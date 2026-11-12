"""Score a trained NEMO policy headlessly.

    uv run python eval.py --policy path/to/walk_policy

When run it starts NEMO at the home pose (pushes off unless --pushes), runs each command for a fixed time,
records whether it fell, how far it moved, and how closely its speed matched the command,
then writes everything to a JSON file and prints a summary.
The policy must have been trained with the same actuator gains as models/nemo/nemo5.xml.
"""
import argparse
import json

import jax
import jax.numpy as jnp
import mujoco.mjx as mjx
import numpy as np
from brax.io import model
from brax.training.acme import running_statistics

from envs.nemo import joystick
from networks.ts_networks import make_inference_fn, make_ppo_networks

# Commands NEMO will run
COMMANDS = [
    ("stand", 0.0, 0.0, 0.0),
    ("forward", 0.4, 0.0, 0.0),
    ("backward", -0.4, 0.0, 0.0),
    ("left", 0.0, 0.3, 0.0),
    ("right", 0.0, -0.3, 0.0),
    ("turn left", 0.0, 0.0, 0.7),
    ("turn right", 0.0, 0.0, -0.7),
    ("forward slow", 0.2, 0.0, 0.0),
    ("forward fast", 0.5, 0.0, 0.0),
    ("forward + turn", 0.4, 0.0, 0.7),
]
FALL_HEIGHT = 0.35

# Command line options.
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--policy", required=True, help="path to a saved Brax params file")
parser.add_argument("--seconds", type=float, default=10.0, help="how long to run each command")
parser.add_argument("--out", default="eval.json", help="where to write the JSON results")
parser.add_argument("--pushes", action="store_true", help="keep the training env's random pushes on")
parser.add_argument("--stochastic", action="store_true", help="sample actions like during training")
args = parser.parse_args()

# Builds NEMO training world.
cfg = joystick.default_config()
cfg.push_config.enable = args.pushes
env = joystick.Joystick(config=cfg)

# Rebuilds the policy's network and loads training params into it.
network = make_ppo_networks(67 * 50, env.action_size,
                            preprocess_observations_fn=running_statistics.normalize,
                            **dict(joystick.rl_config["network_factory"]))
policy = jax.jit(make_inference_fn(network)(model.load_params(args.policy), deterministic=not args.stochastic)) # Deterministic default: uses policy's top action everytime. Turn off with --stochastic
step = jax.jit(env.step)
reset = jax.jit(env.reset)
home = jnp.array(env.mj_model.keyframe("home").qpos)
n_steps = int(args.seconds / env.dt)

# Runs a command and returns its speed history, start, end, and fall time.
def run_command(command):
    state = reset(jax.random.PRNGKey(0))
    data = state.data.replace(qpos=home, qvel=jnp.zeros_like(state.data.qvel), ctrl=home[7:])
    state = state.replace(data=mjx.forward(env.mjx_model, data))
    start = np.array(state.data.qpos[:2])
    speeds, fell_at = [], None
    for i in range(n_steps):
        state.info["command"] = jnp.array(command)
        action, _ = policy(state.obs, jax.random.PRNGKey(i))
        state = step(state, action)
        if float(state.data.qpos[2]) < FALL_HEIGHT:
            fell_at = (i + 1) * env.dt
            break
        forward, sideways = env.get_local_linvel(state.data)[:2]
        turn = env.get_gyro(state.data)[2]
        speeds.append([float(forward), float(sideways), float(turn)])
    end = np.array(state.data.qpos[:2])
    return np.array(speeds), start, end, fell_at


# Runs through all the commands & stores the results.
results = []
for name, vx, vy, yaw in COMMANDS:
    print(f"Running {name}...")
    speeds, start, end, fell_at = run_command((vx, vy, yaw))
    asked = np.array([vx, vy, yaw])
    results.append({
        "name": name,
        "command": [vx, vy, yaw],
        "fell": fell_at is not None,
        "fell_at": fell_at,
        "distance": float(np.linalg.norm(end - start)),
        "commanded_distance": float(np.hypot(vx, vy) * args.seconds),
        "mean_speed": speeds.mean(axis=0).tolist(),
        "mean_abs_error": np.abs(speeds - asked).mean(axis=0).tolist(),
    })


# Saves results to JSON file
with open(args.out, "w") as f:
        json.dump({"policy": args.policy, "seconds": args.seconds, "pushes": args.pushes, "stochastic": args.stochastic, "results": results}, f, indent=2)

print()
print(f"{'command':<16}{'fell':<11}{'distance (m)':<15}speed asked -> got (forward, sideways, turn)")
for r in results:
    fell = f"at {r['fell_at']:.1f} s" if r["fell"] else "no"
    distance = f"{r['distance']:.2f} / {r['commanded_distance']:.2f}"
    asked = ", ".join(f"{v:+.2f}" for v in r["command"])
    got = ", ".join(f"{v:+.2f}" for v in r["mean_speed"])
    print(f"{r['name']:<16}{fell:<11}{distance:<15}{asked} -> {got}")
print(f"\nSaved to {args.out}")