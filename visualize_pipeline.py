"""Roll out a trained NEMO policy and replay it in the MuJoCo viewer.

    uv run python visualize_pipeline.py --policy path/to/walk_policy          # Linux
    uv run mjpython visualize_pipeline.py --policy path/to/walk_policy        # macOS
    uv run python visualize_pipeline.py --policy ... --headless               # no viewer, print a summary

The policy must have been trained with the same actuator gains as models/nemo/nemo5.xml.
"""
import argparse
import time

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
from brax.io import model
from brax.training.acme import running_statistics

from envs.nemo import joystick
from networks.ts_networks import make_inference_fn, make_ppo_networks

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--policy", required=True, help="path to a saved Brax params file")
parser.add_argument("--vx", type=float, default=0.4, help="forward command, m/s")
parser.add_argument("--vy", type=float, default=0.0, help="sideways command, m/s")
parser.add_argument("--yaw", type=float, default=0.0, help="yaw-rate command, rad/s")
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument("--pushes", action="store_true", help="keep the training env's random pushes on")
parser.add_argument("--stochastic", action="store_true", help="sample actions like during training")
parser.add_argument("--random-start", action="store_true", help="use reset()'s randomized start instead of the home pose")
parser.add_argument("--headless", action="store_true", help="don't open the viewer; just print a summary")
args = parser.parse_args()

cfg = joystick.default_config()
cfg.push_config.enable = args.pushes
env = joystick.Joystick(config=cfg)

network = make_ppo_networks(67 * 50, env.action_size,
                            preprocess_observations_fn=running_statistics.normalize,
                            **dict(joystick.rl_config["network_factory"]))
policy = jax.jit(make_inference_fn(network)(model.load_params(args.policy),
                                            deterministic=not args.stochastic))
step = jax.jit(env.step)

state = jax.jit(env.reset)(jax.random.PRNGKey(0))
if not args.random_start:
    # reset() scales joint angles by U(0.5, 1.5) and adds random base velocity, which can
    # start NEMO already off balance. For watching a policy, start cleanly at the home pose.
    home = jnp.array(env.mj_model.keyframe("home").qpos)
    data = state.data.replace(qpos=home, qvel=jnp.zeros_like(state.data.qvel), ctrl=home[7:])
    state = state.replace(data=mjx.forward(env.mjx_model, data))

command = jnp.array([args.vx, args.vy, args.yaw])
n_steps = int(args.seconds / env.dt)
x0 = float(state.data.qpos[0])
rng = jax.random.PRNGKey(1)
frames, fell_at = [], None
print(f"Rolling out {n_steps} steps ({args.seconds:.0f} s)...")
for i in range(n_steps):
    rng, act_rng = jax.random.split(rng)
    state.info["command"] = command
    action, _ = policy(state.obs, act_rng)
    state = step(state, action)
    frames.append(state.data)
    if fell_at is None and float(state.data.qpos[2]) < 0.35:
        fell_at = i * env.dt

distance = float(state.data.qpos[0]) - x0
print(f"Forward distance {distance:.2f} m (commanded {args.vx * args.seconds:.2f} m). "
      + (f"Fell at t = {fell_at:.2f} s." if fell_at is not None else "Stayed upright."))

if not args.headless:
    import mujoco.viewer
    viewer_data = mujoco.MjData(env.mj_model)
    with mujoco.viewer.launch_passive(env.mj_model, viewer_data) as viewer:
        while viewer.is_running():
            for frame in frames:
                if not viewer.is_running():
                    break
                mjx.get_data_into(viewer_data, env.mj_model, frame)
                viewer.sync()
                time.sleep(env.dt)