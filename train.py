from envs.nemo import joystick
from envs.nemo.joystick import rl_config as ppo_params
from mujoco_playground.config import locomotion_params
from envs.nemo.randomize import domain_randomize
from datetime import datetime
import argparse
import functools
import json
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from brax.training.agents.ppo import networks as ppo_networks
from networks.ts_networks import make_ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.io import model
from mujoco_playground import wrapper

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out_dir", default=None,
                    help="defaults to training/<timestamp>_seed<seed>")
args = parser.parse_args()

run_start = datetime.now()
out_dir = args.out_dir or os.path.join(
    "training", f"{run_start:%Y%m%d_%H%M%S}_seed{args.seed}")
os.makedirs(out_dir, exist_ok=True)


def git_info():
  try:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    return {"commit": commit, "dirty": dirty}
  except (subprocess.CalledProcessError, FileNotFoundError):
    return {"commit": None, "dirty": None}


env_cfg = joystick.default_config()
env = joystick.Joystick(config=env_cfg)
eval_env = joystick.Joystick(config=env_cfg)
ppo_training_params = dict(ppo_params)


x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]

def progress(num_steps, metrics):

  times.append(datetime.now())
  x_data.append(num_steps)
  y_data.append(metrics["eval/episode_reward"])
  y_dataerr.append(metrics["eval/episode_reward_std"])

  plt.xlim([0, ppo_params["num_timesteps"] * 1.25])
  plt.xlabel("# environment steps")
  plt.ylabel("reward per episode")
  plt.title(f"y={y_data[-1]:.3f}")
  plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")
  plt.show()


network_factory = make_ppo_networks
if "network_factory" in ppo_params:
  del ppo_training_params["network_factory"]
  network_factory = functools.partial(
      make_ppo_networks,
      **ppo_params.network_factory
  )

train_fn = functools.partial(
    ppo.train, **dict(ppo_training_params),
    network_factory=network_factory,
    randomization_fn = domain_randomize,
    progress_fn=progress,
    seed=args.seed,
)

make_inference_fn, params, metrics = train_fn(
    environment=env,
    #eval_env=env,
    eval_env = eval_env,
    wrap_env_fn=wrapper.wrap_for_brax_training,
)
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")

# Same format visualize_pipeline.py / keyboard.py load with model.load_params.
model.save_params(os.path.join(out_dir, "walk_policy"), params)

run_info = {
    "started": run_start.isoformat(),
    "finished": datetime.now().isoformat(),
    "seed": args.seed,
    "git": git_info(),
    "env_config": env_cfg.to_dict(),
    "ppo_config": ppo_params.to_dict(),
    "final_metrics": {k: float(np.asarray(v)) for k, v in metrics.items()},
    "eval_curve": {"num_steps": [int(x) for x in x_data],
                   "episode_reward": [float(y) for y in y_data],
                   "episode_reward_std": [float(e) for e in y_dataerr]},
}
with open(os.path.join(out_dir, "run.json"), "w") as f:
  json.dump(run_info, f, indent=2, default=str)
print(f"saved policy and run info to {out_dir}")