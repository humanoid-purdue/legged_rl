# legged_rl

Reinforcement-learning training for **NEMO**, Purdue HRC's humanoid. We train a
transformer walking policy with PPO ([Brax](https://github.com/google/brax)) in
GPU-parallel MuJoCo ([MJX](https://mujoco.readthedocs.io/en/stable/mjx.html)),
built on [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground).
Trained policies are deployed on the robot by
[prairie-ros-sim](https://github.com/humanoid-purdue/prairie-ros-sim).

If you did the HRC software onboarding, this is the same pipeline as Stages 3–4,
scaled up from Pup to NEMO.

## Getting started

Install [uv](https://docs.astral.sh/uv/), then:

```bash
git clone https://github.com/humanoid-purdue/legged_rl.git
cd legged_rl
uv sync                  # CPU: laptop, CI
uv run pytest            # environment tests (~1 min)
uv run pytest -m slow    # tiny PPO training run on CPU (~3 min)
```

On a Linux machine with an NVIDIA GPU, use `uv sync --extra gpu` instead.

## Layout

| Path | What it is |
|---|---|
| `envs/nemo/joystick.py` | The task: observations, rewards, termination, and `rl_config` (PPO + network settings) |
| `envs/nemo/base.py` | Shared NEMO environment setup |
| `envs/nemo/randomize.py` | Domain randomization (friction, mass, …) |
| `networks/` | Transformer policy. **prairie-ros-sim keeps a copy of this code**; `tests/test_policy_contract.py` checks they match |
| `models/nemo/` | MuJoCo model of NEMO (`scene.xml`, `nemo5.xml`) and meshes |
| `train.py` | Train locally |
| `colab_train.py` | Training entry point for Colab |
| `visualize_pipeline.py`, `keyboard.py` | Watch / drive a trained policy in the MuJoCo viewer |

## Observation contract

The deployed policy sees the last **50** frames of a **67-number** observation
(`3350` inputs) and outputs **18** joint targets:

`gyro (3) · accelerometer (3) · command vx,vy,yaw (3) · joint pos − default (18) · joint vel (18) · last action (18) · phase clock (4)`

Changing this means changing `prairie_control/helpers/policy_network.py` in
prairie-ros-sim **and** retraining. Never one without the other.

## How we work

- **Experiments don't need a PR.** Copy settings, train, record the result
  (config, commit, seed, eval numbers, video) in the experiment log.
- **Code changes do:** new rewards, environment changes, network changes. Branch,
  PR, green CI, review.
- **Promoting a policy to NEMO is a PR** with its config, eval numbers versus the
  current policy, and a video. Then it's published as a GitHub Release and
  prairie-ros-sim updates to it.

## Dependencies

Versions are pinned in `pyproject.toml` / `uv.lock` to what the current policy
was trained with. Newer mujoco-mjx (3.4+) breaks `envs/nemo/base.py`. Upgrade in
a dedicated PR, with the smoke test and contract test green.
