"""End-to-end check: NEMO's deployed policy walks in this repo's training environment.

Starts from the `home` keyframe with pushes off, commands 0.4 m/s forward for 10 s,
and requires that NEMO stays upright and covers most of the commanded distance.
If this fails, the environment or model no longer matches the deployed policy:
check actuator gains in models/nemo/nemo5.xml first.

Run with:  PRAIRIE_ROS_SIM=/path/to/prairie-ros-sim uv run pytest -m contract
"""
import os
from pathlib import Path

import numpy as np
import pytest

ROS_SIM = os.environ.get("PRAIRIE_ROS_SIM")
DEPLOYED = "walk_policy_trans_acc_low_pd"
pytestmark = [pytest.mark.contract,
              pytest.mark.skipif(not ROS_SIM, reason="set PRAIRIE_ROS_SIM to a prairie-ros-sim checkout")]

COMMAND = (0.4, 0.0, 0.0)   # vx m/s, vy m/s, yaw rad/s
SECONDS = 10.0
MIN_PELVIS_Z = 0.35         # m; below this NEMO has fallen
MIN_TRACKING = 0.7          # must cover >= 70% of commanded distance


def test_deployed_policy_walks():
    import jax
    import jax.numpy as jp
    from brax.io import model
    from brax.training.acme import running_statistics
    from mujoco import mjx

    from envs.nemo import joystick
    from networks.ts_networks import make_inference_fn, make_ppo_networks

    cfg = joystick.default_config()
    cfg.push_config.enable = False
    env = joystick.Joystick(config=cfg)

    net = make_ppo_networks(67 * 50, env.action_size,
                            preprocess_observations_fn=running_statistics.normalize,
                            **dict(joystick.rl_config["network_factory"]))
    params = model.load_params(str(Path(ROS_SIM) / "src/prairie_control/walk_policy" / DEPLOYED))
    infer = jax.jit(make_inference_fn(net)(params, deterministic=True))
    step = jax.jit(env.step)

    # Clean start at the home pose (reset() randomizes joints and base velocity).
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    home = jp.array(env.mj_model.keyframe("home").qpos)
    data = state.data.replace(qpos=home, qvel=jp.zeros_like(state.data.qvel), ctrl=home[7:])
    state = state.replace(data=mjx.forward(env.mjx_model, data))

    x0 = float(state.data.qpos[0])
    for i in range(int(SECONDS / env.dt)):
        state.info["command"] = jp.array(COMMAND)
        action, _ = infer(state.obs, jax.random.PRNGKey(i))
        state = step(state, action)
        z = float(state.data.qpos[2])
        assert z > MIN_PELVIS_Z, f"NEMO fell at t={i * env.dt:.2f}s (pelvis z={z:.2f} m)"
        assert np.isfinite(np.asarray(action)).all()

    distance = float(state.data.qpos[0]) - x0
    assert distance >= MIN_TRACKING * COMMAND[0] * SECONDS, f"walked only {distance:.2f} m"
