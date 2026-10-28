"""NEMO's deployed policy must give identical actions through this repo's network code and
prairie-ros-sim's copy of it. Catches the two copies drifting apart.

Run with:  PRAIRIE_ROS_SIM=/path/to/prairie-ros-sim uv run pytest -m contract
"""
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROS_SIM = os.environ.get("PRAIRIE_ROS_SIM")
DEPLOYED = "walk_policy_trans_acc_low_pd"   # loaded by prairie_control/helpers/policy_network.py
pytestmark = [pytest.mark.contract,
              pytest.mark.skipif(not ROS_SIM, reason="set PRAIRIE_ROS_SIM to a prairie-ros-sim checkout")]


def _inference(ts_networks, policy_file):
    import jax
    from brax.io import model
    from brax.training.acme import running_statistics

    from envs.nemo.joystick import rl_config

    net = ts_networks.make_ppo_networks(67 * 50, 18,
                                        preprocess_observations_fn=running_statistics.normalize,
                                        **dict(rl_config["network_factory"]))
    return jax.jit(ts_networks.make_inference_fn(net)(model.load_params(policy_file), deterministic=True))


def test_deployed_policy_matches_training_code():
    import jax

    helpers = Path(ROS_SIM) / "src/prairie_control/helpers"
    policy_file = str(Path(ROS_SIM) / "src/prairie_control/walk_policy" / DEPLOYED)

    train = _inference(importlib.import_module("networks.ts_networks"), policy_file)
    sys.path.insert(0, str(helpers))          # prairie-ros-sim imports these as top-level modules
    try:
        deploy = _inference(importlib.import_module("ts_networks"), policy_file)
    finally:
        sys.path.remove(str(helpers))

    key = jax.random.PRNGKey(0)
    for i in range(10):
        obs = {"history": jax.random.normal(jax.random.PRNGKey(i), (67 * 50,))}
        a, b = np.asarray(train(obs, key)[0]), np.asarray(deploy(obs, key)[0])
        assert a.shape == (18,) and np.all(np.isfinite(a))
        np.testing.assert_array_equal(a, b)
