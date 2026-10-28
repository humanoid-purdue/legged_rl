"""A few thousand PPO steps on CPU: catches broken training code before anyone burns GPU hours."""
import functools

import numpy as np
import pytest


@pytest.mark.slow
def test_ppo_trains_for_a_few_steps():
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper

    from envs.nemo import joystick
    from envs.nemo.randomize import domain_randomize
    from networks.ts_networks import make_ppo_networks

    params = dict(joystick.rl_config)
    network = params.pop("network_factory")
    params.update(num_timesteps=2000, num_evals=1, num_envs=8, batch_size=8,
                  num_minibatches=1, unroll_length=5, num_updates_per_batch=1,
                  episode_length=50, num_resets_per_eval=0)
    train_fn = functools.partial(
        ppo.train, **params,
        network_factory=functools.partial(make_ppo_networks, **network),
        randomization_fn=domain_randomize)
    _, _, metrics = train_fn(environment=joystick.Joystick(), eval_env=joystick.Joystick(),
                             wrap_env_fn=wrapper.wrap_for_brax_training)
    assert np.isfinite(float(metrics["eval/episode_reward"]))
