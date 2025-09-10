from envs.nemo import joystick
from envs.nemo.joystick import rl_config as ppo_params
from mujoco_playground.config import locomotion_params
from envs.nemo.randomize import domain_randomize
from datetime import datetime
import functools
import matplotlib.pyplot as plt
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper

def make_trainfn():
    env = joystick.Joystick()
    eval_env = joystick.Joystick()

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

    ppo_training_params = dict(ppo_params)
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in ppo_params:
        del ppo_training_params["network_factory"]
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **ppo_params.network_factory
    )

    train_fn = functools.partial(
        ppo.train, **dict(ppo_training_params),
        network_factory=network_factory,
        progress_fn=progress,
        randomization_fn = domain_randomize
    )

    return train_fn, env, eval_env, wrapper.wrap_for_brax_training