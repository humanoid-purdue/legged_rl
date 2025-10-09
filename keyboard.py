import threading
import time
from typing import Set

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import mujoco.mjx as mjx
import numpy as np
from pynput import keyboard
from brax.training.acme import running_statistics
from brax.io import model

from envs.nemo import joystick
from envs.nemo.joystick import rl_config as ppo_params
from networks.ts_networks import make_ppo_networks, make_inference_fn

# Environment and network setup (mirrors visualize_pipeline.py)
env = joystick.Joystick()
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)
state = jit_reset(jax.random.PRNGKey(0))


def makeIFN():
    import functools
    network_factory = functools.partial(
        make_ppo_networks,
        **ppo_params.network_factory
    )
    normalize = running_statistics.normalize
    # Match obs size heuristic used in visualize_pipeline
    obs_size = 67 * 100  # env.observation_size
    ppo_network = network_factory(
        obs_size, env.action_size, preprocess_observations_fn=normalize
    )
    return make_inference_fn(ppo_network)


# Load policy parameters
dir = "training/nemo_full"
model_path = dir + "/walk_policy"
saved_params = model.load_params(model_path)

inference_fn = makeIFN()(saved_params)
jit_inference_fn = jax.jit(inference_fn)

# Command state managed by keyboard listener
pressed: Set[str] = set()
pressed_lock = threading.Lock()
exit_flag = threading.Event()

FWD_SPEED = 0.4  # m/s
YAW_SPEED = 0.7  # rad/s


def compute_command():
    with pressed_lock:
        f = (('w' in pressed) - ('s' in pressed)) * FWD_SPEED  # forward/back
        yaw = (('a' in pressed) - ('d' in pressed)) * YAW_SPEED  # left/right yaw
    # Command vector assumed as [forward, lateral(0), yaw]
    return jnp.array([f, 0.0, yaw], dtype=jnp.float32)


def on_press(key):
    try:
        k = key.char.lower()
    except AttributeError:
        if key == keyboard.Key.esc:
            exit_flag.set()
        return
    if k in ('w', 'a', 's', 'd'):
        with pressed_lock:
            pressed.add(k)


def on_release(key):
    try:
        k = key.char.lower()
    except AttributeError:
        return
    if k in ('w', 'a', 's', 'd'):
        with pressed_lock:
            pressed.discard(k)


def main():
    global state
    rng = jax.random.PRNGKey(0)

    # Setup MuJoCo viewer using underlying xml scene
    mj_model = mujoco.MjModel.from_xml_path('models/nemo/scene.xml')
    data = mujoco.MjData(mj_model)
    init_qpos = mj_model.keyframe('home').qpos
    data.qpos = init_qpos

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    viewer = mujoco.viewer.launch_passive(mj_model, data)
    print("Keyboard control active. Use W/S for forward/back, A/D for yaw, ESC to quit.")

    try:
        while not exit_flag.is_set():
            # Update command from current keyboard state
            command = compute_command()

            state.info["command"] = command
            print(state.info["phase"], state.info["command"])

            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, ctrl)

            # Transfer mjx pipeline state into mujoco data for visualization
            pipeline_state = state.data
            mjx.get_data_into(data, mj_model, pipeline_state)
            viewer.sync()
    except KeyboardInterrupt:
        pass
    finally:
        exit_flag.set()
        listener.stop()
        viewer.close()
        print("Exited keyboard control.")


if __name__ == "__main__":
    main()
