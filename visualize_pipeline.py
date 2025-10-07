from brax.io import model
import jax
import mujoco
import mujoco.mjx as mjx
import mujoco.viewer
import jax.numpy as jnp
import numpy as np
from brax.training.acme import running_statistics
from envs.nemo import joystick
from envs.nemo.joystick import rl_config as ppo_params
from networks.ts_networks import make_ppo_networks, make_inference_fn
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
    # normalize = running_statistics.normalize
    #normalize = lambda x, y: x
    normalize = running_statistics.normalize
    obs_size = 67 * 100#env.observation_size
    ppo_network = network_factory(
        obs_size, env.action_size, preprocess_observations_fn=normalize
    )
    make_inference_fn_ = make_inference_fn(ppo_network)
    return make_inference_fn_

command = jnp.array([0.4, 0.0, 0.2])

#dir = "training/nemo_full"
dir = ""

model_path = dir + "/walk_policy"
saved_params = model.load_params(model_path)

# print out stats to catch any NaNs/Infs early

inference_fn = makeIFN()(saved_params)
jit_inference_fn = jax.jit(inference_fn)

rng = jax.random.PRNGKey(0)
mj_model = mujoco.MjModel.from_xml_path('models/nemo/scene.xml')
data = mujoco.MjData(mj_model)
init_qpos = mj_model.keyframe('home').qpos
data.qpos = init_qpos
print("Precomputing rollout")
pipeline_state_list = []
ctrl_list = []
obs_list = []
nn_p_list = []
states = []

for c in range(1000):
    act_rng, rng = jax.random.split(rng)
    obs_list += [state.obs]
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    state.info["command"] = command
    state = jit_step(state, ctrl)
    print("acc: ", env.get_accelerometer(state.data))
    rews = env.test_rewards(state, ctrl)
    pipeline_state = state.data
    ctrl_list += [ctrl]
    states += [state]
    pipeline_state_list += [pipeline_state]


print("Rollout precomputed")

viewer = mujoco.viewer.launch_passive(mj_model, data)
import time
while True:
    for c1 in range(1000):
        #print("=========================")
        #print(ctrl_list[c1])
        #print(nn_p_list[c1])
        #print(obs_list[c1])
        pipeline_state = pipeline_state_list[c1]
        state = states[c1]
        #print(state.info["phase"])
        #print(state.metrics)
        time.sleep(0.02)
        mjx.get_data_into(data, mj_model, pipeline_state)
        viewer.sync()
viewer.close()