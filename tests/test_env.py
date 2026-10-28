"""The NEMO joystick environment builds, resets and steps with the shapes deployment expects."""
import jax
import jax.numpy as jp
import numpy as np
import pytest

from envs.nemo import joystick

FRAME = 67        # per-step observation, see Joystick._get_obs "state"
HISTORY = 50      # frames the transformer sees (joystick.MAX_LEN)
ACTIONS = 18      # 12 leg + 6 arm joints


@pytest.fixture(scope="module")
def env_and_state():
    env = joystick.Joystick()
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    state = jax.jit(env.step)(state, jp.zeros(env.action_size))
    return env, state


def test_sizes_match_deployment(env_and_state):
    env, state = env_and_state
    assert env.action_size == ACTIONS
    assert state.obs["state"].shape == (FRAME,)
    assert state.obs["history"].shape == (FRAME * HISTORY,)
    assert joystick.MAX_LEN == HISTORY


def test_step_is_finite(env_and_state):
    _, state = env_and_state
    for key in ("state", "history", "privileged_state"):
        assert np.all(np.isfinite(state.obs[key])), key
    assert np.isfinite(state.reward)
