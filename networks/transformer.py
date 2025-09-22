import dataclasses
import functools
from typing import Any, Callable, Literal, Mapping, Sequence, Tuple
import warnings

from brax.training import types
from brax.training.acme import running_statistics
from brax.training.spectral_norm import SNDense
from flax import linen
from flax import linen as nn
import jax
import jax.numpy as jnp
from brax.training.networks import FeedForwardNetwork, normalizer_select, _get_obs_state_size
from transformer_policy import TransformerPolicy, TransformerPolicyModuleWithStd

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]

def make_policy_network(
	param_size: int,
	obs_size: types.ObservationSize,
	preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
	emb_dim = 128,
	max_len = 1000,
	num_layers = 8,
	num_heads = 8,
	mlp_dim = 256,
	kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
	obs_key: str = 'state',
	history_key: str = 'history',
	distribution_type: Literal['normal', 'tanh_normal'] = 'tanh_normal',
	noise_std_type: Literal['scalar', 'log'] = 'scalar',
	init_noise_std: float = 1.0,
	state_dependent_std: bool = False,
) -> FeedForwardNetwork:
	"""Creates a policy network."""
	if distribution_type == 'tanh_normal':
		policy_module = TransformerPolicy(
			obs_dim=obs_size,
			action_dim=param_size,
			emb_dim=emb_dim,
			max_len=max_len,
			num_layers=num_layers,
			num_heads=num_heads,
			mlp_dim=mlp_dim,
			kernel_init=kernel_init
        )
	elif distribution_type == 'normal':
		policy_module = TransformerPolicyModuleWithStd(
			obs_dim=obs_size,
			action_dim=param_size,
			emb_dim=emb_dim,
			max_len=max_len,
			num_layers=num_layers,
			num_heads=num_heads,
			mlp_dim=mlp_dim,
			kernel_init=kernel_init,
			noise_std_type=noise_std_type,
            init_noise_std=init_noise_std,
            state_dependent_std=state_dependent_std,
		)
	else:
		raise ValueError(
			f'Unsupported distribution type: {distribution_type}. Must be one'
			' of "normal" or "tanh_normal".'
		)

	def apply(processor_params, policy_params, obs):
		if obs[obs_key].ndim == 1:
			hist_mat = jnp.concatenate([
			obs[obs_key][None, :], obs[history_key]
            ], axis = 0)
			in_axes = 0
			out_axes = 0
		else:
			hist_mat = jnp.concatenate([
			    obs[obs_key][:, None, :], obs[history_key]
            ], axis = 1)
			in_axes = 1
			out_axes = 1
		
		
		if isinstance(obs, Mapping):
			norm_params = normalizer_select(processor_params, obs_key)
			preprocess_batched = jax.vmap(
                    lambda xi: preprocess_observations_fn(xi, norm_params),
                    in_axes=in_axes, out_axes=out_axes
                )
			obs = preprocess_batched(hist_mat)
		else:
			preprocess_batched = jax.vmap(
                    lambda xi: preprocess_observations_fn(xi, processor_params),
                    in_axes=in_axes, out_axes=out_axes
                )
			obs = preprocess_batched(hist_mat)
		return policy_module.apply(policy_params, obs)

	obs_size = _get_obs_state_size(obs_size, obs_key)
	dummy_obs = jnp.zeros((1, obs_size))

	def init(key):
		policy_module_params = policy_module.init(key, dummy_obs)
		return policy_module_params

	return FeedForwardNetwork(init=init, apply=apply)