from typing import Callable

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from jaxatari.environment import JaxEnvironment
from jaxatari.wrappers import JaxatariWrapper

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: nn.Module,
    capture_video: bool = True,
    seed=1,
    object_centric: bool = False,  # ppo+ewc only; whether the environment is object-centric
    padding_width: int = 0,    # ppo+ewc only; padding width for the object-centric observation space
    crl: bool = False,  # flag to indicate if we are in the CRL setting
):
    env: JaxEnvironment | JaxatariWrapper = make_env(env_id, seed, 1)()
    _Network, _Actor, _Critic = Model
    key = jax.random.key(seed)

    def pad_obs(obs):
        """Zero-pad object-centric observations"""
        return jnp.pad(obs, ((0, 0), (0, padding_width)))

    @jax.jit
    def wrapped_reset(key):
        """wrappes the reset function of the environment to correct the observation shape"""
        next_obs, state = env.reset(key)
        # NNs require shape (B, F, H, W), where B is the batch size and F is the frame stack size
        next_obs = next_obs.squeeze()[None, ...]
        # pad observations for object-centric environments to have consistent shape across tasks
        if object_centric:
            next_obs = pad_obs(next_obs)
        return next_obs, state

    @jax.jit 
    def wrapped_step(state, action):
        """wrappes the step function of the environment to correct the observation shape"""
        next_obs, next_state, reward, terminated, truncated, info =  env.step(state, action.squeeze())
        done = jnp.logical_or(terminated, truncated)
        # NNs require shape (B, F, H, W), where B is the batch size and F is the frame stack size
        next_obs = next_obs.squeeze()[None, ...]
        # pad observations for object-centric environments to have consistent shape across tasks
        if object_centric:
            next_obs = pad_obs(next_obs)
        return next_obs, next_state, reward, done, info

    key, reset_key = jax.random.split(key)
    next_obs, handle = wrapped_reset(reset_key)
    network = _Network()
    actor = _Actor(action_dim=env.action_space().n) if not crl else _Actor(action_dim=18)
    critic = _Critic()
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)
    key, network_key_2, actor_key_2, critic_key_2 = jax.random.split(key, 4)
    network_params = network.init(network_key, env.observation_space().sample(network_key_2).squeeze()[None, ...])
    actor_params = actor.init(actor_key, network.apply(network_params, env.observation_space().sample(actor_key_2).squeeze()[None, ...]))
    critic_params = critic.init(critic_key, network.apply(network_params, env.observation_space().sample(critic_key_2).squeeze()[None, ...]))
    # note: critic_params is not used in this script
    with open(model_path, "rb") as f:
        (args, (network_params, actor_params, critic_params)) = flax.serialization.from_bytes(
            (None, (network_params, actor_params, critic_params)), f.read()
        )

    @jax.jit
    def get_action_and_value(
        network_params: flax.core.FrozenDict,
        actor_params: flax.core.FrozenDict,
        next_obs: jnp.ndarray,
        key: jax.random.PRNGKey,
    ):
        hidden = network.apply(network_params, next_obs)
        logits = actor.apply(actor_params, hidden)
        # sample action: Gumbel-softmax trick
        # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        return action, key

    def step_fn(carry, input):
        next_obs, env_state, keys = carry
        actions, keys = jax.vmap(get_action_and_value, in_axes=(None, None, 0, 0))(network_params, actor_params, next_obs, keys)
        next_obs, env_state, reward, done, infos = jax.vmap(wrapped_step)(env_state, jnp.array(actions))
        first_states = jax.tree.map(lambda x: x[0], env_state)
        # since the env is eval_env (without reward clipping and episodic life), we can just accumulate the rewards
        return (next_obs, env_state, keys), (first_states, done, reward, actions) 

    # evaluate eval_episodes concurrently
    reset_keys = jax.random.split(key, eval_episodes)
    next_obs, env_states = jax.vmap(wrapped_reset)(reset_keys)
    _, (first_states, dones, rewards, actions) = jax.lax.scan(step_fn, (next_obs, env_states, reset_keys), None, length=10_000)

    print("scanned rewards: ", rewards.shape, jnp.sum(rewards), jnp.mean(rewards))

    # obs shape: (time, eval_episodes, 1, H, W)
    first_done = jnp.argmax(dones, axis=0)  # shape: (eval_episodes,)
    # print("first dones: ", first_done)
    # reward_mask = jnp.arange(dones.shape[0])[:, None] <= first_done[None, :]  # shape: (time, eval_episodes)
    # rewards = rewards * reward_mask  # shape: (time, eval_episodes)
    # print("masked rewards: ", rewards.shape, jnp.sum(rewards), jnp.mean(rewards))
    has_finished = jax.lax.cummax(dones.astype(jnp.int32), axis=0)
    # shift right by one timestep
    mask_after_first_done = jnp.pad(has_finished[:-1, :], ((1,0),(0,0)), constant_values=0)
    # masked_rewards = rewards * (1 - mask_after_first_done)
    rewards = rewards * (1 - mask_after_first_done)
    print("filtered rewards: ", rewards.shape, jnp.sum(rewards), jnp.mean(rewards))
    episodic_returns = jnp.sum(rewards, axis=0)  # shape: (eval_episodes,)

    # first episode video capture
    # states_until_done = first_obs[:first_done[0] + 1, 0]  # shape: (time_until_done, 1, H, W)
    # env_states_until_done = jax.tree.map(lambda x: x[:first_done[0] + 1], first_states.atari_state.atari_state.env_state)

    # first episode video capture (fix to account for the case where the first episode does not finish within 10,000 steps)
    # Check if the first evaluation episode actually finished
    is_done = bool(jnp.any(dones[:, 0]).item())
    
    # If it finished, slice at the done index. If not, slice to the end of the array (10,000).
    end_idx = int(first_done[0].item()) if is_done else (dones.shape[0] - 1)
    
    env_states_until_done = jax.tree.map(lambda x: x[:end_idx + 1], first_states.atari_state.atari_state.env_state)

    return episodic_returns, env_states_until_done