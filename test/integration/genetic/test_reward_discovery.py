from copy import copy
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

from turbanet.ppo import DualEncoderActorCritic, PPOConfig, RolloutBuffer
from turbanet.turba_train_state import TurbaTrainState

# Initialize random number generators
SEED = 42
SOLVE_THRESHOLD = 475
TOTAL_TIMESTEPS = 100_000  # Per generation
GENERATIONS = 1000

SWARM_SIZE = 1000

np.random.seed(SEED)


class RewardNetwork(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = nn.Dense(features=1, name="reward")(x)
        return nn.tanh(y)


class PPOState(NamedTuple):
    train_state: TurbaTrainState
    reward_state: TurbaTrainState
    rng: jax.Array  # PRNG key
    episode_rewards: list[list]
    timestep_log: list[list]
    episode_rewards_live: np.ndarray
    total_steps: int


def sample_actions(rng: jax.Array, logits: np.ndarray) -> tuple[jax.Array, np.ndarray, np.ndarray]:
    rng, key = jax.random.split(rng)

    actions = np.array(jax.random.categorical(key, jnp.array(logits)))
    shifted = logits - logits.max(axis=1, keepdims=True)

    log_probs_all = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    log_probs = log_probs_all[np.arange(len(actions)), actions]

    return rng, actions, log_probs


# Run simulation
def collect_rollout(
    state: PPOState, envs: gym.vector.VectorEnv, cfg: PPOConfig, obs: np.ndarray
) -> tuple[PPOState, RolloutBuffer]:
    """
    Run cfg.n_steps in each of cfg.n_envs environments.
    Returns an updated PPOState and a filled RolloutBuffer.
    obs: (n_envs, obs_dim) - current observations (carry-in from last rollout).
    """
    T, E, D = cfg.n_steps, cfg.n_envs, obs.shape[1]

    buffer_observations = np.zeros((T, E, D), dtype=np.float32)
    buffer_actions = np.zeros((T, E), dtype=np.int32)
    buffer_rewards = np.zeros((T, E), dtype=np.float32)
    buffer_dones = np.zeros((T, E), dtype=np.float32)
    buffer_action_lps = np.zeros((T, E), dtype=np.float32)
    buffer_values = np.zeros((T, E), dtype=np.float32)

    rng = state.rng
    episode_live = state.episode_rewards_live.copy()
    episode_rewards = state.episode_rewards
    timestep_log = state.timestep_log
    total_steps = state.total_steps
    current_observation = obs.copy()

    # Run rollout
    for t in range(T):
        total_steps += 1

        # Forward pass to get logits/values for current observation
        logits, values = state.train_state.predict(jnp.array(current_observation))

        # Sample actions from logits
        rng, actions, log_probabilities = sample_actions(rng, logits)

        # Store transition data
        buffer_observations[t] = current_observation
        buffer_actions[t] = actions
        buffer_action_lps[t] = log_probabilities
        buffer_values[t] = values

        # Run each environment
        current_observation, env_reward, terminated, truncated, _ = envs.step(actions)

        # Get rewards
        reward_input = np.concatenate([current_observation, actions.reshape(-1, 1)], axis=-1)
        reward = state.reward_state.predict(reward_input)

        done = terminated | truncated
        buffer_rewards[t] = reward.squeeze()
        buffer_dones[t] = done.astype(float)
        episode_live += env_reward

        # Episode finished
        if not done.any():
            continue

        for idx, d in enumerate(done):
            if d:
                # Store episode reward and timestep
                episode_rewards[idx].append(episode_live[idx].astype(float))
                timestep_log[idx].append(total_steps)
        episode_live[done] = 0.0

    # Compute last values
    _, last_values = state.train_state.predict(jnp.array(current_observation))
    last_values = np.array(last_values)

    # Create buffer
    buffer = RolloutBuffer(
        observations=buffer_observations,
        actions=buffer_actions,
        rewards=buffer_rewards,
        dones=buffer_dones,
        action_lps=buffer_action_lps,
        values=buffer_values,
        last_values=last_values,
        next_obs=current_observation,
    )

    # Update PPO state
    new_state = PPOState(
        train_state=state.train_state,
        reward_state=state.reward_state,
        rng=rng,
        episode_rewards=episode_rewards,
        timestep_log=timestep_log,
        episode_rewards_live=episode_live,
        total_steps=total_steps,
    )
    return new_state, buffer


# Gradient updates
def run_update(
    state: PPOState, buffer: RolloutBuffer, cfg: PPOConfig
) -> tuple[PPOState, jnp.ndarray]:
    """
    Run cfg.n_epochs of minibatch PPO updates over buf.
    Returns updated PPOState and mean loss.
    """
    advantages, returns = buffer.compute_gae(cfg)

    # Flatten (T, E) → (N,)
    observation_flat = buffer.observations
    action_flat = buffer.actions
    action_lp_flat = buffer.action_lps
    advantage_flat = advantages
    return_flat = returns
    N = len(observation_flat)

    train_state = state.train_state
    rng = state.rng
    losses = []

    loss_info = np.stack(
        [action_flat, action_lp_flat, advantage_flat, return_flat], axis=-1
    ).transpose(1, 0, 2)

    for _ in range(cfg.n_epochs):
        rng, shuffle_key = jax.random.split(rng)
        idx = np.random.permutation(N)
        for start in range(0, N, cfg.batch_size):
            b = idx[start : start + cfg.batch_size]

            # Get observations
            x = observation_flat[b].transpose(1, 0, 2)

            # Normalize advantages
            advantages = advantage_flat[b]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            y = loss_info[:, b, :]
            y[:, :, 2] = advantages.T
            y = jnp.array(y)

            # Train
            train_state, loss, _ = train_state.train(x, y)

            losses.append(loss)

    new_state = PPOState(
        train_state=train_state,
        reward_state=state.reward_state,
        rng=rng,
        episode_rewards=state.episode_rewards,
        timestep_log=state.timestep_log,
        episode_rewards_live=state.episode_rewards_live,
        total_steps=state.total_steps,
    )
    return new_state, jnp.array(losses).mean(axis=0)


# Genetic
def weights_to_int8(params: dict) -> dict:
    return jax.tree_util.tree_map(lambda x: (x * 127).astype(np.int8), params)


def weights_to_float32(params: dict) -> dict:
    return jax.tree_util.tree_map(lambda x: (x / 127).astype(np.float32), params)


def crossover(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    # Convert to bit representation
    x_bit, y_bit = jnp.unpackbits(x.view(jnp.uint8)), jnp.unpackbits(y.view(jnp.uint8))

    # Crossover (two point crossover) using bit representation (assumes same shape)
    indexes = np.random.randint(0, x_bit.shape[0], x_bit.shape[0])
    p = jnp.concatenate(
        (x_bit[0 : indexes.min()], y_bit[indexes.min() : indexes.max()], x_bit[indexes.max() :])
    )

    # Select parameters to mutate
    mutation_p = 0.99
    mutate_value = np.random.rand(*p.shape) > mutation_p

    # Mutate the selected parameters (1% chance of mutation)
    mutated_p = jnp.where(mutate_value, 1 - p, p)

    # Convert back to uint8
    final_p = jnp.packbits(mutated_p).view(jnp.int8)

    return final_p


def genetic_update(
    train_state, reward_state, rng, action_ip, reward_ip, episode_rewards
) -> PPOState:
    # Get average rewards
    mean_rewards = np.array([np.mean(r) for r in episode_rewards])

    # Normalize rewards
    normalized_rewards = mean_rewards / np.max(mean_rewards)

    # Selection (tournament selection)
    p = 1.0
    k = SWARM_SIZE
    num_offspring = SWARM_SIZE - 1

    if num_offspring == 0:
        return train_state, reward_state, action_ip, reward_ip

    tournaments = np.array(
        [
            np.random.choice(len(normalized_rewards), size=k, replace=False)
            for _ in range(num_offspring * 2)
        ]
    )

    probs = p * (1 - p) ** np.argsort(np.argsort(-normalized_rewards[tournaments]))
    selected = np.argmax(np.random.rand(num_offspring * 2, 1) < np.cumsum(probs, axis=1), axis=1)

    pairs = tournaments[np.arange(num_offspring * 2), selected].reshape(num_offspring, 2)

    # Get organism index sorted by normalized reward
    sorted_index = np.argsort(normalized_rewards)
    worst_performers = sorted_index[-num_offspring:]

    # Crossover/Mutation
    action_params = train_state.params
    for pair, replace_idx in zip(pairs, worst_performers):
        parent_1 = jax.tree_util.tree_map(lambda x: x[pair[0]], action_ip)
        parent_2 = jax.tree_util.tree_map(lambda x: x[pair[1]], action_ip)

        mixed_parameters = jax.tree_util.tree_map(crossover, parent_1, parent_2)
        params_int8 = jax.tree_util.tree_map(
            lambda x: jnp.expand_dims(x, axis=0), mixed_parameters
        )

        # Convert to float32
        params_f32 = weights_to_float32(params_int8)

        # Replace organism in initial parameters
        action_ip = jax.tree_util.tree_map(
            lambda x, y: x.at[replace_idx].set(y.reshape(x[replace_idx].shape)),
            action_ip,
            params_int8,
        )

        # Replace organism in train state
        action_params = jax.tree_util.tree_map(
            lambda x, y: x.at[replace_idx].set(y.reshape(x[replace_idx].shape)),
            action_params,
            params_f32,
        )

    reward_params = reward_state.params
    for pair, replace_idx in zip(pairs, worst_performers):
        parent_1 = jax.tree_util.tree_map(lambda x: x[pair[0]], reward_ip)
        parent_2 = jax.tree_util.tree_map(lambda x: x[pair[1]], reward_ip)

        mixed_parameters = jax.tree_util.tree_map(crossover, parent_1, parent_2)
        params_int8 = jax.tree_util.tree_map(
            lambda x: jnp.expand_dims(x, axis=0), mixed_parameters
        )

        # Convert to float32
        params_f32 = weights_to_float32(params_int8)

        # Replace organism in initial parameters
        reward_ip = jax.tree_util.tree_map(
            lambda x, y: x.at[replace_idx].set(y.reshape(x[replace_idx].shape)),
            reward_ip,
            params_int8,
        )

        # Replace organism in train state
        reward_params = jax.tree_util.tree_map(
            lambda x, y: x.at[replace_idx].set(y.reshape(x[replace_idx].shape)),
            reward_params,
            params_f32,
        )

    # Zero all leaves of replaced Adam opt states
    opt_state = jax.tree_util.tree_map(
        lambda x: x.at[worst_performers].set(0), train_state.opt_state
    )

    # Update params
    train_state = train_state.replace(params=action_params)
    reward_state = reward_state.replace(params=reward_params)

    # Update opt state
    train_state = train_state.replace(opt_state=opt_state)

    # Update step
    train_state = train_state.replace(step=train_state.step.at[worst_performers].set(0))

    return train_state, reward_state, action_ip, reward_ip


def calculate_similarity(train_state: TurbaTrainState, x: jnp.ndarray = None) -> jnp.ndarray:
    def cosine_similarity(x: jnp.ndarray) -> jnp.ndarray:
        x_norm = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
        similarity = x_norm @ x_norm.T
        return similarity[jnp.triu_indices(similarity.shape[0], k=1)]

    params = train_state.params

    parameter_tree = jax.tree_util.tree_map(lambda x: x.reshape(x.shape[0], -1), params)
    parameter_similarity_tree = jax.tree_util.tree_map(cosine_similarity, parameter_tree)
    layer_similarity = jnp.stack(jax.tree_util.tree_leaves(parameter_similarity_tree))
    overall_similarity = jnp.nanmean(layer_similarity, axis=0)

    if x is not None:
        _, intermediates = train_state.predict(x, capture_intermediates=True)

        flat_tree = jax.tree_util.tree_map(
            lambda x: x.reshape(x.shape[0], -1), intermediates["intermediates"]
        )

        activation_tree = {k: v for k, v in flat_tree.items() if "__call__" not in k}
        activation_similarity_tree = jax.tree_util.tree_map(cosine_similarity, activation_tree)
        activation_similarity = jnp.stack(jax.tree_util.tree_leaves(activation_similarity_tree))

        output_tree = {k: v for k, v in flat_tree.items() if "__call__" in k}
        output_similarity_tree = jax.tree_util.tree_map(cosine_similarity, output_tree)
        output_similarity = jnp.stack(jax.tree_util.tree_leaves(output_similarity_tree))

        return overall_similarity, layer_similarity, activation_similarity, output_similarity

    return overall_similarity


# Run Jax PPO
def run_jax_ppo(total_timesteps: int = 100_000) -> tuple[list, list]:
    # Initialize config
    cfg = PPOConfig(total_timesteps=total_timesteps, n_envs=SWARM_SIZE)
    env_id = "CartPole-v1"
    rng = jax.random.PRNGKey(SEED)

    # Initialize environments
    environments = gym.make_vec(env_id, num_envs=cfg.n_envs)

    # Temporarily environment to get dimensions of observation and actions
    probe = gym.make(env_id)
    obs_dim = probe.observation_space.shape[0]
    n_actions = probe.action_space.n
    probe.close()

    # Initialize network/training state
    net = DualEncoderActorCritic(n_actions=n_actions)
    ts = TurbaTrainState.swarm(
        model=net,
        optimizer=optax.chain(
            optax.clip_by_global_norm(cfg.max_grad_norm),
            optax.adam(cfg.lr, eps=1e-5),
        ),
        swarm_size=cfg.n_envs,
        input_size=obs_dim,
        seed=SEED,
    )

    reward_ts = TurbaTrainState.swarm(
        model=RewardNetwork(),
        optimizer=optax.chain(
            optax.clip_by_global_norm(cfg.max_grad_norm),
            optax.adam(cfg.lr, eps=1e-5),
        ),
        swarm_size=cfg.n_envs,
        input_size=obs_dim + 1,
        seed=SEED,
    )

    # Store initial parameters
    action_ip = copy(weights_to_int8(ts.params))
    reward_ip = copy(weights_to_int8(reward_ts.params))

    # Initialize plot directory
    plot_dir = Path(f"test/integration/{datetime.now().strftime('%Y%m%d%H%M%S')}")
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    reward_history = []
    similarity_history = []
    for generation in range(GENERATIONS):
        print(f"\n[JAX] training… Generation {generation + 1}/{GENERATIONS}")

        # Set loss function
        ts = ts.set_loss_fn(cfg.loss_fn)

        # Initialize PPO state
        state = PPOState(
            train_state=ts,
            reward_state=reward_ts,
            rng=rng,
            episode_rewards=[[] for _ in range(cfg.n_envs)],
            timestep_log=[[] for _ in range(cfg.n_envs)],
            episode_rewards_live=np.zeros(cfg.n_envs, dtype=np.float32),
            total_steps=0,
        )
        # Get initial observations
        observation, _ = environments.reset()

        # Calculate similarities
        similarity_history.append(calculate_similarity(ts, observation))

        losses = []
        n_updates = cfg.total_timesteps // cfg.n_steps
        for update in range(1, n_updates + 1):
            print("\t[JAX] update", update, "of", n_updates)

            # Collect rollout
            state, buffer = collect_rollout(state, environments, cfg, observation)
            observation = buffer.next_obs

            # Run update
            state, loss = run_update(state, buffer, cfg)
            losses.append(loss)

            # Log progress
            if state.episode_rewards and update % 1 == 0:
                final_similarity = calculate_similarity(state.train_state)

                recent = [np.mean(r[-20:]) for r in state.episode_rewards]
                print(
                    f"\t[JAX] progress {state.total_steps}/{cfg.total_timesteps} | "
                    f"mean_rew(last 20): {np.mean(recent):7.1f} | "
                    f"loss: {loss.mean():7.1f} | "
                    f"initial similarity: {similarity_history[-1][0].mean():.7f} | "
                    f"final similarity: {final_similarity.mean():.7f}"
                )

        reward_history.append([np.mean(r) for r in state.episode_rewards])

        # Perform genetic optimization step
        ts, reward_ts, action_ip, reward_ip = genetic_update(
            state.train_state,
            state.reward_state,
            state.rng,
            action_ip,
            reward_ip,
            state.episode_rewards,
        )

    # Close environments
    environments.close()

    return state, state.episode_rewards, state.timestep_log


if __name__ == "__main__":
    jax_results = run_jax_ppo(total_timesteps=TOTAL_TIMESTEPS)
