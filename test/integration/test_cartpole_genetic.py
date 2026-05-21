from copy import copy
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from matplotlib.animation import FuncAnimation

from turbanet.ppo import DualEncoderActorCritic, PPOConfig, RolloutBuffer
from turbanet.turba_train_state import TurbaTrainState

# Initialize random number generators
SEED = 42
SOLVE_THRESHOLD = 475
TOTAL_TIMESTEPS = 5_000  # Per generation
GENERATIONS = 200

SWARM_SIZE = 1000

np.random.seed(SEED)


class PPOState(NamedTuple):
    """Everything that changes across updates, passed explicitly."""

    train_state: TurbaTrainState  # flax TurbaTrainState (params + opt)
    rng: jax.Array  # PRNG key
    episode_rewards: list[list]  # list of lists – appended in rollout
    timestep_log: list[list]  # matching global-step for each ep
    episode_rewards_live: np.ndarray  # per-env running reward accumulator
    total_steps: int  # global step counter


def sample_actions(rng: jax.Array, logits: np.ndarray) -> tuple[jax.Array, np.ndarray, np.ndarray]:
    """
    Sample from logits, return (actions, log_probs) as numpy arrays.
    Numerically stable log-softmax done in numpy.
    """
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
        current_observation, reward, terminated, truncated, _ = envs.step(actions)

        done = terminated | truncated
        buffer_rewards[t] = reward
        buffer_dones[t] = done.astype(float)
        episode_live += reward

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
    mutate_value = np.random.rand(*p.shape) > 0.99

    # Mutate the selected parameters (1% chance of mutation)
    mutated_p = jnp.where(mutate_value > 0.99, 1 - p, p)

    # Convert back to uint8
    final_p = jnp.packbits(mutated_p).view(jnp.int8)

    return final_p


def genetic_update(train_state, rng, initial_parameters, episode_rewards) -> PPOState:
    # Get average rewards
    mean_rewards = np.array([np.mean(r) for r in episode_rewards])

    # Normalize rewards
    normalized_rewards = mean_rewards / np.max(mean_rewards)

    # Selection (tournament selection)
    p = 0.5
    k = 5
    num_offspring = SWARM_SIZE

    if num_offspring == 0:
        return train_state, initial_parameters

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
    params = train_state.params
    for pair, replace_idx in zip(pairs, worst_performers):
        parent_1 = jax.tree_util.tree_map(lambda x: x[pair[0]], initial_parameters)
        parent_2 = jax.tree_util.tree_map(lambda x: x[pair[1]], initial_parameters)

        mixed_parameters = jax.tree_util.tree_map(crossover, parent_1, parent_2)
        params_int8 = jax.tree_util.tree_map(
            lambda x: jnp.expand_dims(x, axis=0), mixed_parameters
        )

        # Convert to float32
        params_f32 = weights_to_float32(params_int8)

        # Replace organism in initial parameters
        initial_parameters = jax.tree_util.tree_map(
            lambda x, y: x.at[replace_idx].set(y.reshape(x[replace_idx].shape)),
            initial_parameters,
            params_int8,
        )

        # Replace organism in train state
        params = jax.tree_util.tree_map(
            lambda x, y: x.at[replace_idx].set(y.reshape(x[replace_idx].shape)),
            params,
            params_f32,
        )

    # Zero all leaves of replaced Adam opt states
    opt_state = jax.tree_util.tree_map(
        lambda x: x.at[worst_performers].set(0), train_state.opt_state
    )

    # Update params
    train_state = train_state.replace(params=params)

    # Update opt state
    train_state = train_state.replace(opt_state=opt_state)

    # Update step
    train_state = train_state.replace(step=train_state.step.at[worst_performers].set(0))

    return train_state, initial_parameters


def calculate_similarity(params: dict) -> jnp.ndarray:
    def cosine_similarity(x: jnp.ndarray) -> jnp.ndarray:
        x_norm = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
        return x_norm @ x_norm.T

    flat_tree = jax.tree_util.tree_map(lambda x: x.reshape(x.shape[0], -1), params)
    similarity_tree = jax.tree_util.tree_map(cosine_similarity, flat_tree)
    leaves = jax.tree_util.tree_leaves(similarity_tree)
    overall_sim = jnp.nanmean(jnp.stack(leaves), axis=0)
    return overall_sim[jnp.triu(jnp.ones((1000, 1000), dtype=bool), k=1)]


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

    # Initialize plot directory
    plot_dir = Path(f"test/integration/{datetime.now().strftime('%Y%m%d%H%M%S')}")
    plot_dir.mkdir(parents=True, exist_ok=True)
    max_similarity_bin_height = 0.0
    max_reward_bin_height = 0.0

    # Training loop
    reward_history = []
    similarity_history = []
    for generation in range(GENERATIONS):
        print(f"\n[JAX] training… Generation {generation + 1}/{GENERATIONS}")

        # Store initial parameters
        initial_parameters = copy(weights_to_int8(ts.params))
        initial_similarity = calculate_similarity(initial_parameters)
        similarity_history.append(initial_similarity)

        # Set loss function
        ts = ts.set_loss_fn(cfg.loss_fn)

        # Initialize PPO state
        state = PPOState(
            train_state=ts,
            rng=rng,
            episode_rewards=[[] for _ in range(cfg.n_envs)],
            timestep_log=[[] for _ in range(cfg.n_envs)],
            episode_rewards_live=np.zeros(cfg.n_envs, dtype=np.float32),
            total_steps=0,
        )
        # Get initial observations
        observation, _ = environments.reset()

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
                final_similarity = calculate_similarity(state.train_state.params)

                recent = [np.mean(r[-20:]) for r in state.episode_rewards]
                print(
                    f"\t[JAX] progress {state.total_steps}/{cfg.total_timesteps} | "
                    f"mean_rew(last 20): {np.mean(recent):7.1f} | "
                    f"loss: {loss.mean():7.1f} | "
                    f"initial similarity: {initial_similarity.mean():.7f} | "
                    f"final similarity: {final_similarity.mean():.7f}"
                )

        reward_history.append([np.mean(r) for r in state.episode_rewards])

        # Perform genetic optimization step
        ts, initial_parameters = genetic_update(
            state.train_state, state.rng, initial_parameters, state.episode_rewards
        )

        fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")

        # Set figure title
        fig.suptitle(f"PPO Swarm Similarity/Performance (Total Generations: {generation + 1})")

        def update(frame):
            nonlocal max_similarity_bin_height, max_reward_bin_height
            # Similarity Histogram
            axes[0].cla()
            data = axes[0].hist(similarity_history[frame], bins=200, range=(-1, 1))
            if data[0].max() > max_similarity_bin_height:
                max_similarity_bin_height = data[0].max()
            axes[0].set_ylim(0, max_similarity_bin_height)
            axes[0].set_title(f"Similarity | Generation {frame + 1}")

            # Reward Histogram
            axes[1].cla()
            data = axes[1].hist(reward_history[frame], bins=200, range=(0, 500))
            if data[0].max() > max_reward_bin_height:
                max_reward_bin_height = data[0].max()
            axes[1].set_xlim(0, 500)
            axes[1].set_ylim(0, max_reward_bin_height)
            axes[1].set_title(f"Mean Reward | Generation {frame + 1}")

        ani = FuncAnimation(fig, update, frames=generation + 1)
        ani.save(plot_dir / "ppo_swarm.gif", writer="pillow")

        plt.close()
        # plot(state, plot_dir, generation)

    # Close environments
    environments.close()

    return state, state.episode_rewards, state.timestep_log


if __name__ == "__main__":
    jax_results = run_jax_ppo(total_timesteps=TOTAL_TIMESTEPS)
