from datetime import datetime
from typing import NamedTuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from turbanet.ppo import ActorCritic, DualEncoderActorCritic, PPOConfig, RolloutBuffer
from turbanet.turba_train_state import TurbaTrainState

# Initialize random number generators
SEED = 42
SOLVE_THRESHOLD = 475
TOTAL_TIMESTEPS = 500_000

SWARM_SIZE = 10000

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

    # Training loop
    print("\n[JAX] training…")
    losses = []
    n_updates = cfg.total_timesteps // cfg.n_steps
    for update in range(1, n_updates + 1):
        # Collect rollout
        state, buffer = collect_rollout(state, environments, cfg, observation)
        observation = buffer.next_obs

        # Run update
        state, loss = run_update(state, buffer, cfg)
        losses.append(loss)

        # Log progress
        if state.episode_rewards and update % 5 == 0:
            recent = [r[-20:] for r in state.episode_rewards]
            print(
                f"  [JAX] progress {state.total_steps}/{cfg.total_timesteps} | "
                f"mean_rew(last 20): {np.mean(recent):7.1f} | "
                f"loss: {loss.mean():7.1f}"
            )

    # Close environments
    environments.close()

    return state, state.episode_rewards, state.timestep_log


def new_plot(state: PPOState) -> None:
    rewards = []
    for t_data, r_data in zip(state.timestep_log, state.episode_rewards):
        for t, r in zip(t_data, r_data):
            rewards.append([t, r])
    rewards = np.array(rewards, dtype=np.float32)

    fig, axes = plt.subplots(nrows=3, figsize=(6, 8), layout="constrained")
    # axes[0].scatter(x=rewards[:, 0], y=rewards[:, 1], marker=".", color="C0", alpha=0.1)

    [axes[0].plot(r, color="C0", alpha=0.1) for r in state.episode_rewards]
    axes[0].set_title("Line plot with alpha")

    cmap = plt.colormaps["plasma"]
    cmap = cmap.with_extremes(bad=cmap(0))
    h, xedges, yedges = np.histogram2d(x=rewards[:, 0], y=rewards[:, 1], bins=[400, 100])
    pcm = axes[1].pcolormesh(
        xedges, yedges, h.T, cmap=cmap, norm="log", vmax=1.5e2, rasterized=True
    )
    fig.colorbar(pcm, ax=axes[1], label="# points", pad=0)
    axes[1].set_title("2d histogram and log color scale")

    pcm = axes[2].pcolormesh(xedges, yedges, h.T, cmap=cmap, vmax=1.5e2, rasterized=True)
    fig.colorbar(pcm, ax=axes[2], label="# points", pad=0)
    axes[2].set_title("2d histogram and linear color scale")

    plt.savefig(f"test/integration/ppo_swarm_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    jax_results = run_jax_ppo(total_timesteps=TOTAL_TIMESTEPS)
    new_plot(jax_results[0])
