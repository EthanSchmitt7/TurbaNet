from datetime import datetime
from typing import Callable, NamedTuple

import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import torch
from stable_baselines3 import PPO as SB3PPO
from stable_baselines3.common.callbacks import BaseCallback

from turbanet.turba_train_state import TurbaTrainState

# Initialize random number generators
SEED = 42
SOLVE_THRESHOLD = 475
CLIP_RANGE = 0.2
VF_COEFFICIENT = 0.5
ENTROPY_COEFFICIENT = 0.0
np.random.seed(SEED)


# Jax PPO Structures
class PPOConfig(NamedTuple):
    lr: float = 3e-4
    n_steps: int = 2048  # rollout steps per env per update
    n_envs: int = 16  # parallel envs
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    total_timesteps: int = 100_000


class PPOState(NamedTuple):
    """Everything that changes across updates, passed explicitly."""

    train_state: TurbaTrainState  # flax TurbaTrainState (params + opt)
    rng: jax.Array  # PRNG key
    episode_rewards: list  # mutable list – appended in rollout
    timestep_log: list  # matching global-step for each ep
    episode_rewards_live: np.ndarray  # per-env running reward accumulator
    total_steps: int  # global step counter


class RolloutBuffer(NamedTuple):
    observations: np.ndarray  # (T, E, obs_dim)
    actions: np.ndarray  # (T, E)
    rewards: np.ndarray  # (T, E)
    dones: np.ndarray  # (T, E)
    action_lps: np.ndarray  # (T, E)
    values: np.ndarray  # (T, E)
    last_values: np.ndarray  # (E,)  – bootstrap values
    next_obs: np.ndarray  # (E, obs_dim) – carry-over for next rollout


# Loss function
def ppo_loss(
    params: dict, x: jax.Array, y: jax.Array, apply_fn: Callable
) -> tuple[jax.Array, jax.Array]:
    # Unpack output array
    actions = y[:, 0].astype(int)
    old_action_lp = y[:, 1]
    advantages = y[:, 2]
    returns = y[:, 3]

    # Query the current policy
    logits, values = apply_fn({"params": params}, x)
    log_probabilities = jax.nn.log_softmax(logits)
    new_action_lp = log_probabilities[jnp.arange(len(actions)), actions]

    # Compute ratio between old and new policy
    ratio = jnp.exp(new_action_lp - old_action_lp)

    # Policy loss (Clip surrogate loss)
    policy_loss = jnp.mean(
        jnp.maximum(
            -advantages * ratio,
            -advantages * jnp.clip(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE),
        )
    )

    # Value loss (MSE)
    value_loss = 0.5 * jnp.mean((values - returns) ** 2)

    # Entropy loss
    probabilities = jax.nn.softmax(logits)
    entropy_loss = -jnp.sum(probabilities * log_probabilities, axis=-1).mean()

    # Total loss
    loss = policy_loss + VF_COEFFICIENT * value_loss - ENTROPY_COEFFICIENT * entropy_loss
    return (loss, (policy_loss, value_loss, entropy_loss))


# Network Configs
class ActorCritic(nn.Module):
    n_actions: int

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        logits = nn.Dense(self.n_actions)(x)
        value = jnp.squeeze(nn.Dense(1)(x), axis=-1)
        return logits, value


class DualEncoderActorCritic(nn.Module):
    n_actions: int

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        # Policy encoder
        x_p = nn.Dense(64)(x)
        x_p = nn.tanh(x_p)
        x_p = nn.Dense(64)(x_p)
        x_p = nn.tanh(x_p)

        # Value encoder
        x_v = nn.Dense(64)(x)
        x_v = nn.tanh(x_v)
        x_v = nn.Dense(64)(x_v)
        x_v = nn.tanh(x_v)

        logits = nn.Dense(self.n_actions)(x_p)
        value = jnp.squeeze(nn.Dense(1)(x_v), axis=-1)

        return logits, value


# Utility functions
def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalised Advantage Estimation over (T, E) arrays."""
    T, E = rewards.shape
    advantages = np.zeros_like(rewards)
    last_gae = np.zeros(E, dtype=np.float32)
    for t in reversed(range(T)):
        next_value = last_values if t == T - 1 else values[t + 1]
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_gae = delta + gamma * gae_lambda * mask * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


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
    state: PPOState, envs: list, cfg: PPOConfig, obs: np.ndarray
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
        # Forward pass to get logits/values for current observation
        logits, values = state.train_state.predict(jnp.array(current_observation))
        logits = np.array(logits)
        values = np.array(values)

        # Sample actions from logits
        rng, actions, log_probabilities = sample_actions(rng, logits)

        # Store transition data
        buffer_observations[t] = current_observation
        buffer_actions[t] = actions
        buffer_action_lps[t] = log_probabilities
        buffer_values[t] = values

        # Run each environment
        next_observation = np.zeros_like(current_observation)
        for i, env in enumerate(envs):
            # Step environment
            observation, reward, terminated, truncated, _ = env.step(int(actions[i]))

            # Store transition data
            done = terminated or truncated
            buffer_rewards[t, i] = reward
            buffer_dones[t, i] = float(done)
            episode_live[i] += reward
            total_steps += 1

            # Episode finished
            if done:
                # Store episode reward and timestep
                episode_rewards.append(float(episode_live[i]))
                timestep_log.append(total_steps)
                episode_live[i] = 0.0

                # Reset environment
                observation, _ = env.reset()

            # Store next observation
            next_observation[i] = observation

        # Update current observation
        current_observation = next_observation

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
def run_update(state: PPOState, buffer: RolloutBuffer, cfg: PPOConfig) -> tuple[PPOState, float]:
    """
    Run cfg.n_epochs of minibatch PPO updates over buf.
    Returns updated PPOState and mean loss.
    """
    advantages, returns = compute_gae(
        buffer.rewards,
        buffer.values,
        buffer.dones,
        buffer.last_values,
        cfg.gamma,
        cfg.gae_lambda,
    )

    # Flatten (T, E) → (N,)
    observation_flat = buffer.observations.reshape(-1, buffer.observations.shape[-1])
    activation_flat = buffer.actions.flatten()
    action_lp_flat = buffer.action_lps.flatten()
    advantage_flat = advantages.flatten()
    return_flat = returns.flatten()
    N = len(observation_flat)

    train_state = state.train_state
    rng = state.rng
    losses = []

    for _ in range(cfg.n_epochs):
        rng, shuffle_key = jax.random.split(rng)
        idx = np.random.permutation(N)
        for start in range(0, N, cfg.batch_size):
            b = idx[start : start + cfg.batch_size]
            batch = (
                jnp.array(observation_flat[b]),
                jnp.array(activation_flat[b]),
                jnp.array(action_lp_flat[b]),
                jnp.array(advantage_flat[b]),
                jnp.array(return_flat[b]),
            )

            observations, actions, old_action_lp, advantages, returns = batch
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            train_state, loss, _ = train_state.train(
                observations, jnp.array([actions, old_action_lp, advantages, returns]).T
            )

            losses.append(loss.item())

    new_state = PPOState(
        train_state=train_state,
        rng=rng,
        episode_rewards=state.episode_rewards,
        timestep_log=state.timestep_log,
        episode_rewards_live=state.episode_rewards_live,
        total_steps=state.total_steps,
    )
    return new_state, float(np.mean(losses))


# Run SB3
def run_sb3(total_timesteps: int = 100_000) -> tuple[list, list]:
    class EpisodeRewardCallback(BaseCallback):
        """Collect per-episode rewards from SB3."""

        def __init__(self) -> None:
            super().__init__()
            self.episode_rewards: list = []
            self.timestep_log: list = []
            self._running = 0.0

        def _on_step(self) -> bool:
            self._running += self.locals["rewards"][0]
            if self.locals["dones"][0]:
                self.episode_rewards.append(self._running)
                self.timestep_log.append(self.num_timesteps)
                self._running = 0.0

            # Print progress
            if self.num_timesteps % np.floor(total_timesteps // 5) == 0:
                print(
                    f"  [SB3] progress {self.num_timesteps}/{total_timesteps} | mean_rew(last 20):"
                    f" {np.mean(self.episode_rewards[-20:]):7.1f}"
                )
            return True

    env = gym.make("CartPole-v1")
    cb = EpisodeRewardCallback()
    model = SB3PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[64, 64], activation_fn=torch.nn.Tanh),
        seed=SEED,
        device="cpu",
        verbose=0,
    )

    print("\n[SB3] training …")
    model.learn(total_timesteps=total_timesteps, callback=cb, progress_bar=True)
    env.close()

    return cb.episode_rewards, cb.timestep_log


# Run Jax PPO
def run_jax_ppo(total_timesteps: int = 100_000) -> tuple[list, list]:
    # Initialize config
    cfg = PPOConfig(total_timesteps=total_timesteps, n_envs=1)
    env_id = "CartPole-v1"
    rng = jax.random.PRNGKey(SEED)

    # Initialize environments
    environments = [gym.make(env_id) for _ in range(cfg.n_envs)]

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
        swarm_size=1,
        input_size=obs_dim,
        seed=SEED,
    )
    ts = ts.set_loss_fn(ppo_loss)

    # Initialize PPO state
    state = PPOState(
        train_state=ts,
        rng=rng,
        episode_rewards=[],
        timestep_log=[],
        episode_rewards_live=np.zeros(cfg.n_envs, dtype=np.float32),
        total_steps=0,
    )

    # Get initial observations
    observation = np.array(
        [env.reset(seed=SEED + i)[0] for i, env in enumerate(environments)],
        dtype=np.float32,
    )

    # Training loop
    print("\n[JAX] training…")
    n_updates = cfg.total_timesteps // (cfg.n_steps * cfg.n_envs)
    for update in range(1, n_updates + 1):
        # Collect rollout
        state, buffer = collect_rollout(state, environments, cfg, observation)
        observation = buffer.next_obs

        # Run update
        state, loss = run_update(state, buffer, cfg)

        # Log progress
        if state.episode_rewards and update % 5 == 0:
            recent = state.episode_rewards[-20:]
            print(
                f"  [JAX] progress {state.total_steps}/{cfg.total_timesteps} | "
                f"mean_rew(last 20): {np.mean(recent):7.1f} | "
                f"loss: {loss:7.1f}"
            )

    # Close environments
    for env in environments:
        env.close()

    return state.episode_rewards, state.timestep_log


def plot_comparison(jax_rewards: list, output_path: str = "test/integration/ppo.png") -> None:
    final_values = np.array(jax_rewards[-max(1, int(len(jax_rewards) * 0.25)) :], dtype=np.float32)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")
        for lbl in (ax.xaxis.label, ax.yaxis.label, ax.title):
            lbl.set_color("white")

    color = "#00d4aa"

    def smooth(values: list, window: int = 20) -> np.ndarray:
        arr = np.array(values, dtype=np.float32)
        if len(arr) < window:
            return arr
        return np.convolve(arr, np.ones(window) / window, mode="valid")

    # Training curves
    ax1.plot(jax_rewards, color=color, alpha=0.18, linewidth=0.8)
    ax1.plot(smooth(jax_rewards), color=color, linewidth=2.5, label="JAX PPO (smoothed)")
    ax1.axhline(
        SOLVE_THRESHOLD,
        color="white",
        linestyle="--",
        linewidth=1.2,
        alpha=0.55,
        label=f"Solve threshold ({SOLVE_THRESHOLD:.0f})",
    )
    ax1.set_xlabel("Episode", fontsize=11)
    ax1.set_ylabel("Episode Reward", fontsize=11)
    ax1.set_title("Training Curves — CartPole-v1", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9, facecolor="#2a2d37", labelcolor="white", edgecolor="#555")

    # Final-phase distributions
    bins = np.linspace(0, 510, 30)
    ax2.hist(
        final_values, bins=bins, color=color, alpha=0.65, label=f"μ={final_values.mean():.1f}"
    )
    ax2.axvline(SOLVE_THRESHOLD, color="white", linestyle="--", linewidth=1.2, alpha=0.55)
    ax2.set_xlabel("Episode Reward (final 25% of training)", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Final-Phase Reward Distribution", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9, facecolor="#2a2d37", labelcolor="white", edgecolor="#555")

    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


if __name__ == "__main__":
    jax_results = run_jax_ppo(total_timesteps=75_000)
    plot_comparison(
        jax_results[0], f"test/integration/ppo_jax_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    )

    sb3_results = run_sb3(75_000)
    plot_comparison(
        sb3_results[0], f"test/integration/ppo_sb3_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    )
