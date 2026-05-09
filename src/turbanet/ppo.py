from typing import Callable, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

# Initialize random number generators
SEED = 42
SOLVE_THRESHOLD = 475
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

    @property
    def loss_fn(self) -> Callable:
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
                    -advantages * jnp.clip(ratio, 1 - self.clip_range, 1 + self.clip_range),
                )
            )

            # Value loss (MSE)
            value_loss = 0.5 * jnp.mean((values - returns) ** 2)

            # Entropy loss
            probabilities = jax.nn.softmax(logits)
            entropy_loss = -jnp.sum(probabilities * log_probabilities, axis=-1).mean()

            # Total loss
            loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss
            return (loss, (policy_loss, value_loss, entropy_loss))

        return ppo_loss


class RolloutBuffer(NamedTuple):
    observations: np.ndarray  # (T, E, obs_dim)
    actions: np.ndarray  # (T, E)
    rewards: np.ndarray  # (T, E)
    dones: np.ndarray  # (T, E)
    action_lps: np.ndarray  # (T, E)
    values: np.ndarray  # (T, E)
    last_values: np.ndarray  # (E,)  – bootstrap values
    next_obs: np.ndarray  # (E, obs_dim) – carry-over for next rollout

    # Utility functions
    def compute_gae(self, cfg: PPOConfig) -> tuple[np.ndarray, np.ndarray]:
        """Generalised Advantage Estimation over (T, E) arrays."""
        T, E = self.rewards.shape
        advantages = np.zeros_like(self.rewards)
        last_gae = np.zeros(E, dtype=np.float32)
        for t in reversed(range(T)):
            next_value = self.last_values if t == T - 1 else self.values[t + 1]
            mask = 1.0 - self.dones[t]
            delta = self.rewards[t] + cfg.gamma * next_value * mask - self.values[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * mask * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values
        return advantages, returns


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
