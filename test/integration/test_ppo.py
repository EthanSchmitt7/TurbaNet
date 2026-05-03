from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import distrax
import gymnasium as gym
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from flax import linen as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from torch.utils.tensorboard import SummaryWriter

from turbanet import TurbaTrainState

writer = SummaryWriter()


if TYPE_CHECKING:
    from collections.abc import Callable

    from jax.typing import ArrayLike

LOGGER = logging.getLogger("turba")
LOG_LEVEL = logging.INFO

TOTAL_STEPS = 1e7
NUM_NETWORKS = 1
LR = 3e-4
EPOCHS = 10
ROLLOUT_LENGTH = 1000
BATCH_SIZE = 2000
CLIP_RANGE = 0.2
CLIP_RANGE_VF = 0.2
ENTROPY_COEFFICIENT = 0.01
CRITIC_COEFFICIENT = 0.5
GAMMA = 0.99
GAE_LAMBDA = 0.95

EVAL_FREQ = 1000
EVAL_EPISODES = 3


@dataclass
class EpisodeData:
    steps: int = 0
    observations = jnp.empty((0, 4))
    actions = jnp.empty((0, 1))
    log_probabilities = jnp.empty((0, 1))
    rewards = jnp.empty((0, 1))
    values = jnp.empty((0, 1))
    terminated = jnp.empty((0, 1))
    truncated = jnp.empty((0, 1))

    def add_step(
        self,
        observation: ArrayLike,
        action: ArrayLike,
        log_probabilities: ArrayLike,
        reward: ArrayLike,
        value: ArrayLike,
        terminated: ArrayLike,
        truncated: ArrayLike,
    ) -> None:
        self.steps += 1
        self.observations = np.append(self.observations, observation.reshape(1, 4), axis=0)
        self.log_probabilities = np.append(self.log_probabilities, log_probabilities)
        self.actions = np.append(self.actions, action)
        self.rewards = np.append(self.rewards, reward)
        self.values = np.append(self.values, value)
        self.terminated = np.append(self.terminated, terminated)
        self.truncated = np.append(self.truncated, truncated)

    @property
    def total_reward(self) -> ArrayLike:
        return jnp.sum(self.rewards).item()

    @property
    def is_terminated(self) -> bool:
        return jnp.any(self.terminated).item()

    @property
    def is_truncated(self) -> bool:
        return jnp.any(self.truncated).item()

    def compute_advantage(self, gamma: float = 0.99, gae_lambda: float = 0.95) -> None:
        self.advantages, self.returns = compute_gae(
            self.rewards, self.values, self.terminated, self.truncated, gamma, gae_lambda
        )


@dataclass
class RolloutData:
    episodes: list[EpisodeData] = None

    def __post_init__(self) -> None:
        self.episodes = []

    def add_episode(self, episode: EpisodeData) -> None:
        self.episodes.append(episode)

    def __getitem__(self, index: int) -> EpisodeData:
        return self.episodes[index]

    def as_batch(self) -> tuple[ArrayLike, tuple[ArrayLike, ...]]:
        """Concatenate all episodes into flat arrays ready for training"""
        observations = jnp.concatenate([e.observations for e in self.episodes])
        actions = jnp.concatenate([e.actions for e in self.episodes])
        log_probabilities = jnp.concatenate([e.log_probabilities for e in self.episodes])
        values = jnp.concatenate([e.values for e in self.episodes])
        advantages = jnp.concatenate([e.advantages for e in self.episodes])
        returns = jnp.concatenate([e.returns for e in self.episodes])

        # TODO: Should be done at minibatch level - Standardize advantage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.advantages = advantages

        return (
            jnp.array(observations),
            jnp.array(
                [
                    jnp.array(actions),
                    jnp.array(log_probabilities),
                    jnp.array(values),
                    jnp.array(advantages),
                    jnp.array(returns),
                ]
            ),
        )

    @property
    def mean_episode_return(self) -> float:
        return np.mean([e.returns for e in self.episodes])

    @property
    def total_steps(self) -> int:
        return sum(e.steps for e in self.episodes)

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)


from flax import nnx


class Test(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.pi = []
        self.pi.append(
            nnx.Linear(4, 64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)), rngs=rngs)
        )
        self.pi.append(
            nnx.Linear(64, 64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)), rngs=rngs)
        )

        self.vf = []
        self.vf.append(
            nnx.Linear(4, 64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)), rngs=rngs)
        )
        self.vf.append(
            nnx.Linear(64, 64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)), rngs=rngs)
        )

        self.actor_head = nnx.Linear(
            64, 2, kernel_init=nn.initializers.orthogonal(0.01), rngs=rngs
        )
        self.critic_head = nnx.Linear(64, 1, kernel_init=nn.initializers.orthogonal(), rngs=rngs)

    def __call__(self, x):
        # Actor
        pi_x = x
        for p_layer in self.pi:
            pi_x = nnx.tanh(p_layer(pi_x))

        logits = self.actor_head(pi_x)

        # Critic
        vf_x = x
        for v_layer in self.vf:
            vf_x = nnx.tanh(v_layer(vf_x))

        value = self.critic_head(vf_x)

        return logits, value


class ActorCritic(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Policy Extractor
        pi_x = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        pi_x = nn.tanh(pi_x)
        pi_x = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(pi_x)
        pi_x = nn.tanh(pi_x)

        # Value Extractor
        vf_x = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        vf_x = nn.tanh(vf_x)
        vf_x = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(vf_x)
        vf_x = nn.tanh(vf_x)

        # Actor head - outputs logits over actions (Action Net)
        logits = nn.Dense(2, kernel_init=nn.initializers.orthogonal(0.01))(pi_x)

        # Critic head - outputs scalar value (Value Net)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal())(vf_x)

        return logits, value


def compute_gae(
    rewards: ArrayLike,
    values: ArrayLike,
    terminated: ArrayLike,
    truncated: ArrayLike,
    gamma: float,
    gae_lambda: float,
) -> tuple[ArrayLike, ArrayLike]:
    # Convert to numpy
    last_values = values[-1]

    last_gae_lam = 0
    buffer_size = len(rewards)
    advantages = jnp.zeros(shape=values.shape)
    for step in reversed(range(buffer_size)):
        if step == buffer_size - 1:
            next_non_terminal = 1.0 - terminated[step]
            next_values = last_values
        else:
            next_non_terminal = 1.0
            next_values = values[step + 1]
        delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
        last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        advantages = advantages.at[step].set(last_gae_lam)

    returns = advantages + values

    return advantages, returns


gae_vmap = jax.vmap(compute_gae, in_axes=(0, 0, 0, 0, None, None))


def ppo_loss(
    params: dict, x: ArrayLike, y: ArrayLike, apply_fn: Callable
) -> tuple[ArrayLike, ArrayLike]:
    # Deconstruct y array
    actions = y[0]
    old_log_probabilities = y[1]
    old_values = y[2]
    advantages = y[3]
    returns = y[4]

    # Query the actor-critic network
    new_logits, new_values = apply_fn({"params": params}, x)
    new_values = new_values.squeeze(-1)

    # Get new log probabilities
    dist = distrax.Categorical(logits=new_logits)
    new_log_probabilities = dist.log_prob(actions)

    # Ratio between old and new policy, should be one at the first iteration
    ratio = jnp.exp(new_log_probabilities - old_log_probabilities)

    # Clipped surrogate loss
    policy_loss_1 = advantages * ratio
    policy_loss_2 = advantages * jnp.clip(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE)
    policy_loss = -jnp.mean(jnp.minimum(policy_loss_1, policy_loss_2))

    # Critic loss
    values_clipped = old_values + jnp.clip(new_values - old_values, -CLIP_RANGE, CLIP_RANGE)
    critic_loss = jnp.mean(
        jnp.maximum((new_values - returns) ** 2, (values_clipped - returns) ** 2)
    )

    # Critic loss
    # No clipping
    values_pred = new_values

    # # Vf Clipping
    # # NOTE: this depends on the reward scaling
    # values_pred = old_values + jnp.clip(new_values - old_values, -CLIP_RANGE_VF, CLIP_RANGE_VF)

    # Value loss using the TD(gae_lambda) target
    critic_loss = jnp.mean((returns - values_pred) ** 2)

    # Adding entropy to encourage exploration
    entropy_loss = -dist.entropy().mean()

    # Total loss
    loss = policy_loss + CRITIC_COEFFICIENT * critic_loss + ENTROPY_COEFFICIENT * entropy_loss

    return loss, new_logits


ppo_loss_fn = jax.jit(jax.vmap(ppo_loss, in_axes=(0, 0, 0, None)), static_argnames=("apply_fn",))


def debug_loss(params, x, y, apply_fn):
    # Deconstruct y array
    actions = y[0]
    old_log_probabilities = y[1]
    old_values = y[2]
    advantages = y[3]
    returns = y[4]

    # Query the actor-critic network
    new_logits, new_values = apply_fn({"params": params}, x)
    new_values = new_values.squeeze(-1)

    # Get new log probabilities
    dist = distrax.Categorical(logits=new_logits)
    new_log_probabilities = dist.log_prob(actions)

    # Ratio between old and new policy, should be one at the first iteration
    ratio = jnp.exp(new_log_probabilities - old_log_probabilities)

    # Clipped surrogate loss
    policy_loss_1 = advantages * ratio
    policy_loss_2 = advantages * jnp.clip(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE)
    policy_loss = -jnp.mean(jnp.minimum(policy_loss_1, policy_loss_2))

    # Clip Fraction
    clip_fraction = jnp.mean((jnp.abs(ratio - 1) > CLIP_RANGE))

    # KL Divergence
    log_ratio = new_log_probabilities - old_log_probabilities
    approx_kl_div = jnp.mean((jnp.exp(log_ratio) - 1) - log_ratio)

    # Critic loss
    # No clipping
    values_pred = new_values

    # # Vf Clipping
    # # NOTE: this depends on the reward scaling
    # values_pred = old_values + jnp.clip(new_values - old_values, -CLIP_RANGE_VF, CLIP_RANGE_VF)

    # Value loss using the TD(gae_lambda) target
    critic_loss = jnp.mean((returns - values_pred) ** 2)

    # Adding entropy to encourage exploration
    entropy_loss = -dist.entropy().mean()

    return policy_loss, critic_loss, entropy_loss, clip_fraction, approx_kl_div


def main() -> None:
    # Training Environment
    env = gym.make("CartPole-v1")
    obs, info = env.reset()

    # Test Environment
    eval_env = gym.make("CartPole-v1", render_mode="human")

    # Network
    # optimizer = optax.adam(3e-4)
    optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4))
    network = TurbaTrainState.swarm(
        ActorCritic(),
        optimizer=optimizer,
        swarm_size=NUM_NETWORKS,
        sample_input=jnp.zeros((1, 4)),
    )

    # Initialize RNG
    key = jr.PRNGKey(0)

    iteration = 0
    while iteration < TOTAL_STEPS:
        # Training
        for _ in range(EVAL_FREQ):
            iteration += 1

            # Initialization
            rollout_data = RolloutData()
            episode = EpisodeData()
            episode.add_step(obs, 0, 0, 0, 0, False, False)

            # Rollout
            for i in range(ROLLOUT_LENGTH - 1):
                logit, value = network.predict(obs)

                dist = distrax.Categorical(logits=logit)
                key, subkey = jr.split(key)
                action = dist.sample(seed=subkey)
                log_probability = dist.log_prob(action)

                obs, reward, term, trunc, info = env.step(np.array(action[0]))

                if i == ROLLOUT_LENGTH - 1:
                    trunc = True

                episode.add_step(obs, action, log_probability, reward, value, term, trunc)
                if term or trunc:
                    rollout_data.add_episode(episode)
                    episode.compute_advantage(GAMMA, GAE_LAMBDA)

                    obs, info = env.reset()
                    episode = EpisodeData()

            # Get rollout data for training
            x, y = rollout_data.as_batch()

            # TODO: Remove after debugging
            p_loss, c_loss, e_loss, clip_fraction, approx_kl_div = debug_loss(
                network.get_state(0).params, x, y, network.get_state(0).apply_fn
            )
            writer.add_scalar("train/policy_loss", p_loss.item(), iteration * BATCH_SIZE)
            writer.add_scalar("train/critic_loss", c_loss.item(), iteration * BATCH_SIZE)
            writer.add_scalar("train/entropy_loss", e_loss.item(), iteration * BATCH_SIZE)
            writer.add_scalar("train/clip_fraction", clip_fraction.item(), iteration * BATCH_SIZE)
            writer.add_scalar("train/approx_kl_div", approx_kl_div.item(), iteration * BATCH_SIZE)

            # Update
            for _ in range(EPOCHS):
                network, loss, _ = network.train(x, y, ppo_loss)

            ep_mean_reward = np.mean([episode.total_reward for episode in rollout_data.episodes])
            LOGGER.info(
                f"Iteration: {iteration} | "
                f"Steps: {iteration * ROLLOUT_LENGTH} | "
                f"Average Loss: {loss.mean():.4f} | "
                f"Average Reward: {ep_mean_reward:.4f}"
            )

            var_y = np.var(y[4])
            explained_variance = np.nan if var_y == 0 else float(1 - np.var(y[4] - y[2]) / var_y)

            ep_len_mean = np.mean([episode.steps for episode in rollout_data.episodes])

            # # Logs
            writer.add_scalar("rollout/ep_len_mean", ep_len_mean, iteration * BATCH_SIZE)
            writer.add_scalar("rollout/ep_mean_reward", ep_mean_reward, iteration * BATCH_SIZE)
            writer.add_scalar("train/total_loss", loss.item(), iteration * BATCH_SIZE)
            writer.add_scalar(
                "train/explained_variance", explained_variance, iteration * BATCH_SIZE
            )

            # self.logger.record("train/entropy_loss", np.mean(entropy_losses))
            # self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
            # self.logger.record("train/value_loss", np.mean(value_losses))
            # self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
            # self.logger.record("train/clip_fraction", np.mean(clip_fractions))
            # self.logger.record("train/loss", loss.item())
            # self.logger.record("train/explained_variance", explained_var)
            # if hasattr(self.policy, "log_std"):
            #     self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

            # self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            # self.logger.record("train/clip_range", clip_range)
            writer.flush()

        # Evaluation
        for _ in range(EVAL_EPISODES):
            obs, info = eval_env.reset()

            term = False
            trunc = False
            while not term or not trunc:
                logit, value = network.predict(obs)

                dist = distrax.Categorical(logits=logit)
                key, subkey = jr.split(key)
                action = dist.sample(seed=subkey)

                obs, reward, term, trunc, info = eval_env.step(np.array(action[0]))


def train_sb3_cartpole():
    env = gym.make("CartPole-v1")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=LR,
        n_steps=ROLLOUT_LENGTH,  # rollout length
        batch_size=BATCH_SIZE,
        n_epochs=EPOCHS,  # epochs per update
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENTROPY_COEFFICIENT,  # entropy coefficient
        vf_coef=CRITIC_COEFFICIENT,  # critic coefficient
        verbose=1,
        tensorboard_log="./ppo_cartpole_tensorboard/",
    )

    eval_env = gym.make("CartPole-v1", render_mode="human")
    eval_callback = EvalCallback(
        eval_env,
        eval_freq=EVAL_FREQ * ROLLOUT_LENGTH,
        n_eval_episodes=EVAL_EPISODES,
        verbose=1,
    )

    model.learn(total_timesteps=TOTAL_STEPS, callback=eval_callback)
    return model


if __name__ == "__main__":
    LOGGER.setLevel(LOG_LEVEL)
    handler = logging.StreamHandler()
    handler.setLevel(LOG_LEVEL)
    formatter = logging.Formatter("%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)

    main()
