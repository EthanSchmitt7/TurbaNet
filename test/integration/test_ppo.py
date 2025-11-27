from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from turbanet import TurbaTrainState, mse

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax.typing import ArrayLike

LOGGER = logging.getLogger("turba")
LOG_LEVEL = logging.INFO

# Parameters
NUM_ORGANISMS = 1

# Training Parameters
SUPERVISED = False
EPISODES = int(2e6)
LR = 1e-5
EPOCHS = 10
BATCH_SIZE = 512
CLIP_RANGE = 0.2
ENTROPY_COEFFICIENT = 0.005

# Logging
try:
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir="test/tensorboard2")
    print("Logging to test/tensorboard")

except ImportError:
    writer = None
    print("Torch not installed, tensorboard logging disabled")

# Configure RNGs
np.random.seed(0)


def definitive_cases() -> np.ndarray:
    environment = np.zeros((5, 9))
    environment[0, 4] = Tile.food_organism.value  # Eat
    environment[1, 1] = Tile.food.value  # Up
    environment[2, 3] = Tile.food.value  # Left
    environment[3, 5] = Tile.food.value  # Right
    environment[4, 7] = Tile.food.value  # Down
    return environment


def subjective_cases() -> np.ndarray:
    environment = np.zeros((11, 9))
    environment[0, 1] = Tile.invalid.value  # Down/Right/Left
    environment[1, 3] = Tile.invalid.value  # Right/Up/Down
    environment[2, 5] = Tile.invalid.value  # Left/Up/Down
    environment[3, 7] = Tile.invalid.value  # Up/Right/Left

    # Up/Down/Left/Right
    environment[4, 1] = Tile.food.value
    environment[4, 3] = Tile.food.value
    environment[4, 5] = Tile.food.value
    environment[4, 7] = Tile.food.value

    # Up/Down
    environment[5, 1] = Tile.food.value
    environment[5, 7] = Tile.food.value

    # Right/Left
    environment[6, 3] = Tile.food.value
    environment[6, 5] = Tile.food.value

    # Up/Right
    environment[7, 1] = Tile.food.value
    environment[7, 5] = Tile.food.value

    # Down/Left
    environment[8, 3] = Tile.food.value
    environment[8, 7] = Tile.food.value

    # Up/Left
    environment[9, 1] = Tile.food.value
    environment[9, 3] = Tile.food.value

    # Down/Right
    environment[10, 5] = Tile.food.value
    environment[10, 7] = Tile.food.value

    return environment


def correct_decision(environments: np.ndarray) -> np.ndarray:
    answered = np.full(environments.shape[0], fill_value=False, dtype=bool)
    answers = np.full((environments.shape[0], len(Decision)), fill_value=-0.1)

    # If the current cell has food, tell the organism to eat
    eat = environments[:, 4] == Tile.food_organism.value
    answers[eat, Decision.eat.value] = 2.0
    answered[eat] = True

    # If nearby cells have food and no other organisms, move towards food
    up = environments[:, 1] == Tile.food.value
    left = environments[:, 3] == Tile.food.value
    right = environments[:, 5] == Tile.food.value
    down = environments[:, 7] == Tile.food.value

    answers[np.logical_and(up, ~answered), Decision.up.value] = 1.0
    answers[np.logical_and(left, ~answered), Decision.left.value] = 1.0
    answers[np.logical_and(right, ~answered), Decision.right.value] = 1.0
    answers[np.logical_and(down, ~answered), Decision.down.value] = 1.0

    answered[up] = True
    answered[left] = True
    answered[right] = True
    answered[down] = True

    # If nearby cells have food and other organisms, move towards food
    up = environments[:, 1] == Tile.food_organism.value
    left = environments[:, 3] == Tile.food_organism.value
    right = environments[:, 5] == Tile.food_organism.value
    down = environments[:, 7] == Tile.food_organism.value

    answers[np.logical_and(up, ~answered), Decision.up.value] = 1.0
    answers[np.logical_and(left, ~answered), Decision.left.value] = 1.0
    answers[np.logical_and(right, ~answered), Decision.right.value] = 1.0
    answers[np.logical_and(down, ~answered), Decision.down.value] = 1.0

    answered[up] = True
    answered[left] = True
    answered[right] = True

    # Add move options if no food is nearby
    answers[~answered, Decision.up.value] = 1.0
    answers[~answered, Decision.left.value] = 1.0
    answers[~answered, Decision.right.value] = 1.0
    answers[~answered, Decision.down.value] = 1.0

    return answers


def actor_loss_fn(
    params: dict, x: ArrayLike, y: ArrayLike, actor_apply_fn: Callable
) -> tuple[ArrayLike, ArrayLike]:
    # Deconstruct y array
    actions = y[:, 0].astype(int)
    advantages = y[:, 1]
    old_action_log_probabilities = y[:, 2]

    # Get the output of the actor
    logits = actor_apply_fn({"params": params}, x)
    log_probabilities = jax.nn.log_softmax(logits)
    probabilities = jnp.exp(log_probabilities)
    action_log_probabilities = jnp.take_along_axis(
        log_probabilities, actions[:, None], axis=-1
    ).squeeze()

    # Ratio between old and new policy, should be one at the first iteration
    ratio = jnp.exp(action_log_probabilities - old_action_log_probabilities)

    # clipped surrogate loss
    policy_loss_1 = advantages * ratio
    policy_loss_2 = advantages * jnp.clip(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE)
    policy_loss = -jnp.mean(jnp.minimum(policy_loss_1, policy_loss_2))

    # Adding entropy to encourage exploration
    entropy = -(log_probabilities * probabilities).sum(axis=-1).mean()
    entropy_loss = -entropy

    # Total loss
    loss = policy_loss + ENTROPY_COEFFICIENT * entropy_loss

    return loss, logits


class Decision(Enum):
    idle = 0
    left = 1
    right = 2
    up = 3
    down = 4
    eat = 5

    def __str__(self) -> str:
        return self.name


class Tile(Enum):
    invalid = -1
    empty = 0
    food = 1
    organism = 2
    food_organism = 3

    def __str__(self) -> str:
        return self.name


class Actor(nn.Module):
    hidden_layers: int = 1
    hidden_size: int = 64
    output_size: int = 1
    gain: float = 0.01

    @nn.compact
    def __call__(self, x):  # noqa: ANN001, ANN204
        for _ in range(self.hidden_layers):
            x = nn.Dense(
                self.hidden_size, kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2))
            )(x)
            x = nn.tanh(x)
        x = nn.Dense(self.output_size, kernel_init=jax.nn.initializers.orthogonal(self.gain))(x)
        return x  # noqa: RET504


class Critic(Actor):
    gain = 1.0


def create_agents() -> tuple[TurbaTrainState, TurbaTrainState, TurbaTrainState]:
    # Decision Making | Policy Network | Actor
    actor = TurbaTrainState.swarm(
        Actor(hidden_layers=2, output_size=len(Decision)),
        optimizer=optax.adam(LR),
        swarm_size=NUM_ORGANISMS,
        sample_input=np.zeros((1, 9)),
    )

    # Reward Prediction | Value Network | Critic
    critic = TurbaTrainState.swarm(
        Critic(hidden_layers=2),
        optimizer=optax.adam(LR),
        swarm_size=NUM_ORGANISMS,
        sample_input=np.zeros((1, 9)),
    )

    return actor, critic


def create_random_states(swarm_size: int, batch_size: int) -> np.ndarray:
    states = np.random.randint(0, 4, (swarm_size, batch_size, 9))
    states[:, :, 4] = np.random.choice((2, 3), (swarm_size, batch_size), p=(0.8, 0.2))

    return states


def get_reward(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    correct = correct_decision(state.reshape(-1, 9)).reshape(NUM_ORGANISMS, BATCH_SIZE, 6)

    action = jnp.expand_dims(action, 2)
    reward = jnp.take_along_axis(correct, action, axis=-1).squeeze(axis=2)

    return reward


def execute_episode(
    episode: int, rng, actor: TurbaTrainState, critic: TurbaTrainState
) -> tuple[Any, TurbaTrainState, TurbaTrainState, float]:
    # Create random environment states and get the correct decisions
    states = create_random_states(NUM_ORGANISMS, BATCH_SIZE)

    # Get the action probabilities
    logits = actor.predict(states)
    log_probabilities = jax.nn.log_softmax(logits)
    probabilities = jnp.exp(log_probabilities)
    entropy = -(log_probabilities * probabilities).sum(axis=-1).mean(axis=-1)

    # Sample actions in acordance with action probabilities
    rng, _rng = jax.random.split(rng)
    actions = jax.random.categorical(_rng, logits, shape=(NUM_ORGANISMS, BATCH_SIZE))

    # Get the reward
    rewards = get_reward(states, actions)
    predicted_reward = critic.predict(states)

    # Calculate advantage for each organism
    advantages = (rewards - predicted_reward.squeeze()).reshape(NUM_ORGANISMS, BATCH_SIZE, 1)
    actions = actions.reshape(NUM_ORGANISMS, BATCH_SIZE, 1)
    rewards = rewards.reshape(NUM_ORGANISMS, BATCH_SIZE, 1)

    # Standardize advantage
    advantage_mean = advantages.mean(axis=1, keepdims=True)
    advantage_std = advantages.std(axis=1, keepdims=True)
    std_advantages = (advantages - advantage_mean) / (advantage_std + 1e-8)

    # Prepare data for loss function
    action_log_probabilities = jnp.take_along_axis(log_probabilities, actions, axis=-1)
    y = jnp.concatenate((actions, std_advantages, action_log_probabilities), axis=2)

    actor_losses = []
    critic_losses = []
    all_values = []
    for _ in range(EPOCHS):
        # Reinforcement
        actor, actor_loss, _ = actor.train(states, y, actor_loss_fn)
        critic, critic_loss, values = critic.train(states, rewards, mse)

        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)
        all_values.append(values)

    if writer is not None:
        if NUM_ORGANISMS == 1:
            # Scalars
            writer.add_scalar("Loss/Actor", np.array(actor_loss), episode)
            writer.add_scalar("Loss/Critic", np.array(critic_loss), episode)
            writer.add_scalar("Loss/Entropy", -np.array(entropy), episode)
            writer.add_scalar("Other/Reward", np.array(rewards).mean(), episode)
            writer.add_scalar("Other/Advantage", np.array(advantages).mean(), episode)
            writer.add_scalar("Other/Values", np.array(all_values).mean(), episode)
            writer.add_scalar("Other/Entropy", np.array(entropy), episode)

            # Histograms
            writer.add_histogram("Actor/Logits", np.array(logits).squeeze(), episode)
            writer.add_histogram("Actor/Probabilities", np.array(probabilities).squeeze(), episode)
            writer.add_histogram(
                "Actor/Log Probabilities", np.array(log_probabilities).squeeze(), episode
            )
            writer.add_histogram(
                "Actor/Actions",
                np.array(actions).astype(int).squeeze().round(),
                episode,
                bins=len(Decision),
            )
        else:
            writer.add_histogram("Loss/Actor", np.array(actor_loss), episode)
            writer.add_histogram("Loss/Critic", np.array(critic_loss), episode)
            writer.add_histogram("Loss/Entropy", -np.array(entropy), episode)
            writer.add_histogram("Other/Reward", np.array(rewards).mean(axis=1), episode)
            writer.add_histogram("Other/Advantage", np.array(advantages).mean(axis=1), episode)
            writer.add_histogram("Other/Values", np.array(all_values).mean(axis=1), episode)
            writer.add_histogram("Other/Entropy", np.array(entropy), episode)

    return rng, actor, critic, np.mean(actor_losses), np.mean(critic_losses), np.mean(rewards)


def main() -> None:
    actor, critic = create_agents()

    values = []
    rng = jax.random.PRNGKey(0)
    try:
        for e in range(EPISODES):
            LOGGER.info(f"---- Episode {e} ----")
            rng, actor, critic, actor_loss, critic_loss, reward = execute_episode(
                e, rng, actor, critic
            )
            values.append(reward)

    except KeyboardInterrupt:
        LOGGER.info("Training Stopped")

    if writer is None:
        import matplotlib.pyplot as plt

        plt.plot(values, "o")
        plt.xlabel("Episode")
        if SUPERVISED:
            plt.ylabel("Loss")
        else:
            plt.ylabel("Average Reward")
        plt.show()

    cases = definitive_cases()
    cases[1:, 4] = np.full((cases.shape[0] - 1,), 2)
    logits = actor.predict(cases)
    print("Definitive Cases")
    for logits, case in zip(logits[0], cases):
        print(f"\nCase: \n{case.reshape(3, 3)}")
        probabilities = jnp.exp(logits) / jnp.sum(jnp.exp(logits), axis=-1, keepdims=True)

        for p, d in zip(probabilities, Decision._member_names_):
            print(f"{d}: {np.round(p, 2) * 100:.2f}%")

    cases = subjective_cases()
    cases[:, 4] = np.full((cases.shape[0],), 2)
    logits = actor.predict(cases)
    print("\nSubjective Cases")
    for logits, case in zip(logits[0], cases):
        print(f"\nCase: \n{case.reshape(3, 3)}")
        probabilities = jnp.exp(logits) / jnp.sum(jnp.exp(logits), axis=-1, keepdims=True)

        for p, d in zip(probabilities, Decision._member_names_):
            print(f"{d}: {np.round(p, 2) * 100:.2f}%")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    LOGGER.setLevel(LOG_LEVEL)
    handler = logging.StreamHandler()
    handler.setLevel(LOG_LEVEL)
    formatter = logging.Formatter("%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)

    main()
