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

    writer = SummaryWriter(log_dir="test/tensorboard")
    print("Logging to test/tensorboard")

except ImportError:
    writer = None
    print("Torch not installed, tensorboard logging disabled")
# Configure RNGs
np.random.seed(0)


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


def batched_correct_decision(environments: jnp.ndarray) -> jnp.ndarray:
    """
    environments: (num_orgs, batch_size, 9)
    returns:      (num_orgs, batch_size, len(Decision))
    """
    num_orgs, batch_size, _ = environments.shape
    num_decisions = len(Decision)

    answered = np.full((num_orgs, batch_size), False, dtype=bool)
    answers = np.full((num_orgs, batch_size, num_decisions), -0.1, dtype=float)

    # Case 1: current cell has food+organism -> eat
    eat = environments[..., 4] == Tile.food_organism.value  # now 3
    answers[eat, Decision.eat.value] = 2.0
    answered[eat] = True

    # Case 2: adjacent plain food (Tile.food)
    up = environments[..., 1] == Tile.food.value
    left = environments[..., 3] == Tile.food.value
    right = environments[..., 5] == Tile.food.value
    down = environments[..., 7] == Tile.food.value

    mask = np.logical_and(up, ~answered)
    answers[mask, Decision.up.value] = 1.0

    mask = np.logical_and(left, ~answered)
    answers[mask, Decision.left.value] = 1.0

    mask = np.logical_and(right, ~answered)
    answers[mask, Decision.right.value] = 1.0

    mask = np.logical_and(down, ~answered)
    answers[mask, Decision.down.value] = 1.0

    answered[up] = True
    answered[left] = True
    answered[right] = True
    answered[down] = True

    # Case 3: adjacent food+organism (Tile.food_organism == 3)
    up = environments[..., 1] == Tile.food_organism.value
    left = environments[..., 3] == Tile.food_organism.value
    right = environments[..., 5] == Tile.food_organism.value
    down = environments[..., 7] == Tile.food_organism.value

    mask = np.logical_and(up, ~answered)
    answers[mask, Decision.up.value] = 1.0

    mask = np.logical_and(left, ~answered)
    answers[mask, Decision.left.value] = 1.0

    mask = np.logical_and(right, ~answered)
    answers[mask, Decision.right.value] = 1.0

    mask = np.logical_and(down, ~answered)
    answers[mask, Decision.down.value] = 1.0

    answered[up] = True
    answered[left] = True
    answered[right] = True
    answered[down] = True

    # Final fallback: if still unanswered, enable move options
    unanswered = ~answered
    answers[unanswered, Decision.up.value] = 1.0
    answers[unanswered, Decision.left.value] = 1.0
    answers[unanswered, Decision.right.value] = 1.0
    answers[unanswered, Decision.down.value] = 1.0

    return answers


# Precompute decision indices (4 directions)
_dir_decs = jnp.array(
    [Decision.up.value, Decision.left.value, Decision.right.value, Decision.down.value],
    dtype=jnp.int32,
)


@jax.jit
def jit_correct_decision(environments: jnp.ndarray) -> jnp.ndarray:
    """
    JIT-compatible version of your batched_correct_decision function.
    Preserves sequential mask logic exactly.
    """
    num_orgs, batch_size, _ = environments.shape
    num_decisions = len(Decision)

    answers = jnp.full((num_orgs, batch_size, num_decisions), -0.1, dtype=jnp.float32)
    answered = jnp.zeros((num_orgs, batch_size), dtype=bool)

    # --- Case 1: eat center ---
    eat_mask = environments[..., 4] == Tile.food_organism.value
    answers = answers.at[..., Decision.eat.value].set(
        jnp.where(eat_mask, 2.0, answers[..., Decision.eat.value])
    )
    answered = answered | eat_mask

    # --- Helper function to update neighbors ---
    def update_neighbors(answers, answered, neigh_indices, tile_value, reward):
        """
        answers: (num_orgs, batch, num_decisions)
        answered: (num_orgs, batch)
        neigh_indices: indices of neighbors to check (1,3,5,7)
        tile_value: value to check for in environments
        reward: reward to assign
        """
        new_answered = answered
        for i, dec_idx in enumerate(_dir_decs):
            mask = (environments[..., neigh_indices[i]] == tile_value) & (~answered)
            answers = answers.at[..., dec_idx].set(jnp.where(mask, reward, answers[..., dec_idx]))
            new_answered = new_answered | mask
        return answers, new_answered

    # --- Case 2: adjacent plain food ---
    answers, answered = update_neighbors(answers, answered, [1, 3, 5, 7], Tile.food.value, 1.0)

    # --- Case 3: adjacent food+organism ---
    answers, answered = update_neighbors(
        answers, answered, [1, 3, 5, 7], Tile.food_organism.value, 1.0
    )

    # --- Case 4: fallback moves ---
    unresolved = ~answered
    for dec_idx in _dir_decs:
        answers = answers.at[..., dec_idx].set(jnp.where(unresolved, 1.0, answers[..., dec_idx]))

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
        log_probabilities, actions[..., None], axis=-1
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


class Actor(nn.Module):
    hidden_layers: int = 1
    hidden_size: int = 64
    output_size: int = 1
    gain: float = 0.01

    @nn.compact
    def __call__(self, x):  # noqa: ANN001, ANN204
        for _ in range(self.hidden_layers):
            # kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2))
            x = nn.Dense(self.hidden_size)(x)
            x = nn.tanh(x)

        # kernel_init=jax.nn.initializers.orthogonal(self.gain)
        x = nn.Dense(self.output_size)(x)
        return x  # noqa: RET504


class Critic(Actor):
    gain = 1.0


def create_agents() -> tuple[TurbaTrainState, TurbaTrainState, TurbaTrainState]:
    # Decision Making | Policy Network | Actor
    actor = TurbaTrainState.swarm(
        Actor(hidden_layers=1, hidden_size=8, output_size=len(Decision)),
        optimizer=optax.adam(LR),
        swarm_size=NUM_ORGANISMS,
        sample_input=np.zeros((1, 9)),
    )

    # Reward Prediction | Value Network | Critic
    critic = TurbaTrainState.swarm(
        Critic(hidden_layers=1, hidden_size=8),
        optimizer=optax.adam(LR),
        swarm_size=NUM_ORGANISMS,
        sample_input=np.zeros((1, 9)),
    )

    return actor, critic


def create_random_states(swarm_size: int, batch_size: int) -> np.ndarray:
    states = np.random.randint(0, 4, (swarm_size, batch_size, 9))
    states[..., 4] = np.random.choice((2, 3), (swarm_size, batch_size), p=(0.8, 0.2))

    return states


def get_reward(state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
    correct = jit_correct_decision(state)
    reward = jnp.take_along_axis(correct, action, axis=-1)

    return reward


def train(
    actor: TurbaTrainState, critic: TurbaTrainState
) -> tuple[TurbaTrainState, TurbaTrainState]:
    rng = jax.random.PRNGKey(0)
    for episode in range(EPISODES):
        LOGGER.info(f"---- Episode {episode} ----")
        # Create random environment states and get the correct decisions
        start = perf_counter()
        states = create_random_states(NUM_ORGANISMS, BATCH_SIZE)
        state_time = perf_counter() - start

        # Get the action probabilities
        start = perf_counter()
        logits = actor.predict(states)
        log_probabilities = jax.nn.log_softmax(logits)
        probabilities = jnp.exp(log_probabilities)
        entropy = -(log_probabilities * probabilities).sum(axis=-1).mean(axis=-1)

        # Sample actions in acordance with action probabilities
        rng, _rng = jax.random.split(rng)
        actions = jax.random.categorical(_rng, logits, shape=(NUM_ORGANISMS, BATCH_SIZE))[
            ..., None
        ]
        action_time = perf_counter() - start

        # Get the reward
        start = perf_counter()
        rewards = get_reward(states, actions)
        predicted_reward = critic.predict(states)

        # Calculate advantage for each organism
        advantages = rewards - predicted_reward

        # Standardize advantage
        advantage_mean = advantages.mean(axis=1, keepdims=True)
        advantage_std = advantages.std(axis=1, keepdims=True)
        std_advantages = (advantages - advantage_mean) / (advantage_std + 1e-8)
        reward_time = perf_counter() - start

        # Prepare data for loss function
        start = perf_counter()
        action_log_probabilities = jnp.take_along_axis(log_probabilities, actions, axis=-1)
        y = jnp.concatenate((actions, std_advantages, action_log_probabilities), axis=2)
        prep_time = perf_counter() - start

        start = perf_counter()
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

        train_time = perf_counter() - start

        if writer is not None:
            writer.add_scalar("Time/State", state_time, episode)
            writer.add_scalar("Time/Action", action_time, episode)
            writer.add_scalar("Time/Reward", reward_time, episode)
            writer.add_scalar("Time/Prep", prep_time, episode)
            writer.add_scalar("Time/Train", train_time, episode)

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
                writer.add_histogram(
                    "Actor/Probabilities", np.array(probabilities).squeeze(), episode
                )
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

    return actor, critic


def main() -> None:
    actor, critic = create_agents()
    actor, critic = train(actor, critic)

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
