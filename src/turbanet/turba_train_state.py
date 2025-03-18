from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState

if TYPE_CHECKING:
    from typing import Callable

    from jaxlib.xla_extension import ArrayImpl


__all__ = ["TurbaTrainState"]


def create_fn(
    model: nn.Module,
    optimizer: optax.GradientTransformation,
    sample_input: ArrayImpl,
    seed: int,
) -> TurbaTrainState:
    """Creates an initial `TurbaTrainState`.

    Args:
        model: The model to train.
        optimizer: The optimizer to use.
        input_size: The size of the input.
        seed: The seed to use for initialization.

    Returns:
        TurbaTrainState: A `TurbaTrainState` object.
    """

    # initialize parameters by passing an input template
    params = model.init(jr.PRNGKey(seed), sample_input)["params"]
    return TurbaTrainState.create(apply_fn=model.apply, params=params, tx=optimizer)


create = jax.vmap(create_fn, in_axes=(None, None, None, 0))


@partial(jax.jit, static_argnames=("loss_fn",))
def train_fn(
    state: TurbaTrainState,
    input: ArrayImpl,
    output: ArrayImpl,
    loss_fn: Callable[[dict, dict, Callable], tuple[ArrayImpl, ArrayImpl]],
) -> tuple[TurbaTrainState, ArrayImpl, ArrayImpl]:
    """Train for a single step.

    Args:
        state: The current `TurbaTrainState` object.
        input: The input to the model.
        output: The output of the model.
        loss_fn: The loss function to use.

    Returns:
        tuple[TurbaTrainState, ArrayImpl, ArrayImpl]: The updated
            (state, loss, prediction).
    """

    def wrapped_loss_fn(
        params: dict, input: ArrayImpl, output: ArrayImpl
    ) -> tuple[ArrayImpl, ArrayImpl]:
        return loss_fn(params, input, output, state.apply_fn)

    grad_fn = jax.value_and_grad(wrapped_loss_fn, has_aux=True)
    (loss, prediction), grads = grad_fn(state.params, input, output)
    state = state.apply_gradients(grads=grads)
    return state, loss, prediction


train = jax.vmap(train_fn, in_axes=(0, 0, 0, None))


@jax.jit
def predict_fn(state: TurbaTrainState, input: ArrayImpl) -> ArrayImpl:
    """Predict on a batch of data.

    Args:
        state: The current `TurbaTrainState` object.
        input: The input to the model.

    Returns:
        ArrayImpl: The prediction of the model.
    """
    return state.apply_fn({"params": state.params}, input)


predict = jax.vmap(predict_fn, in_axes=(0, 0))


@partial(jax.jit, static_argnames=("loss_fn",))
def evaluate_fn(
    state: TurbaTrainState,
    input: ArrayImpl,
    output: ArrayImpl,
    loss_fn: Callable[[dict, dict, Callable], tuple[ArrayImpl, ArrayImpl]],
) -> ArrayImpl:
    """Evaluate a loss function.

    Args:
        state: The current `TurbaTrainState` object.
        input: The input to the model.
        output: The expected output of the model.
        loss_fn: The loss function to use.

    Returns:
        ArrayImpl: The loss of the model."""
    return loss_fn(state.params, input, output, state.apply_fn)


evaluate = jax.vmap(evaluate_fn, in_axes=(0, 0, 0, None))


class TurbaTrainState(TrainState):
    """TrainState for TurbaNet."""

    @staticmethod
    def swarm(
        model: nn.Module,
        optimizer: optax.GradientTransformation,
        swarm_size: int,
        sample_input: ArrayImpl,
        seed: ArrayImpl = None,
    ) -> TurbaTrainState:
        """Creates a swarm of initial `TurbaTrainState`s.

        Args:
            model: The model to train.
            optimizer: The optimizer to use.
            swarm_size: The size of the swarm.
            input_size: The size of the input.
            seed: The seed to use for initialization.

        Returns:
            TurbaTrainState: A `TurbaTrainState` object.
        """
        if seed is None:
            seed = 0

        if isinstance(seed, int):
            seed = jnp.linspace(seed, swarm_size - 1, swarm_size).astype(int)

        if len(seed) != swarm_size:
            raise ValueError("Seed and learning rate must be the same length as swarm_size.")

        return create(model, optimizer, sample_input, seed)

    def predict(self, input_data: np.ndarray) -> ArrayImpl:
        """Predicts on a batch of data.

        Args:
            input_data: A batch of data to predict on of shape
                (swarm_size, batch_size, input_size)

        Returns:
            ArrayImpl: A batch of predictions
        """
        if len(self) == 1 and input_data.shape[0] != 1:
            input_data = input_data.reshape(1, *input_data.shape)

        if input_data.shape[0] != len(self):
            raise ValueError(
                f"Batch input shape {input_data.shape} does not match "
                f"TrainState length {len(self)}."
            )

        if isinstance(input_data, jnp.ndarray):
            input_data = jnp.asarray(input_data)

        return predict(self, input_data)

    def evaluate(
        self, input_data: np.ndarray, output_data: np.ndarray, loss_fn: Callable
    ) -> tuple[ArrayImpl, ArrayImpl]:
        """Evaluates a loss function on a batch of data.

        Args:
            input_data: A batch of data to evaluate on of shape
                (swarm_size, batch_size, input_size)
            output_data: A batch of expected outputs of shape
                (swarm_size, batch_size, output_size)
            loss_fn: The loss function to use.

        Returns:
            tuple[ArrayImpl, ArrayImpl]: A batch of (loss, prediction)
        """
        if len(self) == 1 and input_data.shape[0] != 1:
            input_data = input_data.reshape(1, *input_data.shape)

        if input_data.shape[0] != len(self):
            raise ValueError(
                f"Batch input shape {input_data.shape} does not match "
                f"TrainState length {len(self)}."
            )

        if isinstance(input_data, jnp.ndarray):
            input_data = jnp.asarray(input_data)

        if len(self) == 1 and output_data.shape[0] != 1:
            output_data = output_data.reshape(1, *output_data.shape)

        if output_data.shape[0] != len(self):
            raise ValueError(
                f"Batch output shape {output_data.shape} does not match "
                f"TrainState length {len(self)}."
            )

        if isinstance(output_data, jnp.ndarray):
            output_data = jnp.asarray(output_data)

        return evaluate(self, input_data, output_data, loss_fn)

    def train(
        self, input_data: np.ndarray, output_data: np.ndarray, loss_fn: Callable
    ) -> tuple[TurbaTrainState, ArrayImpl, ArrayImpl]:
        """Trains on a batch of data.

        Args:
            input_data: A batch of data to train on of shape
                (swarm_size, batch_size, input_size)
            output_data: A batch of data to train on of shape
                (swarm_size, batch_size, output_size)

        Returns:
            tuple[TurbaTrainState, ArrayImpl, ArrayImpl]: The updated
                (TrainState, loss, prediction)
        """
        if len(self) == 1 and input_data.shape[0] != 1:
            input_data = input_data.reshape(1, *input_data.shape)

        if input_data.shape[0] != len(self):
            raise ValueError(
                f"Batch input shape {input_data.shape} does not match "
                f"TrainState length {len(self)}."
            )

        if not isinstance(input_data, jnp.ndarray):
            input_data = jnp.asarray(input_data)

        if len(self) == 1 and output_data.shape[0] != 1:
            output_data = output_data.reshape(1, *output_data.shape)

        if output_data.shape[0] != len(self):
            raise ValueError(
                f"Batch output shape {output_data.shape} does not match "
                f"TrainState length {len(self)}."
            )

        if not isinstance(output_data, jnp.ndarray):
            output_data = jnp.asarray(output_data)

        return train(self, input_data, output_data, loss_fn)

    @property
    def shape(self) -> tuple[int, ...]:
        param_keys = list(self.params.keys())
        first_kernel = self.params[param_keys[0]]["kernel"].shape
        last_kernel = self.params[param_keys[-1]]["kernel"].shape
        return tuple(list(first_kernel[0:2]) + [last_kernel[-1]])

    def __len__(self) -> int:
        return len(self.opt_state[0].count)

    @jax.jit
    def __add__(self: TurbaTrainState, other: TurbaTrainState) -> TurbaTrainState:
        if len(self) != len(other):
            raise ValueError(
                f"Cannot add TrainStates with incompatible lengths {len(self)}, {len(other)}."
            )
        mu = {}
        nu = {}
        params = {}
        for key in self.params.keys():
            params[key] = {}
            p1 = self.params[key]["bias"]
            p2 = other.params[key]["bias"]
            if p1.shape != p2.shape:
                raise ValueError(
                    f"Cannot append TrainStates with incompatible "
                    f"bias dimensions {p1.shape}, {p2.shape}."
                )
            params[key]["bias"] = p1 + p2

            p1 = self.params[key]["kernel"]
            p2 = other.params[key]["kernel"]
            if p1.shape != p2.shape:
                raise ValueError(
                    f"Cannot append TrainStates with incompatible "
                    f"kernel dimensions {p1.shape}, {p2.shape}."
                )
            params[key]["kernel"] = p1 + p2

            mu[key] = {}
            p1 = self.opt_state[0].mu[key]["bias"]
            p2 = other.opt_state[0].mu[key]["bias"]
            mu[key]["bias"] = p1 + p2

            p1 = self.opt_state[0].mu[key]["kernel"]
            p2 = other.opt_state[0].mu[key]["kernel"]
            mu[key]["kernel"] = p1 + p2

            nu[key] = {}
            p1 = self.opt_state[0].nu[key]["bias"]
            p2 = other.opt_state[0].nu[key]["bias"]
            nu[key]["bias"] = p1 + p2

            p1 = self.opt_state[0].nu[key]["kernel"]
            p2 = other.opt_state[0].nu[key]["kernel"]
            nu[key]["kernel"] = p1 + p2

        return TurbaTrainState(
            step=self.step,
            apply_fn=self.apply_fn,
            params=params,
            tx=self.tx,
            opt_state=(
                optax.ScaleByAdamState(count=self.opt_state[0].count, mu=mu, nu=nu),
                optax.EmptyState(),
            ),
        )

    @jax.jit
    def append(self, other: TurbaTrainState) -> TurbaTrainState:
        mu = {}
        nu = {}
        params = {}
        for key in self.params.keys():
            params[key] = {}
            p1 = self.params[key]["bias"]
            p2 = other.params[key]["bias"]
            if p1.shape != p2.shape:
                raise ValueError(
                    f"Cannot append TrainStates with incompatible "
                    f"bias dimensions {p1.shape}, {p2.shape}."
                )
            params[key]["bias"] = jnp.append(p1, p2, axis=0)

            p1 = self.params[key]["kernel"]
            p2 = other.params[key]["kernel"]
            if p1.shape != p2.shape:
                raise ValueError(
                    f"Cannot append TrainStates with incompatible "
                    f"kernel dimensions {p1.shape}, {p2.shape}."
                )
            params[key]["kernel"] = jnp.append(p1, p2, axis=0)

            mu[key] = {}
            p1 = self.opt_state[0].mu[key]["bias"]
            p2 = other.opt_state[0].mu[key]["bias"]
            mu[key]["bias"] = jnp.append(p1, p2, axis=0)

            p1 = self.opt_state[0].mu[key]["kernel"]
            p2 = other.opt_state[0].mu[key]["kernel"]
            mu[key]["kernel"] = jnp.append(p1, p2, axis=0)

            nu[key] = {}
            p1 = self.opt_state[0].nu[key]["bias"]
            p2 = other.opt_state[0].nu[key]["bias"]
            nu[key]["bias"] = jnp.append(p1, p2, axis=0)

            p1 = self.opt_state[0].nu[key]["kernel"]
            p2 = other.opt_state[0].nu[key]["kernel"]
            nu[key]["kernel"] = jnp.append(p1, p2, axis=0)

        step = jnp.append(self.step, other.step)
        count = jnp.append(self.opt_state[0].count, other.opt_state[0].count)

        return TurbaTrainState(
            step=step,
            apply_fn=self.apply_fn,
            params=params,
            tx=self.tx,
            opt_state=(
                optax.ScaleByAdamState(count=count, mu=mu, nu=nu),
                optax.EmptyState(),
            ),
        )

    @jax.jit
    def merge(self) -> TurbaTrainState:
        mu = {}
        nu = {}
        params = {}
        for key in self.params.keys():
            params[key] = {}
            params[key]["bias"] = jnp.mean(self.params[key]["bias"], axis=0).reshape(1, -1)
            params[key]["kernel"] = jnp.expand_dims(
                jnp.mean(self.params[key]["kernel"], axis=0), 0
            )

            mu[key] = {}
            mu[key]["bias"] = jnp.mean(self.opt_state[0].mu[key]["bias"], axis=0).reshape(1, -1)
            mu[key]["kernel"] = jnp.expand_dims(
                jnp.mean(self.opt_state[0].mu[key]["kernel"], axis=0), 0
            )

            nu[key] = {}
            nu[key]["bias"] = jnp.mean(self.opt_state[0].nu[key]["bias"], axis=0).reshape(1, -1)
            nu[key]["kernel"] = jnp.expand_dims(
                jnp.mean(self.opt_state[0].nu[key]["kernel"], axis=0), 0
            )

        step = jnp.array([self.step.max()])
        count = jnp.array([self.opt_state[0].count.max()], dtype="int32")

        return TurbaTrainState(
            step=step,
            apply_fn=self.apply_fn,
            params=params,
            tx=self.tx,
            opt_state=(
                optax.ScaleByAdamState(count=count, mu=mu, nu=nu),
                optax.EmptyState(),
            ),
        )
