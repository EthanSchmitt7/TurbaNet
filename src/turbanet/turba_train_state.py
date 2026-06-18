from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import flax
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState

if TYPE_CHECKING:
    from typing import Callable


__all__ = ["TurbaTrainState"]


def make_params_fn(model: nn.Module, sample_input: jax.Array, seed: int) -> dict:
    params = model.init(jr.PRNGKey(seed), sample_input)["params"]
    return params


make_params = jax.vmap(make_params_fn, in_axes=(None, None, 0))


def create_fn(
    apply_fn: Callable, optimizer: optax.GradientTransformation, params: dict
) -> TurbaTrainState:
    """Creates an initial `TurbaTrainState`.

    Args:
        apply_fn: The apply function of the model.
        optimizer: The optimizer to use.
        params: The parameters of the model.

    Returns:
        TurbaTrainState: A `TurbaTrainState` object.
    """

    # initialize parameters by passing an input template
    return TurbaTrainState.create(apply_fn=apply_fn, params=params, tx=optimizer)


_create = jax.vmap(create_fn, in_axes=(None, None, 0))


def make_train_step(loss_fn: Callable) -> Callable:
    def train_fn(
        state: TurbaTrainState, input: jax.Array, output: jax.Array
    ) -> tuple[TurbaTrainState, jax.Array, jax.Array]:
        """Train for a single step.

        Args:
            state: The current `TurbaTrainState` object.
            input: The input to the model.
            output: The output of the model.
            loss_fn: The loss function to use.

        Returns:
            tuple[TurbaTrainState, jax.Array, jax.Array]: The updated
                (state, loss, prediction).
        """

        def wrapped_loss_fn(params: dict) -> tuple[jax.Array, jax.Array]:
            return loss_fn(params, input, output, state.apply_fn)

        grad_fn = jax.value_and_grad(wrapped_loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss, aux

    return jax.jit(jax.vmap(train_fn))


def predict_fn(
    state: TurbaTrainState, input: jax.Array, capture_intermediates: bool = False
) -> jax.Array:
    """Predict on a batch of data.

    Args:
        state: The current `TurbaTrainState` object.
        input: The input to the model.

    Returns:
        jax.Array: The prediction of the model.
    """
    return state.apply_fn(
        {"params": state.params}, input, capture_intermediates=capture_intermediates
    )


predict = jax.jit(jax.vmap(predict_fn, in_axes=(0, 0, None)), static_argnums=(2,))


def evaluate_fn(
    state: TurbaTrainState, input: jax.Array, output: jax.Array, loss_fn: Callable
) -> jax.Array:
    """Evaluate a loss function.

    Args:
        state: The current `TurbaTrainState` object.
        input: The input to the model.
        output: The expected output of the model.
        loss_fn: The loss function to use.

    Returns:
        jax.Array: The loss of the model."""
    return loss_fn(state.params, input, output, state.apply_fn)


evaluate = jax.jit(jax.vmap(evaluate_fn, in_axes=(0, 0, 0, None)), static_argnames=("loss_fn",))


class TurbaTrainState(TrainState):
    """TrainState for TurbaNet."""

    train_function: Callable = flax.struct.field(pytree_node=False, default=None)

    @classmethod
    def vmap_create(
        cls, *, apply_fn: Callable, params: dict, tx: optax.GradientTransformation, **kwargs
    ) -> TurbaTrainState:
        return _create(apply_fn, tx, params)

    @staticmethod
    def swarm(
        model: nn.Module,
        optimizer: optax.GradientTransformation,
        swarm_size: int,
        input_size: int = None,
        sample_input: jax.Array = None,
        seed: jax.Array = None,
    ) -> TurbaTrainState:
        """Creates a swarm of initial `TurbaTrainState`s.

        Args:
            model: The model to train.
            optimizer: The optimizer to use.
            swarm_size: The size of the swarm.
            input_size: The size of the input.
            sample_input: A example of an input to the network
            seed: The seed to use for initialization.

        Returns:
            TurbaTrainState: A `TurbaTrainState` object.
        """
        if input_size is None and sample_input is None:
            raise RuntimeError(
                "'input_size' or a 'sample_input' must be provided to TurbaTrainState.swarm."
            )

        if input_size is not None:
            sample_input = jnp.zeros((1, input_size))

        if seed is None:
            seed = 0

        if isinstance(seed, int):
            seed = jnp.linspace(seed, swarm_size - 1, swarm_size).astype(int)

        if len(seed) != swarm_size:
            raise ValueError("Seed and learning rate must be the same length as swarm_size.")

        params = make_params(model, sample_input, seed)
        return _create(model.apply, optimizer, params)

    def set_loss_fn(self, loss_fn: Callable) -> TurbaTrainState:
        """Sets the loss function to use.

        Args:
            loss_fn: The loss function to use.

        Returns:
            TurbaTrainState: The updated `TurbaTrainState` object.
        """
        # Create train step
        return self.replace(train_function=make_train_step(loss_fn))

    def predict(self, input_data: np.ndarray, capture_intermediates: bool = False) -> jax.Array:
        """Predicts on a batch of data.

        Args:
            input_data: A batch of data to predict on of shape
                (swarm_size, batch_size, input_size)

        Returns:
            jax.Array: A batch of predictions
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

        return predict(self, input_data, capture_intermediates)

    def evaluate(
        self, input_data: np.ndarray, output_data: np.ndarray, loss_fn: Callable
    ) -> tuple[jax.Array, jax.Array]:
        """Evaluates a loss function on a batch of data.

        Args:
            input_data: A batch of data to evaluate on of shape
                (swarm_size, batch_size, input_size)
            output_data: A batch of expected outputs of shape
                (swarm_size, batch_size, output_size)
            loss_fn: The loss function to use.

        Returns:
            tuple[jax.Array, jax.Array]: A batch of (loss, prediction)
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
        self, input_data: np.ndarray, output_data: np.ndarray, **kwargs: dict
    ) -> tuple[TurbaTrainState, jax.Array, jax.Array]:
        """Trains on a batch of data.

        Args:
            input_data: A batch of data to train on of shape
                (swarm_size, batch_size, input_size)
            output_data: A batch of data to train on of shape
                (swarm_size, batch_size, output_size)

        Returns:
            tuple[TurbaTrainState, jax.Array, jax.Array]: The updated
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

        return self.train_function(self, input_data, output_data)

    def cost_analysis(self, input: jax.Array) -> dict:
        return (
            jax.jit(self.apply_fn)
            .lower({"params": self.get_state(0).params}, input)
            .cost_analysis()
        )

    @property
    def shape(self) -> tuple[int, ...]:
        param_keys = list(self.params.keys())
        first_kernel = self.params[param_keys[0]]["kernel"].shape
        last_kernel = self.params[param_keys[-1]]["kernel"].shape
        return tuple(list(first_kernel[0:2]) + [last_kernel[-1]])

    def __len__(self) -> int:
        return list(self.params.values())[0]["kernel"].shape[0]

    @jax.jit
    def append(self, state2: TurbaTrainState) -> TurbaTrainState:
        leaves_a, treedef = jax.tree_util.tree_flatten(self)
        leaves_b, _ = jax.tree_util.tree_flatten(state2)

        merged_leaves = [jnp.concatenate([a, b], axis=0) for a, b in zip(leaves_a, leaves_b)]

        return treedef.unflatten(merged_leaves)

    def merge(self) -> TurbaTrainState:
        return jax.tree_util.tree_map(lambda x: jnp.expand_dims(jnp.mean(x, axis=0), 0), self)

    def get_state(self, index: int) -> TurbaTrainState:
        return jax.tree_util.tree_map(lambda x: x[index], self)

    def remove_state(self, index: int) -> TurbaTrainState:
        return jax.tree_util.tree_map(lambda x: jnp.delete(x, index, axis=0), self)
