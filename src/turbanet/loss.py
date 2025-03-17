from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import optax

if TYPE_CHECKING:
    from jaxlib.xla_extension import ArrayImpl

__all__ = ["l2_loss", "mse", "softmax_cross_entropy"]


def l2_loss(
    params: dict, input: ArrayImpl, output: ArrayImpl, apply_fn: Callable
) -> tuple[ArrayImpl, ArrayImpl]:
    prediction = apply_fn({"params": params}, input)
    loss = optax.l2_loss(predictions=prediction, targets=output).mean()
    return loss, output


def mse(
    params: dict, input: ArrayImpl, output: ArrayImpl, apply_fn: Callable
) -> tuple[ArrayImpl, ArrayImpl]:
    prediction = apply_fn({"params": params}, input)
    loss = ((prediction - output) ** 2).mean()
    return loss, output


def softmax_cross_entropy(params: dict, input: ArrayImpl, output: ArrayImpl, apply_fn: Callable):  # noqa ANN201
    logits = apply_fn({"params": params}, input)
    loss = optax.softmax_cross_entropy(logits, output).mean()
    return loss, logits
