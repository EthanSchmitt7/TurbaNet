from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax.numpy as jnp
import optax

if TYPE_CHECKING:
    from jaxlib.xla_extension import ArrayImpl

__all__ = ["l2_loss", "softmax_cross_entropy"]


def l2_loss(params: dict, batch: dict, apply_fn: Callable) -> tuple[ArrayImpl, ArrayImpl]:
    output = apply_fn({"params": params}, batch["input"])
    loss = optax.l2_loss(predictions=output, targets=batch["output"]).mean()
    return loss, output


def softmax_cross_entropy(params: dict, batch: dict, apply_fn: Callable):  # noqa ANN201
    logits = apply_fn({"params": params}, batch["input"])
    loss = optax.softmax_cross_entropy(logits, batch["output"]).mean()
    return loss, logits
