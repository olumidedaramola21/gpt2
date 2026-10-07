from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from ..model.config import GPT2Config
from ..model.transformer import gpt2_forward
from .loss import cross_entropy_loss


class TrainState(NamedTuple):
    """
    Immutable state container representing the complete training state.

    Attributes:
        params: Model parameter pytree dictionary
        opt_state: Optax optimizer accumulator states (momenta, step counts, etc.)
        step: Integer scalar tracking completed gradient steps.
    """

    params: dict
    opt_state: optax.OptState
    step: int


def make_train_step(optimizer: optax.GradientTransformation, config: GPT2Config):
    """
    Constructs a JIT-compiled training step funcion closed over the model config and optimizer.

    Args:
        optimizer: Optax transformation chain (e.g., clipping + AdamW)
        config: Static model configuration class.

    Returns:
        A JIT-compiled callable `train_step(state, batch_x, batch_y, rng)`
    """

    def loss_fn(params, batch_x, batch_y, rng):
        logits = gpt2_forward(params, batch_x, config, rng=rng, train=True)
        return cross_entropy_loss(logits, batch_y)

    @jax.jit
    def train_step(
        state: TrainState, batch_x: jnp.ndarray, batch_y: jnp.ndarray, rng: jax.Array
    ):

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params, batch_x, batch_y, rng)

        updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)

        new_state = TrainState(
            params=new_params, opt_state=new_opt_state, step=state.step + 1
        )
        grad_norm = optax.global_norm(grads)
        return new_state, {"loss": loss, "grad_norm": grad_norm}

    return train_step


def make_eval_step(config: GPT2Config):
    @jax.jit
    def eval_step(params, batch_x, batch_y):
        logits = gpt2_forward(params, batch_x, config, rng=None, train=False)
        return cross_entropy_loss(logits, batch_y)

    return eval_step


def init_train_state(
    params: dict, optimizer: optax.GradientTransformation
) -> TrainState:
    return TrainState(params=params, opt_state=optimizer.init(params), step=0)
