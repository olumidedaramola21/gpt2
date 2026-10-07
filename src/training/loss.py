from __future__ import annotations

import jax.numpy as jnp
import optax


def cross_entropy_loss(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the mean scalar cross entropy loss over a batch of token predictions

    Args:
        logits: Unnormalized prediction scores of shape [B, T, V]
        targets: Ground truth target token IDs of shape [B, T] with dtype int32.

    Returns:
        Scalar JAX float array representing the mean cross-entropy loss
    """
    losses = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=targets
    )
    return jnp.mean(losses)


def perplexity(loss: jnp.ndarray) -> jnp.ndarray:
    """
    Converts average cross-entropy loss into perplexity

    Args:
        loss: Scalar mean cross-entropy loss.

    Returns:
        Scalar perplexity value exp (loss)
    """
    return jnp.exp(loss)
