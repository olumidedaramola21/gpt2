import jax
import numpy as np


def make_rng(seed: int) -> jax.Array:
    """
    Constructs the root PRNGKey array for an excution run

    Args:
      seed: Base integer seed

    Returns:
      A JAX PRNGKey representing the root generator state.
      This root key must be split before being ditributed across modules.

    """
    return jax.random.PRNGKey(seed)


def make_numpy_rng(seed: int) -> np.random.Generator:
    """
    Initializes a standard Numpy Generator for host side operations

    Args:
        seed: Base integer seed for Numpy

    Returns:
        An isolated `np.random.Generator` instance.
    """
    return np.random.default_rng(seed)


def split_keys(rng: jax.Array, n: int) -> jax.Array:
    """
    Splits a single PRNG key into `n` statistically independent subkeys.

    Args:
        rng: The source JAX PRNG key to split
        n: Number of independent subkeys to produce

    Returns:
      An array of `n` independent subkeys of shape `(n, 2)`

    """
    return jax.random.split(rng, num=n)
