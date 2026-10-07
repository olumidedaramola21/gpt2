from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def get_batch(
    data: np.ndarray | np.memmap,
    block_size: int,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Samples a random batch of input and target token sequences from the databases.

    Args:
        data: Flat 1D `uint16` Numpy array or memory-mapped token buffer
        block_size: Context window length (sequence length)
        batch_size: Number of independent sequences to sample
        rng: Standard Numpy `np.random.Generator` instance for host-side index generation

    Returns:
        A tuple `(x, y)` where:
          - `x`: `int32` array of shape `[batch_size, block_size]` (input token IDs).
          - `y`: `int32` array of shape `[batch_size, block_size]` (next-token targets).

    """

    max_start = len(data) - block_size - 1
    starts = rng.integers(0, max_start, size=batch_size)

    x_host = np.stack([data[s : s + block_size] for s in starts]).astype(np.int32)

    y_host = np.stack([data[s + 1 : s + 1 + block_size] for s in starts]).astype(
        np.int32
    )

    return jnp.asarray(x_host), jnp.asarray(y_host)
