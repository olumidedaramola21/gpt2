import jax
import jax.numpy as jnp

from .config import GPT2Config


def init_mlp_params(rng: jax.Array, config: GPT2Config) -> dict:
    """
    Initializes the weights and biases for the position-wise feed forward network

    Args:
      rng: JAX PRNGKey
      config: GPT2Config instance containing model hyperparameters.

    Returns:
      Dictionary containing parameter arrays for linear projection
    """

    k1, k2 = jax.random.split(rng)
    std = 0.02

    params = {
        "c_fc_w": std * jax.random.normal(k1, (config.n_embd, 4 * config.n_embd)),
        "c_proj_w": (std / jnp.sqrt(2.0 * config.n_layer))
        * jax.random.normal(k2, (4 * config.n_embd, config.n_embd)),
    }
    if config.bias:
        params["c_fc_b"] = jnp.zeros((4 * config.n_embd,))
        params["c_proj_b"] = jnp.zeros(config.n_embd)

    return params


def mlp_forward(
    params: dict,
    x: jnp.ndarray,
    config: GPT2Config,
    rng: jax.Array | None = None,
    train: bool = False,
) -> jnp.ndarray:
    """
    Executes the forward pass of the MLP sub-layer using tensor contractions.

    Args:
      params: Dictionary containing 'c_fc_w', 'c_proj_w', and optional bias
      x: Input residual stream tensor of shape [B, T, C].
      config: GPT2Config instance
      rng: PRNGKey for dropout regularization (required if train=true and dropout > 0)
      train: Boolean flag indicating training mode

    Returns:
      Output tensor of shape [B, T C]

    """

    h = jnp.einsum("b t c, c e -> b t e", x, params["c_fc_w"])
    if config.bias:
        h = h + params["c_fc_b"]

    h = jax.nn.gelu(h, approximate=True)

    y = jnp.einsum("b t e, e c -> b t c", h, params["c_proj_w"])
    if config.bias:
        y = y + params["c_proj_b"]

    if train and config.dropout > 0.0:
        assert rng is not None, "rng is required for MLP dropout in train mode"
        keep = 1.0 - config.dropout
        mask = jax.random.bernoulli(rng, keep, y.shape)
        y = jnp.where(mask, y / keep, 0.0)

    return y
