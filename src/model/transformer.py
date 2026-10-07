import jax
import jax.numpy as jnp

from .attention import causal_self_attention, init_attention_params
from .config import GPT2Config
from .embeddings import apply_embeddings, init_embeddings
from .mlp import init_mlp_params, mlp_forward


def init_layernorm_params(config: GPT2Config) -> dict:
    """Initializes scale (gamma) and shift (beta) parameter for layerNorm."""
    p = {"gamma": jnp.ones((config.n_embd,))}
    if config.bias:
        p["beta"] = jnp.zeros((config.n_embd,))

    return p


def layer_norm(params: dict, x: jnp.ndarray, config: GPT2Config) -> jnp.ndarray:
    """
    Applies standard Layer Normalzation over the trailling embedding dimensions

    Args:
        params: dictionary containing 'gamma' and optional 'beta' vectors
        x: Input tensor shape of [B, T, C]
        config (GPT2Config): GPT2Config instance

    Returns:
        jnp.ndarray
    """
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)

    x_norm = (x - mean) / jnp.sqrt(var + config.layer_norm_eps)
    out = x_norm * params["gamma"]  # scale
    if config.bias:
        out = out + params["beta"]  # shift

    return out


def init_block_params(rng: jax.Array, config: GPT2Config) -> dict:
    """Initializes parameters for a single Pre-Ln Transformer block"""
    k_attn, k_mlp = jax.random.split(rng)
    return {
        "ln_1": init_layernorm_params(config),
        "attn": init_attention_params(k_attn, config),
        "ln_2": init_layernorm_params(config),
        "mlp": init_mlp_params(k_mlp, config),
    }


def block_forward(
    params: dict,
    x: jnp.ndarray,
    config: GPT2Config,
    rng: jax.Array | None = None,
    train: bool = False,
) -> jnp.ndarray:
    """
    Executes a single Pre-LN Transformer block forward pass:
      1. x = x + Attention(LN_1(x))
      2. x = x + MLP(LN_2(x))

    Args:
      params: Dictionary containing sublayer parameter dicts
      x: Input residual stream tensor [B, T, C]
      config: GPT2Config instance
      rng: PRNGKey for attention and MLP dropout.
      train: Trainig mode flag
    """
    rng_attn, rng_mlp = jax.random.split(rng) if rng is not None else (None, None)

    normed_x1 = layer_norm(params["ln_1"], x, config)
    attn_out = causal_self_attention(
        params["attn"], normed_x1, config, rng=rng_attn, train=train
    )
    x = x + attn_out

    normed_x2 = layer_norm(params["ln_2"], x, config)
    mlp_out = mlp_forward(params["mlp"], normed_x2, config, rng=rng_mlp, train=train)
    x = x + mlp_out

    return x


def init_gpt2_params(rng: jax.Array, config: GPT2Config) -> dict:
    """
    Initializes all parameters for the complete GPT-2 model.

    Args:
      rng: JAX PRNGKey
      config: GPT2Config instance

    Returns:
      Nested dict hierarchy matching the full model topology.

    """
    keys = jax.random.split(rng, config.n_layer + 1)
    emb_key, block_keys = keys[0], keys[1:]

    return {
        "embeddings": init_embeddings(emb_key, config),
        "blocks": [
            init_block_params(block_keys[i], config) for i in range(config.n_layer)
        ],
        "ln_f": init_layernorm_params(config),
    }


def gpt2_forward(
    params: dict,
    idx: jnp.ndarray,
    config: GPT2Config,
    rng: jax.Array | None = None,
    train: bool = False,
) -> jnp.ndarray:
    """
    Full GPT-2 forward pass from discrete token IDs to unnormalized vocabulary

    Args:
        params: Model parameter pytree dictionary
        idx: Integer token ID array of shape [B, T], with T <= config.block_size
        config: GPT2Config instance
        rng: PRNGKey for dropout operations across all layers
        train: Boolean flag indicating training mode

    Returns:
        A logits tensor of shape [B, T, V]
    """
    _B, T = idx.shape
    assert (
        T <= config.block_size
    ), f"Sequence length {T} exceeds block_size {config.block_size}"

    x = apply_embeddings(params["embeddings"], idx)

    if train and rng is not None:
        block_rngs = list(jax.random.split(rng, config.n_layer))
    else:
        block_rngs = [None] * config.n_layer

    for i, block_params in enumerate(params["blocks"]):
        x = block_forward(
            block_params,
            x,
            config,
            rng=block_rngs[i],
            train=train,
        )

    x = layer_norm(params["ln_f"], x, config)

    # weight tying
    logits = jnp.einsum("b t c, v c -> b t v", x, params["embeddings"]["wte"])

    return logits


def count_params(params: dict) -> int:
    """
    Computes the total number of scalar parameters in the model pytree

    """
    leaves = jax.tree_util.tree_leaves(params)
    return sum(leaf.size for leaf in leaves)
