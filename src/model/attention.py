import jax
import jax.numpy as jnp
from einops import rearrange

from .config import GPT2Config


def init_attention_params(rng: jax.Array, config: GPT2Config) -> dict:
    k1, k2 = jax.random.split(rng)
    std = 0.02
    params = {
        "c_attn_w": std * jax.random.normal(k1, (config.n_embd, 3 * config.n_embd)),
        "c_proj_w": (std / jnp.sqrt(2.0 * config.n_layer))
        * jax.random.normal(k2, (config.n_embd, config.n_embd)),
    }

    if config.bias:
        params["c_attn_b"] = jnp.zeros((3 * config.n_embd,))
        params["c_proj_b"] = jnp.zeros((config.n_embd,))

    return params


def causal_self_attention(
    params: dict,
    x: jnp.ndarray,
    config: GPT2Config,
    rng: jax.Array | None = None,
    train: bool = False,
) -> jnp.ndarray:
    """
    Args:
      x: [B, T, C] residual stream
      rng: PRNGKey for attention dropout
    Returns:
      [B, T, C]

    """
    _B, T, _C = x.shape
    H = config.n_head

    qkv = jnp.einsum("b t c, c d -> b t d", x, params["c_attn_w"])

    if config.bias:
        qkv = qkv + params["c_attn_b"]

    q, k, v = (
        rearrange(t, "b seq (h d) -> b h seq d", h=H)
        for t in jnp.split(qkv, 3, axis=-1)
    )

    scale = 1.0 / jnp.sqrt(jnp.array(config.head_dim, dtype=x.dtype))
    att = jnp.einsum("b h q d, b h k d -> b h q k", q, k) * scale

    causal_mask = jnp.tril(jnp.ones((T, T), dtype=bool))
    att = jnp.where(causal_mask[None, None, :, :], att, jnp.finfo(att.dtype).min)

    att = jax.nn.softmax(att, axis=-1)

    if train and config.dropout > 0.0:
        assert rng is not None, "rng required for attention dropout in train mode"
        keep = 1.0 - config.dropout
        mask = jax.random.bernoulli(rng, keep, att.shape)
        att = jnp.where(mask, att / keep, 0.0)

    out = jnp.einsum(
        "b h q k, b h k d -> b h q d",
        att,
        v,
    )

    out = rearrange(out, "b h seq d -> b seq (h d)")

    out = jnp.einsum("b t c, c d -> b t d", out, params["c_proj_w"])
    if config.bias:
        out = out + params["c_proj_b"]

    return out
