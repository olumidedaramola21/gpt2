import jax
import jax.numpy as jnp
from einops import rearrange
from .config import GPT2Config


def init_attention_params(rng: jax.Array, config: GPT2Config) -> dict:
    k1, k2 = jax.random.split(rng)
    std = 0.02
    params = {
        "c_attn_w": std * jax.random.normal(k1, (config.n_embed, 3 * config.n_embed)),
        "c_proj_w": (std / jnp.sqrt(2.0 * config.n_layer))
        * jax.random.normal(k2, (config.n_embed, config.n_embed)),
    }

    if config.bias:
        params["c_attn_b"] = jnp.zeros((3 * config.n_embed))
        params["c_proj_b"] = jnp.zeros((config.n_embed,))
