import jax
import jax.numpy as jnp
from einops import repeat

from .config import GPT2Config


def init_embeddings(rng: jax.Array, config: GPT2Config) -> dict:

    wte_key, wpe_key = jax.random.split(rng)
    std = 0.02
    wte = std * jax.random.normal(wte_key, (config.vocab_size, config.n_embd))
    wpe = std * jax.random.normal(wpe_key, (config.block_size, config.n_embd))
    return {"wte": wte, "wpe": wpe}


def apply_embeddings(params: dict, idx: jnp.ndarray) -> jnp.ndarray:
    """
    Args:
      params: dict with 'wte' [V, C] and 'wpe' [T_max, C]
      idx: int32 token ids, shape [B, T]

    Returns:
      [B, T, C] float array - the inital residual stream
    """
    B, T = idx.shape
    tok_emb = params["wte"][idx]
    pos_emb = params["wpe"][:T]
    pos_emb_batched = repeat(pos_emb, "t c -> b t c", b=B)

    return tok_emb + pos_emb_batched
