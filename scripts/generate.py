""" 
Autoregressive sampling from a trained checkpoint.

Usage:

    python scripts/generate.py \\ 
        --config configs/debug.yaml \
        --checkpoint checkpoints/debug/ckpt.pkl \\ 
        --prompt "The quick" \\ 
        --max_new_tokens 50  \
        --temperature 0.8 \
        --top_k 40 
      """

from __future__ import annotations

import argparse
import os
import sys

import jax
import jax.numpy as jnp
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


from src.data.tokenizer import build_tokenizer
from src.model.config import GPT2Config
from src.model.transformer import gpt2_forward, init_gpt2_params
from src.utils.checkpointing import load_checkpoint


def sample(
    params,
    config: GPT2Config,
    idx: jnp.ndarray,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    rng: jax.Array,
):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -config.block_size :]

        logits = gpt2_forward(params, idx_cond, config, rng=None, train=False)

        logits = logits[:, -1, :] / max(temperature, 1e-6)

        if top_k is not None:
            top_vals, _ = jax.lax.top_k(logits, min(top_k, logits.shape[-1]))

            threshold = top_vals[:, -1:]

            logits = jnp.where(logits < threshold, -jnp.inf, logits)

        rng, sample_key = jax.random.split(rng)

        # Categorical sampling
        next_id = jax.random.categorical(sample_key, logits, axis=-1)

        idx = jnp.concatenate([idx, next_id[:, None]], axis=1)
    return idx


def main(
    config_path, checkpoint_path, prompt, max_new_tokens, temperature, top_k, seed
):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

        m, d = cfg["model"], cfg["data"]

    config = GPT2Config(
        vocab_size=m["vocab_size"],
        block_size=m["block_size"],
        n_layer=m["n_layer"],
        n_head=m["n_head"],
        n_embd=m["n_embd"],
        dropout=m["dropout"],
        bias=m["bias"],
    )

    params_template = init_gpt2_params(jax.random.PRNGKey(0), config)
    params, extra = load_checkpoint(checkpoint_path, target_params=params_template)

    print(f"loaded checkpoint from step {extra.get('step')}")

    tokenizer = build_tokenizer(d["tokenizer"])

    ids = tokenizer.encode(prompt)

    idx = jnp.asarray(ids, dtype=jnp.int32)[None, :]

    rng = jax.random.PRNGKey(seed)

    out = sample(params, config, idx, max_new_tokens, temperature, top_k, rng)

    text = tokenizer.decode(out[0])
    print("--- generated ---")
    print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    main(
        args.config,
        args.checkpoint,
        args.prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        args.seed,
    )
