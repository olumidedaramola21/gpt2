"""
Standalone evaluation: load a checkpoint, compute validation loss/perplexity over the full (or a capped number of) validation windows.
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import load_token_array, train_val_split
from src.data.loader import get_batch
from src.model.config import GPT2Config
from src.training.loss import perplexity
from src.training.trainer import make_eval_step
from src.utils.checkpointing import load_checkpoint
from src.utils.seed import make_numpy_rng


def main(config_path: str, checkpoint_path: str, max_batches: int):

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    m, d, t = (cfg["model"], cfg["data"], cfg["train"])

    config = GPT2Config(
        vocab_size=m["vocab_size"],
        block_size=m["block_size"],
        n_layer=m["n_layer"],
        n_head=m["n_head"],
        n_embd=m["n_embd"],
        dropout=m["dropout"],
        bias=m["bias"],
    )

    state, extra = load_checkpoint(checkpoint_path)

    print(f"loaded checkpoint from step {extra.get("step")}")

    all_tokens = load_token_array(d["train_bin"])

    _, val_data = train_val_split(all_tokens, d["val_fraction"])

    eval_step = make_eval_step(config)

    np_rng = make_numpy_rng(t["seed"] + 1)

    losses = []

    for _ in range(max_batches):
        xb, yb = get_batch(val_data, config.block_size, t["batch_size"], np_rng)
        losses.append(float(eval_step(state.params, xb, yb)))

    mean_loss = sum(losses) / len(losses)
    print(
        f"val_loss={mean_loss:.4f} "
        f"val_ppl={float(perplexity(mean_loss)):.2f} "
        f"(over {max_batches} batches of size {t["batch_size"]})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max_batches", type=int, default=100)
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.max_batches)
