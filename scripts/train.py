"""
Usage:
  python scripts/train.py --config configs/debug.yaml


"""

from __future__ import annotations

import argparse
import os
import sys
import time

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import load_token_array, prepare_dataset, train_val_split
from src.data.loader import get_batch
from src.model.config import GPT2Config
from src.model.transformer import count_params, init_gpt2_params
from src.training.loss import perplexity
from src.training.optimizer import build_optimizer
from src.training.trainer import init_train_state, make_eval_step, make_train_step
from src.utils.checkpointing import create_checkpoint_manager, save_checkpoint
from src.utils.seed import make_numpy_rng, make_rng, split_keys


def _ensure_debug_corpus(raw_path: str):
    """
    Create a tiny synthetic text file so debug.yaml runs with no external data.
    """
    if os.path.exists(raw_path):
        return
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)

    sample = (
        "The quick brown fox jumps over the lazy dog. " * 500
        + "Jax makes it easy to write pure functions and comos jit, grad, and vmap"
    )
    with open(raw_path, "w") as f:
        f.write(sample)


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    m, d, o, t = (cfg["model"], cfg["data"], cfg["optim"], cfg["train"])

    if d["tokenizer"] == "byte" and not os.path.exists(d["train_bin"]):
        raw_path = "data/debug_raw.txt"
        _ensure_debug_corpus(raw_path)

        bin_path = prepare_dataset(raw_path, d["train_bin"], tokenizer_name="byte")
    else:
        bin_path = d["train_bin"]

    all_tokens = load_token_array(bin_path)

    train_data, val_data = train_val_split(all_tokens, d["val_fraction"])

    print(f"tokens: train={len(train_data):,}" f"val={len(val_data):,}")

    config = GPT2Config(
        vocab_size=m["vocab_size"],
        block_size=m["block_size"],
        n_layer=m["n_layer"],
        n_head=m["n_head"],
        n_embd=m["n_embd"],
        dropout=m["dropout"],
        bias=m["bias"],
    )

    rng = make_rng(t["seed"])

    init_key, rng = split_keys(rng, 2)

    params = init_gpt2_params(init_key, config)

    print(f"model params: {count_params(params):,}")

    optimizer, _schedule = build_optimizer(
        params,
        peak_lr=o["peak_lr"],
        min_lr=o["min_lr"],
        total_steps=t["total_steps"],
        warmup_steps=o["warmup_steps"],
        weight_decay=o["weight_decay"],
        grad_clip_norm=o["grad_clip_norm"],
        b1=o["b1"],
        b2=o["b2"],
    )

    state = init_train_state(params, optimizer)

    train_step = make_train_step(optimizer, config)

    eval_step = make_eval_step(config)

    np_rng = make_numpy_rng(t["seed"])

    def evaluate():
        losses = []

        for _ in range(t["eval_iters"]):
            xb, yb = get_batch(val_data, config.block_size, t["batch_size"], np_rng)

            losses.append(float(eval_step(state.params, xb, yb)))
        return sum(losses) / len(losses)

    t0 = time.time()
    for step in range(1, t["total_steps"] + 1):

        xb, yb = get_batch(train_data, config.block_size, t["batch_size"], np_rng)

        rng, step_key = split_keys(rng, 2)

        state, metrics = train_step(state, xb, yb, step_key)

        if step % t["log_interval"] == 0:
            dt = time.time() - t0
            print(
                f"steps {step:6d} |"
                f"loss {float(metrics["loss"]):.4f}"
                f"| grad_norm {float(metrics["grad_norm"]):.3f}"
                f"| {dt:.1f}s elapsed"
            )

        if step % t["eval_interval"] == 0 or step == t["total_steps"]:
            val_loss = evaluate()
            print(
                f"  [eval] step {step} | "
                f"val_loss {val_loss:.4f} | "
                f"val_ppl {float(perplexity(val_loss)):.2f}"
            )

            manager = create_checkpoint_manager(t["checkpoint_path"])

            save_checkpoint(
                manager,
                state,
                metrics={"config": m, "step": step, "val_loss": val_loss},
            )
            print(f"saved checkpoint -> {t["checkpoint_path"]}")

    print("training complete.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
