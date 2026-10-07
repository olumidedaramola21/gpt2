from __future__ import annotations

from pathlib import Path

import numpy as np

from .tokenizer import build_tokenizer


def prepare_dataset(
    text_path: str | Path,
    out_path: str | Path,
    tokenizer_name: str = "gpt2",
) -> Path:
    """
    Tokenizes raw text into a contiguous uint16 binary file (idempotent)

    Args:
        text_path: Path to the raw input text file.
        out_path: Destination path for the `.bin` output file

    Returns:
        Path object pointing to the generated/exiting binary file
    """
    target_out = Path(out_path)

    if target_out.exists():
        return target_out

    target_out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer(tokenizer_name)

    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    ids = tokenizer.encode(text)
    if not isinstance(ids, np.ndarray) or ids.dtype != np.uint16:
        ids = np.array(ids, dtype=np.uint16)

    ids.tofile(target_out)
    return target_out


def load_token_array(bin_path: str | Path) -> np.memmap:
    """
    Memory-maps a flat binary token file without loading the entire buffer into RAM.

    Args:
        bin_path: Path to the `.bin` tokrnized dataset

    Returns:
        Read-only memory-mapped 1D numpy array of dtype uint16
    """
    target_path = Path(bin_path)
    if not target_path.exists():
        raise FileNotFoundError(f"Binary token not found at: {target_path}")

    return np.memmap(target_path, dtype=np.uint16, mode="r")


def train_val_split(
    data: np.ndarray | np.memmap,
    val_fraction: float = 0.001,
) -> tuple[np.ndarray | np.memmap, np.ndarray | np.memmap]:
    """

    Args:
        data: A flat 1D array or memory-mapped token sequence
        val_fraction (float, optional): _description_. Defaults to 0.001.

    Returns:
        A tuple of `(train_data, val_data)` memory slices.
    """
    n = len(data)
    split_idx = int(n * (1.0 - val_fraction))
    return data[:split_idx], data[split_idx:]
