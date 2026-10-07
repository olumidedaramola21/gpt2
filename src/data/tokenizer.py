from __future__ import annotations

import numpy as np
import tiktoken


class GPT2Tokenizer:
    """
    Exact GPT-2 BPE toknenizer via tiktoken. vocab_size == 50257
    """

    def __init__(self):

        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.enc.n_vocab
        self.eot_token = self.enc.eot_token

    def encode(self, text: str) -> np.ndarray:
        ids = self.enc.encode(text, allowed_special={"<|endoftext|>"})
        return np.array(ids, dtype=np.uint16)

    def decode(self, ids: np.ndarray) -> str:
        return self.enc.decode(list(np.asarray(ids).tolist()))


class ByteTokenizer:
    """
    Trivial fallback: every byte 0-255 is its own token. No BPE merges, no external dependency, deterministic and instant.
    """

    vocab_size = 256

    def encode(self, text: str) -> np.ndarray:
        return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.uint16)

    def decode(self, ids: np.ndarray) -> str:
        return bytes(np.asarray(ids).astype(np.uint8).tolist()).decode(
            "utf-8", errors="replace"
        )


def build_tokenizer(name: str):
    if name == "gpt2":
        return GPT2Tokenizer()
    elif name == "byte":
        return ByteTokenizer()
