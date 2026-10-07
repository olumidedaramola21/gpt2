from dataclasses import dataclass


@dataclass(frozen=True)
class GPT2Config:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True
    layer_norm_eps: float = 1e-5

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head
