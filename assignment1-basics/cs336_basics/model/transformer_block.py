import torch

from .rmsnorm import RMSNorm
from .swiglu import SwiGLU
from .multihead_self_attention import multihead_self_attention


class Transformer_block(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.rmsnorm1 = RMSNorm(d_model)
        self.rmsnorm2 = RMSNorm(d_model)
        self.attention = multihead_self_attention(d_model, num_heads, max_seq_len, theta)
        self.SwiGLU = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = x + self.attention(self.rmsnorm1(x))
        return x_out + self.SwiGLU(self.rmsnorm2(x_out))


# uv run pytest -k test_transformer_block