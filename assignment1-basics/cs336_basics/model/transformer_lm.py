import torch

from .rmsnorm import RMSNorm
from .embedding import Embedding
from .linear import Linear
from .transformer_block import Transformer_block

class Transformer_LM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.transformer_layers = torch.nn.ModuleList(
            [
                Transformer_block(d_model, num_heads, d_ff, context_length, rope_theta)
                for _ in range(num_layers)
            ]
        )
        self.rmsnorm = RMSNorm(d_model)
        self.output_embedding = Linear(d_model, vocab_size)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        x = self.embedding(in_indices)
        for layer in self.transformer_layers:
            x = layer(x)
        # 训练时直接传递logits给交叉熵函数，在推理时需要概率分布时，才用softmax
        return self.output_embedding(self.rmsnorm(x))

# uv run pytest -k test_transformer_lm