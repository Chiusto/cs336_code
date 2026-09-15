import torch

from .linear import Linear
from .rope import RotaryPositionalEmbedding
from .scaled_dot_product_attention import scaled_dot_product_attention


class multihead_self_attention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float | None = None):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)
        # True 表示该位置可以被关注。
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(max_seq_len, max_seq_len, dtype=torch.bool),
            ),
        )
        # 使用 rope.py 中已经实现的 RoPE。
        self.rope = None
        if theta is not None:
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.d_k,
                max_seq_len=max_seq_len,
            )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        # 将张量从「…，seq_len，d_model」重塑为「…，num_heads，seq_len，d_k」。
        Q = Q.reshape(*Q.shape[:-1], self.num_heads, self.d_k).transpose(-3, -2)
        K = K.reshape(*K.shape[:-1], self.num_heads, self.d_k).transpose(-3, -2)
        V = V.reshape(*V.shape[:-1], self.num_heads, self.d_v).transpose(-3, -2)
        seq_len = x.shape[-2]
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            elif token_positions.ndim == Q.ndim - 2:
                # 在 head 维度插入一个轴，让所有 head 使用相同的位置编码。
                token_positions = token_positions.unsqueeze(-2)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        mask = self.causal_mask[:seq_len, :seq_len]
        # 将头维度作为额外的批次维度处理。
        output = scaled_dot_product_attention(Q, K, V, mask)
        # 将各个头合并回「…，seq_len，d_model」。
        output = output.transpose(-3, -2)
        output = output.reshape(*output.shape[:-2], self.d_model)
        output = self.o_proj(output)
        return output
