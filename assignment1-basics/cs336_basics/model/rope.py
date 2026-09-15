import torch
import math

# class RotaryPositionalEmbedding(torch.nn.Module):
#     def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
#         super().__init__()
#         # 初始化旋转矩阵
#         rotated_matrix = []
#         for i in range(max_seq_len):
#             blocks = []
#             for j in range(1, d_k//2):
#                 angle_rad = i / (theta**((2*j-2)/d_k))
#                 sin_val = math.sin(angle_rad)
#                 cos_val = math.cos(angle_rad)
#                 block = torch.tensor([[cos_val,-sin_val],[sin_val,cos_val]])
#                 blocks.append(block)
#             rotated_matrix.append(torch.block_diag(*blocks))
#         # 这里要在第0维上堆叠，这样让forward函数中按位置下标进行索引
#         # self.rotated_matrix 的原始形状为 (max_seq_len, d_k, d_k)
#         self.register_buffer(
#             "rotated_matrix",
#             torch.stack(rotated_matrix),
#             persistent=False
#         )

#     def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
#         # x 这里应该是已经生成好的 Q/K,形状(batch_size, seq_len, d_k)
#         # R的维度：(batch_size, seq_len, d_k, d_k)   
#         R = self.rotated_matrix[token_positions]
#         return torch.matmul(R,x.unsqueeze(-1)).squeeze(-1)

# 这种方案只存储sin cos，不用再用矩阵乘法去做
class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        sin = torch.zeros(max_seq_len, d_k // 2, device=device)
        cos = torch.zeros(max_seq_len, d_k // 2, device=device)
        for i in range(max_seq_len):
            for j in range(0, d_k, 2):
                angle_rad = i / (theta ** (j / d_k))
                pair_index = j // 2
                sin[i][pair_index] = math.sin(angle_rad)
                cos[i][pair_index] = math.cos(angle_rad)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.stack((y1, y2), dim=-1).flatten(-2)