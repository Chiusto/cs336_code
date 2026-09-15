import torch
import torch.nn.init as init
class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        # 嵌入矩阵： 离散ID映射为连续向量
        # 为了更接近官方接口，建议一律使用weight
        self.weight = torch.nn.Parameter(
            torch.empty(num_embeddings,embedding_dim,device=device,dtype=dtype)
        )
        with torch.no_grad():
            init.trunc_normal_(
                self.weight,
                mean = 0.0,
                std = 1,
                a = -3.0,
                b = 3.0
            )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]