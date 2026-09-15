import torch
import torch.nn.init as init
import math
from einops import einsum
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        # nn.Module的__init__方法会创建大量重要的内部属性和数据结构:存储参数（如权重、偏置）等等
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty(out_features,in_features,device = device,dtype = dtype)
        )
        std = math.sqrt(2.0 / (in_features + out_features))
        init.trunc_normal_(
            self.weight,
            mean = 0.0,
            std = std,
            a = -3.0 * std,
            b = 3.0 * std
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "out d_in, ... d_in -> ... out")