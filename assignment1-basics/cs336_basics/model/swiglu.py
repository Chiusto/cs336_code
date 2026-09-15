import torch
import math
from .linear import Linear
class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None):
        if d_ff is None:
            d_ff = math.ceil((8 / 3) * d_model / 64) * 64
        super().__init__()
        # 注意和x同一维度的in_features是第一个参数
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self,x:torch.Tensor)->torch.Tensor:

        tmp = SiLU(self.w1(x))
        return self.w2(torch.mul(tmp,self.w3(x)))

def SiLU(x:torch.Tensor):
    # 注意SiLU是逐元素相乘
    return torch.mul(x,torch.sigmoid(x))