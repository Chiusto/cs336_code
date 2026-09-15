import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # 从第 dim 维的所有元素中减去最大值，保证最大元素变成 0
    x = x - torch.max(x, dim = dim, keepdim=True).values
    exp_x = torch.exp(x)
    # keepdim=True 是为了让最大值和分母能够与原张量正确广播，同时输出 shape 与输入完全一致
    # 理解对张量的操作，每一组的exp_x除以所有组exp_x的总和，这样进行归一化
    return exp_x / torch.sum(exp_x,dim=dim,keepdim=True)