import torch
class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(
                torch.ones(d_model,device = device,dtype = dtype)                
            )
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        # 提升精度，以防对输入平方时溢出
        x = x.to(torch.float32)
        # 这里注意输入的张量有三个维度的
        rms = torch.sqrt(torch.mean(x ** 2,dim=-1,keepdim = True)+self.eps)        
        result = x / rms * self.weight
        return result.to(in_dtype)


# 资源核算
# x**2: d_model次乘法，一次加法
# rms这里是常数，x / rms * self.weight，FLOPs为2*batch_size*d_model
# 2*batch_size*d_model + d_model + 1