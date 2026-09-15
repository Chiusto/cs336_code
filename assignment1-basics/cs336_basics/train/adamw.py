import torch
import math
from typing import Optional, Callable

class AdamW(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 lr = 1e-3,
                 betas = (0.9,0.999),
                 eps = 1e-8,
                 weight_decay=1e-2
                 ):
        if lr<0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
            }        
        # 父类会帮你建立：self.param_groups,self.state
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        # 遍历所有参数（参数组 → params）
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                # 每个参数 p 都有自己独立的一份 state
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                a_t = lr * (math.sqrt(1 - beta2**t) / (1 - beta1**t))
                with torch.no_grad():
                    p -= lr * weight_decay * p
                state["m"] = beta1*state["m"] + (1-beta1)*p.grad
                state["v"] = beta2*state["v"] +  (1-beta2)*(p.grad**2)
                with torch.no_grad():
                    p -= a_t*state["m"]/(torch.sqrt(state["v"])+eps)
                # 统计的是“这个参数被 AdamW 更新了多少次”
        return loss

# uv run pytest -k test_adamw -v