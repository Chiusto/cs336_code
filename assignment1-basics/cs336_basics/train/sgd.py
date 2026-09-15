from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        # 初始化了父类 Optimizer，而 self.param_groups 就是父类创建和维护的。
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        # 某些优化器需要在参数更新过程中重新计算 loss 和梯度，这时就会使用 closure
        loss = None if closure is None else closure()
        # self.param_groups来自父类torch.optim.Optimizer
        for group in self.param_groups:
            lr = group["lr"]  # 获取学习率。
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # 获取与 p 关联的状态。
                t = state.get("t", 0)  # 从状态中获取迭代次数，或 0。
                grad = p.grad.data  # 获取损失对 p 的梯度。
                p.data -= lr / math.sqrt(t + 1) * grad  # 就地更新权重张量。
                state["t"] = t + 1  # 递增迭代次数。

        return loss

weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1)

for t in range(10):
    opt.zero_grad()  # 重置所有可学习参数的梯度。
    loss = (weights**2).mean()  # 计算标量损失值。
    print(loss.cpu().item())

    loss.backward()  # 运行反向传播，计算梯度。
    opt.step()  # 运行优化器步骤。