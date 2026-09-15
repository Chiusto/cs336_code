from collections.abc import Iterable

import torch
def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], 
    max_l2_norm: float) -> None:
    eps = 1e-6
    # 计算所有参数梯度拼起来后的总 L2 norm
    l2_norm = torch.sqrt(
        sum(torch.sum(p.grad ** 2) for p in parameters if p.grad is not None)
    )
    if l2_norm >= max_l2_norm:
        scale = max_l2_norm /(l2_norm+eps)
        for p in parameters:
            if p.grad is not None:
                p.grad *= scale

# uv run pytest -k test_gradient_clipping -q