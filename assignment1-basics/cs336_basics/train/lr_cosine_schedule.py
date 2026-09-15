import torch
import math

def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
    ):
    if it<warmup_iters: 
        return max_learning_rate*it/warmup_iters
    elif warmup_iters<=it and it<=cosine_cycle_iters: 
        return min_learning_rate + (1 + math.cos((it-warmup_iters)/(cosine_cycle_iters-warmup_iters)*torch.pi))*(max_learning_rate-min_learning_rate)/2
    elif it>cosine_cycle_iters:
        return min_learning_rate

    # uv run pytest -k test_gradient_clipping -q