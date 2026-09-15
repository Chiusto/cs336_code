import pathlib
p = pathlib.Path("cs336_basics/train/lr_cosine_schedule.py")
# minimal fix: keep original logic but ensure math.pi and handle warmup 0 safely
content = """import math

def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if warmup_iters > 0 and it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it > cosine_cycle_iters:
        return min_learning_rate
    if cosine_cycle_iters <= warmup_iters:
        # no cosine phase
        return max_learning_rate if it == warmup_iters else min_learning_rate
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (max_learning_rate - min_learning_rate)
"""
p.write_text(content, encoding="utf-8")
print(p.read_text(encoding="utf-8"))
