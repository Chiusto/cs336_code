"""debug_run.py - 一键检查所有中间形状"""
import torch
from cs336_basics.transformer_lm import Transformer_LM
from cs336_basics.utils.debug_shapes import attach_shape_hooks, SHAPE_CHEATSHEET

print(SHAPE_CHEATSHEET)
B, T, V, D, H, Dff, L = 2, 4, 100, 8, 2, 16, 2
model = Transformer_LM(V, T, D, L, H, Dff, rope_theta=10000)
handles = attach_shape_hooks(model)
tokens = torch.randint(0, V, (B, T))
print(f"input tokens: {tuple(tokens.shape)}")
print("="*80)
with torch.no_grad():
    logits = model(tokens)
print("="*80)
print(f"final logits: {tuple(logits.shape)}  expected ({B},{T},{V})")
assert tuple(logits.shape) == (B, T, V), "LM 输出形状错误！"
print("OK shape check passed")
for h in handles: h.remove()

