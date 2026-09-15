"""
debug_shapes.py - 中间张量形状检查工具
无需改动原模型代码，通过 hook / 断点查看形状

用法1 - 自动打印所有子模块形状（推荐）:
    from cs336_basics.utils.debug_shapes import attach_shape_hooks
    model = Transformer_LM(...)
    attach_shape_hooks(model)
    logits = model(tokens)  # 自动打印每层输入输出形状

用法2 - 手动断点:
    # 在任意 forward 中插入：
    breakpoint()  # 然后在调试控制台输入: x.shape, x.dtype

用法3 - VS Code 图形断点见下方 launch.json
"""
import torch
from typing import Optional

def _fmt(t):
    if isinstance(t, torch.Tensor):
        return f"{tuple(t.shape)} {t.dtype} {t.device}"
    if isinstance(t, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in t) + "]"
    return str(type(t))

def attach_shape_hooks(model: torch.nn.Module, filter_prefix: Optional[str] = None, verbose: bool = True):
    """给所有子模块挂 forward hook，打印输入输出形状"""
    handles = []
    for name, mod in model.named_modules():
        if filter_prefix and not name.startswith(filter_prefix):
            continue
        # 跳过空容器
        if len(list(mod.children())) > 0 and name != "":
            continue  # 只打印叶子模块，避免重复
        def make_hook(n, m):
            def hook(mod, inp, out):
                in_s = _fmt(inp[0] if isinstance(inp, tuple) and len(inp)==1 else inp)
                out_s = _fmt(out)
                print(f"[SHAPE] {n:35s} | {m.__class__.__name__:25s} | in: {in_s:40s} -> out: {out_s}")
            return hook
        handles.append(mod.register_forward_hook(make_hook(name or "root", mod)))
    return handles

def shape_breakpoint(x: torch.Tensor, name: str = "tensor"):
    """在代码中调用： shape_breakpoint(x, 'after_rope') 会打印并进入 pdb"""
    print(f"[BREAKPOINT] {name}: shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}")
    breakpoint()
    return x

# 快捷断言：形状不符直接报错，比断点更快定位
def assert_shape(x: torch.Tensor, expected, name=""):
    if tuple(x.shape) != tuple(expected):
        raise AssertionError(f"{name} shape mismatch: got {tuple(x.shape)} expected {tuple(expected)}")

# 各组件期望形状速查表（以 B=2, T=4, D=8, H=2, Dk=4, Dff=16, V=100 为例）
SHAPE_CHEATSHEET = r"""
输入 tokens: (B, T)  e.g. (2, 4)  int64
Embedding:   (B, T) -> (B, T, D)           (2,4,8)
RMSNorm:     (B, T, D) -> (B, T, D)        不改变形状
Linear:      (..., D_in) -> (..., D_out)   einsum "out d_in, ... d_in -> ... out"
RoPE:        Q (B, H, T, Dk) + pos (T,) -> (B, H, T, Dk)  形状不变
  x1,x2 = x[...,0::2], x[...,1::2]  -> (B,H,T,Dk//2)
Attention:
  Q/K/V proj: (B,T,D)->(B,T,D) -> reshape (B,H,T,Dk)
  scores: Q@K^T -> (B,H,T,T) / sqrt(Dk) -> softmax -> (B,H,T,T)
  out: (B,H,T,T) @ V(B,H,T,Dv) -> (B,H,T,Dv) -> merge -> (B,T,D) -> o_proj -> (B,T,D)
SwiGLU:      (B,T,D) -> w1(B,T,Dff) + w3(B,T,Dff) -> SiLU(w1)*w3 -> w2 -> (B,T,D)
Block:       (B,T,D) -> (B,T,D)  残差连接形状不变
LM:          tokens(B,T) -> logits(B,T,V)  (2,4,100)
"""

if __name__ == "__main__":
    # 自检 demo
    from cs336_basics.model.transformer_lm import Transformer_LM
    B, T, V, D, H, Dff, L = 2, 4, 100, 8, 2, 16, 2
    model = Transformer_LM(V, T, D, L, H, Dff, rope_theta=10000)
    attach_shape_hooks(model)
    tokens = torch.randint(0, V, (B, T))
    print(SHAPE_CHEATSHEET)
    print("="*80)
    logits = model(tokens)
    print(f"final logits: {tuple(logits.shape)}  # 期望 (B,T,V) = ({B},{T},{V})")
