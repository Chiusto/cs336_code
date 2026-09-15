"""
norm_monitor.py - 监控 激活 / 权重 / 梯度 范数，防止爆炸/消失
配合 logger.py 使用，3行接入训练循环

原理:
- 范数 = L2 norm = sqrt(sum(x^2)), 反映张量整体量级
- 健康区间: 激活 ~0.5-10, 权重 ~1-20, 梯度 ~1e-4 ~ 10
- 爆炸: >1e3  -> 梯度裁剪 / 降低lr
- 消失: <1e-6 -> 检查初始化 / 残差 / RMSNorm

用法:
    from cs336_basics.utils.norm_monitor import NormMonitor
    from cs336_basics.utils.logger import Logger

    model = Transformer_LM(...)
    logger = Logger("logs/run")
    monitor = NormMonitor(model, logger)  # 自动挂钩激活

    for step in range(max_iters):
        logits = model(tokens)
        loss = criterion(logits, targets)
        loss.backward()

        if step % 100 == 0:
            monitor.log(step)  # 一键记录三类范数 + 自动预警

        optimizer.step()
        optimizer.zero_grad()

    monitor.remove_hooks()
"""
import torch
import math
from typing import Dict, Optional

def tensor_norm(t: torch.Tensor, p: int = 2) -> float:
    """L2范数，float返回"""
    if t is None or t.numel() == 0:
        return 0.0
    # 用 float32 避免溢出
    return t.float().norm(p).item()

def global_norm(tensors, p: int = 2) -> float:
    """多个张量的全局范数: sqrt(sum(norm^2))"""
    total = 0.0
    for t in tensors:
        if t is not None:
            total += (t.float().norm(p).item() ** p)
    return total ** (1.0 / p) if total > 0 else 0.0

class NormMonitor:
    def __init__(self, model: torch.nn.Module, logger=None, warn_threshold: float = 100.0, verbose: bool = True):
        self.model = model
        self.logger = logger
        self.warn_threshold = warn_threshold
        self.verbose = verbose
        self.activation_norms: Dict[str, float] = {}
        self.hooks = []
        self._attach_activation_hooks()

    def _attach_activation_hooks(self):
        """前向钩子: 捕获每层输出激活的范数"""
        for name, mod in self.model.named_modules():
            # 只监控叶子模块 + 关键容器
            if name == "" or len(list(mod.children())) == 0 or "transformer_layers" in name or name in ["embedding", "rmsnorm", "output_embedding"]:
                def make_hook(n):
                    def hook(mod, inp, out):
                        if isinstance(out, torch.Tensor):
                            self.activation_norms[n or "root"] = tensor_norm(out)
                        elif isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                            self.activation_norms[n] = tensor_norm(out[0])
                    return hook
                self.hooks.append(mod.register_forward_hook(make_hook(name)))

    def get_weight_norms(self) -> Dict[str, float]:
        norms = {}
        for name, p in self.model.named_parameters():
            norms[name] = tensor_norm(p.data)
        norms["__global__"] = global_norm([p.data for _, p in self.model.named_parameters()])
        return norms

    def get_grad_norms(self) -> Dict[str, float]:
        norms = {}
        grads = []
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                n = tensor_norm(p.grad)
                norms[name] = n
                grads.append(p.grad)
            else:
                norms[name] = 0.0
        norms["__global__"] = global_norm(grads) if grads else 0.0
        return norms

    def get_activation_norms(self) -> Dict[str, float]:
        # 返回最近一次前向的激活范数
        out = dict(self.activation_norms)
        if out:
            vals = list(out.values())
            out["__global__"] = math.sqrt(sum(v*v for v in vals)) if vals else 0.0
        return out

    def _check(self, name: str, value: float, kind: str):
        if not math.isfinite(value):
            print(f"[WARN] {kind} {name} NaN/Inf -> EXPLODED")
            return
        if value > self.warn_threshold:
            print(f"[WARN] {kind} {name} norm {value:.2e} > {self.warn_threshold} EXPLODE risk -> clip/lower lr")
        elif value < 1e-6 and value != 0:
            print(f"[WARN] {kind} {name} norm {value:.2e} < 1e-6 VANISH risk -> check init/residual")

    def log(self, step: int):
        """计算三类范数，打印预警，并写入 logger"""
        w_norms = self.get_weight_norms()
        g_norms = self.get_grad_norms()
        a_norms = self.get_activation_norms()

        w_global = w_norms.pop("__global__")
        g_global = g_norms.pop("__global__")
        a_global = a_norms.pop("__global__", 0)

        # 控制台摘要
        if self.verbose:
            print(f"[NORM step {step:5d}] act_global {a_global:.3f} | weight_global {w_global:.3f} | grad_global {g_global:.3e}")
            # 详细每层(可选，太多可注释)
            # for k,v in a_norms.items(): print(f"  act {k}: {v:.3f}")

        # 预警
        self._check("__global__", a_global, "act")
        self._check("__global__", w_global, "weight")
        self._check("__global__", g_global, "grad")
        for k,v in list(a_norms.items())[:3]:
            self._check(k, v, "act")
        for k,v in list(g_norms.items())[:3]:
            self._check(k, v, "grad")

        # 写入 logger (同时记录 wall_s)
        if self.logger:
            self.logger.log(step,
                train_loss=None, val_loss=None,  # 占位，由外层训练循环另行记录
                act_norm=a_global, weight_norm=w_global, grad_norm=g_global,
                # 也可展开每层: **{f"act_{k}":v for k,v in a_norms.items()}
            )
        return {"act": a_global, "weight": w_global, "grad": g_global,
                "act_detail": a_norms, "weight_detail": w_norms, "grad_detail": g_norms}

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

# 健康区间速查
HEALTHY_RANGES = """
健康区间 (经验值, Transformer-LM D=512 附近):
- 激活 L2  (每层输出): 0.5 ~ 20      均值 ~ sqrt(D) 约 22
  RMSNorm 后 ~ 1 * sqrt(D) 正常
- 权重 L2  (单矩阵):   5 ~ 30       全局 ~ 50-200
  初始化后 Linear std~0.02, 范数 ~ sqrt(fan) * std
- 梯度 L2  全局:      1e-3 ~ 10     太大->裁剪(clip=1.0), 太小->消失
- 若 act/grad 连续数步 >1e3 或 NaN -> 爆炸，立刻降低 lr / 加 clip / 检查 RoPE/softmax
- 若 梯度 <1e-6 且 loss 不降 -> 消失，检查 残差连接/初始化/RMSNorm
"""

if __name__ == "__main__":
    from cs336_basics.transformer_lm import Transformer_LM
    from cs336_basics.utils.logger import Logger
    print(HEALTHY_RANGES)
    B, T, V, D, H, Dff, L = 2, 4, 100, 8, 2, 16, 2
    model = Transformer_LM(V, T, D, L, H, Dff, rope_theta=10000)
    logger = Logger("tmp_norm_test", config={"test": True})
    monitor = NormMonitor(model, logger)
    tokens = torch.randint(0, V, (B, T))
    logits = model(tokens)
    loss = logits.sum()
    loss.backward()
    monitor.log(0)
    monitor.remove_hooks()
    logger.close()
    print("demo done, check tmp_norm_test/log.jsonl")

