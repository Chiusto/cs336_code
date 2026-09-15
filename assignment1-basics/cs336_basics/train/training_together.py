import torch
import numpy as np
import os
from .data_loading import data_loading
from .cross_entropy import cross_entropy
from .checkpoint import save_checkpoint
from .lr_cosine_schedule import lr_cosine_schedule  # 新增
from .gradient_clipping import gradient_clipping    # 新增
def train(
    model,
    optimizer,
    train_data_path,
    val_data_path,
    batch_size,
    context_length,
    max_iters,
    device,
    checkpoint_path,
    eval_interval=100,
    max_lr=3e-4,
    min_lr=3e-5,
    warmup_iters=1000,
    max_norm=1.0
):
    # memmap:把一个很大的二进制文件，当成 NumPy 数组来使用，但不会一次性把整个文件加载进内存
    train_data = np.memmap(
        train_data_path,
        dtype=np.uint16,
        mode ="r"
    )
    val_data = np.memmap(
        val_data_path,
        dtype=np.uint16,
        mode="r"
    )
    model.to(device) # 移动参数
    model.train() # 指定模式（训练）
    os.makedirs(os.path.dirname(checkpoint_path) or ".",exist_ok=True) # 新增
    for iteration in range(max_iters):
        # 每步动态改lr,这就是warmup+cosine的设置位置
        lr=lr_cosine_schedule(iteration,max_lr,min_lr,warmup_iters,max_iters)
        for g in optimizer.param_groups:
            g["lr"]=lr
        inputs,targets = data_loading(train_data,batch_size,context_length,device)
        logits = model(inputs) # 前向传播
        loss = cross_entropy(logits,targets) # 计算loss
        optimizer.zero_grad() # 清空梯度
        loss.backward() # 反向传播
        # 梯度裁剪，防止梯度爆炸
        gradient_clipping(model.parameters(),max_norm)
        optimizer.step() # 更新参数
        # 定期打印loss
        if iteration % eval_interval==0:
            print(
                f"iteration {iteration}, "
                f"train loss: {loss.item():.4f}"
            )
            # 定时拿一批模型没有参与参数更新的数据，检查它的泛化能力。
            model.eval()
            with torch.no_grad():
                val_inputs,val_targets = data_loading(val_data,batch_size,context_length,device)
                val_logits = model(val_inputs)
                val_loss = cross_entropy(val_logits,val_targets)
            print(
                f"iteration {iteration}, "
                f"val loss: {val_loss.item():.4f}"
            )
            model.train()
        if iteration % eval_interval == 0:
            save_checkpoint(model,optimizer,iteration,checkpoint_path)

