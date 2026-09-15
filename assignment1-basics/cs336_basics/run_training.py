from .model.transformer_lm import Transformer_LM
from .train.adamw import Adamw
from .train.training_together import train
import torch
device="cuda" if torch.cuda.is_available() else "cpu"
# 1. 定死的结构超参
model = Transformer_LM(
    vocab_szie=10000,
    context_length=256,
    d_model=512,
    num_layers=4,
    num_heads=16,
    d_ff=1344,
    rope_theta=10000
)
# 2. 定死的优化器超参
optimizer=Adamw(
    model.parameters(),
    lr=3e-4,
    betas=(0.9,0.95),
    eps=1e-8,
    weight_decay=0.01
)
 # 3. 预算反推步数:处理的总词元=总步数*批次*上下文长度（总步数=总样本数/批次大小）
batch_size,context_length=32,256
max_iters=327680000//(batch_size*context_length)
warmup_iters = 2000

train(
    model,optimizer,
    # # 需要你先用BPE把txt tokenize成uint16的npy
    train_data_path="data/TinyStories_train.npy",
    val_data_path="data/TinyStories_valid.npy",
    batch_size=batch_size,
    context_length=context_length,
    max_iters=max_iters,
    device=device,
    checkpoint_path="checkpoints/model.pt",
    max_lr=3e-4,
    min_lr=3e-5,
    warmup_iters=warmup_iters
)



