import torch
from torch import Tensor
from jaxtyping import jaxtyped, Float, Array

def softmax_tau(
        logits: Float[Tensor,"vocab_size"],
        dim:int,
        tau
        ):
    logits = logits - torch.max(logits, dim = dim, keepdim=True).values
    exp_x = torch.exp(logits/tau)
    return exp_x / torch.sum(exp_x,dim=dim,keepdim=True)

def nucleus_sampling(
        probs: Float[Tensor,"vocab_size"],
        p, # 超参数
        ):
    """
    返回采样后的概率分布
    """
    # 按序排列
    sorted_probs,indices = torch.sort(probs,descending=True)
    # 取前达到p的元素
    cumulative_probs = torch.cumsum(sorted_probs,dim=-1)
    cutoff = torch.where(cumulative_probs >= p)[0][0]
    sorted_probs[cutoff+1:] = 0
    sorted_probs = sorted_probs / sorted_probs.sum()

    new_probs = torch.zeros_like(probs)
    new_probs[indices] = sorted_probs
    return new_probs

def decode(
        x:torch.Tensor, # (batch, sequence_length) x 存的是 token ID
        model,
        dim,
        tau,
        p,
        max_new_tokens
    ):
    for i in range(x.shape[0]):
        for _ in range(max_new_tokens):
            logits = model(x[i]) # (sequence_length, vocab_size)
            next_token_logits = logits[-1]
            probs = softmax_tau(next_token_logits,dim,tau)
            new_probs = nucleus_sampling(probs,p)
            next_token = torch.multinomial(new_probs,1)
            x[i] = torch.cat((x[i],next_token))
    return x

"""x[i]
 ↓
model(x[i])
 ↓
logits
 ↓
logits[-1]
 ↓
next_token_logits
 ↓
softmax / temperature / top-p
 ↓
采样
 ↓
next_token   ← 一个 token ID
 ↓
追加到 x[i]
 ↓
新的 x[i]
 ↓
再次 model(x[i])"""