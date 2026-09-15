import torch

def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor, # (batch)或(batch,1)
    policy_log_probs: torch.Tensor, # 策略模型对token的概率的对数，(batch_size, sequence_length)
    importance_reweighting_method: str = "none",
    old_log_probs: torch.Tensor | None = None, # GSPO必须,暂不实现
    cliprange: float | None = None, # GSPO必须,暂不实现
    response_mask: torch.Tensor | None = None, # GSPO必须,暂不实现
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]: 

    if response_mask is not None:
        response_lengths = response_mask.sum(dim=1)
        # 避免把 padding 部分的 log probability 也加进去
        mean_log_prob = (policy_log_probs*response_mask).sum(dim=1) / response_lengths
    else:
        mean_log_prob = policy_log_probs.sum(dim=1) / policy_log_probs.shape[1]

    advantages = raw_rewards_or_advantages.reshape(-1,1)
    
    if importance_reweighting_method == "none":
        per_token_loss = - advantages * policy_log_probs
    elif importance_reweighting_method == "noclip":
        r_t = torch.exp(policy_log_probs - old_log_probs)
        per_token_loss = - advantages * r_t
    elif importance_reweighting_method == "grpo" and cliprange is not None:
        # 逐 token 重要性比率,(batch, seq_len)
        r_t = torch.exp(policy_log_probs - old_log_probs)
        per_token_loss = - torch.minimum(r_t * advantages,torch.clamp(r_t,min=1-cliprange,max=1+cliprange) * advantages)
    elif importance_reweighting_method == "gspo" and cliprange is not None:
        # 逐 token 的 log 比率,(batch, seq_len)
        log_ratio = policy_log_probs - old_log_probs
        # 这里不是取均值，要考虑到有掩码的影响
        # .clamp(min=1) 保证分母至少是 1,避免除0
        if response_mask is not None:
            seq_log_ratio = (log_ratio*response_mask).sum(-1,keepdim=True)/response_mask.sum(-1,keepdim=True).clamp(min=1)
        else:
            # keepdim=True，但保的是维度数量，不是维度大小；S被压成1，但维度还在
            seq_log_ratio = log_ratio.mean(dim=-1,keepdim=True) 
        r_seq = torch.exp(seq_log_ratio)
        clipped_seq = torch.clamp(r_seq, 1-cliprange, 1+cliprange)
        # .expand_as(policy_log_probs) 是把一个小张量广播成大张量的形状，和 policy_log_probs 完全一样。
        per_token_loss = - torch.minimum(r_seq * advantages,clipped_seq * advantages).expand_as(policy_log_probs)
        
    return per_token_loss,{}