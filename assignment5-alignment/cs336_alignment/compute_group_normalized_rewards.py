import torch

def compute_group_normalized_rewards(
    reward: torch.Tensor, # (batch_size , group_size,)
    group_size: int,
    baseline: str = "mean",
    advantage_normalizer: str = "std",
    advantage_eps: float = 1e-8,
) -> torch.Tensor:

    # 重塑形状
    reward_2d = reward.view(-1,group_size)
    out = reward_2d.clone()

    if baseline == "mean":
        group_mean = out.mean(dim=1,keepdim=True)
        out = out - group_mean

    if advantage_normalizer == "std":
        group_std = out.std(dim=1,keepdim=True)
        out = out / (group_std + advantage_eps)
    elif advantage_normalizer == "mean":
        out = out / (group_mean + advantage_eps)

    advantages = out.view(-1)
    return advantages

