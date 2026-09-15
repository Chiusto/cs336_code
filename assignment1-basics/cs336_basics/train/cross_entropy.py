import torch
from torch import Tensor
from jaxtyping import Int,Float

def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], 
    targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """
    给定输入和目标张量，计算各样本的平均交叉熵损失。
    inputs：其中 inputs[i][j] 是第 i 个样本对第j个词的未归一化 logit。
    targets：形状为 (batch_size,) 的张量，包含正确类别的索引。
        每个值必须介于 0 和 `num_classes - 1` 之间。
    返回：Float[Tensor, ""]：各样本的平均交叉熵损失。
    """
    inputs = inputs - inputs.max(dim=-1,keepdim=True).values
    # 处理任何额外的批次维度
    target_logits = inputs.gather(
        dim=-1,
        index=targets.unsqueeze(-1)
    ).squeeze(-1)
    # 这里消去了 log 和 exp
    loss = -target_logits + torch.logsumexp(inputs, dim=-1)
    return loss.mean()


