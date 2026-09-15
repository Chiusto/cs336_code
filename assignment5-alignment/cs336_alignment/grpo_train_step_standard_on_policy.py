import torch
from typing import Callable, Literal
from transformers import PreTrainedTokenizerBase
from cs336_alignment.compute_policy_gradient_loss import compute_policy_gradient_loss
from cs336_alignment.aggregate_loss_across_microbatch_sequence import aggregate_loss_across_microbatch

def grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean", 
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none", 
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """执行前向和后向传播，使用梯度累积步数（gradient_accumulation_steps）微批次。
    参数：
        model: PreTrainedModel
            要训练的HuggingFace模型。
        tokenizer: PreTrainedTokenizer
            用于分词的tokenizer。
        optimizer: Optimizer
            模型的优化器。
        gradient_accumulation_steps: int
            每个优化器步骤的微批次数量。
        max_grad_norm: float | None
            如果不为None，则在调用optimizer.step()之前将梯度范数裁剪为此值。
        reward_fn: Callable[[str, str], dict[str, float]]
            对生成回复与真实答案进行评分，生成包含"reward"、"format_reward"和"answer_reward"键的字典。
        repeated_prompts: list[str]
            示例的提示词列表。该列表的长度为rollout_batch_size，因为每个示例的提示词会重复group_size次。
        rollout_responses: list[str]
            策略生成的回复列表。长度为rollout_batch_size = n_prompts_per_rollout_batch * group_size。
        repeated_ground_truths: list[str]
            示例的真实答案列表。该列表的长度为rollout_batch_size，因为每个示例的真实答案会重复group_size次。
        group_size: int
            每个问题（组）的回复数量。
        baseline: Literal["mean", "none"]
            如果为mean，则减去每组的平均奖励；如果为none，则不进行操作。
        advantage_eps: float
            用于避免归一化时分母为零的小常数。
        advantage_normalizer: Literal["std", "none", "mean"]
            如果为std，则除以每组的标准差；如果为none，则不进行操作；如果为mean，则除以每组的平均奖励。
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none"：不进行重要性重加权；"noclip"：应用重要性重加权但不裁剪；
            "grpo"：进行PPO/GRPO风格的token级重加权和裁剪；
            "gspo"：进行GSPO风格的序列级重加权和裁剪。
        old_log_probs: torch.Tensor | None
            除非importance_reweighting_method = "none"，否则必须提供；形状为(batch_size, sequence_length)。
        cliprange: float | None = None
            裁剪参数epsilon，当importance_reweighting_method为"grpo"或"gspo"时需要提供。
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence"：先对每个序列的损失求平均，再对序列求平均；
            "constant"：将总损失除以一个常数（整个训练过程固定）。
        normalization_constant: int | None = None
            用于除以总损失的常数；当loss_normalization = "constant"时需要提供。

    返回：
        tuple[torch.Tensor, dict[str, torch.Tensor]]。
            loss
                标量张量。批次损失，已针对梯度累积进行调整。
                返回此项以便记录日志。
            metadata
                包含底层损失调用产生的元数据、裁剪前的梯度范数，
                以及你可能希望记录的任何其他统计信息的字典。
    """
    # 1.rollout(已经给出rollout结果)
    # 2.优势计算
    reward = [] # list[str]
    format_reward = []
    answer_reward = []
    for response,truth in zip(rollout_responses,repeated_ground_truths):
        rewards = reward_fn(response,truth)
        reward.append(rewards["reward"])
        format_reward.append(rewards["format_reward"])
        answer_reward.append(rewards["answer_reward"])
    reward = torch.tensor(reward,dtype=torch.float)
    reward_2d = reward.view(-1,group_size)

    # 优势归一化
    if baseline == "mean":
        group_mean = reward_2d.mean(dim = -1,keepdim = True)
        advantages = reward_2d - group_mean
    else:
        advantages = reward_2d

    if advantage_normalizer == "std":
        std = reward_2d.std(dim = -1,keepdim = True)
        advantages = advantages / (std + advantage_eps)
    elif advantage_normalizer == "mean":
        group_mean = reward_2d.mean(dim = -1,keepdim = True)
        advantages = advantages / (group_mean + advantage_eps)

    # [num_groups, group_size]
    # ->
    # [rollout_batch_size]
    advantages = advantages.view(-1)

    # 3.训练
    # 注意不能整除的情况，需要确保处理最后的批次
    microbatch_size = len(repeated_prompts) // gradient_accumulation_steps
    optimizer.zero_grad()
    total_loss = torch.tensor(0.0)

    for i in range(0, len(repeated_prompts), microbatch_size):
        prompts_microbatch = repeated_prompts[i:i+microbatch_size]
        responses_microbatch = rollout_responses[i:i+microbatch_size]
        inputs_microbatch = [
            # prompt 和 response 中间需要空格，
            # 否则例如 "Hello" + "world" 会变成 "Helloworld"。
            prompt + " " + response
            for prompt,response in zip(prompts_microbatch,responses_microbatch)
        ]

        input = tokenizer(
            inputs_microbatch,
            padding = True, # 注意力掩码来源
            return_tensors = "pt"
        )

        device = next(model.parameters()).device
        input_ids = input["input_ids"].to(device)
        attention_mask = input["attention_mask"].to(device)
        advantages_microbatch = advantages[i:i+microbatch_size].to(device)

        # 为构造response mask做准备
        prompt_inputs = tokenizer(
            prompts_microbatch,
            padding = True,
            return_tensors = "pt"
        )
        prompts_lengths = prompt_inputs["attention_mask"].sum(dim=-1)

        labels_microbatch = repeated_ground_truths[i:i+microbatch_size]

        # 前向传播
        # token得分形状（microbatch_size,sequence_len,vocab_size）
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

        # 形状(microbatch_size,sequence_len,vocab_size）softmax只改变数值，不改变形状
        log_probs = torch.log_softmax(logits,dim=-1)

        # target_ids需要和input_ids错开一位
        # target_ids用来选取token,选取时需要对齐，log_probs中实际是下一个token的预测情况
        target_ids = input_ids[:,1:]

        # 构造 response mask
        # 先构造空位置
        position = torch.arange(
            target_ids.size(1),
            device = input_ids.device
        ).unsqueeze(0)

        # 这里语法用到了矩阵广播的特性
        response_mask = (position >= (prompts_lengths - 1).unsqueeze(1))

        # 排除PAD
        response_mask = response_mask & attention_mask[:,1:].bool()

        log_probs_shifted = log_probs[:,:-1,:]

        # token_log_probs的形状(microbatch_size,sequence_len)
        token_log_probs = log_probs_shifted.gather(
            dim = -1,
            index = target_ids.unsqueeze(-1)
        ).squeeze(-1)

        # seq_log_probs为每个*回答*连乘概率取对数
        response_lengths = response_mask.sum(dim=-1).clamp_min(1)
        seq_log_probs = (
            token_log_probs * response_mask
        ).sum(dim = -1) / response_lengths

        loss = -(advantages_microbatch * seq_log_probs).mean()
        loss = loss / gradient_accumulation_steps

        # 反向传播
        loss.backward()

        # 保存所有 microbatch 的平均 loss
        total_loss = total_loss + loss.detach()

        # max_grad_norm 用于梯度裁剪，在这一步
    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm
        )
    else:
        grad_norm = torch.tensor(float("nan"))

    optimizer.step()
    optimizer.zero_grad()

    return total_loss, {"grad_norm": grad_norm}