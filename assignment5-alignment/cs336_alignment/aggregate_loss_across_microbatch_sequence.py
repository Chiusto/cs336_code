import torch

def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor, # (batch_size, sequence_length)
    mask: torch.Tensor, # (batch_size, sequence_length)
    loss_normalization: str = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:

    loss = per_token_policy_gradient_loss * mask
    if loss_normalization == "sequence":
        return (loss.sum(dim=1) / mask.sum(dim=1)).mean(dim=0)
    elif loss_normalization == "constant":
        if normalization_constant is not None:
            return loss.sum(dim=1).sum(dim=0) /  normalization_constant
        else:
            raise NotImplementedError