from jaxtyping import Bool, Float
import math
import torch
from torch import Tensor
from .softmax import softmax


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.size(-1)
    probability = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        probability = probability.masked_fill(~mask, float("-inf"))
    return torch.matmul(softmax(probability, dim=-1), V)
