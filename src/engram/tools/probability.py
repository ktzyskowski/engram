import torch
import torch.nn.functional as F
from torch import Tensor


def unimix(logits: Tensor, frac: float = 0.01) -> Tensor:
    assert frac > 0
    probs = F.softmax(logits, dim=-1)
    uniform = torch.ones_like(probs) / probs.shape[-1]
    mixed_probs = (1 - frac) * probs + frac * uniform
    mixed_log_probs = torch.log(mixed_probs)
    return mixed_log_probs
