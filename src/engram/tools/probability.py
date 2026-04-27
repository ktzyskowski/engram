import torch
import torch.nn.functional as F
from torch import Tensor


def unimix(logits: Tensor, frac: float = 0.01) -> Tensor:
    probs = F.softmax(logits, dim=-1)
    uniform = torch.ones_like(probs) / probs.shape[-1]
    mixed = (1 - frac) * probs + frac * uniform
    logits = torch.log(mixed)
    return logits
