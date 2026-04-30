import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def resolve_activation(name: str) -> nn.Module:
    if name == "rmsnorm+silu":
        return RMSNormSiLU()
    else:
        raise ValueError(f"Activation '{name}' not recognized.")


class RMSNormSiLU(nn.Module):
    """RMSNorm + SiLU activation function."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x) -> Tensor:
        # x = F.rms_norm(x, (x.shape[-1],))
        rms = x.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
        x = F.silu(x * rms)
        return x
