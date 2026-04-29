import torch
import torch.nn as nn
from torch import Tensor


class EMA(nn.Module):
    """Exponential Moving Average."""

    def __init__(self, decay: float) -> None:
        """Create new EMA module.

        Args:
            decay (float): rate of decay of average.
        """
        super().__init__()
        assert 0 < decay < 1

        self._decay = decay
        self.average: Tensor
        self.step: Tensor
        self.register_buffer("average", torch.zeros(()))
        self.register_buffer("step", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """Update and return the EMA."""
        self.step.add_(1)
        self.average.mul_(self._decay).add_(x.detach(), alpha=1 - self._decay)
        correction = 1 - self._decay**self.step
        return self.average / correction
