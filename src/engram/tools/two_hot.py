import torch
import torch.nn as nn
from torch import Tensor


def symlog(x: Tensor) -> Tensor:
    """Compute symlog function.

    Eq (9) in paper.

    Args:
        x (*): input tensor.
    Returns:
        y (*): output tensor.
    """
    y = torch.sign(x) * torch.log(torch.abs(x) + 1)
    return y


def symexp(x: Tensor) -> Tensor:
    """Compute symexp function.

    Eq (9) in paper.

    Args:
        x (*): input tensor.
    Returns:
        y (*): output tensor.
    """
    y = torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
    return y


class SymlogTwoHot(nn.Module):
    """Symlog two-hot module."""

    def __init__(self, low: float, high: float, n_bins: int):
        """Construct a new symlog two-hot encoder.

        Args:
            low (float): lower bound of bins.
            high (float): upper bound of bins.
            n_bins (int): total number of bins.
        """
        super().__init__()
        self._low = low
        self._high = high
        self._n_bins = n_bins
        self._bin_width = (high - low) / (n_bins - 1)

        self._bins: Tensor
        self.register_buffer("_bins", torch.linspace(low, high, n_bins))

    def encode(self, y: Tensor) -> Tensor:
        """Encode the given scalars into symlog two-hot encodings.

        Args:
            y       (*): tensor of scalar values.
        Returns:
            twhot   (*, N): tensor of symlog two-hot values.
        """
        y_symlog = symlog(y).clamp(self._low, self._high)

        pos = (y_symlog - self._low) / self._bin_width
        k = pos.floor().long().clamp(0, self._n_bins - 2)
        upper_weight = pos - k
        lower_weight = 1.0 - upper_weight

        twohot = torch.zeros(*y.shape, self._n_bins, dtype=y.dtype, device=y.device)
        twohot.scatter_(-1, k.unsqueeze(-1), lower_weight.unsqueeze(-1))
        twohot.scatter_(-1, (k + 1).unsqueeze(-1), upper_weight.unsqueeze(-1))
        return twohot

    def decode_logits(self, logits: Tensor) -> Tensor:
        """Decode the given logits into a scalar value.

        Args:
            logits  (*, N): tensor of symlog two-hot logits.
        Returns:
            y       (*):   tensor of scalar values.
        """
        weighted_bins = torch.softmax(logits, dim=-1) * self._bins
        y_symlog = weighted_bins.sum(-1)
        y = symexp(y_symlog)
        return y
