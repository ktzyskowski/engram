import torch.nn as nn
from torch import Tensor

from engram.nets.activations import resolve_activation


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        activation: str,
        zero_output_weights: bool = False,
    ) -> None:
        super().__init__()
        assert len(hidden_sizes) > 0

        sizes = [input_size] + hidden_sizes + [output_size]
        layers = []
        for in_features, out_features in zip(sizes, sizes[1:]):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(resolve_activation(activation))

        # remove last activation layer
        layers = layers[:-1]
        self.net = nn.Sequential(*layers)

        if zero_output_weights:
            nn.init.zeros_(self.net[-1].weight)  # type: ignore
            nn.init.zeros_(self.net[-1].bias)  # type: ignore

    def forward(self, x: Tensor) -> Tensor:
        y = self.net(x)
        return y

    def requires_grad_(self, mode: bool) -> None:
        for param in self.parameters():
            param.requires_grad_(mode)
