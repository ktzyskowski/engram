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
    ):
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

    def forward(self, x: Tensor) -> Tensor:
        y = self.net(x)
        return y
