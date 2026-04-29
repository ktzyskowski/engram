import pytest
import torch

from engram.nets.mlp import MLP


def test_output_shape():
    mlp = MLP(input_size=4, hidden_sizes=[8, 8], output_size=3, activation="rmsnorm+silu")
    x = torch.randn(5, 4)
    assert mlp(x).shape == (5, 3)


def test_batched_leading_dims_preserved():
    mlp = MLP(input_size=4, hidden_sizes=[8], output_size=3, activation="rmsnorm+silu")
    x = torch.randn(2, 6, 4)
    assert mlp(x).shape == (2, 6, 3)


def test_no_activation_after_final_linear():
    # last module in the Sequential should be Linear, not the activation
    mlp = MLP(input_size=4, hidden_sizes=[8], output_size=3, activation="rmsnorm+silu")
    assert isinstance(mlp.net[-1], torch.nn.Linear)


def test_zero_output_weights_initializes_linear_to_zero():
    mlp = MLP(
        input_size=4,
        hidden_sizes=[8],
        output_size=3,
        activation="rmsnorm+silu",
        zero_output_weights=True,
    )
    last = mlp.net[-1]
    assert last.weight.abs().sum().item() == 0.0
    assert last.bias.abs().sum().item() == 0.0


def test_zero_output_weights_produces_zero_output():
    mlp = MLP(
        input_size=4,
        hidden_sizes=[8],
        output_size=3,
        activation="rmsnorm+silu",
        zero_output_weights=True,
    )
    y = mlp(torch.randn(7, 4))
    assert y.abs().sum().item() == 0.0


def test_empty_hidden_sizes_asserts():
    with pytest.raises(AssertionError):
        MLP(input_size=4, hidden_sizes=[], output_size=3, activation="rmsnorm+silu")


def test_requires_grad_toggle():
    mlp = MLP(input_size=4, hidden_sizes=[8], output_size=3, activation="rmsnorm+silu")
    mlp.requires_grad_(False)
    assert all(not p.requires_grad for p in mlp.parameters())
    mlp.requires_grad_(True)
    assert all(p.requires_grad for p in mlp.parameters())


def test_layer_count_matches_hidden_depth():
    # for hidden_sizes=[h1, h2, h3], expect 4 Linear layers and 3 activations
    mlp = MLP(
        input_size=4,
        hidden_sizes=[8, 8, 8],
        output_size=3,
        activation="rmsnorm+silu",
    )
    n_linear = sum(1 for m in mlp.net if isinstance(m, torch.nn.Linear))
    assert n_linear == 4
