import pytest
import torch
import torch.nn.functional as F

from engram.nets.activations import RMSNormSiLU, resolve_activation


def test_resolve_known_activation():
    act = resolve_activation("rmsnorm+silu")
    assert isinstance(act, RMSNormSiLU)


def test_resolve_unknown_raises():
    with pytest.raises(ValueError):
        resolve_activation("not-a-real-activation")


def test_rmsnorm_silu_output_shape():
    act = RMSNormSiLU()
    x = torch.randn(3, 5, 7)
    y = act(x)
    assert y.shape == x.shape


def test_rmsnorm_silu_matches_manual():
    act = RMSNormSiLU()
    x = torch.randn(4, 8)
    expected = F.silu(F.rms_norm(x, (x.shape[-1],)))
    assert torch.allclose(act(x), expected)


def test_rmsnorm_silu_normalizes_along_last_dim():
    act = RMSNormSiLU()
    # scaling input along the last dim shouldn't change RMS-normalized values
    x = torch.randn(4, 8)
    y1 = act(x)
    y2 = act(x * 100.0)
    assert torch.allclose(y1, y2, atol=1e-4)
