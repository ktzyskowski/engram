import torch

from engram.tools.two_hot import SymlogTwoHot, symexp, symlog


def test_symlog_zero_is_zero():
    assert symlog(torch.zeros(3)).abs().max().item() == 0.0


def test_symexp_zero_is_zero():
    assert symexp(torch.zeros(3)).abs().max().item() == 0.0


def test_symlog_symexp_inverse():
    x = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0, 12345.0])
    assert torch.allclose(symexp(symlog(x)), x, atol=1e-4)


def test_symlog_preserves_sign():
    x = torch.tensor([-5.0, -0.1, 0.0, 0.1, 5.0])
    y = symlog(x)
    assert torch.equal(torch.sign(y), torch.sign(x))


def test_encode_distribution_sums_to_one():
    enc = SymlogTwoHot(low=-20.0, high=20.0, n_bins=255)
    y = torch.tensor([-3.7, 0.0, 1.0, 42.0])
    two_hot = enc.encode(y)
    sums = two_hot.sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_encode_at_bin_center_is_one_hot():
    # symlog(0) = 0, which is the middle of [-20, 20] with odd n_bins -> exact bin center
    enc = SymlogTwoHot(low=-20.0, high=20.0, n_bins=21)
    two_hot = enc.encode(torch.tensor(0.0))
    assert two_hot.max().item() == 1.0
    assert (two_hot == 0.0).sum().item() == 20


def test_encode_decode_roundtrip():
    enc = SymlogTwoHot(low=-20.0, high=20.0, n_bins=255)
    y = torch.tensor([-50.0, -1.0, 0.0, 0.5, 7.0, 200.0])
    two_hot = enc.encode(y)
    # treat the two-hot distribution as logits via log; softmax recovers it
    logits = (two_hot + 1e-8).log()
    y_hat = enc.decode_logits(logits)
    assert torch.allclose(y_hat, y, atol=1e-2, rtol=1e-2)


def test_encode_clamps_out_of_range():
    enc = SymlogTwoHot(low=-2.0, high=2.0, n_bins=5)
    # symlog(huge) >> 2, should clamp into the top bin
    y = torch.tensor([1e9])
    two_hot = enc.encode(y)
    assert two_hot[..., -1].item() == 1.0


def test_encode_shape_preserved():
    enc = SymlogTwoHot(low=-20.0, high=20.0, n_bins=255)
    y = torch.randn(4, 8)
    two_hot = enc.encode(y)
    assert two_hot.shape == (4, 8, 255)
