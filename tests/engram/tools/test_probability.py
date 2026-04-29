import pytest
import torch
import torch.nn.functional as F

from engram.tools.probability import unimix


def test_output_is_log_probs():
    logits = torch.randn(4, 5)
    log_probs = unimix(logits, frac=0.01)
    sums = log_probs.exp().sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_uniform_floor_respected():
    logits = torch.tensor([[100.0, -100.0, -100.0, -100.0]])
    frac = 0.1
    probs = unimix(logits, frac=frac).exp()
    n = probs.shape[-1]
    # every class probability must be at least frac/n
    assert (probs >= frac / n - 1e-6).all().item()


def test_argmax_preserved():
    logits = torch.randn(8, 10)
    log_probs = unimix(logits, frac=0.01)
    assert torch.equal(logits.argmax(-1), log_probs.argmax(-1))


def test_small_frac_close_to_softmax():
    logits = torch.randn(4, 10)
    log_probs = unimix(logits, frac=1e-6)
    assert torch.allclose(log_probs.exp(), F.softmax(logits, dim=-1), atol=1e-5)


def test_frac_zero_asserts():
    logits = torch.randn(4, 5)
    with pytest.raises(AssertionError):
        unimix(logits, frac=0.0)
