import torch

from engram.rl.policy import sample_action
from engram.tools.probability import unimix


def test_action_is_one_hot():
    logits = torch.randn(8, 4)
    action, _ = sample_action(logits)
    assert action.shape == logits.shape
    assert torch.equal(action.sum(-1), torch.ones(8))
    assert ((action == 0) | (action == 1)).all().item()


def test_log_probs_full_distribution_shape():
    logits = torch.randn(8, 4)
    _, log_probs = sample_action(logits)
    assert log_probs.shape == logits.shape


def test_log_probs_match_unimix():
    logits = torch.randn(16, 5)
    _, log_probs = sample_action(logits)
    assert torch.allclose(log_probs, unimix(logits, frac=0.01), atol=1e-5)


def test_log_probs_normalized():
    logits = torch.randn(8, 4)
    _, log_probs = sample_action(logits)
    sums = log_probs.exp().sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_peaked_logits_pick_argmax():
    # with very peaked logits, sampling should always select the argmax
    logits = torch.zeros(32, 4)
    logits[:, 2] = 100.0
    action, _ = sample_action(logits)
    assert (action.argmax(-1) == 2).all().item()


def test_batched_leading_dims_preserved():
    logits = torch.randn(3, 5, 7)
    action, log_probs = sample_action(logits)
    assert action.shape == (3, 5, 7)
    assert log_probs.shape == (3, 5, 7)
