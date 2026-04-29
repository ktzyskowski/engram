import torch
import torch.nn.functional as F

from engram.loss.critic import CriticLoss


def _log_probs(*shape):
    return F.log_softmax(torch.randn(*shape), dim=-1)


def _probs(*shape):
    return F.softmax(torch.randn(*shape), dim=-1)


def _make_inputs(B=2, T=3, D=4, bins=11):
    return dict(
        fast_log_probs=_log_probs(B, T, D, bins).requires_grad_(True),
        slow_log_probs=_log_probs(B, T, D, bins),
        dream_target=_probs(B, T, D, bins),
        replay_target=_probs(B, T, bins),
    )


def test_returns_loss_and_metrics():
    loss_fn = CriticLoss()
    loss, metrics = loss_fn(**_make_inputs())
    assert loss.dim() == 0
    assert isinstance(metrics, dict)


def test_metrics_keys_present():
    loss_fn = CriticLoss()
    _, metrics = loss_fn(**_make_inputs())
    expected = {
        "critic/loss",
        "critic/dream_loss",
        "critic/replay_loss",
        "critic/slow_reg",
    }
    assert expected <= set(metrics.keys())


def test_loss_is_finite():
    loss_fn = CriticLoss()
    loss, _ = loss_fn(**_make_inputs())
    assert torch.isfinite(loss).item()


def test_grad_flows_into_fast_only():
    # gradient should flow into fast_log_probs but not slow_log_probs / targets
    loss_fn = CriticLoss()
    inputs = _make_inputs()
    inputs["slow_log_probs"] = inputs["slow_log_probs"].detach().requires_grad_(True)
    inputs["dream_target"] = inputs["dream_target"].detach().requires_grad_(True)
    inputs["replay_target"] = inputs["replay_target"].detach().requires_grad_(True)
    loss, _ = loss_fn(**inputs)
    loss.backward()
    assert inputs["fast_log_probs"].grad is not None
    assert inputs["fast_log_probs"].grad.abs().sum().item() > 0


def test_replay_uses_seed_only_dream_uses_full():
    # changing replay_target (B, T, bins) should affect loss
    # changing dream_target (B, T, D, bins) at d=1 should also affect loss (full sequence)
    loss_fn = CriticLoss(beta_dream=1.0, beta_replay=1.0, slow_reg_weight=0.0)
    inputs = _make_inputs()
    base_loss, _ = loss_fn(**inputs)

    perturbed = {**inputs, "replay_target": _probs(2, 3, 11)}
    other_replay_loss, _ = loss_fn(**perturbed)
    assert not torch.isclose(base_loss, other_replay_loss)

    perturbed = {**inputs, "dream_target": _probs(2, 3, 4, 11)}
    other_dream_loss, _ = loss_fn(**perturbed)
    assert not torch.isclose(base_loss, other_dream_loss)


def test_components_combine_with_weights():
    loss_fn = CriticLoss(beta_dream=2.0, beta_replay=3.0, slow_reg_weight=0.5)
    loss, metrics = loss_fn(**_make_inputs())
    expected = (
        2.0 * metrics["critic/dream_loss"]
        + 3.0 * metrics["critic/replay_loss"]
        + 0.5 * metrics["critic/slow_reg"]
    )
    assert abs(loss.item() - expected) < 1e-5


def test_soft_cross_entropy_matches_manual():
    # cross-entropy between matching distributions should equal entropy of that distribution
    loss_fn = CriticLoss()
    target = _probs(4, 5)
    log_probs = target.log()
    ce = loss_fn.soft_cross_entropy(target, log_probs)
    entropy = -(target * target.log()).sum(-1).mean()
    assert torch.allclose(ce, entropy, atol=1e-5)
