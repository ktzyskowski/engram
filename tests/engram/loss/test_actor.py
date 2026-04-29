import torch
import torch.nn.functional as F

from engram.loss.actor import ActorLoss


def _log_probs(B, T, D, A):
    return F.log_softmax(torch.randn(B, T, D, A), dim=-1)


def _one_hot_actions(log_probs):
    idx = torch.distributions.Categorical(logits=log_probs).sample()
    return F.one_hot(idx, num_classes=log_probs.shape[-1]).float()


def _make_inputs(B=2, T=3, D=4, A=5):
    log_probs = _log_probs(B, T, D, A).requires_grad_(True)
    return dict(
        values=torch.randn(B, T, D),
        lambda_returns=torch.randn(B, T, D),
        actions=_one_hot_actions(log_probs.detach()),
        action_log_probs=log_probs,
    )


def test_returns_loss_and_metrics():
    loss_fn = ActorLoss(eta=3e-4, advantage_ema_decay=0.99)
    loss, metrics = loss_fn(**_make_inputs())
    assert loss.dim() == 0
    assert isinstance(metrics, dict)


def test_metrics_keys_present():
    loss_fn = ActorLoss(eta=3e-4, advantage_ema_decay=0.99)
    _, metrics = loss_fn(**_make_inputs())
    expected = {
        "actor/loss",
        "actor/reinforce",
        "actor/entropy",
        "actor/advantage_abs_mean",
        "actor/advantage_norm",
        "actor/return_spread",
        "actor/return_percentile_high",
        "actor/return_percentile_low",
    }
    assert expected <= set(metrics.keys())


def test_advantage_norm_clamped_at_one():
    # tiny return spread -> norm should clamp to 1.0
    loss_fn = ActorLoss(eta=0.0, advantage_ema_decay=0.99)
    inputs = _make_inputs()
    inputs["lambda_returns"] = torch.full_like(inputs["lambda_returns"], 0.5)
    _, metrics = loss_fn(**inputs)
    assert metrics["actor/advantage_norm"] == 1.0


def test_no_grad_through_values_or_returns():
    # advantage is detached -> values/lambda_returns should receive no gradient
    loss_fn = ActorLoss(eta=0.0, advantage_ema_decay=0.99)
    inputs = _make_inputs()
    inputs["values"] = inputs["values"].requires_grad_(True)
    inputs["lambda_returns"] = inputs["lambda_returns"].requires_grad_(True)
    loss, _ = loss_fn(**inputs)
    loss.backward()
    assert inputs["values"].grad is None or inputs["values"].grad.abs().sum().item() == 0
    assert inputs["lambda_returns"].grad is None or inputs["lambda_returns"].grad.abs().sum().item() == 0


def test_grad_flows_into_action_log_probs():
    loss_fn = ActorLoss(eta=3e-4, advantage_ema_decay=0.99)
    inputs = _make_inputs()
    loss, _ = loss_fn(**inputs)
    loss.backward()
    assert inputs["action_log_probs"].grad is not None
    assert inputs["action_log_probs"].grad.abs().sum().item() > 0


def test_entropy_is_non_negative():
    loss_fn = ActorLoss(eta=3e-4, advantage_ema_decay=0.99)
    _, metrics = loss_fn(**_make_inputs())
    assert metrics["actor/entropy"] >= 0.0


def test_loss_is_finite():
    loss_fn = ActorLoss(eta=3e-4, advantage_ema_decay=0.99)
    loss, _ = loss_fn(**_make_inputs())
    assert torch.isfinite(loss).item()
