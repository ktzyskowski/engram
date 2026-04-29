import torch
import torch.nn.functional as F

from engram.loss.world_model import WorldModelLoss


def _log_probs(*shape):
    return F.log_softmax(torch.randn(*shape), dim=-1)


def _two_hot(B, T, bins):
    # random valid soft target distribution over bins
    return F.softmax(torch.randn(B, T, bins), dim=-1)


def _make_inputs(B=2, T=3, O=5, bins=11, K=4, C=8):
    return dict(
        obs=torch.randn(B, T, O),
        dones=torch.zeros(B, T),
        target_reward_logits=_two_hot(B, T, bins),
        reconstructed_obs=torch.randn(B, T, O),
        continue_logits=torch.randn(B, T, 1),
        reward_logits=torch.randn(B, T, bins),
        posterior_log_probs=_log_probs(B, T, K, C),
        prior_log_probs=_log_probs(B, T, K, C),
    )


def test_returns_loss_and_metrics():
    loss_fn = WorldModelLoss()
    loss, metrics = loss_fn(**_make_inputs())
    assert loss.dim() == 0
    assert isinstance(metrics, dict)


def test_metrics_keys_present():
    loss_fn = WorldModelLoss()
    _, metrics = loss_fn(**_make_inputs())
    expected = {
        "world_model/loss",
        "world_model/obs_loss",
        "world_model/continue_loss",
        "world_model/reward_loss",
        "world_model/posterior_kl",
        "world_model/prior_kl",
    }
    assert expected <= set(metrics.keys())


def test_loss_is_finite():
    loss_fn = WorldModelLoss()
    loss, _ = loss_fn(**_make_inputs())
    assert torch.isfinite(loss).item()


def test_free_nats_floors_kl():
    # with identical posterior and prior log-probs, raw KL ~= 0; free_nats should floor it
    loss_fn = WorldModelLoss(beta_prediction=0.0, beta_posterior=1.0, beta_prior=1.0, free_nats=2.0)
    inputs = _make_inputs()
    log_probs = _log_probs(2, 3, 4, 8)
    inputs["posterior_log_probs"] = log_probs
    inputs["prior_log_probs"] = log_probs.clone()
    loss, metrics = loss_fn(**inputs)
    # both KL terms clamped at 2.0; total = 1.0 * 2.0 + 1.0 * 2.0 = 4.0
    assert torch.isclose(loss, torch.tensor(4.0), atol=1e-5)


def test_kl_detach_directions():
    # posterior loss should NOT propagate gradients into prior_log_probs (it's detached)
    loss_fn = WorldModelLoss(beta_prediction=0.0, beta_posterior=1.0, beta_prior=0.0, free_nats=0.0)
    inputs = _make_inputs()
    inputs["prior_log_probs"] = inputs["prior_log_probs"].detach().requires_grad_(True)
    inputs["posterior_log_probs"] = inputs["posterior_log_probs"].detach().requires_grad_(True)
    loss, _ = loss_fn(**inputs)
    loss.backward()
    # gradient should reach posterior, but not prior (since we used beta_prior=0 and detach inside)
    assert inputs["posterior_log_probs"].grad is not None
    assert inputs["posterior_log_probs"].grad.abs().sum().item() > 0
    assert inputs["prior_log_probs"].grad is None or inputs["prior_log_probs"].grad.abs().sum().item() == 0


def test_prediction_loss_only_when_kl_betas_zero():
    loss_fn = WorldModelLoss(beta_posterior=0.0, beta_prior=0.0, beta_prediction=1.0, free_nats=0.0)
    inputs = _make_inputs()
    loss, metrics = loss_fn(**inputs)
    expected = (
        metrics["world_model/obs_loss"]
        + metrics["world_model/continue_loss"]
        + metrics["world_model/reward_loss"]
    )
    assert abs(loss.item() - expected) < 1e-5
