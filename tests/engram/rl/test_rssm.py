import pytest
import torch

from engram.rl.rssm import RSSM


@pytest.fixture
def rssm():
    return RSSM(
        n_categoricals=4,
        n_classes=8,
        observation_size=6,
        action_size=3,
        recurrent_size=16,
        posterior_hidden_sizes=[32],
        prior_hidden_sizes=[32],
    )


def test_initial_recurrent_state_shape_and_zero(rssm):
    h = rssm.get_initial_recurrent_state()
    assert h.shape == (16,)
    assert h.abs().sum().item() == 0.0


def test_full_state_size(rssm):
    assert rssm.full_state_size == 16 + 4 * 8


def test_step_output_shape(rssm):
    B = 5
    h = torch.zeros(B, 16)
    z = torch.zeros(B, 4 * 8)
    a = torch.zeros(B, 3)
    h_next = rssm.step(h, z, a)
    assert h_next.shape == (B, 16)


def test_get_posterior_shapes(rssm):
    B = 5
    obs = torch.randn(B, 6)
    h = torch.zeros(B, 16)
    z, log_probs = rssm.get_posterior(obs, h)
    assert z.shape == (B, 4 * 8)
    assert log_probs.shape == (B, 4, 8)


def test_get_prior_shape(rssm):
    h = torch.randn(7, 16)
    z = rssm.get_prior(h)
    assert z.shape == (7, 4 * 8)


def test_stochastic_state_is_one_hot_per_categorical(rssm):
    # z has shape (B, K*C); reshaping to (B, K, C) should give a one-hot per K row
    obs = torch.randn(8, 6)
    h = torch.zeros(8, 16)
    z, _ = rssm.get_posterior(obs, h)
    z = z.unflatten(-1, (4, 8))
    sums = z.sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums))
    assert ((z == 0) | (z == 1)).all().item()


def test_posterior_log_probs_normalized(rssm):
    obs = torch.randn(8, 6)
    h = torch.zeros(8, 16)
    _, log_probs = rssm.get_posterior(obs, h)
    sums = log_probs.exp().sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_forward_output_shapes(rssm):
    B, T = 3, 7
    obs = torch.randn(B, T, 6)
    actions = torch.zeros(B, T, 3)
    dones = torch.zeros(B, T)
    h, z, post_lp, prior_lp = rssm(obs, actions, dones)
    assert h.shape == (B, T, 16)
    assert z.shape == (B, T, 4 * 8)
    assert post_lp.shape == (B, T, 4, 8)
    assert prior_lp.shape == (B, T, 4, 8)


def test_forward_resets_h_after_done(rssm):
    B, T = 2, 4
    obs = torch.randn(B, T, 6)
    actions = torch.randn(B, T, 3)
    # done at t=1 in batch 0 -> recurrent state at t=2 should be zero for batch 0
    dones = torch.zeros(B, T)
    dones[0, 1] = 1.0
    h, _, _, _ = rssm(obs, actions, dones)
    assert h[0, 2].abs().sum().item() == 0.0
    # batch 1 was never done -> h at t=2 should not be zero (with high probability)
    assert h[1, 2].abs().sum().item() > 0.0


def test_forward_first_step_h_is_zero(rssm):
    B, T = 2, 3
    obs = torch.randn(B, T, 6)
    actions = torch.randn(B, T, 3)
    dones = torch.zeros(B, T)
    h, _, _, _ = rssm(obs, actions, dones)
    assert h[:, 0].abs().sum().item() == 0.0
