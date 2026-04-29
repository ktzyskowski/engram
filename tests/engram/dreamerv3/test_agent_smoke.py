import gymnasium as gym
import pytest
import torch

from engram.dreamerv3.agent import DreamerV3


def _make_params():
    return {
        # rssm
        "n_categoricals": 4,
        "n_classes": 4,
        "encoder_output_size": 8,
        "recurrent_size": 16,
        "posterior_hidden_sizes": [16],
        "prior_hidden_sizes": [16],
        "posterior_activation": "rmsnorm+silu",
        "prior_activation": "rmsnorm+silu",
        "unimix": 0.01,
        # heads
        "reward_hidden_sizes": [16],
        "reward_activation": "rmsnorm+silu",
        "continue_hidden_sizes": [16],
        "continue_activation": "rmsnorm+silu",
        # actor + critic
        "actor_hidden_sizes": [16],
        "actor_activation": "rmsnorm+silu",
        "critic_hidden_sizes": [16],
        "critic_activation": "rmsnorm+silu",
        # encoder + decoder
        "encoder_hidden_sizes": [16],
        "encoder_activation": "rmsnorm+silu",
        "decoder_hidden_sizes": [16],
        "decoder_activation": "rmsnorm+silu",
        "observation_size": 4,  # CartPole obs dim
        # two-hot
        "two_hot_low": -10.0,
        "two_hot_high": 10.0,
        "two_hot_n_bins": 21,
        # optim
        "world_model_lr": 1e-3,
        "actor_lr": 1e-3,
        "critic_lr": 1e-3,
        "critic_tau": 0.02,
    }


@pytest.fixture
def agent():
    env = gym.make("CartPole-v1")
    return DreamerV3(
        env=env,
        params=_make_params(),
        device="cpu",
        batch_size=2,
        sequence_length=4,
        prefill_steps=4,
    )


def test_construct(agent):
    assert agent is not None


def test_collect_step_populates_buffer(agent):
    obs, _ = agent._env.reset()
    obs = torch.from_numpy(obs)
    h = agent._rssm.get_initial_recurrent_state()

    for _ in range(3):
        obs, h, reward, done = agent.collect_step(obs, h)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert obs.shape == (4,)
        assert h.shape == (16,)

    assert len(agent._replay_buffer) == 3


def test_update_step_returns_metrics_after_prefill(agent):
    obs, _ = agent._env.reset()
    obs = torch.from_numpy(obs)
    h = agent._rssm.get_initial_recurrent_state()

    # prefill enough transitions to satisfy sequence_length=4
    for _ in range(8):
        obs, h, _, _ = agent.collect_step(obs, h)

    metrics = agent.update_step()

    expected_prefixes = ("world_model/", "actor/", "critic/")
    assert all(any(k.startswith(p) for p in expected_prefixes) for k in metrics.keys())
    assert "world_model/loss" in metrics
    assert "actor/loss" in metrics
    assert "critic/loss" in metrics
    for k, v in metrics.items():
        assert torch.isfinite(torch.tensor(v)).item(), f"{k} is not finite: {v}"


def test_train_yields_metrics_through_prefill(agent):
    # 6 total steps; first 4 are prefill (no updates), last 2 should run updates
    yielded = list(agent.train(steps=6))
    assert len(yielded) == 6

    for step, m in enumerate(yielded):
        assert m["step"] == step

    # prefill steps should NOT have loss metrics; post-prefill should
    prefill = yielded[: agent._prefill_steps]
    post = yielded[agent._prefill_steps :]
    for m in prefill:
        assert "world_model/loss" not in m
    for m in post:
        assert "world_model/loss" in m
        assert "gradient_step" in m


def test_slow_critic_updates_after_training_step(agent):
    # snapshot slow critic params, run an update, verify they moved
    obs, _ = agent._env.reset()
    obs = torch.from_numpy(obs)
    h = agent._rssm.get_initial_recurrent_state()
    for _ in range(8):
        obs, h, _, _ = agent.collect_step(obs, h)

    before = [p.detach().clone() for p in agent._slow_critic.parameters()]
    agent.update_step()
    after = list(agent._slow_critic.parameters())

    # at least one param should have changed (slow EMA tracks fast critic)
    assert any((a - b).abs().sum().item() > 0 for a, b in zip(after, before))
