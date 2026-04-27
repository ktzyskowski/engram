import torch

from engram.rl.returns import calc_lambda_returns


def test_last_timestep_equals_value():
    rewards = torch.tensor([1.0, 2.0, 3.0])
    continues = torch.tensor([1.0, 1.0, 1.0])
    values = torch.tensor([10.0, 20.0, 30.0])

    returns = calc_lambda_returns(rewards, continues, values, discount=0.9, decay=0.5)

    assert returns[-1].item() == 30.0


def test_td0_when_decay_is_zero():
    # decay=0 -> target is just values[t+1] (TD(0))
    rewards = torch.tensor([1.0, 2.0, 3.0])
    continues = torch.tensor([1.0, 1.0, 1.0])
    values = torch.tensor([10.0, 20.0, 30.0])
    discount = 0.9

    returns = calc_lambda_returns(rewards, continues, values, discount=discount, decay=0.0)

    # returns[t] = rewards[t] + gamma * values[t+1]
    assert torch.allclose(returns[0], torch.tensor(1.0 + discount * 20.0))
    assert torch.allclose(returns[1], torch.tensor(2.0 + discount * 30.0))


def test_monte_carlo_when_decay_is_one():
    # decay=1 -> target is returns[t+1] (full Monte Carlo bootstrapped from values[T])
    rewards = torch.tensor([1.0, 2.0, 3.0])
    continues = torch.tensor([1.0, 1.0, 1.0])
    values = torch.tensor([0.0, 0.0, 5.0])
    discount = 0.5

    returns = calc_lambda_returns(rewards, continues, values, discount=discount, decay=1.0)

    # returns[2] = 5
    # returns[1] = 2 + 0.5 * 5 = 4.5
    # returns[0] = 1 + 0.5 * 4.5 = 3.25
    assert torch.allclose(returns, torch.tensor([3.25, 4.5, 5.0]))


def test_continues_zero_terminates_bootstrap():
    # if continues[t] = 0, future is masked: returns[t] = rewards[t]
    rewards = torch.tensor([1.0, 2.0, 3.0])
    continues = torch.tensor([0.0, 0.0, 0.0])
    values = torch.tensor([10.0, 20.0, 30.0])

    returns = calc_lambda_returns(rewards, continues, values, discount=0.9, decay=0.5)

    assert torch.allclose(returns[0], torch.tensor(1.0))
    assert torch.allclose(returns[1], torch.tensor(2.0))
    assert torch.allclose(returns[-1], torch.tensor(30.0))


def test_batched_shape_preserved():
    rewards = torch.randn(4, 8)
    continues = torch.ones(4, 8)
    values = torch.randn(4, 8)

    returns = calc_lambda_returns(rewards, continues, values, discount=0.99, decay=0.95)

    assert returns.shape == rewards.shape
