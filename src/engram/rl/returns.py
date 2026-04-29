import torch
from torch import Tensor


def calc_lambda_returns(
    rewards: Tensor,
    continues: Tensor,
    values: Tensor,
    discount: float = 0.99,
    decay: float = 0.95,
) -> Tensor:
    """Calculate lambda returns for a given sequence of observed rewards and critic values.

    Args:
        rewards     (*): tensor of observed rewards.
        continues   (*): tensor of continue flags.
        values      (*): tensor of critic values.
        discount    (float):  discount factor (gamma).
        decay       (float):  trace decay factor (lambda).
    Returns:
        returns     (*): tensor of lambda returns.
    """
    returns = torch.zeros_like(values)

    # last timestep in sequence
    T = rewards.shape[-1] - 1

    # the return at time T is equal to the value at time T (bootstrapped)
    returns[..., T] = values[..., T]

    for t in reversed(range(T)):
        # future return is a linear interpolation between critic values and monte carlo return
        next_return = (1 - decay) * values[..., t + 1] + decay * returns[..., t + 1]
        returns[..., t] = rewards[..., t] + discount * continues[..., t] * next_return

    return returns
