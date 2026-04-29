import torch
import torch.nn as nn

from engram.tools.ema import EMA


class ActorLoss(nn.Module):
    """Actor loss function.

    Implements REINFORCE loss with entropy regularization and a clamped
    exponential moving average over the advantage, as described in DreamerV3.
    """

    def __init__(
        self,
        eta: float,
        advantage_ema_decay: float,
        percentile_high: float = 0.95,
        percentile_low: float = 0.05,
    ) -> None:
        """Construct a new actor loss function.

        Args:
            eta (float): _description_
            advantage_ema_decay (float): _description_
            percentile_high (float, optional): _description_. Defaults to 0.95.
            percentile_low (float, optional): _description_. Defaults to 0.05.
        """
        super().__init__()
        self._eta = eta
        self._advantage_ema = EMA(advantage_ema_decay)
        self._percentile_high = percentile_high
        self._percentile_low = percentile_low

    def forward(
        self,
        values: torch.Tensor,
        lambda_returns: torch.Tensor,
        actions: torch.Tensor,
        action_log_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute actor loss.

        Args:
            values              (B, T, D): slow-critic value estimates.
            lambda_returns      (B, T, D): lambda-return targets.
            actions             (B, T, D, A): one-hot sampled actions.
            action_log_probs    (B, T, D, A): full action log-probs.
        """
        # advantage normalization (clamped EMA of return percentile spread)
        flat_returns = lambda_returns.flatten()
        high = torch.quantile(flat_returns, q=self._percentile_high)
        low = torch.quantile(flat_returns, q=self._percentile_low)
        spread = high - low
        norm = self._advantage_ema(spread).clamp(min=1.0)
        advantage = ((lambda_returns - values) / norm).detach()

        # REINFORCE term
        log_prob = (actions * action_log_probs).sum(dim=-1)
        reinforce_term = (advantage * log_prob).mean()

        # entropy regularization
        entropy = -(action_log_probs.exp() * action_log_probs).sum(dim=-1).mean()

        loss = -(reinforce_term + self._eta * entropy)
        metrics = {
            "actor/loss": loss.detach().item(),
            "actor/reinforce": reinforce_term.detach().item(),
            "actor/entropy": entropy.detach().item(),
            "actor/advantage_abs_mean": advantage.abs().mean().item(),
            "actor/advantage_norm": norm.detach().item(),
            "actor/return_spread": spread.detach().item(),
            "actor/return_percentile_high": high.detach().item(),
            "actor/return_percentile_low": low.detach().item(),
        }
        return loss, metrics
