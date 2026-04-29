import torch
import torch.nn as nn


class CriticLoss(nn.Module):
    """Critic regression loss, including slow target regularization."""

    def __init__(
        self,
        beta_dream: float = 1.0,
        beta_replay: float = 0.3,
        slow_reg_weight: float = 0.3,
    ) -> None:
        """Construct a new critic loss function.

        Args:
            slow_coef (float): slow critic target regularization strength.
        """
        super().__init__()
        self._beta_dream = beta_dream
        self._beta_replay = beta_replay
        self._slow_reg_weight = slow_reg_weight

    def soft_cross_entropy(
        self,
        target_probs: torch.Tensor,
        input_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        # soft cross-entropy: -sum(target * log_probs) over bins, mean over batch
        loss = (-target_probs * input_log_probs).sum(-1).mean()
        return loss

    def forward(
        self,
        fast_log_probs: torch.Tensor,
        slow_log_probs: torch.Tensor,
        dream_target: torch.Tensor,
        replay_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute critic regression loss.

        Args:
            fast_log_probs  (B, T, D, bins): log-probs of fast critic values.
            slow_log_probs  (B, T, D, bins): log-probs of slow critic values.
            dream_target    (B, T, D, bins): symlog two-hot encoded dream lambda return targets.
            replay_target   (B, T, bins): symlog two-hot encoded replay lambda return targets.
        """
        regularizer_term = self.soft_cross_entropy(slow_log_probs.exp(), fast_log_probs)
        replay_loss = self.soft_cross_entropy(replay_target, fast_log_probs[:, :, 0])
        dream_loss = self.soft_cross_entropy(dream_target, fast_log_probs)
        loss = (
            self._beta_replay * replay_loss
            + self._beta_dream * dream_loss
            + self._slow_reg_weight * regularizer_term
        )
        metrics = {
            "critic/loss": loss.detach().item(),
            "critic/dream_loss": dream_loss.detach().item(),
            "critic/replay_loss": replay_loss.detach().item(),
            "critic/slow_reg": regularizer_term.detach().item(),
        }
        return loss, metrics
