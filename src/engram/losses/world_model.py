import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from engram.tools.two_hot import symlog


class WorldModelLoss(nn.Module):
    def __init__(
        self,
        beta_posterior: float = 0.1,
        beta_prior: float = 0.5,
        beta_prediction: float = 1.0,
        free_nats: float = 1.0,
    ) -> None:
        super().__init__()
        self._beta_posterior = beta_posterior
        self._beta_prior = beta_prior
        self._beta_prediction = beta_prediction
        self._free_nats = free_nats

    def _prediction_loss(
        self,
        obs: Tensor,
        dones: Tensor,
        target_reward_logits: Tensor,
        reconstructed_obs: Tensor,
        continue_logits: Tensor,
        reward_logits: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        obs_loss = F.mse_loss(symlog(obs), reconstructed_obs)
        continues = 1 - dones
        continue_loss = F.binary_cross_entropy_with_logits(
            continue_logits.squeeze(-1), continues
        )
        # cross_entropy expects class dim in position 1: (B, T, C) -> (B, C, T)
        reward_loss = F.cross_entropy(
            reward_logits.permute(0, 2, 1),
            target_reward_logits.permute(0, 2, 1),
        )
        return obs_loss, continue_loss, reward_loss

    def _kl_loss(self, input_log_probs: Tensor, target_log_probs: Tensor) -> Tensor:
        loss = F.kl_div(
            input_log_probs, target_log_probs, log_target=True, reduction="none"
        )
        loss = loss.sum(dim=(-1, -2))  # sum over categorical/class
        loss = loss.clamp(min=self._free_nats)  # clamp to free nats floor
        loss = loss.mean()  # average over batch/timestep
        return loss

    def forward(
        self,
        obs: Tensor,
        dones: Tensor,
        target_reward_logits: Tensor,
        reconstructed_obs: Tensor,
        continue_logits: Tensor,
        reward_logits: Tensor,
        posterior_log_probs: Tensor,
        prior_log_probs: Tensor,
    ) -> Tensor:
        """Compute world model loss.

        Args:
            obs                     (B, T, ...):    tensor of original observations.
            dones                   (B, T):         tensor of termination flags.
            target_reward_logits    (B, T, N_bins): tensor of rewards encoded in symlog two-hot fashion.
            reconstructed_obs       (B, T, ...):    tensor of reconstructed observations.
            continue_logits         (B, T, 1):      tensor of predicted continue logits.
            reward_logits           (B, T, N_bins): tensor of predicted reward logits.
            posterior_log_probs     (B, T, K, C):   tensor of posterior stochastic state log-probs.
            prior_log_probs         (B, T, K, C):   tensor of prior stochastic state log-probs.
        Returns:
            loss                    ():             world model loss value.
        """
        obs_loss, continue_loss, reward_loss = self._prediction_loss(
            obs,
            dones,
            target_reward_logits,
            reconstructed_obs,
            continue_logits,
            reward_logits,
        )
        prediction_loss = obs_loss + continue_loss + reward_loss

        posterior_loss = self._kl_loss(prior_log_probs.detach(), posterior_log_probs)
        prior_loss = self._kl_loss(prior_log_probs, posterior_log_probs.detach())

        loss = (
            self._beta_prediction * prediction_loss
            + self._beta_posterior * posterior_loss
            + self._beta_prior * prior_loss
        )
        return loss
