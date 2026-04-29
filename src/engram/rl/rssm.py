import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from engram.nets.mlp import MLP
from engram.tools.probability import unimix


class RSSM(nn.Module):
    """Recurrent State Space Model."""

    def __init__(
        self,
        n_categoricals: int,
        n_classes: int,
        observation_size: int,
        action_size: int,
        recurrent_size: int,
        posterior_hidden_sizes: list[int],
        prior_hidden_sizes: list[int],
        posterior_activation: str = "rmsnorm+silu",
        prior_activation: str = "rmsnorm+silu",
        unimix: float = 0.01,
    ) -> None:
        super().__init__()
        self._observation_size = observation_size
        self._action_size = action_size
        self._recurrent_size = recurrent_size
        self._stochastic_size = n_categoricals * n_classes
        self._n_categoricals = n_categoricals
        self._n_classes = n_classes
        self._unimix = unimix

        # encoder (sans perception)
        # z ~ p(z|h,x)
        self._posterior_net = MLP(
            input_size=self._recurrent_size + self._observation_size,
            hidden_sizes=posterior_hidden_sizes,
            output_size=self._stochastic_size,
            activation=posterior_activation,
        )

        # dynamics predictor
        # z ~ p(z|h)
        self._prior_net = MLP(
            input_size=self._recurrent_size,
            hidden_sizes=prior_hidden_sizes,
            output_size=self._stochastic_size,
            activation=prior_activation,
        )

        # sequence model
        # h' = f(h, z, a)
        self._recurrent_net = nn.GRUCell(
            input_size=self._stochastic_size + self._action_size,
            hidden_size=self._recurrent_size,
        )

    @property
    def full_state_size(self) -> int:
        """Get full model state size.

        Equivalent to: (n_categoricals x n_classes) + recurrent_size
        """
        return self._recurrent_size + self._stochastic_size

    def get_initial_recurrent_state(self) -> Tensor:
        """Get the initial recurrent state vector (zeros), with no batch dimension.

        Returns:
            h (D_h): initial recurrent state vector.
        """
        h = torch.zeros(self._recurrent_size)
        return h

    def get_stochastic_state(self, logits: Tensor) -> tuple[Tensor, Tensor]:
        logits = logits.unflatten(-1, (self._n_categoricals, self._n_classes))
        log_probs = unimix(logits, frac=self._unimix)
        z = F.gumbel_softmax(log_probs, tau=1.0, hard=True).flatten(start_dim=-2)
        return z, log_probs

    def get_posterior(self, observation: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        logits = self._posterior_net(torch.cat([h, observation], dim=-1))
        z, log_probs = self.get_stochastic_state(logits)
        return z, log_probs

    def get_prior(self, h: Tensor) -> Tensor:
        logits = self._prior_net(h)
        z, _ = self.get_stochastic_state(logits)
        return z

    def step(self, h: Tensor, z: Tensor, action: Tensor) -> Tensor:
        """Step the recurrent sequence model.

        Args:
            h       (*, D_h): recurrent state.
            z       (*, D_z): stochastic state.
            action  (*, A): one-hot action tensor.
        Returns:
            h_next  (*, D_h): next recurrent state.
        """
        h_next = self._recurrent_net(torch.cat([z, action], dim=-1), h)
        return h_next

    def forward(
        self,
        observations: Tensor,
        actions: Tensor,
        dones: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass through the RSSM.

        The model generates full model states (recurrent + stochastic) for downstream
        predictors, recurrent states for seeding dream rollouts, and posterior/prior
        log-probabilities for loss calculation.

        Args:
            observations        (B, T, O):      tensor of encoded observations.
            actions             (B, T, A):      tensor of one-hot actions.
            dones               (B, T):         tensor of episode termination flags.
        Returns:
            full_states         (B, T, H_full): tensor of full model states (h, z).
            recurrent_states    (B, T, H_rec):  tensor of recurrent states (h).
            posterior_log_probs (B, T, K, C):   tensor of posterior state log-probs (categorical, class).
            prior_log_probs     (B, T, K, C):   tensor of prior state log-probs (categorical, class).
        """
        B = observations.shape[0]  # batch size
        T = observations.shape[1]  # sequence length
        device = observations.device

        recurrent_states = []
        stochastic_states = []
        posterior_log_probs = []

        h = torch.zeros((B, self._recurrent_size), device=device)
        for t in range(T):
            z, z_log_probs = self.get_posterior(observations[:, t], h)
            recurrent_states.append(h)
            stochastic_states.append(z)
            posterior_log_probs.append(z_log_probs)
            # reset recurrent state if episode terminates mid-sequence
            not_done = (~dones[:, t].bool()).unsqueeze(-1)
            h_next = self.step(h, z, actions[:, t])
            h = h_next * not_done

        # stack accumulated tensors in sequence dim
        recurrent_states = torch.stack(recurrent_states, dim=1)
        stochastic_states = torch.stack(stochastic_states, dim=1)
        posterior_log_probs = torch.stack(posterior_log_probs, dim=1)

        # prior not used during loop, so we can batch compute prior logits
        prior_logits = self._prior_net(recurrent_states)
        prior_logits = prior_logits.unflatten(
            -1, (self._n_categoricals, self._n_classes)
        )
        prior_log_probs = unimix(prior_logits, frac=self._unimix)

        return recurrent_states, stochastic_states, posterior_log_probs, prior_log_probs
