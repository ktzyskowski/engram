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
    ):
        super().__init__()
        self._observation_size = observation_size
        self._action_size = action_size
        self._recurrent_size = recurrent_size
        self._stochastic_size = n_categoricals * n_classes
        self._full_size = self._recurrent_size + self._stochastic_size
        self._n_categoricals = n_categoricals
        self._n_classes = n_classes

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

    def forward(
        self,
        observations: Tensor,
        actions: Tensor,
        dones: Tensor,
    ) -> dict[str, Tensor]:
        """Forward pass through the RSSM.

        The model generates full model states (recurrent + stochastic) for downstream
        predictors, recurrent states for seeding dream rollouts, and posterior/prior
        logits for loss calculation.

        Args:
            observations        (B, T, O):      tensor of encoded observations.
            actions             (B, T, A):      tensor of one-hot actions.
            dones               (B, T):         tensor of episode termination flags.
        Returns:
            full_states         (B, T, H_full): tensor of full model states (h, z).
            recurrent_states    (B, T, H_rec):  tensor of recurrent states (h).
            posterior_logits    (B, T, K, C):   tensor of posterior logits (categorical, class).
            prior_logits        (B, T, K, C):   tensor of prior logits (categorical, class).
        """
        B = observations.shape[0]  # batch size
        T = observations.shape[1]  # sequence length
        device = observations.device

        output = {
            "full_states": [],
            "recurrent_states": [],
            "posterior_logits": [],
        }

        h = torch.zeros((B, self._recurrent_size), device=device)
        for t in range(T):
            # ----------------------------------------------------------------------------------- #
            # get posterior logits from posterior net, mix with uniform distribution
            z_logits = self._posterior_net(torch.cat([h, observations[:, t]], dim=-1))
            z_logits = z_logits.unflatten(-1, (self._n_categoricals, self._n_classes))
            z_logits = unimix(z_logits, frac=0.01)
            # use gumbel softmax for straight-through gradients with onehot argmax sampling
            z = F.gumbel_softmax(z_logits, tau=1.0, hard=True)
            # flatten (n_categoricals, n_classes) into single feature dimension
            z = z.flatten(start_dim=-2)
            # ----------------------------------------------------------------------------------- #
            # get next recurrent state from stochastic state and prior recurrent state
            h_next = self._recurrent_net(torch.cat([z, actions[:, t]], dim=-1), h)
            # ----------------------------------------------------------------------------------- #
            output["full_states"].append(torch.cat([h, z], dim=-1))
            output["recurrent_states"].append(h)
            output["posterior_logits"].append(z_logits)
            # ----------------------------------------------------------------------------------- #
            # reset recurrent state if episode terminates mid-sequence
            not_done = (~dones[:, t].bool()).unsqueeze(-1)
            h = h_next * not_done

        # stack accumulated tensors in sequence dim
        output = {k: torch.stack(tensors, dim=1) for k, tensors in output.items()}

        # prior not used during loop, so we can batch compute prior logits
        prior_logits = self._prior_net(output["recurrent_states"])
        prior_logits = prior_logits.unflatten(
            -1, (self._n_categoricals, self._n_classes)
        )
        prior_logits = unimix(prior_logits, frac=0.01)
        output["prior_logits"] = prior_logits

        return output
