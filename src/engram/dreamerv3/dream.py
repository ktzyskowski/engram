import torch
import torch.nn as nn

from engram.rl.policy import sample_action
from engram.rl.rssm import RSSM


@torch.compile()
def dream_rollout(
    rssm: RSSM,
    actor: nn.Module,
    h: torch.Tensor,
    z: torch.Tensor,
    horizon: int = 15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate dream trajectories from seed states.

    Dream trajectories are generated from each timestep in the given
    (h, z) tensors, which equates to B x T total dreams. The generated
    dream (h, z) state tensors are stacked into a new dream dimension.

    At t=0, the latent states are from the posterior distribution,
    but for all t>0, the prior is used to step the world model state.

    Args:
        rssm:               (RSSM): recurrent state space model.
        actor               (MLP): actor network.
        h:                  (B, T, D_h): initial recurrent states.
        z:                  (B, T, D_z): initial stochastic states.
        horizon             (float): dream horizon.
    Returns:
        hs                  (B, T, T_dream, D_h): dream recurrent states.
        zs                  (B, T, T_dream, D_z): dream stochastic states.
        actions             (B, T, T_dream, A): dream actions (one-hot).
        action_log_probs    (B, T, T_dream, A): dream action log-probs.
    """
    a, log_prob = sample_action(actor(torch.cat([h, z], dim=-1)))

    hs = [h]
    zs = [z]
    actions = [a]
    action_log_probs = [log_prob]

    for _ in range(horizon):
        h = rssm.step(h, z, a)
        z = rssm.get_prior(h)
        a, log_probs = sample_action(actor(torch.cat([h, z], dim=-1)))

        actions.append(a)
        action_log_probs.append(log_probs)
        hs.append(h)
        zs.append(z)

    # stack accumulated tensors in dream sequence dimension
    hs = torch.stack(hs, dim=2)
    zs = torch.stack(zs, dim=2)
    actions = torch.stack(actions, dim=2)
    action_log_probs = torch.stack(action_log_probs, dim=2)

    return hs, zs, actions, action_log_probs
