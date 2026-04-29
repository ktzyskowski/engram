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

    Each row of the input (h, z) seeds an independent dream trajectory; the
    generated dream states are stacked into a new dream dimension. Callers
    seeding from a (B, T, D) world-model rollout should flatten the (B, T)
    dims into a single batch dim before passing in, then unflatten afterward.

    At t=0, the latent states are from the posterior distribution, but for
    all t>0, the prior is used to step the world model state.

    Args:
        rssm                (RSSM):         recurrent state space model.
        actor               (MLP):          actor network.
        h                   (N, D_h):       initial recurrent states.
        z                   (N, D_z):       initial stochastic states.
        horizon             (int):          dream horizon.
    Returns:
        hs                  (N, T_dream, D_h):  dream recurrent states.
        zs                  (N, T_dream, D_z):  dream stochastic states.
        actions             (N, T_dream, A):    dream actions (one-hot).
        action_log_probs    (N, T_dream):       dream action log-probs.
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
    hs = torch.stack(hs, dim=1)
    zs = torch.stack(zs, dim=1)
    actions = torch.stack(actions, dim=1)
    action_log_probs = torch.stack(action_log_probs, dim=1)

    return hs, zs, actions, action_log_probs
