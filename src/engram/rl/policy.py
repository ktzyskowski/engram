import torch

from engram.tools.probability import unimix


def sample_action(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample an action from the given policy logits.

    Returns the full categorical log-probs (not just the sampled action's
    log-prob) so the actor loss can compute entropy and pick out the taken
    action's log-prob via `(action * log_probs).sum(-1)`.

    Args:
        logits              (*, A):
    Returns:
        action              (*, A): one-hot sampled action.
        action_log_probs    (*, A): full categorical log-probs.
    """
    log_probs = unimix(logits, frac=0.01)
    action = torch.distributions.OneHotCategorical(logits=log_probs).sample()
    return action, log_probs
