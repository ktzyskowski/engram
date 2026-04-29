import torch

from engram.tools.probability import unimix


def sample_action(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample an action from the given policy logits.

    Args:
        logits              (*, D):
    Returns:
        action              (*, D):
        action_log_prob     (*, D):
    """
    log_probs = unimix(logits, frac=0.01)
    dist = torch.distributions.OneHotCategorical(logits=log_probs)
    action = dist.sample()
    action_log_prob = dist.log_prob(action)
    return action, action_log_prob
