from math import prod

import gymnasium as gym


def get_action_size(env: gym.Env) -> int:
    action_space = env.action_space
    assert action_space is not None
    if isinstance(action_space, gym.spaces.Discrete):
        return int(action_space.n)
    elif isinstance(action_space, gym.spaces.Box):
        return prod(action_space.shape)
    else:
        raise ValueError("Unsupported action space.")
