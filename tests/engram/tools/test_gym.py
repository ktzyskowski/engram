import gymnasium as gym
import pytest

from engram.tools.gym import get_action_size


def test_discrete_returns_n():
    env = gym.make("CartPole-v1")
    assert get_action_size(env) == 2


def test_box_returns_product_of_shape():
    env = gym.make("Pendulum-v1")
    # Pendulum action_space is Box(shape=(1,))
    assert get_action_size(env) == 1


def test_box_multidim_returns_product():
    space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2, 3))
    env = gym.Env()
    env.action_space = space
    assert get_action_size(env) == 6


def test_unsupported_space_raises():
    space = gym.spaces.Tuple((gym.spaces.Discrete(2), gym.spaces.Discrete(3)))
    env = gym.Env()
    env.action_space = space
    with pytest.raises(ValueError):
        get_action_size(env)
