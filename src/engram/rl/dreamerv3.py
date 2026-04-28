import copy
import logging
from typing import Any

import gymnasium as gym
import torch
from torch import optim
import torch.nn.functional as F
from torch import Tensor

from engram.data.buffer import ReplayBuffer
from engram.losses.world_model import WorldModelLoss
from engram.nets.mlp import MLP
from engram.rl.rssm import RSSM
from engram.tools.conditionals import Ratio
from engram.tools.probability import unimix
from engram.tools.two_hot import SymlogTwoHot


class DreamerV3:
    def __init__(
        self,
        env: gym.Env,
        params: dict[str, Any],
        device: str = "cuda",
    ) -> None:
        self._env = env

        # TODO: hard-code to MLP encoder for now, change to CNN later
        observation_shape = env.observation_space.shape
        assert observation_shape is not None and len(observation_shape) == 1
        action_shape = env.action_space.shape
        assert action_shape is not None and len(action_shape) == 1
        self._observation_size = observation_shape[0]
        self._action_size = action_shape[0]

        self._train_ratio = Ratio(1.0)

        # world model ------------------------------------------------------- #
        self._rssm = RSSM(
            n_categoricals=params["n_categoricals"],
            n_classes=params["n_classes"],
            observation_size=params["encoder_output_size"],
            action_size=self._action_size,
            recurrent_size=params["recurrent_size"],
            posterior_hidden_sizes=params["posterior_hidden_sizes"],
            prior_hidden_sizes=params["prior_hidden_sizes"],
            posterior_activation=params["posterior_activation"],
            prior_activation=params["prior_activation"],
            unimix=params["unimix"],
        )
        self._reward_head = MLP(
            input_size=self._rssm.full_state_size,
            hidden_sizes=params["reward_hidden_sizes"],
            output_size=params["two_hot_n_bins"],
            activation=params["reward_activation"],
            zero_output_weights=True,
        )
        self._continue_head = MLP(
            input_size=self._rssm.full_state_size,
            hidden_sizes=params["continue_hidden_sizes"],
            output_size=1,
            activation=params["continue_activation"],
        )

        # actor & critic ---------------------------------------------------- #
        self._actor = MLP(
            input_size=self._rssm.full_state_size,
            hidden_sizes=params["actor_hidden_sizes"],
            output_size=self._action_size,
            activation=params["actor_activation"],
        )
        self._fast_critic = MLP(
            input_size=self._rssm.full_state_size,
            hidden_sizes=params["critic_hidden_sizes"],
            output_size=params["two_hot_n_bins"],
            activation=params["critic_activation"],
            zero_output_weights=True,
        )
        self._slow_critic = copy.deepcopy(self._fast_critic)
        self._slow_critic.requires_grad_(False)

        # encoder & decoder ------------------------------------------------- #
        self._encoder = MLP(
            input_size=self._observation_size,
            hidden_sizes=params["encoder_hidden_sizes"],
            output_size=params["encoder_output_size"],
            activation=params["encoder_activation"],
        )
        self._decoder = MLP(
            input_size=self._rssm.full_state_size,
            hidden_sizes=params["decoder_hidden_sizes"],
            output_size=params["observation_size"],
            activation=params["decoder_activation"],
        )

        # utilities --------------------------------------------------------- #
        self._two_hot = SymlogTwoHot(
            low=params["two_hot_low"],
            high=params["two_hot_high"],
            n_bins=params["two_hot_n_bins"],
        )
        self._replay_buffer = ReplayBuffer(
            observation_shape=observation_shape,
            action_size=self._action_size,
            capacity=100_000,
            dtype="float32",
            action_dtype="uint8",
        )

        # loss -------------------------------------------------------------- #

        self._world_model_loss_fn = WorldModelLoss(
            beta_posterior=0.1,
            beta_prior=0.5,
            beta_prediction=1.0,
            free_nats=1.0,
        )

        # torch ------------------------------------------------------------- #
        self._device = device
        self._rssm.compile()

        self._world_model_optimizer = optim.Adam(
            [
                *self._encoder.parameters(),
                *self._decoder.parameters(),
                *self._rssm.parameters(),
                *self._continue_head.parameters(),
                *self._reward_head.parameters(),
            ],
            lr=params["world_model_lr"],
        )
        self.actor_optimizer = optim.Adam(
            self._actor.parameters(),
            lr=params["actor_lr"],
        )
        self.critic_optimizer = optim.Adam(
            self._fast_critic.parameters(),
            lr=params["critic_lr"],
        )

    def train(self, steps: int) -> None:
        obs, _ = self._env.reset()
        recurrent_state = self._rssm.get_initial_recurrent_state().to(self._device)
        for _ in range(steps):
            obs, recurrent_state = self.collect_step(obs, recurrent_state)
            for _ in range(self._train_ratio()):
                self.update_step()

    @torch.no_grad()
    def collect_step(
        self, obs: Tensor, recurrent_state: Tensor
    ) -> tuple[Any | Tensor, Tensor]:
        encoded_obs = self._encoder(obs)
        stochastic_state, _ = self._rssm.get_posterior(encoded_obs, recurrent_state)
        full_state = torch.cat([recurrent_state, stochastic_state])
        action_logits = self._actor(full_state)
        action_logits = unimix(action_logits, frac=0.01)
        action = torch.distributions.OneHotCategorical(action_logits).sample()
        recurrent_state = self._rssm.step(stochastic_state, recurrent_state, action)
        # ------------------------------------------------------------------- #
        next_obs, reward, terminated, truncated, _ = self._env.step(
            action.argmax().item()
        )
        done = truncated or terminated
        self._replay_buffer.add(obs, action, float(reward), done)
        if done:
            obs, _ = self._env.reset()
        else:
            obs = next_obs
        return obs, recurrent_state

    def update_step(self) -> None:
        batch = self._replay_buffer.sample_torch(
            batch_size=32, sequence_length=64, device=self._device
        )

        # world model forward pass ------------------------------------------ #
        self._world_model_optimizer.zero_grad()
        encoded_obs = self._encoder(batch["observations"])
        rssm_output = self._rssm(encoded_obs, batch["actions"], batch["dones"])
        reward_logits = self._reward_head(rssm_output["full_states"])
        continue_logits = self._continue_head(rssm_output["full_states"])
        reconstructed_obs = self._decoder(rssm_output["full_states"])
        world_model_loss = self._world_model_loss_fn(
            obs=batch["observations"],
            dones=batch["dones"],
            target_reward_logits=self._two_hot.encode(batch["rewards"]),
            reconstructed_obs=reconstructed_obs,
            continue_logits=continue_logits,
            reward_logits=reward_logits,
        )
        world_model_loss.backward()
        self._world_model_optimizer.step()

    def save(self, path) -> None:
        pass

    def load(self, path) -> None:
        pass
