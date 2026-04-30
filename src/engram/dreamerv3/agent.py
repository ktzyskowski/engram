import copy
from typing import Any, Generator

import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.optim as optim

from engram.data.buffer import ReplayBuffer
from engram.dreamerv3.dream import dream_rollout
from engram.loss.actor import ActorLoss
from engram.loss.critic import CriticLoss
from engram.loss.world_model import WorldModelLoss
from engram.nets.mlp import MLP
from engram.rl.policy import sample_action
from engram.rl.returns import calc_lambda_returns
from engram.rl.rssm import RSSM
from engram.tools.checkpoint import CheckpointManager
from engram.tools.conditionals import Ratio
from engram.tools.gym import get_action_size
from engram.tools.two_hot import SymlogTwoHot


class DreamerV3:
    def __init__(
        self,
        env: gym.Env,
        params: dict[str, Any],
        device: str = "cuda",
        batch_size: int = 16,
        sequence_length: int = 64,
        prefill_steps: int = 1_024,
    ) -> None:
        self._env = env
        self._eval_env = gym.make(env.spec.id) if env.spec else copy.deepcopy(env)

        # TODO: hard-code to MLP encoder for now, change to CNN later
        observation_shape = env.observation_space.shape
        assert observation_shape is not None and len(observation_shape) == 1
        self._observation_size = observation_shape[0]
        self._action_size = get_action_size(env)

        self._train_ratio = Ratio(1.0)
        self._dream_horizon = 15
        self._device = device
        self._batch_size = batch_size
        self._sequence_length = sequence_length
        self._prefill_steps = prefill_steps

        # utilities --------------------------------------------------------- #

        self._two_hot = SymlogTwoHot(
            low=params["two_hot_low"],
            high=params["two_hot_high"],
            n_bins=params["two_hot_n_bins"],
        ).to(self._device)

        self._replay_buffer = ReplayBuffer(
            observation_shape=observation_shape,
            action_size=self._action_size,
            capacity=100_000,
            dtype="float32",
            action_dtype="float32",
        )
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
        ).to(self._device)
        self._rssm.compile()

        self._reward_head = MLP(
            input_size=self._rssm.full_state_size,
            hidden_sizes=params["reward_hidden_sizes"],
            output_size=params["two_hot_n_bins"],
            activation=params["reward_activation"],
            zero_output_weights=True,
        ).to(self._device)

        self._continue_head = MLP(
            input_size=self._rssm.full_state_size,
            hidden_sizes=params["continue_hidden_sizes"],
            output_size=1,
            activation=params["continue_activation"],
        ).to(self._device)

        # actor & critic ---------------------------------------------------- #

        self._actor = MLP(
            input_size=self._rssm.full_state_size,
            hidden_sizes=params["actor_hidden_sizes"],
            output_size=self._action_size,
            activation=params["actor_activation"],
        ).to(self._device)

        self._fast_critic = MLP(
            input_size=self._rssm.full_state_size,
            hidden_sizes=params["critic_hidden_sizes"],
            output_size=params["two_hot_n_bins"],
            activation=params["critic_activation"],
            zero_output_weights=True,
        ).to(self._device)

        self._slow_critic = copy.deepcopy(self._fast_critic)
        self._slow_critic.requires_grad_(False)

        # encoder & decoder ------------------------------------------------- #

        self._encoder = MLP(
            input_size=self._observation_size,
            hidden_sizes=params["encoder_hidden_sizes"],
            output_size=params["encoder_output_size"],
            activation=params["encoder_activation"],
        ).to(self._device)

        self._decoder = MLP(
            input_size=self._rssm.full_state_size,
            hidden_sizes=params["decoder_hidden_sizes"],
            output_size=self._observation_size,
            activation=params["decoder_activation"],
        ).to(self._device)

        # loss -------------------------------------------------------------- #

        self._world_model_loss_fn = WorldModelLoss(
            beta_posterior=0.1,
            beta_prior=0.5,
            beta_prediction=1.0,
            free_nats=1.0,
        ).to(self._device)

        self._actor_loss_fn = ActorLoss(
            eta=3e-4,
            advantage_ema_decay=0.99,
            percentile_high=0.95,
            percentile_low=0.05,
        ).to(self._device)

        self._critic_loss_fn = CriticLoss(
            slow_reg_weight=1.0,
        ).to(self._device)

        # torch ------------------------------------------------------------- #

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
        self._actor_optimizer = optim.Adam(
            self._actor.parameters(),
            lr=params["actor_lr"],
        )
        self._critic_optimizer = optim.Adam(
            self._fast_critic.parameters(),
            lr=params["critic_lr"],
        )
        self._critic_tau = params["critic_tau"]

    def train(self, steps: int) -> Generator[dict[str, int | float], Any, None]:
        """Train DreamerV3."""
        obs, _ = self._env.reset()
        obs = torch.from_numpy(obs).to(self._device)
        h = self._rssm.get_initial_recurrent_state().to(self._device)

        episode_return = 0.0
        episode_length = 0
        gradient_step = 0

        for step in range(steps):
            metrics: dict[str, int | float] = {"step": step}
            obs, h, reward, done = self.collect_step(obs, h)
            episode_return += reward
            episode_length += 1

            if (
                step >= self._prefill_steps
                and len(self._replay_buffer) >= self._sequence_length
            ):
                for _ in range(self._train_ratio()):
                    metrics |= self.update_step()
                    gradient_step += 1
            metrics["gradient_step"] = gradient_step

            if done:
                metrics |= {
                    "train/episode_length": episode_length,
                    "train/episode_return": episode_return,
                }
                episode_return = 0.0
                episode_length = 0

            yield metrics

    @torch.no_grad()
    def collect_step(
        self,
        obs: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float, bool]:
        """Perform a collection step."""
        encoded_obs = self._encoder(obs)
        z, _ = self._rssm.get_posterior(encoded_obs, h)
        action, _ = sample_action(self._actor(torch.cat([h, z], dim=-1)))
        h_next = self._rssm.step(h, z, action)
        next_obs, reward, terminated, truncated, _ = self._env.step(
            action.argmax().item()
        )
        reward = float(reward)
        done = truncated or terminated
        self._replay_buffer.add(obs, action, reward, done)
        if done:
            obs, _ = self._env.reset()
            obs = torch.from_numpy(obs).to(self._device)
            h_next = self._rssm.get_initial_recurrent_state().to(self._device)
        else:
            obs = torch.from_numpy(next_obs).to(self._device)
        return obs, h_next, reward, done

    def update_step(self) -> dict[str, int | float]:
        """Perform an update step."""
        batch = self._replay_buffer.sample_torch(
            batch_size=self._batch_size,
            sequence_length=self._sequence_length,
            device=self._device,
        )
        h, z, world_model_metrics = self.update_world_model(batch)
        agent_metrics = self.update_agent(batch, h, z)
        return {**world_model_metrics, **agent_metrics}

    def update_world_model(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Perform a single world model update step.

        The generated recurrent and stochastic state vectors are returned from
        this method, for use as initial seed states in dream rollouts during
        actor-critic training.

        Args:
            batch (dict[str, torch.Tensor]): dictionary containing batch tensors:
                - `observations`    (B, T, *O): tensor of observations.
                - `actions`         (B, T, A):  tensor of actions.
                - `rewards`         (B, T):     tensor of rewards.
                - `dones`           (B, T):     tensor of termination flags.
        Returns:
            h (B, T, H_h): generated recurrent states.
            z (B, T, H_z): generated stochastic (posterior) states.
        """
        self._world_model_optimizer.zero_grad()

        encoded_obs = self._encoder(batch["observations"])
        h, z, posterior_log_probs, prior_log_probs = self._rssm(
            encoded_obs, batch["actions"], batch["dones"]
        )
        full_states = torch.cat([h, z], dim=-1)
        reward_logits = self._reward_head(full_states)
        continue_logits = self._continue_head(full_states)
        reconstructed_obs = self._decoder(full_states)

        world_model_loss, metrics = self._world_model_loss_fn(
            obs=batch["observations"],
            dones=batch["dones"],
            target_reward_logits=self._two_hot.encode(batch["rewards"]),
            reconstructed_obs=reconstructed_obs,
            continue_logits=continue_logits,
            reward_logits=reward_logits,
            posterior_log_probs=posterior_log_probs,
            prior_log_probs=prior_log_probs,
        )

        world_model_loss.backward()
        self._world_model_optimizer.step()
        return h, z, metrics

    def update_agent(self, batch, h, z) -> dict[str, float]:
        self._actor_optimizer.zero_grad()
        self._critic_optimizer.zero_grad()

        # flatten (B, T) into a single batch dim of seeds for dream_rollout,
        # then unflatten the dream outputs back to (B, T, T_dream, ...).
        B, T = h.shape[:2]
        h_seed = h.detach().flatten(0, 1)
        z_seed = z.detach().flatten(0, 1)
        h, z, actions, action_log_probs = dream_rollout(
            rssm=self._rssm,
            actor=self._actor,
            h=h_seed,
            z=z_seed,
            horizon=self._dream_horizon,
        )
        h = h.unflatten(0, (B, T))
        z = z.unflatten(0, (B, T))
        actions = actions.unflatten(0, (B, T))
        action_log_probs = action_log_probs.unflatten(0, (B, T))

        full_states = torch.cat([h, z], dim=-1)
        reward_logits = self._reward_head(full_states)
        continue_logits = self._continue_head(full_states)
        fast_critic_logits = self._fast_critic(full_states)
        slow_critic_logits = self._slow_critic(full_states)
        slow_critic_values = self._two_hot.decode_logits(slow_critic_logits)

        # (B, T, T_dream)
        dream_lambda_returns = calc_lambda_returns(
            rewards=self._two_hot.decode_logits(reward_logits),
            continues=F.sigmoid(continue_logits).squeeze(-1),
            values=slow_critic_values,
        )
        # (B, T)
        replay_lambda_returns = calc_lambda_returns(
            rewards=batch["rewards"],
            continues=1 - batch["dones"],
            values=slow_critic_values[:, :, 0],
        )

        actor_loss, actor_metrics = self._actor_loss_fn(
            values=slow_critic_values,
            lambda_returns=dream_lambda_returns,
            actions=actions,
            action_log_probs=action_log_probs,
        )
        critic_loss, critic_metrics = self._critic_loss_fn(
            fast_log_probs=torch.log_softmax(fast_critic_logits, dim=-1),
            slow_log_probs=torch.log_softmax(slow_critic_logits, dim=-1),
            dream_target=self._two_hot.encode(dream_lambda_returns),
            replay_target=self._two_hot.encode(replay_lambda_returns),
        )

        # actor and critic share the dream-rollout graph, so combine into one
        # backward pass. each optimizer steps its own parameter group.
        (actor_loss + critic_loss).backward()
        self._actor_optimizer.step()
        self._critic_optimizer.step()

        # update slow critic with EMA of fast critic parameters
        for p_slow, p_fast in zip(
            self._slow_critic.parameters(),
            self._fast_critic.parameters(),
        ):
            p_slow.data.lerp_(p_fast.data, self._critic_tau)

        return {**actor_metrics, **critic_metrics}

    def _checkpoint_manager(self, directory: str = "checkpoints") -> CheckpointManager:
        return CheckpointManager(
            modules={
                "rssm": self._rssm,
                "encoder": self._encoder,
                "decoder": self._decoder,
                "reward_head": self._reward_head,
                "continue_head": self._continue_head,
                "actor": self._actor,
                "fast_critic": self._fast_critic,
                "slow_critic": self._slow_critic,
                "actor_loss": self._actor_loss_fn,  # has EMA buffers
                "world_model_optimizer": self._world_model_optimizer,
                "actor_optimizer": self._actor_optimizer,
                "critic_optimizer": self._critic_optimizer,
            },
            directory=directory,
        )

    def save(
        self,
        name: str,
        env_step: int = 0,
        gradient_step: int = 0,
        **extra: object,
    ) -> str:
        return self._checkpoint_manager().save(
            name, env_step=env_step, gradient_step=gradient_step, **extra
        )

    def load(self, path: str) -> dict[str, int]:
        return self._checkpoint_manager().load(path, device=self._device)

    @torch.no_grad()
    def eval(self, n_episodes: int = 100) -> dict[str, float]:
        """Evaluate the greedy policy over n_episodes.

        Switches to eval mode, runs deterministic argmax actions, then restores
        train mode. Returns mean and std of per-episode return and length.
        """
        was_training = self._actor.training
        for m in (self._encoder, self._rssm, self._actor):
            m.eval()

        returns: list[float] = []
        lengths: list[int] = []
        for _ in range(n_episodes):
            obs, _ = self._eval_env.reset()
            obs = torch.from_numpy(obs).to(self._device)
            h = self._rssm.get_initial_recurrent_state().to(self._device)
            ep_return = 0.0
            ep_length = 0
            done = False
            while not done:
                encoded_obs = self._encoder(obs)
                z, _ = self._rssm.get_posterior(encoded_obs, h)
                action_logits = self._actor(torch.cat([h, z], dim=-1))
                action_idx = int(action_logits.argmax(-1).item())
                action_one_hot = F.one_hot(
                    torch.tensor(action_idx, device=self._device),
                    num_classes=self._action_size,
                ).float()
                h = self._rssm.step(h, z, action_one_hot)
                next_obs, reward, terminated, truncated, _ = self._eval_env.step(
                    action_idx
                )
                ep_return += float(reward)
                ep_length += 1
                done = bool(terminated or truncated)
                obs = torch.from_numpy(next_obs).to(self._device)
            returns.append(ep_return)
            lengths.append(ep_length)

        if was_training:
            for m in (self._encoder, self._rssm, self._actor):
                m.train()

        returns_t = torch.tensor(returns, dtype=torch.float32)
        lengths_t = torch.tensor(lengths, dtype=torch.float32)
        return {
            "eval/return_mean": returns_t.mean().item(),
            "eval/return_std": returns_t.std(unbiased=False).item(),
            "eval/length_mean": lengths_t.mean().item(),
            "eval/length_std": lengths_t.std(unbiased=False).item(),
        }
