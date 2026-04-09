from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from agents.base_agent import BaseAgent, Transition
from agents.dqn_agent.features import (
    LEGAL_ACTION_DIM,
    LEGAL_ACTION_START,
    build_dqn_features,
)
from agents.dqn_agent.networks import QNetwork
from agents.dqn_agent.replay import EpisodeReplayBuffer, StepExperience
from env.state import State


def _detach_hidden(
    hidden: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    return hidden[0].detach(), hidden[1].detach()


class DoubleDQNAgent(BaseAgent):
    """Exploitative recurrent Double DQN agent with a dueling Q-head."""

    def __init__(
        self,
        *,
        lr: float = 1e-3,
        gamma: float = 1.0,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_hands: int = 20_000,
        replay_capacity: int = 50_000,
        batch_size: int = 32,
        warmup_hands: int = 200,
        target_update_every: int = 200,
        grad_clip: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_hands = max(1, epsilon_decay_hands)
        self.batch_size = batch_size
        self.warmup_hands = warmup_hands
        self.target_update_every = max(1, target_update_every)
        self.grad_clip = grad_clip

        self.q_network = QNetwork().to(device)
        self.target_network = QNetwork().to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay = EpisodeReplayBuffer(capacity=replay_capacity)

        self._game_hidden: tuple[torch.Tensor, torch.Tensor] | None = None
        self._current_episode: list[StepExperience] = []
        self.training_hands = 0
        self.training_steps = 0

    @property
    def epsilon(self) -> float:
        progress = min(self.training_hands / self.epsilon_decay_hands, 1.0)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def reset_hand_state(self) -> None:
        self._game_hidden = None
        self._current_episode = []

    def build_features(self, state: State) -> torch.Tensor:
        return build_dqn_features(state, self.device)

    def act(
        self,
        *,
        state: State,
        training: bool = True,
        action_record: list[tuple[int, str]] | None = None,
    ) -> int:
        del action_record
        if self._game_hidden is None:
            self._game_hidden = self.q_network.init_hidden(self.device)

        features = self.build_features(state).unsqueeze(0)
        with torch.no_grad():
            q_values, hidden_new = self.q_network(features, self._game_hidden)
        self._game_hidden = _detach_hidden(hidden_new)

        legal_actions = list(state.legal_actions.keys())
        if training and random.random() < self.epsilon:
            return int(random.choice(legal_actions))

        masked_q = q_values.squeeze(0).clone()
        illegal_mask = features[
            0, LEGAL_ACTION_START : LEGAL_ACTION_START + LEGAL_ACTION_DIM
        ] < 0.5
        masked_q[illegal_mask] = float("-inf")
        return int(masked_q.argmax().item())

    def observe(self, transition: Transition) -> None:
        self._current_episode.append(
            StepExperience(
                obs=np.asarray(transition["obs"], dtype=np.float32),
                action=transition["action"],
                reward=transition["reward"],
                next_obs=np.asarray(transition["next_obs"], dtype=np.float32),
                done=transition["done"],
            )
        )
        if transition["done"]:
            self.replay.add_episode(self._current_episode)
            self.training_hands += 1
            self.reset_hand_state()

    def update(self) -> None:
        if len(self.replay) < self.warmup_hands:
            return

        batch = self.replay.sample(self.batch_size)
        losses: list[torch.Tensor] = []

        for episode in batch:
            online_hidden = self.q_network.init_hidden(self.device)
            target_hidden = self.target_network.init_hidden(self.device)
            episode_losses: list[torch.Tensor] = []

            for step in episode:
                obs = (
                    torch.from_numpy(step.obs).float().to(self.device).unsqueeze(0)
                )
                next_obs = (
                    torch.from_numpy(step.next_obs).float().to(self.device).unsqueeze(0)
                )

                q_values, online_hidden_after = self.q_network(obs, online_hidden)
                q_selected = q_values[0, step.action]

                with torch.no_grad():
                    _, target_hidden_after = self.target_network(obs, target_hidden)

                    online_next_q, _ = self.q_network(
                        next_obs, _detach_hidden(online_hidden_after)
                    )
                    next_legal_mask = next_obs[
                        0, LEGAL_ACTION_START : LEGAL_ACTION_START + LEGAL_ACTION_DIM
                    ] > 0.5
                    next_online_masked = online_next_q.squeeze(0).clone()
                    next_online_masked[~next_legal_mask] = float("-inf")
                    next_action = int(next_online_masked.argmax().item())

                    target_next_q, _ = self.target_network(
                        next_obs, target_hidden_after
                    )
                    bootstrap = 0.0 if step.done else target_next_q[0, next_action].item()
                    target_value = step.reward + self.gamma * bootstrap
                    target_tensor = torch.tensor(
                        target_value,
                        dtype=torch.float32,
                        device=self.device,
                    )

                episode_losses.append(F.smooth_l1_loss(q_selected, target_tensor))
                online_hidden = online_hidden_after
                target_hidden = _detach_hidden(target_hidden_after)

            if episode_losses:
                losses.append(torch.stack(episode_losses).mean())

        if not losses:
            return

        loss = torch.stack(losses).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip)
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.target_update_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "training_hands": self.training_hands,
                "training_steps": self.training_steps,
            },
            tmp,
        )
        os.replace(tmp, path)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.training_hands = int(checkpoint.get("training_hands", 0))
        self.training_steps = int(checkpoint.get("training_steps", 0))
        self.reset_hand_state()
