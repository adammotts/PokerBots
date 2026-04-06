from __future__ import annotations

import os

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from agents.ac_agent.features import build_features, build_opponent_summary
from agents.ac_agent.networks import (
    ActorNetwork,
    ConfidenceGate,
    CriticNetwork,
    OpponentLSTM,
)
from agents.base_agent import BaseAgent, Transition
from env.state import State


class ActorCriticAgent(BaseAgent):
    """Adaptive Actor-Critic agent with opponent modeling.

    Uses a dual-LSTM architecture:
    - Game LSTMs (inside actor/critic): capture within-hand action sequences.
      Reset every hand.
    - Opponent LSTM (shared): captures cross-hand opponent patterns.
      Persists across hands within a session.

    Optionally applies adaptive KL regularization toward a CFR baseline,
    gated by the opponent model's learned confidence.
    """

    def __init__(
        self,
        *,
        lambda_kl_max: float = 0.0,
        lr: float = 3e-4,
        gamma: float = 1.0,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        grad_clip: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.lambda_kl_max = lambda_kl_max
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.grad_clip = grad_clip

        # Networks
        self.actor = ActorNetwork().to(device)
        self.critic = CriticNetwork().to(device)
        self.opponent_lstm = OpponentLSTM().to(device)
        self.confidence_gate: ConfidenceGate | None = None
        if lambda_kl_max > 0:
            self.confidence_gate = ConfidenceGate().to(device)

        # Single optimizer for all parameters
        params: list[nn.Parameter] = [
            *self.actor.parameters(),
            *self.critic.parameters(),
            *self.opponent_lstm.parameters(),
        ]
        if self.confidence_gate is not None:
            params.extend(self.confidence_gate.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

        # Per-session state (persists across hands within a session)
        self._opp_hidden = self.opponent_lstm.init_hidden(device)

        # Per-hand state (reset each hand via _reset_hand)
        self._game_hidden_actor: tuple[torch.Tensor, torch.Tensor] | None = None
        self._game_hidden_critic: tuple[torch.Tensor, torch.Tensor] | None = None
        self._trajectory: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []
        self._hand_reward: float = 0.0
        self._action_record: list[tuple[int, str]] = []
        self._our_player_id: int = 0

        # CFR agent for KL target (lazy-loaded)
        self._cfr_agent: object | None = None

    # ── Session management ───────────────────────────────────────────

    def reset_opponent_state(self) -> None:
        """Reset opponent model. Call at the start of each new session."""
        self._opp_hidden = self.opponent_lstm.init_hidden(self.device)

    def _reset_hand(self) -> None:
        """Reset per-hand state for a new hand."""
        self._game_hidden_actor = self.actor.init_game_hidden(self.device)
        self._game_hidden_critic = self.critic.init_game_hidden(self.device)
        self._trajectory.clear()
        self._action_record = []

    def _get_opp_context(self) -> torch.Tensor:
        """Get opponent context vector (detached) for actor/critic input."""
        # h component of hidden state: (1, 1, 32) -> (1, 32)
        return self._opp_hidden[0].detach().squeeze(0)

    # ── BaseAgent interface ──────────────────────────────────────────

    def act(
        self,
        *,
        state: State,
        training: bool = True,
        action_record: list[tuple[int, str]] | None = None,
    ) -> int:
        if self._game_hidden_actor is None:
            self._reset_hand()
            self._our_player_id = state.player_id

        if action_record is not None:
            self._action_record = action_record

        features = build_features(state, self.device).unsqueeze(0)  # (1, 77)
        opp_context = self._get_opp_context()  # (1, 32)

        with torch.set_grad_enabled(training):
            logits, self._game_hidden_actor = self.actor(
                features, self._game_hidden_actor, opp_context
            )
            value, self._game_hidden_critic = self.critic(
                features, self._game_hidden_critic, opp_context
            )

        # Mask illegal actions
        mask = torch.full((4,), float("-inf"), device=self.device)
        for a in state.legal_actions:
            mask[a] = 0.0
        masked_logits = logits.squeeze(0) + mask

        dist = Categorical(logits=masked_logits)

        if training:
            action = dist.sample()
            self._trajectory.append(
                (
                    features.detach(),
                    dist.log_prob(action),
                    value.squeeze(),
                    dist.entropy(),
                )
            )
        else:
            action = masked_logits.argmax()

        return int(action.item())

    def act_from_state(self, state: State, *, training: bool = False) -> int:
        """Convenience method for acting directly from a State object."""
        return self.act(
            state.obs,
            list(state.legal_actions.keys()),
            training=training,
            player_id=state.player_id,
        )

    def observe(self, transition: Transition) -> None:
        if transition["done"]:
            self._hand_reward = transition["reward"]

    def update(self) -> None:
        """Compute A2C loss, backprop, and step the opponent LSTM."""
        if not self._trajectory:
            self._game_hidden_actor = None
            self._game_hidden_critic = None
            return

        # Unpack trajectory
        features_list, log_probs, values, entropies = zip(
            *self._trajectory, strict=True
        )
        log_probs_t = torch.stack(log_probs)
        values_t = torch.stack(values)
        entropies_t = torch.stack(entropies)

        # Returns: R for all steps (gamma=1.0, reward only at terminal)
        returns_t = torch.full_like(values_t, self._hand_reward, device=self.device)

        # Advantages
        advantages = (returns_t - values_t).detach()

        # Policy loss (REINFORCE with baseline)
        policy_loss = -(log_probs_t * advantages).mean()

        # Value loss
        value_loss = F.mse_loss(values_t, returns_t)

        # Entropy bonus
        entropy_loss = -entropies_t.mean()

        # Total loss
        total_loss = (
            policy_loss
            + self.value_coef * value_loss
            + self.entropy_coef * entropy_loss
        )

        # KL regularization toward CFR (if enabled)
        if self.lambda_kl_max > 0 and self.confidence_gate is not None:
            kl_loss = self._compute_kl_loss(features_list)
            if kl_loss is not None:
                confidence = self.confidence_gate(self._opp_hidden[0].detach())
                lambda_kl = self.lambda_kl_max * (1.0 - confidence.squeeze())
                total_loss = total_loss + lambda_kl * kl_loss

        # Backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        all_params = (
            list(self.actor.parameters())
            + list(self.critic.parameters())
            + list(self.opponent_lstm.parameters())
        )
        if self.confidence_gate is not None:
            all_params.extend(self.confidence_gate.parameters())
        nn.utils.clip_grad_norm_(all_params, self.grad_clip)
        self.optimizer.step()

        # Step opponent LSTM with hand summary
        self._step_opponent_lstm()

        # Reset per-hand state
        self._game_hidden_actor = None
        self._game_hidden_critic = None
        self._trajectory.clear()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state_dict: dict[str, object] = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "opponent_lstm": self.opponent_lstm.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.confidence_gate is not None:
            state_dict["confidence_gate"] = self.confidence_gate.state_dict()
        tmp = path + ".tmp"
        torch.save(state_dict, tmp)
        os.replace(tmp, path)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.opponent_lstm.load_state_dict(checkpoint["opponent_lstm"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.confidence_gate is not None and "confidence_gate" in checkpoint:
            self.confidence_gate.load_state_dict(checkpoint["confidence_gate"])

    # ── Internal helpers ─────────────────────────────────────────────

    def _step_opponent_lstm(self) -> None:
        """Feed hand summary through opponent LSTM after a hand ends."""
        went_to_showdown = not any(
            action_str == "fold" for _, action_str in self._action_record
        )
        summary = build_opponent_summary(
            self._action_record,
            self._our_player_id,
            self._hand_reward,
            went_to_showdown,
            self.device,
        )
        # (8,) -> (1, 1, 8)
        summary_input = summary.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            _, self._opp_hidden = self.opponent_lstm(summary_input, self._opp_hidden)

    def _compute_kl_loss(
        self,
        features_list: tuple[torch.Tensor, ...],
    ) -> torch.Tensor | None:
        """Compute KL divergence between agent policy and CFR policy.

        Returns None if CFR agent is not available.
        """
        if self._cfr_agent is None:
            return None

        kl_values = []
        opp_context = self._get_opp_context()

        for features in features_list:
            with torch.no_grad():
                logits, _ = self.actor(
                    features,
                    self.actor.init_game_hidden(self.device),
                    opp_context,
                )
            # Get CFR action probs for this state
            obs_np = features[0, :72].cpu().numpy()
            legal_actions = [i for i in range(4) if features[0, 72 + i].item() > 0.5]
            cfr_action = self._cfr_agent.act(obs_np, legal_actions, training=False)
            # Build uniform-ish CFR target (approximate since CFR returns
            # a single action, not a distribution — use eval_step if available)
            cfr_probs = torch.zeros(4, device=self.device)
            cfr_probs[cfr_action] = 1.0
            # Smooth to avoid KL explosion
            cfr_probs = cfr_probs * 0.9 + 0.025
            cfr_probs = cfr_probs / cfr_probs.sum()

            mask = torch.full((4,), float("-inf"), device=self.device)
            for a in legal_actions:
                mask[a] = 0.0
            agent_log_probs = F.log_softmax(logits.squeeze(0) + mask, dim=-1)

            kl = F.kl_div(agent_log_probs, cfr_probs, reduction="batchmean")
            kl_values.append(kl)

        if not kl_values:
            return None
        return torch.stack(kl_values).mean()

    def set_cfr_agent(self, cfr_agent: object) -> None:
        """Set the CFR agent used as KL target during training."""
        self._cfr_agent = cfr_agent


class _StateProxy:
    """Minimal State-like object for build_features."""

    def __init__(
        self,
        obs: npt.NDArray[np.float64],
        legal_actions: dict[int, None],
        player_id: int,
    ) -> None:
        self.obs = obs
        self.legal_actions = legal_actions
        self.player_id = player_id


def _make_state_proxy(
    obs: npt.NDArray[np.float64],
    legal_actions: list[int],
    player_id: int,
) -> _StateProxy:
    from collections import OrderedDict

    return _StateProxy(
        obs=obs,
        legal_actions=OrderedDict.fromkeys(legal_actions),
        player_id=player_id,
    )
