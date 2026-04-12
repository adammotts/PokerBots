from __future__ import annotations

import os
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from rlcard.games.base import Card
from torch.distributions import Categorical

from agents.ac_agent.features import build_opponent_summary
from agents.ac_agent.networks import (
    ActorNetwork,
    CentralizedCritic,
    OpponentLSTM,
    OpponentPredictor,
)
from agents.ac_agent.rollout import HandRollout, StepData
from agents.base_agent import BaseAgent, Transition
from agents.features import build_features, encode_both_hands_onehot
from env.state import State


def _compute_gae(
    rewards: list[float],
    values: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    T = len(values)
    advantages = torch.zeros(T, device=values.device)
    gae = 0.0
    for t in reversed(range(T)):
        next_val = 0.0 if t == T - 1 else values[t + 1].item()
        delta = rewards[t] + gamma * next_val - values[t].item()
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = advantages + values.detach()
    return advantages, returns


class ActorCriticAgent(BaseAgent):
    def __init__(
        self,
        *,
        lr: float = 3e-4,
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.05,
        value_coef: float = 0.5,
        grad_clip: float = 0.5,
        ppo_epochs: int = 4,
        clip_eps: float = 0.2,
        aux_coef: float = 0.3,
        extra_critic_steps: int = 5,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.grad_clip = grad_clip
        self.ppo_epochs = ppo_epochs
        self.clip_eps = clip_eps
        self.aux_coef = aux_coef
        self.extra_critic_steps = extra_critic_steps

        self.actor = ActorNetwork().to(device)
        self.critic = CentralizedCritic().to(device)
        self.opponent_lstm = OpponentLSTM().to(device)
        self.opp_predictor = OpponentPredictor().to(device)

        self.actor_optimizer = torch.optim.Adam(
            [
                *self.actor.parameters(),
                *self.opponent_lstm.parameters(),
                *self.opp_predictor.parameters(),
            ],
            lr=lr,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=lr,
        )

        self._opp_hidden = self.opponent_lstm.init_hidden(device)

        self._hand_reward: float = 0.0
        self._action_record: list[tuple[int, str]] = []
        self._our_player_id: int = 0

        self._collecting: bool = False
        self._collect_steps: list[StepData] = []
        self._current_opp_actions: list[int] = []
        self._rollout: list[HandRollout] = []
        self._opp_hidden_rollout_start: tuple[torch.Tensor, torch.Tensor] | None = None

    def reset_opponent_state(self) -> None:
        self._opp_hidden = self.opponent_lstm.init_hidden(self.device)
        self.reset_hand_state()

    def reset_hand_state(self) -> None:
        self._action_record = []

    def _get_opp_context(self) -> torch.Tensor:
        return self._opp_hidden[0].squeeze(0)

    def act(
        self,
        *,
        state: State,
        training: bool = True,
        action_record: list[tuple[int, str]] | None = None,
        both_hands: tuple[Sequence[Card], Sequence[Card]] | None = None,
    ) -> int:
        self._our_player_id = state.player_id

        if action_record is not None:
            self._action_record = action_record

        features = build_features(state, self.device).unsqueeze(0)
        opp_context = self._get_opp_context()

        with torch.set_grad_enabled(training):
            logits = self.actor(features, opp_context)

        mask = torch.full((4,), float("-inf"), device=self.device)
        for a in state.legal_actions:
            mask[a] = 0.0
        masked_logits = logits.squeeze(0) + mask

        dist = Categorical(logits=masked_logits)

        if training:
            action = dist.sample()
            if self._collecting and both_hands is not None:
                both_onehot = encode_both_hands_onehot(
                    both_hands[0], both_hands[1], self.device
                )
                self._collect_steps.append(
                    StepData(
                        features=features.detach().squeeze(0),
                        both_hands_onehot=both_onehot,
                        action=int(action.item()),
                        log_prob_old=dist.log_prob(action).detach().item(),
                        legal_mask=mask.clone(),
                    )
                )
        else:
            action = masked_logits.argmax()

        return int(action.item())

    def observe(self, transition: Transition) -> None:
        if transition["done"]:
            self._hand_reward = transition["reward"]

    def update(self) -> None:
        pass

    def set_opp_actions(self, opp_actions: list[int]) -> None:
        self._current_opp_actions = opp_actions

    def begin_collect(self) -> None:
        self._collecting = True
        self._rollout = []
        self._collect_steps = []
        self._current_opp_actions = []
        self._opp_hidden_rollout_start = (
            self._opp_hidden[0].detach().clone(),
            self._opp_hidden[1].detach().clone(),
        )

    def finish_hand_collect(self) -> None:
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
        self._rollout.append(
            HandRollout(
                steps=self._collect_steps,
                reward=self._hand_reward,
                hand_summary=summary.detach(),
                opp_actions=list(self._current_opp_actions),
            )
        )
        self._collect_steps = []
        self._current_opp_actions = []
        with torch.no_grad():
            summary_input = summary.unsqueeze(0).unsqueeze(0)
            _, self._opp_hidden = self.opponent_lstm(summary_input, self._opp_hidden)
        self._opp_hidden = (
            self._opp_hidden[0].detach(),
            self._opp_hidden[1].detach(),
        )
        self._action_record = []

    def ppo_update(self) -> dict[str, float]:
        rollout = self._rollout
        if not rollout or self._opp_hidden_rollout_start is None:
            self._collecting = False
            return {}

        total_steps = sum(len(h.steps) for h in rollout)
        if total_steps == 0:
            self._collecting = False
            return {}

        log_probs_old_t = torch.tensor(
            [s.log_prob_old for h in rollout for s in h.steps],
            dtype=torch.float32,
            device=self.device,
        )

        actor_params = (
            list(self.actor.parameters())
            + list(self.opponent_lstm.parameters())
            + list(self.opp_predictor.parameters())
        )

        diagnostics: dict[str, float] = {}

        for epoch in range(self.ppo_epochs):
            opp_hidden = (
                self._opp_hidden_rollout_start[0].clone(),
                self._opp_hidden_rollout_start[1].clone(),
            )

            all_log_probs_new: list[torch.Tensor] = []
            all_entropies: list[torch.Tensor] = []
            all_advantages: list[torch.Tensor] = []
            all_returns: list[torch.Tensor] = []
            all_values: list[torch.Tensor] = []
            aux_losses: list[torch.Tensor] = []

            for hand in rollout:
                if not hand.steps:
                    summary_input = hand.hand_summary.unsqueeze(0).unsqueeze(0)
                    _, opp_hidden = self.opponent_lstm(summary_input, opp_hidden)
                    continue

                opp_context = opp_hidden[0].squeeze(0)
                step_features = torch.stack([s.features for s in hand.steps])
                step_both_hands = torch.stack([s.both_hands_onehot for s in hand.steps])
                step_masks = torch.stack([s.legal_mask for s in hand.steps])
                n_steps = len(hand.steps)
                opp_ctx_expanded = opp_context.expand(n_steps, -1)

                logits = self.actor(step_features, opp_ctx_expanded)
                values = self.critic(
                    step_features, step_both_hands, opp_ctx_expanded
                ).squeeze(-1)

                masked_logits = logits + step_masks
                dist = Categorical(logits=masked_logits)
                actions_t = torch.tensor(
                    [s.action for s in hand.steps], device=self.device
                )
                all_log_probs_new.append(dist.log_prob(actions_t))
                all_entropies.append(dist.entropy())

                rewards = [0.0] * n_steps
                rewards[-1] = hand.reward
                advantages, returns = _compute_gae(
                    rewards, values.detach(), self.gamma, self.gae_lambda
                )
                all_advantages.append(advantages)
                all_returns.append(returns)
                all_values.append(values)

                if hand.opp_actions:
                    pred_logits = self.opp_predictor(opp_context)
                    targets = torch.tensor(
                        hand.opp_actions, dtype=torch.long, device=self.device
                    )
                    expanded = pred_logits.expand(len(hand.opp_actions), -1)
                    aux_losses.append(F.cross_entropy(expanded, targets))

                summary_input = hand.hand_summary.unsqueeze(0).unsqueeze(0)
                _, opp_hidden = self.opponent_lstm(summary_input, opp_hidden)

            if not all_log_probs_new:
                continue

            log_probs_new_t = torch.cat(all_log_probs_new)
            entropies_t = torch.cat(all_entropies)
            advantages_t = torch.cat(all_advantages)
            returns_t = torch.cat(all_returns)
            values_t = torch.cat(all_values)

            if advantages_t.numel() > 1:
                advantages_t = (advantages_t - advantages_t.mean()) / (
                    advantages_t.std() + 1e-8
                )

            ratio = torch.exp(log_probs_new_t - log_probs_old_t)
            surr1 = ratio * advantages_t
            surr2 = (
                torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages_t
            )
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values_t, returns_t)
            entropy_loss = -entropies_t.mean()

            actor_loss = policy_loss + self.entropy_coef * entropy_loss
            if aux_losses:
                aux_loss = torch.stack(aux_losses).mean()
                actor_loss = actor_loss + self.aux_coef * aux_loss
            else:
                aux_loss = torch.tensor(0.0)

            total_loss = actor_loss + self.value_coef * value_loss

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(actor_params, self.grad_clip)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            for _ in range(self.extra_critic_steps):
                opp_h_extra = (
                    self._opp_hidden_rollout_start[0].clone().detach(),
                    self._opp_hidden_rollout_start[1].clone().detach(),
                )
                extra_values_list: list[torch.Tensor] = []
                extra_returns_list: list[torch.Tensor] = []
                for hand in rollout:
                    if not hand.steps:
                        with torch.no_grad():
                            si = hand.hand_summary.unsqueeze(0).unsqueeze(0)
                            _, opp_h_extra = self.opponent_lstm(si, opp_h_extra)
                        continue
                    opp_ctx = opp_h_extra[0].squeeze(0).detach()
                    sf = torch.stack([s.features for s in hand.steps])
                    sbh = torch.stack([s.both_hands_onehot for s in hand.steps])
                    n = len(hand.steps)
                    oc = opp_ctx.expand(n, -1)
                    v = self.critic(sf, sbh, oc).squeeze(-1)
                    rewards = [0.0] * n
                    rewards[-1] = hand.reward
                    _, ret = _compute_gae(
                        rewards, v.detach(), self.gamma, self.gae_lambda
                    )
                    extra_values_list.append(v)
                    extra_returns_list.append(ret)
                    with torch.no_grad():
                        si = hand.hand_summary.unsqueeze(0).unsqueeze(0)
                        _, opp_h_extra = self.opponent_lstm(si, opp_h_extra)
                if extra_values_list:
                    ev = torch.cat(extra_values_list)
                    er = torch.cat(extra_returns_list)
                    extra_loss = self.value_coef * F.mse_loss(ev, er)
                    self.critic_optimizer.zero_grad()
                    extra_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
                    self.critic_optimizer.step()

            opp_hidden = (opp_hidden[0].detach(), opp_hidden[1].detach())

            if epoch == 0:
                diagnostics = {
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "entropy": entropies_t.mean().item(),
                    "aux_loss": aux_loss.item(),
                }

        self._rollout = []
        self._collecting = False
        return diagnostics

    def get_trial(self) -> list[HandRollout]:
        trial = list(self._rollout)
        self._rollout = []
        self._collecting = False
        return trial

    def meta_ppo_update(self, trials: list[list[HandRollout]]) -> dict[str, float]:
        total_steps = sum(len(s.steps) for trial in trials for s in trial)
        if total_steps == 0:
            return {}

        log_probs_old_t = torch.tensor(
            [s.log_prob_old for trial in trials for hand in trial for s in hand.steps],
            dtype=torch.float32,
            device=self.device,
        )

        actor_params = (
            list(self.actor.parameters())
            + list(self.opponent_lstm.parameters())
            + list(self.opp_predictor.parameters())
        )

        diagnostics: dict[str, float] = {}

        for epoch in range(self.ppo_epochs):
            all_log_probs_new: list[torch.Tensor] = []
            all_entropies: list[torch.Tensor] = []
            all_advantages: list[torch.Tensor] = []
            all_returns: list[torch.Tensor] = []
            all_values: list[torch.Tensor] = []
            aux_losses: list[torch.Tensor] = []

            for trial in trials:
                opp_hidden = self.opponent_lstm.init_hidden(self.device)

                for hand in trial:
                    if not hand.steps:
                        summary_input = hand.hand_summary.unsqueeze(0).unsqueeze(0)
                        _, opp_hidden = self.opponent_lstm(summary_input, opp_hidden)
                        continue

                    opp_context = opp_hidden[0].squeeze(0)
                    step_features = torch.stack([s.features for s in hand.steps])
                    step_both_hands = torch.stack(
                        [s.both_hands_onehot for s in hand.steps]
                    )
                    step_masks = torch.stack([s.legal_mask for s in hand.steps])
                    n_steps = len(hand.steps)
                    opp_ctx_expanded = opp_context.expand(n_steps, -1)

                    logits = self.actor(step_features, opp_ctx_expanded)
                    values = self.critic(
                        step_features, step_both_hands, opp_ctx_expanded
                    ).squeeze(-1)

                    masked_logits = logits + step_masks
                    dist = Categorical(logits=masked_logits)
                    actions_t = torch.tensor(
                        [s.action for s in hand.steps], device=self.device
                    )
                    all_log_probs_new.append(dist.log_prob(actions_t))
                    all_entropies.append(dist.entropy())

                    rewards = [0.0] * n_steps
                    rewards[-1] = hand.reward
                    advantages, returns = _compute_gae(
                        rewards, values.detach(), self.gamma, self.gae_lambda
                    )
                    all_advantages.append(advantages)
                    all_returns.append(returns)
                    all_values.append(values)

                    if hand.opp_actions:
                        pred_logits = self.opp_predictor(opp_context)
                        targets = torch.tensor(
                            hand.opp_actions,
                            dtype=torch.long,
                            device=self.device,
                        )
                        expanded = pred_logits.expand(len(hand.opp_actions), -1)
                        aux_losses.append(F.cross_entropy(expanded, targets))

                    summary_input = hand.hand_summary.unsqueeze(0).unsqueeze(0)
                    _, opp_hidden = self.opponent_lstm(summary_input, opp_hidden)

            if not all_log_probs_new:
                continue

            log_probs_new_t = torch.cat(all_log_probs_new)
            entropies_t = torch.cat(all_entropies)
            advantages_t = torch.cat(all_advantages)
            returns_t = torch.cat(all_returns)
            values_t = torch.cat(all_values)

            if advantages_t.numel() > 1:
                advantages_t = (advantages_t - advantages_t.mean()) / (
                    advantages_t.std() + 1e-8
                )

            ratio = torch.exp(log_probs_new_t - log_probs_old_t)
            surr1 = ratio * advantages_t
            surr2 = (
                torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages_t
            )
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values_t, returns_t)
            entropy_loss = -entropies_t.mean()

            actor_loss = policy_loss + self.entropy_coef * entropy_loss
            if aux_losses:
                aux_loss = torch.stack(aux_losses).mean()
                actor_loss = actor_loss + self.aux_coef * aux_loss
            else:
                aux_loss = torch.tensor(0.0)

            total_loss = actor_loss + self.value_coef * value_loss

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(actor_params, self.grad_clip)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            if epoch == 0:
                diagnostics = {
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "entropy": entropies_t.mean().item(),
                    "aux_loss": aux_loss.item(),
                }

        return diagnostics

    def step_opponent_after_hand(
        self,
        action_record: list[tuple[int, str]],
        payoff: float,
    ) -> None:
        self._action_record = action_record
        self._hand_reward = payoff
        self._our_player_id = 0
        with torch.no_grad():
            self._step_opponent_lstm()

    def _step_opponent_lstm(self) -> None:
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
        summary_input = summary.unsqueeze(0).unsqueeze(0)
        _, self._opp_hidden = self.opponent_lstm(summary_input, self._opp_hidden)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state_dict: dict[str, object] = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "opponent_lstm": self.opponent_lstm.state_dict(),
            "opp_predictor": self.opp_predictor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }
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
        if "opp_predictor" in checkpoint:
            self.opp_predictor.load_state_dict(checkpoint["opp_predictor"])
        if "actor_optimizer" in checkpoint:
            try:
                self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            except ValueError:
                pass
        if "critic_optimizer" in checkpoint:
            try:
                self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
            except ValueError:
                pass
