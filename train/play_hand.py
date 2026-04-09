"""Shared training utility: play one hand between a BaseAgent and a BasePlayer."""

from __future__ import annotations

import numpy as np

from agents.base_agent import BaseAgent, Transition
from agents.features import build_features
from env.action import Action
from env.env import PokerEnv
from players.base_player import BasePlayer

ACTION_NAMES = {a.value: a.name.lower() for a in Action}


def _build_agent_features(agent: BaseAgent, state) -> np.ndarray:
    builder = getattr(agent, "build_features", None)
    if callable(builder):
        return builder(state).cpu().numpy()
    return build_features(state).cpu().numpy()


def play_hand(
    env: PokerEnv,
    agent: BaseAgent,
    opponent: BasePlayer,
) -> float:
    """Play one hand. Returns the agent's payoff.

    Args:
        env: The poker environment.
        agent: Any BaseAgent (trains via act/observe/update).
        opponent: Any BasePlayer (fixed policy).
    """
    state = env.reset()
    if hasattr(agent, "reset_hand_state"):
        agent.reset_hand_state()
    action_record: list[tuple[int, str]] = []
    pending_obs: np.ndarray | None = None
    pending_action: int | None = None

    while not env.is_terminal():
        pid = state.player_id

        if pid == 0:
            if pending_obs is not None and pending_action is not None:
                agent.observe(
                    Transition(
                        obs=pending_obs,
                        action=pending_action,
                        reward=0.0,
                        next_obs=_build_agent_features(agent, state),
                        done=False,
                    )
                )

            action = agent.act(
                state=state,
                training=True,
                action_record=action_record,
            )
            pending_obs = _build_agent_features(agent, state)
            pending_action = action
        else:
            action = opponent.act(state)

        action_record.append((pid, ACTION_NAMES[action]))
        state = env.step(action)

    payoffs = env.get_payoffs()
    agent_payoff = float(payoffs[0])

    if pending_obs is not None and pending_action is not None:
        agent.observe(
            Transition(
                obs=pending_obs,
                action=pending_action,
                reward=agent_payoff,
                next_obs=_build_agent_features(agent, state),
                done=True,
            )
        )

    return agent_payoff
