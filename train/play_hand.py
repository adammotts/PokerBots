"""Shared training utility: play one hand between a BaseAgent and a BasePlayer."""

from __future__ import annotations

from agents.base_agent import BaseAgent, Transition
from env.action import Action
from env.env import PokerEnv
from players.base_player import BasePlayer

ACTION_NAMES = {a.value: a.name.lower() for a in Action}


def play_hand(
    env: PokerEnv,
    agent: BaseAgent,
    opponent: BasePlayer,
    agent_seat: int,
) -> float:
    """Play one hand. Returns the agent's payoff.

    Args:
        env: The poker environment.
        agent: Any BaseAgent (trains via act/observe/update).
        opponent: Any BasePlayer (fixed policy).
        agent_seat: Which seat (0 or 1) the agent occupies this hand.
    """
    state = env.reset()
    action_record: list[tuple[int, str]] = []

    while not env.is_terminal():
        pid = state.player_id

        if pid == agent_seat:
            action = agent.act(
                state.obs,
                list(state.legal_actions.keys()),
                training=True,
                player_id=pid,
                action_record=action_record,
            )
        else:
            action = opponent.act(state)

        action_record.append((pid, ACTION_NAMES[action]))
        state = env.step(action)

    payoffs = env.get_payoffs()
    agent_payoff = float(payoffs[agent_seat])

    agent.observe(
        Transition(
            obs=state.obs,
            action=0,
            reward=agent_payoff,
            next_obs=state.obs,
            done=True,
        )
    )

    return agent_payoff
