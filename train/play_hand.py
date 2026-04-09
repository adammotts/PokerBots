from __future__ import annotations

import numpy as np

from agents.base_agent import BaseAgent, Transition
from agents.features import build_features
from env.action import ACTION_NAMES
from env.env import PokerEnv
from players.base_player import BasePlayer


def play_hand(
    env: PokerEnv,
    agent: BaseAgent,
    opponent: BasePlayer,
) -> float:
    state = env.reset()
    if hasattr(agent, "reset_hand_state"):
        agent.reset_hand_state()

    both_hands = (
        tuple(env.env.game.players[0].hand),
        tuple(env.env.game.players[1].hand),
    )

    action_record: list[tuple[int, str]] = []
    opp_actions: list[int] = []
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
                        next_obs=build_features(state).cpu().numpy(),
                        done=False,
                    )
                )

            action = agent.act(
                state=state,
                training=True,
                action_record=action_record,
                both_hands=both_hands,
            )
            pending_obs = build_features(state).cpu().numpy()
            pending_action = action
        else:
            action = opponent.act(state)
            opp_actions.append(action)

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
                next_obs=build_features(state).cpu().numpy(),
                done=True,
            )
        )

    if hasattr(agent, "set_opp_actions"):
        agent.set_opp_actions(opp_actions)

    return agent_payoff
