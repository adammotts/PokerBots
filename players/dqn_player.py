from __future__ import annotations

from agents.dqn_agent import DoubleDQNAgent
from env.state import State
from players.base_player import BasePlayer


class DoubleDQNPlayer(BasePlayer):
    """Wraps DoubleDQNAgent behind the BasePlayer interface for evaluation."""

    def __init__(self, *, agent: DoubleDQNAgent) -> None:
        super().__init__(player_name="DoubleDQN")
        self.agent = agent

    def reset_hand(self) -> None:
        self.agent.reset_hand_state()

    def act(self, state: State) -> int:
        return self.agent.act(state=state, training=False)
