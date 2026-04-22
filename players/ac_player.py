from agents.ac_agent import ActorCriticAgent
from env.state import State
from players.base_player import BasePlayer


class ActorCriticPlayer(BasePlayer):
    def __init__(self, *, agent: ActorCriticAgent) -> None:
        super().__init__(player_name="ActorCritic")
        self.agent = agent
        self.action_record: list[tuple[int, str]] = []

    def reset_session(self) -> None:
        self.agent.reset_opponent_state()

    def reset_hand(self) -> None:
        self.agent.reset_hand_state()
        self.action_record = []

    def act(self, state: State) -> int:
        return self.agent.act(
            state=state,
            training=False,
        )

    def record_action(self, player_id: int, action_name: str) -> None:
        self.action_record.append((player_id, action_name))

    def end_hand(self, payoff: float) -> None:
        self.agent.step_opponent_after_hand(self.action_record, payoff)
