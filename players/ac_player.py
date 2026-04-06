from agents.ac_agent import ActorCriticAgent
from env.state import State
from players.base_player import BasePlayer


class ActorCriticPlayer(BasePlayer):
    """Wraps ActorCriticAgent behind the BasePlayer interface for evaluation."""

    def __init__(self, *, agent: ActorCriticAgent) -> None:
        super().__init__(player_name="ActorCritic")
        self.agent = agent

    def act(self, state: State) -> int:
        return self.agent.act(
            state=state,
            training=False,
        )
