from agents.cfr_agent import CFRAgent
from env.state import State
from players.base_player import BasePlayer


class CFRPlayer(BasePlayer):
    def __init__(self, *, cfr_agent: CFRAgent) -> None:
        super().__init__(player_name="CFR")
        self.cfr_agent: CFRAgent = cfr_agent

    def act(self, state: State) -> int:
        return self.cfr_agent.act(state=state, training=False)
