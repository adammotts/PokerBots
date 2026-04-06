from pathlib import Path

from agents.ac_agent import ActorCriticAgent
from env.state import State
from players.base_player import BasePlayer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "ac_kl" / "final.pt"


class ActorCriticKlPlayer(BasePlayer):
    def __init__(self) -> None:
        super().__init__(player_name="ac-kl", is_agent=True)
        self.agent = ActorCriticAgent()
        self.agent.load(str(MODEL_PATH))

    def act(self, state: State) -> int:
        return self.agent.act(
            state=state,
            training=False,
        )
