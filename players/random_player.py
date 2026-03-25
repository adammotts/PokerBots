import numpy as np

from env.state import State
from players.base_player import BasePlayer


class RandomPlayer(BasePlayer):
    def __init__(self) -> None:
        super().__init__(player_name="Random")

    def act(self, state: State) -> int:
        return int(np.random.sample(state.legal_actions))
