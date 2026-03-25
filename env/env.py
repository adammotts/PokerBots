import numpy as np
import numpy.typing as npt
import rlcard
from rlcard.envs.env import Env

from env.state import State


class PokerEnv:
    def __init__(self, *, allow_step_back: bool = False) -> None:
        self.env: Env = rlcard.make(
            "limit-holdem",
            {"allow_step_back": allow_step_back},
        )

    def reset(self) -> State:
        state, player_id = self.env.reset()
        return State(**state, player_id=player_id)

    def step(self, action: int) -> State:
        state, player_id = self.env.step(action)
        return State(**state, player_id=player_id)

    def is_terminal(self) -> bool:
        return self.env.is_over()

    def get_payoffs(self) -> npt.NDArray[np.float64]:
        return self.env.get_payoffs()
