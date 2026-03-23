from typing import TypedDict

import numpy as np
import numpy.typing as npt
import rlcard


class ProcessedState(TypedDict):
    obs: npt.NDArray[np.float64]
    legal_actions: list[int]


class PokerEnv:
    def __init__(self) -> None:
        self.env: rlcard.envs.Env = rlcard.make("limit-holdem")

    def reset(self) -> ProcessedState:
        state, _ = self.env.reset()
        return self._process(state)

    def step(self, action: int) -> ProcessedState:
        state, _ = self.env.step(action)
        return self._process(state)

    def _process(self, state: dict[str, object]) -> ProcessedState:
        return ProcessedState(
            obs=state["obs"],
            legal_actions=list(state["legal_actions"].keys()),
        )

    def is_terminal(self) -> bool:
        return self.env.is_over()

    def get_payoffs(self) -> npt.NDArray[np.float64]:
        return self.env.get_payoffs()
