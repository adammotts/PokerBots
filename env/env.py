import numpy as np
import numpy.typing as npt
import rlcard

from env.processed_state import ProcessedState


class PokerEnv:
    def __init__(self, *, allow_step_back: bool = False) -> None:
        self.env: rlcard.envs.Env = rlcard.make(
            "limit-holdem",
            {"allow_step_back": allow_step_back},
        )

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
            raw_obs=state["raw_obs"],
        )

    def is_terminal(self) -> bool:
        return self.env.is_over()

    def get_payoffs(self) -> npt.NDArray[np.float64]:
        return self.env.get_payoffs()
