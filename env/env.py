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
        state, player_id = self.env.reset()
        return self._process(state, player_id)

    def step(self, action: int) -> ProcessedState:
        state, player_id = self.env.step(action)
        return self._process(state, player_id)

    def _process(self, state: dict[str, object], player_id: int) -> ProcessedState:
        return ProcessedState(
            obs=state["obs"],
            legal_actions=list(state["legal_actions"].keys()),
            raw_obs=state["raw_obs"],
            player_id=player_id,
        )

    def is_terminal(self) -> bool:
        return self.env.is_over()

    def get_payoffs(self) -> npt.NDArray[np.float64]:
        return self.env.get_payoffs()
