import numpy as np
import numpy.typing as npt

from agents.base_agent import BaseAgent, Transition


class RandomAgent(BaseAgent):
    """Uniformly random action selection. Used as a baseline opponent."""

    def act(
        self,
        obs: npt.NDArray[np.float64],
        legal_actions: list[int],
        *,
        training: bool = True,
        raw_obs: dict[str, object] | None = None,
        action_record: list[tuple[int, str]] | None = None,
        player_id: int = 0,
    ) -> int:
        return int(np.random.choice(legal_actions))

    def observe(self, transition: Transition) -> None:
        pass

    def update(self) -> None:
        pass

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass
