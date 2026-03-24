from abc import ABC, abstractmethod
from typing import TypedDict

import numpy as np
import numpy.typing as npt


class Transition(TypedDict):
    obs: npt.NDArray[np.float64]
    action: int
    reward: float
    next_obs: npt.NDArray[np.float64]
    done: bool


class BaseAgent(ABC):
    @abstractmethod
    def act(
        self,
        obs: npt.NDArray[np.float64],
        legal_actions: list[int],
        *,
        training: bool = True,
        raw_obs: dict[str, object] | None = None,
        action_record: list[tuple[int, str]] | None = None,
        player_id: int = 0,
    ) -> int: ...

    @abstractmethod
    def observe(self, transition: Transition) -> None: ...

    @abstractmethod
    def update(self) -> None: ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...
