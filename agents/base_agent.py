from abc import ABC, abstractmethod
from typing import TypedDict

import numpy as np
import numpy.typing as npt

from env.state import State


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
        *,
        state: State,
        training: bool = True,
        action_record: list[tuple[int, str]] | None = None,
    ) -> int: ...

    @abstractmethod
    def observe(self, transition: Transition) -> None: ...

    @abstractmethod
    def update(self) -> None: ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...
