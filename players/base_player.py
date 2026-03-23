from abc import ABC, abstractmethod

from env.processed_state import ProcessedState


class BasePlayer(ABC):
    @abstractmethod
    def act(self, state: ProcessedState) -> int:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass
