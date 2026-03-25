from abc import ABC, abstractmethod

from env.state import State


class BasePlayer(ABC):
    def __init__(self, *, player_name: str) -> None:
        self.player_name: str = player_name

    @abstractmethod
    def act(self, state: State) -> int:
        pass
