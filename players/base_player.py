from abc import ABC, abstractmethod

from env.state import State


class BasePlayer(ABC):
    def __init__(self, *, player_name: str, is_agent: bool = False) -> None:
        self.player_name: str = player_name
        self.is_agent: bool = is_agent

    @abstractmethod
    def act(self, state: State) -> int:
        pass
