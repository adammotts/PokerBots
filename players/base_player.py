from abc import ABC, abstractmethod

from env.state import State


class BasePlayer(ABC):
    def __init__(self, *, player_id: int) -> None:
        self.player_id: int = player_id
        self.episode_payoffs: list[int] = []

    @abstractmethod
    def act(self, state: State) -> int:
        pass
