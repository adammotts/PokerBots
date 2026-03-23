from abc import ABC

class BasePlayer(ABC):
    def act(self, obs, legal_actions):
        raise NotImplementedError