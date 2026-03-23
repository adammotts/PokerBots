from abc import ABC

class BaseOpponent(ABC):
    def act(self, obs, legal_actions):
        raise NotImplementedError