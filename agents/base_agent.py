from abc import ABC

class BaseAgent(ABC):
    def act(self, obs, legal_actions, training=True):
        raise NotImplementedError

    def observe(self, transition):
        pass

    def update(self):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass