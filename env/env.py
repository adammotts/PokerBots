import rlcard

class PokerEnv:
    def __init__(self):
        self.env = rlcard.make('no-limit-holdem')

    def reset(self):
        state = self.env.reset()
        return self._process(state)

    def step(self, action):
        state, _ = self.env.step(action)
        return self._process(state)

    def _process(self, state):
        return {
            "obs": state["obs"],
            "legal_actions": list(state["legal_actions"].keys())
        }

    def is_terminal(self):
        return self.env.is_over()

    def get_payoffs(self):
        return self.env.get_payoffs()