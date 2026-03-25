from env.action import Action
from env.processed_state import ProcessedState
from players.base_player import BasePlayer


class CallingStationPlayer(BasePlayer):
    def __init__(self) -> None:
        pass

    def act(self, state: ProcessedState) -> Action:
        if Action.CALL in state["legal_actions"]:
            return Action.CALL

        else:
            return Action.CHECK

    def reset(self) -> None:
        pass
