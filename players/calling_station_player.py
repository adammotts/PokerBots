from env.action import Action
from env.state import State
from players.base_player import BasePlayer


class CallingStationPlayer(BasePlayer):
    def __init__(self) -> None:
        super().__init__(player_name="Calling Station")

    def act(self, state: State) -> int:
        if Action.CALL.value in state.legal_actions:
            return Action.CALL.value

        else:
            return Action.CHECK.value
