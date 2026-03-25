from env.action import Action
from env.state import State
from players.base_player import BasePlayer


class CallingStationPlayer(BasePlayer):
    def __init__(self, *, player_id: int) -> None:
        super().__init__(player_id=player_id, player_name="Calling Station")

    def act(self, state: State) -> int:
        if Action.CALL.value in state.legal_actions:
            return Action.CALL.value

        else:
            return Action.CHECK.value
