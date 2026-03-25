from env.action import Action
from env.state import State
from players.base_player import BasePlayer


class ManiacPlayer(BasePlayer):
    def __init__(self, *, player_id: int) -> None:
        super().__init__(player_id=player_id)

    def act(self, state: State) -> int:
        if Action.RAISE.value in state.legal_actions:
            return Action.RAISE.value

        else:
            return Action.CALL.value
