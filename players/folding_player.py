from env.action import Action
from env.state import State
from players.base_player import BasePlayer


class FoldingPlayer(BasePlayer):
    def __init__(self) -> None:
        super().__init__(player_name="Folder")

    def act(self, state: State) -> int:
        assert Action.FOLD.value in state.legal_actions

        return Action.FOLD.value
