from env.action import Action
from env.state import State
from players.base_player import BasePlayer


class OldManCoffeePlayer(BasePlayer):
    def __init__(self) -> None:
        super().__init__(player_name="Old Man Coffee")

    def act(self, state: State) -> int:
        c1, c2 = state.hand

        if (
            c1.rank == c2.rank == "A"
            or c1.rank == c2.rank == "K"
            or c1.rank == c2.rank == "Q"
        ):
            if Action.RAISE.value in state.legal_actions:
                return Action.RAISE.value

            else:
                return Action.CALL.value

        elif Action.CHECK.value in state.legal_actions:
            return Action.CHECK.value

        else:
            return Action.FOLD.value
