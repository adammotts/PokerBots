from env.action import Action
from env.state import State
from players.base_player import BasePlayer


class OldManCoffeePlayer(BasePlayer):
    def __init__(self) -> None:
        super().__init__(player_name="Old Man Coffee")

    def act(self, state: State) -> int:
        sorted_hand = sorted([card.rank for card in state.hand])

        if (
            sorted_hand == ["A", "A"]
            or sorted_hand == ["K", "K"]
            or sorted_hand == ["Q", "Q"]
        ):
            if Action.RAISE.value in state.legal_actions:
                return Action.RAISE.value

            else:
                return Action.CALL.value

        elif Action.CHECK.value in state.legal_actions:
            return Action.CHECK.value

        else:
            return Action.FOLD.value
