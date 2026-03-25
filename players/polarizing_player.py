from env.action import Action
from env.state import State
from players.base_player import BasePlayer


class PolarizingPlayer(BasePlayer):
    def __init__(self) -> None:
        super().__init__(player_name="Polar")

    def act(self, state: State) -> int:
        c1, c2 = state.hand
        sorted_hand = sorted([card.rank for card in state.hand])

        if (
            sorted_hand == ["A", "A"]
            or sorted_hand == ["K", "K"]
            or sorted_hand == ["Q", "Q"]
            or sorted_hand == ["A", "K"]
            or (sorted_hand == ["4", "5"] and c1.suit == c2.suit)
            or (sorted_hand == ["5", "6"] and c1.suit == c2.suit)
            or (sorted_hand == ["6", "7"] and c1.suit == c2.suit)
            or (sorted_hand == ["7", "8"] and c1.suit == c2.suit)
        ):
            if Action.RAISE.value in state.legal_actions:
                return Action.RAISE.value

            else:
                return Action.CALL.value

        elif (
            sorted_hand[0] == sorted_hand[1]
            or c1.suit == c2.suit
            or (sorted_hand[0].isalpha() and sorted_hand[1].isalpha())
        ):
            if Action.CHECK.value in state.legal_actions:
                return Action.CHECK.value

            else:
                return Action.CALL.value

        elif Action.CHECK.value in state.legal_actions:
            return Action.CHECK.value

        else:
            return Action.FOLD.value
