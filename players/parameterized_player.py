from __future__ import annotations

import random

from env.action import Action
from env.state import State
from players.base_player import BasePlayer

RANK_VALUES = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}


def _hand_strength(hand: tuple) -> float:
    r0 = RANK_VALUES[hand[0].rank]
    r1 = RANK_VALUES[hand[1].rank]
    high = max(r0, r1)
    low = min(r0, r1)
    suited = hand[0].suit == hand[1].suit
    paired = r0 == r1

    score = 0.0
    if paired:
        score = 0.5 + (low - 2) / 24.0
    else:
        score = (high - 2) / 24.0 + (low - 2) / 48.0
        if suited:
            score += 0.08
        gap = high - low
        if gap <= 2:
            score += 0.04

    return min(max(score, 0.0), 1.0)


class ParameterizedPlayer(BasePlayer):
    def __init__(
        self,
        *,
        vpip: float = 0.5,
        pfr: float = 0.2,
        aggression: float = 1.0,
        fold_to_raise: float = 0.3,
    ) -> None:
        super().__init__(player_name="Parameterized")
        self.vpip = vpip
        self.pfr = min(pfr, vpip)
        self.aggression = aggression
        self.fold_to_raise = fold_to_raise
        self._preflop = True
        self._facing_raise = False

    def reset_hand(self) -> None:
        self._preflop = True
        self._facing_raise = False

    def act(self, state: State) -> int:
        legal = set(state.legal_actions.keys())
        strength = _hand_strength(state.hand)
        noise = random.uniform(-0.05, 0.05)
        strength = min(max(strength + noise, 0.0), 1.0)

        if self._preflop:
            self._preflop = False
            return self._preflop_action(strength, legal)
        return self._postflop_action(legal)

    def _preflop_action(self, strength: float, legal: set[int]) -> int:
        play_threshold = 1.0 - self.vpip
        raise_threshold = 1.0 - self.pfr

        if strength < play_threshold:
            if Action.CHECK.value in legal:
                return Action.CHECK.value
            return Action.FOLD.value

        if strength >= raise_threshold and Action.RAISE.value in legal:
            return Action.RAISE.value

        if Action.CALL.value in legal:
            return Action.CALL.value
        return Action.CHECK.value

    def _postflop_action(self, legal: set[int]) -> int:
        if self._facing_raise:
            self._facing_raise = False
            if random.random() < self.fold_to_raise:
                if Action.FOLD.value in legal:
                    return Action.FOLD.value

        raise_prob = self.aggression / (1.0 + self.aggression)

        if random.random() < raise_prob and Action.RAISE.value in legal:
            return Action.RAISE.value

        if Action.CALL.value in legal:
            return Action.CALL.value
        return Action.CHECK.value

    def record_action(self, player_id: int, action_name: str) -> None:
        if action_name == "raise":
            self._facing_raise = True
        if action_name in ("call", "check"):
            self._preflop = False
