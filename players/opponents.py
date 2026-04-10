from __future__ import annotations

import random

from players.base_player import BasePlayer
from players.calling_station_player import CallingStationPlayer
from players.folding_player import FoldingPlayer
from players.maniac_player import ManiacPlayer
from players.old_man_coffee_player import OldManCoffeePlayer
from players.parameterized_player import ParameterizedPlayer
from players.polarizing_player import PolarizingPlayer
from players.random_player import RandomPlayer

OPPONENT_CLASSES: dict[str, type[BasePlayer]] = {
    "calling": CallingStationPlayer,
    "folder": FoldingPlayer,
    "maniac": ManiacPlayer,
    "omc": OldManCoffeePlayer,
    "polar": PolarizingPlayer,
    "random": RandomPlayer,
    "calling_station": CallingStationPlayer,
    "old_man_coffee": OldManCoffeePlayer,
    "polarizing": PolarizingPlayer,
}


def make_opponent(name: str) -> BasePlayer:
    return OPPONENT_CLASSES[name]()


def _categorize(vpip: float, aggression: float) -> str:
    tight = vpip < 0.4
    passive = aggression < 1.5
    if tight and passive:
        return "tight-passive"
    if tight:
        return "tight-aggro"
    if passive:
        return "loose-passive"
    return "loose-aggro"


_CATEGORY_RANGES = {
    "tight-passive": {"vpip": (0.15, 0.40), "aggression": (0.3, 1.5)},
    "tight-aggro": {"vpip": (0.15, 0.40), "aggression": (1.5, 4.0)},
    "loose-passive": {"vpip": (0.40, 0.95), "aggression": (0.3, 1.5)},
    "loose-aggro": {"vpip": (0.40, 0.95), "aggression": (1.5, 4.0)},
}
_CATEGORIES = list(_CATEGORY_RANGES.keys())


def make_random_parameterized() -> tuple[BasePlayer, str]:
    category = random.choice(_CATEGORIES)
    ranges = _CATEGORY_RANGES[category]
    vpip = random.uniform(*ranges["vpip"])
    aggression = random.uniform(*ranges["aggression"])
    pfr = random.uniform(0.05, vpip)
    fold_to_raise = random.uniform(0.05, 0.80)
    player = ParameterizedPlayer(
        vpip=vpip,
        pfr=pfr,
        aggression=aggression,
        fold_to_raise=fold_to_raise,
    )
    return player, category
