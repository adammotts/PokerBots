from __future__ import annotations

from players.base_player import BasePlayer
from players.calling_station_player import CallingStationPlayer
from players.folding_player import FoldingPlayer
from players.maniac_player import ManiacPlayer
from players.old_man_coffee_player import OldManCoffeePlayer
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
