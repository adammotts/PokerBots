import os
from pathlib import Path

from env.env import PokerEnv
from evaluation.evaluator import Evaluator
from players.base_player import BasePlayer
from players.calling_station_player import CallingStationPlayer
from players.folding_player import FoldingPlayer
from players.maniac_player import ManiacPlayer
from players.old_man_coffee_player import OldManCoffeePlayer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def get_player(name: str) -> BasePlayer:
    if name == "maniac":
        return ManiacPlayer()
    elif name == "calling":
        return CallingStationPlayer()
    elif name == "omc":
        return OldManCoffeePlayer()
    elif name == "folder":
        return FoldingPlayer()
    else:
        raise ValueError(f"Unknown player: {name}")


def main() -> None:
    env = PokerEnv()

    p0_name = os.getenv("PLAYER0")
    p1_name = os.getenv("PLAYER1")

    player0 = get_player(p0_name)
    player1 = get_player(p1_name)

    evaluator = Evaluator(env=env, player0=player0, player1=player1)

    evaluator.evaluate(num_episodes=10_000, output_directory=RESULTS_DIR)


if __name__ == "__main__":
    main()
