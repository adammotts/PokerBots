import sys
from pathlib import Path

from env.env import PokerEnv
from evaluation.evaluator import Evaluator
from players.calling_station_player import CallingStationPlayer
from players.maniac_player import ManiacPlayer
from players.old_man_coffee_player import OldManCoffeePlayer
from players.folding_player import FoldingPlayer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def main() -> None:
    env = PokerEnv()

    calling_station_player = CallingStationPlayer()
    maniac_player = ManiacPlayer()
    old_man_coffee_player = OldManCoffeePlayer()
    folding_player = FoldingPlayer()

    evaluator = Evaluator(
        env=env, player0=maniac_player, player1=calling_station_player
    )

    evaluator.evaluate(num_episodes=10_000, output_directory=RESULTS_DIR)


if __name__ == "__main__":
    main()
