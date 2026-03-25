import sys
from pathlib import Path

from env.env import PokerEnv
from evaluation.evaluator import Evaluator
from players.calling_station_player import CallingStationPlayer
from players.maniac_player import ManiacPlayer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS_DIR = ROOT / "results"


def main() -> None:
    env = PokerEnv()

    calling_station_player = CallingStationPlayer()
    maniac_player = ManiacPlayer()

    evaluator = Evaluator(
        env=env, player0=calling_station_player, player1=maniac_player
    )

    evaluator.evaluate(num_episodes=10_000, output_directory=RESULTS_DIR)


if __name__ == "__main__":
    main()
