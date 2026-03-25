import os
from pathlib import Path

from env.env import PokerEnv
from evaluation.evaluator import Evaluator
from players.base_player import BasePlayer
from players.calling_station_player import CallingStationPlayer
from players.folding_player import FoldingPlayer
from players.maniac_player import ManiacPlayer
from players.old_man_coffee_player import OldManCoffeePlayer
from players.polarizing_player import PolarizingPlayer
from players.random_player import RandomPlayer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

players: dict[str, BasePlayer] = {
    "calling": CallingStationPlayer(),
    "folder": FoldingPlayer(),
    "maniac": ManiacPlayer(),
    "omc": OldManCoffeePlayer(),
    "polar": PolarizingPlayer(),
    "random": RandomPlayer(),
}


def main() -> None:
    env = PokerEnv()

    run_all = os.getenv("ALL")

    if not run_all:
        p0_name = os.getenv("PLAYER0")
        p1_name = os.getenv("PLAYER1")

        player0 = players[p0_name]
        player1 = players[p1_name]

        evaluator = Evaluator(env=env, player0=player0, player1=player1)

        evaluator.evaluate(num_episodes=10_000, output_directory=RESULTS_DIR)

    else:
        player_list = list(players.keys())
        n = len(player_list)
        for i in range(n):
            for j in range(i + 1, n):
                player0 = players[player_list[i]]
                player1 = players[player_list[j]]

                evaluator = Evaluator(env=env, player0=player0, player1=player1)

                evaluator.evaluate(num_episodes=10_000, output_directory=RESULTS_DIR)


if __name__ == "__main__":
    main()
