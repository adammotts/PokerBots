import os
from pathlib import Path

from env.env import PokerEnv
from evaluation.evaluator import Evaluator
from evaluation.model_loader import load_player, parse_agent_spec
from players.base_player import BasePlayer
from players.calling_station_player import CallingStationPlayer
from players.folding_player import FoldingPlayer
from players.maniac_player import ManiacPlayer
from players.old_man_coffee_player import OldManCoffeePlayer
from players.polarizing_player import PolarizingPlayer
from players.random_player import RandomPlayer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

opponents: dict[str, BasePlayer] = {
    "calling": CallingStationPlayer(),
    "folder": FoldingPlayer(),
    "maniac": ManiacPlayer(),
    "omc": OldManCoffeePlayer(),
    "polar": PolarizingPlayer(),
    "random": RandomPlayer(),
}


def build_agents() -> dict[str, BasePlayer]:
    return {
        "ac-pure": load_player(parse_agent_spec("ac:ac_pure"), MODELS_DIR),
        "dqn-calling": load_player(parse_agent_spec("dqn:dqn_calling"), MODELS_DIR),
        "dqn-maniac": load_player(parse_agent_spec("dqn:dqn_maniac"), MODELS_DIR),
        "dqn-omc": load_player(parse_agent_spec("dqn:dqn_omc"), MODELS_DIR),
        "dqn-polar": load_player(parse_agent_spec("dqn:dqn_polar"), MODELS_DIR),
        "random": RandomPlayer(),
    }


def main() -> None:
    env = PokerEnv()
    agents = build_agents()

    run_all = os.getenv("ALL")

    if not run_all:
        agent_name = os.getenv("AGENT")
        opp_name = os.getenv("OPPONENT")

        agent = agents[agent_name]
        opponent = opponents[opp_name]

        evaluator = Evaluator(env=env, player0=agent, player1=opponent)
        evaluator.evaluate(num_episodes=10_000, output_directory=RESULTS_DIR)

    else:
        for agent_name, agent in agents.items():
            for opp_name, opponent in opponents.items():
                print(f"\n{'=' * 50}")
                print(f"  {agent_name} vs {opp_name}")
                print(f"{'=' * 50}")

                evaluator = Evaluator(env=env, player0=agent, player1=opponent)
                evaluator.evaluate(num_episodes=10_000, output_directory=RESULTS_DIR)


if __name__ == "__main__":
    main()
