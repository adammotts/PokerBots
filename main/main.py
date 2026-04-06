from __future__ import annotations

import argparse

from env.env import PokerEnv
from evaluation.evaluator import (
    DEFAULT_AGENTS,
    PLAYER_FACTORIES,
    RESULTS_DIR,
    Evaluator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run poker evaluations")
    subparsers = parser.add_subparsers(dest="command", required=True)

    matchup = subparsers.add_parser("matchup", help="Run matchup evaluation")
    matchup.add_argument("--agent")
    matchup.add_argument("--opponent")
    matchup.add_argument("--episodes", type=int, default=10_000)
    matchup.add_argument("--all", action="store_true")

    sessions = subparsers.add_parser(
        "sessions",
        help="Run multi-session adaptation evaluation",
    )
    sessions.add_argument("--agents", nargs="+", required=True)
    sessions.add_argument("--opponent", choices=sorted(PLAYER_FACTORIES), required=True)
    sessions.add_argument("--sessions", type=int, default=50)
    sessions.add_argument("--hands", type=int, default=1000)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "matchup":
        if args.all:
            for agent_name in DEFAULT_AGENTS:
                for opponent_name in PLAYER_FACTORIES:
                    print(f"\n{'=' * 50}")
                    print(f"  {agent_name} vs {opponent_name}")
                    print(f"{'=' * 50}")
                    Evaluator(
                        env=PokerEnv(),
                        player0=PLAYER_FACTORIES[agent_name](),
                        player1=PLAYER_FACTORIES[opponent_name](),
                    ).evaluate(
                        num_episodes=args.episodes,
                        output_directory=RESULTS_DIR,
                    )
            return

        if not args.agent or not args.opponent:
            raise ValueError(
                "`matchup` requires --agent and --opponent unless --all is set."
            )

        agent_name = args.agent.strip().lower().replace("_", "-")
        opponent_name = args.opponent.strip().lower().replace("_", "-")
        if agent_name not in PLAYER_FACTORIES:
            raise ValueError(f"Unknown player '{agent_name}'")

        print(f"\n{'=' * 50}")
        print(f"  {agent_name} vs {opponent_name}")
        print(f"{'=' * 50}")
        Evaluator(
            env=PokerEnv(),
            player0=PLAYER_FACTORIES[agent_name](),
            player1=PLAYER_FACTORIES[opponent_name](),
        ).evaluate(
            num_episodes=args.episodes,
            output_directory=RESULTS_DIR,
        )
        return

    results = {}
    opponent_name = args.opponent.strip().lower().replace("_", "-")
    plotter: Evaluator | None = None

    for agent_name in args.agents:
        label = agent_name.strip().lower().replace("_", "-")
        if label not in PLAYER_FACTORIES:
            raise ValueError(f"Unknown player '{label}'")

        print(f"\nEvaluating {label} vs {opponent_name}...")
        plotter = Evaluator(
            env=PokerEnv(),
            player0=PLAYER_FACTORIES[label](),
            player1=PLAYER_FACTORIES[opponent_name](),
        )
        results[label] = plotter.evaluate_sessions(
            label=label,
            opponent_name=opponent_name,
            num_sessions=args.sessions,
            num_hands=args.hands,
            output_directory=RESULTS_DIR,
        )

    if plotter is None:
        raise ValueError("No agents provided for session evaluation.")

    plotter.plot_session_results(
        results=results,
        opponent_name=opponent_name,
        output_path=RESULTS_DIR / f"sessions_vs_{opponent_name}.png",
    )


if __name__ == "__main__":
    main()
