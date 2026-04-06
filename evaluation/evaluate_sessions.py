"""Multi-session evaluation: run an agent against one opponent for many
independent sessions, plot cumulative mbb/h over hands with 95% CI bands.

Shows how the agent adapts within a session as the opponent LSTM builds up.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import trange

from agents.ac_agent import ActorCriticAgent
from env.env import PokerEnv
from players.ac_player import ActorCriticPlayer
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
BIG_BLIND = 2

OPPONENT_CLASSES: dict[str, type[BasePlayer]] = {
    "calling": CallingStationPlayer,
    "folder": FoldingPlayer,
    "maniac": ManiacPlayer,
    "omc": OldManCoffeePlayer,
    "polar": PolarizingPlayer,
    "random": RandomPlayer,
}

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def run_session(
    env: PokerEnv,
    agent_player: ActorCriticPlayer,
    opponent: BasePlayer,
    num_hands: int,
) -> np.ndarray:
    agent_player.agent.reset_opponent_state()
    payoffs = np.zeros(num_hands)

    for hand in range(num_hands):
        state = env.reset()
        agent_seat = hand % 2
        players: dict[int, BasePlayer] = {
            agent_seat: agent_player,
            1 - agent_seat: opponent,
        }

        while not env.is_terminal():
            action = players[state.player_id].act(state)
            state = env.step(action)

        payoffs[hand] = env.get_payoffs()[agent_seat]

    return payoffs


def plot_sessions(
    results: dict[str, np.ndarray],
    opponent_name: str,
    output_path: Path,
) -> None:
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, (label, payoffs_2d) in enumerate(results.items()):
        mbb = payoffs_2d / BIG_BLIND * 1000
        num_hands = mbb.shape[1]
        cumulative = np.cumsum(mbb, axis=1) / np.arange(1, num_hands + 1)

        mean = np.mean(cumulative, axis=0)
        se = 1.96 * np.std(cumulative, axis=0) / np.sqrt(cumulative.shape[0])
        color = COLORS[i % len(COLORS)]

        ax.plot(range(1, num_hands + 1), mean, color=color, label=label)
        ax.fill_between(
            range(1, num_hands + 1),
            mean - se,
            mean + se,
            alpha=0.3,
            color=color,
        )

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Hands played")
    ax.set_ylabel("Cumulative mbb/h")
    ax.set_title(f"Session Performance vs {opponent_name}")
    ax.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"Plot saved to {output_path}")


def load_agent(name: str) -> ActorCriticPlayer:
    agent = ActorCriticAgent()
    model_path = MODELS_DIR / name / "final.pt"
    agent.load(str(model_path))
    return ActorCriticPlayer(agent=agent)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate agent adaptation over hands")
    parser.add_argument(
        "--agents",
        nargs="+",
        required=True,
        help="Agent model names (e.g. ac_pure ac_kl)",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        required=True,
        choices=list(OPPONENT_CLASSES.keys()),
    )
    parser.add_argument("--sessions", type=int, default=50)
    parser.add_argument("--hands", type=int, default=1000)
    args = parser.parse_args()

    env = PokerEnv()
    results: dict[str, np.ndarray] = {}

    for agent_name in args.agents:
        print(f"\nEvaluating {agent_name} vs {args.opponent}...")
        agent_player = load_agent(agent_name)
        payoffs_2d = np.zeros((args.sessions, args.hands))

        for session in trange(args.sessions, desc=f"{agent_name}"):
            opponent = OPPONENT_CLASSES[args.opponent]()
            payoffs_2d[session] = run_session(env, agent_player, opponent, args.hands)

        npz_path = RESULTS_DIR / f"sessions_{agent_name}_vs_{args.opponent}.npz"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(npz_path, payoffs=payoffs_2d)
        print(f"Data saved to {npz_path}")

        results[agent_name] = payoffs_2d

    output_path = RESULTS_DIR / f"sessions_vs_{args.opponent}.png"
    plot_sessions(results, args.opponent, output_path)


if __name__ == "__main__":
    main()
