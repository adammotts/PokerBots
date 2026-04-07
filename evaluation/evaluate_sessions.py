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

from env.env import PokerEnv
from evaluation.model_loader import load_player, parse_agent_spec
from players.base_player import BasePlayer
from players.opponents import OPPONENT_CLASSES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
BIG_BLIND = 2

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def sanitize_label(label: str) -> str:
    return label.replace(":", "_")


def run_session(
    env: PokerEnv,
    agent_player: BasePlayer,
    opponent: BasePlayer,
    num_hands: int,
) -> np.ndarray:
    agent_player.reset_session()
    payoffs = np.zeros(num_hands)
    players: dict[int, BasePlayer] = {
        0: agent_player,
        1: opponent,
    }

    for hand in range(num_hands):
        for player in players.values():
            player.reset_hand()
        state = env.reset()

        while not env.is_terminal():
            action = players[state.player_id].act(state)
            state = env.step(action)

        payoffs[hand] = env.get_payoffs()[0]

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
        agent_spec = parse_agent_spec(agent_name)
        agent_player = load_player(agent_spec, MODELS_DIR)
        payoffs_2d = np.zeros((args.sessions, args.hands))

        for session in trange(args.sessions, desc=f"{agent_name}"):
            opponent = OPPONENT_CLASSES[args.opponent]()
            payoffs_2d[session] = run_session(env, agent_player, opponent, args.hands)

        npz_path = (
            RESULTS_DIR
            / f"sessions_{sanitize_label(agent_name)}_vs_{args.opponent}.npz"
        )
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(npz_path, payoffs=payoffs_2d)
        print(f"Data saved to {npz_path}")

        results[agent_name] = payoffs_2d

    output_path = RESULTS_DIR / f"sessions_vs_{args.opponent}.png"
    plot_sessions(results, args.opponent, output_path)


if __name__ == "__main__":
    main()
