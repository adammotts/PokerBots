from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import trange

from env.env import PokerEnv
from players.ac_kl_player import ActorCriticKlPlayer
from players.ac_pure_player import ActorCriticPurePlayer
from players.base_player import BasePlayer
from players.calling_station_player import CallingStationPlayer
from players.folding_player import FoldingPlayer
from players.maniac_player import ManiacPlayer
from players.old_man_coffee_player import OldManCoffeePlayer
from players.polarizing_player import PolarizingPlayer
from players.random_player import RandomPlayer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_AGENTS = ["ac-pure", "ac-kl"]
PLAYER_FACTORIES: dict[str, type[BasePlayer]] = {
    "ac-pure": ActorCriticPurePlayer,
    "ac-kl": ActorCriticKlPlayer,
    "calling": CallingStationPlayer,
    "folder": FoldingPlayer,
    "maniac": ManiacPlayer,
    "omc": OldManCoffeePlayer,
    "polar": PolarizingPlayer,
    "random": RandomPlayer,
}


class Evaluator:
    BIG_BLIND = 2
    ROLLING_WINDOW = 500
    SESSION_COLORS = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
    ]

    def __init__(
        self, *, env: PokerEnv, player0: BasePlayer, player1: BasePlayer
    ) -> None:
        self.env: PokerEnv = env
        self.players: dict[int, BasePlayer] = {
            0: player0,
            1: player1,
        }

    def run_episode(self) -> np.ndarray:
        '''
        In RL Card, the first env.reset() can either choose the first to act as player ID 0 or 1. Whatever it chooses,
        it doesn't matter, because the next hand it will switch. The players map in the Evaluator class ties player ID
        0 to player ID 0 in RLCard, and player ID 1 to player ID 1 in RLCard
        '''

        state = self.env.reset()

        while not self.env.is_terminal():
            action = self.players[state.player_id].act(state)
            state = self.env.step(action)

        return self.env.get_payoffs()

    def run_session(self, *, num_hands: int) -> np.ndarray:
        if self.players[0].is_agent:
            self.players[0].agent.reset_opponent_state()

        payoffs = np.zeros(num_hands)
        for hand in range(num_hands):
            episode_payoffs = self.run_episode()
            payoffs[hand] = episode_payoffs[0]

        return payoffs

    def run_session_batch(self, *, num_sessions: int, num_hands: int) -> np.ndarray:
        payoffs = np.zeros((num_sessions, num_hands))
        for session_num in trange(num_sessions, desc="Evaluating sessions"):
            payoffs[session_num] = self.run_session(num_hands=num_hands)
        return payoffs

    def plot_session_results(
        self,
        *,
        results: dict[str, np.ndarray],
        opponent_name: str,
        output_path: Path,
    ) -> None:
        sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, (label, payoffs_2d) in enumerate(results.items()):
            mbb = payoffs_2d / self.BIG_BLIND * 1000
            num_hands = mbb.shape[1]
            cumulative = np.cumsum(mbb, axis=1) / np.arange(1, num_hands + 1)

            mean = np.mean(cumulative, axis=0)
            se = 1.96 * np.std(cumulative, axis=0) / np.sqrt(cumulative.shape[0])
            color = self.SESSION_COLORS[i % len(self.SESSION_COLORS)]
            x = range(1, num_hands + 1)

            ax.plot(x, mean, color=color, label=label)
            ax.fill_between(x, mean - se, mean + se, alpha=0.3, color=color)

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

    def evaluate(self, *, num_episodes: int, output_directory: Path) -> None:
        payoffs = self.run_session(num_hands=num_episodes)

        mbb_per_hand = payoffs / self.BIG_BLIND * 1000
        avg_mbb = mbb_per_hand.mean()
        std_mbb = mbb_per_hand.std() / np.sqrt(num_episodes)

        print(f"\n{'=' * 50}")
        print(f"  {self.players[0].player_name} vs {self.players[1].player_name}")
        print(f"  Episodes:    {num_episodes:,}")
        print(f"  Avg payoff:  {payoffs.mean():.4f}")
        print(f"  mbb/h:       {avg_mbb:.1f} ± {std_mbb:.1f}")
        print(f"{'=' * 50}")

        output_directory.mkdir(parents=True, exist_ok=True)
        name = f"{self.players[0].player_name}_vs_{self.players[1].player_name}"

        npz_path = output_directory / f"{name}.npz"
        np.savez(npz_path, payoffs=payoffs)
        print(f"  Data saved to {npz_path}")

        sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots(figsize=(10, 5))

        rolling = np.convolve(
            mbb_per_hand,
            np.ones(self.ROLLING_WINDOW) / self.ROLLING_WINDOW,
            mode="valid",
        )
        ax.plot(range(self.ROLLING_WINDOW - 1, num_episodes), rolling, linewidth=1)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Hands played")
        ax.set_ylabel("mbb/h (rolling avg)")
        ax.set_title(f"{name}  —  {avg_mbb:.1f} ± {std_mbb:.1f} mbb/h")

        plt.tight_layout()
        png_path = output_directory / f"{name}.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"  Plot saved to {png_path}")

    def evaluate_sessions(
        self,
        *,
        label: str,
        opponent_name: str,
        num_sessions: int,
        num_hands: int,
        output_directory: Path,
    ) -> np.ndarray:
        payoffs_2d = self.run_session_batch(
            num_sessions=num_sessions,
            num_hands=num_hands,
        )

        output_directory.mkdir(parents=True, exist_ok=True)
        npz_path = output_directory / f"sessions_{label}_vs_{opponent_name}.npz"
        np.savez(npz_path, payoffs=payoffs_2d)
        print(f"Data saved to {npz_path}")

        return payoffs_2d
