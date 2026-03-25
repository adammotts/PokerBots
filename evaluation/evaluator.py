from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import trange

from env.env import PokerEnv
from players.base_player import BasePlayer


class Evaluator:
    def __init__(
        self, *, env: PokerEnv, player0: BasePlayer, player1: BasePlayer
    ) -> None:
        self.env: PokerEnv = env
        self.players: dict[int, BasePlayer] = {
            0: player0,
            1: player1,
        }

    def run_episode(self) -> np.ndarray:
        state = self.env.reset()

        while not self.env.is_terminal():
            action = self.players[state.player_id].act(state)
            state = self.env.step(action)

        return self.env.get_payoffs()

    def run_matchup(self, *, num_episodes: int) -> np.ndarray:
        rewards = np.zeros(num_episodes)
        for episode_num in trange(num_episodes, desc="Evaluating"):
            payoffs = self.run_episode()

            rewards[episode_num] = payoffs[0]

        return rewards

    def evaluate(self, *, num_episodes: int, output_directory: Path) -> None:
        BIG_BLIND = 2
        WINDOW = 500
        payoffs = self.run_matchup(num_episodes=num_episodes)

        mbb_per_hand = payoffs / BIG_BLIND * 1000
        avg_mbb = mbb_per_hand.mean()
        std_mbb = mbb_per_hand.std() / np.sqrt(num_episodes)

        print(f"\n{'=' * 50}")
        print(f"  {self.players[0].player_name} vs {self.players[1].player_name}")
        print(f"  Episodes:    {num_episodes:,}")
        print(f"  Avg payoff:  {payoffs.mean():.4f}")
        print(f"  mbb/h:       {avg_mbb:.1f} ± {std_mbb:.1f}")
        print(f"{'=' * 50}")

        # ── Plot ────────────────────────────────────────────────────────
        sns.set_theme(style="darkgrid")
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Rolling average mbb/h
        rolling = np.convolve(mbb_per_hand, np.ones(WINDOW) / WINDOW, mode="valid")
        axes[0].plot(range(WINDOW - 1, num_episodes), rolling, linewidth=1)
        axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        axes[0].set_ylabel("mbb/h (rolling avg)")
        axes[0].set_title(
            f"{self.players[0].player_name} vs {self.players[1].player_name}  —  {avg_mbb:.1f} ± {std_mbb:.1f} mbb/h"
        )

        # Cumulative average mbb/h
        cumulative = np.cumsum(mbb_per_hand) / np.arange(1, num_episodes + 1)
        axes[1].plot(cumulative, linewidth=1)
        axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        axes[1].set_xlabel("Hands played")
        axes[1].set_ylabel("Cumulative mbb/h")

        plt.tight_layout()
        output_directory.mkdir(parents=True, exist_ok=True)
        out = (
            output_directory
            / f"{self.players[0].player_name}_vs_{self.players[1].player_name}.png"
        )
        fig.savefig(out, dpi=150)
        print(f"  Plot saved to {out}")
