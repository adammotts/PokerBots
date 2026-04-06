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
        '''
        In RL Card, the first env.reset() can either choose the first to act as player ID 0 or 1. Whatever it chooses,
        it doesn't matter, because the next hand it will switch. The player ID mapping in the evaluator class does not
        need to be tied to the same ID as that in RL Card every hand. All that needs to happen is that the player IDs
        must remain the same in the Evaluator class
        '''

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

        output_directory.mkdir(parents=True, exist_ok=True)
        name = f"{self.players[0].player_name}_vs_{self.players[1].player_name}"

        npz_path = output_directory / f"{name}.npz"
        np.savez(npz_path, payoffs=payoffs)
        print(f"  Data saved to {npz_path}")

        sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots(figsize=(10, 5))

        rolling = np.convolve(mbb_per_hand, np.ones(WINDOW) / WINDOW, mode="valid")
        ax.plot(range(WINDOW - 1, num_episodes), rolling, linewidth=1)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Hands played")
        ax.set_ylabel("mbb/h (rolling avg)")
        ax.set_title(f"{name}  —  {avg_mbb:.1f} ± {std_mbb:.1f} mbb/h")

        plt.tight_layout()
        png_path = output_directory / f"{name}.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"  Plot saved to {png_path}")
