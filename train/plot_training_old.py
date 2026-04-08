from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

OPPONENT_COLORS = {
    "calling_station": "#1f77b4",
    "maniac": "#d62728",
    "old_man_coffee": "#2ca02c",
    "polarizing": "#ff7f0e",
}


def running_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        window = max(1, len(values))
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_training(model_name: str) -> None:
    log_path = MODELS_DIR / model_name / "training_log.npz"
    if not log_path.exists():
        print(f"No training log found at {log_path}")
        return

    data = np.load(log_path, allow_pickle=True)
    rewards = data["episode_rewards"]
    opponents = data["opponent_names"]

    unique_opps = sorted(set(opponents))
    opp_counts = {opp: np.sum(opponents == opp) for opp in unique_opps}
    print(f"Loaded {len(rewards)} episodes from {log_path}")
    for opp, count in opp_counts.items():
        mask = opponents == opp
        avg = np.mean(rewards[mask])
        print(f"  {opp}: {count} episodes, avg payoff: {avg:+.4f}")

    sns.set_theme(style="darkgrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    window_overall = 20
    ra = running_average(rewards, window_overall)
    episodes = np.arange(window_overall, len(rewards) + 1)
    ax1.plot(episodes, ra, color="#1f77b4", linewidth=1.5)
    ax1.scatter(
        np.arange(1, len(rewards) + 1),
        rewards,
        c=[OPPONENT_COLORS.get(o, "gray") for o in opponents],
        alpha=0.3,
        s=10,
    )
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Avg Payoff per Hand")
    ax1.set_title(
        f"{model_name} — Overall Training (running avg, window={window_overall})"
    )

    window_opp = 5
    for opp in unique_opps:
        mask = opponents == opp
        opp_rewards = rewards[mask]
        opp_episodes = np.where(mask)[0] + 1

        ra_opp = running_average(opp_rewards, window_opp)
        ra_episodes = opp_episodes[window_opp - 1 :]

        color = OPPONENT_COLORS.get(opp, "gray")
        ax2.plot(
            ra_episodes,
            ra_opp,
            color=color,
            linewidth=1.5,
            label=f"{opp} (n={opp_counts[opp]})",
        )
        ax2.scatter(opp_episodes, opp_rewards, color=color, alpha=0.2, s=8)

    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Avg Payoff per Hand")
    ax2.set_title(
        f"{model_name} — Per-Opponent Training (running avg, window={window_opp})"
    )
    ax2.legend(loc="lower right")

    plt.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"training_curves_{model_name}.png"
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot AC training curves")
    parser.add_argument(
        "--model",
        type=str,
        default="ac_pure",
        help="Model name (loads from models/<name>/training_log.npz)",
    )
    args = parser.parse_args()
    plot_training(args.model)


if __name__ == "__main__":
    main()
