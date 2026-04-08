from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

OPPONENT_COLORS = {
    "calling_station": "#1f77b4",
    "maniac": "#d62728",
    "old_man_coffee": "#2ca02c",
    "polarizing": "#ff7f0e",
}

SIGMA_OVERALL = 5
SIGMA_OPP = 3
SIGMA_EARLY_LATE = 4


def plot_training(
    rewards: np.ndarray,
    opponents: np.ndarray,
    model_name: str,
    output_path: Path,
    early_rewards: np.ndarray | None = None,
    late_rewards: np.ndarray | None = None,
) -> None:
    unique_opps = sorted(set(opponents))
    opp_counts = {opp: int(np.sum(opponents == opp)) for opp in unique_opps}
    has_early_late = (
        early_rewards is not None
        and late_rewards is not None
        and len(early_rewards) > 0
        and np.any(np.isfinite(early_rewards))
    )

    nrows = 3 if has_early_late else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(14, 5 * nrows))

    ax1 = axes[0]
    episodes_all = np.arange(1, len(rewards) + 1)
    if len(rewards) >= 3:
        smoothed = gaussian_filter1d(rewards.astype(float), sigma=SIGMA_OVERALL)
        ax1.plot(episodes_all, smoothed, color="#1f77b4", linewidth=1.5)
    ax1.scatter(
        episodes_all,
        rewards,
        c=[OPPONENT_COLORS.get(o, "gray") for o in opponents],
        alpha=0.3,
        s=10,
    )
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Avg Payoff per Hand")
    ax1.set_title(f"{model_name} — Overall Training (gaussian σ={SIGMA_OVERALL})")

    ax2 = axes[1]
    for opp in unique_opps:
        mask = opponents == opp
        opp_rewards = rewards[mask].astype(float)
        opp_episodes = np.where(mask)[0] + 1
        color = OPPONENT_COLORS.get(opp, "gray")
        if len(opp_rewards) >= 3:
            smoothed_opp = gaussian_filter1d(opp_rewards, sigma=SIGMA_OPP)
            ax2.plot(
                opp_episodes,
                smoothed_opp,
                color=color,
                linewidth=1.5,
                label=f"{opp} (n={opp_counts[opp]})",
            )
        ax2.scatter(opp_episodes, opp_rewards, color=color, alpha=0.2, s=8)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Avg Payoff per Hand")
    ax2.set_title(f"{model_name} — Per-Opponent Training (gaussian σ={SIGMA_OPP})")
    ax2.legend(loc="lower right")

    if has_early_late:
        ax3 = axes[2]
        valid = np.isfinite(early_rewards) & np.isfinite(late_rewards)
        valid_idx = np.where(valid)[0]
        valid_episodes = valid_idx + 1
        valid_early = early_rewards[valid].astype(float)
        valid_late = late_rewards[valid].astype(float)

        if len(valid_early) >= 3:
            sm_early = gaussian_filter1d(valid_early, sigma=SIGMA_EARLY_LATE)
            sm_late = gaussian_filter1d(valid_late, sigma=SIGMA_EARLY_LATE)
            ax3.plot(
                valid_episodes,
                sm_early,
                color="#d62728",
                linewidth=1.5,
                label="First half",
            )
            ax3.plot(
                valid_episodes,
                sm_late,
                color="#2ca02c",
                linewidth=1.5,
                label="Second half",
            )

        ax3.scatter(valid_episodes, valid_early, color="#d62728", alpha=0.2, s=8)
        ax3.scatter(valid_episodes, valid_late, color="#2ca02c", alpha=0.2, s=8)
        ax3.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Avg Payoff per Hand")
        ax3.set_title(
            f"{model_name} — Adaptation: First Half vs Second Half "
            f"(gaussian σ={SIGMA_EARLY_LATE})"
        )
        ax3.legend()

    sns.set_theme(style="darkgrid")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


def plot_from_log(model_name: str, output_path: Path | None = None) -> None:
    log_path = MODELS_DIR / model_name / "training_log.npz"
    if not log_path.exists():
        print(f"No training log found at {log_path}")
        return

    data = np.load(log_path, allow_pickle=True)
    rewards = data["episode_rewards"]
    opponents = data["opponent_names"]
    early = data["early_rewards"] if "early_rewards" in data else None
    late = data["late_rewards"] if "late_rewards" in data else None

    if output_path is None:
        output_path = MODELS_DIR / model_name / "training_curves.png"

    plot_training(rewards, opponents, model_name, output_path, early, late)
    print(f"Plot saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot AC training curves")
    parser.add_argument("--model", type=str, default="ac_pure")
    args = parser.parse_args()
    plot_from_log(args.model)


if __name__ == "__main__":
    main()
