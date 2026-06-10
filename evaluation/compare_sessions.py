from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
BIG_BLIND = 2


def plot_comparison(
    entries: list[tuple[str, str, str]],
    opponent_name: str,
    output_path: Path,
) -> None:
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    for label, color, npz_file in entries:
        path = RESULTS_DIR / npz_file
        data = np.load(path)
        payoffs_2d = data["payoffs"]
        mbb = payoffs_2d / BIG_BLIND * 1000
        num_hands = mbb.shape[1]
        cumulative = np.cumsum(mbb, axis=1) / np.arange(1, num_hands + 1)

        mean = np.mean(cumulative, axis=0)
        se = 1.96 * np.std(cumulative, axis=0) / np.sqrt(cumulative.shape[0])

        ax.plot(range(1, num_hands + 1), mean, color=color, label=label)
        ax.fill_between(
            range(1, num_hands + 1),
            mean - se,
            mean + se,
            alpha=0.1,
            color=color,
        )

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 400)
    ax.set_xlabel("Hands played")
    ax.set_ylabel("Cumulative mbb/h")
    ax.set_title(f"Session Performance vs {opponent_name}")
    ax.legend(loc="lower left")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    opponent = "Old Man Coffee"

    # (label, color, npz_filename)
    # colors: any matplotlib/seaborn name, e.g. "tab:blue", "royalblue", "coral", "seagreen"
    entries = [
        ("AC-v2-meta", "firebrick", "sessions_ac-v2-meta_vs_omc.npz"),
        ("AC-v2", "coral", "sessions_ac-v2_vs_omc.npz"),
        ("DQN-maniac", "navy", "sessions_dqn-maniac_vs_omc.npz"),
        ("DQN-polar", "royalblue", "sessions_dqn-polar_vs_omc.npz"),
        ("DQN-omc", "cornflowerblue", "sessions_dqn-omc_vs_omc.npz"),
        ("DQN-calling", "lightskyblue", "sessions_dqn-calling_vs_omc.npz"),
        ("Random", "seagreen", "sessions_random_vs_omc.npz"),
    ]

    output = RESULTS_DIR / f"compare_vs_{opponent}.png"
    plot_comparison(entries, opponent, output)
