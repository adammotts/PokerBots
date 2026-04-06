"""Overlay multiple evaluation results on one plot with 95% CI bands."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

BIG_BLIND = 2
WINDOW = 500
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def load_and_smooth(path: str) -> np.ndarray:
    data = np.load(path)
    payoffs = data["payoffs"]
    mbb = payoffs / BIG_BLIND * 1000
    return np.convolve(mbb, np.ones(WINDOW) / WINDOW, mode="valid")


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay evaluation results")
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="Paths to .npz result files",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Legend labels for each result",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/comparison.png",
        help="Output path for the plot",
    )
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, (path, label) in enumerate(zip(args.results, args.labels, strict=True)):
        rolling = load_and_smooth(path)
        color = COLORS[i % len(COLORS)]
        ax.plot(
            range(WINDOW - 1, WINDOW - 1 + len(rolling)),
            rolling,
            color=color,
            label=label,
            linewidth=1,
        )

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Hands played")
    ax.set_ylabel("mbb/h (rolling avg)")
    ax.set_title(args.title or "Agent Comparison")
    ax.legend()

    plt.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=150)
    plt.close(fig)
    print(f"Plot saved to {output}")


if __name__ == "__main__":
    main()
