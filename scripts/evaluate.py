import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from tqdm import trange  # noqa: E402

from agents.base_agent import BaseAgent  # noqa: E402
from agents.cfr_agent import CFRAgent  # noqa: E402
from agents.openspiel_cfr_agent import OpenSpielCFRAgent  # noqa: E402
from env.env import PokerEnv  # noqa: E402

# ── Config ──────────────────────────────────────────────────────────
NUM_EPISODES = 10_000
BIG_BLIND = 10
WINDOW = 500
RESULTS_DIR = ROOT / "results"

AGENT_NAME = "openspiel-cfr"
OPPONENT_NAME = "random"


# ── Agent loading ───────────────────────────────────────────────────
def load_agent(name: str) -> BaseAgent:
    if name == "openspiel-cfr":
        agent = OpenSpielCFRAgent(iterations=1)
        agent.load(str(ROOT / "models" / "openspiel_cfr"))
        return agent
    if name == "cfr":
        agent = CFRAgent(model_path=str(ROOT / "models" / "cfr"))
        agent.load(str(ROOT / "models" / "cfr"))
        return agent
    msg = f"Unknown agent: {name}"
    raise ValueError(msg)


# ── Game loop ───────────────────────────────────────────────────────
def run_episode(env: PokerEnv, agents: list[BaseAgent]) -> np.ndarray:
    """Play one hand and return payoffs."""
    state = env.reset()

    while not env.is_terminal():
        pid = state["player_id"]
        action = agents[pid].act(
            state["obs"],
            state["legal_actions"],
            training=False,
            raw_obs=state["raw_obs"],
            action_record=env.env.action_recorder,
            player_id=pid,
        )
        state = env.step(action)

    return env.get_payoffs()


# ── Main ────────────────────────────────────────────────────────────
def main() -> None:
    env = PokerEnv()
    agent = load_agent(AGENT_NAME)
    opponent = load_agent(OPPONENT_NAME)

    payoffs = np.zeros(NUM_EPISODES)

    for i in trange(NUM_EPISODES, desc="Evaluating"):
        # Alternate positions each hand for fairness
        if i % 2 == 0:
            agents = [agent, opponent]
            seat = 0
        else:
            agents = [opponent, agent]
            seat = 1

        episode_payoffs = run_episode(env, agents)
        payoffs[i] = episode_payoffs[seat]

    # ── Stats ───────────────────────────────────────────────────────
    mbb_per_hand = payoffs / BIG_BLIND * 1000
    avg_mbb = mbb_per_hand.mean()
    std_mbb = mbb_per_hand.std() / np.sqrt(NUM_EPISODES)

    print(f"\n{'=' * 50}")
    print(f"  {AGENT_NAME} vs {OPPONENT_NAME}")
    print(f"  Episodes:    {NUM_EPISODES:,}")
    print(f"  Avg payoff:  {payoffs.mean():.4f}")
    print(f"  mbb/h:       {avg_mbb:.1f} ± {std_mbb:.1f}")
    print(f"{'=' * 50}")

    # ── Plot ────────────────────────────────────────────────────────
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Rolling average mbb/h
    rolling = np.convolve(mbb_per_hand, np.ones(WINDOW) / WINDOW, mode="valid")
    axes[0].plot(range(WINDOW - 1, NUM_EPISODES), rolling, linewidth=1)
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("mbb/h (rolling avg)")
    axes[0].set_title(
        f"{AGENT_NAME} vs {OPPONENT_NAME}  —  {avg_mbb:.1f} ± {std_mbb:.1f} mbb/h"
    )

    # Cumulative average mbb/h
    cumulative = np.cumsum(mbb_per_hand) / np.arange(1, NUM_EPISODES + 1)
    axes[1].plot(cumulative, linewidth=1)
    axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Hands played")
    axes[1].set_ylabel("Cumulative mbb/h")

    plt.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{AGENT_NAME}_vs_{OPPONENT_NAME}.png"
    fig.savefig(out, dpi=150)
    print(f"  Plot saved to {out}")


if __name__ == "__main__":
    main()
