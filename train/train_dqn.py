"""Train a recurrent Double DQN exploitative agent against a fixed archetype."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from agents.dqn_agent import DoubleDQNAgent
from env.env import PokerEnv
from players.opponents import OPPONENT_CLASSES, make_opponent
from train.play_hand import play_hand

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def train(
    *,
    name: str,
    opponent_name: str,
    num_hands: int,
    checkpoint_every: int,
    lr: float,
    device: str,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_hands: int,
    replay_capacity: int,
    batch_size: int,
    warmup_hands: int,
    target_update_every: int,
) -> None:
    save_dir = MODELS_DIR / name
    save_dir.mkdir(parents=True, exist_ok=True)

    agent = DoubleDQNAgent(
        lr=lr,
        device=device,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_hands=epsilon_decay_hands,
        replay_capacity=replay_capacity,
        batch_size=batch_size,
        warmup_hands=warmup_hands,
        target_update_every=target_update_every,
    )

    final_path = save_dir / "final.pt"
    if final_path.exists():
        print(f"Resuming from {final_path}")
        agent.load(str(final_path))

    env = PokerEnv()
    opponent = make_opponent(opponent_name)

    payoffs: list[float] = []
    epsilon_history: list[float] = []
    log_path = save_dir / "training_log.npz"

    start_hand = agent.training_hands
    if start_hand > 0:
        print(f"Resuming from hand {start_hand}")

    def save_log() -> None:
        np.savez(
            log_path,
            payoffs=np.array(payoffs, dtype=np.float32),
            epsilons=np.array(epsilon_history, dtype=np.float32),
            opponent=np.array([opponent_name]),
        )

    for hand_index in trange(start_hand, num_hands, desc=f"DQN vs {opponent_name}"):
        payoff = play_hand(env, agent, opponent)
        agent.update()
        payoffs.append(payoff)
        epsilon_history.append(agent.epsilon)

        hand_num = hand_index + 1
        if hand_num % checkpoint_every == 0:
            ckpt_path = save_dir / f"hand{hand_num}.pt"
            agent.save(str(ckpt_path))
            save_log()
            recent = np.mean(payoffs[-checkpoint_every:])
            print(
                f"Hand {hand_num:6d} | avg payoff: {recent:+.4f} | "
                f"epsilon: {agent.epsilon:.3f}"
            )
            print(f"  Checkpoint saved: {ckpt_path}")

    agent.save(str(final_path))
    save_log()
    print(f"\nTraining complete. Final model saved to {final_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train recurrent Double DQN agent")
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Model name. Saves to models/<name>/",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        required=True,
        choices=sorted(OPPONENT_CLASSES.keys()),
        help="Fixed opponent archetype to train against",
    )
    parser.add_argument("--hands", type=int, default=50_000)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-hands", type=int, default=20_000)
    parser.add_argument("--replay-capacity", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-hands", type=int, default=200)
    parser.add_argument("--target-update-every", type=int, default=200)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train(
        name=args.name,
        opponent_name=args.opponent,
        num_hands=args.hands,
        checkpoint_every=args.checkpoint_every,
        lr=args.lr,
        device=device,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_hands=args.epsilon_decay_hands,
        replay_capacity=args.replay_capacity,
        batch_size=args.batch_size,
        warmup_hands=args.warmup_hands,
        target_update_every=args.target_update_every,
    )


if __name__ == "__main__":
    main()
