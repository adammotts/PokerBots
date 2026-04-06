"""Train the Actor-Critic agent against a mixture of opponent archetypes."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from agents.ac_agent import ActorCriticAgent
from env.env import PokerEnv
from players.base_player import BasePlayer
from players.calling_station_player import CallingStationPlayer
from players.maniac_player import ManiacPlayer
from players.old_man_coffee_player import OldManCoffeePlayer
from players.polarizing_player import PolarizingPlayer
from train.play_hand import play_hand

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

OPPONENTS: dict[str, type[BasePlayer]] = {
    "calling_station": CallingStationPlayer,
    "maniac": ManiacPlayer,
    "old_man_coffee": OldManCoffeePlayer,
    "polarizing": PolarizingPlayer,
}


def make_opponent(name: str) -> BasePlayer:
    return OPPONENTS[name]()


def train(
    *,
    name: str,
    lambda_kl_max: float,
    num_episodes: int,
    hands_per_episode: int,
    checkpoint_every: int,
    lr: float,
    device: str,
) -> None:
    save_dir = MODELS_DIR / name
    save_dir.mkdir(parents=True, exist_ok=True)

    agent = ActorCriticAgent(
        lambda_kl_max=lambda_kl_max,
        lr=lr,
        device=device,
    )

    # Resume from latest checkpoint if it exists
    final_path = save_dir / "final.pt"
    if final_path.exists():
        print(f"Resuming from {final_path}")
        agent.load(str(final_path))

    env = PokerEnv()
    opponent_names = list(OPPONENTS.keys())

    all_episode_rewards: list[float] = []
    all_opponent_names: list[str] = []
    log_path = save_dir / "training_log.npz"

    def save_log() -> None:
        np.savez(
            log_path,
            episode_rewards=np.array(all_episode_rewards),
            opponent_names=np.array(all_opponent_names),
        )

    for episode in range(num_episodes):
        opp_name = random.choice(opponent_names)
        opponent = make_opponent(opp_name)
        agent.reset_opponent_state()

        episode_rewards: list[float] = []

        for hand in trange(
            hands_per_episode,
            desc=f"Ep {episode + 1}/{num_episodes} vs {opp_name}",
            leave=False,
        ):
            # Alternate seats each hand for fairness
            agent_seat = hand % 2
            payoff = play_hand(env, agent, opponent, agent_seat)
            agent.update()
            episode_rewards.append(payoff)

        avg_reward = np.mean(episode_rewards)
        all_episode_rewards.append(avg_reward)
        all_opponent_names.append(opp_name)
        print(
            f"Episode {episode + 1:3d} | vs {opp_name:<16s} | "
            f"avg payoff: {avg_reward:+.4f} | "
            f"running avg: {np.mean(all_episode_rewards):+.4f}"
        )

        if (episode + 1) % checkpoint_every == 0:
            ckpt_path = save_dir / f"ep{episode + 1}.pt"
            agent.save(str(ckpt_path))
            save_log()
            print(f"  Checkpoint saved: {ckpt_path}")

    agent.save(str(final_path))
    save_log()
    print(f"\nTraining complete. Final model saved to {final_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Actor-Critic agent")
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Model name (e.g. ac_pure, ac_kl). Saves to models/<name>/",
    )
    parser.add_argument(
        "--lambda-kl",
        type=float,
        default=0.0,
        help="Max KL regularization weight (0.0 = pure A2C, 0.5 = KL variant)",
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--hands", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train(
        name=args.name,
        lambda_kl_max=args.lambda_kl,
        num_episodes=args.episodes,
        hands_per_episode=args.hands,
        checkpoint_every=args.checkpoint_every,
        lr=args.lr,
        device=device,
    )


if __name__ == "__main__":
    main()
