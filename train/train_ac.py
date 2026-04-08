"""Train the Actor-Critic agent with PPO (or A2C) against opponent archetypes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from agents.ac_agent import ActorCriticAgent
from env.env import PokerEnv
from players.opponents import make_opponent
from train.play_hand import play_hand
from train.plot_training import plot_training

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def pick_opponent(
    names: list[str], counts: dict[str, int], temperature: float = 1.0
) -> str:
    inverse_counts = np.array([1.0 / (counts.get(n, 0) + 1) for n in names])
    logits = inverse_counts / temperature
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return str(np.random.choice(names, p=probs))


def train(
    *,
    name: str,
    lambda_kl_max: float,
    num_episodes: int,
    hands_per_episode: int,
    checkpoint_every: int,
    lr: float,
    ppo: bool,
    rollout_size: int,
    ppo_epochs: int,
    clip_eps: float,
    aux_coef: float,
    device: str,
) -> None:
    save_dir = MODELS_DIR / name
    save_dir.mkdir(parents=True, exist_ok=True)

    agent = ActorCriticAgent(
        lambda_kl_max=lambda_kl_max,
        lr=lr,
        ppo_epochs=ppo_epochs,
        clip_eps=clip_eps,
        aux_coef=aux_coef,
        device=device,
    )

    log_path = save_dir / "training_log.npz"
    plot_path = save_dir / "training_curves.png"
    final_path = save_dir / "final.pt"

    all_episode_rewards: list[float] = []
    all_early_rewards: list[float] = []
    all_late_rewards: list[float] = []
    all_opponent_names: list[str] = []
    if log_path.exists():
        old = np.load(log_path, allow_pickle=True)
        all_episode_rewards = old["episode_rewards"].tolist()
        all_opponent_names = old["opponent_names"].tolist()
        if "early_rewards" in old and "late_rewards" in old:
            all_early_rewards = old["early_rewards"].tolist()
            all_late_rewards = old["late_rewards"].tolist()
        else:
            all_early_rewards = [float("nan")] * len(all_episode_rewards)
            all_late_rewards = [float("nan")] * len(all_episode_rewards)
        print(f"Loaded {len(all_episode_rewards)} prior episodes from log")

    n_prior = len(all_episode_rewards)
    ckpt_path = save_dir / f"ep{n_prior}.pt"
    if ckpt_path.exists():
        print(f"Resuming from {ckpt_path}")
        agent.load(str(ckpt_path))
    elif final_path.exists():
        print(f"Resuming from {final_path}")
        agent.load(str(final_path))

    env = PokerEnv()
    opponent_names = ["calling_station", "maniac", "old_man_coffee", "polarizing"]

    def save_log() -> None:
        np.savez(
            log_path,
            episode_rewards=np.array(all_episode_rewards),
            early_rewards=np.array(all_early_rewards),
            late_rewards=np.array(all_late_rewards),
            opponent_names=np.array(all_opponent_names),
        )

    def save_plot() -> None:
        early = np.array(all_early_rewards) if all_early_rewards else None
        late = np.array(all_late_rewards) if all_late_rewards else None
        plot_training(
            np.array(all_episode_rewards),
            np.array(all_opponent_names),
            name,
            plot_path,
            early,
            late,
        )

    mode = "PPO" if ppo else "A2C"
    print(
        f"{mode} training: rollout={rollout_size}, epochs={ppo_epochs}, "
        f"clip={clip_eps}, lr={lr}, LR/entropy decay enabled"
    )

    opp_counts: dict[str, int] = {}
    start_episode = len(all_episode_rewards)
    remaining = num_episodes - start_episode
    if remaining <= 0:
        print(f"Already have {start_episode}/{num_episodes} episodes. Nothing to do.")
        return
    print(f"Episodes {start_episode + 1} to {num_episodes} ({remaining} remaining)")

    for episode in range(start_episode, num_episodes):
        opp_name = pick_opponent(opponent_names, opp_counts)
        opp_counts[opp_name] = opp_counts.get(opp_name, 0) + 1
        opponent = make_opponent(opp_name)
        agent.reset_opponent_state()

        frac = 1.0 - episode / max(num_episodes - 1, 1)
        current_lr = lr * max(frac, 0.05)
        for pg in agent.optimizer.param_groups:
            pg["lr"] = current_lr
        agent.entropy_coef = 0.01 * max(frac, 0.1)

        episode_rewards: list[float] = []

        if ppo:
            hand = 0
            pbar = trange(
                hands_per_episode,
                desc=f"Ep {episode + 1}/{num_episodes} vs {opp_name}",
                leave=False,
            )
            while hand < hands_per_episode:
                batch = min(rollout_size, hands_per_episode - hand)
                agent.begin_collect()
                for _ in range(batch):
                    payoff = play_hand(env, agent, opponent)
                    agent.finish_hand_collect()
                    episode_rewards.append(payoff)
                    hand += 1
                    pbar.update(1)
                agent.ppo_update()
            pbar.close()
        else:
            for _ in trange(
                hands_per_episode,
                desc=f"Ep {episode + 1}/{num_episodes} vs {opp_name}",
                leave=False,
            ):
                payoff = play_hand(env, agent, opponent)
                agent.update()
                episode_rewards.append(payoff)

        mid = len(episode_rewards) // 2
        avg_reward = np.mean(episode_rewards)
        all_episode_rewards.append(avg_reward)
        all_early_rewards.append(float(np.mean(episode_rewards[:mid])))
        all_late_rewards.append(float(np.mean(episode_rewards[mid:])))
        all_opponent_names.append(opp_name)
        print(
            f"Episode {episode + 1:3d} | vs {opp_name:<16s} | "
            f"avg payoff: {avg_reward:+.4f} | "
            f"running avg: {np.mean(all_episode_rewards):+.4f} | "
            f"lr: {current_lr:.2e}"
        )

        if (episode + 1) % checkpoint_every == 0:
            ckpt_path = save_dir / f"ep{episode + 1}.pt"
            agent.save(str(ckpt_path))
            save_log()
            save_plot()
            print(f"  Checkpoint saved: {ckpt_path}")

    agent.save(str(final_path))
    save_log()
    save_plot()
    print(f"\nTraining complete. Final model saved to {final_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Actor-Critic agent")
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Model name (e.g. ac_pure). Saves to models/<name>/",
    )
    parser.add_argument(
        "--lambda-kl",
        type=float,
        default=0.0,
        help="Max KL regularization weight (0.0 = pure, 0.5 = KL variant)",
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--hands", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument(
        "--ppo",
        action="store_true",
        default=True,
        dest="ppo",
        help="Use PPO with batched rollouts (default)",
    )
    parser.add_argument(
        "--no-ppo",
        action="store_false",
        dest="ppo",
        help="Use A2C per-hand updates instead of PPO",
    )
    parser.add_argument("--rollout-size", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--aux-coef", type=float, default=0.5)
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
        ppo=args.ppo,
        rollout_size=args.rollout_size,
        ppo_epochs=args.ppo_epochs,
        clip_eps=args.clip_eps,
        aux_coef=args.aux_coef,
        device=device,
    )


if __name__ == "__main__":
    main()
