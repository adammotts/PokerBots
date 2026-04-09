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
    num_episodes: int,
    hands_per_episode: int,
    checkpoint_every: int,
    lr: float,
    rollout_size: int,
    ppo_epochs: int,
    clip_eps: float,
    aux_coef: float,
    entropy_coef: float,
    extra_critic_steps: int,
    device: str,
) -> None:
    save_dir = MODELS_DIR / name
    save_dir.mkdir(parents=True, exist_ok=True)

    agent = ActorCriticAgent(
        lr=lr,
        ppo_epochs=ppo_epochs,
        clip_eps=clip_eps,
        aux_coef=aux_coef,
        entropy_coef=entropy_coef,
        extra_critic_steps=extra_critic_steps,
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

    print(
        f"PPO training: rollout={rollout_size}, epochs={ppo_epochs}, "
        f"clip={clip_eps}, lr={lr}, entropy={entropy_coef}, "
        f"aux={aux_coef}, extra_critic={extra_critic_steps}"
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

        current_lr = lr

        episode_rewards: list[float] = []

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
            diag = agent.ppo_update()
            if diag and hand <= rollout_size:
                print(
                    f"  [diag] policy={diag['policy_loss']:.4f} "
                    f"value={diag['value_loss']:.4f} "
                    f"entropy={diag['entropy']:.4f} "
                    f"aux={diag['aux_loss']:.4f}"
                )
        pbar.close()

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
    parser = argparse.ArgumentParser(description="Train Actor-Critic agent (PPO)")
    parser.add_argument(
        "--name",
        type=str,
        required=True,
    )
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--hands", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--rollout-size", type=int, default=512)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--aux-coef", type=float, default=0.3)
    parser.add_argument("--entropy-coef", type=float, default=0.05)
    parser.add_argument("--extra-critic-steps", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train(
        name=args.name,
        num_episodes=args.episodes,
        hands_per_episode=args.hands,
        checkpoint_every=args.checkpoint_every,
        lr=args.lr,
        rollout_size=args.rollout_size,
        ppo_epochs=args.ppo_epochs,
        clip_eps=args.clip_eps,
        aux_coef=args.aux_coef,
        entropy_coef=args.entropy_coef,
        extra_critic_steps=args.extra_critic_steps,
        device=device,
    )


if __name__ == "__main__":
    main()
