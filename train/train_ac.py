from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from agents.ac_agent import ActorCriticAgent
from env.env import PokerEnv
from evaluation.evaluator import Evaluator
from players.ac_player import ActorCriticPlayer
from players.opponents import make_opponent, make_random_parameterized
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


VAL_OPPONENTS = ["calling_station", "maniac", "old_man_coffee", "polarizing"]


def run_validation(
    agent: ActorCriticAgent, env: PokerEnv, num_hands: int
) -> dict[str, float]:
    player = ActorCriticPlayer(agent=agent)
    results: dict[str, float] = {}
    for opp_name in VAL_OPPONENTS:
        opp = make_opponent(opp_name)
        evaluator = Evaluator(env=env, player0=player, player1=opp)
        rewards = evaluator.run_matchup(num_episodes=num_hands)
        results[opp_name] = float(rewards.mean())
    return results


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
    val_every: int,
    use_hardcoded: bool,
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

    val_episodes: list[int] = []
    val_results: dict[str, list[float]] = {f"val_{o}": [] for o in VAL_OPPONENTS}
    if log_path.exists():
        old = np.load(log_path, allow_pickle=True)
        if "val_episodes" in old:
            val_episodes = old["val_episodes"].tolist()
            for opp in VAL_OPPONENTS:
                key = f"val_{opp}"
                if key in old:
                    val_results[key] = old[key].tolist()

    n_prior = len(all_episode_rewards)
    ckpt_path = save_dir / f"ep{n_prior}.pt"
    if ckpt_path.exists():
        print(f"Resuming from {ckpt_path}")
        agent.load(str(ckpt_path))
    elif final_path.exists():
        print(f"Resuming from {final_path}")
        agent.load(str(final_path))

    env = PokerEnv()
    hardcoded_names = ["calling_station", "maniac", "old_man_coffee", "polarizing"]

    def save_log() -> None:
        save_dict = {
            "episode_rewards": np.array(all_episode_rewards),
            "early_rewards": np.array(all_early_rewards),
            "late_rewards": np.array(all_late_rewards),
            "opponent_names": np.array(all_opponent_names),
            "val_episodes": np.array(val_episodes),
        }
        for key, vals in val_results.items():
            save_dict[key] = np.array(vals)
        np.savez(log_path, **save_dict)

    def save_plot() -> None:
        early = np.array(all_early_rewards) if all_early_rewards else None
        late = np.array(all_late_rewards) if all_late_rewards else None
        vd = None
        if val_episodes:
            vd = {"episodes": val_episodes}
            vd.update(val_results)
        plot_training(
            np.array(all_episode_rewards),
            np.array(all_opponent_names),
            name,
            plot_path,
            early,
            late,
            vd,
        )

    mode = "hardcoded" if use_hardcoded else "parameterized"
    print(
        f"PPO training ({mode} opponents): rollout={rollout_size}, "
        f"epochs={ppo_epochs}, clip={clip_eps}, lr={lr}, entropy={entropy_coef}, "
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
        if use_hardcoded:
            opp_name = pick_opponent(hardcoded_names, opp_counts)
            opponent = make_opponent(opp_name)
        elif random.random() < 0.5:
            opp_name = random.choice(hardcoded_names)
            opponent = make_opponent(opp_name)
        else:
            opponent, opp_name = make_random_parameterized()

        agent.reset_opponent_state()

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
            f"lr: {lr:.2e}"
        )

        if val_every > 0 and (episode + 1) % val_every == 0:
            with torch.no_grad():
                vr = run_validation(agent, env, hands_per_episode)
            val_episodes.append(episode + 1)
            for opp, payoff in vr.items():
                val_results[f"val_{opp}"].append(payoff)
            val_str = " | ".join(f"{o}: {p:+.3f}" for o, p in vr.items())
            print(f"  [val] {val_str}")

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


def meta_train(
    *,
    name: str,
    num_meta_iters: int,
    meta_batch_size: int,
    hands_per_trial: int,
    checkpoint_every: int,
    lr: float,
    ppo_epochs: int,
    clip_eps: float,
    aux_coef: float,
    entropy_coef: float,
    val_every: int,
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
        extra_critic_steps=0,
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
        print(f"Loaded {len(all_episode_rewards)} prior meta-iterations from log")

    val_episodes: list[int] = []
    val_results: dict[str, list[float]] = {f"val_{o}": [] for o in VAL_OPPONENTS}
    if log_path.exists():
        old = np.load(log_path, allow_pickle=True)
        if "val_episodes" in old:
            val_episodes = old["val_episodes"].tolist()
            for opp in VAL_OPPONENTS:
                key = f"val_{opp}"
                if key in old:
                    val_results[key] = old[key].tolist()

    n_prior = len(all_episode_rewards)
    ckpt_path = save_dir / f"ep{n_prior}.pt"
    if ckpt_path.exists():
        print(f"Resuming from {ckpt_path}")
        agent.load(str(ckpt_path))
    elif final_path.exists():
        print(f"Resuming from {final_path}")
        agent.load(str(final_path))

    env = PokerEnv()
    hardcoded_names = ["calling_station", "maniac", "old_man_coffee", "polarizing"]

    def save_log() -> None:
        save_dict = {
            "episode_rewards": np.array(all_episode_rewards),
            "early_rewards": np.array(all_early_rewards),
            "late_rewards": np.array(all_late_rewards),
            "opponent_names": np.array(all_opponent_names),
            "val_episodes": np.array(val_episodes),
        }
        for key, vals in val_results.items():
            save_dict[key] = np.array(vals)
        np.savez(log_path, **save_dict)

    def save_plot() -> None:
        early = np.array(all_early_rewards) if all_early_rewards else None
        late = np.array(all_late_rewards) if all_late_rewards else None
        vd = None
        if val_episodes:
            vd = {"episodes": val_episodes}
            vd.update(val_results)
        plot_training(
            np.array(all_episode_rewards),
            np.array(all_opponent_names),
            name,
            plot_path,
            early,
            late,
            vd,
        )

    print(
        f"Meta-batched PPO: batch_size={meta_batch_size}, "
        f"hands/trial={hands_per_trial}, epochs={ppo_epochs}, "
        f"lr={lr}, entropy={entropy_coef}, aux={aux_coef}"
    )

    start_iter = len(all_episode_rewards)
    remaining = num_meta_iters - start_iter
    if remaining <= 0:
        print(f"Already have {start_iter}/{num_meta_iters} iters. Nothing to do.")
        return
    print(f"Meta-iters {start_iter + 1} to {num_meta_iters} ({remaining} remaining)")

    total_hands = meta_batch_size * hands_per_trial
    for meta_iter in range(start_iter, num_meta_iters):
        all_trials: list[list] = []
        iter_rewards: list[float] = []
        iter_early: list[float] = []
        iter_late: list[float] = []
        trial_opp_names: list[str] = []

        pbar = trange(
            total_hands,
            desc=f"Meta {meta_iter + 1}/{num_meta_iters}",
            leave=False,
        )
        for _trial in range(meta_batch_size):
            if random.random() < 0.5:
                opp_name = random.choice(hardcoded_names)
                opponent = make_opponent(opp_name)
            else:
                opponent, opp_name = make_random_parameterized()
            trial_opp_names.append(opp_name)

            agent.reset_opponent_state()
            agent.begin_collect()
            trial_rewards: list[float] = []
            for _ in range(hands_per_trial):
                payoff = play_hand(env, agent, opponent)
                agent.finish_hand_collect()
                trial_rewards.append(payoff)
                pbar.update(1)
            all_trials.append(agent.get_trial())

            iter_rewards.extend(trial_rewards)
            mid = len(trial_rewards) // 2
            iter_early.extend(trial_rewards[:mid])
            iter_late.extend(trial_rewards[mid:])

        pbar.close()
        diag = agent.meta_ppo_update(all_trials)

        avg_reward = float(np.mean(iter_rewards))
        all_episode_rewards.append(avg_reward)
        all_early_rewards.append(float(np.mean(iter_early)))
        all_late_rewards.append(float(np.mean(iter_late)))
        all_opponent_names.append("meta_mix")

        diag_str = ""
        if diag:
            diag_str = (
                f" | policy={diag['policy_loss']:.4f} "
                f"value={diag['value_loss']:.4f} "
                f"entropy={diag['entropy']:.4f} "
                f"aux={diag['aux_loss']:.4f}"
            )
        print(
            f"Meta-iter {meta_iter + 1:3d} | "
            f"avg payoff: {avg_reward:+.4f} | "
            f"running avg: {np.mean(all_episode_rewards):+.4f}"
            f"{diag_str}"
        )

        if val_every > 0 and (meta_iter + 1) % val_every == 0:
            with torch.no_grad():
                vr = run_validation(agent, env, hands_per_trial)
            val_episodes.append(meta_iter + 1)
            for opp, payoff in vr.items():
                val_results[f"val_{opp}"].append(payoff)
            val_str = " | ".join(f"{o}: {p:+.3f}" for o, p in vr.items())
            print(f"  [val] {val_str}")

        if (meta_iter + 1) % checkpoint_every == 0:
            ckpt_path = save_dir / f"ep{meta_iter + 1}.pt"
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
    parser.add_argument("--name", type=str, required=True)
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
    parser.add_argument(
        "--use-hardcoded",
        action="store_true",
        default=False,
        help="Use original 4 hardcoded opponents instead of parameterized",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=5,
        help="Run validation vs hardcoded opponents every N episodes (0 to disable)",
    )
    parser.add_argument("--meta", action="store_true", default=False)
    parser.add_argument("--meta-batch-size", type=int, default=8)
    parser.add_argument("--hands-per-trial", type=int, default=100)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.meta:
        meta_train(
            name=args.name,
            num_meta_iters=args.episodes,
            meta_batch_size=args.meta_batch_size,
            hands_per_trial=args.hands_per_trial,
            checkpoint_every=args.checkpoint_every,
            lr=args.lr,
            ppo_epochs=args.ppo_epochs,
            clip_eps=args.clip_eps,
            aux_coef=args.aux_coef,
            entropy_coef=args.entropy_coef,
            val_every=args.val_every,
            device=device,
        )
    else:
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
            use_hardcoded=args.use_hardcoded,
            val_every=args.val_every,
            device=device,
        )


if __name__ == "__main__":
    main()
