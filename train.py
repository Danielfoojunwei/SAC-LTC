"""
Training loop for the SAC-LFM Dynamic Spectrum Access agent.

Usage:
    python train.py [--num_steps 50000] [--seed 42] [--device cuda]
"""

import argparse
import json
import os
import time
from collections import deque

import numpy as np
import torch

from dsa_env import DSAEnv
from sac_agent import SACAgent


def parse_args():
    p = argparse.ArgumentParser(description="Train SAC-LFM on DSA environment")

    # Environment
    p.add_argument("--num_channels", type=int, default=10)
    p.add_argument("--sequence_length", type=int, default=16)
    p.add_argument("--num_features", type=int, default=3)
    p.add_argument("--max_episode_steps", type=int, default=200)

    # LFM encoder
    p.add_argument("--model_dim", type=int, default=128)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--num_blocks", type=int, default=3)
    p.add_argument("--num_heads", type=int, default=4)

    # SAC
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--buffer_size", type=int, default=1_000_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--learning_starts", type=int, default=1000)

    # Training
    p.add_argument("--num_steps", type=int, default=50_000)
    p.add_argument("--eval_interval", type=int, default=5000)
    p.add_argument("--log_interval", type=int, default=500)
    p.add_argument("--save_interval", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")

    # Output
    p.add_argument("--output_dir", type=str, default="results")

    return p.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_policy(env: DSAEnv, agent: SACAgent, num_episodes: int = 10):
    """Run deterministic evaluation episodes and return metrics."""
    total_rewards = []
    total_successes = 0
    total_collisions = 0
    total_steps = 0

    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            total_successes += int(info["success"])
            total_collisions += int(info["collision"])
            total_steps += 1
            state = next_state

        total_rewards.append(episode_reward)

    return {
        "mean_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "success_rate": total_successes / max(total_steps, 1),
        "collision_rate": total_collisions / max(total_steps, 1),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Environment
    env = DSAEnv(
        num_channels=args.num_channels,
        sequence_length=args.sequence_length,
        num_features=args.num_features,
        max_steps=args.max_episode_steps,
    )
    input_dim = args.num_channels * args.num_features
    state_shape = (args.sequence_length, input_dim)

    # Agent
    agent = SACAgent(
        state_shape=state_shape,
        num_actions=args.num_channels,
        input_dim=input_dim,
        device=device,
        model_dim=args.model_dim,
        latent_dim=args.latent_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        max_seq_len=args.sequence_length,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
    )

    # Save hyperparameters
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ---- Training loop ----
    state, _ = env.reset()
    episode_reward = 0.0
    episode_count = 0
    recent_rewards = deque(maxlen=20)

    # Logging lists
    log_steps = []
    log_rewards = []
    log_eval_rewards = []
    log_eval_steps = []
    log_losses = {"critic1": [], "critic2": [], "actor": [], "alpha": []}

    start_time = time.time()

    for step in range(1, args.num_steps + 1):
        # Select action (random during warmup, policy after)
        if step < args.learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store transition
        agent.replay_buffer.push(state, action, reward, next_state, terminated)
        state = next_state
        episode_reward += reward

        if done:
            recent_rewards.append(episode_reward)
            episode_count += 1
            state, _ = env.reset()
            episode_reward = 0.0

        # Gradient step
        losses = agent.update()

        # ---- Logging ----
        if step % args.log_interval == 0 and losses:
            elapsed = time.time() - start_time
            mean_r = np.mean(recent_rewards) if recent_rewards else 0.0
            print(
                f"Step {step:>7d} | "
                f"Episodes {episode_count:>5d} | "
                f"Mean Reward {mean_r:>7.2f} | "
                f"α {losses.get('alpha', 0):.4f} | "
                f"Critic Loss {losses.get('critic1_loss', 0):.4f} | "
                f"Elapsed {elapsed:.0f}s"
            )
            log_steps.append(step)
            log_rewards.append(mean_r)
            for key in log_losses:
                log_losses[key].append(losses.get(f"{key}_loss", 0.0))

        # ---- Evaluation ----
        if step % args.eval_interval == 0:
            eval_metrics = evaluate_policy(env, agent, num_episodes=10)
            print(
                f"  [EVAL] Step {step} | "
                f"Reward {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f} | "
                f"Success {eval_metrics['success_rate']:.2%} | "
                f"Collision {eval_metrics['collision_rate']:.2%}"
            )
            log_eval_steps.append(step)
            log_eval_rewards.append(eval_metrics["mean_reward"])

        # ---- Save checkpoint ----
        if step % args.save_interval == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_{step}.pt")
            agent.save(ckpt_path)
            print(f"  Saved checkpoint → {ckpt_path}")

    # Final save
    final_path = os.path.join(args.output_dir, "checkpoint_final.pt")
    agent.save(final_path)
    print(f"Training complete. Final checkpoint → {final_path}")

    # Save training curves
    curves = {
        "steps": log_steps,
        "rewards": log_rewards,
        "eval_steps": log_eval_steps,
        "eval_rewards": log_eval_rewards,
        "losses": log_losses,
    }
    curves_path = os.path.join(args.output_dir, "training_curves.json")
    with open(curves_path, "w") as f:
        json.dump(curves, f)
    print(f"Training curves → {curves_path}")


if __name__ == "__main__":
    main()
