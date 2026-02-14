"""
Evaluation and Visualisation for trained SAC-LTC / SAC-LFM DSA agents.

Usage:
    python evaluate.py --checkpoint results/checkpoint_final.pt [--agent sac_ltc]
    python evaluate.py --checkpoint results/checkpoint_final.pt --agent sac_lfm

Produces:
  1. Console metrics (success rate, collision rate, average reward).
  2. Learning curve plot  (reward vs training step).
  3. Evaluation bar chart (success / collision rates).
"""

import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from dsa_env import DSAEnv
from sac_agent import SACAgent


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SAC-LTC/SAC-LFM on DSA environment")
    p.add_argument("--agent", type=str, default="sac_ltc",
                   choices=["sac_ltc", "sac_lfm"],
                   help="Agent type: sac_ltc (proposed) or sac_lfm (baseline)")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    p.add_argument("--curves", type=str, default=None,
                   help="Path to training_curves.json (auto-detected if omitted)")
    p.add_argument("--num_episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output_dir", type=str, default="results")

    # Environment / model shape (must match training config)
    p.add_argument("--num_channels", type=int, default=10)
    p.add_argument("--sequence_length", type=int, default=16)
    p.add_argument("--num_features", type=int, default=3)
    p.add_argument("--max_episode_steps", type=int, default=200)
    p.add_argument("--model_dim", type=int, default=128)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--num_blocks", type=int, default=3)
    p.add_argument("--num_heads", type=int, default=4)

    return p.parse_args()


# ======================================================================
# Evaluation
# ======================================================================

def run_evaluation(env: DSAEnv, agent: SACAgent, num_episodes: int):
    """Run evaluation episodes and collect detailed metrics."""
    episode_rewards = []
    episode_successes = []
    episode_collisions = []
    episode_lengths = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        ep_success = 0
        ep_collision = 0
        ep_len = 0
        done = False

        while not done:
            action = agent.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            ep_success += int(info["success"])
            ep_collision += int(info["collision"])
            ep_len += 1
            state = next_state

        episode_rewards.append(ep_reward)
        episode_successes.append(ep_success)
        episode_collisions.append(ep_collision)
        episode_lengths.append(ep_len)

    episode_rewards = np.array(episode_rewards)
    episode_successes = np.array(episode_successes)
    episode_collisions = np.array(episode_collisions)
    episode_lengths = np.array(episode_lengths)

    # Per-step rates
    total_steps = episode_lengths.sum()
    success_rate = episode_successes.sum() / total_steps
    collision_rate = episode_collisions.sum() / total_steps

    return {
        "episode_rewards": episode_rewards,
        "mean_reward": float(episode_rewards.mean()),
        "std_reward": float(episode_rewards.std()),
        "median_reward": float(np.median(episode_rewards)),
        "success_rate": float(success_rate),
        "collision_rate": float(collision_rate),
        "mean_episode_length": float(episode_lengths.mean()),
    }


# ======================================================================
# Plotting
# ======================================================================

def plot_learning_curve(curves_path: str, output_path: str):
    """Plot training reward vs step from saved curves JSON."""
    with open(curves_path) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: training reward
    ax = axes[0]
    ax.plot(data["steps"], data["rewards"], linewidth=1.2, label="Training (rolling avg)")
    if data.get("eval_steps"):
        ax.plot(data["eval_steps"], data["eval_rewards"], "o-", linewidth=1.5,
                markersize=4, label="Evaluation")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("SAC-LTC  —  Learning Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: losses
    ax = axes[1]
    losses = data.get("losses", {})
    for key in ("critic1", "critic2", "actor"):
        if key in losses and losses[key]:
            ax.plot(data["steps"][:len(losses[key])], losses[key], linewidth=0.9, label=key)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Losses")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Learning curve saved → {output_path}")


def plot_eval_metrics(metrics: dict, output_path: str):
    """Bar chart of success rate vs collision rate."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: reward distribution
    ax = axes[0]
    ax.hist(metrics["episode_rewards"], bins=20, edgecolor="black", alpha=0.75)
    ax.axvline(metrics["mean_reward"], color="red", linestyle="--", label=f"Mean = {metrics['mean_reward']:.1f}")
    ax.set_xlabel("Episode Reward")
    ax.set_ylabel("Count")
    ax.set_title("Evaluation Reward Distribution")
    ax.legend()

    # Right: rates
    ax = axes[1]
    labels = ["Success Rate", "Collision Rate"]
    values = [metrics["success_rate"] * 100, metrics["collision_rate"] * 100]
    colours = ["#2ecc71", "#e74c3c"]
    bars = ax.bar(labels, values, color=colours, edgecolor="black", width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", fontweight="bold")
    ax.set_ylabel("Rate (%)")
    ax.set_title("DSA Performance Metrics")
    ax.set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Evaluation metrics plot saved → {output_path}")


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

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
    )

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    agent.load(args.checkpoint)

    # Run evaluation
    print(f"Running {args.num_episodes} evaluation episodes ...")
    metrics = run_evaluation(env, agent, args.num_episodes)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Episodes          : {args.num_episodes}")
    print(f"  Mean Reward       : {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"  Median Reward     : {metrics['median_reward']:.2f}")
    print(f"  Success Rate      : {metrics['success_rate']:.2%}")
    print(f"  Collision Rate    : {metrics['collision_rate']:.2%}")
    print(f"  Mean Ep. Length   : {metrics['mean_episode_length']:.1f}")
    print("=" * 50)

    # Save metrics
    save_metrics = {k: v for k, v in metrics.items() if k != "episode_rewards"}
    save_metrics["episode_rewards"] = metrics["episode_rewards"].tolist()
    metrics_path = os.path.join(args.output_dir, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(save_metrics, f, indent=2)
    print(f"Metrics saved → {metrics_path}")

    # Plot evaluation metrics
    eval_plot_path = os.path.join(args.output_dir, "eval_metrics.png")
    plot_eval_metrics(metrics, eval_plot_path)

    # Plot learning curve if training curves available
    curves_path = args.curves
    if curves_path is None:
        # Try to auto-detect in same directory as checkpoint
        candidate = os.path.join(os.path.dirname(args.checkpoint), "training_curves.json")
        if os.path.exists(candidate):
            curves_path = candidate

    if curves_path and os.path.exists(curves_path):
        curve_plot_path = os.path.join(args.output_dir, "learning_curve.png")
        plot_learning_curve(curves_path, curve_plot_path)
    else:
        print("No training_curves.json found; skipping learning curve plot.")


if __name__ == "__main__":
    main()
