"""
Benchmarking harness for SAC-LTC vs. baselines (SAC-LFM, SAC-LSTM, PPO-LSTM).

Orchestrates multi-seed training, evaluation, and metric logging
following NeurIPS/ICLR reproducibility standards.

Usage:
    python benchmark.py --config benchmark_config.yaml
    python benchmark.py --config benchmark_config.yaml --agents sac_ltc sac_lfm
    python benchmark.py --config benchmark_config.yaml --eval-only
"""

import argparse
import json
import os
import time
from collections import defaultdict, deque
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

from dsa_env import DSAEnv

# Optional wandb integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ======================================================================
# Agent factory
# ======================================================================

def make_agent(agent_type: str, agent_params: dict, env_cfg: dict, device: torch.device):
    """Instantiate an agent by type string."""
    num_channels = env_cfg["num_channels"]
    seq_len = env_cfg["sequence_length"]
    num_features = env_cfg["num_features"]
    input_dim = num_channels * num_features
    state_shape = (seq_len, input_dim)

    if agent_type == "sac_lfm":
        from sac_agent import SACAgent
        return SACAgent(
            state_shape=state_shape,
            num_actions=num_channels,
            input_dim=input_dim,
            device=device,
            max_seq_len=seq_len,
            **agent_params,
        )

    elif agent_type == "sac_lstm":
        from sac_lstm_agent import SACLSTMAgent
        return SACLSTMAgent(
            state_shape=state_shape,
            num_actions=num_channels,
            input_dim=input_dim,
            device=device,
            **agent_params,
        )

    elif agent_type == "sac_ltc":
        from sac_ltc_agent import SACLTCAgent
        return SACLTCAgent(
            state_shape=state_shape,
            num_actions=num_channels,
            input_dim=input_dim,
            device=device,
            **agent_params,
        )

    elif agent_type == "ppo_lstm":
        from ppo_lstm_agent import PPOLSTMAgent
        env_kwargs = {
            "num_channels": num_channels,
            "sequence_length": seq_len,
            "num_features": num_features,
            "pu_on_prob": env_cfg.get("pu_on_prob", 0.3),
            "pu_off_prob": env_cfg.get("pu_off_prob", 0.5),
            "noise_std": env_cfg.get("noise_std", 0.1),
            "max_steps": env_cfg.get("max_steps", 200),
        }
        return PPOLSTMAgent(
            env_kwargs=env_kwargs,
            device="cpu" if device == torch.device("cpu") else "auto",
            **agent_params,
        )

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


# ======================================================================
# Evaluation
# ======================================================================

def evaluate_agent(
    agent,
    agent_type: str,
    env: DSAEnv,
    num_episodes: int,
) -> Dict[str, Any]:
    """Run deterministic evaluation and compute all metrics."""
    episode_rewards = []
    episode_successes = []
    episode_collisions = []
    episode_lengths = []
    inference_times = []
    per_step_throughputs: List[float] = []   # bits/slot for spectral efficiency proxy

    for _ in range(num_episodes):
        state, _ = env.reset()
        if agent_type == "ppo_lstm":
            agent.reset_eval_state()

        ep_reward = 0.0
        ep_success = 0
        ep_collision = 0
        ep_len = 0
        done = False

        while not done:
            t0 = time.perf_counter()
            action = agent.select_action(state, deterministic=True)
            t1 = time.perf_counter()
            inference_times.append(t1 - t0)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            ep_success += int(info["success"])
            ep_collision += int(info["collision"])
            ep_len += 1
            # Spectral efficiency proxy: 1 bit/slot on success, 0 on collision
            per_step_throughputs.append(1.0 if info["success"] else 0.0)
            state = next_state

        episode_rewards.append(ep_reward)
        episode_successes.append(ep_success)
        episode_collisions.append(ep_collision)
        episode_lengths.append(ep_len)

    episode_rewards = np.array(episode_rewards)
    total_steps = sum(episode_lengths)
    success_rate = sum(episode_successes) / max(total_steps, 1)
    collision_rate = sum(episode_collisions) / max(total_steps, 1)

    # Jain's Fairness Index over per-episode throughput
    per_ep_throughput = np.array(episode_successes, dtype=float) / np.array(
        episode_lengths, dtype=float
    )
    n = len(per_ep_throughput)
    jain_fairness = (per_ep_throughput.sum() ** 2) / (n * (per_ep_throughput ** 2).sum() + 1e-12)

    return {
        "mean_reward": float(episode_rewards.mean()),
        "std_reward": float(episode_rewards.std()),
        "median_reward": float(np.median(episode_rewards)),
        "success_rate": float(success_rate),
        "collision_rate": float(collision_rate),
        "spectral_efficiency": float(np.mean(per_step_throughputs)),
        "jain_fairness": float(jain_fairness),
        "mean_inference_ms": float(np.mean(inference_times) * 1000),
        "std_inference_ms": float(np.std(inference_times) * 1000),
        "p99_inference_ms": float(np.percentile(inference_times, 99) * 1000),
        "episode_rewards": episode_rewards.tolist(),
    }


# ======================================================================
# Training loop for off-policy agents (SAC-LFM, SAC-LSTM)
# ======================================================================

def train_off_policy(
    agent,
    env: DSAEnv,
    total_steps: int,
    log_interval: int,
    eval_interval: int,
    eval_episodes: int,
    run_dir: str,
    wandb_run=None,
) -> Dict[str, list]:
    """Standard off-policy training loop. Returns training curves."""
    state, _ = env.reset()
    episode_reward = 0.0
    episode_count = 0
    recent_rewards: deque = deque(maxlen=20)

    curves: Dict[str, list] = defaultdict(list)
    start = time.time()

    for step in range(1, total_steps + 1):
        if step < agent.learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.replay_buffer.push(state, action, reward, next_state, terminated)
        state = next_state
        episode_reward += reward

        if done:
            recent_rewards.append(episode_reward)
            episode_count += 1
            state, _ = env.reset()
            episode_reward = 0.0

        losses = agent.update()

        # Periodic logging
        if step % log_interval == 0 and losses:
            mean_r = float(np.mean(recent_rewards)) if recent_rewards else 0.0
            curves["step"].append(step)
            curves["mean_reward"].append(mean_r)
            for k, v in losses.items():
                curves[k].append(v)

            if wandb_run:
                wandb_run.log({"step": step, "mean_reward": mean_r, **losses})

            elapsed = time.time() - start
            print(
                f"  Step {step:>7d} | Ep {episode_count:>4d} | "
                f"R {mean_r:>7.2f} | alpha {losses.get('alpha', 0):.4f} | "
                f"{elapsed:.0f}s"
            )

        # Periodic eval
        if step % eval_interval == 0:
            eval_m = evaluate_agent(agent, "sac", env, num_episodes=10)
            curves["eval_step"].append(step)
            curves["eval_reward"].append(eval_m["mean_reward"])
            if wandb_run:
                wandb_run.log({
                    "eval/step": step,
                    "eval/mean_reward": eval_m["mean_reward"],
                    "eval/success_rate": eval_m["success_rate"],
                })

    # Save checkpoint
    ckpt_path = os.path.join(run_dir, "checkpoint_final.pt")
    agent.save(ckpt_path)
    return dict(curves)


# ======================================================================
# Training loop for PPO-LSTM (on-policy, uses SB3 internally)
# ======================================================================

def train_ppo(
    agent,
    total_steps: int,
    run_dir: str,
    wandb_run=None,
) -> Dict[str, list]:
    """
    PPO training is managed by SB3 internally.
    We call agent.learn() which handles rollout + optimization.
    """
    print(f"  PPO-LSTM training for {total_steps} steps (SB3 internal loop) ...")
    agent.learn(total_timesteps=total_steps, log_interval=10)

    ckpt_path = os.path.join(run_dir, "checkpoint_final")
    agent.save(ckpt_path)

    # SB3 doesn't expose per-step curves easily; return empty
    return {"note": "PPO curves managed by SB3 logger"}


# ======================================================================
# Main benchmark driver
# ======================================================================

def run_benchmark(config: dict, agent_filter: List[str] | None = None, eval_only: bool = False):
    glob = config["global"]
    seeds = config["seeds"]
    env_cfg = config["environment"]
    agents_cfg = config["agents"]

    device_str = glob["device"]
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Device: {device}")

    output_root = glob["output_dir"]
    os.makedirs(output_root, exist_ok=True)

    all_results: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for agent_name, agent_cfg in agents_cfg.items():
        if agent_filter and agent_name not in agent_filter:
            continue

        agent_type = agent_cfg["class"]
        agent_params = agent_cfg.get("params", {})
        print(f"\n{'='*60}")
        print(f"Agent: {agent_name}  ({agent_type})")
        print(f"{'='*60}")

        for seed in seeds:
            run_id = f"{agent_name}_seed{seed}"
            run_dir = os.path.join(output_root, run_id)
            os.makedirs(run_dir, exist_ok=True)

            print(f"\n--- {run_id} ---")
            np.random.seed(seed)
            torch.manual_seed(seed)

            # wandb init (optional)
            wandb_run = None
            if WANDB_AVAILABLE and not eval_only:
                try:
                    wandb_run = wandb.init(
                        project="sac-lfm-dsa",
                        name=run_id,
                        config={"agent": agent_name, "seed": seed, **agent_params, **env_cfg},
                        reinit=True,
                    )
                except Exception:
                    wandb_run = None

            # Build agent
            agent = make_agent(agent_type, agent_params, env_cfg, device)

            # ---------- Training ----------
            if not eval_only:
                if agent_type in ("sac_lfm", "sac_lstm", "sac_ltc"):
                    env = DSAEnv(
                        num_channels=env_cfg["num_channels"],
                        sequence_length=env_cfg["sequence_length"],
                        num_features=env_cfg["num_features"],
                        pu_on_prob=env_cfg.get("pu_on_prob", 0.3),
                        pu_off_prob=env_cfg.get("pu_off_prob", 0.5),
                        noise_std=env_cfg.get("noise_std", 0.1),
                        max_steps=env_cfg.get("max_steps", 200),
                    )
                    curves = train_off_policy(
                        agent, env,
                        total_steps=glob["training_steps"],
                        log_interval=glob["log_interval"],
                        eval_interval=glob["eval_interval"],
                        eval_episodes=10,
                        run_dir=run_dir,
                        wandb_run=wandb_run,
                    )
                else:
                    curves = train_ppo(
                        agent,
                        total_steps=glob["training_steps"],
                        run_dir=run_dir,
                        wandb_run=wandb_run,
                    )

                # Save curves
                curves_path = os.path.join(run_dir, "training_curves.json")
                with open(curves_path, "w") as f:
                    json.dump(curves, f)

            else:
                # Load saved checkpoint
                ckpt = os.path.join(run_dir, "checkpoint_final.pt")
                if not os.path.exists(ckpt):
                    ckpt = os.path.join(run_dir, "checkpoint_final.zip")
                if os.path.exists(ckpt):
                    agent.load(ckpt)
                else:
                    print(f"  WARNING: No checkpoint found at {run_dir}, skipping.")
                    continue

            # ---------- Evaluation ----------
            eval_env = DSAEnv(
                num_channels=env_cfg["num_channels"],
                sequence_length=env_cfg["sequence_length"],
                num_features=env_cfg["num_features"],
                pu_on_prob=env_cfg.get("pu_on_prob", 0.3),
                pu_off_prob=env_cfg.get("pu_off_prob", 0.5),
                noise_std=env_cfg.get("noise_std", 0.1),
                max_steps=env_cfg.get("max_steps", 200),
            )
            print(f"  Evaluating ({glob['eval_episodes']} episodes) ...")
            metrics = evaluate_agent(
                agent, agent_type, eval_env, glob["eval_episodes"]
            )
            # Remove large list from summary
            ep_rewards = metrics.pop("episode_rewards")
            metrics["episode_rewards"] = ep_rewards

            for k, v in metrics.items():
                if k != "episode_rewards":
                    all_results[agent_name][k].append(v)

            # Save per-run metrics
            metrics_path = os.path.join(run_dir, "eval_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

            print(
                f"  Result: R={metrics['mean_reward']:.2f}±{metrics['std_reward']:.2f} | "
                f"Success={metrics['success_rate']:.2%} | "
                f"Collision={metrics['collision_rate']:.2%} | "
                f"Infer={metrics['mean_inference_ms']:.3f}ms"
            )

            if wandb_run:
                wandb_run.finish()

    # ---------- Aggregate across seeds ----------
    summary = {}
    for agent_name, metric_lists in all_results.items():
        summary[agent_name] = {}
        for k, vals in metric_lists.items():
            if isinstance(vals[0], (int, float)):
                arr = np.array(vals)
                summary[agent_name][k] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "values": vals,
                }

    summary_path = os.path.join(output_root, "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nBenchmark summary saved → {summary_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Agent':<12} {'Reward':>14} {'Success':>10} {'Collision':>10} "
          f"{'Spec.Eff':>10} {'Jain':>8} {'Infer(ms)':>12}")
    print(f"{'-'*80}")
    for aname, metrics_agg in summary.items():
        def _fmt(key):
            m = metrics_agg.get(key, {})
            return f"{m.get('mean', 0):.3f}±{m.get('std', 0):.3f}"
        print(
            f"{aname:<12} "
            f"{_fmt('mean_reward'):>14} "
            f"{_fmt('success_rate'):>10} "
            f"{_fmt('collision_rate'):>10} "
            f"{_fmt('spectral_efficiency'):>10} "
            f"{_fmt('jain_fairness'):>8} "
            f"{_fmt('mean_inference_ms'):>12}"
        )
    print(f"{'='*80}")

    return summary


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="SAC-LFM Benchmark Harness")
    parser.add_argument("--config", type=str, default="benchmark_config.yaml")
    parser.add_argument("--agents", nargs="*", default=None,
                        help="Subset of agents to run (e.g. sac_lfm sac_lstm)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, evaluate saved checkpoints")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_benchmark(config, agent_filter=args.agents, eval_only=args.eval_only)


if __name__ == "__main__":
    main()
