#!/usr/bin/env python
"""
Full benchmark: 4 agents x 3 seeds x 5000 steps.
Runs sequentially, prints progress, generates summary.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"

sys.path.insert(0, os.path.dirname(__file__))

from benchmark import make_agent, evaluate_agent, train_off_policy, train_ppo
from dsa_env import DSAEnv

CONFIG = {
    "training_steps": 5000,
    "eval_episodes": 50,
    "log_interval": 1000,
    "eval_interval": 2500,
    "seeds": [1, 2, 3],
    "env": {
        "num_channels": 10,
        "sequence_length": 16,
        "num_features": 3,
        "pu_on_prob": 0.3,
        "pu_off_prob": 0.5,
        "noise_std": 0.1,
        "max_steps": 200,
    },
    "agents": {
        "sac_lfm": {
            "class": "sac_lfm",
            "params": {
                "model_dim": 64,
                "latent_dim": 64,
                "num_blocks": 2,
                "num_heads": 4,
                "lr": 3e-4,
                "gamma": 0.99,
                "tau": 0.005,
                "buffer_size": 50000,
                "batch_size": 128,
                "learning_starts": 500,
            },
        },
        "sac_ltc": {
            "class": "sac_ltc",
            "params": {
                "hidden_dim": 64,
                "latent_dim": 64,
                "num_layers": 2,
                "dt": 1.0,
                "lr": 3e-4,
                "gamma": 0.99,
                "tau": 0.005,
                "buffer_size": 50000,
                "batch_size": 128,
                "learning_starts": 500,
            },
        },
        "sac_lstm": {
            "class": "sac_lstm",
            "params": {
                "hidden_dim": 64,
                "latent_dim": 64,
                "num_layers": 2,
                "lr": 3e-4,
                "gamma": 0.99,
                "tau": 0.005,
                "buffer_size": 50000,
                "batch_size": 128,
                "learning_starts": 500,
            },
        },
        "ppo_lstm": {
            "class": "ppo_lstm",
            "params": {
                "lstm_hidden_size": 64,
                "n_lstm_layers": 1,
                "lr": 3e-4,
                "n_steps": 1024,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
            },
        },
    },
}


def main():
    output_root = "benchmark_results_4agent"
    os.makedirs(output_root, exist_ok=True)
    device = torch.device("cpu")

    env_cfg = CONFIG["env"]
    seeds = CONFIG["seeds"]
    all_results = {}

    total_runs = len(CONFIG["agents"]) * len(seeds)
    run_num = 0

    for agent_name, agent_cfg in CONFIG["agents"].items():
        agent_type = agent_cfg["class"]
        agent_params = agent_cfg["params"]
        all_results[agent_name] = {
            "mean_reward": [], "std_reward": [], "success_rate": [],
            "collision_rate": [], "spectral_efficiency": [],
            "jain_fairness": [], "mean_inference_ms": [], "p99_inference_ms": [],
        }

        print(f"\n{'='*60}", flush=True)
        print(f"AGENT: {agent_name} ({agent_type})", flush=True)
        print(f"{'='*60}", flush=True)

        for seed in seeds:
            run_num += 1
            run_id = f"{agent_name}_seed{seed}"
            run_dir = os.path.join(output_root, run_id)
            os.makedirs(run_dir, exist_ok=True)

            print(f"\n[{run_num}/{total_runs}] {run_id} ...", flush=True)
            np.random.seed(seed)
            torch.manual_seed(seed)

            t_start = time.time()
            agent = make_agent(agent_type, agent_params, env_cfg, device)

            # Train
            if agent_type in ("sac_lfm", "sac_lstm", "sac_ltc"):
                env = DSAEnv(
                    num_channels=env_cfg["num_channels"],
                    sequence_length=env_cfg["sequence_length"],
                    num_features=env_cfg["num_features"],
                    pu_on_prob=env_cfg["pu_on_prob"],
                    pu_off_prob=env_cfg["pu_off_prob"],
                    noise_std=env_cfg["noise_std"],
                    max_steps=env_cfg["max_steps"],
                )
                curves = train_off_policy(
                    agent, env,
                    total_steps=CONFIG["training_steps"],
                    log_interval=CONFIG["log_interval"],
                    eval_interval=CONFIG["eval_interval"],
                    eval_episodes=10,
                    run_dir=run_dir,
                )
            else:
                curves = train_ppo(
                    agent,
                    total_steps=CONFIG["training_steps"],
                    run_dir=run_dir,
                )

            train_time = time.time() - t_start
            print(f"  Training done in {train_time:.1f}s", flush=True)

            # Save curves
            with open(os.path.join(run_dir, "training_curves.json"), "w") as f:
                json.dump(curves if isinstance(curves, dict) else {}, f)

            # Evaluate
            eval_env = DSAEnv(
                num_channels=env_cfg["num_channels"],
                sequence_length=env_cfg["sequence_length"],
                num_features=env_cfg["num_features"],
                pu_on_prob=env_cfg["pu_on_prob"],
                pu_off_prob=env_cfg["pu_off_prob"],
                noise_std=env_cfg["noise_std"],
                max_steps=env_cfg["max_steps"],
            )
            print(f"  Evaluating ({CONFIG['eval_episodes']} episodes) ...", flush=True)
            metrics = evaluate_agent(agent, agent_type, eval_env, CONFIG["eval_episodes"])
            ep_rewards = metrics.pop("episode_rewards")

            # Save
            metrics["episode_rewards"] = ep_rewards
            with open(os.path.join(run_dir, "eval_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

            for k in all_results[agent_name]:
                if k in metrics:
                    all_results[agent_name][k].append(metrics[k])

            eval_time = time.time() - t_start - train_time
            print(
                f"  Result: R={metrics['mean_reward']:.2f}±{metrics['std_reward']:.2f} | "
                f"Success={metrics['success_rate']:.2%} | "
                f"Collision={metrics['collision_rate']:.2%} | "
                f"Infer={metrics['mean_inference_ms']:.3f}ms | "
                f"Total={time.time()-t_start:.1f}s",
                flush=True,
            )

    # ---- Aggregate ----
    summary = {}
    for agent_name, metric_lists in all_results.items():
        summary[agent_name] = {}
        for k, vals in metric_lists.items():
            if vals and isinstance(vals[0], (int, float)):
                arr = np.array(vals)
                summary[agent_name][k] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "values": vals,
                }

    with open(os.path.join(output_root, "benchmark_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Print Table ----
    print(f"\n{'='*96}", flush=True)
    print(f"  EMPIRICAL PERFORMANCE BENCHMARK  (3 seeds x {CONFIG['training_steps']} steps)", flush=True)
    print(f"{'='*96}", flush=True)
    print(f"{'Agent':<12} {'Reward':>16} {'Success%':>12} {'Collision%':>12} "
          f"{'Spec.Eff':>10} {'Jain':>10} {'Infer(ms)':>12}", flush=True)
    print(f"{'-'*96}", flush=True)
    for aname, m in summary.items():
        def f(key):
            d = m.get(key, {})
            return f"{d.get('mean',0):.3f}±{d.get('std',0):.3f}"
        print(
            f"{aname:<12} {f('mean_reward'):>16} {f('success_rate'):>12} "
            f"{f('collision_rate'):>12} {f('spectral_efficiency'):>10} "
            f"{f('jain_fairness'):>10} {f('mean_inference_ms'):>12}",
            flush=True,
        )
    print(f"{'='*96}", flush=True)
    print(f"\nSummary saved -> {output_root}/benchmark_summary.json", flush=True)


if __name__ == "__main__":
    main()
