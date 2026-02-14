#!/usr/bin/env python
"""Run SAC-LTC only (3 seeds x 5000 steps) and merge with existing results."""
import json
import os
import sys
import time

import numpy as np
import torch

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(__file__))

from benchmark import make_agent, evaluate_agent, train_off_policy
from dsa_env import DSAEnv

ENV_CFG = {
    "num_channels": 10,
    "sequence_length": 16,
    "num_features": 3,
    "pu_on_prob": 0.3,
    "pu_off_prob": 0.5,
    "noise_std": 0.1,
    "max_steps": 200,
}

LTC_PARAMS = {
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
}

SEEDS = [1, 2, 3]
TRAINING_STEPS = 5000
EVAL_EPISODES = 50
OUTPUT_ROOT = "benchmark_results_full"


def main():
    device = torch.device("cpu")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    ltc_results = {
        "mean_reward": [], "std_reward": [], "success_rate": [],
        "collision_rate": [], "spectral_efficiency": [],
        "jain_fairness": [], "mean_inference_ms": [], "p99_inference_ms": [],
    }

    print(f"\n{'='*60}", flush=True)
    print(f"AGENT: sac_ltc (Liquid Time-Constant)", flush=True)
    print(f"{'='*60}", flush=True)

    for i, seed in enumerate(SEEDS):
        run_id = f"sac_ltc_seed{seed}"
        run_dir = os.path.join(OUTPUT_ROOT, run_id)
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n[{i+1}/{len(SEEDS)}] {run_id} ...", flush=True)
        np.random.seed(seed)
        torch.manual_seed(seed)

        t_start = time.time()
        agent = make_agent("sac_ltc", LTC_PARAMS, ENV_CFG, device)

        env = DSAEnv(
            num_channels=ENV_CFG["num_channels"],
            sequence_length=ENV_CFG["sequence_length"],
            num_features=ENV_CFG["num_features"],
            pu_on_prob=ENV_CFG["pu_on_prob"],
            pu_off_prob=ENV_CFG["pu_off_prob"],
            noise_std=ENV_CFG["noise_std"],
            max_steps=ENV_CFG["max_steps"],
        )
        curves = train_off_policy(
            agent, env,
            total_steps=TRAINING_STEPS,
            log_interval=1000,
            eval_interval=2500,
            eval_episodes=10,
            run_dir=run_dir,
        )

        train_time = time.time() - t_start
        print(f"  Training done in {train_time:.1f}s", flush=True)

        with open(os.path.join(run_dir, "training_curves.json"), "w") as f:
            json.dump(curves if isinstance(curves, dict) else {}, f)

        eval_env = DSAEnv(
            num_channels=ENV_CFG["num_channels"],
            sequence_length=ENV_CFG["sequence_length"],
            num_features=ENV_CFG["num_features"],
            pu_on_prob=ENV_CFG["pu_on_prob"],
            pu_off_prob=ENV_CFG["pu_off_prob"],
            noise_std=ENV_CFG["noise_std"],
            max_steps=ENV_CFG["max_steps"],
        )
        print(f"  Evaluating ({EVAL_EPISODES} episodes) ...", flush=True)
        metrics = evaluate_agent(agent, "sac_ltc", eval_env, EVAL_EPISODES)
        ep_rewards = metrics.pop("episode_rewards")

        metrics["episode_rewards"] = ep_rewards
        with open(os.path.join(run_dir, "eval_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        for k in ltc_results:
            if k in metrics:
                ltc_results[k].append(metrics[k])

        print(
            f"  Result: R={metrics['mean_reward']:.2f}+/-{metrics['std_reward']:.2f} | "
            f"Success={metrics['success_rate']:.2%} | "
            f"Collision={metrics['collision_rate']:.2%} | "
            f"Infer={metrics['mean_inference_ms']:.3f}ms | "
            f"Total={time.time()-t_start:.1f}s",
            flush=True,
        )

    # Merge with existing summary
    summary_path = os.path.join(OUTPUT_ROOT, "benchmark_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {}

    summary["sac_ltc"] = {}
    for k, vals in ltc_results.items():
        if vals and isinstance(vals[0], (int, float)):
            arr = np.array(vals)
            summary["sac_ltc"][k] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "values": vals,
            }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print combined table
    print(f"\n{'='*96}", flush=True)
    print(f"  COMBINED EMPIRICAL BENCHMARK  (3 seeds x {TRAINING_STEPS} steps)", flush=True)
    print(f"{'='*96}", flush=True)
    print(f"{'Agent':<12} {'Reward':>16} {'Success%':>12} {'Collision%':>12} "
          f"{'Spec.Eff':>10} {'Jain':>10} {'Infer(ms)':>12}", flush=True)
    print(f"{'-'*96}", flush=True)
    for aname, m in summary.items():
        def f(key):
            d = m.get(key, {})
            return f"{d.get('mean',0):.3f}+/-{d.get('std',0):.3f}"
        print(
            f"{aname:<12} {f('mean_reward'):>16} {f('success_rate'):>12} "
            f"{f('collision_rate'):>12} {f('spectral_efficiency'):>10} "
            f"{f('jain_fairness'):>10} {f('mean_inference_ms'):>12}",
            flush=True,
        )
    print(f"{'='*96}", flush=True)
    print(f"\nMerged summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
