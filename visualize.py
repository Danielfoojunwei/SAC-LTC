"""
Publication-quality visualisation for SAC-LTC benchmark results.

Reads per-run training curves and eval metrics from the benchmark output
directory and produces:
  1. Learning curves (reward ± std across seeds)
  2. LaTeX-formatted performance table
  3. Ablation / bar-chart comparison

Usage:
    python visualize.py --results_dir benchmark_results
    python visualize.py --results_dir benchmark_results --format pdf
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ---- Consistent publication style ----
AGENT_STYLES = {
    "sac_ltc":  {"color": "#7c3aed", "label": "SAC-LTC (Ours)",  "ls": "-",  "marker": "D"},
    "sac_lfm":  {"color": "#2563eb", "label": "SAC-LFM",         "ls": "--", "marker": "o"},
    "sac_lstm": {"color": "#dc2626", "label": "SAC-LSTM",        "ls": "-.", "marker": "s"},
    "ppo_lstm": {"color": "#16a34a", "label": "PPO-LSTM",        "ls": ":",  "marker": "^"},
}
DEFAULT_STYLE = {"color": "#6b7280", "label": "Unknown", "ls": ":", "marker": "x"}


def _style(name: str) -> dict:
    return AGENT_STYLES.get(name, {**DEFAULT_STYLE, "label": name})


# ======================================================================
# Data loading
# ======================================================================

def load_all_curves(results_dir: str) -> Dict[str, Dict[int, dict]]:
    """
    Returns:
        {agent_name: {seed: {"step": [...], "mean_reward": [...], ...}}}
    """
    data: Dict[str, Dict[int, dict]] = defaultdict(dict)
    for entry in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, entry)
        if not os.path.isdir(path):
            continue
        curves_file = os.path.join(path, "training_curves.json")
        if not os.path.exists(curves_file):
            continue
        # Parse agent_name and seed from directory name: e.g. sac_lfm_seed1
        parts = entry.rsplit("_seed", 1)
        if len(parts) != 2:
            continue
        agent_name = parts[0]
        seed = int(parts[1])
        with open(curves_file) as f:
            data[agent_name][seed] = json.load(f)
    return dict(data)


def load_all_eval(results_dir: str) -> Dict[str, Dict[int, dict]]:
    """
    Returns:
        {agent_name: {seed: {metric: value, ...}}}
    """
    data: Dict[str, Dict[int, dict]] = defaultdict(dict)
    for entry in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, entry)
        if not os.path.isdir(path):
            continue
        eval_file = os.path.join(path, "eval_metrics.json")
        if not os.path.exists(eval_file):
            continue
        parts = entry.rsplit("_seed", 1)
        if len(parts) != 2:
            continue
        agent_name = parts[0]
        seed = int(parts[1])
        with open(eval_file) as f:
            data[agent_name][seed] = json.load(f)
    return dict(data)


def load_summary(results_dir: str) -> dict:
    path = os.path.join(results_dir, "benchmark_summary.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ======================================================================
# 1. Learning Curves
# ======================================================================

def plot_learning_curves(
    curves_data: Dict[str, Dict[int, dict]],
    output_path: str,
):
    """
    Plot mean reward ± 1 std across seeds for each agent.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for agent_name, seed_curves in curves_data.items():
        # Collect per-seed reward arrays, aligning by step
        all_steps = []
        all_rewards = []
        for seed, c in sorted(seed_curves.items()):
            steps = c.get("step", c.get("steps", []))
            rewards = c.get("mean_reward", c.get("rewards", []))
            if not steps or not rewards:
                continue
            all_steps.append(np.array(steps))
            all_rewards.append(np.array(rewards))

        if not all_rewards:
            continue

        # Interpolate onto common step grid (union of all step arrays)
        min_len = min(len(r) for r in all_rewards)
        common_steps = all_steps[0][:min_len]
        aligned = np.array([r[:min_len] for r in all_rewards])

        mean = aligned.mean(axis=0)
        std = aligned.std(axis=0)

        s = _style(agent_name)
        ax.plot(common_steps, mean, color=s["color"], ls=s["ls"],
                linewidth=1.8, label=s["label"])
        ax.fill_between(common_steps, mean - std, mean + std,
                        color=s["color"], alpha=0.15)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Mean Episode Reward", fontsize=12)
    ax.set_title("Learning Curves  (mean ± 1 std over 5 seeds)", fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Learning curves → {output_path}")


# ======================================================================
# 2. Performance Bar Charts
# ======================================================================

def plot_performance_bars(
    eval_data: Dict[str, Dict[int, dict]],
    output_path: str,
):
    """Bar chart comparing success rate, collision rate, spectral efficiency."""
    metrics_to_plot = [
        ("success_rate", "Success Rate", "%"),
        ("collision_rate", "Collision Rate", "%"),
        ("spectral_efficiency", "Spectral Efficiency", "bits/slot"),
        ("mean_inference_ms", "Inference Latency", "ms"),
    ]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(4 * len(metrics_to_plot), 5))
    if len(metrics_to_plot) == 1:
        axes = [axes]

    agent_names = list(eval_data.keys())

    for ax, (metric, title, unit) in zip(axes, metrics_to_plot):
        means = []
        stds = []
        colours = []
        labels = []
        for aname in agent_names:
            seed_vals = [
                v.get(metric, 0) for v in eval_data[aname].values()
                if isinstance(v.get(metric, None), (int, float))
            ]
            if not seed_vals:
                continue
            arr = np.array(seed_vals)
            scale = 100 if unit == "%" else 1
            means.append(arr.mean() * scale)
            stds.append(arr.std() * scale)
            colours.append(_style(aname)["color"])
            labels.append(_style(aname)["label"])

        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, width=0.55,
                      color=colours, edgecolor="black", linewidth=0.5,
                      capsize=4, error_kw={"linewidth": 1.2})
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
        ax.set_ylabel(f"{title} ({unit})", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)

        for bar, m in zip(bars, means):
            fmt = f"{m:.1f}" if unit in ("%", "ms") else f"{m:.3f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    fmt, ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Performance bars → {output_path}")


# ======================================================================
# 3. LaTeX Table
# ======================================================================

def generate_latex_table(summary: dict, output_path: str):
    """Generate a LaTeX-formatted results table."""
    metrics = [
        ("mean_reward", "Reward", "{:.1f}"),
        ("success_rate", "Success Rate", "{:.2%}"),
        ("collision_rate", "Collision Rate", "{:.2%}"),
        ("spectral_efficiency", "Spec. Eff.", "{:.3f}"),
        ("jain_fairness", "Jain's FI", "{:.3f}"),
        ("mean_inference_ms", "Latency (ms)", "{:.2f}"),
    ]

    # Determine best (highest/lowest) for bold formatting
    # Higher is better for reward, success, spec eff, jain; lower for collision, latency
    higher_better = {"mean_reward", "success_rate", "spectral_efficiency", "jain_fairness"}

    agent_names = list(summary.keys())

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Benchmark results (mean $\pm$ std over 5 seeds).}")
    lines.append(r"\label{tab:benchmark}")
    col_spec = "l" + "c" * len(metrics)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    header = "Agent & " + " & ".join(m[1] for m in metrics) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for aname in agent_names:
        agg = summary.get(aname, {})
        cells = [_style(aname)["label"]]
        for mkey, _, fmt in metrics:
            m = agg.get(mkey, {})
            mean_val = m.get("mean", 0)
            std_val = m.get("std", 0)

            # Find best agent for this metric
            best_val = None
            for an in agent_names:
                v = summary.get(an, {}).get(mkey, {}).get("mean", None)
                if v is None:
                    continue
                if best_val is None:
                    best_val = v
                elif mkey in higher_better:
                    best_val = max(best_val, v)
                else:
                    best_val = min(best_val, v)

            cell = fmt.format(mean_val) + r" {\scriptsize$\pm$ " + fmt.format(std_val) + "}"
            if best_val is not None and abs(mean_val - best_val) < 1e-9:
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)

        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(latex)
    print(f"LaTeX table → {output_path}")
    print(latex)


# ======================================================================
# 4. Ablation study plot (reward vs num_blocks for SAC-LFM)
# ======================================================================

def plot_ablation(
    eval_data: Dict[str, Dict[int, dict]],
    output_path: str,
):
    """
    If multiple SAC-LTC variants are present (e.g. sac_ltc_layers1,
    sac_ltc_layers2, ...) plot an ablation chart.
    Falls back to a comparison of all agents if no ablation data exists.
    """
    ablation_agents = {k: v for k, v in eval_data.items() if k.startswith("sac_ltc")}
    if len(ablation_agents) <= 1:
        # No ablation variants — plot reward distribution comparison instead
        fig, ax = plt.subplots(figsize=(8, 5))
        all_labels = []
        all_rewards = []
        all_colours = []
        for aname, seeds in eval_data.items():
            rewards = []
            for sv in seeds.values():
                rewards.extend(sv.get("episode_rewards", []))
            if rewards:
                all_labels.append(_style(aname)["label"])
                all_rewards.append(rewards)
                all_colours.append(_style(aname)["color"])

        bp = ax.boxplot(all_rewards, patch_artist=True, labels=all_labels)
        for patch, colour in zip(bp["boxes"], all_colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.6)

        ax.set_ylabel("Episode Reward", fontsize=11)
        ax.set_title("Reward Distribution Across Seeds", fontsize=12)
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Ablation / distribution plot → {output_path}")
        return

    # True ablation: extract block count from name and plot
    fig, ax = plt.subplots(figsize=(6, 4))
    block_counts = []
    mean_rewards = []
    std_rewards = []

    for aname in sorted(ablation_agents):
        seeds = ablation_agents[aname]
        vals = [v["mean_reward"] for v in seeds.values() if "mean_reward" in v]
        if not vals:
            continue
        # Try to extract block count from name
        parts = aname.split("blocks")
        if len(parts) == 2 and parts[1].isdigit():
            block_counts.append(int(parts[1]))
        else:
            block_counts.append(len(block_counts))
        mean_rewards.append(np.mean(vals))
        std_rewards.append(np.std(vals))

    ax.errorbar(block_counts, mean_rewards, yerr=std_rewards,
                fmt="o-", capsize=5, linewidth=2, markersize=8,
                color=_style("sac_ltc")["color"])
    ax.set_xlabel("Number of LTC Layers", fontsize=11)
    ax.set_ylabel("Mean Reward", fontsize=11)
    ax.set_title("Ablation: LTC Depth", fontsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Ablation plot → {output_path}")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Visualise SAC-LTC benchmark results")
    parser.add_argument("--results_dir", type=str, default="benchmark_results")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for plots (defaults to results_dir/figures)")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    out = args.output_dir or os.path.join(args.results_dir, "figures")
    os.makedirs(out, exist_ok=True)

    ext = args.format

    curves = load_all_curves(args.results_dir)
    evals = load_all_eval(args.results_dir)
    summary = load_summary(args.results_dir)

    if curves:
        plot_learning_curves(curves, os.path.join(out, f"learning_curves.{ext}"))
    else:
        print("No training curves found — skipping learning curve plot.")

    if evals:
        plot_performance_bars(evals, os.path.join(out, f"performance_bars.{ext}"))
        plot_ablation(evals, os.path.join(out, f"ablation.{ext}"))
    else:
        print("No eval metrics found — skipping performance plots.")

    if summary:
        generate_latex_table(summary, os.path.join(out, "results_table.tex"))
    else:
        print("No benchmark summary found — skipping LaTeX table.")

    print(f"\nAll figures saved to: {out}")


if __name__ == "__main__":
    main()
