"""
Zero-shot accuracy bar chart: one subplot per benchmark,
grouped bars for FP16 / Uniform RTN+OP / Hybrid RTN+OP / GPTQ,
averaged across all model scales.

Usage:
    python plot_zeroshot_barchart.py
    python plot_zeroshot_barchart.py --output figures/zeroshot_barchart.png
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS_PATH = "results_zeroshot/zeroshot_results.json"

METHODS = [
    ("fp16", "FP16"),
    ("uniform_4bit_rtn_op1.0_deconly", "Uniform RTN+OP"),
    ("hybrid_gamma0.5_4bit_rtn_op1.0_deconly", "Hybrid RTN+OP"),
    ("uniform_4bit_gptq", "GPTQ"),
]

TASKS = ["lambada", "arc_easy", "arc_challenge", "piqa"]
TASK_LABELS = ["LAMBADA", "ARC-Easy", "ARC-Challenge", "PiQA"]

MODELS = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
]
MODEL_LABELS = ["125M", "350M", "1.3B", "2.7B", "6.7B"]

COLORS = ["#2c3e50", "#e74c3c", "#2ecc71", "#3498db"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=RESULTS_PATH)
    parser.add_argument("--output", default="figures/zeroshot_barchart.png")
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)

    n_models = len(MODELS)
    n_methods = len(METHODS)
    bar_width = 0.18
    x = np.arange(n_models)

    for ax, task, task_label in zip(axes, TASKS, TASK_LABELS):
        for i, (method_key, method_label) in enumerate(METHODS):
            accs = []
            for model in MODELS:
                acc = data[model][method_key]["tasks"][task]["acc"]
                accs.append(acc)
            offset = (i - n_methods / 2 + 0.5) * bar_width
            bars = ax.bar(x + offset, accs, bar_width,
                          label=method_label, color=COLORS[i],
                          edgecolor="white", linewidth=0.5)

        ax.set_title(task_label, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_LABELS, fontsize=10)
        ax.set_xlabel("Model Scale", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Accuracy", fontsize=12)

    # Single legend in top-right of the whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=10, framealpha=0.9)

    fig.suptitle("Zero-Shot Accuracy Across Benchmarks (4-bit Quantization)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
