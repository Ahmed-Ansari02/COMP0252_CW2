"""
Plot focused no-OP vs OP comparisons from results.json.

This figure is meant to make the outlier-protection story visually obvious:
CDF quantization without outlier protection is catastrophic, while adding
row-wise outlier protection brings perplexity back down sharply. A second
panel shows the same comparison for uniform RTN as a reference.

Usage:
    python plot_cdf_op_comparison.py
    python plot_cdf_op_comparison.py --input results_quantization_methods/results.json
    python plot_cdf_op_comparison.py --output figures/cdf_op_comparison.png
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
]

COMPARISONS = [
    (
        "cdf_4bit_rtn_deconly",
        "cdf_4bit_rtn_op1.0_deconly",
        "CDF RTN",
    ),
    (
        "uniform_4bit_rtn_deconly",
        "uniform_4bit_rtn_op1.0_deconly",
        "Uniform RTN",
    ),
]


def _short_name(model_name: str) -> str:
    return model_name.split("/")[-1].upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results_quantization_methods/results.json")
    parser.add_argument("--output", default="figures/cdf_op_comparison.png")
    args = parser.parse_args()

    with open(args.input) as f:
        results = json.load(f)

    required_keys = {key for pair in COMPARISONS for key in pair[:2]}
    models = [
        model for model in MODEL_ORDER
        if model in results and required_keys.issubset(results[model].keys())
    ]

    if not models:
        raise ValueError("No models with all required no-OP vs OP results were found.")

    labels = [_short_name(m) for m in models]
    x = np.arange(len(models))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True)
    color_no_op = "#d94f3d"
    color_op = "#2b6cb0"

    for ax, (no_op_key, op_key, title) in zip(axes, COMPARISONS):
        no_op = np.array([results[m][no_op_key]["perplexity"] for m in models], dtype=float)
        op = np.array([results[m][op_key]["perplexity"] for m in models], dtype=float)

        bars_no_op = ax.bar(
            x - width / 2,
            no_op,
            width,
            color=color_no_op,
            label="without OP",
        )
        bars_op = ax.bar(
            x + width / 2,
            op,
            width,
            color=color_op,
            label="+ 1% OP",
        )

        ax.set_yscale("log")
        ax.set_title(title, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.grid(axis="y", which="both", alpha=0.25)

        for bars in (bars_no_op, bars_op):
            for bar in bars:
                value = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value * 1.05,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=0,
                )

    axes[0].set_ylabel("WikiText-2 perplexity (log scale) -- lower is better", fontsize=11)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        fontsize=10,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
    )
    fig.suptitle(
        "Outlier Protection Improves Decoder-Only RTN Quantization",
        fontsize=15,
        y=0.985,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
