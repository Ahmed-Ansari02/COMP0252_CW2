"""
Plot perplexity bar chart from results_quantization_methods/results.json

Compares all decoder-only + OP methods and GPTQ across OPT model scales.

Usage:
    python plot_perplexity_barchart.py
    python plot_perplexity_barchart.py --input results_quantization_methods/results.json
    python plot_perplexity_barchart.py --output figures/perplexity_barchart.png
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Methods to plot (key suffix -> display label)
METHODS = [
    ("fp16",                                "FP16 (baseline)"),
    ("uniform_4bit_rtn_op1.0_deconly",      "Uniform RTN + OP"),
    ("cdf_4bit_rtn_op1.0_deconly",          "CDF RTN + OP"),
    ("hybrid_gamma0.5_4bit_rtn_op1.0_deconly", "Hybrid RTN + OP"),
    ("uniform_4bit_gptq",                   "GPTQ (uniform)"),
]

MODEL_ORDER = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results_quantization_methods/results.json")
    parser.add_argument("--output", default="figures/perplexity_barchart.png")
    args = parser.parse_args()

    with open(args.input) as f:
        results = json.load(f)

    # Filter to models present in results, preserving order
    models = [m for m in MODEL_ORDER if m in results]
    short_names = [m.split("/")[-1].upper() for m in models]

    # Filter to methods that have at least one result
    methods = [(k, label) for k, label in METHODS
               if any(k in results.get(m, {}) for m in models)]

    n_models = len(models)
    n_methods = len(methods)
    x = np.arange(n_models)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#333333', '#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    for i, (key, label) in enumerate(methods):
        vals = []
        for m in models:
            entry = results.get(m, {}).get(key)
            if entry is None:
                vals.append(float('nan'))
            elif isinstance(entry, dict):
                vals.append(entry.get("perplexity", float('nan')))
            else:
                vals.append(entry)
        bars = ax.bar(x + i * width, vals, width, label=label, color=colors[i % len(colors)])
        # Add value labels on bars
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                        f'{v:.1f}', ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_ylabel('Perplexity (WikiText-2) -- lower is better', fontsize=11)
    ax.set_title('4-bit Quantization: Perplexity Comparison Across OPT Models\n'
                 '(decoder-only, 1% outlier protection for RTN methods)', fontsize=13)
    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels(short_names, fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
