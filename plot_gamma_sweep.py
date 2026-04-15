"""
Plot Gamma Sweep Results — Perplexity vs Gamma

Reads gamma_sweep.json and plots perplexity as a function of gamma
for each model, with the optimal gamma highlighted.

Usage:
    python plot_gamma_sweep.py
    python plot_gamma_sweep.py --input results_quantization_methods/gamma_sweep.json
    python plot_gamma_sweep.py --output figures/gamma_sweep.png
"""

import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def extract_gamma_results(model_results):
    """Extract (gamma, perplexity) pairs from a model's results."""
    pairs = []
    for key, val in model_results.items():
        match = re.match(r"hybrid_gamma([\d.]+)_\d+bit_rtn_op[\d.]+_deconly", key)
        if match:
            gamma = float(match.group(1))
            ppl = val["perplexity"] if isinstance(val, dict) else val
            pairs.append((gamma, ppl))
    pairs.sort(key=lambda x: x[0])
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        default="results_quantization_methods/gamma_sweep.json")
    parser.add_argument("--output", type=str,
                        default="figures/gamma_sweep.png")
    args = parser.parse_args()

    with open(args.input) as f:
        results = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 5))

    markers = ['o', 's', '^', 'D', 'v']
    colors = plt.cm.tab10.colors

    all_gammas = set()

    for idx, (model_name, model_results) in enumerate(results.items()):
        pairs = extract_gamma_results(model_results)
        if not pairs:
            continue

        gammas = [p[0] for p in pairs]
        ppls = [p[1] for p in pairs]
        all_gammas.update(gammas)

        short_name = model_name.split("/")[-1].upper()
        marker = markers[idx % len(markers)]
        color = colors[idx % len(colors)]

        ax.plot(gammas, ppls, marker=marker, color=color, linewidth=2,
                markersize=8, label=short_name)

        # Highlight the best gamma
        best_idx = np.argmin(ppls)
        ax.plot(gammas[best_idx], ppls[best_idx], marker='*', color=color,
                markersize=16, zorder=5)
        ax.annotate(f'  best={gammas[best_idx]}',
                    (gammas[best_idx], ppls[best_idx]),
                    fontsize=9, color=color)

    ax.set_xlabel('Gamma (γ)', fontsize=12)
    ax.set_ylabel('Perplexity (WikiText-2)', fontsize=12)
    ax.set_title('Perplexity vs Hybrid Grid Mixing Coefficient (γ)\n'
                 '4-bit RTN, decoder-only, 1% outlier protection', fontsize=13)
    if all_gammas:
        ax.set_xticks(sorted(all_gammas))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
