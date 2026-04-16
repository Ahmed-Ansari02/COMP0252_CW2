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

    fig, (ax, ax_table) = plt.subplots(1, 2, figsize=(14, 5),
                                        gridspec_kw={'width_ratios': [3, 2]})

    markers = ['o', 's', '^', 'D', 'v']
    colors = plt.cm.tab10.colors

    all_gammas = set()
    table_data = {}  # model_short_name -> {gamma: ppl}

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

        table_data[short_name] = {g: p for g, p in zip(gammas, ppls)}

        ax.plot(gammas, ppls, marker=marker, color=color, linewidth=2,
                markersize=8, label=short_name)

        # Highlight the best gamma
        best_idx = np.argmin(ppls)
        ax.plot(gammas[best_idx], ppls[best_idx], marker='*', color=color,
                markersize=16, zorder=5)
        # Offset labels to avoid overlapping with nearby lines
        y_offset = 14
        x_offset = 0
        if short_name == 'OPT-6.7B':
            y_offset = 10
            x_offset = 20
        ax.annotate(f'γ*={gammas[best_idx]}',
                    (gammas[best_idx], ppls[best_idx]),
                    textcoords="offset points", xytext=(x_offset, y_offset),
                    fontsize=9, fontweight='bold', color=color,
                    ha='center')

    ax.set_xlabel('Gamma (γ)', fontsize=12)
    ax.set_ylabel('Perplexity (WikiText-2)', fontsize=12)
    ax.set_title('Perplexity vs Hybrid Grid Mixing Coefficient (γ)\n'
                 '4-bit RTN, decoder-only, 1% outlier protection', fontsize=13)
    if all_gammas:
        ax.set_xticks(sorted(all_gammas))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Build table on the right
    sorted_gammas = sorted(all_gammas)
    model_names = list(table_data.keys())
    col_labels = [f'γ={g}' for g in sorted_gammas]
    cell_text = []
    cell_colors = []
    for mname in model_names:
        row = []
        row_colors = []
        ppls = [table_data[mname].get(g, float('nan')) for g in sorted_gammas]
        best_ppl = min(ppls)
        for ppl in ppls:
            row.append(f'{ppl:.2f}')
            if ppl == best_ppl:
                row_colors.append('#d5f5e3')
            else:
                row_colors.append('white')
        cell_text.append(row)
        cell_colors.append(row_colors)

    ax_table.axis('off')
    ax_table.set_title('Perplexity by Model and γ', fontsize=12, pad=12)
    tbl = ax_table.table(cellText=cell_text, rowLabels=model_names,
                         colLabels=col_labels, cellColours=cell_colors,
                         loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.0, 1.3)

    # Bold the best value in each row
    for i, mname in enumerate(model_names):
        ppls = [table_data[mname].get(g, float('nan')) for g in sorted_gammas]
        best_j = int(np.argmin(ppls))
        tbl[i + 1, best_j].set_text_props(fontweight='bold')

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved to {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
