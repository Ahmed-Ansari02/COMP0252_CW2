"""
Visualization for CDF Grid Experiments

Usage:
    python visualize.py --results results.json       # bar chart of perplexity
    python visualize.py --grid_demo                  # plot grid comparison
    python visualize.py --model facebook/opt-125m --layer_name model.decoder.layers.0.self_attn.q_proj
"""

import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from cdf_grid import build_uniform_grid, build_cdf_grid, build_hybrid_grid, quantize_row_with_outlier_protection


def plot_grid_comparison(weight_row: torch.Tensor, bits: int = 4,
                          save_path: str = "grid_comparison.png",
                          results_path: str = "results_quantization_methods/results.json",
                          model_name: str = "facebook/opt-125m",
                          gamma: float = 0.15,
                          outlier_pct: float = 1.0):
    """
    Plot a 4x2 subplot grid: left column = standard, right column = with outlier protection.
    Perplexity scores from results.json are shown in subplot titles.
    """
    num_levels = 2 ** bits
    row_f32 = weight_row.float()
    w_np = row_f32.numpy()
    op = outlier_pct

    # Load perplexity results if available
    ppl = {}
    import os
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        model_results = all_results.get(model_name, {})
        for key, val in model_results.items():
            if isinstance(val, dict):
                ppl[key] = val.get("perplexity", val)
            else:
                ppl[key] = val

    # Compute outlier boundaries for shading
    lo = torch.quantile(row_f32, op / 100.0).item()
    hi = torch.quantile(row_f32, 1.0 - op / 100.0).item()

    # Inlier mask for building OP grids
    outlier_mask = (row_f32 < lo) | (row_f32 > hi)
    inliers = row_f32[~outlier_mask]

    # Standard grids (full row)
    uniform = build_uniform_grid(row_f32, num_levels)
    cdf_pinned = build_cdf_grid(row_f32, num_levels, pin_endpoints=True)
    hybrid = build_hybrid_grid(row_f32, num_levels, gamma=gamma)
    cdf_unpinned = build_cdf_grid(row_f32, num_levels, pin_endpoints=False)

    # Outlier-protected grids (built from inliers only)
    uniform_op = build_uniform_grid(inliers, num_levels)
    cdf_pinned_op = build_cdf_grid(inliers, num_levels, pin_endpoints=True)
    hybrid_op = build_hybrid_grid(inliers, num_levels, gamma=gamma)
    cdf_unpinned_op = build_cdf_grid(inliers, num_levels, pin_endpoints=False)

    b = bits
    g = gamma
    rows_data = [
        ("Uniform",           f"uniform_{b}bit_rtn",            uniform,       'red',    '--',
                              f"uniform_{b}bit_rtn_op{op}",     uniform_op,    'red',    '--'),
        ("CDF (pinned)",      f"cdf_{b}bit_rtn",                cdf_pinned,    'blue',   '-',
                              f"cdf_{b}bit_rtn_op{op}",         cdf_pinned_op, 'blue',   '-'),
        (f"Hybrid (γ={g})",   f"hybrid_gamma{g}_{b}bit_rtn",    hybrid,        'green',  '-',
                              f"hybrid_gamma{g}_{b}bit_rtn_op{op}", hybrid_op, 'green',  '-'),
        ("CDF (unpinned)",    f"cdf_{b}bit_rtn_nopin",          cdf_unpinned,  'orange', '-',
                              f"cdf_{b}bit_rtn_nopin_op{op}",   cdf_unpinned_op, 'orange', '-'),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=True, sharey=True)

    for i, (label, key_std, grid, color, ls, key_op, grid_op, color_op, ls_op) in enumerate(rows_data):
        # Left: standard
        ax_l = axes[i, 0]
        ax_l.hist(w_np, bins=200, density=True, alpha=0.4, color='gray')
        for val in grid:
            ax_l.axvline(val.item(), color=color, alpha=0.7, linestyle=ls)
        ppl_std = ppl.get(key_std)
        ppl_str = f' | PPL: {ppl_std:.2f}' if ppl_std is not None else ''
        ax_l.set_title(f'{label}{ppl_str}', fontsize=12)
        ax_l.set_ylabel('Density')

        # Right: with outlier protection
        ax_r = axes[i, 1]
        ax_r.hist(w_np, bins=200, density=True, alpha=0.4, color='gray')
        ax_r.axvspan(w_np.min(), lo, alpha=0.15, color='red')
        ax_r.axvspan(hi, w_np.max(), alpha=0.15, color='red')
        for val in grid_op:
            ax_r.axvline(val.item(), color=color_op, alpha=0.7, linestyle=ls_op)
        ppl_op_val = ppl.get(key_op)
        ppl_op_str = f' | PPL: {ppl_op_val:.2f}' if ppl_op_val is not None else ''
        ax_r.set_title(f'{label} + OP {op}%{ppl_op_str}', fontsize=12)

    axes[-1, 0].set_xlabel('Weight Value')
    axes[-1, 1].set_xlabel('Weight Value')

    fp16_ppl = ppl.get("fp16")
    fp16_str = f' (FP16 baseline: {fp16_ppl:.2f})' if fp16_ppl is not None else ''
    fig.suptitle(f'{bits}-bit Quantization Grid Comparison{fp16_str}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.show()


def plot_results_table(results: dict, bits: int = 4,
                        save_path: str = "perplexity_comparison.png"):
    """
    Plot a grouped bar chart comparing methods across models.
    """
    models = list(results.keys())
    methods = [
        ("fp16", "FP16"),
        (f"uniform_{bits}bit_rtn", f"Uniform {bits}-bit RTN"),
        (f"cdf_{bits}bit_rtn", f"CDF {bits}-bit RTN"),
        (f"hybrid_gamma0.15_{bits}bit_rtn", f"Hybrid γ=0.15 {bits}-bit RTN"),
    ]

    # Filter to methods that have at least one result
    methods = [(k, label) for k, label in methods
               if any(k in results[m] for m in models)]

    if not methods:
        print("No results to plot yet.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.8 / len(methods)

    for i, (key, label) in enumerate(methods):
        vals = [results[m].get(key) for m in models]
        # Use NaN for missing values so bar is skipped
        vals_plot = [v if v is not None else float('nan') for v in vals]
        ax.bar(x + i * width, vals_plot, width, label=label)

    ax.set_ylabel('Perplexity (lower is better)')
    ax.set_title(f'{bits}-bit RTN Quantization: Perplexity Comparison')
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([m.split('/')[-1] for m in models])
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")
    plt.show()


def plot_hybrid_gamma_sweep(results: dict, model_name: str, bits: int = 4,
                             save_path: str = "gamma_sweep.png"):
    """
    Plot the effect of gamma on perplexity for the hybrid grid.
    """
    gammas = [0.05, 0.10, 0.15, 0.20, 0.30]
    model_results = results.get(model_name, {})

    ppls = []
    valid_gammas = []
    for g in gammas:
        key = f"hybrid_gamma{g}_{bits}bit_rtn"
        if key in model_results:
            ppls.append(model_results[key])
            valid_gammas.append(g)

    if not ppls:
        print(f"No hybrid results for {model_name}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(valid_gammas, ppls, marker='o', color='green', label='Hybrid')

    # Reference lines
    uniform_key = f"uniform_{bits}bit_rtn"
    cdf_key = f"cdf_{bits}bit_rtn"
    if uniform_key in model_results:
        ax.axhline(model_results[uniform_key], color='red', linestyle='--',
                   label='Uniform')
    if cdf_key in model_results:
        ax.axhline(model_results[cdf_key], color='blue', linestyle='-',
                   label='CDF (γ=0)')
    if "fp16" in model_results:
        ax.axhline(model_results["fp16"], color='black', linestyle=':',
                   label='FP16')

    ax.set_xlabel('γ (mixing coefficient, 0=CDF, 1=uniform)')
    ax.set_ylabel('Perplexity')
    ax.set_title(f'{model_name.split("/")[-1]}: {bits}-bit Hybrid Grid Gamma Sweep')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")
    plt.show()


def grid_demo_from_model(model_name: str, layer_name: str, bits: int = 4,
                          gamma: float = 0.15, outlier_pct: float = 1.0):
    """
    Load a model, extract a weight row, and plot the grid comparison.
    """
    from transformers import OPTForCausalLM
    print(f"Loading {model_name}...")
    model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

    # Find the requested layer
    for name, module in model.named_modules():
        if name == layer_name and hasattr(module, "weight"):
            row = module.weight.data[0].cpu()
            print(f"Using row 0 from {layer_name} (shape {module.weight.shape})")
            plot_grid_comparison(row, bits=bits,
                                  save_path=f"grid_comparison_{layer_name.replace('.', '_')}.png",
                                  model_name=model_name, gamma=gamma,
                                  outlier_pct=outlier_pct)
            return

    print(f"Layer '{layer_name}' not found. Available linear layers:")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            print(f"  {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results_quantization_methods/results.json",
                        help="Path to results.json")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--grid_demo", action="store_true",
                        help="Plot grid comparison using synthetic weights")
    parser.add_argument("--model", type=str, default="facebook/opt-125m",
                        help="Model name for grid demo from real weights")
    parser.add_argument("--layer_name", type=str,
                        default="model.decoder.layers.0.self_attn.q_proj",
                        help="Layer name for grid demo from real weights")
    parser.add_argument("--gamma", type=float, default=0.15,
                        help="Gamma for hybrid grid (default 0.15)")
    parser.add_argument("--outlier_pct", type=float, default=1.0,
                        help="Outlier percentile for OP column (default 1.0)")
    parser.add_argument("--gamma_sweep", action="store_true",
                        help="Plot gamma sweep for hybrid grid")
    args = parser.parse_args()

    if args.grid_demo:
        # Synthetic Gaussian weights for a quick illustration
        torch.manual_seed(42)
        row = torch.randn(512) * 0.02
        plot_grid_comparison(row, bits=args.bits, gamma=args.gamma,
                              outlier_pct=args.outlier_pct)
        return

    if args.layer_name and not args.grid_demo:
        # Use real model weights
        grid_demo_from_model(args.model, args.layer_name, bits=args.bits,
                              gamma=args.gamma, outlier_pct=args.outlier_pct)

    import os
    if not os.path.exists(args.results):
        print(f"No results file at {args.results}. Run experiments first.")
        return

    with open(args.results) as f:
        results = json.load(f)

    plot_results_table(results, bits=args.bits)

    if args.gamma_sweep:
        plot_hybrid_gamma_sweep(results, args.model, bits=args.bits)


if __name__ == "__main__":
    main()
