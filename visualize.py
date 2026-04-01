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

from cdf_grid import build_uniform_grid, build_cdf_grid, build_hybrid_grid


def plot_grid_comparison(weight_row: torch.Tensor, bits: int = 4,
                          save_path: str = "grid_comparison.png"):
    """
    Plot a weight histogram with uniform and CDF grids overlaid.
    """
    num_levels = 2 ** bits

    uniform = build_uniform_grid(weight_row, num_levels)
    cdf = build_cdf_grid(weight_row, num_levels)
    hybrid = build_hybrid_grid(weight_row, num_levels, gamma=0.15)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(weight_row.float().numpy(), bins=200, density=True, alpha=0.5,
            color='gray', label='Weight distribution')

    for i, val in enumerate(uniform):
        ax.axvline(val.item(), color='red', alpha=0.6, linestyle='--',
                   label='Uniform' if i == 0 else None)

    for i, val in enumerate(cdf):
        ax.axvline(val.item(), color='blue', alpha=0.6, linestyle='-',
                   label='CDF' if i == 0 else None)

    for i, val in enumerate(hybrid):
        ax.axvline(val.item(), color='green', alpha=0.6, linestyle=':',
                   label='Hybrid (γ=0.15)' if i == 0 else None)

    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Density')
    ax.set_title(f'{bits}-bit Quantization Grid Comparison')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
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


def grid_demo_from_model(model_name: str, layer_name: str, bits: int = 4):
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
                                  save_path=f"grid_comparison_{layer_name.replace('.', '_')}.png")
            return

    print(f"Layer '{layer_name}' not found. Available linear layers:")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            print(f"  {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results.json",
                        help="Path to results.json")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--grid_demo", action="store_true",
                        help="Plot grid comparison using synthetic weights")
    parser.add_argument("--model", type=str, default="facebook/opt-125m",
                        help="Model name for grid demo from real weights")
    parser.add_argument("--layer_name", type=str,
                        default="model.decoder.layers.0.self_attn.q_proj",
                        help="Layer name for grid demo from real weights")
    parser.add_argument("--gamma_sweep", action="store_true",
                        help="Plot gamma sweep for hybrid grid")
    args = parser.parse_args()

    if args.grid_demo:
        # Synthetic Gaussian weights for a quick illustration
        torch.manual_seed(42)
        row = torch.randn(512) * 0.02
        plot_grid_comparison(row, bits=args.bits)
        return

    if args.layer_name and not args.grid_demo:
        # Use real model weights
        grid_demo_from_model(args.model, args.layer_name, bits=args.bits)

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
