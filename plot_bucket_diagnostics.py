"""
Quantization Bucket Diagnostics

Generates 5 diagnostic figures to verify how quantization grids
distribute weights across bucket levels.

Usage:
    python plot_bucket_diagnostics.py --model facebook/opt-125m
    python plot_bucket_diagnostics.py --model facebook/opt-125m --layer model.decoder.layers.5.self_attn.q_proj
    python plot_bucket_diagnostics.py --model facebook/opt-125m --output_dir figures/bucket_diagnostics
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from transformers import OPTForCausalLM

from cdf_grid import (
    build_uniform_grid, build_cdf_grid, build_hybrid_grid,
    quantize_to_grid, quantize_matrix_batched,
)


GRID_TYPES = ["uniform", "cdf", "hybrid"]
GRID_COLORS = {"uniform": "#e74c3c", "cdf": "#3498db", "hybrid": "#2ecc71"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_layer_weight(model, layer_name):
    """Extract weight tensor from a named layer."""
    for name, module in model.named_modules():
        if name == layer_name and hasattr(module, "weight"):
            return module.weight.data.clone()
    raise ValueError(f"Layer '{layer_name}' not found")


def build_grid_with_op(weight_row, num_levels, grid_type, gamma, outlier_pct):
    """Build a grid from inliers only (matching OP behavior)."""
    w = weight_row.float()
    lo = torch.quantile(w, outlier_pct / 100.0).item()
    hi = torch.quantile(w, 1.0 - outlier_pct / 100.0).item()
    inliers = w[(w >= lo) & (w <= hi)]

    if grid_type == "uniform":
        return build_uniform_grid(inliers, num_levels)
    elif grid_type == "cdf":
        return build_cdf_grid(inliers, num_levels, pin_endpoints=True)
    elif grid_type == "hybrid":
        return build_hybrid_grid(inliers, num_levels, gamma=gamma)
    else:
        raise ValueError(f"Unknown grid_type: {grid_type}")


def quantize_row_with_op(weight_row, num_levels, grid_type, gamma, outlier_pct):
    """Quantize a single row with outlier protection, return (quantized, outlier_mask)."""
    w = weight_row.float()
    lo = torch.quantile(w, outlier_pct / 100.0).item()
    hi = torch.quantile(w, 1.0 - outlier_pct / 100.0).item()
    outlier_mask = (w < lo) | (w > hi)
    inliers = w[~outlier_mask]

    grid = build_grid_with_op(weight_row, num_levels, grid_type, gamma, outlier_pct)
    quantized = quantize_to_grid(w, grid)
    # Restore outliers
    quantized[outlier_mask] = w[outlier_mask]
    return quantized, outlier_mask, grid


# ---------------------------------------------------------------------------
# Figure 1: Bucket Population
# ---------------------------------------------------------------------------

def plot_bucket_population(W, num_levels, gamma, outlier_pct, output_dir):
    """Bar chart showing how many weights land in each bucket per grid type."""
    row = W[0]  # sample first row for grid building
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, grid_type in zip(axes, GRID_TYPES):
        grid = build_grid_with_op(row, num_levels, grid_type, gamma, outlier_pct)

        # Quantize all rows and count bucket populations
        total_counts = torch.zeros(num_levels)
        for r in range(W.shape[0]):
            q, omask, _ = quantize_row_with_op(W[r], num_levels, grid_type, gamma, outlier_pct)
            inlier_q = q[~omask]
            for i, level in enumerate(grid):
                total_counts[i] += (torch.abs(inlier_q - level) < 1e-6).sum().item()

        counts = total_counts.numpy()
        colors = [GRID_COLORS[grid_type]] * num_levels
        ax.bar(range(num_levels), counts, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(f"{grid_type.upper()}", fontsize=12)
        ax.set_xlabel("Bucket Index")

        # Annotate balance metric
        if counts.sum() > 0:
            normalized = counts / counts.sum()
            ideal = 1.0 / num_levels
            imbalance = np.std(normalized) / ideal
            ax.text(0.98, 0.95, f"Imbalance: {imbalance:.2f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    axes[0].set_ylabel("Weight Count")
    fig.suptitle("Bucket Population Distribution (sample layer, with 1% OP)", fontsize=13)
    fig.tight_layout()
    path = os.path.join(output_dir, "1_bucket_population.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 2: Quantization Error Distribution
# ---------------------------------------------------------------------------

def plot_error_distribution(W, num_levels, gamma, outlier_pct, output_dir):
    """Overlaid histograms of |original - quantized| per grid type."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for grid_type in GRID_TYPES:
        all_errors = []
        for r in range(W.shape[0]):
            q, omask, _ = quantize_row_with_op(W[r], num_levels, grid_type, gamma, outlier_pct)
            errors = torch.abs(W[r].float() - q)
            # Only count inlier errors (outliers have 0 error)
            all_errors.append(errors[~omask].cpu().numpy())

        all_errors = np.concatenate(all_errors)
        mae = np.mean(all_errors)
        max_err = np.max(all_errors)

        ax.hist(all_errors, bins=100, alpha=0.5, color=GRID_COLORS[grid_type],
                label=f"{grid_type.upper()} (MAE={mae:.5f}, max={max_err:.4f})",
                density=True)

    ax.set_xlabel("Quantization Error |w - Q(w)|", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Quantization Error Distribution (inlier weights only)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "2_error_distribution.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 3: Weight Distribution + Grid Lines
# ---------------------------------------------------------------------------

def plot_weight_dist_with_grids(W, num_levels, gamma, outlier_pct, output_dir):
    """Weight histogram overlaid with grid level lines for each grid type."""
    row = W[0]
    w_np = row.float().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    # Outlier boundaries
    lo = np.percentile(w_np, outlier_pct)
    hi = np.percentile(w_np, 100 - outlier_pct)

    for ax, grid_type in zip(axes, GRID_TYPES):
        ax.hist(w_np, bins=200, density=True, alpha=0.4, color="gray")
        # Shade outlier regions
        ax.axvspan(w_np.min(), lo, alpha=0.1, color="red")
        ax.axvspan(hi, w_np.max(), alpha=0.1, color="red")

        grid = build_grid_with_op(row, num_levels, grid_type, gamma, outlier_pct)
        for val in grid:
            ax.axvline(val.item(), color=GRID_COLORS[grid_type], alpha=0.7, linewidth=1.5)

        ax.set_title(f"{grid_type.upper()} + OP {outlier_pct}%", fontsize=12)
        ax.set_xlabel("Weight Value")

    axes[0].set_ylabel("Density")
    fig.suptitle("Weight Distribution with Quantization Grid Levels", fontsize=13)
    fig.tight_layout()
    path = os.path.join(output_dir, "3_weight_dist_grid_lines.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 4: Bucket Spacing
# ---------------------------------------------------------------------------

def plot_bucket_spacing(W, num_levels, gamma, outlier_pct, output_dir):
    """Plot gaps between consecutive grid levels."""
    row = W[0]
    fig, ax = plt.subplots(figsize=(10, 5))

    for grid_type in GRID_TYPES:
        grid = build_grid_with_op(row, num_levels, grid_type, gamma, outlier_pct)
        grid_sorted = torch.sort(grid).values.cpu().numpy()
        gaps = np.diff(grid_sorted)

        ax.plot(range(1, num_levels), gaps, marker="o", color=GRID_COLORS[grid_type],
                linewidth=2, markersize=6, label=grid_type.upper())

    ax.set_xlabel("Bucket Boundary Index", fontsize=11)
    ax.set_ylabel("Gap Size (distance between consecutive levels)", fontsize=11)
    ax.set_title("Bucket Spacing: Gap Between Consecutive Grid Levels", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "4_bucket_spacing.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 5: Per-Layer MAE Heatmap
# ---------------------------------------------------------------------------

def plot_layer_error_heatmap(model, num_levels, gamma, outlier_pct, output_dir):
    """Heatmap of mean absolute quantization error per decoder layer per grid type."""
    layer_names = []
    mae_data = {gt: [] for gt in GRID_TYPES}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.startswith("model.decoder.layers."):
            W = module.weight.data
            layer_names.append(name.replace("model.decoder.layers.", "L"))

            for grid_type in GRID_TYPES:
                Q, _ = quantize_matrix_batched(
                    W, num_levels, grid_type=grid_type, gamma=gamma,
                    pin_endpoints=True, protect_outliers=True,
                    outlier_percentile=outlier_pct)
                mae = torch.abs(W.float() - Q.float()).mean().item()
                mae_data[grid_type].append(mae)

    # Build heatmap matrix: (n_layers, 3)
    matrix = np.array([mae_data[gt] for gt in GRID_TYPES]).T  # (n_layers, 3)

    fig, ax = plt.subplots(figsize=(6, max(8, len(layer_names) * 0.3)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(GRID_TYPES)))
    ax.set_xticklabels([gt.upper() for gt in GRID_TYPES], fontsize=10)
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels(layer_names, fontsize=6)
    ax.set_xlabel("Grid Type", fontsize=11)
    ax.set_ylabel("Decoder Layer", fontsize=11)
    ax.set_title("Mean Absolute Quantization Error per Layer", fontsize=13)

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center",
                    fontsize=5, color="black" if matrix[i, j] < matrix.max() * 0.7 else "white")

    fig.colorbar(im, ax=ax, label="MAE")
    fig.tight_layout()
    path = os.path.join(output_dir, "5_layer_error_heatmap.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Quantization bucket diagnostic plots")
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--outlier_percentile", type=float, default=1.0)
    parser.add_argument("--layer", type=str,
                        default="model.decoder.layers.0.self_attn.q_proj",
                        help="Layer to use for per-row diagnostics (figures 1-4)")
    parser.add_argument("--output_dir", type=str,
                        default="figures/bucket_diagnostics")
    args = parser.parse_args()

    num_levels = 2 ** args.bits
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.model}...")
    model = OPTForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    print(f"Extracting layer: {args.layer}")
    W = get_layer_weight(model, args.layer).to(device)
    print(f"  Shape: {W.shape}")

    print("\n--- Figure 1: Bucket Population ---")
    plot_bucket_population(W, num_levels, args.gamma, args.outlier_percentile,
                           args.output_dir)

    print("--- Figure 2: Quantization Error Distribution ---")
    plot_error_distribution(W, num_levels, args.gamma, args.outlier_percentile,
                            args.output_dir)

    print("--- Figure 3: Weight Distribution + Grid Lines ---")
    plot_weight_dist_with_grids(W, num_levels, args.gamma, args.outlier_percentile,
                                args.output_dir)

    print("--- Figure 4: Bucket Spacing ---")
    plot_bucket_spacing(W, num_levels, args.gamma, args.outlier_percentile,
                        args.output_dir)

    print("--- Figure 5: Per-Layer MAE Heatmap ---")
    plot_layer_error_heatmap(model, num_levels, args.gamma, args.outlier_percentile,
                             args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
