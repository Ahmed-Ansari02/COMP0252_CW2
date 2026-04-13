"""
Visualization for Selective Layer-Level Outlier Protection

Generates four key figures:
  1. Layer sensitivity heatmap (per-layer stats across metrics)
  2. Perplexity vs top-k curve (per scoring method)
  3. Perplexity vs model size Pareto plot
  4. Scoring method comparison (which metric best identifies critical layers)

Also generates profile distribution plots (histograms of per-layer statistics).

Usage:
    python -m src.visualize_selective                                    # defaults
    python -m src.visualize_selective --profile results/opt-125m_profile.json
    python -m src.visualize_selective --results results/selective_results.json
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Consistent styling
COLORS = {
    "kurtosis": "#E63946",
    "outlier_fraction": "#457B9D",
    "range_sigma": "#2A9D8F",
    "fp16": "#264653",
    "all_protection": "#E9C46A",
    "no_protection": "#F4A261",
}

SCORING_LABELS = {
    "kurtosis": "Kurtosis",
    "outlier_fraction": "Outlier Fraction",
    "range_sigma": "Range / σ",
}


def plot_layer_heatmap(profile: dict, save_dir: str = "figures"):
    """
    Plot a heatmap of per-layer statistics.
    Rows = layers, columns = metrics (kurtosis, outlier fraction, range/σ, variance).
    """
    layer_stats = profile["layer_stats"]
    layer_names = profile["layer_names"]

    # Shorten layer names for display
    short_names = []
    for name in layer_names:
        parts = name.split(".")
        # e.g. model.decoder.layers.0.self_attn.q_proj -> L0.attn.q
        short = name
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                layer_num = parts[i + 1]
                remaining = ".".join(parts[i + 2:])
                remaining = (remaining
                             .replace("self_attn.", "attn.")
                             .replace("_proj", "")
                             .replace("model.decoder.", ""))
                short = f"L{layer_num}.{remaining}"
                break
        short_names.append(short)

    metrics = ["kurtosis", "outlier_fraction_3sigma", "range_sigma_ratio", "variance"]
    metric_labels = ["Kurtosis", "Outlier Frac (3σ)", "Range / σ", "Variance"]

    data = np.zeros((len(layer_names), len(metrics)))
    for i, name in enumerate(layer_names):
        for j, m in enumerate(metrics):
            data[i, j] = layer_stats[name][m]

    # Normalise each column for display
    for j in range(data.shape[1]):
        col = data[:, j]
        cmin, cmax = col.min(), col.max()
        if cmax > cmin:
            data[:, j] = (col - cmin) / (cmax - cmin)

    fig, ax = plt.subplots(figsize=(8, max(6, len(layer_names) * 0.25)))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(short_names)))
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_xlabel("Metric (normalised)")
    ax.set_ylabel("Layer")
    ax.set_title(f"Layer Sensitivity Heatmap — {profile['model_name'].split('/')[-1]}")

    plt.colorbar(im, ax=ax, shrink=0.6, label="Normalised value")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    model_short = profile["model_name"].split("/")[-1]
    path = os.path.join(save_dir, f"{model_short}_layer_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_profile_distributions(profile: dict, save_dir: str = "figures"):
    """
    Plot histograms of per-layer statistics (distributions across layers).
    One subplot per metric.
    """
    layer_stats = profile["layer_stats"]
    layer_names = profile["layer_names"]

    metrics = {
        "kurtosis": "Kurtosis",
        "excess_kurtosis": "Excess Kurtosis",
        "outlier_fraction_3sigma": "Outlier Fraction (3σ)",
        "range_sigma_ratio": "Range / σ Ratio",
        "variance": "Variance",
        "max_abs": "Max |w|",
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for idx, (key, label) in enumerate(metrics.items()):
        ax = axes[idx]
        values = [layer_stats[name][key] for name in layer_names]
        ax.hist(values, bins=20, color="#457B9D", edgecolor="white", alpha=0.8)
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("# Layers", fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.axvline(np.mean(values), color="#E63946", linestyle="--",
                   label=f"mean={np.mean(values):.4f}")
        ax.legend(fontsize=7)

    model_short = profile["model_name"].split("/")[-1]
    fig.suptitle(f"Layer Statistics Distribution — {model_short}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{model_short}_profile_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_ppl_vs_topk(results: dict, model_name: str, save_dir: str = "figures"):
    """
    Plot perplexity vs top-k for each scoring method.
    Includes FP16 and all-protection baselines as horizontal lines.
    """
    model_results = results.get(model_name, {})
    if not model_results:
        print(f"No results for {model_name}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract baselines
    fp16_ppl = model_results.get("fp16")
    if isinstance(fp16_ppl, dict):
        fp16_ppl = fp16_ppl.get("perplexity")

    all_prot_key = [k for k in model_results if k.startswith("all_protection_")]
    no_prot_key = [k for k in model_results if k.startswith("no_protection_")]
    all_prot_ppl = model_results[all_prot_key[0]]["perplexity"] if all_prot_key else None
    no_prot_ppl = model_results[no_prot_key[0]]["perplexity"] if no_prot_key else None

    # Plot baselines
    if fp16_ppl:
        ax.axhline(fp16_ppl, color=COLORS["fp16"], linestyle=":",
                   linewidth=1.5, label=f"FP16 ({fp16_ppl:.1f})")
    if all_prot_ppl:
        ax.axhline(all_prot_ppl, color=COLORS["all_protection"], linestyle="--",
                   linewidth=1.5, label=f"All protected ({all_prot_ppl:.1f})")
    if no_prot_ppl:
        ax.axhline(no_prot_ppl, color=COLORS["no_protection"], linestyle="--",
                   linewidth=1.5, label=f"No protection ({no_prot_ppl:.1f})")

    # Plot selective results per scoring method
    for scoring in ["kurtosis", "outlier_fraction", "range_sigma"]:
        # Find all selective keys with this scoring method
        selective_keys = [k for k in model_results
                          if k.startswith(f"selective_{scoring}_")]
        if not selective_keys:
            continue

        points = []
        for key in selective_keys:
            entry = model_results[key]
            topk = entry.get("topk", 0)
            total = entry.get("total_layers", 1)
            frac = topk / total
            ppl = entry["perplexity"]
            points.append((frac, ppl))

        points.sort()
        fracs, ppls = zip(*points)

        ax.plot(fracs, ppls, marker="o", markersize=5, linewidth=2,
                color=COLORS.get(scoring, "gray"),
                label=SCORING_LABELS.get(scoring, scoring))

    ax.set_xlabel("Fraction of Layers Protected (top-k / total)", fontsize=11)
    ax.set_ylabel("Perplexity (WikiText-2)", fontsize=11)
    ax.set_title(f"Selective Outlier Protection — {model_name.split('/')[-1]}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    model_short = model_name.split("/")[-1]
    path = os.path.join(save_dir, f"{model_short}_ppl_vs_topk.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_pareto(results: dict, model_name: str, save_dir: str = "figures"):
    """
    Plot perplexity vs effective model size (bits/param) — Pareto frontier.
    Shows the tradeoff between compression and quality.
    """
    model_results = results.get(model_name, {})
    if not model_results:
        print(f"No results for {model_name}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Collect all points
    for scoring in ["kurtosis", "outlier_fraction", "range_sigma"]:
        selective_keys = [k for k in model_results
                          if k.startswith(f"selective_{scoring}_")]
        if not selective_keys:
            continue

        bits_per_param = []
        ppls = []
        for key in selective_keys:
            entry = model_results[key]
            bpp = entry.get("size", {}).get("effective_bits_per_param")
            ppl = entry["perplexity"]
            if bpp:
                bits_per_param.append(bpp)
                ppls.append(ppl)

        ax.scatter(bits_per_param, ppls, s=40, alpha=0.7,
                   color=COLORS.get(scoring, "gray"),
                   label=SCORING_LABELS.get(scoring, scoring))

    # Add baselines as special markers
    for key_prefix, label, marker, color in [
        ("all_protection_", "All Protected", "D", COLORS["all_protection"]),
        ("no_protection_", "No Protection", "s", COLORS["no_protection"]),
    ]:
        matching = [k for k in model_results if k.startswith(key_prefix)]
        if matching:
            entry = model_results[matching[0]]
            bpp = entry.get("size", {}).get("effective_bits_per_param")
            ppl = entry["perplexity"]
            if bpp:
                ax.scatter([bpp], [ppl], s=120, marker=marker, color=color,
                           edgecolors="black", linewidths=1.5, label=label, zorder=5)

    # FP16 baseline
    fp16_ppl = model_results.get("fp16")
    if isinstance(fp16_ppl, dict):
        fp16_ppl = fp16_ppl.get("perplexity")
    if fp16_ppl:
        ax.scatter([16.0], [fp16_ppl], s=120, marker="*", color=COLORS["fp16"],
                   edgecolors="black", linewidths=1.5, label="FP16", zorder=5)

    ax.set_xlabel("Effective Bits per Parameter", fontsize=11)
    ax.set_ylabel("Perplexity (WikiText-2)", fontsize=11)
    ax.set_title(f"Compression vs Quality — {model_name.split('/')[-1]}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    model_short = model_name.split("/")[-1]
    path = os.path.join(save_dir, f"{model_short}_pareto.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_scoring_comparison(results: dict, model_name: str, save_dir: str = "figures"):
    """
    Bar chart comparing the three scoring methods at a fixed top-k fraction.
    Shows which scoring method best identifies the critical layers.
    """
    model_results = results.get(model_name, {})
    if not model_results:
        return

    # Find common top-k fractions across scoring methods
    scoring_data = {}
    for scoring in ["kurtosis", "outlier_fraction", "range_sigma"]:
        selective_keys = [k for k in model_results
                          if k.startswith(f"selective_{scoring}_")]
        fracs = {}
        for key in selective_keys:
            entry = model_results[key]
            frac = entry.get("topk_fraction", 0)
            fracs[round(frac, 2)] = entry["perplexity"]
        scoring_data[scoring] = fracs

    if not scoring_data:
        return

    # Find common fractions
    common_fracs = set.intersection(*[set(d.keys()) for d in scoring_data.values()])
    if not common_fracs:
        return

    common_fracs = sorted(common_fracs)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(common_fracs))
    width = 0.25

    for i, scoring in enumerate(["kurtosis", "outlier_fraction", "range_sigma"]):
        vals = [scoring_data[scoring].get(f, float("nan")) for f in common_fracs]
        ax.bar(x + i * width, vals, width,
               label=SCORING_LABELS[scoring],
               color=COLORS[scoring], alpha=0.85)

    ax.set_xlabel("Fraction of Layers Protected", fontsize=11)
    ax.set_ylabel("Perplexity (WikiText-2)", fontsize=11)
    ax.set_title(f"Scoring Method Comparison — {model_name.split('/')[-1]}",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{f:.0%}" for f in common_fracs])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    model_short = model_name.split("/")[-1]
    path = os.path.join(save_dir, f"{model_short}_scoring_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize selective outlier protection results")
    parser.add_argument("--profile", type=str, default=None,
                        help="Path to profile JSON (for heatmap / distribution plots)")
    parser.add_argument("--results", type=str,
                        default="results/selective_results.json",
                        help="Path to selective results JSON")
    parser.add_argument("--model", type=str, default="facebook/opt-125m",
                        help="Model to plot for (used when results contain multiple)")
    parser.add_argument("--figures_dir", type=str, default="figures",
                        help="Directory to save figures")
    args = parser.parse_args()

    base_dir = os.path.join(os.path.dirname(__file__), "..")
    figures_dir = os.path.join(base_dir, args.figures_dir)

    # Profile plots
    if args.profile:
        profile_path = args.profile
        if not os.path.isabs(profile_path):
            profile_path = os.path.join(base_dir, profile_path)
        if os.path.exists(profile_path):
            with open(profile_path) as f:
                profile = json.load(f)
            plot_layer_heatmap(profile, save_dir=figures_dir)
            plot_profile_distributions(profile, save_dir=figures_dir)
        else:
            print(f"Profile not found: {profile_path}")
    else:
        # Try to find profile for the requested model
        model_short = args.model.split("/")[-1]
        profile_path = os.path.join(base_dir, "results", f"{model_short}_profile.json")
        if os.path.exists(profile_path):
            with open(profile_path) as f:
                profile = json.load(f)
            plot_layer_heatmap(profile, save_dir=figures_dir)
            plot_profile_distributions(profile, save_dir=figures_dir)
        else:
            print(f"No profile found at {profile_path}. Run profiler first or pass --profile.")

    # Results plots
    results_path = args.results
    if not os.path.isabs(results_path):
        results_path = os.path.join(base_dir, results_path)

    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)

        for model_name in results.keys():
            if args.model and model_name != args.model:
                continue
            plot_ppl_vs_topk(results, model_name, save_dir=figures_dir)
            plot_pareto(results, model_name, save_dir=figures_dir)
            plot_scoring_comparison(results, model_name, save_dir=figures_dir)
    else:
        print(f"No results found at {results_path}. Run experiments first.")

    print(f"\nAll figures saved to {figures_dir}/")


if __name__ == "__main__":
    main()
