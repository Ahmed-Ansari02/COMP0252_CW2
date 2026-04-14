"""
Layer Profiler — First Pass Weight Statistics

Loads a model in FP16 and computes per-layer statistics for all Linear layers
without modifying any weights. Used to identify which layers are most sensitive
to quantization and benefit most from outlier protection.

Statistics computed per layer:
  - variance: spread of weight values
  - kurtosis: tail heaviness (>3 = heavier than Gaussian)
  - outlier_count: number of weights beyond ±3σ
  - outlier_fraction: outlier_count / total weights
  - max_abs: maximum absolute weight value
  - range_sigma_ratio: (max - min) / σ — long tails relative to bulk

Usage:
    python -m src.layer_profiler --model facebook/opt-125m
    python -m src.layer_profiler --model facebook/opt-125m --output results/opt-125m_profile.json
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
from collections import OrderedDict

# Add parent directory to path for cdf_grid imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def compute_layer_stats(weight: torch.Tensor) -> dict:
    """
    Compute weight statistics for a single Linear layer.

    Args:
        weight: 2D weight tensor (out_features × in_features)

    Returns:
        dict with all computed statistics
    """
    W = weight.float().flatten()
    n = W.numel()

    mu = W.mean().item()
    sigma = W.std().item()
    var = W.var().item()

    # Kurtosis: E[(W - μ)^4] / σ^4
    # Using excess kurtosis (subtract 3) so Gaussian = 0
    if sigma > 0:
        centered = W - mu
        kurt = (centered ** 4).mean().item() / (sigma ** 4)  # raw kurtosis
        excess_kurt = kurt - 3.0  # excess kurtosis (Gaussian = 0)
    else:
        kurt = 0.0
        excess_kurt = 0.0

    # Outlier detection: beyond ±3σ
    lo = mu - 3 * sigma
    hi = mu + 3 * sigma
    outlier_mask = (W < lo) | (W > hi)
    outlier_count = outlier_mask.sum().item()
    outlier_fraction = outlier_count / n if n > 0 else 0.0

    # Max absolute value
    max_abs = W.abs().max().item()

    # Range / σ ratio
    w_min = W.min().item()
    w_max = W.max().item()
    range_sigma = (w_max - w_min) / sigma if sigma > 0 else 0.0

    return {
        "num_params": n,
        "mean": round(mu, 8),
        "std": round(sigma, 8),
        "variance": round(var, 10),
        "kurtosis": round(kurt, 6),
        "excess_kurtosis": round(excess_kurt, 6),
        "outlier_count_3sigma": int(outlier_count),
        "outlier_fraction_3sigma": round(outlier_fraction, 8),
        "max_abs": round(max_abs, 8),
        "min": round(w_min, 8),
        "max": round(w_max, 8),
        "range_sigma_ratio": round(range_sigma, 6),
    }


def profile_model(model_name: str) -> dict:
    """
    Profile all Linear layers in a model.

    Args:
        model_name: HuggingFace model name (e.g. 'facebook/opt-125m')

    Returns:
        dict with model metadata and per-layer statistics
    """
    from transformers import OPTForCausalLM

    print(f"Loading {model_name} for profiling...")
    model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

    layer_stats = OrderedDict()
    total_linear_params = 0

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            print(f"  Profiling {name} ({module.weight.shape})...")
            stats = compute_layer_stats(module.weight.data)
            layer_stats[name] = stats
            total_linear_params += stats["num_params"]

    # Compute rankings for each scoring metric
    # (higher score = more sensitive = should be protected first)
    layer_names = list(layer_stats.keys())

    def rank_by_metric(metric_key, reverse=True):
        """Rank layers by a metric. Returns dict mapping layer_name -> rank (1 = most sensitive)."""
        values = [(name, layer_stats[name][metric_key]) for name in layer_names]
        values.sort(key=lambda x: x[1], reverse=reverse)
        return {name: rank + 1 for rank, (name, _) in enumerate(values)}

    def normalise_metric(metric_key):
        """Min-max normalise a metric across layers. Returns dict mapping layer_name -> [0, 1]."""
        values = [layer_stats[name][metric_key] for name in layer_names]
        vmin, vmax = min(values), max(values)
        if vmax == vmin:
            return {name: 0.5 for name in layer_names}
        return {name: (layer_stats[name][metric_key] - vmin) / (vmax - vmin)
                for name in layer_names}

    # Compute normalised scores for each metric
    norm_kurtosis = normalise_metric("kurtosis")
    norm_outlier = normalise_metric("outlier_fraction_3sigma")
    norm_max_abs = normalise_metric("max_abs")
    norm_range_sigma = normalise_metric("range_sigma_ratio")
    norm_variance = normalise_metric("variance")

    # Attach scores and rankings to each layer
    for name in layer_names:
        layer_stats[name]["scores"] = {
            "kurtosis": round(norm_kurtosis[name], 6),
            "outlier_fraction": round(norm_outlier[name], 6),
            "range_sigma": round(norm_range_sigma[name], 6),
        }
        layer_stats[name]["ranks"] = {
            "kurtosis": rank_by_metric("kurtosis")[name],
            "outlier_fraction": rank_by_metric("outlier_fraction_3sigma")[name],
            "range_sigma": rank_by_metric("range_sigma_ratio")[name],
        }

    profile = {
        "model_name": model_name,
        "total_linear_layers": len(layer_names),
        "total_linear_params": total_linear_params,
        "total_model_params": sum(p.numel() for p in model.parameters()),
        "layer_names": layer_names,
        "layer_stats": layer_stats,
    }

    del model
    torch.cuda.empty_cache()

    return profile


def print_profile_summary(profile: dict, top_n: int = 10):
    """Print a human-readable summary of the profile."""
    layer_stats = profile["layer_stats"]
    layer_names = profile["layer_names"]
    n = len(layer_names)

    print(f"\n{'='*80}")
    print(f"LAYER PROFILE: {profile['model_name']}")
    print(f"{'='*80}")
    print(f"Total Linear layers: {n}")
    print(f"Total Linear params: {profile['total_linear_params']:,}")
    print(f"Total model params:  {profile['total_model_params']:,}")

    for metric, metric_key in [("Kurtosis", "kurtosis"),
                                ("Outlier Fraction", "outlier_fraction_3sigma"),
                                ("Range/σ", "range_sigma_ratio")]:
        sorted_layers = sorted(layer_names,
                               key=lambda x: layer_stats[x][metric_key],
                               reverse=True)
        print(f"\n--- Top {top_n} by {metric} ---")
        for i, name in enumerate(sorted_layers[:top_n]):
            val = layer_stats[name][metric_key]
            print(f"  {i+1:3d}. {name:60s}  {val:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Profile Linear layer weights for selective outlier protection")
    parser.add_argument("--model", type=str, default="facebook/opt-125m",
                        help="HuggingFace model name")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: results/{model_short}_profile.json)")
    args = parser.parse_args()

    # Default output path
    if args.output is None:
        model_short = args.model.split("/")[-1]
        args.output = os.path.join(
            os.path.dirname(__file__), "..", "results", f"{model_short}_profile.json")

    profile = profile_model(args.model)
    print_profile_summary(profile)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"\nProfile saved to {args.output}")


if __name__ == "__main__":
    main()
