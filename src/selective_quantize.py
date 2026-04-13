"""
Selective Layer-Level Quantization with Outlier Protection

Quantizes a model using the profile from layer_profiler.py.
Only the top-k most sensitive layers (by a chosen scoring metric) get
outlier protection; the remaining layers are fully quantized.

Usage:
    python -m src.selective_quantize \
        --model facebook/opt-125m \
        --profile results/opt-125m_profile.json \
        --bits 4 --grid_type hybrid --gamma 0.15 \
        --topk 12 --scoring kurtosis \
        --outlier_percentile 1.0
"""

import argparse
import json
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cdf_grid import (build_uniform_grid, build_cdf_grid, build_hybrid_grid,
                       quantize_to_grid, quantize_row_with_outlier_protection,
                       quantize_standard_rtn_row)
from rtn_baseline import load_model, restore_weights, tokenize_dataset, evaluate_perplexity


def get_topk_layers(profile: dict, k: int, scoring: str) -> set:
    """
    Select the top-k layers that should receive outlier protection,
    ranked by the given scoring metric.

    Args:
        profile: profile dict from layer_profiler.py
        k: number of layers to protect (0 = none, len(layers) = all)
        scoring: scoring metric name — one of:
                 'kurtosis', 'outlier_fraction', 'range_sigma'

    Returns:
        set of layer names to protect
    """
    layer_stats = profile["layer_stats"]
    layer_names = profile["layer_names"]

    # Map scoring name to the raw stat key used for ranking
    metric_map = {
        "kurtosis": "kurtosis",
        "outlier_fraction": "outlier_fraction_3sigma",
        "range_sigma": "range_sigma_ratio",
    }

    if scoring not in metric_map:
        raise ValueError(f"Unknown scoring '{scoring}'. Choose from: {list(metric_map.keys())}")

    stat_key = metric_map[scoring]

    # Sort layers by metric value (descending = most sensitive first)
    ranked = sorted(layer_names,
                    key=lambda name: layer_stats[name][stat_key],
                    reverse=True)

    return set(ranked[:k])


def quantize_model_selective(model, original_weights, profile: dict,
                              bits: int, grid_type: str, gamma: float,
                              topk: int, scoring: str,
                              outlier_percentile: float = 1.0):
    """
    Quantize a model with outlier protection applied selectively.

    Args:
        model: pre-loaded model
        original_weights: cached FP16 weights for restore
        profile: layer profile dict
        bits: quantization bit width
        grid_type: 'uniform', 'cdf', or 'hybrid'
        gamma: hybrid grid mixing coefficient
        topk: number of layers to protect
        scoring: scoring metric for layer ranking
        outlier_percentile: percentage of weights at each tail to keep in FP16

    Returns:
        (model, size_stats, protected_layers)
    """
    # Restore to original FP16 weights
    restore_weights(model, original_weights)

    num_levels = 2 ** bits
    protected_set = get_topk_layers(profile, topk, scoring)

    total_layers = len(profile["layer_names"])
    print(f"  Selective quantization: {bits}-bit {grid_type}, "
          f"protecting {topk}/{total_layers} layers by {scoring}, "
          f"α={outlier_percentile}%")

    # Track size statistics
    total_quantized_weights = 0
    total_outlier_weights = 0
    total_unquantized_weights = 0
    protected_layers_info = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            W = module.weight.data.clone()
            protect_this_layer = name in protected_set

            for row_idx in range(W.shape[0]):
                row = W[row_idx]

                if protect_this_layer:
                    quantized_row, num_outliers = quantize_row_with_outlier_protection(
                        row, num_levels, grid_type=grid_type,
                        gamma=gamma, outlier_percentile=outlier_percentile
                    )
                    W[row_idx] = quantized_row
                    total_outlier_weights += num_outliers
                    total_quantized_weights += row.numel() - num_outliers
                elif grid_type == "uniform":
                    W[row_idx] = quantize_standard_rtn_row(row, bits)
                    total_quantized_weights += row.numel()
                else:
                    row_f32 = row.float()
                    if grid_type == "cdf":
                        grid = build_cdf_grid(row_f32, num_levels)
                    elif grid_type == "hybrid":
                        grid = build_hybrid_grid(row_f32, num_levels, gamma)
                    else:
                        raise ValueError(f"Unknown grid_type: {grid_type}")
                    W[row_idx] = quantize_to_grid(row_f32, grid).to(W.dtype)
                    total_quantized_weights += row.numel()

            module.weight.data = W

            if protect_this_layer:
                layer_stats = profile["layer_stats"].get(name, {})
                protected_layers_info.append({
                    "name": name,
                    "kurtosis": layer_stats.get("kurtosis", 0),
                    "outlier_fraction": layer_stats.get("outlier_fraction_3sigma", 0),
                    "range_sigma": layer_stats.get("range_sigma_ratio", 0),
                })

    # Count unquantized params (biases, embeddings, layernorms)
    total_params = sum(p.numel() for p in model.parameters())
    total_unquantized_weights = total_params - total_quantized_weights - total_outlier_weights

    # Compute effective model size
    quantized_bits = total_quantized_weights * bits
    outlier_bits = total_outlier_weights * 16
    unquantized_bits = total_unquantized_weights * 16
    effective_size_mb = (quantized_bits + outlier_bits + unquantized_bits) / 8 / 1024**2
    effective_avg_bits = (quantized_bits + outlier_bits + unquantized_bits) / total_params

    size_stats = {
        "total_params": total_params,
        "quantized_weights": total_quantized_weights,
        "outlier_weights": total_outlier_weights,
        "unquantized_weights": total_unquantized_weights,
        "effective_bits_per_param": round(effective_avg_bits, 3),
        "effective_size_mb": round(effective_size_mb, 2),
        "fp16_size_mb": round(total_params * 16 / 8 / 1024**2, 2),
    }

    print(f"  Size: {size_stats['effective_size_mb']} MB "
          f"({size_stats['effective_bits_per_param']} bits/param), "
          f"outlier weights: {total_outlier_weights:,}")

    return model, size_stats, protected_layers_info


def run_single_selective(model_name: str, profile: dict,
                          bits: int, grid_type: str, gamma: float,
                          topk: int, scoring: str,
                          outlier_percentile: float = 1.0,
                          model=None, original_weights=None,
                          input_ids=None, tokenizer=None):
    """
    Run a single selective quantization + perplexity evaluation.

    Returns:
        (perplexity, size_stats, protected_layers_info)
    """
    model, size_stats, protected_info = quantize_model_selective(
        model, original_weights, profile,
        bits, grid_type, gamma,
        topk, scoring, outlier_percentile
    )
    ppl = evaluate_perplexity(model, tokenizer=tokenizer, input_ids=input_ids)
    print(f"  Perplexity: {ppl:.2f}")
    return ppl, size_stats, protected_info


def make_selective_key(scoring: str, topk: int, bits: int,
                        grid_type: str, gamma: float,
                        outlier_percentile: float) -> str:
    """Generate a unique key for a selective experiment."""
    if grid_type == "hybrid":
        base = f"selective_{scoring}_top{topk}_hybrid_g{gamma}_{bits}bit_op{outlier_percentile}"
    else:
        base = f"selective_{scoring}_top{topk}_{grid_type}_{bits}bit_op{outlier_percentile}"
    return base


def main():
    parser = argparse.ArgumentParser(
        description="Selective layer-level quantization with outlier protection")
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--profile", type=str, required=True,
                        help="Path to profile JSON from layer_profiler.py")
    parser.add_argument("--bits", type=int, default=4, choices=[3, 4])
    parser.add_argument("--grid_type", type=str, default="hybrid",
                        choices=["uniform", "cdf", "hybrid"])
    parser.add_argument("--gamma", type=float, default=0.15)
    parser.add_argument("--topk", type=int, required=True,
                        help="Number of layers to protect")
    parser.add_argument("--scoring", type=str, default="kurtosis",
                        choices=["kurtosis", "outlier_fraction", "range_sigma"])
    parser.add_argument("--outlier_percentile", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="results/selective_results.json")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    # Load profile
    with open(args.profile) as f:
        profile = json.load(f)

    # Load model
    model, original_weights = load_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    input_ids = tokenize_dataset(tokenizer)

    # Run
    ppl, size_stats, protected_info = run_single_selective(
        args.model, profile,
        args.bits, args.grid_type, args.gamma,
        args.topk, args.scoring,
        args.outlier_percentile,
        model=model, original_weights=original_weights,
        input_ids=input_ids, tokenizer=tokenizer
    )

    # Save results
    key = make_selective_key(args.scoring, args.topk, args.bits,
                              args.grid_type, args.gamma, args.outlier_percentile)

    output_path = os.path.join(
        os.path.dirname(__file__), "..", args.output)
    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
    else:
        results = {}

    if args.model not in results:
        results[args.model] = {}

    results[args.model][key] = {
        "perplexity": ppl,
        "size": size_stats,
        "topk": args.topk,
        "scoring": args.scoring,
        "protected_layers": protected_info,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    del model, original_weights
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
