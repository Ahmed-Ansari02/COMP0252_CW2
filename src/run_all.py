"""
Main Experiment Runner — Selective Layer-Level Outlier Protection

Orchestrates the full evaluation pipeline:
  1. Profile models (compute per-layer weight statistics)
  2. Sweep top-k × scoring methods (kurtosis, outlier_count, range/σ)
  3. Evaluate perplexity on WikiText-2
  4. Save results incrementally

Usage:
    python -m src.run_all                                      # full sweep on opt-125m
    python -m src.run_all --models facebook/opt-125m facebook/opt-350m
    python -m src.run_all --scoring kurtosis                   # single scoring method
    python -m src.run_all --skip_profile                       # reuse existing profiles
"""

import argparse
import json
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.layer_profiler import profile_model, print_profile_summary
from src.selective_quantize import (
    quantize_model_selective, make_selective_key, get_topk_layers
)
from rtn_baseline import (
    load_model, restore_weights, tokenize_dataset,
    evaluate_perplexity, quantize_model_rtn
)


MODELS = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
]

SCORING_METHODS = ["kurtosis", "outlier_fraction", "range_sigma", "variance",
                   "bookend", "random"]

# Default quantization config (best from prior experiments)
BITS = 4
GRID_TYPE = "hybrid"
GAMMA = 0.5
OUTLIER_PCT = 1.0


def get_topk_values(total_layers: int) -> list:
    """
    Generate a range of top-k values to sweep.
    Includes 0%, 10%, 25%, 50%, 75%, 100% of layers as well as a few
    fine-grained values near the low end.
    """
    fractions = [0.25, 0.50, 0.75]
    topk_set = set()

    for f in fractions:
        k = int(round(f * total_layers))
        topk_set.add(min(k, total_layers))
    return sorted(topk_set)


def load_or_create_profile(model_name: str, results_dir: str,
                            skip_profile: bool = False) -> dict:
    """Load an existing profile or create a new one."""
    model_short = model_name.split("/")[-1]
    profile_path = os.path.join(results_dir, f"{model_short}_profile.json")

    if skip_profile and os.path.exists(profile_path):
        print(f"Loading existing profile from {profile_path}")
        with open(profile_path) as f:
            return json.load(f)

    profile = profile_model(model_name)
    print_profile_summary(profile)

    os.makedirs(results_dir, exist_ok=True)
    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"Profile saved to {profile_path}")

    return profile


def load_results(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_results(results: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def run_baselines(model, original_weights, input_ids, model_name,
                   results, results_path):
    """Run FP16 and no-protection baselines if not already computed."""
    if "fp16" not in results.get(model_name, {}):
        print(f"\n{'='*60}")
        print(f"[Baseline: FP16] {model_name}")
        print(f"{'='*60}")
        restore_weights(model, original_weights)
        ppl = evaluate_perplexity(model, input_ids=input_ids)
        if model_name not in results:
            results[model_name] = {}
        results[model_name]["fp16"] = ppl
        print(f"  FP16 perplexity: {ppl:.2f}")
        save_results(results, results_path)

    no_protect_key = f"no_protection_{GRID_TYPE}_g{GAMMA}_{BITS}bit"
    if no_protect_key not in results.get(model_name, {}):
        print(f"\n{'='*60}")
        print(f"[Baseline: No Protection] {model_name}")
        print(f"{'='*60}")
        _, size_stats = quantize_model_rtn(
            model_name, BITS, GRID_TYPE, GAMMA,
            protect_outliers=False,
            model=model, original_weights=original_weights
        )
        ppl = evaluate_perplexity(model, input_ids=input_ids)
        results[model_name][no_protect_key] = {
            "perplexity": ppl,
            "size": size_stats,
        }
        print(f"  No-protection perplexity: {ppl:.2f}")
        save_results(results, results_path)

    all_protect_key = f"all_protection_{GRID_TYPE}_g{GAMMA}_{BITS}bit_op{OUTLIER_PCT}"
    if all_protect_key not in results.get(model_name, {}):
        print(f"\n{'='*60}")
        print(f"[Baseline: All Layers Protected] {model_name}")
        print(f"{'='*60}")
        _, size_stats = quantize_model_rtn(
            model_name, BITS, GRID_TYPE, GAMMA,
            protect_outliers=True, outlier_percentile=OUTLIER_PCT,
            model=model, original_weights=original_weights
        )
        ppl = evaluate_perplexity(model, input_ids=input_ids)
        results[model_name][all_protect_key] = {
            "perplexity": ppl,
            "size": size_stats,
        }
        print(f"  All-protected perplexity: {ppl:.2f}")
        save_results(results, results_path)


def run_selective_sweep(model, original_weights, input_ids, model_name,
                         profile, scoring_methods, results, results_path):
    """Sweep top-k values for each scoring method."""
    total_layers = profile["total_linear_layers"]
    topk_values = get_topk_values(total_layers)

    for scoring in scoring_methods:
        print(f"\n{'='*60}")
        print(f"SCORING METHOD: {scoring}")
        print(f"{'='*60}")

        for topk in topk_values:
            key = make_selective_key(scoring, topk, BITS,
                                     GRID_TYPE, GAMMA, OUTLIER_PCT)

            if key in results.get(model_name, {}):
                existing = results[model_name][key]
                ppl_val = existing["perplexity"]
                print(f"  [skip] {key} already computed ({ppl_val:.2f})")
                continue

            print(f"\n--- {key} ---")

            model_q, size_stats, protected_info = quantize_model_selective(
                model, original_weights, profile,
                BITS, GRID_TYPE, GAMMA,
                topk, scoring, OUTLIER_PCT
            )
            ppl = evaluate_perplexity(model, input_ids=input_ids)

            if model_name not in results:
                results[model_name] = {}

            results[model_name][key] = {
                "perplexity": ppl,
                "size": size_stats,
                "topk": topk,
                "total_layers": total_layers,
                "topk_fraction": round(topk / total_layers, 4),
                "scoring": scoring,
                "protected_layers": [p["name"] for p in protected_info],
            }
            print(f"  Perplexity: {ppl:.2f}")
            save_results(results, results_path)


def print_summary(results: dict):
    """Print a formatted summary of all selective experiments."""
    print(f"\n{'='*80}")
    print("SELECTIVE QUANTIZATION RESULTS SUMMARY")
    print(f"{'='*80}")

    for model_name in sorted(results.keys()):
        model_results = results[model_name]
        print(f"\n--- {model_name} ---")
        print(f"  {'Method':<65} {'PPL':>8}  {'Bits/P':>7}  {'Size MB':>8}")
        print(f"  {'-'*90}")

        for key in sorted(model_results.keys()):
            val = model_results[key]
            if isinstance(val, dict):
                ppl = val["perplexity"]
                bits_p = val.get("size", {}).get("effective_bits_per_param", "?")
                size = val.get("size", {}).get("effective_size_mb", "?")
                print(f"  {key:<65} {ppl:>8.2f}  {bits_p:>7}  {size:>8}")
            else:
                print(f"  {key:<65} {val:>8.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Run selective outlier protection experiments")
    parser.add_argument("--models", nargs="+", default=["facebook/opt-350m"],
                        help="Models to evaluate")
    parser.add_argument("--scoring", nargs="+", default=SCORING_METHODS,
                        choices=SCORING_METHODS,
                        help="Scoring methods to sweep")
    parser.add_argument("--skip_profile", action="store_true",
                        help="Reuse existing profile JSON if available")
    parser.add_argument("--skip_baselines", action="store_true",
                        help="Skip FP16 and all-protection baselines")
    parser.add_argument("--output", type=str,
                        default="results/selective_results.json",
                        help="Path to save results")
    args = parser.parse_args()

    base_dir = os.path.join(os.path.dirname(__file__), "..")
    results_dir = os.path.join(base_dir, "results")
    results_path = os.path.join(base_dir, args.output)

    results = load_results(results_path)

    for model_name in args.models:
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name}")
        print(f"{'='*80}")

        # Step 1: Profile
        profile = load_or_create_profile(model_name, results_dir,
                                          skip_profile=args.skip_profile)

        # Step 2: Load model + data
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model, original_weights = load_model(model_name)
        # print(f'Model device: {next(model.parameters()).device}')
        input_ids = tokenize_dataset(tokenizer)

        if model_name not in results:
            results[model_name] = {}

        # Step 3: Baselines
        if not args.skip_baselines:
            run_baselines(model, original_weights, input_ids, model_name,
                          results, results_path)

        # Step 4: Selective sweep
        run_selective_sweep(model, original_weights, input_ids, model_name,
                            profile, args.scoring, results, results_path)

        # Free GPU
        del model, original_weights
        torch.cuda.empty_cache()

    # Summary
    print_summary(results)
    print(f"\nAll results saved to {results_path}")


if __name__ == "__main__":
    main()
