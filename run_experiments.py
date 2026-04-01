"""
Full Experiment Runner

Runs all RTN experiments across models, bit widths, and grid types,
saving incremental results to results.json.

Usage:
    python run_experiments.py                          # all models
    python run_experiments.py --models facebook/opt-125m  # single model
    python run_experiments.py --skip_fp16              # skip FP16 baselines
    python run_experiments.py --bits 4                 # only 4-bit
"""

import argparse
import json
import os
import torch
from transformers import OPTForCausalLM, AutoTokenizer

from rtn_baseline import (quantize_model_rtn, evaluate_perplexity,
                          load_model, restore_weights, tokenize_dataset)


MODELS = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
]

BITS = [4]

# Grid types: (grid_type, gamma)
GRID_CONFIGS = [
    ("uniform", None),
    ("cdf", None),
    ("hybrid", 0.05),
    ("hybrid", 0.10),
    ("hybrid", 0.15),
    ("hybrid", 0.20),
    ("hybrid", 0.30),
    ("hybrid", 0.40),
    ("hybrid", 0.50),
]


def make_key(grid_type, gamma, bits):
    if grid_type == "hybrid":
        return f"hybrid_gamma{gamma}_{bits}bit_rtn"
    return f"{grid_type}_{bits}bit_rtn"


def load_results(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_results(results, path):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--bits", nargs="+", type=int, default=BITS)
    parser.add_argument("--skip_fp16", action="store_true")
    parser.add_argument("--outlier_sweep", action="store_true",
                        help="Run hybrid+outlier protection sweep from 1%% to 10%%")
    parser.add_argument("--gamma", type=float, default=0.15,
                        help="Gamma for outlier sweep (default 0.15)")
    parser.add_argument("--output", type=str, default="results.json")
    args = parser.parse_args()

    results = load_results(args.output)

    for model_name in args.models:
        if model_name not in results:
            results[model_name] = {}

        # Load model, weights, and dataset ONCE per model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model, original_weights = load_model(model_name)
        input_ids = tokenize_dataset(tokenizer)

        # FP16 baseline (evaluate before any quantization)
        if not args.skip_fp16 and "fp16" not in results[model_name]:
            print(f"\n{'='*60}")
            print(f"[FP16 baseline] {model_name}")
            print(f"{'='*60}")
            ppl = evaluate_perplexity(model, input_ids=input_ids)
            results[model_name]["fp16"] = ppl
            print(f"  Result: {ppl:.2f}")
            save_results(results, args.output)

        if args.outlier_sweep:
            # Outlier protection sweep: each hybrid gamma × OP 1%-10%
            hybrid_gammas = [g for gt, g in GRID_CONFIGS if gt == "hybrid"]
            for bits in args.bits:
                for gamma in hybrid_gammas:
                    for pct in range(1, 11):
                        key = f"hybrid_gamma{gamma}_{bits}bit_rtn_op{float(pct)}"

                        if key in results[model_name]:
                            existing = results[model_name][key]
                            ppl_val = existing["perplexity"] if isinstance(existing, dict) else existing
                            print(f"  [skip] {key} already computed ({ppl_val:.2f})")
                            continue

                        print(f"\n{'='*60}")
                        print(f"[{key}] {model_name}")
                        print(f"{'='*60}")

                        _, size_stats = quantize_model_rtn(
                            model_name, bits, "hybrid", gamma,
                            protect_outliers=True, outlier_percentile=float(pct),
                            model=model, original_weights=original_weights)
                        ppl = evaluate_perplexity(model, input_ids=input_ids)

                        results[model_name][key] = {
                            "perplexity": ppl,
                            "size": size_stats,
                        }
                        print(f"  Result: {ppl:.2f}")
                        save_results(results, args.output)
        else:
            # Standard grid config sweep
            for bits in args.bits:
                for grid_type, gamma in GRID_CONFIGS:
                    key = make_key(grid_type, gamma, bits)

                    if key in results[model_name]:
                        existing = results[model_name][key]
                        ppl_val = existing["perplexity"] if isinstance(existing, dict) else existing
                        print(f"  [skip] {key} already computed ({ppl_val:.2f})")
                        continue

                    print(f"\n{'='*60}")
                    print(f"[{key}] {model_name}")
                    print(f"{'='*60}")

                    _, size_stats = quantize_model_rtn(
                        model_name, bits, grid_type,
                        gamma if gamma is not None else 0.15,
                        model=model, original_weights=original_weights)
                    ppl = evaluate_perplexity(model, input_ids=input_ids)

                    results[model_name][key] = {
                        "perplexity": ppl,
                        "size": size_stats,
                    }
                    print(f"  Result: {ppl:.2f}")
                    save_results(results, args.output)

        # Free model before loading the next one
        del model, original_weights
        torch.cuda.empty_cache()

    print(f"\nAll results saved to {args.output}")
    print_summary(results)


def print_summary(results):
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    key_order = [
        "fp16",
        "uniform_4bit_rtn", "cdf_4bit_rtn",
        "hybrid_gamma0.05_4bit_rtn", "hybrid_gamma0.10_4bit_rtn",
        "hybrid_gamma0.15_4bit_rtn", "hybrid_gamma0.20_4bit_rtn",
        "hybrid_gamma0.30_4bit_rtn",
        "uniform_3bit_rtn", "cdf_3bit_rtn",
        "hybrid_gamma0.15_3bit_rtn",
    ]

    models = list(results.keys())
    header = f"{'Method':<35}" + "".join(f"{m.split('/')[-1]:>12}" for m in models)
    print(header)
    print("-" * len(header))

    all_keys = set()
    for m in models:
        all_keys |= set(results[m].keys())

    for key in key_order:
        if key not in all_keys:
            continue
        row = f"{key:<35}"
        for m in models:
            val = results[m].get(key)
            if val is None:
                row += f"{'N/A':>12}"
            elif isinstance(val, dict):
                row += f"{val['perplexity']:>12.2f}"
            else:
                row += f"{val:>12.2f}"
        print(row)


if __name__ == "__main__":
    main()
