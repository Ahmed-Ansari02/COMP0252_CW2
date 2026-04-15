"""
Gamma Sweep Runner.

Runs hybrid RTN with 1% outlier protection and decoder-only
quantization for a single explicit gamma value across one or more models.

Usage:
    python gamma_sweep.py --models facebook/opt-125m --gamma 0.0
    python gamma_sweep.py --models facebook/opt-125m facebook/opt-350m --gamma 0.3
    python gamma_sweep.py --models facebook/opt-125m --bits 3 --gamma 0.5
"""

import argparse
import json
import os
import time
import torch
from transformers import OPTForCausalLM, AutoTokenizer

from rtn_baseline import (load_model, restore_weights, tokenize_dataset,
                           evaluate_perplexity, quantize_model_rtn)


OUTPUT_PATH = "results_quantization_methods/gamma_sweep.json"


def main():
    parser = argparse.ArgumentParser(
        description="Sweep gamma for hybrid RTN + OP (decoder-only)")
    parser.add_argument("--models", nargs="+",
                        default=["facebook/opt-125m"],
                        help="Models to sweep")
    parser.add_argument("--bits", type=int, default=4, choices=[3, 4])
    parser.add_argument("--gamma", type=float, required=True,
                        help="Single gamma value to run for all passed models")
    parser.add_argument("--outlier_percentile", type=float, default=1.0)
    parser.add_argument("--output", type=str, default=OUTPUT_PATH)
    args = parser.parse_args()

    # Load existing results
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)
    else:
        results = {}

    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"  Gamma run: {model_name}, {args.bits}-bit, gamma={args.gamma}")
        print(f"{'='*60}")

        if model_name not in results:
            results[model_name] = {}

        # Load model once, reuse across gamma values
        model, original_weights = load_model(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        input_ids = tokenize_dataset(tokenizer)

        key = f"hybrid_gamma{args.gamma}_{args.bits}bit_rtn_op{args.outlier_percentile}_deconly"

        if key in results[model_name]:
            print(f"\n  [{key}] already exists, skipping")
            del model, original_weights
            torch.cuda.empty_cache()
            continue

        print(f"\n  [{key}]")

        # Restore original weights
        restore_weights(model, original_weights)

        # Quantize
        torch.cuda.synchronize()
        t0 = time.time()
        model, size_stats = quantize_model_rtn(
            model_name, args.bits, "hybrid", args.gamma,
            protect_outliers=True,
            outlier_percentile=args.outlier_percentile,
            pin_endpoints=True,
            decoder_only=True,
            model=model, original_weights=original_weights)
        torch.cuda.synchronize()
        quant_time = time.time() - t0
        size_stats["quantization_time_s"] = round(quant_time, 3)

        # Evaluate
        ppl = evaluate_perplexity(model, tokenizer=tokenizer, input_ids=input_ids)

        results[model_name][key] = {
            "perplexity": ppl,
            "gamma": args.gamma,
            "size": size_stats,
        }
        print(f"  Perplexity: {ppl:.2f}")

        # Save incrementally
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

        del model, original_weights
        torch.cuda.empty_cache()

    print(f"\nAll results saved to {args.output}")


if __name__ == "__main__":
    main()
