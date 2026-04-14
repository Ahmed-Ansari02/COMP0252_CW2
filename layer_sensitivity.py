"""
Layer Sensitivity Analysis

Quantizes only early, middle, or late decoder layers to measure
which layers are most sensitive to quantization.

OPT decoder layers are split into three equal groups:
  - early:  layers 0 .. N/3-1
  - middle: layers N/3 .. 2N/3-1
  - late:   layers 2N/3 .. N-1

Usage:
    python layer_sensitivity.py --model facebook/opt-125m --bits 4 --grid_type hybrid --gamma 0.15
    python layer_sensitivity.py --model facebook/opt-125m --bits 4 --grid_type uniform
    python layer_sensitivity.py --models facebook/opt-125m facebook/opt-350m facebook/opt-1.3b
"""

import argparse
import json
import os
import torch
from transformers import OPTForCausalLM, AutoTokenizer

from cdf_grid import (build_uniform_grid, build_cdf_grid, build_hybrid_grid,
                       quantize_to_grid, quantize_standard_rtn_row)
from rtn_baseline import load_model, restore_weights, tokenize_dataset, evaluate_perplexity


def get_layer_groups(model):
    """Split decoder layers into early/middle/late thirds."""
    num_layers = model.config.num_hidden_layers
    third = num_layers // 3
    remainder = num_layers % 3

    # Distribute remainder layers to later groups
    early_end = third
    middle_end = third * 2 + (1 if remainder >= 2 else 0)

    groups = {
        "early": list(range(0, early_end)),
        "middle": list(range(early_end, middle_end)),
        "late": list(range(middle_end, num_layers)),
    }
    return groups


def quantize_layers(model, layer_indices, bits, grid_type="hybrid", gamma=0.15,
                    bg_bits=3):
    """
    Quantize Linear layers within the specified decoder layer indices at `bits`,
    and all other decoder Linear layers at `bg_bits` (background bit width).

    Args:
        model: OPT model (modified in-place)
        layer_indices: list of decoder layer indices to quantize at `bits`
        bits: bit width for target layers
        grid_type: "uniform", "cdf", or "hybrid"
        gamma: mixing coefficient for hybrid grid
        bg_bits: bit width for non-target decoder layers (default 3)
    """
    num_layers = model.config.num_hidden_layers
    all_decoder_prefixes = tuple(f"model.decoder.layers.{i}." for i in range(num_layers))
    target_prefixes = tuple(f"model.decoder.layers.{i}." for i in layer_indices)

    target_count = 0
    bg_count = 0
    skipped_count = 0

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        is_target = name.startswith(target_prefixes)
        is_decoder = name.startswith(all_decoder_prefixes)

        if not is_decoder:
            skipped_count += module.weight.numel()
            continue

        b = bits if is_target else bg_bits
        num_levels = 2 ** b

        W = module.weight.data.clone()
        for row_idx in range(W.shape[0]):
            row = W[row_idx]
            if grid_type == "uniform":
                W[row_idx] = quantize_standard_rtn_row(row, b)
            else:
                row_f32 = row.float()
                if grid_type == "cdf":
                    grid = build_cdf_grid(row_f32, num_levels)
                elif grid_type == "hybrid":
                    grid = build_hybrid_grid(row_f32, num_levels, gamma)
                else:
                    raise ValueError(f"Unknown grid_type: {grid_type}")
                W[row_idx] = quantize_to_grid(row_f32, grid).to(W.dtype)
        module.weight.data = W

        if is_target:
            target_count += W.numel()
        else:
            bg_count += W.numel()

    return target_count, bg_count, skipped_count


def make_key(grid_type, gamma, bits, group_name, bg_bits=None):
    if grid_type == "hybrid":
        base = f"hybrid_gamma{gamma}_{bits}bit_{group_name}"
    else:
        base = f"{grid_type}_{bits}bit_{group_name}"
    if bg_bits is not None and bg_bits != bits:
        base += f"_bg{bg_bits}bit"
    return base


def main():
    parser = argparse.ArgumentParser(description="Layer sensitivity analysis")
    parser.add_argument("--models", nargs="+",
                        default=["facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b"])
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--grid_type", type=str, default="hybrid",
                        choices=["uniform", "cdf", "hybrid"])
    parser.add_argument("--gamma", type=float, default=0.15)
    parser.add_argument("--bg_bits", type=int, default=3,
                        help="Bit width for non-target decoder layers (default 3)")
    parser.add_argument("--output", type=str, default="layer_sensitivity_results.json")
    args = parser.parse_args()

    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)
    else:
        results = {}

    for model_name in args.models:
        if model_name not in results:
            results[model_name] = {}

        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model, original_weights = load_model(model_name)
        input_ids = tokenize_dataset(tokenizer)

        # FP16 baseline
        if "fp16" not in results[model_name]:
            print(f"\n  [FP16 baseline]")
            ppl = evaluate_perplexity(model, input_ids=input_ids)
            results[model_name]["fp16"] = ppl
            print(f"  Perplexity: {ppl:.2f}")
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)

        # All layers at target bits (for reference)
        all_key = make_key(args.grid_type, args.gamma, args.bits, "all")
        if all_key not in results[model_name]:
            print(f"\n  [All layers at {args.bits}-bit] {all_key}")
            restore_weights(model, original_weights)
            all_layers = list(range(model.config.num_hidden_layers))
            t_count, bg_count, s_count = quantize_layers(
                model, all_layers, args.bits, args.grid_type, args.gamma,
                bg_bits=args.bits)
            print(f"  Quantized {t_count:,} weights at {args.bits}-bit")
            ppl = evaluate_perplexity(model, input_ids=input_ids)
            results[model_name][all_key] = ppl
            print(f"  Perplexity: {ppl:.2f}")
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)

        # All layers at bg bits (for reference)
        all_bg_key = make_key(args.grid_type, args.gamma, args.bg_bits, "all")
        if all_bg_key not in results[model_name]:
            print(f"\n  [All layers at {args.bg_bits}-bit] {all_bg_key}")
            restore_weights(model, original_weights)
            all_layers = list(range(model.config.num_hidden_layers))
            t_count, bg_count, s_count = quantize_layers(
                model, all_layers, args.bg_bits, args.grid_type, args.gamma,
                bg_bits=args.bg_bits)
            print(f"  Quantized {t_count:,} weights at {args.bg_bits}-bit")
            ppl = evaluate_perplexity(model, input_ids=input_ids)
            results[model_name][all_bg_key] = ppl
            print(f"  Perplexity: {ppl:.2f}")
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)

        # Per-group quantization
        layer_groups = get_layer_groups(model)
        print(f"\n  Layer groups: { {k: f'{v[0]}-{v[-1]}' for k, v in layer_groups.items()} }")

        for group_name, layer_indices in layer_groups.items():
            key = make_key(args.grid_type, args.gamma, args.bits, group_name, args.bg_bits)

            if key in results[model_name]:
                ppl_val = results[model_name][key]
                if isinstance(ppl_val, dict):
                    ppl_val = ppl_val["perplexity"]
                print(f"  [skip] {key} already computed ({ppl_val:.2f})")
                continue

            print(f"\n  [{group_name}] layers {layer_indices[0]}-{layer_indices[-1]} at {args.bits}-bit, rest at {args.bg_bits}-bit")
            restore_weights(model, original_weights)
            t_count, bg_count, s_count = quantize_layers(
                model, layer_indices, args.bits, args.grid_type, args.gamma,
                bg_bits=args.bg_bits)
            print(f"  Target: {t_count:,} weights at {args.bits}-bit, background: {bg_count:,} at {args.bg_bits}-bit")

            ppl = evaluate_perplexity(model, input_ids=input_ids)
            results[model_name][key] = ppl
            print(f"  Perplexity: {ppl:.2f}")

            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)

        del model, original_weights
        torch.cuda.empty_cache()

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    models = list(results.keys())
    header = f"{'Method':<40}" + "".join(f"{m.split('/')[-1]:>12}" for m in models)
    print(header)
    print("-" * len(header))

    key_order = ["fp16",
                 make_key(args.grid_type, args.gamma, args.bits, "all"),
                 make_key(args.grid_type, args.gamma, args.bg_bits, "all"),
                 make_key(args.grid_type, args.gamma, args.bits, "early", args.bg_bits),
                 make_key(args.grid_type, args.gamma, args.bits, "middle", args.bg_bits),
                 make_key(args.grid_type, args.gamma, args.bits, "late", args.bg_bits)]

    for key in key_order:
        row = f"{key:<40}"
        for m in models:
            val = results[m].get(key)
            if val is None:
                row += f"{'N/A':>12}"
            elif isinstance(val, dict):
                row += f"{val['perplexity']:>12.2f}"
            else:
                row += f"{val:>12.2f}"
        print(row)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
