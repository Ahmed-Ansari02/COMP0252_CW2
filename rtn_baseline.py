"""
RTN (Round-To-Nearest) Quantization Baselines

Quantizes OPT models using uniform, CDF, or hybrid grids without
GPTQ compensation. Run independently of the GPTQ repo.

Usage:
    python rtn_baseline.py --model facebook/opt-125m --bits 4 --grid_type cdf
    python rtn_baseline.py --model facebook/opt-125m --bits 4 --grid_type hybrid --gamma 0.15
    python rtn_baseline.py --model facebook/opt-125m --bits 4 --grid_type uniform
    python rtn_baseline.py --model facebook/opt-125m --bits 4 --grid_type cdf --protect_outliers
    python rtn_baseline.py --model facebook/opt-125m --bits 4 --grid_type cdf --protect_outliers --outlier_percentile 0.5
"""

import argparse
import json
import os
import torch
from transformers import OPTForCausalLM, AutoTokenizer
from datasets import load_dataset

from cdf_grid import (build_uniform_grid, build_cdf_grid, build_hybrid_grid,
                       quantize_to_grid, quantize_row_with_outlier_protection,
                       quantize_standard_rtn_row)


def load_model(model_name: str):
    """Load a model and cache its original Linear weights for fast restore."""
    print(f"  Loading {model_name}...")
    model = OPTForCausalLM.from_pretrained(model_name, dtype=torch.float16)
    model = model.cuda()
    # Cache original weights so we can restore after each quantization run
    original_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            original_weights[name] = module.weight.data.clone()
    return model, original_weights


def restore_weights(model, original_weights):
    """Restore model Linear weights to their original FP16 values."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name in original_weights:
            module.weight.data.copy_(original_weights[name])


def tokenize_dataset(tokenizer, dataset_name="wikitext",
                     dataset_config="wikitext-2-raw-v1"):
    """Tokenize WikiText-2 once for reuse across evaluations."""
    print("  Tokenizing WikiText-2...")
    dataset = load_dataset(dataset_name, dataset_config, split="test")
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    return encodings.input_ids


def quantize_model_rtn(model_name: str, bits: int, grid_type: str = "uniform",
                        gamma: float = 0.15, protect_outliers: bool = False,
                        outlier_percentile: float = 1.0,
                        model=None, original_weights=None):
    """
    Quantize all linear layers of a model using round-to-nearest
    with either uniform or CDF grids.

    Args:
        model_name: HuggingFace model name (used only if model is None)
        bits: bit width (3 or 4)
        grid_type: "uniform", "cdf", or "hybrid"
        gamma: mixing coefficient for hybrid grid
        protect_outliers: if True, keep top/bottom outlier_percentile% of
                          weights at FP16 (LLM.int8()-style outlier handling)
        outlier_percentile: percentage of weights at each tail to keep in FP16
        model: pre-loaded model (if None, loads from model_name)
        original_weights: cached original weights for restore (if None, no restore)

    Returns:
        (model, size_stats) — if model was passed in, it's modified in-place
    """
    loaded_fresh = False
    if model is None:
        model, original_weights = load_model(model_name)
        loaded_fresh = True
    elif original_weights is not None:
        restore_weights(model, original_weights)

    num_levels = 2 ** bits
    outlier_str = f" + outlier protection (p={outlier_percentile}%)" if protect_outliers else ""
    print(f"  Quantizing to {bits}-bit with {grid_type} grid{outlier_str}...")

    # Track size statistics
    total_quantized_weights = 0
    total_outlier_weights = 0
    total_unquantized_weights = 0  # biases, embeddings, layernorms

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            W = module.weight.data.clone()

            for row_idx in range(W.shape[0]):
                row = W[row_idx]

                if protect_outliers:
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

    # Count unquantized params (biases, embeddings, layernorms, etc.)
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

    print(f"  Model size: {size_stats['effective_size_mb']} MB "
          f"(FP16: {size_stats['fp16_size_mb']} MB, "
          f"avg {size_stats['effective_bits_per_param']} bits/param)")

    return model, size_stats


def evaluate_perplexity(model, tokenizer=None, input_ids=None,
                         dataset_name="wikitext",
                         dataset_config="wikitext-2-raw-v1", max_length=2048):
    """
    Evaluate perplexity on WikiText-2.
    Standard evaluation protocol following GPTQ / LLM.int8() papers.

    Args:
        model: the model to evaluate
        tokenizer: tokenizer (used only if input_ids is None)
        input_ids: pre-tokenized input IDs (if None, tokenizes from dataset)
        max_length: context window size
    """
    print("  Evaluating perplexity on WikiText-2...")
    if input_ids is None:
        input_ids = tokenize_dataset(tokenizer, dataset_name, dataset_config)
    input_ids = input_ids.to(model.device)

    seq_len = input_ids.shape[1]
    nlls = []

    for i in range(0, seq_len - max_length, max_length):
        segment = input_ids[:, i:i + max_length]
        with torch.no_grad():
            outputs = model(segment, labels=segment)
            nlls.append(outputs.loss.item())

    ppl = torch.exp(torch.tensor(nlls).mean()).item()
    return ppl


def make_key(grid_type: str, bits: int, gamma: float,
              protect_outliers: bool, outlier_percentile: float) -> str:
    if grid_type == "hybrid":
        base = f"hybrid_gamma{gamma}_{bits}bit_rtn"
    else:
        base = f"{grid_type}_{bits}bit_rtn"
    if protect_outliers:
        base += f"_op{outlier_percentile}"
    return base


def run_single_experiment(model_name: str, bits: int, grid_type: str,
                           gamma: float, tokenizer=None,
                           protect_outliers: bool = False,
                           outlier_percentile: float = 1.0,
                           model=None, original_weights=None,
                           input_ids=None):
    """Run a single quantization + evaluation experiment.

    If model/original_weights/input_ids are provided, reuses them
    (no reload). Otherwise loads fresh (backwards compatible).
    """
    cached = model is not None
    model, size_stats = quantize_model_rtn(
        model_name, bits, grid_type, gamma,
        protect_outliers, outlier_percentile,
        model=model, original_weights=original_weights)
    ppl = evaluate_perplexity(model, tokenizer=tokenizer, input_ids=input_ids)
    if not cached:
        del model
        torch.cuda.empty_cache()
    return ppl, size_stats


def main():
    parser = argparse.ArgumentParser(description="RTN quantization with CDF/uniform/hybrid grids")
    parser.add_argument("--model", type=str, default="facebook/opt-125m",
                        help="HuggingFace model name")
    parser.add_argument("--bits", type=int, default=4, choices=[3, 4],
                        help="Quantization bit width")
    parser.add_argument("--grid_type", type=str, default="uniform",
                        choices=["uniform", "cdf", "hybrid"],
                        help="Grid type for quantization")
    parser.add_argument("--gamma", type=float, default=0.15,
                        help="Mixing coefficient for hybrid grid (0=pure CDF, 1=pure uniform)")
    parser.add_argument("--protect_outliers", action="store_true",
                        help="Keep outlier weights at FP16 (LLM.int8()-style)")
    parser.add_argument("--outlier_percentile", type=float, default=1.0,
                        help="Percentage of weights at each tail to keep in FP16 (default 1.0%%)")
    parser.add_argument("--output", type=str, default="results.json",
                        help="Path to save/append results JSON")
    parser.add_argument("--fp16_only", action="store_true",
                        help="Only run FP16 baseline (no quantization)")
    args = parser.parse_args()

    # Load or initialize results dict
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)
    else:
        results = {}

    if args.model not in results:
        results[args.model] = {}

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.fp16_only:
        print(f"\n[FP16 baseline] {args.model}")
        model = OPTForCausalLM.from_pretrained(args.model, dtype=torch.float16).cuda()
        ppl = evaluate_perplexity(model, tokenizer)
        del model
        torch.cuda.empty_cache()
        results[args.model]["fp16"] = ppl
        print(f"  FP16 perplexity: {ppl:.2f}")
    else:
        key = make_key(args.grid_type, args.bits, args.gamma,
                       args.protect_outliers, args.outlier_percentile)

        print(f"\n[{key}] {args.model}")
        ppl, size_stats = run_single_experiment(args.model, args.bits, args.grid_type,
                                                args.gamma, tokenizer,
                                                args.protect_outliers, args.outlier_percentile)
        results[args.model][key] = {
            "perplexity": ppl,
            "size": size_stats,
        }
        print(f"  Perplexity: {ppl:.2f}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
