"""
Pack 4-bit quantized models into actual compressed storage.

Supports two inference paths for packed checkpoints:
1. eager_unpacked: reconstruct full FP16 decoder weights at load time
2. lazy_layerwise: keep weights packed and dequantize inside each layer forward

The lazy runtime is an experimental upper bound on dequantization overhead.
It materializes a temporary FP16 weight and runs a standard GEMM on every
forward pass rather than using fused quantized kernels.
"""

import argparse
import gc
import json
import os
import time
from typing import Callable, Dict, List

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, OPTForCausalLM

from src.packed_runtime import (
    PackedLinearRuntime,
    build_packed_runtime_model,
    load_packed_checkpoint,
)


PACK_FORMAT_VERSION = 2
RUNTIME_NOTE = (
    "Upper-bound runtime: dequantize packed weights inside each layer forward "
    "and run standard FP16/BF16/FP32 GEMM. This is not fused int4 inference."
)


# ---------------------------------------------------------------------------
# Packing: FP16 weights -> 4-bit integers + metadata
# ---------------------------------------------------------------------------

def quantize_and_pack_layer(weight: torch.Tensor, bits: int,
                            outlier_percentile: float):
    """
    Quantize a weight matrix and pack into compact storage.

    Returns a dict with:
        - packed: int32 tensor with 8x 4-bit values packed per element
        - scale: FP16 per-row scale factors (nrows,)
        - zero_point: FP16 per-row zero points (nrows,)
        - outlier_indices: int32 flat indices of outlier positions
        - outlier_values: FP16 values for outliers
        - shape: original weight shape
        - pad: column padding added for packing
    """
    W = weight.float()
    nrows, ncols = W.shape
    maxq = 2 ** bits - 1

    # Detect outliers per row
    lo = torch.quantile(W, outlier_percentile / 100.0, dim=1, keepdim=True)
    hi = torch.quantile(W, 1.0 - outlier_percentile / 100.0, dim=1, keepdim=True)
    outlier_mask = (W < lo) | (W > hi)

    # Compute scale/zero from inliers only
    sorted_W = torch.sort(W, dim=1).values
    k = max(1, int(round(ncols * outlier_percentile / 100.0)))
    inlier_min = sorted_W[:, k].unsqueeze(1)
    inlier_max = sorted_W[:, ncols - k - 1].unsqueeze(1)

    scale = (inlier_max - inlier_min) / maxq
    scale[scale == 0] = 1.0
    zero_point = torch.round(-inlier_min / scale)

    # Quantize to integer indices [0, maxq]
    Q_int = torch.clamp(torch.round(W / scale) + zero_point, 0, maxq).to(torch.uint8)

    # Pack 8x 4-bit values into each int32
    pad = (8 - ncols % 8) % 8
    if pad > 0:
        Q_int = torch.nn.functional.pad(Q_int, (0, pad), value=0)

    Q_int = Q_int.reshape(nrows, -1, 8)
    packed = torch.zeros(nrows, Q_int.shape[1], dtype=torch.int32,
                         device=weight.device)
    for i in range(8):
        packed |= Q_int[:, :, i].int() << (i * 4)

    outlier_flat_indices = outlier_mask.flatten().nonzero(as_tuple=False).squeeze(1)
    outlier_values = W.flatten()[outlier_flat_indices].half()

    return {
        "packed": packed,
        "scale": scale.squeeze(1).half(),
        "zero_point": zero_point.squeeze(1).half(),
        "outlier_indices": outlier_flat_indices.int(),
        "outlier_values": outlier_values,
        "shape": list(weight.shape),
        "pad": pad,
    }


# ---------------------------------------------------------------------------
# Unpacking: 4-bit integers + metadata -> FP16 weights
# ---------------------------------------------------------------------------

def unpack_layer(packed_data: dict, bits: int) -> torch.Tensor:
    """Unpack a packed layer back to FP16 weights."""
    packed = packed_data["packed"]
    scale = packed_data["scale"].float()
    zero_point = packed_data["zero_point"].float()
    shape = packed_data["shape"]
    pad = packed_data["pad"]
    nrows, ncols = shape
    maxq = 2 ** bits - 1

    ncols_padded = ncols + pad
    Q_int = torch.zeros(
        nrows,
        ncols_padded,
        dtype=torch.float32,
        device=packed.device,
    )
    for i in range(8):
        vals = (packed >> (i * 4)) & maxq
        Q_int[:, i::8] = vals.float()

    Q_int = Q_int[:, :ncols]
    W = scale.unsqueeze(1) * (Q_int - zero_point.unsqueeze(1))

    outlier_indices = packed_data["outlier_indices"].long()
    outlier_values = packed_data["outlier_values"].float()
    if outlier_indices.numel() > 0:
        W_flat = W.flatten()
        W_flat[outlier_indices] = outlier_values
        W = W_flat.reshape(shape)

    return W.half()


# ---------------------------------------------------------------------------
# Pack/save/load helpers
# ---------------------------------------------------------------------------

def pack_model(model, bits, outlier_percentile, decoder_only=True):
    """Pack all quantizable layers in a model."""
    packed_layers = {}
    unquantized_layers = {}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        if decoder_only and not name.startswith("model.decoder.layers."):
            unquantized_layers[name] = module.weight.data.cpu().half()
        else:
            packed_layers[name] = quantize_and_pack_layer(
                module.weight.data, bits, outlier_percentile)
            for key, value in packed_layers[name].items():
                if isinstance(value, torch.Tensor):
                    packed_layers[name][key] = value.cpu()

    return packed_layers, unquantized_layers


def packed_checkpoint_size_bytes(output_dir: str) -> int:
    total = 0
    for filename in ("packed_layers.pt", "unquantized_layers.pt", "meta.json"):
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            total += os.path.getsize(path)
    return total


def save_packed_model(
    packed_layers,
    unquantized_layers,
    model_name,
    bits,
    outlier_percentile,
    output_dir,
    decoder_only=True,
):
    """Save packed model to disk."""
    os.makedirs(output_dir, exist_ok=True)

    torch.save(packed_layers, os.path.join(output_dir, "packed_layers.pt"))
    torch.save(unquantized_layers, os.path.join(output_dir, "unquantized_layers.pt"))

    meta = {
        "pack_format_version": PACK_FORMAT_VERSION,
        "model_name": model_name,
        "bits": bits,
        "outlier_percentile": outlier_percentile,
        "decoder_only": decoder_only,
        "quantized_layer_names": sorted(packed_layers.keys()),
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    packed_size = os.path.getsize(os.path.join(output_dir, "packed_layers.pt"))
    unquant_size = os.path.getsize(os.path.join(output_dir, "unquantized_layers.pt"))
    total_size = packed_checkpoint_size_bytes(output_dir)

    print(f"\nPacked model saved to {output_dir}/")
    print(f"  packed_layers.pt:      {packed_size / 1e6:.1f} MB")
    print(f"  unquantized_layers.pt: {unquant_size / 1e6:.1f} MB")
    print(f"  Total on disk:         {total_size / 1e6:.1f} MB")
    return total_size


def load_fp16_model(model_name, device="cuda"):
    model = OPTForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()
    return model


def load_packed_model(output_dir, device="cuda"):
    """Load a packed model and reconstruct the full model eagerly."""
    packed_layers, unquantized_layers, meta = load_packed_checkpoint(output_dir)
    model = OPTForCausalLM.from_pretrained(
        meta["model_name"],
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name in packed_layers:
            packed_data = {}
            for key, value in packed_layers[name].items():
                if isinstance(value, torch.Tensor):
                    packed_data[key] = value.to(device)
                else:
                    packed_data[key] = value
            module.weight.data = unpack_layer(packed_data, meta["bits"]).to(device)
        elif name in unquantized_layers:
            module.weight.data = unquantized_layers[name].to(device)

    model = model.to(device)
    model.eval()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, meta


def load_runtime_model(output_dir, runtime_mode, device):
    if runtime_mode == "eager_unpacked":
        return load_packed_model(output_dir, device)
    if runtime_mode == "lazy_layerwise":
        return build_packed_runtime_model(output_dir, device, unpack_layer)
    raise ValueError(f"Unsupported runtime_mode: {runtime_mode}")


# ---------------------------------------------------------------------------
# Evaluation and validation
# ---------------------------------------------------------------------------

def evaluate_perplexity(model, model_name, device="cuda", max_chunks=None):
    """Evaluate WikiText-2 perplexity, optionally on a limited number of chunks."""
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    input_ids = testenc.input_ids.to(device)

    seqlen = 2048
    nsamples = input_ids.numel() // seqlen
    if max_chunks is not None:
        nsamples = min(nsamples, max_chunks)

    model.eval()
    nlls = []
    with torch.no_grad():
        for i in range(nsamples):
            batch = input_ids[:, i * seqlen:(i + 1) * seqlen]
            out = model(batch)
            shift_logits = out.logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            nlls.append(loss.float().item())

    ppl = torch.exp(torch.tensor(nlls).mean()).item()
    return ppl


def build_fixed_prompt(tokenizer, prompt_length, batch_size, device):
    base_text = (
        "Quantized transformers can trade memory for compute. "
        "This benchmark measures the cost of unpacking weights during inference. "
    )
    text = base_text
    while True:
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        if encoded.input_ids.shape[1] >= prompt_length:
            break
        text += base_text

    input_ids = encoded.input_ids[:, :prompt_length].to(device)
    input_ids = input_ids.repeat(batch_size, 1)
    attention_mask = torch.ones_like(input_ids, device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def sync_device(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def gpu_peak_memory_mb(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(device) / 1e6
    return 0.0


def reset_gpu_peak_memory(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def benchmark_prefill(model, prompt, warmup_runs, timed_runs, device):
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(**prompt)

    times = []
    peak_memories = []
    with torch.no_grad():
        for _ in range(timed_runs):
            reset_gpu_peak_memory(device)
            sync_device(device)
            start = time.perf_counter()
            _ = model(**prompt)
            sync_device(device)
            times.append(time.perf_counter() - start)
            peak_memories.append(gpu_peak_memory_mb(device))

    mean_time = sum(times) / len(times)
    return {
        "mean_time_s": round(mean_time, 6),
        "mean_latency_ms": round(mean_time * 1000.0, 3),
        "mean_peak_gpu_memory_mb": round(sum(peak_memories) / len(peak_memories), 2),
    }


def benchmark_decode(model, tokenizer, prompt, new_tokens, warmup_runs, timed_runs, device):
    generate_kwargs = {
        "input_ids": prompt["input_ids"],
        "attention_mask": prompt["attention_mask"],
        "max_new_tokens": new_tokens,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model.generate(**generate_kwargs)

    times = []
    peak_memories = []
    with torch.no_grad():
        for _ in range(timed_runs):
            reset_gpu_peak_memory(device)
            sync_device(device)
            start = time.perf_counter()
            _ = model.generate(**generate_kwargs)
            sync_device(device)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            peak_memories.append(gpu_peak_memory_mb(device))

    mean_time = sum(times) / len(times)
    return {
        "mean_total_time_s": round(mean_time, 6),
        "mean_decode_ms_per_token": round((mean_time / new_tokens) * 1000.0, 3),
        "mean_peak_gpu_memory_mb": round(sum(peak_memories) / len(peak_memories), 2),
    }


def benchmark_model_load(loader_fn: Callable[[], nn.Module], device: str):
    process = psutil.Process()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    reset_gpu_peak_memory(device)
    sync_device(device)
    start = time.perf_counter()
    model = loader_fn()
    sync_device(device)
    load_time = time.perf_counter() - start

    return model, {
        "load_time_s": round(load_time, 6),
        "peak_gpu_memory_mb": round(gpu_peak_memory_mb(device), 2),
        "host_rss_mb": round(process.memory_info().rss / 1e6, 2),
    }


def environment_info():
    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "transformers_version": __import__("transformers").__version__,
    }


def runtime_loader(mode, output_dir, model_name, device):
    if mode == "hf_fp16":
        return lambda: load_fp16_model(model_name, device)
    if mode == "eager_unpacked":
        return lambda: load_packed_model(output_dir, device)[0]
    if mode == "lazy_layerwise":
        return lambda: build_packed_runtime_model(output_dir, device, unpack_layer)[0]
    raise ValueError(f"Unknown benchmark mode: {mode}")


def benchmark_runtime_modes(output_dir, prompt_lengths, new_tokens, batch_size,
                            warmup_runs, timed_runs, results_file, device):
    _, _, meta = load_packed_checkpoint(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(meta["model_name"], use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    packed_size_mb = round(packed_checkpoint_size_bytes(output_dir) / 1e6, 2)
    all_results = {}

    for mode in ("hf_fp16", "eager_unpacked", "lazy_layerwise"):
        print(f"\nBenchmarking [{mode}] for {meta['model_name']}...")
        loader = runtime_loader(mode, output_dir, meta["model_name"], device)
        model, load_metrics = benchmark_model_load(loader, device)

        prompt_results = {}
        for prompt_length in prompt_lengths:
            prompt = build_fixed_prompt(tokenizer, prompt_length, batch_size, device)
            prompt_results[str(prompt_length)] = {
                "prefill": benchmark_prefill(
                    model, prompt, warmup_runs, timed_runs, device
                ),
                "decode": benchmark_decode(
                    model, tokenizer, prompt, new_tokens, warmup_runs, timed_runs, device
                ),
            }

        all_results[mode] = {
            "environment": environment_info(),
            "runtime_mode": mode,
            "packed_checkpoint_dir": output_dir,
            "packed_checkpoint_size_mb": packed_size_mb,
            "load_metrics": load_metrics,
            "prompt_lengths": prompt_results,
            "note": RUNTIME_NOTE,
        }

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    eager_rss = all_results["eager_unpacked"]["load_metrics"]["host_rss_mb"]
    lazy_rss = all_results["lazy_layerwise"]["load_metrics"]["host_rss_mb"]
    compare_prompt = str(prompt_lengths[0])
    eager_decode = all_results["eager_unpacked"]["prompt_lengths"][compare_prompt]["decode"][
        "mean_decode_ms_per_token"
    ]
    lazy_decode = all_results["lazy_layerwise"]["prompt_lengths"][compare_prompt]["decode"][
        "mean_decode_ms_per_token"
    ]
    sanity_checks = {
        "lazy_lower_post_load_rss_than_eager": lazy_rss < eager_rss,
        "lazy_slower_decode_than_eager": lazy_decode > eager_decode,
    }

    save_runtime_benchmark_results(
        results_file,
        meta["model_name"],
        all_results,
        sanity_checks,
    )
    return all_results, sanity_checks


def validate_runtime_paths(output_dir, device, max_eval_chunks=2):
    packed_layers, _, meta = load_packed_checkpoint(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(meta["model_name"], use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    sample_layer = next(iter(packed_layers.values()))
    unit_results = {}
    for label, use_outliers in (("with_outliers", True), ("without_outliers", False)):
        layer_payload = {}
        for key, value in sample_layer.items():
            if isinstance(value, torch.Tensor):
                layer_payload[key] = value.clone()
            else:
                layer_payload[key] = value
        if not use_outliers:
            layer_payload["outlier_indices"] = torch.tensor([], dtype=torch.int32)
            layer_payload["outlier_values"] = torch.tensor([], dtype=torch.float16)

        runtime = PackedLinearRuntime(layer_payload, meta["bits"], None, unpack_layer).to(device)
        x = torch.randn(2, 3, layer_payload["shape"][1], device=device, dtype=torch.float16)
        reference = F.linear(
            x,
            unpack_layer({
                "packed": layer_payload["packed"].to(device),
                "scale": layer_payload["scale"].to(device),
                "zero_point": layer_payload["zero_point"].to(device),
                "outlier_indices": layer_payload["outlier_indices"].to(device),
                "outlier_values": layer_payload["outlier_values"].to(device),
                "shape": layer_payload["shape"],
                "pad": layer_payload["pad"],
            }, meta["bits"]).to(device),
            None,
        )
        candidate = runtime(x)
        unit_results[label] = torch.allclose(candidate, reference, atol=1e-3, rtol=1e-3)
        if not unit_results[label]:
            raise AssertionError(f"PackedLinearRuntime unit validation failed for {label}.")

    eager_model, _ = load_packed_model(output_dir, device)
    lazy_model, _ = build_packed_runtime_model(output_dir, device, unpack_layer)
    prompt = build_fixed_prompt(tokenizer, 128, 1, device)
    with torch.no_grad():
        eager_logits = eager_model(**prompt).logits
        lazy_logits = lazy_model(**prompt).logits
    logits_match = torch.allclose(eager_logits, lazy_logits, atol=1e-3, rtol=1e-3)
    if not logits_match:
        raise AssertionError("Lazy runtime logits do not match eager-unpacked logits.")

    eager_ppl = evaluate_perplexity(
        eager_model, meta["model_name"], device, max_chunks=max_eval_chunks
    )
    lazy_ppl = evaluate_perplexity(
        lazy_model, meta["model_name"], device, max_chunks=max_eval_chunks
    )
    ppl_delta = abs(eager_ppl - lazy_ppl)
    if ppl_delta > 1e-3:
        raise AssertionError(
            f"Lazy runtime perplexity drift too large: {ppl_delta:.6f}."
        )

    del eager_model
    del lazy_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "unit_results": unit_results,
        "logits_match": logits_match,
        "eager_perplexity": round(eager_ppl, 6),
        "lazy_perplexity": round(lazy_ppl, 6),
        "perplexity_delta": round(ppl_delta, 6),
    }


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def save_verification_result(results_file, model_name, method_key,
                             fp16_size_mb, packed_size_mb, compression,
                             ppl=None):
    """Append a packing verification result to a JSON file."""
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)
    else:
        results = {}

    if model_name not in results:
        results[model_name] = {}

    entry = {
        "fp16_size_mb": round(fp16_size_mb, 2),
        "packed_size_mb": round(packed_size_mb, 2),
        "compression_ratio": round(compression, 2),
    }
    if ppl is not None:
        entry["perplexity_after_unpack"] = round(ppl, 4)

    results[model_name][method_key] = entry
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Result saved to {results_file} [{model_name}][{method_key}]")


def save_runtime_benchmark_results(results_file, model_name, mode_entries, sanity_checks):
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)
    else:
        results = {}

    if model_name not in results:
        results[model_name] = {}
    results[model_name].update(mode_entries)
    results[model_name]["sanity_checks"] = sanity_checks

    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Runtime benchmark results saved to {results_file} [{model_name}]")


# ---------------------------------------------------------------------------
# GPTQ packing
# ---------------------------------------------------------------------------

def pack_pretrained_gptq(saved_dir, bits, device="cuda"):
    """
    Pack an already-quantized GPTQ model (saved as FP16 safetensors) into int4.
    """
    model = OPTForCausalLM.from_pretrained(saved_dir, dtype=torch.float16)
    model = model.to(device)
    model.eval()

    packed_layers = {}
    unquantized_layers = {}
    maxq = 2 ** bits - 1

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        W = module.weight.data.float()
        nrows, ncols = W.shape
        sample_unique = len(torch.unique(W[0]))
        if sample_unique > maxq + 1:
            unquantized_layers[name] = module.weight.data.cpu().half()
            continue

        wmin = W.min(dim=1, keepdim=True).values
        wmax = W.max(dim=1, keepdim=True).values
        scale = (wmax - wmin) / maxq
        scale[scale == 0] = 1.0
        zero_point = torch.round(-wmin / scale)

        Q_int = torch.clamp(torch.round(W / scale) + zero_point, 0, maxq).to(torch.uint8)
        pad = (8 - ncols % 8) % 8
        if pad > 0:
            Q_int = torch.nn.functional.pad(Q_int, (0, pad), value=0)

        Q_int = Q_int.reshape(nrows, -1, 8)
        packed = torch.zeros(nrows, Q_int.shape[1], dtype=torch.int32, device=device)
        for i in range(8):
            packed |= Q_int[:, :, i].int() << (i * 4)

        packed_layers[name] = {
            "packed": packed.cpu(),
            "scale": scale.squeeze(1).cpu().half(),
            "zero_point": zero_point.squeeze(1).cpu().half(),
            "outlier_indices": torch.tensor([], dtype=torch.int32),
            "outlier_values": torch.tensor([], dtype=torch.float16),
            "shape": list(module.weight.shape),
            "pad": pad,
        }

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return packed_layers, unquantized_layers


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def default_results_file(args):
    if args.benchmark_runtime:
        return "results/runtime_overhead.json"
    if args.validate_runtime:
        return "results/runtime_validation.json"
    return "results_quantization_methods/results_pack_verify.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--load", type=str, default=None,
                        help="Load a previously packed model")
    parser.add_argument("--pack_gptq", type=str, default=None,
                        help="Path to saved GPTQ model dir to pack")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--outlier_percentile", type=float, default=1.0)
    parser.add_argument("--runtime_mode", type=str, default="eager_unpacked",
                        choices=["eager_unpacked", "lazy_layerwise"],
                        help="Inference path for a packed checkpoint")
    parser.add_argument("--verify", action="store_true",
                        help="Unpack and compare perplexity to simulated quantization")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate perplexity of a loaded packed model")
    parser.add_argument("--benchmark_runtime", action="store_true",
                        help="Benchmark FP16 vs eager-unpacked vs lazy-layerwise runtime")
    parser.add_argument("--validate_runtime", action="store_true",
                        help="Run runtime correctness checks for the packed checkpoint")
    parser.add_argument("--prompt_length", type=int, nargs="+", default=[128, 512],
                        help="Prompt lengths to benchmark")
    parser.add_argument("--new_tokens", type=int, default=32,
                        help="New tokens for decode benchmark")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for runtime benchmarks")
    parser.add_argument("--warmup_runs", type=int, default=3,
                        help="Warmup runs per benchmark")
    parser.add_argument("--timed_runs", type=int, default=10,
                        help="Timed runs per benchmark")
    parser.add_argument("--max_eval_chunks", type=int, default=None,
                        help="Optional limit on WikiText-2 chunks during eval")
    parser.add_argument("--results_file", type=str, default=None,
                        help="JSON file to append results to")
    args = parser.parse_args()

    args.results_file = args.results_file or default_results_file(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.load:
        if args.benchmark_runtime:
            benchmark_runtime_modes(
                args.load,
                prompt_lengths=args.prompt_length,
                new_tokens=args.new_tokens,
                batch_size=args.batch_size,
                warmup_runs=args.warmup_runs,
                timed_runs=args.timed_runs,
                results_file=args.results_file,
                device=device,
            )
            return

        if args.validate_runtime:
            print(f"Validating packed runtime from {args.load}...")
            results = validate_runtime_paths(
                args.load,
                device=device,
                max_eval_chunks=args.max_eval_chunks or 2,
            )
            os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
            with open(args.results_file, "w") as f:
                json.dump(results, f, indent=2)
            print(json.dumps(results, indent=2))
            return

        print(f"Loading packed model from {args.load} [{args.runtime_mode}]...")
        model, meta = load_runtime_model(args.load, args.runtime_mode, device)
        print(
            f"Model: {meta['model_name']}, {meta['bits']}-bit, "
            f"OP={meta['outlier_percentile']}%, decoder_only={meta['decoder_only']}"
        )

        if args.eval:
            ppl = evaluate_perplexity(
                model,
                meta["model_name"],
                device,
                max_chunks=args.max_eval_chunks,
            )
            print(f"Perplexity ({args.runtime_mode}): {ppl:.2f}")
        return

    if args.pack_gptq:
        output_dir = args.output or args.pack_gptq.rstrip("/") + "-packed"

        print(f"Packing GPTQ model from {args.pack_gptq}...")
        fp16_size = sum(
            os.path.getsize(os.path.join(args.pack_gptq, filename))
            for filename in os.listdir(args.pack_gptq)
            if filename.endswith((".safetensors", ".bin"))
        )
        print(f"Original saved size: {fp16_size / 1e6:.1f} MB")

        packed_layers, unquantized_layers = pack_pretrained_gptq(
            args.pack_gptq, args.bits, device)

        total_disk = save_packed_model(
            packed_layers,
            unquantized_layers,
            args.model,
            args.bits,
            0.0,
            output_dir,
            decoder_only=True,
        )

        fp16_size_mb = fp16_size / 1e6
        packed_size_mb = total_disk / 1e6
        compression = fp16_size / total_disk
        print(f"  Compression ratio:     {compression:.2f}x")

        ppl = None
        if args.verify:
            print("\nVerifying: unpacking and evaluating perplexity...")
            model, meta = load_packed_model(output_dir, device)
            ppl = evaluate_perplexity(
                model,
                meta["model_name"],
                device,
                max_chunks=args.max_eval_chunks,
            )
            print(f"Perplexity (pack -> unpack): {ppl:.2f}")

        save_verification_result(
            args.results_file,
            args.model,
            f"uniform_{args.bits}bit_gptq_packed",
            fp16_size_mb,
            packed_size_mb,
            compression,
            ppl,
        )
        return

    output_dir = args.output or f"packed_models/{args.model.split('/')[-1]}-{args.bits}bit"

    print(f"Loading {args.model}...")
    model = load_fp16_model(args.model, device)
    fp16_params = sum(p.numel() for p in model.parameters())
    fp16_size_mb = fp16_params * 2 / 1e6

    print(f"FP16 model size: {fp16_size_mb:.1f} MB ({fp16_params:,} params)")
    print(f"Packing with {args.bits}-bit uniform + {args.outlier_percentile}% OP...")

    t0 = time.time()
    packed_layers, unquantized_layers = pack_model(
        model, args.bits, args.outlier_percentile, decoder_only=True)
    pack_time = time.time() - t0
    print(f"Pack time: {pack_time:.2f}s")

    total_disk = save_packed_model(
        packed_layers,
        unquantized_layers,
        args.model,
        args.bits,
        args.outlier_percentile,
        output_dir,
        decoder_only=True,
    )

    packed_size_mb = total_disk / 1e6
    compression = fp16_size_mb / packed_size_mb
    print(f"  FP16 size:             {fp16_size_mb:.1f} MB")
    print(f"  Compression ratio:     {compression:.2f}x")

    ppl = None
    if args.verify:
        print("\nVerifying: unpacking and evaluating perplexity...")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model, meta = load_packed_model(output_dir, device)
        ppl = evaluate_perplexity(
            model,
            meta["model_name"],
            device,
            max_chunks=args.max_eval_chunks,
        )
        print(f"Perplexity (pack -> unpack): {ppl:.2f}")

    save_verification_result(
        args.results_file,
        args.model,
        f"uniform_{args.bits}bit_rtn_op{args.outlier_percentile}_packed",
        fp16_size_mb,
        packed_size_mb,
        compression,
        ppl,
    )


if __name__ == "__main__":
    main()
