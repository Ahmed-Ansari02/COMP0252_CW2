"""
Pack 4-bit quantized models into actual compressed storage.

Demonstrates real size savings from uniform RTN + outlier protection.
Stores quantized weights as 4-bit integers packed into int32, with
per-row scale/zero_point and a sparse outlier map at FP16.

Can unpack back to FP16 for inference with no accuracy loss vs the
simulated quantization in rtn_baseline.py.

Usage:
    # Pack a model
    python pack_quantized.py --model facebook/opt-125m --output packed_models/opt-125m-4bit

    # Pack and verify (unpack + compare perplexity)
    python pack_quantized.py --model facebook/opt-125m --output packed_models/opt-125m-4bit --verify

    # Load and evaluate a packed model
    python pack_quantized.py --load packed_models/opt-125m-4bit --eval
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from transformers import OPTForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Packing: FP16 weights -> 4-bit integers + metadata
# ---------------------------------------------------------------------------

def quantize_and_pack_layer(weight: torch.Tensor, bits: int,
                            outlier_percentile: float):
    """
    Quantize a weight matrix and pack into compact storage.

    Returns a dict with:
        - packed_indices: int32 tensor with 8x 4-bit values packed per element
        - scale: FP16 per-row scale factors (nrows,)
        - zero_point: FP16 per-row zero points (nrows,)
        - outlier_indices: int32 flat indices of outlier positions
        - outlier_values: FP16 values for outliers
        - shape: original weight shape
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
    # Pad columns to multiple of 8
    pad = (8 - ncols % 8) % 8
    if pad > 0:
        Q_int = torch.nn.functional.pad(Q_int, (0, pad), value=0)

    Q_int = Q_int.reshape(nrows, -1, 8)  # (nrows, ncols_packed, 8)
    packed = torch.zeros(nrows, Q_int.shape[1], dtype=torch.int32,
                         device=weight.device)
    for i in range(8):
        packed |= Q_int[:, :, i].int() << (i * 4)

    # Store outliers sparsely
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
    """
    Unpack a 4-bit packed layer back to FP16 weights.
    """
    packed = packed_data["packed"]
    scale = packed_data["scale"].float()
    zero_point = packed_data["zero_point"].float()
    shape = packed_data["shape"]
    pad = packed_data["pad"]
    nrows, ncols = shape
    maxq = 2 ** bits - 1

    # Unpack 8x 4-bit values from each int32
    ncols_padded = ncols + pad
    Q_int = torch.zeros(nrows, ncols_padded, dtype=torch.float32,
                        device=packed.device)
    for i in range(8):
        col_start = i
        vals = (packed >> (i * 4)) & maxq
        Q_int[:, col_start::8] = vals.float()

    # Remove padding
    Q_int = Q_int[:, :ncols]

    # Dequantize: weight = scale * (q - zero_point)
    W = scale.unsqueeze(1) * (Q_int - zero_point.unsqueeze(1))

    # Restore outliers
    outlier_indices = packed_data["outlier_indices"].long()
    outlier_values = packed_data["outlier_values"].float()
    W_flat = W.flatten()
    W_flat[outlier_indices] = outlier_values
    W = W_flat.reshape(shape)

    return W.half()


# ---------------------------------------------------------------------------
# Pack/save/load a full model
# ---------------------------------------------------------------------------

def pack_model(model, bits, outlier_percentile, decoder_only=True):
    """Pack all quantizable layers in a model."""
    packed_layers = {}
    unquantized_layers = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if decoder_only and not name.startswith("model.decoder.layers."):
                # Keep non-decoder layers at FP16
                unquantized_layers[name] = module.weight.data.cpu().half()
            else:
                packed_layers[name] = quantize_and_pack_layer(
                    module.weight.data, bits, outlier_percentile)
                # Move tensors to CPU for saving
                for k, v in packed_layers[name].items():
                    if isinstance(v, torch.Tensor):
                        packed_layers[name][k] = v.cpu()

    return packed_layers, unquantized_layers


def save_packed_model(packed_layers, unquantized_layers, model_name, bits,
                      outlier_percentile, output_dir):
    """Save packed model to disk."""
    os.makedirs(output_dir, exist_ok=True)

    # Save packed layers
    torch.save(packed_layers, os.path.join(output_dir, "packed_layers.pt"))

    # Save unquantized layers
    torch.save(unquantized_layers, os.path.join(output_dir, "unquantized_layers.pt"))

    # Save metadata
    meta = {
        "model_name": model_name,
        "bits": bits,
        "outlier_percentile": outlier_percentile,
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Compute and report sizes
    packed_size = os.path.getsize(os.path.join(output_dir, "packed_layers.pt"))
    unquant_size = os.path.getsize(os.path.join(output_dir, "unquantized_layers.pt"))
    total_size = packed_size + unquant_size

    print(f"\nPacked model saved to {output_dir}/")
    print(f"  packed_layers.pt:      {packed_size / 1e6:.1f} MB")
    print(f"  unquantized_layers.pt: {unquant_size / 1e6:.1f} MB")
    print(f"  Total on disk:         {total_size / 1e6:.1f} MB")

    return total_size


def load_packed_model(output_dir, device="cuda"):
    """Load a packed model and reconstruct the full model."""
    with open(os.path.join(output_dir, "meta.json")) as f:
        meta = json.load(f)

    packed_layers = torch.load(os.path.join(output_dir, "packed_layers.pt"),
                               map_location="cpu", weights_only=True)
    unquantized_layers = torch.load(os.path.join(output_dir, "unquantized_layers.pt"),
                                    map_location="cpu", weights_only=True)

    # Load the model skeleton (no actual weights needed, we'll overwrite)
    model = OPTForCausalLM.from_pretrained(meta["model_name"],
                                           torch_dtype=torch.float16)
    model.eval()

    # Unpack quantized layers
    bits = meta["bits"]
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if name in packed_layers:
                # Move packed data to device for unpacking
                pd = packed_layers[name]
                for k, v in pd.items():
                    if isinstance(v, torch.Tensor):
                        pd[k] = v.to(device)
                module.weight.data = unpack_layer(pd, bits).to(device)
            elif name in unquantized_layers:
                module.weight.data = unquantized_layers[name].to(device)

    model = model.to(device)
    return model, meta


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_perplexity(model, model_name, device="cuda"):
    """Evaluate WikiText-2 perplexity."""
    from datasets import load_dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    input_ids = testenc.input_ids.to(device)

    seqlen = 2048
    nsamples = input_ids.numel() // seqlen
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
                shift_labels.view(-1))
            nlls.append(loss.float().item())

    ppl = torch.exp(torch.tensor(nlls).mean()).item()
    return ppl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def save_verification_result(results_file, model_name, method_key,
                             fp16_size_mb, packed_size_mb, compression,
                             ppl=None):
    """Append a verification result to the results JSON file."""
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


def pack_pretrained_gptq(saved_dir, bits, device="cuda"):
    """
    Pack an already-quantized GPTQ model (saved as FP16 safetensors) into int4.

    GPTQ uses uniform quantization with no outlier protection on all decoder
    layers. We recover scale/zero per row from the weight min/max and pack
    into 4-bit integers.
    """
    model = OPTForCausalLM.from_pretrained(saved_dir, torch_dtype=torch.float16)
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

        # Check if this layer was quantized: quantized layers have <= 2^bits
        # unique values per row. Non-decoder layers (embed, lm_head) are FP16.
        sample_unique = len(torch.unique(W[0]))
        if sample_unique > maxq + 1:
            unquantized_layers[name] = module.weight.data.cpu().half()
            continue

        # Recover scale/zero from weight range (GPTQ uniform: no outliers)
        wmin = W.min(dim=1, keepdim=True).values
        wmax = W.max(dim=1, keepdim=True).values
        scale = (wmax - wmin) / maxq
        scale[scale == 0] = 1.0
        zero_point = torch.round(-wmin / scale)

        Q_int = torch.clamp(torch.round(W / scale) + zero_point, 0, maxq).to(torch.uint8)

        # Pack 8x 4-bit into int32
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
    torch.cuda.empty_cache()
    return packed_layers, unquantized_layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--load", type=str, default=None,
                        help="Load a previously packed model")
    parser.add_argument("--pack_gptq", type=str, default=None,
                        help="Path to saved GPTQ model dir to pack (e.g. saved_models/opt-125m-gptq-4bit)")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--outlier_percentile", type=float, default=1.0)
    parser.add_argument("--verify", action="store_true",
                        help="Unpack and compare perplexity to simulated quantization")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate perplexity of a loaded packed model")
    parser.add_argument("--results_file", type=str,
                        default="results_quantization_methods/results_pack_verify.json",
                        help="JSON file to append verification results to")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.load:
        # Load and evaluate a packed model
        print(f"Loading packed model from {args.load}...")
        model, meta = load_packed_model(args.load, device)
        print(f"Model: {meta['model_name']}, {meta['bits']}-bit, "
              f"OP={meta['outlier_percentile']}%")

        if args.eval:
            ppl = evaluate_perplexity(model, meta["model_name"], device)
            print(f"Perplexity (unpacked): {ppl:.2f}")
        return

    if args.pack_gptq:
        # Pack an already-quantized GPTQ model
        output_dir = args.output or args.pack_gptq.rstrip("/") + "-packed"

        print(f"Packing GPTQ model from {args.pack_gptq}...")
        fp16_size = sum(
            os.path.getsize(os.path.join(args.pack_gptq, f))
            for f in os.listdir(args.pack_gptq)
            if f.endswith((".safetensors", ".bin"))
        )
        print(f"Original saved size: {fp16_size / 1e6:.1f} MB")

        packed_layers, unquantized_layers = pack_pretrained_gptq(
            args.pack_gptq, args.bits, device)

        total_disk = save_packed_model(
            packed_layers, unquantized_layers, args.model, args.bits,
            0.0, output_dir)

        fp16_size_mb = fp16_size / 1e6
        packed_size_mb = total_disk / 1e6
        compression = fp16_size / total_disk
        print(f"  Compression ratio:     {compression:.2f}x")

        ppl = None
        if args.verify:
            print("\nVerifying: unpacking and evaluating perplexity...")
            model, meta = load_packed_model(output_dir, device)
            ppl = evaluate_perplexity(model, args.model, device)
            print(f"Perplexity (pack -> unpack): {ppl:.2f}")

        save_verification_result(
            args.results_file, args.model, f"uniform_{args.bits}bit_gptq_packed",
            fp16_size_mb, packed_size_mb, compression, ppl)
        return

    # Pack a fresh model with uniform RTN + OP
    output_dir = args.output or f"packed_models/{args.model.split('/')[-1]}-{args.bits}bit"

    print(f"Loading {args.model}...")
    model = OPTForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    model = model.to(device)
    model.eval()

    # Compute FP16 baseline size
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
        packed_layers, unquantized_layers, args.model, args.bits,
        args.outlier_percentile, output_dir)

    packed_size_mb = total_disk / 1e6
    compression = fp16_size_mb / packed_size_mb
    print(f"  FP16 size:             {fp16_size_mb:.1f} MB")
    print(f"  Compression ratio:     {compression:.2f}x")

    ppl = None
    if args.verify:
        print("\nVerifying: unpacking and evaluating perplexity...")
        del model
        torch.cuda.empty_cache()

        model, meta = load_packed_model(output_dir, device)
        ppl = evaluate_perplexity(model, args.model, device)
        print(f"Perplexity (pack -> unpack): {ppl:.2f}")

    save_verification_result(
        args.results_file, args.model,
        f"uniform_{args.bits}bit_rtn_op{args.outlier_percentile}_packed",
        fp16_size_mb, packed_size_mb, compression, ppl)


if __name__ == "__main__":
    main()
