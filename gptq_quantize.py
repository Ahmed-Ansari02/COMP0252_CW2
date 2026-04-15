"""
GPTQ Quantization using the official GPTQ repo

Uses the official GPTQ codebase (cloned into ./gptq/) for:
  - Calibration data loading (datautils.get_loaders)
  - Hessian collection and GPTQ algorithm (gptq.GPTQ)
  - Quantizer parameters (quant.Quantizer, quant.quantize)
  - Layer-by-layer sequential processing (opt.opt_sequential pattern)
  - Evaluation (opt.opt_eval pattern)

This ensures an apples-to-apples comparison: the ONLY difference
between "uniform_4bit_gptq" and "hybrid_4bit_gptq" is the
quantize() call inside fasterquant's inner loop.

Usage:
    # Standard GPTQ (uniform, matching paper exactly):
    python gptq_quantize.py --model facebook/opt-125m --wbits 4

    # Save a quantized checkpoint for later zero-shot evaluation:
    python gptq_quantize.py --model facebook/opt-125m --wbits 4 \
        --save_quantized_dir saved_models/opt-125m-gptq-4bit

    # GPTQ with hybrid CDF grid:
    python gptq_quantize.py --model facebook/opt-125m --wbits 4 --grid_type hybrid --gamma 0.5

    # GPTQ with hybrid CDF grid + outlier protection:
    python gptq_quantize.py --model facebook/opt-125m --wbits 4 --grid_type hybrid --gamma 0.5 --protect_outliers --outlier_percentile 1.0
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import transformers

# Add the official GPTQ repo to the path
GPTQ_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gptq")
sys.path.insert(0, GPTQ_PATH)

from gptq import GPTQ
from quant import Quantizer, quantize
from datautils import get_loaders, set_seed
from modelutils import find_layers, DEV

from cdf_grid import (build_cdf_grids_batched, build_hybrid_grids_batched,
                       build_uniform_grids_batched, quantize_to_grids_batched)


def get_opt(model_name):
    """Load OPT model, matching the official repo's get_opt()."""
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model_name, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    return model


def make_cdf_quantize_fn(W_full, bits, grid_type, gamma, pin_endpoints,
                          protect_outliers, outlier_percentile):
    """
    Create a quantize function that uses CDF/hybrid grids instead of
    uniform rounding. Precomputes per-row grids from the full weight
    matrix for efficiency.

    Returns a function with signature: quantize_fn(w_col) -> q_col
    where w_col is shape (nrows,) — a single column of weights.
    """
    num_levels = 2 ** bits
    W_f32 = W_full.float()
    nrows, ncols = W_f32.shape

    if protect_outliers:
        lo = torch.quantile(W_f32, outlier_percentile / 100.0, dim=1, keepdim=True)
        hi = torch.quantile(W_f32, 1.0 - outlier_percentile / 100.0, dim=1, keepdim=True)
        sorted_W = torch.sort(W_f32, dim=1).values
        k = max(1, int(round(ncols * outlier_percentile / 100.0)))
        inlier_sorted = sorted_W[:, k:ncols - k]
        # Build grids from inlier-trimmed sorted weights
        n_inlier = inlier_sorted.shape[1]
        quantile_positions = torch.linspace(
            0.5 / num_levels, 1.0 - 0.5 / num_levels,
            num_levels, device=W_full.device)
        indices = (quantile_positions * (n_inlier - 1)).long().clamp(0, n_inlier - 1)

        if grid_type == "cdf":
            grids = inlier_sorted[:, indices]
            if pin_endpoints:
                grids[:, 0] = inlier_sorted[:, 0]
                grids[:, -1] = inlier_sorted[:, -1]
        elif grid_type == "hybrid":
            cdf_grids = inlier_sorted[:, indices]
            if pin_endpoints:
                cdf_grids[:, 0] = inlier_sorted[:, 0]
                cdf_grids[:, -1] = inlier_sorted[:, -1]
            inlier_min = inlier_sorted[:, 0].unsqueeze(1)
            inlier_max = inlier_sorted[:, -1].unsqueeze(1)
            t = torch.linspace(0, 1, num_levels, device=W_full.device).unsqueeze(0)
            grids = cdf_grids + gamma * (inlier_min + (inlier_max - inlier_min) * t - cdf_grids)
            grids = torch.sort(grids, dim=1).values
        else:
            grids = build_uniform_grids_batched(inlier_sorted, num_levels)

        # Outlier thresholds per row
        lo_flat = lo.squeeze(1)  # (nrows,)
        hi_flat = hi.squeeze(1)  # (nrows,)
    else:
        if grid_type == "cdf":
            grids = build_cdf_grids_batched(W_f32, num_levels, pin_endpoints=pin_endpoints)
        elif grid_type == "hybrid":
            grids = build_hybrid_grids_batched(W_f32, num_levels, gamma, pin_endpoints=pin_endpoints)
        else:
            grids = build_uniform_grids_batched(W_f32, num_levels)
        lo_flat = hi_flat = None

    # grids: (nrows, num_levels), sorted per row
    def quantize_col(w):
        """Quantize a column vector w (nrows,) using precomputed per-row grids."""
        w_2d = w.unsqueeze(1).float()  # (nrows, 1)
        q_2d = quantize_to_grids_batched(w_2d, grids)  # (nrows, 1)
        q = q_2d.squeeze(1)

        if protect_outliers:
            # Keep outlier values unquantized
            outlier_mask = (w < lo_flat) | (w > hi_flat)
            q[outlier_mask] = w[outlier_mask].float()

        return q

    return quantize_col


@torch.no_grad()
def opt_sequential_cdf(model, dataloader, dev, args):
    """
    Layer-by-layer GPTQ quantization, matching the official opt_sequential()
    exactly, but with a pluggable quantize function for CDF/hybrid grids.

    Uses the official GPTQ class for Hessian collection and the GPTQ
    algorithm structure. The only modification is swapping the quantize()
    call when grid_type != "uniform".
    """
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    # Catcher: identical to official repo
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print('Ready.')

    total_error = 0
    total_quantized_weights = 0
    total_outlier_weights = 0
    hessian_time = 0
    quant_time = 0

    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        gptq = {}
        for name in subset:
            gptq[name] = GPTQ(subset[name])
            gptq[name].quantizer = Quantizer()
            gptq[name].quantizer.configure(
                args.wbits, perchannel=True, sym=args.sym, mse=False
            )

        # Collect Hessians — identical to official repo
        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        torch.cuda.synchronize()
        h_start = time.time()
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()
        torch.cuda.synchronize()
        hessian_time += time.time() - h_start

        # Quantize each sublayer
        torch.cuda.synchronize()
        q_start = time.time()
        for name in subset:
            print(i, name)
            print('Quantizing ...')

            if args.grid_type == "uniform":
                # Use the official GPTQ fasterquant — completely unmodified
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=args.groupsize,
                    actorder=args.act_order
                )
            else:
                # Use GPTQ algorithm with CDF/hybrid quantization grid
                error = _fasterquant_cdf(
                    gptq[name], args.wbits, args.grid_type, args.gamma,
                    not args.no_pin_endpoints,
                    args.protect_outliers, args.outlier_percentile,
                    blocksize=args.blocksize, percdamp=args.percdamp,
                    groupsize=args.groupsize, actorder=args.act_order
                )
                total_error += error

            total_quantized_weights += gptq[name].rows * gptq[name].columns
            gptq[name].free()

        torch.cuda.synchronize()
        quant_time += time.time() - q_start

        # Propagate through quantized layer — identical to official repo
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    timing = {
        'hessian_time_s': round(hessian_time, 3),
        'quantization_time_s': round(quant_time, 3),
        'total_time_s': round(hessian_time + quant_time, 3),
        'total_quantized_weights': total_quantized_weights,
    }
    return timing


def _fasterquant_cdf(gptq_layer, bits, grid_type, gamma, pin_endpoints,
                      protect_outliers, outlier_percentile,
                      blocksize=128, percdamp=0.01,
                      groupsize=-1, actorder=False):
    """
    GPTQ's fasterquant algorithm with CDF/hybrid grid quantization.

    This is a near-copy of the official fasterquant() — the ONLY change
    is replacing the `quantize(w, scale, zero, maxq)` call with our
    CDF grid-based quantization.

    Everything else (Hessian processing, Cholesky, error compensation,
    block structure) is identical to the official code.
    """
    layer = gptq_layer
    W = layer.layer.weight.data.clone()
    if isinstance(layer.layer, nn.Conv2d):
        W = W.flatten(1)
    if isinstance(layer.layer, transformers.Conv1D):
        W = W.t()
    W = W.float()

    if not layer.quantizer.ready():
        layer.quantizer.find_params(W, weight=True)

    H = layer.H
    del layer.H
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    if actorder:
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]
        invperm = torch.argsort(perm)

    Losses = torch.zeros_like(W)
    Q = torch.zeros_like(W)

    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(layer.columns, device=layer.dev)
    H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H

    # Precompute per-row CDF/hybrid grids from original weights
    quantize_col = make_cdf_quantize_fn(
        W, bits, grid_type, gamma, pin_endpoints,
        protect_outliers, outlier_percentile
    )

    for i1 in range(0, layer.columns, blocksize):
        i2 = min(i1 + blocksize, layer.columns)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Losses1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for j in range(count):
            w = W1[:, j]
            d = Hinv1[j, j]

            # ---- CDF/HYBRID QUANTIZATION (replaces uniform quantize()) ----
            q = quantize_col(w)
            # ---- END ----

            Q1[:, j] = q
            Losses1[:, j] = (w - q) ** 2 / d ** 2

            err1 = (w - q) / d
            W1[:, j:] -= err1.unsqueeze(1).matmul(Hinv1[j, j:].unsqueeze(0))
            Err1[:, j] = err1

        Q[:, i1:i2] = Q1
        Losses[:, i1:i2] = Losses1 / 2

        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    torch.cuda.synchronize()
    error = torch.sum(Losses).item()
    print('error', error)

    if actorder:
        Q = Q[:, invperm]

    layer.layer.weight.data = Q.reshape(layer.layer.weight.shape).to(
        layer.layer.weight.data.dtype
    )
    return error


@torch.no_grad()
def opt_eval(model, testenc, dev):
    """
    Evaluate perplexity — identical to the official repo's opt_eval(),
    but without the RTN (--nearest) path since we've already quantized.
    """
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f'Perplexity: {ppl.item():.2f}')

    model.config.use_cache = use_cache
    return ppl.item()


def make_key(args):
    """Build results dict key from args."""
    if args.grid_type == "uniform":
        key = f"uniform_{args.wbits}bit_gptq"
    elif args.grid_type == "hybrid":
        key = f"hybrid_gamma{args.gamma}_{args.wbits}bit_gptq"
    elif args.grid_type == "cdf":
        key = f"cdf_{args.wbits}bit_gptq"
    else:
        key = f"{args.grid_type}_{args.wbits}bit_gptq"

    if args.protect_outliers:
        key += f"_op{args.outlier_percentile}"
    if args.no_pin_endpoints:
        key += "_nopin"
    if args.act_order:
        key += "_actorder"
    if args.groupsize != -1:
        key += f"_g{args.groupsize}"
    return key


def save_quantized_artifacts(model, model_name, output_dir):
    """Save a quantized model/tokenizer for reuse in zero-shot evaluation."""
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = model.cpu()
    torch.cuda.empty_cache()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f'Saved quantized model to {output_dir}')


def main():
    parser = argparse.ArgumentParser(
        description="GPTQ quantization using official repo infrastructure")

    parser.add_argument('--model', type=str, default='facebook/opt-125m',
                        help='OPT model to quantize')
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        choices=['wikitext2', 'ptb', 'c4'],
                        help='Calibration dataset (default: wikitext2, matching paper)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for calibration data sampling')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration samples')
    parser.add_argument('--wbits', type=int, default=4, choices=[2, 3, 4],
                        help='Quantization bit width')
    parser.add_argument('--blocksize', type=int, default=128,
                        help='GPTQ block size')
    parser.add_argument('--percdamp', type=float, default=0.01,
                        help='Hessian dampening percentage')
    parser.add_argument('--groupsize', type=int, default=-1,
                        help='Groupsize for quantization (-1 = full row)')
    parser.add_argument('--sym', action='store_true',
                        help='Symmetric quantization')
    parser.add_argument('--act-order', action='store_true',
                        help='Activation order heuristic')

    # CDF/hybrid grid options
    parser.add_argument('--grid_type', type=str, default='uniform',
                        choices=['uniform', 'cdf', 'hybrid'],
                        help='Quantization grid type')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='Hybrid grid mixing coefficient (0=CDF, 1=uniform)')
    parser.add_argument('--no_pin_endpoints', action='store_true',
                        help='Do not pin CDF grid endpoints to min/max')
    parser.add_argument('--protect_outliers', action='store_true',
                        help='Keep outlier weights at FP16')
    parser.add_argument('--outlier_percentile', type=float, default=1.0,
                        help='Percentage of weights at each tail to keep in FP16')

    parser.add_argument('--output', type=str, default='results_quantization_methods/results.json',
                        help='Path to save/append results')
    parser.add_argument('--save_quantized_dir', type=str, default=None,
                        help='Directory to save quantized model/tokenizer for later evaluation')

    args = parser.parse_args()

    set_seed(args.seed)

    # Load model — matching official repo
    print(f'\nLoading {args.model}...')
    model = get_opt(args.model)
    model.eval()

    # Load calibration + test data — from official datautils
    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed,
        model=args.model, seqlen=model.seqlen
    )

    key = make_key(args)
    print(f'\n[{key}] {args.model}')

    # Quantize
    torch.cuda.synchronize()
    total_start = time.time()

    timing = opt_sequential_cdf(model, dataloader, DEV, args)

    torch.cuda.synchronize()
    total_time = time.time() - total_start

    # Evaluate on wikitext2 test set — matching official repo exactly
    model = model.to(DEV)
    ppl = opt_eval(model, testloader, DEV)

    # Size stats
    total_params = sum(p.numel() for p in model.parameters())
    total_quantized = timing['total_quantized_weights']
    total_unquantized = total_params - total_quantized

    quantized_bits = total_quantized * args.wbits
    unquantized_bits = total_unquantized * 16
    effective_size_mb = (quantized_bits + unquantized_bits) / 8 / 1024**2
    effective_avg_bits = (quantized_bits + unquantized_bits) / total_params

    size_stats = {
        "total_params": total_params,
        "quantized_weights": total_quantized,
        "outlier_weights": 0,
        "unquantized_weights": total_unquantized,
        "effective_bits_per_param": round(effective_avg_bits, 3),
        "effective_size_mb": round(effective_size_mb, 2),
        "fp16_size_mb": round(total_params * 16 / 8 / 1024**2, 2),
        "hessian_time_s": timing['hessian_time_s'],
        "quantization_time_s": timing['quantization_time_s'],
        "total_time_s": round(total_time, 3),
    }

    print(f'\n  Model size: {size_stats["effective_size_mb"]} MB '
          f'(FP16: {size_stats["fp16_size_mb"]} MB, '
          f'avg {size_stats["effective_bits_per_param"]} bits/param)')
    print(f'  Hessian time: {timing["hessian_time_s"]}s')
    print(f'  Quantization time: {timing["quantization_time_s"]}s')
    print(f'  Total time: {total_time:.3f}s')
    print(f'  Perplexity: {ppl:.2f}')

    # Save results
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)
    else:
        results = {}

    if args.model not in results:
        results[args.model] = {}

    results[args.model][key] = {
        "perplexity": ppl,
        "size": size_stats,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {args.output}')

    if args.save_quantized_dir:
        save_quantized_artifacts(model, args.model, args.save_quantized_dir)

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
