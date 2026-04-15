"""
Zero-Shot Task Evaluation for Quantized Models

Self-contained evaluation of quantized OPT models on zero-shot tasks.
No dependency on the GPTQ zeroShot harness — loads datasets directly
from HuggingFace Hub and does log-likelihood scoring.

Supported tasks: lambada, arc_easy, arc_challenge, piqa, boolq
Supported methods: fp16, hybrid_rtn, gptq

Usage:
    python zeroshot_eval.py --model facebook/opt-125m --method fp16 \
        --tasks lambada,arc_easy,arc_challenge,piqa

    python zeroshot_eval.py --model facebook/opt-125m --method hybrid_rtn \
        --tasks lambada,arc_easy,arc_challenge,piqa --bits 4

    python zeroshot_eval.py --model facebook/opt-125m --method gptq \
        --tasks lambada,arc_easy,arc_challenge,piqa --bits 4
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import OPTForCausalLM, AutoTokenizer
from datasets import load_dataset

# For GPTQ quantization path
GPTQ_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gptq")
sys.path.insert(0, GPTQ_PATH)


# ---------------------------------------------------------------------------
# Dataset loaders — each returns a list of dicts with task-specific fields
# ---------------------------------------------------------------------------

def load_lambada():
    ds = load_dataset("EleutherAI/lambada_openai", "default", split="test")
    return [{"text": row["text"]} for row in ds]


def load_arc(split_name):
    ds = load_dataset("ai2_arc", split_name, split="test")
    items = []
    num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
    for row in ds:
        key = num_to_letter.get(row["answerKey"], row["answerKey"])
        choices = row["choices"]["text"]
        gold = ["A", "B", "C", "D", "E"].index(key)
        items.append({
            "question": row["question"],
            "choices": choices,
            "gold": gold,
        })
    return items


def load_piqa():
    ds = load_dataset("piqa", split="validation")
    items = []
    for row in ds:
        items.append({
            "question": row["goal"],
            "choices": [row["sol1"], row["sol2"]],
            "gold": row["label"],
        })
    return items


def load_boolq():
    ds = load_dataset("super_glue", "boolq", split="validation")
    items = []
    for row in ds:
        items.append({
            "passage": row["passage"],
            "question": row["question"],
            "gold": row["label"],  # 0=False, 1=True
        })
    return items


TASK_LOADERS = {
    "lambada": load_lambada,
    "arc_easy": lambda: load_arc("ARC-Easy"),
    "arc_challenge": lambda: load_arc("ARC-Challenge"),
    "piqa": load_piqa,
    "boolq": load_boolq,
}


# ---------------------------------------------------------------------------
# Scoring — compute log-likelihood of a continuation given a context
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_continuation(model, tokenizer, context_str, continuation_str, device, max_length=2048):
    """
    Compute the total log-probability of continuation tokens given context.
    Returns (log_prob, num_tokens, is_greedy).
    """
    ctx_ids = tokenizer.encode(context_str, add_special_tokens=False)
    cont_ids = tokenizer.encode(continuation_str, add_special_tokens=False)

    # Truncate from left if too long
    all_ids = (ctx_ids + cont_ids)[-(max_length + 1):]
    # We need at least 1 context token
    if len(all_ids) <= len(cont_ids):
        ctx_len = 1
    else:
        ctx_len = len(all_ids) - len(cont_ids)

    input_ids = torch.tensor([all_ids[:-1]], device=device)
    target_ids = torch.tensor([all_ids[1:]], device=device)

    logits = model(input_ids).logits  # (1, seq_len, vocab)
    log_probs = F.log_softmax(logits, dim=-1)

    # Only score the continuation tokens
    cont_start = ctx_len - 1  # offset by 1 due to shifting
    cont_log_probs = log_probs[0, cont_start:, :]
    cont_targets = target_ids[0, cont_start:]

    token_log_probs = cont_log_probs.gather(1, cont_targets.unsqueeze(1)).squeeze(1)
    total_log_prob = token_log_probs.sum().item()

    # Check greedy match
    greedy_tokens = cont_log_probs.argmax(dim=-1)
    is_greedy = (greedy_tokens == cont_targets).all().item()

    return total_log_prob, len(cont_ids), is_greedy


# ---------------------------------------------------------------------------
# Task evaluators
# ---------------------------------------------------------------------------

def eval_lambada(model, tokenizer, device, data):
    """LAMBADA: predict last word. Accuracy = greedy match."""
    correct = 0
    total_nll = 0.0
    n = len(data)

    for item in tqdm(data, desc="LAMBADA"):
        text = item["text"]
        # Split into context and last word
        parts = text.strip().rsplit(" ", 1)
        if len(parts) != 2:
            n -= 1
            continue
        context = "\n" + parts[0]
        target = " " + parts[1]

        log_prob, num_tokens, is_greedy = score_continuation(
            model, tokenizer, context, target, device)
        correct += int(is_greedy)
        total_nll += -log_prob

    acc = correct / n if n > 0 else 0
    ppl = math.exp(total_nll / n) if n > 0 else float("inf")
    return {"acc": round(acc, 4), "ppl": round(ppl, 4), "n": n}


def eval_multiple_choice(model, tokenizer, device, data, task_name):
    """Generic multiple-choice: pick answer with highest log-likelihood."""
    correct = 0
    n = len(data)

    for item in tqdm(data, desc=task_name):
        if task_name in ("arc_easy", "arc_challenge"):
            context = "Question: " + item["question"] + "\nAnswer:"
        elif task_name == "piqa":
            context = "Question: " + item["question"] + "\nAnswer:"
        elif task_name == "boolq":
            context = item["passage"] + "\nQuestion: " + item["question"] + "?\nAnswer:"
            item["choices"] = ["No", "Yes"]
        else:
            context = item["question"]

        best_score = float("-inf")
        best_idx = 0

        for i, choice in enumerate(item["choices"]):
            continuation = " " + choice
            log_prob, num_tokens, _ = score_continuation(
                model, tokenizer, context, continuation, device)
            # Length-normalize
            score = log_prob / num_tokens if num_tokens > 0 else log_prob
            if score > best_score:
                best_score = score
                best_idx = i

        correct += int(best_idx == item["gold"])

    acc = correct / n if n > 0 else 0
    return {"acc": round(acc, 4), "n": n}


# ---------------------------------------------------------------------------
# Quantization methods (reuse existing code)
# ---------------------------------------------------------------------------

def quantize_rtn(model, bits, grid_type, gamma, outlier_percentile, decoder_only):
    """Quantize model in-place using RTN with a given grid type + outlier protection."""
    from cdf_grid import quantize_matrix_batched

    num_levels = 2 ** bits
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if decoder_only and not name.startswith("model.decoder.layers."):
                continue
            W = module.weight.data
            W, _ = quantize_matrix_batched(
                W, num_levels, grid_type=grid_type, gamma=gamma,
                pin_endpoints=True,
                protect_outliers=True,
                outlier_percentile=outlier_percentile)
            module.weight.data = W
    return model


def quantize_gptq(model, model_name, bits, nsamples, seed):
    """Quantize model in-place using GPTQ (uniform grid)."""
    import torch.nn as nn
    from gptq import GPTQ
    from quant import Quantizer
    from datautils import get_loaders, set_seed
    from modelutils import find_layers, DEV

    set_seed(seed)
    dataloader, _ = get_loaders(
        "wikitext2", nsamples=nsamples, seed=seed,
        model=model_name, seqlen=model.seqlen
    )

    dev = DEV
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

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        gptq = {}
        for name in subset:
            gptq[name] = GPTQ(subset[name])
            gptq[name].quantizer = Quantizer()
            gptq[name].quantizer.configure(bits, perchannel=True, sym=False, mse=False)

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f'  Quantizing layer {i} {name} ...')
            gptq[name].fasterquant(percdamp=0.01, groupsize=-1)
            gptq[name].free()

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer, gptq
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_result_key(args):
    if args.method == "fp16":
        return "fp16"
    elif args.method == "gptq":
        return f"uniform_{args.bits}bit_gptq"
    elif args.method == "uniform_rtn":
        return f"uniform_{args.bits}bit_rtn_op{args.outlier_percentile}_deconly"
    else:
        return f"hybrid_gamma{args.gamma}_{args.bits}bit_rtn_op{args.outlier_percentile}_deconly"


def save_quantized_artifacts(model, tokenizer, output_dir):
    """Save a quantized model/tokenizer so evaluation can run in a fresh process."""
    os.makedirs(output_dir, exist_ok=True)
    model = model.cpu()
    torch.cuda.empty_cache()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved quantized model to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot evaluation of quantized OPT models")

    parser.add_argument('--model', type=str, default='facebook/opt-125m')
    parser.add_argument('--method', type=str, default='hybrid_rtn',
                        choices=['hybrid_rtn', 'uniform_rtn', 'gptq', 'fp16'])
    parser.add_argument('--tasks', type=str, default='lambada,arc_easy,arc_challenge,piqa',
                        help='Comma-separated tasks: lambada,arc_easy,arc_challenge,piqa,boolq')
    parser.add_argument('--bits', type=int, default=4, choices=[3, 4])
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--outlier_percentile', type=float, default=1.0)
    parser.add_argument('--decoder_only', action='store_true', default=True)
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=str,
                        default='results_zeroshot/zeroshot_results.json')
    parser.add_argument('--save_quantized_dir', type=str, default=None,
                        help='Directory to save quantized model/tokenizer after quantization')
    parser.add_argument('--load_quantized_dir', type=str, default=None,
                        help='Directory containing a previously saved quantized model/tokenizer')
    parser.add_argument('--save_only', action='store_true',
                        help='Save the quantized model and exit without evaluation')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_names = [t.strip() for t in args.tasks.split(",")]

    for t in task_names:
        if t not in TASK_LOADERS:
            print(f"Unknown task: {t}. Available: {list(TASK_LOADERS.keys())}")
            return

    # --- Load model ---
    load_source = args.load_quantized_dir or args.model
    print(f'\nLoading {load_source}...')

    def skip(*a, **k):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    model = OPTForCausalLM.from_pretrained(load_source, torch_dtype=torch.float16)
    model.seqlen = model.config.max_position_embeddings
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(load_source, use_fast=False)

    # --- Quantize ---
    if args.load_quantized_dir:
        print('Using pre-quantized checkpoint (skipping quantization)')
        model = model.to(device)
        quant_time = 0

    elif args.method == "uniform_rtn":
        print(f'Quantizing with uniform RTN (OP={args.outlier_percentile}%, decoder_only)...')
        tick = time.time()
        model = model.to(device)
        quantize_rtn(model, args.bits, grid_type="uniform", gamma=0.0,
                     outlier_percentile=args.outlier_percentile,
                     decoder_only=args.decoder_only)
        quant_time = time.time() - tick
        print(f'Quantization time: {quant_time:.2f}s')

    elif args.method == "hybrid_rtn":
        print(f'Quantizing with hybrid RTN (γ={args.gamma}, OP={args.outlier_percentile}%)...')
        tick = time.time()
        model = model.to(device)
        quantize_rtn(model, args.bits, grid_type="hybrid", gamma=args.gamma,
                     outlier_percentile=args.outlier_percentile,
                     decoder_only=args.decoder_only)
        quant_time = time.time() - tick
        print(f'Quantization time: {quant_time:.2f}s')

    elif args.method == "gptq":
        print(f'Quantizing with GPTQ (uniform, {args.bits}-bit)...')
        tick = time.time()
        quantize_gptq(model, args.model, args.bits, args.nsamples, args.seed)
        model = model.to(device)
        quant_time = time.time() - tick
        print(f'Quantization time: {quant_time:.2f}s')

    else:
        print('Using FP16 (no quantization)')
        model = model.to(device)
        quant_time = 0

    if args.save_quantized_dir:
        save_quantized_artifacts(model, tokenizer, args.save_quantized_dir)
        if args.save_only:
            del model
            torch.cuda.empty_cache()
            return
        model = model.to(device)

    # --- Load tasks and evaluate ---
    print(f'\nRunning zero-shot evaluation: {task_names}')
    all_task_results = {}

    for task_name in task_names:
        print(f'\n--- Loading {task_name} ---')
        data = TASK_LOADERS[task_name]()
        print(f'  {len(data)} examples')

        if task_name == "lambada":
            result = eval_lambada(model, tokenizer, device, data)
        else:
            result = eval_multiple_choice(model, tokenizer, device, data, task_name)

        all_task_results[task_name] = result
        print(f'  Result: {result}')

    # --- Print summary ---
    print(f'\n{"="*60}')
    print(f'  {args.model} — {args.method}')
    print(f'{"="*60}')
    for task_name, result in all_task_results.items():
        metrics = ", ".join(f"{k}: {v}" for k, v in result.items())
        print(f'  {task_name}: {metrics}')
    if quant_time > 0:
        print(f'\n  Quantization time: {quant_time:.2f}s')

    # --- Save results ---
    key = make_result_key(args)
    output_path = args.output

    if os.path.exists(output_path):
        with open(output_path) as f:
            saved = json.load(f)
    else:
        saved = {}

    if args.model not in saved:
        saved[args.model] = {}

    saved[args.model][key] = {
        "tasks": all_task_results,
        "quantization_time_s": round(quant_time, 3),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(saved, f, indent=2)
    print(f'\nResults saved to {output_path}')

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
