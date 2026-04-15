# COMP0252 CW2 — Selective Layer-Level Outlier Protection for Quantized LLMs

## Motivation

The [LLM.int8()](https://arxiv.org/abs/2208.07339) paper showed that a small fraction of "outlier" features in transformer weights cause disproportionate quality loss under quantization. Their solution: identify outlier columns at **every** layer and keep them in FP16 while quantizing the rest to INT8.

This works, but protecting every layer is expensive in bits — many layers have near-Gaussian weight distributions and don't actually need outlier protection.

**Our approach:** Instead of blindly protecting all layers, we do a **two-pass** process:

1. **Pass 1 (Profile):** Scan all Linear layer weights and compute statistics that indicate how "outlier-prone" each layer is.
2. **Pass 2 (Selective Quantize):** Rank layers by sensitivity, and apply outlier protection **only to the top-k most sensitive layers**. The rest get fully quantized.

The hypothesis is that we can recover most of the quality benefit of full outlier protection while protecting far fewer layers (e.g., 25–50%), resulting in a better compression–quality tradeoff.

## Method

### Layer Sensitivity Metrics

We profile each Linear layer with three independent scoring methods (each tested separately as its own experiment):

| Metric | What it measures | Why it matters |
|--------|-----------------|----------------|
| **Kurtosis** | Tail heaviness of the weight distribution (kurtosis > 3 = heavier tails than Gaussian) | Layers with heavy tails have extreme values that get crushed by uniform quantization |
| **Outlier Fraction** | Fraction of weights beyond ±3σ from the mean | Direct count of "problem" weights — layers with more outliers lose more information |
| **Range / σ Ratio** | `(max − min) / std` — how wide the range is relative to the bulk | High ratio means a few extreme values stretch the quantization grid, wasting resolution on empty space |

Each metric produces a separate layer ranking. We run the top-k sweep independently for each to determine which metric best identifies the critical layers.

### Selective Quantization

Given a profile and a chosen scoring metric:
1. Rank all Linear layers by the metric (descending = most sensitive first)
2. Select the top-k layers to protect
3. **Protected layers:** Keep the top/bottom α% of weights at FP16, quantize inliers using the chosen grid (hybrid CDF/uniform)
4. **Unprotected layers:** Fully quantize all weights (no FP16 outlier preservation)

### Quantization Grids

We build on the existing CDF-based grid infrastructure from Part 1 of the coursework:

- **Uniform grid:** Standard evenly-spaced levels between min and max
- **CDF grid:** Levels placed at quantile positions (more levels where weights are dense)
- **Hybrid grid:** `(1 − γ) × CDF + γ × Uniform` — balances density-awareness with tail coverage

Default config: **4-bit hybrid with γ = 0.15** (best from prior experiments).

## Project Structure

```
COMP0252_CW2/
├── src/                          # New selective protection code
│   ├── layer_profiler.py         # Pass 1: compute per-layer weight statistics
│   ├── selective_quantize.py     # Pass 2: quantize with top-k layer protection
│   ├── run_all.py                # Main experiment orchestrator
│   └── visualize_selective.py    # Generate all figures
│
├── scripts/
│   └── run.sh                    # One-command pipeline runner
│
├── cdf_grid.py                   # Grid construction (uniform, CDF, hybrid)
├── rtn_baseline.py               # RTN quantization + perplexity evaluation
├── zeroshot_eval.py              # Zero-shot downstream evaluation
├── gptq_quantize.py              # GPTQ quantization + optional checkpoint saving
├── run_experiments.py             # Original grid sweep runner (Part 1)
├── visualize.py                   # Original visualization (Part 1)
├── gptq_cdf_patch.py             # GPTQ integration patch (Part 1)
│
├── results/                      # Output: profile JSONs + experiment results
├── results_zeroshot/             # Output: zero-shot benchmark results
├── saved_models/                 # Saved quantized checkpoints for reuse
├── figures/                      # Output: generated plots
├── results.json                  # Part 1 experiment results
├── 4bits_outlier*.json           # Part 1 outlier sweep results
└── requirements.txt
```

## How to Run

### Prerequisites

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers datasets accelerate matplotlib numpy scipy
```

### Quick Test (single scoring method, 4 top-k values)

```bash
# Pick one scoring method: kurtosis | outlier_fraction | range_sigma
python -m src.run_all \
    --models facebook/opt-125m \
    --scoring kurtosis \
    --quick
```

This runs:
1. Profile opt-125m (compute layer stats → `results/opt-125m_profile.json`)
2. Baselines: FP16, no-protection, all-protection
3. Selective sweep: top-k = 0%, 25%, 50%, 100% of layers
4. Saves results to `results/selective_results.json`

### Full Sweep (all scoring methods, fine-grained top-k)

```bash
# All three scoring methods, ~11 top-k values each
python -m src.run_all --models facebook/opt-125m

# Or use the run script (also generates figures)
bash scripts/run.sh
```

### Generate Figures Only (from existing results)

```bash
python -m src.visualize_selective \
    --results results/selective_results.json \
    --model facebook/opt-125m
```

### Useful Flags

| Flag | Effect |
|------|--------|
| `--quick` | Only test 4 top-k values (0%, 25%, 50%, 100%) instead of 11 |
| `--scoring kurtosis` | Run only one scoring method |
| `--skip_profile` | Reuse existing profile JSON (don't re-profile) |
| `--skip_baselines` | Skip FP16 / no-protection / all-protection baselines |
| `--output path.json` | Custom output path for results |

## Evaluation

All experiments are evaluated on **WikiText-2 perplexity** (standard protocol matching GPTQ / LLM.int8() papers) using 2048-token context windows.

### Baselines
- **FP16:** Unquantized model (upper bound on quality)
- **No protection:** All layers quantized, no outliers kept in FP16
- **All protection (α=1%):** Every layer keeps top/bottom 1% of weights in FP16

### Key Questions
1. **Can we match all-layers-protected quality with only 25–50% of layers protected?**
2. **Which scoring metric best identifies the critical layers?**
3. **Is there a clear "elbow" in the PPL-vs-top-k curve (diminishing returns)?**

### Figures Generated

| Figure | What it shows |
|--------|--------------|
| `{model}_layer_heatmap.png` | Per-layer statistics as a heatmap — which layers are flagged as sensitive |
| `{model}_profile_distributions.png` | Histograms of kurtosis, outlier fraction, etc. across all layers |
| `{model}_ppl_vs_topk.png` | Perplexity vs fraction of layers protected (the key result) |
| `{model}_pareto.png` | Perplexity vs effective bits/param — compression–quality tradeoff |
| `{model}_scoring_comparison.png` | Bar chart comparing scoring methods at each top-k level |

## Quantization Method Comparison (GPTQ vs Hybrid RTN)

We compare two quantization approaches across OPT model scales:

- **GPTQ (uniform):** The standard GPTQ algorithm with uniform quantization grids. Uses Hessian-based error compensation via the official GPTQ codebase (vendored in `gptq/`).
- **Hybrid RTN + OP (decoder-only):** Our hybrid CDF/uniform grid (γ=0.5) with 1% outlier protection, applied only to decoder layers. Simple round-to-nearest with no Hessian computation.

### Perplexity Results (WikiText-2, 4-bit)

| Model | GPTQ uniform | Hybrid RTN + OP | PPL gap |
|-------|-------------|-----------------|---------|
| OPT-125M | 30.51 | 30.85 | +0.34 |
| OPT-350M | 24.13 | **23.20** | **−0.93** |
| OPT-1.3B | 15.39 | 15.53 | +0.14 |
| OPT-2.7B | 13.13 | 13.62 | +0.49 |
| OPT-6.7B | 11.22 | 11.89 | +0.67 |

### Quantization Speed

| Model | GPTQ time (s) | Hybrid RTN + OP time (s) | Speedup |
|-------|--------------|--------------------------|---------|
| OPT-125M | 15.0 | 0.15 | **103x** |
| OPT-350M | 41.4 | 0.26 | **159x** |
| OPT-1.3B | 111.2 | 0.83 | **134x** |
| OPT-2.7B | 209.7 | 1.71 | **123x** |
| OPT-6.7B | 638.6 | 5.20 | **123x** |

### Key Findings

- Hybrid RTN + OP is **100–160x faster** than GPTQ, since it skips Hessian collection and Cholesky-based error compensation entirely.
- Despite the simplicity, it stays within 0.1–0.7 PPL of GPTQ at all scales, and **beats GPTQ at OPT-350M**.
- The effective bits/param is ~0.2 higher due to the 1% of weights kept at FP16 for outlier protection.
- Results are stored in `results_quantization_methods/results.json`.

### Running the Comparison

```bash
# GPTQ (uniform, matching the paper)
python gptq_quantize.py --model facebook/opt-125m --wbits 4

# Hybrid RTN + outlier protection (decoder-only, matching GPTQ scope)
python rtn_baseline.py --model facebook/opt-125m --bits 4 --grid_type hybrid --gamma 0.5 \
    --protect_outliers --outlier_percentile 1.0 --decoder_only
```

## Zero-Shot Evaluation

We also evaluate quantized models on zero-shot downstream tasks using `zeroshot_eval.py`.

Tasks used:
- **LAMBADA:** long-context next-word prediction
- **ARC-Easy / ARC-Challenge:** science multiple-choice QA
- **PIQA:** physical commonsense reasoning

Results are stored in `results_zeroshot/zeroshot_results.json`.

### Zero-Shot Results

| Model | Method | LAMBADA acc | LAMBADA ppl | ARC-Easy | ARC-Challenge | PIQA |
|-------|--------|-------------|-------------|----------|---------------|------|
| OPT-125M | FP16 | 0.3823 | 24.3722 | 0.3859 | 0.2227 | 0.6202 |
| OPT-125M | Hybrid RTN | 0.3078 | 39.4451 | 0.3746 | 0.2133 | 0.6192 |
| OPT-125M | GPTQ | 0.3474 | 34.3699 | 0.3965 | 0.2167 | 0.6175 |
| OPT-350M | Hybrid RTN | 0.4463 | 16.7976 | 0.3834 | 0.2406 | 0.6442 |
| OPT-350M | GPTQ | 0.4386 | 17.0626 | 0.3708 | 0.2312 | 0.6349 |
| OPT-1.3B | Hybrid RTN | 0.5764 | 6.9348 | 0.4798 | 0.2594 | 0.7116 |
| OPT-1.3B | GPTQ | 0.5643 | 7.1945 | 0.4832 | 0.2739 | 0.7100 |
| OPT-2.7B | Hybrid RTN | 0.6041 | 5.9711 | 0.5189 | 0.2918 | 0.7416 |
| OPT-2.7B | GPTQ | 0.6169 | 5.5953 | 0.5299 | 0.2995 | 0.7323 |
| OPT-6.7B | Hybrid RTN | 0.6433 | 4.6621 | 0.5779 | 0.3166 | 0.7530 |
| OPT-6.7B | GPTQ | 0.6625 | 4.6803 | 0.5711 | 0.3328 | 0.7699 |

### Zero-Shot Takeaways

- At **125M**, FP16 is still strongest overall, and GPTQ is clearly better than Hybrid RTN on LAMBADA.
- At **350M**, Hybrid RTN slightly outperforms GPTQ on all four tasks in this benchmark.
- At **1.3B**, the two methods are very close: Hybrid RTN is better on LAMBADA and PIQA, while GPTQ is slightly better on ARC.
- At **2.7B** and **6.7B**, GPTQ is generally stronger on LAMBADA and the harder reasoning tasks, while Hybrid RTN remains competitive and occasionally wins on ARC-Easy or PIQA.

### Running `zeroshot_eval.py`

```bash
# FP16 baseline
python zeroshot_eval.py --model facebook/opt-125m --method fp16 \
    --tasks lambada,arc_easy,arc_challenge,piqa

# Hybrid RTN zero-shot evaluation
python zeroshot_eval.py --model facebook/opt-125m --method hybrid_rtn \
    --bits 4 --gamma 0.5 --outlier_percentile 1.0 \
    --tasks lambada,arc_easy,arc_challenge,piqa

# GPTQ: quantize and save a reusable checkpoint first
python gptq_quantize.py --model facebook/opt-125m --wbits 4 \
    --save_quantized_dir saved_models/opt-125m-gptq-4bit

# GPTQ zero-shot evaluation from a saved checkpoint
python zeroshot_eval.py --model facebook/opt-125m --method gptq \
    --load_quantized_dir saved_models/opt-125m-gptq-4bit \
    --tasks lambada,arc_easy,arc_challenge,piqa
```

Useful flags:

| Flag | Effect |
|------|--------|
| `--tasks lambada,arc_easy,arc_challenge,piqa` | Select which zero-shot tasks to run |
| `--output results_zeroshot/zeroshot_results.json` | Save to a custom results file |
| `--load_quantized_dir path/` | Evaluate a previously saved GPTQ checkpoint |
| `--save_quantized_dir path/` | Save a quantized checkpoint for later reuse |

## Prior Results (Part 1)

For reference, existing results from the grid type / outlier sweep experiments:

| Model | FP16 | Uniform 4-bit | Hybrid γ=0.15 | Hybrid + OP 1% |
|-------|------|---------------|---------------|----------------|
| opt-125m | 27.66 | 40.78 | 47.00 | 30.67 |
| opt-350m | 22.00 | 30.10 | 38.31 | 28.10 |
| opt-1.3b | 14.62 | 50.04 | 20.64 | 15.76 |

The gap between "no protection" and "all protection" is the budget we're trying to recover selectively.
