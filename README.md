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
├── run_experiments.py             # Original grid sweep runner (Part 1)
├── visualize.py                   # Original visualization (Part 1)
├── gptq_cdf_patch.py             # GPTQ integration patch (Part 1)
│
├── results/                      # Output: profile JSONs + experiment results
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

## Prior Results (Part 1)

For reference, existing results from the grid type / outlier sweep experiments:

| Model | FP16 | Uniform 4-bit | Hybrid γ=0.15 | Hybrid + OP 1% |
|-------|------|---------------|---------------|----------------|
| opt-125m | 27.66 | 40.78 | 47.00 | 30.67 |
| opt-350m | 22.00 | 30.10 | 38.31 | 28.10 |
| opt-1.3b | 14.62 | 50.04 | 20.64 | 15.76 |

The gap between "no protection" and "all protection" is the budget we're trying to recover selectively.
