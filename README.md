# COMP0252 CW2 — Outlier Protection and Distribution-Aware Quantization for LLMs

Investigates CDF-based non-uniform quantization grids and static outlier protection for weight-only 4-bit quantization of OPT language models (125M to 6.7B). Compares uniform, CDF, and hybrid grids using RTN and GPTQ, evaluates perplexity on WikiText-2 and zero-shot accuracy on LAMBADA, ARC, and PiQA.

## Repository Structure

```
COMP0252_CW2/
├── cdf_grid.py                 # Core quantization grid library (uniform, CDF, hybrid construction + quantize/dequantize)
├── rtn_baseline.py             # RTN quantization: loads model, quantizes Linear layers in-place, evaluates WikiText-2 perplexity
├── gptq_quantize.py            # GPTQ quantization with optional CDF/hybrid grids (uses vendored gptq/ repo)
├── gptq_cdf_patch.py           # Patches GPTQ's internal quantize() call to use CDF/hybrid grids
├── zeroshot_eval.py            # Zero-shot evaluation on LAMBADA, ARC-Easy, ARC-Challenge, PiQA
├── gamma_sweep.py              # Sweeps hybrid mixing coefficient gamma across model scales
├── layer_sensitivity.py        # Evaluates layer-wise sensitivity with different scoring heuristics
├── pack_quantized.py           # Packs quantized weights into actual INT4 format for storage compression
├── run_experiments.py           # Batch runner for grid type / outlier sweep experiments
│
├── src/                        # Selective layer protection pipeline
│   ├── layer_profiler.py       # Computes per-layer weight statistics (kurtosis, outlier fraction, range/sigma, variance)
│   ├── selective_quantize.py   # Applies outlier protection only to top-k most sensitive layers
│   ├── run_all.py              # Orchestrates full sweep: profile -> sweep top-k x scoring -> evaluate
│   ├── visualize_selective.py  # Generates heatmaps, top-k curves, Pareto plots from results
│   └── packed_runtime.py       # Inference benchmarking with packed INT4 checkpoints (eager vs lazy modes)
│
├── Plotting scripts
│   ├── plot_bucket_diagnostics.py   # Grid-level spacing and weight distribution visualizations
│   ├── plot_gamma_sweep.py          # Perplexity vs gamma curves
│   ├── plot_cdf_op_comparison.py    # Effect of outlier protection on CDF quantization
│   ├── plot_perplexity_barchart.py  # Perplexity comparison bar charts
│   └── plot_zeroshot_barchart.py    # Zero-shot accuracy bar charts
│
├── gptq/                       # Vendored GPTQ repository (official implementation)
│   ├── opt.py                  # OPT-specific GPTQ entry point
│   ├── gptq.py                 # Core GPTQ algorithm (Hessian + compensation loop)
│   ├── datautils.py            # Calibration data loading
│   └── zeroShot/               # Zero-shot evaluation harness (LAMBADA, ARC, PiQA, etc.)
│
├── scripts/
│   └── run.sh                  # One-command pipeline: profile + sweep + visualize
│
├── tests/
│   └── test_packed_runtime.py  # Tests for INT4 pack/unpack round-trip correctness
│
├── COMP0252-Project/           # LaTeX report source
│   ├── main.tex
│   └── refs.bib
│
├── results/                    # Layer profiling and selective quantization results (JSON)
├── results_quantization_methods/ # Grid comparison and gamma sweep results (JSON)
├── results_zeroshot/           # Zero-shot benchmark results (JSON)
├── figures/                    # Generated plots used in the report
├── packed_models/              # Packed INT4 model checkpoints
├── saved_models/               # Saved quantized model checkpoints (for reuse)
└── requirements.txt
```

## Setup

Requires Python 3.9+ and a CUDA GPU.

```bash
python3 -m venv VIRTUAL_ENV
source VIRTUAL_ENV/bin/activate
pip install -r requirements.txt
```

## Running Experiments

### RTN Quantization (uniform, CDF, hybrid grids)

```bash
# FP16 baseline (no quantization)
python rtn_baseline.py --model facebook/opt-125m --fp16_only

# Uniform RTN, 4-bit
python rtn_baseline.py --model facebook/opt-125m --bits 4 --grid_type uniform

# CDF RTN with 1% outlier protection
python rtn_baseline.py --model facebook/opt-125m --bits 4 --grid_type cdf \
    --protect_outliers --outlier_percentile 1.0

# Hybrid RTN with gamma=0.5 and 1% outlier protection (decoder-only)
python rtn_baseline.py --model facebook/opt-125m --bits 4 --grid_type hybrid --gamma 0.5 \
    --protect_outliers --outlier_percentile 1.0 --decoder_only
```

### GPTQ Quantization

```bash
# Standard GPTQ with uniform grid
python gptq_quantize.py --model facebook/opt-125m --wbits 4

# GPTQ with hybrid grid
python gptq_quantize.py --model facebook/opt-125m --wbits 4 --grid_type hybrid --gamma 0.5

# Save a GPTQ checkpoint for later reuse
python gptq_quantize.py --model facebook/opt-125m --wbits 4 \
    --save_quantized_dir saved_models/opt-125m-gptq-4bit
```

### Gamma Sweep

```bash
python gamma_sweep.py --model facebook/opt-125m
```

Sweeps gamma in {0.0, 0.1, ..., 1.0} with 1% outlier protection. Results saved to `results_quantization_methods/gamma_sweep.json`.

### Zero-Shot Evaluation

```bash
# FP16 baseline
python zeroshot_eval.py --model facebook/opt-125m --method fp16 \
    --tasks lambada,arc_easy,arc_challenge,piqa

# Hybrid RTN
python zeroshot_eval.py --model facebook/opt-125m --method hybrid_rtn \
    --bits 4 --gamma 0.5 --outlier_percentile 1.0 \
    --tasks lambada,arc_easy,arc_challenge,piqa

# GPTQ from saved checkpoint
python zeroshot_eval.py --model facebook/opt-125m --method gptq \
    --load_quantized_dir saved_models/opt-125m-gptq-4bit \
    --tasks lambada,arc_easy,arc_challenge,piqa
```

### Layer Sensitivity Analysis

```bash
# Profile a model's layer statistics
python -m src.layer_profiler --model facebook/opt-125m

# Selective quantization with a specific scoring metric
python -m src.selective_quantize --model facebook/opt-125m \
    --profile results/opt-125m_profile.json \
    --bits 4 --grid_type hybrid --gamma 0.15 --topk 12 --scoring kurtosis

# Full pipeline: profile + sweep all scoring methods + evaluate
python -m src.run_all --models facebook/opt-125m

# Quick test (only 25%, 50%, 75%, 100% budgets)
python -m src.run_all --models facebook/opt-125m --scoring kurtosis --quick

# Full pipeline via shell script (also generates figures)
bash scripts/run.sh
```

### Storage Compression (INT4 Packing)

```bash
python pack_quantized.py --model facebook/opt-125m --bits 4 \
    --grid_type uniform --outlier_percentile 1.0
```

Packs quantized weights into actual 4-bit integers with outlier weights stored separately at FP16. Verifies round-trip correctness.

### Inference Benchmarking

```bash
python -m src.packed_runtime --model facebook/opt-125m \
    --checkpoint packed_models/opt-125m-gptq-4bit
```

Benchmarks decode throughput and GPU memory for eager-unpacked vs lazy-layerwise inference modes.

### Generating Figures

```bash
# Selective quantization figures (heatmaps, top-k curves, Pareto plots)
python -m src.visualize_selective --results results/selective_results.json --model facebook/opt-125m

# Grid comparison and bar charts
python visualize.py --results results.json
python visualize.py --grid_demo --bits 4

# Gamma sweep plot
python plot_gamma_sweep.py

# Bucket spacing diagnostics
python plot_bucket_diagnostics.py

# CDF outlier protection comparison
python plot_cdf_op_comparison.py

# Zero-shot bar charts
python plot_zeroshot_barchart.py
```

## Key Results

- Uniform RTN + 1% outlier protection matches or beats GPTQ at all OPT scales while being 96-155x faster to quantize.
- Outlier protection is the dominant factor: without it, CDF quantization degrades catastrophically (580 PPL at 1.3B).
- Hybrid CDF/uniform grids improve over pure CDF codebooks at all scales, but the standalone affine uniform method remains competitive.
- Static layer sensitivity heuristics (kurtosis, variance, etc.) fail to consistently outperform random layer selection, suggesting static weight statistics are unreliable proxies for runtime sensitivity.

## Environment

All experiments run on a single NVIDIA 3090 Ti GPU. Models loaded in FP16 via HuggingFace Transformers.
