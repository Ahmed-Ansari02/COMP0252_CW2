#!/bin/bash
# Setup and run script for CDF quantization experiments
# Priority order from the plan:
#   1. Uniform RTN baseline on OPT-125M
#   2. CDF grid RTN on OPT-125M
#   3. Hybrid grid RTN on OPT-125M
#   4. Grid visualisation
#   5-6. Repeat on OPT-350M, OPT-1.3B
#   7. GPTQ integration (stretch)

set -e
cd "$(dirname "$0")"

# ---- Install dependencies ----
pip install torch transformers datasets accelerate matplotlib --quiet

# ---- Priority 1: Uniform RTN baseline on OPT-125M ----
echo "=== Priority 1: Uniform RTN baseline OPT-125M 4-bit ==="
python rtn_baseline.py --model facebook/opt-125m --bits 4 --grid_type uniform

# ---- Priority 1b: FP16 baseline ----
echo "=== FP16 baseline OPT-125M ==="
python rtn_baseline.py --model facebook/opt-125m --fp16_only

# ---- Priority 2: CDF grid RTN on OPT-125M ----
echo "=== Priority 2: CDF grid RTN OPT-125M 4-bit ==="
python rtn_baseline.py --model facebook/opt-125m --bits 4 --grid_type cdf

# ---- Priority 3: Hybrid grid RTN on OPT-125M ----
echo "=== Priority 3: Hybrid grid RTN OPT-125M 4-bit gamma=0.15 ==="
python rtn_baseline.py --model facebook/opt-125m --bits 4 --grid_type hybrid --gamma 0.15

# ---- Outlier protection variants on OPT-125M ----
echo "=== Outlier protection: CDF + protect outliers OPT-125M 4-bit ==="
python rtn_baseline.py --model facebook/opt-125m --bits 4 --grid_type cdf \
    --protect_outliers --outlier_percentile 1.0
python rtn_baseline.py --model facebook/opt-125m --bits 4 --grid_type cdf \
    --protect_outliers --outlier_percentile 0.5
python rtn_baseline.py --model facebook/opt-125m --bits 4 --grid_type hybrid --gamma 0.15 \
    --protect_outliers --outlier_percentile 1.0

# ---- Priority 4: Grid visualisation ----
echo "=== Priority 4: Grid comparison plot ==="
python visualize.py --grid_demo --bits 4
python visualize.py --model facebook/opt-125m \
    --layer_name model.decoder.layers.0.self_attn.q_proj --bits 4

# ---- Priority 5: OPT-350M ----
echo "=== Priority 5: OPT-350M experiments ==="
python rtn_baseline.py --model facebook/opt-350m --fp16_only
python rtn_baseline.py --model facebook/opt-350m --bits 4 --grid_type uniform
python rtn_baseline.py --model facebook/opt-350m --bits 4 --grid_type cdf
python rtn_baseline.py --model facebook/opt-350m --bits 4 --grid_type hybrid --gamma 0.15
python rtn_baseline.py --model facebook/opt-350m --bits 4 --grid_type cdf \
    --protect_outliers --outlier_percentile 1.0
python rtn_baseline.py --model facebook/opt-350m --bits 4 --grid_type hybrid --gamma 0.15 \
    --protect_outliers --outlier_percentile 1.0

# ---- Priority 6: OPT-1.3B ----
echo "=== Priority 6: OPT-1.3B experiments ==="
python rtn_baseline.py --model facebook/opt-1.3b --fp16_only
python rtn_baseline.py --model facebook/opt-1.3b --bits 4 --grid_type uniform
python rtn_baseline.py --model facebook/opt-1.3b --bits 4 --grid_type cdf
python rtn_baseline.py --model facebook/opt-1.3b --bits 4 --grid_type hybrid --gamma 0.15
python rtn_baseline.py --model facebook/opt-1.3b --bits 4 --grid_type cdf \
    --protect_outliers --outlier_percentile 1.0
python rtn_baseline.py --model facebook/opt-1.3b --bits 4 --grid_type hybrid --gamma 0.15 \
    --protect_outliers --outlier_percentile 1.0

# ---- Outlier protection sweep: hybrid + OP 1%-10% (OPT-125M) ----
echo "=== Outlier protection sweep OPT-125M ==="
python rtn_baseline.py --model facebook/opt-125m --bits 4 --grid_type hybrid --gamma 0.15 \
    --outlier_sweep

# ---- Hybrid gamma sweep (OPT-125M) ----
echo "=== Hybrid gamma sweep OPT-125M ==="
for gamma in 0.05 0.10 0.20 0.30; do
    python rtn_baseline.py --model facebook/opt-125m --bits 4 \
        --grid_type hybrid --gamma $gamma
done

# ---- Final results table ----
echo "=== Results summary ==="
python run_experiments.py --models facebook/opt-125m facebook/opt-350m facebook/opt-1.3b \
    --skip_fp16  # already computed, just print summary

echo "Done. Results in results.json"
