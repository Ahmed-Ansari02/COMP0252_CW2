#!/bin/bash
# Selective Layer-Level Outlier Protection — Run
# Runs the full experiment pipeline: profile → sweep → visualize.
#
# Usage:
#   bash scripts/run.sh                                    # full pipeline on opt-125m
#   bash scripts/run.sh --models facebook/opt-125m facebook/opt-350m
#   bash scripts/run.sh --skip_profile                     # reuse existing profiles
#   bash scripts/run.sh --scoring kurtosis                 # single scoring method

set -e
cd "$(dirname "$0")/.."

# ---- Create output directories ----
mkdir -p results figures

# ---- Run the main pipeline ----
echo ""
echo "============================================================"
echo "  Selective Layer-Level Outlier Protection Experiments"
echo "============================================================"
echo ""

python -m src.run_all "$@"

# ---- Generate visualizations ----
echo ""
echo "=== Generating figures ==="
python -m src.visualize_selective

echo ""
echo "Done. Results in results/, figures in figures/"
