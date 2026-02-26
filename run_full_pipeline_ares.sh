#!/bin/bash
# run_full_pipeline_ares.sh
# Automates the sequential execution of the Ares ML pipeline after TBM optimization.

set -e # Exit immediately if a command exits with a non-zero status.

LOG_DIR="logs/pipeline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "🚀 Starting Full Pipeline Sequence: $(date)" | tee -a "$LOG_DIR/execution.log"

# 1. Label Generation
echo "--- [1/5] LABEL GENERATION ---" | tee -a "$LOG_DIR/execution.log"
python3 extreme_price_movements/run_pipeline.py labels 2>&1 | tee "$LOG_DIR/labels.log"

# 2. Base Model Training
echo "--- [2/5] BASE MODEL TRAINING ---" | tee -a "$LOG_DIR/execution.log"
python3 extreme_price_movements/run_pipeline.py train 2>&1 | tee "$LOG_DIR/train_base.log"

# 3. Meta Model Training
echo "--- [3/5] META MODEL TRAINING ---" | tee -a "$LOG_DIR/execution.log"
export MKL_NUM_THREADS=2
export OMP_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
python3 extreme_price_movements/run_pipeline.py train_meta 2>&1 | tee "$LOG_DIR/train_meta.log"

# 4. Ridge Position Sizer
echo "--- [4/5] RIDGE POSITION SIZER ---" | tee -a "$LOG_DIR/execution.log"
export MKL_NUM_THREADS=2
export OMP_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
python3 extreme_price_movements/run_pipeline.py ridge_sizer 2>&1 | tee "$LOG_DIR/ridge_sizer.log"

# 5. Final Optimise Step
echo "--- [5/5] OPTIMISE STEP ---" | tee -a "$LOG_DIR/execution.log"
python3 extreme_price_movements/run_pipeline.py optimise 2>&1 | tee "$LOG_DIR/optimise.log"

echo "✅ Pipeline Sequence Complete: $(date)" | tee -a "$LOG_DIR/execution.log"
echo "Logs saved to: $LOG_DIR" | tee -a "$LOG_DIR/execution.log"
