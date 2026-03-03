#!/bin/bash
set -euo pipefail
export PYTHONPATH=/Users/remyroche/Documents/Ares:$PYTHONPATH
LOGDIR=/Users/remyroche/Documents/Ares/reports

log() { echo "[$(date)] $*"; }

log "=== Ares ML Pipeline Runner ==="

# ── Step 1: TBM Optimization ────────────────────────────────────────────────
log "Checking for existing TBM Optimization..."
EXISTING_TBM_PID=$(pgrep -f compare_tbm_parameters.py || true)

if [ -n "$EXISTING_TBM_PID" ]; then
    log "Detected existing TBM process (PID: $EXISTING_TBM_PID). Waiting..."
    while kill -0 "$EXISTING_TBM_PID" 2>/dev/null; do sleep 30; done
    log "Existing TBM optimization finished."
else
    log "No existing TBM process. Starting TBM optimization..."
    python3 extreme_price_movements/offline_optimisers/compare_tbm_parameters.py \
        --data-root data --output reports/tbm_comparison.csv \
        > "$LOGDIR/tbm_runner.log" 2>&1
    log "TBM optimization finished."
fi

# ── Step 2: Label Generation ─────────────────────────────────────────────────
log "Starting Label Generation step..."
python3 extreme_price_movements/run_pipeline.py labels \
    --horizons 1 2 4 \
    > "$LOGDIR/pipeline_labels.log" 2>&1
log "Label Generation complete."

# ── Step 3: Base Training ─────────────────────────────────────────────────────
log "Starting Base Training step..."
python3 extreme_price_movements/run_pipeline.py train --base-only \
    > "$LOGDIR/pipeline_base.log" 2>&1
log "Base Training complete."

# ── Step 4: Meta Training ─────────────────────────────────────────────────────
log "Starting Meta Training step..."
python3 extreme_price_movements/run_pipeline.py train_meta \
    > "$LOGDIR/pipeline_meta.log" 2>&1
log "Meta Training complete."

# ── Step 5: Ridge / EV Decomposition Sizer ───────────────────────────────────
log "Starting Sizer step..."
python3 extreme_price_movements/run_pipeline.py sizer \
    > "$LOGDIR/pipeline_sizer.log" 2>&1
log "Sizer step complete."

log "=== All pipeline steps completed successfully! ==="
