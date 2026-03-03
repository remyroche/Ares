#!/bin/bash
export PYTHONPATH=/Users/remyroche/Documents/Ares:$PYTHONPATH

echo "Checking for existing TBM Optimization..."
EXISTING_TBM_PID=$(pgrep -f compare_tbm_parameters.py)

if [ -n "$EXISTING_TBM_PID" ]; then
    echo "Detected existing TBM optimization process (PID: $EXISTING_TBM_PID). Waiting for it to finish..."
    while ps -p $EXISTING_TBM_PID > /dev/null; do
        sleep 30
    done
    echo "Existing TBM optimization finished."
else
    echo "No existing TBM optimization process found. Starting a new one..."
    python3 extreme_price_movements/offline_optimisers/compare_tbm_parameters.py --data-root data --output reports/tbm_comparison.csv > reports/tbm_runner.log 2>&1
    if [ $? -ne 0 ]; then
        echo "TBM Optimization failed!"
        exit 1
    fi
echo "TBM optimization finished."
fi

echo "Starting Feature Generation step..."
python3 extreme_price_movements/run_pipeline.py features > reports/pipeline_features.log 2>&1
if [ $? -ne 0 ]; then
    echo "Feature Generation failed!"
    exit 1
fi

echo "Starting Label step..."
python3 extreme_price_movements/run_pipeline.py labels > reports/pipeline_labels.log 2>&1
if [ $? -ne 0 ]; then
    echo "Label step failed! Check reports/pipeline_labels.log"
    exit 1
fi
echo "Label step completed."

echo "Starting Base Training step..."
python3 extreme_price_movements/run_pipeline.py train --base-only > reports/pipeline_base.log 2>&1
if [ $? -ne 0 ]; then
    echo "Base Training step failed! Check reports/pipeline_base.log"
    exit 1
fi
echo "Base Training step completed."

echo "Starting Meta Training step..."
python3 extreme_price_movements/run_pipeline.py train_meta > reports/pipeline_meta.log 2>&1
if [ $? -ne 0 ]; then
    echo "Meta Training step failed! Check reports/pipeline_meta.log"
    exit 1
fi
echo "Meta Training step completed."

echo "Starting Ridge Sizer step..."
python3 extreme_price_movements/run_pipeline.py ridge_sizer > reports/pipeline_sizer.log 2>&1
if [ $? -ne 0 ]; then
    echo "Ridge Sizer step failed! Check reports/pipeline_sizer.log"
    exit 1
fi
echo "Ridge Sizer step completed."

echo "All pipeline steps completed successfully!"
