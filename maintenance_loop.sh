#!/bin/bash

MAINT_LOG="reports/maintenance.log"

while true; do
    echo "[$(date)] --- Maintenance Check Starting ---" >> "$MAINT_LOG"
    
    # 1. Check Processes
    TBM_PID=$(pgrep -f compare_tbm_parameters.py)
    RUNNER_PID=$(pgrep -f run_remaining_pipeline.sh)
    SYNC_PID=$(pgrep -f git_sync_loop.sh)
    
    if [ -z "$TBM_PID" ] && [ -z "$RUNNER_PID" ]; then
        echo "[$(date)] ALERT: Both TBM and Runner processes are missing!" >> "$MAINT_LOG"
    fi
    
    # 2. Check Logs for Errors
    for LOG in reports/*.log; do
        if [ -f "$LOG" ]; then
            # Look for Tracebacks in the last 100 lines
            ERROR_COUNT=$(tail -n 100 "$LOG" | grep -Ei "traceback|error|exception|fail" | grep -v "FutureWarning" | wc -l)
            if [ "$ERROR_COUNT" -gt 0 ]; then
                echo "[$(date)] ALERT: Found $ERROR_COUNT potential errors in $LOG" >> "$MAINT_LOG"
                tail -n 20 "$LOG" >> "$MAINT_LOG"
            fi
        fi
    done
    
    # 3. Process Status Summary
    echo "[$(date)] Status - TBM: ${TBM_PID:-DEAD}, Runner: ${RUNNER_PID:-DEAD}, Sync: ${SYNC_PID:-DEAD}" >> "$MAINT_LOG"
    echo "[$(date)] --- Maintenance Check Completed ---" >> "$MAINT_LOG"
    
    sleep 3600
done
