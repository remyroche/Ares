#!/bin/bash

while true; do
    echo "[$(date)] Starting hourly git sync..."
    git add .
    # Check if there are changes to commit
    if git diff-index --quiet HEAD --; then
        echo "[$(date)] No changes to commit."
    else
        git commit -m "Auto-commit: hourly pipeline update [$(date)]"
        git push origin $(git rev-parse --abbrev-ref HEAD)
        echo "[$(date)] Hourly git sync completed."
    fi
    sleep 3600
done
