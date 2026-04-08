#!/bin/zsh
set -euo pipefail

POLL_SECS="${1:-60}"
READY_STREAK=0

while true; do
  if env PYTHONPATH=. python3 - <<'PY'
from extreme_price_movements.config import CFG
from extreme_price_movements.run_pipeline import (
    _configure_report_roots,
    _label_artifacts_ready,
    _normalize_cfg_paths,
    _resolve_ts_sig,
)
cfg = dict(CFG)
_normalize_cfg_paths(cfg)
_configure_report_roots(cfg)
ts_sig = _resolve_ts_sig(cfg, None)
raise SystemExit(0 if (ts_sig is not None and _label_artifacts_ready(cfg, ts_sig)) else 2)
PY
  then
    READY_STREAK=$((READY_STREAK + 1))
    echo "[watcher] labels ready poll streak=${READY_STREAK}"
    if [ "$READY_STREAK" -ge 2 ]; then
      echo "[watcher] labels ready; running verification gate"
      if env PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/verify_labels_ready.py; then
        echo "[watcher] verification passed; starting train_base"
        exec env PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u extreme_price_movements/run_pipeline.py train_base
      fi
      echo "[watcher] verification failed; refusing to launch train_base"
      exit 1
    fi
  else
    READY_STREAK=0
    echo "[watcher] labels not ready yet"
  fi
  sleep "$POLL_SECS"
done
