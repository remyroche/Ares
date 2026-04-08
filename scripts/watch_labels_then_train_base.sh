#!/bin/zsh
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <labels_pid> [poll_secs]" >&2
  exit 2
fi

PID="$1"
POLL_SECS="${2:-60}"

while ps -p "$PID" -o pid= >/dev/null 2>&1; do
  echo "[watcher] labels pid $PID still running"
  sleep "$POLL_SECS"
done

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
  echo "[watcher] labels ready; starting train_base"
  exec env PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u extreme_price_movements/run_pipeline.py train_base
else
  echo "[watcher] labels not ready after pid exit; skipping train_base"
  exit 1
fi
