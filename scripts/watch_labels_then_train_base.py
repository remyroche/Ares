#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.config import CFG
from extreme_price_movements.run_pipeline import (
    _configure_report_roots,
    _label_artifacts_ready,
    _normalize_cfg_paths,
    _resolve_ts_sig,
)


def _pid_alive(pid: int) -> bool:
    proc = subprocess.run(
        ["ps", "-p", str(pid), "-o", "pid="],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0 and bool(proc.stdout.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--poll-secs", type=int, default=60)
    args = ap.parse_args()

    while _pid_alive(int(args.pid)):
        print(f"[watcher] labels pid={args.pid} still running", flush=True)
        time.sleep(max(5, int(args.poll_secs)))

    cfg = dict(CFG)
    _normalize_cfg_paths(cfg)
    _configure_report_roots(cfg)
    ts_sig = _resolve_ts_sig(cfg, None)
    if ts_sig is None:
        print("[watcher] no ts_sig found; skipping train_base", flush=True)
        return 1

    if not _label_artifacts_ready(cfg, ts_sig):
        print("[watcher] labels not ready after pid exit; skipping train_base", flush=True)
        return 2

    print("[watcher] labels ready; starting train_base", flush=True)
    cmd = [
        sys.executable,
        "-u",
        "extreme_price_movements/run_pipeline.py",
        "train_base",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), env={**os.environ, "PYTHONPATH": "."})
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
