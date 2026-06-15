#!/usr/bin/env python3
"""Daemonize the top-two recency pipeline launcher for long monitored runs."""
from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
_LOG_PATH_RAW = os.environ.get("EPM_TOP2_DAEMON_LOG_PATH", "").strip()
if _LOG_PATH_RAW:
    LOG_PATH = Path(_LOG_PATH_RAW)
    if not LOG_PATH.is_absolute():
        LOG_PATH = ROOT / LOG_PATH
else:
    _RUN_ID = os.environ.get("EPM_TOP2_RESELECT_RUN_ID", "20260611").strip() or "20260611"
    LOG_PATH = ROOT / "logs" / f"top2_recency_pipeline_nohup_{_RUN_ID}.log"


def _fork() -> int:
    try:
        return os.fork()
    except OSError as exc:
        raise SystemExit(f"fork failed: {exc}") from exc


def main() -> int:
    first_pid = _fork()
    if first_pid:
        print(first_pid, flush=True)
        return 0

    os.setsid()
    second_pid = _fork()
    if second_pid:
        os._exit(0)

    os.chdir(ROOT)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTHONPATH", ".")
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib_codex")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("ab", buffering=0) as log_fp:
        os.dup2(log_fp.fileno(), sys.stdout.fileno())
        os.dup2(log_fp.fileno(), sys.stderr.fileno())
        os.execv(
            sys.executable,
            [
                sys.executable,
                "-u",
                "scripts/run_top2_recency_pipeline_20260611.py",
            ],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
