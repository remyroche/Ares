#!/usr/bin/env python3
"""Launch the no-mkt4 HPO matrix wrapper as a detached double-fork daemon."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path("/Users/remyroche/Documents/Ares")
LOG_DIR = ROOT / "logs"
PID_PATH = LOG_DIR / "no_mkt4_fullhpo_labelhpo_drift_daemon.pid"
OUT_PATH = LOG_DIR / "no_mkt4_fullhpo_labelhpo_drift_daemon.out.log"
ERR_PATH = LOG_DIR / "no_mkt4_fullhpo_labelhpo_drift_daemon.err.log"


def _fork_exit_parent() -> None:
    pid = os.fork()
    if pid > 0:
        raise SystemExit(0)


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    Path("/private/tmp/mplconfig").mkdir(parents=True, exist_ok=True)

    _fork_exit_parent()
    os.setsid()
    _fork_exit_parent()

    os.chdir(str(ROOT))
    os.umask(0o022)
    with open(os.devnull, "rb", buffering=0) as stdin:
        os.dup2(stdin.fileno(), 0)
    with OUT_PATH.open("ab", buffering=0) as stdout:
        os.dup2(stdout.fileno(), 1)
    with ERR_PATH.open("ab", buffering=0) as stderr:
        os.dup2(stderr.fileno(), 2)

    PID_PATH.write_text(f"{os.getpid()}\n", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": ".",
            "MPLCONFIGDIR": "/private/tmp/mplconfig",
            "MPLBACKEND": "Agg",
        }
    )
    os.execve(
        sys.executable,
        [
            sys.executable,
            "-u",
            "scripts/run_no_mkt4_fullhpo_labelhpo_drift_20260614.py",
        ],
        env,
    )


if __name__ == "__main__":
    main()
