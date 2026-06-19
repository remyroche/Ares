#!/usr/bin/env python3
"""Launch the current Kraken perps shadow inference run as a detached process."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path("/Users/remyroche/Documents/Ares")
RUN_ID = "20260617_090000_no_mkt4_labelhpo_final_fit"
LOG_PATH = ROOT / "logs" / "shadow_inference_20260617_090000_no_mkt4_labelhpo_final_fit_restarted_20260618.log"
FALLBACK_LOG = "logs/tprint_fallback_shadow_20260618.log"


def main() -> None:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "extreme_price_movements.inference.run_inference",
        "--shadow",
        "--perps",
        "--data-root",
        "data_perp",
        "--run-id",
        RUN_ID,
        "--model-artifact-run-id",
        RUN_ID,
        "--policy-artifact-run-id",
        RUN_ID,
        "--lookback-hours",
        "1440",
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = env.get("PYTHONPATH") or "."
    env["EPM_TPRINT_FALLBACK_LOG"] = FALLBACK_LOG

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    pid = os.fork()
    if pid:
        print(pid)
        return

    os.setsid()
    os.chdir(ROOT)
    stdin_fd = os.open(os.devnull, os.O_RDONLY)
    log_fd = os.open(LOG_PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.dup2(stdin_fd, 0)
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
    finally:
        for fd in {stdin_fd, log_fd}:
            try:
                os.close(fd)
            except OSError:
                pass
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
