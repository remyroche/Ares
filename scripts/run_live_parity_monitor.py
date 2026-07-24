#!/usr/bin/env python3
"""Run the canonical three-hour live parity monitor under job control."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-pid", type=int)
    parser.add_argument("--inference-log")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_live_parity_monitor_loop.sh"
    env = os.environ.copy()
    if args.inference_pid is not None:
        env["EPM_INFERENCE_PID"] = str(args.inference_pid)
    if args.inference_log:
        env["EPM_INFERENCE_LOG"] = args.inference_log
    raise SystemExit(subprocess.call(["bash", str(script)], cwd=root, env=env))


if __name__ == "__main__":
    main()
