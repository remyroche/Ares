#!/usr/bin/env python3
"""Run low-performance specialist train commands sequentially.

This intentionally uses subprocess env dictionaries instead of the generated
shell command lines. The env blocks are long, and direct subprocess execution is
less fragile than shell assignment syntax for long strategy IDs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _stage_command(pack: dict[str, Any], stage: str) -> list[str]:
    source_runs = pack.get("source_runs") or {}
    feature_run_id = str(source_runs.get("feature_source_run_id") or pack.get("env", {}).get("EPM_FEATURE_SOURCE_RUN_ID") or "")
    if not feature_run_id:
        raise RuntimeError(f"{pack.get('head')} missing feature_source_run_id")
    return [
        sys.executable,
        "-u",
        "extreme_price_movements/run_pipeline.py",
        stage,
        "--market-mode",
        "perps",
        "--model-backend",
        "lgbm_pipeline",
        "--run-id",
        str(pack["run_id"]),
        "--ts",
        feature_run_id,
    ]


def run_commands(plan_dir: Path, *, start_at: str | None = None, stop_after_failure: bool = True) -> int:
    commands_path = plan_dir / "train_commands.json"
    packs = json.loads(commands_path.read_text(encoding="utf-8"))
    status_path = plan_dir / "sequential_runner_status.jsonl"
    start_seen = start_at is None
    rc_final = 0
    for pack in packs:
        head = str(pack.get("head", ""))
        for stage in ("train_base", "train_meta"):
            key = f"{head}:{stage}"
            if not start_seen:
                start_seen = key == str(start_at)
                if not start_seen:
                    continue
            log_path = Path(str(pack.get(f"{stage}_log") or f"logs/{pack['run_id']}_{head}_{stage}.log"))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update({str(k): str(v) for k, v in (pack.get("env") or {}).items()})
            cmd = _stage_command(pack, stage)
            start = time.time()
            _append_jsonl(
                status_path,
                {
                    "event": "start",
                    "head": head,
                    "stage": stage,
                    "run_id": pack.get("run_id"),
                    "log": str(log_path),
                    "cmd": cmd,
                    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
            with log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.run(cmd, cwd=".", env=env, stdout=log, stderr=subprocess.STDOUT)
            elapsed = time.time() - start
            _append_jsonl(
                status_path,
                {
                    "event": "end",
                    "head": head,
                    "stage": stage,
                    "run_id": pack.get("run_id"),
                    "returncode": int(proc.returncode),
                    "elapsed_seconds": elapsed,
                    "log": str(log_path),
                    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
            if int(proc.returncode) != 0:
                rc_final = int(proc.returncode)
                if stop_after_failure:
                    return rc_final
    return rc_final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", required=True)
    parser.add_argument("--start-at", default=None, help="Optional key like long_dist:train_base")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()
    raise SystemExit(
        run_commands(
            Path(args.plan_dir),
            start_at=args.start_at,
            stop_after_failure=not bool(args.keep_going),
        )
    )


if __name__ == "__main__":
    main()
