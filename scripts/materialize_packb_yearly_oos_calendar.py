#!/usr/bin/env python3
"""Run bounded monthly Pack-B OOS folds and finalise the 21-day simulation."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCORER = ROOT / "scripts/run_packb_yearly_side_local_oos.py"
FINALIZER = ROOT / "scripts/finalize_packb_yearly_side_local_oos.py"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--months-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-train-rows", type=int, default=250000)
    p.add_argument("--base-lookback-days", type=int, default=90)
    p.add_argument("--base-target", default="__first_touch_target_soft__")
    p.add_argument("--base-binary", action="store_true")
    p.add_argument("--feature-store", type=Path, default=None)
    p.add_argument("--inner-oof-days", type=int, default=30)
    p.add_argument("--inner-base-warmup-days", type=int, default=30)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    a.months_root.mkdir(parents=True, exist_ok=True)
    for month in pd.period_range("2025-08", "2026-07", freq="M"):
        start = pd.Timestamp(month.start_time, tz="UTC")
        end = min(start + pd.offsets.MonthBegin(1), pd.Timestamp("2026-07-11", tz="UTC"))
        destination = a.months_root / str(month)
        raw = destination / "raw_oos_predictions.parquet"
        if raw.exists():
            print(f"reusing completed {month}", flush=True)
            continue
        if destination.exists():
            # A prior interruption can leave only the destination directory:
            # remove that empty checkpoint and rerun the month.  Refuse to
            # discard any non-empty partial artifact because its provenance is
            # ambiguous.
            if any(destination.iterdir()):
                raise RuntimeError(f"incomplete non-empty monthly checkpoint: {destination}")
            destination.rmdir()
        command = [
            sys.executable, str(SCORER), "--scored-start", start.strftime("%Y-%m-%d"),
            "--end", end.strftime("%Y-%m-%d"), "--max-train-rows", str(a.max_train_rows),
            "--base-lookback-days", str(a.base_lookback_days), "--inner-oof-days", str(a.inner_oof_days),
            "--inner-base-warmup-days", str(a.inner_base_warmup_days),
            "--base-target", a.base_target,
            "--skip-admission", "--out", str(destination),
        ]
        if a.base_binary:
            command.append("--base-binary")
        if a.feature_store is not None:
            command.extend(["--feature-store", str(a.feature_store)])
        subprocess.run(command, cwd=ROOT, check=True)
    subprocess.run([
        sys.executable, str(FINALIZER), "--monthly-root", str(a.months_root), "--out", str(a.out),
    ], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
