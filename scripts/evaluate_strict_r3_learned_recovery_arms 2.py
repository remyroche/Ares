#!/usr/bin/env python3
"""Evaluate chronological learned recovery overlays on matched strict-R3 rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_strict_r3_recovery_detection_arms import _summaries


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--funnel-root", type=Path, required=True)
    parser.add_argument("--learned-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)

    maps = pd.read_parquet(args.funnel_root / "multiwindow_ev_maps.parquet")
    maps["__decision_ts__"] = pd.to_datetime(maps["__decision_ts__"], utc=True)
    maps["snapshot_utc"] = maps["__decision_ts__"].dt.normalize()
    pred = pd.read_parquet(args.learned_root / "learned_recovery_daily_predictions.parquet")
    pred["snapshot_utc"] = pd.to_datetime(pred["snapshot_utc"], utc=True)
    # Only chronological 2025 OOF predictions and the once-frozen 2026 fit are
    # admissible.  No in-sample daily prediction is evaluated.
    pred = pred.loc[pred["fold"].isin(["f1", "f2", "f3", "f4", "confirmation_2026"])]
    pred = pred.drop_duplicates(["snapshot_utc", "model_family"], keep="last")
    wide = pred.pivot(index="snapshot_utc", columns="model_family", values="predicted_fast_advantage_bps")
    wide.columns = [f"pred_{value}" for value in wide.columns]
    frame = maps.merge(wide.reset_index(), on="snapshot_utc", how="inner", validate="many_to_one")
    fast = 0.5 * (frame["m14"] + frame["m21"])
    slow = frame[["m28", "m35", "m42"]].median(axis=1)
    arm_columns: dict[str, str] = {
        "A0_robust28_matched": "A0_robust28",
        "A2_robust21_matched": "A2_robust21",
        "A5_mean5_matched": "A5_mean5",
    }
    for family in ("linear_ridge", "gam_ridge", "lightgbm"):
        prediction = frame[f"pred_{family}"]
        lam = 1.0 / (1.0 + np.exp(-prediction.clip(-500, 500) / 50.0))
        frame[f"learned_{family}_continuous"] = (1.0 - lam) * slow + lam * fast
        frame[f"learned_{family}_bounded50"] = (1.0 - 0.5 * lam) * slow + 0.5 * lam * fast
        arm_columns[f"L_{family}_continuous"] = f"learned_{family}_continuous"
        arm_columns[f"L_{family}_bounded50"] = f"learned_{family}_bounded50"
    summary, monthly, daily = _summaries(frame, arm_columns)
    summary.to_csv(args.out_dir / "ablation_summary.csv", index=False)
    monthly.to_csv(args.out_dir / "monthly_summary.csv", index=False)
    daily.to_parquet(args.out_dir / "daily_summary.parquet", index=False)
    manifest = {
        "schema": "strict_r3_learned_recovery_economics_v1",
        "status": "complete",
        "selection": "2025 chronological OOF only; 2026 confirmation frozen before inspection",
        "resolved_set_rule": "upstream features use policy_label_available_ts <= snapshot; unresolved rows ignored",
        "models": ["linear_ridge", "gam_ridge", "lightgbm"],
        "authority": "sigmoid(predicted future-7d fast-minus-slow opportunity EV / 50bps), continuous and bounded-50%",
        "maps_sha256": _sha(args.funnel_root / "multiwindow_ev_maps.parquet"),
        "predictions_sha256": _sha(args.learned_root / "learned_recovery_daily_predictions.parquet"),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "learned_recovery_economics_complete", "rows": len(frame), "arms": len(arm_columns)}))


if __name__ == "__main__":
    main()
