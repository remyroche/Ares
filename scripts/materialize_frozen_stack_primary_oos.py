#!/usr/bin/env python3
"""Materialize older strict-OOS predictions for the current frozen stack.

The current ATR2/q4h residual artifact starts in July 2024.  This runner
reuses the exact frozen specialist contract and winning residual parameters on
the three pre-transport primary folds, so those rows can be used as prior
conversion-training data without mixing the incompatible legacy TP6/SL4 stack.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_frozen_residual_query_hpo import _fold_scores, _load
from scripts.run_market_spine_covariance_meta import LONG_HISTORY_FOLDS

OUT = ROOT / "data_perp/artifacts/frozen_specialist_primary_oos_20260810_v1"
PARAMS = {
    "n_estimators": 220, "learning_rate": .03, "max_depth": 5, "num_leaves": 52,
    "min_child_samples": 893, "min_sum_hessian_in_leaf": 1.1298052513600887,
    "min_gain_to_split": .0089300561896448, "colsample_bytree": .7882182037573211,
    "subsample": .8666554346312396, "subsample_freq": 1, "reg_alpha": .030925476912139326,
    "reg_lambda": .16986488135579808, "max_bin": 63,
    "label_gain": [0., .25, 1., 3., 7., 12.], "verbosity": -1,
    "random_state": 20260810, "n_jobs": 1,
}


def run(out: Path = OUT) -> Path:
    if out.exists() and (out / "predictions.parquet").exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    base, views, ae, ctx = _load()
    folds = LONG_HISTORY_FOLDS[:3]
    pieces = []
    for fold in folds:
        piece = _fold_scores(base, views, ae, ctx, fold, "q4h_side", "q4h_side", PARAMS)
        piece["source_contract"] = "frozen_atr2_specialist_q4h_residual_v1"
        pieces.append(piece)
    pred = pd.concat(pieces, ignore_index=True).sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if pred.candidate_id.duplicated().any():
        raise ValueError("primary strict-OOS candidate IDs are duplicated")
    pred.to_parquet(out / "predictions.parquet", index=False, compression="zstd")
    fold_rows = [{"name": f.name, "test_start": f.test_start, "test_end": f.test_end, "rows": int(len(p))} for f, p in zip(folds, pieces)]
    manifest = {
        "schema": "frozen_specialist_primary_oos_v1",
        "status": "COMPLETED_STRICT_OOS_HISTORICAL_EXTENSION",
        "contract": "same frozen seven-view ATR2 specialists, q4h×side query, ordinal residual LambdaRank parameters as transport winner",
        "folds": fold_rows,
        "rows": int(len(pred)),
        "timestamp_min": pred["__ts__"].min().isoformat(),
        "timestamp_max": pred["__ts__"].max().isoformat(),
        "selection": "no new HPO; reused frozen transport winner",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


if __name__ == "__main__":
    print(run())
