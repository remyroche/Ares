#!/usr/bin/env python3
"""Remove local overlay normalization from a fitted residual representation arm."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.meta_residual_archetypes import (  # noqa: E402
    OUTCOME_COLUMNS,
    REFERENCE_DERIVED_COLUMNS,
)
from scripts.run_train_meta_residual_archetype_enhancement import (  # noqa: E402
    DEFAULT_OUT_DIR,
    _calibrate,
    _fit_platt,
    _merge_residual_features,
    _selection_mask,
    metrics_by_scope,
    surprise_calendar,
)

DEFAULT_PARENT = "lifecycle_residual_aware_ae_gmm_overlay_pca8_clip8_baseline"
DEFAULT_CACHE = "residual_walkforward_ae_gmm_eval_mar_jun_pca8_clip8_baseline.parquet"


def _safe(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(
        columns=[
            name
            for name in OUTCOME_COLUMNS | REFERENCE_DERIVED_COLUMNS
            if name in frame.columns
        ],
        errors="ignore",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-arm", default=DEFAULT_PARENT)
    parser.add_argument("--generated-cache", default=DEFAULT_CACHE)
    args = parser.parse_args()
    root = DEFAULT_OUT_DIR
    parent = str(args.parent_arm)
    arm = f"{parent}_globaloverlay"
    arm_dir = root / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    generated = pd.read_parquet(root / "cache" / str(args.generated_cache))
    burnin = pd.read_parquet(
        root / "lifecycle_only_burnin" / "oos_predictions_march_burnin.parquet"
    )
    burnin = _merge_residual_features(burnin, generated)
    burnin["calendar_month"] = "2026-03"
    burnin["score_lifecycle_only"] = pd.to_numeric(
        burnin["score_alternative"], errors="coerce"
    ).astype(np.float32)
    parent_state = joblib.load(root / parent / "residual_overlay_state.joblib")
    state = replace(parent_state, local_hit_alpha=0.0, local_dirty_lambda=0.0)
    burnin["score_global_overlay"] = state.transform(
        _safe(burnin),
        burnin["score_lifecycle_only"].fillna(0.5).to_numpy(dtype=np.float32),
    )
    calibration_mask = _selection_mask(
        burnin,
        "score_global_overlay",
        0.10,
        ["calendar_month", "side_name"],
    )
    calibrator = _fit_platt(
        burnin.loc[calibration_mask, "score_global_overlay"],
        burnin.loc[calibration_mask, "clean_exec"],
    )
    oos = pd.read_parquet(root / parent / "oos_predictions.parquet")
    oos["score_alternative"] = state.transform(
        _safe(oos),
        pd.to_numeric(oos["score_lifecycle_only"], errors="coerce")
        .fillna(0.5)
        .to_numpy(dtype=np.float32),
    )
    oos["hit_prob_alternative"] = _calibrate(calibrator, oos["score_alternative"])
    oos.to_parquet(arm_dir / "oos_predictions.parquet", index=False, compression="zstd")
    metrics = metrics_by_scope(oos, arm)
    calendar, autocorr, comparison = surprise_calendar(oos, arm)
    metrics.to_csv(arm_dir / "metrics_by_scope.csv", index=False)
    calendar.to_csv(arm_dir / "hit_surprise_calendar.csv", index=False)
    autocorr.to_csv(arm_dir / "hit_surprise_autocorrelation.csv", index=False)
    comparison.to_csv(arm_dir / "high_surprise_period_comparison.csv", index=False)
    joblib.dump(state, arm_dir / "residual_overlay_state.joblib", compress=3)
    joblib.dump(calibrator, arm_dir / "hit_calibrator.joblib", compress=3)
    manifest = {
        "schema": "meta_residual_overlay_locality_ablation_v1",
        "arm": arm,
        "parent_arm": parent,
        "generated_cache": str(args.generated_cache),
        "local_hit_alpha_parent": float(parent_state.local_hit_alpha),
        "local_dirty_lambda_parent": float(parent_state.local_dirty_lambda),
        "local_hit_alpha": 0.0,
        "local_dirty_lambda": 0.0,
        "selection_rule": "OOS placebo rejection of local archetype normalization",
        "current_meta_model_overwritten": False,
    }
    (arm_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
