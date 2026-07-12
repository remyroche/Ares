#!/usr/bin/env python3
"""Diagnostic-only in-sample capacity check for residual meta archetypes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_train_meta_residual_archetype_enhancement import (  # noqa: E402
    DEFAULT_OUT_DIR,
    _add_reference_fold_features,
    _calibrate,
    _fit_platt,
    metrics_by_scope,
    surprise_calendar,
)

ARM = "lifecycle_residual_aware_ae_gmm_overlay_insample_capacity"


def main() -> None:
    root = DEFAULT_OUT_DIR
    out_dir = root / "final_report"
    data = pd.read_parquet(root / "cache" / "compact_reference_with_lifecycle.parquet")
    data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True, errors="coerce")
    july_start = pd.Timestamp("2026-07-01", tz="UTC")
    train = data[data["__ts__"].lt(july_start)].copy()
    valid = data[data["__ts__"].ge(july_start)].copy()
    train, _ = _add_reference_fold_features(train, valid)
    march = train[
        train["__ts__"].ge(pd.Timestamp("2026-03-01", tz="UTC"))
        & train["__ts__"].lt(pd.Timestamp("2026-04-01", tz="UTC"))
    ].copy()
    bundle_dir = root / "inference_bundle_residual_ae_gmm"
    bundle = joblib.load(bundle_dir / "alternative_meta_residual_ae_gmm_bundle.joblib")
    predicted = bundle.predict(march)
    keep = [
        name
        for name in (
            "__ts__",
            "__symbol__",
            "side_name",
            "archetype_policy_key",
            "archetype_label_family",
            "ev_after_1pct",
            "exec_margin",
            "clean_exec",
            "dirty_positive",
            "first_touch_bad_mae_1r",
            "full_path_bad_mae_1r",
            "timeout",
            "score_meta_base_soft_label",
        )
        if name in march.columns
    ]
    scored = march[keep].copy()
    scored["calendar_month"] = "2026-03"
    scored["week_start"] = scored["__ts__"].dt.floor("D") - pd.to_timedelta(
        scored["__ts__"].dt.weekday,
        unit="D",
    )
    scored["score_current_reference"] = pd.to_numeric(
        scored["score_meta_base_soft_label"], errors="coerce"
    )
    scored["score_alternative"] = predicted["score_residual_overlay"].to_numpy()
    scored["hit_prob_alternative_frozen_bundle"] = predicted[
        "hit_probability"
    ].to_numpy()
    capacity_calibrator = _fit_platt(scored["score_alternative"], scored["clean_exec"])
    scored["hit_prob_alternative"] = _calibrate(
        capacity_calibrator,
        scored["score_alternative"],
    ).astype(np.float32)
    burnin = pd.read_parquet(
        root / "lifecycle_only_burnin" / "oos_predictions_march_burnin.parquet"
    )
    ref = burnin[
        [
            "__ts__",
            "__symbol__",
            "side_name",
            "archetype_policy_key",
            "hit_prob_current_reference",
        ]
    ].drop_duplicates(
        ["__ts__", "__symbol__", "side_name", "archetype_policy_key"], keep="last"
    )
    scored = scored.merge(
        ref,
        on=["__ts__", "__symbol__", "side_name", "archetype_policy_key"],
        how="left",
        validate="one_to_one",
    )
    scored["hit_prob_current_reference"] = pd.to_numeric(
        scored["hit_prob_current_reference"], errors="coerce"
    ).fillna(0.5)
    metrics = metrics_by_scope(scored, ARM)
    calendar, autocorr, comparison = surprise_calendar(scored, ARM)
    scored.to_parquet(
        bundle_dir / "march_insample_capacity_predictions.parquet",
        index=False,
        compression="zstd",
    )
    metrics.to_csv(out_dir / "march_insample_capacity_metrics.csv", index=False)
    calendar.to_csv(out_dir / "march_insample_capacity_calendar.csv", index=False)
    autocorr.to_csv(
        out_dir / "march_insample_capacity_autocorrelation.csv", index=False
    )
    comparison.to_csv(
        out_dir / "march_insample_capacity_high_surprise_comparison.csv", index=False
    )
    overall = metrics[metrics["scope"].eq("overall") & metrics["fraction"].eq(0.10)]
    base = overall[overall["selector"].eq("current_reference")].iloc[0]
    alt = overall[overall["selector"].eq(ARM)].iloc[0]
    tail = comparison[comparison["baseline_high_surprise"].fillna(False).astype(bool)]
    manifest = {
        "schema": "meta_residual_insample_capacity_v1",
        "diagnostic_only": True,
        "rows": int(len(scored)),
        "period": "2026-03",
        "model_fit_through": str(bundle.fit_through),
        "top10_ev_current": float(base["mean_ev_after_1pct"]),
        "top10_ev_alternative": float(alt["mean_ev_after_1pct"]),
        "top10_clean_current": float(base["clean_exec_precision"]),
        "top10_clean_alternative": float(alt["clean_exec_precision"]),
        "high_surprise_cells": int(len(tail)),
        "high_surprise_cells_improved": int(
            tail["high_surprise_significantly_improved"].sum()
        ),
        "high_surprise_improvement_rate": float(
            tail["high_surprise_significantly_improved"].mean()
        ),
        "capacity_probability_calibration": "same_sample_platt_diagnostic_only",
        "interpretation": (
            "This deliberately reuses a model/recognizer fitted through June on March rows. "
            "Its probability map is also refitted on March to separate representation capacity "
            "from frozen-calibrator distribution shift. It is excluded from every OOS and "
            "inference claim."
        ),
    }
    (out_dir / "march_insample_capacity_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
