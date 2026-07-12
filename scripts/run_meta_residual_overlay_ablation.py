#!/usr/bin/env python3
"""Tune a causal residual-archetype score overlay on March and test Apr-Jun OOS."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.meta_residual_overlay import (
    ResidualOverlayState,  # noqa: E402
)
from scripts.run_train_meta_residual_archetype_enhancement import (  # noqa: E402
    DEFAULT_OUT_DIR,
    _calibrate,
    _fit_platt,
    _merge_residual_features,
    _metric_record,
    _selection_mask,
    metrics_by_scope,
    surprise_calendar,
    train_arm_oos,
)

RISK_COLUMNS = (
    "meta_resid_arch_expected_dirty_positive",
    "meta_resid_arch_expected_bad_mae",
    "meta_resid_arch_expected_timeout",
    "meta_resid_arch_expected_hit_surprise",
)


def _load_contract(
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    data = pd.read_parquet(root / "cache" / "compact_reference_with_lifecycle.parquet")
    data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True, errors="coerce")
    residual = pd.read_parquet(root / "cache" / "residual_walkforward_raw.parquet")
    selected_rows = pd.read_csv(root / "lifecycle_only" / "selected_features.csv")
    selected_mask = (
        selected_rows["selected"].fillna(False).astype(bool)
        if "selected" in selected_rows
        else True
    )
    selected = selected_rows.loc[selected_mask, "feature"].astype(str).tolist()
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    params = dict(manifest["reference_model_params"])
    return data, residual, selected, params


def _objective(frame: pd.DataFrame, score_col: str) -> dict[str, float]:
    frame["hit_prob_alternative"] = pd.to_numeric(
        frame[score_col], errors="coerce"
    ).clip(0.0, 1.0)
    records: dict[float, dict[str, Any]] = {}
    for frac in (0.10, 0.15, 0.20):
        mask = _selection_mask(frame, score_col, frac, ["calendar_month", "side_name"])
        records[frac] = _metric_record(frame, mask, score_col, "burnin", frac)
    week_values: list[float] = []
    for _, week in frame.groupby("week_start", sort=True):
        mask = _selection_mask(week, score_col, 0.10, ["calendar_month", "side_name"])
        selected = week.loc[mask]
        week_values.append(
            float(pd.to_numeric(selected["ev_after_1pct"], errors="coerce").mean())
        )
    worst_week = float(np.nanmin(week_values)) if week_values else float("nan")
    r10 = records[0.10]
    score = (
        100.0
        * (
            0.45 * float(r10["mean_ev_after_1pct"])
            + 0.20 * float(records[0.15]["mean_ev_after_1pct"])
            + 0.10 * float(records[0.20]["mean_ev_after_1pct"])
            + 0.15 * worst_week
        )
        + 0.10
        * (float(r10["clean_exec_precision"]) - float(r10["dirty_positive_rate"]))
        - 0.05 * float(r10["first_touch_bad_mae_rate"])
    )
    return {
        "objective": float(score),
        "top10_ev": float(r10["mean_ev_after_1pct"]),
        "top15_ev": float(records[0.15]["mean_ev_after_1pct"]),
        "top20_ev": float(records[0.20]["mean_ev_after_1pct"]),
        "worst_week_top10_ev": worst_week,
        "top10_clean": float(r10["clean_exec_precision"]),
        "top10_dirty": float(r10["dirty_positive_rate"]),
        "top10_first_bad": float(r10["first_touch_bad_mae_rate"]),
    }


def _overlay_score(
    frame: pd.DataFrame, *, hit_alpha: float, dirty_lambda: float
) -> np.ndarray:
    base = (
        pd.to_numeric(frame["score_alternative"], errors="coerce")
        .fillna(0.5)
        .to_numpy(dtype=np.float32)
    )
    hit = (
        pd.to_numeric(frame[RISK_COLUMNS[3]], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    dirty = (
        pd.to_numeric(frame[RISK_COLUMNS[0]], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    return np.clip(
        base + np.float32(hit_alpha) * hit - np.float32(dirty_lambda) * dirty, 0.0, 1.0
    )


def _local_overlay_objective(
    frame: pd.DataFrame,
    state: ResidualOverlayState,
    *,
    base_col: str,
) -> dict[str, float]:
    safe = frame.drop(
        columns=[
            name
            for name in (
                "clean_exec",
                "dirty_positive",
                "first_touch_bad_mae_1r",
                "full_path_bad_mae_1r",
                "timeout",
                "ev_after_1pct",
                "exec_margin",
            )
            if name in frame.columns
        ]
    )
    frame["score_overlay_local"] = state.transform(
        safe,
        pd.to_numeric(frame[base_col], errors="coerce")
        .fillna(0.5)
        .to_numpy(dtype=np.float32),
    )
    return _objective(frame, "score_overlay_local")


def main() -> None:
    root = DEFAULT_OUT_DIR
    data, residual, selected, params = _load_contract(root)
    burnin, _ = train_arm_oos(
        arm="lifecycle_only_burnin",
        data=data,
        selected_features=selected,
        params=params,
        output_dir=root,
        seed=20260711,
        eval_months=("2026-03",),
        artifact_tag="march_burnin",
    )
    burnin = _merge_residual_features(burnin, residual)
    burnin["week_start"] = (
        pd.to_datetime(burnin["__ts__"], utc=True)
        .dt.to_period("W-MON")
        .dt.start_time.dt.tz_localize("UTC")
    )
    burnin["calendar_month"] = "2026-03"
    search_rows: list[dict[str, Any]] = []
    for hit_alpha in (0.0, 0.025, 0.05, 0.10, 0.20):
        for dirty_lambda in (0.0, 0.025, 0.05, 0.10, 0.20):
            burnin["score_overlay"] = _overlay_score(
                burnin, hit_alpha=hit_alpha, dirty_lambda=dirty_lambda
            )
            row = _objective(burnin, "score_overlay")
            search_rows.append(
                {"hit_alpha": hit_alpha, "dirty_lambda": dirty_lambda, **row}
            )
    search = pd.DataFrame(search_rows).sort_values(
        "objective", ascending=False, kind="stable"
    )
    best = search.iloc[0]
    search.to_csv(root / "residual_overlay_burnin_search.csv", index=False)

    # The raw priors have different levels by side/archetype.  Fit their local
    # centering on March only, then tune a small relative-position nudge without
    # touching April-June outcomes.
    local_search_rows: list[dict[str, Any]] = []
    normalizer = ResidualOverlayState(
        hit_alpha=float(best["hit_alpha"]),
        dirty_lambda=float(best["dirty_lambda"]),
    ).fit_normalization(burnin)
    for local_hit_alpha in (-0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.03, 0.04):
        for local_dirty_lambda in (0.0, 0.0025, 0.005, 0.01, 0.02):
            state = ResidualOverlayState(
                hit_alpha=float(best["hit_alpha"]),
                dirty_lambda=float(best["dirty_lambda"]),
                local_hit_alpha=float(local_hit_alpha),
                local_dirty_lambda=float(local_dirty_lambda),
                group_stats=dict(normalizer.group_stats),
                side_stats=dict(normalizer.side_stats),
                global_stats=normalizer.global_stats,
                calibration_start=normalizer.calibration_start,
                calibration_end=normalizer.calibration_end,
            )
            row = _local_overlay_objective(burnin, state, base_col="score_alternative")
            local_search_rows.append(
                {
                    "hit_alpha": float(best["hit_alpha"]),
                    "dirty_lambda": float(best["dirty_lambda"]),
                    "local_hit_alpha": local_hit_alpha,
                    "local_dirty_lambda": local_dirty_lambda,
                    **row,
                }
            )
    local_search = pd.DataFrame(local_search_rows).sort_values(
        "objective", ascending=False, kind="stable"
    )
    # A near-tied larger local nudge is more likely to memorize one burn-in
    # month.  Apply a deterministic one-standard-error-style simplification:
    # retain candidates within a small absolute objective tolerance, then use
    # the least local adjustment before considering the raw objective tie-break.
    best_objective = float(local_search["objective"].max())
    local_objective_tolerance = max(0.002, 0.005 * abs(best_objective))
    local_search["within_simplification_tolerance"] = local_search["objective"].ge(
        best_objective - local_objective_tolerance
    )
    local_search["local_adjustment_l1"] = (
        local_search["local_hit_alpha"].abs() + local_search["local_dirty_lambda"].abs()
    )
    local_best = (
        local_search[local_search["within_simplification_tolerance"]]
        .sort_values(
            ["local_adjustment_l1", "objective"],
            ascending=[True, False],
            kind="stable",
        )
        .iloc[0]
    )
    local_search.to_csv(root / "residual_overlay_local_burnin_search.csv", index=False)
    local_state = ResidualOverlayState(
        hit_alpha=float(local_best["hit_alpha"]),
        dirty_lambda=float(local_best["dirty_lambda"]),
        local_hit_alpha=float(local_best["local_hit_alpha"]),
        local_dirty_lambda=float(local_best["local_dirty_lambda"]),
        group_stats=dict(normalizer.group_stats),
        side_stats=dict(normalizer.side_stats),
        global_stats=normalizer.global_stats,
        calibration_start=normalizer.calibration_start,
        calibration_end=normalizer.calibration_end,
    )
    _local_overlay_objective(burnin, local_state, base_col="score_alternative")

    lifecycle_path = root / "lifecycle_only" / "oos_predictions.parquet"
    if not lifecycle_path.exists():
        lifecycle_path = root / "lifecycle_only" / "oos_predictions_apr_may_jun.parquet"
    lifecycle = pd.read_parquet(lifecycle_path)
    scored = _merge_residual_features(lifecycle, residual)
    scored["score_lifecycle_only"] = pd.to_numeric(
        scored["score_alternative"], errors="coerce"
    ).astype(np.float32)
    scored["score_alternative"] = _overlay_score(
        scored,
        hit_alpha=float(best["hit_alpha"]),
        dirty_lambda=float(best["dirty_lambda"]),
    )
    platt = _fit_platt(
        burnin.assign(
            score_overlay=_overlay_score(
                burnin,
                hit_alpha=float(best["hit_alpha"]),
                dirty_lambda=float(best["dirty_lambda"]),
            )
        )["score_overlay"],
        burnin["clean_exec"],
    )
    scored["hit_prob_alternative"] = _calibrate(platt, scored["score_alternative"])
    arm = "lifecycle_residual_overlay"
    arm_dir = root / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(
        arm_dir / "oos_predictions.parquet", index=False, compression="zstd"
    )
    metrics = metrics_by_scope(scored, arm)
    calendar, autocorr, comparison = surprise_calendar(scored, arm)
    metrics.to_csv(arm_dir / "metrics_by_scope.csv", index=False)
    calendar.to_csv(arm_dir / "hit_surprise_calendar.csv", index=False)
    autocorr.to_csv(arm_dir / "hit_surprise_autocorrelation.csv", index=False)
    comparison.to_csv(arm_dir / "high_surprise_period_comparison.csv", index=False)
    manifest = {
        "schema": "lifecycle_residual_overlay_v1",
        "burnin_month": "2026-03",
        "burnin_is_final_oos": False,
        "oos_months": ["2026-04", "2026-05", "2026-06"],
        "hit_alpha": float(best["hit_alpha"]),
        "dirty_lambda": float(best["dirty_lambda"]),
        "burnin_metrics": best.to_dict(),
        "leakage_contract": "Overlay strength selected on March burn-in only; April-June untouched.",
    }
    (arm_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    safe_scored = scored.drop(
        columns=[
            name
            for name in (
                "clean_exec",
                "dirty_positive",
                "first_touch_bad_mae_1r",
                "full_path_bad_mae_1r",
                "timeout",
                "ev_after_1pct",
                "exec_margin",
            )
            if name in scored.columns
        ]
    )
    local_scored = scored.copy()
    local_scored["score_alternative"] = local_state.transform(
        safe_scored,
        pd.to_numeric(scored["score_lifecycle_only"], errors="coerce")
        .fillna(0.5)
        .to_numpy(dtype=np.float32),
    )
    local_platt = _fit_platt(
        burnin["score_overlay_local"],
        burnin["clean_exec"],
    )
    local_scored["hit_prob_alternative"] = _calibrate(
        local_platt, local_scored["score_alternative"]
    )
    local_arm = "lifecycle_residual_local_overlay"
    local_dir = root / local_arm
    local_dir.mkdir(parents=True, exist_ok=True)
    local_scored.to_parquet(
        local_dir / "oos_predictions.parquet", index=False, compression="zstd"
    )
    local_metrics = metrics_by_scope(local_scored, local_arm)
    local_calendar, local_autocorr, local_comparison = surprise_calendar(
        local_scored, local_arm
    )
    local_metrics.to_csv(local_dir / "metrics_by_scope.csv", index=False)
    local_calendar.to_csv(local_dir / "hit_surprise_calendar.csv", index=False)
    local_autocorr.to_csv(local_dir / "hit_surprise_autocorrelation.csv", index=False)
    local_comparison.to_csv(
        local_dir / "high_surprise_period_comparison.csv", index=False
    )
    joblib.dump(local_state, local_dir / "residual_overlay_state.joblib")
    (local_dir / "residual_overlay_state.json").write_text(
        json.dumps(local_state.manifest(), indent=2),
        encoding="utf-8",
    )
    local_manifest = {
        "schema": "lifecycle_residual_local_overlay_v1",
        "parent_arm": arm,
        "burnin_month": "2026-03",
        "burnin_is_final_oos": False,
        "oos_months": ["2026-04", "2026-05", "2026-06"],
        "hit_alpha": float(local_best["hit_alpha"]),
        "dirty_lambda": float(local_best["dirty_lambda"]),
        "local_hit_alpha": float(local_best["local_hit_alpha"]),
        "local_dirty_lambda": float(local_best["local_dirty_lambda"]),
        "local_selection_rule": "minimum_l1_within_objective_tolerance",
        "local_objective_best": best_objective,
        "local_objective_tolerance": local_objective_tolerance,
        "burnin_metrics": local_best.to_dict(),
        "state_artifact": str(local_dir / "residual_overlay_state.joblib"),
        "current_meta_model_overwritten": False,
        "leakage_contract": (
            "Raw and local overlay strengths plus normalization selected/fitted on March burn-in only; "
            "April-June untouched; inference transform rejects outcomes."
        ),
    }
    (local_dir / "manifest.json").write_text(
        json.dumps(local_manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    print(json.dumps(local_manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
