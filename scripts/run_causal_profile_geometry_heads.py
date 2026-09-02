#!/usr/bin/env python3
"""Fit 2025-only causal profile/geometry source heads for 2026 confirmation.

Unlike the rolling MC1 map, these source models are fit once using only
resolved 2025 events.  June, July, and August 2026 are confirmation-only
months.  The output is a target-free candidate-time feature source for a
separate downstream MC1 challenger; it never changes candidate eligibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from extreme_price_movements.causal_profile_geometry import PROFILE_FEATURES

SOURCE = ROOT / "data_perp/artifacts/causal_profile_geometry_2025_train_2026_score_20260831_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/causal_profile_geometry_heads_2025train_2026confirm_20260831_v1"
SEED = 1729
TRAIN_END = pd.Timestamp("2026-01-01T00:00:00Z")
HELD_DEFAULT = ("2026-06", "2026-07", "2026-08")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _matrix(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[:, PROFILE_FEATURES].copy()
    for column in PROFILE_FEATURES:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.replace([np.inf, -np.inf], np.nan)


def _regressor(*, quantile: bool = False) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="quantile" if quantile else "regression_l1", alpha=.50,
        n_estimators=360 if quantile else 320, learning_rate=.03, max_depth=3, num_leaves=7,
        min_child_samples=180, subsample=.80, colsample_bytree=.85, reg_lambda=14.0,
        random_state=SEED, n_jobs=2, verbosity=-1,
    )


def _classifier() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=340, learning_rate=.03, max_depth=3, num_leaves=7,
        min_child_samples=180, subsample=.80, colsample_bytree=.85, reg_lambda=16.0,
        random_state=SEED, n_jobs=2, verbosity=-1,
    )


def _score(train: pd.DataFrame, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train, x_score = _matrix(train), _matrix(frame)
    utility = _regressor().fit(x_train, pd.to_numeric(train.y_profile_utility_atr, errors="raise"))
    magnitude = _regressor(quantile=True).fit(x_train, pd.to_numeric(train.y_profile_favorable_mfe_atr, errors="raise"))
    classifier = _classifier().fit(x_train, pd.to_numeric(train.y_profile_adverse_break, errors="raise").astype(int))
    result = frame.copy()
    result["profile_conditional_utility"] = utility.predict(x_score)
    result["profile_favorable_magnitude_q50"] = magnitude.predict(x_score)
    result["profile_adverse_break_probability"] = classifier.predict_proba(x_score)[:, 1]
    importance = pd.DataFrame({
        "feature": PROFILE_FEATURES,
        "utility_gain": utility.booster_.feature_importance(importance_type="gain"),
        "magnitude_gain": magnitude.booster_.feature_importance(importance_type="gain"),
        "adverse_break_gain": classifier.booster_.feature_importance(importance_type="gain"),
    })
    return result, importance


def _metrics(frame: pd.DataFrame, held: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for prediction, target in (
        ("profile_conditional_utility", "y_profile_utility_atr"),
        ("profile_favorable_magnitude_q50", "y_profile_favorable_mfe_atr"),
    ):
        pred = pd.to_numeric(frame[prediction], errors="coerce")
        truth = pd.to_numeric(frame[target], errors="coerce")
        valid = pred.notna() & truth.notna()
        records.append({
            "held_month": held, "head": prediction, "target": target, "rows": int(valid.sum()),
            "mae": float(mean_absolute_error(truth[valid], pred[valid])) if valid.any() else np.nan,
            "spearman": float(truth[valid].corr(pred[valid], method="spearman")) if valid.sum() > 2 else np.nan,
        })
    pred = pd.to_numeric(frame.profile_adverse_break_probability, errors="coerce")
    truth = pd.to_numeric(frame.y_profile_adverse_break, errors="coerce")
    valid = pred.notna() & truth.notna()
    records.append({
        "held_month": held, "head": "profile_adverse_break_probability", "target": "y_profile_adverse_break", "rows": int(valid.sum()),
        "base_rate": float(truth[valid].mean()) if valid.any() else np.nan,
        "auc": float(roc_auc_score(truth[valid].astype(int), pred[valid])) if valid.sum() > 2 and truth[valid].nunique() == 2 else np.nan,
        "brier": float(brier_score_loss(truth[valid].astype(int), pred[valid])) if valid.any() else np.nan,
        "spearman": float(truth[valid].corr(pred[valid], method="spearman")) if valid.sum() > 2 else np.nan,
    })
    return records


def _calibration(frame: pd.DataFrame, held: str) -> pd.DataFrame:
    work = frame.loc[:, ["profile_conditional_utility", "y_profile_utility_atr"]].copy()
    work = work.dropna()
    if len(work) < 20:
        return pd.DataFrame()
    work["prediction_decile"] = pd.qcut(work.profile_conditional_utility.rank(method="first"), q=min(10, len(work)), labels=False)
    output = work.groupby("prediction_decile", sort=True).agg(
        rows=("y_profile_utility_atr", "size"), predicted_utility_atr=("profile_conditional_utility", "mean"), realised_utility_atr=("y_profile_utility_atr", "mean"),
    ).reset_index()
    output.insert(0, "held_month", held)
    return output


def _price_diagnostics(frame: pd.DataFrame, held: str) -> pd.DataFrame:
    fields = ("profile_inside_value_area", "bb_zscore", "donchian_position", "profile_oi_positioning_imbalance")
    rows: list[dict[str, object]] = []
    target = pd.to_numeric(frame.y_profile_utility_atr, errors="coerce")
    for field in fields:
        values = pd.to_numeric(frame[field], errors="coerce")
        valid = values.notna() & target.notna()
        rows.append({"held_month": held, "field": field, "rows": int(valid.sum()), "spearman_to_future_utility_atr": float(values[valid].corr(target[valid], method="spearman")) if valid.sum() > 2 else np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--held-month", action="append", default=list(HELD_DEFAULT))
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    events = pd.read_parquet(source / "profile_events.parquet")
    snapshots = pd.read_parquet(source / "profile_snapshots.parquet")
    events["event_ts"] = pd.to_datetime(events.event_ts, utc=True, errors="raise")
    events["label_available_ts"] = pd.to_datetime(events.label_available_ts, utc=True, errors="raise")
    snapshots["snapshot_ts"] = pd.to_datetime(snapshots.snapshot_ts, utc=True, errors="raise")
    required = set(PROFILE_FEATURES).union({"y_profile_utility_atr", "y_profile_favorable_mfe_atr", "y_profile_adverse_break"})
    missing = sorted(required.difference(events.columns))
    if missing:
        raise AssertionError(f"profile source contract missing {missing}")
    train = events.loc[events.event_ts.lt(TRAIN_END) & events.label_available_ts.lt(TRAIN_END)].copy()
    if len(train) < 10_000:
        raise RuntimeError(f"insufficient 2025 resolved profile events: {len(train)}")
    output.mkdir(parents=True, exist_ok=False)
    event_frames: list[pd.DataFrame] = []
    snapshot_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    calibration: list[pd.DataFrame] = []
    price_diagnostics: list[pd.DataFrame] = []
    importances: list[pd.DataFrame] = []
    folds: list[dict[str, object]] = []
    for held_raw in args.held_month:
        held = pd.Timestamp(f"{held_raw}-01", tz="UTC")
        end = held + pd.offsets.MonthBegin(1)
        event_test = events.loc[events.event_ts.ge(held) & events.event_ts.lt(end) & events.label_available_ts.lt(end)].copy()
        snapshot_test = snapshots.loc[snapshots.snapshot_ts.ge(held) & snapshots.snapshot_ts.lt(end)].copy()
        if event_test.empty or snapshot_test.empty:
            raise RuntimeError(f"missing 2026 confirmation population for {held_raw}: events={len(event_test)}, snapshots={len(snapshot_test)}")
        scored_event, importance = _score(train, event_test)
        scored_snapshot, _ = _score(train, snapshot_test)
        scored_event["held_month"] = held_raw; scored_snapshot["held_month"] = held_raw
        metric_rows.extend(_metrics(scored_event, held_raw)); calibration.append(_calibration(scored_event, held_raw)); price_diagnostics.append(_price_diagnostics(scored_event, held_raw))
        importance.insert(0, "held_month", held_raw); importances.append(importance)
        event_frames.append(scored_event); snapshot_frames.append(scored_snapshot)
        folds.append({"held_month": held_raw, "train_rows": int(len(train)), "train_label_max": str(train.label_available_ts.max()), "event_test_rows": int(len(event_test)), "snapshot_test_rows": int(len(snapshot_test))})
    event_result = pd.concat(event_frames, ignore_index=True)
    snapshot_result = pd.concat(snapshot_frames, ignore_index=True)
    snapshot_result["profile_snapshot_available"] = snapshot_result.loc[:, PROFILE_FEATURES].notna().sum(axis=1).ge(16).astype("int8")
    event_result.to_parquet(output / "profile_head_oof_predictions.parquet", index=False, compression="zstd")
    snapshot_result.to_parquet(output / "entry_profile_geometry_oof_features.parquet", index=False, compression="zstd")
    pd.DataFrame(metric_rows).to_parquet(output / "head_metrics_by_month.parquet", index=False)
    pd.concat(calibration, ignore_index=True).to_parquet(output / "utility_calibration_by_month.parquet", index=False)
    pd.concat(price_diagnostics, ignore_index=True).to_parquet(output / "price_diagnostics_by_month.parquet", index=False)
    pd.concat(importances, ignore_index=True).to_parquet(output / "feature_importance_by_month.parquet", index=False)
    pd.DataFrame(folds).to_parquet(output / "fold_trace.parquet", index=False)
    manifest = {
        "schema": "causal-profile-geometry-heads-2025train-2026confirm-v1",
        "scope": "offline source-model research only; no live mutation or exchange calls",
        "source": str(source), "source_manifest_sha256": _sha256(source / "run_manifest.json"),
        "training": "all and only 2025 event labels with label_available_ts before 2026-01-01 UTC",
        "confirmation": list(args.held_month), "folds": folds,
        "heads": {
            "profile_conditional_utility": {"target": "next-8h MFE minus MAE, ATR units", "model": "LGBM L1 depth3 leaves7"},
            "profile_favorable_magnitude_q50": {"target": "next-8h long MFE, ATR units", "model": "LGBM median quantile depth3 leaves7"},
            "profile_adverse_break_probability": {"target": "next-8h adverse move >= .75 ATR before favourable excursion", "model": "LGBM binary depth3 leaves7"},
        },
        "features": list(PROFILE_FEATURES),
        "causality": "fixed log-price profile grid and trailing completed-hour inputs; OI is strict-prior; labels remain absent from snapshots; 2026 does not train source heads",
        "seed": SEED,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
