#!/usr/bin/env python3
"""Causal hourly CNN/TCN benchmark for calendar-event warning.

This is intentionally a small research challenger to the lag-summary LightGBM
detector.  Inputs are limited to frozen 1h-or-longer OHLCV/OI/funding state
features.  Each fold fits preprocessing, matching, the sequence model, and
the fixed-FPR threshold only before its embargoed OOS period.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from extreme_price_movements.hourly_extreme_event_detection import (
    HourlyEventConfig,
    available_hourly_features,
    build_hourly_market_state,
    calendar_hourly_targets,
    matched_control_weights,
)
from scripts.run_hourly_extreme_event_detector import (
    DEFAULT_CALENDAR,
    DEFAULT_FOLDS,
    DEFAULT_STATE,
    DEFAULT_TAXONOMY,
    _load_calendar,
    _load_hourly_rows,
    _load_taxonomy,
    _timestamp,
)
from scripts.run_residual_hard_period_cnn import _cnn_fit_predict, _fill_scale


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data_perp/reports/hourly_extreme_event_sequence_benchmark_20260714_v1"


def _folds(values: list[str]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    return [(_timestamp(value.split("::", 1)[0]), _timestamp(value.split("::", 1)[1])) for value in values]


def _windows(values: np.ndarray, window: int) -> np.ndarray:
    """Build compact left-padded causal windows without dataframe copies."""

    rows, features = values.shape
    result = np.zeros((rows, features, window), dtype=np.float32)
    for index in range(rows):
        left = max(0, index - window + 1)
        segment = values[left : index + 1]
        result[index, :, -len(segment) :] = segment.T
    return result


def _calibrated_scores(
    train_values: np.ndarray,
    target: np.ndarray,
    timestamps: pd.Series,
    oos_values: np.ndarray,
    *,
    window: int,
    seed: int,
    epochs: int,
    architecture: str,
    fpr: float,
) -> tuple[float, np.ndarray]:
    """Fit once on an early train block and preserve its calibration scale.

    Neural scores are not inherently calibrated across independently fitted
    models.  The core fit is therefore deliberately retained for calibration
    and OOS scoring; refitting after threshold selection would invalidate a
    fixed-FPR claim.
    """

    split = int(len(train_values) * 0.75)
    if split < window + 48 or len(train_values) - split < 24:
        return 1.0, np.full(len(oos_values), np.nan, dtype=np.float32)
    core_x, all_x = _fill_scale(
        train_values[:split].copy(),
        np.vstack([train_values, oos_values]).astype(np.float32, copy=False),
    )
    core_y = target[:split]
    calibration_y = target[split:]
    weights = matched_control_weights(
        pd.DataFrame({"__ts__": timestamps.iloc[:split].to_numpy(), "target": core_y, "vol": core_x[:, 0]}),
        target_column="target",
        volatility_column="vol",
        control_ratio=4,
        seed=seed,
    )
    keep = weights > 0
    if keep.sum() < 16 or core_y[keep].sum() < 3:
        return 1.0, np.full(len(oos_values), np.nan, dtype=np.float32)
    all_windows = _windows(all_x, window)
    model_scores = _cnn_fit_predict(
        _windows(core_x, window)[keep],
        core_y[keep],
        all_windows[split:],
        seed=seed,
        epochs=epochs,
        architecture=architecture,
    )
    calibration_scores = model_scores[: len(calibration_y)]
    normal = calibration_scores[(calibration_y == 0) & np.isfinite(calibration_scores)]
    threshold = float(np.quantile(normal, 1.0 - fpr)) if len(normal) >= 10 else 1.0
    return threshold, model_scores[len(calibration_y) :]


def _metrics(score: np.ndarray, y: np.ndarray, threshold: float) -> dict[str, float | int]:
    selected = np.nan_to_num(score, nan=-np.inf) >= threshold
    normal = y == 0
    precision = float(y[selected].mean()) if selected.any() else np.nan
    prevalence = float(y.mean()) if len(y) else np.nan
    return {
        "event_hours": int(y.sum()),
        "selected_hours": int(selected.sum()),
        "fpr": float(selected[normal].mean()) if normal.any() else np.nan,
        "recall": float((selected & (y > 0)).sum() / max(int(y.sum()), 1)),
        "precision": precision,
        "lift": precision / prevalence if np.isfinite(precision) and prevalence > 0 else np.nan,
    }


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    schema = pq.ParquetFile(args.state_artifact).schema.names
    features = available_hourly_features(schema)
    config = HourlyEventConfig(embargo_hours=args.embargo_hours, lead_hours=args.lead_hours)
    hourly = build_hourly_market_state(_load_hourly_rows(args.state_artifact, features), feature_columns=features, config=config)
    labels = calendar_hourly_targets(hourly, _load_calendar(args.calendar), _load_taxonomy(args.taxonomy), config=config)
    state_columns = [*hourly.attrs["observable_features"], *hourly.attrs["transition_features"]]
    panel = hourly.merge(labels.drop(columns="day"), on="__ts__", how="left")
    fold_specs = args.fold or [f"{start}::{end}" for start, end in DEFAULT_FOLDS]
    reports: list[dict[str, object]] = []
    outputs: list[pd.DataFrame] = []
    for fold, (start, end) in enumerate(_folds(fold_specs)):
        train_end = start - pd.Timedelta(hours=args.embargo_hours)
        train = panel.loc[panel["__ts__"].lt(train_end)].copy()
        score = panel.loc[panel["__ts__"].ge(start) & panel["__ts__"].lt(end)].copy()
        if len(train) < args.window_hours + 72 or score.empty:
            continue
        raw_train = train[state_columns].to_numpy(np.float32, copy=True)
        raw_score = score[state_columns].to_numpy(np.float32, copy=True)
        y_train = train["event_onset_next_window"].to_numpy(np.int8)
        y_score = score["event_onset_next_window"].to_numpy(np.int8)
        threshold, probabilities = _calibrated_scores(
            raw_train,
            y_train,
            train["__ts__"].reset_index(drop=True),
            raw_score,
            window=args.window_hours,
            seed=args.seed + fold,
            epochs=args.epochs,
            architecture=args.architecture,
            fpr=args.fixed_fpr,
        )
        output = score.loc[:, ["__ts__", "event_state", "event_onset", "event_onset_next_window"]].copy()
        output["sequence_score"] = probabilities
        output["event_threshold"] = threshold
        output["fold"] = fold
        output["architecture"] = args.architecture
        outputs.append(output)
        reports.append({
            "fold": fold,
            "architecture": args.architecture,
            "oos_start": start,
            "oos_end": end,
            "train_end": train_end,
            "feature_count": len(state_columns),
            "window_hours": args.window_hours,
            "threshold": threshold,
            **_metrics(probabilities, y_score, threshold),
        })
    pd.DataFrame(reports).to_csv(args.output / "hourly_sequence_fold_metrics.csv", index=False)
    predictions = pd.concat(outputs, ignore_index=True, copy=False) if outputs else pd.DataFrame()
    predictions.to_parquet(args.output / "hourly_sequence_oos_predictions.parquet", index=False, compression="zstd")
    (args.output / "manifest.json").write_text(json.dumps({
        "purpose": "research-only causal hourly CNN/TCN eventness challenger",
        "resolution": "1h only",
        "subhour_data_used": False,
        "architecture": args.architecture,
        "window_hours": args.window_hours,
        "embargo_hours": args.embargo_hours,
        "features": state_columns,
        "folds": fold_specs,
        "train_only": "robust scaling, matched controls, model fit, fixed-FPR threshold",
        "no_policy_wiring": True,
    }, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-artifact", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--architecture", choices=("cnn", "tcn"), default="cnn")
    parser.add_argument("--fold", action="append", default=[])
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument("--lead-hours", type=int, default=12)
    parser.add_argument("--embargo-hours", type=int, default=36)
    parser.add_argument("--fixed-fpr", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
