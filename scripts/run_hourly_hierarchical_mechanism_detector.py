#!/usr/bin/env python3
"""Leakage-safe, mechanism-specific hourly adverse-period detector.

The generic layer is deliberately unsupervised: causal state-change and
distance-from-normal estimates answer whether the current market is unusual.
Separate shallow LightGBM ensembles then model only *pre-onset* windows for
each named mechanism.  This avoids forcing liquidation, compression,
fragmentation, and divergence into one contradictory binary target.

Outputs are research-only.  They are not a policy, recovery, sizing, or live
inference input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score

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
    _fit_lgbm,
    _load_calendar,
    _load_hourly_rows,
    _load_taxonomy,
    _novelty,
    _screen_features,
    _timestamp,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data_perp/reports/hourly_hierarchical_mechanism_detector_20260714_v1"


def _folds(values: list[str]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    parsed: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for value in values:
        start, separator, end = value.partition("::")
        if not separator:
            raise ValueError(f"Expected START::END, got {value!r}")
        parsed.append((_timestamp(start), _timestamp(end)))
    return parsed


def _rank_normalize(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Empirical upper-tail percentile using train-only normal references."""

    reference = np.sort(reference[np.isfinite(reference)])
    if len(reference) < 16:
        return np.zeros(len(values), dtype=np.float32)
    return (np.searchsorted(reference, values, side="right") / len(reference)).astype(np.float32)


def _generic_abnormality(train: pd.DataFrame, score: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Causal change plus novelty, with no event target in the generic layer."""

    _, train_novelty = _novelty(train, train, features, "event_state")
    _, score_novelty = _novelty(train, score, features, "event_state")
    normal = train["event_state"].to_numpy(np.int8) == 0
    change_column = "evt_causal_change_score"
    if change_column in train:
        train_change = pd.to_numeric(train[change_column], errors="coerce").to_numpy(np.float32)
        score_change = pd.to_numeric(score[change_column], errors="coerce").to_numpy(np.float32)
    else:
        train_change = np.zeros(len(train), dtype=np.float32)
        score_change = np.zeros(len(score), dtype=np.float32)
    novelty_pct_train = _rank_normalize(train_novelty, train_novelty[normal])
    novelty_pct_score = _rank_normalize(score_novelty, train_novelty[normal])
    change_pct_train = _rank_normalize(train_change, train_change[normal])
    change_pct_score = _rank_normalize(score_change, train_change[normal])
    return (
        (0.5 * novelty_pct_train + 0.5 * change_pct_train).astype(np.float32),
        (0.5 * novelty_pct_score + 0.5 * change_pct_score).astype(np.float32),
    )


def _fit_seed_ensemble(
    train: pd.DataFrame,
    score: pd.DataFrame,
    *,
    target: str,
    candidates: list[str],
    config: HourlyEventConfig,
    seeds: list[int],
    fpr: float,
    maximum_features: int,
) -> tuple[np.ndarray, float, list[str], list[dict[str, float]]]:
    """Fit core-only ensembles and calibrate them on later train history."""

    split = int(len(train) * 0.75)
    core = train.iloc[:split].copy()
    calibration = train.iloc[split:].copy()
    if len(core) < 96 or len(calibration) < 24:
        return np.full(len(score), np.nan, dtype=np.float32), np.nan, [], []
    selected = _screen_features(core, candidates, target, maximum_features)
    if not selected:
        return np.full(len(score), np.nan, dtype=np.float32), np.nan, [], []
    y_core = pd.to_numeric(core[target], errors="coerce").fillna(0).to_numpy(np.int8)
    if y_core.sum() < 3:
        return np.full(len(score), np.nan, dtype=np.float32), np.nan, selected, []
    volatility = "mkt_rv_4h" if "mkt_rv_4h" in selected else selected[0]
    calibration_scores: list[np.ndarray] = []
    oos_scores: list[np.ndarray] = []
    seed_rows: list[dict[str, float]] = []
    y_score = pd.to_numeric(score[target], errors="coerce").fillna(0).to_numpy(np.int8)
    for seed in seeds:
        weights = matched_control_weights(
            core,
            target_column=target,
            volatility_column=volatility,
            control_ratio=config.control_ratio,
            seed=seed,
        )
        keep = weights > 0
        if keep.sum() < 16 or y_core[keep].sum() < 3:
            continue
        _, calibration_score = _fit_lgbm(
            core.loc[keep], calibration, features=selected, target=target,
            weights=weights[keep], seed=seed,
        )
        _, oos_score = _fit_lgbm(
            core.loc[keep], score, features=selected, target=target,
            weights=weights[keep], seed=seed,
        )
        calibration_scores.append(calibration_score)
        oos_scores.append(oos_score)
        finite = np.isfinite(oos_score)
        seed_rows.append({
            "seed": float(seed),
            "roc_auc": float(roc_auc_score(y_score[finite], oos_score[finite]))
            if finite.sum() >= 20 and y_score[finite].sum() >= 3 and (y_score[finite] == 0).sum() >= 10
            else np.nan,
        })
    if not oos_scores:
        return np.full(len(score), np.nan, dtype=np.float32), np.nan, selected, seed_rows
    calibration_mean = np.nanmean(np.vstack(calibration_scores), axis=0)
    calibration_y = pd.to_numeric(calibration[target], errors="coerce").fillna(0).to_numpy(np.int8)
    normal = calibration_mean[(calibration_y == 0) & np.isfinite(calibration_mean)]
    threshold = float(np.quantile(normal, 1.0 - fpr)) if len(normal) >= 10 else np.nan
    matrix = np.vstack(oos_scores)
    return np.nanmean(matrix, axis=0).astype(np.float32), threshold, selected, seed_rows


def _metric_row(score: np.ndarray, y: np.ndarray, threshold: float) -> dict[str, float | int]:
    finite = np.isfinite(score)
    if finite.sum() < 20 or y[finite].sum() < 3 or (y[finite] == 0).sum() < 10:
        return {"status": "insufficient_oos_support"}
    selected = finite & (score >= threshold)
    normal = y == 0
    precision = float(y[selected].mean()) if selected.any() else np.nan
    prevalence = float(y[finite].mean())
    return {
        "status": "ok",
        "oos_positive_hours": int(y[finite].sum()),
        "roc_auc": float(roc_auc_score(y[finite], score[finite])),
        "average_precision": float(average_precision_score(y[finite], score[finite])),
        "selected_hours": int(selected.sum()),
        "fpr": float(selected[normal].mean()) if normal.any() else np.nan,
        "recall": float((selected & (y > 0)).sum() / max(int(y.sum()), 1)),
        "precision": precision,
        "lift": precision / prevalence if np.isfinite(precision) and prevalence > 0 else np.nan,
    }


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    config = HourlyEventConfig(lead_hours=args.lead_hours, embargo_hours=args.embargo_hours, control_ratio=args.control_ratio)
    schema = pq.ParquetFile(args.state_artifact).schema.names
    observable = available_hourly_features(schema)
    hourly = build_hourly_market_state(_load_hourly_rows(args.state_artifact, observable), feature_columns=observable, config=config)
    labels = calendar_hourly_targets(hourly, _load_calendar(args.calendar), _load_taxonomy(args.taxonomy), config=config)
    features = [*hourly.attrs["observable_features"], *hourly.attrs["transition_features"]]
    panel = hourly.merge(labels.drop(columns="day"), on="__ts__", how="left")
    targets = [name for name in panel if name.startswith("mechanism__") and name.endswith("__pre_onset_next_window")]
    fold_specs = args.fold or [f"{start}::{end}" for start, end in DEFAULT_FOLDS]
    reports: list[dict[str, object]] = []
    seed_reports: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for fold, (start, end) in enumerate(_folds(fold_specs)):
        train = panel.loc[panel["__ts__"].lt(start - pd.Timedelta(hours=config.embargo_hours))].copy()
        score = panel.loc[panel["__ts__"].ge(start) & panel["__ts__"].lt(end)].copy()
        if train.empty or score.empty:
            continue
        generic_train, generic_oos = _generic_abnormality(train, score, features)
        output = score.loc[:, ["__ts__", "event_state", "event_onset", "event_pre_onset_next_window"]].copy()
        output["generic_abnormal_score"] = generic_oos
        output["fold"] = fold
        for target in targets:
            mechanism = target.removeprefix("mechanism__").removesuffix("__pre_onset_next_window")
            score_values, threshold, selected, seed_rows = _fit_seed_ensemble(
                train, score, target=target, candidates=features, config=config,
                seeds=[args.seed + index for index in range(args.seeds)], fpr=args.fixed_fpr,
                maximum_features=args.max_features,
            )
            column = f"mechanism_pre_onset_score__{mechanism}"
            output[column] = score_values
            output[f"mechanism_pre_onset_threshold__{mechanism}"] = threshold
            y = pd.to_numeric(score[target], errors="coerce").fillna(0).to_numpy(np.int8)
            seed_auc = np.asarray([row["roc_auc"] for row in seed_rows], dtype=np.float32)
            for row in seed_rows:
                seed_reports.append({"fold": fold, "mechanism": mechanism, **row})
            reports.append({
                "fold": fold,
                "oos_start": start,
                "oos_end": end,
                "mechanism": mechanism,
                "target": "pre_onset_next_window",
                "train_positive_hours": int(pd.to_numeric(train[target], errors="coerce").fillna(0).sum()),
                "feature_count": len(selected),
                "features": "|".join(selected),
                "threshold": threshold,
                "seeds": args.seeds,
                "seed_auc_mean": float(np.nanmean(seed_auc)) if np.isfinite(seed_auc).any() else np.nan,
                "seed_auc_std": float(np.nanstd(seed_auc)) if np.isfinite(seed_auc).any() else np.nan,
                **_metric_row(score_values, y, threshold),
            })
        predictions.append(output)
    pd.DataFrame(reports).to_csv(args.output / "hierarchical_mechanism_fold_metrics.csv", index=False)
    pd.DataFrame(seed_reports).to_csv(args.output / "hierarchical_mechanism_seed_stability.csv", index=False)
    result = pd.concat(predictions, ignore_index=True, copy=False) if predictions else pd.DataFrame()
    result.to_parquet(args.output / "hierarchical_mechanism_oos_predictions.parquet", index=False, compression="zstd")
    (args.output / "manifest.json").write_text(json.dumps({
        "purpose": "research-only generic-abnormal plus mechanism-specific pre-onset detector",
        "resolution": "1h only",
        "subhour_data_used": False,
        "generic_layer": "train-only robust novelty plus causal change score; no event target",
        "mechanism_layer": "10-seed shallow LGBM ensembles, train-only feature selection/control matching/calibration",
        "phase_targets": ["pre_onset_next_window", "onset", "active_stress", "recovery"],
        "folds": fold_specs,
        "embargo_hours": args.embargo_hours,
        "no_policy_wiring": True,
    }, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-artifact", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fold", action="append", default=[])
    parser.add_argument("--lead-hours", type=int, default=12)
    parser.add_argument("--embargo-hours", type=int, default=36)
    parser.add_argument("--control-ratio", type=int, default=4)
    parser.add_argument("--fixed-fpr", type=float, default=0.05)
    parser.add_argument("--max-features", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
