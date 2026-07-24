#!/usr/bin/env python3
"""Walk-forward hourly detector for known and unknown adverse market periods.

This is a research-only context producer.  It reads a frozen hourly residual
state artifact, predicts broad eventness, mechanism probabilities, and novelty
with train-only transforms, then evaluates fixed-FPR alerts on future periods.
It never uses trade outcomes, residuals, recent hit-rate, or policy decisions
as inference features and does not modify the live overlay or policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score

from extreme_price_movements.hourly_extreme_event_detection import (
    HourlyEventConfig,
    available_hourly_features,
    build_hourly_market_state,
    calendar_hourly_targets,
    causal_episode_memory,
    matched_control_weights,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = ROOT / (
    "data_perp/reports/residual_event_archetype_true_base_oof_compactlocal_market_20260712_v3/"
    "oos_residual_event_states.parquet"
)
DEFAULT_CALENDAR = ROOT / (
    "data_perp/reports/residual_episode_recognition_calendar_20260712_v1/"
    "calendar_recognized_vs_ignored.csv"
)
DEFAULT_TAXONOMY = ROOT / (
    "data_perp/reports/residual_event_block_taxonomy_20260714_v7_full_mechanism_calendar/"
    "event_block_mechanism_calendar.csv"
)
DEFAULT_OUTPUT = ROOT / "data_perp/reports/hourly_extreme_event_detector_20260714_v1"
DEFAULT_FOLDS = (
    ("2025-10-01", "2026-01-01"),
    ("2026-01-01", "2026-04-01"),
    ("2026-04-01", "2026-07-01"),
)


def _timestamp(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def _load_hourly_rows(path: Path, features: list[str]) -> pd.DataFrame:
    schema = pq.ParquetFile(path).schema.names
    available = [name for name in features if name in schema]
    if not available:
        raise ValueError("No hourly observable state columns are present in the source artifact")
    source = pq.ParquetFile(path)
    pieces: list[pd.DataFrame] = []
    for batch in source.iter_batches(columns=["__ts__", *available], batch_size=100_000):
        frame = batch.to_pandas()
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
        frame = frame.loc[frame["__ts__"].notna()]
        if frame.empty:
            continue
        for name in available:
            frame[name] = pd.to_numeric(frame[name], errors="coerce", downcast="float")
        # Hourly source rows are sorted in timestamp order.  Reducing batches
        # before concatenation prevents materializing the full 1M+ row matrix.
        pieces.append(frame.groupby("__ts__", as_index=False, sort=False)[available].median())
    if not pieces:
        raise ValueError(f"No usable rows in {path}")
    result = pd.concat(pieces, ignore_index=True, copy=False)
    return result.groupby("__ts__", as_index=False, sort=True)[available].median()


def _load_calendar(path: Path) -> pd.DataFrame:
    calendar = pd.read_csv(path)
    calendar["day"] = pd.to_datetime(calendar["day"], utc=True).dt.floor("D")
    return calendar.drop_duplicates(["day", "side_name", "archetype_policy_key"], keep="last")


def _load_taxonomy(path: Path) -> pd.DataFrame:
    taxonomy = pd.read_csv(path)
    taxonomy["event_start"] = pd.to_datetime(taxonomy["event_start"], utc=True).dt.floor("D")
    return taxonomy


def _fit_robust_scale(train: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    median = np.nanmedian(train, axis=0)
    q25 = np.nanquantile(train, 0.25, axis=0)
    q75 = np.nanquantile(train, 0.75, axis=0)
    scale = np.maximum(q75 - q25, 1e-4)
    median = np.nan_to_num(median, nan=0.0).astype(np.float32)
    scale = np.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)
    for values in (train, score):
        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = np.take(median, np.nonzero(missing)[1])
        values -= median
        values /= scale
        np.clip(values, -8.0, 8.0, out=values)
    return train.astype(np.float32, copy=False), score.astype(np.float32, copy=False), median, scale


def _screen_features(train: pd.DataFrame, candidates: list[str], target: str, maximum: int) -> list[str]:
    y = pd.to_numeric(train[target], errors="coerce").fillna(0).to_numpy(np.int8)
    positive = y > 0
    if positive.sum() < 3 or (~positive).sum() < 10:
        return []
    ranked: list[tuple[str, float]] = []
    for name in candidates:
        values = pd.to_numeric(train[name], errors="coerce").to_numpy(np.float32, copy=False)
        finite = np.isfinite(values)
        if finite.mean() < 0.8 or not finite[positive].any() or not finite[~positive].any():
            continue
        q25, q75 = np.nanquantile(values[finite], [0.25, 0.75])
        effect = abs(float(np.nanmedian(values[positive]) - np.nanmedian(values[~positive]))) / max(float(q75 - q25), 1e-4)
        ranked.append((name, effect))
    return [name for name, _ in sorted(ranked, key=lambda item: item[1], reverse=True)[:maximum]]


def _fit_lgbm(
    train: pd.DataFrame,
    score: pd.DataFrame,
    *,
    features: list[str],
    target: str,
    weights: np.ndarray | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_train = train[features].to_numpy(np.float32, copy=True)
    x_score = score[features].to_numpy(np.float32, copy=True)
    x_train, x_score, _, _ = _fit_robust_scale(x_train, x_score)
    y = pd.to_numeric(train[target], errors="coerce").fillna(0).to_numpy(np.int8)
    if y.sum() < 3 or (y == 0).sum() < 10:
        return np.full(len(train), np.nan, dtype=np.float32), np.full(len(score), np.nan, dtype=np.float32)
    model = lgb.train(
        {
            "objective": "binary",
            "metric": "None",
            "learning_rate": 0.035,
            "max_depth": 3,
            "num_leaves": 7,
            "min_data_in_leaf": max(12, min(96, len(train) // 30)),
            "min_gain_to_split": 0.04,
            "lambda_l1": 3.0,
            "lambda_l2": 14.0,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "seed": int(seed),
            "num_threads": 1,
            "verbosity": -1,
            "force_col_wise": True,
        },
        lgb.Dataset(x_train, label=y, weight=weights, feature_name=features),
        num_boost_round=140,
    )
    return (
        np.asarray(model.predict(x_train), dtype=np.float32),
        np.asarray(model.predict(x_score), dtype=np.float32),
    )


def _threshold_from_calibration(
    train: pd.DataFrame,
    features: list[str],
    target: str,
    *,
    config: HourlyEventConfig,
    fpr: float,
    seed: int,
) -> float:
    split = int(len(train) * 0.75)
    core = train.iloc[:split].copy()
    calibration = train.iloc[split:].copy()
    if len(core) < 48 or len(calibration) < 24:
        return 1.0
    volatility = "mkt_rv_4h" if "mkt_rv_4h" in features else features[0]
    weights = matched_control_weights(
        core,
        target_column=target,
        volatility_column=volatility,
        control_ratio=config.control_ratio,
        seed=seed,
    )
    used = weights > 0
    if used.sum() < 16:
        return 1.0
    _, scores = _fit_lgbm(
        core.loc[used], calibration, features=features, target=target, weights=weights[used], seed=seed
    )
    normal = scores[pd.to_numeric(calibration[target], errors="coerce").fillna(0).to_numpy(np.int8) == 0]
    normal = normal[np.isfinite(normal)]
    return float(np.quantile(normal, 1.0 - fpr)) if len(normal) >= 10 else 1.0


def _novelty(train: pd.DataFrame, score: pd.DataFrame, features: list[str], normal_target: str) -> tuple[np.ndarray, np.ndarray]:
    x_train = train[features].to_numpy(np.float32, copy=True)
    x_score = score[features].to_numpy(np.float32, copy=True)
    x_train, x_score, _, _ = _fit_robust_scale(x_train, x_score)
    normal = pd.to_numeric(train[normal_target], errors="coerce").fillna(0).to_numpy(np.int8) == 0
    reference = x_train[normal] if normal.sum() >= 32 else x_train
    # A diagonal robust distance is intentionally used instead of a covariance
    # inverse: it is stable with sparse rare events and has a direct frozen
    # inference contract.
    center = np.median(reference, axis=0)
    dist_train = np.sqrt(np.mean(np.square(x_train - center), axis=1)).astype(np.float32)
    dist_score = np.sqrt(np.mean(np.square(x_score - center), axis=1)).astype(np.float32)
    return dist_train, dist_score


def _lead_time_hours(frame: pd.DataFrame, threshold: float, lead_hours: int) -> float | None:
    onsets = frame.loc[frame["event_onset"].eq(1), "__ts__"]
    lead_values: list[float] = []
    for onset in onsets:
        before = frame.loc[
            frame["__ts__"].between(onset - pd.Timedelta(hours=lead_hours), onset)
            & frame["event_score"].ge(threshold),
            "__ts__",
        ]
        if not before.empty:
            lead_values.append(float((onset - before.min()).total_seconds() / 3600.0))
    return float(np.median(lead_values)) if lead_values else None


def _metrics(frame: pd.DataFrame, threshold: float, target: str) -> dict[str, float | int | None]:
    selected = frame["event_score"].ge(threshold).to_numpy(bool)
    y = pd.to_numeric(frame[target], errors="coerce").fillna(0).to_numpy(np.int8)
    normal = y == 0
    precision = float(y[selected].mean()) if selected.any() else None
    prevalence = float(y.mean()) if len(y) else None
    return {
        "rows": int(len(frame)),
        "event_rows": int(y.sum()),
        "selected_rows": int(selected.sum()),
        "selected_rate": float(selected.mean()) if len(selected) else None,
        "precision": precision,
        "fpr": float(selected[normal].mean()) if normal.any() else None,
        "recall": float((selected & (y > 0)).sum() / max(int(y.sum()), 1)),
        "lift": precision / prevalence if precision is not None and prevalence and prevalence > 0 else None,
    }


def _mechanism_metric_rows(frame: pd.DataFrame, fold: int) -> list[dict[str, Any]]:
    """Evaluate known-family probability without treating absent classes as wins."""

    rows: list[dict[str, Any]] = []
    for probability_column in (name for name in frame if name.startswith("mechanism_prob__")):
        target = probability_column.replace("mechanism_prob__", "mechanism__")
        if target not in frame:
            continue
        probability = pd.to_numeric(frame[probability_column], errors="coerce").to_numpy(np.float32)
        y = pd.to_numeric(frame[target], errors="coerce").fillna(0).to_numpy(np.int8)
        finite = np.isfinite(probability)
        if finite.sum() < 20 or y[finite].sum() < 3 or (y[finite] == 0).sum() < 10:
            rows.append({
                "fold": fold,
                "mechanism": target.removeprefix("mechanism__"),
                "status": "insufficient_oos_support",
                "oos_positive_hours": int(y[finite].sum()),
            })
            continue
        selected_count = max(1, int(np.ceil(0.05 * finite.sum())))
        threshold = float(np.partition(probability[finite], -selected_count)[-selected_count])
        selected = finite & (probability >= threshold)
        rows.append({
            "fold": fold,
            "mechanism": target.removeprefix("mechanism__"),
            "status": "ok",
            "oos_positive_hours": int(y[finite].sum()),
            "roc_auc": float(roc_auc_score(y[finite], probability[finite])),
            "average_precision": float(average_precision_score(y[finite], probability[finite])),
            "top05_precision": float(y[selected].mean()) if selected.any() else np.nan,
            "top05_recall": float((selected & (y > 0)).sum() / max(int(y.sum()), 1)),
            "top05_fpr": float(selected[y == 0].mean()) if (y == 0).any() else np.nan,
        })
    return rows


def _event_block_rows(
    predictions: pd.DataFrame,
    taxonomy: pd.DataFrame,
    *,
    lead_hours: int,
) -> list[dict[str, Any]]:
    """Score taxonomy cells by pre-onset alert, never retrospective selection."""

    rows: list[dict[str, Any]] = []
    if predictions.empty:
        return rows
    for event in taxonomy.itertuples(index=False):
        start = pd.Timestamp(event.event_start)
        end = pd.Timestamp(event.event_end)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        fold_frame = predictions.loc[
            predictions["__ts__"].ge(start - pd.Timedelta(hours=lead_hours))
            & predictions["__ts__"].lt(max(end, start + pd.Timedelta(days=1)))
        ]
        if fold_frame.empty:
            continue
        before = fold_frame.loc[fold_frame["__ts__"].lt(start)]
        during = fold_frame.loc[fold_frame["__ts__"].ge(start) & fold_frame["__ts__"].lt(end + pd.Timedelta(days=1))]
        threshold = float(fold_frame["event_threshold"].iloc[0])
        expected = str(event.onset_primary_mechanism)
        probability_column = f"mechanism_prob__{expected}"
        pre_alerts = before.loc[before["event_score"].ge(threshold), "__ts__"]
        mechanism_score = (
            float(during[probability_column].max())
            if probability_column in during and during[probability_column].notna().any()
            else np.nan
        )
        rows.append({
            "event_start": start,
            "event_end": end,
            "side_name": str(event.side_name),
            "archetype_policy_key": str(event.archetype_policy_key),
            "event_block": str(event.event_block),
            "expected_mechanism": expected,
            "fold": int(fold_frame["fold"].iloc[0]),
            "recognized_pre_onset": bool(len(pre_alerts)),
            "lead_hours": float((start - pre_alerts.min()).total_seconds() / 3600.0) if len(pre_alerts) else np.nan,
            "alert_during_event": bool(during["event_score"].ge(threshold).any()),
            "max_pre_event_score": float(before["event_score"].max()) if not before.empty else np.nan,
            "max_event_score": float(during["event_score"].max()) if not during.empty else np.nan,
            "max_unknown_abnormal_score": float(during["unknown_abnormal_score"].max()) if not during.empty else np.nan,
            "expected_mechanism_max_probability": mechanism_score,
        })
    return rows


def _folds(values: list[str]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    result = []
    for value in values:
        start, _, end = value.partition("::")
        if not end:
            raise ValueError(f"Expected START::END fold, got {value!r}")
        result.append((_timestamp(start), _timestamp(end)))
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=True)
    schema = pq.ParquetFile(args.state_artifact).schema.names
    features = available_hourly_features(schema)
    source = _load_hourly_rows(args.state_artifact, features)
    config = HourlyEventConfig(
        decision_lag_hours=args.decision_lag_hours,
        lead_hours=args.lead_hours,
        state_hours=args.state_hours,
        embargo_hours=args.embargo_hours,
        control_ratio=args.control_ratio,
        max_features=args.max_features,
    )
    hourly = build_hourly_market_state(source, feature_columns=features, config=config)
    calendar = _load_calendar(args.calendar)
    taxonomy = _load_taxonomy(args.taxonomy)
    labelled = calendar_hourly_targets(hourly, calendar, taxonomy, config=config)
    state_columns = list(hourly.attrs["observable_features"]) + list(hourly.attrs["transition_features"])
    panel = hourly.merge(labelled.drop(columns=["day"]), on="__ts__", how="left")
    # The phase-specific mechanism labels are used by the hierarchical
    # challenger.  This first-generation runner retains its historical
    # active-state target only, avoiding an accidental explosion of models.
    mechanism_targets = [
        name for name in panel.columns
        if name.startswith("mechanism__") and "__" not in name.removeprefix("mechanism__")
    ]
    all_predictions: list[pd.DataFrame] = []
    reports: list[dict[str, Any]] = []
    mechanism_reports: list[dict[str, Any]] = []

    fold_specs = args.fold or [f"{start}::{end}" for start, end in DEFAULT_FOLDS]
    for fold_index, (start, end) in enumerate(_folds(fold_specs)):
        train_end = start - pd.Timedelta(hours=config.embargo_hours)
        train = panel.loc[panel["__ts__"].lt(train_end)].copy()
        score = panel.loc[panel["__ts__"].ge(start) & panel["__ts__"].lt(end)].copy()
        if train.empty or score.empty:
            continue
        selected = _screen_features(train, state_columns, "event_onset_next_window", config.max_features)
        if not selected:
            continue
        threshold = _threshold_from_calibration(
            train,
            selected,
            "event_onset_next_window",
            config=config,
            fpr=args.fixed_fpr,
            seed=args.seed + fold_index,
        )
        volatility = "mkt_rv_4h" if "mkt_rv_4h" in selected else selected[0]
        weights = matched_control_weights(
            train,
            target_column="event_onset_next_window",
            volatility_column=volatility,
            control_ratio=config.control_ratio,
            seed=args.seed + fold_index,
        )
        used = weights > 0
        _, event_score = _fit_lgbm(
            train.loc[used], score, features=selected, target="event_onset_next_window",
            weights=weights[used], seed=args.seed + fold_index,
        )
        novelty_train, novelty_score = _novelty(train, score, selected, "event_state")
        novelty_threshold = float(np.quantile(novelty_train, 1.0 - args.fixed_fpr))
        score["event_score"] = event_score
        score["novelty_score"] = novelty_score
        score["event_threshold"] = threshold
        score["novelty_threshold"] = novelty_threshold
        mechanism_scores: dict[str, np.ndarray] = {}
        for mechanism_target in mechanism_targets:
            if pd.to_numeric(train[mechanism_target], errors="coerce").fillna(0).sum() < args.min_mechanism_events:
                score[mechanism_target.replace("mechanism__", "mechanism_prob__")] = np.nan
                continue
            mechanism_features = _screen_features(
                train,
                selected,
                mechanism_target,
                max(8, min(16, len(selected))),
            )
            if not mechanism_features:
                continue
            local_weights = matched_control_weights(
                train,
                target_column=mechanism_target,
                volatility_column=volatility,
                control_ratio=config.control_ratio,
                seed=args.seed + fold_index + len(mechanism_scores) + 1,
            )
            keep = local_weights > 0
            _, probabilities = _fit_lgbm(
                train.loc[keep], score, features=mechanism_features, target=mechanism_target,
                weights=local_weights[keep], seed=args.seed + fold_index + len(mechanism_scores) + 1,
            )
            name = mechanism_target.replace("mechanism__", "mechanism_prob__")
            score[name] = probabilities
            mechanism_scores[mechanism_target.removeprefix("mechanism__")] = probabilities
            mechanism_reports.append(
                {
                    "fold": fold_index,
                    "start": start,
                    "end": end,
                    "mechanism": mechanism_target.removeprefix("mechanism__"),
                    "train_events": int(pd.to_numeric(train[mechanism_target], errors="coerce").fillna(0).sum()),
                    "features": "|".join(mechanism_features),
                }
            )
        probability_columns = [name for name in score.columns if name.startswith("mechanism_prob__")]
        if probability_columns:
            values = score[probability_columns].to_numpy(np.float32, copy=True)
            values[~np.isfinite(values)] = 0.0
            ordered = np.sort(values, axis=1)
            score["mechanism_class_margin"] = ordered[:, -1] - (ordered[:, -2] if values.shape[1] > 1 else 0.0)
        else:
            score["mechanism_class_margin"] = 0.0
        novelty_pct = np.clip(novelty_score / max(novelty_threshold, 1e-4), 0.0, 2.0) / 2.0
        score["unknown_abnormal_score"] = (
            0.60 * np.nan_to_num(event_score, nan=0.0)
            + 0.25 * novelty_pct
            + 0.15 * (1.0 - np.clip(score["mechanism_class_margin"].to_numpy(np.float32), 0.0, 1.0))
        ).astype(np.float32)
        memory = causal_episode_memory(event_score, mechanism_scores, threshold=threshold)
        score = pd.concat([score.reset_index(drop=True), memory], axis=1)
        score["fold"] = fold_index
        reports.append(
            {
                "fold": fold_index,
                "train_end": train_end,
                "oos_start": start,
                "oos_end": end,
                "selected_features": "|".join(selected),
                "feature_count": len(selected),
                "event_threshold": threshold,
                "novelty_threshold": novelty_threshold,
                "median_lead_hours": _lead_time_hours(score, threshold, config.lead_hours),
                **_metrics(score, threshold, "event_onset_next_window"),
            }
        )
        mechanism_reports.extend(_mechanism_metric_rows(score, fold_index))
        all_predictions.append(score)

    predictions = pd.concat(all_predictions, ignore_index=True, copy=False) if all_predictions else pd.DataFrame()
    pd.DataFrame(reports).to_csv(args.output / "hourly_event_detector_fold_metrics.csv", index=False)
    pd.DataFrame(mechanism_reports).to_csv(args.output / "hourly_event_detector_mechanism_metrics.csv", index=False)
    pd.DataFrame(
        _event_block_rows(predictions, taxonomy, lead_hours=config.lead_hours)
    ).to_csv(args.output / "hourly_event_detector_event_block_metrics.csv", index=False)
    predictions.to_parquet(args.output / "hourly_event_detector_oos_predictions.parquet", index=False, compression="zstd")
    manifest = {
        "purpose": "research-only hourly extreme-period context; no live policy activation",
        "resolution": "1h only",
        "subhour_data_used": False,
        "state_artifact": str(args.state_artifact),
        "calendar": str(args.calendar),
        "taxonomy": str(args.taxonomy),
        "available_observable_features": features,
        "state_feature_count": len(state_columns),
        "folds": [(str(start), str(end)) for start, end in _folds(fold_specs)],
        "config": vars(args),
        "causal_contract": {
            "decision_lag_hours": config.decision_lag_hours,
            "transition_features": "computed after the hourly decision lag",
            "train_only": "screening, imputation, scaling, model fit, thresholds, novelty references",
            "purge_embargo_hours": config.embargo_hours,
            "labels": "daily calendar labels expanded to hourly targets; never copied into inference features",
            "no_policy_wiring": True,
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-artifact", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fold", action="append", default=[], help="OOS fold START::END; defaults to the canonical three folds")
    parser.add_argument("--decision-lag-hours", type=int, default=1)
    parser.add_argument("--lead-hours", type=int, default=12)
    parser.add_argument("--state-hours", type=int, default=24)
    parser.add_argument("--embargo-hours", type=int, default=36)
    parser.add_argument("--control-ratio", type=int, default=4)
    parser.add_argument("--max-features", type=int, default=32)
    parser.add_argument("--min-mechanism-events", type=int, default=8)
    parser.add_argument("--fixed-fpr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(f"completed hourly-only folds={len(result['folds'])} features={len(result['available_observable_features'])}")
