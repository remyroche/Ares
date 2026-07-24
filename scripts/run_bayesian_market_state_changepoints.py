#!/usr/bin/env python3
"""Evaluate causal Bayesian market-state breaks against adverse episode blocks.

This is an episode-recognition diagnostic, not a trading policy.  It tests the
hypothesis that adverse residual episodes arise when several observable market
dynamics break simultaneously.  The detector sees only hourly OHLCV/OI/funding
state features.  Calendar outcomes are used exclusively after scoring to
measure block recall and false-alert rates.

Feature candidates come from prior RuleFit/BRL/contrastive model selections
when available.  Each chronological fold uses only selections from strictly
earlier reported folds; a compact, outcome-free market feature fallback covers
the first fold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.bayesian_changepoint import (
    BOCPDConfig,
    bocpd_student_t,
    robust_scale_train_oos,
    synchronized_break_score,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = ROOT / (
    "data_perp/reports/meta_residual_interpretable_rule_overlay_20260714_v10_corrected_clock/"
    "negative_residual_market_features_20250301_20260712.parquet"
)
DEFAULT_FEATURE_REPORT = ROOT / (
    "data_perp/reports/meta_residual_interpretable_rule_overlay_20260714_v18_episode_intervention/"
    "model_report.csv"
)
DEFAULT_CALENDAR = ROOT / (
    "data_perp/reports/residual_event_block_taxonomy_20260714_v7_full_mechanism_calendar/"
    "event_block_mechanism_calendar.csv"
)
DEFAULT_OUTPUT = ROOT / "data_perp/reports/bayesian_market_state_changepoints_20260714_v1"
DEFAULT_FOLDS = (
    "2025-07-01::2025-10-01",
    "2025-10-01::2026-01-01",
    "2026-01-01::2026-04-01",
    "2026-04-01::2026-07-01",
)

# Observables only.  The fallback deliberately excludes model scores, recent
# realized damage, residual labels, and any past-performance proxies.
FALLBACK_MARKET_FEATURES = (
    "negative_breadth_pct",
    "extreme_negative_breadth_pct",
    "median_alt_minus_btc",
    "breadth_dispersion",
    "downside_breadth_intensity",
    "btc_oi_dominance_z_ratio",
    "btc_ex_eth_oi_dominance_z_ratio",
    "short_covering_score_market",
    "funding_confirmed_long_flush",
    "funding_confirmed_short_covering",
    "flush_recovery_state",
    "compression_quality_consistency",
    "fragile_compression_score",
    "breakout_disagreement_score",
    "mkt_regime_change__funding__delta_1h",
    "mkt_regime_change__oi_contraction__delta_1h",
    "mkt_regime_change__negative_breadth__delta_1h",
    "mkt_regime_change__eth_correlation__delta_1h",
    "mkt_regime_change__btc_alt_relative_strength__delta_1h",
    "mkt_regime_change__short_covering__delta_1h",
    "mkt_regime_change__flush_recovery__delta_1h",
)
FORBIDDEN_FEATURE_TOKENS = (
    "short_default_damage",
    "resid_event",
    "period_",
    "parent_rank",
    "meta_score",
    "hit_probability",
    "expected_",
    "adverse_",
    "clean_exec",
    "ev_after",
)


def _timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _folds(values: list[str]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    output: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for value in values:
        start, delimiter, end = value.partition("::")
        if not delimiter:
            raise ValueError(f"Expected START::END, got {value!r}")
        output.append((_timestamp(start), _timestamp(end)))
    return output


def _load_market_state(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    timestamp_column = "__index_level_0__" if "__index_level_0__" in frame else "__ts__"
    if timestamp_column in frame:
        timestamp = frame.pop(timestamp_column)
    elif frame.index.name in {"__index_level_0__", "__ts__"}:
        timestamp = frame.index.to_series(index=frame.index)
    elif isinstance(frame.index, pd.DatetimeIndex):
        timestamp = frame.index.to_series(index=frame.index)
    else:
        raise KeyError("Market state needs a timestamp index column")
    frame["__ts__"] = pd.to_datetime(timestamp, utc=True, errors="coerce").to_numpy()
    frame = frame.loc[frame["__ts__"].notna()].sort_values("__ts__", kind="stable")
    return frame.drop_duplicates("__ts__", keep="last").reset_index(drop=True)


def _is_observable_market_feature(name: str, available: set[str]) -> bool:
    lower = name.lower()
    return (
        name in available
        and not any(token in lower for token in FORBIDDEN_FEATURE_TOKENS)
    )


def _prior_rule_features(
    report: pd.DataFrame,
    available: set[str],
    fold_start: pd.Timestamp,
    maximum: int,
) -> tuple[list[str], pd.DataFrame]:
    """Use only feature selections reported strictly before this OOS fold."""

    frame = report.copy()
    if "stage" not in frame or "fold_start" not in frame or "selected_features" not in frame:
        return [], pd.DataFrame(columns=["feature", "prior_model_uses"])
    frame["fold_start"] = pd.to_datetime(frame["fold_start"], utc=True, errors="coerce")
    frame = frame.loc[frame["stage"].eq("oof") & frame["fold_start"].lt(fold_start)]
    counts: dict[str, int] = {}
    for encoded in frame["selected_features"].dropna().astype(str):
        for feature in encoded.split("|"):
            if _is_observable_market_feature(feature, available):
                counts[feature] = counts.get(feature, 0) + 1
    ordered = sorted(counts, key=lambda name: (-counts[name], name))[:maximum]
    provenance = pd.DataFrame(
        [{"feature": name, "prior_model_uses": counts[name]} for name in ordered]
    )
    return ordered, provenance


def _feature_set(
    report: pd.DataFrame,
    available: set[str],
    fold_start: pd.Timestamp,
    maximum: int,
) -> tuple[list[str], str, pd.DataFrame]:
    selected, provenance = _prior_rule_features(report, available, fold_start, maximum)
    if len(selected) >= min(8, maximum):
        return selected, "prior_rulefit_brl_union", provenance
    fallback = [name for name in FALLBACK_MARKET_FEATURES if name in available]
    merged = list(dict.fromkeys([*selected, *fallback]))[:maximum]
    if not provenance.empty:
        known = set(provenance["feature"])
        extra = pd.DataFrame(
            [{"feature": name, "prior_model_uses": 0} for name in merged if name not in known]
        )
        provenance = pd.concat([provenance, extra], ignore_index=True, copy=False)
    else:
        provenance = pd.DataFrame(
            [{"feature": name, "prior_model_uses": 0} for name in merged]
        )
    return merged, "fallback_plus_prior_rulefit_brl_union", provenance


def _raw_change_scores(train: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    full = np.concatenate([train, score])
    delta = np.abs(np.diff(full, axis=0, prepend=full[:1])).astype(np.float32)
    return delta[: len(train)], delta[len(train) :]


def _build_method_scores(
    train: np.ndarray,
    score: np.ndarray,
    expected_run_hours: int,
    max_run_hours: int,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    """Return train/OOS per-feature score matrices for three fixed methods."""

    feature_count = train.shape[1]
    scaled_train = np.empty_like(train, dtype=np.float32)
    scaled_score = np.empty_like(score, dtype=np.float32)
    cp_fast_train = np.empty_like(train, dtype=np.float32)
    cp_fast_score = np.empty_like(score, dtype=np.float32)
    cp_slow_train = np.empty_like(train, dtype=np.float32)
    cp_slow_score = np.empty_like(score, dtype=np.float32)
    reference_rows: list[dict[str, float]] = []
    for column in range(feature_count):
        local_train, local_score, median, iqr = robust_scale_train_oos(train[:, column], score[:, column])
        scaled_train[:, column], scaled_score[:, column] = local_train, local_score
        full = np.concatenate([local_train, local_score])
        fast = bocpd_student_t(
            full,
            BOCPDConfig(expected_run_hours=expected_run_hours, max_run_hours=max_run_hours),
        )
        slow = bocpd_student_t(
            full,
            BOCPDConfig(expected_run_hours=expected_run_hours * 3, max_run_hours=max_run_hours),
        )
        cp_fast_train[:, column], cp_fast_score[:, column] = fast[: len(train)], fast[len(train) :]
        cp_slow_train[:, column], cp_slow_score[:, column] = slow[: len(train)], slow[len(train) :]
        reference_rows.append({"feature_index": column, "train_median": float(median), "train_iqr": float(iqr)})
    raw_train, raw_score = _raw_change_scores(scaled_train, scaled_score)
    return {
        "raw_abs_change": (raw_train, raw_score),
        f"bocpd_h{expected_run_hours}": (cp_fast_train, cp_fast_score),
        f"bocpd_h{expected_run_hours * 3}": (cp_slow_train, cp_slow_score),
    }, pd.DataFrame(reference_rows)


def _alert_metrics(
    timestamps: pd.Series,
    alerts: np.ndarray,
    score: np.ndarray,
    events: pd.DataFrame,
    *,
    warning_hours: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    stamps = pd.to_datetime(timestamps, utc=True).reset_index(drop=True)
    alert = np.asarray(alerts, dtype=bool)
    values = np.asarray(score, dtype=np.float32)
    normal = np.ones(len(stamps), dtype=bool)
    rows: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        start = _timestamp(event.event_start)
        end = _timestamp(event.event_end)
        warning_start = start - pd.Timedelta(hours=warning_hours)
        window = stamps.ge(warning_start) & stamps.lt(start)
        active = stamps.ge(start) & stamps.le(end)
        normal &= ~(window.to_numpy() | active.to_numpy())
        selected = np.flatnonzero(window.to_numpy() & alert)
        rows.append({
            "event_start": start,
            "event_end": end,
            "side_name": event.side_name,
            "archetype_policy_key": event.archetype_policy_key,
            "event_block": event.event_block,
            "onset_primary_mechanism": event.onset_primary_mechanism,
            "detected": bool(len(selected)),
            "lead_hours": float((start - stamps.iloc[selected[0]]).total_seconds() / 3600.0) if len(selected) else np.nan,
            "max_window_score": float(np.nanmax(values[window.to_numpy()])) if window.any() else np.nan,
        })
    event_rows = pd.DataFrame(rows)
    normal_alerts = alert & normal
    alert_starts = alert & np.r_[True, ~alert[:-1]]
    normal_alert_starts = alert_starts & normal
    elapsed_days = len(stamps) / 24.0
    summary: dict[str, float | int] = {
        "event_cells": int(len(event_rows)),
        "recognized_event_cells": int(event_rows["detected"].sum()) if not event_rows.empty else 0,
        "event_cell_recall": float(event_rows["detected"].mean()) if not event_rows.empty else np.nan,
        "median_lead_hours": float(event_rows.loc[event_rows["detected"], "lead_hours"].median()) if not event_rows.empty and event_rows["detected"].any() else np.nan,
        "normal_hours": int(normal.sum()),
        "false_positive_rate": float(normal_alerts.sum() / max(normal.sum(), 1)),
        "false_alert_hours_per_30d": float(normal_alerts.sum() / max(elapsed_days / 30.0, 1e-6)),
        "false_alert_episodes_per_30d": float(normal_alert_starts.sum() / max(elapsed_days / 30.0, 1e-6)),
        "alert_rate": float(alert.mean()),
    }
    return event_rows, summary


def _evaluate_method(
    method: str,
    train_matrix: np.ndarray,
    score_matrix: np.ndarray,
    timestamps: pd.Series,
    events: pd.DataFrame,
    feature_names: list[str],
    *,
    warning_hours: int,
    tail: float,
    min_simultaneous_breaks: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str], np.ndarray, np.ndarray, np.ndarray]:
    burn = min(max(24, train_matrix.shape[0] // 20), train_matrix.shape[0] - 1)
    reference = train_matrix[burn:] if train_matrix.shape[0] > burn + 16 else train_matrix
    synchronized, count, feature_thresholds = synchronized_break_score(reference, score_matrix, individual_tail=tail)
    train_sync, _, _ = synchronized_break_score(reference, reference, individual_tail=tail)
    threshold = float(np.quantile(train_sync, tail)) if np.nanstd(train_sync) > 1e-8 else np.nan
    alerts = (
        (synchronized > threshold) & (count >= min_simultaneous_breaks)
        if np.isfinite(threshold)
        else np.zeros(len(synchronized), dtype=bool)
    )
    event_rows, summary = _alert_metrics(timestamps, alerts, synchronized, events, warning_hours=warning_hours)
    status = "ok" if np.isfinite(threshold) and np.nanstd(synchronized) > 1e-8 else "degenerate_score"
    summary.update({
        "method": method,
        "status": status,
        "feature_count": int(len(feature_names)),
        "tail": float(tail),
        "alert_threshold": threshold,
        "mean_simultaneous_break_count": float(np.mean(count)),
        "max_simultaneous_break_count": int(np.max(count)) if len(count) else 0,
        "min_simultaneous_breaks": int(min_simultaneous_breaks),
    })
    feature_rows = pd.DataFrame({
        "feature": feature_names,
        "individual_alert_threshold": feature_thresholds,
        "oos_mean_change_probability": np.nanmean(score_matrix, axis=0),
        "oos_max_change_probability": np.nanmax(score_matrix, axis=0),
    })
    return event_rows, feature_rows, summary, synchronized, alerts, count


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output.mkdir(parents=True, exist_ok=True)
    state = _load_market_state(args.market_state)
    available = set(state.columns)
    report = pd.read_csv(args.feature_report)
    calendar = pd.read_csv(args.calendar)
    calendar["event_start"] = pd.to_datetime(calendar["event_start"], utc=True)
    calendar["event_end"] = pd.to_datetime(calendar["event_end"], utc=True)
    folds = _folds(args.fold or list(DEFAULT_FOLDS))
    fold_metrics: list[dict[str, object]] = []
    event_outputs: list[pd.DataFrame] = []
    feature_outputs: list[pd.DataFrame] = []
    score_outputs: list[pd.DataFrame] = []
    selection_outputs: list[pd.DataFrame] = []

    for fold, (start, end) in enumerate(folds):
        train = state.loc[state["__ts__"].lt(start)].copy()
        score = state.loc[state["__ts__"].ge(start) & state["__ts__"].lt(end)].copy()
        events = calendar.loc[
            calendar["event_start"].ge(start) & calendar["event_start"].lt(end)
        ].copy()
        if len(train) < args.min_train_hours or score.empty or events.empty:
            continue
        features, source, provenance = _feature_set(report, available, start, args.max_features)
        if len(features) < 4:
            continue
        provenance["fold"] = fold
        provenance["oos_start"] = start
        provenance["feature_source"] = source
        selection_outputs.append(provenance)
        train_values = train[features].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32, copy=True)
        score_values = score[features].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32, copy=True)
        minimum_finite = max(64, int(len(train_values) * 0.10))
        valid = (
            (np.isfinite(train_values).sum(axis=0) >= minimum_finite)
            & (np.isfinite(score_values).sum(axis=0) >= 24)
        )
        dropped = [name for name, keep in zip(features, valid, strict=True) if not keep]
        if dropped:
            provenance.loc[provenance["feature"].isin(dropped), "dropped_reason"] = "insufficient_fold_coverage"
        features = [name for name, keep in zip(features, valid, strict=True) if keep]
        train_values = train_values[:, valid]
        score_values = score_values[:, valid]
        if len(features) < 4:
            continue
        methods, scaling = _build_method_scores(
            train_values,
            score_values,
            expected_run_hours=args.expected_run_hours,
            max_run_hours=args.max_run_hours,
        )
        scaling["fold"] = fold
        scaling["oos_start"] = start
        scaling["feature"] = features
        for method, (train_matrix, score_matrix) in methods.items():
            for min_breaks in (args.min_simultaneous_breaks or [1, 2, 3]):
                method_id = f"{method}_sync{min_breaks}"
                event_rows, feature_rows, summary, sync_score, alerts, count = _evaluate_method(
                    method_id,
                    train_matrix,
                    score_matrix,
                    score["__ts__"],
                    events,
                    features,
                    warning_hours=args.warning_hours,
                    tail=args.tail,
                    min_simultaneous_breaks=min_breaks,
                )
                summary.update({"fold": fold, "oos_start": start, "oos_end": end, "feature_source": source})
                fold_metrics.append(summary)
                event_rows["fold"] = fold
                event_rows["method"] = method_id
                event_rows["detector_score"] = event_rows["max_window_score"]
                event_outputs.append(event_rows)
                feature_rows["fold"] = fold
                feature_rows["method"] = method_id
                feature_outputs.append(feature_rows)
                score_outputs.append(pd.DataFrame({
                    "__ts__": score["__ts__"].to_numpy(),
                    "fold": fold,
                    "method": method_id,
                    "synchronized_break_score": sync_score,
                    "simultaneous_break_count": count,
                    "simultaneous_break_alert": alerts.astype(np.int8),
                }))

    metrics = pd.DataFrame(fold_metrics)
    events = pd.concat(event_outputs, ignore_index=True, copy=False) if event_outputs else pd.DataFrame()
    feature_metrics = pd.concat(feature_outputs, ignore_index=True, copy=False) if feature_outputs else pd.DataFrame()
    scores = pd.concat(score_outputs, ignore_index=True, copy=False) if score_outputs else pd.DataFrame()
    selections = pd.concat(selection_outputs, ignore_index=True, copy=False) if selection_outputs else pd.DataFrame()
    metrics.to_csv(args.output / "fold_metrics.csv", index=False)
    events.to_csv(args.output / "event_cell_metrics.csv", index=False)
    feature_metrics.to_csv(args.output / "feature_changepoint_metrics.csv", index=False)
    selections.to_csv(args.output / "feature_selection_provenance.csv", index=False)
    # CSV keeps the diagnostic portable in environments where Arrow's CPU
    # introspection is unavailable.  These are research scores, not a live
    # inference artifact, and are intentionally not wired into policy.
    scores.to_csv(args.output / "hourly_changepoint_scores.csv.gz", index=False, compression="gzip")
    summary = {
        "purpose": "research-only causal Bayesian changepoint assessment for simultaneous market-dynamics breaks",
        "resolution": "1h",
        "outcome_inputs_to_detector": False,
        "feature_selection": "prior RuleFit/BRL/contrastive selections strictly before each OOS fold; observable fallback only for first fold",
        "methods": ["raw_abs_change", f"bocpd_h{args.expected_run_hours}", f"bocpd_h{args.expected_run_hours * 3}"],
        "coordination_ablation": args.min_simultaneous_breaks or [1, 2, 3],
        "synchronization": "count plus mean excess of per-feature train-tail changepoint probabilities",
        "tail": args.tail,
        "warning_hours": args.warning_hours,
        "folds": [f"{start.isoformat()}::{end.isoformat()}" for start, end in folds],
        "no_policy_wiring": True,
        "score_format": "gzip_csv",
        "promotion_rule": "Require chronology-stable event recall, FPR <= 15%, and incremental event recognition beyond raw-change baseline before any overlay test.",
    }
    (args.output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--feature-report", type=Path, default=DEFAULT_FEATURE_REPORT)
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fold", action="append", default=[])
    parser.add_argument("--max-features", type=int, default=18)
    parser.add_argument("--expected-run-hours", type=int, default=48)
    parser.add_argument("--max-run-hours", type=int, default=96)
    parser.add_argument("--warning-hours", type=int, default=48)
    parser.add_argument("--tail", type=float, default=0.95)
    parser.add_argument("--min-simultaneous-breaks", type=int, action="append", default=[])
    parser.add_argument("--min-train-hours", type=int, default=24 * 60)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
