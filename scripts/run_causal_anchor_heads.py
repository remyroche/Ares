#!/usr/bin/env python3
"""Strict-OOS Anchor Discovery source heads and 2026 entry snapshots.

The first stage answers forward-path questions about each causal anchor rather
than trading PnL.  Model and feature-hierarchy selection is restricted to
walk-forward 2025 source labels.  June--August 2026 are confirmation-only;
the outputs are optional MC1 inputs and never alter candidate identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import brier_score_loss, mean_absolute_error, mutual_info_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.causal_anchor_geometry import (
    ANCHOR_BASE_FEATURES,
    ANCHOR_FEATURE_GROUPS,
    ANCHOR_KALMAN_STATE_FEATURES,
    ANCHOR_KALMAN_TRANSITION_FEATURES,
    ANCHOR_MARKET_FEATURES,
    ANCHOR_TRANSITION_FEATURES,
)


SOURCE = ROOT / "data_perp/artifacts/causal_anchor_geometry_2025_train_2026_score_20260831_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/causal_anchor_heads_2025select_2026confirm_20260831_v1"
SEED = 1729
MAX_TRAIN_ROWS = 300_000
SELECTION_MONTHS = tuple(pd.date_range("2025-07-01", "2025-12-01", freq="MS", tz="UTC"))
CONFIRMATION_MONTHS = tuple(pd.date_range("2026-06-01", "2026-08-01", freq="MS", tz="UTC"))
HEADS = (
    "anchor_expected_utility_atr",
    "anchor_revisit_probability",
    "anchor_rejection_probability",
    "anchor_accepted_cross_probability",
    "anchor_continuation_probability",
)
ENTRY_HEADS = (
    "anchor_bullish_expected_utility_atr",
    "anchor_bearish_expected_utility_atr",
    "anchor_bullish_rejection_probability",
    "anchor_bearish_rejection_probability",
    "anchor_bullish_continuation_probability",
    "anchor_bearish_continuation_probability",
    "anchor_bullish_accepted_cross_probability",
    "anchor_bearish_accepted_cross_probability",
    "anchor_long_structure_utility",
    "anchor_long_structure_continuation_balance",
)
AVAILABLE = "anchor_snapshot_available"
EVENT_TARGET_COLUMNS = (
    "y_anchor_utility_atr",
    "y_anchor_revisit",
    "y_anchor_rejection",
    "y_anchor_accepted_cross",
    "y_anchor_continuation",
)
EVENT_METADATA_COLUMNS = (
    "__symbol__",
    "event_ts",
    "anchor_id",
    "anchor_event_family",
    "anchor_price_type",
    "label_available_ts",
    "anchor_random_price_distance_atr",
    "anchor_rolling_vwap_distance_atr",
    "anchor_signed_distance_atr",
)
TRAIN_HASH_MODULUS = 20


def _event_columns() -> tuple[str, ...]:
    """The complete source-head panel, without unrelated event-table fields."""
    fields = set(EVENT_METADATA_COLUMNS) | set(EVENT_TARGET_COLUMNS)
    for group in ANCHOR_FEATURE_GROUPS.values():
        fields.update(group)
    return tuple(sorted(fields))


def _coerce_event_times(frame: pd.DataFrame) -> pd.DataFrame:
    frame["event_ts"] = pd.to_datetime(frame.event_ts, utc=True, errors="raise")
    frame["label_available_ts"] = pd.to_datetime(frame.label_available_ts, utc=True, errors="raise")
    return frame


def _read_event_period(
    event_path: Path,
    columns: tuple[str, ...],
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    label_before: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Read only one held source window from the wide event table.

    The source parquet is symbol-partitioned rather than globally ordered by
    event time.  Iterating row groups keeps the resident population bounded
    while preserving every row in the requested held window.
    """
    parts: list[pd.DataFrame] = []
    parquet = pq.ParquetFile(event_path)
    for batch in parquet.iter_batches(columns=list(columns), batch_size=200_000):
        work = _coerce_event_times(batch.to_pandas())
        mask = pd.Series(True, index=work.index)
        if start is not None:
            mask &= work.event_ts.ge(start)
        if end is not None:
            mask &= work.event_ts.lt(end)
        if label_before is not None:
            mask &= work.label_available_ts.lt(label_before)
        if mask.any():
            parts.append(work.loc[mask].copy())
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True)


def _build_2025_train_sample(event_path: Path, cache_path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    """Build a deterministic source-identity sample, never target-selected.

    We cap fitting at ``MAX_TRAIN_ROWS`` anyway.  Sampling each source event
    by its immutable identity before model fitting avoids materialising the
    full 9m-row panel simply to discard most rows in ``_bounded``.  The sample
    is uniform across event identities and is fixed before any label/model
    outcome is observed.
    """
    selected: list[pd.DataFrame] = []
    boundary = pd.Timestamp("2026-01-01", tz="UTC")
    parquet = pq.ParquetFile(event_path)
    for batch in parquet.iter_batches(columns=list(columns), batch_size=200_000):
        work = _coerce_event_times(batch.to_pandas())
        work = work.loc[work.event_ts.lt(boundary)].copy()
        if work.empty:
            continue
        identity = pd.util.hash_pandas_object(work.loc[:, ["__symbol__", "event_ts", "anchor_id"]], index=False).to_numpy(dtype="uint64")
        selected.append(work.loc[(identity % TRAIN_HASH_MODULUS) == 0].copy())
    if not selected:
        raise RuntimeError("deterministic 2025 anchor training sample is empty")
    sample = pd.concat(selected, ignore_index=True)
    sample.to_parquet(cache_path, index=False, compression="zstd")
    return sample


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _bounded(frame: pd.DataFrame, timestamp: str) -> pd.DataFrame:
    if len(frame) <= MAX_TRAIN_ROWS:
        return frame
    ordered = frame.sort_values(timestamp, kind="stable")
    return ordered.iloc[np.linspace(0, len(ordered) - 1, MAX_TRAIN_ROWS, dtype=np.int64)].copy()


def _matrix(train: pd.DataFrame, score: pd.DataFrame, features: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_x = train.loc[:, list(features)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    median = train_x.median(axis=0, numeric_only=True).fillna(0.0)
    return train_x.fillna(median), score.loc[:, list(features)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(median)


def _regressor() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression_l1", n_estimators=300, learning_rate=.03, max_depth=3, num_leaves=7,
        min_child_samples=220, subsample=.80, colsample_bytree=.85, reg_lambda=18.0,
        random_state=SEED, n_jobs=2, verbosity=-1,
    )


def _classifier() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=300, learning_rate=.03, max_depth=3, num_leaves=7,
        min_child_samples=220, subsample=.80, colsample_bytree=.85, reg_lambda=20.0,
        random_state=SEED, n_jobs=2, verbosity=-1,
    )


def _fit_score(train: pd.DataFrame, score: pd.DataFrame, features: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = _bounded(train, "event_ts")
    x_train, x_score = _matrix(train, score, features)
    utility = _regressor().fit(x_train, pd.to_numeric(train.y_anchor_utility_atr, errors="raise"))
    revisit = _classifier().fit(x_train, pd.to_numeric(train.y_anchor_revisit, errors="raise").astype(int))
    rejection = _classifier().fit(x_train, pd.to_numeric(train.y_anchor_rejection, errors="raise").astype(int))
    accepted = _classifier().fit(x_train, pd.to_numeric(train.y_anchor_accepted_cross, errors="raise").astype(int))
    continuation = _classifier().fit(x_train, pd.to_numeric(train.y_anchor_continuation, errors="raise").astype(int))
    out = score.copy()
    out["anchor_expected_utility_atr"] = utility.predict(x_score)
    out["anchor_revisit_probability"] = revisit.predict_proba(x_score)[:, 1]
    out["anchor_rejection_probability"] = rejection.predict_proba(x_score)[:, 1]
    out["anchor_accepted_cross_probability"] = accepted.predict_proba(x_score)[:, 1]
    out["anchor_continuation_probability"] = continuation.predict_proba(x_score)[:, 1]
    importance = pd.DataFrame({
        "feature": features,
        "utility_gain": utility.booster_.feature_importance(importance_type="gain"),
        "revisit_gain": revisit.booster_.feature_importance(importance_type="gain"),
        "rejection_gain": rejection.booster_.feature_importance(importance_type="gain"),
        "accepted_cross_gain": accepted.booster_.feature_importance(importance_type="gain"),
        "continuation_gain": continuation.booster_.feature_importance(importance_type="gain"),
    })
    return out, importance


def _metrics(frame: pd.DataFrame, *, period: str, variant: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    utility = pd.to_numeric(frame.y_anchor_utility_atr, errors="coerce")
    prediction = pd.to_numeric(frame.anchor_expected_utility_atr, errors="coerce")
    valid = utility.notna() & prediction.notna()
    rows.append({
        "period": period, "variant": variant, "head": "utility", "rows": int(valid.sum()),
        "mae": float(mean_absolute_error(utility[valid], prediction[valid])) if valid.any() else np.nan,
        "spearman": float(utility[valid].corr(prediction[valid], method="spearman")) if valid.sum() > 2 else np.nan,
    })
    for prediction_column, target_column, name in (
        ("anchor_revisit_probability", "y_anchor_revisit", "revisit"),
        ("anchor_rejection_probability", "y_anchor_rejection", "rejection"),
        ("anchor_accepted_cross_probability", "y_anchor_accepted_cross", "accepted_cross"),
        ("anchor_continuation_probability", "y_anchor_continuation", "continuation"),
    ):
        pred = pd.to_numeric(frame[prediction_column], errors="coerce")
        target = pd.to_numeric(frame[target_column], errors="coerce")
        valid = pred.notna() & target.notna()
        rows.append({
            "period": period, "variant": variant, "head": name, "rows": int(valid.sum()),
            "base_rate": float(target[valid].mean()) if valid.any() else np.nan,
            "auc": float(roc_auc_score(target[valid].astype(int), pred[valid])) if valid.sum() > 2 and target[valid].nunique() == 2 else np.nan,
            "brier": float(brier_score_loss(target[valid].astype(int), pred[valid])) if valid.any() else np.nan,
            "spearman": float(target[valid].corr(pred[valid], method="spearman")) if valid.sum() > 2 else np.nan,
        })
    return rows


def _quantile_codes(values: pd.Series, bins: int = 8) -> pd.Series:
    rank = pd.to_numeric(values, errors="coerce").rank(method="first", pct=True)
    return np.floor(rank * min(bins, max(2, int(rank.notna().sum() // 40)))).clip(upper=bins - 1).fillna(-1).astype(int)


def _conditional_mi(frame: pd.DataFrame, *, variant: str, period: str) -> pd.DataFrame:
    """Discrete CMI proxy inside market-model score bins, never used for fitting."""
    target = pd.to_numeric(frame.y_anchor_utility_atr, errors="coerce")
    # The condition is *always* the M0 market-only prediction.  Conditioning
    # on the current enriched model would hide the very incremental anchor
    # information this diagnostic is meant to measure.
    condition = _quantile_codes(frame["anchor_market_control_prediction"], bins=6)
    target_code = _quantile_codes(target, bins=8)
    output: list[dict[str, object]] = []
    anchors = tuple(field for field in ANCHOR_FEATURE_GROUPS[variant] if field not in ANCHOR_MARKET_FEATURES)
    for field in anchors:
        value_code = _quantile_codes(frame[field], bins=8)
        total = 0; score = 0.0
        for bucket in sorted(condition.unique()):
            mask = (condition == bucket) & (value_code >= 0) & (target_code >= 0)
            n = int(mask.sum())
            if n < 80:
                continue
            score += n * float(mutual_info_score(value_code[mask], target_code[mask]))
            total += n
        output.append({"period": period, "variant": variant, "feature": field, "rows": total, "conditional_mi": score / total if total else np.nan})
    return pd.DataFrame(output)


def _price_controls(frame: pd.DataFrame, *, period: str) -> pd.DataFrame:
    target = pd.to_numeric(frame.y_anchor_utility_atr, errors="coerce")
    rows: list[dict[str, object]] = []
    for (family, price_type), part in frame.groupby(["anchor_event_family", "anchor_price_type"], sort=True):
        values = pd.to_numeric(part.anchor_signed_distance_atr, errors="coerce")
        truth = pd.to_numeric(part.y_anchor_utility_atr, errors="coerce")
        valid = values.notna() & truth.notna()
        rows.append({
            "period": period, "anchor_event_family": family, "anchor_price_type": price_type,
            "rows": int(valid.sum()), "future_utility_atr": float(truth[valid].mean()) if valid.any() else np.nan,
            "price_location_ic": float(values[valid].corr(truth[valid], method="spearman")) if valid.sum() > 2 else np.nan,
        })
    for name in ("anchor_random_price_distance_atr", "anchor_rolling_vwap_distance_atr", "anchor_signed_distance_atr"):
        values = pd.to_numeric(frame[name], errors="coerce")
        valid = values.notna() & target.notna()
        rows.append({
            "period": period, "anchor_event_family": "__control__", "anchor_price_type": name,
            "rows": int(valid.sum()), "future_utility_atr": float(target[valid].mean()) if valid.any() else np.nan,
            "price_location_ic": float(values[valid].corr(target[valid], method="spearman")) if valid.sum() > 2 else np.nan,
        })
    return pd.DataFrame(rows)


def _age_decay(frame: pd.DataFrame, *, variant: str, period: str) -> pd.DataFrame:
    """Age-banded incremental utility evidence for an empirical half-life."""
    work = frame.loc[:, ["anchor_age_bars", "anchor_expected_utility_atr", "y_anchor_utility_atr"]].copy()
    work["anchor_age_bars"] = pd.to_numeric(work.anchor_age_bars, errors="coerce")
    work["prediction"] = pd.to_numeric(work.anchor_expected_utility_atr, errors="coerce")
    work["target"] = pd.to_numeric(work.y_anchor_utility_atr, errors="coerce")
    edges = [-.1, 4, 8, 16, 32, 64, 96, np.inf]
    labels = ("0-1h", "1-2h", "2-4h", "4-8h", "8-16h", "16-24h", "24h+")
    work["age_band"] = pd.cut(work.anchor_age_bars, bins=edges, labels=labels)
    rows: list[dict[str, object]] = []
    for age_band, part in work.groupby("age_band", observed=False, sort=True):
        valid = part.prediction.notna() & part.target.notna()
        rows.append({
            "period": period, "variant": variant, "age_band": str(age_band), "rows": int(valid.sum()),
            "head_ic": float(part.loc[valid, "prediction"].corr(part.loc[valid, "target"], method="spearman")) if valid.sum() > 2 else np.nan,
            "mean_future_utility_atr": float(part.loc[valid, "target"].mean()) if valid.any() else np.nan,
        })
    return pd.DataFrame(rows)


def _wide_snapshots(rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["candidate_id", "snapshot_ts", "__symbol__", "target_kind", "target_id"]
    wide = rows.pivot_table(index=keys, columns="anchor_role", values=HEADS, aggfunc="first").reset_index()
    wide.columns = ["_".join(part for part in value if part) if isinstance(value, tuple) else value for value in wide.columns]
    rename: dict[str, str] = {}
    for head in HEADS:
        for role in ("bullish", "bearish"):
            source = f"{head}_{role}"
            name = head.removeprefix("anchor_")
            rename[source] = f"anchor_{role}_{name}"
    wide = wide.rename(columns=rename)
    for field in ENTRY_HEADS:
        if field not in wide:
            wide[field] = np.nan
    wide["anchor_long_structure_utility"] = (
        pd.to_numeric(wide["anchor_bullish_expected_utility_atr"], errors="coerce")
        - pd.to_numeric(wide["anchor_bearish_expected_utility_atr"], errors="coerce")
    )
    wide["anchor_long_structure_continuation_balance"] = (
        pd.to_numeric(wide["anchor_bullish_continuation_probability"], errors="coerce")
        - pd.to_numeric(wide["anchor_bearish_continuation_probability"], errors="coerce")
    )
    wide[AVAILABLE] = wide.loc[:, list(ENTRY_HEADS)].notna().sum(axis=1).ge(4).astype("int8")
    return wide


def _selection_score(metrics: pd.DataFrame) -> float:
    head = metrics["head"]
    utility = metrics.loc[head.eq("utility"), "spearman"].mean()
    classifiers = metrics.loc[head.isin(("revisit", "rejection", "accepted_cross", "continuation")), "auc"].mean()
    return float(utility if np.isfinite(utility) else -1.0) + .25 * float((classifiers - .5) if np.isfinite(classifiers) else -1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    # ``append`` must not inherit a non-empty default: otherwise a focused
    # smoke invocation silently runs every default month *plus* the supplied
    # month (and duplicates it).  Defaults are applied only when the caller
    # supplies no explicit months.
    parser.add_argument("--selection-month", action="append", default=None)
    parser.add_argument("--confirmation-month", action="append", default=None)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "causal-anchor-geometry-v1":
        raise AssertionError("source is not the causal anchor geometry contract")
    event_path = source / "anchor_events.parquet"
    snapshots = pd.read_parquet(source / "anchor_snapshots.parquet")
    snapshots["snapshot_ts"] = pd.to_datetime(snapshots.snapshot_ts, utc=True, errors="raise")
    event_columns = _event_columns()
    required = {"y_anchor_utility_atr", "y_anchor_revisit", "y_anchor_rejection", "y_anchor_accepted_cross", "y_anchor_continuation", *ANCHOR_FEATURE_GROUPS["m4_kalman_transition"]}
    missing = sorted(required.difference(pq.ParquetFile(event_path).schema_arrow.names))
    if missing:
        raise AssertionError(f"anchor source lacks fields: {missing}")
    output.mkdir(parents=True, exist_ok=False)
    train_cache_path = output / "anchor_events_2025_identity_sample.parquet"
    train_cache = _build_2025_train_sample(event_path, train_cache_path, event_columns)
    if train_cache.label_available_ts.le(train_cache.event_ts).any():
        raise AssertionError("anchor labels resolve at or before their source timestamp")
    selected_month_args = args.selection_month or [item.strftime("%Y-%m") for item in SELECTION_MONTHS]
    confirmation_month_args = args.confirmation_month or [item.strftime("%Y-%m") for item in CONFIRMATION_MONTHS]
    selection_months = tuple(pd.Timestamp(f"{month}-01", tz="UTC") for month in selected_month_args)
    confirmation_months = tuple(pd.Timestamp(f"{month}-01", tz="UTC") for month in confirmation_month_args)
    selection_metrics: list[dict[str, object]] = []
    selection_cmi: list[pd.DataFrame] = []
    selection_age_decay: list[pd.DataFrame] = []
    control_rows: list[pd.DataFrame] = []
    for held in selection_months:
        end = held + pd.offsets.MonthBegin(1)
        train = train_cache.loc[train_cache.event_ts.lt(held) & train_cache.label_available_ts.lt(held)].copy()
        test = _read_event_period(event_path, event_columns, start=held, end=end, label_before=end)
        if len(train) < 5_000 or len(test) < 500:
            continue
        # Score M0 first, then stream each enriched variant.  Holding every
        # wide event frame at once is unnecessary and can turn this source
        # study into a memory-bound selection rather than a model comparison.
        # The market-control prediction is the only cross-variant value used
        # by CMI, so the selection contract is unchanged.
        market_scored, _ = _fit_score(train, test, ANCHOR_FEATURE_GROUPS["m0_market"])
        market = market_scored.loc[:, ["__symbol__", "event_ts", "anchor_id", "anchor_expected_utility_atr"]].rename(columns={"anchor_expected_utility_atr": "anchor_market_control_prediction"})
        for variant, fields in ANCHOR_FEATURE_GROUPS.items():
            scored = market_scored if variant == "m0_market" else _fit_score(train, test, fields)[0]
            scored = scored.merge(market, on=["__symbol__", "event_ts", "anchor_id"], how="left", validate="one_to_one")
            selection_metrics.extend(_metrics(scored, period=held.strftime("%Y-%m"), variant=variant))
            selection_cmi.append(_conditional_mi(scored, variant=variant, period=held.strftime("%Y-%m")))
            selection_age_decay.append(_age_decay(scored, variant=variant, period=held.strftime("%Y-%m")))
            del scored
        del market_scored, market
        control_rows.append(_price_controls(test, period=held.strftime("%Y-%m")))
    selected_metrics = pd.DataFrame(selection_metrics)
    if selected_metrics.empty:
        raise RuntimeError("insufficient 2025 source support for anchor selection")
    selection_summary = selected_metrics.groupby("variant", sort=True).apply(_selection_score, include_groups=False).rename("selection_score").reset_index()
    # M0 is a market-only negative control; it cannot be selected as an anchor
    # challenger even if the source heads prove no incremental anchor value.
    anchor_candidates = selection_summary.loc[~selection_summary.variant.eq("m0_market")].sort_values(["selection_score", "variant"], ascending=[False, True], kind="stable")
    selected_variant = str(anchor_candidates.iloc[0].variant)
    selected_metrics.to_parquet(output / "selection_head_metrics_2025.parquet", index=False, compression="zstd")
    selection_summary.to_parquet(output / "selection_summary_2025.parquet", index=False, compression="zstd")
    pd.concat(selection_cmi, ignore_index=True).to_parquet(output / "selection_conditional_information_2025.parquet", index=False, compression="zstd")
    pd.concat(selection_age_decay, ignore_index=True).to_parquet(output / "selection_age_decay_2025.parquet", index=False, compression="zstd")
    pd.concat(control_rows, ignore_index=True).to_parquet(output / "anchor_price_control_comparison_2025.parquet", index=False, compression="zstd")
    train = train_cache.loc[train_cache.label_available_ts.lt(pd.Timestamp("2026-01-01", tz="UTC"))].copy()
    if len(train) < 10_000:
        raise RuntimeError(f"insufficient all-2025 resolved anchor training support: {len(train)}")
    confirmation_metrics: list[dict[str, object]] = []
    confirmation_cmi: list[pd.DataFrame] = []
    confirmation_age_decay: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    fold_trace: list[dict[str, object]] = []
    # All predeclared variants are scored in 2026.  Only the 2025-selected
    # variant is eligible for a later MC1 evaluation.
    # Only the 2025-selected variant may advance to an MC1 input.  Keep its
    # compact predictions; stream all other confirmation diagnostics so their
    # raw wide feature matrices cannot accumulate in memory.
    selected_snapshots: list[pd.DataFrame] = []
    selected_events: list[pd.DataFrame] = []
    for held in confirmation_months:
        end = held + pd.offsets.MonthBegin(1)
        event_test = _read_event_period(event_path, event_columns, start=held, end=end, label_before=end)
        snapshot_test = snapshots.loc[snapshots.snapshot_ts.ge(held) & snapshots.snapshot_ts.lt(end)].copy()
        if event_test.empty or snapshot_test.empty:
            raise RuntimeError(f"missing 2026 confirmation anchor population for {held:%Y-%m}")
        market_event, market_importance = _fit_score(train, event_test, ANCHOR_FEATURE_GROUPS["m0_market"])
        market = market_event.loc[:, ["__symbol__", "event_ts", "anchor_id", "anchor_expected_utility_atr"]].rename(columns={"anchor_expected_utility_atr": "anchor_market_control_prediction"})
        for variant, fields in ANCHOR_FEATURE_GROUPS.items():
            if variant == "m0_market":
                scored_event, importance = market_event, market_importance
            else:
                scored_event, importance = _fit_score(train, event_test, fields)
            scored_snapshot, _ = _fit_score(train, snapshot_test, fields)
            scored_event = scored_event.merge(market, on=["__symbol__", "event_ts", "anchor_id"], how="left", validate="one_to_one")
            scored_event["held_month"] = held.strftime("%Y-%m")
            scored_snapshot["held_month"] = held.strftime("%Y-%m")
            confirmation_metrics.extend(_metrics(scored_event, period=held.strftime("%Y-%m"), variant=variant))
            confirmation_cmi.append(_conditional_mi(scored_event, variant=variant, period=held.strftime("%Y-%m")))
            confirmation_age_decay.append(_age_decay(scored_event, variant=variant, period=held.strftime("%Y-%m")))
            importance.insert(0, "held_month", held.strftime("%Y-%m")); importance.insert(0, "variant", variant); importance_frames.append(importance)
            if variant == selected_variant:
                selected_snapshots.append(scored_snapshot.loc[:, ["candidate_id", "snapshot_ts", "__symbol__", "target_kind", "target_id", "anchor_role", *HEADS, "held_month"]].copy())
                selected_events.append(scored_event.loc[:, ["__symbol__", "event_ts", "anchor_id", "anchor_event_family", "anchor_price_type", *HEADS, "held_month"]].copy())
            del scored_snapshot, scored_event
        del market_event, market_importance, market
        fold_trace.append({"held_month": held.strftime("%Y-%m"), "train_rows": int(len(train)), "train_label_max": str(train.label_available_ts.max()), "event_test_rows": int(len(event_test)), "snapshot_test_rows": int(len(snapshot_test))})
    wide = _wide_snapshots(pd.concat(selected_snapshots, ignore_index=True))
    wide.to_parquet(output / f"entry_anchor_{selected_variant}_oof_features.parquet", index=False, compression="zstd")
    pd.concat(selected_events, ignore_index=True).to_parquet(output / f"anchor_event_{selected_variant}_oof_predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(confirmation_metrics).to_parquet(output / "confirmation_head_metrics_2026.parquet", index=False, compression="zstd")
    pd.concat(confirmation_cmi, ignore_index=True).to_parquet(output / "confirmation_conditional_information_2026.parquet", index=False, compression="zstd")
    pd.concat(confirmation_age_decay, ignore_index=True).to_parquet(output / "confirmation_age_decay_2026.parquet", index=False, compression="zstd")
    pd.concat(importance_frames, ignore_index=True).to_parquet(output / "feature_importance_2026.parquet", index=False, compression="zstd")
    pd.DataFrame(fold_trace).to_parquet(output / "fold_trace.parquet", index=False, compression="zstd")
    output_manifest: dict[str, Any] = {
        "schema": "causal-anchor-heads-2025select-2026confirm-v1",
        "scope": "offline anchor-source research only; no live/canonical/policy/execution mutation",
        "source": str(source), "source_manifest_sha256": _sha256(source / "run_manifest.json"),
        "training_sample": {"file": train_cache_path.name, "rule": f"deterministic source-identity hash modulo {TRAIN_HASH_MODULUS}; fixed before fitting and independent of labels", "rows": int(len(train_cache))},
        "selection": {"period": [item.strftime("%Y-%m") for item in selection_months], "selected_variant": selected_variant, "selection_metric": "mean source utility Spearman + 0.25 * mean classifier AUC excess over 0.5; M0 excluded from promotion"},
        "confirmation": [item.strftime("%Y-%m") for item in confirmation_months],
        "features": {variant: list(fields) for variant, fields in ANCHOR_FEATURE_GROUPS.items()},
        "heads": {"utility": "next-8h away-MFE minus toward-MAE (ATR), LGBM L1 d3/l7", "classifiers": "next-8h revisit/rejection/accepted-cross/continuation, LGBM binary d3/l7"},
        "causality": "models select exclusively on 2025 resolved forward labels; 2026 confirmation is not used to select the variant; source snapshot fields are target-free; Kalman is a forward fixed-parameter filter only",
        "downstream_contract": {"selected_variant": selected_variant, "entry_file": f"entry_anchor_{selected_variant}_oof_features.parquet", "entry_heads": list(ENTRY_HEADS), "availability": AVAILABLE},
        "no_exchange_calls": True,
    }
    (output / "run_manifest.json").write_text(json.dumps(output_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
