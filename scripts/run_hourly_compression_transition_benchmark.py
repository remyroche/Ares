#!/usr/bin/env python3
"""Block-level, hourly-only benchmark for compression-to-expansion hazards.

The unit is an event block and its pre-defined 12-hour lead anchor, rather
than every correlated event hour.  A deterministic compression-state layer
limits a shallow hazard model to comparable low-volatility/low-activity
lookalikes.  All thresholds, matching scales, feature screens, fits and alert
cutoffs are learned only from the training portion of each chronological fold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, roc_auc_score

from extreme_price_movements.hourly_extreme_event_detection import (
    HourlyEventConfig,
    available_hourly_features,
    build_hourly_market_state,
)
from scripts.run_hourly_extreme_event_detector import (
    DEFAULT_FOLDS,
    DEFAULT_STATE,
    DEFAULT_TAXONOMY,
    _load_hourly_rows,
    _load_taxonomy,
    _screen_features,
    _timestamp,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data_perp/reports/hourly_compression_transition_benchmark_20260714_v1"
STATE_COLUMNS = ("mkt_rv_4h", "mkt_rv_ratio_1h_24h", "mkt_atr_expansion_1h", "mkt_volume_z_24h")
MATCH_COLUMNS = (
    "mkt_rv_4h", "mkt_atr_expansion_1h", "mkt_volume_z_24h",
    "mkt_oi_chg_4h", "mkt_funding_mean_z_30d", "mkt_ret_4h",
)


def _folds(values: list[str]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    result = []
    for value in values:
        start, separator, end = value.partition("::")
        if not separator:
            raise ValueError(f"Expected START::END, got {value!r}")
        result.append((_timestamp(start), _timestamp(end)))
    return result


def _continuous_duration(active: np.ndarray) -> np.ndarray:
    result = np.zeros(len(active), dtype=np.float32)
    count = 0
    for index, value in enumerate(active):
        count = count + 1 if value else 0
        result[index] = count
    return result


def _fit_state(train: pd.DataFrame, score: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Train-only compression definition from low RV/ATR/volume conditions."""

    available = [name for name in STATE_COLUMNS if name in train]
    if len(available) < 3:
        raise ValueError(f"Missing compression-state columns: {available}")
    thresholds = {
        name: float(pd.to_numeric(train[name], errors="coerce").quantile(0.35))
        for name in available
    }
    outputs: list[pd.DataFrame] = []
    for frame in (train, score):
        local = frame.copy()
        flags = np.column_stack([
            pd.to_numeric(local[name], errors="coerce").to_numpy(np.float32) <= thresholds[name]
            for name in available
        ])
        local["compression_state_score"] = flags.mean(axis=1).astype(np.float32)
        local["compression_state_active"] = (local["compression_state_score"] >= (2.0 / 3.0)).astype(np.int8)
        outputs.append(local)
    combined_active = np.concatenate([outputs[0]["compression_state_active"].to_numpy(bool), outputs[1]["compression_state_active"].to_numpy(bool)])
    duration = _continuous_duration(combined_active)
    outputs[0]["compression_duration_hours"] = duration[: len(outputs[0])]
    outputs[1]["compression_duration_hours"] = duration[len(outputs[0]) :]
    return outputs[0], outputs[1], thresholds


def _add_hazard_features(frame: pd.DataFrame) -> pd.DataFrame:
    local = frame.copy()
    oi = pd.to_numeric(local.get("mkt_oi_chg_4h"), errors="coerce").fillna(0.0).to_numpy(np.float32)
    volume = pd.to_numeric(local.get("mkt_volume_z_24h"), errors="coerce").fillna(0.0).to_numpy(np.float32)
    funding = pd.to_numeric(local.get("mkt_funding_mean_z_30d"), errors="coerce").fillna(0.0).to_numpy(np.float32)
    local["hazard_oi_volume_divergence"] = (oi - volume).astype(np.float32)
    local["hazard_oi_accumulation_duration"] = (np.maximum(oi, 0.0) * local["compression_duration_hours"].to_numpy(np.float32)).astype(np.float32)
    local["hazard_funding_persistence_12h"] = pd.Series(funding).rolling(12, min_periods=3).mean().to_numpy(np.float32)
    local["hazard_funding_acceleration"] = pd.Series(funding).diff().diff().to_numpy(np.float32)
    local["hazard_range_imbalance"] = (
        pd.to_numeric(local.get("mkt_ret_1h"), errors="coerce").fillna(0.0).abs().to_numpy(np.float32)
        / (pd.to_numeric(local.get("mkt_rv_4h"), errors="coerce").fillna(0.0).abs().to_numpy(np.float32) + 1e-4)
    ).astype(np.float32)
    return local


def _event_blocks(taxonomy: pd.DataFrame, lead_hours: int) -> pd.DataFrame:
    events = taxonomy.loc[taxonomy["onset_primary_mechanism"].eq("volatility_compression_transition")].copy()
    events["event_start"] = pd.to_datetime(events["event_start"], utc=True).dt.floor("h")
    events["event_end"] = pd.to_datetime(events["event_end"], utc=True).dt.floor("h")
    # One block per timestamp; side/archetype membership remains an annotation.
    return events.groupby("event_start", as_index=False).agg(
        event_end=("event_end", "max"),
        affected_cells=("event_block", "size"),
        archetypes=("archetype_policy_key", lambda values: "|".join(sorted(set(map(str, values))))),
    ).assign(anchor=lambda x: x["event_start"] - pd.Timedelta(hours=lead_hours))


def _exclude_event_neighborhood(frame: pd.DataFrame, events: pd.DataFrame, lead_hours: int) -> np.ndarray:
    allowed = np.ones(len(frame), dtype=bool)
    timestamp = frame["__ts__"]
    for event in events.itertuples(index=False):
        allowed &= ~timestamp.between(event.anchor - pd.Timedelta(hours=lead_hours), event.event_end + pd.Timedelta(days=1)).to_numpy()
    return allowed


def _nearest_controls(
    train: pd.DataFrame,
    events: pd.DataFrame,
    *,
    lead_hours: int,
    controls_per_event: int,
    seed: int,
) -> pd.DataFrame:
    """Select state-matched benign blocks, never generic normal hours."""

    positives = train.merge(events[["event_start", "anchor"]], left_on="__ts__", right_on="anchor", how="inner")
    candidates = train.loc[train["compression_state_active"].eq(1)].copy()
    candidates = candidates.loc[_exclude_event_neighborhood(candidates, events, lead_hours)]
    if positives.empty or candidates.empty:
        return pd.DataFrame(columns=train.columns)
    columns = [name for name in MATCH_COLUMNS if name in train]
    med = train[columns].median(numeric_only=True)
    scale = (train[columns].quantile(0.75) - train[columns].quantile(0.25)).clip(lower=1e-4)
    candidate_values = ((candidates[columns] - med) / scale).to_numpy(np.float32)
    chosen: list[int] = []
    rng = np.random.default_rng(seed)
    for event in positives.itertuples(index=False):
        row = ((pd.Series(event._asdict())[columns] - med) / scale).to_numpy(np.float32)
        distance = np.nanmean(np.square(candidate_values - row), axis=1)
        # Randomize only among the closest ten to test control sensitivity.
        nearest = np.argsort(np.nan_to_num(distance, nan=np.inf))[: min(10, len(candidates))]
        if len(nearest):
            picked = rng.choice(nearest, size=min(controls_per_event, len(nearest)), replace=False)
            chosen.extend(candidates.index[picked].tolist())
    return candidates.loc[sorted(set(chosen))].copy()


def _fit_scale(train: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(train, axis=0)
    q25 = np.nanquantile(train, 0.25, axis=0)
    q75 = np.nanquantile(train, 0.75, axis=0)
    scale = np.maximum(q75 - q25, 1e-4)
    median = np.nan_to_num(median, nan=0.0).astype(np.float32)
    for values in (train, score):
        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = np.take(median, np.nonzero(missing)[1])
        values -= median
        values /= scale
        np.clip(values, -8, 8, out=values)
    return train, score


def _fit_predict(train: pd.DataFrame, score: pd.DataFrame, features: list[str], seed: int) -> np.ndarray:
    x_train = train[features].to_numpy(np.float32, copy=True)
    x_score = score[features].to_numpy(np.float32, copy=True)
    x_train, x_score = _fit_scale(x_train, x_score)
    y = train["target"].to_numpy(np.int8)
    return np.asarray(lgb.train(
        {"objective": "binary", "metric": "None", "learning_rate": 0.035, "max_depth": 2,
         "num_leaves": 4, "min_data_in_leaf": max(4, min(16, len(train) // 10)),
         "min_gain_to_split": 0.05, "lambda_l1": 4.0, "lambda_l2": 20.0,
         "feature_fraction": 0.80, "bagging_fraction": 0.85, "bagging_freq": 1,
         "seed": seed, "num_threads": 1, "verbosity": -1, "force_col_wise": True},
        lgb.Dataset(x_train, label=y), num_boost_round=100,
    ).predict(x_score), dtype=np.float32)


def _event_metrics(events: pd.DataFrame, scored: pd.DataFrame, threshold: float, lead_hours: int) -> pd.DataFrame:
    rows = []
    for event in events.itertuples(index=False):
        window = scored.loc[scored["__ts__"].between(event.anchor, event.event_start)]
        # A calibration quantile can coincide with a broad tied score mass.  A
        # non-strict comparison would then turn a non-discriminating model into
        # an all-alert detector.  For a fixed-FPR research benchmark, ties at
        # the cutoff remain unselected rather than being promoted arbitrarily.
        selected = window.loc[window["hazard_score"].gt(threshold), "__ts__"]
        rows.append({
            "event_start": event.event_start, "affected_cells": event.affected_cells, "archetypes": event.archetypes,
            "state_active_at_anchor": bool(window["compression_state_active"].iloc[0]) if not window.empty else False,
            "detected": bool(len(selected)),
            "lead_hours": float((event.event_start - selected.min()).total_seconds() / 3600.0) if len(selected) else np.nan,
            "max_lead_score": float(window["hazard_score"].max()) if not window.empty else np.nan,
        })
    return pd.DataFrame(rows)


def _subtype_rows(train: pd.DataFrame, oos: pd.DataFrame, train_events: pd.DataFrame, oos_events: pd.DataFrame) -> pd.DataFrame:
    """Train-only descriptive clustering of compression-event lead anchors."""

    columns = [name for name in (*MATCH_COLUMNS, "compression_duration_hours", "hazard_oi_volume_divergence") if name in train]
    train_anchor = train.merge(train_events[["event_start", "anchor"]], left_on="__ts__", right_on="anchor", how="inner")
    oos_anchor = oos.merge(oos_events[["event_start", "anchor"]], left_on="__ts__", right_on="anchor", how="inner")
    if len(train_anchor) < 6 or not columns:
        return pd.DataFrame()
    x_train = train_anchor[columns].to_numpy(np.float32, copy=True)
    median = np.nanmedian(x_train, axis=0)
    scale = np.maximum(np.nanquantile(x_train, .75, axis=0) - np.nanquantile(x_train, .25, axis=0), 1e-4)
    for values in (x_train,):
        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = np.take(median, np.nonzero(missing)[1])
        values[:] = np.clip((values - median) / scale, -6, 6)
    clusters = min(3, max(2, len(train_anchor) // 4))
    model = KMeans(n_clusters=clusters, n_init=20, random_state=20260714).fit(x_train)
    rows = []
    for split, anchors in (("train", train_anchor), ("oos", oos_anchor)):
        if anchors.empty:
            continue
        values = anchors[columns].to_numpy(np.float32, copy=True)
        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = np.take(median, np.nonzero(missing)[1])
        values[:] = np.clip((values - median) / scale, -6, 6)
        assigned = model.predict(values)
        rows.append(pd.DataFrame({
            "sample": split,
            "event_start": anchors["event_start"].to_numpy(),
            "compression_subtype": assigned.astype(np.int8),
        }))
    return pd.concat(rows, ignore_index=True, copy=False) if rows else pd.DataFrame()


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    config = HourlyEventConfig(lead_hours=args.lead_hours, embargo_hours=args.embargo_hours)
    schema = pq.ParquetFile(args.state_artifact).schema.names
    observable = available_hourly_features(schema)
    raw = _load_hourly_rows(args.state_artifact, observable)
    hourly = build_hourly_market_state(raw, feature_columns=observable, config=config)
    taxonomy = _load_taxonomy(args.taxonomy)
    events = _event_blocks(taxonomy, args.lead_hours)
    base_features = [*hourly.attrs["observable_features"], *hourly.attrs["transition_features"]]
    folds = args.fold or [f"{start}::{end}" for start, end in DEFAULT_FOLDS]
    reports: list[dict[str, object]] = []
    all_events: list[pd.DataFrame] = []
    subtype_reports: list[pd.DataFrame] = []
    for fold, (start, end) in enumerate(_folds(folds)):
        train_raw = hourly.loc[hourly["__ts__"].lt(start - pd.Timedelta(hours=args.embargo_hours))].copy()
        oos_raw = hourly.loc[hourly["__ts__"].ge(start) & hourly["__ts__"].lt(end)].copy()
        train, oos, thresholds = _fit_state(train_raw, oos_raw)
        train, oos = _add_hazard_features(train), _add_hazard_features(oos)
        train_events = events.loc[events["anchor"].lt(start - pd.Timedelta(hours=args.embargo_hours))].copy()
        oos_events = events.loc[events["anchor"].ge(start) & events["anchor"].lt(end)].copy()
        positives = train.merge(train_events[["anchor"]], left_on="__ts__", right_on="anchor", how="inner").drop(columns="anchor")
        if positives.empty or oos_events.empty:
            continue
        split = int(len(train) * 0.75)
        core = train.iloc[:split].copy()
        calibration = train.iloc[split:].copy()
        core_events = train_events.loc[train_events["anchor"].lt(core["__ts__"].max())]
        subtype = _subtype_rows(core, oos, core_events, oos_events)
        if not subtype.empty:
            subtype["fold"] = fold
            subtype_reports.append(subtype)
        controls = _nearest_controls(core, core_events, lead_hours=args.lead_hours, controls_per_event=args.controls_per_event, seed=args.seed + fold)
        core_pos = core.merge(core_events[["anchor"]], left_on="__ts__", right_on="anchor", how="inner").drop(columns="anchor")
        sample = pd.concat([core_pos.assign(target=1), controls.assign(target=0)], ignore_index=True, copy=False)
        candidate_features = [*base_features, "compression_state_score", "compression_duration_hours", "hazard_oi_volume_divergence", "hazard_oi_accumulation_duration", "hazard_funding_persistence_12h", "hazard_funding_acceleration", "hazard_range_imbalance"]
        candidate_features = [name for name in candidate_features if name in sample]
        selected = _screen_features(sample, candidate_features, "target", args.max_features)
        if len(sample) < 12 or sample["target"].sum() < 3 or not selected:
            continue
        calibration_candidates = calibration.loc[calibration["compression_state_active"].eq(1)].copy()
        oos_candidates = oos.loc[oos["compression_state_active"].eq(1)].copy()
        if calibration_candidates.empty or oos_candidates.empty:
            continue
        matrices = []
        calibration_matrix = []
        for seed in range(args.seed, args.seed + args.seeds):
            calibration_matrix.append(_fit_predict(sample, calibration_candidates, selected, seed))
            matrices.append(_fit_predict(sample, oos_candidates, selected, seed))
        calibration_score = np.mean(np.vstack(calibration_matrix), axis=0)
        threshold = float(np.quantile(calibration_score, 1.0 - args.fixed_fpr))
        oos_candidates = oos_candidates.copy()
        oos_candidates["hazard_score"] = np.mean(np.vstack(matrices), axis=0)
        oos_scored = oos.copy()
        oos_scored["hazard_score"] = 0.0
        oos_scored.loc[oos_candidates.index, "hazard_score"] = oos_candidates["hazard_score"]
        event_rows = _event_metrics(oos_events, oos_scored, threshold, args.lead_hours)
        event_rows["fold"] = fold
        all_events.append(event_rows)
        calibration_alert_rate = float((calibration_score > threshold).mean())
        non_event_alerts = int((oos_candidates["hazard_score"] > threshold).sum())
        calibration_score_std = float(np.nanstd(calibration_score))
        oos_score_std = float(np.nanstd(oos_candidates["hazard_score"]))
        status = (
            "non_discriminating_calibration"
            # Float32 LightGBM constants can retain numerical noise around
            # 1e-8; that is still a flat score for this decision problem.
            if calibration_score_std < 1e-6
            else "ok"
        )
        reports.append({
            "fold": fold, "oos_start": start, "oos_end": end, "train_events": len(core_events),
            "oos_events": len(oos_events), "state_active_rate": float(oos["compression_state_active"].mean()),
            "state_event_recall": float(event_rows["state_active_at_anchor"].mean()),
            "event_detection_rate": float(event_rows["detected"].mean()),
            "median_lead_hours": float(event_rows.loc[event_rows["detected"], "lead_hours"].median()),
            "false_alerts_per_30d": non_event_alerts / max(len(oos) / (24 * 30), 1e-6),
            "calibration_alert_rate": calibration_alert_rate,
            "calibration_score_std": calibration_score_std,
            "oos_score_std": oos_score_std,
            "status": status,
            "alert_threshold": threshold, "features": "|".join(selected),
            "compression_state_thresholds": json.dumps(thresholds, sort_keys=True),
        })
    pd.DataFrame(reports).to_csv(args.output / "compression_block_fold_metrics.csv", index=False)
    block_output = pd.concat(all_events, ignore_index=True, copy=False) if all_events else pd.DataFrame()
    block_output.to_csv(args.output / "compression_block_event_metrics.csv", index=False)
    pd.concat(subtype_reports, ignore_index=True, copy=False).to_csv(
        args.output / "compression_block_subtypes.csv", index=False
    ) if subtype_reports else pd.DataFrame().to_csv(args.output / "compression_block_subtypes.csv", index=False)
    (args.output / "manifest.json").write_text(json.dumps({
        "purpose": "research-only compression state plus transition-hazard benchmark",
        "resolution": "1h only", "subhour_data_used": False,
        "unit": "one event lead block plus matched benign compression blocks",
        "validation": "chronological folds, 36h embargo; event-block metrics",
        "no_policy_wiring": True,
    }, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-artifact", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fold", action="append", default=[])
    parser.add_argument("--lead-hours", type=int, default=12)
    parser.add_argument("--embargo-hours", type=int, default=36)
    parser.add_argument("--controls-per-event", type=int, default=4)
    parser.add_argument("--fixed-fpr", type=float, default=0.05)
    parser.add_argument("--max-features", type=int, default=14)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
