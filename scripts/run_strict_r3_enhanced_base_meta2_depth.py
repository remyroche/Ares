#!/usr/bin/env python3
"""Strict-OOS Stage-G residual-depth ablation for the enhanced-base stack.

The Stage-C P2/T1 score receipts are the immutable first-layer ledger.  This
script never reuses a first-layer prediction made on a row that trained it:
the Meta-2 target is formed from a daily prequential policy map of those
already-OOS P2 scores.  It then evaluates a small, sequential set of
second-layer authorities while preserving the frozen MC1 class, dual +30-bps
admission, BCF mapped-EV auction priority and portfolio replay.

Research only.  It has no live/exchange path.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.isotonic import IsotonicRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_strict_r3_enhanced_base_live_stack_challenger as parent


SCHEMA = "strict_r3_enhanced_base_meta2_depth_v1"
SEED = 1729
ANCHOR_HISTORY_DAYS = 90
ANCHOR_BINS = 20
ANCHOR_MIN_SUPPORT = 1_000
META2_TRAIN_MONTHS = 6
META2_RESERVE_DAYS = 28
META2_TRAIN_CAP = 180_000
META2_CORRECTION_LAMBDA = 0.25
META2_CORRECTION_CAP_BPS = 100.0
META2_ADVERSE_BPS = -100.0
META2_ADVERSE_DEMOTION_CAP_BPS = 50.0
META2_FIRST_HELD_MONTH = pd.Timestamp("2025-10-01T00:00:00Z")

ARMS = ("m0_first_layer_control", "m2_identical_residual", "m3_trust_residual", "m4_adverse_tail")


@dataclass(frozen=True)
class Meta2Bundle:
    arm: str
    family: str
    fields: tuple[str, ...]
    medians: np.ndarray
    model: object
    train_rows: int
    residual_mean_bps: float
    residual_std_bps: float


def _month_end(month: pd.Timestamp) -> pd.Timestamp:
    return month + pd.offsets.MonthBegin(1)


def _score_files(root: Path, family: str) -> list[Path]:
    paths = sorted((root / "target_free_scores" / family).glob("month=*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no {family} P2 target-free score receipts under {root}")
    return paths


def _target_free_files(root: Path) -> list[Path]:
    paths = sorted(root.glob("month=*/scores_features.parquet"))
    if not paths:
        raise FileNotFoundError(f"no target-free source receipts under {root}")
    return paths


def _read_score_ledger(score_root: Path, feature_root: Path) -> tuple[pd.DataFrame, list[str], list[Path]]:
    """Read the thin causal ledger without materialising all 120 raw fields.

    The wide raw contract is joined one calendar month at a time only after
    anchors/state features have been created.  This keeps the Stage-G source
    builder bounded even when it spans thirteen months of candidates.
    """

    current = pd.concat([pd.read_parquet(path) for path in _score_files(score_root, "current")], ignore_index=True)
    bcf = pd.concat([
        pd.read_parquet(path, columns=["candidate_id", "__decision_ts__", "final_score"])
        for path in _score_files(score_root, "bcf")
    ], ignore_index=True).rename(columns={"final_score": "stage1_bcf_score"})
    current = current.rename(columns={"final_score": "stage1_current_score"})
    if current["candidate_id"].duplicated().any() or bcf["candidate_id"].duplicated().any():
        raise AssertionError("first-layer score ledger has duplicate candidate identities")
    frame = current.merge(bcf, on=["candidate_id", "__decision_ts__"], how="inner", validate="one_to_one")
    feature_paths = _target_free_files(feature_root)
    first_columns = pd.read_parquet(feature_paths[0]).columns.tolist()
    identity = {
        "candidate_id", "__decision_ts__", "base_bps", "efficiency_bps", "timing_bps",
        "enhanced_base_bps", "base_rank_ts", "enhanced_base_routed", "e_minus_t",
        "e_minus_b0", "t_minus_b0", "base_component_std", "side_name",
    }
    raw_fields = [column for column in first_columns if column not in identity]
    feature_columns = ["candidate_id", "__decision_ts__", "base_bps", "efficiency_bps", "timing_bps"]
    features = pd.concat([
        pd.read_parquet(path, columns=feature_columns) for path in feature_paths
    ], ignore_index=True)
    if features["candidate_id"].duplicated().any():
        raise AssertionError("target-free base source has duplicate candidate identities")
    frame = frame.merge(features, on=["candidate_id", "__decision_ts__"], how="left", validate="one_to_one")
    if frame.loc[:, ["base_bps", "efficiency_bps", "timing_bps"]].isna().all(axis=1).any():
        raise AssertionError("P2 score receipt lacks its frozen target-free component row")
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    return frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True), raw_fields, feature_paths


def _policy_labels(path: Path) -> pd.DataFrame:
    fields = [
        "candidate_id", "policy_path_valid", "policy_label_available_ts", "policy_net_bps",
        "policy_gross_bps", "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price",
        "policy_exit_reason", "policy_cost_bps",
    ]
    labels = pd.read_parquet(path, columns=fields)
    labels["policy_label_available_ts"] = pd.to_datetime(labels["policy_label_available_ts"], utc=True, errors="coerce")
    if labels["candidate_id"].duplicated().any():
        raise AssertionError("policy labels duplicate candidate identities")
    return labels


def _score_bin(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    return np.clip(np.floor(np.nan_to_num(numeric, nan=0.0) * ANCHOR_BINS).astype(int), 0, ANCHOR_BINS - 1)


def _calendar_days(frame: pd.DataFrame) -> pd.DatetimeIndex:
    start = frame["__decision_ts__"].min().normalize()
    end = frame["__decision_ts__"].max().normalize()
    return pd.date_range(start, end, freq="D", tz="UTC")


def _prequential_daily_anchor(
    frame: pd.DataFrame,
    *,
    score_field: str,
    prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map first-layer score to bps using only earlier resolved P2 outcomes.

    A map is frozen once per UTC day.  This is intentionally conservative:
    labels resolving during the current day are not used until the next day.
    The daily contract makes the target available at every intraday decision
    without relying on later same-day outcomes.
    """

    work = frame.loc[:, ["candidate_id", "__decision_ts__", score_field, "policy_path_valid", "policy_label_available_ts", "policy_net_bps"]].copy()
    work["__day__"] = work["__decision_ts__"].dt.normalize()
    score = pd.to_numeric(work[score_field], errors="coerce")
    policy = pd.to_numeric(work["policy_net_bps"], errors="coerce")
    valid = (
        work["policy_path_valid"].fillna(False).astype(bool)
        & work["policy_label_available_ts"].notna()
        & np.isfinite(score.to_numpy(float))
        & np.isfinite(policy.to_numpy(float))
    )
    labelled = work.loc[valid].copy()
    labelled["__eligible_day__"] = labelled["policy_label_available_ts"].dt.normalize() + pd.Timedelta(days=1)
    labelled["__expire_day__"] = labelled["__day__"] + pd.Timedelta(days=ANCHOR_HISTORY_DAYS + 1)
    labelled["__bin__"] = _score_bin(labelled[score_field])
    labelled["__policy__"] = policy.loc[labelled.index].clip(-500.0, 500.0).to_numpy(float)
    additions: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]] = {}
    removals: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]] = {}
    for key, group in labelled.groupby("__eligible_day__", sort=True):
        bins = group["__bin__"].to_numpy(int)
        values = group["__policy__"].to_numpy(float)
        additions[pd.Timestamp(key)] = (bins, values)
    for key, group in labelled.loc[labelled["__eligible_day__"].le(labelled["__expire_day__"])].groupby("__expire_day__", sort=True):
        bins = group["__bin__"].to_numpy(int)
        values = group["__policy__"].to_numpy(float)
        removals[pd.Timestamp(key)] = (bins, values)

    count = np.zeros(ANCHOR_BINS, dtype=float)
    total = np.zeros(ANCHOR_BINS, dtype=float)
    day_anchor: dict[pd.Timestamp, tuple[np.ndarray | None, int]] = {}
    centers = (np.arange(ANCHOR_BINS, dtype=float) + .5) / ANCHOR_BINS
    audit_rows: list[dict[str, object]] = []
    for day in _calendar_days(work):
        if day in removals:
            bins, values = removals[day]
            count -= np.bincount(bins, minlength=ANCHOR_BINS)
            total -= np.bincount(bins, weights=values, minlength=ANCHOR_BINS)
        if day in additions:
            bins, values = additions[day]
            count += np.bincount(bins, minlength=ANCHOR_BINS)
            total += np.bincount(bins, weights=values, minlength=ANCHOR_BINS)
        if (count < -1e-6).any():
            raise AssertionError("prequential anchor support became negative")
        count = np.maximum(count, 0.0)
        support = int(count.sum())
        present = count > 0
        curve: np.ndarray | None = None
        if support >= ANCHOR_MIN_SUPPORT and int(present.sum()) >= 2:
            means = total[present] / count[present]
            curve = IsotonicRegression(
                increasing=True, out_of_bounds="clip", y_min=-500.0, y_max=500.0,
            ).fit(centers[present], means, sample_weight=count[present]).predict(centers)
        day_anchor[day] = (curve, support)
        audit_rows.append({
            "day": day, "score_field": score_field, "resolved_rows": support,
            "nonempty_bins": int(present.sum()), "map_available": curve is not None,
        })
    raw = pd.to_numeric(work[score_field], errors="coerce").to_numpy(float)
    anchors = np.full(len(work), np.nan, dtype=np.float32)
    support_out = np.zeros(len(work), dtype=np.int32)
    for day, positions in work.groupby("__day__", sort=False).indices.items():
        curve, support = day_anchor[pd.Timestamp(day)]
        support_out[positions] = support
        if curve is not None:
            anchors[positions] = np.interp(np.clip(raw[positions], 0.0, 1.0), centers, curve).astype(np.float32)
    result = pd.DataFrame({
        "candidate_id": work["candidate_id"].to_numpy(),
        f"{prefix}_anchor_bps": anchors,
        f"{prefix}_anchor_support": support_out,
    })
    return result, pd.DataFrame(audit_rows)


def _recent_residual_state(frame: pd.DataFrame, *, anchor_field: str, prefix: str) -> pd.DataFrame:
    """Causal global residual state, updated only after label availability."""

    work = frame.loc[:, ["candidate_id", "__decision_ts__", "policy_path_valid", "policy_label_available_ts", "policy_net_bps", anchor_field]].copy()
    work["__day__"] = work["__decision_ts__"].dt.normalize()
    policy = pd.to_numeric(work["policy_net_bps"], errors="coerce")
    anchor = pd.to_numeric(work[anchor_field], errors="coerce")
    valid = (
        work["policy_path_valid"].fillna(False).astype(bool)
        & work["policy_label_available_ts"].notna()
        & np.isfinite(policy.to_numpy(float)) & np.isfinite(anchor.to_numpy(float))
    )
    resolved = work.loc[valid].copy()
    resolved["__available_day__"] = resolved["policy_label_available_ts"].dt.normalize()
    resolved["__residual__"] = (policy.loc[resolved.index] - anchor.loc[resolved.index]).clip(-500.0, 500.0).to_numpy(float)
    daily = resolved.groupby("__available_day__", sort=True)["__residual__"].agg(["count", "sum", lambda x: float(np.square(x).sum())])
    daily.columns = ["count", "sum", "sq_sum"]
    days = _calendar_days(work)
    daily = daily.reindex(days, fill_value=0.0)
    prior = daily.shift(1, fill_value=0.0)
    def mean(window: int) -> pd.Series:
        count = prior["count"].rolling(window, min_periods=1).sum()
        return prior["sum"].rolling(window, min_periods=1).sum() / count.replace(0.0, np.nan)
    mean3, mean7, mean14 = mean(3), mean(7), mean(14)
    count7 = prior["count"].rolling(7, min_periods=1).sum()
    sq7 = prior["sq_sum"].rolling(7, min_periods=1).sum() / count7.replace(0.0, np.nan)
    std7 = np.sqrt(np.maximum(sq7 - mean7 * mean7, 0.0))
    lookup = pd.DataFrame({
        "__day__": days,
        f"{prefix}_recent_residual_mean_3d": mean3.to_numpy(float),
        f"{prefix}_recent_residual_mean_7d": mean7.to_numpy(float),
        f"{prefix}_recent_residual_mean_14d": mean14.to_numpy(float),
        f"{prefix}_recent_residual_std_7d": std7.to_numpy(float),
        f"{prefix}_recent_residual_slope_3d_14d": (mean3 - mean14).to_numpy(float),
        f"{prefix}_recent_residual_support_log1p_7d": np.log1p(count7.to_numpy(float)),
    })
    return work.loc[:, ["candidate_id", "__day__"]].merge(lookup, on="__day__", how="left", validate="many_to_one").drop(columns="__day__")


def _score_state(frame: pd.DataFrame, *, score_field: str, prefix: str) -> pd.DataFrame:
    """Target-free score density/OOD state using strictly preceding days."""

    work = frame.loc[:, ["candidate_id", "__decision_ts__", score_field]].copy()
    work["__day__"] = work["__decision_ts__"].dt.normalize()
    values = pd.to_numeric(work[score_field], errors="coerce")
    work["__value__"] = values.fillna(0.0).clip(0.0, 1.0)
    work["__bin__"] = _score_bin(work[score_field])
    days = _calendar_days(work)
    daily = work.groupby("__day__", sort=True)["__value__"].agg(["count", "sum", lambda x: float(np.square(x).sum())])
    daily.columns = ["count", "sum", "sq_sum"]
    daily = daily.reindex(days, fill_value=0.0)
    bins = pd.crosstab(work["__day__"], work["__bin__"]).reindex(index=days, columns=np.arange(ANCHOR_BINS), fill_value=0.0)
    prior = daily.shift(1, fill_value=0.0)
    prior_count = prior["count"].rolling(28, min_periods=1).sum()
    prior_sum = prior["sum"].rolling(28, min_periods=1).sum()
    prior_mean = prior_sum / prior_count.replace(0.0, np.nan)
    prior_sq = prior["sq_sum"].rolling(28, min_periods=1).sum() / prior_count.replace(0.0, np.nan)
    prior_std = np.sqrt(np.maximum(prior_sq - prior_mean * prior_mean, 1e-8))
    prior_bins = bins.shift(1, fill_value=0.0).rolling(28, min_periods=1).sum()
    day_pos = pd.Index(days).get_indexer(work["__day__"])
    row_bins = work["__bin__"].to_numpy(int)
    support = prior_bins.to_numpy(float)[day_pos, row_bins] / np.maximum(prior_count.to_numpy(float)[day_pos], 1.0)
    mean = prior_mean.to_numpy(float)[day_pos]
    std = prior_std.to_numpy(float)[day_pos]
    ood = np.abs(work["__value__"].to_numpy(float) - mean) / std
    return pd.DataFrame({
        "candidate_id": work["candidate_id"].to_numpy(),
        f"{prefix}_score_support": support.astype(np.float32),
        f"{prefix}_score_ood": np.nan_to_num(ood, nan=10.0, posinf=10.0).astype(np.float32),
    })


def _rank_desc(frame: pd.DataFrame, field: str) -> np.ndarray:
    part = frame.loc[:, ["__decision_ts__", "candidate_id", field]].copy()
    part["__pos__"] = np.arange(len(part), dtype=np.int64)
    part = part.sort_values(["__decision_ts__", field, "candidate_id"], ascending=[True, False, True], kind="stable")
    rank = part.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float)
    count = part.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy(float)
    part["__rank__"] = 1.0 - (rank + .5) / count
    return part.sort_values("__pos__", kind="stable")["__rank__"].to_numpy(np.float32)


def _geometry(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.loc[:, ["candidate_id"]].copy()
    b = _rank_desc(frame, "base_bps")
    e = _rank_desc(frame, "efficiency_bps")
    t = _rank_desc(frame, "timing_bps")
    matrix = np.column_stack([b, e, t]).astype(np.float32)
    ordered = np.sort(matrix, axis=1)
    median = ordered[:, 1]
    out["m2_base_rank"] = b
    out["m2_efficiency_rank"] = e
    out["m2_timing_rank"] = t
    out["m2_rank_min"] = ordered[:, 0]
    out["m2_rank_median"] = median
    out["m2_rank_max"] = ordered[:, 2]
    out["m2_rank_range"] = ordered[:, 2] - ordered[:, 0]
    out["m2_rank_mad"] = np.median(np.abs(matrix - median[:, None]), axis=1)
    out["m2_rank_std"] = matrix.std(axis=1)
    out["m2_base_high_path_low"] = (b - e) + (b - t)
    out["m2_path_high_base_low"] = (e - b) + (t - b)
    out["m2_efficiency_minus_timing_rank"] = e - t
    for level in (.90, .95, .98):
        out[f"m2_fraction_above_p{int(level * 100):02d}"] = (matrix >= level).mean(axis=1)
    return out


def _build_target_free_ledger(
    score_root: Path,
    feature_root: Path,
    policy_root: Path,
    out: Path,
) -> tuple[Path, dict[str, object]]:
    ledger_root = out / "stageg_target_free_ledger"
    marker = ledger_root / "manifest.json"
    if marker.exists():
        manifest = json.loads(marker.read_text())
        return ledger_root, manifest
    ledger_root.mkdir(parents=True, exist_ok=False)
    frame, raw_fields, feature_paths = _read_score_ledger(score_root, feature_root)
    labels = _policy_labels(policy_root)
    frame = frame.merge(labels, on="candidate_id", how="left", validate="one_to_one")
    anchors: list[pd.DataFrame] = []
    states: list[pd.DataFrame] = []
    score_states: list[pd.DataFrame] = []
    anchor_audits: list[pd.DataFrame] = []
    for family, score_field in (("current", "stage1_current_score"), ("bcf", "stage1_bcf_score")):
        prefix = f"m2_{family}"
        anchor, audit = _prequential_daily_anchor(frame, score_field=score_field, prefix=prefix)
        temp = frame.loc[:, ["candidate_id"]].merge(anchor, on="candidate_id", how="left", validate="one_to_one")
        expanded = frame.loc[:, ["candidate_id", "__decision_ts__", "policy_path_valid", "policy_label_available_ts", "policy_net_bps"]].merge(
            temp, on="candidate_id", how="left", validate="one_to_one",
        )
        anchors.append(anchor)
        states.append(_recent_residual_state(expanded, anchor_field=f"{prefix}_anchor_bps", prefix=prefix))
        score_states.append(_score_state(frame, score_field=score_field, prefix=prefix))
        anchor_audits.append(audit.assign(family=family))
    target_free = frame.drop(columns=[
        "policy_path_valid", "policy_label_available_ts", "policy_net_bps", "policy_gross_bps",
        "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price", "policy_exit_reason", "policy_cost_bps",
    ])
    for table in (*anchors, *states, *score_states):
        target_free = target_free.merge(table, on="candidate_id", how="left", validate="one_to_one")
    target_free = target_free.merge(_geometry(target_free), on="candidate_id", how="left", validate="one_to_one")
    prohibited = {"policy_path_valid", "policy_label_available_ts", "policy_net_bps", "policy_gross_bps"}
    if prohibited.intersection(target_free.columns):
        raise AssertionError("outcome field entered Stage-G target-free ledger")
    # Join the broad 120-field contract only within its own month.  The core
    # ledger stays narrow while daily anchors and recent state are calculated.
    source_by_month = {path.parent.name.split("=", 1)[1]: path for path in feature_paths}
    for month, group in target_free.groupby(target_free["__decision_ts__"].dt.strftime("%Y-%m"), sort=True):
        source = source_by_month.get(str(month))
        if source is None:
            raise FileNotFoundError(f"target-free raw source is missing month={month}")
        raw = pd.read_parquet(source, columns=["candidate_id", "__decision_ts__", *raw_fields])
        group = group.merge(raw, on=["candidate_id", "__decision_ts__"], how="left", validate="one_to_one")
        if group.loc[:, raw_fields].isna().all(axis=1).any():
            raise AssertionError(f"month={month}: missing complete raw Meta-2 source row")
        path = ledger_root / f"month={month}.parquet"
        group.to_parquet(path, index=False, compression="zstd")
        del raw, group
        gc.collect()
    audit = pd.concat(anchor_audits, ignore_index=True)
    audit.to_parquet(out / "meta1_prequential_anchor_audit.parquet", index=False, compression="zstd")
    coverage = pd.DataFrame({
        "month": target_free["__decision_ts__"].dt.strftime("%Y-%m"),
        "current_anchor_available": target_free["m2_current_anchor_bps"].notna().astype(float),
        "bcf_anchor_available": target_free["m2_bcf_anchor_bps"].notna().astype(float),
    }).groupby("month", as_index=False).mean()
    coverage.to_parquet(out / "stageg_anchor_coverage.parquet", index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA,
        "ledger": "target-free P2/T1 first-layer receipts plus causal historical-state features",
        "score_root": str(score_root.resolve()),
        "feature_root": str(feature_root.resolve()),
        "policy_root": str(policy_root.resolve()),
        "raw_feature_count": len(raw_fields),
        "anchor_contract": {
            "history_days": ANCHOR_HISTORY_DAYS,
            "bins": ANCHOR_BINS,
            "minimum_resolved_rows": ANCHOR_MIN_SUPPORT,
            "availability": "only labels available before the UTC decision day; same-day labels are deferred",
        },
        "target_free_assertion": "no policy outcome or label-validity field is persisted in the Stage-G ledger",
    }
    marker.write_text(json.dumps(manifest, indent=2) + "\n")
    return ledger_root, manifest


def _load_ledger_months(root: Path, start: pd.Timestamp, end: pd.Timestamp, columns: Sequence[str] | None = None) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for path in sorted(root.glob("month=*.parquet")):
        month = pd.Timestamp(path.stem.split("=", 1)[1] + "-01", tz="UTC")
        if month < end and _month_end(month) > start:
            piece = pd.read_parquet(path, columns=list(columns) if columns is not None else None)
            pieces.append(piece)
    if not pieces:
        raise FileNotFoundError(f"no Stage-G ledger panels for [{start}, {end})")
    frame = pd.concat(pieces, ignore_index=True)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    return frame.loc[frame["__decision_ts__"].ge(start) & frame["__decision_ts__"].lt(end)].copy()


def _label_join(frame: pd.DataFrame, policy_index: pd.DataFrame) -> pd.DataFrame:
    """Attach labels only for training/evaluation, never score persistence."""

    return frame.join(policy_index, on="candidate_id", how="left", validate="many_to_one")


def _stream_train_sample(
    root: Path,
    policy_index: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Build a deterministic day-balanced cap without wide six-month concat."""

    active_days = max(1, len(pd.date_range(start.normalize(), (end - pd.Timedelta(days=1)).normalize(), freq="D", tz="UTC")))
    per_day = max(1, int(math.ceil(META2_TRAIN_CAP / active_days)))
    pieces: list[pd.DataFrame] = []
    for path in sorted(root.glob("month=*.parquet")):
        month = pd.Timestamp(path.stem.split("=", 1)[1] + "-01", tz="UTC")
        if month >= end or _month_end(month) <= start:
            continue
        chunk = pd.read_parquet(path, columns=list(columns))
        chunk["__decision_ts__"] = pd.to_datetime(chunk["__decision_ts__"], utc=True, errors="raise")
        chunk = chunk.loc[chunk["__decision_ts__"].ge(start) & chunk["__decision_ts__"].lt(end)].copy()
        chunk = _label_join(chunk, policy_index)
        valid = (
            chunk["enhanced_base_routed"].fillna(False).astype(bool)
            & chunk["policy_path_valid"].fillna(False).astype(bool)
            & chunk["policy_label_available_ts"].lt(end)
            & np.isfinite(pd.to_numeric(chunk["policy_net_bps"], errors="coerce"))
        )
        chunk = chunk.loc[valid].copy()
        if chunk.empty:
            continue
        chunk["__day__"] = chunk["__decision_ts__"].dt.normalize()
        pieces.extend(
            group.sort_values("candidate_id", kind="stable").head(per_day).drop(columns="__day__")
            for _, group in chunk.groupby("__day__", sort=True)
        )
        del chunk
        gc.collect()
    if not pieces:
        raise ValueError("streamed Meta-2 training sample has no valid rows")
    result = pd.concat(pieces, ignore_index=True)
    if len(result) > META2_TRAIN_CAP:
        result = _sample(result, META2_TRAIN_CAP, SEED + 97)
    return result


def _numeric(frame: pd.DataFrame, fields: Sequence[str], medians: np.ndarray) -> np.ndarray:
    values = frame.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return np.where(np.isfinite(values), values, medians)


def _medians(frame: pd.DataFrame, fields: Sequence[str]) -> np.ndarray:
    medians = frame.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").median().to_numpy(float)
    return np.nan_to_num(medians, nan=0.0)


def _sample(frame: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame
    work = frame.copy()
    work["__day__"] = work["__decision_ts__"].dt.normalize()
    per_day = max(1, int(math.ceil(cap / work["__day__"].nunique())))
    selected = []
    for _, group in work.groupby("__day__", sort=True):
        selected.append(group.sort_values(["candidate_id"], kind="stable").head(per_day))
    return pd.concat(selected, ignore_index=True).head(cap).drop(columns="__day__")


def _m2_fields(arm: str, raw_fields: Sequence[str]) -> tuple[str, ...]:
    geometry = (
        "stage1_current_score", "stage1_bcf_score", "conditional_consensus_rank",
        "ordinary_shadow_consensus_rank", "upstream", "correctness_rank", "head_agreement_std",
        "m2_base_rank", "m2_efficiency_rank", "m2_timing_rank", "m2_rank_min", "m2_rank_median",
        "m2_rank_max", "m2_rank_range", "m2_rank_mad", "m2_rank_std", "m2_base_high_path_low",
        "m2_path_high_base_low", "m2_efficiency_minus_timing_rank", "m2_fraction_above_p90",
        "m2_fraction_above_p95", "m2_fraction_above_p98",
        "head__cap100_ordinary__rank", "head__cap80_ordinary__rank", "head__cap120_equal_month__rank",
        "head__cap40_equal_month__rank", "head__cap60_equal_month__rank",
    )
    state = tuple(parent.META_STATE_FIELDS)
    if arm == "m2_identical_residual":
        return tuple(dict.fromkeys((*raw_fields, *geometry)))
    trust = (
        *geometry,
        "m2_family_anchor_bps", "m2_family_anchor_support", "m2_family_score_support", "m2_family_score_ood",
        "m2_family_recent_residual_mean_3d", "m2_family_recent_residual_mean_7d",
        "m2_family_recent_residual_mean_14d", "m2_family_recent_residual_std_7d",
        "m2_family_recent_residual_slope_3d_14d", "m2_family_recent_residual_support_log1p_7d",
        *state,
    )
    return tuple(dict.fromkeys(trust))


def _persisted_ledger_fields(fields: Sequence[str]) -> tuple[str, ...]:
    """Drop family-local aliases that are materialised only after the read."""

    return tuple(field for field in fields if not field.startswith("m2_family_"))


def _family_view(frame: pd.DataFrame, family: str) -> pd.DataFrame:
    prefix = f"m2_{family}"
    score = "stage1_current_score" if family == "current" else "stage1_bcf_score"
    view = frame.copy()
    view["m2_stage1_score"] = view[score].to_numpy(float)
    for suffix in ("anchor_bps", "anchor_support", "score_support", "score_ood", "recent_residual_mean_3d", "recent_residual_mean_7d", "recent_residual_mean_14d", "recent_residual_std_7d", "recent_residual_slope_3d_14d", "recent_residual_support_log1p_7d"):
        view[f"m2_family_{suffix}"] = view[f"{prefix}_{suffix}"].to_numpy(float)
    return view


def _fit_meta2(train: pd.DataFrame, *, arm: str, family: str, fields: Sequence[str]) -> Meta2Bundle:
    valid = np.isfinite(pd.to_numeric(train["m2_family_anchor_bps"], errors="coerce").to_numpy(float))
    work = train.loc[valid].copy()
    residual = (
        pd.to_numeric(work["policy_net_bps"], errors="coerce").to_numpy(float)
        - pd.to_numeric(work["m2_family_anchor_bps"], errors="coerce").to_numpy(float)
    )
    if len(work) < 5_000 or not np.isfinite(residual).all():
        raise ValueError(f"{family}/{arm}: insufficient strictly OOF residual support")
    work["__target__"] = np.clip(residual, -500.0, 500.0)
    work = _sample(work, META2_TRAIN_CAP, SEED + (11 if family == "current" else 17))
    medians = _medians(work, fields)
    matrix = _numeric(work, fields, medians)
    common = dict(
        n_estimators=140, learning_rate=.035, max_depth=3, num_leaves=15,
        min_child_samples=max(180, int(.02 * len(work))), colsample_bytree=.80,
        subsample=.82, subsample_freq=1, reg_alpha=.10, reg_lambda=8.0,
        max_bin=127, random_state=SEED + (31 if family == "current" else 37),
        n_jobs=1, deterministic=True, force_col_wise=True, verbosity=-1,
    )
    if arm == "m2_identical_residual":
        model: object = LGBMRegressor(objective="huber", alpha=.90, **common).fit(matrix, work["__target__"])
    elif arm == "m3_trust_residual":
        model = LGBMRegressor(objective="quantile", alpha=.20, **common).fit(matrix, work["__target__"])
    elif arm == "m4_adverse_tail":
        severe = (work["__target__"].to_numpy(float) <= META2_ADVERSE_BPS).astype(np.int8)
        if np.unique(severe).size < 2:
            raise ValueError(f"{family}/{arm}: adverse target has one class")
        model = LGBMClassifier(objective="binary", **common).fit(matrix, severe)
    else:
        raise ValueError(f"unsupported Meta-2 arm: {arm}")
    return Meta2Bundle(
        arm=arm, family=family, fields=tuple(fields), medians=medians, model=model,
        train_rows=len(work), residual_mean_bps=float(work["__target__"].mean()), residual_std_bps=float(work["__target__"].std(ddof=0)),
    )


def _correct(bundle: Meta2Bundle, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(bundle.model.predict(_numeric(frame, bundle.fields, bundle.medians)), dtype=float)
    anchor = pd.to_numeric(frame["m2_family_anchor_bps"], errors="coerce").to_numpy(float)
    if bundle.arm == "m2_identical_residual":
        correction = META2_CORRECTION_LAMBDA * np.clip(raw, -META2_CORRECTION_CAP_BPS, META2_CORRECTION_CAP_BPS)
    elif bundle.arm == "m3_trust_residual":
        # Trust may reduce an overconfident first-layer value, never manufacture
        # a new opportunity via a positive correction.
        correction = META2_CORRECTION_LAMBDA * np.clip(raw, -META2_CORRECTION_CAP_BPS, 0.0)
    elif bundle.arm == "m4_adverse_tail":
        probability = np.asarray(bundle.model.predict_proba(_numeric(frame, bundle.fields, bundle.medians))[:, 1], dtype=float)
        raw = probability
        correction = -META2_ADVERSE_DEMOTION_CAP_BPS * np.clip(probability, 0.0, 1.0)
    else:
        raise AssertionError(bundle.arm)
    return (anchor + correction).astype(np.float32), raw.astype(np.float32)


def _final_score(reference: pd.DataFrame, combined: pd.DataFrame, corrected: np.ndarray) -> np.ndarray:
    ref_mask = combined["__reference__"].to_numpy(bool)
    values = corrected[ref_mask]
    values = values[np.isfinite(values)]
    if len(values) < 1_000:
        raise ValueError("same-model reference lacks support for Meta-2 score CDF")
    return parent.ScoreReference.fit(values, source="same_model_meta2_prior_reference").cdf(corrected).astype(np.float32)


def _score_arm(
    ledger_root: Path,
    policy: pd.DataFrame,
    raw_fields: Sequence[str],
    *,
    arm: str,
    out: Path,
) -> pd.DataFrame:
    score_root = out / "target_free_scores"
    score_root.mkdir(parents=True, exist_ok=False)
    fit_rows: list[dict[str, object]] = []
    policy_index = policy.set_index("candidate_id")
    label_fields = list(policy_index.columns)
    for month in parent.SCORE_MONTHS:
        end = _month_end(month)
        reserve_start = month - pd.Timedelta(days=META2_RESERVE_DAYS)
        train_start = month - pd.DateOffset(months=META2_TRAIN_MONTHS)
        ref_start = month - pd.Timedelta(days=parent.BCF_REFERENCE_DAYS)
        print(json.dumps({"event": "meta2_month_begin", "arm": arm, "month": f"{month:%Y-%m}"}), flush=True)
        fields = _m2_fields(arm, raw_fields)
        output_fields = [
            "candidate_id", "__decision_ts__", "side_name", "enhanced_base_routed", "enhanced_base_bps",
            "base_rank42", "base_anchor_bps", "conditional_consensus_rank", "ordinary_shadow_consensus_rank",
            "upstream", "correctness_rank", "head_agreement_std",
            "head__cap100_ordinary__rank", "head__cap80_ordinary__rank", "head__cap120_equal_month__rank",
            "head__cap40_equal_month__rank", "head__cap60_equal_month__rank",
        ]
        if month < META2_FIRST_HELD_MONTH:
            # P2/T1 itself is already strict-OOS in July--September.  It is
            # the required first-layer history for October's first valid
            # Meta-2 fit, so it must remain untouched rather than fabricating
            # an in-sample residual-of-residual warm-up.
            warm = _load_ledger_months(ledger_root, month, end, columns=tuple(dict.fromkeys((*output_fields, "stage1_current_score", "stage1_bcf_score"))))
            for family, score_field in (("current", "stage1_current_score"), ("bcf", "stage1_bcf_score")):
                output = warm.loc[:, output_fields].copy()
                output["final_score"] = pd.to_numeric(warm[score_field], errors="coerce").to_numpy(np.float32)
                path = score_root / family / f"month={month:%Y-%m}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                output.to_parquet(path, index=False, compression="zstd")
                fit_rows.append({
                    "month": f"{month:%Y-%m}", "family": family, "arm": arm,
                    "train_start": None, "reserve_start": None, "train_rows": 0,
                    "reference_rows": 0, "held_rows": int(len(warm)), "feature_count": 0,
                    "residual_mean_bps": np.nan, "residual_std_bps": np.nan,
                    "authority": "immutable P2/T1 warm-up; no earlier OOF first-layer ledger exists",
                })
            del warm
            gc.collect()
            print(json.dumps({"event": "meta2_month_warmup_complete", "arm": arm, "month": f"{month:%Y-%m}"}), flush=True)
            continue
        # All model columns are target-free.  The only wide read is the
        # 42-day reference plus current held month; the six-month fit is
        # streamed and capped before concatenation.
        family_state = tuple(
            f"m2_{family}_{suffix}"
            for family in ("current", "bcf")
            for suffix in (
                "anchor_bps", "anchor_support", "score_support", "score_ood",
                "recent_residual_mean_3d", "recent_residual_mean_7d", "recent_residual_mean_14d",
                "recent_residual_std_7d", "recent_residual_slope_3d_14d",
                "recent_residual_support_log1p_7d",
            )
        )
        # ``m2_family_*`` are deliberately family-local aliases created by
        # ``_family_view`` below.  They are not physical ledger columns, so
        # they must never be requested from Parquet.  Read both persisted
        # family states, then materialise the one relevant to the current or
        # BCF fit in-memory.  This keeps the score receipt target-free and
        # prevents the trust-only arms from silently falling back to a shared
        # or missing state field.
        persisted_model_fields = _persisted_ledger_fields(fields)
        required = tuple(dict.fromkeys((
            *output_fields,
            *persisted_model_fields,
            *family_state,
            "stage1_current_score",
            "stage1_bcf_score",
        )))
        train = _stream_train_sample(ledger_root, policy_index, start=train_start, end=reserve_start, columns=required)
        reference = _label_join(_load_ledger_months(ledger_root, ref_start, month, columns=required), policy_index)
        held = _label_join(_load_ledger_months(ledger_root, month, end, columns=required), policy_index)
        print(json.dumps({"event": "meta2_fold_loaded", "arm": arm, "month": f"{month:%Y-%m}", "train_rows": len(train), "reference_rows": len(reference), "held_rows": len(held)}), flush=True)
        if held.empty or reference.empty:
            raise ValueError(f"{month:%Y-%m}: missing P2 reference or held rows")
        for family in ("current", "bcf"):
            family_train = _family_view(train.copy(), family)
            bundle = _fit_meta2(family_train, arm=arm, family=family, fields=fields)
            combined = pd.concat([reference.assign(__reference__=True), held.assign(__reference__=False)], ignore_index=True)
            combined = _family_view(combined, family)
            corrected, raw = _correct(bundle, combined)
            # Current uses the canonical 28-day same-model reference; BCF keeps
            # its 42-day reference.  Both references are scored by this exact
            # fitted Meta-2 bundle and contain no held outcomes.
            if family == "current":
                reference_cut = combined["__decision_ts__"].ge(reserve_start).to_numpy(bool)
                reference_for_cdf = combined.loc[reference_cut & combined["__reference__"].to_numpy(bool)]
                # Rebuild the indicator so the helper uses exactly this slice.
                cdf_frame = combined.copy()
                cdf_frame["__reference__"] = reference_cut & combined["__reference__"].to_numpy(bool)
            else:
                cdf_frame = combined
            final = _final_score(cdf_frame.loc[cdf_frame["__reference__"]].copy(), cdf_frame, corrected)
            held_mask = ~combined["__reference__"].to_numpy(bool)
            original_score = "stage1_current_score" if family == "current" else "stage1_bcf_score"
            output = combined.loc[held_mask, output_fields].copy()
            output["final_score"] = final[held_mask]
            output["meta2_anchor_bps"] = pd.to_numeric(combined.loc[held_mask, "m2_family_anchor_bps"], errors="coerce").to_numpy(np.float32)
            output["meta2_raw"] = raw[held_mask]
            output["meta2_corrected_bps"] = corrected[held_mask]
            output["meta2_stage1_score"] = pd.to_numeric(combined.loc[held_mask, original_score], errors="coerce").to_numpy(np.float32)
            forbidden = {"policy_net_bps", "policy_path_valid", "policy_label_available_ts", "policy_gross_bps"}
            if forbidden.intersection(output.columns):
                raise AssertionError("Meta-2 target-free score receipt contains outcomes")
            path = score_root / family / f"month={month:%Y-%m}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            output.to_parquet(path, index=False, compression="zstd")
            print(json.dumps({"event": "meta2_family_scored", "arm": arm, "month": f"{month:%Y-%m}", "family": family}), flush=True)
            fit_rows.append({
                "month": f"{month:%Y-%m}", "family": family, "arm": arm,
                "train_start": train_start.isoformat(), "reserve_start": reserve_start.isoformat(),
                "train_rows": bundle.train_rows, "reference_rows": int(len(reference)), "held_rows": int(len(held)),
                "feature_count": len(fields), "residual_mean_bps": bundle.residual_mean_bps,
                "residual_std_bps": bundle.residual_std_bps,
                "authority": (
                    "symmetric 0.25 x clipped [-100,+100] bps residual"
                    if arm == "m2_identical_residual" else
                    "downside-only 0.25 x q20 residual, clipped [-100,0] bps"
                    if arm == "m3_trust_residual" else
                    "downside-only severe-residual probability demotion, capped 50 bps"
                ),
            })
        del train, reference, held
        gc.collect()
        print(json.dumps({"event": "meta2_month_complete", "arm": arm, "month": f"{month:%Y-%m}"}), flush=True)
    audit = pd.DataFrame(fit_rows)
    audit.to_parquet(out / "meta2_fit_audit.parquet", index=False, compression="zstd")
    return audit


def _copy_m0(score_root: Path, out: Path) -> pd.DataFrame:
    target = out / "target_free_scores"
    target.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, object]] = []
    for family in ("current", "bcf"):
        for src in _score_files(score_root, family):
            dst = target / family / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            frame = pd.read_parquet(src)
            frame.to_parquet(dst, index=False, compression="zstd")
            rows.append({"month": src.stem.split("=", 1)[1], "family": family, "arm": "m0_first_layer_control", "train_rows": 0, "reference_rows": 0, "held_rows": len(frame), "feature_count": 0, "authority": "none; immutable P2/T1 first-layer control"})
    audit = pd.DataFrame(rows)
    audit.to_parquet(out / "meta2_fit_audit.parquet", index=False, compression="zstd")
    return audit


def _evaluate(
    paths: parent.Paths,
    out: Path,
    *,
    fit_audit: pd.DataFrame,
    source_manifest: dict[str, object],
    arm: str,
) -> None:
    policy = parent._load_policy(paths)
    current_panel = parent._read_score_panels(out, "current", policy)
    _, current_audit = parent._mc1_predictions(current_panel, "current", out)
    del current_panel
    gc.collect()
    bcf_panel = parent._read_score_panels(out, "bcf", policy)
    _, bcf_audit = parent._mc1_predictions(bcf_panel, "bcf", out)
    del bcf_panel
    gc.collect()
    current = pd.read_parquet(out / "enhanced_current_mc1_predictions.parquet")
    bcf = pd.read_parquet(out / "enhanced_bcf_mc1_predictions.parquet")
    challenger = parent._combined_challenger(current, bcf)
    baseline = parent._baseline(paths, policy)
    baseline_ids = pd.Index(baseline["candidate_id"].astype(str).unique())
    matched = challenger.loc[challenger["candidate_id"].astype(str).isin(baseline_ids)].copy()
    rows: list[dict[str, object]] = []
    for period, (start, end) in parent.EVALUATION_PERIODS.items():
        for label, part in (
            ("live_baseline", baseline),
            ("stageg_matched_stack", matched),
            ("stageg_full_stack_coverage_only", challenger),
        ):
            rows.append(parent._portfolio_metrics(part.loc[part["__decision_ts__"].ge(start) & part["__decision_ts__"].lt(end)].copy(), label, period, out))
    metrics = pd.DataFrame(rows)
    metrics.to_parquet(out / "live_like_portfolio_metrics.parquet", index=False, compression="zstd")
    left = metrics.loc[metrics["arm"].eq("live_baseline")].set_index("period")
    right = metrics.loc[metrics["arm"].eq("stageg_matched_stack")].set_index("period")
    shared = left.index.intersection(right.index)
    delta = pd.DataFrame({"period": shared})
    for field in ("accepted_rows", "realised_rows", "net_ev_bps_per_realised_trade", "net_sum_bps_realised", "worst_month_bps", "worst_week_bps", "max_drawdown"):
        delta[f"delta_{field}"] = right.loc[shared, field].to_numpy(float) - left.loc[shared, field].to_numpy(float)
    delta.to_parquet(out / "delta_vs_live_baseline.parquet", index=False, compression="zstd")
    mc1_audit = pd.concat([current_audit, bcf_audit], ignore_index=True)
    mc1_audit.to_parquet(out / "mc1_fit_audit.parquet", index=False, compression="zstd")
    sample = pd.read_parquet(next((out / "target_free_scores" / "current").glob("*.parquet")))
    prohibited = {"policy_net_bps", "policy_path_valid", "policy_label_available_ts", "policy_gross_bps"}
    if prohibited.intersection(sample.columns):
        raise AssertionError("Stage-G target-free receipt has outcome leakage")
    active_meta2 = fit_audit.loc[pd.to_datetime(fit_audit["month"] + "-01", utc=True).ge(META2_FIRST_HELD_MONTH)]
    if arm != "m0_first_layer_control" and active_meta2["train_rows"].le(0).any():
        raise AssertionError("Stage-G Meta-2 fold lacks strict training support")
    manifest = {
        "schema": SCHEMA, "scope": "offline research only; no live configuration changes",
        "arm": arm,
        "first_layer": "immutable strict-OOS P2/T1 score ledger; no in-sample Meta-1 predictions enter Meta-2 fitting",
        "meta2_target": "canonical rich-policy net minus daily prequential first-layer expected-policy-bps anchor",
        "meta2_anchor": source_manifest["anchor_contract"],
        "reserve": f"{META2_RESERVE_DAYS} calendar days excluded from Meta-2 fits",
        "mc1": "same fixed HGB class/hyperparameters, prequentially refit per score family",
        "admission": "dual current and BCF MC1 >= +30 bps; BCF MC1 expected bps auction priority",
        "portfolio": "same canonical constrained global auction and rich policy outcomes",
        "causality": {
            "score_receipts": "P2/T1 first-layer OOS predictions persisted before policy-label join",
            "anchor": "daily map uses only labels resolved before the decision day",
            "meta2_train": "only first-layer OOS rows whose labels resolved before the fold reserve",
            "held_scores": "target-free receipts persisted before MC1 outcome join",
            "no_held_window_percentile": True,
        },
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def run(args: argparse.Namespace) -> None:
    root = args.out.resolve()
    if root.exists() and not args.resume:
        raise FileExistsError(root)
    if not root.exists():
        root.mkdir(parents=True)
    ledger_root, source_manifest = _build_target_free_ledger(
        args.p2_score_root.resolve(), args.target_free_source.resolve(), args.policy_root.resolve(), root,
    )
    # The M0 first-layer control needs no Meta-2 outcome-derived state.  The
    # ledger is still generated once so all successors share exactly the same
    # causal source and anchor receipt.
    # Do not materialise a 120-field panel merely to discover its contract;
    # even one month is materially larger than the lightweight Stage-G state.
    raw_columns = pq.ParquetFile(next(ledger_root.glob("month=*.parquet"))).schema.names
    meta = {
        "schema": SCHEMA, "scope": "offline research only",
        "p2_score_root": str(args.p2_score_root.resolve()),
        "target_free_source": str(args.target_free_source.resolve()),
        "policy_root": str(args.policy_root.resolve()),
        "raw_feature_count": int(source_manifest["raw_feature_count"]),
    }
    for arm in args.arms:
        arm_out = root / arm
        arm_out.mkdir(parents=True)
        paths = parent.Paths(
            raw_ledger=args.raw_ledger.resolve(), direct_root=args.direct_root.resolve(), policy_root=args.policy_root.resolve(),
            current_mc1=args.current_mc1.resolve(), bcf_mc1=args.bcf_mc1.resolve(), bundle_root=args.bundle_root.resolve(),
        )
        if arm == "m0_first_layer_control":
            fit_audit = _copy_m0(args.p2_score_root.resolve(), arm_out)
        else:
            policy = _policy_labels(args.policy_root.resolve())
            raw_fields = [column for column in raw_columns if column not in {
                "candidate_id", "__decision_ts__", "side_name", "enhanced_base_routed", "base_bps", "efficiency_bps", "timing_bps", "enhanced_base_bps", "base_rank_ts", "e_minus_t", "e_minus_b0", "t_minus_b0", "base_component_std",
                "stage1_current_score", "stage1_bcf_score", "base_rank42", "base_anchor_bps", "conditional_consensus_rank", "ordinary_shadow_consensus_rank", "upstream", "correctness_rank", "head_agreement_std",
            } and not column.startswith("m2_") and not column.startswith("head__")]
            fit_audit = _score_arm(ledger_root, policy, raw_fields, arm=arm, out=arm_out)
        _evaluate(paths, arm_out, fit_audit=fit_audit, source_manifest=source_manifest, arm=arm)
        del fit_audit
        gc.collect()
    root_manifest = root / "run_manifest.json"
    prior_arms: list[str] = []
    if root_manifest.exists():
        prior_arms = list(json.loads(root_manifest.read_text()).get("arms", []))
    root_manifest.write_text(json.dumps({**meta, "arms": list(dict.fromkeys([*prior_arms, *args.arms]))}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--p2-score-root", type=Path, required=True)
    parser.add_argument("--target-free-source", type=Path, required=True)
    parser.add_argument("--raw-ledger", type=Path, required=True)
    parser.add_argument("--direct-root", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--current-mc1", type=Path, required=True)
    parser.add_argument("--bcf-mc1", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=["m0_first_layer_control", "m2_identical_residual", "m3_trust_residual", "m4_adverse_tail"])
    parser.add_argument("--resume", action="store_true", help="reuse the immutable Stage-G target-free ledger and add only missing arm outputs")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
