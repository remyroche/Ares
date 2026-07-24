#!/usr/bin/env python3
"""Compare the causal M0--M5 residual-head stack on a fixed meta stream.

This is intentionally a relative, short-window ablation.  The base/meta score
in the candidate ledger is frozen; every additional head is fit only on the
three calendar months preceding each OOS month.  The train-fitted global top20
score population is retained and every arm keeps its best half, preserving the
same top10 activity budget.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_negative_hit_residual_leaf_rules import (  # noqa: E402
    LEAF_AUDIT_FEATURES,
    _parse_months,
)
from scripts.run_executable_quality_transition_ablation import (  # noqa: E402
    _breakdown,
    _causal_residual_target,
    _fit_predict,
    _metrics,
    _negative_residual_event,
    _safe,
    _select_top10,
    _write_json,
)

DEFAULT_OUTPUT = ROOT / "data_perp/reports/meta_residual_head_stack_ablation_20260719_v1"
DEFAULT_INPUT = (
    ROOT
    / "data_perp/reports/s59_h5_benchmark66_matchedaegmm_refit_wf30_20260716_v1"
    / "meta_handoff_top30_allsafe_newaegmm_20260717/train_meta_regime_handoff.parquet"
)
DEFAULT_LABELS_ROOT = ROOT / "data_perp/artifacts/20260713_s59_h5_fullthroughjul10_trailing_cost100bps_labels/labels"
OBSERVABLE_FEATURE_HINTS = (
    "gmm_", "dae_", "ae_", "cluster", "latent", "mahal", "reconstruction",
    "base_", "score", "adx", "atr", "breakout", "climax", "clv", "comp",
    "convex", "delta", "decel", "dip", "dir_", "dist_", "ema", "evr",
    "flow", "impulse", "innovation", "jump", "leverage", "loc_", "memory",
    "oi", "pullback", "range", "resid", "rsi", "shock", "speed", "spread",
    "thrust", "trend", "vol", "vw_", "wick", "zscore", "xasset",
)
OBSERVABLE_EXCLUDED_PREFIXES = ("regime_", "meta_action_", "meta_context_", "meta_threshold_")
OBSERVABLE_PRIORITY = (
    "score", "base_rank_pct_by_timestamp", "base_rank_pct_by_timestamp_side",
    "base_score_z_by_timestamp", "base_score_z_by_timestamp_side",
    "base_margin_to_cutoff", "base_margin_to_cutoff_z", "base_signal_zscore_within_archetype",
    "gmm_cluster_id", "gmm_posterior_max", "gmm_posterior_margin", "gmm_entropy",
    "gmm_ood_score", "min_mahalanobis", "expected_mahalanobis", "mahalanobis_distance",
    "dae_reconstruction_error", "dae_reconstruction_error_zscore", "AE_reconstruction_error",
    "cluster_speed", "cluster_acceleration", "latent_speed", "latent_acceleration",
    "latent_mahalanobis_drift", "time_since_cluster_change", "rolling_cluster_stability",
    "gmm_prob_0", "gmm_prob_1", "gmm_prob_2", "gmm_prob_3", "gmm_prob_4", "gmm_prob_5",
    "gmm_mahal_0", "gmm_mahal_1", "gmm_mahal_2", "gmm_mahal_3",
    "xasset_ob_liquidity_peer_resid", "oi_rank", "leverage_build_score", "breakout_24h",
    "pullback_depth", "support_distance", "dist_ema20_atr", "dist_ema200_atr",
    "vol_shock_asym_4_12", "vol_shock_asym_8_24", "vol_compression", "trend_r2_24",
    "trend_acceleration", "flow_ratio", "impulse", "jump_intensity", "climax_range_24",
    "range_24h_pct", "wick_ratio_4h_max", "ret4h_peer_resid", "resid_strength",
)
STATE_CELLS = {
    "long_mixed_latent_misfire": ("long", "long_mixed_wideslow_tentative"),
    "short_mixed_off_manifold": ("short", "short_mixed_clean_path"),
    "short_default_latent_uncertainty": ("short", "short_default_clean_path"),
}


def _numeric(frame: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    if name not in frame:
        return np.full(len(frame), default, dtype=np.float32)
    return pd.to_numeric(frame[name], errors="coerce").fillna(default).to_numpy(dtype=np.float32)


def _good_trade_target(frame: pd.DataFrame) -> np.ndarray:
    """Correct direction with positive executable EV and a clean path."""

    return (
        (_numeric(frame, "clean_exec") > 0.5)
        & (_numeric(frame, "ev_after_1pct") > 0.0)
        & (_numeric(frame, "full_path_bad_mae_1r") <= 0.5)
        & (_numeric(frame, "timeout") <= 0.5)
    ).astype(np.float32)


def _conditional_path_target(frame: pd.DataFrame) -> np.ndarray:
    return (
        (_numeric(frame, "full_path_bad_mae_1r") > 0.5)
        | (_numeric(frame, "timeout") > 0.5)
        | (_numeric(frame, "first_touch_bad_mae_1r") > 0.5)
    ).astype(np.float32)


def _local_multiplier(train: pd.DataFrame) -> tuple[dict[str, float], float]:
    """Fit a support-shrunk side x archetype risk/size multiplier."""

    work = pd.DataFrame(
        {
            "key": train["side_name"].astype(str).str.lower()
            + "|"
            + train["archetype_policy_key"].astype(str),
            "good": _good_trade_target(train),
            "path": _conditional_path_target(train),
            "ev": _numeric(train, "ev_after_1pct"),
        }
    )
    global_quality = float(work["good"].mean() - 0.60 * work["path"].mean())
    global_ev = float(work["ev"].mean())
    result: dict[str, float] = {}
    for key, part in work.groupby("key", observed=True, sort=False):
        rows = len(part)
        local_quality = float(part["good"].mean() - 0.60 * part["path"].mean())
        local_ev = float(part["ev"].mean())
        weight = rows / (rows + 450.0)
        quality = weight * local_quality + (1.0 - weight) * global_quality
        ev = weight * local_ev + (1.0 - weight) * global_ev
        # EV breaks ties between equally clean states; the range is deliberately
        # narrow so this is a calibrated size/risk adjustment, not a hard gate.
        result[str(key)] = float(np.clip(1.0 + 0.70 * (quality - global_quality) + 8.0 * (ev - global_ev), 0.85, 1.15))
    return result, global_quality


def _apply_local_multiplier(frame: pd.DataFrame, state: dict[str, float]) -> np.ndarray:
    keys = frame["side_name"].astype(str).str.lower() + "|" + frame["archetype_policy_key"].astype(str)
    return np.asarray([state.get(str(key), 1.0) for key in keys], dtype=np.float32)


def _state_prediction(
    residual_train: pd.DataFrame,
    residual_target: np.ndarray,
    test: pd.DataFrame,
    *,
    side: str,
    archetype: str,
    features: list[str],
    seed: int,
) -> np.ndarray:
    """Predict one named residual state only inside its corresponding cell."""

    output = np.zeros(len(test), dtype=np.float32)
    train_mask = (
        residual_train["side_name"].astype(str).str.lower().eq(side)
        & residual_train["archetype_policy_key"].astype(str).eq(archetype)
    ).to_numpy()
    test_mask = (
        test["side_name"].astype(str).str.lower().eq(side)
        & test["archetype_policy_key"].astype(str).eq(archetype)
    ).to_numpy()
    if int(train_mask.sum()) < 1_500 or int(test_mask.sum()) == 0:
        return output
    predicted, _ = _fit_predict(
        residual_train.loc[train_mask].reset_index(drop=True),
        test.loc[test_mask].reset_index(drop=True),
        features=features,
        target=residual_target[train_mask],
        seed=seed,
    )
    output[np.flatnonzero(test_mask)] = predicted
    return output


def _head_metric(y: np.ndarray, pred: np.ndarray, prefix: str) -> dict[str, float | None]:
    if len(y) == 0 or np.unique(y).size < 2:
        return {f"{prefix}_auc": None, f"{prefix}_ap": None}
    return {
        f"{prefix}_auc": float(roc_auc_score(y, pred)),
        f"{prefix}_ap": float(average_precision_score(y, pred)),
    }


def _month_label_paths(labels_root: Path, month: pd.Period) -> list[Path]:
    """Return the long/short label shards for one UTC calendar month."""

    stamp = f"{month.year:04d}_{month.month:02d}"
    paths = sorted(labels_root.glob(f"train_global_*_5_{stamp}.parquet"))
    if len(paths) != 2:
        raise FileNotFoundError(f"Expected long/short label shards for {month}: found {paths}")
    return paths


def _read_parquet(path: Path, columns: list[str], filters: list[tuple[str, str, Any]] | None = None) -> pd.DataFrame:
    """Read only available columns; parquet schemas vary across historical runs."""

    import pyarrow.parquet as pq

    schema = set(pq.ParquetFile(path).schema_arrow.names)
    return pq.read_table(
        path,
        columns=[name for name in columns if name in schema],
        filters=filters,
    ).to_pandas()


def _load_handoff_with_labels(
    handoff_path: Path,
    labels_root: Path,
    months: list[pd.Period],
    *,
    extra_feature_names: list[str] | None = None,
    full_months: set[str] | None = None,
    max_rows_per_train_month: int | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    """Join current observable handoff context to the matching realized labels.

    The handoff contains the frozen base candidate features; it deliberately has
    no outcomes.  This attaches the policy-label outcomes only after loading,
    keyed by the canonical UTC timestamp, symbol, and side.
    """

    import pyarrow.parquet as pq

    import pyarrow as pa

    schema = pq.ParquetFile(handoff_path).schema_arrow
    handoff_schema = set(schema.names)
    # The extended historical handoff predates the current candidate-ledger
    # schema and calls the base score ``score_base``.  Normalize that known,
    # observable alias at the ingestion boundary.  A score is a model output,
    # not a static feature: falling through without this alias would later
    # cause static-feature hydration to look for an impossible ``score`` key.
    if "score" in handoff_schema:
        score_source = "score"
    elif "score_base" in handoff_schema:
        score_source = "score_base"
    else:
        raise RuntimeError(
            f"Handoff {handoff_path} has neither score nor score_base; "
            "cannot construct a base-candidate residual-state stream"
        )
    present_features = [name for name in LEAF_AUDIT_FEATURES if name in handoff_schema]
    requested_extra = list(dict.fromkeys(str(name) for name in (extra_feature_names or [])))
    # The prior leaf audit intentionally used only its fixed 45-feature basket.
    # This M0--M5 stack needs the full observable handoff context, including
    # base anchors and newly materialized AE/GMM variables.  Do not admit the
    # existing supervised regime scores: they are outcome-derived diagnostics.
    for field in schema:
        name = str(field.name)
        lower = name.lower()
        if name in present_features or name.startswith("__"):
            continue
        if lower.startswith(OBSERVABLE_EXCLUDED_PREFIXES):
            continue
        if not (pa.types.is_floating(field.type) or pa.types.is_integer(field.type) or pa.types.is_boolean(field.type)):
            continue
        if any(token in lower for token in OBSERVABLE_FEATURE_HINTS):
            present_features.append(name)
    present_set = set(present_features)
    present_features = [name for name in OBSERVABLE_PRIORITY if name in present_set]
    # A fixed compact basket prevents the residual-state recognizer from
    # repeatedly allocating a wide raw matrix for each side × archetype cell.
    # The priority list is deliberately capped rather than inferred from OOS.
    present_features = list(dict.fromkeys(present_features))[:48]
    # Keep physical copies of an explicit model contract when they are already
    # in the handoff.  Missing contract columns are hydrated later through the
    # canonical static reader; this loader must not silently reduce the model
    # contract merely because a compact handoff omitted them.
    embedded_extra = [name for name in requested_extra if name in handoff_schema]
    handoff_columns = list(
        dict.fromkeys(
            [
                "__ts__", "__symbol__", "side_name", "archetype_policy_key",
                "__archetype_policy_key__", score_source, "selected_top30",
                *present_features, *embedded_extra,
            ]
        )
    )
    first_month = min(months)
    after_month = max(months) + 1
    full_months = set(full_months or ())
    cap = int(max_rows_per_train_month or 0)
    lower = pd.Timestamp(first_month.start_time, tz="UTC")
    upper = pd.Timestamp(after_month.start_time, tz="UTC")

    # Full-history handoffs can exceed a million rows.  A Parquet file with a
    # single large row group cannot reliably prune by timestamp, so stream it
    # and retain only a deterministic B/M/E sample for train-only months.
    # Evaluation months remain complete, preserving their exact OOS top-k
    # denominator and side/archetype composition.
    full_parts: list[pd.DataFrame] = []
    sampled: dict[tuple[str, int], pd.DataFrame] = {}
    per_bin = max(1, cap // 3) if cap > 0 else 0
    file = pq.ParquetFile(handoff_path)
    for batch in file.iter_batches(columns=handoff_columns, batch_size=50_000):
        part = batch.to_pandas()
        part["__ts__"] = pd.to_datetime(part["__ts__"], utc=True, errors="coerce")
        part = part.loc[part["__ts__"].ge(lower) & part["__ts__"].lt(upper)].copy()
        if part.empty:
            continue
        if "archetype_policy_key" not in part and "__archetype_policy_key__" in part:
            part["archetype_policy_key"] = part["__archetype_policy_key__"]
        if "selected_top30" in part:
            # A combined candidate ledger can inherit this nullable column
            # from one shard while another shard was already materialized as
            # a candidate-only stream.  ``NaN`` means "pre-filtered upstream",
            # not an observed rejection; only an explicit false is removed.
            part = part.loc[part["selected_top30"].fillna(True)].copy()
        if part.empty:
            continue
        part["__handoff_period__"] = part["__ts__"].dt.to_period("M").astype(str)
        for period, group in part.groupby("__handoff_period__", observed=True, sort=False):
            group = group.drop(columns="__handoff_period__").reset_index(drop=True)
            if str(period) in full_months or cap <= 0:
                full_parts.append(group)
                continue
            month_start = pd.Timestamp(pd.Period(str(period), freq="M").start_time, tz="UTC")
            month_end = pd.Timestamp((pd.Period(str(period), freq="M") + 1).start_time, tz="UTC")
            span_ns = max(int((month_end - month_start).value), 1)
            offset_ns = (group["__ts__"].astype("int64") - int(month_start.value)).to_numpy(dtype=np.int64)
            bins = np.minimum(2, np.maximum(0, (offset_ns * 3) // span_ns)).astype(np.int8)
            group["__handoff_bin__"] = bins
            group["__handoff_hash__"] = pd.util.hash_pandas_object(
                group.loc[:, ["__ts__", "__symbol__", "side_name"]], index=False
            ).to_numpy(dtype=np.uint64)
            for bin_id, cell in group.groupby("__handoff_bin__", observed=True, sort=False):
                key = (str(period), int(bin_id))
                existing = sampled.get(key)
                combined = cell if existing is None else pd.concat([existing, cell], ignore_index=True, copy=False)
                sampled[key] = combined.nsmallest(per_bin, "__handoff_hash__", keep="first")
    retained = [*full_parts, *sampled.values()]
    if not retained:
        raise RuntimeError(f"No handoff rows in requested window {first_month}--{after_month}")
    handoff = pd.concat(retained, ignore_index=True, copy=False).drop(
        columns=["__handoff_bin__", "__handoff_hash__"], errors="ignore"
    )
    if score_source != "score":
        handoff = handoff.rename(columns={score_source: "score"})
    # Some replay/source-regime shards are already trimmed and persist the
    # original full-stream rank; other shards retain the full base candidate
    # stream and have no rank column.  A mixed historical ledger must retain
    # the persisted rank where present, while computing it only for the
    # full-stream rows that lack it.  Leaving those rows as NaN silently drops
    # an otherwise valid OOS month at the later top-20 handoff filter.
    observed_rank = pd.to_numeric(
        handoff.get("base_rank_pct_by_timestamp", pd.Series(np.nan, index=handoff.index)),
        errors="coerce",
    )
    computed_rank = handoff.groupby("__ts__", observed=True)["score"].rank(
        method="average", pct=True
    )
    handoff["base_rank_pct_by_timestamp"] = observed_rank.where(
        observed_rank.notna(), computed_rank
    ).astype(np.float32)

    label_columns = [
        "__ts__", "__symbol__", "side_name", "__u_policy_net__",
        "__first_touch_target_soft__",
        "__long_path_clean_exec_label__", "__long_path_dirty_positive_label__",
        "__path_full_bad_mae_1r__", "__first_touch_mae_to_sl__",
        "__first_touch_timeout__", "__is_timeout__", "__first_touch_stop__",
    ]
    labels = pd.concat(
        [_read_parquet(path, label_columns) for month in months for path in _month_label_paths(labels_root, month)],
        ignore_index=True,
    )
    labels["__ts__"] = pd.to_datetime(labels["__ts__"], utc=True)
    labels = labels.drop_duplicates(["__ts__", "__symbol__", "side_name"], keep="last")
    labels = labels.rename(
        columns={
            "__u_policy_net__": "ev_after_1pct",
            "__first_touch_target_soft__": "meta_target_soft",
            "__path_full_bad_mae_1r__": "full_path_bad_mae_1r",
            "__first_touch_timeout__": "timeout",
        }
    )
    merged = handoff.merge(labels, on=["__ts__", "__symbol__", "side_name"], how="left", validate="one_to_one")
    # The generic label contract represents a bad first-touch path when the
    # adverse move reached one SL; the raw MAE-to-SL field is more portable than
    # a geometry-specific boolean name.
    merged["first_touch_bad_mae_1r"] = (
        pd.to_numeric(merged.get("__first_touch_mae_to_sl__"), errors="coerce").fillna(0.0) >= 1.0
    ).astype(np.float32)
    if "timeout" not in merged:
        merged["timeout"] = pd.to_numeric(merged.get("__is_timeout__"), errors="coerce").fillna(0.0)
    merged["timeout"] = pd.to_numeric(merged["timeout"], errors="coerce").fillna(
        pd.to_numeric(merged.get("__is_timeout__"), errors="coerce").fillna(0.0)
    )
    # __long_path_* labels are populated only on long-side label shards.  Use
    # one executable definition for both sides rather than losing every short
    # row due to that historical naming asymmetry.
    net = pd.to_numeric(merged["ev_after_1pct"], errors="coerce")
    full_path_bad = pd.to_numeric(merged["full_path_bad_mae_1r"], errors="coerce")
    timeout = pd.to_numeric(merged["timeout"], errors="coerce")
    merged["clean_exec"] = ((net > 0.0) & (full_path_bad <= 0.5) & (timeout <= 0.5)).astype(np.float32)
    merged["dirty_positive"] = ((net > 0.0) & ((full_path_bad > 0.5) | (timeout > 0.5))).astype(np.float32)
    required = ["ev_after_1pct", "full_path_bad_mae_1r", "timeout"]
    missing = merged[required].isna().any(axis=1)
    coverage = {
        str(period): int(group.loc[~missing.loc[group.index]].shape[0])
        for period, group in merged.groupby(merged["__ts__"].dt.to_period("M"), observed=True)
    }
    # A legacy January label shard has a different contract.  It is never an
    # evaluation month here; retain only rows with a verified matching outcome
    # and make OOS-month completeness an explicit guard in the fold loop.
    merged = merged.loc[~missing].copy()
    for column in merged.select_dtypes(include=["float64"]).columns:
        merged[column] = merged[column].astype(np.float32)
    return merged, present_features, coverage


def _time_spread_sample(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Keep a deterministic beginning/middle/end spread without pandas copies."""

    if len(frame) <= max_rows:
        return frame
    order = np.argsort(frame["__ts__"].to_numpy(dtype="datetime64[ns]"), kind="stable")
    take = np.linspace(0, len(order) - 1, max_rows, dtype=np.int64)
    return frame.iloc[order[take]].copy()


def _residual_arch_context(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Causal side × archetype × frozen-GMM reliability context for M5.

    The previous generic recognizer allocates one wide model per local cell,
    which is unsuitable for a comparative M0--M5 run.  This prior keeps the
    same inference-safe information: a frozen, observable latent-state key;
    train-only support; a shrinkage estimate of clean-hit surprise; dirty-path
    probability; and Bernoulli uncertainty entropy.  No test outcome enters
    the key, estimate, or fallback.
    """

    def _key(frame: pd.DataFrame, include_gmm: bool) -> pd.Series:
        base = frame["side_name"].astype(str).str.lower() + "|" + frame["archetype_policy_key"].astype(str)
        if not include_gmm:
            return base
        raw_cluster = frame.get("gmm_cluster_id")
        if raw_cluster is None:
            # A side-specific MDA contract may retain posterior/distance fields
            # while pruning the hard ID.  Do not reconstruct an ID from partial
            # components; the causal parent reliability key remains valid.
            return base
        cluster = pd.to_numeric(raw_cluster, errors="coerce").fillna(-1).astype(np.int16).astype(str)
        return base + "|gmm_" + cluster

    work = pd.DataFrame(
        {
            "local_key": _key(train, True),
            "parent_key": _key(train, False),
            "clean": _good_trade_target(train),
            "dirty": _conditional_path_target(train),
        }
    )
    global_clean = float(work["clean"].mean())
    global_dirty = float(work["dirty"].mean())
    parent = work.groupby("parent_key", observed=True).agg(clean=("clean", "mean"), dirty=("dirty", "mean"), n=("clean", "size"))
    local = work.groupby("local_key", observed=True).agg(clean=("clean", "mean"), dirty=("dirty", "mean"), n=("clean", "size"))
    out = pd.DataFrame(index=test.index)
    keys = _key(test, True)
    parent_keys = _key(test, False)
    for idx, (local_key, parent_key) in enumerate(zip(keys, parent_keys, strict=False)):
        row = local.loc[local_key] if local_key in local.index else None
        parent_row = parent.loc[parent_key] if parent_key in parent.index else None
        parent_n = float(parent_row["n"]) if parent_row is not None else 0.0
        parent_clean = float(parent_row["clean"]) if parent_row is not None else global_clean
        parent_dirty = float(parent_row["dirty"]) if parent_row is not None else global_dirty
        n = float(row["n"]) if row is not None else 0.0
        weight = n / (n + 200.0)
        clean = weight * float(row["clean"]) + (1.0 - weight) * parent_clean if row is not None else parent_clean
        dirty = weight * float(row["dirty"]) + (1.0 - weight) * parent_dirty if row is not None else parent_dirty
        # Parent support stabilizes sparse latent states without treating their
        # absence as a negative signal.
        support = n + 0.15 * parent_n
        out.loc[test.index[idx], "meta_resid_arch_expected_hit_surprise"] = clean - global_clean
        out.loc[test.index[idx], "meta_resid_arch_expected_dirty_positive"] = dirty
        out.loc[test.index[idx], "meta_resid_arch_support_log1p"] = np.log1p(support)
        p = float(np.clip(clean, 1e-5, 1.0 - 1e-5))
        out.loc[test.index[idx], "meta_resid_arch_entropy"] = -(p * np.log(p) + (1.0 - p) * np.log1p(-p)) / np.log(2.0)
    return out.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--labels-root", type=Path, default=DEFAULT_LABELS_ROOT)
    parser.add_argument("--months", default="2026-04,2026-05,2026-06")
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--max-train-rows-per-month", type=int, default=6_000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    months = _parse_months(args.months)
    required_months = sorted(
        {
            period
            for month in months
            for period in pd.period_range(month - args.train_months, month, freq="M")
        }
    )
    handoff, features, outcome_coverage = _load_handoff_with_labels(args.input_parquet, args.labels_root, required_months)
    if len(features) < 12:
        raise RuntimeError(f"Only {len(features)} observable audit features are available from {args.input_parquet}")
    rows: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []

    for fold, month in enumerate(months):
        start = pd.Timestamp(month.start_time, tz="UTC")
        end = pd.Timestamp((month + 1).start_time, tz="UTC")
        earliest = start - pd.DateOffset(months=int(args.train_months))
        train_full = handoff.loc[(handoff["__ts__"] >= earliest) & (handoff["__ts__"] < start)]
        test_full = handoff.loc[(handoff["__ts__"] >= start) & (handoff["__ts__"] < end)]
        expected_oos = int(
            ((pd.to_datetime(handoff["__ts__"], utc=True) >= start) & (pd.to_datetime(handoff["__ts__"], utc=True) < end)).sum()
        )
        if expected_oos == 0:
            raise RuntimeError(f"No outcome-covered OOS rows remain for {month}")
        if len(train_full) == 0 or len(test_full) == 0:
            continue
        # The handoff is the frozen base top-30 candidate stream.  The base
        # rank percentile was computed cross-sectionally at each timestamp;
        # rank >= .80 therefore defines the same global top-20 source across
        # sides without relying on incomparable raw-score levels.
        rank_col = "base_rank_pct_by_timestamp"
        if rank_col not in handoff:
            raise RuntimeError(f"{rank_col} is required to form the fixed top20 candidate source")
        train_full = train_full.loc[pd.to_numeric(train_full[rank_col], errors="coerce") >= 0.80]
        test = test_full.loc[pd.to_numeric(test_full[rank_col], errors="coerce") >= 0.80].reset_index(drop=True)
        train = pd.concat(
            [
                _time_spread_sample(part, args.max_train_rows_per_month)
                for _, part in train_full.groupby(train_full["__ts__"].dt.to_period("M"), observed=True, sort=True)
            ],
            ignore_index=True,
        )
        if len(train) < 8_000 or len(test) < 1_000:
            continue
        good_target = _good_trade_target(train)
        good_pred, good_report = _fit_predict(train, test, features=features, target=good_target, seed=args.seed + fold)
        # Conditional path risk is only meaningful among opportunities that
        # remain economically plausible.  Conditioning on clean_exec made the
        # adverse label tautologically zero and silently reduced M2 to M1.
        economically_plausible = (
            (_numeric(train, "ev_after_1pct") > 0.0)
            | (_numeric(train, "clean_exec") > 0.5)
        )
        path_train = train.loc[economically_plausible].reset_index(drop=True)
        path_target = _conditional_path_target(path_train)
        path_pred, path_report = _fit_predict(path_train, test, features=features, target=path_target, seed=args.seed + fold + 1_000)
        local_state, _ = _local_multiplier(train)
        residual_train, residual_target, residual_state = _causal_residual_target(
            train, value_col="clean_exec", label_col="__negative_hit_residual_event__"
        )
        residual_pred, residual_report = _fit_predict(
            residual_train, test, features=features, target=residual_target, seed=args.seed + fold + 2_000
        )
        state_predictions = {
            name: _state_prediction(
                residual_train, residual_target, test, side=side, archetype=arch,
                features=features, seed=args.seed + fold + 3_000 + idx,
            )
            for idx, (name, (side, arch)) in enumerate(STATE_CELLS.items())
        }
        residual_context = _residual_arch_context(train, test)
        expected_hit = _numeric(residual_context, "meta_resid_arch_expected_hit_surprise")
        expected_dirty = _numeric(residual_context, "meta_resid_arch_expected_dirty_positive")
        entropy = _numeric(residual_context, "meta_resid_arch_entropy", default=1.0)
        support = _numeric(residual_context, "meta_resid_arch_support_log1p")
        support_z = (support - float(np.nanmean(support))) / max(float(np.nanstd(support)), 1e-3)
        # Use the frozen cross-sectional base rank as the M0 ordering.  Raw
        # score levels from global LGBM are side-dependent and cannot compete
        # fairly in a unified auction; this rank is the actual comparable base
        # opportunity ordering available at decision time.
        base = _numeric(test, "base_rank_pct_by_timestamp")
        m1 = base * (0.60 + 0.40 * good_pred)
        m2 = m1 * (1.0 - 0.50 * path_pred)
        m3 = m2 * _apply_local_multiplier(test, local_state)
        state_risk = np.maximum.reduce(list(state_predictions.values())) if state_predictions else np.zeros(len(test), dtype=np.float32)
        m4 = m3 * (1.0 - 0.45 * residual_pred) * (1.0 - 0.30 * state_risk)
        # M5 is deliberately bounded: support/entropy and expected surprise are
        # contextual calibration, not an uncontrolled second policy.
        m5_factor = np.clip(1.0 + 0.10 * np.clip(expected_hit, -1.5, 1.5) - 0.10 * expected_dirty - 0.05 * entropy + 0.02 * np.clip(support_z, -2.0, 2.0), 0.75, 1.25)
        scores = {"M0_current_head": base, "M1_good_trade": m1, "M2_conditional_path": m2, "M3_local_size_risk": m3, "M4_residual_states": m4, "M5_residual_arch_context": m4 * m5_factor}
        baseline_tail = np.ones(len(test), dtype=bool)
        observed_residual = _negative_residual_event(test, residual_state, value_col="clean_exec")
        observed_path = _conditional_path_target(test)
        for arm, score in scores.items():
            selected = _select_top10(test, score, baseline_tail)
            rows.append({"month": str(month), **_metrics(selected, arm)})
            details.append(_breakdown(selected, arm))
        diagnostic = {
            "month": str(month), "train_rows": int(len(train)), "oos_rows": int(len(test)),
            "train_months": int(args.train_months), "source_rank_cutoff": 0.80,
            "good_models": good_report, "path_models": path_report, "residual_models": residual_report,
            **_head_metric(_good_trade_target(test), good_pred, "good_trade"),
            **_head_metric(observed_path, path_pred, "conditional_path"),
            **_head_metric(observed_residual, residual_pred, "negative_hit_residual"),
        }
        for name, prediction in state_predictions.items():
            side, arch = STATE_CELLS[name]
            mask = test["side_name"].astype(str).str.lower().eq(side) & test["archetype_policy_key"].astype(str).eq(arch)
            diagnostic.update(_head_metric(observed_residual[mask.to_numpy()], prediction[mask.to_numpy()], name))
        diagnostics.append(diagnostic)
        print(json.dumps({"event": "fold_complete", "month": str(month), "train_rows": len(train), "oos_rows": len(test)}), flush=True)
        del train, test, residual_train, residual_context
        gc.collect()

    scorecard = pd.DataFrame(rows)
    scorecard.to_csv(args.output / "oos_scorecard_by_month.csv", index=False)
    aggregate = scorecard.groupby("arm", observed=True).mean(numeric_only=True).reset_index() if not scorecard.empty else pd.DataFrame()
    if not aggregate.empty:
        baseline = aggregate.loc[aggregate["arm"].eq("M0_current_head")].iloc[0]
        for name in ("mean_ev_after_1pct", "worst_week_ev", "worst_month_ev", "clean_exec_precision", "full_path_bad_mae_rate", "timeout_rate"):
            aggregate[f"delta_{name}_vs_M0"] = aggregate[name] - float(baseline[name])
    aggregate.to_csv(args.output / "oos_scorecard_aggregate.csv", index=False)
    (pd.concat(details, ignore_index=True) if details else pd.DataFrame()).to_csv(args.output / "oos_side_archetype_breakdown.csv", index=False)
    pd.DataFrame(diagnostics).to_json(args.output / "head_diagnostics.json", orient="records", indent=2)
    _write_json(args.output / "manifest.json", {
        "schema": "meta_residual_head_stack_ablation_v1",
        "months": [str(month) for month in months], "train_months": int(args.train_months),
        "input_parquet": str(args.input_parquet), "labels_root": str(args.labels_root),
        "outcome_joined_rows_by_month": outcome_coverage,
        "max_train_rows_per_month": int(args.max_train_rows_per_month), "feature_count": len(features), "features": features,
        "arms": {"M0": "frozen cross-sectional base rank", "M1": "M0 plus local good-trade prediction", "M2": "M1 plus conditional adverse-path prediction", "M3": "M2 plus support-shrunk local size/risk multiplier", "M4": "M3 plus negative-hit residual and three supervised leaf-state probabilities", "M5": "M4 plus causal residual-archetype support, entropy, expected hit surprise, and expected dirty probability"},
        "state_labels": STATE_CELLS,
        "selection_contract": "fixed base top30 handoff; cross-sectional base-rank top20 source at every UTC timestamp; select its best half per timestamp to form the global top10 budget",
        "leakage_contract": "all heads, local multipliers, residual archetypes, and normalization fit only on prior resolved rows; OOS transforms receive no outcome columns",
    })
    print(json.dumps({"event": "complete", "output": str(args.output), "scorecard_rows": len(scorecard)}), flush=True)


if __name__ == "__main__":
    main()
