#!/usr/bin/env python3
"""Strictly-prequential MC1 admissions ablation funnel, long-only.

This is an *offline research* producer.  It consumes the target-free strict-R3
score ledger and the regenerated, persistent ten-head sidecar, then joins
policy outcomes only for fitting/evaluation after their availability timestamp.
It never reads a live state, never scores an incomplete candidate universe, and
never changes the sealed live MC1_d2 artifact.

The funnel is deliberately sequential:
  1. build causal 6-hour agreement/state features;
  2. screen them against a prequential MC1-shaped residual;
  3. add up to three portable representatives per family;
  4. HPO small robust regressors on 2025 only;
  5. replay frozen choices over 2026, with Robust-21 blend and auction tests;
  6. fit a two-stage LambdaRank challenger only after the regression winner.

All selection is 2025 development evidence.  2026 is reported separately as
opened validation/model-selection evidence, never as untouched promotion data.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata, spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_selection import mutual_info_regression
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / (
    "data_perp/artifacts/strict_r3_lockstep_history_long_2024apr_jul2026_"
    "strictfull_prior28_optimizedpolicy_20260813_v2/"
    "walkforward_scored_label_ledger.parquet"
)
DEFAULT_HEADS = ROOT / (
    "data_perp/artifacts/strict_r3_ten_head_history_long_2024apr_today_"
    "20260816_v2/ten_head_target_free_scores.parquet"
)

SEED = 1729
BASE_FEATURES = (
    "final_score", "base_rank42", "conditional_consensus_rank", "upstream",
    "ordinary_shadow_consensus_rank", "correctness_rank",
)
POLICY_TIMEOUT = pd.Timedelta(hours=12)
TARGET_EDGES = (-np.inf, -200.0, -50.0, 50.0, 150.0, 250.0, np.inf)
TARGET_CENTRES = np.asarray((-300.0, -125.0, 0.0, 100.0, 200.0, 350.0))
DEV_START = pd.Timestamp("2025-01-01", tz="UTC")
DEV_END = pd.Timestamp("2026-01-01", tz="UTC")
VALIDATION_END = pd.Timestamp("2026-08-01", tz="UTC")


def utc(value: object) -> pd.Timestamp:
    value = pd.Timestamp(value)
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rank_fraction(frame: pd.DataFrame, field: str, fraction: float) -> pd.Series:
    """Timestamp-local ranking.  No outcome or held-period distribution is used."""
    ordered = frame.loc[:, ["__decision_ts__", "candidate_id", field]].copy()
    ordered["__source_index__"] = np.arange(len(ordered), dtype=np.int64)
    ordered = ordered.sort_values(
        ["__decision_ts__", field, "candidate_id"],
        ascending=[True, False, True], kind="stable", na_position="last",
    )
    position = ordered.groupby("__decision_ts__", sort=False).cumcount().to_numpy()
    count = ordered.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy()
    ordered["__keep__"] = position < np.maximum(1, np.ceil(count * fraction).astype(int))
    return pd.Series(ordered["__keep__"].to_numpy(bool), index=ordered["__source_index__"].to_numpy()).reindex(frame.index, fill_value=False)


def _mad(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    return float(np.median(np.abs(values - np.median(values))))


def _entropy_rows(values: np.ndarray, bins: int = 5) -> np.ndarray:
    encoded = np.clip((np.nan_to_num(values, nan=0.5) * bins).astype(int), 0, bins - 1)
    output = np.zeros(len(values), dtype=float)
    for bin_id in range(bins):
        mass = np.mean(encoded == bin_id, axis=1)
        valid = mass > 0.0
        output[valid] -= mass[valid] * np.log(mass[valid])
    return output


def _rank_percentile_prior(series: pd.Series, window: int) -> pd.Series:
    """Percentile of each point against *earlier* complete 6h summaries."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    answer = np.full(len(values), np.nan, dtype=float)
    for index, value in enumerate(values):
        if not np.isfinite(value):
            continue
        start = max(0, index - window)
        prior = values[start:index]
        prior = prior[np.isfinite(prior)]
        if len(prior) >= max(6, window // 8):
            answer[index] = (np.sum(prior < value) + 0.5 * np.sum(prior == value)) / len(prior)
    return pd.Series(answer, index=series.index)


def _month_key(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True).dt.strftime("%Y-%m")


def load_panel(ledger_path: Path, head_path: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    if not ledger_path.exists() or not head_path.exists():
        raise FileNotFoundError("strict-R3 historical score or ten-head ledger is unavailable")
    label_columns = [
        "candidate_id", "__decision_ts__", "__symbol__", "side_name",
        "policy_label_available_ts", "policy_path_valid", "policy_net_bps",
        "policy_exit_bar_15m", "base_score", *BASE_FEATURES,
    ]
    labels = pd.read_parquet(ledger_path, columns=label_columns)
    # Do not materialise the raw head columns merely to inspect their names.
    # The sidecar is intentionally large because it preserves raw outputs too.
    head_columns = pq.ParquetFile(head_path).schema_arrow.names
    conditional = sorted(c for c in head_columns if c.startswith("conditional_head__") and c.endswith("__rank"))
    ordinary = sorted(c for c in head_columns if c.startswith("ordinary_shadow_head__") and c.endswith("__rank"))
    if len(conditional) != 10 or len(ordinary) != 10:
        raise ValueError(f"expected 10 persistent conditional + 10 ordinary head ranks; found {len(conditional)}/{len(ordinary)}")
    heads = pd.read_parquet(
        head_path,
        # The ten persistent conditional outputs are the only individual-head
        # contract used by MC1.  Ordinary-shadow is already represented by the
        # frozen aggregate field in the primary ledger, so loading ten more raw
        # ranks here would only double memory without creating a stable feature.
        columns=["candidate_id", "__decision_ts__", "source_kind", "upstream_bundle_sha256", *conditional],
        filters=[("source_kind", "=", "historical_exact_source_panel")],
    )
    heads = heads.drop(columns="source_kind")
    if heads.candidate_id.duplicated().any() or labels.candidate_id.duplicated().any():
        raise ValueError("the target-free or labelled ledger has duplicate candidate identity")
    labels["__decision_ts__"] = pd.to_datetime(labels["__decision_ts__"], utc=True)
    labels["policy_label_available_ts"] = pd.to_datetime(labels["policy_label_available_ts"], utc=True)
    heads["__decision_ts__"] = pd.to_datetime(heads["__decision_ts__"], utc=True)
    # Both files are deliberately produced from the same target-free source
    # panel in the same immutable candidate order.  A merge would temporarily
    # multiply a 2.7m-row string-heavy table several times.  Verify that exact
    # ordering, then assign the rank vectors without materialising a join.
    if len(heads) != len(labels) or not labels["candidate_id"].equals(heads["candidate_id"]) or not labels["__decision_ts__"].equals(heads["__decision_ts__"]):
        raise ValueError("ten-head sidecar is not in exact candidate order; refuse memory-unsafe implicit alignment")
    panel = labels
    panel["upstream_bundle_sha256"] = heads["upstream_bundle_sha256"].to_numpy(copy=False)
    for column in conditional:
        panel[column] = heads[column].to_numpy(copy=False)
    del heads
    gc.collect()
    if not panel.side_name.astype(str).str.lower().eq("long").all():
        raise ValueError("MC1 admissions ablation is deliberately long-only")
    finite = panel.loc[:, conditional].apply(pd.to_numeric, errors="coerce").notna().mean()
    if float(finite.min()) < 1.0:
        raise ValueError("persistent head rank coverage is incomplete")
    return panel, conditional, []


def add_static_agreement_geometry(
    frame: pd.DataFrame, conditional: Sequence[str], ordinary: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Head geometry uses stable, regenerated conditional-head semantics only."""
    out = frame
    # Process the 10-head matrix in chunks.  A whole 2.7m x 10 float64 matrix
    # plus quantile temporaries breaches the bounded research runtime even
    # though the resulting float32 features are compact.
    n_rows = len(out)
    arrays: dict[str, np.ndarray] = {}
    def reserve(name: str) -> np.ndarray:
        values = np.empty(n_rows, dtype=np.float32)
        arrays[name] = values
        return values
    for name in ("agr_head_mean", "agr_head_median", "agr_head_consensus_gap",
                 "agr_rank_mad", "agr_rank_std", "agr_rank_iqr", "agr_rank_range",
                 "agr_max_minus_median", "agr_second_best", "agr_third_best", "agr_min_top3",
                 "agr_vote_imbalance", "agr_polarisation", "agr_upper_mass", "agr_lower_mass",
                 "agr_support_minus_opposition", "agr_rank_entropy", "ordinary_head_mean"):
        reserve(name)
    thresholds = (.90, .95, .97, .98, .99)
    level: list[str] = ["agr_head_mean", "agr_head_median", "agr_head_consensus_gap"]
    for threshold in thresholds:
        name = str(threshold).replace(".", "")
        reserve(f"agr_frac_ge_{name}"); reserve(f"agr_excess_ge_{name}")
        level.extend((f"agr_frac_ge_{name}", f"agr_excess_ge_{name}"))
    dispersion = ["agr_rank_mad", "agr_rank_std", "agr_rank_iqr", "agr_rank_range", "agr_max_minus_median"]
    for distance in (1.0, 1.25, 1.5, 1.75, 2.0):
        tag = str(distance).replace(".", "")
        column = f"agr_frac_far_{tag}sd"
        reserve(column)
        dispersion.append(column)
    tail = ["agr_second_best", "agr_third_best", "agr_min_top3"]
    for threshold in (.99, .98, .95, .90):
        tag = str(threshold).replace(".", "")
        column = f"agr_tail_{tag}"
        reserve(column)
        tail.append(column)
    shape = ["agr_vote_imbalance", "agr_polarisation", "agr_upper_mass", "agr_lower_mass", "agr_support_minus_opposition"]
    for tolerance in (.03, .05, .10):
        tag = str(tolerance).replace(".", "")
        column = f"agr_within_{tag}_median"
        reserve(column)
        shape.append(column)
    shape.append("agr_rank_entropy")
    for start in range(0, n_rows, 100_000):
        stop = min(n_rows, start + 100_000)
        x = out.iloc[start:stop].loc[:, conditional].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        med = np.nanmedian(x, axis=1)
        arrays["agr_head_mean"][start:stop] = np.nanmean(x, axis=1)
        arrays["agr_head_median"][start:stop] = med
        arrays["agr_head_consensus_gap"][start:stop] = (
            pd.to_numeric(out.iloc[start:stop]["conditional_consensus_rank"], errors="coerce").to_numpy(np.float32) - arrays["agr_head_mean"][start:stop]
        )
        arrays["agr_rank_mad"][start:stop] = np.nanmedian(np.abs(x - med[:, None]), axis=1)
        arrays["agr_rank_std"][start:stop] = np.nanstd(x, axis=1)
        arrays["agr_rank_iqr"][start:stop] = np.nanquantile(x, .75, axis=1) - np.nanquantile(x, .25, axis=1)
        arrays["agr_rank_range"][start:stop] = np.nanmax(x, axis=1) - np.nanmin(x, axis=1)
        arrays["agr_max_minus_median"][start:stop] = np.nanmax(x, axis=1) - med
        scale = np.maximum(arrays["agr_rank_std"][start:stop], .01)
        for threshold in thresholds:
            tag = str(threshold).replace(".", "")
            arrays[f"agr_frac_ge_{tag}"][start:stop] = np.mean(x >= threshold, axis=1)
            arrays[f"agr_excess_ge_{tag}"][start:stop] = np.mean(np.maximum(x - threshold, 0.0), axis=1)
        for distance in (1.0, 1.25, 1.5, 1.75, 2.0):
            tag = str(distance).replace(".", "")
            arrays[f"agr_frac_far_{tag}sd"][start:stop] = np.mean(np.abs(x - med[:, None]) / scale[:, None] >= distance, axis=1)
        ordered = np.sort(x, axis=1)
        arrays["agr_second_best"][start:stop] = ordered[:, -2]
        arrays["agr_third_best"][start:stop] = ordered[:, -3]
        arrays["agr_min_top3"][start:stop] = ordered[:, -3]
        for threshold in (.99, .98, .95, .90):
            arrays[f"agr_tail_{str(threshold).replace('.', '')}"][start:stop] = np.mean(x >= threshold, axis=1)
        arrays["agr_vote_imbalance"][start:stop] = np.abs(np.sum(np.sign(x - .5), axis=1)) / x.shape[1]
        arrays["agr_polarisation"][start:stop] = np.mean(np.abs(x - .5), axis=1)
        arrays["agr_upper_mass"][start:stop] = np.mean(x >= .75, axis=1)
        arrays["agr_lower_mass"][start:stop] = np.mean(x <= .25, axis=1)
        arrays["agr_support_minus_opposition"][start:stop] = np.nanmax(x, axis=1) - (1.0 - np.nanmin(x, axis=1))
        for tolerance in (.03, .05, .10):
            arrays[f"agr_within_{str(tolerance).replace('.', '')}_median"][start:stop] = np.mean(np.abs(x - med[:, None]) <= tolerance, axis=1)
        arrays["agr_rank_entropy"][start:stop] = _entropy_rows(x)
        arrays["ordinary_head_mean"][start:stop] = pd.to_numeric(out.iloc[start:stop]["ordinary_shadow_consensus_rank"], errors="coerce").to_numpy(np.float32)
    for name, values in arrays.items():
        out[name] = values
    del arrays
    gc.collect()
    # Ordinary-shadow is deliberately an aggregate contrast here.  The ten
    # individual conditional heads have stable semantic names; treating the
    # shadow ranks as a second persistent specialist universe would be noise.
    out["ordinary_head_mean"] = pd.to_numeric(out["ordinary_shadow_consensus_rank"], errors="coerce").astype(np.float32)
    base_vs = ["ordinary_head_mean"]
    for left, right, name in (
        ("base_rank42", "conditional_consensus_rank", "gap_base_conditional"),
        ("base_rank42", "ordinary_shadow_consensus_rank", "gap_base_ordinary"),
        ("conditional_consensus_rank", "correctness_rank", "gap_consensus_correctness"),
        ("base_rank42", "agr_head_median", "gap_base_head_median"),
        ("conditional_consensus_rank", "ordinary_head_mean", "gap_conditional_ordinary_heads"),
        ("agr_second_best", "base_rank42", "gap_second_head_base"),
    ):
        out[name] = pd.to_numeric(out[left], errors="coerce") - pd.to_numeric(out[right], errors="coerce")
        base_vs.append(name)
    families = {
        "base_contract": list(BASE_FEATURES),
        "agreement_level": level,
        "agreement_dispersion": dispersion,
        "tail_agreement": tail,
        "agreement_shape": shape,
        "base_vs_specialist": base_vs,
    }
    return out, families


def _independence_features(
    train: pd.DataFrame, current: pd.DataFrame, head_columns: Sequence[str], threshold: float = .80,
) -> pd.DataFrame:
    """Fit head-correlation clusters on earlier *features*, then score current.

    The fit deliberately belongs to each fold.  The output is semantic (effective
    independent supporters / cluster strength), not a transient cluster ID.
    """
    train_x = train.loc[:, head_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if len(train_x) > 75_000:
        rng = np.random.default_rng(SEED)
        train_x = train_x[rng.choice(len(train_x), 75_000, replace=False)]
    corr = pd.DataFrame(train_x, columns=head_columns).corr(method="spearman").fillna(0.0).to_numpy(float)
    distance = 1.0 - np.abs(corr)
    np.fill_diagonal(distance, 0.0)
    # sklearn compatibility: metric replaced affinity in recent versions.
    try:
        labels = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average", distance_threshold=1.0 - threshold).fit_predict(distance)
    except TypeError:
        labels = AgglomerativeClustering(n_clusters=None, affinity="precomputed", linkage="average", distance_threshold=1.0 - threshold).fit_predict(distance)
    groups = [np.flatnonzero(labels == group) for group in sorted(set(labels))]
    n_rows = len(current)
    result = {name: np.empty(n_rows, dtype=np.float32) for name in (
        "ind_effective_supporters", "ind_cluster_mean", "ind_cluster_median",
        "ind_cluster_min_high90", "ind_cluster_strength", "ind_cluster_count",
    )}
    for start in range(0, n_rows, 100_000):
        stop = min(n_rows, start + 100_000)
        x = current.iloc[start:stop].loc[:, head_columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        cluster_values = np.column_stack([np.nanmean(x[:, group], axis=1) for group in groups])
        support = np.maximum(cluster_values - .5, 0.0)
        total = np.sum(support, axis=1)
        denominator = np.sum(support * support, axis=1)
        result["ind_effective_supporters"][start:stop] = np.divide(total * total, denominator, out=np.zeros(len(x)), where=denominator > 0)
        result["ind_cluster_mean"][start:stop] = np.nanmean(cluster_values, axis=1)
        result["ind_cluster_median"][start:stop] = np.nanmedian(cluster_values, axis=1)
        high = cluster_values >= .90
        result["ind_cluster_min_high90"][start:stop] = np.array([
            row[mask].min() if mask.any() else np.nan for row, mask in zip(cluster_values, high)
        ], dtype=np.float32)
        result["ind_cluster_strength"][start:stop] = total / max(len(groups), 1)
        result["ind_cluster_count"][start:stop] = float(len(groups))
    return pd.DataFrame(result, index=current.index)


def build_causal_6h_state(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build a compact 6-hour state table; do not widen the candidate panel.

    The state has only a few thousand rows.  Selected state columns are joined
    lazily into each fit/calibration/current matrix, which preserves the exact
    causal meaning while avoiding an unnecessary 2.7m-row expansion.
    """
    out = frame
    out["__bucket6h__"] = out["__decision_ts__"].dt.floor("6h")
    summary = out.groupby("__bucket6h__", sort=True).agg(
        status_base_mean=("base_score", "mean"),
        status_consensus_mean=("conditional_consensus_rank", "mean"),
        status_disagreement=("agr_rank_std", "mean"),
        status_rank_mad=("agr_rank_mad", "mean"),
        status_base_minus_consensus=("gap_base_conditional", "mean"),
        status_entropy=("agr_rank_entropy", "mean"),
        status_independent_support=("ind_effective_supporters", "mean"),
    )
    calendar = pd.date_range(summary.index.min(), summary.index.max(), freq="6h", tz="UTC")
    summary = summary.reindex(calendar)
    feature_columns: list[str] = []
    percentile_columns: list[str] = []
    # The requested 0.5/1/2 days mean, 3/7 day median and 7/21 day robust
    # dispersion; all are strictly shifted by one complete 6h summary.
    for source in list(summary.columns):
        prior = summary[source].shift(1)
        for label, buckets in (("05d", 2), ("1d", 4), ("2d", 8)):
            name = f"{source}_{label}_mean"
            summary[name] = prior.rolling(buckets, min_periods=max(2, buckets // 2)).mean()
            feature_columns.append(name)
        for label, buckets in (("3d", 12), ("7d", 28)):
            name = f"{source}_{label}_median"
            summary[name] = prior.rolling(buckets, min_periods=max(4, buckets // 3)).median()
            feature_columns.append(name)
        for label, buckets in (("7d", 28), ("21d", 84)):
            name = f"{source}_{label}_mad"
            summary[name] = prior.rolling(buckets, min_periods=max(8, buckets // 3)).apply(_mad, raw=True)
            feature_columns.append(name)
        for label, buckets in (("1d", 4), ("3d", 12), ("7d", 28), ("14d", 56)):
            name = f"{source}_priorpct_{label}"
            summary[name] = _rank_percentile_prior(summary[source], buckets)
            percentile_columns.append(name)
    # Slope/acceleration use timestamp-level summary, not candidate observations.
    for source in ("status_base_mean", "status_consensus_mean", "status_disagreement", "status_base_minus_consensus"):
        prior = summary[source].shift(1)
        slope3 = prior.rolling(12, min_periods=6).mean() - prior.rolling(56, min_periods=16).mean()
        slope7 = prior.rolling(28, min_periods=12).mean() - prior.rolling(84, min_periods=32).mean()
        for name, value in ((f"{source}_slope_3v14", slope3), (f"{source}_slope_7v21", slope7), (f"{source}_acceleration", slope3 - slope7)):
            summary[name] = value
            feature_columns.append(name)
    # Outcomes are first attached to their availability bucket.  Every
    # decision bucket accesses only *earlier* resolved-output summaries.
    valid = out.loc[
        out["policy_path_valid"].fillna(False).astype(bool) & out["policy_net_bps"].notna()
    ].copy()
    valid["__available_bucket__"] = valid["policy_label_available_ts"].dt.floor("6h")
    # Per-timestamp rank IC is calculated before availability aggregation.
    valid["__ts_base_ic__"] = valid.groupby("__decision_ts__", sort=False).apply(
        lambda group: group["base_score"].corr(group["policy_net_bps"], method="spearman") if len(group) >= 4 else np.nan,
        include_groups=False,
    ).reindex(valid["__decision_ts__"]).to_numpy()
    valid["__ts_consensus_ic__"] = valid.groupby("__decision_ts__", sort=False).apply(
        lambda group: group["conditional_consensus_rank"].corr(group["policy_net_bps"], method="spearman") if len(group) >= 4 else np.nan,
        include_groups=False,
    ).reindex(valid["__decision_ts__"]).to_numpy()
    # Surprise is event hit rate less a timestamp-local base-score top-quintile
    # proxy, avoiding reuse of a future outcome at the decision.
    valid["__hit__"] = (valid["policy_net_bps"] > 0.0).astype(float)
    valid["__top_score__"] = rank_fraction(valid, "base_score", .20).astype(float)
    perf = valid.groupby("__available_bucket__", sort=True).agg(
        perf_ev=("policy_net_bps", "mean"),
        perf_hit=("__hit__", "mean"),
        perf_base_ic=("__ts_base_ic__", "mean"),
        perf_consensus_ic=("__ts_consensus_ic__", "mean"),
        perf_top_score_hit=("__hit__", lambda s: float(s.mean())),
    ).reindex(calendar)
    # The following proxy is deliberately simple and causal: residual hit
    # relative to the contemporaneously resolved bucket average.
    perf["perf_hit_surprise"] = perf["perf_hit"] - perf["perf_hit"].shift(28).rolling(84, min_periods=28).mean()
    perf_columns: list[str] = []
    for source in list(perf.columns):
        prior = perf[source].shift(1)
        for label, buckets in (("3d", 12), ("7d", 28), ("14d", 56), ("28d", 112)):
            name = f"{source}_{label}_mean"
            perf[name] = prior.rolling(buckets, min_periods=max(4, buckets // 4)).mean()
            perf_columns.append(name)
        for label, buckets in (("3v14", (12, 56)), ("7v28", (28, 112))):
            name = f"{source}_slope_{label}"
            perf[name] = prior.rolling(buckets[0], min_periods=max(4, buckets[0] // 3)).mean() - prior.rolling(buckets[1], min_periods=max(12, buckets[1] // 3)).mean()
            perf_columns.append(name)
    state = pd.concat([summary[feature_columns + percentile_columns], perf[perf_columns]], axis=1)
    state = state.loc[:, ~state.columns.duplicated()].astype(np.float32)
    return state, feature_columns + percentile_columns, perf_columns


def attach_state(frame: pd.DataFrame, state: pd.DataFrame | None, features: Sequence[str]) -> pd.DataFrame:
    """Attach only state features which the immediate matrix actually needs."""
    if state is None:
        return frame
    required = [feature for feature in features if feature in state.columns and feature not in frame.columns]
    if not required:
        return frame
    bucket = frame["__decision_ts__"].dt.floor("6h")
    additions = pd.DataFrame(
        {feature: bucket.map(state[feature]).astype(np.float32) for feature in required},
        index=frame.index,
    )
    return pd.concat([frame, additions], axis=1, copy=False)


def add_pools(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame
    out["pool_base30"] = rank_fraction(out, "base_score", .30).to_numpy(bool)
    out["pool_consensus30"] = rank_fraction(out, "conditional_consensus_rank", .30).to_numpy(bool)
    out["pool_union30"] = out["pool_base30"] | out["pool_consensus30"]
    return out


def sample_train(frame: pd.DataFrame, cap: int, seed: int = SEED) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame
    # Equal-month subsampling prevents high-volume months from owning model fits.
    work = frame.copy()
    months = _month_key(work["__decision_ts__"])
    groups = list(work.groupby(months, sort=True))
    per = max(1, cap // len(groups))
    pieces = [group.sample(min(len(group), per), random_state=seed + idx) for idx, (_, group) in enumerate(groups)]
    sample = pd.concat(pieces, ignore_index=False)
    if len(sample) > cap:
        sample = sample.sample(cap, random_state=seed)
    return sample


def ordinal_contract(params: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    """Return a versioned ordinal target contract without global mutation.

    ``ordinal_spec`` contains finite internal edges and one numeric centre per
    resulting class.  The immutable default preserves the original six-bin
    MC1 target exactly; research callers can test finer partitions without
    altering another mapper's target semantics.
    """
    spec = params.get("ordinal_spec")
    if spec is None:
        return np.asarray(TARGET_EDGES, dtype=float), np.asarray(TARGET_CENTRES, dtype=float)
    if not isinstance(spec, Mapping):
        raise TypeError("ordinal_spec must be a mapping")
    inner = np.asarray(spec.get("internal_edges", ()), dtype=float)
    centres = np.asarray(spec.get("centres", ()), dtype=float)
    edges = np.concatenate(([-np.inf], inner, [np.inf]))
    if len(centres) != len(edges) - 1 or not np.isfinite(centres).all() or not np.all(np.diff(edges) > 0.0):
        raise ValueError("ordinal_spec needs strictly increasing finite internal_edges and one finite centre per class")
    return edges, centres


def regression_target(
    train: pd.DataFrame, kind: str, *, ordinal_edges: np.ndarray | None = None,
    ordinal_centres: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    y = pd.to_numeric(train["policy_net_bps"], errors="coerce").to_numpy(float)
    lo, hi = np.quantile(y, (.02, .98))
    clipped = np.clip(y, lo, hi)
    if kind.endswith("asin"):
        scale = max(abs(lo), abs(hi), 250.0)
        return np.arcsin(np.clip(clipped / scale, -.999, .999)), {"lo": float(lo), "hi": float(hi), "scale": float(scale)}
    if kind == "ordinal":
        edges = np.asarray(TARGET_EDGES if ordinal_edges is None else ordinal_edges, dtype=float)
        centres = np.asarray(TARGET_CENTRES if ordinal_centres is None else ordinal_centres, dtype=float)
        return np.digitize(y, edges[1:-1], right=True), {
            "lo": float(lo), "hi": float(hi), "ordinal_centres": centres.tolist(),
        }
    return clipped, {"lo": float(lo), "hi": float(hi)}


def build_matrix(train: pd.DataFrame, current: pd.DataFrame, features: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    medians = train.loc[:, features].apply(pd.to_numeric, errors="coerce").median(numeric_only=True)
    x = train.loc[:, features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(medians)
    z = current.loc[:, features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(medians)
    return x, z


@dataclass
class FittedModel:
    kind: str
    model: object
    features: tuple[str, ...]
    medians: pd.Series
    transform: Mapping[str, float]
    iso: IsotonicRegression | None

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame.loc[:, self.features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(self.medians)
        if self.kind == "ordinal":
            probability = self.model.predict_proba(x)
            centres = np.asarray(self.transform.get("ordinal_centres", TARGET_CENTRES), dtype=float)
            raw = probability.dot(centres)
        else:
            raw = np.asarray(self.model.predict(x), dtype=float)
            if self.kind.endswith("asin"):
                raw = np.sin(raw) * float(self.transform["scale"])
        return np.asarray(self.iso.predict(raw) if self.iso is not None else raw, dtype=float)


def fit_calibrated_model(
    train_fit: pd.DataFrame,
    calibration: pd.DataFrame,
    features: Sequence[str],
    kind: str,
    params: Mapping[str, float | int],
    seed: int,
) -> FittedModel:
    if train_fit.empty or calibration.empty:
        raise ValueError("strict prequential model needs both fit and calibration reserves")
    medians = train_fit.loc[:, features].apply(pd.to_numeric, errors="coerce").median(numeric_only=True)
    x = train_fit.loc[:, features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(medians)
    ordinal_edges, ordinal_centres = ordinal_contract(params) if kind == "ordinal" else (None, None)
    y, transform = regression_target(
        train_fit, kind, ordinal_edges=ordinal_edges, ordinal_centres=ordinal_centres,
    )
    common = dict(
        n_estimators=int(params.get("n_estimators", 700)),
        learning_rate=float(params["learning_rate"]),
        num_leaves=int(params["num_leaves"]),
        max_depth=int(params["max_depth"]),
        min_child_samples=int(params["min_child_samples"]),
        feature_fraction=float(params["feature_fraction"]),
        subsample=float(params.get("subsample", .8)),
        reg_lambda=float(params["reg_lambda"]),
        reg_alpha=float(params.get("reg_alpha", 0.0)),
        random_state=seed, n_jobs=min(8, os.cpu_count() or 2), verbosity=-1,
    )
    if kind == "ordinal":
        model: object = lgb.LGBMClassifier(objective="multiclass", num_class=len(ordinal_centres), **common)
        model.fit(x, y)
    else:
        objective = "regression_l1" if kind.startswith("l1") else "huber"
        model = lgb.LGBMRegressor(objective=objective, **common)
        # The separate 28-day calibration reserve must remain untouched by the
        # booster.  Use the fixed HPO ceiling here rather than pretending that
        # in-sample loss is an early-stopping validation set.
        model.fit(x, y)
    fitted = FittedModel(kind, model, tuple(features), medians, transform, None)
    raw = fitted.predict(calibration)
    observed = pd.to_numeric(calibration["policy_net_bps"], errors="coerce").to_numpy(float)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(raw, observed)
    fitted.iso = iso
    return fitted


def chronological_split(
    data: pd.DataFrame, start: pd.Timestamp, *, reserve_days: int = 28,
    pool: str, train_cap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reserve_start = start - pd.Timedelta(days=reserve_days)
    label_valid = data["policy_path_valid"].fillna(False).astype(bool) & data["policy_net_bps"].notna()
    eligible = data.loc[data["policy_label_available_ts"].lt(start) & data[pool] & label_valid].copy()
    fit = eligible.loc[eligible["policy_label_available_ts"].lt(reserve_start)].copy()
    calibration = eligible.loc[eligible["policy_label_available_ts"].ge(reserve_start)].copy()
    return sample_train(fit, train_cap), sample_train(calibration, min(80_000, train_cap // 2), seed=SEED + 11)


def robust21_expected(frame: pd.DataFrame) -> pd.Series:
    """Causal equal-day 15%-trim Robust-21 score map at each 6h decision.

    The implementation maintains only resolved outcomes keyed by their original
    decision date and 20 frozen-score cells.  This makes its previous-21-day
    convention explicit and prevents accidental same-bucket outcome use.
    """
    # A mapper blend only needs these columns; copying all selected geometry
    # features here would needlessly duplicate several GB of data.
    work = frame.loc[:, ["__decision_ts__", "policy_label_available_ts", "policy_path_valid", "policy_net_bps", "final_score"]].copy()
    work["__bucket6h__"] = work["__decision_ts__"].dt.floor("6h")
    work["__score_cell__"] = np.minimum(19, np.floor(pd.to_numeric(work["final_score"], errors="coerce") * 20)).astype("Int64")
    valid = work["policy_path_valid"].fillna(False).astype(bool) & work["policy_net_bps"].notna() & work["__score_cell__"].notna()
    events = work.loc[valid, ["policy_label_available_ts", "__decision_ts__", "__score_cell__", "policy_net_bps"]].copy()
    events["__available_bucket__"] = events["policy_label_available_ts"].dt.floor("6h")
    event_groups = {stamp: group for stamp, group in events.groupby("__available_bucket__", sort=True)}
    by_bucket = {stamp: group for stamp, group in work.groupby("__bucket6h__", sort=True)}
    running: dict[tuple[pd.Timestamp, int], list[float]] = {}
    output = pd.Series(np.nan, index=work.index, dtype=float)
    for bucket in sorted(by_bucket):
        # strictly earlier label buckets; the current one is appended afterwards.
        stale = [key for key in running if key[0] < (bucket - pd.Timedelta(days=21)).normalize()]
        for key in stale:
            del running[key]
        means = np.full(20, np.nan)
        supports = np.zeros(20, dtype=int)
        for cell in range(20):
            day_values = np.asarray([np.mean(value) for (day, present_cell), value in running.items() if present_cell == cell], dtype=float)
            if len(day_values):
                ordered = np.sort(day_values)
                trim = int(math.floor(.15 * len(ordered)))
                kept = ordered[trim:len(ordered) - trim] if trim and len(ordered) > 2 * trim else ordered
                means[cell] = float(np.mean(kept))
                supports[cell] = len(kept)
        usable = np.isfinite(means)
        if usable.sum() >= 2:
            xs = (np.arange(20) + .5) / 20.0
            means[usable] = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(
                xs[usable], means[usable], sample_weight=supports[usable],
            ).predict(xs[usable])
            means[~usable] = np.interp(xs[~usable], xs[usable], means[usable])
        group = by_bucket[bucket]
        cells = group["__score_cell__"].to_numpy(dtype=float, na_value=np.nan)
        selected = np.isfinite(cells) & np.isfinite(means[np.nan_to_num(cells, nan=0).astype(int)])
        values = np.full(len(group), np.nan)
        values[selected] = means[cells[selected].astype(int)]
        output.loc[group.index] = values
        arrived = event_groups.get(bucket)
        if arrived is not None:
            # Do not use named ``itertuples`` attributes here: pandas rewrites
            # double-underscore field names, which can silently detach a label
            # from its original decision date.  The explicit projection keeps
            # the causal attribution (decision day × score cell) exact.
            for decision_ts, score_cell, net_bps in arrived.loc[:, [
                "__decision_ts__", "__score_cell__", "policy_net_bps",
            ]].itertuples(index=False, name=None):
                key = (pd.Timestamp(decision_ts).normalize(), int(score_cell))
                running.setdefault(key, []).append(float(net_bps))
    return output.reindex(frame.index)


def auction(
    frame: pd.DataFrame, expected: str, auction_score: str, *, admission_threshold_bps: float = 50.0,
) -> pd.DataFrame:
    """Exact policy-shape proxy: two hourly entries, eight concurrent slots.

    Selection never uses policy label validity.  An invalid selected outcome
    occupies a conservative 12-hour slot and is kept as missing in EV metrics.
    """
    work = frame.copy()
    work["__eligible__"] = pd.to_numeric(work[expected], errors="coerce").ge(float(admission_threshold_bps))
    work = work.loc[work["__eligible__"]].sort_values(
        ["__decision_ts__", auction_score, "candidate_id"], ascending=[True, False, True], kind="stable",
    ).copy()
    active: list[tuple[pd.Timestamp, str]] = []
    decisions: list[bool] = []
    reasons: list[str] = []
    for stamp, group in work.groupby("__decision_ts__", sort=True):
        active = [(exit_stamp, symbol) for exit_stamp, symbol in active if exit_stamp > stamp]
        symbols = {symbol for _, symbol in active}
        chosen = 0
        symbols_now = group["__symbol__"].astype(str).to_numpy()
        valid_now = group["policy_path_valid"].fillna(False).to_numpy(bool)
        net_now = pd.to_numeric(group["policy_net_bps"], errors="coerce").to_numpy(float)
        bars_now = pd.to_numeric(group["policy_exit_bar_15m"], errors="coerce").to_numpy(float)
        for position in range(len(group)):
            symbol = symbols_now[position]
            if symbol in symbols:
                decisions.append(False); reasons.append("symbol_already_open"); continue
            if len(active) >= 8:
                decisions.append(False); reasons.append("max_concurrent_positions"); continue
            if chosen >= 2:
                decisions.append(False); reasons.append("max_new_entries_per_bar"); continue
            valid = bool(valid_now[position]) and np.isfinite(net_now[position])
            bars = float(bars_now[position]) if valid and np.isfinite(bars_now[position]) else 48.0
            exit_stamp = stamp + pd.Timedelta(minutes=15 * max(1.0, bars))
            active.append((exit_stamp, symbol)); symbols.add(symbol); chosen += 1
            decisions.append(True); reasons.append("accepted")
    work["portfolio_accepted"] = decisions
    work["portfolio_rejection_reason"] = reasons
    return work


def hpo_metric(
    frame: pd.DataFrame, expected: str, auction_score: str, *, admission_threshold_bps: float = 50.0,
) -> dict[str, float]:
    """Fast HPO objective: exact constraint shape, no diagnostic-only work."""
    replay = auction(frame, expected, auction_score, admission_threshold_bps=admission_threshold_bps)
    realised = replay.loc[
        replay.portfolio_accepted
        & replay.policy_path_valid.fillna(False).astype(bool)
        & replay.policy_net_bps.notna()
    ].copy()
    monthly = realised.groupby(_month_key(realised["__decision_ts__"])).policy_net_bps.mean()
    weekly = realised.groupby(realised["__decision_ts__"].dt.strftime("%G-W%V")).policy_net_bps.mean()
    return {
        "portfolio_net_ev_bps": float(realised.policy_net_bps.mean()) if len(realised) else -1e6,
        "worst_week_bps": float(weekly.min()) if len(weekly) else -1e6,
        "worst_month_bps": float(monthly.min()) if len(monthly) else -1e6,
        "accepted": float(len(realised)),
    }


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    paired = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(paired) < 4 or paired.left.nunique() < 2 or paired.right.nunique() < 2:
        return float("nan")
    return float(paired.left.corr(paired.right, method="spearman"))


def metric_row(
    frame: pd.DataFrame, expected: str, auction_score: str, *, admission_threshold_bps: float = 50.0,
) -> dict[str, float]:
    replay = auction(frame, expected, auction_score, admission_threshold_bps=admission_threshold_bps)
    accepted = replay.loc[replay.portfolio_accepted].copy()
    valid_all = frame.loc[frame.policy_path_valid.fillna(False).astype(bool) & frame.policy_net_bps.notna()].copy()
    admitted = valid_all.loc[
        pd.to_numeric(valid_all[expected], errors="coerce").ge(float(admission_threshold_bps))
    ].copy()
    realised = accepted.loc[accepted.policy_path_valid.fillna(False).astype(bool) & accepted.policy_net_bps.notna()].copy()
    ic_values = admitted.groupby("__decision_ts__", sort=False).apply(
        lambda group: _safe_spearman(group[expected], group.policy_net_bps), include_groups=False,
    ).dropna()
    if len(admitted) >= 8 and admitted[expected].nunique() > 1:
        slope, intercept = np.polyfit(admitted[expected].to_numpy(float), admitted.policy_net_bps.to_numpy(float), 1)
    else:
        slope = intercept = np.nan
    monthly = realised.groupby(_month_key(realised["__decision_ts__"])).policy_net_bps.mean()
    weekly = realised.groupby(realised["__decision_ts__"].dt.strftime("%G-W%V")).policy_net_bps.mean()
    # Candidate-level contention answers whether the auction score chooses the
    # better side of timestamps where more than two candidates cleared EV.
    contenders = admitted.groupby("__decision_ts__", sort=False).filter(lambda group: len(group) > 2)
    selected_ids = set(accepted.candidate_id)
    contested_selected = contenders.loc[contenders.candidate_id.isin(selected_ids)]
    contested_rejected = contenders.loc[~contenders.candidate_id.isin(selected_ids)]
    return {
        "candidate_rows": float(len(frame)), "admitted_valid_rows": float(len(admitted)),
        "portfolio_selected_rows": float(len(accepted)), "selected_label_coverage": float(len(realised) / len(accepted)) if len(accepted) else np.nan,
        "portfolio_net_ev_bps": float(realised.policy_net_bps.mean()) if len(realised) else np.nan,
        "portfolio_net_sum_bps": float(realised.policy_net_bps.sum()) if len(realised) else np.nan,
        "within_admission_ic": float(ic_values.mean()) if len(ic_values) else np.nan,
        "calibration_slope": float(slope), "calibration_intercept": float(intercept),
        "worst_month_bps": float(monthly.min()) if len(monthly) else np.nan,
        "worst_week_bps": float(weekly.min()) if len(weekly) else np.nan,
        "positive_month_fraction": float((monthly > 0).mean()) if len(monthly) else np.nan,
        "contested_selected_ev_bps": float(contested_selected.policy_net_bps.mean()) if len(contested_selected) else np.nan,
        "contested_rejected_ev_bps": float(contested_rejected.policy_net_bps.mean()) if len(contested_rejected) else np.nan,
    }


def forward_selection_gate(
    candidate: Mapping[str, float],
    incumbent: Mapping[str, float],
    *,
    max_worst_month_degradation_bps: float = 10.0,
    max_worst_week_degradation_bps: float = 15.0,
) -> dict[str, float | bool]:
    """Gate one cumulative MC1 feature against the *retained* contract.

    This deliberately differs from an experiment-versus-baseline table: a
    feature must add portfolio net contribution beyond the features already
    retained, while keeping worst-period degradation inside the declared
    tolerance.  It prevents an individually baseline-positive but redundant
    field from entering the next forward-selection step.
    """
    delta_sum = float(candidate["portfolio_net_sum_bps"] - incumbent["portfolio_net_sum_bps"])
    delta_month = float(candidate["worst_month_bps"] - incumbent["worst_month_bps"])
    delta_week = float(candidate["worst_week_bps"] - incumbent["worst_week_bps"])
    return {
        "delta_portfolio_net_sum_bps_vs_incumbent": delta_sum,
        "delta_worst_month_bps_vs_incumbent": delta_month,
        "delta_worst_week_bps_vs_incumbent": delta_week,
        "total_positive_vs_incumbent": delta_sum > 0.0,
        "worst_month_ge_guardrail": delta_month >= -max_worst_month_degradation_bps,
        "worst_week_ge_guardrail": delta_week >= -max_worst_week_degradation_bps,
        "keep": (
            delta_sum > 0.0
            and delta_month >= -max_worst_month_degradation_bps
            and delta_week >= -max_worst_week_degradation_bps
        ),
    }


def hpo_params(trial: optuna.Trial) -> dict[str, float | int]:
    return {
        "n_estimators": 700,
        "learning_rate": trial.suggest_float("learning_rate", .015, .075),
        "num_leaves": trial.suggest_int("num_leaves", 7, 31),
        "max_depth": trial.suggest_int("max_depth", 2, 4),
        "min_child_samples": trial.suggest_int("min_child_samples", 150, 1_200),
        "feature_fraction": trial.suggest_float("feature_fraction", .65, .95),
        "subsample": trial.suggest_float("subsample", .70, .95),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 3.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", .1, 30.0, log=True),
    }


def prequential_prediction_blocks(
    panel: pd.DataFrame,
    features: Sequence[str],
    kind: str,
    params: Mapping[str, float | int],
    pool: str,
    starts: Sequence[pd.Timestamp],
    months: int,
    train_cap: int,
    add_independence: bool,
    conditional_columns: Sequence[str],
    state: pd.DataFrame | None = None,
    retain_features: Sequence[str] = (),
    block_output_dir: Path | None = None,
) -> pd.DataFrame | list[Path]:
    outputs: list[pd.DataFrame] = []
    output_paths: list[Path] = []
    if block_output_dir is not None:
        block_output_dir.mkdir(parents=True, exist_ok=False)
    for fold_index, start in enumerate(starts):
        stop = start + pd.DateOffset(months=months)
        current = panel.loc[panel["__decision_ts__"].between(start, stop, inclusive="left")].copy()
        if current.empty:
            continue
        fit, calibration = chronological_split(panel, start, pool=pool, train_cap=train_cap)
        if len(fit) < 10_000 or len(calibration) < 1_000:
            continue
        if add_independence:
            indep = _independence_features(fit, current, conditional_columns)
            current = current.join(indep)
            fit = fit.join(_independence_features(fit, fit, conditional_columns))
            calibration = calibration.join(_independence_features(fit, calibration, conditional_columns))
        fit = attach_state(fit, state, features)
        calibration = attach_state(calibration, state, features)
        current = attach_state(current, state, list(dict.fromkeys([*features, *retain_features])))
        model = fit_calibrated_model(fit, calibration, features, kind, params, SEED + fold_index)
        current["mapper_expected_bps"] = model.predict(current)
        current["fold_start"] = start
        current["mapper_kind"] = kind
        current["mapper_pool"] = pool
        # Preserve only the selected feature contract plus replay lineage.  The
        # full candidate panel stays resident for the next fold instead of
        # being duplicated inside each saved prediction block.
        keep = list(dict.fromkeys([
            "candidate_id", "__decision_ts__", "__symbol__", "side_name",
            "policy_label_available_ts", "policy_path_valid", "policy_net_bps",
            "policy_exit_bar_15m", "final_score", "pool_base30", "pool_consensus30", "pool_union30",
            *features, *retain_features, "mapper_expected_bps", "fold_start", "mapper_kind", "mapper_pool",
        ]))
        projection = current.loc[:, keep].copy()
        if block_output_dir is not None:
            path = block_output_dir / f"fold_{start.strftime('%Y%m%dT%H%M%SZ')}.parquet"
            projection.to_parquet(path, index=False, compression="zstd")
            output_paths.append(path)
            # The fold objects can otherwise survive until the next iteration
            # through sklearn/LightGBM references.  Explicitly release them
            # after checkpointing so a long chronological replay has bounded
            # peak memory rather than accumulating one fit per block.
            del projection, model, fit, calibration, current
            gc.collect()
        else:
            outputs.append(projection)
    if not outputs:
        if output_paths:
            return output_paths
        raise ValueError("no prequential blocks had a usable fit/calibration history")
    return pd.concat(outputs, ignore_index=True)


def mi_screen(
    panel: pd.DataFrame, features: Sequence[str], families: Mapping[str, Sequence[str]],
    train_cap: int, state: pd.DataFrame,
) -> pd.DataFrame:
    """Binned conditional-MI proxy against an OOF MC1-shaped residual.

    Rather than use a contemporaneous outcome mean, the residual is produced
    by a prequential six-feature Huber control over four 2025 development
    blocks.  MI is then evaluated inside final-score deciles, month by month.
    """
    control = {"n_estimators": 450, "learning_rate": .04, "num_leaves": 15, "max_depth": 2, "min_child_samples": 500, "feature_fraction": .85, "subsample": .85, "reg_alpha": .02, "reg_lambda": 5.0}
    starts = [utc(value) for value in ("2025-01-01", "2025-04-01", "2025-07-01", "2025-10-01")]
    static = [feature for feature in features if feature in panel.columns]
    prediction = prequential_prediction_blocks(
        panel, BASE_FEATURES, "huber_clip", control, "pool_consensus30", starts, 3,
        train_cap, False, (), state=state, retain_features=static,
    )
    prediction = prediction.loc[prediction.policy_path_valid.fillna(False).astype(bool) & prediction.policy_net_bps.notna()].copy()
    prediction["mc1_proxy_residual"] = prediction.policy_net_bps - prediction.mapper_expected_bps
    prediction["__score_bin__"] = np.minimum(9, np.floor(prediction.final_score * 10)).astype(int)
    lookup = {feature: family for family, items in families.items() for feature in items}
    rows: list[dict[str, object]] = []
    bucket = prediction["__decision_ts__"].dt.floor("6h")
    for feature in features:
        # Keep the 6h state compact until this individual CMI calculation.
        # Static agreement geometry is retained on the candidate prediction;
        # state fields are mapped lazily and immediately released.
        if feature in prediction.columns:
            value = pd.to_numeric(prediction[feature], errors="coerce")
        else:
            value = pd.to_numeric(bucket.map(state[feature]), errors="coerce")
        monthly_values: list[float] = []
        for month_index, (_, month) in enumerate(prediction.assign(__month__=_month_key(prediction["__decision_ts__"])).groupby("__month__", sort=True)):
            valid = value.loc[month.index].notna() & month.mc1_proxy_residual.notna()
            if valid.sum() < 400:
                continue
            conditioned: list[float] = []
            for bin_index, (_, group) in enumerate(month.loc[valid].groupby("__score_bin__", sort=True)):
                x = value.loc[group.index]
                if len(group) < 80 or x.nunique() < 5:
                    continue
                # The selection criterion is portability, not a high-precision
                # one-month estimate.  Equal score-bin subsampling avoids both
                # dominant high-volume months and an O(features x millions)
                # KNN MI computation.
                if len(group) > 750:
                    group = group.sample(750, random_state=SEED + month_index * 31 + bin_index)
                    x = value.loc[group.index]
                bins = pd.qcut(x, q=min(12, x.nunique()), labels=False, duplicates="drop")
                if bins.nunique() < 2:
                    continue
                conditioned.append(float(mutual_info_regression(
                    pd.DataFrame({"x": bins.astype(float)}), group.mc1_proxy_residual.to_numpy(float),
                    discrete_features=True, random_state=SEED,
                )[0]))
            if conditioned:
                monthly_values.append(float(np.mean(conditioned)))
        if monthly_values:
            med = float(np.median(monthly_values))
            rows.append({"feature": feature, "family": lookup[feature], "cmi_binned_median": med,
                         "cmi_binned_mad": float(np.median(np.abs(np.asarray(monthly_values) - med))),
                         "months": len(monthly_values)})
    return pd.DataFrame(rows).sort_values(["cmi_binned_median", "cmi_binned_mad"], ascending=[False, True])


def choose_representatives(mi: pd.DataFrame, *, per_family: int = 3) -> list[str]:
    chosen: list[str] = []
    # Portable threshold avoids selecting a feature that wins only one month.
    eligible = mi.loc[(mi.cmi_binned_median > 0.001) & (mi.cmi_binned_mad <= mi.cmi_binned_median * 1.5)]
    for _, group in eligible.groupby("family", sort=True):
        chosen.extend(group.sort_values(["cmi_binned_median", "cmi_binned_mad"], ascending=[False, True]).head(per_family).feature.tolist())
    return chosen


def add_independence_all(panel: pd.DataFrame, conditional: Sequence[str]) -> tuple[pd.DataFrame, list[str]]:
    """Adds a point-in-time-independent pre-2025 clustering diagnostic.

    It is used only for screening and family ablations.  Production candidates
    are re-derived fold by fold in `prequential_prediction_blocks`.
    """
    definition = panel.loc[panel["__decision_ts__"] < DEV_START]
    derived = _independence_features(definition, panel, conditional)
    output = panel
    for column in derived:
        output[column] = derived[column].astype(np.float32)
    del derived
    gc.collect()
    columns = ["ind_effective_supporters", "ind_cluster_mean", "ind_cluster_median", "ind_cluster_min_high90", "ind_cluster_strength", "ind_cluster_count"]
    return output, columns


def run_hpo(
    panel: pd.DataFrame, features: Sequence[str], kind: str, pool: str, trials: int, train_cap: int,
    state: pd.DataFrame,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    starts = [utc(value) for value in ("2025-04-01", "2025-07-01", "2025-10-01")]
    records: list[dict[str, object]] = []
    def objective(trial: optuna.Trial) -> float:
        params = hpo_params(trial)
        values: list[float] = []
        for index, start in enumerate(starts):
            stop = start + pd.DateOffset(months=3)
            test = panel.loc[panel["__decision_ts__"].between(start, stop, inclusive="left")].copy()
            fit, calibration = chronological_split(panel, start, pool=pool, train_cap=train_cap)
            if len(fit) < 10_000 or len(calibration) < 1_000:
                continue
            fit = attach_state(fit, state, features)
            calibration = attach_state(calibration, state, features)
            test = attach_state(test, state, features)
            model = fit_calibrated_model(fit, calibration, features, kind, params, SEED + index)
            test["mapper_expected_bps"] = model.predict(test)
            metric = hpo_metric(test.loc[test[pool]], "mapper_expected_bps", "final_score")
            # Lexicographic intent translated to a conservative scalar: do not
            # win HPO by sacrificing +50 admission EV or a bad weekly period.
            value = metric["portfolio_net_ev_bps"] - .35 * max(0.0, -metric["worst_week_bps"])
            values.append(value)
            # HPO reuses the same chronological panels many times.  Dropping
            # each fitted booster and its three frame copies here keeps the
            # search bounded rather than retaining one full fold per trial.
            del model, test, fit, calibration
            gc.collect()
            trial.report(float(np.nanmedian(values)), index)
            if trial.should_prune():
                raise optuna.TrialPruned()
        score = float(np.nanmedian(values)) if values else -1e9
        records.append({"trial": trial.number, "kind": kind, "pool": pool, "score": score, **params})
        return score
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED), pruner=optuna.pruners.MedianPruner(n_startup_trials=3))
    study.optimize(objective, n_trials=trials, gc_after_trial=True)
    return {key: value for key, value in study.best_params.items()} | {"n_estimators": 700}, pd.DataFrame(records)


def test_blends(prediction: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    # The blend never needs the selected feature matrix retained for the later
    # LambdaRank challenger.  Keeping only replay lineage prevents a second
    # large copy of the consensus prediction panel.
    core = list(dict.fromkeys([
        "candidate_id", "__decision_ts__", "__symbol__", "side_name",
        "policy_label_available_ts", "policy_path_valid", "policy_net_bps",
        "policy_exit_bar_15m", "final_score", "pool_consensus30",
        "mapper_expected_bps",
    ]))
    work = prediction.loc[:, core].copy()
    work["robust21_expected_bps"] = robust21_expected(work)
    rows: list[dict[str, object]] = []
    for weight in (0.0, .25, .50, .75, 1.0):
        column = f"blend_mc1_{int(weight * 100):03d}"
        work[column] = weight * work.mapper_expected_bps + (1.0 - weight) * work.robust21_expected_bps
        for auction_score in ("final_score", column):
            for year in (2025, 2026):
                part = work.loc[work.__decision_ts__.dt.year.eq(year) & work.pool_consensus30].copy()
                metric = metric_row(part, column, auction_score)
                rows.append({"arm": f"mc1={weight:.2f}|auction={auction_score}", "year": year, "mc1_weight": weight, "auction_score": auction_score, **metric})
    work.to_parquet(out_dir / "regression_blend_predictions.parquet", index=False, compression="zstd")
    return pd.DataFrame(rows)


def two_stage_ranker(prediction: pd.DataFrame, features: Sequence[str], out_dir: Path) -> pd.DataFrame:
    """Strict two-stage challenger: regression EV admits; LambdaRank auctions.

    The ranker sees only prequential first-stage outputs.  It is trained on
    2025 OOF records and assessed on 2026, so this is intentionally a single
    forward architectural test rather than a hidden retrain-on-test result.
    """
    # A caller may retain lineage fields that also appear in the selected MC1
    # contract (notably ``final_score``).  LightGBM refuses duplicated column
    # names, so canonicalise the projection at this downstream boundary.
    prediction = prediction.loc[:, ~prediction.columns.duplicated()].copy()
    train = prediction.loc[prediction.__decision_ts__.dt.year.eq(2025) & prediction.pool_consensus30].copy()
    test = prediction.loc[prediction.__decision_ts__.dt.year.eq(2026) & prediction.pool_consensus30].copy()
    if train.empty or test.empty:
        return pd.DataFrame()
    rank_features = list(dict.fromkeys(["mapper_expected_bps", *features]))
    train_x, test_x = build_matrix(train, test, rank_features)
    labels = np.digitize(train.policy_net_bps.to_numpy(float), TARGET_EDGES[1:-1], right=True)
    ordered = train.assign(__label__=labels).sort_values(["__decision_ts__", "candidate_id"], kind="stable")
    ordered_x = train_x.loc[ordered.index]
    groups = ordered.groupby("__decision_ts__", sort=False).size().to_numpy(int)
    ranker = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", ndcg_eval_at=[1, 2, 5],
        n_estimators=500, learning_rate=.03, num_leaves=15, max_depth=3,
        min_child_samples=500, feature_fraction=.8, reg_lambda=5.0,
        lambdarank_truncation_level=8, label_gain=[0, 1, 2, 4, 7, 10],
        random_state=SEED, n_jobs=min(8, os.cpu_count() or 2), verbosity=-1,
    )
    # The 2026 rows are the forward test and cannot be used as an early-stop
    # set.  Keep the frozen fixed tree ceiling for this first-stage challenger.
    ranker.fit(ordered_x, ordered["__label__"], group=groups)
    test["two_stage_rank"] = ranker.predict(test_x)
    rows: list[dict[str, object]] = []
    for year in (2026,):
        part = test.loc[test.__decision_ts__.dt.year.eq(year)].copy()
        rows.append({"arm": "two_stage_regression_plus_lambdarank", "year": year, **metric_row(part, "mapper_expected_bps", "two_stage_rank")})
    test.to_parquet(out_dir / "two_stage_predictions_2026.parquet", index=False, compression="zstd")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--heads", type=Path, default=DEFAULT_HEADS)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, default=None,
                        help="immutable prepare-stage artifact; skips raw-ledger reconstruction")
    parser.add_argument("--screen-dir", type=Path, default=None,
                        help="immutable screen-stage artifact; skips recomputing the CMI selector")
    parser.add_argument("--hpo-dir", type=Path, default=None,
                        help="immutable HPO-stage artifact; skips recomputing HPO before replay")
    parser.add_argument("--hpo-trials", type=int, default=8)
    parser.add_argument("--max-train-rows", type=int, default=180_000)
    parser.add_argument("--stage", choices=("all", "prepare", "screen", "hpo", "replay"), default="all")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"output path must be immutable: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    preselected_contract: dict[str, object] | None = None
    if args.screen_dir is not None:
        contract_path = args.screen_dir.resolve() / "selected_feature_contract.json"
        if not contract_path.exists():
            raise FileNotFoundError("screen-dir lacks selected_feature_contract.json")
        preselected_contract = json.loads(contract_path.read_text())
    if args.prepared_dir is not None:
        prepared = args.prepared_dir.resolve()
        panel_path = prepared / "candidate_static_panel.parquet"
        state_path = prepared / "causal_6h_state.parquet"
        prepared_manifest = prepared / "run_manifest.json"
        if not panel_path.exists() or not state_path.exists() or not prepared_manifest.exists():
            raise FileNotFoundError("prepared-dir lacks the immutable panel/state/manifest trio")
        print(f"[mc1-v2] load immutable prepared panel {prepared}", flush=True)
        # Once a selector is frozen, load only the actual model/replay contract
        # from parquet.  Dropping 130 fields after a whole-file read does not
        # reliably return allocator memory before consensus OOF generation.
        if preselected_contract is not None:
            static_features = set(preselected_contract["features"])
            core_columns = {
                "candidate_id", "__decision_ts__", "__symbol__", "side_name",
                "policy_label_available_ts", "policy_path_valid", "policy_net_bps",
                "policy_exit_bar_15m", "pool_base30", "pool_consensus30", "pool_union30",
            }
            available = set(pq.ParquetFile(panel_path).schema_arrow.names)
            selected_columns = sorted((core_columns | static_features).intersection(available))
            panel = pd.read_parquet(panel_path, columns=selected_columns)
        else:
            panel = pd.read_parquet(panel_path)
        state = pd.read_parquet(state_path)
        prior = json.loads(prepared_manifest.read_text())
        conditional = list(prior["conditional_heads"])
        ordinary = list(prior.get("ordinary_shadow_heads", ()))
        families = {str(key): list(value) for key, value in prior["feature_families"].items()}
        status = list(families["causal_recent_status"])
        performance = list(families["causal_recent_performance"])
    else:
        print("[mc1-v2] load compact primary and ten-head panels", flush=True)
        panel, conditional, ordinary = load_panel(args.ledger, args.heads)
        # MC1 is only ever evaluated on candidates that can reach it.  Retaining
        # the target-free base/consensus union gives the requested consensus-top30
        # expansion and a matched base-top30 control, while excluding the 40%+
        # population that no ablation can admit or auction.
        panel = add_pools(panel)
        panel = panel.loc[panel["pool_union30"]].copy()
        gc.collect()
        print(f"[mc1-v2] loaded {len(panel):,} rows; derive static agreement geometry", flush=True)
        panel, families = add_static_agreement_geometry(panel, conditional, ordinary)
        print("[mc1-v2] derive stable independence aggregates", flush=True)
        panel, independent = add_independence_all(panel, conditional)
        families["independence_weighted"] = independent
        # All downstream features are semantic aggregates.  Retaining ten raw
        # rank columns after this point only consumes memory and invites accidental
        # use as unstable named specialists.
        panel.drop(columns=list(conditional), inplace=True)
        gc.collect()
        print("[mc1-v2] derive compact causal six-hour state", flush=True)
        state, status, performance = build_causal_6h_state(panel)
        families["causal_recent_status"] = status
        families["causal_recent_performance"] = performance
    print(f"[mc1-v2] state rows={len(state):,}; candidate pools complete", flush=True)
    if args.stage == "prepare":
        state.to_parquet(args.out_dir / "causal_6h_state.parquet", compression="zstd")
        panel.to_parquet(args.out_dir / "candidate_static_panel.parquet", index=False, compression="zstd")
        _manifest(args, panel, conditional, ordinary, list(BASE_FEATURES), "prepare_complete", families=families)
        return
    # All model fits exclude absent/invalid supervised outcomes.  Candidate
    # pools and current score features remain target-free until this point.
    if args.screen_dir is not None:
        screen_root = args.screen_dir.resolve()
        contract = preselected_contract
        assert contract is not None
        feature_set = list(contract["features"])
        (args.out_dir / "selected_feature_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
        print(f"[mc1-v2] reuse frozen selector {screen_root}", flush=True)
    else:
        static_candidates = [column for items in families.values() for column in items if column not in BASE_FEATURES]
        print("[mc1-v2] construct prequential MC1-shaped residual and CMI screen", flush=True)
        screen = mi_screen(panel, static_candidates, families, args.max_train_rows, state)
        screen.to_parquet(args.out_dir / "feature_binned_cmi_prequential_residual.parquet", index=False)
        selected = choose_representatives(screen)
        # Base features are never removed.  This is the first, deliberately
        # compact representative set used for HPO and family additions.
        feature_set = list(dict.fromkeys([*BASE_FEATURES, *selected]))
        (args.out_dir / "selected_feature_contract.json").write_text(json.dumps({
            "schema": "strict_r3_mc1_admission_ablation_v2_feature_contract",
            "selection": "prequential MC1-shaped residual binned conditional MI; median monthly MI, low MAD; up to three per family",
            "features": feature_set, "families": families,
        }, indent=2) + "\n")
    if args.stage == "screen":
        _manifest(args, panel, conditional, ordinary, feature_set, "screen_complete", families=families)
        return
    hpo_records: list[pd.DataFrame] = []
    winners: dict[str, dict[str, float | int]] = {}
    if args.hpo_dir is not None:
        hpo_root = args.hpo_dir.resolve()
        winner_path = hpo_root / "hpo_winners.json"
        trial_path = hpo_root / "hpo_trials.parquet"
        if not winner_path.exists() or not trial_path.exists():
            raise FileNotFoundError("hpo-dir lacks hpo_winners.json or hpo_trials.parquet")
        winners = json.loads(winner_path.read_text())
        hpo_records = [pd.read_parquet(trial_path)]
        print(f"[mc1-v2] reuse frozen 2025 HPO {hpo_root}", flush=True)
    else:
        # 2025-only HPO on the intended extended top-30 consensus domain.  A
        # base-top30 control is run with the same loss alternatives.
        for pool in ("pool_consensus30", "pool_base30"):
            for kind in ("huber_clip", "huber_asin", "l1_clip", "l1_asin", "ordinal"):
                print(f"[mc1-v2] HPO {pool} {kind}", flush=True)
                params, trials = run_hpo(panel, feature_set, kind, pool, args.hpo_trials, args.max_train_rows, state)
                winners[f"{pool}|{kind}"] = params
                hpo_records.append(trials)
        pd.concat(hpo_records, ignore_index=True).to_parquet(args.out_dir / "hpo_trials.parquet", index=False)
        (args.out_dir / "hpo_winners.json").write_text(json.dumps(winners, indent=2) + "\n")
    if args.stage == "hpo":
        _manifest(args, panel, conditional, ordinary, feature_set, "hpo_complete", families=families)
        return
    # Selection is complete.  Do not carry the other 130+ screened fields into
    # OOF replay: they are neither part of the frozen contract nor needed to
    # derive its compact six-hour state.  In-place dropping keeps one source
    # panel resident while the consensus prediction blocks are accumulated.
    replay_required = set([
        "candidate_id", "__decision_ts__", "__symbol__", "side_name",
        "policy_label_available_ts", "policy_path_valid", "policy_net_bps",
        "policy_exit_bar_15m", "pool_base30", "pool_consensus30", "pool_union30",
        *feature_set,
    ])
    drop_for_replay = [column for column in panel.columns if column not in replay_required]
    if drop_for_replay:
        panel.drop(columns=drop_for_replay, inplace=True)
        gc.collect()
    # Sequential promotion here remains development-only: choose the HPO
    # winner per pool from 2025 trial score, then replay both 2025 and 2026
    # quarterly prequential blocks.  2026 is not used to pick a winner.
    trial_table = pd.concat(hpo_records, ignore_index=True)
    winner_rows = trial_table.sort_values("score", ascending=False, kind="stable").groupby("pool", as_index=False).head(1)
    consensus_prediction: pd.DataFrame | None = None
    metric_rows: list[dict[str, object]] = []
    starts = list(pd.date_range("2025-01-01", "2026-07-01", freq="3MS", tz="UTC"))
    for row in winner_rows.itertuples(index=False):
        params = winners[f"{row.pool}|{row.kind}"]
        prediction = prequential_prediction_blocks(panel, feature_set, row.kind, params, row.pool, starts, 3, args.max_train_rows, False, conditional, state=state)
        prediction["selected_hpo_pool"] = row.pool
        prediction["selected_hpo_kind"] = row.kind
        for auction_score in ("final_score", "mapper_expected_bps"):
            for year in (2025, 2026):
                part = prediction.loc[prediction.__decision_ts__.dt.year.eq(year) & prediction[row.pool]].copy()
                metric_rows.append({"arm": f"regression|{row.pool}|{row.kind}|auction={auction_score}", "year": year, "pool": row.pool, "kind": row.kind, "auction_score": auction_score, **metric_row(part, "mapper_expected_bps", auction_score)})
        if str(row.pool) == "pool_base30":
            # Stream this matched control immediately.  It never participates
            # in the feature-rich LambdaRank stage, so retaining its feature
            # matrix until consensus scoring is pure memory pressure.
            core_control = list(dict.fromkeys([
                "candidate_id", "__decision_ts__", "__symbol__", "side_name",
                "policy_label_available_ts", "policy_path_valid", "policy_net_bps",
                "policy_exit_bar_15m", "final_score", "pool_base30", "pool_consensus30",
                "pool_union30", "mapper_expected_bps", "fold_start", "mapper_kind",
                "mapper_pool", "selected_hpo_pool", "selected_hpo_kind",
            ]))
            prediction.loc[:, core_control].to_parquet(
                args.out_dir / "prequential_base_pool_control_predictions.parquet", index=False, compression="zstd",
            )
            del prediction
            gc.collect()
        else:
            consensus_prediction = prediction
    # The extended consensus winner is the only arm eligible for blend/ranker
    # analysis; it is selected from 2025 HPO only.
    consensus_winner = winner_rows.loc[winner_rows.pool.eq("pool_consensus30")].iloc[0]
    if consensus_prediction is None:
        raise RuntimeError("the frozen 2025 HPO did not yield a consensus-pool winner")
    del panel, state
    gc.collect()
    print("[mc1-v2] base control persisted; blend consensus with causal Robust-21", flush=True)
    blend = test_blends(consensus_prediction, args.out_dir)
    print("[mc1-v2] fit two-stage LambdaRank challenger", flush=True)
    ranker = two_stage_ranker(consensus_prediction, feature_set, args.out_dir)
    metric_rows.extend(blend.to_dict("records"))
    metric_rows.extend(ranker.to_dict("records"))
    pd.DataFrame(metric_rows).to_parquet(args.out_dir / "ablation_metrics.parquet", index=False)
    consensus_prediction.to_parquet(args.out_dir / "prequential_consensus_predictions.parquet", index=False, compression="zstd")
    _manifest(args, panel, conditional, ordinary, feature_set, "complete", extra={
        "2025_hpo_winners": winner_rows.to_dict("records"),
        "selection_note": "2026 reports are opened validation/model-selection evidence, never untouched promotion evidence",
        "six_hour_state": "timestamp summaries first, then prior-only six-hour rolling / resolved-performance fields",
        "two_stage": "ranker trained only on 2025 prequential first-stage predictions and evaluated on 2026",
    }, families=families)


def _manifest(args: argparse.Namespace, panel: pd.DataFrame, conditional: Sequence[str], ordinary: Sequence[str], features: Sequence[str], status: str, extra: Mapping[str, object] | None = None, families: Mapping[str, Sequence[str]] | None = None) -> None:
    payload: dict[str, object] = {
        "schema": "strict_r3_mc1_admission_ablation_v2",
        "status": status,
        "purpose": "offline, causal MC1 admissions optimization; does not alter the sealed live MC1_d2 artifact",
        "ledger": str(args.ledger), "ledger_sha256": sha256(args.ledger),
        "heads": str(args.heads), "heads_sha256": sha256(args.heads),
        "script_sha256": sha256(Path(__file__)), "panel_rows": len(panel),
        "period": {"start": str(panel.__decision_ts__.min()), "end": str(panel.__decision_ts__.max())},
        "conditional_heads": list(conditional), "ordinary_shadow_heads": list(ordinary),
        "feature_contract": list(features), "hpo_development": "2025 chronological three-month folds", "validation": "2026 chronological three-month folds",
        "supervision": "policy_path_valid and policy_net_bps, with policy_label_available_ts strictly before fold start",
        "calibration": "28-day pre-fold resolved reserve; model fit excludes reserve",
        "pool": "base top-30 / consensus top-30 / union controls are timestamp-local and target-free",
        "portfolio": "matched long-only 2-new-entry / 8-concurrent proxy; outcome-invalid selections occupy 12h and are not erased",
    }
    if families is not None:
        payload["feature_families"] = {str(key): list(value) for key, value in families.items()}
    if extra:
        payload.update(extra)
    (args.out_dir / "run_manifest.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
