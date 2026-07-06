#!/usr/bin/env python3
"""Materialize second-generation causal source archetypes.

This script consumes the existing candidate source-tag artifact and builds a
diagnostic-only v2 archetype layer. Archetype scores are constructed only from
prediction-time source scores/tags and lagged same-symbol history. Realized
outcomes are joined only after the v2 artifact has been written, for quality
diagnostics and promotion/readiness reporting.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_LABELS_DIR,
    _json_safe,
    _load_labels,
    _path_metrics,
    _safe_mean,
    _safe_quantile,
    _spearman,
)
from scripts.run_source_quality_label_walkforward_ablation import _load_joined_frame  # noqa: E402


DEFAULT_SOURCE_TAGS = Path(
    "data_perp/reports/source_tags_s10_policy_net_v17_proxy_alignment_diagnostic/candidate_source_tags.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1_sideaware_20260702/source_archetypes_v2"
)

OUTCOME_LIKE_RE = re.compile(
    r"(future|fwd|mfe|mae|pnl|profit|utility|target|label|oracle|hit|timeout|realized|outcome|barrier_result|__y_|__r_policy|__u_policy)",
    re.IGNORECASE,
)

SCORE_COLS_USED = (
    "base_positive_source_score",
    "calm_positive_source_score",
    "risk_adjusted_capture_candidate_score",
    "clean_economic_capture_candidate_score",
    "quiet_continuation_score",
    "compression_capture_candidate_score",
    "loud_clean_source_score",
    "dirty_shock_avoid_score",
    "misleading_location_risk_score",
    "prior_recent_source_strength",
    "run_entry_score",
    "clean_run_entry_score",
    "late_run_continuation_score",
    "execution_quality_score",
    "barrier_relief_score",
    "clean_execution_context_score",
    "location_quality_score",
    "barrier_pressure_score",
    "shock_impulse_score",
    "compression_score",
    "trend_path_score",
    "oi_agreement_score",
    "volume_confirmation_score",
)

TAG_COLS_USED = (
    "tag_dirty_shock_avoid",
    "tag_misleading_location_risk",
    "tag_late_run_continuation",
    "tag_ambiguous_none",
)

V2_SCORE_COLS = (
    "source_evidence_archetype_score",
    "source_independence_archetype_score",
    "source_freshness_archetype_score",
    "path_geometry_archetype_score",
    "timeout_holding_archetype_score",
    "regime_archetype_score",
    "symbol_behavior_archetype_score",
)

V2_TAGS = tuple(f"tag_{col.replace('_score', '')}" for col in V2_SCORE_COLS)
SIDE_CONTRACT_COLUMNS = ("side", "side_name", "__side__", "timeframe", "candidate_id")

PRIMARY_PRIORITY = (
    ("tag_path_geometry_archetype", "path_geometry_archetype"),
    ("tag_timeout_holding_archetype", "timeout_holding_archetype"),
    ("tag_source_freshness_archetype", "source_freshness_archetype"),
    ("tag_source_independence_archetype", "source_independence_archetype"),
    ("tag_source_evidence_archetype", "source_evidence_archetype"),
    ("tag_symbol_behavior_archetype", "symbol_behavior_archetype"),
    ("tag_regime_archetype", "regime_archetype"),
)


@dataclass(frozen=True)
class ThresholdSpec:
    score_col: str
    tag_col: str
    top_frac: float


THRESHOLDS = (
    ThresholdSpec("source_evidence_archetype_score", "tag_source_evidence_archetype", 0.20),
    ThresholdSpec("source_independence_archetype_score", "tag_source_independence_archetype", 0.25),
    ThresholdSpec("source_freshness_archetype_score", "tag_source_freshness_archetype", 0.20),
    ThresholdSpec("path_geometry_archetype_score", "tag_path_geometry_archetype", 0.25),
    ThresholdSpec("timeout_holding_archetype_score", "tag_timeout_holding_archetype", 0.25),
    ThresholdSpec("regime_archetype_score", "tag_regime_archetype", 0.25),
    ThresholdSpec("symbol_behavior_archetype_score", "tag_symbol_behavior_archetype", 0.20),
)


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _clip01(values: Any) -> pd.Series:
    return _safe_numeric(values).clip(0.0, 1.0)


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".gz"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported source tags path: {path}")


def _timestamp_key(values: Any) -> pd.Series:
    ts = pd.to_datetime(values, utc=True, errors="coerce")
    return ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalise_side_contract(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    numeric = pd.Series(np.nan, index=out.index, dtype=np.float32)
    if "side" in out.columns:
        numeric = _safe_numeric(out["side"])
    elif "__side__" in out.columns:
        numeric = _safe_numeric(out["__side__"])
    if "side_name" in out.columns:
        text = out["side_name"].fillna("").astype(str).str.strip().str.lower()
        numeric = numeric.where(~text.isin({"short", "sell"}), -1.0)
        numeric = numeric.where(~text.isin({"long", "buy"}), 1.0)
    valid = numeric.notna() & numeric.ne(0.0)
    if not bool(valid.any()):
        return pd.DataFrame(index=out.index)
    out["side"] = np.where(numeric.fillna(1.0) < 0.0, -1, 1).astype(np.int8)
    out["side_name"] = np.where(out["side"].to_numpy(dtype=np.int8) < 0, "short", "long")
    out["__side__"] = out["side"]
    if "timeframe" in out.columns:
        out["timeframe"] = out["timeframe"].fillna("").astype(str)
    if "candidate_id" in out.columns:
        out["candidate_id"] = out["candidate_id"].fillna("").astype(str)
    return out


def _load_label_side_contract(labels_path: Path, *, timestamp_col: str, symbol_col: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    labels = _load_labels(labels_path)
    if timestamp_col not in labels.columns or symbol_col not in labels.columns:
        return pd.DataFrame(), {
            "side_contract_source": str(labels_path),
            "side_contract_rows": 0,
            "side_contract_available": False,
            "side_contract_reason": "missing_timestamp_or_symbol",
        }
    cols = [timestamp_col, symbol_col, *[col for col in SIDE_CONTRACT_COLUMNS if col in labels.columns]]
    contract = labels.loc[:, list(dict.fromkeys(cols))].copy()
    contract[timestamp_col] = pd.to_datetime(contract[timestamp_col], utc=True, errors="coerce")
    contract[symbol_col] = contract[symbol_col].astype(str)
    contract = _normalise_side_contract(contract)
    if contract.empty:
        return contract, {
            "side_contract_source": str(labels_path),
            "side_contract_rows": 0,
            "side_contract_available": False,
            "side_contract_reason": "missing_side",
        }
    if "timeframe" not in contract.columns or not contract["timeframe"].astype(str).str.len().gt(0).any():
        contract["timeframe"] = "1h"
    if "candidate_id" not in contract.columns or not contract["candidate_id"].astype(str).str.len().gt(0).any():
        contract["candidate_id"] = (
            contract[symbol_col].astype(str)
            + "|"
            + _timestamp_key(contract[timestamp_col])
            + "|"
            + contract["timeframe"].astype(str)
            + "|"
            + contract["side_name"].astype(str)
        )
    contract = contract[contract[timestamp_col].notna() & contract[symbol_col].astype(str).str.len().gt(0)]
    contract = contract[contract["candidate_id"].astype(str).str.len().gt(0)]
    contract = contract.drop_duplicates(["candidate_id"], keep="last")
    keep = [timestamp_col, symbol_col, *SIDE_CONTRACT_COLUMNS]
    contract = contract.loc[:, [col for col in keep if col in contract.columns]].reset_index(drop=True)
    return contract, {
        "side_contract_source": str(labels_path),
        "side_contract_rows": int(len(contract)),
        "side_contract_available": bool(len(contract)),
        "side_contract_reason": "ok" if len(contract) else "empty_after_filter",
    }


def _expand_materialized_with_side_contract(
    materialized: pd.DataFrame,
    *,
    labels_path: Path,
    timestamp_col: str,
    symbol_col: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if {"side", "side_name", "timeframe", "candidate_id"} <= set(materialized.columns):
        return materialized, {
            "side_contract_source": "source_tags",
            "side_contract_rows": int(len(materialized)),
            "side_contract_available": True,
            "side_contract_reason": "already_materialized",
            "side_contract_expanded": False,
        }
    contract, report = _load_label_side_contract(labels_path, timestamp_col=timestamp_col, symbol_col=symbol_col)
    if contract.empty:
        report["side_contract_expanded"] = False
        return materialized, report
    base = materialized.copy()
    base[timestamp_col] = pd.to_datetime(base[timestamp_col], utc=True, errors="coerce")
    base[symbol_col] = base[symbol_col].astype(str)
    contract_cols = [
        col
        for col in [timestamp_col, symbol_col, *SIDE_CONTRACT_COLUMNS]
        if col in contract.columns and col not in base.columns
    ]
    expanded = base.merge(
        contract[[timestamp_col, symbol_col, *[col for col in contract_cols if col not in {timestamp_col, symbol_col}]]],
        on=[timestamp_col, symbol_col],
        how="inner",
        validate="one_to_many",
    )
    report.update(
        {
            "side_contract_expanded": True,
            "side_contract_input_rows": int(len(materialized)),
            "side_contract_output_rows": int(len(expanded)),
            "side_contract_match_rate_vs_materialized": float(len(expanded) / len(materialized))
            if len(materialized)
            else 0.0,
        }
    )
    return expanded, report


def _score(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(float(default), index=frame.index, dtype=np.float32)
    return _clip01(frame[col]).fillna(float(default)).astype(np.float32)


def _bool_score(frame: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(1.0 if default else 0.0, index=frame.index, dtype=np.float32)
    values = frame[col]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(default).astype(float).astype(np.float32)
    lowered = values.astype(str).str.lower()
    return lowered.isin({"1", "true", "t", "yes", "y"}).astype(float).astype(np.float32)


def _timestamp_rank(frame: pd.DataFrame, score: pd.Series, timestamp_col: str) -> pd.Series:
    ranks = score.groupby(frame[timestamp_col], dropna=False).rank(method="average", pct=True)
    return ranks.fillna(0.0).clip(0.0, 1.0).astype(np.float32)


def _tag_from_timestamp_rank(
    frame: pd.DataFrame,
    *,
    score_col: str,
    tag_col: str,
    top_frac: float,
    timestamp_col: str,
    min_timestamp_rows: int,
) -> int:
    score = _safe_numeric(frame[score_col])
    group_sizes = score.groupby(frame[timestamp_col], dropna=False).transform("size")
    ranks = score.groupby(frame[timestamp_col], dropna=False).rank(method="average", pct=True)
    eligible = group_sizes.ge(int(min_timestamp_rows)) & ranks.notna()
    frame[tag_col] = (eligible & ranks.ge(1.0 - float(top_frac))).astype(bool)
    return int((~eligible).sum())


def _lagged_symbol_event_density(
    frame: pd.DataFrame,
    *,
    symbol_col: str,
    timestamp_col: str,
    strength: pd.Series,
    window: int,
) -> pd.Series:
    temp = pd.DataFrame(
        {
            "__orig__": np.arange(len(frame), dtype=np.int64),
            "__symbol__": frame[symbol_col].astype(str).to_numpy(),
            "__ts__": pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce").to_numpy(),
            "__event__": _clip01(strength).ge(0.60).astype(float).to_numpy(),
        }
    ).sort_values(["__symbol__", "__ts__", "__orig__"], kind="mergesort")
    temp["__lag_event__"] = temp.groupby("__symbol__", sort=False)["__event__"].shift(1)
    temp["__density__"] = (
        temp.groupby("__symbol__", sort=False)["__lag_event__"]
        .rolling(int(window), min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    out = pd.Series(0.0, index=frame.index, dtype=np.float32)
    out.iloc[temp["__orig__"].to_numpy(dtype=np.int64)] = temp["__density__"].fillna(0.0).to_numpy(dtype=np.float32)
    return out.clip(0.0, 1.0)


def _weighted_sum(terms: list[tuple[float, pd.Series]]) -> pd.Series:
    out = pd.Series(0.0, index=terms[0][1].index, dtype=np.float64)
    for weight, values in terms:
        out = out + float(weight) * _safe_numeric(values).fillna(0.0)
    return out.clip(0.0, 1.0).astype(np.float32)


def _validate_source_inputs() -> None:
    offenders = [col for col in SCORE_COLS_USED + TAG_COLS_USED if OUTCOME_LIKE_RE.search(col)]
    allowed = {"barrier_pressure_score", "barrier_relief_score"}
    offenders = [col for col in offenders if col not in allowed]
    if offenders:
        raise ValueError(f"Outcome-like columns are not allowed in archetype v2 construction: {offenders}")


def _reason_codes(frame: pd.DataFrame) -> pd.Series:
    reasons: list[list[str]] = [[] for _ in range(len(frame))]
    mapping = (
        ("tag_source_evidence_archetype", "source_evidence_stack"),
        ("tag_source_independence_archetype", "independent_source"),
        ("tag_source_freshness_archetype", "fresh_symbol_run"),
        ("tag_path_geometry_archetype", "clean_path_geometry"),
        ("tag_timeout_holding_archetype", "low_timeout_holding_proxy"),
        ("tag_regime_archetype", "regime_fit"),
        ("tag_symbol_behavior_archetype", "clean_symbol_behavior"),
    )
    for tag_col, code in mapping:
        if tag_col not in frame.columns:
            continue
        mask = frame[tag_col].fillna(False).astype(bool).to_numpy()
        for idx in np.flatnonzero(mask):
            reasons[int(idx)].append(code)
    return pd.Series(["|".join(items) if items else "archetype_v2_none" for items in reasons], index=frame.index)


def build_archetypes_v2(
    source_tags: pd.DataFrame,
    *,
    timestamp_col: str,
    symbol_col: str,
    min_timestamp_rows: int,
    prior_symbol_window: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _validate_source_inputs()
    frame = source_tags.copy()
    if timestamp_col not in frame.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_col}")
    if symbol_col not in frame.columns:
        raise ValueError(f"Missing symbol column: {symbol_col}")
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
    frame[symbol_col] = frame[symbol_col].astype(str)
    frame = frame.sort_values([timestamp_col, symbol_col], kind="mergesort").reset_index(drop=True)

    base_positive = _score(frame, "base_positive_source_score")
    calm_positive = _score(frame, "calm_positive_source_score")
    risk_adjusted = _score(frame, "risk_adjusted_capture_candidate_score")
    clean_economic = _score(frame, "clean_economic_capture_candidate_score")
    quiet = _score(frame, "quiet_continuation_score")
    compression_capture = _score(frame, "compression_capture_candidate_score")
    loud_clean = _score(frame, "loud_clean_source_score")
    dirty = _score(frame, "dirty_shock_avoid_score")
    misleading = _score(frame, "misleading_location_risk_score")
    prior_recent = _score(frame, "prior_recent_source_strength")
    run_entry = _score(frame, "run_entry_score")
    clean_run_entry = _score(frame, "clean_run_entry_score")
    late_run = _score(frame, "late_run_continuation_score")
    execution_quality = _score(frame, "execution_quality_score")
    barrier_relief = _score(frame, "barrier_relief_score")
    clean_execution = _score(frame, "clean_execution_context_score")
    location_quality = _score(frame, "location_quality_score")
    barrier_pressure = _score(frame, "barrier_pressure_score")
    shock_impulse = _score(frame, "shock_impulse_score")
    compression = _score(frame, "compression_score")
    trend_path = _score(frame, "trend_path_score")
    oi_agreement = _score(frame, "oi_agreement_score")
    volume_confirmation = _score(frame, "volume_confirmation_score")
    ambiguous_tag = _bool_score(frame, "tag_ambiguous_none")

    prior_symbol_density = _lagged_symbol_event_density(
        frame,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
        strength=base_positive,
        window=prior_symbol_window,
    )
    frame["prior_symbol_event_density_score"] = prior_symbol_density
    frame["prior_symbol_event_density_rank"] = _timestamp_rank(frame, prior_symbol_density, timestamp_col)

    source_evidence = _weighted_sum(
        [
            (0.22, base_positive),
            (0.16, calm_positive),
            (0.16, risk_adjusted),
            (0.14, clean_economic),
            (0.10, quiet),
            (0.10, compression_capture),
            (0.12, loud_clean),
            (-0.12, dirty),
            (-0.08, misleading),
        ]
    )
    source_independence = _weighted_sum(
        [
            (0.28, 1.0 - prior_recent),
            (0.20, run_entry),
            (0.16, clean_run_entry),
            (0.16, 1.0 - late_run),
            (0.10, 1.0 - dirty),
            (0.06, 1.0 - ambiguous_tag),
            (0.04, 1.0 - prior_symbol_density),
        ]
    )
    source_freshness = _weighted_sum(
        [
            (0.34, run_entry),
            (0.24, clean_run_entry),
            (0.18, 1.0 - prior_recent),
            (0.14, source_evidence),
            (0.10, 1.0 - prior_symbol_density),
            (-0.20, late_run),
        ]
    )
    path_geometry = _weighted_sum(
        [
            (0.26, execution_quality),
            (0.22, barrier_relief),
            (0.18, clean_execution),
            (0.14, location_quality),
            (0.12, 1.0 - barrier_pressure),
            (0.08, 1.0 - misleading),
            (-0.14, dirty),
        ]
    )
    timeout_holding = _weighted_sum(
        [
            (0.24, path_geometry),
            (0.18, source_freshness),
            (0.18, execution_quality),
            (0.14, 1.0 - late_run),
            (0.10, 1.0 - shock_impulse),
            (0.08, compression),
            (0.08, 1.0 - prior_symbol_density),
            (-0.16, barrier_pressure),
        ]
    )
    regime_fit = _weighted_sum(
        [
            (0.24, trend_path),
            (0.18, volume_confirmation),
            (0.18, execution_quality),
            (0.14, oi_agreement),
            (0.12, 1.0 - shock_impulse),
            (0.10, compression),
            (0.04, source_evidence),
            (-0.10, barrier_pressure),
        ]
    )
    symbol_behavior = _weighted_sum(
        [
            (0.26, source_independence),
            (0.24, source_freshness),
            (0.18, clean_run_entry),
            (0.14, 1.0 - late_run),
            (0.10, 1.0 - prior_symbol_density),
            (0.08, 1.0 - ambiguous_tag),
        ]
    )

    frame["source_evidence_archetype_score"] = source_evidence
    frame["source_independence_archetype_score"] = source_independence
    frame["source_freshness_archetype_score"] = source_freshness
    frame["path_geometry_archetype_score"] = path_geometry
    frame["timeout_holding_archetype_score"] = timeout_holding
    frame["regime_archetype_score"] = regime_fit
    frame["symbol_behavior_archetype_score"] = symbol_behavior

    low_timestamp_rows: dict[str, int] = {}
    for spec in THRESHOLDS:
        low_timestamp_rows[spec.tag_col] = _tag_from_timestamp_rank(
            frame,
            score_col=spec.score_col,
            tag_col=spec.tag_col,
            top_frac=spec.top_frac,
            timestamp_col=timestamp_col,
            min_timestamp_rows=min_timestamp_rows,
        )

    primary = pd.Series("archetype_v2_none", index=frame.index, dtype=object)
    assigned = pd.Series(False, index=frame.index)
    for tag_col, value in PRIMARY_PRIORITY:
        mask = frame[tag_col].fillna(False).astype(bool) & ~assigned
        primary.loc[mask] = value
        assigned.loc[mask] = True
    frame["primary_source_archetype_v2"] = primary
    frame["archetype_v2_reason_codes"] = _reason_codes(frame)

    source_columns_used = [col for col in SCORE_COLS_USED + TAG_COLS_USED if col in frame.columns]
    report = {
        "rows": int(len(frame)),
        "date_min": frame[timestamp_col].min().isoformat() if len(frame) else None,
        "date_max": frame[timestamp_col].max().isoformat() if len(frame) else None,
        "symbols": int(frame[symbol_col].nunique()),
        "source_columns_used": source_columns_used,
        "missing_source_columns": [col for col in SCORE_COLS_USED + TAG_COLS_USED if col not in frame.columns],
        "low_timestamp_rank_rows": low_timestamp_rows,
        "min_timestamp_rows": int(min_timestamp_rows),
        "prior_symbol_window": int(prior_symbol_window),
    }
    return frame, report


def _week_start(ts: pd.Series) -> pd.Series:
    return (
        pd.to_datetime(ts, utc=True, errors="coerce")
        .dt.tz_convert(None)
        .dt.to_period("W-SUN")
        .apply(lambda value: value.start_time.date().isoformat() if pd.notna(value) else "")
    )


def _effective_n(values: Any) -> float:
    counts = pd.Series(values, dtype=object).value_counts(dropna=False)
    if counts.empty:
        return 0.0
    shares = counts.to_numpy(dtype=np.float64) / float(counts.sum())
    denom = float(np.sum(shares * shares))
    return 1.0 / denom if denom > 0.0 else 0.0


def _side_name_series(frame: pd.DataFrame) -> pd.Series:
    if "side_name" in frame.columns:
        values = frame["side_name"].fillna("").astype(str).str.strip().str.lower()
        mapped = values.where(values.isin({"long", "short"}), "")
        if mapped.ne("").any():
            return mapped[mapped.ne("")]
    if "side" in frame.columns:
        numeric = _safe_numeric(frame["side"])
    elif "__side__" in frame.columns:
        numeric = _safe_numeric(frame["__side__"])
    else:
        return pd.Series(dtype=object)
    numeric = numeric.dropna()
    if numeric.empty:
        return pd.Series(dtype=object)
    return pd.Series(np.where(numeric < 0.0, "short", "long"), index=numeric.index, dtype=object)


def _metric_summary(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    mask: pd.Series,
) -> dict[str, Any]:
    selected = metrics.loc[mask]
    selected_frame = frame.loc[mask]
    symbols = selected_frame.get("__symbol__", pd.Series(dtype=object))
    side_names = _side_name_series(selected_frame)
    side_counts = side_names.value_counts(normalize=True) if len(side_names) else pd.Series(dtype=float)
    return {
        "rows": int(mask.sum()),
        "mean_utility": _safe_mean(selected.get("u_policy_net")),
        "median_utility": _safe_quantile(selected.get("u_policy_net"), 0.50),
        "p25_utility": _safe_quantile(selected.get("u_policy_net"), 0.25),
        "hit_utility_rate": _safe_mean(selected.get("u_policy_net") > 0.0),
        "bad_mae_1r_rate": _safe_mean(selected.get("mae_norm") >= 1.0),
        "p90_mae_norm": _safe_quantile(selected.get("mae_norm"), 0.90),
        "timeout_rate": _safe_mean(selected.get("is_timeout").astype(float)) if len(selected) else float("nan"),
        "wide_barrier_25bps_rate": _safe_mean(selected.get("barrier") > 0.025),
        "wide_barrier_35bps_rate": _safe_mean(selected.get("barrier") > 0.035),
        "mean_barrier": _safe_mean(selected.get("barrier")),
        "top_symbol_share": float(symbols.value_counts(normalize=True, dropna=False).iloc[0]) if len(symbols) else 0.0,
        "symbol_effective_n": _effective_n(symbols),
        "unique_symbols": int(symbols.nunique(dropna=False)) if len(symbols) else 0,
        "top_side": str(side_counts.index[0]) if len(side_counts) else "",
        "top_side_share": float(side_counts.iloc[0]) if len(side_counts) else float("nan"),
        "side_effective_n": _effective_n(side_names) if len(side_names) else float("nan"),
        "long_share": float((side_names.eq("long")).mean()) if len(side_names) else float("nan"),
        "short_share": float((side_names.eq("short")).mean()) if len(side_names) else float("nan"),
    }


def _quality_by_group(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    group_col: str,
    scope_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if group_col not in frame.columns:
        return pd.DataFrame()
    groups = frame[group_col].astype(str) if group_col == "primary_source_archetype_v2" else frame[group_col].fillna(False).astype(bool)
    if group_col == "primary_source_archetype_v2":
        for value, idx in groups.groupby(groups, dropna=False).groups.items():
            mask = pd.Series(False, index=frame.index)
            mask.loc[idx] = True
            rows.append({"scope": scope_name, "archetype": str(value), **_metric_summary(frame=frame, metrics=metrics, mask=mask)})
    else:
        mask = groups
        rows.append(
            {
                "scope": scope_name,
                "archetype": group_col.replace("tag_", ""),
                **_metric_summary(frame=frame, metrics=metrics, mask=mask),
            }
        )
    return pd.DataFrame(rows)


def _quality_over_time(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    period_col: str,
    period_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for tag_col in V2_TAGS:
        if tag_col not in frame.columns:
            continue
        for period, idx in frame.groupby(period_col, dropna=False, observed=True).groups.items():
            base_mask = pd.Series(False, index=frame.index)
            base_mask.loc[idx] = True
            tag_mask = base_mask & frame[tag_col].fillna(False).astype(bool)
            total = int(base_mask.sum())
            row = {
                period_name: str(period),
                "scope": "multi_tag",
                "archetype": tag_col.replace("tag_", ""),
                "period_rows": total,
                "coverage": float(tag_mask.sum() / total) if total else 0.0,
                **_metric_summary(frame=frame, metrics=metrics, mask=tag_mask),
            }
            rows.append(row)
    if "primary_source_archetype_v2" in frame.columns:
        for (period, archetype), idx in frame.groupby([period_col, "primary_source_archetype_v2"], dropna=False, observed=True).groups.items():
            base_mask = frame[period_col].astype(str).eq(str(period))
            tag_mask = pd.Series(False, index=frame.index)
            tag_mask.loc[idx] = True
            total = int(base_mask.sum())
            rows.append(
                {
                    period_name: str(period),
                    "scope": "primary",
                    "archetype": str(archetype),
                    "period_rows": total,
                    "coverage": float(tag_mask.sum() / total) if total else 0.0,
                    **_metric_summary(frame=frame, metrics=metrics, mask=tag_mask),
                }
            )
    return pd.DataFrame(rows)


def _scorecard(
    *,
    archetypes: pd.DataFrame,
    joined: pd.DataFrame,
    metrics: pd.DataFrame,
    monthly: pd.DataFrame,
    timestamp_col: str,
    symbol_col: str,
    min_coverage: float,
    max_top_symbol_share: float,
) -> pd.DataFrame:
    overall = _metric_summary(frame=joined, metrics=metrics, mask=pd.Series(True, index=joined.index))
    rows: list[dict[str, Any]] = []
    full_total = len(archetypes)
    joined_total = len(joined)
    month_overall = (
        monthly[monthly["scope"].eq("multi_tag")]
        .groupby("month", dropna=False, observed=True)["period_rows"]
        .max()
        .to_dict()
        if not monthly.empty
        else {}
    )
    for spec in THRESHOLDS:
        tag_col = spec.tag_col
        score_col = spec.score_col
        full_mask = archetypes[tag_col].fillna(False).astype(bool)
        joined_mask = joined[tag_col].fillna(False).astype(bool) if tag_col in joined.columns else pd.Series(False, index=joined.index)
        summary = _metric_summary(frame=joined, metrics=metrics, mask=joined_mask)
        monthly_tag = monthly[(monthly["scope"].eq("multi_tag")) & (monthly["archetype"].eq(tag_col.replace("tag_", "")))]
        coverage_cv = float(monthly_tag["coverage"].std(ddof=0) / monthly_tag["coverage"].mean()) if len(monthly_tag) and monthly_tag["coverage"].mean() else float("nan")
        min_month_coverage = _safe_quantile(monthly_tag.get("coverage", pd.Series(dtype=float)), 0.0)
        monthly_delta_u = []
        for _, row in monthly_tag.iterrows():
            period = str(row["month"])
            period_rows = float(month_overall.get(period, 0.0) or 0.0)
            if period_rows <= 0:
                continue
            period_key = (
                pd.to_datetime(joined[timestamp_col], utc=True, errors="coerce")
                .dt.tz_convert(None)
                .dt.to_period("M")
                .astype(str)
            )
            period_frame = joined[period_key.eq(period)]
            period_metrics = metrics.loc[period_frame.index]
            period_overall_u = _safe_mean(period_metrics["u_policy_net"])
            if math.isfinite(period_overall_u) and math.isfinite(float(row.get("mean_utility", float("nan")))):
                monthly_delta_u.append(float(row["mean_utility"]) - period_overall_u)
        stable_sign_months = 0
        if monthly_delta_u:
            signs = [1 if value > 0 else -1 if value < 0 else 0 for value in monthly_delta_u]
            stable_sign_months = max(signs.count(1), signs.count(-1), signs.count(0))
        score = _safe_numeric(joined[score_col]) if score_col in joined.columns else pd.Series(dtype=float)
        utility_ic = _spearman(score, metrics["u_policy_net"]) if len(score) else float("nan")
        timeout_ic = _spearman(score, metrics["is_timeout"].astype(float)) if len(score) else float("nan")
        bad_mae_ic = _spearman(score, metrics["mae_norm"] >= 1.0) if len(score) else float("nan")
        wide_ic = _spearman(score, metrics["barrier"] > 0.025) if len(score) else float("nan")
        diffs = {
            "mean_utility_diff": summary["mean_utility"] - overall["mean_utility"]
            if math.isfinite(summary["mean_utility"]) and math.isfinite(overall["mean_utility"])
            else float("nan"),
            "bad_mae_diff": summary["bad_mae_1r_rate"] - overall["bad_mae_1r_rate"]
            if math.isfinite(summary["bad_mae_1r_rate"]) and math.isfinite(overall["bad_mae_1r_rate"])
            else float("nan"),
            "timeout_diff": summary["timeout_rate"] - overall["timeout_rate"]
            if math.isfinite(summary["timeout_rate"]) and math.isfinite(overall["timeout_rate"])
            else float("nan"),
            "wide_barrier_diff": summary["wide_barrier_25bps_rate"] - overall["wide_barrier_25bps_rate"]
            if math.isfinite(summary["wide_barrier_25bps_rate"]) and math.isfinite(overall["wide_barrier_25bps_rate"])
            else float("nan"),
        }
        coverage = float(full_mask.sum() / full_total) if full_total else 0.0
        joined_coverage = float(joined_mask.sum() / joined_total) if joined_total else 0.0
        economic_distinction = max(
            [
                abs(value)
                for value in [
                    diffs["mean_utility_diff"] / 0.005 if math.isfinite(diffs["mean_utility_diff"]) else float("nan"),
                    diffs["bad_mae_diff"],
                    diffs["timeout_diff"],
                    diffs["wide_barrier_diff"],
                ]
                if math.isfinite(value)
            ]
            or [0.0]
        )
        has_material_difference = (
            (math.isfinite(diffs["mean_utility_diff"]) and abs(diffs["mean_utility_diff"]) >= 0.001)
            or (math.isfinite(diffs["bad_mae_diff"]) and abs(diffs["bad_mae_diff"]) >= 0.03)
            or (math.isfinite(diffs["timeout_diff"]) and abs(diffs["timeout_diff"]) >= 0.02)
            or (math.isfinite(diffs["wide_barrier_diff"]) and abs(diffs["wide_barrier_diff"]) >= 0.02)
        )
        has_score_relation = any(
            math.isfinite(value) and abs(value) >= 0.03
            for value in (utility_ic, timeout_ic, bad_mae_ic, wide_ic)
        )
        if coverage < min_coverage:
            decision = "too_sparse"
        elif summary["top_symbol_share"] > max_top_symbol_share:
            decision = "diagnostic_symbol_concentrated"
        elif has_material_difference and (has_score_relation or stable_sign_months >= 2):
            decision = "candidate_feature"
        else:
            decision = "diagnostic_only"
        rows.append(
            {
                "decision": decision,
                "archetype": tag_col.replace("tag_", ""),
                "score_col": score_col,
                "rows_full": int(full_mask.sum()),
                "coverage_full": coverage,
                "rows_joined": int(joined_mask.sum()),
                "coverage_joined": joined_coverage,
                "min_month_coverage": min_month_coverage,
                "monthly_coverage_cv": coverage_cv,
                "stable_sign_months": int(stable_sign_months),
                "score_ic_utility": utility_ic,
                "score_ic_timeout": timeout_ic,
                "score_ic_bad_mae": bad_mae_ic,
                "score_ic_wide_barrier": wide_ic,
                **summary,
                **diffs,
                "economic_distinction_score": economic_distinction,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["decision", "economic_distinction_score", "coverage_full"],
        ascending=[True, False, False],
        na_position="last",
        kind="mergesort",
    )


def _regime_matrix(joined: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    regime_cols = [
        col
        for col in [
            "G_VOL",
            "__regime_vol_12h__",
            "__regime_vol_48h__",
            "__regime_volume_12h__",
            "__regime_volume_48h__",
            "__regime_trend_12h__",
            "__regime_trend_48h__",
        ]
        if col in joined.columns
    ]
    rows: list[dict[str, Any]] = []
    for regime_col in regime_cols:
        for (regime_value, primary), idx in joined.groupby([regime_col, "primary_source_archetype_v2"], dropna=False, observed=True).groups.items():
            mask = pd.Series(False, index=joined.index)
            mask.loc[idx] = True
            regime_mask = joined[regime_col].astype(str).eq(str(regime_value))
            total = int(regime_mask.sum())
            rows.append(
                {
                    "regime_col": regime_col,
                    "regime_value": str(regime_value),
                    "primary_source_archetype_v2": str(primary),
                    "regime_rows": total,
                    "source_concentration_within_regime": float(mask.sum() / total) if total else 0.0,
                    **_metric_summary(frame=joined, metrics=metrics, mask=mask),
                }
            )
    return pd.DataFrame(rows)


def _table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
    if frame.empty:
        return "No rows."
    view = frame[[col for col in cols if col in frame.columns]].copy()
    if limit is not None:
        view = view.head(limit)
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: f"{float(value):.4f}" if pd.notna(value) else "")
    return view.to_markdown(index=False)


def _write_report(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    scorecard: pd.DataFrame,
    quality: pd.DataFrame,
    monthly: pd.DataFrame,
) -> Path:
    path = output_dir / "source_archetypes_v2_report.md"
    cols = [
        "decision",
        "archetype",
        "coverage_full",
        "coverage_joined",
        "min_month_coverage",
        "monthly_coverage_cv",
        "score_ic_utility",
        "score_ic_timeout",
        "score_ic_bad_mae",
        "score_ic_wide_barrier",
        "mean_utility",
        "mean_utility_diff",
        "bad_mae_1r_rate",
        "bad_mae_diff",
        "timeout_rate",
        "timeout_diff",
        "wide_barrier_25bps_rate",
        "wide_barrier_diff",
        "top_symbol_share",
        "top_side",
        "top_side_share",
        "stable_sign_months",
    ]
    quality_cols = [
        "scope",
        "archetype",
        "rows",
        "mean_utility",
        "bad_mae_1r_rate",
        "timeout_rate",
        "wide_barrier_25bps_rate",
        "p90_mae_norm",
        "top_symbol_share",
        "top_side",
        "top_side_share",
        "unique_symbols",
    ]
    month_cols = [
        "month",
        "scope",
        "archetype",
        "coverage",
        "rows",
        "mean_utility",
        "bad_mae_1r_rate",
        "timeout_rate",
        "wide_barrier_25bps_rate",
        "top_symbol_share",
        "top_side",
        "top_side_share",
    ]
    lines = [
        "# Candidate Source Archetypes V2",
        "",
        "Diagnostic-only v2 archetype layer. Scores/tags are causal; realized outcomes are used only after materialization for reporting.",
        "",
        f"Rows: `{manifest['rows']}`",
        f"Joined outcome rows: `{manifest['join_report']['joined_rows']}`",
        f"Utility source: `{manifest.get('utility_source', '')}`",
        f"Date range: `{manifest['date_min']}` to `{manifest['date_max']}`",
        f"Symbols: `{manifest['symbols']}`",
        f"Side counts full/joined: `{manifest.get('side_counts_full', {})}` / `{manifest.get('side_counts_joined', {})}`",
        "",
        "## Leakage Audit",
        "",
        f"Source columns used: `{', '.join(manifest['source_columns_used'])}`",
        f"Missing source columns: `{', '.join(manifest['missing_source_columns']) or 'none'}`",
        f"Low timestamp-rank rows: `{manifest['low_timestamp_rank_rows']}`",
        "No realized outcome, MFE, MAE, PnL, utility, label, oracle, or timeout columns are used before the v2 artifact is written.",
        "",
        "## Scorecard",
        "",
        _table(scorecard, cols, limit=80),
        "",
        "## Quality By Archetype",
        "",
        _table(quality.sort_values(["scope", "mean_utility"], ascending=[True, False]), quality_cols, limit=120),
        "",
        "## Monthly Coverage And Quality",
        "",
        _table(monthly.sort_values(["scope", "archetype", "month"]), month_cols, limit=160),
        "",
        "## Outputs",
        "",
        f"- Parquet: `{manifest['outputs']['parquet']}`",
        f"- CSV: `{manifest['outputs']['csv']}`",
        f"- Scorecard: `{manifest['outputs']['scorecard']}`",
        f"- Monthly: `{manifest['outputs']['monthly']}`",
        f"- Weekly: `{manifest['outputs']['weekly']}`",
        f"- Regime matrix: `{manifest['outputs']['regime_matrix']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_materialization(
    *,
    source_tags_path: Path,
    labels_path: Path,
    output_dir: Path,
    timestamp_col: str,
    symbol_col: str,
    min_timestamp_rows: int,
    prior_symbol_window: int,
    min_coverage: float,
    max_top_symbol_share: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_tags = _load_table(source_tags_path)
    archetypes, build_report = build_archetypes_v2(
        source_tags,
        timestamp_col=timestamp_col,
        symbol_col=symbol_col,
        min_timestamp_rows=min_timestamp_rows,
        prior_symbol_window=prior_symbol_window,
    )

    keep_cols = [
        col
        for col in [
            "__source_row_idx__",
            "__source_key__",
            timestamp_col,
            symbol_col,
            "side",
            "side_name",
            "__side__",
            "timeframe",
            "candidate_id",
            "G_VOL",
            "__regime_vol_12h__",
            "__regime_vol_48h__",
            "__regime_volume_12h__",
            "__regime_volume_48h__",
            "__regime_trend_12h__",
            "__regime_trend_48h__",
            "primary_source_tag",
            "source_tag_reason_codes",
            "prior_symbol_event_density_score",
            "prior_symbol_event_density_rank",
            *V2_SCORE_COLS,
            *V2_TAGS,
            "primary_source_archetype_v2",
            "archetype_v2_reason_codes",
        ]
        if col in archetypes.columns
    ]
    materialized = archetypes.loc[:, keep_cols].copy()
    materialized, side_contract_report = _expand_materialized_with_side_contract(
        materialized,
        labels_path=labels_path,
        timestamp_col=timestamp_col,
        symbol_col=symbol_col,
    )

    paths = {
        "parquet": output_dir / "candidate_source_archetypes_v2.parquet",
        "csv": output_dir / "candidate_source_archetypes_v2.csv",
        "quality": output_dir / "source_archetypes_v2_quality.csv",
        "monthly": output_dir / "source_archetypes_v2_by_month.csv",
        "weekly": output_dir / "source_archetypes_v2_by_week.csv",
        "scorecard": output_dir / "source_archetypes_v2_scorecard.csv",
        "regime_matrix": output_dir / "source_archetypes_v2_regime_matrix.csv",
        "manifest": output_dir / "manifest.json",
    }
    materialized.to_parquet(paths["parquet"], index=False)
    materialized.to_csv(paths["csv"], index=False)

    joined, join_report = _load_joined_frame(quality_labels_path=paths["parquet"], labels_path=labels_path)
    metrics = _path_metrics(joined)
    joined[timestamp_col] = pd.to_datetime(joined[timestamp_col], utc=True, errors="coerce")
    joined["month"] = joined[timestamp_col].dt.tz_convert(None).dt.to_period("M").astype(str)
    joined["week_start"] = _week_start(joined[timestamp_col])

    quality_parts: list[pd.DataFrame] = []
    for tag_col in V2_TAGS:
        quality_parts.append(_quality_by_group(frame=joined, metrics=metrics, group_col=tag_col, scope_name="multi_tag"))
    quality_parts.append(
        _quality_by_group(
            frame=joined,
            metrics=metrics,
            group_col="primary_source_archetype_v2",
            scope_name="primary",
        )
    )
    quality = pd.concat([part for part in quality_parts if not part.empty], ignore_index=True)
    monthly = _quality_over_time(frame=joined, metrics=metrics, period_col="month", period_name="month")
    weekly = _quality_over_time(frame=joined, metrics=metrics, period_col="week_start", period_name="week_start")
    scorecard = _scorecard(
        archetypes=materialized,
        joined=joined,
        metrics=metrics,
        monthly=monthly,
        timestamp_col=timestamp_col,
        symbol_col=symbol_col,
        min_coverage=min_coverage,
        max_top_symbol_share=max_top_symbol_share,
    )
    regime_matrix = _regime_matrix(joined, metrics)

    quality.to_csv(paths["quality"], index=False)
    monthly.to_csv(paths["monthly"], index=False)
    weekly.to_csv(paths["weekly"], index=False)
    scorecard.to_csv(paths["scorecard"], index=False)
    regime_matrix.to_csv(paths["regime_matrix"], index=False)

    manifest = {
        "scope": "candidate_source_archetypes_v2",
        "source_tags_path": str(source_tags_path),
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(materialized)),
        "utility_source": metrics.attrs.get("utility_source"),
        "date_min": build_report["date_min"],
        "date_max": build_report["date_max"],
        "symbols": build_report["symbols"],
        "side_counts_full": _side_name_series(materialized).value_counts(dropna=False).to_dict(),
        "side_counts_joined": _side_name_series(joined).value_counts(dropna=False).to_dict(),
        "side_contract_report": side_contract_report,
        "source_columns_used": build_report["source_columns_used"],
        "missing_source_columns": build_report["missing_source_columns"],
        "low_timestamp_rank_rows": build_report["low_timestamp_rank_rows"],
        "min_timestamp_rows": int(min_timestamp_rows),
        "prior_symbol_window": int(prior_symbol_window),
        "min_coverage": float(min_coverage),
        "max_top_symbol_share": float(max_top_symbol_share),
        "join_report": join_report,
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    report = _write_report(
        output_dir=output_dir,
        manifest=manifest,
        scorecard=scorecard,
        quality=quality,
        monthly=monthly,
    )
    manifest["outputs"]["markdown"] = str(report)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tags-path", type=Path, default=DEFAULT_SOURCE_TAGS)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timestamp-col", type=str, default="__ts__")
    parser.add_argument("--symbol-col", type=str, default="__symbol__")
    parser.add_argument("--min-timestamp-rows", type=int, default=5)
    parser.add_argument("--prior-symbol-window", type=int, default=12)
    parser.add_argument("--min-coverage", type=float, default=0.05)
    parser.add_argument("--max-top-symbol-share", type=float, default=0.35)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_materialization(
        source_tags_path=args.source_tags_path,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        timestamp_col=args.timestamp_col,
        symbol_col=args.symbol_col,
        min_timestamp_rows=int(args.min_timestamp_rows),
        prior_symbol_window=int(args.prior_symbol_window),
        min_coverage=float(args.min_coverage),
        max_top_symbol_share=float(args.max_top_symbol_share),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
