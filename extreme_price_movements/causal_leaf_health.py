"""Causal H1--H5 health states for strict-OOF base reasoning families.

This module is deliberately the *second* boundary after
``leaf_family_contributions``.  It accepts only token-free, same-artifact
rule-family contributions and strict-OOF candidate outcomes.  It then emits
compact candidate-level features for a later meta learner:

* H1: prior-resolved hierarchical correctness/support;
* H2: completed-period portability and excess variance;
* H3: small, month-frozen context compatibility models;
* H4: selected-family causal covariance compatibility; and
* H5: selected-family causal relationship-break exposure.

Two ordering rules are intentionally non-negotiable:

1. A candidate is scored before its own label is available; and
2. every candidate at the same ``feature_generation_ts`` sees the same
   history.  Labels with ``label_available_ts == feature_generation_ts`` are
   therefore still unavailable.

The public builder works on in-memory frames to make the contract testable.
The accompanying artifact materialiser may use bounded temporary parquet
partitions, but it must feed this exact builder and may not introduce raw
leaf tokens, leaf IDs, or cross-model leaf alignment.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .causal_leaf_covariance import (
    CausalLeafCovarianceConfig,
    build_causal_leaf_covariance_state,
)


SCHEMA = "causal_leaf_health_v1"
STATUS = "CAUSAL_LEAF_HEALTH_MATERIALIZED"
HEADS: tuple[str, ...] = ("p_adverse", "p_weak", "p_clear")
DIRECTIONS: tuple[str, ...] = ("positive", "negative")
RAW_LEAF_MARKERS: tuple[str, ...] = (
    "leaf_token", "leaf_id", "leaf_assignment", "raw_leaf",
)
FORBIDDEN_CONTEXT_TOKENS: tuple[str, ...] = (
    "target", "label", "outcome", "future", "realized", "realised",
    "pnl", "net_ev", "gross_ev", "mfe", "mae", "barrier", "timeout",
    "exit", "post_entry", "postentry",
)
RELATIONSHIP_BREAK_PATTERN = re.compile(
    r"^continuous_regime__relationship_break__(?P<pair>[a-z0-9_]+)__residual_abs_(?P<window>\d+)d$"
)


class CausalLeafHealthError(ValueError):
    """Raised when strict OOF or prequential lineage cannot be proved."""


@dataclass(frozen=True)
class CausalLeafHealthConfig:
    """Bounded, target-independent controls for H1--H5 state generation."""

    global_alpha: float = 1.0
    global_beta: float = 1.0
    side_head_prior_strength: float = 16.0
    family_prior_strength: float = 12.0
    min_timestamp_support: int = 24
    min_day_support: int = 5
    min_symbol_support: int = 5
    min_period_rows: int = 24
    min_periods: int = 3
    period_close_lag_hours: int = 24
    h2_calibration_drift_threshold: float = 0.15
    h2_instability_threshold: float = 2.5
    h2_support_failure_threshold: float = 0.50
    h3_min_rows: int = 64
    h3_ridge_alpha: float = 8.0
    h3_max_rows_per_family: int = 4_000
    h3_context_scale_floor_bps: float = 25.0
    covariance_max_fields: int = 10
    covariance_min_reference_rows: int = 2
    relationship_break_threshold: float = 0.20
    # Selection is external and must be frozen before the scored partition.
    # A key is ``(feature_contract_sha256, side, head, rule_signature,
    # direction)``.  Empty selections are valid and result in availability=0,
    # rather than silently choosing families on the current evaluation rows.
    selected_context_families: frozenset[tuple[str, str, str, str, str]] = frozenset()
    selected_covariance_families: frozenset[tuple[str, str, str, str, str]] = frozenset()
    selected_relationship_families: frozenset[tuple[str, str, str, str, str]] = frozenset()
    # New frozen predecessor selections become usable only from their own
    # declared cutoff.  This keeps a single chronological health pass safe:
    # selected families may retain prior raw/context history for a later H3/H4
    # state, but no pre-cutoff candidate can receive a feature chosen later.
    family_selection_effective_utc: str | None = None

    def validate(self) -> None:
        nonnegative = (
            self.global_alpha, self.global_beta, self.side_head_prior_strength,
            self.family_prior_strength, self.h3_ridge_alpha,
        )
        if any(not np.isfinite(value) or float(value) < 0.0 for value in nonnegative):
            raise CausalLeafHealthError("prior and ridge strengths must be finite and non-negative")
        if float(self.global_alpha) + float(self.global_beta) <= 0.0:
            raise CausalLeafHealthError("global beta prior must have positive total mass")
        if int(self.min_timestamp_support) <= 0 or int(self.min_day_support) <= 0 or int(self.min_symbol_support) <= 0:
            raise CausalLeafHealthError("support thresholds must be positive")
        if int(self.min_period_rows) <= 0 or int(self.min_periods) <= 0:
            raise CausalLeafHealthError("period thresholds must be positive")
        if int(self.period_close_lag_hours) < 0:
            raise CausalLeafHealthError("period_close_lag_hours must be non-negative")
        if int(self.h3_min_rows) < 2 or int(self.h3_max_rows_per_family) < int(self.h3_min_rows):
            raise CausalLeafHealthError("H3 row bounds are invalid")
        if int(self.covariance_max_fields) < 1 or int(self.covariance_max_fields) > 15:
            raise CausalLeafHealthError("covariance_max_fields must be in [1, 15]")
        if int(self.covariance_min_reference_rows) < 2:
            raise CausalLeafHealthError("covariance_min_reference_rows must be at least two")
        if not np.isfinite(float(self.relationship_break_threshold)) or float(self.relationship_break_threshold) < 0.0:
            raise CausalLeafHealthError("relationship_break_threshold must be finite and non-negative")
        for selection in (
            self.selected_context_families,
            self.selected_covariance_families,
            self.selected_relationship_families,
        ):
            for key in selection:
                if len(tuple(key)) != 5:
                    raise CausalLeafHealthError("selected family keys must have five fields")
        if self.family_selection_effective_utc is not None:
            effective = pd.to_datetime(self.family_selection_effective_utc, utc=True, errors="coerce")
            if pd.isna(effective):
                raise CausalLeafHealthError("family_selection_effective_utc must be a finite UTC timestamp")


@dataclass(frozen=True)
class CausalLeafHealthResult:
    """All required H1--H5 tables for one immutable strict-OOF hand-off."""

    family_candidate_states: pd.DataFrame
    health_features: pd.DataFrame
    period_metrics: pd.DataFrame
    portability_scores: pd.DataFrame
    covariance_diagnostics: pd.DataFrame
    relationship_breaks: pd.DataFrame
    covariance_explainability: pd.DataFrame
    manifest: dict[str, Any]


@dataclass
class _PeriodStats:
    rows: int = 0
    successes: float = 0.0
    prediction_sum: float = 0.0
    net_sum: float = 0.0
    base_expected_sum: float = 0.0
    residual_sum: float = 0.0
    residual_sq_sum: float = 0.0
    calibration_sum: float = 0.0
    false_positive_loss_sum: float = 0.0
    timestamps: set[int] = field(default_factory=set)
    days: set[str] = field(default_factory=set)
    symbols: set[str] = field(default_factory=set)

    def update(
        self,
        *, success: float, prediction: float, net_bps: float,
        base_expected_bps: float, decision_ts: pd.Timestamp, asset: str,
    ) -> None:
        residual = float(net_bps) - float(base_expected_bps)
        calibration = float(success) - float(prediction)
        self.rows += 1
        self.successes += float(success)
        self.prediction_sum += float(prediction)
        self.net_sum += float(net_bps)
        self.base_expected_sum += float(base_expected_bps)
        self.residual_sum += residual
        self.residual_sq_sum += residual * residual
        self.calibration_sum += calibration
        if float(prediction) >= 0.5 and float(success) <= 0.0:
            self.false_positive_loss_sum += max(-float(net_bps), 0.0)
        self.timestamps.add(int(decision_ts.value))
        self.days.add(str(decision_ts.date()))
        self.symbols.add(str(asset))

    def effect_bps(self) -> float:
        return float(self.residual_sum / self.rows) if self.rows else np.nan

    def effect_se_bps(self) -> float:
        if self.rows < 2:
            return np.nan
        mean = self.effect_bps()
        variance = max(self.residual_sq_sum / self.rows - mean * mean, 0.0)
        return float(math.sqrt(variance / self.rows))


@dataclass
class _FamilyStats:
    rows: int = 0
    successes: float = 0.0
    prediction_sum: float = 0.0
    net_sum: float = 0.0
    base_expected_sum: float = 0.0
    false_positive_loss_sum: float = 0.0
    timestamps: set[int] = field(default_factory=set)
    days: set[str] = field(default_factory=set)
    symbols: set[str] = field(default_factory=set)
    periods: dict[str, _PeriodStats] = field(default_factory=dict)
    # H3 keeps only selected families and only label-resolved observations.
    h3_records: deque[tuple[np.ndarray, float]] = field(default_factory=deque)

    def update(
        self,
        *, success: float, prediction: float, net_bps: float,
        base_expected_bps: float, decision_ts: pd.Timestamp, asset: str,
        context: np.ndarray | None, h3_selected: bool, max_h3_rows: int,
    ) -> None:
        self.rows += 1
        self.successes += float(success)
        self.prediction_sum += float(prediction)
        self.net_sum += float(net_bps)
        self.base_expected_sum += float(base_expected_bps)
        if float(prediction) >= 0.5 and float(success) <= 0.0:
            self.false_positive_loss_sum += max(-float(net_bps), 0.0)
        self.timestamps.add(int(decision_ts.value))
        self.days.add(str(decision_ts.date()))
        self.symbols.add(str(asset))
        period = str(decision_ts.strftime("%Y-%m"))
        item = self.periods.setdefault(period, _PeriodStats())
        item.update(
            success=success, prediction=prediction, net_bps=net_bps,
            base_expected_bps=base_expected_bps, decision_ts=decision_ts,
            asset=asset,
        )
        if h3_selected and context is not None and np.isfinite(context).all():
            target = float(net_bps) - float(base_expected_bps)
            self.h3_records.append((context.astype(np.float64, copy=True), target))
            while len(self.h3_records) > int(max_h3_rows):
                self.h3_records.popleft()


@dataclass(frozen=True)
class _RidgeSnapshot:
    coefficients: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    residual_scale: float
    rows: int

    def predict(self, values: np.ndarray) -> float:
        normalised = (values.astype(np.float64, copy=False) - self.center) / self.scale
        return float(self.coefficients[0] + normalised @ self.coefficients[1:])


def _utc(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        raise CausalLeafHealthError(f"required timestamp column is missing: {column}")
    value = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if value.isna().any():
        raise CausalLeafHealthError(f"{column} must contain finite UTC timestamps")
    return value


def _forbid_raw_leaf(columns: Iterable[object], *, source: str) -> None:
    bad = [
        str(column) for column in columns
        if any(token in str(column).lower() for token in RAW_LEAF_MARKERS)
    ]
    if bad:
        raise CausalLeafHealthError(f"{source} contains raw local leaf identifiers: {sorted(bad)}")


def _require_columns(frame: pd.DataFrame, required: Sequence[str], *, source: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise CausalLeafHealthError(f"{source} is missing required columns: {missing}")


def _normalise_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    required = (
        "candidate_id", "decision_ts", "feature_generation_ts", "label_available_ts",
        "side_name", "head_name", "fold_id", "transport", "meta_partition",
        "feature_contract_sha256", "semantic_label", "head_prediction",
        "net_bps", "base_expected_bps", "asset",
    )
    _forbid_raw_leaf(candidates.columns, source="candidate health input")
    _require_columns(candidates, required, source="candidate health input")
    work = candidates.copy()
    for column in ("decision_ts", "feature_generation_ts", "label_available_ts"):
        work[column] = _utc(work, column)
    work["candidate_id"] = work["candidate_id"].astype("string")
    work["side_name"] = work["side_name"].astype("string").str.lower()
    work["head_name"] = work["head_name"].astype("string")
    for column in ("fold_id", "transport", "meta_partition", "feature_contract_sha256", "asset"):
        work[column] = work[column].astype("string")
    if not set(work["side_name"].dropna().unique()).issubset({"long", "short"}):
        raise CausalLeafHealthError("candidate health input contains an unknown side")
    if not set(work["head_name"].dropna().unique()).issubset(set(HEADS)):
        raise CausalLeafHealthError("candidate health input contains an unknown head")
    if not set(work["meta_partition"].dropna().unique()).issubset({"inner_oof", "outer_test"}):
        raise CausalLeafHealthError("candidate health input contains an unknown meta partition")
    if not work["feature_generation_ts"].le(work["decision_ts"]).all():
        raise CausalLeafHealthError("health features cannot be generated after decision time")
    if not work["label_available_ts"].ge(work["decision_ts"]).all():
        raise CausalLeafHealthError("health labels cannot resolve before decision time")
    for column in ("semantic_label", "head_prediction", "net_bps", "base_expected_bps"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    if not np.isfinite(work[["semantic_label", "head_prediction", "net_bps", "base_expected_bps"]].to_numpy(float)).all():
        raise CausalLeafHealthError("candidate outcomes and base predictions must be finite")
    if not np.isin(work["semantic_label"].to_numpy(float), (0.0, 1.0)).all():
        raise CausalLeafHealthError("semantic_label must be binary")
    key = ("candidate_id", "side_name", "head_name", "fold_id", "transport", "meta_partition")
    if work.duplicated(list(key)).any():
        raise CausalLeafHealthError("candidate health input duplicates a candidate/head identity")
    if work.loc[:, list(key)].isna().any().any() or work["candidate_id"].str.strip().eq("").any():
        raise CausalLeafHealthError("candidate health input has null identities")
    return work.sort_values(["feature_generation_ts", "candidate_id", "head_name"], kind="stable").reset_index(drop=True)


def _normalise_contributions(contributions: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    required = (
        "candidate_id", "__ts__", "side_name", "fold_id", "head_name",
        "rule_signature", "contribution_direction", "family_ensemble_tree_contribution",
    )
    _forbid_raw_leaf(contributions.columns, source="family contribution input")
    _require_columns(contributions, required, source="family contribution input")
    work = contributions.loc[:, list(required)].copy()
    work["candidate_id"] = work["candidate_id"].astype("string")
    work["__ts__"] = _utc(work, "__ts__")
    work["side_name"] = work["side_name"].astype("string").str.lower()
    for column in ("fold_id", "head_name", "rule_signature", "contribution_direction"):
        work[column] = work[column].astype("string")
    work["family_ensemble_tree_contribution"] = pd.to_numeric(
        work["family_ensemble_tree_contribution"], errors="coerce"
    )
    if not np.isfinite(work["family_ensemble_tree_contribution"].to_numpy(float)).all():
        raise CausalLeafHealthError("family contributions must be finite")
    if not set(work["contribution_direction"].dropna().unique()).issubset(set(DIRECTIONS)):
        raise CausalLeafHealthError("family contributions contain an unknown direction")
    if work["rule_signature"].str.strip().eq("").any():
        raise CausalLeafHealthError("family contributions contain a blank rule signature")
    lookup = candidates.loc[:, [
        "candidate_id", "decision_ts", "side_name", "head_name", "fold_id",
        "transport", "meta_partition", "feature_contract_sha256",
    ]].copy()
    merged = work.merge(
        lookup,
        left_on=["candidate_id", "__ts__", "side_name", "head_name", "fold_id"],
        right_on=["candidate_id", "decision_ts", "side_name", "head_name", "fold_id"],
        how="left", validate="many_to_one", indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise CausalLeafHealthError("family contributions do not have an identical strict candidate identity")
    merged = merged.drop(columns=["_merge", "decision_ts"])
    exact_key = [
        "candidate_id", "__ts__", "side_name", "head_name", "fold_id",
        "transport", "meta_partition", "rule_signature", "contribution_direction",
    ]
    if merged.duplicated(exact_key).any():
        raise CausalLeafHealthError("family contributions must already collapse duplicate trees per candidate/family")
    if (merged["family_ensemble_tree_contribution"] == 0.0).any():
        raise CausalLeafHealthError("zero-valued family contributions must be removed before health aggregation")
    return merged


def _normalise_context(
    candidates: pd.DataFrame,
    context: pd.DataFrame | None,
    context_feature_columns: Sequence[str] | None,
    config: CausalLeafHealthConfig,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """As-of attach a causal context timeline, never an outcome table."""

    base = candidates.copy()
    if context is None:
        base["regime_available_utc"] = pd.Series(
            pd.NaT, index=base.index, dtype="datetime64[ns, UTC]"
        )
        return base, ()
    _forbid_raw_leaf(context.columns, source="causal regime context")
    if "regime_available_utc" not in context:
        raise CausalLeafHealthError("causal regime context lacks regime_available_utc")
    work = context.copy()
    work["regime_available_utc"] = _utc(work, "regime_available_utc")
    forbidden = [
        str(column) for column in work.columns
        if any(token in str(column).lower() for token in FORBIDDEN_CONTEXT_TOKENS)
    ]
    if forbidden:
        raise CausalLeafHealthError(
            f"causal regime context contains outcome-derived fields: {sorted(forbidden)[:8]}"
        )
    identity_columns = {"candidate_id", "decision_ts", "feature_generation_ts", "regime_available_utc"}
    if context_feature_columns is None:
        candidate = [
            name for name in work.columns
            if name not in identity_columns and pd.api.types.is_numeric_dtype(work[name])
        ]
    else:
        candidate = [str(name) for name in context_feature_columns]
        _require_columns(work, candidate, source="causal regime context")
    if len(candidate) > int(config.covariance_max_fields):
        # H3/covariance deliberately use a compact, predeclared field surface.
        candidate = candidate[: int(config.covariance_max_fields)]
    if not candidate:
        raise CausalLeafHealthError("causal regime context has no declared numeric fields")
    for column in candidate:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    if "candidate_id" in work:
        # Candidate-specific context is permitted only with an explicit as-of
        # timestamp.  One candidate row must never be selected by a later
        # context observation.
        work["candidate_id"] = work["candidate_id"].astype("string")
        if work.duplicated(["candidate_id", "regime_available_utc"]).any():
            raise CausalLeafHealthError("causal regime context duplicates candidate/as-of timestamps")
        joined = base.merge(
            work[["candidate_id", "regime_available_utc", *candidate]],
            on="candidate_id", how="left", validate="many_to_many",
        )
        valid = joined["regime_available_utc"].le(joined["feature_generation_ts"])
        joined = joined.loc[valid].sort_values(
            ["candidate_id", "regime_available_utc"], kind="stable"
        )
        # Several semantic heads share a candidate id, so choose the latest
        # valid as-of context *per original candidate row*, not per id.
        joined["__candidate_row__"] = joined.index.to_numpy(dtype=np.int64)
        joined = joined.groupby("__candidate_row__", as_index=False, sort=False).tail(1)
        if len(joined) != len(base):
            raise CausalLeafHealthError("every candidate requires a causal as-of regime context row")
        joined = joined.sort_values("__candidate_row__", kind="stable").drop(columns="__candidate_row__")
    else:
        # A shared market timeline is joined backwards by availability, not by
        # its nominal observation timestamp.
        work = work.sort_values("regime_available_utc", kind="stable")
        if work["regime_available_utc"].duplicated().any():
            raise CausalLeafHealthError("shared causal regime context duplicates availability timestamps")
        left = base.sort_values("feature_generation_ts", kind="stable")
        joined = pd.merge_asof(
            left, work[["regime_available_utc", *candidate]],
            left_on="feature_generation_ts", right_on="regime_available_utc",
            direction="backward", allow_exact_matches=True,
        ).sort_index(kind="stable")
        if len(joined) != len(base) or joined["regime_available_utc"].isna().any():
            raise CausalLeafHealthError("every candidate requires a prior-available regime context row")
    if not joined["regime_available_utc"].le(joined["feature_generation_ts"]).all():
        raise CausalLeafHealthError("regime context became available after its candidate feature time")
    return joined.reset_index(drop=True), tuple(candidate)


def _family_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row["feature_contract_sha256"]), str(row["side_name"]),
        str(row["head_name"]), str(row["rule_signature"]),
        str(row["contribution_direction"]),
    )


def _scope_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["feature_contract_sha256"]), str(row["side_name"]),
        str(row["head_name"]),
    )


def _global_key(row: Mapping[str, Any]) -> str:
    return str(row["feature_contract_sha256"])


def _family_selection_active(
    feature_generation_ts: pd.Timestamp,
    config: CausalLeafHealthConfig,
) -> bool:
    """Whether a frozen family selector was available at this candidate time."""

    if config.family_selection_effective_utc is None:
        return True
    effective = pd.Timestamp(pd.to_datetime(config.family_selection_effective_utc, utc=True))
    return pd.Timestamp(feature_generation_ts) >= effective


def _posterior(
    family: _FamilyStats, side_head: _FamilyStats, global_stats: _FamilyStats,
    config: CausalLeafHealthConfig,
) -> tuple[float, float]:
    global_mean = (
        float(global_stats.successes) + float(config.global_alpha)
    ) / max(
        float(global_stats.rows) + float(config.global_alpha) + float(config.global_beta), 1e-12
    )
    side_strength = float(config.side_head_prior_strength)
    side_mean = (
        float(side_head.successes) + side_strength * global_mean
    ) / max(float(side_head.rows) + side_strength, 1e-12)
    family_strength = float(config.family_prior_strength)
    alpha = float(family.successes) + family_strength * side_mean
    beta = float(family.rows - family.successes) + family_strength * (1.0 - side_mean)
    total = max(alpha + beta, 1e-12)
    mean = alpha / total
    # A normal approximation is deterministic and has no SciPy version
    # dependency.  It is deliberately conservative for low support.
    lower = max(0.0, mean - 1.96 * math.sqrt(max(mean * (1.0 - mean) / (total + 1.0), 0.0)))
    return float(mean), float(lower)


def _support_score(stats: _FamilyStats, config: CausalLeafHealthConfig) -> float:
    pieces = (
        len(stats.timestamps) / float(config.min_timestamp_support),
        len(stats.days) / float(config.min_day_support),
        len(stats.symbols) / float(config.min_symbol_support),
    )
    return float(np.clip(min(pieces), 0.0, 1.0))


def _closed_periods(
    stats: _FamilyStats, current_time: pd.Timestamp, config: CausalLeafHealthConfig,
) -> list[tuple[str, _PeriodStats]]:
    close_before = current_time - pd.Timedelta(hours=int(config.period_close_lag_hours))
    result: list[tuple[str, _PeriodStats]] = []
    for period, period_stats in stats.periods.items():
        end = (pd.Timestamp(f"{period}-01", tz="UTC") + pd.offsets.MonthBegin(1))
        if end <= close_before:
            result.append((period, period_stats))
    return sorted(result, key=lambda item: item[0])


def _portability_metrics(
    stats: _FamilyStats, current_time: pd.Timestamp, config: CausalLeafHealthConfig,
) -> dict[str, float | str]:
    periods = _closed_periods(stats, current_time, config)
    supported = [
        item for item in periods
        if item[1].rows >= int(config.min_period_rows)
        and len(item[1].timestamps) >= int(config.min_timestamp_support)
        and len(item[1].days) >= int(config.min_day_support)
        and len(item[1].symbols) >= int(config.min_symbol_support)
    ]
    support_failure = 1.0 - (len(supported) / max(len(periods), 1))
    if not supported:
        return {
            "period_count": float(len(periods)), "supported_period_count": 0.0,
            "observed_variance": np.nan, "sampling_variance": np.nan,
            "excess_variance": np.nan, "sign_reversal_rate": np.nan,
            "worst_damage_bps": np.nan, "calibration_drift": np.nan,
            "support_failure": float(support_failure),
            "instability": np.nan, "classification": "LOW_SUPPORT_UNCERTAIN",
        }
    effects = np.asarray([entry.effect_bps() for _, entry in supported], dtype=np.float64)
    errors = np.asarray([entry.effect_se_bps() for _, entry in supported], dtype=np.float64)
    weights = np.asarray([entry.rows for _, entry in supported], dtype=np.float64)
    centre = float(np.average(effects, weights=weights))
    observed = float(np.average((effects - centre) ** 2, weights=weights))
    sampling = float(np.nanmean(errors ** 2)) if np.isfinite(errors).any() else 0.0
    excess = max(0.0, observed - sampling)
    credible = np.abs(effects) > 1.96 * np.nan_to_num(errors, nan=np.inf)
    reversals: list[float] = []
    for left, right, left_ok, right_ok in zip(effects[:-1], effects[1:], credible[:-1], credible[1:], strict=True):
        if bool(left_ok) and bool(right_ok):
            reversals.append(float(np.sign(left) != np.sign(right)))
    sign_reversal = float(np.mean(reversals)) if reversals else 0.0
    worst_damage = max(0.0, -float(np.nanmin(effects)))
    calibration_drift = float(np.average(
        np.abs(np.asarray([
            entry.calibration_sum / max(entry.rows, 1) for _, entry in supported
        ], dtype=np.float64)), weights=weights,
    ))
    # The cross-family robust z-score is applied in a monthly frozen snapshot
    # below.  Keep this intrinsic component to make audit artifacts complete.
    classification = "STABLE_PORTABLE"
    if len(supported) < int(config.min_periods):
        classification = "LOW_SUPPORT_UNCERTAIN"
    elif support_failure >= float(config.h2_support_failure_threshold):
        classification = "SUPPORT_SHIFT_ONLY"
    elif calibration_drift >= float(config.h2_calibration_drift_threshold):
        classification = "CALIBRATION_DRIFT"
    elif sign_reversal > 0.0 or excess > 0.0:
        classification = "GLOBAL_PERIOD_SENSITIVE"
    return {
        "period_count": float(len(periods)), "supported_period_count": float(len(supported)),
        "observed_variance": observed, "sampling_variance": sampling,
        "excess_variance": excess, "sign_reversal_rate": sign_reversal,
        "worst_damage_bps": worst_damage, "calibration_drift": calibration_drift,
        "support_failure": float(support_failure),
        "instability": np.nan, "classification": classification,
    }


def _snapshot_portability(
    family_stats: Mapping[tuple[str, str, str, str, str], _FamilyStats],
    current_time: pd.Timestamp, config: CausalLeafHealthConfig,
) -> dict[tuple[str, str, str, str, str], dict[str, float | str]]:
    raw = {
        key: _portability_metrics(stats, current_time, config)
        for key, stats in family_stats.items()
    }
    by_bucket: dict[tuple[str, str, str, int], list[tuple[tuple[str, str, str, str, str], float]]] = defaultdict(list)
    for key, metrics in raw.items():
        value = float(metrics["excess_variance"])
        if np.isfinite(value):
            support_bucket = int(np.floor(np.log2(max(int(family_stats[key].rows), 1))))
            by_bucket[(key[0], key[1], key[2], support_bucket)].append((key, value))
    robust_z: dict[tuple[str, str, str, str, str], float] = {}
    for entries in by_bucket.values():
        values = np.asarray([value for _, value in entries], dtype=np.float64)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        scale = max(1.4826 * mad, 1e-8)
        for key, value in entries:
            robust_z[key] = float((value - median) / scale)
    for key, metrics in raw.items():
        z = robust_z.get(key, np.nan)
        damage_scale = max(float(np.nanmedian([
            float(item["worst_damage_bps"]) for item in raw.values()
            if np.isfinite(float(item["worst_damage_bps"]))
        ] or [1.0])), 1.0)
        damage = float(metrics["worst_damage_bps"])
        instability = (
            float(np.nan_to_num(z, nan=0.0))
            + float(metrics["sign_reversal_rate"])
            + damage / damage_scale
            + 0.5 * float(metrics["support_failure"])
        )
        metrics["robust_z_excess_variance"] = z
        metrics["instability"] = float(instability)
        if metrics["classification"] != "LOW_SUPPORT_UNCERTAIN":
            if instability > float(config.h2_instability_threshold):
                metrics["classification"] = "GLOBAL_PERIOD_SENSITIVE"
    return raw


def _fit_ridge(records: deque[tuple[np.ndarray, float]], config: CausalLeafHealthConfig) -> _RidgeSnapshot | None:
    if len(records) < int(config.h3_min_rows):
        return None
    values = np.vstack([item[0] for item in records]).astype(np.float64, copy=False)
    targets = np.asarray([item[1] for item in records], dtype=np.float64)
    if not np.isfinite(values).all() or not np.isfinite(targets).all():
        return None
    centre = np.median(values, axis=0)
    scale = np.median(np.abs(values - centre), axis=0) * 1.4826
    scale = np.maximum(scale, 1e-6)
    x = (values - centre) / scale
    design = np.column_stack((np.ones(len(x)), x))
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(config.h3_ridge_alpha)
    penalty[0, 0] = 0.0
    try:
        coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    except np.linalg.LinAlgError:
        return None
    residual = targets - design @ coefficient
    residual_scale = max(
        float(np.median(np.abs(residual - np.median(residual))) * 1.4826),
        float(config.h3_context_scale_floor_bps),
    )
    return _RidgeSnapshot(coefficient, centre, scale, residual_scale, len(records))


def _context_model_snapshot(
    family_stats: Mapping[tuple[str, str, str, str, str], _FamilyStats],
    config: CausalLeafHealthConfig,
) -> dict[tuple[str, str, str, str, str], _RidgeSnapshot]:
    return {
        key: model
        for key, stats in family_stats.items()
        if key in config.selected_context_families
        for model in (_fit_ridge(stats.h3_records, config),)
        if model is not None
    }


def _family_h1(
    family: _FamilyStats, side_head: _FamilyStats, global_stats: _FamilyStats,
    config: CausalLeafHealthConfig,
) -> dict[str, float]:
    posterior, lower = _posterior(family, side_head, global_stats, config)
    rows = max(family.rows, 1)
    return {
        "posterior_correctness": posterior,
        "posterior_lower_95": lower,
        "row_support": float(family.rows),
        "timestamp_support": float(len(family.timestamps)),
        "day_support": float(len(family.days)),
        "symbol_support": float(len(family.symbols)),
        "support_score": _support_score(family, config),
        "calibration_residual": float(family.successes / rows - family.prediction_sum / rows),
        "economic_residual_bps": float(family.net_sum / rows - family.base_expected_sum / rows),
        "false_positive_loss_bps": float(family.false_positive_loss_sum / rows),
    }


def _candidate_columns() -> list[str]:
    return [
        "candidate_id", "decision_ts", "side_name", "fold_id", "transport", "meta_partition",
    ]


def _state_columns(prefix: str) -> list[str]:
    return [name for name in prefix.split("|") if name]


def _aggregate_family_metrics(
    candidates: pd.DataFrame,
    states: pd.DataFrame,
    *, section: str, metrics: Sequence[str],
) -> pd.DataFrame:
    """Contribution-weighted H fields, separated by semantic head/direction."""

    identity = _candidate_columns()
    output = candidates.loc[:, identity].drop_duplicates(identity).copy()
    if states.empty:
        for head in HEADS:
            for direction in DIRECTIONS:
                for metric in metrics:
                    output[f"base_health__{section}__{head}__{direction}__{metric}"] = np.float32(0.0)
        return output
    work = states.copy()
    work["__weight__"] = np.abs(pd.to_numeric(work["family_ensemble_tree_contribution"], errors="coerce"))
    for head in HEADS:
        for direction in DIRECTIONS:
            mask = work["head_name"].eq(head) & work["contribution_direction"].eq(direction)
            cell = work.loc[mask]
            names = {
                metric: f"base_health__{section}__{head}__{direction}__{metric}"
                for metric in metrics
            }
            if cell.empty:
                for name in names.values():
                    output[name] = np.float32(0.0)
                continue
            grouped_weight = cell.groupby(identity, observed=True, sort=False)["__weight__"].sum().rename("__total_weight__")
            aggregate = grouped_weight.reset_index()
            for metric, name in names.items():
                values = pd.to_numeric(cell[metric], errors="coerce") if metric in cell else pd.Series(np.nan, index=cell.index)
                numerator = (values.fillna(0.0) * cell["__weight__"]).groupby(
                    [cell[column] for column in identity], observed=True, sort=False,
                ).sum().rename(name)
                availability = (values.notna().astype(float) * cell["__weight__"]).groupby(
                    [cell[column] for column in identity], observed=True, sort=False,
                ).sum()
                joined = numerator.to_frame().join(availability.rename("__available_weight__"), how="left")
                available_weight = joined["__available_weight__"].to_numpy(float)
                joined[name] = np.divide(
                    joined[name].to_numpy(float), available_weight,
                    out=np.zeros(len(joined), dtype=np.float64),
                    where=available_weight > 0.0,
                )
                aggregate = aggregate.merge(joined[[name]].reset_index(), on=identity, how="left", validate="one_to_one")
            weight_name = f"base_health__{section}__{head}__{direction}__active_abs_contribution"
            aggregate = aggregate.rename(columns={"__total_weight__": weight_name})
            output = output.merge(aggregate, on=identity, how="left", validate="one_to_one")
            for name in [*names.values(), weight_name]:
                output[name] = pd.to_numeric(output[name], errors="coerce").fillna(0.0).astype(np.float32)
    return output


def _join_feature_sections(sections: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not sections:
        return pd.DataFrame(columns=_candidate_columns())
    out = sections[0].copy()
    for section in sections[1:]:
        out = out.merge(section, on=_candidate_columns(), how="inner", validate="one_to_one")
    return out


def _relationship_break_columns(context_columns: Sequence[str]) -> dict[str, list[str]]:
    pairs: dict[str, list[str]] = defaultdict(list)
    for column in context_columns:
        match = RELATIONSHIP_BREAK_PATTERN.match(str(column))
        if match:
            pairs[match.group("pair")].append(str(column))
    return {pair: sorted(columns) for pair, columns in pairs.items()}


def _config_payload(config: CausalLeafHealthConfig) -> dict[str, Any]:
    """Serialise frozen family selections deterministically for provenance."""

    payload = asdict(config)
    for name in (
        "selected_context_families", "selected_covariance_families",
        "selected_relationship_families",
    ):
        payload[name] = [list(key) for key in sorted(getattr(config, name))]
    return payload


def _materialise_h4_h5(
    states: pd.DataFrame,
    *, context_columns: Sequence[str], config: CausalLeafHealthConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build selected H4 covariance and H5 pair-break state without labels."""

    identity = _candidate_columns()
    family_keys = states.apply(_family_key, axis=1) if not states.empty else pd.Series(dtype=object)
    selected_covariance = states.loc[family_keys.isin(config.selected_covariance_families)].copy() if not states.empty else states.copy()
    covariance = pd.DataFrame()
    if not selected_covariance.empty and context_columns:
        usable = [
            column for column in context_columns
            if selected_covariance[column].notna().all() and np.isfinite(pd.to_numeric(selected_covariance[column], errors="coerce")).all()
        ][: int(config.covariance_max_fields)]
        if usable:
            # Outer transport lineages can replay the same historical decision
            # timestamps.  They must never become one global covariance
            # history: that would either make the reference block order
            # ambiguous or allow a duplicate from one transport to influence
            # the other.  Within an individual transport, fold blocks are
            # chronological and use the normal frozen-reference contract.
            covariance_parts: list[pd.DataFrame] = []
            for _, transport_states in selected_covariance.groupby("transport", sort=True):
                cov_input = transport_states.copy()
                cov_input["source_utc"] = cov_input["feature_generation_ts"]
                cov_input["evaluation_block"] = cov_input["fold_id"].astype(str)
                cov_input["family"] = cov_input.apply(
                    lambda row: "|".join(_family_key(row)), axis=1,
                )
                covariance_parts.append(build_causal_leaf_covariance_state(
                    cov_input.sort_values(["source_utc", "candidate_id", "head_name"], kind="stable").reset_index(drop=True),
                    usable,
                    config=CausalLeafCovarianceConfig(
                        min_reference_rows=int(config.covariance_min_reference_rows),
                        max_fields_per_family=int(config.covariance_max_fields),
                    ),
                    prefix="base_health__h4",
                ).frame)
            covariance = pd.concat(covariance_parts, ignore_index=True)
            keep = [
                "candidate_id", "decision_ts", "side_name", "fold_id", "transport", "meta_partition",
                "head_name", "contribution_direction", "rule_signature",
                "family_ensemble_tree_contribution",
                *[name for name in covariance.columns if name.startswith("base_health__h4__")],
            ]
            covariance = covariance.loc[:, keep].copy()
    if covariance.empty:
        covariance = pd.DataFrame(columns=[*identity, "head_name", "contribution_direction", "rule_signature", "family_ensemble_tree_contribution"])

    relationship_rows: list[pd.DataFrame] = []
    pairs = _relationship_break_columns(context_columns)
    selected_relationship = states.loc[
        family_keys.isin(config.selected_relationship_families)
        & states.get("h5_selection_active", pd.Series(1.0, index=states.index)).astype(bool)
    ].copy() if not states.empty else states.copy()
    if not selected_relationship.empty:
        for pair, columns in pairs.items():
            # The maximum 30/90d residual is intentionally conservative: it
            # identifies a break at either causal relationship horizon without
            # treating future reconciliation as a feature.
            value = selected_relationship.loc[:, columns].abs().max(axis=1, skipna=True)
            portable = 1.0 / (1.0 + np.maximum(
                pd.to_numeric(selected_relationship.get("h2_instability"), errors="coerce").fillna(0.0), 0.0
            ))
            economic = 1.0 + np.abs(pd.to_numeric(
                selected_relationship.get("h1_economic_residual_bps"), errors="coerce").fillna(0.0)
            ) / 100.0
            table = selected_relationship.loc[:, [
                *identity, "head_name", "contribution_direction", "rule_signature",
                "family_ensemble_tree_contribution",
            ]].copy()
            table["relationship_pair"] = pair
            table["relationship_break"] = pd.to_numeric(value, errors="coerce")
            table["material_break"] = table["relationship_break"].ge(float(config.relationship_break_threshold))
            table["portable_economic_weight"] = (
                np.abs(table["family_ensemble_tree_contribution"].to_numpy(float))
                * portable.to_numpy(float) * economic.to_numpy(float)
            )
            relationship_rows.append(table)
    relationships = pd.concat(relationship_rows, ignore_index=True) if relationship_rows else pd.DataFrame(
        columns=[*identity, "head_name", "contribution_direction", "rule_signature", "family_ensemble_tree_contribution", "relationship_pair", "relationship_break", "material_break", "portable_economic_weight"]
    )
    return covariance, relationships, pd.DataFrame({
        "context_feature": list(context_columns),
        "used_for_covariance": [column in set(context_columns[: int(config.covariance_max_fields)]) for column in context_columns],
    })


def _aggregate_h5(
    candidates: pd.DataFrame, relationships: pd.DataFrame,
) -> pd.DataFrame:
    identity = _candidate_columns()
    output = candidates.loc[:, identity].drop_duplicates(identity).copy()
    pairs = sorted(set(relationships.get("relationship_pair", pd.Series(dtype="string")).astype(str)))
    for head in HEADS:
        for direction in DIRECTIONS:
            base_name = f"base_health__h5__{head}__{direction}"
            cell = relationships.loc[
                relationships.get("head_name", pd.Series(dtype="string")).eq(head)
                & relationships.get("contribution_direction", pd.Series(dtype="string")).eq(direction)
            ].copy()
            if cell.empty:
                for suffix in ("weighted_break", "material_break_share", "worst_break", "availability"):
                    output[f"{base_name}__{suffix}"] = np.float32(0.0)
                for pair in pairs:
                    output[f"{base_name}__{pair}__material_break_share"] = np.float32(0.0)
                continue
            cell["__weight__"] = pd.to_numeric(cell["portable_economic_weight"], errors="coerce").fillna(0.0)
            aggregate = cell.groupby(identity, observed=True, sort=False).agg(
                __weight__=("__weight__", "sum"),
                __break_numerator__=("relationship_break", lambda values: 0.0),
            ).reset_index()
            # Avoid custom groupby aggregation closures with an implicit
            # frame: numerators are explicit vectorised columns.
            cell["__break_weighted__"] = cell["relationship_break"].fillna(0.0) * cell["__weight__"]
            cell["__material_weighted__"] = cell["material_break"].astype(float) * cell["__weight__"]
            sums = cell.groupby(identity, observed=True, sort=False).agg(
                __break_weighted__=("__break_weighted__", "sum"),
                __material_weighted__=("__material_weighted__", "sum"),
                __worst__=("relationship_break", "max"),
            ).reset_index()
            aggregate = aggregate.drop(columns="__break_numerator__").merge(sums, on=identity, how="inner", validate="one_to_one")
            denom = aggregate["__weight__"].to_numpy(float)
            aggregate[f"{base_name}__weighted_break"] = np.divide(aggregate["__break_weighted__"].to_numpy(float), denom, out=np.zeros(len(aggregate)), where=denom > 0.0)
            aggregate[f"{base_name}__material_break_share"] = np.divide(aggregate["__material_weighted__"].to_numpy(float), denom, out=np.zeros(len(aggregate)), where=denom > 0.0)
            aggregate[f"{base_name}__worst_break"] = aggregate["__worst__"].fillna(0.0)
            aggregate[f"{base_name}__availability"] = (denom > 0.0).astype(np.float32)
            for pair in pairs:
                pair_cell = cell.loc[cell["relationship_pair"].eq(pair)]
                if pair_cell.empty:
                    aggregate[f"{base_name}__{pair}__material_break_share"] = 0.0
                else:
                    pair_sum = pair_cell.groupby(identity, observed=True, sort=False)["__material_weighted__"].sum().rename("__pair__").reset_index()
                    aggregate = aggregate.merge(pair_sum, on=identity, how="left", validate="one_to_one")
                    aggregate[f"{base_name}__{pair}__material_break_share"] = np.divide(aggregate["__pair__"].fillna(0.0).to_numpy(float), denom, out=np.zeros(len(aggregate)), where=denom > 0.0)
                    aggregate = aggregate.drop(columns="__pair__")
            keep = [*identity, *[name for name in aggregate if name.startswith(base_name)]]
            output = output.merge(aggregate[keep], on=identity, how="left", validate="one_to_one")
            for name in [column for column in output if column.startswith(base_name)]:
                output[name] = pd.to_numeric(output[name], errors="coerce").fillna(0.0).astype(np.float32)
    return output


def _append_h4_to_states(states: pd.DataFrame, covariance: pd.DataFrame) -> pd.DataFrame:
    if states.empty:
        return states
    names = [name for name in covariance if name.startswith("base_health__h4__")]
    if not names:
        out = states.copy()
        out["h4_availability"] = 0.0
        return out
    keys = [
        "candidate_id", "decision_ts", "side_name", "fold_id", "transport", "meta_partition",
        "head_name", "contribution_direction", "rule_signature",
    ]
    joined = states.merge(covariance.loc[:, [*keys, *names]], on=keys, how="left", validate="one_to_one")
    active = joined.get("h4_selection_active", pd.Series(1.0, index=joined.index)).astype(bool)
    joined["h4_availability"] = (joined[names].notna().any(axis=1) & active).astype(np.float32)
    for name in names:
        joined[name] = (
            pd.to_numeric(joined[name], errors="coerce").where(active, 0.0)
            .fillna(0.0).astype(np.float32)
        )
    return joined


def build_causal_leaf_health_states(
    candidates: pd.DataFrame,
    family_contributions: pd.DataFrame,
    *,
    causal_context: pd.DataFrame | None = None,
    context_feature_columns: Sequence[str] | None = None,
    config: CausalLeafHealthConfig = CausalLeafHealthConfig(),
) -> CausalLeafHealthResult:
    """Materialise strict-prequential H1--H5 states from token-free inputs.

    ``family_contributions`` must have been produced by
    :func:`leaf_family_contributions.materialize_leaf_family_contributions`.
    It is intentionally illegal to substitute raw leaf assignments here.
    """

    config.validate()
    candidate_frame = _normalise_candidates(candidates)
    candidate_frame, context_columns = _normalise_context(
        candidate_frame, causal_context, context_feature_columns, config
    )
    contributions = _normalise_contributions(family_contributions, candidate_frame)
    candidate_frame = candidate_frame.reset_index(drop=True)
    candidate_frame["__event_id__"] = np.arange(len(candidate_frame), dtype=np.int64)
    contribution_join = contributions.merge(
        candidate_frame.loc[:, [
            "candidate_id", "decision_ts", "side_name", "head_name", "fold_id",
            "transport", "meta_partition", "feature_generation_ts", "label_available_ts",
            "feature_contract_sha256", "semantic_label", "head_prediction", "net_bps",
            "base_expected_bps", "asset", "__event_id__", "regime_available_utc", *context_columns,
        ]],
        left_on=["candidate_id", "__ts__", "side_name", "head_name", "fold_id", "transport", "meta_partition", "feature_contract_sha256"],
        right_on=["candidate_id", "decision_ts", "side_name", "head_name", "fold_id", "transport", "meta_partition", "feature_contract_sha256"],
        how="left", validate="many_to_one", indicator=True,
    )
    if not contribution_join["_merge"].eq("both").all():
        raise CausalLeafHealthError("contribution/candidate strict identity reconciliation failed")
    contribution_join = contribution_join.drop(columns=["_merge", "decision_ts"])
    by_event = {
        int(event): value.drop(columns="__event_id__").copy()
        for event, value in contribution_join.groupby("__event_id__", sort=False, observed=True)
    }

    families: dict[tuple[str, str, str, str, str], _FamilyStats] = {}
    side_heads: dict[tuple[str, str, str], _FamilyStats] = {}
    globals_: dict[str, _FamilyStats] = {}
    resolved: list[tuple[pd.Timestamp, int]] = []
    # Each scheduled event is a *candidate/head*.  Its family rows are only
    # touched after the outcome is genuinely label-available.
    state_rows: list[dict[str, Any]] = []
    h2_snapshot: dict[tuple[str, str, str, str, str], dict[str, float | str]] = {}
    h3_snapshot: dict[tuple[str, str, str, str, str], _RidgeSnapshot] = {}
    snapshot_period: str | None = None

    def apply_resolved(event_id: int) -> None:
        row = candidate_frame.iloc[event_id]
        scope = _scope_key(row)
        global_key = _global_key(row)
        side_stats = side_heads.setdefault(scope, _FamilyStats())
        global_stats = globals_.setdefault(global_key, _FamilyStats())
        args = dict(
            success=float(row["semantic_label"]), prediction=float(row["head_prediction"]),
            net_bps=float(row["net_bps"]), base_expected_bps=float(row["base_expected_bps"]),
            decision_ts=pd.Timestamp(row["decision_ts"]), asset=str(row["asset"]),
        )
        side_stats.update(
            **args, context=None, h3_selected=False,
            max_h3_rows=int(config.h3_max_rows_per_family),
        )
        global_stats.update(
            **args, context=None, h3_selected=False,
            max_h3_rows=int(config.h3_max_rows_per_family),
        )
        for _, contribution in by_event.get(event_id, pd.DataFrame()).iterrows():
            key = _family_key(contribution)
            stats = families.setdefault(key, _FamilyStats())
            context = (
                contribution.loc[list(context_columns)].to_numpy(dtype=np.float64, copy=True)
                if context_columns else None
            )
            stats.update(
                **args,
                context=context,
                h3_selected=key in config.selected_context_families,
                max_h3_rows=int(config.h3_max_rows_per_family),
            )

    position = 0
    while position < len(candidate_frame):
        current_time = pd.Timestamp(candidate_frame.loc[position, "feature_generation_ts"])
        end = position + 1
        while end < len(candidate_frame) and pd.Timestamp(candidate_frame.loc[end, "feature_generation_ts"]) == current_time:
            end += 1
        # Strict inequality is intentional: same-timestamp labels are not
        # causally available to any candidate in this decision batch.
        while resolved and resolved[0][0] < current_time:
            _, event_id = resolved.pop(0)
            apply_resolved(event_id)

        period = str(current_time.strftime("%Y-%m"))
        if period != snapshot_period:
            h2_snapshot = _snapshot_portability(families, current_time, config)
            h3_snapshot = _context_model_snapshot(families, config)
            snapshot_period = period

        # Score every family row before scheduling (and therefore before any
        # current candidate label can alter) its own health history.
        for event_id in range(position, end):
            row = candidate_frame.iloc[event_id]
            scope = _scope_key(row)
            global_key = _global_key(row)
            side_stats = side_heads.get(scope, _FamilyStats())
            global_stats = globals_.get(global_key, _FamilyStats())
            for _, contribution in by_event.get(event_id, pd.DataFrame()).iterrows():
                key = _family_key(contribution)
                stats = families.get(key, _FamilyStats())
                h1 = _family_h1(stats, side_stats, global_stats, config)
                h2 = h2_snapshot.get(key, _portability_metrics(stats, current_time, config))
                context = (
                    contribution.loc[list(context_columns)].to_numpy(dtype=np.float64, copy=True)
                    if context_columns else None
                )
                selection_active = _family_selection_active(current_time, config)
                h3_selected_active = selection_active and key in config.selected_context_families
                h4_selected_active = selection_active and key in config.selected_covariance_families
                h5_selected_active = selection_active and key in config.selected_relationship_families
                model = h3_snapshot.get(key) if h3_selected_active else None
                if model is not None and context is not None and np.isfinite(context).all():
                    expected_error = model.predict(context)
                    compatibility = float(np.exp(-abs(expected_error) / max(model.residual_scale, 1e-6)))
                    confidence = float(model.rows / (model.rows + float(config.h3_min_rows)))
                    unexplained = float((1.0 - compatibility) * confidence)
                    h3 = {
                        "availability": 1.0, "compatibility": compatibility,
                        "expected_error_bps": expected_error, "confidence": confidence,
                        "unexplained_break": unexplained,
                    }
                else:
                    h3 = {
                        "availability": 0.0, "compatibility": 0.0,
                        "expected_error_bps": 0.0, "confidence": 0.0,
                        "unexplained_break": 0.0,
                    }
                state_rows.append({
                    "candidate_id": row["candidate_id"], "decision_ts": row["decision_ts"],
                    "feature_generation_ts": row["feature_generation_ts"],
                    "label_available_ts": row["label_available_ts"], "side_name": row["side_name"],
                    "fold_id": row["fold_id"], "transport": row["transport"],
                    "meta_partition": row["meta_partition"], "feature_contract_sha256": row["feature_contract_sha256"],
                    "head_name": row["head_name"], "rule_signature": contribution["rule_signature"],
                    "contribution_direction": contribution["contribution_direction"],
                    "family_ensemble_tree_contribution": float(contribution["family_ensemble_tree_contribution"]),
                    "h4_selection_active": float(h4_selected_active),
                    "h5_selection_active": float(h5_selected_active),
                    "h1_posterior_correctness": h1["posterior_correctness"],
                    "h1_posterior_lower_95": h1["posterior_lower_95"],
                    "h1_row_support": h1["row_support"], "h1_timestamp_support": h1["timestamp_support"],
                    "h1_day_support": h1["day_support"], "h1_symbol_support": h1["symbol_support"],
                    "h1_support_score": h1["support_score"], "h1_calibration_residual": h1["calibration_residual"],
                    "h1_economic_residual_bps": h1["economic_residual_bps"],
                    "h1_false_positive_loss_bps": h1["false_positive_loss_bps"],
                    "h2_period_count": h2["period_count"], "h2_supported_period_count": h2["supported_period_count"],
                    "h2_observed_variance": h2["observed_variance"], "h2_sampling_variance": h2["sampling_variance"],
                    "h2_excess_variance": h2["excess_variance"], "h2_robust_z_excess_variance": h2.get("robust_z_excess_variance", np.nan),
                    "h2_sign_reversal_rate": h2["sign_reversal_rate"], "h2_worst_damage_bps": h2["worst_damage_bps"],
                    "h2_calibration_drift": h2["calibration_drift"], "h2_support_failure": h2["support_failure"], "h2_instability": h2["instability"],
                    "h2_classification": h2["classification"],
                    **{f"h3_{name}": value for name, value in h3.items()},
                    **{column: contribution[column] for column in context_columns},
                })
            resolved.append((pd.Timestamp(row["label_available_ts"]), event_id))
        resolved.sort(key=lambda item: (item[0].value, item[1]))
        position = end

    # Resolve only for retrospective audit artifacts; no candidate output has
    # ever read these events because the loop above has ended.
    while resolved:
        _, event_id = resolved.pop(0)
        apply_resolved(event_id)

    states = pd.DataFrame(state_rows)
    if states.empty:
        states = pd.DataFrame(columns=[
            *(_candidate_columns()), "feature_generation_ts", "label_available_ts",
            "feature_contract_sha256", "head_name", "rule_signature", "contribution_direction",
            "family_ensemble_tree_contribution",
        ])
    covariance, relationships, covariance_field_audit = _materialise_h4_h5(
        states, context_columns=context_columns, config=config
    )
    states = _append_h4_to_states(states, covariance)

    h1_metrics = (
        "posterior_correctness", "posterior_lower_95", "row_support", "timestamp_support",
        "day_support", "symbol_support", "support_score", "calibration_residual",
        "economic_residual_bps", "false_positive_loss_bps",
    )
    h2_metrics = (
        "period_count", "supported_period_count", "observed_variance", "sampling_variance",
        "excess_variance", "robust_z_excess_variance", "sign_reversal_rate", "worst_damage_bps",
        "calibration_drift", "support_failure", "instability",
    )
    h3_metrics = ("availability", "compatibility", "expected_error_bps", "confidence", "unexplained_break")
    h4_metrics = [name.removeprefix("base_health__h4__") for name in states if name.startswith("base_health__h4__")] + ["availability"]
    h1 = _aggregate_family_metrics(candidate_frame, states.rename(columns={f"h1_{name}": name for name in h1_metrics}), section="h1", metrics=h1_metrics)
    h2 = _aggregate_family_metrics(candidate_frame, states.rename(columns={f"h2_{name}": name for name in h2_metrics}), section="h2", metrics=h2_metrics)
    h3 = _aggregate_family_metrics(candidate_frame, states.rename(columns={f"h3_{name}": name for name in h3_metrics}), section="h3", metrics=h3_metrics)
    h4_renames = {f"base_health__h4__{name}": name for name in h4_metrics if name != "availability"}
    h4 = _aggregate_family_metrics(candidate_frame, states.rename(columns=h4_renames), section="h4", metrics=h4_metrics)
    h5 = _aggregate_h5(candidate_frame, relationships)
    health = _join_feature_sections((h1, h2, h3, h4, h5))
    health = health.sort_values(["transport", "meta_partition", "decision_ts", "candidate_id"], kind="stable").reset_index(drop=True)
    feature_columns = [name for name in health if name.startswith("base_health__")]
    if not np.isfinite(health[feature_columns].to_numpy(dtype=float)).all():
        raise CausalLeafHealthError("all emitted H1--H5 candidate features must be finite")
    _forbid_raw_leaf(health.columns, source="candidate health features")

    period_rows: list[dict[str, Any]] = []
    now = candidate_frame["feature_generation_ts"].max() + pd.Timedelta(hours=int(config.period_close_lag_hours) + 1)
    portability_final = _snapshot_portability(families, now, config)
    for key, stats in families.items():
        contract, side, head, signature, direction = key
        for period, value in sorted(stats.periods.items()):
            period_rows.append({
                "feature_contract_sha256": contract, "side_name": side, "head_name": head,
                "rule_signature": signature, "contribution_direction": direction, "period": period,
                "rows": value.rows, "independent_timestamps": len(value.timestamps),
                "trading_days": len(value.days), "symbols": len(value.symbols),
                "mean_prediction": value.prediction_sum / max(value.rows, 1),
                "posterior_correctness_raw": value.successes / max(value.rows, 1),
                "calibration_residual": value.calibration_sum / max(value.rows, 1),
                "mean_net_bps": value.net_sum / max(value.rows, 1),
                "economic_residual_bps": value.effect_bps(),
                "economic_residual_se_bps": value.effect_se_bps(),
                "false_positive_loss_bps": value.false_positive_loss_sum / max(value.rows, 1),
            })
    period_metrics = pd.DataFrame(period_rows)
    portability_rows = [
        {
            "feature_contract_sha256": key[0], "side_name": key[1], "head_name": key[2],
            "rule_signature": key[3], "contribution_direction": key[4], **metrics,
        }
        for key, metrics in portability_final.items()
    ]
    portability = pd.DataFrame(portability_rows)
    if not portability.empty:
        portability["is_portable"] = portability["classification"].eq("STABLE_PORTABLE")
    explainability = pd.DataFrame({
        "status": ["NOT_FITTED_IN_STATE_MATERIALISATION"],
        "reason": ["C5/C6 held-out explanatory regressions belong to the later transport ablation, not prequential feature generation"],
        "uses_outcomes": [False],
    })
    manifest = {
        "schema": SCHEMA,
        "status": STATUS,
        "contract": {
            "family_identity": "feature_contract_sha256, side, head, rule_signature, contribution_direction",
            "raw_leaf_ids": "rejected; only token-free same-artifact rule-family contributions are accepted",
            "history": "only label_available_ts < feature_generation_ts; all same-timestamp candidates are scored before any update",
            "periodicity": "H2 uses prior completed months only; H3 ridge snapshots are refit at month boundaries from prior-resolved rows only",
            "covariance": "H4 uses selected families, compact predeclared causal context, two horizons and no outcome fields",
            "relationship_breaks": "H5 uses predeclared causal relationship residuals, thresholded at the declared magnitude and weighted by portability/economic relevance",
        },
        "config": _config_payload(config),
        "row_counts": {
            "candidates": int(len(candidate_frame)), "family_candidate_states": int(len(states)),
            "health_features": int(len(health)), "period_metrics": int(len(period_metrics)),
            "portability_scores": int(len(portability)), "covariance_diagnostics": int(len(covariance)),
            "relationship_breaks": int(len(relationships)),
        },
        "context_columns": list(context_columns),
        "covariance_field_audit": covariance_field_audit.to_dict("records"),
    }
    return CausalLeafHealthResult(
        family_candidate_states=states, health_features=health,
        period_metrics=period_metrics, portability_scores=portability,
        covariance_diagnostics=covariance, relationship_breaks=relationships,
        covariance_explainability=explainability, manifest=manifest,
    )


def write_immutable_causal_leaf_health(
    result: CausalLeafHealthResult, output_dir: str | os.PathLike[str],
) -> Path:
    """Write all H1--H5 artifacts atomically; never overwrite a run root."""

    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite causal leaf health artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        tables = {
            "base_leaf_family_candidate_states.parquet": result.family_candidate_states,
            "base_leaf_health_features_oof.parquet": result.health_features,
            "leaf_period_metrics.parquet": result.period_metrics,
            "leaf_portability_scores.parquet": result.portability_scores,
            "leaf_covariance_diagnostics.parquet": result.covariance_diagnostics,
            "leaf_relationship_breaks.parquet": result.relationship_breaks,
            "covariance_explainability.parquet": result.covariance_explainability,
        }
        hashes: dict[str, str] = {}
        for name, table in tables.items():
            path = temporary / name
            table.to_parquet(path, index=False, compression="zstd")
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        (temporary / "leaf_covariance_reference_manifest.json").write_text(
            json.dumps({
                "schema": SCHEMA,
                "status": "CAUSAL_COVARIANCE_REFERENCE_DECLARED",
                "contract": result.manifest["contract"]["covariance"],
                "context_columns": result.manifest["context_columns"],
                "covariance_field_audit": result.manifest["covariance_field_audit"],
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        # JSON is valid YAML 1.2 and avoids introducing a serializer-only
        # dependency into this low-level strict contract.
        classifications = result.portability_scores.get("classification", pd.Series(dtype="string"))
        (temporary / "leaf_failure_classification.yaml").write_text(
            json.dumps({
                "schema": SCHEMA,
                "classification_counts": classifications.astype(str).value_counts().to_dict(),
                "labels": [
                    "STABLE_PORTABLE", "SUPPORT_SHIFT_ONLY", "CALIBRATION_DRIFT",
                    "COMPOSITION_DRIFT", "REGIME_CONDITIONAL", "COVARIANCE_CONDITIONAL",
                    "GLOBAL_PERIOD_SENSITIVE", "UNEXPLAINED_CONCEPT_BREAK",
                    "LOW_SUPPORT_UNCERTAIN", "META_HARMFUL",
                ],
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        payload = dict(result.manifest)
        payload["created_utc"] = datetime.now(timezone.utc).isoformat()
        payload["sha256"] = hashes
        (temporary / "health_materialization_manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8",
        )
        os.replace(temporary, target)
        return target
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


__all__ = [
    "CausalLeafHealthConfig", "CausalLeafHealthError", "CausalLeafHealthResult",
    "SCHEMA", "STATUS", "build_causal_leaf_health_states", "write_immutable_causal_leaf_health",
]
