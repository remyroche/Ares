"""Executable schema-v2 strict-R3 forward stack.

The module is deliberately artifact-path free.  A monthly bundle owns every
model, transform, score reference and contract required to reproduce a score.
Future outcomes are accepted by the training APIs only; the scoring APIs reject
them.  Schema-v1 research artifacts are therefore unable to enter this path by
accident.
"""

from __future__ import annotations

import hashlib
import json
import math
import gc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker
from sklearn.cluster import MiniBatchKMeans
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import OneHotEncoder

from .stage_i_causal_admission import (
    Causal21dAdmissionSpec,
    apply_causal_21d_side_admission,
)


SCHEMA = "strict_r3_canonical_forward_v2"
BUNDLE_SCHEMA = "strict_r3_canonical_monthly_bundle_v2"
GEOMETRY_SCHEMA = "strict_r3_geometry_k9_oct_dec_2024_v2"
POLICY_RESIDUAL_GEOMETRY_SCHEMA = "strict_r3_geometry_k9_oct_dec_2024_policy_residual_v1"
# Versioned research representation: identical Oct--Dec 2024 definition
# window and K9 semantics, plus frozen within-cluster encoder-geometry break
# references.  It must never silently substitute for the schema-v2 scorer.
STRUCTURAL_BREAK_GEOMETRY_SCHEMA = "strict_r3_geometry_k9_oct_dec_2024_structural_break_v1"
GEOMETRY_TARGET_H12_TP6_VS_BASE = "h12_tp6_sl4_net_gt_prequential_base_anchor"
GEOMETRY_TARGET_POLICY_RESIDUAL = "policy_net_gt_prequential_base_anchor_plus_hurdle"
SEED = 20260817
GEOMETRY_START = pd.Timestamp("2024-10-01", tz="UTC")
GEOMETRY_END = pd.Timestamp("2025-01-01", tz="UTC")
BASE_TRAIN_CAP = 240_000
GEOMETRY_TRAIN_CAP = 240_000
K9_TRAIN_CAP = 100_000
GEOMETRY_TREES = 64
K9_CLUSTERS = 9
K9_STRUCTURAL_BREAK_DIM = 8
RESIDUAL_CAPS = (40, 60, 80, 100, 120)
RESIDUAL_WEIGHT_MODES = ("ordinary", "equal_month")
RESIDUAL_BANDS_BPS = (-150.0, -50.0, 50.0, 150.0)
FORBIDDEN_SCORING_TOKENS = (
    "future", "outcome", "label_valid", "label_available", "target_invalid",
    "h12", "policy_net", "policy_gross", "tp6", "sl4", "mfe", "mae",
)

BASE_PARAMS: dict[str, Any] = {
    "objective": "multiclass", "num_class": 3, "n_estimators": 220,
    "learning_rate": 0.035, "max_depth": 5, "num_leaves": 24,
    "min_child_samples": 2400, "colsample_bytree": 0.85,
    "reg_lambda": 20.0, "subsample": 1.0, "random_state": SEED,
    "n_jobs": 1, "deterministic": True, "force_col_wise": True,
    "verbosity": -1,
}
RANK_PARAMS: dict[str, Any] = {
    "objective": "lambdarank", "n_estimators": 120,
    "learning_rate": 0.035, "max_depth": 5, "num_leaves": 31,
    "min_child_samples": 300, "colsample_bytree": 0.82,
    "subsample": 0.82, "subsample_freq": 1, "reg_alpha": 0.02,
    "reg_lambda": 2.0, "max_bin": 127,
    "label_gain": [0, 0.25, 1, 3, 7],
    "lambdarank_truncation_level": 10, "random_state": SEED,
    "n_jobs": 1, "deterministic": True, "force_col_wise": True,
    "verbosity": -1,
}
GEOMETRY_PARAMS: dict[str, Any] = {
    "objective": "binary", "n_estimators": GEOMETRY_TREES,
    "learning_rate": 0.04, "max_depth": 5, "num_leaves": 31,
    "min_child_samples": 350, "colsample_bytree": 0.80,
    "subsample": 0.80, "subsample_freq": 1, "reg_lambda": 8.0,
    "random_state": SEED + 1, "n_jobs": 1, "deterministic": True,
    "force_col_wise": True, "verbosity": -1,
}
SEVERE_PARAMS: dict[str, Any] = {
    "objective": "binary", "n_estimators": 35,
    "learning_rate": 0.0444772418995553, "max_depth": 5,
    "num_leaves": 15, "min_child_samples": 103,
    "colsample_bytree": 0.7393319822815638,
    "subsample": 0.7853518403594505, "subsample_freq": 1,
    "reg_alpha": 0.02534130367151813,
    "reg_lambda": 16.57892339556902, "max_bin": 127,
    "random_state": SEED + 2, "n_jobs": 1, "deterministic": True,
    "force_col_wise": True, "verbosity": -1,
}


@dataclass(frozen=True)
class FrozenPolicyContract:
    entry_delay_hours: int = 1
    stop_loss_atr: float = 3.0
    trailing_activation_atr: float = 0.5
    trailing_giveback_atr: float = 0.25
    timeout_hours: int = 12
    cost_bps_once: float = 100.0
    source_bar_minutes: int = 15


@dataclass(frozen=True)
class CandidateSpec:
    spread_limit_bps: float
    required_feature_fraction: float = 1.0
    side_names: tuple[str, ...] = ("long", "short")

    def __post_init__(self) -> None:
        if self.spread_limit_bps <= 0.0:
            raise ValueError("spread_limit_bps must be positive")
        if not 0.0 < self.required_feature_fraction <= 1.0:
            raise ValueError("required_feature_fraction must be in (0, 1]")
        if not self.side_names:
            raise ValueError("at least one side is required")


@dataclass
class ScoreReference:
    """Immutable empirical CDF reference fitted outside the held window."""

    sorted_values: np.ndarray
    source: str

    @classmethod
    def fit(cls, values: Sequence[float], *, source: str) -> "ScoreReference":
        array = np.asarray(values, dtype=float)
        array = np.sort(array[np.isfinite(array)], kind="stable")
        if len(array) < 2:
            raise ValueError(f"{source} has insufficient finite score support")
        return cls(array, source)

    def cdf(self, values: Sequence[float]) -> np.ndarray:
        current = np.asarray(values, dtype=float)
        result = np.full(len(current), np.nan, dtype=float)
        valid = np.isfinite(current)
        if valid.any():
            left = np.searchsorted(self.sorted_values, current[valid], side="left")
            right = np.searchsorted(self.sorted_values, current[valid], side="right")
            result[valid] = (0.5 * (left + right) + 0.5) / len(self.sorted_values)
        return np.clip(result, 0.0, 1.0)


@dataclass
class BinnedPolicyNetMap:
    model: IsotonicRegression
    bin_x: np.ndarray
    bin_y: np.ndarray
    bin_support: np.ndarray
    source_rows: int

    def predict(self, rank_values: Sequence[float]) -> np.ndarray:
        values = np.asarray(rank_values, dtype=float)
        output = np.full(len(values), np.nan, dtype=float)
        valid = np.isfinite(values)
        if valid.any():
            output[valid] = self.model.predict(values[valid])
        return output


@dataclass
class ResidualHead:
    name: str
    cap: int
    weight_mode: str
    model: LGBMRanker
    score_reference: ScoreReference


@dataclass
class FrozenGeometryK9:
    encoder_fields: tuple[str, ...]
    medians: np.ndarray
    encoder: LGBMClassifier
    leaf_categories: tuple[tuple[int, ...], ...]
    leaf_support_counts: tuple[np.ndarray, ...]
    one_hot: OneHotEncoder
    kmeans: MiniBatchKMeans
    cluster_order: np.ndarray
    temperature: float
    state_history: pd.DataFrame
    fit_audit: dict[str, Any]
    # Stable, soft-K9-cluster reference statistics.  These are deliberately
    # cluster-level quantities: no active leaf identity or leaf-path feature
    # is exported to downstream models.
    cluster_fit_support: np.ndarray | None = None
    cluster_distance_mean: np.ndarray | None = None
    cluster_distance_var: np.ndarray | None = None
    cluster_membership_mean: np.ndarray | None = None
    cluster_membership_covariance: np.ndarray | None = None
    cluster_membership_correlation: np.ndarray | None = None
    # Frozen feature-geometry references, separate from the existing
    # timestamp-wide covariance of K9 memberships.  They quantify how the
    # *features represented by an active K9 cluster* depart from that
    # cluster's Oct--Dec training geometry.
    structural_median: np.ndarray | None = None
    structural_scale: np.ndarray | None = None
    structural_projection: np.ndarray | None = None
    cluster_structural_covariance: np.ndarray | None = None
    cluster_structural_correlation: np.ndarray | None = None
    cluster_structural_support: np.ndarray | None = None
    bundle_sha256: str = ""

    @property
    def severe_structural_fields(self) -> tuple[str, ...]:
        """Original 45-field geometry contract consumed by Severe-200.

        It is frozen at schema-v2 creation.  Later N5-only aggregate state
        must never silently change the 123-field Severe input contract.
        """
        rule = (
            "rule_support_effective", "rule_support_p05", "rule_support_p50",
            "rule_support_median", "rule_support_p95",
            "rule_support_contribution_weighted",
            "rule_support_adequate_fraction", "rule_support_leaf_coverage",
            "rule_ood_marginal", "rule_ood_joint_factorised",
        )
        k9 = tuple(
            f"k09__cluster_{cluster:02d}__{suffix}"
            for cluster in range(K9_CLUSTERS)
            for suffix in ("membership", "negative_distance", "confidence")
        )
        path = (
            "path_support_effective_28d", "path_support_adequate_fraction",
            "path_ood_marginal", "path_ood_conditioned",
            "model_ood_marginal", "model_ood_mahalanobis_diag",
            "model_drift_prototype_psi", "model_drift_prototype_ks",
        )
        return rule + k9 + path

    @property
    def structural_fields(self) -> tuple[str, ...]:
        k9_weighted = (
            "k9_cluster_weighted_fit_support",
            "k9_cluster_weighted_fit_support_log",
            "k9_cluster_support_adequate_fraction",
            "k9_cluster_weighted_distance",
            "k9_cluster_weighted_ood",
            "k9_cluster_weighted_mahalanobis_train",
            "k9_cluster_timestamp_cov_break_train",
            "k9_cluster_timestamp_corr_break_train",
            "k9_cluster_timestamp_mahalanobis_train",
            "k9_cluster_timestamp_support_weighted",
            "k9_cluster_timestamp_support_p05",
            "k9_cluster_timestamp_ood_weighted",
        )
        within_cluster = ()
        if getattr(self, "cluster_structural_covariance", None) is not None:
            within_cluster = (
                "k9_cluster_activation_weighted_within_cov_break_train",
                "k9_cluster_activation_weighted_within_corr_break_train",
                "k9_cluster_activation_weighted_within_support_train",
            )
        return self.severe_structural_fields + k9_weighted + within_cluster

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        return _numeric_matrix(frame, self.encoder_fields, self.medians)

    def _leaves_membership(
        self,
        frame: pd.DataFrame,
        *,
        temperature_scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        leaves = np.asarray(self.encoder.predict(self._matrix(frame), pred_leaf=True), dtype=np.int32)
        if leaves.shape[1] != GEOMETRY_TREES:
            raise AssertionError("geometry tree count changed")
        encoded = self.one_hot.transform(leaves)
        distances = self.kmeans.transform(encoded).astype(np.float32)[:, self.cluster_order]
        if not np.isfinite(float(temperature_scale)) or float(temperature_scale) <= 0.0:
            raise ValueError("geometry temperature_scale must be finite and positive")
        effective_temperature = float(self.temperature) * float(temperature_scale)
        logits = -distances / max(effective_temperature, 1e-6)
        logits -= logits.max(axis=1, keepdims=True)
        membership = np.exp(logits, dtype=np.float32)
        membership /= np.maximum(membership.sum(axis=1, keepdims=True), 1e-12)
        return leaves, distances, membership

    def transform(
        self,
        frame: pd.DataFrame,
        *,
        temperature_scale: float = 1.0,
    ) -> pd.DataFrame:
        _require_columns(frame, ["__decision_ts__", *self.encoder_fields], "geometry scoring")
        leaves, distances, membership = self._leaves_membership(
            frame, temperature_scale=temperature_scale,
        )
        support_columns: list[np.ndarray] = []
        for tree in range(GEOMETRY_TREES):
            counts = self.leaf_support_counts[tree]
            values = leaves[:, tree]
            supported = values < len(counts)
            column = np.zeros(len(values), dtype=np.float32)
            column[supported] = counts[values[supported]]
            support_columns.append(column)
        support = np.column_stack(support_columns)
        denominator = float(self.fit_audit["complete_warmup_rows"])
        surprise = -np.log(np.clip((support + 1.0) / (denominator + 1.0), 1e-12, 1.0))
        values: dict[str, np.ndarray] = {
            "rule_support_effective": support.mean(axis=1),
            "rule_support_p05": np.quantile(support, 0.05, axis=1),
            "rule_support_p50": np.quantile(support, 0.50, axis=1),
            "rule_support_median": np.quantile(support, 0.50, axis=1),
            "rule_support_p95": np.quantile(support, 0.95, axis=1),
            "rule_support_contribution_weighted": support.mean(axis=1),
            "rule_support_adequate_fraction": (support >= 30.0).mean(axis=1),
            "rule_support_leaf_coverage": (support > 0.0).mean(axis=1),
            "rule_ood_marginal": surprise.mean(axis=1),
            "rule_ood_joint_factorised": surprise.mean(axis=1),
        }
        for cluster in range(K9_CLUSTERS):
            prefix = f"k09__cluster_{cluster:02d}"
            values[f"{prefix}__membership"] = membership[:, cluster]
            values[f"{prefix}__negative_distance"] = -distances[:, cluster]
            values[f"{prefix}__confidence"] = membership[:, cluster] ** 2
        output = pd.DataFrame(values, index=frame.index, dtype=np.float32)
        dynamic = _dynamic_k9_state(
            pd.to_datetime(frame["__decision_ts__"], utc=True),
            membership,
            history=self.state_history,
        )
        weighted = _k9_weighted_cluster_state(
            pd.to_datetime(frame["__decision_ts__"], utc=True),
            distances,
            membership,
            cluster_fit_support=self.cluster_fit_support,
            cluster_distance_mean=self.cluster_distance_mean,
            cluster_distance_var=self.cluster_distance_var,
            cluster_membership_mean=self.cluster_membership_mean,
            cluster_membership_covariance=self.cluster_membership_covariance,
            cluster_membership_correlation=self.cluster_membership_correlation,
            index=frame.index,
        )
        within_cluster = pd.DataFrame(index=frame.index)
        if getattr(self, "cluster_structural_covariance", None) is not None:
            matrix = self._matrix(frame)
            within_cluster = _k9_weighted_within_cluster_geometry_breaks(
                pd.to_datetime(frame["__decision_ts__"], utc=True), matrix, membership,
                structural_median=getattr(self, "structural_median", None),
                structural_scale=getattr(self, "structural_scale", None),
                structural_projection=getattr(self, "structural_projection", None),
                cluster_covariance=getattr(self, "cluster_structural_covariance", None),
                cluster_correlation=getattr(self, "cluster_structural_correlation", None),
                cluster_support=getattr(self, "cluster_structural_support", None),
                index=frame.index,
            )
        output = pd.concat([output, dynamic, weighted, within_cluster], axis=1)
        if tuple(output.columns) != self.structural_fields:
            raise AssertionError("geometry transform changed its ordered field contract")
        return output.astype(np.float32)


@dataclass
class CanonicalMonthlyBundle:
    cutoff: pd.Timestamp
    side_name: str
    base_fields: tuple[str, ...]
    context_fields: tuple[str, ...]
    severe_fields: tuple[str, ...]
    base_medians: np.ndarray
    base_model: LGBMClassifier
    policy_net_map: BinnedPolicyNetMap
    residual_heads: tuple[ResidualHead, ...]
    geometry: FrozenGeometryK9
    severe_medians: np.ndarray
    severe_model: LGBMClassifier
    policy: FrozenPolicyContract = field(default_factory=FrozenPolicyContract)
    schema: str = BUNDLE_SCHEMA
    manifest: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cutoff = _utc(self.cutoff)
        self.side_name = str(self.side_name).lower()
        if self.side_name not in {"long", "short"}:
            raise ValueError("monthly bundle must be side-local")
        if len(self.base_fields) != 120:
            raise ValueError(f"schema-v2 requires exactly 120 base fields, got {len(self.base_fields)}")
        if len(self.residual_heads) != 10:
            raise ValueError("schema-v2 requires exactly ten residual heads")
        if len(self.severe_fields) != 123:
            raise ValueError(f"schema-v2 requires exactly 123 Severe fields, got {len(self.severe_fields)}")


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], purpose: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{purpose} lacks required columns: {missing[:20]}")


def _numeric_matrix(
    frame: pd.DataFrame, fields: Sequence[str], medians: Sequence[float] | None = None,
) -> np.ndarray:
    values = frame.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians_array = values.median().fillna(0.0).to_numpy(dtype=np.float32)
    else:
        medians_array = np.asarray(medians, dtype=np.float32)
        if len(medians_array) != len(fields):
            raise ValueError("imputation medians do not match field contract")
    return values.fillna(pd.Series(medians_array, index=fields)).fillna(0.0).to_numpy(dtype=np.float32)


def _fit_medians(frame: pd.DataFrame, fields: Sequence[str]) -> np.ndarray:
    values = frame.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return values.median().fillna(0.0).to_numpy(dtype=np.float32)


def _json_hash(payload: Any) -> str:
    serialised = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(serialised).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_scoring_frame_is_target_free(frame: pd.DataFrame) -> None:
    allowed = {"candidate_id"}
    offending = [
        column for column in frame.columns
        if column not in allowed and any(token in column.lower() for token in FORBIDDEN_SCORING_TOKENS)
    ]
    if offending:
        raise ValueError(f"scoring frame contains outcome/future fields: {sorted(offending)[:20]}")


def require_single_geometry_hash(frame: pd.DataFrame) -> str:
    """Return the one frozen geometry identity or fail closed."""
    if "geometry_bundle_sha256" not in frame:
        raise ValueError("schema-v2 replay requires geometry_bundle_sha256 on every scored row")
    if frame["geometry_bundle_sha256"].isna().any():
        raise ValueError("geometry_bundle_sha256 is null on one or more scored rows")
    hashes = sorted(frame["geometry_bundle_sha256"].astype(str).unique().tolist())
    if len(hashes) != 1 or not hashes[0]:
        raise ValueError(
            "all replay months must use one identical frozen geometry/K9 bundle; "
            f"found {hashes}"
        )
    return hashes[0]


def build_point_in_time_candidates(
    market: pd.DataFrame,
    *,
    universe: Sequence[str],
    feature_fields: Sequence[str],
    cross_sectional_sources: Sequence[str],
    spec: CandidateSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build target-free symbol × signal-hour rows and a rejection ledger.

    Cross-sectional transforms are computed on the complete supplied market
    snapshot before any spread, entry or feature eligibility filter is applied.
    Columns not explicitly declared as decision-time inputs are ignored.
    """
    market = market.copy()
    if "__ts__" not in market and "__decision_ts__" in market:
        market["__ts__"] = pd.to_datetime(market["__decision_ts__"], utc=True) - pd.Timedelta(hours=1)
    if "__decision_ts__" not in market and "__ts__" in market:
        market["__decision_ts__"] = pd.to_datetime(market["__ts__"], utc=True) + pd.Timedelta(hours=1)
    required = [
        "__ts__", "__decision_ts__", "__symbol__", "instrument_available",
        "spread_bps", "entry_executable", *feature_fields, *cross_sectional_sources,
    ]
    _require_columns(market, required, "point-in-time market panel")
    if len(set(universe)) != len(universe):
        raise ValueError("frozen universe contains duplicate symbols")
    safe_columns = list(dict.fromkeys(required))
    frame = market.loc[:, safe_columns].copy()
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    if not frame["__decision_ts__"].eq(frame["__ts__"] + pd.Timedelta(hours=1)).all():
        raise ValueError("candidate entry/decision timestamp must equal signal timestamp + one hour")
    frame = frame.loc[frame["__symbol__"].astype(str).isin(set(map(str, universe)))].copy()
    if frame.duplicated(["__ts__", "__symbol__"]).any():
        raise ValueError("market panel has duplicate timestamp/symbol identities")
    timestamps = pd.Index(sorted(frame["__ts__"].unique()))
    identity = pd.MultiIndex.from_product(
        [timestamps, list(map(str, universe))], names=["__ts__", "__symbol__"],
    ).to_frame(index=False)
    complete = identity.merge(frame, on=["__ts__", "__symbol__"], how="left", validate="one_to_one")
    complete["__decision_ts__"] = complete["__ts__"] + pd.Timedelta(hours=1)
    for source in cross_sectional_sources:
        numeric = pd.to_numeric(complete[source], errors="coerce")
        grouped = numeric.groupby(complete["__ts__"], sort=False)
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0.0, np.nan)
        complete[f"xs__{source}__z"] = ((numeric - mean) / std).fillna(0.0).astype(np.float32)
        complete[f"xs__{source}__rank"] = grouped.rank(pct=True, method="average").astype(np.float32)
    feature_columns = [*feature_fields, *[f"xs__{s}__{kind}" for s in cross_sectional_sources for kind in ("z", "rank")]]
    finite_fraction = (
        complete[feature_columns].apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan).notna().mean(axis=1)
        if feature_columns else pd.Series(1.0, index=complete.index)
    )
    reason = np.full(len(complete), "eligible", dtype=object)
    missing_market = complete["instrument_available"].isna()
    reason[missing_market] = "missing_point_in_time_market_row"
    unavailable = ~complete["instrument_available"].fillna(False).astype(bool)
    reason[(reason == "eligible") & unavailable] = "instrument_unavailable"
    invalid_spread = ~np.isfinite(pd.to_numeric(complete["spread_bps"], errors="coerce"))
    reason[(reason == "eligible") & invalid_spread] = "spread_unavailable"
    too_wide = pd.to_numeric(complete["spread_bps"], errors="coerce").gt(spec.spread_limit_bps)
    reason[(reason == "eligible") & too_wide] = "spread_above_frozen_limit"
    not_executable = ~complete["entry_executable"].fillna(False).astype(bool)
    reason[(reason == "eligible") & not_executable] = "entry_not_executable"
    insufficient = finite_fraction.lt(spec.required_feature_fraction)
    reason[(reason == "eligible") & insufficient] = "insufficient_decision_feature_coverage"
    complete["eligibility_reason"] = reason
    complete["decision_feature_fraction"] = finite_fraction.astype(np.float32)
    side_frames: list[pd.DataFrame] = []
    for side in spec.side_names:
        block = complete.copy()
        block["side_name"] = str(side)
        block["candidate_id"] = (
            block["__symbol__"].astype(str) + "|" + str(side) + "|"
            + block["__ts__"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        side_frames.append(block)
    population = pd.concat(side_frames, ignore_index=True).sort_values(
        ["__ts__", "__symbol__", "side_name"], kind="stable",
    ).reset_index(drop=True)
    if population["candidate_id"].duplicated().any():
        raise AssertionError("candidate identity is not unique")
    rejection = population.loc[population["eligibility_reason"].ne("eligible")].copy()
    eligible = population.loc[population["eligibility_reason"].eq("eligible")].copy()
    return population, eligible, rejection


def fit_policy_net_map(
    base_rank42: Sequence[float], policy_net_bps: Sequence[float], *, bins: int = 20,
    trim_fraction: float = 0.05,
) -> BinnedPolicyNetMap:
    score = np.asarray(base_rank42, dtype=float)
    target = np.asarray(policy_net_bps, dtype=float)
    valid = np.isfinite(score) & np.isfinite(target)
    score, target = score[valid], target[valid]
    if len(score) < max(100, bins * 4) or np.unique(score).size < 4:
        raise ValueError("policy-net map has insufficient prequential support")
    order = np.argsort(score, kind="stable")
    group = np.minimum(np.arange(len(order)) * bins // len(order), bins - 1)
    rows: list[tuple[float, float, int]] = []
    for bin_id in range(bins):
        positions = order[group == bin_id]
        if not len(positions):
            continue
        outcomes = np.sort(target[positions])
        trim = int(math.floor(len(outcomes) * trim_fraction))
        kept = outcomes[trim:len(outcomes) - trim] if len(outcomes) > 2 * trim else outcomes
        rows.append((float(np.median(score[positions])), float(kept.mean()), len(positions)))
    x = np.asarray([row[0] for row in rows])
    y = np.asarray([row[1] for row in rows])
    w = np.asarray([row[2] for row in rows])
    model = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, y, sample_weight=w)
    return BinnedPolicyNetMap(model, x, y, w.astype(np.int64), len(score))


def residual_grades(values: Sequence[float]) -> np.ndarray:
    residual = np.asarray(values, dtype=float)
    return np.select(
        [residual <= boundary for boundary in RESIDUAL_BANDS_BPS],
        [0, 1, 2, 3], default=4,
    ).astype(np.int8)


def _fit_ranker(
    frame: pd.DataFrame, fields: Sequence[str], grade: np.ndarray, *,
    cap: int, mode: str, medians: np.ndarray,
) -> tuple[LGBMRanker, ScoreReference]:
    timestamp = pd.to_datetime(frame["__decision_ts__"], utc=True)
    side = frame["side_name"].astype(str)
    query = timestamp.dt.floor("4h").astype(str) + "|" + side
    counts = query.value_counts()
    keep = query.map(counts).ge(2).to_numpy()
    if keep.sum() < 20 or np.unique(grade[keep]).size < 2:
        raise ValueError(f"residual head cap={cap} mode={mode} lacks query/class support")
    positions = np.flatnonzero(keep)
    order = np.argsort(query.iloc[positions].to_numpy(), kind="stable")
    positions = positions[order]
    ordered_query = query.iloc[positions].to_numpy()
    _, group_sizes = np.unique(ordered_query, return_counts=True)
    weights = None
    if mode == "equal_month":
        months = timestamp.dt.strftime("%Y-%m").iloc[positions]
        frequency = months.value_counts()
        weights = months.map(lambda month: 1.0 / float(frequency.loc[month])).to_numpy(float)
        weights *= len(weights) / max(weights.sum(), 1e-12)
    matrix = _numeric_matrix(frame, fields[:cap], medians[:cap])
    model = LGBMRanker(**RANK_PARAMS)
    model.fit(matrix[positions], grade[positions], group=group_sizes, sample_weight=weights)
    raw = model.predict(matrix)
    reference = ScoreReference.fit(raw, source=f"residual_training_distribution_cap{cap}_{mode}")
    return model, reference


def _equal_month_sample(frame: pd.DataFrame, cap: int, *, seed: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable")
    months = pd.to_datetime(frame["__decision_ts__"], utc=True).dt.strftime("%Y-%m")
    pieces: list[pd.DataFrame] = []
    grouped = list(frame.assign(__sample_month__=months).groupby("__sample_month__", sort=True))
    quota = int(math.ceil(cap / len(grouped)))
    for index, (_, block) in enumerate(grouped):
        pieces.append(block.sample(min(quota, len(block)), random_state=seed + index))
    sampled = pd.concat(pieces, ignore_index=True).drop(columns="__sample_month__")
    if len(sampled) > cap:
        sampled = sampled.sample(cap, random_state=seed + 997)
    return sampled.sort_values(["__decision_ts__", "candidate_id"], kind="stable")


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", dtype=np.float32, sparse_output=True)
    except TypeError:  # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", dtype=np.float32, sparse=True)


def _iter_slices(length: int, size: int = 50_000) -> Iterable[slice]:
    for start in range(0, length, size):
        yield slice(start, min(start + size, length))


def _geometry_definition_months(
    start: Any,
    end_exclusive: Any,
) -> tuple[pd.Timestamp, pd.Timestamp, set[str]]:
    """Validate a full-calendar-month geometry definition interval.

    The canonical schema uses October--December 2024, but an explicitly
    labelled research ablation can fit a *separate immutable* geometry on any
    later three-month definition interval.  Keeping this validation next to
    the fitter makes it impossible to accidentally use a partial or
    non-calendar definition as an interchangeable K9 state.
    """

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end_exclusive)
    start_ts = (
        start_ts.tz_localize("UTC")
        if start_ts.tzinfo is None
        else start_ts.tz_convert("UTC")
    )
    end_ts = (
        end_ts.tz_localize("UTC")
        if end_ts.tzinfo is None
        else end_ts.tz_convert("UTC")
    )
    if (
        start_ts != start_ts.normalize().replace(day=1)
        or end_ts != end_ts.normalize().replace(day=1)
        or end_ts <= start_ts
    ):
        raise ValueError("geometry definition must use complete UTC calendar months")
    months = pd.period_range(
        start_ts.tz_convert(None).to_period("M"),
        (end_ts - pd.Timedelta(nanoseconds=1)).tz_convert(None).to_period("M"),
        freq="M",
    )
    if len(months) != 3:
        raise ValueError("geometry definition must span exactly three complete months")
    return start_ts, end_ts, {str(month) for month in months}


def fit_frozen_geometry_k9(
    warmup: pd.DataFrame,
    *,
    encoder_fields: Sequence[str],
    seed: int = SEED,
    definition_start: Any = GEOMETRY_START,
    definition_end_exclusive: Any = GEOMETRY_END,
    target_mode: str = GEOMETRY_TARGET_H12_TP6_VS_BASE,
    policy_residual_hurdle_bps: float = 50.0,
) -> FrozenGeometryK9:
    common = [
        "candidate_id", "__decision_ts__", "prequential_base_anchor_bps",
        "stack_is_prequential", "geometry_definition_population_complete", *encoder_fields,
    ]
    if target_mode == GEOMETRY_TARGET_H12_TP6_VS_BASE:
        required = [
            *common, "h12_label_valid", "h12_label_available_ts", "h12_tp6_sl4_net_bps",
        ]
    elif target_mode == GEOMETRY_TARGET_POLICY_RESIDUAL:
        if not np.isfinite(float(policy_residual_hurdle_bps)):
            raise ValueError("policy_residual_hurdle_bps must be finite")
        required = [*common, "policy_label_available_ts", "policy_net_bps"]
    else:
        raise ValueError(f"unknown frozen geometry target mode: {target_mode}")
    _require_columns(warmup, required, "geometry warm-up")
    definition_start_ts, definition_end_ts, expected_months = _geometry_definition_months(
        definition_start, definition_end_exclusive,
    )
    frame = warmup.copy()
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
    if target_mode == GEOMETRY_TARGET_H12_TP6_VS_BASE:
        frame["h12_label_available_ts"] = pd.to_datetime(frame["h12_label_available_ts"], utc=True)
    else:
        frame["policy_label_available_ts"] = pd.to_datetime(frame["policy_label_available_ts"], utc=True)
    if not frame["geometry_definition_population_complete"].fillna(False).astype(bool).all():
        raise ValueError("geometry definition input is not the declared complete population")
    in_window = (
        frame["__decision_ts__"].ge(definition_start_ts)
        & frame["__decision_ts__"].lt(definition_end_ts)
    )
    valid_common = (
        in_window
        & np.isfinite(pd.to_numeric(frame["prequential_base_anchor_bps"], errors="coerce"))
        & frame["stack_is_prequential"].fillna(False).astype(bool)
    )
    if target_mode == GEOMETRY_TARGET_H12_TP6_VS_BASE:
        valid = (
            valid_common & frame["h12_label_valid"].fillna(False).astype(bool)
            & frame["h12_label_available_ts"].lt(definition_end_ts)
            & np.isfinite(pd.to_numeric(frame["h12_tp6_sl4_net_bps"], errors="coerce"))
        )
    else:
        valid = (
            valid_common & frame["policy_label_available_ts"].lt(definition_end_ts)
            & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
        )
    frame = frame.loc[valid].copy()
    if frame.empty or frame["candidate_id"].duplicated().any():
        raise ValueError("geometry warm-up is empty or has duplicate identities")
    months = frame["__decision_ts__"].dt.strftime("%Y-%m")
    if set(months.unique()) != expected_months:
        raise ValueError(
            "geometry warm-up must cover all three declared months: "
            f"expected={sorted(expected_months)} observed={sorted(months.unique())}",
        )
    if target_mode == GEOMETRY_TARGET_H12_TP6_VS_BASE:
        positive = (
            pd.to_numeric(frame["h12_tp6_sl4_net_bps"], errors="coerce")
            > pd.to_numeric(frame["prequential_base_anchor_bps"], errors="coerce")
        ).astype(np.int8)
    else:
        positive = (
            pd.to_numeric(frame["policy_net_bps"], errors="coerce")
            - pd.to_numeric(frame["prequential_base_anchor_bps"], errors="coerce")
            > float(policy_residual_hurdle_bps)
        ).astype(np.int8)
    medians = _fit_medians(frame, encoder_fields)
    encoder_sample = _equal_month_sample(frame.assign(__target__=positive), GEOMETRY_TRAIN_CAP, seed=seed)
    encoder = LGBMClassifier(**{**GEOMETRY_PARAMS, "random_state": seed + 1})
    encoder.fit(
        _numeric_matrix(encoder_sample, encoder_fields, medians),
        encoder_sample["__target__"].to_numpy(np.int8),
    )
    full_matrix = _numeric_matrix(frame, encoder_fields, medians)
    support_counts = [np.zeros(1, dtype=np.float32) for _ in range(GEOMETRY_TREES)]
    full_leaves_parts: list[np.ndarray] = []
    for chunk in _iter_slices(len(frame)):
        leaves = np.asarray(encoder.predict(full_matrix[chunk], pred_leaf=True), dtype=np.int32)
        full_leaves_parts.append(leaves)
        for tree in range(GEOMETRY_TREES):
            local = np.bincount(leaves[:, tree]).astype(np.float32)
            if len(local) > len(support_counts[tree]):
                support_counts[tree] = np.pad(support_counts[tree], (0, len(local) - len(support_counts[tree])))
            support_counts[tree][:len(local)] += local
    full_leaves = np.concatenate(full_leaves_parts, axis=0)
    one_hot = _one_hot_encoder()
    one_hot.fit(full_leaves)
    k9_sample = _equal_month_sample(frame, K9_TRAIN_CAP, seed=seed + 101)
    # Equal-month sampling may reset its index.  Candidate identity, not a
    # positional index, is the only valid join back to the complete warm-up.
    identity_position = pd.Series(np.arange(len(frame)), index=frame["candidate_id"])
    sample_positions = identity_position.loc[k9_sample["candidate_id"]].to_numpy(int)
    encoded_sample = one_hot.transform(full_leaves[sample_positions])
    kmeans = MiniBatchKMeans(
        n_clusters=K9_CLUSTERS, batch_size=4096, n_init=5,
        random_state=seed + 2,
    ).fit(encoded_sample)
    center_hash = [hashlib.sha256(np.asarray(center, dtype=np.float32).tobytes()).hexdigest() for center in kmeans.cluster_centers_]
    cluster_order = np.argsort(np.asarray(center_hash), kind="stable")
    nearest: list[np.ndarray] = []
    for chunk in _iter_slices(len(frame)):
        distance = kmeans.transform(one_hot.transform(full_leaves[chunk]))[:, cluster_order]
        nearest.append(distance.min(axis=1).astype(np.float32))
    temperature = max(float(np.median(np.concatenate(nearest))), 1e-3)
    membership_parts: list[pd.DataFrame] = []
    for chunk in _iter_slices(len(frame)):
        distance = kmeans.transform(one_hot.transform(full_leaves[chunk]))[:, cluster_order]
        logits = -distance / temperature
        logits -= logits.max(axis=1, keepdims=True)
        membership = np.exp(logits, dtype=np.float32)
        membership /= np.maximum(membership.sum(axis=1, keepdims=True), 1e-12)
        block = pd.DataFrame(membership, columns=[f"k{index}" for index in range(K9_CLUSTERS)])
        block["__decision_ts__"] = frame["__decision_ts__"].iloc[chunk].to_numpy()
        membership_parts.append(block)
    membership_frame = pd.concat(membership_parts, ignore_index=True)
    membership_matrix = membership_frame[[f"k{index}" for index in range(K9_CLUSTERS)]].to_numpy(np.float64)
    # These references are fixed once for the October--December definition
    # population.  They intentionally describe soft K9 clusters, not leaf
    # identities, so their meaning survives every later scoring fold.
    distance_parts: list[np.ndarray] = []
    for chunk in _iter_slices(len(frame)):
        distance_parts.append(
            kmeans.transform(one_hot.transform(full_leaves[chunk]))[:, cluster_order]
        )
    full_distances = np.concatenate(distance_parts, axis=0).astype(np.float64)
    cluster_mass = np.maximum(membership_matrix.sum(axis=0), 1e-12)
    cluster_distance_mean = (membership_matrix * full_distances).sum(axis=0) / cluster_mass
    cluster_distance_var = (
        membership_matrix * np.square(full_distances - cluster_distance_mean[None, :])
    ).sum(axis=0) / cluster_mass
    cluster_membership_mean = membership_matrix.mean(axis=0)
    cluster_membership_covariance = np.cov(membership_matrix, rowvar=False)
    cluster_membership_covariance = np.asarray(cluster_membership_covariance, dtype=np.float64)
    cluster_membership_covariance += np.eye(K9_CLUSTERS, dtype=np.float64) * 1e-6
    cluster_membership_std = np.sqrt(np.maximum(np.diag(cluster_membership_covariance), 1e-12))
    cluster_membership_correlation = cluster_membership_covariance / np.outer(
        cluster_membership_std, cluster_membership_std,
    )
    # Freeze an efficient representation of the *encoder input* geometry for
    # each soft K9 cluster.  It is deliberately fit on the full Oct--Dec
    # definition population, not re-estimated by scoring folds or by current
    # timestamps.  Eight projected dimensions are enough for covariance-break
    # detection while avoiding a prohibitively large covariance per cluster.
    structural_median = np.nanmedian(full_matrix, axis=0)
    structural_median = np.where(np.isfinite(structural_median), structural_median, 0.0)
    structural_q25 = np.nanquantile(full_matrix, 0.25, axis=0)
    structural_q75 = np.nanquantile(full_matrix, 0.75, axis=0)
    structural_scale = np.where(
        np.isfinite(structural_q75 - structural_q25) & ((structural_q75 - structural_q25) > 1e-6),
        structural_q75 - structural_q25,
        1.0,
    )
    structural_projection = _structural_projection(len(encoder_fields), seed=seed)
    structural_values = np.clip(
        (full_matrix - structural_median) / structural_scale,
        -8.0, 8.0,
    ) @ structural_projection
    cluster_structural_covariance: list[np.ndarray] = []
    cluster_structural_correlation: list[np.ndarray] = []
    for cluster in range(K9_CLUSTERS):
        covariance, _mass = _weighted_covariance(structural_values, membership_matrix[:, cluster])
        cluster_structural_covariance.append(covariance)
        cluster_structural_correlation.append(_covariance_correlation(covariance))
    state_history = membership_frame.groupby("__decision_ts__", sort=True)[
        [f"k{index}" for index in range(K9_CLUSTERS)]
    ].sum().reset_index()
    month_counts = months.loc[frame.index].value_counts().sort_index().to_dict()
    encoder_month_counts = encoder_sample["__decision_ts__"].dt.strftime("%Y-%m").value_counts().sort_index().to_dict()
    k9_month_counts = k9_sample["__decision_ts__"].dt.strftime("%Y-%m").value_counts().sort_index().to_dict()
    audit = {
        "schema": GEOMETRY_SCHEMA,
        "definition_start": definition_start_ts.isoformat(),
        "definition_end_exclusive": definition_end_ts.isoformat(),
        "complete_warmup_rows": int(len(frame)),
        "encoder_fit_rows": int(len(encoder_sample)),
        "k9_fit_rows": int(len(k9_sample)),
        "month_rows": {str(key): int(value) for key, value in month_counts.items()},
        "encoder_sample_month_rows": {str(key): int(value) for key, value in encoder_month_counts.items()},
        "k9_sample_month_rows": {str(key): int(value) for key, value in k9_month_counts.items()},
        "natural_positive_rate": float(positive.mean()),
        "geometry_target_mode": str(target_mode),
        "policy_residual_hurdle_bps": (
            None if target_mode != GEOMETRY_TARGET_POLICY_RESIDUAL
            else float(policy_residual_hurdle_bps)
        ),
        "encoder_sample_positive_rate": float(encoder_sample["__target__"].mean()),
        "temperature": temperature,
        "temperature_fit_rows": int(len(frame)),
        "cluster_reference_rows": int(len(frame)),
        "cluster_fit_support": [float(value) for value in cluster_mass],
        "within_cluster_geometry_break": {
            "enabled": True,
            "projected_dimensions": int(structural_projection.shape[1]),
            "fit_population": "complete_oct_dec_2024_definition",
            "reference": "soft-membership-weighted encoder-input covariance/correlation per frozen K9 cluster",
            "projection_sha256": hashlib.sha256(np.asarray(structural_projection, dtype=np.float32).tobytes()).hexdigest(),
            "cluster_covariance_sha256": hashlib.sha256(np.asarray(cluster_structural_covariance, dtype=np.float32).tobytes()).hexdigest(),
        },
        "encoder_fields": list(encoder_fields),
        "encoder_fields_sha256": _json_hash(list(encoder_fields)),
        "seed": seed,
    }
    return FrozenGeometryK9(
        encoder_fields=tuple(encoder_fields), medians=medians, encoder=encoder,
        leaf_categories=tuple(tuple(int(value) for value in category) for category in one_hot.categories_),
        leaf_support_counts=tuple(support_counts), one_hot=one_hot, kmeans=kmeans,
        cluster_order=cluster_order.astype(np.int16), temperature=temperature,
        state_history=state_history, fit_audit=audit,
        cluster_fit_support=cluster_mass.astype(np.float32),
        cluster_distance_mean=cluster_distance_mean.astype(np.float32),
        cluster_distance_var=cluster_distance_var.astype(np.float32),
        cluster_membership_mean=cluster_membership_mean.astype(np.float32),
        cluster_membership_covariance=cluster_membership_covariance.astype(np.float32),
        cluster_membership_correlation=cluster_membership_correlation.astype(np.float32),
        structural_median=np.asarray(structural_median, dtype=np.float32),
        structural_scale=np.asarray(structural_scale, dtype=np.float32),
        structural_projection=np.asarray(structural_projection, dtype=np.float32),
        cluster_structural_covariance=np.asarray(cluster_structural_covariance, dtype=np.float32),
        cluster_structural_correlation=np.asarray(cluster_structural_correlation, dtype=np.float32),
        cluster_structural_support=cluster_mass.astype(np.float32),
    )


def _dynamic_k9_state(
    timestamp: pd.Series, membership: np.ndarray, *, history: pd.DataFrame,
) -> pd.DataFrame:
    names = [f"k{index}" for index in range(K9_CLUSTERS)]
    current = pd.DataFrame(membership, columns=names)
    current["__decision_ts__"] = pd.to_datetime(timestamp, utc=True).to_numpy()
    current_state = current.groupby("__decision_ts__", sort=True)[names].sum()
    historic = history.copy()
    historic["__decision_ts__"] = pd.to_datetime(historic["__decision_ts__"], utc=True)
    historic = historic.set_index("__decision_ts__")[names]
    first = current_state.index.min()
    historic = historic.loc[historic.index < first]
    state = pd.concat([historic, current_state]).sort_index()
    state = state.loc[~state.index.duplicated(keep="last")]
    prior = state.shift(1).fillna(0.0)
    rolling = prior.rolling("28D", min_periods=1).sum()
    current_prob = state.div(state.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    reference_prob = rolling.div(rolling.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    mean = prior.rolling("28D", min_periods=4).mean()
    std = prior.rolling("28D", min_periods=8).std().replace(0.0, np.nan)
    z = (state - mean) / std
    drift = pd.DataFrame(index=state.index)
    drift["model_ood_marginal"] = z.abs().mean(axis=1)
    drift["model_ood_mahalanobis_diag"] = np.sqrt((z * z).mean(axis=1))
    drift["model_drift_prototype_psi"] = (
        (current_prob - reference_prob)
        * np.log(np.clip(current_prob, 1e-12, None) / np.clip(reference_prob, 1e-12, None))
    ).sum(axis=1)
    drift["model_drift_prototype_ks"] = (
        current_prob.cumsum(axis=1) - reference_prob.cumsum(axis=1)
    ).abs().max(axis=1)
    rows = pd.DataFrame({"__decision_ts__": pd.to_datetime(timestamp, utc=True)}, index=timestamp.index)
    rolling_rows = rows.merge(rolling.reset_index(), on="__decision_ts__", how="left", validate="many_to_one")[names].fillna(0.0).to_numpy(np.float32)
    drift_rows = rows.merge(drift.reset_index(), on="__decision_ts__", how="left", validate="many_to_one").drop(columns="__decision_ts__").fillna(0.0)
    total = np.maximum(rolling_rows.sum(axis=1, keepdims=True), 1.0)
    surprise = -np.log(np.clip((rolling_rows + 1.0) / (total + K9_CLUSTERS), 1e-12, 1.0))
    path = pd.DataFrame({
        "path_support_effective_28d": (membership * rolling_rows).sum(axis=1),
        "path_support_adequate_fraction": (membership * (rolling_rows >= 30.0)).sum(axis=1),
        "path_ood_marginal": (membership * surprise).sum(axis=1),
        "path_ood_conditioned": (
            (membership * surprise * (rolling_rows >= 30.0)).sum(axis=1)
            / np.maximum((membership * (rolling_rows >= 30.0)).sum(axis=1), 1e-12)
        ),
    }, index=timestamp.index)
    drift_rows.index = timestamp.index
    return pd.concat([path, drift_rows], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def _structural_projection(feature_count: int, *, seed: int) -> np.ndarray:
    """Fixed low-dimensional proxy for the geometry encoder's input space."""

    if feature_count < 1:
        raise ValueError("structural projection needs at least one feature")
    width = min(K9_STRUCTURAL_BREAK_DIM, int(feature_count))
    rng = np.random.default_rng(int(seed) + 811)
    projection = rng.standard_normal((int(feature_count), width)).astype(np.float64)
    projection /= np.maximum(np.linalg.norm(projection, axis=0, keepdims=True), 1e-12)
    return projection.astype(np.float32)


def _weighted_covariance(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, float]:
    """Regularised soft-membership covariance and effective support."""

    x = np.asarray(values, dtype=np.float64)
    w = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    mass = float(w.sum())
    width = x.shape[1]
    if mass <= 1e-12:
        return np.eye(width, dtype=np.float64), 0.0
    mean = (x * w[:, None]).sum(axis=0) / mass
    centered = x - mean
    covariance = (centered * w[:, None]).T @ centered / mass
    covariance = np.asarray(covariance, dtype=np.float64)
    covariance += np.eye(width, dtype=np.float64) * 1e-6
    return covariance, mass


def _covariance_correlation(covariance: np.ndarray) -> np.ndarray:
    standard_deviation = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
    return covariance / np.outer(standard_deviation, standard_deviation)


def _k9_weighted_within_cluster_geometry_breaks(
    timestamp: pd.Series,
    matrix: np.ndarray,
    membership: np.ndarray,
    *,
    structural_median: np.ndarray | None,
    structural_scale: np.ndarray | None,
    structural_projection: np.ndarray | None,
    cluster_covariance: np.ndarray,
    cluster_correlation: np.ndarray | None,
    cluster_support: np.ndarray | None,
    index: pd.Index,
) -> pd.DataFrame:
    """Causal activation-weighted *within-cluster* geometry break features.

    Every K9 cluster owns a frozen October--December feature-geometry
    covariance/correlation reference.  At a decision timestamp we calculate
    each active cluster's soft-membership-weighted current covariance, shrink
    sparse activation to that frozen reference, and report a candidate's own
    membership-weighted aggregate.  This is intentionally not a covariance of
    the nine membership coordinates themselves.
    """

    if structural_median is None or structural_scale is None or structural_projection is None:
        raise ValueError("within-cluster breaks need a complete frozen structural reference")
    x = np.asarray(matrix, dtype=np.float64)
    median = np.asarray(structural_median, dtype=np.float64)
    scale = np.maximum(np.asarray(structural_scale, dtype=np.float64), 1e-6)
    projection = np.asarray(structural_projection, dtype=np.float64)
    if x.shape[1] != len(median) or projection.shape[0] != x.shape[1]:
        raise ValueError("within-cluster geometry reference does not match encoder fields")
    projected = np.clip((x - median) / scale, -8.0, 8.0) @ projection
    reference_covariance = np.asarray(cluster_covariance, dtype=np.float64)
    reference_correlation = (
        np.asarray(cluster_correlation, dtype=np.float64)
        if cluster_correlation is not None
        else np.stack([_covariance_correlation(item) for item in reference_covariance], axis=0)
    )
    if reference_covariance.shape[0] != K9_CLUSTERS or membership.shape[1] != K9_CLUSTERS:
        raise ValueError("within-cluster geometry breaks require nine K9 clusters")
    reference_support = (
        np.maximum(np.asarray(cluster_support, dtype=np.float64), 1.0)
        if cluster_support is not None else np.ones(K9_CLUSTERS, dtype=np.float64)
    )
    output = np.zeros((len(projected), 3), dtype=np.float32)
    positions = pd.Series(np.arange(len(projected)), index=index)
    timestamps = pd.to_datetime(timestamp, utc=True, errors="raise")
    upper = np.triu_indices(projected.shape[1], k=1)
    for _current_timestamp, rows in positions.groupby(timestamps, sort=False):
        row = rows.to_numpy(dtype=int)
        local = projected[row]
        local_membership = np.asarray(membership[row], dtype=np.float64)
        cov_break = np.zeros(K9_CLUSTERS, dtype=np.float64)
        corr_break = np.zeros(K9_CLUSTERS, dtype=np.float64)
        current_support = np.zeros(K9_CLUSTERS, dtype=np.float64)
        for cluster in range(K9_CLUSTERS):
            current_covariance, mass = _weighted_covariance(local, local_membership[:, cluster])
            shrink = mass / (mass + 30.0)
            covariance = shrink * current_covariance + (1.0 - shrink) * reference_covariance[cluster]
            correlation = _covariance_correlation(covariance)
            covariance_scale = max(float(np.linalg.norm(reference_covariance[cluster], ord="fro")), 1e-8)
            cov_break[cluster] = float(np.linalg.norm(covariance - reference_covariance[cluster], ord="fro") / covariance_scale)
            corr_break[cluster] = float(np.sqrt(np.mean(np.square(correlation[upper] - reference_correlation[cluster][upper]))))
            current_support[cluster] = mass
        output[row, 0] = local_membership @ cov_break
        output[row, 1] = local_membership @ corr_break
        output[row, 2] = local_membership @ np.minimum(current_support / reference_support, 1.0)
    return pd.DataFrame(
        output,
        columns=(
            "k9_cluster_activation_weighted_within_cov_break_train",
            "k9_cluster_activation_weighted_within_corr_break_train",
            "k9_cluster_activation_weighted_within_support_train",
        ),
        index=index,
    )


def _k9_weighted_cluster_state(
    timestamp: pd.Series,
    distances: np.ndarray,
    membership: np.ndarray,
    *,
    cluster_fit_support: np.ndarray | None,
    cluster_distance_mean: np.ndarray | None,
    cluster_distance_var: np.ndarray | None,
    cluster_membership_mean: np.ndarray | None,
    cluster_membership_covariance: np.ndarray | None,
    cluster_membership_correlation: np.ndarray | None,
    index: pd.Index,
) -> pd.DataFrame:
    """Decision-time structural state, aggregated by soft K9 membership.

    The K9 state is fitted once with the geometry bundle.  Candidate-level
    fields are soft weighted across those nine persistent clusters; timestamp
    fields compare only the contemporaneous candidate distribution with the
    same frozen K9 reference.  Thus this function neither exposes raw K9
    labels nor reads outcomes, future rows, or rolling correctness.
    """
    n_rows = len(membership)
    if n_rows == 0:
        return pd.DataFrame(index=index, dtype=np.float32)
    support = (
        np.asarray(cluster_fit_support, dtype=np.float64)
        if cluster_fit_support is not None else np.ones(K9_CLUSTERS, dtype=np.float64)
    )
    distance_mean = (
        np.asarray(cluster_distance_mean, dtype=np.float64)
        if cluster_distance_mean is not None else np.zeros(K9_CLUSTERS, dtype=np.float64)
    )
    distance_var = (
        np.asarray(cluster_distance_var, dtype=np.float64)
        if cluster_distance_var is not None else np.ones(K9_CLUSTERS, dtype=np.float64)
    )
    membership_mean = (
        np.asarray(cluster_membership_mean, dtype=np.float64)
        if cluster_membership_mean is not None else np.full(K9_CLUSTERS, 1.0 / K9_CLUSTERS)
    )
    membership_covariance = (
        np.asarray(cluster_membership_covariance, dtype=np.float64)
        if cluster_membership_covariance is not None else np.eye(K9_CLUSTERS, dtype=np.float64)
    )
    membership_correlation = (
        np.asarray(cluster_membership_correlation, dtype=np.float64)
        if cluster_membership_correlation is not None else np.eye(K9_CLUSTERS, dtype=np.float64)
    )
    if support.shape != (K9_CLUSTERS,) or distance_mean.shape != (K9_CLUSTERS,):
        raise ValueError("frozen K9 cluster reference has invalid shape")
    z_distance = (np.asarray(distances, dtype=np.float64) - distance_mean[None, :]) / np.sqrt(
        np.maximum(distance_var[None, :], 1e-8)
    )
    # A cluster is only unusual when it lies farther than its frozen geometry
    # radius.  This avoids treating a nearer-than-usual K9 center as OOD.
    positive_z = np.maximum(z_distance, 0.0)
    weighted_support = (membership * support[None, :]).sum(axis=1)
    weighted_distance = (membership * distances).sum(axis=1)
    weighted_ood = (membership * positive_z).sum(axis=1)
    weighted_mahalanobis = np.sqrt((membership * np.square(z_distance)).sum(axis=1))
    adequate = (membership * (support[None, :] >= 30.0)).sum(axis=1)
    names = (
        "k9_cluster_weighted_fit_support",
        "k9_cluster_weighted_fit_support_log",
        "k9_cluster_support_adequate_fraction",
        "k9_cluster_weighted_distance",
        "k9_cluster_weighted_ood",
        "k9_cluster_weighted_mahalanobis_train",
        "k9_cluster_timestamp_cov_break_train",
        "k9_cluster_timestamp_corr_break_train",
        "k9_cluster_timestamp_mahalanobis_train",
        "k9_cluster_timestamp_support_weighted",
        "k9_cluster_timestamp_support_p05",
        "k9_cluster_timestamp_ood_weighted",
    )
    output = np.zeros((n_rows, len(names)), dtype=np.float32)
    output[:, 0] = weighted_support
    output[:, 1] = np.log1p(weighted_support)
    output[:, 2] = adequate
    output[:, 3] = weighted_distance
    output[:, 4] = weighted_ood
    output[:, 5] = weighted_mahalanobis
    timestamps = pd.to_datetime(timestamp, utc=True, errors="raise")
    positions = pd.Series(np.arange(n_rows), index=index)
    for _current_timestamp, rows in positions.groupby(timestamps, sort=False):
        row = rows.to_numpy(dtype=int)
        current = membership[row]
        current_mean = current.mean(axis=0)
        current_covariance = np.cov(current, rowvar=False) if len(row) > 1 else membership_covariance
        current_covariance = np.asarray(current_covariance, dtype=np.float64)
        alpha = len(row) / (len(row) + 30.0)
        current_covariance = alpha * current_covariance + (1.0 - alpha) * membership_covariance
        current_std = np.sqrt(np.maximum(np.diag(current_covariance), 1e-12))
        current_correlation = current_covariance / np.outer(current_std, current_std)
        delta = current_mean - membership_mean
        timestamp_mahalanobis = np.sqrt(
            np.mean(np.square(delta) / np.maximum(np.diag(membership_covariance), 1e-8))
        )
        upper = np.triu_indices(K9_CLUSTERS, k=1)
        cov_scale = max(float(np.linalg.norm(membership_covariance, ord="fro")), 1e-8)
        cov_break = float(np.linalg.norm(current_covariance - membership_covariance, ord="fro") / cov_scale)
        corr_break = float(np.sqrt(np.mean(np.square(
            current_correlation[upper] - membership_correlation[upper],
        ))))
        output[row, 6] = cov_break
        output[row, 7] = corr_break
        output[row, 8] = timestamp_mahalanobis
        output[row, 9] = float(np.mean(weighted_support[row]))
        output[row, 10] = float(np.quantile(weighted_support[row], 0.05))
        output[row, 11] = float(np.mean(weighted_ood[row]))
    return pd.DataFrame(output, columns=names, index=index).replace(
        [np.inf, -np.inf], np.nan,
    ).fillna(0.0).astype(np.float32)


def _validate_prequential_ledger(ledger: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    required = [
        "candidate_id", "__decision_ts__", "side_name", "stack_is_prequential",
        "policy_label_available_ts", "policy_net_bps", "h12_label_available_ts",
        "h12_label_valid", "h12_tp6_sl4_net_bps", "prequential_base_rank42",
        "prequential_base_anchor_bps", "prequential_consensus_rank",
        "prequential_residual_rank", "prequential_upstream",
    ]
    _require_columns(ledger, required, "prequential stack ledger")
    out = ledger.copy()
    for column in ("__decision_ts__", "policy_label_available_ts", "h12_label_available_ts"):
        out[column] = pd.to_datetime(out[column], utc=True, errors="raise")
    if out["candidate_id"].duplicated().any():
        raise ValueError("prequential ledger has duplicate candidate IDs")
    if not out["stack_is_prequential"].fillna(False).astype(bool).all():
        raise ValueError("downstream fitting received non-prequential rows")
    if (out["__decision_ts__"] >= cutoff).any():
        raise ValueError("prequential training ledger crosses held cutoff")
    return out


def build_prequential_stack_ledger(
    panel: pd.DataFrame,
    *,
    base_fields: Sequence[str],
    first_held_month: Any,
    last_held_month: Any | None = None,
    reference_days: int = 28,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build monthly OOF/prequential base-map-residual handoffs.

    The function performs two chronological passes.  Pass one produces strict
    monthly base OOF scores.  Pass two fits the policy map and residual heads
    only from earlier pass-one outputs whose policy labels have resolved.
    """
    required = [
        "candidate_id", "__decision_ts__", "side_name", "r3_class",
        "r3_label_available_ts", "policy_net_bps", "policy_label_available_ts",
        "h12_label_valid", "h12_label_available_ts", "h12_tp6_sl4_net_bps",
        *base_fields,
    ]
    _require_columns(panel, required, "prequential source panel")
    if len(base_fields) != 120 or len(set(base_fields)) != 120:
        raise ValueError("prequential source requires the frozen 120-field contract")
    if reference_days < 7:
        raise ValueError("reference_days must be at least 7")
    frame = panel.copy()
    for column in (
        "__decision_ts__", "r3_label_available_ts", "policy_label_available_ts",
        "h12_label_available_ts",
    ):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    if frame["candidate_id"].duplicated().any():
        raise ValueError("prequential source panel has duplicate candidate IDs")
    sides = sorted(frame["side_name"].astype(str).str.lower().unique())
    if len(sides) > 1:
        ledgers: list[pd.DataFrame] = []
        audits: list[pd.DataFrame] = []
        for side in sides:
            side_ledger, side_audit = build_prequential_stack_ledger(
                frame.loc[frame["side_name"].astype(str).str.lower().eq(side)].copy(),
                base_fields=base_fields, first_held_month=first_held_month,
                last_held_month=last_held_month,
                reference_days=reference_days,
            )
            ledgers.append(side_ledger)
            audits.append(side_audit.assign(side_name=side))
        return (
            pd.concat(ledgers, ignore_index=True).sort_values(
                ["__decision_ts__", "side_name", "candidate_id"], kind="stable",
            ).reset_index(drop=True),
            pd.concat(audits, ignore_index=True),
        )
    if len(sides) != 1:
        raise ValueError("prequential source has no canonical side")
    first = _utc(first_held_month).normalize().replace(day=1)
    final = (
        _utc(last_held_month).normalize().replace(day=1)
        if last_held_month is not None else frame["__decision_ts__"].max().normalize().replace(day=1)
    )
    month_starts = list(pd.date_range(first, final, freq="MS"))
    base_outputs: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for fold_index, start in enumerate(month_starts):
        end = start + pd.offsets.MonthBegin(1)
        held = frame.loc[frame["__decision_ts__"].ge(start) & frame["__decision_ts__"].lt(end)].copy()
        reference = frame.loc[
            frame["__decision_ts__"].ge(start - pd.Timedelta(days=reference_days))
            & frame["__decision_ts__"].lt(start)
        ].copy()
        base_fit = frame.loc[
            frame["r3_label_available_ts"].lt(start) & frame["r3_class"].notna()
        ].sort_values("r3_label_available_ts", kind="stable").tail(BASE_TRAIN_CAP)
        if held.empty or len(reference) < 2 or len(base_fit) < 100 or base_fit["r3_class"].nunique() < 2:
            audit_rows.append({
                "held_month": start.strftime("%Y-%m"), "pass": "base",
                "status": "skipped_insufficient_support", "held_rows": len(held),
                "reference_rows": len(reference), "fit_rows": len(base_fit),
            })
            continue
        medians = _fit_medians(base_fit, base_fields)
        model = LGBMClassifier(**{**BASE_PARAMS, "random_state": SEED + fold_index}).fit(
            _numeric_matrix(base_fit, base_fields, medians),
            base_fit["r3_class"].astype(int).to_numpy(),
        )
        ref_probability = model.predict_proba(_numeric_matrix(reference, base_fields, medians))
        held_probability = model.predict_proba(_numeric_matrix(held, base_fields, medians))
        class_index = {int(label): index for index, label in enumerate(model.classes_)}
        ref_score = (
            ref_probability[:, class_index.get(2, ref_probability.shape[1] - 1)]
            - 0.5 * ref_probability[:, class_index.get(0, 0)]
        )
        held["prequential_p_adverse"] = held_probability[:, class_index.get(0, 0)]
        held["prequential_p_weak"] = held_probability[:, class_index.get(1, min(1, held_probability.shape[1] - 1))]
        held["prequential_p_clear"] = held_probability[:, class_index.get(2, held_probability.shape[1] - 1)]
        held["prequential_base_score"] = held["prequential_p_clear"] - 0.5 * held["prequential_p_adverse"]
        held["prequential_base_rank42"] = ScoreReference.fit(
            ref_score, source=f"{start:%Y-%m}_same_model_prior{reference_days}_base",
        ).cdf(held["prequential_base_score"])
        held["held_month"] = start.strftime("%Y-%m")
        base_outputs.append(held)
        audit_rows.append({
            "held_month": start.strftime("%Y-%m"), "pass": "base", "status": "complete",
            "held_rows": len(held), "reference_rows": len(reference), "fit_rows": len(base_fit),
            "reference_start": start - pd.Timedelta(days=reference_days),
            "reference_end_exclusive": start, "same_model_reference": True,
            "reference_window_days": reference_days,
        })
    if not base_outputs:
        raise ValueError("no strict prequential base fold could be materialised")
    base_ledger = pd.concat(base_outputs, ignore_index=True).sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    ).reset_index(drop=True)
    stack_outputs: list[pd.DataFrame] = []
    for fold_index, start in enumerate(month_starts):
        month = start.strftime("%Y-%m")
        held = base_ledger.loc[base_ledger["held_month"].eq(month)].copy()
        earlier = base_ledger.loc[
            base_ledger["__decision_ts__"].lt(start)
            & base_ledger["policy_label_available_ts"].lt(start)
            & np.isfinite(pd.to_numeric(base_ledger["policy_net_bps"], errors="coerce"))
            & np.isfinite(pd.to_numeric(base_ledger["prequential_base_rank42"], errors="coerce"))
        ].copy()
        if held.empty or len(earlier) < 100:
            audit_rows.append({
                "held_month": month, "pass": "map_residual", "status": "skipped_insufficient_prior_oof",
                "held_rows": len(held), "fit_rows": len(earlier),
            })
            continue
        try:
            policy_map = fit_policy_net_map(earlier["prequential_base_rank42"], earlier["policy_net_bps"])
            earlier["__anchor__"] = policy_map.predict(earlier["prequential_base_rank42"])
            earlier["__residual__"] = earlier["policy_net_bps"] - earlier["__anchor__"]
            grade = residual_grades(earlier["__residual__"])
            medians = _fit_medians(earlier, base_fields)
            held_matrix = _numeric_matrix(held, base_fields, medians)
            ranks: list[np.ndarray] = []
            for cap in RESIDUAL_CAPS:
                for mode in RESIDUAL_WEIGHT_MODES:
                    model, reference = _fit_ranker(
                        earlier, base_fields, grade, cap=cap, mode=mode, medians=medians,
                    )
                    raw = model.predict(held_matrix[:, :cap])
                    ranks.append(reference.cdf(raw))
        except ValueError as error:
            audit_rows.append({
                "held_month": month, "pass": "map_residual", "status": "skipped_model_support",
                "held_rows": len(held), "fit_rows": len(earlier), "reason": str(error),
            })
            continue
        held["prequential_base_anchor_bps"] = policy_map.predict(held["prequential_base_rank42"])
        held["prequential_consensus_rank"] = np.nanmedian(np.column_stack(ranks), axis=1)
        held["prequential_residual_rank"] = held["prequential_consensus_rank"]
        held["prequential_upstream"] = (
            0.75 * held["prequential_base_rank42"] + 0.25 * held["prequential_consensus_rank"]
        )
        held["stack_is_prequential"] = True
        stack_outputs.append(held)
        audit_rows.append({
            "held_month": month, "pass": "map_residual", "status": "complete",
            "held_rows": len(held), "fit_rows": len(earlier),
            "map_fit_rows": policy_map.source_rows, "residual_heads": len(ranks),
            "target": "canonical policy net residual", "held_outcomes_consumed": False,
        })
    if not stack_outputs:
        raise ValueError("no full prequential stack fold could be materialised")
    ledger = pd.concat(stack_outputs, ignore_index=True).sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    ).reset_index(drop=True)
    return ledger, pd.DataFrame(audit_rows)


def train_monthly_bundle(
    *,
    cutoff: Any,
    training_ledger: pd.DataFrame,
    frozen_geometry: FrozenGeometryK9,
    base_fields: Sequence[str],
    context_fields: Sequence[str],
    train_cap: int = BASE_TRAIN_CAP,
    source_hashes: Mapping[str, str] | None = None,
) -> CanonicalMonthlyBundle:
    cutoff_ts = _utc(cutoff)
    if train_cap < 10_000:
        raise ValueError("monthly training cap must be at least 10,000")
    if len(base_fields) != 120 or len(set(base_fields)) != 120:
        raise ValueError("base field contract must contain 120 unique fields")
    if len(context_fields) != 73 or len(set(context_fields)) != 73:
        raise ValueError("Severe context contract must contain 73 unique fields")
    if not frozen_geometry.bundle_sha256:
        raise ValueError(
            "monthly training requires a persisted immutable geometry/K9 bundle; "
            "in-memory refits are prohibited"
        )
    ledger = _validate_prequential_ledger(training_ledger, cutoff_ts)
    _require_columns(ledger, ["__symbol__", "r3_class", "r3_label_available_ts", *base_fields, *context_fields], "monthly training")
    ledger["r3_label_available_ts"] = pd.to_datetime(ledger["r3_label_available_ts"], utc=True)
    base_fit = ledger.loc[
        ledger["r3_label_available_ts"].lt(cutoff_ts)
        & ledger["r3_class"].notna()
    ].sort_values("r3_label_available_ts", kind="stable").tail(train_cap)
    if len(base_fit) < 100 or base_fit["r3_class"].nunique() < 2:
        raise ValueError("strict-R3 base has insufficient resolved class support")
    base_fit_rows = int(len(base_fit))
    base_medians = _fit_medians(base_fit, base_fields)
    base_model = LGBMClassifier(**BASE_PARAMS).fit(
        _numeric_matrix(base_fit, base_fields, base_medians),
        base_fit["r3_class"].astype(int).to_numpy(),
    )
    del base_fit
    gc.collect()
    map_population = ledger.loc[
        ledger["policy_label_available_ts"].lt(cutoff_ts)
        & np.isfinite(pd.to_numeric(ledger["policy_net_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(ledger["prequential_base_rank42"], errors="coerce"))
    ]
    # The rank-domain map and all ten residual heads share a bounded,
    # deterministic equal-month prequential population.  Keeping the cap here
    # prevents ten full-history dense matrices from being created in a single
    # monthly fit, while retaining far more than the support needed for the
    # 20-bin map and 4-hour LambdaRank queries.
    map_fit = _equal_month_sample(
        map_population, train_cap, seed=SEED + 211,
    ).copy()
    map_population_rows = int(len(map_population))
    map_fit_rows = int(len(map_fit))
    policy_map = fit_policy_net_map(map_fit["prequential_base_rank42"], map_fit["policy_net_bps"])
    map_fit["refit_anchor_bps"] = policy_map.predict(map_fit["prequential_base_rank42"])
    map_fit["policy_residual_bps"] = map_fit["policy_net_bps"] - map_fit["refit_anchor_bps"]
    grade = residual_grades(map_fit["policy_residual_bps"])
    residual_medians = _fit_medians(map_fit, base_fields)
    heads: list[ResidualHead] = []
    for cap in RESIDUAL_CAPS:
        for mode in RESIDUAL_WEIGHT_MODES:
            model, reference = _fit_ranker(
                map_fit, base_fields, grade, cap=cap, mode=mode,
                medians=residual_medians,
            )
            heads.append(ResidualHead(f"cap{cap}_{mode}", cap, mode, model, reference))
            gc.collect()
    del map_fit, map_population
    gc.collect()
    geometry = frozen_geometry
    sides = sorted(ledger["side_name"].astype(str).str.lower().unique())
    if len(sides) != 1:
        raise ValueError("monthly bundle training must receive exactly one side")
    side_name = sides[0]
    severe_population = ledger.loc[
        ledger["h12_label_available_ts"].lt(cutoff_ts)
        & ledger["h12_label_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(ledger["h12_tp6_sl4_net_bps"], errors="coerce"))
        & ~ledger["__decision_ts__"].between(GEOMETRY_START, GEOMETRY_END, inclusive="left")
    ]
    if severe_population.empty:
        raise ValueError("Severe training has no older support outside the geometry-definition window")
    # The Severe learner has the same 240k supervised cap as the canonical
    # base.  Apply it before geometry transformation: materialising the full
    # multi-year state and then discarding most rows causes avoidable OOMs,
    # while the deterministic equal-month sample preserves chronological and
    # regime coverage.
    severe_train = _equal_month_sample(
        severe_population, train_cap, seed=SEED + 301,
    ).copy()
    severe_population_rows = int(len(severe_population))
    severe_fit_rows = int(len(severe_train))
    geometry_values = geometry.transform(severe_train).loc[:, list(geometry.severe_structural_fields)]
    core = pd.DataFrame({
        "strict_r3_base_bps": severe_train["prequential_base_anchor_bps"].to_numpy(float),
        "base_rank": severe_train["prequential_base_rank42"].to_numpy(float),
        "consensus_rank": severe_train["prequential_consensus_rank"].to_numpy(float),
        "residual_rank": severe_train["prequential_residual_rank"].to_numpy(float),
        "base_plus_consensus25": severe_train["prequential_upstream"].to_numpy(float),
    }, index=severe_train.index)
    severe_input = pd.concat([core, severe_train.loc[:, list(context_fields)], geometry_values], axis=1)
    severe_fields = tuple(severe_input.columns)
    if len(severe_fields) != 123 or len(set(severe_fields)) != 123:
        raise AssertionError(f"Severe contract must be 123 unique fields, got {len(severe_fields)}")
    severe_medians = _fit_medians(severe_input, severe_fields)
    severe_target = severe_train["h12_tp6_sl4_net_bps"].to_numpy(float) <= -200.0
    severe_model = LGBMClassifier(**SEVERE_PARAMS).fit(
        _numeric_matrix(severe_input, severe_fields, severe_medians),
        severe_target.astype(np.int8),
    )
    del severe_input, geometry_values, severe_train, severe_population
    gc.collect()
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "cutoff": cutoff_ts.isoformat(),
        "side_name": side_name,
        "base_fit_rows": base_fit_rows,
        "map_fit_rows": map_fit_rows,
        "map_population_rows": map_population_rows,
        "residual_fit_rows": map_fit_rows,
        "severe_population_rows": severe_population_rows,
        "severe_fit_rows": severe_fit_rows,
        "geometry_definition_rows_excluded_from_severe": True,
        "geometry_bundle_sha256": geometry.bundle_sha256,
        "base_fields_sha256": _json_hash(list(base_fields)),
        "context_fields_sha256": _json_hash(list(context_fields)),
        "severe_fields_sha256": _json_hash(list(severe_fields)),
        "base_params": BASE_PARAMS, "rank_params": RANK_PARAMS,
        "geometry_params": GEOMETRY_PARAMS, "severe_params": SEVERE_PARAMS,
        "policy": asdict(FrozenPolicyContract()),
        "query": "4-hour UTC x side",
        "residual_target": "canonical policy net bps - prequential rank-domain base anchor",
        "severe_target": "H12 TP6/SL4 net bps <= -200",
        "demotion_alpha": 0.5,
        "seed": SEED,
        "training_cap": int(train_cap),
        "universe_symbols": sorted(ledger["__symbol__"].dropna().astype(str).unique().tolist()),
        "universe_sha256": _json_hash(sorted(ledger["__symbol__"].dropna().astype(str).unique().tolist())),
        "source_hashes": dict(source_hashes or {}),
    }
    return CanonicalMonthlyBundle(
        cutoff=cutoff_ts, side_name=side_name,
        base_fields=tuple(base_fields), context_fields=tuple(context_fields),
        severe_fields=severe_fields, base_medians=base_medians,
        base_model=base_model, policy_net_map=policy_map,
        residual_heads=tuple(heads), geometry=geometry,
        severe_medians=severe_medians, severe_model=severe_model,
        manifest=manifest,
    )


def score_same_model_reference(
    bundle: CanonicalMonthlyBundle,
    *,
    reference: pd.DataFrame,
    held: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score prior-42 and held rows with one fitted monthly bundle.

    Prior rows are a calibration reference only.  Their within-reference base
    percentile is permitted; held rows never affect base, specialist or final
    percentiles.
    """
    for name, frame in (("reference", reference), ("held", held)):
        _require_columns(
            frame,
            ["candidate_id", "__decision_ts__", "side_name", *bundle.base_fields, *bundle.context_fields],
            f"{name} score frame",
        )
        assert_scoring_frame_is_target_free(frame)
        if frame["candidate_id"].duplicated().any():
            raise ValueError(f"{name} score frame has duplicate identities")
        if not frame["side_name"].astype(str).str.lower().eq(bundle.side_name).all():
            raise ValueError(f"{name} score frame does not match side-local bundle {bundle.side_name}")
    reference = reference.copy()
    held = held.copy()
    reference["__decision_ts__"] = pd.to_datetime(reference["__decision_ts__"], utc=True)
    held["__decision_ts__"] = pd.to_datetime(held["__decision_ts__"], utc=True)
    start = bundle.cutoff - pd.Timedelta(days=42)
    if not (reference["__decision_ts__"].ge(start) & reference["__decision_ts__"].lt(bundle.cutoff)).all():
        raise ValueError("reference rows must be inside the preceding 42-day half-open window")
    if not held["__decision_ts__"].ge(bundle.cutoff).all():
        raise ValueError("held rows precede the monthly cutoff")
    combined = pd.concat([
        reference.assign(__score_role__="reference"),
        held.assign(__score_role__="held"),
    ], ignore_index=True)
    matrix = _numeric_matrix(combined, bundle.base_fields, bundle.base_medians)
    probability = bundle.base_model.predict_proba(matrix)
    class_index = {int(label): index for index, label in enumerate(bundle.base_model.classes_)}
    p_adverse = probability[:, class_index.get(0, 0)]
    p_weak = probability[:, class_index.get(1, min(1, probability.shape[1] - 1))]
    p_clear = probability[:, class_index.get(2, probability.shape[1] - 1)]
    combined["p_adverse"] = p_adverse
    combined["p_weak"] = p_weak
    combined["p_clear"] = p_clear
    combined["base_score"] = p_clear - 0.5 * p_adverse
    reference_mask = combined["__score_role__"].eq("reference").to_numpy()
    base_reference = ScoreReference.fit(
        combined.loc[reference_mask, "base_score"], source="same_model_prior_42d_base_score",
    )
    combined["base_rank42"] = base_reference.cdf(combined["base_score"])
    combined["base_anchor_bps"] = bundle.policy_net_map.predict(combined["base_rank42"])
    head_ranks: list[np.ndarray] = []
    for head in bundle.residual_heads:
        raw = head.model.predict(matrix[:, :head.cap])
        combined[f"residual_head__{head.name}__raw"] = raw
        rank = head.score_reference.cdf(raw)
        combined[f"residual_head__{head.name}__rank"] = rank
        head_ranks.append(rank)
    combined["consensus_rank"] = np.nanmedian(np.column_stack(head_ranks), axis=1)
    combined["residual_rank"] = combined["consensus_rank"]
    combined["upstream"] = 0.75 * combined["base_rank42"] + 0.25 * combined["consensus_rank"]
    geometry_state = bundle.geometry.transform(combined)
    geometry_values = geometry_state.loc[:, list(bundle.geometry.severe_structural_fields)]
    # Reliability/MDA receives only stable, soft-K9 weighted aggregates.  Raw
    # membership slots remain internal to the frozen geometry and Severe
    # contract, never an exported downstream model input.
    k9_weighted_fields = [
        field for field in geometry_state.columns if field.startswith("k9_cluster_")
    ]
    combined = pd.concat(
        [combined, geometry_state.loc[:, k9_weighted_fields].reset_index(drop=True)],
        axis=1,
    )
    core = pd.DataFrame({
        "strict_r3_base_bps": combined["base_anchor_bps"].to_numpy(float),
        "base_rank": combined["base_rank42"].to_numpy(float),
        "consensus_rank": combined["consensus_rank"].to_numpy(float),
        "residual_rank": combined["residual_rank"].to_numpy(float),
        "base_plus_consensus25": combined["upstream"].to_numpy(float),
    }, index=combined.index)
    severe_input = pd.concat([
        core,
        combined.loc[:, list(bundle.context_fields)].reset_index(drop=True),
        geometry_values.reset_index(drop=True),
    ], axis=1)
    if tuple(severe_input.columns) != bundle.severe_fields:
        raise AssertionError("Severe received a different field order")
    severe_probability = bundle.severe_model.predict_proba(
        _numeric_matrix(severe_input, bundle.severe_fields, bundle.severe_medians)
    )[:, 1]
    combined["severe200_probability"] = severe_probability
    combined["raw_severe"] = combined["upstream"] * (1.0 - 0.5 * severe_probability)
    final_reference = ScoreReference.fit(
        combined.loc[reference_mask, "raw_severe"], source="same_model_prior_42d_raw_severe",
    )
    combined["final_score"] = final_reference.cdf(combined["raw_severe"])
    bundle_hash = bundle.manifest.get("bundle_sha256", "unpersisted_bundle")
    combined["bundle_sha256"] = bundle_hash
    combined["geometry_bundle_sha256"] = bundle.geometry.bundle_sha256
    audit = pd.DataFrame([{
        "cutoff": bundle.cutoff,
        "reference_start": start,
        "reference_end_exclusive": bundle.cutoff,
        "reference_rows": int(reference_mask.sum()),
        "held_rows": int((~reference_mask).sum()),
        "bundle_sha256": bundle_hash,
        "held_reference_identical_bundle": True,
        "held_percentile_operations": 0,
        "base_rank_reference": base_reference.source,
        "final_rank_reference": final_reference.source,
    }])
    columns = [
        "candidate_id", "__decision_ts__", "side_name", "p_adverse", "p_weak",
        "p_clear", "base_score", "base_rank42", "base_anchor_bps",
        "consensus_rank", "residual_rank", "upstream", "severe200_probability",
        "raw_severe", "final_score", "bundle_sha256",
        "geometry_bundle_sha256", "__score_role__",
        *k9_weighted_fields,
        *[column for column in combined.columns if column.startswith("residual_head__")],
    ]
    return combined.loc[:, columns].copy(), audit


def apply_schema_v2_admission(
    scored_label_ledger: pd.DataFrame,
    *,
    score_column: str = "final_score",
    net_column: str = "policy_net_bps",
    label_available_column: str = "policy_label_available_ts",
    spec: Causal21dAdmissionSpec | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if spec is None:
        spec = Causal21dAdmissionSpec(
            mode="hierarchical_tail_side_shrinkage_v2",
        )
    if spec.net_floor_bps != 50.0:
        raise ValueError("schema-v2 live admission is frozen at +50 bps")
    admitted, audit = apply_causal_21d_side_admission(
        scored_label_ledger,
        score_column=score_column,
        net_column=net_column,
        decision_column="__decision_ts__",
        label_available_column=label_available_column,
        identity_column="candidate_id",
        spec=spec,
    )
    return admitted, audit


def persist_monthly_bundle(bundle: CanonicalMonthlyBundle, directory: Path) -> dict[str, Any]:
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"immutable bundle directory already exists: {directory}")
    directory.mkdir(parents=True)
    bundle_path = directory / "monthly_bundle.joblib"
    joblib.dump(bundle, bundle_path, compress=3)
    bundle_sha256 = _file_hash(bundle_path)
    manifest = {
        **bundle.manifest,
        "schema": BUNDLE_SCHEMA,
        "bundle_file": bundle_path.name,
        "bundle_sha256": bundle_sha256,
        "geometry": bundle.geometry.fit_audit,
        "geometry_bundle_sha256": bundle.geometry.bundle_sha256,
        "geometry_leaf_categories_sha256": _json_hash(bundle.geometry.leaf_categories),
        "geometry_cluster_order": bundle.geometry.cluster_order.tolist(),
        "residual_heads": [
            {"name": head.name, "cap": head.cap, "weight_mode": head.weight_mode,
             "rank_reference_rows": len(head.score_reference.sorted_values)}
            for head in bundle.residual_heads
        ],
    }
    (directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    # Rewrite only the in-memory provenance after hashing the immutable fitted
    # object.  Predictions persist this externally verified hash.
    bundle.manifest["bundle_sha256"] = bundle_sha256
    return manifest


def persist_geometry_bundle(
    geometry: FrozenGeometryK9,
    directory: Path,
    *,
    schema: str = GEOMETRY_SCHEMA,
) -> dict[str, Any]:
    """Persist one immutable geometry definition.

    ``GEOMETRY_SCHEMA`` is reserved for the canonical October--December 2024
    representation.  Research callers that intentionally compare isolated
    later geometry episodes must provide a different schema, preventing an
    episodic fit from being accepted by the canonical loader.
    """

    if schema == GEOMETRY_SCHEMA and (
        geometry.fit_audit.get("definition_start") != GEOMETRY_START.isoformat()
        or geometry.fit_audit.get("definition_end_exclusive") != GEOMETRY_END.isoformat()
    ):
        raise ValueError("canonical geometry schema is reserved for October-December 2024")
    directory = Path(directory)
    if geometry.bundle_sha256:
        raise ValueError("a persisted geometry bundle cannot be refit/re-persisted as a new monthly definition")
    if directory.exists():
        raise FileExistsError(f"immutable geometry directory already exists: {directory}")
    directory.mkdir(parents=True)
    payload_path = directory / "frozen_geometry_k9.joblib"
    joblib.dump(geometry, payload_path, compress=3)
    payload_hash = _file_hash(payload_path)
    manifest = {
        **geometry.fit_audit,
        "schema": schema,
        "bundle_file": payload_path.name,
        "bundle_sha256": payload_hash,
        "leaf_categories_sha256": _json_hash(geometry.leaf_categories),
        "leaf_support_sha256": _json_hash(
            [hashlib.sha256(values.tobytes()).hexdigest() for values in geometry.leaf_support_counts]
        ),
        "cluster_centres_sha256": hashlib.sha256(
            np.asarray(geometry.kmeans.cluster_centers_, dtype=np.float32).tobytes()
        ).hexdigest(),
        "cluster_order": geometry.cluster_order.tolist(),
        "input_order": list(geometry.encoder_fields),
        "imputation_medians": geometry.medians.tolist(),
    }
    (directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    geometry.bundle_sha256 = payload_hash
    return manifest


def load_geometry_bundle(
    directory: Path,
    *,
    expected_schema: str = GEOMETRY_SCHEMA,
    definition_start: Any | None = GEOMETRY_START,
    definition_end_exclusive: Any | None = GEOMETRY_END,
) -> FrozenGeometryK9:
    """Load an immutable geometry bundle with an explicit identity contract.

    The defaults intentionally remain the canonical Oct--Dec 2024 contract.
    An ablation must opt into a distinct schema and its exact definition dates;
    it can never silently enter a canonical scorer.
    """

    directory = Path(directory)
    manifest = json.loads((directory / "run_manifest.json").read_text())
    if manifest.get("schema") != expected_schema:
        raise ValueError(f"not a {expected_schema} geometry bundle")
    payload_path = directory / manifest["bundle_file"]
    if _file_hash(payload_path) != manifest["bundle_sha256"]:
        raise ValueError("geometry/K9 bundle hash mismatch")
    geometry = joblib.load(payload_path)
    if not isinstance(geometry, FrozenGeometryK9):
        raise TypeError("geometry bundle payload has the wrong type")
    if (definition_start is None) != (definition_end_exclusive is None):
        raise ValueError("geometry definition validation requires both bounds or neither")
    if definition_start is not None:
        start_ts, end_ts, _ = _geometry_definition_months(
            definition_start, definition_end_exclusive,
        )
        if geometry.fit_audit.get("definition_start") != start_ts.isoformat():
            raise ValueError("geometry definition start differs from the declared contract")
        if geometry.fit_audit.get("definition_end_exclusive") != end_ts.isoformat():
            raise ValueError("geometry definition end differs from the declared contract")
    geometry.bundle_sha256 = manifest["bundle_sha256"]
    return geometry


def load_monthly_bundle(directory: Path) -> CanonicalMonthlyBundle:
    directory = Path(directory)
    manifest = json.loads((directory / "run_manifest.json").read_text())
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("not a schema-v2 canonical monthly bundle")
    path = directory / manifest["bundle_file"]
    if _file_hash(path) != manifest["bundle_sha256"]:
        raise ValueError("monthly bundle hash mismatch")
    bundle = joblib.load(path)
    if not isinstance(bundle, CanonicalMonthlyBundle) or bundle.schema != BUNDLE_SCHEMA:
        raise TypeError("monthly bundle payload has the wrong type/schema")
    declared_geometry_hash = manifest.get("geometry_bundle_sha256", "")
    embedded_geometry_hash = bundle.geometry.bundle_sha256
    if not declared_geometry_hash or embedded_geometry_hash != declared_geometry_hash:
        raise ValueError("monthly bundle geometry/K9 hash does not match its manifest")
    if bundle.geometry.fit_audit.get("definition_start") != GEOMETRY_START.isoformat():
        raise ValueError("monthly bundle embeds a non-canonical geometry definition start")
    if bundle.geometry.fit_audit.get("definition_end_exclusive") != GEOMETRY_END.isoformat():
        raise ValueError("monthly bundle embeds a non-canonical geometry definition end")
    bundle.manifest["bundle_sha256"] = manifest["bundle_sha256"]
    return bundle
