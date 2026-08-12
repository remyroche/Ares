"""Executable long-only strict-R3 upstream/conversion stack (schema v5).

This module owns the frozen-geometry score producer used by the current
executable-research contract.  It keeps
the older schema-v2 producer importable for historical replay, but it does not
reuse its ordinary-consensus or active Severe-200 score.  A schema-v5 bundle
contains everything required to score one four-week block and its preceding
28-day reference with identical fitted state.

The canonical score is:

    D2 strict-R3 base
      + frozen conditional-usefulness ten-head policy-residual consensus
      -> 75/25 rank blend
      -> one frozen October--December 2024 C3 geometry/K9 view (temperature x0.25)
      -> top-30%-trained policy-residual correctness rank
      -> same-bundle prior-28-day CDF

Raw K9 memberships never enter either the consensus heads or the correctness
head.  Severe-200 is fitted and emitted only as an exact-H12 diagnostic; it
cannot change ``final_score``.  The ordinary ten-head score is retained as a
shadow/rollback output.  Final admission is deliberately outside this module:
the canonical downstream path is the causal 28-day Cell-day trim-15 map,
followed by the independently fitted nine-month R5 posterior trust model and
the +50-bps posterior expected-net gate.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans

from .stage_i_causal_admission import (
    Causal21dAdmissionSpec,
    apply_causal_21d_side_admission,
)
from .strict_r3_canonical_v2 import (
    BASE_PARAMS,
    BASE_TRAIN_CAP,
    RESIDUAL_BANDS_BPS,
    SEED,
    ScoreReference,
    FrozenGeometryK9,
    _equal_month_sample,
    _file_hash,
    _fit_medians,
    _json_hash,
    _numeric_matrix,
    _require_columns,
    assert_scoring_frame_is_target_free,
    fit_policy_net_map,
)
from .strict_r3_self_distillation import (
    DistillationWeightSpec,
    build_distillation_weights,
)
from .strict_r3_ev_bridge import (
    StrictR3EVBridgeBundle,
    apply_strict_r3_ev_bridge,
)


SCHEMA = "strict_r3_canonical_current_v5_frozen_geometry"
BUNDLE_SCHEMA = "strict_r3_canonical_current_four_week_bundle_v5_frozen_geometry"
CONSENSUS_SCHEMA = "strict_r3_conditional_consensus_v1"
CONSENSUS_CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "strict_r3_conditional_consensus_v1.json"
)
SIDE = "long"
FOUR_WEEK_DAYS = 28
REFERENCE_DAYS = 28
REQUIRED_REFERENCE_HOURS = REFERENCE_DAYS * 24
CALIBRATION_RESERVE_DAYS = 28
META_TRAIN_MONTHS = 6
MODEL_CAP = 240_000
GEOMETRY_CAP = 100_000
K9_CLUSTERS = 9
CORRECTNESS_HURDLE_BPS = 100.0
CORRECTNESS_TRAIN_FRACTION = 0.30
SEVERE_HURDLE_BPS = -200.0
BASE_BLEND_WEIGHT = 0.75
CONSENSUS_BLEND_WEIGHT = 0.25
CORRECTNESS_FLOOR = 0.25
CORRECTNESS_SPAN = 0.75
K9_TEMPERATURE_SCALE = 0.25
STRUCTURAL_PROJECTION_DIM = 12
STRUCTURAL_SHRINKAGE_SUPPORT = 24.0
GEOMETRY_DEFINITION_START = pd.Timestamp("2024-10-01T00:00:00Z")
GEOMETRY_DEFINITION_END_EXCLUSIVE = pd.Timestamp("2025-01-01T00:00:00Z")


@dataclass(frozen=True)
class OptimizedPolicyContract:
    source: str = "simple_policy_optimiser_pre2025_winner"
    entry_delay_hours: int = 1
    stop_loss_atr: float = 4.1520006
    trailing_activation_atr: float = 2.3262249
    trailing_giveback_atr: float = 0.1023720
    timeout_hours: int = 12
    cost_bps_once: float = 100.0
    preferred_source_bar_minutes: int = 15
    hourly_proxy_allowed_for_missing_paths: bool = True


D2_SPEC = DistillationWeightSpec(
    "D2_top20_clear_boost1.5",
    positive_top_fraction=0.20,
    positive_boost=1.5,
    minimum_weight=0.25,
    maximum_weight=4.0,
)


@dataclass(frozen=True)
class ConsensusHeadSpec:
    name: str
    cap: int
    weight_mode: str
    query: str
    fields: tuple[str, ...]
    target_edges_bps: tuple[float, ...]
    params: dict[str, Any]


@dataclass
class FittedConsensusHead:
    spec: ConsensusHeadSpec
    medians: np.ndarray
    model: LGBMRanker
    score_reference: ScoreReference

    def predict_rank(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        raw = self.model.predict(_numeric_matrix(frame, self.spec.fields, self.medians))
        return raw.astype(np.float32), self.score_reference.cdf(raw).astype(np.float32)


@dataclass
class RollingRawK9:
    bundle_id: str
    fit_start: pd.Timestamp
    fit_end: pd.Timestamp
    fields: tuple[str, ...]
    medians: np.ndarray
    scale: np.ndarray
    kmeans: MiniBatchKMeans
    permutation: np.ndarray
    temperature: float
    fit_rows: int
    bundle_sha256: str
    temperature_scale: float = K9_TEMPERATURE_SCALE
    previous_bundle_id: str | None = None
    matched_center_cosine: float = float("nan")
    structural_projection: np.ndarray | None = None
    structural_mean: np.ndarray | None = None
    structural_covariance: np.ndarray | None = None
    structural_correlation: np.ndarray | None = None
    cluster_structural_mean: np.ndarray | None = None
    cluster_structural_covariance: np.ndarray | None = None
    cluster_structural_correlation: np.ndarray | None = None
    cluster_structural_support: np.ndarray | None = None

    @property
    def raw_membership_fields(self) -> tuple[str, ...]:
        return tuple(f"k09__cluster_{index:02d}__membership" for index in range(K9_CLUSTERS))

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        matrix = _numeric_matrix(frame, self.fields, self.medians)
        z = np.clip((matrix - self.medians) / self.scale, -8.0, 8.0)
        distance = self.kmeans.transform(z)[:, self.permutation].astype(np.float32)
        effective_temperature = float(self.temperature) * float(self.temperature_scale)
        logits = -distance / max(effective_temperature, 1e-6)
        logits -= logits.max(axis=1, keepdims=True)
        membership = np.exp(logits, dtype=np.float32)
        membership /= np.maximum(membership.sum(axis=1, keepdims=True), 1e-12)
        values: dict[str, np.ndarray] = {}
        for cluster in range(K9_CLUSTERS):
            prefix = f"k09__cluster_{cluster:02d}"
            values[f"{prefix}__membership"] = membership[:, cluster]
            values[f"{prefix}__negative_distance"] = -distance[:, cluster]
            values[f"{prefix}__confidence"] = membership[:, cluster] ** 2
        values["k9_entropy"] = (
            -membership * np.log(np.clip(membership, 1e-12, 1.0))
        ).sum(axis=1)
        values["k9_top2_margin"] = (
            np.partition(membership, -2, axis=1)[:, -1]
            - np.partition(membership, -2, axis=1)[:, -2]
        )
        values["k9_ood_distance"] = distance.min(axis=1)
        output = pd.DataFrame(values, index=frame.index, dtype=np.float32)
        pieces = [output, _dynamic_k9_state(frame, membership)]
        if self.structural_projection is not None:
            pieces.append(_structural_geometry_breaks(frame, z, membership, self))
        return pd.concat(pieces, axis=1).astype(np.float32)


@dataclass(frozen=True)
class FrozenGeometryK9View:
    """A stable, temperature-selected view over one persisted K9 definition.

    The underlying encoder, leaves, K9 centres, order, support counts and
    history are all fitted once on October--December 2024.  The x0.25
    temperature is a frozen scorer setting, not a re-fit or a per-fold
    calibration.  It receives a derived identity so a conversion bundle cannot
    silently mix the parent geometry with another membership interpretation.
    """

    parent: FrozenGeometryK9
    temperature_scale: float = K9_TEMPERATURE_SCALE

    def __post_init__(self) -> None:
        if not self.parent.bundle_sha256:
            raise ValueError("frozen geometry view requires a persisted parent bundle")
        if not np.isclose(float(self.temperature_scale), K9_TEMPERATURE_SCALE):
            raise ValueError("canonical frozen geometry view requires temperature scale 0.25")

    @property
    def parent_bundle_sha256(self) -> str:
        return str(self.parent.bundle_sha256)

    @property
    def bundle_sha256(self) -> str:
        return _json_hash({
            "parent_geometry_bundle_sha256": self.parent_bundle_sha256,
            "temperature_scale": float(self.temperature_scale),
            "view_schema": "strict_r3_frozen_geometry_view_v1",
        })

    @property
    def definition_start(self) -> str:
        return str(self.parent.fit_audit.get("definition_start"))

    @property
    def definition_end_exclusive(self) -> str:
        return str(self.parent.fit_audit.get("definition_end_exclusive"))

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.parent.transform(frame, temperature_scale=self.temperature_scale)


@dataclass
class LeafTrustBundle:
    fields: tuple[str, ...]
    medians: np.ndarray
    model: LGBMClassifier
    support_counts: tuple[np.ndarray, ...]
    train_rows: int
    leaf_values: tuple[np.ndarray, ...] | None = None
    # Frozen, contribution-weighted feature-path signatures for every active
    # leaf.  These let the scorer compare the *rules active now* with the
    # training-rule geometry without exposing unstable raw K9 labels.
    leaf_feature_paths: tuple[np.ndarray, ...] | None = None
    rule_activation_mean: np.ndarray | None = None
    rule_activation_covariance: np.ndarray | None = None
    rule_activation_correlation: np.ndarray | None = None

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        leaves = np.asarray(
            self.model.predict(_numeric_matrix(frame, self.fields, self.medians), pred_leaf=True),
            dtype=np.int32,
        )
        support = np.zeros_like(leaves, dtype=np.float32)
        coverage = np.zeros_like(leaves, dtype=bool)
        contribution = np.zeros_like(leaves, dtype=np.float32)
        for tree, counts in enumerate(self.support_counts):
            token = leaves[:, tree]
            valid = token < len(counts)
            support[valid, tree] = counts[token[valid]]
            coverage[valid, tree] = True
            values = None if self.leaf_values is None else self.leaf_values[tree]
            if values is not None:
                value_valid = token < len(values)
                contribution[value_valid, tree] = np.abs(values[token[value_valid]])
        surprise = -np.log(
            np.clip((support + 1.0) / (self.train_rows + 1.0), 1e-12, 1.0),
        )
        contribution_total = contribution.sum(axis=1)
        contribution_weighted = np.divide(
            (support * contribution).sum(axis=1),
            contribution_total,
            out=support.mean(axis=1),
            where=contribution_total > 1e-12,
        )
        contribution_weighted_log = np.divide(
            (np.log1p(support) * contribution).sum(axis=1),
            contribution_total,
            out=np.log1p(support).mean(axis=1),
            where=contribution_total > 1e-12,
        )
        high_contribution = contribution >= np.quantile(
            contribution, 0.75, axis=1, keepdims=True,
        )
        high_contribution_support = np.where(high_contribution, support, np.nan)
        effective_contributors = np.divide(
            contribution_total**2,
            np.square(contribution).sum(axis=1),
            out=np.zeros_like(contribution_total),
            where=np.square(contribution).sum(axis=1) > 1e-12,
        )
        output = pd.DataFrame(
            {
                "leaf_support_effective": support.mean(axis=1),
                "leaf_support_p05": np.quantile(support, 0.05, axis=1),
                "leaf_support_p50": np.quantile(support, 0.50, axis=1),
                "leaf_support_p95": np.quantile(support, 0.95, axis=1),
                "leaf_support_contribution_weighted": contribution_weighted,
                "leaf_support_contribution_weighted_log": contribution_weighted_log,
                "leaf_support_high_contribution_min": np.nanmin(
                    high_contribution_support, axis=1,
                ),
                "leaf_contributor_effective_n": effective_contributors,
                "leaf_support_adequate_fraction": (support >= 30.0).mean(axis=1),
                "leaf_support_leaf_coverage": coverage.mean(axis=1),
                "leaf_ood_marginal": surprise.mean(axis=1),
                "leaf_ood_joint": surprise.sum(axis=1),
                "leaf_ood_joint_rms": np.sqrt(np.mean(np.square(surprise), axis=1)),
            },
            index=frame.index,
            dtype=np.float32,
        )
        # Legacy bundles predate the frozen active-rule baseline.  Retain
        # their existing output contract rather than emitting silent constants;
        # a new contract that requests these fields must use a refitted bundle.
        if (
            self.leaf_feature_paths is not None
            and self.rule_activation_mean is not None
            and self.rule_activation_covariance is not None
            and self.rule_activation_correlation is not None
        ):
            activation = _contribution_weighted_rule_activation(
                leaves, contribution, self.leaf_feature_paths,
            )
            output = pd.concat(
                [
                    output,
                    _active_rule_timestamp_state(
                        frame,
                        activation=activation,
                        support=support,
                        surprise=surprise,
                        contribution=contribution,
                        baseline_mean=self.rule_activation_mean,
                        baseline_covariance=self.rule_activation_covariance,
                        baseline_correlation=self.rule_activation_correlation,
                    ),
                ],
                axis=1,
            )
        return output.astype(np.float32)


def _contribution_weighted_rule_activation(
    leaves: np.ndarray,
    contribution: np.ndarray,
    leaf_feature_paths: Sequence[np.ndarray],
) -> np.ndarray:
    """Return one normalised feature-path signature per active candidate.

    A leaf signature records the base features encountered on its actual tree
    path.  The candidate signature is the absolute-leaf-value weighted blend
    over all active trees, which means it describes the rules contributing to
    the current score rather than a broad, semantic feature family.
    """

    if len(leaf_feature_paths) != leaves.shape[1]:
        raise ValueError("active-rule path contract does not match leaf-tree count")
    field_count = int(leaf_feature_paths[0].shape[1])
    activation = np.zeros((len(leaves), field_count), dtype=np.float64)
    fallback = np.zeros_like(activation)
    for tree, paths in enumerate(leaf_feature_paths):
        token = leaves[:, tree]
        valid = token < len(paths)
        path = np.zeros((len(leaves), field_count), dtype=np.float64)
        path[valid] = paths[token[valid]]
        weight = contribution[:, tree]
        activation += path * weight[:, None]
        fallback += path
    weight_total = contribution.sum(axis=1, keepdims=True)
    return np.divide(
        activation,
        weight_total,
        out=np.divide(
            fallback,
            float(max(leaves.shape[1], 1)),
            out=np.zeros_like(fallback),
        ),
        where=weight_total > 1e-12,
    )


def _active_rule_timestamp_state(
    frame: pd.DataFrame,
    *,
    activation: np.ndarray,
    support: np.ndarray,
    surprise: np.ndarray,
    contribution: np.ndarray,
    baseline_mean: np.ndarray,
    baseline_covariance: np.ndarray,
    baseline_correlation: np.ndarray,
) -> pd.DataFrame:
    """Causal active-rule drift/support state at every decision timestamp.

    Unlike the prior broad geometry break, the baseline is the frozen
    contribution-weighted feature-path distribution of the leaf rules that
    make up the classifier.  Timestamp values are cross-sectional over only
    candidates available at that timestamp; no outcome or future row is read.
    """

    timestamp_field = "__decision_ts__" if "__decision_ts__" in frame else "__ts__"
    timestamps = pd.to_datetime(frame[timestamp_field], utc=True, errors="raise")
    baseline_mean = np.asarray(baseline_mean, dtype=np.float64)
    baseline_covariance = np.asarray(baseline_covariance, dtype=np.float64)
    baseline_correlation = np.asarray(baseline_correlation, dtype=np.float64)
    diagonal = np.maximum(np.diag(baseline_covariance), 1e-8)
    candidate_mahalanobis = np.sqrt(
        np.mean(np.square(activation - baseline_mean) / diagonal[None, :], axis=1),
    )
    activation_mass = np.maximum(activation.sum(axis=1, keepdims=True), 1e-12)
    activation_probability = activation / activation_mass
    candidate_entropy = -np.sum(
        activation_probability * np.log(np.maximum(activation_probability, 1e-12)),
        axis=1,
    ) / np.log(max(activation.shape[1], 2))
    contribution_total = contribution.sum(axis=1)
    weighted_support = np.divide(
        (support * contribution).sum(axis=1),
        contribution_total,
        out=support.mean(axis=1),
        where=contribution_total > 1e-12,
    )
    weighted_ood = np.divide(
        (surprise * contribution).sum(axis=1),
        contribution_total,
        out=surprise.mean(axis=1),
        where=contribution_total > 1e-12,
    )
    names = (
        "active_rule_candidate_mahalanobis_train",
        "active_rule_feature_entropy",
        "active_rule_support_contribution_weighted",
        "active_rule_ood_contribution_weighted",
        "active_rule_timestamp_cov_break_train",
        "active_rule_timestamp_corr_break_train",
        "active_rule_timestamp_mahalanobis_train",
        "active_rule_timestamp_support_weighted",
        "active_rule_timestamp_support_p05",
        "active_rule_timestamp_ood_weighted",
    )
    output = np.zeros((len(frame), len(names)), dtype=np.float32)
    output[:, 0] = candidate_mahalanobis.astype(np.float32)
    output[:, 1] = candidate_entropy.astype(np.float32)
    output[:, 2] = weighted_support.astype(np.float32)
    output[:, 3] = weighted_ood.astype(np.float32)
    positions = pd.Series(np.arange(len(frame)), index=frame.index)
    for _timestamp, index in positions.groupby(timestamps, sort=False):
        row = index.to_numpy(dtype=int)
        current_mean, covariance_raw, _ = _weighted_moments(activation[row], None)
        alpha = len(row) / (len(row) + STRUCTURAL_SHRINKAGE_SUPPORT)
        covariance = alpha * covariance_raw + (1.0 - alpha) * baseline_covariance
        correlation = _correlation_matrix(covariance)
        output[row, 4] = _covariance_distance(covariance, baseline_covariance)
        output[row, 5] = _correlation_distance(correlation, baseline_correlation)
        output[row, 6] = _mahalanobis_diag(
            current_mean, baseline_mean, baseline_covariance,
        )
        output[row, 7] = float(np.mean(weighted_support[row]))
        output[row, 8] = float(np.quantile(weighted_support[row], 0.05))
        output[row, 9] = float(np.mean(weighted_ood[row]))
    return pd.DataFrame(output, columns=names, index=frame.index)


@dataclass
class CorrectnessHead:
    fields: tuple[str, ...]
    medians: np.ndarray
    model: LGBMRanker
    score_reference: ScoreReference
    training_score_floor: float = 0.0
    training_fraction: float = CORRECTNESS_TRAIN_FRACTION


@dataclass
class SevereDiagnostic:
    fields: tuple[str, ...]
    medians: np.ndarray
    model: LGBMClassifier | None
    target: str = "exact H12 TP6/SL4 net bps <= -200"
    affects_final_score: bool = False


@dataclass
class CanonicalCurrentBundle:
    cutoff: pd.Timestamp
    held_end_exclusive: pd.Timestamp
    base_fields: tuple[str, ...]
    base_medians: np.ndarray
    base_model: LGBMClassifier
    policy_net_map: Any
    conditional_heads: tuple[FittedConsensusHead, ...]
    ordinary_shadow_heads: tuple[FittedConsensusHead, ...]
    geometry: FrozenGeometryK9View
    leaf_trust: LeafTrustBundle
    correctness: CorrectnessHead
    severe_diagnostic: SevereDiagnostic
    policy: OptimizedPolicyContract = field(default_factory=OptimizedPolicyContract)
    schema: str = BUNDLE_SCHEMA
    manifest: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cutoff = _utc(self.cutoff)
        self.held_end_exclusive = _utc(self.held_end_exclusive)
        if self.held_end_exclusive != self.cutoff + pd.Timedelta(days=FOUR_WEEK_DAYS):
            raise ValueError("schema-v5 bundles score exactly one four-week block")
        if len(self.base_fields) != 120 or len(set(self.base_fields)) != 120:
            raise ValueError("schema-v5 requires the frozen 120-field long base contract")
        if len(self.conditional_heads) != 10 or len(self.ordinary_shadow_heads) != 10:
            raise ValueError("schema-v5 requires ten canonical and ten shadow consensus heads")
        if any("k09__cluster_" in field for field in self.correctness.fields):
            raise ValueError("raw K9 cluster fields are prohibited in correctness")
        if not np.isclose(self.correctness.training_fraction, CORRECTNESS_TRAIN_FRACTION):
            raise ValueError("schema-v5 correctness must use the top-30% training curriculum")
        if not isinstance(self.geometry, FrozenGeometryK9View):
            raise ValueError("schema-v5 bundles require the frozen geometry/K9 view")
        if self.geometry.definition_start != "2024-10-01T00:00:00+00:00" or self.geometry.definition_end_exclusive != "2025-01-01T00:00:00+00:00":
            raise ValueError("schema-v5 geometry definition is not Oct-Dec 2024")
        if self.severe_diagnostic.affects_final_score:
            raise ValueError("Severe-200 is diagnostic-only in schema v5")


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _month_add(value: pd.Timestamp, months: int) -> pd.Timestamp:
    return (value.tz_convert(None).to_period("M") + months).to_timestamp().tz_localize("UTC")


def _canonical_base_hash(fields: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(fields), sort_keys=True, separators=(",", ":")).encode(),
    ).hexdigest()


def load_conditional_consensus_contract(
    base_fields: Sequence[str], path: Path = CONSENSUS_CONTRACT,
) -> tuple[ConsensusHeadSpec, ...]:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != CONSENSUS_SCHEMA or payload.get("side") != SIDE:
        raise ValueError("not the frozen long conditional-consensus contract")
    if payload.get("base_contract_sha256") != _canonical_base_hash(base_fields):
        raise ValueError("conditional-consensus field indices target another base contract")
    edges = tuple(float(value) for value in payload["target"]["edges_bps"])
    if edges != RESIDUAL_BANDS_BPS:
        raise ValueError("canonical residual target edges changed")
    params = dict(payload["ranker_params"])
    output: list[ConsensusHeadSpec] = []
    for raw in payload["heads"]:
        indices = tuple(int(value) for value in raw["field_indices"])
        if not indices or min(indices) < 0 or max(indices) >= len(base_fields):
            raise ValueError(f"{raw['name']} has invalid base-field indices")
        output.append(
            ConsensusHeadSpec(
                name=str(raw["name"]),
                cap=int(raw["cap"]),
                weight_mode=str(raw["weight_mode"]),
                query=str(raw["query"]),
                fields=tuple(str(base_fields[index]) for index in indices),
                target_edges_bps=edges,
                params=params,
            )
        )
    names = [spec.name for spec in output]
    if len(output) != 10 or len(set(names)) != 10:
        raise ValueError("conditional-consensus contract is not the exact ten-head set")
    if sum(spec.query == "exact_timestamp_side" for spec in output) != 6:
        raise ValueError("canonical consensus must contain six exact-timestamp heads")
    return tuple(output)


def _ordinary_shadow_contract(base_fields: Sequence[str]) -> tuple[ConsensusHeadSpec, ...]:
    params = load_conditional_consensus_contract(base_fields)[0].params
    return tuple(
        ConsensusHeadSpec(
            name=f"shadow_cap{cap}_{mode}",
            cap=cap,
            weight_mode=mode,
            query="cycle_4h_side",
            fields=tuple(base_fields[:cap]),
            target_edges_bps=RESIDUAL_BANDS_BPS,
            params=params,
        )
        for cap in (40, 60, 80, 100, 120)
        for mode in ("ordinary", "equal_month")
    )


def _residual_grade(values: Sequence[float], edges: Sequence[float]) -> np.ndarray:
    residual = np.asarray(values, dtype=float)
    return np.select(
        [residual <= float(edge) for edge in edges],
        [0, 1, 2, 3],
        default=4,
    ).astype(np.int32)


def _query(frame: pd.DataFrame, mode: str) -> pd.Series:
    timestamp = pd.to_datetime(frame["__decision_ts__"], utc=True)
    side = frame["side_name"].astype(str).str.lower()
    if mode == "exact_timestamp_side":
        token = timestamp
    elif mode == "cycle_4h_side":
        token = timestamp.dt.floor("4h")
    else:
        raise ValueError(f"unsupported canonical query: {mode}")
    return token.astype(str) + "|" + side


def _sample_complete_consensus_queries(
    frame: pd.DataFrame,
    grade: np.ndarray,
    spec: ConsensusHeadSpec,
    *,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Apply the validated 240k cap without splitting LambdaRank queries."""

    work = frame.copy()
    work["__target_grade__"] = np.asarray(grade, dtype=np.int32)
    work["__query__"] = _query(work, spec.query).to_numpy()
    work["__month__"] = pd.to_datetime(
        work["__decision_ts__"], utc=True,
    ).dt.strftime("%Y-%m")
    query_sizes = work["__query__"].value_counts()
    work = work.loc[work["__query__"].map(query_sizes).ge(2)].copy()
    if work.empty:
        raise ValueError(f"{spec.name} lacks multi-row query support")

    if len(work) > MODEL_CAP:
        group_meta = (
            work.groupby("__query__", sort=False)
            .agg(
                rows=("candidate_id", "size"),
                month=("__month__", "first"),
                first_ts=("__decision_ts__", "min"),
            )
            .reset_index()
        )
        generator = np.random.default_rng(seed)
        retained: list[str] = []
        if spec.weight_mode == "equal_month":
            months = sorted(group_meta["month"].unique())
            allowance = max(2, MODEL_CAP // max(len(months), 1))
            for month in months:
                candidate = group_meta.loc[group_meta["month"].eq(month)].copy()
                candidate["__random__"] = generator.random(len(candidate))
                candidate = candidate.sort_values(
                    ["__random__", "first_ts", "__query__"], kind="stable",
                )
                used = 0
                for row in candidate.to_dict("records"):
                    rows = int(row["rows"])
                    if used + rows > allowance:
                        continue
                    retained.append(str(row["__query__"]))
                    used += rows
        else:
            group_meta["__random__"] = generator.random(len(group_meta))
            group_meta = group_meta.sort_values(
                ["__random__", "first_ts", "__query__"], kind="stable",
            )
            used = 0
            for row in group_meta.to_dict("records"):
                rows = int(row["rows"])
                if used + rows > MODEL_CAP:
                    continue
                retained.append(str(row["__query__"]))
                used += rows
        work = work.loc[work["__query__"].isin(retained)].copy()

    work = work.sort_values(
        ["__query__", "__decision_ts__", "candidate_id"], kind="stable",
    ).reset_index(drop=True)
    groups = work.groupby("__query__", sort=False).size().to_numpy(dtype=np.int32)
    target = work.pop("__target_grade__").to_numpy(dtype=np.int32)
    return work, target, groups


def _fit_consensus_head(
    frame: pd.DataFrame,
    grade: np.ndarray,
    spec: ConsensusHeadSpec,
    *,
    seed: int,
) -> FittedConsensusHead:
    sampled, target, groups = _sample_complete_consensus_queries(
        frame, grade, spec, seed=seed,
    )
    return _fit_consensus_head_from_sample(
        sampled, target, groups, spec, seed=seed,
    )


def _fit_consensus_head_from_sample(
    sampled: pd.DataFrame,
    target: np.ndarray,
    groups: np.ndarray,
    spec: ConsensusHeadSpec,
    *,
    seed: int,
) -> FittedConsensusHead:
    """Fit one head from an already selected, complete-query sample.

    This is deliberately separate from query selection so a large historical
    ledger can first be sampled using only identity/query fields, then load
    just the selected head's feature columns.  The ranker still receives the
    identical selected rows, target, groups, weights and seed as the regular
    path above.
    """
    if len(sampled) < 20 or np.unique(target).size < 2:
        raise ValueError(f"{spec.name} lacks query/class support")
    medians = _fit_medians(sampled, spec.fields)
    weights = None
    if spec.weight_mode == "equal_month":
        months = sampled["__month__"]
        frequency = months.value_counts()
        weights = months.map(lambda month: 1.0 / float(frequency.loc[month])).to_numpy(float)
        weights *= len(weights) / max(float(weights.sum()), 1e-12)
    elif spec.weight_mode != "ordinary":
        raise ValueError(f"unknown consensus weight mode: {spec.weight_mode}")
    params = dict(spec.params)
    params["random_state"] = int(seed)
    # Fixed single-thread construction prevents equal inputs from changing
    # ranking ties/leaf assignments between repeated causal replays.
    params.setdefault("n_jobs", 1)
    params.setdefault("deterministic", True)
    params.setdefault("force_col_wise", True)
    model = LGBMRanker(**params).fit(
        _numeric_matrix(sampled, spec.fields, medians),
        target,
        group=groups,
        sample_weight=weights,
    )
    training_raw = model.predict(_numeric_matrix(sampled, spec.fields, medians))
    reference = ScoreReference.fit(
        training_raw,
        source=f"{spec.name}_resolved_prequential_training_distribution",
    )
    return FittedConsensusHead(spec, medians, model, reference)


def _fit_consensus_head_compact(
    ledger: pd.DataFrame,
    sampling_frame: pd.DataFrame,
    grade: np.ndarray,
    spec: ConsensusHeadSpec,
    *,
    seed: int,
) -> FittedConsensusHead:
    """Fit a head without copying every frozen field for every head.

    ``sampling_frame`` carries only candidate identity, timestamp, side and
    immutable ledger row position.  `_sample_complete_consensus_queries`
    therefore makes precisely the canonical capped complete-query choice.
    The selected rows are then projected directly from the immutable ledger
    into the head's declared field contract.  No sampling, labels, weights or
    model parameters are changed by this memory-bound path.
    """
    sampled_identity, target, groups = _sample_complete_consensus_queries(
        sampling_frame, grade, spec, seed=seed,
    )
    positions = sampled_identity["__ledger_position__"].to_numpy(dtype=np.int64)
    source_ids = ledger.iloc[positions]["candidate_id"].to_numpy()
    selected_ids = sampled_identity["candidate_id"].to_numpy()
    if not np.array_equal(source_ids, selected_ids):
        raise AssertionError("compact consensus sample no longer matches ledger identities")
    features = ledger.iloc[positions].loc[:, list(spec.fields)].reset_index(drop=True)
    sampled = pd.concat(
        [sampled_identity.reset_index(drop=True), features], axis=1,
    )
    return _fit_consensus_head_from_sample(
        sampled, target, groups, spec, seed=seed,
    )


def _fit_raw_k9(
    frame: pd.DataFrame,
    fields: Sequence[str],
    *,
    fit_start: pd.Timestamp,
    fit_end: pd.Timestamp,
    previous: RollingRawK9 | None,
) -> RollingRawK9:
    fit = frame.loc[
        frame["__decision_ts__"].ge(fit_start)
        & frame["__decision_ts__"].lt(fit_end)
    ].copy()
    if fit.empty:
        raise ValueError("C3 geometry burn-in is empty")
    sample = _equal_month_sample(fit, GEOMETRY_CAP, seed=SEED + 701)
    medians = _fit_medians(sample, fields)
    matrix = _numeric_matrix(sample, fields, medians)
    q25 = np.quantile(matrix, 0.25, axis=0)
    q75 = np.quantile(matrix, 0.75, axis=0)
    scale = np.maximum(q75 - q25, 1e-4).astype(np.float32)
    z = np.clip((matrix - medians) / scale, -8.0, 8.0)
    model = MiniBatchKMeans(
        n_clusters=K9_CLUSTERS,
        batch_size=4096,
        n_init=5,
        random_state=SEED + 703,
    ).fit(z)
    permutation = np.arange(K9_CLUSTERS, dtype=int)
    matched_cosine = float("nan")
    if previous is not None:
        old = previous.kmeans.cluster_centers_[previous.permutation]
        new = model.cluster_centers_
        old_norm = old / np.maximum(np.linalg.norm(old, axis=1, keepdims=True), 1e-12)
        new_norm = new / np.maximum(np.linalg.norm(new, axis=1, keepdims=True), 1e-12)
        cosine = old_norm @ new_norm.T
        row, column = linear_sum_assignment(1.0 - cosine)
        permutation = np.empty(K9_CLUSTERS, dtype=int)
        permutation[row] = column
        matched_cosine = float(cosine[row, column].mean())
    distance = model.transform(z)[:, permutation]
    temperature = max(float(np.median(distance.min(axis=1))), 1e-3)
    effective_temperature = temperature * K9_TEMPERATURE_SCALE
    logits = -distance / max(effective_temperature, 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    membership = np.exp(logits, dtype=np.float32)
    membership /= np.maximum(membership.sum(axis=1, keepdims=True), 1e-12)
    projection = _structural_projection(fields)
    projected = np.asarray(z @ projection, dtype=np.float64)
    structural_mean, structural_covariance, structural_correlation = _weighted_moments(
        projected, None,
    )
    cluster_mean: list[np.ndarray] = []
    cluster_covariance: list[np.ndarray] = []
    cluster_correlation: list[np.ndarray] = []
    cluster_support: list[float] = []
    for cluster in range(K9_CLUSTERS):
        mean, covariance, correlation = _weighted_moments(
            projected, membership[:, cluster],
        )
        cluster_mean.append(mean)
        cluster_covariance.append(covariance)
        cluster_correlation.append(correlation)
        cluster_support.append(float(membership[:, cluster].sum()))
    identity = hashlib.sha256()
    identity.update(np.asarray(model.cluster_centers_[permutation], dtype=np.float32).tobytes())
    identity.update(np.asarray(medians, dtype=np.float32).tobytes())
    identity.update(np.asarray(scale, dtype=np.float32).tobytes())
    identity.update(
        np.asarray([temperature, K9_TEMPERATURE_SCALE], dtype=np.float32).tobytes(),
    )
    identity.update(np.asarray(projection, dtype=np.float32).tobytes())
    identity.update(np.asarray(structural_covariance, dtype=np.float32).tobytes())
    identity.update(np.asarray(cluster_covariance, dtype=np.float32).tobytes())
    return RollingRawK9(
        bundle_id=f"c3_{fit_start:%Y%m%d}_{fit_end:%Y%m%d}",
        fit_start=fit_start,
        fit_end=fit_end,
        fields=tuple(fields),
        medians=medians,
        scale=scale,
        kmeans=model,
        permutation=permutation,
        temperature=temperature,
        fit_rows=len(sample),
        bundle_sha256=identity.hexdigest(),
        temperature_scale=K9_TEMPERATURE_SCALE,
        previous_bundle_id=None if previous is None else previous.bundle_id,
        matched_center_cosine=matched_cosine,
        structural_projection=projection.astype(np.float32),
        structural_mean=structural_mean.astype(np.float32),
        structural_covariance=structural_covariance.astype(np.float32),
        structural_correlation=structural_correlation.astype(np.float32),
        cluster_structural_mean=np.asarray(cluster_mean, dtype=np.float32),
        cluster_structural_covariance=np.asarray(cluster_covariance, dtype=np.float32),
        cluster_structural_correlation=np.asarray(cluster_correlation, dtype=np.float32),
        cluster_structural_support=np.asarray(cluster_support, dtype=np.float32),
    )


def _structural_projection(fields: Sequence[str]) -> np.ndarray:
    """Deterministic signed projection using every geometry input field."""

    dimension = min(STRUCTURAL_PROJECTION_DIM, max(2, len(fields)))
    seed = int.from_bytes(
        hashlib.sha256("\n".join(map(str, fields)).encode()).digest()[:8],
        "little",
    )
    rng = np.random.default_rng(seed)
    projection = rng.choice(
        np.asarray([-1.0, 1.0], dtype=np.float32),
        size=(len(fields), dimension),
    )
    return projection / np.sqrt(float(dimension))


def _correlation_matrix(covariance: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.maximum(np.diag(covariance), 1e-8))
    output = covariance / np.maximum(np.outer(scale, scale), 1e-8)
    output = np.clip((output + output.T) * 0.5, -1.0, 1.0)
    np.fill_diagonal(output, 1.0)
    return output


def _weighted_moments(
    matrix: np.ndarray,
    weight: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=np.float64)
    if weight is None:
        weights = np.ones(len(values), dtype=np.float64)
    else:
        weights = np.maximum(np.asarray(weight, dtype=np.float64), 0.0)
    total = float(weights.sum())
    if total <= 1e-12 or not len(values):
        dimension = values.shape[1]
        mean = np.zeros(dimension, dtype=np.float64)
        covariance = np.eye(dimension, dtype=np.float64)
        return mean, covariance, covariance.copy()
    mean = np.sum(values * weights[:, None], axis=0) / total
    centered = values - mean
    covariance = (centered * weights[:, None]).T @ centered / total
    covariance = (covariance + covariance.T) * 0.5
    covariance[np.diag_indices_from(covariance)] = np.maximum(
        np.diag(covariance), 1e-6,
    )
    return mean, covariance, _correlation_matrix(covariance)


def _covariance_distance(current: np.ndarray, baseline: np.ndarray) -> float:
    scale = 1.0 / np.sqrt(np.maximum(np.diag(baseline), 1e-8))
    weight = np.outer(scale, scale)
    numerator = np.linalg.norm((current - baseline) * weight, "fro")
    denominator = max(float(np.linalg.norm(baseline * weight, "fro")), 1e-8)
    return float(np.clip(numerator / denominator, 0.0, 100.0))


def _correlation_distance(current: np.ndarray, baseline: np.ndarray) -> float:
    upper = np.triu_indices_from(current, k=1)
    if not upper[0].size:
        return 0.0
    return float(np.sqrt(np.mean(np.square(current[upper] - baseline[upper]))))


def _mahalanobis_diag(mean: np.ndarray, baseline_mean: np.ndarray, baseline_cov: np.ndarray) -> float:
    delta = mean - baseline_mean
    return float(np.sqrt(np.mean(np.square(delta) / np.maximum(np.diag(baseline_cov), 1e-8))))


def _structural_geometry_breaks(
    frame: pd.DataFrame,
    z: np.ndarray,
    membership: np.ndarray,
    bundle: RollingRawK9,
) -> pd.DataFrame:
    """Compare each decision-time feature geometry with its frozen fit baseline."""

    projection = np.asarray(bundle.structural_projection, dtype=np.float64)
    projected = np.asarray(z @ projection, dtype=np.float64)
    timestamp_field = "__decision_ts__" if "__decision_ts__" in frame else "__ts__"
    timestamps = pd.to_datetime(frame[timestamp_field], utc=True)
    names = (
        "geometry_cov_break_train",
        "geometry_corr_break_train",
        "geometry_mahalanobis_train",
        "geometry_cluster_cov_break_train",
        "geometry_cluster_corr_break_train",
        "geometry_cluster_mahalanobis_train",
    )
    output = np.zeros((len(frame), len(names)), dtype=np.float32)
    global_mean = np.asarray(bundle.structural_mean, dtype=np.float64)
    global_cov = np.asarray(bundle.structural_covariance, dtype=np.float64)
    global_corr = np.asarray(bundle.structural_correlation, dtype=np.float64)
    cluster_mean = np.asarray(bundle.cluster_structural_mean, dtype=np.float64)
    cluster_cov = np.asarray(bundle.cluster_structural_covariance, dtype=np.float64)
    cluster_corr = np.asarray(bundle.cluster_structural_correlation, dtype=np.float64)
    positions = pd.Series(np.arange(len(frame)), index=frame.index)
    for _timestamp, index in positions.groupby(timestamps, sort=False):
        row = index.to_numpy(dtype=int)
        current_mean, current_cov_raw, _current_corr_raw = _weighted_moments(
            projected[row], None,
        )
        alpha = len(row) / (len(row) + STRUCTURAL_SHRINKAGE_SUPPORT)
        current_cov = alpha * current_cov_raw + (1.0 - alpha) * global_cov
        current_corr = _correlation_matrix(current_cov)
        output[row, 0] = _covariance_distance(current_cov, global_cov)
        output[row, 1] = _correlation_distance(current_corr, global_corr)
        output[row, 2] = _mahalanobis_diag(current_mean, global_mean, global_cov)
        cluster_values = np.zeros((len(row), 3), dtype=np.float64)
        for cluster in range(K9_CLUSTERS):
            weight = membership[row, cluster]
            local_mean, local_cov_raw, _ = _weighted_moments(projected[row], weight)
            effective = float(weight.sum())
            local_alpha = effective / (effective + STRUCTURAL_SHRINKAGE_SUPPORT)
            local_cov = local_alpha * local_cov_raw + (1.0 - local_alpha) * cluster_cov[cluster]
            local_corr = _correlation_matrix(local_cov)
            diagnostics = np.asarray(
                [
                    _covariance_distance(local_cov, cluster_cov[cluster]),
                    _correlation_distance(local_corr, cluster_corr[cluster]),
                    _mahalanobis_diag(local_mean, cluster_mean[cluster], cluster_cov[cluster]),
                ],
                dtype=np.float64,
            )
            cluster_values += membership[row, cluster, None] * diagnostics[None, :]
        output[row, 3:] = cluster_values.astype(np.float32)
    return pd.DataFrame(output, columns=names, index=frame.index)


def _dynamic_k9_state(frame: pd.DataFrame, membership: np.ndarray) -> pd.DataFrame:
    fields = [f"k{index}" for index in range(K9_CLUSTERS)]
    event = pd.DataFrame(membership, columns=fields)
    event["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True).to_numpy()
    state = event.groupby("__decision_ts__", sort=True)[fields].sum().sort_index()
    prior = state.shift(1).fillna(0.0)
    support = prior.rolling("28D", min_periods=1).sum()
    current_probability = state.div(state.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    reference_probability = support.div(support.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    mean = prior.rolling("28D", min_periods=4).mean()
    std = prior.rolling("28D", min_periods=8).std().replace(0.0, np.nan)
    z = (state - mean) / std
    rows = pd.DataFrame(
        {"__decision_ts__": pd.to_datetime(frame["__decision_ts__"], utc=True)},
        index=frame.index,
    )
    support_rows = rows.merge(
        support.reset_index(), on="__decision_ts__", how="left", validate="many_to_one",
    )[fields].fillna(0.0).to_numpy(float)
    total = np.maximum(support_rows.sum(axis=1, keepdims=True), 1.0)
    surprise = -np.log(
        np.clip((support_rows + 1.0) / (total + K9_CLUSTERS), 1e-12, 1.0),
    )
    current_rows = membership
    output = pd.DataFrame(
        {
            "k9_path_support_effective_28d": (current_rows * support_rows).sum(axis=1),
            "k9_path_support_adequate_fraction": (
                current_rows * (support_rows >= 30.0)
            ).sum(axis=1),
            "k9_path_ood_marginal": (current_rows * surprise).sum(axis=1),
            "k9_model_ood_marginal": rows.merge(
                z.abs().mean(axis=1).rename("_v").reset_index(),
                on="__decision_ts__",
                how="left",
                validate="many_to_one",
            )["_v"].fillna(0.0).to_numpy(float),
            "k9_model_drift_psi": rows.merge(
                (
                    (current_probability - reference_probability)
                    * np.log(
                        np.clip(current_probability, 1e-12, None)
                        / np.clip(reference_probability, 1e-12, None)
                    )
                ).sum(axis=1).rename("_v").reset_index(),
                on="__decision_ts__",
                how="left",
                validate="many_to_one",
            )["_v"].fillna(0.0).to_numpy(float),
        },
        index=frame.index,
        dtype=np.float32,
    )
    return output.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _fit_leaf_trust(
    frame: pd.DataFrame, fields: Sequence[str], cutoff: pd.Timestamp,
) -> LeafTrustBundle:
    fit = frame.loc[
        frame["__decision_ts__"].lt(cutoff)
        & frame["r3_label_available_ts"].lt(cutoff)
        & frame["r3_class"].notna()
    ].copy()
    fit = _equal_month_sample(fit, MODEL_CAP, seed=SEED + 711)
    if len(fit) < 1_000 or fit["r3_class"].nunique() < 2:
        raise ValueError("current-base leaf trust lacks R3 support")
    medians = _fit_medians(fit, fields)
    model = LGBMClassifier(
        objective="binary",
        n_estimators=64,
        learning_rate=0.035,
        max_depth=5,
        num_leaves=24,
        min_child_samples=max(300, int(0.01 * len(fit))),
        colsample_bytree=0.85,
        reg_lambda=20.0,
        random_state=SEED + 713,
        n_jobs=1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    ).fit(
        _numeric_matrix(fit, fields, medians),
        fit["r3_class"].eq(2).astype(np.int8).to_numpy(),
    )
    leaves = np.asarray(
        model.predict(_numeric_matrix(fit, fields, medians), pred_leaf=True),
        dtype=np.int32,
    )
    support = tuple(
        np.bincount(leaves[:, tree]).astype(np.float32)
        for tree in range(leaves.shape[1])
    )
    dump = model.booster_.dump_model()
    leaf_values: list[np.ndarray] = []
    leaf_feature_paths: list[np.ndarray] = []
    for tree_index, tree in enumerate(dump["tree_info"]):
        values = np.zeros(len(support[tree_index]), dtype=np.float32)
        paths = np.zeros((len(support[tree_index]), len(fields)), dtype=np.float32)

        def visit(node: dict[str, Any], active: np.ndarray) -> None:
            if "leaf_index" in node:
                leaf_index = int(node["leaf_index"])
                if leaf_index < len(values):
                    values[leaf_index] = float(node.get("leaf_value", 0.0))
                    paths[leaf_index] = active
                return
            feature_index = int(node.get("split_feature", -1))
            next_active = active.copy()
            if 0 <= feature_index < len(fields):
                next_active[feature_index] += 1.0
            visit(node["left_child"], next_active)
            visit(node["right_child"], next_active)

        visit(tree["tree_structure"], np.zeros(len(fields), dtype=np.float32))
        leaf_values.append(values)
        leaf_feature_paths.append(paths)
    contribution = np.column_stack(
        [values[leaves[:, tree]] for tree, values in enumerate(leaf_values)]
    ).astype(np.float64)
    rule_activation = _contribution_weighted_rule_activation(
        leaves, np.abs(contribution), tuple(leaf_feature_paths),
    )
    rule_mean, rule_covariance, rule_correlation = _weighted_moments(
        rule_activation, None,
    )
    # Keep K9 frozen as the persistent archetype contract, but expose a
    # *separate*, model-vintage reliability surface.  The semantic fields
    # below are support/OOD/drift of the currently active feature paths versus
    # this bundle's strictly pre-cutoff training distribution.  They never
    # expose raw leaf IDs, so a downstream model sees the same meaning after
    # the proxy is re-fit at a later conversion boundary.
    return LeafTrustBundle(
        tuple(fields), medians, model, support, len(fit), tuple(leaf_values),
        tuple(leaf_feature_paths), rule_mean, rule_covariance, rule_correlation,
    )


def _aggregate_state_fields(state: pd.DataFrame) -> tuple[str, ...]:
    fields = tuple(
        column for column in state.columns
        if not column.startswith("k09__cluster_")
    )
    if any("k09__cluster_" in field for field in fields):
        raise AssertionError("raw K9 field survived the aggregate-state filter")
    return fields


def _canonical_ldf_geometry_aliases(state: pd.DataFrame) -> pd.DataFrame:
    """Expose stable v3-LDF names from the frozen Geometry/K9 representation.

    The October--December 2024 frozen encoder uses explicit ``rule_*``,
    ``path_*`` and ``model_*`` state names.  The selected 45-field incumbent
    LDF predates that renamed representation and expects six corresponding
    K9 summary names.  Derive those summaries *inside the scorer* while raw
    memberships are available, then let ``_aggregate_state_fields`` remove
    the raw membership slots.  This is a target-free representation alias,
    not a re-fit, calibration, or imputation.
    """

    output = state.copy()
    membership_fields = sorted(
        column
        for column in output.columns
        if column.startswith("k09__cluster_") and column.endswith("__membership")
    )
    if membership_fields:
        membership = output.loc[:, membership_fields].to_numpy(dtype=np.float64)
        membership = np.clip(membership, 0.0, None)
        total = membership.sum(axis=1, keepdims=True)
        membership = np.divide(
            membership,
            total,
            out=np.full_like(membership, 1.0 / len(membership_fields)),
            where=total > 1e-12,
        )
        output["k9_entropy"] = (
            -membership * np.log(np.clip(membership, 1e-12, 1.0))
        ).sum(axis=1).astype(np.float32)
        if membership.shape[1] >= 2:
            ordered = np.partition(membership, -2, axis=1)
            output["k9_top2_margin"] = (ordered[:, -1] - ordered[:, -2]).astype(np.float32)
        distance_fields = sorted(
            column
            for column in output.columns
            if column.startswith("k09__cluster_") and column.endswith("__negative_distance")
        )
        if distance_fields:
            distance = -output.loc[:, distance_fields].to_numpy(dtype=np.float64)
            output["k9_ood_distance"] = np.nanmin(distance, axis=1).astype(np.float32)
    aliases = {
        "path_support_effective_28d": "k9_path_support_effective_28d",
        "model_ood_marginal": "k9_model_ood_marginal",
        "model_drift_prototype_psi": "k9_model_drift_psi",
    }
    for source, target in aliases.items():
        if source in output:
            output[target] = pd.to_numeric(output[source], errors="coerce").astype(np.float32)
    return output


def _fit_correctness(
    train: pd.DataFrame,
    fields: Sequence[str],
) -> CorrectnessHead:
    score = pd.to_numeric(train["upstream"], errors="coerce")
    valid_score = score[np.isfinite(score)]
    if len(valid_score) < 1_000:
        raise ValueError("correctness training score has insufficient finite support")
    training_score_floor = float(
        np.quantile(
            valid_score.to_numpy(float),
            1.0 - CORRECTNESS_TRAIN_FRACTION,
            method="higher",
        ),
    )
    retained = train.loc[score.ge(training_score_floor)].copy()
    ordered = retained.assign(
        __query__=pd.to_datetime(retained["__decision_ts__"], utc=True).dt.floor("4h"),
    ).sort_values(["__query__", "__decision_ts__", "candidate_id"], kind="stable")
    query_counts = ordered["__query__"].value_counts()
    ordered = ordered.loc[ordered["__query__"].map(query_counts).ge(2)].copy()
    if len(ordered) < 1_000:
        raise ValueError("correctness model has insufficient resolved support")
    _, groups = np.unique(ordered["__query__"].to_numpy(), return_counts=True)
    target = (
        pd.to_numeric(ordered["policy_net_bps"], errors="coerce")
        - pd.to_numeric(ordered["base_anchor_bps"], errors="coerce")
        > CORRECTNESS_HURDLE_BPS
    ).astype(np.int8).to_numpy()
    if np.unique(target).size < 2:
        raise ValueError("correctness target is degenerate")
    medians = _fit_medians(ordered, fields)
    model = LGBMRanker(
        objective="lambdarank",
        n_estimators=120,
        learning_rate=0.035,
        max_depth=4,
        num_leaves=15,
        min_child_samples=max(120, int(0.03 * len(ordered))),
        colsample_bytree=0.80,
        subsample=0.82,
        subsample_freq=1,
        reg_alpha=0.05,
        reg_lambda=5.0,
        max_bin=127,
        label_gain=[0, 1],
        lambdarank_truncation_level=10,
        random_state=SEED + 719,
        n_jobs=1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    ).fit(
        _numeric_matrix(ordered, fields, medians),
        target,
        group=groups,
    )
    raw = model.predict(_numeric_matrix(ordered, fields, medians))
    return CorrectnessHead(
        tuple(fields),
        medians,
        model,
        ScoreReference.fit(raw, source="correctness_resolved_training_distribution"),
        training_score_floor=training_score_floor,
        training_fraction=CORRECTNESS_TRAIN_FRACTION,
    )


def _fit_severe_diagnostic(
    train: pd.DataFrame, fields: Sequence[str], *, cutoff: pd.Timestamp,
) -> SevereDiagnostic:
    decision = pd.to_datetime(train["__decision_ts__"], utc=True, errors="raise")
    outside_geometry_definition = ~decision.between(
        GEOMETRY_DEFINITION_START,
        GEOMETRY_DEFINITION_END_EXCLUSIVE,
        inclusive="left",
    )
    valid = train.loc[
        outside_geometry_definition
        & train["h12_label_available_ts"].lt(cutoff)
        & train["h12_label_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(train["h12_tp6_sl4_net_bps"], errors="coerce"))
    ].copy()
    if len(valid) < 1_000:
        return SevereDiagnostic(tuple(fields), np.zeros(len(fields), dtype=np.float32), None)
    target = valid["h12_tp6_sl4_net_bps"].le(SEVERE_HURDLE_BPS).astype(np.int8).to_numpy()
    if np.unique(target).size < 2:
        return SevereDiagnostic(tuple(fields), np.zeros(len(fields), dtype=np.float32), None)
    medians = _fit_medians(valid, fields)
    model = LGBMClassifier(
        objective="binary",
        n_estimators=35,
        learning_rate=0.0444772418995553,
        max_depth=5,
        num_leaves=15,
        min_child_samples=max(103, int(0.01 * len(valid))),
        colsample_bytree=0.7393319822815638,
        subsample=0.7853518403594505,
        subsample_freq=1,
        reg_alpha=0.02534130367151813,
        reg_lambda=16.57892339556902,
        max_bin=127,
        random_state=SEED + 727,
        n_jobs=1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    ).fit(_numeric_matrix(valid, fields, medians), target)
    return SevereDiagnostic(tuple(fields), medians, model)


def train_current_bundle(
    *,
    cutoff: Any,
    training_ledger: pd.DataFrame,
    frozen_geometry: FrozenGeometryK9,
    base_fields: Sequence[str],
    source_hashes: Mapping[str, str] | None = None,
    calibration_reserve_days: int = 0,
) -> CanonicalCurrentBundle:
    """Fit one causal four-week schema-v5 bundle.

    ``training_ledger`` must already contain strict-prequential D2 base and
    conditional-consensus handoffs for every row used by the map/correctness
    learner.  The held block is never accepted by this API.
    """

    cutoff_ts = _utc(cutoff)
    held_end = cutoff_ts + pd.Timedelta(days=FOUR_WEEK_DAYS)
    if not 0 <= int(calibration_reserve_days) <= REFERENCE_DAYS:
        raise ValueError(
            "current bundle calibration reserve must lie between zero and the "
            "same-model reference horizon",
        )
    calibration_reserve_start = cutoff_ts - pd.Timedelta(
        days=int(calibration_reserve_days),
    )
    train_start = _month_add(cutoff_ts, -META_TRAIN_MONTHS)
    base_fields = tuple(str(field) for field in base_fields)
    if len(base_fields) != 120 or len(set(base_fields)) != 120:
        raise ValueError("schema-v5 requires 120 unique base fields")
    required = [
        "candidate_id", "__decision_ts__", "side_name", "r3_class",
        "r3_label_available_ts", "policy_net_bps", "policy_label_available_ts",
        "h12_label_valid", "h12_label_available_ts", "h12_tp6_sl4_net_bps",
        "stack_is_prequential", "prequential_base_score",
        "prequential_base_rank42", "prequential_base_anchor_bps",
        "prequential_conditional_consensus_rank",
        "prequential_conditional_upstream", *base_fields,
    ]
    _require_columns(training_ledger, required, "schema-v5 training ledger")
    ledger = training_ledger.copy()
    for column in (
        "__decision_ts__", "r3_label_available_ts", "policy_label_available_ts",
        "h12_label_available_ts",
    ):
        ledger[column] = pd.to_datetime(ledger[column], utc=True, errors="raise")
    if ledger["candidate_id"].duplicated().any():
        raise ValueError("schema-v5 training ledger has duplicate identities")
    if not ledger["side_name"].astype(str).str.lower().eq(SIDE).all():
        raise ValueError("schema-v5 production contract is long-only")
    if (ledger["__decision_ts__"] >= cutoff_ts).any():
        raise ValueError("schema-v5 training ledger crosses the held cutoff")
    if not ledger["stack_is_prequential"].fillna(False).astype(bool).all():
        raise ValueError("schema-v5 downstream fitting requires strict prequential rows")

    base_fit = ledger.loc[
        ledger["__decision_ts__"].lt(calibration_reserve_start)
        & ledger["r3_label_available_ts"].lt(calibration_reserve_start)
        & ledger["r3_class"].notna()
    ].sort_values("r3_label_available_ts", kind="stable").tail(BASE_TRAIN_CAP).copy()
    if len(base_fit) < 1_000 or base_fit["r3_class"].nunique() < 3:
        raise ValueError("D2 strict-R3 base has insufficient three-class support")
    base_weight, weight_audit = build_distillation_weights(
        base_fit,
        teacher_rank_column="prequential_base_rank42",
        layer="base",
        spec=D2_SPEC,
    )
    base_medians = _fit_medians(base_fit, base_fields)
    base_model = LGBMClassifier(**BASE_PARAMS).fit(
        _numeric_matrix(base_fit, base_fields, base_medians),
        base_fit["r3_class"].astype(int).to_numpy(),
        sample_weight=base_weight,
    )

    map_fit = ledger.loc[
        ledger["__decision_ts__"].lt(calibration_reserve_start)
        & ledger["policy_label_available_ts"].lt(calibration_reserve_start)
        & np.isfinite(pd.to_numeric(ledger["policy_net_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(ledger["prequential_base_rank42"], errors="coerce"))
    ].copy()
    policy_map = fit_policy_net_map(
        map_fit["prequential_base_rank42"], map_fit["policy_net_bps"],
    )
    map_fit["base_anchor_bps"] = policy_map.predict(map_fit["prequential_base_rank42"])
    map_fit["policy_residual_bps"] = (
        map_fit["policy_net_bps"].to_numpy(float)
        - map_fit["base_anchor_bps"].to_numpy(float)
    )
    grade = _residual_grade(map_fit["policy_residual_bps"], RESIDUAL_BANDS_BPS)
    conditional_specs = load_conditional_consensus_contract(base_fields)
    shadow_specs = _ordinary_shadow_contract(base_fields)
    conditional_heads = tuple(
        _fit_consensus_head(map_fit, grade, spec, seed=SEED + 1000 + index)
        for index, spec in enumerate(conditional_specs)
    )
    shadow_heads = tuple(
        _fit_consensus_head(map_fit, grade, spec, seed=SEED + 2000 + index)
        for index, spec in enumerate(shadow_specs)
    )

    geometry = FrozenGeometryK9View(frozen_geometry)

    meta = ledger.loc[
        ledger["__decision_ts__"].ge(train_start)
        & ledger["__decision_ts__"].lt(calibration_reserve_start)
        & ledger["policy_label_available_ts"].lt(calibration_reserve_start)
        & np.isfinite(pd.to_numeric(ledger["policy_net_bps"], errors="coerce"))
    ].copy()
    leaf_source = ledger.loc[
        ledger["__decision_ts__"].ge(train_start)
        & ledger["__decision_ts__"].lt(calibration_reserve_start)
    ].copy()
    if len(meta) < 1_000:
        raise ValueError("schema-v5 correctness layer has insufficient six-month support")
    leaf_trust = _fit_leaf_trust(
        leaf_source, base_fields, calibration_reserve_start,
    )
    state = pd.concat(
        [geometry.transform(meta), leaf_trust.transform(meta)], axis=1,
    )
    aggregate_state = _aggregate_state_fields(state)
    meta = meta.reset_index(drop=True)
    state = state.reset_index(drop=True)
    meta["base_score"] = meta["prequential_base_score"].to_numpy(float)
    meta["base_rank"] = meta["prequential_base_rank42"].to_numpy(float)
    meta["base_anchor_bps"] = meta["prequential_base_anchor_bps"].to_numpy(float)
    meta["consensus_rank"] = meta["prequential_conditional_consensus_rank"].to_numpy(float)
    meta["upstream"] = meta["prequential_conditional_upstream"].to_numpy(float)
    meta = pd.concat([meta, state.loc[:, list(aggregate_state)]], axis=1)
    correctness_fields = (
        "base_score", "base_anchor_bps", "base_rank", "consensus_rank", "upstream",
        *aggregate_state,
    )
    meta_fit = _equal_month_sample(meta, MODEL_CAP, seed=SEED + 733)
    correctness = _fit_correctness(meta_fit, correctness_fields)
    severe = _fit_severe_diagnostic(
        meta_fit, correctness_fields, cutoff=calibration_reserve_start,
    )

    manifest = {
        "schema": BUNDLE_SCHEMA,
        "cutoff": cutoff_ts.isoformat(),
        "held_end_exclusive": held_end.isoformat(),
        "side": SIDE,
        "base": "strict-R3 D2 top-20 robust-clear x1.5",
        "base_weight_audit": weight_audit,
        "policy": asdict(OptimizedPolicyContract()),
        "calibration_reserve_days": int(calibration_reserve_days),
        "calibration_reserve_start": calibration_reserve_start.isoformat(),
        "calibration_reserve_contract": (
            "preceding target-free calibration reserve excluded from every "
            "supervised base/consensus/leaf/correctness/Severe fit"
            if calibration_reserve_days else "disabled_legacy_contract"
        ),
        "policy_map_target": "selected SimplePolicyOptimiser net bps",
        "residual_target": "selected-policy net bps - strict-prequential D2 base anchor",
        "conditional_consensus_contract": str(CONSENSUS_CONTRACT),
        "conditional_consensus_contract_sha256": _file_hash(CONSENSUS_CONTRACT),
        "ordinary_consensus_role": "shadow_rollback_only",
        "blend": {"base_rank": BASE_BLEND_WEIGHT, "conditional_consensus": CONSENSUS_BLEND_WEIGHT},
        "geometry": {
            "mode": "one_frozen_oct_dec_2024_K9_view_never_refit",
            "definition_start": geometry.definition_start,
            "definition_end_exclusive": geometry.definition_end_exclusive,
            "parent_bundle_sha256": geometry.parent_bundle_sha256,
            "bundle_sha256": geometry.bundle_sha256,
            "raw_k9_used_by_consensus": False,
            "raw_k9_used_by_correctness": False,
            "temperature_scale": geometry.temperature_scale,
            "effective_temperature": (
                float(geometry.parent.temperature) * geometry.temperature_scale
            ),
        },
        "correctness_target": "selected-policy net bps - base anchor bps > +100",
        "correctness_query": "4-hour UTC x side",
        "correctness_training_fraction": correctness.training_fraction,
        "correctness_training_score_floor": correctness.training_score_floor,
        "correctness_gate_domain": "pooled-global training upstream score only",
        "severe_target": severe.target,
        "geometry_definition_rows_excluded_from_severe": True,
        "geometry_definition_exclusion_scope": (
            "Severe-200 only; frozen geometry is target-free and correctness may "
            "use its aggregate OOS transform"
        ),
        "severe_role": "shadow_diagnostic_only",
        "severe_affects_final_score": False,
        "final_normalization": "same-bundle prior-28-day CDF",
        "admission": "exact-producer 28-day equal-day Cell-day trim-15 EV >= +50 bps; fail closed",
        "refit_cadence_days": FOUR_WEEK_DAYS,
        "meta_training_months": META_TRAIN_MONTHS,
        "seed": SEED,
        "source_hashes": dict(source_hashes or {}),
    }
    return CanonicalCurrentBundle(
        cutoff=cutoff_ts,
        held_end_exclusive=held_end,
        base_fields=base_fields,
        base_medians=base_medians,
        base_model=base_model,
        policy_net_map=policy_map,
        conditional_heads=conditional_heads,
        ordinary_shadow_heads=shadow_heads,
        geometry=geometry,
        leaf_trust=leaf_trust,
        correctness=correctness,
        severe_diagnostic=severe,
        manifest=manifest,
    )


def _score_base(bundle: CanonicalCurrentBundle, frame: pd.DataFrame) -> tuple[np.ndarray, ...]:
    probability = bundle.base_model.predict_proba(
        _numeric_matrix(frame, bundle.base_fields, bundle.base_medians),
    )
    lookup = {int(label): index for index, label in enumerate(bundle.base_model.classes_)}
    adverse = probability[:, lookup[0]]
    weak = probability[:, lookup[1]]
    clear = probability[:, lookup[2]]
    return adverse, weak, clear, clear - 0.5 * adverse


def score_current_bundle(
    bundle: CanonicalCurrentBundle,
    *,
    reference: pd.DataFrame,
    held: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score target-free reference and held candidates with one v3 bundle."""

    for role, frame in (("reference", reference), ("held", held)):
        _require_columns(
            frame,
            ["candidate_id", "__decision_ts__", "side_name", *bundle.base_fields],
            f"schema-v5 {role} scoring",
        )
        assert_scoring_frame_is_target_free(frame)
        if frame["candidate_id"].duplicated().any():
            raise ValueError(f"schema-v5 {role} scoring has duplicate identities")
        if not frame["side_name"].astype(str).str.lower().eq(SIDE).all():
            raise ValueError("schema-v5 is long-only")
    reference = reference.copy()
    held = held.copy()
    reference["__decision_ts__"] = pd.to_datetime(reference["__decision_ts__"], utc=True)
    held["__decision_ts__"] = pd.to_datetime(held["__decision_ts__"], utc=True)
    start = bundle.cutoff - pd.Timedelta(days=REFERENCE_DAYS)
    if not reference["__decision_ts__"].between(start, bundle.cutoff, inclusive="left").all():
        raise ValueError("schema-v5 reference must be the preceding 28-day half-open window")
    if not held["__decision_ts__"].between(
        bundle.cutoff, bundle.held_end_exclusive, inclusive="left",
    ).all():
        raise ValueError("held rows fall outside this four-week bundle")
    combined = pd.concat(
        [reference.assign(__score_role__="reference"), held.assign(__score_role__="held")],
        ignore_index=True,
    ).sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    adverse, weak, clear, base_score = _score_base(bundle, combined)
    combined["p_adverse"] = adverse
    combined["p_weak"] = weak
    combined["p_clear"] = clear
    combined["base_score"] = base_score
    reference_mask = combined["__score_role__"].eq("reference").to_numpy()
    base_reference = ScoreReference.fit(
        combined.loc[reference_mask, "base_score"],
        source="same_bundle_prior28_base_score",
    )
    combined["base_rank42"] = base_reference.cdf(combined["base_score"])
    combined["base_anchor_bps"] = bundle.policy_net_map.predict(combined["base_rank42"])

    conditional_ranks: list[np.ndarray] = []
    for head in bundle.conditional_heads:
        raw, rank = head.predict_rank(combined)
        combined[f"conditional_head__{head.spec.name}__raw"] = raw
        combined[f"conditional_head__{head.spec.name}__rank"] = rank
        conditional_ranks.append(rank)
    combined["conditional_consensus_rank"] = np.nanmedian(
        np.column_stack(conditional_ranks), axis=1,
    )
    combined["upstream"] = (
        BASE_BLEND_WEIGHT * combined["base_rank42"].to_numpy(float)
        + CONSENSUS_BLEND_WEIGHT * combined["conditional_consensus_rank"].to_numpy(float)
    )

    shadow_ranks: list[np.ndarray] = []
    for head in bundle.ordinary_shadow_heads:
        raw, rank = head.predict_rank(combined)
        combined[f"ordinary_shadow_head__{head.spec.name}__raw"] = raw
        combined[f"ordinary_shadow_head__{head.spec.name}__rank"] = rank
        shadow_ranks.append(rank)
    combined["ordinary_shadow_consensus_rank"] = np.nanmedian(
        np.column_stack(shadow_ranks), axis=1,
    )
    combined["ordinary_shadow_upstream"] = (
        BASE_BLEND_WEIGHT * combined["base_rank42"].to_numpy(float)
        + CONSENSUS_BLEND_WEIGHT * combined["ordinary_shadow_consensus_rank"].to_numpy(float)
    )

    state = _canonical_ldf_geometry_aliases(
        pd.concat(
            [bundle.geometry.transform(combined), bundle.leaf_trust.transform(combined)],
            axis=1,
        ),
    )
    aggregate_state = _aggregate_state_fields(state)
    combined = pd.concat(
        [combined, state.loc[:, list(aggregate_state)].reset_index(drop=True)],
        axis=1,
    )
    correctness_input = combined.loc[:, list(bundle.correctness.fields)]
    correctness_raw = bundle.correctness.model.predict(
        _numeric_matrix(
            correctness_input,
            bundle.correctness.fields,
            bundle.correctness.medians,
        ),
    )
    combined["correctness_raw"] = correctness_raw
    combined["correctness_rank"] = bundle.correctness.score_reference.cdf(correctness_raw)
    combined["correctness_gate_active"] = combined["upstream"].ge(
        bundle.correctness.training_score_floor,
    )
    active_multiplier = (
        CORRECTNESS_FLOOR + CORRECTNESS_SPAN * combined["correctness_rank"].to_numpy(float)
    )
    correctness_multiplier = np.where(
        combined["correctness_gate_active"].to_numpy(bool),
        active_multiplier,
        1.0,
    )
    combined["raw_correctness_demote"] = (
        combined["upstream"].to_numpy(float) * correctness_multiplier
    )
    final_reference = ScoreReference.fit(
        combined.loc[reference_mask, "raw_correctness_demote"],
        source="same_bundle_prior28_correctness_score",
    )
    combined["final_score"] = final_reference.cdf(combined["raw_correctness_demote"])

    severe = bundle.severe_diagnostic
    if severe.model is None:
        combined["severe200_probability_shadow"] = np.nan
    else:
        combined["severe200_probability_shadow"] = severe.model.predict_proba(
            _numeric_matrix(combined, severe.fields, severe.medians),
        )[:, 1]
    combined["severe_affects_final_score"] = False
    combined["bundle_sha256"] = bundle.manifest.get("bundle_sha256", "unpersisted_bundle")
    combined["geometry_bundle_sha256"] = bundle.geometry.bundle_sha256
    combined["ev_score_family_id"] = _current_ev_score_family_id(
        bundle.geometry.bundle_sha256,
    )

    score_columns = [
        "candidate_id", "__decision_ts__", "side_name", "p_adverse", "p_weak",
        "p_clear", "base_score", "base_rank42", "base_anchor_bps",
        "conditional_consensus_rank", "upstream", "correctness_raw",
        "correctness_rank", "correctness_gate_active",
        "raw_correctness_demote", "final_score",
        "ordinary_shadow_consensus_rank", "ordinary_shadow_upstream",
        "severe200_probability_shadow", "severe_affects_final_score",
        "bundle_sha256", "geometry_bundle_sha256", "ev_score_family_id",
        "__score_role__",
        # These are the stable Geometry/K9 and active-leaf summaries.  They
        # are target-free at score time and are required by the downstream
        # 45-field LDF contract.  Raw ``k09__cluster_*`` membership slots are
        # deliberately absent: their semantics are bundle-local and must not
        # be consumed by a pooled downstream model.
        *aggregate_state,
        *[
            column for column in combined.columns
            if column.startswith("conditional_head__")
            or column.startswith("ordinary_shadow_head__")
        ],
    ]
    audit = pd.DataFrame(
        [{
            "schema": SCHEMA,
            "cutoff": bundle.cutoff,
            "held_end_exclusive": bundle.held_end_exclusive,
            "reference_start": start,
            "reference_rows": int(reference_mask.sum()),
            "held_rows": int((~reference_mask).sum()),
            "held_percentile_operations": 0,
            "same_bundle_reference_and_held": True,
            "canonical_consensus": "conditional_usefulness_ten_head_v1",
            "ordinary_consensus": "shadow_rollback_only",
            "raw_k9_in_consensus": False,
            "raw_k9_in_correctness": False,
            "correctness_training_fraction": bundle.correctness.training_fraction,
            "correctness_training_score_floor": bundle.correctness.training_score_floor,
            "k9_temperature_scale": bundle.geometry.temperature_scale,
            "geometry_refit_cadence": "never",
            "geometry_parent_bundle_sha256": bundle.geometry.parent_bundle_sha256,
            "severe_affects_final_score": False,
            "final_reference": final_reference.source,
            "bundle_sha256": bundle.manifest.get("bundle_sha256", "unpersisted_bundle"),
            "geometry_bundle_sha256": bundle.geometry.bundle_sha256,
            "ev_score_family_id": _current_ev_score_family_id(
                bundle.geometry.bundle_sha256,
            ),
        }]
    )
    return combined.loc[:, score_columns].copy(), audit


def apply_current_admission(
    scored_label_ledger: pd.DataFrame,
    *,
    score_column: str = "final_score",
    net_column: str = "policy_net_bps",
    label_available_column: str = "policy_label_available_ts",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = Causal21dAdmissionSpec(mode="hierarchical_tail_side_shrinkage_v2")
    if spec.net_floor_bps != 50.0 or spec.hierarchy_windows_days != (21, 42, 84):
        raise AssertionError("canonical admission contract changed")
    return apply_causal_21d_side_admission(
        scored_label_ledger,
        score_column=score_column,
        net_column=net_column,
        decision_column="__decision_ts__",
        label_available_column=label_available_column,
        identity_column="candidate_id",
        spec=spec,
    )


def _current_ev_score_family_id(geometry_bundle_sha256: str) -> str:
    """Return the stable semantic domain of a current-v5 executable score.

    The exact fitted conversion booster is deliberately *not* part of this
    identifier: it is the producer vintage and is handled as a separate,
    stricter admission partition.  This family identifies only score outputs
    with the same policy outcome, frozen geometry meaning and CDF score
    construction.  It prevents a historical/legacy score from being treated
    as a parent merely because it happens to lie in ``[0, 1]``.
    """
    policy = OptimizedPolicyContract()
    payload = {
        "schema": SCHEMA,
        # Stable wire identity retained for sealed v5 artifacts.  ``prior42``
        # is a legacy name only: REFERENCE_DAYS and all runtime bounds are 28.
        "score": "same_conversion_model_prior42_cdf_after_correctness_demote",
        "upstream": "strict_r3_d2_75_base_25_conditional_ten_head_consensus",
        "policy": {
            "timeout_hours": policy.timeout_hours,
            "stop_loss_atr": policy.stop_loss_atr,
            "trailing_activation_atr": policy.trailing_activation_atr,
            "trailing_giveback_atr": policy.trailing_giveback_atr,
            "cost_bps_once": policy.cost_bps_once,
        },
        "geometry_bundle_sha256": str(geometry_bundle_sha256),
        "raw_k9_memberships": "excluded",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),
    ).hexdigest()


def _apply_current_admission_by_score_vintage(
    scored_label_ledger: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the canonical map separately to exact compatible producers.

    ``final_score`` is a CDF-normalised score, but a new fitted conversion
    booster can still have a different score-to-policy-net relationship.  A
    raw 21/42/84-day window that crosses a conversion cutoff is therefore not
    an economically homogeneous calibration population.  The safe unit is
    the exact conversion × monthly-upstream producer pair: both fitted states
    can alter score calibration.  Its 21/42/84-day windows may be shorter at
    a model boundary; insufficient support fails closed rather than borrowing
    raw outcomes from an older producer.
    """
    required = {
        "candidate_id", "__decision_ts__", "side_name", "final_score",
        "policy_net_bps", "policy_label_available_ts",
        "conversion_bundle_sha256", "upstream_bundle_sha256",
        "geometry_bundle_sha256",
        "ev_score_family_id", "stack_is_prequential",
    }
    missing = sorted(required.difference(scored_label_ledger.columns))
    if missing:
        raise ValueError(
            "vintage-aware current admission lacks score lineage columns: "
            f"{missing}"
        )
    work = scored_label_ledger.copy()
    for column in (
        "conversion_bundle_sha256", "upstream_bundle_sha256",
        "geometry_bundle_sha256",
        "ev_score_family_id",
    ):
        if work[column].isna().any() or work[column].astype(str).eq("").any():
            raise ValueError(f"vintage-aware current admission has null {column}")
        work[column] = work[column].astype(str)
    if not work["stack_is_prequential"].fillna(False).astype(bool).all():
        raise ValueError("vintage-aware current admission requires strict prequential scores")

    parts: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    domain_columns = [
        "ev_score_family_id", "conversion_bundle_sha256",
        "upstream_bundle_sha256", "geometry_bundle_sha256",
    ]
    # ``sort=False`` preserves chronological producer order and reindex below
    # restores the caller's identity order exactly.
    for domain, positions in work.groupby(domain_columns, sort=False).groups.items():
        subset = work.loc[positions].copy()
        mapped, audit = apply_current_admission(subset)
        family, conversion, upstream, geometry = (str(value) for value in domain)
        mapped["ev_mapping_score_family_id"] = family
        mapped["ev_mapping_conversion_vintage"] = conversion
        mapped["ev_mapping_upstream_vintage"] = upstream
        mapped["ev_mapping_geometry_bundle_sha256"] = geometry
        mapped["ev_mapping_vintage_mode"] = (
            "strict_full_producer_vintage_fail_closed_v2"
        )
        parts.append(mapped)
        audit = audit.copy()
        audit["ev_mapping_score_family_id"] = family
        audit["ev_mapping_conversion_vintage"] = conversion
        audit["ev_mapping_upstream_vintage"] = upstream
        audit["ev_mapping_geometry_bundle_sha256"] = geometry
        audit["ev_mapping_vintage_mode"] = (
            "strict_full_producer_vintage_fail_closed_v2"
        )
        audit["upstream_vintage_count"] = 1
        audits.append(audit)
    if not parts:
        raise ValueError("vintage-aware current admission received no score rows")
    output = pd.concat(parts, axis=0).reindex(scored_label_ledger.index)
    audit = pd.concat(audits, ignore_index=True)
    if len(output) != len(scored_label_ledger) or output["candidate_id"].duplicated().any():
        raise AssertionError("vintage-aware admission changed candidate identity")
    return output, audit


def apply_current_admission_snapshot(
    *,
    resolved_score_ledger: pd.DataFrame,
    current_scores: pd.DataFrame,
    ev_bridge_bundle: StrictR3EVBridgeBundle | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map target-free current scores with only prior resolved policy outcomes.

    ``apply_current_admission`` operates on a chronological labelled ledger.
    At inference, however, the candidate being considered has a score but no
    outcome yet.  Represent it explicitly as unresolved (``policy_net_bps``
    is NaN) with its earliest possible 12-hour label timestamp.  The admission
    implementation then cannot use it as reference support.  Without an EV
    bridge this retains the exact-producer 21/42/84 control; with a frozen OOF
    bridge it applies a common-bps prior plus a causal residual correction and
    does not mechanically cold-start at an upstream refit.
    """
    required_current = {
        "candidate_id", "__decision_ts__", "side_name", "final_score",
        "conversion_bundle_sha256", "upstream_bundle_sha256",
        "geometry_bundle_sha256",
        "ev_score_family_id", "stack_is_prequential",
    }
    missing_current = sorted(required_current.difference(current_scores.columns))
    if missing_current:
        raise ValueError(f"current admission snapshot lacks: {missing_current}")
    assert_scoring_frame_is_target_free(current_scores)
    current = current_scores.copy()
    if current["candidate_id"].isna().any() or current["candidate_id"].duplicated().any():
        raise ValueError("current admission snapshot requires immutable unique candidate identities")
    if {"policy_net_bps", "policy_label_available_ts"}.intersection(current.columns):
        raise ValueError("current admission snapshot must not receive policy outcomes or label timestamps")
    if set(current["candidate_id"]).intersection(set(resolved_score_ledger.get("candidate_id", []))):
        raise ValueError("current admission snapshot overlaps resolved candidate identities")
    current["__decision_ts__"] = pd.to_datetime(current["__decision_ts__"], utc=True, errors="raise")
    current["policy_net_bps"] = np.nan
    current["policy_label_available_ts"] = (
        current["__decision_ts__"] + pd.Timedelta(hours=OptimizedPolicyContract().timeout_hours)
    )
    current["__current_admission_snapshot__"] = True

    resolved = resolved_score_ledger.copy()
    required_resolved = {
        "candidate_id", "__decision_ts__", "side_name", "final_score",
        "policy_net_bps", "policy_label_available_ts",
        "conversion_bundle_sha256", "upstream_bundle_sha256",
        "geometry_bundle_sha256",
        "ev_score_family_id", "stack_is_prequential",
    }
    missing_resolved = sorted(required_resolved.difference(resolved.columns))
    if missing_resolved:
        raise ValueError(
            "current admission snapshot lacks resolved score lineage columns: "
            f"{missing_resolved}"
        )
    if not resolved["stack_is_prequential"].fillna(False).astype(bool).all():
        raise ValueError("current admission snapshot requires strict prequential resolved scores")
    current_domain = current.loc[:, [
        "ev_score_family_id", "conversion_bundle_sha256",
        "upstream_bundle_sha256", "geometry_bundle_sha256",
    ]].drop_duplicates()
    if len(current_domain) != 1:
        raise ValueError("one live admission snapshot must use one score-domain vintage")
    family, conversion, upstream, geometry = current_domain.iloc[0].astype(str).tolist()
    if ev_bridge_bundle is None:
        # This filter is the exact-producer audit control.  It intentionally
        # does not use older conversion/upstream history as a raw score parent.
        # It remains available to quantify the bridge, but its cold-start
        # fail-closed behavior is not the executable no-drought path.
        resolved = resolved.loc[
            resolved["ev_score_family_id"].astype(str).eq(family)
            & resolved["conversion_bundle_sha256"].astype(str).eq(conversion)
            & resolved["upstream_bundle_sha256"].astype(str).eq(upstream)
            & resolved["geometry_bundle_sha256"].astype(str).eq(geometry)
        ].copy()
    else:
        if ev_bridge_bundle.fit_cutoff > current["__decision_ts__"].min():
            raise ValueError("EV bridge was fit after the current live decision")
        if ev_bridge_bundle.ev_score_family_id != family:
            raise ValueError("EV bridge and current score family differ")
        if ev_bridge_bundle.geometry_bundle_sha256 != geometry:
            raise ValueError("EV bridge and current frozen geometry differ")
        producer_lineage = dict(ev_bridge_bundle.producer_lineage)
        if producer_lineage.get("conversion_bundle_sha256") not in (None, conversion):
            raise ValueError("immediate EV calibration belongs to another conversion bundle")
        if producer_lineage.get("upstream_bundle_sha256") not in (None, upstream):
            raise ValueError("immediate EV calibration belongs to another upstream bundle")
        if producer_lineage:
            # An immediate-reserve calibrator is more restrictive than the
            # old common-bps bridge: both its prior map and its residual
            # correction belong to exactly one upstream x conversion pair.
            # Mixing an older producer here would either make the map reject
            # the combined ledger or, worse in a future relaxed validator,
            # reintroduce a cross-vintage score calibration.  A newly fitted
            # producer therefore starts with its own reserve prior and only
            # accumulates residuals from rows that it scored itself.
            resolved = resolved.loc[
                resolved["ev_score_family_id"].astype(str).eq(family)
                & resolved["geometry_bundle_sha256"].astype(str).eq(geometry)
                & resolved["conversion_bundle_sha256"].astype(str).eq(conversion)
                & resolved["upstream_bundle_sha256"].astype(str).eq(upstream)
            ].copy()
        else:
            # A legacy common-bps bridge has no exact producer map.  It may
            # consume only prior realised *bps residuals* across producers;
            # it never pools their raw score domains.
            resolved = resolved.loc[
                resolved["ev_score_family_id"].astype(str).eq(family)
                & resolved["geometry_bundle_sha256"].astype(str).eq(geometry)
            ].copy()
    resolved["__current_admission_snapshot__"] = False
    combined = pd.concat([resolved, current], ignore_index=True, sort=False)
    if ev_bridge_bundle is None:
        mapped, audit = _apply_current_admission_by_score_vintage(combined)
    else:
        mapped, audit = apply_strict_r3_ev_bridge(
            combined,
            bundle=ev_bridge_bundle,
            net_column="policy_net_bps",
            decision_column="__decision_ts__",
            label_available_column="policy_label_available_ts",
            identity_column="candidate_id",
        )
    output = mapped.loc[
        mapped["__current_admission_snapshot__"].fillna(False).astype(bool)
    ].drop(columns="__current_admission_snapshot__")
    if len(output) != len(current) or output["candidate_id"].duplicated().any():
        raise AssertionError("current admission snapshot changed candidate identity")
    if output["policy_net_bps"].notna().any():
        raise AssertionError("current admission snapshot unexpectedly has realised policy outcomes")
    return output, audit


def apply_current_admission_by_geometry(
    scored_label_ledger: pd.DataFrame,
    *,
    geometry_mode: str = "frozen",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the EV map under either frozen or isolated K9 semantics.

    Canonical scoring has one immutable geometry identity and a continuous
    causal admission history.  The episodic ablation is deliberately stricter:
    all 21/42/84-day support is reset at a geometry boundary.  Thus no
    score/outcome history from an incompatible K9 representation can increase
    the support of a later episode.
    """

    if geometry_mode == "frozen":
        lineage = {
            "conversion_bundle_sha256", "geometry_bundle_sha256",
            "stack_is_prequential",
        }
        if lineage.issubset(scored_label_ledger.columns):
            work = scored_label_ledger.copy()
            # Pre-vintage-repair v5 ledgers did persist the conversion and
            # frozen geometry identities.  Attach the deterministic semantic
            # family on read so they can be replayed under the repaired map;
            # this is lineage migration, not a new score or outcome fit.
            if "ev_score_family_id" not in work:
                work["ev_score_family_id"] = work[
                    "geometry_bundle_sha256"
                ].astype(str).map(_current_ev_score_family_id)
            return _apply_current_admission_by_score_vintage(work)
        return apply_current_admission(scored_label_ledger)
    if geometry_mode != "episode-isolated":
        raise ValueError(f"unsupported geometry admission mode: {geometry_mode}")
    if "geometry_bundle_sha256" not in scored_label_ledger:
        raise ValueError("episode-isolated admission requires geometry_bundle_sha256")
    if scored_label_ledger["geometry_bundle_sha256"].isna().any():
        raise ValueError("episode-isolated admission encountered null geometry identity")
    admitted_parts: list[pd.DataFrame] = []
    audit_parts: list[pd.DataFrame] = []
    for geometry_id, positions in scored_label_ledger.groupby(
        "geometry_bundle_sha256", sort=False,
    ).groups.items():
        subset = scored_label_ledger.loc[positions].copy()
        if {
            "conversion_bundle_sha256", "stack_is_prequential",
        }.issubset(subset.columns):
            if "ev_score_family_id" not in subset:
                subset["ev_score_family_id"] = subset[
                    "geometry_bundle_sha256"
                ].astype(str).map(_current_ev_score_family_id)
            admitted, audit = _apply_current_admission_by_score_vintage(subset)
        else:
            admitted, audit = apply_current_admission(subset)
        admitted_parts.append(admitted)
        audit_parts.append(audit.assign(geometry_bundle_sha256=str(geometry_id)))
    if not admitted_parts:
        raise ValueError("episode-isolated admission received no rows")
    output = pd.concat(admitted_parts, axis=0).reindex(scored_label_ledger.index)
    audit = pd.concat(audit_parts, ignore_index=True)
    if len(output) != len(scored_label_ledger) or output["candidate_id"].duplicated().any():
        raise AssertionError("geometry-isolated admission changed candidate identity")
    return output, audit


def persist_current_bundle(
    bundle: CanonicalCurrentBundle, directory: Path,
) -> dict[str, Any]:
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"immutable schema-v5 bundle already exists: {directory}")
    directory.mkdir(parents=True)
    payload = directory / "canonical_current_bundle.joblib"
    joblib.dump(bundle, payload, compress=3)
    bundle_hash = _file_hash(payload)
    manifest = {
        **bundle.manifest,
        "schema": BUNDLE_SCHEMA,
        "bundle_file": payload.name,
        "bundle_sha256": bundle_hash,
        "base_fields_sha256": _json_hash(list(bundle.base_fields)),
        "conditional_heads": [
            {
                "name": head.spec.name,
                "query": head.spec.query,
                "weight_mode": head.spec.weight_mode,
                "field_count": len(head.spec.fields),
                "fields_sha256": _json_hash(list(head.spec.fields)),
                "rank_reference_rows": len(head.score_reference.sorted_values),
            }
            for head in bundle.conditional_heads
        ],
        "ordinary_shadow_heads": [head.spec.name for head in bundle.ordinary_shadow_heads],
        "correctness_fields": list(bundle.correctness.fields),
        "correctness_fields_sha256": _json_hash(list(bundle.correctness.fields)),
        "correctness_training_fraction": bundle.correctness.training_fraction,
        "correctness_training_score_floor": bundle.correctness.training_score_floor,
        "k9_temperature_scale": bundle.geometry.temperature_scale,
        "geometry_refit_cadence": "never",
        "geometry_parent_bundle_sha256": bundle.geometry.parent_bundle_sha256,
        "raw_k9_prohibited_from_correctness": True,
        "severe_affects_final_score": False,
    }
    (directory / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n",
    )
    bundle.manifest["bundle_sha256"] = bundle_hash
    return manifest


def load_current_bundle(directory: Path) -> CanonicalCurrentBundle:
    directory = Path(directory)
    manifest = json.loads((directory / "run_manifest.json").read_text())
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("not a canonical schema-v5 bundle")
    path = directory / manifest["bundle_file"]
    if _file_hash(path) != manifest["bundle_sha256"]:
        raise ValueError("schema-v5 bundle hash mismatch")
    bundle = joblib.load(path)
    if not isinstance(bundle, CanonicalCurrentBundle) or bundle.schema != BUNDLE_SCHEMA:
        raise TypeError("schema-v5 bundle payload has the wrong type/schema")
    bundle.__post_init__()
    if manifest.get("severe_affects_final_score") is not False:
        raise ValueError("schema-v5 manifest attempts to activate Severe-200")
    if manifest.get("raw_k9_prohibited_from_correctness") is not True:
        raise ValueError("schema-v5 manifest does not enforce the K9 input veto")
    if not np.isclose(
        float(manifest.get("correctness_training_fraction", np.nan)),
        CORRECTNESS_TRAIN_FRACTION,
    ):
        raise ValueError("schema-v5 manifest has the wrong correctness curriculum")
    if not np.isclose(
        float(manifest.get("k9_temperature_scale", np.nan)),
        K9_TEMPERATURE_SCALE,
    ):
        raise ValueError("schema-v5 manifest has the wrong K9 temperature scale")
    if manifest.get("geometry_refit_cadence") != "never":
        raise ValueError("schema-v5 manifest permits geometry/K9 re-fitting")
    if manifest.get("geometry_parent_bundle_sha256") != bundle.geometry.parent_bundle_sha256:
        raise ValueError("schema-v5 manifest parent geometry identity mismatch")
    bundle.manifest["bundle_sha256"] = manifest["bundle_sha256"]
    return bundle


# ---------------------------------------------------------------------------
# Exact production cadence: monthly upstream + four-week conversion
# ---------------------------------------------------------------------------

UPSTREAM_BUNDLE_SCHEMA = "strict_r3_canonical_monthly_upstream_v3"
CONVERSION_BUNDLE_SCHEMA = "strict_r3_canonical_four_week_conversion_v5_frozen_geometry"


@dataclass
class MonthlyUpstreamBundle:
    cutoff: pd.Timestamp
    end_exclusive: pd.Timestamp
    base_fields: tuple[str, ...]
    base_medians: np.ndarray
    base_model: LGBMClassifier
    base_score_reference: ScoreReference
    policy_net_map: Any
    conditional_heads: tuple[FittedConsensusHead, ...]
    ordinary_shadow_heads: tuple[FittedConsensusHead, ...]
    policy: OptimizedPolicyContract = field(default_factory=OptimizedPolicyContract)
    schema: str = UPSTREAM_BUNDLE_SCHEMA
    manifest: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cutoff = _utc(self.cutoff)
        self.end_exclusive = _utc(self.end_exclusive)
        expected = self.cutoff + pd.offsets.MonthBegin(1)
        lockstep = self.manifest.get("refit_cadence") == "explicit_lockstep_window"
        if lockstep:
            if self.end_exclusive != self.cutoff + pd.Timedelta(days=FOUR_WEEK_DAYS):
                raise ValueError("explicit lock-step upstream bundles use exactly one 28-day window")
        elif self.cutoff.day != 1 or self.cutoff != self.cutoff.normalize() or self.end_exclusive != expected:
            raise ValueError("upstream bundles use one complete UTC calendar month")
        if len(self.base_fields) != 120 or len(self.conditional_heads) != 10:
            raise ValueError("monthly upstream bundle violates the frozen feature/head contract")
        if len(self.ordinary_shadow_heads) != 10:
            raise ValueError("monthly upstream bundle must retain ten shadow heads")


@dataclass
class FourWeekConversionBundle:
    cutoff: pd.Timestamp
    end_exclusive: pd.Timestamp
    base_fields: tuple[str, ...]
    geometry: FrozenGeometryK9View
    leaf_trust: LeafTrustBundle
    correctness: CorrectnessHead
    severe_diagnostic: SevereDiagnostic
    policy: OptimizedPolicyContract = field(default_factory=OptimizedPolicyContract)
    schema: str = CONVERSION_BUNDLE_SCHEMA
    manifest: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cutoff = _utc(self.cutoff)
        self.end_exclusive = _utc(self.end_exclusive)
        if self.end_exclusive != self.cutoff + pd.Timedelta(days=FOUR_WEEK_DAYS):
            raise ValueError("conversion bundles use an exact four-week UTC block")
        if len(self.base_fields) != 120:
            raise ValueError("conversion bundle requires the frozen 120-field contract")
        if any("k09__cluster_" in field for field in self.correctness.fields):
            raise ValueError("raw K9 memberships are prohibited from conversion correctness")
        if not np.isclose(self.correctness.training_fraction, CORRECTNESS_TRAIN_FRACTION):
            raise ValueError("canonical conversion correctness must use top-30% training")
        if not isinstance(self.geometry, FrozenGeometryK9View):
            raise ValueError(
                "canonical conversion requires one frozen Oct-Dec geometry/K9 view; "
                "rolling geometry is prohibited",
            )
        if self.geometry.definition_start != "2024-10-01T00:00:00+00:00" or self.geometry.definition_end_exclusive != "2025-01-01T00:00:00+00:00":
            raise ValueError("canonical conversion geometry definition window is not Oct-Dec 2024")
        if self.severe_diagnostic.affects_final_score:
            raise ValueError("Severe-200 cannot affect the canonical conversion score")


def train_monthly_upstream_bundle(
    *,
    cutoff: Any,
    training_ledger: pd.DataFrame,
    prior42_features: pd.DataFrame,
    base_fields: Sequence[str],
    source_hashes: Mapping[str, str] | None = None,
    calibration_reserve_days: int = 0,
    held_end_exclusive: Any | None = None,
) -> MonthlyUpstreamBundle:
    """Fit the monthly D2 base/map/conditional consensus producer.

    ``teacher_base_rank42`` must be an older strict-prequential base rank.  It
    is only a training weight and never enters the inference feature matrix.
    Map and residual fitting consume the row's latest available causal
    ``prequential_base_rank42``; callers may progressively replace the warm
    start ranks with current monthly predictions as history accumulates.
    """

    # Legacy callers pass a calendar-month start and retain their historical
    # end-of-month scoring envelope.  Lock-step producers pass an explicit
    # 28-day end, which must not be silently rounded back to month start.
    cutoff_ts = _utc(cutoff)
    end = (
        cutoff_ts + pd.offsets.MonthBegin(1)
        if held_end_exclusive is None
        else _utc(held_end_exclusive)
    )
    if end <= cutoff_ts:
        raise ValueError("monthly upstream held end must be after cutoff")
    if not 0 <= int(calibration_reserve_days) <= REFERENCE_DAYS:
        raise ValueError(
            "monthly upstream calibration reserve must lie between zero and the "
            "same-model reference horizon",
        )
    calibration_reserve_start = cutoff_ts - pd.Timedelta(
        days=int(calibration_reserve_days),
    )
    fields = tuple(str(field) for field in base_fields)
    required = [
        "candidate_id", "__decision_ts__", "side_name", "r3_class",
        "r3_label_available_ts", "policy_net_bps", "policy_label_available_ts",
        "teacher_base_rank42", "prequential_base_rank42", "stack_is_prequential",
        *fields,
    ]
    _require_columns(training_ledger, required, "monthly upstream training ledger")
    # The replay already normalises the ledger timestamps before this call.
    # Avoid a multi-gigabyte copy merely to reassign the same timezone-aware
    # values; retain the defensive conversion for generic callers.
    timestamp_columns = (
        "__decision_ts__", "r3_label_available_ts", "policy_label_available_ts",
    )
    requires_timestamp_copy = any(
        not pd.api.types.is_datetime64tz_dtype(training_ledger[column])
        for column in timestamp_columns
    )
    ledger = training_ledger.copy() if requires_timestamp_copy else training_ledger
    if requires_timestamp_copy:
        for column in timestamp_columns:
            ledger[column] = pd.to_datetime(ledger[column], utc=True, errors="raise")
    if (ledger["__decision_ts__"] >= cutoff_ts).any():
        raise ValueError("monthly upstream training crosses its held month")
    if not ledger["stack_is_prequential"].fillna(False).astype(bool).all():
        raise ValueError("monthly upstream training received a non-prequential row")
    if not ledger["side_name"].astype(str).str.lower().eq(SIDE).all():
        raise ValueError("monthly upstream producer is long-only")

    base_fit = ledger.loc[
        ledger["__decision_ts__"].lt(calibration_reserve_start)
        & ledger["r3_label_available_ts"].lt(calibration_reserve_start)
        & ledger["r3_class"].notna()
    ].sort_values("r3_label_available_ts", kind="stable").tail(BASE_TRAIN_CAP).copy()
    if len(base_fit) < 1_000 or base_fit["r3_class"].nunique() < 3:
        raise ValueError("monthly upstream base lacks the required row/class support")
    weights, weight_audit = build_distillation_weights(
        base_fit,
        teacher_rank_column="teacher_base_rank42",
        layer="base",
        spec=D2_SPEC,
    )
    medians = _fit_medians(base_fit, fields)
    model = LGBMClassifier(**BASE_PARAMS).fit(
        _numeric_matrix(base_fit, fields, medians),
        base_fit["r3_class"].astype(int).to_numpy(),
        sample_weight=weights,
    )
    del base_identity, base_fit, weights
    gc.collect()
    print(
        json.dumps({"event": "compact_upstream_base_complete", "cutoff": cutoff_ts.isoformat()}),
        flush=True,
    )
    _require_columns(
        prior42_features,
        ["candidate_id", "__decision_ts__", "side_name", *fields],
        "monthly upstream prior-42 reference",
    )
    prior = prior42_features.copy()
    prior["__decision_ts__"] = pd.to_datetime(prior["__decision_ts__"], utc=True)
    if not prior["__decision_ts__"].between(
        cutoff_ts - pd.Timedelta(days=REFERENCE_DAYS), cutoff_ts, inclusive="left",
    ).all():
        raise ValueError("monthly upstream base reference is not the preceding 28 days")
    reference_hours = pd.DatetimeIndex(prior["__decision_ts__"].drop_duplicates()).sort_values()
    expected_hours = pd.date_range(
        cutoff_ts - pd.Timedelta(days=REFERENCE_DAYS),
        cutoff_ts - pd.Timedelta(hours=1),
        freq="h",
        tz="UTC",
    )
    if len(reference_hours) != REQUIRED_REFERENCE_HOURS or not reference_hours.equals(expected_hours):
        raise ValueError(
            "monthly upstream base reference must cover every one of the preceding "
            f"{REFERENCE_DAYS} UTC calendar days; observed {len(reference_hours)} "
            f"of {REQUIRED_REFERENCE_HOURS} hourly timestamps"
        )
    assert_scoring_frame_is_target_free(prior)
    probability = model.predict_proba(_numeric_matrix(prior, fields, medians))
    lookup = {int(label): index for index, label in enumerate(model.classes_)}
    base_reference = ScoreReference.fit(
        probability[:, lookup[2]] - 0.5 * probability[:, lookup[0]],
        source=f"{cutoff_ts:%Y-%m}_same_monthly_base_prior42",
    )

    map_mask = (
        ledger["__decision_ts__"].lt(calibration_reserve_start).to_numpy()
        & ledger["policy_label_available_ts"].lt(calibration_reserve_start).to_numpy()
        & np.isfinite(pd.to_numeric(ledger["policy_net_bps"], errors="coerce").to_numpy(float))
        & np.isfinite(
            pd.to_numeric(ledger["prequential_base_rank42"], errors="coerce").to_numpy(float),
        )
    )
    # The policy map needs all valid rows, but it needs only the two scalar
    # columns below.  Keeping this compact population lets each consensus head
    # reproduce its capped query sample without a 120-field copy.
    map_fit = ledger.loc[
        map_mask,
        [
            "candidate_id", "__decision_ts__", "side_name",
            "prequential_base_rank42", "policy_net_bps",
        ],
    ].copy()
    map_fit["__ledger_position__"] = np.flatnonzero(map_mask)
    mapping = fit_policy_net_map(map_fit["prequential_base_rank42"], map_fit["policy_net_bps"])
    anchor = mapping.predict(map_fit["prequential_base_rank42"])
    residual = map_fit["policy_net_bps"].to_numpy(float) - np.asarray(anchor, dtype=float)
    grade = _residual_grade(residual, RESIDUAL_BANDS_BPS)
    sampling_frame = map_fit.loc[
        :, ["candidate_id", "__decision_ts__", "side_name", "__ledger_position__"],
    ]
    conditional = tuple(
        _fit_consensus_head_compact(
            ledger, sampling_frame, grade, spec, seed=SEED + 3000 + index,
        )
        for index, spec in enumerate(load_conditional_consensus_contract(fields))
    )
    shadow = tuple(
        _fit_consensus_head_compact(
            ledger, sampling_frame, grade, spec, seed=SEED + 4000 + index,
        )
        for index, spec in enumerate(_ordinary_shadow_contract(fields))
    )
    manifest = {
        "schema": UPSTREAM_BUNDLE_SCHEMA,
        "cutoff": cutoff_ts.isoformat(),
        "end_exclusive": end.isoformat(),
        "refit_cadence": (
            "calendar_month" if held_end_exclusive is None else "explicit_lockstep_window"
        ),
        "side": SIDE,
        "base": "strict-R3 D2 top-20 robust-clear x1.5",
        "base_weight_audit": weight_audit,
        "base_reference": base_reference.source,
        "base_reference_rows": int(len(prior)),
        "base_reference_hours": int(len(reference_hours)),
        "calibration_reserve_days": int(calibration_reserve_days),
        "calibration_reserve_start": calibration_reserve_start.isoformat(),
        "calibration_reserve_contract": (
            "preceding target-free reference rows excluded from all upstream "
            "supervised base/map/consensus fits"
            if calibration_reserve_days else "disabled_legacy_contract"
        ),
        "policy": asdict(OptimizedPolicyContract()),
        "conditional_consensus_contract_sha256": _file_hash(CONSENSUS_CONTRACT),
        "ordinary_consensus_role": "shadow_rollback_only",
        "residual_target": "selected-policy net bps - causal base anchor",
        "source_hashes": dict(source_hashes or {}),
    }
    return MonthlyUpstreamBundle(
        cutoff_ts, end, fields, medians, model, base_reference, mapping,
        conditional, shadow, manifest=manifest,
    )


def train_monthly_upstream_bundle_compact_features(
    *,
    cutoff: Any,
    training_ledger: pd.DataFrame,
    prior42_features: pd.DataFrame,
    base_fields: Sequence[str],
    feature_loader: Callable[[pd.DataFrame, Sequence[str], str], pd.DataFrame],
    permitted_empty_reference_hours: Sequence[Any] = (),
    source_hashes: Mapping[str, str] | None = None,
    calibration_reserve_days: int = 0,
    held_end_exclusive: Any | None = None,
) -> MonthlyUpstreamBundle:
    """Exact upstream training with source fields materialised after sampling.

    The canonical ledger is intentionally reusable and therefore very wide.
    This variant keeps the supervisor/map/query population compact, selects
    the exact same 240k complete LambdaRank queries as
    :func:`train_monthly_upstream_bundle`, then asks ``feature_loader`` for
    only the selected candidate identities and the head's declared fields.
    It is an execution-memory optimisation: target, chronology, sampling,
    weights, model parameters, score references and bundle contents are
    unchanged.
    """
    cutoff_ts = _utc(cutoff)
    end = (
        cutoff_ts + pd.offsets.MonthBegin(1)
        if held_end_exclusive is None
        else _utc(held_end_exclusive)
    )
    if end <= cutoff_ts:
        raise ValueError("monthly upstream held end must be after cutoff")
    if not 0 <= int(calibration_reserve_days) <= REFERENCE_DAYS:
        raise ValueError(
            "monthly upstream calibration reserve must lie between zero and the "
            "same-model reference horizon",
        )
    calibration_reserve_start = cutoff_ts - pd.Timedelta(days=int(calibration_reserve_days))
    fields = tuple(str(field) for field in base_fields)
    required = [
        "candidate_id", "__decision_ts__", "side_name", "r3_class",
        "r3_label_available_ts", "policy_net_bps", "policy_label_available_ts",
        "teacher_base_rank42", "prequential_base_rank42", "stack_is_prequential",
    ]
    _require_columns(training_ledger, required, "compact monthly upstream training ledger")
    timestamp_columns = (
        "__decision_ts__", "r3_label_available_ts", "policy_label_available_ts",
    )
    requires_timestamp_copy = any(
        not pd.api.types.is_datetime64tz_dtype(training_ledger[column])
        for column in timestamp_columns
    )
    ledger = training_ledger.copy() if requires_timestamp_copy else training_ledger
    if requires_timestamp_copy:
        for column in timestamp_columns:
            ledger[column] = pd.to_datetime(ledger[column], utc=True, errors="raise")
    if (ledger["__decision_ts__"] >= cutoff_ts).any():
        raise ValueError("monthly upstream training crosses its held month")
    if not ledger["stack_is_prequential"].fillna(False).astype(bool).all():
        raise ValueError("monthly upstream training received a non-prequential row")
    if not ledger["side_name"].astype(str).str.lower().eq(SIDE).all():
        raise ValueError("monthly upstream producer is long-only")

    base_identity = ledger.loc[
        ledger["__decision_ts__"].lt(calibration_reserve_start)
        & ledger["r3_label_available_ts"].lt(calibration_reserve_start)
        & ledger["r3_class"].notna()
    ].sort_values("r3_label_available_ts", kind="stable").tail(BASE_TRAIN_CAP).copy()
    base_fit = feature_loader(base_identity, fields, "monthly upstream base training")
    print(
        json.dumps({"event": "compact_upstream_base_features_complete", "cutoff": cutoff_ts.isoformat(), "rows": int(len(base_fit))}),
        flush=True,
    )
    if len(base_fit) != len(base_identity) or base_fit["candidate_id"].duplicated().any():
        raise AssertionError("compact base feature loader changed selected identities")
    if len(base_fit) < 1_000 or base_fit["r3_class"].nunique() < 3:
        raise ValueError("monthly upstream base lacks the required row/class support")
    weights, weight_audit = build_distillation_weights(
        base_fit,
        teacher_rank_column="teacher_base_rank42",
        layer="base",
        spec=D2_SPEC,
    )
    medians = _fit_medians(base_fit, fields)
    print(
        json.dumps({"event": "compact_upstream_base_fit_begin", "cutoff": cutoff_ts.isoformat()}),
        flush=True,
    )
    model = LGBMClassifier(**BASE_PARAMS).fit(
        _numeric_matrix(base_fit, fields, medians),
        base_fit["r3_class"].astype(int).to_numpy(),
        sample_weight=weights,
    )
    _require_columns(
        prior42_features,
        ["candidate_id", "__decision_ts__", "side_name", *fields],
        "monthly upstream prior-42 reference",
    )
    prior = prior42_features.copy()
    prior["__decision_ts__"] = pd.to_datetime(prior["__decision_ts__"], utc=True)
    if not prior["__decision_ts__"].between(
        cutoff_ts - pd.Timedelta(days=REFERENCE_DAYS), cutoff_ts, inclusive="left",
    ).all():
        raise ValueError("monthly upstream base reference is not the preceding 28 days")
    reference_hours = pd.DatetimeIndex(prior["__decision_ts__"].drop_duplicates()).sort_values()
    expected_hours = pd.date_range(
        cutoff_ts - pd.Timedelta(days=REFERENCE_DAYS), cutoff_ts - pd.Timedelta(hours=1),
        freq="h", tz="UTC",
    )
    observed_empty_hours = expected_hours.difference(reference_hours)
    # ``DatetimeIndex([])`` defaults to timezone-naive.  The observed index
    # is UTC-aware even when both sets are empty, so normalise explicitly
    # before the equality guard.  Without this, a complete reference window
    # can be rejected solely because there are zero permitted empty hours.
    permitted_values = [_utc(value) for value in permitted_empty_reference_hours]
    permitted = pd.DatetimeIndex(permitted_values)
    if permitted.tz is None:
        permitted = permitted.tz_localize("UTC")
    else:
        permitted = permitted.tz_convert("UTC")
    permitted = permitted.sort_values()
    if not observed_empty_hours.equals(permitted):
        raise ValueError(
            "monthly upstream base reference must cover every one of the preceding "
            f"{REFERENCE_DAYS} UTC calendar days, except explicitly declared "
            f"zero-candidate source hours; observed missing={list(map(str, observed_empty_hours))}",
        )
    assert_scoring_frame_is_target_free(prior)
    probability = model.predict_proba(_numeric_matrix(prior, fields, medians))
    lookup = {int(label): index for index, label in enumerate(model.classes_)}
    base_reference = ScoreReference.fit(
        probability[:, lookup[2]] - 0.5 * probability[:, lookup[0]],
        source=f"{cutoff_ts:%Y-%m}_same_monthly_base_prior42",
    )

    map_mask = (
        ledger["__decision_ts__"].lt(calibration_reserve_start).to_numpy()
        & ledger["policy_label_available_ts"].lt(calibration_reserve_start).to_numpy()
        & np.isfinite(pd.to_numeric(ledger["policy_net_bps"], errors="coerce").to_numpy(float))
        & np.isfinite(
            pd.to_numeric(ledger["prequential_base_rank42"], errors="coerce").to_numpy(float),
        )
    )
    map_fit = ledger.loc[
        map_mask,
        [
            "candidate_id", "__decision_ts__", "side_name",
            "prequential_base_rank42", "policy_net_bps",
        ],
    ].copy()
    mapping = fit_policy_net_map(map_fit["prequential_base_rank42"], map_fit["policy_net_bps"])
    residual = map_fit["policy_net_bps"].to_numpy(float) - np.asarray(
        mapping.predict(map_fit["prequential_base_rank42"]), dtype=float,
    )
    grade = _residual_grade(residual, RESIDUAL_BANDS_BPS)
    print(
        json.dumps({"event": "compact_upstream_map_complete", "cutoff": cutoff_ts.isoformat(), "rows": int(len(map_fit))}),
        flush=True,
    )

    def fit_head(spec: ConsensusHeadSpec, seed: int) -> FittedConsensusHead:
        sampled_identity, target, groups = _sample_complete_consensus_queries(
            map_fit.loc[:, ["candidate_id", "__decision_ts__", "side_name"]].reset_index(drop=True),
            grade, spec, seed=seed,
        )
        sampled = feature_loader(
            sampled_identity, spec.fields, f"monthly consensus {spec.name}",
        )
        if not np.array_equal(
            sampled["candidate_id"].to_numpy(), sampled_identity["candidate_id"].to_numpy(),
        ):
            raise AssertionError("compact consensus feature loader changed sampled identities")
        # Query/month metadata are determined before source materialisation.
        sampled["__query__"] = sampled_identity["__query__"].to_numpy()
        sampled["__month__"] = sampled_identity["__month__"].to_numpy()
        return _fit_consensus_head_from_sample(sampled, target, groups, spec, seed=seed)

    conditional_list: list[FittedConsensusHead] = []
    for index, spec in enumerate(load_conditional_consensus_contract(fields)):
        conditional_list.append(fit_head(spec, SEED + 3000 + index))
        gc.collect()
        print(
            json.dumps({"event": "compact_upstream_conditional_head_complete", "cutoff": cutoff_ts.isoformat(), "head": spec.name}),
            flush=True,
        )
    conditional = tuple(conditional_list)
    shadow_list: list[FittedConsensusHead] = []
    for index, spec in enumerate(_ordinary_shadow_contract(fields)):
        shadow_list.append(fit_head(spec, SEED + 4000 + index))
        gc.collect()
        print(
            json.dumps({"event": "compact_upstream_shadow_head_complete", "cutoff": cutoff_ts.isoformat(), "head": spec.name}),
            flush=True,
        )
    shadow = tuple(shadow_list)
    manifest = {
        "schema": UPSTREAM_BUNDLE_SCHEMA,
        "cutoff": cutoff_ts.isoformat(),
        "end_exclusive": end.isoformat(),
        "refit_cadence": (
            "calendar_month" if held_end_exclusive is None else "explicit_lockstep_window"
        ),
        "side": SIDE,
        "base": "strict-R3 D2 top-20 robust-clear x1.5",
        "base_weight_audit": weight_audit,
        "base_reference": base_reference.source,
        "base_reference_rows": int(len(prior)),
        "base_reference_hours": int(len(reference_hours)),
        "base_reference_zero_candidate_hours": [
            value.isoformat() for value in observed_empty_hours
        ],
        "calibration_reserve_days": int(calibration_reserve_days),
        "calibration_reserve_start": calibration_reserve_start.isoformat(),
        "calibration_reserve_contract": (
            "preceding target-free reference rows excluded from all upstream "
            "supervised base/map/consensus fits"
            if calibration_reserve_days else "disabled_legacy_contract"
        ),
        "policy": asdict(OptimizedPolicyContract()),
        "conditional_consensus_contract_sha256": _file_hash(CONSENSUS_CONTRACT),
        "ordinary_consensus_role": "shadow_rollback_only",
        "residual_target": "selected-policy net bps - causal base anchor",
        "feature_materialisation": "post-query-selection target-free source projection",
        "source_hashes": dict(source_hashes or {}),
    }
    return MonthlyUpstreamBundle(
        cutoff_ts, end, fields, medians, model, base_reference, mapping,
        conditional, shadow, manifest=manifest,
    )


def score_monthly_upstream_bundle(
    bundle: MonthlyUpstreamBundle,
    frame: pd.DataFrame,
    *,
    allow_prior_reference: bool = False,
    prior_reference_start: Any | None = None,
) -> pd.DataFrame:
    _require_columns(
        frame,
        ["candidate_id", "__decision_ts__", "side_name", *bundle.base_fields],
        "monthly upstream scoring",
    )
    assert_scoring_frame_is_target_free(frame)
    output = frame[["candidate_id", "__decision_ts__", "side_name"]].copy()
    timestamp = pd.to_datetime(frame["__decision_ts__"], utc=True)
    if prior_reference_start is not None and not allow_prior_reference:
        raise ValueError(
            "prior_reference_start requires allow_prior_reference=True"
        )
    lower = (
        _utc(prior_reference_start)
        if prior_reference_start is not None
        else (
            bundle.cutoff - pd.Timedelta(days=REFERENCE_DAYS)
            if allow_prior_reference
            else bundle.cutoff
        )
    )
    if allow_prior_reference and lower >= bundle.cutoff:
        raise ValueError("monthly upstream prior reference must begin before its cutoff")
    if not timestamp.between(lower, bundle.end_exclusive, inclusive="left").all():
        role = (
            "same-bundle prior reference/held frame"
            if allow_prior_reference else "held month"
        )
        raise ValueError(f"monthly upstream scorer received rows outside its {role}")
    probability = bundle.base_model.predict_proba(
        _numeric_matrix(frame, bundle.base_fields, bundle.base_medians),
    )
    lookup = {int(label): index for index, label in enumerate(bundle.base_model.classes_)}
    output["p_adverse"] = probability[:, lookup[0]]
    output["p_weak"] = probability[:, lookup[1]]
    output["p_clear"] = probability[:, lookup[2]]
    output["base_score"] = output["p_clear"] - 0.5 * output["p_adverse"]
    output["base_rank42"] = bundle.base_score_reference.cdf(output["base_score"])
    output["base_anchor_bps"] = bundle.policy_net_map.predict(output["base_rank42"])
    ranks: list[np.ndarray] = []
    for head in bundle.conditional_heads:
        raw, rank = head.predict_rank(frame)
        output[f"conditional_head__{head.spec.name}__raw"] = raw
        output[f"conditional_head__{head.spec.name}__rank"] = rank
        ranks.append(rank)
    output["conditional_consensus_rank"] = np.nanmedian(np.column_stack(ranks), axis=1)
    output["upstream"] = (
        BASE_BLEND_WEIGHT * output["base_rank42"]
        + CONSENSUS_BLEND_WEIGHT * output["conditional_consensus_rank"]
    )
    shadow_ranks: list[np.ndarray] = []
    for head in bundle.ordinary_shadow_heads:
        raw, rank = head.predict_rank(frame)
        output[f"ordinary_shadow_head__{head.spec.name}__raw"] = raw
        output[f"ordinary_shadow_head__{head.spec.name}__rank"] = rank
        shadow_ranks.append(rank)
    output["ordinary_shadow_consensus_rank"] = np.nanmedian(
        np.column_stack(shadow_ranks), axis=1,
    )
    output["ordinary_shadow_upstream"] = (
        BASE_BLEND_WEIGHT * output["base_rank42"]
        + CONSENSUS_BLEND_WEIGHT * output["ordinary_shadow_consensus_rank"]
    )
    output["upstream_bundle_sha256"] = bundle.manifest.get("bundle_sha256", "unpersisted")
    return output


def train_four_week_conversion_bundle(
    *,
    cutoff: Any,
    upstream_ledger: pd.DataFrame,
    frozen_geometry: FrozenGeometryK9,
    base_fields: Sequence[str],
    source_hashes: Mapping[str, str] | None = None,
    calibration_reserve_days: int = 0,
) -> FourWeekConversionBundle:
    cutoff_ts = _utc(cutoff)
    end = cutoff_ts + pd.Timedelta(days=FOUR_WEEK_DAYS)
    if not 0 <= int(calibration_reserve_days) <= REFERENCE_DAYS:
        raise ValueError(
            "four-week conversion calibration reserve must lie between zero and "
            "the same-model reference horizon",
        )
    calibration_reserve_start = cutoff_ts - pd.Timedelta(
        days=int(calibration_reserve_days),
    )
    fields = tuple(str(field) for field in base_fields)
    train_start = _month_add(cutoff_ts, -META_TRAIN_MONTHS)
    required = [
        "candidate_id", "__decision_ts__", "side_name", "r3_class",
        "r3_label_available_ts", "policy_net_bps", "policy_label_available_ts",
        "h12_label_valid", "h12_label_available_ts", "h12_tp6_sl4_net_bps",
        "base_score", "base_rank42", "base_anchor_bps",
        "conditional_consensus_rank", "upstream", "stack_is_prequential", *fields,
    ]
    _require_columns(upstream_ledger, required, "four-week conversion ledger")
    timestamp_columns = (
        "__decision_ts__", "r3_label_available_ts", "policy_label_available_ts",
        "h12_label_available_ts",
    )
    requires_timestamp_copy = any(
        not pd.api.types.is_datetime64tz_dtype(upstream_ledger[column])
        for column in timestamp_columns
    )
    ledger = upstream_ledger.copy() if requires_timestamp_copy else upstream_ledger
    if requires_timestamp_copy:
        for column in timestamp_columns:
            ledger[column] = pd.to_datetime(ledger[column], utc=True, errors="raise")
    if (ledger["__decision_ts__"] >= cutoff_ts).any():
        raise ValueError("conversion training ledger crosses the held cutoff")
    if not ledger["stack_is_prequential"].fillna(False).astype(bool).all():
        raise ValueError("conversion training requires prequential upstream scores")
    # Never re-fit geometry/K9 at a conversion boundary.  The parent bundle is
    # one target-free October--December 2024 representation and is embedded in
    # every downstream conversion model through this immutable view.
    geometry = FrozenGeometryK9View(frozen_geometry)
    meta = ledger.loc[
        ledger["__decision_ts__"].ge(train_start)
        & ledger["__decision_ts__"].lt(calibration_reserve_start)
        & ledger["policy_label_available_ts"].lt(calibration_reserve_start)
        & np.isfinite(pd.to_numeric(ledger["policy_net_bps"], errors="coerce"))
    ].copy()
    leaf = _fit_leaf_trust(
        ledger.loc[
            ledger["__decision_ts__"].ge(train_start)
            & ledger["__decision_ts__"].lt(calibration_reserve_start)
        ].copy(), fields, calibration_reserve_start,
    )
    # Dynamic K9 support/OOD is a market-universe state, not a supervised
    # meta-row state.  Materialise it across the complete target-free ledger
    # span first, in chronological order, then select the label-resolved rows
    # used to fit the conversion targets.  This keeps all future policy paths
    # and label-validity out of the representation itself.
    geometry_population = ledger.loc[
        ledger["__decision_ts__"].ge(train_start)
        & ledger["__decision_ts__"].lt(calibration_reserve_start)
    ].copy()
    geometry_population, geometry_state = _transform_frozen_geometry_causally(
        geometry, geometry_population,
    )
    geometry_state = _canonical_ldf_geometry_aliases(geometry_state)
    geometry_state["candidate_id"] = geometry_population["candidate_id"].to_numpy()
    if geometry_state["candidate_id"].duplicated().any():
        raise AssertionError("causal geometry population duplicated candidate identities")
    meta_geometry = meta.loc[:, ["candidate_id"]].merge(
        geometry_state,
        on="candidate_id", how="left", validate="one_to_one",
    ).drop(columns="candidate_id")
    if meta_geometry.isna().all(axis=1).any():
        raise ValueError("causal geometry state does not cover every conversion meta row")
    state = pd.concat(
        [meta_geometry.reset_index(drop=True), leaf.transform(meta).reset_index(drop=True)],
        axis=1,
    )
    aggregate = _aggregate_state_fields(state)
    meta = meta.reset_index(drop=True)
    state = state.reset_index(drop=True)
    meta = pd.concat([meta, state.loc[:, list(aggregate)]], axis=1)
    correctness_fields = (
        "base_score", "base_anchor_bps", "base_rank42",
        "conditional_consensus_rank", "upstream", *aggregate,
    )
    meta_fit = _equal_month_sample(meta, MODEL_CAP, seed=SEED + 5001)
    correctness = _fit_correctness(meta_fit, correctness_fields)
    severe = _fit_severe_diagnostic(
        meta_fit, correctness_fields, cutoff=calibration_reserve_start,
    )
    manifest = {
        "schema": CONVERSION_BUNDLE_SCHEMA,
        "cutoff": cutoff_ts.isoformat(),
        "end_exclusive": end.isoformat(),
        "side": SIDE,
        "training_start": train_start.isoformat(),
        "geometry_refit_cadence": "never",
        "geometry_definition_start": geometry.definition_start,
        "geometry_definition_end_exclusive": geometry.definition_end_exclusive,
        "geometry_parent_bundle_sha256": geometry.parent_bundle_sha256,
        "geometry_bundle_sha256": geometry.bundle_sha256,
        "geometry_temperature_scale": geometry.temperature_scale,
        "geometry_effective_temperature": (
            float(geometry.parent.temperature) * geometry.temperature_scale
        ),
        "geometry_dynamic_history_training": (
            "complete target-free conversion window, chronologically advanced; "
            "seed restricted strictly before that window"
        ),
        "latest_fit_rule_reliability": {
            "enabled": True,
            "semantic_contract": (
                "contribution-weighted support/OOD/Mahalanobis/covariance/correlation "
                "of active current-model feature paths versus its strict pre-cutoff "
                "training distribution; raw leaf IDs are never emitted"
            ),
            "fit_cutoff": calibration_reserve_start.isoformat(),
            "training_rows": int(leaf.train_rows),
            "input_fields": list(leaf.fields),
            "input_fields_sha256": _json_hash(list(leaf.fields)),
        },
        "correctness_target": "selected-policy net bps - base anchor > +100",
        "correctness_query": "4-hour UTC x side",
        "correctness_training_fraction": correctness.training_fraction,
        "correctness_training_score_floor": correctness.training_score_floor,
        "correctness_gate_domain": "pooled-global training upstream score only",
        "raw_k9_used_by_correctness": False,
        "severe_target": severe.target,
        "geometry_definition_rows_excluded_from_severe": True,
        "geometry_definition_exclusion_scope": (
            "Severe-200 only; frozen geometry is target-free and correctness may "
            "use its aggregate OOS transform"
        ),
        "severe_role": "shadow_diagnostic_only",
        "severe_affects_final_score": False,
        "normalization": "same-conversion-model prior-28-day CDF",
        "calibration_reserve_days": int(calibration_reserve_days),
        "calibration_reserve_start": calibration_reserve_start.isoformat(),
        "calibration_reserve_contract": (
            "preceding target-free reference rows excluded from all conversion "
            "supervised leaf-trust/correctness/Severe fits"
            if calibration_reserve_days else "disabled_legacy_contract"
        ),
        "source_hashes": dict(source_hashes or {}),
    }
    return FourWeekConversionBundle(
        cutoff_ts, end, fields, geometry, leaf, correctness, severe, manifest=manifest,
    )


def score_four_week_conversion_bundle(
    bundle: FourWeekConversionBundle,
    *,
    reference: pd.DataFrame,
    held: pd.DataFrame,
    precomputed_state: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    upstream_fields = [
        "base_score", "base_rank42", "base_anchor_bps",
        "conditional_consensus_rank", "upstream",
        "ordinary_shadow_consensus_rank", "ordinary_shadow_upstream",
    ]
    for role, frame in (("reference", reference), ("held", held)):
        _require_columns(
            frame,
            ["candidate_id", "__decision_ts__", "side_name", *upstream_fields, *bundle.base_fields],
            f"four-week conversion {role}",
        )
        if frame["candidate_id"].duplicated().any():
            raise ValueError(f"four-week conversion {role} has duplicate identities")
        assert_scoring_frame_is_target_free(frame)
    reference = reference.copy()
    held = held.copy()
    reference["__decision_ts__"] = pd.to_datetime(reference["__decision_ts__"], utc=True)
    held["__decision_ts__"] = pd.to_datetime(held["__decision_ts__"], utc=True)
    if not reference["__decision_ts__"].between(
        bundle.cutoff - pd.Timedelta(days=REFERENCE_DAYS), bundle.cutoff, inclusive="left",
    ).all():
        raise ValueError("conversion reference is not the preceding 28 days")
    if not held["__decision_ts__"].between(
        bundle.cutoff, bundle.end_exclusive, inclusive="left",
    ).all():
        raise ValueError("conversion held rows fall outside the four-week block")
    combined = pd.concat(
        [reference.assign(__score_role__="reference"), held.assign(__score_role__="held")],
        ignore_index=True,
    ).sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if precomputed_state is None:
        state = _canonical_ldf_geometry_aliases(
            pd.concat(
                [
                    bundle.geometry.transform(combined),
                    bundle.leaf_trust.transform(combined),
                ],
                axis=1,
            ),
        )
        aggregate = _aggregate_state_fields(state)
        combined = pd.concat(
            [combined, state.loc[:, list(aggregate)].reset_index(drop=True)], axis=1,
        )
    else:
        state = precomputed_state.copy()
        state_keys = ["candidate_id", "__decision_ts__", "side_name"]
        _require_columns(state, state_keys, "precomputed conversion state")
        if state["candidate_id"].duplicated().any():
            raise ValueError("precomputed conversion state has duplicate candidate IDs")
        state["__decision_ts__"] = pd.to_datetime(
            state["__decision_ts__"], utc=True, errors="raise",
        )
        aggregate = tuple(column for column in state.columns if column not in state_keys)
        if not aggregate:
            raise ValueError("precomputed conversion state has no state features")
        combined = combined.merge(
            state.loc[:, [*state_keys, *aggregate]],
            on=state_keys,
            how="left",
            validate="one_to_one",
        )
        if combined.loc[:, list(aggregate)].isna().all(axis=1).any():
            raise ValueError("precomputed conversion state does not cover every score row")
    raw = bundle.correctness.model.predict(
        _numeric_matrix(combined, bundle.correctness.fields, bundle.correctness.medians),
    )
    combined["correctness_raw"] = raw
    combined["correctness_rank"] = bundle.correctness.score_reference.cdf(raw)
    combined["correctness_gate_active"] = combined["upstream"].ge(
        bundle.correctness.training_score_floor,
    )
    active_multiplier = (
        CORRECTNESS_FLOOR + CORRECTNESS_SPAN * combined["correctness_rank"].to_numpy(float)
    )
    correctness_multiplier = np.where(
        combined["correctness_gate_active"].to_numpy(bool),
        active_multiplier,
        1.0,
    )
    combined["raw_correctness_demote"] = (
        combined["upstream"].to_numpy(float) * correctness_multiplier
    )
    reference_mask = combined["__score_role__"].eq("reference").to_numpy()
    final_reference = ScoreReference.fit(
        combined.loc[reference_mask, "raw_correctness_demote"],
        source="same_conversion_model_prior28",
    )
    combined["final_score"] = final_reference.cdf(combined["raw_correctness_demote"])
    severe = bundle.severe_diagnostic
    combined["severe200_probability_shadow"] = (
        np.nan
        if severe.model is None
        else severe.model.predict_proba(_numeric_matrix(combined, severe.fields, severe.medians))[:, 1]
    )
    combined["severe_affects_final_score"] = False
    combined["conversion_bundle_sha256"] = bundle.manifest.get("bundle_sha256", "unpersisted")
    combined["geometry_bundle_sha256"] = bundle.geometry.bundle_sha256
    combined["ev_score_family_id"] = _current_ev_score_family_id(
        bundle.geometry.bundle_sha256,
    )
    audit = pd.DataFrame([{
        "schema": CONVERSION_BUNDLE_SCHEMA,
        "cutoff": bundle.cutoff,
        "reference_rows": int(reference_mask.sum()),
        "held_rows": int((~reference_mask).sum()),
        "same_conversion_model_reference_and_held": True,
        "upstream_scores_are_prequential_monthly": True,
        "held_percentile_operations": 0,
        "raw_k9_in_correctness": False,
        "correctness_training_fraction": bundle.correctness.training_fraction,
        "correctness_training_score_floor": bundle.correctness.training_score_floor,
        "k9_temperature_scale": bundle.geometry.temperature_scale,
        "geometry_refit_cadence": "never",
        "geometry_parent_bundle_sha256": bundle.geometry.parent_bundle_sha256,
        "severe_affects_final_score": False,
        "final_reference": final_reference.source,
        "ev_score_family_id": _current_ev_score_family_id(
            bundle.geometry.bundle_sha256,
        ),
    }])
    columns = [
        "candidate_id", "__decision_ts__",
        *(["__symbol__"] if "__symbol__" in combined.columns else []),
        "side_name", *upstream_fields,
        "correctness_raw", "correctness_rank", "correctness_gate_active",
        "raw_correctness_demote", "final_score",
        "severe200_probability_shadow", "severe_affects_final_score",
        "conversion_bundle_sha256", "geometry_bundle_sha256", "ev_score_family_id",
        *( ["upstream_bundle_sha256"] if "upstream_bundle_sha256" in combined.columns else []),
        "__score_role__",
        # Persist the same target-free, bundle-invariant state fields as the
        # one-shot scorer.  The 45-field incumbent LDF uses these summaries;
        # no raw K9 memberships leave the conversion scorer.
        *aggregate,
    ]
    return combined.loc[:, columns].copy(), audit


def _conversion_timestamp_chunks(frame: pd.DataFrame, *, hours: int):
    """Yield complete decision-hour slices without splitting a cross-section."""
    timestamps = pd.Index(
        pd.to_datetime(frame["__decision_ts__"], utc=True).unique(),
    ).sort_values()
    for start in range(0, len(timestamps), hours):
        values = timestamps[start:start + hours]
        yield frame.loc[frame["__decision_ts__"].isin(values)].copy()


def _initial_frozen_geometry_history(
    geometry: FrozenGeometryK9View,
    *,
    first_timestamp: Any,
) -> pd.DataFrame:
    """Return only geometry state available strictly before a score span.

    The K9 encoder/KMeans/support definitions are frozen, but the dynamic
    path-support and drift fields are chronological market state.  A geometry
    bundle fitted on Oct--Dec 2024 therefore cannot seed, for example, a
    November 2024 replay with its complete Dec history.  Keep only the
    target-free aggregate membership states that precede the span's first
    decision hour; the scorer then advances that state hour by hour.
    """
    first = _utc(first_timestamp)
    history = geometry.parent.state_history.copy()
    if "__decision_ts__" not in history:
        raise ValueError("frozen geometry state history lacks decision timestamps")
    history["__decision_ts__"] = pd.to_datetime(
        history["__decision_ts__"], utc=True, errors="raise",
    )
    history = history.loc[history["__decision_ts__"].lt(first)].copy()
    history = history.sort_values("__decision_ts__", kind="stable")
    if history["__decision_ts__"].duplicated().any():
        raise ValueError("frozen geometry state history has duplicate timestamps")
    return history.reset_index(drop=True)


def _transform_frozen_geometry_causally(
    geometry: FrozenGeometryK9View,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transform a complete target-free span with strict prior-only state.

    ``FrozenGeometryK9View`` keeps membership semantics fixed.  This wrapper
    only controls the dynamic-history seed and chronological ordering.  It is
    used for conversion training as well as replay scoring so a label-valid
    subset can never define the K9 support/OOD state seen by the meta layer.
    """
    if frame.empty:
        raise ValueError("causal frozen-geometry transform received an empty frame")
    ordered = frame.copy()
    ordered["__decision_ts__"] = pd.to_datetime(
        ordered["__decision_ts__"], utc=True, errors="raise",
    )
    if ordered["candidate_id"].duplicated().any():
        raise ValueError("causal frozen-geometry transform has duplicate candidate IDs")
    ordered = ordered.sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    ).reset_index(drop=True)
    history = _initial_frozen_geometry_history(
        geometry, first_timestamp=ordered["__decision_ts__"].iloc[0],
    )
    parent = copy.copy(geometry.parent)
    parent.state_history = history
    view = FrozenGeometryK9View(
        parent=parent, temperature_scale=geometry.temperature_scale,
    )
    state = view.transform(ordered).reset_index(drop=True)
    return ordered, state


def _append_geometry_history(
    history: pd.DataFrame,
    *,
    frame: pd.DataFrame,
    geometry_state: pd.DataFrame,
) -> pd.DataFrame:
    """Advance only target-free K9 history after a complete score slice."""
    membership_columns = [
        f"k09__cluster_{index:02d}__membership" for index in range(K9_CLUSTERS)
    ]
    values = geometry_state.loc[:, membership_columns].copy()
    values.columns = [f"k{index}" for index in range(K9_CLUSTERS)]
    values["__decision_ts__"] = pd.to_datetime(
        frame["__decision_ts__"], utc=True,
    ).to_numpy()
    event = values.groupby("__decision_ts__", sort=True)[
        [f"k{index}" for index in range(K9_CLUSTERS)]
    ].sum().reset_index()
    output = pd.concat([history, event], ignore_index=True).sort_values(
        "__decision_ts__", kind="stable",
    )
    if output["__decision_ts__"].duplicated().any():
        raise AssertionError("lock-step geometry history repeated a scored timestamp")
    return output.reset_index(drop=True)


def _score_four_week_conversion_piece(
    bundle: FourWeekConversionBundle,
    *,
    frame: pd.DataFrame,
    role: str,
    geometry_history: pd.DataFrame,
    final_reference: ScoreReference | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score a complete-hour piece with the persisted, causal K9 history.

    The frozen K9 encoder itself is never re-fit.  Copying its parent only
    gives the current score slice the immediately preceding target-free state,
    preventing a batch boundary from resetting support/OOD semantics.
    """
    local_parent = copy.copy(bundle.geometry.parent)
    local_parent.state_history = geometry_history
    local_geometry = FrozenGeometryK9View(
        parent=local_parent,
        temperature_scale=bundle.geometry.temperature_scale,
    )
    geometry_state = local_geometry.transform(frame)
    next_history = _append_geometry_history(
        geometry_history, frame=frame, geometry_state=geometry_state,
    )
    state = _canonical_ldf_geometry_aliases(
        pd.concat([geometry_state, bundle.leaf_trust.transform(frame)], axis=1),
    )
    aggregate = _aggregate_state_fields(state)
    combined = pd.concat(
        [frame.reset_index(drop=True), state.loc[:, list(aggregate)].reset_index(drop=True)],
        axis=1,
    )
    raw = bundle.correctness.model.predict(
        _numeric_matrix(combined, bundle.correctness.fields, bundle.correctness.medians),
    )
    combined["correctness_raw"] = raw
    combined["correctness_rank"] = bundle.correctness.score_reference.cdf(raw)
    combined["correctness_gate_active"] = combined["upstream"].ge(
        bundle.correctness.training_score_floor,
    )
    multiplier = CORRECTNESS_FLOOR + CORRECTNESS_SPAN * combined[
        "correctness_rank"
    ].to_numpy(float)
    combined["raw_correctness_demote"] = combined["upstream"].to_numpy(float) * np.where(
        combined["correctness_gate_active"].to_numpy(bool), multiplier, 1.0,
    )
    combined["final_score"] = (
        np.nan if final_reference is None
        else final_reference.cdf(combined["raw_correctness_demote"])
    )
    severe = bundle.severe_diagnostic
    combined["severe200_probability_shadow"] = (
        np.nan if severe.model is None else severe.model.predict_proba(
            _numeric_matrix(combined, severe.fields, severe.medians),
        )[:, 1]
    )
    combined["severe_affects_final_score"] = False
    combined["conversion_bundle_sha256"] = bundle.manifest.get("bundle_sha256", "unpersisted")
    combined["geometry_bundle_sha256"] = bundle.geometry.bundle_sha256
    combined["ev_score_family_id"] = _current_ev_score_family_id(
        bundle.geometry.bundle_sha256,
    )
    upstream_fields = [
        "base_score", "base_rank42", "base_anchor_bps", "conditional_consensus_rank", "upstream",
        "ordinary_shadow_consensus_rank", "ordinary_shadow_upstream",
    ]
    columns = [
        "candidate_id", "__decision_ts__", *( ["__symbol__"] if "__symbol__" in combined else []),
        "side_name", *upstream_fields, "correctness_raw", "correctness_rank",
        "correctness_gate_active", "raw_correctness_demote", "final_score",
        "severe200_probability_shadow", "severe_affects_final_score",
        "conversion_bundle_sha256", "geometry_bundle_sha256", "ev_score_family_id",
        *( ["upstream_bundle_sha256"] if "upstream_bundle_sha256" in combined else []),
        *aggregate,
    ]
    output = combined.loc[:, columns].copy()
    output["__score_role__"] = role
    return output, next_history


def score_four_week_conversion_bundle_lockstep(
    bundle: FourWeekConversionBundle,
    *,
    reference: pd.DataFrame,
    held: pd.DataFrame,
    chunk_hours: int = 72,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score one exact lock-step producer using causal frozen-K9 history.

    This is the inference counterpart of the lock-step walk-forward producer:
    the exact same persisted conversion bundle scores its preceding 28-day
    target-free reserve and its held/live rows.  It is intentionally distinct
    from the staggered monthly-upstream helper because it has one upstream
    producer and one conversion producer for the complete block.
    """
    if chunk_hours < 1:
        raise ValueError("lock-step conversion score chunk must contain at least one hour")
    upstream_fields = [
        "base_score", "base_rank42", "base_anchor_bps",
        "conditional_consensus_rank", "upstream",
        "ordinary_shadow_consensus_rank", "ordinary_shadow_upstream",
    ]
    for role, frame in (("reference", reference), ("held", held)):
        _require_columns(
            frame,
            ["candidate_id", "__decision_ts__", "side_name", *upstream_fields, *bundle.base_fields],
            f"lock-step conversion {role}",
        )
        if frame["candidate_id"].duplicated().any():
            raise ValueError(f"lock-step conversion {role} has duplicate identities")
        assert_scoring_frame_is_target_free(frame)
    reference = reference.copy()
    held = held.copy()
    reference["__decision_ts__"] = pd.to_datetime(reference["__decision_ts__"], utc=True)
    held["__decision_ts__"] = pd.to_datetime(held["__decision_ts__"], utc=True)
    if not reference["__decision_ts__"].between(
        bundle.cutoff - pd.Timedelta(days=REFERENCE_DAYS), bundle.cutoff, inclusive="left",
    ).all():
        raise ValueError("lock-step conversion reference is not the preceding 28 days")
    if not held["__decision_ts__"].between(
        bundle.cutoff, bundle.end_exclusive, inclusive="left",
    ).all():
        raise ValueError("lock-step conversion held rows fall outside the bundle window")
    history = _initial_frozen_geometry_history(
        bundle.geometry,
        first_timestamp=reference["__decision_ts__"].min(),
    )
    reference_parts: list[pd.DataFrame] = []
    for piece in _conversion_timestamp_chunks(reference, hours=chunk_hours):
        scored, history = _score_four_week_conversion_piece(
            bundle, frame=piece, role="reference", geometry_history=history,
            final_reference=None,
        )
        reference_parts.append(scored)
    scored_reference = pd.concat(reference_parts, ignore_index=True).sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    ).reset_index(drop=True)
    final_reference = ScoreReference.fit(
        scored_reference["raw_correctness_demote"],
        source="same_conversion_model_prior28",
    )
    scored_reference["final_score"] = final_reference.cdf(
        scored_reference["raw_correctness_demote"],
    )
    held_parts: list[pd.DataFrame] = []
    for piece in _conversion_timestamp_chunks(held, hours=chunk_hours):
        scored, history = _score_four_week_conversion_piece(
            bundle, frame=piece, role="held", geometry_history=history,
            final_reference=final_reference,
        )
        held_parts.append(scored)
    scored_held = pd.concat(held_parts, ignore_index=True).sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    ).reset_index(drop=True)
    output = pd.concat([scored_reference, scored_held], ignore_index=True)
    audit = pd.DataFrame([{
        "schema": CONVERSION_BUNDLE_SCHEMA,
        "cutoff": bundle.cutoff,
        "reference_rows": int(len(scored_reference)),
        "held_rows": int(len(scored_held)),
        "score_chunk_hours": int(chunk_hours),
        "same_conversion_model_reference_and_held": True,
        "same_upstream_bundle_reference_and_held": True,
        "upstream_scores_are_prequential_lockstep": True,
        "held_percentile_operations": 0,
        "raw_k9_in_correctness": False,
        "geometry_refit_cadence": "never",
        "geometry_parent_bundle_sha256": bundle.geometry.parent_bundle_sha256,
        "geometry_dynamic_history": (
            "frozen target-free history strictly before reference; then "
            "complete-universe chronological score chunks"
        ),
        "final_reference": final_reference.source,
        "memory_bound_complete_hour_chunks": True,
        "severe_affects_final_score": False,
    }])
    return output, audit


def _precompute_conversion_state(
    bundle: FourWeekConversionBundle,
    *,
    reference: pd.DataFrame,
    held: pd.DataFrame,
) -> pd.DataFrame:
    """Generate causal geometry/leaf state once over a contiguous score span.

    Dynamic K9 support/drift uses preceding decision timestamps.  When one
    conversion block crosses an upstream monthly refit, this state must remain
    continuous: the new upstream producer can use previous *market state*,
    even though it must not borrow the previous producer's score-to-outcome
    calibration.  The returned table is target-free and contains no raw K9
    memberships.
    """
    keys = ["candidate_id", "__decision_ts__", "side_name"]
    full = pd.concat([reference, held], ignore_index=True)
    if full["candidate_id"].duplicated().any():
        raise ValueError("conversion state frame has duplicate candidate IDs")
    full["__decision_ts__"] = pd.to_datetime(full["__decision_ts__"], utc=True, errors="raise")
    full = full.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    state = _canonical_ldf_geometry_aliases(
        pd.concat(
            [bundle.geometry.transform(full), bundle.leaf_trust.transform(full)], axis=1,
        ),
    )
    aggregate = _aggregate_state_fields(state)
    if any(column.startswith("k09__cluster_") for column in aggregate):
        raise AssertionError("raw K9 membership leaked into precomputed conversion state")
    return pd.concat([full.loc[:, keys], state.loc[:, list(aggregate)]], axis=1)


def score_four_week_conversion_by_upstream_vintage(
    bundle: FourWeekConversionBundle,
    *,
    reference: pd.DataFrame,
    held: pd.DataFrame,
    upstream_bundles: Mapping[str, MonthlyUpstreamBundle],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score each monthly upstream producer in its own prior-28 CDF domain.

    The four-week conversion model is shared across its block, but the monthly
    base/consensus producer may change inside that block.  A CDF reference
    made by a different upstream bundle is not in the same score domain even
    when both outputs lie in ``[0, 1]``.  For every held upstream vintage this
    helper therefore rescales the bounded conversion reference with that exact
    upstream model before calling the conversion scorer.

    Reference rows are duplicated only as score-domain provenance when a
    four-week block spans monthly upstream vintages.  Held candidate IDs remain
    unique.  The extended prior scoring bound is explicit and still target
    free: the conversion bundle fixes the reference start to its preceding 28
    calendar days, and no outcomes are read during this operation.
    """
    reference = reference.copy()
    held = held.copy()
    for name, frame in (("reference", reference), ("held", held)):
        frame["__decision_ts__"] = pd.to_datetime(
            frame["__decision_ts__"], utc=True, errors="raise",
        )
        assert_scoring_frame_is_target_free(frame)
        if frame["candidate_id"].duplicated().any():
            raise ValueError(f"upstream-vintage {name} frame has duplicate candidates")
    reference_start = pd.Timestamp(bundle.cutoff) - pd.Timedelta(days=REFERENCE_DAYS)
    precomputed_state = _precompute_conversion_state(
        bundle, reference=reference, held=held,
    )
    outputs: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    for month, held_block in held.groupby(
        held["__decision_ts__"].dt.strftime("%Y-%m"), sort=True,
    ):
        upstream = upstream_bundles.get(str(month))
        if upstream is None:
            raise KeyError(f"missing monthly upstream bundle for held month {month}")
        upstream_hash = str(upstream.manifest.get("bundle_sha256", "unpersisted"))
        held_score = score_monthly_upstream_bundle(upstream, held_block)
        reference_score = score_monthly_upstream_bundle(
            upstream,
            reference,
            allow_prior_reference=True,
            prior_reference_start=reference_start,
        )
        held_input = held_block.merge(
            held_score,
            on=["candidate_id", "__decision_ts__", "side_name"],
            validate="one_to_one",
        )
        reference_input = reference.merge(
            reference_score,
            on=["candidate_id", "__decision_ts__", "side_name"],
            validate="one_to_one",
        )
        scored, audit = score_four_week_conversion_bundle(
            bundle,
            reference=reference_input,
            held=held_input,
            precomputed_state=precomputed_state,
        )
        scored["cdf_reference_upstream_bundle_sha256"] = upstream_hash
        upstream_cutoff = _utc(getattr(upstream, "cutoff", bundle.cutoff))
        upstream_reserve_start = _utc(
            upstream.manifest.get("calibration_reserve_start", upstream_cutoff),
        )
        conversion_reserve_start = _utc(
            getattr(bundle, "manifest", {}).get(
                "calibration_reserve_start", bundle.cutoff,
            ),
        )
        calibration_reserve_start = max(
            upstream_reserve_start, conversion_reserve_start,
        )
        calibration_activation = max(upstream_cutoff, _utc(bundle.cutoff))
        role_reference = scored["__score_role__"].eq("reference")
        score_ts = pd.to_datetime(scored["__decision_ts__"], utc=True, errors="raise")
        # This flag is computed without outcomes.  Once exact policy labels are
        # joined after scoring, only rows whose label timestamp precedes the
        # declared activation can enter the immediate producer calibration.
        # It is deliberately stricter than a generic prior-28 CDF reference:
        # every retained row was excluded from both the active upstream and
        # conversion supervised fits.
        scored["calibration_reserve_start"] = calibration_reserve_start
        scored["calibration_activation_ts"] = calibration_activation
        scored["calibration_reference_oos_to_all_active_fits"] = (
            role_reference
            & score_ts.ge(calibration_reserve_start)
            & score_ts.lt(calibration_activation)
        )
        audit = audit.copy()
        audit["cdf_reference_upstream_bundle_sha256"] = upstream_hash
        audit["upstream_bundle_month"] = str(month)
        audit["same_upstream_bundle_for_reference_and_held"] = True
        audit["calibration_reserve_start"] = calibration_reserve_start
        audit["calibration_activation_ts"] = calibration_activation
        audit["calibration_reference_oos_contract"] = (
            "reference candidate decision >= max(active component reserve starts) "
            "and < active producer activation; labels joined after scoring"
        )
        outputs.append(scored)
        audits.append(audit)
    if not outputs:
        raise ValueError("upstream-vintage conversion scorer received no held candidates")
    output = pd.concat(outputs, ignore_index=True).sort_values(
        ["__score_role__", "__decision_ts__", "candidate_id"], kind="stable",
    )
    held_output = output.loc[output["__score_role__"].eq("held")]
    if held_output["candidate_id"].duplicated().any():
        raise AssertionError("upstream-vintage conversion scorer duplicated a held candidate")
    return output, pd.concat(audits, ignore_index=True)


def _persist_split_bundle(bundle: Any, directory: Path, schema: str, filename: str) -> dict[str, Any]:
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"immutable {schema} directory already exists: {directory}")
    directory.mkdir(parents=True)
    path = directory / filename
    joblib.dump(bundle, path, compress=3)
    digest = _file_hash(path)
    manifest = {**bundle.manifest, "schema": schema, "bundle_file": filename, "bundle_sha256": digest}
    (directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    bundle.manifest["bundle_sha256"] = digest
    return manifest


def persist_monthly_upstream_bundle(bundle: MonthlyUpstreamBundle, directory: Path) -> dict[str, Any]:
    return _persist_split_bundle(bundle, directory, UPSTREAM_BUNDLE_SCHEMA, "monthly_upstream_bundle.joblib")


def persist_four_week_conversion_bundle(bundle: FourWeekConversionBundle, directory: Path) -> dict[str, Any]:
    return _persist_split_bundle(bundle, directory, CONVERSION_BUNDLE_SCHEMA, "four_week_conversion_bundle.joblib")


def _load_split_bundle(directory: Path, schema: str, expected_type: type) -> Any:
    directory = Path(directory)
    manifest = json.loads((directory / "run_manifest.json").read_text())
    if manifest.get("schema") != schema:
        raise ValueError(f"not a {schema} bundle")
    path = directory / manifest["bundle_file"]
    if _file_hash(path) != manifest["bundle_sha256"]:
        raise ValueError(f"{schema} bundle hash mismatch")
    bundle = joblib.load(path)
    if not isinstance(bundle, expected_type) or bundle.schema != schema:
        raise TypeError(f"{schema} payload has the wrong type/schema")
    bundle.__post_init__()
    if isinstance(bundle, FourWeekConversionBundle):
        if not np.isclose(
            float(manifest.get("correctness_training_fraction", np.nan)),
            CORRECTNESS_TRAIN_FRACTION,
        ):
            raise ValueError("conversion manifest has the wrong correctness curriculum")
        if not np.isclose(
            float(manifest.get("geometry_temperature_scale", np.nan)),
            K9_TEMPERATURE_SCALE,
        ):
            raise ValueError("conversion manifest has the wrong K9 temperature scale")
        if manifest.get("geometry_refit_cadence") != "never":
            raise ValueError("conversion manifest permits geometry/K9 re-fitting")
        if manifest.get("geometry_parent_bundle_sha256") != bundle.geometry.parent_bundle_sha256:
            raise ValueError("conversion manifest parent geometry identity mismatch")
    bundle.manifest["bundle_sha256"] = manifest["bundle_sha256"]
    return bundle


def load_monthly_upstream_bundle(directory: Path) -> MonthlyUpstreamBundle:
    return _load_split_bundle(directory, UPSTREAM_BUNDLE_SCHEMA, MonthlyUpstreamBundle)


def load_four_week_conversion_bundle(directory: Path) -> FourWeekConversionBundle:
    return _load_split_bundle(directory, CONVERSION_BUNDLE_SCHEMA, FourWeekConversionBundle)
