"""Strict chronological L/H/C leaf-reasoning meta-ablation engine.

This module is deliberately self contained: it consumes an already-issued,
same-side base OOF ledger and candidate *reasoning summaries* (never raw tree
leaf IDs).  It does not discover leaves, tune on a test period, or apply a
portfolio allocator.  Every prediction at decision time ``t`` is fitted only
on rows of that side whose labels resolved before ``t``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.linalg import cho_factor, cho_solve
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge

from extreme_price_movements.tp6_portability_data import FROZEN_META_CONTEXT


RAW_LEAF_ID_TOKENS = ("leaf_id", "leaf_token", "leaf_assignment", "raw_leaf")
L_ARMS = tuple(f"L{i}" for i in range(5))
H_ARMS = tuple(f"H{i}" for i in range(7))
C_ARMS = tuple(f"C{i}" for i in range(7))
S_ALLOWED = ("S0", "S1", "S2")
# This is a lineage-only input to the strict predecessor materialiser.  It is
# intentionally not an L/H/C arm bucket: the successor consumes the resulting
# predecessor OOF fields, never this source field directly.
S2_REASONING_ENTROPY_GROUP = "S2_reasoning_entropy"
S2_REASONING_ENTROPY_FIELD = "base_reasoning__family_contribution_entropy"
STAGES = ("L", "H", "C")
CLUSTER_THRESHOLD_BY_ARM = {
    "C1": 0.60,
    "C2": 0.70,
    "C3": 0.80,
    "C4": 0.90,
}
FROZEN_META_CONTROL_FEATURES = (
    "p_adverse", "p_weak", "p_clear", "base_expected_bps", *FROZEN_META_CONTEXT,
)
# An opaque candidate ID is not a primary key once strict base OOF data from
# several folds/transports is assembled.  These internal normalised columns
# retain the real sealed values when available and give legacy unit-ledgers a
# deterministic singleton namespace without weakening the strict path.
_STRICT_FOLD = "__strict_fold_id__"
_STRICT_TRANSPORT = "__strict_transport__"
_STRICT_PARTITION = "__strict_meta_partition__"
_H6_MDA_REPEATS = 3
# These bounds keep the selector's additional working set small beside the
# already-materialised train matrix.  They are implementation details only:
# every candidate and every declared phantom is still evaluated.
_H6_REAL_FEATURE_BLOCK = 16
_H6_PHANTOM_BLOCK = 8


class MetaFunnelError(ValueError):
    """Raised when a chronological meta-funnel contract is not satisfied."""


class MetaRegressor(Protocol):
    """Small injectable surface used by tests and fixed production models."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> Any: ...

    def predict(self, x: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class FrozenMetaModelSpec:
    """A fixed side-local residual learner; HPO is deliberately impossible here."""

    family: str
    params: Mapping[str, Any]
    contract_id: str

    def __post_init__(self) -> None:
        if self.family != "lightgbm_lgbmregressor":
            raise MetaFunnelError("the production meta control must be lightgbm_lgbmregressor")
        if not self.contract_id.strip():
            raise MetaFunnelError("frozen meta model contract_id must be non-empty")
        objective = str(dict(self.params).get("objective", "")).lower()
        if objective != "huber":
            raise MetaFunnelError("frozen meta model must explicitly declare objective='huber'")
        try:
            json.dumps(dict(self.params), sort_keys=True, default=str)
        except TypeError as exc:  # pragma: no cover - defensive contract check
            raise MetaFunnelError("frozen meta model parameters must be JSON serializable") from exc

    @property
    def params_hash(self) -> str:
        payload = json.dumps({"family": self.family, "contract_id": self.contract_id, "params": dict(self.params)}, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


MetaModelFactory = Callable[[FrozenMetaModelSpec], MetaRegressor]


@dataclass(frozen=True)
class MetaFunnelColumns:
    candidate_id: str = "candidate_id"
    side: str = "side_name"
    decision: str = "decision_ts"
    fold_id: str = "fold_id"
    label_available: str = "label_available_ts"
    base_expected_bps: str = "base_expected_bps"
    realized_gross_bps: str = "realized_gross_bps"
    realized_cost_bps: str = "realized_cost_bps"
    realized_net_bps: str = "realized_net_bps"
    base_fit_end: str = "base_oof_fit_end_ts"
    base_generated: str = "base_oof_generated_ts"
    base_strict_oof: str = "base_same_side_strict_oof"


@dataclass(frozen=True)
class NestedPredecessorOOFContract:
    """Lineage required before the S2 successor may consume predecessor OOFs."""

    feature_columns: tuple[str, ...]
    fit_end_column: str = "predecessor_oof_fit_end_ts"
    generated_column: str = "predecessor_oof_generated_ts"
    available_column: str = "predecessor_oof_available_ts"
    strict_oof_column: str = "predecessor_same_side_strict_oof"


@dataclass(frozen=True)
class ClusterTaxonomyContract:
    """Frozen development-only cluster-taxonomy metadata for C0--C6.

    The runner receives already-materialised *compact* cluster aggregates; it
    deliberately cannot cluster raw leaf identifiers.  This contract records
    the only permissible linkage/threshold grid and the pre-declared coverage
    choices which produced those aggregates.  ``cluster_ids_by_arm`` is not an
    inference input.  It is a validation handle that makes the soft/hard
    representation caps auditable before the C arms are evaluated.
    """

    linkage: str
    cluster_ids_by_arm: Mapping[str, Sequence[str]]
    threshold_by_arm: Mapping[str, float] = field(
        default_factory=lambda: dict(CLUSTER_THRESHOLD_BY_ARM)
    )
    c5_source_arm: str = "C1"
    c6_source_arm: str = "C5"
    top_decile_coverage_target: float = 0.95
    top_decile_coverage_by_arm: Mapping[str, float] = field(default_factory=dict)
    portable_top_decile_coverage_by_arm: Mapping[str, float] = field(default_factory=dict)
    production_soft_cap: int = 12
    exploratory_hard_cap: int = 20
    # C6 may only be called compact when its pre-declared development score is
    # within one standard error of the best C1--C5 cross-era score.
    c6_best_cross_era_score: float | None = None
    c6_best_cross_era_standard_error: float | None = None
    c6_compact_cross_era_score: float | None = None
    # ``threshold_sweep`` is the only permitted pre-selection C stage: C0 and
    # C1--C4 are evaluated before any arm may be named C5/C6.  ``final`` is
    # issued only by the post-C1--C4 selector with its immutable MDA evidence.
    selection_phase: str = "final"

    def __post_init__(self) -> None:
        linkage = str(self.linkage).lower()
        if linkage not in {"average", "complete"}:
            raise MetaFunnelError(
                "cluster taxonomy linkage must be 'average' or 'complete'; "
                "single linkage is forbidden"
            )
        expected = dict(CLUSTER_THRESHOLD_BY_ARM)
        supplied = {str(key): float(value) for key, value in dict(self.threshold_by_arm).items()}
        if supplied != expected:
            raise MetaFunnelError(
                "cluster taxonomy thresholds must be exactly "
                "C1=.60, C2=.70, C3=.80, C4=.90"
            )
        if not 0.0 < float(self.top_decile_coverage_target) <= 1.0:
            raise MetaFunnelError("top-decile coverage target must be in (0, 1]")
        if self.production_soft_cap < 1 or self.exploratory_hard_cap < self.production_soft_cap:
            raise MetaFunnelError("invalid cluster soft/hard representation caps")
        phase = str(self.selection_phase).lower()
        if phase not in {"threshold_sweep", "final"}:
            raise MetaFunnelError("cluster taxonomy selection_phase must be threshold_sweep or final")
        allowed_arms = set(CLUSTER_THRESHOLD_BY_ARM) if phase == "threshold_sweep" else set(C_ARMS)
        invalid = sorted(set(map(str, self.cluster_ids_by_arm)).difference(allowed_arms))
        if invalid:
            raise MetaFunnelError(f"cluster taxonomy has unknown arms: {invalid}")
        required = set(CLUSTER_THRESHOLD_BY_ARM)
        if phase == "final":
            required |= {"C5", "C6"}
        missing = sorted(required.difference(set(map(str, self.cluster_ids_by_arm))))
        if missing:
            raise MetaFunnelError(f"cluster taxonomy must declare cluster IDs for {missing}")
        ids = {
            str(arm): tuple(map(str, values))
            for arm, values in self.cluster_ids_by_arm.items()
        }
        if any(not value.strip() for values in ids.values() for value in values):
            raise MetaFunnelError("cluster IDs must be non-empty strings")
        if any(len(values) != len(set(values)) for values in ids.values()):
            raise MetaFunnelError("cluster IDs may not repeat within an arm")
        if phase == "final":
            if self.c5_source_arm not in CLUSTER_THRESHOLD_BY_ARM:
                raise MetaFunnelError("C5 must be selected from one of C1--C4")
            if self.c6_source_arm != "C5":
                raise MetaFunnelError("C6 must be a compact one-SE subset of C5")
            if not set(ids["C5"]).issubset(ids[self.c5_source_arm]):
                raise MetaFunnelError("C5 cluster IDs must be a subset of its selected threshold arm")
            if not set(ids["C6"]).issubset(ids["C5"]):
                raise MetaFunnelError("C6 cluster IDs must be a subset of C5")
            for arm in ("C5", "C6"):
                if len(ids[arm]) > self.exploratory_hard_cap:
                    raise MetaFunnelError(
                        f"{arm} exposes {len(ids[arm])} clusters, above the hard cap "
                        f"{self.exploratory_hard_cap}; aggregate into compact bundles"
                    )
        coverage = {str(key): float(value) for key, value in dict(self.top_decile_coverage_by_arm).items()}
        portable = {
            str(key): float(value)
            for key, value in dict(self.portable_top_decile_coverage_by_arm).items()
        }
        for name, values in (("top-decile", coverage), ("portable top-decile", portable)):
            if any(not 0.0 <= value <= 1.0 for value in values.values()):
                raise MetaFunnelError(f"{name} contribution coverage must be in [0, 1]")
        # C5 is the 95%-coverage manifest.  A lower portable coverage is
        # allowed and explicitly reported; it must never be back-filled with
        # unstable clusters merely to claim 95% coverage.
        if phase == "final":
            if "C5" in coverage and coverage["C5"] + 1e-12 < self.top_decile_coverage_target:
                raise MetaFunnelError("C5 must meet the declared top-decile contribution coverage target")
            one_se_values = (
                self.c6_best_cross_era_score,
                self.c6_best_cross_era_standard_error,
                self.c6_compact_cross_era_score,
            )
            if any(value is None for value in one_se_values):
                raise MetaFunnelError(
                    "C6 requires frozen development-only best score, standard error, "
                    "and compact score to prove the one-SE selection"
                )
            best_score, standard_error, compact_score = map(float, one_se_values)
            if not np.isfinite([best_score, standard_error, compact_score]).all() or standard_error < 0.0:
                raise MetaFunnelError("C6 one-SE evidence must be finite with a non-negative standard error")
            if compact_score + 1e-12 < best_score - standard_error:
                raise MetaFunnelError("C6 compact score is outside the one-SE development envelope")


@dataclass(frozen=True)
class MetaTransportGateConfig:
    """Pre-declared advancement gates; these do not select a final OOS arm."""

    required_transport_count: int = 2
    primary_tail_fractions: tuple[float, ...] = (0.05, 0.10)
    minimum_positive_environment_rate: float = 0.70
    max_worst_month_net_drop_bps: float | None = None

    def __post_init__(self) -> None:
        if self.required_transport_count < 1:
            raise MetaFunnelError("required_transport_count must be positive")
        allowed = {0.01, 0.05, 0.10}
        if not self.primary_tail_fractions or not set(self.primary_tail_fractions).issubset(allowed):
            raise MetaFunnelError("primary tail fractions must be a non-empty subset of .01/.05/.10")
        if not 0.0 <= float(self.minimum_positive_environment_rate) <= 1.0:
            raise MetaFunnelError("minimum positive environment rate must be in [0, 1]")
        if self.max_worst_month_net_drop_bps is not None and self.max_worst_month_net_drop_bps > 0.0:
            raise MetaFunnelError("max_worst_month_net_drop_bps must be <= 0 when declared")


@dataclass(frozen=True)
class MetaFunnelConfig:
    min_train_rows: int = 32
    ridge_alpha: float = 10.0
    # Refit in bounded chronological blocks, never once per candidate timestamp.
    # A block is fitted only on labels resolved before its first decision time.
    refit_interval_hours: int = 24
    # ``prequential_batched`` is useful for a fully prequential ledger.  The
    # transport protocol is the nested experiment contract: inner base-OOF
    # rows train the meta model and outer base-OOF rows are scored once.
    fit_protocol: str = "prequential_batched"
    meta_partition_column: str = "meta_partition"
    inner_partition_value: str = "inner_oof"
    outer_partition_value: str = "outer_test"
    transport_column: str = "transport"
    h6_max_features: int = 20
    h6_max_correlation: float = 0.80
    h6_holdout_fraction: float = 0.20
    h6_min_holdout_rows: int = 12
    random_seed: int = 20260804

    def __post_init__(self) -> None:
        if self.min_train_rows < 2:
            raise MetaFunnelError("min_train_rows must be at least two")
        if int(self.refit_interval_hours) < 1:
            raise MetaFunnelError("refit_interval_hours must be positive")
        if self.fit_protocol not in {"prequential_batched", "transport_outer_frozen"}:
            raise MetaFunnelError("fit_protocol must be prequential_batched or transport_outer_frozen")
        if not self.meta_partition_column or not self.transport_column:
            raise MetaFunnelError("meta_partition_column and transport_column must be non-empty")
        if not 0 < self.h6_holdout_fraction < 1:
            raise MetaFunnelError("h6_holdout_fraction must be in (0, 1)")
        if self.h6_max_features < 1 or not 0 < self.h6_max_correlation <= 1:
            raise MetaFunnelError("invalid H6 selection limits")


@dataclass(frozen=True)
class ArmSpec:
    arm: str
    feature_groups: tuple[str, ...]
    features: tuple[str, ...]
    stage: str
    control_arm: str
    h6_train_selected: bool = False
    h6_fixed_features: tuple[str, ...] = ()
    h6_candidate_features: tuple[str, ...] = ()
    cluster_similarity_threshold: float | None = None


@dataclass(frozen=True)
class MetaFunnelResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    side_metrics: pd.DataFrame
    side_decile_metrics: pd.DataFrame
    month_metrics: pd.DataFrame
    transport_metrics: pd.DataFrame
    complexity: pd.DataFrame
    h6_selection: pd.DataFrame
    provenance: pd.DataFrame
    ablation_results: pd.DataFrame
    transport_gates: pd.DataFrame
    arms: tuple[ArmSpec, ...]
    model_spec: FrozenMetaModelSpec
    successor: str
    cluster_taxonomy: ClusterTaxonomyContract | None
    stages: tuple[str, ...]
    gate_config: MetaTransportGateConfig
    # The normal research API retains the compact prediction ledger in memory.
    # The CLI supplies a cache path so a wide multi-arm funnel never needs to
    # accumulate every arm before immutable output is written.
    prediction_cache_path: Path | None = None
    prediction_rows: int = 0


def _as_utc(frame: pd.DataFrame, name: str) -> pd.Series:
    result = pd.to_datetime(frame[name], utc=True, errors="coerce")
    if result.isna().any():
        raise MetaFunnelError(f"{name} must contain valid UTC timestamps")
    return result


def _finite(frame: pd.DataFrame, name: str) -> pd.Series:
    result = pd.to_numeric(frame[name], errors="coerce")
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise MetaFunnelError(f"{name} must be finite")
    return result.astype(float)


def reject_raw_leaf_columns(columns: Sequence[str]) -> None:
    """Fail closed on fold-local leaf identifiers, including requested features."""
    forbidden = sorted(
        str(column) for column in columns
        if not str(column).lower().startswith("base_reasoning__g1_leaf_assignment_count")
        and any(token in str(column).lower() for token in RAW_LEAF_ID_TOKENS)
    )
    if forbidden:
        raise MetaFunnelError(
            "raw fold-local leaf identifiers are forbidden; pass stable reasoning "
            f"summaries or cluster features instead: {forbidden}"
        )


def validate_nested_predecessor_oof_contract(
    frame: pd.DataFrame,
    contract: NestedPredecessorOOFContract | None,
    *,
    columns: MetaFunnelColumns = MetaFunnelColumns(),
) -> None:
    """Validate S2 predecessor predictions against the same decision timestamp.

    This is intentionally row-level.  A manifest assertion without a per-row
    fit cutoff cannot prove that the S2 feature was available at that decision.
    """
    if contract is None:
        raise MetaFunnelError("S2 requires a nested predecessor OOF feature contract")
    required = {
        *contract.feature_columns, contract.fit_end_column, contract.generated_column,
        contract.available_column, contract.strict_oof_column, columns.decision,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise MetaFunnelError(f"S2 predecessor OOF contract missing columns: {missing}")
    reject_raw_leaf_columns(contract.feature_columns)
    decision = _as_utc(frame, columns.decision)
    fit_end = _as_utc(frame, contract.fit_end_column)
    generated = _as_utc(frame, contract.generated_column)
    available = _as_utc(frame, contract.available_column)
    if not frame[contract.strict_oof_column].fillna(False).astype(bool).all():
        raise MetaFunnelError("S2 predecessor rows must all be same-side strict OOF")
    if not fit_end.lt(decision).all():
        raise MetaFunnelError("S2 predecessor fit end must strictly precede decision_ts")
    if not generated.le(decision).all() or not available.le(decision).all():
        raise MetaFunnelError("S2 predecessor feature was not available by decision_ts")
    if not fit_end.le(generated).all() or not generated.le(available).all():
        raise MetaFunnelError(
            "S2 predecessor lineage must order fit_end <= generated <= available <= decision_ts"
        )
    for feature in contract.feature_columns:
        _finite(frame, feature)


def validate_successor_meta_contract(
    arms: Sequence[ArmSpec],
    *,
    successor: str,
    predecessor_contract: NestedPredecessorOOFContract | None,
    declared_base_reasoning_features: Sequence[str],
) -> None:
    """Enforce the MG0/MG1/MG2 feature boundary before fitting any model.

    ``S2`` is not merely a differently named rerun: every declared
    predecessor feature must enter the successor feature contract and a compact
    base-reasoning representation must remain present.  Conversely S0/S1 may
    not silently consume predecessor-meta output.  This cannot prove the
    predecessor materialiser itself was nested (that is handled row-by-row by
    :func:`validate_nested_predecessor_oof_contract`), but it prevents a
    manifest-only no-op from being reported as recursion.
    """
    all_features = set(name for spec in arms for name in spec.features)
    predecessor_features = set(predecessor_contract.feature_columns) if predecessor_contract is not None else set()
    base_reasoning_features = set(map(str, declared_base_reasoning_features)).difference(predecessor_features)
    if successor in {"S1", "S2"} and not base_reasoning_features:
        raise MetaFunnelError(f"{successor} requires compact base reasoning features beyond the frozen current-meta control")
    if successor == "S2":
        if predecessor_contract is None:
            raise MetaFunnelError("S2 requires a nested predecessor OOF feature contract")
        missing_predecessor = sorted(set(predecessor_contract.feature_columns).difference(all_features))
        if missing_predecessor:
            raise MetaFunnelError(
                "S2 predecessor OOF features are declared but do not enter any successor arm: "
                f"{missing_predecessor}"
            )
    elif predecessor_contract is not None:
        raise MetaFunnelError(f"{successor} may not declare predecessor-meta features; only S2 may consume them")


def _strict_identity_columns(columns: MetaFunnelColumns) -> tuple[str, ...]:
    """Return the effective complete row key carried through the funnel."""

    return (
        columns.candidate_id, columns.decision, columns.side,
        _STRICT_FOLD, _STRICT_TRANSPORT, _STRICT_PARTITION,
    )


def _normalise_identity_component(
    frame: pd.DataFrame, *, source: str, target: str, fallback: str, label: str,
) -> None:
    """Copy an explicit strict-ID component or mark a legacy singleton scope."""

    if source not in frame:
        frame[target] = fallback
        return
    values = frame[source]
    if values.isna().any() or values.astype(str).str.strip().eq("").any():
        raise MetaFunnelError(f"strict candidate identity {label} must be non-null and non-empty")
    frame[target] = values.astype(str)


def _ranking_tie_columns(columns: MetaFunnelColumns) -> list[str]:
    """Stable full-row tie-breaker; score ranking itself remains global."""

    return list(_strict_identity_columns(columns))


def validate_base_oof_rows(
    frame: pd.DataFrame, *, columns: MetaFunnelColumns = MetaFunnelColumns(),
    config: MetaFunnelConfig | None = None,
) -> pd.DataFrame:
    """Return a normalized ledger after strict same-side base provenance checks."""
    # The compact leaf-reasoning materializer and earlier target ledgers use a
    # few harmless spelling variants.  Accept them at this boundary, then use
    # one explicit in-memory schema for all downstream checks and artifacts.
    aliases = {
        columns.decision: ("__decision_ts__", "__ts__"),
        columns.label_available: ("__label_available_at__", "label_available_at"),
        columns.realized_gross_bps: ("gross_bps",),
        columns.realized_cost_bps: ("cost_bps",),
        columns.realized_net_bps: ("net_bps", "exact_net_bps"),
        columns.base_fit_end: ("base_fit_cutoff_ts", "prediction_fit_end_ts"),
        columns.base_generated: ("feature_generation_ts", "base_map_cutoff_ts", "prediction_generated_ts"),
        columns.base_strict_oof: ("strict_oof", "is_strict_oof", "strict_prequential_oof"),
    }
    frame = frame.copy()
    for target, choices in aliases.items():
        if target not in frame:
            source = next((name for name in choices if name in frame), None)
            if source is not None:
                frame[target] = frame[source]
    required = {
        columns.candidate_id, columns.side, columns.decision, columns.label_available,
        columns.base_expected_bps, columns.realized_gross_bps, columns.realized_cost_bps,
        columns.realized_net_bps, columns.base_fit_end, columns.base_generated,
        columns.base_strict_oof,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise MetaFunnelError(f"base OOF ledger missing columns: {missing}")
    reject_raw_leaf_columns(frame.columns)
    work = frame.copy()
    for name in (columns.decision, columns.label_available, columns.base_fit_end, columns.base_generated):
        work[name] = _as_utc(work, name)
    for name in (columns.base_expected_bps, columns.realized_gross_bps, columns.realized_cost_bps, columns.realized_net_bps):
        work[name] = _finite(work, name)
    if work[[columns.candidate_id, columns.side]].isna().any().any():
        raise MetaFunnelError("candidate_id and side_name must be non-null")
    transport_column = config.transport_column if config is not None else "transport"
    partition_column = config.meta_partition_column if config is not None else "meta_partition"
    _normalise_identity_component(
        work, source=columns.fold_id, target=_STRICT_FOLD,
        fallback="__LEGACY_SINGLE_FOLD__", label="fold_id",
    )
    _normalise_identity_component(
        work, source=transport_column, target=_STRICT_TRANSPORT,
        fallback="__LEGACY_SINGLE_TRANSPORT__", label="transport",
    )
    _normalise_identity_component(
        work, source=partition_column, target=_STRICT_PARTITION,
        fallback="__LEGACY_SINGLE_PARTITION__", label="meta_partition",
    )
    strict_identity = list(_strict_identity_columns(columns))
    if work.duplicated(strict_identity).any():
        raise MetaFunnelError("one base OOF row is required per full strict candidate identity")
    if not work[columns.base_strict_oof].fillna(False).astype(bool).all():
        raise MetaFunnelError("every base row must be same-side strict OOF")
    if not work[columns.base_fit_end].lt(work[columns.decision]).all():
        raise MetaFunnelError("base OOF fit end must strictly precede decision_ts")
    if not work[columns.base_generated].le(work[columns.decision]).all():
        raise MetaFunnelError("base OOF prediction cannot be generated after decision_ts")
    if not work[columns.base_fit_end].lt(work[columns.base_generated]).all():
        raise MetaFunnelError("base OOF fit end must strictly precede its generation timestamp")
    if not work[columns.label_available].gt(work[columns.decision]).all():
        raise MetaFunnelError("label_available_ts must strictly follow decision_ts")
    if not np.allclose(
        work[columns.realized_gross_bps] - work[columns.realized_cost_bps],
        work[columns.realized_net_bps], atol=1e-8, rtol=0.0,
    ):
        raise MetaFunnelError("realized gross bps minus cost bps must equal realized net bps")
    return work.sort_values(_ranking_tie_columns(columns), kind="mergesort").reset_index(drop=True)


def _attach_transport_ids(work: pd.DataFrame, *, config: MetaFunnelConfig) -> pd.DataFrame:
    """Attach a visible transport label without ever pooling ranking across it.

    Legacy one-transport ledgers may omit the label.  They remain runnable for
    focused unit diagnostics, but their output states ``UNSPECIFIED`` rather
    than claiming A/B transport evidence.  Production transport runs must pass
    a real, non-empty label for every candidate row.
    """
    out = work.copy()
    if config.transport_column not in out:
        out["__transport__"] = "UNSPECIFIED"
        return out
    values = out[config.transport_column]
    if values.isna().any() or values.astype(str).str.strip().eq("").any():
        raise MetaFunnelError("transport_id must be non-null and non-empty when supplied")
    out["__transport__"] = values.astype(str)
    return out


def _refit_block(values: pd.Series, *, interval_hours: int) -> pd.Series:
    """Return deterministic UTC refit anchors for bounded causal batches."""
    timestamps = pd.to_datetime(values, utc=True)
    width = int(pd.Timedelta(hours=int(interval_hours)).value)
    anchors = (timestamps.astype("int64") // width) * width
    return pd.to_datetime(anchors, utc=True)


def _normalise_stages(stages: Sequence[str]) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(str(stage).upper() for stage in stages))
    if not values:
        raise MetaFunnelError("at least one ablation stage is required")
    unknown = sorted(set(values).difference(STAGES))
    if unknown:
        raise MetaFunnelError(f"unknown meta-ablation stages: {unknown}; expected a subset of {STAGES}")
    return values


def _select_stage_arms(
    all_arms: Sequence[ArmSpec],
    *,
    stages: Sequence[str],
    feature_groups: Mapping[str, Sequence[str]],
    cluster_groups: Mapping[str, Sequence[str]] | None,
) -> tuple[ArmSpec, ...]:
    """Return a sequentially valid arm set, never fabricated empty H/C arms."""
    requested = _normalise_stages(stages)
    if "H" in requested:
        missing_l4_inputs = [arm for arm in ("L2", "L3") if not tuple(feature_groups.get(arm, ()))]
        if missing_l4_inputs:
            raise MetaFunnelError(
                "H-stage requires the frozen L4 representation (both recurrent "
                f"rule-family L2 and contribution-bundle L3 inputs); missing {missing_l4_inputs}"
            )
        missing_health = [f"H{number}" for number in range(1, 6) if not tuple(feature_groups.get(f"H{number}", ()))]
        if missing_health:
            raise MetaFunnelError(
                "H-stage requires materialised causal H1--H5 features; refusing "
                f"empty/proxy health ablations for {missing_health}"
            )
    if "C" in requested and not cluster_groups:
        raise MetaFunnelError("C-stage requires frozen C0--C6 compact cluster groups")
    if "C" in requested:
        c0 = set(map(str, cluster_groups.get("C0", ()))) if cluster_groups else set()
        required_upstream = set(map(str, feature_groups.get("L0", ()))) | set(map(str, feature_groups.get("L2", ()))) | set(map(str, feature_groups.get("L3", ())))
        missing_upstream = sorted(required_upstream.difference(c0))
        if missing_upstream:
            raise MetaFunnelError(
                "C0 must be the frozen upstream compact representation and retain "
                f"all L4 control fields; missing {missing_upstream}"
            )
    by_arm = {spec.arm: spec for spec in all_arms}
    selected: list[str] = []
    if "L" in requested:
        selected.extend(L_ARMS)
    if "H" in requested:
        # L0 is retained as the universal frozen current-meta diagnostic and
        # L4 is the explicit stage control for H0.
        selected.extend(("L0", "L4", *H_ARMS))
    if "C" in requested:
        # C0 contains the frozen selected upstream compact contract.  L0
        # remains a diagnostic anchor only; C comparisons use C0.  A
        # threshold-sweep taxonomy deliberately exposes only C0--C4: C5/C6
        # do not exist until the immutable post-sweep finalisation has issued
        # their coverage/one-SE overlay.  Select the arms the sealed contract
        # actually constructed rather than indexing fabricated descendants.
        selected.extend(("L0", *(arm for arm in C_ARMS if arm in by_arm)))
    unique = tuple(dict.fromkeys(selected))
    return tuple(by_arm[arm] for arm in unique)


def build_sequential_arms(
    feature_groups: Mapping[str, Sequence[str]],
    cluster_groups: Mapping[str, Sequence[str]] | None = None,
    *,
    cluster_taxonomy: ClusterTaxonomyContract | None = None,
    successor: str = "S0",
) -> tuple[ArmSpec, ...]:
    """Build the linked-spec sequential L/H/C ablation contract.

    The representation arms are intentionally *alternatives*, not an
    accidental factorial/cumulative search:

    - L1 evaluates individual leaf aggregates, L2 evaluates recurrent rule
      families, L3 evaluates contribution bundles, and L4 is exactly L2+L3.
    - H0 is exactly L4.  H1--H5 add health categories cumulatively to H0;
      H6 retains H0 and selects only H1--H5 candidates inside each training
      fold.
    - C0 is the frozen upstream compact representation supplied in the C
      stage; C1--C4 are independent taxonomy thresholds and C5/C6 are their
      pre-declared 95%-coverage/one-SE compact descendants.

    The function never constructs a raw leaf or a cluster.  It only validates
    and wires already-materialised compact aggregates.
    """
    if successor not in S_ALLOWED:
        raise MetaFunnelError(f"successor must be one of {S_ALLOWED}")
    reject_raw_leaf_columns([name for values in feature_groups.values() for name in values])
    if cluster_groups:
        reject_raw_leaf_columns([name for values in cluster_groups.values() for name in values])
    permitted_feature_groups = set(L_ARMS) | set(H_ARMS)
    if successor == "S2":
        permitted_feature_groups.add(S2_REASONING_ENTROPY_GROUP)
        entropy_fields = tuple(map(str, feature_groups.get(S2_REASONING_ENTROPY_GROUP, ())))
        if S2_REASONING_ENTROPY_GROUP in feature_groups and entropy_fields != (S2_REASONING_ENTROPY_FIELD,):
            raise MetaFunnelError(
                f"{S2_REASONING_ENTROPY_GROUP} is reserved for the exact causal "
                f"ledger field {S2_REASONING_ENTROPY_FIELD}"
            )
    unknown = sorted(set(feature_groups).difference(permitted_feature_groups))
    if unknown:
        raise MetaFunnelError(
            "feature groups must be named L0--L4 or H0--H6; only S2 may additionally "
            f"declare {S2_REASONING_ENTROPY_GROUP}, got {unknown}"
        )
    if cluster_groups:
        permitted_cluster_groups = {"C0", *CLUSTER_THRESHOLD_BY_ARM}
        if cluster_taxonomy is not None and str(cluster_taxonomy.selection_phase).lower() == "final":
            permitted_cluster_groups |= {"C5", "C6"}
        unknown_clusters = sorted(set(cluster_groups).difference(permitted_cluster_groups))
        if unknown_clusters:
            raise MetaFunnelError(f"cluster groups must be named C0--C6, got {unknown_clusters}")
        if cluster_taxonomy is None:
            raise MetaFunnelError(
                "C-stage groups require a frozen ClusterTaxonomyContract; "
                "the runner will not infer linkage, thresholds, coverage, or compactness"
            )
    elif cluster_taxonomy is not None:
        raise MetaFunnelError("a ClusterTaxonomyContract requires C0--C6 compact feature groups")

    def extend(current: list[str], names: Sequence[str]) -> list[str]:
        return list(dict.fromkeys([*current, *map(str, names)]))

    l0_features = tuple(map(str, feature_groups.get("L0", ())))
    l1_features = tuple(extend(list(l0_features), feature_groups.get("L1", ())))
    l2_features = tuple(extend(list(l0_features), feature_groups.get("L2", ())))
    l3_features = tuple(extend(list(l0_features), feature_groups.get("L3", ())))
    l4_features = tuple(extend(list(l2_features), feature_groups.get("L3", ())))
    # L4, H0 and H6 are derived representation identities, never hidden
    # feature buckets.  Accept an omitted key, but fail closed on a value.
    for arm in ("L4", "H0", "H6"):
        if tuple(feature_groups.get(arm, ())) != ():
            raise MetaFunnelError(
                f"{arm} is a derived ablation arm and may not declare hidden feature fields"
            )
    specs: list[ArmSpec] = []
    specs.extend((
        ArmSpec("L0", ("L0",), l0_features, "L", "L0"),
        ArmSpec("L1", ("L0", "L1"), l1_features, "L", "L0"),
        ArmSpec("L2", ("L0", "L2"), l2_features, "L", "L0"),
        ArmSpec("L3", ("L0", "L3"), l3_features, "L", "L0"),
        ArmSpec("L4", ("L0", "L2", "L3"), l4_features, "L", "L0"),
    ))
    h0_features = l4_features
    current_h = list(h0_features)
    specs.append(ArmSpec("H0", ("L4",), h0_features, "H", "L4"))
    for number in range(1, 6):
        arm = f"H{number}"
        current_h = extend(current_h, feature_groups.get(arm, ()))
        specs.append(
            ArmSpec(
                arm,
                tuple(["L4", *(f"H{index}" for index in range(1, number + 1))]),
                tuple(current_h),
                "H",
                "H0",
            )
        )
    h6_candidates = tuple(
        dict.fromkeys(name for number in range(1, 6) for name in feature_groups.get(f"H{number}", ()))
    )
    specs.append(
        ArmSpec(
            "H6",
            ("L4", "H1", "H2", "H3", "H4", "H5", "H6_train_only_mda"),
            tuple(dict.fromkeys([*h0_features, *h6_candidates])),
            "H",
            "H0",
            h6_train_selected=True,
            h6_fixed_features=h0_features,
            h6_candidate_features=h6_candidates,
        )
    )
    if cluster_groups:
        c0 = tuple(map(str, cluster_groups.get("C0", ())))
        if not c0:
            raise MetaFunnelError(
                "C0 must contain the frozen upstream compact representation "
                "(for example the selected H6 contract), not leaf clusters"
            )
        specs.append(ArmSpec("C0", ("C0",), c0, "C", "C0"))
        cluster_arms = list(CLUSTER_THRESHOLD_BY_ARM)
        if str(cluster_taxonomy.selection_phase).lower() == "final":
            cluster_arms.extend(("C5", "C6"))
        for arm in cluster_arms:
            additions = tuple(map(str, cluster_groups.get(arm, ())))
            ids = tuple(map(str, cluster_taxonomy.cluster_ids_by_arm.get(arm, ())))
            if bool(ids) != bool(additions):
                raise MetaFunnelError(
                    f"{arm} cluster IDs and compact feature additions disagree; "
                    "every selected cluster taxonomy must have materialised features"
                )
            specs.append(
                ArmSpec(
                    arm,
                    ("C0", arm),
                    tuple(dict.fromkeys([*c0, *additions])),
                    "C",
                    "C0",
                    cluster_similarity_threshold=cluster_taxonomy.threshold_by_arm.get(arm),
                )
            )
    return tuple(specs)


def _matrix(frame: pd.DataFrame, features: Sequence[str], *, allow_nan: bool = False) -> np.ndarray:
    if not features:
        return np.empty((len(frame), 0), dtype=float)
    missing = sorted(set(features).difference(frame.columns))
    if missing:
        raise MetaFunnelError(f"candidate reasoning features missing from input: {missing}")
    values = frame.loc[:, list(features)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if np.isinf(values).any():
        raise MetaFunnelError("candidate reasoning features cannot contain infinity")
    if not allow_nan and not np.isfinite(values).all():
        raise MetaFunnelError("candidate reasoning features must be finite")
    return values


def _train_median_impute(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Impute only from the fitted rows; preserve an explicit missingness audit."""
    train_missing = int(np.isnan(train_x).sum())
    test_missing = int(np.isnan(test_x).sum())
    if not train_missing and not test_missing:
        return train_x, test_x, {"train_missing_cells": 0, "prediction_missing_cells": 0, "all_missing_features": 0}
    medians = np.zeros(train_x.shape[1], dtype=float)
    all_missing = 0
    for index in range(train_x.shape[1]):
        observed = train_x[:, index][np.isfinite(train_x[:, index])]
        if len(observed):
            medians[index] = float(np.median(observed))
        else:
            # The model receives an explicit constant rather than silently
            # dropping an under-covered frozen field; the audit records it.
            all_missing += 1
    filled_train = np.where(np.isnan(train_x), medians, train_x)
    filled_test = np.where(np.isnan(test_x), medians, test_x)
    return filled_train, filled_test, {
        "train_missing_cells": train_missing,
        "prediction_missing_cells": test_missing,
        "all_missing_features": all_missing,
    }


def _h6_sklearn_permutation_indices(
    rows: int,
    *,
    seed: int,
    repeats: int = _H6_MDA_REPEATS,
) -> tuple[np.ndarray, ...]:
    """Reproduce sklearn's dense-column permutation sequence exactly.

    ``permutation_importance`` makes a fresh ``RandomState`` per feature from
    one derived seed.  Its repeat loop mutates the copied column in place, so
    the later permutations compose rather than independently re-indexing the
    original column.  The compact linear path must retain both behaviours.
    """
    random_state = np.random.RandomState(seed)
    derived_seed = int(random_state.randint(np.iinfo(np.int32).max + 1))
    shuffler = np.random.RandomState(derived_seed)
    shuffle_indices = np.arange(rows)
    source_indices = np.arange(rows)
    result: list[np.ndarray] = []
    for _ in range(repeats):
        shuffler.shuffle(shuffle_indices)
        source_indices = source_indices[shuffle_indices]
        result.append(source_indices.copy())
    return tuple(result)


def _h6_linear_permutation_mda(
    model: Ridge,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Exact dense Ridge MDA without copying/predicting a full matrix per field."""
    base_prediction = np.asarray(model.predict(valid_x), dtype=float)
    base_mse = float(np.mean((valid_y - base_prediction) ** 2))
    coefficients = np.asarray(model.coef_, dtype=float).reshape(-1)
    if len(coefficients) != valid_x.shape[1]:
        raise MetaFunnelError("H6 Ridge coefficient shape does not match candidate features")
    importances = np.zeros(valid_x.shape[1], dtype=float)
    for source_indices in _h6_sklearn_permutation_indices(len(valid_x), seed=seed):
        for start in range(0, valid_x.shape[1], _H6_REAL_FEATURE_BLOCK):
            stop = min(start + _H6_REAL_FEATURE_BLOCK, valid_x.shape[1])
            current = valid_x[:, start:stop]
            permuted = valid_x[source_indices, start:stop]
            changed_prediction = base_prediction[:, None] + (
                permuted - current
            ) * coefficients[None, start:stop]
            changed_mse = np.mean((valid_y[:, None] - changed_prediction) ** 2, axis=0)
            importances[start:stop] += changed_mse - base_mse
    return importances / _H6_MDA_REPEATS


def _h6_phantom_score_reference(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    phantom_fit: np.ndarray,
    phantom_valid: np.ndarray,
    permuted_valid: np.ndarray,
    *,
    alpha: float,
) -> float:
    """Reference calculation for a singular/ill-conditioned phantom block."""
    phantom_model = Ridge(alpha=alpha).fit(np.column_stack([fit_x, phantom_fit]), fit_y)
    baseline = np.mean(
        (valid_y - phantom_model.predict(np.column_stack([valid_x, phantom_valid]))) ** 2
    )
    changed = np.mean(
        (valid_y - phantom_model.predict(np.column_stack([valid_x, permuted_valid]))) ** 2
    )
    return float(changed - baseline)


def _h6_batched_phantom_mda(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    *,
    alpha: float,
    seed: int,
) -> np.ndarray:
    """Calculate the existing independent-phantom MDA distribution in blocks.

    Appending one phantom to a Ridge system is a rank-one update.  Factoring
    the fitted candidate Gram matrix once and applying the Schur complement
    yields the same augmented Ridge prediction as a separate fit for every
    phantom.  Keeping the phantoms in small blocks bounds memory while
    retaining every existing q95 draw and its RNG order.
    """
    rows, feature_count = fit_x.shape
    phantom_count = max(8, feature_count)
    x_mean = np.mean(fit_x, axis=0)
    y_mean = float(np.mean(fit_y))
    gram = fit_x.T @ fit_x
    gram -= float(rows) * np.outer(x_mean, x_mean)
    gram.flat[:: feature_count + 1] += float(alpha)
    factor = cho_factor(gram, lower=True, overwrite_a=True, check_finite=False)
    rhs = fit_x.T @ fit_y - float(rows) * x_mean * y_mean
    base_beta = cho_solve(factor, rhs, check_finite=False)
    base_intercept = y_mean - float(x_mean @ base_beta)
    base_prediction = valid_x @ base_beta + base_intercept
    rng = np.random.default_rng(seed)
    scores = np.empty(phantom_count, dtype=float)

    for start in range(0, phantom_count, _H6_PHANTOM_BLOCK):
        stop = min(start + _H6_PHANTOM_BLOCK, phantom_count)
        width = stop - start
        phantom_fit = np.empty((len(fit_x), width), dtype=float)
        phantom_valid = np.empty((len(valid_x), width), dtype=float)
        permuted_valid = np.empty((len(valid_x), width), dtype=float)
        for offset, number in enumerate(range(start, stop)):
            source = number % feature_count
            # Keep exactly the source selection and Generator consumption of
            # the former per-phantom reference loop.
            phantom_fit[:, offset] = rng.permutation(fit_x[:, source])
            phantom_valid[:, offset] = rng.permutation(valid_x[:, source])
            permuted_valid[:, offset] = rng.permutation(phantom_valid[:, offset])

        phantom_mean = np.mean(phantom_fit, axis=0)
        cross = fit_x.T @ phantom_fit - float(rows) * x_mean[:, None] * phantom_mean[None, :]
        solved_cross = cho_solve(factor, cross, check_finite=False)
        response_cross = phantom_fit.T @ fit_y - float(rows) * phantom_mean * y_mean
        denominator = (
            np.einsum("ij,ij->j", phantom_fit, phantom_fit)
            - float(rows) * phantom_mean**2
            + float(alpha)
            - np.einsum("ij,ij->j", cross, solved_cross)
        )
        numerator = response_cross - base_beta @ cross
        valid = np.isfinite(denominator) & np.isfinite(numerator) & (denominator > 0.0)
        if valid.any():
            gamma = numerator[valid] / denominator[valid]
            projection = valid_x @ solved_cross[:, valid] - (x_mean @ solved_cross[:, valid])[None, :]
            baseline_prediction = base_prediction[:, None] + gamma[None, :] * (
                phantom_valid[:, valid] - phantom_mean[None, valid] - projection
            )
            baseline_error = valid_y[:, None] - baseline_prediction
            changed_error = baseline_error - gamma[None, :] * (
                permuted_valid[:, valid] - phantom_valid[:, valid]
            )
            local_scores = scores[start:stop]
            local_scores[valid] = (
                np.mean(changed_error**2, axis=0) - np.mean(baseline_error**2, axis=0)
            )
        for local_index in np.flatnonzero(~valid):
            scores[start + int(local_index)] = _h6_phantom_score_reference(
                fit_x,
                fit_y,
                valid_x,
                valid_y,
                phantom_fit[:, local_index],
                phantom_valid[:, local_index],
                permuted_valid[:, local_index],
                alpha=alpha,
            )
    return scores


def _h6_reference_mda_and_phantoms(
    model: Ridge,
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    *,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The original implementation retained for deterministic parity tests."""
    real = permutation_importance(
        model,
        valid_x,
        valid_y,
        scoring="neg_mean_squared_error",
        n_repeats=_H6_MDA_REPEATS,
        random_state=seed,
    )
    rng = np.random.default_rng(seed)
    phantom_scores: list[float] = []
    for number in range(max(8, fit_x.shape[1])):
        source = number % fit_x.shape[1]
        phantom_fit = rng.permutation(fit_x[:, source])
        phantom_valid = rng.permutation(valid_x[:, source])
        permuted_valid = rng.permutation(phantom_valid)
        phantom_scores.append(
            _h6_phantom_score_reference(
                fit_x,
                fit_y,
                valid_x,
                valid_y,
                phantom_fit,
                phantom_valid,
                permuted_valid,
                alpha=alpha,
            )
        )
    return np.asarray(real.importances_mean, dtype=float), np.asarray(phantom_scores, dtype=float)


def _h6_fit_scale_diagnostics(fit_x: np.ndarray) -> dict[str, Any]:
    """Return observational H6 fit diagnostics without touching solver inputs.

    This deliberately reports column-scale information rather than computing a
    full condition number.  The latter requires another SVD precisely on the
    rare ill-conditioned fallback path and can dominate the selector's work.
    The diagnostics are therefore safe to emit for every train-only H6 job,
    while preserving both the numerical inputs and the selection calculation.
    """
    scales = np.std(fit_x, axis=0)
    finite_scales = scales[np.isfinite(scales)]
    positive_scales = finite_scales[finite_scales > 0.0]
    min_scale = float(np.min(positive_scales)) if len(positive_scales) else np.nan
    max_scale = float(np.max(positive_scales)) if len(positive_scales) else np.nan
    return {
        "h6_fit_rows": int(fit_x.shape[0]),
        "h6_valid_rows": None,  # populated by the selector once the holdout is known
        "h6_fit_feature_count": int(fit_x.shape[1]),
        "h6_fit_nonfinite_cells": int((~np.isfinite(fit_x)).sum()),
        "h6_fit_zero_variance_features": int(np.sum(np.isfinite(scales) & (scales == 0.0))),
        "h6_fit_scale_min": min_scale,
        "h6_fit_scale_median": float(np.median(positive_scales)) if len(positive_scales) else np.nan,
        "h6_fit_scale_max": max_scale,
        # A diagonal-scale proxy is intentionally not a matrix condition
        # number.  It is cheap, transparent, and cannot invoke an additional
        # SVD on a fallback job.
        "h6_fit_scale_ratio": (max_scale / min_scale) if np.isfinite(min_scale) and min_scale > 0.0 else np.nan,
        "h6_fit_condition_number": np.nan,
        "h6_fit_condition_diagnostic": "not_computed_to_preserve_selector_runtime",
    }


def _select_h6_features_train_only_impl(
    train: pd.DataFrame,
    candidate_features: Sequence[str],
    *,
    columns: MetaFunnelColumns,
    config: MetaFunnelConfig,
    use_batched_linear_mda: bool,
) -> tuple[str, ...] | tuple[tuple[str, ...], pd.DataFrame]:
    """Shared H6 contract with either the reference or compact MDA backend."""
    features = tuple(dict.fromkeys(map(str, candidate_features)))
    reject_raw_leaf_columns(features)
    if not features:
        return (), pd.DataFrame(columns=["feature", "mda", "phantom_q95", "selected"])
    x = _matrix(train, features, allow_nan=True)
    target = train[columns.realized_net_bps].to_numpy(float) - train[columns.base_expected_bps].to_numpy(float)
    split = max(config.h6_min_holdout_rows, int(np.ceil(len(train) * config.h6_holdout_fraction)))
    split = min(max(1, split), len(train) - 2)
    if len(train) < max(config.min_train_rows, 4) or split < 1:
        audit = pd.DataFrame({"feature": features, "mda": np.nan, "phantom_q95": np.nan, "selected": False, "reason": "insufficient_train_only_mda_rows", "imputation_method": "train_only_median"})
        return (), audit
    fit_x, valid_x, imputation = _train_median_impute(x[:-split], x[-split:])
    fit_y, valid_y = target[:-split], target[-split:]
    if len(fit_x) < 2:
        return (), pd.DataFrame({"feature": features, "mda": np.nan, "phantom_q95": np.nan, "selected": False, "reason": "insufficient_fit_rows", "imputation_method": "train_only_median"})
    model = Ridge(alpha=config.ridge_alpha).fit(fit_x, fit_y)
    seed = int(config.random_seed + len(train) + len(features))
    solver_telemetry: dict[str, Any] = {
        **_h6_fit_scale_diagnostics(fit_x),
        "h6_valid_rows": int(valid_x.shape[0]),
        "h6_batched_attempted": bool(use_batched_linear_mda),
        "h6_batched_fallback": False,
        "h6_batched_fallback_exception_type": None,
        "h6_batched_fallback_reason": None,
        "h6_mda_backend": "reference" if not use_batched_linear_mda else "batched_linear",
    }
    if use_batched_linear_mda:
        try:
            real_mda = _h6_linear_permutation_mda(model, valid_x, valid_y, seed=seed)
            phantom_scores = _h6_batched_phantom_mda(
                fit_x,
                fit_y,
                valid_x,
                valid_y,
                alpha=config.ridge_alpha,
                seed=seed,
            )
        except np.linalg.LinAlgError as exc:
            # Ridge's declared alpha should make this unreachable, but the
            # reference path is deliberately retained rather than changing a
            # train-only selection decision on an unusual linear algebra host.
            solver_telemetry.update({
                "h6_batched_fallback": True,
                "h6_batched_fallback_exception_type": type(exc).__name__,
                "h6_batched_fallback_reason": str(exc),
                "h6_mda_backend": "reference_after_batched_linear_algebra_failure",
            })
            real_mda, phantom_scores = _h6_reference_mda_and_phantoms(
                model,
                fit_x,
                fit_y,
                valid_x,
                valid_y,
                alpha=config.ridge_alpha,
                seed=seed,
            )
    else:
        real_mda, phantom_scores = _h6_reference_mda_and_phantoms(
            model,
            fit_x,
            fit_y,
            valid_x,
            valid_y,
            alpha=config.ridge_alpha,
            seed=seed,
        )
    q95 = float(np.quantile(phantom_scores, .95))
    order = np.argsort(-real_mda, kind="stable")
    feature_positions = {feature: index for index, feature in enumerate(features)}
    selected: list[str] = []
    rows: list[dict[str, Any]] = []
    for index in order:
        feature = features[int(index)]
        mda = float(real_mda[int(index)])
        correlations: list[float] = []
        for chosen in selected:
            other = fit_x[:, feature_positions[chosen]]
            current = fit_x[:, int(index)]
            if np.std(current) == 0.0 or np.std(other) == 0.0:
                correlations.append(1.0)
            else:
                correlations.append(abs(float(np.corrcoef(current, other)[0, 1])))
        max_corr = max(correlations, default=0.0)
        if not np.isfinite(max_corr):
            max_corr = 1.0
        keep = bool(mda > q95 and max_corr <= config.h6_max_correlation and len(selected) < config.h6_max_features)
        if keep:
            selected.append(feature)
        rows.append({"feature": feature, "mda": mda, "phantom_q95": q95, "max_abs_correlation_to_selected": max_corr, "selected": keep, "selection_scope": "chronological_train_only_internal_holdout", "imputation_method": "train_only_median", **imputation, **solver_telemetry})
    return tuple(selected), pd.DataFrame(rows)


def _select_h6_features_train_only_reference(
    train: pd.DataFrame,
    candidate_features: Sequence[str],
    *,
    columns: MetaFunnelColumns = MetaFunnelColumns(),
    config: MetaFunnelConfig = MetaFunnelConfig(),
) -> tuple[str, ...] | tuple[tuple[str, ...], pd.DataFrame]:
    """Test/reference path for the original H6 MDA implementation."""
    return _select_h6_features_train_only_impl(
        train,
        candidate_features,
        columns=columns,
        config=config,
        use_batched_linear_mda=False,
    )


def select_h6_features_train_only(
    train: pd.DataFrame,
    candidate_features: Sequence[str],
    *,
    columns: MetaFunnelColumns = MetaFunnelColumns(),
    config: MetaFunnelConfig = MetaFunnelConfig(),
) -> tuple[str, ...] | tuple[tuple[str, ...], pd.DataFrame]:
    """Choose H6 fields with the unchanged causal train-only phantom q95 gate.

    The final chronological slice of *the training rows* is an internal MDA
    holdout.  The compact backend is algebraically equivalent to the former
    dense sklearn/Ridge loop, but avoids one full copy/prediction per real
    feature and one full Ridge factorisation per phantom.
    """
    return _select_h6_features_train_only_impl(
        train,
        candidate_features,
        columns=columns,
        config=config,
        use_batched_linear_mda=True,
    )


def _fixed_lgbm_factory(spec: FrozenMetaModelSpec) -> MetaRegressor:
    """Instantiate the already-selected Huber control; never tune it here."""
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise MetaFunnelError("LightGBM is required for the frozen meta control") from exc
    return LGBMRegressor(**dict(spec.params))


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    *,
    columns: MetaFunnelColumns,
    config: MetaFunnelConfig,
    model_spec: FrozenMetaModelSpec,
    model_factory: MetaModelFactory,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    if not features or len(train) < config.min_train_rows:
        return np.zeros(len(test), dtype=float), "base_only_fallback", {"missing_value_handling": "not_fitted", "train_missing_cells": 0, "prediction_missing_cells": 0, "all_missing_features": 0}
    target = train[columns.realized_net_bps].to_numpy(float) - train[columns.base_expected_bps].to_numpy(float)
    train_x = _matrix(train, features, allow_nan=True)
    test_x = _matrix(test, features, allow_nan=True)
    if model_factory is _fixed_lgbm_factory:
        model = model_factory(model_spec).fit(train_x, target)
        audit: dict[str, Any] = {
            "missing_value_handling": "lightgbm_native_nan",
            "train_missing_cells": int(np.isnan(train_x).sum()),
            "prediction_missing_cells": int(np.isnan(test_x).sum()),
            "all_missing_features": int(np.isnan(train_x).all(axis=0).sum()),
        }
    else:
        train_x, test_x, audit_counts = _train_median_impute(train_x, test_x)
        model = model_factory(model_spec).fit(train_x, target)
        audit = {"missing_value_handling": "train_only_median_for_injected_factory", **audit_counts}
    return np.asarray(model.predict(test_x), dtype=float), "side_local_frozen_huber_lgbm", audit


def _compact_prediction_columns(
    frame: pd.DataFrame,
    *,
    columns: MetaFunnelColumns,
    config: MetaFunnelConfig,
) -> list[str]:
    """Return the minimum replayable, full-lineage prediction schema.

    A meta prediction does not need a second copy of every several-hundred
    column input feature.  The ledger is immutable upstream evidence; this
    compact panel retains the strict row identity, realised economics, base
    lineage, and every value needed to replay the funnel metrics.  Keeping the
    contract explicit avoids the previous accidental multiplication of the
    full feature matrix by every ablation arm.
    """

    wanted = [
        *_strict_identity_columns(columns),
        columns.label_available,
        columns.base_fit_end,
        columns.base_generated,
        columns.base_strict_oof,
        columns.base_expected_bps,
        columns.realized_gross_bps,
        columns.realized_cost_bps,
        columns.realized_net_bps,
        "__transport__",
        columns.fold_id,
        config.transport_column,
        config.meta_partition_column,
    ]
    # Legacy ledgers may intentionally omit the visible fold/transport/
    # partition fields; their normalised strict counterparts above remain the
    # binding identity in that case.
    return list(dict.fromkeys(name for name in wanted if name in frame.columns))


def _compact_prediction_frame(
    test: pd.DataFrame,
    *,
    columns: MetaFunnelColumns,
    config: MetaFunnelConfig,
) -> pd.DataFrame:
    """Copy only replay/provenance fields before an arm frame is retained."""

    return test.loc[:, _compact_prediction_columns(test, columns=columns, config=config)].copy()


class _IncrementalPredictionParquet:
    """Append compact arm ledgers to one Zstandard Parquet cache.

    The writer owns no pandas frame after :meth:`append` returns.  A cache is
    deliberately *not* an immutable result: the caller must still publish it
    through :func:`write_immutable_meta_funnel_output`, which validates that a
    manifest is written last.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.exists():
            raise MetaFunnelError(f"prediction cache path already exists: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer: pq.ParquetWriter | None = None
        self._schema: pa.Schema | None = None
        self.rows = 0

    def append(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self._writer is None:
            self._schema = table.schema
            self._writer = pq.ParquetWriter(
                self.path,
                table.schema,
                compression="zstd",
                use_dictionary=True,
                write_statistics=True,
            )
        else:
            assert self._schema is not None  # constructor invariant
            if table.schema.names != self._schema.names:
                raise MetaFunnelError("incremental prediction chunks changed schema")
            # pandas occasionally omits non-semantic schema metadata on a
            # later chunk.  Casting preserves the first complete contract and
            # still rejects a genuine type mismatch.
            if not table.schema.equals(self._schema, check_metadata=False):
                try:
                    table = table.cast(self._schema)
                except pa.ArrowInvalid as exc:
                    raise MetaFunnelError("incremental prediction chunks changed field types") from exc
        assert self._writer is not None
        self._writer.write_table(table, row_group_size=131_072)
        self.rows += len(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def abort(self) -> None:
        self.close()
        if self.path.exists():
            self.path.unlink()


def _identity_signature(frame: pd.DataFrame, *, columns: MetaFunnelColumns) -> tuple[int, str, str]:
    """Return two independent canonical strict-identity fingerprints.

    The candidate key itself is validated exactly before fitting.  These two
    keyed, sorted fingerprints then verify that every arm emitted the same
    evaluated identity without retaining a second multi-million-row identity
    panel solely for pairing.
    """

    identity_columns = _ranking_tie_columns(columns)
    identity = frame.loc[:, identity_columns]
    if identity.duplicated(identity_columns).any():
        raise MetaFunnelError("an ablation arm emitted duplicate full strict candidate identities")

    def _digest(key: str) -> str:
        hashes = pd.util.hash_pandas_object(identity, index=False, hash_key=key).to_numpy(dtype=np.uint64)
        hashes.sort()
        return hashlib.sha256(hashes.tobytes()).hexdigest()

    return len(identity), _digest("0123456789abcdef"), _digest("fedcba9876543210")


def _mean_or_nan(frame: pd.DataFrame, column: str) -> float:
    return float(frame[column].mean()) if len(frame) else float("nan")


def _rank_ic(score: pd.Series, outcome: pd.Series) -> float:
    if len(score) < 2 or score.nunique(dropna=False) < 2 or outcome.nunique(dropna=False) < 2:
        return float("nan")
    return float(score.corr(outcome, method="spearman"))


def _evaluate_arm(
    predictions: pd.DataFrame,
    *,
    arm: str,
    columns: MetaFunnelColumns,
    feature_count: int,
    group_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Score each transport independently, with one pooled cross-side tail.

    Ranking never happens per timestamp or per side.  Transport separation is
    intentionally retained: pooling A and B into one ranked book would turn a
    transport comparison into a cross-period allocation experiment.
    """
    work = predictions.loc[predictions.arm.eq(arm)].copy()
    metric_rows: list[dict[str, Any]] = []
    side_rows: list[dict[str, Any]] = []
    month_rows: list[dict[str, Any]] = []
    decile_rows: list[dict[str, Any]] = []
    transport_rows: list[dict[str, Any]] = []
    for transport, transport_work in work.groupby("__transport__", observed=True, sort=True):
        transport_work = transport_work.copy()
        transport_rows.append({
            "arm": arm, "transport_id": str(transport), "population_rows": len(transport_work),
            "decision_start_ts": transport_work[columns.decision].min(),
            "decision_end_ts": transport_work[columns.decision].max(),
            "month_count": int(transport_work["__month__"].nunique()),
            "side_count": int(transport_work[columns.side].nunique()),
            "selection_scope": "one_pooled_global_post_common_bps_top_k_per_transport",
        })
        for fraction in (.01, .05, .10):
            selected_n = max(1, int(np.ceil(len(transport_work) * fraction)))
            selected = transport_work.sort_values(
                ["common_bps_score", *_ranking_tie_columns(columns)],
                ascending=[False, *([True] * len(_ranking_tie_columns(columns)))], kind="mergesort",
            ).head(selected_n)
            by_month = selected.groupby("__month__", observed=True)[columns.realized_net_bps].mean()
            latest_month = str(by_month.index.max()) if len(by_month) else ""
            metric_rows.append({
                "arm": arm,
                "transport_id": str(transport),
                "selection_scope": "one_pooled_global_post_common_bps_top_k_per_transport",
                "top_fraction": fraction,
                "population_rows": len(transport_work),
                "selected_rows": len(selected),
                "gross_bps": _mean_or_nan(selected, columns.realized_gross_bps),
                "cost_bps": _mean_or_nan(selected, columns.realized_cost_bps),
                "net_bps": _mean_or_nan(selected, columns.realized_net_bps),
                "gross_bps_sum": float(selected[columns.realized_gross_bps].sum()),
                "cost_bps_sum": float(selected[columns.realized_cost_bps].sum()),
                "net_bps_sum": float(selected[columns.realized_net_bps].sum()),
                "worst_month_net_bps": float(by_month.min()) if len(by_month) else float("nan"),
                "positive_month_count": int((by_month > 0.0).sum()),
                "selected_month_count": int(len(by_month)),
                "positive_month_fraction": float((by_month > 0.0).mean()) if len(by_month) else float("nan"),
                "latest_selected_month": latest_month,
                "latest_selected_month_net_bps": float(by_month.loc[latest_month]) if latest_month else float("nan"),
                "feature_count": feature_count,
                "feature_group_count": group_count,
            })
            for side, local in selected.groupby(columns.side, observed=True, sort=True):
                population_side = transport_work.loc[transport_work[columns.side].eq(side)]
                side_rows.append({
                    "arm": arm, "transport_id": str(transport), "top_fraction": fraction,
                    "side_name": str(side), "metric_scope": "pooled_global_selected_tail_side_composition",
                    "population_rows": len(population_side), "selected_rows": len(local),
                    "selected_share": float(len(local) / len(selected)),
                    "population_share": float(len(population_side) / len(transport_work)),
                    "gross_bps": _mean_or_nan(local, columns.realized_gross_bps),
                    "cost_bps": _mean_or_nan(local, columns.realized_cost_bps),
                    "net_bps": _mean_or_nan(local, columns.realized_net_bps),
                    "net_bps_sum": float(local[columns.realized_net_bps].sum()),
                })
            for month, month_local in selected.groupby("__month__", observed=True, sort=True):
                for side_name, side_local in [("ALL", month_local), *[(str(side), local) for side, local in month_local.groupby(columns.side, observed=True, sort=True)]]:
                    month_rows.append({
                        "arm": arm, "transport_id": str(transport), "top_fraction": fraction,
                        "month": str(month), "side_name": side_name, "selected_rows": len(side_local),
                        "gross_bps": _mean_or_nan(side_local, columns.realized_gross_bps),
                        "cost_bps": _mean_or_nan(side_local, columns.realized_cost_bps),
                        "net_bps": _mean_or_nan(side_local, columns.realized_net_bps),
                        "net_bps_sum": float(side_local[columns.realized_net_bps].sum()),
                    })
        for side, local in transport_work.groupby(columns.side, observed=True, sort=True):
            local = local.sort_values(
                ["common_bps_score", *_ranking_tie_columns(columns)],
                ascending=[False, *([True] * len(_ranking_tie_columns(columns)))], kind="mergesort",
            ).copy()
            local_score = local["common_bps_score"]
            local_net = local[columns.realized_net_bps]
            false_positive = (local_score > 0.0) & (local_net <= 0.0)
            ranks = np.arange(1, len(local) + 1, dtype=float)
            local["__score_decile__"] = np.minimum(10, np.ceil(10.0 * ranks / max(len(local), 1))).astype(int)
            decile_mean = local.groupby("__score_decile__", observed=True)[columns.realized_net_bps].mean().sort_index()
            monotonic = float((np.diff(decile_mean.to_numpy(float)) <= 0.0).mean()) if len(decile_mean) > 1 else float("nan")
            side_rows.append({
                "arm": arm, "transport_id": str(transport), "top_fraction": np.nan,
                "side_name": str(side), "population_rows": len(local), "metric_scope": "all_scored_rows",
                "within_side_linear_ic": float(local_score.corr(local_net)) if len(local) > 1 and local_score.nunique() > 1 and local_net.nunique() > 1 else float("nan"),
                "within_side_rank_ic": _rank_ic(local_score, local_net),
                "within_side_decile_monotonic_nonincreasing_fraction": monotonic,
                "false_positive_rows": int(false_positive.sum()),
                "false_positive_rate": float(false_positive.mean()),
                "false_positive_gross_bps": _mean_or_nan(local.loc[false_positive], columns.realized_gross_bps),
                "false_positive_cost_bps": _mean_or_nan(local.loc[false_positive], columns.realized_cost_bps),
                "false_positive_net_bps": _mean_or_nan(local.loc[false_positive], columns.realized_net_bps),
                "false_positive_net_bps_sum": float(local.loc[false_positive, columns.realized_net_bps].sum()),
            })
            for decile, decile_local in local.groupby("__score_decile__", observed=True, sort=True):
                decile_rows.append({
                    "arm": arm, "transport_id": str(transport), "side_name": str(side),
                    "score_decile": int(decile), "score_decile_semantics": "1_is_highest_common_bps_score",
                    "population_rows": len(local), "rows": len(decile_local),
                    "score_mean_bps": float(decile_local["common_bps_score"].mean()),
                    "gross_bps": _mean_or_nan(decile_local, columns.realized_gross_bps),
                    "cost_bps": _mean_or_nan(decile_local, columns.realized_cost_bps),
                    "net_bps": _mean_or_nan(decile_local, columns.realized_net_bps),
                    "net_bps_sum": float(decile_local[columns.realized_net_bps].sum()),
                })
    return metric_rows, side_rows, month_rows, decile_rows, transport_rows


def _median_absolute_deviation(values: Iterable[float]) -> float:
    series = pd.Series(list(values), dtype=float).dropna()
    if series.empty:
        return float("nan")
    median = float(series.median())
    return float((series - median).abs().median())


def _attach_control_metrics(
    metrics: pd.DataFrame,
    *,
    arms: Sequence[ArmSpec],
) -> pd.DataFrame:
    """Attach the appropriate frozen upstream control to every arm.

    Existing L0-relative columns are retained for backwards-compatible
    diagnostics, but promotion and grouped MDA must use the stage control:
    H1--H6 against H0 and C1--C6 against C0.  This prevents an apparent lift
    from being attributed to a later stage when it was actually supplied by an
    upstream representation.
    """
    control_by_arm = {spec.arm: spec.control_arm for spec in arms}
    result = metrics.copy()
    result["control_arm"] = result["arm"].map(control_by_arm)
    if result["control_arm"].isna().any():  # pragma: no cover - defensive
        raise MetaFunnelError("every ablation arm must declare a control arm")
    available = result.loc[:, ["arm", "transport_id", "top_fraction", "gross_bps", "net_bps", "net_bps_sum", "worst_month_net_bps"]].rename(
        columns={
            "arm": "control_arm",
            "gross_bps": "control_gross_bps",
            "net_bps": "control_net_bps",
            "net_bps_sum": "control_net_bps_sum",
            "worst_month_net_bps": "control_worst_month_net_bps",
        }
    )
    result = result.merge(
        available,
        on=["control_arm", "transport_id", "top_fraction"],
        how="left",
        validate="many_to_one",
    )
    if result["control_net_bps"].isna().any():
        missing = result.loc[result["control_net_bps"].isna(), ["arm", "control_arm"]].drop_duplicates().to_dict("records")
        raise MetaFunnelError(f"ablation arms lack their declared control metrics: {missing}")
    result["incremental_global_top_k_gross_bps_vs_control"] = result["gross_bps"] - result["control_gross_bps"]
    result["incremental_global_top_k_net_bps_vs_control"] = result["net_bps"] - result["control_net_bps"]
    result["incremental_global_top_k_net_bps_sum_vs_control"] = result["net_bps_sum"] - result["control_net_bps_sum"]
    result["worst_month_net_bps_delta_vs_control"] = result["worst_month_net_bps"] - result["control_worst_month_net_bps"]
    return result


def _month_control_lifts(
    month_metrics: pd.DataFrame,
    *,
    arms: Sequence[ArmSpec],
) -> pd.DataFrame:
    """Compare calendar-month economics against the same arm's stage control."""
    if month_metrics.empty:
        return pd.DataFrame()
    control_by_arm = {spec.arm: spec.control_arm for spec in arms}
    result = month_metrics.loc[month_metrics["side_name"].eq("ALL")].copy()
    result["control_arm"] = result["arm"].map(control_by_arm)
    baseline = result.loc[:, ["arm", "transport_id", "top_fraction", "month", "net_bps", "net_bps_sum"]].rename(
        columns={"arm": "control_arm", "net_bps": "control_net_bps", "net_bps_sum": "control_net_bps_sum"}
    )
    result = result.merge(
        baseline,
        on=["control_arm", "transport_id", "top_fraction", "month"],
        how="left",
        validate="many_to_one",
    )
    result["monthly_net_bps_lift_vs_control"] = result["net_bps"] - result["control_net_bps"]
    result["monthly_net_bps_sum_lift_vs_control"] = result["net_bps_sum"] - result["control_net_bps_sum"]
    return result


def build_meta_ablation_gates(
    metrics: pd.DataFrame,
    month_metrics: pd.DataFrame,
    complexity: pd.DataFrame,
    *,
    arms: Sequence[ArmSpec],
    config: MetaTransportGateConfig = MetaTransportGateConfig(),
    grouped_mda: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise transport evidence without selecting or touching final OOS.

    This deliberately makes the grouped-MDA requirement explicit.  A positive
    development tail alone cannot mark an arm as promoted.  The optional
    table is produced by the separate immutable grouped-MDA sidecar because
    it must refit only prior-resolved rows and cannot be inferred from the
    already-scored prediction panel.
    """
    stage_by_arm = {spec.arm: spec.stage for spec in arms}
    controls = {spec.arm: spec.control_arm for spec in arms}
    specs = {spec.arm: spec for spec in arms}
    mda: pd.DataFrame | None = None
    if grouped_mda is not None:
        required_mda = {
            "arm", "transport_id", "transport_mda_bps", "stable_transport_mda_bps",
            "phantom_q95_bps",
        }
        missing_mda = sorted(required_mda.difference(grouped_mda.columns))
        if missing_mda:
            raise MetaFunnelError(f"grouped MDA evidence lacks required columns: {missing_mda}")
        mda = grouped_mda.loc[:, list(required_mda)].copy()
        mda["arm"] = mda["arm"].astype(str)
        mda["transport_id"] = mda["transport_id"].astype(str)
        for name in required_mda.difference({"arm", "transport_id"}):
            mda[name] = pd.to_numeric(mda[name], errors="coerce")
        if mda.duplicated(["arm", "transport_id"]).any():
            raise MetaFunnelError("grouped MDA evidence duplicates an arm/transport row")
    monthly = _month_control_lifts(month_metrics, arms=arms)
    summary_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for arm in sorted(stage_by_arm):
        arm_metrics = metrics.loc[metrics["arm"].eq(arm)].copy()
        arm_complexity = complexity.loc[complexity["arm"].eq(arm)].copy()
        row: dict[str, Any] = {
            "arm": arm,
            "stage": stage_by_arm[arm],
            "control_arm": controls[arm],
            "transport_count": int(arm_metrics["transport_id"].nunique()),
            "feature_count_declared": float(arm_complexity["feature_count_declared"].max()) if not arm_complexity.empty else float("nan"),
            "mean_selected_feature_count": float(arm_complexity["mean_selected_feature_count"].mean()) if not arm_complexity.empty else float("nan"),
            "fit_count": int(arm_complexity["fit_count"].sum()) if not arm_complexity.empty else 0,
        }
        gate: dict[str, Any] = {
            "arm": arm,
            "stage": stage_by_arm[arm],
            "control_arm": controls[arm],
            "transport_count": row["transport_count"],
            "two_transport_evidence": row["transport_count"] >= config.required_transport_count,
            "grouped_transport_mda_evidence_present": False,
            "grouped_transport_mda_pass": False,
            "promotion_status": "DEVELOPMENT_METRICS_ONLY_GROUPED_MDA_REQUIRED",
        }
        incremental_features = tuple(
            field for field in specs[arm].features
            if field not in set(specs[controls[arm]].features)
        )
        expected_transports = set(arm_metrics["transport_id"].astype(str))
        mda_local = (
            mda.loc[mda["arm"].eq(arm)].copy()
            if mda is not None else pd.DataFrame()
        )
        mda_complete = bool(
            incremental_features
            and len(mda_local) == len(expected_transports)
            and set(mda_local["transport_id"]) == expected_transports
        )
        if incremental_features:
            finite_mda = bool(
                mda_complete
                and np.isfinite(mda_local[["transport_mda_bps", "stable_transport_mda_bps", "phantom_q95_bps"]].to_numpy(float)).all()
            )
            mda_pass = bool(
                finite_mda
                and (mda_local["stable_transport_mda_bps"] > 0.0).all()
                and (mda_local["stable_transport_mda_bps"] > mda_local["phantom_q95_bps"]).all()
            )
            gate["grouped_transport_mda_evidence_present"] = bool(mda_complete)
            gate["grouped_transport_mda_pass"] = mda_pass
            gate["grouped_transport_mda_status"] = (
                "PASS" if mda_pass else "FAIL" if mda_complete else "MISSING_OR_INCOMPLETE"
            )
            row["median_transport_mda_bps"] = float(mda_local["transport_mda_bps"].median()) if finite_mda else float("nan")
            row["stable_transport_mda_bps"] = float(mda_local["stable_transport_mda_bps"].median()) if finite_mda else float("nan")
            row["phantom_q95_bps"] = float(mda_local["phantom_q95_bps"].median()) if finite_mda else float("nan")
        else:
            gate["grouped_transport_mda_status"] = "NOT_APPLICABLE_STAGE_CONTROL"
            row["median_transport_mda_bps"] = float("nan")
            row["stable_transport_mda_bps"] = float("nan")
            row["phantom_q95_bps"] = float("nan")
        for fraction in config.primary_tail_fractions:
            local = arm_metrics.loc[arm_metrics["top_fraction"].eq(float(fraction))]
            lifts = local["incremental_global_top_k_net_bps_vs_control"]
            key = f"top{int(fraction * 100):02d}"
            row[f"{key}_median_transport_net_lift_bps"] = float(lifts.median()) if not lifts.empty else float("nan")
            row[f"{key}_mad_transport_net_lift_bps"] = _median_absolute_deviation(lifts)
            row[f"{key}_worst_transport_net_lift_bps"] = float(lifts.min()) if not lifts.empty else float("nan")
            row[f"{key}_positive_transport_fraction"] = float((lifts > 0.0).mean()) if not lifts.empty else float("nan")
            gate[f"{key}_positive_in_every_transport"] = bool(
                len(local) >= config.required_transport_count and local["transport_id"].nunique() >= config.required_transport_count and (lifts > 0.0).all()
            )
            monthly_local = monthly.loc[(monthly["arm"].eq(arm)) & (monthly["top_fraction"].eq(float(fraction)))] if not monthly.empty else pd.DataFrame()
            month_lifts = monthly_local.get("monthly_net_bps_lift_vs_control", pd.Series(dtype=float)).dropna()
            row[f"{key}_positive_environment_rate"] = float((month_lifts > 0.0).mean()) if not month_lifts.empty else float("nan")
            row[f"{key}_environment_count"] = int(len(month_lifts))
            gate[f"{key}_environment_rate_pass"] = bool(
                len(month_lifts) and float((month_lifts > 0.0).mean()) >= config.minimum_positive_environment_rate
            )
            worst = local["worst_month_net_bps_delta_vs_control"].dropna()
            row[f"{key}_worst_month_net_delta_bps"] = float(worst.min()) if not worst.empty else float("nan")
            if config.max_worst_month_net_drop_bps is None:
                gate[f"{key}_worst_month_gate_configured"] = False
                gate[f"{key}_no_catastrophic_worst_month_reversal"] = False
            else:
                gate[f"{key}_worst_month_gate_configured"] = True
                gate[f"{key}_no_catastrophic_worst_month_reversal"] = bool(
                    len(worst) and (worst >= config.max_worst_month_net_drop_bps).all()
                )
        primary_checks = [
            gate[f"top{int(fraction * 100):02d}_positive_in_every_transport"]
            and gate[f"top{int(fraction * 100):02d}_environment_rate_pass"]
            and gate[f"top{int(fraction * 100):02d}_no_catastrophic_worst_month_reversal"]
            for fraction in config.primary_tail_fractions
        ]
        gate["development_economic_gates_pass"] = bool(gate["two_transport_evidence"] and all(primary_checks))
        gate["passes_all_advancement_gates"] = bool(
            gate["development_economic_gates_pass"]
            and (gate["grouped_transport_mda_pass"] if incremental_features else False)
        )
        if gate["passes_all_advancement_gates"]:
            gate["promotion_status"] = "DEVELOPMENT_ADVANCEMENT_EVIDENCE_COMPLETE_FINAL_OOS_STILL_REQUIRED"
        elif grouped_mda is not None:
            gate["promotion_status"] = "DEVELOPMENT_GATE_FAILED_OR_GROUPED_MDA_REJECTED"
        summary_rows.append(row)
        gate_rows.append(gate)
    return pd.DataFrame(summary_rows), pd.DataFrame(gate_rows)


def compare_successor_meta_generations(
    metrics_by_generation: Mapping[str, pd.DataFrame],
    *,
    selected_arm_by_generation: Mapping[str, str],
    gate_config: MetaTransportGateConfig = MetaTransportGateConfig(),
) -> pd.DataFrame:
    """Compare S0/S1/S2 only on matched transport rows and primary tails.

    It is intentionally a post-run development validator rather than a model
    trainer.  S2 can be labelled ``PREDECESSOR_META_REASONING_ADDS_VALUE`` only
    when it beats S1 (not merely S0) on *every* required transport and both
    top-5/top-10 tails.  The input must come from runs which already enforced
    the row-level nested predecessor OOF contract; this function never
    reconstructs predecessor predictions from a score column.
    """
    required_generations = ("S0", "S1", "S2")
    missing = [name for name in required_generations if name not in metrics_by_generation or name not in selected_arm_by_generation]
    if missing:
        raise MetaFunnelError(f"successor comparison requires S0/S1/S2 metrics and selected arms; missing {missing}")
    prepared: list[pd.DataFrame] = []
    for generation in required_generations:
        table = metrics_by_generation[generation]
        required = {"arm", "transport_id", "top_fraction", "net_bps", "gross_bps", "cost_bps"}
        absent = sorted(required.difference(table.columns))
        if absent:
            raise MetaFunnelError(f"{generation} metrics lack required successor-comparison fields: {absent}")
        arm = str(selected_arm_by_generation[generation])
        local = table.loc[
            table["arm"].eq(arm) & table["top_fraction"].isin(gate_config.primary_tail_fractions),
            ["transport_id", "top_fraction", "net_bps", "gross_bps", "cost_bps"],
        ].copy()
        expected_rows = int(local["transport_id"].nunique()) * len(gate_config.primary_tail_fractions)
        if local.empty or len(local) != expected_rows or local.duplicated(["transport_id", "top_fraction"]).any():
            raise MetaFunnelError(
                f"{generation}/{arm} must have exactly one matched metric row per transport and primary tail"
            )
        prepared.append(local.rename(columns={name: f"{generation.lower()}_{name}" for name in ("net_bps", "gross_bps", "cost_bps")}))
    comparison = prepared[0]
    for next_table in prepared[1:]:
        comparison = comparison.merge(next_table, on=["transport_id", "top_fraction"], how="outer", validate="one_to_one", indicator=True)
        if not comparison.pop("_merge").eq("both").all():
            raise MetaFunnelError("S0/S1/S2 successor comparison must use identical transport/tail cells")
    comparison["s1_incremental_net_bps_vs_s0"] = comparison["s1_net_bps"] - comparison["s0_net_bps"]
    comparison["s2_incremental_net_bps_vs_s1"] = comparison["s2_net_bps"] - comparison["s1_net_bps"]
    comparison["s2_beats_s1_cell"] = comparison["s2_incremental_net_bps_vs_s1"].gt(0.0)
    comparison["s1_selected_arm"] = str(selected_arm_by_generation["S1"])
    comparison["s2_selected_arm"] = str(selected_arm_by_generation["S2"])
    comparison["s0_selected_arm"] = str(selected_arm_by_generation["S0"])
    enough_transports = comparison["transport_id"].nunique() >= gate_config.required_transport_count
    s2_advances = bool(enough_transports and comparison["s2_beats_s1_cell"].all())
    comparison["required_transport_count"] = gate_config.required_transport_count
    comparison["s2_beats_s1_every_transport_and_primary_tail"] = s2_advances
    comparison["terminal_decision"] = (
        "PREDECESSOR_META_REASONING_ADDS_VALUE"
        if s2_advances else "PREDECESSOR_META_RECURSION_REJECTED"
    )
    return comparison.sort_values(["top_fraction", "transport_id"], kind="mergesort").reset_index(drop=True)


def run_leaf_reasoning_meta_funnel(
    frame: pd.DataFrame,
    *,
    feature_groups: Mapping[str, Sequence[str]],
    cluster_groups: Mapping[str, Sequence[str]] | None = None,
    cluster_taxonomy: ClusterTaxonomyContract | None = None,
    stages: Sequence[str] = ("L",),
    successor: str = "S0",
    predecessor_contract: NestedPredecessorOOFContract | None = None,
    model_spec: FrozenMetaModelSpec | None = None,
    model_factory: MetaModelFactory | None = None,
    columns: MetaFunnelColumns = MetaFunnelColumns(),
    config: MetaFunnelConfig = MetaFunnelConfig(),
    gate_config: MetaTransportGateConfig = MetaTransportGateConfig(),
    prediction_cache_path: str | Path | None = None,
) -> MetaFunnelResult:
    """Run side-local frozen-Huber ablations under an explicit causal protocol.

    ``transport_outer_frozen`` is the nested experiment mode: fit exactly once
    per transport/side/arm on eligible inner base-OOF rows, then score the
    corresponding outer rows.  ``prequential_batched`` is retained only for a
    ledger which contains no outer partition; it refits at fixed time blocks,
    never per decision timestamp.
    """
    if successor not in S_ALLOWED:
        raise MetaFunnelError(f"successor must be one of {S_ALLOWED}")
    work = _attach_transport_ids(validate_base_oof_rows(frame, columns=columns, config=config), config=config)
    if successor == "S2":
        validate_nested_predecessor_oof_contract(work, predecessor_contract, columns=columns)
    if model_spec is None:
        raise MetaFunnelError("a frozen LightGBM Huber meta model specification is required")
    if model_factory is None:
        model_factory = _fixed_lgbm_factory
    all_arms = build_sequential_arms(
        feature_groups,
        cluster_groups,
        cluster_taxonomy=cluster_taxonomy,
        successor=successor,
    )
    selected_stages = _normalise_stages(stages)
    arms = _select_stage_arms(
        all_arms,
        stages=selected_stages,
        feature_groups=feature_groups,
        cluster_groups=cluster_groups,
    )
    validate_successor_meta_contract(
        arms,
        successor=successor,
        predecessor_contract=predecessor_contract,
        declared_base_reasoning_features=tuple(
            dict.fromkeys(
                [*map(str, feature_groups.get("L2", ())), *map(str, feature_groups.get("L3", ()))]
            )
        ),
    )
    l0 = next(spec for spec in arms if spec.arm == "L0")
    if not l0.features:
        raise MetaFunnelError("L0 must declare the frozen current meta-control feature subset")
    missing_control = sorted(set(FROZEN_META_CONTROL_FEATURES).difference(l0.features))
    if missing_control:
        raise MetaFunnelError(
            "L0 must include every frozen current-meta control field; missing "
            f"{missing_control}"
        )
    # Keep the normal library API convenient for small diagnostics, but let
    # the CLI stream each compact arm panel to disk.  The latter is essential
    # for H/C: retaining a full wide ledger once per arm previously multiplied
    # memory by the number of ablations.
    predictions: list[pd.DataFrame] = []
    prediction_sink = (
        _IncrementalPredictionParquet(prediction_cache_path)
        if prediction_cache_path is not None else None
    )
    prediction_rows = 0
    h6_rows: list[pd.DataFrame] = []
    provenance_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []

    if config.fit_protocol == "transport_outer_frozen":
        required = {config.meta_partition_column, config.transport_column}
        missing = sorted(required.difference(work.columns))
        if missing:
            raise MetaFunnelError(f"transport_outer_frozen requires explicit columns: {missing}")
        allowed = {config.inner_partition_value, config.outer_partition_value}
        actual = set(work[config.meta_partition_column].astype(str))
        unexpected = sorted(actual.difference(allowed))
        if unexpected:
            raise MetaFunnelError(f"unknown meta partition values: {unexpected}")

    def _jobs_for_arm() -> list[tuple[str, str, pd.Timestamp, pd.DataFrame, pd.DataFrame, str]]:
        jobs: list[tuple[str, str, pd.Timestamp, pd.DataFrame, pd.DataFrame, str]] = []
        if config.fit_protocol == "prequential_batched":
            if config.meta_partition_column in work and work[config.meta_partition_column].astype(str).eq(config.outer_partition_value).any():
                raise MetaFunnelError("outer meta rows require transport_outer_frozen; prequential scoring would misuse the test partition")
            staged = work.copy()
            staged["__refit_block__"] = _refit_block(staged[columns.decision], interval_hours=config.refit_interval_hours)
            for (transport, block, side), test in staged.groupby(["__transport__", "__refit_block__", columns.side], sort=True, observed=True):
                train = staged.loc[
                    staged["__transport__"].eq(transport)
                    & staged[columns.side].eq(side)
                    & staged[columns.label_available].lt(block)
                ].copy()
                jobs.append((str(transport), str(side), pd.Timestamp(block), train, test.copy(), "prequential_refit_block"))
            return jobs
        for transport, cell in work.groupby("__transport__", sort=True, observed=True):
            partition = cell[config.meta_partition_column].astype(str)
            outer = cell.loc[partition.eq(config.outer_partition_value)].copy()
            if outer.empty:
                raise MetaFunnelError(f"transport {transport!r} has no outer meta evaluation rows")
            inner = cell.loc[partition.eq(config.inner_partition_value)].copy()
            if inner.empty:
                raise MetaFunnelError(f"transport {transport!r} has no inner base-OOF meta training rows")
            outer_start = outer[columns.decision].min()
            if not outer[columns.decision].ge(outer_start).all():  # defensive, documents the cutoff
                raise AssertionError("outer start computation is inconsistent")
            for side, test in outer.groupby(columns.side, sort=True, observed=True):
                train = inner.loc[
                    inner[columns.side].eq(side) & inner[columns.label_available].lt(outer_start)
                ].copy()
                if train.empty:
                    raise MetaFunnelError(f"transport {transport!r}/{side!r} has no prior-resolved inner meta rows")
                if not train[columns.label_available].lt(outer_start).all():
                    raise AssertionError("unresolved label entered frozen outer meta fit")
                jobs.append((str(transport), str(side), pd.Timestamp(outer_start), train, test.copy(), "inner_oof_frozen_outer"))
        return jobs

    metric_rows: list[dict[str, Any]] = []
    side_rows: list[dict[str, Any]] = []
    month_rows: list[dict[str, Any]] = []
    side_decile_rows: list[dict[str, Any]] = []
    transport_rows: list[dict[str, Any]] = []
    complexity_rows: list[dict[str, Any]] = []
    # Jobs are derived exclusively from the sealed base ledger and are feature
    # independent.  Materialise them once so every arm is paired to precisely
    # the same train/evaluation rows.
    jobs = _jobs_for_arm()
    reference_identity: tuple[int, str, str] | None = None
    try:
        for spec in arms:
            arm_parts: list[pd.DataFrame] = []
            for transport, side, fit_reference, train, test, protocol_detail in jobs:
                selected_features = spec.features
                if spec.h6_train_selected:
                    selected_candidates, audit = select_h6_features_train_only(
                        train,
                        spec.h6_candidate_features,
                        columns=columns,
                        config=config,
                    )
                    # H6 is a compact *health* selection on top of the frozen L4
                    # control.  It must never make the control disappear merely
                    # because a train-only MDA split did not retain one of its
                    # fields.
                    selected_features = tuple(dict.fromkeys([*spec.h6_fixed_features, *selected_candidates]))
                    if not audit.empty:
                        h6_rows.append(
                            audit.assign(
                                arm=spec.arm,
                                transport_id=transport,
                                side_name=side,
                                scoring_decision_ts=fit_reference,
                                train_rows=len(train),
                                selection_role="H1_H5_train_only_candidate",
                                frozen_l4_feature_count=len(spec.h6_fixed_features),
                            )
                        )
                residual, mode, missingness = _fit_predict(
                    train, test, selected_features, columns=columns, config=config,
                    model_spec=model_spec, model_factory=model_factory,
                )
                if config.fit_protocol == "transport_outer_frozen" and mode == "base_only_fallback":
                    raise MetaFunnelError(
                        f"transport {transport!r}/{side!r}/{spec.arm} has fewer than min_train_rows; do not evaluate a frozen outer arm as base-only"
                    )
                # Do not retain the wide feature matrix in the result.  The
                # immutable source ledger remains the feature evidence; this is a
                # compact, replayable prediction/economics/provenance panel.
                out = _compact_prediction_frame(test, columns=columns, config=config)
                out["arm"] = spec.arm
                out["predicted_residual_bps"] = residual
                # Common-bps reconstruction precedes the one pooled global rank.
                out["common_bps_score"] = out[columns.base_expected_bps].to_numpy(float) + residual
                out["base_plus_residual_bps"] = out["common_bps_score"]
                out["fit_mode"] = mode
                out["meta_train_rows"] = len(train)
                out["meta_train_label_available_max"] = train[columns.label_available].max() if len(train) else pd.NaT
                out["meta_fit_reference_ts"] = fit_reference
                out["meta_fit_protocol"] = config.fit_protocol
                out["meta_fit_protocol_detail"] = protocol_detail
                out["selected_feature_count"] = len(selected_features)
                out["selected_h6_candidate_feature_count"] = (
                    len(selected_features) - len(spec.h6_fixed_features)
                    if spec.h6_train_selected else 0
                )
                out["selected_features_json"] = json.dumps(list(selected_features))
                out["frozen_meta_model_contract_id"] = model_spec.contract_id
                out["frozen_meta_model_params_hash"] = model_spec.params_hash
                out["missing_value_handling"] = str(missingness["missing_value_handling"])
                arm_parts.append(out)
                provenance_rows.append({
                    "arm": spec.arm, "transport_id": transport, "fit_reference_ts": fit_reference,
                    "side_name": side, "train_rows": len(train), "evaluation_rows": len(test),
                    "max_label_available_used": train[columns.label_available].max() if len(train) else pd.NaT,
                    "strict_prior_resolved": bool(len(train) == 0 or train[columns.label_available].lt(fit_reference).all()),
                    "fit_mode": mode, "fit_protocol": config.fit_protocol,
                    "fit_protocol_detail": protocol_detail,
                    "selected_features_json": json.dumps(list(selected_features)),
                    "selected_h6_candidate_feature_count": (
                        len(selected_features) - len(spec.h6_fixed_features)
                        if spec.h6_train_selected else 0
                    ),
                    "frozen_meta_model_contract_id": model_spec.contract_id,
                    "frozen_meta_model_params_hash": model_spec.params_hash,
                    **missingness,
                })
                fit_rows.append({
                    "arm": spec.arm, "transport_id": transport, "side_name": side,
                    "fit_reference_ts": fit_reference, "train_rows": len(train), "prediction_rows": len(test),
                    "selected_feature_count": len(selected_features),
                    "selected_h6_candidate_feature_count": (
                        len(selected_features) - len(spec.h6_fixed_features)
                        if spec.h6_train_selected else 0
                    ),
                    "train_matrix_bytes_estimate": int(len(train) * len(selected_features) * np.dtype(float).itemsize),
                    "prediction_matrix_bytes_estimate": int(len(test) * len(selected_features) * np.dtype(float).itemsize),
                    "fit_protocol": config.fit_protocol,
                    **missingness,
                })
            arm_frame = pd.concat(arm_parts, ignore_index=True)
            arm_frame["__month__"] = arm_frame[columns.decision].dt.strftime("%Y-%m")
            signature = _identity_signature(arm_frame, columns=columns)
            if reference_identity is None:
                reference_identity = signature
            elif signature != reference_identity:
                raise MetaFunnelError("all ablation arms must score identical candidate rows")
            one, two, three, four, five = _evaluate_arm(
                arm_frame,
                arm=spec.arm,
                columns=columns,
                feature_count=len(spec.features),
                group_count=len(spec.feature_groups),
            )
            metric_rows.extend(one); side_rows.extend(two); month_rows.extend(three); side_decile_rows.extend(four); transport_rows.extend(five)
            fit_audit = pd.DataFrame(fit_rows).loc[lambda x: x.arm.eq(spec.arm)]
            for transport, local_fit in fit_audit.groupby("transport_id", observed=True, sort=True):
                complexity_rows.append({
                    "arm": spec.arm, "transport_id": str(transport),
                    "feature_count_declared": len(spec.features), "feature_group_count": len(spec.feature_groups),
                    "mean_selected_feature_count": float(arm_frame.loc[arm_frame.__transport__.eq(transport), "selected_feature_count"].mean()),
                    "h6_train_only_selection": spec.h6_train_selected,
                    "fit_count": len(local_fit), "total_fit_rows": int(local_fit.train_rows.sum()),
                    "max_train_rows": int(local_fit.train_rows.max()), "max_prediction_rows_per_fit": int(local_fit.prediction_rows.max()),
                    "max_train_matrix_bytes_estimate": int(local_fit.train_matrix_bytes_estimate.max()),
                    "max_prediction_matrix_bytes_estimate": int(local_fit.prediction_matrix_bytes_estimate.max()),
                    "estimated_peak_matrix_bytes": int(local_fit.train_matrix_bytes_estimate.max() + local_fit.prediction_matrix_bytes_estimate.max()),
                    "refit_interval_hours": int(config.refit_interval_hours), "fit_protocol": config.fit_protocol,
                    "frozen_meta_model_contract_id": model_spec.contract_id,
                    "frozen_meta_model_params_hash": model_spec.params_hash,
                    "features_json": json.dumps(list(spec.features)),
                })
            prediction_rows += len(arm_frame)
            if prediction_sink is None:
                predictions.append(arm_frame)
            else:
                prediction_sink.append(arm_frame)
    except BaseException:
        if prediction_sink is not None:
            prediction_sink.abort()
        raise
    else:
        if prediction_sink is not None:
            prediction_sink.close()
    all_predictions = (
        pd.concat(predictions, ignore_index=True)
        if prediction_sink is None else pd.DataFrame()
    )
    metrics = pd.DataFrame(metric_rows)
    baseline = metrics.loc[metrics.arm.eq("L0"), ["transport_id", "top_fraction", "gross_bps", "net_bps", "net_bps_sum"]].rename(columns={"gross_bps": "l0_gross_bps", "net_bps": "l0_net_bps", "net_bps_sum": "l0_net_bps_sum"})
    metrics = metrics.merge(baseline, on=["transport_id", "top_fraction"], how="left", validate="many_to_one")
    metrics["incremental_global_top_k_gross_bps_vs_l0"] = metrics.gross_bps - metrics.l0_gross_bps
    metrics["incremental_global_top_k_net_bps_vs_l0"] = metrics.net_bps - metrics.l0_net_bps
    metrics["incremental_global_top_k_net_bps_sum_vs_l0"] = metrics.net_bps_sum - metrics.l0_net_bps_sum
    metrics = _attach_control_metrics(metrics, arms=arms)
    # Pairing was checked after every arm against the same two keyed,
    # canonical strict-identity fingerprints before that arm could be released
    # to the bounded prediction cache.
    ablation_results, transport_gates = build_meta_ablation_gates(
        metrics,
        pd.DataFrame(month_rows),
        pd.DataFrame(complexity_rows),
        arms=arms,
        config=gate_config,
    )
    return MetaFunnelResult(
        all_predictions,
        metrics,
        pd.DataFrame(side_rows),
        pd.DataFrame(side_decile_rows),
        pd.DataFrame(month_rows),
        pd.DataFrame(transport_rows),
        pd.DataFrame(complexity_rows),
        pd.concat(h6_rows, ignore_index=True) if h6_rows else pd.DataFrame(),
        pd.DataFrame(provenance_rows),
        ablation_results,
        transport_gates,
        arms,
        model_spec,
        successor,
        cluster_taxonomy,
        selected_stages,
        gate_config,
        Path(prediction_cache_path) if prediction_cache_path is not None else None,
        prediction_rows,
    )


def write_immutable_meta_funnel_output(
    result: MetaFunnelResult,
    output_root: str | Path,
    *,
    config: MetaFunnelConfig = MetaFunnelConfig(),
    gate_config: MetaTransportGateConfig | None = None,
    consume_prediction_cache: bool = False,
) -> Path:
    """Write one compact, immutable research artifact atomically.

    The output directory is deliberately invisible to consumers until every
    table and its manifest have been written and hashed.  This matters for the
    large prediction panel: a full CSV can exhaust disk mid-write, and a
    partially populated directory must never look like a completed research
    result.  All tables use Zstandard-compressed Parquet so the high-cardinality
    prediction panel is not inflated by CSV serialisation.
    """
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = root / f"leaf_reasoning_meta_funnel_{stamp}_{uuid4().hex[:10]}"
    if output.exists():  # UUID collision must fail rather than weaken immutability.
        raise MetaFunnelError(f"immutable meta-funnel output already exists: {output}")
    temporary = root / f".{output.name}.tmp-{uuid4().hex}"
    temporary.mkdir()  # no exist_ok: a staging collision is never reusable
    cache_path = result.prediction_cache_path
    if cache_path is not None and not cache_path.is_file():
        raise MetaFunnelError(f"bounded prediction cache is missing: {cache_path}")
    if cache_path is not None and not result.predictions.empty:
        raise MetaFunnelError("a result may use either in-memory predictions or a bounded prediction cache, not both")
    tables = {
        "metrics": result.metrics,
        "side_metrics": result.side_metrics,
        "side_decile_metrics": result.side_decile_metrics,
        "month_metrics": result.month_metrics,
        "transport_metrics": result.transport_metrics,
        "complexity": result.complexity,
        "h6_train_only_selection": result.h6_selection,
        "provenance": result.provenance,
        "ablation_results": result.ablation_results,
        "transport_gates": result.transport_gates,
    }
    try:
        hashes: dict[str, str] = {}
        prediction_path = temporary / "predictions.parquet"
        if cache_path is None:
            result.predictions.to_parquet(prediction_path, index=False, compression="zstd")
        elif consume_prediction_cache:
            # The CLI locates the cache beside this staging directory.  Moving
            # it avoids a second multi-gigabyte copy while the staged directory
            # remains invisible until its manifest is written and renamed.
            cache_path.replace(prediction_path)
        else:
            shutil.copyfile(cache_path, prediction_path)
        if result.prediction_rows:
            cached_rows = int(pq.ParquetFile(prediction_path).metadata.num_rows)
            if cached_rows != result.prediction_rows:
                raise MetaFunnelError(
                    "bounded prediction cache row count differs from the completed funnel result"
                )
        hashes[prediction_path.name] = _sha256_file(prediction_path)
        for name, table in tables.items():
            path = temporary / f"{name}.parquet"
            table.to_parquet(path, index=False, compression="zstd")
            hashes[path.name] = _sha256_file(path)
        effective_gate_config = result.gate_config if gate_config is None else gate_config
        manifest = {
            "created_utc": stamp,
            "immutable_output": True,
            "artifact_state": "COMPLETE",
            "table_format": "parquet_zstd",
            "prediction_materialization": (
                "bounded_incremental_parquet_cache"
                if cache_path is not None else "in_memory_compact_prediction_ledger"
            ),
            "prediction_rows": int(result.prediction_rows or len(result.predictions)),
            "common_bps_mapping": "base_expected_bps + predicted_residual_bps",
            "global_rank_scope": "one_pooled_cross_side_book_per_arm_per_transport_after_common_bps_mapping",
            "h6_selection": "chronological_train_only_phantom_q95_mda_correlation_le_0.80_max20_with_frozen_l4_controls_retained",
            "selection_status": "DEVELOPMENT_METRICS_ONLY; grouped chronological transport-aware MDA and final untouched OOS remain required",
            "successor": result.successor,
            "stages": list(result.stages),
            "frozen_meta_model": {
                "family": result.model_spec.family,
                "contract_id": result.model_spec.contract_id,
                "params": dict(result.model_spec.params),
                "params_hash": result.model_spec.params_hash,
            },
            "config": asdict(config),
            "transport_gate_config": asdict(effective_gate_config),
            "cluster_taxonomy": asdict(result.cluster_taxonomy) if result.cluster_taxonomy is not None else None,
            "arms": [asdict(arm) for arm in result.arms],
            "sha256": hashes,
        }
        # The manifest is last inside staging; its publication happens only
        # through the same-filesystem directory rename below.
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if output.exists():  # defensive check immediately before publication
            raise MetaFunnelError(f"immutable meta-funnel output already exists: {output}")
        temporary.rename(output)
    except BaseException:
        # A failed output must not be confused with a sealed result and should
        # not strand multi-gigabyte staging data after an interrupted run.
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def _sha256_file(path: Path) -> str:
    """Hash a potentially multi-gigabyte Parquet table without loading it."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
