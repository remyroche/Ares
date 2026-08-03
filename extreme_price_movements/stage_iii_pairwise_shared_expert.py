"""Bounded shared residual expert with an optional within-context rank head.

Round F of the Stage-III funnel compares exactly three *shared* arms:

``F0``
    a robust pointwise candidate-residual model;
``F1``
    that same model plus a small 50-bps within-context ranking correction;
``F2``
    that same model plus a small 100-bps within-context ranking correction.

The ranking head is deliberately auxiliary.  It is fit on pairs emitted by
``construct_context_matched_residual_pairs`` (same side, date, comparable soft
regime, base value and cost-to-ATR).  A training-only affine calibration maps
its otherwise unitless LambdaRank score back into *candidate-residual bps* and
the final score is a bounded blend with the pointwise bps prediction.  The
result therefore remains directly comparable across long, short and soft
regime states after the standard causal common-bps reconstruction.

This module is a modelling primitive, not a Stage-III runner.  It neither
selects an arm nor opens data/artifacts.  Callers must provide a frozen feature
contract and a strictly prior-resolved training ledger for every fit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .shared_regime_residual_expert import (
    SCHEMA as SHARED_RESIDUAL_SCHEMA,
    SharedResidualColumns,
    SharedResidualExpertError,
    SharedResidualExpertFit,
    fit_shared_regime_residual_expert,
    reconstruct_shared_regime_expected_net_bps,
)
from .stage_iii_residual_target_challengers import (
    ONE_SHARED_MODEL,
    ORDINAL_EDGES_BPS,
    PairColumns,
    PairConstructionConfig,
    ResidualTargetColumns,
    StageIIIResidualTargetError,
    candidate_residual_bps as challenger_candidate_residual_bps,
    construct_context_matched_residual_pairs,
)


SCHEMA = "stage_iii_pairwise_shared_expert_v1"
COMMON_BPS_RECONSTRUCTION = (
    "frozen_base_expected_net_bps_plus_prior_resolved_soft_regime_prior_residual_bps_"
    "plus_predicted_candidate_residual_bps"
)
ONE_SHARED_BOTH_SIDE_MODEL = "one_shared_both_side_model_no_local_or_hard_routing"
PairwiseArm = Literal["F0_pointwise", "F1_pairwise_50bps", "F2_pairwise_100bps"]
_PAIRWISE_ARMS: tuple[PairwiseArm, ...] = (
    "F0_pointwise",
    "F1_pairwise_50bps",
    "F2_pairwise_100bps",
)
_PAIRWISE_SEPARATION: dict[PairwiseArm, float | None] = {
    "F0_pointwise": None,
    "F1_pairwise_50bps": 50.0,
    "F2_pairwise_100bps": 100.0,
}
_EPS = 1e-12
_FORBIDDEN_FEATURE_TOKENS = (
    "exact_net", "realised_net", "realized_net", "outcome_resolved",
    "candidate_residual", "target", "label", "future_", "mfe", "mae",
)
_HARD_REGIME_FEATURE_TOKENS = (
    "regime_id", "regime_code", "regime_class", "hard_regime", "argmax_regime",
)


class PairwiseSharedExpertError(ValueError):
    """Raised when a Round-F shared-expert contract is violated."""


@dataclass(frozen=True)
class PairwiseSharedResidualColumns:
    """Narrow ledger contract shared by pointwise and pairwise Round-F arms."""

    decision_timestamp: str = "decision_ts"
    label_available_timestamp: str = "label_available_ts"
    side: str = "side_name"
    candidate_id: str = "candidate_id"
    symbol: str = "symbol"
    exact_net_bps: str = "exact_net_bps"
    base_expected_net_bps: str = "prequential_base_expected_net_bps"
    regime_prior_residual_bps: str = "prequential_soft_regime_prior_residual_bps"
    candidate_residual_bps: str = "candidate_residual_bps"
    cost_to_atr: str = "cost_to_atr"
    base_map_prequential_flag: str = "base_map_is_prequential"
    soft_regime_causal_flag: str = "soft_regime_is_causal_prequential"
    cost_atr_causal_flag: str = "cost_atr_is_causal"

    def shared_columns(self) -> SharedResidualColumns:
        return SharedResidualColumns(
            decision_timestamp=self.decision_timestamp,
            label_available_timestamp=self.label_available_timestamp,
            side=self.side,
            exact_net_bps=self.exact_net_bps,
            base_expected_net_bps=self.base_expected_net_bps,
        )

    def pair_columns(self) -> PairColumns:
        return PairColumns(
            decision_timestamp=self.decision_timestamp,
            label_available_timestamp=self.label_available_timestamp,
            side=self.side,
            candidate_id=self.candidate_id,
            base_expected_net_bps=self.base_expected_net_bps,
            cost_to_atr=self.cost_to_atr,
            base_map_prequential_flag=self.base_map_prequential_flag,
            soft_regime_causal_flag=self.soft_regime_causal_flag,
            cost_atr_causal_flag=self.cost_atr_causal_flag,
        )


@dataclass(frozen=True)
class PairwiseSharedResidualConfig:
    """Small, deterministic limits for the optional shared pairwise head."""

    pairwise_blend_weight: float = 0.10
    pairwise_prediction_clip_bps: float = 400.0
    calibration_ridge: float = 16.0
    ranker_estimators: int = 96
    ranker_learning_rate: float = 0.035
    ranker_num_leaves: int = 15
    ranker_min_child_samples: int = 32
    ranker_l2: float = 4.0
    random_state: int = 1729

    def validate(self) -> None:
        if not 0.0 <= self.pairwise_blend_weight <= 0.25:
            raise PairwiseSharedExpertError(
                "pairwise_blend_weight must lie in [0, 0.25]"
            )
        if not 1.0 <= self.pairwise_prediction_clip_bps <= 2_000.0:
            raise PairwiseSharedExpertError("pairwise_prediction_clip_bps must lie in [1, 2000]")
        if self.calibration_ridge < 0.0:
            raise PairwiseSharedExpertError("calibration_ridge must be non-negative")
        if not 1 <= self.ranker_estimators <= 512:
            raise PairwiseSharedExpertError("ranker_estimators must lie in [1, 512]")
        if not 0.001 <= self.ranker_learning_rate <= 0.25:
            raise PairwiseSharedExpertError("ranker_learning_rate must lie in [0.001, 0.25]")
        if not 2 <= self.ranker_num_leaves <= 64:
            raise PairwiseSharedExpertError("ranker_num_leaves must lie in [2, 64]")
        if not 1 <= self.ranker_min_child_samples <= 1_024:
            raise PairwiseSharedExpertError("ranker_min_child_samples must lie in [1, 1024]")
        if self.ranker_l2 < 0.0:
            raise PairwiseSharedExpertError("ranker_l2 must be non-negative")


@dataclass(frozen=True)
class PairwiseBpsCalibration:
    """Training-only map from a raw shared rank score to residual bps."""

    score_mean: float
    score_scale: float
    intercept_bps: float
    slope_bps_per_standard_score: float
    ridge: float
    rows: int
    fit_before_utc: pd.Timestamp
    max_label_available_utc: pd.Timestamp
    source: str = "training_only_weighted_affine_rank_score_to_candidate_residual_bps"

    def predict_bps(self, raw_score: Sequence[float], *, clip_bps: float) -> np.ndarray:
        value = np.asarray(raw_score, dtype=np.float64).reshape(-1)
        if not np.isfinite(value).all():
            raise PairwiseSharedExpertError("pairwise ranker returned non-finite scores")
        if self.score_scale <= 0.0 or not np.isfinite(self.score_scale):
            raise PairwiseSharedExpertError("pairwise calibration has an invalid score scale")
        calibrated = self.intercept_bps + self.slope_bps_per_standard_score * (
            (value - self.score_mean) / self.score_scale
        )
        return np.clip(calibrated, -clip_bps, clip_bps).astype(np.float32)


@dataclass(frozen=True)
class PairSupportAudit:
    """Compact exact-support evidence for a bounded context-pair ledger."""

    separation_bps: float | None
    pair_selection: str
    constructed_pairs: int
    selected_pairs: int
    selected_pair_rows: int
    pair_ledger_sha256: str | None
    selected_pair_ledger_sha256: str | None
    selected_pairs_by_side: tuple[tuple[str, int], ...]
    selected_unique_candidates: int
    max_pair_label_available_utc: pd.Timestamp | None
    pair_builder_schema: str
    pair_builder_routing: str
    pair_config: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["selected_pairs_by_side"] = [list(value) for value in self.selected_pairs_by_side]
        result["max_pair_label_available_utc"] = (
            None if self.max_pair_label_available_utc is None
            else self.max_pair_label_available_utc.isoformat()
        )
        result["pair_config"] = dict(self.pair_config)
        return result


@dataclass(frozen=True)
class PairwiseSharedResidualAudit:
    """Immutable model/feature/cutoff/support evidence for one Round-F arm."""

    schema: str
    arm: PairwiseArm
    routing: str
    reconstruction: str
    feature_names: tuple[str, ...]
    feature_sha256: str
    training_row_count: int
    training_candidate_ids_sha256: str
    training_rows_by_side: tuple[tuple[str, int], ...]
    training_cutoff_utc: pd.Timestamp
    max_label_available_utc: pd.Timestamp
    pointwise_target_mode: str
    pointwise_model_class: str
    pointwise_params: Mapping[str, Any]
    pairwise_model_class: str | None
    pairwise_params: Mapping[str, Any] | None
    pairwise_config: Mapping[str, Any]
    pair_support: PairSupportAudit
    pairwise_calibration: PairwiseBpsCalibration | None
    shared_residual_schema: str = SHARED_RESIDUAL_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["feature_names"] = list(self.feature_names)
        result["training_rows_by_side"] = [list(value) for value in self.training_rows_by_side]
        result["training_cutoff_utc"] = self.training_cutoff_utc.isoformat()
        result["max_label_available_utc"] = self.max_label_available_utc.isoformat()
        result["pointwise_params"] = dict(self.pointwise_params)
        result["pairwise_params"] = (
            None if self.pairwise_params is None else dict(self.pairwise_params)
        )
        result["pairwise_config"] = dict(self.pairwise_config)
        result["pair_support"] = self.pair_support.to_dict()
        if self.pairwise_calibration is not None:
            calibration = asdict(self.pairwise_calibration)
            calibration["fit_before_utc"] = self.pairwise_calibration.fit_before_utc.isoformat()
            calibration["max_label_available_utc"] = self.pairwise_calibration.max_label_available_utc.isoformat()
            result["pairwise_calibration"] = calibration
        return result


@dataclass(frozen=True)
class PairwiseSharedResidualExpertFit:
    """One both-side residual expert with an optional, small pairwise head."""

    arm: PairwiseArm
    pointwise_fit: SharedResidualExpertFit
    pairwise_model: Any | None
    pairwise_calibration: PairwiseBpsCalibration | None
    config: PairwiseSharedResidualConfig
    audit: PairwiseSharedResidualAudit
    columns: PairwiseSharedResidualColumns = PairwiseSharedResidualColumns()

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.pointwise_fit.feature_names

    def predict_candidate_residual_bps(self, frame: pd.DataFrame) -> np.ndarray:
        """Return a bounded common-unit candidate-residual correction in bps."""
        _validate_feature_frame(frame, self.feature_names)
        try:
            pointwise = self.pointwise_fit.predict_candidate_residual_bps(frame).astype(np.float64)
        except SharedResidualExpertError as exc:
            raise PairwiseSharedExpertError(str(exc)) from exc
        if self.arm == "F0_pointwise":
            return pointwise.astype(np.float32)
        if self.pairwise_model is None or self.pairwise_calibration is None:
            raise PairwiseSharedExpertError("pairwise arm is missing its frozen ranker/calibration")
        raw = np.asarray(
            self.pairwise_model.predict(frame.loc[:, self.feature_names]), dtype=np.float64
        ).reshape(-1)
        pairwise_bps = self.pairwise_calibration.predict_bps(
            raw, clip_bps=self.config.pairwise_prediction_clip_bps
        ).astype(np.float64)
        if len(pairwise_bps) != len(pointwise):
            raise PairwiseSharedExpertError("pairwise prediction lost row alignment")
        # This is intentionally a small, explicit correction—not a hard ranker
        # replacement—so pointwise bps semantics remain dominant and global.
        weight = self.config.pairwise_blend_weight
        return ((1.0 - weight) * pointwise + weight * pairwise_bps).astype(np.float32)

    def predict_expected_net_bps(self, frame: pd.DataFrame) -> np.ndarray:
        """Causally reconstruct a pooled-global comparable expected-net score."""
        return reconstruct_shared_regime_expected_net_bps(
            frame,
            self.predict_candidate_residual_bps(frame),
            columns=self.columns.shared_columns(),
        )


@dataclass(frozen=True)
class PreservedBaseTargetAudit:
    """Identity evidence proving the auxiliary ranker did not replace T3/T4.

    The adapter requires the original model's own frozen audit to describe the
    exact same input rows, features, cutoff and challenger label digest.  This
    makes a silent Huber/pointwise substitution impossible at this boundary.
    """

    base_model_class: str
    base_target_arm: str
    base_target_formulation: str
    base_audit_sha256: str
    base_feature_sha256: str
    base_target_label_sha256: str
    base_training_prediction_sha256: str
    base_training_cutoff_utc: pd.Timestamp
    base_max_label_available_utc: pd.Timestamp
    base_training_rows: int
    base_training_candidate_ids_sha256: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["base_training_cutoff_utc"] = self.base_training_cutoff_utc.isoformat()
        result["base_max_label_available_utc"] = self.base_max_label_available_utc.isoformat()
        return result


@dataclass(frozen=True)
class TargetPreservingPairwiseAudit:
    """Round-F audit for an adapter layered over one already-fitted T3/T4 base."""

    schema: str
    arm: PairwiseArm
    routing: str
    reconstruction: str
    feature_names: tuple[str, ...]
    feature_sha256: str
    training_row_count: int
    training_candidate_ids_sha256: str
    training_rows_by_side: tuple[tuple[str, int], ...]
    training_cutoff_utc: pd.Timestamp
    max_label_available_utc: pd.Timestamp
    preserved_base_target: PreservedBaseTargetAudit
    candidate_residual_label_sha256: str
    pairwise_model_class: str | None
    pairwise_params: Mapping[str, Any] | None
    pairwise_config: Mapping[str, Any]
    pair_support: PairSupportAudit
    pairwise_calibration: PairwiseBpsCalibration | None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["feature_names"] = list(self.feature_names)
        result["training_rows_by_side"] = [list(value) for value in self.training_rows_by_side]
        result["training_cutoff_utc"] = self.training_cutoff_utc.isoformat()
        result["max_label_available_utc"] = self.max_label_available_utc.isoformat()
        result["preserved_base_target"] = self.preserved_base_target.to_dict()
        result["pairwise_params"] = (
            None if self.pairwise_params is None else dict(self.pairwise_params)
        )
        result["pairwise_config"] = dict(self.pairwise_config)
        result["pair_support"] = self.pair_support.to_dict()
        if self.pairwise_calibration is not None:
            calibration = asdict(self.pairwise_calibration)
            calibration["fit_before_utc"] = self.pairwise_calibration.fit_before_utc.isoformat()
            calibration["max_label_available_utc"] = self.pairwise_calibration.max_label_available_utc.isoformat()
            result["pairwise_calibration"] = calibration
        return result


@dataclass(frozen=True)
class TargetPreservingPairwiseAdapterFit:
    """A small shared F1/F2 ranker layered over an immutable T3/T4 base fit."""

    arm: PairwiseArm
    base_model: Any
    feature_names: tuple[str, ...]
    pairwise_model: Any | None
    pairwise_calibration: PairwiseBpsCalibration | None
    config: PairwiseSharedResidualConfig
    audit: TargetPreservingPairwiseAudit
    columns: PairwiseSharedResidualColumns = PairwiseSharedResidualColumns()

    def _base_prediction(self, frame: pd.DataFrame) -> np.ndarray:
        predictor = getattr(self.base_model, "predict_candidate_residual_bps", None)
        if not callable(predictor):
            raise PairwiseSharedExpertError(
                "preserved base model no longer exposes predict_candidate_residual_bps"
            )
        value = np.asarray(predictor(frame)).reshape(-1)
        if len(value) != len(frame) or not np.isfinite(value).all():
            raise PairwiseSharedExpertError(
                "preserved base target prediction must be finite and row-aligned"
            )
        return value

    def predict_candidate_residual_bps(self, frame: pd.DataFrame) -> np.ndarray:
        """Return F0 unchanged, or a capped shared pairwise blend over T3/T4."""
        _validate_feature_frame(frame, self.feature_names)
        base = self._base_prediction(frame)
        # F0 is a strict no-op around the supplied target fit.  In particular,
        # do not cast/recalibrate/rerank its output at this adapter boundary.
        if self.arm == "F0_pointwise":
            return base
        if self.pairwise_model is None or self.pairwise_calibration is None:
            raise PairwiseSharedExpertError("target-preserving pairwise arm lacks ranker/calibration")
        raw = np.asarray(
            self.pairwise_model.predict(frame.loc[:, self.feature_names]), dtype=np.float64
        ).reshape(-1)
        pairwise_bps = self.pairwise_calibration.predict_bps(
            raw, clip_bps=self.config.pairwise_prediction_clip_bps
        ).astype(np.float64)
        if len(pairwise_bps) != len(base):
            raise PairwiseSharedExpertError("target-preserving pairwise prediction lost row alignment")
        return (
            (1.0 - self.config.pairwise_blend_weight) * base.astype(np.float64)
            + self.config.pairwise_blend_weight * pairwise_bps
        ).astype(np.float32)

    def predict_expected_net_bps(self, frame: pd.DataFrame) -> np.ndarray:
        return reconstruct_shared_regime_expected_net_bps(
            frame,
            self.predict_candidate_residual_bps(frame),
            columns=self.columns.shared_columns(),
        )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _utc(value: object, *, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise PairwiseSharedExpertError(f"{name} is not a valid timestamp")
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _utc_series(frame: pd.DataFrame, column: str, *, name: str) -> pd.Series:
    if column not in frame:
        raise PairwiseSharedExpertError(f"training ledger lacks {name} column {column!r}")
    series = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if series.isna().any():
        raise PairwiseSharedExpertError(f"{name} contains invalid timestamps")
    return series


def _finite(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise PairwiseSharedExpertError(f"training ledger lacks required column {column!r}")
    value = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(value).all():
        raise PairwiseSharedExpertError(f"{column!r} must be finite")
    return value


def _strict_true_flag(frame: pd.DataFrame, column: str) -> None:
    if column not in frame:
        raise PairwiseSharedExpertError(f"causal lineage flag {column!r} is missing")
    value = frame[column]
    if value.isna().any() or not value.isin((True, 1)).all():
        raise PairwiseSharedExpertError(
            f"causal lineage flag {column!r} must contain only explicit true booleans"
        )


def _normalise_feature_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(dict.fromkeys(str(name) for name in feature_names if str(name).strip()))
    if not names:
        raise PairwiseSharedExpertError("Round-F shared expert needs a frozen non-empty feature list")
    suspicious = [
        name for name in names
        if any(token in name.lower() for token in _FORBIDDEN_FEATURE_TOKENS)
    ]
    if suspicious:
        raise PairwiseSharedExpertError(
            f"outcome-derived fields cannot enter the Round-F shared expert: {suspicious[:12]}"
        )
    hard = [
        name for name in names
        if any(token in name.lower() for token in _HARD_REGIME_FEATURE_TOKENS)
    ]
    if hard:
        raise PairwiseSharedExpertError(
            "hard regime identifiers cannot enter the Round-F shared expert; "
            f"use causal soft probabilities instead: {hard[:12]}"
        )
    return names


def _validate_feature_frame(frame: pd.DataFrame, names: Sequence[str]) -> None:
    missing = [name for name in names if name not in frame]
    if missing:
        raise PairwiseSharedExpertError(
            f"Round-F shared inference is missing frozen features: {missing[:12]}"
        )
    converted = frame.loc[:, list(names)].apply(pd.to_numeric, errors="coerce")
    if converted.isna().any().any():
        raise PairwiseSharedExpertError("Round-F frozen features must be numeric and finite")
    values = converted.to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise PairwiseSharedExpertError("Round-F frozen features must be numeric and finite")


def _validate_arm(arm: str) -> PairwiseArm:
    if arm not in _PAIRWISE_ARMS:
        raise PairwiseSharedExpertError(
            f"unknown Round-F arm {arm!r}; expected one of {list(_PAIRWISE_ARMS)}"
        )
    return arm  # type: ignore[return-value]


def _validate_training_ledger(
    frame: pd.DataFrame,
    *,
    fit_before_utc: object,
    columns: PairwiseSharedResidualColumns,
    feature_names: Sequence[str],
) -> tuple[pd.Timestamp, pd.Series, pd.Series, np.ndarray, tuple[tuple[str, int], ...]]:
    if frame.empty:
        raise PairwiseSharedExpertError("Round-F shared expert cannot fit an empty ledger")
    cutoff = _utc(fit_before_utc, name="fit_before_utc")
    decision = _utc_series(frame, columns.decision_timestamp, name="decision timestamp")
    available = _utc_series(frame, columns.label_available_timestamp, name="label availability")
    if (available <= decision).any():
        raise PairwiseSharedExpertError("labels must resolve strictly after their decision timestamp")
    if not (decision < cutoff).all():
        raise PairwiseSharedExpertError("Round-F fit includes decisions at/after its cutoff")
    if not (available < cutoff).all():
        raise PairwiseSharedExpertError("Round-F fit includes unresolved/current/future labels")
    if columns.candidate_id not in frame:
        raise PairwiseSharedExpertError(f"training ledger lacks candidate identity {columns.candidate_id!r}")
    candidate = frame[columns.candidate_id].astype(str).str.strip()
    if candidate.eq("").any() or candidate.duplicated().any():
        raise PairwiseSharedExpertError("candidate_id must be non-empty and unique")
    if columns.symbol not in frame:
        raise PairwiseSharedExpertError(f"training ledger lacks symbol identity {columns.symbol!r}")
    symbol = frame[columns.symbol].astype(str).str.strip()
    if symbol.eq("").any():
        raise PairwiseSharedExpertError("symbol identity must be non-empty")
    if columns.side not in frame:
        raise PairwiseSharedExpertError(f"training ledger lacks side column {columns.side!r}")
    side = frame[columns.side].astype(str).str.lower().str.strip()
    if side.eq("").any() or not {"long", "short"}.issubset(set(side)):
        raise PairwiseSharedExpertError("one shared both-side expert requires non-empty long and short rows")
    side_counts = tuple((str(name), int(count)) for name, count in side.value_counts().sort_index().items())
    for flag in (
        columns.base_map_prequential_flag,
        columns.soft_regime_causal_flag,
        columns.cost_atr_causal_flag,
    ):
        _strict_true_flag(frame, flag)
    if (_finite(frame, columns.cost_to_atr) < 0.0).any():
        raise PairwiseSharedExpertError("cost_to_atr must be non-negative")
    exact = _finite(frame, columns.exact_net_bps)
    base = _finite(frame, columns.base_expected_net_bps)
    prior = _finite(frame, columns.regime_prior_residual_bps)
    residual = _finite(frame, columns.candidate_residual_bps)
    # The pair target, calibration target and output correction must all refer
    # to the same common-bps candidate residual—not a side score conversion.
    if not np.allclose(residual, exact - base - prior, rtol=0.0, atol=1e-3):
        raise PairwiseSharedExpertError(
            "candidate_residual_bps must equal exact_net_bps minus causal base and prior residual"
        )
    challenger = challenger_candidate_residual_bps(
        frame,
        columns=ResidualTargetColumns(
            decision_timestamp=columns.decision_timestamp,
            label_available_timestamp=columns.label_available_timestamp,
            side=columns.side,
            exact_net_bps=columns.exact_net_bps,
            base_expected_net_bps=columns.base_expected_net_bps,
            regime_prior_residual_bps=columns.regime_prior_residual_bps,
        ),
    )
    if not np.allclose(residual, challenger, rtol=0.0, atol=1e-3):
        raise PairwiseSharedExpertError("candidate residual disagrees with the Stage-III challenger contract")
    _validate_feature_frame(frame, feature_names)
    return cutoff, decision, available, residual, side_counts


def _effective_pointwise_params(
    params: Mapping[str, Any] | None,
    *,
    config: PairwiseSharedResidualConfig,
) -> dict[str, Any]:
    effective = dict(params or {})
    requested = str(effective.get("objective", "huber")).strip().lower()
    if requested not in {"", "huber"}:
        raise PairwiseSharedExpertError("Round-F pointwise component must use a robust Huber objective")
    estimators = int(effective.get("n_estimators", 128))
    if not 1 <= estimators <= 512:
        raise PairwiseSharedExpertError("pointwise n_estimators must lie in [1, 512]")
    effective.update(
        {
            "objective": "huber",
            "n_estimators": estimators,
            "random_state": int(config.random_state),
            "seed": int(config.random_state),
            "n_jobs": 1,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
        }
    )
    return effective


def _effective_ranker_params(config: PairwiseSharedResidualConfig) -> dict[str, Any]:
    return {
        "objective": "lambdarank",
        "metric": "ndcg",
        "n_estimators": int(config.ranker_estimators),
        "learning_rate": float(config.ranker_learning_rate),
        "num_leaves": int(config.ranker_num_leaves),
        "min_child_samples": int(config.ranker_min_child_samples),
        "reg_lambda": float(config.ranker_l2),
        "random_state": int(config.random_state),
        "seed": int(config.random_state),
        "n_jobs": 1,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }


def _weighted_affine_bps_calibration(
    raw_score: np.ndarray,
    target_bps: np.ndarray,
    *,
    sample_weight: np.ndarray | None,
    ridge: float,
    cutoff: pd.Timestamp,
    max_available: pd.Timestamp,
) -> PairwiseBpsCalibration:
    if raw_score.ndim != 1 or target_bps.shape != raw_score.shape:
        raise PairwiseSharedExpertError("rank-score calibration inputs must be aligned one-dimensional arrays")
    if not (np.isfinite(raw_score).all() and np.isfinite(target_bps).all()):
        raise PairwiseSharedExpertError("rank-score calibration inputs must be finite")
    if sample_weight is None:
        weight = np.ones(len(raw_score), dtype=np.float64)
    else:
        weight = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if len(weight) != len(raw_score) or not np.isfinite(weight).all() or (weight < 0.0).any() or weight.sum() <= 0:
            raise PairwiseSharedExpertError("calibration weights must be aligned, finite and positive in aggregate")
    total = float(weight.sum())
    score_mean = float(np.sum(weight * raw_score) / total)
    centered = raw_score - score_mean
    score_scale = float(np.sqrt(np.sum(weight * centered * centered) / total))
    # Degenerate ranker output should not silently masquerade as a bps signal.
    if not np.isfinite(score_scale) or score_scale <= _EPS:
        raise PairwiseSharedExpertError("pairwise ranker has no score variation for bps calibration")
    z = centered / score_scale
    intercept = float(np.sum(weight * target_bps) / total)
    numerator = float(np.sum(weight * z * (target_bps - intercept)))
    denominator = float(np.sum(weight * z * z) + ridge)
    slope = numerator / max(denominator, _EPS)
    return PairwiseBpsCalibration(
        score_mean=score_mean,
        score_scale=score_scale,
        intercept_bps=intercept,
        slope_bps_per_standard_score=float(slope),
        ridge=float(ridge),
        rows=int(len(raw_score)),
        fit_before_utc=cutoff,
        max_label_available_utc=max_available,
    )


def _pair_ledger_digest(ledger: pd.DataFrame) -> str | None:
    if ledger.empty:
        return None
    fields = [
        "better_candidate_id", "worse_candidate_id", "side_name", "decision_date_utc",
        "residual_gap_bps", "soft_regime_similarity", "base_ev_difference_bps",
        "cost_atr_difference", "pair_direction", "fit_before_utc", "routing",
        *[name for name in ledger.columns if name.startswith("eligible_")],
    ]
    return _sha256(ledger.loc[:, fields].to_dict(orient="records"))


def _pair_support_audit(
    ledger: pd.DataFrame,
    *,
    arm: PairwiseArm,
    separation_bps: float | None,
    config: PairConstructionConfig,
    available: pd.Series,
    candidate_id: pd.Series,
) -> tuple[PairSupportAudit, pd.DataFrame]:
    pair_config = asdict(config)
    if separation_bps is None:
        return PairSupportAudit(
            separation_bps=None,
            pair_selection="disabled_for_F0_pointwise",
            constructed_pairs=0,
            selected_pairs=0,
            selected_pair_rows=0,
            pair_ledger_sha256=None,
            selected_pair_ledger_sha256=None,
            selected_pairs_by_side=(),
            selected_unique_candidates=0,
            max_pair_label_available_utc=None,
            pair_builder_schema="stage_iii_residual_target_challengers_v1",
            pair_builder_routing=ONE_SHARED_MODEL,
            pair_config=pair_config,
        ), ledger.iloc[0:0].copy()
    flag = f"eligible_{int(separation_bps)}bps"
    if flag not in ledger:
        raise PairwiseSharedExpertError(
            f"pair builder did not emit the required {separation_bps:.0f}-bps eligibility flag"
        )
    selected = ledger.loc[ledger[flag].astype(bool)].copy()
    if selected.empty:
        raise PairwiseSharedExpertError(
            f"Round-F {arm} has no bounded context-matched pairs at {separation_bps:.0f} bps"
        )
    pair_candidate_ids = pd.Index(
        pd.concat([selected["better_candidate_id"], selected["worse_candidate_id"]], ignore_index=True)
    ).unique()
    lookup = pd.DataFrame(
        {"candidate_id": candidate_id.to_numpy(), "label_available_ts": available.to_numpy()}
    ).set_index("candidate_id")["label_available_ts"]
    max_available = pd.Timestamp(lookup.loc[pair_candidate_ids].max())
    by_side = tuple(
        (str(side), int(count))
        for side, count in selected["side_name"].value_counts().sort_index().items()
    )
    return PairSupportAudit(
        separation_bps=float(separation_bps),
        pair_selection=f"eligible_{int(separation_bps)}bps_from_context_matched_prior_resolved_ledger",
        constructed_pairs=int(len(ledger)),
        selected_pairs=int(len(selected)),
        selected_pair_rows=int(2 * len(selected)),
        pair_ledger_sha256=_pair_ledger_digest(ledger),
        selected_pair_ledger_sha256=_pair_ledger_digest(selected),
        selected_pairs_by_side=by_side,
        selected_unique_candidates=int(len(pair_candidate_ids)),
        max_pair_label_available_utc=max_available,
        pair_builder_schema="stage_iii_residual_target_challengers_v1",
        pair_builder_routing=ONE_SHARED_MODEL,
        pair_config=pair_config,
    ), selected


def _fit_pairwise_ranker(
    frame: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    selected_pairs: pd.DataFrame,
    sample_weight: np.ndarray | None,
    config: PairwiseSharedResidualConfig,
) -> Any:
    import lightgbm as lgb

    better = selected_pairs["better_position"].to_numpy(np.int64)
    worse = selected_pairs["worse_position"].to_numpy(np.int64)
    if (
        (better < 0).any() or (worse < 0).any()
        or (better >= len(frame)).any() or (worse >= len(frame)).any()
    ):
        raise PairwiseSharedExpertError("pair ledger positions are outside the frozen training ledger")
    # Every generated group is exactly one context-matched better/worse pair.
    # Repeated source rows are intentional: they are the bounded empirical
    # pairwise loss support, not distinct local experts.
    order = np.empty(2 * len(selected_pairs), dtype=np.int64)
    order[0::2] = better
    order[1::2] = worse
    x_pair = frame.iloc[order].loc[:, list(feature_names)].reset_index(drop=True)
    relevance = np.tile(np.asarray([1, 0], dtype=np.int32), len(selected_pairs))
    group = np.full(len(selected_pairs), 2, dtype=np.int32)
    fit_kwargs: dict[str, Any] = {"group": group}
    if sample_weight is not None:
        source_weight = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        pair_weight = np.sqrt(source_weight[better] * source_weight[worse]).astype(np.float32)
        fit_kwargs["sample_weight"] = np.repeat(pair_weight, 2)
    model = lgb.LGBMRanker(**_effective_ranker_params(config))
    model.fit(x_pair, relevance, **fit_kwargs)
    return model


def fit_pairwise_shared_residual_expert(
    frame: pd.DataFrame,
    *,
    arm: PairwiseArm | str,
    feature_names: Sequence[str],
    soft_regime_columns: Sequence[str],
    fit_before_utc: object,
    columns: PairwiseSharedResidualColumns = PairwiseSharedResidualColumns(),
    pair_config: PairConstructionConfig = PairConstructionConfig(),
    config: PairwiseSharedResidualConfig = PairwiseSharedResidualConfig(),
    pointwise_target_mode: Literal["huber", "clipped", "regime_standardized"] = "huber",
    pointwise_params: Mapping[str, Any] | None = None,
    sample_weight: Sequence[float] | None = None,
) -> PairwiseSharedResidualExpertFit:
    """Fit one frozen both-side Round-F arm on strict prior-resolved rows.

    F1/F2 never fit a side-local or regime-local model.  They add a single
    shared LambdaRank head, trained only on the helper's bounded context pairs,
    then map its raw score back to candidate-residual bps using only these
    already-resolved training rows.
    """
    chosen_arm = _validate_arm(str(arm))
    config.validate()
    pair_config.validate()
    names = _normalise_feature_names(feature_names)
    separation = _PAIRWISE_SEPARATION[chosen_arm]
    if separation is not None and config.pairwise_blend_weight <= 0.0:
        raise PairwiseSharedExpertError(
            "F1/F2 require a small positive pairwise_blend_weight"
        )
    if separation is not None and separation not in pair_config.separation_bps:
        raise PairwiseSharedExpertError(
            f"pair_config must predeclare the selected {separation:.0f}-bps threshold"
        )
    cutoff, _decision, available, target_bps, side_counts = _validate_training_ledger(
        frame,
        fit_before_utc=fit_before_utc,
        columns=columns,
        feature_names=names,
    )
    if sample_weight is None:
        weight: np.ndarray | None = None
    else:
        weight = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if len(weight) != len(frame) or not np.isfinite(weight).all() or (weight < 0.0).any() or weight.sum() <= 0.0:
            raise PairwiseSharedExpertError(
                "sample_weight must be finite, non-negative, aligned and positive in aggregate"
            )
    pointwise_effective = _effective_pointwise_params(pointwise_params, config=config)
    try:
        pointwise_fit = fit_shared_regime_residual_expert(
            frame,
            feature_names=names,
            fit_before_utc=cutoff,
            columns=columns.shared_columns(),
            target_mode=pointwise_target_mode,
            sample_weight=weight,
            params=pointwise_effective,
        )
    except SharedResidualExpertError as exc:
        raise PairwiseSharedExpertError(str(exc)) from exc

    # The F0 control deliberately does not inspect outcome-derived pair
    # selection.  Its audit says exactly that pairwise support is disabled.
    full_pair_ledger = pd.DataFrame()
    pair_audit, selected_pairs = _pair_support_audit(
        full_pair_ledger,
        arm=chosen_arm,
        separation_bps=separation,
        config=pair_config,
        available=available,
        candidate_id=frame[columns.candidate_id].astype(str),
    ) if separation is None else (None, None)
    pairwise_model: Any | None = None
    calibration: PairwiseBpsCalibration | None = None
    ranker_params: Mapping[str, Any] | None = None
    if separation is not None:
        try:
            full_pair_ledger = construct_context_matched_residual_pairs(
                frame,
                target_bps,
                soft_regime_columns=soft_regime_columns,
                fit_before_utc=cutoff,
                columns=columns.pair_columns(),
                config=pair_config,
            )
        except StageIIIResidualTargetError as exc:
            raise PairwiseSharedExpertError(str(exc)) from exc
        pair_audit, selected_pairs = _pair_support_audit(
            full_pair_ledger,
            arm=chosen_arm,
            separation_bps=separation,
            config=pair_config,
            available=available,
            candidate_id=frame[columns.candidate_id].astype(str),
        )
        pairwise_model = _fit_pairwise_ranker(
            frame,
            feature_names=names,
            selected_pairs=selected_pairs,
            sample_weight=weight,
            config=config,
        )
        raw_train = np.asarray(
            pairwise_model.predict(frame.loc[:, names]), dtype=np.float64
        ).reshape(-1)
        calibration = _weighted_affine_bps_calibration(
            raw_train,
            target_bps,
            sample_weight=weight,
            ridge=config.calibration_ridge,
            cutoff=cutoff,
            max_available=pd.Timestamp(available.max()),
        )
        ranker_params = _effective_ranker_params(config)
    assert pair_audit is not None and selected_pairs is not None

    candidate_ids = frame[columns.candidate_id].astype(str).tolist()
    audit = PairwiseSharedResidualAudit(
        schema=SCHEMA,
        arm=chosen_arm,
        routing=ONE_SHARED_BOTH_SIDE_MODEL,
        reconstruction=COMMON_BPS_RECONSTRUCTION,
        feature_names=names,
        feature_sha256=_sha256(list(names)),
        training_row_count=int(len(frame)),
        training_candidate_ids_sha256=_sha256(candidate_ids),
        training_rows_by_side=side_counts,
        training_cutoff_utc=cutoff,
        max_label_available_utc=pd.Timestamp(available.max()),
        pointwise_target_mode=pointwise_target_mode,
        pointwise_model_class=type(pointwise_fit.model).__name__,
        pointwise_params=pointwise_effective,
        pairwise_model_class=None if pairwise_model is None else type(pairwise_model).__name__,
        pairwise_params=ranker_params,
        pairwise_config=asdict(config),
        pair_support=pair_audit,
        pairwise_calibration=calibration,
    )
    return PairwiseSharedResidualExpertFit(
        arm=chosen_arm,
        pointwise_fit=pointwise_fit,
        pairwise_model=pairwise_model,
        pairwise_calibration=calibration,
        config=config,
        audit=audit,
        columns=columns,
    )


def _audit_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    serializer = getattr(value, "to_dict", None)
    if callable(serializer):
        payload = serializer()
        if isinstance(payload, Mapping):
            return dict(payload)
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise PairwiseSharedExpertError(
        "preserved base must expose a serialisable frozen target audit"
    )


def _payload_utc(payload: Mapping[str, Any], key: str) -> pd.Timestamp:
    if key not in payload:
        raise PairwiseSharedExpertError(f"preserved base audit lacks {key!r}")
    return _utc(payload[key], name=f"preserved base audit {key}")


def _expected_preserved_target_digest(
    *, base_arm: str, candidate_residual: np.ndarray
) -> str:
    if base_arm == "T3_ordinal":
        labels = np.digitize(candidate_residual, ORDINAL_EDGES_BPS, right=True).astype(int)
        return _sha256(labels.tolist())
    if base_arm == "T4_quantile":
        return _sha256(np.round(candidate_residual, 8).tolist())
    raise PairwiseSharedExpertError(
        "target-preserving Round-F adapter accepts only T3_ordinal or T4_quantile bases"
    )


def _preserved_base_target_audit(
    base_model: Any,
    *,
    frame: pd.DataFrame,
    feature_names: tuple[str, ...],
    cutoff: pd.Timestamp,
    available: pd.Series,
    candidate_residual: np.ndarray,
    columns: PairwiseSharedResidualColumns,
) -> tuple[PreservedBaseTargetAudit, np.ndarray]:
    """Validate one immutable T3/T4 model before adding any pairwise head."""
    audit_source = getattr(base_model, "audit", None)
    payload = _audit_payload(audit_source)
    base_arm = str(payload.get("arm", ""))
    if base_arm not in {"T3_ordinal", "T4_quantile"}:
        raise PairwiseSharedExpertError(
            "preserved base audit must declare T3_ordinal or T4_quantile; target substitution is forbidden"
        )
    if str(payload.get("routing", "")) != ONE_SHARED_BOTH_SIDE_MODEL:
        raise PairwiseSharedExpertError("preserved base target is not one shared both-side model")
    if str(payload.get("reconstruction", "")) != COMMON_BPS_RECONSTRUCTION:
        raise PairwiseSharedExpertError("preserved base target does not reconstruct common candidate-residual bps")
    audited_features = tuple(str(name) for name in payload.get("feature_names", ()))
    if audited_features != feature_names:
        raise PairwiseSharedExpertError("preserved base feature contract does not match the pairwise adapter")
    expected_feature_sha = _sha256(list(feature_names))
    if str(payload.get("feature_sha256", "")) != expected_feature_sha:
        raise PairwiseSharedExpertError("preserved base feature digest does not match frozen features")
    model_features = tuple(str(name) for name in getattr(base_model, "feature_names", ()))
    if model_features != feature_names:
        raise PairwiseSharedExpertError("preserved base model feature list does not match its audit")
    base_cutoff = _payload_utc(payload, "training_cutoff_utc")
    base_max_available = _payload_utc(payload, "max_label_available_utc")
    if base_cutoff != cutoff or base_max_available != pd.Timestamp(available.max()):
        raise PairwiseSharedExpertError(
            "preserved base cutoff/label support does not exactly match the pairwise training ledger"
        )
    if int(payload.get("training_row_count", -1)) != len(frame):
        raise PairwiseSharedExpertError("preserved base training row count does not match adapter ledger")
    candidate_ids = frame[columns.candidate_id].astype(str).tolist()
    candidate_digest = _sha256(candidate_ids)
    if str(payload.get("training_candidate_ids_sha256", "")) != candidate_digest:
        raise PairwiseSharedExpertError("preserved base candidate identity support does not match adapter ledger")
    expected_target_digest = _expected_preserved_target_digest(
        base_arm=base_arm, candidate_residual=candidate_residual
    )
    if str(payload.get("target_label_sha256", "")) != expected_target_digest:
        raise PairwiseSharedExpertError(
            "preserved base target-label digest does not match the candidate-residual target; substitution rejected"
        )
    predictor = getattr(base_model, "predict_candidate_residual_bps", None)
    if not callable(predictor):
        raise PairwiseSharedExpertError("preserved base must expose predict_candidate_residual_bps")
    base_prediction = np.asarray(predictor(frame)).reshape(-1)
    if len(base_prediction) != len(frame) or not np.isfinite(base_prediction).all():
        raise PairwiseSharedExpertError("preserved base prediction is not finite/aligned on its audited support")
    formulation = str(payload.get("formulation", ""))
    if not formulation:
        raise PairwiseSharedExpertError("preserved base audit lacks target formulation")
    return PreservedBaseTargetAudit(
        base_model_class=type(base_model).__name__,
        base_target_arm=base_arm,
        base_target_formulation=formulation,
        base_audit_sha256=_sha256(payload),
        base_feature_sha256=expected_feature_sha,
        base_target_label_sha256=expected_target_digest,
        base_training_prediction_sha256=_sha256(
            np.round(base_prediction.astype(np.float64), 8).tolist()
        ),
        base_training_cutoff_utc=base_cutoff,
        base_max_label_available_utc=base_max_available,
        base_training_rows=int(len(frame)),
        base_training_candidate_ids_sha256=candidate_digest,
    ), base_prediction


def fit_target_preserving_pairwise_adapter(
    frame: pd.DataFrame,
    *,
    base_model: Any,
    arm: PairwiseArm | str,
    feature_names: Sequence[str],
    soft_regime_columns: Sequence[str],
    fit_before_utc: object,
    columns: PairwiseSharedResidualColumns = PairwiseSharedResidualColumns(),
    pair_config: PairConstructionConfig = PairConstructionConfig(),
    config: PairwiseSharedResidualConfig = PairwiseSharedResidualConfig(),
    sample_weight: Sequence[float] | None = None,
) -> TargetPreservingPairwiseAdapterFit:
    """Add F1/F2 ranking evidence without fitting or changing the T3/T4 target.

    The only supervised model trained here is the shared bounded LambdaRank
    head.  F0 returns the supplied target model's candidate-residual bps exactly
    unchanged; F1/F2 blend it with a small ranker-derived bps correction.
    """
    chosen_arm = _validate_arm(str(arm))
    config.validate()
    pair_config.validate()
    names = _normalise_feature_names(feature_names)
    separation = _PAIRWISE_SEPARATION[chosen_arm]
    if separation is not None and config.pairwise_blend_weight <= 0.0:
        raise PairwiseSharedExpertError("F1/F2 require a small positive pairwise_blend_weight")
    if separation is not None and separation not in pair_config.separation_bps:
        raise PairwiseSharedExpertError(
            f"pair_config must predeclare the selected {separation:.0f}-bps threshold"
        )
    cutoff, _decision, available, target_bps, side_counts = _validate_training_ledger(
        frame,
        fit_before_utc=fit_before_utc,
        columns=columns,
        feature_names=names,
    )
    if sample_weight is None:
        weight: np.ndarray | None = None
    else:
        weight = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if (
            len(weight) != len(frame) or not np.isfinite(weight).all()
            or (weight < 0.0).any() or weight.sum() <= 0.0
        ):
            raise PairwiseSharedExpertError(
                "sample_weight must be finite, non-negative, aligned and positive in aggregate"
            )
    base_audit, _base_training_prediction = _preserved_base_target_audit(
        base_model,
        frame=frame,
        feature_names=names,
        cutoff=cutoff,
        available=available,
        candidate_residual=target_bps,
        columns=columns,
    )
    full_pair_ledger = pd.DataFrame()
    pair_audit, selected_pairs = _pair_support_audit(
        full_pair_ledger,
        arm=chosen_arm,
        separation_bps=separation,
        config=pair_config,
        available=available,
        candidate_id=frame[columns.candidate_id].astype(str),
    ) if separation is None else (None, None)
    pairwise_model: Any | None = None
    calibration: PairwiseBpsCalibration | None = None
    ranker_params: Mapping[str, Any] | None = None
    if separation is not None:
        try:
            full_pair_ledger = construct_context_matched_residual_pairs(
                frame,
                target_bps,
                soft_regime_columns=soft_regime_columns,
                fit_before_utc=cutoff,
                columns=columns.pair_columns(),
                config=pair_config,
            )
        except StageIIIResidualTargetError as exc:
            raise PairwiseSharedExpertError(str(exc)) from exc
        pair_audit, selected_pairs = _pair_support_audit(
            full_pair_ledger,
            arm=chosen_arm,
            separation_bps=separation,
            config=pair_config,
            available=available,
            candidate_id=frame[columns.candidate_id].astype(str),
        )
        pairwise_model = _fit_pairwise_ranker(
            frame,
            feature_names=names,
            selected_pairs=selected_pairs,
            sample_weight=weight,
            config=config,
        )
        raw_train = np.asarray(pairwise_model.predict(frame.loc[:, names]), dtype=np.float64).reshape(-1)
        calibration = _weighted_affine_bps_calibration(
            raw_train,
            target_bps,
            sample_weight=weight,
            ridge=config.calibration_ridge,
            cutoff=cutoff,
            max_available=pd.Timestamp(available.max()),
        )
        ranker_params = _effective_ranker_params(config)
    assert pair_audit is not None and selected_pairs is not None
    audit = TargetPreservingPairwiseAudit(
        schema=SCHEMA,
        arm=chosen_arm,
        routing=ONE_SHARED_BOTH_SIDE_MODEL,
        reconstruction=COMMON_BPS_RECONSTRUCTION,
        feature_names=names,
        feature_sha256=_sha256(list(names)),
        training_row_count=int(len(frame)),
        training_candidate_ids_sha256=_sha256(frame[columns.candidate_id].astype(str).tolist()),
        training_rows_by_side=side_counts,
        training_cutoff_utc=cutoff,
        max_label_available_utc=pd.Timestamp(available.max()),
        preserved_base_target=base_audit,
        candidate_residual_label_sha256=_sha256(np.round(target_bps, 8).tolist()),
        pairwise_model_class=None if pairwise_model is None else type(pairwise_model).__name__,
        pairwise_params=ranker_params,
        pairwise_config=asdict(config),
        pair_support=pair_audit,
        pairwise_calibration=calibration,
    )
    return TargetPreservingPairwiseAdapterFit(
        arm=chosen_arm,
        base_model=base_model,
        feature_names=names,
        pairwise_model=pairwise_model,
        pairwise_calibration=calibration,
        config=config,
        audit=audit,
        columns=columns,
    )


__all__ = [
    "SCHEMA", "COMMON_BPS_RECONSTRUCTION", "ONE_SHARED_BOTH_SIDE_MODEL",
    "PairwiseArm", "PairwiseBpsCalibration", "PairwiseSharedExpertError",
    "PairwiseSharedResidualAudit", "PairwiseSharedResidualColumns",
    "PairwiseSharedResidualConfig", "PairwiseSharedResidualExpertFit",
    "PairSupportAudit", "PreservedBaseTargetAudit", "TargetPreservingPairwiseAdapterFit",
    "TargetPreservingPairwiseAudit", "fit_pairwise_shared_residual_expert",
    "fit_target_preserving_pairwise_adapter",
]
