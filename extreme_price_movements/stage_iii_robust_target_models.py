"""Actual shared robust-target model fits for Stage-III Round T.

The target challenger module owns label semantics and bps reconstruction.  This
module turns its two non-regression challengers into bounded deterministic
LightGBM fits, while retaining one model family over both sides:

* ``T3`` is a cumulative ordinal formulation: three shared binary models for
  ``P(candidate_residual <= edge)`` at the predeclared residual edges.  It is
  intentionally *not* a nominal four-class model.
* ``T4`` is five independent shared conditional-quantile heads
  ``q10/q25/q50/q75/q90``.  Inference always applies the challenger's
  deterministic crossing repair before exposing the median/residual bps.

Neither target formulation creates a side- or regime-local model.  All target
statistics, class means, labels and calibrations are frozen from a ledger whose
labels resolved strictly before the supplied UTC fit cutoff.  Both outputs map
back to candidate-residual bps and then to the canonical common expected-net
bps score, suitable for one pooled-global ranking.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .stage_iii_residual_target_challengers import (
    ONE_SHARED_MODEL,
    ORDINAL_EDGES_BPS,
    QUANTILE_LEVELS,
    OrdinalResidualTargetFit,
    QuantileResidualTargetFit,
    ResidualTargetColumns,
    StageIIIResidualTargetError,
    candidate_residual_bps,
    fit_quantile_residual_targets,
    fit_regime_centered_ordinal_residual,
    reconstruct_expected_net_bps,
    reconstruct_ordinal_candidate_residual_bps,
    reconstruct_quantile_residual_outputs,
)


SCHEMA = "stage_iii_robust_target_models_v1"
ONE_SHARED_BOTH_SIDE_MODEL = "one_shared_both_side_model_no_local_or_hard_routing"
COMMON_BPS_RECONSTRUCTION = (
    "frozen_base_expected_net_bps_plus_prior_resolved_soft_regime_prior_residual_bps_"
    "plus_predicted_candidate_residual_bps"
)
ORDINAL_FORMULATION = "cumulative_three_threshold_binary_heads_not_nominal_multiclass"
QUANTILE_FORMULATION = "five_shared_conditional_quantile_heads_with_deterministic_crossing_repair"
_EPS = 1e-12
_FORBIDDEN_FEATURE_TOKENS = (
    "exact_net", "realised_net", "realized_net", "outcome_resolved",
    "candidate_residual", "target", "label", "future_", "mfe", "mae",
)
_HARD_REGIME_FEATURE_TOKENS = (
    "regime_id", "regime_code", "regime_class", "hard_regime", "argmax_regime",
)


class RobustTargetModelError(ValueError):
    """Raised when an ordinal/quantile shared-target contract is violated."""


@dataclass(frozen=True)
class RobustTargetColumns:
    """Identity and lineage columns required by the challenger target contract."""

    decision_timestamp: str = "decision_ts"
    label_available_timestamp: str = "label_available_ts"
    side: str = "side_name"
    candidate_id: str = "candidate_id"
    exact_net_bps: str = "exact_net_bps"
    base_expected_net_bps: str = "prequential_base_expected_net_bps"
    regime_prior_residual_bps: str = "prequential_soft_regime_prior_residual_bps"
    base_map_prequential_flag: str = "base_map_is_prequential"
    base_map_source_side: str = "base_map_source_side"
    base_map_max_label_available_timestamp: str = "base_map_max_label_available_ts"
    soft_regime_causal_flag: str = "soft_regime_is_causal_prequential"
    soft_regime_fit_end_timestamp: str = "soft_regime_fit_end_ts"
    regime_prior_max_label_available_timestamp: str = "prior_resolved_max_label_available_ts"

    def target_columns(self) -> ResidualTargetColumns:
        return ResidualTargetColumns(
            decision_timestamp=self.decision_timestamp,
            label_available_timestamp=self.label_available_timestamp,
            side=self.side,
            exact_net_bps=self.exact_net_bps,
            base_expected_net_bps=self.base_expected_net_bps,
            regime_prior_residual_bps=self.regime_prior_residual_bps,
            base_map_prequential_flag=self.base_map_prequential_flag,
            base_map_source_side=self.base_map_source_side,
            base_map_max_label_available_timestamp=self.base_map_max_label_available_timestamp,
            soft_regime_causal_flag=self.soft_regime_causal_flag,
            soft_regime_fit_end_timestamp=self.soft_regime_fit_end_timestamp,
            regime_prior_max_label_available_timestamp=self.regime_prior_max_label_available_timestamp,
        )


@dataclass(frozen=True)
class RobustTargetModelConfig:
    """No-HPO, deterministic capacity limits for a frozen Round-T fit."""

    n_estimators: int = 128
    learning_rate: float = 0.035
    num_leaves: int = 15
    min_child_samples: int = 32
    l2_regularization: float = 4.0
    random_state: int = 1729

    def validate(self) -> None:
        if not 1 <= self.n_estimators <= 512:
            raise RobustTargetModelError("n_estimators must lie in [1, 512]")
        if not 0.001 <= self.learning_rate <= 0.25:
            raise RobustTargetModelError("learning_rate must lie in [0.001, 0.25]")
        if not 2 <= self.num_leaves <= 64:
            raise RobustTargetModelError("num_leaves must lie in [2, 64]")
        if not 1 <= self.min_child_samples <= 1_024:
            raise RobustTargetModelError("min_child_samples must lie in [1, 1024]")
        if self.l2_regularization < 0.0:
            raise RobustTargetModelError("l2_regularization must be non-negative")


@dataclass(frozen=True)
class OrdinalCumulativeHeadAudit:
    """One shared cumulative threshold head, including frozen support."""

    threshold_index: int
    threshold_edge_bps: float
    event_definition: str
    negative_support: int
    positive_support: int
    model_class: str


@dataclass(frozen=True)
class RobustTargetAudit:
    """Exact model/feature/cutoff target audit for T3 or T4."""

    schema: str
    arm: Literal["T3_ordinal", "T4_quantile"]
    routing: str
    formulation: str
    reconstruction: str
    feature_names: tuple[str, ...]
    feature_sha256: str
    training_row_count: int
    training_candidate_ids_sha256: str
    training_rows_by_side: tuple[tuple[str, int], ...]
    training_cutoff_utc: pd.Timestamp
    max_label_available_utc: pd.Timestamp
    target_label_sha256: str
    target_support: tuple[int, ...]
    model_params: Mapping[str, Any]
    ordinal_heads: tuple[OrdinalCumulativeHeadAudit, ...] = ()
    quantile_heads: tuple[str, ...] = ()
    target_challenger_schema: str = "stage_iii_residual_target_challengers_v1"
    target_challenger_routing: str = ONE_SHARED_MODEL

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["feature_names"] = list(self.feature_names)
        result["training_rows_by_side"] = [list(value) for value in self.training_rows_by_side]
        result["training_cutoff_utc"] = self.training_cutoff_utc.isoformat()
        result["max_label_available_utc"] = self.max_label_available_utc.isoformat()
        result["model_params"] = dict(self.model_params)
        result["ordinal_heads"] = [asdict(head) for head in self.ordinal_heads]
        result["quantile_heads"] = list(self.quantile_heads)
        return result


@dataclass(frozen=True)
class OrdinalCumulativeHead:
    """A binary CDF head belonging to one shared ordinal target model."""

    edge_bps: float
    model: Any


@dataclass(frozen=True)
class OrdinalSharedRobustTargetFit:
    """T3: three shared cumulative CDF heads and frozen class economics."""

    heads: tuple[OrdinalCumulativeHead, ...]
    target_fit: OrdinalResidualTargetFit
    feature_names: tuple[str, ...]
    audit: RobustTargetAudit
    columns: RobustTargetColumns = RobustTargetColumns()

    def predict_outputs(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return repaired CDF/classes plus candidate-residual bps on frozen inputs."""
        _validate_feature_frame(frame, self.feature_names)
        raw_columns: list[np.ndarray] = []
        for head in self.heads:
            probability = _binary_positive_probability(head.model, frame.loc[:, self.feature_names])
            raw_columns.append(probability)
        raw_cdf = np.column_stack(raw_columns)
        repaired_cdf, repaired = repair_cumulative_ordinal_probabilities(raw_cdf)
        probability = _cdf_to_class_probabilities(repaired_cdf)
        try:
            residual = reconstruct_ordinal_candidate_residual_bps(probability, self.target_fit)
        except StageIIIResidualTargetError as exc:
            raise RobustTargetModelError(str(exc)) from exc
        output = pd.DataFrame(index=frame.index)
        for index, edge in enumerate(self.target_fit.edges_bps):
            output[f"cdf_candidate_residual_le_{_edge_name(edge)}bps"] = repaired_cdf[:, index].astype(np.float32)
        for klass in range(self.target_fit.class_count):
            output[f"ordinal_class_{klass}_probability"] = probability[:, klass].astype(np.float32)
        output["ordinal_cdf_crossing_repaired"] = repaired
        output["candidate_residual_bps"] = residual
        return output

    def predict_candidate_residual_bps(self, frame: pd.DataFrame) -> np.ndarray:
        return self.predict_outputs(frame)["candidate_residual_bps"].to_numpy(np.float32)

    def predict_expected_net_bps(self, frame: pd.DataFrame) -> np.ndarray:
        try:
            return reconstruct_expected_net_bps(
                frame,
                self.predict_candidate_residual_bps(frame),
                columns=self.columns.target_columns(),
            )
        except StageIIIResidualTargetError as exc:
            raise RobustTargetModelError(str(exc)) from exc


@dataclass(frozen=True)
class QuantileSharedRobustTargetFit:
    """T4: five shared conditional-quantile heads with repaired inference."""

    models: Mapping[str, Any]
    target_fit: QuantileResidualTargetFit
    feature_names: tuple[str, ...]
    audit: RobustTargetAudit
    columns: RobustTargetColumns = RobustTargetColumns()

    def predict_outputs(self, frame: pd.DataFrame) -> pd.DataFrame:
        _validate_feature_frame(frame, self.feature_names)
        raw = {
            name: np.asarray(model.predict(frame.loc[:, self.feature_names]), dtype=np.float64).reshape(-1)
            for name, model in self.models.items()
        }
        try:
            output = reconstruct_quantile_shared_outputs(raw, self.target_fit)
        except StageIIIResidualTargetError as exc:
            raise RobustTargetModelError(str(exc)) from exc
        output.index = frame.index
        return output

    def predict_candidate_residual_bps(self, frame: pd.DataFrame) -> np.ndarray:
        return self.predict_outputs(frame)["candidate_residual_median_bps"].to_numpy(np.float32)

    def predict_expected_net_bps(self, frame: pd.DataFrame) -> np.ndarray:
        try:
            return reconstruct_expected_net_bps(
                frame,
                self.predict_candidate_residual_bps(frame),
                columns=self.columns.target_columns(),
            )
        except StageIIIResidualTargetError as exc:
            raise RobustTargetModelError(str(exc)) from exc


def _canonical_json(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _utc(value: object, *, name: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise RobustTargetModelError(f"{name} is not a valid timestamp")
    return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")


def _utc_series(frame: pd.DataFrame, column: str, *, name: str) -> pd.Series:
    if column not in frame:
        raise RobustTargetModelError(f"ledger lacks {name} column {column!r}")
    result = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if result.isna().any():
        raise RobustTargetModelError(f"{name} contains invalid timestamps")
    return result


def _normalise_feature_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(dict.fromkeys(str(name) for name in feature_names if str(name).strip()))
    if not names:
        raise RobustTargetModelError("shared target model requires a frozen non-empty feature list")
    forbidden = [
        name for name in names
        if any(token in name.lower() for token in _FORBIDDEN_FEATURE_TOKENS)
    ]
    if forbidden:
        raise RobustTargetModelError(
            f"outcome-derived fields cannot enter the shared target model: {forbidden[:12]}"
        )
    hard_regime = [
        name for name in names
        if any(token in name.lower() for token in _HARD_REGIME_FEATURE_TOKENS)
    ]
    if hard_regime:
        raise RobustTargetModelError(
            "hard regime identifiers cannot enter the shared target model; "
            f"use causal soft probabilities instead: {hard_regime[:12]}"
        )
    return names


def _validate_feature_frame(frame: pd.DataFrame, feature_names: Sequence[str]) -> None:
    missing = [name for name in feature_names if name not in frame]
    if missing:
        raise RobustTargetModelError(
            f"shared target inference is missing frozen features: {missing[:12]}"
        )
    values = frame.loc[:, list(feature_names)].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise RobustTargetModelError("shared target frozen features must be numeric and finite")


def _validate_training_ledger(
    frame: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    fit_before_utc: object,
    columns: RobustTargetColumns,
) -> tuple[pd.Timestamp, pd.Series, tuple[tuple[str, int], ...]]:
    if frame.empty:
        raise RobustTargetModelError("shared target model cannot fit an empty ledger")
    cutoff = _utc(fit_before_utc, name="fit_before_utc")
    decision = _utc_series(frame, columns.decision_timestamp, name="decision timestamp")
    available = _utc_series(frame, columns.label_available_timestamp, name="label availability")
    if (available <= decision).any():
        raise RobustTargetModelError("labels must resolve strictly after their decision timestamp")
    if not (decision < cutoff).all():
        raise RobustTargetModelError("shared target fit includes decisions at/after its cutoff")
    if not (available < cutoff).all():
        raise RobustTargetModelError("shared target fit includes unresolved/current/future labels")
    if columns.candidate_id not in frame:
        raise RobustTargetModelError(f"ledger lacks candidate identity {columns.candidate_id!r}")
    candidate = frame[columns.candidate_id].astype(str).str.strip()
    if candidate.eq("").any() or candidate.duplicated().any():
        raise RobustTargetModelError("candidate_id must be non-empty and unique")
    if columns.side not in frame:
        raise RobustTargetModelError(f"ledger lacks side column {columns.side!r}")
    side = frame[columns.side].astype(str).str.lower().str.strip()
    if side.eq("").any() or not {"long", "short"}.issubset(set(side)):
        raise RobustTargetModelError("one shared both-side model requires non-empty long and short rows")
    _validate_feature_frame(frame, feature_names)
    counts = tuple((str(key), int(value)) for key, value in side.value_counts().sort_index().items())
    return cutoff, available, counts


def _validate_sample_weight(sample_weight: Sequence[float] | None, rows: int) -> np.ndarray | None:
    if sample_weight is None:
        return None
    weight = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if (
        len(weight) != rows or not np.isfinite(weight).all()
        or (weight < 0.0).any() or weight.sum() <= 0.0
    ):
        raise RobustTargetModelError(
            "sample_weight must be aligned, finite, non-negative and positive in aggregate"
        )
    return weight.astype(np.float32)


def _model_params(config: RobustTargetModelConfig, *, objective: str, alpha: float | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {
        "objective": objective,
        "n_estimators": int(config.n_estimators),
        "learning_rate": float(config.learning_rate),
        "num_leaves": int(config.num_leaves),
        "min_child_samples": int(config.min_child_samples),
        "reg_lambda": float(config.l2_regularization),
        "random_state": int(config.random_state),
        "seed": int(config.random_state),
        "n_jobs": 1,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    if alpha is not None:
        params["alpha"] = float(alpha)
    return params


def _edge_name(edge: float) -> str:
    return f"m{abs(int(edge))}" if edge < 0 else str(int(edge))


def _binary_positive_probability(model: Any, x: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise RobustTargetModelError("ordinal cumulative head does not expose binary probabilities")
    probability = np.asarray(model.predict_proba(x), dtype=np.float64)
    if probability.ndim != 2 or probability.shape[1] != 2:
        raise RobustTargetModelError("ordinal cumulative head must emit exactly two probabilities")
    positive = probability[:, 1].reshape(-1)
    if not np.isfinite(positive).all():
        raise RobustTargetModelError("ordinal cumulative head returned non-finite probabilities")
    return np.clip(positive, 0.0, 1.0)


def repair_cumulative_ordinal_probabilities(
    cumulative_probabilities: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Repair only CDF crossing; no target/outcome information is consulted."""
    raw = np.asarray(cumulative_probabilities, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != len(ORDINAL_EDGES_BPS):
        raise RobustTargetModelError("cumulative ordinal probabilities must have three threshold columns")
    if not np.isfinite(raw).all():
        raise RobustTargetModelError("cumulative ordinal probabilities must be finite")
    clipped = np.clip(raw, 0.0, 1.0)
    repaired = np.maximum.accumulate(clipped, axis=1)
    crossing = np.any(np.abs(repaired - clipped) > 1e-12, axis=1)
    return repaired.astype(np.float64), crossing


def _cdf_to_class_probabilities(cdf: np.ndarray) -> np.ndarray:
    if cdf.ndim != 2 or cdf.shape[1] != len(ORDINAL_EDGES_BPS):
        raise RobustTargetModelError("ordinal CDF must have three ordered threshold columns")
    probabilities = np.column_stack(
        [cdf[:, 0], cdf[:, 1] - cdf[:, 0], cdf[:, 2] - cdf[:, 1], 1.0 - cdf[:, 2]]
    )
    probabilities = np.clip(probabilities, 0.0, 1.0)
    total = probabilities.sum(axis=1, keepdims=True)
    if (total <= _EPS).any() or not np.isfinite(total).all():
        raise RobustTargetModelError("ordinal CDF could not be converted to a probability simplex")
    return probabilities / total


def reconstruct_quantile_shared_outputs(
    predictions: Mapping[str, Sequence[float]],
    target_fit: QuantileResidualTargetFit,
) -> pd.DataFrame:
    """Use the challenger's deterministic q10→q90 crossing repair contract."""
    return reconstruct_quantile_residual_outputs(predictions, target_fit, repair_crossing=True)


def fit_ordinal_shared_robust_target(
    frame: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    fit_before_utc: object,
    columns: RobustTargetColumns = RobustTargetColumns(),
    config: RobustTargetModelConfig = RobustTargetModelConfig(),
    sample_weight: Sequence[float] | None = None,
) -> OrdinalSharedRobustTargetFit:
    """Fit T3 as three shared cumulative ordinal LightGBM heads.

    Head ``k`` learns ``P(Y <= edge_k)``.  This cumulative construction
    respects the order of large overestimate → approximately correct → large
    underestimate, unlike a nominal four-class softmax.
    """
    import lightgbm as lgb

    config.validate()
    names = _normalise_feature_names(feature_names)
    cutoff, available, side_counts = _validate_training_ledger(
        frame, feature_names=names, fit_before_utc=fit_before_utc, columns=columns
    )
    target_columns = columns.target_columns()
    try:
        target_fit, labels = fit_regime_centered_ordinal_residual(
            frame, fit_before_utc=cutoff, columns=target_columns
        )
    except StageIIIResidualTargetError as exc:
        raise RobustTargetModelError(str(exc)) from exc
    weight = _validate_sample_weight(sample_weight, len(frame))
    heads: list[OrdinalCumulativeHead] = []
    audits: list[OrdinalCumulativeHeadAudit] = []
    x = frame.loc[:, names]
    for index, edge in enumerate(target_fit.edges_bps):
        event = (labels <= index).astype(np.int8)
        positive = int(event.sum())
        negative = int(len(event) - positive)
        if positive == 0 or negative == 0:
            raise RobustTargetModelError(
                f"ordinal edge {edge:g} bps has no two-class support for its cumulative head"
            )
        model = lgb.LGBMClassifier(**_model_params(config, objective="binary"))
        fit_kwargs: dict[str, Any] = {}
        if weight is not None:
            fit_kwargs["sample_weight"] = weight
        model.fit(x, event, **fit_kwargs)
        heads.append(OrdinalCumulativeHead(edge_bps=float(edge), model=model))
        audits.append(
            OrdinalCumulativeHeadAudit(
                threshold_index=index,
                threshold_edge_bps=float(edge),
                event_definition=f"candidate_residual_bps <= {edge:g}",
                negative_support=negative,
                positive_support=positive,
                model_class=type(model).__name__,
            )
        )
    audit = RobustTargetAudit(
        schema=SCHEMA,
        arm="T3_ordinal",
        routing=ONE_SHARED_BOTH_SIDE_MODEL,
        formulation=ORDINAL_FORMULATION,
        reconstruction=COMMON_BPS_RECONSTRUCTION,
        feature_names=names,
        feature_sha256=_digest(list(names)),
        training_row_count=int(len(frame)),
        training_candidate_ids_sha256=_digest(frame[columns.candidate_id].astype(str).tolist()),
        training_rows_by_side=side_counts,
        training_cutoff_utc=cutoff,
        max_label_available_utc=pd.Timestamp(available.max()),
        target_label_sha256=_digest(labels.astype(int).tolist()),
        target_support=target_fit.class_support,
        model_params=_model_params(config, objective="binary"),
        ordinal_heads=tuple(audits),
    )
    return OrdinalSharedRobustTargetFit(
        heads=tuple(heads),
        target_fit=target_fit,
        feature_names=names,
        audit=audit,
        columns=columns,
    )


def fit_quantile_shared_robust_target(
    frame: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    fit_before_utc: object,
    columns: RobustTargetColumns = RobustTargetColumns(),
    config: RobustTargetModelConfig = RobustTargetModelConfig(),
    sample_weight: Sequence[float] | None = None,
) -> QuantileSharedRobustTargetFit:
    """Fit T4 as five shared conditional-quantile LightGBM residual heads."""
    import lightgbm as lgb

    config.validate()
    names = _normalise_feature_names(feature_names)
    cutoff, available, side_counts = _validate_training_ledger(
        frame, feature_names=names, fit_before_utc=fit_before_utc, columns=columns
    )
    target_columns = columns.target_columns()
    try:
        target_fit, targets = fit_quantile_residual_targets(
            frame, fit_before_utc=cutoff, columns=target_columns
        )
        residual = candidate_residual_bps(frame, columns=target_columns)
    except StageIIIResidualTargetError as exc:
        raise RobustTargetModelError(str(exc)) from exc
    weight = _validate_sample_weight(sample_weight, len(frame))
    models: dict[str, Any] = {}
    x = frame.loc[:, names]
    for quantile, name in zip(target_fit.quantiles, target_fit.head_names):
        model = lgb.LGBMRegressor(**_model_params(config, objective="quantile", alpha=quantile))
        fit_kwargs: dict[str, Any] = {}
        if weight is not None:
            fit_kwargs["sample_weight"] = weight
        model.fit(x, np.asarray(targets[name], dtype=np.float32), **fit_kwargs)
        models[name] = model
    audit = RobustTargetAudit(
        schema=SCHEMA,
        arm="T4_quantile",
        routing=ONE_SHARED_BOTH_SIDE_MODEL,
        formulation=QUANTILE_FORMULATION,
        reconstruction=COMMON_BPS_RECONSTRUCTION,
        feature_names=names,
        feature_sha256=_digest(list(names)),
        training_row_count=int(len(frame)),
        training_candidate_ids_sha256=_digest(frame[columns.candidate_id].astype(str).tolist()),
        training_rows_by_side=side_counts,
        training_cutoff_utc=cutoff,
        max_label_available_utc=pd.Timestamp(available.max()),
        target_label_sha256=_digest(np.round(residual, 8).tolist()),
        target_support=tuple(int(len(residual)) for _ in QUANTILE_LEVELS),
        model_params=_model_params(config, objective="quantile"),
        quantile_heads=target_fit.head_names,
    )
    return QuantileSharedRobustTargetFit(
        models=models,
        target_fit=target_fit,
        feature_names=names,
        audit=audit,
        columns=columns,
    )


__all__ = [
    "SCHEMA", "COMMON_BPS_RECONSTRUCTION", "ONE_SHARED_BOTH_SIDE_MODEL",
    "ORDINAL_FORMULATION", "QUANTILE_FORMULATION", "OrdinalCumulativeHead",
    "OrdinalCumulativeHeadAudit", "OrdinalSharedRobustTargetFit",
    "QuantileSharedRobustTargetFit", "RobustTargetAudit", "RobustTargetColumns",
    "RobustTargetModelConfig", "RobustTargetModelError",
    "fit_ordinal_shared_robust_target", "fit_quantile_shared_robust_target",
    "reconstruct_quantile_shared_outputs", "repair_cumulative_ordinal_probabilities",
]
