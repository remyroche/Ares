"""Guarded Stage-I meta-target diagnostics on a frozen strict-OOF base ledger.

The Stage-I meta model is an *action* around a same-side OOF base prediction;
it is not entitled to replace the base ordering merely because a residual loss
can be fitted.  This module keeps target construction, score reconstruction and
promotion deliberately separate:

* every arm is evaluated on identical candidate/fold support;
* raw base and causal-map controls are mandatory;
* reliability corrections are standardized and bounded;
* overestimate-risk models act as vetoes rather than free rerankers;
* ordinal payoff values are fitted side-locally on training rows only; and
* the exact raw-base no-op wins unless an arm clears pooled and worst-period
  economic gates.

Model fitting remains injectable.  The helpers here define the target and
evaluation contracts used by a sequential Stage-I experiment; they do not
launch an experiment or select features.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .stage_i_ranking import RANKING_POLICY, stable_stage_i_topk_positions


SCHEMA = "stage_i_guarded_meta_target_funnel_v1"
TOP_FRACTIONS = (0.01, 0.05, 0.10, 0.20)
ORDINAL_RESIDUAL_EDGES_BPS = (-100.0, 0.0, 100.0)
QUANTILE_RESIDUAL_LABELS = ("lower", "middle", "upper")
QUANTILE_RESIDUAL_QUANTILES = (1.0 / 3.0, 2.0 / 3.0)
QUANTILE_RESIDUAL_LOCATION = (
    "training_class_winsorized_mean_q05_q95_shrunk_to_global_winsorized_mean"
)
QUANTILE_METHOD = "linear"


class StageIMetaTargetError(ValueError):
    """Raised when a target or comparison violates the frozen OOF contract."""


@dataclass(frozen=True)
class MetaTargetSpec:
    """One predeclared meta target/action arm."""

    arm_id: str
    family: str
    hurdle_bps: float = 0.0
    base_tail_fraction: float = 0.30
    tail_floor_weight: float = 0.0
    residual_clip_bps: float = 100.0
    correction_cap_score_std: float = 0.25
    shrinkage_support: float = 100.0
    veto_probability: float = 0.50

    def __post_init__(self) -> None:
        families = {
            "reliability", "overestimate_risk", "ordinal_residual",
            "quantile_ordinal_residual",
            "clipped_residual", "huber_residual",
        }
        if self.family not in families:
            raise StageIMetaTargetError(f"unknown meta target family {self.family!r}")
        if not self.arm_id.strip():
            raise StageIMetaTargetError("arm_id must be non-empty")
        if not 0.0 < self.base_tail_fraction <= 1.0:
            raise StageIMetaTargetError("base_tail_fraction must lie in (0, 1]")
        if not 0.0 <= self.tail_floor_weight <= 1.0:
            raise StageIMetaTargetError("tail_floor_weight must lie in [0, 1]")
        if self.residual_clip_bps <= 0.0 or self.correction_cap_score_std <= 0.0:
            raise StageIMetaTargetError("residual clip and correction cap must be positive")
        if self.shrinkage_support < 0.0:
            raise StageIMetaTargetError("shrinkage support must be non-negative")
        if not 0.0 < self.veto_probability < 1.0:
            raise StageIMetaTargetError("veto_probability must lie in (0, 1)")


@dataclass(frozen=True)
class MetaTargetFit:
    """Training-only reconstruction state for one arm and one side/fold."""

    spec: MetaTargetSpec
    side: str
    target: np.ndarray
    sample_weight: np.ndarray
    class_payoff_bps: tuple[float, ...] = ()
    residual_thresholds_bps: tuple[float, ...] = ()
    class_support: tuple[int, ...] = ()
    class_median_bps: tuple[float, ...] = ()
    class_location_uncertainty_bps: tuple[float, ...] = ()
    residual_winsor_bounds_bps: tuple[float, float] = ()
    class_location_method: str = ""
    quantile_method: str = ""
    prediction_center: float = 0.0
    prediction_scale: float = 1.0
    raw_base_scale: float = 1.0
    fit_rows: int = 0
    max_label_available_utc: str = ""


@dataclass(frozen=True)
class MetaOOFArm:
    """An aligned strict-OOF arm ready for economic comparison."""

    arm_id: str
    score: np.ndarray
    action_admitted: np.ndarray
    fold_id: np.ndarray
    target_family: str
    target: np.ndarray | None = None
    prediction: np.ndarray | None = None
    prior_prediction: np.ndarray | None = None
    semantic_valid: bool = True


@dataclass(frozen=True)
class StrictMetaArmResult:
    """One arm's predictions and causal fold-fit audit on common support."""

    arm: MetaOOFArm
    evaluation_positions: np.ndarray
    fold_provenance: pd.DataFrame


def default_meta_target_specs() -> tuple[MetaTargetSpec, ...]:
    """Return the small predeclared target funnel from the v3 failure audit."""
    arms: list[MetaTargetSpec] = []
    for fraction in (0.20, 0.30):
        for hurdle in (0.0, 25.0, 50.0):
            arms.append(MetaTargetSpec(
                f"T1_reliable_h{int(hurdle)}_top{int(fraction*100)}",
                "reliability", hurdle_bps=hurdle, base_tail_fraction=fraction,
            ))
            arms.append(MetaTargetSpec(
                f"T2_overestimate_d{int(hurdle)}_top{int(fraction*100)}",
                "overestimate_risk", hurdle_bps=hurdle,
                base_tail_fraction=fraction,
            ))
    arms.append(MetaTargetSpec("T3_ordinal_residual", "ordinal_residual"))
    arms.append(MetaTargetSpec(
        "T3Q_fold_quantile_ordinal_residual",
        "quantile_ordinal_residual",
        # This cap is in bps for this arm, not raw-score standard deviations.
        residual_clip_bps=200.0,
        shrinkage_support=50.0,
    ))
    for clip in (50.0, 100.0, 200.0):
        arms.append(MetaTargetSpec(
            f"T4_clipped_residual_c{int(clip)}", "clipped_residual",
            residual_clip_bps=clip, tail_floor_weight=0.25,
        ))
    arms.append(MetaTargetSpec("C3_current_map_huber", "huber_residual"))
    return tuple(arms)


def focused_quantile_meta_target_specs() -> tuple[MetaTargetSpec, ...]:
    """Focused user-requested arm plus the current Huber negative control."""
    return (
        MetaTargetSpec(
            "T3Q_fold_quantile_ordinal_residual",
            "quantile_ordinal_residual",
            residual_clip_bps=200.0,
            shrinkage_support=50.0,
        ),
        MetaTargetSpec("C3_current_map_huber", "huber_residual"),
    )


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise StageIMetaTargetError(f"required column is absent: {column}")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise StageIMetaTargetError(f"{column} must be finite")
    return values


def _utc(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        raise StageIMetaTargetError(f"required column is absent: {column}")
    values = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if values.isna().any():
        raise StageIMetaTargetError(f"{column} must contain valid UTC timestamps")
    return values


def _top_tail_mask(score: np.ndarray, fraction: float) -> np.ndarray:
    n = len(score)
    count = max(1, int(np.ceil(float(fraction) * n)))
    positions = np.arange(n, dtype=np.int64)
    order = np.lexsort((positions, score))
    mask = np.zeros(n, dtype=bool)
    mask[order[-count:]] = True
    return mask


def _shrunk_class_payoffs(
    residual: np.ndarray,
    labels: np.ndarray,
    *,
    classes: int,
    support: float,
) -> tuple[float, ...]:
    global_mean = float(np.mean(residual))
    output: list[float] = []
    for value in range(classes):
        local = residual[labels == value]
        n = int(len(local))
        local_sum = float(local.sum()) if n else 0.0
        output.append((local_sum + float(support) * global_mean) / (n + float(support)))
    return tuple(output)


def _quantile_residual_state(
    residual: np.ndarray, *, shrinkage_support: float,
) -> tuple[
    np.ndarray, tuple[float, float], tuple[float, float, float],
    tuple[int, int, int], tuple[float, float, float],
    tuple[float, float, float], tuple[float, float],
]:
    """Fit the three-class ordinal residual state on training rows only."""
    values = np.asarray(residual, dtype=np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise StageIMetaTargetError("quantile residual fit requires finite training residuals")
    thresholds = tuple(
        float(value)
        for value in np.quantile(
            values, QUANTILE_RESIDUAL_QUANTILES, method=QUANTILE_METHOD
        )
    )
    if thresholds[0] >= thresholds[1]:
        raise StageIMetaTargetError(
            "quantile residual thresholds are degenerate; three-class supervision is unavailable"
        )
    labels = np.digitize(values, thresholds, right=True).astype(np.int8)
    support = tuple(int(np.sum(labels == value)) for value in range(3))
    if any(value <= 0 for value in support):
        raise StageIMetaTargetError("quantile residual fit lacks one of its three classes")
    winsor_bounds = tuple(
        float(value) for value in np.quantile(values, (0.05, 0.95), method=QUANTILE_METHOD)
    )
    winsorized = np.clip(values, winsor_bounds[0], winsor_bounds[1])
    global_location = float(np.mean(winsorized))
    medians = tuple(float(np.median(values[labels == value])) for value in range(3))
    locations: list[float] = []
    uncertainty: list[float] = []
    for value in range(3):
        local = winsorized[labels == value]
        locations.append(float(
            (local.sum() + float(shrinkage_support) * global_location)
            / (len(local) + float(shrinkage_support))
        ))
        uncertainty.append(float(
            np.std(local, ddof=1) / np.sqrt(len(local)) if len(local) > 1 else 0.0
        ))
    return (
        labels, thresholds, tuple(locations), support, medians,
        tuple(uncertainty), winsor_bounds,
    )


def fit_meta_target(
    train: pd.DataFrame,
    spec: MetaTargetSpec,
    *,
    side: str,
    fit_before_utc: Any,
    raw_base_column: str = "r3_opportunity_score",
    mapped_base_column: str = "prequential_base_expected_net_bps",
    net_column: str = "exact_net_bps",
    decision_column: str = "decision_ts",
    label_available_column: str = "label_available_ts",
) -> MetaTargetFit:
    """Build one target using only labels resolved before ``fit_before_utc``."""
    if train.empty:
        raise StageIMetaTargetError("meta target fit cannot use an empty frame")
    side_value = str(side).lower().strip()
    if "side_name" not in train or not train.side_name.astype(str).str.lower().eq(side_value).all():
        raise StageIMetaTargetError("meta target fitting must be strictly side-local")
    decision = _utc(train, decision_column)
    available = _utc(train, label_available_column)
    cutoff = pd.Timestamp(fit_before_utc)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    if not (decision < cutoff).all() or not (available < cutoff).all():
        raise StageIMetaTargetError("meta target fit admitted unresolved/current/future rows")
    if not (available > decision).all():
        raise StageIMetaTargetError("label availability must follow the decision")
    raw = _numeric(train, raw_base_column)
    mapped = _numeric(train, mapped_base_column)
    net = _numeric(train, net_column)
    residual = net - mapped
    tail = _top_tail_mask(raw, spec.base_tail_fraction)
    weight = np.where(tail, 1.0, spec.tail_floor_weight).astype(np.float32)
    payoffs: tuple[float, ...] = ()
    thresholds: tuple[float, ...] = ()
    class_support: tuple[int, ...] = ()
    class_location_method = ""
    quantile_method = ""
    medians: tuple[float, ...] = ()
    uncertainty: tuple[float, ...] = ()
    winsor_bounds: tuple[float, float] = ()
    if spec.family == "reliability":
        target = (net >= spec.hurdle_bps).astype(np.int8)
    elif spec.family == "overestimate_risk":
        target = (net - mapped < -spec.hurdle_bps).astype(np.int8)
    elif spec.family == "ordinal_residual":
        target = np.digitize(residual, ORDINAL_RESIDUAL_EDGES_BPS, right=True).astype(np.int8)
        payoffs = _shrunk_class_payoffs(
            residual, target, classes=4, support=spec.shrinkage_support
        )
        weight = np.ones(len(train), dtype=np.float32)
    elif spec.family == "quantile_ordinal_residual":
        (
            target, thresholds, payoffs, class_support, medians,
            uncertainty, winsor_bounds,
        ) = _quantile_residual_state(
            residual, shrinkage_support=spec.shrinkage_support
        )
        weight = np.ones(len(train), dtype=np.float32)
        class_location_method = QUANTILE_RESIDUAL_LOCATION
        quantile_method = QUANTILE_METHOD
    elif spec.family == "clipped_residual":
        target = np.clip(residual, -spec.residual_clip_bps, spec.residual_clip_bps).astype(np.float32)
        # Tail relevance without hindsight top-k membership: membership is
        # defined only by the causal OOF raw-base score.
        weight = np.where(tail, 1.0, max(spec.tail_floor_weight, 0.25)).astype(np.float32)
    else:
        target = residual.astype(np.float32)
        weight = np.ones(len(train), dtype=np.float32)
    if weight.sum() <= 0:
        raise StageIMetaTargetError("target arm has no positive training weight")
    target_array = np.asarray(target)
    target_scale = float(np.std(target_array.astype(np.float64)))
    if target_scale <= 1e-12:
        target_scale = 1.0
    raw_scale = float(np.std(raw))
    if raw_scale <= 1e-12:
        raw_scale = 1.0
    return MetaTargetFit(
        spec=spec, side=side_value, target=np.asarray(target), sample_weight=weight,
        class_payoff_bps=payoffs,
        residual_thresholds_bps=thresholds,
        class_support=class_support,
        class_median_bps=medians,
        class_location_uncertainty_bps=uncertainty,
        residual_winsor_bounds_bps=winsor_bounds,
        class_location_method=class_location_method,
        quantile_method=quantile_method,
        prediction_center=float(np.average(target_array, weights=weight)),
        prediction_scale=target_scale, raw_base_scale=raw_scale, fit_rows=len(train),
        max_label_available_utc=available.max().isoformat(),
    )


def _evaluation_target(
    frame: pd.DataFrame, spec: MetaTargetSpec, fit: MetaTargetFit | None = None
) -> np.ndarray:
    net = _numeric(frame, "exact_net_bps")
    mapped = _numeric(frame, "prequential_base_expected_net_bps")
    residual = net - mapped
    if spec.family == "reliability":
        return (net >= spec.hurdle_bps).astype(np.int8)
    if spec.family == "overestimate_risk":
        return (residual < -spec.hurdle_bps).astype(np.int8)
    if spec.family == "ordinal_residual":
        return np.digitize(residual, ORDINAL_RESIDUAL_EDGES_BPS, right=True).astype(np.int8)
    if spec.family == "quantile_ordinal_residual":
        if fit is None or len(fit.residual_thresholds_bps) != 2:
            raise StageIMetaTargetError(
                "quantile ordinal evaluation requires training-fold thresholds"
            )
        return np.digitize(
            residual, fit.residual_thresholds_bps, right=True
        ).astype(np.int8)
    if spec.family == "clipped_residual":
        return np.clip(residual, -spec.residual_clip_bps, spec.residual_clip_bps).astype(np.float32)
    return residual.astype(np.float32)


def run_strict_meta_target_arm(
    frame: pd.DataFrame,
    spec: MetaTargetSpec,
    *,
    feature_columns: Sequence[str],
    fold_id: Sequence[int],
    predictor: Callable[[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, MetaTargetSpec], np.ndarray],
) -> StrictMetaArmResult:
    """Fit/predict an arm on the frozen base folds with prior-resolved labels.

    Negative fold IDs are burn-in/training-only rows.  Every non-negative fold
    is evaluated once, and its training rows satisfy
    ``label_available_ts < validation_start``.  The callback owns only model
    fitting; it cannot alter row selection, target construction or weights.
    """
    if frame.empty:
        raise StageIMetaTargetError("strict meta arm requires a non-empty frame")
    features = tuple(dict.fromkeys(map(str, feature_columns)))
    if not features or any(feature not in frame for feature in features):
        raise StageIMetaTargetError("strict meta arm requires an exact available feature list")
    side_values = frame.side_name.astype(str).str.lower().unique() if "side_name" in frame else []
    if len(side_values) != 1:
        raise StageIMetaTargetError("strict meta target arms must be fitted side-locally")
    side = str(side_values[0])
    fold = np.asarray(fold_id, dtype=np.int32).reshape(-1)
    if len(fold) != len(frame) or not np.any(fold >= 0):
        raise StageIMetaTargetError("strict meta fold IDs must be aligned and contain evaluation folds")
    decision = _utc(frame, "decision_ts")
    available = _utc(frame, "label_available_ts")
    if not (available > decision).all():
        raise StageIMetaTargetError("strict meta labels must resolve after decision")
    evaluation_positions = np.flatnonzero(fold >= 0)
    fold_values = list(dict.fromkeys(fold[evaluation_positions].tolist()))
    validation_starts = [decision.iloc[np.flatnonzero(fold == value)].min() for value in fold_values]
    if fold_values != sorted(fold_values) or validation_starts != sorted(validation_starts):
        raise StageIMetaTargetError("strict meta fold IDs are not chronologically ordered")
    score = np.full(len(frame), np.nan, dtype=np.float64)
    admitted = np.zeros(len(frame), dtype=bool)
    target_oof = np.full(len(frame), np.nan, dtype=np.float64)
    if spec.family in {"ordinal_residual", "quantile_ordinal_residual"}:
        classes = 4 if spec.family == "ordinal_residual" else 3
        prediction_oof: np.ndarray = np.full((len(frame), classes), np.nan, dtype=np.float64)
        prior_prediction_oof: np.ndarray | None = (
            np.full((len(frame), classes), np.nan, dtype=np.float64)
            if spec.family == "quantile_ordinal_residual" else None
        )
    else:
        prediction_oof = np.full(len(frame), np.nan, dtype=np.float64)
        prior_prediction_oof = None
    provenance: list[dict[str, Any]] = []
    for value, validation_start in zip(fold_values, validation_starts):
        validation_idx = np.flatnonzero(fold == value)
        train_idx = np.flatnonzero(available.lt(validation_start).to_numpy())
        if not len(train_idx) or not available.iloc[train_idx].lt(validation_start).all():
            raise StageIMetaTargetError(f"fold {value} has no strict prior-resolved training support")
        train = frame.iloc[train_idx]
        validation = frame.iloc[validation_idx]
        target_fit = fit_meta_target(
            train, spec, side=side, fit_before_utc=validation_start
        )
        prediction = np.asarray(predictor(
            train.loc[:, list(features)], target_fit.target,
            target_fit.sample_weight, validation.loc[:, list(features)], spec,
        ))
        fold_score, fold_admitted = reconstruct_meta_action(validation, target_fit, prediction)
        if len(fold_score) != len(validation_idx):
            raise StageIMetaTargetError(f"fold {value} predictor output is not aligned")
        score[validation_idx] = fold_score
        admitted[validation_idx] = fold_admitted
        target_oof[validation_idx] = _evaluation_target(validation, spec, target_fit)
        if prediction_oof.ndim == 2:
            expected_classes = prediction_oof.shape[1]
            if prediction.shape != (len(validation_idx), expected_classes):
                raise StageIMetaTargetError(
                    f"fold {value} ordinal output must be n x {expected_classes}"
                )
            prediction_oof[validation_idx] = prediction
            if spec.family == "quantile_ordinal_residual":
                prior = np.asarray(target_fit.class_support, dtype=np.float64)
                prior /= prior.sum()
                assert prior_prediction_oof is not None
                prior_prediction_oof[validation_idx] = prior
        else:
            vector = prediction[:, 1] if prediction.ndim == 2 and prediction.shape[1] == 2 else prediction.reshape(-1)
            if len(vector) != len(validation_idx):
                raise StageIMetaTargetError(f"fold {value} predictor output is not aligned")
            prediction_oof[validation_idx] = vector
        provenance.append({
            "arm_id": spec.arm_id, "target_family": spec.family,
            "side": side, "fold_id": int(value), "train_rows": int(len(train_idx)),
            "validation_rows": int(len(validation_idx)),
            "validation_start_utc": validation_start.isoformat(),
            "validation_end_utc": decision.iloc[validation_idx].max().isoformat(),
            "train_max_label_available_utc": available.iloc[train_idx].max().isoformat(),
            "strict_prior_resolved": True,
            "residual_q33_bps": (
                target_fit.residual_thresholds_bps[0]
                if target_fit.residual_thresholds_bps else np.nan
            ),
            "residual_q67_bps": (
                target_fit.residual_thresholds_bps[1]
                if target_fit.residual_thresholds_bps else np.nan
            ),
            "class_0_support": (
                target_fit.class_support[0] if target_fit.class_support else np.nan
            ),
            "class_1_support": (
                target_fit.class_support[1] if target_fit.class_support else np.nan
            ),
            "class_2_support": (
                target_fit.class_support[2] if target_fit.class_support else np.nan
            ),
            "class_0_training_prior": (
                target_fit.class_support[0] / target_fit.fit_rows
                if target_fit.class_support else np.nan
            ),
            "class_1_training_prior": (
                target_fit.class_support[1] / target_fit.fit_rows
                if target_fit.class_support else np.nan
            ),
            "class_2_training_prior": (
                target_fit.class_support[2] / target_fit.fit_rows
                if target_fit.class_support else np.nan
            ),
            "class_0_residual_location_bps": (
                target_fit.class_payoff_bps[0] if len(target_fit.class_payoff_bps) == 3 else np.nan
            ),
            "class_1_residual_location_bps": (
                target_fit.class_payoff_bps[1] if len(target_fit.class_payoff_bps) == 3 else np.nan
            ),
            "class_2_residual_location_bps": (
                target_fit.class_payoff_bps[2] if len(target_fit.class_payoff_bps) == 3 else np.nan
            ),
            "class_0_residual_median_bps": (
                target_fit.class_median_bps[0] if target_fit.class_median_bps else np.nan
            ),
            "class_1_residual_median_bps": (
                target_fit.class_median_bps[1] if target_fit.class_median_bps else np.nan
            ),
            "class_2_residual_median_bps": (
                target_fit.class_median_bps[2] if target_fit.class_median_bps else np.nan
            ),
            "class_0_location_uncertainty_bps": (
                target_fit.class_location_uncertainty_bps[0]
                if target_fit.class_location_uncertainty_bps else np.nan
            ),
            "class_1_location_uncertainty_bps": (
                target_fit.class_location_uncertainty_bps[1]
                if target_fit.class_location_uncertainty_bps else np.nan
            ),
            "class_2_location_uncertainty_bps": (
                target_fit.class_location_uncertainty_bps[2]
                if target_fit.class_location_uncertainty_bps else np.nan
            ),
            "residual_winsor_lower_bps": (
                target_fit.residual_winsor_bounds_bps[0]
                if target_fit.residual_winsor_bounds_bps else np.nan
            ),
            "residual_winsor_upper_bps": (
                target_fit.residual_winsor_bounds_bps[1]
                if target_fit.residual_winsor_bounds_bps else np.nan
            ),
            "class_location_shrinkage_support": float(spec.shrinkage_support),
            "class_location_method": target_fit.class_location_method,
            "quantile_method": target_fit.quantile_method,
            "zero_in_middle_tercile": (
                bool(
                    target_fit.residual_thresholds_bps[0]
                    < 0.0
                    <= target_fit.residual_thresholds_bps[1]
                )
                if len(target_fit.residual_thresholds_bps) == 2 else None
            ),
            "fold_semantic_valid": (
                bool(
                    target_fit.residual_thresholds_bps[0]
                    < 0.0
                    <= target_fit.residual_thresholds_bps[1]
                )
                if len(target_fit.residual_thresholds_bps) == 2 else True
            ),
            "class_0_name": "lower_residual_tercile",
            "class_1_name": "middle_residual_tercile",
            "class_2_name": "upper_residual_tercile",
            "economic_class_interpretation": (
                "overestimate|approximately_right|underestimate"
                if len(target_fit.residual_thresholds_bps) == 2
                and target_fit.residual_thresholds_bps[0] < 0.0
                <= target_fit.residual_thresholds_bps[1]
                else "not_authorized"
            ),
            "correction_bound_bps": (
                spec.residual_clip_bps
                if spec.family == "quantile_ordinal_residual" else np.nan
            ),
        })
    if not np.isfinite(score[evaluation_positions]).all() or not np.isfinite(prediction_oof[evaluation_positions]).all():
        raise StageIMetaTargetError("strict meta arm emitted incomplete evaluation predictions")
    if prior_prediction_oof is not None and not np.isfinite(
        prior_prediction_oof[evaluation_positions]
    ).all():
        raise StageIMetaTargetError("strict meta arm emitted incomplete prior predictions")
    provenance_frame = pd.DataFrame(provenance)
    semantic_valid = bool(
        provenance_frame.fold_semantic_valid.fillna(False).all()
        if spec.family == "quantile_ordinal_residual" else True
    )
    if spec.family == "quantile_ordinal_residual" and not semantic_valid:
        provenance_frame["economic_class_interpretation"] = "not_authorized"
    return StrictMetaArmResult(
        arm=MetaOOFArm(
            spec.arm_id, score[evaluation_positions], admitted[evaluation_positions],
            fold[evaluation_positions], spec.family,
            target=target_oof[evaluation_positions],
            prediction=prediction_oof[evaluation_positions],
            prior_prediction=(
                prior_prediction_oof[evaluation_positions]
                if prior_prediction_oof is not None else None
            ),
            semantic_valid=semantic_valid,
        ),
        evaluation_positions=evaluation_positions,
        fold_provenance=provenance_frame,
    )


def _probability_vector(prediction: Sequence[float], n: int) -> np.ndarray:
    values = np.asarray(prediction, dtype=np.float64)
    if values.ndim == 2:
        if values.shape != (n, 2):
            raise StageIMetaTargetError("binary prediction must be n rows x 2 classes")
        values = values[:, 1]
    values = values.reshape(-1)
    if len(values) != n or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise StageIMetaTargetError("binary predictions must be aligned finite probabilities")
    return values


def reconstruct_meta_action(
    evaluation: pd.DataFrame,
    fit: MetaTargetFit,
    prediction: Sequence[float] | np.ndarray,
    *,
    raw_base_column: str = "r3_opportunity_score",
    mapped_base_column: str = "prequential_base_expected_net_bps",
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a model output into a bounded correction or explicit veto.

    All promoted arms anchor on raw base ordering.  Only the current Huber
    negative control reconstructs around the causal map, preserving an exact
    diagnostic of the v3 behavior without silently making it the baseline.
    """
    raw = _numeric(evaluation, raw_base_column)
    mapped = _numeric(evaluation, mapped_base_column)
    n = len(evaluation)
    spec = fit.spec
    admitted = np.ones(n, dtype=bool)
    if spec.family in {"reliability", "overestimate_risk"}:
        probability = _probability_vector(prediction, n)
        if spec.family == "overestimate_risk":
            admitted = probability < spec.veto_probability
            return raw.copy(), admitted
        # The probability delta is intrinsically standardized around the
        # training prevalence; bounding prevents a low-information head from
        # freely replacing the base ordering.
        prior = float(np.average(fit.target, weights=fit.sample_weight))
        standardized = (probability - prior) / max(float(np.sqrt(prior * (1.0 - prior))), 1e-6)
        correction = np.clip(
            standardized,
            -spec.correction_cap_score_std,
            spec.correction_cap_score_std,
        ) * fit.raw_base_scale
        return raw + correction, admitted
    values = np.asarray(prediction, dtype=np.float64)
    if spec.family == "ordinal_residual":
        if values.shape != (n, 4) or not np.isfinite(values).all() or (values < 0).any():
            raise StageIMetaTargetError("ordinal predictions must be a finite n x 4 simplex")
        if not np.allclose(values.sum(axis=1), 1.0, atol=1e-5):
            raise StageIMetaTargetError("ordinal probabilities must sum to one")
        correction_bps = values @ np.asarray(fit.class_payoff_bps, dtype=np.float64)
        standardized = (correction_bps - fit.prediction_center) / fit.prediction_scale
        correction = np.clip(
            standardized,
            -spec.correction_cap_score_std,
            spec.correction_cap_score_std,
        ) * fit.raw_base_scale
        return raw + correction, admitted
    if spec.family == "quantile_ordinal_residual":
        if values.shape != (n, 3) or not np.isfinite(values).all() or (values < 0).any():
            raise StageIMetaTargetError(
                "quantile ordinal predictions must be a finite n x 3 simplex"
            )
        if not np.allclose(values.sum(axis=1), 1.0, atol=1e-5):
            raise StageIMetaTargetError("quantile ordinal probabilities must sum to one")
        if len(fit.class_payoff_bps) != 3:
            raise StageIMetaTargetError("quantile ordinal reconstruction lacks class locations")
        prior = np.asarray(fit.class_support, dtype=np.float64)
        prior /= prior.sum()
        # Conversion-head correction: a no-skill classifier which emits its
        # training prior produces exactly zero correction and therefore the
        # hidden causal-bps anchor, rather than a spurious unconditional shift.
        expected_correction_bps = (values - prior) @ np.asarray(
            fit.class_payoff_bps, dtype=np.float64
        )
        bounded = np.clip(
            expected_correction_bps,
            -spec.residual_clip_bps,
            spec.residual_clip_bps,
        )
        return mapped + bounded, admitted
    values = values.reshape(-1)
    if len(values) != n or not np.isfinite(values).all():
        raise StageIMetaTargetError("regression predictions must be aligned and finite")
    if spec.family == "huber_residual":
        return mapped + values, admitted
    standardized = (values - fit.prediction_center) / fit.prediction_scale
    correction = np.clip(
        standardized,
        -spec.correction_cap_score_std,
        spec.correction_cap_score_std,
    ) * fit.raw_base_scale
    return raw + correction, admitted


def mandatory_control_arms(frame: pd.DataFrame, fold_id: Sequence[int]) -> tuple[MetaOOFArm, ...]:
    """Create model-free raw, map, and raw-plus-zero bounded controls."""
    raw = _numeric(frame, "r3_opportunity_score")
    mapped = _numeric(frame, "prequential_base_expected_net_bps")
    fold = np.asarray(fold_id, dtype=np.int32).reshape(-1)
    if len(fold) != len(frame):
        raise StageIMetaTargetError("control fold IDs must be aligned")
    admitted = np.ones(len(frame), dtype=bool)
    return (
        MetaOOFArm("C0_raw_base_exact_noop", raw, admitted, fold, "control"),
        MetaOOFArm("C1_causal_map_only", mapped, admitted, fold, "control"),
        MetaOOFArm("C2_raw_base_bounded_zero", raw.copy(), admitted, fold, "control"),
    )


def current_huber_control_arm(
    frame: pd.DataFrame,
    fold_id: Sequence[int],
    residual_prediction_bps: Sequence[float],
) -> MetaOOFArm:
    """Create the matched v3 map-plus-Huber negative control."""
    mapped = _numeric(frame, "prequential_base_expected_net_bps")
    prediction = np.asarray(residual_prediction_bps, dtype=np.float64).reshape(-1)
    fold = np.asarray(fold_id, dtype=np.int32).reshape(-1)
    if len(prediction) != len(frame) or len(fold) != len(frame) or not np.isfinite(prediction).all():
        raise StageIMetaTargetError("current Huber control must be finite and aligned")
    return MetaOOFArm(
        "C3_current_map_huber", mapped + prediction,
        np.ones(len(frame), dtype=bool), fold, "huber_residual",
        target=_numeric(frame, "exact_net_bps") - mapped,
        prediction=prediction,
    )


def quantile_prior_conversion_control_arm(
    frame: pd.DataFrame,
    fold_id: Sequence[int],
    learned: MetaOOFArm,
) -> MetaOOFArm:
    """No-skill economic control for the prior-centered conversion head."""
    if learned.target_family != "quantile_ordinal_residual":
        raise StageIMetaTargetError("prior conversion control requires the T3Q arm")
    prior = np.asarray(learned.prior_prediction, dtype=np.float64)
    if prior.shape != (len(frame), 3):
        raise StageIMetaTargetError("prior conversion control lacks fold-local priors")
    mapped = _numeric(frame, "prequential_base_expected_net_bps")
    fold = np.asarray(fold_id, dtype=np.int32).reshape(-1)
    return MetaOOFArm(
        "C4_T3Q_fold_prior_conversion",
        mapped.copy(),
        np.ones(len(frame), dtype=bool),
        fold,
        "quantile_prior_conversion_control",
        semantic_valid=learned.semantic_valid,
    )


def _selected_indices(
    ledger: pd.DataFrame, score: np.ndarray, admitted: np.ndarray, count: int
) -> np.ndarray:
    eligible = np.flatnonzero(admitted & np.isfinite(score))
    count = min(len(eligible), max(0, int(count)))
    if not count:
        return np.asarray([], dtype=np.int64)
    candidate_ids = (
        ledger.candidate_id.to_numpy(object)
        if "candidate_id" in ledger else ledger.candidate_key.to_numpy(object)
    )
    return stable_stage_i_topk_positions(
        score,
        candidate_ids=candidate_ids,
        side_names=ledger.side_name.to_numpy(object),
        decision_timestamps=ledger.decision_ts,
        signal_timestamps=ledger["__ts__"] if "__ts__" in ledger else None,
        symbols=ledger["__symbol__"].to_numpy(object) if "__symbol__" in ledger else None,
        count=count,
        valid_mask=admitted & np.isfinite(score),
    ).astype(np.int64, copy=False)


def _rank_auc(target: np.ndarray, score: np.ndarray) -> float:
    positive = target == 1
    n_positive, n_negative = int(positive.sum()), int((~positive).sum())
    if not n_positive or not n_negative:
        return np.nan
    ranks = pd.Series(score).rank(method="average").to_numpy(np.float64)
    return float((ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))


def _target_diagnostics(arm: MetaOOFArm, n: int) -> dict[str, Any]:
    if arm.target is None or arm.prediction is None:
        return {}
    target = np.asarray(arm.target)
    prediction = np.asarray(arm.prediction, dtype=np.float64)
    if len(target) != n or len(prediction) != n or not np.isfinite(prediction).all():
        raise StageIMetaTargetError(f"arm {arm.arm_id} target diagnostics are not aligned/finite")
    if arm.target_family in {"reliability", "overestimate_risk"}:
        probability = _probability_vector(prediction, n)
        binary = target.astype(np.int8).reshape(-1)
        if not np.isin(binary, (0, 1)).all():
            raise StageIMetaTargetError(f"arm {arm.arm_id} binary target is invalid")
        clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
        return {
            "target_auc": _rank_auc(binary, probability),
            "target_brier": float(np.mean(np.square(probability - binary))),
            "target_log_loss": float(-np.mean(binary * np.log(clipped) + (1 - binary) * np.log(1 - clipped))),
        }
    if arm.target_family in {"ordinal_residual", "quantile_ordinal_residual"}:
        labels = target.astype(np.int8).reshape(-1)
        classes = 4 if arm.target_family == "ordinal_residual" else 3
        if prediction.shape != (n, classes) or not np.allclose(prediction.sum(axis=1), 1.0, atol=1e-5):
            raise StageIMetaTargetError(f"arm {arm.arm_id} ordinal diagnostic prediction is invalid")
        if not np.isin(labels, np.arange(classes)).all():
            raise StageIMetaTargetError(f"arm {arm.arm_id} ordinal labels are invalid")
        chosen = np.clip(prediction[np.arange(n), labels], 1e-7, 1.0)
        one_hot = np.eye(classes, dtype=np.float64)[labels]
        output: dict[str, Any] = {
            "target_ordinal_accuracy": float(np.mean(np.argmax(prediction, axis=1) == labels)),
            "target_log_loss": float(-np.mean(np.log(chosen))),
            "target_multiclass_brier": float(
                np.mean(np.sum(np.square(prediction - one_hot), axis=1) / classes)
            ),
            "target_rps": float(
                np.mean(
                    np.mean(
                        np.square(
                            np.cumsum(prediction, axis=1)[:, :-1]
                            - np.cumsum(one_hot, axis=1)[:, :-1]
                        ),
                        axis=1,
                    )
                )
            ),
        }
        if arm.target_family == "quantile_ordinal_residual":
            predicted = np.argmax(prediction, axis=1)
            confusion = [
                [int(np.sum((labels == actual) & (predicted == forecast))) for forecast in range(3)]
                for actual in range(3)
            ]
            confidence = prediction.max(axis=1)
            correct = (predicted == labels).astype(float)
            bins = np.minimum((confidence * 10).astype(int), 9)
            output["target_confusion_json"] = json.dumps(confusion, separators=(",", ":"))
            output["target_calibration_ece_10"] = float(
                sum(
                    np.mean(bins == index)
                    * abs(float(confidence[bins == index].mean()) - float(correct[bins == index].mean()))
                    for index in range(10) if np.any(bins == index)
                )
            )
            for value, label in enumerate(QUANTILE_RESIDUAL_LABELS):
                output[f"target_{label}_support"] = int(np.sum(labels == value))
                output[f"target_{label}_probability_mean"] = float(prediction[:, value].mean())
                output[f"target_{label}_prevalence"] = float(np.mean(labels == value))
                output[f"target_{label}_calibration_gap"] = float(
                    prediction[:, value].mean() - np.mean(labels == value)
                )
            prior = np.asarray(arm.prior_prediction, dtype=np.float64)
            if (
                prior.shape != (n, 3)
                or not np.isfinite(prior).all()
                or (prior < 0).any()
                or not np.allclose(prior.sum(axis=1), 1.0, atol=1e-6)
            ):
                raise StageIMetaTargetError(
                    f"arm {arm.arm_id} lacks aligned fold-local prior probabilities"
                )
            prior_chosen = np.clip(prior[np.arange(n), labels], 1e-7, 1.0)
            prior_predicted = np.argmax(prior, axis=1)
            model_accuracy = float(np.mean(predicted == labels))
            prior_accuracy = float(np.mean(prior_predicted == labels))

            def balanced_accuracy(forecast: np.ndarray) -> float:
                return float(
                    np.mean([
                        np.mean(forecast[labels == value] == value)
                        for value in range(3) if np.any(labels == value)
                    ])
                )

            def ordinal_spearman(probability: np.ndarray) -> float:
                expected = probability @ np.arange(3, dtype=np.float64)
                if np.std(expected) <= 1e-12 or np.std(labels) <= 1e-12:
                    return 0.0
                value = pd.Series(expected).corr(
                    pd.Series(labels.astype(float)), method="spearman"
                )
                return float(value) if np.isfinite(value) else 0.0

            prior_log_loss = float(-np.mean(np.log(prior_chosen)))
            prior_brier = float(
                np.mean(np.sum(np.square(prior - one_hot), axis=1) / 3.0)
            )
            prior_rps = float(
                np.square(
                    np.cumsum(prior, axis=1)[:, :-1]
                    - np.cumsum(one_hot, axis=1)[:, :-1]
                ).mean()
            )
            model_brier = float(output["target_multiclass_brier"])
            model_log_loss = float(output["target_log_loss"])
            model_rps = float(output["target_rps"])
            model_balanced = balanced_accuracy(predicted)
            prior_balanced = balanced_accuracy(prior_predicted)
            model_spearman = ordinal_spearman(prediction)
            prior_spearman = ordinal_spearman(prior)
            output.update({
                "target_balanced_accuracy": model_balanced,
                "target_ordinal_expected_class_spearman": model_spearman,
                "target_prior_accuracy": prior_accuracy,
                "target_majority_accuracy": prior_accuracy,
                "target_prior_balanced_accuracy": prior_balanced,
                "target_prior_log_loss": prior_log_loss,
                "target_prior_multiclass_brier": prior_brier,
                "target_prior_rps": prior_rps,
                "target_prior_ordinal_expected_class_spearman": prior_spearman,
                "target_accuracy_delta_vs_prior": model_accuracy - prior_accuracy,
                "target_accuracy_ratio_to_prior": model_accuracy / max(prior_accuracy, 1e-12),
                "target_balanced_accuracy_delta_vs_prior": model_balanced - prior_balanced,
                "target_balanced_accuracy_ratio_to_prior": model_balanced / max(prior_balanced, 1e-12),
                "target_log_loss_delta_vs_prior": model_log_loss - prior_log_loss,
                "target_log_loss_ratio_to_prior": model_log_loss / max(prior_log_loss, 1e-12),
                "target_log_loss_skill": 1.0 - model_log_loss / max(prior_log_loss, 1e-12),
                "target_brier_delta_vs_prior": model_brier - prior_brier,
                "target_brier_ratio_to_prior": model_brier / max(prior_brier, 1e-12),
                "target_brier_skill": 1.0 - model_brier / max(prior_brier, 1e-12),
                "target_rps_delta_vs_prior": model_rps - prior_rps,
                "target_rps_ratio_to_prior": model_rps / max(prior_rps, 1e-12),
                "target_rps_skill": 1.0 - model_rps / max(prior_rps, 1e-12),
                "target_ordinal_spearman_delta_vs_prior": model_spearman - prior_spearman,
            })
        return output
    pred = prediction.reshape(-1)
    truth = target.astype(np.float64).reshape(-1)
    if float(np.std(pred)) <= 1e-12 or float(np.std(truth)) <= 1e-12:
        ic = 0.0
    else:
        ic = float(pd.Series(pred).corr(pd.Series(truth), method="spearman"))
    return {"target_rank_ic": ic, "target_mae_bps": float(np.mean(np.abs(pred - truth)))}


def evaluate_meta_oof_arms(
    ledger: pd.DataFrame,
    arms: Sequence[MetaOOFArm],
    *,
    identity_column: str = "candidate_key",
    net_column: str = "exact_net_bps",
    decision_column: str = "decision_ts",
) -> pd.DataFrame:
    """Report same-support pooled tails plus worst month/fold for every arm."""
    if not arms:
        raise StageIMetaTargetError("meta evaluation requires at least one arm")
    if identity_column not in ledger or ledger[identity_column].astype(str).duplicated().any():
        raise StageIMetaTargetError("evaluation needs unique candidate identities")
    net = _numeric(ledger, net_column)
    month = _utc(ledger, decision_column).dt.strftime("%Y-%m").to_numpy()
    n = len(ledger)
    original_population_rows = (
        int(pd.to_numeric(ledger.original_side_population_rows).iloc[0])
        if "original_side_population_rows" in ledger else n
    )
    if original_population_rows < n:
        raise StageIMetaTargetError("original population denominator is smaller than candidate support")
    required = {
        "C0_raw_base_exact_noop", "C1_causal_map_only",
        "C2_raw_base_bounded_zero", "C3_current_map_huber",
    }
    names = [arm.arm_id for arm in arms]
    if len(names) != len(set(names)) or not required.issubset(names):
        raise StageIMetaTargetError("arms must be unique and include all mandatory controls")
    reference_fold = np.asarray(arms[0].fold_id, dtype=np.int32).reshape(-1)
    by_name = {arm.arm_id: arm for arm in arms}
    raw_control = np.asarray(by_name["C0_raw_base_exact_noop"].score, dtype=np.float64)
    zero_control = np.asarray(by_name["C2_raw_base_bounded_zero"].score, dtype=np.float64)
    if not np.array_equal(raw_control, zero_control, equal_nan=True):
        raise StageIMetaTargetError("bounded-correction no-op is not exactly the raw-base score")
    rows: list[dict[str, Any]] = []
    for arm in arms:
        score = np.asarray(arm.score, dtype=np.float64).reshape(-1)
        admitted = np.asarray(arm.action_admitted)
        fold = np.asarray(arm.fold_id, dtype=np.int32).reshape(-1)
        if len(score) != n or admitted.shape != (n,) or fold.shape != (n,):
            raise StageIMetaTargetError(f"arm {arm.arm_id} is not row-aligned")
        if admitted.dtype.kind != "b" or not np.array_equal(fold, reference_fold):
            raise StageIMetaTargetError(f"arm {arm.arm_id} changed fold/support identity")
        target_metrics = _target_diagnostics(arm, n)
        for fraction in TOP_FRACTIONS:
            requested = max(1, int(np.ceil(fraction * original_population_rows)))
            chosen = _selected_indices(ledger, score, admitted, requested)
            if not len(chosen):
                pooled = worst_month = worst_fold = np.nan
            else:
                pooled = float(np.mean(net[chosen]))
                worst_month = float(min(np.mean(net[chosen[month[chosen] == value]]) for value in np.unique(month[chosen])))
                worst_fold = float(min(np.mean(net[chosen[fold[chosen] == value]]) for value in np.unique(fold[chosen])))
            rows.append({
                "schema": SCHEMA, "arm_id": arm.arm_id,
                "target_family": arm.target_family, "top_fraction": fraction,
                "target_semantic_valid": bool(arm.semantic_valid),
                "candidate_rows": n, "admitted_rows": int(admitted.sum()),
                "original_population_rows": original_population_rows,
                "candidate_population_fraction": float(n / original_population_rows),
                "requested_topk_rows_original_population": requested,
                "topk_saturated_due_candidate_or_admission_support": bool(
                    int(admitted.sum()) < requested
                ),
                "ranking_tie_policy": RANKING_POLICY,
                "selected_rows": int(len(chosen)), "net_bps_per_trade": pooled,
                "worst_month_net_bps_per_trade": worst_month,
                "worst_fold_net_bps_per_trade": worst_fold,
                **target_metrics,
            })
    result = pd.DataFrame(rows)
    baseline = result[result.arm_id.eq("C0_raw_base_exact_noop")].set_index("top_fraction")
    mapped = result[result.arm_id.eq("C1_causal_map_only")].set_index("top_fraction")
    for index, row in result.iterrows():
        fraction = row.top_fraction
        result.loc[index, "delta_vs_raw_net_bps"] = row.net_bps_per_trade - baseline.loc[fraction, "net_bps_per_trade"]
        result.loc[index, "delta_vs_raw_worst_month_bps"] = row.worst_month_net_bps_per_trade - baseline.loc[fraction, "worst_month_net_bps_per_trade"]
        result.loc[index, "delta_vs_raw_worst_fold_bps"] = row.worst_fold_net_bps_per_trade - baseline.loc[fraction, "worst_fold_net_bps_per_trade"]
        result.loc[index, "delta_vs_map_net_bps"] = row.net_bps_per_trade - mapped.loc[fraction, "net_bps_per_trade"]
    return result


def select_meta_arm_with_noop_gate(metrics: pd.DataFrame) -> Mapping[str, Any]:
    """Promote only an arm beating raw base at top-10 and worst-period gates."""
    required_columns = {
        "arm_id", "top_fraction", "net_bps_per_trade",
        "worst_month_net_bps_per_trade", "worst_fold_net_bps_per_trade",
    }
    if not required_columns.issubset(metrics.columns):
        raise StageIMetaTargetError("meta gate metrics are incomplete")
    top10 = metrics[np.isclose(metrics.top_fraction, 0.10)].copy()
    baseline_rows = top10[top10.arm_id.eq("C0_raw_base_exact_noop")]
    if len(baseline_rows) != 1:
        raise StageIMetaTargetError("meta gate requires exactly one raw-base no-op")
    baseline = baseline_rows.iloc[0]
    eligible = top10[
        top10.net_bps_per_trade.gt(baseline.net_bps_per_trade)
        & top10.worst_month_net_bps_per_trade.ge(baseline.worst_month_net_bps_per_trade)
        & top10.worst_fold_net_bps_per_trade.ge(baseline.worst_fold_net_bps_per_trade)
        & top10.target_semantic_valid.astype(bool)
    ].copy()
    eligible = eligible[~eligible.arm_id.str.startswith("C")]
    if eligible.empty:
        return {
            "winner_arm_id": "C0_raw_base_exact_noop", "deployment_action": "no_op",
            "learned_meta_promoted": False,
            "reason": "no learned arm cleared pooled top10 and worst-month/fold raw-base gates",
        }
    winner = eligible.sort_values(
        ["net_bps_per_trade", "worst_month_net_bps_per_trade", "worst_fold_net_bps_per_trade", "arm_id"],
        ascending=[False, False, False, True], kind="stable",
    ).iloc[0]
    return {
        "winner_arm_id": str(winner.arm_id), "deployment_action": "learned_meta",
        "learned_meta_promoted": True, "reason": "cleared all mandatory raw-base gates",
    }


__all__ = [
    "MetaOOFArm", "MetaTargetFit", "MetaTargetSpec", "SCHEMA",
    "StageIMetaTargetError", "current_huber_control_arm", "default_meta_target_specs",
    "focused_quantile_meta_target_specs",
    "evaluate_meta_oof_arms", "fit_meta_target", "mandatory_control_arms",
    "reconstruct_meta_action", "select_meta_arm_with_noop_gate",
    "quantile_prior_conversion_control_arm",
    "run_strict_meta_target_arm", "StrictMetaArmResult",
]
