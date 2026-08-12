"""Concrete chronological base+meta adapter for nested Stage-I feature plans.

The adapter owns row selection, fold chronology, direct R3 hand-off and metric
reporting.  Model fitting remains injected so merely importing this module (or
building a plan) cannot start a long experiment.  Crucially, the meta matrix
contains direct same-side R3 probabilities and derived trust/context fields,
never a converted bps score.  The causal common-bps map is applied only after
strict OOF scoring when reporting a pooled long/short admission tail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .stage_i_causal_admission import Causal21dAdmissionSpec, apply_causal_21d_side_admission, pooled_global_admission_comparison
from .prequential_r3_value_map import (
    PrequentialR3ValueMapConfig,
    prequential_same_side_r3_value_map,
)
from .stage_i_nested_feature_challenger import IDENTITY_COLUMNS, NestedFeatureChallengePlan, NestedFeatureChallengerError, NestedFeatureSet
from .stage_i_timestamp_contract import resolve_stage_i_timestamp_contract
from .stage_i_ranking import RANKING_POLICY, stable_stage_i_topk_positions


SCHEMA = "stage_i_nested_matched_stack_execution_v1"
TOP_FRACTIONS = (0.01, 0.05, 0.10)


@dataclass(frozen=True)
class NestedStackInput:
    side: str
    frame: pd.DataFrame
    base_feature_universe: tuple[str, ...]
    meta_context_features: tuple[str, ...]
    r3_target_column: str = "r3_class"
    net_column: str = "exact_net_bps"
    decision_column: str = "decision_ts"
    label_available_column: str = "label_available_ts"
    meta_universe_provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NestedStackConfig:
    n_validation_folds: int = 4
    min_base_train_rows: int = 500
    min_meta_train_rows: int = 500
    correction_cap: float = 0.25
    base_candidate_fraction: float = 0.30


@dataclass(frozen=True)
class GuardedMetaArmSpec:
    arm_id: str
    family: str
    hurdle_bps: float = 0.0
    residual_clip_bps: float = 100.0
    veto_probability: float = 0.50
    shrinkage_support: float = 50.0

    def __post_init__(self) -> None:
        if self.family not in {
            "reliability", "overestimate_veto", "ordinal",
            "quantile_ordinal_residual", "clipped_residual",
        }:
            raise NestedFeatureChallengerError(f"unsupported guarded meta family {self.family!r}")
        if not self.arm_id or self.residual_clip_bps <= 0 or not 0 < self.veto_probability < 1 or self.shrinkage_support < 0:
            raise NestedFeatureChallengerError("invalid guarded meta arm specification")


BaseFoldPredictor = Callable[[pd.DataFrame, np.ndarray, pd.DataFrame, NestedFeatureSet], np.ndarray]
MetaFoldPredictor = Callable[[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, GuardedMetaArmSpec], np.ndarray]
MetaFeatureSelector = Callable[
    [pd.DataFrame, np.ndarray, Sequence[str], Sequence[str], GuardedMetaArmSpec],
    tuple[tuple[str, ...], Mapping[str, Any]],
]


@dataclass(frozen=True)
class NestedStackArmOutput:
    feature_set: str
    arm_id: str
    family: str
    frame: pd.DataFrame
    fold_provenance: pd.DataFrame


@dataclass(frozen=True)
class NestedStackExecutionResult:
    side: str
    strict_oof_identity_sha256: str
    base_outputs: Mapping[str, pd.DataFrame]
    meta_outputs: Mapping[tuple[str, str], NestedStackArmOutput]
    metrics: pd.DataFrame


def _utc(frame: pd.DataFrame, column: str) -> pd.Series:
    value = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if value.isna().any():
        raise NestedFeatureChallengerError(f"{column} must contain valid UTC timestamps")
    return value


def _identity_hash(frame: pd.DataFrame) -> str:
    identity = frame.loc[:, list(IDENTITY_COLUMNS)]
    if identity.isna().any().any() or identity.duplicated().any():
        raise NestedFeatureChallengerError("strict stack input requires unique non-null identities")
    return __import__("hashlib").sha256(pd.util.hash_pandas_object(identity, index=False).to_numpy(dtype=np.uint64).tobytes()).hexdigest()


def _folds(decision: pd.Series, available: pd.Series, *, count: int, min_train_rows: int) -> list[np.ndarray]:
    order = np.argsort(decision.to_numpy(dtype="datetime64[ns]"), kind="stable")
    ordered = decision.to_numpy(dtype="datetime64[ns]")[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    groups = [order[start:end] for start, end in zip(starts, np.r_[starts[1:], len(order)])]
    first = next((i for i, group in enumerate(groups) if int(available.lt(decision.iloc[group].min()).sum()) >= min_train_rows), None)
    if first is None:
        raise NestedFeatureChallengerError("no strict OOF support after chronological base burn-in")
    blocks = np.array_split(np.arange(len(groups) - first), min(count, len(groups) - first))
    return [np.concatenate([groups[first + int(i)] for i in block]).astype(np.int32) for block in blocks if len(block)]


def _simplex(value: np.ndarray, rows: int, *, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (rows, 3) or not np.isfinite(result).all() or (result < 0).any() or not np.allclose(result.sum(axis=1), 1.0, atol=1e-6):
        raise NestedFeatureChallengerError(f"{label} must return an aligned finite R3 probability simplex")
    return result


def _base_context(frame: pd.DataFrame, probability: np.ndarray, context: Sequence[str]) -> pd.DataFrame:
    out = frame.loc[:, list(IDENTITY_COLUMNS) + ["side_name", "decision_ts", "label_available_ts", "r3_class", "exact_net_bps", "fold_id"]].copy()
    out["r3_p_adverse"], out["r3_p_weak"], out["r3_p_clear"] = probability[:, 0], probability[:, 1], probability[:, 2]
    out["r3_opportunity_score"] = probability[:, 2] - probability[:, 0]
    out["base_r3_max_probability"] = probability.max(axis=1)
    out["base_r3_top2_margin"] = np.partition(probability, -2, axis=1)[:, -1] - np.partition(probability, -2, axis=1)[:, -2]
    out["base_r3_entropy"] = -(np.clip(probability, 1e-12, 1.0) * np.log(np.clip(probability, 1e-12, 1.0))).sum(axis=1)
    for column in context:
        out[column] = frame[column].to_numpy()
    return out


def _base_candidate_positions(
    frame: pd.DataFrame, *, fraction: float
) -> np.ndarray:
    """Select the canonical side-local base handoff without outcome inputs.

    The rank is global over the supplied chronological fold/pool, never per
    timestamp.  Callers pass only OOF rows from one side.  The same immutable
    tie policy used by all Stage-I tails makes the handoff row-order invariant.
    """
    value = float(fraction)
    if not 0.0 < value <= 1.0:
        raise NestedFeatureChallengerError(
            "base_candidate_fraction must be in (0, 1]"
        )
    if frame.empty:
        return np.asarray([], dtype=np.int32)
    count = min(len(frame), max(1, int(math.ceil(value * len(frame)))))
    return stable_stage_i_topk_positions(
        frame["r3_opportunity_score"].to_numpy(dtype=float),
        candidate_ids=frame["candidate_id"].to_numpy(dtype=object),
        side_names=frame["side_name"].to_numpy(dtype=object),
        decision_timestamps=frame["decision_ts"],
        signal_timestamps=frame["__ts__"] if "__ts__" in frame else None,
        symbols=(
            frame["__symbol__"].to_numpy(dtype=object)
            if "__symbol__" in frame
            else None
        ),
        count=count,
    ).astype(np.int32, copy=False)


def _binary_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if not np.isfinite(prediction).all() or (prediction < 0).any() or (prediction > 1).any():
        raise NestedFeatureChallengerError("binary guarded meta output must be a probability")
    clipped = np.clip(prediction, 1e-12, 1 - 1e-12)
    metric = {"target_brier": float(np.square(prediction - target).mean()), "target_log_loss": float(-(target * np.log(clipped) + (1 - target) * np.log(1 - clipped)).mean())}
    bins = np.minimum((prediction * 10).astype(int), 9)
    metric["target_ece_10"] = float(sum((bins == i).mean() * abs(prediction[bins == i].mean() - target[bins == i].mean()) for i in range(10) if (bins == i).any()))
    return metric


def _training_target(train: pd.DataFrame, spec: GuardedMetaArmSpec) -> np.ndarray:
    raw, net = train.r3_opportunity_score.to_numpy(float), train.exact_net_bps.to_numpy(float)
    if spec.family == "reliability":
        return (net >= spec.hurdle_bps).astype(float)
    if spec.family == "overestimate_veto":
        return ((raw > 0.0) & (net < spec.hurdle_bps)).astype(float)
    if spec.family == "ordinal":
        return np.digitize(net, (-100.0, 0.0, 100.0), right=True)
    if spec.family == "quantile_ordinal_residual":
        labels, *_unused = _fit_quantile_residual(
            train, shrinkage_support=spec.shrinkage_support
        )
        return labels
    return np.clip(net, -spec.residual_clip_bps, spec.residual_clip_bps)


def _fit_quantile_residual(
    train: pd.DataFrame, *, shrinkage_support: float,
) -> tuple[
    np.ndarray, tuple[float, float], tuple[float, float, float],
    tuple[int, int, int], tuple[float, float, float],
    tuple[float, float, float], tuple[float, float],
]:
    if "prequential_base_expected_net_bps" not in train:
        raise NestedFeatureChallengerError(
            "quantile residual target requires the causal mapped base value"
        )
    residual = (
        train.exact_net_bps.to_numpy(float)
        - train.prequential_base_expected_net_bps.to_numpy(float)
    )
    if not np.isfinite(residual).all():
        raise NestedFeatureChallengerError("quantile residual target is non-finite")
    thresholds = tuple(
        float(value)
        for value in np.quantile(residual, (1.0 / 3.0, 2.0 / 3.0), method="linear")
    )
    if thresholds[0] >= thresholds[1]:
        raise NestedFeatureChallengerError("quantile residual thresholds are degenerate")
    labels = np.digitize(residual, thresholds, right=True).astype(np.int8)
    support = tuple(int(np.sum(labels == value)) for value in range(3))
    if any(value <= 0 for value in support):
        raise NestedFeatureChallengerError("quantile residual target lacks a class")
    bounds = tuple(float(value) for value in np.quantile(residual, (0.05, 0.95), method="linear"))
    winsorized = np.clip(residual, bounds[0], bounds[1])
    global_location = float(winsorized.mean())
    medians = tuple(float(np.median(residual[labels == value])) for value in range(3))
    locations, uncertainty = [], []
    for value in range(3):
        local = winsorized[labels == value]
        locations.append(float(
            (local.sum() + shrinkage_support * global_location)
            / (len(local) + shrinkage_support)
        ))
        uncertainty.append(float(
            np.std(local, ddof=1) / np.sqrt(len(local)) if len(local) > 1 else 0.0
        ))
    return labels, thresholds, tuple(locations), support, medians, tuple(uncertainty), bounds


def _quantile_ordinal_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(target, dtype=np.int8)
    p = np.asarray(probability, dtype=float)
    if (
        p.shape != (len(labels), 3) or not np.isfinite(p).all() or (p < 0).any()
        or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
        or not np.isin(labels, (0, 1, 2)).all()
    ):
        raise NestedFeatureChallengerError(
            "quantile ordinal prediction must be an aligned three-class simplex"
        )
    one_hot = np.eye(3)[labels]
    predicted = np.argmax(p, axis=1)
    confusion = [
        [int(np.sum((labels == actual) & (predicted == forecast))) for forecast in range(3)]
        for actual in range(3)
    ]
    confidence = p.max(axis=1)
    correct = (predicted == labels).astype(float)
    bins = np.minimum((confidence * 10).astype(int), 9)
    return {
        "target_multiclass_log_loss": float(
            -np.log(np.clip(p[np.arange(len(p)), labels], 1e-12, 1.0)).mean()
        ),
        "target_multiclass_brier": float(
            np.square(p - one_hot).sum(axis=1).mean() / 3.0
        ),
        "target_rps": float(
            np.square(
                np.cumsum(p, axis=1)[:, :-1]
                - np.cumsum(one_hot, axis=1)[:, :-1]
            ).mean()
        ),
        "target_accuracy": float(np.mean(predicted == labels)),
        "target_calibration_ece_10": float(
            sum(
                np.mean(bins == index)
                * abs(float(confidence[bins == index].mean()) - float(correct[bins == index].mean()))
                for index in range(10) if np.any(bins == index)
            )
        ),
        "target_confusion_json": json.dumps(confusion, separators=(",", ":")),
    }


def _quantile_prior_skill_metrics(
    target: np.ndarray, probability: np.ndarray, prior_probability: np.ndarray
) -> dict[str, float]:
    labels = np.asarray(target, dtype=np.int8)
    p = np.asarray(probability, dtype=float)
    prior = np.asarray(prior_probability, dtype=float)
    if prior.shape != p.shape or prior.shape != (len(labels), 3):
        raise NestedFeatureChallengerError("quantile prior prediction is misaligned")
    one_hot = np.eye(3)[labels]

    def values(candidate: np.ndarray) -> tuple[float, float, float, float, float]:
        predicted = np.argmax(candidate, axis=1)
        accuracy = float(np.mean(predicted == labels))
        balanced = float(np.mean([
            np.mean(predicted[labels == value] == value)
            for value in range(3) if np.any(labels == value)
        ]))
        chosen = np.clip(candidate[np.arange(len(labels)), labels], 1e-12, 1.0)
        log_loss = float(-np.log(chosen).mean())
        brier = float(np.square(candidate - one_hot).sum(axis=1).mean() / 3.0)
        rps = float(np.square(
            np.cumsum(candidate, axis=1)[:, :-1]
            - np.cumsum(one_hot, axis=1)[:, :-1]
        ).mean())
        return accuracy, balanced, log_loss, brier, rps

    model, null = values(p), values(prior)
    expected = p @ np.arange(3, dtype=float)
    prior_expected = prior @ np.arange(3, dtype=float)
    spearman = (
        float(pd.Series(expected).corr(pd.Series(labels), method="spearman"))
        if np.std(expected) > 1e-12 and np.std(labels) > 1e-12 else 0.0
    )
    prior_spearman = (
        float(pd.Series(prior_expected).corr(pd.Series(labels), method="spearman"))
        if np.std(prior_expected) > 1e-12 and np.std(labels) > 1e-12 else 0.0
    )
    if not np.isfinite(spearman): spearman = 0.0
    if not np.isfinite(prior_spearman): prior_spearman = 0.0
    return {
        "target_balanced_accuracy": model[1],
        "target_ordinal_expected_class_spearman": spearman,
        "target_prior_accuracy": null[0],
        "target_majority_accuracy": null[0],
        "target_prior_balanced_accuracy": null[1],
        "target_prior_log_loss": null[2],
        "target_prior_multiclass_brier": null[3],
        "target_prior_rps": null[4],
        "target_prior_ordinal_expected_class_spearman": prior_spearman,
        "target_accuracy_delta_vs_prior": model[0] - null[0],
        "target_accuracy_ratio_to_prior": model[0] / max(null[0], 1e-12),
        "target_balanced_accuracy_delta_vs_prior": model[1] - null[1],
        "target_balanced_accuracy_ratio_to_prior": model[1] / max(null[1], 1e-12),
        "target_log_loss_delta_vs_prior": model[2] - null[2],
        "target_log_loss_ratio_to_prior": model[2] / max(null[2], 1e-12),
        "target_log_loss_skill": 1.0 - model[2] / max(null[2], 1e-12),
        "target_brier_delta_vs_prior": model[3] - null[3],
        "target_brier_ratio_to_prior": model[3] / max(null[3], 1e-12),
        "target_brier_skill": 1.0 - model[3] / max(null[3], 1e-12),
        "target_rps_delta_vs_prior": model[4] - null[4],
        "target_rps_ratio_to_prior": model[4] / max(null[4], 1e-12),
        "target_rps_skill": 1.0 - model[4] / max(null[4], 1e-12),
        "target_ordinal_spearman_delta_vs_prior": spearman - prior_spearman,
    }


def _target_and_score(train: pd.DataFrame, valid: pd.DataFrame, spec: GuardedMetaArmSpec, prediction: np.ndarray, cap: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    raw_train = train.r3_opportunity_score.to_numpy(float)
    raw_valid = valid.r3_opportunity_score.to_numpy(float)
    net_train, net_valid = train.exact_net_bps.to_numpy(float), valid.exact_net_bps.to_numpy(float)
    raw_scale = max(float(np.std(raw_train)), 1e-6)
    admitted = np.ones(len(valid), dtype=bool)
    if spec.family == "reliability":
        target_train, target_valid = _training_target(train, spec), (net_valid >= spec.hurdle_bps).astype(float)
        p = np.asarray(prediction, dtype=float).reshape(-1)
        if len(p) != len(valid): raise NestedFeatureChallengerError("reliability prediction is misaligned")
        prior = float(target_train.mean())
        score = raw_valid + np.clip((p - prior) / max(math.sqrt(prior * (1 - prior)), 1e-6), -cap, cap) * raw_scale
        return target_train, target_valid, score, _binary_metrics(target_valid, p)
    if spec.family == "overestimate_veto":
        target_train = _training_target(train, spec)
        target_valid = ((raw_valid > 0.0) & (net_valid < spec.hurdle_bps)).astype(float)
        p = np.asarray(prediction, dtype=float).reshape(-1)
        if len(p) != len(valid): raise NestedFeatureChallengerError("veto prediction is misaligned")
        admitted = p < spec.veto_probability
        metrics = _binary_metrics(target_valid, p)
        actual = target_valid > 0.5
        metrics["veto_false_negative_rate"] = float((~admitted & actual).sum() / max(1, actual.sum()))
        return target_train, target_valid, raw_valid, metrics
    if spec.family == "ordinal":
        edges = (-100.0, 0.0, 100.0)
        target_train, target_valid = _training_target(train, spec), np.digitize(net_valid, edges, right=True)
        p = np.asarray(prediction, dtype=float)
        if p.shape != (len(valid), 4) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
            raise NestedFeatureChallengerError("ordinal prediction must be an aligned four-class simplex")
        payoffs = np.asarray([net_train[target_train == i].mean() if np.any(target_train == i) else net_train.mean() for i in range(4)])
        expected = p @ payoffs
        score = raw_valid + np.clip((expected - net_train.mean()) / max(float(np.std(net_train)), 1e-6), -cap, cap) * raw_scale
        one_hot = np.eye(4)[target_valid]
        return target_train, target_valid, score, {"target_multiclass_log_loss": float(-np.log(np.clip(p[np.arange(len(p)), target_valid], 1e-12, 1)).mean()), "target_multiclass_brier": float(np.square(p - one_hot).sum(axis=1).mean() / 4), "target_ordinal_expected_mae": float(np.abs(p @ np.arange(4) - target_valid).mean())}
    if spec.family == "quantile_ordinal_residual":
        (
            target_train, thresholds, locations, support, medians,
            uncertainty, winsor_bounds,
        ) = _fit_quantile_residual(
            train, shrinkage_support=spec.shrinkage_support
        )
        residual_valid = (
            valid.exact_net_bps.to_numpy(float)
            - valid.prequential_base_expected_net_bps.to_numpy(float)
        )
        target_valid = np.digitize(residual_valid, thresholds, right=True).astype(np.int8)
        p = np.asarray(prediction, dtype=float)
        metrics = _quantile_ordinal_metrics(target_valid, p)
        prior_vector = np.asarray(support, dtype=float) / float(sum(support))
        prior_probability = np.tile(prior_vector, (len(valid), 1))
        metrics.update(
            _quantile_prior_skill_metrics(target_valid, p, prior_probability)
        )
        correction = np.clip(
            (p - prior_vector) @ np.asarray(locations),
            -spec.residual_clip_bps,
            spec.residual_clip_bps,
        )
        score = valid.prequential_base_expected_net_bps.to_numpy(float) + correction
        metrics.update({
            "target_residual_q33_bps": thresholds[0],
            "target_residual_q67_bps": thresholds[1],
            "target_class_0_support": support[0],
            "target_class_1_support": support[1],
            "target_class_2_support": support[2],
            "target_class_0_residual_location_bps": locations[0],
            "target_class_1_residual_location_bps": locations[1],
            "target_class_2_residual_location_bps": locations[2],
            "target_class_0_residual_median_bps": medians[0],
            "target_class_1_residual_median_bps": medians[1],
            "target_class_2_residual_median_bps": medians[2],
            "target_class_0_location_uncertainty_bps": uncertainty[0],
            "target_class_1_location_uncertainty_bps": uncertainty[1],
            "target_class_2_location_uncertainty_bps": uncertainty[2],
            "target_residual_winsor_lower_bps": winsor_bounds[0],
            "target_residual_winsor_upper_bps": winsor_bounds[1],
            "target_zero_in_middle_tercile": bool(
                thresholds[0] < 0.0 <= thresholds[1]
            ),
            "target_fold_semantic_valid": bool(thresholds[0] < 0.0 <= thresholds[1]),
            "target_correction_bound_bps": float(spec.residual_clip_bps),
            "target_class_location_method": "training_class_winsorized_mean_q05_q95_shrunk_to_global_winsorized_mean",
            "target_class_location_shrinkage_support": float(spec.shrinkage_support),
            "target_quantile_method": "linear",
            "target_class_0_name": "lower_residual_tercile",
            "target_class_1_name": "middle_residual_tercile",
            "target_class_2_name": "upper_residual_tercile",
            "target_economic_class_interpretation": (
                "overestimate|approximately_right|underestimate"
                if thresholds[0] < 0.0 <= thresholds[1] else "not_authorized"
            ),
        })
        return target_train, target_valid, score, metrics
    target_train = _training_target(train, spec)
    target_valid = np.clip(net_valid, -spec.residual_clip_bps, spec.residual_clip_bps)
    p = np.asarray(prediction, dtype=float).reshape(-1)
    if len(p) != len(valid) or not np.isfinite(p).all(): raise NestedFeatureChallengerError("clipped-residual prediction is misaligned")
    score = raw_valid + np.clip((p - target_train.mean()) / max(float(np.std(target_train)), 1e-6), -cap, cap) * raw_scale
    error = p - target_valid
    return target_train, target_valid, score, {"target_clipped_residual_mae": float(np.abs(error).mean()), "target_clipped_residual_rmse": float(np.sqrt(np.square(error).mean())), "target_clipped_residual_signed_bias": float(error.mean())}


def _tail_summary(frame: pd.DataFrame, score: np.ndarray, admitted: np.ndarray) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for fraction in TOP_FRACTIONS:
        eligible = np.flatnonzero(admitted & np.isfinite(score))
        count = min(len(eligible), max(1, int(math.ceil(fraction * len(eligible)))))
        ranked = (
            stable_stage_i_topk_positions(
                score,
                candidate_ids=frame["candidate_id"].to_numpy(dtype=object),
                side_names=frame["side_name"].to_numpy(dtype=object),
                decision_timestamps=frame["decision_ts"],
                signal_timestamps=frame["__ts__"] if "__ts__" in frame else None,
                symbols=(
                    frame["__symbol__"].to_numpy(dtype=object)
                    if "__symbol__" in frame
                    else None
                ),
                count=count,
                valid_mask=admitted,
            )
            if count
            else np.asarray([], dtype=int)
        )
        selected = frame.iloc[ranked]
        out.append({"top_fraction": fraction, "selected_rows": int(len(selected)), "net_bps_per_trade": float(selected.exact_net_bps.mean()) if len(selected) else np.nan, "worst_month_net_bps_per_trade": float(selected.groupby(selected.decision_ts.dt.strftime("%Y-%m")).exact_net_bps.mean().min()) if len(selected) else np.nan, "worst_fold_net_bps_per_trade": float(selected.groupby("fold_id").exact_net_bps.mean().min()) if len(selected) else np.nan, "ranking_tie_policy": RANKING_POLICY})
    return out


def execute_matched_nested_stack(
    data: NestedStackInput, plan: NestedFeatureChallengePlan, *, base_predictor: BaseFoldPredictor,
    meta_predictor: MetaFoldPredictor, meta_arms: Sequence[GuardedMetaArmSpec],
    meta_feature_selector: MetaFeatureSelector | None = None,
    config: NestedStackConfig = NestedStackConfig(),
) -> NestedStackExecutionResult:
    """Refit each nested base arm, then fit guarded meta heads only on prior base OOF rows."""
    if data.side != plan.side or data.side not in {"long", "short"}:
        raise NestedFeatureChallengerError("matched stack data and plan must share a canonical side")
    raw = data.frame.copy().reset_index(drop=True)
    needed = {*IDENTITY_COLUMNS, "side_name", data.r3_target_column, data.net_column, data.decision_column, data.label_available_column, *data.base_feature_universe, *data.meta_context_features}
    if missing := sorted(needed.difference(raw.columns)):
        raise NestedFeatureChallengerError(f"matched stack input lacks {missing[:10]}")
    if not raw.side_name.astype(str).str.lower().eq(data.side).all(): raise NestedFeatureChallengerError("matched stack input is not side-local")
    raw = raw.rename(columns={data.r3_target_column: "r3_class", data.net_column: "exact_net_bps", data.decision_column: "decision_ts", data.label_available_column: "label_available_ts"})
    raw["decision_ts"], raw["label_available_ts"] = _utc(raw, "decision_ts"), _utc(raw, "label_available_ts")
    # Resolve against immutable signal-close identity; this enforces both the
    # next-bar executable decision and the full H12 path availability.
    resolve_stage_i_timestamp_contract(raw)
    if not np.isin(raw.r3_class, (0, 1, 2)).all() or not np.isfinite(pd.to_numeric(raw.exact_net_bps, errors="coerce")).all(): raise NestedFeatureChallengerError("invalid Stage-I target/timing contract")
    identity_hash = _identity_hash(raw)
    blocks = _folds(raw.decision_ts, raw.label_available_ts, count=config.n_validation_folds, min_train_rows=config.min_base_train_rows)
    fold = np.full(len(raw), -1, dtype=int)
    for index, positions in enumerate(blocks): fold[positions] = index
    raw["fold_id"] = fold
    base_outputs: dict[str, pd.DataFrame] = {}
    meta_outputs: dict[tuple[str, str], NestedStackArmOutput] = {}
    metrics: list[dict[str, Any]] = []
    for feature_set in plan.feature_sets:
        if not set(feature_set.features).issubset(data.base_feature_universe): raise NestedFeatureChallengerError(f"{feature_set.name} escapes base feature universe")
        probability = np.full((len(raw), 3), np.nan)
        base_lineage: list[dict[str, Any]] = []
        for fold_id, valid_idx in enumerate(blocks):
            start = raw.decision_ts.iloc[valid_idx].min()
            train_idx = np.flatnonzero(raw.label_available_ts.lt(start).to_numpy())
            prediction = _simplex(base_predictor(raw.iloc[train_idx].loc[:, list(feature_set.features)], raw.r3_class.iloc[train_idx].to_numpy(int), raw.iloc[valid_idx].loc[:, list(feature_set.features)], feature_set), len(valid_idx), label=f"{feature_set.name}/fold{fold_id}")
            probability[valid_idx] = prediction
            base_lineage.append({"fold_id": fold_id, "train_rows": len(train_idx), "validation_rows": len(valid_idx), "validation_start_utc": start.isoformat(), "train_max_label_available_utc": raw.label_available_ts.iloc[train_idx].max().isoformat(), "strict_prior_resolved": True})
        context = _base_context(raw, probability, data.meta_context_features)
        mapped = np.full(len(context), np.nan, dtype=float)
        mapped_positions = np.flatnonzero(context.fold_id.to_numpy(int) >= 0)
        mapped_values, _mapped_audit, _mapped_provenance = prequential_same_side_r3_value_map(
            exact_net_bps=context.exact_net_bps.iloc[mapped_positions].to_numpy(float),
            decision_timestamps=context.decision_ts.iloc[mapped_positions],
            label_available_timestamps=context.label_available_ts.iloc[mapped_positions],
            side=data.side,
            score=context.r3_opportunity_score.iloc[mapped_positions].to_numpy(float),
            config=PrequentialR3ValueMapConfig(side=data.side),
        )
        mapped[mapped_positions] = np.asarray(mapped_values, dtype=float)
        # Target/reconstruction state only. It is deliberately excluded from
        # ``direct_features`` below, so the classifier still receives the raw
        # same-side simplex, trust and declared causal context without a
        # converted base feature.
        context["prequential_base_expected_net_bps"] = mapped
        # The first base fold has no earlier direct base OOF hand-off for a
        # leakage-safe meta fit.  All later folds share one matched population.
        meta_fold_ids: list[int] = []
        for index, valid_idx in enumerate(blocks):
            start = raw.decision_ts.iloc[valid_idx].min()
            train_pool_idx = np.flatnonzero(
                (fold >= 0)
                & (fold < index)
                & raw.label_available_ts.lt(start).to_numpy()
            )
            train_pool = context.iloc[train_pool_idx]
            if len(
                _base_candidate_positions(
                    train_pool, fraction=config.base_candidate_fraction
                )
            ) >= config.min_meta_train_rows:
                meta_fold_ids.append(index)
        if not meta_fold_ids: raise NestedFeatureChallengerError("no matched meta OOF support after direct-base OOF burn-in")
        matched = np.isin(fold, meta_fold_ids)
        base_eval = context.loc[matched].reset_index(drop=True)
        # The causal mapped value is target/reconstruction state for the
        # quantile residual head, not part of the direct same-side base OOF
        # feature handoff or its public base output.
        base_outputs[feature_set.name] = base_eval.drop(
            columns=["prequential_base_expected_net_bps"]
        )
        p = base_eval.loc[:, ["r3_p_adverse", "r3_p_weak", "r3_p_clear"]].to_numpy(float); y = base_eval.r3_class.to_numpy(int); one_hot = np.eye(3)[y]
        base_core = {"base_multiclass_log_loss": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()), "base_multiclass_brier": float(np.square(p - one_hot).sum(axis=1).mean() / 3)}
        for tail in _tail_summary(base_eval, base_eval.r3_opportunity_score.to_numpy(float), np.ones(len(base_eval), dtype=bool)):
            metrics.append({"schema": SCHEMA, "side": data.side, "feature_set": feature_set.name, "layer": "base", "arm_id": "direct_r3_base", "target_family": "r3_multiclass", **base_core, **tail, "side_attribution": data.side})
        direct_features = ("r3_p_adverse", "r3_p_weak", "r3_p_clear", "r3_opportunity_score", "base_r3_max_probability", "base_r3_top2_margin", "base_r3_entropy", *data.meta_context_features)
        for spec in meta_arms:
            output_rows: list[pd.DataFrame] = []; lineage: list[dict[str, Any]] = []
            for fold_id in meta_fold_ids:
                valid_idx = np.flatnonzero(fold == fold_id); start = raw.decision_ts.iloc[valid_idx].min()
                train_idx = np.flatnonzero((raw.label_available_ts.lt(start) & (fold >= 0) & (fold < fold_id)).to_numpy())
                train_pool = context.iloc[train_idx].reset_index(drop=True)
                valid_pool = context.iloc[valid_idx].reset_index(drop=True)
                train_positions = _base_candidate_positions(
                    train_pool, fraction=config.base_candidate_fraction
                )
                valid_positions = _base_candidate_positions(
                    valid_pool, fraction=config.base_candidate_fraction
                )
                train = train_pool.iloc[train_positions].reset_index(drop=True)
                valid = valid_pool.iloc[valid_positions].reset_index(drop=True)
                if len(train) < config.min_meta_train_rows:
                    raise NestedFeatureChallengerError(
                        "meta fold escaped the base-candidate minimum support gate"
                    )
                # No converted score is allowed in this direct R3 hand-off.
                if any(column.startswith("prequential_") or "expected_net" in column for column in direct_features): raise AssertionError("converted base input escaped into meta matrix")
                target_train = _training_target(train, spec)
                weights = np.ones(len(train), dtype=float)
                mandatory_direct = direct_features[:7]
                selected_features, selection_provenance = (
                    meta_feature_selector(train, target_train, direct_features, mandatory_direct, spec)
                    if meta_feature_selector is not None
                    else (tuple(direct_features), {"mode": "no_selector_supplied"})
                )
                selected_features = tuple(dict.fromkeys(map(str, selected_features)))
                if not selected_features or not set(mandatory_direct).issubset(selected_features) or not set(selected_features).issubset(direct_features):
                    raise NestedFeatureChallengerError("meta feature selector violated direct/trust or declared feature contract")
                prediction = meta_predictor(train.loc[:, list(selected_features)], target_train, weights, valid.loc[:, list(selected_features)], spec)
                _target_unused, target_valid, score, target_metrics = _target_and_score(train, valid, spec, prediction, config.correction_cap)
                admitted = np.asarray(score == score, dtype=bool)
                if spec.family == "overestimate_veto": admitted = np.asarray(prediction, dtype=float).reshape(-1) < spec.veto_probability
                piece = valid.copy(); piece["meta_score"] = score; piece["meta_target"] = target_valid; piece["meta_admitted"] = admitted
                if spec.family == "quantile_ordinal_residual":
                    piece["meta_p_lower_residual_tercile"] = np.asarray(prediction)[:, 0]
                    piece["meta_p_middle_residual_tercile"] = np.asarray(prediction)[:, 1]
                    piece["meta_p_upper_residual_tercile"] = np.asarray(prediction)[:, 2]
                    prior = np.asarray(
                        [
                            target_metrics["target_class_0_support"],
                            target_metrics["target_class_1_support"],
                            target_metrics["target_class_2_support"],
                        ],
                        dtype=float,
                    )
                    prior /= prior.sum()
                    piece["meta_prior_p_lower_residual_tercile"] = prior[0]
                    piece["meta_prior_p_middle_residual_tercile"] = prior[1]
                    piece["meta_prior_p_upper_residual_tercile"] = prior[2]
                output_rows.append(piece); lineage.append({"fold_id": fold_id, "train_pool_rows": len(train_pool), "train_rows": len(train), "validation_pool_rows": len(valid_pool), "validation_rows": len(valid), "base_candidate_fraction": float(config.base_candidate_fraction), "base_candidate_ranking_policy": RANKING_POLICY, "base_candidate_ranking_scope": "side_local_global_within_chronological_fold_or_prior_oof_pool; never_per_timestamp", "validation_start_utc": start.isoformat(), "train_max_label_available_utc": train.label_available_ts.max().isoformat(), "strict_prior_resolved": True, "selected_meta_features": list(selected_features), "meta_feature_count": len(selected_features), "meta_selection_provenance": dict(selection_provenance), **target_metrics})
            arm_frame = pd.concat(output_rows, ignore_index=True)
            lineage_frame = pd.DataFrame(lineage)
            semantic_valid = bool(
                lineage_frame.target_fold_semantic_valid.fillna(False).all()
                if spec.family == "quantile_ordinal_residual" else True
            )
            if spec.family == "quantile_ordinal_residual" and not semantic_valid:
                lineage_frame["target_economic_class_interpretation"] = "not_authorized"
            lineage_frame["target_semantic_valid_all_folds"] = semantic_valid
            arm = NestedStackArmOutput(feature_set.name, spec.arm_id, spec.family, arm_frame, lineage_frame)
            meta_outputs[(feature_set.name, spec.arm_id)] = arm
            target_metrics = {key: float(value) for key, value in arm.fold_provenance.iloc[:, :].select_dtypes(include=[np.number]).mean().items() if key.startswith("target_")}
            for tail in _tail_summary(arm_frame, arm_frame.meta_score.to_numpy(float), arm_frame.meta_admitted.to_numpy(bool)):
                metrics.append({"schema": SCHEMA, "side": data.side, "feature_set": feature_set.name, "layer": "meta", "arm_id": spec.arm_id, "target_family": spec.family, "target_semantic_valid": semantic_valid, **target_metrics, **tail, "side_attribution": data.side})
    return NestedStackExecutionResult(data.side, identity_hash, base_outputs, meta_outputs, pd.DataFrame(metrics))


def pooled_global_causal_tail(
    long: NestedStackExecutionResult, short: NestedStackExecutionResult, *, feature_set: str, arm_id: str,
    admission_spec: Causal21dAdmissionSpec = Causal21dAdmissionSpec(),
) -> pd.DataFrame:
    """Apply the 21-day side map only after both side-local stacks are OOF-complete."""
    pieces: list[pd.DataFrame] = []
    for result in (long, short):
        if arm_id == "direct_r3_base":
            base = result.base_outputs.get(feature_set)
            if base is None: raise NestedFeatureChallengerError(f"{result.side}: requested final base checkpoint is absent")
            frame = base.loc[:, list(IDENTITY_COLUMNS) + ["side_name", "decision_ts", "label_available_ts", "exact_net_bps", "r3_opportunity_score"]].rename(columns={"r3_opportunity_score": "meta_score"}).copy()
        else:
            arm = result.meta_outputs.get((feature_set, arm_id))
            if arm is None: raise NestedFeatureChallengerError(f"{result.side}: requested final arm checkpoint is absent")
            frame = arm.frame.loc[:, list(IDENTITY_COLUMNS) + ["side_name", "decision_ts", "label_available_ts", "exact_net_bps", "meta_score"]].copy()
        frame["candidate_key"] = frame.side_name.astype(str) + "::" + frame.candidate_id.astype(str) + "::" + frame["__ts__"].astype(str)
        mapped, _audit = apply_causal_21d_side_admission(frame, score_column="meta_score", net_column="exact_net_bps", decision_column="decision_ts", label_available_column="label_available_ts", identity_column="candidate_key", spec=admission_spec)
        pieces.append(mapped)
    return pooled_global_admission_comparison(pd.concat(pieces, ignore_index=True), raw_score_column="meta_score", net_column="exact_net_bps", identity_column="candidate_key", top_fractions=TOP_FRACTIONS)
