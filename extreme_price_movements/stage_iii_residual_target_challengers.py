"""Leakage-safe Stage-III residual target challengers.

The production Stage-III architecture has one shared residual expert over both
sides.  This module supplies three alternative training contracts without
creating side- or regime-local models:

* a four-class ordinal candidate residual with fixed economic boundaries;
* five shared conditional-quantile residual heads;
* small, context-matched pair ledgers for an auxiliary ranking loss.

All outcome-derived statistics and pairs are fit only from rows whose labels
resolved strictly before an explicit UTC cutoff.  Soft regimes are used as a
continuous similarity/context representation; they are never hard routes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA = "stage_iii_residual_target_challengers_v1"
ONE_SHARED_MODEL = "one_shared_model_no_local_experts"
ORDINAL_EDGES_BPS = (-150.0, 0.0, 100.0)
QUANTILE_LEVELS = (0.10, 0.25, 0.50, 0.75, 0.90)
PAIR_SEPARATIONS_BPS = (50.0, 100.0)
_EPS = 1e-12


class StageIIIResidualTargetError(ValueError):
    """Raised when a challenger violates the shared causal contract."""


@dataclass(frozen=True)
class ResidualTargetColumns:
    decision_timestamp: str = "decision_ts"
    label_available_timestamp: str = "label_available_ts"
    side: str = "side_name"
    exact_net_bps: str = "exact_net_bps"
    base_expected_net_bps: str = "prequential_base_expected_net_bps"
    regime_prior_residual_bps: str = "prequential_soft_regime_prior_residual_bps"
    base_map_prequential_flag: str = "base_map_is_prequential"
    base_map_source_side: str = "base_map_source_side"
    base_map_max_label_available_timestamp: str = "base_map_max_label_available_ts"
    soft_regime_causal_flag: str = "soft_regime_is_causal_prequential"
    soft_regime_fit_end_timestamp: str = "soft_regime_fit_end_ts"
    regime_prior_max_label_available_timestamp: str = "prior_resolved_max_label_available_ts"


@dataclass(frozen=True)
class OrdinalResidualTargetFit:
    """Training-only class economics for one shared ordinal model."""

    edges_bps: tuple[float, ...]
    class_mean_bps: tuple[float, ...]
    class_support: tuple[int, ...]
    fit_before_utc: pd.Timestamp
    max_label_available_utc: pd.Timestamp
    rows: int
    routing: str = ONE_SHARED_MODEL
    schema: str = SCHEMA

    @property
    def class_count(self) -> int:
        return len(self.edges_bps) + 1


@dataclass(frozen=True)
class QuantileResidualTargetFit:
    """Frozen head contract for one shared five-head quantile expert."""

    quantiles: tuple[float, ...]
    fit_before_utc: pd.Timestamp
    max_label_available_utc: pd.Timestamp
    rows: int
    routing: str = ONE_SHARED_MODEL
    schema: str = SCHEMA

    @property
    def head_names(self) -> tuple[str, ...]:
        return tuple(f"q{int(round(q * 100)):02d}" for q in self.quantiles)


@dataclass(frozen=True)
class PairConstructionConfig:
    """Bounded context matching for the auxiliary ranking term.

    Pair search occurs only within side and UTC date.  Continuous soft-regime,
    base-EV and cost/ATR constraints prevent the ranking term from comparing
    candidates with materially different economic context.
    """

    separation_bps: tuple[float, ...] = PAIR_SEPARATIONS_BPS
    min_soft_regime_similarity: float = 0.50
    max_base_ev_difference_bps: float = 75.0
    max_cost_atr_difference: float = 0.50
    max_pairs_per_better_row: int = 4
    max_rows_per_side_date: int = 2_000

    def validate(self) -> None:
        if not self.separation_bps or any(x <= 0 for x in self.separation_bps):
            raise StageIIIResidualTargetError("pair separations must be positive")
        if tuple(sorted(set(self.separation_bps))) != self.separation_bps:
            raise StageIIIResidualTargetError("pair separations must be unique and increasing")
        if not 0.0 <= self.min_soft_regime_similarity <= 1.0:
            raise StageIIIResidualTargetError("soft-regime similarity must lie in [0, 1]")
        if self.max_base_ev_difference_bps < 0 or self.max_cost_atr_difference < 0:
            raise StageIIIResidualTargetError("context tolerances must be non-negative")
        if self.max_pairs_per_better_row < 1 or self.max_rows_per_side_date < 2:
            raise StageIIIResidualTargetError("pair limits must be positive")


@dataclass(frozen=True)
class PairColumns:
    decision_timestamp: str = "decision_ts"
    label_available_timestamp: str = "label_available_ts"
    side: str = "side_name"
    candidate_id: str = "candidate_id"
    base_expected_net_bps: str = "prequential_base_expected_net_bps"
    cost_to_atr: str = "cost_to_atr"
    base_map_prequential_flag: str = "base_map_is_prequential"
    soft_regime_causal_flag: str = "soft_regime_is_causal_prequential"
    cost_atr_causal_flag: str = "cost_atr_is_causal"


def _utc(values: pd.Series, name: str) -> pd.Series:
    result = pd.to_datetime(values, utc=True, errors="coerce")
    if result.isna().any():
        raise StageIIIResidualTargetError(f"{name} contains invalid timestamps")
    return result


def _cutoff(value: object) -> pd.Timestamp:
    result = pd.Timestamp(value)
    result = result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")
    return result


def _require(frame: pd.DataFrame, names: Sequence[str]) -> None:
    missing = [name for name in names if name not in frame]
    if missing:
        raise StageIIIResidualTargetError(f"challenger frame lacks columns: {missing[:12]}")


def _numeric(frame: pd.DataFrame, name: str) -> np.ndarray:
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise StageIIIResidualTargetError(f"{name!r} must be finite")
    return values


def _strict_true_flag(frame: pd.DataFrame, name: str) -> None:
    """Require an explicit canonical true flag; never rely on truthiness."""
    _require(frame, (name,))
    values = frame[name]
    if values.isna().any() or not values.isin((True, 1)).all():
        raise StageIIIResidualTargetError(
            f"causal lineage flag {name!r} must contain only explicit true booleans"
        )


def _validate_target_lineage(
    frame: pd.DataFrame,
    decision: pd.Series,
    *,
    columns: ResidualTargetColumns,
) -> None:
    """Prove that base-map, regime state, and residual prior are causal."""
    required = (
        columns.side,
        columns.base_map_source_side,
        columns.base_map_max_label_available_timestamp,
        columns.soft_regime_fit_end_timestamp,
        columns.regime_prior_max_label_available_timestamp,
    )
    _require(frame, required)
    _strict_true_flag(frame, columns.base_map_prequential_flag)
    _strict_true_flag(frame, columns.soft_regime_causal_flag)
    side = frame[columns.side].astype(str).str.lower().str.strip()
    source_side = frame[columns.base_map_source_side].astype(str).str.lower().str.strip()
    if side.eq("").any() or not source_side.eq(side).all():
        raise StageIIIResidualTargetError("base map must be a direct same-side causal map")
    for name in (
        columns.base_map_max_label_available_timestamp,
        columns.soft_regime_fit_end_timestamp,
        columns.regime_prior_max_label_available_timestamp,
    ):
        timestamp = _utc(frame[name], name)
        if not (timestamp < decision).all():
            raise StageIIIResidualTargetError(
                f"target challenger contains current/future lineage in {name!r}"
            )


def _strict_training_contract(
    frame: pd.DataFrame,
    *,
    decision_column: str,
    available_column: str,
    fit_before_utc: object,
) -> tuple[pd.Series, pd.Series, pd.Timestamp]:
    _require(frame, (decision_column, available_column))
    decision = _utc(frame[decision_column], "decision timestamp")
    available = _utc(frame[available_column], "label availability")
    cutoff = _cutoff(fit_before_utc)
    if (available <= decision).any():
        raise StageIIIResidualTargetError("labels must resolve strictly after decision time")
    if not (decision < cutoff).all():
        raise StageIIIResidualTargetError("shared fit includes decisions at/after its cutoff")
    if not (available < cutoff).all():
        raise StageIIIResidualTargetError("shared fit includes unresolved/current/future labels")
    return decision, available, cutoff


def candidate_residual_bps(
    frame: pd.DataFrame,
    *,
    columns: ResidualTargetColumns = ResidualTargetColumns(),
) -> np.ndarray:
    """Return the regime-centered candidate-specific residual in bps."""
    _require(
        frame,
        (
            columns.exact_net_bps,
            columns.base_expected_net_bps,
            columns.regime_prior_residual_bps,
        ),
    )
    return (
        _numeric(frame, columns.exact_net_bps)
        - _numeric(frame, columns.base_expected_net_bps)
        - _numeric(frame, columns.regime_prior_residual_bps)
    )


def fit_regime_centered_ordinal_residual(
    frame: pd.DataFrame,
    *,
    fit_before_utc: object,
    columns: ResidualTargetColumns = ResidualTargetColumns(),
) -> tuple[OrdinalResidualTargetFit, np.ndarray]:
    """Fit fixed-bin class economics and return aligned ordinal labels.

    Empty-bin reconstruction falls back to the finite interval midpoint (or a
    boundary extrapolation for open tails).  Consequently validation/OOS class
    probabilities always reconstruct to common bps without inspecting OOS
    outcomes.
    """
    decision, available, cutoff = _strict_training_contract(
        frame,
        decision_column=columns.decision_timestamp,
        available_column=columns.label_available_timestamp,
        fit_before_utc=fit_before_utc,
    )
    _validate_target_lineage(frame, decision, columns=columns)
    residual = candidate_residual_bps(frame, columns=columns)
    labels = np.digitize(residual, ORDINAL_EDGES_BPS, right=True).astype(np.int8)
    fallback = (
        ORDINAL_EDGES_BPS[0] - 50.0,
        (ORDINAL_EDGES_BPS[0] + ORDINAL_EDGES_BPS[1]) / 2.0,
        (ORDINAL_EDGES_BPS[1] + ORDINAL_EDGES_BPS[2]) / 2.0,
        ORDINAL_EDGES_BPS[2] + 50.0,
    )
    support = tuple(int(np.sum(labels == klass)) for klass in range(4))
    means = tuple(
        float(np.mean(residual[labels == klass])) if support[klass] else float(fallback[klass])
        for klass in range(4)
    )
    fit = OrdinalResidualTargetFit(
        edges_bps=ORDINAL_EDGES_BPS,
        class_mean_bps=means,
        class_support=support,
        fit_before_utc=cutoff,
        max_label_available_utc=pd.Timestamp(available.max()),
        rows=len(frame),
    )
    return fit, labels


def reconstruct_ordinal_candidate_residual_bps(
    class_probabilities: Sequence[Sequence[float]],
    fit: OrdinalResidualTargetFit,
) -> np.ndarray:
    """Map shared-model ordinal probabilities through frozen class means."""
    if fit.routing != ONE_SHARED_MODEL or fit.edges_bps != ORDINAL_EDGES_BPS:
        raise StageIIIResidualTargetError("ordinal fit is not the frozen shared fixed-bin contract")
    probability = np.asarray(class_probabilities, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[1] != fit.class_count:
        raise StageIIIResidualTargetError("ordinal probabilities must have four columns")
    if not np.isfinite(probability).all() or (probability < -1e-8).any():
        raise StageIIIResidualTargetError("ordinal probabilities must be finite and non-negative")
    if not np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise StageIIIResidualTargetError("ordinal probabilities must sum to one")
    means = np.asarray(fit.class_mean_bps, dtype=np.float64)
    return (np.clip(probability, 0.0, 1.0) @ means).astype(np.float32)


def reconstruct_expected_net_bps(
    frame: pd.DataFrame,
    candidate_residual_prediction_bps: Sequence[float],
    *,
    columns: ResidualTargetColumns = ResidualTargetColumns(),
) -> np.ndarray:
    """Rejoin a challenger output to the causal base and regime baseline."""
    _require(frame, (columns.base_expected_net_bps, columns.regime_prior_residual_bps))
    residual = np.asarray(candidate_residual_prediction_bps, dtype=np.float64).reshape(-1)
    if len(residual) != len(frame) or not np.isfinite(residual).all():
        raise StageIIIResidualTargetError("candidate residual prediction must be aligned and finite")
    return (
        _numeric(frame, columns.base_expected_net_bps)
        + _numeric(frame, columns.regime_prior_residual_bps)
        + residual
    ).astype(np.float32)


def fit_quantile_residual_targets(
    frame: pd.DataFrame,
    *,
    fit_before_utc: object,
    columns: ResidualTargetColumns = ResidualTargetColumns(),
) -> tuple[QuantileResidualTargetFit, Mapping[str, np.ndarray]]:
    """Return the five aligned targets for shared quantile-loss heads."""
    decision, available, cutoff = _strict_training_contract(
        frame,
        decision_column=columns.decision_timestamp,
        available_column=columns.label_available_timestamp,
        fit_before_utc=fit_before_utc,
    )
    _validate_target_lineage(frame, decision, columns=columns)
    residual = candidate_residual_bps(frame, columns=columns).astype(np.float32)
    fit = QuantileResidualTargetFit(
        quantiles=QUANTILE_LEVELS,
        fit_before_utc=cutoff,
        max_label_available_utc=pd.Timestamp(available.max()),
        rows=len(frame),
    )
    # Each head observes the same response; its frozen quantile objective/alpha
    # supplies the different statistical functional.
    return fit, {name: residual.copy() for name in fit.head_names}


def reconstruct_quantile_residual_outputs(
    predictions: Mapping[str, Sequence[float]],
    fit: QuantileResidualTargetFit,
    *,
    repair_crossing: bool = True,
) -> pd.DataFrame:
    """Produce median, downside and width from the five quantile heads.

    Downside is the non-negative median-to-q10 gap.  Width is q90-q10; IQR is
    also retained for diagnostics.  A row-wise isotonic-by-order repair
    (cumulative maximum) is deterministic and does not use outcomes.
    """
    if fit.routing != ONE_SHARED_MODEL or fit.quantiles != QUANTILE_LEVELS:
        raise StageIIIResidualTargetError("quantile fit is not the frozen shared contract")
    missing = [name for name in fit.head_names if name not in predictions]
    if missing:
        raise StageIIIResidualTargetError(f"quantile predictions lack heads: {missing}")
    columns = [np.asarray(predictions[name], dtype=np.float64).reshape(-1) for name in fit.head_names]
    lengths = {len(value) for value in columns}
    if len(lengths) != 1 or not all(np.isfinite(value).all() for value in columns):
        raise StageIIIResidualTargetError("quantile predictions must be aligned and finite")
    matrix = np.column_stack(columns)
    crossing = np.any(np.diff(matrix, axis=1) < 0.0, axis=1)
    if crossing.any() and not repair_crossing:
        raise StageIIIResidualTargetError("quantile predictions cross")
    if repair_crossing:
        matrix = np.maximum.accumulate(matrix, axis=1)
    result = pd.DataFrame(matrix.astype(np.float32), columns=fit.head_names)
    result["candidate_residual_median_bps"] = result["q50"]
    result["candidate_residual_downside_bps"] = np.maximum(
        result["q50"].to_numpy(float) - result["q10"].to_numpy(float), 0.0
    ).astype(np.float32)
    result["candidate_residual_width_bps"] = (
        result["q90"].to_numpy(float) - result["q10"].to_numpy(float)
    ).astype(np.float32)
    result["candidate_residual_iqr_bps"] = (
        result["q75"].to_numpy(float) - result["q25"].to_numpy(float)
    ).astype(np.float32)
    result["quantile_crossing_repaired"] = crossing
    return result


def _validate_soft_regimes(frame: pd.DataFrame, names: Sequence[str]) -> np.ndarray:
    unique = tuple(dict.fromkeys(str(name) for name in names))
    if len(unique) < 2:
        raise StageIIIResidualTargetError("pair construction requires at least two soft regimes")
    _require(frame, unique)
    probability = frame.loc[:, unique].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(probability).all() or (probability < -1e-8).any():
        raise StageIIIResidualTargetError("soft regimes must be finite and non-negative")
    if not np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise StageIIIResidualTargetError("soft regimes must sum to one")
    return np.clip(probability, 0.0, 1.0)


def construct_context_matched_residual_pairs(
    frame: pd.DataFrame,
    candidate_residual: Sequence[float],
    *,
    soft_regime_columns: Sequence[str],
    fit_before_utc: object,
    columns: PairColumns = PairColumns(),
    config: PairConstructionConfig = PairConstructionConfig(),
) -> pd.DataFrame:
    """Construct deterministic training-only pairs for a small ranking term.

    Returned rows are oriented ``better`` minus ``worse`` and include both
    50-bps and 100-bps eligibility flags.  Callers can assign a small loss
    multiplier; this function never replaces the shared pointwise objective.
    """
    config.validate()
    required = (
        columns.decision_timestamp, columns.label_available_timestamp, columns.side,
        columns.candidate_id, columns.base_expected_net_bps, columns.cost_to_atr,
        columns.base_map_prequential_flag, columns.soft_regime_causal_flag,
        columns.cost_atr_causal_flag,
    )
    _require(frame, required)
    decision, _, cutoff = _strict_training_contract(
        frame,
        decision_column=columns.decision_timestamp,
        available_column=columns.label_available_timestamp,
        fit_before_utc=fit_before_utc,
    )
    for flag in (
        columns.base_map_prequential_flag,
        columns.soft_regime_causal_flag,
        columns.cost_atr_causal_flag,
    ):
        _strict_true_flag(frame, flag)
    side = frame[columns.side].astype(str).str.lower().str.strip()
    if side.eq("").any():
        raise StageIIIResidualTargetError("side must be non-empty")
    candidate_id = frame[columns.candidate_id].astype(str)
    if candidate_id.eq("").any() or candidate_id.duplicated().any():
        raise StageIIIResidualTargetError("candidate_id must be non-empty and unique")
    base = _numeric(frame, columns.base_expected_net_bps)
    cost_atr = _numeric(frame, columns.cost_to_atr)
    if (cost_atr < 0).any():
        raise StageIIIResidualTargetError("cost_to_atr must be non-negative")
    regime = _validate_soft_regimes(frame, soft_regime_columns)
    target = np.asarray(candidate_residual, dtype=np.float64).reshape(-1)
    if len(target) != len(frame) or not np.isfinite(target).all():
        raise StageIIIResidualTargetError("candidate residual must be aligned and finite")

    day = decision.dt.floor("D")
    work = pd.DataFrame({"side": side.to_numpy(), "day": day.to_numpy(), "position": np.arange(len(frame))})
    rows: list[dict[str, object]] = []
    min_sep = float(config.separation_bps[0])
    for (_, _), group in work.groupby(["side", "day"], sort=True, observed=True):
        positions = group["position"].to_numpy(np.int64)
        # Deterministic bounded support: retain an even grid over base-EV order,
        # a causal quantity, rather than outcome-dependent subsampling.
        if len(positions) > config.max_rows_per_side_date:
            ordered_base = positions[np.argsort(base[positions], kind="stable")]
            take = np.linspace(0, len(ordered_base) - 1, config.max_rows_per_side_date).round().astype(int)
            positions = ordered_base[take]
        # Better rows are traversed by target only to define ranking labels.
        better_order = positions[np.argsort(-target[positions], kind="stable")]
        for better in better_order:
            gap = target[better] - target[positions]
            similarity = regime[positions] @ regime[better]
            eligible = (
                (gap >= min_sep)
                & (similarity >= config.min_soft_regime_similarity)
                & (np.abs(base[positions] - base[better]) <= config.max_base_ev_difference_bps)
                & (np.abs(cost_atr[positions] - cost_atr[better]) <= config.max_cost_atr_difference)
            )
            worse = positions[eligible]
            if not len(worse):
                continue
            eligible_gap = gap[eligible]
            # Prefer the closest context, then the larger economic separation.
            context_distance = (
                np.abs(base[worse] - base[better]) / max(config.max_base_ev_difference_bps, _EPS)
                + np.abs(cost_atr[worse] - cost_atr[better]) / max(config.max_cost_atr_difference, _EPS)
                + (1.0 - similarity[eligible])
            )
            order = np.lexsort((-eligible_gap, context_distance))
            for inferior in worse[order[: config.max_pairs_per_better_row]]:
                residual_gap = float(target[better] - target[inferior])
                record: dict[str, object] = {
                    "better_position": int(better),
                    "worse_position": int(inferior),
                    "better_candidate_id": candidate_id.iloc[better],
                    "worse_candidate_id": candidate_id.iloc[inferior],
                    "side_name": side.iloc[better],
                    "decision_date_utc": day.iloc[better],
                    "residual_gap_bps": residual_gap,
                    "soft_regime_similarity": float(regime[better] @ regime[inferior]),
                    "base_ev_difference_bps": float(base[better] - base[inferior]),
                    "cost_atr_difference": float(cost_atr[better] - cost_atr[inferior]),
                    "pair_direction": np.int8(1),
                    "fit_before_utc": cutoff,
                    "routing": ONE_SHARED_MODEL,
                }
                for separation in config.separation_bps:
                    record[f"eligible_{int(separation)}bps"] = residual_gap >= separation
                rows.append(record)
    output = pd.DataFrame(rows)
    if output.empty:
        return pd.DataFrame(columns=[
            "better_position", "worse_position", "better_candidate_id", "worse_candidate_id",
            "side_name", "decision_date_utc", "residual_gap_bps", "soft_regime_similarity",
            "base_ev_difference_bps", "cost_atr_difference", "pair_direction",
            "fit_before_utc", "routing", *[f"eligible_{int(x)}bps" for x in config.separation_bps],
        ])
    return output.sort_values(
        ["decision_date_utc", "side_name", "better_candidate_id", "worse_candidate_id"],
        kind="stable",
    ).reset_index(drop=True)


__all__ = [
    "SCHEMA", "ONE_SHARED_MODEL", "ORDINAL_EDGES_BPS", "QUANTILE_LEVELS",
    "PAIR_SEPARATIONS_BPS", "OrdinalResidualTargetFit", "PairColumns",
    "PairConstructionConfig", "QuantileResidualTargetFit", "ResidualTargetColumns",
    "StageIIIResidualTargetError", "candidate_residual_bps",
    "construct_context_matched_residual_pairs", "fit_quantile_residual_targets",
    "fit_regime_centered_ordinal_residual", "reconstruct_ordinal_candidate_residual_bps",
    "reconstruct_expected_net_bps", "reconstruct_quantile_residual_outputs",
]
