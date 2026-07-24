"""Role definitions for the five canonical future-path auxiliary heads.

This module deliberately contains no estimator imports or fitting code.  It is
the small, auditable seam between the canonical target table and a runner that
fits side-local models.  In particular, the roles below make the mixture and
survival structure explicit instead of asking one unconditional regressor to
learn censored, zero-inflated outcomes.

The public functions return *masks*, never filtered data frames.  A caller may
use a role's ``train_mask`` to select rows for fitting, but must retain the
original candidate population when emitting OOF predictions and metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

MODEL_FAMILY_SCHEMA = "path_auxiliary_model_families_v1"
PATH_VALID_COLUMN = "__path_auxiliary_target_valid__"
MEANINGFUL_HIT_COLUMN = "__meaningful_mfe_reached_12h__"
TIMING_COLUMN = "__time_to_first_meaningful_mfe_hours_12h__"
PEAK_COLUMN = "__peak_mfe_atr_12h__"
MAE_COLUMN = "__mae_before_meaningful_mfe_atr_12h__"
MAE_NO_HIT_COLUMN = "__mae_until_horizon_if_no_1_5atr__"
LEGACY_ADVERSE_COLUMNS: tuple[str, ...] = (
    "__bars_to_adverse_extreme_before_mfe_12h__",
    "__bars_before_price_stops_decreasing_12h__",
)
CONFIRMED_ADVERSE_COLUMN = "__bars_to_confirmed_adverse_trough__"
SLOPE_COLUMN = "__future_slope_atr_per_hour_12h__"
TIMING_HORIZONS_HOURS: tuple[int, ...] = (2, 4, 8, 12)


@dataclass(frozen=True)
class RoleSpec:
    """One fitted model role within a path auxiliary head.

    ``target_columns`` are ordered preference columns.  The sole alternate is
    the intentionally equivalent legacy adverse-extreme alias; all other
    roles use one exact canonical column.  The explicit meaningful-MFE event
    is a separate requirement so no caller can silently swap in a 1.5-ATR
    support label whose return-floor semantics differ.
    """

    name: str
    head_name: str
    task: str
    target_columns: tuple[str, ...]
    target_condition: str = "valid"
    quantile: float | None = None
    deployment_status: str = "candidate"
    allow_missing_on_valid: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if self.task not in {"binary", "regression", "quantile"}:
            raise ValueError(f"unsupported role task: {self.task}")
        if not self.name or not self.head_name or not self.target_columns:
            raise ValueError("role name, head name, and target columns are required")
        if self.task == "quantile":
            if self.quantile is None or not 0.0 < float(self.quantile) < 1.0:
                raise ValueError(
                    "a quantile role requires quantile strictly inside (0, 1)"
                )
        elif self.quantile is not None:
            raise ValueError("only quantile roles may declare quantile")
        if self.target_condition not in {
            "valid",
            "meaningful_hit",
            "meaningful_no_hit",
        }:
            raise ValueError(f"unsupported target condition: {self.target_condition}")
        if self.deployment_status not in {
            "candidate",
            "benchmark",
            "diagnostic_only",
        }:
            raise ValueError(f"unsupported deployment status: {self.deployment_status}")


@dataclass(frozen=True)
class HeadSpec:
    """Fixed semantic contract for one named auxiliary head."""

    name: str
    roles: tuple[RoleSpec, ...]
    deployment_status: str
    description: str
    promotion_requirement: str | None = None

    def __post_init__(self) -> None:
        if self.name not in {
            "peak_mfe_12h_atr",
            "time_to_first_meaningful_mfe",
            "mae_before_meaningful_mfe_atr",
            "bars_before_price_stops_decreasing",
            "future_slope_atr_per_hour",
        }:
            raise ValueError(f"unknown fixed auxiliary head: {self.name}")
        if not self.roles or any(role.head_name != self.name for role in self.roles):
            raise ValueError("every role must belong to its enclosing head")


PEAK_HEAD = HeadSpec(
    name="peak_mfe_12h_atr",
    deployment_status="candidate",
    description=(
        "Hurdle model for useful 12-hour peak MFE: meaningful-hit probability, "
        "then conditional natural-unit mean and upper-tail q80."
    ),
    roles=(
        RoleSpec(
            name="peak_mfe_12h_atr.p_hit",
            head_name="peak_mfe_12h_atr",
            task="binary",
            target_columns=(MEANINGFUL_HIT_COLUMN,),
            description="Probability that the explicit meaningful-MFE event occurs.",
        ),
        RoleSpec(
            name="peak_mfe_12h_atr.conditional_mean",
            head_name="peak_mfe_12h_atr",
            task="regression",
            target_columns=(PEAK_COLUMN,),
            target_condition="meaningful_hit",
            description="Peak MFE in ATR units conditional on a meaningful hit.",
        ),
        RoleSpec(
            name="peak_mfe_12h_atr.conditional_q80",
            head_name="peak_mfe_12h_atr",
            task="quantile",
            target_columns=(PEAK_COLUMN,),
            target_condition="meaningful_hit",
            quantile=0.80,
            description="Conditional 80th percentile of peak MFE in ATR units.",
        ),
    ),
)

TIMING_HEAD = HeadSpec(
    name="time_to_first_meaningful_mfe",
    deployment_status="candidate",
    description=(
        "Discrete CDF of the explicit meaningful-MFE event at 2/4/8/12 hours; "
        "unreached paths are retained as right-censored negatives."
    ),
    roles=tuple(
        RoleSpec(
            name=f"time_to_first_meaningful_mfe.hit_by_{hours}h",
            head_name="time_to_first_meaningful_mfe",
            task="binary",
            target_columns=(TIMING_COLUMN,),
            description=f"Probability of a meaningful-MFE hit by {hours} hours.",
        )
        for hours in TIMING_HORIZONS_HOURS
    ),
)

MAE_HEAD = HeadSpec(
    name="mae_before_meaningful_mfe_atr",
    deployment_status="candidate",
    description=(
        "Mixture model for adverse excursion: meaningful-hit probability, then "
        "separate pre-hit and no-hit/horizon risk regressions."
    ),
    roles=(
        RoleSpec(
            name="mae_before_meaningful_mfe_atr.p_hit",
            head_name="mae_before_meaningful_mfe_atr",
            task="binary",
            target_columns=(MEANINGFUL_HIT_COLUMN,),
            description="Probability of the explicit meaningful-MFE event.",
        ),
        RoleSpec(
            name="mae_before_meaningful_mfe_atr.if_hit",
            head_name="mae_before_meaningful_mfe_atr",
            task="regression",
            target_columns=(MAE_COLUMN,),
            target_condition="meaningful_hit",
            description="Pre-meaningful-event MAE in ATR units when it is reached.",
        ),
        RoleSpec(
            name="mae_before_meaningful_mfe_atr.if_no_hit",
            head_name="mae_before_meaningful_mfe_atr",
            task="regression",
            # The primary canonical MAE target already uses the full 12-hour
            # path when the *explicit* meaningful-MFE event is unreached.  Do
            # not use ``__mae_until_horizon_if_no_1_5atr__`` here: that
            # support target keys off an ATR-only event and can disagree with
            # the target's additional 1.5%-return floor.
            target_columns=(MAE_COLUMN,),
            target_condition="meaningful_no_hit",
            description=(
                "Full-horizon MAE in ATR units when the explicit meaningful event "
                "is not reached."
            ),
        ),
    ),
)

ADVERSE_HEAD = HeadSpec(
    name="bars_before_price_stops_decreasing",
    deployment_status="candidate",
    description=(
        "Compare the historical adverse-extreme clock with a two-bar-confirmed "
        "adverse-trough clock. The confirmed clock is the intended decision "
        "semantics; legacy remains an explicit benchmark rather than an alias."
    ),
    roles=(
        RoleSpec(
            name="bars_before_price_stops_decreasing.legacy_adverse_extreme",
            head_name="bars_before_price_stops_decreasing",
            task="regression",
            target_columns=LEGACY_ADVERSE_COLUMNS,
            deployment_status="benchmark",
            description="Legacy adverse extreme before meaningful MFE (one-based bar).",
        ),
        RoleSpec(
            name="bars_before_price_stops_decreasing.confirmed_adverse_trough",
            head_name="bars_before_price_stops_decreasing",
            task="regression",
            target_columns=(CONFIRMED_ADVERSE_COLUMN,),
            allow_missing_on_valid=True,
            description="Two-bar-confirmed adverse trough timing (one-based bar).",
        ),
    ),
)

SLOPE_HEAD = HeadSpec(
    name="future_slope_atr_per_hour",
    deployment_status="diagnostic_only",
    description=(
        "Favorable-path slope is retained only as a diagnostic until it adds "
        "incremental May-July OOF economic value beyond the peak and timing bundles."
    ),
    promotion_requirement=(
        "Demonstrate incremental side-local May-July OOF economic value beyond "
        "peak_mfe_12h_atr and time_to_first_meaningful_mfe before deployment."
    ),
    roles=(
        RoleSpec(
            name="future_slope_atr_per_hour.diagnostic",
            head_name="future_slope_atr_per_hour",
            task="regression",
            target_columns=(SLOPE_COLUMN,),
            deployment_status="diagnostic_only",
            description="Diagnostic 12-hour favorable slope in ATR per hour.",
        ),
    ),
)

# Stable tuple order is part of manifests and tests.  Do not add implicit
# family discovery: the roadmap calls for exactly these five head contracts.
HEAD_SPECS: tuple[HeadSpec, ...] = (
    PEAK_HEAD,
    TIMING_HEAD,
    MAE_HEAD,
    ADVERSE_HEAD,
    SLOPE_HEAD,
)
HEAD_SPECS_BY_NAME: Mapping[str, HeadSpec] = {spec.name: spec for spec in HEAD_SPECS}
ROLE_SPECS: tuple[RoleSpec, ...] = tuple(
    role for head in HEAD_SPECS for role in head.roles
)
ROLE_SPECS_BY_NAME: Mapping[str, RoleSpec] = {spec.name: spec for spec in ROLE_SPECS}


@dataclass(frozen=True)
class RoleTargets:
    """Aligned target vector and masks for one model role.

    ``target`` always has the source-frame length.  Values outside
    ``train_mask`` are ``NaN``; this makes accidental fitting on a conditional
    population visible without dropping or reordering the canonical rows.
    ``valid_mask`` means the path label itself is valid before the role's
    condition and finite-target rule are applied.
    """

    role: RoleSpec
    target: np.ndarray
    train_mask: np.ndarray
    valid_mask: np.ndarray
    source_column: str

    def __post_init__(self) -> None:
        n = len(self.target)
        if (
            self.target.ndim != 1
            or self.train_mask.shape != (n,)
            or self.valid_mask.shape != (n,)
        ):
            raise ValueError(
                "role targets and masks must be aligned one-dimensional arrays"
            )
        if np.any(self.train_mask & ~self.valid_mask):
            raise ValueError(
                "a role train mask cannot include invalid canonical labels"
            )
        if np.any(np.isfinite(self.target[~self.train_mask])):
            raise ValueError("non-training role targets must be NaN")


def _as_numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"canonical auxiliary label is missing: {column}")
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)


def _binary_column(
    frame: pd.DataFrame, column: str, valid_mask: np.ndarray
) -> np.ndarray:
    values = _as_numeric(frame, column)
    invalid = valid_mask & (~np.isfinite(values) | ~np.isin(values, (0.0, 1.0)))
    if np.any(invalid):
        raise ValueError(
            f"{column} must be finite binary on valid canonical rows; "
            f"found {int(invalid.sum())} invalid rows"
        )
    return values


def _select_source_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str:
    for column in candidates:
        if column in frame.columns:
            return column
    raise ValueError(
        f"canonical auxiliary label is missing; expected one of {list(candidates)!r}"
    )


def _require_valid_mask(frame: pd.DataFrame) -> np.ndarray:
    values = _as_numeric(frame, PATH_VALID_COLUMN)
    invalid = np.isfinite(values) & ~np.isin(values, (0.0, 1.0))
    if np.any(invalid):
        raise ValueError(f"{PATH_VALID_COLUMN} must contain only 0/1 values")
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{PATH_VALID_COLUMN} must be present for every canonical row")
    return values.astype(bool)


def _role_requested(spec: RoleSpec, requested: set[str] | None) -> bool:
    return requested is None or spec.name in requested


def validate_canonical_auxiliary_labels(
    frame: pd.DataFrame,
    *,
    role_names: Iterable[str] | None = None,
) -> None:
    """Validate canonical inputs for the requested model roles.

    This deliberately fails if a meaningful-MFE role has only
    ``__mfe_ge_1_5atr__`` available.  That support label omits the target's
    1.5%-of-entry return floor and is therefore not interchangeable.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("canonical auxiliary labels must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("canonical auxiliary labels cannot be empty")
    requested = None if role_names is None else set(map(str, role_names))
    unknown = set() if requested is None else requested.difference(ROLE_SPECS_BY_NAME)
    if unknown:
        raise ValueError(f"unknown auxiliary role names: {sorted(unknown)}")
    valid = _require_valid_mask(frame)
    selected = [spec for spec in ROLE_SPECS if _role_requested(spec, requested)]
    needs_meaningful = any(
        spec.head_name
        in {
            "peak_mfe_12h_atr",
            "time_to_first_meaningful_mfe",
            "mae_before_meaningful_mfe_atr",
        }
        for spec in selected
    )
    if needs_meaningful:
        meaningful = _binary_column(frame, MEANINGFUL_HIT_COLUMN, valid)
        timing_required = any(
            spec.head_name == "time_to_first_meaningful_mfe" for spec in selected
        )
        if timing_required:
            time_values = _as_numeric(frame, TIMING_COLUMN)
            bad_time = valid & (
                ~np.isfinite(time_values)
                | (time_values < 0.0)
                | (time_values > float(TIMING_HORIZONS_HOURS[-1]))
            )
            if np.any(bad_time):
                raise ValueError(
                    f"{TIMING_COLUMN} must be finite inside [0, 12] on valid rows"
                )
            # A hit at a finite timing clock is part of the canonical target
            # geometry. Unreached rows are right censored at 12h.
            inconsistent = valid & (meaningful > 0.5) & (time_values <= 0.0)
            if np.any(inconsistent):
                raise ValueError(
                    "meaningful hits require a strictly positive timing clock"
                )
    for spec in selected:
        source = _select_source_column(frame, spec.target_columns)
        if spec.task == "binary":
            # Timing binary roles derive from timing plus the explicit event,
            # while p_hit uses the explicit event as its source column.
            if spec.target_columns == (MEANINGFUL_HIT_COLUMN,):
                _binary_column(frame, source, valid)
            continue
        values = _as_numeric(frame, source)
        condition = valid.copy()
        if spec.target_condition == "meaningful_hit":
            condition &= _binary_column(frame, MEANINGFUL_HIT_COLUMN, valid) > 0.5
        elif spec.target_condition == "meaningful_no_hit":
            condition &= _binary_column(frame, MEANINGFUL_HIT_COLUMN, valid) <= 0.5
        bad = condition & (values < 0.0)
        if not spec.allow_missing_on_valid:
            bad |= condition & ~np.isfinite(values)
        if np.any(bad):
            raise ValueError(
                f"{source} must be finite and non-negative for role {spec.name}; "
                f"found {int(bad.sum())} invalid rows"
            )


def build_role_targets(
    frame: pd.DataFrame,
    *,
    role_names: Iterable[str] | None = None,
) -> dict[str, RoleTargets]:
    """Build aligned canonical role targets without dropping any source row."""

    validate_canonical_auxiliary_labels(frame, role_names=role_names)
    requested = None if role_names is None else set(map(str, role_names))
    valid = _require_valid_mask(frame)
    meaningful = (
        _binary_column(frame, MEANINGFUL_HIT_COLUMN, valid)
        if any(
            spec.head_name
            in {
                "peak_mfe_12h_atr",
                "time_to_first_meaningful_mfe",
                "mae_before_meaningful_mfe_atr",
            }
            and _role_requested(spec, requested)
            for spec in ROLE_SPECS
        )
        else np.full(len(frame), np.nan, dtype=np.float64)
    )
    result: dict[str, RoleTargets] = {}
    for spec in ROLE_SPECS:
        if not _role_requested(spec, requested):
            continue
        source = _select_source_column(frame, spec.target_columns)
        train_mask = valid.copy()
        target = np.full(len(frame), np.nan, dtype=np.float64)
        if spec.name.startswith("time_to_first_meaningful_mfe.hit_by_"):
            time_values = _as_numeric(frame, TIMING_COLUMN)
            hours = int(spec.name.rsplit("_", 1)[1].removesuffix("h"))
            train_mask &= np.isfinite(time_values)
            target[train_mask] = (
                (meaningful[train_mask] > 0.5) & (time_values[train_mask] <= hours)
            ).astype(np.float64)
        elif spec.task == "binary":
            source_values = _binary_column(frame, source, valid)
            target[train_mask] = source_values[train_mask]
        else:
            source_values = _as_numeric(frame, source)
            if spec.target_condition == "meaningful_hit":
                train_mask &= meaningful > 0.5
            elif spec.target_condition == "meaningful_no_hit":
                train_mask &= meaningful <= 0.5
            train_mask &= np.isfinite(source_values) & (source_values >= 0.0)
            target[train_mask] = source_values[train_mask]
        result[spec.name] = RoleTargets(
            role=spec,
            target=target,
            train_mask=train_mask,
            valid_mask=valid.copy(),
            source_column=source,
        )
    return result


def _vector(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _probability(values: Any, *, name: str) -> np.ndarray:
    array = _vector(values, name=name)
    if not np.isfinite(array).all() or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must be finite probabilities inside [0, 1]")
    return array


def _non_negative(values: Any, *, name: str) -> np.ndarray:
    array = _vector(values, name=name)
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise ValueError(f"{name} must be finite non-negative natural-unit values")
    return array


def _same_length(**arrays: np.ndarray) -> int:
    lengths = {name: len(array) for name, array in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"prediction inputs must have equal lengths: {lengths}")
    return next(iter(lengths.values()), 0)


def compose_peak_predictions(
    p_hit: Any,
    conditional_mean_atr: Any,
    conditional_q80_atr: Any,
) -> dict[str, np.ndarray]:
    """Compose peak-hurdle outputs in natural ATR units."""

    probability = _probability(p_hit, name="p_hit")
    mean = _non_negative(conditional_mean_atr, name="conditional_mean_atr")
    q80 = _non_negative(conditional_q80_atr, name="conditional_q80_atr")
    _same_length(p_hit=probability, conditional_mean_atr=mean, conditional_q80_atr=q80)
    return {
        "p_hit": probability,
        "conditional_mean_atr": mean,
        "conditional_q80_atr": q80,
        "expected_peak_mfe_atr": probability * mean,
    }


def project_monotone_timing_cdf(
    cdf_by_horizon: Mapping[int | float, Any],
    *,
    horizons: Sequence[int | float] = TIMING_HORIZONS_HOURS,
    preserve_final_horizon: bool = True,
) -> dict[float, np.ndarray]:
    """Validate and isotonic-project CDF probabilities across horizons.

    The projection is the least non-decreasing vector in squared-error space
    for each row (pool-adjacent-violators), rather than merely a cumulative
    maximum.  By default the final-horizon probability is an immutable shared
    meaningful-event prediction: earlier horizons are capped at that value and
    projected without changing it.
    """

    ordered = tuple(float(hour) for hour in horizons)
    if ordered != tuple(sorted(ordered)) or len(set(ordered)) != len(ordered):
        raise ValueError("timing horizons must be strictly increasing")
    normalized = {
        float(key): _probability(value, name=f"cdf[{key}]")
        for key, value in cdf_by_horizon.items()
    }
    missing = [hour for hour in ordered if hour not in normalized]
    extra = sorted(set(normalized).difference(ordered))
    if missing or extra:
        raise ValueError(
            f"timing CDF horizons mismatch; missing={missing}, extra={extra}"
        )
    matrix = np.column_stack([normalized[hour] for hour in ordered])
    _same_length(**{str(hour): normalized[hour] for hour in ordered})
    final = matrix[:, -1].copy()
    projected_input = (
        np.minimum(matrix[:, :-1], final[:, None]) if preserve_final_horizon else matrix
    )
    # PAV with unit weights; number of timing bins is intentionally tiny.
    projected = np.empty_like(projected_input)
    for row_i, row in enumerate(projected_input):
        levels: list[float] = []
        counts: list[int] = []
        for value in row:
            levels.append(float(value))
            counts.append(1)
            while len(levels) >= 2 and levels[-2] > levels[-1]:
                total = counts[-2] + counts[-1]
                pooled = (levels[-2] * counts[-2] + levels[-1] * counts[-1]) / total
                levels[-2:] = [pooled]
                counts[-2:] = [total]
        pos = 0
        for level, count in zip(levels, counts):
            projected[row_i, pos : pos + count] = level
            pos += count
    if preserve_final_horizon:
        projected = np.column_stack([projected, final])
    return {hour: projected[:, i] for i, hour in enumerate(ordered)}


def compose_timing_cdf_predictions(
    cdf_by_horizon: Mapping[int | float, Any],
    *,
    horizons: Sequence[int | float] = TIMING_HORIZONS_HOURS,
) -> dict[str, np.ndarray]:
    """Project timing CDFs and return natural-hour, right-censored summaries."""

    ordered = tuple(float(hour) for hour in horizons)
    cdf = project_monotone_timing_cdf(cdf_by_horizon, horizons=ordered)
    matrix = np.column_stack([cdf[hour] for hour in ordered])
    masses = np.column_stack([matrix[:, 0], np.diff(matrix, axis=1)])
    # Events within an interval are represented at its upper endpoint.  The
    # final 12h event and right-censor both contribute 12h, which is exactly
    # the observable resolution horizon.
    expected_censored_hours = (
        masses @ np.asarray(ordered, dtype=np.float64)
        + (1.0 - matrix[:, -1]) * ordered[-1]
    )
    return {
        **{
            f"p_hit_by_{int(hour) if hour.is_integer() else hour}h": cdf[hour]
            for hour in ordered
        },
        "p_hit_12h": cdf[ordered[-1]],
        "expected_censored_time_hours": expected_censored_hours,
    }


def compose_mae_predictions(
    p_hit: Any,
    mae_if_hit_atr: Any,
    mae_if_no_hit_atr: Any,
) -> dict[str, np.ndarray]:
    """Compose separate hit/no-hit MAE risks in natural ATR units."""

    probability = _probability(p_hit, name="p_hit")
    if_hit = _non_negative(mae_if_hit_atr, name="mae_if_hit_atr")
    if_no_hit = _non_negative(mae_if_no_hit_atr, name="mae_if_no_hit_atr")
    _same_length(p_hit=probability, mae_if_hit_atr=if_hit, mae_if_no_hit_atr=if_no_hit)
    return {
        "p_hit": probability,
        "mae_if_hit_atr": if_hit,
        "mae_if_no_hit_atr": if_no_hit,
        "expected_mae_atr": probability * if_hit + (1.0 - probability) * if_no_hit,
    }


def compose_adverse_timing_predictions(
    legacy_adverse_extreme_bars: Any,
    confirmed_adverse_trough_bars: Any,
) -> dict[str, np.ndarray]:
    """Expose both adverse-timing variants in their shared natural unit (bars)."""

    legacy = _non_negative(
        legacy_adverse_extreme_bars, name="legacy_adverse_extreme_bars"
    )
    confirmed = _non_negative(
        confirmed_adverse_trough_bars, name="confirmed_adverse_trough_bars"
    )
    _same_length(
        legacy_adverse_extreme_bars=legacy, confirmed_adverse_trough_bars=confirmed
    )
    return {
        "legacy_adverse_extreme_bars": legacy,
        "confirmed_adverse_trough_bars": confirmed,
        "confirmed_minus_legacy_bars": confirmed - legacy,
    }


def compose_slope_diagnostic_predictions(
    slope_atr_per_hour: Any,
) -> dict[str, np.ndarray]:
    """Return the non-deployable slope diagnostic in natural ATR/hour units."""

    slope = _non_negative(slope_atr_per_hour, name="slope_atr_per_hour")
    return {
        "future_slope_atr_per_hour": slope,
        "deployment_status": np.repeat("diagnostic_only", len(slope)),
    }


def _metric_arrays(
    y_true: Any,
    y_pred: Any,
    *,
    sample_weight: Any | None = None,
    mask: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    truth = _vector(y_true, name="y_true")
    pred = _vector(y_pred, name="y_pred")
    _same_length(y_true=truth, y_pred=pred)
    keep = np.isfinite(truth) & np.isfinite(pred)
    if mask is not None:
        requested = np.asarray(mask, dtype=bool)
        if requested.shape != truth.shape:
            raise ValueError("metric mask must align to y_true")
        keep &= requested
    weights: np.ndarray | None = None
    if sample_weight is not None:
        weights = _vector(sample_weight, name="sample_weight")
        _same_length(y_true=truth, sample_weight=weights)
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError("sample_weight must be finite and non-negative")
        keep &= weights > 0.0
        weights = weights[keep]
        if len(weights) and float(weights.sum()) <= 0.0:
            raise ValueError("sample_weight must retain positive mass")
    return truth[keep], pred[keep], weights


def calibration_reliability_table(
    y_true: Any,
    p_pred: Any,
    *,
    n_bins: int = 10,
    sample_weight: Any | None = None,
    mask: Any | None = None,
) -> list[dict[str, float | int]]:
    """Return fixed-width, weight-aware calibration bins for a binary role."""

    if int(n_bins) < 2:
        raise ValueError("n_bins must be at least two")
    truth, probability, weights = _metric_arrays(
        y_true, p_pred, sample_weight=sample_weight, mask=mask
    )
    if np.any(~np.isin(truth, (0.0, 1.0))):
        raise ValueError("binary metrics require y_true values in {0, 1}")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("binary metrics require p_pred values inside [0, 1]")
    if weights is None:
        weights = np.ones(len(truth), dtype=np.float64)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    bucket = np.minimum(
        np.searchsorted(edges, probability, side="right") - 1, int(n_bins) - 1
    )
    rows: list[dict[str, float | int]] = []
    for bin_i in range(int(n_bins)):
        selected = bucket == bin_i
        if not np.any(selected):
            continue
        local_weight = weights[selected]
        mass = float(local_weight.sum())
        mean_prediction = float(np.average(probability[selected], weights=local_weight))
        observed_rate = float(np.average(truth[selected], weights=local_weight))
        rows.append(
            {
                "bin": int(bin_i),
                "lower": float(edges[bin_i]),
                "upper": float(edges[bin_i + 1]),
                "rows": int(selected.sum()),
                "weight": mass,
                "mean_prediction": mean_prediction,
                "observed_rate": observed_rate,
                "absolute_gap": abs(observed_rate - mean_prediction),
            }
        )
    return rows


def probability_calibration_metrics(
    y_true: Any,
    p_pred: Any,
    *,
    n_bins: int = 10,
    sample_weight: Any | None = None,
    mask: Any | None = None,
) -> dict[str, Any]:
    """Compute binary discrimination, Brier/log loss, and calibration metrics."""

    truth, probability, weights = _metric_arrays(
        y_true, p_pred, sample_weight=sample_weight, mask=mask
    )
    if np.any(~np.isin(truth, (0.0, 1.0))):
        raise ValueError("binary metrics require y_true values in {0, 1}")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("binary metrics require p_pred values inside [0, 1]")
    if weights is None:
        weights = np.ones(len(truth), dtype=np.float64)
    if not len(truth):
        return {
            "rows": 0,
            "weight": 0.0,
            "prevalence": np.nan,
            "brier": np.nan,
            "log_loss": np.nan,
            "roc_auc": np.nan,
            "ece": np.nan,
            "mce": np.nan,
            "calibration_bins": [],
        }
    total_weight = float(weights.sum())
    prevalence = float(np.average(truth, weights=weights))
    brier = float(np.average((probability - truth) ** 2, weights=weights))
    clipped = np.clip(probability, 1e-15, 1.0 - 1e-15)
    log_loss = float(
        -np.average(
            truth * np.log(clipped) + (1.0 - truth) * np.log1p(-clipped),
            weights=weights,
        )
    )
    table = calibration_reliability_table(
        truth, probability, n_bins=n_bins, sample_weight=weights
    )
    gaps = np.asarray([float(row["absolute_gap"]) for row in table], dtype=np.float64)
    masses = np.asarray([float(row["weight"]) for row in table], dtype=np.float64)
    ece = float(np.sum(gaps * masses) / total_weight) if len(gaps) else np.nan
    mce = float(np.max(gaps)) if len(gaps) else np.nan
    try:
        auc = (
            float(roc_auc_score(truth, probability, sample_weight=weights))
            if np.unique(truth).size == 2
            else np.nan
        )
    except ValueError:
        auc = np.nan
    return {
        "rows": int(len(truth)),
        "weight": total_weight,
        "prevalence": prevalence,
        "brier": brier,
        "log_loss": log_loss,
        "roc_auc": auc,
        "ece": ece,
        "mce": mce,
        "calibration_bins": table,
    }


def regression_metrics(
    y_true: Any,
    y_pred: Any,
    *,
    sample_weight: Any | None = None,
    mask: Any | None = None,
) -> dict[str, float | int]:
    """Natural-unit regression metrics with safe empty/constant handling."""

    truth, prediction, weights = _metric_arrays(
        y_true, y_pred, sample_weight=sample_weight, mask=mask
    )
    if weights is None:
        weights = np.ones(len(truth), dtype=np.float64)
    if not len(truth):
        return {
            "rows": 0,
            "weight": 0.0,
            "mean_true": np.nan,
            "mean_pred": np.nan,
            "bias": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "spearman_ic": np.nan,
        }
    error = prediction - truth
    weight_total = float(weights.sum())
    mae = float(np.average(np.abs(error), weights=weights))
    rmse = float(np.sqrt(np.average(error**2, weights=weights)))
    if len(truth) >= 2 and np.unique(truth).size > 1 and np.unique(prediction).size > 1:
        correlation = float(spearmanr(truth, prediction).statistic)
    else:
        correlation = np.nan
    return {
        "rows": int(len(truth)),
        "weight": weight_total,
        "mean_true": float(np.average(truth, weights=weights)),
        "mean_pred": float(np.average(prediction, weights=weights)),
        "bias": float(np.average(error, weights=weights)),
        "mae": mae,
        "rmse": rmse,
        "spearman_ic": correlation,
    }


def conditional_regression_metrics(
    y_true: Any,
    y_pred: Any,
    meaningful_hit: Any,
    *,
    sample_weight: Any | None = None,
    mask: Any | None = None,
) -> dict[str, dict[str, float | int]]:
    """Report overall, meaningful-hit, and no-hit regression performance."""

    hit = _vector(meaningful_hit, name="meaningful_hit")
    truth = _vector(y_true, name="y_true")
    _same_length(y_true=truth, meaningful_hit=hit)
    if np.any(np.isfinite(hit) & ~np.isin(hit, (0.0, 1.0))):
        raise ValueError("meaningful_hit must contain only 0/1 or NaN")
    base_mask = (
        np.ones(len(truth), dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool)
    )
    if base_mask.shape != truth.shape:
        raise ValueError("metric mask must align to y_true")
    return {
        "overall": regression_metrics(
            y_true, y_pred, sample_weight=sample_weight, mask=base_mask
        ),
        "if_hit": regression_metrics(
            y_true,
            y_pred,
            sample_weight=sample_weight,
            mask=base_mask & (hit > 0.5),
        ),
        "if_no_hit": regression_metrics(
            y_true,
            y_pred,
            sample_weight=sample_weight,
            mask=base_mask & (hit <= 0.5),
        ),
    }


__all__ = [
    "ADVERSE_HEAD",
    "CONFIRMED_ADVERSE_COLUMN",
    "HEAD_SPECS",
    "HEAD_SPECS_BY_NAME",
    "LEGACY_ADVERSE_COLUMNS",
    "MAE_HEAD",
    "MEANINGFUL_HIT_COLUMN",
    "MODEL_FAMILY_SCHEMA",
    "PATH_VALID_COLUMN",
    "PEAK_HEAD",
    "ROLE_SPECS",
    "ROLE_SPECS_BY_NAME",
    "SLOPE_HEAD",
    "TIMING_HEAD",
    "TIMING_HORIZONS_HOURS",
    "HeadSpec",
    "RoleSpec",
    "RoleTargets",
    "build_role_targets",
    "calibration_reliability_table",
    "compose_adverse_timing_predictions",
    "compose_mae_predictions",
    "compose_peak_predictions",
    "compose_slope_diagnostic_predictions",
    "compose_timing_cdf_predictions",
    "conditional_regression_metrics",
    "probability_calibration_metrics",
    "project_monotone_timing_cdf",
    "regression_metrics",
    "validate_canonical_auxiliary_labels",
]
