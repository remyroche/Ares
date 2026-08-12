"""Causal feature-portability transforms and diagnostics.

This module deliberately has no model, target-construction, or runner
dependency.  It provides one small contract for research code that needs to
answer four distinct questions before a field is promoted into a reusable
base/meta feature set:

* what is the field's semantic role;
* can a scale-dependent field be made portable with point-in-time transforms;
* is coverage, support, distribution, and optional outcome association stable
  across declared eras; and
* should the field be kept, transformed, reviewed, diagnostic-only, or
  excluded.

The rolling transforms are per declared entity, sorted by UTC decision time,
and use only the current and earlier observations.  They never fit a statistic
on a later era.  The diagnostic helpers only describe a supplied frame; any
caller that uses outcome effects for selection must still keep its evaluation
fold outside that selection process.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd


EPS = 1e-12


class FeaturePortabilityError(ValueError):
    """Raised when a portability input violates its explicit data contract."""


class FeatureRole:
    """String constants used in the role and disposition tables."""

    PORTABLE = "portable"
    CAUSAL_TRANSFORM_REQUIRED = "causal_transform_required"
    IDENTITY = "identity"
    FOLD_LOCAL_STATE = "fold_local_state"
    LATENT_REGIME_OUTPUT = "latent_regime_output"
    OUTCOME_DERIVED = "outcome_derived"
    CONTROL = "control"
    UNKNOWN = "unknown"


class FeatureSemanticRole:
    """The Stage-A taxonomy used to decide how a portable field may be used."""

    LEVEL = "LEVEL"
    RELATIVE_LEVEL = "RELATIVE_LEVEL"
    CHANGE = "CHANGE"
    ACCELERATION = "ACCELERATION"
    RELATIONSHIP_BREAK = "RELATIONSHIP_BREAK"
    SETUP_ALIGNMENT = "SETUP_ALIGNMENT"
    SUPPORT_OR_TRUST = "SUPPORT_OR_TRUST"
    MODEL_OUTPUT = "MODEL_OUTPUT"


@dataclass(frozen=True)
class PortabilityPolicy:
    """Conservative, target-agnostic thresholds for a feature disposition.

    The policy is a diagnostic contract, not a replacement for side-local MDA
    or chronological OOS selection.  ``max_*`` values are deliberately
    review gates: crossing one produces a review disposition rather than a
    silent automatic drop.
    """

    # Stage-A readiness is deliberately tighter than ordinary feature
    # selection: an input unavailable on more than 1% of rows is not a
    # production-portable feature merely because a model can impute it.
    min_coverage: float = 0.99
    min_finite_support: int = 100
    min_unique_values: int = 2
    # The pre-registered support gate compares the test distribution to the
    # train p0.5--p99.5 envelope.  With a 5% cap it catches material support
    # loss without rejecting a stationary distribution by construction.
    max_extrapolation_rate: float = 0.05
    min_bin_support: int = 5
    min_bins_represented: int = 8
    max_abs_robust_median_shift: float = 3.0
    max_psi: float = 0.25
    min_effect_support: int = 100
    max_effect_range: float = 0.20
    min_effect_magnitude_for_sign: float = 0.02

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_coverage <= 1.0 or not 0.0 <= self.max_extrapolation_rate <= 1.0:
            raise FeaturePortabilityError("coverage and extrapolation rates must be in [0, 1]")
        if self.min_finite_support < 1 or self.min_unique_values < 1 or self.min_bin_support < 1:
            raise FeaturePortabilityError("support and unique-value thresholds must be positive")
        if not 1 <= self.min_bins_represented <= 10:
            raise FeaturePortabilityError("min_bins_represented must be in [1, 10]")
        if self.max_abs_robust_median_shift < 0.0 or self.max_psi < 0.0:
            raise FeaturePortabilityError("drift thresholds must be non-negative")
        if self.min_effect_support < 1 or self.max_effect_range < 0.0:
            raise FeaturePortabilityError("effect thresholds are invalid")


@dataclass(frozen=True)
class CausalTransformMemoryEstimate:
    """Lower-bound output and bounded scratch estimates for rolling transforms.

    ``materialized_output_bytes`` is unavoidable when a caller elects to keep
    every generated float32 column in memory.  ``peak_batch_working_bytes`` is
    the deliberately bounded temporary footprint of the batched API; it does
    not include the caller's source frame or model matrix.
    """

    rows: int
    source_features: int
    generated_columns_per_source: int
    feature_batch_size: int
    materialized_output_bytes: int
    peak_batch_working_bytes: int


def estimate_causal_rolling_transform_memory(
    *,
    rows: int,
    source_features: int,
    rank_windows: Sequence[int] = (96,),
    robust_z_windows: Sequence[int] = (96,),
    change_periods: Sequence[int] = (1,),
    include_relative_change: bool = True,
    feature_batch_size: int = 1,
) -> CausalTransformMemoryEstimate:
    """Return a deterministic, conservative memory estimate before work starts."""
    if rows < 0 or source_features < 0 or feature_batch_size < 1:
        raise FeaturePortabilityError("rows/source_features must be non-negative and batch size positive")
    output_per_source = (
        len(tuple(dict.fromkeys(rank_windows)))
        + len(tuple(dict.fromkeys(robust_z_windows)))
        + (1 + int(bool(include_relative_change))) * len(tuple(dict.fromkeys(change_periods)))
    )
    batch_features = min(int(source_features), int(feature_batch_size))
    # Per source: float32 output columns plus ordered/current float64 vectors,
    # three robust rolling summaries, one rank/change vector, and indices.  It
    # deliberately overstates the temporary requirement instead of pretending
    # pandas rolling internals are free.
    per_row_per_source = output_per_source * np.dtype(np.float32).itemsize + 7 * np.dtype(np.float64).itemsize + np.dtype(np.int64).itemsize
    return CausalTransformMemoryEstimate(
        rows=int(rows), source_features=int(source_features),
        generated_columns_per_source=int(output_per_source), feature_batch_size=int(feature_batch_size),
        materialized_output_bytes=int(rows) * int(source_features) * int(output_per_source) * np.dtype(np.float32).itemsize,
        peak_batch_working_bytes=int(rows) * int(batch_features) * int(per_row_per_source) + int(rows) * (np.dtype(np.int64).itemsize + np.dtype(np.int32).itemsize),
    )


_OUTCOME_RE = re.compile(
    r"(?:^|_)(?:label|target|outcome|realized|realised|future|mfe|mae|pnl|"
    r"first_touch|exit|take_profit|stop_loss|path|event|exact_net|"
    r"exact_gross|expected_net|mapped_net|residual)(?:_|$)",
    flags=re.IGNORECASE,
)
_CONTROL_RE = re.compile(
    r"(?:^|_)(?:timestamp|datetime|date|month|week|year|candidate_id|row_id|"
    r"fold|split|source_month|label_available)(?:_|$)|^ts$",
    flags=re.IGNORECASE,
)
_IDENTITY_RE = re.compile(
    r"^(?:symbol|asset|instrument|ticker|exchange|venue|quote_currency|contract|strategy)$|"
    r"(?:^|_)(?:symbol_id|asset_id|instrument_id|ticker_id|exchange_id|venue_id)(?:_|$)",
    flags=re.IGNORECASE,
)
_FOLD_LOCAL_STATE_RE = re.compile(
    r"(?:^|_)(?:state_id|cluster_id|archetype_id|leaf_id|regime_id)(?:_|$)",
    flags=re.IGNORECASE,
)
_LATENT_REGIME_RE = re.compile(
    r"(?:^|_)(?:latent|gmm|autoencoder|archetype|cluster|membership|posterior|"
    r"reconstruction)(?:_|$)|(?:^|_)ae(?:_|\d|$)|(?:^|_)regime_state_p?_?\d+(?:_|$)",
    flags=re.IGNORECASE,
)
_ACCELERATION_RE = re.compile(r"(?:^|_)(?:accel(?:eration)?|velocity|second_diff|curvature)(?:_|$)", re.IGNORECASE)
_CHANGE_RE = re.compile(r"(?:^|_)(?:chg|change|delta|diff|impulse)(?:_|$)", re.IGNORECASE)
_RELATIONSHIP_BREAK_RE = re.compile(
    r"(?:^|_)(?:break|divergen(?:ce|t)|dislocation|decoupl(?:e|ing)|structural)(?:_|$)", re.IGNORECASE
)
_SETUP_ALIGNMENT_RE = re.compile(
    r"(?:^|_)(?:setup|alignment|align|confluence|trigger|condition)(?:_|$)", re.IGNORECASE
)
_SUPPORT_TRUST_RE = re.compile(
    r"(?:^|_)(?:support|trust|ood|drift|uncertainty|confidence|reliability|coverage)(?:_|$)", re.IGNORECASE
)
_MODEL_OUTPUT_RE = re.compile(
    r"^(?:base|meta|model|prediction|pred|probability|logit)(?:_|$)|^(?:p|prob)_(?:clear|weak|adverse|\d+)(?:_|$)",
)
_PORTABLE_RE = re.compile(
    r"(?:ret(?:urn)?(?:_|$)|pct(?:_|$)|bps(?:_|$)|atr(?:_|$)|z(?:score)?(?:_|$)|"
    r"rank(?:_|$)|percentile(?:_|$)|ratio(?:_|$)|relative(?:_|$)|normalized(?:_|$)|"
    r"normalised(?:_|$)|fraction(?:_|$)|entropy(?:_|$)|posterior(?:_|$)|probability(?:_|$)|"
    r"(?:^|_)p_\d+(?:_|$)|vol(?:atility)?(?:_|$)|spread(?:_|$)|distance(?:_|$)|"
    r"margin(?:_|$)|drawdown(?:_|$)|correlation(?:_|$)|corr(?:_|$)|beta(?:_|$)|slope(?:_|$)|"
    r"efficiency(?:_|$)|chop(?:_|$)|switch(?:_|$)|transition(?:_|$)|age(?:_|$)|"
    r"duration(?:_|$)|bars(?:_|$)|resid(?:_|$)|loc(?:_|$)|imbalance(?:_|$)|pressure(?:_|$)|recovery(?:_|$)|intensity(?:_|$)|"
    r"funding(?:_|$)|liquidity(?:_|$)|range(?:_|$)|donchian(?:_|$)|wick(?:_|$)|"
    r"flush(?:_|$)|exhaustion(?:_|$)|phase(?:_|$)|covering(?:_|$)|oiw(?:_|$)|"
    r"support(?:_|$)|resistance(?:_|$)|divergence(?:_|$)|deleveraging(?:_|$)|rebound(?:_|$)|"
    r"resilience(?:_|$)|breadth(?:_|$)|deceleration(?:_|$)|breakout(?:_|$)|(?:^|_)is_long(?:_|$))",
    flags=re.IGNORECASE,
)
_RAW_LEVEL_RE = re.compile(
    r"(?:^|_)(?:open|high|low|close|price|vwap|volume|notional|market_cap|"
    r"open_interest|oi_value|liquidity|depth|turnover|trade_count|tick_size)(?:_|$)|"
    r"(?:^|_)(?:usd|usdt|eur|gbp)(?:_|$)",
    flags=re.IGNORECASE,
)


def classify_feature_role(
    feature_name: str,
    *,
    overrides: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    """Classify one feature name without inspecting realised outcomes.

    Exact-name ``overrides`` are deliberately applied first.  The built-in
    rules then bias toward safety: fields that look outcome-derived are
    excluded before portable-unit suffixes are considered; raw scale-dependent
    levels are marked for causal transformation rather than silently promoted.
    """
    name = str(feature_name)
    if not name.strip():
        raise FeaturePortabilityError("feature names must be non-empty")
    if overrides and name in overrides:
        role = str(overrides[name])
        valid = {
            FeatureRole.PORTABLE,
            FeatureRole.CAUSAL_TRANSFORM_REQUIRED,
            FeatureRole.IDENTITY,
            FeatureRole.FOLD_LOCAL_STATE,
            FeatureRole.LATENT_REGIME_OUTPUT,
            FeatureRole.OUTCOME_DERIVED,
            FeatureRole.CONTROL,
            FeatureRole.UNKNOWN,
        }
        if role not in valid:
            raise FeaturePortabilityError(f"unknown role override {role!r} for {name!r}")
        return role, "explicit_override"
    lower = name.lower()
    if _OUTCOME_RE.search(lower):
        return FeatureRole.OUTCOME_DERIVED, "outcome_or_path_namespace"
    if _CONTROL_RE.search(lower):
        return FeatureRole.CONTROL, "row_or_split_control_namespace"
    if _IDENTITY_RE.search(lower):
        return FeatureRole.IDENTITY, "asset_or_venue_identity_namespace"
    if _LATENT_REGIME_RE.search(lower):
        return FeatureRole.LATENT_REGIME_OUTPUT, "latent_regime_representation_namespace"
    if _FOLD_LOCAL_STATE_RE.search(lower):
        return FeatureRole.FOLD_LOCAL_STATE, "hard_fold_local_state_namespace"
    if _PORTABLE_RE.search(lower):
        return FeatureRole.PORTABLE, "portable_unit_or_context_namespace"
    if _RAW_LEVEL_RE.search(lower):
        return FeatureRole.CAUSAL_TRANSFORM_REQUIRED, "raw_scale_dependent_namespace"
    return FeatureRole.UNKNOWN, "no_portability_semantics_match"


def _semantic_role(feature_name: str, portability_role: str) -> str:
    """Classify use semantics separately from safety/lineage semantics."""
    lower = str(feature_name).lower()
    if _SUPPORT_TRUST_RE.search(lower):
        return FeatureSemanticRole.SUPPORT_OR_TRUST
    if _RELATIONSHIP_BREAK_RE.search(lower):
        return FeatureSemanticRole.RELATIONSHIP_BREAK
    if _SETUP_ALIGNMENT_RE.search(lower):
        return FeatureSemanticRole.SETUP_ALIGNMENT
    if _ACCELERATION_RE.search(lower):
        return FeatureSemanticRole.ACCELERATION
    if _CHANGE_RE.search(lower):
        return FeatureSemanticRole.CHANGE
    if _MODEL_OUTPUT_RE.search(lower):
        return FeatureSemanticRole.MODEL_OUTPUT
    if portability_role == FeatureRole.CAUSAL_TRANSFORM_REQUIRED:
        return FeatureSemanticRole.LEVEL
    if portability_role == FeatureRole.PORTABLE:
        return FeatureSemanticRole.RELATIVE_LEVEL
    # Unsafe/control fields keep the narrow MODEL_OUTPUT label; their lineage
    # still controls the final REJECTED_LINEAGE disposition.
    return FeatureSemanticRole.MODEL_OUTPUT


def classify_feature_roles(
    feature_names: Iterable[str],
    *,
    overrides: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Return Stage-A semantic roles plus independent portability lineage."""
    names = [str(name) for name in feature_names]
    if len(set(names)) != len(names):
        raise FeaturePortabilityError("feature-role inventory requires unique names")
    rows = []
    for name in names:
        portability_role, reason = classify_feature_role(name, overrides=overrides)
        rows.append({
            "feature": name,
            "role": _semantic_role(name, portability_role),
            "portability_role": portability_role,
            "role_reason": reason,
        })
    return pd.DataFrame(rows, columns=["feature", "role", "portability_role", "role_reason"])


def _validate_columns(frame: pd.DataFrame, names: Sequence[str], *, kind: str) -> None:
    missing = [name for name in names if name not in frame]
    if missing:
        raise FeaturePortabilityError(f"missing {kind} columns: {missing[:12]}")


def _ordered_frame(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    group_columns: Sequence[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    _validate_columns(frame, [timestamp_column, *group_columns], kind="rolling-key")
    timestamp = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
    if timestamp.isna().any():
        raise FeaturePortabilityError("rolling transforms require finite UTC decision timestamps")
    if group_columns and frame.loc[:, list(group_columns)].isna().any().any():
        raise FeaturePortabilityError("rolling transforms require non-missing entity keys")
    ordered = frame.copy(deep=False)
    ordered = ordered.assign(
        __feature_portability_ts__=timestamp.to_numpy(),
        __feature_portability_position__=np.arange(len(frame), dtype=np.int64),
    )
    ordered = ordered.sort_values(
        [*group_columns, "__feature_portability_ts__", "__feature_portability_position__"],
        kind="stable",
    ).reset_index(drop=True)
    if group_columns:
        ordered["__feature_portability_group__"] = pd.factorize(
            pd.MultiIndex.from_frame(ordered.loc[:, list(group_columns)]), sort=False
        )[0]
    else:
        ordered["__feature_portability_group__"] = 0
    return ordered, ordered["__feature_portability_position__"].to_numpy(np.int64)


def _minimum_periods(window: int, minimum_periods: int | None) -> int:
    if int(window) < 1:
        raise FeaturePortabilityError("rolling windows must be positive")
    if minimum_periods is None:
        value = max(3, int(window) // 4)
    else:
        value = int(minimum_periods)
    if not 1 <= value <= int(window):
        raise FeaturePortabilityError("minimum_periods must be in [1, window]")
    return value


def _restore_rolling(values: pd.Series, ordered: pd.DataFrame) -> pd.Series:
    """Drop the internal group index produced by groupby.rolling."""
    result = values.reset_index(level=0, drop=True)
    return result.reindex(ordered.index)


def _thin_rolling_order(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    group_columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Sort only key columns, never a full wide source panel.

    The historical implementation sorted ``frame.copy()``.  On a million-row
    panel this could duplicate every raw/model column before the transform
    matrix was even allocated.  This version retains only timestamp, entity
    keys and a positional permutation; all value work is subsequently one
    source feature at a time.
    """
    _validate_columns(frame, [timestamp_column, *group_columns], kind="rolling-key")
    timestamp = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
    if timestamp.isna().any():
        raise FeaturePortabilityError("rolling transforms require finite UTC decision timestamps")
    if group_columns and frame.loc[:, list(group_columns)].isna().any().any():
        raise FeaturePortabilityError("rolling transforms require non-missing entity keys")
    keys = pd.DataFrame({
        "__feature_portability_ts__": timestamp.to_numpy(),
        "__feature_portability_position__": np.arange(len(frame), dtype=np.int64),
    })
    for column in group_columns:
        # A shallow Series reference is sufficient until pandas needs the
        # compact sort-key table.  It never carries source feature columns.
        keys[column] = frame[column].to_numpy(copy=False)
    keys = keys.sort_values(
        [*group_columns, "__feature_portability_ts__", "__feature_portability_position__"],
        kind="stable",
    ).reset_index(drop=True)
    if group_columns:
        group = pd.factorize(
            pd.MultiIndex.from_frame(keys.loc[:, list(group_columns)]), sort=False
        )[0].astype(np.int32, copy=False)
    else:
        group = np.zeros(len(keys), dtype=np.int32)
    return keys["__feature_portability_position__"].to_numpy(np.int64, copy=False), group


def _ordered_rolling_array(values: pd.Series, order: np.ndarray) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64, copy=True)
    numeric[~np.isfinite(numeric)] = np.nan
    return numeric[order]


def _unpack_rolling(values: pd.Series) -> np.ndarray:
    """Recover the row-order vector from a one-level groupby.rolling result."""
    return values.reset_index(level=0, drop=True).to_numpy(dtype=np.float64, copy=False)


def causal_rolling_portability_transform_batches(
    frame: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    timestamp_column: str,
    group_columns: Sequence[str] = (),
    rank_windows: Sequence[int] = (96,),
    robust_z_windows: Sequence[int] = (96,),
    change_periods: Sequence[int] = (1,),
    include_relative_change: bool = True,
    minimum_periods: int | None = None,
    feature_batch_size: int = 1,
    max_batch_working_bytes: int = 512 * 1024 * 1024,
) -> Iterator[pd.DataFrame]:
    """Yield strictly-causal transform batches without copying a wide frame.

    Each yielded frame is in the *caller* row order and contains at most
    ``feature_batch_size`` source features' derived columns.  The generator is
    designed for callers which attach/write one batch at a time.  It keeps one
    float64 source vector and its rolling intermediates in memory, rather than
    sorting an entire side panel or building all generated columns at once.
    """
    names = tuple(dict.fromkeys(map(str, feature_names)))
    if not names:
        raise FeaturePortabilityError("at least one feature is required for transforms")
    _validate_columns(frame, names, kind="feature")
    if int(feature_batch_size) < 1:
        raise FeaturePortabilityError("feature_batch_size must be positive")
    if int(max_batch_working_bytes) < 1:
        raise FeaturePortabilityError("max_batch_working_bytes must be positive")
    rank_windows = tuple(dict.fromkeys(int(item) for item in rank_windows))
    robust_z_windows = tuple(dict.fromkeys(int(item) for item in robust_z_windows))
    change_periods = tuple(dict.fromkeys(int(item) for item in change_periods))
    if any(item < 1 for item in (*rank_windows, *robust_z_windows, *change_periods)):
        raise FeaturePortabilityError("rolling windows and change periods must be positive")
    if len(frame) == 0:
        return
    estimate = estimate_causal_rolling_transform_memory(
        rows=len(frame), source_features=len(names), rank_windows=rank_windows,
        robust_z_windows=robust_z_windows, change_periods=change_periods,
        include_relative_change=bool(include_relative_change),
        feature_batch_size=int(feature_batch_size),
    )
    if estimate.peak_batch_working_bytes > int(max_batch_working_bytes):
        raise FeaturePortabilityError(
            "causal transform batch exceeds memory budget "
            f"({estimate.peak_batch_working_bytes:,} > {int(max_batch_working_bytes):,} bytes); "
            "lower feature_batch_size or use fewer transform families"
        )
    order, group = _thin_rolling_order(
        frame, timestamp_column=timestamp_column, group_columns=group_columns
    )
    rows = len(frame)
    for start in range(0, len(names), int(feature_batch_size)):
        batch_names = names[start:start + int(feature_batch_size)]
        data: dict[str, np.ndarray] = {}
        for name in batch_names:
            ordered_values = _ordered_rolling_array(frame[name], order)
            values = pd.Series(ordered_values, copy=False)
            grouped = values.groupby(group, sort=False, observed=True, dropna=False)
            for window in rank_windows:
                minp = _minimum_periods(window, minimum_periods)
                ranked = _unpack_rolling(
                    grouped.rolling(window=window, min_periods=minp).rank(pct=True, method="average")
                )
                restored = np.full(rows, np.nan, dtype=np.float32)
                restored[order] = ranked
                data[f"{name}__causal_rank_w{window}"] = restored
            for window in robust_z_windows:
                minp = _minimum_periods(window, minimum_periods)
                rolling = grouped.rolling(window=window, min_periods=minp)
                median = _unpack_rolling(rolling.median())
                q25 = _unpack_rolling(rolling.quantile(0.25))
                q75 = _unpack_rolling(rolling.quantile(0.75))
                scale = (q75 - q25) / 1.349
                z = (ordered_values - median) / np.where(np.abs(scale) > EPS, scale, np.nan)
                restored = np.full(rows, np.nan, dtype=np.float32)
                restored[order] = z
                data[f"{name}__causal_robust_z_w{window}"] = restored
            for period in change_periods:
                prior = grouped.shift(periods=period).to_numpy(dtype=np.float64, copy=False)
                delta = ordered_values - prior
                restored_delta = np.full(rows, np.nan, dtype=np.float32)
                restored_delta[order] = delta
                data[f"{name}__causal_delta_p{period}"] = restored_delta
                if include_relative_change:
                    relative = delta / np.where(np.abs(prior) > EPS, prior, np.nan)
                    restored_relative = np.full(rows, np.nan, dtype=np.float32)
                    restored_relative[order] = relative
                    data[f"{name}__causal_relative_change_p{period}"] = restored_relative
        yield pd.DataFrame(data, index=frame.index, copy=False)


def causal_rolling_portability_transforms(
    frame: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    timestamp_column: str,
    group_columns: Sequence[str] = (),
    rank_windows: Sequence[int] = (96,),
    robust_z_windows: Sequence[int] = (96,),
    change_periods: Sequence[int] = (1,),
    include_relative_change: bool = True,
    minimum_periods: int | None = None,
) -> pd.DataFrame:
    """Materialize all rolling outputs through the bounded batch engine.

    Keeping every output column is intrinsically wide; for million-row panels
    prefer :func:`causal_rolling_portability_transform_batches` and attach or
    persist each batch before requesting the next.  This compatibility helper
    nevertheless avoids the former full-frame source copy and full-frame sort.
    """
    result = pd.DataFrame(index=frame.index)
    for batch in causal_rolling_portability_transform_batches(
        frame, feature_names=feature_names, timestamp_column=timestamp_column,
        group_columns=group_columns, rank_windows=rank_windows,
        robust_z_windows=robust_z_windows, change_periods=change_periods,
        include_relative_change=bool(include_relative_change),
        minimum_periods=minimum_periods,
    ):
        for name in batch.columns:
            # Assign one source batch at a time: this avoids ``concat`` making
            # an additional full output copy while retaining the old API.
            result[name] = batch[name].to_numpy(copy=False)
    return result


def _finite_numeric(values: pd.Series) -> pd.Series:
    result = pd.to_numeric(values, errors="coerce").astype(float)
    return result.mask(~np.isfinite(result))


def _era_order(values: pd.Series) -> list[object]:
    if isinstance(values.dtype, pd.CategoricalDtype) and values.dtype.ordered:
        available = set(values.dropna().tolist())
        return [item for item in values.dtype.categories.tolist() if item in available]
    unique = values.drop_duplicates().tolist()
    try:
        return sorted(unique)
    except TypeError:
        return unique


def _safe_correlation(
    count: pd.Series,
    sum_x: pd.Series,
    sum_y: pd.Series,
    sum_x2: pd.Series,
    sum_y2: pd.Series,
    sum_xy: pd.Series,
) -> pd.Series:
    n = count.astype(float)
    cov = sum_xy - sum_x * sum_y / n
    var_x = sum_x2 - sum_x * sum_x / n
    var_y = sum_y2 - sum_y * sum_y / n
    denom = np.sqrt(np.maximum(var_x, 0.0) * np.maximum(var_y, 0.0))
    return cov.div(denom.where((denom > EPS) & (n >= 3)))


def _psi(reference: np.ndarray, current: np.ndarray, *, bins: int = 10) -> float:
    """Population-stability index on reference quantile bins, safely bounded."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) < 2 or len(current) < 1:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        return 0.0 if np.allclose(reference, current[0], equal_nan=False) else float("inf")
    edges[0] = -np.inf
    edges[-1] = np.inf
    p = np.histogram(reference, bins=edges)[0].astype(float) / len(reference)
    q = np.histogram(current, bins=edges)[0].astype(float) / len(current)
    p = np.clip(p, EPS, None)
    q = np.clip(q, EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


def _diagnostic_keys(
    frame: pd.DataFrame,
    *,
    era_column: str,
    strata_columns: Sequence[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    _validate_columns(frame, [era_column, *strata_columns], kind="diagnostic-key")
    work = frame.copy(deep=False)
    if work[era_column].isna().any():
        raise FeaturePortabilityError("era labels must be non-missing")
    if strata_columns and work.loc[:, list(strata_columns)].isna().any().any():
        raise FeaturePortabilityError("diagnostic strata must be non-missing")
    scope = "__feature_portability_scope__"
    work = work.assign(**{scope: "all"})
    return work, [scope, *strata_columns], [scope, *strata_columns, era_column]


def _per_era_effects(
    work: pd.DataFrame,
    *,
    keys: Sequence[str],
    value_column: str,
    target_column: str,
) -> pd.DataFrame:
    """Vectorized per-group Pearson/Spearman and 80:20 target separation."""
    local = work.loc[:, [*keys, value_column, target_column]].copy()
    valid = local[value_column].notna() & local[target_column].notna()
    local["__valid__"] = valid.astype(np.int64)
    local["__x__"] = local[value_column].where(valid, 0.0)
    local["__y__"] = local[target_column].where(valid, 0.0)
    local["__x2__"] = local["__x__"] * local["__x__"]
    local["__y2__"] = local["__y__"] * local["__y__"]
    local["__xy__"] = local["__x__"] * local["__y__"]
    grouped = local.groupby(list(keys), observed=True, dropna=False, sort=False)
    totals = grouped[["__valid__", "__x__", "__y__", "__x2__", "__y2__", "__xy__"]].sum()
    pearson = _safe_correlation(
        totals["__valid__"], totals["__x__"], totals["__y__"],
        totals["__x2__"], totals["__y2__"], totals["__xy__"],
    )

    # Ranking is within era/stratum, so Spearman remains an effect diagnostic
    # even when raw feature units differ across assets or eras.
    local["__rank_x__"] = grouped[value_column].rank(method="average", pct=True)
    local["__rank_y__"] = grouped[target_column].rank(method="average", pct=True)
    valid_rank = local["__rank_x__"].notna() & local["__rank_y__"].notna()
    local["__rx__"] = local["__rank_x__"].where(valid_rank, 0.0)
    local["__ry__"] = local["__rank_y__"].where(valid_rank, 0.0)
    local["__rx2__"] = local["__rx__"] * local["__rx__"]
    local["__ry2__"] = local["__ry__"] * local["__ry__"]
    local["__rxry__"] = local["__rx__"] * local["__ry__"]
    local["__rank_valid__"] = valid_rank.astype(np.int64)
    ranked = local.groupby(list(keys), observed=True, dropna=False, sort=False)[
        ["__rank_valid__", "__rx__", "__ry__", "__rx2__", "__ry2__", "__rxry__"]
    ].sum()
    spearman = _safe_correlation(
        ranked["__rank_valid__"], ranked["__rx__"], ranked["__ry__"],
        ranked["__rx2__"], ranked["__ry2__"], ranked["__rxry__"],
    )

    q20 = grouped[value_column].quantile(0.20).rename("__q20__")
    q80 = grouped[value_column].quantile(0.80).rename("__q80__")
    cutoffs = pd.concat([q20, q80], axis=1).reset_index()
    local = local.merge(cutoffs, on=list(keys), how="left", validate="many_to_one")
    local["__bottom_y__"] = local[target_column].where(
        valid & local[value_column].le(local["__q20__"])
    )
    local["__top_y__"] = local[target_column].where(
        valid & local[value_column].ge(local["__q80__"])
    )
    separation = local.groupby(list(keys), observed=True, dropna=False, sort=False).agg(
        effect_bottom_mean=("__bottom_y__", "mean"),
        effect_top_mean=("__top_y__", "mean"),
    )
    result = pd.DataFrame({
        "effect_support": totals["__valid__"].astype(np.int64),
        "effect_pearson": pearson,
        "effect_spearman": spearman,
    }).join(separation)
    result["effect_top_bottom_delta"] = (
        result["effect_top_mean"] - result["effect_bottom_mean"]
    )
    return result.reset_index()


def feature_portability_diagnostics(
    frame: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    era_column: str,
    target_column: str | None = None,
    strata_columns: Sequence[str] = (),
    reference_era: object | None = None,
) -> pd.DataFrame:
    """Measure per-era coverage, support, drift, and optional target effects.

    ``reference_era`` is compared inside each stratum.  If omitted, the first
    ordered era is used.  PSI is computed from that reference era's deciles;
    robust median shift is expressed in reference-IQR standard-deviation
    units.  This function never estimates a reference from later rows when an
    explicit earlier era is supplied.
    """
    names = tuple(dict.fromkeys(map(str, feature_names)))
    if not names:
        raise FeaturePortabilityError("at least one feature is required for diagnostics")
    _validate_columns(frame, names, kind="feature")
    if target_column is not None:
        _validate_columns(frame, [target_column], kind="target")
    work, scope_keys, keys = _diagnostic_keys(
        frame, era_column=era_column, strata_columns=strata_columns
    )
    eras = _era_order(work[era_column])
    if not eras:
        raise FeaturePortabilityError("diagnostics require at least one era")
    ref_era = eras[0] if reference_era is None else reference_era
    if ref_era not in set(eras):
        raise FeaturePortabilityError("reference_era is absent from the frame")
    target = _finite_numeric(work[target_column]) if target_column else None
    all_rows: list[pd.DataFrame] = []

    for name in names:
        value = _finite_numeric(work[name])
        local = work.loc[:, [*keys]].copy()
        local["__value__"] = value
        local["__finite__"] = value.notna().astype(np.int64)
        local["__nonzero__"] = (value.abs() > EPS).astype(np.int64)
        grouped = local.groupby(keys, observed=True, dropna=False, sort=False)
        stats = grouped.agg(
            rows=("__finite__", "size"),
            finite_rows=("__finite__", "sum"),
            nonzero_rows=("__nonzero__", "sum"),
            median=("__value__", "median"),
            mean=("__value__", "mean"),
            std=("__value__", "std"),
            minimum=("__value__", "min"),
            maximum=("__value__", "max"),
            unique_values=("__value__", "nunique"),
        )
        quantiles = grouped["__value__"].quantile([0.05, 0.25, 0.75, 0.95]).unstack()
        quantiles = quantiles.rename(columns={0.05: "q05", 0.25: "q25", 0.75: "q75", 0.95: "q95"})
        stats = stats.join(quantiles).reset_index()
        stats["coverage"] = stats["finite_rows"].div(stats["rows"])
        stats["nonzero_coverage"] = stats["nonzero_rows"].div(stats["rows"])
        stats["iqr"] = stats["q75"] - stats["q25"]
        stats["feature"] = name

        if target_column is not None:
            effect_source = work.loc[:, [*keys]].copy()
            effect_source["__value__"] = value
            effect_source["__target__"] = target
            effects = _per_era_effects(
                effect_source, keys=keys, value_column="__value__", target_column="__target__"
            )
            stats = stats.merge(effects, on=keys, how="left", validate="one_to_one")
        else:
            stats["effect_support"] = 0
            for column in (
                "effect_pearson", "effect_spearman", "effect_bottom_mean",
                "effect_top_mean", "effect_top_bottom_delta",
            ):
                stats[column] = np.nan

        reference = stats.loc[stats[era_column].eq(ref_era), [
            *scope_keys, "coverage", "median", "mean", "std", "iqr",
            "effect_pearson", "effect_spearman", "effect_top_bottom_delta",
        ]].copy()
        rename = {column: f"reference_{column}" for column in reference.columns if column not in scope_keys}
        reference = reference.rename(columns=rename)
        stats = stats.merge(reference, on=scope_keys, how="left", validate="many_to_one")
        if stats[["reference_coverage", "reference_median"]].isna().any(axis=None):
            raise FeaturePortabilityError("each stratum must contain the declared reference era")
        robust_scale = stats["reference_iqr"] / 1.349
        stats["coverage_delta"] = stats["coverage"] - stats["reference_coverage"]
        stats["robust_median_shift"] = (stats["median"] - stats["reference_median"]).div(
            robust_scale.where(robust_scale.abs() > EPS)
        )
        stats.loc[
            stats["median"].eq(stats["reference_median"]) & robust_scale.abs().le(EPS),
            "robust_median_shift",
        ] = 0.0
        stats["mean_shift_std"] = (stats["mean"] - stats["reference_mean"]).div(
            stats["reference_std"].where(stats["reference_std"].abs() > EPS)
        )
        stats.loc[
            stats["mean"].eq(stats["reference_mean"]) & stats["reference_std"].abs().le(EPS),
            "mean_shift_std",
        ] = 0.0
        stats["effect_spearman_delta"] = stats["effect_spearman"] - stats["reference_effect_spearman"]
        stats["effect_top_bottom_delta_change"] = (
            stats["effect_top_bottom_delta"] - stats["reference_effect_top_bottom_delta"]
        )
        stats["reference_era"] = ref_era
        stats["psi"] = np.nan
        # PSI needs the two empirical distributions.  This is intentionally a
        # small feature × stratum × era loop over NumPy arrays, not a row-wise
        # apply, so wide candidate panels remain memory bounded.
        for scope, scoped_stats in stats.groupby(scope_keys, observed=True, dropna=False, sort=False):
            scope_values = scope if isinstance(scope, tuple) else (scope,)
            mask = np.ones(len(work), dtype=bool)
            for column, value_key in zip(scope_keys, scope_values, strict=False):
                mask &= work[column].eq(value_key).to_numpy(bool)
            reference_values = value.loc[mask & work[era_column].eq(ref_era).to_numpy(bool)].to_numpy(float)
            for row_index, row in scoped_stats.iterrows():
                current_mask = mask & work[era_column].eq(row[era_column]).to_numpy(bool)
                stats.loc[row_index, "psi"] = _psi(reference_values, value.loc[current_mask].to_numpy(float))
        all_rows.append(stats)

    result = pd.concat(all_rows, ignore_index=True)
    result = result.sort_values(["feature", *scope_keys, era_column], kind="stable").reset_index(drop=True)
    # The synthetic scope makes vectorized reference joins uniform; it is an
    # implementation detail and never part of a published diagnostic schema.
    result = result.drop(columns="__feature_portability_scope__")
    return result


def assign_feature_dispositions(
    diagnostics: pd.DataFrame,
    *,
    roles: pd.DataFrame | Mapping[str, str] | None = None,
    policy: PortabilityPolicy = PortabilityPolicy(),
) -> pd.DataFrame:
    """Assign a conservative, auditable feature disposition from diagnostics.

    The return is one row per feature.  It does not select a model feature set;
    it separates safe portable candidates from fields requiring a causal
    transform, review, or a hard exclusion before later side-local selection.
    """
    required = {
        "feature", "coverage", "finite_rows", "unique_values",
        "robust_median_shift", "psi", "effect_support", "effect_spearman",
    }
    missing = sorted(required.difference(diagnostics.columns))
    if missing:
        raise FeaturePortabilityError(f"diagnostics lack required columns: {missing}")
    if diagnostics.empty:
        return pd.DataFrame(columns=["feature", "role", "disposition", "disposition_reason"])
    if roles is None:
        role_table = classify_feature_roles(diagnostics["feature"].drop_duplicates().tolist())
    elif isinstance(roles, pd.DataFrame):
        if not {"feature", "role"}.issubset(roles.columns):
            raise FeaturePortabilityError("role table must contain feature and role")
        role_table = roles.loc[:, [column for column in ("feature", "role", "portability_role", "role_reason") if column in roles]].copy()
        if "portability_role" not in role_table:
            # Backward-compatible supplied role tables use the old safety role
            # as ``role``.  Keep their declared label for auditability.
            role_table["portability_role"] = role_table["role"]
        if "role_reason" not in role_table:
            role_table["role_reason"] = "supplied_role_table"
    else:
        role_table = pd.DataFrame(
            [{"feature": str(name), "role": FeatureSemanticRole.MODEL_OUTPUT,
              "portability_role": str(role), "role_reason": "supplied_role_mapping"}
             for name, role in roles.items()]
        )
    if role_table["feature"].duplicated().any():
        raise FeaturePortabilityError("role table must have one row per feature")

    rows: list[dict[str, object]] = []
    for feature, local in diagnostics.groupby("feature", sort=True, observed=True):
        if feature not in set(role_table["feature"]):
            raise FeaturePortabilityError(f"diagnostics feature {feature!r} lacks a role")
        role_row = role_table.loc[role_table["feature"].eq(feature)].iloc[0]
        finite_effect = local.loc[
            local["effect_support"].ge(policy.min_effect_support), "effect_spearman"
        ].dropna().to_numpy(float)
        signs = finite_effect[np.abs(finite_effect) >= policy.min_effect_magnitude_for_sign]
        sign_reversal = bool((signs > 0).any() and (signs < 0).any())
        effect_range = float(np.ptp(finite_effect)) if len(finite_effect) >= 2 else float("nan")
        min_coverage = float(local["coverage"].min())
        min_support = int(local["finite_rows"].min())
        min_unique = int(local["unique_values"].min())
        max_shift = float(local["robust_median_shift"].abs().max(skipna=True))
        max_psi = float(local["psi"].max(skipna=True))
        support_ok = (
            min_coverage >= policy.min_coverage
            and min_support >= policy.min_finite_support
            and min_unique >= policy.min_unique_values
        )
        drift_ok = (
            (not np.isfinite(max_shift) or max_shift <= policy.max_abs_robust_median_shift)
            and (not np.isfinite(max_psi) or max_psi <= policy.max_psi)
        )
        effect_ok = (
            len(finite_effect) == 0
            or (not sign_reversal and (not np.isfinite(effect_range) or effect_range <= policy.max_effect_range))
        )
        semantic_role = str(role_row["role"])
        portability_role = str(role_row["portability_role"])
        if portability_role == FeatureRole.OUTCOME_DERIVED:
            disposition, reason = "EXCLUDE_OUTCOME_DERIVED", "outcome/path semantics cannot be inference inputs"
        elif portability_role in {FeatureRole.IDENTITY, FeatureRole.FOLD_LOCAL_STATE, FeatureRole.LATENT_REGIME_OUTPUT, FeatureRole.CONTROL}:
            disposition, reason = "DIAGNOSTIC_ONLY", "identity, control, or fold-local state is not a portable model feature"
        elif portability_role == FeatureRole.CAUSAL_TRANSFORM_REQUIRED:
            if support_ok:
                disposition, reason = "TRANSFORM_CAUSALLY", "raw scale-dependent field has adequate support"
            else:
                disposition, reason = "REVIEW_INSUFFICIENT_SUPPORT", "raw field needs both a causal transform and better support"
        elif portability_role == FeatureRole.UNKNOWN:
            disposition, reason = "REVIEW_UNKNOWN_SEMANTICS", "no explicit portable-unit or source semantics"
        elif not support_ok:
            disposition, reason = "REVIEW_INSUFFICIENT_SUPPORT", "coverage, finite support, or uniqueness fails policy"
        elif not drift_ok:
            disposition, reason = "REVIEW_DISTRIBUTION_DRIFT", "reference-era distribution drift exceeds policy"
        elif not effect_ok:
            disposition, reason = "REVIEW_EFFECT_INSTABILITY", "target association changes materially across eras"
        else:
            disposition, reason = "KEEP_PORTABLE", "portable role with sufficient support and stable diagnostics"
        rows.append({
            "feature": feature,
            "role": semantic_role,
            "portability_role": portability_role,
            "role_reason": str(role_row.get("role_reason", "")),
            "era_count": int(len(local)),
            "min_coverage": min_coverage,
            "min_finite_support": min_support,
            "min_unique_values": min_unique,
            "max_abs_robust_median_shift": max_shift,
            "max_psi": max_psi,
            "effect_era_count": int(len(finite_effect)),
            "effect_spearman_range": effect_range,
            "effect_sign_reversal": sign_reversal,
            "disposition": disposition,
            "disposition_reason": reason,
        })
    return pd.DataFrame(rows).sort_values("feature", kind="stable").reset_index(drop=True)


__all__ = [
    "EPS",
    "CausalTransformMemoryEstimate",
    "FeaturePortabilityError",
    "FeatureRole",
    "FeatureSemanticRole",
    "PortabilityPolicy",
    "assign_feature_dispositions",
    "causal_rolling_portability_transforms",
    "causal_rolling_portability_transform_batches",
    "classify_feature_role",
    "classify_feature_roles",
    "estimate_causal_rolling_transform_memory",
    "feature_portability_diagnostics",
]
