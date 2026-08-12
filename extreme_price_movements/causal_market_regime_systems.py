"""Frozen, causal soft market-regime systems for 6--12 hour decisions.

This module deliberately contains neither targets nor outcome-conditioned
selection.  Each system is fit on a supplied historical observable panel and
then transformed sequentially with the frozen imputer/scaler/GMM parameters.
The small recursive filter below is *forward only*: unlike common
minimum-duration smoothers it never revises an earlier state after observing a
later bar.

The public ensemble exposes four complementary geometries:

* ``trend_volatility`` -- price path, trend, volatility and compression;
* ``breadth_dependence`` -- market breadth, dispersion and correlation shape;
* ``leverage_flow`` -- funding, OI, basis and liquidation-flow context;
* ``liquidity`` -- spread, depth, turnover and price-impact context.

All generated numeric fields are float32.  Expensive parameter assessment is
performed on a deterministic, bounded proxy sample; the final five-component
diagonal GMM is still fit only on historical rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler


CAUSAL_MARKET_REGIME_SCHEMA = "causal_market_regime_systems_v1"
PHASE_NAMES: tuple[str, ...] = ("stable", "onset", "active", "settling")

# The primary five-state coordinate system is not permitted to inherit an
# arbitrary GMM-component ordering.  These names describe *market geometry*,
# not a signed directional view.  The latter is emitted separately below.
PRIMARY_SEMANTIC_STATE_NAMES: tuple[str, ...] = (
    "s0_quiet_choppy",
    "s1_coherent_directional_expansion",
    "s2_fragmented",
    "s3_systemic_stress_deleveraging",
    "s4_recovery_settling",
)
_DIRECTION_TOKENS: tuple[str, ...] = (
    "return", "ret_", "momentum", "slope", "trend", "direction",
)

# A small, deliberately named set of *continuous observables*.  This contract
# is distinct from the soft GMM systems below: it never consumes a state id,
# posterior, or any other cluster-membership output.  The source names are
# present in the multiview hourly store and span trend quality, volatility,
# breadth, dependence, leverage, funding, and liquidity.  Keeping the public
# names independent from the raw store names makes the downstream meta feature
# contract stable when a source is migrated in the store.
CONTINUOUS_CONTEXT_SOURCE_CONTRACT: Mapping[str, str] = {
    "trend_quality": "mv__breakout_efficiency_4h__delta_6h",
    "volatility": "mv__breadth_dispersion__vol_of_vol_6h",
    "breadth": "mv__breadth_dispersion__delta_6h",
    "dependence": "mv__correlation_heterogeneity_dispersion__delta_6h",
    "leverage": "mv__mkt_regime_change__oi_contraction__cumulative_change_2d__delta_6h",
    "funding": "mv__mkt_regime_change__funding__cumulative_change_2d__delta_6h",
    "spread": "mv__liquidity_xs__ob_spread_bps_z_24h__mean__robust_z_6h",
    "turnover": "mv__liquidity_market__mkt_quote_volume_z_24h__robust_z_6h",
    "dispersion": "mv__btc_decoupling_dispersion__delta_6h",
}
CONTINUOUS_CONTEXT_OPERATORS: tuple[str, ...] = (
    "rank_90d", "z_90d", "rank_180d", "z_180d",
    "change_4h", "change_24h", "distance_recent_median_30d",
)
CONTINUOUS_CONTEXT_FEATURE_KEYS: tuple[str, ...] = tuple(
    f"continuous_regime__{source}__{operator}"
    for source in CONTINUOUS_CONTEXT_SOURCE_CONTRACT
    for operator in CONTINUOUS_CONTEXT_OPERATORS
)

# Relationship breaks intentionally use the same raw, continuous source
# contract as the relative context fields above.  They express whether a
# decision-time observation is unusual *conditional on another observable*,
# rather than introducing a second latent state or a cluster-derived feature.
# ``trend_quality`` is the compact price/trend proxy available in the panel.
RELATIONSHIP_BREAK_SOURCE_PAIRS: Mapping[str, tuple[str, str]] = {
    "trend_breadth": ("trend_quality", "breadth"),
    "trend_turnover": ("trend_quality", "turnover"),
    "volatility_dependence": ("volatility", "dependence"),
    "price_leverage": ("trend_quality", "leverage"),
    # These two complete the small H5 geometry contract.  ``spread`` is the
    # observable, decision-time liquidity impairment proxy; ``dispersion``
    # is the market-isolation proxy.  They remain ordinary strict-prequential
    # residuals and never consume a latent-state posterior.
    "volatility_liquidity": ("volatility", "spread"),
    "isolation_dependence": ("dispersion", "dependence"),
}
RELATIONSHIP_BREAK_OPERATORS: tuple[str, ...] = ("residual_signed", "residual_abs")
RELATIONSHIP_BREAK_FEATURE_KEYS: tuple[str, ...] = tuple(
    f"continuous_regime__relationship_break__{relationship}__{operator}_{window}d"
    for relationship in RELATIONSHIP_BREAK_SOURCE_PAIRS
    for window in (30, 90)
    for operator in RELATIONSHIP_BREAK_OPERATORS
)

# Deliberately conservative.  A name-based gate is not the only lineage
# defence in the calling pipeline, but it prevents accidental target plumbing
# into this isolated unsupervised module.
FORBIDDEN_INPUT_TOKENS: tuple[str, ...] = (
    "target", "label", "outcome", "future", "post_entry", "postentry",
    "realized", "realised", "pnl", "net_ev", "gross_ev", "mfe", "mae",
    "barrier", "timeout", "exit", "time_to", "policy_return",
)


@dataclass(frozen=True)
class RegimeGeometrySpec:
    """One intentionally distinct observable market-geometry view."""

    name: str
    include_tokens: tuple[str, ...]
    exclude_tokens: tuple[str, ...] = ()
    max_features: int = 12
    min_features: int = 2
    # ``5`` is reserved for the primary market state.  Geometry systems may
    # choose a compact K from the predeclared diagnostic candidates.
    fixed_state_count: int | None = None


DEFAULT_GEOMETRY_SPECS: tuple[RegimeGeometrySpec, ...] = (
    RegimeGeometrySpec(
        "primary",
        ("trend", "ema", "momentum", "return", "ret", "vol", "atr", "range", "compression", "chop", "efficiency", "breadth", "dispersion", "corr", "funding", "oi", "spread", "depth", "volume"),
        # Direction is emitted separately from frozen signed sources.  Do not
        # let obvious signed return/momentum fields determine the geometry
        # coordinate that will later be named coherent expansion.
        ("return", "ret_", "momentum", "slope", "direction"),
        max_features=16,
        fixed_state_count=5,
    ),
    RegimeGeometrySpec(
        "trend_volatility",
        ("trend", "ema", "momentum", "return", "ret", "vol", "atr", "range", "compression", "chop", "efficiency", "coherence", "entropy"),
        # A ``range`` field whose name also encodes a deleveraging/flow event
        # belongs to leverage_flow, not this price/volatility view.
        ("fund", "oi", "open_interest", "spread", "depth", "liquid", "breadth", "corr", "dispersion", "volume", "turnover", "deleverag", "liquidation", "crowd", "basis", "premium", "carry", "flow"),
    ),
    RegimeGeometrySpec(
        "breadth_dependence",
        ("breadth", "dispersion", "correlation", "corr", "dependence", "eigen", "effective_rank", "synchron", "cross_section"),
        ("fund", "oi", "open_interest", "spread", "depth", "liquid", "deleverag", "liquidation", "crowd", "basis", "premium", "carry", "flow"),
    ),
    RegimeGeometrySpec(
        "leverage_flow",
        ("fund", "funding", "open_interest", "oi_", "oi", "basis", "premium", "carry", "liquidation", "deleverag", "crowding", "flow"),
        ("spread", "depth", "orderbook"),
    ),
    RegimeGeometrySpec(
        "liquidity",
        ("liquid", "spread", "depth", "amihud", "amivest", "volume", "turnover", "orderbook", "ob_", "impact"),
        ("fund", "oi", "open_interest", "basis", "deleverag", "liquidation", "crowd", "premium", "carry", "flow"),
    ),
)

# The primary five-state representation has its own contract and lifecycle.
# Geometry specialists are intentionally exposed separately so callers do not
# accidentally refit/replace the primary system merely to add one slow market
# lens.  Keep the order fixed: it is part of the output schema and manifest.
LATENT_GEOMETRY_SPECS: tuple[RegimeGeometrySpec, ...] = tuple(
    spec for spec in DEFAULT_GEOMETRY_SPECS if spec.name != "primary"
)
LATENT_GEOMETRY_SYSTEM_NAMES: tuple[str, ...] = tuple(
    spec.name for spec in LATENT_GEOMETRY_SPECS
)


def latent_geometry_output_feature_names(
    *,
    include_memberships: bool = True,
    max_state_count: int = 6,
) -> tuple[str, ...]:
    """Return the stable padded candidate-sidecar geometry schema.

    Geometry K is selected from a bounded label-free proxy per frozen OOF
    fold.  ``state_count`` records local support and posterior coordinates are
    padded to six values in the materializer.  The named invariant fields are
    safe meta candidates; membership coordinates remain separately declared
    diagnostics unless a downstream fold-alignment gate explicitly admits
    them.
    """

    fields: list[str] = []
    for system in LATENT_GEOMETRY_SYSTEM_NAMES:
        prefix = f"geometry_regime__{system}"
        if include_memberships:
            fields.extend(f"{prefix}__state_p_{state}" for state in range(int(max_state_count)))
        fields.extend((
            f"{prefix}__state_count",
            f"{prefix}__assigned_centroid_distance",
            f"{prefix}__within_state_radius_percentile",
            f"{prefix}__state_boundary_margin",
            f"{prefix}__centroid_distance_velocity",
            f"{prefix}__entropy",
            f"{prefix}__top2_margin",
            f"{prefix}__state_age_hours",
            f"{prefix}__state_switch_probability",
            f"{prefix}__ood_distance_percentile",
            f"{prefix}__input_coverage",
            *(f"{prefix}__phase_p_{phase}" for phase in PHASE_NAMES),
        ))
    return tuple(fields)


@dataclass(frozen=True)
class CausalMarketRegimeConfig:
    """Bounded, deterministic configuration for the frozen state systems."""

    timestamp_col: str = "source_utc"
    group_columns: tuple[str, ...] = ()
    diagnostic_k_values: tuple[int, ...] = (3, 4, 5, 6)
    stickiness_values: tuple[float, ...] = (0.0, 0.35, 0.60, 0.80)
    max_train_rows: int = 50_000
    max_proxy_rows: int = 6_000
    max_iter: int = 120
    reg_covar: float = 1e-5
    random_state: int = 20260803
    minimum_feature_coverage: float = 0.80
    minimum_rows_per_state: int = 20
    max_gap_hours: float = 2.0
    transition_horizon_hours: float = 6.0
    robust_scaled_clip: float = 12.0
    # The production-default primary representation remains five states.  The
    # controlled generator funnel may explicitly request K=3/4 or collapse a
    # low-support K=5 component after the label-free fit.
    primary_state_count: int = 5
    primary_merge_low_support_state: bool = False
    minimum_state_soft_occupancy: float = 0.02
    # A low switch rate is not automatically desirable.  This target merely
    # prevents HPO from selecting maximum stickiness for every geometry.
    target_switch_probability: float = 0.08


@dataclass(frozen=True)
class CausalContinuousContextConfig:
    """Causal rolling-reference configuration for continuous market context.

    The 90/180-day rank and z-score fields compare a decision-time value with
    strictly earlier observations only.  Ranks use pandas' compiled rolling
    rank primitive and remove the current row's self-rank algebraically; this
    is exact for the empirical mid-rank while avoiding an O(rows × window)
    Python loop.  The only intentionally expensive rolling quantile is the
    30-day median, which is a bounded proxy for a recent market reference.
    """

    timestamp_col: str = "source_utc"
    group_columns: tuple[str, ...] = ()
    rank_windows_days: tuple[int, int] = (90, 180)
    recent_median_days: int = 30
    change_hours: tuple[int, int] = (4, 24)
    min_reference_rows: int = 72
    z_clip: float = 12.0


@dataclass(frozen=True)
class CausalRelationshipBreakConfig:
    """Strict-prequential, low-capacity relationship-break configuration.

    Each declared pair fits only a trailing, intercept-plus-one-slope linear
    reference using earlier complete observations.  The relationship is a
    deliberately cheap proxy for a time-varying conditional expectation: it
    avoids cluster fitting, targets and an unbounded pairwise history matrix.
    The 30/90-day windows are slow enough to be stable yet responsive for the
    6--12 hour trading horizon.
    """

    timestamp_col: str = "source_utc"
    group_columns: tuple[str, ...] = ()
    windows_days: tuple[int, int] = (30, 90)
    min_reference_rows: int = 72
    variance_epsilon: float = 1e-8
    slope_clip: float = 12.0


def _continuous_context_group_keys(frame: pd.DataFrame, config: CausalContinuousContextConfig) -> pd.Series:
    missing = [column for column in config.group_columns if column not in frame]
    if missing:
        raise KeyError(f"continuous context group columns missing: {missing}")
    if not config.group_columns:
        return pd.Series("__all__", index=frame.index, dtype="string")
    return frame.loc[:, list(config.group_columns)].astype("string").fillna("<null>").agg("\x1f".join, axis=1)


def _relationship_break_group_keys(frame: pd.DataFrame, config: CausalRelationshipBreakConfig) -> pd.Series:
    """Use the identical partitioning contract as continuous context."""

    return _continuous_context_group_keys(
        frame,
        CausalContinuousContextConfig(
            timestamp_col=config.timestamp_col,
            group_columns=config.group_columns,
        ),
    )


def _strict_prequential_context_for_group(
    values: pd.Series,
    timestamps: pd.Series,
    *,
    config: CausalContinuousContextConfig,
) -> dict[str, np.ndarray]:
    """Compute one group with timestamp-indexed, left-closed rolling windows.

    Inputs are already chronological and unique by timestamp.  ``closed='left'``
    is the causality guard for moments and medians.  For rank, the compiled
    inclusive current-row rank is converted to a strict historical mid-rank by
    removing the current observation's half-tie contribution.
    """

    index = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True, errors="raise"))
    series = pd.Series(pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64, copy=False), index=index)
    output: dict[str, np.ndarray] = {}
    epsilon = 1e-8
    for days in config.rank_windows_days:
        window = f"{int(days)}D"
        history = series.rolling(window, closed="left", min_periods=int(config.min_reference_rows))
        mean = history.mean()
        std = history.std(ddof=0)
        z = ((series - mean) / std.where(std.gt(epsilon))).clip(-float(config.z_clip), float(config.z_clip))
        # ``closed='both'`` contains exactly the left-closed historical set
        # plus the current value.  With average tie ranking, subtracting one
        # gives lower-count plus half of prior equal values: the strict
        # decision-time empirical mid-rank without retaining a window matrix.
        inclusive_rank = series.rolling(window, closed="both", min_periods=int(config.min_reference_rows) + 1).rank(method="average")
        count = history.count()
        rank = ((inclusive_rank - 1.0) / count.where(count.gt(0))).clip(0.0, 1.0)
        output[f"rank_{int(days)}d"] = rank.to_numpy(dtype=np.float32, na_value=np.nan)
        output[f"z_{int(days)}d"] = z.to_numpy(dtype=np.float32, na_value=np.nan)

    recent_window = f"{int(config.recent_median_days)}D"
    recent_median = series.rolling(
        recent_window,
        closed="left",
        min_periods=int(config.min_reference_rows),
    ).median()
    output[f"distance_recent_median_{int(config.recent_median_days)}d"] = (
        series - recent_median
    ).to_numpy(dtype=np.float32, na_value=np.nan)
    for hours in config.change_hours:
        # Exact timestamp reindexing is intentional: a gap must remain absent
        # rather than silently borrowing a stale market observation.
        prior = series.reindex(index - pd.Timedelta(hours=int(hours))).to_numpy(dtype=np.float64, copy=False)
        output[f"change_{int(hours)}h"] = (series.to_numpy(dtype=np.float64, copy=False) - prior).astype(np.float32)
    return output


def build_causal_continuous_context_features(
    frame: pd.DataFrame,
    source_contract: Mapping[str, str] = CONTINUOUS_CONTEXT_SOURCE_CONTRACT,
    *,
    config: CausalContinuousContextConfig = CausalContinuousContextConfig(),
) -> pd.DataFrame:
    """Build bounded, strict-prequential relative fields for raw observables.

    No membership-derived columns are accepted.  The function processes one
    source at a time and emits float32 arrays, so peak working memory is
    bounded by a few vectors per source rather than a source × 180-day window
    tensor.  Source values are allowed to be missing; resulting references are
    unavailable until the declared prior support exists.
    """

    if config.timestamp_col not in frame:
        raise KeyError(f"continuous context timestamp column missing: {config.timestamp_col}")
    if not source_contract:
        raise ValueError("continuous context source contract must not be empty")
    aliases = list(source_contract)
    source_columns = list(source_contract.values())
    if len(set(aliases)) != len(aliases) or len(set(source_columns)) != len(source_columns):
        raise ValueError("continuous context aliases and source columns must be unique")
    forbidden_membership = [
        name for name in source_columns
        if any(token in name.lower() for token in ("state_p_", "membership", "posterior", "cluster"))
    ]
    if forbidden_membership:
        raise ValueError(f"continuous context cannot use membership outputs: {forbidden_membership[:4]}")
    _require_observable_columns(frame, source_columns)
    timestamps = pd.to_datetime(frame[config.timestamp_col], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("continuous context timestamps must be valid UTC values")
    group_keys = _continuous_context_group_keys(frame, config)
    ordered = pd.DataFrame({"__timestamp__": timestamps, "__group__": group_keys}, index=frame.index)
    ordered["__position__"] = np.arange(len(ordered), dtype=np.int64)
    ordered = ordered.sort_values(["__group__", "__timestamp__", "__position__"], kind="stable")
    if ordered.duplicated(["__group__", "__timestamp__"]).any():
        raise ValueError("continuous context requires unique timestamps within each group")
    expected = continuous_context_feature_names(source_contract, config=config)
    output: dict[str, np.ndarray] = {
        name: np.full(len(frame), np.nan, dtype=np.float32)
        for name in expected
    }
    # This group loop is only across calendar/segment partitions.  All rolling
    # statistics inside it remain pandas/numpy vector operations.
    for alias, source in source_contract.items():
        numeric = pd.to_numeric(frame[source], errors="coerce")
        for _, positions in ordered.groupby("__group__", sort=False)["__position__"]:
            local_positions = positions.to_numpy(dtype=np.int64, copy=False)
            local = _strict_prequential_context_for_group(
                # ``frame`` may legitimately carry a non-unique external
                # index.  Label-based ``.loc`` would then duplicate/reorder
                # rows and write them into a positional output incorrectly.
                # The ordered grouping already stores canonical row positions,
                # so keep the complete calculation positional end-to-end.
                numeric.iloc[local_positions],
                timestamps.iloc[local_positions],
                config=config,
            )
            for operator, values in local.items():
                output[f"continuous_regime__{alias}__{operator}"][local_positions] = values
    result = pd.DataFrame(output, index=frame.index, dtype=np.float32)
    if tuple(result.columns) != expected:
        raise RuntimeError("continuous context output does not match its named contract")
    return result


def _strict_prequential_linear_break_for_group(
    predictor: pd.Series,
    response: pd.Series,
    timestamps: pd.Series,
    *,
    config: CausalRelationshipBreakConfig,
) -> dict[str, np.ndarray]:
    """Return residuals from rolling prior-only one-variable OLS references.

    All sufficient statistics are rolling, left-closed sums.  Consequently a
    row's response (and every future row) cannot affect its own expected value
    or an earlier residual.  Invalid pair observations are absent from the
    regression support rather than coerced to zero.
    """

    index = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True, errors="raise"))
    x = pd.Series(pd.to_numeric(predictor, errors="coerce").to_numpy(dtype=np.float64, copy=False), index=index)
    y = pd.Series(pd.to_numeric(response, errors="coerce").to_numpy(dtype=np.float64, copy=False), index=index)
    valid = x.notna() & y.notna()
    # Filling only after masking makes the sums inexpensive and preserves the
    # valid-count denominator needed for complete-pair OLS.
    x_valid = x.where(valid).fillna(0.0)
    y_valid = y.where(valid).fillna(0.0)
    xy_valid = (x_valid * y_valid).where(valid, 0.0)
    xx_valid = (x_valid * x_valid).where(valid, 0.0)
    count_values = valid.astype(np.float64)
    output: dict[str, np.ndarray] = {}
    for days in config.windows_days:
        history = f"{int(days)}D"
        # ``closed='left'`` is the central no-lookahead invariant.
        n = count_values.rolling(history, closed="left", min_periods=1).sum()
        sum_x = x_valid.rolling(history, closed="left", min_periods=1).sum()
        sum_y = y_valid.rolling(history, closed="left", min_periods=1).sum()
        sum_xx = xx_valid.rolling(history, closed="left", min_periods=1).sum()
        sum_xy = xy_valid.rolling(history, closed="left", min_periods=1).sum()
        usable = n.ge(float(config.min_reference_rows))
        n_safe = n.where(usable)
        centered_xx = sum_xx - (sum_x * sum_x / n_safe)
        centered_xy = sum_xy - (sum_x * sum_y / n_safe)
        slope = (centered_xy / centered_xx.where(centered_xx.abs().gt(float(config.variance_epsilon)))).clip(
            -float(config.slope_clip), float(config.slope_clip),
        )
        intercept = sum_y / n_safe - slope * (sum_x / n_safe)
        signed = (y - (intercept + slope * x)).where(valid & usable & slope.notna())
        signed_values = signed.to_numpy(dtype=np.float32, na_value=np.nan)
        output[f"residual_signed_{int(days)}d"] = signed_values
        output[f"residual_abs_{int(days)}d"] = np.abs(signed_values).astype(np.float32, copy=False)
    return output


def relationship_break_feature_names(
    source_contract: Mapping[str, str] = CONTINUOUS_CONTEXT_SOURCE_CONTRACT,
    *,
    config: CausalRelationshipBreakConfig = CausalRelationshipBreakConfig(),
) -> tuple[str, ...]:
    """Return available relationship-break fields in deterministic order."""

    available = set(source_contract)
    return tuple(
        f"continuous_regime__relationship_break__{relationship}__{operator}_{int(days)}d"
        for relationship, (predictor, response) in RELATIONSHIP_BREAK_SOURCE_PAIRS.items()
        if predictor in available and response in available
        for days in config.windows_days
        for operator in RELATIONSHIP_BREAK_OPERATORS
    )


def build_causal_relationship_break_features(
    frame: pd.DataFrame,
    source_contract: Mapping[str, str] = CONTINUOUS_CONTEXT_SOURCE_CONTRACT,
    *,
    config: CausalRelationshipBreakConfig = CausalRelationshipBreakConfig(),
) -> pd.DataFrame:
    """Materialise cheap prior-only continuous relationship-break features.

    Only complete named pairs present in ``source_contract`` are emitted.  The
    implementation processes one pair and one group at a time; its temporary
    working set is a fixed number of one-dimensional rolling vectors, rather
    than a rows-by-window or pairwise correlation tensor.
    """

    if config.timestamp_col not in frame:
        raise KeyError(f"relationship-break timestamp column missing: {config.timestamp_col}")
    if not source_contract:
        raise ValueError("relationship-break source contract must not be empty")
    aliases = list(source_contract)
    source_columns = list(source_contract.values())
    if len(set(aliases)) != len(aliases) or len(set(source_columns)) != len(source_columns):
        raise ValueError("relationship-break aliases and source columns must be unique")
    forbidden_membership = [
        name for name in source_columns
        if any(token in name.lower() for token in ("state_p_", "membership", "posterior", "cluster"))
    ]
    if forbidden_membership:
        raise ValueError(f"relationship breaks cannot use membership outputs: {forbidden_membership[:4]}")
    _require_observable_columns(frame, source_columns)
    timestamps = pd.to_datetime(frame[config.timestamp_col], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("relationship-break timestamps must be valid UTC values")
    group_keys = _relationship_break_group_keys(frame, config)
    ordered = pd.DataFrame({"__timestamp__": timestamps, "__group__": group_keys}, index=frame.index)
    ordered["__position__"] = np.arange(len(ordered), dtype=np.int64)
    ordered = ordered.sort_values(["__group__", "__timestamp__", "__position__"], kind="stable")
    if ordered.duplicated(["__group__", "__timestamp__"]).any():
        raise ValueError("relationship breaks require unique timestamps within each group")
    expected = relationship_break_feature_names(source_contract, config=config)
    output: dict[str, np.ndarray] = {
        name: np.full(len(frame), np.nan, dtype=np.float32)
        for name in expected
    }
    for relationship, (predictor_alias, response_alias) in RELATIONSHIP_BREAK_SOURCE_PAIRS.items():
        if predictor_alias not in source_contract or response_alias not in source_contract:
            continue
        predictor = pd.to_numeric(frame[source_contract[predictor_alias]], errors="coerce")
        response = pd.to_numeric(frame[source_contract[response_alias]], errors="coerce")
        for _, positions in ordered.groupby("__group__", sort=False)["__position__"]:
            local_positions = positions.to_numpy(dtype=np.int64, copy=False)
            local = _strict_prequential_linear_break_for_group(
                predictor.iloc[local_positions],
                response.iloc[local_positions],
                timestamps.iloc[local_positions],
                config=config,
            )
            for operator, values in local.items():
                output[f"continuous_regime__relationship_break__{relationship}__{operator}"][local_positions] = values
    result = pd.DataFrame(output, index=frame.index, dtype=np.float32)
    if tuple(result.columns) != expected:
        raise RuntimeError("relationship-break output does not match its named contract")
    return result


def continuous_context_feature_names(
    source_contract: Mapping[str, str] = CONTINUOUS_CONTEXT_SOURCE_CONTRACT,
    *,
    config: CausalContinuousContextConfig = CausalContinuousContextConfig(),
) -> tuple[str, ...]:
    """Return the deterministic, meta-only feature names for a source map."""

    return tuple(
        f"continuous_regime__{alias}__{operator}"
        for alias in source_contract
        for operator in (
            *(item for days in config.rank_windows_days for item in (f"rank_{int(days)}d", f"z_{int(days)}d")),
            *(f"change_{int(hours)}h" for hours in config.change_hours),
            f"distance_recent_median_{int(config.recent_median_days)}d",
        )
    )


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")


def _forbidden(columns: Iterable[str]) -> list[str]:
    return [str(c) for c in columns if any(token in str(c).lower() for token in FORBIDDEN_INPUT_TOKENS)]


def _require_observable_columns(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    names = list(dict.fromkeys(str(c) for c in columns))
    missing = [name for name in names if name not in frame]
    if missing:
        raise KeyError(f"regime input columns missing: {missing[:8]}")
    forbidden = _forbidden(names)
    if forbidden:
        raise ValueError(f"regime inputs contain label/outcome fields: {forbidden[:8]}")
    bad = [name for name in names if not pd.api.types.is_numeric_dtype(frame[name])]
    if bad:
        raise TypeError(f"regime inputs must be numeric: {bad[:8]}")
    return names


def _time_group_order(frame: pd.DataFrame, cfg: CausalMarketRegimeConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cfg.timestamp_col not in frame:
        raise KeyError(f"timestamp column missing: {cfg.timestamp_col}")
    missing_groups = [name for name in cfg.group_columns if name not in frame]
    if missing_groups:
        raise KeyError(f"group columns missing: {missing_groups}")
    ts = pd.to_datetime(frame[cfg.timestamp_col], utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError("regime timestamps must be valid UTC values")
    if cfg.group_columns:
        group_values = frame.loc[:, list(cfg.group_columns)].astype(str).fillna("<null>").agg("\x1f".join, axis=1).to_numpy(dtype=object)
    else:
        group_values = np.repeat("__all__", len(frame)).astype(object)
    # Stable lexical ordering makes the recursive pass deterministic even when
    # an input arrives in a different row order.
    order = np.lexsort((np.arange(len(frame)), ts.astype("int64").to_numpy(), group_values.astype(str)))
    return order.astype(np.int64, copy=False), ts.astype("int64").to_numpy(), group_values


def _deterministic_sample_positions(n: int, maximum: int) -> np.ndarray:
    if maximum <= 0 or n <= maximum:
        return np.arange(n, dtype=np.int64)
    # Includes the beginning/end and is deterministic across repeated runs.
    return np.unique(np.linspace(0, n - 1, num=int(maximum), dtype=np.int64))


def _softmax_log(log_values: np.ndarray) -> np.ndarray:
    shifted = log_values - np.max(log_values, axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / np.maximum(values.sum(axis=1, keepdims=True), 1e-12)


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    k = max(int(probabilities.shape[1]), 2)
    return (-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1) / np.log(float(k))).astype(np.float32)


def _fit_diag_gmm(
    values: np.ndarray,
    *,
    components: int,
    config: CausalMarketRegimeConfig,
    seed: int,
    max_iter: int,
) -> GaussianMixture:
    """Fit a bounded diagonal GMM, escalating regularisation deterministically.

    Real market panels can contain long stretches of nearly identical,
    coverage-qualified values.  Escalating covariance regularisation preserves
    the predeclared state count and avoids a brittle singleton-covariance
    failure without looking at evaluation rows or economic labels.
    """

    last_error: ValueError | None = None
    for multiplier in (1.0, 10.0, 100.0, 1_000.0):
        try:
            return GaussianMixture(
                n_components=int(components),
                covariance_type="diag",
                reg_covar=float(config.reg_covar) * multiplier,
                max_iter=int(max_iter),
                n_init=1,
                random_state=int(seed),
            ).fit(values)
        except ValueError as error:
            last_error = error
    assert last_error is not None
    raise last_error


def _state_dynamics(
    probabilities: np.ndarray,
    *,
    timestamps_ns: np.ndarray,
    group_values: np.ndarray,
    stickiness: float,
    max_gap_hours: float,
    initial: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, dict[str, Any]]]:
    """Apply an online sticky filter and derive age/switch probability.

    This loop is necessarily sequential (a state filter is recursive), but it
    works on compact float32 arrays and only stores one prior vector per group.
    It does not smooth labels backwards or inspect a following row.
    """

    raw = np.asarray(probabilities, dtype=np.float32)
    n, k = raw.shape
    out = np.empty_like(raw, dtype=np.float32)
    ages = np.zeros(n, dtype=np.float32)
    switch = np.zeros(n, dtype=np.float32)
    first = np.zeros(n, dtype=bool)
    history: dict[str, dict[str, Any]] = {str(key): dict(value) for key, value in (initial or {}).items()}
    rho = float(np.clip(stickiness, 0.0, 0.99))
    max_gap_ns = max(float(max_gap_hours), 0.0) * 3_600_000_000_000.0
    for pos in range(n):
        key = str(group_values[pos])
        state = history.get(key)
        ts = int(timestamps_ns[pos])
        reset = state is None or (max_gap_ns > 0.0 and ts - int(state["timestamp_ns"]) > max_gap_ns)
        if reset:
            filtered = raw[pos]
            # A new sequence/gap has no observed dwell in this state.  Keeping
            # this at zero makes reset semantics explicit and avoids inventing
            # elapsed time from an unavailable interval.
            ages[pos] = np.float32(0.0)
            first[pos] = True
            previous_label = int(np.argmax(filtered))
        else:
            previous = np.asarray(state["posterior"], dtype=np.float32)
            filtered = ((1.0 - rho) * raw[pos] + rho * previous).astype(np.float32, copy=False)
            filtered /= np.maximum(filtered.sum(), np.float32(1e-12))
            switch[pos] = np.float32(np.clip(1.0 - float(np.dot(previous, filtered)), 0.0, 1.0))
            previous_label = int(state["label"])
            elapsed = max(0.0, (ts - int(state["timestamp_ns"])) / 3_600_000_000_000.0)
            ages[pos] = np.float32(float(state["age_hours"]) + elapsed)
        label = int(np.argmax(filtered))
        if not reset and label != previous_label:
            ages[pos] = np.float32(0.0)
        out[pos] = filtered
        history[key] = {
            "posterior": filtered.astype(np.float32, copy=True),
            "label": label,
            "age_hours": float(ages[pos]),
            "timestamp_ns": ts,
        }
    return out, ages, switch, first, history


def _phase_simplex(
    switch_probability: np.ndarray,
    age_hours: np.ndarray,
    initial: np.ndarray,
    horizon_hours: float,
) -> np.ndarray:
    """Causal action-facing transition morphology as a four-state simplex."""

    switch = np.clip(np.asarray(switch_probability, dtype=np.float32), 0.0, 1.0)
    age = np.maximum(np.asarray(age_hours, dtype=np.float32), 0.0)
    progress = np.clip(age / max(float(horizon_hours), 1e-6), 0.0, 1.0)
    onset = switch * (1.0 - progress)
    active = switch * (0.25 + 0.75 * progress)
    settling = (1.0 - switch) * (1.0 - progress) * (age > 0.0)
    stable = (1.0 - switch) * (0.25 + 0.75 * progress)
    raw = np.column_stack((stable, onset, active, settling)).astype(np.float32)
    raw[np.asarray(initial, dtype=bool)] = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32)
    raw /= np.maximum(raw.sum(axis=1, keepdims=True), np.float32(1e-12))
    return raw.astype(np.float32, copy=False)


def _persistence_metrics(
    posterior: np.ndarray,
    *,
    timestamps_ns: np.ndarray,
    group_values: np.ndarray,
    max_gap_hours: float,
) -> dict[str, float]:
    """Structural, label-free state-quality metrics on a chronological proxy."""

    states = np.argmax(posterior, axis=1)
    max_gap_ns = float(max_gap_hours) * 3_600_000_000_000.0
    runs: list[int] = []
    run_length = 1
    transitions = 0
    adjacent = 0
    for index in range(1, len(states)):
        contiguous = (
            str(group_values[index]) == str(group_values[index - 1])
            and int(timestamps_ns[index]) - int(timestamps_ns[index - 1]) <= max_gap_ns
        )
        if contiguous:
            adjacent += 1
            if states[index] != states[index - 1]:
                transitions += 1
                runs.append(run_length)
                run_length = 1
            else:
                run_length += 1
        else:
            runs.append(run_length)
            run_length = 1
    if len(states):
        runs.append(run_length)
    occupancy = np.bincount(states, minlength=posterior.shape[1]) / max(len(states), 1)
    return {
        "median_dwell_hours": float(np.median(runs)) if runs else 0.0,
        "temporal_switch_rate": float(transitions / adjacent) if adjacent else 0.0,
        "minimum_state_occupancy": float(occupancy.min()) if len(occupancy) else 0.0,
        "mean_max_probability": float(np.max(posterior, axis=1).mean()) if len(posterior) else 0.0,
    }


def _merge_low_support_component(
    probabilities: np.ndarray,
    centroids: np.ndarray,
    *,
    minimum_occupancy: float,
) -> tuple[np.ndarray, int | None, int | None, np.ndarray]:
    """Merge one label-free unsupported state into its nearest centroid.

    This is deliberately a *post-fit* candidate, not a refit chosen on
    returns.  Membership mass is preserved exactly; the state count only
    changes if the GMM's soft occupancy is below the declared support floor.
    """

    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 3:
        return values, None, None, np.empty(0, dtype=np.float32)
    occupancy = values.mean(axis=0).astype(np.float32)
    source = int(np.argmin(occupancy))
    if float(occupancy[source]) >= float(minimum_occupancy):
        return values, None, None, occupancy
    means = np.asarray(centroids, dtype=np.float32)
    distances = np.sum((means - means[source]) ** 2, axis=1)
    distances[source] = np.inf
    target = int(np.argmin(distances))
    merged = values.copy()
    merged[:, target] += merged[:, source]
    merged = np.delete(merged, source, axis=1)
    merged /= np.maximum(merged.sum(axis=1, keepdims=True), np.float32(1e-12))
    return merged.astype(np.float32, copy=False), source, target, occupancy


def _collapse_merged_component(
    probabilities: np.ndarray,
    *,
    source: int | None,
    target: int | None,
) -> np.ndarray:
    """Apply the train-decided K=5 merge to a later causal transform."""

    values = np.asarray(probabilities, dtype=np.float32)
    if source is None or target is None:
        return values
    if source == target or source < 0 or target < 0 or source >= values.shape[1] or target >= values.shape[1]:
        raise ValueError("invalid frozen regime state merge")
    merged = values.copy()
    merged[:, target] += merged[:, source]
    merged = np.delete(merged, source, axis=1)
    merged /= np.maximum(merged.sum(axis=1, keepdims=True), np.float32(1e-12))
    return merged.astype(np.float32, copy=False)


def _causal_distance_velocity(
    distances: np.ndarray,
    *,
    timestamps_ns: np.ndarray,
    group_values: np.ndarray,
    max_gap_hours: float,
) -> np.ndarray:
    """One-step, forward-only movement relative to the nearest state centre."""

    values = np.asarray(distances, dtype=np.float32)
    result = np.zeros(len(values), dtype=np.float32)
    last: dict[str, tuple[int, float]] = {}
    max_gap_ns = float(max_gap_hours) * 3_600_000_000_000.0
    for position, value in enumerate(values):
        key = str(group_values[position])
        prior = last.get(key)
        timestamp = int(timestamps_ns[position])
        if prior is not None and timestamp - prior[0] <= max_gap_ns:
            result[position] = np.float32(float(value) - prior[1])
        last[key] = (timestamp, float(value))
    return result


def _select_geometry_columns(frame: pd.DataFrame, available: Sequence[str], spec: RegimeGeometrySpec, cfg: CausalMarketRegimeConfig) -> list[str]:
    names = _require_observable_columns(frame, available)
    include = tuple(token.lower() for token in spec.include_tokens)
    exclude = tuple(token.lower() for token in spec.exclude_tokens)
    candidates = [name for name in names if any(token in name.lower() for token in include) and not any(token in name.lower() for token in exclude)]
    coverage = frame.loc[:, candidates].notna().mean() if candidates else pd.Series(dtype=float)
    variance = frame.loc[:, candidates].var(skipna=True) if candidates else pd.Series(dtype=float)
    chosen = [name for name in candidates if coverage.get(name, 0.0) >= float(cfg.minimum_feature_coverage) and np.isfinite(variance.get(name, np.nan)) and variance.get(name, 0.0) > 1e-12]
    # Preserve the declared input order: it is a deterministic, transparent
    # proxy, avoids target scoring, and can be overridden by a frozen list.
    chosen = chosen[: max(0, int(spec.max_features))]
    if len(chosen) < int(spec.min_features):
        raise ValueError(f"geometry {spec.name!r} has only {len(chosen)} supported observable inputs")
    return chosen


def _direction_feature_indices(feature_columns: Sequence[str]) -> tuple[int, ...]:
    """Return signed-direction candidates without letting them name a state.

    A state such as ``coherent expansion`` should be equally likely in a
    coherent up- or down-move.  Direction is therefore a separate surface,
    derived after state fitting from explicitly signed price/trend inputs.
    This small name-routed proxy is deterministic and only consumes the same
    pre-entry fields already admitted to the frozen primary model.
    """

    return tuple(
        index
        for index, name in enumerate(feature_columns)
        if any(token in str(name).lower() for token in _DIRECTION_TOKENS)
    )


def _semantic_state_mapping(
    feature_columns: Sequence[str], centroids: np.ndarray
) -> tuple[dict[str, int], dict[str, Any]]:
    """Name exactly five fitted components from train-only geometry prototypes.

    The GMM remains fully unsupervised.  This routine merely establishes a
    one-to-one semantic coordinate map from its *training* centroids using
    observable geometry/stress dimensions.  It does not read any future row,
    target, return or PnL.  A 5! exhaustive assignment is both clearer and
    cheaper than a general optimizer here.
    """

    values = np.asarray(centroids, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != len(PRIMARY_SEMANTIC_STATE_NAMES):
        return {}, {
            "status": "unavailable_nonfive_effective_state_count",
            "semantic_state_names": list(PRIMARY_SEMANTIC_STATE_NAMES),
        }
    names = [str(name).lower() for name in feature_columns]

    def mean_for(tokens: tuple[str, ...], *, absolute: bool = False) -> np.ndarray:
        indices = [
            index
            for index, name in enumerate(names)
            if any(token in name for token in tokens)
        ]
        if not indices:
            return np.zeros(values.shape[0], dtype=np.float32)
        selected = values[:, indices]
        if absolute:
            selected = np.abs(selected)
        return selected.mean(axis=1).astype(np.float32)

    # Centroids are already robust-scaled.  Each score intentionally omits
    # signed returns/trend so S1 remains a *coherent* expansion, with the
    # sign exposed by the independent direction surface.
    volatility = mean_for(("vol", "atr", "range", "vov", "variance"), absolute=True)
    chop = mean_for(("chop", "compression", "entropy", "inefficien"), absolute=False)
    coherence = mean_for(("efficiency", "coherence", "trend_strength", "breakout"), absolute=True)
    fragmentation = mean_for(
        ("breadth", "dispersion", "corr", "correlation", "dependence", "eigen", "effective_rank"),
        absolute=True,
    )
    stress = mean_for(
        ("spread", "liquid", "depth", "impact", "amihud", "fund", "oi", "basis", "deleverag"),
        absolute=True,
    )
    turnover = mean_for(("volume", "turnover", "flow"), absolute=True)
    scores = np.column_stack(
        (
            -volatility + chop - 0.25 * coherence - 0.25 * stress,
            0.75 * coherence + 0.35 * turnover - 0.45 * chop - 0.25 * stress,
            0.80 * fragmentation + 0.25 * chop + 0.15 * volatility,
            0.85 * stress + 0.55 * volatility + 0.25 * fragmentation,
            0.45 * coherence + 0.35 * turnover - 0.55 * stress - 0.35 * volatility,
        )
    ).astype(np.float32)
    # Choose the globally consistent one-to-one mapping.  Adding a tiny
    # lexical tie-break keeps model serialization stable in symmetric panels.
    best = max(
        permutations(range(values.shape[0])),
        key=lambda assignment: (
            float(sum(scores[component, semantic] for semantic, component in enumerate(assignment))),
            tuple(-item for item in assignment),
        ),
    )
    mapping = {
        semantic: int(component)
        for semantic, component in zip(PRIMARY_SEMANTIC_STATE_NAMES, best)
    }
    return mapping, {
        "status": "train_only_centroid_prototype_assignment",
        "semantic_state_names": list(PRIMARY_SEMANTIC_STATE_NAMES),
        "semantic_to_frozen_component": mapping,
        "prototype_score_by_component": {
            name: scores[:, index].astype(float).tolist()
            for index, name in enumerate(PRIMARY_SEMANTIC_STATE_NAMES)
        },
        "prototype_inputs": {
            "volatility": [name for name in feature_columns if any(token in str(name).lower() for token in ("vol", "atr", "range", "vov", "variance"))],
            "chop": [name for name in feature_columns if any(token in str(name).lower() for token in ("chop", "compression", "entropy", "inefficien"))],
            "coherence": [name for name in feature_columns if any(token in str(name).lower() for token in ("efficiency", "coherence", "trend_strength", "breakout"))],
            "fragmentation": [name for name in feature_columns if any(token in str(name).lower() for token in ("breadth", "dispersion", "corr", "correlation", "dependence", "eigen", "effective_rank"))],
            "stress": [name for name in feature_columns if any(token in str(name).lower() for token in ("spread", "liquid", "depth", "impact", "amihud", "fund", "oi", "basis", "deleverag"))],
        },
        "direction_excluded_from_semantic_assignment": True,
    }


@dataclass
class FrozenCausalMarketRegimeModel:
    """One train-only five-state GMM plus an online sticky posterior filter."""

    system_name: str
    feature_columns: tuple[str, ...]
    imputer: SimpleImputer
    scaler: RobustScaler
    gmm: GaussianMixture
    component_order: np.ndarray
    state_count: int
    stickiness: float
    config: CausalMarketRegimeConfig
    diagnostics: dict[str, Any]
    training_ood_distances_sorted: np.ndarray
    effective_state_centroids: np.ndarray
    training_assigned_distance_sorted: np.ndarray
    semantic_state_indices: Mapping[str, int] = field(default_factory=dict)
    direction_feature_indices: tuple[int, ...] = ()
    merged_state_source: int | None = None
    merged_state_target: int | None = None
    training_history: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        feature_columns: Sequence[str],
        *,
        system_name: str,
        config: CausalMarketRegimeConfig = CausalMarketRegimeConfig(),
    ) -> "FrozenCausalMarketRegimeModel":
        features = tuple(_require_observable_columns(frame, feature_columns))
        requested_primary_k = int(config.primary_state_count) if str(system_name) == "primary" else None
        if requested_primary_k is not None and not 3 <= requested_primary_k <= 6:
            raise ValueError("primary_state_count must be in [3, 6]")
        minimum_k = requested_primary_k if requested_primary_k is not None else 3
        if len(frame) < max(int(config.minimum_rows_per_state) * minimum_k, 100):
            raise ValueError("insufficient historical rows for causal market states")
        order, timestamps_ns, groups = _time_group_order(frame, config)
        raw = frame.loc[:, features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=True)
        train_positions = _deterministic_sample_positions(len(order), int(config.max_train_rows))
        fit_positions = order[train_positions]
        imputer = SimpleImputer(strategy="median")
        scaler = RobustScaler(quantile_range=(10.0, 90.0))
        fit_values = scaler.fit_transform(imputer.fit_transform(raw[fit_positions])).astype(np.float32, copy=False)
        fit_values = np.clip(
            fit_values,
            -float(config.robust_scaled_clip),
            float(config.robust_scaled_clip),
        ).astype(np.float32, copy=False)
        proxy_positions = _deterministic_sample_positions(len(fit_values), int(config.max_proxy_rows))
        proxy = fit_values[proxy_positions]
        requested_k = requested_primary_k
        k_values = sorted({int(k) for k in config.diagnostic_k_values if 2 <= int(k) <= 8} | ({requested_k} if requested_k is not None else set()))
        k_diagnostics: list[dict[str, float]] = []
        for k in k_values:
            if len(proxy) < max(k * int(config.minimum_rows_per_state), 30):
                continue
            model = _fit_diag_gmm(
                proxy,
                components=k,
                config=config,
                seed=int(config.random_state) + k,
                max_iter=min(int(config.max_iter), 80),
            )
            labels = model.predict(proxy)
            occupancy = float(np.bincount(labels, minlength=k).min() / max(len(labels), 1))
            k_diagnostics.append({"k": float(k), "proxy_bic": float(model.bic(proxy)), "minimum_occupancy": occupancy, "eligible": float(occupancy >= float(config.minimum_state_soft_occupancy)), "fixed_primary": float(k == requested_k and requested_k is not None)})
        if requested_k is not None:
            selected_k = requested_k
        else:
            eligible = [row for row in k_diagnostics if bool(row["eligible"])] or k_diagnostics
            if not eligible:
                raise ValueError("no supported K candidates for geometry regime system")
            selected_k = int(min(eligible, key=lambda row: (float(row["proxy_bic"]), int(row["k"]))) ["k"])
        gmm = _fit_diag_gmm(
            fit_values,
            components=selected_k,
            config=config,
            seed=int(config.random_state),
            max_iter=int(config.max_iter),
        )
        # Stable coordinate order for this frozen model only.  Cross-fold
        # semantic alignment must use a separate frozen-anchor procedure.
        component_order = np.argsort(gmm.means_[:, 0], kind="stable").astype(np.int64)
        values_all = scaler.transform(imputer.transform(raw)).astype(np.float32, copy=False)
        values_all = np.clip(
            values_all,
            -float(config.robust_scaled_clip),
            float(config.robust_scaled_clip),
        ).astype(np.float32, copy=False)
        training_ood_distances = np.maximum(0.0, -gmm.score_samples(values_all)).astype(np.float32)
        raw_prob = gmm.predict_proba(values_all)[:, component_order].astype(np.float32)
        merge_source: int | None = None
        merge_target: int | None = None
        merge_occupancy = raw_prob.mean(axis=0).astype(np.float32)
        if str(system_name) == "primary" and bool(config.primary_merge_low_support_state):
            raw_prob, merge_source, merge_target, merge_occupancy = _merge_low_support_component(
                raw_prob,
                gmm.means_[component_order],
                minimum_occupancy=float(config.minimum_state_soft_occupancy),
            )
        effective_centroids = gmm.means_[component_order].astype(np.float32, copy=True)
        if merge_source is not None and merge_target is not None:
            source_weight = float(merge_occupancy[merge_source])
            target_weight = float(merge_occupancy[merge_target])
            effective_centroids[merge_target] = (
                target_weight * effective_centroids[merge_target]
                + source_weight * effective_centroids[merge_source]
            ) / max(target_weight + source_weight, 1e-12)
            effective_centroids = np.delete(effective_centroids, merge_source, axis=0)
        semantic_state_indices: dict[str, int] = {}
        semantic_diagnostics: dict[str, Any] = {
            "status": "not_primary_system",
            "semantic_state_names": list(PRIMARY_SEMANTIC_STATE_NAMES),
        }
        direction_indices: tuple[int, ...] = ()
        if str(system_name) == "primary":
            semantic_state_indices, semantic_diagnostics = _semantic_state_mapping(
                features, effective_centroids
            )
            direction_indices = _direction_feature_indices(features)
        training_centroid_distances = np.sqrt(
            np.maximum(
                ((values_all[:, None, :] - effective_centroids[None, :, :]) ** 2).sum(axis=2),
                0.0,
            )
        ).astype(np.float32)
        training_assigned_distance = training_centroid_distances[
            np.arange(len(values_all)), np.argmax(raw_prob, axis=1)
        ]
        raw_prob = raw_prob[order]
        ordered_ts, ordered_groups = timestamps_ns[order], groups[order]
        sticky_diagnostics: list[dict[str, float]] = []
        best: tuple[int, float, float] | None = None
        selected_stickiness = 0.0
        for rho in sorted({float(np.clip(value, 0.0, 0.99)) for value in config.stickiness_values}):
            filtered, _age, switch, _initial, _hist = _state_dynamics(raw_prob, timestamps_ns=ordered_ts, group_values=ordered_groups, stickiness=rho, max_gap_hours=float(config.max_gap_hours))
            metrics = _persistence_metrics(filtered, timestamps_ns=ordered_ts, group_values=ordered_groups, max_gap_hours=float(config.max_gap_hours))
            confidence = metrics["mean_max_probability"]
            switch_mean = float(np.mean(switch))
            passed = bool(
                metrics["median_dwell_hours"] >= 6.0
                and metrics["temporal_switch_rate"] <= 0.10
                and metrics["minimum_state_occupancy"] >= float(config.minimum_state_soft_occupancy)
                and confidence >= 0.55
            )
            score = (
                0.40 * confidence
                + 0.25 * min(metrics["median_dwell_hours"] / 6.0, 1.0)
                + 0.20 * metrics["minimum_state_occupancy"]
                + 0.15 * max(0.0, 1.0 - metrics["temporal_switch_rate"] / 0.10)
            )
            sticky_diagnostics.append({"stickiness": rho, "mean_switch_probability": switch_mean, "persistent_state_gate_passed": passed, "objective": score, **metrics})
            candidate = (int(passed), score, -rho)  # gate first, then lower inertia on ties
            if best is None or candidate > best:
                best, selected_stickiness = candidate, rho
        filtered, ages, switches, initial, history = _state_dynamics(raw_prob, timestamps_ns=ordered_ts, group_values=ordered_groups, stickiness=selected_stickiness, max_gap_hours=float(config.max_gap_hours))
        # histories are keyed by group and are carried to the first subsequent
        # transform row, never rebuilt from evaluation rows.
        diagnostics = {
            "schema": CAUSAL_MARKET_REGIME_SCHEMA,
            "system": str(system_name),
            "state_count": int(raw_prob.shape[1]),
            "gmm_state_count_before_merge": int(selected_k),
            "feature_columns": list(features),
            "train_rows": int(len(frame)),
            "fit_rows": int(len(fit_positions)),
            "proxy_rows": int(len(proxy)),
            "k_selection_proxy": k_diagnostics,
            "k_selection_constraint": f"primary_fixed_k{requested_k}" if requested_k is not None else "geometry_proxy_bic_with_support_gate",
            "selected_k": int(selected_k),
            "selected_reg_covar": float(gmm.reg_covar),
            "effective_state_centroids": effective_centroids.tolist(),
            "semantic_ontology": semantic_diagnostics,
            "direction_surface": {
                "feature_columns": [features[index] for index in direction_indices],
                "state_input_role": "separate_from_semantic_state_naming",
                "semantics": "signed_robust_scaled_direction_score_and_logistic_probability",
            },
            "training_assigned_distance_q50": float(np.median(training_assigned_distance)),
            "training_assigned_distance_q90": float(np.quantile(training_assigned_distance, 0.90)),
            "postfit_low_support_merge": {
                "enabled": bool(config.primary_merge_low_support_state) if str(system_name) == "primary" else False,
                "source_state": merge_source,
                "target_state": merge_target,
                "premerge_soft_occupancy": merge_occupancy.tolist(),
            },
            "stickiness_selection": sticky_diagnostics,
            "selected_stickiness": float(selected_stickiness),
            "selected_stickiness_persistent_gate_passed": bool(
                next(row["persistent_state_gate_passed"] for row in sticky_diagnostics if row["stickiness"] == selected_stickiness)
            ),
            "training_mean_switch_probability": float(np.mean(switches)),
            "training_mean_state_age_hours": float(np.mean(ages)),
            "ood_reference_rows": int(len(training_ood_distances)),
            "causality": "frozen train-only imputer/scaler/clipping/GMM; forward-only posterior filter; no label/outcome inputs",
        }
        return cls(
            system_name=str(system_name), feature_columns=features, imputer=imputer,
            scaler=scaler, gmm=gmm, component_order=component_order,
            state_count=int(raw_prob.shape[1]), stickiness=selected_stickiness,
            config=config, diagnostics=diagnostics,
            training_ood_distances_sorted=np.sort(training_ood_distances),
            effective_state_centroids=effective_centroids,
            training_assigned_distance_sorted=np.sort(training_assigned_distance),
            semantic_state_indices=semantic_state_indices,
            direction_feature_indices=direction_indices,
            merged_state_source=merge_source, merged_state_target=merge_target,
            training_history=history,
        )

    @property
    def output_columns(self) -> tuple[str, ...]:
        prefix = "market_regime" if self.system_name == "primary" else f"geometry_regime__{_safe_name(self.system_name)}"
        fields = [
            *(f"{prefix}__state_p_{state}" for state in range(self.state_count)),
            *(f"{prefix}__state_centroid_distance_p_{state}" for state in range(self.state_count)),
            f"{prefix}__assigned_centroid_distance",
            f"{prefix}__within_state_radius_percentile",
            f"{prefix}__state_boundary_margin",
            f"{prefix}__centroid_distance_velocity",
            f"{prefix}__entropy", f"{prefix}__top2_margin",
            f"{prefix}__state_age_hours", f"{prefix}__state_switch_probability",
            f"{prefix}__ood_distance_percentile", f"{prefix}__input_coverage",
            *(f"{prefix}__phase_p_{phase}" for phase in PHASE_NAMES),
        ]
        if self.system_name == "primary":
            fields.extend((f"{prefix}__phase_entropy", f"{prefix}__phase_top2_margin"))
            # Compact public aliases are intentionally emitted only for the
            # primary five-state system.  They make the decision-time regime
            # contract easy to consume without forcing callers to depend on
            # the internal GMM prefix.  Geometry specialists keep their
            # namespaced columns because their state coordinates are not
            # interchangeable.
            fields.extend(
                (
                    *(f"regime_state_probability_{state}" for state in range(self.state_count)),
                    *(
                        f"market_regime__{semantic}_probability"
                        for semantic in self.semantic_state_indices
                    ),
                    *(f"regime_p_{semantic}" for semantic in self.semantic_state_indices),
                    "regime_entropy",
                    "regime_top2_margin",
                    "state_age_hours",
                    "state_age",
                    "state_switch_probability",
                    "market_regime__direction_score",
                    "market_regime__direction_positive_probability",
                    "market_direction_sign",
                    *(f"transition_{phase}_probability" for phase in PHASE_NAMES),
                )
            )
        return tuple(fields)

    def transform(self, frame: pd.DataFrame, *, carry_history: bool = False) -> pd.DataFrame:
        _require_observable_columns(frame, self.feature_columns)
        order, timestamps_ns, groups = _time_group_order(frame, self.config)
        raw = frame.loc[:, self.feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=True)
        input_coverage = np.isfinite(raw).mean(axis=1).astype(np.float32)
        values = self.scaler.transform(self.imputer.transform(raw)).astype(np.float32, copy=False)
        values = np.clip(
            values,
            -float(self.config.robust_scaled_clip),
            float(self.config.robust_scaled_clip),
        ).astype(np.float32, copy=False)
        probability = self.gmm.predict_proba(values)[:, self.component_order].astype(np.float32)
        probability = _collapse_merged_component(
            probability,
            source=self.merged_state_source,
            target=self.merged_state_target,
        )
        ood_distance = np.maximum(0.0, -self.gmm.score_samples(values)).astype(np.float32)
        ood_percentile = (np.searchsorted(self.training_ood_distances_sorted, ood_distance, side="right") / max(len(self.training_ood_distances_sorted), 1)).astype(np.float32)
        filtered, ages, switches, initial, history = _state_dynamics(probability[order], timestamps_ns=timestamps_ns[order], group_values=groups[order], stickiness=self.stickiness, max_gap_hours=float(self.config.max_gap_hours), initial=self.training_history)
        restored = np.empty_like(filtered, dtype=np.float32); restored[order] = filtered
        restored_age = np.empty(len(frame), dtype=np.float32); restored_age[order] = ages
        restored_switch = np.empty(len(frame), dtype=np.float32); restored_switch[order] = switches
        restored_initial = np.empty(len(frame), dtype=bool); restored_initial[order] = initial
        phases = _phase_simplex(restored_switch, restored_age, restored_initial, float(self.config.transition_horizon_hours))
        entropy = _entropy(restored)
        sorted_prob = np.sort(restored, axis=1)
        centroid_distances = np.sqrt(
            np.maximum(
                ((values[:, None, :] - self.effective_state_centroids[None, :, :]) ** 2).sum(axis=2),
                0.0,
            )
        ).astype(np.float32)
        assigned = np.argmax(restored, axis=1)
        assigned_distance = centroid_distances[np.arange(len(frame)), assigned]
        radius_percentile = (
            np.searchsorted(self.training_assigned_distance_sorted, assigned_distance, side="right")
            / max(len(self.training_assigned_distance_sorted), 1)
        ).astype(np.float32)
        distance_order = np.sort(centroid_distances, axis=1)
        boundary_margin = (distance_order[:, 1] - distance_order[:, 0]).astype(np.float32)
        ordered_velocity = _causal_distance_velocity(
            assigned_distance[order], timestamps_ns=timestamps_ns[order],
            group_values=groups[order], max_gap_hours=float(self.config.max_gap_hours),
        )
        distance_velocity = np.empty(len(frame), dtype=np.float32)
        distance_velocity[order] = ordered_velocity
        prefix = "market_regime" if self.system_name == "primary" else f"geometry_regime__{_safe_name(self.system_name)}"
        data: dict[str, np.ndarray] = {
            f"{prefix}__state_p_{state}": restored[:, state]
            for state in range(self.state_count)
        }
        data.update({
            f"{prefix}__state_centroid_distance_p_{state}": centroid_distances[:, state]
            for state in range(self.state_count)
        })
        data.update({
            f"{prefix}__assigned_centroid_distance": assigned_distance.astype(np.float32),
            f"{prefix}__within_state_radius_percentile": radius_percentile,
            f"{prefix}__state_boundary_margin": boundary_margin,
            f"{prefix}__centroid_distance_velocity": distance_velocity,
        })
        data.update({f"{prefix}__entropy": entropy, f"{prefix}__top2_margin": (sorted_prob[:, -1] - sorted_prob[:, -2]).astype(np.float32), f"{prefix}__state_age_hours": restored_age, f"{prefix}__state_switch_probability": restored_switch, f"{prefix}__ood_distance_percentile": ood_percentile, f"{prefix}__input_coverage": input_coverage})
        data.update({f"{prefix}__phase_p_{phase}": phases[:, i] for i, phase in enumerate(PHASE_NAMES)})
        if self.system_name == "primary":
            phase_entropy = _entropy(phases)
            phase_order = np.sort(phases, axis=1)
            data[f"{prefix}__phase_entropy"] = phase_entropy
            data[f"{prefix}__phase_top2_margin"] = (phase_order[:, -1] - phase_order[:, -2]).astype(np.float32)
            # Public, unit-explicit aliases for the required five-state and
            # transition representation.  They are exact copies rather than
            # a second calculation, which avoids any numerical drift between
            # the feature-store and diagnostic interfaces.
            data.update(
                {
                    f"regime_state_probability_{state}": restored[:, state]
                    for state in range(self.state_count)
                }
            )
            data.update(
                {
                    f"market_regime__{semantic}_probability": restored[:, component]
                    for semantic, component in self.semantic_state_indices.items()
                }
            )
            # Short aliases form the stable public five-state simplex.  Unlike
            # raw GMM coordinate IDs, each column has the train-only semantic
            # ontology recorded in the frozen diagnostics/manifest.
            data.update(
                {
                    f"regime_p_{semantic}": restored[:, component]
                    for semantic, component in self.semantic_state_indices.items()
                }
            )
            data["regime_entropy"] = entropy
            data["regime_top2_margin"] = (
                sorted_prob[:, -1] - sorted_prob[:, -2]
            ).astype(np.float32)
            data["state_age_hours"] = restored_age
            # Kept as the requested terse alias; the sibling makes the unit
            # unambiguous in model/config contracts.
            data["state_age"] = restored_age
            data["state_switch_probability"] = restored_switch
            if self.direction_feature_indices:
                direction_score = np.clip(
                    values[:, self.direction_feature_indices].mean(axis=1), -6.0, 6.0
                ).astype(np.float32)
            else:
                # Explicitly neutral when the frozen source contract contains
                # no signed direction field; state geometry still remains
                # available and no arbitrary feature is substituted.
                direction_score = np.zeros(len(frame), dtype=np.float32)
            data["market_regime__direction_score"] = direction_score
            data["market_regime__direction_positive_probability"] = (
                1.0 / (1.0 + np.exp(-direction_score))
            ).astype(np.float32)
            data["market_direction_sign"] = np.sign(direction_score).astype(np.float32)
            data.update(
                {
                    f"transition_{phase}_probability": phases[:, index]
                    for index, phase in enumerate(PHASE_NAMES)
                }
            )
        if carry_history:
            self.training_history = history
        return pd.DataFrame(data, index=frame.index, dtype=np.float32)


@dataclass
class FrozenCausalMarketRegimeSystems:
    """Four independent frozen soft-state systems with a common causal API."""

    models: Mapping[str, FrozenCausalMarketRegimeModel]
    feature_views: Mapping[str, tuple[str, ...]]
    config: CausalMarketRegimeConfig
    diagnostics: dict[str, Any]

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        feature_columns: Sequence[str],
        *,
        specs: Sequence[RegimeGeometrySpec] = DEFAULT_GEOMETRY_SPECS,
        config: CausalMarketRegimeConfig = CausalMarketRegimeConfig(),
    ) -> "FrozenCausalMarketRegimeSystems":
        _require_observable_columns(frame, feature_columns)
        models: dict[str, FrozenCausalMarketRegimeModel] = {}
        views: dict[str, tuple[str, ...]] = {}
        specialist_features: set[str] = set()
        for spec in specs:
            if spec.name in models:
                raise ValueError(f"duplicate regime geometry name: {spec.name}")
            view = tuple(_select_geometry_columns(frame, feature_columns, spec, config))
            if spec.name != "primary":
                # The specialists represent different economic geometries, not
                # four resamplings of a single generic feature surface.  The
                # primary keeps its established broad proxy; specialists are
                # mutually disjoint in their raw inputs in declared order.
                view = tuple(name for name in view if name not in specialist_features)
                if len(view) < int(spec.min_features):
                    raise ValueError(
                        f"geometry {spec.name!r} has insufficient disjoint observable inputs"
                    )
                specialist_features.update(view)
            views[spec.name] = view
            models[spec.name] = FrozenCausalMarketRegimeModel.fit(frame, view, system_name=spec.name, config=config)
        if not models:
            raise ValueError("at least one regime geometry system is required")
        return cls(models, views, config, {"schema": CAUSAL_MARKET_REGIME_SCHEMA, "systems": {name: model.diagnostics for name, model in models.items()}, "causality": "all systems are frozen train-only and forward-filtered"})

    @property
    def output_columns(self) -> tuple[str, ...]:
        return tuple(column for model in self.models.values() for column in model.output_columns)

    def transform(self, frame: pd.DataFrame, *, carry_history: bool = False) -> pd.DataFrame:
        parts = [model.transform(frame, carry_history=carry_history) for model in self.models.values()]
        result = pd.concat(parts, axis=1)
        if result.columns.duplicated().any():
            raise RuntimeError("causal market regime systems generated colliding columns")
        return result.astype(np.float32, copy=False)


def fit_causal_market_regime_systems(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    specs: Sequence[RegimeGeometrySpec] = DEFAULT_GEOMETRY_SPECS,
    config: CausalMarketRegimeConfig = CausalMarketRegimeConfig(),
) -> FrozenCausalMarketRegimeSystems:
    """Convenience entry point for a frozen, label-free market-regime fit."""

    return FrozenCausalMarketRegimeSystems.fit(frame, feature_columns, specs=specs, config=config)


def fit_causal_market_geometry_systems(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    specs: Sequence[RegimeGeometrySpec] = LATENT_GEOMETRY_SPECS,
    config: CausalMarketRegimeConfig = CausalMarketRegimeConfig(),
) -> FrozenCausalMarketRegimeSystems:
    """Fit only the independent latent geometry systems.

    This entry point is deliberately non-overlapping with the primary five-
    state engine.  It is useful for a geometry-only sidecar or for a caller
    that already owns a frozen primary model.
    """

    names = tuple(spec.name for spec in specs)
    if not names or "primary" in names:
        raise ValueError("geometry systems must be non-empty and exclude primary")
    return FrozenCausalMarketRegimeSystems.fit(
        frame, feature_columns, specs=specs, config=config,
    )


# Short aliases form the intended OOF-materializer API:
# ``CausalMarketRegimeSystem.fit(train, ...)`` then ``.transform(eval)``;
# use ``CausalMarketRegimeSystems`` for the four geometry views plus primary.
MarketRegimeSystemConfig = CausalMarketRegimeConfig
CausalMarketRegimeSystem = FrozenCausalMarketRegimeModel
CausalMarketRegimeSystems = FrozenCausalMarketRegimeSystems


__all__ = [
    "CAUSAL_MARKET_REGIME_SCHEMA", "DEFAULT_GEOMETRY_SPECS", "LATENT_GEOMETRY_SPECS",
    "LATENT_GEOMETRY_SYSTEM_NAMES", "latent_geometry_output_feature_names", "FORBIDDEN_INPUT_TOKENS",
    "PHASE_NAMES", "PRIMARY_SEMANTIC_STATE_NAMES", "CONTINUOUS_CONTEXT_SOURCE_CONTRACT", "CONTINUOUS_CONTEXT_OPERATORS",
    "CONTINUOUS_CONTEXT_FEATURE_KEYS", "RELATIONSHIP_BREAK_SOURCE_PAIRS",
    "RELATIONSHIP_BREAK_OPERATORS", "RELATIONSHIP_BREAK_FEATURE_KEYS",
    "CausalContinuousContextConfig", "CausalRelationshipBreakConfig",
    "build_causal_continuous_context_features", "continuous_context_feature_names",
    "build_causal_relationship_break_features", "relationship_break_feature_names",
    "CausalMarketRegimeConfig", "MarketRegimeSystemConfig", "CausalMarketRegimeSystem",
    "CausalMarketRegimeSystems", "FrozenCausalMarketRegimeModel", "FrozenCausalMarketRegimeSystems",
    "RegimeGeometrySpec", "fit_causal_market_regime_systems", "fit_causal_market_geometry_systems",
]
