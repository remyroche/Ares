"""Causal market-state composites associated with adverse model residuals.

The features in this module use only OHLCV, open interest, and funding inputs.
They are market/trust context for the meta model, never outcome residuals and
never hard policy gates.  Robust normalization is fitted causally from prior
observations at each timestamp.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
import warnings

import numpy as np
import pandas as pd

from extreme_price_movements.market_regime_change_contract import (
    MARKET_REGIME_CHANGE_2D_HOURS,
    MARKET_REGIME_CHANGE_FEATURE_KEYS,
    MARKET_REGIME_CHANGE_OPERATORS,
    MARKET_REGIME_CHANGE_SCHEMA_VERSION,
    MARKET_REGIME_CHANGE_SOURCES,
    market_regime_change_feature_name,
)


NEGATIVE_RESIDUAL_FEATURE_SCHEMA_VERSION = 2
NEGATIVE_RESIDUAL_CAUSAL_WINDOW_HOURS = 24 * 30
NEGATIVE_RESIDUAL_MIN_HISTORY_HOURS = 24 * 7
NEGATIVE_RESIDUAL_MIN_MARKET_ASSETS = 3


NEGATIVE_RESIDUAL_PRIMITIVE_FEATURE_KEYS = [
    "negative_breadth_pct",
    "extreme_negative_breadth_pct",
    "median_alt_minus_btc",
    "breadth_dispersion",
    "downside_breadth_intensity",
    "btc_resilience_alt_weakness",
    "btc_oi_dominance_z_ratio",
    "btc_ex_eth_oi_dominance_z_ratio",
    "btc_over_eth_dominance_roc",
    "peer_decoupling_acceleration",
    "short_covering_score_market",
]

NEGATIVE_RESIDUAL_COMPOSITE_FEATURE_KEYS = [
    "correlation_breakdown_dispersion",
    "btc_decoupling_dispersion",
    "correlation_heterogeneity_dispersion",
    "fragmented_new_low_breadth",
    "deleveraging_without_followthrough",
    "short_breakout_exhaustion",
    "funding_deleveraging_divergence",
    "funding_confirmed_short_covering",
    "funding_confirmed_long_flush",
    "post_flush_leverage_rebuild",
    "fragile_leverage_rebuild",
    "flush_recovery_state",
    "fragmented_flush_recovery",
    "broad_washout_recovery",
    "range_climax_reversal",
    "deleveraged_range_climax_reversal",
    "compressed_index_fragmented_assets",
    "peer_volatility_decoupling",
    "thin_compression",
    "unconfirmed_long_breakout",
    "leveraged_long_breakout_risk",
    "false_clean_short",
    "short_signal_recovery_conflict",
    "late_short_after_deleveraging",
]

# Compact, direction-agnostic temporal mechanisms.  These deliberately describe
# whether an observable state is coherent and persistent, not whether a prior
# model recently won or lost.
NEGATIVE_RESIDUAL_TEMPORAL_MECHANISM_FEATURE_KEYS = [
    # Compression quality.
    "compression_quality_consistency",
    "compression_confirmation_ratio",
    "healthy_compression_score",
    "exhausted_compression_score",
    "fragile_compression_score",
    "compression_duration_72h_norm",
    "compression_integral_72h",
    "compression_onset_shock_24h",
    # Persistent adverse context and failed recovery.
    "short_default_damage_pressure",
    "short_default_damage_ema_5d",
    "short_default_damage_integral_5d",
    "short_default_damage_max_5d",
    "short_default_adverse_duration_5d_norm",
    "market_state_transition_entropy_5d",
    "market_state_persistence_5d",
    "recovery_failure_score_24h",
    # Direction-agnostic breakout quality.
    "breakout_efficiency_4h",
    "breakout_participation_4h",
    "breakout_retention_4h",
    "breakout_confirmation_ratio",
    "breakout_disagreement_score",
    "breakout_bilateral_failure_score",
]

NEGATIVE_RESIDUAL_META_FEATURE_KEYS = list(
    dict.fromkeys(
        NEGATIVE_RESIDUAL_PRIMITIVE_FEATURE_KEYS
        + NEGATIVE_RESIDUAL_COMPOSITE_FEATURE_KEYS
        + NEGATIVE_RESIDUAL_TEMPORAL_MECHANISM_FEATURE_KEYS
        + MARKET_REGIME_CHANGE_FEATURE_KEYS
    )
)

# Narrow OOS-supported subset exposed to meta training.  The complete library
# remains materializable for diagnostics and regime research.
NEGATIVE_RESIDUAL_PROMOTED_META_FEATURE_KEYS = [
    "short_covering_score_market",
    "funding_confirmed_short_covering",
    "flush_recovery_state",
    "negative_breadth_pct",
    "extreme_negative_breadth_pct",
    "downside_breadth_intensity",
    "post_flush_leverage_rebuild",
    "funding_deleveraging_divergence",
    "funding_confirmed_long_flush",
]

RESIDUAL_STATE_TARGET_HORIZONS_HOURS: tuple[int, ...] = (6, 12, 24)
RESIDUAL_STATE_TARGET_METRICS: tuple[str, ...] = (
    "directional_ev_divergence",
    "bullish_tape_adverse_ev",
    "timestamp_ev_sign_disagreement",
    "positive_timestamp_negative_ev",
    "tail_absence_factor",
    "persistent_subthreshold_damage",
    "persistent_material_nontail",
)


def residual_state_target_feature_names(
    horizons: Sequence[int] = RESIDUAL_STATE_TARGET_HORIZONS_HOURS,
) -> list[str]:
    """Outcome-only discovery targets; never part of the model feature schema."""

    return [
        f"resid_target_{scope}_{metric}_{int(horizon)}h"
        for scope in ("side", "side_archetype")
        for horizon in horizons
        for metric in RESIDUAL_STATE_TARGET_METRICS
    ]


def negative_residual_feature_contract() -> dict[str, object]:
    """Stable schema recorded by training/inference manifests."""
    payload: dict[str, object] = {
        "schema_version": NEGATIVE_RESIDUAL_FEATURE_SCHEMA_VERSION,
        "feature_family": "negative_residual_market_context",
        "primitive_features": list(NEGATIVE_RESIDUAL_PRIMITIVE_FEATURE_KEYS),
        "composite_features": list(NEGATIVE_RESIDUAL_COMPOSITE_FEATURE_KEYS),
        "temporal_mechanism_features": list(
            NEGATIVE_RESIDUAL_TEMPORAL_MECHANISM_FEATURE_KEYS
        ),
        "promoted_meta_features": list(
            NEGATIVE_RESIDUAL_PROMOTED_META_FEATURE_KEYS
        ),
        "market_regime_change": {
            "schema_version": MARKET_REGIME_CHANGE_SCHEMA_VERSION,
            "sources": dict(MARKET_REGIME_CHANGE_SOURCES),
            "operators": list(MARKET_REGIME_CHANGE_OPERATORS),
            "features": list(MARKET_REGIME_CHANGE_FEATURE_KEYS),
            "cumulative_change_hours": MARKET_REGIME_CHANGE_2D_HOURS,
        },
        "allowed_sources": ["OHLCV", "open_interest", "funding"],
        "forbidden_sources": ["outcomes", "future_path", "orderbook", "spread"],
        "causal_window_hours": NEGATIVE_RESIDUAL_CAUSAL_WINDOW_HOURS,
        "minimum_history_hours": NEGATIVE_RESIDUAL_MIN_HISTORY_HOURS,
        "minimum_market_assets": NEGATIVE_RESIDUAL_MIN_MARKET_ASSETS,
        "composite_transform": "arcsinh",
        "dtype": "float32",
    }
    stable = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["contract_hash"] = "sha256:" + sha256(stable.encode("utf-8")).hexdigest()
    return payload

_SOURCE_KEYS = {
    "ret4h",
    "ret_resid_btc_4h",
    "corr_btc_24h",
    "corr_eth_24h",
    "market_dispersion_4h",
    "pct_assets_new_low_24h",
    "price_recovery_from_low_24h_atr",
    "mkt_median_oi_drawdown_from_peak_24h",
    "mkt_median_oi_chg_4h_rz",
    "mkt_pct_oi_chg_4h_rz_lt_minus1",
    "funding_1d_chg_ts_resid",
    "price_down_oi_down_4h_rz",
    "price_up_oi_up_4h_rz",
    "mkt_pct_price_up_oi_up_4h",
    "mkt_flush_exhaustion_score",
    "breadth_recovery_from_6h_min",
    "range_climax_decay_4h",
    "log_realized_vol_cp_absratio_8_32",
    "rv_24h_peer_resid",
    "range_per_volume",
    "mkt_ret_4h",
    "mkt_ret_1h",
    "market_downside_pairwise_corr_24h",
    "asset_short_covering_score",
    "oi_value_z_30d",
    "oi_value_1d_chg_z_90d",
    "symbol_minus_mkt_ret_1h",
}


def expand_negative_residual_feature_dependencies(keys: Sequence[str]) -> set[str]:
    requested = {str(key) for key in keys}
    if requested.intersection(NEGATIVE_RESIDUAL_META_FEATURE_KEYS):
        requested.update(_SOURCE_KEYS)
    if requested.intersection(MARKET_REGIME_CHANGE_FEATURE_KEYS):
        requested.update(MARKET_REGIME_CHANGE_SOURCES.values())
    return requested


def _symbol(columns: pd.Index, root: str) -> object | None:
    root = root.upper()
    preferred: list[object] = []
    fallback: list[object] = []
    for column in columns:
        token = str(column).upper().replace("-", "/")
        if token == root or token.startswith(root + "/") or token.startswith(root + ":"):
            # Perpetual feature panels can carry a collateral-denominated
            # duplicate (for example BTC/USD:BTC) before the tradeable USD
            # contract. Prefer USD settlement so benchmark series retain the
            # same availability as the tradable universe.
            if "/USD:USD" in token:
                preferred.append(column)
            else:
                fallback.append(column)
    return preferred[0] if preferred else (fallback[0] if fallback else None)


def _aligned(
    features: Mapping[str, pd.DataFrame],
    name: str,
    index: pd.Index,
    columns: pd.Index,
) -> pd.DataFrame:
    value = features.get(name)
    if not isinstance(value, pd.DataFrame):
        return pd.DataFrame(np.nan, index=index, columns=columns, dtype=np.float32)
    return value.reindex(index=index, columns=columns).astype(np.float32)


def _market_series(frame: pd.DataFrame, *, statistic: str = "median") -> pd.Series:
    values = frame.to_numpy(dtype=np.float32, copy=False)
    valid_count = np.sum(np.isfinite(values), axis=1)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if statistic == "q10":
            result = np.nanquantile(values, 0.10, axis=1)
        elif statistic == "q90":
            result = np.nanquantile(values, 0.90, axis=1)
        elif statistic == "std":
            result = np.nanstd(values, axis=1)
        elif statistic == "iqr":
            result = np.nanquantile(values, 0.75, axis=1) - np.nanquantile(
                values, 0.25, axis=1
            )
        else:
            result = np.nanmedian(values, axis=1)
    result = np.asarray(result, dtype=np.float32)
    result[valid_count < NEGATIVE_RESIDUAL_MIN_MARKET_ASSETS] = np.nan
    return pd.Series(result, index=frame.index, dtype=np.float32)


def _broadcast(series: pd.Series, columns: pd.Index) -> pd.DataFrame:
    values = np.broadcast_to(
        series.to_numpy(dtype=np.float32, copy=False)[:, None],
        (len(series), len(columns)),
    )
    return pd.DataFrame(values, index=series.index, columns=columns).astype(np.float32)


def _causal_robust_z(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype(np.float32)
    history = numeric.shift(1)
    median = history.rolling(window, min_periods=min_periods).median()
    absolute = history.sub(median).abs()
    mad = absolute.rolling(window, min_periods=min_periods).median()
    return (
        numeric.sub(median)
        .div(1.4826 * mad + np.float32(1e-6))
        .clip(-5.0, 5.0)
        .astype(np.float32)
    )


def _positive(series: pd.Series) -> pd.Series:
    return series.clip(lower=0.0).astype(np.float32)


def _consecutive_fraction(
    active: pd.Series,
    *,
    maximum_bars: int,
) -> pd.Series:
    """Causal consecutive-state age, normalized and capped."""

    values = active.fillna(False).to_numpy(dtype=bool, copy=False)
    result = np.zeros(len(values), dtype=np.float32)
    count = 0
    denominator = np.float32(max(int(maximum_bars), 1))
    for position, is_active in enumerate(values):
        count = min(count + 1, int(maximum_bars)) if is_active else 0
        result[position] = np.float32(count) / denominator
    return pd.Series(result, index=active.index, dtype=np.float32)


def _row_mean(series: Sequence[pd.Series]) -> pd.Series:
    values = np.column_stack(
        [pd.to_numeric(item, errors="coerce").to_numpy(np.float32) for item in series]
    )
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = np.nanmean(values, axis=1)
    return pd.Series(result.astype(np.float32), index=series[0].index)


def _row_std(series: Sequence[pd.Series]) -> pd.Series:
    values = np.column_stack(
        [pd.to_numeric(item, errors="coerce").to_numpy(np.float32) for item in series]
    )
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = np.nanstd(values, axis=1)
    return pd.Series(result.astype(np.float32), index=series[0].index)


def calculate_market_regime_change_series(
    levels: Mapping[str, pd.Series],
    *,
    bars_per_hour: int = 1,
) -> dict[str, pd.Series]:
    """Calculate causal transition geometry from aligned market-state levels."""

    lag_1h = max(int(bars_per_hour), 1)
    lag_2d = MARKET_REGIME_CHANGE_2D_HOURS * lag_1h
    output: dict[str, pd.Series] = {}
    for source_name in MARKET_REGIME_CHANGE_SOURCES:
        source = levels.get(source_name)
        if source is None:
            continue
        value = pd.to_numeric(source, errors="coerce").astype(np.float32)
        delta = value - value.shift(lag_1h)
        previous_delta = delta.shift(lag_1h)
        acceleration = delta - previous_delta
        cumulative = value - value.shift(lag_2d)
        sign = np.sign(delta)
        previous_sign = np.sign(previous_delta)
        sign_flip = (
            delta.notna()
            & previous_delta.notna()
            & sign.ne(0.0)
            & previous_sign.ne(0.0)
            & sign.ne(previous_sign)
        ).astype(np.float32).where(delta.notna() & previous_delta.notna())
        values = {
            "delta_1h": delta,
            "acceleration_1h": acceleration,
            "cumulative_change_2d": cumulative,
            "sign_flip_1h": sign_flip,
        }
        for operator, series in values.items():
            output[
                market_regime_change_feature_name(source_name, operator)
            ] = series.astype(np.float32)
    return output


def compute_short_default_mechanism_context(
    *,
    asset_short_covering_score: pd.DataFrame,
    funding_1d_chg_ts_resid: pd.DataFrame,
    price_down_oi_down_4h_rz: pd.DataFrame,
    bars_per_hour: int = 1,
) -> dict[str, pd.Series]:
    """Return the two market-only fields used by the short-default mechanism.

    This allocation-conscious helper is mathematically identical to the
    corresponding section of :func:`add_negative_residual_features`, but it
    returns one market series per feature rather than broadcasting every value
    across the symbol panel.  It is intended for causal historical backfills.
    """

    index = asset_short_covering_score.index
    columns = asset_short_covering_score.columns
    short_cover = _market_series(
        asset_short_covering_score.reindex(index=index, columns=columns)
    )
    funding = _market_series(
        funding_1d_chg_ts_resid.reindex(index=index, columns=columns)
    )
    down_oi_down = _market_series(
        price_down_oi_down_4h_rz.reindex(index=index, columns=columns)
    )
    window = max(NEGATIVE_RESIDUAL_CAUSAL_WINDOW_HOURS * max(int(bars_per_hour), 1), 32)
    min_periods = max(NEGATIVE_RESIDUAL_MIN_HISTORY_HOURS * max(int(bars_per_hour), 1), 16)
    short_cover_state = _positive(_causal_robust_z(short_cover, window, min_periods))
    funding_z = _causal_robust_z(funding, window, min_periods)
    down_oi_down_z = _causal_robust_z(down_oi_down, window, min_periods)
    return {
        "short_covering_score_market": short_cover_state.astype(np.float32),
        "funding_confirmed_long_flush": np.arcsinh(
            _positive(funding_z) * _positive(down_oi_down_z)
        ).astype(np.float32),
    }


def add_residual_state_target_composites(
    frame: pd.DataFrame,
    *,
    timestamp_col: str = "__ts__",
    side_col: str = "side_name",
    archetype_col: str = "archetype_policy_key",
    horizons: Sequence[int] = RESIDUAL_STATE_TARGET_HORIZONS_HOURS,
) -> pd.DataFrame:
    """Attach realized residual-state labels for train/discovery only.

    These columns summarize *realized* model residuals.  They may define an
    economic state or train a pre-entry state predictor, but must never enter
    the inference feature matrix directly.  All names therefore use the
    reserved ``resid_target_`` prefix.
    """

    required = {
        timestamp_col,
        side_col,
        archetype_col,
        "resid_event_timestamp_neutral_surprise",
        "resid_event_ev_timestamp_neutral_surprise",
        "resid_event_daily_neutral_z",
        "resid_event_daily_ev_neutral_z",
        "resid_event_persistence_strength",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "residual-state target composites require realized columns: "
            + ", ".join(missing)
        )
    valid_horizons = tuple(
        dict.fromkeys(int(value) for value in horizons if int(value) > 0)
    )
    if not valid_horizons:
        return frame.copy(deep=False)

    out = frame.copy(deep=False)
    ts = pd.to_datetime(out[timestamp_col], utc=True, errors="coerce")
    work = pd.DataFrame(
        {
            "__row_position__": np.arange(len(out), dtype=np.int64),
            "__ts__": ts,
            "__side__": out[side_col].astype(str).str.lower(),
            "__archetype__": out[archetype_col].astype(str),
            "neutral": pd.to_numeric(
                out["resid_event_daily_neutral_z"], errors="coerce"
            ),
            "ev_neutral": pd.to_numeric(
                out["resid_event_daily_ev_neutral_z"], errors="coerce"
            ),
            "timestamp_neutral": pd.to_numeric(
                out["resid_event_timestamp_neutral_surprise"], errors="coerce"
            ),
            "ev_timestamp_neutral": pd.to_numeric(
                out["resid_event_ev_timestamp_neutral_surprise"], errors="coerce"
            ),
            "persistence": pd.to_numeric(
                out["resid_event_persistence_strength"], errors="coerce"
            ).fillna(0.0),
        }
    )
    if "adverse_tail_rows" in out.columns:
        work["adverse_tail_rows"] = pd.to_numeric(
            out["adverse_tail_rows"], errors="coerce"
        ).fillna(0.0)
    else:
        negative = pd.to_numeric(
            out.get("resid_event_negative_large", 0.0), errors="coerce"
        )
        top10 = pd.to_numeric(
            out.get("resid_event_top10_population", 0.0), errors="coerce"
        )
        if not isinstance(negative, pd.Series):
            negative = pd.Series(negative, index=out.index, dtype=np.float32)
        if not isinstance(top10, pd.Series):
            top10 = pd.Series(top10, index=out.index, dtype=np.float32)
        work["adverse_tail_rows"] = (
            negative.fillna(0.0).gt(0.5) & top10.fillna(0.0).gt(0.5)
        ).astype(np.float32)
    if "material_extreme" in out.columns:
        work["material_extreme"] = pd.to_numeric(
            out["material_extreme"], errors="coerce"
        ).fillna(0.0)
    else:
        large = pd.to_numeric(
            out.get("resid_event_large_event_strength", 0.0), errors="coerce"
        )
        if not isinstance(large, pd.Series):
            large = pd.Series(large, index=out.index, dtype=np.float32)
        work["material_extreme"] = (
            large.fillna(0.0).ge(1.0) | work["persistence"].ge(1.0)
        ).astype(np.float32)
    if "adverse_state_detected" in out.columns:
        work["adverse_state_detected"] = pd.to_numeric(
            out["adverse_state_detected"], errors="coerce"
        ).fillna(0.0)
    else:
        work["adverse_state_detected"] = work["adverse_tail_rows"].gt(0.0).astype(
            np.float32
        )
    work = work.loc[work["__ts__"].notna()]

    mean_columns = [
        "neutral",
        "ev_neutral",
        "timestamp_neutral",
        "ev_timestamp_neutral",
    ]
    max_columns = ["persistence", "material_extreme", "adverse_state_detected"]
    for scope, group_columns in (
        ("side", ["__side__"]),
        ("side_archetype", ["__side__", "__archetype__"]),
    ):
        aggregate = (
            work.groupby([*group_columns, "__ts__"], observed=True, sort=True)
            .agg(
                **{name: (name, "mean") for name in mean_columns},
                **{name: (name, "max") for name in max_columns},
                adverse_tail_rows=("adverse_tail_rows", "sum"),
            )
            .reset_index()
        )
        pieces: list[pd.DataFrame] = []
        for group_key, local in aggregate.groupby(
            group_columns, observed=True, sort=False
        ):
            local = local.sort_values("__ts__", kind="stable").set_index("__ts__")
            keys = group_key if isinstance(group_key, tuple) else (group_key,)
            group_values = pd.DataFrame(index=local.index)
            for horizon in valid_horizons:
                rolling = local[
                    [*mean_columns, *max_columns, "adverse_tail_rows"]
                ].rolling(f"{int(horizon)}h", min_periods=1, closed="both")
                means = rolling[mean_columns].mean()
                maxima = rolling[max_columns].max()
                tail_rows = rolling["adverse_tail_rows"].sum()
                signs_differ = np.sign(means["timestamp_neutral"]).ne(
                    np.sign(means["ev_timestamp_neutral"])
                )
                tail_absence = 1.0 / (1.0 + tail_rows.clip(lower=0.0))
                group_values[
                    f"resid_target_{scope}_directional_ev_divergence_{horizon}h"
                ] = means["neutral"] - means["ev_neutral"]
                group_values[
                    f"resid_target_{scope}_bullish_tape_adverse_ev_{horizon}h"
                ] = means["neutral"].clip(lower=0.0) * (
                    -means["ev_neutral"]
                ).clip(lower=0.0)
                group_values[
                    f"resid_target_{scope}_timestamp_ev_sign_disagreement_{horizon}h"
                ] = (
                    means["timestamp_neutral"].abs()
                    * means["ev_timestamp_neutral"].abs()
                    * signs_differ.astype(np.float32)
                )
                group_values[
                    f"resid_target_{scope}_positive_timestamp_negative_ev_{horizon}h"
                ] = means["timestamp_neutral"].clip(lower=0.0) * (
                    -means["ev_timestamp_neutral"]
                ).clip(lower=0.0)
                group_values[
                    f"resid_target_{scope}_tail_absence_factor_{horizon}h"
                ] = tail_absence
                group_values[
                    f"resid_target_{scope}_persistent_subthreshold_damage_{horizon}h"
                ] = (
                    maxima["persistence"]
                    * (-means["ev_neutral"]).clip(lower=0.0)
                    * tail_absence
                )
                group_values[
                    f"resid_target_{scope}_persistent_material_nontail_{horizon}h"
                ] = (
                    np.log1p(maxima["persistence"].clip(lower=0.0))
                    * maxima["material_extreme"].gt(0.5).astype(np.float32)
                    * maxima["adverse_state_detected"].le(0.5).astype(np.float32)
                )
            group_values = group_values.reset_index()
            for name, value in zip(group_columns, keys, strict=True):
                group_values[name] = value
            pieces.append(group_values)
        if not pieces:
            continue
        attach = pd.concat(pieces, ignore_index=True)
        attach_columns = [
            name
            for name in attach.columns
            if name.startswith(f"resid_target_{scope}_")
        ]
        work = work.merge(
            attach[[*group_columns, "__ts__", *attach_columns]],
            on=[*group_columns, "__ts__"],
            how="left",
            sort=False,
            validate="many_to_one",
        )
    target_columns = residual_state_target_feature_names(valid_horizons)
    values = work.set_index("__row_position__").reindex(np.arange(len(out)))
    result = out.copy()
    for name in target_columns:
        result[name] = pd.to_numeric(values.get(name), errors="coerce").astype(
            np.float32
        )
    return result


def add_negative_residual_features(
    features: dict[str, pd.DataFrame],
    *,
    requested_feature_keys: Sequence[str] | None = None,
    cfg: Mapping[str, object] | None = None,
) -> set[str]:
    """Materialize requested meta-only negative-residual state features."""
    requested = set(NEGATIVE_RESIDUAL_META_FEATURE_KEYS)
    if requested_feature_keys is not None:
        requested.intersection_update(map(str, requested_feature_keys))
    if not requested or not features:
        return set()
    exemplar = next((value for value in features.values() if isinstance(value, pd.DataFrame)), None)
    if exemplar is None or exemplar.empty:
        return set()
    index, columns = exemplar.index, exemplar.columns
    cfg = cfg or {}
    bars_per_hour = max(int(cfg.get("feature_bars_per_hour", 1)), 1)
    window = max(NEGATIVE_RESIDUAL_CAUSAL_WINDOW_HOURS * bars_per_hour, 32)
    min_periods = max(NEGATIVE_RESIDUAL_MIN_HISTORY_HOURS * bars_per_hour, 16)
    generated: set[str] = set()

    def frame(name: str) -> pd.DataFrame:
        return _aligned(features, name, index, columns)

    def z(series: pd.Series) -> pd.Series:
        return _causal_robust_z(series, window, min_periods)

    def put(name: str, value: pd.Series) -> None:
        if name in requested:
            if name in NEGATIVE_RESIDUAL_COMPOSITE_FEATURE_KEYS:
                # Preserve ordering/sign without allowing two- and three-way
                # z-products to dominate tree splits solely by scale.
                value = np.arcsinh(value).astype(np.float32)
            features[name] = _broadcast(
                value.replace([np.inf, -np.inf], np.nan).astype(np.float32), columns
            )
            generated.add(name)

    market_state_values: dict[str, pd.Series] = {}

    def state(name: str, value: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(value, errors="coerce").astype(np.float32)
        market_state_values[name] = numeric
        return numeric

    def add_transition_family() -> None:
        for name, output in calculate_market_regime_change_series(
            market_state_values, bars_per_hour=bars_per_hour
        ).items():
            put(name, output)

    ret4 = frame("ret4h")
    corr_eth = frame("corr_eth_24h")
    corr_btc = frame("corr_btc_24h")
    median_corr_eth = _market_series(corr_eth)
    q10_corr_btc = _market_series(corr_btc, statistic="q10")
    std_corr_eth = _market_series(corr_eth, statistic="std")
    dispersion = _market_series(frame("market_dispersion_4h"))
    btc = _symbol(columns, "BTC")
    eth = _symbol(columns, "ETH")
    alt_columns = [column for column in columns if column not in {btc, eth}]
    alt_ret = ret4[alt_columns] if alt_columns else ret4
    median_alt_ret = _market_series(alt_ret)
    btc_ret = (
        ret4[btc].astype(np.float32)
        if btc is not None
        else pd.Series(np.nan, index=index, dtype=np.float32)
    )

    ret4_values = ret4.to_numpy(dtype=np.float32, copy=False)
    valid_ret4 = np.sum(np.isfinite(ret4_values), axis=1)
    negative_breadth_values = np.divide(
        np.sum((ret4_values < 0) & np.isfinite(ret4_values), axis=1),
        np.maximum(valid_ret4, 1),
    ).astype(np.float32)
    negative_breadth_values[valid_ret4 < NEGATIVE_RESIDUAL_MIN_MARKET_ASSETS] = np.nan
    negative_breadth = pd.Series(
        negative_breadth_values,
        index=index,
        dtype=np.float32,
    )
    residual = frame("ret_resid_btc_4h")
    residual_q10 = residual.shift(1).rolling(window, min_periods=min_periods).quantile(0.10)
    extreme_negative_breadth = residual.lt(residual_q10).mean(axis=1).astype(np.float32)
    extreme_valid = residual.notna().sum(axis=1)
    extreme_negative_breadth = extreme_negative_breadth.where(
        extreme_valid >= NEGATIVE_RESIDUAL_MIN_MARKET_ASSETS
    )
    breadth_iqr = _market_series(ret4, statistic="iqr")
    downside_values = ret4.where(ret4.lt(0.0))
    downside_dispersion = _market_series(downside_values, statistic="iqr")
    state("negative_breadth", negative_breadth)
    put("negative_breadth_pct", negative_breadth)
    put("extreme_negative_breadth_pct", extreme_negative_breadth)
    median_alt_minus_btc = state(
        "btc_alt_relative_strength", median_alt_ret - btc_ret
    )
    put("median_alt_minus_btc", median_alt_minus_btc)
    put("breadth_dispersion", breadth_iqr)
    put("downside_breadth_intensity", negative_breadth * downside_dispersion)
    put(
        "btc_resilience_alt_weakness",
        _positive(z(btc_ret)) * _positive(-z(median_alt_ret)),
    )
    oi_z = frame("oi_value_z_30d")
    btc_oi = oi_z[btc] if btc is not None else pd.Series(np.nan, index=index)
    eth_oi = oi_z[eth] if eth is not None else pd.Series(np.nan, index=index)
    alt_oi_columns = [column for column in columns if column not in {btc, eth}]
    alt_oi = _market_series(oi_z[alt_oi_columns]) if alt_oi_columns else _market_series(oi_z)
    non_btc_oi = _market_series(oi_z[[column for column in columns if column != btc]])
    put(
        "btc_oi_dominance_z_ratio",
        np.arcsinh(btc_oi / (non_btc_oi.abs() + np.float32(0.25))),
    )
    put(
        "btc_ex_eth_oi_dominance_z_ratio",
        np.arcsinh(btc_oi / (alt_oi.abs() + np.float32(0.25))),
    )
    oi_change_z = frame("oi_value_1d_chg_z_90d")
    btc_dominance_momentum = (
        oi_change_z[btc] if btc is not None else pd.Series(np.nan, index=index)
    )
    eth_dominance_momentum = (
        oi_change_z[eth] if eth is not None else pd.Series(np.nan, index=index)
    )
    put(
        "btc_over_eth_dominance_roc",
        (btc_dominance_momentum - eth_dominance_momentum).clip(-5.0, 5.0),
    )
    peer = _market_series(frame("symbol_minus_mkt_ret_1h"), statistic="iqr")
    put("peer_decoupling_acceleration", peer.diff().abs() * dispersion)
    short_cover = _market_series(frame("asset_short_covering_score"))
    funding = _market_series(frame("funding_1d_chg_ts_resid"))
    short_covering_state = state("short_covering", _positive(z(short_cover)))
    put("short_covering_score_market", short_covering_state)

    z_corr_eth = z(median_corr_eth)
    z_corr_btc_q10 = z(q10_corr_btc)
    z_corr_eth_std = z(std_corr_eth)
    z_dispersion = z(dispersion)
    z_new_lows = z(_market_series(frame("pct_assets_new_low_24h")))
    z_recovery = z(_market_series(frame("price_recovery_from_low_24h_atr")))
    z_oi_drawdown = z(_market_series(frame("mkt_median_oi_drawdown_from_peak_24h")))
    z_oi_contraction = z(-_market_series(frame("mkt_median_oi_chg_4h_rz")))
    z_oi_breadth = z(_market_series(frame("mkt_pct_oi_chg_4h_rz_lt_minus1")))
    z_funding = z(funding)
    z_down_oi_down = z(_market_series(frame("price_down_oi_down_4h_rz")))
    z_price_up_oi_up = z(_market_series(frame("mkt_pct_price_up_oi_up_4h")))
    z_flush = z(_market_series(frame("mkt_flush_exhaustion_score")))
    z_breadth_recovery = z(_market_series(frame("breadth_recovery_from_6h_min")))
    z_range_decay = z(_market_series(frame("range_climax_decay_4h")))
    z_compression = z(_market_series(frame("log_realized_vol_cp_absratio_8_32")))
    z_peer_vol = z(_market_series(frame("rv_24h_peer_resid")))
    z_range_volume = z(_market_series(frame("range_per_volume")))
    z_market_ret = z(_market_series(frame("mkt_ret_4h")))
    market_ret_4h = _market_series(frame("mkt_ret_4h"))
    market_ret_1h = _market_series(frame("mkt_ret_1h"))
    z_downside_corr = z(_market_series(frame("market_downside_pairwise_corr_24h")))

    state("funding", funding)
    state("oi_contraction", -_market_series(frame("mkt_median_oi_chg_4h_rz")))
    state("eth_correlation", median_corr_eth)

    put("correlation_breakdown_dispersion", _positive(-z_corr_eth) * _positive(z_dispersion))
    put("btc_decoupling_dispersion", _positive(-z_corr_btc_q10) * _positive(z_dispersion))
    put("correlation_heterogeneity_dispersion", _positive(z_corr_eth_std) * _positive(z_dispersion))
    put("fragmented_new_low_breadth", _positive(z_new_lows) * _positive(-z_corr_btc_q10))
    put("deleveraging_without_followthrough", _positive(-z_oi_drawdown) * _positive(z_recovery))
    put("short_breakout_exhaustion", _positive(z_oi_contraction) * _positive(z_recovery) * _positive(-z_corr_eth))
    put("funding_deleveraging_divergence", _positive(-z_funding) * _positive(z_oi_breadth))
    put("funding_confirmed_short_covering", _positive(-z_funding) * _positive(z(short_cover)))
    put("funding_confirmed_long_flush", _positive(z_funding) * _positive(z_down_oi_down))
    put("post_flush_leverage_rebuild", _positive(z_flush) * _positive(z_price_up_oi_up))
    put("fragile_leverage_rebuild", _positive(z_price_up_oi_up) * _positive(-z_corr_eth) * _positive(z_dispersion))
    flush_recovery_raw = _positive(z_flush) * _positive(z_breadth_recovery)
    state(
        "flush_recovery",
        np.arcsinh(flush_recovery_raw).astype(np.float32),
    )
    put("flush_recovery_state", flush_recovery_raw)
    put("fragmented_flush_recovery", _positive(z_flush) * _positive(z_breadth_recovery) * _positive(-z_corr_eth))
    put("broad_washout_recovery", _positive(z_new_lows) * _positive(z_recovery))
    put("range_climax_reversal", _positive(z_range_decay) * _positive(z_recovery))
    put("deleveraged_range_climax_reversal", _positive(z_range_decay) * _positive(z_recovery) * _positive(z_oi_contraction))
    put("compressed_index_fragmented_assets", _positive(-z_compression) * _positive(z_dispersion))
    put("peer_volatility_decoupling", _positive(z_peer_vol) * _positive(-z_corr_eth))
    put("thin_compression", _positive(z_range_volume) * _positive(-z_compression))
    put("unconfirmed_long_breakout", _positive(z_market_ret) * _positive(-z_corr_btc_q10) * _positive(z_dispersion))
    put("leveraged_long_breakout_risk", _positive(z_price_up_oi_up) * _positive(z_funding))
    put("false_clean_short", _positive(-z_market_ret) * _positive(-z_downside_corr) * _positive(z_dispersion))
    put("short_signal_recovery_conflict", _positive(-z_market_ret) * _positive(z_recovery) * _positive(z_breadth_recovery))
    put("late_short_after_deleveraging", _positive(-z_market_ret) * _positive(-z_oi_drawdown) * _positive(z_recovery))

    # Compression quality: genuine compression requires agreement between
    # volatility, cross-sectional dispersion, correlation, positioning, funding,
    # and the range-per-volume liquidity proxy.  The proxy is used because this
    # feature family intentionally has no order-book/spread dependency.
    compression_intensity = _positive(-z_compression).clip(0.0, 5.0)
    oi_stability = np.exp(-z_oi_contraction.abs()).astype(np.float32)
    funding_stability = np.exp(-z_funding.abs()).astype(np.float32)
    liquidity_stability = np.exp(-_positive(z_range_volume)).astype(np.float32)
    compression_components = [
        (compression_intensity / np.float32(5.0)).clip(0.0, 1.0),
        (_positive(-z_dispersion) / np.float32(5.0)).clip(0.0, 1.0),
        (_positive(z_corr_eth) / np.float32(5.0)).clip(0.0, 1.0),
        oi_stability,
        funding_stability,
        liquidity_stability,
    ]
    compression_consistency = (1.0 / (1.0 + _row_std(compression_components))).astype(
        np.float32
    )
    compression_confirmation = _row_mean(
        [
            z_compression.le(0.0).astype(np.float32),
            z_dispersion.le(0.0).astype(np.float32),
            z_corr_eth.ge(0.0).astype(np.float32),
            z_oi_contraction.abs().le(1.0).astype(np.float32),
            z_funding.abs().le(1.0).astype(np.float32),
            z_range_volume.le(0.0).astype(np.float32),
        ]
    ).clip(0.0, 1.0)
    compression_dynamic_instability = _row_mean(
        [
            (_positive(z_dispersion) / np.float32(5.0)).clip(0.0, 1.0),
            (z(median_corr_eth.diff()).abs() / np.float32(5.0)).clip(0.0, 1.0),
            (z((-_market_series(frame("mkt_median_oi_chg_4h_rz"))).diff()).abs() / np.float32(5.0)).clip(0.0, 1.0),
            (z(funding.diff()).abs() / np.float32(5.0)).clip(0.0, 1.0),
            (_positive(z_range_volume) / np.float32(5.0)).clip(0.0, 1.0),
        ]
    ).clip(0.0, 1.0)
    put("compression_quality_consistency", compression_consistency)
    put("compression_confirmation_ratio", compression_confirmation)
    put(
        "healthy_compression_score",
        (compression_intensity * compression_confirmation * compression_consistency).clip(0.0, 5.0),
    )
    exhausted_confirmation = _row_mean(
        [
            (_positive(z_recovery) / np.float32(5.0)).clip(0.0, 1.0),
            (_positive(z_breadth_recovery) / np.float32(5.0)).clip(0.0, 1.0),
            (_positive(-z_oi_contraction) / np.float32(5.0)).clip(0.0, 1.0),
            (_positive(z_funding) / np.float32(5.0)).clip(0.0, 1.0),
        ]
    )
    put(
        "exhausted_compression_score",
        (compression_intensity * exhausted_confirmation).clip(0.0, 5.0),
    )
    put(
        "fragile_compression_score",
        (
            compression_intensity
            * (1.0 - compression_confirmation)
            * compression_dynamic_instability
        ).clip(0.0, 5.0),
    )
    compression_active = z_compression.le(-0.5)
    put(
        "compression_duration_72h_norm",
        _consecutive_fraction(compression_active, maximum_bars=72 * bars_per_hour),
    )
    put(
        "compression_integral_72h",
        compression_intensity.rolling(
            72 * bars_per_hour,
            min_periods=12 * bars_per_hour,
        ).mean().clip(0.0, 5.0),
    )
    put(
        "compression_onset_shock_24h",
        _positive(
            compression_intensity
            - compression_intensity.shift(1).rolling(
                24 * bars_per_hour,
                min_periods=6 * bars_per_hour,
            ).mean()
        ).clip(0.0, 5.0),
    )

    # Persistent short-default pressure.  These are observable market-state
    # integrals, not lagged realized model errors.
    short_damage = _row_mean(
        [
            (_positive(-z_corr_eth) / np.float32(5.0)).clip(0.0, 1.0),
            (_positive(z_dispersion) / np.float32(5.0)).clip(0.0, 1.0),
            (_positive(z_recovery) / np.float32(5.0)).clip(0.0, 1.0),
            (_positive(z_oi_contraction) / np.float32(5.0)).clip(0.0, 1.0),
            (_positive(-z(negative_breadth)) / np.float32(5.0)).clip(0.0, 1.0),
        ]
    ).clip(0.0, 1.0)
    five_days = 5 * 24 * bars_per_hour
    put("short_default_damage_pressure", short_damage)
    put(
        "short_default_damage_ema_5d",
        short_damage.ewm(span=five_days, adjust=False, min_periods=24 * bars_per_hour).mean(),
    )
    put(
        "short_default_damage_integral_5d",
        short_damage.rolling(five_days, min_periods=24 * bars_per_hour).mean(),
    )
    put(
        "short_default_damage_max_5d",
        short_damage.rolling(five_days, min_periods=24 * bars_per_hour).max(),
    )
    damage_active = z(short_damage).ge(0.75)
    put(
        "short_default_adverse_duration_5d_norm",
        _consecutive_fraction(damage_active, maximum_bars=five_days),
    )
    state_signals = [
        np.sign(funding).astype(np.float32),
        np.sign(-_market_series(frame("mkt_median_oi_chg_4h_rz"))).astype(np.float32),
        np.sign(negative_breadth - np.float32(0.5)).astype(np.float32),
        np.sign(median_corr_eth.diff()).astype(np.float32),
    ]
    transition_rate = _row_mean(
        [signal.ne(signal.shift(1)).astype(np.float32) for signal in state_signals]
    ).rolling(five_days, min_periods=24 * bars_per_hour).mean().clip(0.0, 1.0)
    entropy = -(
        transition_rate.clip(1e-6, 1.0 - 1e-6) * np.log2(
            transition_rate.clip(1e-6, 1.0 - 1e-6)
        )
        + (1.0 - transition_rate).clip(1e-6, 1.0 - 1e-6) * np.log2(
            (1.0 - transition_rate).clip(1e-6, 1.0 - 1e-6)
        )
    ).astype(np.float32)
    put("market_state_transition_entropy_5d", entropy.clip(0.0, 1.0))
    put("market_state_persistence_5d", (1.0 - transition_rate).clip(0.0, 1.0))
    prior_recovery = _positive(z_recovery).shift(1).rolling(
        24 * bars_per_hour,
        min_periods=4 * bars_per_hour,
    ).max()
    put(
        "recovery_failure_score_24h",
        (
            prior_recovery
            * _positive(-z_market_ret)
            * _positive(-z_breadth_recovery)
        ).clip(0.0, 25.0),
    )

    # Breakout quality is intentionally direction agnostic: June 17 hurt both
    # long and short breakout archetypes.  Retention only evaluates a breakout
    # already observed four hours earlier.
    path_length_4h = market_ret_1h.abs().rolling(
        4 * bars_per_hour,
        min_periods=4 * bars_per_hour,
    ).sum()
    breakout_efficiency = market_ret_4h.abs().div(path_length_4h + np.float32(1e-6)).clip(0.0, 1.0)
    positive_breadth = (1.0 - negative_breadth).clip(0.0, 1.0)
    breakout_participation = pd.Series(
        np.where(market_ret_4h.ge(0.0), positive_breadth, negative_breadth),
        index=index,
        dtype=np.float32,
    ).clip(0.0, 1.0)
    prior_move = market_ret_4h.shift(4 * bars_per_hour)
    breakout_retention = (
        np.sign(prior_move)
        * market_ret_4h
        / (prior_move.abs() + np.float32(1e-6))
    ).clip(-1.0, 1.0)
    direction = np.sign(market_ret_4h)
    oi_confirmation = pd.Series(
        np.where(direction.ge(0.0), z_price_up_oi_up.ge(0.0), z_down_oi_down.ge(0.0)),
        index=index,
        dtype=np.float32,
    )
    funding_confirmation = (direction * np.sign(funding)).ge(0.0).astype(np.float32)
    breakout_confirmation = _row_mean(
        [
            breakout_efficiency.ge(0.5).astype(np.float32),
            breakout_participation.ge(0.55).astype(np.float32),
            z_corr_eth.ge(0.0).astype(np.float32),
            oi_confirmation,
            funding_confirmation,
        ]
    ).clip(0.0, 1.0)
    put("breakout_efficiency_4h", breakout_efficiency)
    put("breakout_participation_4h", breakout_participation)
    put("breakout_retention_4h", breakout_retention)
    put("breakout_confirmation_ratio", breakout_confirmation)
    put(
        "breakout_disagreement_score",
        (z_market_ret.abs() * (1.0 - breakout_confirmation)).clip(0.0, 5.0),
    )
    put(
        "breakout_bilateral_failure_score",
        (
            z_market_ret.abs()
            * (1.0 - breakout_efficiency)
            * (1.0 - breakout_participation)
            * (1.0 - breakout_confirmation)
        ).clip(0.0, 5.0),
    )
    add_transition_family()
    return generated
