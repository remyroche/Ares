"""Causal OHLCV continuation-versus-exhaustion features.

This module deliberately contains no execution policy, labels, target
transforms, or order-book assertions.  It is a reusable feature generator for
the Stage-C ``P(retain | clear)`` research head.  Inputs are completed OHLCV
bars indexed by symbol and close timestamp.  Every rolling calculation is
trailing and includes, at most, the completed decision bar.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


CONTINUATION_PRICE_FEATURE_KEYS = [
    "cont_ret_1h", "cont_ret_4h", "cont_ret_12h", "cont_return_acceleration_1h_4h",
    "cont_efficiency_12h", "cont_directional_consistency_12h", "cont_slope_12h",
    "cont_slope_r2_12h", "cont_distance_from_high_12h", "cont_distance_from_low_12h",
    "cont_high_recency_12h", "cont_low_recency_12h", "cont_direction_changes_12h",
    "cont_close_location_ohlcv_proxy", "cont_side_wick_imbalance_raw",
    "cont_range_expansion_12h", "cont_failed_up_breakout_count_12h",
    "cont_failed_down_breakout_count_12h", "cont_up_breakout_rejection_12h",
    "cont_down_breakout_rejection_12h",
]
CONTINUATION_VOLUME_FEATURE_KEYS = [
    "cont_volume_z_48h", "cont_volume_persistence_12h", "cont_signed_volume_proxy_12h",
    "cont_range_to_volume_ohlcv_proxy", "cont_high_volume_low_return_churn_12h",
    "cont_volume_price_corr_12h", "cont_volume_concentration_4h_12h",
    "cont_volume_shock_age_hrs", "cont_volume_shock_decay_12h",
]
CONTINUATION_VOLATILITY_FEATURE_KEYS = [
    "cont_rv_12h", "cont_downside_rv_12h", "cont_vol_ratio_4h_12h",
    "cont_vol_of_vol_12h", "cont_range_z_48h", "cont_squared_return_autocorr_12h",
    "cont_atr_slope_12h", "cont_atr_acceleration_12h", "cont_vol_shock_age_hrs",
    "cont_vol_shock_decay_12h", "cont_vol_climax_persistence_12h",
]
CONTINUATION_CROSS_SECTIONAL_FEATURE_KEYS = [
    "cont_cs_universe_size", "cont_cs_ret_rank_4h", "cont_cs_volume_rank_12h",
    "cont_market_median_ret_4h", "cont_market_ret_breadth_4h",
    "cont_asset_minus_market_ret_4h", "cont_market_ret_dispersion_4h",
    "cont_cs_volume_dispersion_12h", "cont_cs_breakout_confirmation",
    "cont_cs_isolated_move_4h",
]
CONTINUATION_COMPOSITE_FEATURE_KEYS = [
    "cont_efficiency_x_volume_persistence", "cont_breakout_x_breadth_confirmation",
    "cont_volatility_x_efficiency", "cont_volume_climax_x_low_efficiency",
    "cont_market_confirmation_x_volume_persistence",
]
CONTINUATION_SIDE_PRICE_FEATURE_KEYS = [
    "side_cont_ret_1h", "side_cont_ret_4h", "side_cont_ret_12h",
    "side_cont_return_acceleration_1h_4h", "side_cont_slope_12h",
    "side_cont_asset_minus_market_ret_4h", "side_cont_wick_imbalance",
    "side_cont_breakout_rejection",
]
CONTINUATION_SIDE_VOLATILITY_FEATURE_KEYS = ["side_cont_adverse_rv_12h"]
# F4/F5 and F7 are registered deliberately as empty research groups until a
# source-side observed/available timestamp contract (or strict OOF provenance)
# is materialised.  Empty is safer than accepting the historical unbounded
# forward-filled sidecars or raw learned regime fields.
CONTINUATION_OI_FEATURE_KEYS: list[str] = []
CONTINUATION_FUNDING_FEATURE_KEYS: list[str] = []
CONTINUATION_REGIME_FEATURE_KEYS: list[str] = []
CONTINUATION_FEATURE_GROUPS = {
    "F1_price_continuation_exhaustion": CONTINUATION_PRICE_FEATURE_KEYS,
    "F2_volume_liquidity_proxies": CONTINUATION_VOLUME_FEATURE_KEYS,
    "F3_volatility_transition": CONTINUATION_VOLATILITY_FEATURE_KEYS,
    "F4_oi_dynamics": CONTINUATION_OI_FEATURE_KEYS,
    "F5_funding_crowding": CONTINUATION_FUNDING_FEATURE_KEYS,
    "F6_cross_sectional_confirmation": CONTINUATION_CROSS_SECTIONAL_FEATURE_KEYS,
    "F7_causal_regime_transition": CONTINUATION_REGIME_FEATURE_KEYS,
    "F8_predeclared_composites": CONTINUATION_COMPOSITE_FEATURE_KEYS,
}


def _safe_divide(numerator: pd.Series, denominator: pd.Series, floor: float = 1e-12) -> pd.Series:
    return numerator / denominator.abs().clip(lower=floor)


def _group_rolling(series: pd.Series, symbols: pd.Series, window: int, operation: str) -> pd.Series:
    """Fast grouped trailing rolling operation retaining the original index."""
    grouped = series.groupby(symbols, observed=True)
    rolling = grouped.rolling(window, min_periods=max(2, min(window, 3)))
    result = getattr(rolling, operation)().reset_index(level=0, drop=True)
    return result.reindex(series.index)


def _group_slope_and_r2(log_close: pd.Series, symbols: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    """Trailing OLS slope/R² via rolling sufficient statistics (no Python loops)."""
    position = log_close.groupby(symbols, observed=True).cumcount().astype(float)
    y = log_close.astype(float)
    valid = y.notna().astype(float)
    x = position.where(y.notna(), 0.0)
    y0 = y.fillna(0.0)
    n = _group_rolling(valid, symbols, window, "sum")
    sx = _group_rolling(x, symbols, window, "sum")
    sy = _group_rolling(y0, symbols, window, "sum")
    sxx = _group_rolling(x * x, symbols, window, "sum")
    syy = _group_rolling(y0 * y0, symbols, window, "sum")
    sxy = _group_rolling(x * y0, symbols, window, "sum")
    denominator_x = n * sxx - sx * sx
    denominator_y = n * syy - sy * sy
    numerator = n * sxy - sx * sy
    slope = (numerator / denominator_x.replace(0.0, np.nan)).where(n >= window)
    r2 = ((numerator * numerator) / (denominator_x * denominator_y).replace(0.0, np.nan)).where(n >= window)
    return slope.astype("float32"), r2.clip(0.0, 1.0).astype("float32")


def _group_event_age(event: pd.Series, symbols: pd.Series) -> pd.Series:
    """Trailing bars since the latest event, with no cross-symbol carry."""
    position = event.groupby(symbols, observed=True).cumcount().astype("float64")
    latest = position.where(event.fillna(False)).groupby(symbols, observed=True).ffill()
    return (position - latest).where(latest.notna())


def _group_extreme_recency(series: pd.Series, symbols: pd.Series, window: int, *, extreme: str) -> pd.Series:
    """Trailing bars since the most recent high or low inside a full window."""
    if extreme not in {"max", "min"}:
        raise ValueError(f"unsupported extreme {extreme!r}")
    reducer = np.argmax if extreme == "max" else np.argmin
    return series.groupby(symbols, observed=True).transform(
        lambda values: values.rolling(window, min_periods=window).apply(
            # Reverse first so tied highs/lows resolve to the most recent bar.
            lambda window_values: float(reducer(window_values[::-1])), raw=True
        )
    )


def materialize_ohlcv_continuation_features(
    bars: pd.DataFrame,
    *,
    timestamp_column: str = "ts",
    symbol_column: str = "symbol",
) -> pd.DataFrame:
    """Return decision-time features from a long, completed hourly OHLCV panel.

    ``bars`` may include unrelated columns.  It must contain OHLCV values and
    is sorted locally by symbol/timestamp.  Cross-sectional values are computed
    only from rows present at the same timestamp, so a delisted/missing symbol
    cannot enter another symbol's statistic.
    """
    required = {timestamp_column, symbol_column, "open", "high", "low", "close", "volume"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"OHLCV continuation features require {sorted(missing)}")
    frame = bars.loc[:, list(required)].copy()
    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], utc=True, errors="raise")
    frame = frame.sort_values([symbol_column, timestamp_column], kind="stable").drop_duplicates([symbol_column, timestamp_column], keep="last").reset_index(drop=True)
    symbol = frame[symbol_column]
    close = pd.to_numeric(frame.close, errors="coerce").where(lambda s: s > 0.0)
    high = pd.to_numeric(frame.high, errors="coerce").where(lambda s: s > 0.0)
    low = pd.to_numeric(frame.low, errors="coerce").where(lambda s: s > 0.0)
    opening = pd.to_numeric(frame.open, errors="coerce").where(lambda s: s > 0.0)
    volume = pd.to_numeric(frame.volume, errors="coerce").where(lambda s: s >= 0.0)
    log_close = np.log(close)
    ret_1 = log_close.groupby(symbol, observed=True).diff(1)
    ret_4 = log_close.groupby(symbol, observed=True).diff(4)
    ret_12 = log_close.groupby(symbol, observed=True).diff(12)
    abs_path_12 = _group_rolling(ret_1.abs(), symbol, 12, "sum")
    high_12 = _group_rolling(high, symbol, 12, "max")
    low_12 = _group_rolling(low, symbol, 12, "min")
    bar_range = (high - low).clip(lower=0.0)
    mean_range_12 = _group_rolling(bar_range, symbol, 12, "mean")
    atr_slope_12 = mean_range_12 - _group_rolling(mean_range_12, symbol, 12, "mean")
    atr_acceleration_12 = atr_slope_12 - atr_slope_12.groupby(symbol, observed=True).shift(4)
    volume_mean_48 = _group_rolling(volume, symbol, 48, "mean")
    volume_std_48 = _group_rolling(volume, symbol, 48, "std")
    volume_mean_12 = _group_rolling(volume, symbol, 12, "mean")
    slope_12, slope_r2_12 = _group_slope_and_r2(log_close, symbol, 12)
    positive_fraction = _group_rolling((ret_1 > 0.0).astype(float), symbol, 12, "mean")
    signed_volume = np.sign(close - opening).fillna(0.0) * volume.fillna(0.0)
    return_abs = (close / opening - 1.0).abs()
    range_scale = _safe_divide(bar_range, close)
    rv_4 = np.sqrt(_group_rolling(ret_1 * ret_1, symbol, 4, "mean"))
    rv_12 = np.sqrt(_group_rolling(ret_1 * ret_1, symbol, 12, "mean"))
    downside_rv = np.sqrt(_group_rolling(ret_1.clip(upper=0.0) ** 2, symbol, 12, "mean"))
    range_z = _safe_divide(bar_range - _group_rolling(bar_range, symbol, 48, "mean"), _group_rolling(bar_range, symbol, 48, "std"))
    squared_autocorr = (ret_1 * ret_1).groupby(symbol, observed=True).transform(lambda s: s.rolling(12, min_periods=6).corr(s.shift(1)))
    direction_changes = ((np.sign(ret_1) != np.sign(ret_1.groupby(symbol, observed=True).shift(1))) & ret_1.ne(0.0)).astype(float)
    upper_wick = (high - pd.concat([opening, close], axis=1).max(axis=1)).clip(lower=0.0)
    lower_wick = (pd.concat([opening, close], axis=1).min(axis=1) - low).clip(lower=0.0)
    high_recency = _group_extreme_recency(high, symbol, 12, extreme="max")
    low_recency = _group_extreme_recency(low, symbol, 12, extreme="min")
    prior_high = high_12.groupby(symbol, observed=True).shift(1)
    prior_low = low_12.groupby(symbol, observed=True).shift(1)
    failed_up_breakout = ((high.gt(prior_high)) & close.lt(prior_high)).fillna(False)
    failed_down_breakout = ((low.lt(prior_low)) & close.gt(prior_low)).fillna(False)
    volume_shock = (volume > (volume_mean_48 + 2.0 * volume_std_48)).fillna(False)
    # A missing event is meaningful rather than a missing measurement: after
    # the 48 completed-bar warm-up, encode it as "no shock in the observable
    # 48h history".  Leaving it null would discard otherwise complete
    # candidates from every common-cohort comparison.
    volume_shock_age = _group_event_age(volume_shock, symbol).fillna(49.0)
    volatility_shock = (range_z > 2.0).fillna(False)
    volatility_shock_age = _group_event_age(volatility_shock, symbol).fillna(49.0)
    result = frame.loc[:, [timestamp_column, symbol_column]].copy()
    result["cont_ret_1h"] = ret_1
    result["cont_ret_4h"] = ret_4
    result["cont_ret_12h"] = ret_12
    result["cont_return_acceleration_1h_4h"] = ret_1 - ret_4 / 4.0
    result["cont_efficiency_12h"] = _safe_divide(ret_12.abs(), abs_path_12)
    result["cont_directional_consistency_12h"] = (2.0 * positive_fraction - 1.0).abs()
    result["cont_slope_12h"] = slope_12
    result["cont_slope_r2_12h"] = slope_r2_12
    result["cont_distance_from_high_12h"] = _safe_divide(close - high_12, close)
    result["cont_distance_from_low_12h"] = _safe_divide(close - low_12, close)
    result["cont_high_recency_12h"] = high_recency
    result["cont_low_recency_12h"] = low_recency
    result["cont_direction_changes_12h"] = _group_rolling(direction_changes, symbol, 12, "sum")
    result["cont_close_location_ohlcv_proxy"] = _safe_divide(close - low, bar_range)
    result["cont_side_wick_imbalance_raw"] = _safe_divide(lower_wick - upper_wick, bar_range)
    result["cont_range_expansion_12h"] = _safe_divide(bar_range, mean_range_12)
    result["cont_failed_up_breakout_count_12h"] = _group_rolling(failed_up_breakout.astype(float), symbol, 12, "sum")
    result["cont_failed_down_breakout_count_12h"] = _group_rolling(failed_down_breakout.astype(float), symbol, 12, "sum")
    result["cont_up_breakout_rejection_12h"] = (failed_up_breakout & (result["cont_close_location_ohlcv_proxy"] < 0.35)).astype(float)
    result["cont_down_breakout_rejection_12h"] = (failed_down_breakout & (result["cont_close_location_ohlcv_proxy"] > 0.65)).astype(float)
    result["cont_volume_z_48h"] = _safe_divide(volume - volume_mean_48, volume_std_48)
    result["cont_volume_persistence_12h"] = _safe_divide(volume_mean_12, volume_mean_48)
    result["cont_signed_volume_proxy_12h"] = _group_rolling(signed_volume, symbol, 12, "sum")
    result["cont_range_to_volume_ohlcv_proxy"] = _safe_divide(range_scale, volume)
    result["cont_high_volume_low_return_churn_12h"] = _group_rolling((result["cont_volume_z_48h"].clip(lower=0.0) * (1.0 - return_abs.clip(upper=1.0))), symbol, 12, "mean")
    result["cont_volume_price_corr_12h"] = volume.groupby(symbol, observed=True).transform(lambda s: s.rolling(12, min_periods=6).corr(ret_1))
    result["cont_volume_concentration_4h_12h"] = _safe_divide(_group_rolling(volume, symbol, 4, "sum"), _group_rolling(volume, symbol, 12, "sum"))
    result["cont_volume_shock_age_hrs"] = volume_shock_age
    result["cont_volume_shock_decay_12h"] = np.exp(-volume_shock_age / 12.0)
    result["cont_rv_12h"] = rv_12
    result["cont_downside_rv_12h"] = downside_rv
    result["cont_vol_ratio_4h_12h"] = _safe_divide(rv_4, rv_12)
    result["cont_vol_of_vol_12h"] = _group_rolling(ret_1.abs(), symbol, 12, "std")
    result["cont_range_z_48h"] = range_z
    result["cont_squared_return_autocorr_12h"] = squared_autocorr
    result["cont_atr_slope_12h"] = atr_slope_12
    result["cont_atr_acceleration_12h"] = atr_acceleration_12
    result["cont_vol_shock_age_hrs"] = volatility_shock_age
    result["cont_vol_shock_decay_12h"] = np.exp(-volatility_shock_age / 12.0)
    result["cont_vol_climax_persistence_12h"] = _group_rolling((range_z > 1.5).astype(float), symbol, 12, "mean")
    # Each timestamp's eligible universe is exactly the rows present here.
    grouped_ts = result[timestamp_column]
    result["cont_cs_universe_size"] = result.groupby(grouped_ts, observed=True)[symbol_column].transform("size")
    result["cont_cs_ret_rank_4h"] = ret_4.groupby(grouped_ts, observed=True).rank(pct=True)
    result["cont_cs_volume_rank_12h"] = result["cont_volume_persistence_12h"].groupby(grouped_ts, observed=True).rank(pct=True)
    result["cont_market_median_ret_4h"] = ret_4.groupby(grouped_ts, observed=True).transform("median")
    result["cont_market_ret_breadth_4h"] = (ret_4 > 0.0).groupby(grouped_ts, observed=True).transform("mean")
    result["cont_asset_minus_market_ret_4h"] = ret_4 - result["cont_market_median_ret_4h"]
    result["cont_market_ret_dispersion_4h"] = ret_4.groupby(grouped_ts, observed=True).transform("std")
    result["cont_cs_volume_dispersion_12h"] = result["cont_volume_persistence_12h"].groupby(grouped_ts, observed=True).transform("std")
    raw_breakout_rejection = np.maximum(result["cont_up_breakout_rejection_12h"], result["cont_down_breakout_rejection_12h"])
    result["cont_cs_breakout_confirmation"] = (1.0 - raw_breakout_rejection) * result["cont_market_ret_breadth_4h"]
    result["cont_cs_isolated_move_4h"] = (result["cont_asset_minus_market_ret_4h"].abs() > result["cont_market_ret_dispersion_4h"]).astype(float)
    result["cont_efficiency_x_volume_persistence"] = result["cont_efficiency_12h"] * result["cont_volume_persistence_12h"]
    result["cont_breakout_x_breadth_confirmation"] = (1.0 - raw_breakout_rejection) * result["cont_market_ret_breadth_4h"]
    result["cont_volatility_x_efficiency"] = result["cont_vol_ratio_4h_12h"] * result["cont_efficiency_12h"]
    result["cont_volume_climax_x_low_efficiency"] = result["cont_volume_z_48h"].clip(lower=0.0) * (1.0 - result["cont_efficiency_12h"].clip(0.0, 1.0))
    result["cont_market_confirmation_x_volume_persistence"] = result["cont_market_ret_breadth_4h"] * result["cont_volume_persistence_12h"]
    return result.replace([np.inf, -np.inf], np.nan).astype({name: "float32" for name in result.columns if name not in {timestamp_column, symbol_column}})


def side_normalize_continuation_features(features: pd.DataFrame, sides: Iterable[str], *, side_column: str = "side") -> pd.DataFrame:
    """Add side-aligned price-shape fields after candidate-side expansion."""
    output = features.copy()
    sign = output[side_column].map({"long": 1.0, "short": -1.0})
    if sign.isna().any():
        raise ValueError("side normalisation requires only long/short rows")
    for name in ("cont_ret_1h", "cont_ret_4h", "cont_ret_12h", "cont_return_acceleration_1h_4h", "cont_slope_12h", "cont_asset_minus_market_ret_4h"):
        output[f"side_{name}"] = sign.to_numpy() * output[name].to_numpy(float)
    output["side_cont_wick_imbalance"] = sign.to_numpy() * output["cont_side_wick_imbalance_raw"].to_numpy(float)
    side_return_1h = output["side_cont_ret_1h"]
    symbol_key = "symbol" if "symbol" in output.columns else "source_symbol"
    side_groups = [output[symbol_key], output[side_column]]
    output["side_cont_adverse_rv_12h"] = np.sqrt(
        side_return_1h.clip(upper=0.0).pow(2).groupby(side_groups, observed=True).transform(lambda s: s.rolling(12, min_periods=3).mean())
    ).astype("float32")
    output["side_cont_breakout_rejection"] = np.where(
        sign.to_numpy() > 0.0,
        output["cont_up_breakout_rejection_12h"].to_numpy(float),
        output["cont_down_breakout_rejection_12h"].to_numpy(float),
    ).astype("float32")
    return output
