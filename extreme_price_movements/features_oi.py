"""Open-interest feature block for hourly perpetual futures panels."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit, prange

import extreme_price_movements.fast_funcs as ff

OI_INTERNAL_FEATURE_KEYS = [
    "oi_to_volume_1d",
    "oi_to_volume_7d",
    "oi_value_log",
    "oi_value_1d_log_chg",
    "oi_value_3d_log_chg",
    "oi_value_7d_log_chg",
    "price_1d_ret_z",
    "price_3d_ret_z",
    "price_7d_ret_z",
    "funding_z_90d",
    "oi_1d_chg_z_90d",
    "oi_3d_chg_z_90d",
    "oi_7d_chg_z_90d",
    "funding_1d_chg",
    "funding_1d_chg_z_90d",
]

OI_ALIAS_FEATURE_KEYS = [
    "oi_value_log_z_30d",
    "oi_value_log_z_90d",
    "oi_value_1d_log_chg_z_90d",
    "oi_value_3d_log_chg_z_90d",
    "oi_value_7d_log_chg_z_90d",
    "oi_value_7d_log_chg_z_180d",
]

OI_NORMALIZED_FEATURE_KEYS = [
    "oi_value_1d_chg_z_90d",
    "oi_value_3d_chg_z_90d",
    "oi_value_7d_chg_z_90d",
    "oi_value_7d_chg_z_180d",
    "oi_value_z_30d",
    "oi_value_z_90d",
    "oi_value_pct_90d",
    "oi_value_log_1d_robust_z",
    "oi_value_log_7d_robust_z",
    *OI_ALIAS_FEATURE_KEYS,
]

OI_CHANGE_POINT_FEATURE_KEYS = [
    "oi_value_log_cp_z_8_32_96",
    "oi_value_log_cp_logstd_8_32",
    "oi_value_log_cp_absratio_8_32",
    "oi_value_1d_log_chg_cp_z_8_32_96",
    "oi_value_1d_log_chg_cp_logstd_8_32",
    "oi_value_1d_log_chg_cp_absratio_8_32",
    "log_oi_to_volume_1d_cp_z_8_32_96",
    "log_oi_to_volume_1d_cp_logstd_8_32",
    "log_oi_to_volume_1d_cp_absratio_8_32",
    "asset_minus_mkt_oi_1d_cp_z_8_32_96",
    "asset_minus_mkt_oi_1d_cp_logstd_8_32",
    "asset_minus_mkt_oi_1d_cp_absratio_8_32",
]

MARKET_OI_REGIME_FEATURE_KEYS = [
    "mkt_oi_chg_15m",
    "mkt_oi_chg_1h",
    "mkt_oi_chg_4h",
    "mkt_oi_chg_24h",
    "mkt_oi_chg_accel_1h",
    "mkt_oi_chg_accel_4h",
    "mkt_oi_drawdown_from_24h_peak",
    "mkt_oi_drawdown_from_7d_peak",
    "mkt_oi_flush_z_30d",
    "pct_assets_oi_down_1h",
    "pct_assets_oi_down_4h",
    "pct_assets_oi_down_24h",
    "pct_assets_extreme_oi_drop_1h",
    "pct_assets_extreme_oi_drop_4h",
    "mkt_oi_dispersion_1h",
    "mkt_oi_dispersion_4h",
    "mkt_oi_concentration_btc_eth",
    "mkt_oi_recovery_from_24h_low",
    "bars_since_mkt_oi_trough",
]

PRICE_OI_STATE_FEATURE_KEYS = [
    "mkt_price_down_oi_down_1h",
    "mkt_price_down_oi_down_4h",
    "mkt_price_down_oi_up_1h",
    "mkt_price_down_oi_up_4h",
    "mkt_price_up_oi_down_1h",
    "mkt_price_up_oi_down_4h",
    "mkt_price_up_oi_up_1h",
    "mkt_price_up_oi_up_4h",
    "pct_assets_price_down_oi_down_1h",
    "pct_assets_price_down_oi_down_4h",
    "pct_assets_price_down_oi_up_1h",
    "pct_assets_price_up_oi_down_1h",
    "mkt_abs_ret_per_oi_drop_1h",
    "mkt_abs_ret_per_oi_drop_4h",
    "mkt_ret_per_oi_change_1h",
    "mkt_ret_per_oi_change_4h",
]

MARKET_FUNDING_REGIME_FEATURE_KEYS = [
    "mkt_funding_mean",
    "mkt_funding_median",
    "mkt_funding_abs_mean",
    "mkt_funding_weighted_by_oi",
    "mkt_funding_chg_1h",
    "mkt_funding_chg_4h",
    "mkt_funding_chg_24h",
    "mkt_funding_accel_4h",
    "pct_assets_positive_funding",
    "pct_assets_negative_funding",
    "pct_assets_extreme_positive_funding",
    "pct_assets_extreme_negative_funding",
    "mkt_funding_dispersion",
    "mkt_funding_skew",
    "mkt_funding_tail_concentration",
    "mkt_funding_mean_z_30d",
    "mkt_funding_dispersion_z_30d",
]

FUNDING_PRICE_OI_INTERACTION_FEATURE_KEYS = [
    "positive_funding_x_price_down",
    "positive_funding_x_oi_drop",
    "negative_funding_x_price_up",
    "negative_funding_x_oi_drop",
    "funding_crowding_x_vol_expansion",
    "funding_flip_x_oi_flush",
    "funding_mean_reversion_after_oi_flush",
]

ASSET_OI_LIFECYCLE_FEATURE_KEYS = [
    "oi_drawdown_from_peak_24h",
    "oi_drawdown_from_peak_72h",
    "oi_drawdown_from_peak_168h",
    "oi_recovery_fraction_24h",
    "oi_recovery_fraction_72h",
    "bars_since_oi_low_24h_norm",
    "bars_since_oi_low_72h_norm",
    "bars_since_max_oi_drop_24h_norm",
    "bars_since_max_oi_drop_72h_norm",
    "oi_drop_acceleration_4h_rz",
    "oi_drop_deceleration_4h_rz",
]

PRICE_OI_QUADRANT_FEATURE_KEYS = [
    "price_down_oi_down_1h_rz",
    "price_down_oi_up_1h_rz",
    "price_up_oi_down_1h_rz",
    "price_up_oi_up_1h_rz",
    "price_down_oi_down_4h_rz",
    "price_down_oi_up_4h_rz",
    "price_up_oi_down_4h_rz",
    "price_up_oi_up_4h_rz",
]

PRICE_OI_RECOVERY_FEATURE_KEYS = [
    "price_recovery_fraction_24h",
    "price_recovery_fraction_72h",
    "price_minus_oi_recovery_24h",
    "price_minus_oi_recovery_72h",
    "price_recovery_oi_still_falling_1h",
    "price_recovery_oi_still_falling_4h",
]

FUNDING_LIFECYCLE_FEATURE_KEYS = [
    "funding_sign_persistence_24h",
    "funding_sign_persistence_72h",
    "hours_since_funding_sign_flip_24h_norm",
    "funding_positive_to_negative_intensity",
    "funding_negative_to_positive_intensity",
    "funding_crowding_release_4h",
]

MARKET_OI_LIFECYCLE_FEATURE_KEYS = [
    "mkt_median_oi_chg_1h_rz",
    "mkt_median_oi_chg_4h_rz",
    "mkt_pct_oi_chg_1h_rz_lt_minus1",
    "mkt_pct_oi_chg_4h_rz_lt_minus1",
    "mkt_pct_oi_chg_4h_rz_lt_minus2",
    "mkt_pct_oi_drawdown_24h_lt_minus5pct",
    "mkt_median_oi_drawdown_from_peak_24h",
    "mkt_median_oi_recovery_fraction_24h",
    "mkt_median_bars_since_max_oi_drop_24h_norm",
    "mkt_oi_flush_breadth_accel_1h",
    "mkt_oi_flush_breadth_recovery_4h",
]

MARKET_PRICE_OI_STATE_FEATURE_KEYS = [
    "mkt_pct_price_down_oi_down_1h",
    "mkt_pct_price_down_oi_up_1h",
    "mkt_pct_price_up_oi_down_1h",
    "mkt_pct_price_up_oi_up_1h",
    "mkt_pct_price_down_oi_down_4h",
    "mkt_pct_price_down_oi_up_4h",
    "mkt_pct_price_up_oi_down_4h",
    "mkt_pct_price_up_oi_up_4h",
    "mkt_median_long_flush_intensity_4h",
    "mkt_median_short_build_intensity_4h",
    "mkt_median_short_cover_intensity_1h",
]

ASSET_MARKET_LIFECYCLE_RESIDUAL_KEYS = [
    "asset_minus_mkt_oi_chg_1h_rz",
    "asset_minus_mkt_oi_chg_4h_rz",
    "asset_minus_mkt_oi_drawdown_24h",
    "asset_minus_mkt_oi_recovery_fraction_24h",
    "asset_minus_mkt_price_recovery_fraction_24h",
    "asset_minus_mkt_long_flush_intensity_4h",
    "asset_minus_mkt_short_cover_intensity_1h",
    "asset_minus_mkt_bars_since_oi_flush_24h",
]

ASSET_MARKET_LIQUIDATION_COMPOSITE_FEATURE_KEYS = [
    "asset_liquidation_phase_score",
    "asset_flush_exhaustion_score",
    "asset_short_covering_score",
    "mkt_systemic_deleveraging_score",
    "mkt_flush_exhaustion_score",
    "mkt_leverage_rebuild_score",
    "asset_mkt_liquidation_phase_divergence",
    "asset_mkt_exhaustion_phase_divergence",
]

OI_CRASH_LIFECYCLE_FEATURE_KEYS = [
    *ASSET_OI_LIFECYCLE_FEATURE_KEYS,
    *PRICE_OI_QUADRANT_FEATURE_KEYS,
    *PRICE_OI_RECOVERY_FEATURE_KEYS,
    *FUNDING_LIFECYCLE_FEATURE_KEYS,
    *MARKET_OI_LIFECYCLE_FEATURE_KEYS,
    *MARKET_PRICE_OI_STATE_FEATURE_KEYS,
    *ASSET_MARKET_LIFECYCLE_RESIDUAL_KEYS,
    *ASSET_MARKET_LIQUIDATION_COMPOSITE_FEATURE_KEYS,
]

OI_TRADING_FEATURE_KEYS = [
    *OI_NORMALIZED_FEATURE_KEYS,
    *OI_CHANGE_POINT_FEATURE_KEYS,
    "log_oi_to_volume_1d",
    "log_oi_to_volume_7d",
    "oi_to_volume_1d_z_90d",
    "oi_to_volume_7d_z_180d",
    "oi_chg_2h_robust_z",
    "oi_chg_4h_robust_z",
    "oi_chg_8h_robust_z",
    "oi_1d_chg_z",
    "oi_3d_chg_z",
    "oi_7d_chg_z",
    "price_x_oi_1d",
    "price_x_oi_3d",
    "price_x_oi_7d",
    "oi_1d_x_funding",
    "oi_3d_x_funding",
    "oi_7d_x_funding",
    "crowded_long_1d",
    "crowded_long_3d",
    "crowded_long_7d",
    "crowded_short_1d",
    "crowded_short_3d",
    "crowded_short_7d",
    "oi_1d_x_funding_1d_chg",
    "oi_3d_x_funding_1d_chg",
    "oi_7d_x_funding_1d_chg",
    "asset_minus_mkt_oi_1d",
    "asset_minus_mkt_oi_7d",
    "asset_minus_mkt_oi_1d_z_90d",
    "asset_minus_mkt_oi_7d_z_180d",
    "mkt_oi_z_30d",
    "mkt_oi_chg_z_24h",
    "mkt_oi_breadth_rising_24h",
    "mkt_oi_dispersion_24h",
    "cs_rank_oi_value_z_30d",
    "cs_rank_oi_chg_1d_z_90d",
    "funding_mean_7d_robust_z",
    "funding_mean_10d_robust_z",
    "funding_mean_15d_robust_z",
    "funding_vol_7d_robust_z",
    "funding_vol_10d_robust_z",
    "funding_vol_15d_robust_z",
    "oi_trend_7d_robust_z",
    "oi_trend_10d_robust_z",
    "oi_trend_15d_robust_z",
    "oi_vol_7d_robust_z",
    "oi_vol_10d_robust_z",
    "oi_vol_15d_robust_z",
    "price_trend_7d_vol_norm",
    "price_trend_10d_vol_norm",
    "price_trend_15d_vol_norm",
    "price_rv_7d_robust_z",
    "price_rv_10d_robust_z",
    "price_rv_15d_robust_z",
    *MARKET_OI_REGIME_FEATURE_KEYS,
    *PRICE_OI_STATE_FEATURE_KEYS,
    *MARKET_FUNDING_REGIME_FEATURE_KEYS,
    *FUNDING_PRICE_OI_INTERACTION_FEATURE_KEYS,
    *OI_CRASH_LIFECYCLE_FEATURE_KEYS,
]


def get_oi_internal_feature_names() -> list[str]:
    return list(OI_INTERNAL_FEATURE_KEYS)


def get_oi_trading_feature_names() -> list[str]:
    return list(OI_TRADING_FEATURE_KEYS)


def get_oi_normalized_feature_names() -> list[str]:
    return list(OI_NORMALIZED_FEATURE_KEYS)


def get_oi_feature_names() -> list[str]:
    return list(
        dict.fromkeys(
            OI_INTERNAL_FEATURE_KEYS + OI_TRADING_FEATURE_KEYS + OI_ALIAS_FEATURE_KEYS
        )
    )


def _short_long_change_point_features(
    frame: pd.DataFrame,
    *,
    prefix: str,
    short_window: int = 8,
    long_window: int = 32,
    sigma_window: int = 96,
    eps: float = 1e-9,
) -> dict[str, pd.DataFrame]:
    """Causal short-vs-long shift diagnostics down each symbol column."""
    x = frame.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    sw = max(2, int(short_window))
    lw = max(sw * 2, int(long_window))
    sigw = max(lw, int(sigma_window))
    mu_short = ff.numba_rolling_mean(x, sw).astype(np.float32)
    mu_long = ff.numba_rolling_mean(x, lw).astype(np.float32)
    sigma_long = ff.numba_rolling_std(x, sigw).astype(np.float32).replace(0.0, np.nan)
    short_std = ff.numba_rolling_std(x, sw).astype(np.float32).replace(0.0, np.nan)
    long_std = ff.numba_rolling_std(x, lw).astype(np.float32).replace(0.0, np.nan)
    z = (
        ((mu_short - mu_long) / (sigma_long + np.float32(eps)))
        .replace([np.inf, -np.inf], np.nan)
        .clip(-12.0, 12.0)
        .fillna(0.0)
    )
    logstd = (
        (np.log(short_std.clip(lower=eps)) - np.log(long_std.clip(lower=eps)))
        .replace([np.inf, -np.inf], np.nan)
        .clip(-8.0, 8.0)
        .fillna(0.0)
    )
    absratio = (
        (mu_short.abs() / (mu_long.abs() + np.float32(eps)))
        .replace([np.inf, -np.inf], np.nan)
        .clip(0.0, 100.0)
        .fillna(1.0)
    )
    return {
        f"{prefix}_cp_z_{sw}_{lw}_{sigw}": z.astype(np.float32),
        f"{prefix}_cp_logstd_{sw}_{lw}": logstd.astype(np.float32),
        f"{prefix}_cp_absratio_{sw}_{lw}": absratio.astype(np.float32),
    }


def rolling_zscore_by_symbol(
    frame: pd.DataFrame, window: int, *, min_periods: int | None = None
) -> pd.DataFrame:
    """Rolling z-score down each symbol column."""
    window = max(1, int(window))
    if min_periods is None:
        # Long-horizon OI features should use a growing window after one month
        # of history instead of waiting for a full 90d/180d window, but they
        # should not become active on just a few early rows.
        min_periods = min(window, max(5, 24 * 30 if window >= 24 * 30 else window // 5))
    mean = frame.rolling(window, min_periods=min_periods).mean()
    std = frame.rolling(window, min_periods=min_periods).std(ddof=0)
    return ((frame - mean) / std.replace(0.0, np.nan)).astype(np.float32)


def rolling_robust_zscore_by_symbol(
    frame: pd.DataFrame,
    window: int,
    *,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Rolling robust z-score down each symbol column using median and IQR."""
    window = max(1, int(window))
    if min_periods is None:
        min_periods = min(window, max(24 * 7, window // 4))
    median = frame.rolling(window, min_periods=min_periods).median()
    q75 = frame.rolling(window, min_periods=min_periods).quantile(0.75)
    q25 = frame.rolling(window, min_periods=min_periods).quantile(0.25)
    iqr = (q75 - q25).replace(0.0, np.nan)
    return ((frame - median) / iqr).astype(np.float32)


def rolling_long_iqr_robust_zscore_by_symbol(
    frame: pd.DataFrame,
    base_window: int,
) -> pd.DataFrame:
    """IQR robust z-score with half-length window and one-sixth minimum."""
    base_window = max(1, int(base_window))
    window = max(1, base_window // 2)
    min_periods = max(1, base_window // 6)
    return rolling_robust_zscore_by_symbol(
        frame,
        window,
        min_periods=min(window, min_periods),
    )


def rolling_percentile_by_symbol(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    window = max(1, int(window))
    min_periods = min(window, max(5, 24 * 30 if window >= 24 * 30 else window // 5))

    def rank_last(values: np.ndarray) -> float:
        cur = values[-1]
        hist = values[np.isfinite(values)]
        if not np.isfinite(cur) or len(hist) < min_periods:
            return np.nan
        return float((hist <= cur).mean())

    return (
        frame.astype(np.float32)
        .rolling(window, min_periods=min_periods)
        .apply(rank_last, raw=True)
        .astype(np.float32)
    )


def cross_sectional_mean(frame: pd.DataFrame) -> pd.Series:
    return frame.mean(axis=1, skipna=True).astype(np.float32)


def cross_sectional_median(frame: pd.DataFrame) -> pd.Series:
    return frame.median(axis=1, skipna=True).astype(np.float32)


@njit(parallel=True, cache=True)
def _leave_one_out_row_median_nb(mat: np.ndarray) -> np.ndarray:
    n_rows, n_cols = mat.shape
    out = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    for i in prange(n_rows):
        vals = np.empty(n_cols, dtype=np.float64)
        count = 0
        for j in range(n_cols):
            val = mat[i, j]
            if np.isfinite(val):
                vals[count] = val
                count += 1
        if count == 0:
            continue
        sorted_vals = np.sort(vals[:count])
        if count % 2 == 1:
            all_median = sorted_vals[count // 2]
        else:
            all_median = 0.5 * (sorted_vals[count // 2 - 1] + sorted_vals[count // 2])
        for j in range(n_cols):
            val = mat[i, j]
            if not np.isfinite(val):
                out[i, j] = np.float32(all_median)
                continue
            if count <= 1:
                continue
            rank = 0
            while rank < count and sorted_vals[rank] < val:
                rank += 1
            excl_count = count - 1
            if excl_count % 2 == 1:
                k = excl_count // 2
                src = k if k < rank else k + 1
                median = sorted_vals[src]
            else:
                k1 = excl_count // 2 - 1
                k2 = excl_count // 2
                src1 = k1 if k1 < rank else k1 + 1
                src2 = k2 if k2 < rank else k2 + 1
                median = 0.5 * (sorted_vals[src1] + sorted_vals[src2])
            out[i, j] = np.float32(median)
    return out


def leave_one_out_cross_sectional_median(frame: pd.DataFrame) -> pd.DataFrame:
    arr = np.ascontiguousarray(frame.to_numpy(dtype=np.float32, copy=False))
    out = _leave_one_out_row_median_nb(arr)
    return pd.DataFrame(out, index=frame.index, columns=frame.columns)


def cross_sectional_std(frame: pd.DataFrame) -> pd.Series:
    return frame.std(axis=1, skipna=True, ddof=0).astype(np.float32)


def cross_sectional_rank_pct(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, method="average", pct=True).astype(np.float32)


def _broadcast(series: pd.Series, like: pd.DataFrame) -> pd.DataFrame:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float32, copy=False)
    return pd.DataFrame(
        np.broadcast_to(arr[:, None], like.shape),
        index=like.index,
        columns=like.columns,
        copy=False,
    ).astype(np.float32, copy=False)


def _rolling_series_robust_z(
    series: pd.Series,
    window: int,
    *,
    min_periods: int | None = None,
) -> pd.Series:
    window = max(1, int(window))
    if min_periods is None:
        min_periods = min(window, max(24 * 7, window // 4))
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = s.rolling(window, min_periods=min_periods).median()
    q75 = s.rolling(window, min_periods=min_periods).quantile(0.75)
    q25 = s.rolling(window, min_periods=min_periods).quantile(0.25)
    iqr = (q75 - q25).replace(0.0, np.nan)
    return ((s - med) / iqr).replace([np.inf, -np.inf], np.nan).astype(np.float32)


def _bars_since_rolling_trough(series: pd.Series, window: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    out = np.full(values.shape, np.nan, dtype=np.float32)
    w = max(1, int(window))
    for i in range(values.shape[0]):
        start = max(0, i - w + 1)
        window_values = values[start : i + 1]
        finite = np.isfinite(window_values)
        if not finite.any():
            continue
        finite_idx = np.flatnonzero(finite)
        trough_local = finite_idx[int(np.nanargmin(window_values[finite]))]
        out[i] = float(i - (start + trough_local))
    return pd.Series(out, index=series.index, dtype=np.float32)


@njit(cache=True, parallel=True)
def _rolling_lifecycle_matrix(
    values: np.ndarray, drop_values: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_rows, n_cols = values.shape
    drawdown = np.empty((n_rows, n_cols), dtype=np.float32)
    recovery = np.empty((n_rows, n_cols), dtype=np.float32)
    bars_since_low = np.empty((n_rows, n_cols), dtype=np.float32)
    bars_since_max_drop = np.empty((n_rows, n_cols), dtype=np.float32)
    w = max(1, int(window))
    eps = np.float32(1e-12)
    for j in prange(n_cols):
        for i in range(n_rows):
            cur = values[i, j]
            drawdown[i, j] = np.nan
            recovery[i, j] = np.nan
            bars_since_low[i, j] = np.nan
            bars_since_max_drop[i, j] = np.nan
            if not np.isfinite(cur):
                continue
            start = max(0, i - w + 1)
            peak_val = -np.inf
            peak_idx = -1
            low_val = np.inf
            low_idx = -1
            min_drop = np.inf
            min_drop_idx = -1
            for k in range(start, i + 1):
                v = values[k, j]
                if np.isfinite(v):
                    if v >= peak_val:
                        peak_val = v
                        peak_idx = k
                    if v <= low_val:
                        low_val = v
                        low_idx = k
                d = drop_values[k, j]
                if np.isfinite(d) and d <= min_drop:
                    min_drop = d
                    min_drop_idx = k
            if peak_idx >= 0 and np.isfinite(peak_val):
                drawdown[i, j] = np.float32(min(cur - peak_val, 0.0))
                post_low = peak_val
                for k in range(peak_idx, i + 1):
                    v = values[k, j]
                    if np.isfinite(v) and v < post_low:
                        post_low = v
                denom = peak_val - post_low
                if np.isfinite(denom) and denom > eps:
                    rec = (cur - post_low) / (denom + eps)
                    if rec < -0.5:
                        rec = -0.5
                    elif rec > 1.5:
                        rec = 1.5
                    recovery[i, j] = np.float32(rec)
                else:
                    recovery[i, j] = np.float32(1.0)
            if low_idx >= 0:
                bars_since_low[i, j] = np.float32((i - low_idx) / float(w))
            if min_drop_idx >= 0:
                bars_since_max_drop[i, j] = np.float32((i - min_drop_idx) / float(w))
    return drawdown, recovery, bars_since_low, bars_since_max_drop


def _rolling_lifecycle_frames(
    values: pd.DataFrame,
    drop_values: pd.DataFrame,
    window: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aligned_drop = drop_values.reindex(index=values.index, columns=values.columns)
    drawdown, recovery, bars_low, bars_drop = _rolling_lifecycle_matrix(
        values.to_numpy(dtype=np.float32, copy=False),
        aligned_drop.to_numpy(dtype=np.float32, copy=False),
        int(window),
    )
    kwargs = {"index": values.index, "columns": values.columns}
    return (
        pd.DataFrame(drawdown, **kwargs),
        pd.DataFrame(recovery, **kwargs),
        pd.DataFrame(bars_low, **kwargs),
        pd.DataFrame(bars_drop, **kwargs),
    )


@njit(cache=True, parallel=True)
def _bars_since_true_matrix(mask: np.ndarray, cap: int) -> np.ndarray:
    n_rows, n_cols = mask.shape
    out = np.empty((n_rows, n_cols), dtype=np.float32)
    cap = max(1, int(cap))
    for j in prange(n_cols):
        last = -1
        for i in range(n_rows):
            if mask[i, j]:
                last = i
            if last >= 0:
                dist = i - last
                if dist > cap:
                    dist = cap
                out[i, j] = np.float32(dist / float(cap))
            else:
                out[i, j] = np.nan
    return out


def _bars_since_true_frame(mask: pd.DataFrame, cap: int) -> pd.DataFrame:
    out = _bars_since_true_matrix(
        mask.fillna(False).to_numpy(dtype=np.bool_, copy=False), int(cap)
    )
    return pd.DataFrame(out, index=mask.index, columns=mask.columns)


def _subset_cols(columns: pd.Index, token: str) -> list[str]:
    token = token.upper()
    out: list[str] = []
    for col in columns:
        text = str(col).upper()
        if token in text:
            out.append(col)
    return out


def _broadcast_clean(
    series: pd.Series, like: pd.DataFrame, *, clip: tuple[float, float] | None = None
) -> pd.DataFrame:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if clip is not None:
        s = s.clip(float(clip[0]), float(clip[1]))
    return _broadcast(s.fillna(0.0).astype(np.float32), like)


def compute_oi_features(
    *,
    oi_native: pd.DataFrame,
    price: pd.DataFrame,
    quote_volume: pd.DataFrame,
    funding_rate: pd.DataFrame | None = None,
    bars_per_day: int = 24,
) -> dict[str, pd.DataFrame]:
    """Compute OI features from wide hourly panels.

    ``oi_native`` is exchange-native OI. ``oi_value`` is normalized to quote
    notional by multiplying by the aligned perp price.
    """
    bpd = max(1, int(bars_per_day))
    w_1d = bpd
    w_3d = 3 * bpd
    w_7d = 7 * bpd
    w_30d = 30 * bpd
    w_90d = 90 * bpd
    w_180d = 180 * bpd

    oi_native = (
        oi_native.reindex(index=price.index, columns=price.columns)
        .replace([np.inf, -np.inf], np.nan)
        .where(lambda f: f > 0.0)
        .ffill(limit=3 * bpd)
        .astype(np.float32)
    )
    price = (
        price.reindex(index=oi_native.index, columns=oi_native.columns)
        .replace([np.inf, -np.inf], np.nan)
        .where(lambda f: f > 0.0)
        .ffill(limit=bpd)
        .astype(np.float32)
    )
    quote_volume = (
        quote_volume.reindex(index=oi_native.index, columns=oi_native.columns)
        .replace([np.inf, -np.inf], np.nan)
        .where(lambda f: f > 0.0)
        .fillna(0.0)
        .astype(np.float32)
    )
    if funding_rate is not None:
        funding_rate = (
            funding_rate.reindex(index=oi_native.index, columns=oi_native.columns)
            .replace([np.inf, -np.inf], np.nan)
            .astype(np.float32)
        )

    out: dict[str, pd.DataFrame] = {}
    oi_value = (
        (oi_native * price).replace([np.inf, -np.inf], np.nan).where(lambda f: f > 0.0)
    )
    oi_value_log = np.log(oi_value.clip(lower=1e-12)).astype(np.float32)
    price_log = np.log(price.clip(lower=1e-12)).astype(np.float32)

    volume_usd_1d = quote_volume.rolling(w_1d, min_periods=1).sum()
    volume_usd_7d = quote_volume.rolling(w_7d, min_periods=1).sum()
    oi_to_volume_1d = (oi_value / volume_usd_1d.clip(lower=1e-12)).replace(
        [np.inf, -np.inf], np.nan
    )
    oi_to_volume_7d = (oi_value / volume_usd_7d.clip(lower=1e-12)).replace(
        [np.inf, -np.inf], np.nan
    )

    out["oi_to_volume_1d"] = oi_to_volume_1d.astype(np.float32)
    out["oi_to_volume_7d"] = oi_to_volume_7d.astype(np.float32)
    out["oi_value_log"] = oi_value_log.astype(np.float32)

    log_chg: dict[str, pd.DataFrame] = {}
    for label, shift in (("1d", w_1d), ("3d", w_3d), ("7d", w_7d)):
        chg = (oi_value_log - oi_value_log.shift(shift)).astype(np.float32)
        log_chg[label] = chg
        out[f"oi_value_{label}_log_chg"] = chg

    price_ret_z: dict[str, pd.DataFrame] = {}
    for label, shift in (("1d", w_1d), ("3d", w_3d), ("7d", w_7d)):
        ret = (price_log - price_log.shift(shift)).astype(np.float32)
        z = (
            rolling_long_iqr_robust_zscore_by_symbol(ret, w_90d)
            .clip(-10, 10)
            .astype(np.float32)
        )
        price_ret_z[label] = z
        out[f"price_{label}_ret_z"] = z

    if funding_rate is None:
        funding_z_90d = pd.DataFrame(
            np.nan, index=oi_native.index, columns=oi_native.columns, dtype=np.float32
        )
        funding_1d_chg = funding_z_90d.copy()
        funding_1d_chg_z_90d = funding_z_90d.copy()
    else:
        funding_z_90d = rolling_long_iqr_robust_zscore_by_symbol(
            funding_rate, w_90d
        ).clip(-10, 10)
        funding_1d_chg = (funding_rate - funding_rate.shift(w_1d)).astype(np.float32)
        funding_1d_chg_z_90d = rolling_long_iqr_robust_zscore_by_symbol(
            funding_1d_chg, w_90d
        ).clip(-10, 10)
    out["funding_z_90d"] = funding_z_90d.astype(np.float32)
    out["funding_1d_chg"] = funding_1d_chg.astype(np.float32)
    out["funding_1d_chg_z_90d"] = funding_1d_chg_z_90d.astype(np.float32)

    oi_chg_z: dict[str, pd.DataFrame] = {}
    for label, window in (("1d", w_90d), ("3d", w_90d), ("7d", w_90d)):
        z = rolling_long_iqr_robust_zscore_by_symbol(log_chg[label], window).clip(
            -10, 10
        )
        oi_chg_z[label] = z.astype(np.float32)
        out[f"oi_{label}_chg_z_90d"] = oi_chg_z[label]
        out[f"oi_value_{label}_chg_z_90d"] = oi_chg_z[label]
        out[f"oi_value_{label}_log_chg_z_90d"] = oi_chg_z[label]
        out[f"oi_{label}_chg_z"] = oi_chg_z[label]

    oi_7d_chg_z_180d = rolling_long_iqr_robust_zscore_by_symbol(
        log_chg["7d"], w_180d
    ).clip(-10, 10)
    out["oi_value_7d_chg_z_180d"] = oi_7d_chg_z_180d.astype(np.float32)
    out["oi_value_7d_log_chg_z_180d"] = oi_7d_chg_z_180d.astype(np.float32)

    oi_value_z_30d = rolling_robust_zscore_by_symbol(
        oi_value_log, w_30d, min_periods=w_7d
    ).clip(-10, 10)
    oi_value_z_90d = rolling_long_iqr_robust_zscore_by_symbol(oi_value_log, w_90d).clip(
        -10, 10
    )
    out["oi_value_z_30d"] = oi_value_z_30d.astype(np.float32)
    out["oi_value_z_90d"] = oi_value_z_90d.astype(np.float32)
    out["oi_value_log_z_30d"] = oi_value_z_30d.astype(np.float32)
    out["oi_value_log_z_90d"] = oi_value_z_90d.astype(np.float32)
    out["oi_value_pct_90d"] = rolling_percentile_by_symbol(oi_value_log, w_90d).astype(
        np.float32
    )

    log_oi_to_volume_1d = np.log1p(oi_to_volume_1d.clip(lower=0.0)).replace(
        [np.inf, -np.inf], np.nan
    )
    log_oi_to_volume_7d = np.log1p(oi_to_volume_7d.clip(lower=0.0)).replace(
        [np.inf, -np.inf], np.nan
    )
    out["log_oi_to_volume_1d"] = log_oi_to_volume_1d.astype(np.float32)
    out["log_oi_to_volume_7d"] = log_oi_to_volume_7d.astype(np.float32)
    out["oi_to_volume_1d_z_90d"] = (
        rolling_long_iqr_robust_zscore_by_symbol(log_oi_to_volume_1d, w_90d)
        .clip(-10, 10)
        .astype(np.float32)
    )
    out["oi_to_volume_7d_z_180d"] = (
        rolling_long_iqr_robust_zscore_by_symbol(log_oi_to_volume_7d, w_180d)
        .clip(-10, 10)
        .astype(np.float32)
    )
    out.update(
        _short_long_change_point_features(
            oi_value_log,
            prefix="oi_value_log",
        )
    )
    out.update(
        _short_long_change_point_features(
            log_chg["1d"],
            prefix="oi_value_1d_log_chg",
        )
    )
    out.update(
        _short_long_change_point_features(
            log_oi_to_volume_1d,
            prefix="log_oi_to_volume_1d",
        )
    )

    oi_value_log_1d = oi_value_log.rolling(w_1d, min_periods=1).mean()
    oi_value_log_7d = oi_value_log.rolling(w_7d, min_periods=1).mean()
    out["oi_value_log_1d_robust_z"] = (
        rolling_robust_zscore_by_symbol(
            oi_value_log_1d,
            w_30d,
            min_periods=w_7d,
        )
        .clip(-10, 10)
        .astype(np.float32)
    )
    out["oi_value_log_7d_robust_z"] = (
        rolling_robust_zscore_by_symbol(
            oi_value_log_7d,
            w_30d,
            min_periods=w_7d,
        )
        .clip(-10, 10)
        .astype(np.float32)
    )

    for hours in (2, 4, 8):
        shift = hours * bpd // 24
        chg = (oi_value_log - oi_value_log.shift(max(1, shift))).astype(np.float32)
        out[f"oi_chg_{hours}h_robust_z"] = (
            rolling_robust_zscore_by_symbol(
                chg,
                w_30d,
                min_periods=w_7d,
            )
            .clip(-10, 10)
            .astype(np.float32)
        )

    for label in ("1d", "3d", "7d"):
        out[f"price_x_oi_{label}"] = (price_ret_z[label] * oi_chg_z[label]).astype(
            np.float32
        )
        out[f"oi_{label}_x_funding"] = (oi_chg_z[label] * funding_z_90d).astype(
            np.float32
        )
        out[f"crowded_long_{label}"] = (
            oi_chg_z[label] * funding_z_90d.clip(lower=0.0)
        ).astype(np.float32)
        out[f"crowded_short_{label}"] = (
            oi_chg_z[label] * (-funding_z_90d).clip(lower=0.0)
        ).astype(np.float32)
        out[f"oi_{label}_x_funding_1d_chg"] = (
            oi_chg_z[label] * funding_1d_chg_z_90d
        ).astype(np.float32)

    mkt_oi_1d = cross_sectional_mean(log_chg["1d"])
    mkt_oi_7d = cross_sectional_mean(log_chg["7d"])
    asset_minus_mkt_oi_1d = log_chg["1d"].sub(mkt_oi_1d, axis=0)
    asset_minus_mkt_oi_7d = log_chg["7d"].sub(mkt_oi_7d, axis=0)
    out["asset_minus_mkt_oi_1d"] = asset_minus_mkt_oi_1d.astype(np.float32)
    out["asset_minus_mkt_oi_7d"] = asset_minus_mkt_oi_7d.astype(np.float32)
    out["asset_minus_mkt_oi_1d_z_90d"] = (
        rolling_long_iqr_robust_zscore_by_symbol(asset_minus_mkt_oi_1d, w_90d)
        .clip(-10, 10)
        .astype(np.float32)
    )
    out["asset_minus_mkt_oi_7d_z_180d"] = (
        rolling_long_iqr_robust_zscore_by_symbol(asset_minus_mkt_oi_7d, w_180d)
        .clip(-10, 10)
        .astype(np.float32)
    )
    out.update(
        _short_long_change_point_features(
            asset_minus_mkt_oi_1d,
            prefix="asset_minus_mkt_oi_1d",
        )
    )

    out["mkt_oi_z_30d"] = _broadcast(cross_sectional_mean(oi_value_z_30d), oi_native)
    out["mkt_oi_chg_z_24h"] = _broadcast(
        cross_sectional_mean(oi_chg_z["1d"]), oi_native
    )
    oi_rising_1d = (log_chg["1d"] > 0.0).astype(np.float32).where(log_chg["1d"].notna())
    out["mkt_oi_breadth_rising_24h"] = _broadcast(
        cross_sectional_mean(oi_rising_1d), oi_native
    )
    out["mkt_oi_dispersion_24h"] = _broadcast(
        cross_sectional_std(oi_chg_z["1d"]), oi_native
    )
    out["cs_rank_oi_value_z_30d"] = cross_sectional_rank_pct(oi_value_z_30d)
    out["cs_rank_oi_chg_1d_z_90d"] = cross_sectional_rank_pct(oi_chg_z["1d"])

    ret_1h = price_log.diff(1).astype(np.float32)
    oi_1h_chg = oi_value_log.diff(1).astype(np.float32)
    h_1h = max(1, bpd // 24)
    h_4h = max(h_1h, 4 * h_1h)
    h_24h = max(h_1h, 24 * h_1h)
    mkt_oi_value = oi_value.sum(axis=1, min_count=1).replace([np.inf, -np.inf], np.nan)
    mkt_oi_log = np.log(mkt_oi_value.clip(lower=1e-12)).astype(np.float32)

    oi_chg_15m = (oi_value_log - oi_value_log.shift(1)).astype(np.float32)
    oi_chg_1h = (oi_value_log - oi_value_log.shift(h_1h)).astype(np.float32)
    oi_chg_4h = (oi_value_log - oi_value_log.shift(h_4h)).astype(np.float32)
    oi_chg_24h = (oi_value_log - oi_value_log.shift(h_24h)).astype(np.float32)
    mkt_oi_chg_15m = cross_sectional_mean(oi_chg_15m)
    mkt_oi_chg_1h = cross_sectional_mean(oi_chg_1h)
    mkt_oi_chg_4h = cross_sectional_mean(oi_chg_4h)
    mkt_oi_chg_24h = cross_sectional_mean(oi_chg_24h)
    out["mkt_oi_chg_15m"] = _broadcast_clean(
        mkt_oi_chg_15m, oi_native, clip=(-0.50, 0.50)
    )
    out["mkt_oi_chg_1h"] = _broadcast_clean(
        mkt_oi_chg_1h, oi_native, clip=(-0.75, 0.75)
    )
    out["mkt_oi_chg_4h"] = _broadcast_clean(
        mkt_oi_chg_4h, oi_native, clip=(-1.50, 1.50)
    )
    out["mkt_oi_chg_24h"] = _broadcast_clean(
        mkt_oi_chg_24h, oi_native, clip=(-3.00, 3.00)
    )
    out["mkt_oi_chg_accel_1h"] = _broadcast_clean(
        mkt_oi_chg_1h - mkt_oi_chg_1h.shift(h_1h), oi_native, clip=(-1.50, 1.50)
    )
    out["mkt_oi_chg_accel_4h"] = _broadcast_clean(
        mkt_oi_chg_4h - mkt_oi_chg_4h.shift(h_4h), oi_native, clip=(-2.50, 2.50)
    )

    mkt_oi_peak_24h = mkt_oi_log.rolling(h_24h, min_periods=max(2, h_1h)).max()
    mkt_oi_peak_7d = mkt_oi_log.rolling(w_7d, min_periods=max(h_24h, h_1h)).max()
    mkt_oi_low_24h = mkt_oi_log.rolling(h_24h, min_periods=max(2, h_1h)).min()
    out["mkt_oi_drawdown_from_24h_peak"] = _broadcast_clean(
        (mkt_oi_peak_24h - mkt_oi_log).clip(lower=0.0), oi_native, clip=(0.0, 5.0)
    )
    out["mkt_oi_drawdown_from_7d_peak"] = _broadcast_clean(
        (mkt_oi_peak_7d - mkt_oi_log).clip(lower=0.0), oi_native, clip=(0.0, 8.0)
    )
    oi_flush_z = _rolling_series_robust_z(
        -mkt_oi_chg_24h, w_30d, min_periods=w_7d
    ).clip(-10, 10)
    out["mkt_oi_flush_z_30d"] = _broadcast_clean(
        oi_flush_z, oi_native, clip=(-10.0, 10.0)
    )
    out["mkt_oi_recovery_from_24h_low"] = _broadcast_clean(
        (mkt_oi_log - mkt_oi_low_24h).clip(lower=0.0), oi_native, clip=(0.0, 5.0)
    )
    out["bars_since_mkt_oi_trough"] = _broadcast_clean(
        _bars_since_rolling_trough(mkt_oi_log, h_24h),
        oi_native,
        clip=(0.0, float(h_24h)),
    )

    for label, frame in (("1h", oi_chg_1h), ("4h", oi_chg_4h), ("24h", oi_chg_24h)):
        out[f"pct_assets_oi_down_{label}"] = _broadcast_clean(
            frame.lt(0.0).where(frame.notna()).mean(axis=1, skipna=True),
            oi_native,
            clip=(0.0, 1.0),
        )
    oi_chg_1h_z = rolling_robust_zscore_by_symbol(
        oi_chg_1h, w_30d, min_periods=w_7d
    ).clip(-10, 10)
    oi_chg_1h_z = (
        oi_chg_1h_z.where(
            oi_chg_1h_z.notna(),
            rolling_zscore_by_symbol(oi_chg_1h, w_30d, min_periods=w_7d).clip(-10, 10),
        )
        .fillna(0.0)
        .astype(np.float32)
    )
    oi_chg_4h_z = rolling_robust_zscore_by_symbol(
        oi_chg_4h, w_30d, min_periods=w_7d
    ).clip(-10, 10)
    oi_chg_4h_z = (
        oi_chg_4h_z.where(
            oi_chg_4h_z.notna(),
            rolling_zscore_by_symbol(oi_chg_4h, w_30d, min_periods=w_7d).clip(-10, 10),
        )
        .fillna(0.0)
        .astype(np.float32)
    )
    out["pct_assets_extreme_oi_drop_1h"] = _broadcast_clean(
        oi_chg_1h_z.lt(-2.0).where(oi_chg_1h_z.notna()).mean(axis=1, skipna=True),
        oi_native,
        clip=(0.0, 1.0),
    )
    out["pct_assets_extreme_oi_drop_4h"] = _broadcast_clean(
        oi_chg_4h_z.lt(-2.0).where(oi_chg_4h_z.notna()).mean(axis=1, skipna=True),
        oi_native,
        clip=(0.0, 1.0),
    )
    h_72h = max(h_24h, 3 * h_24h)
    oi_dd_24h, oi_rec_24h, oi_low_24h, oi_drop_24h = _rolling_lifecycle_frames(
        oi_value_log, oi_chg_1h, h_24h
    )
    oi_dd_72h, oi_rec_72h, oi_low_72h, oi_drop_72h = _rolling_lifecycle_frames(
        oi_value_log, oi_chg_1h, h_72h
    )
    oi_dd_168h, _, _, _ = _rolling_lifecycle_frames(oi_value_log, oi_chg_1h, w_7d)
    out["oi_drawdown_from_peak_24h"] = oi_dd_24h.clip(-5.0, 0.0).astype(np.float32)
    out["oi_drawdown_from_peak_72h"] = oi_dd_72h.clip(-8.0, 0.0).astype(np.float32)
    out["oi_drawdown_from_peak_168h"] = oi_dd_168h.clip(-10.0, 0.0).astype(np.float32)
    out["oi_recovery_fraction_24h"] = oi_rec_24h.clip(-0.5, 1.5).astype(np.float32)
    out["oi_recovery_fraction_72h"] = oi_rec_72h.clip(-0.5, 1.5).astype(np.float32)
    out["bars_since_oi_low_24h_norm"] = oi_low_24h.clip(0.0, 1.0).astype(np.float32)
    out["bars_since_oi_low_72h_norm"] = oi_low_72h.clip(0.0, 1.0).astype(np.float32)
    out["bars_since_max_oi_drop_24h_norm"] = oi_drop_24h.clip(0.0, 1.0).astype(
        np.float32
    )
    out["bars_since_max_oi_drop_72h_norm"] = oi_drop_72h.clip(0.0, 1.0).astype(
        np.float32
    )
    oi_chg_4h_z_delta = (oi_chg_4h_z - oi_chg_4h_z.shift(h_4h)).astype(np.float32)
    out["oi_drop_acceleration_4h_rz"] = (
        (-oi_chg_4h_z_delta).clip(-10.0, 10.0).astype(np.float32)
    )
    out["oi_drop_deceleration_4h_rz"] = oi_chg_4h_z_delta.clip(
        lower=0.0, upper=10.0
    ).astype(np.float32)

    flush_breadth_1h = (
        oi_chg_1h_z.lt(-1.0).where(oi_chg_1h_z.notna()).mean(axis=1, skipna=True)
    )
    flush_breadth_4h = (
        oi_chg_4h_z.lt(-1.0).where(oi_chg_4h_z.notna()).mean(axis=1, skipna=True)
    )
    out["mkt_median_oi_chg_1h_rz"] = _broadcast_clean(
        cross_sectional_median(oi_chg_1h_z), oi_native, clip=(-10.0, 10.0)
    )
    out["mkt_median_oi_chg_4h_rz"] = _broadcast_clean(
        cross_sectional_median(oi_chg_4h_z), oi_native, clip=(-10.0, 10.0)
    )
    out["mkt_pct_oi_chg_1h_rz_lt_minus1"] = _broadcast_clean(
        flush_breadth_1h, oi_native, clip=(0.0, 1.0)
    )
    out["mkt_pct_oi_chg_4h_rz_lt_minus1"] = _broadcast_clean(
        flush_breadth_4h, oi_native, clip=(0.0, 1.0)
    )
    out["mkt_pct_oi_chg_4h_rz_lt_minus2"] = _broadcast_clean(
        oi_chg_4h_z.lt(-2.0).where(oi_chg_4h_z.notna()).mean(axis=1, skipna=True),
        oi_native,
        clip=(0.0, 1.0),
    )
    out["mkt_pct_oi_drawdown_24h_lt_minus5pct"] = _broadcast_clean(
        oi_dd_24h.lt(np.log(0.95)).where(oi_dd_24h.notna()).mean(axis=1, skipna=True),
        oi_native,
        clip=(0.0, 1.0),
    )
    out["mkt_median_oi_drawdown_from_peak_24h"] = _broadcast_clean(
        cross_sectional_median(oi_dd_24h), oi_native, clip=(-5.0, 0.0)
    )
    out["mkt_median_oi_recovery_fraction_24h"] = _broadcast_clean(
        cross_sectional_median(oi_rec_24h), oi_native, clip=(-0.5, 1.5)
    )
    out["mkt_median_bars_since_max_oi_drop_24h_norm"] = _broadcast_clean(
        cross_sectional_median(oi_drop_24h), oi_native, clip=(0.0, 1.0)
    )
    out["mkt_oi_flush_breadth_accel_1h"] = _broadcast_clean(
        (flush_breadth_4h - flush_breadth_4h.shift(h_1h)).fillna(0.0),
        oi_native,
        clip=(-1.0, 1.0),
    )
    out["mkt_oi_flush_breadth_recovery_4h"] = _broadcast_clean(
        (flush_breadth_4h.rolling(h_4h, min_periods=1).max() - flush_breadth_4h).fillna(
            0.0
        ),
        oi_native,
        clip=(0.0, 1.0),
    )
    out["mkt_oi_dispersion_1h"] = _broadcast_clean(
        cross_sectional_std(oi_chg_1h), oi_native, clip=(0.0, 2.0)
    )
    out["mkt_oi_dispersion_4h"] = _broadcast_clean(
        cross_sectional_std(oi_chg_4h), oi_native, clip=(0.0, 3.0)
    )
    btc_cols = _subset_cols(oi_value.columns, "BTC")
    eth_cols = _subset_cols(oi_value.columns, "ETH")
    if btc_cols or eth_cols:
        btc_eth_oi = oi_value.reindex(
            columns=list(dict.fromkeys(btc_cols + eth_cols))
        ).sum(axis=1, min_count=1)
        concentration = (btc_eth_oi / mkt_oi_value.replace(0.0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )
    else:
        concentration = pd.Series(np.nan, index=oi_value.index, dtype=np.float32)
    out["mkt_oi_concentration_btc_eth"] = _broadcast_clean(
        concentration, oi_native, clip=(0.0, 1.0)
    )

    price_ret_1h = (price_log - price_log.shift(h_1h)).astype(np.float32)
    price_ret_4h = (price_log - price_log.shift(h_4h)).astype(np.float32)
    price_ret_1h_z = rolling_robust_zscore_by_symbol(
        price_ret_1h, w_30d, min_periods=w_7d
    ).clip(-10, 10)
    price_ret_1h_z = (
        price_ret_1h_z.where(
            price_ret_1h_z.notna(),
            rolling_zscore_by_symbol(price_ret_1h, w_30d, min_periods=w_7d).clip(
                -10, 10
            ),
        )
        .fillna(0.0)
        .astype(np.float32)
    )
    price_ret_4h_z = rolling_robust_zscore_by_symbol(
        price_ret_4h, w_30d, min_periods=w_7d
    ).clip(-10, 10)
    price_ret_4h_z = (
        price_ret_4h_z.where(
            price_ret_4h_z.notna(),
            rolling_zscore_by_symbol(price_ret_4h, w_30d, min_periods=w_7d).clip(
                -10, 10
            ),
        )
        .fillna(0.0)
        .astype(np.float32)
    )
    out["price_down_oi_down_1h_rz"] = (
        ((-price_ret_1h_z).clip(lower=0.0) * (-oi_chg_1h_z).clip(lower=0.0))
        .clip(0.0, 100.0)
        .astype(np.float32)
    )
    out["price_down_oi_up_1h_rz"] = (
        ((-price_ret_1h_z).clip(lower=0.0) * oi_chg_1h_z.clip(lower=0.0))
        .clip(0.0, 100.0)
        .astype(np.float32)
    )
    out["price_up_oi_down_1h_rz"] = (
        (price_ret_1h_z.clip(lower=0.0) * (-oi_chg_1h_z).clip(lower=0.0))
        .clip(0.0, 100.0)
        .astype(np.float32)
    )
    out["price_up_oi_up_1h_rz"] = (
        (price_ret_1h_z.clip(lower=0.0) * oi_chg_1h_z.clip(lower=0.0))
        .clip(0.0, 100.0)
        .astype(np.float32)
    )
    out["price_down_oi_down_4h_rz"] = (
        ((-price_ret_4h_z).clip(lower=0.0) * (-oi_chg_4h_z).clip(lower=0.0))
        .clip(0.0, 100.0)
        .astype(np.float32)
    )
    out["price_down_oi_up_4h_rz"] = (
        ((-price_ret_4h_z).clip(lower=0.0) * oi_chg_4h_z.clip(lower=0.0))
        .clip(0.0, 100.0)
        .astype(np.float32)
    )
    out["price_up_oi_down_4h_rz"] = (
        (price_ret_4h_z.clip(lower=0.0) * (-oi_chg_4h_z).clip(lower=0.0))
        .clip(0.0, 100.0)
        .astype(np.float32)
    )
    out["price_up_oi_up_4h_rz"] = (
        (price_ret_4h_z.clip(lower=0.0) * oi_chg_4h_z.clip(lower=0.0))
        .clip(0.0, 100.0)
        .astype(np.float32)
    )

    _, price_rec_24h, price_low_24h, _ = _rolling_lifecycle_frames(
        price_log, price_ret_1h, h_24h
    )
    _, price_rec_72h, _, _ = _rolling_lifecycle_frames(price_log, price_ret_1h, h_72h)
    out["price_recovery_fraction_24h"] = price_rec_24h.clip(-0.5, 1.5).astype(
        np.float32
    )
    out["price_recovery_fraction_72h"] = price_rec_72h.clip(-0.5, 1.5).astype(
        np.float32
    )
    out["price_minus_oi_recovery_24h"] = (
        (price_rec_24h - oi_rec_24h).clip(-2.0, 2.0).astype(np.float32)
    )
    out["price_minus_oi_recovery_72h"] = (
        (price_rec_72h - oi_rec_72h).clip(-2.0, 2.0).astype(np.float32)
    )
    out["price_recovery_oi_still_falling_1h"] = (
        (price_rec_24h.clip(lower=0.0) * (-oi_chg_1h_z).clip(lower=0.0))
        .clip(0.0, 20.0)
        .astype(np.float32)
    )
    out["price_recovery_oi_still_falling_4h"] = (
        (price_rec_24h.clip(lower=0.0) * (-oi_chg_4h_z).clip(lower=0.0))
        .clip(0.0, 20.0)
        .astype(np.float32)
    )

    long_flush_intensity_4h = out["price_down_oi_down_4h_rz"]
    short_build_intensity_4h = out["price_down_oi_up_4h_rz"]
    short_cover_intensity_1h = out["price_up_oi_down_1h_rz"]
    out["mkt_pct_price_down_oi_down_1h"] = _broadcast_clean(
        ((price_ret_1h_z < 0.0) & (oi_chg_1h_z < 0.0))
        .where(price_ret_1h_z.notna() & oi_chg_1h_z.notna())
        .mean(axis=1, skipna=True),
        oi_native,
        clip=(0.0, 1.0),
    )
    out["mkt_pct_price_down_oi_up_1h"] = _broadcast_clean(
        ((price_ret_1h_z < 0.0) & (oi_chg_1h_z > 0.0))
        .where(price_ret_1h_z.notna() & oi_chg_1h_z.notna())
        .mean(axis=1, skipna=True),
        oi_native,
        clip=(0.0, 1.0),
    )
    out["mkt_pct_price_up_oi_down_1h"] = _broadcast_clean(
        ((price_ret_1h_z > 0.0) & (oi_chg_1h_z < 0.0))
        .where(price_ret_1h_z.notna() & oi_chg_1h_z.notna())
        .mean(axis=1, skipna=True),
        oi_native,
        clip=(0.0, 1.0),
    )
    out["mkt_pct_price_up_oi_up_1h"] = _broadcast_clean(
        ((price_ret_1h_z > 0.0) & (oi_chg_1h_z > 0.0))
        .where(price_ret_1h_z.notna() & oi_chg_1h_z.notna())
        .mean(axis=1, skipna=True),
        oi_native,
        clip=(0.0, 1.0),
    )
    out["mkt_pct_price_down_oi_down_4h"] = _broadcast_clean(
        ((price_ret_4h_z < 0.0) & (oi_chg_4h_z < 0.0))
        .where(price_ret_4h_z.notna() & oi_chg_4h_z.notna())
        .mean(axis=1, skipna=True),
        oi_native,
        clip=(0.0, 1.0),
    )
    out["mkt_pct_price_down_oi_up_4h"] = _broadcast_clean(
        ((price_ret_4h_z < 0.0) & (oi_chg_4h_z > 0.0))
        .where(price_ret_4h_z.notna() & oi_chg_4h_z.notna())
        .mean(axis=1, skipna=True),
        oi_native,
        clip=(0.0, 1.0),
    )
    out["mkt_pct_price_up_oi_down_4h"] = _broadcast_clean(
        ((price_ret_4h_z > 0.0) & (oi_chg_4h_z < 0.0))
        .where(price_ret_4h_z.notna() & oi_chg_4h_z.notna())
        .mean(axis=1, skipna=True),
        oi_native,
        clip=(0.0, 1.0),
    )
    out["mkt_pct_price_up_oi_up_4h"] = _broadcast_clean(
        ((price_ret_4h_z > 0.0) & (oi_chg_4h_z > 0.0))
        .where(price_ret_4h_z.notna() & oi_chg_4h_z.notna())
        .mean(axis=1, skipna=True),
        oi_native,
        clip=(0.0, 1.0),
    )
    out["mkt_median_long_flush_intensity_4h"] = _broadcast_clean(
        cross_sectional_median(long_flush_intensity_4h), oi_native, clip=(0.0, 100.0)
    )
    out["mkt_median_short_build_intensity_4h"] = _broadcast_clean(
        cross_sectional_median(short_build_intensity_4h), oi_native, clip=(0.0, 100.0)
    )
    out["mkt_median_short_cover_intensity_1h"] = _broadcast_clean(
        cross_sectional_median(short_cover_intensity_1h), oi_native, clip=(0.0, 100.0)
    )

    mkt_oi_chg_4h_rz_med = cross_sectional_median(oi_chg_4h_z)
    mkt_oi_rec_24h_med = cross_sectional_median(oi_rec_24h)
    mkt_price_rec_24h_med = cross_sectional_median(price_rec_24h)
    loo_oi_chg_1h_rz_med = leave_one_out_cross_sectional_median(oi_chg_1h_z)
    loo_oi_chg_4h_rz_med = leave_one_out_cross_sectional_median(oi_chg_4h_z)
    loo_oi_dd_24h_med = leave_one_out_cross_sectional_median(oi_dd_24h)
    loo_oi_rec_24h_med = leave_one_out_cross_sectional_median(oi_rec_24h)
    loo_price_rec_24h_med = leave_one_out_cross_sectional_median(price_rec_24h)
    loo_long_flush_4h_med = leave_one_out_cross_sectional_median(
        long_flush_intensity_4h
    )
    loo_short_cover_1h_med = leave_one_out_cross_sectional_median(
        short_cover_intensity_1h
    )
    loo_oi_drop_24h_med = leave_one_out_cross_sectional_median(oi_drop_24h)
    out["asset_minus_mkt_oi_chg_1h_rz"] = (
        oi_chg_1h_z.sub(loo_oi_chg_1h_rz_med).clip(-20.0, 20.0).astype(np.float32)
    )
    out["asset_minus_mkt_oi_chg_4h_rz"] = (
        oi_chg_4h_z.sub(loo_oi_chg_4h_rz_med).clip(-20.0, 20.0).astype(np.float32)
    )
    out["asset_minus_mkt_oi_drawdown_24h"] = (
        oi_dd_24h.sub(loo_oi_dd_24h_med).clip(-10.0, 10.0).astype(np.float32)
    )
    out["asset_minus_mkt_oi_recovery_fraction_24h"] = (
        oi_rec_24h.sub(loo_oi_rec_24h_med).clip(-2.0, 2.0).astype(np.float32)
    )
    out["asset_minus_mkt_price_recovery_fraction_24h"] = (
        price_rec_24h.sub(loo_price_rec_24h_med).clip(-2.0, 2.0).astype(np.float32)
    )
    out["asset_minus_mkt_long_flush_intensity_4h"] = (
        long_flush_intensity_4h.sub(loo_long_flush_4h_med)
        .clip(-100.0, 100.0)
        .astype(np.float32)
    )
    out["asset_minus_mkt_short_cover_intensity_1h"] = (
        short_cover_intensity_1h.sub(loo_short_cover_1h_med)
        .clip(-100.0, 100.0)
        .astype(np.float32)
    )
    out["asset_minus_mkt_bars_since_oi_flush_24h"] = (
        oi_drop_24h.sub(loo_oi_drop_24h_med).clip(-1.0, 1.0).astype(np.float32)
    )

    mkt_ret_1h = cross_sectional_mean(price_ret_1h)
    mkt_ret_4h = cross_sectional_mean(price_ret_4h)

    def _quadrant(
        prefix: str, ret: pd.Series, oi_chg: pd.Series, ret_up: bool, oi_up: bool
    ) -> pd.DataFrame:
        ret_part = ret.clip(lower=0.0) if ret_up else (-ret).clip(lower=0.0)
        oi_part = oi_chg.clip(lower=0.0) if oi_up else (-oi_chg).clip(lower=0.0)
        return _broadcast_clean(
            (ret_part * oi_part).replace([np.inf, -np.inf], np.nan),
            oi_native,
            clip=(0.0, 2.0),
        )

    out["mkt_price_down_oi_down_1h"] = _quadrant(
        "1h", mkt_ret_1h, mkt_oi_chg_1h, False, False
    )
    out["mkt_price_down_oi_down_4h"] = _quadrant(
        "4h", mkt_ret_4h, mkt_oi_chg_4h, False, False
    )
    out["mkt_price_down_oi_up_1h"] = _quadrant(
        "1h", mkt_ret_1h, mkt_oi_chg_1h, False, True
    )
    out["mkt_price_down_oi_up_4h"] = _quadrant(
        "4h", mkt_ret_4h, mkt_oi_chg_4h, False, True
    )
    out["mkt_price_up_oi_down_1h"] = _quadrant(
        "1h", mkt_ret_1h, mkt_oi_chg_1h, True, False
    )
    out["mkt_price_up_oi_down_4h"] = _quadrant(
        "4h", mkt_ret_4h, mkt_oi_chg_4h, True, False
    )
    out["mkt_price_up_oi_up_1h"] = _quadrant(
        "1h", mkt_ret_1h, mkt_oi_chg_1h, True, True
    )
    out["mkt_price_up_oi_up_4h"] = _quadrant(
        "4h", mkt_ret_4h, mkt_oi_chg_4h, True, True
    )

    out["pct_assets_price_down_oi_down_1h"] = _broadcast_clean(
        ((price_ret_1h < 0.0) & (oi_chg_1h < 0.0))
        .where(price_ret_1h.notna() & oi_chg_1h.notna())
        .mean(axis=1, skipna=True),
        oi_native,
        clip=(0.0, 1.0),
    )
    out["pct_assets_price_down_oi_down_4h"] = _broadcast_clean(
        ((price_ret_4h < 0.0) & (oi_chg_4h < 0.0))
        .where(price_ret_4h.notna() & oi_chg_4h.notna())
        .mean(axis=1, skipna=True),
        oi_native,
        clip=(0.0, 1.0),
    )
    out["pct_assets_price_down_oi_up_1h"] = _broadcast_clean(
        ((price_ret_1h < 0.0) & (oi_chg_1h > 0.0))
        .where(price_ret_1h.notna() & oi_chg_1h.notna())
        .mean(axis=1, skipna=True),
        oi_native,
        clip=(0.0, 1.0),
    )
    out["pct_assets_price_up_oi_down_1h"] = _broadcast_clean(
        ((price_ret_1h > 0.0) & (oi_chg_1h < 0.0))
        .where(price_ret_1h.notna() & oi_chg_1h.notna())
        .mean(axis=1, skipna=True),
        oi_native,
        clip=(0.0, 1.0),
    )
    eps = np.float32(1e-6)
    out["mkt_abs_ret_per_oi_drop_1h"] = _broadcast_clean(
        (mkt_ret_1h.abs() / ((-mkt_oi_chg_1h).clip(lower=0.0) + eps)),
        oi_native,
        clip=(0.0, 50.0),
    )
    out["mkt_abs_ret_per_oi_drop_4h"] = _broadcast_clean(
        (mkt_ret_4h.abs() / ((-mkt_oi_chg_4h).clip(lower=0.0) + eps)),
        oi_native,
        clip=(0.0, 50.0),
    )
    out["mkt_ret_per_oi_change_1h"] = _broadcast_clean(
        (mkt_ret_1h / (mkt_oi_chg_1h.abs() + eps)), oi_native, clip=(-50.0, 50.0)
    )
    out["mkt_ret_per_oi_change_4h"] = _broadcast_clean(
        (mkt_ret_4h / (mkt_oi_chg_4h.abs() + eps)), oi_native, clip=(-50.0, 50.0)
    )

    if funding_rate is not None:
        mkt_funding_mean = funding_rate.mean(axis=1, skipna=True).astype(np.float32)
        mkt_funding_median = funding_rate.median(axis=1, skipna=True).astype(np.float32)
        mkt_funding_abs_mean = (
            funding_rate.abs().mean(axis=1, skipna=True).astype(np.float32)
        )
        funding_weighted = (
            (funding_rate * oi_value).sum(axis=1, min_count=1)
            / mkt_oi_value.replace(0.0, np.nan)
        ).astype(np.float32)
        funding_dispersion = funding_rate.std(axis=1, skipna=True, ddof=0).astype(
            np.float32
        )
        funding_skew = (
            funding_rate.skew(axis=1, skipna=True)
            .replace([np.inf, -np.inf], np.nan)
            .astype(np.float32)
        )
        funding_mean_z_30d = _rolling_series_robust_z(
            mkt_funding_mean, w_30d, min_periods=w_7d
        ).clip(-10, 10)
        funding_dispersion_z_30d = _rolling_series_robust_z(
            funding_dispersion, w_30d, min_periods=w_7d
        ).clip(-10, 10)
        out["mkt_funding_mean"] = _broadcast_clean(
            mkt_funding_mean, oi_native, clip=(-0.10, 0.10)
        )
        out["mkt_funding_median"] = _broadcast_clean(
            mkt_funding_median, oi_native, clip=(-0.10, 0.10)
        )
        out["mkt_funding_abs_mean"] = _broadcast_clean(
            mkt_funding_abs_mean, oi_native, clip=(0.0, 0.10)
        )
        out["mkt_funding_weighted_by_oi"] = _broadcast_clean(
            funding_weighted, oi_native, clip=(-0.10, 0.10)
        )
        out["mkt_funding_chg_1h"] = _broadcast_clean(
            mkt_funding_mean - mkt_funding_mean.shift(h_1h),
            oi_native,
            clip=(-0.10, 0.10),
        )
        out["mkt_funding_chg_4h"] = _broadcast_clean(
            mkt_funding_mean - mkt_funding_mean.shift(h_4h),
            oi_native,
            clip=(-0.10, 0.10),
        )
        out["mkt_funding_chg_24h"] = _broadcast_clean(
            mkt_funding_mean - mkt_funding_mean.shift(h_24h),
            oi_native,
            clip=(-0.10, 0.10),
        )
        out["mkt_funding_accel_4h"] = _broadcast_clean(
            (mkt_funding_mean - mkt_funding_mean.shift(h_4h))
            - (mkt_funding_mean.shift(h_4h) - mkt_funding_mean.shift(2 * h_4h)),
            oi_native,
            clip=(-0.10, 0.10),
        )
        out["pct_assets_positive_funding"] = _broadcast_clean(
            funding_rate.gt(0.0).where(funding_rate.notna()).mean(axis=1, skipna=True),
            oi_native,
            clip=(0.0, 1.0),
        )
        out["pct_assets_negative_funding"] = _broadcast_clean(
            funding_rate.lt(0.0).where(funding_rate.notna()).mean(axis=1, skipna=True),
            oi_native,
            clip=(0.0, 1.0),
        )
        out["pct_assets_extreme_positive_funding"] = _broadcast_clean(
            funding_z_90d.gt(2.0)
            .where(funding_z_90d.notna())
            .mean(axis=1, skipna=True),
            oi_native,
            clip=(0.0, 1.0),
        )
        out["pct_assets_extreme_negative_funding"] = _broadcast_clean(
            funding_z_90d.lt(-2.0)
            .where(funding_z_90d.notna())
            .mean(axis=1, skipna=True),
            oi_native,
            clip=(0.0, 1.0),
        )
        out["mkt_funding_dispersion"] = _broadcast_clean(
            funding_dispersion, oi_native, clip=(0.0, 0.10)
        )
        out["mkt_funding_skew"] = _broadcast_clean(
            funding_skew, oi_native, clip=(-10.0, 10.0)
        )
        out["mkt_funding_tail_concentration"] = _broadcast_clean(
            funding_z_90d.abs()
            .gt(2.0)
            .where(funding_z_90d.notna())
            .mean(axis=1, skipna=True),
            oi_native,
            clip=(0.0, 1.0),
        )
        out["mkt_funding_mean_z_30d"] = _broadcast_clean(
            funding_mean_z_30d, oi_native, clip=(-10.0, 10.0)
        )
        out["mkt_funding_dispersion_z_30d"] = _broadcast_clean(
            funding_dispersion_z_30d, oi_native, clip=(-10.0, 10.0)
        )

        pos_funding = mkt_funding_mean.clip(lower=0.0)
        neg_funding = (-mkt_funding_mean).clip(lower=0.0)
        mkt_ret1_std_4h = mkt_ret_1h.rolling(h_4h, min_periods=max(2, h_1h)).std(ddof=0)
        mkt_ret1_std_24h = mkt_ret_1h.rolling(h_24h, min_periods=max(h_4h, 2)).std(
            ddof=0
        )
        vol_expansion = (
            (mkt_ret1_std_4h / (mkt_ret1_std_24h + eps))
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(0.0, 10.0)
        )
        funding_flip = (
            np.sign(mkt_funding_mean) != np.sign(mkt_funding_mean.shift(h_1h))
        ).astype(np.float32)
        funding_abs_chg = mkt_funding_abs_mean.shift(h_1h) - mkt_funding_abs_mean
        out["positive_funding_x_price_down"] = _broadcast_clean(
            pos_funding * (-mkt_ret_1h).clip(lower=0.0), oi_native, clip=(0.0, 1.0)
        )
        out["positive_funding_x_oi_drop"] = _broadcast_clean(
            pos_funding * (-mkt_oi_chg_1h).clip(lower=0.0), oi_native, clip=(0.0, 1.0)
        )
        out["negative_funding_x_price_up"] = _broadcast_clean(
            neg_funding * mkt_ret_1h.clip(lower=0.0), oi_native, clip=(0.0, 1.0)
        )
        out["negative_funding_x_oi_drop"] = _broadcast_clean(
            neg_funding * (-mkt_oi_chg_1h).clip(lower=0.0), oi_native, clip=(0.0, 1.0)
        )
        out["funding_crowding_x_vol_expansion"] = _broadcast_clean(
            mkt_funding_abs_mean * vol_expansion, oi_native, clip=(0.0, 1.0)
        )
        out["funding_flip_x_oi_flush"] = _broadcast_clean(
            funding_flip * oi_flush_z.clip(lower=0.0), oi_native, clip=(0.0, 10.0)
        )
        out["funding_mean_reversion_after_oi_flush"] = _broadcast_clean(
            funding_abs_chg.clip(lower=0.0) * oi_flush_z.clip(lower=0.0),
            oi_native,
            clip=(0.0, 10.0),
        )
        funding_event = funding_rate.ne(funding_rate.shift(1)) & funding_rate.notna()
        funding_sign = (
            np.sign(funding_rate).where(funding_rate.notna()).astype(np.float32)
        )
        funding_event_sign = funding_sign.where(funding_event)
        for _label, _window in (("24h", h_24h), ("72h", h_72h)):
            event_count = (
                funding_event.astype(np.float32).rolling(_window, min_periods=1).sum()
            )
            sign_sum = (
                funding_event_sign.fillna(0.0).rolling(_window, min_periods=1).sum()
            )
            out[f"funding_sign_persistence_{_label}"] = (
                (sign_sum.abs() / (event_count.replace(0.0, np.nan) + eps))
                .replace([np.inf, -np.inf], np.nan)
                .clip(0.0, 1.0)
                .astype(np.float32)
            )
        sign_flip = funding_event & funding_sign.ne(
            funding_sign.where(funding_event).ffill().shift(1)
        )
        out["hours_since_funding_sign_flip_24h_norm"] = (
            _bars_since_true_frame(sign_flip, h_24h).clip(0.0, 1.0).astype(np.float32)
        )
        prev_event_funding_z = funding_z_90d.where(funding_event).ffill().shift(1)
        out["funding_positive_to_negative_intensity"] = (
            (prev_event_funding_z.clip(lower=0.0) * (-funding_z_90d).clip(lower=0.0))
            .clip(0.0, 100.0)
            .astype(np.float32)
        )
        out["funding_negative_to_positive_intensity"] = (
            ((-prev_event_funding_z).clip(lower=0.0) * funding_z_90d.clip(lower=0.0))
            .clip(0.0, 100.0)
            .astype(np.float32)
        )
        out["funding_crowding_release_4h"] = (
            (
                funding_z_90d.shift(h_4h).clip(lower=0.0)
                * (funding_z_90d.shift(h_4h) - funding_z_90d).clip(lower=0.0)
                * (-oi_chg_4h_z).clip(lower=0.0)
            )
            .clip(0.0, 100.0)
            .astype(np.float32)
        )
    else:
        for key in (
            MARKET_FUNDING_REGIME_FEATURE_KEYS
            + FUNDING_PRICE_OI_INTERACTION_FEATURE_KEYS
            + FUNDING_LIFECYCLE_FEATURE_KEYS
        ):
            out[key] = pd.DataFrame(
                np.nan,
                index=oi_native.index,
                columns=oi_native.columns,
                dtype=np.float32,
            )

    asset_ret_rel_4h = price_ret_4h_z.sub(
        cross_sectional_median(price_ret_4h_z), axis=0
    )
    volume_z_7d = rolling_robust_zscore_by_symbol(
        np.log1p(quote_volume).astype(np.float32), w_7d, min_periods=max(h_24h, h_1h)
    ).clip(-10, 10)
    funding_rz_30d = rolling_robust_zscore_by_symbol(
        funding_rate
        if funding_rate is not None
        else pd.DataFrame(np.nan, index=oi_native.index, columns=oi_native.columns),
        w_30d,
        min_periods=w_7d,
    ).clip(-10, 10)
    # Funding is informative when available, but some otherwise tradable Kraken
    # perps have no usable funding history. Treat absent funding as neutral in
    # these composite state scores instead of invalidating their OI/OHLCV signal.
    funding_rz_30d_neutral = funding_rz_30d.fillna(0.0)
    asset_liq = (
        (-price_ret_4h_z).clip(lower=0.0).fillna(0.0)
        + (-oi_chg_4h_z).clip(lower=0.0).fillna(0.0)
        + volume_z_7d.clip(lower=0.0).fillna(0.0)
        + funding_rz_30d_neutral.clip(lower=0.0)
        + (-asset_ret_rel_4h).clip(lower=0.0).fillna(0.0)
    ) / np.float32(5.0)
    asset_flush = (
        (-oi_dd_24h / np.float32(0.10)).clip(0.0, 3.0)
        + out["oi_drop_deceleration_4h_rz"].clip(lower=0.0)
        + price_rec_24h.clip(lower=0.0)
        + out.get(
            "price_up_oi_down_1h_rz",
            pd.DataFrame(0.0, index=oi_native.index, columns=oi_native.columns),
        ).clip(lower=0.0)
    ) / np.float32(4.0)
    asset_short_cover = (
        price_ret_1h_z.clip(lower=0.0).fillna(0.0)
        + (-oi_chg_1h_z).clip(lower=0.0).fillna(0.0)
        + (-funding_rz_30d_neutral).clip(lower=0.0)
        + price_rec_24h.clip(lower=0.0).fillna(0.0)
        + volume_z_7d.clip(lower=0.0).fillna(0.0)
    ) / np.float32(5.0)
    out["asset_liquidation_phase_score"] = asset_liq.clip(0.0, 20.0).astype(np.float32)
    out["asset_flush_exhaustion_score"] = asset_flush.clip(0.0, 20.0).astype(np.float32)
    out["asset_short_covering_score"] = asset_short_cover.clip(0.0, 20.0).astype(
        np.float32
    )
    median_asset_ret_4h_rz = cross_sectional_median(price_ret_4h_z)
    mkt_systemic = (
        (-median_asset_ret_4h_rz).clip(lower=0.0)
        + (-mkt_oi_chg_4h_rz_med).clip(lower=0.0)
        + flush_breadth_4h.fillna(0.0)
        + out["mkt_pct_price_down_oi_down_4h"].iloc[:, 0].fillna(0.0)
    ) / np.float32(4.0)
    median_downside_decel_4h = cross_sectional_median(
        (
            (-price_ret_4h_z.shift(h_4h)).clip(lower=0.0)
            - (-price_ret_4h_z).clip(lower=0.0)
        ).clip(lower=0.0)
    )
    mkt_flush = (
        out["mkt_pct_oi_chg_4h_rz_lt_minus2"].iloc[:, 0].fillna(0.0)
        + out["mkt_oi_flush_breadth_recovery_4h"].iloc[:, 0].fillna(0.0)
        + out["mkt_pct_price_up_oi_down_1h"].iloc[:, 0].fillna(0.0)
        + mkt_price_rec_24h_med.clip(lower=0.0)
        + median_downside_decel_4h.clip(lower=0.0)
    ) / np.float32(5.0)
    median_funding_chg_4h_rz = cross_sectional_median(
        rolling_robust_zscore_by_symbol(
            (funding_rate - funding_rate.shift(h_4h)).astype(np.float32),
            w_30d,
            min_periods=w_7d,
        ).clip(-10, 10)
        if funding_rate is not None
        else pd.DataFrame(np.nan, index=oi_native.index, columns=oi_native.columns)
    )
    mkt_rebuild = (
        mkt_oi_chg_4h_rz_med.clip(lower=0.0)
        + out["mkt_pct_price_up_oi_up_4h"].iloc[:, 0].fillna(0.0)
        + mkt_oi_rec_24h_med.clip(lower=0.0)
        + mkt_price_rec_24h_med.clip(lower=0.0)
        + median_funding_chg_4h_rz.clip(lower=0.0)
    ) / np.float32(5.0)
    out["mkt_systemic_deleveraging_score"] = _broadcast_clean(
        mkt_systemic, oi_native, clip=(0.0, 20.0)
    )
    out["mkt_flush_exhaustion_score"] = _broadcast_clean(
        mkt_flush, oi_native, clip=(0.0, 20.0)
    )
    out["mkt_leverage_rebuild_score"] = _broadcast_clean(
        mkt_rebuild, oi_native, clip=(0.0, 20.0)
    )
    out["asset_mkt_liquidation_phase_divergence"] = (
        (
            out["asset_liquidation_phase_score"].fillna(0.0)
            - out["mkt_systemic_deleveraging_score"].fillna(0.0)
        )
        .clip(-20.0, 20.0)
        .astype(np.float32)
    )
    out["asset_mkt_exhaustion_phase_divergence"] = (
        (out["asset_flush_exhaustion_score"] - out["mkt_flush_exhaustion_score"])
        .clip(-20.0, 20.0)
        .astype(np.float32)
    )

    for days, window in (("7d", w_7d), ("10d", 10 * bpd), ("15d", 15 * bpd)):
        trend = (price_log - price_log.shift(window)).astype(np.float32)
        trend_norm = trend / (
            ret_1h.abs().rolling(window, min_periods=max(1, window // 6)).sum()
            + np.float32(1e-12)
        )
        out[f"price_trend_{days}_vol_norm"] = (
            trend_norm.replace([np.inf, -np.inf], np.nan)
            .clip(-10, 10)
            .astype(np.float32)
        )
        price_rv = ret_1h.rolling(window, min_periods=max(1, window // 6)).std(ddof=0)
        out[f"price_rv_{days}_robust_z"] = (
            rolling_robust_zscore_by_symbol(
                np.log(price_rv.clip(lower=1e-12)),
                w_30d,
                min_periods=w_7d,
            )
            .clip(-10, 10)
            .astype(np.float32)
        )
        oi_trend = (oi_value_log - oi_value_log.shift(window)).astype(np.float32)
        out[f"oi_trend_{days}_robust_z"] = (
            rolling_robust_zscore_by_symbol(
                oi_trend,
                w_30d,
                min_periods=w_7d,
            )
            .clip(-10, 10)
            .astype(np.float32)
        )
        oi_vol = oi_1h_chg.rolling(window, min_periods=max(1, window // 6)).std(ddof=0)
        out[f"oi_vol_{days}_robust_z"] = (
            rolling_robust_zscore_by_symbol(
                np.log(oi_vol.clip(lower=1e-12)),
                w_30d,
                min_periods=w_7d,
            )
            .clip(-10, 10)
            .astype(np.float32)
        )
        if funding_rate is not None:
            funding_mean = funding_rate.rolling(
                window, min_periods=max(1, window // 6)
            ).mean()
            funding_vol = (
                funding_rate.diff(1)
                .rolling(window, min_periods=max(1, window // 6))
                .std(ddof=0)
            )
            out[f"funding_mean_{days}_robust_z"] = (
                rolling_robust_zscore_by_symbol(
                    funding_mean,
                    w_30d,
                    min_periods=w_7d,
                )
                .clip(-10, 10)
                .astype(np.float32)
            )
            out[f"funding_vol_{days}_robust_z"] = (
                rolling_robust_zscore_by_symbol(
                    np.log(funding_vol.clip(lower=1e-12)),
                    w_30d,
                    min_periods=w_7d,
                )
                .clip(-10, 10)
                .astype(np.float32)
            )
        else:
            out[f"funding_mean_{days}_robust_z"] = funding_z_90d.copy()
            out[f"funding_vol_{days}_robust_z"] = funding_z_90d.copy()

    return {k: v.astype(np.float32) for k, v in out.items()}
