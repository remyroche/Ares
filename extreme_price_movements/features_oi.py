"""Open-interest feature block for hourly perpetual futures panels."""

from __future__ import annotations

import numpy as np
import pandas as pd

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
    z = ((mu_short - mu_long) / (sigma_long + np.float32(eps))).replace(
        [np.inf, -np.inf], np.nan
    ).clip(-12.0, 12.0).fillna(0.0)
    logstd = (
        np.log(short_std.clip(lower=eps)) - np.log(long_std.clip(lower=eps))
    ).replace([np.inf, -np.inf], np.nan).clip(-8.0, 8.0).fillna(0.0)
    absratio = (mu_short.abs() / (mu_long.abs() + np.float32(eps))).replace(
        [np.inf, -np.inf], np.nan
    ).clip(0.0, 100.0).fillna(1.0)
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

    return frame.astype(np.float32).rolling(window, min_periods=min_periods).apply(
        rank_last, raw=True
    ).astype(np.float32)


def cross_sectional_mean(frame: pd.DataFrame) -> pd.Series:
    return frame.mean(axis=1, skipna=True).astype(np.float32)


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
    oi_value = (oi_native * price).replace([np.inf, -np.inf], np.nan).where(
        lambda f: f > 0.0
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
        z = rolling_long_iqr_robust_zscore_by_symbol(ret, w_90d).clip(-10, 10).astype(np.float32)
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
        z = rolling_long_iqr_robust_zscore_by_symbol(log_chg[label], window).clip(-10, 10)
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
    oi_value_z_90d = rolling_long_iqr_robust_zscore_by_symbol(oi_value_log, w_90d).clip(-10, 10)
    out["oi_value_z_30d"] = oi_value_z_30d.astype(np.float32)
    out["oi_value_z_90d"] = oi_value_z_90d.astype(np.float32)
    out["oi_value_log_z_30d"] = oi_value_z_30d.astype(np.float32)
    out["oi_value_log_z_90d"] = oi_value_z_90d.astype(np.float32)
    out["oi_value_pct_90d"] = rolling_percentile_by_symbol(
        oi_value_log, w_90d
    ).astype(np.float32)

    log_oi_to_volume_1d = np.log1p(oi_to_volume_1d.clip(lower=0.0)).replace(
        [np.inf, -np.inf], np.nan
    )
    log_oi_to_volume_7d = np.log1p(oi_to_volume_7d.clip(lower=0.0)).replace(
        [np.inf, -np.inf], np.nan
    )
    out["log_oi_to_volume_1d"] = log_oi_to_volume_1d.astype(np.float32)
    out["log_oi_to_volume_7d"] = log_oi_to_volume_7d.astype(np.float32)
    out["oi_to_volume_1d_z_90d"] = rolling_long_iqr_robust_zscore_by_symbol(
        log_oi_to_volume_1d, w_90d
    ).clip(-10, 10).astype(np.float32)
    out["oi_to_volume_7d_z_180d"] = rolling_long_iqr_robust_zscore_by_symbol(
        log_oi_to_volume_7d, w_180d
    ).clip(-10, 10).astype(np.float32)
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
    out["oi_value_log_1d_robust_z"] = rolling_robust_zscore_by_symbol(
        oi_value_log_1d,
        w_30d,
        min_periods=w_7d,
    ).clip(-10, 10).astype(np.float32)
    out["oi_value_log_7d_robust_z"] = rolling_robust_zscore_by_symbol(
        oi_value_log_7d,
        w_30d,
        min_periods=w_7d,
    ).clip(-10, 10).astype(np.float32)

    for hours in (2, 4, 8):
        shift = hours * bpd // 24
        chg = (oi_value_log - oi_value_log.shift(max(1, shift))).astype(np.float32)
        out[f"oi_chg_{hours}h_robust_z"] = rolling_robust_zscore_by_symbol(
            chg,
            w_30d,
            min_periods=w_7d,
        ).clip(-10, 10).astype(np.float32)

    for label in ("1d", "3d", "7d"):
        out[f"price_x_oi_{label}"] = (
            price_ret_z[label] * oi_chg_z[label]
        ).astype(np.float32)
        out[f"oi_{label}_x_funding"] = (
            oi_chg_z[label] * funding_z_90d
        ).astype(np.float32)
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
    out["asset_minus_mkt_oi_1d_z_90d"] = rolling_long_iqr_robust_zscore_by_symbol(
        asset_minus_mkt_oi_1d, w_90d
    ).clip(-10, 10).astype(np.float32)
    out["asset_minus_mkt_oi_7d_z_180d"] = rolling_long_iqr_robust_zscore_by_symbol(
        asset_minus_mkt_oi_7d, w_180d
    ).clip(-10, 10).astype(np.float32)
    out.update(
        _short_long_change_point_features(
            asset_minus_mkt_oi_1d,
            prefix="asset_minus_mkt_oi_1d",
        )
    )

    out["mkt_oi_z_30d"] = _broadcast(cross_sectional_mean(oi_value_z_30d), oi_native)
    out["mkt_oi_chg_z_24h"] = _broadcast(cross_sectional_mean(oi_chg_z["1d"]), oi_native)
    oi_rising_1d = (log_chg["1d"] > 0.0).astype(np.float32).where(
        log_chg["1d"].notna()
    )
    out["mkt_oi_breadth_rising_24h"] = _broadcast(
        cross_sectional_mean(oi_rising_1d), oi_native
    )
    out["mkt_oi_dispersion_24h"] = _broadcast(cross_sectional_std(oi_chg_z["1d"]), oi_native)
    out["cs_rank_oi_value_z_30d"] = cross_sectional_rank_pct(oi_value_z_30d)
    out["cs_rank_oi_chg_1d_z_90d"] = cross_sectional_rank_pct(oi_chg_z["1d"])

    ret_1h = price_log.diff(1).astype(np.float32)
    oi_1h_chg = oi_value_log.diff(1).astype(np.float32)
    for days, window in (("7d", w_7d), ("10d", 10 * bpd), ("15d", 15 * bpd)):
        trend = (price_log - price_log.shift(window)).astype(np.float32)
        trend_norm = trend / (
            ret_1h.abs().rolling(window, min_periods=max(1, window // 6)).sum()
            + np.float32(1e-12)
        )
        out[f"price_trend_{days}_vol_norm"] = (
            trend_norm.replace([np.inf, -np.inf], np.nan).clip(-10, 10).astype(np.float32)
        )
        price_rv = ret_1h.rolling(window, min_periods=max(1, window // 6)).std(ddof=0)
        out[f"price_rv_{days}_robust_z"] = rolling_robust_zscore_by_symbol(
            np.log(price_rv.clip(lower=1e-12)),
            w_30d,
            min_periods=w_7d,
        ).clip(-10, 10).astype(np.float32)
        oi_trend = (oi_value_log - oi_value_log.shift(window)).astype(np.float32)
        out[f"oi_trend_{days}_robust_z"] = rolling_robust_zscore_by_symbol(
            oi_trend,
            w_30d,
            min_periods=w_7d,
        ).clip(-10, 10).astype(np.float32)
        oi_vol = oi_1h_chg.rolling(window, min_periods=max(1, window // 6)).std(ddof=0)
        out[f"oi_vol_{days}_robust_z"] = rolling_robust_zscore_by_symbol(
            np.log(oi_vol.clip(lower=1e-12)),
            w_30d,
            min_periods=w_7d,
        ).clip(-10, 10).astype(np.float32)
        if funding_rate is not None:
            funding_mean = funding_rate.rolling(window, min_periods=max(1, window // 6)).mean()
            funding_vol = funding_rate.diff(1).rolling(
                window, min_periods=max(1, window // 6)
            ).std(ddof=0)
            out[f"funding_mean_{days}_robust_z"] = rolling_robust_zscore_by_symbol(
                funding_mean,
                w_30d,
                min_periods=w_7d,
            ).clip(-10, 10).astype(np.float32)
            out[f"funding_vol_{days}_robust_z"] = rolling_robust_zscore_by_symbol(
                np.log(funding_vol.clip(lower=1e-12)),
                w_30d,
                min_periods=w_7d,
            ).clip(-10, 10).astype(np.float32)
        else:
            out[f"funding_mean_{days}_robust_z"] = funding_z_90d.copy()
            out[f"funding_vol_{days}_robust_z"] = funding_z_90d.copy()

    return {k: v.astype(np.float32) for k, v in out.items()}
