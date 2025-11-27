import numpy as np
import pandas as pd
from typing import Any, Dict

from src.feature_generation.categories.cross_timeframe import CrossTimeframeFeatureGenerator


def generate_smc_regime_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Generate all core SMC features (without normalization).
    
    OPTIMIZED:
    - Uses float32 for 50% memory reduction
    - Vectorized ATR calculation with numpy
    - EWM instead of rolling for ATR (O(1) vs O(window))
    """
    result = df.copy()

    required_cols = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in result.columns]
    if missing:
        raise ValueError(f"Missing columns for SMC features: {missing}")

    if not isinstance(result.index, pd.DatetimeIndex):
        result.index = pd.to_datetime(result.index)
    result = result.sort_index()

    # OPTIMIZED: Use float32 for memory efficiency
    o = result["open"].astype(np.float32)
    h = result["high"].astype(np.float32)
    l = result["low"].astype(np.float32)
    c = result["close"].astype(np.float32)
    v = result["volume"].astype(np.float32)

    # OPTIMIZED: Vectorized True Range calculation with numpy
    h_arr = h.values
    l_arr = l.values
    c_arr = c.values
    c_prev = np.roll(c_arr, 1)
    c_prev[0] = c_arr[0]
    
    tr1 = h_arr - l_arr
    tr2 = np.abs(h_arr - c_prev)
    tr3 = np.abs(l_arr - c_prev)
    true_range = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # OPTIMIZED: EWM for ATR (O(1) per point vs O(window) for rolling)
    atr_window = int(config.get("smc_atr_window", 14))
    atr = pd.Series(true_range, index=result.index).ewm(span=atr_window, adjust=False).mean()
    result["atr"] = atr.astype(np.float32)

    result = _add_liquidity_features(result, config)
    result = _add_fvg_features(result, config)
    result = _add_premium_discount_features(result, config)
    result = _add_momentum_features(result, config)
    result = _add_volatility_time_features(result, config)
    result = _add_mtf_features(result, config)
    result = _add_volume_profile_features(result, config)
    result = _add_time_categories(result, config)

    if bool(config.get("smc_enable_cross_timeframe_features", True)):
        result = _add_cross_timeframe_features(result)

    return result


def _add_liquidity_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    atr = df["atr"].astype(float)

    day_index = df.index.normalize()
    daily = df.groupby(day_index).agg(high=("high", "max"), low=("low", "min"), open=("open", "first"))
    prev_day_high = daily["high"].shift(1)
    prev_day_low = daily["low"].shift(1)
    day_open = daily["open"]

    pdh = prev_day_high.reindex(day_index).to_numpy()
    pdl = prev_day_low.reindex(day_index).to_numpy()
    day_open_vals = day_open.reindex(day_index).to_numpy()

    df["smc_pdh"] = pdh
    df["smc_pdl"] = pdl
    df["smc_dist_to_pdh_atr"] = (c.values - pdh) / (atr.values + 1e-9)
    df["smc_dist_to_pdl_atr"] = (c.values - pdl) / (atr.values + 1e-9)

    df["smc_day_open"] = day_open_vals
    df["smc_dist_to_day_open"] = (c.values - day_open_vals) / (atr.values + 1e-9)

    week_index = df.index.to_period("W").to_timestamp()
    weekly = df.groupby(week_index).agg(open=("open", "first"), close=("close", "last"))
    week_open = weekly["open"]
    prev_week_close = weekly["close"].shift(1)

    week_open_vals = week_open.reindex(week_index).to_numpy()
    prev_week_close_vals = prev_week_close.reindex(week_index).to_numpy()

    df["smc_week_open"] = week_open_vals
    df["smc_dist_to_week_open"] = (c.values - week_open_vals) / (atr.values + 1e-9)

    nwog_gap = week_open_vals - prev_week_close_vals
    df["smc_nwog_gap_size"] = nwog_gap / (atr.values + 1e-9)

    return df


def _add_fvg_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    o = df["open"].astype(float)
    atr = df["atr"].astype(float)

    high_2 = h.shift(2)
    low_2 = l.shift(2)

    bullish_fvg = l > high_2
    bearish_fvg = h < low_2

    bullish_fvg_size = (l - high_2).clip(lower=0.0)
    bearish_fvg_size = (low_2 - h).clip(lower=0.0)

    fvg_size = bullish_fvg_size.where(bullish_fvg, bearish_fvg_size.where(bearish_fvg, 0.0))
    df["smc_current_fvg_size"] = fvg_size / (atr + 1e-9)

    bullish_fvg_mid = (l + high_2) / 2.0
    bearish_fvg_mid = (h + low_2) / 2.0

    fvg_mid = bullish_fvg_mid.where(bullish_fvg, bearish_fvg_mid.where(bearish_fvg, np.nan))
    fvg_mid_filled = fvg_mid.ffill()
    df["smc_nearest_fvg_dist"] = (c - fvg_mid_filled) / (atr + 1e-9)

    fvg_high = l.where(bullish_fvg, low_2.where(bearish_fvg, np.nan))
    fvg_low = high_2.where(bullish_fvg, h.where(bearish_fvg, np.nan))

    fvg_range = fvg_high - fvg_low
    ce_position = (c - fvg_low) / (fvg_range + 1e-9)
    df["smc_consequent_encroachment"] = ce_position.fillna(0.5)

    volume_imb = (o - c.shift(1)).abs()
    df["smc_volume_imbalance_size"] = volume_imb / (atr + 1e-9)

    fvg_fill = ((h - fvg_low) / (fvg_range + 1e-9)).clip(0.0, 1.0)
    df["smc_gap_fill_ratio"] = fvg_fill.fillna(0.0)

    return df


def _add_premium_discount_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    atr = df["atr"].astype(float)

    sh = (
        (h.shift(2) < h.shift(1))
        & (h.shift(1) < h)
        & (h > h.shift(-1))
        & (h.shift(-1) > h.shift(-2))
    )
    sl = (
        (l.shift(2) > l.shift(1))
        & (l.shift(1) > l)
        & (l < l.shift(-1))
        & (l.shift(-1) < l.shift(-2))
    )

    swing_high = h.where(sh).ffill()
    swing_low = l.where(sl).ffill()

    range_height = swing_high - swing_low
    range_pos = (c - swing_low) / (range_height + 1e-9)
    df["smc_range_position"] = range_pos.clip(0.0, 1.0)

    df["smc_dist_to_swing_high"] = (swing_high - c) / (atr + 1e-9)
    df["smc_dist_to_swing_low"] = (c - swing_low) / (atr + 1e-9)

    fib_level = (swing_high - c) / (range_height + 1e-9)
    df["smc_fib_retracement_level"] = fib_level.clip(0.0, 1.0)

    prev_swing_high = swing_high.shift(1)
    bos_magnitude = (c - prev_swing_high) / (atr + 1e-9)
    df["smc_break_of_structure_mag"] = bos_magnitude.clip(lower=0.0)

    return df


def _add_momentum_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)

    body = (c - o).abs()
    avg_body = body.rolling(window=20).mean()

    df["smc_displacement_strength"] = body / (avg_body + 1e-9)

    upper_wick = h - np.maximum(c, o)
    lower_wick = np.minimum(c, o) - l
    total_wick = upper_wick + lower_wick
    df["smc_wick_body_ratio"] = total_wick / (body + 1e-9)

    candle_range = h - l
    close_pos = (c - l) / (candle_range + 1e-9)
    df["smc_close_position_in_candle"] = close_pos.clip(0.0, 1.0)

    roc = (c - c.shift(3)) / 3.0
    df["smc_velocity_roc"] = roc / (c.shift(3) + 1e-9)

    candle_direction = np.sign(c - o)
    streaks = np.zeros(len(candle_direction))
    current_streak = 0
    for i in range(len(candle_direction)):
        if i == 0:
            current_streak = candle_direction.iloc[i]
        elif candle_direction.iloc[i] == candle_direction.iloc[i - 1] and candle_direction.iloc[i] != 0:
            current_streak += candle_direction.iloc[i]
        else:
            current_streak = candle_direction.iloc[i]
        streaks[i] = current_streak

    df["smc_consecutive_candles"] = streaks

    return df


def _add_volatility_time_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v = df["volume"].astype(float)
    atr = df["atr"].astype(float)

    day_index = df.index.normalize()
    daily_range = df.groupby(day_index).apply(lambda x: x["high"].max() - x["low"].min())
    adr = daily_range.rolling(window=20).mean()

    today_high = df.groupby(day_index)["high"].transform("max")
    today_low = df.groupby(day_index)["low"].transform("min")
    today_range = today_high - today_low
    adr_reindexed = adr.reindex(day_index).to_numpy()
    df["smc_adr_filled_pct"] = today_range / (adr_reindexed + 1e-9)

    avg_vol = v.rolling(window=20).mean()
    df["smc_rel_volume"] = v / (avg_vol + 1e-9)

    df["smc_time_elapsed_session"] = df.index.hour * 60 + df.index.minute

    atr_20 = atr.rolling(window=20).mean()
    df["smc_atr_compression"] = atr / (atr_20 + 1e-9)

    return df


def _add_mtf_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    atr = df["atr"].astype(float)

    try:
        df_1h = df.resample("1H").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        if len(df_1h) > 50:
            ema_20 = df_1h["close"].ewm(span=20).mean()
            ema_50 = df_1h["close"].ewm(span=50).mean()

            h_1h = df_1h["high"]
            l_1h = df_1h["low"]
            c_1h = df_1h["close"]
            tr1_1h = h_1h - l_1h
            tr2_1h = (h_1h - c_1h.shift(1)).abs()
            tr3_1h = (l_1h - c_1h.shift(1)).abs()
            tr_1h = pd.concat([tr1_1h, tr2_1h, tr3_1h], axis=1).max(axis=1)
            atr_1h = tr_1h.rolling(window=14).mean()

            htf_trend_slope = (ema_20 - ema_50) / (atr_1h + 1e-9)

            htf_trend_slope_15m = htf_trend_slope.reindex(df.index, method="ffill")
            df["smc_htf_trend_slope"] = htf_trend_slope_15m.fillna(0.0)
        else:
            df["smc_htf_trend_slope"] = 0.0
    except Exception:
        df["smc_htf_trend_slope"] = 0.0

    day_index = df.index.normalize()
    daily_stats = df.groupby(day_index).agg(
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    )
    prev_day_high = daily_stats["high"].shift(1)
    prev_day_low = daily_stats["low"].shift(1)
    prev_day_close = daily_stats["close"].shift(1)
    prev_day_range = prev_day_high - prev_day_low

    daily_wick_rej = (prev_day_high - prev_day_close) / (prev_day_range + 1e-9)
    df["smc_daily_wick_rejection"] = (
        daily_wick_rej.reindex(day_index).fillna(0.0).to_numpy()
    )

    return df


def _add_volume_profile_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v = df["volume"].astype(float)
    atr = df["atr"].astype(float)

    lookback = int(config.get("smc_vp_lookback", 100))
    bins = int(config.get("smc_vp_bins", 50))

    hvn_gravity_list = []
    poc_dist_list = []
    is_in_value_area_list = []
    profile_skew_list = []

    for i in range(len(df)):
        if i < lookback:
            hvn_gravity_list.append(0.5)
            poc_dist_list.append(0.0)
            is_in_value_area_list.append(1)
            profile_skew_list.append(0.0)
            continue

        window_close = c.iloc[i - lookback : i]
        window_volume = v.iloc[i - lookback : i]
        current_price = c.iloc[i]

        try:
            hist, bin_edges = np.histogram(
                window_close,
                bins=bins,
                weights=window_volume,
            )

            bin_index = np.digitize(current_price, bin_edges) - 1
            bin_index = max(0, min(bins - 1, bin_index))

            vol_at_price = hist[bin_index]
            max_vol = np.max(hist)

            hvn_gravity = vol_at_price / (max_vol + 1e-9)
            hvn_gravity_list.append(float(hvn_gravity))

            poc_price = bin_edges[np.argmax(hist)]
            poc_dist = (current_price - poc_price) / (atr.iloc[i] + 1e-9)
            poc_dist_list.append(float(poc_dist))

            sorted_indices = np.argsort(hist)[::-1]
            cumsum = 0
            value_area_bins = []
            for idx in sorted_indices:
                cumsum += hist[idx]
                value_area_bins.append(idx)
                if cumsum >= 0.7 * hist.sum():
                    break

            is_in_va = 1 if bin_index in value_area_bins else 0
            is_in_value_area_list.append(is_in_va)

            volume_above = hist[bin_index:].sum()
            volume_below = hist[:bin_index].sum()
            skew = (volume_above - volume_below) / (hist.sum() + 1e-9)
            profile_skew_list.append(float(skew))

        except Exception:
            hvn_gravity_list.append(0.5)
            poc_dist_list.append(0.0)
            is_in_value_area_list.append(1)
            profile_skew_list.append(0.0)

    df["smc_hvn_gravity"] = hvn_gravity_list
    df["smc_poc_dist_atr"] = poc_dist_list
    df["smc_is_in_value_area"] = is_in_value_area_list
    df["smc_profile_skew"] = profile_skew_list

    return df


def _add_time_categories(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    hour = df.index.hour

    session_kz = pd.Series("Dead", index=df.index)
    session_kz[(hour >= 0) & (hour < 8)] = "Asia"
    session_kz[(hour >= 8) & (hour < 13)] = "London"
    session_kz[(hour >= 13) & (hour < 17)] = "NY_AM"
    session_kz[(hour >= 17) & (hour < 21)] = "NY_PM"

    for session in ["Asia", "London", "NY_AM", "NY_PM", "Dead"]:
        df[f"smc_session_{session}"] = (session_kz == session).astype(int)

    dow = df.index.dayofweek
    for day_num, day_name in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
        df[f"smc_dow_{day_name}"] = (dow == day_num).astype(int)

    h = df["high"].astype(float)
    l = df["low"].astype(float)

    hh = (h > h.shift(1)) & (h.shift(1) > h.shift(2))
    ll = (l < l.shift(1)) & (l.shift(1) < l.shift(2))
    lh = (h < h.shift(1)) & (h.shift(1) < h.shift(2))
    hl = (l > l.shift(1)) & (l.shift(1) > l.shift(2))

    market_structure = pd.Series("Range", index=df.index)
    market_structure[hh & hl] = "Uptrend"
    market_structure[lh & ll] = "Downtrend"

    df["smc_market_structure_Uptrend"] = (market_structure == "Uptrend").astype(int)
    df["smc_market_structure_Downtrend"] = (market_structure == "Downtrend").astype(int)
    df["smc_market_structure_Range"] = (market_structure == "Range").astype(int)

    df["smc_is_inside_fvg"] = (df["smc_current_fvg_size"] > 0).astype(int)

    c = df["close"].astype(float)
    pdh = df["smc_pdh"]
    pdl = df["smc_pdl"]

    sweep_high = (h > pdh) & (c < pdh)
    sweep_low = (l < pdl) & (c > pdl)
    df["smc_sweep_confirmed"] = (sweep_high | sweep_low).astype(int)

    return df


def _add_cross_timeframe_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        generator = CrossTimeframeFeatureGenerator()
    except Exception:
        return df

    base_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    if not base_cols:
        return df

    try:
        data = df[base_cols].copy()
        features = generator.generate_enhanced_cross_timeframe_features(data)
        if not features:
            return df

        for name, series in features.items():
            col_name = f"smc_ctf_{name}"
            ser = pd.Series(series)
            ser.index = data.index
            df[col_name] = ser.reindex(df.index).astype(float)
    except Exception:
        return df

    return df
