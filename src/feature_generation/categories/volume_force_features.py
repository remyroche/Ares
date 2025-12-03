"""Volume Force and Impulse feature generation.

This module defines features focusing on Volume Force/Impulse, derived from
order flow pressure, volume deltas, and shocks. These features act as
"Triggers" for immediate directional moves.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.stats import entropy

from src.features_common.transforms.scaling_normalization import (
    winsorized_zscore_normalize,
)
from src.utils.feature_common.volume_transforms import (
    log1p_zscore_normalize,
)


def generate_volume_force_features(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Generate normalized volume force/impulse features.

    Features include:
    - Volume Delta / Imbalance (proxy via candle structure)
    - Cumulative Volume Delta (CVD) Slope
    - Force Index (Volume * Price Change)
    - Volume Flow Indicator (VFI) logic
    - Kyle's Lambda (Market Impact)
    - Volume Shocks
    - Chaikin Money Flow (CMF)
    - Volume Price Trend (VPT)
    - Ease of Movement (EMV)
    - Volume Acceleration
    - Cross-Timeframe Volume Proxies (RVOL HTF, Trend Alignment, Breakout)
    - Advanced: Churn, Effort/Result, UV/DV Ratio, Volume RSI, OBV Divergence, Thrust, Elasticity
    - Time of day/week
    - Multi-timeframe acceleration and entropy

    Args:
        df: Input market data with `open`, `high`, `low`, `close`, `volume`.
        config: Configuration dictionary.

    Returns:
        DataFrame with normalized volume force features.
    """
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"Market data missing required columns: {required - set(df.columns)}")

    # Use float32 for memory efficiency
    # Ensure index is datetime for time features
    if not isinstance(df.index, pd.DatetimeIndex):
         # Attempt conversion if not already
         try:
             df_index = pd.to_datetime(df.index)
         except Exception:
             # Fallback if conversion fails, though it should be handled upstream
             df_index = df.index
    else:
        df_index = df.index

    open_p = df["open"].astype(np.float32)
    high = df["high"].astype(np.float32)
    low = df["low"].astype(np.float32)
    close = df["close"].astype(np.float32)
    volume = df["volume"].astype(np.float32)

    eps = 1e-9

    # 1. Volume Delta Proxy (Buyer vs Seller Intensity)
    # Estimate based on close position within range (Buying Pressure)
    candle_range = (high - low).replace(0, eps)
    close_pos = (close - low) / candle_range
    # Clip to [0, 1]
    close_pos = close_pos.clip(0.0, 1.0)

    # Buyer/Seller Volume Proxy
    buy_vol_proxy = volume * close_pos
    sell_vol_proxy = volume * (1.0 - close_pos)

    # Volume Delta: Net Buying Pressure
    # (Buy - Sell) / (Buy + Sell) -> ranges [-1, 1]
    volume_delta = (buy_vol_proxy - sell_vol_proxy) / (volume + eps)

    # 2. Cumulative Volume Delta (CVD) Slope
    # CVD is cumsum of delta. We want the slope/acceleration.
    # Short-term slope (velocity) and change in slope (acceleration)
    cvd = volume_delta.cumsum()

    # Slope over 3 and 6 bars
    cvd_slope_3 = cvd.diff(3)
    cvd_slope_6 = cvd.diff(6)

    # 3. Volume Imbalance (Signed Volume relative to baseline)
    # Signed by price direction
    price_change = close - close.shift(1)
    direction = np.sign(price_change)
    # If flat, use delta direction
    # Ensure consistent types for np.where
    direction_filled = np.where(direction == 0, np.sign(volume_delta), direction)

    signed_volume = volume * direction_filled

    # Normalize by recent volume std
    vol_mean_20 = volume.rolling(20).mean()
    vol_std_20 = volume.rolling(20).std().replace(0, eps)

    volume_imbalance = (volume - vol_mean_20) / vol_std_20 * direction_filled

    # 4. Force Index (Alexander Elder)
    # Volume * (Close - Prev Close)
    # We smooth it (e.g., 2-period and 13-period)
    raw_force = volume * price_change
    force_index_2 = raw_force.ewm(span=2, adjust=False).mean()
    force_index_13 = raw_force.ewm(span=13, adjust=False).mean()

    # Normalize Force Index by volume baseline to make it stationary-ish
    # Or just rely on winsorization later. Let's normalize by Avg Volume * Avg Price Change
    price_volatility = price_change.abs().rolling(20).mean().replace(0, eps)
    force_norm_factor = vol_mean_20 * price_volatility

    force_index_norm = force_index_13 / force_norm_factor

    # 5. Volume Flow Indicator (VFI) Proxy
    # Simplified: (Typical Price Change > 0 ? Vol : -Vol) smoothed
    typical_price = (high + low + close) / 3.0
    tp_change = typical_price.diff()

    # Cutoff for "noise" (optional, simplified here to just sign)
    vfi_raw = np.where(tp_change > 0, volume, -volume)
    vfi_raw = np.where(tp_change == 0, 0, vfi_raw)

    # VFI is typically a ratio of smoothed signed vol to smoothed total vol
    # Ensure index alignment
    vfi_num = pd.Series(vfi_raw, index=df.index).ewm(span=13, adjust=False).mean()
    vfi_den = volume.ewm(span=13, adjust=False).mean().replace(0, eps)
    vfi = vfi_num / vfi_den

    # 6. Kyle's Lambda Proxy (Market Impact)
    # |Price Change| / Volume
    # High value = Low Liquidity (Big impact per unit vol)
    # Low value = High Liquidity (Small impact per unit vol)
    kyles_lambda = price_change.abs() / (volume + eps)

    # 7. Volume Shock
    # Volume / MA Volume
    volume_shock = volume / vol_mean_20

    # 8. Chaikin Money Flow (CMF)
    # Money Flow Multiplier: ((Close - Low) - (High - Close)) / (High - Low)
    mf_multiplier = ((close - low) - (high - close)) / candle_range
    money_flow_vol = mf_multiplier * volume

    # Classic CMF is sum(MFV, 20) / sum(Vol, 20)
    cmf_window = 20
    cmf = (money_flow_vol.rolling(cmf_window).sum() /
           volume.rolling(cmf_window).sum().replace(0, eps))

    # 9. Volume Price Trend (VPT)
    # VPT = VPT_prev + Volume * (Close - Close_prev) / Close_prev
    # We want a stationary version: rolling slope of VPT or just (Volume * %Chg) smoothed
    # Let's use smoothed (Volume * %Chg) normalized by avg volume
    pct_change = price_change / close.shift(1).replace(0, eps)
    vpt_raw = volume * pct_change
    # Smooth over 13 periods
    vpt_smoothed = vpt_raw.ewm(span=13, adjust=False).mean()
    # Normalize by Average Volume to make it comparable
    vpt_norm = vpt_smoothed / vol_mean_20.replace(0, eps) * 100 # scale up

    # 10. Ease of Movement (EMV)
    # Distance moved / (Volume / Range)
    # Midpoint Move = (High + Low)/2 - (High_prev + Low_prev)/2
    midpoint = (high + low) / 2.0
    midpoint_move = midpoint - midpoint.shift(1)

    # EMV = Midpoint Move * (Range / Volume)
    # High EMV = Price rising easily on low volume
    # Low EMV = Price falling easily on low volume
    # Near 0 = Heavy volume required to move price
    emv_raw = midpoint_move * (candle_range / (volume + eps))
    # Smooth EMV
    emv = emv_raw.ewm(span=14, adjust=False).mean()

    # 11. Volume Acceleration
    # 2nd derivative of volume
    # diff(diff(volume))
    # Normalize by volume level
    vol_acc = volume.diff().diff()
    vol_acc_norm = vol_acc / vol_mean_20.replace(0, eps)

    # 12. Cross-Timeframe Volume Proxies
    # Assuming base dataframe is 15m.
    # 4h = 16 bars, Daily = 96 bars (approx for 24h market)

    # HTF Relative Volume
    vol_mean_4h = volume.rolling(16).mean()
    vol_mean_daily = volume.rolling(96).mean()

    rvol_htf_4h = volume / vol_mean_4h.replace(0, eps)
    rvol_htf_daily = volume / vol_mean_daily.replace(0, eps)

    # Volume Trend Alignment
    # Slope of volume over 15m (3 bars), 1h (4 bars), 4h (16 bars)
    # Simple linear regression slope is expensive windowed.
    # Proxy: (Vol - Vol_lag) / Vol_lag (ROC) smoothed

    vol_roc_15m = volume.pct_change(3) # 45m trend
    vol_roc_1h = volume.pct_change(4) # 1h trend
    vol_roc_4h = volume.pct_change(16) # 4h trend

    # Alignment: 1 if all same sign, -1 if opposed, 0 mixed
    # We want "Strength of Alignment". Sum of signs?
    # Let's do a weighted sum or simply mean sign
    vol_trend_alignment = (np.sign(vol_roc_15m) + np.sign(vol_roc_1h) + np.sign(vol_roc_4h)) / 3.0

    # 13. HTF Volume Breakout (Volume / Rolling Max)
    # How close is current volume to the max volume of the last 4h/Daily?
    # Ratio > 1 means breakout.
    vol_max_4h = volume.rolling(16).max()
    vol_htf_breakout_4h = volume / vol_max_4h.replace(0, eps)

    vol_max_daily = volume.rolling(96).max()
    vol_htf_breakout_daily = volume / vol_max_daily.replace(0, eps)

    # ==============================================================================
    # NEW ADVANCED FEATURES (Requested)
    # ==============================================================================

    # 14. Churn Index & Absorption (High Volume, Low Progress)
    # Churn = Volume / EffectivePriceMovement. Using Range for "Movement capability".
    # Absorption = High Volume / Low Range
    churn_index = volume / candle_range
    # Normalize churn by its moving average to detect anomalies
    churn_index_norm = churn_index / churn_index.rolling(20).mean().replace(0, eps)

    # 15. Effort vs Result Ratio (Wyckoff)
    # Effort = Volume, Result = abs(Close - Close_prev)
    # Ratio = Effort / Result. High ratio = Inefficiency/Absorption.
    effort_result_ratio = volume / price_change.abs().replace(0, eps)
    # Log transform due to heavy tails when price change is near zero
    effort_result_log = np.log1p(effort_result_ratio)

    # 16. UV/DV Ratio (Up Volume / Down Volume)
    # Defined over a rolling window (e.g., 14 bars) to be stable
    up_vol = np.where(price_change > 0, volume, 0)
    down_vol = np.where(price_change < 0, volume, 0)

    up_vol_sum = pd.Series(up_vol, index=df.index).rolling(14).sum()
    down_vol_sum = pd.Series(down_vol, index=df.index).rolling(14).sum()

    uv_dv_ratio = up_vol_sum / down_vol_sum.replace(0, eps)
    # Normalize to -1 to 1 range: (UV - DV) / (UV + DV) = Money Flow Ratio style
    uv_dv_oscillator = (up_vol_sum - down_vol_sum) / (up_vol_sum + down_vol_sum + eps)

    # 17. Volume RSI
    # RSI formula applied to Volume changes
    vol_diff = volume.diff()
    gain = vol_diff.where(vol_diff > 0, 0)
    loss = -vol_diff.where(vol_diff < 0, 0)

    avg_gain = gain.ewm(com=13, min_periods=13).mean()
    avg_loss = loss.ewm(com=13, min_periods=13).mean()
    rs = avg_gain / avg_loss.replace(0, eps)
    volume_rsi = 100 - (100 / (1 + rs))

    # 18. On-Balance Volume (OBV) Features
    # OBV Trend and Divergence
    obv = (np.sign(price_change) * volume).cumsum()

    # OBV Slope (Trend Confirmation)
    obv_slope = obv.diff(5)
    # Normalize slope by recent volume sum to make scale-invariant
    obv_slope_norm = obv_slope / volume.rolling(5).sum().replace(0, eps)

    # OBV-Price Divergence
    # Compare z-score of OBV vs z-score of Price over a window
    # If Price Z is High and OBV Z is Low -> Bearish Divergence (Weak support)
    # If Price Z is Low and OBV Z is High -> Bullish Divergence (Accumulation)
    div_window = 20
    price_z = (close - close.rolling(div_window).mean()) / close.rolling(div_window).std().replace(0, eps)
    obv_z = (obv - obv.rolling(div_window).mean()) / obv.rolling(div_window).std().replace(0, eps)
    obv_price_divergence = obv_z - price_z

    # 19. Volume Thrust Index
    # Volume * Price Change, smoothed. (Force Index logic, but focused on surge)
    # Thrust = (Vol * PctChange) EWM / Vol EWM
    # Often calculated as: (V * C - V_prev * C_prev) ... let's use Force Index norm as proxy or VPT.
    # Let's create a specific "Thrust" oscillator:
    # (ShortMA(Vol*CloseChg) - LongMA(Vol*CloseChg))
    force = volume * price_change
    thrust = force.ewm(span=5).mean() - force.ewm(span=20).mean()
    # Normalize
    thrust_norm = thrust / (volume * close * 0.01).rolling(20).mean().replace(0, eps)

    # 20. Volume/Volatility Elasticity
    # % Change in Volume / % Change in Price (Volatility)
    # Sensitivity of participation to price moves.
    vol_pct = volume.pct_change()
    price_pct = price_change.abs() / close.shift(1).replace(0, eps)
    vol_elasticity = vol_pct / price_pct.replace(0, eps)
    # Clip extreme values
    vol_elasticity = vol_elasticity.clip(-10, 10)

    # 21. VWAP Deviation Bands
    # Rolling VWAP (24h approx = 96 bars for 15m)
    vwap_window = 96
    cum_vol = volume.rolling(vwap_window).sum()
    cum_vol_price = (volume * typical_price).rolling(vwap_window).sum()
    rolling_vwap = cum_vol_price / cum_vol.replace(0, eps)

    # Standard deviation of price relative to VWAP? Or just ATR bands?
    # Usually bands are VWAP +/- k * StdDev(Price)
    # Deviation = (Close - VWAP) / StdDev(Price, 96)
    price_std_long = close.rolling(vwap_window).std().replace(0, eps)
    vwap_deviation = (close - rolling_vwap) / price_std_long

    # 22. Climax / Dry-up Flags (Continuous features for ML)
    # Climax: Volume Z-Score
    vol_zscore = (volume - vol_mean_20) / vol_std_20

    # 23. Aggressive vs Passive (Close location vs Volume)
    # If Close is near High on High Volume -> Aggressive Buying
    # If Close is near Low on High Volume -> Aggressive Selling
    # Passive = High Volume but Close in middle (Doji) -> Churn/Absorption
    # Feature: Volume * (2*ClosePos - 1).
    # Range [-Vol, +Vol]. High + = Aggr Buy. High - = Aggr Sell. Near 0 = Passive/Indecision.
    aggressive_flow = volume * (2 * close_pos - 1)
    # Normalize
    aggressive_flow_norm = aggressive_flow / vol_mean_20.replace(0, eps)

    # ==============================================================================
    # 24. NEW MULTI-TASK & TIME FEATURES (Specific Request)
    # ==============================================================================

    # Assemble DataFrame
    features = pd.DataFrame(index=df.index)

    # Add Core Features
    features["volume_delta"] = volume_delta
    features["cvd_slope_3"] = cvd_slope_3
    features["cvd_slope_6"] = cvd_slope_6
    features["volume_imbalance"] = volume_imbalance
    features["force_index_norm"] = force_index_norm
    features["vfi"] = vfi
    features["kyles_lambda"] = kyles_lambda
    features["volume_shock"] = volume_shock
    features["cmf"] = cmf
    features["vpt_norm"] = vpt_norm
    features["emv"] = emv
    features["volume_acceleration"] = vol_acc_norm
    features["rvol_htf_4h"] = rvol_htf_4h
    features["rvol_htf_daily"] = rvol_htf_daily
    features["volume_trend_alignment"] = vol_trend_alignment
    features["vol_htf_breakout_4h"] = vol_htf_breakout_4h
    features["vol_htf_breakout_daily"] = vol_htf_breakout_daily

    # Add Advanced Features
    features["churn_index_norm"] = churn_index_norm
    features["effort_result_log"] = effort_result_log
    features["uv_dv_ratio"] = uv_dv_ratio
    features["uv_dv_oscillator"] = uv_dv_oscillator
    features["volume_rsi"] = volume_rsi
    features["obv_slope_norm"] = obv_slope_norm
    features["obv_price_divergence"] = obv_price_divergence
    features["volume_thrust"] = thrust_norm
    features["vol_elasticity"] = vol_elasticity
    features["vwap_deviation"] = vwap_deviation
    features["vol_zscore"] = vol_zscore
    features["aggressive_flow_norm"] = aggressive_flow_norm

    # Time Features
    if isinstance(df_index, pd.DatetimeIndex):
        day_of_week = df_index.dayofweek
        hour = df_index.hour
        minute = df_index.minute

        # Day of week (0-6)
        features["day_of_week"] = day_of_week

        # Time of day encoded cyclically
        minutes_in_day = hour * 60 + minute
        features["time_day_sin"] = np.sin(2 * np.pi * minutes_in_day / 1440)
        features["time_day_cos"] = np.cos(2 * np.pi * minutes_in_day / 1440)

        # Minutes since last funding (every 8h: 00:00, 08:00, 16:00)
        # 8h = 480 minutes
        features["minutes_since_funding"] = minutes_in_day % 480
    else:
        # Fallback if no datetime index
        features["day_of_week"] = 0
        features["time_day_sin"] = 0
        features["time_day_cos"] = 0
        features["minutes_since_funding"] = 0

    # Ensure day_of_week is numeric
    features["day_of_week"] = pd.to_numeric(features["day_of_week"], errors='coerce').fillna(0)

    # Volume Z-Score over 1d (96 bars)
    vol_mean_96 = volume.rolling(96).mean()
    vol_std_96 = volume.rolling(96).std().replace(0, eps)
    features["vol_15m_zscore_over_1d"] = (volume - vol_mean_96) / vol_std_96

    # Rolling Rank of volume (last 96 bars)
    features["rolling_rank_vol_96"] = volume.rolling(96).rank(pct=True)

    # Delta Volume 15m (Ratio to previous)
    features["delta_vol_15m"] = volume / volume.shift(1).replace(0, eps)

    # 1h Volume Acceleration
    # vol_1h = rolling(4).sum()
    vol_1h_sum = volume.rolling(4).sum()
    features["accel_1h"] = vol_1h_sum / vol_1h_sum.shift(4).replace(0, eps)

    # Multi-TF Acceleration
    # (vol_15m / vol_1h_ma) * (vol_1h / vol_4h_ma)
    # vol_1h_ma defined above as vol_mean_4 (rolling 4 mean) = vol_1h_sum / 4?
    # Usually "MA" means mean volume over that window.
    # vol_1h_ma = rolling(4).mean()
    # vol_4h_ma = rolling(16).mean()
    # vol_1h (instantaneous) -> we can use rolling(4).sum() or rolling(4).mean() * 4
    # Let's interpret strictly:
    # Ratio 1: vol / vol_mean_4h (used rvol_htf_4h above, but request says 1h ma)
    vol_mean_1h = volume.rolling(4).mean()
    ratio_15m_1h = volume / vol_mean_1h.replace(0, eps)
    features["vol_15m_div_vol_1h_ma"] = ratio_15m_1h

    ratio_15m_4h = volume / vol_mean_4h.replace(0, eps)
    features["vol_15m_div_vol_4h_ma"] = ratio_15m_4h

    # Ratio 2: vol_1h / vol_4h_ma
    # vol_1h here likely means "volume over last 1h" = rolling(4).sum() or mean
    # Let's use mean to keep units consistent
    ratio_1h_4h = vol_mean_1h / vol_mean_4h.replace(0, eps)

    features["multi_tf_accel"] = ratio_15m_1h * ratio_1h_4h

    # CVD Slope Ratios
    # cvd_slope_1h (4 bars diff) vs cvd_slope_15m (1 bar diff)
    # Request: cvd_15m_slope / cvd_1h_slope
    # cvd is accumulated delta.
    cvd_slope_15m = cvd.diff(1)
    cvd_slope_1h = cvd.diff(4)
    features["cvd_slope_ratio"] = cvd_slope_15m / cvd_slope_1h.replace(0, eps)

    # CVD 1h / CVD 4h (Absolute levels or slopes? "cvd_1h / cvd_4h")
    # Likely means slope or accumulated change over that window
    cvd_slope_4h = cvd.diff(16)
    features["cvd_1h_div_cvd_4h"] = cvd_slope_1h / cvd_slope_4h.replace(0, eps)

    # 15m CVD Z-Score over 1d
    cvd_change = cvd.diff(1)
    cvd_change_mean_96 = cvd_change.rolling(96).mean()
    cvd_change_std_96 = cvd_change.rolling(96).std().replace(0, eps)
    features["cvd_15m_zscore_over_1d"] = (cvd_change - cvd_change_mean_96) / cvd_change_std_96

    # Entropy of Volume across TF
    # H([vol_15m, vol_1h, vol_4h])
    # Convert to equivalent units (e.g. per-minute volume or just total)
    # If we use sums: 15m sum, 1h sum, 4h sum.
    # 1h sum includes 15m sum. 4h sum includes 1h sum.
    # The request likely implies entropy of the *distribution* of volume at these scales
    # or entropy of the vector [v15, v1h, v4h] normalized.
    v15 = volume
    v1h = vol_1h_sum
    v4h = volume.rolling(16).sum()

    # Normalize to probability distribution for entropy calculation
    # p_i = v_i / sum(v)
    sum_v = v15 + v1h + v4h + eps
    p15 = v15 / sum_v
    p1h = v1h / sum_v
    p4h = v4h / sum_v

    # Calculate entropy row-wise
    # Entropy = -sum(p * log(p))
    # Note: using numpy directly for speed
    def row_entropy(row):
         p = np.array([row[0], row[1], row[2]])
         return entropy(p)

    # Vectorized approximate entropy
    # - (p15*log(p15) + p1h*log(p1h) + p4h*log(p4h))
    # Clip p to avoid log(0)
    p15 = p15.clip(1e-9, 1.0)
    p1h = p1h.clip(1e-9, 1.0)
    p4h = p4h.clip(1e-9, 1.0)

    features["entropy_volume_tf"] = -(p15 * np.log(p15) + p1h * np.log(p1h) + p4h * np.log(p4h))

    # Volume Fragmentation
    # vol_1h / (vol_4h + vol_1d)
    vol_1d_sum = volume.rolling(96).sum()
    features["vol_fragmentation"] = v1h / (v4h + vol_1d_sum + eps)

    # Assemble final set
    # Horizon-aligned aggregations using volume_force_lookahead as bar horizon
    horizon = int(config.get("volume_force_lookahead", 12))
    horizon = max(1, horizon)
    features["volume_imbalance_roll_h"] = volume_imbalance.rolling(horizon, min_periods=1).sum()
    features["volume_shock_roll_h"] = volume_shock.rolling(horizon, min_periods=1).sum()
    features["aggressive_flow_roll_h"] = aggressive_flow_norm.rolling(horizon, min_periods=1).sum()

    # Normalization
    # Group 1: Oscillators (Center around 0 or 50)
    oscillators = [
        "volume_delta", "cvd_slope_3", "cvd_slope_6", "volume_imbalance",
        "force_index_norm", "vfi", "cmf", "vpt_norm", "emv", "volume_acceleration",
        "volume_trend_alignment", "uv_dv_oscillator", "volume_rsi", "obv_slope_norm",
        "obv_price_divergence", "volume_thrust", "vol_elasticity", "vwap_deviation",
        "vol_zscore", "aggressive_flow_norm",
        "vol_15m_zscore_over_1d", "delta_vol_15m", "cvd_slope_ratio", "cvd_1h_div_cvd_4h",
        "cvd_15m_zscore_over_1d", "time_day_sin", "time_day_cos"
    ]

    # Group 2: Magnitude/Ratios (Positive, heavy tail)
    magnitudes = [
        "kyles_lambda", "volume_shock",
        "rvol_htf_4h", "rvol_htf_daily",
        "vol_htf_breakout_4h", "vol_htf_breakout_daily",
        "churn_index_norm", "effort_result_log", "uv_dv_ratio",
        "volume_imbalance_roll_h", "volume_shock_roll_h", "aggressive_flow_roll_h",
        "rolling_rank_vol_96", "accel_1h", "multi_tf_accel",
        "vol_15m_div_vol_1h_ma", "vol_15m_div_vol_4h_ma",
        "entropy_volume_tf", "vol_fragmentation", "minutes_since_funding"
    ]

    window_size = int(config.get("volume_force_normalization_window", 100))

    # Apply winsorized z-score to oscillators
    features[oscillators] = winsorized_zscore_normalize(
        features[oscillators],
        window=window_size
    )

    # Apply log1p + z-score to magnitudes
    for col in magnitudes:
        features[col] = log1p_zscore_normalize(features[col], window=window_size)

    # Day of week is categorical/cyclical, leaving as is or normalizing?
    # Usually treated as categorical, but for simple XGB integration, 0-6 is fine.
    # No normalization applied to day_of_week.

    return features
