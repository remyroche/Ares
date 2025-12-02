"""Volume Force and Impulse feature generation.

This module defines features focusing on Volume Force/Impulse, derived from
order flow pressure, volume deltas, and shocks. These features act as
"Triggers" for immediate directional moves.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

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
    - Cross-Timeframe Volume Proxies (RVOL HTF, Trend Alignment)

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

    # Assemble DataFrame
    features = pd.DataFrame(index=df.index)
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

    # Normalization
    # Group 1: Oscillators (Center around 0)
    oscillators = [
        "volume_delta", "cvd_slope_3", "cvd_slope_6", "volume_imbalance",
        "force_index_norm", "vfi", "cmf", "vpt_norm", "emv", "volume_acceleration",
        "volume_trend_alignment"
    ]

    # Group 2: Magnitude/Ratios (Positive, heavy tail)
    magnitudes = ["kyles_lambda", "volume_shock", "rvol_htf_4h", "rvol_htf_daily"]

    window_size = int(config.get("volume_force_normalization_window", 500))

    # Apply winsorized z-score to oscillators
    features[oscillators] = winsorized_zscore_normalize(
        features[oscillators],
        window=window_size
    )

    # Apply log1p + z-score to magnitudes
    for col in magnitudes:
        features[col] = log1p_zscore_normalize(features[col], window=window_size)

    return features
