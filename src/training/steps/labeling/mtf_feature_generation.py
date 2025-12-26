"""
Multi-Timeframe Feature Generation Module.
Extracted and enhanced from feature_generation_meta_labeling_step.py.
"""
from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import logging
try:
    from scipy.signal import hilbert
except ImportError:
    hilbert = None

from src.features_common.transforms.scaling_normalization import winsorized_zscore_normalize
from src.utils.feature_common.volume_transforms import log1p_zscore_normalize
from src.utils.feature_common.atr_normalization import atr_normalize, should_use_atr_normalization, calculate_atr

logger = logging.getLogger(__name__)

class KalmanFilter1D:
    """
    Simple 1D Kalman filter for signal smoothing.

    State-space model:
        x_t = x_{t-1} + mu + w_t    (state evolution)
        y_t = x_t + v_t              (observation)

    Where w_t ~ N(0, Q) and v_t ~ N(0, R)
    """

    def __init__(self, Q: float = 1e-5, R: float = 0.01, initial_value: float = 0.0):
        """
        Args:
            Q: Process variance (smaller = smoother evolution)
            R: Observation variance (larger = more smoothing)
            initial_value: Initial state estimate
        """
        self.Q = Q  # Process variance
        self.R = R  # Observation variance
        self.x = initial_value  # State estimate
        self.P = 1.0  # State variance

    def update(self, measurement: float) -> Tuple[float, float]:
        """
        Update filter with new measurement.

        Returns:
            Tuple of (filtered_value, state_variance)
        """
        # Predict
        x_prior = self.x
        P_prior = self.P + self.Q

        # Update
        K = P_prior / (P_prior + self.R)  # Kalman gain
        self.x = x_prior + K * (measurement - x_prior)
        self.P = (1 - K) * P_prior

        return self.x, self.P

    def filter_series(self, series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Filter entire time series.

        Returns:
            Tuple of (filtered_series, variance_series)
        """
        filtered = []
        variances = []

        for val in series:
            if pd.isna(val):
                filtered.append(np.nan)
                variances.append(np.nan)
            else:
                x_filt, P_filt = self.update(val)
                filtered.append(x_filt)
                variances.append(P_filt)

        return (
            pd.Series(filtered, index=series.index),
            pd.Series(variances, index=series.index)
        )

def kalman_smooth_trend(prices: pd.Series, Q: float = 1e-5, R: float = 0.01) -> Tuple[pd.Series, pd.Series]:
    """
    Extract smoothed trend from prices using Kalman filter.

    Args:
        prices: Price series
        Q: Process variance (smaller = smoother)
        R: Observation variance (larger = more smoothing)

    Returns:
        Tuple of (trend, uncertainty)
    """
    kf = KalmanFilter1D(Q=Q, R=R, initial_value=prices.iloc[0] if len(prices) > 0 else 0.0)
    return kf.filter_series(prices)

def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI (Relative Strength Index)."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute MACD (Moving Average Convergence Divergence).

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def get_efficiency_ratio(close: pd.Series, window: int = 14) -> pd.Series:
    """Compute Kaufman Efficiency Ratio."""
    change = close.diff(window).abs()
    volatility = close.diff().abs().rolling(window).sum()
    return change / (volatility + 1e-9)

def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Compute Stochastic Oscillator (%K and %D)."""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-9))
    d_percent = k_percent.rolling(window=d_period).mean()
    return k_percent, d_percent

def compute_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Compute Commodity Channel Index (CCI)."""
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=period).mean()
    # Mean deviation from the moving average
    mean_dev = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma_tp) / (0.015 * mean_dev + 1e-9)
    return cci

def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute ADX, Plus DI, Minus DI."""
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm = plus_dm.where(plus_dm > 0, 0.0)
    minus_dm = minus_dm.where(minus_dm > 0, 0.0)

    mask_plus = (plus_dm > minus_dm) & (plus_dm > 0)
    mask_minus = (minus_dm > plus_dm) & (minus_dm > 0)

    plus_dm_final = pd.Series(0.0, index=close.index)
    minus_dm_final = pd.Series(0.0, index=close.index)

    plus_dm_final[mask_plus] = plus_dm[mask_plus]
    minus_dm_final[mask_minus] = minus_dm[mask_minus]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()

    plus_di = 100 * (plus_dm_final.rolling(period).mean() / (atr + 1e-9))
    minus_di = 100 * (minus_dm_final.rolling(period).mean() / (atr + 1e-9))

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    adx = dx.rolling(period).mean()
    return adx, plus_di, minus_di

def compute_bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands (Upper, Middle, Lower, Width)."""
    middle = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    width = (upper - lower) / (middle + 1e-9)
    return upper, middle, lower, width

def compute_choppiness_index(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Choppiness Index."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_sum = tr.rolling(period).sum()
    high_max = high.rolling(period).max()
    low_min = low.rolling(period).min()

    ci = 100 * np.log10(atr_sum / (high_max - low_min + 1e-9)) / np.log10(period)
    return ci

def compute_cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """Compute Chaikin Money Flow."""
    mf_multiplier = ((close - low) - (high - close)) / (high - low + 1e-9)
    mf_volume = mf_multiplier * volume
    cmf = mf_volume.rolling(period).sum() / (volume.rolling(period).sum() + 1e-9)
    return cmf

def compute_force_index(close: pd.Series, volume: pd.Series, period: int = 13) -> pd.Series:
    """Compute Force Index."""
    fi = close.diff(1) * volume
    return fi.ewm(span=period).mean()

def compute_hurst_proxy(close: pd.Series, window: int = 100) -> pd.Series:
    """Vectorized Hurst Exponent proxy using Rolling R/S analysis."""
    roll = close.rolling(window)
    r = roll.max() - roll.min()
    s = roll.std()
    rs = r / (s + 1e-9)
    hurst = np.log(rs + 1e-9) / np.log(window)
    return hurst

def compute_parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """Compute Parkinson Volatility."""
    log_hl = np.log(high / (low + 1e-9)) ** 2
    return np.sqrt((1.0 / (4.0 * np.log(2.0))) * log_hl.rolling(window).mean())

def compute_ema_slope(series: pd.Series, window: int = 20) -> pd.Series:
    """Compute EMA slope (normalized)."""
    ema = series.ewm(span=window, adjust=False).mean()
    # Slope as pct change of EMA
    slope = ema.pct_change()
    return slope

def compute_donchian_channel(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Donchian Channel (Upper, Lower, Position)."""
    upper = high.rolling(window).max()
    lower = low.rolling(window).min()
    width = upper - lower
    # Position: 0 to 1
    position = (close - lower) / (width + 1e-9)
    return upper, lower, position

def compute_garman_klass_volatility(open_p: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """Compute Garman-Klass Volatility."""
    log_hl = np.log(high / (low + 1e-9)) ** 2
    log_co = np.log(close / (open_p + 1e-9)) ** 2
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return np.sqrt(gk.rolling(window).mean())

def compute_volume_delta(close: pd.Series, open_p: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute Volume Delta Proxy."""
    # 1 if Close > Open (Buy), -1 if Close < Open (Sell)
    direction = np.sign(close - open_p)
    # If flat, assume 0 or maintain previous? 0 is safer for proxy.
    return direction * volume

def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute On-Balance Volume (OBV)."""
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    return (direction * volume).cumsum()

def compute_rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """Compute Rolling Z-Score."""
    roll = series.rolling(window)
    return (series - roll.mean()) / (roll.std() + 1e-9)

def compute_rolling_percentile(series: pd.Series, window: int = 20) -> pd.Series:
    """Compute Rolling Percentile (Rank)."""
    return series.rolling(window).rank(pct=True)

def compute_bars_since(condition: pd.Series) -> pd.Series:
    """Count bars since condition was True."""
    idx = pd.Series(np.arange(len(condition)), index=condition.index)
    last_occurrence = idx.where(condition).ffill()
    return (idx - last_occurrence).fillna(len(condition))

def compute_rogers_satchell_volatility(open_p: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """Compute Rogers-Satchell Volatility."""
    h_c = np.log(high / (close + 1e-9))
    h_o = np.log(high / (open_p + 1e-9))
    l_c = np.log(low / (close + 1e-9))
    l_o = np.log(low / (open_p + 1e-9))
    rs_var = (h_c * h_o) + (l_c * l_o)
    return np.sqrt(rs_var.rolling(window).mean())

def compute_yang_zhang_volatility(open_p: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """Compute Yang-Zhang Volatility."""
    # Log variances
    log_oc_prev = np.log(open_p / (close.shift(1) + 1e-9))
    log_co = np.log(close / (open_p + 1e-9))

    var_o = log_oc_prev.rolling(window).var()
    var_c = log_co.rolling(window).var()

    # Rogers-Satchell part
    # Re-using internal RS calc for efficiency or call function?
    # Calling function does rolling mean, we need that.
    h_c = np.log(high / (close + 1e-9))
    h_o = np.log(high / (open_p + 1e-9))
    l_c = np.log(low / (close + 1e-9))
    l_o = np.log(low / (open_p + 1e-9))
    rs_var = (h_c * h_o) + (l_c * l_o)
    rs_var_mean = rs_var.rolling(window).mean()

    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    yz_var = var_o + k * var_c + (1 - k) * rs_var_mean
    return np.sqrt(yz_var)

def compute_aroon(high: pd.Series, low: pd.Series, window: int = 25) -> pd.Series:
    """Compute Aroon Oscillator."""
    # How many bars since high/low
    high_idx = high.rolling(window).apply(np.argmax, raw=True)
    low_idx = low.rolling(window).apply(np.argmin, raw=True)

    aroon_up = ((window - (window - 1 - high_idx)) / window) * 100
    aroon_down = ((window - (window - 1 - low_idx)) / window) * 100
    return aroon_up - aroon_down

def compute_ease_of_movement(high: pd.Series, low: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
    """Compute Ease of Movement (EOM)."""
    dm = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
    box_ratio = (volume + 1e-9) / ((high - low) + 1e-9)
    eom = dm / box_ratio
    return eom.rolling(window).mean()

def compute_mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
    """Compute Money Flow Index (Volume-Weighted RSI)."""
    tp = (high + low + close) / 3
    rmf = tp * volume

    diff = tp.diff()
    pos_flow = rmf.where(diff > 0, 0)
    neg_flow = rmf.where(diff < 0, 0)

    pos_mean = pos_flow.rolling(window).mean()
    neg_mean = neg_flow.rolling(window).mean()

    mfi = 100 - (100 / (1 + (pos_mean / (neg_mean + 1e-9))))
    return mfi

def compute_fisher_transform(high: pd.Series, low: pd.Series, window: int = 10) -> pd.Series:
    """Compute Ehlers Fisher Transform."""
    mid = (high + low) / 2
    # Normalize to -1..1 over window
    rolling_min = low.rolling(window).min()
    rolling_max = high.rolling(window).max()

    val = 2 * ((mid - rolling_min) / (rolling_max - rolling_min + 1e-9) - 0.5)
    # Smooth with EMA
    val = val.ewm(alpha=0.33).mean()
    # Clamp
    val = val.clip(-0.999, 0.999)

    fisher = 0.5 * np.log((1 + val) / (1 - val))
    return fisher.ewm(alpha=0.5).mean()

def compute_hilbert_phase(series: pd.Series) -> pd.Series:
    """
    Compute Causal Hilbert Transform Phase Proxy.
    Uses Ehlers' Two-Bar Hilbert Transform logic to avoid future leakage.
    (Simple implementation: Quadrature = Price[t] - Price[t-2], InPhase = Price[t-1] - Price[t-3])
    """
    # Detrend
    centered = series - series.rolling(30).mean()

    # Ehlers Simple Hilbert Transform (Causal)
    # Q[t] = trend[t] - trend[t-2]?
    # Using simple difference proxy for quadrature
    # Q = centered.diff(2) * -1 (90 deg lag)?
    # Let's use standard approximation:
    # Q = I shifted by 90 deg.
    # Q[t] = 0.0962*P[t] + 0.5769*P[t-2] - 0.5769*P[t-4] - 0.0962*P[t-6] (Ehlers Coefficients)
    # But simple diff is often used as rough proxy.

    # Using Quadrature = series.diff(3) and InPhase = series.shift(1).diff(2)
    # This is a crude proxy.

    # Better: Use discrete Hilbert filter kernel (Type III or IV) but that's complex.
    # We will use the simplest "Ehlers Loop" inputs:
    # Q1 = (Price - Price[2])
    # I1 = (Price[1] - Price[3])
    # Phase = arctan(Q1 / I1)

    q1 = centered.diff(2)
    i1 = centered.shift(1).diff(2)

    phase = np.arctan2(q1, i1 + 1e-9)
    return phase.fillna(0)

def _align_to_features(arr: Any, n: int) -> np.ndarray:
    """Helper to align 1D array to feature index length."""
    values = np.asarray(arr)
    if len(values) == n:
        return values
    if len(values) > n:
        return values[:n]
    padded = np.full(n, np.nan, dtype=float)
    padded[: len(values)] = values
    return padded

def create_meta_features(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    volume_available: bool = True,
    include_raw_signals: bool = False,
    use_kalman: bool = True,
    windows: List[int] = [5, 10, 20, 50, 100, 150],
) -> pd.DataFrame:
    """
    Create features for the meta-model with Multi-Timeframe support.
    """
    # Hard-align df and signals to a shared tail window
    len_df = len(df)
    len_sig = len(signals)

    if len_df != len_sig:
        target_len = min(len_df, len_sig)
        if len_df > target_len: df = df.iloc[-target_len:, :]
        if len_sig > target_len: signals = signals.iloc[-target_len:, :]

    # Reset index to avoid duplicate index issues
    if (not df.index.equals(signals.index)) or df.index.has_duplicates or signals.index.has_duplicates:
        df = df.reset_index(drop=True)
        signals = signals.reset_index(drop=True)

    features = pd.DataFrame(index=df.index)
    n_features = len(features)

    def _norm(data: Union[pd.Series, np.ndarray], name: str) -> np.ndarray:
        """
        Normalize feature data based on feature name and type.
        """
        if not isinstance(data, pd.Series):
             if len(data) == len(df):
                 series = pd.Series(data, index=df.index)
             else:
                 series = pd.Series(data)
        else:
            series = data

        name_lower = name.lower()
        exclude_keywords = [
            'ratio', 'trend', 'corr', 'rel', 'per', 'norm',
            'imbalance', 'signed', 'pressure', 'delta', 'cvd', 'spike',
            'asymmetry', 'location', 'oscillator', 'phase', 'flag', 'count'
        ]

        if 'volume' in name_lower and not any(x in name_lower for x in exclude_keywords):
             return log1p_zscore_normalize(series, window=600).fillna(0).to_numpy()

        if should_use_atr_normalization(name):
             if any(x in name_lower for x in ['position', 'ratio', 'score', 'percent', 'pct', 'oscillator', 'index', 'flag', 'count']):
                 pass
             else:
                 return atr_normalize(series, df['high'], df['low'], df['close'], window=14).fillna(0).to_numpy()

        return winsorized_zscore_normalize(series, window=600).fillna(0).to_numpy()

    # Pre-calc common series
    returns = df['close'].pct_change()
    log_ret = np.log(df['close']).diff()
    features['log_ret'] = log_ret

    high = df['high']
    low = df['low']
    close = df['close']
    open_p = df.get('open', close) # Fallback if open missing
    volume = df['volume'] if volume_available and 'volume' in df.columns else pd.Series(1, index=df.index)

    # ===== 1. CANDLE GEOMETRY & MICRO-SENTIMENT =====
    candle_range = high - low
    upper_shadow = high - pd.concat([open_p, close], axis=1).max(axis=1)
    lower_shadow = pd.concat([open_p, close], axis=1).min(axis=1) - low
    real_body = (close - open_p).abs()

    # Body to Range
    features['body_to_range'] = _align_to_features(_norm(real_body / (candle_range + 1e-9), 'body_to_range'), n_features)

    # Shadow Asymmetry: (Upper - Lower) / Range. Positive = Bearish rejection, Negative = Bullish rejection
    features['shadow_asymmetry'] = _align_to_features(_norm((upper_shadow - lower_shadow) / (candle_range + 1e-9), 'shadow_asymmetry'), n_features)

    # Close Location Value (CLV): ((C - L) - (H - C)) / (H - L) -> range [-1, 1]
    clv = ((close - low) - (high - close)) / (candle_range + 1e-9)
    features['close_location_value'] = _align_to_features(_norm(clv, 'close_location_value'), n_features)

    # ===== 2. VOLATILITY REGIME (GLOBAL) =====
    # Global/Base features
    vol_short_20 = log_ret.rolling(window=20).std()
    vol_long_mean = vol_short_20.rolling(window=200, min_periods=50).mean()
    vol_long_std = vol_short_20.rolling(window=200, min_periods=50).std()
    rv_z_short = (vol_short_20 - vol_long_mean) / (vol_long_std + 1e-8)
    features['rv_z_short'] = _norm(rv_z_short.fillna(0.0), 'rv_z_short')

    # Volatility Trend Slope
    features['volatility_trend_slope'] = _align_to_features(_norm(vol_short_20.diff(5), 'volatility_trend_slope'), n_features)

    # ===== SPECIALIST SCALAR FEATURES =====
    specialist_cols = [
        "risk_score", "path_risk_score", "macro_trend_score_continuous",
        "mr_probability_dense", "mr_probability", "mr_raw_score",
        "mr_trend_state", "mr_trend_is_mr", "sr_labeling_xgb_prob",
        "vol_force_scalar", "smc_predicted"
    ]
    for col in df.columns:
        if col in specialist_cols or col.startswith("mr_") or col.startswith("smc_"):
            if col not in features.columns:
                features[col] = _norm(df[col], col)

    # ===== KALMAN & BASE INDICATORS =====
    df_local = df.copy()
    df_local['rsi'] = compute_rsi(close, period=14)
    df_local['momentum_30'] = close.pct_change(30)
    features['momentum_30'] = _align_to_features(_norm(df_local['momentum_30'], 'momentum_30'), n_features)

    # Kalman
    if use_kalman:
        kalman_trend, kalman_uncertainty = kalman_smooth_trend(close, Q=1e-5, R=0.01)
        features['kalman_trend'] = _align_to_features(_norm(kalman_trend, 'kalman_trend'), n_features)
        features['kalman_uncertainty'] = _align_to_features(_norm(kalman_uncertainty, 'kalman_uncertainty'), n_features)

        kf_rsi = KalmanFilter1D(Q=1e-4, R=0.1, initial_value=50.0)
        kalman_rsi, _ = kf_rsi.filter_series(df_local['rsi'])
        features['rsi_kalman'] = _align_to_features(_norm(kalman_rsi, 'rsi_kalman'), n_features)
    else:
        features['rsi'] = _align_to_features(_norm(df_local['rsi'], 'rsi'), n_features)

    # Base ATR
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = true_range.rolling(14).mean()
    features['atr_14'] = _align_to_features(_norm(atr_14, 'atr_14'), n_features)

    # Base Sharpe
    roll_mean_50 = returns.rolling(50).mean()
    roll_std_50 = returns.rolling(50).std()
    features['rolling_sharpe'] = _align_to_features(_norm((roll_mean_50/(roll_std_50+1e-9)).fillna(0), 'rolling_sharpe'), n_features)

    # Kaufman ER (Base)
    features['kaufman_efficiency_ratio'] = _align_to_features(_norm(get_efficiency_ratio(close, 30), 'kaufman_efficiency_ratio'), n_features)

    # ===== MTF LOOP =====
    for w in windows:
        # --- 2. ADVANCED VOLATILITY ---
        # Yang-Zhang
        yz_vol = compute_yang_zhang_volatility(open_p, high, low, close, window=w)
        features[f'yang_zhang_volatility_w{w}'] = _align_to_features(_norm(yz_vol, f'yang_zhang_volatility_w{w}'), n_features)

        # Rogers-Satchell
        rs_vol = compute_rogers_satchell_volatility(open_p, high, low, close, window=w)
        features[f'rogers_satchell_volatility_w{w}'] = _align_to_features(_norm(rs_vol, f'rogers_satchell_volatility_w{w}'), n_features)

        # Parkinson
        park_vol = compute_parkinson_volatility(high, low, window=w)
        features[f'parkinson_volatility_w{w}'] = _align_to_features(_norm(park_vol, f'parkinson_volatility_w{w}'), n_features)

        # Standard & Z-Score Vol
        vol_w = log_ret.rolling(w).std()
        features[f'volatility_w{w}'] = _align_to_features(_norm(vol_w, f'volatility_w{w}'), n_features)
        vol_z = compute_rolling_zscore(vol_w, window=w)
        features[f'volatility_zscore_w{w}'] = _align_to_features(_norm(vol_z, f'volatility_zscore_w{w}'), n_features)

        # --- 3. VOLATILITY REGIME DYNAMICS ---
        # Volatility Trend Slope
        vol_slope = vol_w.diff(max(1, w//4))
        features[f'volatility_trend_slope_w{w}'] = _align_to_features(_norm(vol_slope, f'volatility_trend_slope_w{w}'), n_features)

        # Vol Acceleration
        vol_accel = vol_slope.diff()
        features[f'vol_acceleration_w{w}'] = _align_to_features(_norm(vol_accel, f'vol_acceleration_w{w}'), n_features)

        # Compression Duration: Bars since BB Width expanded?
        # Or bars since vol > mean?
        # Let's use Bars Since Vol Spike (Z > 2)
        is_vol_spike = vol_z > 2.0
        features[f'bars_since_vol_spike_w{w}'] = _align_to_features(_norm(compute_bars_since(is_vol_spike), f'bars_since_vol_spike_w{w}'), n_features)

        # Compression: Low volatility state. Z < -1.0
        is_compression = vol_z < -1.0
        # Bars since compression ENDED = bars since NOT compression.
        # But commonly we want "Duration of current compression".
        # Which is `bars_since_not_compression`.
        features[f'vol_compression_duration_w{w}'] = _align_to_features(_norm(compute_bars_since(~is_compression), f'vol_compression_duration_w{w}'), n_features)

        # Bars since compression ended
        features[f'bars_since_compression_ended_w{w}'] = _align_to_features(_norm(compute_bars_since(is_compression), f'bars_since_compression_ended_w{w}'), n_features)

        # Breakout after compression: Interaction
        # If we broke out (price change high) AND we were compressed recently
        price_breakout = compute_rolling_zscore(returns, w).abs() > 2.0
        was_compressed = features[f'vol_compression_duration_w{w}'] > w
        features[f'breakout_after_compression_flag_w{w}'] = _align_to_features(_norm(price_breakout.astype(float) * was_compressed.astype(float), f'breakout_after_compression_flag_w{w}'), n_features)

        # --- 4. TREND QUALITY & EFFICIENCY ---
        # Efficiency Ratio (Kaufman)
        er_w = get_efficiency_ratio(close, window=w)
        features[f'trend_efficiency_ratio_w{w}'] = _align_to_features(_norm(er_w, f'trend_efficiency_ratio_w{w}'), n_features)

        # Directional Consistency: sum(sign(returns)) / w
        dir_consistency = np.sign(returns).rolling(w).sum() / w
        features[f'directional_consistency_w{w}'] = _align_to_features(_norm(dir_consistency.abs(), f'directional_consistency_w{w}'), n_features)

        # Trend Duration: Bars since MA slope flip
        ma_w = close.rolling(w).mean()
        ma_slope = ma_w.diff()
        slope_sign_change = np.sign(ma_slope) != np.sign(ma_slope.shift(1))
        features[f'trend_duration_w{w}'] = _align_to_features(_norm(compute_bars_since(slope_sign_change), f'trend_duration_w{w}'), n_features)

        # Momentum Decay
        roc = close.pct_change(w)
        max_roc = roc.rolling(w).max()
        features[f'momentum_decay_w{w}'] = _align_to_features(_norm(roc / (max_roc + 1e-9), f'momentum_decay_w{w}'), n_features)

        # Trend per Vol (Sharpe proxy)
        sharpe_w = (returns.rolling(w).mean() / (returns.rolling(w).std() + 1e-9)).fillna(0)
        features[f'trend_per_vol_w{w}'] = _align_to_features(_norm(sharpe_w, f'trend_per_vol_w{w}'), n_features)

        # Trend Slope Stability: Mean(Slope) / Std(Slope)
        slope = ma_w.diff()
        slope_stab = slope.rolling(w).mean() / (slope.rolling(w).std() + 1e-9)
        features[f'trend_slope_stability_w{w}'] = _align_to_features(_norm(slope_stab, f'trend_slope_stability_w{w}'), n_features)

        # --- 5. CYCLE & REGIME ---
        # Fisher Transform
        fisher = compute_fisher_transform(high, low, window=w)
        features[f'ehlers_fisher_transform_w{w}'] = _align_to_features(_norm(fisher, f'ehlers_fisher_transform_w{w}'), n_features)

        # Hilbert Phase
        # Only compute for one main window to save compute, or cheap version
        if w == 20:
            features[f'hilbert_transform_phase'] = _align_to_features(_norm(compute_hilbert_phase(close), f'hilbert_transform_phase'), n_features)

        # Aroon
        features[f'aroon_oscillator_w{w}'] = _align_to_features(_norm(compute_aroon(high, low, w), f'aroon_oscillator_w{w}'), n_features)

        # --- 6. VOLUME-PRICE EFFICIENCY ---
        if volume_available:
            # Price Impact: Abs(Ret) / Vol
            # Normalize volume first or ratio?
            # Log Return / Log Volume is better, or Amihud illiquidity
            impact = returns.abs() / (volume + 1e-9)
            features[f'price_impact_w{w}'] = _align_to_features(_norm(impact.rolling(w).mean(), f'price_impact_w{w}'), n_features)

            # Signed Price Impact
            signed_impact = returns / (volume + 1e-9)
            features[f'signed_price_impact_w{w}'] = _align_to_features(_norm(signed_impact.rolling(w).mean(), f'signed_price_impact_w{w}'), n_features)

            # Range per Volume (Kyber's liquidity metric)
            rpv = (high - low) / (volume + 1e-9)
            features[f'range_per_volume_w{w}'] = _align_to_features(_norm(rpv.rolling(w).mean(), f'range_per_volume_w{w}'), n_features)

            # Volume without progress (Churn): Vol * (1 - Efficiency)
            churn = volume * (1 - er_w)
            features[f'volume_without_progress_w{w}'] = _align_to_features(_norm(churn.rolling(w).mean(), f'volume_without_progress_w{w}'), n_features)

            # Delta-Volume Divergence: Corr(PriceChange, VolDelta)
            # Or Divergence between Price Trend and Vol Trend
            vol_trend = volume.rolling(w).mean().diff()
            price_trend = close.rolling(w).mean().diff()
            # Simple interaction
            features[f'delta_volume_divergence_w{w}'] = _align_to_features(_norm(vol_trend * price_trend, f'delta_volume_divergence_w{w}'), n_features)

            # Climax Volume: Vol > 3 * mean
            vol_mean = volume.rolling(w).mean()
            is_climax = volume > 3 * vol_mean
            features[f'climax_volume_flag_w{w}'] = _align_to_features(_norm(is_climax.astype(float), f'climax_volume_flag_w{w}'), n_features)

        # --- 7. VWAP & CONTEXT ---
        if volume_available:
            # VWAP Z-Score
            tp = (high + low + close) / 3
            # rolling vwap
            pv = tp * volume
            vwap = pv.rolling(w).sum() / (volume.rolling(w).sum() + 1e-9)
            vwap_std = tp.rolling(w).std() # Approx std dev of price
            features[f'vwap_zscore_w{w}'] = _align_to_features(_norm((close - vwap)/(vwap_std+1e-9), f'vwap_zscore_w{w}'), n_features)

            # EOM
            features[f'ease_of_movement_w{w}'] = _align_to_features(_norm(compute_ease_of_movement(high, low, volume, w), f'ease_of_movement_w{w}'), n_features)

            # MFI
            features[f'volume_weighted_rsi_w{w}'] = _align_to_features(_norm(compute_mfi(high, low, close, volume, w), f'volume_weighted_rsi_w{w}'), n_features)

        # --- 8. STRUCTURAL SR ---
        # Touch count: Count close near High/Low of window
        win_high = high.rolling(w).max()
        win_low = low.rolling(w).min()
        # Near = within 0.5% range?
        rng = win_high - win_low
        thresh = rng * 0.05
        # Current bar touch?
        touch_high = (high > (win_high - thresh)).astype(float)
        touch_low = (low < (win_low + thresh)).astype(float)
        # Sum touches in window
        features[f'touch_count_near_price_w{w}'] = _align_to_features(_norm((touch_high + touch_low).rolling(w).sum(), f'touch_count_near_price_w{w}'), n_features)

        # Time-weighted SR strength
        # Weighted sum of touches (decay over time)
        # Touch event series
        touch_series = (touch_high + touch_low)
        # Exponential moving sum
        features[f'time_weighted_sr_strength_w{w}'] = _align_to_features(_norm(touch_series.ewm(span=w).sum(), f'time_weighted_sr_strength_w{w}'), n_features)

        # Range Expansion Ratio: Range / Avg Range
        avg_range = (high - low).rolling(w).mean()
        features[f'range_expansion_ratio_w{w}'] = _align_to_features(_norm((high - low) / (avg_range + 1e-9), f'range_expansion_ratio_w{w}'), n_features)

        # Breakout Follow Through: (Close - BreakLevel) / BreakCandleSize
        # Defined as momentum continuity
        # Simplified: Return(t) * Return(t-1) > 0 ?
        features[f'breakout_follow_through_w{w}'] = _align_to_features(_norm((returns * returns.shift(1)), f'breakout_follow_through_w{w}'), n_features)

        # False Break Rate (Proxy)
        # Breakout: High > Rolling Max High (w)
        # False: Breakout AND Close < Rolling Max High (w)
        roll_high = high.rolling(w).max().shift(1)
        breakout = high > roll_high
        failed_break = breakout & (close < roll_high)
        # Rate in window
        features[f'false_break_rate_w{w}'] = _align_to_features(_norm(failed_break.rolling(w).sum() / (breakout.rolling(w).sum() + 1e-9), f'false_break_rate_w{w}'), n_features)

        # --- 9. TAIL RISK ---
        # Rolling Skew/Kurt
        features[f'rolling_skewness_w{w}'] = _align_to_features(_norm(returns.rolling(w).skew(), f'rolling_skewness_w{w}'), n_features)
        features[f'rolling_kurtosis_w{w}'] = _align_to_features(_norm(returns.rolling(w).kurt(), f'rolling_kurtosis_w{w}'), n_features)

        # Downside Semivariance
        neg_ret = returns.where(returns < 0, 0)
        downside_var = neg_ret.rolling(w).var()
        features[f'downside_semivariance_w{w}'] = _align_to_features(_norm(downside_var, f'downside_semivariance_w{w}'), n_features)

        # Left Tail Var Ratio
        total_var = returns.rolling(w).var()
        features[f'left_tail_var_ratio_w{w}'] = _align_to_features(_norm(downside_var / (total_var + 1e-9), f'left_tail_var_ratio_w{w}'), n_features)

        # Max Runup (MFE Proxy)
        # Max high relative to close
        # "How high did it go in last w bars?" relative to min in window?
        # Rolling Max - Rolling Min / Rolling Min
        win_min = low.rolling(w).min()
        win_max = high.rolling(w).max()
        features[f'max_runup_w{w}'] = _align_to_features(_norm((win_max - win_min) / (win_min + 1e-9), f'max_runup_w{w}'), n_features)
        # This is effectively max amplitude

        # Drawdown Depth (Current price vs Rolling Max)
        dd_w = (df['close'] / df['close'].rolling(w).max()) - 1.0
        features[f'drawdown_w{w}'] = _align_to_features(_norm(dd_w, f'drawdown_w{w}'), n_features) if use_kalman else _norm(dd_w, f'drawdown_w{w}')

        # Max Adverse Excursion (MAE) Proxy -> Alias to Drawdown
        features[f'max_adverse_excursion_w{w}'] = features[f'drawdown_w{w}']

        # Tail Event Flag
        is_tail = returns.abs() > (returns.rolling(w).std() * 3)
        features[f'tail_event_flag_w{w}'] = _align_to_features(_norm(is_tail.astype(float), f'tail_event_flag_w{w}'), n_features)

        # --- 11. EVENT BASED TIME IN STATE ---
        # Bars since breakout
        # Breakout = Close > Rolling High (prev)
        is_breakout = close > high.shift(1).rolling(w).max()
        features[f'bars_since_breakout_attempt_w{w}'] = _align_to_features(_norm(compute_bars_since(is_breakout), f'bars_since_breakout_attempt_w{w}'), n_features)

        # Bars since trend exhaustion (e.g. RSI > 80 or < 20)
        # Use RSI of window w
        rsi_w = compute_rsi(close, period=w)
        is_exhaustion = (rsi_w > 80) | (rsi_w < 20)
        features[f'bars_since_trend_exhaustion_signal_w{w}'] = _align_to_features(_norm(compute_bars_since(is_exhaustion), f'bars_since_trend_exhaustion_signal_w{w}'), n_features)

        # --- 12. INTERACTIONS ---
        # Trend Alignment: RSI direction == Slope direction
        rsi_slope = rsi_w.diff(max(1, w//4)) # Use window-aligned slope for RSI
        aligned = np.sign(rsi_slope) == np.sign(ma_slope)
        features[f'trend_alignment_score_w{w}'] = _align_to_features(_norm(aligned.astype(float), f'trend_alignment_score_w{w}'), n_features)

        # Vol Trend Conflict: Vol rising, Price falling (Panic?) or Vol falling, Price rising (Drift)
        # Vol Slope * Price Slope
        features[f'vol_trend_conflict_w{w}'] = _align_to_features(_norm(vol_slope * ma_slope, f'vol_trend_conflict_w{w}'), n_features)

        # Compression x Momentum (Already partially done)
        bb_width = compute_bollinger_bands(close, w)[3] # raw width
        mom_abs = roc.abs()
        features[f'compression_x_momentum_w{w}'] = _align_to_features(_norm(mom_abs / (bb_width + 1e-9), f'compression_x_momentum_w{w}'), n_features)

        # Absorption x Vol Spike
        # Absorption ~ (High Vol, Low Move). Vol Spike ~ High Vol.
        # Absorption defined in OFI section as absorption_ratio
        # We can construct local proxy: Vol / Range
        if volume_available:
            features[f'absorption_x_vol_spike_w{w}'] = _align_to_features(_norm(rpv * is_vol_spike.astype(float), f'absorption_x_vol_spike_w{w}'), n_features)

    # ===== LEGACY SUPPORT / CROSS-TIMEFRAME SPECIFIC =====
    # Add back some key legacy features if not covered
    close_1h = df['close'].rolling(4).mean()
    features['returns_1h'] = _align_to_features(_norm(close_1h.pct_change(), 'returns_1h'), n_features)
    close_4h = df['close'].rolling(16).mean()
    features['returns_4h'] = _align_to_features(_norm(close_4h.pct_change(), 'returns_4h'), n_features)

    return features
