"""
Multi-Timeframe Feature Generation Module.
Extracted and enhanced from feature_generation_meta_labeling_step.py.
"""
from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import logging

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

    CRITICAL: By default, does NOT include raw signal values to avoid circular behavior.
    Features capture market context, not the signals themselves.

    Args:
        df: DataFrame with OHLCV data
        signals: DataFrame with primary signals (used only for context)
        volume_available: Whether volume data is available
        include_raw_signals: WARNING: Set True only for ablation tests
        use_kalman: Whether to use Kalman filtering for indicators
        windows: List of timeframes for MTF feature generation

    Returns:
        DataFrame of features for meta-model
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
        Prioritizes:
        1. Log1p Z-Score for raw volume features (strictly positive magnitude)
        2. ATR Normalization for spatial/distance/level features
        3. Winsorized Z-Score (Robust Scaler) for everything else (default)
        """
        # Convert to Series if needed
        if not isinstance(data, pd.Series):
             # Try to recover index if possible, else default index
             if len(data) == len(df):
                 series = pd.Series(data, index=df.index)
             else:
                 series = pd.Series(data)
        else:
            series = data

        name_lower = name.lower()

        # 1. Volume features (raw magnitude)
        # Strictly exclude signed/relative metrics that might be negative or centered on zero.
        # log1p is undefined for x <= -1 and distorts negative values.
        # Exclude: ratio, trend, corr, rel, per, norm, imbalance, signed, pressure, delta, cvd, spike
        exclude_keywords = [
            'ratio', 'trend', 'corr', 'rel', 'per', 'norm',
            'imbalance', 'signed', 'pressure', 'delta', 'cvd', 'spike'
        ]

        if 'volume' in name_lower and not any(x in name_lower for x in exclude_keywords):
             # Use log1p_zscore for raw-ish volume (strictly positive)
             return log1p_zscore_normalize(series, window=600).fillna(0).to_numpy()

        # 2. Distance/Level features -> ATR Normalization
        # Exclude ratio/score/position/oscillator features that might match 'range' or 'level' keywords
        # but are actually unitless or 0-1 bounded.
        if should_use_atr_normalization(name):
             if any(x in name_lower for x in ['position', 'ratio', 'score', 'percent', 'pct', 'oscillator', 'index']):
                 pass # Fallback to winsorized
             else:
                 # atr_normalize expects the series to be in price units (distance)
                 return atr_normalize(series, df['high'], df['low'], df['close'], window=14).fillna(0).to_numpy()

        # 3. Default: Winsorized Z-Score (Robust Scaler)
        return winsorized_zscore_normalize(series, window=600).fillna(0).to_numpy()

    # ===== VOLATILITY FEATURES (ENHANCED) =====
    log_ret = np.log(df['close']).diff()
    features['log_ret'] = log_ret

    features['volatility_1h'] = _norm(log_ret.rolling(window=4).std(), 'volatility_1h')
    features['volatility_4h'] = _norm(log_ret.rolling(window=16).std(), 'volatility_4h')
    features['volatility_1d'] = _norm(log_ret.rolling(window=96).std(), 'volatility_1d')
    features['vol_of_vol'] = _norm(features['volatility_1h'].rolling(window=20).std(), 'vol_of_vol')

    # ===== VOLATILITY REGIME LABELING (Z-SCORE) =====
    vol_short_20 = log_ret.rolling(window=20).std()
    vol_long_mean = vol_short_20.rolling(window=200, min_periods=50).mean()
    vol_long_std = vol_short_20.rolling(window=200, min_periods=50).std()
    rv_z_short = (vol_short_20 - vol_long_mean) / (vol_long_std + 1e-8)
    features['rv_z_short'] = _norm(rv_z_short.fillna(0.0), 'rv_z_short')
    features['vol_ratio_20_200'] = _norm(vol_short_20 / (vol_long_mean + 1e-8), 'vol_ratio_20_200')
    features['volatility_20'] = _norm(vol_short_20, 'volatility_20')

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

    # ===== KALMAN-FILTERED TECHNICAL INDICATORS =====
    df_local = df.copy()
    df_local['rsi'] = compute_rsi(df_local['close'], period=14)
    df_local['sma_fast'] = df_local['close'].rolling(10).mean()
    df_local['sma_slow'] = df_local['close'].rolling(30).mean()
    df_local['momentum'] = df_local['close'].pct_change(10)

    # New: Add Momentum 30 (Mandatory feature)
    df_local['momentum_30'] = df_local['close'].pct_change(30)
    features['momentum_30'] = _align_to_features(_norm(df_local['momentum_30'], 'momentum_30'), n_features)

    if use_kalman:
        # Kalman-filtered trend
        kalman_trend, kalman_uncertainty = kalman_smooth_trend(df['close'], Q=1e-5, R=0.01)
        # kalman_trend is absolute price, probably not useful as feature directly.
        # But we normalize it anyway if requested. Usually trend features are diffs.
        # Let's keep existing logic but normalize.
        features['kalman_trend'] = _align_to_features(_norm(kalman_trend, 'kalman_trend'), n_features)
        features['kalman_uncertainty'] = _align_to_features(_norm(kalman_uncertainty, 'kalman_uncertainty'), n_features)

        # Kalman-filtered RSI
        kf_rsi = KalmanFilter1D(Q=1e-4, R=0.1, initial_value=50.0)
        kalman_rsi, _ = kf_rsi.filter_series(df_local['rsi'])
        features['rsi_kalman'] = _align_to_features(_norm(kalman_rsi, 'rsi_kalman'), n_features)

        # Kalman-filtered MA distance
        # MA distance is price distance. Should use ATR norm.
        ma_distance = df_local['sma_fast'] - df_local['sma_slow']
        kf_ma = KalmanFilter1D(Q=1e-5, R=0.01, initial_value=0.0)
        kalman_ma_distance, _ = kf_ma.filter_series(ma_distance)
        features['ma_distance_kalman'] = _align_to_features(_norm(kalman_ma_distance, 'ma_distance_kalman'), n_features)

        # Kalman-filtered momentum
        kf_mom = KalmanFilter1D(Q=1e-4, R=0.01, initial_value=0.0)
        kalman_momentum, _ = kf_mom.filter_series(df_local['momentum'])
        features['momentum_kalman'] = _align_to_features(_norm(kalman_momentum, 'momentum_kalman'), n_features)
    else:
        features['rsi'] = _norm(df_local['rsi'], 'rsi')
        features['ma_distance'] = _norm(df_local['sma_fast'] - df_local['sma_slow'], 'ma_distance')
        features['momentum'] = _norm(df_local['momentum'], 'momentum')

    # ===== VOLATILITY-NORMALIZED FEATURES =====
    # These are already ratios, so _norm will default to winsorized z-score which is fine.
    vol_1h_series = features['volatility_1h'].replace(0, np.nan)
    vol_1h_arr = _align_to_features(vol_1h_series, n_features) if use_kalman else vol_1h_series.to_numpy()
    close_arr = _align_to_features(df['close'], n_features) if use_kalman else df['close'].to_numpy()

    mom_feat = features['momentum_kalman'] if use_kalman else features['momentum']
    ma_feat = features['ma_distance_kalman'] if use_kalman else features['ma_distance']

    # Recalculate raw values for normalization context if needed, but these are ratios
    features['momentum_per_vol'] = _norm(mom_feat / (vol_1h_arr + 1e-8), 'momentum_per_vol')
    features['ma_distance_per_vol'] = _norm(ma_feat / (close_arr * vol_1h_arr + 1e-8), 'ma_distance_per_vol')

    # ===== TRADITIONAL VOLATILITY FEATURES =====
    returns = df['close'].pct_change()
    vol5_series = returns.rolling(5).std()
    vol20_series = returns.rolling(20).std()

    if use_kalman:
        features['volatility_5'] = _align_to_features(_norm(vol5_series, 'volatility_5'), n_features)
        features['volatility_20'] = _align_to_features(_norm(vol20_series, 'volatility_20'), n_features)
    else:
        features['volatility_5'] = _norm(vol5_series, 'volatility_5')
        features['volatility_20'] = _norm(vol20_series, 'volatility_20')

    features['volatility_ratio'] = _norm(features['volatility_5'] / (features['volatility_20'] + 1e-8), 'volatility_ratio')

    vol_ema_series = (returns**2).ewm(alpha=0.1, adjust=False).mean()
    vol_ema = _align_to_features(vol_ema_series, n_features) if use_kalman else vol_ema_series.to_numpy()
    features['volatility_ema'] = _norm(np.sqrt(vol_ema), 'volatility_ema')

    # ===== TREND STRENGTH =====
    sma_10 = df['close'].rolling(10).mean()
    sma_slope_series = sma_10.pct_change(5)
    sma20 = df['close'].rolling(20).mean()
    # price_vs_sma20 is usually a ratio (close-sma)/sma.
    # If we want ATR norm, we should use (close-sma).
    # Current name 'price_vs_sma20' implies ratio.
    # But 'distance_to_sma' triggers ATR norm.
    # Let's keep it as ratio and use winsorized.
    price_vs_sma20_series = (df['close'] - sma20) / (sma20 + 1e-8)

    # TRUE RANGE Calculation
    if 'high' in df.columns and 'low' in df.columns:
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    else:
        true_range = (df['close'] - df['close'].shift(1)).abs()

    # ATR 14 (Mandatory feature)
    atr_14_series = true_range.rolling(14).mean()

    if use_kalman:
        features['sma_slope'] = _align_to_features(_norm(sma_slope_series, 'sma_slope'), n_features)
        features['price_vs_sma20'] = _align_to_features(_norm(price_vs_sma20_series, 'price_vs_sma20'), n_features)
        features['atr_14'] = _align_to_features(_norm(atr_14_series, 'atr_14'), n_features)
    else:
        features['sma_slope'] = _norm(sma_slope_series, 'sma_slope')
        features['price_vs_sma20'] = _norm(price_vs_sma20_series, 'price_vs_sma20')
        features['atr_14'] = _norm(atr_14_series, 'atr_14')

    features['atr_ratio'] = _norm(features['atr_14'] / (close_arr + 1e-8), 'atr_ratio')

    # Rolling Sharpe (50) - Mandatory Feature
    roll_mean_50 = returns.rolling(50).mean()
    roll_std_50 = returns.rolling(50).std()
    rolling_sharpe_series = (roll_mean_50 / (roll_std_50 + 1e-9)).fillna(0)
    features['rolling_sharpe'] = _align_to_features(_norm(rolling_sharpe_series, 'rolling_sharpe'), n_features) if use_kalman else _norm(rolling_sharpe_series, 'rolling_sharpe')

    # ===== VOLUME CONTEXT =====
    if volume_available and 'volume' in df.columns:
        vol_sma = df['volume'].rolling(20).mean()
        volume_ratio_series = df['volume'] / (vol_sma + 1e-8)
        volume_trend_series = df['volume'].rolling(5).mean() / (vol_sma + 1e-8)
        vol_price_corr_series = returns.rolling(20).corr(df['volume'].pct_change())

        # volume_zscore should be log1p_zscore normalized raw volume now, as per instruction
        # But here it calculates manual zscore.
        # We replace this with _norm(df['volume'])
        volume_zscore_series = df['volume']

        volume_spike_series = df['volume'] / (df['volume'].rolling(96).mean() + 1e-8)

        signed_volume_raw = np.sign(returns.fillna(0.0).to_numpy()) * df['volume'].to_numpy()
        signed_volume_ema_series = pd.Series(signed_volume_raw).ewm(span=20).mean()

        if use_kalman:
            features['volume_ratio'] = _align_to_features(_norm(volume_ratio_series, 'volume_ratio'), n_features)
            features['volume_trend'] = _align_to_features(_norm(volume_trend_series, 'volume_trend'), n_features)
            features['vol_price_corr'] = _align_to_features(_norm(vol_price_corr_series, 'vol_price_corr'), n_features)
            features['volume_zscore'] = _align_to_features(_norm(volume_zscore_series, 'volume_zscore'), n_features)
            features['volume_spike'] = _align_to_features(_norm(volume_spike_series, 'volume_spike'), n_features)
            features['signed_volume_ema'] = _align_to_features(_norm(signed_volume_ema_series, 'signed_volume_ema'), n_features)
        else:
            features['volume_ratio'] = _norm(volume_ratio_series, 'volume_ratio')
            features['volume_trend'] = _norm(volume_trend_series, 'volume_trend')
            features['vol_price_corr'] = _norm(vol_price_corr_series, 'vol_price_corr')
            features['volume_zscore'] = _norm(volume_zscore_series, 'volume_zscore')
            features['volume_spike'] = _norm(volume_spike_series, 'volume_spike')
            features['signed_volume_ema'] = _norm(signed_volume_ema_series, 'signed_volume_ema')
    else:
        features['volume_ratio'] = 1.0
        features['volume_trend'] = 1.0
        features['vol_price_corr'] = 0.0
        features['volume_zscore'] = 0.0
        features['volume_spike'] = 1.0
        features['signed_volume_ema'] = 0.0

    # ===== MARKET MOMENTUM =====
    mom5_series = df['close'].pct_change(5)
    mom10_series = df['close'].pct_change(10)
    mom20_series = df['close'].pct_change(20)

    if use_kalman:
        features['momentum_5'] = _align_to_features(_norm(mom5_series, 'momentum_5'), n_features)
        features['momentum_10'] = _align_to_features(_norm(mom10_series, 'momentum_10'), n_features)
        features['momentum_20'] = _align_to_features(_norm(mom20_series, 'momentum_20'), n_features)
        features['momentum_ema'] = _align_to_features(_norm(mom10_series.ewm(span=5).mean(), 'momentum_ema'), n_features)
    else:
        features['momentum_5'] = _norm(mom5_series, 'momentum_5')
        features['momentum_10'] = _norm(mom10_series, 'momentum_10')
        features['momentum_20'] = _norm(mom20_series, 'momentum_20')
        features['momentum_ema'] = _norm(mom10_series.ewm(span=5).mean(), 'momentum_ema')

    # ACF
    autocorr_series = returns.rolling(window=50, min_periods=10).corr(returns.shift(1))
    features['return_autocorr_lag1_w50'] = _align_to_features(_norm(autocorr_series, 'return_autocorr_lag1_w50'), n_features) if use_kalman else _norm(autocorr_series, 'return_autocorr_lag1_w50')

    # ===== RANGE POSITION =====
    recent_high = df['high'].rolling(20).max()
    recent_low = df['low'].rolling(20).min()
    range_position_series = (df['close'] - recent_low) / (recent_high - recent_low + 1e-8)
    features['range_position'] = _align_to_features(_norm(range_position_series, 'range_position'), n_features) if use_kalman else _norm(range_position_series, 'range_position')

    if 'close' in df.columns and 'volume' in df.columns:
        try:
            dollar_volume = df['close'] * df['volume']
            cum_volume = df['volume'].cumsum()
            vwap_series = dollar_volume.cumsum() / (cum_volume + 1e-8)
            vwap_diff_series = df['close'] - vwap_series
            # vwap_diff is price distance
            features['close_minus_vwap'] = _align_to_features(_norm(vwap_diff_series, 'close_minus_vwap'), n_features) if use_kalman else _norm(vwap_diff_series, 'close_minus_vwap')
        except: pass

    # ===== ENTROPY =====
    returns_abs = returns.abs().rolling(20).mean()
    returns_entropy_series = -returns_abs * np.log(returns_abs + 1e-8)
    features['returns_entropy'] = _align_to_features(_norm(returns_entropy_series, 'returns_entropy'), n_features) if use_kalman else _norm(returns_entropy_series, 'returns_entropy')

    # Hurst Proxy
    features['hurst_100'] = _align_to_features(_norm(compute_hurst_proxy(df['close'], 100), 'hurst_100'), n_features) if use_kalman else _norm(compute_hurst_proxy(df['close'], 100), 'hurst_100')

    # ===== ADVANCED MOMENTUM =====
    # Stochastic
    stoch_k, stoch_d = compute_stochastic(df['high'], df['low'], df['close'])
    features['stoch_k'] = _align_to_features(_norm(stoch_k, 'stoch_k'), n_features) if use_kalman else _norm(stoch_k, 'stoch_k')
    features['stoch_d'] = _align_to_features(_norm(stoch_d, 'stoch_d'), n_features) if use_kalman else _norm(stoch_d, 'stoch_d')

    # CCI
    features['cci_14'] = _align_to_features(_norm(compute_cci(df['high'], df['low'], df['close'], 14), 'cci_14'), n_features) if use_kalman else _norm(compute_cci(df['high'], df['low'], df['close'], 14), 'cci_14')
    features['cci_40'] = _align_to_features(_norm(compute_cci(df['high'], df['low'], df['close'], 40), 'cci_40'), n_features) if use_kalman else _norm(compute_cci(df['high'], df['low'], df['close'], 40), 'cci_40')

    # ADX
    adx, plus_di, minus_di = compute_adx(df['high'], df['low'], df['close'], 14)
    features['adx_14'] = _align_to_features(_norm(adx, 'adx_14'), n_features) if use_kalman else _norm(adx, 'adx_14')
    features['plus_di_14'] = _align_to_features(_norm(plus_di, 'plus_di_14'), n_features) if use_kalman else _norm(plus_di, 'plus_di_14')
    features['minus_di_14'] = _align_to_features(_norm(minus_di, 'minus_di_14'), n_features) if use_kalman else _norm(minus_di, 'minus_di_14')
    features['adx_trend'] = _norm(features['plus_di_14'] - features['minus_di_14'], 'adx_trend') # Simple trend strength

    # ===== ADVANCED VOLATILITY =====
    # Bollinger Bands
    # bb_width usually calculated as % (width / middle).
    # If we want ATR norm, we need raw width (upper - lower).
    # 'bb_width' triggers ATR norm.
    upper, middle, lower, width_pct = compute_bollinger_bands(df['close'], 20, 2.0)
    width_raw = upper - lower
    features['bb_width'] = _align_to_features(_norm(width_raw, 'bb_width'), n_features) if use_kalman else _norm(width_raw, 'bb_width')
    features['price_vs_bb'] = _align_to_features(_norm((df['close'] - lower) / (upper - lower + 1e-9), 'price_vs_bb'), n_features) if use_kalman else _norm((df['close'] - lower) / (upper - lower + 1e-9), 'price_vs_bb')

    # Choppiness
    features['choppiness_14'] = _align_to_features(_norm(compute_choppiness_index(df['high'], df['low'], df['close'], 14), 'choppiness_14'), n_features) if use_kalman else _norm(compute_choppiness_index(df['high'], df['low'], df['close'], 14), 'choppiness_14')

    # Parkinson Volatility
    features['parkinson_volatility_20'] = _align_to_features(_norm(compute_parkinson_volatility(df['high'], df['low'], 20), 'parkinson_volatility_20'), n_features) if use_kalman else _norm(compute_parkinson_volatility(df['high'], df['low'], 20), 'parkinson_volatility_20')

    # ===== ADVANCED VOLUME =====
    if volume_available and 'volume' in df.columns:
        features['cmf_20'] = _align_to_features(_norm(compute_cmf(df['high'], df['low'], df['close'], df['volume'], 20), 'cmf_20'), n_features) if use_kalman else _norm(compute_cmf(df['high'], df['low'], df['close'], df['volume'], 20), 'cmf_20')
        features['force_index_13'] = _align_to_features(_norm(compute_force_index(df['close'], df['volume'], 13), 'force_index_13'), n_features) if use_kalman else _norm(compute_force_index(df['close'], df['volume'], 13), 'force_index_13')
        # Normalized Force Index (by volume MA)
        fi_norm = features['force_index_13'] / (df['volume'].rolling(20).mean() * df['close'] + 1e-9)
        features['force_index_norm'] = _norm(fi_norm, 'force_index_norm')

    # ===== INTERACTION FEATURES =====
    # Volatility * Momentum
    if 'volatility_20' in features.columns and 'momentum_20' in features.columns:
        features['vol_x_mom'] = _norm(features['volatility_20'] * features['momentum_20'], 'vol_x_mom')

    # ADX * Trend
    if 'adx_14' in features.columns and 'momentum_10' in features.columns:
        features['adx_x_mom'] = _norm((features['adx_14'] / 100.0) * features['momentum_10'], 'adx_x_mom')

    # BB Squeeze * Volume
    if 'bb_width' in features.columns and 'volume_ratio' in features.columns:
        features['squeeze_x_vol'] = _norm((1.0 / (features['bb_width'] + 1e-9)) * features['volume_ratio'], 'squeeze_x_vol')

    # ===== TIME-BASED FEATURES =====
    if isinstance(df.index, pd.DatetimeIndex):
        hour_arr = df.index.hour.to_numpy()
        dow_arr = df.index.dayofweek.to_numpy()
        features['hour'] = hour_arr
        features['day_of_week'] = dow_arr
        features['hour_sin'] = np.sin(2 * np.pi * hour_arr / 24.0)
        features['hour_cos'] = np.cos(2 * np.pi * hour_arr / 24.0)
        features['is_good_hour'] = np.isin(hour_arr, [3, 5, 10]).astype(float)
        features['is_bad_hour'] = np.isin(hour_arr, [0, 13, 19]).astype(float)
        features['is_sunday'] = (dow_arr == 6).astype(float)
    else:
        features['hour'] = 0
        features['day_of_week'] = 0
        features['hour_sin'] = 0.0
        features['hour_cos'] = 1.0
        features['is_good_hour'] = 0.0
        features['is_bad_hour'] = 0.0
        features['is_sunday'] = 0.0

    # ===== ORDER FLOW IMBALANCE (OFI) PROXY =====
    if 'volume' in df.columns:
        volume = df['volume']
        close = df['close']
        high = df['high']
        low = df['low']
        open_price = df.get('open', close)

        price_direction = np.sign(close - open_price)
        signed_volume = volume * price_direction
        cvd_proxy = signed_volume.cumsum()
        cvd_normalized = (cvd_proxy - cvd_proxy.rolling(96).mean()) / (cvd_proxy.rolling(96).std() + 1e-8)
        features['cvd_proxy'] = _align_to_features(_norm(cvd_normalized, 'cvd_proxy'), n_features) if use_kalman else _norm(cvd_normalized, 'cvd_proxy')

        close_in_range = (close - low) / (high - low + 1e-8)
        volume_pressure = (close_in_range - 0.5) * volume
        volume_pressure_ema = volume_pressure.ewm(span=20).mean()
        features['volume_pressure'] = _align_to_features(_norm(volume_pressure_ema, 'volume_pressure'), n_features) if use_kalman else _norm(volume_pressure_ema, 'volume_pressure')

        upper_wick = high - pd.concat([open_price, close], axis=1).max(axis=1)
        lower_wick = pd.concat([open_price, close], axis=1).min(axis=1) - low
        total_range = high - low + 1e-8
        supply_rejection = (upper_wick / total_range) * volume
        demand_rejection = (lower_wick / total_range) * volume
        ofi_proxy = (demand_rejection - supply_rejection).rolling(20).sum()
        ofi_normalized = ofi_proxy / (ofi_proxy.rolling(96).std() + 1e-8)
        features['ofi_proxy'] = _align_to_features(_norm(ofi_normalized, 'ofi_proxy'), n_features) if use_kalman else _norm(ofi_normalized, 'ofi_proxy')

        buy_volume = volume * close_in_range
        sell_volume = volume * (1 - close_in_range)
        volume_imbalance = (buy_volume - sell_volume) / (volume + 1e-8)
        volume_imbalance_ema = volume_imbalance.ewm(span=20).mean()
        features['volume_imbalance'] = _align_to_features(_norm(volume_imbalance_ema, 'volume_imbalance'), n_features) if use_kalman else _norm(volume_imbalance_ema, 'volume_imbalance')

        is_at_extreme = (close_in_range < 0.2) | (close_in_range > 0.8)
        extreme_volume = volume.where(is_at_extreme, 0).rolling(20).sum()
        total_volume = volume.rolling(20).sum()
        absorption_ratio = extreme_volume / (total_volume + 1e-8)
        features['absorption_ratio'] = _align_to_features(_norm(absorption_ratio, 'absorption_ratio'), n_features) if use_kalman else _norm(absorption_ratio, 'absorption_ratio')

        trade_aggressor_ratio_series = close_in_range.ewm(span=20).mean()
        features['trade_aggressor_ratio'] = _align_to_features(_norm(trade_aggressor_ratio_series, 'trade_aggressor_ratio'), n_features) if use_kalman else _norm(trade_aggressor_ratio_series, 'trade_aggressor_ratio')

        prev_close = close.shift(1)
        gap_raw = open_price - prev_close
        # liquidity_gap is not ratio, it's normalized here manually.
        # But user wants standard transforms.
        # "gap" triggers ATR norm.
        # So we should pass raw gap if possible.
        features['liquidity_gap_up'] = _align_to_features(_norm(np.maximum(gap_raw, 0), 'liquidity_gap_up'), n_features) if use_kalman else _norm(np.maximum(gap_raw, 0), 'liquidity_gap_up')
        features['liquidity_gap_down'] = _align_to_features(_norm(np.maximum(-gap_raw, 0), 'liquidity_gap_down'), n_features) if use_kalman else _norm(np.maximum(-gap_raw, 0), 'liquidity_gap_down')
        features['liquidity_gap_abs'] = _align_to_features(_norm(gap_raw.abs(), 'liquidity_gap_abs'), n_features) if use_kalman else _norm(gap_raw.abs(), 'liquidity_gap_abs')
    else:
        features['cvd_proxy'] = 0.0
        features['volume_pressure'] = 0.0
        features['ofi_proxy'] = 0.0
        features['volume_imbalance'] = 0.0
        features['absorption_ratio'] = 0.0
        features['trade_aggressor_ratio'] = 0.5
        features['liquidity_gap_up'] = 0.0
        features['liquidity_gap_down'] = 0.0
        features['liquidity_gap_abs'] = 0.0

    # ===== SIGNAL FEATURES =====
    if 'consensus' in signals.columns:
        signal_active = (signals['consensus'] != 0).astype(int)
        features['signal_active'] = _align_to_features(signal_active, n_features) if use_kalman else signal_active.to_numpy()
    else:
        features['signal_active'] = 0

    base_signal_cols = [col for col in ['rsi', 'rsi_long', 'macd', 'macd_long', 'ma', 'mom'] if col in signals.columns]
    if base_signal_cols:
        abs_signals = signals[base_signal_cols].abs()
        features['signal_strength_all'] = _align_to_features(_norm(abs_signals.sum(axis=1), 'signal_strength_all'), n_features) if use_kalman else _norm(abs_signals.sum(axis=1), 'signal_strength_all')
        features['signal_count_active'] = _align_to_features(_norm((abs_signals > 0).sum(axis=1), 'signal_count_active'), n_features) if use_kalman else _norm((abs_signals > 0).sum(axis=1), 'signal_count_active')

        if 'rsi' in signals.columns and 'macd' in signals.columns:
            align_series = np.sign(signals['rsi'] * signals['macd']).replace(0, 0)
            features['signal_rsi_macd_alignment'] = _align_to_features(align_series, n_features) if use_kalman else align_series.to_numpy()

    # Pass through signal component values
    for col, new_col in [('rsi_value', 'signal_rsi_distance_50'), ('rsi_long_value', 'signal_rsi_long_distance_50')]:
        if col in signals.columns:
            dist = (signals[col] - 50.0).abs()
            features[new_col] = _align_to_features(_norm(dist, new_col.replace('distance', 'diff')), n_features) if use_kalman else _norm(dist, new_col.replace('distance', 'diff'))

    if 'macd_hist_value' in signals.columns:
        features['signal_macd_hist_abs'] = _align_to_features(_norm(signals['macd_hist_value'].abs(), 'signal_macd_hist_abs'), n_features) if use_kalman else _norm(signals['macd_hist_value'].abs(), 'signal_macd_hist_abs')
    if 'macd_hist_long_value' in signals.columns:
        features['signal_macd_hist_long_abs'] = _align_to_features(_norm(signals['macd_hist_long_value'].abs(), 'signal_macd_hist_long_abs'), n_features) if use_kalman else _norm(signals['macd_hist_long_value'].abs(), 'signal_macd_hist_long_abs')

    if 'sma_fast_value' in signals.columns and 'sma_slow_value' in signals.columns:
        ma_dist = (signals['sma_fast_value'] - signals['sma_slow_value'])
        features['signal_ma_distance_raw'] = _align_to_features(_norm(ma_dist, 'signal_ma_distance_raw'), n_features) if use_kalman else _norm(ma_dist, 'signal_ma_distance_raw')

    if 'momentum_value' in signals.columns:
        features['signal_momentum_value'] = _align_to_features(_norm(signals['momentum_value'], 'signal_momentum_value'), n_features) if use_kalman else _norm(signals['momentum_value'], 'signal_momentum_value')

    # Pass through regime flags
    for col in ['trend_regime', 'candle_trend', 'candle_reversal']:
        if col in signals.columns:
            features[col] = _align_to_features(signals[col], n_features) if use_kalman else signals[col].to_numpy()

    # Interactions
    if 'trend_regime' in features.columns and 'signal_macd_hist_abs' in features.columns:
        features['signal_trend_regime_x_macd_hist_abs'] = _norm(features['trend_regime'] * features['signal_macd_hist_abs'], 'signal_trend_regime_x_macd_hist_abs')
    if 'candle_trend' in features.columns and 'signal_rsi_distance_50' in features.columns:
        features['signal_candle_trend_x_rsi_distance_50'] = _norm(features['candle_trend'] * features['signal_rsi_distance_50'], 'signal_candle_trend_x_rsi_distance_50')

    # ===== CROSS-TIMEFRAME FEATURES =====
    close_1h = df['close'].rolling(4).mean()
    features['returns_1h'] = _align_to_features(_norm(close_1h.pct_change(), 'returns_1h'), n_features) if use_kalman else _norm(close_1h.pct_change(), 'returns_1h')
    features['momentum_1h'] = _align_to_features(_norm(df['close'].pct_change(4), 'momentum_1h'), n_features) if use_kalman else _norm(df['close'].pct_change(4), 'momentum_1h')
    features['volatility_1h_agg'] = _align_to_features(_norm(close_1h.pct_change().rolling(16).std(), 'volatility_1h_agg'), n_features) if use_kalman else _norm(close_1h.pct_change().rolling(16).std(), 'volatility_1h_agg')

    close_4h = df['close'].rolling(16).mean()
    features['returns_4h'] = _align_to_features(_norm(close_4h.pct_change(), 'returns_4h'), n_features) if use_kalman else _norm(close_4h.pct_change(), 'returns_4h')
    features['momentum_4h'] = _align_to_features(_norm(df['close'].pct_change(16), 'momentum_4h'), n_features) if use_kalman else _norm(df['close'].pct_change(16), 'momentum_4h')
    features['volatility_4h_agg'] = _align_to_features(_norm(close_4h.pct_change().rolling(16).std(), 'volatility_4h_agg'), n_features) if use_kalman else _norm(close_4h.pct_change().rolling(16).std(), 'volatility_4h_agg')

    # 4H RSI Proxy
    rsi_4h = compute_rsi(close_4h, 14)
    features['rsi_4h'] = _align_to_features(_norm(rsi_4h, 'rsi_4h'), n_features) if use_kalman else _norm(rsi_4h, 'rsi_4h')

    # Kaufman ER (Mandatory feature)
    features['kaufman_efficiency_ratio'] = _align_to_features(_norm(get_efficiency_ratio(df['close'], 30), 'kaufman_efficiency_ratio'), n_features) if use_kalman else _norm(get_efficiency_ratio(df['close'], 30), 'kaufman_efficiency_ratio')

    # ACF Mean
    acf_vals = []
    for lag in [1, 2, 5]:
        acf_vals.append(log_ret.rolling(20).corr(log_ret.shift(lag)))
    features['acf_mean_lags_1_2_5'] = _align_to_features(_norm(pd.concat(acf_vals, axis=1).mean(axis=1), 'acf_mean_lags_1_2_5'), n_features) if use_kalman else _norm(pd.concat(acf_vals, axis=1).mean(axis=1), 'acf_mean_lags_1_2_5')

    if include_raw_signals:
        features['signal_strength'] = _norm(signals[['rsi', 'ma', 'mom']].abs().sum(axis=1), 'signal_strength')
        features['signal_consensus'] = _norm(signals['consensus'].abs(), 'signal_consensus')

    # ===== MTF LOOP (Enhanced) =====
    # Generate key features on multiple timeframes
    for w in windows:
        # Removed hard check to skip 20 to ensure all features are generated for all windows
        # if w == 20: continue

        # ===== BASIC ROLLING FEATURES (Previously in separate loop) =====
        features[f'returns_mean_{w}'] = _align_to_features(_norm(returns.rolling(w).mean(), f'returns_mean_{w}'), n_features) if use_kalman else _norm(returns.rolling(w).mean(), f'returns_mean_{w}')
        features[f'returns_std_{w}'] = _align_to_features(_norm(returns.rolling(w).std(), f'returns_std_{w}'), n_features) if use_kalman else _norm(returns.rolling(w).std(), f'returns_std_{w}')

        # Use HIGH and LOW for max/min if available, else fall back to close
        if 'high' in df.columns and 'low' in df.columns:
            win_high = df['high'].rolling(w).max()
            win_low = df['low'].rolling(w).min()
        else:
            win_high = df['close'].rolling(w).max()
            win_low = df['close'].rolling(w).min()

        # Close min/max based on Close
        close_min = df['close'].rolling(w).min()
        close_max = df['close'].rolling(w).max()
        features[f'close_min_{w}'] = _align_to_features(_norm(close_min, f'close_min_{w}'), n_features) if use_kalman else _norm(close_min, f'close_min_{w}')
        features[f'close_max_{w}'] = _align_to_features(_norm(close_max, f'close_max_{w}'), n_features) if use_kalman else _norm(close_max, f'close_max_{w}')

        # For drawdown, we usually look at close vs rolling max close
        rolling_max_close = df['close'].rolling(w).max()
        drawdown_val = (df['close'] - rolling_max_close) / (rolling_max_close + 1e-8)
        features[f'drawdown_{w}'] = _align_to_features(_norm(drawdown_val, f'drawdown_{w}'), n_features) if use_kalman else _norm(drawdown_val, f'drawdown_{w}')

        # Close range uses high/low extremes
        close_range = (win_high - win_low) / (df['close'] + 1e-8)
        features[f'close_range_{w}'] = _align_to_features(_norm(close_range, f'close_range_{w}'), n_features) if use_kalman else _norm(close_range, f'close_range_{w}')

        # Distance from recent high/low
        # Change to raw distance for ATR normalization
        dist_high = df['close'] - win_high
        dist_low = df['close'] - win_low
        features[f'dist_from_recent_high_{w}'] = _align_to_features(_norm(dist_high, f'dist_from_recent_high_{w}'), n_features) if use_kalman else _norm(dist_high, f'dist_from_recent_high_{w}')
        features[f'dist_from_recent_low_{w}'] = _align_to_features(_norm(dist_low, f'dist_from_recent_low_{w}'), n_features) if use_kalman else _norm(dist_low, f'dist_from_recent_low_{w}')

        # Entropy (MTF)
        returns_abs_w = returns.abs().rolling(w).mean()
        returns_entropy_w = -returns_abs_w * np.log(returns_abs_w + 1e-8)
        features[f'returns_entropy_w{w}'] = _align_to_features(_norm(returns_entropy_w, f'returns_entropy_w{w}'), n_features) if use_kalman else _norm(returns_entropy_w, f'returns_entropy_w{w}')

        # Volatility
        vol_w = log_ret.rolling(w).std()
        features[f'volatility_w{w}'] = _align_to_features(_norm(vol_w, f'volatility_w{w}'), n_features) if use_kalman else _norm(vol_w, f'volatility_w{w}')

        # Returns Std (for compatibility with return_std_{horizon}b)
        returns_std_w = returns.rolling(w).std()
        features[f'returns_std_w{w}'] = _align_to_features(_norm(returns_std_w, f'returns_std_w{w}'), n_features) if use_kalman else _norm(returns_std_w, f'returns_std_w{w}')

        # Momentum / Simple Return
        mom_w = df['close'].pct_change(w)
        if use_kalman:
            kf = KalmanFilter1D(Q=1e-4, R=0.01)
            mom_w_filt, _ = kf.filter_series(mom_w)
            features[f'momentum_w{w}'] = _align_to_features(_norm(mom_w_filt, f'momentum_w{w}'), n_features)
            features[f'return_w{w}'] = features[f'momentum_w{w}']
        else:
            features[f'momentum_w{w}'] = _norm(mom_w, f'momentum_w{w}')
            features[f'return_w{w}'] = features[f'momentum_w{w}']

        # RSI
        rsi_w = compute_rsi(df['close'], period=w)
        features[f'rsi_w{w}'] = _align_to_features(_norm(rsi_w, f'rsi_w{w}'), n_features) if use_kalman else _norm(rsi_w, f'rsi_w{w}')

        # Efficiency
        eff_w = get_efficiency_ratio(df['close'], window=w)
        features[f'kaufman_efficiency_ratio_w{w}'] = _align_to_features(_norm(eff_w, f'kaufman_efficiency_ratio_w{w}'), n_features) if use_kalman else _norm(eff_w, f'kaufman_efficiency_ratio_w{w}')

        # Range Position
        high_w = df['high'].rolling(w).max()
        low_w = df['low'].rolling(w).min()
        rng_pos_w = (df['close'] - low_w) / (high_w - low_w + 1e-8)
        features[f'range_position_w{w}'] = _align_to_features(_norm(rng_pos_w, f'range_position_w{w}'), n_features) if use_kalman else _norm(rng_pos_w, f'range_position_w{w}')

        # Autocorr
        ac_w = returns.rolling(window=w, min_periods=max(5, w//4)).corr(returns.shift(1))
        features[f'autocorr_w{w}'] = _align_to_features(_norm(ac_w, f'autocorr_w{w}'), n_features) if use_kalman else _norm(ac_w, f'autocorr_w{w}')

        # SMA Slope (MTF)
        # Use w for SMA window, and w//2 for slope lookback
        sma_w = df['close'].rolling(w).mean()
        slope_w = sma_w.pct_change(max(1, w//2))
        features[f'sma_slope_w{w}'] = _align_to_features(_norm(slope_w, f'sma_slope_w{w}'), n_features) if use_kalman else _norm(slope_w, f'sma_slope_w{w}')

        # Price vs SMA (MTF)
        # Ratio
        price_vs_sma_w = (df['close'] - sma_w) / (sma_w + 1e-8)
        features[f'price_vs_sma_w{w}'] = _align_to_features(_norm(price_vs_sma_w, f'price_vs_sma_w{w}'), n_features) if use_kalman else _norm(price_vs_sma_w, f'price_vs_sma_w{w}')

        # ATR Ratio (MTF)
        # Use True Range logic if high/low available
        if 'high' in df.columns and 'low' in df.columns:
            high_w = df['high']
            low_w = df['low']
            close_w = df['close']
            tr1_w = high_w - low_w
            tr2_w = (high_w - close_w.shift(1)).abs()
            tr3_w = (low_w - close_w.shift(1)).abs()
            tr_w = pd.concat([tr1_w, tr2_w, tr3_w], axis=1).max(axis=1)
        else:
            tr_w = (df['close'] - df['close'].shift(1)).abs()

        atr_w = tr_w.rolling(w).mean()
        atr_ratio_w = atr_w / (df['close'] + 1e-8)
        features[f'atr_ratio_w{w}'] = _align_to_features(_norm(atr_ratio_w, f'atr_ratio_w{w}'), n_features) if use_kalman else _norm(atr_ratio_w, f'atr_ratio_w{w}')

        # Rolling Sharpe (MTF)
        sharpe_mean = returns.rolling(w).mean()
        sharpe_std = returns.rolling(w).std()
        sharpe_w = (sharpe_mean / (sharpe_std + 1e-9)).fillna(0)
        features[f'rolling_sharpe_w{w}'] = _align_to_features(_norm(sharpe_w, f'rolling_sharpe_w{w}'), n_features) if use_kalman else _norm(sharpe_w, f'rolling_sharpe_w{w}')

        # Volume Metrics (MTF)
        if volume_available and 'volume' in df.columns:
            vol_sma_w = df['volume'].rolling(w).mean()
            vol_ratio_w = df['volume'] / (vol_sma_w + 1e-8)
            features[f'volume_ratio_w{w}'] = _align_to_features(_norm(vol_ratio_w, f'volume_ratio_w{w}'), n_features) if use_kalman else _norm(vol_ratio_w, f'volume_ratio_w{w}')

            # Volume Trend (Short SMA / Long SMA) - using w as Long, w/4 as Short
            vol_short_sma = df['volume'].rolling(max(1, w//4)).mean()
            vol_trend_w = vol_short_sma / (vol_sma_w + 1e-8)
            features[f'volume_trend_w{w}'] = _align_to_features(_norm(vol_trend_w, f'volume_trend_w{w}'), n_features) if use_kalman else _norm(vol_trend_w, f'volume_trend_w{w}')

            # CMF (MTF)
            cmf_w = compute_cmf(df['high'], df['low'], df['close'], df['volume'], w)
            features[f'cmf_w{w}'] = _align_to_features(_norm(cmf_w, f'cmf_w{w}'), n_features) if use_kalman else _norm(cmf_w, f'cmf_w{w}')

            # Force Index (MTF)
            fi_w = compute_force_index(df['close'], df['volume'], w)
            # Normalize Force Index
            fi_norm_w = fi_w / (vol_sma_w * df['close'] + 1e-9)
            features[f'force_index_w{w}'] = _align_to_features(_norm(fi_norm_w, f'force_index_w{w}'), n_features) if use_kalman else _norm(fi_norm_w, f'force_index_w{w}')

            # Volume-Price Correlation (MTF)
            vol_price_corr_w = returns.rolling(w).corr(df['volume'].pct_change())
            features[f'vol_price_corr_w{w}'] = _align_to_features(_norm(vol_price_corr_w, f'vol_price_corr_w{w}'), n_features) if use_kalman else _norm(vol_price_corr_w, f'vol_price_corr_w{w}')

        # Stochastic (MTF)
        stoch_k_w, stoch_d_w = compute_stochastic(df['high'], df['low'], df['close'], k_period=w, d_period=max(3, w // 5))
        features[f'stoch_k_w{w}'] = _align_to_features(_norm(stoch_k_w, f'stoch_k_w{w}'), n_features) if use_kalman else _norm(stoch_k_w, f'stoch_k_w{w}')
        features[f'stoch_d_w{w}'] = _align_to_features(_norm(stoch_d_w, f'stoch_d_w{w}'), n_features) if use_kalman else _norm(stoch_d_w, f'stoch_d_w{w}')

        # CCI (MTF)
        cci_w = compute_cci(df['high'], df['low'], df['close'], period=w)
        features[f'cci_w{w}'] = _align_to_features(_norm(cci_w, f'cci_w{w}'), n_features) if use_kalman else _norm(cci_w, f'cci_w{w}')

        # ADX (MTF)
        adx_w, plus_di_w, minus_di_w = compute_adx(df['high'], df['low'], df['close'], period=w)
        features[f'adx_w{w}'] = _align_to_features(_norm(adx_w, f'adx_w{w}'), n_features) if use_kalman else _norm(adx_w, f'adx_w{w}')
        features[f'adx_trend_w{w}'] = _align_to_features(_norm(plus_di_w - minus_di_w, f'adx_trend_w{w}'), n_features) if use_kalman else _norm(plus_di_w - minus_di_w, f'adx_trend_w{w}')

        # Bollinger Bands (MTF)
        # raw width for ATR norm
        bb_up_w, bb_mid_w, bb_low_w, bb_width_pct = compute_bollinger_bands(df['close'], period=w, num_std=2.0)
        bb_width_w = bb_up_w - bb_low_w
        features[f'bb_width_w{w}'] = _align_to_features(_norm(bb_width_w, f'bb_width_w{w}'), n_features) if use_kalman else _norm(bb_width_w, f'bb_width_w{w}')
        price_vs_bb_w = (df['close'] - bb_low_w) / (bb_up_w - bb_low_w + 1e-9)
        features[f'price_vs_bb_w{w}'] = _align_to_features(_norm(price_vs_bb_w, f'price_vs_bb_w{w}'), n_features) if use_kalman else _norm(price_vs_bb_w, f'price_vs_bb_w{w}')

        # Choppiness (MTF)
        chop_w = compute_choppiness_index(df['high'], df['low'], df['close'], period=w)
        features[f'choppiness_w{w}'] = _align_to_features(_norm(chop_w, f'choppiness_w{w}'), n_features) if use_kalman else _norm(chop_w, f'choppiness_w{w}')

        # Parkinson Volatility (MTF)
        park_vol_w = compute_parkinson_volatility(df['high'], df['low'], window=w)
        features[f'parkinson_volatility_w{w}'] = _align_to_features(_norm(park_vol_w, f'parkinson_volatility_w{w}'), n_features) if use_kalman else _norm(park_vol_w, f'parkinson_volatility_w{w}')

        # Hurst (MTF) - Only for larger windows to avoid noise
        if w >= 50:
            hurst_w = compute_hurst_proxy(df['close'], window=w)
            features[f'hurst_w{w}'] = _align_to_features(_norm(hurst_w, f'hurst_w{w}'), n_features) if use_kalman else _norm(hurst_w, f'hurst_w{w}')

        # ==============================================================================
        # NEW CORE FEATURES (MOMENTUM, VOLUME, VOLATILITY) - Top 50 Core
        # ==============================================================================

        # Momentum
        # EMA Slope
        ema_slope_w = compute_ema_slope(df['close'], window=w)
        features[f'ema_slope_w{w}'] = _align_to_features(_norm(ema_slope_w, f'ema_slope_w{w}'), n_features) if use_kalman else _norm(ema_slope_w, f'ema_slope_w{w}')

        # ROC
        roc_w = df['close'].pct_change(w)
        features[f'roc_w{w}'] = _align_to_features(_norm(roc_w, f'roc_w{w}'), n_features) if use_kalman else _norm(roc_w, f'roc_w{w}')

        # Price Acceleration (2nd derivative of price)
        # Using difference of ROC as proxy for acceleration
        accel_w = roc_w.diff()
        features[f'price_accel_w{w}'] = _align_to_features(_norm(accel_w, f'price_accel_w{w}'), n_features) if use_kalman else _norm(accel_w, f'price_accel_w{w}')

        # Donchian Channel
        donch_up, donch_low, donch_pos = compute_donchian_channel(df['high'], df['low'], df['close'], window=w)
        # raw dist for ATR norm
        features[f'donchian_breakout_dist_w{w}'] = _align_to_features(_norm((df['close'] - donch_up).abs(), f'donchian_breakout_dist_w{w}'), n_features) if use_kalman else _norm((df['close'] - donch_up).abs(), f'donchian_breakout_dist_w{w}')
        features[f'close_loc_range_w{w}'] = _align_to_features(_norm(donch_pos, f'close_loc_range_w{w}'), n_features) if use_kalman else _norm(donch_pos, f'close_loc_range_w{w}')

        # Momentum Percentile
        mom_rank_w = compute_rolling_percentile(mom_w, window=w)
        features[f'momentum_rank_w{w}'] = _align_to_features(_norm(mom_rank_w, f'momentum_rank_w{w}'), n_features) if use_kalman else _norm(mom_rank_w, f'momentum_rank_w{w}')

        # Z-Scored Returns
        z_ret_w = compute_rolling_zscore(df['close'].pct_change(), window=w)
        features[f'return_zscore_w{w}'] = _align_to_features(_norm(z_ret_w, f'return_zscore_w{w}'), n_features) if use_kalman else _norm(z_ret_w, f'return_zscore_w{w}')

        # Momentum Decay Rate (ROC / Max ROC in window)
        max_roc_w = roc_w.rolling(w).max()
        features[f'momentum_decay_w{w}'] = _align_to_features(_norm(roc_w / (max_roc_w + 1e-9), f'momentum_decay_w{w}'), n_features) if use_kalman else _norm(roc_w / (max_roc_w + 1e-9), f'momentum_decay_w{w}')

        # Volume
        if volume_available and 'volume' in df.columns:
            # Volume Z-Score -> raw volume for log1p
            features[f'volume_zscore_w{w}'] = _align_to_features(_norm(df['volume'], f'volume_zscore_w{w}'), n_features) if use_kalman else _norm(df['volume'], f'volume_zscore_w{w}')

            # Volume Delta
            vol_delta = compute_volume_delta(df['close'], df.get('open', df['close']), df['volume'])
            # Rolling sum of volume delta
            vol_delta_w = vol_delta.rolling(w).sum()
            features[f'volume_delta_w{w}'] = _align_to_features(_norm(vol_delta_w, f'volume_delta_w{w}'), n_features) if use_kalman else _norm(vol_delta_w, f'volume_delta_w{w}')

            # Volume Acceleration
            vol_accel_w = df['volume'].pct_change().rolling(w).mean()
            features[f'volume_accel_w{w}'] = _align_to_features(_norm(vol_accel_w, f'volume_accel_w{w}'), n_features) if use_kalman else _norm(vol_accel_w, f'volume_accel_w{w}')

            # OBV (on window? OBV is usually cumulative. Maybe OBV ROC?)
            obv = compute_obv(df['close'], df['volume'])
            obv_roc_w = obv.pct_change(w)
            features[f'obv_roc_w{w}'] = _align_to_features(_norm(obv_roc_w, f'obv_roc_w{w}'), n_features) if use_kalman else _norm(obv_roc_w, f'obv_roc_w{w}')

            # Volume per unit volatility
            vol_per_vol_w = df['volume'] / (vol_w * df['close'] + 1e-9)
            features[f'vol_per_vol_w{w}'] = _align_to_features(_norm(vol_per_vol_w, f'vol_per_vol_w{w}'), n_features) if use_kalman else _norm(vol_per_vol_w, f'vol_per_vol_w{w}')

        # Volatility
        # Garman-Klass
        if 'open' in df.columns:
            gk_vol_w = compute_garman_klass_volatility(df['open'], df['high'], df['low'], df['close'], window=w)
            features[f'garman_klass_vol_w{w}'] = _align_to_features(_norm(gk_vol_w, f'garman_klass_vol_w{w}'), n_features) if use_kalman else _norm(gk_vol_w, f'garman_klass_vol_w{w}')

        # Volatility Z-Score
        vol_zscore_w = compute_rolling_zscore(vol_w, window=w)
        features[f'volatility_zscore_w{w}'] = _align_to_features(_norm(vol_zscore_w, f'volatility_zscore_w{w}'), n_features) if use_kalman else _norm(vol_zscore_w, f'volatility_zscore_w{w}')

        # Range / ATR
        range_atr_w = (df['high'] - df['low']) / (atr_w + 1e-9)
        features[f'range_div_atr_w{w}'] = _align_to_features(_norm(range_atr_w, f'range_div_atr_w{w}'), n_features) if use_kalman else _norm(range_atr_w, f'range_div_atr_w{w}')

        # Volatility of Volume
        if volume_available:
            vol_of_vol_w = df['volume'].rolling(w).std() / (df['volume'].rolling(w).mean() + 1e-9)
            features[f'vol_of_vol_w{w}'] = _align_to_features(_norm(vol_of_vol_w, f'vol_of_vol_w{w}'), n_features) if use_kalman else _norm(vol_of_vol_w, f'vol_of_vol_w{w}')

        # Drawdown Depth (Current price vs Rolling Max)
        dd_w = (df['close'] / df['close'].rolling(w).max()) - 1.0
        features[f'drawdown_w{w}'] = _align_to_features(_norm(dd_w, f'drawdown_w{w}'), n_features) if use_kalman else _norm(dd_w, f'drawdown_w{w}')

        # Volatility Compression Ratio (Narrowest Range in W / Average Range)
        # Proxy: Min TR / ATR
        min_tr_w = (df['high'] - df['low']).rolling(w).min()
        features[f'vol_compression_w{w}'] = _align_to_features(_norm(min_tr_w / (atr_w + 1e-9), f'vol_compression_w{w}'), n_features) if use_kalman else _norm(min_tr_w / (atr_w + 1e-9), f'vol_compression_w{w}')

        # ==============================================================================
        # NEW INTERACTION FEATURES - Top 30
        # ==============================================================================

        if volume_available and 'volume' in df.columns:
            # RSI x Volume Z-Score
            # Recomputing locally to keep simple interaction logic, then normalize interaction.
            vol_z_w = compute_rolling_zscore(df['volume'], window=w)
            features[f'rsi_x_vol_z_w{w}'] = _norm(rsi_w * vol_z_w, f'rsi_x_vol_z_w{w}')

            # Momentum x ATR
            features[f'mom_x_atr_w{w}'] = _norm(mom_w * atr_w, f'mom_x_atr_w{w}')

            # Trend strength (ADX) x Pullback (Dist from high)
            pullback = dd_w.abs()
            features[f'adx_x_pullback_w{w}'] = _norm(adx_w * pullback, f'adx_x_pullback_w{w}')

            # Volume Delta x Candle Body Ratio
            body_size = (df['close'] - df.get('open', df['close'])).abs()
            range_size = df['high'] - df['low']
            body_ratio = body_size / (range_size + 1e-9)
            features[f'vol_delta_x_body_w{w}'] = _norm(vol_delta_w * body_ratio, f'vol_delta_x_body_w{w}')

            # VWAP Distance x Volatility
            if 'close_minus_vwap' in features.columns:
                try:
                    dollar_volume = df['close'] * df['volume']
                    cum_volume = df['volume'].cumsum()
                    vwap_series = dollar_volume.cumsum() / (cum_volume + 1e-8)
                    vwap_diff_series = df['close'] - vwap_series
                    features[f'vwap_dist_x_vol_w{w}'] = _norm(vwap_diff_series * vol_w, f'vwap_dist_x_vol_w{w}')
                except: pass

            # ADX x BB Width
            # bb_width_w local var is raw width now.
            features[f'adx_x_bbw_w{w}'] = _norm(adx_w * bb_width_w, f'adx_x_bbw_w{w}')

            # EMA Slope x Volume Surge (Vol Ratio)
            features[f'slope_x_vol_surge_w{w}'] = _norm(ema_slope_w * vol_ratio_w, f'slope_x_vol_surge_w{w}')

            # ATR x Liquidity (Volume/Range)
            liquidity_proxy = df['volume'] / (range_size + 1e-9)
            features[f'atr_x_liq_w{w}'] = _norm(atr_w * liquidity_proxy, f'atr_x_liq_w{w}')

            # Breakout Size (Return) x Prior Consolidation (Choppiness)
            features[f'ret_x_chop_w{w}'] = _norm(returns * chop_w, f'ret_x_chop_w{w}')

            # Volatility Compression (Low BBW) x Momentum Burst (High Mom)
            features[f'compress_x_burst_w{w}'] = _norm((1.0 / (bb_width_w + 1e-9)) * mom_w.abs(), f'compress_x_burst_w{w}')

            # Volume Spike x Resistance Proximity
            high_w = df['high'].rolling(w).max()
            dist_pct = (high_w - df['close']) / (high_w + 1e-9)
            res_prox = 1.0 / (1.0 + dist_pct)
            features[f'vol_spike_x_res_prox_w{w}'] = _norm(vol_ratio_w * res_prox, f'vol_spike_x_res_prox_w{w}')

            # Trend Slope x Drawdown
            features[f'slope_x_dd_w{w}'] = _norm(ema_slope_w * dd_w, f'slope_x_dd_w{w}')

            # EMA Distance x ATR
            ema_dist_val = (df['close'] - df['close'].ewm(span=w).mean()).abs()
            features[f'ema_dist_x_atr_w{w}'] = _norm(ema_dist_val * atr_w, f'ema_dist_x_atr_w{w}')

        # ==============================================================================
        # CROSS-TIMEFRAME FEATURES (Dynamic HTF) - Top 20
        # ==============================================================================
        w_htf = w * 4
        if w_htf <= 600:
            # 1. HTF EMA Slope vs LTF EMA Slope
            ema_slope_htf = compute_ema_slope(df['close'], window=w_htf)
            features[f'slope_div_w{w}_w{w_htf}'] = _align_to_features(_norm(ema_slope_w - ema_slope_htf, f'slope_div_w{w}_w{w_htf}'), n_features) if use_kalman else _norm(ema_slope_w - ema_slope_htf, f'slope_div_w{w}_w{w_htf}')

            # 2. LTF Close / HTF VWAP
            tp = (df['high'] + df['low'] + df['close']) / 3
            if volume_available:
                vwap_htf = (tp * df['volume']).rolling(w_htf).sum() / (df['volume'].rolling(w_htf).sum() + 1e-9)
            else:
                vwap_htf = tp.rolling(w_htf).mean()
            features[f'close_div_vwap_w{w}_w{w_htf}'] = _align_to_features(_norm(df['close'] / (vwap_htf + 1e-9), f'close_div_vwap_w{w}_w{w_htf}'), n_features) if use_kalman else _norm(df['close'] / (vwap_htf + 1e-9), f'close_div_vwap_w{w}_w{w_htf}')

            # 3. LTF RSI - HTF RSI
            rsi_htf = compute_rsi(df['close'], period=w_htf)
            features[f'rsi_div_w{w}_w{w_htf}'] = _align_to_features(_norm(rsi_w - rsi_htf, f'rsi_div_w{w}_w{w_htf}'), n_features) if use_kalman else _norm(rsi_w - rsi_htf, f'rsi_div_w{w}_w{w_htf}')

            # 4. LTF ATR / HTF ATR
            tr_htf = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()], axis=1).max(axis=1)
            atr_htf = tr_htf.rolling(w_htf).mean()
            features[f'atr_ratio_w{w}_w{w_htf}'] = _align_to_features(_norm(atr_w / (atr_htf + 1e-9), f'atr_ratio_w{w}_w{w_htf}'), n_features) if use_kalman else _norm(atr_w / (atr_htf + 1e-9), f'atr_ratio_w{w}_w{w_htf}')

            # 5. HTF Trend Direction * LTF Pullback
            htf_trend_dir = np.sign(ema_slope_htf)
            features[f'htf_trend_x_pullback_w{w}'] = _align_to_features(_norm(htf_trend_dir * pullback, f'htf_trend_x_pullback_w{w}'), n_features) if use_kalman else _norm(htf_trend_dir * pullback, f'htf_trend_x_pullback_w{w}')

            # 7. LTF Breakout above HTF High (Normalized by ATR)
            htf_high = df['high'].rolling(w_htf).max()
            # use raw dist for ATR norm
            dist_htf = df['close'] - htf_high
            features[f'breakout_vs_htf_w{w}'] = _align_to_features(_norm(dist_htf, f'breakout_vs_htf_w{w}'), n_features) if use_kalman else _norm(dist_htf, f'breakout_vs_htf_w{w}')

            # 8. HTF BB Width * LTF Return
            bb_up_htf, bb_mid_htf, bb_low_htf, bb_width_pct_htf = compute_bollinger_bands(df['close'], period=w_htf, num_std=2.0)
            bb_width_htf = bb_up_htf - bb_low_htf # raw width
            features[f'htf_bbw_x_ret_w{w}'] = _align_to_features(_norm(bb_width_htf * returns, f'htf_bbw_x_ret_w{w}'), n_features) if use_kalman else _norm(bb_width_htf * returns, f'htf_bbw_x_ret_w{w}')

            # 9. HTF ADX * LTF Momentum
            adx_htf, _, _ = compute_adx(df['high'], df['low'], df['close'], period=w_htf)
            features[f'htf_adx_x_mom_w{w}'] = _align_to_features(_norm(adx_htf * mom_w, f'htf_adx_x_mom_w{w}'), n_features) if use_kalman else _norm(adx_htf * mom_w, f'htf_adx_x_mom_w{w}')

            # 11. LTF Volume Spike relative to HTF Volume Mean
            if volume_available:
                vol_mean_htf = df['volume'].rolling(w_htf).mean()
                features[f'vol_spike_vs_htf_w{w}'] = _align_to_features(_norm(df['volume'] / (vol_mean_htf + 1e-9), f'vol_spike_vs_htf_w{w}'), n_features) if use_kalman else _norm(df['volume'] / (vol_mean_htf + 1e-9), f'vol_spike_vs_htf_w{w}')

            # 15. LTF Return / HTF ATR
            features[f'ret_div_htf_atr_w{w}'] = _align_to_features(_norm(returns / (atr_htf + 1e-9), f'ret_div_htf_atr_w{w}'), n_features) if use_kalman else _norm(returns / (atr_htf + 1e-9), f'ret_div_htf_atr_w{w}')

    # ===== MORE INTERACTIONS =====
    if 'volatility_1d' in features.columns and 'momentum_20' in features.columns:
        features['vol_momentum_interaction'] = _norm(features['volatility_1d'] * features['momentum_20'], 'vol_momentum_interaction')

    if 'volatility_1d' in features.columns:
        if 'momentum_10' in features.columns:
            features['momentum_10_div_volatility_1d'] = _norm(features['momentum_10'] / (features['volatility_1d'] + 1e-8), 'momentum_10_div_volatility_1d')
        if 'momentum_5' in features.columns:
            features['momentum_5_div_volatility_1d'] = _norm(features['momentum_5'] / (features['volatility_1d'] + 1e-8), 'momentum_5_div_volatility_1d')

    if 'rv_z_short' in features.columns:
        for col in ['momentum_5', 'momentum_10', 'momentum_20']:
            if col in features.columns:
                features[f'{col}_x_rv_z'] = _norm(features[col] * features['rv_z_short'], f'{col}_x_rv_z')

    if 'atr_ratio' in features.columns and 'momentum_20' in features.columns:
        features['atr_momentum'] = _norm(features['atr_ratio'] * features['momentum_20'], 'atr_momentum')

    if 'volatility_1d' in features.columns:
        # These will only exist if 50 was in 'windows'
        if 'dist_from_recent_high_50' in features.columns:
            features['high_dist_x_vol'] = _norm(features['dist_from_recent_high_50'] * features['volatility_1d'], 'high_dist_x_vol')
        if 'dist_from_recent_low_50' in features.columns:
            features['low_dist_x_vol'] = _norm(features['dist_from_recent_low_50'] * features['volatility_1d'], 'low_dist_x_vol')

    return features
