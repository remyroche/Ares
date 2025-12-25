"""
Multi-Timeframe Feature Generation Module.
Extracted and enhanced from feature_generation_meta_labeling_step.py.
"""
from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import logging

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

    # ===== VOLATILITY FEATURES (ENHANCED) =====
    log_ret = np.log(df['close']).diff()
    features['log_ret'] = log_ret

    features['volatility_1h'] = log_ret.rolling(window=4).std()
    features['volatility_4h'] = log_ret.rolling(window=16).std()
    features['volatility_1d'] = log_ret.rolling(window=96).std()
    features['vol_of_vol'] = features['volatility_1h'].rolling(window=20).std()

    # ===== VOLATILITY REGIME LABELING (Z-SCORE) =====
    vol_short_20 = log_ret.rolling(window=20).std()
    vol_long_mean = vol_short_20.rolling(window=200, min_periods=50).mean()
    vol_long_std = vol_short_20.rolling(window=200, min_periods=50).std()
    rv_z_short = (vol_short_20 - vol_long_mean) / (vol_long_std + 1e-8)
    features['rv_z_short'] = rv_z_short.fillna(0.0)
    features['vol_ratio_20_200'] = vol_short_20 / (vol_long_mean + 1e-8)
    features['volatility_20'] = vol_short_20

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
                features[col] = df[col]

    # ===== KALMAN-FILTERED TECHNICAL INDICATORS =====
    df_local = df.copy()
    df_local['rsi'] = compute_rsi(df_local['close'], period=14)
    df_local['sma_fast'] = df_local['close'].rolling(10).mean()
    df_local['sma_slow'] = df_local['close'].rolling(30).mean()
    df_local['momentum'] = df_local['close'].pct_change(10)

    # New: Add Momentum 30 (Missing from original but required)
    df_local['momentum_30'] = df_local['close'].pct_change(30)
    features['momentum_30'] = df_local['momentum_30'].to_numpy()

    if use_kalman:
        # Kalman-filtered trend
        kalman_trend, kalman_uncertainty = kalman_smooth_trend(df['close'], Q=1e-5, R=0.01)
        features['kalman_trend'] = _align_to_features(kalman_trend, n_features)
        features['kalman_uncertainty'] = _align_to_features(kalman_uncertainty, n_features)

        # Kalman-filtered RSI
        kf_rsi = KalmanFilter1D(Q=1e-4, R=0.1, initial_value=50.0)
        kalman_rsi, _ = kf_rsi.filter_series(df_local['rsi'])
        features['rsi_kalman'] = _align_to_features(kalman_rsi, n_features)

        # Kalman-filtered MA distance
        ma_distance = df_local['sma_fast'] - df_local['sma_slow']
        kf_ma = KalmanFilter1D(Q=1e-5, R=0.01, initial_value=0.0)
        kalman_ma_distance, _ = kf_ma.filter_series(ma_distance)
        features['ma_distance_kalman'] = _align_to_features(kalman_ma_distance, n_features)

        # Kalman-filtered momentum
        kf_mom = KalmanFilter1D(Q=1e-4, R=0.01, initial_value=0.0)
        kalman_momentum, _ = kf_mom.filter_series(df_local['momentum'])
        features['momentum_kalman'] = _align_to_features(kalman_momentum, n_features)
    else:
        features['rsi'] = df_local['rsi']
        features['ma_distance'] = df_local['sma_fast'] - df_local['sma_slow']
        features['momentum'] = df_local['momentum']

    # ===== VOLATILITY-NORMALIZED FEATURES =====
    vol_1h_series = features['volatility_1h'].replace(0, np.nan)
    vol_1h_arr = _align_to_features(vol_1h_series, n_features) if use_kalman else vol_1h_series.to_numpy()
    close_arr = _align_to_features(df['close'], n_features) if use_kalman else df['close'].to_numpy()

    mom_feat = features['momentum_kalman'] if use_kalman else features['momentum']
    ma_feat = features['ma_distance_kalman'] if use_kalman else features['ma_distance']

    features['momentum_per_vol'] = mom_feat / (vol_1h_arr + 1e-8)
    features['ma_distance_per_vol'] = ma_feat / (close_arr * vol_1h_arr + 1e-8)

    # ===== TRADITIONAL VOLATILITY FEATURES =====
    returns = df['close'].pct_change()
    vol5_series = returns.rolling(5).std()
    vol20_series = returns.rolling(20).std()

    if use_kalman:
        features['volatility_5'] = _align_to_features(vol5_series, n_features)
        features['volatility_20'] = _align_to_features(vol20_series, n_features)
    else:
        features['volatility_5'] = vol5_series.to_numpy()
        features['volatility_20'] = vol20_series.to_numpy()

    features['volatility_ratio'] = features['volatility_5'] / (features['volatility_20'] + 1e-8)

    vol_ema_series = (returns**2).ewm(alpha=0.1, adjust=False).mean()
    vol_ema = _align_to_features(vol_ema_series, n_features) if use_kalman else vol_ema_series.to_numpy()
    features['volatility_ema'] = np.sqrt(vol_ema)

    # ===== TREND STRENGTH =====
    sma_10 = df['close'].rolling(10).mean()
    sma_slope_series = sma_10.pct_change(5)
    sma20 = df['close'].rolling(20).mean()
    price_vs_sma20_series = (df['close'] - sma20) / (sma20 + 1e-8)

    high_low = df['high'] - df['low']
    atr_14_series = high_low.rolling(14).mean()

    if use_kalman:
        features['sma_slope'] = _align_to_features(sma_slope_series, n_features)
        features['price_vs_sma20'] = _align_to_features(price_vs_sma20_series, n_features)
        features['atr_14'] = _align_to_features(atr_14_series, n_features)
    else:
        features['sma_slope'] = sma_slope_series.to_numpy()
        features['price_vs_sma20'] = price_vs_sma20_series.to_numpy()
        features['atr_14'] = atr_14_series.to_numpy()

    features['atr_ratio'] = features['atr_14'] / (close_arr + 1e-8)

    # New: Rolling Sharpe (50) - Missing from original
    roll_mean_50 = returns.rolling(50).mean()
    roll_std_50 = returns.rolling(50).std()
    rolling_sharpe_series = (roll_mean_50 / (roll_std_50 + 1e-9)).fillna(0)
    features['rolling_sharpe'] = _align_to_features(rolling_sharpe_series, n_features) if use_kalman else rolling_sharpe_series.to_numpy()

    # ===== VOLUME CONTEXT =====
    if volume_available and 'volume' in df.columns:
        vol_sma = df['volume'].rolling(20).mean()
        volume_ratio_series = df['volume'] / (vol_sma + 1e-8)
        volume_trend_series = df['volume'].rolling(5).mean() / (vol_sma + 1e-8)
        vol_price_corr_series = returns.rolling(20).corr(df['volume'].pct_change())

        vol_mean = df['volume'].rolling(96).mean()
        vol_std = df['volume'].rolling(96).std()
        volume_zscore_series = (df['volume'] - vol_mean) / (vol_std + 1e-8)

        volume_spike_series = df['volume'] / (df['volume'].rolling(96).mean() + 1e-8)

        signed_volume_raw = np.sign(returns.fillna(0.0).to_numpy()) * df['volume'].to_numpy()
        signed_volume_ema_series = pd.Series(signed_volume_raw).ewm(span=20).mean()

        if use_kalman:
            features['volume_ratio'] = _align_to_features(volume_ratio_series, n_features)
            features['volume_trend'] = _align_to_features(volume_trend_series, n_features)
            features['vol_price_corr'] = _align_to_features(vol_price_corr_series, n_features)
            features['volume_zscore'] = _align_to_features(volume_zscore_series, n_features)
            features['volume_spike'] = _align_to_features(volume_spike_series, n_features)
            features['signed_volume_ema'] = _align_to_features(signed_volume_ema_series, n_features)
        else:
            features['volume_ratio'] = volume_ratio_series.to_numpy()
            features['volume_trend'] = volume_trend_series.to_numpy()
            features['vol_price_corr'] = vol_price_corr_series.to_numpy()
            features['volume_zscore'] = volume_zscore_series.to_numpy()
            features['volume_spike'] = volume_spike_series.to_numpy()
            features['signed_volume_ema'] = signed_volume_ema_series.to_numpy()
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
        features['momentum_5'] = _align_to_features(mom5_series, n_features)
        features['momentum_10'] = _align_to_features(mom10_series, n_features)
        features['momentum_20'] = _align_to_features(mom20_series, n_features)
        features['momentum_ema'] = _align_to_features(mom10_series.ewm(span=5).mean(), n_features)
    else:
        features['momentum_5'] = mom5_series.to_numpy()
        features['momentum_10'] = mom10_series.to_numpy()
        features['momentum_20'] = mom20_series.to_numpy()
        features['momentum_ema'] = mom10_series.ewm(span=5).mean().to_numpy()

    # ACF
    autocorr_series = returns.rolling(window=50, min_periods=10).corr(returns.shift(1))
    features['return_autocorr_lag1_w50'] = _align_to_features(autocorr_series, n_features) if use_kalman else autocorr_series.to_numpy()

    # ===== RANGE POSITION =====
    recent_high = df['high'].rolling(20).max()
    recent_low = df['low'].rolling(20).min()
    range_position_series = (df['close'] - recent_low) / (recent_high - recent_low + 1e-8)
    features['range_position'] = _align_to_features(range_position_series, n_features) if use_kalman else range_position_series.to_numpy()

    if 'close' in df.columns and 'volume' in df.columns:
        try:
            dollar_volume = df['close'] * df['volume']
            cum_volume = df['volume'].cumsum()
            vwap_series = dollar_volume.cumsum() / (cum_volume + 1e-8)
            vwap_diff_series = df['close'] - vwap_series
            features['close_minus_vwap'] = _align_to_features(vwap_diff_series, n_features) if use_kalman else vwap_diff_series.to_numpy()
        except: pass

    # ===== ENTROPY =====
    returns_abs = returns.abs().rolling(20).mean()
    returns_entropy_series = -returns_abs * np.log(returns_abs + 1e-8)
    features['returns_entropy'] = _align_to_features(returns_entropy_series, n_features) if use_kalman else returns_entropy_series.to_numpy()

    # Hurst Proxy
    features['hurst_100'] = _align_to_features(compute_hurst_proxy(df['close'], 100), n_features) if use_kalman else compute_hurst_proxy(df['close'], 100).to_numpy()

    # ===== ADVANCED MOMENTUM =====
    # Stochastic
    stoch_k, stoch_d = compute_stochastic(df['high'], df['low'], df['close'])
    features['stoch_k'] = _align_to_features(stoch_k, n_features) if use_kalman else stoch_k.to_numpy()
    features['stoch_d'] = _align_to_features(stoch_d, n_features) if use_kalman else stoch_d.to_numpy()

    # CCI
    features['cci_14'] = _align_to_features(compute_cci(df['high'], df['low'], df['close'], 14), n_features) if use_kalman else compute_cci(df['high'], df['low'], df['close'], 14).to_numpy()
    features['cci_40'] = _align_to_features(compute_cci(df['high'], df['low'], df['close'], 40), n_features) if use_kalman else compute_cci(df['high'], df['low'], df['close'], 40).to_numpy()

    # ADX
    adx, plus_di, minus_di = compute_adx(df['high'], df['low'], df['close'], 14)
    features['adx_14'] = _align_to_features(adx, n_features) if use_kalman else adx.to_numpy()
    features['plus_di_14'] = _align_to_features(plus_di, n_features) if use_kalman else plus_di.to_numpy()
    features['minus_di_14'] = _align_to_features(minus_di, n_features) if use_kalman else minus_di.to_numpy()
    features['adx_trend'] = features['plus_di_14'] - features['minus_di_14'] # Simple trend strength

    # ===== ADVANCED VOLATILITY =====
    # Bollinger Bands
    bb_upper, bb_mid, bb_lower, bb_width = compute_bollinger_bands(df['close'], 20, 2.0)
    features['bb_width'] = _align_to_features(bb_width, n_features) if use_kalman else bb_width.to_numpy()
    features['price_vs_bb'] = _align_to_features((df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-9), n_features) if use_kalman else ((df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-9)).to_numpy()

    # Choppiness
    features['choppiness_14'] = _align_to_features(compute_choppiness_index(df['high'], df['low'], df['close'], 14), n_features) if use_kalman else compute_choppiness_index(df['high'], df['low'], df['close'], 14).to_numpy()

    # Parkinson Volatility
    features['parkinson_volatility_20'] = _align_to_features(compute_parkinson_volatility(df['high'], df['low'], 20), n_features) if use_kalman else compute_parkinson_volatility(df['high'], df['low'], 20).to_numpy()

    # ===== ADVANCED VOLUME =====
    if volume_available and 'volume' in df.columns:
        features['cmf_20'] = _align_to_features(compute_cmf(df['high'], df['low'], df['close'], df['volume'], 20), n_features) if use_kalman else compute_cmf(df['high'], df['low'], df['close'], df['volume'], 20).to_numpy()
        features['force_index_13'] = _align_to_features(compute_force_index(df['close'], df['volume'], 13), n_features) if use_kalman else compute_force_index(df['close'], df['volume'], 13).to_numpy()
        # Normalized Force Index (by volume MA)
        fi_norm = features['force_index_13'] / (df['volume'].rolling(20).mean() * df['close'] + 1e-9)
        features['force_index_norm'] = fi_norm

    # ===== INTERACTION FEATURES =====
    # Volatility * Momentum
    if 'volatility_20' in features.columns and 'momentum_20' in features.columns:
        features['vol_x_mom'] = features['volatility_20'] * features['momentum_20']

    # ADX * Trend
    if 'adx_14' in features.columns and 'momentum_10' in features.columns:
        features['adx_x_mom'] = (features['adx_14'] / 100.0) * features['momentum_10']

    # BB Squeeze * Volume
    if 'bb_width' in features.columns and 'volume_ratio' in features.columns:
        features['squeeze_x_vol'] = (1.0 / (features['bb_width'] + 1e-9)) * features['volume_ratio']

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
        features['cvd_proxy'] = _align_to_features(cvd_normalized, n_features) if use_kalman else cvd_normalized.to_numpy()

        close_in_range = (close - low) / (high - low + 1e-8)
        volume_pressure = (close_in_range - 0.5) * volume
        volume_pressure_ema = volume_pressure.ewm(span=20).mean()
        features['volume_pressure'] = _align_to_features(volume_pressure_ema, n_features) if use_kalman else volume_pressure_ema.to_numpy()

        upper_wick = high - pd.concat([open_price, close], axis=1).max(axis=1)
        lower_wick = pd.concat([open_price, close], axis=1).min(axis=1) - low
        total_range = high - low + 1e-8
        supply_rejection = (upper_wick / total_range) * volume
        demand_rejection = (lower_wick / total_range) * volume
        ofi_proxy = (demand_rejection - supply_rejection).rolling(20).sum()
        ofi_normalized = ofi_proxy / (ofi_proxy.rolling(96).std() + 1e-8)
        features['ofi_proxy'] = _align_to_features(ofi_normalized, n_features) if use_kalman else ofi_normalized.to_numpy()

        buy_volume = volume * close_in_range
        sell_volume = volume * (1 - close_in_range)
        volume_imbalance = (buy_volume - sell_volume) / (volume + 1e-8)
        volume_imbalance_ema = volume_imbalance.ewm(span=20).mean()
        features['volume_imbalance'] = _align_to_features(volume_imbalance_ema, n_features) if use_kalman else volume_imbalance_ema.to_numpy()

        is_at_extreme = (close_in_range < 0.2) | (close_in_range > 0.8)
        extreme_volume = volume.where(is_at_extreme, 0).rolling(20).sum()
        total_volume = volume.rolling(20).sum()
        absorption_ratio = extreme_volume / (total_volume + 1e-8)
        features['absorption_ratio'] = _align_to_features(absorption_ratio, n_features) if use_kalman else absorption_ratio.to_numpy()

        trade_aggressor_ratio_series = close_in_range.ewm(span=20).mean()
        features['trade_aggressor_ratio'] = _align_to_features(trade_aggressor_ratio_series, n_features) if use_kalman else trade_aggressor_ratio_series.to_numpy()

        prev_close = close.shift(1)
        gap_raw = open_price - prev_close
        features['liquidity_gap_up'] = _align_to_features(np.maximum(gap_raw, 0) / (prev_close + 1e-8), n_features) if use_kalman else (np.maximum(gap_raw, 0) / (prev_close + 1e-8)).to_numpy()
        features['liquidity_gap_down'] = _align_to_features(np.maximum(-gap_raw, 0) / (prev_close + 1e-8), n_features) if use_kalman else (np.maximum(-gap_raw, 0) / (prev_close + 1e-8)).to_numpy()
        features['liquidity_gap_abs'] = _align_to_features(gap_raw.abs() / (features['atr_14'] + 1e-8), n_features) if use_kalman else (gap_raw.abs() / (features['atr_14'] + 1e-8)).to_numpy()
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
        features['signal_strength_all'] = _align_to_features(abs_signals.sum(axis=1), n_features) if use_kalman else abs_signals.sum(axis=1).to_numpy()
        features['signal_count_active'] = _align_to_features((abs_signals > 0).sum(axis=1), n_features) if use_kalman else (abs_signals > 0).sum(axis=1).to_numpy()

        if 'rsi' in signals.columns and 'macd' in signals.columns:
            align_series = np.sign(signals['rsi'] * signals['macd']).replace(0, 0)
            features['signal_rsi_macd_alignment'] = _align_to_features(align_series, n_features) if use_kalman else align_series.to_numpy()

    # Pass through signal component values
    for col, new_col in [('rsi_value', 'signal_rsi_distance_50'), ('rsi_long_value', 'signal_rsi_long_distance_50')]:
        if col in signals.columns:
            dist = (signals[col] - 50.0).abs()
            features[new_col] = _align_to_features(dist, n_features) if use_kalman else dist.to_numpy()

    if 'macd_hist_value' in signals.columns:
        features['signal_macd_hist_abs'] = _align_to_features(signals['macd_hist_value'].abs(), n_features) if use_kalman else signals['macd_hist_value'].abs().to_numpy()
    if 'macd_hist_long_value' in signals.columns:
        features['signal_macd_hist_long_abs'] = _align_to_features(signals['macd_hist_long_value'].abs(), n_features) if use_kalman else signals['macd_hist_long_value'].abs().to_numpy()

    if 'sma_fast_value' in signals.columns and 'sma_slow_value' in signals.columns:
        ma_dist = (signals['sma_fast_value'] - signals['sma_slow_value']) / (df['close'] + 1e-8)
        features['signal_ma_distance_raw'] = _align_to_features(ma_dist, n_features) if use_kalman else ma_dist.to_numpy()

    if 'momentum_value' in signals.columns:
        features['signal_momentum_value'] = _align_to_features(signals['momentum_value'], n_features) if use_kalman else signals['momentum_value'].to_numpy()

    # Pass through regime flags
    for col in ['trend_regime', 'candle_trend', 'candle_reversal']:
        if col in signals.columns:
            features[col] = _align_to_features(signals[col], n_features) if use_kalman else signals[col].to_numpy()

    # Interactions
    if 'trend_regime' in features.columns and 'signal_macd_hist_abs' in features.columns:
        features['signal_trend_regime_x_macd_hist_abs'] = features['trend_regime'] * features['signal_macd_hist_abs']
    if 'candle_trend' in features.columns and 'signal_rsi_distance_50' in features.columns:
        features['signal_candle_trend_x_rsi_distance_50'] = features['candle_trend'] * features['signal_rsi_distance_50']

    # ===== CROSS-TIMEFRAME FEATURES =====
    close_1h = df['close'].rolling(4).mean()
    features['returns_1h'] = _align_to_features(close_1h.pct_change(), n_features) if use_kalman else close_1h.pct_change().to_numpy()
    features['momentum_1h'] = _align_to_features(df['close'].pct_change(4), n_features) if use_kalman else df['close'].pct_change(4).to_numpy()
    features['volatility_1h_agg'] = _align_to_features(close_1h.pct_change().rolling(16).std(), n_features) if use_kalman else close_1h.pct_change().rolling(16).std().to_numpy()

    # 1H RSI Proxy
    # Removed specific 1H RSI to rely on MTF loop

    # 1H BB Proxy
    # Removed specific 1H BB Width to rely on MTF loop

    close_4h = df['close'].rolling(16).mean()
    features['returns_4h'] = _align_to_features(close_4h.pct_change(), n_features) if use_kalman else close_4h.pct_change().to_numpy()
    features['momentum_4h'] = _align_to_features(df['close'].pct_change(16), n_features) if use_kalman else df['close'].pct_change(16).to_numpy()
    features['volatility_4h_agg'] = _align_to_features(close_4h.pct_change().rolling(16).std(), n_features) if use_kalman else close_4h.pct_change().rolling(16).std().to_numpy()

    # 4H RSI Proxy
    rsi_4h = compute_rsi(close_4h, 14)
    features['rsi_4h'] = _align_to_features(rsi_4h, n_features) if use_kalman else rsi_4h.to_numpy()

    # ===== ROLLING WINDOW FEATURES (FOR TREES) =====
    close_arr_full = _align_to_features(df['close'], n_features) if use_kalman else df['close'].to_numpy()

    for window in [5, 10, 20, 50]:
        features[f'returns_mean_{window}'] = _align_to_features(returns.rolling(window).mean(), n_features) if use_kalman else returns.rolling(window).mean().to_numpy()
        features[f'returns_std_{window}'] = _align_to_features(returns.rolling(window).std(), n_features) if use_kalman else returns.rolling(window).std().to_numpy()

        close_min = df['close'].rolling(window).min()
        close_max = df['close'].rolling(window).max()
        features[f'close_min_{window}'] = _align_to_features(close_min, n_features) if use_kalman else close_min.to_numpy()
        features[f'close_max_{window}'] = _align_to_features(close_max, n_features) if use_kalman else close_max.to_numpy()

        close_range = (close_max - close_min) / (df['close'] + 1e-8)
        features[f'close_range_{window}'] = _align_to_features(close_range, n_features) if use_kalman else close_range.to_numpy()

        dist_high = (df['close'] - close_max) / (df['close'] + 1e-8)
        dist_low = (df['close'] - close_min) / (df['close'] + 1e-8)
        features[f'dist_from_recent_high_{window}'] = _align_to_features(dist_high, n_features) if use_kalman else dist_high.to_numpy()
        features[f'dist_from_recent_low_{window}'] = _align_to_features(dist_low, n_features) if use_kalman else dist_low.to_numpy()

    # Kaufman ER
    features['kaufman_efficiency_ratio'] = _align_to_features(get_efficiency_ratio(df['close'], 14), n_features) if use_kalman else get_efficiency_ratio(df['close'], 14).to_numpy()

    # ACF Mean
    acf_vals = []
    for lag in [1, 2, 5]:
        acf_vals.append(log_ret.rolling(20).corr(log_ret.shift(lag)))
    features['acf_mean_lags_1_2_5'] = _align_to_features(pd.concat(acf_vals, axis=1).mean(axis=1), n_features) if use_kalman else pd.concat(acf_vals, axis=1).mean(axis=1).to_numpy()

    # More Interactions
    if 'volatility_1d' in features.columns and 'momentum_20' in features.columns:
        features['vol_momentum_interaction'] = features['volatility_1d'] * features['momentum_20']

    if 'volatility_1d' in features.columns:
        if 'momentum_10' in features.columns:
            features['momentum_10_div_volatility_1d'] = features['momentum_10'] / (features['volatility_1d'] + 1e-8)
        if 'momentum_5' in features.columns:
            features['momentum_5_div_volatility_1d'] = features['momentum_5'] / (features['volatility_1d'] + 1e-8)

    if 'rv_z_short' in features.columns:
        for col in ['momentum_5', 'momentum_10', 'momentum_20']:
            if col in features.columns:
                features[f'{col}_x_rv_z'] = features[col] * features['rv_z_short']

    if 'atr_ratio' in features.columns and 'momentum_20' in features.columns:
        features['atr_momentum'] = features['atr_ratio'] * features['momentum_20']

    if 'volatility_1d' in features.columns:
        if 'dist_from_recent_high_50' in features.columns:
            features['high_dist_x_vol'] = features['dist_from_recent_high_50'] * features['volatility_1d']
        if 'dist_from_recent_low_50' in features.columns:
            features['low_dist_x_vol'] = features['dist_from_recent_low_50'] * features['volatility_1d']

    if include_raw_signals:
        features['signal_strength'] = signals[['rsi', 'ma', 'mom']].abs().sum(axis=1)
        features['signal_consensus'] = signals['consensus'].abs()

    # ===== MTF LOOP (Enhanced) =====
    # Generate key features on multiple timeframes
    for w in windows:
        if w == 20: continue # Already have many 20-period features

        # Volatility
        vol_w = log_ret.rolling(w).std()
        features[f'volatility_w{w}'] = _align_to_features(vol_w, n_features) if use_kalman else vol_w.to_numpy()

        # Momentum
        mom_w = df['close'].pct_change(w)
        if use_kalman:
            kf = KalmanFilter1D(Q=1e-4, R=0.01)
            mom_w_filt, _ = kf.filter_series(mom_w)
            features[f'momentum_w{w}'] = _align_to_features(mom_w_filt, n_features)
        else:
            features[f'momentum_w{w}'] = mom_w.to_numpy()

        # RSI
        rsi_w = compute_rsi(df['close'], period=w)
        features[f'rsi_w{w}'] = _align_to_features(rsi_w, n_features) if use_kalman else rsi_w.to_numpy()

        # Efficiency
        eff_w = get_efficiency_ratio(df['close'], window=w)
        features[f'kaufman_efficiency_ratio_w{w}'] = _align_to_features(eff_w, n_features) if use_kalman else eff_w.to_numpy()

        # Range Position
        high_w = df['high'].rolling(w).max()
        low_w = df['low'].rolling(w).min()
        rng_pos_w = (df['close'] - low_w) / (high_w - low_w + 1e-8)
        features[f'range_position_w{w}'] = _align_to_features(rng_pos_w, n_features) if use_kalman else rng_pos_w.to_numpy()

        # Autocorr
        ac_w = returns.rolling(window=w, min_periods=max(5, w//4)).corr(returns.shift(1))
        features[f'autocorr_w{w}'] = _align_to_features(ac_w, n_features) if use_kalman else ac_w.to_numpy()

        # SMA Slope (MTF)
        # Use w for SMA window, and w//2 for slope lookback
        sma_w = df['close'].rolling(w).mean()
        slope_w = sma_w.pct_change(max(1, w//2))
        features[f'sma_slope_w{w}'] = _align_to_features(slope_w, n_features) if use_kalman else slope_w.to_numpy()

        # Price vs SMA (MTF)
        price_vs_sma_w = (df['close'] - sma_w) / (sma_w + 1e-8)
        features[f'price_vs_sma_w{w}'] = _align_to_features(price_vs_sma_w, n_features) if use_kalman else price_vs_sma_w.to_numpy()

        # ATR Ratio (MTF)
        high_low = df['high'] - df['low']
        atr_w = high_low.rolling(w).mean()
        atr_ratio_w = atr_w / (df['close'] + 1e-8)
        features[f'atr_ratio_w{w}'] = _align_to_features(atr_ratio_w, n_features) if use_kalman else atr_ratio_w.to_numpy()

        # Rolling Sharpe (MTF)
        # We need a longer window for Sharpe usually, so let's stick to w if w >= 20, else use max(20, w*2)
        # Or just use w for consistency, accepting it might be noisy for small w
        sharpe_mean = returns.rolling(w).mean()
        sharpe_std = returns.rolling(w).std()
        sharpe_w = (sharpe_mean / (sharpe_std + 1e-9)).fillna(0)
        features[f'rolling_sharpe_w{w}'] = _align_to_features(sharpe_w, n_features) if use_kalman else sharpe_w.to_numpy()

        # Volume Metrics (MTF)
        if volume_available and 'volume' in df.columns:
            vol_sma_w = df['volume'].rolling(w).mean()
            vol_ratio_w = df['volume'] / (vol_sma_w + 1e-8)
            features[f'volume_ratio_w{w}'] = _align_to_features(vol_ratio_w, n_features) if use_kalman else vol_ratio_w.to_numpy()

            # Volume Trend (Short SMA / Long SMA) - using w as Long, w/4 as Short
            vol_short_sma = df['volume'].rolling(max(1, w//4)).mean()
            vol_trend_w = vol_short_sma / (vol_sma_w + 1e-8)
            features[f'volume_trend_w{w}'] = _align_to_features(vol_trend_w, n_features) if use_kalman else vol_trend_w.to_numpy()

            # CMF (MTF)
            cmf_w = compute_cmf(df['high'], df['low'], df['close'], df['volume'], w)
            features[f'cmf_w{w}'] = _align_to_features(cmf_w, n_features) if use_kalman else cmf_w.to_numpy()

            # Force Index (MTF)
            fi_w = compute_force_index(df['close'], df['volume'], w)
            # Normalize Force Index
            fi_norm_w = fi_w / (vol_sma_w * df['close'] + 1e-9)
            features[f'force_index_w{w}'] = _align_to_features(fi_norm_w, n_features) if use_kalman else fi_norm_w.to_numpy()

        # Stochastic (MTF)
        stoch_k_w, stoch_d_w = compute_stochastic(df['high'], df['low'], df['close'], k_period=w, d_period=max(3, w // 5))
        features[f'stoch_k_w{w}'] = _align_to_features(stoch_k_w, n_features) if use_kalman else stoch_k_w.to_numpy()
        features[f'stoch_d_w{w}'] = _align_to_features(stoch_d_w, n_features) if use_kalman else stoch_d_w.to_numpy()

        # CCI (MTF)
        cci_w = compute_cci(df['high'], df['low'], df['close'], period=w)
        features[f'cci_w{w}'] = _align_to_features(cci_w, n_features) if use_kalman else cci_w.to_numpy()

        # ADX (MTF)
        adx_w, plus_di_w, minus_di_w = compute_adx(df['high'], df['low'], df['close'], period=w)
        features[f'adx_w{w}'] = _align_to_features(adx_w, n_features) if use_kalman else adx_w.to_numpy()
        features[f'adx_trend_w{w}'] = _align_to_features(plus_di_w - minus_di_w, n_features) if use_kalman else (plus_di_w - minus_di_w).to_numpy()

        # Bollinger Bands (MTF)
        bb_up_w, bb_mid_w, bb_low_w, bb_width_w = compute_bollinger_bands(df['close'], period=w, num_std=2.0)
        features[f'bb_width_w{w}'] = _align_to_features(bb_width_w, n_features) if use_kalman else bb_width_w.to_numpy()
        price_vs_bb_w = (df['close'] - bb_low_w) / (bb_up_w - bb_low_w + 1e-9)
        features[f'price_vs_bb_w{w}'] = _align_to_features(price_vs_bb_w, n_features) if use_kalman else price_vs_bb_w.to_numpy()

        # Choppiness (MTF)
        chop_w = compute_choppiness_index(df['high'], df['low'], df['close'], period=w)
        features[f'choppiness_w{w}'] = _align_to_features(chop_w, n_features) if use_kalman else chop_w.to_numpy()

        # Parkinson Volatility (MTF)
        park_vol_w = compute_parkinson_volatility(df['high'], df['low'], window=w)
        features[f'parkinson_volatility_w{w}'] = _align_to_features(park_vol_w, n_features) if use_kalman else park_vol_w.to_numpy()

        # Hurst (MTF) - Only for larger windows to avoid noise
        if w >= 50:
            hurst_w = compute_hurst_proxy(df['close'], window=w)
            features[f'hurst_w{w}'] = _align_to_features(hurst_w, n_features) if use_kalman else hurst_w.to_numpy()

    return features
