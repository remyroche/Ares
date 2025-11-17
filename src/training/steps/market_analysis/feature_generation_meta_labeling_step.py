"""
Feature Generation Meta-Labeling Step (Production Version - Enhanced).

Major enhancements:
1. Ensemble models (LGBM + LogisticRegression + RF) with soft voting
2. K-fold cross-fitting to prevent leakage (purged time-series CV)
3. Volatility-adaptive labeling with Kalman filtering
4. Robust feature engineering with RobustScaler
5. Vectorized operations for performance
6. Comprehensive diagnostics and calibration (Platt + Isotonic)
7. Production TPSL parameters (1% profit, 0.5% stop, 0.15% fee)

NEW FEATURES (v2):
8. Enhanced volatility-normalized features (MACD/ATR, slope/delta normalized by vol)
9. Short-term autocorrelation and momentum metrics (3-bar autocorrelation, 2-4 bar momentum)
10. Signal uncertainty features (model agreement count, probability range, entropy)
11. Feature interactions (RSI slope × MACD, price distance × volatility-adjusted RSI)
12. Signal persistence features aligned to 2-4 bars (direction persistence)
13. Feature selection (low variance removal <1e-5, correlation pruning >0.98)
14. Enhanced metrics tracking (Precision@K, Brier score, ECE, hit rates)

Based on guidance from "Advances in Financial Machine Learning" by Marcos López de Prado.
"""

import asyncio
import logging
import json
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import gc
import warnings

# ML/Stats libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, log_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb

# Vectorized computation (if available)
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    warnings.warn("vectorbt not available - using slower triple barrier implementation")

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error

logger = logging.getLogger(__name__)

# Production TPSL Parameters (overridable via config)
DEFAULT_PROFIT_THRESHOLD = 0.01  # 1%
DEFAULT_STOP_THRESHOLD = 0.005   # 0.5%
DEFAULT_TRANSACTION_COST = 0.0015  # 0.15% per trade


def purge_training_idxs(
    train_idxs: np.ndarray,
    val_start_idx: int,
    val_end_idx: int,
    horizon: int
) -> np.ndarray:
    """
    Remove training indices that would create lookahead bias.

    CRITICAL: A training sample at position i uses data up to i and predicts i+horizon.
    We must remove training samples where the prediction horizon reaches into validation.

    Args:
        train_idxs: Array of training indices (positions in DataFrame)
        val_start_idx: Start of validation period (inclusive)
        val_end_idx: End of validation period (exclusive)
        horizon: Number of periods the label looks ahead

    Returns:
        Filtered training indices without lookahead bias
    """
    filtered = []
    for i in train_idxs:
        # Drop if prediction horizon reaches into validation
        if (i + horizon) >= val_start_idx and i < val_end_idx:
            continue
        # Drop if entry is in validation window
        if (i >= val_start_idx) and (i < val_end_idx):
            continue
        # Drop if lookahead overlaps validation start
        if (i + horizon) >= val_start_idx and i < val_start_idx:
            continue
        filtered.append(i)

    return np.array(filtered, dtype=int)


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


def kalman_smooth_labels(
    binary_labels: pd.Series,
    Q: float = 1e-4,
    R: float = 0.01,
    volatility: Optional[pd.Series] = None
) -> Tuple[pd.Series, pd.Series]:
    """
    Smooth binary meta-labels using Kalman filter to get continuous probability.

    This treats observed 0/1 outcomes as noisy observations of latent signal quality.

    Args:
        binary_labels: Raw binary labels (0/1)
        Q: Process variance (how fast quality can change)
        R: Observation variance (noise in labels)
        volatility: Optional volatility series for adaptive Q

    Returns:
        Tuple of (smoothed_labels, state_variance)
    """
    n = len(binary_labels)
    smoothed = np.full(n, np.nan)
    variances = np.full(n, np.nan)

    # Initialize
    x = 0.5  # Start at neutral
    P = 1.0

    for i in range(n):
        if pd.isna(binary_labels.iloc[i]):
            # No observation, just predict forward
            x = x  # State stays same
            P = P + Q
            smoothed[i] = x
            variances[i] = P
        else:
            # Adaptive process noise based on volatility
            Q_adaptive = Q
            if volatility is not None and not pd.isna(volatility.iloc[i]):
                # Increase Q in high volatility (labels adapt faster)
                vol_factor = volatility.iloc[i] / (volatility.mean() + 1e-8)
                Q_adaptive = Q * (1 + vol_factor)

            # Predict
            x_prior = x
            P_prior = P + Q_adaptive

            # Update
            y = binary_labels.iloc[i]
            K = P_prior / (P_prior + R)
            x = x_prior + K * (y - x_prior)
            P = (1 - K) * P_prior

            smoothed[i] = x
            variances[i] = P

    return (
        pd.Series(smoothed, index=binary_labels.index),
        pd.Series(variances, index=binary_labels.index)
    )


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI (Relative Strength Index)."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
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


def compute_autocorrelation(series: pd.Series, lag: int = 1) -> pd.Series:
    """
    Compute rolling autocorrelation at specified lag.

    Args:
        series: Time series data
        lag: Lag for autocorrelation

    Returns:
        Rolling autocorrelation series
    """
    # Use rolling correlation with shifted version of itself
    return series.rolling(window=20).apply(
        lambda x: x[:-lag].corr(x[lag:]) if len(x) > lag else np.nan,
        raw=False
    )


def generate_primary_signals(
    df: pd.DataFrame,
    rsi_period: int = 14,
    rsi_period_long: int = 56,  # 4x longer for multi-timeframe
    sma_fast: int = 10,
    sma_slow: int = 30,
    momentum_period: int = 10,
    rsi_oversold: float = 25.0,  # LOOSER (was 30)
    rsi_overbought: float = 75.0,  # LOOSER (was 70)
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    macd_fast_long: int = 48,  # 4x longer
    macd_slow_long: int = 104,  # 4x longer
    macd_signal_long: int = 36,  # 4x longer
    macd_threshold: float = 0.02,  # LOOSER difference threshold
    momentum_threshold: float = 0.005
) -> pd.DataFrame:
    """
    Generate primary trading signals from technical indicators with LOOSER thresholds.

    ENHANCED: Uses looser thresholds to generate more candidate signals for meta-model filtering.
    Includes long-term indicators (4x periods) for multi-timeframe analysis.

    CRITICAL: These signals are FIXED and must never be re-optimized during CV.
    They define the "primary model" whose signals we will meta-label.

    Returns:
        DataFrame with signal columns including raw indicator values for meta-features
    """
    signals = pd.DataFrame(index=df.index)
    df_local = df.copy()

    # ===== RSI SIGNALS (short + long term) =====
    df_local['rsi'] = compute_rsi(df_local['close'], period=rsi_period)
    df_local['rsi_long'] = compute_rsi(df_local['close'], period=rsi_period_long)

    # Short-term RSI signals (LOOSER thresholds: 25/75)
    signals['rsi'] = 0
    signals.loc[df_local['rsi'] < rsi_oversold, 'rsi'] = 1
    signals.loc[df_local['rsi'] > rsi_overbought, 'rsi'] = -1

    # Long-term RSI signals (for trend confirmation)
    signals['rsi_long'] = 0
    signals.loc[df_local['rsi_long'] < rsi_oversold, 'rsi_long'] = 1
    signals.loc[df_local['rsi_long'] > rsi_overbought, 'rsi_long'] = -1

    # ===== MACD SIGNALS (short + long term) =====
    macd, macd_signal_line, macd_hist = compute_macd(
        df_local['close'], macd_fast, macd_slow, macd_signal
    )
    macd_long, macd_signal_long_line, macd_hist_long = compute_macd(
        df_local['close'], macd_fast_long, macd_slow_long, macd_signal_long
    )

    df_local['macd'] = macd
    df_local['macd_signal'] = macd_signal_line
    df_local['macd_hist'] = macd_hist
    df_local['macd_hist_long'] = macd_hist_long

    # MACD signals (based on histogram and threshold)
    signals['macd'] = 0
    # Bullish: histogram positive AND difference > threshold
    signals.loc[(macd_hist > 0) & (macd_hist > macd_threshold), 'macd'] = 1
    # Bearish: histogram negative AND difference < -threshold
    signals.loc[(macd_hist < 0) & (macd_hist < -macd_threshold), 'macd'] = -1

    # Long-term MACD (for trend confirmation)
    signals['macd_long'] = 0
    signals.loc[(macd_hist_long > 0) & (macd_hist_long > macd_threshold), 'macd_long'] = 1
    signals.loc[(macd_hist_long < 0) & (macd_hist_long < -macd_threshold), 'macd_long'] = -1

    # ===== MOVING AVERAGE CROSSOVER =====
    df_local['sma_fast'] = df_local['close'].rolling(sma_fast).mean()
    df_local['sma_slow'] = df_local['close'].rolling(sma_slow).mean()
    signals['ma'] = 0
    signals.loc[df_local['sma_fast'] > df_local['sma_slow'], 'ma'] = 1
    signals.loc[df_local['sma_fast'] < df_local['sma_slow'], 'ma'] = -1

    # ===== MOMENTUM SIGNALS =====
    df_local['momentum'] = df_local['close'].pct_change(momentum_period)
    signals['mom'] = 0
    signals.loc[df_local['momentum'] > momentum_threshold, 'mom'] = 1
    signals.loc[df_local['momentum'] < -momentum_threshold, 'mom'] = -1

    # ===== CONSENSUS SIGNAL =====
    # Use all signals for consensus (including long-term for multi-timeframe agreement)
    signal_cols = ['rsi', 'rsi_long', 'macd', 'macd_long', 'ma', 'mom']
    signals['consensus'] = signals[signal_cols].sum(axis=1).apply(np.sign)

    # Store raw indicator values for meta-features (signal disagreement, magnitude, etc.)
    signals['rsi_value'] = df_local['rsi']
    signals['rsi_long_value'] = df_local['rsi_long']
    signals['macd_hist_value'] = df_local['macd_hist']
    signals['macd_hist_long_value'] = df_local['macd_hist_long']
    signals['sma_fast_value'] = df_local['sma_fast']
    signals['sma_slow_value'] = df_local['sma_slow']
    signals['momentum_value'] = df_local['momentum']

    return signals


def compute_realized_returns(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    profit_threshold: Union[float, pd.Series] = 0.015,
    stop_threshold: Union[float, pd.Series] = 0.010,
    horizon: int = 16,
    transaction_cost: float = 0.0005,
    min_event_spacing: int = 4
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute realized returns for each signal event.

    IMPROVED: Returns continuous values (realized return) instead of binary labels.
    This allows isotonic regression to map probabilities to expected returns.

    ENHANCED: Supports adaptive thresholds based on volatility.

    Args:
        df: DataFrame with price data
        signals: DataFrame with signal columns
        profit_threshold: Profit target as fraction (float or Series for adaptive)
        stop_threshold: Stop loss as fraction (float or Series for adaptive)
        horizon: Maximum bars to look ahead
        transaction_cost: Transaction cost per trade (round trip)
        min_event_spacing: Minimum bars between signals (prevents overlapping events)

    Returns:
        Tuple of (realized_returns, binary_labels)
        - realized_returns: Actual returns achieved (NaN where no signal)
        - binary_labels: Binary success/failure (for model training)
    """
    realized_returns = pd.Series(index=df.index, dtype=float)
    realized_returns[:] = np.nan

    binary_labels = pd.Series(index=df.index, dtype=float)
    binary_labels[:] = np.nan

    close_prices = df['close'].values
    consensus_signals = signals['consensus'].values

    # Convert thresholds to arrays for adaptive support
    if isinstance(profit_threshold, (int, float)):
        profit_thresholds = np.full(len(df), profit_threshold)
    else:
        profit_thresholds = profit_threshold.values

    if isinstance(stop_threshold, (int, float)):
        stop_thresholds = np.full(len(df), stop_threshold)
    else:
        stop_thresholds = stop_threshold.values

    last_event_idx = -min_event_spacing  # Track last signal to avoid overlaps

    for i in range(len(df) - horizon):
        signal = consensus_signals[i]

        # Only create labels where we have a signal
        if signal == 0:
            continue

        # Handle overlapping events: skip if too close to previous signal
        if (i - last_event_idx) < min_event_spacing:
            continue

        # Edge window handling: skip events too close to end of available data
        if i + horizon >= len(df):
            # Mark as NaN - incomplete forward window
            continue

        entry_price = close_prices[i]
        exit_price = None
        exit_reason = None

        # Get adaptive thresholds for this event
        profit_thr = profit_thresholds[i]
        stop_thr = stop_thresholds[i]

        # Look ahead up to horizon bars
        for j in range(1, horizon + 1):
            if i + j >= len(df):
                break

            future_price = close_prices[i + j]

            if signal > 0:  # Long signal
                pnl = (future_price - entry_price) / entry_price

                # Hit profit target
                if pnl >= profit_thr:
                    exit_price = future_price
                    exit_reason = 'profit'
                    break
                # Hit stop loss
                elif pnl <= -stop_thr:
                    exit_price = future_price
                    exit_reason = 'stop'
                    break

            elif signal < 0:  # Short signal
                pnl = (entry_price - future_price) / entry_price

                # Hit profit target
                if pnl >= profit_thr:
                    exit_price = future_price
                    exit_reason = 'profit'
                    break
                # Hit stop loss
                elif pnl <= -stop_thr:
                    exit_price = future_price
                    exit_reason = 'stop'
                    break

        # If no exit, use end-of-horizon price (timeout)
        if exit_price is None:
            exit_price = close_prices[min(i + horizon, len(df) - 1)]
            exit_reason = 'timeout'

        # Compute realized return accounting for transaction costs
        if signal > 0:  # Long
            gross_return = (exit_price - entry_price) / entry_price
        else:  # Short
            gross_return = (entry_price - exit_price) / entry_price

        net_return = gross_return - transaction_cost  # Subtract costs

        # Store results
        realized_returns.iloc[i] = net_return
        binary_labels.iloc[i] = 1.0 if net_return > 0 else 0.0

        last_event_idx = i  # Update last event position

    return realized_returns, binary_labels


def create_meta_features(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    volume_available: bool = True,
    include_raw_signals: bool = False,
    use_kalman: bool = True
) -> pd.DataFrame:
    """
    Create features for the meta-model.

    CRITICAL: By default, does NOT include raw signal values to avoid circular behavior.
    Features capture market context, not the signals themselves.

    ENHANCED: Includes Kalman-filtered versions of technical indicators and
    volatility-based regime features.

    Args:
        df: DataFrame with OHLCV data
        signals: DataFrame with primary signals (used only for context)
        volume_available: Whether volume data is available
        include_raw_signals: WARNING: Set True only for ablation tests
        use_kalman: Whether to use Kalman filtering for indicators

    Returns:
        DataFrame of features for meta-model
    """
    features = pd.DataFrame(index=df.index)

    # ===== VOLATILITY FEATURES (ENHANCED) =====

    # Log returns (more stable)
    log_ret = np.log(df['close']).diff()
    features['log_ret'] = log_ret

    # Multiple volatility windows
    features['volatility_1h'] = log_ret.rolling(window=4).std()  # 4 x 15min bars
    features['volatility_4h'] = log_ret.rolling(window=16).std()
    features['volatility_1d'] = log_ret.rolling(window=96).std()  # daily

    # Volatility of volatility (regime instability)
    features['vol_of_vol'] = features['volatility_1h'].rolling(window=20).std()

    # Volatility ratio (current vs baseline)
    vol_baseline = features['volatility_1d'].rolling(96).mean()
    features['vol_ratio'] = features['volatility_1d'] / (vol_baseline + 1e-8)

    # ===== VOLATILITY REGIME LABELING =====

    # Compute rolling volatility for regime detection
    vol_for_regime = log_ret.rolling(96).std()

    # Create regime labels using quantiles
    try:
        regime_labels = pd.qcut(
            vol_for_regime.dropna(),
            q=3,
            labels=['low', 'medium', 'high'],
            duplicates='drop'
        )
        # Reindex to match original data
        features['volatility_regime'] = regime_labels.reindex(df.index)

        # One-hot encode (drop first to avoid multicollinearity)
        regime_dummies = pd.get_dummies(features['volatility_regime'], prefix='vol_regime', drop_first=True)
        features = features.join(regime_dummies)
    except Exception as e:
        tprint(f"⚠️ Warning: Could not create volatility regimes: {e}", "WARNING")
        features['vol_regime_medium'] = 0
        features['vol_regime_high'] = 0

    # ===== KALMAN-FILTERED TECHNICAL INDICATORS =====

    # Compute raw indicators
    df_local = df.copy()
    df_local['rsi'] = compute_rsi(df_local['close'], period=14)
    df_local['sma_fast'] = df_local['close'].rolling(10).mean()
    df_local['sma_slow'] = df_local['close'].rolling(30).mean()
    df_local['momentum'] = df_local['close'].pct_change(10)

    if use_kalman:
        # Kalman-filtered trend
        kalman_trend, kalman_uncertainty = kalman_smooth_trend(df['close'], Q=1e-5, R=0.01)
        features['kalman_trend'] = kalman_trend
        features['kalman_uncertainty'] = kalman_uncertainty

        # Kalman-filtered RSI
        kf_rsi = KalmanFilter1D(Q=1e-4, R=0.1, initial_value=50.0)
        kalman_rsi, _ = kf_rsi.filter_series(df_local['rsi'])
        features['rsi_kalman'] = kalman_rsi

        # Kalman-filtered MA distance
        ma_distance = df_local['sma_fast'] - df_local['sma_slow']
        kf_ma = KalmanFilter1D(Q=1e-5, R=0.01, initial_value=0.0)
        kalman_ma_distance, _ = kf_ma.filter_series(ma_distance)
        features['ma_distance_kalman'] = kalman_ma_distance

        # Kalman-filtered momentum
        kf_mom = KalmanFilter1D(Q=1e-4, R=0.01, initial_value=0.0)
        kalman_momentum, _ = kf_mom.filter_series(df_local['momentum'])
        features['momentum_kalman'] = kalman_momentum

        # Keep raw for reference (diagnostic purposes)
        features['rsi_raw'] = df_local['rsi']
        features['ma_distance_raw'] = ma_distance
        features['momentum_raw'] = df_local['momentum']
    else:
        # Use raw indicators
        features['rsi'] = df_local['rsi']
        features['ma_distance'] = df_local['sma_fast'] - df_local['sma_slow']
        features['momentum'] = df_local['momentum']

    # ===== VOLATILITY-NORMALIZED FEATURES (ENHANCED) =====

    # Normalize momentum and MA distance by current volatility
    vol_1h = features['volatility_1h'].replace(0, np.nan)  # Avoid division by zero

    if use_kalman:
        features['momentum_per_vol'] = features['momentum_kalman'] / (vol_1h + 1e-8)
        features['ma_distance_per_vol'] = features['ma_distance_kalman'] / (df['close'] * vol_1h + 1e-8)
    else:
        features['momentum_per_vol'] = features['momentum'] / (vol_1h + 1e-8)
        features['ma_distance_per_vol'] = features['ma_distance'] / (df['close'] * vol_1h + 1e-8)

    # NEW: MACD histogram normalized by ATR
    macd_line, macd_signal_line, macd_hist = compute_macd(df['close'])
    features['macd_hist_raw'] = macd_hist
    features['macd_hist_per_atr'] = macd_hist / (features['atr_14'] + 1e-8)

    # NEW: RSI slope normalized by short-term volatility
    rsi_slope = df_local['rsi'].diff()
    features['rsi_slope'] = rsi_slope
    features['rsi_slope_per_vol'] = rsi_slope / (vol_1h + 1e-8)

    # NEW: Price change normalized by ATR (volatility-adjusted momentum)
    price_change = df['close'].diff()
    features['price_change_per_atr'] = price_change / (features['atr_14'] + 1e-8)

    # NEW: MACD delta (change in histogram) normalized by volatility
    macd_delta = macd_hist.diff()
    features['macd_delta'] = macd_delta
    features['macd_delta_per_vol'] = macd_delta / (vol_1h + 1e-8)

    # ===== TRADITIONAL VOLATILITY FEATURES (BACKWARD COMPATIBLE) =====

    returns = df['close'].pct_change()
    features['volatility_5'] = returns.rolling(5).std()
    features['volatility_20'] = returns.rolling(20).std()
    features['volatility_ratio'] = features['volatility_5'] / (features['volatility_20'] + 1e-8)

    # Use EMA for smoothing (vectorized - MUCH faster than row-by-row iteration)
    # EMA of squared returns, then take sqrt
    alpha = 0.1  # Same as previous manual calculation
    features['volatility_ema'] = np.sqrt((returns**2).ewm(alpha=alpha, adjust=False).mean())

    # ===== TREND STRENGTH =====

    features['sma_slope'] = df['close'].rolling(10).mean().pct_change(5)
    features['price_vs_sma20'] = (
        (df['close'] - df['close'].rolling(20).mean()) /
        (df['close'].rolling(20).mean() + 1e-8)
    )

    # ADX-like trend strength (simplified)
    high_low = df['high'] - df['low']
    features['atr_14'] = high_low.rolling(14).mean()
    features['atr_ratio'] = features['atr_14'] / (df['close'] + 1e-8)

    # ===== VOLUME CONTEXT =====

    if volume_available and 'volume' in df.columns:
        vol_sma = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / (vol_sma + 1e-8)
        features['volume_trend'] = (
            df['volume'].rolling(5).mean() / (vol_sma + 1e-8)
        )
        # Volume-price divergence
        features['vol_price_corr'] = returns.rolling(20).corr(
            df['volume'].pct_change()
        )

        # Volume z-score (regime measure)
        vol_mean = df['volume'].rolling(96).mean()
        vol_std = df['volume'].rolling(96).std()
        features['volume_zscore'] = (df['volume'] - vol_mean) / (vol_std + 1e-8)
    else:
        features['volume_ratio'] = 1.0
        features['volume_trend'] = 1.0
        features['vol_price_corr'] = 0.0
        features['volume_zscore'] = 0.0

    # ===== MARKET MOMENTUM =====

    features['momentum_5'] = df['close'].pct_change(5)
    features['momentum_10'] = df['close'].pct_change(10)
    features['momentum_20'] = df['close'].pct_change(20)

    # Smoothed momentum using EMA
    features['momentum_ema'] = features['momentum_10'].ewm(span=5).mean()

    # ===== RANGE POSITION =====

    recent_high = df['high'].rolling(20).max()
    recent_low = df['low'].rolling(20).min()
    features['range_position'] = (
        (df['close'] - recent_low) / (recent_high - recent_low + 1e-8)
    )

    # ===== ENTROPY (SIMPLE MEASURE) =====

    # Price entropy using returns distribution
    returns_abs = returns.abs().rolling(20).mean()
    features['returns_entropy'] = -returns_abs * np.log(returns_abs + 1e-8)

    # ===== SHORT-TERM AUTOCORRELATION / MOMENTUM METRICS (NEW) =====

    # 3-bar autocorrelation for price, RSI, MACD histogram
    features['price_autocorr_3'] = compute_autocorrelation(df['close'].pct_change(), lag=3)
    features['rsi_autocorr_3'] = compute_autocorrelation(df_local['rsi'], lag=3)
    features['macd_hist_autocorr_3'] = compute_autocorrelation(macd_hist, lag=3)

    # Short-term momentum over last 2-4 bars
    # RSI delta sum (momentum in RSI)
    features['rsi_delta_2bar'] = df_local['rsi'].diff().rolling(2).sum()
    features['rsi_delta_3bar'] = df_local['rsi'].diff().rolling(3).sum()
    features['rsi_delta_4bar'] = df_local['rsi'].diff().rolling(4).sum()

    # MACD histogram delta sum (momentum in MACD)
    features['macd_delta_2bar'] = macd_hist.diff().rolling(2).sum()
    features['macd_delta_3bar'] = macd_hist.diff().rolling(3).sum()
    features['macd_delta_4bar'] = macd_hist.diff().rolling(4).sum()

    # Price momentum over 2-4 bars
    features['price_momentum_2bar'] = df['close'].pct_change(2)
    features['price_momentum_3bar'] = df['close'].pct_change(3)
    features['price_momentum_4bar'] = df['close'].pct_change(4)

    # Persistence: How many consecutive bars has signal been positive/negative?
    price_change_sign = np.sign(df['close'].diff())
    features['price_direction_persistence'] = price_change_sign.rolling(4).apply(
        lambda x: (x == x.iloc[-1]).sum() if len(x) > 0 else 0,
        raw=False
    )

    # RSI direction persistence (useful for mean reversion)
    rsi_change_sign = np.sign(df_local['rsi'].diff())
    features['rsi_direction_persistence'] = rsi_change_sign.rolling(4).apply(
        lambda x: (x == x.iloc[-1]).sum() if len(x) > 0 else 0,
        raw=False
    )

    # MACD histogram persistence
    macd_hist_sign = np.sign(macd_hist)
    features['macd_hist_persistence'] = macd_hist_sign.rolling(4).apply(
        lambda x: (x == x.iloc[-1]).sum() if len(x) > 0 else 0,
        raw=False
    )

    # ===== FEATURE INTERACTIONS (NEW) =====

    # RSI slope × MACD histogram magnitude (strong combo indicator)
    features['rsi_slope_x_macd_mag'] = features['rsi_slope'] * macd_hist.abs()

    # Price distance from EMA × volatility-adjusted RSI change
    ema_20 = df['close'].ewm(span=20).mean()
    price_distance_from_ema = (df['close'] - ema_20) / (ema_20 + 1e-8)
    rsi_change_vol_adj = features['rsi_slope_per_vol']
    features['price_dist_x_rsi_change'] = price_distance_from_ema * rsi_change_vol_adj

    # Volatility × momentum (high vol + strong momentum = potential breakout)
    features['vol_x_momentum'] = features['volatility_1h'] * features['momentum_10'].abs()

    # ATR ratio × MACD histogram (volatility context for MACD)
    features['atr_ratio_x_macd'] = features['atr_ratio'] * macd_hist.abs()

    # Volume ratio × momentum (volume confirmation)
    features['volume_x_momentum'] = features['volume_ratio'] * features['momentum_5'].abs()

    # ===== TIME-BASED FEATURES =====

    if isinstance(df.index, pd.DatetimeIndex):
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
    else:
        features['hour'] = 0
        features['day_of_week'] = 0

    # ===== RAW SIGNALS (OPTIONAL, FOR DIAGNOSTICS) =====

    if include_raw_signals:
        tprint("⚠️ WARNING: Including raw signal features - may cause circular behavior", "WARNING")
        features['signal_strength'] = signals[['rsi', 'ma', 'mom']].abs().sum(axis=1)
        features['signal_consensus'] = signals['consensus'].abs()

    return features


def fit_probability_to_return_mapping(
    probabilities: np.ndarray,
    realized_returns: np.ndarray,
    method: str = 'isotonic'
) -> IsotonicRegression:
    """
    Fit mapping from predicted probability to expected return.

    Uses isotonic regression to create a monotonic mapping that captures
    the empirical relationship between model confidence and realized returns.

    CRITICAL: Must use out-of-fold probabilities to avoid leakage.

    Args:
        probabilities: Out-of-fold predicted probabilities
        realized_returns: Realized returns for those events
        method: 'isotonic' or 'binned'

    Returns:
        Fitted IsotonicRegression model
    """
    # Remove NaN values
    mask = ~(np.isnan(probabilities) | np.isnan(realized_returns))
    p_clean = probabilities[mask]
    r_clean = realized_returns[mask]

    if len(p_clean) < 10:
        tprint("⚠️ Warning: Very few samples for probability mapping", "WARNING")

    if method == 'isotonic':
        # Isotonic regression: monotonic mapping
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(p_clean, r_clean)
        return iso

    elif method == 'binned':
        # Alternative: binned approach (less smooth but more robust)
        # Not implemented here, but would bin probabilities and take mean return per bin
        raise NotImplementedError("Binned method not yet implemented")

    else:
        raise ValueError(f"Unknown method: {method}")


def translate_to_targets_with_isotonic(
    realized_returns: pd.Series,
    probabilities: np.ndarray,
    signals: pd.DataFrame,
    iso_regressor: IsotonicRegression
) -> Tuple[pd.Series, pd.Series]:
    """
    Translate probabilities to continuous targets using isotonic regression.

    This creates economically meaningful targets based on expected returns.

    Args:
        realized_returns: Actual returns (used only for validation)
        probabilities: Predicted probabilities from meta-model
        signals: Signal directions
        iso_regressor: Fitted isotonic regression model

    Returns:
        Tuple of (target_long, target_short)
    """
    target_long = pd.Series(0.0, index=realized_returns.index)
    target_short = pd.Series(0.0, index=realized_returns.index)

    consensus = signals['consensus'].values

    # VECTORIZED: Predict on entire probability array at once (much faster)
    expected_returns = iso_regressor.predict(probabilities)
    expected_returns = np.maximum(0, expected_returns)  # Clip negative returns to 0

    # Vectorized assignment based on signal direction
    long_mask = (consensus > 0) & (~realized_returns.isna())
    short_mask = (consensus < 0) & (~realized_returns.isna())

    target_long.iloc[long_mask] = expected_returns[long_mask]
    target_short.iloc[short_mask] = expected_returns[short_mask]

    return target_long, target_short


def generate_diagnostics_report(
    labeled_data: pd.DataFrame,
    meta_features: pd.DataFrame,
    binary_labels: pd.Series,
    realized_returns: pd.Series,
    smoothed_labels: pd.Series,
    probabilities: np.ndarray,
    final_model: RandomForestClassifier,
    config: Dict[str, Any],
    output_dir: Path
) -> str:
    """
    Generate comprehensive diagnostics report for meta-labeling.

    This creates a markdown document with:
    1. Label distribution analysis
    2. Signal coverage/sparsity
    3. Feature correlation analysis
    4. P&L distribution per label
    5. Time-series stability and regime analysis
    6. Out-of-fold probability diagnostics
    7. SHAP/feature importance

    Args:
        labeled_data: Full labeled dataset
        meta_features: Meta-features used for training
        binary_labels: Binary 0/1 labels
        realized_returns: Realized returns for each event
        smoothed_labels: Kalman-smoothed continuous labels
        probabilities: Model predicted probabilities
        final_model: Trained meta-model
        config: Configuration dictionary
        output_dir: Directory to save report

    Returns:
        Path to saved report
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"meta_labeling_diagnostics_{timestamp}.md"

    # Prepare data
    labeled_mask = ~binary_labels.isna()
    n_labeled = labeled_mask.sum()

    report_lines = []
    report_lines.append("# Meta-Labeling Diagnostics Report")
    report_lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"\n**Symbol:** {config.get('symbol', 'N/A')}")
    report_lines.append(f"**Timeframe:** {config.get('timeframe', 'N/A')}")
    report_lines.append(f"**Horizon:** {config.get('horizon', 'N/A')} bars")
    report_lines.append("\n---\n")

    # ===== 1. LABEL DISTRIBUTION =====
    report_lines.append("\n## 1. Label Distribution Analysis\n")

    n_positive = (binary_labels == 1.0).sum()
    n_negative = (binary_labels == 0.0).sum()
    positive_rate = n_positive / n_labeled if n_labeled > 0 else 0

    report_lines.append(f"- **Total labeled events:** {n_labeled}")
    report_lines.append(f"- **Positive labels (profitable):** {n_positive} ({positive_rate:.1%})")
    report_lines.append(f"- **Negative labels (unprofitable):** {n_negative} ({(1-positive_rate):.1%})")

    # Check for balance
    if positive_rate < 0.3:
        report_lines.append(f"\n⚠️ **Warning:** Low positive label rate ({positive_rate:.1%}) - most signals are unprofitable")
    elif positive_rate > 0.7:
        report_lines.append(f"\n⚠️ **Warning:** High positive label rate ({positive_rate:.1%}) - may indicate overfitting or leakage")
    else:
        report_lines.append(f"\n✅ **OK:** Reasonable label balance ({positive_rate:.1%})")

    # Time series of labels
    report_lines.append("\n### Label Distribution Over Time\n")

    # Resample labels by day to see trends
    if isinstance(labeled_data.index, pd.DatetimeIndex):
        try:
            daily_positive_rate = binary_labels.resample('1D').apply(
                lambda x: x.sum() / len(x) if len(x) > 0 else np.nan
            ).dropna()

            report_lines.append(f"- **Daily positive rate - Mean:** {daily_positive_rate.mean():.1%}")
            report_lines.append(f"- **Daily positive rate - Std:** {daily_positive_rate.std():.1%}")
            report_lines.append(f"- **Daily positive rate - Min:** {daily_positive_rate.min():.1%}")
            report_lines.append(f"- **Daily positive rate - Max:** {daily_positive_rate.max():.1%}")

            # Check for periods with extreme values
            extreme_low = (daily_positive_rate < 0.1).sum()
            extreme_high = (daily_positive_rate > 0.9).sum()

            if extreme_low > 0:
                report_lines.append(f"\n⚠️ **Warning:** {extreme_low} days with <10% positive labels")
            if extreme_high > 0:
                report_lines.append(f"\n⚠️ **Warning:** {extreme_high} days with >90% positive labels")
        except Exception as e:
            report_lines.append(f"\n⚠️ Could not compute daily statistics: {e}")
    else:
        report_lines.append("\n⚠️ Index is not datetime, skipping time-series analysis")

    # ===== 2. SIGNAL COVERAGE / SPARSITY =====
    report_lines.append("\n## 2. Signal Coverage and Sparsity\n")

    n_samples = len(labeled_data)
    coverage = n_labeled / n_samples if n_samples > 0 else 0

    report_lines.append(f"- **Total samples:** {n_samples}")
    report_lines.append(f"- **Labeled samples:** {n_labeled}")
    report_lines.append(f"- **Coverage:** {coverage:.1%}")

    if coverage < 0.05:
        report_lines.append(f"\n⚠️ **Warning:** Very sparse signals ({coverage:.1%}) - consider lowering signal thresholds")
    elif coverage > 0.5:
        report_lines.append(f"\n⚠️ **Warning:** Very dense signals ({coverage:.1%}) - may lead to overlapping events")
    else:
        report_lines.append(f"\n✅ **OK:** Reasonable signal coverage ({coverage:.1%})")

    # ===== 3. CORRELATION ANALYSIS =====
    report_lines.append("\n## 3. Feature-Label Correlation Analysis\n")

    # Compute correlations
    features_clean = meta_features[labeled_mask].fillna(0)
    labels_clean = binary_labels[labeled_mask]

    try:
        correlations = {}
        for col in features_clean.columns:
            corr = features_clean[col].corr(labels_clean)
            if not pd.isna(corr):
                correlations[col] = corr

        # Sort by absolute correlation
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

        report_lines.append("\n### Top 10 Most Correlated Features:\n")
        for feat, corr in sorted_corr[:10]:
            report_lines.append(f"- **{feat}:** {corr:.4f}")

        # Check for concerning correlations
        report_lines.append("\n### Correlation Health Check:\n")

        very_high = [(f, c) for f, c in sorted_corr if abs(c) > 0.8]
        very_low = [(f, c) for f, c in sorted_corr if abs(c) < 0.01]

        if very_high:
            report_lines.append(f"\n⚠️ **Warning:** {len(very_high)} features with |corr| > 0.8 (possible leakage):")
            for feat, corr in very_high[:5]:
                report_lines.append(f"  - {feat}: {corr:.4f}")

        if len(very_low) > len(sorted_corr) * 0.8:
            report_lines.append(f"\n⚠️ **Warning:** {len(very_low)} features with |corr| < 0.01 (mostly uninformative)")

    except Exception as e:
        report_lines.append(f"\n⚠️ Could not compute correlations: {e}")

    # ===== 4. P&L DISTRIBUTION PER LABEL =====
    report_lines.append("\n## 4. P&L Distribution Per Label\n")

    returns_labeled = realized_returns[labeled_mask]
    labels_clean = binary_labels[labeled_mask]

    returns_positive = returns_labeled[labels_clean == 1]
    returns_negative = returns_labeled[labels_clean == 0]

    report_lines.append("### Label = 1 (Profitable Signals):\n")
    report_lines.append(f"- **Count:** {len(returns_positive)}")
    report_lines.append(f"- **Mean return:** {returns_positive.mean():.2%}")
    report_lines.append(f"- **Median return:** {returns_positive.median():.2%}")
    report_lines.append(f"- **Std return:** {returns_positive.std():.2%}")
    report_lines.append(f"- **% Actually positive:** {(returns_positive > 0).sum() / len(returns_positive):.1%}")

    report_lines.append("\n### Label = 0 (Unprofitable Signals):\n")
    report_lines.append(f"- **Count:** {len(returns_negative)}")
    report_lines.append(f"- **Mean return:** {returns_negative.mean():.2%}")
    report_lines.append(f"- **Median return:** {returns_negative.median():.2%}")
    report_lines.append(f"- **Std return:** {returns_negative.std():.2%}")
    report_lines.append(f"- **% Actually positive:** {(returns_negative > 0).sum() / len(returns_negative):.1%}")

    # Overlap check
    overlap_pos_in_neg = (returns_positive < 0).sum()
    overlap_neg_in_pos = (returns_negative > 0).sum()

    report_lines.append("\n### Labeling Quality:\n")
    if len(returns_positive) > 0 and len(returns_negative) > 0:
        pct_overlap = (overlap_pos_in_neg + overlap_neg_in_pos) / (len(returns_positive) + len(returns_negative))
        report_lines.append(f"- **Label overlap:** {pct_overlap:.1%}")

        if pct_overlap > 0.4:
            report_lines.append(f"\n⚠️ **Warning:** High label overlap ({pct_overlap:.1%}) - labels may be too noisy or horizon too short")
        else:
            report_lines.append(f"\n✅ **OK:** Acceptable label overlap ({pct_overlap:.1%})")

    # ===== 5. TIME-SERIES STABILITY / REGIME CHECK =====
    report_lines.append("\n## 5. Time-Series Stability and Regime Analysis\n")

    if isinstance(labeled_data.index, pd.DatetimeIndex):
        try:
            # Compute daily metrics
            daily_win_rate = binary_labels.resample('1D').mean()
            daily_mean_return = realized_returns.resample('1D').mean()

            # Volatility
            log_ret = labeled_data['log_ret'] if 'log_ret' in labeled_data.columns else np.log(labeled_data['close']).diff()
            daily_volatility = log_ret.resample('1D').std()

            # Volume z-score
            if 'volume' in labeled_data.columns:
                vol_mean = labeled_data['volume'].rolling(96).mean()
                vol_std = labeled_data['volume'].rolling(96).std()
                volume_zscore = (labeled_data['volume'] - vol_mean) / (vol_std + 1e-8)
                daily_volume_z = volume_zscore.resample('1D').mean()
            else:
                daily_volume_z = pd.Series(0, index=daily_volatility.index)

            # Correlation with volatility
            corr_winrate_vol = daily_win_rate.corr(daily_volatility)
            corr_return_vol = daily_mean_return.corr(daily_volatility)

            report_lines.append(f"- **Win rate vs Volatility correlation:** {corr_winrate_vol:.4f}")
            report_lines.append(f"- **Mean return vs Volatility correlation:** {corr_return_vol:.4f}")

            if abs(corr_winrate_vol) > 0.5:
                report_lines.append(f"\n⚠️ **Warning:** Strong correlation between win rate and volatility - performance is regime-dependent")
            else:
                report_lines.append(f"\n✅ **OK:** Win rate not strongly correlated with volatility")

            # Detect regime shifts
            report_lines.append("\n### Regime Shift Detection:\n")

            # Rolling correlation
            rolling_corr = daily_win_rate.rolling(30).corr(daily_volatility)
            report_lines.append(f"- **30-day rolling correlation (win rate vs vol) - Mean:** {rolling_corr.mean():.4f}")
            report_lines.append(f"- **30-day rolling correlation (win rate vs vol) - Std:** {rolling_corr.std():.4f}")

        except Exception as e:
            report_lines.append(f"\n⚠️ Could not compute regime analysis: {e}")
    else:
        report_lines.append("\n⚠️ Index is not datetime, skipping time-series analysis")

    # ===== 6. OUT-OF-FOLD PROBABILITY DIAGNOSTICS =====
    report_lines.append("\n## 6. Model Probability Diagnostics\n")

    try:
        # Calibration analysis
        prob_clean = probabilities[labeled_mask]
        labels_clean_array = binary_labels[labeled_mask].values

        # Bin probabilities
        prob_bins = pd.cut(prob_clean, bins=10, labels=False)
        calibration_data = []

        for bin_idx in range(10):
            mask = (prob_bins == bin_idx)
            if mask.sum() > 0:
                mean_prob = prob_clean[mask].mean()
                mean_label = labels_clean_array[mask].mean()
                count = mask.sum()
                calibration_data.append((mean_prob, mean_label, count))

        report_lines.append("\n### Calibration (Predicted Probability vs Actual Success Rate):\n")
        report_lines.append("\n| Predicted Prob | Actual Success Rate | Count |")
        report_lines.append("|---------------|---------------------|-------|")

        for mean_prob, mean_label, count in calibration_data:
            report_lines.append(f"| {mean_prob:.3f} | {mean_label:.3f} | {count} |")

        # Calculate calibration error
        calibration_error = np.mean([abs(p - l) for p, l, c in calibration_data])
        report_lines.append(f"\n- **Mean calibration error:** {calibration_error:.3f}")

        if calibration_error > 0.15:
            report_lines.append(f"\n⚠️ **Warning:** High calibration error - model probabilities may not be well-calibrated")
        else:
            report_lines.append(f"\n✅ **OK:** Reasonable calibration")

    except Exception as e:
        report_lines.append(f"\n⚠️ Could not compute calibration: {e}")

    # Probability distribution
    report_lines.append("\n### Probability Distribution:\n")
    report_lines.append(f"- **Mean probability:** {probabilities.mean():.3f}")
    report_lines.append(f"- **Median probability:** {np.median(probabilities):.3f}")
    report_lines.append(f"- **Std probability:** {probabilities.std():.3f}")

    # Check for collapsed predictions
    if probabilities.std() < 0.05:
        report_lines.append(f"\n⚠️ **Warning:** Very low probability variance - model may not be learning useful patterns")

    # ===== 7. SHAP / FEATURE IMPORTANCE =====
    report_lines.append("\n## 7. Feature Importance Analysis\n")

    try:
        # Get feature importances from model
        feature_importances = dict(zip(meta_features.columns, final_model.feature_importances_))
        sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)

        report_lines.append("\n### Top 20 Features by Importance:\n")
        for i, (feat, imp) in enumerate(sorted_importances[:20], 1):
            report_lines.append(f"{i}. **{feat}:** {imp:.4f}")

        # Check for suspicious features
        report_lines.append("\n### Feature Importance Health Check:\n")

        # Check if any single feature dominates
        top_importance = sorted_importances[0][1] if sorted_importances else 0
        if top_importance > 0.5:
            report_lines.append(f"\n⚠️ **Warning:** Single feature dominates ({sorted_importances[0][0]}: {top_importance:.2%}) - possible leakage")
        else:
            report_lines.append(f"\n✅ **OK:** No single feature dominates")

        # Check if Kalman features are used
        kalman_features = [f for f, _ in sorted_importances if 'kalman' in f.lower()]
        if kalman_features:
            report_lines.append(f"\n✅ Kalman-filtered features are being used ({len(kalman_features)} features)")

        # Check if volatility features are used
        vol_features = [f for f, _ in sorted_importances[:10] if 'vol' in f.lower()]
        if vol_features:
            report_lines.append(f"\n✅ Volatility-based features in top 10: {', '.join(vol_features)}")

    except Exception as e:
        report_lines.append(f"\n⚠️ Could not compute feature importance: {e}")

    # ===== 8. KALMAN SMOOTHED LABELS =====
    report_lines.append("\n## 8. Kalman Smoothed Labels Analysis\n")

    if smoothed_labels is not None and not smoothed_labels.isna().all():
        smoothed_labeled = smoothed_labels[labeled_mask]

        report_lines.append(f"- **Mean smoothed label:** {smoothed_labeled.mean():.3f}")
        report_lines.append(f"- **Median smoothed label:** {smoothed_labeled.median():.3f}")
        report_lines.append(f"- **Std smoothed label:** {smoothed_labeled.std():.3f}")

        # Correlation with binary labels
        corr_smoothed_binary = smoothed_labeled.corr(binary_labels[labeled_mask])
        report_lines.append(f"- **Correlation with binary labels:** {corr_smoothed_binary:.3f}")

        # Correlation with realized returns
        corr_smoothed_returns = smoothed_labeled.corr(realized_returns[labeled_mask])
        report_lines.append(f"- **Correlation with realized returns:** {corr_smoothed_returns:.3f}")

        if corr_smoothed_binary < 0.5:
            report_lines.append(f"\n⚠️ **Warning:** Low correlation between smoothed and binary labels - Kalman filter may be over-smoothing")
        else:
            report_lines.append(f"\n✅ **OK:** Good correlation between smoothed and binary labels")

    else:
        report_lines.append("\n⚠️ Smoothed labels not available")

    # ===== SUMMARY =====
    report_lines.append("\n---\n")
    report_lines.append("\n## Summary and Recommendations\n")

    report_lines.append("\n### Key Findings:\n")
    report_lines.append(f"1. Label balance: {positive_rate:.1%} positive")
    report_lines.append(f"2. Signal coverage: {coverage:.1%}")
    report_lines.append(f"3. Mean return (label=1): {returns_positive.mean():.2%}")
    report_lines.append(f"4. Mean return (label=0): {returns_negative.mean():.2%}")
    report_lines.append(f"5. Calibration error: {calibration_error:.3f}")

    # Write report
    report_content = "\n".join(report_lines)

    with open(report_path, 'w') as f:
        f.write(report_content)

    tprint(f"📊 Diagnostics report saved to {report_path}", "SUCCESS")

    return str(report_path)


def create_base_models(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create base models for ensemble with proper regularization.

    Three models optimized for different strengths:
    - LGBM: Gradient boosting for ranking and AUC
    - LogisticRegression: Linear blender with elastic net regularization
    - RandomForest: Non-linear ensemble for robustness

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of {model_name: model_instance}
    """
    models = {}

    # LightGBM: Optimized for AUC with early stopping and regularization
    models['lgbm'] = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        n_estimators=500,
        max_depth=5,
        learning_rate=0.01,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=0.1,  # L2 regularization
        n_jobs=-1,
        verbose=-1,
        random_state=42
    )

    # Logistic Regression with elastic net (L1 + L2)
    models['logreg'] = LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        C=1.0,
        l1_ratio=0.5,  # Balance between L1 and L2
        max_iter=1000,
        n_jobs=-1,
        random_state=42
    )

    # Random Forest with conservative parameters
    models['rf'] = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=20,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    )

    return models


def train_ensemble_with_kfold(
    X: pd.DataFrame,
    y: pd.Series,
    horizon: int,
    n_splits: int = 5,
    verbose: bool = True
) -> Tuple[Dict[str, Any], pd.Series]:
    """
    Train ensemble models with K-fold cross-fitting to prevent leakage.

    CRITICAL: Uses purged time-series CV to avoid lookahead bias.
    Each model is trained on fold ∖i and predicts on fold i.

    Args:
        X: Feature matrix
        y: Binary labels
        horizon: Forward-looking horizon (for purging)
        n_splits: Number of CV folds
        verbose: Whether to print progress

    Returns:
        Tuple of (trained_models_dict, out_of_fold_predictions_series)
    """
    # Initialize storage
    trained_models = {'lgbm': [], 'logreg': [], 'rf': []}
    oof_predictions = {
        'lgbm': pd.Series(np.nan, index=X.index),
        'logreg': pd.Series(np.nan, index=X.index),
        'rf': pd.Series(np.nan, index=X.index)
    }

    # Time-series CV
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        if verbose:
            tprint(f"  Fold {fold_idx + 1}/{n_splits}...", "INFO")

        # Purge training indices to avoid lookahead
        train_idx_purged = purge_training_idxs(
            train_idx,
            test_idx[0],
            test_idx[-1] + 1,
            horizon=horizon
        )

        if len(train_idx_purged) == 0:
            if verbose:
                tprint(f"    ⚠️ All training samples purged, skipping fold", "WARNING")
            continue

        # Get train/test splits
        X_train = X.iloc[train_idx_purged]
        y_train = y.iloc[train_idx_purged]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        # Filter NaN labels
        train_mask = ~y_train.isna()
        test_mask = ~y_test.isna()

        if train_mask.sum() < 10 or test_mask.sum() < 5:
            if verbose:
                tprint(f"    ⚠️ Too few samples, skipping fold", "WARNING")
            continue

        X_train_clean = X_train[train_mask].fillna(0)
        y_train_clean = y_train[train_mask]
        X_test_clean = X_test[test_mask].fillna(0)
        y_test_clean = y_test[test_mask]

        # Train each base model
        base_models = create_base_models({})

        for model_name, model in base_models.items():
            try:
                # Train
                model.fit(X_train_clean, y_train_clean)
                trained_models[model_name].append(model)

                # Predict on test fold
                y_pred_proba = model.predict_proba(X_test_clean)[:, 1]

                # Store OOF predictions
                test_indices_with_labels = test_idx[test_mask]
                oof_predictions[model_name].iloc[test_indices_with_labels] = y_pred_proba

                # Metrics
                try:
                    auc = roc_auc_score(y_test_clean, y_pred_proba)
                    if verbose:
                        tprint(f"    ✓ {model_name}: AUC={auc:.3f}", "INFO")
                except:
                    if verbose:
                        tprint(f"    ⚠️ {model_name}: Could not compute AUC", "WARNING")

            except Exception as e:
                if verbose:
                    tprint(f"    ❌ {model_name} failed: {e}", "ERROR")

    # Combine OOF predictions into DataFrame
    oof_df = pd.DataFrame(oof_predictions, index=X.index)

    return trained_models, oof_df


def calibrate_ensemble(
    oof_predictions: pd.DataFrame,
    y_true: pd.Series,
    realized_returns: pd.Series,
    meta_features: pd.DataFrame,
    method: str = 'platt_isotonic',
    include_context: bool = True
) -> Tuple[Dict[str, Any], IsotonicRegression]:
    """
    Calibrate ensemble predictions using Platt scaling + isotonic regression.

    Two-stage calibration:
    1. Platt scaling per individual model (sigmoid calibration)
    2. Isotonic regression on final blended output (with optional entropy/volatility)

    Args:
        oof_predictions: Out-of-fold predictions from each base model (DataFrame)
        y_true: Binary ground truth labels
        realized_returns: Realized returns for isotonic mapping
        meta_features: Meta-features (for optional entropy/volatility context)
        method: 'platt_isotonic' or 'isotonic_only'
        include_context: Whether to include entropy/volatility in final calibration

    Returns:
        Tuple of (platt_calibrators_dict, isotonic_regressor)
    """
    platt_calibrators = {}

    # Valid data mask
    valid_mask = ~y_true.isna()
    for col in oof_predictions.columns:
        valid_mask &= ~oof_predictions[col].isna()

    if valid_mask.sum() < 20:
        tprint("⚠️ Warning: Too few samples for calibration", "WARNING")
        return {}, None

    y_valid = y_true[valid_mask]
    returns_valid = realized_returns[valid_mask]

    # STAGE 1: Platt scaling per model
    calibrated_predictions = pd.DataFrame(index=oof_predictions.index)

    if method == 'platt_isotonic':
        tprint("  📈 Stage 1: Applying Platt scaling to each base model...", "INFO")

        for model_name in oof_predictions.columns:
            try:
                # Get OOF predictions for this model
                oof_model = oof_predictions[model_name][valid_mask].values.reshape(-1, 1)

                # Fit Platt scaler (logistic regression on top of predictions)
                platt_scaler = LogisticRegression(max_iter=1000, random_state=42)
                platt_scaler.fit(oof_model, y_valid)
                platt_calibrators[model_name] = platt_scaler

                # Apply calibration to full predictions
                all_preds = oof_predictions[model_name].fillna(0.5).values.reshape(-1, 1)
                calibrated_predictions[model_name] = platt_scaler.predict_proba(all_preds)[:, 1]

                tprint(f"    ✓ {model_name} calibrated", "INFO")

            except Exception as e:
                tprint(f"    ⚠️ {model_name} calibration failed: {e}", "WARNING")
                calibrated_predictions[model_name] = oof_predictions[model_name]
    else:
        # Skip Platt, use raw predictions
        calibrated_predictions = oof_predictions.copy()

    # STAGE 2: Blend models with soft voting
    tprint("  📊 Stage 2: Blending models with soft voting...", "INFO")

    # Simple average (can be weighted based on validation performance)
    ensemble_probs = calibrated_predictions.mean(axis=1)

    # STAGE 3: Isotonic regression on ensemble output
    tprint("  📈 Stage 3: Applying isotonic regression to ensemble...", "INFO")

    ensemble_valid = ensemble_probs[valid_mask].values

    # Optional: Include entropy/volatility as context
    if include_context and meta_features is not None:
        try:
            # Compute prediction entropy (uncertainty)
            pred_entropy = -ensemble_valid * np.log(ensemble_valid + 1e-8) - \
                           (1 - ensemble_valid) * np.log(1 - ensemble_valid + 1e-8)

            # Get volatility from meta-features
            if 'volatility_1h' in meta_features.columns:
                vol_valid = meta_features['volatility_1h'][valid_mask].fillna(0).values
            else:
                vol_valid = np.zeros(len(ensemble_valid))

            # Combine: probability, entropy, volatility
            # Weight: prob has highest weight, entropy/vol are context
            calibration_input = (
                0.7 * ensemble_valid +
                0.2 * (1 - pred_entropy / pred_entropy.max()) +  # Lower entropy = more confident
                0.1 * (vol_valid / (vol_valid.max() + 1e-8))     # Volatility normalization
            )

            tprint(f"    ✓ Including entropy + volatility context", "INFO")

        except Exception as e:
            tprint(f"    ⚠️ Could not include context, using probabilities only: {e}", "WARNING")
            calibration_input = ensemble_valid
    else:
        calibration_input = ensemble_valid

    # Fit isotonic regression: calibrated_prob -> expected_return
    try:
        iso_regressor = IsotonicRegression(out_of_bounds='clip')
        iso_regressor.fit(calibration_input, returns_valid.values)

        tprint(f"    ✓ Isotonic regression fitted on {len(calibration_input)} samples", "SUCCESS")

    except Exception as e:
        tprint(f"    ❌ Isotonic regression failed: {e}", "ERROR")
        iso_regressor = None

    return platt_calibrators, iso_regressor


def add_signal_disagreement(
    oof_predictions: pd.DataFrame,
    meta_features: pd.DataFrame
) -> pd.DataFrame:
    """
    Add signal uncertainty / model disagreement features (ENHANCED).

    This captures model consensus and uncertainty, which are strong predictors of signal quality.
    High disagreement = models don't agree = lower confidence = worse signal quality.

    NEW FEATURES:
    - signal_disagreement: Std dev across ensemble predictions
    - signal_disagreement_ema: Smoothed disagreement
    - n_models_agreeing: Number of models with similar predictions (within 0.1)
    - model_prob_range: Range between min and max model probabilities
    - model_entropy: Entropy of model predictions (measure of uncertainty)

    Args:
        oof_predictions: Out-of-fold predictions from each base model
        meta_features: Existing meta-features

    Returns:
        Updated meta-features with disagreement columns
    """
    # Compute std dev across model predictions
    disagreement = oof_predictions.std(axis=1, skipna=True)

    # Add to meta-features
    meta_features_updated = meta_features.copy()
    meta_features_updated['signal_disagreement'] = disagreement

    # Smoothed disagreement
    meta_features_updated['signal_disagreement_ema'] = disagreement.ewm(span=10).mean()

    # NEW: Count how many models agree (within 0.1 threshold of mean)
    mean_pred = oof_predictions.mean(axis=1)
    n_agreeing = []
    for idx in range(len(oof_predictions)):
        row = oof_predictions.iloc[idx]
        mean_val = mean_pred.iloc[idx]
        if pd.isna(mean_val):
            n_agreeing.append(np.nan)
        else:
            # Count models within 0.1 of mean
            count = ((row - mean_val).abs() <= 0.1).sum()
            n_agreeing.append(count)

    meta_features_updated['n_models_agreeing'] = n_agreeing

    # NEW: Range of model predictions (max - min)
    model_prob_range = oof_predictions.max(axis=1) - oof_predictions.min(axis=1)
    meta_features_updated['model_prob_range'] = model_prob_range

    # NEW: Entropy of model predictions (measure of uncertainty)
    # Normalize predictions to sum to 1, then compute entropy
    model_entropy = []
    for idx in range(len(oof_predictions)):
        row = oof_predictions.iloc[idx].dropna()
        if len(row) > 0:
            # Normalize
            row_norm = row / (row.sum() + 1e-8)
            # Compute entropy
            entropy = -(row_norm * np.log(row_norm + 1e-8)).sum()
            model_entropy.append(entropy)
        else:
            model_entropy.append(np.nan)

    meta_features_updated['model_entropy'] = model_entropy

    tprint(f"  ✓ Added signal uncertainty features:", "INFO")
    tprint(f"    - Disagreement (std): mean={disagreement.mean():.3f}, std={disagreement.std():.3f}", "INFO")
    tprint(f"    - Model agreement count: mean={np.nanmean(n_agreeing):.1f}", "INFO")
    tprint(f"    - Probability range: mean={model_prob_range.mean():.3f}", "INFO")
    tprint(f"    - Model entropy: mean={np.nanmean(model_entropy):.3f}", "INFO")

    return meta_features_updated


def compute_precision_at_k(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    k_values: List[int] = [10, 20, 50, 100]
) -> Dict[str, float]:
    """
    Compute Precision@K - precision for top-K predicted positives.

    This measures how many of the top-K highest probability predictions are actually positive.
    Critical metric for trading: we only want to trade the highest confidence signals.

    Args:
        y_true: Binary ground truth labels
        y_pred_proba: Predicted probabilities
        k_values: List of K values to evaluate

    Returns:
        Dictionary mapping k to precision@k
    """
    results = {}

    # Sort by probability (descending)
    sorted_indices = np.argsort(-y_pred_proba)

    for k in k_values:
        if k > len(y_true):
            k = len(y_true)

        # Get top-K predictions
        top_k_indices = sorted_indices[:k]
        top_k_labels = y_true[top_k_indices]

        # Precision = fraction of top-K that are positive
        precision_k = top_k_labels.sum() / k if k > 0 else 0.0
        results[f'precision@{k}'] = precision_k

    return results


def compute_brier_score(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Compute Brier score - mean squared error between predicted probabilities and outcomes.

    Lower is better (0 = perfect, 1 = worst possible).
    Measures both calibration and refinement (discrimination).

    Args:
        y_true: Binary ground truth labels
        y_pred_proba: Predicted probabilities

    Returns:
        Brier score
    """
    return np.mean((y_pred_proba - y_true) ** 2)


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures calibration: do predicted probabilities match actual frequencies?
    - ECE = 0: Perfect calibration
    - ECE > 0.1: Poor calibration

    Args:
        y_true: Binary ground truth labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins for calibration curve

    Returns:
        Expected calibration error
    """
    # Bin predictions
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    total_samples = len(y_true)

    for bin_idx in range(n_bins):
        # Find samples in this bin
        in_bin = (bin_indices == bin_idx)
        n_in_bin = in_bin.sum()

        if n_in_bin == 0:
            continue

        # Mean predicted probability in this bin
        mean_pred_prob = y_pred_proba[in_bin].mean()

        # Actual frequency of positives in this bin
        actual_freq = y_true[in_bin].mean()

        # Weighted absolute difference
        ece += (n_in_bin / total_samples) * abs(mean_pred_prob - actual_freq)

    return ece


def compute_hit_rate(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    Compute hit rate (accuracy) above threshold.

    Args:
        y_true: Binary ground truth labels
        y_pred_proba: Predicted probabilities
        threshold: Probability threshold

    Returns:
        Hit rate (accuracy)
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    return (y_pred == y_true).mean()


def compute_all_metrics(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    probabilities_valid: np.ndarray,
    k_values: List[int] = [10, 20, 50, 100]
) -> Dict[str, float]:
    """
    Compute all enhanced metrics for model evaluation.

    Args:
        y_true: Binary ground truth labels
        y_pred_proba: Predicted probabilities (full)
        probabilities_valid: Probabilities for valid samples only
        k_values: K values for Precision@K

    Returns:
        Dictionary of all metrics
    """
    metrics = {}

    # Filter to valid samples
    valid_mask = ~y_true.isna()
    y_true_clean = y_true[valid_mask].values
    y_pred_clean = probabilities_valid

    if len(y_true_clean) < 10:
        return metrics

    try:
        # Precision@K
        precision_k_results = compute_precision_at_k(y_true_clean, y_pred_clean, k_values)
        metrics.update(precision_k_results)

        # Brier score
        metrics['brier_score'] = compute_brier_score(y_true_clean, y_pred_clean)

        # Expected Calibration Error
        metrics['ece'] = compute_expected_calibration_error(y_true_clean, y_pred_clean, n_bins=10)

        # Hit rate at different thresholds
        metrics['hit_rate_0.5'] = compute_hit_rate(y_true_clean, y_pred_clean, threshold=0.5)
        metrics['hit_rate_0.6'] = compute_hit_rate(y_true_clean, y_pred_clean, threshold=0.6)
        metrics['hit_rate_0.7'] = compute_hit_rate(y_true_clean, y_pred_clean, threshold=0.7)

        # Standard metrics
        metrics['auc'] = roc_auc_score(y_true_clean, y_pred_clean)

        y_pred_binary = (y_pred_clean >= 0.5).astype(int)
        metrics['precision'] = precision_score(y_true_clean, y_pred_binary, zero_division=0)
        metrics['recall'] = recall_score(y_true_clean, y_pred_binary, zero_division=0)
        metrics['f1'] = f1_score(y_true_clean, y_pred_binary, zero_division=0)

    except Exception as e:
        tprint(f"⚠️ Warning: Could not compute some metrics: {e}", "WARNING")

    return metrics


def apply_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    variance_threshold: float = 1e-5,
    correlation_threshold: float = 0.98,
    verbose: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply feature selection to remove low-variance and highly correlated features.

    Two-stage feature selection:
    1. Remove features with very low variance (< variance_threshold on normalized scale)
    2. Remove highly correlated features (correlation > correlation_threshold)

    Args:
        X: Feature matrix
        y: Target labels
        variance_threshold: Minimum variance threshold (normalized scale)
        correlation_threshold: Maximum correlation threshold (0.98 for tree models, 0.95 for linear)
        verbose: Print selection statistics

    Returns:
        Tuple of (filtered_features, removed_feature_names)
    """
    if verbose:
        tprint(f"🔍 Feature selection: Starting with {len(X.columns)} features", "INFO")

    removed_features = []
    X_filtered = X.copy()

    # STAGE 1: Remove low-variance features
    if verbose:
        tprint(f"  Stage 1: Removing features with variance < {variance_threshold}", "INFO")

    # Normalize features first (to put them on same scale)
    X_normalized = (X_filtered - X_filtered.mean()) / (X_filtered.std() + 1e-8)

    # Compute variance on normalized features
    feature_variances = X_normalized.var()

    # Find low-variance features
    low_var_features = feature_variances[feature_variances < variance_threshold].index.tolist()

    if low_var_features:
        if verbose:
            tprint(f"    Removing {len(low_var_features)} low-variance features", "INFO")
        X_filtered = X_filtered.drop(columns=low_var_features)
        removed_features.extend(low_var_features)

    # STAGE 2: Remove highly correlated features
    if verbose:
        tprint(f"  Stage 2: Removing features with correlation > {correlation_threshold}", "INFO")

    # Compute correlation matrix
    corr_matrix = X_filtered.corr().abs()

    # Find pairs of highly correlated features
    highly_correlated = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > correlation_threshold:
                # Keep the feature with higher variance (more informative)
                col_i = corr_matrix.columns[i]
                col_j = corr_matrix.columns[j]

                var_i = X_filtered[col_i].var()
                var_j = X_filtered[col_j].var()

                # Remove the one with lower variance
                to_remove = col_j if var_i > var_j else col_i
                highly_correlated.append(to_remove)

    # Remove duplicates
    highly_correlated = list(set(highly_correlated))

    if highly_correlated:
        if verbose:
            tprint(f"    Removing {len(highly_correlated)} highly correlated features", "INFO")
        X_filtered = X_filtered.drop(columns=highly_correlated, errors='ignore')
        removed_features.extend(highly_correlated)

    if verbose:
        tprint(f"✅ Feature selection complete: {len(X_filtered.columns)} features remaining", "SUCCESS")
        tprint(f"   Removed {len(removed_features)} features total", "INFO")

    return X_filtered, removed_features


def select_top_k_signals(
    signals: pd.DataFrame,
    probabilities: np.ndarray,
    volatility: pd.Series,
    k_per_day: float = 5.0,
    min_probability: float = 0.6
) -> pd.Series:
    """
    Select top-K signals per day based on calibrated probability.

    Implements volatility-normalized ranking to ensure quality selection.

    Args:
        signals: Signal DataFrame with 'consensus' column
        probabilities: Calibrated probabilities from ensemble
        volatility: Volatility series for normalization
        k_per_day: Target number of trades per day (on average)
        min_probability: Minimum probability threshold

    Returns:
        Boolean mask of selected signals
    """
    n_samples = len(signals)
    selected_mask = pd.Series(False, index=signals.index)

    # Only consider signals with non-zero consensus
    signal_mask = (signals['consensus'] != 0) & (probabilities >= min_probability)

    if signal_mask.sum() == 0:
        tprint("  ⚠️ No signals meet minimum probability threshold", "WARNING")
        return selected_mask

    # Compute volatility-adjusted score
    # Higher probability + lower volatility = better signal
    vol_normalized = volatility / (volatility.mean() + 1e-8)
    vol_normalized = vol_normalized.clip(0.5, 2.0)  # Prevent extreme values

    # Score = probability / volatility_factor
    # This favors high-probability signals in low-volatility environments
    scores = probabilities / (vol_normalized + 0.5)
    scores[~signal_mask] = -np.inf  # Exclude non-signals

    # Determine how many signals to select
    # Assume ~96 bars per day (15min timeframe)
    bars_per_day = 96
    n_days = n_samples / bars_per_day
    total_k = int(k_per_day * n_days)

    # Select top-K by score
    if total_k > 0:
        threshold_idx = min(total_k, signal_mask.sum())
        score_threshold = np.partition(scores, -threshold_idx)[-threshold_idx]
        selected_mask = scores >= score_threshold

        n_selected = selected_mask.sum()
        actual_per_day = n_selected / n_days if n_days > 0 else 0

        tprint(f"  ✓ Selected {n_selected} signals ({actual_per_day:.1f} per day)", "SUCCESS")
    else:
        tprint(f"  ⚠️ No signals selected (total_k={total_k})", "WARNING")

    return selected_mask


class FeatureGenerationMetaLabelingStep(BaseStep):
    """
    Feature Generation Meta-Labeling Step (Enhanced).

    Improvements over basic version:
    - Computes realized returns (not just binary labels)
    - Uses isotonic regression for probability → expected return mapping
    - Avoids circular behavior (doesn't include raw signals in features)
    - Handles overlapping events and edge windows
    - Includes transaction costs
    - Uses economic metrics
    """

    def __init__(self, step_name: str = "feature_generation_meta_labeling_step"):
        """Initialize the meta-labeling step."""
        super().__init__(step_name)
        self.logger = system_logger.getChild('FeatureGenerationMetaLabeling')

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute feature generation meta-labeling with enhanced methodology.

        Args:
            config: Configuration dictionary

        Returns:
            Result dictionary with targets and metrics
        """
        start_time = datetime.now()

        # Validate required config
        required = ('symbol', 'exchange', 'timeframe')
        missing = [k for k in required if k not in config or not config[k]]
        if missing:
            error_msg = f"Missing required config keys: {', '.join(missing)}"
            tprint(f"❌ {error_msg}", "ERROR")
            return {'success': False, 'error': error_msg}

        tprint(f"🏷️ Starting enhanced meta-labeling for {config['symbol']}", "INFO")

        try:
            # Extract config parameters (using production defaults)
            profit_threshold = config.get('profit_threshold', DEFAULT_PROFIT_THRESHOLD)  # 1%
            stop_threshold = config.get('stop_threshold', DEFAULT_STOP_THRESHOLD)  # 0.5%
            horizon = config.get('horizon', 16)
            transaction_cost = config.get('transaction_cost', DEFAULT_TRANSACTION_COST)  # 0.15%
            min_event_spacing = config.get('min_event_spacing', 4)

            # Load market data
            tprint("📊 Loading market data...", "INFO")
            from src.utils.data.klines_parquet import get_klines_manager
            klines_manager = get_klines_manager(data_dir=config.get('data_dir', 'historical_data'))

            market_data = klines_manager.read_data(
                symbol=config['symbol'],
                interval=config['timeframe'],
                data_type="processed"
            )

            if market_data is None or market_data.empty:
                raise ValueError(f"No market data available for {config['symbol']} {config['timeframe']}")

            tprint(f"✅ Loaded {len(market_data)} samples", "SUCCESS")

            if 'close' not in market_data.columns:
                raise ValueError("Missing required 'close' column in market data")

            volume_available = 'volume' in market_data.columns

            # STEP 1: Generate FIXED primary signals
            tprint("🎯 Generating fixed primary signals...", "INFO")
            primary_signals = generate_primary_signals(market_data)

            n_long_signals = (primary_signals['consensus'] > 0).sum()
            n_short_signals = (primary_signals['consensus'] < 0).sum()
            tprint(f"📊 Primary signals: {n_long_signals} long, {n_short_signals} short", "INFO")

            # STEP 2: Compute volatility for adaptive thresholds
            tprint("📊 Computing volatility for adaptive thresholds...", "INFO")
            log_ret = np.log(market_data['close']).diff()
            volatility_1d = log_ret.rolling(96).std()  # Daily volatility
            vol_baseline = volatility_1d.rolling(96).mean()
            vol_factor = volatility_1d / (vol_baseline + 1e-8)

            # Adaptive thresholds based on volatility
            adaptive_profit_threshold = profit_threshold * vol_factor
            adaptive_stop_threshold = stop_threshold * vol_factor

            # Clip to reasonable bounds (e.g., 0.5x to 2x base threshold)
            adaptive_profit_threshold = adaptive_profit_threshold.clip(
                lower=profit_threshold * 0.5,
                upper=profit_threshold * 2.0
            )
            adaptive_stop_threshold = adaptive_stop_threshold.clip(
                lower=stop_threshold * 0.5,
                upper=stop_threshold * 2.0
            )

            tprint(f"📊 Adaptive thresholds: Profit {adaptive_profit_threshold.mean():.2%} ± {adaptive_profit_threshold.std():.2%}", "INFO")
            tprint(f"📊 Adaptive thresholds: Stop {adaptive_stop_threshold.mean():.2%} ± {adaptive_stop_threshold.std():.2%}", "INFO")

            # STEP 3: Compute realized returns (continuous) and binary labels with adaptive thresholds
            tprint("💰 Computing realized returns with adaptive thresholds and transaction costs...", "INFO")
            realized_returns, binary_labels = compute_realized_returns(
                market_data,
                primary_signals,
                profit_threshold=adaptive_profit_threshold,
                stop_threshold=adaptive_stop_threshold,
                horizon=horizon,
                transaction_cost=transaction_cost,
                min_event_spacing=min_event_spacing
            )

            # Statistics
            labeled_mask = ~binary_labels.isna()
            n_labeled = labeled_mask.sum()
            n_positive = (binary_labels == 1.0).sum()
            n_negative = (binary_labels == 0.0).sum()

            if n_labeled > 0:
                mean_return = realized_returns[labeled_mask].mean()
                median_return = realized_returns[labeled_mask].median()
                win_rate = n_positive / n_labeled

                tprint(f"📊 Events: {n_labeled} total ({n_positive} wins, {n_negative} losses)", "INFO")
                tprint(f"📈 Win rate: {win_rate:.1%}, Mean return: {mean_return:.2%}, Median: {median_return:.2%}", "INFO")
            else:
                tprint("⚠️ Warning: No labeled events found", "WARNING")
                mean_return = 0.0
                median_return = 0.0
                win_rate = 0.0

            # STEP 4: Apply Kalman smoothing to binary labels
            tprint("📈 Applying Kalman smoothing to binary labels...", "INFO")
            smoothed_labels, label_uncertainty = kalman_smooth_labels(
                binary_labels,
                Q=1e-4,
                R=0.01,
                volatility=volatility_1d
            )
            tprint(f"📊 Smoothed labels: Mean={smoothed_labels[labeled_mask].mean():.3f}, Std={smoothed_labels[labeled_mask].std():.3f}", "INFO")

            # STEP 5: Create meta-features with Kalman filtering
            tprint("🔧 Creating meta-features with Kalman filtering...", "INFO")
            meta_features = create_meta_features(
                market_data,
                primary_signals,
                volume_available,
                include_raw_signals=False,  # CRITICAL: avoid circular behavior
                use_kalman=True  # Enable Kalman filtering
            )

            # STEP 5.5: Apply feature selection to remove low-variance and highly correlated features
            tprint("🔍 Applying feature selection (variance + correlation pruning)...", "INFO")

            # Use correlation threshold of 0.98 for tree models (LGBM, RF)
            # For pure linear models, you might use 0.95
            meta_features_selected, removed_features = apply_feature_selection(
                X=meta_features,
                y=binary_labels,
                variance_threshold=1e-5,
                correlation_threshold=0.98,
                verbose=True
            )

            tprint(f"✅ Feature selection complete: {len(meta_features_selected.columns)} features selected", "SUCCESS")

            # STEP 6: Train ensemble meta-models with K-fold cross-fitting
            tprint("🎓 Training ensemble meta-models (LGBM + LogReg + RF) with purged K-fold CV...", "INFO")

            # Train ensemble and get OOF predictions using SELECTED features
            trained_models, oof_predictions_df = train_ensemble_with_kfold(
                X=meta_features_selected,
                y=binary_labels,
                horizon=horizon,
                n_splits=5,
                verbose=True
            )

            # STEP 7: Add signal disagreement / uncertainty features
            tprint("🔧 Adding signal uncertainty features (disagreement, agreement count, entropy)...", "INFO")
            meta_features_enhanced = add_signal_disagreement(
                oof_predictions=oof_predictions_df,
                meta_features=meta_features_selected  # Use selected features
            )

            # STEP 8: Calibrate ensemble with Platt + isotonic regression
            tprint("📈 Calibrating ensemble (Platt per model + isotonic on blend with entropy/volatility)...", "INFO")

            platt_calibrators, iso_regressor = calibrate_ensemble(
                oof_predictions=oof_predictions_df,
                y_true=binary_labels,
                realized_returns=realized_returns,
                meta_features=meta_features_enhanced,
                method='platt_isotonic',
                include_context=True  # Include entropy/volatility
            )

            # STEP 9: Generate final ensemble predictions
            tprint("🎯 Generating final ensemble predictions...", "INFO")

            # Average OOF predictions after Platt calibration (for final probabilities)
            if platt_calibrators:
                # Apply Platt calibration to OOF predictions
                calibrated_oof = pd.DataFrame(index=oof_predictions_df.index)
                for model_name in oof_predictions_df.columns:
                    if model_name in platt_calibrators:
                        try:
                            oof_vals = oof_predictions_df[model_name].fillna(0.5).values.reshape(-1, 1)
                            calibrated_oof[model_name] = platt_calibrators[model_name].predict_proba(oof_vals)[:, 1]
                        except:
                            calibrated_oof[model_name] = oof_predictions_df[model_name]
                    else:
                        calibrated_oof[model_name] = oof_predictions_df[model_name]

                ensemble_probs = calibrated_oof.mean(axis=1).values
            else:
                ensemble_probs = oof_predictions_df.mean(axis=1).values

            probabilities = ensemble_probs

            # Compute CV metrics for reporting (ENHANCED with Precision@K, Brier, ECE)
            tprint("📊 Computing enhanced metrics (Precision@K, Brier score, ECE)...", "INFO")

            cv_results = []
            oof_mask = ~binary_labels.isna()
            for col in oof_predictions_df.columns:
                oof_mask &= ~oof_predictions_df[col].isna()

            if oof_mask.sum() > 0:
                y_oof = binary_labels[oof_mask]
                for model_name in oof_predictions_df.columns:
                    try:
                        y_pred_proba = oof_predictions_df[model_name][oof_mask].values

                        # Compute all enhanced metrics
                        model_metrics = compute_all_metrics(
                            y_true=binary_labels,
                            y_pred_proba=oof_predictions_df[model_name].values,
                            probabilities_valid=y_pred_proba,
                            k_values=[10, 20, 50, 100]
                        )

                        model_metrics['model'] = model_name

                        cv_results.append(model_metrics)

                        # Print summary
                        tprint(f"  📊 {model_name}:", "INFO")
                        tprint(f"      AUC={model_metrics.get('auc', 0):.3f}, "
                               f"Brier={model_metrics.get('brier_score', 0):.3f}, "
                               f"ECE={model_metrics.get('ece', 0):.3f}", "INFO")
                        tprint(f"      Precision@10={model_metrics.get('precision@10', 0):.3f}, "
                               f"Precision@50={model_metrics.get('precision@50', 0):.3f}", "INFO")

                    except Exception as e:
                        tprint(f"  ⚠️ Could not compute metrics for {model_name}: {e}", "WARNING")

                # Compute ensemble metrics
                try:
                    ensemble_probs_oof = oof_predictions_df[oof_mask].mean(axis=1).values
                    ensemble_metrics = compute_all_metrics(
                        y_true=binary_labels,
                        y_pred_proba=ensemble_probs,
                        probabilities_valid=ensemble_probs_oof,
                        k_values=[10, 20, 50, 100]
                    )
                    ensemble_metrics['model'] = 'ensemble'
                    cv_results.append(ensemble_metrics)

                    tprint(f"  📊 ENSEMBLE:", "INFO")
                    tprint(f"      AUC={ensemble_metrics.get('auc', 0):.3f}, "
                           f"Brier={ensemble_metrics.get('brier_score', 0):.3f}, "
                           f"ECE={ensemble_metrics.get('ece', 0):.3f}", "INFO")
                    tprint(f"      Precision@10={ensemble_metrics.get('precision@10', 0):.3f}, "
                           f"Precision@50={ensemble_metrics.get('precision@50', 0):.3f}", "INFO")

                except Exception as e:
                    tprint(f"  ⚠️ Could not compute ensemble metrics: {e}", "WARNING")

            # STEP 10: Train final models on full dataset (for deployment)
            tprint("🎓 Training final ensemble models on full dataset...", "INFO")

            full_mask = ~binary_labels.isna()
            X_full = meta_features_enhanced[full_mask].fillna(0)
            y_full = binary_labels[full_mask]

            final_models = create_base_models({})
            for model_name, model in final_models.items():
                try:
                    model.fit(X_full, y_full)
                    tprint(f"  ✓ Trained final {model_name}", "INFO")
                except Exception as e:
                    tprint(f"  ❌ Failed to train final {model_name}: {e}", "ERROR")

            # Use first final model for feature importance reporting (RF preferred)
            final_model = final_models.get('rf', list(final_models.values())[0])

            # STEP 11: Translate to targets using isotonic regression
            tprint("🔄 Translating probabilities to economic targets...", "INFO")

            if iso_regressor is not None:
                target_long, target_short = translate_to_targets_with_isotonic(
                    realized_returns,
                    probabilities,
                    primary_signals,
                    iso_regressor
                )
            else:
                # Fallback: simple threshold-based approach
                tprint("⚠️ Using fallback threshold-based translation", "WARNING")
                target_long = pd.Series(0.0, index=market_data.index)
                target_short = pd.Series(0.0, index=market_data.index)

                threshold = 0.6
                for i in range(len(market_data)):
                    if probabilities[i] >= threshold:
                        if primary_signals['consensus'].iloc[i] > 0:
                            target_long.iloc[i] = probabilities[i] - threshold
                        elif primary_signals['consensus'].iloc[i] < 0:
                            target_short.iloc[i] = probabilities[i] - threshold

            # STEP 12: Create output DataFrame with enhanced features
            tprint("📦 Creating output DataFrame...", "INFO")
            labeled_data = market_data.copy()

            # Add log returns and volatility (for diagnostics)
            labeled_data['log_ret'] = log_ret
            labeled_data['volatility_1d'] = volatility_1d

            # Add labeling results
            labeled_data['realized_return'] = realized_returns
            labeled_data['binary_label'] = binary_labels
            labeled_data['smoothed_label'] = smoothed_labels
            labeled_data['label_uncertainty'] = label_uncertainty
            labeled_data['meta_probability'] = probabilities

            # Rename targets to "fused" for backward compatibility with subsequent steps
            labeled_data['fused_target_long'] = target_long
            labeled_data['fused_target_short'] = target_short

            # Add primary signal for reference
            labeled_data['primary_signal'] = primary_signals['consensus']

            # Add adaptive thresholds for transparency
            labeled_data['adaptive_profit_threshold'] = adaptive_profit_threshold
            labeled_data['adaptive_stop_threshold'] = adaptive_stop_threshold

            # Save labeled data
            tprint("💾 Saving labeled data...", "INFO")

            labeled_data_path = self._save_artifact(
                data=labeled_data,
                artifact_name=f"{config['symbol']}_{config['timeframe']}_meta_labeled_data_v2",
                artifact_type="data",
                metadata={
                    'symbol': config['symbol'],
                    'exchange': config['exchange'],
                    'timeframe': config['timeframe'],
                    'profit_threshold': profit_threshold,
                    'stop_threshold': stop_threshold,
                    'horizon': horizon,
                    'transaction_cost': transaction_cost,
                    'n_samples': len(labeled_data),
                    'n_labeled': int(n_labeled),
                    'n_positive': int(n_positive),
                    'win_rate': float(win_rate),
                    'mean_return': float(mean_return),
                    'median_return': float(median_return)
                }
            )

            # STEP 13: Generate comprehensive diagnostics report
            tprint("📊 Generating diagnostics report...", "INFO")

            # Create outcomes directory if it doesn't exist
            outcomes_dir = Path("outcomes")
            outcomes_dir.mkdir(exist_ok=True)

            try:
                diagnostics_path = generate_diagnostics_report(
                    labeled_data=labeled_data,
                    meta_features=meta_features_enhanced,  # Use enhanced features with disagreement
                    binary_labels=binary_labels,
                    realized_returns=realized_returns,
                    smoothed_labels=smoothed_labels,
                    probabilities=probabilities,
                    final_model=final_model,
                    config=config,
                    output_dir=outcomes_dir
                )
                tprint(f"✅ Diagnostics report saved: {diagnostics_path}", "SUCCESS")
            except Exception as e:
                tprint(f"⚠️ Warning: Could not generate diagnostics report: {e}", "WARNING")
                diagnostics_path = None

            # Calculate aggregate metrics from CV results (ENHANCED)
            avg_auc = np.mean([r.get('auc', 0.5) for r in cv_results if 'auc' in r]) if cv_results else 0.5
            avg_precision = np.mean([r.get('precision', 0) for r in cv_results if 'precision' in r]) if cv_results else 0.0
            avg_brier = np.mean([r.get('brier_score', 0) for r in cv_results if 'brier_score' in r]) if cv_results else 0.0
            avg_ece = np.mean([r.get('ece', 0) for r in cv_results if 'ece' in r]) if cv_results else 0.0
            avg_precision_at_10 = np.mean([r.get('precision@10', 0) for r in cv_results if 'precision@10' in r]) if cv_results else 0.0
            avg_precision_at_50 = np.mean([r.get('precision@50', 0) for r in cv_results if 'precision@50' in r]) if cv_results else 0.0

            # Feature importances (use enhanced features)
            try:
                feature_importances = dict(zip(meta_features_enhanced.columns, final_model.feature_importances_))
                top_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:10]

                tprint("🎯 Top 10 features:", "INFO")
                for feat, imp in top_features:
                    tprint(f"  {feat}: {imp:.4f}", "INFO")
            except Exception as e:
                tprint(f"⚠️ Could not compute feature importances: {e}", "WARNING")
                top_features = []

            elapsed_time = (datetime.now() - start_time).total_seconds()

            result = {
                'success': True,
                'artifacts': {
                    'labeled_data_path': labeled_data_path,
                    'diagnostics_report_path': diagnostics_path
                },
                'metrics': {
                    'n_samples': len(labeled_data),
                    'n_labeled': int(n_labeled),
                    'n_positive': int(n_positive),
                    'win_rate': float(win_rate),
                    'mean_return': float(mean_return),
                    'median_return': float(median_return),
                    'cv_mean_auc': float(avg_auc),
                    'cv_mean_precision': float(avg_precision),
                    'cv_mean_brier_score': float(avg_brier),
                    'cv_mean_ece': float(avg_ece),
                    'cv_mean_precision@10': float(avg_precision_at_10),
                    'cv_mean_precision@50': float(avg_precision_at_50),
                    'n_cv_folds': len(cv_results),
                    'elapsed_seconds': elapsed_time,
                    'n_features_before_selection': len(meta_features.columns),
                    'n_features_after_selection': len(meta_features_selected.columns),
                    'n_features_removed': len(removed_features),
                    'top_features': dict(top_features),
                    'config': {
                        'profit_threshold': profit_threshold,
                        'stop_threshold': stop_threshold,
                        'horizon': horizon,
                        'transaction_cost': transaction_cost,
                        'min_event_spacing': min_event_spacing,
                        'use_kalman_filtering': True,
                        'use_adaptive_thresholds': True,
                        'use_kalman_label_smoothing': True,
                        'use_ensemble': True,
                        'ensemble_models': ['lgbm', 'logreg', 'rf'],
                        'use_platt_calibration': True,
                        'use_isotonic_calibration': True,
                        'include_signal_disagreement': True
                    },
                    'enhancements': {
                        'kalman_filtering': True,
                        'adaptive_thresholds': True,
                        'volatility_regimes': True,
                        'smoothed_labels': True,
                        'ensemble_models': True,
                        'platt_calibration': True,
                        'isotonic_calibration': True,
                        'signal_disagreement': True,
                        'kfold_cross_fitting': True,
                        'volatility_normalized_features': True,
                        'short_term_autocorrelation': True,
                        'signal_uncertainty_features': True,
                        'feature_interactions': True,
                        'signal_persistence': True,
                        'feature_selection': True,
                        'enhanced_metrics': ['precision@k', 'brier_score', 'ece', 'hit_rate']
                    }
                },
                'cv_results': cv_results
            }

            tprint(f"✅ Enhanced meta-labeling completed in {elapsed_time:.1f}s", "SUCCESS")
            tprint(f"📊 Performance:", "SUCCESS")
            tprint(f"   - AUC={avg_auc:.3f}, Brier={avg_brier:.3f}, ECE={avg_ece:.3f}", "SUCCESS")
            tprint(f"   - Precision@10={avg_precision_at_10:.3f}, Precision@50={avg_precision_at_50:.3f}", "SUCCESS")
            tprint(f"   - Win Rate={win_rate:.1%}, Mean Return={mean_return:.2%}", "SUCCESS")
            tprint(f"   - Features: {len(meta_features_selected.columns)} selected (removed {len(removed_features)})", "SUCCESS")

            return result

        except Exception as e:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Meta-labeling failed: {str(e)}"
            tprint(f"❌ {error_msg}", "ERROR")
            self.logger.exception("Meta-labeling error")

            return {
                'success': False,
                'error': error_msg,
                'elapsed_seconds': elapsed_time
            }
