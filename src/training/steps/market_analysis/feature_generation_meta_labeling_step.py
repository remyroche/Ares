"""
Feature Generation Meta-Labeling Step (Production Version).

Major enhancements:
1. Ensemble models (LGBM + LogisticRegression + RF) with soft voting
2. K-fold cross-fitting to prevent leakage
3. Volatility-adaptive labeling with Kalman filtering
4. Robust feature engineering with RobustScaler
5. Vectorized operations for performance
6. Comprehensive diagnostics and calibration
7. Production TPSL parameters (1% profit, 0.5% stop, 0.15% fee)

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
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    brier_score_loss,
    average_precision_score,
    mutual_info_score,
)
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
R_MULTIPLE_POS_THRESHOLD = 0.7
R_MULTIPLE_NEG_THRESHOLD = -0.25
ECON_MIN_RETURN_MULTIPLE = 1.25
TARGET_POWER = 1.5


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
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
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

    exit_reasons = pd.Series(index=df.index, dtype=object)
    exit_reasons[:] = pd.NA

    event_durations = pd.Series(index=df.index, dtype=float)
    event_durations[:] = np.nan

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

    i = 0
    n = len(df)
    max_start = n - horizon

    while i < max_start:
        signal = consensus_signals[i]

        # Only create labels where we have a signal
        if signal == 0:
            i += 1
            continue

        # Handle overlapping events: skip ahead if too close to previous signal
        if (i - last_event_idx) < min_event_spacing:
            i = last_event_idx + min_event_spacing
            continue

        # Edge window handling: skip events too close to end of available data
        if i + horizon >= n:
            break

        entry_price = close_prices[i]
        exit_price = None
        exit_reason = None
        event_end_idx = i

        # Get adaptive thresholds for this event
        profit_thr = profit_thresholds[i]
        stop_thr = stop_thresholds[i]

        # Look ahead up to horizon bars
        for j in range(1, horizon + 1):
            idx = i + j
            if idx >= n:
                break

            future_price = close_prices[idx]

            if signal > 0:  # Long signal
                pnl = (future_price - entry_price) / entry_price

                # Hit profit target
                if pnl >= profit_thr:
                    exit_price = future_price
                    exit_reason = 'profit'
                    event_end_idx = idx
                    break
                # Hit stop loss
                elif pnl <= -stop_thr:
                    exit_price = future_price
                    exit_reason = 'stop'
                    event_end_idx = idx
                    break

            elif signal < 0:  # Short signal
                pnl = (entry_price - future_price) / entry_price

                # Hit profit target
                if pnl >= profit_thr:
                    exit_price = future_price
                    exit_reason = 'profit'
                    event_end_idx = idx
                    break
                # Hit stop loss
                elif pnl <= -stop_thr:
                    exit_price = future_price
                    exit_reason = 'stop'
                    event_end_idx = idx
                    break

        # If no exit, use end-of-horizon price (timeout)
        if exit_price is None:
            event_end_idx = min(i + horizon, n - 1)
            exit_price = close_prices[event_end_idx]
            exit_reason = 'timeout'

        # Compute realized return accounting for transaction costs
        if signal > 0:  # Long
            gross_return = (exit_price - entry_price) / entry_price
        else:  # Short
            gross_return = (entry_price - exit_price) / entry_price

        net_return = gross_return - transaction_cost

        event_length = event_end_idx - i

        realized_returns.iloc[i] = net_return
        exit_reasons.iloc[i] = exit_reason
        event_durations.iloc[i] = float(event_length)

        econ_min_return = ECON_MIN_RETURN_MULTIPLE * transaction_cost

        if abs(net_return) < econ_min_return:
            binary_labels.iloc[i] = np.nan
        else:
            risk_unit = stop_thr if stop_thr > 0 else profit_thr
            if risk_unit <= 0:
                r_multiple = 0.0
            else:
                r_multiple = net_return / risk_unit

            if r_multiple >= R_MULTIPLE_POS_THRESHOLD:
                binary_labels.iloc[i] = 1.0
            elif r_multiple <= R_MULTIPLE_NEG_THRESHOLD:
                binary_labels.iloc[i] = 0.0
            else:
                binary_labels.iloc[i] = np.nan

        last_event_idx = i  # Update last event position
        i += 1

    return realized_returns, binary_labels, exit_reasons, event_durations


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

    # Create regime labels using quantiles estimated on early history to reduce lookahead
    try:
        vol_non_null = vol_for_regime.dropna()
        if len(vol_non_null) >= 10:
            split_idx = max(1, int(len(vol_non_null) * 0.7))
            vol_train = vol_non_null.iloc[:split_idx]
        else:
            vol_train = vol_non_null

        if len(vol_train) >= 3:
            q1, q2 = vol_train.quantile([1 / 3, 2 / 3])
            bins = [-np.inf, q1, q2, np.inf]
            labels = ['low', 'medium', 'high']
            regime_full = pd.cut(vol_for_regime, bins=bins, labels=labels)
            features['volatility_regime'] = regime_full

            regime_dummies = pd.get_dummies(features['volatility_regime'], prefix='vol_regime', drop_first=True)
            features = features.join(regime_dummies)
        else:
            raise ValueError("Not enough non-null volatility samples for regime estimation")
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

    # ===== VOLATILITY-NORMALIZED FEATURES =====

    # Normalize momentum and MA distance by current volatility
    vol_1h = features['volatility_1h'].replace(0, np.nan)  # Avoid division by zero

    if use_kalman:
        features['momentum_per_vol'] = features['momentum_kalman'] / (vol_1h + 1e-8)
        features['ma_distance_per_vol'] = features['ma_distance_kalman'] / (df['close'] * vol_1h + 1e-8)
    else:
        features['momentum_per_vol'] = features['momentum'] / (vol_1h + 1e-8)
        features['ma_distance_per_vol'] = features['ma_distance'] / (df['close'] * vol_1h + 1e-8)

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

        # Additional volume/flow proxies
        volume_long_mean = df['volume'].rolling(96).mean()
        features['volume_spike'] = df['volume'] / (volume_long_mean + 1e-8)
        features['volume_spike_ema'] = features['volume_spike'].ewm(span=20).mean()

        signed_volume = np.sign(returns.fillna(0.0)) * df['volume']
        features['signed_volume_ema'] = signed_volume.ewm(span=20).mean()
    else:
        features['volume_ratio'] = 1.0
        features['volume_trend'] = 1.0
        features['vol_price_corr'] = 0.0
        features['volume_zscore'] = 0.0
        features['volume_spike'] = 1.0
        features['volume_spike_ema'] = 1.0
        features['signed_volume_ema'] = 0.0

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

    # ===== TIME-BASED FEATURES =====

    if isinstance(df.index, pd.DatetimeIndex):
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
    else:
        features['hour'] = 0
        features['day_of_week'] = 0

    # Volatility / trend interaction features
    if 'kalman_trend' in features.columns and 'vol_ratio' in features.columns:
        features['kalman_trend_x_vol_ratio'] = features['kalman_trend'] * features['vol_ratio']
    if 'sma_slope' in features.columns and 'vol_ratio' in features.columns:
        features['sma_slope_x_vol_ratio'] = features['sma_slope'] * features['vol_ratio']
    if 'price_vs_sma20' in features.columns and 'vol_ratio' in features.columns:
        features['price_vs_sma20_x_vol_ratio'] = features['price_vs_sma20'] * features['vol_ratio']
    if 'range_position' in features.columns and 'vol_ratio' in features.columns:
        features['range_position_x_vol_ratio'] = features['range_position'] * features['vol_ratio']

    # Signal-aware diagnostic features (without leaking raw rule logic)
    if 'consensus' in signals.columns:
        signal_consensus = signals['consensus']
        signal_active = (signal_consensus != 0).astype(int)
        features['signal_active'] = signal_active

        idx = np.arange(len(df))
        last_signal_idx = np.where(signal_active == 1, idx, np.nan)
        last_signal_idx_series = pd.Series(last_signal_idx, index=df.index).ffill()

        signal_age = idx - last_signal_idx_series.values
        signal_age[last_signal_idx_series.isna().values] = np.nan
        features['bars_since_last_signal'] = signal_age

        features['signal_density_50'] = signal_consensus.abs().rolling(50).sum()
    else:
        features['signal_active'] = 0
        features['bars_since_last_signal'] = np.nan
        features['signal_density_50'] = 0.0

    # ===== RAW SIGNALS (OPTIONAL, FOR DIAGNOSTICS) =====

    if include_raw_signals:
        tprint("⚠️ WARNING: Including raw signal features - may cause circular behavior", "WARNING")
        features['signal_strength'] = signals[['rsi', 'ma', 'mom']].abs().sum(axis=1)
        features['signal_consensus'] = signals['consensus'].abs()

    return features


def prepare_feature_matrix(features: pd.DataFrame) -> pd.DataFrame:
    numeric = features.select_dtypes(include=[np.number])
    if numeric.empty:
        return numeric
    numeric = numeric.astype(np.float32, copy=False)
    return numeric


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
    iso_regressor: IsotonicRegression,
    cost_threshold: float = DEFAULT_TRANSACTION_COST,
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
    expected_returns = np.maximum(0.0, expected_returns)

    cost_thr = float(cost_threshold)
    if cost_thr > 0.0:
        below_cost_mask = expected_returns < cost_thr
        expected_returns[below_cost_mask] = 0.0
        above_cost_mask = expected_returns >= cost_thr
        if np.any(above_cost_mask):
            ratio = expected_returns[above_cost_mask] / cost_thr
            ratio = np.maximum(1.0, ratio)
            expected_returns[above_cost_mask] = cost_thr * np.power(ratio, TARGET_POWER)

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

    # Represent probabilities as Series aligned with index for richer diagnostics
    prob_series = pd.Series(probabilities, index=labeled_data.index)

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

    # Compute correlations (numeric features only to avoid categorical fill issues)
    features_clean = meta_features[labeled_mask]
    features_clean = features_clean.select_dtypes(include=[np.number]).fillna(0)
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
    if len(returns_positive) > 0:
        report_lines.append(f"- **Mean return:** {returns_positive.mean():.2%}")
        report_lines.append(f"- **Median return:** {returns_positive.median():.2%}")
        report_lines.append(f"- **Std return:** {returns_positive.std():.2%}")
        pct_pos_pos = (returns_positive > 0).sum() / len(returns_positive)
        report_lines.append(f"- **% Actually positive:** {pct_pos_pos:.1%}")
    else:
        report_lines.append("- **Mean return:** N/A")
        report_lines.append("- **Median return:** N/A")
        report_lines.append("- **Std return:** N/A")
        report_lines.append("- **% Actually positive:** N/A")

    report_lines.append("\n### Label = 0 (Unprofitable Signals):\n")
    report_lines.append(f"- **Count:** {len(returns_negative)}")
    if len(returns_negative) > 0:
        report_lines.append(f"- **Mean return:** {returns_negative.mean():.2%}")
        report_lines.append(f"- **Median return:** {returns_negative.median():.2%}")
        report_lines.append(f"- **Std return:** {returns_negative.std():.2%}")
        pct_pos_neg = (returns_negative > 0).sum() / len(returns_negative)
        report_lines.append(f"- **% Actually positive:** {pct_pos_neg:.1%}")
    else:
        report_lines.append("- **Mean return:** N/A")
        report_lines.append("- **Median return:** N/A")
        report_lines.append("- **Std return:** N/A")
        report_lines.append("- **% Actually positive:** N/A")

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

    # Cost-aware economic summary
    try:
        tx_cost = float(config.get('transaction_cost', DEFAULT_TRANSACTION_COST))
    except Exception:
        tx_cost = float(DEFAULT_TRANSACTION_COST)

    if len(returns_labeled) > 0:
        unconditional_mean = float(returns_labeled.mean())
        frac_small = float((returns_labeled.abs() < tx_cost).mean())
    else:
        unconditional_mean = 0.0
        frac_small = 0.0

    if len(returns_positive) > 0:
        mean_pos_ret = float(returns_positive.mean())
    else:
        mean_pos_ret = 0.0

    report_lines.append("\n### Cost-Aware Event Quality Summary\n")
    report_lines.append(f"- **Transaction cost (per event, approx):** {tx_cost:.3%}")
    report_lines.append(f"- **Unconditional mean event return:** {unconditional_mean:.2%}")
    report_lines.append(f"- **Mean return (label=1) minus cost:** {(mean_pos_ret - tx_cost):.2%}")
    report_lines.append(f"- **Fraction of labeled events with |return| < cost:** {frac_small:.1%}")

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
        prob_clean = prob_series[labeled_mask]
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

        # Monotonicity and slope diagnostics: probability bins vs realized returns
        try:
            mean_ret_by_bin = []
            mean_prob_by_bin = []
            unique_bins = sorted([b for b in prob_bins.dropna().unique()])
            for b in unique_bins:
                mask_bin = (prob_bins == b)
                if mask_bin.sum() == 0:
                    continue
                mean_prob_bin = prob_clean[mask_bin].mean()
                mean_ret_bin = returns_labeled[mask_bin].mean()
                mean_prob_by_bin.append(float(mean_prob_bin))
                mean_ret_by_bin.append(float(mean_ret_bin))

            if len(mean_ret_by_bin) >= 2:
                violations = 0
                for i in range(len(mean_ret_by_bin) - 1):
                    if mean_ret_by_bin[i + 1] < mean_ret_by_bin[i] - 1e-8:
                        violations += 1
                denom = max(len(mean_ret_by_bin) - 1, 1)
                violation_frac = violations / denom

                # Simple slope in upper half of probability spectrum
                mid = len(mean_prob_by_bin) // 2
                x = np.array(mean_prob_by_bin[mid:], dtype=float)
                y = np.array(mean_ret_by_bin[mid:], dtype=float)
                if x.size >= 2 and np.var(x) > 0:
                    slope_high = float(np.cov(x, y)[0, 1] / (np.var(x) + 1e-8))
                else:
                    slope_high = float('nan')

                report_lines.append("\n### Monotonicity and Slope Checks (Prob → Realized Return)\n")
                report_lines.append(f"- **Monotonicity violations (adjacent bins):** {violations} / {denom}")
                report_lines.append(f"- **Approx. slope in high-probability region:** {slope_high:.6f}")
        except Exception as e_mono:
            report_lines.append(f"\n⚠️ Could not compute monotonicity diagnostics: {e_mono}")

    except Exception as e:
        report_lines.append(f"\n⚠️ Could not compute calibration: {e}")

    # Probability distribution
    report_lines.append("\n### Probability Distribution:\n")
    report_lines.append(f"- **Mean probability:** {prob_series.mean():.3f}")
    report_lines.append(f"- **Median probability:** {prob_series.median():.3f}")
    report_lines.append(f"- **Std probability:** {prob_series.std():.3f}")

    # Check for collapsed predictions
    if prob_series.std() < 0.05:
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

    # ===== 9. EVENT MECHANICS: EXIT REASONS, DURATIONS, R-MULTIPLES =====
    report_lines.append("\n## 9. Event Mechanics and R-Multiples\n")
    report_lines.append(
        "These diagnostics describe how trades exit (profit, stop, or timeout), "
        "how long they stay open, and how large the realized return is relative "
        "to the configured stop-loss (R-multiple). They help verify that the "
        "triple-barrier / TPSL configuration produces economically meaningful events.\n"
    )

    try:
        exit_reasons_series = labeled_data.get('exit_reason')
        durations_series = labeled_data.get('event_duration_bars')
        stop_threshold_series = labeled_data.get('adaptive_stop_threshold')

        if exit_reasons_series is not None:
            exit_labeled = exit_reasons_series[labeled_mask].dropna()
            total_events = len(exit_labeled)
            if total_events > 0:
                value_counts = exit_labeled.value_counts(normalize=True)
                report_lines.append("\n### Exit Reason Mix (Labeled Events)\n")
                for reason, frac in value_counts.items():
                    report_lines.append(f"- **{reason}:** {frac:.1%}")

        if durations_series is not None:
            dur_clean = durations_series[labeled_mask].dropna()
            if len(dur_clean) > 0:
                report_lines.append("\n### Event Duration Distribution (Bars)\n")
                report_lines.append(f"- **Mean duration:** {dur_clean.mean():.2f}")
                report_lines.append(f"- **Median duration:** {dur_clean.median():.2f}")
                report_lines.append(f"- **90th percentile:** {dur_clean.quantile(0.9):.2f}")

        if stop_threshold_series is not None:
            stop_labeled = stop_threshold_series[labeled_mask]
            # Avoid division by zero
            r_multiple = returns_labeled / (stop_labeled.replace(0, np.nan) + 1e-8)
            r_multiple_pos = r_multiple[labels_clean == 1]
            r_multiple_neg = r_multiple[labels_clean == 0]

            report_lines.append("\n### R-Multiple Distribution (Return / Stop Threshold)\n")
            if len(r_multiple_pos) > 0:
                report_lines.append(f"- **R-multiple (label=1) mean:** {r_multiple_pos.mean():.2f}")
                report_lines.append(f"- **R-multiple (label=1) median:** {r_multiple_pos.median():.2f}")
            if len(r_multiple_neg) > 0:
                report_lines.append(f"- **R-multiple (label=0) mean:** {r_multiple_neg.mean():.2f}")
                report_lines.append(f"- **R-multiple (label=0) median:** {r_multiple_neg.median():.2f}")

    except Exception as e:
        report_lines.append(f"\n⚠️ Could not compute event mechanics diagnostics: {e}")

    # ===== 10. REGIME AND TIME-CONDITIONAL LABEL CHECKS =====
    report_lines.append("\n## 10. Regime and Time-Conditional Label Checks\n")
    report_lines.append(
        "These metrics show how label base-rates and realized returns change "
        "across volatility regimes, trend states, and time-of-day/weekday. "
        "Large differences indicate that learnability and edge are "
        "regime-dependent, which is important for downstream model conditioning.\n"
    )

    try:
        # Volatility regime base-rates
        regime_series = None
        if 'volatility_regime' in meta_features.columns:
            regime_series = meta_features['volatility_regime']
        elif 'vol_regime_high' in meta_features.columns or 'vol_regime_medium' in meta_features.columns:
            # Derive simple labels from dummies
            regime_labels = []
            for idx in meta_features.index:
                if 'vol_regime_high' in meta_features.columns and meta_features.at[idx, 'vol_regime_high'] == 1:
                    regime_labels.append('high')
                elif 'vol_regime_medium' in meta_features.columns and meta_features.at[idx, 'vol_regime_medium'] == 1:
                    regime_labels.append('medium')
                else:
                    regime_labels.append('low')
            regime_series = pd.Series(regime_labels, index=meta_features.index)

        if regime_series is not None:
            regime_labeled = regime_series[labeled_mask]
            report_lines.append("\n### Label Base-Rate by Volatility Regime\n")
            for regime in sorted(regime_labeled.dropna().unique()):
                mask_reg = labeled_mask & (regime_series == regime)
                if mask_reg.sum() < 10:
                    continue
                pos_rate_reg = binary_labels[mask_reg].mean()
                mean_ret_reg = realized_returns[mask_reg].mean()
                report_lines.append(
                    f"- **Regime {regime}:** positive={pos_rate_reg:.1%}, mean_return={mean_ret_reg:.2%}"
                )

        # Trend-conditional checks using price_vs_sma20 if available
        if 'price_vs_sma20' in meta_features.columns:
            trend_measure = meta_features['price_vs_sma20']
            trend_labeled = trend_measure[labeled_mask]
            high_trend = trend_labeled.quantile(0.75)
            low_trend = trend_labeled.quantile(0.25)

            strong_up = labeled_mask & (trend_measure >= high_trend)
            strong_down = labeled_mask & (trend_measure <= low_trend)

            if strong_up.sum() >= 10:
                report_lines.append("\n### Trend-Conditional (Price vs SMA20)\n")
                pos_up = binary_labels[strong_up].mean()
                mean_ret_up = realized_returns[strong_up].mean()
                report_lines.append(
                    f"- **Strong uptrend:** positive={pos_up:.1%}, mean_return={mean_ret_up:.2%}"
                )
            if strong_down.sum() >= 10:
                pos_down = binary_labels[strong_down].mean()
                mean_ret_down = realized_returns[strong_down].mean()
                report_lines.append(
                    f"- **Strong downtrend:** positive={pos_down:.1%}, mean_return={mean_ret_down:.2%}"
                )

        # Time-of-day / weekday conditional
        if 'hour' in meta_features.columns:
            hour_labeled = meta_features['hour'][labeled_mask]
            pos_by_hour = labels_clean.groupby(hour_labeled).mean()
            if len(pos_by_hour) > 0:
                top_hours = pos_by_hour.sort_values(ascending=False).head(3)
                bottom_hours = pos_by_hour.sort_values().head(3)
                report_lines.append("\n### Time-of-Day Positive Rates (Top/Bottom)\n")
                report_lines.append("- **Top hours by positive rate:**")
                for h, v in top_hours.items():
                    report_lines.append(f"  - Hour {int(h)}: {v:.1%}")
                report_lines.append("- **Bottom hours by positive rate:**")
                for h, v in bottom_hours.items():
                    report_lines.append(f"  - Hour {int(h)}: {v:.1%}")

        if 'day_of_week' in meta_features.columns:
            dow_labeled = meta_features['day_of_week'][labeled_mask]
            pos_by_dow = labels_clean.groupby(dow_labeled).mean()
            if len(pos_by_dow) > 0:
                report_lines.append("\n### Day-of-Week Positive Rates\n")
                for d, v in pos_by_dow.items():
                    report_lines.append(f"- Day {int(d)}: {v:.1%}")

    except Exception as e:
        report_lines.append(f"\n⚠️ Could not compute regime/time-conditional diagnostics: {e}")

    # ===== 11. LABEL–RETURN SEPARATION AND INFORMATION CONTENT =====
    report_lines.append("\n## 11. Label–Return Separation and Information Content\n")
    report_lines.append(
        "This section quantifies how well the labels separate profitable from "
        "unprofitable events (effect size) and how much information they carry "
        "about the sign of future returns compared to a random baseline "
        "(mutual information and permutation tests).\n"
    )

    try:
        if len(returns_positive) > 0 and len(returns_negative) > 0:
            mean_diff = returns_positive.mean() - returns_negative.mean()
            var_pos = returns_positive.var()
            var_neg = returns_negative.var()
            pooled_std = np.sqrt(((len(returns_positive) - 1) * var_pos + (len(returns_negative) - 1) * var_neg) /
                                 max(len(returns_positive) + len(returns_negative) - 2, 1))
            effect_size = mean_diff / pooled_std if pooled_std > 0 else np.nan

            report_lines.append("\n### Separation Metrics\n")
            report_lines.append(f"- **Mean return difference (label=1 - label=0):** {mean_diff:.2%}")
            report_lines.append(f"- **Cohen's d effect size:** {effect_size:.3f}")

        # Information content vs trivial baseline using mutual information
        label_array = labels_clean.values.astype(int)
        # Binary sign of realized return
        ret_sign = (returns_labeled > 0).astype(int).values
        if len(label_array) > 0:
            mi_observed = mutual_info_score(label_array, ret_sign)
            # Baseline: shuffle returns relative to labels multiple times
            rng = np.random.default_rng(42)
            mi_baseline_samples = []
            for _ in range(50):
                perm = rng.permutation(len(ret_sign))
                mi_baseline_samples.append(mutual_info_score(label_array, ret_sign[perm]))
            mi_baseline_mean = float(np.mean(mi_baseline_samples))
            report_lines.append("\n### Information Content vs Permutation Baseline\n")
            report_lines.append(f"- **Mutual information (labels vs realized sign):** {mi_observed:.4f}")
            report_lines.append(f"- **Baseline MI (mean over permutations):** {mi_baseline_mean:.4f}")

    except Exception as e:
        report_lines.append(f"\n⚠️ Could not compute separation/information diagnostics: {e}")

    # ===== 12. TARGET AND EXPECTED-RETURN ALIGNMENT =====
    report_lines.append("\n## 12. Target and Expected-Return Alignment\n")
    report_lines.append(
        "Here we check whether the continuous targets produced by the isotonic "
        "mapping are consistent with realized returns: higher targets should "
        "correspond to higher average realized P&L, and most non-zero targets "
        "should exceed transaction costs.\n"
    )

    try:
        # Build unified target magnitude series
        target_long = labeled_data.get('target_long')
        target_short = labeled_data.get('target_short')
        if target_long is not None and target_short is not None:
            target_mag = pd.Series(0.0, index=labeled_data.index)
            long_mask_target = target_long > 0
            short_mask_target = target_short > 0
            target_mag[long_mask_target] = target_long[long_mask_target]
            target_mag[short_mask_target] = target_short[short_mask_target]

            trade_mask = labeled_mask & (target_mag > 0) & ~realized_returns.isna()
            target_trades = target_mag[trade_mask]
            returns_trades = realized_returns[trade_mask]

            if len(target_trades) > 0:
                corr_tr = target_trades.corr(returns_trades)
                mse_tr = float(np.mean((target_trades - returns_trades) ** 2))
                report_lines.append("\n### Target vs Realized Return\n")
                report_lines.append(f"- **Correlation (target, realized):** {corr_tr:.3f}")
                report_lines.append(f"- **MSE (target vs realized):** {mse_tr:.6f}")

                # Decile check
                try:
                    deciles = pd.qcut(target_trades, 10, labels=False, duplicates='drop')
                    mean_target_by_decile = target_trades.groupby(deciles).mean()
                    mean_ret_by_decile = returns_trades.groupby(deciles).mean()
                    report_lines.append("\n### Target/Return by Target Decile\n")
                    for d in mean_target_by_decile.index:
                        report_lines.append(
                            f"- Decile {int(d)}: target={mean_target_by_decile[d]:.4f}, realized={mean_ret_by_decile[d]:.4f}"
                        )
                except Exception as e_dec:
                    report_lines.append(f"\n⚠️ Could not compute decile diagnostics: {e_dec}")

            # Target distribution sanity
            target_nz = target_mag[target_mag > 0]
            if len(target_nz) > 0:
                report_lines.append("\n### Target Distribution Sanity\n")
                report_lines.append(f"- **Non-zero target fraction (all samples):** {len(target_nz) / len(target_mag):.1%}")
                report_lines.append(f"- **Mean non-zero target:** {target_nz.mean():.4f}")
                report_lines.append(f"- **Median non-zero target:** {target_nz.median():.4f}")
                tx_cost = float(config.get('transaction_cost', DEFAULT_TRANSACTION_COST))
                frac_below_cost = (target_nz < tx_cost).mean()
                report_lines.append(f"- **Fraction of targets below transaction cost ({tx_cost:.3%}):** {frac_below_cost:.1%}")

    except Exception as e:
        report_lines.append(f"\n⚠️ Could not compute target/return alignment diagnostics: {e}")

    # ===== 13. COST-AWARE METRICS AND THRESHOLD P&L =====
    report_lines.append("\n## 13. Cost-Aware Metrics and Threshold P&L\n")
    report_lines.append(
        "These diagnostics evaluate the meta-model from a trading perspective: "
        "global AUC/Brier/PR-AUC, and how mean and cumulative returns evolve "
        "as you raise the probability threshold used to filter trades. This "
        "helps choose probability cutoffs that are profitable after costs.\n"
    )

    try:
        # Cost-aware classification metrics
        if len(prob_clean) > 0:
            try:
                brier = brier_score_loss(labels_clean_array, prob_clean)
                ap = average_precision_score(labels_clean_array, prob_clean)
                auc_global = roc_auc_score(labels_clean_array, prob_clean)
                report_lines.append("\n### Cost-Aware Classification Metrics (OOF)\n")
                report_lines.append(f"- **AUC:** {auc_global:.3f}")
                report_lines.append(f"- **Brier score:** {brier:.4f}")
                report_lines.append(f"- **Average precision (PR-AUC):** {ap:.3f}")
            except Exception as e_metrics:
                report_lines.append(f"\n⚠️ Could not compute cost-aware metrics: {e_metrics}")

        # Threshold sweep P&L curves
        report_lines.append("\n### Threshold-Sweep P&L (Using Meta Probability)\n")
        report_lines.append("\n| Threshold | Trades | Mean Return | Cum Return | Sharpe (per trade) |")
        report_lines.append("|----------|--------|-------------|------------|---------------------|")

        # Evaluate a dense grid of thresholds in the operational region [0.50, 0.65]
        thresholds = [round(x, 2) for x in np.arange(0.5, 0.651, 0.01)]
        for thr in thresholds:
            mask_thr = (prob_series >= thr) & labeled_mask & ~realized_returns.isna()
            n_trades_thr = int(mask_thr.sum())
            if n_trades_thr == 0:
                report_lines.append(f"| {thr:.2f} | 0 | N/A | N/A | N/A |")
                continue
            ret_thr = realized_returns[mask_thr]
            mean_ret_thr = ret_thr.mean()
            std_ret_thr = ret_thr.std()
            cum_ret_thr = ret_thr.sum()
            sharpe_thr = (
                mean_ret_thr / (std_ret_thr + 1e-8) * np.sqrt(n_trades_thr)
                if std_ret_thr > 0
                else np.nan
            )
            report_lines.append(
                f"| {thr:.2f} | {n_trades_thr} | {mean_ret_thr:.2%} | {cum_ret_thr:.2%} | {sharpe_thr:.3f} |"
            )

    except Exception as e:
        report_lines.append(f"\n⚠️ Could not compute threshold/P&L diagnostics: {e}")

    # ===== 14. STABILITY AND BOOTSTRAP STATISTICS =====
    report_lines.append("\n## 14. Stability and Bootstrap Statistics\n")
    report_lines.append(
        "Finally, we assess how stable the labeling and model performance are "
        "across time (years and time-series folds) and how sensitive key "
        "metrics such as positive rate and label–return separation are under "
        "bootstrap resampling. Tight confidence intervals indicate robust "
        "signal rather than fragile noise.\n"
    )

    try:
        # Per-year stability (if datetime index)
        if isinstance(labeled_data.index, pd.DatetimeIndex):
            years = labeled_data.index.year
            report_lines.append("\n### Per-Year Label and Return Stability\n")
            for year in sorted(np.unique(years)):
                year_mask = labeled_mask & (years == year)
                if year_mask.sum() < 20:
                    continue
                pos_rate_y = binary_labels[year_mask].mean()
                mean_ret_y = realized_returns[year_mask].mean()
                try:
                    auc_y = roc_auc_score(binary_labels[year_mask], prob_series[year_mask])
                except Exception:
                    auc_y = float('nan')
                report_lines.append(
                    f"- **Year {int(year)}:** positive={pos_rate_y:.1%}, mean_return={mean_ret_y:.2%}, AUC={auc_y:.3f}"
                )

        # Approximate per-fold stability using TimeSeriesSplit on chronology
        try:
            tscv_diag = TimeSeriesSplit(n_splits=5)
            indices = np.arange(len(labeled_data))
            report_lines.append("\n### Time-Series Fold Stability (Approximate)\n")
            for fold_idx, (_, test_idx) in enumerate(tscv_diag.split(indices)):
                fold_mask = pd.Series(False, index=labeled_data.index)
                fold_mask.iloc[test_idx] = True
                fold_mask &= labeled_mask
                if fold_mask.sum() < 20:
                    continue
                pos_rate_f = binary_labels[fold_mask].mean()
                mean_ret_f = realized_returns[fold_mask].mean()
                try:
                    auc_f = roc_auc_score(binary_labels[fold_mask], prob_series[fold_mask])
                except Exception:
                    auc_f = float('nan')
                report_lines.append(
                    f"- **Fold {fold_idx + 1}:** positive={pos_rate_f:.1%}, mean_return={mean_ret_f:.2%}, AUC={auc_f:.3f}"
                )
        except Exception as e_fold:
            report_lines.append(f"\n⚠️ Could not compute fold stability: {e_fold}")

        # Bootstrap label statistics
        report_lines.append("\n### Bootstrap Label Statistics (Labeled Events)\n")
        try:
            n_boot = 100
            idx = np.where(labeled_mask.values)[0]
            if len(idx) >= 10:
                rng = np.random.default_rng(123)
                boot_pos_rates = []
                boot_mean_diffs = []
                for _ in range(n_boot):
                    sample_idx = rng.choice(idx, size=len(idx), replace=True)
                    y_boot = binary_labels.iloc[sample_idx]
                    r_boot = realized_returns.iloc[sample_idx]
                    pos_rate_b = y_boot.mean()
                    boot_pos_rates.append(pos_rate_b)

                    r_pos_b = r_boot[y_boot == 1]
                    r_neg_b = r_boot[y_boot == 0]
                    if len(r_pos_b) > 0 and len(r_neg_b) > 0:
                        boot_mean_diffs.append(r_pos_b.mean() - r_neg_b.mean())

                def _ci(arr: List[float], lower: float = 2.5, upper: float = 97.5) -> Tuple[float, float]:
                    arr_np = np.array(arr, dtype=float)
                    arr_np = arr_np[np.isfinite(arr_np)]
                    if arr_np.size == 0:
                        return float('nan'), float('nan')
                    return float(np.percentile(arr_np, lower)), float(np.percentile(arr_np, upper))

                pos_low, pos_high = _ci(boot_pos_rates)
                diff_low, diff_high = _ci(boot_mean_diffs)
                report_lines.append(f"- **Positive rate 95% CI:** [{pos_low:.1%}, {pos_high:.1%}]")
                if not np.isnan(diff_low):
                    report_lines.append(
                        f"- **Mean return diff (label=1 - label=0) 95% CI:** [{diff_low:.2%}, {diff_high:.2%}]"
                    )
        except Exception as e_boot:
            report_lines.append(f"\n⚠️ Could not compute bootstrap statistics: {e_boot}")

    except Exception as e:
        report_lines.append(f"\n⚠️ Could not compute stability/bootstrap diagnostics: {e}")

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

    # LightGBM: Slightly higher capacity and weaker regularization for richer patterns
    models['lgbm'] = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        n_estimators=800,
        max_depth=6,
        learning_rate=0.01,
        num_leaves=63,
        min_child_samples=10,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.05,  # L1 regularization
        reg_lambda=0.05,  # L2 regularization
        n_jobs=-1,
        verbose=-1,
        random_state=42
    )

    # Logistic Regression with elastic net (L1 + L2), slightly weaker regularization
    models['logreg'] = Pipeline([
        (
            'scaler',
            RobustScaler()
        ),
        (
            'logreg',
            LogisticRegression(
                penalty='elasticnet',
                solver='saga',
                C=2.0,
                l1_ratio=0.3,
                max_iter=1000,
                n_jobs=-1,
                random_state=42
            )
        ),
    ])

    # Random Forest with slightly higher capacity
    models['rf'] = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=10,
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
        # Skip Platt, use raw predictions but fill missing with neutral 0.5
        calibrated_predictions = oof_predictions.copy()
        calibrated_predictions = calibrated_predictions.fillna(0.5)

    # STAGE 2: Blend models with soft voting
    tprint("  📊 Stage 2: Blending models with soft voting...", "INFO")

    # Simple average (can be weighted based on validation performance)
    ensemble_probs = calibrated_predictions.mean(axis=1)

    # STAGE 3: Isotonic regression on ensemble output
    tprint("  📈 Stage 3: Applying isotonic regression to ensemble...", "INFO")

    ensemble_valid = ensemble_probs[valid_mask].values

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
    Add signal disagreement feature (std dev across ensemble predictions).

    This is one of the strongest predictors of signal failure.
    High disagreement = models don't agree = lower confidence.

    Args:
        oof_predictions: Out-of-fold predictions from each base model
        meta_features: Existing meta-features

    Returns:
        Updated meta-features with disagreement column
    """
    # Compute std dev across model predictions
    disagreement = oof_predictions.std(axis=1, skipna=True)

    # Add to meta-features
    meta_features_updated = meta_features.copy()
    meta_features_updated['signal_disagreement'] = disagreement

    # Also add mean and max disagreement as features
    meta_features_updated['signal_disagreement_ema'] = disagreement.ewm(span=10).mean()

    tprint(f"  ✓ Added signal disagreement (mean={disagreement.mean():.3f}, std={disagreement.std():.3f})", "INFO")

    return meta_features_updated


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

    # Align volatility with signals index
    if volatility is not None:
        vol_aligned = volatility.reindex(signals.index)
    else:
        vol_aligned = pd.Series(np.nan, index=signals.index)

    vol_values = vol_aligned.to_numpy(dtype=float)
    vol_mean = np.nanmean(vol_values)
    if not np.isfinite(vol_mean) or vol_mean == 0.0:
        vol_mean = 1.0

    vol_normalized = vol_values / (vol_mean + 1e-8)
    vol_normalized = np.clip(vol_normalized, 0.5, 2.0)

    scores = probabilities.astype(float)
    scores = scores / (vol_normalized + 0.5)
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    scores[~signal_mask.to_numpy()] = -np.inf

    # Determine how many signals to select
    # Assume ~96 bars per day (15min timeframe)
    bars_per_day = 96
    n_days = n_samples / bars_per_day if bars_per_day > 0 else 0
    total_k = int(k_per_day * n_days) if n_days > 0 else 0

    # Select top-K by score
    if total_k > 0 and np.isfinite(scores).any():
        threshold_idx = min(total_k, int(signal_mask.sum()))
        finite_scores = scores[np.isfinite(scores)]
        if threshold_idx > 0 and finite_scores.size >= threshold_idx:
            score_threshold = np.partition(finite_scores, -threshold_idx)[-threshold_idx]
            selected_mask = pd.Series(scores >= score_threshold, index=signals.index)

            n_selected = selected_mask.sum()
            actual_per_day = n_selected / n_days if n_days > 0 else 0

            tprint(f"  ✓ Selected {n_selected} signals ({actual_per_day:.1f} per day)", "SUCCESS")
        else:
            tprint("  ⚠️ Not enough finite scores to select signals", "WARNING")
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
            tprint(f" {error_msg}", "ERROR")
            return {'success': False, 'error': error_msg}

        tprint(f" Starting enhanced meta-labeling for {config['symbol']}", "INFO")

        try:
            self.set_context(
                symbol=config['symbol'],
                exchange=config['exchange'],
                timeframe=config['timeframe'],
                direction=config.get('direction', 'long'),
                model=config.get('model', 'analyst')
            )

            # Extract config parameters (using production defaults)
            profit_threshold = config.get('profit_threshold', DEFAULT_PROFIT_THRESHOLD)  # 1%
            stop_threshold = config.get('stop_threshold', DEFAULT_STOP_THRESHOLD)  # 0.5%
            horizon = config.get('horizon', 16)
            transaction_cost = config.get('transaction_cost', DEFAULT_TRANSACTION_COST)  # 0.15%
            min_event_spacing = config.get('min_event_spacing', 4)

            # Extended labeling parameters (Kalman, volatility adaptation, clipping)
            kalman_Q = float(config.get('kalman_Q', 1e-4))
            kalman_R = float(config.get('kalman_R', 0.01))
            vol_baseline_window = int(config.get('vol_baseline_window', 96))
            profit_mult_min = float(config.get('profit_mult_min', 0.5))
            profit_mult_max = float(config.get('profit_mult_max', 2.0))
            stop_mult_min = float(config.get('stop_mult_min', 0.5))
            stop_mult_max = float(config.get('stop_mult_max', 2.0))
            iso_min_prob_param = float(config.get('iso_min_prob', 0.0))
            target_clip_high_q_param = config.get('target_clip_high_q', None)

            # Optionally override labeling parameters using latest HPO results
            # enable_labeling_hpo_params: if True (default), try to load best params JSON
            if config.get('enable_labeling_hpo_params', True):
                try:
                    outcomes_dir = Path('outcomes')
                    symbol = str(config['symbol'])
                    timeframe = str(config['timeframe'])
                    pattern = f"meta_labeling_hpo_best_params_{symbol}_{timeframe}_*.json"
                    candidates = sorted(outcomes_dir.glob(pattern)) if outcomes_dir.exists() else []
                    if candidates:
                        latest = candidates[-1]
                        tprint(f"🔍 Using labeling HPO params from {latest}", "INFO")
                        import json as _json
                        with open(latest, 'r') as f:
                            hpo_cfg = _json.load(f)
                        best_params = hpo_cfg.get('best_params', {}) if isinstance(hpo_cfg, dict) else {}

                        # Map HPO params → step parameters with safety clamps
                        if 'profit_thr_base' in best_params:
                            profit_threshold = float(best_params['profit_thr_base'])
                        if 'stop_to_profit_ratio' in best_params:
                            stop_ratio = float(best_params['stop_to_profit_ratio'])
                            stop_threshold = max(0.0005, profit_threshold * stop_ratio)
                        if 'horizon_bars' in best_params:
                            horizon = int(best_params['horizon_bars'])
                        if 'min_event_spacing' in best_params:
                            min_event_spacing = int(best_params['min_event_spacing'])

                        if 'kalman_Q' in best_params:
                            kalman_Q = float(best_params['kalman_Q'])
                        if 'kalman_R' in best_params:
                            kalman_R = float(best_params['kalman_R'])
                        if 'vol_baseline_window' in best_params:
                            vol_baseline_window = int(best_params['vol_baseline_window'])
                        if 'profit_mult_min' in best_params:
                            profit_mult_min = float(best_params['profit_mult_min'])
                        if 'profit_mult_max' in best_params:
                            profit_mult_max = float(best_params['profit_mult_max'])
                        if 'stop_mult_min' in best_params:
                            stop_mult_min = float(best_params['stop_mult_min'])
                        if 'stop_mult_max' in best_params:
                            stop_mult_max = float(best_params['stop_mult_max'])
                        if 'iso_min_prob' in best_params:
                            iso_min_prob_param = float(best_params['iso_min_prob'])
                        if 'target_clip_high_q' in best_params:
                            target_clip_high_q_param = float(best_params['target_clip_high_q'])

                        vol_baseline_window = max(8, min(512, vol_baseline_window))
                        if profit_mult_min > profit_mult_max:
                            profit_mult_min, profit_mult_max = profit_mult_max, profit_mult_min
                        if stop_mult_min > stop_mult_max:
                            stop_mult_min, stop_mult_max = stop_mult_max, stop_mult_min

                        tprint(
                            f"⚙️ HPO overrides → profit={profit_threshold:.3%}, stop={stop_threshold:.3%}, horizon={horizon}, spacing={min_event_spacing}",
                            "INFO",
                        )
                    else:
                        tprint("ℹ️ No HPO best-params file found; using configured/default labeling parameters", "INFO")
                except Exception as hpo_exc:
                    tprint(f"⚠️ Failed to load labeling HPO params: {hpo_exc}", "WARNING")

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
            volatility_1d = log_ret.rolling(96).std()  # Short volatility estimate

            # Baseline volatility over configurable window (HPO-aware)
            vol_baseline = volatility_1d.rolling(vol_baseline_window).mean()
            vol_factor = volatility_1d / (vol_baseline + 1e-8)

            # Adaptive thresholds based on volatility and HPO multipliers
            adaptive_profit_threshold = profit_threshold * vol_factor
            adaptive_stop_threshold = stop_threshold * vol_factor

            adaptive_profit_threshold = adaptive_profit_threshold.clip(
                lower=profit_threshold * profit_mult_min,
                upper=profit_threshold * profit_mult_max,
            )
            adaptive_stop_threshold = adaptive_stop_threshold.clip(
                lower=stop_threshold * stop_mult_min,
                upper=stop_threshold * stop_mult_max,
            )

            tprint(f"📊 Adaptive thresholds: Profit {adaptive_profit_threshold.mean():.2%} ± {adaptive_profit_threshold.std():.2%}", "INFO")
            tprint(f"📊 Adaptive thresholds: Stop {adaptive_stop_threshold.mean():.2%} ± {adaptive_stop_threshold.std():.2%}", "INFO")

            # STEP 3: Compute realized returns (continuous) and binary labels with adaptive thresholds
            tprint("💰 Computing realized returns with adaptive thresholds and transaction costs...", "INFO")
            realized_returns, binary_labels, exit_reasons, event_durations = compute_realized_returns(
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
                Q=kalman_Q,
                R=kalman_R,
                volatility=volatility_1d
            )
            tprint(f"📊 Smoothed labels: Mean={smoothed_labels[labeled_mask].mean():.3f}, Std={smoothed_labels[labeled_mask].std():.3f}", "INFO")

            # Event-centric and label-history features (event-only where applicable)
            event_mask = ~binary_labels.isna()

            # Bars since last labeled event
            idx = np.arange(len(market_data))
            last_event_idx = np.where(event_mask.to_numpy(), idx, np.nan)
            last_event_idx_series = pd.Series(last_event_idx, index=market_data.index).ffill()

            bars_since_last_event = idx - last_event_idx_series.values
            bars_since_last_event[last_event_idx_series.isna().values] = np.nan

            # Distance to recent highs/lows and recent drawdown
            recent_high_50 = market_data['high'].rolling(50).max()
            recent_low_50 = market_data['low'].rolling(50).min()
            dist_from_recent_high_50 = (market_data['close'] - recent_high_50) / (recent_high_50 + 1e-8)
            dist_from_recent_low_50 = (market_data['close'] - recent_low_50) / (recent_low_50 + 1e-8)

            rolling_max_100 = market_data['close'].rolling(100).max()
            drawdown_100 = (market_data['close'] - rolling_max_100) / (rolling_max_100 + 1e-8)

            # Label-history (rolling over past events only)
            event_returns = realized_returns[event_mask]
            event_labels = binary_labels[event_mask]

            rolling_win_rate_50 = event_labels.rolling(window=50, min_periods=1).mean()
            rolling_mean_ret_50 = event_returns.rolling(window=50, min_periods=1).mean()

            win_rate_50_full = pd.Series(np.nan, index=market_data.index)
            win_rate_50_full.loc[rolling_win_rate_50.index] = rolling_win_rate_50

            mean_ret_50_full = pd.Series(np.nan, index=market_data.index)
            mean_ret_50_full.loc[rolling_mean_ret_50.index] = rolling_mean_ret_50

            # STEP 5: Create meta-features with Kalman filtering
            tprint("🔧 Creating meta-features with Kalman filtering...", "INFO")
            meta_features = create_meta_features(
                market_data,
                primary_signals,
                volume_available,
                include_raw_signals=False,  # CRITICAL: avoid circular behavior
                use_kalman=True  # Enable Kalman filtering
            )

            # Attach event-centric and label-history features
            event_meta_features = pd.DataFrame(index=market_data.index)
            event_meta_features['bars_since_last_event'] = bars_since_last_event
            event_meta_features['dist_from_recent_high_50'] = dist_from_recent_high_50
            event_meta_features['dist_from_recent_low_50'] = dist_from_recent_low_50
            event_meta_features['drawdown_100'] = drawdown_100
            event_meta_features['event_win_rate_last_50'] = win_rate_50_full
            event_meta_features['event_mean_return_last_50'] = mean_ret_50_full

            meta_features = meta_features.join(event_meta_features)

            meta_features_model = prepare_feature_matrix(meta_features)

            # STEP 6: Train ensemble meta-models with K-fold cross-fitting
            tprint("🎓 Training ensemble meta-models (LGBM + LogReg + RF) with purged K-fold CV...", "INFO")

            # Train ensemble and get OOF predictions
            trained_models, oof_predictions_df = train_ensemble_with_kfold(
                X=meta_features_model,
                y=binary_labels,
                horizon=horizon,
                n_splits=5,
                verbose=True
            )

            # STEP 7: Add signal disagreement feature
            tprint("🔧 Adding signal disagreement feature...", "INFO")
            meta_features_enhanced = add_signal_disagreement(
                oof_predictions=oof_predictions_df,
                meta_features=meta_features
            )

            # STEP 8: Calibrate ensemble with isotonic regression only (preserve variance)
            tprint("📈 Calibrating ensemble (isotonic on blended predictions)...", "INFO")

            platt_calibrators, iso_regressor = calibrate_ensemble(
                oof_predictions=oof_predictions_df,
                y_true=binary_labels,
                realized_returns=realized_returns,
                meta_features=meta_features_enhanced,
                method='isotonic_only',
                include_context=False
            )

            # STEP 9: Generate final ensemble predictions
            tprint("🎯 Generating final ensemble predictions...", "INFO")

            # Average OOF predictions after calibration (for final probabilities)
            if platt_calibrators:
                # Apply Platt calibration to OOF predictions
                calibrated_oof = pd.DataFrame(index=oof_predictions_df.index)
                for model_name in oof_predictions_df.columns:
                    if model_name in platt_calibrators:
                        try:
                            oof_vals = oof_predictions_df[model_name].fillna(0.5).values.reshape(-1, 1)
                            calibrated_oof[model_name] = platt_calibrators[model_name].predict_proba(oof_vals)[:, 1]
                        except Exception:
                            calibrated_oof[model_name] = oof_predictions_df[model_name]
                    else:
                        calibrated_oof[model_name] = oof_predictions_df[model_name]

                ensemble_probs_series = calibrated_oof.fillna(0.5).mean(axis=1)
            else:
                # No per-model Platt calibration; blend raw OOF predictions with neutral fill
                oof_filled = oof_predictions_df.fillna(0.5)
                ensemble_probs_series = oof_filled.mean(axis=1)

            ensemble_probs = ensemble_probs_series.values
            probabilities = ensemble_probs

            # Compute CV metrics for reporting
            cv_results = []
            oof_mask = ~binary_labels.isna()
            for col in oof_predictions_df.columns:
                oof_mask &= ~oof_predictions_df[col].isna()

            if oof_mask.sum() > 0:
                y_oof = binary_labels[oof_mask]
                for model_name in oof_predictions_df.columns:
                    try:
                        y_pred_proba = oof_predictions_df[model_name][oof_mask]
                        auc = roc_auc_score(y_oof, y_pred_proba)
                        y_pred = (y_pred_proba >= 0.5).astype(int)
                        precision = precision_score(y_oof, y_pred, zero_division=0)
                        recall = recall_score(y_oof, y_pred, zero_division=0)
                        f1 = f1_score(y_oof, y_pred, zero_division=0)

                        cv_results.append({
                            'model': model_name,
                            'auc': auc,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1
                        })

                        tprint(f"  📊 {model_name}: AUC={auc:.3f}, Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}", "INFO")
                    except Exception as e:
                        tprint(f"  ⚠️ Could not compute metrics for {model_name}: {e}", "WARNING")

            # STEP 10: Train final models on full dataset (for deployment)
            tprint("🎓 Training final ensemble models on full dataset...", "INFO")

            full_mask = ~binary_labels.isna()
            meta_features_enhanced_model = prepare_feature_matrix(meta_features_enhanced)
            X_full = meta_features_enhanced_model[full_mask].fillna(0)
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
                # Apply symmetric probability clipping if configured/HPO-provided
                iso_min_prob = max(0.0, min(0.1, iso_min_prob_param))
                iso_max_prob = 1.0 - iso_min_prob
                iso_max_prob = max(0.9, min(1.0, iso_max_prob))

                prob_array = np.asarray(probabilities, dtype=float)
                prob_clipped = np.clip(prob_array, iso_min_prob, iso_max_prob)

                target_long, target_short = translate_to_targets_with_isotonic(
                    realized_returns,
                    prob_clipped,
                    primary_signals,
                    iso_regressor
                )

                # Optional symmetric quantile clipping of target magnitudes
                if target_clip_high_q_param is not None:
                    try:
                        q_high = float(target_clip_high_q_param)
                        q_high = max(0.90, min(0.99, q_high))
                        q_low = max(0.0, min(0.5, 1.0 - q_high))

                        for series in (target_long, target_short):
                            nz = series[series > 0]
                            if len(nz) >= 100:
                                low_val = nz.quantile(q_low)
                                high_val = nz.quantile(q_high)
                                if low_val < high_val:
                                    series_mask = series > 0
                                    series.loc[series_mask] = series.loc[series_mask].clip(low_val, high_val)
                    except Exception as clip_exc:
                        tprint(f"⚠️ Failed to apply target quantile clipping: {clip_exc}", "WARNING")
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
            labeled_data['exit_reason'] = exit_reasons
            labeled_data['event_duration_bars'] = event_durations

            try:
                r_unit = labeled_data['adaptive_stop_threshold'].abs().replace(0.0, np.nan)
                r_multiple = (labeled_data['realized_return'] / r_unit).replace([np.inf, -np.inf], np.nan)

                strength = r_multiple.abs().clip(lower=0.5, upper=3.0)
                confidence = labeled_data['meta_probability'].astype(float).clip(lower=0.2, upper=1.0)

                raw_weight = np.sqrt(strength * confidence)
                sample_weight = raw_weight.clip(lower=0.1, upper=5.0)

                if 'binary_label' in labeled_data.columns:
                    sample_weight = sample_weight.where(~labeled_data['binary_label'].isna())

                labeled_data['r_multiple'] = r_multiple.astype(np.float32)
                labeled_data['target_sample_weight'] = sample_weight.astype(np.float32)
            except Exception:
                labeled_data['target_sample_weight'] = np.float32(1.0)

            # Rename targets to "fused" for backward compatibility with subsequent steps
            labeled_data['fused_target_long'] = target_long
            labeled_data['fused_target_short'] = target_short
            labeled_data['target_long'] = target_long
            labeled_data['target_short'] = target_short
            labeled_data['target_long_fused'] = target_long
            labeled_data['target_short_fused'] = target_short

            # Add primary signal for reference
            labeled_data['primary_signal'] = primary_signals['consensus']

            # Add adaptive thresholds for transparency
            labeled_data['adaptive_profit_threshold'] = adaptive_profit_threshold
            labeled_data['adaptive_stop_threshold'] = adaptive_stop_threshold

            timestamp_ns = np.int64(pd.Timestamp.utcnow().value)
            labeled_data['labeling_timestamp'] = timestamp_ns
            labeled_data['labeling_method_id'] = np.int8(2)

            # Save labeled data
            tprint("Saving labeled data...", "INFO")

            labeled_data_path = self._save_artifact(
                data=labeled_data,
                artifact_name=f"labeled_data_{config['symbol']}_{config['timeframe']}",
                artifact_type="data",
                compression="auto",
                data_category="features",
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

            # Calculate metrics
            avg_auc = np.mean([r['auc'] for r in cv_results]) if cv_results else 0.5
            avg_precision = np.mean([r['precision'] for r in cv_results]) if cv_results else 0.0

            # Feature importances (use enhanced features)
            try:
                feature_importances = dict(zip(X_full.columns, final_model.feature_importances_))
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
                    'labeled_data_file': labeled_data_path,
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
                    'n_cv_folds': len(cv_results),
                    'elapsed_seconds': elapsed_time,
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
                        'kfold_cross_fitting': True
                    }
                },
                'cv_results': cv_results
            }

            tprint(f"✅ Enhanced meta-labeling completed in {elapsed_time:.1f}s", "SUCCESS")
            tprint(f"📊 Performance: AUC={avg_auc:.3f}, Win Rate={win_rate:.1%}, Mean Return={mean_return:.2%}", "SUCCESS")

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
