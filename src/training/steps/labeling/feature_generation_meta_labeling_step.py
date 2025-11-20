"""
Feature Generation Meta-Labeling Step (Production Version).

Major enhancements:
1. Ensemble models (LGBM + XGBoost + RF) with soft voting
2. K-fold cross-fitting to prevent leakage
3. Volatility-adaptive labeling with Kalman filtering
4. Robust feature engineering with RobustScaler
5. Vectorized operations for performance
6. Comprehensive diagnostics and calibration
7. Production TPSL parameters (1% profit, 0.5% stop, 0.15% fee)

2025-11-18 DATA STARVATION FIX:
- Removed vol_expansion FILTER → now a continuous FEATURE
- Dynamic threshold tuning: targets 1-3 trades/day (500-1500 signals/year)
- Removed cost subtraction & power scaling from target generation
- Wider volatility-adjusted horizons: 0.5x to 3.0x (was 0.75x to 1.5x)
- Stronger regularization in tree models (prevent overfitting)
- Feature selection (remove correlated features, limit max features)
- Sequential bootstrapping sample weights (handle overlapping events)

IMPLEMENTED ADVANCED FEATURES:
- [✓] Focal Loss: Custom loss function for LGBM/XGB (optional, use_focal_loss parameter)
- [✓] CUSUM Filters: de Prado's structural break detector (use_cusum_filter parameter)
- [✓] Multi-Class Labels: (0=Timeout, 1=Profit, 2=Stop) - use_multiclass_labels parameter
- [✓] HPO System: Label quality discovery with learnability scoring and entropy constraints

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
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
import xgboost as xgb
import hashlib
import pickle

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
from .labeled_data_schema import (
    LABELED_DATA_SCHEMA_VERSION,
    get_required_labeled_data_columns,
    validate_labeled_data_schema,
)
from .label_config import (
    LABEL_CONFIG_VERSION,
    build_label_config,
    compute_label_config_id,
)

logger = logging.getLogger(__name__)

# Production TPSL Parameters (overridable via config)
DEFAULT_PROFIT_THRESHOLD = 0.01  # 1%
DEFAULT_STOP_THRESHOLD = 0.005   # 0.5%
DEFAULT_TRANSACTION_COST = 0.0015  # 0.15% per trade
R_MULTIPLE_POS_THRESHOLD = 0.7
R_MULTIPLE_NEG_THRESHOLD = -0.25
ECON_MIN_RETURN_MULTIPLE = 2.0
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


def detect_cusum_events(
    returns: pd.Series,
    threshold: float = 0.02,
    drift: float = 0.0
) -> Tuple[pd.Series, pd.Series]:
    """
    CUSUM (Cumulative Sum) Filter for event detection.

    Detects structural breaks by accumulating deviations from a drift level.
    More robust than simple momentum thresholds as it captures persistent directional moves.

    Based on de Prado's "Advances in Financial Machine Learning" Chapter 2.5.

    Args:
        returns: Log returns series
        threshold: CUSUM threshold (e.g., 0.02 = 2% cumulative move)
        drift: Expected drift per period (usually 0.0 for log returns)

    Returns:
        Tuple of (long_events, short_events) boolean series
    """
    cusum_pos = pd.Series(0.0, index=returns.index)
    cusum_neg = pd.Series(0.0, index=returns.index)

    long_events = pd.Series(False, index=returns.index)
    short_events = pd.Series(False, index=returns.index)

    s_pos = 0.0
    s_neg = 0.0

    for i in range(1, len(returns)):
        if np.isnan(returns.iloc[i]):
            continue

        # Positive CUSUM (detects upward structural breaks)
        s_pos = max(0.0, s_pos + returns.iloc[i] - drift)
        cusum_pos.iloc[i] = s_pos

        if s_pos > threshold:
            long_events.iloc[i] = True
            s_pos = 0.0  # Reset

        # Negative CUSUM (detects downward structural breaks)
        s_neg = min(0.0, s_neg + returns.iloc[i] - drift)
        cusum_neg.iloc[i] = s_neg

        if s_neg < -threshold:
            short_events.iloc[i] = True
            s_neg = 0.0  # Reset

    return long_events, short_events


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
    momentum_threshold: Optional[float] = None,  # If None, will be auto-tuned
    target_trades_per_day: float = 2.0,  # Target signal density for dynamic tuning
    enable_dynamic_tuning: bool = True,  # Enable auto-tuning of momentum threshold
    use_cusum_filter: bool = True,  # Use CUSUM filter instead of momentum threshold
    cusum_threshold: float = 0.015  # CUSUM threshold for event detection
) -> pd.DataFrame:
    """
    Generate primary trading signals from technical indicators.

    ENHANCED IMPROVEMENTS (2025-11-18):
    - Removed volatility expansion FILTER → now a continuous FEATURE for ML
    - Dynamic threshold tuning to target specific sample sizes (1-3 trades/day)
    - Volatility weighting for consensus signals
    - Signal funnel logging

    CRITICAL: These signals are FIXED and must never be re-optimized during CV.
    They define the "primary model" whose signals we will meta-label.

    Returns:
        DataFrame with signal columns including raw indicator values for meta-features
    """
    signals = pd.DataFrame(index=df.index)
    df_local = df.copy()

    # SIGNAL FUNNEL TRACKING
    funnel = {'total_bars': len(df)}
    raw_signal_count = 0

    # ===== DYNAMIC THRESHOLD TUNING =====
    # Auto-tune momentum_threshold to achieve target signal density
    if enable_dynamic_tuning and momentum_threshold is None:
        # Estimate dataset duration (bars_per_day depends on timeframe, assume 15m → 96 bars/day)
        bars_per_day = 96  # 15m timeframe: 24h * 4 bars/hour = 96
        n_days = len(df) / bars_per_day
        target_total_signals = int(target_trades_per_day * n_days)

        tprint(f"🔧 Dynamic threshold tuning: Target {target_trades_per_day:.1f} trades/day × {n_days:.1f} days = {target_total_signals} signals", "INFO")

        # Binary search for optimal threshold
        low_thresh, high_thresh = 0.001, 0.015  # Search range
        best_thresh = 0.006  # Fallback
        tolerance = 0.15  # Accept within 15% of target

        for iteration in range(12):  # Max 12 iterations
            mid_thresh = (low_thresh + high_thresh) / 2

            # Test with candidate threshold (run momentum calc only)
            test_momentum = df_local['close'].pct_change(momentum_period)
            test_count_long = (test_momentum > mid_thresh).sum()
            test_count_short = (test_momentum < -mid_thresh).sum()
            test_count = test_count_long + test_count_short

            error_ratio = abs(test_count - target_total_signals) / (target_total_signals + 1)

            if error_ratio < tolerance:
                best_thresh = mid_thresh
                tprint(f"  ✓ Converged at iteration {iteration+1}: threshold={best_thresh:.4f} → {test_count} signals", "INFO")
                break
            elif test_count < target_total_signals:
                high_thresh = mid_thresh  # Loosen (lower threshold)
            else:
                low_thresh = mid_thresh  # Tighten (raise threshold)

            if iteration == 11:  # Last iteration
                best_thresh = mid_thresh
                tprint(f"  ⚠️ Max iterations reached: threshold={best_thresh:.4f} → {test_count} signals (target: {target_total_signals})", "WARNING")

        momentum_threshold = best_thresh
    elif momentum_threshold is None:
        momentum_threshold = 0.006  # Default fallback

    tprint(f"📊 Using momentum_threshold={momentum_threshold:.4f}", "INFO")

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

    # ===== MOMENTUM / CUSUM SIGNALS =====
    if use_cusum_filter:
        # Use CUSUM filter (de Prado's structural break detector)
        log_returns = np.log(df_local['close']).diff()
        cusum_long, cusum_short = detect_cusum_events(
            log_returns,
            threshold=cusum_threshold,
            drift=0.0
        )

        signals['mom'] = 0
        signals.loc[cusum_long, 'mom'] = 1
        signals.loc[cusum_short, 'mom'] = -1

        tprint(f"  CUSUM events: {cusum_long.sum()} long, {cusum_short.sum()} short", "INFO")
    else:
        # Use simple momentum threshold
        df_local['momentum'] = df_local['close'].pct_change(momentum_period)
        signals['mom'] = 0
        signals.loc[df_local['momentum'] > momentum_threshold, 'mom'] = 1
        signals.loc[df_local['momentum'] < -momentum_threshold, 'mom'] = -1

    # ===== VOLATILITY EXPANSION CALCULATION =====
    # Compute short and long volatility
    log_ret = np.log(df_local['close']).diff()
    vol_short = log_ret.rolling(20).std()  # Short-term volatility (20 bars ~5h on 15m)
    vol_long = log_ret.rolling(96).std()   # Long-term volatility (96 bars ~1 day on 15m)

    # Calculate volatility expansion ratio (NO LONGER A FILTER - now a feature for ML)
    vol_expansion_ratio = vol_short / (vol_long + 1e-8)
    vol_expansion_ratio = vol_expansion_ratio.fillna(1.0)

    # Store volatility values for meta-features and diagnostics
    signals['vol_short'] = vol_short
    signals['vol_long'] = vol_long
    signals['vol_expansion'] = vol_expansion_ratio  # Continuous ratio, not boolean

    # ===== CONSENSUS SIGNAL WITH VOLATILITY WEIGHTING =====
    # Use all signals for consensus (including long-term for multi-timeframe)
    signal_cols = ['rsi', 'rsi_long', 'macd', 'macd_long', 'ma', 'mom']

    # Count raw signals before filtering
    raw_consensus = signals[signal_cols].sum(axis=1).apply(np.sign)
    raw_signal_count = (raw_consensus != 0).sum()
    funnel['raw_signals'] = raw_signal_count

    # Apply volatility-weighted consensus
    # Weight signals by current volatility regime
    vol_weight = (vol_short / (vol_long + 1e-8)).clip(0.5, 2.0)  # Boost signals in expansion
    weighted_sum = signals[signal_cols].sum(axis=1) * vol_weight
    signals['consensus'] = weighted_sum.apply(np.sign)

    # SIGNAL FUNNEL LOGGING (NO FILTER - wider net for ML to learn from)
    final_signal_count = (signals['consensus'] != 0).sum()
    funnel['final_signals'] = final_signal_count

    tprint(f"📊 Signal Funnel:", "INFO")
    tprint(f"  Total bars: {funnel['total_bars']}", "INFO")
    tprint(f"  Raw signals generated: {funnel['raw_signals']}", "INFO")
    tprint(f"  Final signals (no vol filter): {funnel['final_signals']}", "INFO")
    tprint(f"  ℹ️  Vol expansion now used as ML feature, not filter", "INFO")

    # Store raw indicator values for meta-features (signal disagreement, magnitude, etc.)
    signals['rsi_value'] = df_local['rsi']
    signals['rsi_long_value'] = df_local['rsi_long']
    signals['macd_hist_value'] = df_local['macd_hist']
    signals['macd_hist_long_value'] = df_local['macd_hist_long']
    signals['sma_fast_value'] = df_local['sma_fast']
    signals['sma_slow_value'] = df_local['sma_slow']

    if 'momentum' not in df_local.columns:
        df_local['momentum'] = df_local['close'].pct_change(momentum_period)

    signals['momentum_value'] = df_local['momentum']

    return signals


def compute_realized_returns(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    profit_threshold: Union[float, pd.Series] = 0.015,
    stop_threshold: Union[float, pd.Series] = 0.010,
    horizon: int = 16,
    transaction_cost: float = 0.0005,
    min_event_spacing: int = 4,
    volatility_series: Optional[pd.Series] = None,
    use_multiclass_labels: bool = False  # NEW: 3-class labels (0=timeout, 1=profit, 2=stop)
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Compute realized returns for each signal event.

    ENHANCED IMPROVEMENTS:
    - Uses High/Low prices for TP/SL checks (more realistic)
    - Adds velocity/efficiency penalty for slow trades
    - Dynamic horizon based on volatility (linear scaling, 2x max cap)
    - Tracks MFE/MAE for diagnostics
    - Supports adaptive thresholds based on volatility
    - NEW: Multi-class labels (0=timeout, 1=profit, 2=stop) for more nuanced learning

    Args:
        df: DataFrame with OHLCV data
        signals: DataFrame with signal columns
        profit_threshold: Profit target as fraction (float or Series for adaptive)
        stop_threshold: Stop loss as fraction (float or Series for adaptive)
        horizon: Base maximum bars to look ahead
        transaction_cost: Transaction cost per trade (round trip)
        min_event_spacing: Minimum bars between signals (prevents overlapping events)
        volatility_series: Volatility series for dynamic horizon scaling (optional)
        use_multiclass_labels: If True, returns 3-class labels (0=timeout, 1=profit, 2=stop)
                               If False, returns binary labels (0=loss/timeout, 1=profit)

    Returns:
        Tuple of (realized_returns, labels, exit_reasons, event_durations, mfe_series, mae_series)
        - realized_returns: Actual returns achieved (NaN where no signal)
        - labels: Binary (0/1) or Multi-class (0/1/2) depending on use_multiclass_labels
        - exit_reasons: How each event exited ('profit', 'stop', 'timeout')
        - event_durations: Bars held for each event
        - mfe_series: Maximum Favorable Excursion for each event
        - mae_series: Maximum Adverse Excursion for each event
    """
    realized_returns = pd.Series(index=df.index, dtype=float)
    realized_returns[:] = np.nan

    binary_labels = pd.Series(index=df.index, dtype=float)
    binary_labels[:] = np.nan

    exit_reasons = pd.Series(index=df.index, dtype=object)
    exit_reasons[:] = pd.NA

    event_durations = pd.Series(index=df.index, dtype=float)
    event_durations[:] = np.nan

    # NEW: Track MFE/MAE for each event
    mfe_series = pd.Series(index=df.index, dtype=float)
    mfe_series[:] = np.nan

    mae_series = pd.Series(index=df.index, dtype=float)
    mae_series[:] = np.nan

    close_prices = df['close'].values
    high_prices = df['high'].values if 'high' in df.columns else close_prices
    low_prices = df['low'].values if 'low' in df.columns else close_prices
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

    # Dynamic horizon based on volatility (LINEAR with 2x max cap)
    # Lower vol = More time needed (price moves slower in low vol environments)
    # Higher vol = Less time needed (price moves faster in high vol environments)
    if volatility_series is not None:
        vol_array = volatility_series.values
        # Normalize volatility to [0, 1] range using quantiles
        vol_clean = vol_array[~np.isnan(vol_array)]
        if len(vol_clean) > 10:
            vol_min = np.percentile(vol_clean, 10)
            vol_max = np.percentile(vol_clean, 90)
            # normalized_vol = 0 for low vol, 1 for high vol
            normalized_vol = np.clip((vol_array - vol_min) / (vol_max - vol_min + 1e-8), 0, 1)

            # LINEAR SCALING: 2.0x at low vol (slow moves), 0.5x at high vol (fast moves)
            # time_multiplier = 2.0 - 1.5 * normalized_vol → Range: [0.5, 2.0]
            time_multiplier = 2.0 - 1.5 * normalized_vol
            dynamic_horizons = (horizon * time_multiplier).astype(int)

            # Safety bounds: [horizon/2, horizon*2]
            dynamic_horizons = np.clip(dynamic_horizons, max(4, horizon // 2), horizon * 2)
        else:
            dynamic_horizons = np.full(len(df), horizon)
    else:
        dynamic_horizons = np.full(len(df), horizon)

    last_event_idx = -min_event_spacing  # Track last signal to avoid overlaps

    i = 0
    n = len(df)

    while i < n - 1:
        signal = consensus_signals[i]

        # Only create labels where we have a signal
        if signal == 0:
            i += 1
            continue

        # Handle overlapping events: skip ahead if too close to previous signal
        if (i - last_event_idx) < min_event_spacing:
            i = last_event_idx + min_event_spacing
            continue

        # Get dynamic horizon for this event
        event_horizon = int(dynamic_horizons[i])

        # Edge window handling: skip events too close to end of available data
        if i + event_horizon >= n:
            i += 1
            continue

        entry_price = close_prices[i]
        exit_price = None
        exit_reason = None
        event_end_idx = i

        # Get adaptive thresholds for this event
        profit_thr = profit_thresholds[i]
        stop_thr = stop_thresholds[i]

        # Track MFE/MAE during the event
        max_favorable = 0.0
        max_adverse = 0.0

        # Look ahead up to dynamic horizon bars
        for j in range(1, event_horizon + 1):
            idx = i + j
            if idx >= n:
                break

            # NEW: Use High/Low prices for more realistic TP/SL checks
            high_price = high_prices[idx]
            low_price = low_prices[idx]
            close_price = close_prices[idx]

            if signal > 0:  # Long signal
                # Check high for profit target (intra-bar high could have hit TP)
                pnl_high = (high_price - entry_price) / entry_price
                # Check low for stop loss (intra-bar low could have hit SL)
                pnl_low = (low_price - entry_price) / entry_price
                # Current P&L based on close
                pnl_close = (close_price - entry_price) / entry_price

                # Track MFE/MAE
                max_favorable = max(max_favorable, pnl_high)
                max_adverse = min(max_adverse, pnl_low)

                # Hit profit target (check high first)
                if pnl_high >= profit_thr:
                    # Assume we got filled at profit target price
                    exit_price = entry_price * (1 + profit_thr)
                    exit_reason = 'profit'
                    event_end_idx = idx
                    break
                # Hit stop loss (check low)
                elif pnl_low <= -stop_thr:
                    # Assume we got filled at stop loss price
                    exit_price = entry_price * (1 - stop_thr)
                    exit_reason = 'stop'
                    event_end_idx = idx
                    break

            elif signal < 0:  # Short signal
                # For shorts: check low for profit, high for stop
                pnl_high = (entry_price - high_price) / entry_price  # High is bad for shorts
                pnl_low = (entry_price - low_price) / entry_price  # Low is good for shorts
                pnl_close = (entry_price - close_price) / entry_price

                # Track MFE/MAE
                max_favorable = max(max_favorable, pnl_low)
                max_adverse = min(max_adverse, pnl_high)

                # Hit profit target (check low for shorts)
                if pnl_low >= profit_thr:
                    exit_price = entry_price * (1 - profit_thr)
                    exit_reason = 'profit'
                    event_end_idx = idx
                    break
                # Hit stop loss (check high for shorts)
                elif pnl_high <= -stop_thr:
                    exit_price = entry_price * (1 + stop_thr)
                    exit_reason = 'stop'
                    event_end_idx = idx
                    break

        # If no exit, use end-of-horizon price (timeout)
        if exit_price is None:
            event_end_idx = min(i + event_horizon, n - 1)
            exit_price = close_prices[event_end_idx]
            exit_reason = 'timeout'

        # Compute realized return accounting for transaction costs
        if signal > 0:  # Long
            gross_return = (exit_price - entry_price) / entry_price
        else:  # Short
            gross_return = (entry_price - exit_price) / entry_price

        net_return = gross_return - transaction_cost

        event_length = event_end_idx - i

        # Store realized return and event info
        realized_returns.iloc[i] = net_return
        exit_reasons.iloc[i] = exit_reason
        event_durations.iloc[i] = float(event_length)
        mfe_series.iloc[i] = max_favorable
        mae_series.iloc[i] = abs(max_adverse)  # Store as positive value

        # Label assignment: Binary or Multi-class
        econ_min_return = ECON_MIN_RETURN_MULTIPLE * transaction_cost

        if use_multiclass_labels:
            # MULTI-CLASS LABELS: 0=timeout, 1=profit, 2=stop
            # This allows model to learn different patterns for each exit type
            if exit_reason == 'timeout':
                binary_labels.iloc[i] = 0.0  # Timeout/noise
            elif exit_reason == 'profit':
                binary_labels.iloc[i] = 1.0  # Hit profit target
            elif exit_reason == 'stop':
                binary_labels.iloc[i] = 2.0  # Hit stop loss (bad entry)
            else:
                binary_labels.iloc[i] = np.nan  # Should not happen
        else:
            # BINARY LABELS: Velocity/efficiency-adjusted (legacy) with SOFT-LABELING
            # NEW: Instead of dropping small returns as NaN, soft-label them as class 0 (neutral)
            # with implicit low sample weight (0.01). This helps model learn "what to avoid"
            # without being penalized for missing "grey" trades.
            if abs(net_return) < econ_min_return:
                # SOFT-LABELING: Treat small/economically trivial returns as class 0 (neutral)
                # instead of dropping them. The model will see these as "grey" data but won't
                # be heavily penalized for misclassifying them. Sample weights should be set to 0.01
                # downstream in model training (src/training/steps/models_training/core/model_trainer.py)
                binary_labels.iloc[i] = 0.0
            else:
                # Efficiency ratio: 1.0 / log(1 + duration)
                # Fast trades (1-2 bars) get full credit, slow trades (16+ bars) get penalized
                efficiency_ratio = 1.0 / np.log1p(event_length)

                # Velocity-adjusted return for binary labeling only
                velocity_adjusted_return = net_return * efficiency_ratio

                risk_unit = stop_thr if stop_thr > 0 else profit_thr
                if risk_unit <= 0:
                    r_multiple = 0.0
                else:
                    # Use velocity-adjusted return for R-multiple calculation
                    r_multiple = velocity_adjusted_return / risk_unit

                if r_multiple >= R_MULTIPLE_POS_THRESHOLD:
                    binary_labels.iloc[i] = 1.0
                elif net_return < 0:  # Losses are losses regardless of speed
                    binary_labels.iloc[i] = 0.0
                else:
                    # Profitable but too slow = noise/drift, soft-label as class 0
                    binary_labels.iloc[i] = 0.0

        last_event_idx = i  # Update last event position
        i += 1

    return realized_returns, binary_labels, exit_reasons, event_durations, mfe_series, mae_series


def compute_vol_scaled_returns_for_events(
    realized_returns: pd.Series,
    volatility: Optional[pd.Series],
) -> pd.Series:
    vol_scaled = pd.Series(index=realized_returns.index, dtype=float)
    vol_scaled[:] = np.nan

    if volatility is None:
        return vol_scaled

    try:
        vol_aligned = volatility.reindex(realized_returns.index)
        vol_aligned = vol_aligned.astype(float)
        vol_aligned = vol_aligned.replace(0.0, np.nan)
        vol_scaled = realized_returns.astype(float) / (vol_aligned.abs() + 1e-8)

        # Drop economically trivial events from the vol-scaled series so that
        # quantile-based labels focus on meaningful moves.
        econ_floor = ECON_MIN_RETURN_MULTIPLE * DEFAULT_TRANSACTION_COST
        small_mask = realized_returns.abs() < econ_floor
        vol_scaled[small_mask] = np.nan

        # Robust outlier handling: winsorize extreme vol-scaled returns so that
        # a handful of large moves do not dominate quantile thresholds.
        try:
            v = vol_scaled.dropna()
            if len(v) >= 100:
                low_clip = float(v.quantile(0.005))
                high_clip = float(v.quantile(0.995))
                if np.isfinite(low_clip) and np.isfinite(high_clip) and low_clip < high_clip:
                    vol_scaled = vol_scaled.clip(lower=low_clip, upper=high_clip)
        except Exception:
            # Never let defensive winsorisation break labeling; fall back to raw vol_scaled
            pass
    except Exception:
        vol_scaled[:] = np.nan

    return vol_scaled


def create_quantile_labels_from_vol_scaled_returns(
    vol_scaled: pd.Series,
    low_q: float = 0.3,
    high_q: float = 0.7,
) -> pd.Series:
    labels = pd.Series(index=vol_scaled.index, dtype=float)
    labels[:] = np.nan

    try:
        v = vol_scaled.dropna()
        if len(v) < 100:
            return labels

        low_val = float(v.quantile(low_q))
        high_val = float(v.quantile(high_q))

        if not np.isfinite(low_val) or not np.isfinite(high_val) or low_val >= high_val:
            return labels

        labels.loc[vol_scaled >= high_val] = 1.0
        labels.loc[vol_scaled <= low_val] = 0.0
    except Exception:
        labels[:] = np.nan

    return labels


def create_regime_aware_quantile_labels_from_vol_scaled_returns(
    vol_scaled: pd.Series,
    regimes: Optional[pd.Series] = None,
    low_q: float = 0.3,
    high_q: float = 0.7,
    min_samples_per_regime: int = 100,
) -> pd.Series:
    """Regime-aware wrapper around quantile-based label generation.

    If a regime series is provided, compute quantile thresholds separately
    within each regime. If there are not enough samples per regime or if
    regimes is None, fall back to global quantiles.
    """

    labels = pd.Series(index=vol_scaled.index, dtype=float)
    labels[:] = np.nan

    # Fallback to global behavior if no regimes are provided
    if regimes is None:
        return create_quantile_labels_from_vol_scaled_returns(
            vol_scaled=vol_scaled,
            low_q=low_q,
            high_q=high_q,
        )

    try:
        regimes_aligned = regimes.reindex(vol_scaled.index)
        v_global = vol_scaled.dropna()
        if len(v_global) < min_samples_per_regime:
            return create_quantile_labels_from_vol_scaled_returns(
                vol_scaled=vol_scaled,
                low_q=low_q,
                high_q=high_q,
            )

        unique_regimes = pd.unique(regimes_aligned.dropna())
        for reg_val in unique_regimes:
            try:
                mask = regimes_aligned == reg_val
                v_reg = vol_scaled[mask].dropna()
                if len(v_reg) < min_samples_per_regime:
                    continue

                low_val = float(v_reg.quantile(low_q))
                high_val = float(v_reg.quantile(high_q))

                if not np.isfinite(low_val) or not np.isfinite(high_val) or low_val >= high_val:
                    continue

                labels.loc[mask & (vol_scaled >= high_val)] = 1.0
                labels.loc[mask & (vol_scaled <= low_val)] = 0.0
            except Exception:
                # Never let a single regime failure break global labeling
                continue
    except Exception:
        # On any unexpected error, fall back to global behavior
        return create_quantile_labels_from_vol_scaled_returns(
            vol_scaled=vol_scaled,
            low_q=low_q,
            high_q=high_q,
        )

    return labels


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

    # ===== EXTERNAL REGIME FEATURES (e.g., HMM regimes) =====

    # If upstream steps have attached HMM regime labels/probabilities to the
    # market_data frame (e.g., via rolling_hmm_regime_* artifacts), expose them
    # as meta-features so that downstream meta-models and HPO can use them.
    try:
        if 'hmm_regime_label_1h' in df.columns:
            features['hmm_regime_label_1h'] = df['hmm_regime_label_1h']

        # Raw per-regime probabilities (regime_0_prob, regime_1_prob, ...)
        regime_prob_cols = [
            c for c in df.columns
            if c.startswith('regime_') and c.endswith('_prob')
        ]
        for col in regime_prob_cols:
            features[col] = df[col]
    except Exception as e_reg:
        tprint(f"⚠️ Warning: Could not attach external regime features: {e_reg}", "WARNING")

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

    # ===== CROSS-TIMEFRAME FEATURES (1H, 4H AGGREGATIONS) =====
    # Aggregate 15m data to higher timeframes for multi-horizon analysis

    # 1h aggregation (4 bars of 15m data)
    close_1h = df['close'].rolling(4).mean()
    high_1h = df['high'].rolling(4).max()
    low_1h = df['low'].rolling(4).min()

    features['returns_1h'] = close_1h.pct_change()
    features['momentum_1h'] = df['close'].pct_change(4)
    features['volatility_1h_agg'] = features['returns_1h'].rolling(16).std()  # 16h of 1h bars
    features['range_1h'] = (high_1h - low_1h) / (close_1h + 1e-8)

    # 4h aggregation (16 bars of 15m data)
    close_4h = df['close'].rolling(16).mean()
    high_4h = df['high'].rolling(16).max()
    low_4h = df['low'].rolling(16).min()

    features['returns_4h'] = close_4h.pct_change()
    features['momentum_4h'] = df['close'].pct_change(16)
    features['volatility_4h_agg'] = features['returns_4h'].rolling(16).std()
    features['range_4h'] = (high_4h - low_4h) / (close_4h + 1e-8)

    # ===== ROLLING WINDOW FEATURES (FOR TREE MODELS) =====
    # Trees work better with explicitly computed rolling statistics

    for window in [5, 10, 20, 50]:
        # Rolling returns statistics
        features[f'returns_mean_{window}'] = returns.rolling(window).mean()
        features[f'returns_std_{window}'] = returns.rolling(window).std()

        # Rolling price statistics
        features[f'close_min_{window}'] = df['close'].rolling(window).min()
        features[f'close_max_{window}'] = df['close'].rolling(window).max()
        features[f'close_range_{window}'] = (
            features[f'close_max_{window}'] - features[f'close_min_{window}']
        ) / (df['close'] + 1e-8)

        # Distance from recent high/low
        features[f'dist_from_recent_high_{window}'] = (
            df['close'] - features[f'close_max_{window}']
        ) / (df['close'] + 1e-8)
        features[f'dist_from_recent_low_{window}'] = (
            df['close'] - features[f'close_min_{window}']
        ) / (df['close'] + 1e-8)

    # ===== MORE INTERACTION FEATURES =====
    # Combine features to capture non-linear relationships

    # Volatility × Momentum interactions
    if 'volatility_1d' in features.columns and 'momentum_20' in features.columns:
        features['vol_momentum_interaction'] = features['volatility_1d'] * features['momentum_20']

    if 'volatility_regime' in features.columns:
        # Regime-conditional momentum
        for col in ['momentum_5', 'momentum_10', 'momentum_20']:
            if col in features.columns:
                # Create dummy variables for regime if they don't exist
                if 'vol_regime_high' in features.columns:
                    features[f'{col}_x_regime_high'] = features[col] * features['vol_regime_high']
                if 'vol_regime_medium' in features.columns:
                    features[f'{col}_x_regime_medium'] = features[col] * features['vol_regime_medium']

    # ATR × Momentum
    if 'atr_ratio' in features.columns and 'momentum_20' in features.columns:
        features['atr_momentum'] = features['atr_ratio'] * features['momentum_20']

    # Volatility × Range Position
    if 'vol_ratio' in features.columns and 'range_position' in features.columns:
        features['vol_range_interaction'] = features['vol_ratio'] * features['range_position']

    # Distance features × Volatility
    if 'dist_from_recent_high_50' in features.columns and 'volatility_1d' in features.columns:
        features['high_dist_x_vol'] = features['dist_from_recent_high_50'] * features['volatility_1d']
        features['low_dist_x_vol'] = features['dist_from_recent_low_50'] * features['volatility_1d']

    # ===== EVENT HISTORY FEATURES (FOR PRE-FILTERING) =====
    # Track historical event performance to filter low-quality signals
    # NOTE: These will only be populated after first run; use with caution to avoid leakage

    # Placeholder for event history features (to be populated from previous runs)
    # These should be computed from historical realized returns, NOT current data
    features['event_win_rate_last_50'] = 0.0  # Will be updated externally
    features['event_mean_return_last_50'] = 0.0  # Will be updated externally
    features['bars_since_last_event'] = np.nan  # Will be computed from signals

    # Compute bars since last event (non-leaking, based on past signals only)
    if 'consensus' in signals.columns:
        signal_active = (signals['consensus'] != 0).astype(int)
        idx_array = np.arange(len(df))
        last_event_idx = np.where(signal_active == 1, idx_array, np.nan)
        last_event_idx_series = pd.Series(last_event_idx, index=df.index).ffill()

        bars_since_event = idx_array - last_event_idx_series.values
        bars_since_event[last_event_idx_series.isna().values] = np.nan
        features['bars_since_last_event'] = bars_since_event

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


def winsorize_features(
    X: pd.DataFrame,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    if X.empty:
        return X
    lower = X.quantile(lower_quantile)
    upper = X.quantile(upper_quantile)
    X_clipped = X.clip(lower=lower, upper=upper, axis=1)
    return X_clipped


def rolling_robust_scale_features(
    X: pd.DataFrame,
    window: int = 256,
    min_periods: int = 64,
    skip_binary: bool = True,
    skip_low_cardinality_int: bool = True,
) -> pd.DataFrame:
    if X.empty:
        return X

    if window <= 1:
        window = 2
    if min_periods <= 0 or min_periods > window:
        min_periods = max(1, window // 4)

    X_scaled = pd.DataFrame(index=X.index)

    for col in X.columns:
        s = X[col]

        # Non-numeric columns are passed through
        if not pd.api.types.is_numeric_dtype(s):
            X_scaled[col] = s
            continue

        # Skip strictly binary indicators (0/1) if requested
        if skip_binary:
            unique_vals = pd.unique(s.dropna())
            if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
                X_scaled[col] = s.astype(np.float32, copy=False)
                continue

        # Skip very low-cardinality features if requested (likely categorical encodings)
        if skip_low_cardinality_int:
            n_unique = s.dropna().nunique()
            if n_unique <= 5:
                X_scaled[col] = s.astype(np.float32, copy=False)
                continue

        # Rolling median and IQR
        rolling_median = s.rolling(window=window, min_periods=min_periods).median()
        rolling_q75 = s.rolling(window=window, min_periods=min_periods).quantile(0.75)
        rolling_q25 = s.rolling(window=window, min_periods=min_periods).quantile(0.25)
        rolling_iqr = rolling_q75 - rolling_q25

        # Global fallback statistics for early samples / flat windows
        global_median = s.median()
        global_q75 = s.quantile(0.75)
        global_q25 = s.quantile(0.25)
        global_iqr = global_q75 - global_q25

        if not np.isfinite(global_iqr) or global_iqr <= 0:
            global_iqr = s.std()
        if not np.isfinite(global_iqr) or global_iqr <= 0:
            global_iqr = 1.0

        rolling_median = rolling_median.fillna(global_median)
        rolling_iqr = rolling_iqr.replace(0, np.nan).fillna(global_iqr)

        scaled = (s - rolling_median) / (rolling_iqr + 1e-6)
        X_scaled[col] = scaled.astype(np.float32)

    return X_scaled


def select_features_by_importance(
    X: pd.DataFrame,
    y: pd.Series,
    max_features: Optional[int] = None,
    correlation_threshold: float = 0.95,
    method: str = 'tree'
) -> List[str]:
    """
    Select important features while removing highly correlated ones.

    NEW (2025-11-18): Proper feature selection to prevent overfitting with
    increased signal count and feature complexity.

    Args:
        X: Feature matrix
        y: Binary labels
        max_features: Maximum number of features to keep (None = no limit)
        correlation_threshold: Remove features with correlation > this value
        method: 'tree' for tree-based importance, 'mutual_info' for MI

    Returns:
        List of selected feature names
    """
    # Remove features with NaN/Inf
    clean_mask = ~y.isna()
    X_clean = X[clean_mask].fillna(0)
    y_clean = y[clean_mask]

    if len(y_clean) < 20:
        tprint("⚠️ Too few samples for feature selection, using all features", "WARNING")
        return list(X.columns)

    # Step 1: Remove highly correlated features
    tprint(f"🔍 Feature selection: Starting with {len(X.columns)} features", "INFO")

    corr_matrix = X_clean.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features to drop (keep first of each correlated pair)
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > correlation_threshold)]
    features_after_corr = [col for col in X.columns if col not in to_drop]

    tprint(f"  ✓ Removed {len(to_drop)} highly correlated features (>{correlation_threshold})", "INFO")
    X_reduced = X_clean[features_after_corr]

    # Step 2: Rank by importance (if max_features specified)
    if max_features is not None and len(features_after_corr) > max_features:
        if method == 'tree':
            # Quick RandomForest for feature importance
            from sklearn.ensemble import RandomForestClassifier
            rf_quick = RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                min_samples_leaf=20,
                n_jobs=-1,
                random_state=42
            )
            rf_quick.fit(X_reduced, y_clean)
            importances = rf_quick.feature_importances_

        elif method == 'mutual_info':
            # Mutual information
            from sklearn.feature_selection import mutual_info_classif
            importances = mutual_info_classif(X_reduced, y_clean, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Select top K
        top_indices = np.argsort(importances)[::-1][:max_features]
        selected_features = [features_after_corr[i] for i in top_indices]

        tprint(f"  ✓ Selected top {max_features} features by {method} importance", "INFO")
    else:
        selected_features = features_after_corr

    tprint(f"  ✓ Final feature count: {len(selected_features)}", "INFO")

    return selected_features


def build_meta_features_for_model(
    market_data: pd.DataFrame,
    primary_signals: pd.DataFrame,
    realized_returns: pd.Series,
    binary_labels: pd.Series,
    event_durations: pd.Series,
    mfe_series: pd.Series,
    mae_series: pd.Series,
    adaptive_stop_threshold: pd.Series,
    horizon: int,
    volume_available: bool,
    meta_feature_cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Optional[np.ndarray]]:
    # Optional: label uncertainty can be passed via meta_feature_cfg to enable
    # quality-aware weighting without requiring direct access to the caller's
    # local variables.
    if isinstance(meta_feature_cfg, dict):
        label_uncertainty = meta_feature_cfg.get('_label_uncertainty')

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

    # Event-mechanics history (R-multiple, TTO, MFE/MAE) based only on past events
    try:
        # Per-event R-multiple using adaptive stop as risk unit
        r_unit_series = adaptive_stop_threshold.abs().replace(0.0, np.nan)
        r_multiple_series = (realized_returns / (r_unit_series + 1e-8)).replace([np.inf, -np.inf], np.nan)
        event_r_multiple = r_multiple_series[event_mask]

        # Time-to-outcome ratio (TTO): duration normalized by horizon
        if horizon > 0:
            event_tto = (event_durations[event_mask] / float(horizon)).replace([np.inf, -np.inf], np.nan)
        else:
            event_tto = pd.Series(index=event_returns.index, dtype=float)

        # MFE/MAE ratio
        event_mfe = mfe_series[event_mask]
        event_mae = mae_series[event_mask]
        mfe_mae_ratio_series = (event_mfe / (event_mae + 1e-6)).replace([np.inf, -np.inf], np.nan)

        # Rolling histories over past events only
        rolling_r_multiple_50 = event_r_multiple.rolling(window=50, min_periods=1).mean()
        rolling_tto_50 = event_tto.rolling(window=50, min_periods=1).mean()
        rolling_mfe_mae_ratio_50 = mfe_mae_ratio_series.rolling(window=50, min_periods=1).mean()

        r_mult_50_full = pd.Series(np.nan, index=market_data.index)
        r_mult_50_full.loc[rolling_r_multiple_50.index] = rolling_r_multiple_50

        tto_50_full = pd.Series(np.nan, index=market_data.index)
        tto_50_full.loc[rolling_tto_50.index] = rolling_tto_50

        mfe_mae_ratio_50_full = pd.Series(np.nan, index=market_data.index)
        mfe_mae_ratio_50_full.loc[rolling_mfe_mae_ratio_50.index] = rolling_mfe_mae_ratio_50
    except Exception:
        r_mult_50_full = pd.Series(np.nan, index=market_data.index)
        tto_50_full = pd.Series(np.nan, index=market_data.index)
        mfe_mae_ratio_50_full = pd.Series(np.nan, index=market_data.index)

    # STEP 5: Create meta-features with Kalman filtering
    tprint("🔧 [5/13] Creating meta-features with Kalman filtering...", "INFO")
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
    event_meta_features['event_r_multiple_mean_last_50'] = r_mult_50_full
    event_meta_features['event_tto_mean_last_50'] = tto_50_full
    event_meta_features['event_mfe_mae_ratio_mean_last_50'] = mfe_mae_ratio_50_full

    # Attach/overwrite event-centric features without creating duplicate columns
    meta_features[event_meta_features.columns] = event_meta_features

    meta_features_model = prepare_feature_matrix(meta_features)

    meta_features_model_processed = meta_features_model
    if not isinstance(meta_feature_cfg, dict):
        meta_feature_cfg = {}

    if meta_feature_cfg.get('enable_winsorisation', False):
        try:
            lower_q = float(meta_feature_cfg.get('winsor_lower_q', 0.01))
            upper_q = float(meta_feature_cfg.get('winsor_upper_q', 0.99))
            robust_window = int(meta_feature_cfg.get('robust_window', 256))
            robust_min_periods = int(meta_feature_cfg.get('robust_min_periods', max(1, robust_window // 4)))

            meta_features_model_processed = rolling_robust_scale_features(
                meta_features_model_processed,
                window=robust_window,
                min_periods=robust_min_periods,
                skip_binary=True,
                skip_low_cardinality_int=True,
            )
            meta_features_model_processed = winsorize_features(
                meta_features_model_processed,
                lower_quantile=lower_q,
                upper_quantile=upper_q,
            )
            tprint(
                f"📊 Applied rolling robust scaling (w={robust_window}) + winsorisation to meta-features (q={lower_q:.3f}-{upper_q:.3f})",
                "INFO",
            )
        except Exception as e_w:
            tprint(f"⚠️ Winsorisation failed, using raw features: {e_w}", "WARNING")

    selected_feature_names = list(meta_features_model_processed.columns)
    if meta_feature_cfg.get('enable_feature_selection', False):
        try:
            max_feats = meta_feature_cfg.get('max_features', None)
            if max_feats is not None:
                max_feats = int(max_feats)
            corr_threshold = float(meta_feature_cfg.get('correlation_threshold', 0.95))
            fs_method = meta_feature_cfg.get('selection_method', 'tree')
            selected_feature_names = select_features_by_importance(
                X=meta_features_model_processed,
                y=binary_labels,
                max_features=max_feats,
                correlation_threshold=corr_threshold,
                method=fs_method,
            )
            meta_features_model_processed = meta_features_model_processed[selected_feature_names]
        except Exception as e_fs:
            tprint(f"⚠️ Feature selection failed, using all features: {e_fs}", "WARNING")
            selected_feature_names = list(meta_features_model_processed.columns)

    sample_weights: Optional[np.ndarray] = None
    if meta_feature_cfg.get('enable_sample_weighting', False):
        try:
            if isinstance(market_data.index, pd.DatetimeIndex):
                event_mask_sw = ~binary_labels.isna()
                if event_mask_sw.any():
                    index_series = pd.Series(market_data.index, index=market_data.index)
                    event_start_times = pd.Series(pd.NaT, index=market_data.index)
                    event_start_times[event_mask_sw] = index_series[event_mask_sw]

                    # Use median bar spacing as scalar Timedelta; fall back to no adjustment on failure
                    bar_delta = index_series.diff().dropna().median()
                    if not isinstance(bar_delta, pd.Timedelta) or bar_delta <= pd.Timedelta(0):
                        event_end_times = event_start_times.copy()
                    else:
                        durations_bars = event_durations.fillna(0).round().astype(int)
                        event_end_times = event_start_times.copy()
                        event_end_times[event_mask_sw] = (
                            event_start_times[event_mask_sw]
                            + durations_bars[event_mask_sw] * bar_delta
                        )
                    sample_weights = compute_sample_weights_with_uniqueness(
                        event_start_times=event_start_times,
                        event_end_times=event_end_times,
                        y=binary_labels,
                        class_weight_mult=float(meta_feature_cfg.get('class_weight_mult', 5.0)),
                    )

                    # Optional cost- and R-multiple-aware reweighting
                    if sample_weights is not None and meta_feature_cfg.get('enable_cost_aware_weighting', True):
                        try:
                            weights = np.asarray(sample_weights, dtype=float)

                            # Compute per-event R-multiple at entry bars
                            r_unit_series = adaptive_stop_threshold.abs().replace(0.0, np.nan)
                            r_multiple_series = (realized_returns / (r_unit_series + 1e-8)).replace([np.inf, -np.inf], np.nan)
                            r_multiple_arr = r_multiple_series.fillna(0.0).to_numpy(dtype=float)

                            # Emphasize high-R winners; clip to avoid extreme weights
                            r_pos = np.clip(r_multiple_arr, 0.0, 3.0)
                            r_factor = 1.0 + r_pos  # 1..4

                            # Down-weight clearly bad losers if configured
                            neg_weight_mult = float(meta_feature_cfg.get('neg_weight_mult', 0.7))
                            label_arr = binary_labels.fillna(-1.0).to_numpy(dtype=float)

                            class_factor = np.ones_like(weights, dtype=float)
                            class_factor = np.where(label_arr == 1.0, r_factor, class_factor)
                            class_factor = np.where(label_arr == 0.0, class_factor * neg_weight_mult, class_factor)

                            # Additional noise-aware quality weighting (event diagnostics)
                            try:
                                quality_factor = np.ones_like(weights, dtype=float)

                                # 1) Kalman label uncertainty: higher uncertainty → lower weight
                                if 'label_uncertainty' in locals() and isinstance(label_uncertainty, pd.Series):
                                    lu = pd.to_numeric(label_uncertainty, errors="coerce")
                                    if lu.notna().any():
                                        lu_filled = lu.fillna(lu.median())
                                        lu_norm = (lu_filled - lu_filled.min()) / (lu_filled.max() - lu_filled.min() + 1e-8)
                                        lu_factor = 1.2 - 0.4 * lu_norm.clip(0.0, 1.0)  # ≈ [0.8, 1.2]
                                        quality_factor *= lu_factor.to_numpy(dtype=float)

                                # 2) MFE/MAE ratio: reward efficient trends, penalize noisy paths
                                if 'mfe_series' in locals() and 'mae_series' in locals() and isinstance(mfe_series, pd.Series) and isinstance(mae_series, pd.Series):
                                    mfe_local = pd.to_numeric(mfe_series, errors="coerce")
                                    mae_local = pd.to_numeric(mae_series, errors="coerce")
                                    mfe_mae = (mfe_local / (mae_local + 1e-6)).replace([np.inf, -np.inf], np.nan)
                                    mfe_mae_clipped = mfe_mae.clip(lower=0.0, upper=3.0).fillna(1.0)
                                    mfe_factor = 0.7 + 0.3 * mfe_mae_clipped  # ≈ [0.7, 1.6]
                                    quality_factor *= mfe_factor.to_numpy(dtype=float)

                                # 3) Time-to-outcome ratio (TTO): down-weight near-timeout events
                                if 'event_durations' in locals() and isinstance(event_durations, pd.Series) and horizon > 0:
                                    tto = (event_durations / float(horizon)).replace([np.inf, -np.inf], np.nan)
                                    tto_clipped = tto.clip(lower=0.0, upper=2.0).fillna(1.0)
                                    tto_factor = 1.1 - 0.4 * tto_clipped.clip(0.0, 1.5)  # ≈ [0.5, 1.1]
                                    quality_factor *= tto_factor.to_numpy(dtype=float)
                            except Exception as w_quality_exc:
                                tprint(f"⚠️ Quality-aware weighting failed, using cost/uniqueness only: {w_quality_exc}", "WARNING")
                                quality_factor = np.ones_like(weights, dtype=float)

                            # Combine class-based and quality-based weights
                            weights *= class_factor * quality_factor

                            # Normalize back to mean 1 to keep scale stable
                            if np.isfinite(weights).any() and weights.mean() > 0:
                                weights = weights / weights.mean()

                            sample_weights = weights
                        except Exception as w_cost_exc:
                            tprint(f"⚠️ Cost-aware weighting failed, using uniqueness weights only: {w_cost_exc}", "WARNING")
        except Exception as w_exc:
            tprint(f"⚠️ Sample weight computation failed, using uniform weights: {w_exc}", "WARNING")
            sample_weights = None

    return meta_features, meta_features_model_processed, selected_feature_names, sample_weights


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
    n_nan_prob = np.isnan(probabilities).sum()
    n_nan_ret = np.isnan(realized_returns).sum()
    n_inf_prob = np.isinf(probabilities).sum()
    n_inf_ret = np.isinf(realized_returns).sum()

    # Remove NaN values and ignore economically trivial events (below cost floor)
    econ_floor = ECON_MIN_RETURN_MULTIPLE * DEFAULT_TRANSACTION_COST
    base_mask = ~(np.isnan(probabilities) | np.isnan(realized_returns))
    econ_mask = np.abs(realized_returns) >= econ_floor
    mask = base_mask & econ_mask

    if not np.any(mask):
        tprint("⚠️ Warning: No economically meaningful samples for probability mapping", "WARNING")
        logger.warning("No samples with returns above economic floor for isotonic regression")
        # Fallback: use all non-NaN samples
        mask = base_mask

    p_clean = probabilities[mask]
    r_clean = realized_returns[mask]

    # Compact isotonic diagnostics: counts and filtering
    try:
        tprint(
            f"📊 [META_ISO] n_prob={len(probabilities)}, n_ret={len(realized_returns)}, "
            f"nan_prob={n_nan_prob}, nan_ret={n_nan_ret}, "
            f"inf_prob={n_inf_prob}, inf_ret={n_inf_ret}, "
            f"filtered={len(p_clean)}, econ_floor={econ_floor:.6f}",
            "INFO",
        )
    except Exception:
        # Never let diagnostics break the main flow
        logger.debug("Isotonic diagnostics logging failed", exc_info=True)

    if len(p_clean) < 10:
        tprint("⚠️ Warning: Very few samples for probability mapping", "WARNING")
        logger.warning(f"Only {len(p_clean)} samples available for isotonic regression fitting")

    if method == 'isotonic':
        # Isotonic regression: monotonic mapping
        iso = IsotonicRegression(out_of_bounds='clip')

        # Cost-aware weighting: emphasize large absolute returns
        try:
            weights = np.abs(r_clean)
            if np.isfinite(weights).any() and weights.mean() > 0:
                weights = weights / weights.mean()
            else:
                weights = None

            if weights is not None:
                iso.fit(p_clean, r_clean, sample_weight=weights)
            else:
                iso.fit(p_clean, r_clean)
        except TypeError:
            # Older sklearn versions may not support sample_weight here
            iso.fit(p_clean, r_clean)

        # Compact mapping & correlation diagnostics
        if len(p_clean) > 2:
            from scipy.stats import spearmanr
            try:
                corr, pval = spearmanr(p_clean, r_clean)

                probe_probs = np.array([0.0, 0.5, 1.0])
                probe_returns = iso.predict(probe_probs)

                tprint(
                    "📊 [META_ISO] mapping & corr → "
                    f"p=0.0:{probe_returns[0]:.4f}, "
                    f"0.5:{probe_returns[1]:.4f}, "
                    f"1.0:{probe_returns[2]:.4f}, "
                    f"spearman={corr:.4f}, p={pval:.2e}",
                    "INFO",
                )
            except Exception:
                logger.warning("  Could not compute Spearman correlation", exc_info=True)

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

    UPDATED APPROACH (2025-11-18): Use net-of-cost expected returns as targets to
    avoid collapsing around zero while preserving monotonicity. The mapping is
    still learned on OOF probabilities, but small, economically trivial moves are
    de-emphasized via the fit_probability_to_return_mapping filtering and the
    cost subtraction here.

    Args:
        realized_returns: Actual returns (used only for validation)
        probabilities: Predicted probabilities from meta-model
        signals: Signal directions
        iso_regressor: Fitted isotonic regression model
        cost_threshold: Transaction cost per trade (used only for logging)

    Returns:
        Tuple of (target_long, target_short) with raw expected returns
    """
    target_long = pd.Series(0.0, index=realized_returns.index)
    target_short = pd.Series(0.0, index=realized_returns.index)

    consensus = signals['consensus'].values

    # VECTORIZED: Predict on entire probability array at once (much faster)
    expected_returns = iso_regressor.predict(probabilities)

    # DEBUG LOGGING: Check for anomalies in isotonic predictions
    n_nan = np.isnan(expected_returns).sum()
    n_inf = np.isinf(expected_returns).sum()
    if n_nan > 0 or n_inf > 0:
        tprint(f"⚠️ WARNING: Isotonic predictions contain {n_nan} NaN and {n_inf} Inf values", "WARNING")
        expected_returns = np.nan_to_num(expected_returns, nan=0.0, posinf=0.1, neginf=-0.1)

    # Convert to net-of-cost expected returns and suppress negative expectations
    net_expected = expected_returns - cost_threshold
    net_positive = np.maximum(net_expected, 0.0)

    # Only apply minimal clipping to avoid extreme outliers
    final_targets = np.clip(net_positive, 0.0, 0.15)  # Cap at 15% to avoid outliers

    # DEBUG LOGGING: Target statistics (compact summary)
    n_nonzero = (final_targets > 1e-6).sum()
    pct_nonzero = n_nonzero / len(final_targets) * 100 if len(final_targets) > 0 else 0
    n_above_cost = (final_targets > cost_threshold).sum()
    pct_above_cost = n_above_cost / len(final_targets) * 100 if len(final_targets) > 0 else 0

    # Vectorized assignment based on signal direction
    long_mask = (consensus > 0) & (~realized_returns.isna())
    short_mask = (consensus < 0) & (~realized_returns.isna())

    target_long.iloc[long_mask] = final_targets[long_mask]
    target_short.iloc[short_mask] = final_targets[short_mask]

    # DEBUG LOGGING: Verify assignment coverage (compact)
    n_long_assigned = (target_long > 0).sum()
    n_short_assigned = (target_short > 0).sum()

    try:
        tprint(
            "📊 [META_TARGETS] nonzero="
            f"{n_nonzero}/{len(final_targets)} ({pct_nonzero:.1f}%), "
            f"above_cost={n_above_cost}/{len(final_targets)} ({pct_above_cost:.1f}%), "
            f"mean={final_targets.mean():.6f}, std={final_targets.std():.6f}, max={final_targets.max():.6f}, "
            f"assigned_long={n_long_assigned}, assigned_short={n_short_assigned}",
            "INFO",
        )
    except Exception:
        logger.debug("Target diagnostics logging failed", exc_info=True)

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
    output_dir: Path,
    exit_reasons: Optional[pd.Series] = None,
    event_durations: Optional[pd.Series] = None,
    mfe_series: Optional[pd.Series] = None,
    mae_series: Optional[pd.Series] = None,
    target_long: Optional[pd.Series] = None,
    target_short: Optional[pd.Series] = None
) -> str:
    """
    Generate comprehensive diagnostics report for meta-labeling.

    ENHANCED: Includes new diagnostic metrics:
    - Path Efficiency Ratio (PER)
    - Time-to-Outcome Ratio (TTO)
    - MFE/MAE Ratio
    - Target Volatility health check
    - Information Coefficient (IC)

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
        exit_reasons: How each event exited ('profit', 'stop', 'timeout')
        event_durations: Bars held for each event
        mfe_series: Maximum Favorable Excursion for each event
        mae_series: Maximum Adverse Excursion for each event
        target_long: Continuous target values for long positions
        target_short: Continuous target values for short positions
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

    # ===== 3. FEATURE–LABEL CORRELATION ANALYSIS (POST-FILTER) =====
    report_lines.append("\n## 3. Feature-Label Correlation Analysis (Post-Filter)\n")

    # Compute correlations (numeric features only to avoid categorical fill issues)
    features_clean = meta_features[labeled_mask]
    features_clean = features_clean.select_dtypes(include=[np.number]).fillna(0)
    labels_clean = binary_labels[labeled_mask]

    correlations_post = {}

    try:
        for col in features_clean.columns:
            corr = features_clean[col].corr(labels_clean)
            if not pd.isna(corr):
                correlations_post[col] = corr

        # Sort by absolute correlation
        sorted_corr_post = sorted(correlations_post.items(), key=lambda x: abs(x[1]), reverse=True)

        report_lines.append("\n### Top 10 Most Correlated Features (Post-Filter):\n")
        for feat, corr in sorted_corr_post[:10]:
            report_lines.append(f"- **{feat}:** {corr:.4f}")

        # Check for concerning correlations
        report_lines.append("\n### Correlation Health Check (Post-Filter):\n")

        very_high = [(f, c) for f, c in sorted_corr_post if abs(c) > 0.8]
        very_low = [(f, c) for f, c in sorted_corr_post if abs(c) < 0.01]

        if very_high:
            report_lines.append(f"\n⚠️ **Warning:** {len(very_high)} features with |corr| > 0.8 (possible leakage):")
            for feat, corr in very_high[:5]:
                report_lines.append(f"  - {feat}: {corr:.4f}")

        if len(very_low) > len(sorted_corr_post) * 0.8:
            report_lines.append(f"\n⚠️ **Warning:** {len(very_low)} features with |corr| < 0.01 (mostly uninformative)")

    except Exception as e:
        report_lines.append(f"\n⚠️ Could not compute correlations: {e}")

    # ===== 3B. PRE- VS POST-FILTER COMPARISON & FEATURE SHIFT =====
    report_lines.append("\n## 3B. Pre- vs Post-Filter Comparison and Feature Shift\n")

    try:
        # Pre-filter: all events with realized returns
        pre_mask = ~realized_returns.isna()
        n_pre_total = int(pre_mask.sum())

        if n_pre_total > 0:
            tx_cost_local = config.get('transaction_cost', DEFAULT_TRANSACTION_COST)
            try:
                tx_cost_local = float(tx_cost_local)
            except Exception:
                tx_cost_local = float(DEFAULT_TRANSACTION_COST)

            # Raw pre-filter labels: simple economic sign after costs
            pre_returns = realized_returns[pre_mask]
            raw_label_pre = (pre_returns > tx_cost_local).astype(int)

            n_pre_pos = int((raw_label_pre == 1).sum())
            n_pre_neg = int((raw_label_pre == 0).sum())

            n_post_total = int(n_labeled)
            n_post_pos = int((binary_labels == 1.0).sum())
            n_post_neg = int((binary_labels == 0.0).sum())

            retention_total = n_post_total / max(n_pre_total, 1)
            retention_pos = n_post_pos / max(n_pre_pos, 1) if n_pre_pos > 0 else 0.0
            retention_neg = n_post_neg / max(n_pre_neg, 1) if n_pre_neg > 0 else 0.0

            report_lines.append("\n### Sample Counts and Retention\n")
            report_lines.append(f"- **Pre-filter events (realized_return not NaN):** {n_pre_total}")
            report_lines.append(f"- **Pre-filter positive/negative (raw economic):** {n_pre_pos} / {n_pre_neg}")
            report_lines.append(f"- **Post-filter labeled events:** {n_post_total}")
            report_lines.append(f"- **Post-filter positive/negative (binary_labels):** {n_post_pos} / {n_post_neg}")
            report_lines.append(f"- **Total retention (post / pre):** {retention_total:.1%}")
            report_lines.append(f"- **Positive retention:** {retention_pos:.1%}")
            report_lines.append(f"- **Negative retention:** {retention_neg:.1%}")

            if retention_pos < 0.2 and n_pre_pos >= 20:
                report_lines.append("\n⚠️ **Warning:** Filters removed >80% of economically positive candidates – risk of losing learnable signal")

            # Pre-filter effect size and simple SNR
            pre_pos_ret = pre_returns[raw_label_pre == 1]
            pre_neg_ret = pre_returns[raw_label_pre == 0]

            # Post-filter realized returns (only where labels exist)
            returns_post = realized_returns[labeled_mask]

            def _safe_stats(x: pd.Series) -> Tuple[float, float]:
                return (float(x.mean()) if len(x) > 0 else 0.0, float(x.std() if len(x) > 1 else 0.0))

            pre_pos_mean, pre_pos_std = _safe_stats(pre_pos_ret)
            pre_neg_mean, pre_neg_std = _safe_stats(pre_neg_ret)
            post_pos_mean, post_pos_std = _safe_stats(returns_post[labels_clean == 1])
            post_neg_mean, post_neg_std = _safe_stats(returns_post[labels_clean == 0])

            def _cohens_d(m1, s1, n1, m2, s2, n2) -> float:
                if n1 <= 1 or n2 <= 1:
                    return float('nan')
                pooled = ((n1 - 1) * (s1 ** 2) + (n2 - 1) * (s2 ** 2)) / max(n1 + n2 - 2, 1)
                if pooled <= 0:
                    return float('nan')
                return (m1 - m2) / np.sqrt(pooled)

            d_pre = _cohens_d(pre_pos_mean, pre_pos_std, max(len(pre_pos_ret), 1),
                               pre_neg_mean, pre_neg_std, max(len(pre_neg_ret), 1))
            d_post = _cohens_d(post_pos_mean, post_pos_std, max(len(returns_post[labels_clean == 1]), 1),
                                post_neg_mean, post_neg_std, max(len(returns_post[labels_clean == 0]), 1))

            # SNR-style metric (mean/std) for positive returns
            snr_pre = pre_pos_mean / (pre_pos_std + 1e-8) if pre_pos_std > 0 else 0.0
            snr_post = post_pos_mean / (post_pos_std + 1e-8) if post_pos_std > 0 else 0.0

            report_lines.append("\n### Pre- vs Post-Filter Signal Quality\n")
            report_lines.append(f"- **Pre-filter mean return (label=1/0):** {pre_pos_mean:.2%} / {pre_neg_mean:.2%}")
            report_lines.append(f"- **Post-filter mean return (label=1/0):** {post_pos_mean:.2%} / {post_neg_mean:.2%}")
            report_lines.append(f"- **Pre-filter Cohen's d (label=1 vs 0):** {d_pre:.3f}")
            report_lines.append(f"- **Post-filter Cohen's d (label=1 vs 0):** {d_post:.3f}")
            report_lines.append(f"- **Pre-filter SNR (mean/std, label=1):** {snr_pre:.3f}")
            report_lines.append(f"- **Post-filter SNR (mean/std, label=1):** {snr_post:.3f}")

            if np.isfinite(d_pre) and np.isfinite(d_post) and d_post < d_pre - 0.05:
                report_lines.append("\n⚠️ **Warning:** Post-filter effect size is materially worse than pre-filter – filters may be discarding informative events")

            # Pre- vs post-filter feature correlation shifts
            features_clean_pre = meta_features.select_dtypes(include=[np.number]).fillna(0)
            correlations_pre = {}
            for col in features_clean_pre.columns:
                try:
                    correlations_pre[col] = float(features_clean_pre[col].loc[pre_mask].corr(raw_label_pre))
                except Exception:
                    continue

            common_feats = set(correlations_pre.keys()).intersection(set(correlations_post.keys()))
            delta_corr = []
            for feat in common_feats:
                pre_val = correlations_pre.get(feat, 0.0)
                post_val = correlations_post.get(feat, 0.0)
                delta_corr.append((feat, pre_val, post_val, abs(post_val) - abs(pre_val)))

            delta_corr_sorted = sorted(delta_corr, key=lambda x: abs(x[3]), reverse=True)

            report_lines.append("\n### Largest Feature Correlation Shifts (|post|-|pre|)\n")
            for feat, pre_val, post_val, delta in delta_corr_sorted[:10]:
                report_lines.append(
                    f"- **{feat}:** pre={pre_val:.4f}, post={post_val:.4f}, Δ|corr|={delta:.4f}"
                )

    except Exception as e:
        report_lines.append(f"\n⚠️ Could not compute pre/post-filter comparison: {e}")

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

            # Rolling SNR on daily returns
            daily_std_ret = realized_returns.resample('1D').std()
            daily_snr = daily_mean_return / (daily_std_ret + 1e-8)

            if len(daily_snr.dropna()) > 0:
                report_lines.append("\n### Rolling Daily SNR (Return / Std by Day)\n")
                report_lines.append(f"- **Daily SNR mean:** {daily_snr.mean():.4f}")
                report_lines.append(f"- **Daily SNR std:** {daily_snr.std():.4f}")
                report_lines.append(f"- **Daily SNR min/max:** {daily_snr.min():.4f} / {daily_snr.max():.4f}")

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

    # ===== 9B. ENHANCED EVENT DIAGNOSTICS (PER, TTO, MFE/MAE) =====
    report_lines.append("\n## 9B. Enhanced Event Quality Metrics\n")
    report_lines.append(
        "These advanced metrics help distinguish between efficient momentum capture "
        "and random drift trades. They measure path efficiency, timing quality, and "
        "entry/exit effectiveness.\n"
    )

    try:
        # Path Efficiency Ratio (PER)
        if event_durations is not None and mfe_series is not None:
            labeled_durations = event_durations[labeled_mask].dropna()
            labeled_returns = returns_labeled
            labeled_mfe = mfe_series[labeled_mask].dropna() if mfe_series is not None else None

            # PER: Net price change / sum of absolute moves (approximation)
            # Higher PER = more direct path to profit
            if labeled_mfe is not None and len(labeled_mfe) > 0:
                # Approx: PER = abs(return) / MFE
                per_values = np.abs(labeled_returns) / (labeled_mfe + 1e-6)
                per_values = per_values.replace([np.inf, -np.inf], np.nan).dropna()

                if len(per_values) > 0:
                    report_lines.append("\n### Path Efficiency Ratio (PER)\n")
                    report_lines.append(f"- **Mean PER:** {per_values.mean():.3f}")
                    report_lines.append(f"- **Median PER:** {per_values.median():.3f}")

                    if per_values.mean() < 0.3:
                        report_lines.append(f"\n⚠️ **Alert:** Mean PER < 0.3 indicates excessive random walk / drift")
                    else:
                        report_lines.append(f"\n✅ **OK:** Reasonable path efficiency")

            # Time-to-Outcome Ratio (TTO)
            horizon_config = config.get('horizon', 16)
            if len(labeled_durations) > 0:
                tto_values = labeled_durations / horizon_config
                report_lines.append("\n### Time-to-Outcome Ratio (TTO)\n")
                report_lines.append(f"- **Mean TTO:** {tto_values.mean():.3f}")
                report_lines.append(f"- **Median TTO:** {tto_values.median():.3f}")

                if tto_values.mean() > 0.9:
                    report_lines.append(f"\n⚠️ **Alert:** Mean TTO > 0.9 confirms excessive timeouts (not hitting barriers)")
                elif tto_values.mean() < 0.4 or tto_values.mean() > 0.6:
                    report_lines.append(f"\n⚠️ **Warning:** TTO outside target range [0.4, 0.6]")
                else:
                    report_lines.append(f"\n✅ **OK:** TTO in healthy range [0.4, 0.6]")

            # MFE/MAE Ratio
            if mfe_series is not None and mae_series is not None:
                labeled_mfe = mfe_series[labeled_mask].dropna()
                labeled_mae = mae_series[labeled_mask].dropna()

                if len(labeled_mfe) > 0 and len(labeled_mae) > 0:
                    mfe_mae_ratio = labeled_mfe / (labeled_mae + 1e-6)
                    mfe_mae_ratio = mfe_mae_ratio.replace([np.inf, -np.inf], np.nan).dropna()

                    if len(mfe_mae_ratio) > 0:
                        report_lines.append("\n### MFE/MAE Ratio (Maximum Favorable vs Adverse Excursion)\n")
                        report_lines.append(f"- **Mean MFE/MAE:** {mfe_mae_ratio.mean():.3f}")
                        report_lines.append(f"- **Median MFE/MAE:** {mfe_mae_ratio.median():.3f}")

                        if mfe_mae_ratio.mean() < 1.0:
                            report_lines.append(f"\n⚠️ **Alert:** Average MFE/MAE < 1.0 indicates poor entry timing or fundamentally random signals")
                        else:
                            report_lines.append(f"\n✅ **OK:** Favorable excursions exceed adverse excursions")

    except Exception as e:
        report_lines.append(f"\n⚠️ Could not compute enhanced event diagnostics: {e}")

    # ===== 9C. TARGET HEALTH CHECK =====
    report_lines.append("\n## 9C. Target Volatility Health Check\n")
    report_lines.append(
        "Verifies that the continuous target values have sufficient variance to train a regression model. "
        "If targets are constant or near-zero, the model cannot learn meaningful patterns.\n"
    )

    try:
        # Combine long and short targets
        combined_targets = pd.Series(0.0, index=labeled_data.index)
        if target_long is not None:
            combined_targets = combined_targets + target_long.fillna(0)
        if target_short is not None:
            combined_targets = combined_targets + target_short.fillna(0)

        target_nonzero = combined_targets[combined_targets > 1e-6]
        target_std = combined_targets.std()
        target_mean = combined_targets.mean()
        pct_nonzero = len(target_nonzero) / len(combined_targets) * 100 if len(combined_targets) > 0 else 0

        report_lines.append(f"- **Target mean:** {target_mean:.6f}")
        report_lines.append(f"- **Target std:** {target_std:.6f}")
        report_lines.append(f"- **Non-zero targets:** {len(target_nonzero)} / {len(combined_targets)} ({pct_nonzero:.1f}%)")

        if target_std < 1e-5:
            report_lines.append(f"\n🚨 **CRITICAL:** Target std < 1e-5 - targets are effectively constant! Abort training.")
        elif pct_nonzero < 1.0:
            report_lines.append(f"\n⚠️ **Warning:** Very few non-zero targets ({pct_nonzero:.1f}%)")
        else:
            report_lines.append(f"\n✅ **OK:** Targets have sufficient variance")

    except Exception as e:
        report_lines.append(f"\n⚠️ Could not compute target health check: {e}")

    # ===== 9D. INFORMATION COEFFICIENT (IC) =====
    report_lines.append("\n## 9D. Information Coefficient (IC)\n")
    report_lines.append(
        "Measures the rank correlation between predicted probabilities and realized returns. "
        "This is the purest measure of ranking ability, independent of absolute calibration.\n"
    )

    try:
        from scipy.stats import spearmanr

        prob_labeled = probabilities[labeled_mask]
        ret_labeled = returns_labeled

        # Remove NaN values
        valid_mask = ~(np.isnan(prob_labeled) | np.isnan(ret_labeled))
        if valid_mask.sum() > 10:
            ic_corr, ic_pval = spearmanr(prob_labeled[valid_mask], ret_labeled[valid_mask])

            report_lines.append(f"- **Spearman IC (prob, return):** {ic_corr:.4f}")
            report_lines.append(f"- **P-value:** {ic_pval:.4e}")

            if abs(ic_corr) < 0.05:
                report_lines.append(f"\n⚠️ **Warning:** Very weak IC (|IC| < 0.05) - model has minimal ranking ability")
            elif abs(ic_corr) < 0.1:
                report_lines.append(f"\n⚠️ **Caution:** Weak IC (|IC| < 0.1) - limited practical value")
            else:
                report_lines.append(f"\n✅ **OK:** Meaningful rank correlation")
        else:
            report_lines.append(f"\n⚠️ Too few valid samples to compute IC")

    except Exception as e:
        report_lines.append(f"\n⚠️ Could not compute Information Coefficient: {e}")

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

    # ===== 11. LABEL–RETURN SEPARATION, INFORMATION CONTENT, AND SAMPLE SIZE =====
    report_lines.append("\n## 11. Label–Return Separation, Information Content, and Sample Size\n")
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

            # Simple power heuristic: N_required ≈ 16 / d^2 for 80% power
            try:
                if np.isfinite(effect_size) and effect_size != 0:
                    n_required = 16.0 / (effect_size ** 2)
                else:
                    n_required = float('inf')
            except Exception:
                n_required = float('inf')

            n_current = float(len(returns_positive) + len(returns_negative))
            report_lines.append(f"- **Approx. required samples for 80% power (heuristic):** {n_required:.1f}")
            report_lines.append(f"- **Current labeled samples used in separation:** {n_current:.1f}")

            if np.isfinite(n_required) and n_current < n_required:
                report_lines.append("\n⚠️ **Warning:** Current labeled sample size is below heuristic requirement for stable separation statistics")

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


def focal_loss_lgb(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal Loss for LightGBM (custom objective function).

    Focal Loss: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Down-weights easy examples and focuses on hard misclassifications.
    Critical for imbalanced datasets with noisy borderline samples.

    Args:
        y_pred: Raw predictions (logits)
        dtrain: LightGBM Dataset with labels
        alpha: Weighting factor for positive class (0.25 = slight positive bias)
        gamma: Focusing parameter (2.0 = strong focus on hard examples)

    Returns:
        Tuple of (gradient, hessian)
    """

    # Sigmoid to get probabilities
    p = 1.0 / (1.0 + np.exp(-y_pred))

    # Compute focal weights
    p_t = np.where(y_true == 1, p, 1 - p)
    focal_weight = alpha * np.power(1 - p_t, gamma)

    # Gradient and Hessian for binary cross-entropy with focal weighting
    grad = focal_weight * (p - y_true)
    hess = focal_weight * p * (1 - p)

    return grad, hess


def focal_loss_xgb(y_pred, dtrain, alpha=0.25, gamma=2.0):
    """
    Focal Loss for XGBoost (custom objective function).

    Args:
        y_pred: Raw predictions (logits)
        dtrain: XGBoost DMatrix with labels
        alpha: Weighting factor for positive class
        gamma: Focusing parameter

    Returns:
        Tuple of (gradient, hessian)
    """
    if hasattr(dtrain, "get_label"):
        y_true = dtrain.get_label()
    else:
        y_true = dtrain

    y_true = np.asarray(y_true, dtype=float).ravel()

    # Sigmoid to get probabilities
    p = 1.0 / (1.0 + np.exp(-y_pred))

    # Compute focal weights
    p_t = np.where(y_true == 1, p, 1 - p)
    focal_weight = alpha * np.power(1 - p_t, gamma)

    # Gradient and Hessian
    grad = focal_weight * (p - y_true)
    hess = focal_weight * p * (1 - p)

    return grad, hess


def create_base_models(config: Dict[str, Any], use_focal_loss: bool = True) -> Dict[str, Any]:
    """
    Create base models for ensemble with proper regularization.

    ENHANCED: Three powerful tree-based models optimized for different strengths:
    - LGBM: Deeper gradient boosting for ranking and AUC (10 depth from 6)
    - XGBoost: Powerful gradient boosting with class weighting (replaces LogReg)
    - RandomForest: Non-linear ensemble for robustness

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of {model_name: model_instance}
    """
    models = {}

    # LightGBM: Balanced capacity with STRONGER regularization (2025-11-18 update)
    if use_focal_loss:
        # Use focal loss custom objective
        models['lgbm'] = lgb.LGBMClassifier(
            objective=focal_loss_lgb,  # Custom focal loss
            metric='auc',
            n_estimators=800,
            max_depth=8,
            learning_rate=0.01,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=0.2,
            n_jobs=-1,
            verbose=-1,
            random_state=42
        )
        models['lgbm']._use_focal = True  # Flag for prediction handling
    else:
        # Standard binary cross-entropy
        models['lgbm'] = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            n_estimators=800,
            max_depth=8,
            learning_rate=0.01,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=0.2,
            class_weight='balanced',
            n_jobs=-1,
            verbose=-1,
            random_state=42
        )
        models['lgbm']._use_focal = False

    # XGBoost: Strong regularization (2025-11-18 update)
    if use_focal_loss:
        # Use focal loss custom objective
        models['xgb'] = xgb.XGBClassifier(
            objective=focal_loss_xgb,  # Custom focal loss
            eval_metric='auc',
            n_estimators=800,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.75,
            colsample_bytree=0.7,
            min_child_weight=8,
            gamma=0.2,
            reg_alpha=0.1,
            reg_lambda=0.3,
            n_jobs=-1,
            random_state=42,
            verbosity=0
        )
        models['xgb']._use_focal = True
    else:
        # Standard binary logistic
        models['xgb'] = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            n_estimators=800,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.75,
            colsample_bytree=0.7,
            min_child_weight=8,
            gamma=0.2,
            reg_alpha=0.1,
            reg_lambda=0.3,
            scale_pos_weight=4.3,  # Handle imbalance
            n_jobs=-1,
            random_state=42,
            verbosity=0
        )
        models['xgb']._use_focal = False

    # Random Forest: Balanced capacity with regularization (2025-11-18 update)
    models['rf'] = RandomForestClassifier(
        n_estimators=200,  # Reduced from 300 for speed
        max_depth=10,  # REDUCED from 12 for regularization
        min_samples_leaf=15,  # INCREASED from 8 for regularization
        min_samples_split=30,  # NEW: Require more samples for splits
        max_features='sqrt',  # Good default for high-dimensional data
        max_samples=0.8,  # NEW: Bootstrap sample size (regularization)
        class_weight='balanced',  # Handle imbalance
        n_jobs=-1,
        random_state=42
    )
    models['logreg'] = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs',
    )

    return models


def compute_sample_weights_with_uniqueness(
    event_start_times: pd.Series,
    event_end_times: pd.Series,
    y: pd.Series,
    class_weight_mult: float = 5.0
) -> np.ndarray:
    """
    Compute sample weights combining class weighting and sequential bootstrapping.

    Sequential Bootstrapping (de Prado): Down-weight samples that overlap heavily
    with others to prevent overfitting to clustered events (e.g., 10 trades during
    a single volatility spike).

    NEW (2025-11-18): Critical for handling increased signal count from loosened filters.

    Args:
        event_start_times: Start timestamps for each event
        event_end_times: End timestamps for each event
        y: Binary labels
        class_weight_mult: Multiplier for positive class (5.0 = 5x more important)

    Returns:
        Sample weights array
    """
    labeled_mask = ~y.isna()
    n_labeled = labeled_mask.sum()

    if n_labeled == 0:
        return np.ones(len(y))

    # Base class weights (handle imbalance)
    n_positive = (y == 1.0).sum()
    n_negative = (y == 0.0).sum()

    if n_positive > 0 and n_negative > 0:
        # Rare winners get higher weight
        pos_weight = n_negative / (n_positive + 1e-8) * class_weight_mult
        neg_weight = 1.0
    else:
        pos_weight = class_weight_mult
        neg_weight = 1.0

    class_weights = np.where(y == 1.0, pos_weight, neg_weight)
    class_weights = np.where(labeled_mask, class_weights, 0.0)

    # Sequential bootstrapping: compute uniqueness weights
    # Map concurrency (how many events are active at each time)
    try:
        # Get all unique timestamps
        all_times = pd.DatetimeIndex(
            sorted(set(event_start_times.dropna()).union(set(event_end_times.dropna())))
        )

        if len(all_times) < 2:
            # No overlap possible
            uniqueness_weights = np.ones(len(y))
        else:
            # Count concurrency at each event
            concurrency = pd.Series(0, index=all_times)

            for start, end in zip(event_start_times[labeled_mask], event_end_times[labeled_mask]):
                if pd.notna(start) and pd.notna(end):
                    mask = (all_times >= start) & (all_times <= end)
                    concurrency[mask] += 1

            # Compute average uniqueness for each event
            uniqueness_weights = np.ones(len(y))

            for idx, (start, end) in enumerate(zip(event_start_times, event_end_times)):
                if labeled_mask.iloc[idx] and pd.notna(start) and pd.notna(end):
                    event_times = all_times[(all_times >= start) & (all_times <= end)]
                    if len(event_times) > 0:
                        avg_concurrency = concurrency[event_times].mean()
                        uniqueness_weights[idx] = 1.0 / max(avg_concurrency, 1.0)

            tprint(f"  ✓ Sequential bootstrapping: avg uniqueness = {uniqueness_weights[labeled_mask].mean():.3f}", "INFO")

    except Exception as e:
        tprint(f"  ⚠️ Sequential bootstrapping failed: {e}, using class weights only", "WARNING")
        uniqueness_weights = np.ones(len(y))

    # Combined weights
    final_weights = class_weights * uniqueness_weights

    # Normalize
    if final_weights.sum() > 0:
        final_weights = final_weights / final_weights.mean()

    return final_weights


def train_ensemble_with_kfold(
    X: pd.DataFrame,
    y: pd.Series,
    horizon: int,
    n_splits: int = 5,
    sample_weights: Optional[np.ndarray] = None,
    verbose: bool = True
) -> Tuple[Dict[str, Any], pd.Series]:
    """
    Train ensemble models with K-fold cross-fitting to prevent leakage.

    CRITICAL: Uses purged time-series CV to avoid lookahead bias.
    Each model is trained on fold ∖i and predicts on fold i.

    NEW (2025-11-18): Support for sample weights (class imbalance + sequential bootstrapping)

    Args:
        X: Feature matrix
        y: Binary labels
        horizon: Forward-looking horizon (for purging)
        n_splits: Number of CV folds
        sample_weights: Optional sample weights (e.g., from sequential bootstrapping)
        verbose: Whether to print progress

    Returns:
        Tuple of (trained_models_dict, out_of_fold_predictions_series)
    """
    # Initialize storage
    trained_models = {'lgbm': [], 'xgb': [], 'rf': [], 'logreg': []}
    oof_predictions = {
        'lgbm': pd.Series(np.nan, index=X.index),
        'xgb': pd.Series(np.nan, index=X.index),
        'rf': pd.Series(np.nan, index=X.index),
        'logreg': pd.Series(np.nan, index=X.index),
    }
    oof_aucs = {
        'lgbm': [],
        'xgb': [],
        'rf': [],
        'logreg': [],
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

        # Extract sample weights for this fold (if provided)
        if sample_weights is not None:
            weights_train_clean = sample_weights[train_idx_purged][train_mask]
        else:
            weights_train_clean = None

        # Train each base model
        # NOTE: use_focal_loss=False for now (standard objectives work better with predict_proba)
        # Set to True to enable focal loss (focuses on hard examples, good for noise)
        base_models = create_base_models({}, use_focal_loss=False)

        for model_name, model in base_models.items():
            try:
                # Train with sample weights
                if weights_train_clean is not None:
                    model.fit(X_train_clean, y_train_clean, sample_weight=weights_train_clean)
                else:
                    model.fit(X_train_clean, y_train_clean)
                trained_models[model_name].append(model)

                # Predict on test fold
                y_pred_proba = model.predict_proba(X_test_clean)[:, 1]

                # Store OOF predictions
                test_indices_with_labels = test_idx[test_mask]
                oof_predictions[model_name].iloc[test_indices_with_labels] = y_pred_proba

                # Metrics (track for early stopping)
                try:
                    auc = roc_auc_score(y_test_clean, y_pred_proba)
                    oof_aucs[model_name].append(auc)
                    if verbose:
                        tprint(f"    ✓ {model_name}: AUC={auc:.3f}", "INFO")
                except:
                    oof_aucs[model_name].append(np.nan)
                    if verbose:
                        tprint(f"    ⚠️ {model_name}: Could not compute AUC", "WARNING")

            except Exception as e:
                if verbose:
                    tprint(f"    ❌ {model_name} failed: {e}", "ERROR")

    # EARLY STOPPING CHECK: Compute mean OOF AUC across folds
    for model_name in ['lgbm', 'xgb', 'rf', 'logreg']:
        valid_aucs = [a for a in oof_aucs[model_name] if not np.isnan(a)]
        if valid_aucs:
            mean_auc = np.mean(valid_aucs)
            std_auc = np.std(valid_aucs)
            if verbose:
                tprint(f"  {model_name} Mean OOF AUC: {mean_auc:.4f} ± {std_auc:.4f}", "INFO")
        else:
            if verbose:
                tprint(f"  {model_name}: No valid AUC scores", "WARNING")

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


# ========================================================================================
# HPO SYSTEM FOR LABEL QUALITY DISCOVERY
# ========================================================================================

def compute_learnability_score(
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 3,
    time_aware_cv: bool = True
) -> Tuple[float, float]:
    """Measure how "learnable" a specific set of labels is given the features.

    Uses a lightweight, depth-constrained probe model to prevent overfitting.
    Returns the cross-validated AUC penalized by stability (std).

    Based on concept: If AUC = 0.5, labels are random noise. If AUC = 0.7,
    labels capture structural inefficiency that features can explain.

    Args:
        X: Feature matrix
        y: Binary labels (or multi-class in future)
        cv_splits: Number of CV splits
        time_aware_cv: Use TimeSeriesSplit instead of KFold

    Returns:
        Tuple of (learnability_score, mean_auc)
    """
    # Remove NaN labels and ensure we only work with numeric features to avoid
    # categorical setitem issues when filling NaNs.
    valid_mask = ~y.isna()
    X_num = X.select_dtypes(include=[np.number]) if isinstance(X, pd.DataFrame) else X
    if isinstance(X_num, pd.DataFrame) and X_num.empty:
        return 0.0, 0.5

    X_clean = X_num[valid_mask].fillna(0)
    y_clean = y[valid_mask]

    if len(y_clean) < 50:
        return 0.0, 0.5  # Too few samples

    # Check if labels are degenerate (all same class)
    if len(y_clean.unique()) < 2:
        return 0.0, 0.5  # No signal

    # Lightweight Probe Model (shallow, fast)
    probe = lgb.LGBMClassifier(
        boosting_type='gbdt',
        objective='binary',
        max_depth=3,  # Very shallow
        n_estimators=50,  # Very few trees
        learning_rate=0.1,  # Fast convergence
        subsample=0.7,  # Stochastic for stability
        colsample_bytree=0.7,  # Feature subsampling
        min_child_samples=20,  # Prevent overfitting
        n_jobs=-1,
        verbose=-1,
        random_state=42
    )

    # Time-aware CV
    if time_aware_cv:
        from sklearn.model_selection import TimeSeriesSplit
        cv = TimeSeriesSplit(n_splits=cv_splits)
    else:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    # Cross-validate
    try:
        scores = cross_val_score(probe, X_clean, y_clean, cv=cv, scoring='roc_auc', n_jobs=-1)

        mean_auc = scores.mean()
        std_auc = scores.std()

        # Learnability score: penalize instability
        learnability = mean_auc - (0.5 * std_auc)

        return learnability, mean_auc

    except Exception as e:
        tprint(f"⚠️ Learnability scoring failed: {e}", "WARNING")
        return 0.0, 0.5


def compute_label_entropy_score(
    y: pd.Series,
    min_positive_rate: float = 0.30,
    max_positive_rate: float = 0.70,
    min_samples: int = 50
) -> float:
    """
    Compute entropy-based balance score for labels.

    Penalizes extremes (too many or too few positive labels) using a parabolic curve.
    Ensures we have enough samples for statistical significance.

    Args:
        y: Binary labels
        min_positive_rate: Minimum acceptable positive rate
        max_positive_rate: Maximum acceptable positive rate
        min_samples: Minimum number of positive samples required

    Returns:
        Balance score in [0, 1], where 1 is perfectly balanced (50/50)
    """
    valid_mask = ~y.isna()
    y_clean = y[valid_mask]

    if len(y_clean) < min_samples:
        return 0.0  # Not enough samples

    pos_rate = y_clean.mean()
    n_positive = (y_clean == 1).sum()

    # Hard constraint: too few positive samples
    if n_positive < min_samples:
        return 0.0

    # Hard constraint: too extreme distribution
    if pos_rate < min_positive_rate or pos_rate > max_positive_rate:
        return 0.0

    # Balance score: parabolic curve peaking at 0.5
    # balance = 1 - (2 * |0.5 - pos_rate|)^2
    balance_score = 1.0 - (2.0 * abs(0.5 - pos_rate)) ** 2

    return balance_score


def combined_label_quality_objective(
    X: pd.DataFrame,
    y: pd.Series,
    learnability_weight: float = 0.7,
    balance_weight: float = 0.3,
    cv_splits: int = 3
) -> Tuple[float, Dict[str, float]]:
    """
    Combined objective for HPO: Learnability * Balance.

    Finds labeling parameters that produce:
    1. High learnability (AUC >> 0.5)
    2. Sufficient sample size (not too sparse)
    3. Balanced distribution (not too extreme)

    Args:
        X: Feature matrix
        y: Binary labels
        learnability_weight: Weight for learnability component (0.7 = 70%)
        balance_weight: Weight for balance component (0.3 = 30%)
        cv_splits: Number of CV splits for learnability

    Returns:
        Tuple of (combined_score, diagnostics_dict)
    """
    # Compute components
    learnability, mean_auc = compute_learnability_score(X, y, cv_splits=cv_splits)
    balance = compute_label_entropy_score(y)

    # Combined score
    combined = (learnability_weight * learnability) + (balance_weight * balance)

    # Diagnostics
    diagnostics = {
        'learnability': learnability,
        'mean_auc': mean_auc,
        'balance': balance,
        'combined': combined,
        'n_samples': (~y.isna()).sum(),
        'positive_rate': y.mean() if (~y.isna()).sum() > 0 else 0.0
    }

    return combined, diagnostics


def compute_label_quality_score_from_components(
    coverage: float,
    retention_total: float,
    snr_post: float,
    d_post: float,
    econ_margin: float,
) -> Tuple[float, str, str]:
    """Map label-quality components into a scalar score and rating.

    This mirrors the scoring used in snr_diagnostics.run_label_quality so that
    both diagnostics and HPO can refer to the same transformation from
    coverage / retention / SNR / effect size / econ margin → [0, 1] score.
    """

    def _score_component(value: float, low: float, high: float) -> float:
        if value is None or not np.isfinite(value):
            return 0.0
        if value <= low:
            return 0.0
        if value >= high:
            return 1.0
        return float((value - low) / (high - low))

    coverage_score = _score_component(coverage, 0.05, 0.2)
    retention_score = _score_component(retention_total, 0.1, 0.3)
    snr_score = _score_component(snr_post, 0.5, 1.0)
    d_score = _score_component(abs(d_post) if np.isfinite(d_post) else float("nan"), 0.2, 1.5)
    econ_score = _score_component(econ_margin, 0.0, 0.02)

    components = [coverage_score, retention_score, snr_score, d_score, econ_score]
    label_quality_score = float(np.mean(components))

    if label_quality_score < 0.4:
        rating = "Bad"
        comment = (
            "Low coverage/SNR or weak economic separation; labels are likely "
            "noisy or too sparse."
        )
    elif label_quality_score < 0.7:
        rating = "Pass"
        comment = (
            "Mixed label quality; some usable signal but economic separation "
            "or coverage may be modest."
        )
    else:
        rating = "Great"
        comment = (
            "Strong label quality with good coverage, separation and "
            "economic margins."
        )

    return label_quality_score, rating, comment


def attach_rolling_hmm_regimes_to_market_data(
    step: BaseStep,
    market_data: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    if not isinstance(market_data, pd.DataFrame) or market_data.empty:
        return market_data

    try:
        symbol = str(config.get("symbol", "ETHUSDT"))
        exchange = str(config.get("exchange", "binance"))
        base_timeframe = str(config.get("timeframe", "15m"))
        direction = str(config.get("direction", "long"))
        regime_timeframe = str(config.get("regime_timeframe", config.get("timeframe", "1h")))
    except Exception:
        return market_data

    original_context: Optional[Dict[str, Any]] = None
    if hasattr(step, "_current_context") and isinstance(step._current_context, dict):
        original_context = step._current_context.copy()

    labels = None
    probs = None

    try:
        step.set_context(
            symbol=symbol,
            exchange=exchange,
            timeframe=regime_timeframe,
            direction=direction,
            model="regime",
        )

        labels = step._get_artifact(
            "rolling_hmm_regime_labels",
            artifact_type="data",
        )
        probs = step._get_artifact(
            "rolling_hmm_regime_probabilities",
            artifact_type="data",
        )
    except Exception as e:
        tprint(f"⚠️ Could not load rolling HMM regime artifacts: {e}", "WARNING")
    finally:
        try:
            step.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=base_timeframe,
                direction=direction,
                model=config.get("model", "analyst"),
            )
        except Exception:
            if isinstance(original_context, dict) and original_context:
                try:
                    step.set_context(**original_context)
                except Exception:
                    pass

    if labels is None and (probs is None or (isinstance(probs, pd.DataFrame) and probs.empty)):
        return market_data

    md = market_data.copy()

    try:
        labels_series = None
        if isinstance(labels, pd.DataFrame) and not labels.empty:
            labels_df = labels.copy()
            if "timestamp" in labels_df.columns:
                labels_df["timestamp"] = pd.to_datetime(labels_df["timestamp"])
                labels_df.set_index("timestamp", inplace=True)
            if isinstance(labels_df.index, pd.DatetimeIndex):
                if "regime_label_ml" in labels_df.columns:
                    regime_col = "regime_label_ml"
                elif "regime_label" in labels_df.columns:
                    regime_col = "regime_label"
                else:
                    regime_col = None
                if regime_col is not None:
                    labels_series = labels_df[regime_col].sort_index()
        if labels_series is not None:
            aligned_labels = labels_series.reindex(md.index, method="ffill")
            md["hmm_regime_label_1h"] = aligned_labels
    except Exception as e_lab:
        tprint(f"⚠️ Failed to align rolling HMM regime labels: {e_lab}", "WARNING")

    try:
        if isinstance(probs, pd.DataFrame) and not probs.empty:
            probs_df = probs.copy()
            if "timestamp" in probs_df.columns:
                probs_df["timestamp"] = pd.to_datetime(probs_df["timestamp"])
                probs_df.set_index("timestamp", inplace=True)
            if isinstance(probs_df.index, pd.DatetimeIndex):
                prob_cols = [
                    c for c in probs_df.columns
                    if c.startswith("regime_") and c.endswith("_prob")
                ]
                if prob_cols:
                    probs_sub = probs_df[prob_cols].sort_index()
                    probs_aligned = probs_sub.reindex(md.index, method="ffill")
                    for col in prob_cols:
                        md[col] = probs_aligned[col]
    except Exception as e_prob:
        tprint(f"⚠️ Failed to align rolling HMM regime probabilities: {e_prob}", "WARNING")

    return md


class HPOCache:
    """Simple cache for HPO computations to avoid recomputing labels."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("/tmp/hpo_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = {}

    def _get_key(self, params: Dict[str, Any]) -> str:
        """Generate cache key from parameters."""
        # Sort keys for consistent hashing
        param_str = str(sorted(params.items()))
        return hashlib.md5(param_str.encode()).hexdigest()

    def get(self, params: Dict[str, Any]) -> Optional[Tuple[pd.Series, pd.Series]]:
        """Retrieve cached labels."""
        key = self._get_key(params)

        # Check memory cache first
        if key in self.cache:
            return self.cache[key]

        # Check disk cache
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                self.cache[key] = data  # Load into memory
                return data
            except Exception as e:
                tprint(f"⚠️ Cache load failed: {e}", "WARNING")
                return None

        return None

    def put(self, params: Dict[str, Any], labels: Tuple[pd.Series, pd.Series]):
        """Store labels in cache."""
        key = self._get_key(params)

        # Store in memory
        self.cache[key] = labels

        # Store on disk
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(labels, f)
        except Exception as e:
            tprint(f"⚠️ Cache save failed: {e}", "WARNING")


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

        tprint(f"🚀 [0/13] Starting enhanced meta-labeling for {config['symbol']} ({config.get('timeframe', 'N/A')})", "INFO")

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

            used_hpo_params = False

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
                        used_hpo_params = True
                    else:
                        tprint("ℹ️ No HPO best-params file found; using configured/default labeling parameters", "INFO")
                except Exception as hpo_exc:
                    tprint(f"⚠️ Failed to load labeling HPO params: {hpo_exc}", "WARNING")

            # Load market data
            tprint("📊 [prep] Loading market data...", "INFO")
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

            try:
                market_data = attach_rolling_hmm_regimes_to_market_data(
                    self,
                    market_data,
                    config,
                )
            except Exception as e_reg:
                tprint(f"⚠️ Failed to attach rolling HMM regimes to market_data: {e_reg}", "WARNING")

            # STEP 1: Generate FIXED primary signals
            tprint("🎯 [1/13] Generating fixed primary signals...", "INFO")
            primary_signals = generate_primary_signals(market_data)

            n_long_signals = (primary_signals['consensus'] > 0).sum()
            n_short_signals = (primary_signals['consensus'] < 0).sum()
            tprint(f"📊 Primary signals: {n_long_signals} long, {n_short_signals} short", "INFO")

            # STEP 2: Compute volatility for adaptive thresholds
            tprint("📊 [2/13] Computing volatility for adaptive thresholds...", "INFO")
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
            tprint("💰 [3/13] Computing realized returns with adaptive thresholds and transaction costs...", "INFO")
            realized_returns, binary_labels, exit_reasons, event_durations, mfe_series, mae_series = compute_realized_returns(
                market_data,
                primary_signals,
                profit_threshold=adaptive_profit_threshold,
                stop_threshold=adaptive_stop_threshold,
                horizon=horizon,
                transaction_cost=transaction_cost,
                min_event_spacing=min_event_spacing,
                volatility_series=volatility_1d,  # Enable dynamic horizon based on volatility
            )

            # NEW: Volatility-scaled returns and quantile-based labels to improve
            # balance and focus labels on economically meaningful moves.
            tprint("📊 [3b/13] Computing volatility-scaled returns and quantile-based labels...", "INFO")
            vol_scaled_returns = compute_vol_scaled_returns_for_events(
                realized_returns=realized_returns,
                volatility=volatility_1d,
            )

            quantile_low_q = float(config.get("quantile_low_q", 0.3))
            quantile_high_q = float(config.get("quantile_high_q", 0.7))

            regimes_for_labeling = None
            if config.get("enable_regime_aware_quantiles", True) and "hmm_regime_label_1h" in market_data.columns:
                regimes_for_labeling = market_data["hmm_regime_label_1h"]

            if regimes_for_labeling is not None:
                quantile_labels = create_regime_aware_quantile_labels_from_vol_scaled_returns(
                    vol_scaled=vol_scaled_returns,
                    regimes=regimes_for_labeling,
                    low_q=quantile_low_q,
                    high_q=quantile_high_q,
                )
            else:
                quantile_labels = create_quantile_labels_from_vol_scaled_returns(
                    vol_scaled=vol_scaled_returns,
                    low_q=quantile_low_q,
                    high_q=quantile_high_q,
                )

            # Always use quantile-based labels for meta-labeling. If they are
            # very sparse, downstream diagnostics will reflect that directly.
            binary_labels = quantile_labels

            # Statistics (based on the effective binary_labels used for training)
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
            tprint("📈 [4/13] Applying Kalman smoothing to binary labels...", "INFO")
            smoothed_labels, label_uncertainty = kalman_smooth_labels(
                binary_labels,
                Q=kalman_Q,
                R=kalman_R,
                volatility=volatility_1d
            )
            tprint(f"📊 Smoothed labels: Mean={smoothed_labels[labeled_mask].mean():.3f}, Std={smoothed_labels[labeled_mask].std():.3f}", "INFO")

            meta_feature_cfg_raw = config.get('meta_feature_engineering', {})
            meta_feature_cfg = dict(meta_feature_cfg_raw) if isinstance(meta_feature_cfg_raw, dict) else {}
            meta_feature_cfg["_label_uncertainty"] = label_uncertainty

            meta_features, meta_features_model_processed, selected_feature_names, sample_weights = build_meta_features_for_model(
                market_data=market_data,
                primary_signals=primary_signals,
                realized_returns=realized_returns,
                binary_labels=binary_labels,
                event_durations=event_durations,
                mfe_series=mfe_series,
                mae_series=mae_series,
                adaptive_stop_threshold=adaptive_stop_threshold,
                horizon=horizon,
                volume_available=volume_available,
                meta_feature_cfg=meta_feature_cfg,
            )

            # STEP 6: Train ensemble meta-models with K-fold cross-fitting
            tprint("🎓 [6/13] Training ensemble meta-models (LGBM + LogReg + RF) with purged K-fold CV...", "INFO")

            # Train ensemble and get OOF predictions
            trained_models, oof_predictions_df = train_ensemble_with_kfold(
                X=meta_features_model_processed,
                y=binary_labels,
                horizon=horizon,
                n_splits=5,
                sample_weights=sample_weights,
                verbose=True
            )

            # STEP 7: Add signal disagreement feature
            tprint("🔧 [7/13] Adding signal disagreement feature...", "INFO")
            meta_features_enhanced = add_signal_disagreement(
                oof_predictions=oof_predictions_df,
                meta_features=meta_features
            )

            # STEP 8: Calibrate ensemble with isotonic regression only (preserve variance)
            tprint("📈 [8/13] Calibrating ensemble (isotonic on blended predictions)...", "INFO")

            platt_calibrators, iso_regressor = calibrate_ensemble(
                oof_predictions=oof_predictions_df,
                y_true=binary_labels,
                realized_returns=realized_returns,
                meta_features=meta_features_enhanced,
                method='isotonic_only',
                include_context=False
            )

            # STEP 9: Generate final ensemble predictions
            tprint("🎯 [9/13] Generating final ensemble predictions...", "INFO")

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
            tprint("🎓 [10/13] Training final ensemble models on full dataset...", "INFO")

            full_mask = ~binary_labels.isna()
            meta_features_enhanced_model = prepare_feature_matrix(meta_features_enhanced)

            train_columns = list(meta_features_enhanced_model.columns)
            try:
                extra_cols = [c for c in meta_features_enhanced_model.columns if c.startswith('signal_disagreement')]
                base_selected = selected_feature_names if 'selected_feature_names' in locals() else train_columns
                train_columns = [c for c in meta_features_enhanced_model.columns if c in base_selected or c in extra_cols]
                if not train_columns:
                    train_columns = list(meta_features_enhanced_model.columns)
            except Exception:
                train_columns = list(meta_features_enhanced_model.columns)

            X_full = meta_features_enhanced_model[train_columns]
            if meta_feature_cfg.get('enable_winsorisation', False):
                try:
                    lower_q = float(meta_feature_cfg.get('winsor_lower_q', 0.01))
                    upper_q = float(meta_feature_cfg.get('winsor_upper_q', 0.99))
                    robust_window = int(meta_feature_cfg.get('robust_window', 256))
                    robust_min_periods = int(meta_feature_cfg.get('robust_min_periods', max(1, robust_window // 4)))

                    X_full = rolling_robust_scale_features(
                        pd.DataFrame(X_full),
                        window=robust_window,
                        min_periods=robust_min_periods,
                        skip_binary=True,
                        skip_low_cardinality_int=True,
                    )
                    X_full = winsorize_features(
                        X_full,
                        lower_quantile=lower_q,
                        upper_quantile=upper_q,
                    )
                except Exception as e_w_full:
                    tprint(f"⚠️ Winsorisation for final models failed, using raw features: {e_w_full}", "WARNING")

            X_full = pd.DataFrame(X_full)[full_mask].fillna(0)
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
            tprint("🔄 [11/13] Translating probabilities to economic targets...", "INFO")

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
            tprint("📦 [12/13] Creating output DataFrame...", "INFO")
            labeled_data = market_data.copy()

            # Add log returns and volatility (for diagnostics)
            labeled_data['log_ret'] = log_ret
            labeled_data['volatility_1d'] = volatility_1d

            # Store volatility-scaled returns for diagnostics
            try:
                labeled_data['vol_scaled_return'] = vol_scaled_returns
            except Exception:
                labeled_data['vol_scaled_return'] = np.nan

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

                # Exploit ensemble disagreement as anti-signal for downstream training weights
                try:
                    if 'meta_features_enhanced' in locals() and isinstance(meta_features_enhanced, pd.DataFrame):
                        if 'signal_disagreement' in meta_features_enhanced.columns:
                            dis = pd.to_numeric(meta_features_enhanced['signal_disagreement'], errors="coerce")
                            dis = dis.reindex(labeled_data.index)
                            if dis.notna().any():
                                dis_filled = dis.fillna(dis.median())
                                dis_norm = (dis_filled - dis_filled.min()) / (dis_filled.max() - dis_filled.min() + 1e-8)
                                # High disagreement → stronger down-weight (≈0.5–1.0)
                                dis_factor = 1.0 - 0.5 * dis_norm.clip(0.0, 1.0)
                                sample_weight = sample_weight * dis_factor
                except Exception as w_disc_exc:
                    tprint(f"⚠️ Disagreement-based weighting failed, keeping base sample_weight: {w_disc_exc}", "WARNING")

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

            # Attach schema version for downstream consumers
            labeled_data['labeled_data_schema_version'] = LABELED_DATA_SCHEMA_VERSION

            timestamp_ns = np.int64(pd.Timestamp.utcnow().value)
            labeled_data['labeling_timestamp'] = timestamp_ns
            labeled_data['labeling_method_id'] = np.int8(2)

            # Save labeled data
            tprint("💾 [12b/13] Saving labeled data...", "INFO")

            # Guard: ensure required columns are present before persisting
            validate_labeled_data_schema(
                labeled_data,
                required_cols=get_required_labeled_data_columns(
                    [
                        'meta_probability',
                        'event_duration_bars',
                    ]
                ),
                context='FeatureGenerationMetaLabelingStep',
            )

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
            tprint("📊 [13/13] Generating diagnostics report...", "INFO")

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
                    output_dir=outcomes_dir,
                    exit_reasons=exit_reasons,
                    event_durations=event_durations,
                    mfe_series=mfe_series,
                    mae_series=mae_series,
                    target_long=target_long,
                    target_short=target_short
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

            # ------------------------------------------------------------------
            # Save isotonic regressor + meta-gating config for live trading
            # ------------------------------------------------------------------
            try:
                symbol = config.get('symbol', 'UNKNOWN')
                exchange = config.get('exchange', 'binance')
                timeframe = config.get('timeframe', '15m')
                direction = config.get('direction', 'long')

                if iso_regressor is not None:
                    # Versioned artifact directory for analyst models
                    va_dir = Path("versioned_artifacts") / f"{symbol}_{exchange}_{timeframe}_{direction}_analyst"
                    artifacts_dir = va_dir / "artifacts"
                    artifacts_dir.mkdir(parents=True, exist_ok=True)

                    # Save isotonic regressor artifact
                    iso_rel_path = "artifacts/iso_regressor_analyst_meta.pkl"
                    iso_path = va_dir / iso_rel_path
                    try:
                        with open(iso_path, "wb") as f_iso:
                            pickle.dump(iso_regressor, f_iso)
                        tprint(f"💾 Saved isotonic regressor artifact to {iso_path}", "INFO")
                    except Exception as iso_exc:
                        tprint(f"⚠️ Could not save isotonic regressor artifact: {iso_exc}", "WARNING")

                    # Derive simple meta-gating thresholds from OOF probabilities + realized returns
                    try:
                        event_mask = ~realized_returns.isna()
                        n_events = int(event_mask.sum())

                        best_cfg = None
                        if n_events >= 20:
                            # Use ensemble_probs_series (pandas Series) for safe masking
                            p_series = ensemble_probs_series.loc[event_mask].astype(float)
                            r_series = realized_returns.loc[event_mask].astype(float)

                            p_array = p_series.to_numpy()
                            r_array = r_series.to_numpy()

                            # Expected returns from isotonic mapping (if possible)
                            try:
                                E_hat_array = iso_regressor.predict(p_array)
                            except Exception:
                                E_hat_array = None

                            prob_thresholds = [0.55, 0.60, 0.65, 0.70, 0.75]
                            er_multipliers = [1.0, 2.0, 3.0]
                            tx_cost = float(transaction_cost)

                            for p_thr in prob_thresholds:
                                for k in er_multipliers:
                                    # If isotonic mapping is not available, fall back to prob-only gating
                                    E_thr = tx_cost * k if E_hat_array is not None else 0.0

                                    gate = p_array >= p_thr
                                    if E_hat_array is not None:
                                        gate &= (E_hat_array >= E_thr)

                                    gated_r = r_array[gate]
                                    n_trades = int(len(gated_r))
                                    if n_trades < 10:
                                        continue

                                    mean_r = float(np.mean(gated_r))
                                    std_r = float(np.std(gated_r, ddof=1)) if n_trades > 1 else 0.0
                                    sharpe = float(mean_r / std_r) if std_r > 0 else 0.0

                                    # Score: prefer higher Sharpe and more trades
                                    score = sharpe * np.sqrt(max(n_trades, 1))

                                    if (best_cfg is None) or (score > best_cfg["score"]):
                                        best_cfg = {
                                            "prob_threshold": float(p_thr),
                                            "expected_return_threshold": float(E_thr),
                                            "mean_return": mean_r,
                                            "sharpe": sharpe,
                                            "n_trades": n_trades,
                                            "score": float(score),
                                        }

                        # Fallback if no valid configuration found
                        if best_cfg is None:
                            best_cfg = {
                                "prob_threshold": 0.60,
                                "expected_return_threshold": float(transaction_cost * 2.0),
                                "mean_return": 0.0,
                                "sharpe": 0.0,
                                "n_trades": 0,
                                "score": 0.0,
                            }

                        regime_gating = {}
                        try:
                            if "hmm_regime_label_1h" in labeled_data.columns:
                                reg_all_events = labeled_data.loc[event_mask, "hmm_regime_label_1h"]
                                unique_regs = pd.unique(reg_all_events.dropna())
                                for reg_val in unique_regs:
                                    try:
                                        reg_mask = (reg_all_events == reg_val).to_numpy()
                                        n_reg_events = int(reg_mask.sum())
                                        if n_reg_events < 20:
                                            continue
                                        best_reg_cfg = None
                                        for p_thr in prob_thresholds:
                                            for k in er_multipliers:
                                                E_thr_reg = tx_cost * k if E_hat_array is not None else 0.0
                                                gate_reg = p_array >= p_thr
                                                if E_hat_array is not None:
                                                    gate_reg &= (E_hat_array >= E_thr_reg)
                                                gate_reg &= reg_mask
                                                gated_r_reg = r_array[gate_reg]
                                                n_trades_reg = int(len(gated_r_reg))
                                                if n_trades_reg < 10:
                                                    continue
                                                mean_r_reg = float(np.mean(gated_r_reg))
                                                std_r_reg = float(np.std(gated_r_reg, ddof=1)) if n_trades_reg > 1 else 0.0
                                                sharpe_reg = float(mean_r_reg / std_r_reg) if std_r_reg > 0 else 0.0
                                                score_reg = sharpe_reg * np.sqrt(max(n_trades_reg, 1))
                                                if (best_reg_cfg is None) or (score_reg > best_reg_cfg["score"]):
                                                    best_reg_cfg = {
                                                        "prob_threshold": float(p_thr),
                                                        "expected_return_threshold": float(E_thr_reg),
                                                        "mean_return": mean_r_reg,
                                                        "sharpe": sharpe_reg,
                                                        "n_trades": n_trades_reg,
                                                        "score": float(score_reg),
                                                    }
                                        if best_reg_cfg is not None:
                                            regime_gating[str(reg_val)] = best_reg_cfg
                                    except Exception:
                                        continue
                        except Exception:
                            regime_gating = {}

                        # Build meta-gating configuration payload
                        meta_gating_config = {
                            "symbol": symbol,
                            "exchange": exchange,
                            "timeframe": timeframe,
                            "direction": direction,
                            "model_family": "analyst_meta",
                            "meta_gating": {
                                "version": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
                                "transaction_cost": float(transaction_cost),
                                "entry": {
                                    "prob_threshold": best_cfg["prob_threshold"],
                                    "use_expected_return": bool(best_cfg["expected_return_threshold"] > 0),
                                    "expected_return_threshold": best_cfg["expected_return_threshold"],
                                    "expected_return_unit": "fraction",
                                    "min_trades": int(best_cfg["n_trades"]),
                                },
                                "calibration": {
                                    "iso_regressor_artifact": iso_rel_path,
                                    "fitted_on": "train_oof",
                                },
                                # Expose the triple-barrier configuration used during labeling
                                # so live trading can align TPSL/horizon with the same setup.
                                "triple_barrier": {
                                    "profit_threshold": float(profit_threshold),
                                    "stop_threshold": float(stop_threshold),
                                    "horizon_bars": int(horizon),
                                    "min_event_spacing": int(min_event_spacing),
                                },
                                "backtest_metrics": {
                                    "auc_oof": float(avg_auc),
                                    "mean_return_gated": float(best_cfg["mean_return"]),
                                    "sharpe_gated": float(best_cfg["sharpe"]),
                                    "trades_gated": int(best_cfg["n_trades"]),
                                },
                                "regime_specific": regime_gating,
                            },
                        }

                        gating_path = va_dir / "meta_gating_config.json"
                        with open(gating_path, "w") as f_gate:
                            json.dump(meta_gating_config, f_gate, indent=2)

                        tprint(f"💾 Saved meta-gating config to {gating_path}", "INFO")
                    except Exception as gate_exc:
                        tprint(f"⚠️ Could not compute/save meta-gating config: {gate_exc}", "WARNING")

            except Exception as e:
                tprint(f"⚠️ Meta-gating artifact saving failed: {e}", "WARNING")

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
                        'ensemble_models': ['lgbm', 'xgb', 'rf'],
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

            tprint(f"✅ [done] Enhanced meta-labeling completed in {elapsed_time:.1f}s", "SUCCESS")
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
