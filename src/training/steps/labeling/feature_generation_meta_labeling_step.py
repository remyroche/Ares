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

# CatBoost for ensemble (with graceful fallback)
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    warnings.warn("CatBoost not available - ensemble will use LightGBM/XGBoost/RF only")

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
from src.utils.ml_common.get_specialist_models_outputs import get_specialist_models_outputs
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
from src.feature_generation.utils.step06_labeling_components.trend_aware_meta_labeling import (
    TrendAwareMetaLabeler,
    MultiTimeframeConfig,
)

logger = logging.getLogger(__name__)


def _load_latest_labeling_hpo_params(
    symbol: str,
    timeframe: str,
) -> Tuple[Dict[str, Any], Optional[Path], str]:
    """Load latest HPO parameters from outcomes directory.

    Args:
        symbol: Trading symbol (e.g., 'ETHUSDT')
        timeframe: Timeframe (e.g., '15m')

    Returns:
        Tuple of (params_dict, file_path, source_key)
        - params_dict: HPO parameters (knee_params preferred, fallback to best_params)
        - file_path: Path to JSON file (or None if not found)
        - source_key: Either 'knee_params' or 'best_params' to indicate source
    """
    outcomes_dir = Path("outcomes")
    if not outcomes_dir.exists():
        return {}, None, ""

    pattern = f"meta_labeling_hpo_best_params_{symbol}_{timeframe}_*.json"
    candidates = sorted(outcomes_dir.glob(pattern))
    if not candidates:
        return {}, None, ""

    latest = candidates[-1]
    with open(latest, "r") as f:
        hpo_cfg = json.load(f)

    params: Dict[str, Any] = {}
    source_key = ""
    if isinstance(hpo_cfg, dict):
        knee = hpo_cfg.get("knee_params")
        best = hpo_cfg.get("best_params")
        if isinstance(knee, dict) and knee:
            params = knee
            source_key = "knee_params"
        elif isinstance(best, dict) and best:
            params = best
            source_key = "best_params"

    return params, latest, source_key


def create_triple_barrier_from_hpo(
    symbol: str,
    timeframe: str,
    fallback_profit_take: float = 0.004,
    fallback_stop_loss: float = 0.003,
    fallback_time_barrier: int = 30,
    fallback_max_lookahead: int = 100,
    binary_classification: bool = True,
    transaction_cost: float = 0.0008,
) -> Tuple[Any, Dict[str, Any], bool]:
    """
    Create OptimizedTripleBarrierLabeling instance aligned with HPO results.

    This function provides programmatic alignment between meta_labeling_hpo_experiment
    results and triple barrier configuration. It loads the best HPO parameters and
    creates a properly configured triple barrier labeler.

    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        fallback_profit_take: Fallback profit take multiplier if HPO not found (default: 0.4%)
        fallback_stop_loss: Fallback stop loss multiplier if HPO not found (default: 0.3%)
        fallback_time_barrier: Fallback time barrier in minutes (default: 30)
        fallback_max_lookahead: Fallback max lookahead bars (default: 100)
        binary_classification: Whether to use binary classification (default: True)
        transaction_cost: Transaction cost as percentage (default: 0.08%)

    Returns:
        Tuple of (labeler, hpo_params, used_hpo)
        - labeler: Configured OptimizedTripleBarrierLabeling instance
        - hpo_params: Dictionary of HPO parameters (or empty dict if not found)
        - used_hpo: Boolean indicating whether HPO params were found and used
    """
    from src.feature_generation.utils.step06_labeling_components.optimized_triple_barrier_labeling import (
        OptimizedTripleBarrierLabeling,
    )

    # Try to load HPO parameters
    hpo_params, latest_path, params_source = _load_latest_labeling_hpo_params(symbol, timeframe)

    used_hpo = False
    profit_take = fallback_profit_take
    stop_loss = fallback_stop_loss
    time_barrier = fallback_time_barrier
    max_lookahead = fallback_max_lookahead

    if latest_path is not None and hpo_params:
        tprint(f"🔍 [TripleBarrier] Found HPO params from {latest_path} (source: {params_source})", "INFO")
        used_hpo = True

        # Extract and convert HPO parameters to triple barrier settings
        if 'profit_thr_base' in hpo_params:
            profit_take = float(hpo_params['profit_thr_base'])
            tprint(f"  ✓ profit_take_multiplier: {profit_take:.4f} ({profit_take*100:.2f}%)", "INFO")

        if 'stop_to_profit_ratio' in hpo_params and 'profit_thr_base' in hpo_params:
            stop_ratio = float(hpo_params['stop_to_profit_ratio'])
            stop_loss = profit_take * stop_ratio
            # Ensure minimum stop loss
            stop_loss = max(0.0005, stop_loss)
            tprint(f"  ✓ stop_loss_multiplier: {stop_loss:.4f} ({stop_loss*100:.2f}%) [ratio: {stop_ratio:.2f}]", "INFO")

        if 'horizon_bars' in hpo_params:
            horizon_bars = int(hpo_params['horizon_bars'])
            # Convert horizon bars to approximate time barrier in minutes (assuming 15m timeframe)
            # This is a heuristic - adjust if using different timeframes
            timeframe_minutes = 15  # Default assumption
            if timeframe.endswith('m'):
                try:
                    timeframe_minutes = int(timeframe[:-1])
                except:
                    pass
            time_barrier = min(horizon_bars * timeframe_minutes, 240)  # Cap at 4 hours
            tprint(f"  ✓ time_barrier_minutes: {time_barrier} (from horizon_bars={horizon_bars})", "INFO")

        # Use horizon_bars as max_lookahead directly
        if 'horizon_bars' in hpo_params:
            max_lookahead = int(hpo_params['horizon_bars'])
            # Ensure reasonable bounds
            max_lookahead = max(10, min(max_lookahead, 200))
            tprint(f"  ✓ max_lookahead: {max_lookahead} bars", "INFO")
    else:
        tprint(f"ℹ️ [TripleBarrier] No HPO params found for {symbol}_{timeframe}, using fallback settings", "INFO")
        tprint(f"  → profit_take: {profit_take:.4f}, stop_loss: {stop_loss:.4f}", "INFO")
        tprint(f"  → time_barrier: {time_barrier}min, max_lookahead: {max_lookahead}", "INFO")

    # Create the labeler with aligned parameters
    try:
        labeler = OptimizedTripleBarrierLabeling(
            profit_take_multiplier=profit_take,
            stop_loss_multiplier=stop_loss,
            time_barrier_minutes=time_barrier,
            max_lookahead=max_lookahead,
            binary_classification=binary_classification,
            transaction_cost=transaction_cost,
        )

        if used_hpo:
            tprint(f"✅ [TripleBarrier] Created labeler aligned with HPO parameters", "SUCCESS")
        else:
            tprint(f"✅ [TripleBarrier] Created labeler with fallback parameters", "SUCCESS")

        return labeler, hpo_params, used_hpo

    except Exception as e:
        tprint(f"❌ [TripleBarrier] Failed to create labeler: {e}", "ERROR")
        tprint(f"   Using very conservative fallback settings", "WARNING")
        # Emergency fallback with very conservative settings
        labeler = OptimizedTripleBarrierLabeling(
            profit_take_multiplier=0.005,
            stop_loss_multiplier=0.004,
            time_barrier_minutes=30,
            max_lookahead=50,
            binary_classification=binary_classification,
            transaction_cost=transaction_cost,
        )
        return labeler, {}, False

# Production TPSL Parameters (overridable via config)
DEFAULT_PROFIT_THRESHOLD = 0.01  # 1%
DEFAULT_STOP_THRESHOLD = 0.005   # 0.5%
DEFAULT_TRANSACTION_COST = 0.003  # 0.30% per trade (increased from 0.15% for more realistic modeling)
R_MULTIPLE_POS_THRESHOLD = 0.7
R_MULTIPLE_NEG_THRESHOLD = -0.25
ECON_MIN_RETURN_MULTIPLE = 2.0
TARGET_POWER = 1.5
# Hard floor for profit targets to ensure viability after transaction costs
PROFIT_TARGET_FLOOR_BPS = 50  # 0.5% = 50 basis points (must exceed slippage + fees)
PROFITABLE_TIMEOUT_RETURN_THRESHOLD = 0.005
# Default probability threshold for meta-gating
DEFAULT_PROBABILITY_THRESHOLD = 0.60
# Default expected return threshold (lowered from 0.45% to 0.30%)
DEFAULT_EXPECTED_RETURN_THRESHOLD = 0.003  # 0.30%


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
    rsi_oversold: float = 30.0,  
    rsi_overbought: float = 70.0,  # LOOSER (was 70)
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    macd_fast_long: int = 48,  # 4x longer
    macd_slow_long: int = 104,  # 4x longer
    macd_signal_long: int = 36,  # 4x longer
    macd_threshold: float = 0.02,  # LOOSER difference threshold
    momentum_threshold: Optional[float] = None,  # If None, will be auto-tuned
    target_trades_per_day: float = 4.0,  # Target signal density (increased from 2.0 for more signals)
    enable_dynamic_tuning: bool = True,  # Enable auto-tuning of momentum threshold
    use_cusum_filter: bool = True,  # Use CUSUM filter instead of momentum threshold
    cusum_threshold: float = 0.015,  # CUSUM threshold for event detection
    # New parameters for enhanced signals
    bb_window: int = 20,  # Bollinger Band window
    bb_std: float = 2.0,  # Bollinger Band standard deviations
    atr_period: int = 14,  # ATR period for breakout signals
    atr_mult: float = 1.5,  # ATR multiplier for breakout threshold
    volume_spike_threshold: float = 2.0,  # Volume spike threshold (multiples of mean)
    range_window: int = 48,  # Range window for fade signals (12 hours at 15m)
    mtf_lookback: int = 4,  # Multi-timeframe lookback (4 bars = 1 hour at 15m)
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
        low_thresh, high_thresh = 0.001, 0.2  # Search range
        best_thresh = 0.01  # Fallback
        tolerance = 0.15  # Accept within 15% of target

        for iteration in range(15):  # Max 15 iterations
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

            if iteration == 14:  # Last iteration
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

    # ===== NEW SIGNALS: BOLLINGER BAND FADE =====
    bb_mid = df_local['close'].rolling(bb_window).mean()
    bb_std_series = df_local['close'].rolling(bb_window).std()
    bb_upper = bb_mid + bb_std * bb_std_series
    bb_lower = bb_mid - bb_std * bb_std_series

    signals['bb_fade'] = 0
    # Long when price touches/crosses lower band (mean-reversion signal)
    signals.loc[df_local['close'] <= bb_lower, 'bb_fade'] = 1
    # Short when price touches/crosses upper band (mean-reversion signal)
    signals.loc[df_local['close'] >= bb_upper, 'bb_fade'] = -1

    # Store BB values for features
    signals['bb_upper'] = bb_upper
    signals['bb_lower'] = bb_lower
    signals['bb_mid'] = bb_mid
    signals['bb_width'] = (bb_upper - bb_lower) / (bb_mid + 1e-8)

    # ===== NEW SIGNALS: ATR BREAKOUT =====
    atr_raw = (df_local['high'] - df_local['low']).rolling(atr_period).mean()
    close_change = df_local['close'].diff()
    atr_breakout_threshold = atr_mult * atr_raw

    signals['atr_breakout'] = 0
    # Long on strong upward breakout
    signals.loc[close_change > atr_breakout_threshold, 'atr_breakout'] = 1
    # Short on strong downward breakout
    signals.loc[close_change < -atr_breakout_threshold, 'atr_breakout'] = -1

    # ===== NEW SIGNALS: VOLUME SPIKE =====
    if 'volume' in df_local.columns:
        vol_mean = df_local['volume'].rolling(96).mean()  # 1-day mean
        vol_ratio = df_local['volume'] / (vol_mean + 1e-8)
        price_direction = np.sign(df_local['close'].diff())

        signals['volume_spike'] = 0
        # Volume spike in direction of price move
        spike_mask = vol_ratio > volume_spike_threshold
        signals.loc[spike_mask & (price_direction > 0), 'volume_spike'] = 1
        signals.loc[spike_mask & (price_direction < 0), 'volume_spike'] = -1
        signals['volume_ratio_signal'] = vol_ratio
    else:
        signals['volume_spike'] = 0
        signals['volume_ratio_signal'] = 1.0

    # ===== NEW SIGNALS: RANGE FADE (Mean-Reversion at Range Extremes) =====
    range_high = df_local['high'].rolling(range_window).max()
    range_low = df_local['low'].rolling(range_window).min()
    range_mid = (range_high + range_low) / 2
    range_position = (df_local['close'] - range_low) / (range_high - range_low + 1e-8)

    signals['range_fade'] = 0
    # Long at bottom of range (mean-reversion)
    signals.loc[range_position < 0.15, 'range_fade'] = 1
    # Short at top of range (mean-reversion)
    signals.loc[range_position > 0.85, 'range_fade'] = -1
    signals['range_position'] = range_position

    # ===== NEW SIGNALS: RSI MEAN-REVERSION (tighter thresholds for low-vol) =====
    signals['rsi_mr'] = 0
    # Less extreme thresholds than momentum RSI (35/65 vs 25/75)
    signals.loc[df_local['rsi'] < 35, 'rsi_mr'] = 1
    signals.loc[df_local['rsi'] > 65, 'rsi_mr'] = -1

    # ===== NEW SIGNALS: MULTI-TIMEFRAME CONFLUENCE =====
    # Use higher timeframe signals (aggregated from current data)
    # 4-bar lookback = 1 hour at 15m timeframe
    close_mtf = df_local['close'].rolling(mtf_lookback).mean()
    momentum_mtf = close_mtf.pct_change(mtf_lookback * 2)  # 2-hour momentum

    signals['mtf_trend'] = 0
    signals.loc[momentum_mtf > 0.005, 'mtf_trend'] = 1  # Bullish MTF
    signals.loc[momentum_mtf < -0.005, 'mtf_trend'] = -1  # Bearish MTF

    # MTF confluence: current signal agrees with higher timeframe
    current_momentum = df_local['close'].pct_change(momentum_period)
    signals['mtf_confluence'] = 0
    # Strong long: short-term momentum up AND MTF momentum up
    signals.loc[(current_momentum > 0) & (momentum_mtf > 0), 'mtf_confluence'] = 1
    # Strong short: short-term momentum down AND MTF momentum down
    signals.loc[(current_momentum < 0) & (momentum_mtf < 0), 'mtf_confluence'] = -1

    # ===== VOL-AWARE DUAL-MODE CONSENSUS =====
    # Separate signal types into momentum and mean-reversion categories
    momentum_cols = ['rsi', 'rsi_long', 'macd', 'macd_long', 'ma', 'mom', 'atr_breakout', 'volume_spike', 'mtf_trend', 'mtf_confluence']
    mr_cols = ['bb_fade', 'range_fade', 'rsi_mr']

    # Ensure all columns exist
    for col in momentum_cols + mr_cols:
        if col not in signals.columns:
            signals[col] = 0

    # Calculate scores for each signal type
    momentum_score = signals[momentum_cols].sum(axis=1)
    mr_score = signals[mr_cols].sum(axis=1)

    # Vol ratio for regime detection (using linear formula, not thresholds)
    vol_ratio = (vol_short / (vol_long + 1e-8)).fillna(1.0)

    # Linear vol-aware weighting:
    # vol_ratio < 0.7: favor mean-reversion (low vol, ranging market)
    # vol_ratio > 1.3: favor momentum (high vol, trending market)
    # Linear interpolation in between
    # momentum_weight = clip((vol_ratio - 0.7) / 0.6, 0.2, 0.9)
    # This gives: vol_ratio=0.7 → mom_weight=0.2, vol_ratio=1.3 → mom_weight=0.9
    momentum_weight = np.clip((vol_ratio - 0.7) / 0.6, 0.2, 0.9)
    mr_weight = 1.0 - momentum_weight

    # Count raw signals for funnel
    all_signal_cols = momentum_cols + mr_cols
    raw_consensus = signals[all_signal_cols].sum(axis=1).apply(np.sign)
    raw_signal_count = int((raw_consensus != 0).sum())
    funnel['raw_signals'] = raw_signal_count
    funnel['momentum_signals'] = int((momentum_score != 0).sum())
    funnel['mr_signals'] = int((mr_score != 0).sum())

    # Weighted consensus: blend momentum and mean-reversion based on vol regime
    weighted_score = momentum_weight * momentum_score + mr_weight * mr_score
    strict_consensus = weighted_score.apply(np.sign)

    # Relax consensus slightly: where the strict vol-weighted consensus is 0 but
    # there is a clear raw consensus, fall back to the raw consensus direction.
    consensus_relaxed = strict_consensus.copy()
    relaxed_mask = (consensus_relaxed == 0) & (raw_consensus != 0)
    consensus_relaxed[relaxed_mask] = raw_consensus[relaxed_mask]
    signals['consensus'] = consensus_relaxed

    # Store diagnostic info
    signals['momentum_weight'] = momentum_weight
    signals['mr_weight'] = mr_weight
    signals['vol_ratio_for_consensus'] = vol_ratio
    signals['momentum_score'] = momentum_score
    signals['mr_score'] = mr_score

    # SIGNAL FUNNEL LOGGING
    final_signal_count = int((signals['consensus'] != 0).sum())
    funnel['final_signals'] = final_signal_count
    funnel['raw_long_signals'] = int((raw_consensus > 0).sum())
    funnel['raw_short_signals'] = int((raw_consensus < 0).sum())
    funnel['final_long_signals'] = int((signals['consensus'] > 0).sum())
    funnel['final_short_signals'] = int((signals['consensus'] < 0).sum())
    funnel['relaxed_extra_signals'] = int(relaxed_mask.sum())
    funnel['raw_to_final_ratio'] = float(final_signal_count) / max(raw_signal_count, 1)

    # Attach funnel statistics to the signal DataFrame for downstream diagnostics
    try:
        signals.attrs['signal_funnel'] = funnel
    except Exception:
        pass

    tprint(f"📊 Signal Funnel (Vol-Aware Dual-Mode):", "INFO")
    tprint(f"  Total bars: {funnel['total_bars']}", "INFO")
    tprint(f"  Raw signals generated: {funnel['raw_signals']}", "INFO")
    tprint(f"  Final consensus signals: {funnel['final_signals']} (ratio={funnel['raw_to_final_ratio']:.3f})", "INFO")
    tprint(f"  Long/short raw: {funnel['raw_long_signals']}/{funnel['raw_short_signals']}", "INFO")
    tprint(f"  Long/short final: {funnel['final_long_signals']}/{funnel['final_short_signals']}", "INFO")
    tprint(f"  Relaxed extra signals (strict=0 but raw≠0): {funnel['relaxed_extra_signals']}", "INFO")
    tprint(f"  ℹ️  Using linear vol-aware weighting: low-vol favors MR, high-vol favors momentum", "INFO")

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

    body = df_local['close'] - df_local.get('open', df_local['close'])
    high_low_range = (df_local['high'] - df_local['low']).replace(0.0, np.nan)
    upper_wick = df_local['high'] - df_local[['open', 'close']].max(axis=1)
    lower_wick = df_local[['open', 'close']].min(axis=1) - df_local['low']
    body_ratio = (body.abs() / (high_low_range + 1e-8)).fillna(0.0)

    signals['trend_regime'] = 0
    trend_window = 32
    trend_mean = df_local['close'].rolling(trend_window).mean()
    trend_slope = trend_mean.pct_change(max(trend_window // 2, 1))
    atr_lookback = 14
    atr_series = (df_local['high'] - df_local['low']).rolling(atr_lookback).mean()
    atr_norm = atr_series / (df_local['close'] + 1e-8)
    atr_threshold = atr_norm.median()
    slope_threshold = 0.003
    signals.loc[(trend_slope > slope_threshold) & (atr_norm > atr_threshold), 'trend_regime'] = 1
    signals.loc[(trend_slope < -slope_threshold) & (atr_norm > atr_threshold), 'trend_regime'] = -1

    signals['candle_trend'] = 0
    signals.loc[(body > 0) & (body_ratio > 0.6), 'candle_trend'] = 1
    signals.loc[(body < 0) & (body_ratio > 0.6), 'candle_trend'] = -1

    signals['candle_reversal'] = 0
    signals.loc[(body > 0) & (lower_wick > 2.0 * body.abs()) & (body_ratio < 0.4), 'candle_reversal'] = 1
    signals.loc[(body < 0) & (upper_wick > 2.0 * body.abs()) & (body_ratio < 0.4), 'candle_reversal'] = -1

    return signals


def compute_realized_returns(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    profit_threshold: Union[float, pd.Series] = 0.015,
    stop_threshold: Union[float, pd.Series] = 0.010,
    horizon: int = 16,
    transaction_cost: float = 0.0005,
    min_event_spacing: int = 2,
    volatility_series: Optional[pd.Series] = None,
    use_multiclass_labels: bool = False,  # NEW: 3-class labels (0=timeout, 1=profit, 2=stop)
    atr_series: Optional[pd.Series] = None,  # NEW: For trailing stops
    trail_distance_atr_mult: Optional[float] = None,  # NEW: Trailing distance in ATR
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Compute realized returns for each signal event.

    ENHANCED IMPROVEMENTS:
    - Uses High/Low prices for TP/SL checks (more realistic)
    - Adds velocity/efficiency penalty for slow trades
    - Dynamic horizon based on volatility (linear scaling, 2x max cap)
    - Tracks MFE/MAE for diagnostics
    - Supports adaptive thresholds based on volatility
    - NEW: Multi-class labels (0=timeout, 1=profit, 2=stop) for more nuanced learning
    - NEW: Directional binary labels (binary_labels_long, binary_labels_short) for training
           direction-specific classifiers
    - NEW: Simulated trailing profit (activates at TP, exits on reversal)

    Args:
        df: DataFrame with OHLCV data
        signals: DataFrame with signal columns
        profit_threshold: Profit target as fraction (float or Series for adaptive)
                          If trailing is enabled, this acts as the INITIAL activation threshold.
        stop_threshold: Stop loss as fraction (float or Series for adaptive)
        horizon: Base maximum bars to look ahead
        transaction_cost: Transaction cost per trade (round trip)
        min_event_spacing: Minimum bars between signals (prevents overlapping events)
        volatility_series: Volatility series for dynamic horizon scaling (optional)
        use_multiclass_labels: If True, returns 3-class labels (0=timeout, 1=profit, 2=stop)
                               If False, returns binary labels (0=loss/timeout, 1=profit)
        atr_series: ATR series for trailing stop calculation (optional)
        trail_distance_atr_mult: Trailing distance in ATR multiples (optional, enables trailing)

    Returns:
        Tuple of (realized_returns, binary_labels, exit_reasons, event_durations, 
                  mfe_series, mae_series, binary_labels_long, binary_labels_short)
        - realized_returns: Actual returns achieved (NaN where no signal)
        - binary_labels: Binary (0/1) or Multi-class (0/1/2) depending on use_multiclass_labels
        - exit_reasons: How each event exited ('profit', 'stop', 'timeout')
        - event_durations: Bars held for each event
        - mfe_series: Maximum Favorable Excursion for each event
        - mae_series: Maximum Adverse Excursion for each event
        - binary_labels_long: Binary success/failure for LONG trades only (NaN for shorts)
        - binary_labels_short: Binary success/failure for SHORT trades only (NaN for longs)
    """
    realized_returns = pd.Series(index=df.index, dtype=float)
    realized_returns[:] = np.nan

    binary_labels = pd.Series(index=df.index, dtype=float)
    binary_labels[:] = np.nan

    # NEW: Directional binary labels for training separate long/short classifiers
    # These allow training models specifically for longs or shorts without mixing signals
    binary_labels_long = pd.Series(index=df.index, dtype=float)
    binary_labels_long[:] = np.nan
    
    binary_labels_short = pd.Series(index=df.index, dtype=float)
    binary_labels_short[:] = np.nan

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

    # Prepare trailing stop arrays if enabled
    use_trailing = (atr_series is not None) and (trail_distance_atr_mult is not None) and (trail_distance_atr_mult > 0)
    # Ensure atr_values is aligned with df
    if use_trailing:
        if len(atr_series) != len(df):
            atr_values = atr_series.reindex(df.index).values
        else:
            atr_values = atr_series.values
    else:
        atr_values = None

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

        # Initialize trailing state for this event
        trailing_active = False
        peak_price = entry_price  # Initialize peak/trough
        # Pre-calculate trailing distance if trailing enabled
        if use_trailing:
            # Trailing distance = ATR at entry * mult
            current_atr = atr_values[i]
            if pd.isna(current_atr):
                # Fallback if ATR missing: use % of price approx or disable trailing
                # For safety, disable trailing for this specific event if ATR is missing
                event_trail_dist = 0.0
                event_use_trailing = False
            else:
                event_trail_dist = current_atr * trail_distance_atr_mult
                event_use_trailing = True
        else:
            event_trail_dist = 0.0
            event_use_trailing = False

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
                # Current P&L states
                pnl_high = (high_price - entry_price) / entry_price
                pnl_low = (low_price - entry_price) / entry_price

                # Track MFE/MAE
                max_favorable = max(max_favorable, pnl_high)
                max_adverse = min(max_adverse, pnl_low)

                # Determine effective stop level
                fixed_stop_price = entry_price * (1 - stop_thr)
                effective_stop = fixed_stop_price

                current_trailing_stop = -np.inf
                if event_use_trailing and trailing_active:
                    peak_price = max(peak_price, high_price)
                    current_trailing_stop = peak_price - event_trail_dist
                    effective_stop = max(fixed_stop_price, current_trailing_stop)
                    # With trailing enabled, never realize less than the base profit
                    min_profit_price_long = entry_price * (1 + profit_thr)
                    effective_stop = max(effective_stop, min_profit_price_long)

                # 1. Check Stop Hit (Highest priority exit)
                if low_price <= effective_stop:
                    exit_price = effective_stop
                    # If stop was tightened by trailing, reason is trailing. If stuck at fixed floor, reason is stop.
                    # Note: If trailing_stop > fixed_stop, reason is trailing.
                    # If trailing_active is false, effective_stop == fixed_stop, reason is stop.
                    exit_reason = 'trailing' if (effective_stop > fixed_stop_price + 1e-8) else 'stop'
                    event_end_idx = idx
                    break

                # 2. Check Activation (if not already active)
                if event_use_trailing and not trailing_active:
                    activation_price = entry_price * (1 + profit_thr)
                    if high_price >= activation_price:
                        # Intra-bar conflict check
                        # Assume we hit High (activation) first, then trail
                        peak_price = high_price
                        intra_bar_stop = peak_price - event_trail_dist
                        eff_intra_stop = max(fixed_stop_price, intra_bar_stop)

                        if low_price <= eff_intra_stop:
                            # Activated and stopped out in same bar
                            # Assume mid-point exit, but never below base profit
                            raw_exit_price = (high_price + low_price) / 2
                            min_profit_price_long = entry_price * (1 + profit_thr)
                            exit_price = max(raw_exit_price, min_profit_price_long)
                            exit_reason = 'trailing'
                            event_end_idx = idx
                            break
                        else:
                            # Activated and survived
                            trailing_active = True

                # 3. Standard Fixed Take Profit (if trailing disabled)
                if not event_use_trailing:
                    if pnl_high >= profit_thr:
                        exit_price = entry_price * (1 + profit_thr)
                        exit_reason = 'profit'
                        event_end_idx = idx
                        break

            elif signal < 0:  # Short signal
                # For shorts: check low for profit, high for stop
                pnl_high = (entry_price - high_price) / entry_price  # High is bad for shorts
                pnl_low = (entry_price - low_price) / entry_price  # Low is good for shorts

                # Track MFE/MAE
                max_favorable = max(max_favorable, pnl_low)
                max_adverse = min(max_adverse, pnl_high)

                # Determine effective stop level
                fixed_stop_price = entry_price * (1 + stop_thr)
                effective_stop = fixed_stop_price

                current_trailing_stop = np.inf
                if event_use_trailing and trailing_active:
                    peak_price = min(peak_price, low_price) # "peak" variable reused for trough
                    current_trailing_stop = peak_price + event_trail_dist
                    effective_stop = min(fixed_stop_price, current_trailing_stop)
                    # With trailing enabled, never realize less than the base profit
                    min_profit_price_short = entry_price * (1 - profit_thr)
                    effective_stop = min(effective_stop, min_profit_price_short)

                # 1. Check Stop Hit (Highest priority exit)
                if high_price >= effective_stop:
                    exit_price = effective_stop
                    # If stop was tightened (lowered) by trailing, reason is trailing.
                    exit_reason = 'trailing' if (effective_stop < fixed_stop_price - 1e-8) else 'stop'
                    event_end_idx = idx
                    break

                # 2. Check Activation (if not already active)
                if event_use_trailing and not trailing_active:
                    activation_price = entry_price * (1 - profit_thr)
                    if low_price <= activation_price:
                        # Activated!
                        peak_price = low_price
                        intra_bar_stop = peak_price + event_trail_dist
                        eff_intra_stop = min(fixed_stop_price, intra_bar_stop)

                        if high_price >= eff_intra_stop:
                            # Intra-bar conflict
                            raw_exit_price = (high_price + low_price) / 2
                            min_profit_price_short = entry_price * (1 - profit_thr)
                            exit_price = min(raw_exit_price, min_profit_price_short)
                            exit_reason = 'trailing'
                            event_end_idx = idx
                            break
                        else:
                            trailing_active = True

                # 3. Standard Fixed Take Profit (if trailing disabled)
                if not event_use_trailing:
                    if pnl_low >= profit_thr:
                        exit_price = entry_price * (1 - profit_thr)
                        exit_reason = 'profit'
                        event_end_idx = idx
                        break

        # If no exit, use end-of-horizon price (timeout), but clamp to avoid
        # synthetic losses beyond the nominal stop level.
        if exit_price is None:
            event_end_idx = min(i + event_horizon, n - 1)
            final_close = close_prices[event_end_idx]
            if signal > 0:
                fixed_stop_price = entry_price * (1 - stop_thr)
                if final_close < fixed_stop_price:
                    exit_price = 0.5 * (final_close + fixed_stop_price)
                else:
                    exit_price = final_close
            else:
                fixed_stop_price = entry_price * (1 + stop_thr)
                if final_close > fixed_stop_price:
                    exit_price = 0.5 * (final_close + fixed_stop_price)
                else:
                    exit_price = final_close
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

        # Compute the unified binary label first (used for backward compatibility)
        unified_label = np.nan
        
        if use_multiclass_labels:
            # MULTI-CLASS LABELS: 0=timeout, 1=profit, 2=stop
            # This allows model to learn different patterns for each exit type
            if exit_reason == 'timeout':
                unified_label = 0.0  # Timeout/noise
            elif exit_reason == 'profit':
                unified_label = 1.0  # Hit profit target
            elif exit_reason == 'stop':
                unified_label = 2.0  # Hit stop loss (bad entry)
            else:
                unified_label = np.nan  # Should not happen
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
                unified_label = 0.0
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
                    unified_label = 1.0
                elif net_return < 0:  # Losses are losses regardless of speed
                    unified_label = 0.0
                else:
                    # Profitable but too slow = noise/drift, soft-label as class 0
                    unified_label = 0.0

            if exit_reason == 'timeout' and net_return > PROFITABLE_TIMEOUT_RETURN_THRESHOLD:
                unified_label = 1.0

        # Assign the unified label (backward compatible)
        binary_labels.iloc[i] = unified_label
        
        # DIRECTIONAL LABELS: Assign to direction-specific series
        # This allows training separate models for longs vs shorts
        # - binary_labels_long: Only populated for long signals (signal > 0)
        # - binary_labels_short: Only populated for short signals (signal < 0)
        # The other direction remains NaN, so models can be trained on subsets
        if signal > 0:
            # Long signal - populate binary_labels_long, leave binary_labels_short as NaN
            binary_labels_long.iloc[i] = unified_label
        elif signal < 0:
            # Short signal - populate binary_labels_short, leave binary_labels_long as NaN  
            binary_labels_short.iloc[i] = unified_label

        last_event_idx = i  # Update last event position
        i += 1

    return (
        realized_returns, 
        binary_labels, 
        exit_reasons, 
        event_durations, 
        mfe_series, 
        mae_series,
        binary_labels_long,
        binary_labels_short
    )


def compute_vol_scaled_returns_for_events(
    realized_returns: pd.Series,
    volatility: Optional[pd.Series],
    econ_min_return_multiple: Optional[float] = None,
) -> pd.Series:
    """Compute volatility-scaled returns for events.

    Args:
        realized_returns: Per-event realized returns (after costs).
        volatility: Daily volatility series aligned to market data.
        econ_min_return_multiple: Optional economic floor multiplier expressed
            in units of transaction cost. When provided, the economic floor is
            ``econ_min_return_multiple × DEFAULT_TRANSACTION_COST``; otherwise
            the global ``ECON_MIN_RETURN_MULTIPLE`` constant is used.

    Returns:
        Series of volatility-scaled returns with economically trivial events
        masked as NaN and extreme values winsorised.
    """

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
        # quantile-based labels focus on meaningful moves. Use a low floor tied
        # to transaction costs rather than an arbitrary fixed threshold. When a
        # custom ``econ_min_return_multiple`` is provided (e.g. by HPO), honor
        # it; otherwise fall back to the global ECON_MIN_RETURN_MULTIPLE.
        econ_mult = (
            float(econ_min_return_multiple)
            if econ_min_return_multiple is not None
            else float(ECON_MIN_RETURN_MULTIPLE)
        )
        econ_floor = DEFAULT_TRANSACTION_COST * econ_mult
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
    high_q: float = 0.8,
) -> pd.Series:
    """Create binary labels using GLOBAL quantile thresholds.
    
    WARNING: This uses look-ahead bias as thresholds are computed across ALL data.
    For production use, prefer create_rolling_quantile_labels_from_vol_scaled_returns().
    """
    labels = pd.Series(index=vol_scaled.index, dtype=float)
    labels[:] = np.nan

    try:
        v = vol_scaled.dropna()
        if len(v) < 50:
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


def create_rolling_quantile_labels_from_vol_scaled_returns(
    vol_scaled: pd.Series,
    low_q: float = 0.3,
    high_q: float = 0.7,
    lookback_bars: int = 3000,
    min_periods: int = 300,
    expanding_start: bool = True,
) -> pd.Series:
    """Create binary labels using ROLLING quantile thresholds (no look-ahead bias).
    
    This function computes quantile thresholds using only PAST data at each point,
    eliminating the look-ahead bias present in global quantile approaches.
    
    Args:
        vol_scaled: Volatility-scaled returns series
        low_q: Lower quantile for negative labels (e.g., 0.3 = bottom 30%)
        high_q: Upper quantile for positive labels (e.g., 0.7 = top 30%)
        lookback_bars: Rolling window size in bars (default: 3000 ~= 31 days at 15m)
        min_periods: Minimum observations required before computing quantiles
        expanding_start: If True, use expanding window until lookback_bars is reached
    
    Returns:
        Series with labels: 1.0 (positive), 0.0 (negative), NaN (unlabeled/insufficient data)
    """
    labels = pd.Series(index=vol_scaled.index, dtype=float)
    labels[:] = np.nan
    
    try:
        # Get non-NaN values and their positions
        valid_mask = ~vol_scaled.isna()
        if valid_mask.sum() < min_periods:
            tprint(f"⚠️ Rolling quantiles: insufficient data ({valid_mask.sum()} < {min_periods})", "WARNING")
            return labels
        
        # Compute rolling quantiles using only past data
        # Use shift(1) to ensure we don't include the current observation in its own threshold
        if expanding_start:
            # Start with expanding window, switch to rolling after lookback_bars
            rolling_low = vol_scaled.expanding(min_periods=min_periods).quantile(low_q).shift(1)
            rolling_high = vol_scaled.expanding(min_periods=min_periods).quantile(high_q).shift(1)
            
            # After we have enough data, switch to fixed rolling window
            rolling_low_fixed = vol_scaled.rolling(window=lookback_bars, min_periods=min_periods).quantile(low_q).shift(1)
            rolling_high_fixed = vol_scaled.rolling(window=lookback_bars, min_periods=min_periods).quantile(high_q).shift(1)
            
            # Use fixed rolling where we have enough history
            enough_history = pd.Series(range(len(vol_scaled)), index=vol_scaled.index) >= lookback_bars
            rolling_low = rolling_low.where(~enough_history, rolling_low_fixed)
            rolling_high = rolling_high.where(~enough_history, rolling_high_fixed)
        else:
            # Pure rolling window (NaN until min_periods reached)
            rolling_low = vol_scaled.rolling(window=lookback_bars, min_periods=min_periods).quantile(low_q).shift(1)
            rolling_high = vol_scaled.rolling(window=lookback_bars, min_periods=min_periods).quantile(high_q).shift(1)
        
        # Apply thresholds
        valid_thresholds = (
            rolling_low.notna() & 
            rolling_high.notna() & 
            (rolling_high > rolling_low) &
            vol_scaled.notna()
        )
        
        # Label based on rolling thresholds
        labels.loc[valid_thresholds & (vol_scaled >= rolling_high)] = 1.0
        labels.loc[valid_thresholds & (vol_scaled <= rolling_low)] = 0.0
        
        # Diagnostics
        n_labeled = labels.notna().sum()
        n_pos = (labels == 1.0).sum()
        n_neg = (labels == 0.0).sum()
        tprint(
            f"📊 Rolling quantile labels: {n_labeled} labeled ({n_pos} pos, {n_neg} neg), "
            f"lookback={lookback_bars}, q=[{low_q:.2f}, {high_q:.2f}]",
            "INFO"
        )
        
    except Exception as e:
        tprint(f"⚠️ Rolling quantile labeling failed: {e}", "WARNING")
        labels[:] = np.nan
    
    return labels


def create_rolling_regime_aware_quantile_labels_from_vol_scaled_returns(
    vol_scaled: pd.Series,
    regimes: Optional[pd.Series] = None,
    low_q: float = 0.3,
    high_q: float = 0.7,
    lookback_bars: int = 3000,
    min_periods: int = 300,
    min_samples_per_regime: int = 50,
    expanding_start: bool = True,
) -> pd.Series:
    """Regime-aware rolling quantile labeling (no look-ahead bias).
    
    Combines rolling quantile thresholds with regime conditioning. Within each
    regime, computes quantiles using only past data from that regime.
    
    Args:
        vol_scaled: Volatility-scaled returns series
        regimes: Optional regime labels (e.g., HMM states)
        low_q: Lower quantile for negative labels
        high_q: Upper quantile for positive labels
        lookback_bars: Rolling window size in bars (default: 3000 ~= 31 days at 15m)
        min_periods: Minimum observations before computing quantiles
        min_samples_per_regime: Minimum samples per regime for regime-specific thresholds
        expanding_start: If True, use expanding window until lookback_bars reached
    
    Returns:
        Series with labels: 1.0 (positive), 0.0 (negative), NaN (unlabeled)
    """
    labels = pd.Series(index=vol_scaled.index, dtype=float)
    labels[:] = np.nan
    
    # Fall back to non-regime-aware if no regimes provided
    if regimes is None:
        return create_rolling_quantile_labels_from_vol_scaled_returns(
            vol_scaled=vol_scaled,
            low_q=low_q,
            high_q=high_q,
            lookback_bars=lookback_bars,
            min_periods=min_periods,
            expanding_start=expanding_start,
        )
    
    try:
        regimes_aligned = regimes.reindex(vol_scaled.index)
        unique_regimes = pd.unique(regimes_aligned.dropna())
        
        if len(unique_regimes) == 0:
            return create_rolling_quantile_labels_from_vol_scaled_returns(
                vol_scaled=vol_scaled,
                low_q=low_q,
                high_q=high_q,
                lookback_bars=lookback_bars,
                min_periods=min_periods,
                expanding_start=expanding_start,
            )
        
        for reg_val in unique_regimes:
            try:
                regime_mask = regimes_aligned == reg_val
                regime_data = vol_scaled.where(regime_mask)
                
                # Count samples in this regime
                n_regime = regime_data.notna().sum()
                if n_regime < min_samples_per_regime:
                    continue
                
                # Compute rolling quantiles within this regime
                # Use cumcount to track regime-specific sample count
                if expanding_start:
                    reg_rolling_low = regime_data.expanding(min_periods=min(min_periods, min_samples_per_regime)).quantile(low_q).shift(1)
                    reg_rolling_high = regime_data.expanding(min_periods=min(min_periods, min_samples_per_regime)).quantile(high_q).shift(1)
                else:
                    reg_rolling_low = regime_data.rolling(window=lookback_bars, min_periods=min_periods).quantile(low_q).shift(1)
                    reg_rolling_high = regime_data.rolling(window=lookback_bars, min_periods=min_periods).quantile(high_q).shift(1)
                
                # Apply thresholds for this regime
                valid_regime = (
                    regime_mask &
                    reg_rolling_low.notna() &
                    reg_rolling_high.notna() &
                    (reg_rolling_high > reg_rolling_low) &
                    vol_scaled.notna()
                )
                
                labels.loc[valid_regime & (vol_scaled >= reg_rolling_high)] = 1.0
                labels.loc[valid_regime & (vol_scaled <= reg_rolling_low)] = 0.0
                
            except Exception:
                # Skip this regime on error
                continue
        
        # If no labels assigned (regimes too sparse), fall back to global rolling
        if labels.dropna().empty:
            tprint("⚠️ Regime-aware rolling quantiles: falling back to global", "WARNING")
            return create_rolling_quantile_labels_from_vol_scaled_returns(
                vol_scaled=vol_scaled,
                low_q=low_q,
                high_q=high_q,
                lookback_bars=lookback_bars,
                min_periods=min_periods,
                expanding_start=expanding_start,
            )
        
        n_labeled = labels.notna().sum()
        n_pos = (labels == 1.0).sum()
        n_neg = (labels == 0.0).sum()
        tprint(
            f"📊 Rolling regime-aware quantile labels: {n_labeled} labeled ({n_pos} pos, {n_neg} neg)",
            "INFO"
        )
        
    except Exception as e:
        tprint(f"⚠️ Rolling regime-aware quantile labeling failed: {e}", "WARNING")
        return create_rolling_quantile_labels_from_vol_scaled_returns(
            vol_scaled=vol_scaled,
            low_q=low_q,
            high_q=high_q,
            lookback_bars=lookback_bars,
            min_periods=min_periods,
            expanding_start=expanding_start,
        )
    
    return labels


def compute_volatility_normalized_zscore(
    realized_returns: pd.Series,
    volatility: pd.Series,
    vol_lookback: int = 100,
    vol_min_periods: int = 20,
    clip_zscore: float = 5.0,
) -> pd.Series:
    """Compute volatility-normalized z-scores for trend-preserving labeling.
    
    Normalizes future returns by rolling volatility to produce z-scores that:
    - Preserve trend-following signal (magnitude matters)
    - Are volatility-aware (scale-normalized)
    - Are comparable across different volatility regimes
    
    z = future_return / rolling_volatility
    
    Args:
        realized_returns: Raw realized returns series
        volatility: Volatility series (e.g., volatility_1d or ATR-based)
        vol_lookback: Lookback for volatility smoothing (default: 100 bars)
        vol_min_periods: Minimum periods for volatility calculation
        clip_zscore: Maximum absolute z-score to prevent outliers
    
    Returns:
        Series of volatility-normalized z-scores
    """
    z_scores = pd.Series(index=realized_returns.index, dtype=float)
    z_scores[:] = np.nan
    
    try:
        # Align volatility to returns
        vol_aligned = volatility.reindex(realized_returns.index)
        
        # Use rolling volatility for smoothing (EMA-style for responsiveness)
        rolling_vol = vol_aligned.ewm(span=vol_lookback, min_periods=vol_min_periods).mean()
        
        # Ensure positive volatility
        rolling_vol = rolling_vol.replace(0.0, np.nan).abs()
        
        # Compute z-scores
        z_scores = realized_returns / (rolling_vol + 1e-8)
        
        # Clip extreme values to prevent outlier domination
        z_scores = z_scores.clip(lower=-clip_zscore, upper=clip_zscore)
        
        # Fill any infinities
        z_scores = z_scores.replace([np.inf, -np.inf], np.nan)
        
        n_valid = z_scores.notna().sum()
        z_mean = float(z_scores.mean()) if n_valid > 0 else 0.0
        z_std = float(z_scores.std()) if n_valid > 1 else 1.0
        
        tprint(
            f"📊 Z-score normalization: n={n_valid}, mean={z_mean:.3f}, std={z_std:.3f}",
            "INFO"
        )
        
    except Exception as e:
        tprint(f"⚠️ Z-score computation failed: {e}", "WARNING")
        z_scores[:] = np.nan
    
    return z_scores


def create_conditional_quantile_labels(
    realized_returns: pd.Series,
    features: pd.DataFrame,
    volatility: pd.Series,
    quantile_long: float = 0.6,
    quantile_short: float = 0.35,
    lookback_bars: int = 3000,
    min_train_samples: int = 500,
    retrain_frequency: int = 500,
    vol_lookback: int = 100,
    use_lightgbm: bool = True,
    n_estimators: int = 50,
    max_depth: int = 4,
    feature_subset: Optional[List[str]] = None,
    asymmetric_crypto: bool = True,
) -> Tuple[pd.Series, pd.Series, pd.Series, Dict[str, Any]]:
    """Create labels using conditional quantile regression with asymmetric tails.
    
    Predicts conditional quantiles Q_τ(z | X) for both long and short directions,
    supporting asymmetric selection thresholds for crypto markets.
    
    Labeling logic:
        z > Q_long(z|X)  → long signal (label = 1)
        z < Q_short(z|X) → short signal (label = -1)
        else             → no trade (label = NaN)
    
    For crypto, asymmetric quantiles are recommended (asymmetric_crypto=True):
        - Longs: Q_0.6 (more selective, need to beat 60th percentile)
        - Shorts: Q_0.35 (less selective, below 35th percentile)
    
    For symmetric selection (asymmetric_crypto=False):
        - Longs: Q_0.6
        - Shorts: Q_0.4 (symmetric = 1 - quantile_long)
    
    Args:
        realized_returns: Per-event realized returns
        features: Feature matrix X for conditioning
        volatility: Volatility series for z-score normalization
        quantile_long: Upper quantile for long signals (default: 0.6)
        quantile_short: Lower quantile for short signals (default: 0.35 for crypto asymmetry)
        lookback_bars: Training window size (default: 3000 ~= 31 days at 15m)
        min_train_samples: Minimum samples before model training starts
        retrain_frequency: Retrain model every N bars
        vol_lookback: Lookback for volatility smoothing
        use_lightgbm: If True, use LightGBM; else use sklearn GBR
        n_estimators: Number of trees in ensemble
        max_depth: Maximum tree depth
        feature_subset: Optional list of features to use (for speed)
        asymmetric_crypto: If True, use asymmetric quantiles (crypto-optimized)
    
    Returns:
        Tuple of (labels, labels_long, labels_short, diagnostics)
        - labels: Combined directional labels (1=long, -1=short, NaN=no trade)
        - labels_long: Binary long labels (1=long, 0=not long, NaN=no signal)
        - labels_short: Binary short labels (1=short, 0=not short, NaN=no signal)
        - diagnostics: Dict with model performance metrics
    """
    labels = pd.Series(index=realized_returns.index, dtype=float)
    labels[:] = np.nan
    
    labels_long = pd.Series(index=realized_returns.index, dtype=float)
    labels_long[:] = np.nan
    
    labels_short = pd.Series(index=realized_returns.index, dtype=float)
    labels_short[:] = np.nan
    
    # Use symmetric quantiles if not crypto-asymmetric
    if not asymmetric_crypto:
        quantile_short = 1.0 - quantile_long  # e.g., 0.4 if long is 0.6
    
    diagnostics: Dict[str, Any] = {
        "n_labeled": 0,
        "n_long": 0,
        "n_short": 0,
        "n_no_trade": 0,
        "quantile_long": quantile_long,
        "quantile_short": quantile_short,
        "asymmetric_crypto": asymmetric_crypto,
        "mean_predicted_q_long": np.nan,
        "mean_predicted_q_short": np.nan,
        "coverage_long": np.nan,
        "coverage_short": np.nan,
        "model_type": "lightgbm" if use_lightgbm else "sklearn_gbr",
    }
    
    try:
        # Step 1: Compute volatility-normalized z-scores
        z_scores = compute_volatility_normalized_zscore(
            realized_returns=realized_returns,
            volatility=volatility,
            vol_lookback=vol_lookback,
        )
        
        # Step 2: Prepare features
        if feature_subset is not None and len(feature_subset) > 0:
            available_features = [f for f in feature_subset if f in features.columns]
            if len(available_features) < 5:
                # Fallback to all numeric features
                X = features.select_dtypes(include=[np.number])
            else:
                X = features[available_features]
        else:
            X = features.select_dtypes(include=[np.number])
        
        # Remove columns that might leak future info
        drop_cols = [c for c in X.columns if any(
            pat in c.lower() for pat in ['target', 'label', 'return', 'future', 'forward']
        )]
        X = X.drop(columns=drop_cols, errors='ignore')
        
        # Align indices
        common_idx = z_scores.dropna().index.intersection(X.dropna(how='all').index)
        if len(common_idx) < min_train_samples:
            tprint(f"⚠️ Conditional quantile: insufficient data ({len(common_idx)} < {min_train_samples})", "WARNING")
            return labels, labels_long, labels_short, diagnostics
        
        z_aligned = z_scores.loc[common_idx]
        X_aligned = X.loc[common_idx].fillna(0.0)
        
        # Step 3: Rolling prediction with periodic retraining
        tprint(
            f"📊 Conditional quantile regression: Q_long={quantile_long}, Q_short={quantile_short}, "
            f"asymmetric={asymmetric_crypto}, training on {len(common_idx)} samples...",
            "INFO"
        )
        
        # Convert to numpy for speed
        z_arr = z_aligned.to_numpy(dtype=float)
        X_arr = X_aligned.to_numpy(dtype=float)
        
        pred_q_long = np.full(len(common_idx), np.nan)
        pred_q_short = np.full(len(common_idx), np.nan)
        
        model_long = None
        model_short = None
        last_train_idx = -retrain_frequency  # Force initial training
        
        # Rolling prediction
        for i in range(min_train_samples, len(common_idx)):
            # Retrain periodically
            if (i - last_train_idx) >= retrain_frequency or model_long is None:
                train_start = max(0, i - lookback_bars)
                train_end = i
                
                X_train = X_arr[train_start:train_end]
                z_train = z_arr[train_start:train_end]
                
                # Remove NaN from training
                valid_train = ~np.isnan(z_train)
                if valid_train.sum() < 50:
                    continue
                
                X_train = X_train[valid_train]
                z_train = z_train[valid_train]
                
                if use_lightgbm:
                    # LightGBM quantile regression for upper quantile (longs)
                    model_long = lgb.LGBMRegressor(
                        objective='quantile',
                        alpha=quantile_long,
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        min_child_samples=20,
                        n_jobs=-1,
                        verbose=-1,
                        random_state=42,
                    )
                    # LightGBM quantile regression for lower quantile (shorts)
                    model_short = lgb.LGBMRegressor(
                        objective='quantile',
                        alpha=quantile_short,
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        min_child_samples=20,
                        n_jobs=-1,
                        verbose=-1,
                        random_state=42,
                    )
                else:
                    # Sklearn GradientBoostingRegressor
                    from sklearn.ensemble import GradientBoostingRegressor
                    model_long = GradientBoostingRegressor(
                        loss='quantile',
                        alpha=quantile_long,
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=0.05,
                        subsample=0.8,
                        min_samples_leaf=20,
                        random_state=42,
                    )
                    model_short = GradientBoostingRegressor(
                        loss='quantile',
                        alpha=quantile_short,
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=0.05,
                        subsample=0.8,
                        min_samples_leaf=20,
                        random_state=42,
                    )
                
                try:
                    model_long.fit(X_train, z_train)
                    model_short.fit(X_train, z_train)
                    last_train_idx = i
                except Exception as fit_err:
                    tprint(f"⚠️ Model fit failed at i={i}: {fit_err}", "WARNING")
                    continue
            
            # Predict for current observation
            if model_long is not None and model_short is not None:
                try:
                    X_pred = X_arr[i:i+1]
                    pred_q_long[i] = float(model_long.predict(X_pred)[0])
                    pred_q_short[i] = float(model_short.predict(X_pred)[0])
                except Exception:
                    pass
        
        # Step 4: Generate directional labels based on conditional quantiles
        # z > Q_long → long signal (1)
        # z < Q_short → short signal (-1)
        # else → no trade (NaN)
        
        for i, idx in enumerate(common_idx):
            if np.isnan(pred_q_long[i]) or np.isnan(pred_q_short[i]) or np.isnan(z_arr[i]):
                continue
            
            z_val = z_arr[i]
            q_long_val = pred_q_long[i]
            q_short_val = pred_q_short[i]
            
            if z_val >= q_long_val:
                # Long signal: z exceeds upper quantile
                labels.loc[idx] = 1.0
                labels_long.loc[idx] = 1.0
                labels_short.loc[idx] = 0.0
            elif z_val <= q_short_val:
                # Short signal: z below lower quantile
                labels.loc[idx] = -1.0
                labels_long.loc[idx] = 0.0
                labels_short.loc[idx] = 1.0
            else:
                # No trade zone (between quantiles)
                labels.loc[idx] = 0.0  # Explicit no-trade
                labels_long.loc[idx] = 0.0
                labels_short.loc[idx] = 0.0
        
        # Diagnostics
        n_labeled = int((labels != 0).sum())  # Exclude no-trade
        n_long = int((labels == 1.0).sum())
        n_short = int((labels == -1.0).sum())
        n_no_trade = int((labels == 0.0).sum())
        
        valid_pred = ~np.isnan(pred_q_long)
        mean_pred_q_long = float(np.nanmean(pred_q_long)) if valid_pred.any() else np.nan
        mean_pred_q_short = float(np.nanmean(pred_q_short)) if valid_pred.any() else np.nan
        
        # Coverage diagnostics
        if valid_pred.any():
            above_q_long = z_arr[valid_pred] >= pred_q_long[valid_pred]
            below_q_short = z_arr[valid_pred] <= pred_q_short[valid_pred]
            coverage_long = float(above_q_long.mean())
            coverage_short = float(below_q_short.mean())
        else:
            coverage_long = np.nan
            coverage_short = np.nan
        
        diagnostics.update({
            "n_labeled": n_labeled,
            "n_long": n_long,
            "n_short": n_short,
            "n_no_trade": n_no_trade,
            "mean_predicted_q_long": mean_pred_q_long,
            "mean_predicted_q_short": mean_pred_q_short,
            "coverage_long": coverage_long,
            "coverage_short": coverage_short,
            "expected_coverage_long": 1.0 - quantile_long,
            "expected_coverage_short": quantile_short,
        })
        
        tprint(
            f"📊 Conditional quantile labels: {n_long} long, {n_short} short, {n_no_trade} no-trade",
            "INFO"
        )
        tprint(
            f"   Coverage: long={coverage_long:.1%} (exp={1.0-quantile_long:.1%}), "
            f"short={coverage_short:.1%} (exp={quantile_short:.1%})",
            "INFO"
        )
        
        # Calibration checks
        if np.isfinite(coverage_long) and abs(coverage_long - (1.0 - quantile_long)) > 0.1:
            tprint(
                f"⚠️ Long quantile may be miscalibrated: coverage={coverage_long:.1%} vs expected={1.0-quantile_long:.1%}",
                "WARNING"
            )
        if np.isfinite(coverage_short) and abs(coverage_short - quantile_short) > 0.1:
            tprint(
                f"⚠️ Short quantile may be miscalibrated: coverage={coverage_short:.1%} vs expected={quantile_short:.1%}",
                "WARNING"
            )
        
    except Exception as e:
        tprint(f"⚠️ Conditional quantile labeling failed: {e}", "WARNING")
        import traceback
        traceback.print_exc()
    
    return labels, labels_long, labels_short, diagnostics


# ==============================================================================
# VOLATILITY-BASED LABELING SYSTEM
# ==============================================================================
# These functions implement the new volatility-scaled labeling approach:
#   y = future_return > k_t * rolling_volatility
# Where k_t is a dynamic threshold that adapts to market regimes.
# ==============================================================================


def compute_ema_volatility(
    returns: pd.Series,
    span: int = 48,
    min_periods: int = 10,
) -> pd.Series:
    """Compute rolling volatility using EMA of squared returns.
    
    Uses the formula: volatility_t = sqrt(EMA(return^2, span=N))
    
    This provides a responsive, smooth estimate of recent volatility that:
    - Avoids excessive lag from simple rolling windows
    - Responds quickly to volatility regime changes
    - Is suitable for dynamic threshold calibration
    
    Args:
        returns: Log returns or percentage returns series
        span: EMA span in bars (recommended: 32-64 bars for 15m data = 8-16h)
        min_periods: Minimum periods before computing volatility
    
    Returns:
        Series of EMA-based volatility estimates
    """
    # Compute squared returns
    squared_returns = returns ** 2
    
    # Compute EMA of squared returns
    ema_sq_ret = squared_returns.ewm(span=span, min_periods=min_periods, adjust=False).mean()
    
    # Take square root to get volatility (standard deviation estimate)
    ema_volatility = np.sqrt(ema_sq_ret)
    
    # Replace zeros and infinities with NaN
    ema_volatility = ema_volatility.replace([0.0, np.inf, -np.inf], np.nan)
    
    return ema_volatility


def compute_regime_metrics(
    volatility: pd.Series,
    returns: pd.Series,
    median_lookback_bars: int = 400,
    trend_lookback_bars: int = 20,
    min_periods: int = 50,
) -> Tuple[pd.Series, pd.Series]:
    """Compute regime metrics for dynamic threshold adaptation.
    
    Computes two regime metrics:
    1. Volatility ratio: current_vol / median_vol_over_window
       - > 1 means higher than typical volatility
       - < 1 means lower than typical volatility
       
    2. Trend strength: rolling_mean(returns) / rolling_vol
       - Positive = upward trend
       - Negative = downward trend
       - Near zero = mean-reverting / choppy
    
    Args:
        volatility: EMA-based volatility series
        returns: Returns series
        median_lookback_bars: Window for computing median volatility (300-500 bars = 3-5 days)
        trend_lookback_bars: Window for computing trend strength (10-30 bars)
        min_periods: Minimum periods for rolling calculations
    
    Returns:
        Tuple of (volatility_ratio, trend_strength) series
    """
    # Volatility ratio: current_vol / median_vol
    # Use rolling median for robustness to outliers
    median_vol = volatility.rolling(
        window=median_lookback_bars, 
        min_periods=min_periods
    ).median()
    
    volatility_ratio = volatility / (median_vol + 1e-8)
    volatility_ratio = volatility_ratio.replace([np.inf, -np.inf], np.nan)
    
    # Trend strength: rolling_mean(returns) / rolling_vol
    rolling_mean_returns = returns.rolling(
        window=trend_lookback_bars, 
        min_periods=min_periods // 2
    ).mean()
    rolling_vol = volatility.rolling(
        window=trend_lookback_bars, 
        min_periods=min_periods // 2
    ).mean()
    
    trend_strength = rolling_mean_returns / (rolling_vol + 1e-8)
    trend_strength = trend_strength.replace([np.inf, -np.inf], np.nan)
    
    # Clip extreme values
    volatility_ratio = volatility_ratio.clip(0.1, 10.0)
    trend_strength = trend_strength.clip(-5.0, 5.0)
    
    return volatility_ratio, trend_strength


def compute_target_positive_fraction(
    trend_strength: pd.Series,
    volatility_ratio: pd.Series,
    p_min: float = 0.30,
    p_max: float = 0.70,
    sigmoid_slope: float = 1.5,
    sigmoid_midpoint: float = 0.0,
) -> pd.Series:
    """Compute target positive label fraction using sigmoid function.
    
    Uses a sigmoid function combining trend and volatility metrics:
    
    p_positive(t) = p_min + (p_max - p_min) / (1 + exp(-a * (trend_t / vol_ratio_t - b)))
    
    Where:
        - p_min, p_max = min/max target positive fraction (e.g., 0.3, 0.7)
        - a = sigmoid_slope (controls sensitivity, e.g., 1.5)
        - b = sigmoid_midpoint (central reference point, e.g., 0.0)
    
    This creates a dynamic target that:
        - In strong uptrends → higher positive fraction (up to p_max)
        - In strong downtrends → lower positive fraction (down to p_min)
        - In neutral markets → around 0.5 positive fraction
    
    Args:
        trend_strength: Trend strength series (mean returns / vol)
        volatility_ratio: Volatility ratio series (current_vol / median_vol)
        p_min: Minimum target positive fraction (default: 0.30)
        p_max: Maximum target positive fraction (default: 0.70)
        sigmoid_slope: Slope parameter 'a' (default: 1.5)
        sigmoid_midpoint: Midpoint parameter 'b' (default: 0.0)
    
    Returns:
        Series of target positive fractions in [p_min, p_max]
    """
    # Compute the regime signal: trend / vol_ratio
    # High trend + low vol_ratio → strong bullish signal
    # Low trend + high vol_ratio → strong bearish signal
    regime_signal = trend_strength / (volatility_ratio + 1e-8)
    
    # Apply sigmoid function
    exponent = -sigmoid_slope * (regime_signal - sigmoid_midpoint)
    
    # Clip exponent to avoid overflow
    exponent = exponent.clip(-20, 20)
    
    sigmoid_output = 1.0 / (1.0 + np.exp(exponent))
    
    # Scale to [p_min, p_max]
    p_positive = p_min + (p_max - p_min) * sigmoid_output
    
    return p_positive


def compute_dynamic_threshold_k(
    z_scores: pd.Series,
    p_positive: pd.Series,
    rolling_window: int = 400,
    min_periods: int = 100,
) -> pd.Series:
    """Compute dynamic threshold k_t for volatility-scaled labeling.
    
    Computes: k_t = quantile_{1 - p_positive(t)}(z over rolling window)
    
    This threshold adapts to:
    - The target positive label fraction (regime-dependent)
    - Recent z-score distribution (rolling quantile)
    
    The label is then assigned as:
        y_t = 1 if future_return_t > k_t * rolling_volatility_t
    
    Args:
        z_scores: Volatility-normalized z-scores (future_return / vol)
        p_positive: Target positive fraction series (from sigmoid function)
        rolling_window: Rolling window for quantile computation (300-500 bars)
        min_periods: Minimum periods for rolling quantile
    
    Returns:
        Series of dynamic threshold k values
    """
    k_t = pd.Series(index=z_scores.index, dtype=float)
    k_t[:] = np.nan
    
    # We need to compute rolling quantiles at the (1 - p_positive) level
    # This is done per-timestep with expanding/rolling windows
    
    valid_idx = z_scores.dropna().index
    if len(valid_idx) < min_periods:
        return k_t
    
    # Convert to numpy for efficient computation
    z_arr = z_scores.to_numpy()
    p_arr = p_positive.reindex(z_scores.index).to_numpy()
    
    # Use rolling window approach
    for i in range(min_periods, len(z_arr)):
        if np.isnan(p_arr[i]) or np.isnan(z_arr[i]):
            continue
            
        # Get rolling window of z-scores
        start_idx = max(0, i - rolling_window)
        z_window = z_arr[start_idx:i]  # Exclude current (no lookahead)
        z_window = z_window[~np.isnan(z_window)]
        
        if len(z_window) < min_periods // 2:
            continue
        
        # Compute quantile at (1 - p_positive)
        quantile_level = 1.0 - p_arr[i]
        quantile_level = np.clip(quantile_level, 0.05, 0.95)  # Safety bounds
        
        k_t.iloc[i] = float(np.quantile(z_window, quantile_level))
    
    return k_t


def create_volatility_scaled_labels(
    future_returns: pd.Series,
    close_prices: pd.Series,
    # Volatility parameters
    volatility_ema_span: int = 48,
    # Dynamic threshold parameters
    rolling_k_window: int = 400,
    median_vol_lookback: int = 400,
    trend_lookback: int = 20,
    # Sigmoid parameters for p_positive
    p_min: float = 0.30,
    p_max: float = 0.70,
    sigmoid_slope: float = 1.5,
    sigmoid_midpoint: float = 0.0,
    # General parameters
    min_periods: int = 100,
    clip_zscore: float = 5.0,
    econ_floor: Optional[float] = None,
) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
    """Create volatility-scaled labels using dynamic threshold k.
    
    This implements the full volatility-based labeling pipeline:
    
    1. Volatility Estimation: vol_t = sqrt(EMA(return^2, span=N))
    2. Regime Metrics:
       - volatility_ratio = current_vol / median_vol
       - trend_strength = mean(returns) / vol
    3. Target Positive Fraction via sigmoid:
       p_positive(t) = p_min + (p_max - p_min) / (1 + exp(-a * (trend/vol_ratio - b)))
    4. Z-score computation: z_t = future_return_t / vol_t
    5. Dynamic threshold: k_t = quantile_{1-p_positive}(z over rolling window)
    6. Label assignment: y_t = 1 if future_return_t > k_t * vol_t
    
    Key benefits:
    - Labels are relative to market volatility (not fixed percentiles)
    - Threshold adapts to regime (trend + volatility conditions)
    - Works across different market environments
    - Z-score preserves trend magnitude while being volatility-aware
    
    Args:
        future_returns: Future returns at each timestamp (the target to predict)
        close_prices: Close price series for computing returns
        volatility_ema_span: EMA span for volatility (32-64 bars = 8-16h at 15m)
        rolling_k_window: Window for dynamic k quantile (300-500 bars = 3-5 days)
        median_vol_lookback: Window for median volatility (300-500 bars)
        trend_lookback: Window for trend strength (10-30 bars)
        p_min: Minimum target positive fraction
        p_max: Maximum target positive fraction
        sigmoid_slope: Sigmoid slope parameter
        sigmoid_midpoint: Sigmoid midpoint parameter
        min_periods: Minimum periods for rolling calculations
        clip_zscore: Maximum |z-score| to prevent outliers
        econ_floor: Optional economic floor for filtering trivial returns
    
    Returns:
        Tuple of (labels, z_scores, diagnostics)
        - labels: Binary labels (1=positive, 0=negative, NaN=filtered)
        - z_scores: Volatility-normalized z-scores
        - diagnostics: Dict with computation diagnostics
    """
    labels = pd.Series(index=future_returns.index, dtype=float)
    labels[:] = np.nan
    
    z_scores = pd.Series(index=future_returns.index, dtype=float)
    z_scores[:] = np.nan
    
    diagnostics: Dict[str, Any] = {
        "n_total": len(future_returns),
        "n_labeled": 0,
        "n_positive": 0,
        "n_negative": 0,
        "positive_ratio": np.nan,
        "mean_k_threshold": np.nan,
        "mean_volatility": np.nan,
        "mean_vol_ratio": np.nan,
        "mean_trend_strength": np.nan,
        "mean_p_positive": np.nan,
        "mean_z_score": np.nan,
        "std_z_score": np.nan,
    }
    
    try:
        # Step 1: Compute returns for volatility estimation
        log_returns = np.log(close_prices / close_prices.shift(1))
        log_returns = log_returns.replace([np.inf, -np.inf], np.nan)
        
        # Step 2: Compute EMA-based volatility
        ema_volatility = compute_ema_volatility(
            returns=log_returns,
            span=volatility_ema_span,
            min_periods=min_periods // 4,
        )
        
        # Step 3: Compute regime metrics
        volatility_ratio, trend_strength = compute_regime_metrics(
            volatility=ema_volatility,
            returns=log_returns,
            median_lookback_bars=median_vol_lookback,
            trend_lookback_bars=trend_lookback,
            min_periods=min_periods,
        )
        
        # Step 4: Compute target positive fraction via sigmoid
        p_positive = compute_target_positive_fraction(
            trend_strength=trend_strength,
            volatility_ratio=volatility_ratio,
            p_min=p_min,
            p_max=p_max,
            sigmoid_slope=sigmoid_slope,
            sigmoid_midpoint=sigmoid_midpoint,
        )
        
        # Step 5: Compute z-scores (future returns normalized by volatility)
        vol_aligned = ema_volatility.reindex(future_returns.index)
        z_scores = future_returns / (vol_aligned + 1e-8)
        z_scores = z_scores.replace([np.inf, -np.inf], np.nan)
        z_scores = z_scores.clip(lower=-clip_zscore, upper=clip_zscore)
        
        # Step 6: Compute dynamic threshold k_t
        k_t = compute_dynamic_threshold_k(
            z_scores=z_scores,
            p_positive=p_positive,
            rolling_window=rolling_k_window,
            min_periods=min_periods,
        )
        
        # Step 7: Assign labels: y_t = 1 if future_return > k_t * volatility
        # Equivalently: y_t = 1 if z_t > k_t
        valid_mask = (
            z_scores.notna() & 
            k_t.notna() & 
            vol_aligned.notna() &
            future_returns.notna()
        )
        
        # Apply economic floor if specified
        if econ_floor is not None and econ_floor > 0:
            valid_mask = valid_mask & (future_returns.abs() >= econ_floor)
        
        # Assign labels based on z-score vs dynamic threshold
        labels.loc[valid_mask & (z_scores > k_t)] = 1.0
        labels.loc[valid_mask & (z_scores <= k_t)] = 0.0
        
        # Compute diagnostics
        n_labeled = int(labels.notna().sum())
        n_positive = int((labels == 1.0).sum())
        n_negative = int((labels == 0.0).sum())
        
        diagnostics.update({
            "n_labeled": n_labeled,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "positive_ratio": n_positive / n_labeled if n_labeled > 0 else np.nan,
            "mean_k_threshold": float(k_t.mean()) if k_t.notna().any() else np.nan,
            "mean_volatility": float(ema_volatility.mean()) if ema_volatility.notna().any() else np.nan,
            "mean_vol_ratio": float(volatility_ratio.mean()) if volatility_ratio.notna().any() else np.nan,
            "mean_trend_strength": float(trend_strength.mean()) if trend_strength.notna().any() else np.nan,
            "mean_p_positive": float(p_positive.mean()) if p_positive.notna().any() else np.nan,
            "mean_z_score": float(z_scores.mean()) if z_scores.notna().any() else np.nan,
            "std_z_score": float(z_scores.std()) if z_scores.notna().any() else np.nan,
        })
        
        tprint(
            f"📊 Volatility-scaled labels: {n_labeled} labeled ({n_positive} pos, {n_negative} neg), "
            f"ratio={n_positive/n_labeled:.1%}" if n_labeled > 0 else "📊 No labels generated",
            "INFO"
        )
        tprint(
            f"   Vol: mean={diagnostics['mean_volatility']:.4f}, "
            f"k: mean={diagnostics['mean_k_threshold']:.3f}, "
            f"z: mean={diagnostics['mean_z_score']:.3f} std={diagnostics['std_z_score']:.3f}",
            "INFO"
        )
        
    except Exception as e:
        tprint(f"⚠️ Volatility-scaled labeling failed: {e}", "WARNING")
        import traceback
        traceback.print_exc()
    
    return labels, z_scores, diagnostics


def create_volatility_scaled_labels_for_events(
    realized_returns: pd.Series,
    market_data: pd.DataFrame,
    # Volatility parameters (HPO tunable)
    volatility_ema_span: int = 48,
    # Dynamic threshold parameters (HPO tunable)
    rolling_k_window: int = 400,
    median_vol_lookback: int = 400,
    trend_lookback: int = 20,
    # Sigmoid parameters (HPO tunable)
    p_min: float = 0.30,
    p_max: float = 0.70,
    sigmoid_slope: float = 1.5,
    sigmoid_midpoint: float = 0.0,
    # General parameters
    min_periods: int = 100,
    clip_zscore: float = 5.0,
    econ_floor: Optional[float] = None,
) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
    """Create volatility-scaled labels for event-based returns.
    
    This is a convenience wrapper for create_volatility_scaled_labels that
    works with event-based realized returns (from triple barrier or similar).
    
    The key difference from the base function:
    - Takes event-based realized_returns (sparse, with NaN for non-events)
    - Computes volatility from underlying market_data close prices
    - Aligns all computations to the event timestamps
    
    This is the function to use in HPO for label generation.
    
    Args:
        realized_returns: Event-based realized returns (NaN for non-events)
        market_data: Full market data with 'close' column
        volatility_ema_span: EMA span for volatility (32-64 bars)
        rolling_k_window: Window for dynamic k quantile (300-500 bars)
        median_vol_lookback: Window for median volatility (300-500 bars)
        trend_lookback: Window for trend strength (10-30 bars)
        p_min: Minimum target positive fraction
        p_max: Maximum target positive fraction
        sigmoid_slope: Sigmoid slope parameter
        sigmoid_midpoint: Sigmoid midpoint parameter
        min_periods: Minimum periods for calculations
        clip_zscore: Maximum |z-score| 
        econ_floor: Optional economic floor for filtering
    
    Returns:
        Tuple of (labels, z_scores, diagnostics)
    """
    # Extract close prices
    close_col = 'close' if 'close' in market_data.columns else 'Close'
    if close_col not in market_data.columns:
        tprint("⚠️ No 'close' column in market_data", "WARNING")
        return (
            pd.Series(np.nan, index=realized_returns.index),
            pd.Series(np.nan, index=realized_returns.index),
            {"error": "no close column"}
        )
    
    close_prices = market_data[close_col]
    
    # Compute returns for volatility estimation (from full market data)
    log_returns = np.log(close_prices / close_prices.shift(1))
    log_returns = log_returns.replace([np.inf, -np.inf], np.nan)
    
    # Compute EMA-based volatility from market data
    ema_volatility = compute_ema_volatility(
        returns=log_returns,
        span=volatility_ema_span,
        min_periods=min_periods // 4,
    )
    
    # Compute regime metrics from market data
    volatility_ratio, trend_strength = compute_regime_metrics(
        volatility=ema_volatility,
        returns=log_returns,
        median_lookback_bars=median_vol_lookback,
        trend_lookback_bars=trend_lookback,
        min_periods=min_periods,
    )
    
    # Compute target positive fraction
    p_positive = compute_target_positive_fraction(
        trend_strength=trend_strength,
        volatility_ratio=volatility_ratio,
        p_min=p_min,
        p_max=p_max,
        sigmoid_slope=sigmoid_slope,
        sigmoid_midpoint=sigmoid_midpoint,
    )
    
    # Align volatility to event timestamps
    vol_aligned = ema_volatility.reindex(realized_returns.index)
    p_pos_aligned = p_positive.reindex(realized_returns.index)
    
    # Compute z-scores for events
    z_scores = realized_returns / (vol_aligned + 1e-8)
    z_scores = z_scores.replace([np.inf, -np.inf], np.nan)
    z_scores = z_scores.clip(lower=-clip_zscore, upper=clip_zscore)
    
    # Compute dynamic threshold k_t
    # For events, we use the full z-score series but only label at event times
    full_z_for_quantile = pd.Series(index=market_data.index, dtype=float)
    full_z_for_quantile[:] = np.nan
    # Fill with event z-scores at event times
    full_z_for_quantile.loc[z_scores.index] = z_scores
    
    k_t = compute_dynamic_threshold_k(
        z_scores=full_z_for_quantile,
        p_positive=p_positive,
        rolling_window=rolling_k_window,
        min_periods=min_periods,
    )
    
    # Align k_t to event timestamps
    k_t_aligned = k_t.reindex(realized_returns.index)
    
    # Initialize labels
    labels = pd.Series(index=realized_returns.index, dtype=float)
    labels[:] = np.nan
    
    # Valid mask
    valid_mask = (
        z_scores.notna() &
        k_t_aligned.notna() &
        vol_aligned.notna() &
        realized_returns.notna()
    )
    
    # Apply economic floor if specified
    if econ_floor is not None and econ_floor > 0:
        valid_mask = valid_mask & (realized_returns.abs() >= econ_floor)
    
    # Assign labels: y = 1 if z > k_t
    labels.loc[valid_mask & (z_scores > k_t_aligned)] = 1.0
    labels.loc[valid_mask & (z_scores <= k_t_aligned)] = 0.0
    
    # Compute diagnostics
    n_labeled = int(labels.notna().sum())
    n_positive = int((labels == 1.0).sum())
    n_negative = int((labels == 0.0).sum())
    
    diagnostics: Dict[str, Any] = {
        "n_total": len(realized_returns),
        "n_events": int(realized_returns.notna().sum()),
        "n_labeled": n_labeled,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "positive_ratio": n_positive / n_labeled if n_labeled > 0 else np.nan,
        "mean_k_threshold": float(k_t_aligned.mean()) if k_t_aligned.notna().any() else np.nan,
        "mean_volatility": float(vol_aligned.mean()) if vol_aligned.notna().any() else np.nan,
        "mean_vol_ratio": float(volatility_ratio.mean()) if volatility_ratio.notna().any() else np.nan,
        "mean_trend_strength": float(trend_strength.mean()) if trend_strength.notna().any() else np.nan,
        "mean_p_positive": float(p_pos_aligned.mean()) if p_pos_aligned.notna().any() else np.nan,
        "mean_z_score": float(z_scores.mean()) if z_scores.notna().any() else np.nan,
        "std_z_score": float(z_scores.std()) if z_scores.notna().any() else np.nan,
        "volatility_ema_span": volatility_ema_span,
        "rolling_k_window": rolling_k_window,
        "p_min": p_min,
        "p_max": p_max,
        "sigmoid_slope": sigmoid_slope,
    }
    
    tprint(
        f"📊 Vol-scaled event labels: {n_labeled} labeled ({n_positive} pos, {n_negative} neg), "
        f"ratio={n_positive/n_labeled:.1%}" if n_labeled > 0 else "📊 No event labels generated",
        "INFO"
    )
    
    return labels, z_scores, diagnostics


def compute_zscore_gated_triple_barrier_labels(
    df: pd.DataFrame,
    features: pd.DataFrame,
    signals: pd.DataFrame,
    volatility: pd.Series,
    # Conditional quantile parameters
    quantile_long: float = 0.6,
    quantile_short: float = 0.35,
    asymmetric_crypto: bool = True,
    quantile_lookback: int = 3000,
    quantile_min_train: int = 500,
    quantile_retrain_freq: int = 500,
    # Barrier parameters
    k_tp_base: float = 1.5,
    k_sl_base: float = 1.0,
    k_tp_long_mult: float = 1.1,   # Slightly larger TP for longs (crypto upward bias)
    k_sl_long_mult: float = 0.9,  # Slightly smaller SL for longs (capture pullbacks)
    k_tp_short_mult: float = 1.0,
    k_sl_short_mult: float = 1.0,
    # Z-score magnitude scaling
    z_magnitude_scale: float = 0.3,  # k_TP = k0 * (1 + z_magnitude_scale * |z|)
    # Trend adjustment
    trend_alpha: float = 0.3,  # TP_adj = TP * (1 + alpha * trend_strength)
    trend_lookback: int = 20,
    # Clipping bounds (as multiples of base volatility)
    tp_min_mult: float = 0.5,
    tp_max_mult: float = 4.0,
    sl_min_mult: float = 0.3,
    sl_max_mult: float = 2.0,
    # Horizon and other parameters
    horizon: int = 26,
    transaction_cost: float = 0.003,
    min_event_spacing: int = 2,
    # Trailing profit
    atr_series: Optional[pd.Series] = None,
    trail_distance_atr_mult: Optional[float] = None,
    # Feature subset for quantile model
    feature_subset: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Z-Score Gated Triple Barrier Labeling Pipeline.
    
    This implements the full pipeline:
    
        [ Market Features ]
                ↓
        Conditional Quantile Filter
                ↓   (z > Q_long OR z < Q_short)
           Trade Entry Candidate
                ↓
        Triple Barrier Labeling (volatility-aware, trend-adjusted)
                ↓
           Final Supervised Target
    
    Key Features:
    1. Entry gating via conditional quantile regression on z-scores
    2. Volatility-aware barriers: TP = k_TP × σ_rolling, SL = k_SL × σ_rolling
    3. Z-score magnitude scaling: k_TP = k_base × (1 + scale × |z|)
    4. Trend-aware adjustment: TP_adj = TP × (1 + α × trend_strength)
    5. Asymmetric multipliers for crypto (longs vs shorts)
    6. Clipping to avoid extreme TP/SL in low/high volatility
    7. Trailing profit support
    
    Args:
        df: OHLCV DataFrame with 'close', 'high', 'low' columns
        features: Feature matrix for conditional quantile model
        signals: Signal DataFrame with 'consensus' column
        volatility: Volatility series (e.g., volatility_1d)
        
        # Quantile parameters
        quantile_long: Upper quantile threshold for longs (default: 0.6)
        quantile_short: Lower quantile threshold for shorts (default: 0.35)
        asymmetric_crypto: Use asymmetric quantiles for crypto
        quantile_lookback: Rolling window for quantile model training
        quantile_min_train: Minimum samples before training
        quantile_retrain_freq: Retrain model every N bars
        
        # Barrier parameters
        k_tp_base: Base take-profit multiplier (× volatility)
        k_sl_base: Base stop-loss multiplier (× volatility)
        k_tp_long_mult: Additional TP multiplier for longs
        k_sl_long_mult: Additional SL multiplier for longs
        k_tp_short_mult: Additional TP multiplier for shorts
        k_sl_short_mult: Additional SL multiplier for shorts
        
        # Z-score scaling
        z_magnitude_scale: Scale factor for |z| adjustment
        
        # Trend adjustment (linear, not thresholds)
        trend_alpha: Trend adjustment factor (0.2-0.4 recommended)
        trend_lookback: Lookback for trend calculation
        
        # Clipping
        tp_min_mult: Minimum TP as multiple of base volatility
        tp_max_mult: Maximum TP as multiple of base volatility
        sl_min_mult: Minimum SL as multiple of base volatility
        sl_max_mult: Maximum SL as multiple of base volatility
        
        # Other parameters
        horizon: Maximum bars to look ahead
        transaction_cost: Transaction cost per trade
        min_event_spacing: Minimum bars between events
        atr_series: ATR series for trailing stops
        trail_distance_atr_mult: Trailing distance in ATR multiples
        feature_subset: Optional feature subset for quantile model
    
    Returns:
        Tuple of (labeled_data DataFrame, diagnostics dict)
    """
    n = len(df)
    
    # Initialize output DataFrame
    labeled_data = pd.DataFrame(index=df.index)
    labeled_data['close'] = df['close']
    
    diagnostics: Dict[str, Any] = {
        "n_signals": 0,
        "n_gated_long": 0,
        "n_gated_short": 0,
        "n_trades": 0,
        "quantile_diagnostics": {},
        "barrier_stats": {},
    }
    
    try:
        # =====================================================================
        # STEP 1: Compute z-scores and conditional quantile filter
        # =====================================================================
        tprint("📊 Step 1: Computing conditional z-scores and quantile filter...", "INFO")
        
        # Compute future returns for z-score calculation
        # rt+1:t+h = return over horizon period
        future_returns = df['close'].pct_change(horizon).shift(-horizon)
        
        # Compute volatility-normalized z-scores
        z_scores = compute_volatility_normalized_zscore(
            realized_returns=future_returns,
            volatility=volatility,
            vol_lookback=100,
            clip_zscore=5.0,
        )
        labeled_data['z_score'] = z_scores
        
        # Run conditional quantile regression
        cond_labels, labels_long, labels_short, quantile_diag = create_conditional_quantile_labels(
            realized_returns=future_returns,
            features=features,
            volatility=volatility,
            quantile_long=quantile_long,
            quantile_short=quantile_short,
            lookback_bars=quantile_lookback,
            min_train_samples=quantile_min_train,
            retrain_frequency=quantile_retrain_freq,
            vol_lookback=100,
            use_lightgbm=True,
            n_estimators=50,
            max_depth=4,
            feature_subset=feature_subset,
            asymmetric_crypto=asymmetric_crypto,
        )
        
        diagnostics["quantile_diagnostics"] = quantile_diag
        
        # Entry candidates: where quantile filter passes
        # labels_long == 1 means z > Q_long (good long candidate)
        # labels_short == 1 means z < Q_short (good short candidate)
        long_candidates = labels_long == 1.0
        short_candidates = labels_short == 1.0
        
        n_long_cand = int(long_candidates.sum())
        n_short_cand = int(short_candidates.sum())
        
        tprint(f"   Entry candidates: {n_long_cand} longs, {n_short_cand} shorts", "INFO")
        
        # =====================================================================
        # STEP 2: Compute volatility-aware barriers with z-score & trend scaling
        # =====================================================================
        tprint("📊 Step 2: Computing volatility-aware barriers...", "INFO")
        
        # Rolling volatility for barrier computation
        rolling_vol = volatility.rolling(50, min_periods=10).mean()
        labeled_data['rolling_vol'] = rolling_vol
        
        # Trend strength: EMA slope normalized (linear, not threshold)
        ema_fast = df['close'].ewm(span=10).mean()
        ema_slow = df['close'].ewm(span=30).mean()
        ema_slope = (ema_fast - ema_slow) / df['close']
        
        # Z-score momentum (rolling mean of z-scores)
        z_momentum = z_scores.rolling(trend_lookback, min_periods=5).mean().fillna(0)
        
        # Combined trend strength (normalized to roughly [-1, 1])
        trend_strength = (ema_slope / (rolling_vol + 1e-8)).clip(-2, 2) / 2
        labeled_data['trend_strength'] = trend_strength
        
        # Compute per-bar barrier thresholds
        z_abs = z_scores.abs().fillna(0)
        
        # Base barriers (in price %)
        tp_base = k_tp_base * rolling_vol
        sl_base = k_sl_base * rolling_vol
        
        # Z-score magnitude scaling: k = k_base × (1 + scale × |z|)
        # Stronger signals → larger targets
        z_scale_factor = 1.0 + z_magnitude_scale * z_abs.clip(0, 3)
        
        # Direction-specific multipliers
        # For longs: apply long multipliers
        # For shorts: apply short multipliers
        consensus = signals['consensus'] if 'consensus' in signals.columns else pd.Series(0, index=df.index)
        
        # Compute direction-aware TP/SL
        tp_scaled = pd.Series(index=df.index, dtype=float)
        sl_scaled = pd.Series(index=df.index, dtype=float)
        
        # Long positions
        long_mask = consensus > 0
        tp_scaled.loc[long_mask] = tp_base.loc[long_mask] * k_tp_long_mult * z_scale_factor.loc[long_mask]
        sl_scaled.loc[long_mask] = sl_base.loc[long_mask] * k_sl_long_mult
        
        # Short positions
        short_mask = consensus < 0
        tp_scaled.loc[short_mask] = tp_base.loc[short_mask] * k_tp_short_mult * z_scale_factor.loc[short_mask]
        sl_scaled.loc[short_mask] = sl_base.loc[short_mask] * k_sl_short_mult
        
        # =====================================================================
        # STEP 3: Trend-aware adjustment (LINEAR, not thresholds)
        # =====================================================================
        # TP_adj = TP × (1 + α × trend_strength)
        # For longs: positive trend → increase TP, negative trend → decrease TP
        # For shorts: negative trend → increase TP, positive trend → decrease TP
        
        trend_adjustment_long = 1.0 + trend_alpha * trend_strength.clip(-1, 1)
        trend_adjustment_short = 1.0 - trend_alpha * trend_strength.clip(-1, 1)  # Inverted for shorts
        
        tp_trend_adjusted = pd.Series(index=df.index, dtype=float)
        tp_trend_adjusted.loc[long_mask] = tp_scaled.loc[long_mask] * trend_adjustment_long.loc[long_mask]
        tp_trend_adjusted.loc[short_mask] = tp_scaled.loc[short_mask] * trend_adjustment_short.loc[short_mask]
        
        # =====================================================================
        # STEP 4: Clipping to avoid extreme TP/SL
        # =====================================================================
        tp_min = tp_min_mult * rolling_vol
        tp_max = tp_max_mult * rolling_vol
        sl_min = sl_min_mult * rolling_vol
        sl_max = sl_max_mult * rolling_vol
        
        tp_final = tp_trend_adjusted.clip(lower=tp_min, upper=tp_max)
        sl_final = sl_scaled.clip(lower=sl_min, upper=sl_max)
        
        # Ensure minimum absolute thresholds
        tp_final = tp_final.clip(lower=0.002)  # At least 0.2%
        sl_final = sl_final.clip(lower=0.001)  # At least 0.1%
        
        labeled_data['tp_threshold'] = tp_final
        labeled_data['sl_threshold'] = sl_final
        
        tprint(
            f"   Barrier stats: TP mean={tp_final.mean():.4f}, SL mean={sl_final.mean():.4f}",
            "INFO"
        )
        
        # =====================================================================
        # STEP 5: Apply triple barrier labeling with gated entries
        # =====================================================================
        tprint("📊 Step 3: Applying triple barrier labeling...", "INFO")
        
        # Create gated signals: only where quantile filter passes
        gated_signals = signals.copy()
        
        # Gate: only allow long signals where long_candidates == True
        # Gate: only allow short signals where short_candidates == True
        original_consensus = gated_signals['consensus'].copy()
        gated_consensus = pd.Series(0.0, index=df.index)
        
        # Apply long gate
        gated_consensus.loc[long_candidates & (original_consensus > 0)] = 1.0
        # Apply short gate
        gated_consensus.loc[short_candidates & (original_consensus < 0)] = -1.0
        
        gated_signals['consensus'] = gated_consensus
        
        n_gated_long = int((gated_consensus > 0).sum())
        n_gated_short = int((gated_consensus < 0).sum())
        diagnostics["n_gated_long"] = n_gated_long
        diagnostics["n_gated_short"] = n_gated_short
        
        tprint(f"   Gated signals: {n_gated_long} longs, {n_gated_short} shorts", "INFO")
        
        # Run triple barrier with volatility-aware thresholds
        (
            realized_returns,
            binary_labels,
            exit_reasons,
            event_durations,
            mfe_series,
            mae_series,
            binary_labels_long,
            binary_labels_short,
        ) = compute_realized_returns(
            df=df,
            signals=gated_signals,
            profit_threshold=tp_final,  # Volatility-aware, trend-adjusted
            stop_threshold=sl_final,    # Volatility-aware
            horizon=horizon,
            transaction_cost=transaction_cost,
            min_event_spacing=min_event_spacing,
            volatility_series=volatility,
            atr_series=atr_series,
            trail_distance_atr_mult=trail_distance_atr_mult,
        )
        
        # Store results
        labeled_data['realized_return'] = realized_returns
        labeled_data['binary_label'] = binary_labels
        labeled_data['binary_label_long'] = binary_labels_long
        labeled_data['binary_label_short'] = binary_labels_short
        labeled_data['exit_reason'] = exit_reasons
        labeled_data['event_duration'] = event_durations
        labeled_data['mfe'] = mfe_series
        labeled_data['mae'] = mae_series
        labeled_data['gated_consensus'] = gated_consensus
        
        # Compute final statistics
        n_trades = int(realized_returns.notna().sum())
        n_profitable = int((realized_returns > 0).sum())
        mean_return = float(realized_returns.mean()) if n_trades > 0 else 0.0
        
        diagnostics["n_trades"] = n_trades
        diagnostics["n_profitable"] = n_profitable
        diagnostics["win_rate"] = n_profitable / n_trades if n_trades > 0 else 0.0
        diagnostics["mean_return"] = mean_return
        diagnostics["barrier_stats"] = {
            "tp_mean": float(tp_final.mean()),
            "tp_std": float(tp_final.std()),
            "sl_mean": float(sl_final.mean()),
            "sl_std": float(sl_final.std()),
            "trend_alpha": trend_alpha,
            "z_magnitude_scale": z_magnitude_scale,
        }
        
        tprint(
            f"📊 Final results: {n_trades} trades, {n_profitable} profitable ({diagnostics['win_rate']:.1%}), "
            f"mean return={mean_return:.4f}",
            "INFO"
        )
        
    except Exception as e:
        tprint(f"⚠️ Z-score gated triple barrier labeling failed: {e}", "WARNING")
        import traceback
        traceback.print_exc()
    
    return labeled_data, diagnostics


def diagnose_quantile_lookahead_bias(
    vol_scaled: pd.Series,
    low_q: float = 0.3,
    high_q: float = 0.7,
    print_results: bool = True,
) -> Dict[str, Any]:
    """Diagnose look-ahead bias in quantile-based labeling.
    
    Computes per-year statistics to detect if global quantile thresholds
    cause systematic bias across time periods (a sign of look-ahead bias).
    
    Args:
        vol_scaled: Volatility-scaled returns series (must have DatetimeIndex)
        low_q: Lower quantile threshold
        high_q: Upper quantile threshold
        print_results: If True, print diagnostic summary
    
    Returns:
        Dictionary with per-year statistics and bias indicators
    """
    diagnostics: Dict[str, Any] = {
        "global_thresholds": {},
        "per_year": {},
        "bias_detected": False,
        "bias_severity": "none",
        "recommendation": "",
    }
    
    try:
        v = vol_scaled.dropna()
        if len(v) < 50:
            diagnostics["error"] = "Insufficient data for diagnosis"
            return diagnostics
        
        # Global thresholds (what global quantile labeling uses)
        global_low = float(v.quantile(low_q))
        global_high = float(v.quantile(high_q))
        global_median = float(v.median())
        
        diagnostics["global_thresholds"] = {
            "low_q": low_q,
            "high_q": high_q,
            "low_val": global_low,
            "high_val": global_high,
            "median": global_median,
        }
        
        # Per-year analysis
        if not isinstance(vol_scaled.index, pd.DatetimeIndex):
            diagnostics["error"] = "Index must be DatetimeIndex for per-year analysis"
            return diagnostics
        
        years = vol_scaled.index.year
        unique_years = sorted(years.unique())
        
        year_stats = []
        for year in unique_years:
            year_mask = years == year
            year_data = vol_scaled[year_mask].dropna()
            
            if len(year_data) < 20:
                continue
            
            year_median = float(year_data.median())
            year_q30 = float(year_data.quantile(low_q))
            year_q70 = float(year_data.quantile(high_q))
            
            # How does this year's data compare to global thresholds?
            n_above_global_high = int((year_data >= global_high).sum())
            n_below_global_low = int((year_data <= global_low).sum())
            n_total = len(year_data)
            
            pct_positive_global = n_above_global_high / n_total if n_total > 0 else 0
            pct_negative_global = n_below_global_low / n_total if n_total > 0 else 0
            
            stats = {
                "year": int(year),
                "n_samples": n_total,
                "median": year_median,
                f"q{int(low_q*100)}": year_q30,
                f"q{int(high_q*100)}": year_q70,
                "pct_positive_global_thresh": pct_positive_global,
                "pct_negative_global_thresh": pct_negative_global,
                "median_vs_global_high": year_median - global_high,
            }
            year_stats.append(stats)
            diagnostics["per_year"][int(year)] = stats
        
        # Detect bias: if early years have very high positive rates
        if len(year_stats) >= 2:
            early_years = year_stats[:len(year_stats)//2]
            late_years = year_stats[len(year_stats)//2:]
            
            early_pos_rate = np.mean([s["pct_positive_global_thresh"] for s in early_years])
            late_pos_rate = np.mean([s["pct_positive_global_thresh"] for s in late_years])
            
            early_neg_rate = np.mean([s["pct_negative_global_thresh"] for s in early_years])
            late_neg_rate = np.mean([s["pct_negative_global_thresh"] for s in late_years])
            
            diagnostics["early_vs_late"] = {
                "early_years": [s["year"] for s in early_years],
                "late_years": [s["year"] for s in late_years],
                "early_positive_rate": early_pos_rate,
                "late_positive_rate": late_pos_rate,
                "early_negative_rate": early_neg_rate,
                "late_negative_rate": late_neg_rate,
                "positive_rate_diff": early_pos_rate - late_pos_rate,
            }
            
            # Detect bias severity
            pos_diff = early_pos_rate - late_pos_rate
            if pos_diff > 0.4 or early_pos_rate > 0.9:
                diagnostics["bias_detected"] = True
                diagnostics["bias_severity"] = "severe"
                diagnostics["recommendation"] = (
                    "SEVERE look-ahead bias detected. Early years have nearly 100% positive labels. "
                    "Use rolling quantiles (use_rolling_quantiles=True) to eliminate bias."
                )
            elif pos_diff > 0.2:
                diagnostics["bias_detected"] = True
                diagnostics["bias_severity"] = "moderate"
                diagnostics["recommendation"] = (
                    "Moderate look-ahead bias detected. Consider using rolling quantiles."
                )
            elif pos_diff > 0.1:
                diagnostics["bias_detected"] = True
                diagnostics["bias_severity"] = "mild"
                diagnostics["recommendation"] = (
                    "Mild temporal drift detected. Rolling quantiles recommended for robustness."
                )
            else:
                diagnostics["bias_severity"] = "none"
                diagnostics["recommendation"] = (
                    "No significant look-ahead bias detected. Global quantiles are acceptable."
                )
        
        if print_results:
            print("\n" + "=" * 70)
            print("QUANTILE LOOK-AHEAD BIAS DIAGNOSTIC")
            print("=" * 70)
            print(f"\nGlobal Thresholds (computed across ALL data):")
            print(f"  q{int(low_q*100)} (negative threshold): {global_low:.4f}")
            print(f"  q{int(high_q*100)} (positive threshold): {global_high:.4f}")
            print(f"  median: {global_median:.4f}")
            
            print(f"\nPer-Year Statistics:")
            print("-" * 70)
            print(f"{'Year':<6} {'N':<6} {'Median':<10} {'q30':<10} {'q70':<10} {'%Pos(global)':<12} {'%Neg(global)':<12}")
            print("-" * 70)
            for s in year_stats:
                print(
                    f"{s['year']:<6} {s['n_samples']:<6} {s['median']:<10.4f} "
                    f"{s[f'q{int(low_q*100)}']:<10.4f} {s[f'q{int(high_q*100)}']:<10.4f} "
                    f"{s['pct_positive_global_thresh']*100:<12.1f} "
                    f"{s['pct_negative_global_thresh']*100:<12.1f}"
                )
            
            print("\n" + "-" * 70)
            if diagnostics["bias_detected"]:
                print(f"⚠️  BIAS DETECTED: {diagnostics['bias_severity'].upper()}")
            else:
                print("✅ No significant bias detected")
            print(f"\n{diagnostics['recommendation']}")
            print("=" * 70 + "\n")
    
    except Exception as e:
        diagnostics["error"] = str(e)
        if print_results:
            print(f"⚠️ Diagnosis failed: {e}")
    
    return diagnostics


def create_regime_aware_quantile_labels_from_vol_scaled_returns(
    vol_scaled: pd.Series,
    regimes: Optional[pd.Series] = None,
    low_q: float = 0.3,
    high_q: float = 0.8,
    min_samples_per_regime: int = 50,
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

    # If no labels were assigned in any regime (e.g. when each regime has too
    # few samples to satisfy min_samples_per_regime), fall back to global
    # quantile-based labeling so that we still obtain a usable label set.
    if labels.dropna().empty:
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
    zigzag_features = None
    try:
        labeler = TrendAwareMetaLabeler()
        zigzag_single = labeler.detect_zigzag_trend(df)
        mtf_config = MultiTimeframeConfig()
        zigzag_mtf = labeler.detect_zigzag_multi_timeframe(df, mtf_config=mtf_config, base_zigzag=zigzag_single)
        zigzag_features = zigzag_single.join(zigzag_mtf, how="outer", rsuffix="_mtf")
    except Exception:
        zigzag_features = None

    # Hard-align df and signals to a shared tail window to avoid any
    # length mismatch when assigning signal-based features. We align
    # positionally (most recent data) and then construct features on
    # this aligned window.
    len_df = len(df)
    len_sig = len(signals)

    if len_df != len_sig:
        try:
            target_len = min(len_df, len_sig)
            if target_len <= 0:
                raise ValueError("[meta_features] Non-positive target length after alignment")

            tprint(
                f"⚠️ [meta_features] df length={len_df} != signals length={len_sig}; "
                f"using shared tail window of length={target_len}",
                "WARNING",
            )

            if len_df > target_len:
                df = df.iloc[-target_len:, :]
            if len_sig > target_len:
                signals = signals.iloc[-target_len:, :]
        except Exception as align_exc:
            tprint(f"⚠️ [meta_features] Failed to align df/signals by tail: {align_exc}", "WARNING")

    # After alignment, enforce identical index by resetting to a simple
    # RangeIndex so that downstream operations are purely positional and
    # not affected by duplicate datetime labels.
    if (not df.index.equals(signals.index)) or df.index.has_duplicates or signals.index.has_duplicates:
        df = df.reset_index(drop=True)
        signals = signals.reset_index(drop=True)
        if zigzag_features is not None:
            if len(zigzag_features) > len(df):
                zigzag_features = zigzag_features.iloc[-len(df):, :]
            zigzag_features = zigzag_features.reset_index(drop=True)

    features = pd.DataFrame(index=df.index)
    n_features = len(features)
    if zigzag_features is not None:
        if len(zigzag_features) > n_features:
            zigzag_features = zigzag_features.iloc[-n_features:, :]
        elif len(zigzag_features) < n_features:
            pad = pd.DataFrame(
                np.nan,
                index=range(n_features - len(zigzag_features)),
                columns=zigzag_features.columns,
            )
            zigzag_features = pd.concat([pad, zigzag_features], axis=0, ignore_index=True)
        for col in zigzag_features.columns:
            if col not in features.columns:
                features[col] = zigzag_features[col].to_numpy()

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
        # External regime features are no longer attached directly to the
        # meta-feature matrix; they remain available on market_data for
        # diagnostics and specialized consumers.
        pass
    except Exception as e_reg:
        tprint(f"⚠️ Warning: Could not attach external regime features: {e_reg}", "WARNING")

    try:
        banned_cols = []
        if "hmm_regime_label_1h" in features.columns:
            banned_cols.append("hmm_regime_label_1h")
        banned_cols.extend(
            [c for c in features.columns if c.startswith("regime_") and c.endswith("_prob")]
        )
        banned_cols.extend(
            [c for c in features.columns if c.startswith("hmm_alpha_")]
        )
        if banned_cols:
            features = features.drop(columns=list(set(banned_cols)), errors="ignore")
    except Exception:
        pass

    # ===== SPECIALIST SCALAR FEATURES (canonical per-specialist signals) =====

    # When get_specialist_models_outputs(..., use_canonical_specialist_scalars=True)
    # is used upstream, several canonical scalar columns become available on the
    # market_data frame (df). Here we expose them directly as meta-features so
    # that the meta-model can leverage specialist risk/liquidity/MR/path/macro
    # context without relying on raw multi-column regime blocks.
    try:
        specialist_cols: List[str] = []

        # Explicit canonical scalar names used across the unified training
        # pipelines. Only attach those that are actually present on df.
        for col in [
            "risk_score",
            "path_risk_score",
            "macro_trend_score_continuous",
            "mr_probability_dense",
            "mr_probability",
            "mr_raw_score",
            "mr_trend_state",
            "mr_trend_is_mr",
            "sr_labeling_xgb_prob",
            "vol_force_scalar",
            "smc_predicted",
        ]:
            if col in df.columns:
                specialist_cols.append(col)

        # Include any remaining scalar MR/SMC-style columns if they follow the
        # standard prefixes. This keeps the feature surface aligned with
        # specialist diagnostics without hard-coding every variant.
        specialist_cols.extend(
            [c for c in df.columns if c.startswith("mr_") or c.startswith("smc_")]
        )

        # De-duplicate while preserving order, then attach as-is. At this
        # point df and features share the same index, so direct assignment is
        # safe and avoids additional alignment work.
        seen: set[str] = set()
        specialist_cols_unique: List[str] = []
        for c in specialist_cols:
            if c not in seen:
                seen.add(c)
                specialist_cols_unique.append(c)

        for col in specialist_cols_unique:
            if col not in features.columns:
                features[col] = df[col]
    except Exception:
        # Specialist features are optional; never let failures here break the
        # core meta-feature pipeline.
        pass

    # ===== KALMAN-FILTERED TECHNICAL INDICATORS =====

    # Compute raw indicators
    df_local = df.copy()
    df_local['rsi'] = compute_rsi(df_local['close'], period=14)
    df_local['sma_fast'] = df_local['close'].rolling(10).mean()
    df_local['sma_slow'] = df_local['close'].rolling(30).mean()
    df_local['momentum'] = df_local['close'].pct_change(10)

    if use_kalman:
        # Helper to align any 1D array/Series to the feature index length
        def _align_to_features(arr: Any, n: int) -> np.ndarray:
            values = np.asarray(arr)
            if len(values) == n:
                return values
            if len(values) > n:
                return values[:n]
            padded = np.full(n, np.nan, dtype=float)
            padded[: len(values)] = values
            return padded

        n_features = len(features)

        # Kalman-filtered trend
        kalman_trend, kalman_uncertainty = kalman_smooth_trend(df['close'], Q=1e-5, R=0.01)
        kalman_trend_values = _align_to_features(kalman_trend, n_features)
        kalman_uncertainty_values = _align_to_features(kalman_uncertainty, n_features)
        features['kalman_trend'] = kalman_trend_values
        features['kalman_uncertainty'] = kalman_uncertainty_values

        # Kalman-filtered RSI
        kf_rsi = KalmanFilter1D(Q=1e-4, R=0.1, initial_value=50.0)
        kalman_rsi, _ = kf_rsi.filter_series(df_local['rsi'])
        kalman_rsi_values = _align_to_features(kalman_rsi, n_features)
        features['rsi_kalman'] = kalman_rsi_values

        # Kalman-filtered MA distance
        ma_distance = df_local['sma_fast'] - df_local['sma_slow']
        kf_ma = KalmanFilter1D(Q=1e-5, R=0.01, initial_value=0.0)
        kalman_ma_distance, _ = kf_ma.filter_series(ma_distance)
        kalman_ma_distance_values = _align_to_features(kalman_ma_distance, n_features)
        features['ma_distance_kalman'] = kalman_ma_distance_values

        # Kalman-filtered momentum
        kf_mom = KalmanFilter1D(Q=1e-4, R=0.01, initial_value=0.0)
        kalman_momentum, _ = kf_mom.filter_series(df_local['momentum'])
        kalman_momentum_values = _align_to_features(kalman_momentum, n_features)
        features['momentum_kalman'] = kalman_momentum_values

        # Keep raw for reference (diagnostic purposes)
    else:
        # Use raw indicators
        features['rsi'] = df_local['rsi']
        features['ma_distance'] = df_local['sma_fast'] - df_local['sma_slow']
        features['momentum'] = df_local['momentum']

    # ===== VOLATILITY-NORMALIZED FEATURES =====

    # Normalize momentum and MA distance by current volatility
    vol_1h_series = features['volatility_1h'].replace(0, np.nan)  # Avoid division by zero

    if use_kalman:
        # Use the existing alignment helper to ensure all arrays match the
        # feature index length and avoid index-based alignment (which can
        # trigger tz-aware vs tz-naive issues).
        vol_1h_arr = _align_to_features(vol_1h_series, n_features)
        close_arr = _align_to_features(df['close'], n_features)

        features['momentum_per_vol'] = features['momentum_kalman'] / (vol_1h_arr + 1e-8)
        features['ma_distance_per_vol'] = features['ma_distance_kalman'] / (close_arr * vol_1h_arr + 1e-8)
    else:
        vol_1h = vol_1h_series
        features['momentum_per_vol'] = features['momentum'] / (vol_1h + 1e-8)
        features['ma_distance_per_vol'] = features['ma_distance'] / (df['close'] * vol_1h + 1e-8)

    # ===== TRADITIONAL VOLATILITY FEATURES (BACKWARD COMPATIBLE) =====

    returns = df['close'].pct_change()
    if use_kalman:
        # Align rolling volatility series to feature index length to avoid
        # reindex-on-duplicate-index errors when the underlying index has
        # duplicate timestamps.
        vol5_series = returns.rolling(5).std()
        vol20_series = returns.rolling(20).std()
        vol5 = _align_to_features(vol5_series, n_features)
        vol20 = _align_to_features(vol20_series, n_features)
        features['volatility_5'] = vol5
        features['volatility_20'] = vol20
    else:
        vol5_series = returns.rolling(5).std()
        vol20_series = returns.rolling(20).std()
        features['volatility_5'] = vol5_series.to_numpy()
        features['volatility_20'] = vol20_series.to_numpy()

    features['volatility_ratio'] = features['volatility_5'] / (features['volatility_20'] + 1e-8)

    # Use EMA for smoothing (vectorized - MUCH faster than row-by-row iteration)
    # EMA of squared returns, then take sqrt. Use aligned numpy arrays to avoid
    # triggering pandas reindex on duplicate indices.
    alpha = 0.1  # Same as previous manual calculation
    vol_ema_series = (returns**2).ewm(alpha=alpha, adjust=False).mean()
    if use_kalman:
        vol_ema = _align_to_features(vol_ema_series, n_features)
    else:
        vol_ema = vol_ema_series.to_numpy()
    features['volatility_ema'] = np.sqrt(vol_ema)

    # ===== TREND STRENGTH =====

    sma_10 = df['close'].rolling(10).mean()
    sma_slope_series = sma_10.pct_change(5)
    if use_kalman:
        features['sma_slope'] = _align_to_features(sma_slope_series, n_features)
    else:
        features['sma_slope'] = sma_slope_series.to_numpy()

    sma20 = df['close'].rolling(20).mean()
    price_vs_sma20_series = (df['close'] - sma20) / (sma20 + 1e-8)
    if use_kalman:
        features['price_vs_sma20'] = _align_to_features(price_vs_sma20_series, n_features)
    else:
        features['price_vs_sma20'] = price_vs_sma20_series.to_numpy()

    # ADX-like trend strength (simplified)
    high_low = df['high'] - df['low']
    atr_14_series = high_low.rolling(14).mean()
    if use_kalman:
        atr_14 = _align_to_features(atr_14_series, n_features)
        close_arr_for_atr = _align_to_features(df['close'], n_features)
    else:
        atr_14 = atr_14_series.to_numpy()
        close_arr_for_atr = df['close'].to_numpy()
    features['atr_14'] = atr_14
    features['atr_ratio'] = atr_14 / (close_arr_for_atr + 1e-8)

    # ===== VOLUME CONTEXT =====

    if volume_available and 'volume' in df.columns:
        vol_sma = df['volume'].rolling(20).mean()
        volume_ratio_series = df['volume'] / (vol_sma + 1e-8)
        volume_trend_series = df['volume'].rolling(5).mean() / (vol_sma + 1e-8)
        vol_price_corr_series = returns.rolling(20).corr(df['volume'].pct_change())

        vol_mean = df['volume'].rolling(96).mean()
        vol_std = df['volume'].rolling(96).std()
        volume_zscore_series = (df['volume'] - vol_mean) / (vol_std + 1e-8)

        volume_long_mean = df['volume'].rolling(96).mean()
        volume_spike_series = df['volume'] / (volume_long_mean + 1e-8)
        volume_spike_ema_series = volume_spike_series.ewm(span=20).mean()

        signed_volume_raw = np.sign(returns.fillna(0.0).to_numpy()) * df['volume'].to_numpy()
        signed_volume_series = pd.Series(signed_volume_raw)
        signed_volume_ema_series = signed_volume_series.ewm(span=20).mean()

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
        momentum_ema_series = mom10_series.ewm(span=5).mean()
        features['momentum_ema'] = _align_to_features(momentum_ema_series, n_features)
    else:
        features['momentum_5'] = mom5_series.to_numpy()
        features['momentum_10'] = mom10_series.to_numpy()
        features['momentum_20'] = mom20_series.to_numpy()
        features['momentum_ema'] = mom10_series.ewm(span=5).mean().to_numpy()

    # Autocorrelation of returns (lag 1) to capture trend vs mean reversion
    autocorr_window = 50
    lag = 1
    returns_for_autocorr = returns
    shifted_returns = returns_for_autocorr.shift(lag)
    autocorr_series = returns_for_autocorr.rolling(window=autocorr_window, min_periods=10).corr(shifted_returns)

    if use_kalman:
        features['return_autocorr_lag1_w50'] = _align_to_features(autocorr_series, n_features)
    else:
        features['return_autocorr_lag1_w50'] = autocorr_series.to_numpy()

    # ===== RANGE POSITION =====

    recent_high = df['high'].rolling(20).max()
    recent_low = df['low'].rolling(20).min()
    range_position_series = (df['close'] - recent_low) / (recent_high - recent_low + 1e-8)
    if use_kalman:
        features['range_position'] = _align_to_features(range_position_series, n_features)
    else:
        features['range_position'] = range_position_series.to_numpy()

    # VWAP-based mean-reversion distance
    if 'close' in df.columns and 'volume' in df.columns:
        try:
            dollar_volume = df['close'] * df['volume']
            cum_volume = df['volume'].cumsum()
            vwap_series = dollar_volume.cumsum() / (cum_volume + 1e-8)
            vwap_diff_series = df['close'] - vwap_series
            if use_kalman:
                features['close_minus_vwap'] = _align_to_features(vwap_diff_series, n_features)
            else:
                features['close_minus_vwap'] = vwap_diff_series.to_numpy()
        except Exception:
            pass

    # ===== ENTROPY (SIMPLE MEASURE) =====

    # Price entropy using returns distribution
    returns_abs = returns.abs().rolling(20).mean()
    returns_entropy_series = -returns_abs * np.log(returns_abs + 1e-8)
    if use_kalman:
        features['returns_entropy'] = _align_to_features(returns_entropy_series, n_features)
    else:
        features['returns_entropy'] = returns_entropy_series.to_numpy()

    # ===== TIME-BASED FEATURES =====

    if isinstance(df.index, pd.DatetimeIndex):
        features['hour'] = df.index.hour.to_numpy()
        features['day_of_week'] = df.index.dayofweek.to_numpy()

        # ===== SELECTIVE TIME-OF-DAY PATTERNS (NEW 2025-12-04) =====
        # Only 5 features: cyclical hour encoding + known good/bad patterns
        hour_arr = df.index.hour.to_numpy()
        dow_arr = df.index.dayofweek.to_numpy()

        # Cyclical encoding for hour (captures 24-hour cycle in 2 features)
        features['hour_sin'] = np.sin(2 * np.pi * hour_arr / 24.0)
        features['hour_cos'] = np.cos(2 * np.pi * hour_arr / 24.0)

        # Known good/bad hours (from diagnostics - high impact)
        # Best hours: 3, 5, 10 (win rate > 56%)
        # Worst hours: 0, 13, 19 (win rate < 45%)
        features['is_good_hour'] = np.isin(hour_arr, [3, 5, 10]).astype(float)
        features['is_bad_hour'] = np.isin(hour_arr, [0, 13, 19]).astype(float)

        # Sunday indicator (worst day at 39.5% win rate)
        features['is_sunday'] = (dow_arr == 6).astype(float)

    else:
        features['hour'] = 0
        features['day_of_week'] = 0
        features['hour_sin'] = 0.0
        features['hour_cos'] = 1.0
        features['is_good_hour'] = 0.0
        features['is_bad_hour'] = 0.0
        features['is_sunday'] = 0.0

    # ===== ORDER FLOW IMBALANCE (OFI) PROXY (NEW 2025-12-04) =====
    # Without direct order book data, we approximate OFI using price/volume patterns

    if 'volume' in df.columns:
        volume = df['volume']
        close = df['close']
        high = df['high']
        low = df['low']
        open_price = df.get('open', close)

        # 1. Cumulative Volume Delta (CVD) Proxy
        # Positive when close > open (buying pressure), negative when close < open
        price_direction = np.sign(close - open_price)
        signed_volume = volume * price_direction
        cvd_proxy = signed_volume.cumsum()
        cvd_normalized = (cvd_proxy - cvd_proxy.rolling(96).mean()) / (cvd_proxy.rolling(96).std() + 1e-8)

        if use_kalman:
            features['cvd_proxy'] = _align_to_features(cvd_normalized, n_features)
        else:
            features['cvd_proxy'] = cvd_normalized.to_numpy()

        # 2. Volume-Weighted Price Pressure
        # High volume at highs = distribution, high volume at lows = accumulation
        close_in_range = (close - low) / (high - low + 1e-8)  # 0 = at low, 1 = at high
        volume_pressure = (close_in_range - 0.5) * volume  # Positive at highs, negative at lows
        volume_pressure_ema = volume_pressure.ewm(span=20).mean()

        if use_kalman:
            features['volume_pressure'] = _align_to_features(volume_pressure_ema, n_features)
        else:
            features['volume_pressure'] = volume_pressure_ema.to_numpy()

        # 3. OFI Proxy: Buying vs Selling Pressure Ratio
        # Volume at bar high vs volume at bar low (approximated by body position)
        upper_wick = high - pd.concat([open_price, close], axis=1).max(axis=1)
        lower_wick = pd.concat([open_price, close], axis=1).min(axis=1) - low
        body = (close - open_price).abs()
        total_range = high - low + 1e-8

        # Rejection from highs (supply) vs rejection from lows (demand)
        supply_rejection = (upper_wick / total_range) * volume
        demand_rejection = (lower_wick / total_range) * volume
        ofi_proxy = (demand_rejection - supply_rejection).rolling(20).sum()
        ofi_normalized = ofi_proxy / (ofi_proxy.rolling(96).std() + 1e-8)

        if use_kalman:
            features['ofi_proxy'] = _align_to_features(ofi_normalized, n_features)
        else:
            features['ofi_proxy'] = ofi_normalized.to_numpy()

        # 4. Volume Imbalance: Buy Volume vs Sell Volume Estimation
        # Using close position in high-low range as proxy
        buy_volume = volume * close_in_range
        sell_volume = volume * (1 - close_in_range)
        volume_imbalance = (buy_volume - sell_volume) / (volume + 1e-8)
        volume_imbalance_ema = volume_imbalance.ewm(span=20).mean()

        if use_kalman:
            features['volume_imbalance'] = _align_to_features(volume_imbalance_ema, n_features)
        else:
            features['volume_imbalance'] = volume_imbalance_ema.to_numpy()

        # 5. Absorption Ratio: Volume at extremes vs mid-range
        is_at_extreme = (close_in_range < 0.2) | (close_in_range > 0.8)
        extreme_volume = volume.where(is_at_extreme, 0).rolling(20).sum()
        total_volume = volume.rolling(20).sum()
        absorption_ratio = extreme_volume / (total_volume + 1e-8)

        # 6. Trade Aggressor Ratio (proxy): smoothed share of "buy-side" volume
        trade_aggressor_ratio_series = close_in_range.ewm(span=20).mean()

        # 7. Liquidity Gaps: open vs previous close, normalized
        prev_close = close.shift(1)
        gap_raw = open_price - prev_close
        liquidity_gap_up_series = np.maximum(gap_raw, 0) / (prev_close + 1e-8)
        liquidity_gap_down_series = np.maximum(-gap_raw, 0) / (prev_close + 1e-8)
        liquidity_gap_abs_series = gap_raw.abs() / (atr_14_series + 1e-8)

        if use_kalman:
            features['absorption_ratio'] = _align_to_features(absorption_ratio, n_features)
            features['trade_aggressor_ratio'] = _align_to_features(trade_aggressor_ratio_series, n_features)
            features['liquidity_gap_up'] = _align_to_features(liquidity_gap_up_series, n_features)
            features['liquidity_gap_down'] = _align_to_features(liquidity_gap_down_series, n_features)
            features['liquidity_gap_abs'] = _align_to_features(liquidity_gap_abs_series, n_features)
        else:
            features['absorption_ratio'] = absorption_ratio.to_numpy()
            features['trade_aggressor_ratio'] = trade_aggressor_ratio_series.to_numpy()
            features['liquidity_gap_up'] = liquidity_gap_up_series.to_numpy()
            features['liquidity_gap_down'] = liquidity_gap_down_series.to_numpy()
            features['liquidity_gap_abs'] = liquidity_gap_abs_series.to_numpy()

    else:
        # No volume data - set defaults
        features['cvd_proxy'] = 0.0
        features['volume_pressure'] = 0.0
        features['ofi_proxy'] = 0.0
        features['volume_imbalance'] = 0.0
        features['absorption_ratio'] = 0.0
        features['trade_aggressor_ratio'] = 0.5
        features['liquidity_gap_up'] = 0.0
        features['liquidity_gap_down'] = 0.0
        features['liquidity_gap_abs'] = 0.0

    # Volatility / trend interaction features
    if 'kalman_trend' in features.columns and 'vol_ratio' in features.columns:
        features['kalman_trend_x_vol_ratio'] = features['kalman_trend'] * features['vol_ratio']
    if 'sma_slope' in features.columns and 'vol_ratio' in features.columns:
        features['sma_slope_x_vol_ratio'] = features['sma_slope'] * features['vol_ratio']
    if 'price_vs_sma20' in features.columns and 'vol_ratio' in features.columns:
        features['price_vs_sma20_x_vol_ratio'] = features['price_vs_sma20'] * features['vol_ratio']
    if 'range_position' in features.columns and 'vol_ratio' in features.columns:
        features['range_position_x_vol_ratio'] = features['range_position'] * features['vol_ratio']

    if 'consensus' in signals.columns:
        signal_consensus = signals['consensus']
        signal_active = (signal_consensus != 0).astype(int)

        if use_kalman:
            features['signal_active'] = _align_to_features(signal_active, n_features)
        else:
            features['signal_active'] = signal_active.to_numpy()

        idx = np.arange(len(df))
        last_signal_idx = np.where(signal_active.to_numpy() == 1, idx, np.nan)
        last_signal_idx_series = pd.Series(last_signal_idx, index=df.index).ffill()

        signal_age = idx - last_signal_idx_series.values
        signal_age[last_signal_idx_series.isna().values] = np.nan
        if use_kalman:
            features['bars_since_last_signal'] = _align_to_features(signal_age, n_features)
        else:
            features['bars_since_last_signal'] = signal_age

        density_50 = signal_consensus.abs().rolling(50).sum()
        if use_kalman:
            features['signal_density_50'] = _align_to_features(density_50, n_features)
        else:
            features['signal_density_50'] = density_50.to_numpy()
    else:
        features['signal_active'] = 0
        features['bars_since_last_signal'] = np.nan
        features['signal_density_50'] = 0.0

    base_signal_cols = [
        col for col in ['rsi', 'rsi_long', 'macd', 'macd_long', 'ma', 'mom']
        if col in signals.columns
    ]
    if base_signal_cols:
        abs_signals = signals[base_signal_cols].abs()
        if use_kalman:
            strength_series = abs_signals.sum(axis=1)
            count_series = (abs_signals > 0).sum(axis=1)
            features['signal_strength_all'] = _align_to_features(strength_series, n_features)
            features['signal_count_active'] = _align_to_features(count_series, n_features)
        else:
            features['signal_strength_all'] = abs_signals.sum(axis=1).to_numpy()
            features['signal_count_active'] = (abs_signals > 0).sum(axis=1).to_numpy()

        if 'rsi' in signals.columns and 'macd' in signals.columns:
            align_series = np.sign(signals['rsi'] * signals['macd']).replace(0, 0)
            if use_kalman:
                features['signal_rsi_macd_alignment'] = _align_to_features(align_series, n_features)
            else:
                features['signal_rsi_macd_alignment'] = align_series.to_numpy()

    if 'rsi_value' in signals.columns:
        rsi_dist_series = (signals['rsi_value'] - 50.0).abs()
        if use_kalman:
            features['signal_rsi_distance_50'] = _align_to_features(rsi_dist_series, n_features)
        else:
            features['signal_rsi_distance_50'] = rsi_dist_series.to_numpy()
    if 'rsi_long_value' in signals.columns:
        rsi_long_dist_series = (signals['rsi_long_value'] - 50.0).abs()
        if use_kalman:
            features['signal_rsi_long_distance_50'] = _align_to_features(rsi_long_dist_series, n_features)
        else:
            features['signal_rsi_long_distance_50'] = rsi_long_dist_series.to_numpy()
    if 'macd_hist_value' in signals.columns:
        macd_hist_abs_series = signals['macd_hist_value'].abs()
        if use_kalman:
            features['signal_macd_hist_abs'] = _align_to_features(macd_hist_abs_series, n_features)
        else:
            features['signal_macd_hist_abs'] = macd_hist_abs_series.to_numpy()
    if 'macd_hist_long_value' in signals.columns:
        macd_hist_long_abs_series = signals['macd_hist_long_value'].abs()
        if use_kalman:
            features['signal_macd_hist_long_abs'] = _align_to_features(macd_hist_long_abs_series, n_features)
        else:
            features['signal_macd_hist_long_abs'] = macd_hist_long_abs_series.to_numpy()
    if 'sma_fast_value' in signals.columns and 'sma_slow_value' in signals.columns:
        ma_dist_series = (signals['sma_fast_value'] - signals['sma_slow_value']) / (df['close'] + 1e-8)
        if use_kalman:
            features['signal_ma_distance_raw'] = _align_to_features(ma_dist_series, n_features)
        else:
            features['signal_ma_distance_raw'] = ma_dist_series.to_numpy()
    if 'momentum_value' in signals.columns:
        if use_kalman:
            features['signal_momentum_value'] = _align_to_features(signals['momentum_value'], n_features)
        else:
            features['signal_momentum_value'] = signals['momentum_value'].to_numpy()

    if 'trend_regime' in signals.columns:
        if use_kalman:
            features['trend_regime'] = _align_to_features(signals['trend_regime'], n_features)
        else:
            features['trend_regime'] = signals['trend_regime'].to_numpy()
    if 'candle_trend' in signals.columns:
        if use_kalman:
            features['candle_trend'] = _align_to_features(signals['candle_trend'], n_features)
        else:
            features['candle_trend'] = signals['candle_trend'].to_numpy()
    if 'candle_reversal' in signals.columns:
        if use_kalman:
            features['candle_reversal'] = _align_to_features(signals['candle_reversal'], n_features)
        else:
            features['candle_reversal'] = signals['candle_reversal'].to_numpy()

    # Targeted trend×signal interaction features
    if 'trend_regime' in features.columns and 'signal_macd_hist_abs' in features.columns:
        tr_arr = np.asarray(features['trend_regime'])
        macd_abs_arr = np.asarray(features['signal_macd_hist_abs'])
        features['signal_trend_regime_x_macd_hist_abs'] = tr_arr * macd_abs_arr
    if 'candle_trend' in features.columns and 'signal_rsi_distance_50' in features.columns:
        ct_arr = np.asarray(features['candle_trend'])
        rsi_dist_arr = np.asarray(features['signal_rsi_distance_50'])
        features['signal_candle_trend_x_rsi_distance_50'] = ct_arr * rsi_dist_arr

    # ===== CROSS-TIMEFRAME FEATURES (1H, 4H AGGREGATIONS) =====
    # Aggregate 15m data to higher timeframes for multi-horizon analysis

    # 1h aggregation (4 bars of 15m data)
    close_1h = df['close'].rolling(4).mean()
    high_1h = df['high'].rolling(4).max()
    low_1h = df['low'].rolling(4).min()

    returns_1h_series = close_1h.pct_change()
    momentum_1h_series = df['close'].pct_change(4)
    volatility_1h_agg_series = returns_1h_series.rolling(16).std()  # 16h of 1h bars
    range_1h_series = (high_1h - low_1h) / (close_1h + 1e-8)

    if use_kalman:
        features['returns_1h'] = _align_to_features(returns_1h_series, n_features)
        features['momentum_1h'] = _align_to_features(momentum_1h_series, n_features)
        features['volatility_1h_agg'] = _align_to_features(volatility_1h_agg_series, n_features)
        features['range_1h'] = _align_to_features(range_1h_series, n_features)
    else:
        features['returns_1h'] = returns_1h_series.to_numpy()
        features['momentum_1h'] = momentum_1h_series.to_numpy()
        features['volatility_1h_agg'] = volatility_1h_agg_series.to_numpy()
        features['range_1h'] = range_1h_series.to_numpy()

    # 4h aggregation (16 bars of 15m data)
    close_4h = df['close'].rolling(16).mean()
    high_4h = df['high'].rolling(16).max()
    low_4h = df['low'].rolling(16).min()

    returns_4h_series = close_4h.pct_change()
    momentum_4h_series = df['close'].pct_change(16)
    volatility_4h_agg_series = returns_4h_series.rolling(16).std()
    range_4h_series = (high_4h - low_4h) / (close_4h + 1e-8)

    if use_kalman:
        features['returns_4h'] = _align_to_features(returns_4h_series, n_features)
        features['momentum_4h'] = _align_to_features(momentum_4h_series, n_features)
        features['volatility_4h_agg'] = _align_to_features(volatility_4h_agg_series, n_features)
        features['range_4h'] = _align_to_features(range_4h_series, n_features)
    else:
        features['returns_4h'] = returns_4h_series.to_numpy()
        features['momentum_4h'] = momentum_4h_series.to_numpy()
        features['volatility_4h_agg'] = volatility_4h_agg_series.to_numpy()
        features['range_4h'] = range_4h_series.to_numpy()

    # ===== ROLLING WINDOW FEATURES (FOR TREE MODELS) =====
    # Trees work better with explicitly computed rolling statistics

    close_arr_full = _align_to_features(df['close'], len(features)) if use_kalman else df['close'].to_numpy()

    for window in [5, 10, 20, 50]:
        # Rolling returns statistics
        ret_mean_series = returns.rolling(window).mean()
        ret_std_series = returns.rolling(window).std()

        if use_kalman:
            features[f'returns_mean_{window}'] = _align_to_features(ret_mean_series, len(features))
            features[f'returns_std_{window}'] = _align_to_features(ret_std_series, len(features))
        else:
            features[f'returns_mean_{window}'] = ret_mean_series.to_numpy()
            features[f'returns_std_{window}'] = ret_std_series.to_numpy()

        # Rolling price statistics
        close_min_series = df['close'].rolling(window).min()
        close_max_series = df['close'].rolling(window).max()
        if use_kalman:
            close_min_arr = _align_to_features(close_min_series, len(features))
            close_max_arr = _align_to_features(close_max_series, len(features))
        else:
            close_min_arr = close_min_series.to_numpy()
            close_max_arr = close_max_series.to_numpy()

        features[f'close_min_{window}'] = close_min_arr
        features[f'close_max_{window}'] = close_max_arr

        close_range_arr = (close_max_arr - close_min_arr) / (close_arr_full + 1e-8)
        features[f'close_range_{window}'] = close_range_arr

        # Distance from recent high/low
        dist_high_arr = (close_arr_full - close_max_arr) / (close_arr_full + 1e-8)
        dist_low_arr = (close_arr_full - close_min_arr) / (close_arr_full + 1e-8)
        features[f'dist_from_recent_high_{window}'] = dist_high_arr
        features[f'dist_from_recent_low_{window}'] = dist_low_arr

    # ===== MORE INTERACTION FEATURES =====
    # Combine features to capture non-linear relationships

    # Volatility × Momentum interactions
    if 'volatility_1d' in features.columns and 'momentum_20' in features.columns:
        features['vol_momentum_interaction'] = features['volatility_1d'] * features['momentum_20']

    # Sharpe-like momentum/volatility ratios
    if 'volatility_1d' in features.columns and 'momentum_10' in features.columns:
        denom_10 = features['volatility_1d'].replace(0.0, np.nan)
        features['momentum_10_div_volatility_1d'] = features['momentum_10'] / (denom_10 + 1e-8)
    if 'volatility_1d' in features.columns and 'momentum_5' in features.columns:
        denom_5 = features['volatility_1d'].replace(0.0, np.nan)
        features['momentum_5_div_volatility_1d'] = features['momentum_5'] / (denom_5 + 1e-8)

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

        if use_kalman:
            # Align to feature index length to avoid length mismatches in
            # scenarios where df/signals underwent tail alignment.
            features['bars_since_last_event'] = _align_to_features(bars_since_event, len(features))
        else:
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
    # Cache static, label-independent meta-features (e.g., ZigZag, volatility,
    # external regimes) per unique (market_data, primary_signals, volume
    # availability) combination so that expensive computations are reused
    # across different labeling configurations.
    if not hasattr(build_meta_features_for_model, "_static_meta_cache"):
        setattr(build_meta_features_for_model, "_static_meta_cache", {})

    static_cache = getattr(build_meta_features_for_model, "_static_meta_cache")

    cache_key = (
        id(market_data),
        id(primary_signals),
        bool(volume_available),
    )
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
    event_positions = np.flatnonzero(event_mask.to_numpy())

    rolling_win_rate_50 = event_labels.rolling(window=50, min_periods=1).mean()
    rolling_mean_ret_50 = event_returns.rolling(window=50, min_periods=1).mean()

    win_rate_50_full = pd.Series(np.nan, index=market_data.index)
    mean_ret_50_full = pd.Series(np.nan, index=market_data.index)

    if len(event_positions) == len(rolling_win_rate_50):
        win_rate_50_full.iloc[event_positions] = rolling_win_rate_50.to_numpy()
        mean_ret_50_full.iloc[event_positions] = rolling_mean_ret_50.to_numpy()

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

        # Rolling histories over past events only. For TTO, we shift the rolling
        # window by one event so that the feature at event k depends only on
        # events strictly before k (no self-outcome leakage).
        rolling_r_multiple_50 = event_r_multiple.rolling(window=50, min_periods=1).mean()
        rolling_tto_50 = event_tto.rolling(window=50, min_periods=1).mean()
        rolling_tto_50_past = rolling_tto_50.shift(1)
        rolling_mfe_mae_ratio_50 = mfe_mae_ratio_series.rolling(window=50, min_periods=1).mean()

        r_mult_50_full = pd.Series(np.nan, index=market_data.index)
        tto_50_full = pd.Series(np.nan, index=market_data.index)
        mfe_mae_ratio_50_full = pd.Series(np.nan, index=market_data.index)

        if len(event_positions) == len(rolling_r_multiple_50):
            r_mult_50_full.iloc[event_positions] = rolling_r_multiple_50.to_numpy()
            # Use the past-only rolling TTO (shifted by one event) so that the
            # TTO feature at event k cannot incorporate the outcome of event k
            # itself.
            tto_50_full.iloc[event_positions] = rolling_tto_50_past.to_numpy()
            mfe_mae_ratio_50_full.iloc[event_positions] = rolling_mfe_mae_ratio_50.to_numpy()
    except Exception:
        r_mult_50_full = pd.Series(np.nan, index=market_data.index)
        tto_50_full = pd.Series(np.nan, index=market_data.index)
        mfe_mae_ratio_50_full = pd.Series(np.nan, index=market_data.index)

    # STEP 5: Create meta-features with Kalman filtering
    tprint("🔧 [5/13] Creating meta-features with Kalman filtering...", "INFO")
    if cache_key in static_cache:
        meta_features = static_cache[cache_key].copy()
    else:
        meta_features = create_meta_features(
            market_data,
            primary_signals,
            volume_available,
            include_raw_signals=False,  # CRITICAL: avoid circular behavior
            use_kalman=True,  # Enable Kalman filtering
        )
        static_cache[cache_key] = meta_features.copy()

    # Attach event-centric and label-history features
    event_meta_features = pd.DataFrame(index=market_data.index)
    event_meta_features['bars_since_last_event'] = bars_since_last_event
    event_meta_features['dist_from_recent_high_50'] = dist_from_recent_high_50
    event_meta_features['dist_from_recent_low_50'] = dist_from_recent_low_50
    event_meta_features['drawdown_100'] = drawdown_100
    event_meta_features['event_tto_mean_last_50'] = tto_50_full

    # Attach/overwrite event-centric features without triggering index-based
    # reindexing (which is sensitive to duplicate datetime labels). Reset to a
    # simple RangeIndex and align positionally to the meta_features length.
    emf = event_meta_features.reset_index(drop=True)
    n_meta = len(meta_features)
    if len(emf) > n_meta:
        # Align to the most recent window, consistent with tail alignment in
        # create_meta_features where df/signals are truncated from the tail.
        emf = emf.iloc[-n_meta:, :].reset_index(drop=True)
    elif len(emf) < n_meta:
        pad_rows = n_meta - len(emf)
        pad = pd.DataFrame(np.nan, index=range(pad_rows), columns=emf.columns)
        emf = pd.concat([pad, emf], axis=0, ignore_index=True)

    meta_features[event_meta_features.columns] = emf.to_numpy()

    try:
        if isinstance(horizon, (int, float)) and horizon > 0:
            horizon_int = int(horizon)
            close_series = market_data['close']
            returns_series = close_series.pct_change()
            meta_features[f'return_{horizon_int}b'] = close_series.pct_change(horizon_int)
            meta_features[f'return_std_{horizon_int}b'] = returns_series.rolling(horizon_int).std()
            slope_window = max(horizon_int // 2, 1)
            rolling_mean = close_series.rolling(horizon_int).mean()
            meta_features[f'sma_slope_{horizon_int}b'] = rolling_mean.pct_change(slope_window)
    except Exception:
        pass

    meta_features_model = prepare_feature_matrix(meta_features)
    n_features_before_forbidden = int(meta_features_model.shape[1])

    # Drop high-leakage structural features from the meta-model feature matrix.
    # These include ZigZag / Renko / Swing High-Low derivatives and raw
    # volatility level features that can create tautological relationships
    # with fixed or poorly normalised labels.
    forbidden_exact = {
        "vol_ratio",
        "vol_expansion",
        "returns_std_50",
        "volume_spike_ema",
        "event_r_multiple_mean_last_50",
    }
    forbidden_prefixes = ("zigzag_",)
    # Case-insensitive substrings for structural / memory / proxy features.
    forbidden_substrings = (
        # ZigZag / pivot / swing / Renko structure
        "zigzag",
        "pivot",
        "swing",
        "renko",
        # Memory-style rolling P&L features
        "last_50",
        "last_100",
        "cumulative",
        "streak",
        # Volatility ratio, signal density proxies
        "vol_expansion",
        "signal_density",
    )

    cols_to_drop: List[str] = []
    for col in list(meta_features_model.columns):
        col_str = str(col)
        col_lower = col_str.lower()
        if col_str in forbidden_exact:
            cols_to_drop.append(col_str)
            continue
        if any(col_str.startswith(pref) for pref in forbidden_prefixes):
            cols_to_drop.append(col_str)
            continue
        if any(sub in col_lower for sub in forbidden_substrings):
            cols_to_drop.append(col_str)

    if cols_to_drop:
        meta_features_model = meta_features_model.drop(columns=list(set(cols_to_drop)), errors="ignore")

    n_features_after_forbidden = int(meta_features_model.shape[1])

    meta_features_model_processed = meta_features_model
    if not isinstance(meta_feature_cfg, dict):
        meta_feature_cfg = {}

    # Default meta-feature engineering behaviour: enable robust scaling,
    # winsorisation, feature selection, and sample weighting unless explicitly
    # disabled via the config. This keeps the feature matrix numerically
    # stable for tree-based models and consistent with unified training.
    if 'enable_winsorisation' not in meta_feature_cfg:
        meta_feature_cfg['enable_winsorisation'] = True
    if 'enable_feature_selection' not in meta_feature_cfg:
        meta_feature_cfg['enable_feature_selection'] = True
    if 'enable_sample_weighting' not in meta_feature_cfg:
        meta_feature_cfg['enable_sample_weighting'] = True

    if meta_feature_cfg.get('enable_winsorisation', False):
        try:
            lower_q = float(meta_feature_cfg.get('winsor_lower_q', 0.01))
            upper_q = float(meta_feature_cfg.get('winsor_upper_q', 0.99))
            robust_window = int(meta_feature_cfg.get('robust_window', 256))
            robust_min_periods = int(
                meta_feature_cfg.get('robust_min_periods', max(1, robust_window // 4))
            )

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

    # Compact diagnostics: feature counts before/after structural drops and processing.
    try:
        n_features_final = int(meta_features_model_processed.shape[1])
        tprint(
            f"[META_FEATURES_DIAG] n_raw={n_features_before_forbidden}, "
            f"after_forbidden={n_features_after_forbidden}, final={n_features_final}",
            "INFO",
        )
        if selected_feature_names:
            n_sel = len(selected_feature_names)
            sample_names = selected_feature_names[: min(10, n_sel)]
            tprint(
                f"[META_FEATURES_DIAG] selected_features_count={n_sel}, sample={sample_names}",
                "INFO",
            )
    except Exception:
        # Never let diagnostics break the main feature pipeline.
        pass

    # Ensure the event-history TTO diagnostic remains available as a model feature
    critical_tto_feature = 'event_tto_mean_last_50'
    if critical_tto_feature in meta_features_model.columns and critical_tto_feature not in meta_features_model_processed.columns:
        meta_features_model_processed[critical_tto_feature] = meta_features_model[critical_tto_feature]
    if critical_tto_feature in meta_features_model_processed.columns and critical_tto_feature not in selected_feature_names:
        selected_feature_names.append(critical_tto_feature)

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
                    # Extract volatility series for inverse volatility weighting
                    volatility_series = None
                    if 'volatility_1h' in market_data.columns:
                        volatility_series = market_data['volatility_1h']

                    # Time decay halflife (default: 180 days = ~6 months)
                    time_decay_halflife = float(meta_feature_cfg.get('time_decay_halflife', 180.0))

                    sample_weights = compute_sample_weights_with_uniqueness(
                        event_start_times=event_start_times,
                        event_end_times=event_end_times,
                        y=binary_labels,
                        class_weight_mult=float(meta_feature_cfg.get('class_weight_mult', 5.0)),
                        volatility_series=volatility_series,
                        time_decay_halflife=time_decay_halflife,
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
    method: str = 'isotonic',
    econ_min_return_multiple: Optional[float] = None,
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

    # Remove NaN values and ignore economically trivial events (below cost floor).
    # When a custom econ_min_return_multiple is provided (e.g. by HPO), honor it;
    # otherwise fall back to the global ECON_MIN_RETURN_MULTIPLE constant.
    econ_mult = (
        float(econ_min_return_multiple)
        if econ_min_return_multiple is not None
        else float(ECON_MIN_RETURN_MULTIPLE)
    )
    econ_floor = econ_mult * DEFAULT_TRANSACTION_COST
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

        # Enhanced weighting: emphasize both large absolute returns AND high confidence
        try:
            # Component 1: Cost-aware weighting (large absolute returns)
            return_weights = np.abs(r_clean)

            # Component 2: Confidence weighting (distance from 0.5 probability)
            # High confidence predictions (near 0 or 1) get more weight
            confidence_weights = np.abs(p_clean - 0.5) * 2.0  # Range [0, 1]
            confidence_weights = confidence_weights ** 1.5  # Emphasize extreme confidence more

            # Combined weighting
            weights = return_weights * (1.0 + confidence_weights)

            if np.isfinite(weights).any() and weights.mean() > 0:
                weights = weights / weights.mean()
                iso.fit(p_clean, r_clean, sample_weight=weights)
            else:
                iso.fit(p_clean, r_clean)

            # Log weighting diagnostics
            if len(p_clean) > 10:
                high_conf_mask = confidence_weights > 0.75
                low_conf_mask = confidence_weights < 0.25
                if high_conf_mask.any():
                    avg_weight_high_conf = weights[high_conf_mask].mean()
                    avg_weight_low_conf = weights[low_conf_mask].mean() if low_conf_mask.any() else 0
                    tprint(f"  ✓ Calibration confidence weighting: high_conf={avg_weight_high_conf:.2f}, low_conf={avg_weight_low_conf:.2f}", "INFO")
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
    target_long = pd.Series(np.nan, index=realized_returns.index)
    target_short = pd.Series(np.nan, index=realized_returns.index)

    consensus = signals['consensus'].values

    # Align realized_returns, probabilities and signals on a common
    # tail-aligned window so that all arrays and masks have consistent
    # lengths. Use realized_returns as the reference length since it
    # defines the target index.
    n_rr = len(realized_returns)
    if n_rr == 0:
        empty = pd.Series(0.0, index=realized_returns.index)
        return empty.copy(), empty.copy()

    n_prob = len(probabilities)
    n_sig = len(signals)
    n_common = min(n_rr, n_prob, n_sig)
    rr_tail = realized_returns.iloc[-n_common:]
    sig_tail = signals.iloc[-n_common:]
    prob_tail = np.asarray(probabilities, dtype=float)[-n_common:]

    target_long_tail = pd.Series(np.nan, index=rr_tail.index)
    target_short_tail = pd.Series(np.nan, index=rr_tail.index)

    consensus = sig_tail['consensus'].to_numpy()

    # VECTORIZED: Predict on entire probability array at once (much faster)
    expected_returns = iso_regressor.predict(prob_tail)

    # DEBUG LOGGING: Check for anomalies in isotonic predictions
    n_nan = np.isnan(expected_returns).sum()
    n_inf = np.isinf(expected_returns).sum()
    if n_nan > 0 or n_inf > 0:
        tprint(f" WARNING: Isotonic predictions contain {n_nan} NaN and {n_inf} Inf values", "WARNING")
        expected_returns = np.nan_to_num(expected_returns, nan=0.0, posinf=0.1, neginf=-0.1)

    # Convert to net-of-cost expected returns
    # NOTE: We preserve negative expected returns to help downstream models learn
    # what to avoid. This allows regressors to predict negative outcomes.
    net_expected = expected_returns - cost_threshold

    # Apply symmetric clipping to avoid extreme outliers while preserving negative values
    final_targets = np.clip(net_expected, -0.15, 0.15)  # Symmetric range [-15%, +15%]

    # DEBUG LOGGING: Target statistics (compact summary)
    n_positive = (final_targets > 1e-6).sum()
    n_negative = (final_targets < -1e-6).sum()
    pct_positive = n_positive / len(final_targets) * 100 if len(final_targets) > 0 else 0
    pct_negative = n_negative / len(final_targets) * 100 if len(final_targets) > 0 else 0
    n_above_cost = (final_targets > cost_threshold).sum()
    pct_above_cost = n_above_cost / len(final_targets) * 100 if len(final_targets) > 0 else 0

    # Vectorized assignment based on signal direction (on aligned tail
    # window, purely positional to avoid index alignment issues)
    rr_tail_notna = ~rr_tail.isna().to_numpy()
    long_mask = (consensus > 0) & rr_tail_notna
    short_mask = (consensus < 0) & rr_tail_notna

    if len(final_targets) == len(long_mask):
        target_long_tail.iloc[long_mask] = final_targets[long_mask]
    if len(final_targets) == len(short_mask):
        target_short_tail.iloc[short_mask] = final_targets[short_mask]

    # DEBUG LOGGING: Verify assignment coverage (compact)
    # Count non-zero assignments (both positive and negative targets are meaningful)
    n_long_assigned = target_long_tail.notna().sum()
    n_short_assigned = target_short_tail.notna().sum()

    try:
        tprint(
            " [META_TARGETS] positive="
            f"{n_positive}/{len(final_targets)} ({pct_positive:.1f}%), "
            f"negative={n_negative}/{len(final_targets)} ({pct_negative:.1f}%), "
            f"above_cost={n_above_cost}/{len(final_targets)} ({pct_above_cost:.1f}%), "
            f"mean={final_targets.mean():.6f}, std={final_targets.std():.6f}, "
            f"min={final_targets.min():.6f}, max={final_targets.max():.6f}, "
            f"assigned_long={n_long_assigned}, assigned_short={n_short_assigned}",
            "INFO",
        )
    except Exception:
        logger.debug("Target diagnostics logging failed", exc_info=True)

    # Reindex tail targets back to the full realized_returns index so that
    # downstream code can safely assign them to labeled_data without index
    # length mismatches.
    target_long = target_long_tail.reindex(realized_returns.index)
    target_short = target_short_tail.reindex(realized_returns.index)

    return target_long, target_short


def generate_strategy_aware_targets(
    realized_returns: pd.Series,
    probabilities: np.ndarray,
    signals: pd.DataFrame,
    iso_regressor: IsotonicRegression,
    strategy_type: str = 'trend_following',
    cost_threshold: float = DEFAULT_TRANSACTION_COST,
) -> Tuple[pd.Series, pd.Series]:
    """
    Generate different targets/labels based on whether the strategy is trend following or mean reversion.
    
    Strategy-specific target generation:
    
    **Trend Following** (strategy_type='trend_following'):
    - Uses momentum signals (MACD, MA crossover, ATR breakout, volume spike)
    - Rewards following the established trend direction
    - Higher targets when momentum and MTF signals agree
    - Labels are based on continuation of price movement
    
    **Mean Reversion** (strategy_type='mean_reversion'):
    - Uses mean-reversion signals (Bollinger band fade, RSI extremes, range fade)
    - Rewards betting against extreme price movements
    - Higher targets when price is at extreme levels with reversal signals
    - Labels are based on price returning toward mean
    
    Args:
        realized_returns: Actual returns from triple barrier labeling
        probabilities: Predicted probabilities from meta-model
        signals: Signal directions with individual signal columns
        iso_regressor: Fitted isotonic regression model
        strategy_type: 'trend_following' or 'mean_reversion'
        cost_threshold: Transaction cost per trade
        
    Returns:
        Tuple of (target_long, target_short) with strategy-specific targets
    """
    target_long = pd.Series(0.0, index=realized_returns.index)
    target_short = pd.Series(0.0, index=realized_returns.index)
    
    n_rr = len(realized_returns)
    if n_rr == 0:
        return target_long.copy(), target_short.copy()
    
    # Align all arrays to common length
    n_prob = len(probabilities)
    n_sig = len(signals)
    n_common = min(n_rr, n_prob, n_sig)
    
    rr_tail = realized_returns.iloc[-n_common:]
    sig_tail = signals.iloc[-n_common:]
    prob_tail = np.asarray(probabilities, dtype=float)[-n_common:]
    
    target_long_tail = pd.Series(0.0, index=rr_tail.index)
    target_short_tail = pd.Series(0.0, index=rr_tail.index)
    
    # Base expected returns from isotonic regression
    expected_returns = iso_regressor.predict(prob_tail)
    expected_returns = np.nan_to_num(expected_returns, nan=0.0, posinf=0.1, neginf=-0.1)
    
    # Net-of-cost expected returns
    net_expected = expected_returns - cost_threshold
    
    if strategy_type == 'trend_following':
        # TREND FOLLOWING: Boost targets when momentum signals are strong
        # Use momentum-based signal columns
        momentum_cols = ['rsi', 'rsi_long', 'macd', 'macd_long', 'ma', 'mom', 
                        'atr_breakout', 'volume_spike', 'mtf_trend', 'mtf_confluence']
        
        # Calculate momentum agreement score (how many momentum signals agree)
        momentum_agreement = pd.Series(0.0, index=sig_tail.index)
        n_momentum_signals = 0
        
        for col in momentum_cols:
            if col in sig_tail.columns:
                momentum_agreement += sig_tail[col].fillna(0).abs()
                n_momentum_signals += 1
        
        # Normalize to [0, 1] range
        if n_momentum_signals > 0:
            momentum_agreement = momentum_agreement / n_momentum_signals
            momentum_agreement = momentum_agreement.clip(0, 1)
        
        # Apply momentum boost factor (1.0 to 1.5x based on momentum agreement)
        momentum_boost = 1.0 + 0.5 * momentum_agreement.values
        
        # For trend following, we reward strong momentum agreement
        final_targets = net_expected * momentum_boost
        
        # Additional boost when MTF confluence is positive
        if 'mtf_confluence' in sig_tail.columns:
            mtf_conf = sig_tail['mtf_confluence'].fillna(0).values
            mtf_boost = np.where(np.abs(mtf_conf) > 0.5, 1.2, 1.0)
            final_targets = final_targets * mtf_boost
        
        tprint(f"  📈 Trend Following targets: momentum_boost mean={momentum_boost.mean():.3f}", "INFO")
        
    elif strategy_type == 'mean_reversion':
        # MEAN REVERSION: Boost targets when mean-reversion signals are strong
        # Use mean-reversion signal columns
        mr_cols = ['bb_fade', 'range_fade', 'rsi_mr']
        
        # Calculate mean-reversion strength
        mr_strength = pd.Series(0.0, index=sig_tail.index)
        n_mr_signals = 0
        
        for col in mr_cols:
            if col in sig_tail.columns:
                mr_strength += sig_tail[col].fillna(0).abs()
                n_mr_signals += 1
        
        # Normalize to [0, 1] range
        if n_mr_signals > 0:
            mr_strength = mr_strength / n_mr_signals
            mr_strength = mr_strength.clip(0, 1)
        
        # Apply mean-reversion boost factor (1.0 to 1.5x based on MR strength)
        mr_boost = 1.0 + 0.5 * mr_strength.values
        
        # For mean reversion, check if price is at extremes (from range_position)
        if 'range_position' in sig_tail.columns:
            range_pos = sig_tail['range_position'].fillna(0.5).values
            # Boost when at extreme positions (< 0.2 or > 0.8)
            extreme_mask = (range_pos < 0.2) | (range_pos > 0.8)
            extreme_boost = np.where(extreme_mask, 1.3, 1.0)
            mr_boost = mr_boost * extreme_boost
        
        # For mean reversion, also boost when volatility is low (ranging market)
        if 'vol_ratio_for_consensus' in sig_tail.columns:
            vol_ratio = sig_tail['vol_ratio_for_consensus'].fillna(1.0).values
            # Low volatility ratio (< 0.8) favors mean reversion
            low_vol_boost = np.where(vol_ratio < 0.8, 1.2, 1.0)
            mr_boost = mr_boost * low_vol_boost
        
        final_targets = net_expected * mr_boost
        
        tprint(f"  📉 Mean Reversion targets: mr_boost mean={mr_boost.mean():.3f}", "INFO")
        
    else:
        # Default: use standard approach without strategy-specific modifications
        final_targets = net_expected
        tprint(f"  ⚠️ Unknown strategy_type '{strategy_type}', using default targets", "WARNING")
    
    # Apply symmetric clipping
    final_targets = np.clip(final_targets, -0.15, 0.15)
    
    # Assign targets based on signal direction
    consensus = sig_tail['consensus'].to_numpy() if 'consensus' in sig_tail.columns else np.zeros(n_common)
    rr_tail_notna = ~rr_tail.isna().to_numpy()
    
    long_mask = (consensus > 0) & rr_tail_notna
    short_mask = (consensus < 0) & rr_tail_notna
    
    if len(final_targets) == len(long_mask):
        target_long_tail.iloc[long_mask] = final_targets[long_mask]
    if len(final_targets) == len(short_mask):
        target_short_tail.iloc[short_mask] = final_targets[short_mask]
    
    # Reindex to full index
    target_long = target_long_tail.reindex(realized_returns.index).fillna(0.0)
    target_short = target_short_tail.reindex(realized_returns.index).fillna(0.0)
    
    # Log statistics
    n_long = (target_long != 0).sum()
    n_short = (target_short != 0).sum()
    tprint(f"  📊 Strategy-aware targets ({strategy_type}): {n_long} long, {n_short} short", "INFO")
    
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
    target_short: Optional[pd.Series] = None,
    selected_feature_names: Optional[List[str]] = None,
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
        selected_feature_names: Optional list of final selected feature names
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
    labeled_mask_arr = labeled_mask.to_numpy(dtype=bool, copy=False)
    n_labeled = int(labeled_mask_arr.sum())

    # Represent probabilities as Series aligned with index for richer diagnostics
    prob_series = pd.Series(probabilities, index=labeled_data.index)

    report_lines = []
    report_lines.append("# Meta-Labeling Diagnostics Report")
    report_lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"\n**Symbol:** {config.get('symbol', 'N/A')}")
    report_lines.append(f"**Timeframe:** {config.get('timeframe', 'N/A')}")
    report_lines.append(f"**Horizon:** {config.get('horizon', 'N/A')} bars")
    if config.get('trail_distance'):
        report_lines.append(f"**Trailing Distance:** {config.get('trail_distance')} ATR")
    # Compact summary of final feature set if available. Surface the full list
    # of selected feature names so that diagnostics consumers can inspect the
    # exact meta-feature surface used by the model.
    if isinstance(selected_feature_names, (list, tuple)) and selected_feature_names:
        try:
            n_feats = len(selected_feature_names)
            report_lines.append("\n**Final selected features (count):** ")
            report_lines.append(f"{n_feats} features\n")

            report_lines.append("\n**Final selected features (full list):**\n")
            for feat in selected_feature_names:
                report_lines.append(f"- {feat}")
        except Exception:
            # Never let diagnostics metadata break the main report
            pass

    report_lines.append("\n---\n")

    # ===== 0. SIGNAL FUNNEL (if available) =====
    signal_funnel = config.get('signal_funnel') or {}
    if isinstance(signal_funnel, dict) and signal_funnel:
        report_lines.append("\n## 0. Signal Funnel (Primary Signals)\n")
        # Use total_bars from signal_funnel when available; otherwise fall back to
        # the length of labeled_data to avoid referencing n_samples before it is defined.
        total_bars_sf = int(signal_funnel.get('total_bars', len(labeled_data)))
        raw_sf = int(signal_funnel.get('raw_signals', 0))
        final_sf = int(signal_funnel.get('final_signals', 0))
        ratio_sf = float(signal_funnel.get('raw_to_final_ratio', (final_sf / max(raw_sf, 1)) if raw_sf else 0.0))

        report_lines.append(f"- **Total bars:** {total_bars_sf}")
        report_lines.append(f"- **Raw non-zero signals:** {raw_sf}")
        report_lines.append(f"- **Final consensus signals:** {final_sf} (ratio={ratio_sf:.3f})")

        # Long/short breakdown
        rl = int(signal_funnel.get('raw_long_signals', 0))
        rs = int(signal_funnel.get('raw_short_signals', 0))
        fl = int(signal_funnel.get('final_long_signals', 0))
        fs = int(signal_funnel.get('final_short_signals', 0))
        extra = int(signal_funnel.get('relaxed_extra_signals', 0))

        report_lines.append(f"- **Raw long/short:** {rl}/{rs}")
        report_lines.append(f"- **Final long/short:** {fl}/{fs}")
        report_lines.append(f"- **Relaxed extra signals (strict=0 but raw≠0):** {extra}")

        # Densities per bar and per day (approximate)
        raw_density_bar = raw_sf / max(total_bars_sf, 1)
        final_density_bar = final_sf / max(total_bars_sf, 1)

        report_lines.append(f"- **Raw signal density:** {raw_density_bar:.5f} per bar")
        report_lines.append(f"- **Final signal density:** {final_density_bar:.5f} per bar")

        # If we have a DatetimeIndex, estimate per-day densities directly
        raw_per_day = None
        final_per_day = None
        if isinstance(labeled_data.index, pd.DatetimeIndex) and len(labeled_data.index) > 1:
            try:
                dt_start = labeled_data.index[0]
                dt_end = labeled_data.index[-1]
                days_span_sf = max((dt_end - dt_start).total_seconds() / 86400.0, 1e-6)
                raw_per_day = raw_sf / days_span_sf
                final_per_day = final_sf / days_span_sf
                report_lines.append(f"- **Raw signals per day (approx):** {raw_per_day:.3f}")
                report_lines.append(f"- **Final consensus signals per day (approx):** {final_per_day:.3f}")
            except Exception:
                pass

        # Heuristic warnings for extreme pruning or density
        if raw_sf > 0 and ratio_sf < 0.25:
            report_lines.append("\n⚠️ **Warning:** Consensus is pruning heavily (final/raw < 0.25). Consider relaxing signal gating.")
        if raw_sf > 0 and ratio_sf > 0.9:
            report_lines.append("\nℹ️ **Note:** Consensus preserves most raw signals (final/raw > 0.90).")

        if final_per_day is not None:
            if final_per_day < 0.3:
                report_lines.append(f"\n⚠️ **Warning:** Very sparse primary signals ({final_per_day:.3f} trades/day). Downstream labels may be too sparse.")
            elif final_per_day > 10.0:
                report_lines.append(f"\n⚠️ **Warning:** Very dense primary signals ({final_per_day:.3f} trades/day). Overlapping events may be frequent.")

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
    # Use positional masking to avoid index-alignment issues.
    features_clean = meta_features.iloc[labeled_mask_arr]
    features_clean = features_clean.select_dtypes(include=[np.number]).fillna(0)
    labels_clean = binary_labels.iloc[labeled_mask_arr]

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
        pre_mask_arr = pre_mask.to_numpy(dtype=bool, copy=False)
        n_pre_total = int(pre_mask_arr.sum())

        if n_pre_total > 0:
            tx_cost_local = config.get('transaction_cost', DEFAULT_TRANSACTION_COST)
            try:
                tx_cost_local = float(tx_cost_local)
            except Exception:
                tx_cost_local = float(DEFAULT_TRANSACTION_COST)

            # Raw pre-filter labels: simple economic sign after costs
            pre_returns = realized_returns.iloc[pre_mask_arr]
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
            returns_post = realized_returns.iloc[labeled_mask_arr]

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
            try:
                raw_label_pre_arr = raw_label_pre.to_numpy()
                for col in features_clean_pre.columns:
                    feat_vals_full = features_clean_pre[col].to_numpy()

                    # Tail-align lengths if they differ (defensive); operate positionally.
                    if feat_vals_full.shape[0] != pre_mask_arr.shape[0]:
                        min_len = min(feat_vals_full.shape[0], pre_mask_arr.shape[0])
                        if min_len <= 1:
                            continue
                        feat_vals = feat_vals_full[-min_len:]
                        mask_use = pre_mask_arr[-min_len:]
                    else:
                        feat_vals = feat_vals_full
                        mask_use = pre_mask_arr

                    if mask_use.sum() <= 1:
                        continue

                    feat_vals_pre = feat_vals[mask_use]
                    labels_arr = raw_label_pre_arr
                    if labels_arr.shape[0] != feat_vals_pre.shape[0]:
                        min_len2 = min(labels_arr.shape[0], feat_vals_pre.shape[0])
                        if min_len2 <= 1:
                            continue
                        feat_vals_pre = feat_vals_pre[-min_len2:]
                        labels_arr = labels_arr[-min_len2:]

                    if labels_arr.size <= 1:
                        continue

                    corr = np.corrcoef(feat_vals_pre, labels_arr)[0, 1]
                    if np.isfinite(corr):
                        correlations_pre[col] = float(corr)
            except Exception:
                correlations_pre = {}

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

    returns_labeled = realized_returns.iloc[labeled_mask_arr]
    labels_clean = binary_labels.iloc[labeled_mask_arr]

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
        # Calibration analysis (use positional masking to avoid index issues)
        prob_clean = prob_series.iloc[labeled_mask_arr]
        labels_clean_array = binary_labels.iloc[labeled_mask_arr].to_numpy()

        # Bin probabilities
        prob_bins = pd.cut(prob_clean, bins=10, labels=False)
        calibration_data = []

        for bin_idx in range(10):
            mask = (prob_bins == bin_idx)
            if mask.sum() > 0:
                mask_arr = mask.to_numpy(dtype=bool, copy=False)
                mean_prob = float(prob_clean.iloc[mask_arr].mean())
                mean_label = float(labels_clean_array[mask_arr].mean())
                count = int(mask_arr.sum())
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
                mask_bin_arr = mask_bin.to_numpy(dtype=bool, copy=False)
                mean_prob_bin = prob_clean.iloc[mask_bin_arr].mean()
                mean_ret_bin = returns_labeled.iloc[mask_bin_arr].mean()
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
        # Use positional mask to avoid index alignment issues
        smoothed_labeled = smoothed_labels.iloc[labeled_mask_arr]

        report_lines.append(f"- **Mean smoothed label:** {smoothed_labeled.mean():.3f}")
        report_lines.append(f"- **Median smoothed label:** {smoothed_labeled.median():.3f}")
        report_lines.append(f"- **Std smoothed label:** {smoothed_labeled.std():.3f}")

        # Correlation with binary labels (already aligned labeled subset)
        corr_smoothed_binary = smoothed_labeled.corr(labels_clean)
        report_lines.append(f"- **Correlation with binary labels:** {corr_smoothed_binary:.3f}")

        # Correlation with realized returns (labeled subset)
        corr_smoothed_returns = smoothed_labeled.corr(returns_labeled)
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
            # Positional mask to avoid index alignment issues
            exit_labeled = exit_reasons_series.iloc[labeled_mask_arr].dropna()
            total_events = len(exit_labeled)
            if total_events > 0:
                value_counts = exit_labeled.value_counts(normalize=True)
                report_lines.append("\n### Exit Reason Mix (Labeled Events)\n")
                for reason, frac in value_counts.items():
                    report_lines.append(f"- **{reason}:** {frac:.1%}")

        if durations_series is not None:
            # Coerce to numeric to avoid mixed int/Timestamp comparisons in quantiles
            dur_numeric = pd.to_numeric(durations_series, errors="coerce")
            dur_clean = dur_numeric.iloc[labeled_mask_arr].dropna()
            if len(dur_clean) > 0:
                report_lines.append("\n### Event Duration Distribution (Bars)\n")
                report_lines.append(f"- **Mean duration:** {dur_clean.mean():.2f}")
                report_lines.append(f"- **Median duration:** {dur_clean.median():.2f}")
                report_lines.append(f"- **90th percentile:** {dur_clean.quantile(0.9):.2f}")

        if stop_threshold_series is not None:
            # Coerce to numeric and compute R-multiples using numpy arrays to avoid
            # any dtype surprises from extension arrays.
            stop_numeric = pd.to_numeric(stop_threshold_series, errors="coerce")
            stop_labeled = stop_numeric.iloc[labeled_mask_arr]

            ret_arr = returns_labeled.to_numpy(dtype=float, copy=False)
            stop_arr = stop_labeled.to_numpy(dtype=float, copy=False)

            with np.errstate(divide='ignore', invalid='ignore'):
                denom = np.where(np.isnan(stop_arr) | (stop_arr == 0.0), np.nan, stop_arr) + 1e-8
                r_multiple_arr = ret_arr / denom

            r_multiple = pd.Series(r_multiple_arr, index=returns_labeled.index)
            r_multiple = r_multiple.replace([np.inf, -np.inf], np.nan)
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
        # Path Efficiency Ratio (PER) and Time-to-Outcome Ratio (TTO)
        if event_durations is not None:
            labeled_durations = event_durations.iloc[labeled_mask_arr].dropna()

            # Path Efficiency Ratio (PER)
            # - Restrict to profitable events only (net return > 0)
            # - Use denominator that includes both favorable and adverse excursions (|MFE| + |MAE|)
            # - Report robust statistics (median, quantiles, clipped mean) and outlier counts
            if mfe_series is not None and mae_series is not None:
                per_df = pd.DataFrame(
                    {
                        "ret": returns_labeled,
                        "mfe": mfe_series.iloc[labeled_mask_arr],
                        "mae": mae_series.iloc[labeled_mask_arr],
                    }
                ).dropna()

                # Restrict to genuinely profitable events
                per_df = per_df[per_df["ret"] > 0]

                if len(per_df) > 0:
                    denom = np.abs(per_df["mfe"]) + np.abs(per_df["mae"]) + 1e-6
                    per_values = (per_df["ret"].abs() / denom).replace([np.inf, -np.inf], np.nan).dropna()

                    if len(per_values) > 0:
                        report_lines.append("\n### Path Efficiency Ratio (PER)\n")

                        # Robust summary statistics
                        if len(per_values) > 10:
                            clip_threshold = float(per_values.quantile(0.995))
                            per_clipped = per_values.clip(upper=clip_threshold)
                            mean_per = float(per_clipped.mean())
                        else:
                            mean_per = float(per_values.mean())

                        median_per = float(per_values.median())
                        p90_per = float(per_values.quantile(0.90)) if len(per_values) > 1 else median_per
                        p99_per = float(per_values.quantile(0.99)) if len(per_values) > 1 else median_per

                        report_lines.append(f"- **Mean PER (clipped 99.5%):** {mean_per:.3f}")
                        report_lines.append(f"- **Median PER:** {median_per:.3f}")
                        report_lines.append(f"- **90th percentile PER:** {p90_per:.3f}")
                        report_lines.append(f"- **99th percentile PER:** {p99_per:.3f}")

                        # Outlier count for very high PER values
                        high_per_count = int((per_values > 5.0).sum())
                        if high_per_count > 0:
                            report_lines.append(f"- **Trades with PER > 5.0:** {high_per_count}")

                        # Health check based on median PER (typical winner path quality)
                        if median_per < 0.3:
                            report_lines.append(
                                "\n⚠️ **Alert:** Median PER < 0.3 indicates highly noisy / random-walk paths even for winners"
                            )
                        elif median_per < 0.5:
                            report_lines.append(
                                "\n⚠️ **Warning:** Median PER in [0.3, 0.5] – many winners meander significantly before paying off"
                            )
                        else:
                            report_lines.append("\n✅ **OK:** Good path efficiency on profitable events")

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
            labeled_mfe = mfe_series.iloc[labeled_mask_arr].dropna()
            labeled_mae = mae_series.iloc[labeled_mask_arr].dropna()
            
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
        # Combine long and short targets using purely positional, numeric arrays.
        n = len(labeled_data)
        combined_arr = np.zeros(n, dtype=float)

        def _tail_align_numeric(series: Optional[pd.Series], n: int) -> np.ndarray:
            if series is None or len(series) == 0:
                return np.zeros(n, dtype=float)
            num = pd.to_numeric(series, errors="coerce")
            arr = num.to_numpy(dtype=float, copy=False)
            if arr.size >= n:
                return arr[-n:]
            # Left-pad with zeros if shorter than labeled_data
            pad = np.zeros(n - arr.size, dtype=float)
            return np.concatenate([pad, arr])

        combined_arr += _tail_align_numeric(target_long, n)
        combined_arr += _tail_align_numeric(target_short, n)

        combined_targets = pd.Series(combined_arr, index=labeled_data.index)

        target_nonzero = combined_targets[combined_targets > 1e-6]
        target_std = float(combined_targets.std())
        target_mean = float(combined_targets.mean())
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

        # Use positional mask and numeric arrays for IC computation
        prob_labeled = prob_series.iloc[labeled_mask_arr].to_numpy(dtype=float, copy=False)
        ret_labeled = returns_labeled.to_numpy(dtype=float, copy=False)

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
            # Work on labeled subset positionally to avoid index misalignment
            regime_labeled = regime_series.iloc[labeled_mask_arr]
            report_lines.append("\n### Label Base-Rate by Volatility Regime\n")

            labels_arr = labels_clean.to_numpy(dtype=float, copy=False)
            returns_arr = returns_labeled.to_numpy(dtype=float, copy=False)

            for regime in sorted(regime_labeled.dropna().unique()):
                mask_reg = (regime_labeled == regime).to_numpy(dtype=bool, copy=False)
                if mask_reg.sum() < 10:
                    continue
                pos_rate_reg = float(labels_arr[mask_reg].mean())
                mean_ret_reg = float(returns_arr[mask_reg].mean())
                report_lines.append(
                    f"- **Regime {regime}:** positive={pos_rate_reg:.1%}, mean_return={mean_ret_reg:.2%}"
                )

        # Trend-conditional checks using price_vs_sma20 if available
        if 'price_vs_sma20' in meta_features.columns:
            trend_measure = meta_features['price_vs_sma20']
            trend_labeled = trend_measure.iloc[labeled_mask_arr]
            high_trend = trend_labeled.quantile(0.75)
            low_trend = trend_labeled.quantile(0.25)

            strong_up = trend_labeled >= high_trend
            strong_down = trend_labeled <= low_trend

            labels_arr = labels_clean.to_numpy(dtype=float, copy=False)
            returns_arr = returns_labeled.to_numpy(dtype=float, copy=False)

            if strong_up.sum() >= 10:
                report_lines.append("\n### Trend-Conditional (Price vs SMA20)\n")
                mask_up = strong_up.to_numpy(dtype=bool, copy=False)
                pos_up = float(labels_arr[mask_up].mean())
                mean_ret_up = float(returns_arr[mask_up].mean())
                report_lines.append(
                    f"- **Strong uptrend:** positive={pos_up:.1%}, mean_return={mean_ret_up:.2%}"
                )
            if strong_down.sum() >= 10:
                mask_down = strong_down.to_numpy(dtype=bool, copy=False)
                pos_down = float(labels_arr[mask_down].mean())
                mean_ret_down = float(returns_arr[mask_down].mean())
                report_lines.append(
                    f"- **Strong downtrend:** positive={pos_down:.1%}, mean_return={mean_ret_down:.2%}"
                )

        # Time-of-day / weekday conditional
        if 'hour' in meta_features.columns:
            hour_series = meta_features['hour']
            hour_labeled = hour_series.iloc[labeled_mask_arr]
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
            dow_series = meta_features['day_of_week']
            dow_labeled = dow_series.iloc[labeled_mask_arr]
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
        # Build unified target magnitude series (force numeric and positional alignment)
        target_long = labeled_data.get('target_long')
        target_short = labeled_data.get('target_short')
        n = len(labeled_data)

        def _tail_align_numeric(series: Optional[pd.Series], n: int) -> np.ndarray:
            if series is None or len(series) == 0:
                return np.zeros(n, dtype=float)
            num = pd.to_numeric(series, errors="coerce")
            arr = num.to_numpy(dtype=float, copy=False)
            if arr.size >= n:
                return arr[-n:]
            pad = np.zeros(n - arr.size, dtype=float)
            return np.concatenate([pad, arr])

        if target_long is not None or target_short is not None:
            long_arr = _tail_align_numeric(target_long, n)
            short_arr = _tail_align_numeric(target_short, n)
            target_vals = long_arr + short_arr

            target_mag = pd.Series(target_vals, index=labeled_data.index, dtype=float)

            trade_mask = labeled_mask & (target_mag > 0) & ~realized_returns.isna()
            target_trades = target_mag[trade_mask]
            returns_trades = realized_returns[trade_mask]

            if len(target_trades) > 0:
                corr_tr = float(target_trades.corr(returns_trades))
                mse_tr = float(np.mean((target_trades.to_numpy(dtype=float) - returns_trades.to_numpy(dtype=float)) ** 2))
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
                frac_below_cost = float((target_nz < tx_cost).mean())
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

                # Add regime-wise breakdown
                try:
                    regime_metrics = compute_regime_wise_metrics(
                        X=meta_features,
                        y=binary_labels,
                        y_pred=probabilities,
                        realized_returns=realized_returns
                    )

                    if 'volatility_regimes' in regime_metrics:
                        report_lines.append("\n### Volatility Regime Breakdown\n")
                        report_lines.append("| Regime | Samples | Pos Rate | AUC | Mean Ret | Sharpe |")
                        report_lines.append("|--------|---------|----------|-----|----------|--------|")

                        for regime in ['low', 'medium', 'high']:
                            if regime in regime_metrics['volatility_regimes']:
                                r = regime_metrics['volatility_regimes'][regime]
                                auc_str = f"{r['auc']:.3f}" if 'auc' in r else "N/A"
                                ret_str = f"{r['mean_return']:.2%}" if 'mean_return' in r else "N/A"
                                sharpe_str = f"{r['sharpe']:.2f}" if 'sharpe' in r else "N/A"
                                report_lines.append(
                                    f"| {regime.capitalize()} | {r['n_samples']} | {r['pos_rate']:.1%} | "
                                    f"{auc_str} | {ret_str} | {sharpe_str} |"
                                )

                        # Analyze regime dependencies
                        vol_regimes = regime_metrics['volatility_regimes']
                        if 'low' in vol_regimes and 'high' in vol_regimes:
                            low_pos_rate = vol_regimes['low']['pos_rate']
                            high_pos_rate = vol_regimes['high']['pos_rate']
                            if abs(high_pos_rate - low_pos_rate) > 0.3:
                                report_lines.append(
                                    f"\n⚠️ **Warning:** Large win-rate disparity between regimes "
                                    f"(low: {low_pos_rate:.1%}, high: {high_pos_rate:.1%}). "
                                    "Performance is highly regime-dependent."
                                )
                except Exception as e_regime:
                    report_lines.append(f"\n⚠️ Could not compute regime breakdown: {e_regime}")

            except Exception as e_metrics:
                report_lines.append(f"\n⚠️ Could not compute cost-aware metrics: {e_metrics}")

        # Threshold sweep P&L curves (purely positional masks)
        report_lines.append("\n### Threshold-Sweep P&L (Using Meta Probability)\n")
        report_lines.append("\n| Threshold | Trades | Mean Return | Cum Return | Sharpe (per trade) |")
        report_lines.append("|----------|--------|-------------|------------|---------------------|")

        prob_values = prob_series.to_numpy(dtype=float, copy=False)
        ret_values = realized_returns.to_numpy(dtype=float, copy=False)
        mask_labeled_arr = labeled_mask_arr
        mask_ret_not_nan = ~np.isnan(ret_values)

        # Evaluate a dense grid of thresholds in the operational region [0.50, 0.65]
        thresholds = [round(x, 2) for x in np.arange(0.5, 0.651, 0.01)]
        for thr in thresholds:
            mask_thr_arr = (prob_values >= thr) & mask_labeled_arr & mask_ret_not_nan
            n_trades_thr = int(mask_thr_arr.sum())
            if n_trades_thr == 0:
                report_lines.append(f"| {thr:.2f} | 0 | N/A | N/A | N/A |")
                continue
            ret_thr = ret_values[mask_thr_arr]
            mean_ret_thr = float(np.mean(ret_thr))
            std_ret_thr = float(np.std(ret_thr, ddof=1)) if n_trades_thr > 1 else 0.0
            cum_ret_thr = float(np.sum(ret_thr))
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
            years_arr = np.asarray(years, dtype=int)
            years_labeled = years_arr[labeled_mask_arr]

            prob_labeled_series = prob_series.iloc[labeled_mask_arr]

            report_lines.append("\n### Per-Year Label and Return Stability\n")
            for year in sorted(np.unique(years_labeled)):
                year_mask_arr = years_labeled == year
                if year_mask_arr.sum() < 20:
                    continue
                pos_rate_y = labels_clean.iloc[year_mask_arr].mean()
                mean_ret_y = returns_labeled.iloc[year_mask_arr].mean()
                try:
                    auc_y = roc_auc_score(
                        labels_clean.iloc[year_mask_arr],
                        prob_labeled_series.iloc[year_mask_arr],
                    )
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
                fold_mask_arr = np.zeros(len(labeled_data), dtype=bool)
                fold_mask_arr[test_idx] = True
                fold_mask_arr &= labeled_mask_arr
                if fold_mask_arr.sum() < 20:
                    continue
                pos_rate_f = binary_labels.iloc[fold_mask_arr].mean()
                mean_ret_f = realized_returns.iloc[fold_mask_arr].mean()
                try:
                    auc_f = roc_auc_score(
                        binary_labels.iloc[fold_mask_arr],
                        prob_series.iloc[fold_mask_arr],
                    )
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


def focal_loss_lgb(y_pred, dtrain, alpha=0.25, gamma=2.0):
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

    # LightGBM Python API expects a custom objective with signature
    #   func(y_pred, dtrain[, alpha, gamma])
    # where y_pred are raw scores and dtrain is a Dataset providing labels.

    # Extract labels from Dataset if available
    if hasattr(dtrain, "get_label"):
        y_true = dtrain.get_label()
    else:
        y_true = dtrain

    # Ensure numeric arrays
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    # Guard against unexpected alpha/gamma values
    if not np.isscalar(alpha) or alpha is None:
        alpha = 0.25
    if not np.isscalar(gamma) or gamma is None:
        gamma = 2.0

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
    class_weight_mult: float = 5.0,
    volatility_series: Optional[pd.Series] = None,
    time_decay_halflife: Optional[float] = None
) -> np.ndarray:
    """
    Compute sample weights combining class weighting, sequential bootstrapping,
    inverse volatility weighting, and time decay.

    Sequential Bootstrapping (de Prado): Down-weight samples that overlap heavily
    with others to prevent overfitting to clustered events (e.g., 10 trades during
    a single volatility spike).

    Inverse Volatility Weighting: Make a 0.5% move in low volatility as "important"
    to the loss function as a 2% move in high volatility. Counteracts model bias
    towards high-volatility patterns.

    Time Decay: Apply exponential decay to make recent data more important than
    older data, addressing regime drift and AUC instability across folds.

    NEW (2025-11-18): Critical for handling increased signal count from loosened filters.
    NEW (2025-11-20): Added inverse volatility weighting and time decay for regime stability.

    Args:
        event_start_times: Start timestamps for each event
        event_end_times: End timestamps for each event
        y: Binary labels
        class_weight_mult: Multiplier for positive class (5.0 = 5x more important)
        volatility_series: Optional volatility series for inverse weighting
        time_decay_halflife: Optional halflife (in days) for exponential time decay

    Returns:
        Sample weights array
    """
    def _to_utc_naive(series: pd.Series) -> pd.Series:
        if not isinstance(series, pd.Series):
            return series
        try:
            dt = pd.to_datetime(series, utc=True, errors="coerce")
            try:
                dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
            except Exception:
                dt = dt.dt.tz_localize(None)
            return dt
        except Exception:
            return series

    event_start_times = _to_utc_naive(event_start_times)
    event_end_times = _to_utc_naive(event_end_times)
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

    # ===== INVERSE VOLATILITY WEIGHTING =====
    # Makes a 0.5% move in low vol as important as a 2% move in high vol
    # Counteracts model bias towards high-volatility patterns
    if volatility_series is not None:
        try:
            vol_array = volatility_series.values
            vol_clean = vol_array[~np.isnan(vol_array)]

            if len(vol_clean) > 10:
                # Normalize volatility to [0, 1] using percentiles
                vol_min = np.percentile(vol_clean, 10)
                vol_max = np.percentile(vol_clean, 90)
                normalized_vol = np.clip((vol_array - vol_min) / (vol_max - vol_min + 1e-8), 0.1, 1.0)

                # Inverse weighting: low vol gets higher weight
                # Range: [1.0, 10.0] where low vol = 10x weight, high vol = 1x weight
                inv_vol_weights = 1.0 / normalized_vol

                # Normalize to have mean = 1.0
                inv_vol_weights = inv_vol_weights / inv_vol_weights.mean()

                final_weights = final_weights * inv_vol_weights

                avg_weight_low = inv_vol_weights[normalized_vol < 0.33].mean() if (normalized_vol < 0.33).any() else 0
                avg_weight_high = inv_vol_weights[normalized_vol > 0.67].mean() if (normalized_vol > 0.67).any() else 0
                tprint(f"  ✓ Inverse volatility weighting: low_vol_weight={avg_weight_low:.2f}, high_vol_weight={avg_weight_high:.2f}", "INFO")
        except Exception as e:
            tprint(f"  ⚠️ Inverse volatility weighting failed: {e}, skipping", "WARNING")

    # ===== LINEAR TIME DECAY =====
    # Recent data more important than older data (addresses regime drift)
    if time_decay_halflife is not None and time_decay_halflife > 0:
        try:
            # Use event start times for decay calculation
            if event_start_times is not None and len(event_start_times) > 0:
                start_times = event_start_times[labeled_mask]

                if len(start_times) > 0 and pd.notna(start_times).any():
                    # Get time range in days
                    latest_time = start_times.max()
                    time_diffs = (latest_time - start_times).dt.total_seconds() / (24 * 3600)  # Convert to days

                    # Exponential decay: weight = exp(-ln(2) * time_diff / halflife)
                    # This ensures weight halves every 'halflife' days
                    decay_weights = np.ones(len(y))
                    decay_weights[labeled_mask] = np.exp(-np.log(2) * time_diffs / time_decay_halflife)

                    # Normalize to mean = 1.0
                    decay_weights = decay_weights / decay_weights[labeled_mask].mean()

                    final_weights = final_weights * decay_weights

                    tprint(f"  ✓ Time decay (halflife={time_decay_halflife:.0f}d): "
                           f"oldest_weight={decay_weights[labeled_mask].min():.3f}, "
                           f"newest_weight={decay_weights[labeled_mask].max():.3f}", "INFO")
        except Exception as e:
            tprint(f"  ⚠️ Time decay weighting failed: {e}, skipping", "WARNING")

    # Normalize final weights
    if final_weights.sum() > 0:
        final_weights = final_weights / final_weights.mean()

    return final_weights


def tune_lgbm_hyperparameters_meta(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weights: Optional[np.ndarray],
    horizon: int,
    n_splits: int = 4,
    max_trials: int = 20,
) -> Dict[str, Any]:
    """Time-aware hyperparameter tuning for the meta-model LGBM.

    Uses purged TimeSeriesSplit CV over a small random search grid focusing on
    max_depth, min_child_samples and learning_rate, with early stopping.
    """

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    if not isinstance(y, pd.Series):
        y = pd.Series(y, index=X.index)

    # Use only numeric features and drop NaN labels
    X_num = X.select_dtypes(include=[np.number])
    valid_mask = ~y.isna()
    X_clean = X_num[valid_mask].fillna(0)
    y_clean = y[valid_mask]

    if len(y_clean) < 100 or len(X_clean.columns) == 0 or len(y_clean.unique()) < 2:
        return {}

    sw_clean: Optional[np.ndarray] = None
    if sample_weights is not None:
        try:
            sw = np.asarray(sample_weights, dtype=float)
            if sw.shape[0] != len(X):
                if sw.shape[0] > len(X):
                    sw = sw[-len(X):]
                else:
                    pad = np.ones(len(X) - sw.shape[0], dtype=float)
                    sw = np.concatenate([pad, sw])
            sw_clean = sw[valid_mask.to_numpy()]
        except Exception:
            sw_clean = None

    tscv = TimeSeriesSplit(n_splits=n_splits)
    rng = np.random.RandomState(42)

    best_score = float("-inf")
    best_params: Dict[str, Any] = {}

    for trial in range(int(max_trials)):
        max_depth = int(rng.choice([4, 5, 6, 8]))
        min_child_samples = int(rng.choice([20, 50]))
        learning_rate = float(rng.choice([0.01, 0.03, 0.05]))
        feature_fraction = float(rng.choice([0.6, 0.7, 0.8]))
        bagging_fraction = float(rng.choice([0.7, 0.8, 0.9]))
        reg_alpha = float(rng.choice([0.0, 0.1, 0.3]))
        reg_lambda = float(rng.choice([0.0, 0.2, 0.7, 1.0]))

        num_leaves = int(min(255, 2 ** (max_depth + 1)))

        params: Dict[str, Any] = {
            "max_depth": max_depth,
            "min_child_samples": min_child_samples,
            "learning_rate": learning_rate,
            "feature_fraction": feature_fraction,
            "bagging_fraction": bagging_fraction,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "num_leaves": num_leaves,
            "n_estimators": 1000,
        }

        fold_aucs: List[float] = []

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X_clean)):
            train_idx_purged = purge_training_idxs(
                train_idx,
                test_idx[0],
                test_idx[-1] + 1,
                horizon=horizon,
            )
            if len(train_idx_purged) < 50 or len(test_idx) < 20:
                continue

            X_train_cv = X_clean.iloc[train_idx_purged]
            y_train_cv = y_clean.iloc[train_idx_purged]
            X_val_cv = X_clean.iloc[test_idx]
            y_val_cv = y_clean.iloc[test_idx]

            if sw_clean is not None:
                w_train_cv = sw_clean[train_idx_purged]
            else:
                w_train_cv = None

            model = lgb.LGBMClassifier(
                boosting_type="gbdt",
                objective="binary",
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                num_leaves=params["num_leaves"],
                learning_rate=params["learning_rate"],
                min_child_samples=params["min_child_samples"],
                feature_fraction=params["feature_fraction"],
                bagging_fraction=params["bagging_fraction"],
                bagging_freq=1,
                reg_alpha=params["reg_alpha"],
                reg_lambda=params["reg_lambda"],
                n_jobs=-1,
                verbose=-1,
                random_state=int(rng.randint(0, 1_000_000)),
            )

            fit_kwargs: Dict[str, Any] = {}
            if w_train_cv is not None:
                fit_kwargs["sample_weight"] = w_train_cv

            try:
                model.fit(
                    X_train_cv,
                    y_train_cv,
                    eval_set=[(X_val_cv, y_val_cv)],
                    eval_metric="auc",
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
                    **fit_kwargs,
                )
                y_pred = model.predict_proba(X_val_cv)[:, 1]
                auc = roc_auc_score(y_val_cv, y_pred)
                fold_aucs.append(float(auc))
            except Exception as e:
                tprint(
                    f"[META_LGBM_HPO] Trial {trial + 1}, fold {fold_idx + 1} failed: {str(e)[:120]}...",
                    "WARNING",
                )
                continue

        if not fold_aucs:
            continue

        mean_auc = float(np.mean(fold_aucs))
        std_auc = float(np.std(fold_aucs))
        score = mean_auc - 0.2 * std_auc

        if score > best_score:
            best_score = score
            best_params = params

    if best_params:
        tprint(
            f"[META_LGBM_HPO] Best params: {best_params}, score={best_score:.4f}",
            "INFO",
        )

    return best_params


def train_ensemble_with_kfold(
    X: pd.DataFrame,
    y: pd.Series,
    horizon: int,
    n_splits: int = 5,
    sample_weights: Optional[np.ndarray] = None,
    verbose: bool = True,
    lgbm_params_override: Optional[Dict[str, Any]] = None,
    model_names: Optional[List[str]] = None,
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
    # Hard-align labels (and optional weights) to the feature matrix so that
    # TimeSeriesSplit indices are always valid and purely positional.
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    n_samples = len(X)

    if not isinstance(y, pd.Series):
        y = pd.Series(y, index=X.index)
    else:
        # If indices or lengths differ, reindex on X.index; this is safe even
        # with duplicate datetime labels as long as no fill method is used.
        if len(y) != n_samples or not y.index.equals(X.index):
            try:
                y = y.reindex(X.index)
            except ValueError:
                # Fallback: align positionally on the most recent window.
                y_arr = y.to_numpy()
                if len(y_arr) > n_samples:
                    y_arr = y_arr[-n_samples:]
                elif len(y_arr) < n_samples:
                    pad = np.full(n_samples - len(y_arr), np.nan, dtype=float)
                    y_arr = np.concatenate([pad, y_arr])
                y = pd.Series(y_arr, index=X.index, name=y.name)

    # Align sample_weights length to X if provided (positionally, tail-aligned).
    if sample_weights is not None:
        try:
            sw = np.asarray(sample_weights, dtype=float)
            if sw.shape[0] != n_samples:
                if sw.shape[0] > n_samples:
                    sw = sw[-n_samples:]
                else:
                    pad = np.ones(n_samples - sw.shape[0], dtype=float)
                    sw = np.concatenate([pad, sw])
            sample_weights = sw
        except Exception:
            sample_weights = None

    # Initialize storage
    if model_names is None:
        model_names = ['lgbm', 'xgb', 'rf']

    trained_models = {name: [] for name in model_names}
    oof_predictions = {
        name: pd.Series(np.nan, index=X.index) for name in model_names
    }
    oof_aucs = {name: [] for name in model_names}

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

        # Get train/test splits (positional indices from TimeSeriesSplit)
        X_train = X.iloc[train_idx_purged]
        y_train = y.iloc[train_idx_purged]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        # Filter NaN labels using positional boolean masks to avoid any
        # index-alignment quirks with pandas Series.
        train_mask = ~y_train.isna()
        test_mask = ~y_test.isna()

        if train_mask.sum() < 10 or test_mask.sum() < 5:
            if verbose:
                tprint(f"    ⚠️ Too few samples, skipping fold", "WARNING")
            continue

        train_mask_arr = train_mask.to_numpy(dtype=bool, copy=False)
        test_mask_arr = test_mask.to_numpy(dtype=bool, copy=False)

        X_train_clean = X_train.iloc[train_mask_arr].fillna(0)
        y_train_clean = y_train.iloc[train_mask_arr]
        X_test_clean = X_test.iloc[test_mask_arr].fillna(0)
        y_test_clean = y_test.iloc[test_mask_arr]

        # Extract sample weights for this fold (if provided)
        if sample_weights is not None:
            weights_train_clean = sample_weights[train_idx_purged][train_mask]
        else:
            weights_train_clean = None

        try:
            fold_start = X.index[test_idx[0]]
            fold_end = X.index[test_idx[-1]]

            y_train_arr = y_train_clean.to_numpy()
            y_test_arr = y_test_clean.to_numpy()

            n_train = y_train_arr.shape[0]
            n_test = y_test_arr.shape[0]

            pos_train = int((y_train_arr == 1.0).sum())
            pos_test = int((y_test_arr == 1.0).sum())

            neg_train = n_train - pos_train
            neg_test = n_test - pos_test

            vol_cols = [c for c in X_train_clean.columns if "vol" in c.lower()]
            if vol_cols:
                train_vol_proxy = float(np.nanmedian(X_train_clean[vol_cols].to_numpy()))
                test_vol_proxy = float(np.nanmedian(X_test_clean[vol_cols].to_numpy()))
            else:
                train_vol_proxy = float("nan")
                test_vol_proxy = float("nan")

            if verbose:
                tprint(
                    (
                        f"    Fold {fold_idx + 1}: test_range=[{fold_start} -> {fold_end}], "
                        f"train_n={n_train}, test_n={n_test}, "
                        f"train_pos={pos_train} ({pos_train / n_train:.1%}), "
                        f"test_pos={pos_test} ({pos_test / n_test:.1%}), "
                        f"vol_proxy_train={train_vol_proxy:.4g}, "
                        f"vol_proxy_test={test_vol_proxy:.4g}"
                    ),
                    "INFO",
                )
        except Exception as diag_exc:
            if verbose:
                tprint(
                    f"    Fold {fold_idx + 1}: diagnostics failed: {diag_exc}",
                    "WARNING",
                )

        # Train each base model
        # NOTE: use_focal_loss=False for now (standard objectives work better with predict_proba)
        # Set to True to enable focal loss (focuses on hard examples, good for noise)
        base_models = create_base_models({}, use_focal_loss=False)

        # Optionally override LGBM hyperparameters from meta-model HPO
        if lgbm_params_override is not None and 'lgbm' in base_models:
            try:
                base_models['lgbm'].set_params(**lgbm_params_override)
            except ValueError:
                # If any params are incompatible, ignore override gracefully
                tprint("⚠️ LGBM param override incompatible; using default base LGBM params", "WARNING")

        for model_name in model_names:
            model = base_models[model_name]
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
    for model_name in trained_models.keys():
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


def train_bagged_lgbm_with_kfold(
    X: pd.DataFrame,
    y: pd.Series,
    horizon: int,
    n_splits: int = 5,
    sample_weights: Optional[np.ndarray] = None,
    n_bags: int = 10,
    lgbm_base_params: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Train a 10x bagged LGBM meta-model with time-series CV.

    Returns a DataFrame with two columns:
        - 'lgbm_bag_mean': mean probability across bags
        - 'lgbm_bag_lower': mean - 1 * std, clipped to [0, 1]
    """

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    n_samples = len(X)

    if not isinstance(y, pd.Series):
        y = pd.Series(y, index=X.index)
    else:
        if len(y) != n_samples or not y.index.equals(X.index):
            try:
                y = y.reindex(X.index)
            except ValueError:
                y_arr = y.to_numpy()
                if len(y_arr) > n_samples:
                    y_arr = y_arr[-n_samples:]
                elif len(y_arr) < n_samples:
                    pad = np.full(n_samples - len(y_arr), np.nan, dtype=float)
                    y_arr = np.concatenate([pad, y_arr])
                y = pd.Series(y_arr, index=X.index, name=y.name)

    if sample_weights is not None:
        try:
            sw = np.asarray(sample_weights, dtype=float)
            if sw.shape[0] != n_samples:
                if sw.shape[0] > n_samples:
                    sw = sw[-n_samples:]
                else:
                    pad = np.ones(n_samples - sw.shape[0], dtype=float)
                    sw = np.concatenate([pad, sw])
            sample_weights = sw
        except Exception:
            sample_weights = None

    # Base LGBM parameters from create_base_models (no focal loss)
    base_models = create_base_models({}, use_focal_loss=False)
    base_lgbm = base_models['lgbm']
    base_params = base_lgbm.get_params()

    if lgbm_base_params is not None:
        try:
            base_params.update(lgbm_base_params)
        except Exception:
            pass

    # Force bagging-related parameters
    base_params.setdefault('n_estimators', 1000)
    if 'feature_fraction' not in base_params:
        base_params['feature_fraction'] = 1.0
    base_params['colsample_bytree'] = base_params.get('colsample_bytree', base_params['feature_fraction'])
    if 'subsample' not in base_params:
        base_params['subsample'] = 1.0
    if 'bagging_fraction' not in base_params:
        base_params['bagging_fraction'] = 1.0
    base_params['bagging_freq'] = base_params.get('bagging_freq', 0)

    external_feature_fraction = 0.7
    external_sample_fraction = 0.7
    rng = np.random.RandomState(42)

    oof_mean = pd.Series(np.nan, index=X.index)
    oof_lower = pd.Series(np.nan, index=X.index)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        if verbose:
            tprint(f"  [BAGGED_LGBM] Fold {fold_idx + 1}/{n_splits}...", "INFO")

        train_idx_purged = purge_training_idxs(
            train_idx,
            test_idx[0],
            test_idx[-1] + 1,
            horizon=horizon,
        )

        if len(train_idx_purged) == 0:
            if verbose:
                tprint("    ⚠️ All training samples purged, skipping fold", "WARNING")
            continue

        X_train = X.iloc[train_idx_purged]
        y_train = y.iloc[train_idx_purged]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        train_mask = ~y_train.isna()
        test_mask = ~y_test.isna()

        if train_mask.sum() < 10 or test_mask.sum() < 5:
            if verbose:
                tprint("    ⚠️ Too few samples, skipping fold", "WARNING")
            continue

        train_mask_arr = train_mask.to_numpy(dtype=bool, copy=False)
        test_mask_arr = test_mask.to_numpy(dtype=bool, copy=False)

        X_train_clean = X_train.iloc[train_mask_arr].fillna(0)
        y_train_clean = y_train.iloc[train_mask_arr]
        X_test_clean = X_test.iloc[test_mask_arr].fillna(0)

        if sample_weights is not None:
            weights_train_clean = sample_weights[train_idx_purged][train_mask]
        else:
            weights_train_clean = None

        if len(y_train_clean.unique()) < 2:
            if verbose:
                tprint("    ⚠️ Degenerate labels in fold, skipping", "WARNING")
            continue

        test_indices_with_labels = test_idx[test_mask]

        fold_preds = []
        for bag_idx in range(int(max(1, n_bags))):
            params = dict(base_params)
            params['random_state'] = int(params.get('random_state', 42)) + bag_idx

            model = lgb.LGBMClassifier(**params)

            n_features = X_train_clean.shape[1]
            n_feat_sub = max(1, int(round(external_feature_fraction * n_features)))
            feat_indices = rng.choice(n_features, size=n_feat_sub, replace=False)
            feat_indices.sort()
            cols_sub = X_train_clean.columns[feat_indices]

            X_train_bag = X_train_clean[cols_sub]
            X_test_bag = X_test_clean[cols_sub]

            n_rows = X_train_bag.shape[0]
            n_rows_sub = max(10, int(round(external_sample_fraction * n_rows)))
            n_rows_sub = min(n_rows_sub, n_rows)
            row_indices = rng.choice(n_rows, size=n_rows_sub, replace=False)
            row_indices.sort()

            X_train_bag_sub = X_train_bag.iloc[row_indices]
            y_train_bag_sub = y_train_clean.iloc[row_indices]
            if weights_train_clean is not None:
                weights_bag_sub = weights_train_clean[row_indices]
            else:
                weights_bag_sub = None

            try:
                if weights_bag_sub is not None:
                    model.fit(X_train_bag_sub, y_train_bag_sub, sample_weight=weights_bag_sub)
                else:
                    model.fit(X_train_bag_sub, y_train_bag_sub)
                y_pred_proba = model.predict_proba(X_test_bag)[:, 1]
                fold_preds.append(y_pred_proba)
            except Exception as e:
                if verbose:
                    tprint(f"    ❌ Bag {bag_idx + 1} failed: {e}", "ERROR")
                continue

        if not fold_preds:
            continue

        preds_mat = np.vstack(fold_preds).T  # shape: (n_test_clean, n_bags_effective)
        mu = np.mean(preds_mat, axis=1)
        sigma = np.std(preds_mat, axis=1)

        oof_mean.iloc[test_indices_with_labels] = mu
        lower = np.clip(mu - 1.0 * sigma, 0.0, 1.0)
        oof_lower.iloc[test_indices_with_labels] = lower

    oof_mean = oof_mean.fillna(0.5)
    oof_lower = oof_lower.fillna(0.5)

    return pd.DataFrame(
        {
            'lgbm_bag_mean': oof_mean,
            'lgbm_bag_lower': oof_lower,
        },
        index=X.index,
    )


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

    # Normalize indices to UTC-naive DatetimeIndex where applicable to avoid
    # tz-naive vs tz-aware alignment issues when combining masks and slicing.
    def _normalize_index(obj: Any) -> Any:
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            idx = obj.index
            if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
                try:
                    obj = obj.copy()
                    obj.index = idx.tz_convert("UTC").tz_localize(None)
                except Exception:
                    obj = obj.copy()
                    obj.index = idx.tz_localize(None)
        return obj

    oof_predictions = _normalize_index(oof_predictions)
    y_true = _normalize_index(y_true)
    realized_returns = _normalize_index(realized_returns)
    meta_features = _normalize_index(meta_features)

    def _align_to_common_index(
        preds: pd.DataFrame,
        y: pd.Series,
        rets: pd.Series,
        feats: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
        lengths = [len(preds), len(y), len(rets)]
        if isinstance(feats, pd.DataFrame):
            lengths.append(len(feats))
        min_len = min(lengths) if lengths else 0
        if min_len <= 0:
            return (
                preds.iloc[0:0],
                y.iloc[0:0],
                rets.iloc[0:0],
                feats.iloc[0:0] if isinstance(feats, pd.DataFrame) else feats,
            )
        preds = preds.iloc[-min_len:]
        y = y.iloc[-min_len:]
        rets = rets.iloc[-min_len:]
        if isinstance(feats, pd.DataFrame):
            feats = feats.iloc[-min_len:]
        common_index = pd.RangeIndex(min_len)
        preds.index = common_index
        y.index = common_index
        rets.index = common_index
        if isinstance(feats, pd.DataFrame):
            feats.index = common_index
        return preds, y, rets, feats

    oof_predictions, y_true, realized_returns, meta_features = _align_to_common_index(
        oof_predictions,
        y_true,
        realized_returns,
        meta_features,
    )

    # Valid data mask (purely positional to avoid any index alignment issues)
    y_arr = y_true.to_numpy()
    valid_mask = ~pd.isna(y_arr)
    for col in oof_predictions.columns:
        col_arr = oof_predictions[col].to_numpy()
        valid_mask &= ~pd.isna(col_arr)

    n_valid = int(valid_mask.sum())
    if n_valid < 20:
        tprint("⚠️ Warning: Too few samples for calibration", "WARNING")
        return {}, None

    y_valid = y_true.iloc[valid_mask]
    returns_valid = realized_returns.iloc[valid_mask]

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

    # STAGE 2: Blend models with STACKED GENERALIZATION
    tprint("  📊 Stage 2: Blending models with stacked generalization...", "INFO")

    # Use a meta-learner (LogisticRegression) to learn optimal weights from OOF predictions
    # This is stacked generalization: base model outputs become features for a meta-model
    try:
        # Prepare stacking features (calibrated OOF predictions from each base model)
        stack_features_valid = calibrated_predictions.iloc[valid_mask].fillna(0.5)
        stack_features_all = calibrated_predictions.fillna(0.5)

        if len(stack_features_valid.columns) >= 2 and len(stack_features_valid) > 50:
            # Fit meta-learner (simple logistic regression for interpretability)
            meta_learner = LogisticRegression(
                max_iter=1000,
                solver='lbfgs',
                random_state=42
            )
            meta_learner.fit(stack_features_valid, y_valid)

            # Get meta-learner weights for diagnostics
            meta_weights = dict(zip(stack_features_valid.columns, meta_learner.coef_[0]))
            tprint(f"    ✓ Stacked generalization meta-weights: {meta_weights}", "INFO")

            # Predict probabilities using meta-learner
            ensemble_probs = pd.Series(
                meta_learner.predict_proba(stack_features_all)[:, 1],
                index=calibrated_predictions.index
            )
            tprint("    ✓ Stacked generalization applied", "SUCCESS")
        else:
            # Fallback to simple average if insufficient data
            tprint("    ⚠️ Insufficient data for stacking, using simple average", "WARNING")
            ensemble_probs = calibrated_predictions.mean(axis=1)

    except Exception as e:
        tprint(f"    ⚠️ Stacked generalization failed: {e}, using simple average", "WARNING")
        ensemble_probs = calibrated_predictions.mean(axis=1)

    # STAGE 3: Isotonic regression on ensemble output
    tprint("  📈 Stage 3: Applying isotonic regression to ensemble...", "INFO")

    ensemble_valid = ensemble_probs.iloc[valid_mask].values

    calibration_input = ensemble_valid

    # Fit isotonic regression: calibrated_prob -> expected_return
    try:
        iso_regressor = IsotonicRegression(out_of_bounds='clip')
        iso_regressor.fit(calibration_input, returns_valid.values)

        tprint(f"    ✓ Isotonic regression fitted on {len(calibration_input)} samples", "SUCCESS")
        tprint("    ℹ️ iso_regressor.fit completed", "INFO")

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

def compute_regime_wise_metrics(
    X: pd.DataFrame,
    y: pd.Series,
    y_pred: Optional[np.ndarray] = None,
    realized_returns: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """Compute performance metrics broken down by volatility regimes.

    Implementation is fully positional/array-based to avoid any pandas boolean
    index alignment issues. All inputs are tail-aligned to a common window
    before applying masks.

    Args:
        X: Feature matrix (should include 'volatility_regime' column)
        y: Binary labels
        y_pred: Optional predicted probabilities (same length as y)
        realized_returns: Optional realized returns (same length as y)

    Returns:
        Dictionary with regime-wise metrics
    """
    metrics: Dict[str, Any] = {}

    # Basic sanity checks
    if not isinstance(X, pd.DataFrame) or len(X) == 0 or len(y) == 0:
        return {"error": "Insufficient samples"}

    lengths = [len(X), len(y)]
    if y_pred is not None:
        lengths.append(len(y_pred))
    if realized_returns is not None:
        lengths.append(len(realized_returns))

    n_common = min(lengths) if lengths else 0
    if n_common < 10:
        return {"error": "Insufficient valid samples"}

    # Tail-align all inputs on a common window
    X_tail = X.iloc[-n_common:]
    y_tail = y.iloc[-n_common:]
    rr_tail = realized_returns.iloc[-n_common:] if realized_returns is not None else None
    y_pred_tail = np.asarray(y_pred, dtype=float)[-n_common:] if y_pred is not None else None

    # Volatility regime must be present to compute metrics
    if 'volatility_regime' not in X_tail.columns:
        return metrics

    vol_regime = X_tail['volatility_regime']

    # Convert labels and optional returns to numpy for robust masking
    y_arr = y_tail.to_numpy(dtype=float, copy=False)
    valid_mask_arr = ~np.isnan(y_arr)
    if valid_mask_arr.sum() < 10:
        return {"error": "Insufficient valid samples"}

    vol_arr = vol_regime.to_numpy()
    y_valid = y_arr[valid_mask_arr]
    vol_valid = vol_arr[valid_mask_arr]

    if y_pred_tail is not None:
        y_pred_valid = y_pred_tail[valid_mask_arr]
    else:
        y_pred_valid = None

    if rr_tail is not None:
        rr_arr = rr_tail.to_numpy(dtype=float, copy=False)
        rr_valid = rr_arr[valid_mask_arr]
    else:
        rr_valid = None

    vol_metrics: Dict[str, Any] = {}
    for regime in ['low', 'medium', 'high']:
        regime_mask = vol_valid == regime
        n_regime = int(regime_mask.sum())
        if n_regime < 10:
            continue

        y_regime = y_valid[regime_mask]
        pos_rate = float((y_regime == 1.0).mean())
        reg_dict: Dict[str, Any] = {
            'n_samples': n_regime,
            'pos_rate': pos_rate,
        }

        # AUC by regime if predictions available
        if y_pred_valid is not None and len(y_pred_valid) == len(y_valid):
            y_pred_regime = y_pred_valid[regime_mask]
            if np.unique(y_regime).size > 1:
                try:
                    auc = roc_auc_score(y_regime, y_pred_regime)
                    reg_dict['auc'] = float(auc)
                except Exception:
                    pass

        # Return-based metrics by regime if available
        if rr_valid is not None and len(rr_valid) == len(y_valid):
            ret_regime = rr_valid[regime_mask]
            if ret_regime.size > 0:
                mean_ret = float(ret_regime.mean())
                std_ret = float(ret_regime.std())
                reg_dict['mean_return'] = mean_ret
                reg_dict['sharpe'] = float(mean_ret / (std_ret + 1e-8))

        vol_metrics[regime] = reg_dict

    if vol_metrics:
        metrics['volatility_regimes'] = vol_metrics

    return metrics


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
    attach_probs = bool(config.get("attach_hmm_probabilities", True))

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
        if attach_probs:
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
                ts = pd.to_datetime(labels_df["timestamp"], utc=True, errors="coerce")
                try:
                    ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
                except Exception:
                    ts = ts.dt.tz_localize(None)
                labels_df["timestamp"] = ts
                labels_df.set_index("timestamp", inplace=True)
            if isinstance(labels_df.index, pd.DatetimeIndex):
                if labels_df.index.tz is not None:
                    try:
                        labels_df.index = labels_df.index.tz_convert("UTC").tz_localize(None)
                    except Exception:
                        labels_df.index = labels_df.index.tz_localize(None)
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
                ts = pd.to_datetime(probs_df["timestamp"], utc=True, errors="coerce")
                try:
                    ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
                except Exception:
                    ts = ts.dt.tz_localize(None)
                probs_df["timestamp"] = ts
                probs_df.set_index("timestamp", inplace=True)
            if isinstance(probs_df.index, pd.DatetimeIndex):
                if probs_df.index.tz is not None:
                    try:
                        probs_df.index = probs_df.index.tz_convert("UTC").tz_localize(None)
                    except Exception:
                        probs_df.index = probs_df.index.tz_localize(None)
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

    # Attach HMM Alpha expectation (1h) from hmm_ml_alpha_step if available.
    # This uses the regime_alpha namespace and the hmm_alpha_training_data_1h
    # artifact produced by HMMMLAlphaStep and aligns it to the base timeframe.
    try:
        # Use shared specialist loader (HMM Alpha + ML Risk) and then map
        # columns back to the schema expected by the meta-labeling pipeline.
        specialist_config = {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": base_timeframe,
            "regime_timeframe": regime_timeframe,
            "direction": direction,
            "enable_risk_hmm_specialist": False,
            # Use canonical per-specialist scalars so downstream consumers
            # (including meta-labeling) see a compact, well-defined set of
            # specialist features instead of raw multi-column blocks.
            "use_canonical_specialist_scalars": True,
        }

        specialist_df = get_specialist_models_outputs(
            artifact_router=step.artifact_router,
            training_index=md.index,
            config=specialist_config,
            logger=getattr(step, "logger", None),
            strict=False,
        )

        if specialist_df is not None and not specialist_df.empty:
            # HMM Alpha expectations (1h), renamed to the existing 1h-aware
            # column names on md. The shared specialist loader already
            # aligns indices to md.index; avoid an additional reindex step
            # here to prevent duplicate-index reindex errors.
            try:
                alpha_cols = []
                rename_map = {}

                # Prefer the unified calibrated score if available
                if "alpha_score_continuous" in specialist_df.columns:
                    alpha_cols = [
                        c
                        for c in specialist_df.columns
                        if c == "alpha_score_continuous" or c.startswith("alpha_score_continuous_ewm_")
                    ]
                    at_aligned = specialist_df[alpha_cols].copy()
                    rename_map = {}
                    for c in alpha_cols:
                        if c == "alpha_score_continuous":
                            rename_map[c] = "hmm_alpha_score_continuous_1h"
                        elif c.startswith("alpha_score_continuous_ewm_"):
                            rename_map[c] = f"hmm_{c}_1h"
                else:
                    # Backward compatibility: fall back to expectation columns
                    alpha_cols = [
                        c
                        for c in specialist_df.columns
                        if c.startswith("alpha_expectation_")
                    ]
                    if alpha_cols:
                        at_aligned = specialist_df[alpha_cols].copy()
                        rename_map = {}
                        for c in alpha_cols:
                            if c == "alpha_expectation_raw_01":
                                rename_map[c] = "hmm_alpha_expectation_raw_1h"
                            elif c == "alpha_expectation_ema_01":
                                rename_map[c] = "hmm_alpha_expectation_ema_1h"
                            else:
                                rename_map[c] = f"hmm_alpha_{c}"

                if alpha_cols:
                    # Safety: if indices somehow differ, fall back to a
                    # one-time ffill reindex without allowing duplicates to
                    # raise. This should be rare since the specialist loader
                    # already reindexes to md.index.
                    if not at_aligned.index.equals(md.index):
                        try:
                            at_aligned = at_aligned.reindex(md.index, method="ffill")
                        except ValueError:
                            # As a last resort, align positionally on the
                            # tail window without changing md.index.
                            at_arr = at_aligned.reset_index(drop=True)
                            n_md = len(md)
                            if len(at_arr) > n_md:
                                at_arr = at_arr.iloc[-n_md:, :].reset_index(drop=True)
                            elif len(at_arr) < n_md:
                                pad_rows = n_md - len(at_arr)
                                pad = pd.DataFrame(np.nan, index=range(pad_rows), columns=at_arr.columns)
                                at_arr = pd.concat([pad, at_arr], axis=0, ignore_index=True)
                            at_arr.index = md.index
                            at_aligned = at_arr

                    at_aligned = at_aligned.rename(columns=rename_map)
                    for c in at_aligned.columns:
                        md[c] = at_aligned[c]
            except Exception as e_alpha:
                tprint(
                    f"⚠️ Failed to attach HMM Alpha expectation features from specialist outputs: {e_alpha}",
                    "WARNING",
                )

            # ML Risk regimes / scores (1h expectations aligned to base
            # timeframe). Again, rely on the shared specialist loader for
            # index alignment and avoid redundant reindexing.
            try:
                risk_cols = [
                    c
                    for c in specialist_df.columns
                    if c.startswith("risk_regime") or c.startswith("risk_score")
                ]
                if risk_cols:
                    rt_aligned = specialist_df[risk_cols].copy()

                    if not rt_aligned.index.equals(md.index):
                        try:
                            rt_aligned = rt_aligned.reindex(md.index, method="ffill")
                        except ValueError:
                            rt_arr = rt_aligned.reset_index(drop=True)
                            n_md = len(md)
                            if len(rt_arr) > n_md:
                                rt_arr = rt_arr.iloc[-n_md:, :].reset_index(drop=True)
                            elif len(rt_arr) < n_md:
                                pad_rows = n_md - len(rt_arr)
                                pad = pd.DataFrame(np.nan, index=range(pad_rows), columns=rt_arr.columns)
                                rt_arr = pd.concat([pad, rt_arr], axis=0, ignore_index=True)
                            rt_arr.index = md.index
                            rt_aligned = rt_arr

                    for col in risk_cols:
                        md[col] = rt_aligned[col]
            except Exception as e_risk:
                tprint(
                    f"⚠️ Failed to attach ML Risk regime features from specialist outputs: {e_risk}",
                    "WARNING",
                )
    except Exception as e_spec:
        tprint(
            f"⚠️ Failed to load specialist model outputs for HMM Alpha / ML Risk attachment: {e_spec}",
            "WARNING",
        )

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
            min_event_spacing = config.get('min_event_spacing', 2)

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

            meta_model_family = str(config.get('meta_model_family', 'lgbm_bag_lower')).lower()
            enable_meta_lgbm_hpo = bool(config.get('enable_meta_lgbm_hpo', True))
            meta_lgbm_n_bags = int(config.get('meta_lgbm_n_bags', 10))
            meta_prob_source = 'ensemble'
            best_lgbm_params: Dict[str, Any] = {}
            bagged_mean_series: Optional[pd.Series] = None
            bagged_lower_series: Optional[pd.Series] = None

            used_hpo_params = False

            # Optionally override labeling parameters using latest HPO results
            # enable_labeling_hpo_params: if True (default), try to load best params JSON
            if config.get('enable_labeling_hpo_params', True):
                try:
                    symbol = str(config['symbol'])
                    timeframe = str(config['timeframe'])
                    hpo_params, latest_path, params_source = _load_latest_labeling_hpo_params(
                        symbol,
                        timeframe,
                    )
                    if latest_path is not None and hpo_params:
                        label_source = params_source or 'params'
                        tprint(f"🔍 Using labeling HPO {label_source} from {latest_path}", "INFO")

                        # Map HPO params → step parameters with safety clamps
                        if 'profit_thr_base' in hpo_params:
                            profit_threshold = float(hpo_params['profit_thr_base'])
                        if 'stop_to_profit_ratio' in hpo_params:
                            stop_ratio = float(hpo_params['stop_to_profit_ratio'])
                            stop_threshold = max(0.0005, profit_threshold * stop_ratio)
                        if 'horizon_bars' in hpo_params:
                            horizon = int(hpo_params['horizon_bars'])
                        if 'min_event_spacing' in hpo_params:
                            min_event_spacing = int(hpo_params['min_event_spacing'])

                        if 'kalman_Q' in hpo_params:
                            kalman_Q = float(hpo_params['kalman_Q'])
                        if 'kalman_R' in hpo_params:
                            kalman_R = float(hpo_params['kalman_R'])
                        if 'vol_baseline_window' in hpo_params:
                            vol_baseline_window = int(hpo_params['vol_baseline_window'])
                        if 'profit_mult_min' in hpo_params:
                            profit_mult_min = float(hpo_params['profit_mult_min'])
                        if 'profit_mult_max' in hpo_params:
                            profit_mult_max = float(hpo_params['profit_mult_max'])
                        if 'stop_mult_min' in hpo_params:
                            stop_mult_min = float(hpo_params['stop_mult_min'])
                        if 'stop_mult_max' in hpo_params:
                            stop_mult_max = float(hpo_params['stop_mult_max'])
                        if 'iso_min_prob' in hpo_params:
                            iso_min_prob_param = float(hpo_params['iso_min_prob'])
                        if 'target_clip_high_q' in hpo_params:
                            target_clip_high_q_param = float(hpo_params['target_clip_high_q'])

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

            # Load market data via BaseStep helpers so execution mode and lookback days are centralized
            tprint("📊 [prep] Loading market data via BaseStep...", "INFO")
            pipeline_state: Dict[str, Any] = {}
            market_data, source = self.load_market_data_or_fail(
                config,
                pipeline_state=pipeline_state,
                allow_config_override=True,
            )

            if not isinstance(market_data, pd.DataFrame) or market_data.empty:
                raise ValueError(f"No market data available for {config['symbol']} {config['timeframe']}")

            # Index diagnostics and normalization (timezone-safe)
            idx = market_data.index
            if isinstance(idx, pd.DatetimeIndex):
                tprint(
                    f"🕒 [index] market_data.index tz={idx.tz}, name={idx.name}, len={len(idx)}",
                    "INFO",
                )
                if idx.tz is not None:
                    try:
                        market_data = market_data.copy()
                        market_data.index = market_data.index.tz_convert("UTC").tz_localize(None)
                        tprint("🕒 [index] Normalized market_data.index to UTC-naive", "INFO")
                    except Exception as tz_exc:
                        tprint(f"⚠️ [index] Failed to normalize market_data.index timezone: {tz_exc}", "WARNING")
            else:
                tprint(
                    f"🕒 [index] market_data.index type={type(idx)}, len={len(idx)}",
                    "INFO",
                )

            tprint(f"📊 Loaded {len(market_data)} samples from {source}", "SUCCESS")

            if 'close' not in market_data.columns:
                raise ValueError("Missing required 'close' column in market data")

            # STEP 1: Generate FIXED primary signals
            tprint("🎯 [1/13] Generating fixed primary signals...", "INFO")
            primary_signals = generate_primary_signals(market_data)

            n_long_signals = int((primary_signals['consensus'] > 0).sum())
            n_short_signals = int((primary_signals['consensus'] < 0).sum())
            tprint(f"📊 Primary signals: {n_long_signals} long, {n_short_signals} short", "INFO")

            # Surface signal funnel statistics from the signal generator if available
            signal_funnel = {}
            try:
                signal_funnel = primary_signals.attrs.get('signal_funnel', {}) or {}
            except Exception:
                signal_funnel = {}

            if signal_funnel:
                total_bars_sf = int(signal_funnel.get('total_bars', len(primary_signals)))
                raw_sf = int(signal_funnel.get('raw_signals', n_long_signals + n_short_signals))
                final_sf = int(signal_funnel.get('final_signals', n_long_signals + n_short_signals))
                ratio_sf = float(signal_funnel.get('raw_to_final_ratio', final_sf / max(raw_sf, 1)))

                tprint("📊 Signal funnel summary (from generator):", "INFO")
                tprint(f"  Bars: {total_bars_sf}, Raw signals: {raw_sf}, Final consensus: {final_sf} (ratio={ratio_sf:.3f})", "INFO")

            # Define canonical training index based on primary signals
            train_index = primary_signals.index

            # Align market_data to train_index (positionally via reindex) to avoid
            # length and duplicate-index issues downstream.
            if not market_data.index.equals(train_index):
                try:
                    tprint(
                        f"⚠️ [train_index] Aligning market_data (len={len(market_data)}) "
                        f"to primary_signals index (len={len(train_index)})",
                        "WARNING",
                    )
                    market_data = market_data.reindex(train_index, method="ffill")
                except Exception as align_exc:
                    tprint(
                        f"⚠️ [train_index] Failed to align market_data to train_index: {align_exc}",
                        "WARNING",
                    )

            # Defensive: ensure primary_signals also uses train_index exactly
            if not primary_signals.index.equals(train_index):
                try:
                    primary_signals = primary_signals.reindex(train_index, method="ffill")
                except Exception as sig_align_exc:
                    tprint(
                        f"⚠️ [train_index] Failed to align primary_signals to train_index: {sig_align_exc}",
                        "WARNING",
                    )

            # Volume flag after potential realignment
            volume_available = 'volume' in market_data.columns

            # Attach rolling HMM regimes and specialist features aligned to train_index
            try:
                market_data = attach_rolling_hmm_regimes_to_market_data(
                    self,
                    market_data,
                    config,
                )
            except Exception as e_reg:
                tprint(f"⚠️ Failed to attach rolling HMM regimes to market_data: {e_reg}", "WARNING")

            # Attach specialist model outputs (liquidity regimes, canonical scalars, etc.)
            # aligned to train_index
            try:
                specialist_config = dict(config)
                specialist_config.setdefault("enable_risk_hmm_specialist", False)
                specialist_config.setdefault("use_canonical_specialist_scalars", True)
                specialist_df = get_specialist_models_outputs(
                    artifact_router=self.artifact_router,
                    training_index=train_index,
                    config=specialist_config,
                    logger=self.logger,
                    strict=False,
                )
                if specialist_df is not None and not specialist_df.empty:
                    # Liquidity regime probabilities
                    prob_cols = [
                        c for c in specialist_df.columns
                        if c.startswith('liquidity_regime_') and 'prob_' in c
                    ]
                    if prob_cols:
                        liquidity_features = specialist_df[prob_cols].reindex(train_index, method='ffill')
                        for col in liquidity_features.columns:
                            market_data[f'liquidity_{col}'] = liquidity_features[col]
                        tprint(
                            f"✅ Added {len(prob_cols)} liquidity regime probability features to market_data via specialist loader",
                            "SUCCESS",
                        )

                    # Canonical specialist scalar signals (risk, path, macro trend,
                    # mean-reversion, SR labeling, volume force, SMC, etc.). These
                    # are aligned to train_index inside get_specialist_models_outputs,
                    # so we can attach them directly and let create_meta_features
                    # pick them up as meta-features.
                    try:
                        scalar_cols: List[str] = []

                        for col in [
                            "risk_score",
                            "path_risk_score",
                            "macro_trend_score_continuous",
                            "mr_probability_dense",
                            "mr_probability",
                            "mr_raw_score",
                            "mr_trend_state",
                            "mr_trend_is_mr",
                            "sr_labeling_xgb_prob",
                            "vol_force_scalar",
                            "smc_predicted",
                        ]:
                            if col in specialist_df.columns:
                                scalar_cols.append(col)

                        # Include any remaining MR / SMC-prefixed scalars without
                        # hard-coding every variant name.
                        scalar_cols.extend(
                            [
                                c
                                for c in specialist_df.columns
                                if c.startswith("mr_") or c.startswith("smc_")
                            ]
                        )

                        seen_scalars: set[str] = set()
                        scalar_cols_unique: List[str] = []
                        for c in scalar_cols:
                            if c not in seen_scalars:
                                seen_scalars.add(c)
                                scalar_cols_unique.append(c)

                        for col in scalar_cols_unique:
                            if col not in market_data.columns:
                                market_data[col] = specialist_df[col]
                    except Exception as e_spec_scalars:
                        tprint(
                            f"⚠️ Failed to attach canonical specialist scalars to market_data: {e_spec_scalars}",
                            "WARNING",
                        )
            except Exception as e_liquidity:
                tprint(f"⚠️ Failed to attach specialist liquidity regime probabilities: {e_liquidity}", "WARNING")

            # STEP 2: Compute volatility for adaptive thresholds
            tprint("📊 [2/13] Computing volatility for adaptive thresholds...", "INFO")
            log_ret = np.log(market_data['close']).diff()
            volatility_1d = log_ret.rolling(96).std()  # Short volatility estimate

            # Baseline volatility over configurable window (HPO-aware)
            vol_baseline = volatility_1d.rolling(vol_baseline_window).mean()
            vol_factor = volatility_1d / (vol_baseline + 1e-8)

            # === Trend-aware ATR modulation for triple-barrier distances ===
            high_prices = market_data['high'] if 'high' in market_data.columns else market_data['close']
            low_prices = market_data['low'] if 'low' in market_data.columns else market_data['close']
            close_prices = market_data['close']

            tr1 = high_prices - low_prices
            tr2 = (high_prices - close_prices.shift(1)).abs()
            tr3 = (low_prices - close_prices.shift(1)).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            trend_atr_window = int(config.get("trend_strength_atr_window", 14))
            atr_series = true_range.rolling(window=trend_atr_window, min_periods=1).mean()

            trend_delta_lookback = int(config.get("trend_strength_delta_lookback", 4))
            price_delta = close_prices.diff(trend_delta_lookback).abs()

            trend_strength = (price_delta / (atr_series + 1e-8)).replace([np.inf, -np.inf], np.nan)
            trend_strength = trend_strength.clip(
                lower=0.0,
                upper=float(config.get("trend_strength_clip", 5.0)),
            ).fillna(0.0)

            trend_alpha = float(config.get("trend_strength_alpha_profit", 0.5))
            trend_beta = float(config.get("trend_strength_beta_stop", 0.5))

            profit_factor = 1.0 + trend_alpha * trend_strength
            stop_factor = 1.0 + trend_beta * trend_strength

            # Adaptive thresholds based on volatility and trend-aware multipliers
            adaptive_profit_threshold = profit_threshold * vol_factor * profit_factor
            adaptive_stop_threshold = stop_threshold * vol_factor * stop_factor

            # Enforce hard floor based on transaction costs (0.5% = 50 bps)
            # This ensures profit targets remain viable after slippage + fees
            profit_floor = PROFIT_TARGET_FLOOR_BPS / 10000.0  # Convert basis points to decimal (0.005)

            adaptive_profit_threshold = adaptive_profit_threshold.clip(
                lower=max(profit_threshold * profit_mult_min, profit_floor),
                upper=profit_threshold * profit_mult_max,
            )
            adaptive_stop_threshold = adaptive_stop_threshold.clip(
                lower=stop_threshold * stop_mult_min,
                upper=stop_threshold * stop_mult_max,
            )

            # Log if any targets were floored
            n_floored = (adaptive_profit_threshold <= profit_floor * 1.001).sum()
            if n_floored > 0:
                tprint(f"  ⚠️ Enforced profit floor (0.5%) on {n_floored}/{len(adaptive_profit_threshold)} bars", "WARNING")

            tprint(f"📊 Adaptive thresholds: Profit {adaptive_profit_threshold.mean():.2%} ± {adaptive_profit_threshold.std():.2%} (floor: {profit_floor:.2%})", "INFO")
            tprint(f"📊 Adaptive thresholds: Stop {adaptive_stop_threshold.mean():.2%} ± {adaptive_stop_threshold.std():.2%}", "INFO")

            # STEP 3: Compute realized returns (continuous) and binary labels with adaptive thresholds
            tprint("💰 [3/13] Computing realized returns with adaptive thresholds and transaction costs...", "INFO")

            # ATR series for trailing stops (aligned with HPO behaviour). We use a
            # True Range based ATR so trailing distance is comparable across steps.
            try:
                high_prices = market_data["high"] if "high" in market_data.columns else market_data["close"]
                low_prices = market_data["low"] if "low" in market_data.columns else market_data["close"]
                close_prices = market_data["close"]

                tr1 = high_prices - low_prices
                tr2 = (high_prices - close_prices.shift(1)).abs()
                tr3 = (low_prices - close_prices.shift(1)).abs()
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

                atr_lookback = int(config.get("trail_atr_window", 14))
                atr_lookback = max(2, atr_lookback)
                atr_series = true_range.rolling(window=atr_lookback, min_periods=1).mean()
            except Exception:
                atr_series = None

            # Trailing distance in ATR multiples. Prefer explicit trail_distance_atr,
            # but fall back to the generic trail_distance used by HPO if present.
            trail_dist = float(config.get("trail_distance_atr", config.get("trail_distance", 0.0)))
            if not np.isfinite(trail_dist):
                trail_dist = 0.0

            # For type safety: only pass a Series (or None) into compute_realized_returns.
            if isinstance(atr_series, pd.Series):
                atr_series_trailing = atr_series
            else:
                atr_series_trailing = None

            (
                realized_returns, 
                binary_labels, 
                exit_reasons, 
                event_durations, 
                mfe_series, 
                mae_series,
                binary_labels_long,
                binary_labels_short
            ) = compute_realized_returns(
                market_data,
                primary_signals,
                profit_threshold=adaptive_profit_threshold,
                stop_threshold=adaptive_stop_threshold,
                horizon=horizon,
                transaction_cost=transaction_cost,
                min_event_spacing=min_event_spacing,
                volatility_series=volatility_1d,  # Enable dynamic horizon based on volatility
                atr_series=atr_series_trailing,
                trail_distance_atr_mult=trail_dist,
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
            
            # Labeling method selection:
            # - "rolling": Rolling quantiles with fixed thresholds (default, no look-ahead bias)
            # - "conditional": Conditional quantile regression Q_τ(z|X) - context-adaptive
            # - "zscore_gated": Full pipeline - Conditional quantile filter THEN triple barrier
            #                   [ Features ] → Quantile Filter → Entry Candidate → Triple Barrier → Labels
            # - "global": Global quantiles (legacy, has look-ahead bias)
            labeling_method = str(config.get("labeling_method", "rolling")).lower()
            
            # Rolling quantile parameters
            use_rolling_quantiles = labeling_method == "rolling" or bool(config.get("use_rolling_quantiles", True))
            rolling_lookback_bars = int(config.get("rolling_quantile_lookback_bars", 3000))  # ~31 days at 15m
            rolling_min_periods = int(config.get("rolling_quantile_min_periods", 300))  # ~3 days at 15m
            
            # Conditional quantile regression parameters (for "conditional" and "zscore_gated")
            use_conditional_quantiles = labeling_method == "conditional"
            use_zscore_gated_pipeline = labeling_method == "zscore_gated"
            # Asymmetric quantiles for crypto (longs more selective, shorts less selective)
            conditional_quantile_long = float(config.get("conditional_quantile_long", 0.6))  # Q_0.6(z|X) for longs
            conditional_quantile_short = float(config.get("conditional_quantile_short", 0.35))  # Q_0.35(z|X) for shorts
            conditional_asymmetric = bool(config.get("conditional_asymmetric_crypto", True))  # Use crypto-optimized asymmetry
            conditional_retrain_freq = int(config.get("conditional_retrain_frequency", 500))
            conditional_min_train = int(config.get("conditional_min_train_samples", 500))
            
            # Z-score gated pipeline parameters (for "zscore_gated" method)
            # Barrier multipliers (× volatility)
            zg_k_tp_base = float(config.get("zscore_gated_k_tp_base", 1.5))
            zg_k_sl_base = float(config.get("zscore_gated_k_sl_base", 1.0))
            # Asymmetric multipliers for longs vs shorts (crypto upward bias)
            zg_k_tp_long_mult = float(config.get("zscore_gated_k_tp_long_mult", 1.1))
            zg_k_sl_long_mult = float(config.get("zscore_gated_k_sl_long_mult", 0.9))
            zg_k_tp_short_mult = float(config.get("zscore_gated_k_tp_short_mult", 1.0))
            zg_k_sl_short_mult = float(config.get("zscore_gated_k_sl_short_mult", 1.0))
            # Z-score magnitude scaling: k_TP = k0 × (1 + scale × |z|)
            zg_z_magnitude_scale = float(config.get("zscore_gated_z_magnitude_scale", 0.3))
            # Trend adjustment (linear): TP_adj = TP × (1 + α × trend_strength)
            zg_trend_alpha = float(config.get("zscore_gated_trend_alpha", 0.3))
            zg_trend_lookback = int(config.get("zscore_gated_trend_lookback", 20))
            # Clipping bounds (as multiples of base volatility)
            zg_tp_min_mult = float(config.get("zscore_gated_tp_min_mult", 0.5))
            zg_tp_max_mult = float(config.get("zscore_gated_tp_max_mult", 4.0))
            zg_sl_min_mult = float(config.get("zscore_gated_sl_min_mult", 0.3))
            zg_sl_max_mult = float(config.get("zscore_gated_sl_max_mult", 2.0))
            
            # Run look-ahead bias diagnostic if enabled
            if config.get("diagnose_quantile_bias", False):
                tprint("🔍 Running quantile look-ahead bias diagnostic...", "INFO")
                bias_diag = diagnose_quantile_lookahead_bias(
                    vol_scaled=vol_scaled_returns,
                    low_q=quantile_low_q,
                    high_q=quantile_high_q,
                    print_results=True,
                )
                if bias_diag.get("bias_detected", False) and labeling_method == "global":
                    tprint(
                        f"⚠️ Look-ahead bias detected ({bias_diag.get('bias_severity', 'unknown')}) "
                        "with global quantiles. Consider using labeling_method='rolling', 'conditional', or 'zscore_gated'",
                        "WARNING"
                    )

            regimes_for_labeling = None
            if config.get("enable_regime_aware_quantiles", True) and "hmm_regime_label_1h" in market_data.columns:
                regimes_for_labeling = market_data["hmm_regime_label_1h"]

            # Store conditional quantile diagnostics if used
            conditional_quantile_diagnostics: Dict[str, Any] = {}
            zscore_gated_diagnostics: Dict[str, Any] = {}

            if use_zscore_gated_pipeline:
                # =====================================================================
                # FULL PIPELINE: Z-Score Gated Triple Barrier
                # =====================================================================
                # [ Market Features ]
                #         ↓
                # Conditional Quantile Filter
                #         ↓   (z > Q_long OR z < Q_short)
                #    Trade Entry Candidate
                #         ↓
                # Triple Barrier Labeling (volatility-aware, trend-adjusted)
                #         ↓
                #    Final Supervised Target
                # =====================================================================
                tprint(
                    f"📊 Using Z-SCORE GATED TRIPLE BARRIER PIPELINE:\n"
                    f"   - Quantile filter: Q_long={conditional_quantile_long}, Q_short={conditional_quantile_short}\n"
                    f"   - Barrier scaling: k_TP={zg_k_tp_base}, k_SL={zg_k_sl_base}\n"
                    f"   - Z-score magnitude scale: {zg_z_magnitude_scale}\n"
                    f"   - Trend alpha: {zg_trend_alpha}",
                    "INFO"
                )
                
                # Build meta-features for conditional quantile model
                meta_features_for_cond = create_meta_features(
                    df=market_data,
                    signals=primary_signals,
                    volume_available=volume_available,
                    include_raw_signals=False,
                    use_kalman=True,
                )
                
                # Stable features for conditioning
                stable_features = [
                    'volatility_1h', 'volatility_4h', 'volatility_1d', 'volatility_ema',
                    'vol_of_vol', 'momentum_20', 'momentum_ema', 'rsi_kalman',
                    'ma_distance_kalman', 'kalman_trend', 'range_position',
                    'hour_sin', 'hour_cos', 'day_of_week',
                ]
                feature_subset = [f for f in stable_features if f in meta_features_for_cond.columns]
                
                # Prepare ATR series for trailing stops
                atr_for_trailing = None
                trail_mult_for_pipeline = None
                if 'atr_14' in market_data.columns:
                    atr_for_trailing = market_data['atr_14']
                    # Use a reasonable trailing distance (1.5 ATR by default)
                    trail_mult_for_pipeline = float(config.get("zscore_gated_trail_atr_mult", 1.5))
                
                # Run the full z-score gated pipeline
                gated_labeled_data, zscore_gated_diagnostics = compute_zscore_gated_triple_barrier_labels(
                    df=market_data,
                    features=meta_features_for_cond,
                    signals=primary_signals,
                    volatility=volatility_1d,
                    # Conditional quantile parameters
                    quantile_long=conditional_quantile_long,
                    quantile_short=conditional_quantile_short,
                    asymmetric_crypto=conditional_asymmetric,
                    quantile_lookback=rolling_lookback_bars,
                    quantile_min_train=conditional_min_train,
                    quantile_retrain_freq=conditional_retrain_freq,
                    # Barrier parameters
                    k_tp_base=zg_k_tp_base,
                    k_sl_base=zg_k_sl_base,
                    k_tp_long_mult=zg_k_tp_long_mult,
                    k_sl_long_mult=zg_k_sl_long_mult,
                    k_tp_short_mult=zg_k_tp_short_mult,
                    k_sl_short_mult=zg_k_sl_short_mult,
                    # Z-score scaling
                    z_magnitude_scale=zg_z_magnitude_scale,
                    # Trend adjustment
                    trend_alpha=zg_trend_alpha,
                    trend_lookback=zg_trend_lookback,
                    # Clipping
                    tp_min_mult=zg_tp_min_mult,
                    tp_max_mult=zg_tp_max_mult,
                    sl_min_mult=zg_sl_min_mult,
                    sl_max_mult=zg_sl_max_mult,
                    # Other
                    horizon=horizon,
                    transaction_cost=transaction_cost,
                    min_event_spacing=min_event_spacing,
                    atr_series=atr_for_trailing,
                    trail_distance_atr_mult=trail_mult_for_pipeline,
                    feature_subset=feature_subset if feature_subset else None,
                )
                
                # Extract labels from the gated pipeline output
                # The pipeline returns a DataFrame with all labeling information
                quantile_labels = gated_labeled_data.get('binary_label', pd.Series(index=market_data.index, dtype=float))
                quantile_labels_short = gated_labeled_data.get('binary_label_short', pd.Series(index=market_data.index, dtype=float))
                
                # Override realized_returns with the gated pipeline results
                realized_returns = gated_labeled_data.get('realized_return', realized_returns)
                
                # Use gated consensus instead of original
                gated_consensus = gated_labeled_data.get('gated_consensus', pd.Series(0, index=market_data.index))
                primary_signals['consensus'] = gated_consensus
                
                # Log diagnostics
                n_trades = zscore_gated_diagnostics.get('n_trades', 0)
                win_rate = zscore_gated_diagnostics.get('win_rate', 0.0)
                mean_ret = zscore_gated_diagnostics.get('mean_return', 0.0)
                n_gated_long = zscore_gated_diagnostics.get('n_gated_long', 0)
                n_gated_short = zscore_gated_diagnostics.get('n_gated_short', 0)
                
                tprint(
                    f"📊 Z-Score Gated Pipeline Results:\n"
                    f"   - Gated entries: {n_gated_long} longs, {n_gated_short} shorts\n"
                    f"   - Trades executed: {n_trades}, Win rate: {win_rate:.1%}, Mean return: {mean_ret:.4f}",
                    "INFO"
                )
                
                # Store additional columns from the gated pipeline
                if 'tp_threshold' in gated_labeled_data.columns:
                    labeled_data['tp_threshold'] = gated_labeled_data['tp_threshold']
                if 'sl_threshold' in gated_labeled_data.columns:
                    labeled_data['sl_threshold'] = gated_labeled_data['sl_threshold']
                if 'trend_strength' in gated_labeled_data.columns:
                    labeled_data['trend_strength'] = gated_labeled_data['trend_strength']
                if 'z_score' in gated_labeled_data.columns:
                    labeled_data['z_score'] = gated_labeled_data['z_score']
                
            elif use_conditional_quantiles:
                # ADVANCED: Conditional quantile regression with asymmetric tails
                # Predicts Q_long(z|X) and Q_short(z|X) separately for crypto markets
                tprint(
                    f"📊 Using CONDITIONAL quantile regression: "
                    f"Q_long={conditional_quantile_long}, Q_short={conditional_quantile_short}, "
                    f"asymmetric={conditional_asymmetric}",
                    "INFO"
                )
                
                # Build meta-features first (needed for conditioning)
                # Use a minimal feature set for speed during labeling
                meta_features_for_cond = create_meta_features(
                    df=market_data,
                    signals=primary_signals,
                    volume_available=volume_available,
                    include_raw_signals=False,
                    use_kalman=True,
                )
                
                # Select stable features for conditioning (avoid noisy/leaky ones)
                stable_features = [
                    'volatility_1h', 'volatility_4h', 'volatility_1d', 'volatility_ema',
                    'vol_of_vol', 'momentum_20', 'momentum_ema', 'rsi_kalman',
                    'ma_distance_kalman', 'kalman_trend', 'range_position',
                    'hour_sin', 'hour_cos', 'day_of_week',
                ]
                feature_subset = [f for f in stable_features if f in meta_features_for_cond.columns]
                
                # Returns: labels (directional), labels_long, labels_short, diagnostics
                cond_labels, cond_labels_long, cond_labels_short, conditional_quantile_diagnostics = create_conditional_quantile_labels(
                    realized_returns=realized_returns,
                    features=meta_features_for_cond,
                    volatility=volatility_1d,
                    quantile_long=conditional_quantile_long,
                    quantile_short=conditional_quantile_short,
                    lookback_bars=rolling_lookback_bars,
                    min_train_samples=conditional_min_train,
                    retrain_frequency=conditional_retrain_freq,
                    vol_lookback=100,
                    use_lightgbm=True,
                    n_estimators=50,
                    max_depth=4,
                    feature_subset=feature_subset if feature_subset else None,
                    asymmetric_crypto=conditional_asymmetric,
                )
                
                # Convert directional labels to binary for compatibility with downstream code
                # For binary_labels: 1 = profitable trade (long or short), 0 = unprofitable
                # We use the long labels for the main binary_labels since direction is in consensus
                quantile_labels = cond_labels_long.copy()
                # Store short labels separately for potential use
                quantile_labels_short = cond_labels_short.copy()
                
                # Log diagnostics
                if conditional_quantile_diagnostics:
                    n_long = conditional_quantile_diagnostics.get('n_long', 0)
                    n_short = conditional_quantile_diagnostics.get('n_short', 0)
                    cov_long = conditional_quantile_diagnostics.get('coverage_long', np.nan)
                    cov_short = conditional_quantile_diagnostics.get('coverage_short', np.nan)
                    tprint(
                        f"📊 Conditional quantile results: {n_long} longs, {n_short} shorts",
                        "INFO"
                    )
                    if np.isfinite(cov_long) and np.isfinite(cov_short):
                        tprint(
                            f"   Calibration: long={cov_long:.1%} (exp={1.0-conditional_quantile_long:.1%}), "
                            f"short={cov_short:.1%} (exp={conditional_quantile_short:.1%})",
                            "INFO"
                        )
                    
            elif use_rolling_quantiles:
                # Use rolling quantiles to eliminate look-ahead bias
                tprint(f"📊 Using ROLLING quantiles (lookback={rolling_lookback_bars} bars) to prevent look-ahead bias", "INFO")
                if regimes_for_labeling is not None:
                    quantile_labels = create_rolling_regime_aware_quantile_labels_from_vol_scaled_returns(
                        vol_scaled=vol_scaled_returns,
                        regimes=regimes_for_labeling,
                        low_q=quantile_low_q,
                        high_q=quantile_high_q,
                        lookback_bars=rolling_lookback_bars,
                        min_periods=rolling_min_periods,
                    )
                else:
                    quantile_labels = create_rolling_quantile_labels_from_vol_scaled_returns(
                        vol_scaled=vol_scaled_returns,
                        low_q=quantile_low_q,
                        high_q=quantile_high_q,
                        lookback_bars=rolling_lookback_bars,
                        min_periods=rolling_min_periods,
                    )
            else:
                # Legacy: global quantiles (has look-ahead bias)
                tprint("⚠️ Using GLOBAL quantiles (may have look-ahead bias)", "WARNING")
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

            relabel_profitable_timeouts = bool(config.get("relabel_profitable_timeouts", True))
            profitable_timeout_return_threshold = float(
                config.get("profitable_timeout_return_threshold", PROFITABLE_TIMEOUT_RETURN_THRESHOLD)
            )
            if relabel_profitable_timeouts:
                try:
                    timeout_mask = (exit_reasons == 'timeout') & (realized_returns > profitable_timeout_return_threshold)
                    if isinstance(quantile_labels, pd.Series):
                        ql = quantile_labels.copy()
                        ql.loc[timeout_mask] = 1.0
                        quantile_labels = ql
                except Exception:
                    pass

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
            lgbm_params_override: Optional[Dict[str, Any]] = None
            if enable_meta_lgbm_hpo:
                try:
                    best_lgbm_params = tune_lgbm_hyperparameters_meta(
                        X=meta_features_model_processed,
                        y=binary_labels,
                        sample_weights=sample_weights,
                        horizon=horizon,
                    )
                    if best_lgbm_params:
                        lgbm_params_override = best_lgbm_params
                        used_hpo_params = True
                        tprint(f"⚙️ Meta LGBM HPO applied with params: {best_lgbm_params}", "INFO")
                except Exception as e_meta_hpo:
                    tprint(f"⚠️ Meta LGBM HPO failed, using default LGBM params: {e_meta_hpo}", "WARNING")

            ensemble_model_names = ['lgbm', 'xgb', 'rf']
            try:
                include_logreg = bool(config.get('include_logreg_in_meta_ensemble', True))
            except Exception:
                include_logreg = True
            if include_logreg and 'logreg' not in ensemble_model_names:
                ensemble_model_names.append('logreg')

            trained_models, oof_predictions_df = train_ensemble_with_kfold(
                X=meta_features_model_processed,
                y=binary_labels,
                horizon=horizon,
                n_splits=5,
                sample_weights=sample_weights,
                verbose=True,
                lgbm_params_override=lgbm_params_override,
                model_names=ensemble_model_names,
            )

            if meta_model_family in ('all', 'lgbm_bag_mean', 'lgbm_bag_lower'):
                try:
                    bagged_oof_df = train_bagged_lgbm_with_kfold(
                        X=meta_features_model_processed,
                        y=binary_labels,
                        horizon=horizon,
                        n_splits=5,
                        sample_weights=sample_weights,
                        n_bags=meta_lgbm_n_bags,
                        lgbm_base_params=best_lgbm_params if best_lgbm_params else None,
                        verbose=True,
                    )
                    if isinstance(bagged_oof_df, pd.DataFrame):
                        bagged_mean_series = bagged_oof_df.get('lgbm_bag_mean')
                        bagged_lower_series = bagged_oof_df.get('lgbm_bag_lower')
                except Exception as e_bagged:
                    tprint(f"⚠️ Bagged LGBM training failed, skipping bagged variants: {e_bagged}", "WARNING")
                    bagged_mean_series = None
                    bagged_lower_series = None

            # STEP 7: Add signal disagreement feature
            tprint("🔧 [7/13] Adding signal disagreement feature...", "INFO")
            meta_features_enhanced = add_signal_disagreement(
                oof_predictions=oof_predictions_df,
                meta_features=meta_features
            )

            # STEP 8: Calibrate ensemble with isotonic regression only (preserve variance)
            tprint("📈 [8/13] Calibrating ensemble (isotonic on blended predictions)...", "INFO")
            tprint(
                f"    Calibration inputs: n_oof={len(oof_predictions_df)}, "
                f"n_labels={binary_labels.notna().sum()}, "
                f"n_returns={realized_returns.notna().sum()}",
                "INFO",
            )

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

            chosen_probs_series = ensemble_probs_series
            if meta_model_family == 'lgbm_bag_mean' and bagged_mean_series is not None:
                try:
                    chosen_probs_series = bagged_mean_series.reindex(ensemble_probs_series.index).fillna(ensemble_probs_series)
                    meta_prob_source = 'lgbm_bag_mean'
                except Exception:
                    meta_prob_source = 'ensemble'
            elif meta_model_family == 'lgbm_bag_lower' and bagged_lower_series is not None:
                try:
                    chosen_probs_series = bagged_lower_series.reindex(ensemble_probs_series.index).fillna(ensemble_probs_series)
                    meta_prob_source = 'lgbm_bag_lower'
                except Exception:
                    meta_prob_source = 'ensemble'
            elif meta_model_family == 'all':
                meta_prob_source = 'ensemble'
            else:
                meta_prob_source = 'ensemble'

            probabilities = chosen_probs_series.values

            # Compute CV metrics for reporting
            cv_results = []

            n_metrics = min(len(binary_labels), len(oof_predictions_df))
            if n_metrics > 0:
                labels_for_metrics = binary_labels.iloc[-n_metrics:]
                preds_for_metrics = oof_predictions_df.iloc[-n_metrics:]

                mask = ~pd.isna(labels_for_metrics.to_numpy())
                for col in preds_for_metrics.columns:
                    col_arr = preds_for_metrics[col].to_numpy()
                    mask &= ~pd.isna(col_arr)

                if mask.sum() > 0:
                    y_oof = labels_for_metrics.iloc[mask]
                    for model_name in preds_for_metrics.columns:
                        try:
                            y_pred_proba = preds_for_metrics[model_name].iloc[mask]
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

            try:
                try:
                    enable_fs60_comparison = bool(config.get('enable_fs60_comparison', True))
                except Exception:
                    enable_fs60_comparison = True

                date_range_days: Optional[float]
                try:
                    if isinstance(market_data.index, pd.DatetimeIndex) and len(market_data.index) > 1:
                        delta = market_data.index[-1] - market_data.index[0]
                        date_range_days = max(delta.total_seconds() / 86400.0, 1.0)
                    else:
                        date_range_days = None
                except Exception:
                    date_range_days = None

                y_eval = binary_labels
                ens_meta_series = None
                if isinstance(oof_predictions_df, pd.DataFrame) and not oof_predictions_df.empty:
                    try:
                        ens_meta_series = oof_predictions_df.mean(axis=1)
                    except Exception:
                        ens_meta_series = None

                comparison_rows: List[Dict[str, Any]] = []

                def _align_arrays(y_series: pd.Series, prob_series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
                    """Align label and probability series positionally and return numpy arrays."""
                    y_local = y_series
                    p_local = prob_series
                    if len(y_local) != len(p_local):
                        if len(y_local) > len(p_local):
                            y_local = y_local.iloc[-len(p_local):]
                        else:
                            p_local = p_local.iloc[-len(y_local):]
                    y_arr = y_local.to_numpy()
                    p_arr = p_local.to_numpy()
                    return y_arr, p_arr

                def _compute_auc_series(y_series: pd.Series, prob_series: pd.Series) -> Optional[float]:
                    if prob_series is None:
                        return None
                    try:
                        y_arr, p_arr = _align_arrays(y_series, prob_series)
                        mask_local = ~(np.isnan(y_arr) | np.isnan(p_arr))
                        if mask_local.sum() < 10:
                            return None
                        y_clean = y_arr[mask_local]
                        p_clean = p_arr[mask_local]
                        if np.unique(y_clean).size < 2:
                            return None
                        return float(roc_auc_score(y_clean, p_clean))
                    except Exception:
                        return None

                def _compute_metrics_row(
                    feature_set: str,
                    model_variant: str,
                    y_series: pd.Series,
                    prob_series: Optional[pd.Series],
                ) -> Optional[Dict[str, Any]]:
                    if prob_series is None:
                        return None
                    try:
                        y_arr, p_arr = _align_arrays(y_series, prob_series)
                        mask_local = ~(np.isnan(y_arr) | np.isnan(p_arr))
                        n_eff = int(mask_local.sum())
                        if n_eff < 10:
                            return None
                        y_clean = y_arr[mask_local]
                        p_clean = p_arr[mask_local]
                        if np.unique(y_clean).size < 2:
                            return None

                        auc_val = float(roc_auc_score(y_clean, p_clean))

                        y_pred_05 = (p_clean >= 0.5).astype(int)
                        precision_05 = float(precision_score(y_clean, y_pred_05, zero_division=0))
                        recall_05 = float(recall_score(y_clean, y_pred_05, zero_division=0))
                        f1_05 = float(f1_score(y_clean, y_pred_05, zero_division=0))

                        y_pred_06 = (p_clean >= 0.6).astype(int)
                        precision_06 = float(precision_score(y_clean, y_pred_06, zero_division=0))
                        recall_06 = float(recall_score(y_clean, y_pred_06, zero_division=0))
                        f1_06 = float(f1_score(y_clean, y_pred_06, zero_division=0))

                        y_pred_07 = (p_clean >= 0.7).astype(int)
                        precision_07 = float(precision_score(y_clean, y_pred_07, zero_division=0))
                        recall_07 = float(recall_score(y_clean, y_pred_07, zero_division=0))
                        f1_07 = float(f1_score(y_clean, y_pred_07, zero_division=0))
                        try:
                            brier_val = float(brier_score_loss(y_clean, p_clean))
                        except Exception:
                            brier_val = float('nan')
                        try:
                            logloss_val = float(log_loss(y_clean, p_clean, eps=1e-15))
                        except Exception:
                            logloss_val = float('nan')
                        try:
                            ap_val = float(average_precision_score(y_clean, p_clean))
                        except Exception:
                            ap_val = float('nan')

                        pos_rate = float((y_clean == 1.0).mean())

                        trades_per_day_05 = float(((p_clean >= 0.5).sum() / date_range_days)) if date_range_days is not None else float('nan')
                        trades_per_day_06 = float(((p_clean >= 0.6).sum() / date_range_days)) if date_range_days is not None else float('nan')
                        trades_per_day_07 = float(((p_clean >= 0.7).sum() / date_range_days)) if date_range_days is not None else float('nan')

                        return {
                            'feature_set': feature_set,
                            'model_variant': model_variant,
                            'n_samples': n_eff,
                            'auc': auc_val,
                            'brier': brier_val,
                            'log_loss': logloss_val,
                            'average_precision': ap_val,
                            'precision_at_0_5': precision_05,
                            'recall_at_0_5': recall_05,
                            'f1_at_0_5': f1_05,
                            'precision_at_0_6': precision_06,
                            'recall_at_0_6': recall_06,
                            'f1_at_0_6': f1_06,
                            'precision_at_0_7': precision_07,
                            'recall_at_0_7': recall_07,
                            'f1_at_0_7': f1_07,
                            'positive_rate': pos_rate,
                            'trades_per_day_0_5': trades_per_day_05,
                            'trades_per_day_0_6': trades_per_day_06,
                            'trades_per_day_0_7': trades_per_day_07,
                        }
                    except Exception:
                        return None

                def _fmt_auc(v: Optional[float]) -> str:
                    try:
                        if v is None or not np.isfinite(v):
                            return "nan"
                        return f"{float(v):.4f}"
                    except Exception:
                        return "nan"

                meta_ensemble_auc = _compute_auc_series(y_eval, ens_meta_series) if ens_meta_series is not None else None
                meta_bag_mean_auc = _compute_auc_series(y_eval, bagged_mean_series) if isinstance(bagged_mean_series, pd.Series) else None
                meta_bag_lower_auc = _compute_auc_series(y_eval, bagged_lower_series) if isinstance(bagged_lower_series, pd.Series) else None

                row = _compute_metrics_row('meta_features', 'ensemble', y_eval, ens_meta_series) if ens_meta_series is not None else None
                if row is not None:
                    comparison_rows.append(row)
                row = _compute_metrics_row('meta_features', 'lgbm_bag_mean', y_eval, bagged_mean_series) if isinstance(bagged_mean_series, pd.Series) else None
                if row is not None:
                    comparison_rows.append(row)
                row = _compute_metrics_row('meta_features', 'lgbm_bag_lower', y_eval, bagged_lower_series) if isinstance(bagged_lower_series, pd.Series) else None
                if row is not None:
                    comparison_rows.append(row)

                if meta_ensemble_auc is not None or meta_bag_mean_auc is not None or meta_bag_lower_auc is not None:
                    tprint(
                        f"[META_MODEL_COMPARISON] meta_features ensemble AUC={_fmt_auc(meta_ensemble_auc)} "
                        f"lgbm_bag_mean AUC={_fmt_auc(meta_bag_mean_auc)} "
                        f"lgbm_bag_lower AUC={_fmt_auc(meta_bag_lower_auc)}",
                        "INFO",
                    )

                if enable_fs60_comparison:
                    fs_candidates = [
                        'selected_feature_dataframe_60',
                        f"{config.get('execution_mode', 'analyst')}_selected_feature_dataframe_60",
                        f"final_{config.get('execution_mode', 'analyst')}_dataset_60",
                    ]
                    fs_df = None
                    for artifact_name in fs_candidates:
                        try:
                            candidate_df = self._get_artifact(artifact_name, artifact_type='data')
                        except Exception:
                            candidate_df = None
                        if isinstance(candidate_df, pd.DataFrame) and not candidate_df.empty:
                            fs_df = candidate_df
                            break

                    if fs_df is not None:
                        feature_cols_60 = [
                            c for c in fs_df.columns
                            if c not in {'timestamp', 'target', 'label', 'target_long', 'target_short'}
                            and not str(c).lower().endswith('_target')
                            and not str(c).lower().endswith('_label')
                        ]
                        if feature_cols_60:
                            fs_features = fs_df[feature_cols_60]
                            base_index = meta_features_model_processed.index
                            fs_aligned = fs_features
                            try:
                                if isinstance(fs_aligned.index, pd.DatetimeIndex) and isinstance(base_index, pd.DatetimeIndex):
                                    fs_aligned = fs_aligned.reindex(base_index, method='ffill')
                                else:
                                    fs_arr = fs_aligned.reset_index(drop=True)
                                    n_base = len(base_index)
                                    if len(fs_arr) > n_base:
                                        fs_arr = fs_arr.iloc[-n_base:, :].reset_index(drop=True)
                                    elif len(fs_arr) < n_base:
                                        pad_rows = n_base - len(fs_arr)
                                        pad = pd.DataFrame(np.nan, index=range(pad_rows), columns=fs_arr.columns)
                                        fs_arr = pd.concat([pad, fs_arr], axis=0, ignore_index=True)
                                    fs_arr.index = base_index
                                    fs_aligned = fs_arr
                            except Exception:
                                pass

                            X_fs = prepare_feature_matrix(fs_aligned)
                            if not X_fs.empty:
                                fs_trained_models, fs_oof_df = train_ensemble_with_kfold(
                                    X=X_fs,
                                    y=binary_labels,
                                    horizon=horizon,
                                    n_splits=5,
                                    sample_weights=sample_weights,
                                    verbose=False,
                                    lgbm_params_override=lgbm_params_override,
                                    model_names=ensemble_model_names,
                                )

                                fs_ens_series = None
                                if isinstance(fs_oof_df, pd.DataFrame) and not fs_oof_df.empty:
                                    try:
                                        fs_ens_series = fs_oof_df.mean(axis=1)
                                    except Exception:
                                        fs_ens_series = None

                                fs_bagged_mean = None
                                fs_bagged_lower = None
                                if meta_model_family in ('all', 'lgbm_bag_mean', 'lgbm_bag_lower'):
                                    try:
                                        fs_bagged_df = train_bagged_lgbm_with_kfold(
                                            X=X_fs,
                                            y=binary_labels,
                                            horizon=horizon,
                                            n_splits=5,
                                            sample_weights=sample_weights,
                                            n_bags=meta_lgbm_n_bags,
                                            lgbm_base_params=best_lgbm_params if best_lgbm_params else None,
                                            verbose=False,
                                        )
                                        if isinstance(fs_bagged_df, pd.DataFrame):
                                            fs_bagged_mean = fs_bagged_df.get('lgbm_bag_mean')
                                            fs_bagged_lower = fs_bagged_df.get('lgbm_bag_lower')
                                    except Exception:
                                        fs_bagged_mean = None
                                        fs_bagged_lower = None

                                fs_ens_auc = _compute_auc_series(y_eval, fs_ens_series) if fs_ens_series is not None else None
                                fs_bag_mean_auc = _compute_auc_series(y_eval, fs_bagged_mean) if isinstance(fs_bagged_mean, pd.Series) else None
                                fs_bag_lower_auc = _compute_auc_series(y_eval, fs_bagged_lower) if isinstance(fs_bagged_lower, pd.Series) else None

                                row = _compute_metrics_row('fs60', 'ensemble', y_eval, fs_ens_series) if fs_ens_series is not None else None
                                if row is not None:
                                    comparison_rows.append(row)
                                row = _compute_metrics_row('fs60', 'lgbm_bag_mean', y_eval, fs_bagged_mean) if isinstance(fs_bagged_mean, pd.Series) else None
                                if row is not None:
                                    comparison_rows.append(row)
                                row = _compute_metrics_row('fs60', 'lgbm_bag_lower', y_eval, fs_bagged_lower) if isinstance(fs_bagged_lower, pd.Series) else None
                                if row is not None:
                                    comparison_rows.append(row)

                                tprint(
                                    f"[META_MODEL_COMPARISON_FS60] fs60 ensemble AUC={_fmt_auc(fs_ens_auc)} "
                                    f"lgbm_bag_mean AUC={_fmt_auc(fs_bag_mean_auc)} "
                                    f"lgbm_bag_lower AUC={_fmt_auc(fs_bag_lower_auc)}",
                                    "INFO",
                                )

                if comparison_rows:
                    try:
                        comparison_df = pd.DataFrame(comparison_rows)
                        ts_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                        symbol = str(config.get('symbol', 'UNKNOWN'))
                        exchange = str(config.get('exchange', 'binance'))
                        timeframe = str(config.get('timeframe', '15m'))
                        direction = str(config.get('direction', 'long'))
                        model = str(config.get('model', 'analyst'))

                        filename = (
                            f"meta_model_feature_set_comparison_"
                            f"{symbol}_{exchange}_{timeframe}_{direction}_{model}_{ts_str}.csv"
                        )
                        output_dir = Path('outcomes')
                        try:
                            output_dir.mkdir(parents=True, exist_ok=True)
                        except Exception:
                            pass

                        csv_path = output_dir / filename
                        comparison_df.to_csv(csv_path, index=False)
                        tprint(f"✅ Saved meta-model feature-set comparison CSV to {csv_path}", "SUCCESS")
                    except Exception as csv_exc:
                        tprint(f"⚠️ Failed to save meta-model feature-set comparison CSV: {csv_exc}", "WARNING")
            except Exception as comp_exc:
                tprint(f"⚠️ Meta-model feature-set comparison skipped due to error: {comp_exc}", "WARNING")

            # STEP 10: Train final models on full dataset (for deployment)
            tprint("🎓 [10/13] Training final ensemble models on full dataset...", "INFO")

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

            # Align features and labels positionally (tail-aligned) to avoid
            # index alignment issues (including any datetime tz mismatches).
            X_full = pd.DataFrame(X_full)
            y_series = binary_labels

            n_full = min(len(X_full), len(y_series))
            if n_full > 0:
                X_tail = X_full.iloc[-n_full:]
                y_tail = y_series.iloc[-n_full:]
                full_mask_arr = ~pd.isna(y_tail.to_numpy())
                X_full = X_tail.iloc[full_mask_arr].fillna(0)
                y_full = y_tail.iloc[full_mask_arr]
            else:
                X_full = X_full.iloc[0:0]
                y_full = y_series.iloc[0:0]

            final_models = create_base_models({}, use_focal_loss=False)
            for model_name, model in final_models.items():
                if model_name == 'logreg':
                    tprint("  ⏭ Skipping final logreg training (not used in meta ensemble)", "INFO")
                    continue
                try:
                    model.fit(X_full, y_full)
                    tprint(f"  ✓ Trained final {model_name}", "INFO")
                except Exception as e:
                    tprint(f"  ❌ Failed to train final {model_name}: {e}", "ERROR")

            # Use first final model for feature importance reporting (RF preferred)
            final_model = final_models.get('rf', list(final_models.values())[0])

            # STEP 11: Translate to targets using isotonic regression
            # Strategy-aware target generation: different labels for trend following vs mean reversion
            strategy_type = config.get('strategy_type', 'auto')  # 'trend_following', 'mean_reversion', or 'auto'
            tprint(f"🔄 [11/13] Translating probabilities to economic targets (strategy_type={strategy_type})...", "INFO")

            if iso_regressor is not None:
                # Apply symmetric probability clipping if configured/HPO-provided
                iso_min_prob = max(0.0, min(0.1, iso_min_prob_param))
                iso_max_prob = 1.0 - iso_min_prob
                iso_max_prob = max(0.9, min(1.0, iso_max_prob))

                prob_array = np.asarray(probabilities, dtype=float)
                prob_clipped = np.clip(prob_array, iso_min_prob, iso_max_prob)

                # Use strategy-aware target generation if strategy_type is specified
                if strategy_type in ['trend_following', 'mean_reversion']:
                    tprint(f"  📊 Using strategy-aware target generation: {strategy_type}", "INFO")
                    target_long, target_short = generate_strategy_aware_targets(
                        realized_returns,
                        prob_clipped,
                        primary_signals,
                        iso_regressor,
                        strategy_type=strategy_type,
                        cost_threshold=transaction_cost
                    )
                elif strategy_type == 'auto':
                    # Auto-detect strategy based on signal composition
                    # Use volatility ratio to determine dominant regime
                    if 'vol_ratio_for_consensus' in primary_signals.columns:
                        mean_vol_ratio = primary_signals['vol_ratio_for_consensus'].mean()
                        if mean_vol_ratio < 0.85:
                            detected_strategy = 'mean_reversion'
                            tprint(f"  🔍 Auto-detected strategy: mean_reversion (vol_ratio={mean_vol_ratio:.3f})", "INFO")
                        else:
                            detected_strategy = 'trend_following'
                            tprint(f"  🔍 Auto-detected strategy: trend_following (vol_ratio={mean_vol_ratio:.3f})", "INFO")
                        
                        target_long, target_short = generate_strategy_aware_targets(
                            realized_returns,
                            prob_clipped,
                            primary_signals,
                            iso_regressor,
                            strategy_type=detected_strategy,
                            cost_threshold=transaction_cost
                        )
                    else:
                        # Fallback to default isotonic translation
                        tprint("  ⚠️ Cannot auto-detect strategy, using default translation", "WARNING")
                        target_long, target_short = translate_to_targets_with_isotonic(
                            realized_returns,
                            prob_clipped,
                            primary_signals,
                            iso_regressor
                        )
                else:
                    # Default: use standard isotonic translation
                    target_long, target_short = translate_to_targets_with_isotonic(
                        realized_returns,
                        prob_clipped,
                        primary_signals,
                        iso_regressor
                    )

                # Optional symmetric quantile clipping of target magnitudes
                # Now handles both positive and negative targets
                if target_clip_high_q_param is not None:
                    try:
                        q_high = float(target_clip_high_q_param)
                        q_high = max(0.90, min(0.99, q_high))
                        q_low = 1.0 - q_high  # Symmetric quantile for negative side

                        for series in (target_long, target_short):
                            # Handle positive values
                            pos_mask = series > 0
                            pos_vals = series[pos_mask]
                            if len(pos_vals) >= 100:
                                high_val = pos_vals.quantile(q_high)
                                series.loc[pos_mask] = series.loc[pos_mask].clip(upper=high_val)
                            
                            # Handle negative values (clip extreme negatives symmetrically)
                            neg_mask = series < 0
                            neg_vals = series[neg_mask]
                            if len(neg_vals) >= 100:
                                low_val = neg_vals.quantile(q_low)  # q_low quantile of negatives
                                series.loc[neg_mask] = series.loc[neg_mask].clip(lower=low_val)
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
            # NEW: Directional binary labels for training long-only or short-only classifiers
            # - binary_label_long: Success/failure for long trades only (NaN for shorts)
            # - binary_label_short: Success/failure for short trades only (NaN for longs)
            # These allow training models that specialize in one direction without mixing signals
            labeled_data['binary_label_long'] = binary_labels_long
            labeled_data['binary_label_short'] = binary_labels_short
            labeled_data['smoothed_label'] = smoothed_labels
            labeled_data['label_uncertainty'] = label_uncertainty
            labeled_data['meta_probability'] = probabilities

            try:
                if 'ensemble_probs_series' in locals():
                    ens_series = pd.Series(np.nan, index=labeled_data.index)
                    ens_src = ensemble_probs_series
                    if len(ens_src) >= len(ens_series):
                        ens_vals = ens_src.iloc[-len(ens_series):].astype(float).to_numpy()
                        ens_series.iloc[:] = ens_vals
                    else:
                        ens_vals = ens_src.astype(float).to_numpy()
                        ens_series.iloc[-len(ens_vals):] = ens_vals
                    labeled_data['meta_probability_ensemble'] = ens_series.astype(np.float32)

                if bagged_mean_series is not None:
                    mean_series = pd.Series(np.nan, index=labeled_data.index)
                    mean_src = bagged_mean_series
                    if len(mean_src) >= len(mean_series):
                        mean_vals = mean_src.iloc[-len(mean_series):].astype(float).to_numpy()
                        mean_series.iloc[:] = mean_vals
                    else:
                        mean_vals = mean_src.astype(float).to_numpy()
                        mean_series.iloc[-len(mean_vals):] = mean_vals
                    labeled_data['meta_probability_lgbm_bag_mean'] = mean_series.astype(np.float32)

                if bagged_lower_series is not None:
                    lower_series = pd.Series(np.nan, index=labeled_data.index)
                    lower_src = bagged_lower_series
                    if len(lower_src) >= len(lower_series):
                        lower_vals = lower_src.iloc[-len(lower_series):].astype(float).to_numpy()
                        lower_series.iloc[:] = lower_vals
                    else:
                        lower_vals = lower_src.astype(float).to_numpy()
                        lower_series.iloc[-len(lower_vals):] = lower_vals
                    labeled_data['meta_probability_lgbm_bag_lower'] = lower_series.astype(np.float32)

                labeled_data['meta_probability_source'] = str(meta_prob_source)
            except Exception:
                pass
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

            # Ensure all object-dtype columns are HDF5-safe. For HDF5, we
            # prefer numeric encodings over generic Python objects.
            #
            # - exit_reason: encode as categorical integer codes
            # - labeled_data_schema_version: store as a numeric version id
            if 'exit_reason' in labeled_data.columns:
                try:
                    cat = labeled_data['exit_reason'].astype('category')
                    labeled_data['exit_reason'] = cat.cat.codes.astype('int16')
                except Exception:
                    # Fallback: treat missing/unknown as -1
                    labeled_data['exit_reason'] = pd.Series(-1, index=labeled_data.index, dtype='int16')

            if 'labeled_data_schema_version' in labeled_data.columns:
                try:
                    version_numeric = float(LABELED_DATA_SCHEMA_VERSION)
                except Exception:
                    version_numeric = 1.0
                labeled_data['labeled_data_schema_version'] = np.full(
                    len(labeled_data), version_numeric, dtype='float32'
                )

            # Ensure index is HDF5-safe: prefer DatetimeIndex, else fall back
            # to a simple RangeIndex so that the HDF5 backend never sees an
            # object-typed index.
            if not isinstance(labeled_data.index, pd.DatetimeIndex):
                try:
                    idx_dt = pd.to_datetime(labeled_data.index, errors="coerce")
                    if isinstance(idx_dt, pd.DatetimeIndex) and idx_dt.notna().all():
                        labeled_data.index = idx_dt
                        tprint("    ℹ️ Coerced labeled_data index to DatetimeIndex for HDF5 storage", "INFO")
                    else:
                        labeled_data.index = pd.RangeIndex(len(labeled_data))
                        tprint("    ℹ️ Using RangeIndex for labeled_data (HDF5-safe)", "INFO")
                except Exception:
                    labeled_data.index = pd.RangeIndex(len(labeled_data))
                    tprint("    ℹ️ Fallback: Using RangeIndex for labeled_data (HDF5-safe)", "INFO")

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
                    'trail_distance_atr': float(trail_dist) if 'trail_dist' in locals() else 0.0,
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
                # Create effective config with actual parameters used (including HPO overrides)
                effective_config = config.copy()
                effective_config.update({
                    'profit_threshold': profit_threshold,
                    'stop_threshold': stop_threshold,
                    'horizon': horizon,
                    'transaction_cost': transaction_cost,
                    'trail_distance': float(trail_dist) if 'trail_dist' in locals() else 0.0,
                })

                # Attach signal funnel statistics if they were produced during signal generation
                try:
                    signal_funnel = primary_signals.attrs.get('signal_funnel', {})  # type: ignore[name-defined]
                except Exception:
                    signal_funnel = {}
                if signal_funnel:
                    effective_config['signal_funnel'] = signal_funnel

                diagnostics_path = generate_diagnostics_report(
                    labeled_data=labeled_data,
                    meta_features=meta_features_enhanced,  # Use enhanced features with disagreement
                    binary_labels=binary_labels,
                    realized_returns=realized_returns,
                    smoothed_labels=smoothed_labels,
                    probabilities=probabilities,
                    final_model=final_model,
                    config=effective_config,
                    output_dir=outcomes_dir,
                    exit_reasons=exit_reasons,
                    event_durations=event_durations,
                    mfe_series=mfe_series,
                    mae_series=mae_series,
                    target_long=target_long,
                    target_short=target_short,
                    selected_feature_names=selected_feature_names,
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
                        event_mask_arr = event_mask.to_numpy(dtype=bool, copy=False)
                        # Positions of valid events in time order
                        event_positions = np.flatnonzero(event_mask_arr)
                        n_events = int(event_positions.size)

                        best_cfg = None
                        train_metrics = None
                        holdout_metrics = None

                        if n_events >= 20:
                            # Use positional masking to avoid any index alignment issues
                            p_all = ensemble_probs_series.to_numpy(dtype=float)
                            r_all = realized_returns.to_numpy(dtype=float)

                            # Expected returns from isotonic mapping (if possible)
                            try:
                                E_hat_all = iso_regressor.predict(p_all)
                            except Exception:
                                E_hat_all = None

                            # Time-ordered split: earlier fraction for gate tuning, later for internal holdout
                            train_frac = float(config.get("meta_gating_train_fraction", 0.7) or 0.7)
                            if train_frac <= 0.0 or train_frac >= 1.0:
                                train_frac = 0.7

                            n_train = int(max(20, min(n_events - 10, int(round(n_events * train_frac)))))
                            if n_train <= 0 or n_train >= n_events:
                                n_train = max(20, n_events // 2)

                            train_pos = event_positions[:n_train]
                            holdout_pos = event_positions[n_train:]

                            p_train = p_all[train_pos]
                            r_train = r_all[train_pos]
                            E_train = E_hat_all[train_pos] if E_hat_all is not None else None

                            train_days = 1.0
                            try:
                                train_dates = realized_returns.index[train_pos]
                                if len(train_dates) >= 2:
                                    delta = train_dates[-1] - train_dates[0]
                                    train_days = max(
                                        1.0,
                                        delta.days + delta.seconds / 86400.0,
                                    )
                            except Exception:
                                train_days = 1.0

                            target_trades_per_day_min = float(
                                config.get("meta_gating_target_trades_per_day_min", 1.0) or 1.0
                            )
                            target_trades_per_day_max = float(
                                config.get("meta_gating_target_trades_per_day_max", 2.0) or 2.0
                            )
                            if target_trades_per_day_max < target_trades_per_day_min:
                                target_trades_per_day_max = target_trades_per_day_min

                            # Thresholds for meta-gating search: probability thresholds starting at 0.55
                            # to ensure meaningful filtering, with reduced expected-return multipliers
                            prob_thresholds = [0.55, 0.60, 0.65, 0.70, 0.75]
                            # With tx_cost ≈ 0.3%, these multipliers correspond to ≈0.075%–0.30%
                            # expected-return thresholds, instead of the previous 0.15%–0.60% range.
                            er_multipliers = [0.25, 0.5, 0.75, 1.0]
                            tx_cost = float(transaction_cost)

                            def _evaluate_gate_local(p_arr, r_arr, E_arr, p_thr_val, E_thr_val):
                                gate_local = p_arr >= p_thr_val
                                if E_arr is not None and E_thr_val > 0.0:
                                    gate_local &= (E_arr >= E_thr_val)
                                gated_r_local = r_arr[gate_local]
                                n_trades_local = int(gated_r_local.size)
                                if n_trades_local == 0:
                                    return n_trades_local, 0.0, 0.0
                                mean_r_local = float(np.mean(gated_r_local))
                                std_r_local = float(np.std(gated_r_local, ddof=1)) if n_trades_local > 1 else 0.0
                                sharpe_local = float(mean_r_local / std_r_local) if std_r_local > 0.0 else 0.0
                                return n_trades_local, mean_r_local, sharpe_local

                            for p_thr in prob_thresholds:
                                for k in er_multipliers:
                                    # If isotonic mapping is not available, fall back to prob-only gating
                                    E_thr = tx_cost * k if E_train is not None else 0.0

                                    n_trades, mean_r, sharpe = _evaluate_gate_local(
                                        p_train, r_train, E_train, float(p_thr), float(E_thr)
                                    )

                                    if n_trades == 0:
                                        continue
                                    if mean_r <= 0.0:
                                        continue

                                    trades_per_day = float(n_trades) / float(train_days)
                                    freq_penalty = 1.0
                                    if trades_per_day < target_trades_per_day_min:
                                        freq_penalty = trades_per_day / target_trades_per_day_min
                                    elif trades_per_day > target_trades_per_day_max:
                                        freq_penalty = target_trades_per_day_max / trades_per_day

                                    if freq_penalty <= 0.0:
                                        continue

                                    score = sharpe * np.sqrt(max(n_trades, 1)) * freq_penalty

                                    if (best_cfg is None) or (score > best_cfg["score"]):
                                        best_cfg = {
                                            "prob_threshold": float(p_thr),
                                            "expected_return_threshold": float(E_thr),
                                            "mean_return": mean_r,
                                            "sharpe": sharpe,
                                            "n_trades": n_trades,
                                            "score": float(score),
                                        }

                            # Fallback: if expected-return gated search yields zero trades,
                            # derive a gate using probability-only thresholds.
                            if best_cfg is None or int(best_cfg.get("n_trades", 0)) == 0:
                                fallback_cfg = None
                                for p_thr in prob_thresholds:
                                    n_fb, mean_fb, sharpe_fb = _evaluate_gate_local(
                                        p_train,
                                        r_train,
                                        None,
                                        float(p_thr),
                                        0.0,
                                    )
                                    if n_fb == 0:
                                        continue
                                    if mean_fb <= 0.0:
                                        continue

                                    trades_per_day_fb = float(n_fb) / float(train_days)
                                    freq_penalty_fb = 1.0
                                    if trades_per_day_fb < target_trades_per_day_min:
                                        freq_penalty_fb = trades_per_day_fb / target_trades_per_day_min
                                    elif trades_per_day_fb > target_trades_per_day_max:
                                        freq_penalty_fb = target_trades_per_day_max / trades_per_day_fb

                                    if freq_penalty_fb <= 0.0:
                                        continue

                                    score_fb = sharpe_fb * np.sqrt(max(n_fb, 1)) * freq_penalty_fb
                                    if (fallback_cfg is None) or (score_fb > fallback_cfg["score"]):
                                        fallback_cfg = {
                                            "prob_threshold": float(p_thr),
                                            "expected_return_threshold": 0.0,
                                            "mean_return": mean_fb,
                                            "sharpe": sharpe_fb,
                                            "n_trades": int(n_fb),
                                            "score": float(score_fb),
                                        }

                                if fallback_cfg is not None:
                                    best_cfg = fallback_cfg

                            # Compute simple internal holdout metrics for information only
                            if best_cfg is not None and holdout_pos.size > 0:
                                p_hold = p_all[holdout_pos]
                                r_hold = r_all[holdout_pos]
                                E_hold = E_hat_all[holdout_pos] if E_hat_all is not None else None
                                n_tr_train, mean_train, sharpe_train = _evaluate_gate_local(
                                    p_train,
                                    r_train,
                                    E_train,
                                    best_cfg["prob_threshold"],
                                    best_cfg["expected_return_threshold"],
                                )
                                n_tr_hold, mean_hold, sharpe_hold = _evaluate_gate_local(
                                    p_hold,
                                    r_hold,
                                    E_hold,
                                    best_cfg["prob_threshold"],
                                    best_cfg["expected_return_threshold"],
                                )
                                train_metrics = {
                                    "n_trades": int(n_tr_train),
                                    "mean_return": float(mean_train),
                                    "sharpe": float(sharpe_train),
                                }
                                holdout_metrics = {
                                    "n_trades": int(n_tr_hold),
                                    "mean_return": float(mean_hold),
                                    "sharpe": float(sharpe_hold),
                                }

                        regime_gating = {}
                        try:
                            if "hmm_regime_label_1h" in labeled_data.columns:
                                # Restrict to the same event window using positional masking
                                reg_all_events = labeled_data["hmm_regime_label_1h"].iloc[event_mask_arr]
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
                        if best_cfg is None or int(best_cfg.get("n_trades", 0)) <= 0:
                            tprint(
                                "⚠️ Meta-gating config not created: no valid gate found; skipping save.",
                                "WARNING",
                            )
                        else:
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
                                        "use_expected_return": bool(
                                            best_cfg["expected_return_threshold"] > 0
                                        ),
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
                                        "trail_distance_atr": float(trail_dist) if 'trail_dist' in locals() else 0.0,
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
                        'trail_distance_atr': float(trail_dist) if 'trail_dist' in locals() else 0.0,
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
