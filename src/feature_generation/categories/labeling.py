"""
Labeling Feature Generator for Meta-Labeling

This module provides features specifically designed for meta-labeling models,
optimized for 2-4 period trading windows. All features are normalized, stationary,
and vectorized for maximum performance.

Features are organized into two categories:
1. Market Features: Input to base signal models (RSI, MACD, etc.)
2. Signal Features: Input to meta-learner (disagreement, persistence, etc.)

Key Design Principles:
- No non-stationary raw values (price, volume, raw ATR)
- All features normalized (ratios, %, z-scores)
- Vectorized with numba/numpy/vectorbt
- Aligned with 2-4 period horizon
"""

import warnings
import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from numba import njit, prange

from ..core.feature_generator import (
    FeatureGenerator,
    FeatureResult,
    VectorizedFeatureGenerator,
    FeatureConfig,
    FeatureCategory
)

# Import normalization utilities
try:
    from src.features_common.normalization import NormalizationFeatureGenerator
    NORMALIZATION_AVAILABLE = True
except ImportError:
    NORMALIZATION_AVAILABLE = False

# VectorBT for performance
try:
    import vectorbt as vbt
    from src.utils.vectorbt_compat import (
        rolling_mean, rolling_std, rolling_min, rolling_max,
        rolling_sum, zscore as vbt_zscore
    )
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    warnings.warn("VectorBT not available - using slower implementations")

logger = logging.getLogger(__name__)


# ============================================================================
# NUMBA-ACCELERATED UTILITY FUNCTIONS
# ============================================================================

@njit
def fast_rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Numba-accelerated rolling mean."""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        result[i] = np.mean(arr[i - window + 1:i + 1])
    return result


@njit
def fast_rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Numba-accelerated rolling std."""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        result[i] = np.std(arr[i - window + 1:i + 1])
    return result


@njit
def fast_rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
    """Numba-accelerated rolling min."""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        result[i] = np.min(arr[i - window + 1:i + 1])
    return result


@njit
def fast_rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    """Numba-accelerated rolling max."""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        result[i] = np.max(arr[i - window + 1:i + 1])
    return result


@njit(parallel=True)
def fast_bollinger_bands(close: np.ndarray, window: int = 20, num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-accelerated Bollinger Bands calculation.

    Returns: (upper, middle, lower)
    """
    n = len(close)
    middle = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)

    for i in prange(window - 1, n):
        window_data = close[i - window + 1:i + 1]
        mean_val = np.mean(window_data)
        std_val = np.std(window_data)

        middle[i] = mean_val
        upper[i] = mean_val + num_std * std_val
        lower[i] = mean_val - num_std * std_val

    return upper, middle, lower


@njit
def fast_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> np.ndarray:
    """
    Numba-accelerated Average True Range.
    """
    n = len(close)
    tr = np.full(n, np.nan)
    atr = np.full(n, np.nan)

    # True Range
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    # ATR (simple moving average of TR)
    for i in range(window, n):
        atr[i] = np.mean(tr[i - window + 1:i + 1])

    return atr


@njit
def fast_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Numba-accelerated RSI calculation.
    """
    n = len(prices)
    rsi = np.full(n, np.nan)

    # Calculate price changes
    deltas = np.diff(prices)

    for i in range(period, n):
        gains = np.where(deltas[i - period:i] > 0, deltas[i - period:i], 0.0)
        losses = np.where(deltas[i - period:i] < 0, -deltas[i - period:i], 0.0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


@njit
def fast_macd(prices: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-accelerated MACD calculation.

    Returns: (macd_line, signal_line, histogram)
    """
    n = len(prices)

    # EMA calculation
    fast_ema = np.full(n, np.nan)
    slow_ema = np.full(n, np.nan)

    # Initialize
    fast_ema[fast_period - 1] = np.mean(prices[:fast_period])
    slow_ema[slow_period - 1] = np.mean(prices[:slow_period])

    fast_multiplier = 2.0 / (fast_period + 1)
    slow_multiplier = 2.0 / (slow_period + 1)

    # Calculate EMAs
    for i in range(fast_period, n):
        fast_ema[i] = (prices[i] - fast_ema[i - 1]) * fast_multiplier + fast_ema[i - 1]

    for i in range(slow_period, n):
        slow_ema[i] = (prices[i] - slow_ema[i - 1]) * slow_multiplier + slow_ema[i - 1]

    # MACD line
    macd_line = fast_ema - slow_ema

    # Signal line (EMA of MACD)
    signal_line = np.full(n, np.nan)
    start_idx = slow_period + signal_period - 1

    if start_idx < n:
        signal_line[start_idx] = np.nanmean(macd_line[slow_period:start_idx + 1])
        signal_multiplier = 2.0 / (signal_period + 1)

        for i in range(start_idx + 1, n):
            if not np.isnan(macd_line[i]) and not np.isnan(signal_line[i - 1]):
                signal_line[i] = (macd_line[i] - signal_line[i - 1]) * signal_multiplier + signal_line[i - 1]

    # Histogram
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================

@dataclass
class LabelingFeatureConfig(FeatureConfig):
    """Configuration for labeling features."""
    # Window sizes (aligned with 2-4 period horizon)
    short_window: int = 4   # For 2-4 period features
    medium_window: int = 12  # 3x short
    long_window: int = 48    # 12x short (4x medium)

    # RSI parameters (LOOSER thresholds)
    rsi_period: int = 14
    rsi_period_long: int = 56  # 4x longer
    rsi_oversold: float = 25.0  # LOOSER (was 30)
    rsi_overbought: float = 75.0  # LOOSER (was 70)

    # MACD parameters (LOOSER thresholds)
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    macd_fast_long: int = 48  # 4x longer
    macd_slow_long: int = 104  # 4x longer
    macd_signal_long: int = 36  # 4x longer
    macd_threshold: float = 0.02  # LOOSER difference threshold

    # Bollinger Bands
    bb_window: int = 20
    bb_num_std: float = 2.0

    # ATR
    atr_window: int = 14

    # Volume
    volume_window: int = 20

    # Normalization
    use_robust_scaling: bool = True
    outlier_threshold: float = 3.0


# ============================================================================
# MARKET FEATURES (Input to base signal models)
# ============================================================================

class MarketFeaturesGenerator(VectorizedFeatureGenerator):
    """
    Generates market context features for base signal models.

    These features describe market conditions and are inputs to
    primary signal models (RSI, MACD, etc.).
    """

    def __init__(self, config: Optional[LabelingFeatureConfig] = None):
        if config is None:
            config = LabelingFeatureConfig(
                name="market_features",
                category=FeatureCategory.TECHNICAL_INDICATORS,
                description="Market context features for signal generation",
                required_columns=["open", "high", "low", "close", "volume"],
                default_lookback=48
            )
        super().__init__(config)
        self.config = config

    def generate(self, df: pd.DataFrame) -> FeatureResult:
        """Generate all market features."""
        features = {}

        # Convert to numpy for speed
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values if 'volume' in df.columns else None

        # ===== RSI FEATURES (short + long term) =====
        rsi_short = fast_rsi(close, self.config.rsi_period)
        rsi_long = fast_rsi(close, self.config.rsi_period_long)

        # Normalized RSI (0-100 → -1 to +1)
        features['rsi_norm'] = (rsi_short - 50) / 50
        features['rsi_long_norm'] = (rsi_long - 50) / 50
        features['rsi_delta'] = np.diff(rsi_short, prepend=np.nan)  # ∆RSI

        # RSI z-score (volatility-adjusted)
        rsi_mean = fast_rolling_mean(rsi_short, 20)
        rsi_std = fast_rolling_std(rsi_short, 20)
        features['rsi_zscore'] = (rsi_short - rsi_mean) / (rsi_std + 1e-8)

        # ===== MACD FEATURES (short + long term) =====
        macd, signal, hist = fast_macd(close, self.config.macd_fast, self.config.macd_slow, self.config.macd_signal)
        macd_long, signal_long, hist_long = fast_macd(close, self.config.macd_fast_long, self.config.macd_slow_long, self.config.macd_signal_long)

        # Normalized MACD (as % of price)
        features['macd_pct'] = macd / (close + 1e-8)
        features['macd_signal_pct'] = signal / (close + 1e-8)
        features['macd_hist_pct'] = hist / (close + 1e-8)
        features['macd_long_pct'] = macd_long / (close + 1e-8)
        features['macd_delta'] = np.diff(hist, prepend=np.nan)  # ∆MACD histogram

        # MACD histogram consistency (how many consecutive bars increasing/decreasing)
        features['macd_hist_consistency'] = self._calculate_consistency(hist)

        # ===== BOLLINGER BANDS =====
        bb_upper, bb_mid, bb_lower = fast_bollinger_bands(close, self.config.bb_window, self.config.bb_num_std)

        # % distance from bands (normalized)
        features['bb_pct_from_upper'] = (close - bb_upper) / (bb_upper + 1e-8)
        features['bb_pct_from_lower'] = (close - bb_lower) / (bb_lower + 1e-8)
        features['bb_pct_from_mid'] = (close - bb_mid) / (bb_mid + 1e-8)

        # Bandwidth (volatility measure)
        features['bb_bandwidth'] = (bb_upper - bb_lower) / (bb_mid + 1e-8)
        features['bb_bandwidth_pct_change'] = np.diff(features['bb_bandwidth'], prepend=np.nan) / (features['bb_bandwidth'] + 1e-8)

        # ===== ATR FEATURES =====
        atr = fast_atr(high, low, close, self.config.atr_window)

        # ATR as % of price (stationary)
        features['atr_pct'] = atr / (close + 1e-8)
        features['atr_pct_change'] = np.diff(atr, prepend=np.nan) / (atr + 1e-8)

        # ATR z-score
        atr_mean = fast_rolling_mean(atr, 20)
        atr_std = fast_rolling_std(atr, 20)
        features['atr_zscore'] = (atr - atr_mean) / (atr_std + 1e-8)

        # Volatility spike detection
        atr_change_5 = (atr - np.roll(atr, 5)) / (np.roll(atr, 5) + 1e-8)
        features['atr_spike_5bar'] = atr_change_5

        # ===== MOVING AVERAGES =====
        ema_fast = pd.Series(close).ewm(span=self.config.short_window).mean().values
        ema_medium = pd.Series(close).ewm(span=self.config.medium_window).mean().values
        ema_slow = pd.Series(close).ewm(span=self.config.long_window).mean().values

        # % distance from EMAs
        features['price_vs_ema_fast'] = (close - ema_fast) / (ema_fast + 1e-8)
        features['price_vs_ema_medium'] = (close - ema_medium) / (ema_medium + 1e-8)
        features['price_vs_ema_slow'] = (close - ema_slow) / (ema_slow + 1e-8)

        # EMA slope (momentum)
        features['ema_fast_slope'] = np.diff(ema_fast, prepend=np.nan) / (ema_fast + 1e-8)
        features['ema_medium_slope'] = np.diff(ema_medium, prepend=np.nan) / (ema_medium + 1e-8)

        # ===== VOLATILITY-ADJUSTED FEATURES =====
        # Price change / ATR (normalizes for volatility)
        price_change = np.diff(close, prepend=np.nan)
        features['price_change_per_atr'] = price_change / (atr + 1e-8)

        # RSI change / ATR
        features['rsi_change_per_atr'] = features['rsi_delta'] / (features['atr_pct'] + 1e-8)

        # ===== VOLUME FEATURES =====
        if volume is not None:
            # Relative volume (current / rolling mean)
            volume_ma = fast_rolling_mean(volume, self.config.volume_window)
            features['volume_ratio'] = volume / (volume_ma + 1e-8)

            # Volume spike
            features['volume_spike'] = (volume - volume_ma) / (volume_ma + 1e-8)

            # Volume z-score
            volume_std = fast_rolling_std(volume, self.config.volume_window)
            features['volume_zscore'] = (volume - volume_ma) / (volume_std + 1e-8)
        else:
            features['volume_ratio'] = np.ones_like(close)
            features['volume_spike'] = np.zeros_like(close)
            features['volume_zscore'] = np.zeros_like(close)

        # ===== VOLATILITY MEASURES =====
        # Rolling standard deviation of returns
        returns = np.diff(np.log(close), prepend=np.nan)
        features['returns_std_4bar'] = fast_rolling_std(returns, 4)
        features['returns_std_12bar'] = fast_rolling_std(returns, 12)

        # ===== MEAN REVERSION =====
        # Z-score of short-term price vs medium-term EMA
        price_vs_ema_std = fast_rolling_std(close - ema_medium, 20)
        features['mean_reversion_zscore'] = (close - ema_medium) / (price_vs_ema_std + 1e-8)

        # ===== MULTI-TIMEFRAME ALIGNMENT =====
        # EMA slope agreement
        ema_fast_up = features['ema_fast_slope'] > 0
        ema_medium_up = features['ema_medium_slope'] > 0
        features['ema_slope_agreement'] = (ema_fast_up == ema_medium_up).astype(float)

        # MACD histogram sign alignment (short vs long)
        macd_hist_sign = np.sign(hist)
        macd_hist_long_sign = np.sign(hist_long)
        features['macd_sign_alignment'] = (macd_hist_sign == macd_hist_long_sign).astype(float)

        # ===== AUTOCORRELATION (simplified) =====
        # Rolling correlation of price with 1-bar lag
        features['price_autocorr_lag1'] = self._rolling_autocorr(close, lag=1, window=20)
        features['rsi_autocorr_lag1'] = self._rolling_autocorr(rsi_short, lag=1, window=20)

        # Convert to DataFrame
        result_df = pd.DataFrame(features, index=df.index)

        return FeatureResult(
            success=True,
            features=result_df,
            feature_names=list(features.keys()),
            metadata={
                'n_features': len(features),
                'config': self.config.__dict__
            }
        )

    @staticmethod
    @njit
    def _calculate_consistency(arr: np.ndarray) -> np.ndarray:
        """
        Calculate how many consecutive bars the array has been increasing/decreasing.

        Positive = consecutive increases, Negative = consecutive decreases.
        """
        n = len(arr)
        consistency = np.zeros(n)

        for i in range(1, n):
            if np.isnan(arr[i]) or np.isnan(arr[i - 1]):
                consistency[i] = 0
            elif arr[i] > arr[i - 1]:
                consistency[i] = max(1, consistency[i - 1] + 1) if consistency[i - 1] > 0 else 1
            elif arr[i] < arr[i - 1]:
                consistency[i] = min(-1, consistency[i - 1] - 1) if consistency[i - 1] < 0 else -1
            else:
                consistency[i] = 0

        return consistency

    @staticmethod
    def _rolling_autocorr(arr: np.ndarray, lag: int, window: int) -> np.ndarray:
        """Calculate rolling autocorrelation."""
        n = len(arr)
        result = np.full(n, np.nan)

        for i in range(window + lag, n):
            x = arr[i - window:i]
            y = arr[i - window - lag:i - lag]

            if not np.any(np.isnan(x)) and not np.any(np.isnan(y)):
                result[i] = np.corrcoef(x, y)[0, 1]

        return result


# ============================================================================
# SIGNAL FEATURES (Input to meta-learner)
# ============================================================================

class SignalFeaturesGenerator(VectorizedFeatureGenerator):
    """
    Generates signal-specific features for the meta-learner.

    These features describe signal quality, consistency, and context.
    They are used ONLY by the meta-model to determine if a signal
    from the primary model should be taken.
    """

    def __init__(self, config: Optional[LabelingFeatureConfig] = None):
        if config is None:
            config = LabelingFeatureConfig(
                name="signal_features",
                category=FeatureCategory.TECHNICAL_INDICATORS,
                description="Signal quality features for meta-learning",
                required_columns=["open", "high", "low", "close"],
                default_lookback=48
            )
        super().__init__(config)
        self.config = config

    def generate(self, df: pd.DataFrame, signals: Optional[pd.DataFrame] = None) -> FeatureResult:
        """
        Generate signal features.

        Args:
            df: Market data
            signals: Primary signals DataFrame (optional)
        """
        features = {}

        close = df['close'].values

        # ===== SIGNAL MAGNITUDE =====
        # If signals provided, use them; otherwise calculate
        if signals is not None and 'macd_hist' in signals.columns:
            features['signal_magnitude_macd'] = np.abs(signals['macd_hist'].values)
        else:
            _, _, hist = fast_macd(close, self.config.macd_fast, self.config.macd_slow, self.config.macd_signal)
            features['signal_magnitude_macd'] = np.abs(hist) / (close + 1e-8)

        if signals is not None and 'rsi' in signals.columns:
            # Distance from neutral (50)
            features['signal_magnitude_rsi'] = np.abs(signals['rsi'].values - 50) / 50
        else:
            rsi = fast_rsi(close, self.config.rsi_period)
            features['signal_magnitude_rsi'] = np.abs(rsi - 50) / 50

        # ===== SIGNAL PERSISTENCE =====
        # How long has RSI been below/above threshold?
        if signals is not None and 'rsi' in signals.columns:
            rsi_vals = signals['rsi'].values
        else:
            rsi_vals = fast_rsi(close, self.config.rsi_period)

        features['rsi_below_threshold_bars'] = self._count_consecutive(rsi_vals < self.config.rsi_oversold)
        features['rsi_above_threshold_bars'] = self._count_consecutive(rsi_vals > self.config.rsi_overbought)

        # How long has MACD histogram been positive/negative?
        if signals is not None and 'macd_hist' in signals.columns:
            macd_hist = signals['macd_hist'].values
        else:
            _, _, macd_hist = fast_macd(close, self.config.macd_fast, self.config.macd_slow, self.config.macd_signal)

        features['macd_positive_bars'] = self._count_consecutive(macd_hist > 0)
        features['macd_negative_bars'] = self._count_consecutive(macd_hist < 0)

        # ===== SIGNAL FREQUENCY =====
        # Number of signals in past N bars (high frequency → noisier)
        if signals is not None and 'consensus' in signals.columns:
            signal_binary = (signals['consensus'].values != 0).astype(float)
        else:
            # Placeholder: any RSI extreme or MACD cross
            signal_binary = ((rsi_vals < self.config.rsi_oversold) | (rsi_vals > self.config.rsi_overbought)).astype(float)

        features['signal_frequency_5bar'] = fast_rolling_sum(signal_binary, 5)
        features['signal_frequency_10bar'] = fast_rolling_sum(signal_binary, 10)

        # ===== EVENT CLUSTERING =====
        # Signals in clusters tend to be less stable
        features['signal_clustering'] = self._calculate_clustering(signal_binary, window=10)

        # ===== MARKET REGIME INDICATORS =====
        # Realized volatility (already calculated in market features, but repeated here for meta-model)
        returns = np.diff(np.log(close), prepend=np.nan)
        features['realized_vol_1h'] = fast_rolling_std(returns, 4)  # 4 x 15min bars
        features['realized_vol_4h'] = fast_rolling_std(returns, 16)
        features['realized_vol_1d'] = fast_rolling_std(returns, 96)

        # Recent drawdown % over N bars
        recent_high_5 = fast_rolling_max(close, 5)
        features['drawdown_5bar'] = (close - recent_high_5) / (recent_high_5 + 1e-8)

        recent_high_20 = fast_rolling_max(close, 20)
        features['drawdown_20bar'] = (close - recent_high_20) / (recent_high_20 + 1e-8)

        # Bullish/bearish trend labels (from long-term moving averages)
        ema_slow = pd.Series(close).ewm(span=self.config.long_window).mean().values
        features['bullish_trend'] = (close > ema_slow).astype(float)

        # Convert to DataFrame
        result_df = pd.DataFrame(features, index=df.index)

        return FeatureResult(
            success=True,
            features=result_df,
            feature_names=list(features.keys()),
            metadata={
                'n_features': len(features),
                'config': self.config.__dict__
            }
        )

    @staticmethod
    @njit
    def _count_consecutive(condition: np.ndarray) -> np.ndarray:
        """Count consecutive True values in a boolean array."""
        n = len(condition)
        count = np.zeros(n)

        for i in range(1, n):
            if condition[i]:
                count[i] = count[i - 1] + 1 if condition[i - 1] else 1
            else:
                count[i] = 0

        return count

    @staticmethod
    def _calculate_clustering(signal_binary: np.ndarray, window: int) -> np.ndarray:
        """
        Calculate signal clustering metric.

        High clustering = signals bunched together (less stable).
        Uses variance of signal positions within window.
        """
        n = len(signal_binary)
        clustering = np.full(n, np.nan)

        for i in range(window, n):
            window_signals = signal_binary[i - window:i]
            signal_indices = np.where(window_signals > 0)[0]

            if len(signal_indices) > 1:
                # Variance of positions (high = spread out, low = clustered)
                clustering[i] = 1.0 - (np.var(signal_indices) / (window ** 2 / 12))  # Normalize
            else:
                clustering[i] = 0.0

        return clustering


@njit
def fast_rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    """Numba-accelerated rolling sum."""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        result[i] = np.sum(arr[i - window + 1:i + 1])
    return result


# ============================================================================
# COMBINED LABELING FEATURE GENERATOR
# ============================================================================

class LabelingFeatureGenerator(VectorizedFeatureGenerator):
    """
    Combined generator for all labeling features (market + signal).

    This is the main entry point for generating features for meta-labeling.
    """

    def __init__(self, config: Optional[LabelingFeatureConfig] = None):
        if config is None:
            config = LabelingFeatureConfig(
                name="labeling_features",
                category=FeatureCategory.TECHNICAL_INDICATORS,
                description="Complete feature set for meta-labeling",
                required_columns=["open", "high", "low", "close", "volume"],
                default_lookback=96  # Ensure enough data for long-term features
            )
        super().__init__(config)
        self.config = config

        # Initialize sub-generators
        self.market_gen = MarketFeaturesGenerator(config)
        self.signal_gen = SignalFeaturesGenerator(config)

    def generate(self, df: pd.DataFrame, signals: Optional[pd.DataFrame] = None) -> FeatureResult:
        """
        Generate all labeling features.

        Args:
            df: Market data (OHLCV)
            signals: Optional primary signals for signal features

        Returns:
            FeatureResult with all market + signal features
        """
        # Generate market features
        market_result = self.market_gen.generate(df)
        if not market_result.success:
            return market_result

        # Generate signal features
        signal_result = self.signal_gen.generate(df, signals)
        if not signal_result.success:
            return signal_result

        # Combine features
        combined_features = pd.concat([
            market_result.features,
            signal_result.features
        ], axis=1)

        # Apply robust scaling if enabled
        if self.config.use_robust_scaling and NORMALIZATION_AVAILABLE:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()

            # Fit and transform
            scaled_values = scaler.fit_transform(combined_features.fillna(0))
            combined_features = pd.DataFrame(
                scaled_values,
                index=combined_features.index,
                columns=combined_features.columns
            )

        all_feature_names = list(combined_features.columns)

        return FeatureResult(
            success=True,
            features=combined_features,
            feature_names=all_feature_names,
            metadata={
                'n_market_features': len(market_result.feature_names),
                'n_signal_features': len(signal_result.feature_names),
                'n_total_features': len(all_feature_names),
                'market_features': market_result.feature_names,
                'signal_features': signal_result.feature_names,
                'config': self.config.__dict__
            }
        )
