"""
Trend-Aware Meta-Labeling with ZigZag and Triple Barrier Integration

This module provides comprehensive trend-aware meta-labeling capabilities including:
- Bollinger Bands signals (squeeze detection, price above upper band)
- OBV Divergence detection (price vs volume divergence)
- ZigZag trend detection and swing analysis
- Trend-aware triple barrier integration with dynamic barriers
- Categorical trend/zigzag labels for ML models

Key Features:
- Trend detection as filter/weighting factor for triple barrier
- Dynamic barrier adjustment based on trend strength
- Label enrichment with trend direction and zigzag swings
- ZigZag-based target smoothing for meaningful market structure

References:
- Trend vs Mean Reversion confluence model
- Triple barrier method (De Prado)
- ZigZag indicator for swing detection
"""

import warnings
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.core.decorators import handles_errors
from src.utils.logger import get_logger
from src.utils.comprehensive_function_logger import log_important_calls, log_all_calls

# Import numba for acceleration if available
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    numba = None

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from src.utils.vectorbt_compat import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max,
        rolling_sum, rolling_apply, rolling_corr, rolling_cov,
        scale, rank, zscore, winsorize, clip, quantile
    )
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = rolling_std = rolling_var = rolling_min = rolling_max = None
    rolling_sum = rolling_apply = rolling_corr = rolling_cov = None
    scale = rank = zscore = winsorize = clip = quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Trend direction enumeration."""
    UP = 1
    DOWN = -1
    SIDEWAYS = 0


class ZigZagSwing(Enum):
    """ZigZag swing type enumeration."""
    PEAK = 1
    TROUGH = -1
    NEUTRAL = 0


@dataclass
class BollingerBandsSignal:
    """Bollinger Bands signal detection result."""
    is_squeeze: bool = False
    squeeze_strength: float = 0.0
    price_above_upper: bool = False
    price_below_lower: bool = False
    bandwidth: float = 0.0
    bandwidth_percentile: float = 0.0
    position_in_bands: float = 0.5  # 0 = lower, 1 = upper


@dataclass
class OBVDivergence:
    """OBV Divergence detection result."""
    has_divergence: bool = False
    divergence_type: str = "none"  # "bullish", "bearish", "none"
    divergence_strength: float = 0.0
    price_trend: float = 0.0
    obv_trend: float = 0.0
    price_new_high: bool = False
    obv_fails_new_high: bool = False
    price_new_low: bool = False
    obv_fails_new_low: bool = False


@dataclass
class ZigZagResult:
    """ZigZag analysis result."""
    trend_direction: TrendDirection = TrendDirection.SIDEWAYS
    current_swing: ZigZagSwing = ZigZagSwing.NEUTRAL
    last_pivot_price: float = 0.0
    last_pivot_index: int = -1
    swing_magnitude: float = 0.0
    swing_slope: float = 0.0
    swing_count: int = 0
    pivots: List[Tuple[int, float, str]] = field(default_factory=list)  # (index, price, type)


@dataclass
class TrendAwareTripleBarrierConfig:
    """Configuration for trend-aware triple barrier labeling."""
    # Base triple barrier parameters
    base_profit_take_multiplier: float = 0.004
    base_stop_loss_multiplier: float = 0.003
    time_barrier_minutes: int = 30
    max_lookahead: int = 100
    transaction_cost: float = 0.0008
    binary_classification: bool = True
    
    # Trend adjustment parameters
    uptrend_pt_scale: float = 1.5  # Scale profit-take in uptrend
    uptrend_sl_scale: float = 0.8  # Loosen stop-loss in uptrend
    downtrend_pt_scale: float = 1.5  # Scale profit-take in downtrend
    downtrend_sl_scale: float = 0.8  # Loosen stop-loss in downtrend
    sideways_pt_scale: float = 0.7  # Tighten in sideways
    sideways_sl_scale: float = 1.2  # Tighten in sideways
    
    # Signal weighting
    uptrend_long_weight: float = 1.5
    uptrend_short_weight: float = 0.5
    downtrend_long_weight: float = 0.5
    downtrend_short_weight: float = 1.5
    sideways_signal_weight: float = 0.7  # Down-weight signals in sideways
    
    # Confluence thresholds
    min_trend_strength: float = 0.3
    min_divergence_strength: float = 0.5
    squeeze_lookback: int = 20
    divergence_lookback: int = 20
    
    # ZigZag parameters
    zigzag_pct_threshold: float = 0.03  # 3% price movement for pivot
    zigzag_atr_multiplier: float = 2.0  # Alternative: use ATR-based threshold
    use_atr_for_zigzag: bool = True
    
    # Feature generation
    include_trend_features: bool = True
    include_zigzag_features: bool = True
    include_confluence_features: bool = True


class TrendAwareMetaLabeler:
    """
    Comprehensive trend-aware meta-labeling system.
    
    Integrates:
    - Bollinger Bands signals (squeeze, breakout)
    - OBV Divergence detection
    - ZigZag trend analysis
    - Trend-aware triple barrier method
    """
    
    @log_important_calls
    def __init__(self, config: Optional[TrendAwareTripleBarrierConfig] = None) -> None:
        """Initialize the trend-aware meta-labeler.
        
        Args:
            config: Configuration for trend-aware labeling
        """
        self.config = config or TrendAwareTripleBarrierConfig()
        self.logger = get_logger('TrendAwareMetaLabeler')
        
        # Validate configuration
        self._validate_config()
        
        self.logger.info("🎯 TrendAwareMetaLabeler initialized")
        self.logger.info(f"   Base PT: {self.config.base_profit_take_multiplier:.4f}")
        self.logger.info(f"   Base SL: {self.config.base_stop_loss_multiplier:.4f}")
        self.logger.info(f"   ZigZag threshold: {self.config.zigzag_pct_threshold:.2%}")
        
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config.base_profit_take_multiplier <= 0:
            raise ValueError("base_profit_take_multiplier must be positive")
        if self.config.base_stop_loss_multiplier <= 0:
            raise ValueError("base_stop_loss_multiplier must be positive")
        if self.config.zigzag_pct_threshold <= 0:
            raise ValueError("zigzag_pct_threshold must be positive")
            
    # =========================================================================
    # BOLLINGER BANDS SIGNAL DETECTION
    # =========================================================================
    
    @log_all_calls
    def calculate_bollinger_bands(
        self,
        close: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands.
        
        Args:
            close: Close prices
            period: SMA period (default 20)
            std_dev: Standard deviation multiplier (default 2)
            
        Returns:
            Tuple of (middle, upper, lower, bandwidth)
        """
        middle = close.rolling(window=period, min_periods=1).mean()
        std = close.rolling(window=period, min_periods=1).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        bandwidth = upper - lower
        
        return middle, upper, lower, bandwidth
    
    @log_all_calls
    def detect_bollinger_signals(
        self,
        data: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0
    ) -> pd.DataFrame:
        """Detect Bollinger Bands signals including squeeze and breakout.
        
        Condition A: Bandwidth is at a 20-period Low (The Squeeze)
        Condition B: Price closes Above the Upper Band
        
        Args:
            data: DataFrame with 'close' column
            period: Bollinger period
            std_dev: Standard deviation multiplier
            
        Returns:
            DataFrame with Bollinger signals
        """
        close = data['close']
        n = len(close)
        
        middle, upper, lower, bandwidth = self.calculate_bollinger_bands(
            close, period, std_dev
        )
        
        # Calculate bandwidth percentile (for squeeze detection)
        squeeze_lookback = self.config.squeeze_lookback
        bandwidth_min = bandwidth.rolling(window=squeeze_lookback, min_periods=1).min()
        
        # Condition A: Squeeze - bandwidth at lookback period low
        is_squeeze = bandwidth <= bandwidth_min * 1.01  # Within 1% of minimum
        
        # Calculate squeeze strength (how tight is the squeeze)
        bandwidth_mean = bandwidth.rolling(window=squeeze_lookback, min_periods=1).mean()
        squeeze_strength = np.where(
            bandwidth_mean > 0,
            1 - (bandwidth / bandwidth_mean),
            0
        )
        squeeze_strength = np.clip(squeeze_strength, 0, 1)
        
        # Condition B: Price closes above upper band
        price_above_upper = close > upper
        
        # Additional signals
        price_below_lower = close < lower
        
        # Position within bands (0 = at lower, 1 = at upper)
        band_range = upper - lower
        position_in_bands = np.where(
            band_range > 0,
            (close - lower) / band_range,
            0.5
        )
        position_in_bands = np.clip(position_in_bands, 0, 1)
        
        # Bandwidth percentile
        bandwidth_percentile = bandwidth.rolling(window=squeeze_lookback, min_periods=1).apply(
            lambda x: (x[-1:] <= x).mean() if len(x) > 0 else 0.5,
            raw=False
        )
        
        # Create result DataFrame
        result = pd.DataFrame(index=data.index)
        result['bb_middle'] = middle
        result['bb_upper'] = upper
        result['bb_lower'] = lower
        result['bb_bandwidth'] = bandwidth
        result['bb_is_squeeze'] = is_squeeze.astype(int)
        result['bb_squeeze_strength'] = squeeze_strength
        result['bb_price_above_upper'] = price_above_upper.astype(int)
        result['bb_price_below_lower'] = price_below_lower.astype(int)
        result['bb_position'] = position_in_bands
        result['bb_bandwidth_percentile'] = bandwidth_percentile
        
        # Confluence signals
        # Squeeze breakout: was in squeeze and now breaking above upper
        is_squeeze_shifted = is_squeeze.shift(1).fillna(False)
        result['bb_squeeze_breakout_up'] = (is_squeeze_shifted & price_above_upper).astype(int)
        result['bb_squeeze_breakout_down'] = (is_squeeze_shifted & price_below_lower).astype(int)
        
        self.logger.info(f"📊 Bollinger signals detected:")
        self.logger.info(f"   Squeeze periods: {is_squeeze.sum()}/{n} ({is_squeeze.mean()*100:.1f}%)")
        self.logger.info(f"   Price above upper: {price_above_upper.sum()}/{n}")
        self.logger.info(f"   Squeeze breakouts up: {result['bb_squeeze_breakout_up'].sum()}")
        
        return result
    
    # =========================================================================
    # OBV DIVERGENCE DETECTION
    # =========================================================================
    
    @log_all_calls
    def calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume (OBV).
        
        Args:
            close: Close prices
            volume: Volume data
            
        Returns:
            OBV series
        """
        price_change = close.diff()
        direction = np.sign(price_change)
        obv = (direction * volume).cumsum()
        return obv
    
    @log_all_calls
    def detect_obv_divergence(
        self,
        data: pd.DataFrame,
        lookback: int = 20
    ) -> pd.DataFrame:
        """Detect OBV Divergence signals.
        
        Condition A: Price makes a New 20-Day High
        Condition B: OBV fails to make a New 20-Day High (bearish divergence)
        
        Also detects bullish divergence:
        Condition A: Price makes a New 20-Day Low  
        Condition B: OBV fails to make a New 20-Day Low
        
        Args:
            data: DataFrame with 'close' and 'volume' columns
            lookback: Lookback period for new highs/lows
            
        Returns:
            DataFrame with OBV divergence signals
        """
        close = data['close']
        volume = data['volume'] if 'volume' in data.columns else pd.Series(1.0, index=data.index)
        
        obv = self.calculate_obv(close, volume)
        
        # Rolling max/min for price and OBV
        price_max = close.rolling(window=lookback, min_periods=1).max()
        price_min = close.rolling(window=lookback, min_periods=1).min()
        obv_max = obv.rolling(window=lookback, min_periods=1).max()
        obv_min = obv.rolling(window=lookback, min_periods=1).min()
        
        # Detect new highs/lows
        price_new_high = close >= price_max
        price_new_low = close <= price_min
        obv_new_high = obv >= obv_max
        obv_new_low = obv <= obv_min
        
        # Bearish divergence: price new high but OBV fails
        # Need to look at previous price peak vs current
        obv_fails_new_high = price_new_high & ~obv_new_high
        
        # Bullish divergence: price new low but OBV fails
        obv_fails_new_low = price_new_low & ~obv_new_low
        
        # Calculate divergence strength
        # Normalize price and OBV changes for comparison
        price_pct_from_high = (close - price_max) / price_max
        obv_pct_from_high = np.where(
            obv_max != 0,
            (obv - obv_max) / np.abs(obv_max),
            0
        )
        
        price_pct_from_low = np.where(
            price_min != 0,
            (close - price_min) / price_min,
            0
        )
        obv_pct_from_low = np.where(
            obv_min != 0,
            (obv - obv_min) / np.abs(obv_min),
            0
        )
        
        # Bearish divergence strength: price is high but OBV is weak
        bearish_div_strength = np.where(
            obv_fails_new_high,
            np.abs(obv_pct_from_high),  # How much OBV is lagging
            0
        )
        
        # Bullish divergence strength: price is low but OBV is strong
        bullish_div_strength = np.where(
            obv_fails_new_low,
            np.abs(obv_pct_from_low),
            0
        )
        
        # Combined divergence signal
        has_bearish_divergence = (obv_fails_new_high) & (bearish_div_strength > 0.05)
        has_bullish_divergence = (obv_fails_new_low) & (bullish_div_strength > 0.05)
        
        # Create result DataFrame
        result = pd.DataFrame(index=data.index)
        result['obv'] = obv
        result['obv_normalized'] = (obv - obv.rolling(lookback).mean()) / obv.rolling(lookback).std().replace(0, 1)
        result['price_new_high_20'] = price_new_high.astype(int)
        result['price_new_low_20'] = price_new_low.astype(int)
        result['obv_new_high_20'] = obv_new_high.astype(int)
        result['obv_new_low_20'] = obv_new_low.astype(int)
        result['obv_fails_new_high'] = obv_fails_new_high.astype(int)
        result['obv_fails_new_low'] = obv_fails_new_low.astype(int)
        result['obv_bearish_divergence'] = has_bearish_divergence.astype(int)
        result['obv_bullish_divergence'] = has_bullish_divergence.astype(int)
        result['obv_bearish_div_strength'] = bearish_div_strength
        result['obv_bullish_div_strength'] = bullish_div_strength
        
        # Combined divergence type (-1 = bearish, 0 = none, 1 = bullish)
        result['obv_divergence_type'] = np.where(
            has_bullish_divergence, 1,
            np.where(has_bearish_divergence, -1, 0)
        )
        
        self.logger.info(f"📊 OBV Divergence signals detected:")
        self.logger.info(f"   Bearish divergences: {has_bearish_divergence.sum()}")
        self.logger.info(f"   Bullish divergences: {has_bullish_divergence.sum()}")
        
        return result
    
    # =========================================================================
    # ZIGZAG TREND DETECTION
    # =========================================================================
    
    @log_all_calls
    def calculate_atr(
        self,
        data: pd.DataFrame,
        period: int = 14
    ) -> pd.Series:
        """Calculate Average True Range.
        
        Args:
            data: DataFrame with high, low, close columns
            period: ATR period
            
        Returns:
            ATR series
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
        
        return atr
    
    def _find_zigzag_pivots_numba(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        threshold: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find ZigZag pivots using threshold-based detection.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            threshold: Dynamic threshold array (e.g., ATR-based)
            
        Returns:
            Tuple of (pivot_types, pivot_prices, pivot_indices)
            pivot_types: 1 = peak, -1 = trough, 0 = no pivot
        """
        n = len(close)
        pivot_types = np.zeros(n, dtype=np.int8)
        pivot_prices = np.zeros(n, dtype=np.float64)
        pivot_indices = np.zeros(n, dtype=np.int64)
        
        if n < 3:
            return pivot_types, pivot_prices, pivot_indices
        
        # Initialize with first point
        last_pivot_type = 0  # 1 = looking for trough (after peak), -1 = looking for peak
        last_pivot_idx = 0
        last_pivot_price = close[0]
        
        # Find initial direction
        if high[1] > high[0] and low[1] >= low[0]:
            last_pivot_type = 1  # First pivot is peak-seeking
            last_pivot_price = low[0]
        elif low[1] < low[0] and high[1] <= high[0]:
            last_pivot_type = -1  # First pivot is trough-seeking
            last_pivot_price = high[0]
        
        pivot_count = 0
        
        for i in range(1, n):
            current_threshold = threshold[i] if i < len(threshold) else threshold[-1]
            
            if last_pivot_type == 1:  # Looking for peak
                if high[i] > last_pivot_price + current_threshold:
                    # New potential peak
                    last_pivot_price = high[i]
                    last_pivot_idx = i
                elif low[i] < last_pivot_price - current_threshold:
                    # Found peak, now looking for trough
                    pivot_types[last_pivot_idx] = 1  # Peak
                    pivot_prices[last_pivot_idx] = last_pivot_price
                    pivot_indices[pivot_count] = last_pivot_idx
                    pivot_count += 1
                    
                    last_pivot_type = -1  # Now looking for trough
                    last_pivot_price = low[i]
                    last_pivot_idx = i
                    
            elif last_pivot_type == -1:  # Looking for trough
                if low[i] < last_pivot_price - current_threshold:
                    # New potential trough
                    last_pivot_price = low[i]
                    last_pivot_idx = i
                elif high[i] > last_pivot_price + current_threshold:
                    # Found trough, now looking for peak
                    pivot_types[last_pivot_idx] = -1  # Trough
                    pivot_prices[last_pivot_idx] = last_pivot_price
                    pivot_indices[pivot_count] = last_pivot_idx
                    pivot_count += 1
                    
                    last_pivot_type = 1  # Now looking for peak
                    last_pivot_price = high[i]
                    last_pivot_idx = i
            else:
                # Initialize direction
                if high[i] > close[i-1] + current_threshold:
                    last_pivot_type = 1
                    last_pivot_price = high[i]
                    last_pivot_idx = i
                elif low[i] < close[i-1] - current_threshold:
                    last_pivot_type = -1
                    last_pivot_price = low[i]
                    last_pivot_idx = i
        
        return pivot_types, pivot_prices, pivot_indices[:pivot_count]
    
    @log_all_calls
    def detect_zigzag_trend(
        self,
        data: pd.DataFrame,
        use_atr: bool = True,
        pct_threshold: float = 0.03,
        atr_multiplier: float = 2.0
    ) -> pd.DataFrame:
        """Detect ZigZag pivots and trend direction.
        
        Args:
            data: DataFrame with OHLC data
            use_atr: If True, use ATR-based threshold; else use percentage
            pct_threshold: Percentage threshold for price movement
            atr_multiplier: ATR multiplier for threshold
            
        Returns:
            DataFrame with ZigZag trend features
        """
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        n = len(close)
        
        # Calculate threshold
        if use_atr and 'high' in data.columns and 'low' in data.columns:
            atr = self.calculate_atr(data, period=14).values
            threshold = atr * atr_multiplier
        else:
            threshold = close * pct_threshold
        
        # Find ZigZag pivots
        pivot_types, pivot_prices, pivot_indices = self._find_zigzag_pivots_numba(
            high, low, close, threshold
        )
        
        # Create result arrays
        trend_direction = np.zeros(n, dtype=np.int8)  # 1=up, -1=down, 0=sideways
        current_swing_type = np.zeros(n, dtype=np.int8)  # 1=peak, -1=trough, 0=neutral
        swing_magnitude = np.zeros(n, dtype=np.float64)
        swing_slope = np.zeros(n, dtype=np.float64)
        last_pivot_idx = np.zeros(n, dtype=np.int64)
        last_pivot_price = np.zeros(n, dtype=np.float64)
        bars_since_pivot = np.zeros(n, dtype=np.int64)
        
        # Initialize
        last_pivot_price[:] = close[0]
        
        # Process pivots to determine trend
        if len(pivot_indices) >= 2:
            for i in range(1, len(pivot_indices)):
                curr_idx = int(pivot_indices[i])
                prev_idx = int(pivot_indices[i-1])
                
                if curr_idx >= n or prev_idx >= n:
                    continue
                    
                curr_price = pivot_prices[curr_idx]
                prev_price = pivot_prices[prev_idx]
                curr_type = pivot_types[curr_idx]
                
                # Determine trend direction based on higher highs/lower lows
                if i >= 2:
                    prev_prev_idx = int(pivot_indices[i-2])
                    if prev_prev_idx < n:
                        prev_prev_price = pivot_prices[prev_prev_idx]
                        
                        # Higher high and higher low = uptrend
                        if curr_type == 1 and curr_price > prev_prev_price:  # Higher peak
                            trend_direction[curr_idx:] = 1
                        elif curr_type == -1 and curr_price < prev_prev_price:  # Lower trough
                            trend_direction[curr_idx:] = -1
                        else:
                            # Sideways if no clear higher/lower pattern
                            trend_direction[curr_idx:] = 0
                
                # Set swing information
                current_swing_type[curr_idx:] = curr_type
                last_pivot_idx[curr_idx:] = curr_idx
                last_pivot_price[curr_idx:] = curr_price
                
                # Calculate swing magnitude and slope
                price_diff = curr_price - prev_price
                bar_diff = max(curr_idx - prev_idx, 1)
                swing_magnitude[curr_idx:] = abs(price_diff) / max(prev_price, 1e-8)
                swing_slope[curr_idx:] = price_diff / (bar_diff * max(prev_price, 1e-8))
        
        # Calculate bars since last pivot
        for i in range(n):
            if last_pivot_idx[i] > 0:
                bars_since_pivot[i] = i - last_pivot_idx[i]
        
        # Determine if currently at or near a pivot
        near_pivot = np.zeros(n, dtype=np.int8)
        for idx in pivot_indices:
            if idx < n:
                # Mark 3 bars around pivot
                start = max(0, int(idx) - 1)
                end = min(n, int(idx) + 2)
                near_pivot[start:end] = pivot_types[int(idx)]
        
        # Create result DataFrame
        result = pd.DataFrame(index=data.index)
        result['zigzag_pivot_type'] = pivot_types
        result['zigzag_trend_direction'] = trend_direction
        result['zigzag_current_swing'] = current_swing_type
        result['zigzag_swing_magnitude'] = swing_magnitude
        result['zigzag_swing_slope'] = swing_slope
        result['zigzag_last_pivot_idx'] = last_pivot_idx
        result['zigzag_last_pivot_price'] = last_pivot_price
        result['zigzag_bars_since_pivot'] = bars_since_pivot
        result['zigzag_near_pivot'] = near_pivot
        
        # Add categorical trend labels
        result['trend_direction_cat'] = pd.Categorical(
            np.where(trend_direction == 1, 'up',
                     np.where(trend_direction == -1, 'down', 'sideways')),
            categories=['up', 'down', 'sideways']
        )
        result['zigzag_swing_cat'] = pd.Categorical(
            np.where(current_swing_type == 1, 'peak',
                     np.where(current_swing_type == -1, 'trough', 'neutral')),
            categories=['peak', 'trough', 'neutral']
        )
        
        # Calculate trend strength
        result['trend_strength'] = np.abs(swing_slope) * np.sqrt(swing_magnitude + 1e-8)
        
        pivot_count = len(pivot_indices)
        self.logger.info(f"📊 ZigZag analysis complete:")
        self.logger.info(f"   Total pivots: {pivot_count}")
        self.logger.info(f"   Uptrend periods: {(trend_direction == 1).sum()}/{n}")
        self.logger.info(f"   Downtrend periods: {(trend_direction == -1).sum()}/{n}")
        self.logger.info(f"   Sideways periods: {(trend_direction == 0).sum()}/{n}")
        
        return result
    
    # =========================================================================
    # TREND-AWARE TRIPLE BARRIER LABELING
    # =========================================================================
    
    @log_all_calls
    def calculate_dynamic_barriers(
        self,
        data: pd.DataFrame,
        trend_direction: np.ndarray,
        trend_strength: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate dynamic profit-take and stop-loss barriers based on trend.
        
        If uptrend is strong: Increase profit-taking distance, loosen stop-loss
        If downtrend is strong: Similar for shorts
        If sideways: Tighten both barriers
        
        Args:
            data: Market data
            trend_direction: Array of trend directions (1, -1, 0)
            trend_strength: Array of trend strength values
            
        Returns:
            Tuple of (pt_multipliers, sl_multipliers)
        """
        n = len(data)
        base_pt = self.config.base_profit_take_multiplier
        base_sl = self.config.base_stop_loss_multiplier
        
        pt_multipliers = np.full(n, base_pt, dtype=np.float64)
        sl_multipliers = np.full(n, base_sl, dtype=np.float64)
        
        # Normalize trend strength to 0-1 range
        strength_normalized = np.clip(trend_strength / (trend_strength.max() + 1e-8), 0, 1)
        
        for i in range(n):
            direction = trend_direction[i]
            strength = strength_normalized[i]
            
            if direction == 1:  # Uptrend
                # Scale barriers by trend strength
                pt_scale = 1 + (self.config.uptrend_pt_scale - 1) * strength
                sl_scale = 1 + (self.config.uptrend_sl_scale - 1) * strength
                pt_multipliers[i] = base_pt * pt_scale
                sl_multipliers[i] = base_sl * sl_scale
                
            elif direction == -1:  # Downtrend
                pt_scale = 1 + (self.config.downtrend_pt_scale - 1) * strength
                sl_scale = 1 + (self.config.downtrend_sl_scale - 1) * strength
                pt_multipliers[i] = base_pt * pt_scale
                sl_multipliers[i] = base_sl * sl_scale
                
            else:  # Sideways
                pt_multipliers[i] = base_pt * self.config.sideways_pt_scale
                sl_multipliers[i] = base_sl * self.config.sideways_sl_scale
        
        return pt_multipliers, sl_multipliers
    
    @log_all_calls
    def calculate_signal_weights(
        self,
        trend_direction: np.ndarray,
        trend_strength: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        """Calculate signal weights based on trend alignment.
        
        - Up-weight long signals in uptrend
        - Up-weight short signals in downtrend
        - Down-weight signals in sideways market
        
        Args:
            trend_direction: Array of trend directions
            trend_strength: Array of trend strength values
            labels: Array of signal labels (1 = long, -1 = short)
            
        Returns:
            Array of signal weights
        """
        n = len(labels)
        weights = np.ones(n, dtype=np.float64)
        
        # Normalize trend strength
        strength_normalized = np.clip(trend_strength / (trend_strength.max() + 1e-8), 0, 1)
        
        for i in range(n):
            direction = trend_direction[i]
            strength = strength_normalized[i]
            label = labels[i]
            
            if direction == 1:  # Uptrend
                if label == 1:  # Long signal in uptrend
                    weights[i] = 1 + (self.config.uptrend_long_weight - 1) * strength
                elif label == -1:  # Short signal in uptrend (contrarian)
                    weights[i] = 1 + (self.config.uptrend_short_weight - 1) * strength
                    
            elif direction == -1:  # Downtrend
                if label == 1:  # Long signal in downtrend (contrarian)
                    weights[i] = 1 + (self.config.downtrend_long_weight - 1) * strength
                elif label == -1:  # Short signal in downtrend
                    weights[i] = 1 + (self.config.downtrend_short_weight - 1) * strength
                    
            else:  # Sideways
                weights[i] = self.config.sideways_signal_weight
        
        return weights
    
    @handles_errors(
        exceptions=(Exception,),
        default_return=pd.DataFrame(),
        context="trend_aware_triple_barrier_labeling"
    )
    @log_important_calls
    def apply_trend_aware_triple_barrier(
        self,
        data: pd.DataFrame,
        zigzag_features: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Apply trend-aware triple barrier labeling.
        
        Integrates:
        - ZigZag trend detection
        - Dynamic barrier adjustment
        - Signal weighting
        - Optional ZigZag-based target smoothing
        
        Args:
            data: DataFrame with OHLC data
            zigzag_features: Optional pre-computed ZigZag features
            
        Returns:
            DataFrame with trend-aware labels
        """
        self.logger.info("🎯 Applying trend-aware triple barrier labeling")
        
        # Validate input
        required_cols = ['close', 'high', 'low']
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        n = len(data)
        if n < 2:
            labeled_data = data.copy()
            labeled_data['label'] = 0
            labeled_data['signal_weight'] = 1.0
            return labeled_data
        
        # Get ZigZag features
        if zigzag_features is None:
            zigzag_features = self.detect_zigzag_trend(
                data,
                use_atr=self.config.use_atr_for_zigzag,
                pct_threshold=self.config.zigzag_pct_threshold,
                atr_multiplier=self.config.zigzag_atr_multiplier
            )
        
        trend_direction = zigzag_features['zigzag_trend_direction'].values
        trend_strength = zigzag_features['trend_strength'].values
        
        # Calculate dynamic barriers
        pt_multipliers, sl_multipliers = self.calculate_dynamic_barriers(
            data, trend_direction, trend_strength
        )
        
        # Prepare data for labeling
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        idx = data.index
        
        # Calculate time barrier
        use_time_barrier = isinstance(idx, pd.DatetimeIndex)
        arange_n = np.arange(n, dtype=np.int64)
        end_by_lookahead = np.minimum(arange_n + 1 + self.config.max_lookahead, n)
        
        if use_time_barrier:
            try:
                idx_ns = idx.view(np.int64)
                delta_ns = np.int64(self.config.time_barrier_minutes) * np.int64(60_000_000_000)
                end_times = idx_ns + delta_ns
                end_by_time = np.searchsorted(idx_ns, end_times, side='right')
            except Exception:
                end_by_time = end_by_lookahead
        else:
            end_by_time = end_by_lookahead
        
        end_idx_arr = np.minimum(end_by_lookahead, end_by_time).astype(np.int64)
        
        # Apply triple barrier labeling with dynamic barriers
        labels = np.zeros(n, dtype=np.int8)
        profit_pcts = np.zeros(n, dtype=np.float64)
        barrier_hit_type = np.zeros(n, dtype=np.int8)  # 1=PT, -1=SL, 0=time
        
        for i in range(n - 1):
            entry_price = close[i]
            pt_mult = pt_multipliers[i]
            sl_mult = sl_multipliers[i]
            
            profit_barrier = entry_price * (1.0 + pt_mult)
            stop_barrier = entry_price * (1.0 - sl_mult)
            end_idx = int(end_idx_arr[i])
            
            if end_idx <= i + 1:
                labels[i] = 0
                profit_pcts[i] = 0.0
                barrier_hit_type[i] = 0
                continue
            
            win_high = high[i + 1:end_idx]
            win_low = low[i + 1:end_idx]
            
            profit_hits = np.where(win_high >= profit_barrier)[0]
            stop_hits = np.where(win_low <= stop_barrier)[0]
            
            if profit_hits.size == 0 and stop_hits.size == 0:
                labels[i] = 0
                profit_pcts[i] = 0.0
                barrier_hit_type[i] = 0
            elif profit_hits.size == 0:
                labels[i] = -1
                profit_pcts[i] = -(sl_mult + self.config.transaction_cost)
                barrier_hit_type[i] = -1
            elif stop_hits.size == 0:
                labels[i] = 1
                profit_pcts[i] = pt_mult - self.config.transaction_cost
                barrier_hit_type[i] = 1
            elif profit_hits[0] <= stop_hits[0]:
                labels[i] = 1
                profit_pcts[i] = pt_mult - self.config.transaction_cost
                barrier_hit_type[i] = 1
            else:
                labels[i] = -1
                profit_pcts[i] = -(sl_mult + self.config.transaction_cost)
                barrier_hit_type[i] = -1
        
        # Calculate signal weights
        signal_weights = self.calculate_signal_weights(trend_direction, trend_strength, labels)
        
        # Create labeled DataFrame
        labeled_data = data.copy()
        labeled_data['label'] = labels
        labeled_data['potential_profit_pct'] = profit_pcts
        labeled_data['barrier_hit_type'] = barrier_hit_type
        labeled_data['pt_multiplier_used'] = pt_multipliers
        labeled_data['sl_multiplier_used'] = sl_multipliers
        labeled_data['signal_weight'] = signal_weights
        
        # Add trend features
        for col in zigzag_features.columns:
            if col not in labeled_data.columns:
                labeled_data[col] = zigzag_features[col].values
        
        # Filter HOLD samples if binary classification
        if self.config.binary_classification:
            original_count = len(labeled_data)
            hold_samples = (labeled_data['label'] == 0).sum()
            labeled_data = labeled_data[labeled_data['label'] != 0].copy()
            
            self.logger.info(f"📊 Label distribution after filtering:")
            self.logger.info(f"   LONG (1): {(labeled_data['label'] == 1).sum()}")
            self.logger.info(f"   SHORT (-1): {(labeled_data['label'] == -1).sum()}")
            self.logger.info(f"   HOLD removed: {hold_samples}")
            self.logger.info(f"   Retained: {len(labeled_data)}/{original_count}")
        
        return labeled_data
    
    # =========================================================================
    # COMPREHENSIVE FEATURE GENERATION
    # =========================================================================
    
    @handles_errors(
        exceptions=(Exception,),
        default_return=pd.DataFrame(),
        context="generate_trend_aware_features"
    )
    @log_important_calls
    def generate_trend_aware_features(
        self,
        data: pd.DataFrame,
        include_labels: bool = True
    ) -> pd.DataFrame:
        """Generate comprehensive trend-aware features for meta-labeling.
        
        Includes:
        - Bollinger Bands signals (squeeze, breakout)
        - OBV Divergence detection
        - ZigZag trend analysis
        - Trend-aware labels (if requested)
        
        Args:
            data: DataFrame with OHLCV data
            include_labels: Whether to include triple barrier labels
            
        Returns:
            DataFrame with all trend-aware features
        """
        self.logger.info("🔧 Generating comprehensive trend-aware features")
        
        result = data.copy()
        
        # 1. Bollinger Bands signals
        self.logger.info("   Computing Bollinger Bands signals...")
        bb_features = self.detect_bollinger_signals(data)
        for col in bb_features.columns:
            result[col] = bb_features[col].values
        
        # 2. OBV Divergence
        if 'volume' in data.columns:
            self.logger.info("   Computing OBV Divergence signals...")
            obv_features = self.detect_obv_divergence(data)
            for col in obv_features.columns:
                result[col] = obv_features[col].values
        else:
            self.logger.warning("   Skipping OBV (no volume data)")
        
        # 3. ZigZag Trend
        self.logger.info("   Computing ZigZag trend analysis...")
        zigzag_features = self.detect_zigzag_trend(
            data,
            use_atr=self.config.use_atr_for_zigzag,
            pct_threshold=self.config.zigzag_pct_threshold,
            atr_multiplier=self.config.zigzag_atr_multiplier
        )
        for col in zigzag_features.columns:
            if col not in result.columns:
                result[col] = zigzag_features[col].values
        
        # 4. Confluence features
        self.logger.info("   Computing confluence signals...")
        result = self._add_confluence_features(result, bb_features, zigzag_features)
        
        # 5. Triple barrier labels with trend awareness
        if include_labels:
            self.logger.info("   Computing trend-aware triple barrier labels...")
            labeled_result = self.apply_trend_aware_triple_barrier(data, zigzag_features)
            
            # Add labeling columns
            label_cols = ['label', 'potential_profit_pct', 'barrier_hit_type',
                         'pt_multiplier_used', 'sl_multiplier_used', 'signal_weight']
            for col in label_cols:
                if col in labeled_result.columns:
                    # Handle different lengths (due to HOLD filtering)
                    if len(labeled_result) == len(result):
                        result[col] = labeled_result[col].values
                    else:
                        # Merge on index
                        result = result.loc[labeled_result.index].copy()
                        result[col] = labeled_result[col].values
        
        self.logger.info(f"✅ Generated {len(result.columns)} features for {len(result)} samples")
        
        return result
    
    def _add_confluence_features(
        self,
        result: pd.DataFrame,
        bb_features: pd.DataFrame,
        zigzag_features: pd.DataFrame
    ) -> pd.DataFrame:
        """Add confluence features combining multiple signals.
        
        Args:
            result: Result DataFrame
            bb_features: Bollinger Bands features
            zigzag_features: ZigZag features
            
        Returns:
            DataFrame with confluence features added
        """
        # Trend-confirming signals
        trend_up = zigzag_features['zigzag_trend_direction'].values == 1
        trend_down = zigzag_features['zigzag_trend_direction'].values == -1
        trend_sideways = zigzag_features['zigzag_trend_direction'].values == 0
        
        bb_breakout_up = bb_features['bb_price_above_upper'].values == 1
        bb_breakout_down = bb_features['bb_price_below_lower'].values == 1
        bb_squeeze = bb_features['bb_is_squeeze'].values == 1
        
        # Confluence: Trend + Breakout alignment
        result['confluence_uptrend_breakout'] = (trend_up & bb_breakout_up).astype(int)
        result['confluence_downtrend_breakdown'] = (trend_down & bb_breakout_down).astype(int)
        
        # Squeeze in trend context (potential breakout setup)
        result['confluence_squeeze_uptrend'] = (bb_squeeze & trend_up).astype(int)
        result['confluence_squeeze_downtrend'] = (bb_squeeze & trend_down).astype(int)
        result['confluence_squeeze_sideways'] = (bb_squeeze & trend_sideways).astype(int)
        
        # OBV divergence in trend context (if available)
        if 'obv_divergence_type' in result.columns:
            obv_bullish_div = result['obv_divergence_type'].values == 1
            obv_bearish_div = result['obv_divergence_type'].values == -1
            
            # Divergence alignment with trend (contrarian signals)
            result['confluence_bullish_div_downtrend'] = (obv_bullish_div & trend_down).astype(int)
            result['confluence_bearish_div_uptrend'] = (obv_bearish_div & trend_up).astype(int)
            
            # Divergence confirmation (trend exhaustion signals)
            result['trend_exhaustion_signal'] = (
                (obv_bearish_div & trend_up) | (obv_bullish_div & trend_down)
            ).astype(int)
        
        # Combined signal strength
        trend_strength = zigzag_features['trend_strength'].values
        bb_squeeze_strength = bb_features['bb_squeeze_strength'].values
        
        result['combined_signal_strength'] = (
            trend_strength * 0.5 + bb_squeeze_strength * 0.5
        )
        
        return result
    
    # =========================================================================
    # ZIGZAG-BASED TARGET SMOOTHING
    # =========================================================================
    
    @log_all_calls
    def calculate_zigzag_based_barriers(
        self,
        data: pd.DataFrame,
        zigzag_features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate barriers relative to ZigZag pivots.
        
        For longs:
        - Profit-taking = next ZigZag swing high
        - Stop-loss = previous ZigZag swing low
        
        For shorts:
        - Profit-taking = next ZigZag swing low
        - Stop-loss = previous ZigZag swing high
        
        Args:
            data: OHLC data
            zigzag_features: Pre-computed ZigZag features
            
        Returns:
            DataFrame with ZigZag-based barrier levels
        """
        n = len(data)
        close = data['close'].values
        pivot_types = zigzag_features['zigzag_pivot_type'].values
        
        # Find all pivot prices and indices
        peak_indices = np.where(pivot_types == 1)[0]
        trough_indices = np.where(pivot_types == -1)[0]
        
        # Initialize barrier levels
        next_peak_price = np.full(n, np.nan)
        next_trough_price = np.full(n, np.nan)
        prev_peak_price = np.full(n, np.nan)
        prev_trough_price = np.full(n, np.nan)
        
        # Forward fill from peaks
        if len(peak_indices) > 0:
            for i, peak_idx in enumerate(peak_indices):
                # Previous peak for indices after this peak
                if peak_idx < n:
                    end_idx = peak_indices[i + 1] if i + 1 < len(peak_indices) else n
                    prev_peak_price[peak_idx:end_idx] = close[peak_idx]
                
            # Next peak (look forward)
            for i in range(n):
                future_peaks = peak_indices[peak_indices > i]
                if len(future_peaks) > 0:
                    next_peak_price[i] = close[future_peaks[0]]
        
        # Forward fill from troughs
        if len(trough_indices) > 0:
            for i, trough_idx in enumerate(trough_indices):
                if trough_idx < n:
                    end_idx = trough_indices[i + 1] if i + 1 < len(trough_indices) else n
                    prev_trough_price[trough_idx:end_idx] = close[trough_idx]
                    
            # Next trough (look forward)
            for i in range(n):
                future_troughs = trough_indices[trough_indices > i]
                if len(future_troughs) > 0:
                    next_trough_price[i] = close[future_troughs[0]]
        
        # Calculate barrier distances as percentages
        result = pd.DataFrame(index=data.index)
        result['zigzag_long_pt_level'] = next_peak_price
        result['zigzag_long_sl_level'] = prev_trough_price
        result['zigzag_short_pt_level'] = next_trough_price
        result['zigzag_short_sl_level'] = prev_peak_price
        
        # Calculate potential returns
        result['zigzag_long_potential_return'] = (next_peak_price - close) / close
        result['zigzag_long_risk'] = (close - prev_trough_price) / close
        result['zigzag_short_potential_return'] = (close - next_trough_price) / close
        result['zigzag_short_risk'] = (prev_peak_price - close) / close
        
        # Risk-reward ratios
        result['zigzag_long_rr_ratio'] = np.where(
            result['zigzag_long_risk'] > 0,
            result['zigzag_long_potential_return'] / result['zigzag_long_risk'],
            0
        )
        result['zigzag_short_rr_ratio'] = np.where(
            result['zigzag_short_risk'] > 0,
            result['zigzag_short_potential_return'] / result['zigzag_short_risk'],
            0
        )
        
        return result


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_trend_aware_meta_labeler(
    config: Optional[TrendAwareTripleBarrierConfig] = None
) -> TrendAwareMetaLabeler:
    """Create a trend-aware meta-labeler with optional configuration.
    
    Args:
        config: Optional configuration
        
    Returns:
        Configured TrendAwareMetaLabeler instance
    """
    return TrendAwareMetaLabeler(config)


def apply_trend_aware_meta_labeling(
    data: pd.DataFrame,
    config: Optional[TrendAwareTripleBarrierConfig] = None,
    include_labels: bool = True
) -> pd.DataFrame:
    """Apply trend-aware meta-labeling to data.
    
    Convenience function for full feature generation.
    
    Args:
        data: DataFrame with OHLCV data
        config: Optional configuration
        include_labels: Whether to include triple barrier labels
        
    Returns:
        DataFrame with trend-aware features and labels
    """
    labeler = create_trend_aware_meta_labeler(config)
    return labeler.generate_trend_aware_features(data, include_labels=include_labels)


# =============================================================================
# MAIN / EXAMPLE
# =============================================================================

if __name__ == "__main__":
    from src.utils.tprint import tprint
    
    # Create sample data
    np.random.seed(42)
    n = 1000
    dates = pd.date_range('2024-01-01', periods=n, freq='15min')
    
    # Generate price data with trends
    trend = np.cumsum(np.random.randn(n) * 0.001)
    price = 100 * np.exp(trend)
    
    data = pd.DataFrame({
        'open': price * (1 + np.random.uniform(-0.002, 0.002, n)),
        'high': price * (1 + np.random.uniform(0, 0.005, n)),
        'low': price * (1 + np.random.uniform(-0.005, 0, n)),
        'close': price,
        'volume': np.random.uniform(1000, 10000, n)
    }, index=dates)
    
    # Ensure high >= open, close and low <= open, close
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    tprint("=" * 60)
    tprint("Testing TrendAwareMetaLabeler")
    tprint("=" * 60)
    
    # Create labeler with custom config
    config = TrendAwareTripleBarrierConfig(
        base_profit_take_multiplier=0.005,
        base_stop_loss_multiplier=0.003,
        zigzag_pct_threshold=0.02,
        use_atr_for_zigzag=True
    )
    
    labeler = TrendAwareMetaLabeler(config)
    
    # Generate features
    result = labeler.generate_trend_aware_features(data, include_labels=True)
    
    tprint(f"\nGenerated features: {len(result.columns)}")
    tprint(f"Sample count: {len(result)}")
    
    # Show feature categories
    bb_cols = [c for c in result.columns if c.startswith('bb_')]
    obv_cols = [c for c in result.columns if c.startswith('obv')]
    zigzag_cols = [c for c in result.columns if c.startswith('zigzag_')]
    confluence_cols = [c for c in result.columns if c.startswith('confluence_')]
    
    tprint(f"\nBollinger Bands features: {len(bb_cols)}")
    tprint(f"OBV features: {len(obv_cols)}")
    tprint(f"ZigZag features: {len(zigzag_cols)}")
    tprint(f"Confluence features: {len(confluence_cols)}")
    
    if 'label' in result.columns:
        tprint(f"\nLabel distribution:")
        tprint(f"   LONG (1): {(result['label'] == 1).sum()}")
        tprint(f"   SHORT (-1): {(result['label'] == -1).sum()}")
        
        if 'signal_weight' in result.columns:
            tprint(f"\nSignal weight statistics:")
            tprint(f"   Mean: {result['signal_weight'].mean():.3f}")
            tprint(f"   Std: {result['signal_weight'].std():.3f}")
            tprint(f"   Min: {result['signal_weight'].min():.3f}")
            tprint(f"   Max: {result['signal_weight'].max():.3f}")
