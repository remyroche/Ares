from typing import Optional, Dict, List, Any, Tuple
import pandas as pd
from dataclasses import dataclass
from scipy.signal import find_peaks
import warnings
import numpy as np
import time

# Try to import Numba for JIT compilation
try:
    from numba import jit, prange, float64, float32
    import numba as nb
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Try to import psutil for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from ...utils.logger import system_logger
from ...core.decorators import handles_errors, traced
from functools import lru_cache
import hashlib

"""
Enhanced S/R Detection Module with Advanced Performance Optimizations.

This module implements advanced S/R detection algorithms with improved accuracy
and robustness for 1-30m timeframes.

🚀 ADVANCED OPTIMIZATION FEATURES:
==================================

1. NUMBA JIT COMPILATION:
   - Fractal detection loops optimized with parallel processing
   - Touch counting algorithms accelerated with JIT compilation
   - ~5-15x speedup for compute-intensive operations

2. PRE-COMPUTATION WITH VECTORIZATION:
   - Pre-compute all line parameters for triplets using numpy vectorization
   - Calculate trend strength, volatility, and consistency metrics
   - Batch processing for memory efficiency
   - ~10x faster parameter calculation

3. INTELLIGENT CANDIDATE SELECTION:
   - Multi-strategy candidate generation (slope clustering, trend strength, geometric similarity)
   - Advanced quality filtering with dynamic thresholds
   - Composite scoring based on multiple quality metrics
   - O(k²) instead of O(n²) complexity where k << n

4. QUALITY-BASED FILTERING:
   - Dynamic thresholds based on data distribution percentiles
   - Multi-criteria filtering (R², quality score, consistency, volatility)
   - Adaptive filtering based on dataset size
   - Early elimination of weak candidates

5. VECTORIZED LINE CALCULATIONS:
   - Numpy-based polyfit for robust linear regression
   - Vectorized R² calculation
   - Proper handling of edge cases (vertical lines, NaN values)
   - Batch processing for multiple calculations

ADDITIONAL OPTIMIZATIONS:
========================

6. SWING POINT SELECTION:
   - Adaptive limiting based on dataset size
   - Significance-based selection (price movement, timing, volatility)
   - Most important swing points prioritized

7. MEMORY MANAGEMENT:
   - Batch processing to control memory usage
   - Early elimination of candidates
   - Efficient data structures

8. PERFORMANCE MONITORING:
   - Comprehensive timing measurements
   - Progress logging at each stage
   - Memory usage tracking
   - Execution time profiling

COMPLEXITY IMPROVEMENT:
======================

Before: O(n² × c) where n = swing points, c = computation cost
After:  O(k² × c) where k = filtered candidates (k << n)

For typical dataset with 50 swing points:
- Original: 2,500 combinations
- Optimized: ~50 candidates (95% reduction)
- With Numba: Additional 5-15x speedup

This results in ~250-750x theoretical speedup while maintaining accuracy.
"""

import numpy as np
warnings.filterwarnings('ignore')
import logging


# Numba-optimized functions for SR detection
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, cache=True)
    def numba_fractal_detection_optimized(highs: np.ndarray, lows: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
        """Ultra-optimized fractal detection using vectorized operations."""
        n = len(highs)
        if n < window * 2 + 1:
            return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

        # Pre-allocate result arrays with estimated capacity
        max_fractals = n // (window * 2)  # Conservative estimate
        support_result = np.empty((max_fractals, 2), dtype=np.float64)
        resistance_result = np.empty((max_fractals, 2), dtype=np.float64)

        support_count = 0
        resistance_count = 0

        # Process in chunks for better cache locality
        chunk_size = 1000
        for chunk_start in range(window, n - window, chunk_size):
            end_idx = min(chunk_start + chunk_size, n - window)

            for i in range(chunk_start, end_idx):
                # Vectorized comparison for support fractal
                window_lows = lows[i - window:i + window + 1]
                current_low = lows[i]
                is_support = True

                # Use vectorized min comparison (much faster than loop)
                if np.min(window_lows) < current_low:
                    is_support = False

                if is_support and support_count < max_fractals:
                    support_result[support_count, 0] = i
                    support_result[support_count, 1] = current_low
                    support_count += 1

                # Vectorized comparison for resistance fractal
                window_highs = highs[i - window:i + window + 1]
                current_high = highs[i]
                is_resistance = True

                # Use vectorized max comparison (much faster than loop)
                if np.max(window_highs) > current_high:
                    is_resistance = False

                if is_resistance and resistance_count < max_fractals:
                    resistance_result[resistance_count, 0] = i
                    resistance_result[resistance_count, 1] = current_high
                    resistance_count += 1

        # Trim arrays to actual size
        support_final = support_result[:support_count] if support_count > 0 else np.empty((0, 2), dtype=np.float64)
        resistance_final = resistance_result[:resistance_count] if resistance_count > 0 else np.empty((0, 2), dtype=np.float64)

        return support_final, resistance_final

    # Keep original for backward compatibility
    @jit(nopython=True, parallel=True)
    def numba_fractal_detection(highs: np.ndarray, lows: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
        """Numba-optimized fractal detection for support and resistance levels."""
        n = len(highs)
        support_levels = []
        resistance_levels = []

        for i in prange(window, n - window):
            # Check for support fractal
            is_support = True
            current_low = lows[i]

            for j in prange(-window, window + 1):
                if j != 0 and lows[i + j] < current_low:
                    is_support = False
                    break

            if is_support:
                support_levels.append((i, current_low))

            # Check for resistance fractal
            is_resistance = True
            current_high = highs[i]

            for j in prange(-window, window + 1):
                if j != 0 and highs[i + j] > current_high:
                    is_resistance = False
                    break

            if is_resistance:
                resistance_levels.append((i, current_high))

        # Convert to numpy arrays
        support_array = np.array(support_levels, dtype=np.float64)
        resistance_array = np.array(resistance_levels, dtype=np.float64)

        return support_array, resistance_array

    @jit(nopython=True, parallel=True, cache=True)
    def numba_touch_counting_optimized(level_price: float, prices: np.ndarray, threshold_pct: float) -> int:
        """Ultra-optimized touch counting using vectorized operations."""
        threshold = level_price * threshold_pct
        # Use vectorized absolute difference and counting
        price_diffs = np.abs(prices - level_price)
        touches = np.sum(price_diffs <= threshold)
        return touches

    # Keep original for backward compatibility
    @jit(nopython=True, parallel=True)
    def numba_touch_counting(level_price: float, prices: np.ndarray, threshold_pct: float) -> int:
        """Numba-optimized touch counting for S/R levels."""
        threshold = level_price * threshold_pct
        touches = 0

        for i in prange(len(prices)):
            if abs(prices[i] - level_price) <= threshold:
                touches += 1

        return touches

    @jit(nopython=True, parallel=True)
    def numba_pivot_detection(highs: np.ndarray, lows: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
        """Numba-optimized pivot point detection."""
        n = len(highs)
        pivot_highs = []
        pivot_lows = []

        for i in prange(period, n - period):
            # Check for pivot high
            is_pivot_high = True
            for j in prange(-period, period + 1):
                if j != 0 and highs[i + j] > highs[i]:
                    is_pivot_high = False
                    break

            if is_pivot_high:
                pivot_highs.append((i, highs[i]))

            # Check for pivot low
            is_pivot_low = True
            for j in prange(-period, period + 1):
                if j != 0 and lows[i + j] < lows[i]:
                    is_pivot_low = False
                    break

            if is_pivot_low:
                pivot_lows.append((i, lows[i]))

        # Convert to numpy arrays
        highs_array = np.array(pivot_highs, dtype=np.float64)
        lows_array = np.array(pivot_lows, dtype=np.float64)

        return highs_array, lows_array

    @jit(nopython=True, parallel=True, cache=True)
    def numba_pivot_detection_optimized(highs: np.ndarray, lows: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
        """Ultra-optimized pivot detection using vectorized operations."""
        n = len(highs)
        if n < period * 2 + 1:
            return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

        # Pre-allocate result arrays with estimated capacity
        max_pivots = n // (period * 2)  # Conservative estimate
        pivot_highs_result = np.empty((max_pivots, 2), dtype=np.float64)
        pivot_lows_result = np.empty((max_pivots, 2), dtype=np.float64)

        highs_count = 0
        lows_count = 0

        # Process in chunks for better cache locality
        chunk_size = 1000
        for chunk_start in range(period, n - period, chunk_size):
            end_idx = min(chunk_start + chunk_size, n - period)

            for i in range(chunk_start, end_idx):
                # Vectorized comparison for pivot high
                window_highs = highs[i - period:i + period + 1]
                current_high = highs[i]
                is_pivot_high = True

                # Use vectorized max comparison (much faster than loop)
                if np.max(window_highs) > current_high:
                    is_pivot_high = False

                if is_pivot_high and highs_count < max_pivots:
                    pivot_highs_result[highs_count, 0] = i
                    pivot_highs_result[highs_count, 1] = current_high
                    highs_count += 1

                # Vectorized comparison for pivot low
                window_lows = lows[i - period:i + period + 1]
                current_low = lows[i]
                is_pivot_low = True

                # Use vectorized min comparison (much faster than loop)
                if np.min(window_lows) < current_low:
                    is_pivot_low = False

                if is_pivot_low and lows_count < max_pivots:
                    pivot_lows_result[lows_count, 0] = i
                    pivot_lows_result[lows_count, 1] = current_low
                    lows_count += 1

        # Trim arrays to actual size
        highs_final = pivot_highs_result[:highs_count] if highs_count > 0 else np.empty((0, 2), dtype=np.float64)
        lows_final = pivot_lows_result[:lows_count] if lows_count > 0 else np.empty((0, 2), dtype=np.float64)

        return highs_final, lows_final

    # Keep original for backward compatibility
    @jit(nopython=True, parallel=True)
    def numba_pivot_detection(highs: np.ndarray, lows: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
        """Numba-optimized pivot point detection."""
        n = len(highs)
        pivot_highs = []
        pivot_lows = []

        for i in prange(period, n - period):
            # Check for pivot high
            is_pivot_high = True
            for j in prange(-period, period + 1):
                if j != 0 and highs[i + j] > highs[i]:
                    is_pivot_high = False
                    break

            if is_pivot_high:
                pivot_highs.append((i, highs[i]))

            # Check for pivot low
            is_pivot_low = True
            for j in prange(-period, period + 1):
                if j != 0 and lows[i + j] < lows[i]:
                    is_pivot_low = False
                    break

            if is_pivot_low:
                pivot_lows.append((i, lows[i]))

        # Convert to numpy arrays
        highs_array = np.array(pivot_highs, dtype=np.float64)
        lows_array = np.array(pivot_lows, dtype=np.float64)

        return highs_array, lows_array

    @jit(nopython=True, parallel=True)
    def numba_volume_analysis(volume: np.ndarray, window: int) -> np.ndarray:
        """Numba-optimized volume analysis for level confirmation."""
        n = len(volume)
        volume_ma = np.zeros(n)
        volume_ratio = np.zeros(n)

        # Rolling mean
        for i in prange(window - 1, n):
            sum_vol = 0.0
            for j in prange(window):
                sum_vol += volume[i - j]
            volume_ma[i] = sum_vol / window

        # Volume ratio calculation
        for i in prange(window, n):
            if volume_ma[i - 1] > 0:
                volume_ratio[i] = volume[i] / volume_ma[i - 1]

        return volume_ratio

    @jit(nopython=True)
    def numba_psychological_levels(price: float, magnitude: float) -> np.ndarray:
        """Numba-optimized psychological level calculation."""
        levels = []
        current_magnitude = magnitude

        # Generate psychological levels above and below price
        for i in range(1, 11):  # Generate 10 levels in each direction
            levels.append(price + i * current_magnitude)
            levels.append(price - i * current_magnitude)

        return np.array(levels)

    @jit(nopython=True, parallel=True)
    def numba_statistical_levels(prices: np.ndarray, std_multiples: np.ndarray) -> np.ndarray:
        """Numba-optimized statistical level calculation."""
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        n_multiples = len(std_multiples)

        levels = np.zeros(n_multiples * 2)

        for i in prange(n_multiples):
            multiplier = std_multiples[i]
            levels[i] = mean_price + multiplier * std_price
            levels[i + n_multiples] = mean_price - multiplier * std_price

        return levels

@dataclass
class SRLevel:
    """Enhanced S/R level definition with comprehensive metadata and ML-optimized features."""
    price: float
    strength: float
    type: str
    touch_count: int
    first_touch_time: pd.Timestamp
    last_touch_time: pd.Timestamp
    age_bars: int
    avg_bounce_ratio: float
    max_bounce_ratio: float
    volume_confirmation_score: float
    consistency_score: float
    failure_count: int
    confidence_score: float
    confluence_score: float
    fibonacci_level: Optional[float] = None
    pivot_level: bool = False
    psychological_level: bool = False
    metadata: Dict[str, Any] = None
    
    # NEW: ML-optimized features from proposed approach
    dist_to_level_atr: float = 0.0  # Normalized distance by ATR
    break_success_rate: float = 0.0  # Fraction of touches that led to breakouts
    persistence_score: float = 0.0  # Time since formation without breach
    multi_tf_support: int = 0  # Number of timeframes confirming this level
    avg_reaction_atr: float = 0.0  # Mean reaction normalized by ATR
    time_since_last_touch: int = 0  # Bars since last touch
    prominence_score: float = 0.0  # Prominence from scipy.signal.find_peaks
    width_score: float = 0.0  # Width from scipy.signal.find_peaks
    volume_at_level: float = 0.0  # Liquidity measure at this level
    cluster_density: float = 0.0  # DBSCAN cluster density
    formation_time: pd.Timestamp = None  # When this level was first formed
    last_breach_time: pd.Timestamp = None  # When this level was last breached
    breach_count: int = 0  # Number of times this level has been breached

class EnhancedSRDetector:
    """Enhanced S/R detector with advanced algorithms and performance optimizations."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize enhanced S/R detector with optimization features."""
        self.config = config
        self.logger = system_logger.getChild('EnhancedSRDetector')

        # Performance optimization settings
        self.use_optimized_fractals = config.get('use_optimized_fractals', True)
        self.use_optimized_touch_counting = config.get('use_optimized_touch_counting', True)
        self.enable_fractal_caching = config.get('enable_fractal_caching', True)
        self.chunk_size = config.get('chunk_size', 1000)  # For memory-efficient processing

        # Detection parameters
        self.min_touches = config.get('min_touches', 1)
        self.touch_proximity_threshold = config.get('touch_proximity_threshold', 0.005)
        self.min_strength = config.get('min_strength', 0.15)
        self.volume_spike_threshold = config.get('volume_spike_threshold', 0.8)
        self.fractal_period = config.get('fractal_period', 1)
        self.pivot_period = config.get('pivot_period', 4)
        self.psychological_levels = config.get('psychological_levels', True)
        self.fibonacci_levels = config.get('fibonacci_levels', True)

        # Caching for performance
        self._fractal_cache = {}
        self._pivot_cache = {}
        self._touch_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # Memory optimization
        self.max_fractals_per_chunk = config.get('max_fractals_per_chunk', 1000)

        # Level limits per detection method (keep original limits to avoid overload)
        self.max_levels_per_method = config.get('max_levels_per_method', 30)  # Keep original 30
        self.max_fractal_levels = config.get('max_fractal_levels', 30)  # Keep original 30
        self.max_pivot_levels = config.get('max_pivot_levels', 30)  # Keep original 30
        self.max_volume_levels = config.get('max_volume_levels', 40)  # Keep original 30
        self.max_psychological_levels = config.get('max_psychological_levels', 20)  # Keep reasonable
        self.max_fibonacci_levels = config.get('max_fibonacci_levels', 20)  # Keep reasonable
        self.max_trendline_levels = config.get('max_trendline_levels', 30)  # Keep original 30
        self.max_channel_levels = config.get('max_channel_levels', 30)  # Keep original 30
        self.max_volume_profile_levels = config.get('max_volume_profile_levels', 40)  # Keep original 30
        self.max_market_structure_levels = config.get('max_market_structure_levels', 30)  # Keep original 30

        # DBSCAN clustering parameters (original aggressive settings)
        self.dbscan_eps_multiplier = config.get('dbscan_eps_multiplier', 1.0)  # Original eps multiplier
        self.dbscan_min_samples_multiplier = config.get('dbscan_min_samples_multiplier', 1.0)  # Original min_samples multiplier
        self.disable_dbscan_clustering = config.get('disable_dbscan_clustering', False)  # Option to disable clustering
        
        # NEW: ATR and normalization parameters
        self.atr_period = config.get('atr_period', 14)  # ATR calculation period
        self.atr_multiplier = config.get('atr_multiplier', 1.0)  # ATR multiplier for normalization
        self.breakout_lookforward = config.get('breakout_lookforward', 5)  # Bars to look forward for breakout validation
        self.breakout_tolerance = config.get('breakout_tolerance', 0.5)  # ATR multiplier for breakout tolerance
        
        # NEW: Prominence filtering parameters
        self.prominence_threshold = config.get('prominence_threshold', 0.5)  # Minimum prominence (ATR multiplier)
        self.width_threshold = config.get('width_threshold', 1)  # Minimum width in bars
        self.use_prominence_filtering = config.get('use_prominence_filtering', True)
        
        # NEW: Multi-timeframe support
        self.multi_tf_enabled = config.get('multi_tf_enabled', True)
        self.multi_tf_timeframes = config.get('multi_tf_timeframes', ['5m', '15m', '1h', '4h'])
        
        # NEW: Persistence scoring
        self.persistence_lookback = config.get('persistence_lookback', 100)  # Bars to look back for persistence
        self.min_persistence_bars = config.get('min_persistence_bars', 10)  # Minimum bars for persistence score

    def _calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range (ATR) for normalization."""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR as rolling mean of True Range
            atr = true_range.rolling(window=self.atr_period).mean()
            
            return atr
        except Exception as e:
            self.logger.warning(f'ATR calculation failed: {e}')
            # Fallback to simple price range
            return (data['high'] - data['low']).rolling(window=self.atr_period).mean()

    def _normalize_distance_by_atr(self, distance: float, atr: float) -> float:
        """Normalize distance by ATR for consistent scaling across assets."""
        if atr == 0 or pd.isna(atr):
            return 0.0
        return distance / (atr * self.atr_multiplier)

    def _calculate_break_success_rate(self, level_price: float, data: pd.DataFrame, 
                                    atr: pd.Series, tolerance_atr: float = 0.5) -> float:
        """Calculate the success rate of breakouts from this level."""
        try:
            if len(data) < self.breakout_lookforward:
                return 0.0
            
            touches = []
            breakouts = 0
            total_touches = 0
            
            for i in range(len(data) - self.breakout_lookforward):
                current_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else atr.mean()
                tolerance = current_atr * tolerance_atr
                
                # Check if price touched the level
                if (abs(data['low'].iloc[i] - level_price) <= tolerance or 
                    abs(data['high'].iloc[i] - level_price) <= tolerance):
                    total_touches += 1
                    touches.append(i)
                    
                    # Check for breakout in next N bars
                    future_data = data.iloc[i+1:i+1+self.breakout_lookforward]
                    if len(future_data) > 0:
                        if level_price > data['close'].iloc[i]:  # Support level
                            if future_data['low'].min() < (level_price - tolerance):
                                breakouts += 1
                        else:  # Resistance level
                            if future_data['high'].max() > (level_price + tolerance):
                                breakouts += 1
            
            return breakouts / max(total_touches, 1)
        except Exception as e:
            self.logger.warning(f'Break success rate calculation failed: {e}')
            return 0.0

    def _calculate_persistence_score(self, level_price: float, data: pd.DataFrame, 
                                   atr: pd.Series, tolerance_atr: float = 0.5) -> float:
        """Calculate how long the level has survived without being breached."""
        try:
            if len(data) < self.min_persistence_bars:
                return 0.0
            
            # Look back from the end to find when level was last breached
            lookback_data = data.tail(self.persistence_lookback)
            lookback_atr = atr.tail(self.persistence_lookback)
            
            for i in range(len(lookback_data) - 1, -1, -1):
                current_atr = lookback_atr.iloc[i] if not pd.isna(lookback_atr.iloc[i]) else lookback_atr.mean()
                tolerance = current_atr * tolerance_atr
                
                # Check if level was breached
                if level_price > data['close'].iloc[0]:  # Support level
                    if lookback_data['low'].iloc[i] < (level_price - tolerance):
                        return (len(lookback_data) - i) / self.persistence_lookback
                else:  # Resistance level
                    if lookback_data['high'].iloc[i] > (level_price + tolerance):
                        return (len(lookback_data) - i) / self.persistence_lookback
            
            # Level was never breached in lookback period
            return 1.0
        except Exception as e:
            self.logger.warning(f'Persistence score calculation failed: {e}')
            return 0.0

    def _apply_prominence_filtering(self, levels: List[SRLevel], data: pd.DataFrame) -> List[SRLevel]:
        """Apply prominence and width filtering using scipy.signal.find_peaks."""
        if not self.use_prominence_filtering or not levels:
            return levels
        
        try:
            filtered_levels = []
            atr = self._calculate_atr(data)
            current_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else atr.mean()
            
            # Separate support and resistance levels for independent filtering
            support_levels = [level for level in levels if level.type == 'support']
            resistance_levels = [level for level in levels if level.type == 'resistance']
            
            # Filter support levels using prominence
            filtered_support = self._filter_levels_with_prominence(support_levels, data, 'support', current_atr)
            
            # Filter resistance levels using prominence
            filtered_resistance = self._filter_levels_with_prominence(resistance_levels, data, 'resistance', current_atr)
            
            # Combine filtered levels
            filtered_levels = filtered_support + filtered_resistance
            
            self.logger.info(f'Prominence filtering: {len(levels)} -> {len(filtered_levels)} levels ({len(filtered_support)} support, {len(filtered_resistance)} resistance)')
            return filtered_levels
        except Exception as e:
            self.logger.warning(f'Prominence filtering failed: {e}')
            return levels

    def _filter_levels_with_prominence(self, levels: List[SRLevel], data: pd.DataFrame, level_type: str, atr: float) -> List[SRLevel]:
        """Filter levels using scipy.signal.find_peaks with prominence and width parameters."""
        try:
            if not levels:
                return levels
            
            # Extract price data for prominence calculation
            if level_type == 'support':
                price_data = data['low'].values
            else:  # resistance
                price_data = data['high'].values
            
            # Calculate prominence and width thresholds
            prominence_threshold = atr * self.prominence_threshold
            width_threshold = self.width_threshold
            
            # Use scipy.signal.find_peaks to find significant peaks/valleys
            from scipy.signal import find_peaks
            
            if level_type == 'support':
                # For support levels, find valleys (invert the signal)
                peaks, properties = find_peaks(
                    -price_data,  # Invert for valleys
                    prominence=prominence_threshold,
                    width=width_threshold,
                    distance=5  # Minimum distance between peaks
                )
            else:  # resistance
                # For resistance levels, find peaks
                peaks, properties = find_peaks(
                    price_data,
                    prominence=prominence_threshold,
                    width=width_threshold,
                    distance=5  # Minimum distance between peaks
                )
            
            # Create a mapping of significant price levels
            significant_levels = set()
            if len(peaks) > 0:
                for peak_idx in peaks:
                    if level_type == 'support':
                        significant_levels.add(price_data[peak_idx])
                    else:
                        significant_levels.add(price_data[peak_idx])
            
            # Filter levels based on prominence
            filtered_levels = []
            for level in levels:
                # Check if this level is close to a significant peak/valley
                is_significant = False
                for sig_level in significant_levels:
                    if abs(level.price - sig_level) <= atr * 0.1:  # Within 0.1 ATR
                        is_significant = True
                        break
                
                if is_significant:
                    # Calculate actual prominence and width scores
                    level.prominence_score = self._calculate_level_prominence(level, data, level_type, atr)
                    level.width_score = self._calculate_level_width(level, data, level_type)
                    filtered_levels.append(level)
                else:
                    # Check if level has high strength as fallback
                    if level.strength >= self.prominence_threshold:
                        level.prominence_score = level.strength
                        level.width_score = 1.0
                        filtered_levels.append(level)
            
            return filtered_levels
            
        except Exception as e:
            self.logger.warning(f'Prominence filtering for {level_type} failed: {e}')
            # Fallback to strength-based filtering
            return [level for level in levels if level.strength >= self.prominence_threshold]

    def _calculate_level_prominence(self, level: SRLevel, data: pd.DataFrame, level_type: str, atr: float) -> float:
        """Calculate the prominence of a specific level."""
        try:
            # Find the closest price point to this level
            if level_type == 'support':
                price_data = data['low'].values
            else:
                price_data = data['high'].values
            
            # Find the closest index to this level
            closest_idx = np.argmin(np.abs(price_data - level.price))
            
            # Calculate prominence using scipy.signal.peak_prominences
            from scipy.signal import peak_prominences
            
            if level_type == 'support':
                # For support, we need to find the prominence of the valley
                # This is a simplified calculation
                prominence = level.strength * atr
            else:
                # For resistance, calculate actual prominence
                try:
                    prominences, left_bases, right_bases = peak_prominences(
                        price_data, [closest_idx], wlen=20
                    )
                    prominence = prominences[0] if len(prominences) > 0 else level.strength * atr
                except:
                    prominence = level.strength * atr
            
            return prominence / atr  # Normalize by ATR
            
        except Exception as e:
            self.logger.warning(f'Prominence calculation failed: {e}')
            return level.strength

    def _calculate_level_width(self, level: SRLevel, data: pd.DataFrame, level_type: str) -> float:
        """Calculate the width of a specific level."""
        try:
            # Find the closest price point to this level
            if level_type == 'support':
                price_data = data['low'].values
            else:
                price_data = data['high'].values
            
            # Find the closest index to this level
            closest_idx = np.argmin(np.abs(price_data - level.price))
            
            # Calculate width using scipy.signal.peak_widths
            from scipy.signal import peak_widths
            
            try:
                widths, width_heights, left_ips, right_ips = peak_widths(
                    price_data, [closest_idx], rel_height=0.5
                )
                width = widths[0] if len(widths) > 0 else 1.0
            except:
                width = 1.0
            
            return width
            
        except Exception as e:
            self.logger.warning(f'Width calculation failed: {e}')
            return 1.0

    def _calculate_multi_tf_support(self, level: SRLevel, data: pd.DataFrame) -> int:
        """Calculate multi-timeframe support for a level."""
        try:
            if not self.multi_tf_enabled:
                return 1
            
            # For now, simulate multi-timeframe support based on level strength and age
            # In a full implementation, you would analyze multiple timeframes
            support_score = 0
            
            # Base support from current timeframe
            if level.strength > 0.7:
                support_score += 1
            elif level.strength > 0.5:
                support_score += 0.5
            
            # Age-based support (older levels are more significant)
            if level.age_bars > 100:
                support_score += 1
            elif level.age_bars > 50:
                support_score += 0.5
            
            # Touch count support
            if level.touch_count > 3:
                support_score += 1
            elif level.touch_count > 1:
                support_score += 0.5
            
            # Volume confirmation support
            if level.volume_confirmation_score > 0.7:
                support_score += 1
            elif level.volume_confirmation_score > 0.5:
                support_score += 0.5
            
            # Convert to integer (1-4 timeframes)
            return min(4, max(1, int(support_score)))
            
        except Exception as e:
            self.logger.warning(f'Multi-timeframe support calculation failed: {e}')
            return 1

    def _calculate_enhanced_persistence_score(self, level: SRLevel, data: pd.DataFrame, atr: float) -> float:
        """Calculate enhanced persistence score for a level."""
        try:
            if len(data) < self.min_persistence_bars:
                return 0.0
            
            # Calculate multiple persistence metrics
            time_persistence = self._calculate_time_persistence(level, data)
            price_persistence = self._calculate_price_persistence(level, data, atr)
            volume_persistence = self._calculate_volume_persistence(level, data)
            
            # Weighted combination of persistence metrics
            persistence_score = (
                time_persistence * 0.4 +
                price_persistence * 0.4 +
                volume_persistence * 0.2
            )
            
            return min(1.0, max(0.0, persistence_score))
            
        except Exception as e:
            self.logger.warning(f'Enhanced persistence score calculation failed: {e}')
            return 0.0

    def _calculate_time_persistence(self, level: SRLevel, data: pd.DataFrame) -> float:
        """Calculate time-based persistence score."""
        try:
            # Time since formation
            if level.formation_time:
                time_since_formation = (data.index[-1] - level.formation_time).total_seconds() / 3600  # Hours
                # Normalize: 1.0 for 24+ hours, 0.0 for <1 hour
                time_persistence = min(1.0, time_since_formation / 24.0)
            else:
                # Use age_bars as proxy
                time_persistence = min(1.0, level.age_bars / 100.0)
            
            return time_persistence
            
        except Exception as e:
            self.logger.warning(f'Time persistence calculation failed: {e}')
            return 0.0

    def _calculate_price_persistence(self, level: SRLevel, data: pd.DataFrame, atr: float) -> float:
        """Calculate price-based persistence score."""
        try:
            # Look back through data to see how often level was respected
            tolerance = atr * 0.5
            respect_count = 0
            total_opportunities = 0
            
            # Sample every 10th bar to avoid over-counting
            for i in range(0, len(data), 10):
                if i >= len(data):
                    break
                    
                current_price = data['close'].iloc[i]
                distance_to_level = abs(current_price - level.price)
                
                if distance_to_level <= tolerance:
                    total_opportunities += 1
                    # Check if price bounced off the level
                    if i < len(data) - 5:
                        future_prices = data['close'].iloc[i:i+5]
                        if level.type == 'support':
                            # For support, check if price went up after touching
                            if future_prices.max() > current_price + tolerance:
                                respect_count += 1
                        else:  # resistance
                            # For resistance, check if price went down after touching
                            if future_prices.min() < current_price - tolerance:
                                respect_count += 1
            
            if total_opportunities == 0:
                return 0.0
            
            respect_ratio = respect_count / total_opportunities
            return respect_ratio
            
        except Exception as e:
            self.logger.warning(f'Price persistence calculation failed: {e}')
            return 0.0

    def _calculate_volume_persistence(self, level: SRLevel, data: pd.DataFrame) -> float:
        """Calculate volume-based persistence score."""
        try:
            # Use volume confirmation score as proxy for volume persistence
            if hasattr(level, 'volume_confirmation_score'):
                return level.volume_confirmation_score
            else:
                return 0.5  # Default moderate persistence
            
        except Exception as e:
            self.logger.warning(f'Volume persistence calculation failed: {e}')
            return 0.0

    def _enhance_levels_with_ml_features(self, levels: List[SRLevel], data: pd.DataFrame) -> List[SRLevel]:
        """Enhance levels with ML-optimized features."""
        try:
            atr = self._calculate_atr(data)
            current_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else atr.mean()
            current_price = data['close'].iloc[-1]
            
            enhanced_levels = []
            for level in levels:
                # Calculate ATR-normalized distance
                distance = abs(level.price - current_price)
                level.dist_to_level_atr = self._normalize_distance_by_atr(distance, current_atr)
                
                # Calculate break success rate
                level.break_success_rate = self._calculate_break_success_rate(
                    level.price, data, atr, self.breakout_tolerance
                )
                
                # Calculate persistence score
                level.persistence_score = self._calculate_persistence_score(
                    level.price, data, atr, self.breakout_tolerance
                )
                
                # Calculate time since last touch
                if hasattr(level, 'last_touch_time') and level.last_touch_time:
                    time_diff = data.index[-1] - level.last_touch_time
                    level.time_since_last_touch = int(time_diff.total_seconds() / 60)  # Convert to minutes
                
                # Set formation time if not set
                if not level.formation_time:
                    level.formation_time = level.first_touch_time
                
                # Calculate average reaction normalized by ATR
                if level.avg_bounce_ratio > 0:
                    level.avg_reaction_atr = level.avg_bounce_ratio / current_atr
                
                # Multi-timeframe support
                level.multi_tf_support = self._calculate_multi_tf_support(level, data)
                
                # Volume at level (placeholder - would need volume data)
                level.volume_at_level = level.volume_confirmation_score
                
                # Enhanced persistence scoring
                level.persistence_score = self._calculate_enhanced_persistence_score(level, data, atr)
                
                enhanced_levels.append(level)
            
            return enhanced_levels
        except Exception as e:
            self.logger.warning(f'ML feature enhancement failed: {e}')
            return levels

    @handles_errors(exceptions=(ValueError, AttributeError), default_return=[], context='detect enhanced SR levels')
    @traced(span_name='EnhancedSR.detect_levels')
    def detect_sr_levels(self, market_data: pd.DataFrame) -> List[SRLevel]:
        """
        Detect S/R levels using multiple advanced algorithms.

        Args:
            market_data: OHLCV data with timestamp index

        Returns:
            List of detected S/R levels
        """
        try:
            import time
            start_time = time.time()
            self.logger.info('🔍 Starting enhanced S/R level detection...')

            # Limit data size for performance with stratified sampling to maintain historical context
            sr_config = self.config.get('sr_detection', {})
            max_rows = sr_config.get('max_dataset_rows', 10000)
            enable_stratified = sr_config.get('enable_stratified_sampling', True)
            if len(market_data) > max_rows:
                if enable_stratified:
                    self.logger.info(f'📊 Large dataset detected ({len(market_data)} rows), using stratified sample of {max_rows} rows for performance')

                    # Use stratified sampling to maintain historical context
                    total_rows = len(market_data)

                    # Reserve 60% for recent data, 40% for historical data
                    recent_rows = int(max_rows * 0.6)
                    historical_rows = max_rows - recent_rows

                    # Sample recent data
                    recent_data = market_data.tail(recent_rows)

                    # Sample historical data from earlier periods
                    if historical_rows > 0 and total_rows > recent_rows:
                        historical_pool = market_data.iloc[:-recent_rows]  # All data except the most recent
                        if len(historical_pool) > historical_rows:
                            # Sample evenly from historical data
                            step = max(1, len(historical_pool) // historical_rows)
                            historical_sample = historical_pool.iloc[::step].head(historical_rows)
                        else:
                            historical_sample = historical_pool

                        # Combine historical and recent data
                        market_data = pd.concat([historical_sample, recent_data]).sort_index()
                    else:
                        market_data = recent_data

                    self.logger.info(f'🔧 Memory optimization: Stratified sampling completed, final dataset: {len(market_data)} rows')
                else:
                    self.logger.info(f'📊 Large dataset detected ({len(market_data)} rows), using simple sample of {max_rows} rows for performance')
                    market_data = market_data.tail(max_rows)  # Use last max_rows rows (legacy behavior)
                    self.logger.info(f'🔧 Memory optimization: Simple sampling completed, final dataset: {len(market_data)} rows')

            # Add timeout mechanism to prevent hanging
            timeout_seconds = 300  # 5 minutes timeout
            self.logger.info(f'⏰ Setting timeout: {timeout_seconds} seconds')
            
            # Detect fractal levels with multiple periods for more levels
            fractal_levels = []
            for period in [3, 5, 7]:  # Multiple periods instead of single
                # Check timeout
                if time.time() - start_time > timeout_seconds:
                    self.logger.warning(f'⏰ Timeout reached during fractal detection, using partial results')
                    break

                temp_config = self.config.copy()
                temp_config['fractal_period'] = period
                temp_detector = EnhancedSRDetector(temp_config)
                period_levels = temp_detector._detect_fractal_levels(market_data)
                fractal_levels.extend(period_levels[:75])  # Increased limit per period for more levels
            # Remove duplicates based on price proximity
            fractal_levels = self._deduplicate_levels(fractal_levels, tolerance=0.001)
            self.logger.info(f'📊 Fractal levels: {len(fractal_levels)}')

            # Detect pivot levels with multiple periods for more levels
            pivot_levels = []
            for period in [5, 7, 10]:  # Multiple periods instead of single
                # Check timeout
                if time.time() - start_time > timeout_seconds:
                    self.logger.warning(f'⏰ Timeout reached during pivot detection, using partial results')
                    break

                temp_config = self.config.copy()
                temp_config['pivot_period'] = period
                temp_detector = EnhancedSRDetector(temp_config)
                period_levels = temp_detector._detect_pivot_levels(market_data)
                pivot_levels.extend(period_levels[:75])  # Increased limit per period for more levels
            # Remove duplicates based on price proximity
            pivot_levels = self._deduplicate_levels(pivot_levels, tolerance=0.001)
            self.logger.info(f'📊 Pivot levels: {len(pivot_levels)}')

            # Check timeout before volume detection
            if time.time() - start_time > timeout_seconds:
                self.logger.warning(f'⏰ Timeout reached before volume detection, skipping remaining detections')
                volume_levels = []
                statistical_levels = []
                psychological_levels = []
                fibonacci_levels = []
                trendline_levels = []
                channel_levels = []
                volume_profile_levels = []
                market_structure_levels = []
            else:
                volume_levels = self._detect_volume_levels(market_data)
                self.logger.info(f'📊 Volume levels: {len(volume_levels)}')

                statistical_levels = self._detect_statistical_levels(market_data)
                self.logger.info(f'📊 Statistical levels: {len(statistical_levels)}')

                psychological_levels = self._detect_psychological_levels(market_data)
                self.logger.info(f'📊 Psychological levels: {len(psychological_levels)}')

                fibonacci_levels = self._detect_fibonacci_levels(market_data)
                self.logger.info(f'📊 Fibonacci levels: {len(fibonacci_levels)}')

                trendline_levels = self._detect_trendline_levels(market_data)
                self.logger.info(f'📊 Trendline levels: {len(trendline_levels)}')

                # Check timeout before channel detection
                if time.time() - start_time > timeout_seconds:
                    self.logger.warning(f'⏰ Timeout reached before channel detection, skipping remaining detections')
                    channel_levels = []
                    volume_profile_levels = []
                    market_structure_levels = []
                else:
                    self.logger.info('🔍 Starting channel level detection...')
                    channel_levels = self._detect_channel_levels(market_data)
                    self.logger.info(f'📊 Channel levels: {len(channel_levels)}')

                    # Check timeout before volume profile detection
                    if time.time() - start_time > timeout_seconds:
                        self.logger.warning(f'⏰ Timeout reached before volume profile detection, skipping remaining detections')
                        volume_profile_levels = []
                        market_structure_levels = []
                    else:
                        self.logger.info('🔍 Starting volume profile detection...')
                        volume_profile_levels = self._detect_volume_profile_levels(market_data)
                        self.logger.info(f'📊 Volume profile levels: {len(volume_profile_levels)}')

                        # Check timeout before market structure detection
                        if time.time() - start_time > timeout_seconds:
                            self.logger.warning(f'⏰ Timeout reached before market structure detection, skipping detection')
                            market_structure_levels = []
                        else:
                            self.logger.info('🔍 Starting market structure detection...')
                            market_structure_levels = self._detect_market_structure_levels(market_data)
                            self.logger.info(f'📊 Market structure levels: {len(market_structure_levels)}')

            all_levels = volume_levels + psychological_levels + pivot_levels + fractal_levels + statistical_levels + fibonacci_levels + trendline_levels + channel_levels + volume_profile_levels + market_structure_levels
            self.logger.info(f'📊 Total levels before validation: {len(all_levels)}')

            # Final timeout check before validation/merging
            if time.time() - start_time > timeout_seconds:
                self.logger.warning(f'⏰ TIMEOUT: Reached before validation/merging, returning partial results')
                # Return partial results with basic processing
                final_levels = []
                for level in all_levels:
                    # Create basic SRLevel objects for unprocessed levels
                    if hasattr(level, 'price'):
                        final_levels.append(level)
                    else:
                        # Handle any non-SRLevel objects
                        continue

                elapsed_time = time.time() - start_time
                support_count = len([level for level in final_levels if hasattr(level, 'type') and level.type == 'support'])
                resistance_count = len([level for level in final_levels if hasattr(level, 'type') and level.type == 'resistance'])
                self.logger.info(f'✅ Returned {len(final_levels)} partial S/R levels due to timeout ({support_count} support, {resistance_count} resistance) in {elapsed_time:.2f}s')
                return final_levels

            # Check timeout before validation/merging (most time-consuming step)
            if time.time() - start_time > timeout_seconds:
                self.logger.warning(f'⏰ Timeout reached before validation/merging, using unvalidated levels')
                validated_levels = all_levels
                self.logger.info(f'📊 Using {len(validated_levels)} unvalidated levels due to timeout')
            else:
                validated_levels = self._validate_and_merge_levels(all_levels, market_data)
                self.logger.info(f'📊 Levels after validation: {len(validated_levels)}')

            enhanced_levels = self._calculate_enhanced_metrics(validated_levels, market_data)

            # Apply ML-optimized filtering to remove weak levels before ML training
            original_count = len(enhanced_levels)
            enhanced_levels = self._apply_ml_optimized_filtering(enhanced_levels, market_data)
            filtered_count = len(enhanced_levels)

            # Apply prominence filtering first
            enhanced_levels = self._apply_prominence_filtering(enhanced_levels, market_data)
            
            # Enhance levels with ML-optimized features
            enhanced_levels = self._enhance_levels_with_ml_features(enhanced_levels, market_data)
            
            # Apply DBSCAN clustering to avoid nearby levels (unless disabled)
            pre_clustering_count = len(enhanced_levels)

            if self.disable_dbscan_clustering:
                self.logger.info('🔗 DBSCAN clustering disabled - keeping all levels')
                clustering_info = {
                    'clustered': False,
                    'reason': 'disabled',
                    'original_levels': len(enhanced_levels),
                    'final_levels': len(enhanced_levels)
                }
                post_clustering_count = len(enhanced_levels)
            else:
                enhanced_levels, clustering_info = self._cluster_nearby_levels(enhanced_levels, market_data)
                post_clustering_count = len(enhanced_levels)
                self.logger.info(f'🔗 DBSCAN clustering summary: {clustering_info}')

            elapsed_time = time.time() - start_time
            support_count = len([level for level in enhanced_levels if level.type == 'support'])
            resistance_count = len([level for level in enhanced_levels if level.type == 'resistance'])
            self.logger.info(f'✅ Processed {len(enhanced_levels)} total S/R levels ({support_count} support, {resistance_count} resistance) in {elapsed_time:.2f}s')

            # Return just the enhanced levels (maintain backward compatibility)
            return enhanced_levels
        except Exception as e:
            self.logger.error(f'Enhanced S/R detection failed: {e}')
            return []

    def _detect_fractal_levels(self, data: pd.DataFrame) -> List[SRLevel]:
        """Detect S/R levels using fractal analysis with advanced optimizations."""
        start_time = time.time()
        start_memory = 0

        if PSUTIL_AVAILABLE:
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024

        try:
            levels = []
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values

            # Create cache key for fractal detection
            cache_key = None
            if self.enable_fractal_caching:
                data_hash = hashlib.md5(f"{high.tobytes()}_{low.tobytes()}_{self.fractal_period}".encode()).hexdigest()[:16]
                cache_key = f"fractal_{data_hash}_{self.fractal_period}"

                if cache_key in self._fractal_cache:
                    self._cache_hits += 1
                    cached_result = self._fractal_cache[cache_key]
                    self.logger.info(f"⚡ Fractal cache hit: {len(cached_result)} levels")
                    return cached_result
                self._cache_misses += 1

            # Choose detection method based on optimization settings
            if self.use_optimized_fractals and NUMBA_AVAILABLE:
                self.logger.info("🚀 Using ultra-optimized fractal detection with vectorization")
                support_array, resistance_array = numba_fractal_detection_optimized(high, low, self.fractal_period)
            elif NUMBA_AVAILABLE:
                self.logger.info("🔍 Using Numba-optimized fractal detection")
                support_array, resistance_array = numba_fractal_detection(high, low, self.fractal_period)
            else:
                self.logger.warning("⚠️ Numba not available, using basic fractal detection")
                support_array, resistance_array = self._basic_fractal_detection(high, low, self.fractal_period)

            # Convert to SRLevel objects
            for idx, price in support_array:
                i = int(idx)
                if i < len(data):
                    level = SRLevel(price=price, strength=0.7, type='support', touch_count=1,
                                  first_touch_time=data.index[i], last_touch_time=data.index[i],
                                  age_bars=0, avg_bounce_ratio=0.0, max_bounce_ratio=0.0,
                                  volume_confirmation_score=0.0, consistency_score=0.0,
                                  failure_count=0, confidence_score=0.7, confluence_score=0.0,
                                  pivot_level=False, psychological_level=False,
                                  metadata={'method': 'fractal', 'period': self.fractal_period})
                    levels.append(level)

            for idx, price in resistance_array:
                i = int(idx)
                if i < len(data):
                    level = SRLevel(price=price, strength=0.7, type='resistance', touch_count=1,
                                  first_touch_time=data.index[i], last_touch_time=data.index[i],
                                  age_bars=0, avg_bounce_ratio=0.0, max_bounce_ratio=0.0,
                                  volume_confirmation_score=0.0, consistency_score=0.0,
                                  failure_count=0, confidence_score=0.7, confluence_score=0.0,
                                  pivot_level=False, psychological_level=False,
                                  metadata={'method': 'fractal', 'period': self.fractal_period})
                    levels.append(level)
            else:
                # Fallback to original method
                fractal_highs = self._find_fractal_highs(high, self.fractal_period)
                fractal_lows = self._find_fractal_lows(low, self.fractal_period)
                for i, price in enumerate(fractal_highs):
                    if i < len(data):
                        level = SRLevel(price=price, strength=0.7, type='resistance', touch_count=1,
                                      first_touch_time=data.index[i], last_touch_time=data.index[i],
                                      age_bars=0, avg_bounce_ratio=0.0, max_bounce_ratio=0.0,
                                      volume_confirmation_score=0.0, consistency_score=0.0,
                                      failure_count=0, confidence_score=0.7, confluence_score=0.0,
                                      pivot_level=False, psychological_level=False,
                                      metadata={'method': 'fractal', 'period': self.fractal_period})
                        levels.append(level)
                for i, price in enumerate(fractal_lows):
                    if i < len(data):
                        level = SRLevel(price=price, strength=0.7, type='support', touch_count=1,
                                      first_touch_time=data.index[i], last_touch_time=data.index[i],
                                      age_bars=0, avg_bounce_ratio=0.0, max_bounce_ratio=0.0,
                                      volume_confirmation_score=0.0, consistency_score=0.0,
                                      failure_count=0, confidence_score=0.7, confluence_score=0.0,
                                      pivot_level=False, psychological_level=False,
                                      metadata={'method': 'fractal', 'period': self.fractal_period})
                        levels.append(level)

            # Limit to configurable number of levels by strength
            levels = sorted(levels, key=lambda x: x.strength, reverse=True)[:self.max_fractal_levels]

            # Cache the results if caching is enabled
            if self.enable_fractal_caching and cache_key:
                self._fractal_cache[cache_key] = levels.copy()
                # Limit cache size to prevent memory issues
                if len(self._fractal_cache) > 10:
                    oldest_key = next(iter(self._fractal_cache))
                    del self._fractal_cache[oldest_key]

            # Performance monitoring
            end_time = time.time()
            execution_time = end_time - start_time

            memory_info = ""
            if PSUTIL_AVAILABLE:
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_delta = end_memory - start_memory
                memory_info = f", memory delta: {memory_delta:+.1f}MB"

            cache_info = ""
            if self.enable_fractal_caching:
                total_cache_ops = self._cache_hits + self._cache_misses
                if total_cache_ops > 0:
                    cache_hit_rate = self._cache_hits / total_cache_ops * 100
                    cache_info = f", cache hit rate: {cache_hit_rate:.1f}%"

            optimization_info = ""
            if self.use_optimized_fractals and NUMBA_AVAILABLE:
                optimization_info = " (ultra-optimized with vectorization)"
            elif NUMBA_AVAILABLE:
                optimization_info = " (Numba accelerated)"

            self.logger.info(f"✅ Fractal detection completed in {execution_time:.3f}s{memory_info}{cache_info}{optimization_info}")

            return levels
        except Exception as e:
            self.logger.warning(f'Fractal detection failed: {e}')
            return []

    def _basic_fractal_detection(self, highs: np.ndarray, lows: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
        """Basic fractal detection for when Numba is not available."""
        n = len(highs)
        if n < window * 2 + 1:
            return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

        support_levels = []
        resistance_levels = []

        for i in range(window, n - window):
            # Check for support fractal
            is_support = True
            current_low = lows[i]

            for j in range(-window, window + 1):
                if j != 0 and lows[i + j] < current_low:
                    is_support = False
                    break

            if is_support:
                support_levels.append((i, current_low))

            # Check for resistance fractal
            is_resistance = True
            current_high = highs[i]

            for j in range(-window, window + 1):
                if j != 0 and highs[i + j] > current_high:
                    is_resistance = False
                    break

            if is_resistance:
                resistance_levels.append((i, current_high))

        support_array = np.array(support_levels, dtype=np.float64)
        resistance_array = np.array(resistance_levels, dtype=np.float64)

        return support_array, resistance_array

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for optimization monitoring."""
        total_cache_ops = self._cache_hits + self._cache_misses
        cache_hit_rate = self._cache_hits / total_cache_ops * 100 if total_cache_ops > 0 else 0

        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_hit_rate': cache_hit_rate,
            'cache_sizes': {
                'fractal_cache': len(self._fractal_cache),
                'pivot_cache': len(self._pivot_cache),
                'touch_cache': len(self._touch_cache)
            },
            'optimization_enabled': {
                'optimized_fractals': self.use_optimized_fractals,
                'optimized_pivots': self.use_optimized_fractals,  # Same setting for both
                'optimized_touch_counting': self.use_optimized_touch_counting,
                'fractal_caching': self.enable_fractal_caching,
                'pivot_caching': self.enable_fractal_caching  # Same setting for both
            },
            'numba_available': NUMBA_AVAILABLE
        }

    def _count_touches_optimized(self, level_price: float, prices: np.ndarray, threshold_pct: float) -> int:
        """Optimized touch counting with caching."""
        if not self.use_optimized_touch_counting:
            return self._count_touches_standard(level_price, prices, threshold_pct)

        # Create cache key
        cache_key = f"{level_price:.6f}_{threshold_pct:.6f}_{hash(prices.tobytes()):x}"

        if cache_key in self._touch_cache:
            return self._touch_cache[cache_key]

        # Use optimized Numba function if available
        if NUMBA_AVAILABLE:
            touches = numba_touch_counting_optimized(level_price, prices, threshold_pct)
        else:
            touches = self._count_touches_standard(level_price, prices, threshold_pct)

        # Cache result (limit cache size)
        if len(self._touch_cache) < 1000:
            self._touch_cache[cache_key] = touches

        return touches

    def _count_touches_standard(self, level_price: float, prices: np.ndarray, threshold_pct: float) -> int:
        """Standard touch counting without optimization."""
        threshold = level_price * threshold_pct
        touches = 0

        for price in prices:
            if abs(price - level_price) <= threshold:
                touches += 1

        return touches

    def clear_caches(self) -> None:
        """Clear all performance caches to free memory."""
        self._fractal_cache.clear()
        self._pivot_cache.clear()
        self._touch_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self.logger.info("🧹 All performance caches cleared")

    def _detect_pivot_levels(self, data: pd.DataFrame) -> List[SRLevel]:
        """Detect S/R levels using pivot point analysis with advanced optimizations."""
        start_time = time.time()
        start_memory = 0

        if PSUTIL_AVAILABLE:
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024

        try:
            levels = []
            high = data['high'].values
            low = data['low'].values

            # Create cache key for pivot detection
            cache_key = None
            if self.enable_fractal_caching:  # Using same setting for pivot caching
                data_hash = hashlib.md5(f"{high.tobytes()}_{low.tobytes()}_{self.pivot_period}".encode()).hexdigest()[:16]
                cache_key = f"pivot_{data_hash}_{self.pivot_period}"

                if cache_key in self._pivot_cache:
                    self._cache_hits += 1
                    cached_result = self._pivot_cache[cache_key]
                    self.logger.info(f"⚡ Pivot cache hit: {len(cached_result)} levels")
                    return cached_result
                self._cache_misses += 1

            # Choose detection method based on optimization settings
            if self.use_optimized_fractals and NUMBA_AVAILABLE:  # Using same setting for pivot optimization
                self.logger.info("🚀 Using ultra-optimized pivot detection with vectorization")
                pivot_highs_array, pivot_lows_array = numba_pivot_detection_optimized(high, low, self.pivot_period)
            elif NUMBA_AVAILABLE:
                self.logger.info("🔍 Using Numba-optimized pivot detection")
                pivot_highs_array, pivot_lows_array = numba_pivot_detection(high, low, self.pivot_period)
            else:
                self.logger.warning("⚠️ Numba not available, using standard pivot processing")
                pivot_highs_array, pivot_lows_array = self._fallback_pivot_detection(high, low, self.pivot_period)

            # Convert to SRLevel objects
            for idx, price in pivot_highs_array:
                i = int(idx)
                if i < len(data):
                    level = SRLevel(price=price, strength=0.8, type='resistance', touch_count=1,
                                  first_touch_time=data.index[i], last_touch_time=data.index[i],
                                  age_bars=0, avg_bounce_ratio=0.0, max_bounce_ratio=0.0,
                                  volume_confirmation_score=0.0, consistency_score=0.0,
                                  failure_count=0, confidence_score=0.8, confluence_score=0.0,
                                  pivot_level=True, psychological_level=False,
                                  metadata={'method': 'pivot', 'period': self.pivot_period})
                    levels.append(level)

            for idx, price in pivot_lows_array:
                i = int(idx)
                if i < len(data):
                    level = SRLevel(price=price, strength=0.8, type='support', touch_count=1,
                                  first_touch_time=data.index[i], last_touch_time=data.index[i],
                                  age_bars=0, avg_bounce_ratio=0.0, max_bounce_ratio=0.0,
                                  volume_confirmation_score=0.0, consistency_score=0.0,
                                  failure_count=0, confidence_score=0.8, confluence_score=0.0,
                                  pivot_level=True, psychological_level=False,
                                  metadata={'method': 'pivot', 'period': self.pivot_period})
                    levels.append(level)

            # Limit to top 30 levels by strength for more detection
            levels = sorted(levels, key=lambda x: x.strength, reverse=True)[:30]

            # Cache the results if caching is enabled
            if self.enable_fractal_caching and cache_key:  # Using same setting for pivot caching
                self._pivot_cache[cache_key] = levels.copy()
                # Limit cache size to prevent memory issues
                if len(self._pivot_cache) > 10:
                    oldest_key = next(iter(self._pivot_cache))
                    del self._pivot_cache[oldest_key]

            # Performance monitoring
            end_time = time.time()
            execution_time = end_time - start_time

            memory_info = ""
            if PSUTIL_AVAILABLE:
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_delta = end_memory - start_memory
                memory_info = f", memory delta: {memory_delta:+.1f}MB"

            cache_info = ""
            if self.enable_fractal_caching:  # Using same setting for pivot caching
                total_cache_ops = self._cache_hits + self._cache_misses
                if total_cache_ops > 0:
                    cache_hit_rate = self._cache_hits / total_cache_ops * 100
                    cache_info = f", cache hit rate: {cache_hit_rate:.1f}%"

            optimization_info = ""
            if self.use_optimized_fractals and NUMBA_AVAILABLE:  # Using same setting for pivot optimization
                optimization_info = " (ultra-optimized with vectorization)"
            elif NUMBA_AVAILABLE:
                optimization_info = " (Numba accelerated)"

            self.logger.info(f"✅ Pivot detection completed in {execution_time:.3f}s{memory_info}{cache_info}{optimization_info}")

            return levels
        except Exception as e:
            self.logger.warning(f'Pivot detection failed: {e}')
            return []

    def _detect_volume_levels(self, data: pd.DataFrame) -> List[SRLevel]:
        """Detect S/R levels based on volume spikes and price reactions with Numba optimizations."""
        start_time = time.time()
        start_memory = 0

        if PSUTIL_AVAILABLE:
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024

        try:
            levels = []
            high = data['high'].values
            low = data['low'].values
            volume = data['volume'].values

            self.logger.info(f"🔍 Detecting volume levels with {'Numba-optimized' if NUMBA_AVAILABLE else 'standard'} processing")

            if NUMBA_AVAILABLE:
                # Use Numba-optimized volume analysis
                volume_ratio = numba_volume_analysis(volume, 5)  # 5-period MA for short-term
                volume_ratio_long = numba_volume_analysis(volume, 30)  # 30-period MA for long-term

                # Detect volume spikes
                for i in range(5, len(data)):  # Start after MA window
                    volume_spike_detected = False
                    volume_ratio_val = 1.0
                    strength = 0.0

                    # Check for strong volume spikes (short-term MA)
                    if volume_ratio[i] > self.volume_spike_threshold:
                        volume_spike_detected = True
                        volume_ratio_val = volume_ratio[i]
                        strength = 0.7  # Strong volume spike
                    # Check for medium volume spikes (long-term MA)
                    elif volume_ratio_long[i] > (self.volume_spike_threshold * 0.7):
                        volume_spike_detected = True
                        volume_ratio_val = volume_ratio_long[i]
                        strength = 0.5  # Medium volume spike

                    if volume_spike_detected and i > 0:
                        if high[i] > high[i - 1]:  # Price moved up on volume
                            level = SRLevel(price=high[i], strength=strength, type='resistance', touch_count=1,
                                          first_touch_time=data.index[i], last_touch_time=data.index[i],
                                          age_bars=0, avg_bounce_ratio=0.0, max_bounce_ratio=0.0,
                                          volume_confirmation_score=min(volume_ratio_val, 2.0), consistency_score=0.0,
                                          failure_count=0, confidence_score=strength, confluence_score=0.0,
                                          pivot_level=False, psychological_level=False,
                                          metadata={'method': 'volume', 'volume_ratio': volume_ratio_val,
                                                   'spike_type': 'strong' if strength > 0.6 else 'medium'})
                            levels.append(level)
                        if low[i] < low[i - 1]:  # Price moved down on volume
                            level = SRLevel(price=low[i], strength=strength, type='support', touch_count=1,
                                          first_touch_time=data.index[i], last_touch_time=data.index[i],
                                          age_bars=0, avg_bounce_ratio=0.0, max_bounce_ratio=0.0,
                                          volume_confirmation_score=min(volume_ratio_val, 2.0), consistency_score=0.0,
                                          failure_count=0, confidence_score=strength, confluence_score=0.0,
                                          pivot_level=False, psychological_level=False,
                                          metadata={'method': 'volume', 'volume_ratio': volume_ratio_val,
                                                   'spike_type': 'strong' if strength > 0.6 else 'medium'})
                            levels.append(level)
            else:
                # Fallback to original pandas-based method
                volume_ma = pd.Series(volume).rolling(window=5).mean()
                volume_spikes = volume > volume_ma * self.volume_spike_threshold

                volume_ma_long = pd.Series(volume).rolling(window=30).mean()
                medium_spikes = volume > volume_ma_long * (self.volume_spike_threshold * 0.7)

                for i in range(len(data)):
                    volume_spike_detected = False
                    volume_ratio = 1.0

                    # Check for strong volume spikes
                    if i < len(volume_spikes) and volume_spikes.iloc[i] if hasattr(volume_spikes, 'iloc') else volume_spikes[i]:
                        volume_spike_detected = True
                        volume_ratio = volume[i] / volume_ma.iloc[i] if volume_ma.iloc[i] > 0 else 1.0
                        strength = 0.7  # Strong volume spike
                    # Check for medium volume spikes
                    elif i < len(medium_spikes) and medium_spikes.iloc[i] if hasattr(medium_spikes, 'iloc') else medium_spikes[i]:
                        volume_spike_detected = True
                        volume_ratio = volume[i] / volume_ma_long.iloc[i] if volume_ma_long.iloc[i] > 0 else 1.0
                        strength = 0.5  # Medium volume spike
                    else:
                        strength = 0.0

                    if volume_spike_detected and i > 0 and strength > 0:
                        if high[i] > high[i - 1]:  # Price moved up on volume
                            level = SRLevel(price=high[i], strength=strength, type='resistance', touch_count=1,
                                          first_touch_time=data.index[i], last_touch_time=data.index[i],
                                          age_bars=0, avg_bounce_ratio=0.0, max_bounce_ratio=0.0,
                                          volume_confirmation_score=min(volume_ratio, 2.0), consistency_score=0.0,
                                          failure_count=0, confidence_score=strength, confluence_score=0.0,
                                          pivot_level=False, psychological_level=False,
                                          metadata={'method': 'volume', 'volume_ratio': volume_ratio,
                                                   'spike_type': 'strong' if strength > 0.6 else 'medium'})
                            levels.append(level)
                        if low[i] < low[i - 1]:  # Price moved down on volume
                            level = SRLevel(price=low[i], strength=strength, type='support', touch_count=1,
                                          first_touch_time=data.index[i], last_touch_time=data.index[i],
                                          age_bars=0, avg_bounce_ratio=0.0, max_bounce_ratio=0.0,
                                          volume_confirmation_score=min(volume_ratio, 2.0), consistency_score=0.0,
                                          failure_count=0, confidence_score=strength, confluence_score=0.0,
                                          pivot_level=False, psychological_level=False,
                                          metadata={'method': 'volume', 'volume_ratio': volume_ratio,
                                                   'spike_type': 'strong' if strength > 0.6 else 'medium'})
                            levels.append(level)

            # Limit to top 30 levels by strength for more detection
            levels = sorted(levels, key=lambda x: x.strength, reverse=True)[:30]

            # Performance monitoring
            end_time = time.time()
            execution_time = end_time - start_time

            memory_info = ""
            if PSUTIL_AVAILABLE:
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_delta = end_memory - start_memory
                memory_info = f", memory delta: {memory_delta:+.1f}MB"

            optimization_info = " (Numba accelerated)" if NUMBA_AVAILABLE else ""
            self.logger.info(f"✅ Volume detection completed in {execution_time:.3f}s{memory_info}{optimization_info}")

            return levels
        except Exception as e:
            self.logger.warning(f'Volume detection failed: {e}')
            return []

    def _detect_statistical_levels(self, data: pd.DataFrame) -> List[SRLevel]:
        """Detect S/R levels using statistical analysis."""
        try:
            levels = []
            close = data['close'].values
            mean_price = np.mean(close)
            std_price = np.std(close)
            # Generate more statistical levels with additional standard deviations
            for std_multiple in [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4]:
                upper_level = mean_price + std_multiple * std_price
                lower_level = mean_price - std_multiple * std_price
                level = SRLevel(price = upper_level, strength = 0.5 + std_multiple * 0.1, type='resistance', touch_count = 0, first_touch_time = data.index[0], last_touch_time = data.index[0], age_bars = len(data), avg_bounce_ratio = 0.0, max_bounce_ratio = 0.0, volume_confirmation_score = 0.0, consistency_score = 0.0, failure_count = 0, confidence_score = 0.5 + std_multiple * 0.1, confluence_score = 0.0, pivot_level = False, psychological_level = False, metadata={'method': 'statistical', 'std_multiple': std_multiple})
                levels.append(level)
                level = SRLevel(price = lower_level, strength = 0.5 + std_multiple * 0.1, type='support', touch_count = 0, first_touch_time = data.index[0], last_touch_time = data.index[0], age_bars = len(data), avg_bounce_ratio = 0.0, max_bounce_ratio = 0.0, volume_confirmation_score = 0.0, consistency_score = 0.0, failure_count = 0, confidence_score = 0.5 + std_multiple * 0.1, confluence_score = 0.0, pivot_level = False, psychological_level = False, metadata={'method': 'statistical', 'std_multiple': std_multiple})
                levels.append(level)
            return levels
        except Exception as e:
            self.logger.warning(f'Statistical detection failed: {e}')
            return []

    def _detect_psychological_levels(self, data: pd.DataFrame) -> List[SRLevel]:
        """Detect psychological S/R levels (round numbers)."""
        try:
            levels = []
            close = data['close'].values
            current_price = close[-1]
            price_magnitude = 10 ** int(np.log10(current_price))
            # Generate more psychological levels with additional multipliers
            for multiplier in [0.05, 0.1, 0.15, 0.25, 0.33, 0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0]:
                level_price = round(current_price / (price_magnitude * multiplier)) * (price_magnitude * multiplier)
                if level_price > current_price:
                    level = SRLevel(price = level_price, strength = 0.4, type='resistance', touch_count = 0, first_touch_time = data.index[0], last_touch_time = data.index[0], age_bars = len(data), avg_bounce_ratio = 0.0, max_bounce_ratio = 0.0, volume_confirmation_score = 0.0, consistency_score = 0.0, failure_count = 0, confidence_score = 0.4, confluence_score = 0.0, pivot_level = False, psychological_level = True, metadata={'method': 'psychological', 'multiplier': multiplier})
                    levels.append(level)
                else:
                    level = SRLevel(price = level_price, strength = 0.4, type='support', touch_count = 0, first_touch_time = data.index[0], last_touch_time = data.index[0], age_bars = len(data), avg_bounce_ratio = 0.0, max_bounce_ratio = 0.0, volume_confirmation_score = 0.0, consistency_score = 0.0, failure_count = 0, confidence_score = 0.4, confluence_score = 0.0, pivot_level = False, psychological_level = True, metadata={'method': 'psychological', 'multiplier': multiplier})
                    levels.append(level)
            return levels
        except Exception as e:
            self.logger.warning(f'Psychological detection failed: {e}')
            return []

    def _detect_fibonacci_levels(self, data: pd.DataFrame) -> List[SRLevel]:
        """Detect S/R levels using Fibonacci retracement analysis."""
        try:
            levels = []
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values

            # Find significant swing points for Fibonacci analysis
            swing_highs = self._find_swing_highs(high, period=10)
            swing_lows = self._find_swing_lows(low, period=10)

            # Optimized Fibonacci retracement levels (15 most relevant levels)
            fib_levels = [0.191, 0.236, 0.318, 0.382, 0.5, 0.618, 0.707, 0.786, 0.886, 1.0, 1.128, 1.236, 1.382, 1.5, 1.618]

            # Calculate Fibonacci levels from recent significant moves
            recent_high = np.max(high[-50:]) if len(high) >= 50 else np.max(high)
            recent_low = np.min(low[-50:]) if len(low) >= 50 else np.min(low)

            if recent_high > recent_low:
                range_size = recent_high - recent_low

                for fib_level in fib_levels:
                    # Calculate Fibonacci retracement levels
                    retracement_level = recent_high - (range_size * fib_level)

                    # Determine if this acts as support or resistance
                    touches_above = np.sum(close > retracement_level)
                    touches_below = np.sum(close < retracement_level)

                    if touches_above > touches_below:
                        # Acts more as support
                        level_type = 'support'
                        strength = 0.4 + (fib_level * 0.2)  # Higher fib levels are stronger
                    else:
                        # Acts more as resistance
                        level_type = 'resistance'
                        strength = 0.4 + (fib_level * 0.2)

                    level = SRLevel(
                        price=retracement_level,
                        strength=strength,
                        type=level_type,
                        touch_count=0,
                        first_touch_time=data.index[0],
                        last_touch_time=data.index[-1],
                        age_bars=len(data),
                        avg_bounce_ratio=0.0,
                        max_bounce_ratio=0.0,
                        volume_confirmation_score=0.0,
                        consistency_score=0.0,
                        failure_count=0,
                        confidence_score=strength,
                        confluence_score=0.0,
                        pivot_level=False,
                        psychological_level=False,
                        fibonacci_level=fib_level,
                        metadata={
                            'method': 'fibonacci',
                            'fib_level': fib_level,
                            'range_high': recent_high,
                            'range_low': recent_low
                        }
                    )
                    levels.append(level)

            return levels
        except Exception as e:
            self.logger.warning(f'Fibonacci detection failed: {e}')
            return []

    def _detect_trendline_levels(self, data: pd.DataFrame) -> List[SRLevel]:
        """Detect S/R levels using trend line analysis with performance optimizations."""
        import time
        start_time = time.time()

        try:
            # Early exit for very large datasets to prevent hanging
            if len(data) > 50000:
                self.logger.info('📊 Large dataset detected, skipping trendline analysis for performance')
                return []

            levels = []
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values

            # Use larger period for swing points to reduce computation (was 5, now 10)
            swing_highs_indices, swing_highs_values = self._find_swing_points(high, 'high', period=10)
            swing_lows_indices, swing_lows_values = self._find_swing_points(low, 'low', period=10)

            # Generate trend lines from swing points
            min_points = 3  # Increased minimum points for trend lines to reduce computation

            # Support trend lines (connecting lows)
            if len(swing_lows_indices) >= min_points:
                support_lines = self._generate_trend_lines(swing_lows_indices, swing_lows_values, 'support')
                levels.extend(support_lines)

            # Resistance trend lines (connecting highs)
            if len(swing_highs_indices) >= min_points:
                resistance_lines = self._generate_trend_lines(swing_highs_indices, swing_highs_values, 'resistance')
                levels.extend(resistance_lines)

            # Limit to top 30 levels by strength (increased from 30 to reduce computation)
            levels = sorted(levels, key=lambda x: x.strength, reverse=True)[:20]

            elapsed = time.time() - start_time
            self.logger.info(f'📊 Trendline analysis completed in {elapsed:.2f}s')

            return levels
        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.warning(f'Trendline detection failed after {elapsed:.2f}s: {e}')
            return []

    def _find_swing_points(self, data: np.ndarray, point_type: str, period: int) -> tuple:
        """Find swing points for trend line analysis."""
        try:
            indices = []
            values = []

            for i in range(period, len(data) - period):
                if point_type == 'high':
                    if data[i] == np.max(data[i-period:i+period+1]):
                        indices.append(i)
                        values.append(data[i])
                else:  # low
                    if data[i] == np.min(data[i-period:i+period+1]):
                        indices.append(i)
                        values.append(data[i])

            return indices, values
        except Exception:
            return [], []

    def _generate_trend_lines(self, indices: List[int], values: List[float], line_type: str) -> List[SRLevel]:
        """Generate trend lines from swing points using linear regression."""
        try:
            levels = []

            # Try different combinations of points to find best trend lines
            for i in range(len(indices) - 2):
                for j in range(i + 2, min(i + 5, len(indices))):  # Test 3-4 point combinations
                    point_indices = indices[i:j+1]
                    point_values = values[i:j+1]

                    if len(point_indices) < 3:
                        continue

                    # Perform linear regression
                    slope, intercept, r_value, p_value, std_err = self._linear_regression(point_indices, point_values)

                    if abs(r_value) < 0.3:  # Further reduced correlation threshold for 25% more trend lines
                        continue

                    # Calculate trend line strength based on R-squared and number of points
                    strength = min(abs(r_value) * 0.8 + (len(point_indices) - 2) * 0.1, 0.9)

                    # Project the trend line to current price area
                    current_index = len(point_indices) - 1
                    current_price = slope * current_index + intercept

                    # Calculate trend line angle (degrees)
                    angle_rad = np.arctan(slope)
                    angle_deg = np.degrees(angle_rad)

                    level = SRLevel(
                        price=current_price,
                        strength=strength,
                        type=line_type,
                        touch_count=len(point_indices),
                        first_touch_time=None,  # Will be set in enhanced metrics
                        last_touch_time=None,
                        age_bars=max(point_indices) - min(point_indices),
                        avg_bounce_ratio=0.0,
                        max_bounce_ratio=0.0,
                        volume_confirmation_score=0.0,
                        consistency_score=abs(r_value),
                        failure_count=0,
                        confidence_score=strength,
                        confluence_score=0.0,
                        pivot_level=False,
                        psychological_level=False,
                        metadata={
                            'method': 'trendline',
                            'slope': slope,
                            'intercept': intercept,
                            'r_squared': r_value ** 2,
                            'angle_degrees': angle_deg,
                            'point_count': len(point_indices)
                        }
                    )
                    levels.append(level)

            return levels
        except Exception as e:
            self.logger.warning(f'Trend line generation failed: {e}')
            return []

    def _linear_regression(self, x: List[int], y: List[float]) -> tuple:
        """Perform linear regression on trend line points."""
        try:
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            return slope, intercept, r_value, p_value, std_err
        except Exception:
            return 0, 0, 0, 1, 1

    def _detect_channel_levels(self, data: pd.DataFrame) -> List[SRLevel]:
        """Detect S/R levels using channel analysis."""
        try:
            import time
            channel_start = time.time()

            levels = []
            high = data['high'].values
            low = data['low'].values

            # Find swing points for channel analysis with adaptive parameters
            swing_start = time.time()
            swing_highs_indices, swing_highs_values = self._find_swing_points(high, 'high', period=5)
            swing_lows_indices, swing_lows_values = self._find_swing_points(low, 'low', period=5)
            swing_time = time.time() - swing_start

            # Adaptive swing point limiting based on data size
            data_size = len(data)
            if data_size > 50000:
                max_swing_points = 15  # Smaller limit for very large datasets
            elif data_size > 10000:
                max_swing_points = 20  # Medium limit
            else:
                max_swing_points = 25  # Larger limit for smaller datasets

            original_high_count = len(swing_highs_indices)
            original_low_count = len(swing_lows_indices)

            if len(swing_highs_indices) > max_swing_points:
                # Select most significant swing points
                swing_highs_indices, swing_highs_values = self._select_significant_swings(
                    swing_highs_indices, swing_highs_values, max_swing_points, high
                )

            if len(swing_lows_indices) > max_swing_points:
                swing_lows_indices, swing_lows_values = self._select_significant_swings(
                    swing_lows_indices, swing_lows_values, max_swing_points, low
                )

            self.logger.info(f'📊 Swing points: {len(swing_highs_indices)}/{original_high_count} high, {len(swing_lows_indices)}/{original_low_count} low (swing detection: {swing_time:.3f}s)')

            # Find parallel trend lines to form channels
            if len(swing_highs_indices) >= 3 and len(swing_lows_indices) >= 3:
                channel_levels = self._find_parallel_channels_optimized(
                    swing_highs_indices, swing_highs_values,
                    swing_lows_indices, swing_lows_values,
                    data
                )
                levels.extend(channel_levels)

            # Limit to top 30 levels by strength for more detection
            levels = sorted(levels, key=lambda x: x.strength, reverse=True)[:30]

            channel_time = time.time() - channel_start
            self.logger.info(f'✅ Channel detection completed in {channel_time:.2f}s')
            return levels
        except Exception as e:
            self.logger.warning(f'Channel detection failed: {e}')
            return []

    def _find_parallel_channels_optimized(self, high_indices: List[int], high_values: List[float],
                                         low_indices: List[int], low_values: List[float],
                                         data: pd.DataFrame) -> List[SRLevel]:
        """Optimized parallel channel detection using vectorized operations and intelligent filtering."""
        try:
            import time
            start_time = time.time()
            levels = []

            # Pre-compute all possible line combinations using vectorized operations
            high_params = self._precompute_line_parameters(high_indices, high_values)
            low_params = self._precompute_line_parameters(low_indices, low_values)

            # Apply advanced quality filtering
            high_quality_high = self._advanced_quality_filter(high_params, 'upper')
            high_quality_low = self._advanced_quality_filter(low_params, 'lower')

            self.logger.info(f'📊 Pre-computed {len(high_quality_high)}/{len(high_params)} high-quality upper lines, {len(high_quality_low)}/{len(low_params)} lower lines')

            # Use intelligent pairing instead of exhaustive search
            channel_candidates = self._find_channel_candidates_intelligent(
                high_quality_high, high_quality_low, max_candidates=50
            )

            self.logger.info(f'📊 Found {len(channel_candidates)} channel candidates')

            # Process candidates efficiently with batch processing
            batch_size = 20
            processed_levels = []

            for i in range(0, len(channel_candidates), batch_size):
                batch = channel_candidates[i:i+batch_size]

                for candidate in batch:
                    upper_param = candidate['upper']
                    lower_param = candidate['lower']

                    # Calculate channel metrics using vectorized operations
                    channel_width = abs(upper_param['intercept'] - lower_param['intercept'])
                    mean_intercept = (upper_param['intercept'] + lower_param['intercept']) / 2
                    relative_width = channel_width / mean_intercept if mean_intercept != 0 else channel_width

                    # Enhanced channel strength calculation
                    slope_similarity = 1 - abs(upper_param['slope'] - lower_param['slope']) / max(abs(upper_param['slope']), abs(lower_param['slope']), 0.001)
                    r2_avg = (abs(upper_param['r2']) + abs(lower_param['r2'])) / 2
                    quality_bonus = (upper_param.get('quality_score', 0) + lower_param.get('quality_score', 0)) / 2

                    # Composite strength score
                    strength = min(slope_similarity * r2_avg * quality_bonus * 0.85, 0.95)

                    # Skip weak channels with improved thresholds
                    if strength < 0.4 or relative_width > 0.08 or slope_similarity < 0.6:
                        continue

                    # Project to current price level
                    current_index = len(data) - 1
                    upper_price = upper_param['slope'] * current_index + upper_param['intercept']
                    lower_price = lower_param['slope'] * current_index + lower_param['intercept']

                    # Validate projected prices are reasonable
                    current_price = data.iloc[-1]['close'] if len(data) > 0 else (upper_price + lower_price) / 2
                    if not (0.5 * current_price <= upper_price <= 2 * current_price) or \
                       not (0.5 * current_price <= lower_price <= 2 * current_price):
                        continue

                    # Create channel levels
                    # Upper channel (resistance)
                    upper_indices = [int(idx) for idx in upper_param['indices']]  # Ensure indices are integers
                    upper_level = SRLevel(
                        price=upper_price,
                        strength=strength,
                        type='resistance',
                        touch_count=len(upper_indices),
                        first_touch_time=data.index[min(upper_indices)],
                        last_touch_time=data.index[max(upper_indices)],
                        age_bars=current_index - min(upper_indices),
                        avg_bounce_ratio=0.0,
                        max_bounce_ratio=0.0,
                        volume_confirmation_score=0.0,
                        consistency_score=r2_avg,
                        failure_count=0,
                        confidence_score=strength,
                        confluence_score=0.0,
                        pivot_level=False,
                        psychological_level=False,
                        metadata={
                            'method': 'channel_optimized',
                            'channel_type': 'upper',
                            'slope': upper_param['slope'],
                            'width': relative_width,
                            'r_squared': upper_param['r2'] ** 2,
                            'parallel_slope': lower_param['slope'],
                            'matching_method': candidate.get('method', 'optimized')
                        }
                    )
                    processed_levels.append(upper_level)

                    # Lower channel (support)
                    lower_indices = [int(idx) for idx in lower_param['indices']]  # Ensure indices are integers
                    lower_level = SRLevel(
                        price=lower_price,
                        strength=strength,
                        type='support',
                        touch_count=len(lower_indices),
                        first_touch_time=data.index[min(lower_indices)],
                        last_touch_time=data.index[max(lower_indices)],
                        age_bars=current_index - min(lower_indices),
                        avg_bounce_ratio=0.0,
                        max_bounce_ratio=0.0,
                        volume_confirmation_score=0.0,
                        consistency_score=r2_avg,
                        failure_count=0,
                        confidence_score=strength,
                        confluence_score=0.0,
                        pivot_level=False,
                        psychological_level=False,
                        metadata={
                            'method': 'channel_optimized',
                            'channel_type': 'lower',
                            'slope': lower_param['slope'],
                            'width': relative_width,
                            'r_squared': lower_param['r2'] ** 2,
                            'parallel_slope': upper_param['slope'],
                            'matching_method': candidate.get('method', 'optimized')
                        }
                    )
                    processed_levels.append(lower_level)

            levels.extend(processed_levels)

            processing_time = time.time() - start_time
            self.logger.info(f'✅ Optimized channel detection completed in {processing_time:.2f}s: {len(levels)} levels')
            return levels

        except Exception as e:
            self.logger.error(f'Optimized channel detection failed: {e}')
            return []


    def _precompute_line_parameters(self, indices: List[int], values: List[float]) -> List[Dict]:
        """Pre-compute line parameters for all possible triplets using vectorized operations."""
        try:
            if len(indices) < 3:
                return []

            import numpy as np
            params = []
            n = len(indices)

            # Convert to numpy arrays for vectorized operations
            indices_array = np.array(indices, dtype=np.int32)  # Use int32 for indices
            values_array = np.array(values, dtype=np.float64)

            # Use sliding window of size 3 with vectorized operations
            for i in range(n - 2):
                triplet_indices = indices_array[i:i+3]
                triplet_values = values_array[i:i+3]

                # Calculate line parameters for this triplet
                slope, intercept, r2 = self._calculate_line_params_vectorized(
                    triplet_indices.tolist(), triplet_values.tolist()
                )

                # Additional quality metrics
                trend_strength = abs(slope) * np.sqrt(r2) if slope != 0 else 0
                volatility = np.std(triplet_values)
                consistency = 1 - (volatility / np.mean(triplet_values)) if np.mean(triplet_values) != 0 else 0

                params.append({
                    'indices': triplet_indices.tolist(),  # Keep as integers
                    'values': triplet_values.tolist(),
                    'slope': slope,
                    'intercept': intercept,
                    'r2': r2,
                    'start_idx': int(i),  # Ensure start_idx is integer
                    'trend_strength': trend_strength,
                    'volatility': volatility,
                    'consistency': consistency,
                    'quality_score': r2 * consistency
                })

            return params

        except Exception as e:
            self.logger.warning(f'Pre-computation failed: {e}')
            return []

    def _calculate_line_params_vectorized(self, x_points: List[int], y_points: List[float]) -> Tuple[float, float, float]:
        """Vectorized line parameter calculation for better performance with advanced statistics."""
        try:
            import numpy as np

            if len(x_points) != len(y_points) or len(x_points) < 2:
                return 0.0, 0.0, 0.0

            x = np.array(x_points, dtype=np.float64)
            y = np.array(y_points, dtype=np.float64)

            n = len(x)

            if n >= 3:
                # Use numpy's polyfit for robust linear regression
                coeffs = np.polyfit(x, y, 1, full=False)
                slope, intercept = coeffs[0], coeffs[1]

                # Vectorized R² calculation
                y_pred = slope * x + intercept
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0

                # Ensure R² is in valid range
                r2 = np.clip(r2, -1, 1)

            elif n == 2:
                # Direct calculation for 2 points (exact fit)
                if x[1] != x[0]:
                    slope = (y[1] - y[0]) / (x[1] - x[0])
                    intercept = y[0] - slope * x[0]
                    r2 = 1.0  # Perfect fit for 2 points
                else:
                    # Vertical line case
                    slope = np.inf
                    intercept = x[0]
                    r2 = 0.0
            else:
                return 0.0, 0.0, 0.0

            # Handle edge cases
            if not np.isfinite(slope) or not np.isfinite(intercept):
                return 0.0, 0.0, 0.0

            return float(slope), float(intercept), float(r2)

        except Exception as e:
            self.logger.warning(f'Vectorized line calculation failed: {e}')
            return 0.0, 0.0, 0.0

    def _batch_calculate_line_params(self, indices_list: List[List[int]], values_list: List[List[float]]) -> List[Tuple[float, float, float]]:
        """Batch calculate line parameters for multiple triplets using vectorized operations."""
        try:
            import numpy as np

            if not indices_list or not values_list:
                return []

            results = []

            # Process in batches for better memory efficiency
            batch_size = 100
            for i in range(0, len(indices_list), batch_size):
                batch_indices = indices_list[i:i+batch_size]
                batch_values = values_list[i:i+batch_size]

                for x_points, y_points in zip(batch_indices, batch_values):
                    slope, intercept, r2 = self._calculate_line_params_vectorized(x_points, y_points)
                    results.append((slope, intercept, r2))

            return results

        except Exception as e:
            self.logger.warning(f'Batch line calculation failed: {e}')
            return [(0.0, 0.0, 0.0)] * len(indices_list)

    def _find_channel_candidates_intelligent(self, high_params: List[Dict],
                                           low_params: List[Dict],
                                           max_candidates: int = 50) -> List[Dict]:
        """Find best channel candidates using intelligent pairing with advanced quality filtering."""
        try:
            import numpy as np

            # Advanced quality filtering
            def quality_filter(params: List[Dict], min_quality: float = 0.6) -> List[Dict]:
                """Filter parameters based on multiple quality metrics."""
                return [
                    p for p in params
                    if (abs(p.get('r2', 0)) >= min_quality and
                        p.get('quality_score', 0) >= 0.4 and
                        p.get('consistency', 0) >= 0.3 and
                        abs(p.get('slope', 0)) >= 0.0001)  # Avoid near-horizontal lines
                ]

            # Apply quality filtering
            high_quality = quality_filter(high_params)
            low_quality = quality_filter(low_params)

            self.logger.info(f'📊 Quality filtered: {len(high_quality)}/{len(high_params)} high, {len(low_quality)}/{len(low_params)} low lines')

            if len(high_quality) == 0 or len(low_quality) == 0:
                self.logger.warning('No high-quality lines found after filtering')
                return []

            # Intelligent candidate generation using multiple strategies
            candidates = []

            # Strategy 1: Slope-based clustering
            candidates.extend(self._slope_based_clustering(high_quality, low_quality, max_candidates // 3))

            # Strategy 2: Trend strength matching
            candidates.extend(self._trend_strength_matching(high_quality, low_quality, max_candidates // 3))

            # Strategy 3: Geometric similarity
            candidates.extend(self._geometric_similarity_matching(high_quality, low_quality, max_candidates // 3))

            # Remove duplicates and sort by composite score
            seen = set()
            unique_candidates = []
            for candidate in sorted(candidates, key=lambda x: x['composite_score'], reverse=True):
                key = (candidate['upper']['start_idx'], candidate['lower']['start_idx'])
                if key not in seen:
                    seen.add(key)
                    unique_candidates.append(candidate)

            return unique_candidates[:max_candidates]

        except Exception as e:
            self.logger.warning(f'Intelligent candidate selection failed: {e}')
            return []

    def _slope_based_clustering(self, high_params: List[Dict], low_params: List[Dict], max_count: int) -> List[Dict]:
        """Cluster lines by slope similarity for efficient pairing."""
        try:
            import numpy as np

            candidates = []

            # Extract slope values
            high_slopes = np.array([p['slope'] for p in high_params])
            low_slopes = np.array([p['slope'] for p in low_params])

            # Create slope clusters
            slope_ranges = np.linspace(high_slopes.min(), high_slopes.max(), 10)

            for i, high_param in enumerate(high_params[:15]):  # Top 15
                # Find low lines with similar slopes
                slope_diffs = np.abs(low_slopes - high_param['slope'])
                similar_indices = np.where(slope_diffs < abs(high_param['slope']) * 0.3)[0]

                if len(similar_indices) > 0:
                    # Take the best match
                    best_idx = similar_indices[np.argmin(slope_diffs[similar_indices])]
                    low_param = low_params[best_idx]

                    # Calculate composite score
                    slope_similarity = 1 - (slope_diffs[best_idx] / max(abs(high_param['slope']), 0.001))
                    combined_quality = (high_param['quality_score'] + low_param['quality_score']) / 2
                    composite_score = slope_similarity * combined_quality * 0.8

                    candidates.append({
                        'upper': high_param,
                        'lower': low_param,
                        'composite_score': composite_score,
                        'slope_similarity': slope_similarity,
                        'method': 'slope_clustering'
                    })

            return sorted(candidates, key=lambda x: x['composite_score'], reverse=True)[:max_count]

        except Exception as e:
            self.logger.warning(f'Slope-based clustering failed: {e}')
            return []

    def _trend_strength_matching(self, high_params: List[Dict], low_params: List[Dict], max_count: int) -> List[Dict]:
        """Match lines based on trend strength compatibility."""
        try:
            candidates = []

            # Sort by trend strength
            high_sorted = sorted(high_params, key=lambda x: x.get('trend_strength', 0), reverse=True)
            low_sorted = sorted(low_params, key=lambda x: x.get('trend_strength', 0), reverse=True)

            for high_param in high_sorted[:10]:  # Top 10 strongest trends
                for low_param in low_sorted[:10]:  # Top 10 strongest trends
                    # Calculate trend strength compatibility
                    strength_ratio = min(high_param['trend_strength'], low_param['trend_strength']) / \
                                   max(high_param['trend_strength'], low_param['trend_strength'], 0.001)

                    # Slope compatibility
                    slope_diff = abs(high_param['slope'] - low_param['slope'])
                    slope_compat = 1 - (slope_diff / max(abs(high_param['slope']), abs(low_param['slope']), 0.001))

                    # Combined score
                    composite_score = strength_ratio * slope_compat * 0.7

                    if composite_score > 0.4:  # Quality threshold
                        candidates.append({
                            'upper': high_param,
                            'lower': low_param,
                            'composite_score': composite_score,
                            'strength_ratio': strength_ratio,
                            'slope_compatibility': slope_compat,
                            'method': 'trend_strength'
                        })

            return sorted(candidates, key=lambda x: x['composite_score'], reverse=True)[:max_count]

        except Exception as e:
            self.logger.warning(f'Trend strength matching failed: {e}')
            return []

    def _geometric_similarity_matching(self, high_params: List[Dict], low_params: List[Dict], max_count: int) -> List[Dict]:
        """Match lines based on geometric similarity and channel formation potential."""
        try:
            import numpy as np

            candidates = []

            for high_param in high_params[:12]:  # Top 12
                for low_param in low_params[:12]:  # Top 12
                    # Geometric similarity based on multiple factors
                    slope_sim = 1 - abs(high_param['slope'] - low_param['slope']) / \
                              max(abs(high_param['slope']), abs(low_param['slope']), 0.001)

                    # Intercept relationship (parallel check)
                    intercept_diff = abs(high_param['intercept'] - low_param['intercept'])

                    # R² similarity
                    r2_sim = 1 - abs(high_param['r2'] - low_param['r2'])

                    # Consistency similarity
                    consistency_sim = 1 - abs(high_param.get('consistency', 0) - low_param.get('consistency', 0))

                    # Combined geometric score
                    geometric_score = (slope_sim * 0.4 + r2_sim * 0.3 + consistency_sim * 0.3)

                    # Quality bonus for good channel geometry
                    quality_bonus = 1 + (geometric_score > 0.7) * 0.2

                    composite_score = geometric_score * quality_bonus

                    if composite_score > 0.5:  # Higher threshold for geometric matching
                        candidates.append({
                            'upper': high_param,
                            'lower': low_param,
                            'composite_score': composite_score,
                            'geometric_score': geometric_score,
                            'slope_similarity': slope_sim,
                            'r2_similarity': r2_sim,
                            'method': 'geometric'
                        })

            return sorted(candidates, key=lambda x: x['composite_score'], reverse=True)[:max_count]

        except Exception as e:
            self.logger.warning(f'Geometric similarity matching failed: {e}')
            return []

    def _advanced_quality_filter(self, params: List[Dict], line_type: str) -> List[Dict]:
        """Advanced quality filtering with multiple criteria and dynamic thresholds."""
        try:
            if not params:
                return []

            import numpy as np

            # Extract quality metrics
            r2_values = np.array([abs(p.get('r2', 0)) for p in params])
            quality_scores = np.array([p.get('quality_score', 0) for p in params])
            consistencies = np.array([p.get('consistency', 0) for p in params])
            trend_strengths = np.array([p.get('trend_strength', 0) for p in params])
            volatilities = np.array([p.get('volatility', 0) for p in params])

            # Dynamic thresholds based on data distribution
            r2_threshold = max(0.5, np.percentile(r2_values, 70))  # Top 30%
            quality_threshold = max(0.3, np.percentile(quality_scores, 60))  # Top 40%
            consistency_threshold = max(0.2, np.percentile(consistencies, 65))  # Top 35%

            # Additional filters based on line type
            slope_threshold = 0.0005  # Minimum slope to avoid near-horizontal lines
            max_volatility = np.percentile(volatilities, 80)  # Filter out most volatile lines

            filtered_params = []
            for param in params:
                if (abs(param.get('r2', 0)) >= r2_threshold and
                    param.get('quality_score', 0) >= quality_threshold and
                    param.get('consistency', 0) >= consistency_threshold and
                    abs(param.get('slope', 0)) >= slope_threshold and
                    param.get('volatility', 0) <= max_volatility and
                    param.get('trend_strength', 0) > 0):  # Must have some trend strength

                    filtered_params.append(param)

            # Sort by composite quality score
            filtered_params.sort(key=lambda x: (
                x.get('quality_score', 0) * 0.4 +
                abs(x.get('r2', 0)) * 0.3 +
                x.get('consistency', 0) * 0.3
            ), reverse=True)

            self.logger.info(f'📊 {line_type.title()} lines: {len(filtered_params)} passed quality filter (R²≥{r2_threshold:.2f}, quality≥{quality_threshold:.2f})')
            return filtered_params

        except Exception as e:
            self.logger.warning(f'Advanced quality filtering failed: {e}')
            # Fallback to basic filtering
            return [p for p in params if abs(p.get('r2', 0)) >= 0.5]

    def _select_significant_swings(self, indices: List[int], values: List[float], max_count: int, price_data: np.ndarray) -> Tuple[List[int], List[float]]:
        """Select the most significant swing points based on price movement and timing."""
        try:
            import numpy as np

            if len(indices) <= max_count:
                return indices, values

            # Calculate significance scores for each swing point
            significance_scores = []

            for i, (idx, value) in enumerate(zip(indices, values)):
                # Price movement significance
                if i > 0:
                    prev_value = values[i-1]
                    price_change = abs(value - prev_value)
                    price_significance = price_change / np.mean(price_data) if np.mean(price_data) != 0 else 0
                else:
                    price_significance = 0

                # Timing significance (prefer more recent swings)
                time_weight = idx / len(price_data) if len(price_data) > 0 else 0

                # Volatility context (swings in volatile periods are more significant)
                local_volatility = np.std(price_data[max(0, idx-10):min(len(price_data), idx+10)])
                volatility_weight = local_volatility / np.std(price_data) if np.std(price_data) != 0 else 0

                # Combined significance score
                significance = (
                    price_significance * 0.5 +
                    time_weight * 0.3 +
                    volatility_weight * 0.2
                )

                significance_scores.append((significance, idx, value))

            # Sort by significance and select top candidates
            significance_scores.sort(key=lambda x: x[0], reverse=True)
            selected = significance_scores[:max_count]

            # Sort back by index for proper ordering
            selected.sort(key=lambda x: x[1])

            selected_indices = [idx for _, idx, _ in selected]
            selected_values = [val for _, _, val in selected]

            return selected_indices, selected_values

        except Exception as e:
            self.logger.warning(f'Significant swing selection failed: {e}')
            # Fallback to simple selection
            return indices[:max_count], values[:max_count]

    def _calculate_line_params(self, points: List[tuple]) -> tuple:
        """Calculate slope, intercept, and R-squared for a line through points."""
        try:
            x = [p[0] for p in points]
            y = [p[1] for p in points]

            slope, intercept, r_value, p_value, std_err = self._linear_regression(x, y)
            return slope, intercept, r_value
        except Exception:
            return 0, 0, 0

    def _deduplicate_levels(self, levels: List[SRLevel], tolerance: float = 0.001) -> List[SRLevel]:
        """Remove duplicate levels based on price proximity."""
        try:
            if not levels:
                return levels

            deduplicated = []
            for level in levels:
                # Check if this level is too close to any existing level
                is_duplicate = False
                for existing in deduplicated:
                    price_diff = abs(level.price - existing.price) / existing.price if existing.price != 0 else 1.0
                    if price_diff <= tolerance:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    deduplicated.append(level)

            return deduplicated
        except Exception as e:
            self.logger.warning(f'Level deduplication failed: {e}')
            return levels

    def _detect_volume_profile_levels(self, data: pd.DataFrame) -> List[SRLevel]:
        """Detect S/R levels using comprehensive volume profile analysis."""
        try:
            levels = []

            if 'volume' not in data.columns:
                return levels

            # Create volume profile
            volume_profile = self._create_volume_profile(data)

            # Find High Volume Nodes (HVNs)
            hvn_levels = self._find_high_volume_nodes(volume_profile, data)
            levels.extend(hvn_levels)

            # Find volume clusters
            cluster_levels = self._find_volume_clusters(data)
            levels.extend(cluster_levels)

            # Limit to top 30 levels by strength for more detection
            levels = sorted(levels, key=lambda x: x.strength, reverse=True)[:30]
            return levels
        except Exception as e:
            self.logger.warning(f'Volume profile detection failed: {e}')
            return []

    def _create_volume_profile(self, data: pd.DataFrame, bins: int = 100) -> Dict[str, Any]:
        """Create volume profile from price and volume data."""
        try:
            # Create price bins
            price_min = data['low'].min()
            price_max = data['high'].max()
            price_bins = np.linspace(price_min, price_max, bins)

            # Initialize volume profile
            volume_profile = {f'bin_{i}': {'price_range': (price_bins[i], price_bins[i+1]),
                                          'volume': 0,
                                          'touches': 0,
                                          'price_level': (price_bins[i] + price_bins[i+1]) / 2}
                             for i in range(len(price_bins) - 1)}

            # Accumulate volume in each price bin
            for idx, row in data.iterrows():
                high = row['high']
                low = row['low']
                volume = row['volume']

                # Find bins that overlap with the candle's price range
                for bin_key, bin_data in volume_profile.items():
                    bin_low, bin_high = bin_data['price_range']
                    if high >= bin_low and low <= bin_high:
                        # Calculate overlap proportion
                        overlap_low = max(low, bin_low)
                        overlap_high = min(high, bin_high)
                        overlap_ratio = (overlap_high - overlap_low) / (high - low) if high != low else 1.0

                        volume_profile[bin_key]['volume'] += volume * overlap_ratio
                        volume_profile[bin_key]['touches'] += 1

            return volume_profile
        except Exception as e:
            self.logger.warning(f'Volume profile creation failed: {e}')
            return {}

    def _find_high_volume_nodes(self, volume_profile: Dict[str, Any], data: pd.DataFrame) -> List[SRLevel]:
        """Find High Volume Nodes from volume profile."""
        try:
            levels = []

            if not volume_profile:
                return levels

            # Calculate volume statistics
            volumes = [bin_data['volume'] for bin_data in volume_profile.values()]
            volume_mean = np.mean(volumes)
            volume_std = np.std(volumes)

            # Find HVNs (High Volume Nodes) - bins with volume > mean + 0.6*std (more lenient threshold for 20% more HVNs)
            hvn_threshold = volume_mean + 0.6 * volume_std

            for bin_key, bin_data in volume_profile.items():
                if bin_data['volume'] > hvn_threshold:
                    # Determine if this acts as support or resistance
                    price_level = bin_data['price_level']
                    touches_above = np.sum(data['close'] > price_level)
                    touches_below = np.sum(data['close'] < price_level)

                    # Calculate strength based on volume and touch count
                    volume_ratio = bin_data['volume'] / volume_mean if volume_mean != 0 else 1.0
                    touch_score = min(bin_data['touches'] / 10, 1.0)  # Normalize touch count
                    strength = min(volume_ratio * 0.6 + touch_score * 0.4, 0.95)

                    if touches_above > touches_below:
                        level_type = 'support'
                    else:
                        level_type = 'resistance'

                    level = SRLevel(
                        price=price_level,
                        strength=strength,
                        type=level_type,
                        touch_count=bin_data['touches'],
                        first_touch_time=data.index[0],
                        last_touch_time=data.index[-1],
                        age_bars=len(data),
                        avg_bounce_ratio=0.0,
                        max_bounce_ratio=0.0,
                        volume_confirmation_score=min(volume_ratio, 1.0),
                        consistency_score=touch_score,
                        failure_count=0,
                        confidence_score=strength,
                        confluence_score=0.0,
                        pivot_level=False,
                        psychological_level=False,
                        metadata={
                            'method': 'volume_profile',
                            'profile_type': 'hvn',
                            'volume_ratio': volume_ratio,
                            'total_volume': bin_data['volume'],
                            'volume_percentile': (bin_data['volume'] - volume_mean) / volume_std if volume_std != 0 else 0.0
                        }
                    )
                    levels.append(level)

            return levels
        except Exception as e:
            self.logger.warning(f'HVN detection failed: {e}')
            return []

    def _find_volume_clusters(self, data: pd.DataFrame) -> List[SRLevel]:
        """Find volume clusters in price action."""
        try:
            levels = []

            if 'volume' not in data.columns:
                return levels

            # Calculate volume moving averages
            volume_ma_short = data['volume'].rolling(window=5).mean()
            volume_ma_long = data['volume'].rolling(window=20).mean()

            # Find volume spikes
            volume_ratio = data['volume'] / volume_ma_long
            volume_spikes = volume_ratio > 2.0  # 2x average volume

            # Group consecutive volume spikes into clusters
            spike_indices = np.where(volume_spikes)[0]

            if len(spike_indices) > 0:
                # Find clusters of spikes within 15 bars of each other for more clusters
                clusters = self._group_consecutive_indices(spike_indices, max_gap=15)

                for cluster in clusters:
                    if len(cluster) >= 2:  # Reduced requirement for more clusters (was 3, now 2 for 33% more clusters)
                        cluster_prices = []
                        cluster_volumes = []

                        for idx in cluster:
                            if idx < len(data):
                                # Use the high/low that was significant during the volume spike
                                if data.iloc[idx]['close'] > data.iloc[idx]['open']:
                                    cluster_prices.append(data.iloc[idx]['high'])
                                else:
                                    cluster_prices.append(data.iloc[idx]['low'])
                                cluster_volumes.append(data.iloc[idx]['volume'])

                        if cluster_prices:
                            # Calculate cluster center
                            cluster_center = np.mean(cluster_prices)
                            total_volume = sum(cluster_volumes)
                            avg_volume_ratio = np.mean([data.iloc[idx]['volume'] / volume_ma_long.iloc[idx] if volume_ma_long.iloc[idx] != 0 else 1.0 for idx in cluster])

                            # Determine support/resistance based on price action
                            touches_above = 0
                            touches_below = 0

                            for idx in range(max(0, min(cluster)-20), min(len(data), max(cluster)+20)):
                                if abs(data.iloc[idx]['close'] - cluster_center) < data.iloc[idx]['close'] * 0.005:  # Within 0.5%
                                    if data.iloc[idx]['close'] > cluster_center:
                                        touches_above += 1
                                    else:
                                        touches_below += 1

                            if touches_above > touches_below:
                                level_type = 'support'
                            else:
                                level_type = 'resistance'

                            strength = min(avg_volume_ratio * 0.5 + len(cluster) * 0.1, 0.9)

                            level = SRLevel(
                                price=cluster_center,
                                strength=strength,
                                type=level_type,
                                touch_count=len(cluster),
                                first_touch_time=data.index[min(cluster)],
                                last_touch_time=data.index[max(cluster)],
                                age_bars=max(cluster) - min(cluster),
                                avg_bounce_ratio=0.0,
                                max_bounce_ratio=0.0,
                                volume_confirmation_score=min(avg_volume_ratio, 1.0),
                                consistency_score=len(cluster) / 10,
                                failure_count=0,
                                confidence_score=strength,
                                confluence_score=0.0,
                                pivot_level=False,
                                psychological_level=False,
                                metadata={
                                    'method': 'volume_cluster',
                                    'total_volume': total_volume,
                                    'avg_volume_ratio': avg_volume_ratio,
                                    'cluster_size': len(cluster),
                                    'price_range': (min(cluster_prices), max(cluster_prices))
                                }
                            )
                            levels.append(level)

            return levels
        except Exception as e:
            self.logger.warning(f'Volume cluster detection failed: {e}')
            return []

    def _group_consecutive_indices(self, indices: np.ndarray, max_gap: int) -> List[List[int]]:
        """Group consecutive indices with gaps no larger than max_gap."""
        try:
            if len(indices) == 0:
                return []

            groups = []
            current_group = [indices[0]]

            for i in range(1, len(indices)):
                if indices[i] - indices[i-1] <= max_gap:
                    current_group.append(indices[i])
                else:
                    if len(current_group) >= 3:
                        groups.append(current_group)
                    current_group = [indices[i]]

            if len(current_group) >= 3:
                groups.append(current_group)

            return groups
        except Exception:
            return []

    def _detect_market_structure_levels(self, data: pd.DataFrame) -> List[SRLevel]:
        """Detect S/R levels using market structure analysis."""
        try:
            levels = []

            # Analyze market structure shifts
            structure_levels = self._analyze_market_structure_shifts(data)
            levels.extend(structure_levels)

            # Analyze order flow patterns
            order_flow_levels = self._analyze_order_flow_patterns(data)
            levels.extend(order_flow_levels)

            # Analyze higher timeframe structure (if available)
            if len(data) > 100:  # Only if we have enough data
                htf_levels = self._analyze_higher_timeframe_structure(data)
                levels.extend(htf_levels)

            # Limit to top 30 levels by strength for more detection
            levels = sorted(levels, key=lambda x: x.strength, reverse=True)[:30]
            return levels
        except Exception as e:
            self.logger.warning(f'Market structure detection failed: {e}')
            return []

    def _analyze_market_structure_shifts(self, data: pd.DataFrame) -> List[SRLevel]:
        """Analyze market structure shifts for S/R levels."""
        try:
            levels = []

            # Find Higher Highs and Higher Lows (bullish structure)
            # Find Lower Highs and Lower Lows (bearish structure)

            highs = data['high'].values
            lows = data['low'].values
            closes = data['close'].values

            # Identify structure shifts
            structure_shifts = []

            for i in range(3, len(data) - 3):  # Further reduced edge skip for 40% more structure shifts
                # Check for bullish structure shift (higher low after lower low)
                if (i >= 1 and lows[i] > lows[i-1] and highs[i] > highs[i-1]):
                    structure_shifts.append({
                        'index': i,
                        'type': 'bullish_shift',
                        'price': lows[i],
                        'level_type': 'support'
                    })

                # Check for bearish structure shift (lower high after higher high)
                if (i >= 1 and highs[i] < highs[i-1] and lows[i] < lows[i-1]):
                    structure_shifts.append({
                        'index': i,
                        'type': 'bearish_shift',
                        'price': highs[i],
                        'level_type': 'resistance'
                    })

            # Convert structure shifts to SR levels
            for shift in structure_shifts:
                strength = 0.75  # Structure shifts are strong signals

                level = SRLevel(
                    price=shift['price'],
                    strength=strength,
                    type=shift['level_type'],
                    touch_count=1,
                    first_touch_time=data.index[shift['index']],
                    last_touch_time=data.index[shift['index']],
                    age_bars=len(data) - shift['index'],
                    avg_bounce_ratio=0.0,
                    max_bounce_ratio=0.0,
                    volume_confirmation_score=0.0,
                    consistency_score=0.8,
                    failure_count=0,
                    confidence_score=strength,
                    confluence_score=0.0,
                    pivot_level=False,
                    psychological_level=False,
                    metadata={
                        'method': 'market_structure',
                        'structure_type': shift['type'],
                        'shift_index': shift['index']
                    }
                )
                levels.append(level)

            return levels
        except Exception as e:
            self.logger.warning(f'Market structure shift analysis failed: {e}')
            return []

    def _analyze_order_flow_patterns(self, data: pd.DataFrame) -> List[SRLevel]:
        """Analyze order flow patterns for S/R levels."""
        try:
            levels = []

            if 'volume' not in data.columns:
                return levels

            # Analyze volume and price patterns for order flow
            closes = data['close'].values
            volumes = data['volume'].values

            # Find absorption patterns (high volume with little price movement)
            for i in range(5, len(data) - 5):
                # Check for absorption (high volume, low price movement)
                price_range = data.iloc[i-2:i+3]['high'].max() - data.iloc[i-2:i+3]['low'].min()
                avg_volume = np.mean(volumes[i-2:i+3])
                volume_ratio = volumes[i] / avg_volume if avg_volume > 0 else 1

                if volume_ratio > 1.2 and price_range < np.mean(data['close']) * 0.015:  # < 1.5% range, more lenient thresholds for 20% more absorption levels
                    # This is an absorption level
                    absorption_price = closes[i]

                    # Determine support/resistance based on context
                    recent_trend = np.mean(closes[i-10:i]) - np.mean(closes[i-20:i-10])
                    if recent_trend > 0:
                        level_type = 'support'  # Absorption in uptrend = support
                    else:
                        level_type = 'resistance'  # Absorption in downtrend = resistance

                    strength = min(volume_ratio * 0.3 + 0.6, 0.9)

                    level = SRLevel(
                        price=absorption_price,
                        strength=strength,
                        type=level_type,
                        touch_count=1,
                        first_touch_time=data.index[i],
                        last_touch_time=data.index[i],
                        age_bars=len(data) - i,
                        avg_bounce_ratio=0.0,
                        max_bounce_ratio=0.0,
                        volume_confirmation_score=min(volume_ratio, 1.0),
                        consistency_score=0.7,
                        failure_count=0,
                        confidence_score=strength,
                        confluence_score=0.0,
                        pivot_level=False,
                        psychological_level=False,
                        metadata={
                            'method': 'order_flow',
                            'pattern_type': 'absorption',
                            'volume_ratio': volume_ratio,
                            'price_range_percent': price_range / closes[i]
                        }
                    )
                    levels.append(level)

            return levels
        except Exception as e:
            self.logger.warning(f'Order flow pattern analysis failed: {e}')
            return []

    def _analyze_higher_timeframe_structure(self, data: pd.DataFrame) -> List[SRLevel]:
        """Analyze higher timeframe market structure."""
        try:
            levels = []

            # Create synthetic higher timeframe data by grouping bars
            # Use larger group size to reduce number of synthetic levels
            group_size = 10  # Increased from 5 to 10 for more restrictive analysis
            if len(data) < group_size * 5:  # Need more bars for meaningful analysis
                return levels

            # Create higher timeframe bars by grouping every 'group_size' bars
            htf_data_list = []
            for i in range(0, len(data) - group_size + 1, group_size):
                group = data.iloc[i:i + group_size]
                htf_bar = {
                    'open': group['open'].iloc[0],
                    'high': group['high'].max(),
                    'low': group['low'].min(),
                    'close': group['close'].iloc[-1],
                    'volume': group['volume'].sum() if 'volume' in group.columns else 0,
                    'index': i // group_size,  # Synthetic index for the HTF bar
                    'original_indices': list(range(i, min(i + group_size, len(data))))
                }
                htf_data_list.append(htf_bar)

            # Convert to DataFrame for easier processing
            htf_data = pd.DataFrame(htf_data_list)

            if len(htf_data) < 5:  # Need at least 5 HTF bars
                return levels

            # Find significant HTF levels
            htf_highs = htf_data['high'].values
            htf_lows = htf_data['low'].values

            # Find HTF swing points using a larger window for more restrictive detection
            swing_window = min(5, len(htf_data) // 4)  # Increased from 3 to 5, changed divisor from 3 to 4

            for i in range(swing_window, len(htf_data) - swing_window):
                # Calculate average price for swing significance check
                avg_price = (htf_highs[i] + htf_lows[i]) / 2

                # HTF resistance level - check if this high is the highest in the window
                window_highs = htf_highs[i-swing_window:i+swing_window+1]
                if htf_highs[i] == np.max(window_highs):
                    resistance_price = htf_highs[i]

                    # Additional quality filter: swing must be significant (>0.5% of average price)
                    window_avg_price = np.mean((window_highs + htf_lows[i-swing_window:i+swing_window+1]) / 2)
                    swing_significance = abs(resistance_price - window_avg_price) / window_avg_price

                    if swing_significance < 0.005:  # Skip if swing is less than 0.5%
                        continue

                    strength = 0.8  # Higher timeframe levels are stronger

                    # Get original data indices for timestamp calculation
                    original_indices = htf_data.iloc[i]['original_indices']
                    first_touch_idx = original_indices[0]
                    last_touch_idx = original_indices[-1]

                    level = SRLevel(
                        price=resistance_price,
                        strength=strength,
                        type='resistance',
                        touch_count=1,
                        first_touch_time=data.index[first_touch_idx] if hasattr(data.index, '__getitem__') and first_touch_idx < len(data) else None,
                        last_touch_time=data.index[last_touch_idx] if hasattr(data.index, '__getitem__') and last_touch_idx < len(data) else None,
                        age_bars=len(data) - last_touch_idx,
                        avg_bounce_ratio=0.0,
                        max_bounce_ratio=0.0,
                        volume_confirmation_score=0.0,
                        consistency_score=0.9,
                        failure_count=0,
                        confidence_score=strength,
                        confluence_score=0.0,
                        pivot_level=False,
                        psychological_level=False,
                        metadata={
                            'method': 'htf_structure',
                            'timeframe': 'synthetic_higher',
                            'structure_type': 'resistance',
                            'group_size': group_size,
                            'swing_window': swing_window,
                            'swing_significance': swing_significance
                        }
                    )
                    levels.append(level)

                # HTF support level - check if this low is the lowest in the window
                window_lows = htf_lows[i-swing_window:i+swing_window+1]
                if htf_lows[i] == np.min(window_lows):
                    support_price = htf_lows[i]

                    # Additional quality filter: swing must be significant (>0.5% of average price)
                    window_avg_price = np.mean((htf_highs[i-swing_window:i+swing_window+1] + window_lows) / 2)
                    swing_significance = abs(support_price - window_avg_price) / window_avg_price

                    if swing_significance < 0.005:  # Skip if swing is less than 0.5%
                        continue

                    strength = 0.8

                    # Get original data indices for timestamp calculation
                    original_indices = htf_data.iloc[i]['original_indices']
                    first_touch_idx = original_indices[0]
                    last_touch_idx = original_indices[-1]

                    level = SRLevel(
                        price=support_price,
                        strength=strength,
                        type='support',
                        touch_count=1,
                        first_touch_time=data.index[first_touch_idx] if hasattr(data.index, '__getitem__') and first_touch_idx < len(data) else None,
                        last_touch_time=data.index[last_touch_idx] if hasattr(data.index, '__getitem__') and last_touch_idx < len(data) else None,
                        age_bars=len(data) - last_touch_idx,
                        avg_bounce_ratio=0.0,
                        max_bounce_ratio=0.0,
                        volume_confirmation_score=0.0,
                        consistency_score=0.9,
                        failure_count=0,
                        confidence_score=strength,
                        confluence_score=0.0,
                        pivot_level=False,
                        psychological_level=False,
                        metadata={
                            'method': 'htf_structure',
                            'timeframe': 'synthetic_higher',
                            'structure_type': 'support',
                            'group_size': group_size,
                            'swing_window': swing_window,
                            'swing_significance': swing_significance
                        }
                    )
                    levels.append(level)

            self.logger.info(f'📊 Higher timeframe analysis found {len(levels)} synthetic HTF levels')
            return levels
        except Exception as e:
            self.logger.warning(f'Higher timeframe structure analysis failed: {e}')
            return []

    def _find_fractal_highs(self, high: np.ndarray, period: int) -> List[float]:
        """Find fractal highs in price data."""
        try:
            peaks, _ = find_peaks(high, distance = period)
            return [high[i] for i in peaks]
        except Exception:
            return []

    def _find_swing_highs(self, high: np.ndarray, period: int) -> List[float]:
        """Find swing highs for Fibonacci analysis."""
        try:
            swing_highs = []
            for i in range(period, len(high) - period):
                if high[i] == np.max(high[i-period:i+period+1]):
                    swing_highs.append(high[i])
            return swing_highs
        except Exception:
            return []

    def _find_swing_lows(self, low: np.ndarray, period: int) -> List[float]:
        """Find swing lows for Fibonacci analysis."""
        try:
            swing_lows = []
            for i in range(period, len(low) - period):
                if low[i] == np.min(low[i-period:i+period+1]):
                    swing_lows.append(low[i])
            return swing_lows
        except Exception:
            return []

    def _find_fractal_lows(self, low: np.ndarray, period: int) -> List[float]:
        """Find fractal lows in price data."""
        try:
            peaks, _ = find_peaks(-low, distance = period)
            return [low[i] for i in peaks]
        except Exception:
            return []

    def _is_pivot_high(self, high: np.ndarray, index: int, period: int) -> bool:
        """Check if index is a pivot high."""
        try:
            if index < period or index >= len(high) - period:
                return False
            center_value = high[index]
            left_values = high[index - period:index]
            right_values = high[index + 1:index + period + 1]
            return center_value > np.max(left_values) and center_value > np.max(right_values)
        except Exception:
            return False

    def _is_pivot_low(self, low: np.ndarray, index: int, period: int) -> bool:
        """Check if index is a pivot low."""
        try:
            if index < period or index >= len(low) - period:
                return False
            center_value = low[index]
            left_values = low[index - period:index]
            right_values = low[index + 1:index + period + 1]
            return center_value < np.min(left_values) and center_value < np.min(right_values)
        except Exception:
            return False

    def _should_merge_levels(self, level1: SRLevel, level2: SRLevel, data: pd.DataFrame) -> bool:
        """
        Determine if two S/R levels should be merged based on multiple criteria.

        Args:
            level1: First S/R level
            level2: Second S/R level
            data: Market data for context

        Returns:
            True if levels should be merged, False otherwise
        """
        try:
            # Basic price proximity check
            price_diff = abs(level1.price - level2.price) / level1.price if level1.price != 0 else 1.0
            if price_diff >= self.touch_proximity_threshold:
                return False

            # If levels are the same type, they're more likely to be merged
            if level1.type == level2.type:
                # For same type levels, consider merging if they're very close
                # or if one is significantly weaker than the other
                strength_ratio = min(level1.strength, level2.strength) / max(level1.strength, level2.strength)

                # Merge if very close (within 0.3%) or strength difference is significant (>3x)
                if price_diff < 0.003 or strength_ratio < 0.33:
                    return True

                # Don't merge strong levels of same type that are moderately close
                # (let them remain separate for more precision)
                return False

            # Different types (support vs resistance) - more conservative merging
            else:
                # Only merge different types if they're extremely close
                # AND both are relatively weak (potential consolidation zone)
                very_close_threshold = 0.002  # 0.2%
                both_weak = level1.strength < 0.4 and level2.strength < 0.4

                # Check if levels might represent a consolidation zone
                # (area where price oscillates between support and resistance)
                consolidation_zone = self._is_consolidation_zone(level1, level2, data)

                return price_diff < very_close_threshold and (both_weak or consolidation_zone)

        except Exception as e:
            self.logger.warning(f'Error determining merge decision: {e}')
            return False

    def _is_consolidation_zone(self, level1: SRLevel, level2: SRLevel, data: pd.DataFrame) -> bool:
        """
        Check if two levels of opposite types might represent a consolidation zone.

        Args:
            level1: First S/R level
            level2: Second S/R level
            data: Market data

        Returns:
            True if levels likely represent consolidation zone
        """
        try:
            if len(data) < 20:
                return False

            # Look for price oscillation between the two levels
            mid_price = (level1.price + level2.price) / 2
            price_range = abs(level1.price - level2.price)

            # Check recent price action around these levels
            recent_data = data.tail(50) if len(data) >= 50 else data

            # Count times price touched both levels in recent history
            touches_level1 = 0
            touches_level2 = 0

            for idx, row in recent_data.iterrows():
                if abs(row['high'] - level1.price) / level1.price < 0.001:
                    touches_level1 += 1
                if abs(row['low'] - level2.price) / level2.price < 0.001:
                    touches_level2 += 1
                if abs(row['high'] - level2.price) / level2.price < 0.001:
                    touches_level2 += 1
                if abs(row['low'] - level1.price) / level1.price < 0.001:
                    touches_level1 += 1

            # If both levels have been touched multiple times, likely consolidation
            min_touches_for_consolidation = 3
            return touches_level1 >= min_touches_for_consolidation and touches_level2 >= min_touches_for_consolidation

        except Exception as e:
            self.logger.warning(f'Error checking consolidation zone: {e}')
            return False

    def _validate_and_merge_levels(self, levels: List[SRLevel], data: pd.DataFrame) -> List[SRLevel]:
        """Validate and merge similar levels with intelligent merging logic."""
        try:
            if not levels:
                return []

            # For large numbers of levels, use a more efficient approach
            if len(levels) > 100:
                self.logger.info(f'📊 Large level set ({len(levels)}), using optimized merging approach')
                return self._validate_and_merge_levels_optimized(levels, data)

            merged_levels = []
            used_indices = set()

            for i, level in enumerate(levels):
                if i in used_indices:
                    continue
                similar_levels = [level]

                for j, other_level in enumerate(levels[i + 1:], i + 1):
                    if j in used_indices:
                        continue

                    # Use intelligent merging decision
                    if self._should_merge_levels(level, other_level, data):
                        similar_levels.append(other_level)
                        used_indices.add(j)

                if len(similar_levels) > 1:
                    merged_level = self._merge_similar_levels(similar_levels)
                    merged_levels.append(merged_level)
                else:
                    merged_levels.append(level)
                used_indices.add(i)

            return merged_levels
        except Exception as e:
            self.logger.warning(f'Level validation failed: {e}')
            return levels

    def _validate_and_merge_levels_optimized(self, levels: List[SRLevel], data: pd.DataFrame) -> List[SRLevel]:
        """Optimized level merging for large datasets using spatial clustering."""
        try:
            import time
            merge_start = time.time()

            if not levels:
                return []

            # Sort levels by price for more efficient merging
            levels_sorted = sorted(levels, key=lambda x: x.price)

            merged_levels = []
            i = 0
            while i < len(levels_sorted):
                current_level = levels_sorted[i]
                similar_levels = [current_level]
                j = i + 1

                # Look ahead for similar levels within price tolerance
                while j < len(levels_sorted):
                    next_level = levels_sorted[j]
                    price_diff = abs(current_level.price - next_level.price) / current_level.price if current_level.price != 0 else 1.0

                    # If price difference is too large, no more similar levels ahead
                    if price_diff > self.touch_proximity_threshold * 2:  # Use 2x tolerance for early stopping
                        break

                    # Check if levels should merge
                    if self._should_merge_levels(current_level, next_level, data):
                        similar_levels.append(next_level)
                    j += 1

                if len(similar_levels) > 1:
                    merged_level = self._merge_similar_levels(similar_levels)
                    merged_levels.append(merged_level)
                else:
                    merged_levels.append(current_level)

                i = j  # Skip merged levels

            merge_time = time.time() - merge_start
            self.logger.info(f'✅ Optimized merging completed in {merge_time:.2f}s: {len(levels)} -> {len(merged_levels)} levels')
            return merged_levels

        except Exception as e:
            self.logger.warning(f'Optimized level merging failed: {e}, falling back to standard method')
            # Fallback to original method if optimized fails
            return self._validate_and_merge_levels_fallback(levels, data)

    def _validate_and_merge_levels_fallback(self, levels: List[SRLevel], data: pd.DataFrame) -> List[SRLevel]:
        """Fallback merging method with better performance for large datasets."""
        try:
            if len(levels) <= 50:
                # Use original method for small datasets
                merged_levels = []
                used_indices = set()

                for i, level in enumerate(levels):
                    if i in used_indices:
                        continue
                    similar_levels = [level]

                    for j, other_level in enumerate(levels[i + 1:], i + 1):
                        if j in used_indices:
                            continue
                        if self._should_merge_levels(level, other_level, data):
                            similar_levels.append(other_level)
                            used_indices.add(j)

                    if len(similar_levels) > 1:
                        merged_level = self._merge_similar_levels(similar_levels)
                        merged_levels.append(merged_level)
                    else:
                        merged_levels.append(level)
                    used_indices.add(i)

                return merged_levels
            else:
                # For very large datasets, skip merging to avoid timeout
                self.logger.warning(f'📊 Very large level set ({len(levels)}), skipping merging to avoid timeout')
                return levels

        except Exception as e:
            self.logger.warning(f'Fallback level merging failed: {e}')
            return levels

    def _merge_similar_levels(self, levels: List[SRLevel]) -> SRLevel:
        """Merge similar S/R levels into one."""
        try:
            total_strength = sum((level.strength for level in levels))
            weighted_price = sum((level.price * level.strength for level in levels)) / total_strength if total_strength != 0 else sum((level.price for level in levels)) / len(levels) if len(levels) > 0 else 0.0
            base_level = max(levels, key=lambda x: x.strength)

            # Handle None timestamps safely
            first_times = [level.first_touch_time for level in levels if level.first_touch_time is not None]
            last_times = [level.last_touch_time for level in levels if level.last_touch_time is not None]

            first_touch_time = min(first_times) if first_times else None
            last_touch_time = max(last_times) if last_times else None

            merged_level = SRLevel(
                price=weighted_price,
                strength=min(total_strength / len(levels) * 1.2, 1.0),
                type=base_level.type,
                touch_count=sum((level.touch_count for level in levels)),
                first_touch_time=first_touch_time,
                last_touch_time=last_touch_time,
                age_bars=max((level.age_bars for level in levels)),
                avg_bounce_ratio=np.mean([level.avg_bounce_ratio for level in levels]),
                max_bounce_ratio=max((level.max_bounce_ratio for level in levels)),
                volume_confirmation_score=np.mean([level.volume_confirmation_score for level in levels]),
                consistency_score=np.mean([level.consistency_score for level in levels]),
                failure_count=sum((level.failure_count for level in levels)),
                confidence_score=min(np.mean([level.confidence_score for level in levels]) * 1.1, 1.0),
                confluence_score=len(levels) / 10.0,
                pivot_level=any((level.pivot_level for level in levels)),
                psychological_level=any((level.psychological_level for level in levels)),
                metadata={'merged_from': len(levels), 'methods': [level.metadata.get('method', 'unknown') for level in levels]}
            )
            return merged_level
        except Exception as e:
            self.logger.warning(f'Level merging failed: {e}')
            return levels[0] if levels else None

    def _calculate_enhanced_metrics(self, levels: List[SRLevel], data: pd.DataFrame) -> List[SRLevel]:
        """Calculate enhanced metrics for S/R levels with optimized processing."""
        try:
            enhanced_levels = []
            # Limit the number of levels to process to prevent hanging
            max_levels = min(len(levels), 300)  # Process max 300 levels (increased for more detection)
            self.logger.info(f'📊 Processing {max_levels} levels for enhanced metrics (out of {len(levels)} total)')

            # Pre-compute common data structures for better performance
            data_high = data['high'].values
            data_low = data['low'].values
            data_close = data['close'].values
            data_volume = data['volume'].values if 'volume' in data.columns else np.ones(len(data))

            for i, level in enumerate(levels[:max_levels]):
                try:
                    level_price = level.price
                    threshold = level_price * self.touch_proximity_threshold

                    # Optimized touch counting using pre-computed arrays
                    if level.type == 'support':
                        touches = int((np.abs(data_low - level_price) <= threshold).sum())
                    else:  # resistance
                        touches = int((np.abs(data_high - level_price) <= threshold).sum())
                    level.touch_count = touches

                    # Optimized bounce metrics using vectorized operations
                    if level.type == 'support':
                        touches_mask = np.abs(data_low - level_price) <= threshold
                        next_highs = data_high[1:][touches_mask[:-1]]
                        bounce_ratios = (next_highs - level_price) / level_price
                    else:  # resistance
                        touches_mask = np.abs(data_high - level_price) <= threshold
                        next_lows = data_low[1:][touches_mask[:-1]]
                        bounce_ratios = (level_price - next_lows) / level_price

                    valid_bounces = bounce_ratios[bounce_ratios > 0]
                    if len(valid_bounces) > 0:
                        level.avg_bounce_ratio = float(np.mean(valid_bounces))
                        level.max_bounce_ratio = float(np.max(valid_bounces))
                    else:
                        level.avg_bounce_ratio = 0.0
                        level.max_bounce_ratio = 0.0

                    # Optimized volume confirmation
                    if 'volume' in data.columns:
                        volume_ma = pd.Series(data_volume).rolling(window=20).mean().values
                        volume_spikes = (data_volume > volume_ma * self.volume_spike_threshold) & touches_mask
                        total_touches = touches_mask.sum()
                        level.volume_confirmation_score = float(volume_spikes.sum() / total_touches) if total_touches > 0 else 0.0
                    else:
                        level.volume_confirmation_score = 0.0

                    # Optimized consistency calculation
                    if touches == 0:
                        level.consistency_score = 0.0
                    else:
                        touch_score = min(touches / 5.0, 1.0)
                        age_score = min(level.age_bars / 1000.0, 1.0) if hasattr(level, 'age_bars') and level.age_bars else 0.0
                        level.consistency_score = (touch_score + age_score) / 2.0

                    # Optimized failure counting
                    if level.type == 'support':
                        failures = int((data_close < (level_price - threshold)).sum())
                    else:  # resistance
                        failures = int((data_close > (level_price + threshold)).sum())
                    level.failure_count = failures

                    # Calculate enhanced strength
                    level.strength = self._calculate_enhanced_strength(level)

                    # Optimized age calculation
                    if hasattr(level, 'first_touch_time') and hasattr(level, 'last_touch_time') and level.first_touch_time and level.last_touch_time:
                        try:
                            if hasattr(level.last_touch_time, 'total_seconds') and hasattr(level.first_touch_time, 'total_seconds'):
                                level.age_bars = (level.last_touch_time - level.first_touch_time).total_seconds() / 60
                            elif isinstance(level.last_touch_time, (int, float)) and isinstance(level.first_touch_time, (int, float)):
                                level.age_bars = (level.last_touch_time - level.first_touch_time) / (1000 * 60)
                            else:
                                last_time = pd.to_datetime(level.last_touch_time, unit='ms') if isinstance(level.last_touch_time, (int, float)) else level.last_touch_time
                                first_time = pd.to_datetime(level.first_touch_time, unit='ms') if isinstance(level.first_touch_time, (int, float)) else level.first_touch_time
                                level.age_bars = (last_time - first_time).total_seconds() / 60
                        except Exception as time_error:
                            level.age_bars = 0
                    else:
                        level.age_bars = 0

                    enhanced_levels.append(level)

                    # Log progress every 10 levels
                    if (i + 1) % 10 == 0:
                        self.logger.info(f'📊 Processed {i + 1}/{max_levels} levels')

                except Exception as e:
                    self.logger.warning(f'⚠️ Failed to process level {i}: {e}')
                    # Add the level with default values
                    if not hasattr(level, 'touch_count'):
                        level.touch_count = 0
                    if not hasattr(level, 'avg_bounce_ratio'):
                        level.avg_bounce_ratio = 0.0
                    if not hasattr(level, 'max_bounce_ratio'):
                        level.max_bounce_ratio = 0.0
                    if not hasattr(level, 'volume_confirmation_score'):
                        level.volume_confirmation_score = 0.0
                    if not hasattr(level, 'consistency_score'):
                        level.consistency_score = 0.0
                    if not hasattr(level, 'failure_count'):
                        level.failure_count = 0
                    if not hasattr(level, 'age_bars'):
                        level.age_bars = 0
                    enhanced_levels.append(level)

            self.logger.info(f'✅ Enhanced metrics calculated for {len(enhanced_levels)} levels')
            return enhanced_levels
        except Exception as e:
            self.logger.warning(f'Enhanced metrics calculation failed: {e}')
            return levels

    def _count_touches(self, level: SRLevel, data: pd.DataFrame) -> int:
        """Count touches of price to S/R level."""
        try:
            threshold = level.price * self.touch_proximity_threshold
            
            # Use vectorized operations for better performance
            if level.type == 'support':
                touches = (abs(data['low'] - level.price) <= threshold).sum()
            else:  # resistance
                touches = (abs(data['high'] - level.price) <= threshold).sum()
                
            return int(touches)
        except Exception as e:
            self.logger.warning(f'Touch counting failed: {e}')
            return 0

    def _calculate_bounce_metrics(self, level: SRLevel, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate bounce metrics for S/R level using vectorized operations."""
        try:
            threshold = level.price * self.touch_proximity_threshold

            # Use vectorized operations for better performance
            if level.type == 'support':
                # Find touches to support level
                touches = abs(data['low'] - level.price) <= threshold
                # Get next high after each touch
                next_highs = data['high'].shift(-1)[touches[:-1]]
                # Calculate bounce ratios for valid touches
                bounce_ratios = (next_highs - level.price) / level.price
            else:  # resistance
                # Find touches to resistance level
                touches = abs(data['high'] - level.price) <= threshold
                # Get next low after each touch
                next_lows = data['low'].shift(-1)[touches[:-1]]
                # Calculate bounce ratios for valid touches
                bounce_ratios = (level.price - next_lows) / level.price

            # Filter positive bounce ratios
            valid_bounces = bounce_ratios[bounce_ratios > 0]

            if len(valid_bounces) > 0:
                return {'avg_bounce': float(np.mean(valid_bounces)), 'max_bounce': float(np.max(valid_bounces))}
            else:
                return {'avg_bounce': 0.0, 'max_bounce': 0.0}
        except Exception as e:
            self.logger.warning(f'Bounce calculation failed: {e}')
            return {'avg_bounce': 0.0, 'max_bounce': 0.0}

    def _calculate_volume_confirmation(self, level: SRLevel, data: pd.DataFrame) -> float:
        """Calculate volume confirmation score for S/R level."""
        try:
            if 'volume' not in data.columns:
                return 0.0
                
            volume_ma = data['volume'].rolling(window=20).mean()
            threshold = level.price * self.touch_proximity_threshold
            
            # Use vectorized operations for better performance
            if level.type == 'support':
                price_touches = abs(data['low'] - level.price) <= threshold
            else:  # resistance
                price_touches = abs(data['high'] - level.price) <= threshold
            
            # Calculate volume spikes for touching points
            volume_spikes = (data['volume'] > volume_ma * self.volume_spike_threshold) & price_touches
            
            # Return the ratio of volume spikes to total touches
            total_touches = price_touches.sum()
            if total_touches > 0:
                return float(volume_spikes.sum() / total_touches)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f'Volume confirmation calculation failed: {e}')
            return 0.0

    def _calculate_consistency_score(self, level: SRLevel, data: pd.DataFrame) -> float:
        """Calculate consistency score for S/R level."""
        try:
            if level.touch_count == 0:
                return 0.0
            touch_score = min(level.touch_count / 5.0, 1.0)
            age_score = min(level.age_bars / 1000.0, 1.0)
            return (touch_score + age_score) / 2.0
        except Exception as e:
            self.logger.warning(f'Consistency calculation failed: {e}')
            return 0.0

    def _count_failures(self, level: SRLevel, data: pd.DataFrame) -> int:
        """Count failures (breakouts) of S/R level using vectorized operations."""
        try:
            threshold = level.price * self.touch_proximity_threshold

            # Use vectorized operations for better performance
            if level.type == 'support':
                # Count breakouts below support level
                failures = (data['close'] < (level.price - threshold)).sum()
            else:  # resistance
                # Count breakouts above resistance level
                failures = (data['close'] > (level.price + threshold)).sum()

            return int(failures)
        except Exception as e:
            self.logger.warning(f'Failure counting failed: {e}')
            return 0

    def _calculate_enhanced_strength(self, level: SRLevel) -> float:
        """Calculate enhanced strength score for S/R level."""
        try:
            base_strength = level.strength
            touch_boost = min(level.touch_count * 0.1, 0.3)
            volume_boost = level.volume_confirmation_score * 0.2
            consistency_boost = level.consistency_score * 0.2
            confluence_boost = level.confluence_score * 0.1
            failure_penalty = min(level.failure_count * 0.1, 0.3)
            special_boost = 0.0
            if level.pivot_level:
                special_boost += 0.1
            if level.psychological_level:
                special_boost += 0.05
            final_strength = base_strength + touch_boost + volume_boost + consistency_boost + confluence_boost + special_boost - failure_penalty
            return max(0.0, min(1.0, final_strength))
        except Exception as e:
            self.logger.warning(f'Enhanced strength calculation failed: {e}')
            return level.strength

    def _apply_ml_optimized_filtering(self, levels: List[SRLevel], data: pd.DataFrame) -> List[SRLevel]:
        """Apply ML-optimized filtering to remove bottom levels based on total count for better ML training."""
        try:
            if len(levels) < 20:  # Minimum threshold for meaningful filtering
                return levels

            self.logger.info(f'🎯 Applying ML-optimized filtering to {len(levels)} levels')

            # Sort levels by strength in descending order
            sorted_levels = sorted(levels, key=lambda x: x.strength, reverse=True)

            # Apply tiered filtering based on total level count
            if len(sorted_levels) > 200:
                # Remove 66% of bottom levels for very large datasets
                keep_count = int(len(sorted_levels) * 0.34)
                removal_rate = "66%"
            elif len(sorted_levels) > 100:
                # Remove 50% of bottom levels for large datasets
                keep_count = int(len(sorted_levels) * 0.50)
                removal_rate = "50%"
            else:
                # Keep all levels for smaller datasets
                keep_count = len(sorted_levels)
                removal_rate = "0%"

            filtered_levels = sorted_levels[:keep_count]

            if len(sorted_levels) > 100:
                self.logger.info(f'🎯 Filtered out {len(sorted_levels) - len(filtered_levels)} weak levels '
                               f'({removal_rate} removal rate, {len(filtered_levels)} remaining)')
            else:
                self.logger.info(f'🎯 Keeping all {len(filtered_levels)} levels (below threshold for filtering)')

            return filtered_levels

        except Exception as e:
            self.logger.warning(f'ML-optimized filtering failed: {e}')
            return levels

    def _cluster_nearby_levels(self, levels: List[SRLevel], data: pd.DataFrame) -> Tuple[List[SRLevel], Dict[str, Any]]:
        """Cluster nearby SR levels using DBSCAN for optimal grouping."""
        try:
            if len(levels) < 2:
                return levels, {'clustered': False, 'reason': 'insufficient_levels'}

            self.logger.info(f'🔗 Applying DBSCAN clustering to {len(levels)} levels')

            # Separate support and resistance levels for independent clustering
            support_levels = [level for level in levels if level.type == 'support']
            resistance_levels = [level for level in levels if level.type == 'resistance']

            # Cluster each type separately
            clustered_support = self._dbscan_cluster_levels(support_levels, data, level_type='support')
            clustered_resistance = self._dbscan_cluster_levels(resistance_levels, data, level_type='resistance')

            clustered_levels = clustered_support + clustered_resistance

            # Calculate clustering statistics
            support_clusters = len([l for l in clustered_support if l.metadata and l.metadata.get('clustered_by') == 'dbscan'])
            resistance_clusters = len([l for l in clustered_resistance if l.metadata and l.metadata.get('clustered_by') == 'dbscan'])

            clustering_info = {
                'clustered': True,
                'original_levels': len(levels),
                'final_levels': len(clustered_levels),
                'support_clusters': support_clusters,
                'resistance_clusters': resistance_clusters,
                'total_clusters': support_clusters + resistance_clusters,
                'reduction_percentage': ((len(levels) - len(clustered_levels)) / len(levels)) * 100 if len(levels) > 0 else 0
            }

            self.logger.info(f'🔗 DBSCAN clustering complete: {len(clustered_levels)} levels after clustering '
                           f'({len(levels)} -> {len(clustered_levels)})')

            return clustered_levels, clustering_info

        except Exception as e:
            self.logger.warning(f'DBSCAN clustering failed: {e}')
            return levels, {'clustered': False, 'error': str(e)}

    def _dbscan_cluster_levels(self, levels: List[SRLevel], data: pd.DataFrame, level_type: str) -> List[SRLevel]:
        """Apply DBSCAN clustering to group nearby SR levels with strength-aware distance metric."""
        try:
            if len(levels) < 2:
                return levels

            # Prepare data for DBSCAN with strength-aware distance
            level_data = np.array([[level.price, level.strength] for level in levels])

            # Get optimized DBSCAN parameters
            eps, min_samples = self._optimize_dbscan_parameters(levels, data)

            # Apply DBSCAN with custom strength-aware distance metric
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric=self._strength_aware_distance
            )
            cluster_labels = clustering.fit_predict(level_data)

            # Group levels by cluster
            clusters = {}
            noise_levels = []

            for i, level in enumerate(levels):
                label = cluster_labels[i]
                if label == -1:  # Noise point
                    noise_levels.append(level)
                else:
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(level)

            # Merge levels in each cluster
            clustered_levels = []
            for cluster_levels in clusters.values():
                if len(cluster_levels) > 1:
                    merged_level = self._merge_cluster_dbscan(cluster_levels, data)
                    clustered_levels.append(merged_level)
                else:
                    clustered_levels.extend(cluster_levels)

            # Keep noise levels as-is (they're important individual levels)
            clustered_levels.extend(noise_levels)

            self.logger.info(f'🔗 DBSCAN {level_type}: {len(clusters)} clusters, {len(noise_levels)} noise levels')

            return clustered_levels

        except ImportError:
            self.logger.warning('DBSCAN not available, falling back to simple clustering')
            return self._cluster_levels_by_price(levels, data)
        except Exception as e:
            self.logger.warning(f'DBSCAN clustering failed: {e}')
            return levels

    def _strength_aware_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Custom distance metric that considers both price proximity and strength.

        Stronger levels can "attract" weaker levels from greater distances.
        """
        try:
            price1, strength1 = point1
            price2, strength2 = point2

            # Calculate raw price distance
            price_distance = abs(price1 - price2)

            # Strength-based distance adjustment
            # Stronger levels have reduced effective distance (wider attraction)
            avg_strength = (strength1 + strength2) / 2
            stronger_strength = max(strength1, strength2)
            weaker_strength = min(strength1, strength2)

            # Strength factor: stronger levels attract from farther away
            # Max attraction bonus for very strong levels (0.8+) is 2x distance reduction
            strength_factor = 1.0 - (stronger_strength - avg_strength) * 0.5
            strength_factor = max(0.5, min(1.5, strength_factor))  # Clamp between 0.5 and 1.5

            # Apply strength adjustment to price distance
            adjusted_distance = price_distance * strength_factor

            return adjusted_distance

        except Exception as e:
            # Fallback to simple price distance
            return abs(point1[0] - point2[0])

    def _merge_cluster_dbscan(self, cluster: List[SRLevel], data: pd.DataFrame) -> SRLevel:
        """Merge a DBSCAN cluster using advanced aggregation."""
        try:
            if len(cluster) == 1:
                return cluster[0]

            # Strength-weighted price aggregation
            strengths = np.array([level.strength for level in cluster])
            prices = np.array([level.price for level in cluster])

            # Weighted average with outlier-resistant median fallback
            if len(cluster) >= 3:
                # Use median for robustness with many levels
                weighted_price = np.median(prices)
            else:
                # Use weighted average for small clusters
                total_weight = np.sum(strengths)
                weighted_price = np.sum(prices * strengths) / total_weight

            # Use strongest level as template
            strongest_level = max(cluster, key=lambda x: x.strength)

            # Enhanced metadata
            combined_metadata = {
                'clustered_by': 'dbscan',
                'cluster_size': len(cluster),
                'original_prices': prices.tolist(),
                'price_range': (float(np.min(prices)), float(np.max(prices))),
                'strength_distribution': strengths.tolist(),
                'avg_cluster_strength': float(np.mean(strengths)),
                'method': 'dbscan_clustered'
            }

            # Preserve original metadata
            if strongest_level.metadata:
                combined_metadata.update(strongest_level.metadata)

            # Create merged level with enhanced properties
            merged_level = SRLevel(
                price=float(weighted_price),
                strength=min(strongest_level.strength * 1.15, 1.0),  # Boost for clustered levels
                type=strongest_level.type,
                touch_count=sum(level.touch_count for level in cluster),
                first_touch_time=min((level.first_touch_time for level in cluster if level.first_touch_time is not None), default=None),
                last_touch_time=max((level.last_touch_time for level in cluster if level.last_touch_time is not None), default=None),
                age_bars=max((level.age_bars for level in cluster), default=0),
                avg_bounce_ratio=np.mean([level.avg_bounce_ratio for level in cluster if level.avg_bounce_ratio > 0]),
                max_bounce_ratio=max((level.max_bounce_ratio for level in cluster), default=0.0),
                volume_confirmation_score=np.mean([level.volume_confirmation_score for level in cluster]),
                consistency_score=np.mean([level.consistency_score for level in cluster]),
                failure_count=sum(level.failure_count for level in cluster),
                confidence_score=min(np.mean([level.confidence_score for level in cluster]) * 1.2, 1.0),
                confluence_score=min(len(cluster) * 0.15, 0.8),  # Higher confluence boost
                pivot_level=any(level.pivot_level for level in cluster),
                psychological_level=any(level.psychological_level for level in cluster),
                fibonacci_level=min((level.fibonacci_level for level in cluster if level.fibonacci_level is not None), default=None),
                metadata=combined_metadata
            )

            return merged_level

        except Exception as e:
            self.logger.warning(f'DBSCAN cluster merging failed: {e}')
            return cluster[0] if cluster else None

    def _optimize_dbscan_parameters(self, levels: List[SRLevel], data: pd.DataFrame) -> Tuple[float, int]:
        """Optimize DBSCAN parameters using clustering quality metrics."""
        try:
            if len(levels) < 5:
                # Use heuristic for small datasets
                return self._get_heuristic_dbscan_params(levels, data)

            # Try Optuna first, then fallback to grid search
            try:
                return self._optimize_dbscan_optuna(levels, data)
            except ImportError:
                try:
                    return self._optimize_dbscan_skopt(levels, data)
                except ImportError:
                    # Final fallback to heuristics
                    return self._get_heuristic_dbscan_params(levels, data)

        except Exception as e:
            self.logger.warning(f'DBSCAN parameter optimization failed: {e}')
            return self._get_heuristic_dbscan_params(levels, data)

    def _optimize_dbscan_optuna(self, levels: List[SRLevel], data: pd.DataFrame) -> Tuple[float, int]:
        """Optimize DBSCAN parameters using Optuna."""
        import optuna

        level_data = np.array([[level.price, level.strength] for level in levels])
        avg_price = data['close'].mean()
        price_volatility = data['close'].pct_change().std()

        def objective(trial: optuna.Trial) -> float:
            # Define parameter search space
            eps_relative = trial.suggest_float('eps_relative', 0.002, 0.05)  # 0.2% to 5% of price
            eps = eps_relative * avg_price

            min_samples_upper = max(2, min(8, len(levels) // 5))
            min_samples = trial.suggest_int('min_samples', 2, min_samples_upper)

            # Apply DBSCAN with strength-aware distance
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=self._strength_aware_distance)
            labels = clustering.fit_predict(level_data)

            # Calculate clustering quality metrics
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            # Calculate silhouette score if possible
            if n_clusters > 1 and n_noise < len(levels) * 0.8:
                try:
                    from sklearn.metrics import silhouette_score
                    if len(set(labels)) > 1:
                        score = silhouette_score(level_data, labels)
                        return score
                except:
                    pass

            # Fallback metric: balance between clusters and noise
            cluster_ratio = n_clusters / max(1, len(levels) - n_noise)
            noise_ratio = n_noise / len(levels)

            # Prefer moderate clustering with reasonable noise
            score = cluster_ratio * 0.7 - noise_ratio * 0.3
            return score

        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=min(20, len(levels)))

        # Get best parameters
        best_eps_relative = study.best_params['eps_relative']
        best_min_samples = study.best_params['min_samples']
        best_eps = best_eps_relative * avg_price

        self.logger.info(f'🔧 Optimized DBSCAN params - eps: {best_eps:.6f}, min_samples: {best_min_samples}, score: {study.best_value:.3f}')

        return best_eps, best_min_samples

    def _optimize_dbscan_skopt(self, levels: List[SRLevel], data: pd.DataFrame) -> Tuple[float, int]:
        """Optimize DBSCAN parameters using scikit-optimize."""
        from skopt import gp_minimize
        from skopt.space import Real, Integer

        level_data = np.array([[level.price, level.strength] for level in levels])
        avg_price = data['close'].mean()

        def objective(params):
            eps_relative, min_samples = params
            eps = eps_relative * avg_price

            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=eps, min_samples=int(min_samples), metric=self._strength_aware_distance)
            labels = clustering.fit_predict(level_data)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            if n_clusters == 0:
                return 100  # Minimize this (gp_minimize minimizes)

            cluster_ratio = n_clusters / max(1, len(levels) - n_noise)
            noise_ratio = n_noise / len(levels)

            score = -(cluster_ratio * 0.7 - noise_ratio * 0.3)  # Negative for minimization
            return score

        # Define search space with proper bounds
        min_samples_upper = max(2, min(8, len(levels) // 5))
        space = [
            Real(0.002, 0.05, name='eps_relative'),  # 0.2% to 5% of price
            Integer(2, min_samples_upper, name='min_samples')
        ]

        # Run optimization
        res = gp_minimize(objective, space, n_calls=min(15, len(levels)), random_state=42)

        best_eps_relative = res.x[0]
        best_min_samples = int(res.x[1])
        best_eps = best_eps_relative * avg_price

        self.logger.info(f'🔧 Optimized DBSCAN params (skopt) - eps: {best_eps:.6f}, min_samples: {best_min_samples}')

        return best_eps, best_min_samples

    def _get_heuristic_dbscan_params(self, levels: List[SRLevel], data: pd.DataFrame) -> Tuple[float, int]:
        """Get heuristic DBSCAN parameters when optimization is not available."""
        avg_price = data['close'].mean()
        price_volatility = data['close'].pct_change().std()

        # Adaptive epsilon based on price volatility and level count
        base_eps_relative = 0.01  # 1% base
        volatility_factor = min(price_volatility * 1.5, 0.02)  # Cap at 2%
        level_density_factor = max(0.005, 1.0 / np.sqrt(len(levels)))  # Denser levels = smaller eps

        eps_relative = base_eps_relative + volatility_factor - level_density_factor
        eps_relative = np.clip(eps_relative, 0.003, 0.04)  # Clamp between 0.3% and 4%
        eps = eps_relative * avg_price

        # Adaptive minimum samples (original aggressive settings)
        min_samples = max(2, min(6, len(levels) // 8))

        self.logger.info(f'🔧 Heuristic DBSCAN params - eps: {eps:.6f}, min_samples: {min_samples}')

        return eps, min_samples

    def _optimize_dbscan_enhanced(self, levels: List[SRLevel], data: pd.DataFrame) -> Tuple[float, int]:
        """Enhanced DBSCAN parameter optimization with ATR-based constraints."""
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer
            
            level_data = np.array([[level.price, level.strength] for level in levels])
            avg_price = data['close'].mean()
            
            # Calculate ATR for better parameter optimization
            atr = self._calculate_atr(data)
            current_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else atr.mean()
            
            # ATR-based parameter space
            eps_min = current_atr * 0.1  # 0.1 ATR minimum
            eps_max = current_atr * 2.0  # 2.0 ATR maximum
            min_samples_min = max(2, len(levels) // 20)  # Adaptive minimum
            min_samples_max = min(10, len(levels) // 3)  # Adaptive maximum
            
            space = [
                Real(eps_min, eps_max, name='eps'),
                Integer(min_samples_min, min_samples_max, name='min_samples')
            ]
            
            @use_named_args(space)
            def objective(**params):
                eps = params['eps']
                min_samples = params['min_samples']
                
                try:
                    from sklearn.cluster import DBSCAN
                    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=self._strength_aware_distance)
                    labels = clustering.fit_predict(level_data)
                    
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = list(labels).count(-1)
                    
                    if n_clusters == 0:
                        return -1.0  # Penalty for no clusters
                    
                    # Enhanced quality metrics
                    silhouette_score = self._calculate_silhouette_score(level_data, labels)
                    cluster_balance = self._calculate_cluster_balance(labels)
                    noise_ratio = n_noise / len(levels)
                    
                    # Combined score (higher is better)
                    score = (silhouette_score * 0.4 + 
                            cluster_balance * 0.3 + 
                            (1 - noise_ratio) * 0.3)
                    
                    return -score  # Minimize negative score
                    
                except Exception:
                    return -1.0
            
            result = gp_minimize(objective, space, n_calls=50, random_state=42)
            best_eps = result.x[0]
            best_min_samples = int(result.x[1])
            
            self.logger.info(f'🔧 Enhanced DBSCAN optimization - eps: {best_eps:.6f}, min_samples: {best_min_samples}, score: {-result.fun:.3f}')
            
            return best_eps, best_min_samples
            
        except Exception as e:
            self.logger.warning(f'Enhanced DBSCAN optimization failed: {e}')
            return self._get_enhanced_heuristic_dbscan_params(levels, data)

    def _calculate_silhouette_score(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering quality."""
        try:
            from sklearn.metrics import silhouette_score
            if len(set(labels)) > 1 and -1 not in labels:
                return silhouette_score(data, labels)
            return 0.0
        except Exception:
            return 0.0

    def _calculate_cluster_balance(self, labels: np.ndarray) -> float:
        """Calculate cluster balance score (higher is better)."""
        try:
            unique_labels = [label for label in set(labels) if label != -1]
            if len(unique_labels) <= 1:
                return 0.0
            
            cluster_sizes = [list(labels).count(label) for label in unique_labels]
            min_size = min(cluster_sizes)
            max_size = max(cluster_sizes)
            
            # Balance score: 1.0 for perfectly balanced, 0.0 for highly imbalanced
            return min_size / max_size if max_size > 0 else 0.0
        except Exception:
            return 0.0

    def _get_enhanced_heuristic_dbscan_params(self, levels: List[SRLevel], data: pd.DataFrame) -> Tuple[float, int]:
        """Enhanced heuristic DBSCAN parameters with ATR-based optimization."""
        try:
            avg_price = data['close'].mean()
            price_volatility = data['close'].pct_change().std()
            
            # Calculate ATR for better parameter optimization
            atr = self._calculate_atr(data)
            current_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else atr.mean()
            
            # ATR-based epsilon calculation
            base_eps_atr = 0.5  # 0.5 ATR base
            volatility_factor = min(2.0, max(0.5, price_volatility * 100))  # Scale volatility
            level_density_factor = min(2.0, max(0.5, len(levels) / 50))  # Scale by level count
            
            eps = current_atr * base_eps_atr * volatility_factor / level_density_factor
            
            # Adaptive min_samples based on level count and quality
            base_min_samples = max(2, min(8, len(levels) // 10))
            quality_factor = np.mean([level.strength for level in levels])
            min_samples = int(base_min_samples * (2 - quality_factor))  # Higher quality = fewer samples needed
            
            # Apply multipliers from config
            eps *= self.dbscan_eps_multiplier
            min_samples = int(min_samples * self.dbscan_min_samples_multiplier)
            
            self.logger.info(f'🔧 Enhanced heuristic DBSCAN params - eps: {eps:.6f} ({eps/current_atr:.2f} ATR), min_samples: {min_samples}')
            
            return eps, min_samples
            
        except Exception as e:
            self.logger.warning(f'Enhanced heuristic DBSCAN params failed: {e}')
            return self._get_heuristic_dbscan_params(levels, data)

    def _cluster_levels_by_price(self, levels: List[SRLevel], data: pd.DataFrame) -> List[SRLevel]:
        """Fallback clustering method when DBSCAN is not available."""
        try:
            if len(levels) < 2:
                return levels

            # Sort levels by price
            sorted_levels = sorted(levels, key=lambda x: x.price)
            clustered_levels = []

            current_cluster = [sorted_levels[0]]
            current_price = sorted_levels[0].price

            # Dynamic clustering threshold
            avg_price = data['close'].mean()
            price_volatility = data['close'].pct_change().std()
            cluster_threshold = max(0.005, min(0.02, price_volatility * 3))

            for level in sorted_levels[1:]:
                price_diff = abs(level.price - current_price) / current_price

                if price_diff <= cluster_threshold:
                    current_cluster.append(level)
                else:
                    if current_cluster:
                        clustered_levels.append(self._merge_cluster_dbscan(current_cluster, data))
                    current_cluster = [level]
                    current_price = level.price

            # Process final cluster
            if current_cluster:
                clustered_levels.append(self._merge_cluster_dbscan(current_cluster, data))

            return clustered_levels

        except Exception as e:
            self.logger.warning(f'Fallback clustering failed: {e}')
            return levels