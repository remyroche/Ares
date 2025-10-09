"""
Advanced Filters for 15m Timeframe Profit Labeling

This module implements advanced filtering techniques specifically optimized for 15-minute
timeframe data to improve label quality and reduce noise in profit labeling.

Key Features:
1. Bar Efficiency Ratio - Measures directional price action vs. choppy conditions
2. Close-Location Value (CLV) - Tracks buying/selling pressure and control
3. ATR Volatility Ratio - Normalizes volatility for adaptive filtering
4. Trend Coherence Features - Ensures trend continuity and direction consistency

These filters are designed to work with the existing VolatilityAwareMultiHorizonLabeler
and integrate seamlessly with the current profit labeling pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import warnings

# Import existing utilities
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.common_operations import (
    safe_divide, safe_log, safe_sqrt, safe_mean, safe_std,
    validate_finite, validate_positive, validate_range, safe_correlation
)
from src.utils.math_validation import MathValidation

# Import matrix operations for optimized rolling calculations
from src.utils.matrix_operations import vectorized_rolling_features


class FilterType(Enum):
    """Enumeration of filter types."""
    EFFICIENCY_RATIO = "efficiency_ratio"
    CLV = "clv"
    ATR_RATIO = "atr_ratio"
    TREND_COHERENCE = "trend_coherence"
    COMBINED = "combined"


@dataclass
class AdvancedFiltersConfig:
    """Configuration for advanced 15m timeframe filters."""
    
    # Global enable/disable
    enabled: bool = True
    
    # Bar Efficiency Ratio settings
    enable_efficiency_ratio: bool = True
    efficiency_window: int = 6  # Rolling window for efficiency (4-8 bars = 1-2 hours)
    efficiency_threshold_high: float = 0.6  # High efficiency = directional
    efficiency_threshold_low: float = 0.3   # Low efficiency = choppy
    
    # Close-Location Value (CLV) settings
    enable_clv: bool = True
    clv_window: int = 8  # Rolling window for CLV smoothing
    clv_threshold_positive: float = 0.2   # Sustained positive CLV = bullish
    clv_threshold_negative: float = -0.2  # Sustained negative CLV = bearish
    clv_volatility_threshold: float = 0.5  # Avoid when CLV fluctuates rapidly
    
    # ATR Volatility Ratio settings
    enable_atr_ratio: bool = True
    atr_short_window: int = 4   # Short-term ATR window (1 hour)
    atr_long_window: int = 20   # Long-term ATR window (5 hours)
    atr_ratio_threshold_high: float = 1.5  # Too jumpy
    atr_ratio_threshold_low: float = 0.5   # Too quiet
    
    # Trend Coherence settings
    enable_trend_coherence: bool = True
    direction_window: int = 8   # Window for direction consistency check
    min_direction_consistency: float = 0.6  # 60% of bars in same direction
    ema_period: int = 12        # EMA period for slope calculation
    min_slope_threshold: float = 0.001  # Minimum slope for trend continuity
    
    # Combined filtering
    filter_type: FilterType = FilterType.COMBINED
    min_eligibility_ratio: float = 0.3  # Minimum ratio of eligible samples
    strict_mode: bool = False  # Use strict eligibility criteria
    
    # Quality checks
    min_eligible_samples: int = 50
    max_filter_failure_rate: float = 0.7  # Maximum allowed filter failure rate


@dataclass
class FilterResult:
    """Result container for advanced filtering."""
    
    # Core results
    eligibility_mask: pd.Series
    eligibility_ratio: float
    
    # Filter-specific results
    efficiency_mask: Optional[pd.Series] = None
    clv_mask: Optional[pd.Series] = None
    atr_ratio_mask: Optional[pd.Series] = None
    trend_coherence_mask: Optional[pd.Series] = None
    
    # Statistics
    n_total_samples: int = 0
    n_eligible_samples: int = 0
    n_filtered_samples: int = 0
    
    # Filter performance
    filter_effectiveness: Dict[str, float] = field(default_factory=dict)
    filter_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Quality metrics
    overall_quality_score: float = 0.0
    noise_reduction_ratio: float = 0.0
    
    # Metadata
    config_used: AdvancedFiltersConfig = None
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class AdvancedFilters15m:
    """
    Advanced Filters for 15m Timeframe Profit Labeling
    
    This class implements sophisticated filtering techniques optimized for 15-minute
    timeframe data to improve label quality and reduce noise.
    
    Key Features:
    1. **Bar Efficiency Ratio**: Measures directional price action vs. choppy conditions
    2. **Close-Location Value (CLV)**: Tracks buying/selling pressure and control
    3. **ATR Volatility Ratio**: Normalizes volatility for adaptive filtering
    4. **Trend Coherence**: Ensures trend continuity and direction consistency
    """
    
    def __init__(self, config: Optional[AdvancedFiltersConfig] = None):
        """Initialize advanced filters for 15m timeframe."""
        self.config = config or AdvancedFiltersConfig()
        self.logger = logging.getLogger('AdvancedFilters15m')
        
        tprint_info("🔍 Advanced Filters for 15m Timeframe initialized")
        tprint_info(f"   → Efficiency ratio: {self.config.enable_efficiency_ratio}")
        tprint_info(f"   → CLV filtering: {self.config.enable_clv}")
        tprint_info(f"   → ATR ratio: {self.config.enable_atr_ratio}")
        tprint_info(f"   → Trend coherence: {self.config.enable_trend_coherence}")
    
    def apply_filters(self, data: pd.DataFrame) -> FilterResult:
        """
        Apply advanced filters to 15m timeframe data.
        
        Args:
            data: OHLCV data with 15m timeframe
            
        Returns:
            FilterResult with eligibility mask and statistics
        """
        start_time = datetime.now()
        tprint_info("🔍 Applying advanced 15m filters")
        
        # Validate input data
        self._validate_input_data(data)
        
        # Initialize result container
        result = FilterResult(
            eligibility_mask=pd.Series(True, index=data.index),
            eligibility_ratio=1.0,
            n_total_samples=len(data),
            config_used=self.config
        )
        
        try:
            # Apply individual filters
            if self.config.enable_efficiency_ratio:
                result.efficiency_mask = self._apply_efficiency_ratio_filter(data)
                result.filter_statistics['efficiency'] = self._calculate_efficiency_stats(data, result.efficiency_mask)
            
            if self.config.enable_clv:
                result.clv_mask = self._apply_clv_filter(data)
                result.filter_statistics['clv'] = self._calculate_clv_stats(data, result.clv_mask)
            
            if self.config.enable_atr_ratio:
                result.atr_ratio_mask = self._apply_atr_ratio_filter(data)
                result.filter_statistics['atr_ratio'] = self._calculate_atr_ratio_stats(data, result.atr_ratio_mask)
            
            if self.config.enable_trend_coherence:
                result.trend_coherence_mask = self._apply_trend_coherence_filter(data)
                result.filter_statistics['trend_coherence'] = self._calculate_trend_coherence_stats(data, result.trend_coherence_mask)
            
            # Combine filters based on configuration
            result.eligibility_mask = self._combine_filters(result)
            result.eligibility_ratio = result.eligibility_mask.mean()
            result.n_eligible_samples = result.eligibility_mask.sum()
            result.n_filtered_samples = result.n_total_samples - result.n_eligible_samples
            
            # Calculate quality metrics
            result.overall_quality_score = self._calculate_overall_quality_score(result)
            result.noise_reduction_ratio = result.n_filtered_samples / result.n_total_samples
            
            # Calculate filter effectiveness
            result.filter_effectiveness = self._calculate_filter_effectiveness(result)
            
            # Validate results
            self._validate_filter_results(result)
            
            result.processing_time = (datetime.now() - start_time).total_seconds()
            
            tprint_success(f"✅ Advanced filters applied: {result.n_eligible_samples}/{result.n_total_samples} samples eligible ({result.eligibility_ratio:.1%})")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Error applying advanced filters: {e}")
            raise
    
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input data format and requirements."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if len(data) < max(self.config.efficiency_window, self.config.clv_window, 
                          self.config.atr_long_window, self.config.direction_window):
            raise ValueError(f"Insufficient data: need at least {max(self.config.efficiency_window, self.config.clv_window, self.config.atr_long_window, self.config.direction_window)} samples")
        
        # Check for valid OHLCV data
        for col in ['open', 'high', 'low', 'close']:
            if not pd.api.types.is_numeric_dtype(data[col]):
                raise ValueError(f"Column {col} must be numeric")
        
        # Validate OHLC relationships
        invalid_ohlc = (data['high'] < data['low']) | (data['high'] < data['open']) | (data['high'] < data['close']) | (data['low'] > data['open']) | (data['low'] > data['close'])
        if invalid_ohlc.any():
            tprint_warning(f"⚠️ Found {invalid_ohlc.sum()} invalid OHLC relationships")
    
    def _apply_efficiency_ratio_filter(self, data: pd.DataFrame) -> pd.Series:
        """
        Apply bar efficiency ratio filter.
        
        Efficiency_t = |close_t - open_t| / (high_t - low_t)
        High efficiency (>0.6) = directional, Low efficiency (<0.3) = choppy
        """
        tprint_info("📊 Applying efficiency ratio filter")
        
        # Calculate efficiency ratio for each bar
        price_range = data['high'] - data['low']
        price_range = price_range.replace(0, np.nan)  # Avoid division by zero
        
        efficiency = np.abs(data['close'] - data['open']) / price_range
        efficiency = efficiency.fillna(0)  # Set to 0 for zero-range bars
        
        # Calculate rolling mean efficiency
        rolling_efficiency = efficiency.rolling(window=self.config.efficiency_window, min_periods=1).mean()
        
        # Create eligibility mask
        # Keep bars with moderate to high efficiency (avoid both very low and very high)
        efficiency_mask = (rolling_efficiency >= self.config.efficiency_threshold_low) & \
                         (rolling_efficiency <= 1.0)  # Cap at 1.0 (perfect efficiency)
        
        tprint_info(f"   → Efficiency filter: {efficiency_mask.sum()}/{len(efficiency_mask)} samples passed")
        
        return efficiency_mask
    
    def _apply_clv_filter(self, data: pd.DataFrame) -> pd.Series:
        """
        Apply Close-Location Value (CLV) filter.
        
        CLV_t = (2*close_t - high_t - low_t) / (high_t - low_t)
        Sustained positive CLV → bullish control, sustained negative → bearish
        """
        tprint_info("📊 Applying CLV filter")
        
        # Calculate CLV for each bar
        price_range = data['high'] - data['low']
        price_range = price_range.replace(0, np.nan)  # Avoid division by zero
        
        clv = (2 * data['close'] - data['high'] - data['low']) / price_range
        clv = clv.fillna(0)  # Set to 0 for zero-range bars
        
        # Calculate rolling mean CLV
        rolling_clv = clv.rolling(window=self.config.clv_window, min_periods=1).mean()
        
        # Calculate CLV volatility (standard deviation of rolling CLV)
        clv_volatility = rolling_clv.rolling(window=self.config.clv_window, min_periods=1).std()
        
        # Create eligibility mask
        # Keep bars with sustained directional CLV and low volatility
        clv_directional = (rolling_clv >= self.config.clv_threshold_positive) | \
                         (rolling_clv <= self.config.clv_threshold_negative)
        clv_stable = clv_volatility <= self.config.clv_volatility_threshold
        
        clv_mask = clv_directional & clv_stable
        
        tprint_info(f"   → CLV filter: {clv_mask.sum()}/{len(clv_mask)} samples passed")
        
        return clv_mask
    
    def _apply_atr_ratio_filter(self, data: pd.DataFrame) -> pd.Series:
        """
        Apply ATR volatility ratio filter.
        
        r_t = ATR_short / ATR_long
        Skip when r_t > 1.5-2.0 (too jumpy) or < 0.5 (too quiet)
        """
        tprint_info("📊 Applying ATR ratio filter")
        
        # Calculate True Range
        tr1 = data['high'] - data['low']
        tr2 = np.abs(data['high'] - data['close'].shift(1))
        tr3 = np.abs(data['low'] - data['close'].shift(1))
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Calculate ATR for short and long windows
        atr_short = true_range.rolling(window=self.config.atr_short_window, min_periods=1).mean()
        atr_long = true_range.rolling(window=self.config.atr_long_window, min_periods=1).mean()
        
        # Calculate ATR ratio
        atr_ratio = atr_short / atr_long
        atr_ratio = atr_ratio.fillna(1.0)  # Fill NaN values with 1.0
        atr_ratio = atr_ratio.replace([np.inf, -np.inf], 1.0)  # Replace infinite values with 1.0
        
        # Create eligibility mask
        # Keep bars with moderate volatility (not too quiet, not too jumpy)
        atr_ratio_mask = (atr_ratio >= self.config.atr_ratio_threshold_low) & \
                        (atr_ratio <= self.config.atr_ratio_threshold_high)
        
        tprint_info(f"   → ATR ratio filter: {atr_ratio_mask.sum()}/{len(atr_ratio_mask)} samples passed")
        
        return atr_ratio_mask
    
    def _apply_trend_coherence_filter(self, data: pd.DataFrame) -> pd.Series:
        """
        Apply trend coherence filter.
        
        Combines direction consistency and EMA slope for trend continuity.
        """
        tprint_info("📊 Applying trend coherence filter")
        
        # Calculate direction consistency
        # Check if bars close in the same direction as previous bars
        close_direction = np.sign(data['close'].diff())
        direction_consistency = close_direction.rolling(window=self.config.direction_window, min_periods=1).apply(
            lambda x: (x == x.iloc[-1]).sum() / len(x) if len(x) > 0 else 0
        )
        
        # Calculate EMA slope
        ema = data['close'].ewm(span=self.config.ema_period, min_periods=1).mean()
        ema_slope = ema.diff()
        
        # Create eligibility mask
        # Keep bars with consistent direction and positive slope trend
        direction_consistent = direction_consistency >= self.config.min_direction_consistency
        slope_positive = ema_slope >= self.config.min_slope_threshold
        
        trend_coherence_mask = direction_consistent & slope_positive
        
        tprint_info(f"   → Trend coherence filter: {trend_coherence_mask.sum()}/{len(trend_coherence_mask)} samples passed")
        
        return trend_coherence_mask
    
    def _combine_filters(self, result: FilterResult) -> pd.Series:
        """Combine individual filter results into final eligibility mask."""
        masks = []
        
        if result.efficiency_mask is not None:
            masks.append(result.efficiency_mask)
        if result.clv_mask is not None:
            masks.append(result.clv_mask)
        if result.atr_ratio_mask is not None:
            masks.append(result.atr_ratio_mask)
        if result.trend_coherence_mask is not None:
            masks.append(result.trend_coherence_mask)
        
        if not masks:
            # If no filters are enabled, return all True
            return pd.Series(True, index=result.eligibility_mask.index)
        
        # Combine masks based on filter type
        if self.config.filter_type == FilterType.COMBINED:
            # Use AND logic - all filters must pass
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = combined_mask & mask
        else:
            # Use OR logic - any filter can pass
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = combined_mask | mask
        
        return combined_mask
    
    def _calculate_efficiency_stats(self, data: pd.DataFrame, mask: pd.Series) -> Dict[str, float]:
        """Calculate efficiency ratio statistics."""
        price_range = data['high'] - data['low']
        price_range = price_range.replace(0, np.nan)
        efficiency = np.abs(data['close'] - data['open']) / price_range
        efficiency = efficiency.fillna(0)
        
        return {
            'mean_efficiency': float(efficiency.mean()),
            'std_efficiency': float(efficiency.std()),
            'min_efficiency': float(efficiency.min()),
            'max_efficiency': float(efficiency.max()),
            'eligible_ratio': float(mask.mean()) if mask is not None else 0.0
        }
    
    def _calculate_clv_stats(self, data: pd.DataFrame, mask: pd.Series) -> Dict[str, float]:
        """Calculate CLV statistics."""
        price_range = data['high'] - data['low']
        price_range = price_range.replace(0, np.nan)
        clv = (2 * data['close'] - data['high'] - data['low']) / price_range
        clv = clv.fillna(0)
        
        return {
            'mean_clv': float(clv.mean()),
            'std_clv': float(clv.std()),
            'min_clv': float(clv.min()),
            'max_clv': float(clv.max()),
            'eligible_ratio': float(mask.mean()) if mask is not None else 0.0
        }
    
    def _calculate_atr_ratio_stats(self, data: pd.DataFrame, mask: pd.Series) -> Dict[str, float]:
        """Calculate ATR ratio statistics."""
        tr1 = data['high'] - data['low']
        tr2 = np.abs(data['high'] - data['close'].shift(1))
        tr3 = np.abs(data['low'] - data['close'].shift(1))
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        
        atr_short = true_range.rolling(window=self.config.atr_short_window, min_periods=1).mean()
        atr_long = true_range.rolling(window=self.config.atr_long_window, min_periods=1).mean()
        atr_ratio = atr_short / atr_long
        atr_ratio = atr_ratio.fillna(1.0)  # Fill NaN values with 1.0
        atr_ratio = atr_ratio.replace([np.inf, -np.inf], 1.0)  # Replace infinite values with 1.0
        
        return {
            'mean_atr_ratio': float(atr_ratio.mean()),
            'std_atr_ratio': float(atr_ratio.std()),
            'min_atr_ratio': float(atr_ratio.min()),
            'max_atr_ratio': float(atr_ratio.max()),
            'eligible_ratio': float(mask.mean()) if mask is not None else 0.0
        }
    
    def _calculate_trend_coherence_stats(self, data: pd.DataFrame, mask: pd.Series) -> Dict[str, float]:
        """Calculate trend coherence statistics."""
        close_direction = np.sign(data['close'].diff())
        direction_consistency = close_direction.rolling(window=self.config.direction_window, min_periods=1).apply(
            lambda x: (x == x.iloc[-1]).sum() / len(x) if len(x) > 0 else 0
        )
        
        ema = data['close'].ewm(span=self.config.ema_period, min_periods=1).mean()
        ema_slope = ema.diff()
        
        return {
            'mean_direction_consistency': float(direction_consistency.mean()),
            'mean_ema_slope': float(ema_slope.mean()),
            'std_ema_slope': float(ema_slope.std()),
            'eligible_ratio': float(mask.mean()) if mask is not None else 0.0
        }
    
    def _calculate_overall_quality_score(self, result: FilterResult) -> float:
        """Calculate overall quality score based on filter results."""
        if result.n_total_samples == 0:
            return 0.0
        
        # Base score from eligibility ratio
        base_score = result.eligibility_ratio
        
        # Bonus for noise reduction
        noise_reduction_bonus = min(result.noise_reduction_ratio * 0.2, 0.2)
        
        # Penalty for too many filters failing
        if result.eligibility_ratio < self.config.min_eligibility_ratio:
            penalty = (self.config.min_eligibility_ratio - result.eligibility_ratio) * 0.5
        else:
            penalty = 0.0
        
        quality_score = base_score + noise_reduction_bonus - penalty
        return max(0.0, min(1.0, quality_score))
    
    def _calculate_filter_effectiveness(self, result: FilterResult) -> Dict[str, float]:
        """Calculate effectiveness of each filter."""
        effectiveness = {}
        
        for filter_name, stats in result.filter_statistics.items():
            if 'eligible_ratio' in stats:
                effectiveness[filter_name] = stats['eligible_ratio']
        
        return effectiveness
    
    def _validate_filter_results(self, result: FilterResult) -> None:
        """Validate filter results and raise warnings if needed."""
        if result.eligibility_ratio < self.config.min_eligibility_ratio:
            tprint_warning(f"⚠️ Low eligibility ratio: {result.eligibility_ratio:.1%} < {self.config.min_eligibility_ratio:.1%}")
        
        if result.n_eligible_samples < self.config.min_eligible_samples:
            tprint_warning(f"⚠️ Insufficient eligible samples: {result.n_eligible_samples} < {self.config.min_eligible_samples}")
        
        # Check for filter failure rates
        for filter_name, effectiveness in result.filter_effectiveness.items():
            failure_rate = 1.0 - effectiveness
            if failure_rate > self.config.max_filter_failure_rate:
                tprint_warning(f"⚠️ High failure rate for {filter_name}: {failure_rate:.1%} > {self.config.max_filter_failure_rate:.1%}")


# Convenience function for external usage
def apply_advanced_filters_15m(
    data: pd.DataFrame,
    config: Optional[AdvancedFiltersConfig] = None,
    **kwargs
) -> FilterResult:
    """
    Apply advanced filters to 15m timeframe data.
    
    Args:
        data: OHLCV data with 15m timeframe
        config: Optional configuration
        **kwargs: Additional parameters
        
    Returns:
        FilterResult with eligibility mask and statistics
    """
    filter_system = AdvancedFilters15m(config)
    return filter_system.apply_filters(data, **kwargs)