"""
Prefix Sums/EMA Reuse for Rolling Aggregations

This module implements efficient reuse of prefix sums and EMA computations
to eliminate redundant rolling calculations across multiple indicators.

Key Features:
- Prefix sums for multiple rolling windows
- EMA reuse across indicators
- Fused computations for RSI/MACD/BB
- Memory-efficient rolling operations
- Vectorized rolling statistics
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from collections import defaultdict
import time

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance
)

logger = logging.getLogger(__name__)


@dataclass
class PrefixSumsConfig:
    """Configuration for prefix sums reuse."""
    enable_reuse: bool = True
    cache_emas: bool = True
    cache_prefix_sums: bool = True
    max_cache_size: int = 1000  # Maximum number of cached computations
    memory_limit_mb: int = 500  # Memory limit for cache
    vectorized_rolling: bool = True  # Use vectorized rolling operations


class PrefixSumsReuse:
    """Efficient reuse of prefix sums and EMA computations."""
    
    def __init__(self, config: PrefixSumsConfig):
        self.config = config
        self.ema_cache = {}
        self.prefix_sums_cache = {}
        self.rolling_cache = {}
        self.computation_stats = {}
        
        tprint_info("📊 Prefix sums reuse initialized")
        tprint_info(f"📊 Cache EMAs: {config.cache_emas}")
        tprint_info(f"📊 Cache prefix sums: {config.cache_prefix_sums}")
        tprint_info(f"📊 Vectorized rolling: {config.vectorized_rolling}")
    
    def compute_rolling_features(self, 
                               data: pd.DataFrame,
                               windows: List[int],
                               feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compute rolling features with prefix sums reuse.
        
        Args:
            data: Input data
            windows: List of rolling windows
            feature_names: List of feature names to process
            
        Returns:
            DataFrame with rolling features
        """
        if feature_names is None:
            feature_names = list(data.columns)
        
        tprint_info(f"📊 Computing rolling features for {len(feature_names)} features")
        tprint_info(f"📊 Windows: {windows}")
        
        # Initialize statistics
        self.computation_stats = {
            'total_features': len(feature_names),
            'total_windows': len(windows),
            'cached_computations': 0,
            'new_computations': 0,
            'processing_time': 0
        }
        
        start_time = time.time()
        
        # Compute rolling features
        rolling_features = {}
        
        for feature in feature_names:
            if feature not in data.columns:
                continue
            
            feature_data = data[feature].values
            
            # Compute rolling features for this feature
            feature_rolling = self._compute_feature_rolling(
                feature_data, feature, windows
            )
            
            rolling_features.update(feature_rolling)
        
        # Update statistics
        self.computation_stats['processing_time'] = time.time() - start_time
        
        tprint_success(f"✅ Computed {len(rolling_features)} rolling features")
        tprint_info(f"📊 Cached computations: {self.computation_stats['cached_computations']}")
        tprint_info(f"📊 New computations: {self.computation_stats['new_computations']}")
        
        return pd.DataFrame(rolling_features, index=data.index)
    
    def _compute_feature_rolling(self, 
                                data: np.ndarray,
                                feature_name: str,
                                windows: List[int]) -> Dict[str, np.ndarray]:
        """Compute rolling features for a single feature."""
        rolling_features = {}
        
        # Compute prefix sums for this feature
        prefix_sums = self._get_or_compute_prefix_sums(data, feature_name)
        
        for window in windows:
            # Compute rolling statistics using prefix sums
            rolling_stats = self._compute_rolling_from_prefix_sums(
                data, prefix_sums, window, feature_name
            )
            
            # Add to results
            for stat_name, stat_values in rolling_stats.items():
                rolling_features[f'{feature_name}_{stat_name}_{window}'] = stat_values
        
        return rolling_features
    
    def _get_or_compute_prefix_sums(self, 
                                   data: np.ndarray,
                                   feature_name: str) -> Dict[str, np.ndarray]:
        """Get or compute prefix sums for a feature."""
        cache_key = f'{feature_name}_prefix_sums'
        
        if cache_key in self.prefix_sums_cache:
            self.computation_stats['cached_computations'] += 1
            return self.prefix_sums_cache[cache_key]
        
        # Compute prefix sums
        prefix_sums = self._compute_prefix_sums(data)
        
        # Cache the result
        if self.config.cache_prefix_sums:
            self.prefix_sums_cache[cache_key] = prefix_sums
        
        self.computation_stats['new_computations'] += 1
        return prefix_sums
    
    def _compute_prefix_sums(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute prefix sums for efficient rolling calculations."""
        n = len(data)
        
        # Handle NaN values
        valid_mask = ~np.isnan(data)
        data_clean = np.where(valid_mask, data, 0.0)
        
        # Compute cumulative sums
        cumsum = np.cumsum(data_clean)
        cumsum_sq = np.cumsum(data_clean ** 2)
        cumsum_abs = np.cumsum(np.abs(data_clean))
        
        # Compute cumulative counts of valid values
        cumcount = np.cumsum(valid_mask.astype(int))
        
        return {
            'cumsum': cumsum,
            'cumsum_sq': cumsum_sq,
            'cumsum_abs': cumsum_abs,
            'cumcount': cumcount,
            'valid_mask': valid_mask
        }
    
    def _compute_rolling_from_prefix_sums(self, 
                                        data: np.ndarray,
                                        prefix_sums: Dict[str, np.ndarray],
                                        window: int,
                                        feature_name: str) -> Dict[str, np.ndarray]:
        """Compute rolling statistics from prefix sums."""
        n = len(data)
        rolling_stats = {}
        
        # Get prefix sums
        cumsum = prefix_sums['cumsum']
        cumsum_sq = prefix_sums['cumsum_sq']
        cumsum_abs = prefix_sums['cumsum_abs']
        cumcount = prefix_sums['cumcount']
        valid_mask = prefix_sums['valid_mask']
        
        # Compute rolling mean
        rolling_mean = np.full(n, np.nan)
        for i in range(window - 1, n):
            start_idx = i - window + 1
            if start_idx >= 0:
                count = cumcount[i] - (cumcount[start_idx - 1] if start_idx > 0 else 0)
                if count > 0:
                    sum_val = cumsum[i] - (cumsum[start_idx - 1] if start_idx > 0 else 0)
                    rolling_mean[i] = sum_val / count
        
        rolling_stats['mean'] = rolling_mean
        
        # Compute rolling standard deviation
        rolling_std = np.full(n, np.nan)
        for i in range(window - 1, n):
            start_idx = i - window + 1
            if start_idx >= 0:
                count = cumcount[i] - (cumcount[start_idx - 1] if start_idx > 0 else 0)
                if count > 1:
                    sum_val = cumsum[i] - (cumsum[start_idx - 1] if start_idx > 0 else 0)
                    sum_sq = cumsum_sq[i] - (cumsum_sq[start_idx - 1] if start_idx > 0 else 0)
                    
                    mean_val = sum_val / count
                    variance = (sum_sq / count) - (mean_val ** 2)
                    rolling_std[i] = np.sqrt(max(0, variance))
        
        rolling_stats['std'] = rolling_std
        
        # Compute rolling min/max using sliding window
        rolling_min = np.full(n, np.nan)
        rolling_max = np.full(n, np.nan)
        
        for i in range(window - 1, n):
            start_idx = i - window + 1
            if start_idx >= 0:
                window_data = data[start_idx:i+1]
                valid_window_data = window_data[valid_mask[start_idx:i+1]]
                
                if len(valid_window_data) > 0:
                    rolling_min[i] = np.min(valid_window_data)
                    rolling_max[i] = np.max(valid_window_data)
        
        rolling_stats['min'] = rolling_min
        rolling_stats['max'] = rolling_max
        
        return rolling_stats
    
    def compute_ema_features(self, 
                           data: pd.DataFrame,
                           periods: List[int],
                           feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compute EMA features with reuse.
        
        Args:
            data: Input data
            periods: List of EMA periods
            feature_names: List of feature names to process
            
        Returns:
            DataFrame with EMA features
        """
        if feature_names is None:
            feature_names = list(data.columns)
        
        tprint_info(f"📊 Computing EMA features for {len(feature_names)} features")
        tprint_info(f"📊 Periods: {periods}")
        
        ema_features = {}
        
        for feature in feature_names:
            if feature not in data.columns:
                continue
            
            feature_data = data[feature].values
            
            # Compute EMA features for this feature
            feature_emas = self._compute_feature_emas(
                feature_data, feature, periods
            )
            
            ema_features.update(feature_emas)
        
        return pd.DataFrame(ema_features, index=data.index)
    
    def _compute_feature_emas(self, 
                             data: np.ndarray,
                             feature_name: str,
                             periods: List[int]) -> Dict[str, np.ndarray]:
        """Compute EMA features for a single feature."""
        ema_features = {}
        
        for period in periods:
            # Check cache
            cache_key = f'{feature_name}_ema_{period}'
            
            if cache_key in self.ema_cache:
                ema_features[f'{feature_name}_ema_{period}'] = self.ema_cache[cache_key]
                self.computation_stats['cached_computations'] += 1
                continue
            
            # Compute EMA
            ema = self._compute_ema(data, period)
            ema_features[f'{feature_name}_ema_{period}'] = ema
            
            # Cache the result
            if self.config.cache_emas:
                self.ema_cache[cache_key] = ema
            
            self.computation_stats['new_computations'] += 1
        
        return ema_features
    
    def _compute_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Compute Exponential Moving Average."""
        alpha = 2.0 / (period + 1)
        ema = np.full_like(data, np.nan)
        
        # Find first valid value
        first_valid = np.argmax(~np.isnan(data))
        if first_valid < len(data):
            ema[first_valid] = data[first_valid]
            
            # Compute EMA for remaining values
            for i in range(first_valid + 1, len(data)):
                if not np.isnan(data[i]):
                    ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
                else:
                    ema[i] = ema[i-1]
        
        return ema
    
    def compute_technical_indicators(self, 
                                   data: pd.DataFrame,
                                   feature_name: str) -> pd.DataFrame:
        """
        Compute technical indicators with shared computations.
        
        Args:
            data: Input data with OHLCV columns
            feature_name: Name of the price feature to use
            
        Returns:
            DataFrame with technical indicators
        """
        if feature_name not in data.columns:
            raise ValueError(f"Feature {feature_name} not found in data")
        
        price_data = data[feature_name].values
        indicators = {}
        
        # Compute shared EMAs for multiple indicators
        ema_12 = self._compute_ema(price_data, 12)
        ema_26 = self._compute_ema(price_data, 26)
        
        # RSI (using EMA for smoothing)
        rsi = self._compute_rsi(price_data, 14)
        indicators[f'{feature_name}_rsi_14'] = rsi
        
        # MACD (reusing EMAs)
        macd_line = ema_12 - ema_26
        macd_signal = self._compute_ema(macd_line, 9)
        macd_histogram = macd_line - macd_signal
        
        indicators[f'{feature_name}_macd_line'] = macd_line
        indicators[f'{feature_name}_macd_signal'] = macd_signal
        indicators[f'{feature_name}_macd_histogram'] = macd_histogram
        
        # Bollinger Bands (using rolling mean and std)
        bb_period = 20
        bb_std = 2
        
        # Get rolling statistics
        prefix_sums = self._get_or_compute_prefix_sums(price_data, feature_name)
        rolling_stats = self._compute_rolling_from_prefix_sums(
            price_data, prefix_sums, bb_period, feature_name
        )
        
        bb_middle = rolling_stats['mean']
        bb_std_vals = rolling_stats['std']
        bb_upper = bb_middle + (bb_std_vals * bb_std)
        bb_lower = bb_middle - (bb_std_vals * bb_std)
        
        indicators[f'{feature_name}_bb_upper'] = bb_upper
        indicators[f'{feature_name}_bb_middle'] = bb_middle
        indicators[f'{feature_name}_bb_lower'] = bb_lower
        indicators[f'{feature_name}_bb_width'] = bb_upper - bb_lower
        indicators[f'{feature_name}_bb_position'] = (price_data - bb_lower) / (bb_upper - bb_lower + 1e-8)
        
        return pd.DataFrame(indicators, index=data.index)
    
    def _compute_rsi(self, data: np.ndarray, period: int) -> np.ndarray:
        """Compute Relative Strength Index."""
        # Calculate price changes
        price_changes = np.diff(data)
        
        # Separate gains and losses
        gains = np.where(price_changes > 0, price_changes, 0)
        losses = np.where(price_changes < 0, -price_changes, 0)
        
        # Compute EMA of gains and losses
        avg_gains = self._compute_ema(np.concatenate([[0], gains]), period)
        avg_losses = self._compute_ema(np.concatenate([[0], losses]), period)
        
        # Calculate RSI
        rs = avg_gains / (avg_losses + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def clear_cache(self):
        """Clear all caches."""
        self.ema_cache.clear()
        self.prefix_sums_cache.clear()
        self.rolling_cache.clear()
        tprint_info("📊 Caches cleared")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'ema_cache_size': len(self.ema_cache),
            'prefix_sums_cache_size': len(self.prefix_sums_cache),
            'rolling_cache_size': len(self.rolling_cache),
            'total_cache_size': len(self.ema_cache) + len(self.prefix_sums_cache) + len(self.rolling_cache)
        }
    
    def get_computation_statistics(self) -> Dict[str, Any]:
        """Get computation statistics."""
        return self.computation_stats


# Global instance
_prefix_sums_reuse = None

def get_prefix_sums_reuse() -> PrefixSumsReuse:
    """Get the global prefix sums reuse instance."""
    global _prefix_sums_reuse
    if _prefix_sums_reuse is None:
        config = PrefixSumsConfig()
        _prefix_sums_reuse = PrefixSumsReuse(config)
    return _prefix_sums_reuse

def compute_rolling_features_reuse(data: pd.DataFrame,
                                 windows: List[int],
                                 feature_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compute rolling features with prefix sums reuse.
    
    Args:
        data: Input data
        windows: List of rolling windows
        feature_names: List of feature names to process
        
    Returns:
        DataFrame with rolling features
    """
    reuse = get_prefix_sums_reuse()
    return reuse.compute_rolling_features(data, windows, feature_names)

def compute_ema_features_reuse(data: pd.DataFrame,
                             periods: List[int],
                             feature_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compute EMA features with reuse.
    
    Args:
        data: Input data
        periods: List of EMA periods
        feature_names: List of feature names to process
        
    Returns:
        DataFrame with EMA features
    """
    reuse = get_prefix_sums_reuse()
    return reuse.compute_ema_features(data, periods, feature_names)