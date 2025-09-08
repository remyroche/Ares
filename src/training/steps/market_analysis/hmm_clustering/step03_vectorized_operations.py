#!/usr/bin/env python3
"""Vectorized Operations and Numba JIT Optimizations for Step03.

This module provides highly optimized, vectorized implementations of common
operations used in HMM regime discovery and clustering. Includes Numba JIT
compilation for maximum performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
import logging

from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

# Performance optimization imports
try:
    from numba import jit, prange, vectorize, float64, int64
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def prange(*args):
        return range(*args)

    def vectorize(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    float64 = None
    int64 = None

logger = logging.getLogger(__name__)

# NUMBA JIT-COMPILED FUNCTIONS

@jit(nopython=True, parallel=True, cache=True)
def vectorized_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling mean using vectorized operations."""
    n = len(values)
    result = np.zeros(n, dtype=np.float64)

    for i in prange(window - 1, n):
        result[i] = np.mean(values[i - window + 1:i + 1])

    return result

@jit(nopython=True, parallel=True, cache=True)
def vectorized_rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling standard deviation using vectorized operations."""
    n = len(values)
    result = np.zeros(n, dtype=np.float64)

    for i in prange(window - 1, n):
        window_data = values[i - window + 1:i + 1]
        result[i] = np.std(window_data)

    return result

@jit(nopython=True, parallel=True, cache=True)
def vectorized_rolling_skewness(values: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling skewness using vectorized operations."""
    n = len(values)
    result = np.zeros(n, dtype=np.float64)

    for i in prange(window - 1, n):
        window_data = values[i - window + 1:i + 1]
        mean_val = np.mean(window_data)
        std_val = np.std(window_data)

        if std_val > 1e-10:
            skewness = np.mean(((window_data - mean_val) / std_val) ** 3)
            result[i] = skewness

    return result

@jit(nopython=True, parallel=True, cache=True)
def vectorized_rolling_kurtosis(values: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling kurtosis using vectorized operations."""
    n = len(values)
    result = np.zeros(n, dtype=np.float64)

    for i in prange(window - 1, n):
        window_data = values[i - window + 1:i + 1]
        mean_val = np.mean(window_data)
        std_val = np.std(window_data)

        if std_val > 1e-10:
            kurtosis = np.mean(((window_data - mean_val) / std_val) ** 4) - 3.0
            result[i] = kurtosis

    return result

@jit(nopython=True, cache=True)
def vectorized_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute RSI using vectorized operations."""
    n = len(prices)
    rsi = np.zeros(n, dtype=np.float64)

    if n < period + 1:
        return rsi

    # Calculate price changes
    deltas = np.zeros(n, dtype=np.float64)
    deltas[1:] = prices[1:] - prices[:-1]

    # Calculate gains and losses
    gains = np.maximum(deltas, 0)
    losses = np.maximum(-deltas, 0)

    # Calculate RSI using exponential moving average approach
    # This avoids the variable modification issue in Numba
    alpha = 1.0 / period

    # Calculate initial values
    avg_gain = np.mean(gains[1:period+1])
    avg_loss = np.mean(losses[1:period+1])

    if avg_loss != 0:
        rs = avg_gain / avg_loss
        rsi[period] = 100 - (100 / (1 + rs))
    else:
        rsi[period] = 100

    # Calculate subsequent values using EMA
    for i in range(period + 1, n):
        avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
        avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss

        if avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
        else:
            rsi[i] = 100

    return rsi

@jit(nopython=True, parallel=True, cache=True)
def vectorized_macd(prices: np.ndarray, fast_period: int = 12,
                   slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute MACD using vectorized operations."""
    n = len(prices)

    # Calculate EMAs
    fast_ema = vectorized_ema(prices, fast_period)
    slow_ema = vectorized_ema(prices, slow_period)

    # Calculate MACD line
    macd_line = fast_ema - slow_ema

    # Calculate signal line (EMA of MACD line)
    signal_line = vectorized_ema(macd_line, signal_period)

    # Calculate histogram
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram

@jit(nopython=True, cache=True)
def vectorized_ema(values: np.ndarray, period: int) -> np.ndarray:
    """Compute EMA using vectorized operations."""
    n = len(values)
    ema = np.zeros(n, dtype=np.float64)

    if n < period:
        return ema

    # Calculate multiplier
    multiplier = 2.0 / (period + 1)

    # Initialize with SMA
    ema[period-1] = np.mean(values[:period])

    # Calculate subsequent EMAs
    for i in range(period, n):
        ema[i] = (values[i] - ema[i-1]) * multiplier + ema[i-1]

    return ema

@jit(nopython=True, parallel=True, cache=True)
def vectorized_bollinger_bands(prices: np.ndarray, window: int = 20,
                              num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Bollinger Bands using vectorized operations."""
    n = len(prices)

    upper_band = np.zeros(n, dtype=np.float64)
    middle_band = np.zeros(n, dtype=np.float64)
    lower_band = np.zeros(n, dtype=np.float64)

    for i in prange(window - 1, n):
        window_data = prices[i - window + 1:i + 1]
        mean_val = np.mean(window_data)
        std_val = np.std(window_data)

        middle_band[i] = mean_val
        upper_band[i] = mean_val + num_std * std_val
        lower_band[i] = mean_val - num_std * std_val

    return upper_band, middle_band, lower_band

@jit(nopython=True, parallel=True, cache=True)
def vectorized_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                  period: int = 14) -> np.ndarray:
    """Compute Average True Range using vectorized operations."""
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)

    if n < period + 1:
        return atr

    # Calculate True Range
    tr = np.zeros(n, dtype=np.float64)
    tr[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )

    # Calculate initial ATR
    atr[period] = np.mean(tr[1:period+1])

    # Calculate subsequent ATR values
    for i in prange(period + 1, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

    return atr

@jit(nopython=True, parallel=True, cache=True)
def vectorized_correlation_matrix(data: np.ndarray) -> np.ndarray:
    """Compute correlation matrix using vectorized operations."""
    n_features = data.shape[1]
    corr_matrix = np.zeros((n_features, n_features), dtype=np.float64)

    for i in prange(n_features):
        for j in prange(i, n_features):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                # Compute correlation between columns i and j
                col_i = data[:, i]
                col_j = data[:, j]

                # Remove NaN values
                valid_mask = ~(np.isnan(col_i) | np.isnan(col_j))
                if np.sum(valid_mask) > 1:
                    col_i_clean = col_i[valid_mask]
                    col_j_clean = col_j[valid_mask]

                    corr = np.corrcoef(col_i_clean, col_j_clean)[0, 1]
                    corr_matrix[i, j] = corr if not np.isnan(corr) else 0.0
                    corr_matrix[j, i] = corr_matrix[i, j]

    return corr_matrix

@jit(nopython=True, parallel=True, cache=True)
def vectorized_distance_matrix(data: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    """Compute distance matrix using vectorized operations."""
    n_samples = data.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples), dtype=np.float64)

    if metric == 'euclidean':
        for i in prange(n_samples):
            for j in prange(i + 1, n_samples):
                diff = data[i] - data[j]
                dist = np.sqrt(np.sum(diff ** 2))
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
    elif metric == 'manhattan':
        for i in prange(n_samples):
            for j in prange(i + 1, n_samples):
                diff = np.abs(data[i] - data[j])
                dist = np.sum(diff)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

    return dist_matrix

@jit(nopython=True, parallel=True, cache=True)
def vectorized_zscore_normalization(data: np.ndarray) -> np.ndarray:
    """Perform z-score normalization using vectorized operations."""
    n_features = data.shape[1]
    normalized_data = np.zeros_like(data, dtype=np.float64)

    for i in prange(n_features):
        col = data[:, i]

        # Remove NaN values for calculation
        valid_mask = ~np.isnan(col)
        if np.sum(valid_mask) > 0:
            valid_values = col[valid_mask]
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)

            if std_val > 1e-10:
                normalized_col = (col - mean_val) / std_val
                # Keep NaN values as NaN
                normalized_data[:, i] = np.where(valid_mask, normalized_col, np.nan)
            else:
                normalized_data[:, i] = col
        else:
            normalized_data[:, i] = col

    return normalized_data

@jit(nopython=True, parallel=True, cache=True)
def vectorized_robust_scaling(data: np.ndarray) -> np.ndarray:
    """Perform robust scaling using vectorized operations."""
    n_features = data.shape[1]
    scaled_data = np.zeros_like(data, dtype=np.float64)

    for i in prange(n_features):
        col = data[:, i]

        # Remove NaN values for calculation
        valid_mask = ~np.isnan(col)
        if np.sum(valid_mask) > 0:
            valid_values = col[valid_mask]
            median_val = np.median(valid_values)
            mad_val = np.median(np.abs(valid_values - median_val))

            if mad_val > 1e-10:
                scaled_col = (col - median_val) / mad_val
                # Keep NaN values as NaN
                scaled_data[:, i] = np.where(valid_mask, scaled_col, np.nan)
            else:
                scaled_data[:, i] = col
        else:
            scaled_data[:, i] = col

    return scaled_data

# VECTORIZED FEATURE ENGINEERING FUNCTIONS

class VectorizedFeatureEngineer:
    """Vectorized feature engineering with Numba optimization."""

    def __init__(self):
        self.logger = logging.getLogger('VectorizedFeatureEngineer')

    @log_all_calls
    def compute_technical_indicators(self, df: pd.DataFrame,
                                   config: Dict[str, Any]) -> pd.DataFrame:
        """Compute technical indicators using vectorized operations."""

        result_df = df.copy()

        # Extract price data
        if 'close' in df.columns:
            prices = df['close'].values
        else:
            raise ValueError("DataFrame must contain 'close' column")

        # RSI
        if config.get('compute_rsi', True):
            rsi_values = vectorized_rsi(prices, config.get('rsi_period', 14))
            result_df['rsi'] = rsi_values

        # MACD
        if config.get('compute_macd', True):
            macd_line, signal_line, histogram = vectorized_macd(
                prices,
                config.get('macd_fast', 12),
                config.get('macd_slow', 26),
                config.get('macd_signal', 9)
            )
            result_df['macd'] = macd_line
            result_df['macd_signal'] = signal_line
            result_df['macd_histogram'] = histogram

        # Bollinger Bands
        if config.get('compute_bollinger', True):
            upper, middle, lower = vectorized_bollinger_bands(
                prices,
                config.get('bollinger_window', 20),
                config.get('bollinger_std', 2.0)
            )
            result_df['bb_upper'] = upper
            result_df['bb_middle'] = middle
            result_df['bb_lower'] = lower
            result_df['bb_width'] = (upper - lower) / middle

        # ATR (requires high, low, close)
        if (config.get('compute_atr', True) and
            'high' in df.columns and 'low' in df.columns):

            atr_values = vectorized_atr(
                df['high'].values,
                df['low'].values,
                prices,
                config.get('atr_period', 14)
            )
            result_df['atr'] = atr_values

        # Rolling statistics
        if config.get('compute_rolling_stats', True):
            window = config.get('rolling_window', 20)

            for col in ['close', 'volume'] if 'volume' in df.columns else ['close']:
                if col in df.columns:
                    values = df[col].values

                    result_df[f'{col}_mean_{window}'] = vectorized_rolling_mean(values, window)
                    result_df[f'{col}_std_{window}'] = vectorized_rolling_std(values, window)

                    if config.get('compute_higher_moments', True):
                        result_df[f'{col}_skew_{window}'] = vectorized_rolling_skewness(values, window)
                        result_df[f'{col}_kurt_{window}'] = vectorized_rolling_kurtosis(values, window)

        return result_df

    @log_all_calls
    def normalize_features(self, features: np.ndarray,
                          method: str = 'zscore') -> np.ndarray:
        """Normalize features using vectorized operations."""

        if method == 'zscore':
            return vectorized_zscore_normalization(features)
        elif method == 'robust':
            return vectorized_robust_scaling(features)
        elif method == 'minmax':
            # Vectorized min-max scaling
            min_vals = np.nanmin(features, axis=0)
            max_vals = np.nanmax(features, axis=0)
            ranges = max_vals - min_vals

            # Avoid division by zero
            ranges = np.where(ranges == 0, 1.0, ranges)

            return (features - min_vals) / ranges
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    @log_all_calls
    def compute_correlation_features(self, features: np.ndarray,
                                   threshold: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """Compute correlation-based features using vectorized operations."""

        # Compute correlation matrix
        corr_matrix = vectorized_correlation_matrix(features)

        # Find highly correlated feature pairs
        high_corr_mask = np.abs(corr_matrix) > threshold
        np.fill_diagonal(high_corr_mask, False)  # Remove diagonal

        # Compute correlation statistics
        corr_stats = np.zeros(features.shape[1], dtype=np.float64)

        for i in range(features.shape[1]):
            # Average absolute correlation with other features
            corr_values = corr_matrix[i, :]
            corr_values = corr_values[~np.isnan(corr_values)]
            corr_stats[i] = np.mean(np.abs(corr_values)) if len(corr_values) > 0 else 0.0

        return corr_matrix, corr_stats

    @log_all_calls
    def compute_distance_features(self, features: np.ndarray,
                                metric: str = 'euclidean') -> np.ndarray:
        """Compute distance-based features using vectorized operations."""

        # Compute distance matrix
        dist_matrix = vectorized_distance_matrix(features, metric)

        # Compute distance statistics for each sample
        dist_stats = np.zeros((features.shape[0], 4), dtype=np.float64)

        for i in range(features.shape[0]):
            distances = dist_matrix[i, :]
            valid_distances = distances[~np.isnan(distances)]

            if len(valid_distances) > 0:
                dist_stats[i, 0] = np.mean(valid_distances)      # Mean distance
                dist_stats[i, 1] = np.std(valid_distances)       # Std distance
                dist_stats[i, 2] = np.min(valid_distances)       # Min distance
                dist_stats[i, 3] = np.max(valid_distances)       # Max distance
            else:
                dist_stats[i, :] = np.nan

        return dist_stats

# PERFORMANCE UTILITIES

class PerformanceProfiler:
    """Performance profiler for vectorized operations."""

    def __init__(self):
        self.logger = logging.getLogger('PerformanceProfiler')
        self.timing_stats = {}

    def time_function(self, func_name: str):
        """Decorator to time function execution."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                import time
                start_time = time.time()

                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time

                    if func_name not in self.timing_stats:
                        self.timing_stats[func_name] = []

                    self.timing_stats[func_name].append(execution_time)

                    # Keep only last 100 measurements
                    if len(self.timing_stats[func_name]) > 100:
                        self.timing_stats[func_name] = self.timing_stats[func_name][-100:]

                    return result

                except Exception as e:
                    execution_time = time.time() - start_time
                    self.logger.warning(f"Function {func_name} failed after {execution_time:.4f}s: {e}")
                    raise

            return wrapper
        return decorator

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {}

        for func_name, times in self.timing_stats.items():
            if times:
                stats[func_name] = {
                    'mean_time': np.mean(times),
                    'std_time': np.std(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'call_count': len(times)
                }

        return stats

    def log_performance_summary(self):
        """Log performance summary."""
        stats = self.get_performance_stats()

        self.logger.info("🚀 Vectorized Operations Performance Summary:")

        for func_name, func_stats in stats.items():
            self.logger.info(f"  {func_name}:")
            self.logger.info(f"    Mean: {func_stats['mean_time']:.4f}s")
            self.logger.info(f"    Std: {func_stats['std_time']:.4f}s")
            self.logger.info(f"    Calls: {func_stats['call_count']}")

# MAIN INTERFACE

class VectorizedOperationsManager:
    """Main interface for vectorized operations."""

    def __init__(self):
        self.feature_engineer = VectorizedFeatureEngineer()
        self.profiler = PerformanceProfiler()
        self.logger = logging.getLogger('VectorizedOperationsManager')

        if not NUMBA_AVAILABLE:
            self.logger.warning("⚠️ Numba not available - using fallback implementations")

    @log_important_calls
    def process_dataset(self, data: Union[pd.DataFrame, np.ndarray],
                       config: Dict[str, Any]) -> Union[pd.DataFrame, np.ndarray]:
        """Process dataset using vectorized operations."""

        self.logger.info("🚀 Processing dataset with vectorized operations")

        # Feature engineering for DataFrames
        if isinstance(data, pd.DataFrame):
            result = self.feature_engineer.compute_technical_indicators(data, config)

            # Normalize if requested
            if config.get('normalize_features', False):
                features_array = result.select_dtypes(include=[np.number]).values
                normalized_features = self.feature_engineer.normalize_features(
                    features_array, config.get('normalization_method', 'zscore')
                )

                # Update DataFrame with normalized values
                numeric_cols = result.select_dtypes(include=[np.number]).columns
                for i, col in enumerate(numeric_cols):
                    result[col] = normalized_features[:, i]

            return result

        # Feature processing for numpy arrays
        elif isinstance(data, np.ndarray):
            # Normalization
            if config.get('normalize_features', False):
                data = self.feature_engineer.normalize_features(
                    data, config.get('normalization_method', 'zscore')
                )

            # Distance features
            if config.get('compute_distance_features', False):
                dist_features = self.feature_engineer.compute_distance_features(
                    data, config.get('distance_metric', 'euclidean')
                )

                # Concatenate original features with distance features
                data = np.column_stack([data, dist_features])

            return data

        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    @log_all_calls
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.profiler.get_performance_stats()

# UTILITY FUNCTIONS

def create_vectorized_config(**kwargs) -> Dict[str, Any]:
    """Create configuration for vectorized operations."""

    default_config = {
        # Technical indicators
        'compute_rsi': True,
        'rsi_period': 14,
        'compute_macd': True,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'compute_bollinger': True,
        'bollinger_window': 20,
        'bollinger_std': 2.0,
        'compute_atr': True,
        'atr_period': 14,

        # Rolling statistics
        'compute_rolling_stats': True,
        'rolling_window': 20,
        'compute_higher_moments': True,

        # Normalization
        'normalize_features': True,
        'normalization_method': 'zscore',  # 'zscore', 'robust', 'minmax'

        # Distance features
        'compute_distance_features': False,
        'distance_metric': 'euclidean'
    }

    # Update with provided kwargs
    default_config.update(kwargs)
    return default_config

# Global instance for easy access
_vectorized_manager = None

def get_vectorized_operations_manager() -> VectorizedOperationsManager:
    """Get global vectorized operations manager instance."""
    global _vectorized_manager
    if _vectorized_manager is None:
        _vectorized_manager = VectorizedOperationsManager()
    return _vectorized_manager

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    import numpy as np

    # Create sample data
    np.random.seed(42)
    n_samples = 1000

    data = pd.DataFrame({
        'close': np.random.randn(n_samples).cumsum() + 100,
        'high': np.random.randn(n_samples).cumsum() + 102,
        'low': np.random.randn(n_samples).cumsum() + 98,
        'volume': np.random.randint(1000, 10000, n_samples)
    })

    # Get manager and config
    manager = get_vectorized_operations_manager()
    config = create_vectorized_config()

    # Process data
    processed_data = manager.process_dataset(data, config)

    print(f"Original data shape: {data.shape}")
    print(f"Processed data shape: {processed_data.shape}")
    print(f"New features: {set(processed_data.columns) - set(data.columns)}")

    # Show performance stats
    manager.profiler.log_performance_summary()
