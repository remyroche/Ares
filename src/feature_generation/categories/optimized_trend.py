"""
Optimized Trend Feature Generator

This module provides optimized feature generators for trend-based indicators,
including moving averages, trend lines, and trend strength measures.
Fully optimized with VectorBTRollingOptimizer and UnifiedVectorizationManager.

Key Features:
- Batch rolling operations using VectorBTRollingOptimizer
- UnifiedVectorizationManager integration for cross-category features
- Memory optimization with data type optimization
- Smart caching for frequently computed operations
- Performance monitoring and statistics
- Cross-category feature generation optimization
"""

import numpy as np
import pandas as pd
import warnings
import logging
import time
from typing import Any, Dict, List, Optional, Union

from ..core.optimized_feature_generator import OptimizedFeatureGenerator
from ..core.feature_generator import FeatureConfig, FeatureCategory

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

logger = logging.getLogger(__name__)


class OptimizedTrendFeatureGenerator(OptimizedFeatureGenerator):
    """Optimized feature generator for trend-based features with comprehensive VectorBT optimization."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize optimized trend feature generator."""
        if config is None:
            config = self._create_default_config()
        super().__init__(config)
        
        # Trend-specific parameters
        self.periods = config.get('periods', [10, 20, 50, 100])
        self.adx_period = config.get('adx_period', 14)
        self.ema_alpha = config.get('ema_alpha', 0.1)
        
        self.logger.info("✅ OptimizedTrendFeatureGenerator initialized")
    
    @classmethod
    def _create_default_config(cls) -> Dict[str, Any]:
        """Create default configuration for trend features."""
        return {
            'name': 'optimized_trend_features',
            'category': FeatureCategory.TREND,
            'description': 'Optimized trend features with VectorBT optimization',
            'required_columns': ['close'],
            'optional_columns': ['high', 'low', 'open', 'volume'],
            'default_lookback': 100,
            'min_lookback': 20,
            'max_lookback': 200,
            'periods': [10, 20, 50, 100],
            'adx_period': 14,
            'ema_alpha': 0.1,
            'matrix_optimized': True,
            'gpu_accelerated': True
        }
    
    def generate_trend_features_optimized(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate multiple trend features using optimized batch processing."""
        with self.performance_monitoring("trend_feature_generation"):
            # Optimize DataFrame for processing
            optimized_data = self.optimize_dataframe_processing(data)
            
            # Define feature configurations for batch processing
            feature_configs = self._create_trend_feature_configs(optimized_data)
            
            # Generate features using cross-category optimization
            features = self.generate_cross_category_features(optimized_data, feature_configs)
            
            return features
    
    def _create_trend_feature_configs(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create feature configurations for trend features."""
        feature_configs = []
        
        # Simple Moving Averages
        for period in self.periods:
            feature_configs.append({
                'name': f'sma_{period}',
                'type': 'rolling',
                'params': {
                    'operation': 'mean',
                    'window': period,
                    'column': 'close'
                }
            })
        
        # Exponential Moving Averages
        for period in self.periods:
            feature_configs.append({
                'name': f'ema_{period}',
                'type': 'rolling',
                'params': {
                    'operation': 'ewm',
                    'window': period,
                    'column': 'close',
                    'span': period
                }
            })
        
        # Rolling Standard Deviation
        for period in self.periods:
            feature_configs.append({
                'name': f'std_{period}',
                'type': 'rolling',
                'params': {
                    'operation': 'std',
                    'window': period,
                    'column': 'close'
                }
            })
        
        # ADX (Average Directional Index) - custom function
        if 'high' in data.columns and 'low' in data.columns:
            feature_configs.append({
                'name': f'adx_{self.adx_period}',
                'type': 'custom',
                'params': {
                    'function': self._calculate_adx_optimized,
                    'period': self.adx_period
                }
            })
        
        # MACD (Moving Average Convergence Divergence) - custom function
        feature_configs.append({
            'name': 'macd',
            'type': 'custom',
            'params': {
                'function': self._calculate_macd_optimized,
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9
            }
        })
        
        # RSI (Relative Strength Index) - custom function
        feature_configs.append({
            'name': 'rsi_14',
            'type': 'custom',
            'params': {
                'function': self._calculate_rsi_optimized,
                'period': 14
            }
        })
        
        return feature_configs
    
    def _calculate_adx_optimized(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX using optimized batch operations."""
        if 'high' not in data.columns or 'low' not in data.columns or 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index, name=f'adx_{period}')
        
        high = data['high'].astype(float)
        low = data['low'].astype(float)
        close = data['close'].astype(float)
        
        if len(high) < period:
            return pd.Series(np.nan, index=data.index, name=f'adx_{period}')
        
        # Calculate True Range
        tr = np.maximum.reduce([
            high - low,
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        ])
        tr[0] = np.nan  # First value is NaN
        
        # Calculate Directional Movement
        dm_plus = np.maximum(high - np.roll(high, 1), 0)
        dm_minus = np.maximum(np.roll(low, 1) - low, 0)
        
        # Convert to pandas Series for rolling operations
        tr_series = pd.Series(tr, index=data.index)
        dm_plus_series = pd.Series(dm_plus, index=data.index)
        dm_minus_series = pd.Series(dm_minus, index=data.index)
        
        # Use batch rolling operations for better performance
        rolling_operations = [
            {'name': 'dm_plus_mean', 'operation': 'mean', 'window': period, 'data': dm_plus_series},
            {'name': 'dm_minus_mean', 'operation': 'mean', 'window': period, 'data': dm_minus_series},
            {'name': 'tr_mean', 'operation': 'mean', 'window': period, 'data': tr_series}
        ]
        
        # Create temporary DataFrame for batch processing
        temp_data = pd.DataFrame({
            'dm_plus': dm_plus_series,
            'dm_minus': dm_minus_series,
            'tr': tr_series
        })
        
        # Use batch rolling operations
        rolling_results = self.batch_rolling_operations(temp_data, [
            {'name': 'dm_plus_mean', 'operation': 'mean', 'window': period, 'column': 'dm_plus'},
            {'name': 'dm_minus_mean', 'operation': 'mean', 'window': period, 'column': 'dm_minus'},
            {'name': 'tr_mean', 'operation': 'mean', 'window': period, 'column': 'tr'}
        ])
        
        dm_plus_mean = rolling_results['dm_plus_mean']
        dm_minus_mean = rolling_results['dm_minus_mean']
        tr_mean = rolling_results['tr_mean']
        
        # Calculate DI+ and DI-
        di_plus = 100 * (dm_plus_mean / tr_mean)
        di_minus = 100 * (dm_minus_mean / tr_mean)
        
        # Calculate DX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        
        # Calculate ADX as smoothed DX
        adx = self.get_cached_rolling_result(dx, 'mean', period)
        
        return adx.fillna(0)
    
    def _calculate_macd_optimized(self, data: pd.DataFrame, fast_period: int = 12, 
                                 slow_period: int = 26, signal_period: int = 9) -> pd.Series:
        """Calculate MACD using optimized operations."""
        if 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index, name='macd')
        
        close = data['close'].astype(float)
        
        if len(close) < slow_period:
            return pd.Series(np.nan, index=data.index, name='macd')
        
        # Calculate EMAs using cached rolling results
        ema_fast = self.get_cached_rolling_result(close, 'ewm', fast_period, span=fast_period)
        ema_slow = self.get_cached_rolling_result(close, 'ewm', slow_period, span=slow_period)
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = self.get_cached_rolling_result(macd_line, 'ewm', signal_period, span=signal_period)
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return histogram
    
    def _calculate_rsi_optimized(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI using optimized operations."""
        if 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index, name=f'rsi_{period}')
        
        close = data['close'].astype(float)
        
        if len(close) < period + 1:
            return pd.Series(np.nan, index=data.index, name=f'rsi_{period}')
        
        # Calculate price changes
        delta = close.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses using cached rolling results
        avg_gains = self.get_cached_rolling_result(gains, 'mean', period)
        avg_losses = self.get_cached_rolling_result(losses, 'mean', period)
        
        # Calculate RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)  # Neutral RSI when no data
    
    def generate_sma_features_batch(self, data: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Generate multiple SMA features in batch for better performance."""
        rolling_configs = []
        
        for period in periods:
            rolling_configs.append({
                'name': f'sma_{period}',
                'operation': 'mean',
                'window': period,
                'column': 'close'
            })
        
        return self.batch_rolling_operations(data, rolling_configs)
    
    def generate_ema_features_batch(self, data: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Generate multiple EMA features in batch for better performance."""
        rolling_configs = []
        
        for period in periods:
            rolling_configs.append({
                'name': f'ema_{period}',
                'operation': 'ewm',
                'window': period,
                'column': 'close',
                'span': period
            })
        
        return self.batch_rolling_operations(data, rolling_configs)
    
    def generate_volatility_features_batch(self, data: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Generate multiple volatility features in batch for better performance."""
        rolling_configs = []
        
        for period in periods:
            rolling_configs.extend([
                {
                    'name': f'std_{period}',
                    'operation': 'std',
                    'window': period,
                    'column': 'close'
                },
                {
                    'name': f'var_{period}',
                    'operation': 'var',
                    'window': period,
                    'column': 'close'
                }
            ])
        
        return self.batch_rolling_operations(data, rolling_configs)
    
    def generate_cross_timeframe_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate cross-timeframe trend features."""
        feature_configs = []
        
        # Short-term features (5, 10 periods)
        short_periods = [5, 10]
        for period in short_periods:
            feature_configs.extend([
                {
                    'name': f'sma_short_{period}',
                    'type': 'rolling',
                    'params': {'operation': 'mean', 'window': period, 'column': 'close'}
                },
                {
                    'name': f'ema_short_{period}',
                    'type': 'rolling',
                    'params': {'operation': 'ewm', 'window': period, 'column': 'close', 'span': period}
                }
            ])
        
        # Medium-term features (20, 50 periods)
        medium_periods = [20, 50]
        for period in medium_periods:
            feature_configs.extend([
                {
                    'name': f'sma_medium_{period}',
                    'type': 'rolling',
                    'params': {'operation': 'mean', 'window': period, 'column': 'close'}
                },
                {
                    'name': f'ema_medium_{period}',
                    'type': 'rolling',
                    'params': {'operation': 'ewm', 'window': period, 'column': 'close', 'span': period}
                }
            ])
        
        # Long-term features (100, 200 periods)
        long_periods = [100, 200]
        for period in long_periods:
            feature_configs.extend([
                {
                    'name': f'sma_long_{period}',
                    'type': 'rolling',
                    'params': {'operation': 'mean', 'window': period, 'column': 'close'}
                },
                {
                    'name': f'ema_long_{period}',
                    'type': 'rolling',
                    'params': {'operation': 'ewm', 'window': period, 'column': 'close', 'span': period}
                }
            ])
        
        return self.generate_cross_category_features(data, feature_configs)
    
    def generate_trend_strength_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trend strength features using optimized operations."""
        feature_configs = []
        
        # Price position relative to moving averages
        for period in [20, 50, 100]:
            feature_configs.extend([
                {
                    'name': f'price_vs_sma_{period}',
                    'type': 'custom',
                    'params': {
                        'function': self._calculate_price_vs_sma,
                        'period': period
                    }
                },
                {
                    'name': f'price_vs_ema_{period}',
                    'type': 'custom',
                    'params': {
                        'function': self._calculate_price_vs_ema,
                        'period': period
                    }
                }
            ])
        
        # Moving average crossovers
        feature_configs.extend([
            {
                'name': 'sma_crossover_20_50',
                'type': 'custom',
                'params': {
                    'function': self._calculate_sma_crossover,
                    'fast_period': 20,
                    'slow_period': 50
                }
            },
            {
                'name': 'ema_crossover_12_26',
                'type': 'custom',
                'params': {
                    'function': self._calculate_ema_crossover,
                    'fast_period': 12,
                    'slow_period': 26
                }
            }
        ])
        
        return self.generate_cross_category_features(data, feature_configs)
    
    def _calculate_price_vs_sma(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate price position relative to SMA."""
        if 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        sma = self.get_cached_rolling_result(close, 'mean', period)
        
        return (close - sma) / sma * 100
    
    def _calculate_price_vs_ema(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate price position relative to EMA."""
        if 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        ema = self.get_cached_rolling_result(close, 'ewm', period, span=period)
        
        return (close - ema) / ema * 100
    
    def _calculate_sma_crossover(self, data: pd.DataFrame, fast_period: int, slow_period: int) -> pd.Series:
        """Calculate SMA crossover signal."""
        if 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        sma_fast = self.get_cached_rolling_result(close, 'mean', fast_period)
        sma_slow = self.get_cached_rolling_result(close, 'mean', slow_period)
        
        # Crossover signal: 1 when fast > slow, -1 when fast < slow, 0 otherwise
        crossover = np.where(sma_fast > sma_slow, 1, np.where(sma_fast < sma_slow, -1, 0))
        
        return pd.Series(crossover, index=data.index)
    
    def _calculate_ema_crossover(self, data: pd.DataFrame, fast_period: int, slow_period: int) -> pd.Series:
        """Calculate EMA crossover signal."""
        if 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        ema_fast = self.get_cached_rolling_result(close, 'ewm', fast_period, span=fast_period)
        ema_slow = self.get_cached_rolling_result(close, 'ewm', slow_period, span=slow_period)
        
        # Crossover signal: 1 when fast > slow, -1 when fast < slow, 0 otherwise
        crossover = np.where(ema_fast > ema_slow, 1, np.where(ema_fast < ema_slow, -1, 0))
        
        return pd.Series(crossover, index=data.index)
    
    def generate_all_trend_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate all trend features with comprehensive optimization."""
        with self.performance_monitoring("comprehensive_trend_features"):
            # Optimize DataFrame for processing
            optimized_data = self.optimize_dataframe_processing(data)
            
            # Generate all types of trend features
            features = []
            
            # Basic trend features
            basic_features = self.generate_trend_features_optimized(optimized_data)
            features.append(basic_features)
            
            # Cross-timeframe features
            cross_timeframe_features = self.generate_cross_timeframe_features(optimized_data)
            features.append(cross_timeframe_features)
            
            # Trend strength features
            strength_features = self.generate_trend_strength_features(optimized_data)
            features.append(strength_features)
            
            # Combine all features
            all_features = pd.concat(features, axis=1)
            
            # Log performance statistics
            stats = self.get_performance_stats()
            self.logger.info(f"Generated {len(all_features.columns)} trend features")
            self.logger.info(f"VectorBT usage rate: {stats.get('vectorbt_usage_rate', 0):.2%}")
            self.logger.info(f"Cache hit rate: {stats.get('cache_hit_rate', 0):.2f}%")
            self.logger.info(f"Memory optimizations: {stats.get('memory_optimizations', 0)}")
            
            return all_features


# Convenience functions for easy usage
def create_optimized_trend_generator(periods: List[int] = None, adx_period: int = 14) -> OptimizedTrendFeatureGenerator:
    """Create an optimized trend feature generator with specified parameters."""
    config = {
        'periods': periods or [10, 20, 50, 100],
        'adx_period': adx_period,
        'ema_alpha': 0.1
    }
    return OptimizedTrendFeatureGenerator(config)


def generate_trend_features_optimized(data: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
    """Generate optimized trend features for the given data."""
    generator = create_optimized_trend_generator(periods)
    return generator.generate_all_trend_features(data)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='1min')
    data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(1000) * 0.01),
        'high': 100 + np.cumsum(np.random.randn(1000) * 0.01) + np.abs(np.random.randn(1000) * 0.5),
        'low': 100 + np.cumsum(np.random.randn(1000) * 0.01) - np.abs(np.random.randn(1000) * 0.5),
        'volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)
    
    print("Testing Optimized Trend Feature Generator...")
    
    # Create generator
    generator = create_optimized_trend_generator(periods=[10, 20, 50])
    
    # Generate features
    features = generator.generate_all_trend_features(data)
    
    print(f"Generated {len(features.columns)} trend features")
    print(f"Feature columns: {list(features.columns)}")
    
    # Get performance stats
    stats = generator.get_performance_stats()
    print(f"Performance stats: {stats}")
    
    print("✅ Optimized trend feature generation test completed!")