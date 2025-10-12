"""
Optimized Returns Feature Generator

This module provides optimized feature generators for return-based indicators,
including log returns, cumulative returns, rolling returns, and return statistics.
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


class OptimizedReturnsFeatureGenerator(OptimizedFeatureGenerator):
    """Optimized feature generator for return-based features with comprehensive VectorBT optimization."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize optimized returns feature generator."""
        if config is None:
            config = self._create_default_config()
        super().__init__(config)
        
        # Returns-specific parameters
        self.periods = config.get('periods', [1, 5, 10, 20, 50, 100])
        self.return_types = config.get('return_types', ['simple', 'log', 'cumulative'])
        self.statistics = config.get('statistics', ['mean', 'std', 'skew', 'kurt'])
        
        self.logger.info("✅ OptimizedReturnsFeatureGenerator initialized")
    
    @classmethod
    def _create_default_config(cls) -> Dict[str, Any]:
        """Create default configuration for returns features."""
        return {
            'name': 'optimized_returns_features',
            'category': FeatureCategory.RETURNS,
            'description': 'Optimized returns features with VectorBT optimization',
            'required_columns': ['close'],
            'optional_columns': ['open', 'high', 'low', 'volume'],
            'default_lookback': 100,
            'min_lookback': 20,
            'max_lookback': 200,
            'periods': [1, 5, 10, 20, 50, 100],
            'return_types': ['simple', 'log', 'cumulative'],
            'statistics': ['mean', 'std', 'skew', 'kurt'],
            'matrix_optimized': True,
            'gpu_accelerated': True
        }
    
    def generate_returns_features_optimized(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate multiple returns features using optimized batch processing."""
        with self.performance_monitoring("returns_feature_generation"):
            # Optimize DataFrame for processing
            optimized_data = self.optimize_dataframe_processing(data)
            
            # Define feature configurations for batch processing
            feature_configs = self._create_returns_feature_configs(optimized_data)
            
            # Generate features using cross-category optimization
            features = self.generate_cross_category_features(optimized_data, feature_configs)
            
            return features
    
    def _create_returns_feature_configs(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create feature configurations for returns features."""
        feature_configs = []
        
        # Basic returns
        for period in self.periods:
            for return_type in self.return_types:
                feature_configs.append({
                    'name': f'{return_type}_return_{period}',
                    'type': 'custom',
                    'params': {
                        'function': self._calculate_return,
                        'period': period,
                        'return_type': return_type
                    }
                })
        
        # Rolling return statistics
        for period in self.periods:
            for stat in self.statistics:
                feature_configs.append({
                    'name': f'return_{stat}_{period}',
                    'type': 'custom',
                    'params': {
                        'function': self._calculate_return_statistic,
                        'period': period,
                        'statistic': stat
                    }
                })
        
        # Return volatility features
        for period in self.periods:
            feature_configs.extend([
                {
                    'name': f'return_volatility_{period}',
                    'type': 'custom',
                    'params': {
                        'function': self._calculate_return_volatility,
                        'period': period
                    }
                },
                {
                    'name': f'return_volatility_annualized_{period}',
                    'type': 'custom',
                    'params': {
                        'function': self._calculate_return_volatility_annualized,
                        'period': period
                    }
                }
            ])
        
        # Return momentum features
        for period in self.periods:
            feature_configs.append({
                'name': f'return_momentum_{period}',
                'type': 'custom',
                'params': {
                    'function': self._calculate_return_momentum,
                    'period': period
                }
            })
        
        # Return correlation features
        if 'volume' in data.columns:
            feature_configs.append({
                'name': 'return_volume_correlation',
                'type': 'custom',
                'params': {
                    'function': self._calculate_return_volume_correlation,
                    'period': 20
                }
            })
        
        # Return regime features
        feature_configs.extend([
            {
                'name': 'return_regime',
                'type': 'custom',
                'params': {
                    'function': self._calculate_return_regime,
                    'period': 20
                }
            },
            {
                'name': 'return_trend_strength',
                'type': 'custom',
                'params': {
                    'function': self._calculate_return_trend_strength,
                    'period': 20
                }
            }
        ])
        
        return feature_configs
    
    def _calculate_return(self, data: pd.DataFrame, period: int = 1, return_type: str = 'simple') -> pd.Series:
        """Calculate returns of specified type and period."""
        if 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        
        if return_type == 'simple':
            return close.pct_change(periods=period)
        elif return_type == 'log':
            return np.log(close / close.shift(periods=period))
        elif return_type == 'cumulative':
            return (close / close.iloc[0] - 1).shift(periods=period)
        else:
            raise ValueError(f"Unsupported return type: {return_type}")
    
    def _calculate_return_statistic(self, data: pd.DataFrame, period: int = 20, statistic: str = 'mean') -> pd.Series:
        """Calculate rolling return statistics."""
        if 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        returns = close.pct_change()
        
        if statistic == 'mean':
            return self.get_cached_rolling_result(returns, 'mean', period)
        elif statistic == 'std':
            return self.get_cached_rolling_result(returns, 'std', period)
        elif statistic == 'skew':
            return self.get_cached_rolling_result(returns, 'skew', period)
        elif statistic == 'kurt':
            return self.get_cached_rolling_result(returns, 'kurt', period)
        elif statistic == 'var':
            return self.get_cached_rolling_result(returns, 'var', period)
        elif statistic == 'min':
            return self.get_cached_rolling_result(returns, 'min', period)
        elif statistic == 'max':
            return self.get_cached_rolling_result(returns, 'max', period)
        else:
            raise ValueError(f"Unsupported statistic: {statistic}")
    
    def _calculate_return_volatility(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate rolling return volatility."""
        if 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        returns = close.pct_change()
        
        return self.get_cached_rolling_result(returns, 'std', period)
    
    def _calculate_return_volatility_annualized(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate annualized rolling return volatility."""
        if 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        returns = close.pct_change()
        volatility = self.get_cached_rolling_result(returns, 'std', period)
        
        # Annualize (assuming daily data)
        return volatility * np.sqrt(252)
    
    def _calculate_return_momentum(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate return momentum."""
        if 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        returns = close.pct_change()
        
        # Calculate momentum as sum of returns over period
        return self.get_cached_rolling_result(returns, 'sum', period)
    
    def _calculate_return_volume_correlation(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate correlation between returns and volume."""
        if 'close' not in data.columns or 'volume' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        volume = data['volume']
        
        returns = close.pct_change()
        volume_returns = volume.pct_change()
        
        # Calculate rolling correlation
        return self.get_cached_rolling_result(returns, 'corr', period, other=volume_returns)
    
    def _calculate_return_regime(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate return regime (bull/bear/neutral)."""
        if 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        returns = close.pct_change()
        
        # Calculate rolling mean return
        mean_return = self.get_cached_rolling_result(returns, 'mean', period)
        std_return = self.get_cached_rolling_result(returns, 'std', period)
        
        # Define regime based on mean return and volatility
        regime = np.where(mean_return > std_return, 1,  # Bull market
                         np.where(mean_return < -std_return, -1,  # Bear market
                                 0))  # Neutral market
        
        return pd.Series(regime, index=data.index)
    
    def _calculate_return_trend_strength(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate return trend strength."""
        if 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        returns = close.pct_change()
        
        # Calculate rolling mean and std
        mean_return = self.get_cached_rolling_result(returns, 'mean', period)
        std_return = self.get_cached_rolling_result(returns, 'std', period)
        
        # Trend strength as signal-to-noise ratio
        trend_strength = np.abs(mean_return) / std_return
        
        return trend_strength.fillna(0)
    
    def generate_returns_batch(self, data: pd.DataFrame, periods: List[int], return_types: List[str]) -> pd.DataFrame:
        """Generate multiple returns in batch for better performance."""
        feature_configs = []
        
        for period in periods:
            for return_type in return_types:
                feature_configs.append({
                    'name': f'{return_type}_return_{period}',
                    'type': 'custom',
                    'params': {
                        'function': self._calculate_return,
                        'period': period,
                        'return_type': return_type
                    }
                })
        
        return self.generate_cross_category_features(data, feature_configs)
    
    def generate_return_statistics_batch(self, data: pd.DataFrame, periods: List[int], statistics: List[str]) -> pd.DataFrame:
        """Generate multiple return statistics in batch for better performance."""
        feature_configs = []
        
        for period in periods:
            for statistic in statistics:
                feature_configs.append({
                    'name': f'return_{statistic}_{period}',
                    'type': 'custom',
                    'params': {
                        'function': self._calculate_return_statistic,
                        'period': period,
                        'statistic': statistic
                    }
                })
        
        return self.generate_cross_category_features(data, feature_configs)
    
    def generate_volatility_features_batch(self, data: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Generate multiple volatility features in batch for better performance."""
        feature_configs = []
        
        for period in periods:
            feature_configs.extend([
                {
                    'name': f'return_volatility_{period}',
                    'type': 'custom',
                    'params': {
                        'function': self._calculate_return_volatility,
                        'period': period
                    }
                },
                {
                    'name': f'return_volatility_annualized_{period}',
                    'type': 'custom',
                    'params': {
                        'function': self._calculate_return_volatility_annualized,
                        'period': period
                    }
                }
            ])
        
        return self.generate_cross_category_features(data, feature_configs)
    
    def generate_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum features."""
        feature_configs = []
        
        # Return momentum for different periods
        for period in [5, 10, 20, 50]:
            feature_configs.append({
                'name': f'return_momentum_{period}',
                'type': 'custom',
                'params': {
                    'function': self._calculate_return_momentum,
                    'period': period
                }
            })
        
        # Price momentum (rate of change)
        for period in [5, 10, 20, 50]:
            feature_configs.append({
                'name': f'price_momentum_{period}',
                'type': 'custom',
                'params': {
                    'function': self._calculate_price_momentum,
                    'period': period
                }
            })
        
        return self.generate_cross_category_features(data, feature_configs)
    
    def _calculate_price_momentum(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate price momentum (rate of change)."""
        if 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        
        # Price momentum as (current_price - price_n_periods_ago) / price_n_periods_ago
        momentum = (close - close.shift(periods=period)) / close.shift(periods=period)
        
        return momentum
    
    def generate_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate regime-based features."""
        feature_configs = []
        
        # Return regime
        feature_configs.append({
            'name': 'return_regime',
            'type': 'custom',
            'params': {
                'function': self._calculate_return_regime,
                'period': 20
            }
        })
        
        # Trend strength
        feature_configs.append({
            'name': 'return_trend_strength',
            'type': 'custom',
            'params': {
                'function': self._calculate_return_trend_strength,
                'period': 20
            }
        })
        
        # Volatility regime
        feature_configs.append({
            'name': 'volatility_regime',
            'type': 'custom',
            'params': {
                'function': self._calculate_volatility_regime,
                'period': 20
            }
        })
        
        return self.generate_cross_category_features(data, feature_configs)
    
    def _calculate_volatility_regime(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate volatility regime (high/medium/low)."""
        if 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        returns = close.pct_change()
        volatility = self.get_cached_rolling_result(returns, 'std', period)
        
        # Calculate volatility percentiles
        vol_percentile = volatility.rolling(period * 2).rank(pct=True)
        
        # Define regime based on percentiles
        regime = np.where(vol_percentile > 0.8, 2,  # High volatility
                         np.where(vol_percentile < 0.2, 0,  # Low volatility
                                 1))  # Medium volatility
        
        return pd.Series(regime, index=data.index)
    
    def generate_correlation_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate correlation features."""
        feature_configs = []
        
        # Return-volume correlation
        if 'volume' in data.columns:
            feature_configs.append({
                'name': 'return_volume_correlation',
                'type': 'custom',
                'params': {
                    'function': self._calculate_return_volume_correlation,
                    'period': 20
                }
            })
        
        # Return-high correlation
        if 'high' in data.columns:
            feature_configs.append({
                'name': 'return_high_correlation',
                'type': 'custom',
                'params': {
                    'function': self._calculate_return_high_correlation,
                    'period': 20
                }
            })
        
        # Return-low correlation
        if 'low' in data.columns:
            feature_configs.append({
                'name': 'return_low_correlation',
                'type': 'custom',
                'params': {
                    'function': self._calculate_return_low_correlation,
                    'period': 20
                }
            })
        
        return self.generate_cross_category_features(data, feature_configs)
    
    def _calculate_return_high_correlation(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate correlation between returns and high prices."""
        if 'close' not in data.columns or 'high' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        high = data['high']
        
        returns = close.pct_change()
        high_returns = high.pct_change()
        
        return self.get_cached_rolling_result(returns, 'corr', period, other=high_returns)
    
    def _calculate_return_low_correlation(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate correlation between returns and low prices."""
        if 'close' not in data.columns or 'low' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        low = data['low']
        
        returns = close.pct_change()
        low_returns = low.pct_change()
        
        return self.get_cached_rolling_result(returns, 'corr', period, other=low_returns)
    
    def generate_all_returns_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate all returns features with comprehensive optimization."""
        with self.performance_monitoring("comprehensive_returns_features"):
            # Optimize DataFrame for processing
            optimized_data = self.optimize_dataframe_processing(data)
            
            # Generate all types of returns features
            features = []
            
            # Basic returns features
            basic_features = self.generate_returns_features_optimized(optimized_data)
            features.append(basic_features)
            
            # Momentum features
            momentum_features = self.generate_momentum_features(optimized_data)
            features.append(momentum_features)
            
            # Regime features
            regime_features = self.generate_regime_features(optimized_data)
            features.append(regime_features)
            
            # Correlation features
            correlation_features = self.generate_correlation_features(optimized_data)
            features.append(correlation_features)
            
            # Combine all features
            all_features = pd.concat(features, axis=1)
            
            # Log performance statistics
            stats = self.get_performance_stats()
            self.logger.info(f"Generated {len(all_features.columns)} returns features")
            self.logger.info(f"VectorBT usage rate: {stats.get('vectorbt_usage_rate', 0):.2%}")
            self.logger.info(f"Cache hit rate: {stats.get('cache_hit_rate', 0):.2f}%")
            self.logger.info(f"Memory optimizations: {stats.get('memory_optimizations', 0)}")
            
            return all_features


# Convenience functions for easy usage
def create_optimized_returns_generator(periods: List[int] = None, return_types: List[str] = None) -> OptimizedReturnsFeatureGenerator:
    """Create an optimized returns feature generator with specified parameters."""
    config = {
        'periods': periods or [1, 5, 10, 20, 50, 100],
        'return_types': return_types or ['simple', 'log', 'cumulative'],
        'statistics': ['mean', 'std', 'skew', 'kurt']
    }
    return OptimizedReturnsFeatureGenerator(config)


def generate_returns_features_optimized(data: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
    """Generate optimized returns features for the given data."""
    generator = create_optimized_returns_generator(periods)
    return generator.generate_all_returns_features(data)


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
    
    print("Testing Optimized Returns Feature Generator...")
    
    # Create generator
    generator = create_optimized_returns_generator(periods=[1, 5, 10, 20])
    
    # Generate features
    features = generator.generate_all_returns_features(data)
    
    print(f"Generated {len(features.columns)} returns features")
    print(f"Feature columns: {list(features.columns)}")
    
    # Get performance stats
    stats = generator.get_performance_stats()
    print(f"Performance stats: {stats}")
    
    print("✅ Optimized returns feature generation test completed!")