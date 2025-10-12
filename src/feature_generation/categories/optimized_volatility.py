"""
Optimized Volatility Feature Generator

This module provides optimized feature generators for volatility-based indicators,
including Bollinger Bands, ATR, and other volatility measures.
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


class OptimizedVolatilityFeatureGenerator(OptimizedFeatureGenerator):
    """Optimized feature generator for volatility-based features with comprehensive VectorBT optimization."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize optimized volatility feature generator."""
        if config is None:
            config = self._create_default_config()
        super().__init__(config)
        
        # Volatility-specific parameters
        self.periods = config.get('periods', [10, 20, 50, 100])
        self.atr_period = config.get('atr_period', 14)
        self.bb_period = config.get('bb_period', 20)
        self.bb_std = config.get('bb_std', 2.0)
        self.kc_period = config.get('kc_period', 20)
        self.kc_std = config.get('kc_std', 2.0)
        
        self.logger.info("✅ OptimizedVolatilityFeatureGenerator initialized")
    
    @classmethod
    def _create_default_config(cls) -> Dict[str, Any]:
        """Create default configuration for volatility features."""
        return {
            'name': 'optimized_volatility_features',
            'category': FeatureCategory.VOLATILITY,
            'description': 'Optimized volatility features with VectorBT optimization',
            'required_columns': ['close'],
            'optional_columns': ['high', 'low', 'open', 'volume'],
            'default_lookback': 100,
            'min_lookback': 20,
            'max_lookback': 200,
            'periods': [10, 20, 50, 100],
            'atr_period': 14,
            'bb_period': 20,
            'bb_std': 2.0,
            'kc_period': 20,
            'kc_std': 2.0,
            'matrix_optimized': True,
            'gpu_accelerated': True
        }
    
    def generate_volatility_features_optimized(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate multiple volatility features using optimized batch processing."""
        with self.performance_monitoring("volatility_feature_generation"):
            # Optimize DataFrame for processing
            optimized_data = self.optimize_dataframe_processing(data)
            
            # Define feature configurations for batch processing
            feature_configs = self._create_volatility_feature_configs(optimized_data)
            
            # Generate features using cross-category optimization
            features = self.generate_cross_category_features(optimized_data, feature_configs)
            
            return features
    
    def _create_volatility_feature_configs(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create feature configurations for volatility features."""
        feature_configs = []
        
        # Rolling Standard Deviation
        for period in self.periods:
            feature_configs.append({
                'name': f'volatility_{period}',
                'type': 'rolling',
                'params': {
                    'operation': 'std',
                    'window': period,
                    'column': 'close'
                }
            })
        
        # Rolling Variance
        for period in self.periods:
            feature_configs.append({
                'name': f'variance_{period}',
                'type': 'rolling',
                'params': {
                    'operation': 'var',
                    'window': period,
                    'column': 'close'
                }
            })
        
        # Bollinger Bands
        if 'close' in data.columns:
            feature_configs.extend([
                {
                    'name': 'bb_upper',
                    'type': 'custom',
                    'params': {
                        'function': self._calculate_bollinger_upper,
                        'period': self.bb_period,
                        'std': self.bb_std
                    }
                },
                {
                    'name': 'bb_lower',
                    'type': 'custom',
                    'params': {
                        'function': self._calculate_bollinger_lower,
                        'period': self.bb_period,
                        'std': self.bb_std
                    }
                },
                {
                    'name': 'bb_width',
                    'type': 'custom',
                    'params': {
                        'function': self._calculate_bollinger_width,
                        'period': self.bb_period,
                        'std': self.bb_std
                    }
                },
                {
                    'name': 'bb_position',
                    'type': 'custom',
                    'params': {
                        'function': self._calculate_bollinger_position,
                        'period': self.bb_period,
                        'std': self.bb_std
                    }
                }
            ])
        
        # ATR (Average True Range)
        if 'high' in data.columns and 'low' in data.columns and 'close' in data.columns:
            feature_configs.append({
                'name': f'atr_{self.atr_period}',
                'type': 'custom',
                'params': {
                    'function': self._calculate_atr_optimized,
                    'period': self.atr_period
                }
            })
        
        # Keltner Channels
        if 'high' in data.columns and 'low' in data.columns and 'close' in data.columns:
            feature_configs.extend([
                {
                    'name': 'kc_upper',
                    'type': 'custom',
                    'params': {
                        'function': self._calculate_keltner_upper,
                        'period': self.kc_period,
                        'std': self.kc_std
                    }
                },
                {
                    'name': 'kc_lower',
                    'type': 'custom',
                    'params': {
                        'function': self._calculate_keltner_lower,
                        'period': self.kc_period,
                        'std': self.kc_std
                    }
                },
                {
                    'name': 'kc_width',
                    'type': 'custom',
                    'params': {
                        'function': self._calculate_keltner_width,
                        'period': self.kc_period,
                        'std': self.kc_std
                    }
                }
            ])
        
        # Historical Volatility
        for period in self.periods:
            feature_configs.append({
                'name': f'historical_vol_{period}',
                'type': 'custom',
                'params': {
                    'function': self._calculate_historical_volatility,
                    'period': period
                }
            })
        
        # Parkinson Volatility
        if 'high' in data.columns and 'low' in data.columns:
            feature_configs.append({
                'name': 'parkinson_vol',
                'type': 'custom',
                'params': {
                    'function': self._calculate_parkinson_volatility,
                    'period': 20
                }
            })
        
        return feature_configs
    
    def _calculate_bollinger_upper(self, data: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.Series:
        """Calculate Bollinger Bands upper band."""
        if 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        sma = self.get_cached_rolling_result(close, 'mean', period)
        std_dev = self.get_cached_rolling_result(close, 'std', period)
        
        return sma + (std * std_dev)
    
    def _calculate_bollinger_lower(self, data: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.Series:
        """Calculate Bollinger Bands lower band."""
        if 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        sma = self.get_cached_rolling_result(close, 'mean', period)
        std_dev = self.get_cached_rolling_result(close, 'std', period)
        
        return sma - (std * std_dev)
    
    def _calculate_bollinger_width(self, data: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.Series:
        """Calculate Bollinger Bands width."""
        if 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        std_dev = self.get_cached_rolling_result(close, 'std', period)
        
        return (4 * std * std_dev) / self.get_cached_rolling_result(close, 'mean', period)
    
    def _calculate_bollinger_position(self, data: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.Series:
        """Calculate Bollinger Bands position (0-1 scale)."""
        if 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        upper = self._calculate_bollinger_upper(data, period, std)
        lower = self._calculate_bollinger_lower(data, period, std)
        
        return (close - lower) / (upper - lower)
    
    def _calculate_atr_optimized(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR using optimized batch operations."""
        if 'high' not in data.columns or 'low' not in data.columns or 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index, name=f'atr_{period}')
        
        high = data['high'].astype(float)
        low = data['low'].astype(float)
        close = data['close'].astype(float)
        
        if len(high) < period:
            return pd.Series(np.nan, index=data.index, name=f'atr_{period}')
        
        # Calculate True Range
        tr = np.maximum.reduce([
            high - low,
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        ])
        tr[0] = np.nan  # First value is NaN
        
        # Convert to pandas Series
        tr_series = pd.Series(tr, index=data.index)
        
        # Calculate ATR using cached rolling result
        atr = self.get_cached_rolling_result(tr_series, 'mean', period)
        
        return atr.fillna(0)
    
    def _calculate_keltner_upper(self, data: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.Series:
        """Calculate Keltner Channels upper band."""
        if 'high' not in data.columns or 'low' not in data.columns or 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate typical price
        typical_price = (high + low + close) / 3
        
        # Calculate EMA of typical price
        ema_tp = self.get_cached_rolling_result(typical_price, 'ewm', period, span=period)
        
        # Calculate ATR for volatility
        atr = self._calculate_atr_optimized(data, period)
        
        return ema_tp + (std * atr)
    
    def _calculate_keltner_lower(self, data: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.Series:
        """Calculate Keltner Channels lower band."""
        if 'high' not in data.columns or 'low' not in data.columns or 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate typical price
        typical_price = (high + low + close) / 3
        
        # Calculate EMA of typical price
        ema_tp = self.get_cached_rolling_result(typical_price, 'ewm', period, span=period)
        
        # Calculate ATR for volatility
        atr = self._calculate_atr_optimized(data, period)
        
        return ema_tp - (std * atr)
    
    def _calculate_keltner_width(self, data: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.Series:
        """Calculate Keltner Channels width."""
        upper = self._calculate_keltner_upper(data, period, std)
        lower = self._calculate_keltner_lower(data, period, std)
        
        return upper - lower
    
    def _calculate_historical_volatility(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate historical volatility."""
        if 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        
        # Calculate log returns
        log_returns = np.log(close / close.shift(1))
        
        # Calculate rolling standard deviation
        volatility = self.get_cached_rolling_result(log_returns, 'std', period)
        
        # Annualize (assuming daily data)
        return volatility * np.sqrt(252)
    
    def _calculate_parkinson_volatility(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Parkinson volatility estimator."""
        if 'high' not in data.columns or 'low' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        high = data['high']
        low = data['low']
        
        # Calculate Parkinson volatility
        parkinson = np.log(high / low) ** 2
        
        # Calculate rolling mean
        parkinson_vol = self.get_cached_rolling_result(parkinson, 'mean', period)
        
        # Annualize (assuming daily data)
        return np.sqrt(parkinson_vol * 252 / (4 * np.log(2)))
    
    def generate_bollinger_bands_batch(self, data: pd.DataFrame, periods: List[int], stds: List[float]) -> pd.DataFrame:
        """Generate multiple Bollinger Bands in batch for better performance."""
        feature_configs = []
        
        for period in periods:
            for std in stds:
                feature_configs.extend([
                    {
                        'name': f'bb_upper_{period}_{std}',
                        'type': 'custom',
                        'params': {
                            'function': self._calculate_bollinger_upper,
                            'period': period,
                            'std': std
                        }
                    },
                    {
                        'name': f'bb_lower_{period}_{std}',
                        'type': 'custom',
                        'params': {
                            'function': self._calculate_bollinger_lower,
                            'period': period,
                            'std': std
                        }
                    },
                    {
                        'name': f'bb_width_{period}_{std}',
                        'type': 'custom',
                        'params': {
                            'function': self._calculate_bollinger_width,
                            'period': period,
                            'std': std
                        }
                    }
                ])
        
        return self.generate_cross_category_features(data, feature_configs)
    
    def generate_atr_features_batch(self, data: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Generate multiple ATR features in batch for better performance."""
        feature_configs = []
        
        for period in periods:
            feature_configs.append({
                'name': f'atr_{period}',
                'type': 'custom',
                'params': {
                    'function': self._calculate_atr_optimized,
                    'period': period
                }
            })
        
        return self.generate_cross_category_features(data, feature_configs)
    
    def generate_volatility_ratios(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility ratio features."""
        feature_configs = []
        
        # Short-term vs Long-term volatility ratios
        short_periods = [10, 20]
        long_periods = [50, 100]
        
        for short in short_periods:
            for long in long_periods:
                feature_configs.append({
                    'name': f'vol_ratio_{short}_{long}',
                    'type': 'custom',
                    'params': {
                        'function': self._calculate_volatility_ratio,
                        'short_period': short,
                        'long_period': long
                    }
                })
        
        # ATR vs Price volatility ratios
        feature_configs.append({
            'name': 'atr_price_ratio',
            'type': 'custom',
            'params': {
                'function': self._calculate_atr_price_ratio,
                'period': 20
            }
        })
        
        return self.generate_cross_category_features(data, feature_configs)
    
    def _calculate_volatility_ratio(self, data: pd.DataFrame, short_period: int, long_period: int) -> pd.Series:
        """Calculate volatility ratio between short and long periods."""
        if 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        
        # Calculate returns
        returns = close.pct_change()
        
        # Calculate short and long-term volatility
        vol_short = self.get_cached_rolling_result(returns, 'std', short_period)
        vol_long = self.get_cached_rolling_result(returns, 'std', long_period)
        
        return vol_short / vol_long
    
    def _calculate_atr_price_ratio(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate ATR to price ratio."""
        if 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        atr = self._calculate_atr_optimized(data, period)
        
        return atr / close
    
    def generate_volatility_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility regime features."""
        feature_configs = []
        
        # Volatility percentile ranks
        for period in [20, 50, 100]:
            feature_configs.append({
                'name': f'vol_percentile_{period}',
                'type': 'custom',
                'params': {
                    'function': self._calculate_volatility_percentile,
                    'period': period
                }
            })
        
        # Volatility momentum
        feature_configs.append({
            'name': 'vol_momentum',
            'type': 'custom',
            'params': {
                'function': self._calculate_volatility_momentum,
                'short_period': 10,
                'long_period': 30
            }
        })
        
        # Volatility acceleration
        feature_configs.append({
            'name': 'vol_acceleration',
            'type': 'custom',
            'params': {
                'function': self._calculate_volatility_acceleration,
                'period': 20
            }
        })
        
        return self.generate_cross_category_features(data, feature_configs)
    
    def _calculate_volatility_percentile(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate volatility percentile rank."""
        if 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        returns = close.pct_change()
        volatility = self.get_cached_rolling_result(returns, 'std', period)
        
        # Calculate percentile rank
        return volatility.rolling(period * 2).rank(pct=True)
    
    def _calculate_volatility_momentum(self, data: pd.DataFrame, short_period: int = 10, long_period: int = 30) -> pd.Series:
        """Calculate volatility momentum."""
        if 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        returns = close.pct_change()
        
        vol_short = self.get_cached_rolling_result(returns, 'std', short_period)
        vol_long = self.get_cached_rolling_result(returns, 'std', long_period)
        
        return vol_short - vol_long
    
    def _calculate_volatility_acceleration(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate volatility acceleration."""
        if 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        returns = close.pct_change()
        volatility = self.get_cached_rolling_result(returns, 'std', period)
        
        # Calculate second derivative (acceleration)
        vol_diff = volatility.diff()
        acceleration = vol_diff.diff()
        
        return acceleration
    
    def generate_all_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate all volatility features with comprehensive optimization."""
        with self.performance_monitoring("comprehensive_volatility_features"):
            # Optimize DataFrame for processing
            optimized_data = self.optimize_dataframe_processing(data)
            
            # Generate all types of volatility features
            features = []
            
            # Basic volatility features
            basic_features = self.generate_volatility_features_optimized(optimized_data)
            features.append(basic_features)
            
            # Volatility ratios
            ratio_features = self.generate_volatility_ratios(optimized_data)
            features.append(ratio_features)
            
            # Volatility regime features
            regime_features = self.generate_volatility_regime_features(optimized_data)
            features.append(regime_features)
            
            # Combine all features
            all_features = pd.concat(features, axis=1)
            
            # Log performance statistics
            stats = self.get_performance_stats()
            self.logger.info(f"Generated {len(all_features.columns)} volatility features")
            self.logger.info(f"VectorBT usage rate: {stats.get('vectorbt_usage_rate', 0):.2%}")
            self.logger.info(f"Cache hit rate: {stats.get('cache_hit_rate', 0):.2f}%")
            self.logger.info(f"Memory optimizations: {stats.get('memory_optimizations', 0)}")
            
            return all_features


# Convenience functions for easy usage
def create_optimized_volatility_generator(periods: List[int] = None, atr_period: int = 14) -> OptimizedVolatilityFeatureGenerator:
    """Create an optimized volatility feature generator with specified parameters."""
    config = {
        'periods': periods or [10, 20, 50, 100],
        'atr_period': atr_period,
        'bb_period': 20,
        'bb_std': 2.0,
        'kc_period': 20,
        'kc_std': 2.0
    }
    return OptimizedVolatilityFeatureGenerator(config)


def generate_volatility_features_optimized(data: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
    """Generate optimized volatility features for the given data."""
    generator = create_optimized_volatility_generator(periods)
    return generator.generate_all_volatility_features(data)


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
    
    print("Testing Optimized Volatility Feature Generator...")
    
    # Create generator
    generator = create_optimized_volatility_generator(periods=[10, 20, 50])
    
    # Generate features
    features = generator.generate_all_volatility_features(data)
    
    print(f"Generated {len(features.columns)} volatility features")
    print(f"Feature columns: {list(features.columns)}")
    
    # Get performance stats
    stats = generator.get_performance_stats()
    print(f"Performance stats: {stats}")
    
    print("✅ Optimized volatility feature generation test completed!")