"""
Optimized Cross-Category Feature Generator

This module provides a comprehensive feature generator that combines features
from multiple categories (trend, volatility, returns, momentum, etc.) using
VectorBTRollingOptimizer and UnifiedVectorizationManager for maximum performance.

Key Features:
- Batch rolling operations using VectorBTRollingOptimizer
- UnifiedVectorizationManager integration for cross-category features
- Memory optimization with data type optimization
- Smart caching for frequently computed operations
- Performance monitoring and statistics
- Cross-category feature generation optimization
- Comprehensive feature engineering pipeline
"""

import numpy as np
import pandas as pd
import warnings
import logging
import time
from typing import Any, Dict, List, Optional, Union

from ..core.optimized_feature_generator import OptimizedFeatureGenerator
from ..core.feature_generator import FeatureConfig, FeatureCategory

# Import optimized feature generators
from .optimized_trend import OptimizedTrendFeatureGenerator
from .optimized_volatility import OptimizedVolatilityFeatureGenerator
from .optimized_returns import OptimizedReturnsFeatureGenerator

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


class OptimizedCrossCategoryFeatureGenerator(OptimizedFeatureGenerator):
    """Optimized cross-category feature generator with comprehensive VectorBT optimization."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize optimized cross-category feature generator."""
        if config is None:
            config = self._create_default_config()
        super().__init__(config)
        
        # Cross-category specific parameters
        self.enabled_categories = config.get('enabled_categories', ['trend', 'volatility', 'returns', 'momentum', 'volume'])
        self.periods = config.get('periods', [5, 10, 20, 50, 100])
        self.cross_correlations = config.get('cross_correlations', True)
        self.interaction_features = config.get('interaction_features', True)
        self.regime_features = config.get('regime_features', True)
        
        # Initialize category-specific generators
        self.category_generators = {}
        self._initialize_category_generators()
        
        self.logger.info("✅ OptimizedCrossCategoryFeatureGenerator initialized")
    
    @classmethod
    def _create_default_config(cls) -> Dict[str, Any]:
        """Create default configuration for cross-category features."""
        return {
            'name': 'optimized_cross_category_features',
            'category': FeatureCategory.CROSS_CATEGORY,
            'description': 'Optimized cross-category features with VectorBT optimization',
            'required_columns': ['close'],
            'optional_columns': ['open', 'high', 'low', 'volume'],
            'default_lookback': 200,
            'min_lookback': 50,
            'max_lookback': 500,
            'enabled_categories': ['trend', 'volatility', 'returns', 'momentum', 'volume'],
            'periods': [5, 10, 20, 50, 100],
            'cross_correlations': True,
            'interaction_features': True,
            'regime_features': True,
            'matrix_optimized': True,
            'gpu_accelerated': True
        }
    
    def _initialize_category_generators(self):
        """Initialize category-specific feature generators."""
        if 'trend' in self.enabled_categories:
            self.category_generators['trend'] = OptimizedTrendFeatureGenerator({
                'periods': self.periods
            })
        
        if 'volatility' in self.enabled_categories:
            self.category_generators['volatility'] = OptimizedVolatilityFeatureGenerator({
                'periods': self.periods
            })
        
        if 'returns' in self.enabled_categories:
            self.category_generators['returns'] = OptimizedReturnsFeatureGenerator({
                'periods': self.periods
            })
        
        if 'momentum' in self.enabled_categories:
            self.category_generators['momentum'] = self._create_momentum_generator()
        
        if 'volume' in self.enabled_categories:
            self.category_generators['volume'] = self._create_volume_generator()
    
    def _create_momentum_generator(self) -> OptimizedFeatureGenerator:
        """Create momentum feature generator."""
        class MomentumFeatureGenerator(OptimizedFeatureGenerator):
            def __init__(self, config):
                super().__init__(config)
                self.periods = config.get('periods', [5, 10, 20, 50, 100])
            
            def generate_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
                """Generate momentum features."""
                feature_configs = []
                
                for period in self.periods:
                    feature_configs.extend([
                        {
                            'name': f'roc_{period}',
                            'type': 'custom',
                            'params': {
                                'function': self._calculate_rate_of_change,
                                'period': period
                            }
                        },
                        {
                            'name': f'momentum_{period}',
                            'type': 'custom',
                            'params': {
                                'function': self._calculate_momentum,
                                'period': period
                            }
                        }
                    ])
                
                return self.generate_cross_category_features(data, feature_configs)
            
            def _calculate_rate_of_change(self, data: pd.DataFrame, period: int) -> pd.Series:
                """Calculate rate of change."""
                if 'close' not in data.columns:
                    return pd.Series(np.nan, index=data.index)
                
                close = data['close']
                return (close - close.shift(periods=period)) / close.shift(periods=period) * 100
            
            def _calculate_momentum(self, data: pd.DataFrame, period: int) -> pd.Series:
                """Calculate momentum."""
                if 'close' not in data.columns:
                    return pd.Series(np.nan, index=data.index)
                
                close = data['close']
                return close - close.shift(periods=period)
        
        return MomentumFeatureGenerator({'periods': self.periods})
    
    def _create_volume_generator(self) -> OptimizedFeatureGenerator:
        """Create volume feature generator."""
        class VolumeFeatureGenerator(OptimizedFeatureGenerator):
            def __init__(self, config):
                super().__init__(config)
                self.periods = config.get('periods', [5, 10, 20, 50, 100])
            
            def generate_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
                """Generate volume features."""
                feature_configs = []
                
                if 'volume' not in data.columns:
                    return pd.DataFrame(index=data.index)
                
                for period in self.periods:
                    feature_configs.extend([
                        {
                            'name': f'volume_sma_{period}',
                            'type': 'rolling',
                            'params': {
                                'operation': 'mean',
                                'window': period,
                                'column': 'volume'
                            }
                        },
                        {
                            'name': f'volume_ratio_{period}',
                            'type': 'custom',
                            'params': {
                                'function': self._calculate_volume_ratio,
                                'period': period
                            }
                        }
                    ])
                
                return self.generate_cross_category_features(data, feature_configs)
            
            def _calculate_volume_ratio(self, data: pd.DataFrame, period: int) -> pd.Series:
                """Calculate volume ratio."""
                if 'volume' not in data.columns:
                    return pd.Series(np.nan, index=data.index)
                
                volume = data['volume']
                volume_sma = self.get_cached_rolling_result(volume, 'mean', period)
                
                return volume / volume_sma
        
        return VolumeFeatureGenerator({'periods': self.periods})
    
    def generate_cross_category_features_optimized(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate cross-category features using optimized batch processing."""
        with self.performance_monitoring("cross_category_feature_generation"):
            # Optimize DataFrame for processing
            optimized_data = self.optimize_dataframe_processing(data)
            
            # Generate features from each category
            all_features = []
            
            # Generate category-specific features
            for category, generator in self.category_generators.items():
                if category == 'trend':
                    features = generator.generate_all_trend_features(optimized_data)
                elif category == 'volatility':
                    features = generator.generate_all_volatility_features(optimized_data)
                elif category == 'returns':
                    features = generator.generate_all_returns_features(optimized_data)
                elif category == 'momentum':
                    features = generator.generate_momentum_features(optimized_data)
                elif category == 'volume':
                    features = generator.generate_volume_features(optimized_data)
                else:
                    continue
                
                all_features.append(features)
            
            # Generate cross-category features
            if self.cross_correlations:
                cross_corr_features = self._generate_cross_correlation_features(optimized_data)
                all_features.append(cross_corr_features)
            
            if self.interaction_features:
                interaction_features = self._generate_interaction_features(optimized_data)
                all_features.append(interaction_features)
            
            if self.regime_features:
                regime_features = self._generate_regime_features(optimized_data)
                all_features.append(regime_features)
            
            # Combine all features
            if all_features:
                combined_features = pd.concat(all_features, axis=1)
            else:
                combined_features = pd.DataFrame(index=optimized_data.index)
            
            return combined_features
    
    def _generate_cross_correlation_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate cross-correlation features between different price series."""
        feature_configs = []
        
        # Price-volume correlation
        if 'close' in data.columns and 'volume' in data.columns:
            feature_configs.append({
                'name': 'price_volume_correlation',
                'type': 'custom',
                'params': {
                    'function': self._calculate_price_volume_correlation,
                    'period': 20
                }
            })
        
        # High-low correlation
        if 'high' in data.columns and 'low' in data.columns:
            feature_configs.append({
                'name': 'high_low_correlation',
                'type': 'custom',
                'params': {
                    'function': self._calculate_high_low_correlation,
                    'period': 20
                }
            })
        
        # Open-close correlation
        if 'open' in data.columns and 'close' in data.columns:
            feature_configs.append({
                'name': 'open_close_correlation',
                'type': 'custom',
                'params': {
                    'function': self._calculate_open_close_correlation,
                    'period': 20
                }
            })
        
        return self.generate_cross_category_features(data, feature_configs)
    
    def _calculate_price_volume_correlation(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate correlation between price and volume."""
        if 'close' not in data.columns or 'volume' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        volume = data['volume']
        
        close_returns = close.pct_change()
        volume_returns = volume.pct_change()
        
        return self.get_cached_rolling_result(close_returns, 'corr', period, other=volume_returns)
    
    def _calculate_high_low_correlation(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate correlation between high and low prices."""
        if 'high' not in data.columns or 'low' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        high = data['high']
        low = data['low']
        
        high_returns = high.pct_change()
        low_returns = low.pct_change()
        
        return self.get_cached_rolling_result(high_returns, 'corr', period, other=low_returns)
    
    def _calculate_open_close_correlation(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate correlation between open and close prices."""
        if 'open' not in data.columns or 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        open_price = data['open']
        close = data['close']
        
        open_returns = open_price.pct_change()
        close_returns = close.pct_change()
        
        return self.get_cached_rolling_result(open_returns, 'corr', period, other=close_returns)
    
    def _generate_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate interaction features between different indicators."""
        feature_configs = []
        
        # RSI-Volatility interaction
        feature_configs.append({
            'name': 'rsi_volatility_interaction',
            'type': 'custom',
            'params': {
                'function': self._calculate_rsi_volatility_interaction,
                'period': 20
            }
        })
        
        # MACD-Volume interaction
        feature_configs.append({
            'name': 'macd_volume_interaction',
            'type': 'custom',
            'params': {
                'function': self._calculate_macd_volume_interaction,
                'period': 20
            }
        })
        
        # Price-Volatility interaction
        feature_configs.append({
            'name': 'price_volatility_interaction',
            'type': 'custom',
            'params': {
                'function': self._calculate_price_volatility_interaction,
                'period': 20
            }
        })
        
        return self.generate_cross_category_features(data, feature_configs)
    
    def _calculate_rsi_volatility_interaction(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate RSI-Volatility interaction."""
        if 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        returns = close.pct_change()
        
        # Calculate RSI
        rsi = self._calculate_rsi(close, period)
        
        # Calculate volatility
        volatility = self.get_cached_rolling_result(returns, 'std', period)
        
        # Interaction as product
        return rsi * volatility
    
    def _calculate_macd_volume_interaction(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate MACD-Volume interaction."""
        if 'close' not in data.columns or 'volume' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        volume = data['volume']
        
        # Calculate MACD
        macd = self._calculate_macd(close, 12, 26, 9)
        
        # Calculate volume ratio
        volume_ratio = volume / self.get_cached_rolling_result(volume, 'mean', period)
        
        # Interaction as product
        return macd * volume_ratio
    
    def _calculate_price_volatility_interaction(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Price-Volatility interaction."""
        if 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        returns = close.pct_change()
        
        # Calculate price momentum
        price_momentum = (close - close.shift(periods=period)) / close.shift(periods=period)
        
        # Calculate volatility
        volatility = self.get_cached_rolling_result(returns, 'std', period)
        
        # Interaction as product
        return price_momentum * volatility
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gains = self.get_cached_rolling_result(gains, 'mean', period)
        avg_losses = self.get_cached_rolling_result(losses, 'mean', period)
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD."""
        ema_fast = self.get_cached_rolling_result(prices, 'ewm', fast, span=fast)
        ema_slow = self.get_cached_rolling_result(prices, 'ewm', slow, span=slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.get_cached_rolling_result(macd_line, 'ewm', signal, span=signal)
        
        return macd_line - signal_line
    
    def _generate_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate regime-based features."""
        feature_configs = []
        
        # Market regime
        feature_configs.append({
            'name': 'market_regime',
            'type': 'custom',
            'params': {
                'function': self._calculate_market_regime,
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
        
        # Trend regime
        feature_configs.append({
            'name': 'trend_regime',
            'type': 'custom',
            'params': {
                'function': self._calculate_trend_regime,
                'period': 20
            }
        })
        
        return self.generate_cross_category_features(data, feature_configs)
    
    def _calculate_market_regime(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate market regime (bull/bear/sideways)."""
        if 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        returns = close.pct_change()
        
        # Calculate rolling mean and std
        mean_return = self.get_cached_rolling_result(returns, 'mean', period)
        std_return = self.get_cached_rolling_result(returns, 'std', period)
        
        # Define regime based on mean return and volatility
        regime = np.where(mean_return > std_return, 1,  # Bull market
                         np.where(mean_return < -std_return, -1,  # Bear market
                                 0))  # Sideways market
        
        return pd.Series(regime, index=data.index)
    
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
    
    def _calculate_trend_regime(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate trend regime (uptrend/downtrend/sideways)."""
        if 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        
        # Calculate moving averages
        sma_short = self.get_cached_rolling_result(close, 'mean', period // 2)
        sma_long = self.get_cached_rolling_result(close, 'mean', period)
        
        # Define trend based on moving average relationship
        trend = np.where(sma_short > sma_long, 1,  # Uptrend
                        np.where(sma_short < sma_long, -1,  # Downtrend
                                0))  # Sideways
        
        return pd.Series(trend, index=data.index)
    
    def generate_all_cross_category_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate all cross-category features with comprehensive optimization."""
        with self.performance_monitoring("comprehensive_cross_category_features"):
            # Optimize DataFrame for processing
            optimized_data = self.optimize_dataframe_processing(data)
            
            # Generate all cross-category features
            features = self.generate_cross_category_features_optimized(optimized_data)
            
            # Log performance statistics
            stats = self.get_performance_stats()
            self.logger.info(f"Generated {len(features.columns)} cross-category features")
            self.logger.info(f"VectorBT usage rate: {stats.get('vectorbt_usage_rate', 0):.2%}")
            self.logger.info(f"Cache hit rate: {stats.get('cache_hit_rate', 0):.2f}%")
            self.logger.info(f"Memory optimizations: {stats.get('memory_optimizations', 0)}")
            self.logger.info(f"Enabled categories: {self.enabled_categories}")
            
            return features


# Convenience functions for easy usage
def create_optimized_cross_category_generator(
    enabled_categories: List[str] = None,
    periods: List[int] = None,
    cross_correlations: bool = True,
    interaction_features: bool = True,
    regime_features: bool = True
) -> OptimizedCrossCategoryFeatureGenerator:
    """Create an optimized cross-category feature generator with specified parameters."""
    config = {
        'enabled_categories': enabled_categories or ['trend', 'volatility', 'returns', 'momentum', 'volume'],
        'periods': periods or [5, 10, 20, 50, 100],
        'cross_correlations': cross_correlations,
        'interaction_features': interaction_features,
        'regime_features': regime_features
    }
    return OptimizedCrossCategoryFeatureGenerator(config)


def generate_cross_category_features_optimized(
    data: pd.DataFrame,
    enabled_categories: List[str] = None,
    periods: List[int] = None
) -> pd.DataFrame:
    """Generate optimized cross-category features for the given data."""
    generator = create_optimized_cross_category_generator(enabled_categories, periods)
    return generator.generate_all_cross_category_features(data)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='1min')
    data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(1000) * 0.01),
        'open': 100 + np.cumsum(np.random.randn(1000) * 0.01),
        'high': 100 + np.cumsum(np.random.randn(1000) * 0.01) + np.abs(np.random.randn(1000) * 0.5),
        'low': 100 + np.cumsum(np.random.randn(1000) * 0.01) - np.abs(np.random.randn(1000) * 0.5),
        'volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)
    
    print("Testing Optimized Cross-Category Feature Generator...")
    
    # Create generator
    generator = create_optimized_cross_category_generator(
        enabled_categories=['trend', 'volatility', 'returns'],
        periods=[10, 20, 50]
    )
    
    # Generate features
    features = generator.generate_all_cross_category_features(data)
    
    print(f"Generated {len(features.columns)} cross-category features")
    print(f"Feature columns: {list(features.columns)}")
    
    # Get performance stats
    stats = generator.get_performance_stats()
    print(f"Performance stats: {stats}")
    
    print("✅ Optimized cross-category feature generation test completed!")