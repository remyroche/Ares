"""Legacy features

Legacy features are traditional technical indicators that have been used in 
financial analysis for decades. These include classic indicators like:
- Traditional RSI implementations
- Classic MACD calculations
- Original Bollinger Bands formulations
- Standard moving averages
- Conventional oscillators

These features maintain backward compatibility with existing trading systems
and provide a baseline for comparison with newer, enhanced indicators.

All legacy generators now use the UnifiedVectorizationManager for optimal performance
and consistent optimization across all features.
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from ..core.feature_generator import VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

# Unified optimization imports
from ..utils.unified_vectorization_manager import (
    get_unified_vectorization_manager, 
    UnifiedVectorizationManager,
    OptimizationConfig
)

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    import warnings
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

class LegacyFeatureGeneratorBase(VectorizedFeatureGenerator):
    """
    Base class for all legacy feature generators with unified VectorBT optimization.
    
    This class provides a consistent interface for all legacy features using the
    UnifiedVectorizationManager for optimal performance and memory management.
    """
    
    def __init__(self, config: FeatureConfig, enable_gpu: bool = False, enable_parallel: bool = True):
        """Initialize legacy feature generator with unified optimization."""
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        # Initialize unified vectorization manager
        optimization_config = OptimizationConfig(
            enable_vectorbt=True,
            enable_gpu=enable_gpu,
            enable_parallel=enable_parallel,
            enable_batch_processing=True,
            enable_caching=True
        )
        self.unified_manager = get_unified_vectorization_manager(optimization_config)
        
        # Performance tracking
        self.performance_stats = {
            'total_generations': 0,
            'vectorbt_operations': 0,
            'batch_operations': 0,
            'cache_hits': 0,
            'total_time': 0.0
        }
    
    def _optimize_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame using unified manager."""
        return self.unified_manager.optimize_dataframe(data)
    
    def _rolling_operation(self, data: pd.Series, operation: str, window: int, **kwargs) -> pd.Series:
        """Perform rolling operation using unified manager."""
        return self.unified_manager.rolling_operation(data, operation, window, **kwargs)
    
    def _technical_indicator(self, data: pd.DataFrame, indicator: str, **kwargs) -> pd.Series:
        """Calculate technical indicator using unified manager."""
        return self.unified_manager.technical_indicator(data, indicator, **kwargs)
    
    def _batch_operations(self, data: pd.DataFrame, operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """Perform batch operations using unified manager."""
        return self.unified_manager.batch_operations(data, operations)
    
    def generate_features_batch(self, data: pd.DataFrame, 
                              feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Generate multiple features in batch with unified optimization."""
        with self.unified_manager.batch_processing():
            return self._batch_operations(data, feature_configs)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        unified_stats = self.unified_manager.get_performance_stats()
        stats.update(unified_stats)
        return stats
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_generations': 0,
            'vectorbt_operations': 0,
            'batch_operations': 0,
            'cache_hits': 0,
            'total_time': 0.0
        }
        self.unified_manager.reset_stats()


class LegacyRSIGenerator(LegacyFeatureGeneratorBase):
    """Legacy RSI generator with unified VectorBT optimization."""
    
    def __init__(self, period: int = 14, enable_gpu: bool = False, enable_parallel: bool = True):
        config = FeatureConfig(
            name=f"legacy_rsi_{period}",
            category=FeatureCategory.LEGACY,
            description=f"Legacy RSI {period} - traditional implementation with VectorBT optimization",
            required_columns=["close"],
            default_lookback=period * 2,
            min_lookback=period,
            max_lookback=period * 3
        )
        super().__init__(config, enable_gpu=enable_gpu, enable_parallel=enable_parallel)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate RSI feature using unified VectorBT optimization."""
        # Optimize DataFrame for processing
        data = self._optimize_dataframe(data)
        
        # Use VectorBT native RSI calculation
        rsi = self._technical_indicator(data, 'rsi', window=self.period)
        return rsi.rename(f'legacy_rsi_{self.period}')

class LegacyMACDGenerator(LegacyFeatureGeneratorBase):
    """Legacy MACD generator with unified VectorBT optimization."""
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, enable_gpu: bool = False, enable_parallel: bool = True):
        config = FeatureConfig(
            name=f"legacy_macd_{fast}_{slow}_{signal}",
            category=FeatureCategory.LEGACY,
            description=f"Legacy MACD {fast}/{slow}/{signal} - traditional implementation with VectorBT optimization",
            required_columns=["close"],
            default_lookback=slow * 2,
            min_lookback=slow,
            max_lookback=slow * 3
        )
        super().__init__(config, enable_gpu=enable_gpu, enable_parallel=enable_parallel)
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate MACD feature using unified VectorBT optimization."""
        # Optimize DataFrame for processing
        data = self._optimize_dataframe(data)
        
        # Use VectorBT native MACD calculation
        macd = self._technical_indicator(data, 'macd', 
                                       fast_window=self.fast, 
                                       slow_window=self.slow, 
                                       signal_window=self.signal)
        return macd.rename(f'legacy_macd_{self.fast}_{self.slow}_{self.signal}')


class LegacyBollingerBandsGenerator(LegacyFeatureGeneratorBase):
    """Legacy Bollinger Bands generator with unified VectorBT optimization."""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0, enable_gpu: bool = False, enable_parallel: bool = True):
        config = FeatureConfig(
            name=f"legacy_bollinger_{period}_{std_dev}",
            category=FeatureCategory.LEGACY,
            description=f"Legacy Bollinger Bands {period}/{std_dev} - traditional implementation with VectorBT optimization",
            required_columns=["close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period
        )
        super().__init__(config, enable_gpu=enable_gpu, enable_parallel=enable_parallel)
        self.period = period
        self.std_dev = std_dev
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Bollinger Bands upper band using unified VectorBT optimization."""
        # Optimize DataFrame for processing
        data = self._optimize_dataframe(data)
        
        # Use VectorBT native Bollinger Bands calculation
        bb_upper = self._technical_indicator(data, 'bbands_upper', 
                                           window=self.period, 
                                           alpha=self.std_dev)
        return bb_upper.rename(f'legacy_bollinger_upper_{self.period}_{self.std_dev}')


class LegacySMAGenerator(LegacyFeatureGeneratorBase):
    """Legacy SMA generator with unified VectorBT optimization."""
    
    def __init__(self, period: int = 20, enable_gpu: bool = False, enable_parallel: bool = True):
        config = FeatureConfig(
            name=f"legacy_sma_{period}",
            category=FeatureCategory.LEGACY,
            description=f"Legacy SMA {period} - traditional implementation with VectorBT optimization",
            required_columns=["close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period
        )
        super().__init__(config, enable_gpu=enable_gpu, enable_parallel=enable_parallel)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate SMA feature using unified VectorBT optimization."""
        # Optimize DataFrame for processing
        data = self._optimize_dataframe(data)
        
        # Use VectorBT native SMA calculation
        sma = self._technical_indicator(data, 'sma', window=self.period)
        return sma.rename(f'legacy_sma_{self.period}')


class LegacyEMAGenerator(LegacyFeatureGeneratorBase):
    """Legacy EMA generator with unified VectorBT optimization."""
    
    def __init__(self, period: int = 21, enable_gpu: bool = False, enable_parallel: bool = True):
        config = FeatureConfig(
            name=f"legacy_ema_{period}",
            category=FeatureCategory.LEGACY,
            description=f"Legacy EMA {period} - traditional implementation with VectorBT optimization",
            required_columns=["close"],
            default_lookback=period * 2,
            min_lookback=period,
            max_lookback=period * 3
        )
        super().__init__(config, enable_gpu=enable_gpu, enable_parallel=enable_parallel)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate EMA feature using unified VectorBT optimization."""
        # Optimize DataFrame for processing
        data = self._optimize_dataframe(data)
        
        # Use VectorBT native EMA calculation
        ema = self._technical_indicator(data, 'ema', window=self.period)
        return ema.rename(f'legacy_ema_{self.period}')


class LegacyATRGenerator(LegacyFeatureGeneratorBase):
    """Legacy ATR generator with unified VectorBT optimization."""
    
    def __init__(self, period: int = 14, enable_gpu: bool = False, enable_parallel: bool = True):
        config = FeatureConfig(
            name=f"legacy_atr_{period}",
            category=FeatureCategory.LEGACY,
            description=f"Legacy ATR {period} - traditional implementation with VectorBT optimization",
            required_columns=["high", "low", "close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period
        )
        super().__init__(config, enable_gpu=enable_gpu, enable_parallel=enable_parallel)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate ATR feature using unified VectorBT optimization."""
        # Optimize DataFrame for processing
        data = self._optimize_dataframe(data)
        
        # Use VectorBT native ATR calculation
        atr = self._technical_indicator(data, 'atr', window=self.period)
        return atr.rename(f'legacy_atr_{self.period}')


class LegacyStochasticGenerator(LegacyFeatureGeneratorBase):
    """Legacy Stochastic generator with unified VectorBT optimization."""
    
    def __init__(self, k_period: int = 14, d_period: int = 3, enable_gpu: bool = False, enable_parallel: bool = True):
        config = FeatureConfig(
            name=f"legacy_stochastic_{k_period}_{d_period}",
            category=FeatureCategory.LEGACY,
            description=f"Legacy Stochastic {k_period}/{d_period} - traditional implementation with VectorBT optimization",
            required_columns=["high", "low", "close"],
            default_lookback=k_period,
            min_lookback=k_period,
            max_lookback=k_period
        )
        super().__init__(config, enable_gpu=enable_gpu, enable_parallel=enable_parallel)
        self.k_period = k_period
        self.d_period = d_period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Stochastic %K feature using unified VectorBT optimization."""
        # Optimize DataFrame for processing
        data = self._optimize_dataframe(data)
        
        # Use VectorBT native Stochastic calculation
        stoch_k = self._technical_indicator(data, 'stoch_k', 
                                          k_window=self.k_period, 
                                          d_window=self.d_period)
        return stoch_k.rename(f'legacy_stochastic_k_{self.k_period}_{self.d_period}')


class LegacyWilliamsRGenerator(LegacyFeatureGeneratorBase):
    """Legacy Williams %R generator with unified VectorBT optimization."""
    
    def __init__(self, period: int = 14, enable_gpu: bool = False, enable_parallel: bool = True):
        config = FeatureConfig(
            name=f"legacy_williams_r_{period}",
            category=FeatureCategory.LEGACY,
            description=f"Legacy Williams %R {period} - traditional implementation with VectorBT optimization",
            required_columns=["high", "low", "close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period
        )
        super().__init__(config, enable_gpu=enable_gpu, enable_parallel=enable_parallel)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Williams %R feature using unified VectorBT optimization."""
        # Optimize DataFrame for processing
        data = self._optimize_dataframe(data)
        
        # Use VectorBT native Williams %R calculation
        willr = self._technical_indicator(data, 'willr', window=self.period)
        return willr.rename(f'legacy_williams_r_{self.period}')


class LegacyOBVGenerator(LegacyFeatureGeneratorBase):
    """Legacy OBV generator with unified VectorBT optimization."""
    
    def __init__(self, enable_gpu: bool = False, enable_parallel: bool = True):
        config = FeatureConfig(
            name="legacy_obv",
            category=FeatureCategory.LEGACY,
            description="Legacy OBV - traditional implementation with VectorBT optimization",
            required_columns=["close", "volume"],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1
        )
        super().__init__(config, enable_gpu=enable_gpu, enable_parallel=enable_parallel)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate OBV feature using unified VectorBT optimization."""
        # Optimize DataFrame for processing
        data = self._optimize_dataframe(data)
        
        # Use VectorBT native OBV calculation
        obv = self._technical_indicator(data, 'obv')
        return obv.rename('legacy_obv')

def create_default_legacy_generators(enable_gpu: bool = False, enable_parallel: bool = True) -> List[LegacyFeatureGeneratorBase]:
    """
    Create default legacy feature generators with unified VectorBT optimization.
    
    Legacy features include traditional implementations of classic indicators
    that have been used in technical analysis for decades. These provide
    backward compatibility and serve as benchmarks for enhanced versions.
    
    Args:
        enable_gpu: Enable GPU acceleration
        enable_parallel: Enable parallel processing
        
    Returns:
        List of optimized legacy feature generators
    """
    generators = []
    
    # Classic indicators with standard parameters
    generators.extend([
        LegacyRSIGenerator(14, enable_gpu=enable_gpu, enable_parallel=enable_parallel),
        LegacyMACDGenerator(12, 26, 9, enable_gpu=enable_gpu, enable_parallel=enable_parallel),
        LegacyBollingerBandsGenerator(20, 2.0, enable_gpu=enable_gpu, enable_parallel=enable_parallel),
        LegacySMAGenerator(20, enable_gpu=enable_gpu, enable_parallel=enable_parallel),
        LegacyEMAGenerator(21, enable_gpu=enable_gpu, enable_parallel=enable_parallel),
        LegacyATRGenerator(14, enable_gpu=enable_gpu, enable_parallel=enable_parallel),
        LegacyStochasticGenerator(14, 3, enable_gpu=enable_gpu, enable_parallel=enable_parallel),
        LegacyWilliamsRGenerator(14, enable_gpu=enable_gpu, enable_parallel=enable_parallel),
        LegacyOBVGenerator(enable_gpu=enable_gpu, enable_parallel=enable_parallel),
    ])
    
    # Additional legacy moving averages
    sma_periods = [5, 10, 50, 100, 200]
    for period in sma_periods:
        generators.append(LegacySMAGenerator(period, enable_gpu=enable_gpu, enable_parallel=enable_parallel))
    
    # Additional legacy EMAs
    ema_periods = [8, 12, 26, 50, 100]
    for period in ema_periods:
        generators.append(LegacyEMAGenerator(period, enable_gpu=enable_gpu, enable_parallel=enable_parallel))
    
    # Additional legacy RSI periods
    rsi_periods = [9, 21, 25]
    for period in rsi_periods:
        generators.append(LegacyRSIGenerator(period, enable_gpu=enable_gpu, enable_parallel=enable_parallel))
    
    return generators


def create_legacy_features_batch(data: pd.DataFrame, 
                                feature_configs: List[Dict[str, Any]], 
                                enable_gpu: bool = False, 
                                enable_parallel: bool = True) -> pd.DataFrame:
    """
    Generate multiple legacy features in batch with unified VectorBT optimization.
    
    Args:
        data: Input OHLCV data
        feature_configs: List of feature configuration dictionaries
        enable_gpu: Enable GPU acceleration
        enable_parallel: Enable parallel processing
        
    Returns:
        DataFrame with generated features
    """
    # Create a temporary generator for batch processing
    temp_generator = LegacyFeatureGeneratorBase(
        FeatureConfig(name="batch_legacy", category=FeatureCategory.LEGACY),
        enable_gpu=enable_gpu,
        enable_parallel=enable_parallel
    )
    
    return temp_generator.generate_features_batch(data, feature_configs)


def get_legacy_performance_stats() -> Dict[str, Any]:
    """Get comprehensive performance statistics for all legacy generators."""
    # Get stats from the unified manager
    manager = get_unified_vectorization_manager()
    return manager.get_performance_stats()


def reset_legacy_performance_stats():
    """Reset performance statistics for all legacy generators."""
    # Reset stats from the unified manager
    manager = get_unified_vectorization_manager()
    manager.reset_stats()
