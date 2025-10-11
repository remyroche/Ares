"""
Consolidated Feature Generators

This module provides a unified approach to feature generation that leverages
the centralized utilities from feature_generation/ and features_common/
to eliminate code duplication and ensure consistent VectorBT optimization.

Key Features:
- Uses VectorBTRollingOptimizer for all rolling operations
- Leverages VectorBTScaler for normalization
- Implements feature generators through the FeatureBank system
- Maintains backward compatibility while reducing duplication
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod

# Centralized imports from feature_generation/
from src.feature_generation.core.vectorbt_feature_generator import VectorBTFeatureGenerator
from src.feature_generation.core.feature_generator import FeatureConfig, FeatureCategory
from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
from src.feature_generation.core.feature_bank import get_global_feature_bank

# Centralized imports from features_common/
from src.features_common.transforms.vectorbt_scaler import VectorBTScaler, create_vectorbt_scaler

logger = logging.getLogger(__name__)


class ConsolidatedFeatureGenerator(VectorBTFeatureGenerator):
    """
    Base class for consolidated feature generators that use centralized utilities.
    
    This class ensures all feature generators use:
    - VectorBTRollingOptimizer for rolling operations
    - VectorBTScaler for normalization
    - Consistent error handling and fallbacks
    """
    
    def __init__(self, config: FeatureConfig, enable_gpu: bool = False, enable_parallel: bool = True):
        super().__init__(config, enable_gpu, enable_parallel)
        
        # Initialize centralized utilities
        self.rolling_optimizer = get_vectorbt_rolling_optimizer(
            enable_gpu=enable_gpu, 
            enable_parallel=enable_parallel
        )
        
        # Initialize scaler for normalization
        self.scaler = create_vectorbt_scaler(method='zscore')
        
        # Performance tracking
        self.consolidated_stats = {
            'rolling_operations': 0,
            'scaling_operations': 0,
            'vectorbt_optimizations': 0,
            'fallback_operations': 0
        }
    
    def _optimized_rolling_operation(self, data: pd.Series, operation: str, 
                                   window: int, **kwargs) -> pd.Series:
        """
        Perform rolling operation using centralized VectorBTRollingOptimizer.
        
        Args:
            data: Input data series
            operation: Operation type ('mean', 'std', 'var', 'min', 'max', 'sum')
            window: Rolling window size
            **kwargs: Additional parameters
            
        Returns:
            Result of rolling operation
        """
        try:
            if operation == 'mean':
                result = self.rolling_optimizer.rolling_mean(data, window, **kwargs)
            elif operation == 'std':
                result = self.rolling_optimizer.rolling_std(data, window, **kwargs)
            elif operation == 'var':
                result = self.rolling_optimizer.rolling_var(data, window, **kwargs)
            elif operation == 'min':
                result = self.rolling_optimizer.rolling_min(data, window, **kwargs)
            elif operation == 'max':
                result = self.rolling_optimizer.rolling_max(data, window, **kwargs)
            elif operation == 'sum':
                result = self.rolling_optimizer.rolling_sum(data, window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
            
            self.consolidated_stats['rolling_operations'] += 1
            self.consolidated_stats['vectorbt_optimizations'] += 1
            return result
            
        except Exception as e:
            logger.warning(f"VectorBT rolling operation failed: {e}, using fallback")
            self.consolidated_stats['fallback_operations'] += 1
            return self._fallback_rolling_operation(data, operation, window, **kwargs)
    
    def _fallback_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        rolling_obj = data.rolling(window=window, **kwargs)
        
        if operation == 'mean':
            return rolling_obj.mean()
        elif operation == 'std':
            return rolling_obj.std()
        elif operation == 'var':
            return rolling_obj.var()
        elif operation == 'min':
            return rolling_obj.min()
        elif operation == 'max':
            return rolling_obj.max()
        elif operation == 'sum':
            return rolling_obj.sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _normalize_feature(self, data: pd.Series, method: str = 'zscore') -> pd.Series:
        """
        Normalize feature using centralized VectorBTScaler.
        
        Args:
            data: Input data series
            method: Normalization method
            
        Returns:
            Normalized data series
        """
        try:
            # Create scaler with specified method
            scaler = create_vectorbt_scaler(method=method)
            result = scaler.fit_transform(data)
            
            self.consolidated_stats['scaling_operations'] += 1
            return result
            
        except Exception as e:
            logger.warning(f"VectorBT scaling failed: {e}, using fallback")
            return self._fallback_normalize(data, method)
    
    def _fallback_normalize(self, data: pd.Series, method: str = 'zscore') -> pd.Series:
        """Fallback normalization using pandas/numpy."""
        if method == 'zscore':
            return (data - data.mean()) / data.std()
        elif method == 'minmax':
            return (data - data.min()) / (data.max() - data.min())
        elif method == 'robust':
            median = data.median()
            mad = (data - median).abs().median()
            return (data - median) / mad
        else:
            return data


class ConsolidatedRSIGenerator(ConsolidatedFeatureGenerator):
    """
    Consolidated RSI generator that uses centralized utilities.
    
    This replaces multiple RSI implementations across the codebase.
    """
    
    def __init__(self, period: int = 14, enable_gpu: bool = False, enable_parallel: bool = True):
        config = FeatureConfig(
            name=f"consolidated_rsi_{period}",
            category=FeatureCategory.MOMENTUM,
            description=f"Consolidated RSI over {period} periods using centralized utilities",
            required_columns=["close"],
            default_lookback=period * 2,
            min_lookback=period,
            max_lookback=period * 3,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=enable_gpu
        )
        super().__init__(config, enable_gpu, enable_parallel)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate RSI using centralized rolling operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'consolidated_rsi_{self.period}')
        
        close = data['close']
        
        # Calculate price changes
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Use centralized rolling operations
        avg_gain = self._optimized_rolling_operation(gain, 'mean', self.period)
        avg_loss = self._optimized_rolling_operation(loss, 'mean', self.period)
        
        # Calculate RSI
        rs = avg_gain / avg_loss.replace(0, 1)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.rename(f'consolidated_rsi_{self.period}')


class ConsolidatedMACDGenerator(ConsolidatedFeatureGenerator):
    """
    Consolidated MACD generator that uses centralized utilities.
    
    This replaces multiple MACD implementations across the codebase.
    """
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, 
                 enable_gpu: bool = False, enable_parallel: bool = True):
        config = FeatureConfig(
            name=f"consolidated_macd_{fast}_{slow}_{signal}",
            category=FeatureCategory.MOMENTUM,
            description=f"Consolidated MACD with fast={fast}, slow={slow}, signal={signal}",
            required_columns=["close"],
            default_lookback=slow + signal,
            min_lookback=slow,
            max_lookback=slow + signal,
            parameters={"fast": fast, "slow": slow, "signal": signal},
            matrix_optimized=True,
            gpu_accelerated=enable_gpu
        )
        super().__init__(config, enable_gpu, enable_parallel)
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate MACD using centralized utilities."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'consolidated_macd_{self.fast}_{self.slow}')
        
        close = data['close']
        
        # Calculate EMAs using centralized rolling operations
        # Note: For EMA, we use the ewm method as it's more appropriate than rolling mean
        ema_fast = close.ewm(span=self.fast).mean()
        ema_slow = close.ewm(span=self.slow).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=self.signal).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return histogram.rename(f'consolidated_macd_{self.fast}_{self.slow}')


class ConsolidatedEMAGenerator(ConsolidatedFeatureGenerator):
    """
    Consolidated EMA generator that uses centralized utilities.
    
    This replaces multiple EMA implementations across the codebase.
    """
    
    def __init__(self, period: int = 20, enable_gpu: bool = False, enable_parallel: bool = True):
        config = FeatureConfig(
            name=f"consolidated_ema_{period}",
            category=FeatureCategory.TREND,
            description=f"Consolidated EMA over {period} periods using centralized utilities",
            required_columns=["close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=enable_gpu
        )
        super().__init__(config, enable_gpu, enable_parallel)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate EMA using centralized utilities."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'consolidated_ema_{self.period}')
        
        close = data['close']
        
        # Calculate EMA using pandas ewm (most appropriate for EMA)
        ema = close.ewm(span=self.period).mean()
        
        return ema.rename(f'consolidated_ema_{self.period}')


class ConsolidatedSMAGenerator(ConsolidatedFeatureGenerator):
    """
    Consolidated SMA generator that uses centralized utilities.
    
    This replaces multiple SMA implementations across the codebase.
    """
    
    def __init__(self, period: int = 20, enable_gpu: bool = False, enable_parallel: bool = True):
        config = FeatureConfig(
            name=f"consolidated_sma_{period}",
            category=FeatureCategory.TREND,
            description=f"Consolidated SMA over {period} periods using centralized utilities",
            required_columns=["close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=enable_gpu
        )
        super().__init__(config, enable_gpu, enable_parallel)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate SMA using centralized rolling operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'consolidated_sma_{self.period}')
        
        close = data['close']
        
        # Use centralized rolling operations for SMA
        sma = self._optimized_rolling_operation(close, 'mean', self.period)
        
        return sma.rename(f'consolidated_sma_{self.period}')


def create_consolidated_generators() -> List[ConsolidatedFeatureGenerator]:
    """
    Create a comprehensive set of consolidated feature generators.
    
    This function replaces the need for multiple generator creation functions
    across different modules.
    
    Returns:
        List of consolidated feature generators
    """
    generators = []
    
    # RSI generators with different periods
    for period in [9, 14, 21, 30]:
        generators.append(ConsolidatedRSIGenerator(period))
    
    # MACD generators with different parameters
    macd_configs = [
        (8, 21, 5), (12, 26, 9), (5, 35, 5)
    ]
    for fast, slow, signal in macd_configs:
        generators.append(ConsolidatedMACDGenerator(fast, slow, signal))
    
    # EMA generators with different periods
    for period in [10, 20, 50, 100]:
        generators.append(ConsolidatedEMAGenerator(period))
    
    # SMA generators with different periods
    for period in [10, 20, 50, 100]:
        generators.append(ConsolidatedSMAGenerator(period))
    
    return generators


def register_consolidated_generators():
    """
    Register consolidated generators with the global feature bank.
    
    This ensures all consolidated generators are available through
    the centralized feature bank system.
    """
    feature_bank = get_global_feature_bank()
    generators = create_consolidated_generators()
    
    for generator in generators:
        feature_bank.register_generator(generator)
    
    logger.info(f"Registered {len(generators)} consolidated generators with feature bank")
    return generators


# Example usage and migration guide
def migrate_to_consolidated_generators():
    """
    Migration guide for replacing existing generators with consolidated ones.
    
    This function demonstrates how to replace existing feature generators
    with the consolidated versions that use centralized utilities.
    """
    # Register consolidated generators
    generators = register_consolidated_generators()
    
    # Example: Generate features using consolidated generators
    feature_bank = get_global_feature_bank()
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=1000, freq='1min')
    np.random.seed(42)
    data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(1000) * 0.01),
        'high': 100 + np.cumsum(np.random.randn(1000) * 0.01) + np.random.randn(1000) * 0.5,
        'low': 100 + np.cumsum(np.random.randn(1000) * 0.01) - np.random.randn(1000) * 0.5,
        'volume': np.random.lognormal(10, 1, 1000)
    }, index=dates)
    
    # Generate features using consolidated generators
    features = feature_bank.generate_features(
        data, 
        categories=[FeatureCategory.MOMENTUM, FeatureCategory.TREND]
    )
    
    logger.info(f"Generated {len(features.columns)} features using consolidated generators")
    return features


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Register and test consolidated generators
    generators = register_consolidated_generators()
    features = migrate_to_consolidated_generators()
    
    print(f"Successfully created {len(generators)} consolidated generators")
    print(f"Generated features shape: {features.shape}")
    print(f"Feature columns: {list(features.columns)}")