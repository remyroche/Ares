"""
Custom Support/Resistance Feature Generators

This module provides custom support/resistance feature generators for calculating
derived SR metrics like strength, distance, touch counts, and other analytics.

These features are disabled by default and complement the pre-created SR levels
in the main support_resistance.py module.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
import logging
import warnings

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

# Optimization utilities
try:
    from src.feature_generation.utils.vectorization_optimizer import get_vectorization_optimizer
    from src.feature_generation.utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

# VectorBT optimization imports
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
    from src.feature_generation.utils.vectorbt_optimization_integration import get_optimization_manager, VectorBTOptimizationManager
    VECTORBT_OPTIMIZATION_AVAILABLE = True
except ImportError:
    VECTORBT_OPTIMIZATION_AVAILABLE = False
    VectorBTRollingOptimizer = None
    VectorBTOptimizationManager = None

# Fallback VectorBT imports for direct operations
try:
    import vectorbt as vbt
    from src.utils.vectorbt_compat import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None

logger = logging.getLogger(__name__)


class SRStrengthGenerator(VectorizedFeatureGenerator):
    """Generator for SR strength/quality metrics."""

    def __init__(self, window: int = 20, enable_gpu: bool = False, memory_efficient: bool = True):
        config = FeatureConfig(
            name=f"sr_strength_{window}",
            category=FeatureCategory.CUSTOM_SUPPORT_RESISTANCE,
            description=f"Support/Resistance strength over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        self.window = window
        self.enable_gpu = enable_gpu
        self.memory_efficient = memory_efficient
        
        # Initialize rolling optimizer
        if VECTORBT_OPTIMIZATION_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=enable_gpu,
                enable_parallel=True,
                memory_efficient=memory_efficient
            )
        else:
            self.rolling_optimizer = None

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate SR strength feature."""
        if 'close' not in data.columns:
            return pd.Series(np.zeros(len(data)), index=data.index)
        
        close = data['close']
        
        if self.rolling_optimizer:
            rolling_std = self.rolling_optimizer.rolling_std(close, self.window)
            rolling_mean = self.rolling_optimizer.rolling_mean(close, self.window)
            # Strength based on volatility and mean reversion
            strength = 1 / (1 + rolling_std / rolling_mean)
        else:
            rolling_std = close.rolling(window=self.window).std()
            rolling_mean = close.rolling(window=self.window).mean()
            strength = 1 / (1 + rolling_std / rolling_mean)
        
        return strength


class SRDistanceGenerator(VectorizedFeatureGenerator):
    """Generator for distance to nearest SR level."""

    def __init__(self, window: int = 20, enable_gpu: bool = False, memory_efficient: bool = True):
        config = FeatureConfig(
            name=f"sr_distance_{window}",
            category=FeatureCategory.CUSTOM_SUPPORT_RESISTANCE,
            description=f"Distance to nearest SR level over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        self.window = window
        self.enable_gpu = enable_gpu
        self.memory_efficient = memory_efficient
        
        # Initialize rolling optimizer
        if VECTORBT_OPTIMIZATION_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=enable_gpu,
                enable_parallel=True,
                memory_efficient=memory_efficient
            )
        else:
            self.rolling_optimizer = None

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate SR distance feature."""
        if 'close' not in data.columns:
            return pd.Series(np.zeros(len(data)), index=data.index)
        
        close = data['close']
        
        if self.rolling_optimizer:
            rolling_min = self.rolling_optimizer.rolling_min(close, self.window)
            rolling_max = self.rolling_optimizer.rolling_max(close, self.window)
        else:
            rolling_min = close.rolling(window=self.window).min()
            rolling_max = close.rolling(window=self.window).max()
        
        # Distance to nearest SR level
        distance_to_support = close - rolling_min
        distance_to_resistance = rolling_max - close
        distance = np.minimum(distance_to_support, distance_to_resistance)
        
        return distance


class SRTouchCountGenerator(VectorizedFeatureGenerator):
    """Generator for SR level touch count analysis."""

    def __init__(self, window: int = 50, threshold_pct: float = 0.01, 
                 enable_gpu: bool = False, memory_efficient: bool = True):
        config = FeatureConfig(
            name=f"sr_touch_count_{window}",
            category=FeatureCategory.CUSTOM_SUPPORT_RESISTANCE,
            description=f"Count of touches to SR levels over {window} periods",
            required_columns=["high", "low", "close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'threshold_pct': threshold_pct}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        self.window = window
        self.threshold_pct = threshold_pct
        self.enable_gpu = enable_gpu
        self.memory_efficient = memory_efficient
        
        # Initialize rolling optimizer
        if VECTORBT_OPTIMIZATION_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=enable_gpu,
                enable_parallel=True,
                memory_efficient=memory_efficient
            )
        else:
            self.rolling_optimizer = None

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate SR touch count feature."""
        required_cols = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            return pd.Series(np.zeros(len(data)), index=data.index)
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate SR levels
        if self.rolling_optimizer:
            resistance = self.rolling_optimizer.rolling_max(high, self.window)
            support = self.rolling_optimizer.rolling_min(low, self.window)
        else:
            resistance = high.rolling(window=self.window).max()
            support = low.rolling(window=self.window).min()
        
        # Count touches (simplified version - can be enhanced)
        touch_count = pd.Series(0.0, index=data.index)
        
        for i in range(self.window, len(data)):
            window_data = close.iloc[max(0, i-self.window):i]
            current_resistance = resistance.iloc[i]
            current_support = support.iloc[i]
            
            # Count touches to resistance
            resistance_touches = np.sum(
                np.abs(window_data - current_resistance) / current_resistance <= self.threshold_pct
            )
            
            # Count touches to support
            support_touches = np.sum(
                np.abs(window_data - current_support) / current_support <= self.threshold_pct
            )
            
            touch_count.iloc[i] = resistance_touches + support_touches
        
        return touch_count


class SRQualityGenerator(VectorizedFeatureGenerator):
    """Generator for SR quality/consistency metrics."""

    def __init__(self, window: int = 20, enable_gpu: bool = False, memory_efficient: bool = True):
        config = FeatureConfig(
            name=f"sr_quality_{window}",
            category=FeatureCategory.CUSTOM_SUPPORT_RESISTANCE,
            description=f"Support/Resistance quality score over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        self.window = window
        self.enable_gpu = enable_gpu
        self.memory_efficient = memory_efficient
        
        # Initialize rolling optimizer
        if VECTORBT_OPTIMIZATION_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=enable_gpu,
                enable_parallel=True,
                memory_efficient=memory_efficient
            )
        else:
            self.rolling_optimizer = None

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate SR quality feature."""
        if 'close' not in data.columns:
            return pd.Series(np.zeros(len(data)), index=data.index)
        
        close = data['close']
        
        if self.rolling_optimizer:
            rolling_min = self.rolling_optimizer.rolling_min(close, self.window)
            rolling_max = self.rolling_optimizer.rolling_max(close, self.window)
            rolling_std = self.rolling_optimizer.rolling_std(close, self.window)
        else:
            rolling_min = close.rolling(window=self.window).min()
            rolling_max = close.rolling(window=self.window).max()
            rolling_std = close.rolling(window=self.window).std()
        
        # Quality based on consistency and strength
        range_size = rolling_max - rolling_min
        # Avoid division by zero
        quality = 1 / (1 + rolling_std / (range_size + 1e-10))
        
        return quality


class VolumeWeightedSRGenerator(VectorizedFeatureGenerator):
    """Generator for volume-weighted SR levels."""

    def __init__(self, window: int = 20, enable_gpu: bool = False, memory_efficient: bool = True):
        config = FeatureConfig(
            name=f"sr_volume_weighted_{window}",
            category=FeatureCategory.CUSTOM_SUPPORT_RESISTANCE,
            description=f"Volume-weighted SR level over {window} periods",
            required_columns=["close", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        self.window = window
        self.enable_gpu = enable_gpu
        self.memory_efficient = memory_efficient
        
        # Initialize rolling optimizer
        if VECTORBT_OPTIMIZATION_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=enable_gpu,
                enable_parallel=True,
                memory_efficient=memory_efficient
            )
        else:
            self.rolling_optimizer = None

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate volume-weighted SR feature."""
        required_cols = ['close', 'volume']
        if not all(col in data.columns for col in required_cols):
            return pd.Series(np.zeros(len(data)), index=data.index)
        
        close = data['close']
        volume = data['volume']
        
        if self.rolling_optimizer:
            volume_weighted_price = (
                self.rolling_optimizer.rolling_sum(close * volume, self.window) /
                self.rolling_optimizer.rolling_sum(volume, self.window)
            )
        else:
            volume_weighted_price = (
                (close * volume).rolling(window=self.window).sum() / 
                volume.rolling(window=self.window).sum()
            )
        
        return volume_weighted_price


class DynamicSRGenerator(VectorizedFeatureGenerator):
    """Generator for dynamic/adaptive SR levels."""

    def __init__(self, base_window: int = 20, enable_gpu: bool = False, memory_efficient: bool = True):
        config = FeatureConfig(
            name=f"sr_dynamic_{base_window}",
            category=FeatureCategory.CUSTOM_SUPPORT_RESISTANCE,
            description=f"Dynamic/adaptive SR level with base window {base_window}",
            required_columns=["close"],
            default_lookback=50,  # Need extra for volatility calculation
            min_lookback=base_window,
            max_lookback=200,
            parameters={'base_window': base_window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        self.base_window = base_window
        self.enable_gpu = enable_gpu
        self.memory_efficient = memory_efficient
        
        # Initialize rolling optimizer
        if VECTORBT_OPTIMIZATION_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=enable_gpu,
                enable_parallel=True,
                memory_efficient=memory_efficient
            )
        else:
            self.rolling_optimizer = None

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate dynamic SR feature."""
        if 'close' not in data.columns:
            return pd.Series(np.zeros(len(data)), index=data.index)
        
        close = data['close']
        
        if not self.rolling_optimizer:
            return pd.Series(np.zeros(len(data)), index=data.index)
        
        # Calculate volatility-based adaptive window
        volatility = self.rolling_optimizer.rolling_std(close, window=self.base_window)
        vol_mean = volatility.rolling(50).mean()
        
        # Avoid division by zero
        adaptive_factor = (volatility / (vol_mean + 1e-10) * 10).fillna(0)
        adaptive_window = (self.base_window + adaptive_factor).astype(int)
        adaptive_window = adaptive_window.clip(5, 50)
        
        # Dynamic SR calculation (simplified version)
        dynamic_sr = pd.Series(0.0, index=close.index)
        for i in range(len(close)):
            window = int(adaptive_window.iloc[i])
            if i >= window - 1:
                start_idx = max(0, i - window + 1)
                window_data = close.iloc[start_idx:i+1]
                dynamic_sr.iloc[i] = (window_data.min() + window_data.max() + window_data.mean()) / 3
        
        return dynamic_sr


def create_default_custom_sr_generators(enable_gpu: bool = False, memory_efficient: bool = True) -> List[FeatureGenerator]:
    """
    Create default custom SR feature generators.
    
    Note: These generators are DISABLED by default and must be explicitly enabled.
    """
    windows = [10, 20, 50]
    generators = []
    
    for window in windows:
        generators.extend([
            SRStrengthGenerator(window, enable_gpu=enable_gpu, memory_efficient=memory_efficient),
            SRDistanceGenerator(window, enable_gpu=enable_gpu, memory_efficient=memory_efficient),
            SRQualityGenerator(window, enable_gpu=enable_gpu, memory_efficient=memory_efficient),
            VolumeWeightedSRGenerator(window, enable_gpu=enable_gpu, memory_efficient=memory_efficient),
        ])
    
    # Add dynamic SR with different base windows
    for window in [10, 20]:
        generators.append(DynamicSRGenerator(window, enable_gpu=enable_gpu, memory_efficient=memory_efficient))
    
    # Add touch count generators with longer windows
    for window in [50, 100]:
        generators.append(SRTouchCountGenerator(window, enable_gpu=enable_gpu, memory_efficient=memory_efficient))
    
    return generators


# Export all generators
__all__ = [
    'SRStrengthGenerator',
    'SRDistanceGenerator',
    'SRTouchCountGenerator',
    'SRQualityGenerator',
    'VolumeWeightedSRGenerator',
    'DynamicSRGenerator',
    'create_default_custom_sr_generators',
]

