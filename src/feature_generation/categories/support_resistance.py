"""
Support/Resistance Feature Generator

This module provides feature generators for support/resistance-based indicators.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory
# Optimization utilities
try:
    from src.feature_generation.utils.vectorization_optimizer import get_vectorization_optimizer
    from src.feature_generation.utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

from ..base_calculations import (
    BaseCalculationType,
    create_base_calculator
)

# VectorBT optimization imports
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
    from src.feature_generation.utils.vectorbt_optimization_integration import get_optimization_manager, VectorBTOptimizationManager
    VECTORBT_OPTIMIZATION_AVAILABLE = True
except ImportError:
    VECTORBT_OPTIMIZATION_AVAILABLE = False
    VectorBTRollingOptimizer = None
    VectorBTOptimizationManager = None
    import warnings
    warnings.warn("VectorBT optimization not available. Install with: pip install vectorbt for optimized performance")

# Fallback VectorBT imports for direct operations
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

except ImportError:

    cp = None

class SupportResistanceFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for support/resistance-based features with full VectorBT optimization."""

    def __init__(self, config: Optional[FeatureConfig] = None, enable_gpu: bool = False,
                 memory_efficient: bool = True, enable_monitoring: bool = True):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

        # Initialize VectorBT optimization components
        self.enable_gpu = enable_gpu
        self.memory_efficient = memory_efficient
        self.enable_monitoring = enable_monitoring

        # Initialize optimization manager
        if VECTORBT_OPTIMIZATION_AVAILABLE:
            self.optimization_manager = get_optimization_manager(
                enable_gpu=enable_gpu,
                enable_parallel=True,
                memory_efficient=memory_efficient,
                enable_monitoring=enable_monitoring
            )
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=enable_gpu,
                enable_parallel=True,
                memory_efficient=memory_efficient
            )
        else:
            self.optimization_manager = None
            self.rolling_optimizer = None

    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="support_resistance_features",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description="Comprehensive support/resistance features including pivot points, Fibonacci, and volume profile",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=20,
            min_lookback=1,
            max_lookback=100,
            parameters={
                "pivot_windows": [5, 10, 20],
                "fibonacci_levels": [0.236, 0.382, 0.5, 0.618, 0.786],
                "volume_profile_windows": [5, 10, 20]
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )

    @classmethod
    def create_default(cls) -> 'SupportResistanceFeatureGenerator':
        return cls()

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate comprehensive support/resistance features using VectorBT optimization."""
        # Optimize DataFrame for processing
        if self.optimization_manager:
            data = self.optimization_manager.optimize_dataframe(data)
        elif hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        # Generate comprehensive support/resistance features
        features = {}

        # Basic support/resistance levels
        if 'close' in data.columns:
            close = data['close']
            features['sr_level'] = self._calculate_sr_level(close)
            features['sr_strength'] = self._calculate_sr_strength(close)
            features['sr_distance'] = self._calculate_sr_distance(close)

        # Volume-based support/resistance
        if 'volume' in data.columns and 'close' in data.columns:
            features['volume_sr'] = self._calculate_volume_sr(data['close'], data['volume'])

        # Pivot-based features
        if all(col in data.columns for col in ['high', 'low', 'close']):
            features['pivot_sr'] = self._calculate_pivot_sr(data['high'], data['low'], data['close'])
            features['fibonacci_sr'] = self._calculate_fibonacci_sr(data['high'], data['low'])

        # Combine all features into a single series
        if features:
            # Use the primary SR level as the main feature
            return features.get('sr_level', pd.Series(np.zeros(len(data)), index=data.index))
        else:
            # Fallback to placeholder
            return pd.Series(np.zeros(len(data)), index=data.index, name='sr_placeholder')

    def _calculate_sr_level(self, close: pd.Series) -> pd.Series:
        """Calculate support/resistance level using VectorBT optimization."""
        if self.rolling_optimizer:
            # Use VectorBT rolling operations for efficiency
            window = self.config.parameters.get('pivot_windows', [20])[0]
            rolling_min = self.rolling_optimizer.rolling_min(close, window)
            rolling_max = self.rolling_optimizer.rolling_max(close, window)
            rolling_mean = self.rolling_optimizer.rolling_mean(close, window)

            # Calculate SR level as weighted combination
            sr_level = (rolling_min + rolling_max + rolling_mean) / 3
            return sr_level
        else:
            # Fallback to pandas
            window = self.config.parameters.get('pivot_windows', [20])[0]
            rolling_min = close.rolling(window=window).min()
            rolling_max = close.rolling(window=window).max()
            rolling_mean = close.rolling(window=window).mean()
            return (rolling_min + rolling_max + rolling_mean) / 3

    def _calculate_sr_strength(self, close: pd.Series) -> pd.Series:
        """Calculate support/resistance strength using VectorBT optimization."""
        if self.rolling_optimizer:
            window = self.config.parameters.get('pivot_windows', [20])[0]
            rolling_std = self.rolling_optimizer.rolling_std(close, window)
            rolling_mean = self.rolling_optimizer.rolling_mean(close, window)

            # Strength based on volatility and mean reversion
            strength = 1 / (1 + rolling_std / rolling_mean)
            return strength
        else:
            window = self.config.parameters.get('pivot_windows', [20])[0]
            rolling_std = close.rolling(window=window).std()
            rolling_mean = close.rolling(window=window).mean()
            return 1 / (1 + rolling_std / rolling_mean)

    def _calculate_sr_distance(self, close: pd.Series) -> pd.Series:
        """Calculate distance to nearest support/resistance level."""
        if self.rolling_optimizer:
            window = self.config.parameters.get('pivot_windows', [20])[0]
            rolling_min = self.rolling_optimizer.rolling_min(close, window)
            rolling_max = self.rolling_optimizer.rolling_max(close, window)

            # Distance to nearest SR level
            distance_to_support = close - rolling_min
            distance_to_resistance = rolling_max - close
            distance = np.minimum(distance_to_support, distance_to_resistance)
            return distance
        else:
            window = self.config.parameters.get('pivot_windows', [20])[0]
            rolling_min = close.rolling(window=window).min()
            rolling_max = close.rolling(window=window).max()
            distance_to_support = close - rolling_min
            distance_to_resistance = rolling_max - close
            return np.minimum(distance_to_support, distance_to_resistance)

    def _calculate_volume_sr(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate volume-weighted support/resistance."""
        if self.rolling_optimizer:
            window = self.config.parameters.get('volume_profile_windows', [20])[0]
            # Volume-weighted price
            volume_weighted_price = self.rolling_optimizer.rolling_sum(close * volume, window) / self.rolling_optimizer.rolling_sum(volume, window)
            return volume_weighted_price
        else:
            window = self.config.parameters.get('volume_profile_windows', [20])[0]
            volume_weighted_price = (close * volume).rolling(window=window).sum() / volume.rolling(window=window).sum()
            return volume_weighted_price

    def _calculate_pivot_sr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate pivot-based support/resistance."""
        if self.rolling_optimizer:
            window = self.config.parameters.get('pivot_windows', [20])[0]
            # Pivot point calculation
            pivot = (high + low + close) / 3
            # Rolling pivot for SR level
            rolling_pivot = self.rolling_optimizer.rolling_mean(pivot, window)
            return rolling_pivot
        else:
            window = self.config.parameters.get('pivot_windows', [20])[0]
            pivot = (high + low + close) / 3
            return pivot.rolling(window=window).mean()

    def _calculate_fibonacci_sr(self, high: pd.Series, low: pd.Series) -> pd.Series:
        """Calculate Fibonacci-based support/resistance levels."""
        if self.rolling_optimizer:
            window = self.config.parameters.get('pivot_windows', [20])[0]
            fibonacci_levels = self.config.parameters.get('fibonacci_levels', [0.618])
            level = fibonacci_levels[0]

            rolling_high = self.rolling_optimizer.rolling_max(high, window)
            rolling_low = self.rolling_optimizer.rolling_min(low, window)
            range_size = rolling_high - rolling_low
            fibonacci_level = rolling_low + (range_size * level)
            return fibonacci_level
        else:
            window = self.config.parameters.get('pivot_windows', [20])[0]
            fibonacci_levels = self.config.parameters.get('fibonacci_levels', [0.618])
            level = fibonacci_levels[0]

            rolling_high = high.rolling(window=window).max()
            rolling_low = low.rolling(window=window).min()
            range_size = rolling_high - rolling_low
            return rolling_low + (range_size * level)

# Support Level Generator

    def generate_optimized_support_resistance_features(self, data: pd.DataFrame,
                                                     feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Generate multiple support/resistance features using optimized batch processing.

        Args:
            data: OHLCV data
            feature_configs: List of feature configuration dictionaries

        Returns:
            DataFrame with generated support/resistance features
        """
        # Initialize Unified Vectorization Manager if not already done
        if not hasattr(self, 'unified_manager'):
            try:
                from ...utils.ml_common.unified_vectorization_manager import (
                    get_unified_vectorization_manager, UnifiedVectorizationManager,
                    OperationType, OptimizationStrategy, OperationConfig
                )
                self.unified_manager = get_unified_vectorization_manager()
                self.UNIFIED_MANAGER_AVAILABLE = True
            except ImportError:
                self.unified_manager = None
                self.UNIFIED_MANAGER_AVAILABLE = False

        if hasattr(self, 'unified_manager') and self.unified_manager and len(data) > 100:
            try:
                # Use Unified Vectorization Manager for batch processing
                batch_result = self.unified_manager.optimize_operation(
                    OperationType.FEATURE_ENGINEERING,
                    {
                        'data': data,
                        'feature_configs': feature_configs,
                        'operation_type': 'support_resistance_batch'
                    },
                    OperationConfig(
                        operation_type=OperationType.FEATURE_ENGINEERING,
                        data_size=len(data),
                        data_dimensions=data.shape,
                        memory_budget_mb=1024.0
                    )
                )
                return batch_result.result
            except Exception as e:
                self.logger.warning(f"Unified Vectorization Manager batch processing failed: {e}, using fallback")
                # Fallback to individual processing
                return self._process_support_resistance_features_individually(data, feature_configs)
        else:
            return self._process_support_resistance_features_individually(data, feature_configs)

    def _process_support_resistance_features_individually(self, data: pd.DataFrame, feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process support/resistance features individually as fallback when batch processing fails."""
        results = {}
        for config in feature_configs:
            feature_name = config['name']
            feature_type = config.get('type', 'support_resistance')
            params = config.get('params', {})

            try:
                if feature_type == 'support_resistance':
                    window = params.get('window', 20)
                    if 'close' in data.columns:
                        close = data['close']
                        if self.rolling_optimizer:
                            rolling_min = self.rolling_optimizer.rolling_min(close, window)
                            rolling_max = self.rolling_optimizer.rolling_max(close, window)
                            rolling_mean = self.rolling_optimizer.rolling_mean(close, window)
                            sr_level = (rolling_min + rolling_max + rolling_mean) / 3
                        else:
                            rolling_min = close.rolling(window=window).min()
                            rolling_max = close.rolling(window=window).max()
                            rolling_mean = close.rolling(window=window).mean()
                            sr_level = (rolling_min + rolling_max + rolling_mean) / 3
                        results[feature_name] = sr_level

            except Exception as e:
                self.logger.warning(f"Support/Resistance feature {feature_name} failed: {e}")
                results[feature_name] = pd.Series(np.nan, index=data.index)

        return pd.DataFrame(results, index=data.index)

class SupportLevelGenerator(VectorizedFeatureGenerator):
    """Generator for support level features with VectorBT optimization."""

    def __init__(self, level: int = 1, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 enable_gpu: bool = False, memory_efficient: bool = True, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        if 'low' not in required_columns:
            required_columns.append('low')

        config = FeatureConfig(
            name=f"support_level_{level}_{window}_{base_calculation.value}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"Support level {level} over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'level': level, 'window': window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

        # Initialize VectorBT optimization
        self.level = level
        self.window = window
        self.base_calculation = base_calculation
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
        """Generate support level using VectorBT optimization."""
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            low = data['low']
            # Use VectorBT rolling optimizer for enhanced performance
            if self.rolling_optimizer:
                support_level = self.rolling_optimizer.rolling_min(low, window=self.window)
            elif VECTORBT_AVAILABLE and len(low) >= 1000:
                support_level = rolling_min(low, window=self.window)
            else:
                support_level = low.rolling(window=self.window).min()
        else:
            base_values = self.base_calculator.calculate(data)
            # Use VectorBT rolling optimizer for enhanced performance
            if self.rolling_optimizer:
                support_level = self.rolling_optimizer.rolling_min(base_values, window=self.window)
            elif VECTORBT_AVAILABLE and len(base_values) >= 1000:
                support_level = rolling_min(base_values, window=self.window)
            else:
                support_level = base_values.rolling(window=self.window).min()
        return support_level

# Resistance Level Generator

class ResistanceLevelGenerator(VectorizedFeatureGenerator):
    """Generator for resistance level features with VectorBT optimization."""

    def __init__(self, level: int = 1, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 enable_gpu: bool = False, memory_efficient: bool = True, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        if 'high' not in required_columns:
            required_columns.append('high')

        config = FeatureConfig(
            name=f"resistance_level_{level}_{window}_{base_calculation.value}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"Resistance level {level} over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'level': level, 'window': window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

        # Initialize VectorBT optimization
        self.level = level
        self.window = window
        self.base_calculation = base_calculation
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
        """Generate resistance level using VectorBT optimization."""
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            high = data['high']
            # Use VectorBT rolling optimizer for enhanced performance
            if self.rolling_optimizer:
                resistance_level = self.rolling_optimizer.rolling_max(high, window=self.window)
            elif VECTORBT_AVAILABLE and len(high) >= 1000:
                resistance_level = rolling_max(high, window=self.window)
            else:
                resistance_level = high.rolling(window=self.window).max()
        else:
            base_values = self.base_calculator.calculate(data)
            # Use VectorBT rolling optimizer for enhanced performance
            if self.rolling_optimizer:
                resistance_level = self.rolling_optimizer.rolling_max(base_values, window=self.window)
            elif VECTORBT_AVAILABLE and len(base_values) >= 1000:
                resistance_level = rolling_max(base_values, window=self.window)
            else:
                resistance_level = base_values.rolling(window=self.window).max()
        return resistance_level

# Pivot Point Generator

class PivotPointGenerator(VectorizedFeatureGenerator):
    """Generator for pivot point features with VectorBT optimization."""

    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 enable_gpu: bool = False, memory_efficient: bool = True, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        if 'high' not in required_columns:
            required_columns.append('high')
        if 'low' not in required_columns:
            required_columns.append('low')
        if 'close' not in required_columns:
            required_columns.append('close')

        config = FeatureConfig(
            name=f"pivot_point_{window}_{base_calculation.value}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"Pivot point over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

        # Initialize VectorBT optimization
        self.window = window
        self.base_calculation = base_calculation
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
        """Generate pivot point using VectorBT optimization."""
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            high = data['high']
            low = data['low']
            close = data['close']
            pivot_point = (high + low + close) / 3
        else:
            base_values = self.base_calculator.calculate(data)
            # Use VectorBT rolling optimizer for enhanced performance
            if self.rolling_optimizer:
                pivot_point = self.rolling_optimizer.rolling_mean(base_values, window=self.window)
            elif VECTORBT_AVAILABLE and len(base_values) >= 1000:
                pivot_point = rolling_mean(base_values, window=self.window)
            else:
                pivot_point = base_values.rolling(window=self.window).mean()
        return pivot_point

# Fibonacci Level Generator

class FibonacciLevelGenerator(VectorizedFeatureGenerator):
    """Generator for Fibonacci level features with VectorBT optimization."""

    def __init__(self, level: float = 0.618, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 enable_gpu: bool = False, memory_efficient: bool = True, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        if 'high' not in required_columns:
            required_columns.append('high')
        if 'low' not in required_columns:
            required_columns.append('low')

        config = FeatureConfig(
            name=f"fibonacci_{level}_{window}_{base_calculation.value}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"Fibonacci level {level} over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'level': level, 'window': window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

        # Initialize VectorBT optimization
        self.level = level
        self.window = window
        self.base_calculation = base_calculation
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
        """Generate Fibonacci level using VectorBT optimization."""
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            high = data['high']
            low = data['low']
            # Use VectorBT rolling optimizer for enhanced performance
            if self.rolling_optimizer:
                high_max = self.rolling_optimizer.rolling_max(high, window=self.window)
                low_min = self.rolling_optimizer.rolling_min(low, window=self.window)
                range_size = high_max - low_min
                fibonacci_level = low_min + (range_size * self.level)
            elif VECTORBT_AVAILABLE and len(high) >= 1000:
                high_max = rolling_max(high, window=self.window)
                low_min = rolling_min(low, window=self.window)
                range_size = high_max - low_min
                fibonacci_level = low_min + (range_size * self.level)
            else:
                range_size = high.rolling(window=self.window).max() - low.rolling(window=self.window).min()
                fibonacci_level = low.rolling(window=self.window).min() + (range_size * self.level)
        else:
            base_values = self.base_calculator.calculate(data)
            # Use VectorBT rolling optimizer for enhanced performance
            if self.rolling_optimizer:
                fibonacci_level = self.rolling_optimizer.rolling_quantile(base_values, window=self.window, q=self.level)
            elif VECTORBT_AVAILABLE and len(base_values) >= 1000:
                fibonacci_level = quantile(base_values, q=self.level, window=self.window)
            else:
                fibonacci_level = base_values.rolling(window=self.window).quantile(self.level)
        return fibonacci_level

def create_default_support_resistance_generators(enable_gpu: bool = False, memory_efficient: bool = True) -> List[FeatureGenerator]:
    """Create default support/resistance feature generators with VectorBT optimization."""
    windows = [5, 10, 20]
    fibonacci_levels = [0.236, 0.382, 0.5, 0.618, 0.786]

    generators = []

    # Create generators for each window with VectorBT optimization
    for window in windows:
        generators.extend([
            SupportLevelGenerator(1, window, enable_gpu=enable_gpu, memory_efficient=memory_efficient),
            SupportLevelGenerator(2, window, enable_gpu=enable_gpu, memory_efficient=memory_efficient),
            SupportLevelGenerator(3, window, enable_gpu=enable_gpu, memory_efficient=memory_efficient),
            SupportLevelGenerator(4, window, enable_gpu=enable_gpu, memory_efficient=memory_efficient),
            SupportLevelGenerator(5, window, enable_gpu=enable_gpu, memory_efficient=memory_efficient),
            ResistanceLevelGenerator(1, window, enable_gpu=enable_gpu, memory_efficient=memory_efficient),
            ResistanceLevelGenerator(2, window, enable_gpu=enable_gpu, memory_efficient=memory_efficient),
            ResistanceLevelGenerator(3, window, enable_gpu=enable_gpu, memory_efficient=memory_efficient),
            ResistanceLevelGenerator(4, window, enable_gpu=enable_gpu, memory_efficient=memory_efficient),
            ResistanceLevelGenerator(5, window, enable_gpu=enable_gpu, memory_efficient=memory_efficient),
            PivotPointGenerator(window, enable_gpu=enable_gpu, memory_efficient=memory_efficient),
        ])

    # Create Fibonacci level generators with VectorBT optimization
    for level in fibonacci_levels:
        for window in windows:
            generators.append(FibonacciLevelGenerator(level, window, enable_gpu=enable_gpu, memory_efficient=memory_efficient))

    return generators

# Advanced Support/Resistance Features using VectorBT

class AdvancedSupportResistanceGenerator(VectorizedFeatureGenerator):
    """Advanced support/resistance features using full VectorBT optimization."""

    def __init__(self, config: Optional[FeatureConfig] = None, enable_gpu: bool = False,
                 memory_efficient: bool = True, enable_monitoring: bool = True):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

        # Initialize VectorBT optimization components
        self.enable_gpu = enable_gpu
        self.memory_efficient = memory_efficient
        self.enable_monitoring = enable_monitoring

        # Initialize optimization manager
        if VECTORBT_OPTIMIZATION_AVAILABLE:
            self.optimization_manager = get_optimization_manager(
                enable_gpu=enable_gpu,
                enable_parallel=True,
                memory_efficient=memory_efficient,
                enable_monitoring=enable_monitoring
            )
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=enable_gpu,
                enable_parallel=True,
                memory_efficient=memory_efficient
            )
        else:
            self.optimization_manager = None
            self.rolling_optimizer = None

    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="advanced_support_resistance_features",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description="Advanced support/resistance features with VectorBT optimization",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=50,
            min_lookback=10,
            max_lookback=200,
            parameters={
                "sr_windows": [10, 20, 50],
                "volume_windows": [5, 10, 20],
                "fibonacci_levels": [0.236, 0.382, 0.5, 0.618, 0.786],
                "strength_threshold": 0.7,
                "breakout_threshold": 0.02
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate advanced support/resistance features using VectorBT optimization."""
        # Optimize DataFrame for processing
        if self.optimization_manager:
            data = self.optimization_manager.optimize_dataframe(data)
        elif hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        # Generate comprehensive SR features
        features = {}

        if 'close' in data.columns:
            close = data['close']

            # Multi-timeframe support/resistance
            features['sr_multi_timeframe'] = self._calculate_multi_timeframe_sr(close)

            # Dynamic support/resistance with adaptive windows
            features['sr_dynamic'] = self._calculate_dynamic_sr(close)

            # Support/resistance strength and quality
            features['sr_quality'] = self._calculate_sr_quality(close)

            # Breakout detection
            features['sr_breakout'] = self._calculate_breakout_detection(close)

            # Volume-weighted support/resistance
            if 'volume' in data.columns:
                features['sr_volume_weighted'] = self._calculate_volume_weighted_sr(close, data['volume'])

            # Pivot-based features
            if all(col in data.columns for col in ['high', 'low', 'close']):
                features['sr_pivot_advanced'] = self._calculate_advanced_pivot_sr(data['high'], data['low'], data['close'])
                features['sr_fibonacci_advanced'] = self._calculate_advanced_fibonacci_sr(data['high'], data['low'])

        # Combine features into a single comprehensive score
        if features:
            # Weighted combination of all features
            weights = {
                'sr_multi_timeframe': 0.25,
                'sr_dynamic': 0.25,
                'sr_quality': 0.20,
                'sr_breakout': 0.15,
                'sr_volume_weighted': 0.10,
                'sr_pivot_advanced': 0.05
            }

            combined_sr = pd.Series(0.0, index=data.index)
            for feature_name, feature_series in features.items():
                if feature_name in weights:
                    combined_sr += weights[feature_name] * feature_series

            return combined_sr
        else:
            return pd.Series(np.zeros(len(data)), index=data.index, name='advanced_sr_placeholder')

    def _calculate_multi_timeframe_sr(self, close: pd.Series) -> pd.Series:
        """Calculate multi-timeframe support/resistance levels."""
        if not self.rolling_optimizer:
            return pd.Series(np.zeros(len(close)), index=close.index)

        windows = self.config.parameters.get('sr_windows', [10, 20, 50])
        multi_sr = pd.Series(0.0, index=close.index)

        for window in windows:
            rolling_min = self.rolling_optimizer.rolling_min(close, window)
            rolling_max = self.rolling_optimizer.rolling_max(close, window)
            rolling_mean = self.rolling_optimizer.rolling_mean(close, window)

            # Weighted combination based on window size
            weight = 1.0 / window
            multi_sr += weight * (rolling_min + rolling_max + rolling_mean) / 3

        return multi_sr / len(windows)

    def _calculate_dynamic_sr(self, close: pd.Series) -> pd.Series:
        """Calculate dynamic support/resistance with adaptive windows."""
        if not self.rolling_optimizer:
            return pd.Series(np.zeros(len(close)), index=close.index)

        # Calculate volatility-based adaptive window
        volatility = self.rolling_optimizer.rolling_std(close, window=20)
        adaptive_window = (20 + (volatility / volatility.rolling(50).mean() * 10)).astype(int)
        adaptive_window = adaptive_window.clip(5, 50)

        # Dynamic SR calculation
        dynamic_sr = pd.Series(0.0, index=close.index)
        for i in range(len(close)):
            window = adaptive_window.iloc[i]
            if i >= window - 1:
                start_idx = max(0, i - window + 1)
                window_data = close.iloc[start_idx:i+1]
                dynamic_sr.iloc[i] = (window_data.min() + window_data.max() + window_data.mean()) / 3

        return dynamic_sr

    def _calculate_sr_quality(self, close: pd.Series) -> pd.Series:
        """Calculate support/resistance quality score."""
        if not self.rolling_optimizer:
            return pd.Series(np.zeros(len(close)), index=close.index)

        window = 20
        rolling_min = self.rolling_optimizer.rolling_min(close, window)
        rolling_max = self.rolling_optimizer.rolling_max(close, window)
        rolling_std = self.rolling_optimizer.rolling_std(close, window)

        # Quality based on consistency and strength
        range_size = rolling_max - rolling_min
        quality = 1 / (1 + rolling_std / range_size)

        return quality

    def _calculate_breakout_detection(self, close: pd.Series) -> pd.Series:
        """Detect support/resistance breakouts."""
        if not self.rolling_optimizer:
            return pd.Series(np.zeros(len(close)), index=close.index)

        window = 20
        rolling_min = self.rolling_optimizer.rolling_min(close, window)
        rolling_max = self.rolling_optimizer.rolling_max(close, window)

        # Breakout detection
        breakout_threshold = self.config.parameters.get('breakout_threshold', 0.02)
        breakout = pd.Series(0.0, index=close.index)

        # Resistance breakout
        resistance_breakout = (close > rolling_max * (1 + breakout_threshold)).astype(float)
        # Support breakout
        support_breakout = (close < rolling_min * (1 - breakout_threshold)).astype(float)

        breakout = resistance_breakout - support_breakout
        return breakout

    def _calculate_volume_weighted_sr(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate volume-weighted support/resistance."""
        if not self.rolling_optimizer:
            return pd.Series(np.zeros(len(close)), index=close.index)

        window = self.config.parameters.get('volume_windows', [20])[0]

        # Volume-weighted price
        volume_weighted_price = (self.rolling_optimizer.rolling_sum(close * volume, window) /
                                self.rolling_optimizer.rolling_sum(volume, window))

        return volume_weighted_price

    def _calculate_advanced_pivot_sr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate advanced pivot-based support/resistance."""
        if not self.rolling_optimizer:
            return pd.Series(np.zeros(len(close)), index=close.index)

        window = 20

        # Traditional pivot points
        pivot = (high + low + close) / 3

        # Support and resistance levels
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)

        # Rolling average of pivot levels
        rolling_pivot = self.rolling_optimizer.rolling_mean(pivot, window)
        rolling_r1 = self.rolling_optimizer.rolling_mean(r1, window)
        rolling_s1 = self.rolling_optimizer.rolling_mean(s1, window)

        # Combined pivot SR level
        advanced_pivot = (rolling_pivot + rolling_r1 + rolling_s1) / 3

        return advanced_pivot

    def _calculate_advanced_fibonacci_sr(self, high: pd.Series, low: pd.Series) -> pd.Series:
        """Calculate advanced Fibonacci support/resistance levels."""
        if not self.rolling_optimizer:
            return pd.Series(np.zeros(len(high)), index=high.index)

        window = 20
        fibonacci_levels = self.config.parameters.get('fibonacci_levels', [0.618])

        rolling_high = self.rolling_optimizer.rolling_max(high, window)
        rolling_low = self.rolling_optimizer.rolling_min(low, window)
        range_size = rolling_high - rolling_low

        # Calculate multiple Fibonacci levels
        fibonacci_sr = pd.Series(0.0, index=high.index)
        for level in fibonacci_levels:
            fib_level = rolling_low + (range_size * level)
            fibonacci_sr += fib_level

        return fibonacci_sr / len(fibonacci_levels)
