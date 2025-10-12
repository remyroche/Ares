"""
Acceleration Feature Generators

This module provides feature generators for acceleration, velocity, and jerk indicators,
including momentum derivatives, trend strength, and consistency measures.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from scipy import stats

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

# Optimization utilities
try:
    from ..utils.vectorization_optimizer import get_vectorization_optimizer
    from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

# VectorBT Rolling Optimizer
try:
    from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = True
except ImportError:
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = False
    VectorBTRollingOptimizer = None

# Unified Vectorization Manager
try:
    from ...utils.ml_common.unified_vectorization_manager import get_unified_vectorization_manager, UnifiedVectorizationManager, OperationType, OptimizationStrategy
    UNIFIED_VECTORIZATION_AVAILABLE = True
except ImportError:
    UNIFIED_VECTORIZATION_AVAILABLE = False
    UnifiedVectorizationManager = None
from ..base_calculations import (
    BaseCalculationType,
    create_base_calculator
)
from ...utils.math_validation import safe_divide, validate_finite, safe_percentage_change

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

# Import VectorBT-optimized acceleration generators
try:
    from .vectorbt_acceleration import (
        create_vectorbt_acceleration_generators,
        create_default_vectorbt_acceleration_generators,
        VectorBTMomentumGenerator,
        VectorBTPriceAccelerationGenerator,
        VectorBTPriceJerkGenerator,
        VectorBTTrendStrengthGenerator,
        VectorBTTrendConsistencyGenerator,
        VectorBTVolumeAccelerationGenerator,
        VectorBTVolatilityAccelerationGenerator,
        VectorBTMomentumAccelerationGenerator,
        VectorBTAccelerationMomentumGenerator,
        VectorBTAccelerationVolatilityGenerator,
        VectorBTAccelerationTrendStrengthGenerator,
        VectorBTAccelerationConsistencyGenerator,
        VectorBTAccelerationRegimeGenerator,
        VectorBTMultiTimeframeAccelerationGenerator,
        VectorBTAccelerationCorrelationGenerator,
        VectorBTAccelerationDivergenceGenerator
    )
    VECTORBT_ACCELERATION_AVAILABLE = True
except ImportError:
    VECTORBT_ACCELERATION_AVAILABLE = False

class AccelerationFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for acceleration-based features with full VectorBT optimization."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        # Initialize VectorBT optimization components
        self.vectorbt_optimizer = None
        self.unified_manager = None
        
        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=True, enable_parallel=True)
        
        if UNIFIED_VECTORIZATION_AVAILABLE:
            self.unified_manager = get_unified_vectorization_manager()
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing using VectorBT and UnifiedVectorizationManager."""
        # Use UnifiedVectorizationManager for intelligent optimization
        if self.unified_manager and UNIFIED_VECTORIZATION_AVAILABLE:
            try:
                from ...utils.ml_common.unified_vectorization_manager import OperationConfig
                config = OperationConfig(
                    operation_type=OperationType.FEATURE_ENGINEERING,
                    data_size=len(data),
                    data_dimensions=data.shape,
                    memory_budget_mb=1024.0
                )
                
                # Use UnifiedVectorizationManager for optimization
                result = self.unified_manager.optimize_operation(
                    OperationType.FEATURE_ENGINEERING,
                    data,
                    config
                )
                return result.result
            except Exception as e:
                self.logger.warning(f"UnifiedVectorizationManager optimization failed: {e}")
        
        # Fallback to VectorBT Rolling Optimizer
        if self.vectorbt_optimizer and VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            try:
                # Use VectorBT optimization for data processing
                return self.vectorbt_optimizer._optimize_data_types(data)
            except Exception as e:
                self.logger.warning(f"VectorBTRollingOptimizer optimization failed: {e}")
        
        # Fallback to original vectorization optimizer
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with VectorBT optimization."""
        # Use VectorBT Rolling Optimizer for enhanced performance
        if self.vectorbt_optimizer and VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            try:
                results = {}
                for i, operation in enumerate(operations):
                    window = windows[i] if i < len(windows) else windows[0]
                    column = columns[i] if columns and i < len(columns) else 'close'
                    
                    if column in data.columns:
                        if operation == 'mean':
                            results[f"{operation}_{window}_{column}"] = self.vectorbt_optimizer.rolling_mean(data[column], window)
                        elif operation == 'std':
                            results[f"{operation}_{window}_{column}"] = self.vectorbt_optimizer.rolling_std(data[column], window)
                        elif operation == 'var':
                            results[f"{operation}_{window}_{column}"] = self.vectorbt_optimizer.rolling_var(data[column], window)
                        elif operation == 'min':
                            results[f"{operation}_{window}_{column}"] = self.vectorbt_optimizer.rolling_min(data[column], window)
                        elif operation == 'max':
                            results[f"{operation}_{window}_{column}"] = self.vectorbt_optimizer.rolling_max(data[column], window)
                        elif operation == 'sum':
                            results[f"{operation}_{window}_{column}"] = self.vectorbt_optimizer.rolling_sum(data[column], window)
                
                return pd.DataFrame(results, index=data.index)
            except Exception as e:
                self.logger.warning(f"VectorBTRollingOptimizer rolling operations failed: {e}")
        
        # Fallback to original vectorization optimizer
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="acceleration_features",
            category=FeatureCategory.ACCELERATION,
            description="Comprehensive acceleration features including momentum, acceleration, and jerk",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=20,
            min_lookback=5,
            max_lookback=100,
            parameters={
                "acceleration_windows": [5, 10, 20],
                "momentum_windows": [5, 10, 20, 50]
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )

# Price Momentum Generator

class MomentumGenerator(FeatureGenerator):
    """Generator for price momentum features with VectorBT optimization."""
    
    def __init__(self, period: int = 10, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"momentum_{period}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"Price momentum over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={'period': period, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation
        
        # Initialize VectorBT optimization components
        self.vectorbt_optimizer = None
        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=True, enable_parallel=True)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate momentum with VectorBT optimization and math validation."""
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        base_values = self.base_calculator.calculate(data)

        # Use VectorBT optimization for momentum calculation
        if self.vectorbt_optimizer and VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            try:
                # Calculate momentum using VectorBT optimized operations
                shifted_values = base_values.shift(self.period)
                momentum_series = safe_percentage_change(shifted_values, base_values)
                momentum_series.name = f'momentum_{self.period}_{self.base_calculation.value}'
                
                # Validate finite values
                try:
                    validate_finite(momentum_series.values, f"VectorBT_Momentum_{self.period}_{self.base_calculation.value}")
                except ValueError as e:
                    self.logger.warning(f"⚠️ {e}")
                
                return momentum_series
                
            except Exception as e:
                self.logger.warning(f"VectorBT momentum calculation failed: {e}, using fallback")

        # Fallback to original calculation
        shifted_values = base_values.shift(self.period)

        # Use safe percentage change calculation
        momentum_values = []
        for i in range(len(base_values)):
            current_val = base_values.iloc[i]
            shifted_val = shifted_values.iloc[i]

            # Use safe percentage change function
            momentum_val = safe_percentage_change(shifted_val, current_val)
            momentum_values.append(momentum_val)

        momentum_series = pd.Series(momentum_values, index=data.index, name=f'momentum_{self.period}_{self.base_calculation.value}')

        # Validate that all values are finite and provide detailed information
        try:
            validate_finite(momentum_series.values, f"Momentum_{self.period}_{self.base_calculation.value}")
        except ValueError as e:
            # Get detailed information about where the NaN/inf values are
            non_finite_mask = ~np.isfinite(momentum_series.values)
            if np.any(non_finite_mask):
                non_finite_indices = np.where(non_finite_mask)[0]
                total_count = len(non_finite_indices)

                # Show first few and last few problematic indices
                if total_count <= 10:
                    indices_str = f"indices {non_finite_indices.tolist()}"
                else:
                    first_5 = non_finite_indices[:5].tolist()
                    last_5 = non_finite_indices[-5:].tolist()
                    indices_str = f"indices {first_5} ... {last_5} (total: {total_count})"

                # Only log once per feature globally to reduce verbosity
                feature_key = f"Momentum_{self.period}_{self.base_calculation.value}"
                # Use class-level tracking to prevent duplicate warnings across all instances
                if not hasattr(MomentumGenerator, '_logged_warnings'):
                    MomentumGenerator._logged_warnings = set()
                if feature_key not in MomentumGenerator._logged_warnings:
                    self.logger.warning(f"⚠️ {e} - {indices_str}")
                    MomentumGenerator._logged_warnings.add(feature_key)
            else:
                self.logger.warning(f"⚠️ {e}")

        return momentum_series

# Price Acceleration Generator
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class PriceAccelerationGenerator(FeatureGenerator):
    """Generator for price acceleration features."""
    
    def __init__(self, period: int = 10, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"acceleration_{period}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"Price acceleration over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period * 2,
            min_lookback=period * 2,
            max_lookback=period * 2,
            parameters={'period': period, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate acceleration (second derivative of price) with math validation."""
        base_values = self.base_calculator.calculate(data)

        # Calculate momentum with proper handling using safe math utilities
        shifted_values = base_values.shift(self.period)

        # Use safe percentage change calculation for momentum
        momentum_values = []
        for i in range(len(base_values)):
            current_val = base_values.iloc[i]
            shifted_val = shifted_values.iloc[i]

            # Use safe percentage change function
            momentum_val = safe_percentage_change(shifted_val, current_val)
            momentum_values.append(momentum_val)

        momentum_series = pd.Series(momentum_values, index=data.index)

        # Calculate acceleration (second derivative) using diff
        acceleration = momentum_series.diff(self.period)

        # Validate that all values are finite and provide detailed information
        try:
            validate_finite(acceleration.values, f"Acceleration_{self.period}_{self.base_calculation.value}")
        except ValueError as e:
            # Get detailed information about where the NaN/inf values are
            non_finite_mask = ~np.isfinite(acceleration.values)
            if np.any(non_finite_mask):
                non_finite_indices = np.where(non_finite_mask)[0]
                total_count = len(non_finite_indices)

                # Show first few and last few problematic indices
                if total_count <= 10:
                    indices_str = f"indices {non_finite_indices.tolist()}"
                else:
                    first_5 = non_finite_indices[:5].tolist()
                    last_5 = non_finite_indices[-5:].tolist()
                    indices_str = f"indices {first_5} ... {last_5} (total: {total_count})"

                # Only log once per feature globally to reduce verbosity
                feature_key = f"Acceleration_{self.period}_{self.base_calculation.value}"
                # Use class-level tracking to prevent duplicate warnings across all instances
                if not hasattr(PriceAccelerationGenerator, '_logged_warnings'):
                    PriceAccelerationGenerator._logged_warnings = set()
                if feature_key not in PriceAccelerationGenerator._logged_warnings:
                    self.logger.warning(f"⚠️ {e} - {indices_str}")
                    PriceAccelerationGenerator._logged_warnings.add(feature_key)
            else:
                self.logger.warning(f"⚠️ {e}")

        return acceleration

# Price Jerk Generator
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class PriceJerkGenerator(FeatureGenerator):
    """Generator for price jerk features (third derivative)."""
    
    def __init__(self, period: int = 10, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"jerk_{period}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"Price jerk over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period * 3,
            min_lookback=period * 3,
            max_lookback=period * 3,
            parameters={'period': period, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate jerk (third derivative of price) with math validation."""
        base_values = self.base_calculator.calculate(data)

        # Calculate momentum with proper handling using safe math utilities
        shifted_values = base_values.shift(self.period)

        # Use safe percentage change calculation for momentum
        momentum_values = []
        for i in range(len(base_values)):
            current_val = base_values.iloc[i]
            shifted_val = shifted_values.iloc[i]

            # Use safe percentage change function
            momentum_val = safe_percentage_change(shifted_val, current_val)
            momentum_values.append(momentum_val)

        momentum_series = pd.Series(momentum_values, index=data.index)

        # Calculate acceleration (second derivative) using diff
        acceleration = momentum_series.diff(self.period)

        # Calculate jerk (third derivative) using diff
        jerk = acceleration.diff(self.period)

        # Validate that all values are finite and provide detailed information
        try:
            validate_finite(jerk.values, f"Jerk_{self.period}_{self.base_calculation.value}")
        except ValueError as e:
            # Get detailed information about where the NaN/inf values are
            non_finite_mask = ~np.isfinite(jerk.values)
            if np.any(non_finite_mask):
                non_finite_indices = np.where(non_finite_mask)[0]
                total_count = len(non_finite_indices)

                # Show first few and last few problematic indices
                if total_count <= 10:
                    indices_str = f"indices {non_finite_indices.tolist()}"
                else:
                    first_5 = non_finite_indices[:5].tolist()
                    last_5 = non_finite_indices[-5:].tolist()
                    indices_str = f"indices {first_5} ... {last_5} (total: {total_count})"

                # Only log once per feature globally to reduce verbosity
                feature_key = f"Jerk_{self.period}_{self.base_calculation.value}"
                # Use class-level tracking to prevent duplicate warnings across all instances
                if not hasattr(PriceJerkGenerator, '_logged_warnings'):
                    PriceJerkGenerator._logged_warnings = set()
                if feature_key not in PriceJerkGenerator._logged_warnings:
                    self.logger.warning(f"⚠️ {e} - {indices_str}")
                    PriceJerkGenerator._logged_warnings.add(feature_key)
            else:
                self.logger.warning(f"⚠️ {e}")

        return jerk

# Trend Strength Generator
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class TrendStrengthGenerator(FeatureGenerator):
    """Generator for trend strength features using polyfit."""
    
    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"trend_strength_{window}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"Trend strength over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate trend strength using VectorBT optimization."""
        base_values = self.base_calculator.calculate(data)
        
        def calculate_trend_strength(series):
            if len(series) < 2:
                return 0.0
            try:
                # Calculate linear regression slope
                slope = np.polyfit(range(len(series)), series, 1)[0]
                return slope
            except:
                return 0.0
        
        # Use VectorBT rolling apply if available
        if VECTORBT_AVAILABLE:
            try:
                trend_strength = rolling_apply(base_values, calculate_trend_strength, window=self.window)
                return trend_strength
            except Exception as e:
                self.logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
        
        # Fallback to pandas rolling apply
        trend_strength = base_values.rolling(window=self.window).apply(calculate_trend_strength, raw=False)
        return trend_strength

# Trend Consistency Generator
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class TrendConsistencyGenerator(FeatureGenerator):
    """Generator for trend consistency features."""
    
    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"trend_consistency_{window}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"Trend consistency over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate trend consistency using VectorBT optimization."""
        base_values = self.base_calculator.calculate(data)
        
        def calculate_trend_consistency(series):
            if len(series) < 2:
                return 0
            try:
                # Calculate linear regression slope and return 1 if positive, 0 otherwise
                slope = np.polyfit(range(len(series)), series, 1)[0]
                return 1 if slope > 0 else 0
            except:
                return 0
        
        # Use VectorBT rolling apply if available
        if VECTORBT_AVAILABLE:
            try:
                trend_consistency = rolling_apply(base_values, calculate_trend_consistency, window=self.window)
                return trend_consistency
            except Exception as e:
                self.logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
        
        # Fallback to pandas rolling apply
        trend_consistency = base_values.rolling(window=self.window).apply(calculate_trend_consistency, raw=False)
        return trend_consistency

# Volume Acceleration Generator
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class VolumeAccelerationGenerator(FeatureGenerator):
    """Generator for volume acceleration features."""
    
    def __init__(self, period: int = 5, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.VOLUME_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"volume_acceleration_{period}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"Volume acceleration over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period * 2,
            min_lookback=period * 2,
            max_lookback=period * 2,
            parameters={'period': period, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate volume acceleration."""
        base_values = self.base_calculator.calculate(data)
        volume_acceleration = base_values.diff(self.period).diff(self.period)
        return volume_acceleration

# Volatility Acceleration Generator
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class VolatilityAccelerationGenerator(FeatureGenerator):
    """Generator for volatility acceleration features."""
    
    def __init__(self, period: int = 5, volatility_window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"volatility_acceleration_{period}_{volatility_window}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"Volatility acceleration over {period} periods with {volatility_window} volatility window based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=volatility_window + period * 2,
            min_lookback=volatility_window + period * 2,
            max_lookback=volatility_window + period * 2,
            parameters={'period': period, 'volatility_window': volatility_window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.volatility_window = volatility_window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate volatility acceleration using VectorBT optimization."""
        base_values = self.base_calculator.calculate(data)
        
        # Use VectorBT rolling std if available
        if VECTORBT_AVAILABLE:
            try:
                volatility = rolling_std(base_values, window=self.volatility_window)
                volatility_acceleration = volatility.diff(self.period).diff(self.period)
                return volatility_acceleration
            except Exception as e:
                self.logger.warning(f"VectorBT rolling std failed: {e}, using pandas fallback")
        
        # Fallback to pandas rolling std
        volatility = base_values.rolling(window=self.volatility_window).std()
        volatility_acceleration = volatility.diff(self.period).diff(self.period)
        return volatility_acceleration

def create_acceleration_generators() -> List[FeatureGenerator]:
    """Create all acceleration-based feature generators with VectorBT optimization."""
    generators = []
    
    # Always prioritize VectorBT generators if available
    if VECTORBT_ACCELERATION_AVAILABLE and VECTORBT_AVAILABLE:
        # Use VectorBT-optimized generators
        generators.extend(create_vectorbt_acceleration_generators())
        self.logger.info(f"✅ Created {len(generators)} VectorBT-optimized acceleration generators")
    else:
        self.logger.warning("⚠️ VectorBT acceleration generators not available, using legacy generators")
    
    # Add enhanced legacy generators with VectorBT optimization
    # Momentum generators for different periods
    for period in [5, 10, 20, 50]:
        generators.append(MomentumGenerator(period=period))
    
    # Acceleration generators
    for period in [5, 10]:
        generators.append(PriceAccelerationGenerator(period=period))
    
    # Jerk generators
    for period in [5, 10]:
        generators.append(PriceJerkGenerator(period=period))
    
    # Trend strength generators
    for window in [5, 10, 20, 50]:
        generators.append(TrendStrengthGenerator(window=window))
    
    # Trend consistency generators
    for window in [5, 10, 20, 50]:
        generators.append(TrendConsistencyGenerator(window=window))
    
    # Volume acceleration
    generators.append(VolumeAccelerationGenerator(period=5))
    
    # Volatility acceleration
    generators.append(VolatilityAccelerationGenerator(period=5, volatility_window=20))
    
    return generators

def create_default_acceleration_generators() -> List[FeatureGenerator]:
    """Create default acceleration-based feature generators (alias for create_acceleration_generators)."""
    return create_acceleration_generators()

# Export all generators
__all__ = [
    'AccelerationFeatureGenerator',
    'MomentumGenerator',
    'PriceAccelerationGenerator', 
    'PriceJerkGenerator',
    'TrendStrengthGenerator',
    'TrendConsistencyGenerator',
    'VolumeAccelerationGenerator',
    'VolatilityAccelerationGenerator',
    'create_acceleration_generators',
    'create_default_acceleration_generators'
]
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
