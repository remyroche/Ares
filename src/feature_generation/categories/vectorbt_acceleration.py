"""
VectorBT-Optimized Acceleration Feature Generators

This module provides high-performance acceleration feature generators using VectorBT's
optimized C++ backend for maximum performance in feature generation.

Features:
- Price momentum
- Price acceleration
- Price jerk
- Trend strength
- Trend consistency
- Volume acceleration
- Volatility acceleration
- Momentum acceleration
- Acceleration momentum
- Acceleration volatility
- Acceleration trend strength
- Acceleration consistency
- Acceleration regime detection
- Multi-timeframe acceleration
- Cross-asset acceleration
- Acceleration correlation
- Acceleration divergence
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Optional, Dict, Any, Union
from scipy import stats

from ..core.vectorbt_feature_generator import VectorBTFeatureGenerator
from ..core.feature_generator import FeatureConfig, FeatureCategory
from ..base_calculations import BaseCalculationType, create_base_calculator
from ...utils.math_validation import safe_divide, validate_finite, safe_percentage_change

# Enhanced VectorBT optimization imports
try:
    from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
    from ...utils.ml_common.unified_vectorization_manager import (
        get_unified_vectorization_manager, UnifiedVectorizationManager, 
        OperationType, OptimizationStrategy, OperationConfig
    )
    VECTORBT_OPTIMIZATION_AVAILABLE = True
except ImportError:
    VECTORBT_OPTIMIZATION_AVAILABLE = False
    VectorBTRollingOptimizer = None
    UnifiedVectorizationManager = None

logger = logging.getLogger(__name__)

class VectorBTMomentumGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized momentum generator with enhanced optimization."""
    
    def __init__(self, period: int = 10, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, 
                 enable_optimization: bool = True, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"vectorbt_momentum_{period}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"VectorBT-optimized momentum over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={'period': period, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.period = period
        self.base_calculation = base_calculation
        self.enable_optimization = enable_optimization and VECTORBT_OPTIMIZATION_AVAILABLE
        
        # Initialize optimization components
        if self.enable_optimization:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=True, enable_parallel=True)
            self.unified_manager = get_unified_vectorization_manager()
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate momentum using enhanced VectorBT operations with optimization."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_momentum_{self.period}_{self.base_calculation.value}')
        
        base_values = self.base_calculator.calculate(data)
        
        if self.enable_optimization:
            # Use VectorBTRollingOptimizer for enhanced performance
            try:
                # Calculate momentum using optimized rolling operations
                shifted_values = base_values.shift(self.period)
                momentum = safe_percentage_change(shifted_values, base_values)
                
                # Apply additional VectorBT optimizations if data is large enough
                if len(base_values) > 1000:
                    # Use VectorBTRollingOptimizer for additional smoothing if needed
                    momentum = self.rolling_optimizer.rolling_mean(momentum, window=min(3, self.period))
                
                # GPU acceleration for very large datasets
                if len(base_values) > 10000 and hasattr(self.rolling_optimizer, 'enable_gpu') and self.rolling_optimizer.enable_gpu:
                    try:
                        # Apply GPU-accelerated smoothing
                        momentum = self.rolling_optimizer.rolling_ewm(momentum, window=min(5, self.period), alpha=0.3)
                    except Exception as gpu_e:
                        logger.warning(f"GPU acceleration failed: {gpu_e}, using CPU")
                
            except Exception as e:
                logger.warning(f"VectorBTRollingOptimizer failed: {e}, using fallback")
                # Fallback to standard calculation
                shifted_values = base_values.shift(self.period)
                momentum = safe_percentage_change(shifted_values, base_values)
        else:
            # Standard VectorBT calculation
            shifted_values = base_values.shift(self.period)
            momentum = safe_percentage_change(shifted_values, base_values)
        
        # Validate finite values
        try:
            validate_finite(momentum.values, f"VectorBT_Momentum_{self.period}_{self.base_calculation.value}")
        except ValueError as e:
            logger.warning(f"⚠️ {e}")
        
        return momentum.rename(f'vectorbt_momentum_{self.period}_{self.base_calculation.value}')


class VectorBTPriceAccelerationGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized price acceleration generator with enhanced optimization."""
    
    def __init__(self, period: int = 10, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, 
                 enable_optimization: bool = True, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"vectorbt_acceleration_{period}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"VectorBT-optimized acceleration over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=period * 2,
            min_lookback=period * 2,
            max_lookback=period * 2,
            parameters={'period': period, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.period = period
        self.base_calculation = base_calculation
        self.enable_optimization = enable_optimization and VECTORBT_OPTIMIZATION_AVAILABLE
        
        # Initialize optimization components
        if self.enable_optimization:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=True, enable_parallel=True)
            self.unified_manager = get_unified_vectorization_manager()
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate acceleration using enhanced VectorBT operations with optimization."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_acceleration_{self.period}_{self.base_calculation.value}')
        
        base_values = self.base_calculator.calculate(data)
        
        if self.enable_optimization:
            # Use VectorBTRollingOptimizer for enhanced performance
            try:
                # Calculate momentum first using optimized operations
                shifted_values = base_values.shift(self.period)
                momentum = safe_percentage_change(shifted_values, base_values)
                
                # Calculate acceleration (second derivative) using VectorBTRollingOptimizer
                acceleration = self.rolling_optimizer.rolling_mean(momentum.diff(self.period), window=min(3, self.period))
                
                # Apply additional smoothing for large datasets
                if len(base_values) > 2000:
                    acceleration = self.rolling_optimizer.rolling_std(acceleration, window=min(5, self.period))
                
            except Exception as e:
                logger.warning(f"VectorBTRollingOptimizer failed: {e}, using fallback")
                # Fallback to standard calculation
                shifted_values = base_values.shift(self.period)
                momentum = safe_percentage_change(shifted_values, base_values)
                acceleration = momentum.diff(self.period)
        else:
            # Standard VectorBT calculation
            shifted_values = base_values.shift(self.period)
            momentum = safe_percentage_change(shifted_values, base_values)
            acceleration = momentum.diff(self.period)
        
        # Validate finite values
        try:
            validate_finite(acceleration.values, f"VectorBT_Acceleration_{self.period}_{self.base_calculation.value}")
        except ValueError as e:
            logger.warning(f"⚠️ {e}")
        
        return acceleration.rename(f'vectorbt_acceleration_{self.period}_{self.base_calculation.value}')


class VectorBTPriceJerkGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized price jerk generator."""
    
    def __init__(self, period: int = 10, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"vectorbt_jerk_{period}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"VectorBT-optimized jerk over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=period * 3,
            min_lookback=period * 3,
            max_lookback=period * 3,
            parameters={'period': period, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate jerk using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_jerk_{self.period}_{self.base_calculation.value}')
        
        base_values = self.base_calculator.calculate(data)
        
        # Calculate momentum first
        shifted_values = base_values.shift(self.period)
        momentum = safe_percentage_change(shifted_values, base_values)
        
        # Calculate acceleration (second derivative)
        acceleration = momentum.diff(self.period)
        
        # Calculate jerk (third derivative) using VectorBT
        jerk = acceleration.diff(self.period)
        
        # Validate finite values
        try:
            validate_finite(jerk.values, f"VectorBT_Jerk_{self.period}_{self.base_calculation.value}")
        except ValueError as e:
            logger.warning(f"⚠️ {e}")
        
        return jerk.rename(f'vectorbt_jerk_{self.period}_{self.base_calculation.value}')


class VectorBTTrendStrengthGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized trend strength generator with enhanced optimization."""
    
    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, 
                 enable_optimization: bool = True, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"vectorbt_trend_strength_{window}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"VectorBT-optimized trend strength over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.window = window
        self.base_calculation = base_calculation
        self.enable_optimization = enable_optimization and VECTORBT_OPTIMIZATION_AVAILABLE
        
        # Initialize optimization components
        if self.enable_optimization:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=True, enable_parallel=True)
            self.unified_manager = get_unified_vectorization_manager()
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate trend strength using enhanced VectorBT operations with optimization."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_trend_strength_{self.window}_{self.base_calculation.value}')
        
        base_values = self.base_calculator.calculate(data)
        
        if self.enable_optimization:
            # Use VectorBTRollingOptimizer for enhanced performance
            try:
                # Calculate trend strength using optimized rolling correlation with time
                time_index = pd.Series(range(len(base_values)), index=base_values.index)
                trend_strength = self.rolling_optimizer.rolling_corr(
                    base_values, time_index, window=self.window
                )
                
                # Apply additional smoothing for large datasets
                if len(base_values) > 2000:
                    trend_strength = self.rolling_optimizer.rolling_mean(trend_strength, window=min(5, self.window))
                
            except Exception as e:
                logger.warning(f"VectorBTRollingOptimizer failed: {e}, using fallback")
                # Fallback to standard VectorBT calculation
                time_index = pd.Series(range(len(base_values)), index=base_values.index)
                trend_strength = self._vectorbt_rolling_operation(
                    base_values, 'corr', window=self.window, other=time_index
                )
        else:
            # Standard VectorBT calculation
            time_index = pd.Series(range(len(base_values)), index=base_values.index)
            trend_strength = self._vectorbt_rolling_operation(
                base_values, 'corr', window=self.window, other=time_index
            )
        
        return trend_strength.rename(f'vectorbt_trend_strength_{self.window}_{self.base_calculation.value}')


class VectorBTTrendConsistencyGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized trend consistency generator."""
    
    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"vectorbt_trend_consistency_{window}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"VectorBT-optimized trend consistency over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate trend consistency using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_trend_consistency_{self.window}_{self.base_calculation.value}')
        
        base_values = self.base_calculator.calculate(data)
        
        # Calculate trend consistency as inverse of volatility
        volatility = self._vectorbt_rolling_operation(base_values, 'std', window=self.window)
        consistency = 1.0 / (volatility + 1e-8)  # Add small epsilon to avoid division by zero
        
        return consistency.rename(f'vectorbt_trend_consistency_{self.window}_{self.base_calculation.value}')


class VectorBTVolumeAccelerationGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized volume acceleration generator."""
    
    def __init__(self, period: int = 5, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.VOLUME_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"vectorbt_volume_acceleration_{period}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"VectorBT-optimized volume acceleration over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["high", "low", "open", "close"],
            default_lookback=period * 2,
            min_lookback=period * 2,
            max_lookback=period * 2,
            parameters={'period': period, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate volume acceleration using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_volume_acceleration_{self.period}_{self.base_calculation.value}')
        
        base_values = self.base_calculator.calculate(data)
        
        # Calculate volume acceleration using VectorBT
        volume_acceleration = base_values.diff(self.period).diff(self.period)
        
        return volume_acceleration.rename(f'vectorbt_volume_acceleration_{self.period}_{self.base_calculation.value}')


class VectorBTVolatilityAccelerationGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized volatility acceleration generator."""
    
    def __init__(self, period: int = 5, volatility_window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"vectorbt_volatility_acceleration_{period}_{volatility_window}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"VectorBT-optimized volatility acceleration over {period} periods with {volatility_window} volatility window based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=volatility_window + period * 2,
            min_lookback=volatility_window + period * 2,
            max_lookback=volatility_window + period * 2,
            parameters={'period': period, 'volatility_window': volatility_window, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.period = period
        self.volatility_window = volatility_window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate volatility acceleration using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_volatility_acceleration_{self.period}_{self.volatility_window}_{self.base_calculation.value}')
        
        base_values = self.base_calculator.calculate(data)
        
        # Calculate volatility using VectorBT
        volatility = self._vectorbt_rolling_operation(base_values, 'std', window=self.volatility_window)
        
        # Calculate volatility acceleration using VectorBT
        volatility_acceleration = volatility.diff(self.period).diff(self.period)
        
        return volatility_acceleration.rename(f'vectorbt_volatility_acceleration_{self.period}_{self.volatility_window}_{self.base_calculation.value}')


class VectorBTMomentumAccelerationGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized momentum acceleration generator."""
    
    def __init__(self, period: int = 10, momentum_window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"vectorbt_momentum_acceleration_{period}_{momentum_window}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"VectorBT-optimized momentum acceleration over {period} periods with {momentum_window} momentum window based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=momentum_window + period * 2,
            min_lookback=momentum_window + period * 2,
            max_lookback=momentum_window + period * 2,
            parameters={'period': period, 'momentum_window': momentum_window, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.period = period
        self.momentum_window = momentum_window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate momentum acceleration using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_momentum_acceleration_{self.period}_{self.momentum_window}_{self.base_calculation.value}')
        
        base_values = self.base_calculator.calculate(data)
        
        # Calculate momentum using VectorBT
        momentum = self._vectorbt_rolling_operation(base_values, 'mean', window=self.momentum_window)
        
        # Calculate momentum acceleration using VectorBT
        momentum_acceleration = momentum.diff(self.period).diff(self.period)
        
        return momentum_acceleration.rename(f'vectorbt_momentum_acceleration_{self.period}_{self.momentum_window}_{self.base_calculation.value}')


class VectorBTAccelerationMomentumGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized acceleration momentum generator."""
    
    def __init__(self, period: int = 10, acceleration_window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"vectorbt_acceleration_momentum_{period}_{acceleration_window}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"VectorBT-optimized acceleration momentum over {period} periods with {acceleration_window} acceleration window based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=acceleration_window + period * 2,
            min_lookback=acceleration_window + period * 2,
            max_lookback=acceleration_window + period * 2,
            parameters={'period': period, 'acceleration_window': acceleration_window, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.period = period
        self.acceleration_window = acceleration_window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate acceleration momentum using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_acceleration_momentum_{self.period}_{self.acceleration_window}_{self.base_calculation.value}')
        
        base_values = self.base_calculator.calculate(data)
        
        # Calculate acceleration first
        shifted_values = base_values.shift(self.acceleration_window)
        momentum = safe_percentage_change(shifted_values, base_values)
        acceleration = momentum.diff(self.acceleration_window)
        
        # Calculate acceleration momentum using VectorBT
        acceleration_momentum = self._vectorbt_rolling_operation(acceleration, 'mean', window=self.period)
        
        return acceleration_momentum.rename(f'vectorbt_acceleration_momentum_{self.period}_{self.acceleration_window}_{self.base_calculation.value}')


class VectorBTAccelerationVolatilityGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized acceleration volatility generator."""
    
    def __init__(self, period: int = 10, acceleration_window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"vectorbt_acceleration_volatility_{period}_{acceleration_window}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"VectorBT-optimized acceleration volatility over {period} periods with {acceleration_window} acceleration window based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=acceleration_window + period * 2,
            min_lookback=acceleration_window + period * 2,
            max_lookback=acceleration_window + period * 2,
            parameters={'period': period, 'acceleration_window': acceleration_window, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.period = period
        self.acceleration_window = acceleration_window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate acceleration volatility using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_acceleration_volatility_{self.period}_{self.acceleration_window}_{self.base_calculation.value}')
        
        base_values = self.base_calculator.calculate(data)
        
        # Calculate acceleration first
        shifted_values = base_values.shift(self.acceleration_window)
        momentum = safe_percentage_change(shifted_values, base_values)
        acceleration = momentum.diff(self.acceleration_window)
        
        # Calculate acceleration volatility using VectorBT
        acceleration_volatility = self._vectorbt_rolling_operation(acceleration, 'std', window=self.period)
        
        return acceleration_volatility.rename(f'vectorbt_acceleration_volatility_{self.period}_{self.acceleration_window}_{self.base_calculation.value}')


class VectorBTAccelerationTrendStrengthGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized acceleration trend strength generator."""
    
    def __init__(self, period: int = 10, acceleration_window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"vectorbt_acceleration_trend_strength_{period}_{acceleration_window}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"VectorBT-optimized acceleration trend strength over {period} periods with {acceleration_window} acceleration window based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=acceleration_window + period * 2,
            min_lookback=acceleration_window + period * 2,
            max_lookback=acceleration_window + period * 2,
            parameters={'period': period, 'acceleration_window': acceleration_window, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.period = period
        self.acceleration_window = acceleration_window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate acceleration trend strength using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_acceleration_trend_strength_{self.period}_{self.acceleration_window}_{self.base_calculation.value}')
        
        base_values = self.base_calculator.calculate(data)
        
        # Calculate acceleration first
        shifted_values = base_values.shift(self.acceleration_window)
        momentum = safe_percentage_change(shifted_values, base_values)
        acceleration = momentum.diff(self.acceleration_window)
        
        # Calculate acceleration trend strength using VectorBT
        time_index = pd.Series(range(len(acceleration)), index=acceleration.index)
        acceleration_trend_strength = self._vectorbt_rolling_operation(
            acceleration, 'corr', window=self.period, other=time_index
        )
        
        return acceleration_trend_strength.rename(f'vectorbt_acceleration_trend_strength_{self.period}_{self.acceleration_window}_{self.base_calculation.value}')


class VectorBTAccelerationConsistencyGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized acceleration consistency generator."""
    
    def __init__(self, period: int = 10, acceleration_window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"vectorbt_acceleration_consistency_{period}_{acceleration_window}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"VectorBT-optimized acceleration consistency over {period} periods with {acceleration_window} acceleration window based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=acceleration_window + period * 2,
            min_lookback=acceleration_window + period * 2,
            max_lookback=acceleration_window + period * 2,
            parameters={'period': period, 'acceleration_window': acceleration_window, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.period = period
        self.acceleration_window = acceleration_window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate acceleration consistency using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_acceleration_consistency_{self.period}_{self.acceleration_window}_{self.base_calculation.value}')
        
        base_values = self.base_calculator.calculate(data)
        
        # Calculate acceleration first
        shifted_values = base_values.shift(self.acceleration_window)
        momentum = safe_percentage_change(shifted_values, base_values)
        acceleration = momentum.diff(self.acceleration_window)
        
        # Calculate acceleration consistency using VectorBT
        acceleration_volatility = self._vectorbt_rolling_operation(acceleration, 'std', window=self.period)
        acceleration_consistency = 1.0 / (acceleration_volatility + 1e-8)  # Add small epsilon to avoid division by zero
        
        return acceleration_consistency.rename(f'vectorbt_acceleration_consistency_{self.period}_{self.acceleration_window}_{self.base_calculation.value}')


class VectorBTAccelerationRegimeGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized acceleration regime detection generator."""
    
    def __init__(self, period: int = 10, acceleration_window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"vectorbt_acceleration_regime_{period}_{acceleration_window}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"VectorBT-optimized acceleration regime detection over {period} periods with {acceleration_window} acceleration window based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=acceleration_window + period * 2,
            min_lookback=acceleration_window + period * 2,
            max_lookback=acceleration_window + period * 2,
            parameters={'period': period, 'acceleration_window': acceleration_window, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.period = period
        self.acceleration_window = acceleration_window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate acceleration regime using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_acceleration_regime_{self.period}_{self.acceleration_window}_{self.base_calculation.value}')
        
        base_values = self.base_calculator.calculate(data)
        
        # Calculate acceleration first
        shifted_values = base_values.shift(self.acceleration_window)
        momentum = safe_percentage_change(shifted_values, base_values)
        acceleration = momentum.diff(self.acceleration_window)
        
        # Calculate acceleration regime using VectorBT
        acceleration_momentum = self._vectorbt_rolling_operation(acceleration, 'mean', window=self.period)
        acceleration_volatility = self._vectorbt_rolling_operation(acceleration, 'std', window=self.period)
        
        # Regime classification: 1 for accelerating, -1 for decelerating, 0 for neutral
        regime = np.where(acceleration_momentum > acceleration_volatility, 1, 
                         np.where(acceleration_momentum < -acceleration_volatility, -1, 0))
        
        return pd.Series(regime, index=data.index, name=f'vectorbt_acceleration_regime_{self.period}_{self.acceleration_window}_{self.base_calculation.value}')


class VectorBTMultiTimeframeAccelerationGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized multi-timeframe acceleration generator."""
    
    def __init__(self, short_period: int = 5, long_period: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"vectorbt_multi_timeframe_acceleration_{short_period}_{long_period}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"VectorBT-optimized multi-timeframe acceleration with {short_period} and {long_period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=long_period * 2,
            min_lookback=long_period * 2,
            max_lookback=long_period * 2,
            parameters={'short_period': short_period, 'long_period': long_period, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.short_period = short_period
        self.long_period = long_period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate multi-timeframe acceleration using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_multi_timeframe_acceleration_{self.short_period}_{self.long_period}_{self.base_calculation.value}')
        
        base_values = self.base_calculator.calculate(data)
        
        # Calculate short-term acceleration
        shifted_values_short = base_values.shift(self.short_period)
        momentum_short = safe_percentage_change(shifted_values_short, base_values)
        acceleration_short = momentum_short.diff(self.short_period)
        
        # Calculate long-term acceleration
        shifted_values_long = base_values.shift(self.long_period)
        momentum_long = safe_percentage_change(shifted_values_long, base_values)
        acceleration_long = momentum_long.diff(self.long_period)
        
        # Calculate multi-timeframe acceleration ratio
        multi_timeframe_acceleration = safe_divide(acceleration_short, acceleration_long)
        
        return multi_timeframe_acceleration.rename(f'vectorbt_multi_timeframe_acceleration_{self.short_period}_{self.long_period}_{self.base_calculation.value}')


class VectorBTAccelerationCorrelationGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized acceleration correlation generator."""
    
    def __init__(self, period: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"vectorbt_acceleration_correlation_{period}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"VectorBT-optimized acceleration correlation over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=period * 2,
            min_lookback=period * 2,
            max_lookback=period * 2,
            parameters={'period': period, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate acceleration correlation using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_acceleration_correlation_{self.period}_{self.base_calculation.value}')
        
        base_values = self.base_calculator.calculate(data)
        
        # Calculate acceleration
        shifted_values = base_values.shift(self.period)
        momentum = safe_percentage_change(shifted_values, base_values)
        acceleration = momentum.diff(self.period)
        
        # Calculate acceleration correlation with price
        acceleration_correlation = self._vectorbt_rolling_operation(
            acceleration, 'corr', window=self.period, other=base_values
        )
        
        return acceleration_correlation.rename(f'vectorbt_acceleration_correlation_{self.period}_{self.base_calculation.value}')


class VectorBTAccelerationDivergenceGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized acceleration divergence generator."""
    
    def __init__(self, period: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"vectorbt_acceleration_divergence_{period}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"VectorBT-optimized acceleration divergence over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=period * 2,
            min_lookback=period * 2,
            max_lookback=period * 2,
            parameters={'period': period, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate acceleration divergence using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_acceleration_divergence_{self.period}_{self.base_calculation.value}')
        
        base_values = self.base_calculator.calculate(data)
        
        # Calculate acceleration
        shifted_values = base_values.shift(self.period)
        momentum = safe_percentage_change(shifted_values, base_values)
        acceleration = momentum.diff(self.period)
        
        # Calculate acceleration divergence (difference between acceleration and price trend)
        price_trend = self._vectorbt_rolling_operation(base_values, 'mean', window=self.period)
        acceleration_divergence = acceleration - price_trend.pct_change()
        
        return acceleration_divergence.rename(f'vectorbt_acceleration_divergence_{self.period}_{self.base_calculation.value}')


class VectorBTAccelerationBatchProcessor(VectorBTFeatureGenerator):
    """Enhanced batch processor for acceleration features using UnifiedVectorizationManager."""
    
    def __init__(self, enable_optimization: bool = True, **kwargs):
        config = FeatureConfig(
            name="vectorbt_acceleration_batch_processor",
            category=FeatureCategory.ACCELERATION,
            description="Batch processor for acceleration features using UnifiedVectorizationManager",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=100,
            min_lookback=50,
            max_lookback=200,
            parameters=kwargs,
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.enable_optimization = enable_optimization and VECTORBT_OPTIMIZATION_AVAILABLE
        
        # Initialize optimization components
        if self.enable_optimization:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=True, enable_parallel=True)
            self.unified_manager = get_unified_vectorization_manager()
    
    def generate_batch_acceleration_features(self, data: pd.DataFrame, 
                                           feature_configs: List[Dict[str, Any]],
                                           chunk_size: int = 5000, 
                                           memory_limit_mb: float = 1024.0) -> pd.DataFrame:
        """Generate multiple acceleration features in batch with memory optimization and chunked processing."""
        if data.empty:
            return pd.DataFrame(index=data.index)
        
        # Estimate memory requirements
        data_memory_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
        estimated_memory_mb = data_memory_mb * len(feature_configs) * 2  # Rough estimate
        
        # Use chunked processing for large datasets
        if len(data) > chunk_size or estimated_memory_mb > memory_limit_mb:
            logger.info(f"🔄 Using chunked processing: {len(data)} rows, {estimated_memory_mb:.1f}MB estimated")
            return self._chunked_batch_processing(data, feature_configs, chunk_size)
        
        if self.enable_optimization:
            try:
                # Use UnifiedVectorizationManager for intelligent optimization
                operation_config = OperationConfig(
                    operation_type=OperationType.FEATURE_ENGINEERING,
                    data_size=len(data),
                    data_dimensions=data.shape,
                    memory_budget_mb=memory_limit_mb
                )
                
                # Prepare data for batch processing
                batch_data = {
                    'data': data,
                    'feature_configs': feature_configs,
                    'operation_type': 'acceleration_features'
                }
                
                # Execute with UnifiedVectorizationManager
                result = self.unified_manager.optimize_operation(
                    OperationType.FEATURE_ENGINEERING,
                    batch_data,
                    operation_config
                )
                
                return result.result
                
            except Exception as e:
                logger.warning(f"UnifiedVectorizationManager batch processing failed: {e}, using fallback")
                return self._fallback_batch_processing(data, feature_configs)
        else:
            return self._fallback_batch_processing(data, feature_configs)
    
    def _fallback_batch_processing(self, data: pd.DataFrame, 
                                 feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Fallback batch processing using individual generators."""
        results = {}
        
        for config in feature_configs:
            feature_type = config.get('type', 'momentum')
            period = config.get('period', 10)
            base_calculation = config.get('base_calculation', BaseCalculationType.PRICE_RETURNS)
            
            try:
                if feature_type == 'momentum':
                    generator = VectorBTMomentumGenerator(period=period, base_calculation=base_calculation)
                elif feature_type == 'acceleration':
                    generator = VectorBTPriceAccelerationGenerator(period=period, base_calculation=base_calculation)
                elif feature_type == 'jerk':
                    generator = VectorBTPriceJerkGenerator(period=period, base_calculation=base_calculation)
                else:
                    continue
                
                feature_result = generator.generate(data)
                results[feature_result.name] = feature_result
                
            except Exception as e:
                logger.warning(f"Failed to generate {feature_type} feature: {e}")
                continue
        
        return pd.DataFrame(results, index=data.index)
    
    def _chunked_batch_processing(self, data: pd.DataFrame, 
                                feature_configs: List[Dict[str, Any]], 
                                chunk_size: int) -> pd.DataFrame:
        """Process large datasets in chunks for memory efficiency."""
        logger.info(f"🔄 Processing {len(data)} rows in chunks of {chunk_size}")
        
        # Split data into overlapping chunks
        chunks = []
        for i in range(0, len(data), chunk_size):
            end_idx = min(i + chunk_size, len(data))
            chunk = data.iloc[i:end_idx]
            chunks.append(chunk)
        
        # Process each chunk
        chunk_results = []
        for i, chunk in enumerate(chunks):
            logger.info(f"🔄 Processing chunk {i+1}/{len(chunks)} ({len(chunk)} rows)")
            
            try:
                # Process chunk with fallback method
                chunk_result = self._fallback_batch_processing(chunk, feature_configs)
                chunk_results.append(chunk_result)
                
                # Memory cleanup
                import gc
                gc.collect()
                
            except Exception as e:
                logger.warning(f"Chunk {i+1} processing failed: {e}")
                # Create empty result for failed chunk
                empty_result = pd.DataFrame(index=chunk.index)
                for config in feature_configs:
                    feature_name = f"{config['type']}_{config.get('period', 10)}"
                    empty_result[feature_name] = np.nan
                chunk_results.append(empty_result)
        
        # Combine chunk results
        if chunk_results:
            combined_result = pd.concat(chunk_results, ignore_index=False)
            logger.info(f"✅ Chunked processing completed: {len(combined_result.columns)} features")
            return combined_result
        else:
            return pd.DataFrame(index=data.index)


def create_vectorbt_acceleration_generators() -> List[VectorBTFeatureGenerator]:
    """Create all VectorBT-optimized acceleration feature generators."""
    generators = []
    
    # Basic acceleration features with enhanced optimization
    for period in [5, 10, 20, 50]:
        generators.append(VectorBTMomentumGenerator(period=period, enable_optimization=True))
    
    for period in [5, 10]:
        generators.extend([
            VectorBTPriceAccelerationGenerator(period=period, enable_optimization=True),
            VectorBTPriceJerkGenerator(period=period, enable_optimization=True),
        ])
    
    for window in [5, 10, 20, 50]:
        generators.extend([
            VectorBTTrendStrengthGenerator(window=window),
            VectorBTTrendConsistencyGenerator(window=window),
        ])
    
    # Volume and volatility acceleration
    generators.extend([
        VectorBTVolumeAccelerationGenerator(period=5),
        VectorBTVolatilityAccelerationGenerator(period=5, volatility_window=20),
    ])
    
    # Advanced acceleration features
    for period in [5, 10]:
        for window in [10, 20]:
            generators.extend([
                VectorBTMomentumAccelerationGenerator(period=period, momentum_window=window),
                VectorBTAccelerationMomentumGenerator(period=period, acceleration_window=window),
                VectorBTAccelerationVolatilityGenerator(period=period, acceleration_window=window),
                VectorBTAccelerationTrendStrengthGenerator(period=period, acceleration_window=window),
                VectorBTAccelerationConsistencyGenerator(period=period, acceleration_window=window),
                VectorBTAccelerationRegimeGenerator(period=period, acceleration_window=window),
            ])
    
    # Multi-timeframe and correlation features
    generators.extend([
        VectorBTMultiTimeframeAccelerationGenerator(short_period=5, long_period=20),
        VectorBTAccelerationCorrelationGenerator(period=20),
        VectorBTAccelerationDivergenceGenerator(period=20),
    ])
    
    # Add batch processor
    generators.append(VectorBTAccelerationBatchProcessor(enable_optimization=True))
    
    return generators


class VectorBTAccelerationPerformanceMonitor:
    """Performance monitoring and statistics for acceleration feature generation."""
    
    def __init__(self):
        self.performance_stats = {
            'total_features_generated': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'gpu_accelerations': 0,
            'parallel_operations': 0,
            'memory_optimizations': 0,
            'batch_operations': 0,
            'total_generation_time': 0.0,
            'average_generation_time': 0.0,
            'feature_types': {},
            'optimization_strategies': {}
        }
        self.logger = logging.getLogger(__name__)
    
    def record_feature_generation(self, feature_type: str, generation_time: float, 
                                optimization_strategy: str = 'standard', 
                                vectorbt_used: bool = False, gpu_used: bool = False,
                                parallel_used: bool = False, memory_optimized: bool = False):
        """Record feature generation statistics."""
        self.performance_stats['total_features_generated'] += 1
        self.performance_stats['total_generation_time'] += generation_time
        
        # Update average generation time
        total_features = self.performance_stats['total_features_generated']
        self.performance_stats['average_generation_time'] = (
            self.performance_stats['total_generation_time'] / total_features
        )
        
        # Track feature types
        if feature_type not in self.performance_stats['feature_types']:
            self.performance_stats['feature_types'][feature_type] = 0
        self.performance_stats['feature_types'][feature_type] += 1
        
        # Track optimization strategies
        if optimization_strategy not in self.performance_stats['optimization_strategies']:
            self.performance_stats['optimization_strategies'][optimization_strategy] = 0
        self.performance_stats['optimization_strategies'][optimization_strategy] += 1
        
        # Track specific optimizations
        if vectorbt_used:
            self.performance_stats['vectorbt_operations'] += 1
        else:
            self.performance_stats['pandas_fallbacks'] += 1
        
        if gpu_used:
            self.performance_stats['gpu_accelerations'] += 1
        
        if parallel_used:
            self.performance_stats['parallel_operations'] += 1
        
        if memory_optimized:
            self.performance_stats['memory_optimizations'] += 1
    
    def record_batch_operation(self, num_features: int, batch_time: float, 
                             optimization_strategy: str = 'batch'):
        """Record batch operation statistics."""
        self.performance_stats['batch_operations'] += 1
        self.performance_stats['total_features_generated'] += num_features
        self.performance_stats['total_generation_time'] += batch_time
        
        # Update average generation time
        total_features = self.performance_stats['total_features_generated']
        self.performance_stats['average_generation_time'] = (
            self.performance_stats['total_generation_time'] / total_features
        )
        
        # Track optimization strategy
        if optimization_strategy not in self.performance_stats['optimization_strategies']:
            self.performance_stats['optimization_strategies'][optimization_strategy] = 0
        self.performance_stats['optimization_strategies'][optimization_strategy] += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        stats = self.performance_stats.copy()
        
        # Calculate efficiency metrics
        if stats['total_features_generated'] > 0:
            stats['vectorbt_usage_rate'] = stats['vectorbt_operations'] / stats['total_features_generated']
            stats['gpu_usage_rate'] = stats['gpu_accelerations'] / stats['total_features_generated']
            stats['parallel_usage_rate'] = stats['parallel_operations'] / stats['total_features_generated']
            stats['memory_optimization_rate'] = stats['memory_optimizations'] / stats['total_features_generated']
            stats['batch_usage_rate'] = stats['batch_operations'] / stats['total_features_generated']
        else:
            stats.update({
                'vectorbt_usage_rate': 0.0,
                'gpu_usage_rate': 0.0,
                'parallel_usage_rate': 0.0,
                'memory_optimization_rate': 0.0,
                'batch_usage_rate': 0.0
            })
        
        # Calculate throughput
        if stats['total_generation_time'] > 0:
            stats['features_per_second'] = stats['total_features_generated'] / stats['total_generation_time']
        else:
            stats['features_per_second'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Reset all performance statistics."""
        self.performance_stats = {
            'total_features_generated': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'gpu_accelerations': 0,
            'parallel_operations': 0,
            'memory_optimizations': 0,
            'batch_operations': 0,
            'total_generation_time': 0.0,
            'average_generation_time': 0.0,
            'feature_types': {},
            'optimization_strategies': {}
        }
    
    def log_performance_summary(self):
        """Log performance summary to logger."""
        summary = self.get_performance_summary()
        
        self.logger.info("📊 VectorBT Acceleration Feature Performance Summary:")
        self.logger.info(f"  Total features generated: {summary['total_features_generated']}")
        self.logger.info(f"  Average generation time: {summary['average_generation_time']:.4f}s")
        self.logger.info(f"  Features per second: {summary['features_per_second']:.2f}")
        self.logger.info(f"  VectorBT usage rate: {summary['vectorbt_usage_rate']:.2%}")
        self.logger.info(f"  GPU usage rate: {summary['gpu_usage_rate']:.2%}")
        self.logger.info(f"  Parallel usage rate: {summary['parallel_usage_rate']:.2%}")
        self.logger.info(f"  Memory optimization rate: {summary['memory_optimization_rate']:.2%}")
        self.logger.info(f"  Batch usage rate: {summary['batch_usage_rate']:.2%}")
        
        if summary['feature_types']:
            self.logger.info("  Feature types generated:")
            for feature_type, count in summary['feature_types'].items():
                self.logger.info(f"    {feature_type}: {count}")
        
        if summary['optimization_strategies']:
            self.logger.info("  Optimization strategies used:")
            for strategy, count in summary['optimization_strategies'].items():
                self.logger.info(f"    {strategy}: {count}")


# Global performance monitor instance
_performance_monitor = None

def get_acceleration_performance_monitor() -> VectorBTAccelerationPerformanceMonitor:
    """Get global acceleration performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = VectorBTAccelerationPerformanceMonitor()
    return _performance_monitor


def create_default_vectorbt_acceleration_generators() -> List[VectorBTFeatureGenerator]:
    """Create default VectorBT-optimized acceleration feature generators."""
    return create_vectorbt_acceleration_generators()


# Export all generators
__all__ = [
    'VectorBTMomentumGenerator',
    'VectorBTPriceAccelerationGenerator',
    'VectorBTPriceJerkGenerator',
    'VectorBTTrendStrengthGenerator',
    'VectorBTTrendConsistencyGenerator',
    'VectorBTVolumeAccelerationGenerator',
    'VectorBTVolatilityAccelerationGenerator',
    'VectorBTMomentumAccelerationGenerator',
    'VectorBTAccelerationMomentumGenerator',
    'VectorBTAccelerationVolatilityGenerator',
    'VectorBTAccelerationTrendStrengthGenerator',
    'VectorBTAccelerationConsistencyGenerator',
    'VectorBTAccelerationRegimeGenerator',
    'VectorBTMultiTimeframeAccelerationGenerator',
    'VectorBTAccelerationCorrelationGenerator',
    'VectorBTAccelerationDivergenceGenerator',
    'VectorBTAccelerationBatchProcessor',
    'VectorBTAccelerationPerformanceMonitor',
    'get_acceleration_performance_monitor',
    'create_vectorbt_acceleration_generators',
    'create_default_vectorbt_acceleration_generators'
]