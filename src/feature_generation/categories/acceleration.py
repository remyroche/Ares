"""
Acceleration Feature Generators - VectorBT Optimized

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

logger = logging.getLogger(__name__)

class MomentumGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized momentum generator."""
    
    def __init__(self, period: int = 10, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"momentum_{period}_{base_calculation.value}",
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
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate momentum using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'momentum_{self.period}_{self.base_calculation.value}')
        
        base_values = self.base_calculator.calculate(data)
        
        # Calculate momentum using VectorBT rolling operations
        shifted_values = base_values.shift(self.period)
        momentum = safe_percentage_change(shifted_values, base_values)
        
        # Validate finite values
        try:
            validate_finite(momentum.values, f"VectorBT_Momentum_{self.period}_{self.base_calculation.value}")
        except ValueError as e:
            logger.warning(f"⚠️ {e}")
        
        return momentum.rename(f'momentum_{self.period}_{self.base_calculation.value}')


class PriceAccelerationGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized price acceleration generator."""
    
    def __init__(self, period: int = 10, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"acceleration_{period}_{base_calculation.value}",
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
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate acceleration using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'acceleration_{self.period}_{self.base_calculation.value}')
        
        base_values = self.base_calculator.calculate(data)
        
        # Calculate momentum first
        shifted_values = base_values.shift(self.period)
        momentum = safe_percentage_change(shifted_values, base_values)
        
        # Calculate acceleration (second derivative) using VectorBT
        acceleration = momentum.diff(self.period)
        
        # Validate finite values
        try:
            validate_finite(acceleration.values, f"VectorBT_Acceleration_{self.period}_{self.base_calculation.value}")
        except ValueError as e:
            logger.warning(f"⚠️ {e}")
        
        return acceleration.rename(f'acceleration_{self.period}_{self.base_calculation.value}')


class PriceJerkGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized price jerk generator."""
    
    def __init__(self, period: int = 10, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"jerk_{period}_{base_calculation.value}",
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
            return pd.Series(dtype=float, index=data.index, name=f'jerk_{self.period}_{self.base_calculation.value}')
        
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
        
        return jerk.rename(f'jerk_{self.period}_{self.base_calculation.value}')


class TrendStrengthGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized trend strength generator."""
    
    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"trend_strength_{window}_{base_calculation.value}",
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
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate trend strength using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'trend_strength_{self.window}_{self.base_calculation.value}')
        
        base_values = self.base_calculator.calculate(data)
        
        # Calculate trend strength using rolling correlation with time
        time_index = pd.Series(range(len(base_values)), index=base_values.index)
        trend_strength = self._vectorbt_rolling_operation(
            base_values, 'corr', window=self.window, other=time_index
        )
        
        return trend_strength.rename(f'trend_strength_{self.window}_{self.base_calculation.value}')


class TrendConsistencyGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized trend consistency generator."""
    
    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"trend_consistency_{window}_{base_calculation.value}",
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
            return pd.Series(dtype=float, index=data.index, name=f'trend_consistency_{self.window}_{self.base_calculation.value}')
        
        base_values = self.base_calculator.calculate(data)
        
        # Calculate trend consistency as inverse of volatility
        volatility = self._vectorbt_rolling_operation(base_values, 'std', window=self.window)
        consistency = 1.0 / (volatility + 1e-8)  # Add small epsilon to avoid division by zero
        
        return consistency.rename(f'trend_consistency_{self.window}_{self.base_calculation.value}')


class VolumeAccelerationGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized volume acceleration generator."""
    
    def __init__(self, period: int = 5, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.VOLUME_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"volume_acceleration_{period}_{base_calculation.value}",
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
            return pd.Series(dtype=float, index=data.index, name=f'volume_acceleration_{self.period}_{self.base_calculation.value}')
        
        base_values = self.base_calculator.calculate(data)
        
        # Calculate volume acceleration using VectorBT
        volume_acceleration = base_values.diff(self.period).diff(self.period)
        
        return volume_acceleration.rename(f'volume_acceleration_{self.period}_{self.base_calculation.value}')


class VolatilityAccelerationGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized volatility acceleration generator."""
    
    def __init__(self, period: int = 5, volatility_window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"volatility_acceleration_{period}_{volatility_window}_{base_calculation.value}",
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
            return pd.Series(dtype=float, index=data.index, name=f'volatility_acceleration_{self.period}_{self.volatility_window}_{self.base_calculation.value}')
        
        base_values = self.base_calculator.calculate(data)
        
        # Calculate volatility using VectorBT
        volatility = self._vectorbt_rolling_operation(base_values, 'std', window=self.volatility_window)
        
        # Calculate volatility acceleration using VectorBT
        volatility_acceleration = volatility.diff(self.period).diff(self.period)
        
        return volatility_acceleration.rename(f'volatility_acceleration_{self.period}_{self.volatility_window}_{self.base_calculation.value}')


class MomentumAccelerationGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized momentum acceleration generator."""
    
    def __init__(self, period: int = 10, momentum_window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"momentum_acceleration_{period}_{momentum_window}_{base_calculation.value}",
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
            return pd.Series(dtype=float, index=data.index, name=f'momentum_acceleration_{self.period}_{self.momentum_window}_{self.base_calculation.value}')
        
        base_values = self.base_calculator.calculate(data)
        
        # Calculate momentum using VectorBT
        momentum = self._vectorbt_rolling_operation(base_values, 'mean', window=self.momentum_window)
        
        # Calculate momentum acceleration using VectorBT
        momentum_acceleration = momentum.diff(self.period).diff(self.period)
        
        return momentum_acceleration.rename(f'momentum_acceleration_{self.period}_{self.momentum_window}_{self.base_calculation.value}')


class AccelerationMomentumGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized acceleration momentum generator."""
    
    def __init__(self, period: int = 10, acceleration_window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"acceleration_momentum_{period}_{acceleration_window}_{base_calculation.value}",
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
            return pd.Series(dtype=float, index=data.index, name=f'acceleration_momentum_{self.period}_{self.acceleration_window}_{self.base_calculation.value}')
        
        base_values = self.base_calculator.calculate(data)
        
        # Calculate acceleration first
        shifted_values = base_values.shift(self.acceleration_window)
        momentum = safe_percentage_change(shifted_values, base_values)
        acceleration = momentum.diff(self.acceleration_window)
        
        # Calculate acceleration momentum using VectorBT
        acceleration_momentum = self._vectorbt_rolling_operation(acceleration, 'mean', window=self.period)
        
        return acceleration_momentum.rename(f'acceleration_momentum_{self.period}_{self.acceleration_window}_{self.base_calculation.value}')


class AccelerationVolatilityGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized acceleration volatility generator."""
    
    def __init__(self, period: int = 10, acceleration_window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"acceleration_volatility_{period}_{acceleration_window}_{base_calculation.value}",
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
            return pd.Series(dtype=float, index=data.index, name=f'acceleration_volatility_{self.period}_{self.acceleration_window}_{self.base_calculation.value}')
        
        base_values = self.base_calculator.calculate(data)
        
        # Calculate acceleration first
        shifted_values = base_values.shift(self.acceleration_window)
        momentum = safe_percentage_change(shifted_values, base_values)
        acceleration = momentum.diff(self.acceleration_window)
        
        # Calculate acceleration volatility using VectorBT
        acceleration_volatility = self._vectorbt_rolling_operation(acceleration, 'std', window=self.period)
        
        return acceleration_volatility.rename(f'acceleration_volatility_{self.period}_{self.acceleration_window}_{self.base_calculation.value}')


class AccelerationTrendStrengthGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized acceleration trend strength generator."""
    
    def __init__(self, period: int = 10, acceleration_window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"acceleration_trend_strength_{period}_{acceleration_window}_{base_calculation.value}",
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
            return pd.Series(dtype=float, index=data.index, name=f'acceleration_trend_strength_{self.period}_{self.acceleration_window}_{self.base_calculation.value}')
        
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
        
        return acceleration_trend_strength.rename(f'acceleration_trend_strength_{self.period}_{self.acceleration_window}_{self.base_calculation.value}')


class AccelerationConsistencyGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized acceleration consistency generator."""
    
    def __init__(self, period: int = 10, acceleration_window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"acceleration_consistency_{period}_{acceleration_window}_{base_calculation.value}",
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
            return pd.Series(dtype=float, index=data.index, name=f'acceleration_consistency_{self.period}_{self.acceleration_window}_{self.base_calculation.value}')
        
        base_values = self.base_calculator.calculate(data)
        
        # Calculate acceleration first
        shifted_values = base_values.shift(self.acceleration_window)
        momentum = safe_percentage_change(shifted_values, base_values)
        acceleration = momentum.diff(self.acceleration_window)
        
        # Calculate acceleration consistency using VectorBT
        acceleration_volatility = self._vectorbt_rolling_operation(acceleration, 'std', window=self.period)
        acceleration_consistency = 1.0 / (acceleration_volatility + 1e-8)  # Add small epsilon to avoid division by zero
        
        return acceleration_consistency.rename(f'acceleration_consistency_{self.period}_{self.acceleration_window}_{self.base_calculation.value}')


class AccelerationRegimeGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized acceleration regime detection generator."""
    
    def __init__(self, period: int = 10, acceleration_window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"acceleration_regime_{period}_{acceleration_window}_{base_calculation.value}",
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
            return pd.Series(dtype=float, index=data.index, name=f'acceleration_regime_{self.period}_{self.acceleration_window}_{self.base_calculation.value}')
        
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
        
        return pd.Series(regime, index=data.index, name=f'acceleration_regime_{self.period}_{self.acceleration_window}_{self.base_calculation.value}')


class MultiTimeframeAccelerationGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized multi-timeframe acceleration generator."""
    
    def __init__(self, short_period: int = 5, long_period: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"multi_timeframe_acceleration_{short_period}_{long_period}_{base_calculation.value}",
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
            return pd.Series(dtype=float, index=data.index, name=f'multi_timeframe_acceleration_{self.short_period}_{self.long_period}_{self.base_calculation.value}')
        
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
        
        return multi_timeframe_acceleration.rename(f'multi_timeframe_acceleration_{self.short_period}_{self.long_period}_{self.base_calculation.value}')


class AccelerationCorrelationGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized acceleration correlation generator."""
    
    def __init__(self, period: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"acceleration_correlation_{period}_{base_calculation.value}",
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
            return pd.Series(dtype=float, index=data.index, name=f'acceleration_correlation_{self.period}_{self.base_calculation.value}')
        
        base_values = self.base_calculator.calculate(data)
        
        # Calculate acceleration
        shifted_values = base_values.shift(self.period)
        momentum = safe_percentage_change(shifted_values, base_values)
        acceleration = momentum.diff(self.period)
        
        # Calculate acceleration correlation with price
        acceleration_correlation = self._vectorbt_rolling_operation(
            acceleration, 'corr', window=self.period, other=base_values
        )
        
        return acceleration_correlation.rename(f'acceleration_correlation_{self.period}_{self.base_calculation.value}')


class AccelerationDivergenceGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized acceleration divergence generator."""
    
    def __init__(self, period: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"acceleration_divergence_{period}_{base_calculation.value}",
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
            return pd.Series(dtype=float, index=data.index, name=f'acceleration_divergence_{self.period}_{self.base_calculation.value}')
        
        base_values = self.base_calculator.calculate(data)
        
        # Calculate acceleration
        shifted_values = base_values.shift(self.period)
        momentum = safe_percentage_change(shifted_values, base_values)
        acceleration = momentum.diff(self.period)
        
        # Calculate acceleration divergence (difference between acceleration and price trend)
        price_trend = self._vectorbt_rolling_operation(base_values, 'mean', window=self.period)
        acceleration_divergence = acceleration - price_trend.pct_change()
        
        return acceleration_divergence.rename(f'acceleration_divergence_{self.period}_{self.base_calculation.value}')


def create_acceleration_generators() -> List[VectorBTFeatureGenerator]:
    """Create all VectorBT-optimized acceleration feature generators."""
    generators = []
    
    # Basic acceleration features
    for period in [5, 10, 20, 50]:
        generators.append(MomentumGenerator(period=period))
    
    for period in [5, 10]:
        generators.extend([
            PriceAccelerationGenerator(period=period),
            PriceJerkGenerator(period=period),
        ])
    
    for window in [5, 10, 20, 50]:
        generators.extend([
            TrendStrengthGenerator(window=window),
            TrendConsistencyGenerator(window=window),
        ])
    
    # Volume and volatility acceleration
    generators.extend([
        VolumeAccelerationGenerator(period=5),
        VolatilityAccelerationGenerator(period=5, volatility_window=20),
    ])
    
    # Advanced acceleration features
    for period in [5, 10]:
        for window in [10, 20]:
            generators.extend([
                MomentumAccelerationGenerator(period=period, momentum_window=window),
                AccelerationMomentumGenerator(period=period, acceleration_window=window),
                AccelerationVolatilityGenerator(period=period, acceleration_window=window),
                AccelerationTrendStrengthGenerator(period=period, acceleration_window=window),
                AccelerationConsistencyGenerator(period=period, acceleration_window=window),
                AccelerationRegimeGenerator(period=period, acceleration_window=window),
            ])
    
    # Multi-timeframe and correlation features
    generators.extend([
        MultiTimeframeAccelerationGenerator(short_period=5, long_period=20),
        AccelerationCorrelationGenerator(period=20),
        AccelerationDivergenceGenerator(period=20),
    ])
    
    return generators


def create_default_acceleration_generators() -> List[VectorBTFeatureGenerator]:
    """Create default VectorBT-optimized acceleration feature generators."""
    return create_acceleration_generators()


# Export all generators
__all__ = [
    'MomentumGenerator',
    'PriceAccelerationGenerator',
    'PriceJerkGenerator',
    'TrendStrengthGenerator',
    'TrendConsistencyGenerator',
    'VolumeAccelerationGenerator',
    'VolatilityAccelerationGenerator',
    'MomentumAccelerationGenerator',
    'AccelerationMomentumGenerator',
    'AccelerationVolatilityGenerator',
    'AccelerationTrendStrengthGenerator',
    'AccelerationConsistencyGenerator',
    'AccelerationRegimeGenerator',
    'MultiTimeframeAccelerationGenerator',
    'AccelerationCorrelationGenerator',
    'AccelerationDivergenceGenerator',
    'create_acceleration_generators',
    'create_default_acceleration_generators'
]