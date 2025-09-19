"""
Acceleration Feature Generators

This module provides feature generators for acceleration, velocity, and jerk indicators,
including momentum derivatives, trend strength, and consistency measures.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from scipy import stats

from ..core.feature_generator import (
    FeatureGenerator, 
    FeatureConfig, 
    FeatureCategory,
    VectorizedFeatureGenerator
)
from ..base_calculations import (
    BaseCalculationType,
    create_base_calculator
)

class AccelerationFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for acceleration-based features."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True)
    
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
    """Generator for price momentum features."""
    
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
        super().__init__(config)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate momentum."""
        base_values = self.base_calculator.calculate(data)
        momentum = base_values.pct_change(self.period)
        return momentum

# Price Acceleration Generator
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
        super().__init__(config)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate acceleration (second derivative of price)."""
        base_values = self.base_calculator.calculate(data)
        momentum = base_values.pct_change(self.period)
        acceleration = momentum.diff(self.period)
        return acceleration

# Price Jerk Generator
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
        super().__init__(config)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate jerk (third derivative of price)."""
        base_values = self.base_calculator.calculate(data)
        momentum = base_values.pct_change(self.period)
        acceleration = momentum.diff(self.period)
        jerk = acceleration.diff(self.period)
        return jerk

# Trend Strength Generator
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
        super().__init__(config)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate trend strength using polyfit."""
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
        
        trend_strength = base_values.rolling(window=self.window).apply(calculate_trend_strength, raw=False)
        return trend_strength

# Trend Consistency Generator
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
        super().__init__(config)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate trend consistency (positive slope indicator)."""
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
        
        trend_consistency = base_values.rolling(window=self.window).apply(calculate_trend_consistency, raw=False)
        return trend_consistency

# Volume Acceleration Generator
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
        super().__init__(config)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate volume acceleration."""
        base_values = self.base_calculator.calculate(data)
        volume_acceleration = base_values.diff(self.period).diff(self.period)
        return volume_acceleration

# Volatility Acceleration Generator
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
        super().__init__(config)
        self.period = period
        self.volatility_window = volatility_window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate volatility acceleration."""
        base_values = self.base_calculator.calculate(data)
        volatility = base_values.rolling(window=self.volatility_window).std()
        volatility_acceleration = volatility.diff(self.period).diff(self.period)
        return volatility_acceleration

def create_acceleration_generators() -> List[FeatureGenerator]:
    """Create all acceleration-based feature generators."""
    generators = []
    
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
    'create_acceleration_generators'
]