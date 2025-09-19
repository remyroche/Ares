"""
Feature Interaction Generators

This module provides feature generators for feature interactions, combinations,
and derived features that capture relationships between different indicators.
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

class InteractionFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for interaction-based features."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True)
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="interaction_features",
            category=FeatureCategory.INTERACTION,
            description="Comprehensive interaction features between different indicators",
            required_columns=["close", "volume"],
            optional_columns=["high", "low", "open"],
            default_lookback=20,
            min_lookback=5,
            max_lookback=100,
            parameters={
                "interaction_types": ["momentum_divergence", "momentum_volume", "momentum_volatility", "volatility_volume"]
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )

# Momentum Divergence Generator
class MomentumDivergenceGenerator(FeatureGenerator):
    """Generator for momentum divergence between price and volume."""
    
    def __init__(self, period: int = 5, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns() + ["volume"]
        
        config = FeatureConfig(
            name=f"momentum_divergence_{period}_{base_calculation.value}",
            category=FeatureCategory.INTERACTION,
            description=f"Momentum divergence between price and volume over {period} periods",
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
        """Generate momentum divergence."""
        base_values = self.base_calculator.calculate(data)
        price_momentum = base_values.pct_change(self.period)
        volume_momentum = data['volume'].pct_change(self.period)
        divergence = price_momentum - volume_momentum
        return divergence

# Momentum Volume Generator
class MomentumVolumeGenerator(FeatureGenerator):
    """Generator for momentum-volume interaction."""
    
    def __init__(self, period: int = 5, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns() + ["volume"]
        
        config = FeatureConfig(
            name=f"momentum_volume_{period}_{base_calculation.value}",
            category=FeatureCategory.INTERACTION,
            description=f"Momentum-volume interaction over {period} periods",
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
        """Generate momentum-volume interaction."""
        base_values = self.base_calculator.calculate(data)
        price_momentum = base_values.pct_change(self.period)
        volume_momentum = data['volume'].pct_change(self.period)
        interaction = price_momentum * volume_momentum
        return interaction

# Momentum Volatility Generator
class MomentumVolatilityGenerator(FeatureGenerator):
    """Generator for momentum-volatility interaction."""
    
    def __init__(self, period: int = 5, volatility_window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"momentum_volatility_{period}_{volatility_window}_{base_calculation.value}",
            category=FeatureCategory.INTERACTION,
            description=f"Momentum-volatility interaction over {period} periods with {volatility_window} volatility window",
            required_columns=required_columns,
            default_lookback=max(period, volatility_window),
            min_lookback=max(period, volatility_window),
            max_lookback=max(period, volatility_window),
            parameters={'period': period, 'volatility_window': volatility_window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config)
        self.period = period
        self.volatility_window = volatility_window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate momentum-volatility interaction."""
        base_values = self.base_calculator.calculate(data)
        price_momentum = base_values.pct_change(self.period)
        volatility = base_values.rolling(window=self.volatility_window).std()
        # Normalize momentum by volatility
        interaction = price_momentum / (volatility + 1e-8)  # Add small epsilon to prevent division by zero
        return interaction

# Momentum Trend Generator
class MomentumTrendGenerator(FeatureGenerator):
    """Generator for momentum-trend interaction."""
    
    def __init__(self, momentum_period: int = 5, trend_window: int = 10, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"momentum_trend_{momentum_period}_{trend_window}_{base_calculation.value}",
            category=FeatureCategory.INTERACTION,
            description=f"Momentum-trend interaction over {momentum_period} momentum periods with {trend_window} trend window",
            required_columns=required_columns,
            default_lookback=max(momentum_period, trend_window),
            min_lookback=max(momentum_period, trend_window),
            max_lookback=max(momentum_period, trend_window),
            parameters={'momentum_period': momentum_period, 'trend_window': trend_window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config)
        self.momentum_period = momentum_period
        self.trend_window = trend_window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate momentum-trend interaction."""
        base_values = self.base_calculator.calculate(data)
        price_momentum = base_values.pct_change(self.momentum_period)
        
        def calculate_trend_strength(series):
            if len(series) < 2:
                return 0.0
            try:
                # Calculate linear regression slope
                slope = np.polyfit(range(len(series)), series, 1)[0]
                return slope
            except:
                return 0.0
        
        trend_strength = base_values.rolling(window=self.trend_window).apply(calculate_trend_strength, raw=False)
        interaction = price_momentum * trend_strength
        return interaction

# Volatility Volume Generator
class VolatilityVolumeGenerator(FeatureGenerator):
    """Generator for volatility-volume interaction."""
    
    def __init__(self, volatility_window: int = 20, volume_window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns() + ["volume"]
        
        config = FeatureConfig(
            name=f"volatility_volume_{volatility_window}_{volume_window}_{base_calculation.value}",
            category=FeatureCategory.INTERACTION,
            description=f"Volatility-volume interaction with {volatility_window} volatility window and {volume_window} volume window",
            required_columns=required_columns,
            default_lookback=max(volatility_window, volume_window),
            min_lookback=max(volatility_window, volume_window),
            max_lookback=max(volatility_window, volume_window),
            parameters={'volatility_window': volatility_window, 'volume_window': volume_window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config)
        self.volatility_window = volatility_window
        self.volume_window = volume_window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate volatility-volume interaction."""
        base_values = self.base_calculator.calculate(data)
        volatility = base_values.rolling(window=self.volatility_window).std()
        volume_ma = data['volume'].rolling(window=self.volume_window).mean()
        interaction = volatility * volume_ma
        return interaction

# Volatility Price Generator
class VolatilityPriceGenerator(FeatureGenerator):
    """Generator for volatility-price interaction."""
    
    def __init__(self, volatility_window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"volatility_price_{volatility_window}_{base_calculation.value}",
            category=FeatureCategory.INTERACTION,
            description=f"Volatility-price interaction with {volatility_window} volatility window",
            required_columns=required_columns,
            default_lookback=volatility_window,
            min_lookback=volatility_window,
            max_lookback=volatility_window,
            parameters={'volatility_window': volatility_window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config)
        self.volatility_window = volatility_window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate volatility-price interaction."""
        base_values = self.base_calculator.calculate(data)
        volatility = base_values.rolling(window=self.volatility_window).std()
        # Use close price for interaction
        if 'close' in data.columns:
            interaction = volatility * data['close']
        else:
            interaction = volatility * base_values
        return interaction

# Volatility High-Low Generator
class VolatilityHighLowGenerator(FeatureGenerator):
    """Generator for volatility-high-low range interaction."""
    
    def __init__(self, volatility_window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns() + ["high", "low", "close"]
        
        config = FeatureConfig(
            name=f"volatility_hl_{volatility_window}_{base_calculation.value}",
            category=FeatureCategory.INTERACTION,
            description=f"Volatility-high-low range interaction with {volatility_window} volatility window",
            required_columns=required_columns,
            default_lookback=volatility_window,
            min_lookback=volatility_window,
            max_lookback=volatility_window,
            parameters={'volatility_window': volatility_window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config)
        self.volatility_window = volatility_window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate volatility-high-low range interaction."""
        base_values = self.base_calculator.calculate(data)
        volatility = base_values.rolling(window=self.volatility_window).std()
        hl_range_pct = (data['high'] - data['low']) / data['close']
        interaction = volatility * hl_range_pct
        return interaction

# Volatility Momentum Generator
class VolatilityMomentumGenerator(FeatureGenerator):
    """Generator for volatility-momentum interaction."""
    
    def __init__(self, volatility_window: int = 20, momentum_period: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"volatility_momentum_{volatility_window}_{momentum_period}_{base_calculation.value}",
            category=FeatureCategory.INTERACTION,
            description=f"Volatility-momentum interaction with {volatility_window} volatility window and {momentum_period} momentum period",
            required_columns=required_columns,
            default_lookback=max(volatility_window, momentum_period),
            min_lookback=max(volatility_window, momentum_period),
            max_lookback=max(volatility_window, momentum_period),
            parameters={'volatility_window': volatility_window, 'momentum_period': momentum_period, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config)
        self.volatility_window = volatility_window
        self.momentum_period = momentum_period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate volatility-momentum interaction."""
        base_values = self.base_calculator.calculate(data)
        volatility = base_values.rolling(window=self.volatility_window).std()
        momentum = base_values.pct_change(self.momentum_period)
        interaction = volatility * momentum
        return interaction

# Volatility Trend Generator
class VolatilityTrendGenerator(FeatureGenerator):
    """Generator for volatility-trend interaction."""
    
    def __init__(self, volatility_window: int = 20, trend_window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"volatility_trend_{volatility_window}_{trend_window}_{base_calculation.value}",
            category=FeatureCategory.INTERACTION,
            description=f"Volatility-trend interaction with {volatility_window} volatility window and {trend_window} trend window",
            required_columns=required_columns,
            default_lookback=max(volatility_window, trend_window),
            min_lookback=max(volatility_window, trend_window),
            max_lookback=max(volatility_window, trend_window),
            parameters={'volatility_window': volatility_window, 'trend_window': trend_window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config)
        self.volatility_window = volatility_window
        self.trend_window = trend_window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate volatility-trend interaction."""
        base_values = self.base_calculator.calculate(data)
        volatility = base_values.rolling(window=self.volatility_window).std()
        
        def calculate_trend_strength(series):
            if len(series) < 2:
                return 0.0
            try:
                # Calculate linear regression slope
                slope = np.polyfit(range(len(series)), series, 1)[0]
                return slope
            except:
                return 0.0
        
        trend_strength = base_values.rolling(window=self.trend_window).apply(calculate_trend_strength, raw=False)
        interaction = volatility * trend_strength
        return interaction

def create_interaction_generators() -> List[FeatureGenerator]:
    """Create all interaction feature generators."""
    generators = []
    
    # Momentum divergence
    generators.append(MomentumDivergenceGenerator(period=5))
    
    # Momentum-volume interaction
    generators.append(MomentumVolumeGenerator(period=5))
    
    # Momentum-volatility interaction
    generators.append(MomentumVolatilityGenerator(period=5, volatility_window=20))
    
    # Momentum-trend interaction
    generators.append(MomentumTrendGenerator(momentum_period=5, trend_window=10))
    
    # Volatility-volume interaction
    generators.append(VolatilityVolumeGenerator(volatility_window=20, volume_window=20))
    
    # Volatility-price interaction
    generators.append(VolatilityPriceGenerator(volatility_window=20))
    
    # Volatility-high-low interaction
    generators.append(VolatilityHighLowGenerator(volatility_window=20))
    
    # Volatility-momentum interaction
    generators.append(VolatilityMomentumGenerator(volatility_window=20, momentum_period=20))
    
    # Volatility-trend interaction
    generators.append(VolatilityTrendGenerator(volatility_window=20, trend_window=20))
    
    return generators

# Export all generators
__all__ = [
    'InteractionFeatureGenerator',
    'MomentumDivergenceGenerator',
    'MomentumVolumeGenerator',
    'MomentumVolatilityGenerator',
    'MomentumTrendGenerator',
    'VolatilityVolumeGenerator',
    'VolatilityPriceGenerator',
    'VolatilityHighLowGenerator',
    'VolatilityMomentumGenerator',
    'VolatilityTrendGenerator',
    'create_interaction_generators'
]