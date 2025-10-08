"""
Feature Interaction Generators

This module provides feature generators for feature interactions, combinations,
and derived features that capture relationships between different indicators.
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
from ..base_calculations import (
    BaseCalculationType,
    create_base_calculator
)

class InteractionFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for interaction-based features."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
    
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
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate momentum divergence."""
        base_values = self.base_calculator.calculate(data)
        price_momentum = base_values.pct_change(self.period)
        volume_momentum = data['volume'].pct_change(self.period)
        divergence = price_momentum - volume_momentum
        return divergence

# Momentum Volume Generator
    
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
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate momentum-volume interaction."""
        base_values = self.base_calculator.calculate(data)
        price_momentum = base_values.pct_change(self.period)
        volume_momentum = data['volume'].pct_change(self.period)
        interaction = price_momentum * volume_momentum
        return interaction

# Momentum Volatility Generator
    
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
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.volatility_window = volatility_window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate momentum-volatility interaction."""
        base_values = self.base_calculator.calculate(data)
        price_momentum = base_values.pct_change(self.period)
        volatility = base_values.rolling(window=self.volatility_window).std()
        # Normalize momentum by volatility
        interaction = price_momentum / (volatility + 1e-8)  # Add small epsilon to prevent division by zero
        return interaction

# Momentum Trend Generator
    
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
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.momentum_period = momentum_period
        self.trend_window = trend_window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

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
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.volatility_window = volatility_window
        self.volume_window = volume_window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate volatility-volume interaction."""
        base_values = self.base_calculator.calculate(data)
        volatility = base_values.rolling(window=self.volatility_window).std()
        volume_ma = data['volume'].rolling(window=self.volume_window).mean()
        interaction = volatility * volume_ma
        return interaction

# Volatility Price Generator
    
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
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.volatility_window = volatility_window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

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
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.volatility_window = volatility_window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate volatility-high-low range interaction."""
        base_values = self.base_calculator.calculate(data)
        volatility = base_values.rolling(window=self.volatility_window).std()
        hl_range_pct = (data['high'] - data['low']) / data['close']
        interaction = volatility * hl_range_pct
        return interaction

# Volatility Momentum Generator
    
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
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.volatility_window = volatility_window
        self.momentum_period = momentum_period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate volatility-momentum interaction."""
        base_values = self.base_calculator.calculate(data)
        volatility = base_values.rolling(window=self.volatility_window).std()
        momentum = base_values.pct_change(self.momentum_period)
        interaction = volatility * momentum
        return interaction

# Volatility Trend Generator
    
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
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.volatility_window = volatility_window
        self.trend_window = trend_window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

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


# Legacy Interaction Generators
# =============================

    
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

class CrossTimeframeInteractionGenerator(FeatureGenerator):
    """Generator for cross-timeframe feature interactions."""
    
    def __init__(self, short_period: int = 5, long_period: int = 20, interaction_type: str = "ratio", base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"cross_timeframe_{interaction_type}_{short_period}_{long_period}_{base_calculation.value}",
            category=FeatureCategory.INTERACTION,
            description=f"Cross-timeframe {interaction_type} interaction between {short_period} and {long_period} periods",
            required_columns=required_columns,
            default_lookback=max(short_period, long_period),
            min_lookback=max(short_period, long_period),
            max_lookback=max(short_period, long_period),
            parameters={'short_period': short_period, 'long_period': long_period, 'interaction_type': interaction_type, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.short_period = short_period
        self.long_period = long_period
        self.interaction_type = interaction_type
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate cross-timeframe interaction."""
        base_values = self.base_calculator.calculate(data)
        
        # Calculate short and long period features
        short_ma = base_values.rolling(window=self.short_period).mean()
        long_ma = base_values.rolling(window=self.long_period).mean()
        
        if self.interaction_type == "ratio":
            interaction = short_ma / (long_ma + 1e-8)  # Add small epsilon to prevent division by zero
        elif self.interaction_type == "difference":
            interaction = short_ma - long_ma
        elif self.interaction_type == "momentum":
            interaction = (short_ma - long_ma) / (long_ma + 1e-8)
        elif self.interaction_type == "crossover":
            # Binary signal: 1 if short > long, 0 otherwise
            interaction = (short_ma > long_ma).astype(float)
        else:
            raise ValueError(f"Unknown interaction_type: {self.interaction_type}")
        
        return interaction


    
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

class FeatureRatioGenerator(FeatureGenerator):
    """Generator for ratios between different features."""
    
    def __init__(self, numerator_column: str = "close", denominator_column: str = "volume", window: int = 1):
        config = FeatureConfig(
            name=f"ratio_{numerator_column}_{denominator_column}_{window}",
            category=FeatureCategory.INTERACTION,
            description=f"Ratio between {numerator_column} and {denominator_column} with {window} period smoothing",
            required_columns=[numerator_column, denominator_column],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'numerator_column': numerator_column, 'denominator_column': denominator_column, 'window': window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.numerator_column = numerator_column
        self.denominator_column = denominator_column
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate feature ratio."""
        numerator = data[self.numerator_column]
        denominator = data[self.denominator_column]
        
        if self.window > 1:
            numerator = numerator.rolling(window=self.window).mean()
            denominator = denominator.rolling(window=self.window).mean()
        
        ratio = numerator / (denominator + 1e-8)  # Add small epsilon to prevent division by zero
        return ratio


    
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

class PolynomialFeatureGenerator(FeatureGenerator):
    """Generator for polynomial transformations of features."""
    
    def __init__(self, column: str = "close", degree: int = 2, include_bias: bool = False):
        config = FeatureConfig(
            name=f"poly_{column}_deg{degree}{'_bias' if include_bias else ''}",
            category=FeatureCategory.INTERACTION,
            description=f"Polynomial transformation of {column} with degree {degree}",
            required_columns=[column],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            parameters={'column': column, 'degree': degree, 'include_bias': include_bias}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.column = column
        self.degree = degree
        self.include_bias = include_bias
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate polynomial transformation."""
        values = data[self.column]
        
        # Normalize values to prevent numerical overflow
        values_normalized = (values - values.mean()) / (values.std() + 1e-8)
        
        # Create polynomial features
        result = values_normalized ** self.degree
        
        if self.include_bias:
            result = result + 1.0
        
        return result


    
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

class CorrelationInteractionGenerator(FeatureGenerator):
    """Generator for correlation-based feature interactions."""
    
    def __init__(self, column1: str = "close", column2: str = "volume", window: int = 20, method: str = "pearson"):
        config = FeatureConfig(
            name=f"corr_{column1}_{column2}_{window}_{method}",
            category=FeatureCategory.INTERACTION,
            description=f"{method.capitalize()} correlation between {column1} and {column2} over {window} periods",
            required_columns=[column1, column2],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'column1': column1, 'column2': column2, 'window': window, 'method': method}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.column1 = column1
        self.column2 = column2
        self.window = window
        self.method = method
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate correlation interaction."""
        values1 = data[self.column1]
        values2 = data[self.column2]
        
        # Calculate rolling correlation
        correlation = values1.rolling(window=self.window).corr(values2)
        
        return correlation


def create_default_interaction_generators() -> List[FeatureGenerator]:
    """Create default set of legacy interaction generators."""
    generators = []
    
    # Cross-timeframe interactions
    generators.append(CrossTimeframeInteractionGenerator(short_period=5, long_period=20, interaction_type="ratio"))
    generators.append(CrossTimeframeInteractionGenerator(short_period=10, long_period=50, interaction_type="momentum"))
    
    # Feature ratios
    generators.append(FeatureRatioGenerator(numerator_column="close", denominator_column="volume"))
    generators.append(FeatureRatioGenerator(numerator_column="high", denominator_column="low"))
    
    # Polynomial features
    generators.append(PolynomialFeatureGenerator(column="close", degree=2))
    generators.append(PolynomialFeatureGenerator(column="volume", degree=2))
    
    # Correlation interactions
    generators.append(CorrelationInteractionGenerator(column1="close", column2="volume", window=20))
    generators.append(CorrelationInteractionGenerator(column1="high", column2="low", window=10))
    
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
    'create_interaction_generators',
    # Legacy interaction generators
    'CrossTimeframeInteractionGenerator',
    'FeatureRatioGenerator',
    'PolynomialFeatureGenerator',
    'CorrelationInteractionGenerator',
    'create_default_interaction_generators'
    # Note: RegimeDependentFeatureGenerator, CointegrationResidualGenerator,
    # StructuralRatioGenerator, PairwiseInteractionGenerator are not implemented yet
    # They are referenced but not defined - removed from exports to prevent import errors
]