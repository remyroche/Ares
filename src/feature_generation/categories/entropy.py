"""
Entropy Feature Generator

This module provides feature generators for entropy-based indicators,
including price, volume, and return entropy features.
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

# OPTIMIZED: Shared vectorized entropy calculation function
def calculate_vectorized_entropy(series: pd.Series, window: int) -> pd.Series:
    """
    Calculate entropy using vectorized operations for optimal performance.
    
    This replaces the slow rolling apply operations with fast vectorized calculations
    using variance approximation instead of histogram-based entropy.
    """
    if len(series) < window:
        return pd.Series(np.zeros(len(series)), index=series.index)
    
    # OPTIMIZED: Use variance approximation for entropy (much faster than histogram)
    # Entropy is approximated as log of variance for normal-like distributions
    rolling_var = series.rolling(window=window).var()
    entropy_approx = np.log(rolling_var + 1e-8)
    
    # Normalize entropy to reasonable range
    entropy_normalized = entropy_approx / (entropy_approx.rolling(window=window*2).std() + 1e-8)
    entropy_normalized = np.clip(entropy_normalized, 0, 1)
    
    return entropy_normalized.fillna(0)

class EntropyFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for entropy-based features."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="entropy_features",
            category=FeatureCategory.ENTROPY,
            description="Comprehensive entropy features including price, volume, and return entropy",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=20,
            min_lookback=1,
            max_lookback=100,
            parameters={
                "entropy_windows": [5, 10, 20],
                "entropy_types": ["price", "volume", "return"]
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
    
    @classmethod
    def create_default(cls) -> 'EntropyFeatureGenerator':
        return cls()
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close_prices = data['close'].values
        entropy = np.zeros_like(close_prices)
        return pd.Series(entropy, index=data.index, name='entropy_placeholder')

# Price Entropy Generator
    
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

class PriceEntropyGenerator(VectorizedFeatureGenerator):
    """Generator for price entropy features."""
    
    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"price_entropy_{window}_{base_calculation.value}",
            category=FeatureCategory.ENTROPY,
            description=f"Price entropy over {window} periods based on {base_calculation.value}",
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

        """Generate price entropy - OPTIMIZED."""
        base_values = self.base_calculator.calculate(data)
        
        # OPTIMIZED: Use vectorized entropy calculation instead of rolling apply
        price_entropy = calculate_vectorized_entropy(base_values, self.window)
        return price_entropy

# Volume Entropy Generator
    
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

class VolumeEntropyGenerator(VectorizedFeatureGenerator):
    """Generator for volume entropy features."""
    
    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.VOLUME_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"volume_entropy_{window}_{base_calculation.value}",
            category=FeatureCategory.ENTROPY,
            description=f"Volume entropy over {window} periods based on {base_calculation.value}",
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

        """Generate volume entropy - OPTIMIZED."""
        base_values = self.base_calculator.calculate(data)
        
        # OPTIMIZED: Use vectorized entropy calculation instead of rolling apply
        volume_entropy = calculate_vectorized_entropy(base_values, self.window)
        return volume_entropy

# Return Entropy Generator
    
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

class ReturnEntropyGenerator(VectorizedFeatureGenerator):
    """Generator for return entropy features."""
    
    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"return_entropy_{window}_{base_calculation.value}",
            category=FeatureCategory.ENTROPY,
            description=f"Return entropy over {window} periods based on {base_calculation.value}",
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

        """Generate return entropy - OPTIMIZED."""
        base_values = self.base_calculator.calculate(data)
        
        # OPTIMIZED: Use vectorized entropy calculation instead of rolling apply
        return_entropy = calculate_vectorized_entropy(base_values, self.window)
        return return_entropy

# Price Entropy MA Generator
    
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

class PriceEntropyMAGenerator(VectorizedFeatureGenerator):
    """Generator for price entropy moving average features."""
    
    def __init__(self, window: int = 20, ma_window: int = 5, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"price_entropy_ma_{window}_{ma_window}_{base_calculation.value}",
            category=FeatureCategory.ENTROPY,
            description=f"Price entropy MA over {window} periods with MA {ma_window} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'ma_window': ma_window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.ma_window = ma_window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate price entropy MA - OPTIMIZED."""
        base_values = self.base_calculator.calculate(data)
        
        # OPTIMIZED: Use vectorized entropy calculation instead of rolling apply
        price_entropy = calculate_vectorized_entropy(base_values, self.window)
        price_entropy_ma = price_entropy.rolling(window=self.ma_window).mean()
        return price_entropy_ma

# Volume Entropy MA Generator
    
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

class VolumeEntropyMAGenerator(VectorizedFeatureGenerator):
    """Generator for volume entropy moving average features."""
    
    def __init__(self, window: int = 20, ma_window: int = 5, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.VOLUME_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"volume_entropy_ma_{window}_{ma_window}_{base_calculation.value}",
            category=FeatureCategory.ENTROPY,
            description=f"Volume entropy MA over {window} periods with MA {ma_window} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'ma_window': ma_window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.ma_window = ma_window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate volume entropy MA - OPTIMIZED."""
        base_values = self.base_calculator.calculate(data)
        
        # OPTIMIZED: Use vectorized entropy calculation instead of rolling apply
        volume_entropy = calculate_vectorized_entropy(base_values, self.window)
        volume_entropy_ma = volume_entropy.rolling(window=self.ma_window).mean()
        return volume_entropy_ma

# Return Entropy MA Generator
    
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

class ReturnEntropyMAGenerator(VectorizedFeatureGenerator):
    """Generator for return entropy moving average features."""
    
    def __init__(self, window: int = 20, ma_window: int = 5, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"return_entropy_ma_{window}_{ma_window}_{base_calculation.value}",
            category=FeatureCategory.ENTROPY,
            description=f"Return entropy MA over {window} periods with MA {ma_window} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'ma_window': ma_window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.ma_window = ma_window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate return entropy MA - OPTIMIZED."""
        base_values = self.base_calculator.calculate(data)
        
        # OPTIMIZED: Use vectorized entropy calculation instead of rolling apply
        return_entropy = calculate_vectorized_entropy(base_values, self.window)
        return_entropy_ma = return_entropy.rolling(window=self.ma_window).mean()
        return return_entropy_ma

# High-Low Entropy Generator
    
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

class HighLowEntropyGenerator(VectorizedFeatureGenerator):
    """Generator for high-low range entropy features."""
    
    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_LEVELS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = ["high", "low", "close"]
        
        config = FeatureConfig(
            name=f"hl_entropy_{window}_{base_calculation.value}",
            category=FeatureCategory.ENTROPY,
            description=f"High-low range entropy over {window} periods based on {base_calculation.value}",
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

        """Generate high-low range entropy - OPTIMIZED."""
        hl_range = (data['high'] - data['low']) / data['close']
        
        # OPTIMIZED: Use vectorized entropy calculation instead of rolling apply
        hl_entropy = calculate_vectorized_entropy(hl_range, self.window)
        return hl_entropy

# Volatility Entropy Generator
    
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

class VolatilityEntropyGenerator(VectorizedFeatureGenerator):
    """Generator for volatility entropy features."""
    
    def __init__(self, window: int = 20, volatility_window: int = 10, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"volatility_entropy_{window}_{volatility_window}_{base_calculation.value}",
            category=FeatureCategory.ENTROPY,
            description=f"Volatility entropy over {window} periods with {volatility_window} volatility window based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window + volatility_window,
            min_lookback=window + volatility_window,
            max_lookback=window + volatility_window,
            parameters={'window': window, 'volatility_window': volatility_window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.volatility_window = volatility_window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate volatility entropy - OPTIMIZED."""
        base_values = self.base_calculator.calculate(data)
        volatility = base_values.rolling(window=self.volatility_window).std()
        
        # OPTIMIZED: Use vectorized entropy calculation instead of rolling apply
        volatility_entropy = calculate_vectorized_entropy(volatility, self.window)
        return volatility_entropy

# Add 6 more entropy generators to reach 15 total
    
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

class MomentumEntropyGenerator(VectorizedFeatureGenerator):
    """Generator for momentum entropy features."""
    
    def __init__(self, window: int = 20, momentum_period: int = 5):
        config = FeatureConfig(
            name=f"momentum_entropy_{window}_{momentum_period}",
            category=FeatureCategory.ENTROPY,
            description=f"Momentum entropy over {window} periods with {momentum_period} momentum period",
            required_columns=["close"],
            default_lookback=window + momentum_period,
            min_lookback=window + momentum_period,
            max_lookback=window + momentum_period,
            parameters={'window': window, 'momentum_period': momentum_period}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.momentum_period = momentum_period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate momentum entropy - OPTIMIZED."""
        momentum = data['close'].pct_change(self.momentum_period)
        
        # OPTIMIZED: Use vectorized entropy calculation instead of rolling apply
        return calculate_vectorized_entropy(momentum, self.window)

    
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

class RSIEntropyGenerator(VectorizedFeatureGenerator):
    """Generator for RSI entropy features."""
    
    def __init__(self, window: int = 20, rsi_period: int = 14):
        config = FeatureConfig(
            name=f"rsi_entropy_{window}_{rsi_period}",
            category=FeatureCategory.ENTROPY,
            description=f"RSI entropy over {window} periods with {rsi_period} RSI period",
            required_columns=["close"],
            default_lookback=window + rsi_period,
            min_lookback=window + rsi_period,
            max_lookback=window + rsi_period,
            parameters={'window': window, 'rsi_period': rsi_period}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.rsi_period = rsi_period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate RSI entropy - OPTIMIZED."""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        # OPTIMIZED: Use vectorized entropy calculation instead of rolling apply
        return calculate_vectorized_entropy(rsi, self.window)

    
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

class MACDEntropyGenerator(VectorizedFeatureGenerator):
    """Generator for MACD entropy features."""
    
    def __init__(self, window: int = 20, fast: int = 12, slow: int = 26):
        config = FeatureConfig(
            name=f"macd_entropy_{window}_{fast}_{slow}",
            category=FeatureCategory.ENTROPY,
            description=f"MACD entropy over {window} periods with {fast}/{slow} MACD periods",
            required_columns=["close"],
            default_lookback=window + slow,
            min_lookback=window + slow,
            max_lookback=window + slow,
            parameters={'window': window, 'fast': fast, 'slow': slow}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.fast = fast
        self.slow = slow
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate MACD entropy - OPTIMIZED."""
        ema_fast = data['close'].ewm(span=self.fast).mean()
        ema_slow = data['close'].ewm(span=self.slow).mean()
        macd = ema_fast - ema_slow
        
        # OPTIMIZED: Use vectorized entropy calculation instead of rolling apply
        return calculate_vectorized_entropy(macd, self.window)

    
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

class BollingerBandsEntropyGenerator(VectorizedFeatureGenerator):
    """Generator for Bollinger Bands position entropy features."""
    
    def __init__(self, window: int = 20, bb_period: int = 20, bb_std: float = 2.0):
        config = FeatureConfig(
            name=f"bb_entropy_{window}_{bb_period}_{bb_std}",
            category=FeatureCategory.ENTROPY,
            description=f"Bollinger Bands position entropy over {window} periods",
            required_columns=["close"],
            default_lookback=window + bb_period,
            min_lookback=window + bb_period,
            max_lookback=window + bb_period,
            parameters={'window': window, 'bb_period': bb_period, 'bb_std': bb_std}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.bb_period = bb_period
        self.bb_std = bb_std
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate Bollinger Bands position entropy - OPTIMIZED."""
        sma = data['close'].rolling(window=self.bb_period).mean()
        std = data['close'].rolling(window=self.bb_period).std()
        upper_band = sma + (std * self.bb_std)
        lower_band = sma - (std * self.bb_std)
        bb_position = (data['close'] - lower_band) / (upper_band - lower_band + 1e-8)
        
        # OPTIMIZED: Use vectorized entropy calculation instead of rolling apply
        return calculate_vectorized_entropy(bb_position, self.window)

    
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

class CrossAssetEntropyGenerator(VectorizedFeatureGenerator):
    """Generator for cross-asset correlation entropy features."""
    
    def __init__(self, window: int = 20, correlation_window: int = 10):
        config = FeatureConfig(
            name=f"cross_asset_entropy_{window}_{correlation_window}",
            category=FeatureCategory.ENTROPY,
            description=f"Cross-asset correlation entropy over {window} periods",
            required_columns=["close", "volume"],
            default_lookback=window + correlation_window,
            min_lookback=window + correlation_window,
            max_lookback=window + correlation_window,
            parameters={'window': window, 'correlation_window': correlation_window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.correlation_window = correlation_window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate cross-asset correlation entropy - OPTIMIZED."""
        price_returns = data['close'].pct_change()
        volume_returns = data['volume'].pct_change()
        correlation = price_returns.rolling(window=self.correlation_window).corr(volume_returns)
        
        # OPTIMIZED: Use vectorized entropy calculation instead of rolling apply
        return calculate_vectorized_entropy(correlation, self.window)

    
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

class RegimeEntropyGenerator(VectorizedFeatureGenerator):
    """Generator for regime transition entropy features."""
    
    def __init__(self, window: int = 20, regime_window: int = 50):
        config = FeatureConfig(
            name=f"regime_entropy_{window}_{regime_window}",
            category=FeatureCategory.ENTROPY,
            description=f"Regime transition entropy over {window} periods",
            required_columns=["close"],
            default_lookback=window + regime_window,
            min_lookback=window + regime_window,
            max_lookback=window + regime_window,
            parameters={'window': window, 'regime_window': regime_window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.regime_window = regime_window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate regime transition entropy - OPTIMIZED."""
        volatility = data['close'].rolling(window=20).std()
        regime = pd.cut(volatility.rolling(window=self.regime_window).rank(pct=True), 
                       bins=3, labels=[0, 1, 2]).astype(float)
        
        # OPTIMIZED: Use vectorized entropy calculation instead of rolling apply
        return calculate_vectorized_entropy(regime, self.window)

def create_default_entropy_generators() -> List[FeatureGenerator]:
    """Create default entropy feature generators."""
    windows = [5, 10, 20]
    ma_windows = [5, 10]
    
    generators = []
    
    # Create generators for each window
    for window in windows:
        generators.extend([
            PriceEntropyGenerator(window),
            VolumeEntropyGenerator(window),
            ReturnEntropyGenerator(window),
        ])
        
        # Create MA generators
        for ma_window in ma_windows:
            generators.extend([
                PriceEntropyMAGenerator(window, ma_window),
                VolumeEntropyMAGenerator(window, ma_window),
                ReturnEntropyMAGenerator(window, ma_window),
            ])
    
    return generators

def create_entropy_generators() -> List[FeatureGenerator]:
    """Create all 15 entropy feature generators."""
    generators = []
    
    # Original 7 generators
    generators.append(PriceEntropyGenerator(window=20))
    generators.append(VolumeEntropyGenerator(window=20))
    generators.append(ReturnEntropyGenerator(window=20))
    generators.append(PriceEntropyMAGenerator(window=20, ma_window=5))
    generators.append(VolumeEntropyMAGenerator(window=20, ma_window=5))
    generators.append(ReturnEntropyMAGenerator(window=20, ma_window=5))
    
    # New 8 generators to reach 15 total
    generators.append(HighLowEntropyGenerator(window=20))
    generators.append(VolatilityEntropyGenerator(window=20, volatility_window=10))
    generators.append(MomentumEntropyGenerator(window=20, momentum_period=5))
    generators.append(RSIEntropyGenerator(window=20, rsi_period=14))
    generators.append(MACDEntropyGenerator(window=20, fast=12, slow=26))
    generators.append(BollingerBandsEntropyGenerator(window=20, bb_period=20, bb_std=2.0))
    generators.append(CrossAssetEntropyGenerator(window=20, correlation_window=10))
    generators.append(RegimeEntropyGenerator(window=20, regime_window=50))
    
    return generators