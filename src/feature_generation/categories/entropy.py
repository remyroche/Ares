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
    rolling_var = self._vectorbt_rolling_operation(series, "var", window)
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

# NEW FEATURES - Advanced Entropy Analysis

class ShannonEntropyGenerator(VectorizedFeatureGenerator):
    """Generator for Shannon entropy of discretized returns."""
    
    def __init__(self, window: int = 20, q_bins: int = 10):
        config = FeatureConfig(
            name=f"shannon_entropy_{window}_{q_bins}",
            category=FeatureCategory.ENTROPY,
            description=f"Shannon entropy of discretized returns over {window} periods with {q_bins} bins",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'q_bins': q_bins},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.q_bins = q_bins
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close'].values
        if len(close) < self.window + 1:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate returns
        returns = np.diff(close) / close[:-1]
        returns = np.concatenate([[np.nan], returns])
        
        # Calculate Shannon entropy
        shannon_entropy = np.full(len(close), np.nan)
        for i in range(self.window, len(close)):
            window_returns = returns[i - self.window + 1:i + 1]
            valid_returns = window_returns[np.isfinite(window_returns)]
            
            if len(valid_returns) > 1:
                # Discretize returns into q_bins
                bins = np.linspace(np.min(valid_returns), np.max(valid_returns), self.q_bins + 1)
                digitized = np.digitize(valid_returns, bins) - 1
                digitized = np.clip(digitized, 0, self.q_bins - 1)
                
                # Calculate probabilities
                counts = np.bincount(digitized, minlength=self.q_bins)
                probabilities = counts / len(valid_returns)
                
                # Calculate Shannon entropy
                entropy = 0
                for p in probabilities:
                    if p > 0:
                        entropy -= p * np.log2(p)
                
                shannon_entropy[i] = entropy
        
        return pd.Series(shannon_entropy, index=data.index)

class PermutationEntropyGenerator(VectorizedFeatureGenerator):
    """Generator for permutation entropy on returns."""
    
    def __init__(self, window: int = 20, embedding_dim: int = 3, delay: int = 1):
        config = FeatureConfig(
            name=f"permutation_entropy_{window}_{embedding_dim}_{delay}",
            category=FeatureCategory.ENTROPY,
            description=f"Permutation entropy over {window} periods (embedding dim {embedding_dim}, delay {delay})",
            required_columns=["close"],
            default_lookback=window + embedding_dim * delay,
            min_lookback=window + embedding_dim * delay,
            max_lookback=window + embedding_dim * delay,
            parameters={'window': window, 'embedding_dim': embedding_dim, 'delay': delay},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.embedding_dim = embedding_dim
        self.delay = delay
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close'].values
        if len(close) < self.window + self.embedding_dim * self.delay:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate returns
        returns = np.diff(close) / close[:-1]
        returns = np.concatenate([[np.nan], returns])
        
        # Calculate permutation entropy
        perm_entropy = np.full(len(close), np.nan)
        for i in range(self.window + self.embedding_dim * self.delay - 1, len(close)):
            window_returns = returns[i - self.window + 1:i + 1]
            valid_returns = window_returns[np.isfinite(window_returns)]
            
            if len(valid_returns) >= self.embedding_dim:
                # Create embedding vectors
                vectors = []
                for j in range(len(valid_returns) - (self.embedding_dim - 1) * self.delay):
                    vector = valid_returns[j:j + self.embedding_dim * self.delay:self.delay]
                    if len(vector) == self.embedding_dim:
                        vectors.append(vector)
                
                if len(vectors) > 0:
                    # Calculate permutation patterns
                    patterns = []
                    for vector in vectors:
                        # Get permutation pattern
                        pattern = np.argsort(vector)
                        patterns.append(tuple(pattern))
                    
                    # Calculate probabilities
                    unique_patterns, counts = np.unique(patterns, return_counts=True)
                    probabilities = counts / len(patterns)
                    
                    # Calculate permutation entropy
                    entropy = 0
                    for p in probabilities:
                        if p > 0:
                            entropy -= p * np.log2(p)
                    
                    perm_entropy[i] = entropy
        
        return pd.Series(perm_entropy, index=data.index)

class SampleEntropyGenerator(VectorizedFeatureGenerator):
    """Generator for sample entropy on returns."""
    
    def __init__(self, window: int = 20, m: int = 2, r: float = 0.2):
        config = FeatureConfig(
            name=f"sample_entropy_{window}_{m}_{r}",
            category=FeatureCategory.ENTROPY,
            description=f"Sample entropy over {window} periods (m={m}, r={r})",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'm': m, 'r': r},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.m = m
        self.r = r
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close'].values
        if len(close) < self.window + self.m:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate returns
        returns = np.diff(close) / close[:-1]
        returns = np.concatenate([[np.nan], returns])
        
        # Calculate sample entropy
        sample_entropy = np.full(len(close), np.nan)
        for i in range(self.window + self.m - 1, len(close)):
            window_returns = returns[i - self.window + 1:i + 1]
            valid_returns = window_returns[np.isfinite(window_returns)]
            
            if len(valid_returns) >= self.m + 1:
                # Calculate sample entropy
                entropy = self._calculate_sample_entropy(valid_returns, self.m, self.r)
                sample_entropy[i] = entropy
        
        return pd.Series(sample_entropy, index=data.index)
    
    def _calculate_sample_entropy(self, data: np.ndarray, m: int, r: float) -> float:
        """Calculate sample entropy."""
        N = len(data)
        if N < m + 1:
            return 0.0
        
        # Create template vectors
        def _maxdist(xi, xj, m):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _get_template_vectors(data, m):
            return [data[i:i + m] for i in range(N - m + 1)]
        
        # Calculate phi(m) and phi(m+1)
        def _calculate_phi(data, m):
            template_vectors = _get_template_vectors(data, m)
            N = len(template_vectors)
            C = np.zeros(N)
            
            for i in range(N):
                template_i = template_vectors[i]
                for j in range(N):
                    if i != j:
                        if _maxdist(template_i, template_vectors[j], m) <= r:
                            C[i] += 1
            
            phi = np.sum(np.log(C / (N - 1))) / N
            return phi
        
        phi_m = _calculate_phi(data, m)
        phi_m1 = _calculate_phi(data, m + 1)
        
        return phi_m - phi_m1

class LempelZivComplexityGenerator(VectorizedFeatureGenerator):
    """Generator for Lempel-Ziv complexity of up/down sequence."""
    
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"lempel_ziv_complexity_{window}",
            category=FeatureCategory.ENTROPY,
            description=f"Lempel-Ziv complexity of up/down sequence over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close'].values
        if len(close) < self.window + 1:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate returns and up/down sequence
        returns = np.diff(close) / close[:-1]
        up_down = np.where(returns > 0, 1, 0)  # 1 for up, 0 for down
        up_down = np.concatenate([[0], up_down])  # Add initial value
        
        # Calculate Lempel-Ziv complexity
        lz_complexity = np.full(len(close), np.nan)
        for i in range(self.window, len(close)):
            sequence = up_down[i - self.window + 1:i + 1]
            complexity = self._calculate_lz_complexity(sequence)
            lz_complexity[i] = complexity
        
        return pd.Series(lz_complexity, index=data.index)
    
    def _calculate_lz_complexity(self, sequence: np.ndarray) -> float:
        """Calculate Lempel-Ziv complexity."""
        if len(sequence) == 0:
            return 0.0
        
        # Convert to string for LZ algorithm
        s = ''.join(map(str, sequence))
        n = len(s)
        
        # Lempel-Ziv complexity calculation
        c = 1
        i = 0
        while i + c <= n:
            substring = s[i:i + c]
            if substring in s[:i + c - 1]:
                c += 1
            else:
                i += c
                c = 1
        
        return c

class EntropyRateGenerator(VectorizedFeatureGenerator):
    """Generator for entropy rate of 2-state Markov chain for sign(returns)."""
    
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"entropy_rate_{window}",
            category=FeatureCategory.ENTROPY,
            description=f"Entropy rate of 2-state Markov chain over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close'].values
        if len(close) < self.window + 1:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate returns and signs
        returns = np.diff(close) / close[:-1]
        signs = np.sign(returns)
        signs = np.concatenate([[0], signs])  # Add initial value
        
        # Calculate entropy rate
        entropy_rate = np.full(len(close), np.nan)
        for i in range(self.window, len(close)):
            sequence = signs[i - self.window + 1:i + 1]
            rate = self._calculate_entropy_rate(sequence)
            entropy_rate[i] = rate
        
        return pd.Series(entropy_rate, index=data.index)
    
    def _calculate_entropy_rate(self, sequence: np.ndarray) -> float:
        """Calculate entropy rate of 2-state Markov chain."""
        if len(sequence) < 2:
            return 0.0
        
        # Count transitions
        transitions = {
            (1, 1): 0, (1, -1): 0, (1, 0): 0,
            (-1, 1): 0, (-1, -1): 0, (-1, 0): 0,
            (0, 1): 0, (0, -1): 0, (0, 0): 0
        }
        
        for i in range(len(sequence) - 1):
            transition = (int(sequence[i]), int(sequence[i + 1]))
            if transition in transitions:
                transitions[transition] += 1
        
        # Calculate transition probabilities
        total_transitions = sum(transitions.values())
        if total_transitions == 0:
            return 0.0
        
        # Calculate entropy rate
        entropy_rate = 0.0
        for count in transitions.values():
            if count > 0:
                p = count / total_transitions
                entropy_rate -= p * np.log2(p)
        
        return entropy_rate

class SpectralEntropyGenerator(VectorizedFeatureGenerator):
    """Generator for spectral entropy of returns (normalized PSD)."""
    
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"spectral_entropy_{window}",
            category=FeatureCategory.ENTROPY,
            description=f"Spectral entropy of returns over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close'].values
        if len(close) < self.window + 1:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate returns
        returns = np.diff(close) / close[:-1]
        returns = np.concatenate([[np.nan], returns])
        
        # Calculate spectral entropy
        spectral_entropy = np.full(len(close), np.nan)
        for i in range(self.window, len(close)):
            window_returns = returns[i - self.window + 1:i + 1]
            valid_returns = window_returns[np.isfinite(window_returns)]
            
            if len(valid_returns) > 4:  # Need enough data for FFT
                entropy = self._calculate_spectral_entropy(valid_returns)
                spectral_entropy[i] = entropy
        
        return pd.Series(spectral_entropy, index=data.index)
    
    def _calculate_spectral_entropy(self, data: np.ndarray) -> float:
        """Calculate spectral entropy from power spectral density."""
        try:
            # Calculate FFT
            fft = np.fft.fft(data)
            psd = np.abs(fft) ** 2
            
            # Normalize PSD
            psd = psd / np.sum(psd)
            
            # Calculate spectral entropy
            entropy = 0.0
            for p in psd:
                if p > 0:
                    entropy -= p * np.log2(p)
            
            return entropy
        except:
            return 0.0

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
    
    # NEW FEATURES - Advanced Entropy Analysis
    # Shannon entropy generators
    for window in [20]:
        for q_bins in [10]:
            generators.append(ShannonEntropyGenerator(window, q_bins))
    
    # Permutation entropy generators
    for window in [20]:
        for embedding_dim in [3]:
            for delay in [1]:
                generators.append(PermutationEntropyGenerator(window, embedding_dim, delay))
    
    # Sample entropy generators
    for window in [20]:
        for m in [2]:
            for r in [0.2]:
                generators.append(SampleEntropyGenerator(window, m, r))
    
    # Lempel-Ziv complexity generators
    for window in [20]:
        generators.append(LempelZivComplexityGenerator(window))
    
    # Entropy rate generators
    for window in [20]:
        generators.append(EntropyRateGenerator(window))
    
    # Spectral entropy generators
    for window in [20]:
        generators.append(SpectralEntropyGenerator(window))
    
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
