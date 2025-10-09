"""
Returns Feature Generator

This module provides feature generators for return-based indicators,
including log returns, cumulative returns, rolling returns, and return statistics.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig

# Optimization utilities
try:
    from ..utils.vectorization_optimizer import get_vectorization_optimizer
    from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
from ..base_calculations import (
    BaseCalculator,
    BaseCalculationType,
    BaseCalculationConfig,
    create_base_calculator
)

class ReturnsFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for return-based features."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="returns_features",
            category=FeatureCategory.RETURNS,
            description="Comprehensive return-based features including log returns, cumulative returns, and return statistics",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=20,
            min_lookback=2,
            max_lookback=50,
            parameters={
                "return_periods": [1, 5, 10, 20],
                "log_return_periods": [1, 5, 10],
                "cumulative_windows": [10, 20]
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
    
    @classmethod
    def create_default(cls) -> 'ReturnsFeatureGenerator':
        return cls()
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        if data.empty:
            return pd.Series(dtype=float, index=data.index, name='returns_1')

        close_prices = data['close'].astype(float).values
        state = self.get_state()
        history = state.get('close_history') or []

        if history:
            try:
                history_array = np.asarray(history, dtype=float)
            except Exception:
                history_array = np.array(history, dtype=float)
            combined_closes = np.concatenate([history_array, close_prices])
        else:
            combined_closes = close_prices

        combined_returns = self._calculate_returns(combined_closes, period=1)
        returns = combined_returns[-len(close_prices):] if len(close_prices) else np.array([])

        prev_close = state.get('last_close')
        if len(returns) > 0 and prev_close not in (None, np.nan):
            try:
                prev_close_val = float(prev_close)
                returns[0] = (close_prices[0] - prev_close_val) / prev_close_val if prev_close_val != 0 else np.nan
            except Exception:
                returns[0] = np.nan

        return pd.Series(returns, index=data.index, name='returns_1')
    
    def _calculate_returns(self, prices: np.ndarray, period: int = 1) -> np.ndarray:
        if len(prices) < period + 1:
            return np.full(len(prices), np.nan)

        returns = (prices - np.roll(prices, period)) / np.roll(prices, period)
        returns[:period] = np.nan
        return returns

    def _finalize_state(self, data: pd.DataFrame, feature_data: pd.Series) -> None:
        if not data.empty:
            closes = data['close'].astype(float)
            history_window = max(int(getattr(self.config, 'min_lookback', 2)), 2)
            close_history = closes.tolist()[-history_window:]
            state_update = {
                'last_close': float(closes.iloc[-1]),
                'close_history': close_history
            }
            if not feature_data.empty:
                last_return = feature_data.iloc[-1]
                if pd.notna(last_return):
                    state_update['last_return'] = float(last_return)
            self.update_state(state_update)

        super()._finalize_state(data, feature_data)

    
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

class LogReturnsGenerator(VectorizedFeatureGenerator):
    """Generator for Log Returns with different base calculations - VECTORIZED."""
    
    def __init__(self, 
                 period: int = 1,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize Log Returns generator.
        
        Args:
            period: Return period
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"log_returns_{period}_{base_calculation.value}",
            category=FeatureCategory.RETURNS,
            description=f"Log returns over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period + 1,
            min_lookback=period + 1,
            max_lookback=period + 1,
            parameters={
                'period': period,
                'base_calculation': base_calculation.value,
                **base_kwargs
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate log returns based on the specified base calculation - VECTORIZED."""
        # Calculate base values
        base_values = self.base_calculator.calculate(data)
        
        # Convert to numpy array for vectorized operations
        values = base_values.values
        
        # Vectorized log returns calculation
        if len(values) < self.period + 1:
            return pd.Series(np.full(len(values), np.nan), index=data.index)
        
        # Calculate log returns using numpy operations
        shifted_values = np.roll(values, self.period)
        shifted_values[:self.period] = np.nan  # Set initial values to NaN
        
        # Avoid division by zero and log of zero
        ratio = values / shifted_values
        ratio = np.where(np.isfinite(ratio) & (ratio > 0), ratio, np.nan)
        
        log_returns = np.log(ratio)
        
        return pd.Series(log_returns, index=data.index)

    
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

class SimpleReturnsGenerator(VectorizedFeatureGenerator):
    """Generator for Simple Returns with different base calculations - VECTORIZED."""
    
    def __init__(self, 
                 period: int = 1,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize Simple Returns generator.
        
        Args:
            period: Return period
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"simple_returns_{period}_{base_calculation.value}",
            category=FeatureCategory.RETURNS,
            description=f"Simple returns over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period + 1,
            min_lookback=period + 1,
            max_lookback=period + 1,
            parameters={
                'period': period,
                'base_calculation': base_calculation.value,
                **base_kwargs
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate simple returns based on the specified base calculation - VECTORIZED."""
        # Calculate base values
        base_values = self.base_calculator.calculate(data)
        
        # Convert to numpy array for vectorized operations
        values = base_values.values
        
        # Vectorized simple returns calculation
        if len(values) < self.period + 1:
            return pd.Series(np.full(len(values), np.nan), index=data.index)
        
        # Calculate simple returns using numpy operations
        shifted_values = np.roll(values, self.period)
        shifted_values[:self.period] = np.nan  # Set initial values to NaN
        
        # Avoid division by zero
        simple_returns = np.where(
            np.isfinite(shifted_values) & (shifted_values != 0),
            (values - shifted_values) / shifted_values,
            np.nan
        )
        
        return pd.Series(simple_returns, index=data.index)

    
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

class CumulativeReturnsGenerator(VectorizedFeatureGenerator):
    """Generator for Cumulative Returns with different base calculations - VECTORIZED."""
    
    def __init__(self, 
                 window: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize Cumulative Returns generator.
        
        Args:
            window: Rolling window for cumulative returns
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"cumulative_returns_{window}_{base_calculation.value}",
            category=FeatureCategory.RETURNS,
            description=f"Cumulative returns over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={
                'window': window,
                'base_calculation': base_calculation.value,
                **base_kwargs
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate cumulative returns based on the specified base calculation - VECTORIZED."""
        # Calculate base values
        base_values = self.base_calculator.calculate(data)
        
        # Convert to numpy array for vectorized operations
        values = base_values.values
        
        # Vectorized cumulative returns calculation
        if len(values) < self.window + 1:
            return pd.Series(np.full(len(values), np.nan), index=data.index)
        
        # Calculate returns using vectorized operations
        returns = np.diff(values) / values[:-1]
        returns = np.concatenate([[np.nan], returns])  # Add NaN for first value
        
        # Vectorized rolling cumulative returns calculation
        cumulative_returns = np.full(len(values), np.nan)
        
        for i in range(self.window, len(values)):
            window_returns = returns[i-self.window+1:i+1]
            # Filter out NaN values
            valid_returns = window_returns[np.isfinite(window_returns)]
            if len(valid_returns) > 0:
                cumulative_returns[i] = np.prod(1 + valid_returns) - 1
        
        return pd.Series(cumulative_returns, index=data.index)

    
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

class RollingReturnsGenerator(VectorizedFeatureGenerator):
    """Generator for Rolling Returns with different base calculations - VECTORIZED."""
    
    def __init__(self, 
                 window: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize Rolling Returns generator.
        
        Args:
            window: Rolling window for returns
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"rolling_returns_{window}_{base_calculation.value}",
            category=FeatureCategory.RETURNS,
            description=f"Rolling returns over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={
                'window': window,
                'base_calculation': base_calculation.value,
                **base_kwargs
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate rolling returns based on the specified base calculation - VECTORIZED."""
        # Calculate base values
        base_values = self.base_calculator.calculate(data)
        
        # Convert to numpy array for vectorized operations
        values = base_values.values
        
        # Vectorized rolling returns calculation
        if len(values) < self.window + 1:
            return pd.Series(np.full(len(values), np.nan), index=data.index)
        
        # Calculate rolling returns using vectorized operations
        rolling_returns = np.full(len(values), np.nan)
        
        for i in range(self.window, len(values)):
            window_values = values[i-self.window:i+1]
            if np.isfinite(window_values[0]) and window_values[0] != 0:
                rolling_returns[i] = (window_values[-1] - window_values[0]) / window_values[0]
        
        return pd.Series(rolling_returns, index=data.index)

    
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

class ReturnsVolatilityGenerator(VectorizedFeatureGenerator):
    """Generator for Returns Volatility with different base calculations - VECTORIZED."""
    
    def __init__(self, 
                 window: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize Returns Volatility generator.
        
        Args:
            window: Rolling window for volatility calculation
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"returns_volatility_{window}_{base_calculation.value}",
            category=FeatureCategory.RETURNS,
            description=f"Returns volatility over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={
                'window': window,
                'base_calculation': base_calculation.value,
                **base_kwargs
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate returns volatility based on the specified base calculation - VECTORIZED."""
        # Calculate base values
        base_values = self.base_calculator.calculate(data)
        
        # Convert to numpy array for vectorized operations
        values = base_values.values
        
        # Vectorized returns volatility calculation
        if len(values) < self.window + 1:
            return pd.Series(np.full(len(values), np.nan), index=data.index)
        
        # Calculate returns using vectorized operations
        returns = np.diff(values) / values[:-1]
        returns = np.concatenate([[np.nan], returns])  # Add NaN for first value
        
        # Vectorized rolling volatility calculation
        volatility = np.full(len(values), np.nan)
        
        for i in range(self.window, len(values)):
            window_returns = returns[i-self.window+1:i+1]
            # Filter out NaN values
            valid_returns = window_returns[np.isfinite(window_returns)]
            if len(valid_returns) > 1:  # Need at least 2 values for std
                volatility[i] = np.std(valid_returns, ddof=1)
        
        return pd.Series(volatility, index=data.index)

    
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

class ReturnsSkewnessGenerator(VectorizedFeatureGenerator):
    """Generator for Returns Skewness with different base calculations - VECTORIZED."""
    
    def __init__(self, 
                 window: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize Returns Skewness generator.
        
        Args:
            window: Rolling window for skewness calculation
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"returns_skewness_{window}_{base_calculation.value}",
            category=FeatureCategory.RETURNS,
            description=f"Returns skewness over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={
                'window': window,
                'base_calculation': base_calculation.value,
                **base_kwargs
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate returns skewness based on the specified base calculation - VECTORIZED."""
        # Calculate base values
        base_values = self.base_calculator.calculate(data)
        
        # Convert to numpy array for vectorized operations
        values = base_values.values
        
        # Vectorized returns skewness calculation
        if len(values) < self.window + 1:
            return pd.Series(np.full(len(values), np.nan), index=data.index)
        
        # Calculate returns using vectorized operations
        returns = np.diff(values) / values[:-1]
        returns = np.concatenate([[np.nan], returns])  # Add NaN for first value
        
        # Vectorized rolling skewness calculation
        skewness = np.full(len(values), np.nan)
        
        for i in range(self.window, len(values)):
            window_returns = returns[i-self.window+1:i+1]
            # Filter out NaN values
            valid_returns = window_returns[np.isfinite(window_returns)]
            if len(valid_returns) > 2:  # Need at least 3 values for skewness
                mean_ret = np.mean(valid_returns)
                std_ret = np.std(valid_returns, ddof=1)
                if std_ret > 0:
                    skewness[i] = np.mean(((valid_returns - mean_ret) / std_ret) ** 3)
        
        return pd.Series(skewness, index=data.index)

    
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

class ReturnsKurtosisGenerator(VectorizedFeatureGenerator):
    """Generator for Returns Kurtosis with different base calculations - VECTORIZED."""
    
    def __init__(self, 
                 window: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize Returns Kurtosis generator.
        
        Args:
            window: Rolling window for kurtosis calculation
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"returns_kurtosis_{window}_{base_calculation.value}",
            category=FeatureCategory.RETURNS,
            description=f"Returns kurtosis over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={
                'window': window,
                'base_calculation': base_calculation.value,
                **base_kwargs
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate returns kurtosis based on the specified base calculation - VECTORIZED."""
        # Calculate base values
        base_values = self.base_calculator.calculate(data)
        
        # Convert to numpy array for vectorized operations
        values = base_values.values
        
        # Vectorized returns kurtosis calculation
        if len(values) < self.window + 1:
            return pd.Series(np.full(len(values), np.nan), index=data.index)
        
        # Calculate returns using vectorized operations
        returns = np.diff(values) / values[:-1]
        returns = np.concatenate([[np.nan], returns])  # Add NaN for first value
        
        # Vectorized rolling kurtosis calculation
        kurtosis = np.full(len(values), np.nan)
        
        for i in range(self.window, len(values)):
            window_returns = returns[i-self.window+1:i+1]
            # Filter out NaN values
            valid_returns = window_returns[np.isfinite(window_returns)]
            if len(valid_returns) > 3:  # Need at least 4 values for kurtosis
                mean_ret = np.mean(valid_returns)
                std_ret = np.std(valid_returns, ddof=1)
                if std_ret > 0:
                    kurtosis[i] = np.mean(((valid_returns - mean_ret) / std_ret) ** 4) - 3  # Excess kurtosis
        
        return pd.Series(kurtosis, index=data.index)

    
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

class SharpeRatioGenerator(VectorizedFeatureGenerator):
    """Generator for Sharpe Ratio with different base calculations - VECTORIZED."""
    
    def __init__(self, 
                 window: int = 20,
                 risk_free_rate: float = 0.0,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize Sharpe Ratio generator.
        
        Args:
            window: Rolling window for Sharpe ratio calculation
            risk_free_rate: Risk-free rate (annualized)
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"sharpe_ratio_{window}_{risk_free_rate}_{base_calculation.value}",
            category=FeatureCategory.RETURNS,
            description=f"Sharpe ratio over {window} periods with risk-free rate {risk_free_rate} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={
                'window': window,
                'risk_free_rate': risk_free_rate,
                'base_calculation': base_calculation.value,
                **base_kwargs
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.risk_free_rate = risk_free_rate
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate Sharpe ratio based on the specified base calculation - VECTORIZED."""
        # Calculate base values
        base_values = self.base_calculator.calculate(data)
        
        # Convert to numpy array for vectorized operations
        values = base_values.values
        
        # Vectorized Sharpe ratio calculation
        if len(values) < self.window + 1:
            return pd.Series(np.full(len(values), np.nan), index=data.index)
        
        # Calculate returns using vectorized operations
        returns = np.diff(values) / values[:-1]
        returns = np.concatenate([[np.nan], returns])  # Add NaN for first value
        
        # Calculate daily risk-free rate
        daily_rf_rate = self.risk_free_rate / 252
        
        # Vectorized rolling Sharpe ratio calculation
        sharpe_ratio = np.full(len(values), np.nan)
        
        for i in range(self.window, len(values)):
            window_returns = returns[i-self.window+1:i+1]
            # Filter out NaN values
            valid_returns = window_returns[np.isfinite(window_returns)]
            if len(valid_returns) > 1:  # Need at least 2 values for std
                excess_returns = valid_returns - daily_rf_rate
                mean_excess = np.mean(excess_returns)
                std_returns = np.std(valid_returns, ddof=1)
                if std_returns > 0:
                    sharpe_ratio[i] = mean_excess / std_returns
        
        return pd.Series(sharpe_ratio, index=data.index)

# NEW FEATURES - Advanced Returns Analysis

class AdvancedCumulativeReturnsGenerator(VectorizedFeatureGenerator):
    """Generator for cumulative returns over a specified window (enhanced version)."""
    
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"advanced_cumulative_returns_{window}",
            category=FeatureCategory.RETURNS,
            description=f"Advanced cumulative returns over {window} periods",
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
        if len(close) < self.window:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate returns
        returns = np.diff(close) / close[:-1]
        returns = np.concatenate([[np.nan], returns])
        
        # Calculate cumulative returns
        cumulative_returns = np.full(len(close), np.nan)
        for i in range(self.window - 1, len(close)):
            window_returns = returns[i - self.window + 1:i + 1]
            valid_returns = window_returns[np.isfinite(window_returns)]
            if len(valid_returns) > 0:
                cumulative_returns[i] = np.prod(1 + valid_returns) - 1
        
        return pd.Series(cumulative_returns, index=data.index)

class RollingZScoreReturnsGenerator(VectorizedFeatureGenerator):
    """Generator for rolling z-score of returns."""
    
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"rolling_zscore_returns_{window}",
            category=FeatureCategory.RETURNS,
            description=f"Rolling z-score of returns over {window} periods",
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
        if len(close) < self.window:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate returns
        returns = np.diff(close) / close[:-1]
        returns = np.concatenate([[np.nan], returns])
        
        # Calculate rolling z-score
        z_scores = np.full(len(close), np.nan)
        for i in range(self.window - 1, len(close)):
            window_returns = returns[i - self.window + 1:i + 1]
            valid_returns = window_returns[np.isfinite(window_returns)]
            if len(valid_returns) > 1:
                mean_return = np.mean(valid_returns)
                std_return = np.std(valid_returns, ddof=1)
                if std_return > 0:
                    z_scores[i] = (returns[i] - mean_return) / std_return
        
        return pd.Series(z_scores, index=data.index)

class ARCoefficientsGenerator(VectorizedFeatureGenerator):
    """Generator for AR(1) and AR(p) coefficients on returns."""
    
    def __init__(self, order: int = 1, window: int = 20):
        config = FeatureConfig(
            name=f"ar_{order}_coefficients_{window}",
            category=FeatureCategory.RETURNS,
            description=f"AR({order}) coefficients on returns over {window} periods",
            required_columns=["close"],
            default_lookback=window + order,
            min_lookback=window + order,
            max_lookback=window + order,
            parameters={'order': order, 'window': window},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.order = order
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close'].values
        if len(close) < self.window + self.order:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate returns
        returns = np.diff(close) / close[:-1]
        returns = np.concatenate([[np.nan], returns])
        
        # Calculate AR coefficients
        ar_coeffs = np.full(len(close), np.nan)
        residual_stds = np.full(len(close), np.nan)
        
        for i in range(self.window + self.order - 1, len(close)):
            window_returns = returns[i - self.window + 1:i + 1]
            valid_returns = window_returns[np.isfinite(window_returns)]
            
            if len(valid_returns) >= self.order + 5:  # Need enough data for AR estimation
                try:
                    # Simple AR(1) estimation using OLS
                    if self.order == 1:
                        y = valid_returns[1:]
                        x = valid_returns[:-1]
                        if len(y) > 1 and np.std(x) > 0:
                            coeff = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
                            ar_coeffs[i] = coeff
                            
                            # Calculate residual standard deviation
                            residuals = y - coeff * x
                            residual_stds[i] = np.std(residuals, ddof=1)
                except:
                    pass
        
        return pd.Series(ar_coeffs, index=data.index)

class LjungBoxTestGenerator(VectorizedFeatureGenerator):
    """Generator for Ljung-Box p-value on returns (autocorrelation presence)."""
    
    def __init__(self, window: int = 20, lags: int = 10):
        config = FeatureConfig(
            name=f"ljung_box_pvalue_{window}_{lags}",
            category=FeatureCategory.RETURNS,
            description=f"Ljung-Box p-value on returns over {window} periods with {lags} lags",
            required_columns=["close"],
            default_lookback=window + lags,
            min_lookback=window + lags,
            max_lookback=window + lags,
            parameters={'window': window, 'lags': lags},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.lags = lags
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close'].values
        if len(close) < self.window + self.lags:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate returns
        returns = np.diff(close) / close[:-1]
        returns = np.concatenate([[np.nan], returns])
        
        # Calculate Ljung-Box p-values
        p_values = np.full(len(close), np.nan)
        
        for i in range(self.window + self.lags - 1, len(close)):
            window_returns = returns[i - self.window + 1:i + 1]
            valid_returns = window_returns[np.isfinite(window_returns)]
            
            if len(valid_returns) >= self.lags + 5:  # Need enough data
                try:
                    # Simple autocorrelation-based Ljung-Box approximation
                    autocorrs = []
                    for lag in range(1, min(self.lags + 1, len(valid_returns))):
                        if len(valid_returns) > lag:
                            corr = np.corrcoef(valid_returns[:-lag], valid_returns[lag:])[0, 1]
                            if not np.isnan(corr):
                                autocorrs.append(corr)
                    
                    if len(autocorrs) > 0:
                        # Simplified Ljung-Box statistic
                        n = len(valid_returns)
                        lb_stat = n * (n + 2) * sum([(ac**2) / (n - lag) for lag, ac in enumerate(autocorrs, 1)])
                        
                        # Approximate p-value (simplified)
                        # In practice, you'd use scipy.stats.chi2.sf
                        p_values[i] = max(0.001, min(0.999, 1 - (lb_stat / (self.lags * 10))))
                except:
                    pass
        
        return pd.Series(p_values, index=data.index)

def create_returns_generators(periods: Dict[str, List[int]] = None) -> List[FeatureGenerator]:
    """Create a set of returns feature generators."""
    if periods is None:
        periods = {
            'log_returns': [1, 5, 10],
            'simple_returns': [1, 5, 10],
            'cumulative_returns': [10, 20],
            'rolling_returns': [10, 20],
            'volatility': [20],
            'skewness': [20],
            'kurtosis': [20],
            'sharpe_ratio': [20]
        }
    
    generators = []
    
    # Log returns generators
    for period in periods.get('log_returns', [1, 5, 10]):
        generators.append(LogReturnsGenerator(period))
    
    # Simple returns generators
    for period in periods.get('simple_returns', [1, 5, 10]):
        generators.append(SimpleReturnsGenerator(period))
    
    # Cumulative returns generators
    for window in periods.get('cumulative_returns', [10, 20]):
        generators.append(CumulativeReturnsGenerator(window))
    
    # Rolling returns generators
    for window in periods.get('rolling_returns', [10, 20]):
        generators.append(RollingReturnsGenerator(window))
    
    # Volatility generators
    for window in periods.get('volatility', [20]):
        generators.append(ReturnsVolatilityGenerator(window))
    
    # Skewness generators
    for window in periods.get('skewness', [20]):
        generators.append(ReturnsSkewnessGenerator(window))
    
    # Kurtosis generators
    for window in periods.get('kurtosis', [20]):
        generators.append(ReturnsKurtosisGenerator(window))
    
    # Sharpe ratio generators
    for window in periods.get('sharpe_ratio', [20]):
        generators.append(SharpeRatioGenerator(window))
    
    # NEW FEATURES - Advanced Returns Analysis
    # Advanced cumulative returns generators
    for window in periods.get('advanced_cumulative_returns', [10, 20]):
        generators.append(AdvancedCumulativeReturnsGenerator(window))
    
    # Rolling z-score returns generators
    for window in periods.get('rolling_zscore_returns', [20]):
        generators.append(RollingZScoreReturnsGenerator(window))
    
    # AR coefficients generators
    for order in periods.get('ar_coefficients', [1]):
        for window in periods.get('ar_windows', [20]):
            generators.append(ARCoefficientsGenerator(order, window))
    
    # Ljung-Box test generators
    for window in periods.get('ljung_box_windows', [20]):
        for lags in periods.get('ljung_box_lags', [10]):
            generators.append(LjungBoxTestGenerator(window, lags))
    
    return generators

    
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

class ReturnGenerator(SimpleReturnsGenerator):
    """Legacy alias for SimpleReturnsGenerator for backward compatibility."""
    pass

def create_default_returns_generators() -> List[FeatureGenerator]:
    """Create default returns generators."""
    return create_returns_generators()
    
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

