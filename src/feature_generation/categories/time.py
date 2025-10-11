"""
Streamlined Time Features

Focus on the most important time features:
- Hourly patterns (intraday trading patterns)
- Cyclical encodings (for machine learning compatibility)
- Intraday patterns (market open, close, lunch effects)
"""
import pandas as pd
import numpy as np
from typing import List, Optional
from ..core.feature_generator import FeatureGenerator, FeatureConfig, FeatureCategory, VectorizedFeatureGenerator

# Optimization utilities
try:
    from ..utils.vectorization_optimizer import get_vectorization_optimizer
    from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline

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
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

# Basic Hour Features
class HourGenerator(VectorizedFeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="hour",
            category=FeatureCategory.TIME,
            description="Hour of day (0-23) - captures intraday trading patterns",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        return pd.Series(data.index.hour, index=data.index)

# Cyclical Encodings for Machine Learning
    
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

class HourSinGenerator(FeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="hour_sin",
            category=FeatureCategory.TIME,
            description="Sine transformation of hour (cyclical) - ML compatible",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        hour = data.index.hour
        return pd.Series(np.sin(2 * np.pi * hour / 24), index=data.index)

    
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

class HourCosGenerator(FeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="hour_cos",
            category=FeatureCategory.TIME,
            description="Cosine transformation of hour (cyclical) - ML compatible",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        hour = data.index.hour
        return pd.Series(np.cos(2 * np.pi * hour / 24), index=data.index)

# Intraday Pattern Features
    
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

class MarketOpenGenerator(FeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="market_open",
            category=FeatureCategory.TIME,
            description="Market open indicator (first 2 hours of trading)",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        hour = data.index.hour
        # Market open: 9-11 AM (assuming 9 AM market open)
        market_open = ((hour >= 9) & (hour < 11)).astype(int)
        return pd.Series(market_open, index=data.index)

    
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

class LunchHourGenerator(FeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="lunch_hour",
            category=FeatureCategory.TIME,
            description="Lunch hour indicator (12-2 PM) - reduced activity period",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        hour = data.index.hour
        # Lunch hour: 12-2 PM
        lunch_hour = ((hour >= 12) & (hour < 14)).astype(int)
        return pd.Series(lunch_hour, index=data.index)

    
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

class MarketCloseGenerator(FeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="market_close",
            category=FeatureCategory.TIME,
            description="Market close indicator (last 2 hours of trading)",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        hour = data.index.hour
        # Market close: 3-5 PM (assuming 5 PM market close)
        market_close = ((hour >= 15) & (hour < 17)).astype(int)
        return pd.Series(market_close, index=data.index)

    
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

class AfterHoursGenerator(FeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="after_hours",
            category=FeatureCategory.TIME,
            description="After hours indicator (outside normal trading hours)",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        hour = data.index.hour
        # After hours: before 9 AM or after 5 PM
        after_hours = ((hour < 9) | (hour >= 17)).astype(int)
        return pd.Series(after_hours, index=data.index)

    
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

class HighActivityHoursGenerator(FeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="high_activity_hours",
            category=FeatureCategory.TIME,
            description="High activity hours (10 AM - 2 PM) - peak trading period",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        hour = data.index.hour
        # High activity: 10 AM - 2 PM (excluding lunch hour)
        high_activity = ((hour >= 10) & (hour < 12)) | ((hour >= 14) & (hour < 16))
        return pd.Series(high_activity.astype(int), index=data.index)

# Day of Week Cyclical Encoding (important for weekly patterns)
    
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

class DayOfWeekSinGenerator(FeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="day_of_week_sin",
            category=FeatureCategory.TIME,
            description="Sine transformation of day of week (cyclical) - weekly patterns",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        day_of_week = data.index.dayofweek
        return pd.Series(np.sin(2 * np.pi * day_of_week / 7), index=data.index)

    
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

class DayOfWeekCosGenerator(FeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="day_of_week_cos",
            category=FeatureCategory.TIME,
            description="Cosine transformation of day of week (cyclical) - weekly patterns",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        day_of_week = data.index.dayofweek
        return pd.Series(np.cos(2 * np.pi * day_of_week / 7), index=data.index)

def create_default_time_generators() -> List[FeatureGenerator]:
    """Create streamlined time feature generators focusing on hourly patterns and intraday effects."""
    return [
        # Basic hour features
        HourGenerator(),
        
        # Cyclical encodings (ML compatible)
        HourSinGenerator(),
        HourCosGenerator(),
        DayOfWeekSinGenerator(),
        DayOfWeekCosGenerator(),
        
        # Intraday pattern features
        MarketOpenGenerator(),
        LunchHourGenerator(),
        MarketCloseGenerator(),
        AfterHoursGenerator(),
        HighActivityHoursGenerator(),
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
