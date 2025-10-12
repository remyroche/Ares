"""
Streamlined Time Features with Full VectorBT Optimization

Focus on the most important time features with maximum performance:
- Hourly patterns (intraday trading patterns)
- Cyclical encodings (for machine learning compatibility)
- Intraday patterns (market open, close, lunch effects)

Optimized with VectorBTRollingOptimizer and UnifiedVectorizationManager for maximum performance.
"""
import pandas as pd
import numpy as np
import warnings
from typing import List, Optional, Dict, Any, Union
from ..core.feature_generator import FeatureGenerator, FeatureConfig, FeatureCategory, VectorizedFeatureGenerator

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

# VectorBT Rolling Optimizer
try:
    from ..utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = True
except ImportError:
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = False
    VectorBTRollingOptimizer = None
    get_vectorbt_rolling_optimizer = None

# Unified Vectorization Manager
try:
    from ...utils.ml_common.unified_vectorization_manager import UnifiedVectorizationManager, get_unified_vectorization_manager, OperationType, OptimizationStrategy
    UNIFIED_VECTORIZATION_AVAILABLE = True
except ImportError:
    UNIFIED_VECTORIZATION_AVAILABLE = False
    UnifiedVectorizationManager = None
    get_unified_vectorization_manager = None

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# Base class for optimized time features
class OptimizedTimeFeatureGenerator(VectorizedFeatureGenerator):
    """
    Base class for time features with full VectorBT optimization.
    
    Provides optimized rolling operations, unified vectorization management,
    and intelligent performance optimization.
    """
    
    def __init__(self, config: FeatureConfig, enable_vectorbt: bool = True, enable_unified_vectorization: bool = True):
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        self.enable_vectorbt = enable_vectorbt and VECTORBT_AVAILABLE
        self.enable_unified_vectorization = enable_unified_vectorization and UNIFIED_VECTORIZATION_AVAILABLE
        
        # Initialize optimizers
        self.rolling_optimizer = None
        self.unified_manager = None
        
        if self.enable_vectorbt and VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=CUPY_AVAILABLE, enable_parallel=True)
        
        if self.enable_unified_vectorization and UNIFIED_VECTORIZATION_AVAILABLE:
            self.unified_manager = get_unified_vectorization_manager()
        
        # Performance tracking
        self.performance_stats = {
            'vectorbt_operations': 0,
            'unified_vectorization_operations': 0,
            'optimization_operations': 0,
            'total_operations': 0
        }
    
    def _optimized_rolling_operation(self, data: Union[pd.Series, pd.DataFrame], 
                                   operation: str, window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """
        Perform optimized rolling operation using the best available method.
        
        Args:
            data: Input data
            operation: Operation type ('mean', 'std', 'var', 'min', 'max', 'sum', 'corr', 'cov')
            window: Rolling window size
            **kwargs: Additional parameters
            
        Returns:
            Result of rolling operation
        """
        self.performance_stats['total_operations'] += 1
        
        # Try VectorBT rolling optimizer first
        if self.rolling_optimizer and self.enable_vectorbt:
            try:
                result = getattr(self.rolling_optimizer, f'rolling_{operation}')(data, window, **kwargs)
                self.performance_stats['vectorbt_operations'] += 1
                return result
            except Exception as e:
                warnings.warn(f"VectorBT rolling operation failed: {e}, using fallback")
        
        # Try unified vectorization manager
        if self.unified_manager and self.enable_unified_vectorization:
            try:
                # Create operation config
                from ...utils.ml_common.unified_vectorization_manager import OperationConfig
                op_config = OperationConfig(
                    operation_type=OperationType.TECHNICAL_INDICATORS,
                    data_size=len(data),
                    data_dimensions=data.shape if hasattr(data, 'shape') else (len(data),),
                    memory_budget_mb=1024.0,
                    time_budget_seconds=60.0
                )
                
                # Execute with unified manager
                result = self.unified_manager.execute_rolling_operation(
                    data, operation, window, op_config, **kwargs
                )
                self.performance_stats['unified_vectorization_operations'] += 1
                return result
            except Exception as e:
                warnings.warn(f"Unified vectorization failed: {e}, using fallback")
        
        # Fallback to pandas
        return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: Union[pd.Series, pd.DataFrame], 
                                operation: str, window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Fallback rolling operation using pandas."""
        rolling_obj = data.rolling(window=window, **kwargs)
        
        if operation == 'mean':
            return rolling_obj.mean()
        elif operation == 'std':
            return rolling_obj.std()
        elif operation == 'var':
            return rolling_obj.var()
        elif operation == 'min':
            return rolling_obj.min()
        elif operation == 'max':
            return rolling_obj.max()
        elif operation == 'sum':
            return rolling_obj.sum()
        elif operation == 'corr':
            other = kwargs.get('other')
            return rolling_obj.corr(other)
        elif operation == 'cov':
            other = kwargs.get('other')
            return rolling_obj.cov(other)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _optimized_vectorized_operation(self, data: pd.DataFrame, operation: str, **kwargs) -> pd.DataFrame:
        """
        Perform optimized vectorized operation using unified vectorization manager.
        
        Args:
            data: Input DataFrame
            operation: Operation type
            **kwargs: Additional parameters
            
        Returns:
            Result DataFrame
        """
        if not self.unified_manager or not self.enable_unified_vectorization:
            return data
        
        try:
            from ...utils.ml_common.unified_vectorization_manager import OperationConfig
            op_config = OperationConfig(
                operation_type=OperationType.FEATURE_ENGINEERING,
                data_size=len(data),
                data_dimensions=data.shape,
                memory_budget_mb=1024.0,
                time_budget_seconds=60.0
            )
            
            result = self.unified_manager.execute_vectorized_operation(
                data, operation, op_config, **kwargs
            )
            self.performance_stats['unified_vectorization_operations'] += 1
            return result
        except Exception as e:
            warnings.warn(f"Unified vectorization operation failed: {e}")
            return data
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'vectorbt_operations': 0,
            'unified_vectorization_operations': 0,
            'optimization_operations': 0,
            'total_operations': 0
        }

# Basic Hour Features
class HourGenerator(OptimizedTimeFeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="hour",
            category=FeatureCategory.TIME,
            description="Hour of day (0-23) - captures intraday trading patterns with VectorBT optimization",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config, enable_vectorbt=True, enable_unified_vectorization=True)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate hour feature with VectorBT optimization."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name='hour')
        
        # Use optimized vectorized operation for large datasets
        if len(data) > 1000 and self.unified_manager:
            try:
                # Convert to DataFrame for unified processing
                hour_data = pd.DataFrame({'hour': data.index.hour}, index=data.index)
                result = self._optimized_vectorized_operation(hour_data, 'extract_hour')
                return result['hour'] if 'hour' in result.columns else pd.Series(data.index.hour, index=data.index)
            except Exception as e:
                warnings.warn(f"Optimized hour extraction failed: {e}, using direct method")
        
        # Direct method for smaller datasets or fallback
        return pd.Series(data.index.hour, index=data.index, name='hour')

# Cyclical Encodings for Machine Learning
class HourSinGenerator(OptimizedTimeFeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="hour_sin",
            category=FeatureCategory.TIME,
            description="Sine transformation of hour (cyclical) - ML compatible with VectorBT optimization",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config, enable_vectorbt=True, enable_unified_vectorization=True)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate hour sine feature with VectorBT optimization."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name='hour_sin')
        
        hour = data.index.hour
        
        # Use VectorBT for large datasets
        if len(data) > 1000 and VECTORBT_AVAILABLE:
            try:
                # Convert to VectorBT array for optimized computation
                hour_array = vbt.array_wrapper(hour, freq=data.index.freq if hasattr(data.index, 'freq') else None)
                result = np.sin(2 * np.pi * hour_array / 24)
                return pd.Series(result, index=data.index, name='hour_sin')
            except Exception as e:
                warnings.warn(f"VectorBT sine computation failed: {e}, using numpy")
        
        # Fallback to numpy
        return pd.Series(np.sin(2 * np.pi * hour / 24), index=data.index, name='hour_sin')

class HourCosGenerator(OptimizedTimeFeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="hour_cos",
            category=FeatureCategory.TIME,
            description="Cosine transformation of hour (cyclical) - ML compatible with VectorBT optimization",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config, enable_vectorbt=True, enable_unified_vectorization=True)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate hour cosine feature with VectorBT optimization."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name='hour_cos')
        
        hour = data.index.hour
        
        # Use VectorBT for large datasets
        if len(data) > 1000 and VECTORBT_AVAILABLE:
            try:
                # Convert to VectorBT array for optimized computation
                hour_array = vbt.array_wrapper(hour, freq=data.index.freq if hasattr(data.index, 'freq') else None)
                result = np.cos(2 * np.pi * hour_array / 24)
                return pd.Series(result, index=data.index, name='hour_cos')
            except Exception as e:
                warnings.warn(f"VectorBT cosine computation failed: {e}, using numpy")
        
        # Fallback to numpy
        return pd.Series(np.cos(2 * np.pi * hour / 24), index=data.index, name='hour_cos')

# Intraday Pattern Features
class MarketOpenGenerator(OptimizedTimeFeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="market_open",
            category=FeatureCategory.TIME,
            description="Market open indicator (first 2 hours of trading) with VectorBT optimization",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config, enable_vectorbt=True, enable_unified_vectorization=True)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate market open indicator with VectorBT optimization."""
        if data.empty:
            return pd.Series(dtype=int, index=data.index, name='market_open')
        
        hour = data.index.hour
        
        # Use VectorBT for large datasets
        if len(data) > 1000 and VECTORBT_AVAILABLE:
            try:
                # Convert to VectorBT array for optimized boolean operations
                hour_array = vbt.array_wrapper(hour, freq=data.index.freq if hasattr(data.index, 'freq') else None)
                # Market open: 9-11 AM (assuming 9 AM market open)
                market_open = ((hour_array >= 9) & (hour_array < 11)).astype(int)
                return pd.Series(market_open, index=data.index, name='market_open')
            except Exception as e:
                warnings.warn(f"VectorBT market open computation failed: {e}, using numpy")
        
        # Fallback to numpy
        market_open = ((hour >= 9) & (hour < 11)).astype(int)
        return pd.Series(market_open, index=data.index, name='market_open')

class LunchHourGenerator(OptimizedTimeFeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="lunch_hour",
            category=FeatureCategory.TIME,
            description="Lunch hour indicator (12-2 PM) - reduced activity period with VectorBT optimization",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config, enable_vectorbt=True, enable_unified_vectorization=True)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate lunch hour indicator with VectorBT optimization."""
        if data.empty:
            return pd.Series(dtype=int, index=data.index, name='lunch_hour')
        
        hour = data.index.hour
        
        # Use VectorBT for large datasets
        if len(data) > 1000 and VECTORBT_AVAILABLE:
            try:
                # Convert to VectorBT array for optimized boolean operations
                hour_array = vbt.array_wrapper(hour, freq=data.index.freq if hasattr(data.index, 'freq') else None)
                # Lunch hour: 12-2 PM
                lunch_hour = ((hour_array >= 12) & (hour_array < 14)).astype(int)
                return pd.Series(lunch_hour, index=data.index, name='lunch_hour')
            except Exception as e:
                warnings.warn(f"VectorBT lunch hour computation failed: {e}, using numpy")
        
        # Fallback to numpy
        lunch_hour = ((hour >= 12) & (hour < 14)).astype(int)
        return pd.Series(lunch_hour, index=data.index, name='lunch_hour')

class MarketCloseGenerator(OptimizedTimeFeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="market_close",
            category=FeatureCategory.TIME,
            description="Market close indicator (last 2 hours of trading) with VectorBT optimization",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config, enable_vectorbt=True, enable_unified_vectorization=True)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate market close indicator with VectorBT optimization."""
        if data.empty:
            return pd.Series(dtype=int, index=data.index, name='market_close')
        
        hour = data.index.hour
        
        # Use VectorBT for large datasets
        if len(data) > 1000 and VECTORBT_AVAILABLE:
            try:
                # Convert to VectorBT array for optimized boolean operations
                hour_array = vbt.array_wrapper(hour, freq=data.index.freq if hasattr(data.index, 'freq') else None)
                # Market close: 3-5 PM (assuming 5 PM market close)
                market_close = ((hour_array >= 15) & (hour_array < 17)).astype(int)
                return pd.Series(market_close, index=data.index, name='market_close')
            except Exception as e:
                warnings.warn(f"VectorBT market close computation failed: {e}, using numpy")
        
        # Fallback to numpy
        market_close = ((hour >= 15) & (hour < 17)).astype(int)
        return pd.Series(market_close, index=data.index, name='market_close')

class AfterHoursGenerator(OptimizedTimeFeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="after_hours",
            category=FeatureCategory.TIME,
            description="After hours indicator (outside normal trading hours) with VectorBT optimization",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config, enable_vectorbt=True, enable_unified_vectorization=True)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate after hours indicator with VectorBT optimization."""
        if data.empty:
            return pd.Series(dtype=int, index=data.index, name='after_hours')
        
        hour = data.index.hour
        
        # Use VectorBT for large datasets
        if len(data) > 1000 and VECTORBT_AVAILABLE:
            try:
                # Convert to VectorBT array for optimized boolean operations
                hour_array = vbt.array_wrapper(hour, freq=data.index.freq if hasattr(data.index, 'freq') else None)
                # After hours: before 9 AM or after 5 PM
                after_hours = ((hour_array < 9) | (hour_array >= 17)).astype(int)
                return pd.Series(after_hours, index=data.index, name='after_hours')
            except Exception as e:
                warnings.warn(f"VectorBT after hours computation failed: {e}, using numpy")
        
        # Fallback to numpy
        after_hours = ((hour < 9) | (hour >= 17)).astype(int)
        return pd.Series(after_hours, index=data.index, name='after_hours')

class HighActivityHoursGenerator(OptimizedTimeFeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="high_activity_hours",
            category=FeatureCategory.TIME,
            description="High activity hours (10 AM - 2 PM) - peak trading period with VectorBT optimization",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config, enable_vectorbt=True, enable_unified_vectorization=True)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate high activity hours indicator with VectorBT optimization."""
        if data.empty:
            return pd.Series(dtype=int, index=data.index, name='high_activity_hours')
        
        hour = data.index.hour
        
        # Use VectorBT for large datasets
        if len(data) > 1000 and VECTORBT_AVAILABLE:
            try:
                # Convert to VectorBT array for optimized boolean operations
                hour_array = vbt.array_wrapper(hour, freq=data.index.freq if hasattr(data.index, 'freq') else None)
                # High activity: 10 AM - 2 PM (excluding lunch hour)
                high_activity = ((hour_array >= 10) & (hour_array < 12)) | ((hour_array >= 14) & (hour_array < 16))
                return pd.Series(high_activity.astype(int), index=data.index, name='high_activity_hours')
            except Exception as e:
                warnings.warn(f"VectorBT high activity computation failed: {e}, using numpy")
        
        # Fallback to numpy
        high_activity = ((hour >= 10) & (hour < 12)) | ((hour >= 14) & (hour < 16))
        return pd.Series(high_activity.astype(int), index=data.index, name='high_activity_hours')

# Day of Week Cyclical Encoding (important for weekly patterns)
class DayOfWeekSinGenerator(OptimizedTimeFeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="day_of_week_sin",
            category=FeatureCategory.TIME,
            description="Sine transformation of day of week (cyclical) - weekly patterns with VectorBT optimization",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config, enable_vectorbt=True, enable_unified_vectorization=True)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate day of week sine feature with VectorBT optimization."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name='day_of_week_sin')
        
        day_of_week = data.index.dayofweek
        
        # Use VectorBT for large datasets
        if len(data) > 1000 and VECTORBT_AVAILABLE:
            try:
                # Convert to VectorBT array for optimized computation
                dow_array = vbt.array_wrapper(day_of_week, freq=data.index.freq if hasattr(data.index, 'freq') else None)
                result = np.sin(2 * np.pi * dow_array / 7)
                return pd.Series(result, index=data.index, name='day_of_week_sin')
            except Exception as e:
                warnings.warn(f"VectorBT day of week sine computation failed: {e}, using numpy")
        
        # Fallback to numpy
        return pd.Series(np.sin(2 * np.pi * day_of_week / 7), index=data.index, name='day_of_week_sin')

class DayOfWeekCosGenerator(OptimizedTimeFeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="day_of_week_cos",
            category=FeatureCategory.TIME,
            description="Cosine transformation of day of week (cyclical) - weekly patterns with VectorBT optimization",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config, enable_vectorbt=True, enable_unified_vectorization=True)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate day of week cosine feature with VectorBT optimization."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name='day_of_week_cos')
        
        day_of_week = data.index.dayofweek
        
        # Use VectorBT for large datasets
        if len(data) > 1000 and VECTORBT_AVAILABLE:
            try:
                # Convert to VectorBT array for optimized computation
                dow_array = vbt.array_wrapper(day_of_week, freq=data.index.freq if hasattr(data.index, 'freq') else None)
                result = np.cos(2 * np.pi * dow_array / 7)
                return pd.Series(result, index=data.index, name='day_of_week_cos')
            except Exception as e:
                warnings.warn(f"VectorBT day of week cosine computation failed: {e}, using numpy")
        
        # Fallback to numpy
        return pd.Series(np.cos(2 * np.pi * day_of_week / 7), index=data.index, name='day_of_week_cos')

# Additional optimized time features
class TimeOfDayGenerator(OptimizedTimeFeatureGenerator):
    """Generate time of day as continuous feature (0-1) with VectorBT optimization."""
    
    def __init__(self):
        config = FeatureConfig(
            name="time_of_day",
            category=FeatureCategory.TIME,
            description="Time of day as continuous feature (0-1) with VectorBT optimization",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config, enable_vectorbt=True, enable_unified_vectorization=True)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate time of day feature with VectorBT optimization."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name='time_of_day')
        
        # Calculate time of day as fraction of day
        time_of_day = (data.index.hour * 3600 + data.index.minute * 60 + data.index.second) / 86400
        
        # Use VectorBT for large datasets
        if len(data) > 1000 and VECTORBT_AVAILABLE:
            try:
                # Convert to VectorBT array for optimized computation
                tod_array = vbt.array_wrapper(time_of_day, freq=data.index.freq if hasattr(data.index, 'freq') else None)
                return pd.Series(tod_array, index=data.index, name='time_of_day')
            except Exception as e:
                warnings.warn(f"VectorBT time of day computation failed: {e}, using numpy")
        
        # Fallback to numpy
        return pd.Series(time_of_day, index=data.index, name='time_of_day')

class WeekdayGenerator(OptimizedTimeFeatureGenerator):
    """Generate weekday indicator (1-7) with VectorBT optimization."""
    
    def __init__(self):
        config = FeatureConfig(
            name="weekday",
            category=FeatureCategory.TIME,
            description="Weekday indicator (1-7) with VectorBT optimization",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config, enable_vectorbt=True, enable_unified_vectorization=True)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate weekday feature with VectorBT optimization."""
        if data.empty:
            return pd.Series(dtype=int, index=data.index, name='weekday')
        
        weekday = data.index.dayofweek + 1  # Convert 0-6 to 1-7
        
        # Use VectorBT for large datasets
        if len(data) > 1000 and VECTORBT_AVAILABLE:
            try:
                # Convert to VectorBT array for optimized computation
                wd_array = vbt.array_wrapper(weekday, freq=data.index.freq if hasattr(data.index, 'freq') else None)
                return pd.Series(wd_array, index=data.index, name='weekday')
            except Exception as e:
                warnings.warn(f"VectorBT weekday computation failed: {e}, using numpy")
        
        # Fallback to numpy
        return pd.Series(weekday, index=data.index, name='weekday')

def create_default_time_generators() -> List[OptimizedTimeFeatureGenerator]:
    """Create streamlined time feature generators with full VectorBT optimization."""
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
        
        # Additional optimized features
        TimeOfDayGenerator(),
        WeekdayGenerator(),
    ]

def create_optimized_time_generators(enable_vectorbt: bool = True, 
                                   enable_unified_vectorization: bool = True) -> List[OptimizedTimeFeatureGenerator]:
    """Create time feature generators with specified optimization settings."""
    generators = []
    
    # Basic hour features
    generators.append(HourGenerator())
    
    # Cyclical encodings
    generators.append(HourSinGenerator())
    generators.append(HourCosGenerator())
    generators.append(DayOfWeekSinGenerator())
    generators.append(DayOfWeekCosGenerator())
    
    # Intraday pattern features
    generators.append(MarketOpenGenerator())
    generators.append(LunchHourGenerator())
    generators.append(MarketCloseGenerator())
    generators.append(AfterHoursGenerator())
    generators.append(HighActivityHoursGenerator())
    
    # Additional optimized features
    generators.append(TimeOfDayGenerator())
    generators.append(WeekdayGenerator())
    
    return generators

def get_time_feature_performance_stats(generators: List[OptimizedTimeFeatureGenerator]) -> Dict[str, Any]:
    """Get aggregated performance statistics from time feature generators."""
    total_stats = {
        'vectorbt_operations': 0,
        'unified_vectorization_operations': 0,
        'optimization_operations': 0,
        'total_operations': 0
    }
    
    for generator in generators:
        stats = generator.get_performance_stats()
        for key in total_stats:
            total_stats[key] += stats.get(key, 0)
    
    return total_stats
