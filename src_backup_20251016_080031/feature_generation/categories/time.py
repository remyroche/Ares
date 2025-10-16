"""
Streamlined Time Features with Full VectorBT Optimization

Focus on the most important time features with native VectorBT integration:
- Hourly patterns (intraday trading patterns)
- Cyclical encodings (for machine learning compatibility)
- Intraday patterns (market open, close, lunch effects)
- Advanced time-based rolling features using VectorBTRollingOptimizer
- Unified vectorization using UnifiedVectorizationManager
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import warnings
from ..core.feature_generator import FeatureGenerator, FeatureConfig, FeatureCategory, VectorizedFeatureGenerator

# VectorBT imports for native optimization - no longer using features_common.utils to avoid circular imports
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

# Import VectorBT optimization utilities
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
    from src.feature_generation.utils.unified_vectorization_manager import get_unified_vectorization_manager, VectorizationConfig
    VECTORBT_OPTIMIZATION_AVAILABLE = True
except ImportError:
    VECTORBT_OPTIMIZATION_AVAILABLE = False
    get_vectorbt_rolling_optimizer = None
    get_unified_vectorization_manager = None
    VectorizationConfig = None

except ImportError:
    
    cp = None

# Base class for VectorBT-optimized time features
class VectorBTTimeFeatureGenerator(VectorizedFeatureGenerator):
    """
    Base class for time features with full VectorBT optimization.
    
    Provides unified VectorBT integration, performance monitoring,
    and memory optimization for all time-based features.
    """
    
    def __init__(self, config: FeatureConfig):
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        # Initialize VectorBT optimization components
        self.vectorbt_rolling_optimizer = None
        self.unified_vectorization_manager = None
        
        if VECTORBT_OPTIMIZATION_AVAILABLE:
            try:
                self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer()
                self.unified_vectorization_manager = get_unified_vectorization_manager()
            except Exception as e:
                self.logger.warning(f"Failed to initialize VectorBT optimization: {e}")
    
    def _optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for VectorBT processing."""
        if self.unified_vectorization_manager:
            return self.unified_vectorization_manager.optimize_dataframe(data)
        return data
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with optimization."""
        if self.vectorbt_rolling_optimizer:
            return self.vectorbt_rolling_optimizer._rolling_operation(data, operation, window, **kwargs)
        elif VECTORBT_AVAILABLE:
            return self._direct_vectorbt_operation(data, operation, window, **kwargs)
        else:
            return self._pandas_fallback_operation(data, operation, window, **kwargs)
    
    def _direct_vectorbt_operation(self, data: pd.Series, operation: str,
                                 window: int, **kwargs) -> pd.Series:
        """Direct VectorBT operation fallback."""
        if not VECTORBT_AVAILABLE:
            raise RuntimeError("VectorBT not available for direct operations")

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
            raise ValueError(f"Unsupported VectorBT operation: {operation}")
    
    def _pandas_fallback_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Pandas fallback operation."""
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
        else:
            raise ValueError(f"Unsupported pandas operation: {operation}")
    
    def _vectorbt_scale_operation(self, data: pd.Series, method: str, **kwargs) -> pd.Series:
        """Perform VectorBT scaling operation."""
        if self.unified_vectorization_manager:
            return self.unified_vectorization_manager.scale_data(data, method, **kwargs)
        elif VECTORBT_AVAILABLE:
            if method == 'zscore':
                return zscore(data, **kwargs)
            elif method == 'minmax':
                return scale(data, method='minmax', **kwargs)
            elif method == 'rank':
                return rank(data, **kwargs)
            else:
                return self._pandas_scale_fallback(data, method, **kwargs)
        else:
            return self._pandas_scale_fallback(data, method, **kwargs)
    
    def _pandas_scale_fallback(self, data: pd.Series, method: str, **kwargs) -> pd.Series:
        """Pandas scaling fallback."""
        if method == 'zscore':
            return (data - data.mean()) / data.std()
        elif method == 'minmax':
            return (data - data.min()) / (data.max() - data.min())
        elif method == 'rank':
            return data.rank()
        else:
            raise ValueError(f"Unsupported scaling method: {method}")

# Basic Hour Features
class HourGenerator(VectorBTTimeFeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="hour",
            category=FeatureCategory.TIME,
            description="Hour of day (0-23) - captures intraday trading patterns with VectorBT optimization",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            use_vectorbt=True,
            enable_parallel=True
        )
        super().__init__(config)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for VectorBT processing
        data = self._optimize_dataframe_processing(data)
        
        # Generate hour feature with VectorBT optimization
        hour_series = pd.Series(data.index.hour, index=data.index)
        
        # Apply VectorBT scaling if requested
        scale_method = kwargs.get('scale_method')
        if scale_method:
            hour_series = self._vectorbt_scale_operation(hour_series, scale_method)
        
        return hour_series

# Cyclical Encodings for Machine Learning
class HourSinGenerator(VectorBTTimeFeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="hour_sin",
            category=FeatureCategory.TIME,
            description="Sine transformation of hour (cyclical) - ML compatible with VectorBT optimization",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            use_vectorbt=True,
            enable_parallel=True
        )
        super().__init__(config)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for VectorBT processing
        data = self._optimize_dataframe_processing(data)
        
        # Generate cyclical hour feature with VectorBT optimization
        hour = data.index.hour
        sin_hour = np.sin(2 * np.pi * hour / 24)
        
        # Apply VectorBT scaling if requested
        scale_method = kwargs.get('scale_method')
        if scale_method:
            sin_hour = self._vectorbt_scale_operation(pd.Series(sin_hour, index=data.index), scale_method)
            return sin_hour
        
        return pd.Series(sin_hour, index=data.index)

class HourCosGenerator(VectorBTTimeFeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="hour_cos",
            category=FeatureCategory.TIME,
            description="Cosine transformation of hour (cyclical) - ML compatible with VectorBT optimization",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            use_vectorbt=True,
            enable_parallel=True
        )
        super().__init__(config)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for VectorBT processing
        data = self._optimize_dataframe_processing(data)
        
        # Generate cyclical hour feature with VectorBT optimization
        hour = data.index.hour
        cos_hour = np.cos(2 * np.pi * hour / 24)
        
        # Apply VectorBT scaling if requested
        scale_method = kwargs.get('scale_method')
        if scale_method:
            cos_hour = self._vectorbt_scale_operation(pd.Series(cos_hour, index=data.index), scale_method)
            return cos_hour
        
        return pd.Series(cos_hour, index=data.index)

# Intraday Pattern Features
class MarketOpenGenerator(VectorBTTimeFeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="market_open",
            category=FeatureCategory.TIME,
            description="Market open indicator (first 2 hours of trading) with VectorBT optimization",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            use_vectorbt=True,
            enable_parallel=True
        )
        super().__init__(config)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for VectorBT processing
        data = self._optimize_dataframe_processing(data)
        
        # Generate market open feature with VectorBT optimization
        hour = data.index.hour
        # Market open: 9-11 AM (assuming 9 AM market open)
        market_open = ((hour >= 9) & (hour < 11)).astype(int)
        
        # Apply VectorBT scaling if requested
        scale_method = kwargs.get('scale_method')
        if scale_method:
            market_open = self._vectorbt_scale_operation(pd.Series(market_open, index=data.index), scale_method)
            return market_open
        
        return pd.Series(market_open, index=data.index)

class LunchHourGenerator(VectorBTTimeFeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="lunch_hour",
            category=FeatureCategory.TIME,
            description="Lunch hour indicator (12-2 PM) - reduced activity period with VectorBT optimization",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            use_vectorbt=True,
            enable_parallel=True
        )
        super().__init__(config)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for VectorBT processing
        data = self._optimize_dataframe_processing(data)
        
        # Generate lunch hour feature with VectorBT optimization
        hour = data.index.hour
        # Lunch hour: 12-2 PM
        lunch_hour = ((hour >= 12) & (hour < 14)).astype(int)
        
        # Apply VectorBT scaling if requested
        scale_method = kwargs.get('scale_method')
        if scale_method:
            lunch_hour = self._vectorbt_scale_operation(pd.Series(lunch_hour, index=data.index), scale_method)
            return lunch_hour
        
        return pd.Series(lunch_hour, index=data.index)

class MarketCloseGenerator(VectorBTTimeFeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="market_close",
            category=FeatureCategory.TIME,
            description="Market close indicator (last 2 hours of trading) with VectorBT optimization",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            use_vectorbt=True,
            enable_parallel=True
        )
        super().__init__(config)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for VectorBT processing
        data = self._optimize_dataframe_processing(data)
        
        # Generate market close feature with VectorBT optimization
        hour = data.index.hour
        # Market close: 3-5 PM (assuming 5 PM market close)
        market_close = ((hour >= 15) & (hour < 17)).astype(int)
        
        # Apply VectorBT scaling if requested
        scale_method = kwargs.get('scale_method')
        if scale_method:
            market_close = self._vectorbt_scale_operation(pd.Series(market_close, index=data.index), scale_method)
            return market_close
        
        return pd.Series(market_close, index=data.index)

class AfterHoursGenerator(VectorBTTimeFeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="after_hours",
            category=FeatureCategory.TIME,
            description="After hours indicator (outside normal trading hours) with VectorBT optimization",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            use_vectorbt=True,
            enable_parallel=True
        )
        super().__init__(config)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for VectorBT processing
        data = self._optimize_dataframe_processing(data)
        
        # Generate after hours feature with VectorBT optimization
        hour = data.index.hour
        # After hours: before 9 AM or after 5 PM
        after_hours = ((hour < 9) | (hour >= 17)).astype(int)
        
        # Apply VectorBT scaling if requested
        scale_method = kwargs.get('scale_method')
        if scale_method:
            after_hours = self._vectorbt_scale_operation(pd.Series(after_hours, index=data.index), scale_method)
            return after_hours
        
        return pd.Series(after_hours, index=data.index)

class HighActivityHoursGenerator(VectorBTTimeFeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="high_activity_hours",
            category=FeatureCategory.TIME,
            description="High activity hours (10 AM - 2 PM) - peak trading period with VectorBT optimization",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            use_vectorbt=True,
            enable_parallel=True
        )
        super().__init__(config)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for VectorBT processing
        data = self._optimize_dataframe_processing(data)
        
        # Generate high activity feature with VectorBT optimization
        hour = data.index.hour
        # High activity: 10 AM - 2 PM (excluding lunch hour)
        high_activity = ((hour >= 10) & (hour < 12)) | ((hour >= 14) & (hour < 16))
        
        # Apply VectorBT scaling if requested
        scale_method = kwargs.get('scale_method')
        if scale_method:
            high_activity = self._vectorbt_scale_operation(pd.Series(high_activity.astype(int), index=data.index), scale_method)
            return high_activity
        
        return pd.Series(high_activity.astype(int), index=data.index)

# Day of Week Cyclical Encoding (important for weekly patterns)
class DayOfWeekSinGenerator(VectorBTTimeFeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="day_of_week_sin",
            category=FeatureCategory.TIME,
            description="Sine transformation of day of week (cyclical) - weekly patterns with VectorBT optimization",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            use_vectorbt=True,
            enable_parallel=True
        )
        super().__init__(config)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for VectorBT processing
        data = self._optimize_dataframe_processing(data)
        
        # Generate cyclical day of week feature with VectorBT optimization
        day_of_week = data.index.dayofweek
        sin_dow = np.sin(2 * np.pi * day_of_week / 7)
        
        # Apply VectorBT scaling if requested
        scale_method = kwargs.get('scale_method')
        if scale_method:
            sin_dow = self._vectorbt_scale_operation(pd.Series(sin_dow, index=data.index), scale_method)
            return sin_dow
        
        return pd.Series(sin_dow, index=data.index)

class DayOfWeekCosGenerator(VectorBTTimeFeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="day_of_week_cos",
            category=FeatureCategory.TIME,
            description="Cosine transformation of day of week (cyclical) - weekly patterns with VectorBT optimization",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            use_vectorbt=True,
            enable_parallel=True
        )
        super().__init__(config)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for VectorBT processing
        data = self._optimize_dataframe_processing(data)
        
        # Generate cyclical day of week feature with VectorBT optimization
        day_of_week = data.index.dayofweek
        cos_dow = np.cos(2 * np.pi * day_of_week / 7)
        
        # Apply VectorBT scaling if requested
        scale_method = kwargs.get('scale_method')
        if scale_method:
            cos_dow = self._vectorbt_scale_operation(pd.Series(cos_dow, index=data.index), scale_method)
            return cos_dow
        
        return pd.Series(cos_dow, index=data.index)

# Advanced Time-Based Features with VectorBT Rolling Operations
class HourlyVolatilityGenerator(VectorBTTimeFeatureGenerator):
    """Generate hourly volatility patterns using VectorBT rolling operations."""
    
    def __init__(self):
        config = FeatureConfig(
            name="hourly_volatility",
            category=FeatureCategory.TIME,
            description="Hourly volatility patterns using VectorBT rolling operations",
            required_columns=['close'],
            default_lookback=20,
            min_lookback=5,
            max_lookback=100,
            use_vectorbt=True,
            enable_parallel=True
        )
        super().__init__(config)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for VectorBT processing
        data = self._optimize_dataframe_processing(data)
        
        # Calculate rolling volatility by hour using VectorBT
        window = kwargs.get('window', self.config.default_lookback)
        close_prices = data['close']
        
        # Use VectorBT rolling operations for volatility calculation
        rolling_std = self._vectorbt_rolling_operation(close_prices, 'std', window)
        
        # Group by hour and calculate mean volatility
        hourly_vol = rolling_std.groupby(data.index.hour).mean()
        
        # Map back to original index
        result = data.index.hour.map(hourly_vol)
        
        return pd.Series(result, index=data.index)

class TimeBasedMomentumGenerator(VectorBTTimeFeatureGenerator):
    """Generate time-based momentum features using VectorBT optimization."""
    
    def __init__(self):
        config = FeatureConfig(
            name="time_momentum",
            category=FeatureCategory.TIME,
            description="Time-based momentum using VectorBT rolling operations",
            required_columns=['close'],
            default_lookback=20,
            min_lookback=5,
            max_lookback=100,
            use_vectorbt=True,
            enable_parallel=True
        )
        super().__init__(config)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for VectorBT processing
        data = self._optimize_dataframe_processing(data)
        
        # Calculate time-based momentum using VectorBT
        window = kwargs.get('window', self.config.default_lookback)
        close_prices = data['close']
        
        # Calculate rolling mean and momentum
        rolling_mean = self._vectorbt_rolling_operation(close_prices, 'mean', window)
        momentum = (close_prices - rolling_mean) / rolling_mean
        
        # Apply time-based scaling
        hour = data.index.hour
        time_factor = np.where((hour >= 9) & (hour <= 16), 1.0, 0.5)  # Higher weight during trading hours
        
        result = momentum * time_factor
        
        # Apply VectorBT scaling if requested
        scale_method = kwargs.get('scale_method')
        if scale_method:
            result = self._vectorbt_scale_operation(pd.Series(result, index=data.index), scale_method)
            return result
        
        return pd.Series(result, index=data.index)

class TimeBasedVolumeProfileGenerator(VectorBTTimeFeatureGenerator):
    """Generate time-based volume profile using VectorBT batch processing."""
    
    def __init__(self):
        config = FeatureConfig(
            name="time_volume_profile",
            category=FeatureCategory.TIME,
            description="Time-based volume profile using VectorBT batch processing",
            required_columns=['volume'],
            default_lookback=20,
            min_lookback=5,
            max_lookback=100,
            use_vectorbt=True,
            enable_parallel=True
        )
        super().__init__(config)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for VectorBT processing
        data = self._optimize_dataframe_processing(data)
        
        # Use UnifiedVectorizationManager for batch processing
        if self.unified_vectorization_manager:
            # Define feature configurations for batch processing
            feature_configs = [
                {
                    'name': 'volume_mean',
                    'type': 'rolling',
                    'params': {'operation': 'mean', 'window': self.config.default_lookback, 'column': 'volume'}
                },
                {
                    'name': 'volume_std',
                    'type': 'rolling',
                    'params': {'operation': 'std', 'window': self.config.default_lookback, 'column': 'volume'}
                }
            ]
            
            # Process features in batch
            features = self.unified_vectorization_manager.batch_process_features(data, feature_configs)
            
            # Calculate volume profile (normalized volume)
            volume_profile = (data['volume'] - features['volume_mean']) / features['volume_std']
            
            # Apply time-based weighting
            hour = data.index.hour
            time_weight = np.where((hour >= 9) & (hour <= 16), 1.0, 0.3)  # Higher weight during trading hours
            
            result = volume_profile * time_weight
            
            # Apply VectorBT scaling if requested
            scale_method = kwargs.get('scale_method')
            if scale_method:
                result = self._vectorbt_scale_operation(pd.Series(result, index=data.index), scale_method)
                return result
            
            return pd.Series(result, index=data.index)
        else:
            # Fallback to simple calculation
            volume = data['volume']
            rolling_mean = self._vectorbt_rolling_operation(volume, 'mean', self.config.default_lookback)
            rolling_std = self._vectorbt_rolling_operation(volume, 'std', self.config.default_lookback)
            
            result = (volume - rolling_mean) / rolling_std
            
            return pd.Series(result, index=data.index)

def create_default_time_generators() -> List[FeatureGenerator]:
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
        
        # Advanced VectorBT-optimized features
        HourlyVolatilityGenerator(),
        TimeBasedMomentumGenerator(),
        TimeBasedVolumeProfileGenerator(),
    ]

def create_advanced_time_generators() -> List[FeatureGenerator]:
    """Create advanced time feature generators with full VectorBT optimization."""
    return [
        # All basic generators
        *create_default_time_generators(),
        
        # Additional advanced features can be added here
    ]

def create_time_feature_batch(data: pd.DataFrame, generators: List[FeatureGenerator] = None) -> pd.DataFrame:
    """
    Create time features in batch using UnifiedVectorizationManager for maximum performance.
    
    Args:
        data: Input OHLCV data
        generators: List of time feature generators (uses default if None)
        
    Returns:
        DataFrame with generated time features
    """
    if generators is None:
        generators = create_default_time_generators()
    
    # Use UnifiedVectorizationManager for batch processing if available
    try:
        from src.feature_generation.utils.unified_vectorization_manager import get_unified_vectorization_manager
        manager = get_unified_vectorization_manager()
        
        # Generate features using batch processing
        results = {}
        for generator in generators:
            try:
                result = generator.generate(data)
                if result.success:
                    results[generator.config.name] = result.data
                else:
                    print(f"Warning: {generator.config.name} failed: {result.error_message}")
            except Exception as e:
                print(f"Error generating {generator.config.name}: {e}")
        
        return pd.DataFrame(results, index=data.index)
        
    except ImportError:
        # Fallback to individual generation
        results = {}
        for generator in generators:
            try:
                result = generator.generate(data)
                if result.success:
                    results[generator.config.name] = result.data
            except Exception as e:
                print(f"Error generating {generator.config.name}: {e}")
        
        return pd.DataFrame(results, index=data.index)
