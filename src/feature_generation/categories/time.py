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
    from src.utils.vectorbt_compat import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from src.utils.vectorbt_compat import scale, rank, zscore, winsorize, clip, quantile
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

from src.utils.feature_common.volume_transforms import log1p_zscore_normalize

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

class TimeSinceLastVolSpikeGenerator(VectorBTTimeFeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="time_since_last_vol_spike",
            category=FeatureCategory.TIME,
            description="Time since last volatility spike (z-score > threshold)",
            required_columns=['close'],
            default_lookback=20,
            parameters={'threshold': 2.0, 'baseline_window': 100},
            use_vectorbt=True
        )
        super().__init__(config)

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        data = self._optimize_dataframe_processing(data)
        window = kwargs.get('window', self.config.default_lookback)
        threshold = kwargs.get('threshold', self.config.parameters['threshold'])
        baseline_window = kwargs.get('baseline_window', self.config.parameters['baseline_window'])

        # Calculate returns
        close = data['close']
        returns = close.pct_change()

        # Calculate Volatility (rolling std of returns)
        vol = self._vectorbt_rolling_operation(returns, 'std', window)

        # Calculate Z-Score of Volatility (relative to baseline)
        vol_mean = self._vectorbt_rolling_operation(vol, 'mean', baseline_window)
        vol_std = self._vectorbt_rolling_operation(vol, 'std', baseline_window)

        # Avoid division by zero
        vol_zscore = (vol - vol_mean) / vol_std.replace(0, np.nan)

        # Identify spikes
        is_spike = vol_zscore > threshold

        # Calculate time since spike using vectorized approach
        positions = pd.Series(np.arange(len(data)), index=data.index)
        spike_positions = positions.where(is_spike)
        last_spike_position = spike_positions.ffill()

        time_since = positions - last_spike_position

        # Fill initial NaNs with 0
        time_since = time_since.fillna(0)

        return time_since

# --- Helper Functions for Indicators (Vectorized/Pandas) ---

def _calc_rsi(close, window=14):
    delta = close.diff()
    # Use pandas rolling for calculation
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _calc_bb(close, window=20, std_dev=2):
    ma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper = ma + std * std_dev
    lower = ma - std * std_dev
    return upper, lower

def _calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig

def _calc_atr(high, low, close, window=14):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

def _calc_adx(high, low, close, window=14):
    # Simplified ADX implementation
    tr = _calc_atr(high, low, close, window=1) # True Range (1 period)
    up_move = high - high.shift()
    down_move = low.shift() - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Smooth (using simple rolling sum for efficiency in basic features)
    tr_s = pd.Series(tr, index=close.index).rolling(window=window).sum()
    plus_dm_s = pd.Series(plus_dm, index=close.index).rolling(window=window).sum()
    minus_dm_s = pd.Series(minus_dm, index=close.index).rolling(window=window).sum()

    plus_di = 100 * (plus_dm_s / tr_s.replace(0, np.nan))
    minus_di = 100 * (minus_dm_s / tr_s.replace(0, np.nan))

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.rolling(window=window).mean()
    return adx

def _calc_vwap(high, low, close, volume):
    typical_price = (high + low + close) / 3
    cum_pv = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()
    return cum_pv / cum_vol.replace(0, np.nan)

def _calc_time_since(condition, index):
    """Vectorized calculation of time since condition was true."""
    positions = pd.Series(np.arange(len(index)), index=index)
    condition_positions = positions.where(condition)
    last_condition_position = condition_positions.ffill()
    time_since = positions - last_condition_position
    time_since = time_since.fillna(0) # Default to 0 if never happened (or beginning of data)

    # Apply robust normalization (log1p + rolling z-score)
    # This handles the long-tail nature of "time since" features
    return log1p_zscore_normalize(time_since)

# --- New Time-Based Feature Generators ---

class TimeSinceTrendImpulseGenerator(VectorBTTimeFeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="time_since_trend_impulse",
            category=FeatureCategory.TIME,
            description="Time since last trend impulse (ADX > 25 and rising)",
            required_columns=['high', 'low', 'close'],
            default_lookback=14,
            use_vectorbt=True
        )
        super().__init__(config)

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        data = self._optimize_dataframe_processing(data)
        window = kwargs.get('window', self.config.default_lookback)

        adx = _calc_adx(data['high'], data['low'], data['close'], window)

        # Impulse condition: ADX > 25 and Rising
        is_impulse = (adx > 25) & (adx > adx.shift(1))

        return _calc_time_since(is_impulse, data.index)

class TimeSinceLocalHighGenerator(VectorBTTimeFeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="time_since_local_high",
            category=FeatureCategory.TIME,
            description="Time since last confirmed local high (fractal)",
            required_columns=['high'],
            default_lookback=5, # Window for fractal check
            use_vectorbt=True
        )
        super().__init__(config)

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        data = self._optimize_dataframe_processing(data)
        # Fractal High: High[t-2] is max of window 5 centered at t-2
        # Implemented as lagged check: at time t, check if t-2 was a high
        h = data['high']
        # Check if t-2 is greater than t-3, t-4, t-1, t
        is_high = (h.shift(2) > h.shift(3)) & \
                  (h.shift(2) > h.shift(4)) & \
                  (h.shift(2) > h.shift(1)) & \
                  (h.shift(2) > h)

        return _calc_time_since(is_high, data.index)

class TimeSinceLocalLowGenerator(VectorBTTimeFeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="time_since_local_low",
            category=FeatureCategory.TIME,
            description="Time since last confirmed local low (fractal)",
            required_columns=['low'],
            default_lookback=5,
            use_vectorbt=True
        )
        super().__init__(config)

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        data = self._optimize_dataframe_processing(data)
        l = data['low']
        is_low = (l.shift(2) < l.shift(3)) & \
                 (l.shift(2) < l.shift(4)) & \
                 (l.shift(2) < l.shift(1)) & \
                 (l.shift(2) < l)

        return _calc_time_since(is_low, data.index)

class TimeSinceBreakoutGenerator(VectorBTTimeFeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="time_since_breakout",
            category=FeatureCategory.TIME,
            description="Time since last BB breakout (Close outside bands)",
            required_columns=['close'],
            default_lookback=20,
            use_vectorbt=True
        )
        super().__init__(config)

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        data = self._optimize_dataframe_processing(data)
        window = kwargs.get('window', self.config.default_lookback)

        upper, lower = _calc_bb(data['close'], window=window)

        is_breakout = (data['close'] > upper) | (data['close'] < lower)

        return _calc_time_since(is_breakout, data.index)

class TimeSinceLargeCandleGenerator(VectorBTTimeFeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="time_since_large_candle",
            category=FeatureCategory.TIME,
            description="Time since last large candle (Body > 2 * ATR)",
            required_columns=['open', 'high', 'low', 'close'],
            default_lookback=14,
            use_vectorbt=True
        )
        super().__init__(config)

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        data = self._optimize_dataframe_processing(data)
        window = kwargs.get('window', self.config.default_lookback)

        atr = _calc_atr(data['high'], data['low'], data['close'], window)
        body_size = (data['close'] - data['open']).abs()

        is_large = body_size > (2 * atr)

        return _calc_time_since(is_large, data.index)

class TimeSinceLiquiditySweepGenerator(VectorBTTimeFeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="time_since_liquidity_sweep",
            category=FeatureCategory.TIME,
            description="Time since last liquidity sweep (large wicks)",
            required_columns=['open', 'high', 'low', 'close'],
            parameters={'threshold': 0.5},
            use_vectorbt=True
        )
        super().__init__(config)

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        data = self._optimize_dataframe_processing(data)
        threshold = kwargs.get('threshold', self.config.parameters['threshold'])

        o, h, l, c = data['open'], data['high'], data['low'], data['close']

        upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
        lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l

        max_wick = pd.concat([upper_wick, lower_wick], axis=1).max(axis=1)
        total_range = h - l

        # Avoid division by zero
        wick_ratio = max_wick / total_range.replace(0, np.nan)

        is_sweep = wick_ratio > threshold

        return _calc_time_since(is_sweep, data.index)

class TimeSinceSidewaysRegimeGenerator(VectorBTTimeFeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="time_since_sideways",
            category=FeatureCategory.TIME,
            description="Time since last sideways regime (ADX < 20)",
            required_columns=['high', 'low', 'close'],
            default_lookback=14,
            use_vectorbt=True
        )
        super().__init__(config)

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        data = self._optimize_dataframe_processing(data)
        window = kwargs.get('window', self.config.default_lookback)

        adx = _calc_adx(data['high'], data['low'], data['close'], window)

        is_sideways = adx < 20

        return _calc_time_since(is_sideways, data.index)

class TimeSinceRSICrossGenerator(VectorBTTimeFeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="time_since_rsi_cross",
            category=FeatureCategory.TIME,
            description="Time since RSI crossed 50",
            required_columns=['close'],
            default_lookback=14,
            use_vectorbt=True
        )
        super().__init__(config)

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        data = self._optimize_dataframe_processing(data)
        window = kwargs.get('window', self.config.default_lookback)

        rsi = _calc_rsi(data['close'], window)

        # Cross 50 (up or down)
        # (RSI[t] > 50 and RSI[t-1] <= 50) OR (RSI[t] < 50 and RSI[t-1] >= 50)
        # Simplified: sign of (RSI - 50) changed
        rsi_centered = rsi - 50
        is_cross = (np.sign(rsi_centered) != np.sign(rsi_centered.shift(1))) & (rsi_centered.notna()) & (rsi_centered.shift(1).notna())

        return _calc_time_since(is_cross, data.index)

class TimeSinceMACDCrossGenerator(VectorBTTimeFeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="time_since_macd_cross",
            category=FeatureCategory.TIME,
            description="Time since MACD crossed 0",
            required_columns=['close'],
            use_vectorbt=True
        )
        super().__init__(config)

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        data = self._optimize_dataframe_processing(data)

        macd, _ = _calc_macd(data['close'])

        # Cross 0
        is_cross = (np.sign(macd) != np.sign(macd.shift(1))) & (macd.notna()) & (macd.shift(1).notna())

        return _calc_time_since(is_cross, data.index)

class TimeSinceVWAPCrossGenerator(VectorBTTimeFeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="time_since_vwap_cross",
            category=FeatureCategory.TIME,
            description="Time since Price crossed VWAP or Touched VWAP",
            required_columns=['high', 'low', 'close', 'volume'],
            use_vectorbt=True
        )
        super().__init__(config)

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        data = self._optimize_dataframe_processing(data)

        vwap = _calc_vwap(data['high'], data['low'], data['close'], data['volume'])
        c = data['close']

        # Price Cross VWAP (Close crosses VWAP)
        diff = c - vwap
        is_cross = (np.sign(diff) != np.sign(diff.shift(1))) & (diff.notna()) & (diff.shift(1).notna())

        # Price Touch VWAP (High >= VWAP >= Low)
        is_touch = (data['high'] >= vwap) & (data['low'] <= vwap)

        # Combine: either cross or touch
        is_event = is_cross | is_touch

        return _calc_time_since(is_event, data.index)

class TimeSinceMeanReversionSignalGenerator(VectorBTTimeFeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="time_since_mean_reversion_signal",
            category=FeatureCategory.TIME,
            description="Time since Bollinger Band touch",
            required_columns=['high', 'low', 'close'],
            default_lookback=20,
            use_vectorbt=True
        )
        super().__init__(config)

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        data = self._optimize_dataframe_processing(data)
        window = kwargs.get('window', self.config.default_lookback)

        upper, lower = _calc_bb(data['close'], window=window)

        # Touch: High >= Upper OR Low <= Lower
        is_touch = (data['high'] >= upper) | (data['low'] <= lower)

        return _calc_time_since(is_touch, data.index)

def create_default_time_generators() -> List[FeatureGenerator]:
    """Create basic/cyclical time feature generators."""
    return [
        # Basic hour features
        HourGenerator(),

        # Cyclical encodings (ML compatible) - This is what the user specifically requested
        HourSinGenerator(),
        HourCosGenerator(),
        DayOfWeekSinGenerator(),
        DayOfWeekCosGenerator(),

        # Time Since Last Vol Spike
        TimeSinceLastVolSpikeGenerator(),

        # Additional Time-Since Features
        TimeSinceTrendImpulseGenerator(),
        TimeSinceLocalHighGenerator(),
        TimeSinceLocalLowGenerator(),
        TimeSinceBreakoutGenerator(),
        TimeSinceLargeCandleGenerator(),
        TimeSinceLiquiditySweepGenerator(),
        TimeSinceSidewaysRegimeGenerator(),
        TimeSinceRSICrossGenerator(),
        TimeSinceMACDCrossGenerator(),
        TimeSinceVWAPCrossGenerator(),
        TimeSinceMeanReversionSignalGenerator(),
    ]

def create_advanced_time_generators() -> List[FeatureGenerator]:
    """Create advanced time feature generators including intraday patterns and momentum."""
    return [
        # All basic generators
        *create_default_time_generators(),

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
