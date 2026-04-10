"""
Bar Construction Module

This module provides bar construction functionality for profit labeling.
It defines bar types and construction parameters for different data formats.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import logging


from numba import njit

@njit
def _numba_construct_volume_bars(timestamps, opens, highs, lows, closes, volumes, threshold):
    n = len(timestamps)
    out_timestamps = np.zeros(n, dtype=np.int64)
    out_opens = np.zeros(n, dtype=np.float64)
    out_highs = np.zeros(n, dtype=np.float64)
    out_lows = np.zeros(n, dtype=np.float64)
    out_closes = np.zeros(n, dtype=np.float64)
    out_volumes = np.zeros(n, dtype=np.float64)

    bar_idx = 0
    cum_vol = 0.0

    current_ts = 0
    current_open = 0.0
    current_high = -np.inf
    current_low = np.inf
    current_close = 0.0
    current_vol = 0.0
    in_bar = False

    for i in range(n):
        if not in_bar:
            current_ts = timestamps[i]
            current_open = opens[i]
            current_high = highs[i]
            current_low = lows[i]
            current_close = closes[i]
            current_vol = volumes[i]
            cum_vol = volumes[i]
            in_bar = True
        else:
            if highs[i] > current_high:
                current_high = highs[i]
            if lows[i] < current_low:
                current_low = lows[i]
            current_close = closes[i]
            current_vol += volumes[i]
            cum_vol += volumes[i]

        if cum_vol >= threshold:
            out_timestamps[bar_idx] = current_ts
            out_opens[bar_idx] = current_open
            out_highs[bar_idx] = current_high
            out_lows[bar_idx] = current_low
            out_closes[bar_idx] = current_close
            out_volumes[bar_idx] = current_vol
            bar_idx += 1
            in_bar = False
            cum_vol = 0.0

    if in_bar and current_vol > 0:
        out_timestamps[bar_idx] = current_ts
        out_opens[bar_idx] = current_open
        out_highs[bar_idx] = current_high
        out_lows[bar_idx] = current_low
        out_closes[bar_idx] = current_close
        out_volumes[bar_idx] = current_vol
        bar_idx += 1

    return out_timestamps[:bar_idx], out_opens[:bar_idx], out_highs[:bar_idx], out_lows[:bar_idx], out_closes[:bar_idx], out_volumes[:bar_idx]

@njit
def _numba_construct_dollar_bars(timestamps, opens, highs, lows, closes, volumes, price_col, threshold):
    n = len(timestamps)
    out_timestamps = np.zeros(n, dtype=np.int64)
    out_opens = np.zeros(n, dtype=np.float64)
    out_highs = np.zeros(n, dtype=np.float64)
    out_lows = np.zeros(n, dtype=np.float64)
    out_closes = np.zeros(n, dtype=np.float64)
    out_volumes = np.zeros(n, dtype=np.float64)
    out_dollar_volumes = np.zeros(n, dtype=np.float64)

    bar_idx = 0
    cum_dollar_vol = 0.0

    current_ts = 0
    current_open = 0.0
    current_high = -np.inf
    current_low = np.inf
    current_close = 0.0
    current_vol = 0.0
    current_dollar_vol = 0.0
    in_bar = False

    for i in range(n):
        price = price_col[i]
        dollar_volume = price * volumes[i]

        if not in_bar:
            current_ts = timestamps[i]
            current_open = opens[i]
            current_high = highs[i]
            current_low = lows[i]
            current_close = closes[i]
            current_vol = volumes[i]
            current_dollar_vol = dollar_volume
            cum_dollar_vol = dollar_volume
            in_bar = True
        else:
            if highs[i] > current_high:
                current_high = highs[i]
            if lows[i] < current_low:
                current_low = lows[i]
            current_close = closes[i]
            current_vol += volumes[i]
            current_dollar_vol += dollar_volume
            cum_dollar_vol += dollar_volume

        if cum_dollar_vol >= threshold:
            out_timestamps[bar_idx] = current_ts
            out_opens[bar_idx] = current_open
            out_highs[bar_idx] = current_high
            out_lows[bar_idx] = current_low
            out_closes[bar_idx] = current_close
            out_volumes[bar_idx] = current_vol
            out_dollar_volumes[bar_idx] = current_dollar_vol
            bar_idx += 1
            in_bar = False
            cum_dollar_vol = 0.0

    if in_bar and current_dollar_vol > 0:
        out_timestamps[bar_idx] = current_ts
        out_opens[bar_idx] = current_open
        out_highs[bar_idx] = current_high
        out_lows[bar_idx] = current_low
        out_closes[bar_idx] = current_close
        out_volumes[bar_idx] = current_vol
        out_dollar_volumes[bar_idx] = current_dollar_vol
        bar_idx += 1

    return out_timestamps[:bar_idx], out_opens[:bar_idx], out_highs[:bar_idx], out_lows[:bar_idx], out_closes[:bar_idx], out_volumes[:bar_idx], out_dollar_volumes[:bar_idx]

@njit
def _numba_construct_tick_bars(timestamps, opens, highs, lows, closes, price_col, price_change_threshold, threshold):
    n = len(timestamps)
    out_timestamps = np.zeros(n, dtype=np.int64)
    out_opens = np.zeros(n, dtype=np.float64)
    out_highs = np.zeros(n, dtype=np.float64)
    out_lows = np.zeros(n, dtype=np.float64)
    out_closes = np.zeros(n, dtype=np.float64)
    out_tick_counts = np.zeros(n, dtype=np.int64)

    bar_idx = 0
    tick_count = 0

    current_ts = 0
    current_open = 0.0
    current_high = -np.inf
    current_low = np.inf
    current_close = 0.0
    last_price = np.nan
    in_bar = False

    for i in range(n):
        current_price = price_col[i]

        if not np.isnan(last_price):
            price_change = abs(current_price - last_price)
            if price_change >= price_change_threshold:
                tick_count += 1

        if not in_bar:
            current_ts = timestamps[i]
            current_open = opens[i]
            current_high = highs[i]
            current_low = lows[i]
            current_close = closes[i]
            last_price = current_price
            in_bar = True
        else:
            if highs[i] > current_high:
                current_high = highs[i]
            if lows[i] < current_low:
                current_low = lows[i]
            current_close = closes[i]
            last_price = current_price

        if tick_count >= threshold:
            out_timestamps[bar_idx] = current_ts
            out_opens[bar_idx] = current_open
            out_highs[bar_idx] = current_high
            out_lows[bar_idx] = current_low
            out_closes[bar_idx] = current_close
            out_tick_counts[bar_idx] = tick_count
            bar_idx += 1
            in_bar = False
            tick_count = 0
            last_price = np.nan

    if in_bar:
        out_timestamps[bar_idx] = current_ts
        out_opens[bar_idx] = current_open
        out_highs[bar_idx] = current_high
        out_lows[bar_idx] = current_low
        out_closes[bar_idx] = current_close
        out_tick_counts[bar_idx] = tick_count
        bar_idx += 1

    return out_timestamps[:bar_idx], out_opens[:bar_idx], out_highs[:bar_idx], out_lows[:bar_idx], out_closes[:bar_idx], out_tick_counts[:bar_idx]

@njit
def _numba_construct_range_bars(timestamps, opens, highs, lows, closes, range_type_code, use_atr, atr_values, range_threshold):
    # range_type_code: 0=hl, 1=oc, 2=hlc
    n = len(timestamps)
    out_timestamps = np.zeros(n, dtype=np.int64)
    out_opens = np.zeros(n, dtype=np.float64)
    out_highs = np.zeros(n, dtype=np.float64)
    out_lows = np.zeros(n, dtype=np.float64)
    out_closes = np.zeros(n, dtype=np.float64)
    out_ranges = np.zeros(n, dtype=np.float64)

    bar_idx = 0
    current_ts = 0
    current_open = 0.0
    current_high = -np.inf
    current_low = np.inf
    current_close = 0.0
    current_range = 0.0
    in_bar = False

    for i in range(n):
        if not in_bar:
            current_ts = timestamps[i]
            current_open = opens[i]
            current_high = highs[i]
            current_low = lows[i]
            current_close = closes[i]
            current_range = 0.0
            in_bar = True
        else:
            if highs[i] > current_high:
                current_high = highs[i]
            if lows[i] < current_low:
                current_low = lows[i]
            current_close = closes[i]

        if range_type_code == 0: # hl
            current_range = current_high - current_low
        elif range_type_code == 1: # oc
            current_range = abs(current_close - current_open)
        else: # hlc
            current_range = max(current_high - current_low, abs(current_close - current_open))

        effective_threshold = range_threshold
        if use_atr and not np.isnan(atr_values[i]):
            effective_threshold = range_threshold * atr_values[i]

        if current_range >= effective_threshold:
            out_timestamps[bar_idx] = current_ts
            out_opens[bar_idx] = current_open
            out_highs[bar_idx] = current_high
            out_lows[bar_idx] = current_low
            out_closes[bar_idx] = current_close
            out_ranges[bar_idx] = current_range
            bar_idx += 1
            in_bar = False
            current_range = 0.0

    if in_bar:
        out_timestamps[bar_idx] = current_ts
        out_opens[bar_idx] = current_open
        out_highs[bar_idx] = current_high
        out_lows[bar_idx] = current_low
        out_closes[bar_idx] = current_close
        out_ranges[bar_idx] = current_range
        bar_idx += 1

    return out_timestamps[:bar_idx], out_opens[:bar_idx], out_highs[:bar_idx], out_lows[:bar_idx], out_closes[:bar_idx], out_ranges[:bar_idx]



class BarType(Enum):
    """Types of bars for construction."""
    TIME = "time"           # Time-based bars (OHLCV data)
    VOLUME = "volume"        # Volume-based bars
    DOLLAR = "dollar"        # Dollar volume-based bars
    TICK = "tick"           # Tick-based bars
    RANGE = "range"         # Range-based bars (high-low range)
    RENKO = "renko"         # Renko bars
    KAGI = "kagi"           # Kagi bars
    POINT_AND_FIGURE = "point_and_figure"  # Point and Figure bars


@dataclass
class BarConstructionConfig:
    """Configuration for bar construction."""
    bar_type: BarType = BarType.TIME
    bar_size: float = 1.0  # Size of the bar (time in minutes, volume, etc.)
    min_bars_required: int = 10  # Minimum number of bars required
    max_bars: Optional[int] = None  # Maximum number of bars to construct
    overlap_allowed: bool = False  # Whether overlapping bars are allowed
    gap_threshold: float = 0.0  # Threshold for gap detection
    session_filter: Optional[str] = None  # Session filter (e.g., "regular_hours")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'bar_type': self.bar_type.value,
            'bar_size': self.bar_size,
            'min_bars_required': self.min_bars_required,
            'max_bars': self.max_bars,
            'overlap_allowed': self.overlap_allowed,
            'gap_threshold': self.gap_threshold,
            'session_filter': self.session_filter
        }


class BarConstructor:
    """Bar construction utility class."""
    
    def __init__(self, config: BarConstructionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def construct_bars(self, data, **kwargs):
        """
        Construct bars from raw data based on configuration.
        
        Args:
            data: Input data (DataFrame, list of ticks, etc.)
            **kwargs: Additional parameters for bar construction
            
        Returns:
            Constructed bars
        """
        # Validate input data
        validation_results = self.validate_data(data)
        if not validation_results['is_valid']:
            raise ValueError(f"Data validation failed: {validation_results['issues']}")
        
        if validation_results['warnings']:
            for warning in validation_results['warnings']:
                self.logger.warning(warning)
        
        # Construct bars based on type
        if self.config.bar_type == BarType.TIME:
            result = self._construct_time_bars(data, **kwargs)
        elif self.config.bar_type == BarType.VOLUME:
            result = self._construct_volume_bars(data, **kwargs)
        elif self.config.bar_type == BarType.DOLLAR:
            result = self._construct_dollar_bars(data, **kwargs)
        elif self.config.bar_type == BarType.TICK:
            result = self._construct_tick_bars(data, **kwargs)
        elif self.config.bar_type == BarType.RANGE:
            result = self._construct_range_bars(data, **kwargs)
        else:
            raise ValueError(f"Unsupported bar type: {self.config.bar_type}")
        
        # Validate minimum bars requirement
        if len(result) < self.config.min_bars_required:
            self.logger.warning(f"Only {len(result)} bars constructed, minimum required: {self.config.min_bars_required}")
        
        # Apply maximum bars limit if specified
        if self.config.max_bars is not None and len(result) > self.config.max_bars:
            result = result.iloc[:self.config.max_bars]
            self.logger.info(f"Limited bars to {self.config.max_bars} as specified in config")
        
        return result
    
    def _construct_time_bars(self, data, **kwargs):
        """Construct time-based bars.
        
        Args:
            data: DataFrame with OHLCV data or tick data
            **kwargs: Additional parameters
                - timeframe: Timeframe for aggregation (e.g., '1T', '5T', '1H', '1D')
                - resample_method: Method for resampling ('ohlc', 'first', 'last', 'mean')
            
        Returns:
            DataFrame with time-based bars
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        
        if len(data) == 0:
            return data
        
        # Get parameters
        timeframe = kwargs.get('timeframe', None)
        resample_method = kwargs.get('resample_method', 'ohlc')
        
        # If no timeframe specified, return data as-is (already time-based)
        if timeframe is None:
            return data
        
        # Ensure data has datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a datetime index for time-based bar construction")
        
        # Resample based on timeframe
        if resample_method == 'ohlc':
            # Standard OHLC resampling
            result = data.resample(timeframe).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        else:
            # Custom resampling method
            agg_dict = {}
            for col in data.columns:
                if col in ['open', 'high', 'low', 'close']:
                    if resample_method == 'first':
                        agg_dict[col] = 'first'
                    elif resample_method == 'last':
                        agg_dict[col] = 'last'
                    elif resample_method == 'mean':
                        agg_dict[col] = 'mean'
                    else:
                        agg_dict[col] = 'last'  # default
                elif col == 'volume':
                    agg_dict[col] = 'sum'
                else:
                    agg_dict[col] = 'last'  # default for other columns
            
            result = data.resample(timeframe).agg(agg_dict).dropna()
        
        self.logger.info(f"Constructed {len(result)} time bars with timeframe {timeframe}")
        return result
    
    def _construct_volume_bars(self, data, **kwargs):
        """Construct volume-based bars.
        
        Args:
            data: DataFrame with OHLCV data (columns: open, high, low, close, volume)
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with volume-based bars
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if len(data) == 0:
            return pd.DataFrame(columns=required_columns)
        
        # Ensure data is sorted by index (time)
        data = data.sort_index()
        
        volume_threshold = self.config.bar_size
        
        ts_arr = data.index.astype(np.int64).to_numpy()
        op_arr = data['open'].to_numpy(dtype=np.float64)
        hi_arr = data['high'].to_numpy(dtype=np.float64)
        lo_arr = data['low'].to_numpy(dtype=np.float64)
        cl_arr = data['close'].to_numpy(dtype=np.float64)
        vo_arr = data['volume'].to_numpy(dtype=np.float64)
        
        ts, op, hi, lo, cl, vo = _numba_construct_volume_bars(
            ts_arr, op_arr, hi_arr, lo_arr, cl_arr, vo_arr, volume_threshold
        )
        
        if len(ts) == 0:
            self.logger.warning("No volume bars could be constructed")
            return pd.DataFrame(columns=required_columns)

        result = pd.DataFrame({
            'timestamp': pd.to_datetime(ts),
            'open': op,
            'high': hi,
            'low': lo,
            'close': cl,
            'volume': vo
        })
        result.set_index('timestamp', inplace=True)
        
        self.logger.info(f"Constructed {len(result)} volume bars with threshold {volume_threshold}")
        return result
    
    def _construct_dollar_bars(self, data, **kwargs):
        """Construct dollar volume-based bars adapted for crypto (USDT/USDC).
        
        Args:
            data: DataFrame with OHLCV data (columns: open, high, low, close, volume)
            **kwargs: Additional parameters
                - quote_currency: Quote currency for dollar calculation (default: 'USDT')
                - price_column: Column to use for price calculation (default: 'close')
            
        Returns:
            DataFrame with dollar volume-based bars
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if len(data) == 0:
            return pd.DataFrame(columns=required_columns)
        
        # Get parameters
        quote_currency = kwargs.get('quote_currency', 'USDT')
        price_column = kwargs.get('price_column', 'close')
        
        if price_column not in data.columns:
            raise ValueError(f"Price column '{price_column}' not found in data")
        
        # Ensure data is sorted by index (time)
        data = data.sort_index()
        
        dollar_threshold = self.config.bar_size
        
        ts_arr = data.index.astype(np.int64).to_numpy()
        op_arr = data['open'].to_numpy(dtype=np.float64)
        hi_arr = data['high'].to_numpy(dtype=np.float64)
        lo_arr = data['low'].to_numpy(dtype=np.float64)
        cl_arr = data['close'].to_numpy(dtype=np.float64)
        vo_arr = data['volume'].to_numpy(dtype=np.float64)
        pr_arr = data[price_column].to_numpy(dtype=np.float64)
        
        ts, op, hi, lo, cl, vo, d_vo = _numba_construct_dollar_bars(
            ts_arr, op_arr, hi_arr, lo_arr, cl_arr, vo_arr, pr_arr, dollar_threshold
        )
        
        if len(ts) == 0:
            self.logger.warning("No dollar bars could be constructed")
            return pd.DataFrame(columns=required_columns + ['dollar_volume'])

        result = pd.DataFrame({
            'timestamp': pd.to_datetime(ts),
            'open': op,
            'high': hi,
            'low': lo,
            'close': cl,
            'volume': vo,
            'dollar_volume': d_vo
        })
        result.set_index('timestamp', inplace=True)
        
        self.logger.info(f"Constructed {len(result)} dollar bars with threshold {dollar_threshold} {quote_currency}")
        return result
    
    def _construct_tick_bars(self, data, **kwargs):
        """Construct tick-based bars.
        
        Args:
            data: DataFrame with OHLCV data or tick data
            **kwargs: Additional parameters
                - tick_column: Column to use for tick counting (default: 'close')
                - price_change_threshold: Minimum price change to count as a tick (default: 0.0)
            
        Returns:
            DataFrame with tick-based bars
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if len(data) == 0:
            return pd.DataFrame(columns=required_columns)
        
        # Get parameters
        tick_column = kwargs.get('tick_column', 'close')
        price_change_threshold = kwargs.get('price_change_threshold', 0.0)
        
        if tick_column not in data.columns:
            raise ValueError(f"Tick column '{tick_column}' not found in data")
        
        # Ensure data is sorted by index (time)
        data = data.sort_index()
        
        tick_threshold = int(self.config.bar_size)
        
        ts_arr = data.index.astype(np.int64).to_numpy()
        op_arr = data['open'].to_numpy(dtype=np.float64)
        hi_arr = data['high'].to_numpy(dtype=np.float64)
        lo_arr = data['low'].to_numpy(dtype=np.float64)
        cl_arr = data['close'].to_numpy(dtype=np.float64)
        pr_arr = data[tick_column].to_numpy(dtype=np.float64)
        
        ts, op, hi, lo, cl, t_counts = _numba_construct_tick_bars(
            ts_arr, op_arr, hi_arr, lo_arr, cl_arr, pr_arr, price_change_threshold, tick_threshold
        )
        
        if len(ts) == 0:
            self.logger.warning("No tick bars could be constructed")
            return pd.DataFrame(columns=required_columns + ['tick_count'])

        result = pd.DataFrame({
            'timestamp': pd.to_datetime(ts),
            'open': op,
            'high': hi,
            'low': lo,
            'close': cl,
            'tick_count': t_counts
        })
        result.set_index('timestamp', inplace=True)
        
        self.logger.info(f"Constructed {len(result)} tick bars with threshold {tick_threshold} ticks")
        return result
    
    def _construct_range_bars(self, data, **kwargs):
        """Construct range-based bars.
        
        Args:
            data: DataFrame with OHLCV data (columns: open, high, low, close, volume)
            **kwargs: Additional parameters
                - range_type: Type of range calculation ('hl' for high-low, 'oc' for open-close, 'hlc' for high-low-close)
                - use_atr: Whether to use ATR for dynamic range calculation (default: False)
                - atr_period: Period for ATR calculation (default: 14)
            
        Returns:
            DataFrame with range-based bars
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if len(data) == 0:
            return pd.DataFrame(columns=required_columns)
        
        # Get parameters
        range_type = kwargs.get('range_type', 'hl')  # high-low range
        use_atr = kwargs.get('use_atr', False)
        atr_period = kwargs.get('atr_period', 14)
        
        # Ensure data is sorted by index (time)
        data = data.sort_index()
        
        range_threshold = self.config.bar_size
        
        # Calculate ATR if requested
        if use_atr:
            atr_values_series = self._calculate_atr(data, atr_period)
            # Reindex to match data if needed (should already match)
            atr_values_arr = atr_values_series.reindex(data.index).to_numpy(dtype=np.float64)
        else:
            atr_values_arr = np.full(len(data), np.nan, dtype=np.float64)
            
        type_code = 0
        if range_type == 'hl':
            type_code = 0
        elif range_type == 'oc':
            type_code = 1
        elif range_type == 'hlc':
            type_code = 2
        else:
            raise ValueError(f"Unsupported range type: {range_type}")
            
        ts_arr = data.index.astype(np.int64).to_numpy()
        op_arr = data['open'].to_numpy(dtype=np.float64)
        hi_arr = data['high'].to_numpy(dtype=np.float64)
        lo_arr = data['low'].to_numpy(dtype=np.float64)
        cl_arr = data['close'].to_numpy(dtype=np.float64)
        
        ts, op, hi, lo, cl, rngs = _numba_construct_range_bars(
            ts_arr, op_arr, hi_arr, lo_arr, cl_arr, type_code, use_atr, atr_values_arr, range_threshold
        )
        
        if len(ts) == 0:
            self.logger.warning("No range bars could be constructed")
            return pd.DataFrame(columns=required_columns + ['range'])

        result = pd.DataFrame({
            'timestamp': pd.to_datetime(ts),
            'open': op,
            'high': hi,
            'low': lo,
            'close': cl,
            'range': rngs
        })
        result.set_index('timestamp', inplace=True)
        
        self.logger.info(f"Constructed {len(result)} range bars with threshold {range_threshold} ({range_type})")
        return result
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR) for dynamic range calculation.
        
        Args:
            data: DataFrame with OHLC data
            period: ATR calculation period
            
        Returns:
            Series with ATR values
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate True Range components
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        # True Range is the maximum of the three
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR as rolling mean of True Range
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate input data for bar construction.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }
        
        if not isinstance(data, pd.DataFrame):
            validation_results['is_valid'] = False
            validation_results['issues'].append("Data must be a pandas DataFrame")
            return validation_results
        
        if len(data) == 0:
            validation_results['is_valid'] = False
            validation_results['issues'].append("Data is empty")
            return validation_results
        
        # Check for required columns based on bar type
        if self.config.bar_type in [BarType.VOLUME, BarType.DOLLAR, BarType.TICK, BarType.RANGE]:
            required_columns = ['open', 'high', 'low', 'close']
            if self.config.bar_type in [BarType.VOLUME, BarType.DOLLAR]:
                required_columns.append('volume')
            
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                validation_results['is_valid'] = False
                validation_results['issues'].append(f"Missing required columns: {missing_columns}")
        
        # Check for datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            validation_results['warnings'].append("Data does not have a datetime index")
        
        # Check for data consistency
        if 'high' in data.columns and 'low' in data.columns:
            invalid_hl = (data['high'] < data['low']).sum()
            if invalid_hl > 0:
                validation_results['warnings'].append(f"Found {invalid_hl} records where high < low")
        
        # Check for negative values
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in ['open', 'high', 'low', 'close', 'volume']:
                negative_count = (data[col] < 0).sum()
                if negative_count > 0:
                    validation_results['warnings'].append(f"Found {negative_count} negative values in {col}")
        
        return validation_results
    
    def get_bar_statistics(self, bars: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about constructed bars.
        
        Args:
            bars: DataFrame with constructed bars
            
        Returns:
            Dictionary with bar statistics
        """
        if len(bars) == 0:
            return {
                'total_bars': 0,
                'time_span': None,
                'avg_bar_duration': None,
                'volume_stats': {},
                'price_stats': {}
            }
        
        stats = {
            'total_bars': len(bars),
            'time_span': None,
            'avg_bar_duration': None,
            'volume_stats': {},
            'price_stats': {}
        }
        
        # Time span analysis
        if isinstance(bars.index, pd.DatetimeIndex) and len(bars) > 1:
            time_span = bars.index[-1] - bars.index[0]
            stats['time_span'] = str(time_span)
            stats['avg_bar_duration'] = str(time_span / len(bars))
        
        # Volume statistics
        if 'volume' in bars.columns:
            volume = bars['volume']
            stats['volume_stats'] = {
                'mean': float(volume.mean()),
                'std': float(volume.std()),
                'min': float(volume.min()),
                'max': float(volume.max()),
                'total': float(volume.sum())
            }
        
        # Price statistics
        price_columns = ['open', 'high', 'low', 'close']
        available_price_columns = [col for col in price_columns if col in bars.columns]
        
        if available_price_columns:
            price_stats = {}
            for col in available_price_columns:
                price_data = bars[col]
                price_stats[col] = {
                    'mean': float(price_data.mean()),
                    'std': float(price_data.std()),
                    'min': float(price_data.min()),
                    'max': float(price_data.max())
                }
            stats['price_stats'] = price_stats
        
        return stats


def create_bar_constructor(bar_type: BarType = BarType.TIME, 
                          bar_size: float = 1.0,
                          min_bars_required: int = 10,
                          **kwargs) -> BarConstructor:
    """
    Create a bar constructor with the specified configuration.
    
    Args:
        bar_type: Type of bars to construct
        bar_size: Size of the bars
        min_bars_required: Minimum number of bars required
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured BarConstructor instance
    """
    config = BarConstructionConfig(
        bar_type=bar_type,
        bar_size=bar_size,
        min_bars_required=min_bars_required,
        **kwargs
    )
    return BarConstructor(config)


def create_crypto_dollar_bar_constructor(bar_size: float = 10000.0,  # $10k default
                                       quote_currency: str = 'USDT',
                                       **kwargs) -> BarConstructor:
    """
    Create a bar constructor optimized for crypto dollar bars.
    
    Args:
        bar_size: Dollar volume threshold for bar construction
        quote_currency: Quote currency (USDT, USDC, etc.)
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured BarConstructor instance for crypto dollar bars
    """
    config = BarConstructionConfig(
        bar_type=BarType.DOLLAR,
        bar_size=bar_size,
        min_bars_required=kwargs.get('min_bars_required', 10),
        **kwargs
    )
    return BarConstructor(config)


def create_crypto_volume_bar_constructor(bar_size: float = 1000.0,  # 1000 units default
                                       **kwargs) -> BarConstructor:
    """
    Create a bar constructor optimized for crypto volume bars.
    
    Args:
        bar_size: Volume threshold for bar construction
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured BarConstructor instance for crypto volume bars
    """
    config = BarConstructionConfig(
        bar_type=BarType.VOLUME,
        bar_size=bar_size,
        min_bars_required=kwargs.get('min_bars_required', 10),
        **kwargs
    )
    return BarConstructor(config)


def create_crypto_range_bar_constructor(bar_size: float = 0.01,  # 1% range default
                                      range_type: str = 'hl',
                                      use_atr: bool = True,
                                      **kwargs) -> BarConstructor:
    """
    Create a bar constructor optimized for crypto range bars.
    
    Args:
        bar_size: Range threshold for bar construction
        range_type: Type of range calculation ('hl', 'oc', 'hlc')
        use_atr: Whether to use ATR for dynamic range calculation
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured BarConstructor instance for crypto range bars
    """
    config = BarConstructionConfig(
        bar_type=BarType.RANGE,
        bar_size=bar_size,
        min_bars_required=kwargs.get('min_bars_required', 10),
        **kwargs
    )
    return BarConstructor(config)
