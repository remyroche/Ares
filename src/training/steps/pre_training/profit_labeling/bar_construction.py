"""
Bar Construction Module

This module provides bar construction functionality for profit labeling.
It defines bar types and construction parameters for different data formats.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union
import pandas as pd
import numpy as np
import logging

# Import tprint utilities for enhanced troubleshooting
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_warning, tprint_error, tprint_success, 
        tprint_debug, tprint_data_preview, tprint_data_format, tprint_performance,
        tprint_structured, tprint_step, tprint_result
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)
    def tprint_data_preview(*args, **kwargs): print("DATA_PREVIEW:", *args, **kwargs)
    def tprint_data_format(*args, **kwargs): print("DATA_FORMAT:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERFORMANCE:", *args, **kwargs)
    def tprint_structured(*args, **kwargs): print("STRUCTURED:", *args, **kwargs)
    def tprint_step(*args, **kwargs): print("STEP:", *args, **kwargs)
    def tprint_result(*args, **kwargs): print("RESULT:", *args, **kwargs)


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
        tprint_step("🔨 Starting bar construction")
        tprint_data_preview(data, "bar_construction_input_data", level="DEBUG")
        tprint_info(f"📊 Input data type: {type(data)}")
        tprint_info(f"🎯 Bar type: {self.config.bar_type.value}")
        tprint_info(f"📏 Bar size: {self.config.bar_size}")
        tprint_info(f"📋 Min bars required: {self.config.min_bars_required}")
        
        # Validate input data
        tprint_debug("🔍 Validating input data")
        validation_results = self.validate_data(data)
        if not validation_results['is_valid']:
            tprint_error(f"❌ Data validation failed: {validation_results['issues']}")
            raise ValueError(f"Data validation failed: {validation_results['issues']}")
        
        if validation_results['warnings']:
            for warning in validation_results['warnings']:
                self.logger.warning(warning)
                tprint_warning(f"⚠️ {warning}")
        
        tprint_success("✅ Data validation passed")
        
        # Construct bars based on type
        tprint_debug(f"🔨 Constructing {self.config.bar_type.value} bars")
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
            tprint_error(f"❌ Unsupported bar type: {self.config.bar_type}")
            raise ValueError(f"Unsupported bar type: {self.config.bar_type}")
        
        tprint_success(f"✅ Bar construction completed: {len(result)} bars created")
        tprint_data_preview(result, "constructed_bars", level="DEBUG")
        
        # Validate minimum bars requirement
        if len(result) < self.config.min_bars_required:
            tprint_warning(f"⚠️ Only {len(result)} bars constructed, minimum required: {self.config.min_bars_required}")
            self.logger.warning(f"Only {len(result)} bars constructed, minimum required: {self.config.min_bars_required}")
        
        # Apply maximum bars limit if specified
        if self.config.max_bars is not None and len(result) > self.config.max_bars:
            tprint_info(f"📏 Limiting bars to {self.config.max_bars} as specified in config")
            result = result.iloc[:self.config.max_bars]
            self.logger.info(f"Limited bars to {self.config.max_bars} as specified in config")
        
        tprint_result(f"🎯 Bar construction complete: {len(result)} final bars")
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
        bars = []
        current_bar = None
        cumulative_volume = 0.0
        
        for idx, row in data.iterrows():
            if current_bar is None:
                # Start new bar
                current_bar = {
                    'timestamp': idx,
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume']
                }
                cumulative_volume = row['volume']
            else:
                # Update current bar
                current_bar['high'] = max(current_bar['high'], row['high'])
                current_bar['low'] = min(current_bar['low'], row['low'])
                current_bar['close'] = row['close']
                current_bar['volume'] += row['volume']
                cumulative_volume += row['volume']
            
            # Check if we've reached the volume threshold
            if cumulative_volume >= volume_threshold:
                bars.append(current_bar)
                current_bar = None
                cumulative_volume = 0.0
        
        # Add the last incomplete bar if it exists and has minimum volume
        if current_bar is not None and current_bar['volume'] > 0:
            bars.append(current_bar)
        
        if not bars:
            self.logger.warning("No volume bars could be constructed")
            return pd.DataFrame(columns=required_columns)
        
        # Convert to DataFrame
        result = pd.DataFrame(bars)
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
        bars = []
        current_bar = None
        cumulative_dollar_volume = 0.0
        
        for idx, row in data.iterrows():
            # Calculate dollar volume for this row
            price = row[price_column]
            volume = row['volume']
            dollar_volume = price * volume
            
            if current_bar is None:
                # Start new bar
                current_bar = {
                    'timestamp': idx,
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume'],
                    'dollar_volume': dollar_volume
                }
                cumulative_dollar_volume = dollar_volume
            else:
                # Update current bar
                current_bar['high'] = max(current_bar['high'], row['high'])
                current_bar['low'] = min(current_bar['low'], row['low'])
                current_bar['close'] = row['close']
                current_bar['volume'] += row['volume']
                current_bar['dollar_volume'] += dollar_volume
                cumulative_dollar_volume += dollar_volume
            
            # Check if we've reached the dollar volume threshold
            if cumulative_dollar_volume >= dollar_threshold:
                bars.append(current_bar)
                current_bar = None
                cumulative_dollar_volume = 0.0
        
        # Add the last incomplete bar if it exists and has minimum dollar volume
        if current_bar is not None and current_bar['dollar_volume'] > 0:
            bars.append(current_bar)
        
        if not bars:
            self.logger.warning("No dollar bars could be constructed")
            return pd.DataFrame(columns=required_columns + ['dollar_volume'])
        
        # Convert to DataFrame
        result = pd.DataFrame(bars)
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
        bars = []
        current_bar = None
        tick_count = 0
        last_price = None
        
        for idx, row in data.iterrows():
            current_price = row[tick_column]
            
            # Count ticks based on price changes
            if last_price is not None:
                price_change = abs(current_price - last_price)
                if price_change >= price_change_threshold:
                    tick_count += 1
            
            if current_bar is None:
                # Start new bar
                current_bar = {
                    'timestamp': idx,
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'tick_count': 0
                }
                last_price = current_price
            else:
                # Update current bar
                current_bar['high'] = max(current_bar['high'], row['high'])
                current_bar['low'] = min(current_bar['low'], row['low'])
                current_bar['close'] = row['close']
                current_bar['tick_count'] = tick_count
                last_price = current_price
            
            # Check if we've reached the tick threshold
            if tick_count >= tick_threshold:
                bars.append(current_bar)
                current_bar = None
                tick_count = 0
                last_price = None
        
        # Add the last incomplete bar if it exists
        if current_bar is not None:
            bars.append(current_bar)
        
        if not bars:
            self.logger.warning("No tick bars could be constructed")
            return pd.DataFrame(columns=required_columns + ['tick_count'])
        
        # Convert to DataFrame
        result = pd.DataFrame(bars)
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
        bars = []
        current_bar = None
        current_range = 0.0
        
        # Calculate ATR if requested
        if use_atr:
            atr_values = self._calculate_atr(data, atr_period)
        else:
            atr_values = None
        
        for idx, row in data.iterrows():
            if current_bar is None:
                # Start new bar
                current_bar = {
                    'timestamp': idx,
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'range': 0.0
                }
                current_range = 0.0
            else:
                # Update current bar
                current_bar['high'] = max(current_bar['high'], row['high'])
                current_bar['low'] = min(current_bar['low'], row['low'])
                current_bar['close'] = row['close']
            
            # Calculate current range based on type
            if range_type == 'hl':
                current_range = current_bar['high'] - current_bar['low']
            elif range_type == 'oc':
                current_range = abs(current_bar['close'] - current_bar['open'])
            elif range_type == 'hlc':
                current_range = max(
                    current_bar['high'] - current_bar['low'],
                    abs(current_bar['close'] - current_bar['open'])
                )
            else:
                raise ValueError(f"Unsupported range type: {range_type}")
            
            # Use ATR-adjusted threshold if requested
            effective_threshold = range_threshold
            if use_atr and atr_values is not None and idx in atr_values.index:
                atr_value = atr_values.loc[idx]
                effective_threshold = range_threshold * atr_value
            
            current_bar['range'] = current_range
            
            # Check if we've reached the range threshold
            if current_range >= effective_threshold:
                bars.append(current_bar)
                current_bar = None
                current_range = 0.0
        
        # Add the last incomplete bar if it exists
        if current_bar is not None:
            bars.append(current_bar)
        
        if not bars:
            self.logger.warning("No range bars could be constructed")
            return pd.DataFrame(columns=required_columns + ['range'])
        
        # Convert to DataFrame
        result = pd.DataFrame(bars)
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
