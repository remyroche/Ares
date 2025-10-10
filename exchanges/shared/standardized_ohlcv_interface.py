"""
Standardized OHLCV Data Interface

This module provides a unified interface for OHLCV data across all exchanges,
ensuring complete equivalency and standardization for downstream use.

Features:
- Unified MarketData structure
- Standardized data conversion pipeline
- Exchange-agnostic data processing
- Comprehensive validation and error handling
- Consistent field mapping across all exchanges
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

# Configure logging
logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Data source enumeration"""
    BINANCE = "binance"
    BINGX = "bingx"
    OKX = "okx"
    MEXC = "mexc"
    GATEIO = "gateio"
    PHEMEX = "phemex"


class Interval(Enum):
    """Standardized interval enumeration"""
    MINUTE_1 = "1m"
    MINUTE_3 = "3m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_2 = "2h"
    HOUR_4 = "4h"
    HOUR_6 = "6h"
    HOUR_8 = "8h"
    HOUR_12 = "12h"
    DAY_1 = "1d"
    DAY_3 = "3d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"


@dataclass
class StandardizedMarketData:
    """
    Standardized market data structure for all exchanges.
    
    This is the single source of truth for OHLCV data across the entire system.
    All exchanges must convert their data to this format.
    """
    # Core OHLCV data
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    interval: str
    
    # Exchange metadata
    exchange: str
    source: DataSource
    
    # Additional standardized fields
    quote_volume: Optional[float] = None
    trades_count: Optional[int] = None
    taker_buy_base_volume: Optional[float] = None
    taker_buy_quote_volume: Optional[float] = None
    
    # Data quality metrics
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    
    # Processing metadata
    processed_at: Optional[datetime] = None
    raw_data_hash: Optional[str] = None
    
    def __post_init__(self):
        """Validate data after initialization"""
        self._validate_data()
        if not self.processed_at:
            self.processed_at = datetime.utcnow()
    
    def _validate_data(self) -> None:
        """Validate the market data for consistency and quality"""
        errors = []
        
        # Validate required fields
        if not self.symbol or not isinstance(self.symbol, str):
            errors.append("Symbol must be a non-empty string")
        
        if not isinstance(self.timestamp, datetime):
            errors.append("Timestamp must be a datetime object")
        
        # Validate OHLCV values
        ohlcv_values = [self.open, self.high, self.low, self.close, self.volume]
        for field_name, value in zip(['open', 'high', 'low', 'close', 'volume'], ohlcv_values):
            if not isinstance(value, (int, float)) or np.isnan(value) or value < 0:
                errors.append(f"{field_name} must be a non-negative number")
        
        # Validate OHLC relationships
        if self.high < max(self.open, self.close):
            errors.append("High must be >= max(open, close)")
        
        if self.low > min(self.open, self.close):
            errors.append("Low must be <= min(open, close)")
        
        if self.high < self.low:
            errors.append("High must be >= low")
        
        # Validate interval
        valid_intervals = [interval.value for interval in Interval]
        if self.interval not in valid_intervals:
            errors.append(f"Interval must be one of {valid_intervals}")
        
        # Validate exchange
        valid_exchanges = [source.value for source in DataSource]
        if self.exchange not in valid_exchanges:
            errors.append(f"Exchange must be one of {valid_exchanges}")
        
        self.validation_errors = errors
        self.is_valid = len(errors) == 0
        
        if not self.is_valid:
            logger.warning(f"Invalid market data: {errors}")


class OHLCVDataStandardizer:
    """
    Centralized OHLCV data standardizer for all exchanges.
    
    This class ensures that all exchange data is converted to the standardized
    format with consistent field names, data types, and validation.
    """
    
    def __init__(self):
        """Initialize the standardizer with exchange-specific configurations"""
        self.exchange_configs = {
            DataSource.BINANCE: {
                'timestamp_field': 'open_time',
                'timestamp_unit': 'ms',
                'field_mapping': {
                    'open_time': 'timestamp',
                    'close_time': 'close_time',
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume',
                    'quoteVolume': 'quote_volume',
                    'trades': 'trades_count',
                    'takerBuyBase': 'taker_buy_base_volume',
                    'takerBuyQuote': 'taker_buy_quote_volume'
                }
            },
            DataSource.BINGX: {
                'timestamp_field': 'open_time',
                'timestamp_unit': 'ms',
                'field_mapping': {
                    'open_time': 'timestamp',
                    'close_time': 'close_time',
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume',
                    'quoteVolume': 'quote_volume',
                    'trades': 'trades_count',
                    'takerBuyBase': 'taker_buy_base_volume',
                    'takerBuyQuote': 'taker_buy_quote_volume'
                }
            },
            DataSource.OKX: {
                'timestamp_field': 'timestamp',
                'timestamp_unit': 'ms',
                'field_mapping': {
                    'ts': 'timestamp',
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'vol': 'volume',
                    'volCcy': 'quote_volume',
                    'confirm': 'trades_count'
                }
            },
            DataSource.MEXC: {
                'timestamp_field': 'open_time',
                'timestamp_unit': 'ms',
                'field_mapping': {
                    'open_time': 'timestamp',
                    'close_time': 'close_time',
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume',
                    'quoteVolume': 'quote_volume',
                    'trades': 'trades_count',
                    'takerBuyBase': 'taker_buy_base_volume',
                    'takerBuyQuote': 'taker_buy_quote_volume'
                }
            },
            DataSource.GATEIO: {
                'timestamp_field': 'timestamp',
                'timestamp_unit': 's',
                'field_mapping': {
                    'timestamp': 'timestamp',
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume',
                    'quote_volume': 'quote_volume',
                    'trades': 'trades_count'
                }
            },
            DataSource.PHEMEX: {
                'timestamp_field': 'timestamp',
                'timestamp_unit': 'ms',
                'field_mapping': {
                    'timestamp': 'timestamp',
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume',
                    'quote_volume': 'quote_volume',
                    'trades': 'trades_count'
                }
            }
        }
    
    def standardize_data(
        self,
        raw_data: Union[List[Dict], List[List], pd.DataFrame],
        exchange: DataSource,
        symbol: str,
        interval: str
    ) -> List[StandardizedMarketData]:
        """
        Standardize raw exchange data to unified format.
        
        Args:
            raw_data: Raw data from exchange (list of dicts, list of lists, or DataFrame)
            exchange: Exchange source
            symbol: Trading symbol
            interval: Data interval
            
        Returns:
            List of standardized market data objects
        """
        try:
            # Convert to list of dictionaries for consistent processing
            data_list = self._normalize_input_format(raw_data)
            
            # Get exchange configuration
            config = self.exchange_configs.get(exchange)
            if not config:
                raise ValueError(f"No configuration found for exchange: {exchange}")
            
            standardized_data = []
            
            for item in data_list:
                try:
                    # Convert single data point
                    market_data = self._convert_single_data_point(
                        item, exchange, symbol, interval, config
                    )
                    standardized_data.append(market_data)
                    
                except Exception as e:
                    logger.error(f"Failed to convert data point: {e}")
                    # Create invalid data point for tracking
                    invalid_data = StandardizedMarketData(
                        symbol=symbol,
                        timestamp=datetime.utcnow(),
                        open=0.0,
                        high=0.0,
                        low=0.0,
                        close=0.0,
                        volume=0.0,
                        interval=interval,
                        exchange=exchange.value,
                        source=exchange,
                        is_valid=False,
                        validation_errors=[str(e)]
                    )
                    standardized_data.append(invalid_data)
            
            logger.info(f"Standardized {len(standardized_data)} data points from {exchange.value}")
            return standardized_data
            
        except Exception as e:
            logger.error(f"Failed to standardize data from {exchange.value}: {e}")
            raise
    
    def _normalize_input_format(
        self, 
        raw_data: Union[List[Dict], List[List], pd.DataFrame]
    ) -> List[Dict]:
        """Convert various input formats to list of dictionaries"""
        if isinstance(raw_data, pd.DataFrame):
            return raw_data.to_dict('records')
        
        if isinstance(raw_data, list) and len(raw_data) > 0:
            if isinstance(raw_data[0], list):
                # Convert list of lists to list of dicts
                return self._convert_list_of_lists_to_dicts(raw_data)
            elif isinstance(raw_data[0], dict):
                return raw_data
        
        raise ValueError(f"Unsupported data format: {type(raw_data)}")
    
    def _convert_list_of_lists_to_dicts(self, data: List[List]) -> List[Dict]:
        """Convert list of lists (typical exchange format) to list of dictionaries"""
        # Standard kline format: [timestamp, open, high, low, close, volume, ...]
        standard_fields = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote'
        ]
        
        result = []
        for item in data:
            if len(item) >= 6:  # Minimum required fields
                item_dict = {}
                for i, field in enumerate(standard_fields):
                    if i < len(item):
                        item_dict[field] = item[i]
                result.append(item_dict)
        
        return result
    
    def _convert_single_data_point(
        self,
        item: Dict,
        exchange: DataSource,
        symbol: str,
        interval: str,
        config: Dict
    ) -> StandardizedMarketData:
        """Convert a single data point to standardized format"""
        
        # Map fields according to exchange configuration
        mapped_data = {}
        field_mapping = config['field_mapping']
        
        for source_field, target_field in field_mapping.items():
            if source_field in item:
                mapped_data[target_field] = item[source_field]
        
        # Handle timestamp conversion
        timestamp_field = config['timestamp_field']
        if timestamp_field in item:
            timestamp = self._convert_timestamp(
                item[timestamp_field], 
                config['timestamp_unit']
            )
        else:
            timestamp = datetime.utcnow()
        
        # Extract OHLCV data with defaults
        open_price = float(mapped_data.get('open', 0.0))
        high_price = float(mapped_data.get('high', 0.0))
        low_price = float(mapped_data.get('low', 0.0))
        close_price = float(mapped_data.get('close', 0.0))
        volume = float(mapped_data.get('volume', 0.0))
        
        # Extract optional fields
        quote_volume = mapped_data.get('quote_volume')
        if quote_volume is not None:
            quote_volume = float(quote_volume)
        
        trades_count = mapped_data.get('trades_count')
        if trades_count is not None:
            trades_count = int(trades_count)
        
        taker_buy_base_volume = mapped_data.get('taker_buy_base_volume')
        if taker_buy_base_volume is not None:
            taker_buy_base_volume = float(taker_buy_base_volume)
        
        taker_buy_quote_volume = mapped_data.get('taker_buy_quote_volume')
        if taker_buy_quote_volume is not None:
            taker_buy_quote_volume = float(taker_buy_quote_volume)
        
        # Create standardized market data
        return StandardizedMarketData(
            symbol=symbol,
            timestamp=timestamp,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
            interval=interval,
            exchange=exchange.value,
            source=exchange,
            quote_volume=quote_volume,
            trades_count=trades_count,
            taker_buy_base_volume=taker_buy_base_volume,
            taker_buy_quote_volume=taker_buy_quote_volume
        )
    
    def _convert_timestamp(self, timestamp: Union[int, str, float], unit: str) -> datetime:
        """Convert timestamp to datetime object"""
        try:
            if isinstance(timestamp, str):
                timestamp = float(timestamp)
            
            if unit == 'ms':
                return datetime.fromtimestamp(timestamp / 1000, tz=None)
            elif unit == 's':
                return datetime.fromtimestamp(timestamp, tz=None)
            elif unit == 'us':
                return datetime.fromtimestamp(timestamp / 1000000, tz=None)
            else:
                raise ValueError(f"Unsupported timestamp unit: {unit}")
        except Exception as e:
            logger.error(f"Failed to convert timestamp {timestamp} with unit {unit}: {e}")
            return datetime.utcnow()
    
    def validate_data_consistency(
        self, 
        data: List[StandardizedMarketData]
    ) -> Dict[str, Any]:
        """
        Validate data consistency across multiple data points.
        
        Returns:
            Dictionary with validation results
        """
        if not data:
            return {'valid': False, 'errors': ['No data provided']}
        
        errors = []
        warnings = []
        
        # Check for valid data points
        valid_count = sum(1 for item in data if item.is_valid)
        invalid_count = len(data) - valid_count
        
        if invalid_count > 0:
            warnings.append(f"{invalid_count} invalid data points out of {len(data)}")
        
        # Check for consistent symbol
        symbols = set(item.symbol for item in data)
        if len(symbols) > 1:
            errors.append(f"Multiple symbols found: {symbols}")
        
        # Check for consistent interval
        intervals = set(item.interval for item in data)
        if len(intervals) > 1:
            errors.append(f"Multiple intervals found: {intervals}")
        
        # Check for consistent exchange
        exchanges = set(item.exchange for item in data)
        if len(exchanges) > 1:
            errors.append(f"Multiple exchanges found: {exchanges}")
        
        # Check timestamp ordering
        timestamps = [item.timestamp for item in data if item.is_valid]
        if len(timestamps) > 1:
            if timestamps != sorted(timestamps):
                warnings.append("Timestamps are not in chronological order")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'valid_count': valid_count,
            'invalid_count': invalid_count,
            'total_count': len(data)
        }


class ExchangeOHLCVInterface:
    """
    Unified interface for OHLCV data across all exchanges.
    
    This class provides a consistent interface for getting OHLCV data
    from any exchange, ensuring complete standardization.
    """
    
    def __init__(self):
        """Initialize the interface with standardizer"""
        self.standardizer = OHLCVDataStandardizer()
        self._exchange_instances = {}
    
    def register_exchange(
        self, 
        exchange: DataSource, 
        exchange_instance: Any
    ) -> None:
        """Register an exchange instance"""
        self._exchange_instances[exchange] = exchange_instance
        logger.info(f"Registered exchange: {exchange.value}")
    
    async def get_klines(
        self,
        exchange: DataSource,
        symbol: str,
        interval: str,
        limit: int = 100
    ) -> List[StandardizedMarketData]:
        """
        Get standardized klines data from any exchange.
        
        Args:
            exchange: Exchange to get data from
            symbol: Trading symbol
            interval: Data interval
            limit: Maximum number of records
            
        Returns:
            List of standardized market data
        """
        try:
            # Get exchange instance
            exchange_instance = self._exchange_instances.get(exchange)
            if not exchange_instance:
                raise ValueError(f"Exchange {exchange.value} not registered")
            
            # Get raw data from exchange
            raw_data = await self._get_raw_klines(
                exchange_instance, symbol, interval, limit
            )
            
            # Standardize data
            standardized_data = self.standardizer.standardize_data(
                raw_data, exchange, symbol, interval
            )
            
            # Validate consistency
            validation_result = self.standardizer.validate_data_consistency(standardized_data)
            if not validation_result['valid']:
                logger.warning(f"Data consistency issues: {validation_result['errors']}")
            
            return standardized_data
            
        except Exception as e:
            logger.error(f"Failed to get klines from {exchange.value}: {e}")
            raise
    
    async def _get_raw_klines(
        self,
        exchange_instance: Any,
        symbol: str,
        interval: str,
        limit: int
    ) -> Union[List[Dict], List[List]]:
        """Get raw klines data from exchange instance"""
        # Try different method names that exchanges might use
        method_names = ['get_klines', '_get_klines', 'get_klines_raw', '_get_klines_raw']
        
        for method_name in method_names:
            if hasattr(exchange_instance, method_name):
                method = getattr(exchange_instance, method_name)
                try:
                    return await method(symbol, interval, limit)
                except Exception as e:
                    logger.debug(f"Method {method_name} failed: {e}")
                    continue
        
        raise ValueError(f"No suitable klines method found on exchange instance")
    
    def get_exchange_schema(self, exchange: DataSource) -> Dict[str, Any]:
        """Get the expected schema for an exchange"""
        config = self.standardizer.exchange_configs.get(exchange, {})
        return {
            'exchange': exchange.value,
            'timestamp_field': config.get('timestamp_field'),
            'timestamp_unit': config.get('timestamp_unit'),
            'field_mapping': config.get('field_mapping', {}),
            'supported_intervals': [interval.value for interval in Interval]
        }
    
    def validate_exchange_data(
        self,
        raw_data: Union[List[Dict], List[List], pd.DataFrame],
        exchange: DataSource
    ) -> Dict[str, Any]:
        """Validate that raw exchange data can be standardized"""
        try:
            # Try to standardize a small sample
            sample_data = raw_data[:5] if len(raw_data) > 5 else raw_data
            standardized = self.standardizer.standardize_data(
                sample_data, exchange, "TEST", "1m"
            )
            
            return {
                'can_standardize': True,
                'sample_size': len(sample_data),
                'valid_samples': sum(1 for item in standardized if item.is_valid),
                'errors': []
            }
            
        except Exception as e:
            return {
                'can_standardize': False,
                'sample_size': 0,
                'valid_samples': 0,
                'errors': [str(e)]
            }


# Convenience functions for easy usage
def create_standardized_market_data(
    symbol: str,
    timestamp: datetime,
    open_price: float,
    high_price: float,
    low_price: float,
    close_price: float,
    volume: float,
    interval: str,
    exchange: str,
    **kwargs
) -> StandardizedMarketData:
    """Create a standardized market data object"""
    return StandardizedMarketData(
        symbol=symbol,
        timestamp=timestamp,
        open=open_price,
        high=high_price,
        low=low_price,
        close=close_price,
        volume=volume,
        interval=interval,
        exchange=exchange,
        source=DataSource(exchange),
        **kwargs
    )


def standardize_exchange_data(
    raw_data: Union[List[Dict], List[List], pd.DataFrame],
    exchange: str,
    symbol: str,
    interval: str
) -> List[StandardizedMarketData]:
    """Convenience function to standardize exchange data"""
    standardizer = OHLCVDataStandardizer()
    return standardizer.standardize_data(
        raw_data, DataSource(exchange), symbol, interval
    )


# Global instance for easy access
ohlcv_interface = ExchangeOHLCVInterface()