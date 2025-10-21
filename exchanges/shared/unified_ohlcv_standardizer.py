"""
Unified OHLCV Data Standardizer

This module provides a comprehensive, unified OHLCV data standardizer that ensures
complete equivalency between all exchanges (binance, bingx, okx, mexc) and full
compatibility with src/utils/data/ utilities.

Features:
- Unified data format across all exchanges
- Complete integration with src/utils/data/ processing pipeline
- Comprehensive validation and error handling
- Exchange-agnostic data processing
- Memory-efficient data handling
- Full compatibility with existing data utilities
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import hashlib

# Import src/utils/data utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.data import (
    DataProcessor, DataQualityFramework, DataCleaner,
    validate_and_fix_data_quality, optimize_dataframe_dtypes,
    check_dataframe_health
)
from src.utils.logger import system_logger
from src.utils.tprint import tprint_data_preview

logger = logging.getLogger(__name__)


class ExchangeType(Enum):
    """Supported exchange types"""
    BINANCE = "binance"
    BINGX = "bingx"
    OKX = "okx"
    MEXC = "mexc"
    GATEIO = "gateio"
    PHEMEX = "phemex"


class DataQualityLevel(Enum):
    """Data quality validation levels"""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    CRITICAL = "critical"


@dataclass
class StandardizedOHLCVData:
    """
    Standardized OHLCV data structure that all exchanges must conform to.
    
    This is the single source of truth for OHLCV data across the entire system.
    All exchanges must convert their data to this exact format.
    """
    # Core OHLCV data (required)
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    interval: str
    
    # Exchange metadata (required)
    exchange: str
    source: ExchangeType
    
    # Additional standardized fields (optional)
    quote_volume: Optional[float] = None
    trades_count: Optional[int] = None
    taker_buy_base_volume: Optional[float] = None
    taker_buy_quote_volume: Optional[float] = None
    
    # Data quality metrics
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    quality_score: float = 100.0
    
    # Processing metadata
    processed_at: Optional[datetime] = None
    raw_data_hash: Optional[str] = None
    
    def __post_init__(self):
        """Validate data after initialization"""
        self._validate_data()
        if not self.processed_at:
            self.processed_at = datetime.now(timezone.utc)
    
    def _validate_data(self) -> None:
        """Validate the OHLCV data for consistency and quality"""
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
        valid_intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
        if self.interval not in valid_intervals:
            errors.append(f"Interval must be one of {valid_intervals}")
        
        # Validate exchange
        valid_exchanges = [exchange.value for exchange in ExchangeType]
        if self.exchange not in valid_exchanges:
            errors.append(f"Exchange must be one of {valid_exchanges}")
        
        self.validation_errors = errors
        self.is_valid = len(errors) == 0
        
        if not self.is_valid:
            logger.warning(f"Invalid OHLCV data: {errors}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'interval': self.interval,
            'exchange': self.exchange,
            'source': self.source.value,
            'quote_volume': self.quote_volume,
            'trades_count': self.trades_count,
            'taker_buy_base_volume': self.taker_buy_base_volume,
            'taker_buy_quote_volume': self.taker_buy_quote_volume,
            'is_valid': self.is_valid,
            'validation_errors': self.validation_errors,
            'quality_score': self.quality_score,
            'processed_at': self.processed_at,
            'raw_data_hash': self.raw_data_hash
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        return pd.DataFrame([self.to_dict()])


class UnifiedOHLCVStandardizer:
    """
    Unified OHLCV data standardizer that ensures complete equivalency across all exchanges.
    
    This class provides a single interface for standardizing OHLCV data from any exchange
    to a unified format that's fully compatible with src/utils/data/ utilities.
    """
    
    def __init__(self, quality_level: DataQualityLevel = DataQualityLevel.STANDARD):
        """Initialize the unified standardizer"""
        self.quality_level = quality_level
        self.logger = system_logger.getChild("UnifiedOHLCVStandardizer")
        
        # Initialize data processing utilities
        self.data_processor = DataProcessor()
        self.quality_framework = DataQualityFramework()
        self.data_cleaner = DataCleaner()
        
        # Exchange-specific field mappings
        self.exchange_mappings = {
            ExchangeType.BINANCE: {
                'timestamp_field': 'open_time',
                'timestamp_unit': 'ms',
                'field_mapping': {
                    'openTime': 'timestamp',
                    'closeTime': 'close_time',
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
            ExchangeType.BINGX: {
                'timestamp_field': 'open_time',
                'timestamp_unit': 'ms',
                'field_mapping': {
                    'openTime': 'timestamp',
                    'closeTime': 'close_time',
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
            ExchangeType.OKX: {
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
            ExchangeType.MEXC: {
                'timestamp_field': 'open_time',
                'timestamp_unit': 'ms',
                'field_mapping': {
                    'openTime': 'timestamp',
                    'closeTime': 'close_time',
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
            ExchangeType.GATEIO: {
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
            ExchangeType.PHEMEX: {
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
        
        self.logger.info(f"✅ UnifiedOHLCVStandardizer initialized with {quality_level.value} quality level")
    
    def standardize_data(
        self,
        raw_data: Union[List[Dict], List[List], pd.DataFrame],
        exchange: ExchangeType,
        symbol: str,
        interval: str,
        enable_data_preview: bool = True
    ) -> List[StandardizedOHLCVData]:
        """
        Standardize raw exchange data to unified format.
        
        Args:
            raw_data: Raw data from exchange (list of dicts, list of lists, or DataFrame)
            exchange: Exchange source
            symbol: Trading symbol
            interval: Data interval
            enable_data_preview: Whether to show data preview using tprint_data_preview
            
        Returns:
            List of standardized OHLCV data objects
        """
        try:
            # Convert to list of dictionaries for consistent processing
            data_list = self._normalize_input_format(raw_data)
            
            # Show data preview if enabled
            if enable_data_preview and data_list:
                tprint_data_preview(
                    data_list, 
                    name=f"Raw {exchange.value} data for {symbol} ({interval})",
                    max_rows=3,
                    level="DEBUG"
                )
            
            # Get exchange configuration
            config = self.exchange_mappings.get(exchange)
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
                    self.logger.error(f"Failed to convert data point: {e}")
                    # Create invalid data point for tracking
                    invalid_data = StandardizedOHLCVData(
                        symbol=symbol,
                        timestamp=datetime.now(timezone.utc),
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
            
            # Apply quality validation and cleaning
            standardized_data = self._apply_quality_processing(standardized_data)
            
            # Show standardized data preview if enabled
            if enable_data_preview and standardized_data:
                # Convert to DataFrame for preview
                preview_df = pd.DataFrame([item.to_dict() for item in standardized_data])
                tprint_data_preview(
                    preview_df, 
                    name=f"Standardized {exchange.value} data for {symbol} ({interval})",
                    max_rows=3,
                    level="DEBUG"
                )
            
            self.logger.info(f"✅ Standardized {len(standardized_data)} data points from {exchange.value}")
            return standardized_data
            
        except Exception as e:
            self.logger.error(f"Failed to standardize data from {exchange.value}: {e}")
            raise
    
    def standardize_to_dataframe(
        self,
        raw_data: Union[List[Dict], List[List], pd.DataFrame],
        exchange: ExchangeType,
        symbol: str,
        interval: str
    ) -> pd.DataFrame:
        """
        Standardize raw exchange data to unified DataFrame format.
        
        This method ensures full compatibility with src/utils/data/ utilities.
        
        Args:
            raw_data: Raw data from exchange
            exchange: Exchange source
            symbol: Trading symbol
            interval: Data interval
            
        Returns:
            Standardized DataFrame compatible with src/utils/data/
        """
        try:
            # Get standardized data objects
            standardized_objects = self.standardize_data(raw_data, exchange, symbol, interval)
            
            # Convert to DataFrame
            df_data = [obj.to_dict() for obj in standardized_objects]
            df = pd.DataFrame(df_data)
            
            # Set timestamp as index for time series compatibility
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            
            # Apply data processing optimizations
            df = self._apply_data_processing_optimizations(df)
            
            # Validate with src/utils/data/ framework
            self._validate_with_data_framework(df, f"{exchange.value} standardization")
            
            self.logger.info(f"✅ Standardized DataFrame created: {df.shape} with {len(standardized_objects)} records")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to create standardized DataFrame: {e}")
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
        exchange: ExchangeType,
        symbol: str,
        interval: str,
        config: Dict
    ) -> StandardizedOHLCVData:
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
            timestamp = datetime.now(timezone.utc)
        
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
        
        # Create standardized OHLCV data
        return StandardizedOHLCVData(
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
                return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
            elif unit == 's':
                return datetime.fromtimestamp(timestamp, tz=timezone.utc)
            elif unit == 'us':
                return datetime.fromtimestamp(timestamp / 1000000, tz=timezone.utc)
            else:
                raise ValueError(f"Unsupported timestamp unit: {unit}")
        except Exception as e:
            self.logger.error(f"Failed to convert timestamp {timestamp} with unit {unit}: {e}")
            return datetime.now(timezone.utc)
    
    def _apply_quality_processing(self, data: List[StandardizedOHLCVData]) -> List[StandardizedOHLCVData]:
        """Apply quality processing to standardized data"""
        if self.quality_level == DataQualityLevel.BASIC:
            return data
        
        processed_data = []
        for item in data:
            if item.is_valid:
                # Apply quality scoring
                item.quality_score = self._calculate_quality_score(item)
                
                # Apply quality-based filtering
                if self.quality_level in [DataQualityLevel.STRICT, DataQualityLevel.CRITICAL]:
                    if item.quality_score < 80.0:  # Adjust threshold as needed
                        item.is_valid = False
                        item.validation_errors.append(f"Quality score {item.quality_score} below threshold")
            
            processed_data.append(item)
        
        return processed_data
    
    def _calculate_quality_score(self, item: StandardizedOHLCVData) -> float:
        """Calculate quality score for a single data point"""
        score = 100.0
        
        # Check for zero values
        if item.volume == 0:
            score -= 10.0
        
        # Check for negative values
        if any(val < 0 for val in [item.open, item.high, item.low, item.close, item.volume]):
            score -= 20.0
        
        # Check OHLC consistency
        if item.high < max(item.open, item.close) or item.low > min(item.open, item.close):
            score -= 15.0
        
        # Check for extreme values (basic outlier detection)
        if item.volume > 0:
            price_volatility = abs(item.close - item.open) / item.open if item.open > 0 else 0
            if price_volatility > 0.5:  # 50% price change
                score -= 5.0
        
        return max(0.0, score)
    
    def _apply_data_processing_optimizations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply data processing optimizations using src/utils/data/ utilities"""
        try:
            # Optimize data types
            df = self.data_processor.optimize_dataframe_dtypes(df)
            
            # Apply feature-specific optimizations
            df = self.data_processor.apply_feature_specific_optimization(df)
            
            return df
            
        except Exception as e:
            self.logger.warning(f"Data processing optimization failed: {e}")
            return df
    
    def _validate_with_data_framework(self, df: pd.DataFrame, context: str) -> None:
        """Validate DataFrame using src/utils/data/ framework"""
        try:
            # Use the quality framework for validation
            quality_result = self.quality_framework.validate_dataframe_quality(df, context)
            
            if not quality_result.passed:
                self.logger.warning(f"Data quality validation failed: {quality_result.issues}")
            
            # Log quality metrics
            self.logger.info(f"Data quality score: {quality_result.quality_score:.2f}")
            
        except Exception as e:
            self.logger.warning(f"Data framework validation failed: {e}")
    
    def validate_data_consistency(
        self, 
        data: List[StandardizedOHLCVData]
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
    
    def get_exchange_schema(self, exchange: ExchangeType) -> Dict[str, Any]:
        """Get the expected schema for an exchange"""
        config = self.exchange_mappings.get(exchange, {})
        return {
            'exchange': exchange.value,
            'timestamp_field': config.get('timestamp_field'),
            'timestamp_unit': config.get('timestamp_unit'),
            'field_mapping': config.get('field_mapping', {}),
            'supported_intervals': ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
        }


# Global instance for easy access
unified_ohlcv_standardizer = UnifiedOHLCVStandardizer()


# Convenience functions for easy usage
def standardize_exchange_ohlcv(
    raw_data: Union[List[Dict], List[List], pd.DataFrame],
    exchange: str,
    symbol: str,
    interval: str,
    quality_level: str = "standard"
) -> pd.DataFrame:
    """
    Convenience function to standardize exchange OHLCV data.
    
    Args:
        raw_data: Raw data from exchange
        exchange: Exchange name (binance, bingx, okx, mexc, etc.)
        symbol: Trading symbol
        interval: Data interval
        quality_level: Quality validation level
        
    Returns:
        Standardized DataFrame compatible with src/utils/data/
    """
    try:
        exchange_type = ExchangeType(exchange.lower())
        quality_level_enum = DataQualityLevel(quality_level.lower())
        
        standardizer = UnifiedOHLCVStandardizer(quality_level_enum)
        return standardizer.standardize_to_dataframe(raw_data, exchange_type, symbol, interval)
        
    except ValueError as e:
        raise ValueError(f"Invalid exchange or quality level: {e}")


def validate_ohlcv_equivalency(
    data1: pd.DataFrame,
    data2: pd.DataFrame,
    tolerance: float = 1e-6
) -> Dict[str, Any]:
    """
    Validate that two OHLCV DataFrames are equivalent.
    
    Args:
        data1: First DataFrame
        data2: Second DataFrame
        tolerance: Numerical tolerance for comparison
        
    Returns:
        Validation results
    """
    results = {
        'equivalent': True,
        'errors': [],
        'warnings': [],
        'differences': {}
    }
    
    try:
        # Check shapes
        if data1.shape != data2.shape:
            results['equivalent'] = False
            results['errors'].append(f"Shape mismatch: {data1.shape} vs {data2.shape}")
            return results
        
        # Check columns
        if not data1.columns.equals(data2.columns):
            results['equivalent'] = False
            results['errors'].append(f"Column mismatch: {list(data1.columns)} vs {list(data2.columns)}")
            return results
        
        # Check data types
        for col in data1.columns:
            if data1[col].dtype != data2[col].dtype:
                results['warnings'].append(f"Data type mismatch for {col}: {data1[col].dtype} vs {data2[col].dtype}")
        
        # Check numerical values
        for col in data1.select_dtypes(include=[np.number]).columns:
            if not np.allclose(data1[col], data2[col], rtol=tolerance, equal_nan=True):
                diff_count = np.sum(~np.isclose(data1[col], data2[col], rtol=tolerance, equal_nan=True))
                results['differences'][col] = diff_count
                if diff_count > 0:
                    results['warnings'].append(f"Numerical differences in {col}: {diff_count} values")
        
        # Check string values
        for col in data1.select_dtypes(include=['object']).columns:
            if not data1[col].equals(data2[col]):
                diff_count = (data1[col] != data2[col]).sum()
                results['differences'][col] = diff_count
                if diff_count > 0:
                    results['warnings'].append(f"String differences in {col}: {diff_count} values")
        
        # Overall assessment
        if results['differences']:
            results['equivalent'] = False
        
        return results
        
    except Exception as e:
        results['equivalent'] = False
        results['errors'].append(f"Validation error: {str(e)}")
        return results