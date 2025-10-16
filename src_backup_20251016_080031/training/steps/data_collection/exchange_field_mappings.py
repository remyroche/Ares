from src.utils.tprint import tprint

from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
"""
Exchange Field Mappings

This module contains comprehensive field mappings for different exchanges
to standardize data collection and validation across all supported exchanges.

Supported Exchanges:
- Binance
- Coinbase
- Kraken
- Gate.io
- MEXC
- OKX
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum

from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation
from src.utils.error_handler import handles_errors
from src.utils.common_operations import safe_fillna, safe_to_parquet, safe_read_parquet
from src.utils.common_utilities import validate_dataframe_columns, safe_dataframe_operation
from src.utils.validation import validate_data_quality
import logging

class ExchangeType(Enum):
    """Supported exchange types."""
    BINANCE = 'binance'
    COINBASE = 'coinbase'
    KRAKEN = 'kraken'
    GATEIO = 'gateio'
    MEXC = 'mexc'
    OKX = 'okx'

@dataclass
class FieldMapping:
    """Field mapping definition for a specific exchange."""
    exchange: ExchangeType
    data_type: str
    field_mappings: Dict[str, str]
    data_types: Dict[str, str]
    required_fields: List[str]
    optional_fields: List[str]
    timestamp_format: str
    timestamp_field: str
EXCHANGE_FIELD_MAPPINGS = {ExchangeType.BINANCE: {'klines': FieldMapping(exchange = ExchangeType.BINANCE, data_type='klines', field_mappings={'timestamp': 'open_time', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume', 'quote_asset_volume': 'quote_asset_volume', 'number_of_trades': 'number_of_trades', 'taker_buy_base_asset_volume': 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume': 'taker_buy_quote_asset_volume', 'close_time': 'close_time'}, data_types={'timestamp': 'int64', 'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'float64', 'quote_asset_volume': 'float64', 'number_of_trades': 'int64', 'taker_buy_base_asset_volume': 'float64', 'taker_buy_quote_asset_volume': 'float64', 'close_time': 'int64'}, required_fields=['timestamp', 'open', 'high', 'low', 'close', 'volume'], optional_fields=['quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'close_time'], timestamp_format='ms', timestamp_field='open_time'), 'aggtrades': FieldMapping(exchange = ExchangeType.BINANCE, data_type='aggtrades', field_mappings={'timestamp': 'T', 'price': 'p', 'quantity': 'q', 'is_buyer_maker': 'm', 'trade_id': 'a'}, data_types={'timestamp': 'int64', 'price': 'float64', 'quantity': 'float64', 'is_buyer_maker': 'bool', 'trade_id': 'int64'}, required_fields=['timestamp', 'price', 'quantity'], optional_fields=['is_buyer_maker', 'trade_id'], timestamp_format='ms', timestamp_field='T')}, ExchangeType.COINBASE: {'klines': FieldMapping(exchange = ExchangeType.COINBASE, data_type='klines', field_mappings={'timestamp': 'timestamp', 'open': 'price_open', 'high': 'price_high', 'low': 'price_low', 'close': 'price_close', 'volume': 'volume'}, data_types={'timestamp': 'int64', 'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'float64'}, required_fields=['timestamp', 'open', 'high', 'low', 'close', 'volume'], optional_fields=[], timestamp_format='s', timestamp_field='timestamp'), 'aggtrades': FieldMapping(exchange = ExchangeType.COINBASE, data_type='aggtrades', field_mappings={'timestamp': 'timestamp', 'price': 'price', 'quantity': 'size', 'is_buyer_maker': 'side', 'trade_id': 'trade_id'}, data_types={'timestamp': 'int64', 'price': 'float64', 'quantity': 'float64', 'is_buyer_maker': 'string', 'trade_id': 'int64'}, required_fields=['timestamp', 'price', 'quantity'], optional_fields=['is_buyer_maker', 'trade_id'], timestamp_format='s', timestamp_field='timestamp')}, ExchangeType.KRAKEN: {'klines': FieldMapping(exchange = ExchangeType.KRAKEN, data_type='klines', field_mappings={'timestamp': 'time', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'vol'}, data_types={'timestamp': 'int64', 'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'float64'}, required_fields=['timestamp', 'open', 'high', 'low', 'close', 'volume'], optional_fields=[], timestamp_format='s', timestamp_field='time'), 'aggtrades': FieldMapping(exchange = ExchangeType.KRAKEN, data_type='aggtrades', field_mappings={'timestamp': 'time', 'price': 'price', 'quantity': 'vol', 'is_buyer_maker': 'type', 'trade_id': 'id'}, data_types={'timestamp': 'int64', 'price': 'float64', 'quantity': 'float64', 'is_buyer_maker': 'string', 'trade_id': 'string'}, required_fields=['timestamp', 'price', 'quantity'], optional_fields=['is_buyer_maker', 'trade_id'], timestamp_format='s', timestamp_field='time')}, ExchangeType.GATEIO: {'klines': FieldMapping(exchange = ExchangeType.GATEIO, data_type='klines', field_mappings={'timestamp': 't', 'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c', 'volume': 'v'}, data_types={'timestamp': 'int64', 'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'float64'}, required_fields=['timestamp', 'open', 'high', 'low', 'close', 'volume'], optional_fields=[], timestamp_format='s', timestamp_field='t'), 'aggtrades': FieldMapping(exchange = ExchangeType.GATEIO, data_type='aggtrades', field_mappings={'timestamp': 'time', 'price': 'price', 'quantity': 'amount', 'is_buyer_maker': 'side', 'trade_id': 'id'}, data_types={'timestamp': 'int64', 'price': 'float64', 'quantity': 'float64', 'is_buyer_maker': 'string', 'trade_id': 'string'}, required_fields=['timestamp', 'price', 'quantity'], optional_fields=['is_buyer_maker', 'trade_id'], timestamp_format='s', timestamp_field='time')}, ExchangeType.MEXC: {'klines': FieldMapping(exchange = ExchangeType.MEXC, data_type='klines', field_mappings={'timestamp': 'open_time', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}, data_types={'timestamp': 'int64', 'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'float64'}, required_fields=['timestamp', 'open', 'high', 'low', 'close', 'volume'], optional_fields=[], timestamp_format='ms', timestamp_field='open_time'), 'aggtrades': FieldMapping(exchange = ExchangeType.MEXC, data_type='aggtrades', field_mappings={'timestamp': 'T', 'price': 'p', 'quantity': 'q', 'is_buyer_maker': 'm', 'trade_id': 'a'}, data_types={'timestamp': 'int64', 'price': 'float64', 'quantity': 'float64', 'is_buyer_maker': 'bool', 'trade_id': 'int64'}, required_fields=['timestamp', 'price', 'quantity'], optional_fields=['is_buyer_maker', 'trade_id'], timestamp_format='ms', timestamp_field='T')}, ExchangeType.OKX: {'klines': FieldMapping(exchange = ExchangeType.OKX, data_type='klines', field_mappings={'timestamp': 'ts', 'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c', 'volume': 'vol'}, data_types={'timestamp': 'int64', 'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'float64'}, required_fields=['timestamp', 'open', 'high', 'low', 'close', 'volume'], optional_fields=[], timestamp_format='ms', timestamp_field='ts'), 'aggtrades': FieldMapping(exchange = ExchangeType.OKX, data_type='aggtrades', field_mappings={'timestamp': 'ts', 'price': 'px', 'quantity': 'sz', 'is_buyer_maker': 'side', 'trade_id': 'tradeId'}, data_types={'timestamp': 'int64', 'price': 'float64', 'quantity': 'float64', 'is_buyer_maker': 'string', 'trade_id': 'string'}, required_fields=['timestamp', 'price', 'quantity'], optional_fields=['is_buyer_maker', 'trade_id'], timestamp_format='ms', timestamp_field='ts')}}

class ExchangeFieldMapper:
    """Utility class for mapping exchange-specific fields to standardized fields."""
    @log_important_calls

    def __init__(self, exchange: ExchangeType) -> None:
        self.exchange = exchange
        self.mappings = EXCHANGE_FIELD_MAPPINGS.get(exchange, {})

    def get_field_mapping(self, data_type: str) -> Optional[FieldMapping]:
        """Get field mapping for specific data type."""
        return self.mappings.get(data_type)

    def map_fields(self, data_type: str, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map exchange-specific fields to standardized fields."""
        mapping = self.get_field_mapping(data_type)
        if not mapping:
            return raw_data
        mapped_data = {}
        for standard_field, exchange_field in mapping.field_mappings.items():
            if exchange_field in raw_data:
                mapped_data[standard_field] = raw_data[exchange_field]
        return mapped_data

    def get_required_fields(self, data_type: str) -> List[str]:
        """Get list of required fields for data type."""
        mapping = self.get_field_mapping(data_type)
        return mapping.required_fields if mapping else []

    def get_optional_fields(self, data_type: str) -> List[str]:
        """Get list of optional fields for data type."""
        mapping = self.get_field_mapping(data_type)
        return mapping.optional_fields if mapping else []

    def get_timestamp_field(self, data_type: str) -> Optional[str]:
        """Get timestamp field name for data type."""
        mapping = self.get_field_mapping(data_type)
        return mapping.timestamp_field if mapping else None

    def get_timestamp_format(self, data_type: str) -> Optional[str]:
        """Get timestamp format for data type."""
        mapping = self.get_field_mapping(data_type)
        return mapping.timestamp_format if mapping else None

    def convert_timestamp(self, data_type: str, timestamp: Any) -> int:
        """Convert timestamp to milliseconds."""
        mapping = self.get_field_mapping(data_type)
        if not mapping:
            return int(timestamp)
        if mapping.timestamp_format == 'ms':
            return int(timestamp)
        elif mapping.timestamp_format == 's':
            return int(timestamp) * 1000
        else:
            return int(timestamp)

    def validate_required_fields(self, data_type: str, data: Dict[str, Any]) -> List[str]:
        """Validate that all required fields are present."""
        required_fields = self.get_required_fields(data_type)
        missing_fields = []
        for field in required_fields:
            if field not in data:
                missing_fields.append(field)
        return missing_fields

    def get_data_types(self, data_type: str) -> Dict[str, str]:
        """Get expected data types for fields."""
        mapping = self.get_field_mapping(data_type)
        return mapping.data_types if mapping else {}

def get_exchange_mapper(exchange_name: str) -> ExchangeFieldMapper:
    """Get field mapper for exchange by name."""
    try:
        exchange_type = ExchangeType(exchange_name.lower())
        return ExchangeFieldMapper(exchange_type)
    except ValueError:
        raise ValueError(f'Unsupported exchange: {exchange_name}')

def list_supported_exchanges() -> List[str]:
    """List all supported exchanges."""
    return [exchange.value for exchange in ExchangeType]

def get_exchange_field_mapping(exchange: str, data_type: str) -> Optional[FieldMapping]:
    """Get field mapping for specific exchange and data type."""
    mapper = get_exchange_mapper(exchange)
    return mapper.get_field_mapping(data_type)
if __name__ == '__main__':
    tprint('🔍 Testing Exchange Field Mappings')
    tprint('=' * 50)
    binance_mapper = get_exchange_mapper('binance')
    raw_klines = {'open_time': 1640995200000, 'open': '3000.0', 'high': '3100.0', 'low': '2900.0', 'close': '3050.0', 'volume': '1000.0'}
    mapped_klines = binance_mapper.map_fields('klines', raw_klines)
    tprint(f'✅ Binance klines mapping: {mapped_klines}')
    raw_aggtrades = {'T': 1640995200000, 'p': '3050.0', 'q': '1.5', 'm': True}
    mapped_aggtrades = binance_mapper.map_fields('aggtrades', raw_aggtrades)
    tprint(f'✅ Binance aggtrades mapping: {mapped_aggtrades}')
    coinbase_mapper = get_exchange_mapper('coinbase')
    raw_coinbase_klines = {'timestamp': 1640995200, 'price_open': '3000.0', 'price_high': '3100.0', 'price_low': '2900.0', 'price_close': '3050.0', 'volume': '1000.0'}
    mapped_coinbase_klines = coinbase_mapper.map_fields('klines', raw_coinbase_klines)
    tprint(f'✅ Coinbase klines mapping: {mapped_coinbase_klines}')
    tprint(f'📋 Supported exchanges: {list_supported_exchanges()}')
    tprint('=' * 50)
    tprint('🎉 Field mapping tests completed successfully!')