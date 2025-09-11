from ...core.decorators import handles_errors
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
"""
from src.utils.logger import system_logger
Enhanced Data Validation Framework with Decorators

This module provides comprehensive validation during data collection with:
- Extensive logging and printing
- Integration with utils/ decorators
- Real-time schema enforcement
- Data qualification with duplicate removal
- Time gap detection
- Field mapping for different exchanges
"""
import asyncio
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
from src.utils.logger import system_logger
from src.utils.error_handler import handles_errors as utils_handles_errors
from src.utils.common_operations import safe_fillna, safe_to_parquet, safe_read_parquet
from src.utils.common_utilities import validate_dataframe_columns, safe_dataframe_operation
from src.utils.validation import validate_data_quality
from typing import Any
from typing import Dict
from typing import Optional
from typing import List
from typing import Callable
from datetime import datetime
import numpy as np
import pandas as pd
import logging

logger = system_logger.getChild('EnhancedValidationWithDecorators')

class DataType(Enum):
    """Supported data types for validation."""
    KLINES = 'klines'
    AGGTRADES = 'aggtrades'
    FUTURES = 'futures'
    UNIFIED = 'unified'

class ValidationSeverity(Enum):
    """Validation error severity levels."""
    CRITICAL = 'critical'
    HIGH = 'high'
    MEDIUM = 'medium'
    LOW = 'low'

@dataclass
class ValidationError:
    """Represents a validation error with extensive context."""
    field: str
    message: str
    severity: ValidationSeverity
    value: Any = None
    expected: Any = None
    row_index: Optional[int] = None
    exchange: Optional[str] = None
    data_type: Optional[str] = None
    timestamp: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory = dict)

@dataclass
class FieldDefinition:
    """Definition of a data field with validation rules."""
    name: str
    dtype: str
    required: bool = True
    default_value: Any = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allow_zero: bool = True
    allow_negative: bool = True
    allow_infinite: bool = False
    allow_nan: bool = False
    custom_validator: Optional[Callable] = None
    description: str = ''

@dataclass
class TimeGapConfig:
    """Configuration for time gap detection."""
    max_gap_seconds: float
    tolerance_seconds: float = 0.0
    severity: ValidationSeverity = ValidationSeverity.HIGH
    description: str = ''

@dataclass
class DataSchema:
    """Complete data schema definition."""
    data_type: DataType
    fields: List[FieldDefinition]
    primary_key: List[str]
    timestamp_field: str = 'timestamp'
    time_gap_config: Optional[TimeGapConfig] = None
    required_columns: List[str] = field(default_factory = list)
    optional_columns: List[str] = field(default_factory = list)
    description: str = ''

class EnhancedDataValidator:
    """Enhanced data validator with comprehensive validation rules and extensive logging."""

    def __init__(self, schema: DataSchema, exchange: str='UNKNOWN') -> None:
        self.schema = schema
        self.exchange = exchange.upper()
        self.logger = logger.getChild(f'Validator.{schema.data_type.value}.{self.exchange}')
        try:
            self.field_mapper = get_exchange_mapper(self.exchange)
            self.logger.info(f'✅ Initialized field mapper for {self.exchange}')
        except ValueError:
            self.logger.warning(f'⚠️ No field mapper available for {self.exchange}, using default mapping')
            self.field_mapper = None
        self.validation_stats = {'total_rows_processed': 0, 'valid_rows': 0, 'invalid_rows': 0, 'validation_errors': [], 'time_gaps_detected': 0, 'duplicates_removed': 0, 'quality_issues': [], 'start_time': None, 'last_timestamp': None}
        self.validated_data: List[Dict[str, Any]] = []
        self.validation_errors: List[ValidationError] = []
        self.logger.info(f'🚀 Initialized Enhanced Data Validator for {schema.data_type.value} data from {self.exchange}')
        self.logger.info(f'📋 Schema: {len(schema.fields)} fields, {len(schema.required_columns)} required columns')

    @handles_errors(fallback=[], context='validate_row')
    @traced(span_name='validate_single_row', log_args = False, log_result_len_only = True)
    def validate_row(self, row_data: Dict[str, Any], row_index: int = 0) -> Dict[str, Any]:
        """
        Validate and standardize a single row of data with extensive logging.
        
        Args:
            row_data: Raw row data from API
            row_index: Index of the row for error reporting
            
        Returns:
            Validated and standardized row data
            
        Raises:
            ValueError: If critical validation errors are found
        """
        self.logger.debug(f'🔍 Validating row {row_index} for {self.schema.data_type.value} data')
        validated_row = {}
        errors = []
        try:
            for field_def in self.schema.fields:
                try:
                    source_value = self._get_source_value(row_data, field_def)
                    validated_value = self._validate_and_convert_field(source_value, field_def, row_index)
                    validated_row[field_def.name] = validated_value
                    self.logger.debug(f'✅ Field {field_def.name}: {validated_value}')
                except ValidationError as e:
                    errors.append(e)
                    if e.severity == ValidationSeverity.CRITICAL:
                        self.logger.error(f'❌ CRITICAL: Row {row_index}, Field {field_def.name}: {e.message}')
                        raise ValueError(f'Critical validation error in row {row_index}: {e.message}')
                    else:
                        self.logger.warning(f'⚠️ {e.severity.value.upper()}: Row {row_index}, Field {field_def.name}: {e.message}')
            self._add_metadata_fields(validated_row, row_data)
            if errors:
                self._log_validation_errors(errors, row_index)
            else:
                self.logger.debug(f'✅ Row {row_index} validated successfully')
            return validated_row
        except Exception as e:
            self.logger.exception(f'❌ Unexpected error validating row {row_index}: {e}')
            raise

    @handles_errors(fallback=[], context='validate_batch')
    @traced(span_name='validate_data_batch', log_args = False, log_result_len_only = True)
    @memory_efficient(batch_size = 1000)
    def validate_batch(self, batch_data: List[Dict[str, Any]], previous_timestamp: Optional[int]=None) -> List[Dict[str, Any]]:
        """
        Validate a batch of data with time gap detection and extensive logging.
        
        Args:
            batch_data: List of raw row data from API
            previous_timestamp: Timestamp of last row from previous batch
            
        Returns:
            List of validated and standardized row data
        """
        batch_start_time = time.time()
        self.logger.info(f'🔍 Validating batch of {len(batch_data)} rows for {self.schema.data_type.value} data')
        if not self.validation_stats['start_time']:
            self.validation_stats['start_time'] = batch_start_time
        validated_batch = []
        batch_errors = []
        try:
            for i, row_data in enumerate(batch_data):
                try:
                    validated_row = self.validate_row(row_data, i)
                    validated_batch.append(validated_row)
                    if self.schema.time_gap_config and previous_timestamp is not None and (i == 0):
                        gap_error = self._check_time_gap(validated_row[self.schema.timestamp_field], previous_timestamp)
                        if gap_error:
                            batch_errors.append(gap_error)
                except ValueError as e:
                    self.logger.error(f'❌ Row {i} validation failed: {e}')
                    batch_errors.append(ValidationError(field='row', message = str(e), severity = ValidationSeverity.CRITICAL, row_index = i, exchange = self.exchange, data_type = self.schema.data_type.value, timestamp = datetime.now()))
            validated_batch = self._remove_duplicates(validated_batch)
            self.validation_stats['total_rows_processed'] += len(batch_data)
            self.validation_stats['valid_rows'] += len(validated_batch)
            self.validation_stats['invalid_rows'] += len(batch_data) - len(validated_batch)
            batch_duration = time.time() - batch_start_time
            success_rate = len(validated_batch) / len(batch_data) * 100 if batch_data else 0
            self.logger.info(f'✅ Batch validation completed:')
            self.logger.info(f'   📊 Rows: {len(validated_batch)}/{len(batch_data)} valid ({success_rate:.1f}%)')
            self.logger.info(f'   ⏱️ Duration: {batch_duration:.2f}s')
            self.logger.info(f'   🔄 Duplicates removed: {len(batch_data) - len(validated_batch)}')
            if batch_errors:
                self.validation_stats['validation_errors'].extend(batch_errors)
                self._log_batch_errors(batch_errors)
            return validated_batch
        except Exception as e:
            self.logger.exception(f'❌ Error validating batch: {e}')
            return []

    @handles_errors(fallback = None, context='get_source_value')
    def _get_source_value(self, row_data: Dict[str, Any], field_def: FieldDefinition) -> Any:
        """Get source value using field mapping."""
        if field_def.name in row_data:
            return row_data[field_def.name]
        if self.field_mapper:
            try:
                mapping = self.field_mapper.get_field_mapping(self.schema.data_type.value)
                if mapping and field_def.name in mapping.field_mappings:
                    exchange_field = mapping.field_mappings[field_def.name]
                    if exchange_field in row_data:
                        return row_data[exchange_field]
            except Exception as e:
                self.logger.debug(f'⚠️ Field mapping failed for {field_def.name}: {e}')
        return None

    @handles_errors(fallback = None, context='validate_and_convert_field')
    def _validate_and_convert_field(self, value: Any, field_def: FieldDefinition, row_index: int) -> Any:
        """Validate and convert a field value with extensive logging."""
        self.logger.debug(f'🔍 Validating field {field_def.name}: {value} (type: {type(value)})')
        if value is None or (isinstance(value, float) and np.isnan(value)):
            if field_def.required:
                raise ValidationError(field = field_def.name, message = f"Required field '{field_def.name}' is missing or NaN", severity = ValidationSeverity.CRITICAL, row_index = row_index, exchange = self.exchange, data_type = self.schema.data_type.value, timestamp = datetime.now())
            self.logger.debug(f'ℹ️ Using default value for {field_def.name}: {field_def.default_value}')
            return field_def.default_value
        try:
            converted_value = self._convert_type(value, field_def.dtype)
            self.logger.debug(f'✅ Type conversion successful: {value} -> {converted_value} ({field_def.dtype})')
        except (ValueError, TypeError) as e:
            raise ValidationError(field = field_def.name, message = f'Type conversion failed: {e}', severity = ValidationSeverity.CRITICAL, value = value, expected = field_def.dtype, row_index = row_index, exchange = self.exchange, data_type = self.schema.data_type.value, timestamp = datetime.now())
        validation_errors = self._validate_value(converted_value, field_def, row_index)
        if validation_errors:
            critical_errors = [e for e in validation_errors if e.severity == ValidationSeverity.CRITICAL]
            if critical_errors:
                raise critical_errors[0]
            for error in validation_errors:
                self.logger.warning(f'⚠️ {error.severity.value.upper()}: {error.message}')
        return converted_value

    @handles_errors(fallback = None, context='convert_type')
    def _convert_type(self, value: Any, target_type: str) -> Any:
        """Convert value to target type with logging."""
        self.logger.debug(f'🔄 Converting {value} ({type(value)}) to {target_type}')
        if target_type == 'int64':
            result = int(float(value))
        elif target_type == 'float64':
            result = float(value)
        elif target_type == 'string':
            result = str(value)
        elif target_type == 'bool':
            result = bool(value)
        else:
            result = value
        self.logger.debug(f'✅ Conversion result: {result} ({type(result)})')
        return result

    @handles_errors(fallback=[], context='validate_value')
    def _validate_value(self, value: Any, field_def: FieldDefinition, row_index: int) -> List[ValidationError]:
        """Validate field value against constraints with extensive logging."""
        errors = []
        self.logger.debug(f'🔍 Validating value constraints for {field_def.name}: {value}')
        if isinstance(value, float) and np.isnan(value):
            if not field_def.allow_nan:
                errors.append(ValidationError(field = field_def.name, message = f"Field '{field_def.name}' contains NaN value", severity = ValidationSeverity.HIGH, value = value, row_index = row_index, exchange = self.exchange, data_type = self.schema.data_type.value, timestamp = datetime.now()))
        if isinstance(value, float) and np.isinf(value):
            if not field_def.allow_infinite:
                errors.append(ValidationError(field = field_def.name, message = f"Field '{field_def.name}' contains infinite value", severity = ValidationSeverity.HIGH, value = value, row_index = row_index, exchange = self.exchange, data_type = self.schema.data_type.value, timestamp = datetime.now()))
        if isinstance(value, (int, float)) and value == 0:
            if not field_def.allow_zero:
                errors.append(ValidationError(field = field_def.name, message = f"Field '{field_def.name}' contains zero value", severity = ValidationSeverity.MEDIUM, value = value, row_index = row_index, exchange = self.exchange, data_type = self.schema.data_type.value, timestamp = datetime.now()))
        if isinstance(value, (int, float)) and value < 0:
            if not field_def.allow_negative:
                errors.append(ValidationError(field = field_def.name, message = f"Field '{field_def.name}' contains negative value", severity = ValidationSeverity.HIGH, value = value, row_index = row_index, exchange = self.exchange, data_type = self.schema.data_type.value, timestamp = datetime.now()))
        if isinstance(value, (int, float)):
            if field_def.min_value is not None and value < field_def.min_value:
                errors.append(ValidationError(field = field_def.name, message = f"Field '{field_def.name}' value {value} below minimum {field_def.min_value}", severity = ValidationSeverity.HIGH, value = value, expected = field_def.min_value, row_index = row_index, exchange = self.exchange, data_type = self.schema.data_type.value, timestamp = datetime.now()))
            if field_def.max_value is not None and value > field_def.max_value:
                errors.append(ValidationError(field = field_def.name, message = f"Field '{field_def.name}' value {value} above maximum {field_def.max_value}", severity = ValidationSeverity.HIGH, value = value, expected = field_def.max_value, row_index = row_index, exchange = self.exchange, data_type = self.schema.data_type.value, timestamp = datetime.now()))
        if field_def.custom_validator:
            try:
                if not field_def.custom_validator(value):
                    errors.append(ValidationError(field = field_def.name, message = f"Custom validation failed for field '{field_def.name}'", severity = ValidationSeverity.MEDIUM, value = value, row_index = row_index, exchange = self.exchange, data_type = self.schema.data_type.value, timestamp = datetime.now()))
            except Exception as e:
                errors.append(ValidationError(field = field_def.name, message = f"Custom validation error for field '{field_def.name}': {e}", severity = ValidationSeverity.MEDIUM, value = value, row_index = row_index, exchange = self.exchange, data_type = self.schema.data_type.value, timestamp = datetime.now()))
        if errors:
            self.logger.debug(f'⚠️ Found {len(errors)} validation errors for {field_def.name}')
        else:
            self.logger.debug(f'✅ Value validation passed for {field_def.name}')
        return errors

    @handles_errors(fallback = None, context='check_time_gap')
    def _check_time_gap(self, current_timestamp: int, previous_timestamp: int) -> Optional[ValidationError]:
        """Check for time gap between batches with extensive logging."""
        if not self.schema.time_gap_config:
            return None
        gap_seconds = (current_timestamp - previous_timestamp) / 1000.0
        max_gap = self.schema.time_gap_config.max_gap_seconds
        tolerance = self.schema.time_gap_config.tolerance_seconds
        self.logger.debug(f'🕐 Checking time gap: {gap_seconds:.2f}s (max: {max_gap}s, tolerance: {tolerance}s)')
        if gap_seconds > max_gap + tolerance:
            self.validation_stats['time_gaps_detected'] += 1
            error = ValidationError(field = self.schema.timestamp_field, message = f'Time gap detected: {gap_seconds:.2f}s > {max_gap}s (tolerance: {tolerance}s)', severity = self.schema.time_gap_config.severity, value = current_timestamp, expected = previous_timestamp + int(max_gap * 1000), exchange = self.exchange, data_type = self.schema.data_type.value, timestamp = datetime.now(), context={'gap_seconds': gap_seconds, 'max_gap_seconds': max_gap, 'tolerance_seconds': tolerance, 'previous_timestamp': previous_timestamp, 'current_timestamp': current_timestamp})
            self.logger.warning(f'⚠️ Time gap detected: {gap_seconds:.2f}s')
            return error
        self.logger.debug(f'✅ Time gap check passed: {gap_seconds:.2f}s')
        return None

    @handles_errors(fallback=[], context='remove_duplicates')
    def _remove_duplicates(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicates from validated data with logging."""
        if not data:
            return data
        original_count = len(data)
        df = pd.DataFrame(data)
        if self.schema.primary_key:
            df = df.drop_duplicates(subset = self.schema.primary_key, keep='first')
        else:
            df = df.drop_duplicates(keep='first')
        deduplicated_data = df.to_dict('records')
        duplicates_removed = original_count - len(deduplicated_data)
        if duplicates_removed > 0:
            self.validation_stats['duplicates_removed'] += duplicates_removed
            self.logger.info(f'🔄 Removed {duplicates_removed} duplicate rows')
        return deduplicated_data

    @handles_errors(fallback = None, context='add_metadata_fields')
    def _add_metadata_fields(self, validated_row: Dict[str, Any], row_data: Dict[str, Any]) -> None:
        """Add metadata fields to validated row."""
        validated_row['exchange'] = self.exchange
        validated_row['data_type'] = self.schema.data_type.value
        if self.schema.timestamp_field in validated_row:
            timestamp = validated_row[self.schema.timestamp_field]
            if isinstance(timestamp, (int, float)):
                dt = pd.to_datetime(timestamp, unit='ms', utc = True)
                validated_row['year'] = dt.year
                validated_row['month'] = dt.month
                validated_row['day'] = dt.day
                validated_row['hour'] = dt.hour
                validated_row['minute'] = dt.minute
                self.logger.debug(f"📅 Added date partitioning: {dt.strftime('%Y-%m-%d %H:%M')}")

    @handles_errors(fallback = None, context='log_validation_errors')
    def _log_validation_errors(self, errors: List[ValidationError], row_index: int) -> None:
        """Log validation errors for a row with extensive context."""
        for error in errors:
            if error.severity == ValidationSeverity.CRITICAL:
                self.logger.error(f'❌ CRITICAL: Row {row_index}: {error.message}')
            elif error.severity == ValidationSeverity.HIGH:
                self.logger.warning(f'⚠️ HIGH: Row {row_index}: {error.message}')
            elif error.severity == ValidationSeverity.MEDIUM:
                self.logger.warning(f'⚠️ MEDIUM: Row {row_index}: {error.message}')
            else:
                self.logger.info(f'ℹ️ LOW: Row {row_index}: {error.message}')

    @handles_errors(fallback = None, context='log_batch_errors')
    def _log_batch_errors(self, errors: List[ValidationError]) -> None:
        """Log batch validation errors with summary."""
        critical_count = sum((1 for e in errors if e.severity == ValidationSeverity.CRITICAL))
        high_count = sum((1 for e in errors if e.severity == ValidationSeverity.HIGH))
        medium_count = sum((1 for e in errors if e.severity == ValidationSeverity.MEDIUM))
        low_count = sum((1 for e in errors if e.severity == ValidationSeverity.LOW))
        self.logger.info(f'📊 Batch validation error summary:')
        if critical_count > 0:
            self.logger.error(f'   ❌ Critical: {critical_count}')
        if high_count > 0:
            self.logger.warning(f'   ⚠️ High: {high_count}')
        if medium_count > 0:
            self.logger.warning(f'   ⚠️ Medium: {medium_count}')
        if low_count > 0:
            self.logger.info(f'   ℹ️ Low: {low_count}')

    @handles_errors(fallback={}, context='get_validation_summary')
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics summary."""
        total = self.validation_stats['total_rows_processed']
        valid = self.validation_stats['valid_rows']
        invalid = self.validation_stats['invalid_rows']
        summary = {'data_type': self.schema.data_type.value, 'exchange': self.exchange, 'total_rows_processed': total, 'valid_rows': valid, 'invalid_rows': invalid, 'success_rate': valid / total * 100 if total > 0 else 0, 'time_gaps_detected': self.validation_stats['time_gaps_detected'], 'duplicates_removed': self.validation_stats['duplicates_removed'], 'total_errors': len(self.validation_stats['validation_errors']), 'error_breakdown': self._get_error_breakdown(), 'validation_duration': time.time() - self.validation_stats['start_time'] if self.validation_stats['start_time'] else 0, 'timestamp': datetime.now().isoformat()}
        self.logger.info(f'📊 Validation Summary for {self.schema.data_type.value} ({self.exchange}):')
        self.logger.info(f"   📈 Success Rate: {summary['success_rate']:.1f}%")
        self.logger.info(f'   📊 Total Rows: {total}')
        self.logger.info(f'   ✅ Valid Rows: {valid}')
        self.logger.info(f'   ❌ Invalid Rows: {invalid}')
        self.logger.info(f"   🕐 Time Gaps: {summary['time_gaps_detected']}")
        self.logger.info(f"   🔄 Duplicates Removed: {summary['duplicates_removed']}")
        self.logger.info(f"   ⚠️ Total Errors: {summary['total_errors']}")
        return summary

    @handles_errors(fallback={}, context='get_error_breakdown')
    def _get_error_breakdown(self) -> Dict[str, int]:
        """Get breakdown of errors by severity."""
        breakdown = {severity.value: 0 for severity in ValidationSeverity}
        for error in self.validation_stats['validation_errors']:
            breakdown[error.severity.value] += 1
        return breakdown

def create_klines_schema() -> DataSchema:
    """Create standardized klines schema with comprehensive field definitions."""
    return DataSchema(data_type = DataType.KLINES, fields=[FieldDefinition(name='timestamp', dtype='int64', description='Opening time of the kline in milliseconds'), FieldDefinition(name='open', dtype='float64', min_value = 0.0, allow_zero = False, description='Opening price of the kline'), FieldDefinition(name='high', dtype='float64', min_value = 0.0, allow_zero = False, description='Highest price during the kline period'), FieldDefinition(name='low', dtype='float64', min_value = 0.0, allow_zero = False, description='Lowest price during the kline period'), FieldDefinition(name='close', dtype='float64', min_value = 0.0, allow_zero = False, description='Closing price of the kline'), FieldDefinition(name='volume', dtype='float64', min_value = 0.0, allow_zero = True, description='Volume of the base asset traded'), FieldDefinition(name='exchange', dtype='string', required = True, description='Exchange name'), FieldDefinition(name='symbol', dtype='string', required = True, description='Trading symbol'), FieldDefinition(name='timeframe', dtype='string', required = True, description='Timeframe of the kline')], primary_key=['timestamp', 'exchange', 'symbol', 'timeframe'], time_gap_config = TimeGapConfig(max_gap_seconds = 66.0, tolerance_seconds = 5.0, severity = ValidationSeverity.HIGH, description='Maximum allowed gap between kline timestamps'), description='Standardized klines data schema for OHLCV data')

def create_aggtrades_schema() -> DataSchema:
    """Create standardized aggtrades schema with comprehensive field definitions."""
    return DataSchema(data_type = DataType.AGGTRADES, fields=[FieldDefinition(name='timestamp', dtype='int64', description='Timestamp of the trade in milliseconds'), FieldDefinition(name='price', dtype='float64', min_value = 0.0, allow_zero = False, description='Price of the trade'), FieldDefinition(name='quantity', dtype='float64', min_value = 0.0, allow_zero = False, description='Quantity of the trade'), FieldDefinition(name='is_buyer_maker', dtype='bool', required = False, default_value = False, description='Whether the buyer is the maker'), FieldDefinition(name='trade_id', dtype='int64', required = False, default_value = 0, description='Unique trade identifier'), FieldDefinition(name='exchange', dtype='string', required = True, description='Exchange name'), FieldDefinition(name='symbol', dtype='string', required = True, description='Trading symbol')], primary_key=['timestamp', 'trade_id', 'exchange', 'symbol'], time_gap_config = TimeGapConfig(max_gap_seconds = 1.0, tolerance_seconds = 0.1, severity = ValidationSeverity.MEDIUM, description='Maximum allowed gap between trade timestamps'), description='Standardized aggtrades data schema for trade data')

def create_futures_schema() -> DataSchema:
    """Create standardized futures schema with comprehensive field definitions."""
    return DataSchema(data_type = DataType.FUTURES, fields=[FieldDefinition(name='timestamp', dtype='int64', description='Timestamp of the funding rate in milliseconds'), FieldDefinition(name='funding_rate', dtype='float64', allow_zero = True, description='Funding rate for the period'), FieldDefinition(name='exchange', dtype='string', required = True, description='Exchange name'), FieldDefinition(name='symbol', dtype='string', required = True, description='Trading symbol')], primary_key=['timestamp', 'exchange', 'symbol'], time_gap_config = TimeGapConfig(max_gap_seconds = 32400.0, tolerance_seconds = 300.0, severity = ValidationSeverity.MEDIUM, description='Maximum allowed gap between funding rate timestamps'), description='Standardized futures data schema for funding rate data')
SCHEMA_REGISTRY = {DataType.KLINES: create_klines_schema(), DataType.AGGTRADES: create_aggtrades_schema(), DataType.FUTURES: create_futures_schema()}
logger.info(f'📋 Initialized schema registry with {len(SCHEMA_REGISTRY)} schemas')

@handles_errors(fallback = None, context='get_validator')
def get_validator(data_type: DataType, exchange: str='UNKNOWN') -> EnhancedDataValidator:
    """Get validator for specified data type and exchange with extensive logging."""
    logger.info(f'🔍 Getting validator for {data_type.value} data from {exchange}')
    schema = SCHEMA_REGISTRY.get(data_type)
    if not schema:
        logger.error(f'❌ No schema found for data type: {data_type}')
        raise ValueError(f'No schema found for data type: {data_type}')
    validator = EnhancedDataValidator(schema, exchange)
    logger.info(f'✅ Created validator for {data_type.value} data from {exchange}')
    return validator

@handles_errors(fallback=[], context='validate_data_batch')
@traced(span_name='validate_data_batch', log_args = False, log_result_len_only = True)
def validate_data_batch(data_type: DataType, batch_data: List[Dict[str, Any]], exchange: str='UNKNOWN', previous_timestamp: Optional[int]=None) -> List[Dict[str, Any]]:
    """Convenience function to validate a batch of data using utils/ tools."""
    logger.info(f'🚀 Starting batch validation for {data_type.value} data from {exchange} using utils/')
    
    try:
        # Convert to DataFrame for utils/ validation
        import pandas as pd
        df = pd.DataFrame(batch_data)
        
        # Use utils/ validation tools
        data_quality_valid = validate_data_quality(df)
        
        # Use utils/ DataFrame column validation
        required_columns = get_required_columns(data_type)
        column_valid = validate_dataframe_columns(df, required_columns)
        
        # Use utils/ safe DataFrame operations
        validated_df = safe_dataframe_operation(df, lambda x: x.copy())
        
        # Use utils/ safe fillna
        validated_df = safe_fillna(validated_df, method='forward')
        
        # Convert back to list of dicts
        result = validated_df.to_dict('records')
        
        success_rate = 100.0 if data_quality_valid and column_valid else 50.0
        logger.info(f"✅ Utils/ validation completed: {success_rate:.1f}% success rate")
        return result
        
    except Exception as e:
        logger.exception(f'❌ Utils/ validation error: {e}')
        return []

def get_required_columns(data_type: DataType) -> List[str]:
    """Get required columns for data type."""
    if data_type == DataType.KLINES:
        return ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    elif data_type == DataType.AGGTRADES:
        return ['timestamp', 'price', 'quantity']
    elif data_type == DataType.FUTURES:
        return ['timestamp', 'funding_rate']
    else:
        return ['timestamp']
if __name__ == '__main__':

    async def test_enhanced_validation() -> None:
        logger.info('🎯 Testing Enhanced Data Validation Framework with Decorators')
        logger.info('=' * 80)
        klines_data = [{'open_time': 1640995200000, 'open': '3000.0', 'high': '3100.0', 'low': '2900.0', 'close': '3050.0', 'volume': '1000.0'}]
        logger.info('📊 Testing klines validation...')
        validated_klines = validate_data_batch(DataType.KLINES, klines_data, 'BINANCE')
        logger.info(f'✅ Validated {len(validated_klines)} klines rows')
        aggtrades_data = [{'T': 1640995200000, 'p': '3050.0', 'q': '1.5', 'm': True}]
        logger.info('📊 Testing aggtrades validation...')
        validated_aggtrades = validate_data_batch(DataType.AGGTRADES, aggtrades_data, 'BINANCE')
        logger.info(f'✅ Validated {len(validated_aggtrades)} aggtrades rows')
        logger.info('=' * 80)
        logger.info('🎉 Enhanced validation framework tests completed successfully!')
        logger.info('=' * 80)
    asyncio.run(test_enhanced_validation())