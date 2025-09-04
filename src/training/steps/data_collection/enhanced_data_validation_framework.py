#!/usr/bin/env python3
"""
Enhanced Data Validation Framework for Data Collection

This module provides comprehensive validation during data collection for:
- Klines data
- Aggtrades data  
- Futures data

Features:
- Schema enforcement with field mapping
- Time gap detection between batches
- Data quality checks (NaN, infinite, zero values)
- Format validation (string, size, data types)
- Real-time validation during API collection
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger
from src.utils.pipeline_standards import pipeline_standards

logger = system_logger.getChild("EnhancedDataValidation")


class DataType(Enum):
    """Supported data types for validation."""
    KLINES = "klines"
    AGGTRADES = "aggtrades"
    FUTURES = "futures"
    UNIFIED = "unified"


class ValidationSeverity(Enum):
    """Validation error severity levels."""
    CRITICAL = "critical"  # Stop processing
    HIGH = "high"         # Log error, continue with warning
    MEDIUM = "medium"     # Log warning, continue
    LOW = "low"          # Log info, continue


@dataclass
class ValidationError:
    """Represents a validation error."""
    field: str
    message: str
    severity: ValidationSeverity
    value: Any = None
    expected: Any = None
    row_index: Optional[int] = None


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
    source_mapping: Dict[str, str] = field(default_factory=dict)  # Exchange-specific field names
    custom_validator: Optional[Callable] = None


@dataclass
class TimeGapConfig:
    """Configuration for time gap detection."""
    max_gap_seconds: float
    tolerance_seconds: float = 0.0
    severity: ValidationSeverity = ValidationSeverity.HIGH


@dataclass
class DataSchema:
    """Complete data schema definition."""
    data_type: DataType
    fields: List[FieldDefinition]
    primary_key: List[str]
    timestamp_field: str = "timestamp"
    time_gap_config: Optional[TimeGapConfig] = None
    required_columns: List[str] = field(default_factory=list)
    optional_columns: List[str] = field(default_factory=list)


class EnhancedDataValidator:
    """Enhanced data validator with comprehensive validation rules."""
    
    def __init__(self, schema: DataSchema):
        self.schema = schema
        self.logger = logger.getChild(f"Validator.{schema.data_type.value}")
        self.validation_stats = {
            'total_rows_processed': 0,
            'valid_rows': 0,
            'invalid_rows': 0,
            'validation_errors': [],
            'time_gaps_detected': 0,
            'quality_issues': []
        }
    
    def validate_row(self, row_data: Dict[str, Any], row_index: int = 0) -> Dict[str, Any]:
        """
        Validate and standardize a single row of data.
        
        Args:
            row_data: Raw row data from API
            row_index: Index of the row for error reporting
            
        Returns:
            Validated and standardized row data
            
        Raises:
            ValueError: If critical validation errors are found
        """
        validated_row = {}
        errors = []
        
        # Process each field according to schema
        for field_def in self.schema.fields:
            try:
                # Get source value using field mapping
                source_value = self._get_source_value(row_data, field_def)
                
                # Validate and convert value
                validated_value = self._validate_and_convert_field(
                    source_value, field_def, row_index
                )
                
                validated_row[field_def.name] = validated_value
                
            except ValidationError as e:
                errors.append(e)
                if e.severity == ValidationSeverity.CRITICAL:
                    raise ValueError(f"Critical validation error in row {row_index}: {e.message}")
        
        # Add metadata fields if not present
        self._add_metadata_fields(validated_row, row_data)
        
        # Log validation results
        if errors:
            self._log_validation_errors(errors, row_index)
        
        return validated_row
    
    def validate_batch(self, batch_data: List[Dict[str, Any]], 
                      previous_timestamp: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Validate a batch of data with time gap detection.
        
        Args:
            batch_data: List of raw row data from API
            previous_timestamp: Timestamp of last row from previous batch
            
        Returns:
            List of validated and standardized row data
        """
        validated_batch = []
        batch_errors = []
        
        self.logger.info(f"🔍 Validating batch of {len(batch_data)} rows")
        
        for i, row_data in enumerate(batch_data):
            try:
                validated_row = self.validate_row(row_data, i)
                validated_batch.append(validated_row)
                
                # Check time gap if configured and we have previous timestamp
                if (self.schema.time_gap_config and 
                    previous_timestamp is not None and 
                    i == 0):  # Only check first row of batch
                    
                    gap_error = self._check_time_gap(
                        validated_row[self.schema.timestamp_field], 
                        previous_timestamp
                    )
                    if gap_error:
                        batch_errors.append(gap_error)
                
            except ValueError as e:
                self.logger.error(f"❌ Row {i} validation failed: {e}")
                batch_errors.append(ValidationError(
                    field="row",
                    message=str(e),
                    severity=ValidationSeverity.CRITICAL,
                    row_index=i
                ))
        
        # Update validation statistics
        self.validation_stats['total_rows_processed'] += len(batch_data)
        self.validation_stats['valid_rows'] += len(validated_batch)
        self.validation_stats['invalid_rows'] += len(batch_data) - len(validated_batch)
        
        # Log batch validation results
        success_rate = len(validated_batch) / len(batch_data) * 100 if batch_data else 0
        self.logger.info(f"✅ Batch validation: {len(validated_batch)}/{len(batch_data)} rows valid ({success_rate:.1f}%)")
        
        if batch_errors:
            self.validation_stats['validation_errors'].extend(batch_errors)
            self._log_batch_errors(batch_errors)
        
        return validated_batch
    
    def _get_source_value(self, row_data: Dict[str, Any], field_def: FieldDefinition) -> Any:
        """Get source value using field mapping."""
        # Try direct field name first
        if field_def.name in row_data:
            return row_data[field_def.name]
        
        # Try exchange-specific mappings
        for exchange, source_field in field_def.source_mapping.items():
            if source_field in row_data:
                return row_data[source_field]
        
        # Return None if not found
        return None
    
    def _validate_and_convert_field(self, value: Any, field_def: FieldDefinition, 
                                   row_index: int) -> Any:
        """Validate and convert a field value."""
        # Handle missing values
        if value is None or (isinstance(value, float) and np.isnan(value)):
            if field_def.required:
                raise ValidationError(
                    field=field_def.name,
                    message=f"Required field '{field_def.name}' is missing or NaN",
                    severity=ValidationSeverity.CRITICAL,
                    row_index=row_index
                )
            return field_def.default_value
        
        # Type conversion
        try:
            converted_value = self._convert_type(value, field_def.dtype)
        except (ValueError, TypeError) as e:
            raise ValidationError(
                field=field_def.name,
                message=f"Type conversion failed: {e}",
                severity=ValidationSeverity.CRITICAL,
                value=value,
                expected=field_def.dtype,
                row_index=row_index
            )
        
        # Value validation
        validation_errors = self._validate_value(converted_value, field_def, row_index)
        if validation_errors:
            # Use the most severe error
            critical_errors = [e for e in validation_errors if e.severity == ValidationSeverity.CRITICAL]
            if critical_errors:
                raise critical_errors[0]
            # Log non-critical errors
            for error in validation_errors:
                self.logger.warning(f"⚠️ {error.message}")
        
        return converted_value
    
    def _convert_type(self, value: Any, target_type: str) -> Any:
        """Convert value to target type."""
        if target_type == "int64":
            return int(float(value))
        elif target_type == "float64":
            return float(value)
        elif target_type == "string":
            return str(value)
        elif target_type == "bool":
            return bool(value)
        else:
            return value
    
    def _validate_value(self, value: Any, field_def: FieldDefinition, 
                       row_index: int) -> List[ValidationError]:
        """Validate field value against constraints."""
        errors = []
        
        # Check for NaN
        if isinstance(value, float) and np.isnan(value):
            if not field_def.allow_nan:
                errors.append(ValidationError(
                    field=field_def.name,
                    message=f"Field '{field_def.name}' contains NaN value",
                    severity=ValidationSeverity.HIGH,
                    value=value,
                    row_index=row_index
                ))
        
        # Check for infinite values
        if isinstance(value, float) and np.isinf(value):
            if not field_def.allow_infinite:
                errors.append(ValidationError(
                    field=field_def.name,
                    message=f"Field '{field_def.name}' contains infinite value",
                    severity=ValidationSeverity.HIGH,
                    value=value,
                    row_index=row_index
                ))
        
        # Check for zero values
        if isinstance(value, (int, float)) and value == 0:
            if not field_def.allow_zero:
                errors.append(ValidationError(
                    field=field_def.name,
                    message=f"Field '{field_def.name}' contains zero value",
                    severity=ValidationSeverity.MEDIUM,
                    value=value,
                    row_index=row_index
                ))
        
        # Check for negative values
        if isinstance(value, (int, float)) and value < 0:
            if not field_def.allow_negative:
                errors.append(ValidationError(
                    field=field_def.name,
                    message=f"Field '{field_def.name}' contains negative value",
                    severity=ValidationSeverity.HIGH,
                    value=value,
                    row_index=row_index
                ))
        
        # Check min/max constraints
        if isinstance(value, (int, float)):
            if field_def.min_value is not None and value < field_def.min_value:
                errors.append(ValidationError(
                    field=field_def.name,
                    message=f"Field '{field_def.name}' value {value} below minimum {field_def.min_value}",
                    severity=ValidationSeverity.HIGH,
                    value=value,
                    expected=field_def.min_value,
                    row_index=row_index
                ))
            
            if field_def.max_value is not None and value > field_def.max_value:
                errors.append(ValidationError(
                    field=field_def.name,
                    message=f"Field '{field_def.name}' value {value} above maximum {field_def.max_value}",
                    severity=ValidationSeverity.HIGH,
                    value=value,
                    expected=field_def.max_value,
                    row_index=row_index
                ))
        
        # Custom validation
        if field_def.custom_validator:
            try:
                if not field_def.custom_validator(value):
                    errors.append(ValidationError(
                        field=field_def.name,
                        message=f"Custom validation failed for field '{field_def.name}'",
                        severity=ValidationSeverity.MEDIUM,
                        value=value,
                        row_index=row_index
                    ))
            except Exception as e:
                errors.append(ValidationError(
                    field=field_def.name,
                    message=f"Custom validation error for field '{field_def.name}': {e}",
                    severity=ValidationSeverity.MEDIUM,
                    value=value,
                    row_index=row_index
                ))
        
        return errors
    
    def _check_time_gap(self, current_timestamp: int, previous_timestamp: int) -> Optional[ValidationError]:
        """Check for time gap between batches."""
        if not self.schema.time_gap_config:
            return None
        
        gap_seconds = (current_timestamp - previous_timestamp) / 1000.0  # Convert ms to seconds
        max_gap = self.schema.time_gap_config.max_gap_seconds
        tolerance = self.schema.time_gap_config.tolerance_seconds
        
        if gap_seconds > max_gap + tolerance:
            self.validation_stats['time_gaps_detected'] += 1
            return ValidationError(
                field=self.schema.timestamp_field,
                message=f"Time gap detected: {gap_seconds:.2f}s > {max_gap}s (tolerance: {tolerance}s)",
                severity=self.schema.time_gap_config.severity,
                value=current_timestamp,
                expected=previous_timestamp + int(max_gap * 1000)
            )
        
        return None
    
    def _add_metadata_fields(self, validated_row: Dict[str, Any], row_data: Dict[str, Any]):
        """Add metadata fields to validated row."""
        # Add date partitioning fields if timestamp exists
        if self.schema.timestamp_field in validated_row:
            timestamp = validated_row[self.schema.timestamp_field]
            if isinstance(timestamp, (int, float)):
                dt = pd.to_datetime(timestamp, unit='ms', utc=True)
                validated_row['year'] = dt.year
                validated_row['month'] = dt.month
                validated_row['day'] = dt.day
    
    def _log_validation_errors(self, errors: List[ValidationError], row_index: int):
        """Log validation errors for a row."""
        for error in errors:
            if error.severity == ValidationSeverity.CRITICAL:
                self.logger.error(f"❌ Row {row_index}: {error.message}")
            elif error.severity == ValidationSeverity.HIGH:
                self.logger.warning(f"⚠️ Row {row_index}: {error.message}")
            else:
                self.logger.info(f"ℹ️ Row {row_index}: {error.message}")
    
    def _log_batch_errors(self, errors: List[ValidationError]):
        """Log batch validation errors."""
        critical_count = sum(1 for e in errors if e.severity == ValidationSeverity.CRITICAL)
        high_count = sum(1 for e in errors if e.severity == ValidationSeverity.HIGH)
        medium_count = sum(1 for e in errors if e.severity == ValidationSeverity.MEDIUM)
        
        if critical_count > 0:
            self.logger.error(f"❌ Batch validation: {critical_count} critical errors")
        if high_count > 0:
            self.logger.warning(f"⚠️ Batch validation: {high_count} high severity errors")
        if medium_count > 0:
            self.logger.info(f"ℹ️ Batch validation: {medium_count} medium severity errors")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation statistics summary."""
        total = self.validation_stats['total_rows_processed']
        valid = self.validation_stats['valid_rows']
        invalid = self.validation_stats['invalid_rows']
        
        return {
            'data_type': self.schema.data_type.value,
            'total_rows_processed': total,
            'valid_rows': valid,
            'invalid_rows': invalid,
            'success_rate': valid / total * 100 if total > 0 else 0,
            'time_gaps_detected': self.validation_stats['time_gaps_detected'],
            'total_errors': len(self.validation_stats['validation_errors']),
            'error_breakdown': self._get_error_breakdown()
        }
    
    def _get_error_breakdown(self) -> Dict[str, int]:
        """Get breakdown of errors by severity."""
        breakdown = {severity.value: 0 for severity in ValidationSeverity}
        for error in self.validation_stats['validation_errors']:
            breakdown[error.severity.value] += 1
        return breakdown


# Standardized Schema Definitions
def create_klines_schema() -> DataSchema:
    """Create standardized klines schema."""
    return DataSchema(
        data_type=DataType.KLINES,
        fields=[
            FieldDefinition(
                name="timestamp",
                dtype="int64",
                source_mapping={
                    "binance": "open_time",
                    "coinbase": "timestamp",
                    "kraken": "time"
                }
            ),
            FieldDefinition(
                name="open",
                dtype="float64",
                min_value=0.0,
                allow_zero=False,
                source_mapping={
                    "binance": "open",
                    "coinbase": "price_open",
                    "kraken": "open"
                }
            ),
            FieldDefinition(
                name="high",
                dtype="float64",
                min_value=0.0,
                allow_zero=False,
                source_mapping={
                    "binance": "high",
                    "coinbase": "price_high",
                    "kraken": "high"
                }
            ),
            FieldDefinition(
                name="low",
                dtype="float64",
                min_value=0.0,
                allow_zero=False,
                source_mapping={
                    "binance": "low",
                    "coinbase": "price_low",
                    "kraken": "low"
                }
            ),
            FieldDefinition(
                name="close",
                dtype="float64",
                min_value=0.0,
                allow_zero=False,
                source_mapping={
                    "binance": "close",
                    "coinbase": "price_close",
                    "kraken": "close"
                }
            ),
            FieldDefinition(
                name="volume",
                dtype="float64",
                min_value=0.0,
                allow_zero=True,
                source_mapping={
                    "binance": "volume",
                    "coinbase": "volume",
                    "kraken": "vol"
                }
            ),
            FieldDefinition(
                name="exchange",
                dtype="string",
                required=True
            ),
            FieldDefinition(
                name="symbol",
                dtype="string",
                required=True
            ),
            FieldDefinition(
                name="timeframe",
                dtype="string",
                required=True
            )
        ],
        primary_key=["timestamp", "exchange", "symbol", "timeframe"],
        time_gap_config=TimeGapConfig(
            max_gap_seconds=66.0,  # 1.1 minutes for klines
            tolerance_seconds=5.0,
            severity=ValidationSeverity.HIGH
        )
    )


def create_aggtrades_schema() -> DataSchema:
    """Create standardized aggtrades schema."""
    return DataSchema(
        data_type=DataType.AGGTRADES,
        fields=[
            FieldDefinition(
                name="timestamp",
                dtype="int64",
                source_mapping={
                    "binance": "T",
                    "coinbase": "timestamp",
                    "kraken": "time"
                }
            ),
            FieldDefinition(
                name="price",
                dtype="float64",
                min_value=0.0,
                allow_zero=False,
                source_mapping={
                    "binance": "p",
                    "coinbase": "price",
                    "kraken": "price"
                }
            ),
            FieldDefinition(
                name="quantity",
                dtype="float64",
                min_value=0.0,
                allow_zero=False,
                source_mapping={
                    "binance": "q",
                    "coinbase": "size",
                    "kraken": "vol"
                }
            ),
            FieldDefinition(
                name="is_buyer_maker",
                dtype="bool",
                required=False,
                default_value=False,
                source_mapping={
                    "binance": "m",
                    "coinbase": "side",
                    "kraken": "type"
                }
            ),
            FieldDefinition(
                name="trade_id",
                dtype="int64",
                required=False,
                default_value=0,
                source_mapping={
                    "binance": "a",
                    "coinbase": "trade_id",
                    "kraken": "id"
                }
            ),
            FieldDefinition(
                name="exchange",
                dtype="string",
                required=True
            ),
            FieldDefinition(
                name="symbol",
                dtype="string",
                required=True
            )
        ],
        primary_key=["timestamp", "trade_id", "exchange", "symbol"],
        time_gap_config=TimeGapConfig(
            max_gap_seconds=1.0,  # 1 second for aggtrades
            tolerance_seconds=0.1,
            severity=ValidationSeverity.MEDIUM
        )
    )


def create_futures_schema() -> DataSchema:
    """Create standardized futures schema."""
    return DataSchema(
        data_type=DataType.FUTURES,
        fields=[
            FieldDefinition(
                name="timestamp",
                dtype="int64",
                source_mapping={
                    "binance": "fundingTime",
                    "coinbase": "timestamp",
                    "kraken": "time"
                }
            ),
            FieldDefinition(
                name="funding_rate",
                dtype="float64",
                allow_zero=True,
                source_mapping={
                    "binance": "fundingRate",
                    "coinbase": "funding_rate",
                    "kraken": "funding_rate"
                }
            ),
            FieldDefinition(
                name="exchange",
                dtype="string",
                required=True
            ),
            FieldDefinition(
                name="symbol",
                dtype="string",
                required=True
            )
        ],
        primary_key=["timestamp", "exchange", "symbol"],
        time_gap_config=TimeGapConfig(
            max_gap_seconds=32400.0,  # 9 hours for futures
            tolerance_seconds=300.0,  # 5 minutes tolerance
            severity=ValidationSeverity.MEDIUM
        )
    )


def create_unified_schema() -> DataSchema:
    """Create standardized unified schema."""
    klines_schema = create_klines_schema()
    aggtrades_schema = create_aggtrades_schema()
    futures_schema = create_futures_schema()
    
    # Combine all fields
    all_fields = []
    all_fields.extend(klines_schema.fields)
    
    # Add aggtrades-specific fields
    for field in aggtrades_schema.fields:
        if field.name not in [f.name for f in all_fields]:
            field.required = False  # Make optional in unified schema
            all_fields.append(field)
    
    # Add futures-specific fields
    for field in futures_schema.fields:
        if field.name not in [f.name for f in all_fields]:
            field.required = False  # Make optional in unified schema
            all_fields.append(field)
    
    # Add unified-specific fields
    all_fields.extend([
        FieldDefinition(
            name="trade_volume",
            dtype="float64",
            required=False,
            default_value=0.0,
            min_value=0.0,
            allow_zero=True
        ),
        FieldDefinition(
            name="trade_count",
            dtype="int64",
            required=False,
            default_value=0,
            min_value=0,
            allow_zero=True
        ),
        FieldDefinition(
            name="avg_price",
            dtype="float64",
            required=False,
            default_value=0.0,
            min_value=0.0,
            allow_zero=True
        ),
        FieldDefinition(
            name="min_price",
            dtype="float64",
            required=False,
            default_value=0.0,
            min_value=0.0,
            allow_zero=True
        ),
        FieldDefinition(
            name="max_price",
            dtype="float64",
            required=False,
            default_value=0.0,
            min_value=0.0,
            allow_zero=True
        ),
        FieldDefinition(
            name="volume_ratio",
            dtype="float64",
            required=False,
            default_value=0.0,
            allow_zero=True
        )
    ])
    
    return DataSchema(
        data_type=DataType.UNIFIED,
        fields=all_fields,
        primary_key=["timestamp", "exchange", "symbol", "timeframe"],
        time_gap_config=TimeGapConfig(
            max_gap_seconds=66.0,  # Same as klines
            tolerance_seconds=5.0,
            severity=ValidationSeverity.HIGH
        )
    )


# Schema registry
SCHEMA_REGISTRY = {
    DataType.KLINES: create_klines_schema(),
    DataType.AGGTRADES: create_aggtrades_schema(),
    DataType.FUTURES: create_futures_schema(),
    DataType.UNIFIED: create_unified_schema()
}


def get_validator(data_type: DataType) -> EnhancedDataValidator:
    """Get validator for specified data type."""
    schema = SCHEMA_REGISTRY.get(data_type)
    if not schema:
        raise ValueError(f"No schema found for data type: {data_type}")
    return EnhancedDataValidator(schema)


def validate_data_batch(data_type: DataType, batch_data: List[Dict[str, Any]], 
                       previous_timestamp: Optional[int] = None) -> List[Dict[str, Any]]:
    """Convenience function to validate a batch of data."""
    validator = get_validator(data_type)
    return validator.validate_batch(batch_data, previous_timestamp)


if __name__ == "__main__":
    # Example usage
    async def test_validation():
        # Test klines validation
        klines_data = [
            {
                "open_time": 1640995200000,  # Binance format
                "open": "3000.0",
                "high": "3100.0",
                "low": "2900.0",
                "close": "3050.0",
                "volume": "1000.0"
            }
        ]
        
        validator = get_validator(DataType.KLINES)
        validated = validator.validate_batch(klines_data)
        print(f"Validated {len(validated)} klines rows")
        
        # Test aggtrades validation
        aggtrades_data = [
            {
                "T": 1640995200000,  # Binance format
                "p": "3050.0",
                "q": "1.5",
                "m": True
            }
        ]
        
        validator = get_validator(DataType.AGGTRADES)
        validated = validator.validate_batch(aggtrades_data)
        print(f"Validated {len(validated)} aggtrades rows")
        
        # Print validation summary
        print("Validation Summary:", validator.get_validation_summary())
    
    asyncio.run(test_validation())