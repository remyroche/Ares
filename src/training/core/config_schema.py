"""
Configuration Schema Validation System

This module provides comprehensive configuration validation:
1. Schema-based validation with detailed error messages
2. Type checking and conversion
3. Default value handling
4. Configuration inheritance and merging
5. Runtime validation

Usage:
    from src.training.core.config_schema import (
        ConfigSchema, ConfigValidator,
        validate_pipeline_config
    )

    # Define schema
    schema = ConfigSchema({
        'symbol': {'type': 'str', 'required': True, 'choices': ['BTCUSDT', 'ETHUSDT']},
        'exchange': {'type': 'str', 'default': 'binance'},
        'timeframe': {'type': 'str', 'default': '1m', 'pattern': r'^\d+[mhd]$'}
    })

    # Validate configuration
    validator = ConfigValidator(schema)
    try:
        validated_config = validator.validate(raw_config)
        print("Configuration is valid!")
    except ConfigurationError as e:
        print(f"Configuration error: {e}")
"""

import re
import json
from typing import Dict, Any, Optional, List, Union, Type, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod

from src.training.core.errors import ConfigurationError, ErrorContext

class ValidationRule(Enum):
    """Validation rule types."""
    REQUIRED = "required"
    TYPE = "type"
    DEFAULT = "default"
    CHOICES = "choices"
    RANGE = "range"
    PATTERN = "pattern"
    CUSTOM = "custom"
    LENGTH = "length"
    NESTED = "nested"

@dataclass
class FieldSchema:
    """Schema definition for a configuration field."""
    type: str
    required: bool = False
    default: Any = None
    choices: List[Any] = field(default_factory=list)
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    length_min: Optional[int] = None
    length_max: Optional[int] = None
    description: Optional[str] = None
    nested_schema: Optional['ConfigSchema'] = None
    validator: Optional[Callable] = None

    def validate(self, value: Any, field_name: str) -> Any:
        """Validate a single field value."""
        # Check required fields
        if self.required and value is None:
            raise ConfigurationError(
                f"Required field '{field_name}' is missing",
                context=ErrorContext(
                    operation="configuration_validation",
                    custom_data={'field': field_name}
                )
            )

        # Use default if None and not required
        if value is None:
            if self.default is not None:
                value = self.default
            else:
                return None  # Field is optional and not provided

        # Type validation and conversion
        try:
            value = self._convert_type(value)
        except (ValueError, TypeError) as e:
            raise ConfigurationError(
                f"Invalid type for field '{field_name}': {e}",
                context=ErrorContext(
                    operation="type_conversion",
                    custom_data={'field': field_name, 'expected_type': self.type}
                )
            )

        # Choices validation
        if self.choices and value not in self.choices:
            raise ConfigurationError(
                f"Invalid choice for field '{field_name}': {value}. Valid choices: {self.choices}",
                context=ErrorContext(
                    operation="choices_validation",
                    custom_data={'field': field_name, 'choices': self.choices}
                )
            )

        # Range validation
        if self.min_value is not None and value < self.min_value:
            raise ConfigurationError(
                f"Value for field '{field_name}' is below minimum: {value} < {self.min_value}",
                context=ErrorContext(
                    operation="range_validation",
                    custom_data={'field': field_name, 'min_value': self.min_value}
                )
            )

        if self.max_value is not None and value > self.max_value:
            raise ConfigurationError(
                f"Value for field '{field_name}' is above maximum: {value} > {self.max_value}",
                context=ErrorContext(
                    operation="range_validation",
                    custom_data={'field': field_name, 'max_value': self.max_value}
                )
            )

        # Pattern validation
        if self.pattern and not re.match(self.pattern, str(value)):
            raise ConfigurationError(
                f"Value for field '{field_name}' does not match pattern: {self.pattern}",
                context=ErrorContext(
                    operation="pattern_validation",
                    custom_data={'field': field_name, 'pattern': self.pattern}
                )
            )

        # Length validation
        if self.length_min is not None and len(str(value)) < self.length_min:
            raise ConfigurationError(
                f"Value for field '{field_name}' is too short: {len(str(value))} < {self.length_min}",
                context=ErrorContext(
                    operation="length_validation",
                    custom_data={'field': field_name, 'min_length': self.length_min}
                )
            )

        if self.length_max is not None and len(str(value)) > self.length_max:
            raise ConfigurationError(
                f"Value for field '{field_name}' is too long: {len(str(value))} > {self.length_max}",
                context=ErrorContext(
                    operation="length_validation",
                    custom_data={'field': field_name, 'max_length': self.length_max}
                )
            )

        # Custom validation
        if self.validator and not self.validator(value):
            raise ConfigurationError(
                f"Custom validation failed for field '{field_name}'",
                context=ErrorContext(
                    operation="custom_validation",
                    custom_data={'field': field_name}
                )
            )

        return value

    def _convert_type(self, value: Any) -> Any:
        """Convert value to the specified type."""
        type_map = {
            'str': str,
            'int': int,
            'float': float,
            'bool': self._convert_bool,
            'list': list,
            'dict': dict,
            'path': Path,
            'json': self._convert_json
        }

        if self.type not in type_map:
            raise ValueError(f"Unsupported type: {self.type}")

        converter = type_map[self.type]
        return converter(value)

    def _convert_bool(self, value: Any) -> bool:
        """Convert value to boolean."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        if isinstance(value, int):
            return value != 0
        raise ValueError(f"Cannot convert {type(value)} to bool")

    def _convert_json(self, value: Any) -> Any:
        """Convert JSON string to Python object."""
        if isinstance(value, str):
            return json.loads(value)
        return value

class ConfigSchema:
    """Configuration schema definition."""

    def __init__(self, schema_dict: Dict[str, Union[FieldSchema, Dict[str, Any]]]):
        """Initialize schema from dictionary."""
        self.fields: Dict[str, FieldSchema] = {}

        for field_name, field_def in schema_dict.items():
            if isinstance(field_def, FieldSchema):
                self.fields[field_name] = field_def
            elif isinstance(field_def, dict):
                self.fields[field_name] = FieldSchema(**field_def)

    def add_field(self, name: str, field_schema: FieldSchema):
        """Add a field to the schema."""
        self.fields[name] = field_schema

    def get_field(self, name: str) -> FieldSchema:
        """Get a field schema."""
        return self.fields.get(name)

    def validate_field(self, name: str, value: Any) -> Any:
        """Validate a single field."""
        if name not in self.fields:
            raise ConfigurationError(
                f"Unknown field: {name}",
                context=ErrorContext(
                    operation="field_validation",
                    custom_data={'field': name}
                )
            )

        return self.fields[name].validate(value, name)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert schema to dictionary representation."""
        return {
            name: {
                'type': field.type,
                'required': field.required,
                'default': field.default,
                'choices': field.choices,
                'min_value': field.min_value,
                'max_value': field.max_value,
                'pattern': field.pattern,
                'length_min': field.length_min,
                'length_max': field.length_max,
                'description': field.description
            }
            for name, field in self.fields.items()
        }

class ConfigValidator:
    """Validates configurations against schemas."""

    def __init__(self, schema: ConfigSchema):
        self.schema = schema
        self.logger = logging.getLogger(__name__)

    def validate(self, config: Dict[str, Any], strict: bool = True) -> Dict[str, Any]:
        """Validate a configuration dictionary."""
        validated_config = {}

        # Validate all provided fields
        for key, value in config.items():
            try:
                validated_config[key] = self.schema.validate_field(key, value)
            except ConfigurationError:
                if strict:
                    raise
                else:
                    self.logger.warning(f"Validation failed for field '{key}': {value}")
                    # Keep original value if validation fails in non-strict mode

        # Check for missing required fields
        for field_name, field_schema in self.schema.fields.items():
            if field_schema.required and field_name not in validated_config:
                raise ConfigurationError(
                    f"Missing required field: {field_name}",
                    context=ErrorContext(
                        operation="required_field_check",
                        custom_data={'field': field_name}
                    )
                )

        # Add default values for missing optional fields
        for field_name, field_schema in self.schema.fields.items():
            if field_name not in validated_config and field_schema.default is not None:
                validated_config[field_name] = field_schema.default

        return validated_config

    def validate_partial(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate only provided fields, ignore missing ones."""
        return self.validate(config, strict=False)

def create_pipeline_config_schema() -> ConfigSchema:
    """Create schema for pipeline configuration."""
    return ConfigSchema({
        'mode': FieldSchema(
            type='str',
            required=True,
            choices=['full', 'light', 'blank'],
            description='Execution mode for the pipeline'
        ),
        'symbol': FieldSchema(
            type='str',
            required=True,
            pattern=r'^[A-Z0-9]{5,20}$',
            description='Trading symbol (e.g., BTCUSDT)'
        ),
        'exchange': FieldSchema(
            type='str',
            default='binance',
            choices=['binance', 'coinbase', 'kraken'],
            description='Exchange name'
        ),
        'timeframe': FieldSchema(
            type='str',
            default='1m',
            pattern=r'^\d+[mhd]$',
            description='Data timeframe (e.g., 1m, 5m, 1h, 1d)'
        ),
        'data_dir': FieldSchema(
            type='path',
            default=Path('./historical_data'),
            description='Directory for data storage'
        ),
        'start_date': FieldSchema(
            type='str',
            pattern=r'^\d{4}-\d{2}-\d{2}$',
            description='Start date in YYYY-MM-DD format'
        ),
        'end_date': FieldSchema(
            type='str',
            pattern=r'^\d{4}-\d{2}-\d{2}$',
            description='End date in YYYY-MM-DD format'
        ),
        'force_rerun': FieldSchema(
            type='bool',
            default=False,
            description='Force re-execution of all steps'
        ),
        'parallel_processing': FieldSchema(
            type='bool',
            default=True,
            description='Enable parallel processing'
        ),
        'max_workers': FieldSchema(
            type='int',
            default=4,
            min_value=1,
            max_value=32,
            description='Maximum number of worker processes'
        ),
        'validation_enabled': FieldSchema(
            type='bool',
            default=True,
            description='Enable data validation'
        ),
        'monitoring_enabled': FieldSchema(
            type='bool',
            default=True,
            description='Enable monitoring and logging'
        ),
        'intensity_percentage': FieldSchema(
            type='float',
            default=1.0,
            min_value=0.1,
            max_value=1.0,
            description='Training intensity (0.1 to 1.0)'
        )
    })

def create_data_collection_schema() -> ConfigSchema:
    """Create schema for data collection configuration."""
    return ConfigSchema({
        'target_timeframes': FieldSchema(
            type='list',
            default=['5m', '15m', '30m', '1h'],
            description='Target timeframes for resampling'
        ),
        'lookback_days': FieldSchema(
            type='int',
            default=30,
            min_value=1,
            max_value=365,
            description='Days of historical data to download'
        ),
        'add_technical_indicators': FieldSchema(
            type='bool',
            default=False,
            description='Add technical indicators during preparation'
        ),
        'gap_fill_enabled': FieldSchema(
            type='bool',
            default=True,
            description='Enable gap filling'
        ),
        'quality_threshold': FieldSchema(
            type='float',
            default=0.8,
            min_value=0.0,
            max_value=1.0,
            description='Data quality threshold'
        )
    })

def validate_pipeline_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate pipeline configuration with detailed error reporting."""
    schema = create_pipeline_config_schema()
    validator = ConfigValidator(schema)

    try:
        validated_config = validator.validate(config)
        logger.info("Pipeline configuration validation passed")
        return validated_config
    except ConfigurationError as e:
        logger.error(f"Pipeline configuration validation failed: {e}")
        raise

def validate_data_collection_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate data collection configuration."""
    schema = create_data_collection_schema()
    validator = ConfigValidator(schema)

    try:
        validated_config = validator.validate(config)
        logger.info("Data collection configuration validation passed")
        return validated_config
    except ConfigurationError as e:
        logger.error(f"Data collection configuration validation failed: {e}")
        raise

# Global schemas
PIPELINE_CONFIG_SCHEMA = create_pipeline_config_schema()
DATA_COLLECTION_SCHEMA = create_data_collection_schema()

def get_schema_for_stage(stage: str) -> ConfigSchema:
    """Get schema for a specific pipeline stage."""
    schemas = {
        'data_collection': DATA_COLLECTION_SCHEMA,
        'market_analysis': ConfigSchema({}),  # Add as needed
        'model_training': ConfigSchema({}),   # Add as needed
        'backtesting': ConfigSchema({})       # Add as needed
    }

    return schemas.get(stage, ConfigSchema({}))

# Export all classes and functions
__all__ = [
    'ValidationRule', 'FieldSchema', 'ConfigSchema', 'ConfigValidator',
    'create_pipeline_config_schema', 'create_data_collection_schema',
    'validate_pipeline_config', 'validate_data_collection_config',
    'get_schema_for_stage', 'PIPELINE_CONFIG_SCHEMA', 'DATA_COLLECTION_SCHEMA'
]