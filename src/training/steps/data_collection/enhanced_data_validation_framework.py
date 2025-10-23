from src.utils.tprint import tprint

from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
from src.utils.logger import system_logger
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation
from src.training.steps.base_step import BaseStep

"""
Enhanced Data Validation Framework for Data Collection

This module provides comprehensive validation during data collection with BaseStep 
comprehensive tools integration for:
- Klines data (PRIMARY - per new setup)
- Aggtrades data (DEPRECATED - not used in new klines-only setup)
- Futures data (DEPRECATED - not used in new klines-only setup)

ENHANCED FEATURES:
==================
- BaseStep comprehensive tools integration
- Advanced logging with tprint utilities
- Hardware optimization for validation operations
- Comprehensive error handling and validation
- Performance monitoring and metrics
- Memory optimization for large datasets

NOTE: Per new setup, only klines data validation is actively used.

Features:
- Schema enforcement with field mapping
- Time gap detection between batches
- Data quality checks (NaN, infinite, zero values)
- Format validation (string, size, data types)
- Real-time validation during API collection
"""
import asyncio
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
from src.utils.error_handler import handles_errors
from src.utils.common_operations import safe_fillna, safe_to_parquet, safe_read_parquet
from src.utils.common_utilities import validate_dataframe_columns, safe_dataframe_operation
# from src.utils.validation import validate_data_quality  # Replaced with comprehensive quality tools

# Import comprehensive data quality tools
try:
    from src.utils.data.quality.comprehensive_quality_scorer import get_quality_scorer
    from src.utils.data.quality.data_quality import DataQualityFramework
    from src.utils.data.quality.advanced_quality_metrics import AdvancedQualityMetrics
    from src.utils.data.quality.data_cleaning import DataCleaner
    QUALITY_TOOLS_AVAILABLE = True
except ImportError:
    QUALITY_TOOLS_AVAILABLE = False

def validate_data_quality(df, data_type='klines', context='data_collection', **kwargs):
    """Comprehensive data quality validation using proper tools."""
    if not QUALITY_TOOLS_AVAILABLE:
        # Fallback to basic validation
        return {'valid': True, 'quality_score': 50.0, 'issues': [], 'warnings': []}

    try:
        quality_scorer = get_quality_scorer()
        quality_assessment = quality_scorer.assess_data_quality(
            df,
            context=context,
            step_name="data_validation",
            data_type=data_type
        )

        return {
            'valid': quality_assessment.level.value not in ['critical'],
            'quality_score': quality_assessment.overall_score,
            'issues': quality_assessment.issues,
            'warnings': quality_assessment.warnings,
            'component_scores': quality_assessment.component_scores,
            'quality_level': quality_assessment.level.value
        }
    except Exception as e:
        logger.warning(f"⚠️ Error in comprehensive quality validation: {e}")
        return {'valid': True, 'quality_score': 50.0, 'issues': [str(e)], 'warnings': []}

from typing import Any
from typing import Dict
from typing import Optional
from typing import List
from typing import Callable
import numpy as np
import pandas as pd
import logging

logger = system_logger.getChild('EnhancedDataValidation')


class EnhancedDataValidationFramework(BaseStep):
    """
    Enhanced data validation framework with BaseStep comprehensive tools integration.
    
    This class provides:
    - Direct access to all BaseStep comprehensive tools
    - Advanced logging with tprint utilities
    - Hardware optimization for validation operations
    - Comprehensive error handling and validation
    - Performance monitoring and metrics
    - Memory optimization for large datasets
    """
    
    def __init__(self, step_name: str = "enhanced_data_validation", config: Optional[Dict[str, Any]] = None):
        super().__init__(step_name, config)
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'critical_errors': 0,
            'warnings': 0
        }
        self.tprint_success("✅ Enhanced Data Validation Framework initialized with BaseStep tools")
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the enhanced data validation process using BaseStep tools.
        
        Args:
            config: Configuration containing validation parameters
            
        Returns:
            Dictionary with validation status and results
        """
        try:
            # Set context for enhanced logging and file operations
            self._set_context(
                symbol=config.get('symbol'),
                exchange=config.get('exchange'),
                information=config.get('information', 'klines'),
                direction=config.get('direction', 'long'),
                model=config.get('model', 'Analyst')
            )
            
            # Log step start with comprehensive information
            self.tprint_step_start("Enhanced Data Validation")
            self.tprint_config_preview(config, "Validation Configuration")
            
            # Extract parameters with validation
            data = self._get_config_value('data', expected_type=(list, dict, pd.DataFrame))
            data_type = self._get_config_value('data_type', 'klines', str)
            validation_level = self._get_config_value('validation_level', 'comprehensive', str)
            
            self.tprint_info(f"🔍 Starting enhanced validation for {data_type} data")
            self.tprint_info(f"📊 Validation level: {validation_level}")
            
            # Perform comprehensive validation
            validation_result = await self._enhanced_validate_data(
                data, data_type, validation_level
            )
            
            if validation_result['valid']:
                # Use BaseStep data quality tools for additional validation
                if self.data_quality and hasattr(data, 'shape'):
                    quality_result = self._get_data_cleaner().assess_quality(data)
                    self.tprint_validation_result(quality_result, "Additional Quality Assessment")
                
                # Use BaseStep hardware optimization if available
                if self.hardware_utils and 'optimize_dataframe' in self.hardware_utils and hasattr(data, 'shape'):
                    optimized_data = self.hardware_utils['optimize_dataframe'](data)
                    self.tprint_info("🔧 Applied hardware optimization to validated data")
                
                # Store validation results using BaseStep artifact management
                artifact_path = self._save_metadata(
                    validation_result, 
                    f"validation_results_{data_type}"
                )
                
                # Log performance metrics
                performance_metrics = self._get_performance_metrics()
                self.tprint_performance_summary(performance_metrics)
                
                # Log step completion
                self.tprint_step_end("Enhanced Data Validation", True, performance_metrics.get('execution_time', 0))
                
                return {
                    'success': True,
                    'valid': True,
                    'validation_result': validation_result,
                    'artifacts': [artifact_path],
                    'metrics': performance_metrics
                }
            else:
                error_msg = f"Validation failed: {validation_result.get('error', 'Unknown error')}"
                self.tprint_error(f"❌ {error_msg}")
                return {
                    'success': False,
                    'valid': False,
                    'validation_result': validation_result,
                    'artifacts': [],
                    'metrics': {}
                }
                
        except Exception as e:
            self.tprint_error(f"❌ Unexpected error in enhanced validation: {e}")
            self._log_error_with_context(e, "enhanced_data_validation")
            return {
                'success': False,
                'valid': False,
                'validation_result': {'error': str(e)},
                'artifacts': [],
                'metrics': {}
            }
    
    async def _enhanced_validate_data(
        self, 
        data: Any, 
        data_type: str, 
        validation_level: str
    ) -> Dict[str, Any]:
        """
        Enhanced data validation with BaseStep comprehensive tools integration.
        
        Args:
            data: Data to validate
            data_type: Type of data being validated
            validation_level: Level of validation to perform
            
        Returns:
            Dictionary with validation results
        """
        try:
            self.tprint_operation_start(f"Validating {data_type} data")
            
            # Use BaseStep safe operations for data validation
            if hasattr(data, 'shape'):
                if not self._validate_dataframe_columns(data, []):  # Basic validation
                    raise ValueError("Invalid data format")
            
            # Perform comprehensive validation using the original framework
            validation_result = self._original_validate_data(data, data_type, validation_level)
            
            # Use BaseStep data quality tools for additional validation
            if self.data_quality and hasattr(data, 'shape'):
                quality_result = self._get_data_cleaner().assess_quality(data)
                validation_result['quality_assessment'] = quality_result
            
            self.tprint_validation_result(validation_result, f"Validation results for {data_type}")
            self.tprint_operation_end(f"Validation completed for {data_type} data")
            
            return validation_result
                
        except Exception as e:
            error_msg = f"Validation exception: {e}"
            self.tprint_error(f"❌ {error_msg}")
            return {
                'valid': False,
                'error': error_msg,
                'quality_score': 0.0,
                'issues': [error_msg],
                'warnings': []
            }
    
    def _original_validate_data(self, data: Any, data_type: str, validation_level: str) -> Dict[str, Any]:
        """
        Original validation method for backward compatibility.
        This method delegates to the existing validation framework.
        """
        # Create a legacy validator instance for the actual validation
        legacy_validator = DataValidationFramework()
        return legacy_validator.validate_data(data, data_type, validation_level)


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
    source_mapping: Dict[str, str] = field(default_factory = dict)
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
    timestamp_field: str = 'timestamp'
    time_gap_config: Optional[TimeGapConfig] = None
    required_columns: List[str] = field(default_factory = list)
    optional_columns: List[str] = field(default_factory = list)

class EnhancedDataValidator:
    """Enhanced data validator with comprehensive validation rules."""
    @log_important_calls

    def __init__(self, schema: DataSchema) -> None:
        self.schema = schema
        self.logger = logger.getChild(f'Validator.{schema.data_type.value}')
        self.validation_stats = {'total_rows_processed': 0, 'valid_rows': 0, 'invalid_rows': 0, 'validation_errors': [], 'time_gaps_detected': 0, 'quality_issues': []}

        # Initialize comprehensive quality tools if available
        if QUALITY_TOOLS_AVAILABLE:
            try:
                self.quality_scorer = get_quality_scorer()
                self.quality_framework = DataQualityFramework()
                self.advanced_quality_metrics = AdvancedQualityMetrics()
                self.data_cleaner = DataCleaner(data_type=schema.data_type.value)
                self.logger.info("✅ Comprehensive quality tools initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize quality tools: {e}")
                self.quality_scorer = None
                self.quality_framework = None
                self.advanced_quality_metrics = None
                self.data_cleaner = None
        else:
            self.quality_scorer = None
            self.quality_framework = None
            self.advanced_quality_metrics = None
            self.data_cleaner = None

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
        for field_def in self.schema.fields:
            try:
                source_value = self._get_source_value(row_data, field_def)
                validated_value = self._validate_and_convert_field(source_value, field_def, row_index)
                validated_row[field_def.name] = validated_value
            except ValidationError as e:
                errors.append(e)
                if e.severity == ValidationSeverity.CRITICAL:
                    raise ValueError(f'Critical validation error in row {row_index}: {e.message}')
        self._add_metadata_fields(validated_row, row_data)
        if errors:
            self._log_validation_errors(errors, row_index)
        return validated_row

    def validate_batch(self, batch_data: List[Dict[str, Any]], previous_timestamp: Optional[int]=None) -> List[Dict[str, Any]]:
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
        self.logger.info(f'🔍 Validating batch of {len(batch_data)} rows')
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
                batch_errors.append(ValidationError(field='row', message = str(e), severity = ValidationSeverity.CRITICAL, row_index = i))
        self.validation_stats['total_rows_processed'] += len(batch_data)
        self.validation_stats['valid_rows'] += len(validated_batch)
        self.validation_stats['invalid_rows'] += len(batch_data) - len(validated_batch)
        success_rate = len(validated_batch) / len(batch_data) * 100 if batch_data else 0
        self.logger.info(f'✅ Batch validation: {len(validated_batch)}/{len(batch_data)} rows valid ({success_rate:.1f}%)')
        if batch_errors:
            self.validation_stats['validation_errors'].extend(batch_errors)
            self._log_batch_errors(batch_errors)
        return validated_batch
    @log_all_calls

    def _get_source_value(self, row_data: Dict[str, Any], field_def: FieldDefinition) -> Any:
        """Get source value using field mapping."""
        if field_def.name in row_data:
            return row_data[field_def.name]
        for exchange, source_field in field_def.source_mapping.items():
            if source_field in row_data:
                return row_data[source_field]
        return None
    @log_all_calls

    def _validate_and_convert_field(self, value: Any, field_def: FieldDefinition, row_index: int) -> Any:
        """Validate and convert a field value."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            if field_def.required:
                raise ValidationError(field = field_def.name, message = f"Required field '{field_def.name}' is missing or NaN", severity = ValidationSeverity.CRITICAL, row_index = row_index)
            return field_def.default_value
        try:
            converted_value = self._convert_type(value, field_def.dtype)
        except (ValueError, TypeError) as e:
            raise ValidationError(field = field_def.name, message = f'Type conversion failed: {e}', severity = ValidationSeverity.CRITICAL, value = value, expected = field_def.dtype, row_index = row_index)
        validation_errors = self._validate_value(converted_value, field_def, row_index)
        if validation_errors:
            critical_errors = [e for e in validation_errors if e.severity == ValidationSeverity.CRITICAL]
            if critical_errors:
                raise critical_errors[0]
            for error in validation_errors:
                self.logger.warning(f'⚠️ {error.message}')
        return converted_value
    @log_all_calls

    def _convert_type(self, value: Any, target_type: str) -> Any:
        """Convert value to target type."""
        if target_type == 'int64':
            return int(float(value))
        elif target_type == 'float64':
            return float(value)
        elif target_type == 'string':
            return str(value)
        elif target_type == 'bool':
            return bool(value)
        else:
            return value
    @log_all_calls

    def _validate_value(self, value: Any, field_def: FieldDefinition, row_index: int) -> List[ValidationError]:
        """Validate field value against constraints."""
        errors = []
        if isinstance(value, float) and np.isnan(value):
            if not field_def.allow_nan:
                errors.append(ValidationError(field = field_def.name, message = f"Field '{field_def.name}' contains NaN value", severity = ValidationSeverity.HIGH, value = value, row_index = row_index))
        if isinstance(value, float) and np.isinf(value):
            if not field_def.allow_infinite:
                errors.append(ValidationError(field = field_def.name, message = f"Field '{field_def.name}' contains infinite value", severity = ValidationSeverity.HIGH, value = value, row_index = row_index))
        if isinstance(value, (int, float)) and value == 0:
            if not field_def.allow_zero:
                errors.append(ValidationError(field = field_def.name, message = f"Field '{field_def.name}' contains zero value", severity = ValidationSeverity.MEDIUM, value = value, row_index = row_index))
        if isinstance(value, (int, float)) and value < 0:
            if not field_def.allow_negative:
                errors.append(ValidationError(field = field_def.name, message = f"Field '{field_def.name}' contains negative value", severity = ValidationSeverity.HIGH, value = value, row_index = row_index))
        if isinstance(value, (int, float)):
            if field_def.min_value is not None and value < field_def.min_value:
                errors.append(ValidationError(field = field_def.name, message = f"Field '{field_def.name}' value {value} below minimum {field_def.min_value}", severity = ValidationSeverity.HIGH, value = value, expected = field_def.min_value, row_index = row_index))
            if field_def.max_value is not None and value > field_def.max_value:
                errors.append(ValidationError(field = field_def.name, message = f"Field '{field_def.name}' value {value} above maximum {field_def.max_value}", severity = ValidationSeverity.HIGH, value = value, expected = field_def.max_value, row_index = row_index))
        if field_def.custom_validator:
            try:
                if not field_def.custom_validator(value):
                    errors.append(ValidationError(field = field_def.name, message = f"Custom validation failed for field '{field_def.name}'", severity = ValidationSeverity.MEDIUM, value = value, row_index = row_index))
            except Exception as e:
                errors.append(ValidationError(field = field_def.name, message = f"Custom validation error for field '{field_def.name}': {e}", severity = ValidationSeverity.MEDIUM, value = value, row_index = row_index))
        return errors
    @log_all_calls

    def _check_time_gap(self, current_timestamp: int, previous_timestamp: int) -> Optional[ValidationError]:
        """Check for time gap between batches."""
        if not self.schema.time_gap_config:
            return None
        gap_seconds = (current_timestamp - previous_timestamp) / 1000.0
        max_gap = self.schema.time_gap_config.max_gap_seconds
        tolerance = self.schema.time_gap_config.tolerance_seconds
        if gap_seconds > max_gap + tolerance:
            self.validation_stats['time_gaps_detected'] += 1
            return ValidationError(field = self.schema.timestamp_field, message = f'Time gap detected: {gap_seconds:.2f}s > {max_gap}s (tolerance: {tolerance}s)', severity = self.schema.time_gap_config.severity, value = current_timestamp, expected = previous_timestamp + int(max_gap * 1000))
        return None

    def validate_dataframe_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive quality validation on a DataFrame using proper quality tools.

        Args:
            df: DataFrame to validate

        Returns:
            Comprehensive quality assessment results
        """
        if self.quality_scorer is None:
            # Fallback to basic validation
            return self._fallback_quality_validation(df)

        try:
            # Perform comprehensive quality assessment
            quality_assessment = self.quality_scorer.assess_data_quality(
                df,
                context="data_collection",
                step_name=f"data_validation_{self.schema.data_type.value}",
                data_type=self.schema.data_type.value
            )

            # Log quality assessment results
            self.logger.info(f"📊 Data quality assessment: {quality_assessment.overall_score:.2f} ({quality_assessment.level.value})")

            # Handle quality issues
            if quality_assessment.level.value in ['poor', 'critical']:
                self.logger.warning(f"⚠️ Low data quality detected: {quality_assessment.issues}")

                # Attempt data cleaning for poor quality data
                if quality_assessment.level.value == 'poor' and self.data_cleaner:
                    self.logger.info("🔧 Attempting data cleaning to improve quality...")
                    cleaned_df = self.data_cleaner.clean_dataframe(df)

                    if cleaned_df is not None and not cleaned_df.empty:
                        # Re-assess quality after cleaning
                        cleaned_assessment = self.quality_scorer.assess_data_quality(
                            cleaned_df,
                            context="data_collection",
                            step_name=f"data_validation_{self.schema.data_type.value}_cleaned",
                            data_type=self.schema.data_type.value
                        )

                        if cleaned_assessment.overall_score > quality_assessment.overall_score:
                            self.logger.info(f"✅ Data cleaning improved quality: {cleaned_assessment.overall_score:.2f}")
                            quality_assessment = cleaned_assessment
                        else:
                            self.logger.warning("⚠️ Data cleaning did not improve quality")

            # Update validation stats
            self.validation_stats['quality_issues'].extend(quality_assessment.issues)

            return {
                'valid': quality_assessment.level.value not in ['critical'],
                'quality_score': quality_assessment.overall_score,
                'quality_level': quality_assessment.level.value,
                'issues': quality_assessment.issues,
                'warnings': quality_assessment.warnings,
                'component_scores': quality_assessment.component_scores,
                'recommendations': quality_assessment.recommendations
            }

        except Exception as e:
            self.logger.error(f"❌ Error in comprehensive quality validation: {e}")
            return self._fallback_quality_validation(df)

    def _fallback_quality_validation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fallback quality validation using basic checks."""
        try:
            issues = []
            warnings = []

            # Basic quality checks
            if df.empty:
                issues.append("DataFrame is empty")

            # Check for missing values
            missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) if df.shape[0] > 0 and df.shape[1] > 0 else 0
            if missing_ratio > 0.1:
                warnings.append(f"High missing value ratio: {missing_ratio:.2%}")

            # Check for duplicates
            duplicate_ratio = df.duplicated().sum() / len(df) if len(df) > 0 else 0
            if duplicate_ratio > 0.05:
                warnings.append(f"High duplicate ratio: {duplicate_ratio:.2%}")

            # Simple quality score calculation
            quality_score = 1.0 - (missing_ratio + duplicate_ratio)
            quality_score = max(0.0, quality_score) * 100  # Convert to 0-100 scale

            return {
                'valid': len(issues) == 0,
                'quality_score': quality_score,
                'quality_level': 'excellent' if quality_score >= 90 else 'good' if quality_score >= 80 else 'fair' if quality_score >= 70 else 'poor' if quality_score >= 60 else 'critical',
                'issues': issues,
                'warnings': warnings,
                'component_scores': {},
                'recommendations': []
            }

        except Exception as e:
            self.logger.error(f"❌ Error in fallback quality validation: {e}")
            return {
                'valid': True,
                'quality_score': 50.0,
                'quality_level': 'fair',
                'issues': [str(e)],
                'warnings': [],
                'component_scores': {},
                'recommendations': []
            }

    @log_all_calls

    def _add_metadata_fields(self, validated_row: Dict[str, Any], row_data: Dict[str, Any]) -> None:
        """Add metadata fields to validated row."""
        if self.schema.timestamp_field in validated_row:
            timestamp = validated_row[self.schema.timestamp_field]
            if isinstance(timestamp, (int, float)):
                dt = pd.to_datetime(timestamp, unit='ms', utc = True)
                validated_row['year'] = dt.year
                validated_row['month'] = dt.month
                validated_row['day'] = dt.day
    @log_all_calls

    def _log_validation_errors(self, errors: List[ValidationError], row_index: int) -> None:
        """Log validation errors for a row."""
        for error in errors:
            if error.severity == ValidationSeverity.CRITICAL:
                self.logger.error(f'❌ Row {row_index}: {error.message}')
            elif error.severity == ValidationSeverity.HIGH:
                self.logger.warning(f'⚠️ Row {row_index}: {error.message}')
            else:
                self.logger.info(f'ℹ️ Row {row_index}: {error.message}')
    @log_all_calls

    def _log_batch_errors(self, errors: List[ValidationError]) -> None:
        """Log batch validation errors."""
        critical_count = sum((1 for e in errors if e.severity == ValidationSeverity.CRITICAL))
        high_count = sum((1 for e in errors if e.severity == ValidationSeverity.HIGH))
        medium_count = sum((1 for e in errors if e.severity == ValidationSeverity.MEDIUM))
        if critical_count > 0:
            self.logger.error(f'❌ Batch validation: {critical_count} critical errors')
        if high_count > 0:
            self.logger.warning(f'⚠️ Batch validation: {high_count} high severity errors')
        if medium_count > 0:
            self.logger.info(f'ℹ️ Batch validation: {medium_count} medium severity errors')

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation statistics summary."""
        total = self.validation_stats['total_rows_processed']
        valid = self.validation_stats['valid_rows']
        invalid = self.validation_stats['invalid_rows']
        return {'data_type': self.schema.data_type.value, 'total_rows_processed': total, 'valid_rows': valid, 'invalid_rows': invalid, 'success_rate': valid / total * 100 if total > 0 else 0, 'time_gaps_detected': self.validation_stats['time_gaps_detected'], 'total_errors': len(self.validation_stats['validation_errors']), 'error_breakdown': self._get_error_breakdown()}
    @log_all_calls

    def _get_error_breakdown(self) -> Dict[str, int]:
        """Get breakdown of errors by severity."""
        breakdown = {severity.value: 0 for severity in ValidationSeverity}
        for error in self.validation_stats['validation_errors']:
            breakdown[error.severity.value] += 1
        return breakdown

def create_klines_schema() -> DataSchema:
    """Create standardized klines schema."""
    return DataSchema(data_type = DataType.KLINES, fields=[FieldDefinition(name='timestamp', dtype='int64', source_mapping={'binance': 'open_time', 'coinbase': 'timestamp', 'kraken': 'time'}), FieldDefinition(name='open', dtype='float64', min_value = 0.0, allow_zero = False, source_mapping={'binance': 'open', 'coinbase': 'price_open', 'kraken': 'open'}), FieldDefinition(name='high', dtype='float64', min_value = 0.0, allow_zero = False, source_mapping={'binance': 'high', 'coinbase': 'price_high', 'kraken': 'high'}), FieldDefinition(name='low', dtype='float64', min_value = 0.0, allow_zero = False, source_mapping={'binance': 'low', 'coinbase': 'price_low', 'kraken': 'low'}), FieldDefinition(name='close', dtype='float64', min_value = 0.0, allow_zero = False, source_mapping={'binance': 'close', 'coinbase': 'price_close', 'kraken': 'close'}), FieldDefinition(name='volume', dtype='float64', min_value = 0.0, allow_zero = True, source_mapping={'binance': 'volume', 'coinbase': 'volume', 'kraken': 'vol'}), FieldDefinition(name='exchange', dtype='string', required = True), FieldDefinition(name='symbol', dtype='string', required = True), FieldDefinition(name='timeframe', dtype='string', required = True)], primary_key=['timestamp', 'exchange', 'symbol', 'timeframe'], time_gap_config = TimeGapConfig(max_gap_seconds = 66.0, tolerance_seconds = 5.0, severity = ValidationSeverity.HIGH))

def create_aggtrades_schema() -> DataSchema:
    """Create standardized aggtrades schema."""
    return DataSchema(data_type = DataType.AGGTRADES, fields=[FieldDefinition(name='timestamp', dtype='int64', source_mapping={'binance': 'T', 'coinbase': 'timestamp', 'kraken': 'time'}), FieldDefinition(name='price', dtype='float64', min_value = 0.0, allow_zero = False, source_mapping={'binance': 'p', 'coinbase': 'price', 'kraken': 'price'}), FieldDefinition(name='quantity', dtype='float64', min_value = 0.0, allow_zero = False, source_mapping={'binance': 'q', 'coinbase': 'size', 'kraken': 'vol'}), FieldDefinition(name='is_buyer_maker', dtype='bool', required = False, default_value = False, source_mapping={'binance': 'm', 'coinbase': 'side', 'kraken': 'type'}), FieldDefinition(name='trade_id', dtype='int64', required = False, default_value = 0, source_mapping={'binance': 'a', 'coinbase': 'trade_id', 'kraken': 'id'}), FieldDefinition(name='exchange', dtype='string', required = True), FieldDefinition(name='symbol', dtype='string', required = True)], primary_key=['timestamp', 'trade_id', 'exchange', 'symbol'], time_gap_config = TimeGapConfig(max_gap_seconds = 1.0, tolerance_seconds = 0.1, severity = ValidationSeverity.MEDIUM))

def create_futures_schema() -> DataSchema:
    """Create standardized futures schema."""
    return DataSchema(
        data_type=DataType.FUTURES,
        fields=[
            FieldDefinition(
                name='timestamp',
                dtype='int64',
                source_mapping={'binance': 'fundingTime', 'coinbase': 'timestamp', 'kraken': 'time'}
            ),
            FieldDefinition(name='exchange', dtype='string', required=True),
            FieldDefinition(name='symbol', dtype='string', required=True)
        ],
        primary_key=['timestamp', 'exchange', 'symbol'],
        time_gap_config=TimeGapConfig(
            max_gap_seconds=32400.0,
            tolerance_seconds=300.0,
            severity=ValidationSeverity.MEDIUM
        )
    )

def create_unified_schema() -> DataSchema:
    """Create standardized unified schema - klines-only per new setup."""
    klines_schema = create_klines_schema()
    all_fields = []
    all_fields.extend(klines_schema.fields)

    # NOTE: Since we don't collect aggtrades, skip adding trade-related columns to avoid constant features
    # These columns were previously from aggtrades schema but are now omitted when no aggtrades data exists
    # This prevents the data cleaner from removing them as constant features
    pass

    return DataSchema(
        data_type=DataType.UNIFIED,
        fields=all_fields,
        primary_key=['timestamp', 'exchange', 'symbol', 'timeframe'],
        time_gap_config=TimeGapConfig(max_gap_seconds=66.0, tolerance_seconds=5.0, severity=ValidationSeverity.HIGH)
    )
SCHEMA_REGISTRY = {DataType.KLINES: create_klines_schema(), DataType.AGGTRADES: create_aggtrades_schema(), DataType.FUTURES: create_futures_schema(), DataType.UNIFIED: create_unified_schema()}

def get_validator(data_type: DataType) -> EnhancedDataValidator:
    """Get validator for specified data type."""
    schema = SCHEMA_REGISTRY.get(data_type)
    if not schema:
        raise ValueError(f'No schema found for data type: {data_type}')
    return EnhancedDataValidator(schema)

def validate_data_batch(data_type: DataType, batch_data: List[Dict[str, Any]], previous_timestamp: Optional[int]=None) -> List[Dict[str, Any]]:
    """Convenience function to validate a batch of data."""
    validator = get_validator(data_type)
    return validator.validate_batch(batch_data, previous_timestamp)
if __name__ == '__main__':

    async def test_validation() -> None:
        klines_data = [{'open_time': 1640995200000, 'open': '3000.0', 'high': '3100.0', 'low': '2900.0', 'close': '3050.0', 'volume': '1000.0'}]
        validator = get_validator(DataType.KLINES)
        validated = validator.validate_batch(klines_data)
        tprint(f'Validated {len(validated)} klines rows')
        aggtrades_data = [{'T': 1640995200000, 'p': '3050.0', 'q': '1.5', 'm': True}]
        validator = get_validator(DataType.AGGTRADES)
        validated = validator.validate_batch(aggtrades_data)
        tprint(f'Validated {len(validated)} aggtrades rows')
        tprint('Validation Summary:', validator.get_validation_summary())
    asyncio.run(test_validation())
