from .standardized_parquet_handler import standardized_parquet_handler
"""
Enhanced Validation Framework for Training Steps

This module provides comprehensive validation that ensures:
1. No silent failures - all validation failures are properly logged and handled
2. Proper error propagation for validation failures
3. Comprehensive data quality checks
4. Process validation and monitoring
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from src.utils.logger import system_logger
import json
import logging
import os
import time

from .enhanced_error_handling import (

    EnhancedErrorHandler,
    CriticalProcessError,
    ErrorSeverity,
    ErrorCategory,
    ErrorContext,
    ErrorRecord
)

class ValidationLevel(Enum):
    """Validation levels for different scenarios."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    CRITICAL = "critical"

class ValidationResult:
    """Result of a validation check."""
    
    def __init__(self, 
                 passed: bool, 
                 message: str, 
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 details: Optional[Dict[str, Any]] = None):
        self.passed = passed
        self.message = message
        self.severity = severity
        self.details = details or {}
        self.timestamp = datetime.now()
    
    def __bool__(self) -> bool:
        return self.passed

class EnhancedValidator:
    """Enhanced validator with comprehensive checks and fail-fast behavior."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = system_logger.getChild('EnhancedValidator')
        self.error_handler = EnhancedErrorHandler()
        self.validation_results: List[ValidationResult] = []
    
    async def validate_data_quality(self, 
                                  data: pd.DataFrame, 
                                  level: ValidationLevel = ValidationLevel.STANDARD,
                                  context: str = "data_validation") -> ValidationResult:
        """Validate data quality with comprehensive checks."""
        self.logger.info(f'🔍 Validating data quality (level: {level.value})...')
        
        try:
            # Basic checks
            if data is None:
                return ValidationResult(
                    passed=False,
                    message="Data is None",
                    severity=ErrorSeverity.CRITICAL,
                    details={'context': context}
                )
            
            if data.empty:
                return ValidationResult(
                    passed=False,
                    message="Data is empty",
                    severity=ErrorSeverity.CRITICAL,
                    details={'context': context, 'shape': data.shape}
                )
            
            # Shape validation
            if len(data) < 100:  # Minimum rows
                return ValidationResult(
                    passed=False,
                    message=f"Insufficient data rows: {len(data)} (minimum: 100)",
                    severity=ErrorSeverity.HIGH,
                    details={'context': context, 'rows': len(data), 'minimum': 100}
                )
            
            if len(data.columns) < 5:  # Minimum columns
                return ValidationResult(
                    passed=False,
                    message=f"Insufficient data columns: {len(data.columns)} (minimum: 5)",
                    severity=ErrorSeverity.HIGH,
                    details={'context': context, 'columns': len(data.columns), 'minimum': 5}
                )
            
            # NaN validation
            nan_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            max_nan_ratio = 0.1 if level in [ValidationLevel.BASIC, ValidationLevel.STANDARD] else 0.05
            
            if nan_ratio > max_nan_ratio:
                return ValidationResult(
                    passed=False,
                    message=f"Too many NaN values: {nan_ratio:.2%} (maximum: {max_nan_ratio:.2%})",
                    severity=ErrorSeverity.HIGH,
                    details={'context': context, 'nan_ratio': nan_ratio, 'max_nan_ratio': max_nan_ratio}
                )
            
            # Duplicate validation
            duplicate_ratio = data.duplicated().sum() / len(data)
            max_duplicate_ratio = 0.05 if level in [ValidationLevel.BASIC, ValidationLevel.STANDARD] else 0.01
            
            if duplicate_ratio > max_duplicate_ratio:
                return ValidationResult(
                    passed=False,
                    message=f"Too many duplicate rows: {duplicate_ratio:.2%} (maximum: {max_duplicate_ratio:.2%})",
                    severity=ErrorSeverity.MEDIUM,
                    details={'context': context, 'duplicate_ratio': duplicate_ratio, 'max_duplicate_ratio': max_duplicate_ratio}
                )
            
            # Data type validation
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) < 3:  # Minimum numeric columns
                return ValidationResult(
                    passed=False,
                    message=f"Insufficient numeric columns: {len(numeric_columns)} (minimum: 3)",
                    severity=ErrorSeverity.HIGH,
                    details={'context': context, 'numeric_columns': len(numeric_columns), 'minimum': 3}
                )
            
            # Range validation for numeric columns
            for col in numeric_columns:
                if data[col].dtype in ['int64', 'float64']:
                    if data[col].isnull().all():
                        return ValidationResult(
                            passed=False,
                            message=f"Column {col} is entirely NaN",
                            severity=ErrorSeverity.HIGH,
                            details={'context': context, 'column': col}
                        )
                    
                    # Check for infinite values
                    inf_count = np.isinf(data[col]).sum()
                    if inf_count > 0:
                        return ValidationResult(
                            passed=False,
                            message=f"Column {col} contains {inf_count} infinite values",
                            severity=ErrorSeverity.HIGH,
                            details={'context': context, 'column': col, 'inf_count': inf_count}
                        )
            
            # Strict level checks
            if level in [ValidationLevel.STRICT, ValidationLevel.CRITICAL]:
                # Check for constant columns
                constant_columns = []
                for col in numeric_columns:
                    if data[col].nunique() <= 1:
                        constant_columns.append(col)
                
                if constant_columns:
                    return ValidationResult(
                        passed=False,
                        message=f"Constant columns found: {constant_columns}",
                        severity=ErrorSeverity.MEDIUM,
                        details={'context': context, 'constant_columns': constant_columns}
                    )
                
                # Check for highly correlated columns
                if len(numeric_columns) > 1:
                    corr_matrix = data[numeric_columns].corr()
                    high_corr_pairs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            if abs(corr_matrix.iloc[i, j]) > 0.95:
                                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
                    
                    if high_corr_pairs:
                        return ValidationResult(
                            passed=False,
                            message=f"Highly correlated columns found: {high_corr_pairs}",
                            severity=ErrorSeverity.LOW,
                            details={'context': context, 'high_corr_pairs': high_corr_pairs}
                        )
            
            # Critical level checks
            if level == ValidationLevel.CRITICAL:
                # Check for data leakage (future information)
                if 'timestamp' in data.columns or 'datetime' in data.columns:
                    time_col = 'timestamp' if 'timestamp' in data.columns else 'datetime'
                    if not pd.api.types.is_datetime64_any_dtype(data[time_col]):
                        return ValidationResult(
                            passed=False,
                            message=f"Time column {time_col} is not datetime type",
                            severity=ErrorSeverity.HIGH,
                            details={'context': context, 'time_column': time_col}
                        )
                    
                    # Check for proper time ordering
                    if not data[time_col].is_monotonic_increasing:
                        return ValidationResult(
                            passed=False,
                            message=f"Time column {time_col} is not properly ordered",
                            severity=ErrorSeverity.HIGH,
                            details={'context': context, 'time_column': time_col}
                        )
            
            self.logger.info(f'✅ Data quality validation passed (level: {level.value})')
            return ValidationResult(
                passed=True,
                message=f"Data quality validation passed (level: {level.value})",
                severity=ErrorSeverity.LOW,
                details={'context': context, 'shape': data.shape, 'nan_ratio': nan_ratio, 'duplicate_ratio': duplicate_ratio}
            )
            
        except Exception as e:
            self.logger.exception(f'❌ Data quality validation failed: {e}')
            return ValidationResult(
                passed=False,
                message=f"Data quality validation failed: {str(e)}",
                severity=ErrorSeverity.CRITICAL,
                details={'context': context, 'error': str(e)}
            )
    
    async def validate_process_completion(self, 
                                        process_name: str, 
                                        expected_outputs: List[str],
                                        output_directory: str,
                                        level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
        """Validate that a process completed successfully."""
        self.logger.info(f'🔍 Validating process completion: {process_name}...')
        
        try:
            output_path = Path(output_directory)
            if not output_path.exists():
                return ValidationResult(
                    passed=False,
                    message=f"Output directory does not exist: {output_directory}",
                    severity=ErrorSeverity.CRITICAL,
                    details={'process_name': process_name, 'output_directory': output_directory}
                )
            
            missing_outputs = []
            for output_file in expected_outputs:
                file_path = output_path / output_file
                if not file_path.exists():
                    missing_outputs.append(output_file)
                elif file_path.stat().st_size == 0:
                    return ValidationResult(
                        passed=False,
                        message=f"Output file is empty: {output_file}",
                        severity=ErrorSeverity.HIGH,
                        details={'process_name': process_name, 'output_file': output_file}
                    )
            
            if missing_outputs:
                return ValidationResult(
                    passed=False,
                    message=f"Missing output files: {missing_outputs}",
                    severity=ErrorSeverity.HIGH,
                    details={'process_name': process_name, 'missing_outputs': missing_outputs}
                )
            
            # Validate file contents for critical level
            if level == ValidationLevel.CRITICAL:
                for output_file in expected_outputs:
                    file_path = output_path / output_file
                    if file_path.suffix == '.parquet':
                        try:
                            df = standardized_parquet_handler.read_parquet_standardized(file_path)
                            if df.empty:
                                return ValidationResult(
                                    passed=False,
                                    message=f"Output file contains no data: {output_file}",
                                    severity=ErrorSeverity.HIGH,
                                    details={'process_name': process_name, 'output_file': output_file}
                                )
                        except Exception as e:
                            return ValidationResult(
                                passed=False,
                                message=f"Output file is corrupted: {output_file} - {str(e)}",
                                severity=ErrorSeverity.HIGH,
                                details={'process_name': process_name, 'output_file': output_file, 'error': str(e)}
                            )
            
            self.logger.info(f'✅ Process completion validation passed: {process_name}')
            return ValidationResult(
                passed=True,
                message=f"Process completion validation passed: {process_name}",
                severity=ErrorSeverity.LOW,
                details={'process_name': process_name, 'output_files': expected_outputs}
            )
            
        except Exception as e:
            self.logger.exception(f'❌ Process completion validation failed: {e}')
            return ValidationResult(
                passed=False,
                message=f"Process completion validation failed: {str(e)}",
                severity=ErrorSeverity.CRITICAL,
                details={'process_name': process_name, 'error': str(e)}
            )
    
    async def validate_model_quality(self, 
                                   model_path: str, 
                                   test_data: pd.DataFrame,
                                   level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
        """Validate model quality and performance."""
        self.logger.info(f'🔍 Validating model quality: {model_path}...')
        
        try:
            model_file = Path(model_path)
            if not model_file.exists():
                return ValidationResult(
                    passed=False,
                    message=f"Model file does not exist: {model_path}",
                    severity=ErrorSeverity.CRITICAL,
                    details={'model_path': model_path}
                )
            
            if model_file.stat().st_size == 0:
                return ValidationResult(
                    passed=False,
                    message=f"Model file is empty: {model_path}",
                    severity=ErrorSeverity.CRITICAL,
                    details={'model_path': model_path}
                )
            
            # Try to load the model
            try:
                import joblib
                model = joblib.load(model_path)
            except Exception as e:
                return ValidationResult(
                    passed=False,
                    message=f"Model file is corrupted: {str(e)}",
                    severity=ErrorSeverity.CRITICAL,
                    details={'model_path': model_path, 'error': str(e)}
                )
            
            # Basic model validation
            if not hasattr(model, 'predict'):
                return ValidationResult(
                    passed=False,
                    message="Model does not have predict method",
                    severity=ErrorSeverity.HIGH,
                    details={'model_path': model_path, 'model_type': type(model).__name__}
                )
            
            # Test prediction if test data is available
            if test_data is not None and not test_data.empty:
                try:
                    # Prepare features (exclude target columns)
                    feature_columns = [col for col in test_data.columns 
                                     if col not in ['target', 'label', 'y', 'close']]
                    if not feature_columns:
                        return ValidationResult(
                            passed=False,
                            message="No feature columns found in test data",
                            severity=ErrorSeverity.HIGH,
                            details={'model_path': model_path}
                        )
                    
                    X_test = test_data[feature_columns]
                    
                    # Make predictions
                    predictions = model.predict(X_test)
                    
                    if predictions is None or len(predictions) == 0:
                        return ValidationResult(
                            passed=False,
                            message="Model predictions are empty",
                            severity=ErrorSeverity.HIGH,
                            details={'model_path': model_path}
                        )
                    
                    # Check for NaN predictions
                    nan_predictions = np.isnan(predictions).sum() if hasattr(predictions, '__len__') else 0
                    if nan_predictions > 0:
                        return ValidationResult(
                            passed=False,
                            message=f"Model produces {nan_predictions} NaN predictions",
                            severity=ErrorSeverity.HIGH,
                            details={'model_path': model_path, 'nan_predictions': nan_predictions}
                        )
                    
                    # Check for constant predictions
                    if hasattr(predictions, '__len__') and len(predictions) > 1:
                        unique_predictions = len(np.unique(predictions))
                        if unique_predictions == 1:
                            return ValidationResult(
                                passed=False,
                                message="Model produces constant predictions",
                                severity=ErrorSeverity.MEDIUM,
                                details={'model_path': model_path, 'unique_predictions': unique_predictions}
                            )
                    
                except Exception as e:
                    return ValidationResult(
                        passed=False,
                        message=f"Model prediction test failed: {str(e)}",
                        severity=ErrorSeverity.HIGH,
                        details={'model_path': model_path, 'error': str(e)}
                    )
            
            self.logger.info(f'✅ Model quality validation passed: {model_path}')
            return ValidationResult(
                passed=True,
                message=f"Model quality validation passed: {model_path}",
                severity=ErrorSeverity.LOW,
                details={'model_path': model_path, 'model_type': type(model).__name__}
            )
            
        except Exception as e:
            self.logger.exception(f'❌ Model quality validation failed: {e}')
            return ValidationResult(
                passed=False,
                message=f"Model quality validation failed: {str(e)}",
                severity=ErrorSeverity.CRITICAL,
                details={'model_path': model_path, 'error': str(e)}
            )
    
    async def validate_pipeline_step(self, 
                                   step_name: str, 
                                   step_result: Any,
                                   expected_outputs: List[str],
                                   output_directory: str,
                                   level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
        """Validate a complete pipeline step."""
        self.logger.info(f'🔍 Validating pipeline step: {step_name}...')
        
        try:
            # Check step result
            if step_result is None:
                return ValidationResult(
                    passed=False,
                    message=f"Step {step_name} returned None",
                    severity=ErrorSeverity.CRITICAL,
                    details={'step_name': step_name}
                )
            
            # Check if step result indicates success
            if isinstance(step_result, dict):
                if step_result.get('success') is False:
                    return ValidationResult(
                        passed=False,
                        message=f"Step {step_name} reported failure",
                        severity=ErrorSeverity.HIGH,
                        details={'step_name': step_name, 'step_result': step_result}
                    )
            elif isinstance(step_result, bool):
                if not step_result:
                    return ValidationResult(
                        passed=False,
                        message=f"Step {step_name} returned False",
                        severity=ErrorSeverity.HIGH,
                        details={'step_name': step_name}
                    )
            
            # Validate process completion
            process_validation = await self.validate_process_completion(
                step_name, expected_outputs, output_directory, level
            )
            
            if not process_validation.passed:
                return process_validation
            
            self.logger.info(f'✅ Pipeline step validation passed: {step_name}')
            return ValidationResult(
                passed=True,
                message=f"Pipeline step validation passed: {step_name}",
                severity=ErrorSeverity.LOW,
                details={'step_name': step_name, 'expected_outputs': expected_outputs}
            )
            
        except Exception as e:
            self.logger.exception(f'❌ Pipeline step validation failed: {e}')
            return ValidationResult(
                passed=False,
                message=f"Pipeline step validation failed: {str(e)}",
                severity=ErrorSeverity.CRITICAL,
                details={'step_name': step_name, 'error': str(e)}
            )
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results."""
        try:
            total_validations = len(self.validation_results)
            passed_validations = len([r for r in self.validation_results if r.passed])
            failed_validations = total_validations - passed_validations
            
            # Count by severity
            severity_counts = {}
            for result in self.validation_results:
                severity = result.severity.value
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Recent validations (last 24 hours)
            recent_cutoff = datetime.now().timestamp() - 86400
            recent_validations = len([r for r in self.validation_results 
                                    if r.timestamp.timestamp() > recent_cutoff])
            
            return {
                'total_validations': total_validations,
                'passed_validations': passed_validations,
                'failed_validations': failed_validations,
                'success_rate': passed_validations / total_validations if total_validations > 0 else 0.0,
                'severity_counts': severity_counts,
                'recent_validations': recent_validations,
                'validation_results': [
                    {
                        'passed': r.passed,
                        'message': r.message,
                        'severity': r.severity.value,
                        'timestamp': r.timestamp.isoformat(),
                        'details': r.details
                    }
                    for r in self.validation_results
                ]
            }
            
        except Exception as e:
            self.logger.error(f"❌ Validation summary generation failed: {e}")
            return {'error': str(e)}

# Global validator instance
_global_validator = EnhancedValidator()

def get_global_validator() -> EnhancedValidator:
    """Get the global validator instance."""
    return _global_validator

def set_global_validator(validator: EnhancedValidator) -> None:
    """Set the global validator instance."""
    global _global_validator
    _global_validator = validator

# Validation decorators
def validate_step_output(level: ValidationLevel = ValidationLevel.STANDARD):
    """Decorator to validate step output."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # Get validator
            validator = get_global_validator()
            
            # Validate result
            if result is None:
                validation_result = ValidationResult(
                    passed=False,
                    message=f"Function {func.__name__} returned None",
                    severity=ErrorSeverity.CRITICAL
                )
            elif isinstance(result, bool) and not result:
                validation_result = ValidationResult(
                    passed=False,
                    message=f"Function {func.__name__} returned False",
                    severity=ErrorSeverity.HIGH
                )
            else:
                validation_result = ValidationResult(
                    passed=True,
                    message=f"Function {func.__name__} validation passed",
                    severity=ErrorSeverity.LOW
                )
            
            validator.validation_results.append(validation_result)
            
            if not validation_result.passed and level == ValidationLevel.CRITICAL:
                raise CriticalProcessError(
                    f"Critical validation failed: {validation_result.message}",
                    ErrorRecord(
                        error_id=f"validation_{func.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        error_type="ValidationError",
                        error_message=validation_result.message,
                        severity=validation_result.severity,
                        category=ErrorCategory.VALIDATION,
                        context=ErrorContext(
                            function_name=func.__name__,
                            step_name="validation"
                        ),
                        stack_trace="",
                        should_fail_fast=True
                    )
                )
            
            return result
        return wrapper
    return decorator