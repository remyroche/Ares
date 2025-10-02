"""
Shared validation utilities for regime data splitting module.

This module provides standardized validation functions and error message formats
to ensure consistency across all regime data splitting components.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
import logging


class ValidationErrorType(Enum):
    """Standardized validation error types."""
    VALIDATION_ERROR = "VALIDATION_ERROR"
    CONFIG_ERROR = "CONFIG_ERROR"
    CRITICAL_ERROR = "CRITICAL_ERROR"
    DATA_ERROR = "DATA_ERROR"
    FILE_ERROR = "FILE_ERROR"


@dataclass
class ValidationResult:
    """Standardized validation result."""
    valid: bool
    errors: List[str]
    warnings: List[str]
    details: Dict[str, Any]
    
    @classmethod
    def success(cls, details: Dict[str, Any] = None) -> 'ValidationResult':
        """Create a successful validation result."""
        return cls(
            valid=True,
            errors=[],
            warnings=[],
            details=details or {}
        )
    
    @classmethod
    def failure(cls, errors: List[str], warnings: List[str] = None, details: Dict[str, Any] = None) -> 'ValidationResult':
        """Create a failed validation result."""
        return cls(
            valid=False,
            errors=errors,
            warnings=warnings or [],
            details=details or {}
        )


class StandardizedValidator:
    """Standardized validator with consistent error messaging."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def create_error_message(self, error_type: ValidationErrorType, message: str, action: str) -> str:
        """Create standardized error message."""
        return f"{error_type.value}: {message}. Action required: {action}"
    
    def validate_not_none(self, value: Any, name: str, action: str = "Provide a valid value") -> ValidationResult:
        """Validate that a value is not None."""
        if value is None:
            error_msg = self.create_error_message(
                ValidationErrorType.VALIDATION_ERROR,
                f"{name} is None",
                action
            )
            return ValidationResult.failure([error_msg])
        return ValidationResult.success()
    
    def validate_dataframe(self, df: Any, name: str = "DataFrame", 
                          min_rows: int = 0, required_columns: List[str] = None) -> ValidationResult:
        """Validate DataFrame with standardized error messages."""
        errors = []
        warnings = []
        details = {}
        
        # Check if it's None
        if df is None:
            error_msg = self.create_error_message(
                ValidationErrorType.VALIDATION_ERROR,
                f"{name} is None",
                "Provide a valid pandas DataFrame"
            )
            return ValidationResult.failure([error_msg])
        
        # Check if it's a DataFrame
        if not isinstance(df, pd.DataFrame):
            error_msg = self.create_error_message(
                ValidationErrorType.VALIDATION_ERROR,
                f"{name} must be a pandas DataFrame, got {type(df)}",
                "Convert data to pandas DataFrame"
            )
            return ValidationResult.failure([error_msg])
        
        # Check if empty
        if df.empty:
            error_msg = self.create_error_message(
                ValidationErrorType.VALIDATION_ERROR,
                f"{name} is empty",
                "Provide non-empty data"
            )
            return ValidationResult.failure([error_msg])
        
        # Check minimum rows
        if len(df) < min_rows:
            error_msg = self.create_error_message(
                ValidationErrorType.VALIDATION_ERROR,
                f"{name} has {len(df)} rows, minimum required: {min_rows}",
                f"Provide at least {min_rows} rows of data"
            )
            errors.append(error_msg)
        
        # Check required columns
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                error_msg = self.create_error_message(
                    ValidationErrorType.VALIDATION_ERROR,
                    f"{name} missing required columns: {missing_cols}",
                    f"Ensure columns {missing_cols} are present in the data"
                )
                errors.append(error_msg)
        
        # Add details
        details.update({
            'rows': len(df),
            'columns': list(df.columns),
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        })
        
        # Check for data quality issues
        null_count = df.isnull().sum().sum()
        if null_count > 0:
            null_percentage = (null_count / (len(df) * len(df.columns))) * 100
            warnings.append(f"{name} contains {null_count} null values ({null_percentage:.2f}%)")
        
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            warnings.append(f"{name} contains {duplicate_count} duplicate rows")
        
        if errors:
            return ValidationResult.failure(errors, warnings, details)
        return ValidationResult.success(details)
    
    def validate_file_exists(self, file_path: Union[str, Path], name: str = "File") -> ValidationResult:
        """Validate that a file exists."""
        path = Path(file_path)
        
        if not path.exists():
            error_msg = self.create_error_message(
                ValidationErrorType.FILE_ERROR,
                f"{name} does not exist: {path}",
                f"Ensure the file exists at {path}"
            )
            return ValidationResult.failure([error_msg])
        
        if not path.is_file():
            error_msg = self.create_error_message(
                ValidationErrorType.FILE_ERROR,
                f"{name} is not a file: {path}",
                f"Ensure {path} is a valid file"
            )
            return ValidationResult.failure([error_msg])
        
        # Add file details
        stat = path.stat()
        details = {
            'path': str(path),
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / 1024 / 1024,
            'modified_timestamp': stat.st_mtime
        }
        
        return ValidationResult.success(details)
    
    def validate_config_parameters(self, config: Dict[str, Any], required_params: List[str]) -> ValidationResult:
        """Validate required configuration parameters."""
        errors = []
        
        if not isinstance(config, dict):
            error_msg = self.create_error_message(
                ValidationErrorType.CONFIG_ERROR,
                "Configuration must be a dictionary",
                "Provide configuration as a dictionary"
            )
            return ValidationResult.failure([error_msg])
        
        for param in required_params:
            if param not in config or config[param] is None or config[param] == "":
                error_msg = self.create_error_message(
                    ValidationErrorType.CONFIG_ERROR,
                    f"Missing required parameter '{param}'",
                    f"Set configuration parameter '{param}'"
                )
                errors.append(error_msg)
        
        if errors:
            return ValidationResult.failure(errors)
        
        return ValidationResult.success({'validated_params': required_params})
    
    def validate_regime_data(self, regime_states: np.ndarray, market_data: pd.DataFrame) -> ValidationResult:
        """Validate regime data consistency."""
        errors = []
        warnings = []
        details = {}
        
        # Check regime states
        if regime_states is None:
            error_msg = self.create_error_message(
                ValidationErrorType.DATA_ERROR,
                "Regime states are None",
                "Provide valid regime states array"
            )
            return ValidationResult.failure([error_msg])
        
        if not isinstance(regime_states, np.ndarray):
            regime_states = np.array(regime_states)
        
        if len(regime_states) == 0:
            error_msg = self.create_error_message(
                ValidationErrorType.DATA_ERROR,
                "Regime states array is empty",
                "Provide non-empty regime states"
            )
            return ValidationResult.failure([error_msg])
        
        # Check alignment with market data
        if len(regime_states) != len(market_data):
            error_msg = self.create_error_message(
                ValidationErrorType.DATA_ERROR,
                f"Regime states length ({len(regime_states)}) doesn't match market data length ({len(market_data)})",
                "Ensure regime states and market data have the same length"
            )
            errors.append(error_msg)
        
        # Check regime diversity
        unique_regimes = len(np.unique(regime_states))
        if unique_regimes < 2:
            warnings.append(f"Very few regimes detected: {unique_regimes}")
        elif unique_regimes > 20:
            warnings.append(f"Many regimes detected: {unique_regimes} - consider parameter tuning")
        
        # Check for reasonable regime values
        if len(regime_states) > 0:
            min_regime = np.min(regime_states)
            max_regime = np.max(regime_states)
            
            if min_regime < 0:
                warnings.append(f"Negative regime values detected: min={min_regime}")
            
            if max_regime > 100:
                warnings.append(f"Unusually high regime values detected: max={max_regime}")
        
        # Add details
        unique_regimes, counts = np.unique(regime_states, return_counts=True)
        details.update({
            'regime_count': unique_regimes,
            'regime_distribution': {int(k): int(v) for k, v in zip(unique_regimes, counts)},
            'min_regime': int(np.min(regime_states)) if len(regime_states) > 0 else None,
            'max_regime': int(np.max(regime_states)) if len(regime_states) > 0 else None
        })
        
        if errors:
            return ValidationResult.failure(errors, warnings, details)
        
        return ValidationResult.success(details)


# Global validator instance
_global_validator = StandardizedValidator()


def get_validator(logger: Optional[logging.Logger] = None) -> StandardizedValidator:
    """Get a validator instance."""
    if logger:
        return StandardizedValidator(logger)
    return _global_validator


def validate_training_input(training_input: Dict[str, Any]) -> ValidationResult:
    """Validate training input parameters."""
    validator = get_validator()
    
    # Check if training input is a dictionary
    if not isinstance(training_input, dict):
        error_msg = validator.create_error_message(
            ValidationErrorType.VALIDATION_ERROR,
            "Training input must be a dictionary",
            "Provide training_input as a dictionary"
        )
        return ValidationResult.failure([error_msg])
    
    # Check required parameters
    required_params = ['symbol', 'exchange', 'timeframe']
    return validator.validate_config_parameters(training_input, required_params)


def validate_pipeline_state(pipeline_state: Dict[str, Any]) -> ValidationResult:
    """Validate pipeline state."""
    validator = get_validator()
    
    if not isinstance(pipeline_state, dict):
        warning_msg = "Pipeline state is not a dictionary - this may cause issues"
        return ValidationResult.success({'warnings': [warning_msg]})
    
    return ValidationResult.success()


def create_standardized_error(error_type: ValidationErrorType, message: str, action: str) -> str:
    """Create a standardized error message."""
    return f"{error_type.value}: {message}. Action required: {action}"