"""
Fast failing data validation for feature lookback optimization.

This module provides strict validation that fails immediately on invalid data
rather than attempting recovery or fallback mechanisms.

Enhanced with comprehensive tprint logging and generalized validation patterns
for use across all pre-training steps.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import time
import traceback

# Import centralized tprint utilities
from .tprint_utils import (
    tprint, tprint_debug, tprint_info, tprint_warning, 
    tprint_error, tprint_success, tprint_performance,
    TPRINT_AVAILABLE
)

from ..error_handling.error_handler import DataValidationError, OptimizationError
from .nan_handling import SafeNaNHandler


class ValidationSeverity(Enum):
    """Validation error severity levels."""
    CRITICAL = "critical"  # Immediate failure
    HIGH = "high"         # Likely to cause issues
    MEDIUM = "medium"     # Warning but continue
    LOW = "low"          # Info only


@dataclass
class ValidationResult:
    """Result of data validation with fast failing support."""
    is_valid: bool
    severity: ValidationSeverity
    error_message: str
    details: Dict[str, Any]
    should_fail_fast: bool = True
    validation_time: float = 0.0
    validated_columns: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.validated_columns is None:
            self.validated_columns = []
        if self.warnings is None:
            self.warnings = []
    
    def add_warning(self, warning: str):
        """Add a warning to the validation result."""
        if self.warnings is None:
            self.warnings = []
        self.warnings.append(warning)
    
    def to_summary(self) -> str:
        """Generate a human-readable summary of the validation result."""
        status = "✅ VALID" if self.is_valid else "❌ INVALID"
        severity_icon = {
            ValidationSeverity.CRITICAL: "🚨",
            ValidationSeverity.HIGH: "⚠️",
            ValidationSeverity.MEDIUM: "⚡",
            ValidationSeverity.LOW: "ℹ️"
        }.get(self.severity, "❓")
        
        summary = f"{status} {severity_icon} {self.error_message}"
        if self.warnings:
            summary += f" | {len(self.warnings)} warnings"
        if self.validated_columns:
            summary += f" | {len(self.validated_columns)} columns validated"
        summary += f" | {self.validation_time:.3f}s"
        
        return summary


class FastFailingValidator:
    """Validator that fails fast on critical data issues with comprehensive logging."""
    
    def __init__(self, 
                 min_samples: int = 100, 
                 min_variance: float = 1e-8,
                 enable_logging: bool = True,
                 validation_context: str = "general"):
        self.min_samples = min_samples
        self.min_variance = min_variance
        self.nan_handler = SafeNaNHandler()
        self.enable_logging = enable_logging
        self.validation_context = validation_context
        self.validation_stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "total_time": 0.0
        }
    
    def _log_validation_start(self, operation: str, **kwargs):
        """Log the start of a validation operation."""
        if not self.enable_logging:
            return
        tprint_debug(f"🔍 [{self.validation_context}] Starting {operation}")
        for key, value in kwargs.items():
            tprint_debug(f"   → {key}: {value}")
    
    def _log_validation_success(self, operation: str, duration: float, **details):
        """Log successful validation completion."""
        if not self.enable_logging:
            return
        tprint_success(f"✅ [{self.validation_context}] {operation} completed in {duration:.3f}s")
        for key, value in details.items():
            tprint_debug(f"   → {key}: {value}")
    
    def _log_validation_failure(self, operation: str, error: str, duration: float):
        """Log validation failure."""
        if not self.enable_logging:
            return
        tprint_error(f"❌ [{self.validation_context}] {operation} failed in {duration:.3f}s: {error}")
    
    def _log_validation_warning(self, operation: str, warning: str):
        """Log validation warning."""
        if not self.enable_logging:
            return
        tprint_warning(f"⚠️ [{self.validation_context}] {operation}: {warning}")
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        stats = self.validation_stats.copy()
        if stats["total_validations"] > 0:
            stats["success_rate"] = stats["successful_validations"] / stats["total_validations"]
            stats["avg_time"] = stats["total_time"] / stats["total_validations"]
        else:
            stats["success_rate"] = 0.0
            stats["avg_time"] = 0.0
        return stats
    
    def validate_optimization_data(
        self, 
        data: pd.DataFrame, 
        feature_columns: List[str],
        target_columns: List[str],
        lookback_range: Tuple[int, int]
    ) -> ValidationResult:
        """
        Validate data for optimization with fast failing and comprehensive logging.
        
        Args:
            data: Input DataFrame
            feature_columns: List of feature column names
            target_columns: List of target column names
            lookback_range: (min_lookback, max_lookback) tuple
            
        Returns:
            ValidationResult with validation status
            
        Raises:
            DataValidationError: If validation fails critically
        """
        start_time = time.time()
        self.validation_stats["total_validations"] += 1
        
        self._log_validation_start(
            "optimization data validation",
            data_shape=data.shape,
            feature_count=len(feature_columns),
            target_count=len(target_columns),
            lookback_range=lookback_range
        )
        
        try:
            warnings = []
            validated_columns = []
            
            # Critical validations that must pass
            self._validate_dataframe_structure(data)
            self._validate_required_columns(data, feature_columns, target_columns)
            self._validate_data_length(data, lookback_range)
            
            # Feature validation with detailed logging
            for col in feature_columns:
                if col in data.columns:
                    validated_columns.append(col)
                    self._validate_single_feature_column(data, col, warnings)
            
            # Target validation with detailed logging
            for col in target_columns:
                if col in data.columns:
                    validated_columns.append(col)
                    self._validate_single_target_column(data, col, warnings)
            
            # High priority validations
            self._validate_data_quality(data, feature_columns, target_columns, warnings)
            
            duration = time.time() - start_time
            self.validation_stats["successful_validations"] += 1
            self.validation_stats["total_time"] += duration
            
            result = ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.LOW,
                error_message="Validation passed",
                details={
                    "validated_columns": len(validated_columns),
                    "data_shape": data.shape,
                    "feature_columns": feature_columns,
                    "target_columns": target_columns,
                    "lookback_range": lookback_range
                },
                should_fail_fast=False,
                validation_time=duration,
                validated_columns=validated_columns,
                warnings=warnings
            )
            
            self._log_validation_success(
                "optimization data validation",
                duration,
                validated_columns=len(validated_columns),
                warnings=len(warnings)
            )
            
            return result
            
        except DataValidationError as e:
            duration = time.time() - start_time
            self.validation_stats["failed_validations"] += 1
            self.validation_stats["total_time"] += duration
            
            result = ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                error_message=str(e),
                details={"error_type": "DataValidationError", "traceback": traceback.format_exc()},
                should_fail_fast=True,
                validation_time=duration
            )
            
            self._log_validation_failure("optimization data validation", str(e), duration)
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self.validation_stats["failed_validations"] += 1
            self.validation_stats["total_time"] += duration
            
            result = ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                error_message=f"Unexpected validation error: {str(e)}",
                details={"error_type": type(e).__name__, "traceback": traceback.format_exc()},
                should_fail_fast=True,
                validation_time=duration
            )
            
            self._log_validation_failure("optimization data validation", f"Unexpected error: {str(e)}", duration)
            return result
    
    def _validate_dataframe_structure(self, data: pd.DataFrame) -> None:
        """Validate basic DataFrame structure."""
        if not isinstance(data, pd.DataFrame):
            raise DataValidationError(f"Data must be a pandas DataFrame, got {type(data)}")
        
        if data.empty:
            raise DataValidationError("DataFrame is empty")
        
        if len(data) < self.min_samples:
            raise DataValidationError(
                f"Insufficient data: {len(data)} rows < {self.min_samples} required"
            )
    
    def _validate_required_columns(
        self, 
        data: pd.DataFrame, 
        feature_columns: List[str], 
        target_columns: List[str]
    ) -> None:
        """Validate that required columns exist."""
        missing_features = [col for col in feature_columns if col not in data.columns]
        if missing_features:
            raise DataValidationError(
                f"Missing feature columns: {missing_features}"
            )
        
        missing_targets = [col for col in target_columns if col not in data.columns]
        if missing_targets:
            raise DataValidationError(
                f"Missing target columns: {missing_targets}"
            )
    
    def _validate_data_length(self, data: pd.DataFrame, lookback_range: Tuple[int, int]) -> None:
        """Validate data length against lookback requirements."""
        min_lookback, max_lookback = lookback_range
        
        if len(data) < max_lookback + 50:  # Need buffer for validation
            raise DataValidationError(
                f"Insufficient data for lookback range: {len(data)} rows < {max_lookback + 50} required"
            )
    
    def _validate_single_feature_column(self, data: pd.DataFrame, col: str, warnings: List[str]) -> None:
        """Validate a single feature column with detailed logging."""
        if col not in data.columns:
            return
            
        series = data[col]
        
        # Check for all NaN values
        if series.isna().all():
            raise DataValidationError(f"Feature column '{col}' contains only NaN values")
        
        # Check for constant values
        valid_values = series.dropna()
        if len(valid_values) > 0:
            if valid_values.nunique() == 1:
                raise DataValidationError(f"Feature column '{col}' is constant (no variance)")
            
            # Check variance
            variance = valid_values.var()
            if variance < self.min_variance:
                raise DataValidationError(
                    f"Feature column '{col}' has insufficient variance: {variance:.2e} < {self.min_variance:.2e}"
                )
            
            # Check for high NaN rate (warning only)
            nan_rate = series.isna().sum() / len(series)
            if nan_rate > 0.3:  # More than 30% NaN
                warning = f"Feature column '{col}' has high NaN rate: {nan_rate:.1%}"
                warnings.append(warning)
                self._log_validation_warning("feature validation", warning)
    
    def _validate_feature_columns(self, data: pd.DataFrame, feature_columns: List[str]) -> None:
        """Validate feature columns for optimization suitability."""
        for col in feature_columns:
            if col not in data.columns:
                continue
            self._validate_single_feature_column(data, col, [])
    
    def _validate_single_target_column(self, data: pd.DataFrame, col: str, warnings: List[str]) -> None:
        """Validate a single target column with detailed logging."""
        if col not in data.columns:
            return
            
        series = data[col]
        
        # Check for all NaN values
        if series.isna().all():
            raise DataValidationError(f"Target column '{col}' contains only NaN values")
        
        # Check for sufficient valid values
        valid_values = series.dropna()
        if len(valid_values) < self.min_samples // 2:  # At least half the minimum samples
            raise DataValidationError(
                f"Target column '{col}' has insufficient valid values: {len(valid_values)} < {self.min_samples // 2}"
            )
        
        # Check for high NaN rate (warning only)
        nan_rate = series.isna().sum() / len(series)
        if nan_rate > 0.2:  # More than 20% NaN for targets
            warning = f"Target column '{col}' has high NaN rate: {nan_rate:.1%}"
            warnings.append(warning)
            self._log_validation_warning("target validation", warning)
    
    def _validate_target_columns(self, data: pd.DataFrame, target_columns: List[str]) -> None:
        """Validate target columns for optimization suitability."""
        for col in target_columns:
            if col not in data.columns:
                continue
            self._validate_single_target_column(data, col, [])
    
    def _validate_data_quality(
        self, 
        data: pd.DataFrame, 
        feature_columns: List[str], 
        target_columns: List[str],
        warnings: List[str] = None
    ) -> None:
        """Validate overall data quality with comprehensive checks."""
        if warnings is None:
            warnings = []
            
        all_columns = feature_columns + target_columns
        
        # Check for excessive NaN rates
        for col in all_columns:
            if col not in data.columns:
                continue
                
            nan_rate = data[col].isna().sum() / len(data)
            if nan_rate > 0.5:  # More than 50% NaN
                raise DataValidationError(
                    f"Column '{col}' has excessive NaN rate: {nan_rate:.1%} > 50%"
                )
            elif nan_rate > 0.3:  # Warning for high NaN rate
                warning = f"Column '{col}' has high NaN rate: {nan_rate:.1%}"
                warnings.append(warning)
                self._log_validation_warning("data quality", warning)
        
        # Check for infinite values
        for col in all_columns:
            if col not in data.columns:
                continue
                
            if np.isinf(data[col]).any():
                raise DataValidationError(f"Column '{col}' contains infinite values")
        
        # Check for duplicate columns
        duplicate_columns = data.columns[data.columns.duplicated()].tolist()
        if duplicate_columns:
            warning = f"Found duplicate columns: {duplicate_columns}"
            warnings.append(warning)
            self._log_validation_warning("data quality", warning)
        
        # Check for memory usage (warning only)
        memory_usage = data.memory_usage(deep=True).sum()
        if memory_usage > 1e9:  # More than 1GB
            warning = f"Large memory usage: {memory_usage / 1e9:.2f}GB"
            warnings.append(warning)
            self._log_validation_warning("data quality", warning)
    
    def validate_lookback_range(self, lookback_range: Tuple[int, int]) -> None:
        """Validate lookback range parameters."""
        min_lookback, max_lookback = lookback_range
        
        if min_lookback < 1:
            raise DataValidationError(f"Minimum lookback must be >= 1, got {min_lookback}")
        
        if max_lookback < min_lookback:
            raise DataValidationError(
                f"Maximum lookback ({max_lookback}) must be >= minimum lookback ({min_lookback})"
            )
        
        if max_lookback > 1000:  # Reasonable upper limit
            raise DataValidationError(f"Maximum lookback ({max_lookback}) exceeds reasonable limit (1000)")
    
    def validate_feature_target_alignment(
        self, 
        feature_values: np.ndarray, 
        target_values: np.ndarray,
        min_valid_samples: int = 20
    ) -> None:
        """Validate feature-target alignment with fast failing."""
        if len(feature_values) != len(target_values):
            raise DataValidationError(
                f"Feature-target length mismatch: {len(feature_values)} vs {len(target_values)}"
            )
        
        # Use safe NaN handling to check alignment
        try:
            alignment = self.nan_handler.align_arrays_safely(
                feature_values, target_values, min_valid_samples
            )
            
            if alignment.n_valid < min_valid_samples:
                raise DataValidationError(
                    f"Insufficient valid samples after NaN removal: {alignment.n_valid} < {min_valid_samples}"
                )
                
        except DataValidationError:
            raise  # Re-raise validation errors
        except Exception as e:
            raise DataValidationError(f"Feature-target alignment validation failed: {str(e)}")


def validate_optimization_inputs_fast_fail(
    data: pd.DataFrame,
    feature_columns: List[str],
    target_columns: List[str],
    lookback_range: Tuple[int, int],
    min_samples: int = 100,
    validation_context: str = "optimization"
) -> ValidationResult:
    """
    Fast-failing validation of optimization inputs with enhanced logging.
    
    This function will raise DataValidationError immediately on any critical issue
    rather than attempting recovery or returning warnings.
    
    Args:
        data: Input DataFrame
        feature_columns: List of feature column names
        target_columns: List of target column names
        lookback_range: (min_lookback, max_lookback) tuple
        min_samples: Minimum number of samples required
        validation_context: Context for logging (e.g., "optimization", "preprocessing")
        
    Returns:
        ValidationResult if validation passes
        
    Raises:
        DataValidationError: If any critical validation fails
    """
    validator = FastFailingValidator(
        min_samples=min_samples, 
        validation_context=validation_context
    )
    
    # Validate lookback range first (fastest check)
    validator.validate_lookback_range(lookback_range)
    
    # Validate data
    result = validator.validate_optimization_data(
        data, feature_columns, target_columns, lookback_range
    )
    
    if not result.is_valid and result.should_fail_fast:
        raise DataValidationError(result.error_message)
    
    return result


def validate_feature_calculation_inputs(
    data: pd.DataFrame,
    feature_name: str,
    lookback: int,
    validation_context: str = "feature_calculation"
) -> None:
    """
    Fast-failing validation for feature calculation inputs with enhanced logging.
    
    Args:
        data: Input DataFrame
        feature_name: Name of feature to calculate
        lookback: Lookback period
        validation_context: Context for logging
        
    Raises:
        DataValidationError: If inputs are invalid
    """
    start_time = time.time()
    
    if TPRINT_AVAILABLE:
        tprint_debug(f"🔍 [{validation_context}] Validating feature calculation inputs")
        tprint_debug(f"   → feature_name: {feature_name}")
        tprint_debug(f"   → lookback: {lookback}")
        tprint_debug(f"   → data_shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")
    
    if not isinstance(data, pd.DataFrame):
        raise DataValidationError(f"Data must be DataFrame, got {type(data)}")
    
    if data.empty:
        raise DataValidationError("DataFrame is empty")
    
    if lookback < 1:
        raise DataValidationError(f"Lookback must be >= 1, got {lookback}")
    
    if lookback >= len(data):
        raise DataValidationError(
            f"Lookback ({lookback}) must be < data length ({len(data)})"
        )
    
    if not isinstance(feature_name, str) or not feature_name.strip():
        raise DataValidationError(f"Feature name must be non-empty string, got '{feature_name}'")
    
    duration = time.time() - start_time
    if TPRINT_AVAILABLE:
        tprint_success(f"✅ [{validation_context}] Feature calculation inputs validated in {duration:.3f}s")


# ============================================================================
# GENERALIZED VALIDATION FUNCTIONS FOR PRE-TRAINING STEPS
# ============================================================================

def validate_dataframe_basic(
    data: pd.DataFrame,
    min_rows: int = 1,
    min_cols: int = 1,
    validation_context: str = "dataframe_basic"
) -> ValidationResult:
    """
    Basic DataFrame validation for general use across pre-training steps.
    
    Args:
        data: Input DataFrame
        min_rows: Minimum number of rows required
        min_cols: Minimum number of columns required
        validation_context: Context for logging
        
    Returns:
        ValidationResult with validation status
    """
    start_time = time.time()
    validator = FastFailingValidator(validation_context=validation_context)
    
    try:
        if not isinstance(data, pd.DataFrame):
            raise DataValidationError(f"Data must be a pandas DataFrame, got {type(data)}")
        
        if data.empty:
            raise DataValidationError("DataFrame is empty")
        
        if len(data) < min_rows:
            raise DataValidationError(f"Insufficient rows: {len(data)} < {min_rows}")
        
        if len(data.columns) < min_cols:
            raise DataValidationError(f"Insufficient columns: {len(data.columns)} < {min_cols}")
        
        duration = time.time() - start_time
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.LOW,
            error_message="Basic DataFrame validation passed",
            details={
                "data_shape": data.shape,
                "min_rows": min_rows,
                "min_cols": min_cols
            },
            should_fail_fast=False,
            validation_time=duration
        )
        
    except DataValidationError as e:
        duration = time.time() - start_time
        return ValidationResult(
            is_valid=False,
            severity=ValidationSeverity.CRITICAL,
            error_message=str(e),
            details={"error_type": "DataValidationError"},
            should_fail_fast=True,
            validation_time=duration
        )


# Removed unused function: validate_feature_data


# Removed unused function: validate_target_data


# Removed unused function: validate_preprocessing_inputs


# Removed unused function: validate_model_inputs