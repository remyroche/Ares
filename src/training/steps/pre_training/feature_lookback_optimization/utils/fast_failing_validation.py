"""
Fast failing data validation for feature lookback optimization.

This module provides strict validation that fails immediately on invalid data
rather than attempting recovery or fallback mechanisms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum

from .error_handling import DataValidationError, OptimizationError
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


class FastFailingValidator:
    """Validator that fails fast on critical data issues."""
    
    def __init__(self, min_samples: int = 100, min_variance: float = 1e-8):
        self.min_samples = min_samples
        self.min_variance = min_variance
        self.nan_handler = SafeNaNHandler()
    
    def validate_optimization_data(
        self, 
        data: pd.DataFrame, 
        feature_columns: List[str],
        target_columns: List[str],
        lookback_range: Tuple[int, int]
    ) -> ValidationResult:
        """
        Validate data for optimization with fast failing.
        
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
        try:
            # Critical validations that must pass
            self._validate_dataframe_structure(data)
            self._validate_required_columns(data, feature_columns, target_columns)
            self._validate_data_length(data, lookback_range)
            self._validate_feature_columns(data, feature_columns)
            self._validate_target_columns(data, target_columns)
            
            # High priority validations
            self._validate_data_quality(data, feature_columns, target_columns)
            
            return ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.LOW,
                error_message="Validation passed",
                details={"validated_columns": len(feature_columns + target_columns)},
                should_fail_fast=False
            )
            
        except DataValidationError as e:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                error_message=str(e),
                details={"error_type": "DataValidationError"},
                should_fail_fast=True
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                error_message=f"Unexpected validation error: {str(e)}",
                details={"error_type": type(e).__name__},
                should_fail_fast=True
            )
    
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
    
    def _validate_feature_columns(self, data: pd.DataFrame, feature_columns: List[str]) -> None:
        """Validate feature columns for optimization suitability."""
        for col in feature_columns:
            if col not in data.columns:
                continue
                
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
    
    def _validate_target_columns(self, data: pd.DataFrame, target_columns: List[str]) -> None:
        """Validate target columns for optimization suitability."""
        for col in target_columns:
            if col not in data.columns:
                continue
                
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
    
    def _validate_data_quality(
        self, 
        data: pd.DataFrame, 
        feature_columns: List[str], 
        target_columns: List[str]
    ) -> None:
        """Validate overall data quality."""
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
        
        # Check for infinite values
        for col in all_columns:
            if col not in data.columns:
                continue
                
            if np.isinf(data[col]).any():
                raise DataValidationError(f"Column '{col}' contains infinite values")
    
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
    min_samples: int = 100
) -> ValidationResult:
    """
    Fast-failing validation of optimization inputs.
    
    This function will raise DataValidationError immediately on any critical issue
    rather than attempting recovery or returning warnings.
    
    Args:
        data: Input DataFrame
        feature_columns: List of feature column names
        target_columns: List of target column names
        lookback_range: (min_lookback, max_lookback) tuple
        min_samples: Minimum number of samples required
        
    Returns:
        ValidationResult if validation passes
        
    Raises:
        DataValidationError: If any critical validation fails
    """
    validator = FastFailingValidator(min_samples=min_samples)
    
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
    lookback: int
) -> None:
    """
    Fast-failing validation for feature calculation inputs.
    
    Args:
        data: Input DataFrame
        feature_name: Name of feature to calculate
        lookback: Lookback period
        
    Raises:
        DataValidationError: If inputs are invalid
    """
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