"""
Advanced Validation Framework for Unified Data-Driven Pipeline.

This module provides comprehensive validation infrastructure similar to
FeatureLookbackOptimizationComponent but adapted for the unified pipeline.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Import utility modules
from src.utils.common_operations import validate_dataframe_columns, safe_dataframe_operation
from src.utils.common_utilities import CommonUtilities
from src.utils.math_validation import validate_finite, validate_positive
from src.utils.serialization_utils import UniversalSerializer

try:
    from src.utils.tprint import tprint, tprint_error, tprint_warning, tprint_success, tprint_debug
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

import numpy as np
import pandas as pd


class ValidationLevel(Enum):
    """Validation severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ValidationStatus(Enum):
    """Validation result status."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class ValidationRule:
    """Individual validation rule definition."""
    name: str
    description: str
    level: ValidationLevel
    validator_func: callable
    required: bool = True
    auto_fix: bool = False
    fix_func: Optional[callable] = None


@dataclass
class ValidationResult:
    """Result of a validation check."""
    rule_name: str
    status: ValidationStatus
    level: ValidationLevel
    message: str
    details: Optional[Dict[str, Any]] = None
    auto_fixed: bool = False
    fix_applied: Optional[str] = None


@dataclass
class ValidationSummary:
    """Summary of validation results."""
    total_rules: int
    passed: int
    failed: int
    warnings: int
    skipped: int
    critical_failures: int
    overall_status: ValidationStatus
    quality_score: float
    recommendations: List[str]


class AdvancedInputValidator:
    """
    Advanced input validation framework for unified pipeline.
    
    Provides comprehensive validation for data quality, optimization parameters,
    and pipeline state with automatic fixing capabilities.
    """

    def __init__(self, logger=None):
        """Initialize the advanced input validator."""
        self.logger = logger or logging.getLogger(__name__)
        self.common_utils = CommonUtilities()
        self.serializer = UniversalSerializer()

        # Define validation rules
        self.validation_rules = self._initialize_validation_rules()

    def _initialize_validation_rules(self) -> List[ValidationRule]:
        """Initialize validation rules for the unified pipeline."""
        rules = [
            ValidationRule(
                name="dataframe_not_empty",
                description="DataFrame must not be empty",
                level=ValidationLevel.CRITICAL,
                validator_func=self._validate_dataframe_not_empty,
                required=True
            ),
            ValidationRule(
                name="required_columns_present",
                description="Required columns must be present",
                level=ValidationLevel.CRITICAL,
                validator_func=self._validate_required_columns,
                required=True
            ),
            ValidationRule(
                name="data_types_valid",
                description="Data types must be valid",
                level=ValidationLevel.HIGH,
                validator_func=self._validate_data_types,
                required=True
            ),
            ValidationRule(
                name="no_infinite_values",
                description="No infinite values allowed",
                level=ValidationLevel.HIGH,
                validator_func=self._validate_no_infinite_values,
                required=True,
                auto_fix=True,
                fix_func=self._fix_infinite_values
            ),
            ValidationRule(
                name="no_nan_values",
                description="No NaN values allowed",
                level=ValidationLevel.MEDIUM,
                validator_func=self._validate_no_nan_values,
                required=False,
                auto_fix=True,
                fix_func=self._fix_nan_values
            ),
            ValidationRule(
                name="sufficient_data_length",
                description="Sufficient data length for optimization",
                level=ValidationLevel.HIGH,
                validator_func=self._validate_sufficient_data_length,
                required=True
            ),
            ValidationRule(
                name="target_columns_present",
                description="Target columns must be present for supervised learning",
                level=ValidationLevel.MEDIUM,
                validator_func=self._validate_target_columns,
                required=False
            ),
            ValidationRule(
                name="feature_columns_valid",
                description="Feature columns must be valid",
                level=ValidationLevel.HIGH,
                validator_func=self._validate_feature_columns,
                required=True
            )
        ]
        
        return rules

    def validate_data(self, data: pd.DataFrame, 
                     required_columns: Optional[List[str]] = None,
                     lookback_range: Optional[Tuple[int, int]] = None,
                     target_columns: Optional[List[str]] = None) -> Tuple[bool, ValidationSummary, pd.DataFrame]:
        """
        Validate data comprehensively.
        
        Args:
            data: DataFrame to validate
            required_columns: List of required column names
            lookback_range: Tuple of (min_lookback, max_lookback)
            target_columns: List of target column names
            
        Returns:
            Tuple of (is_valid, validation_summary, cleaned_data)
        """
        tprint_debug("🔍 Starting comprehensive data validation")
        
        if data is None or not isinstance(data, pd.DataFrame):
            return False, self._create_failed_summary("Data is None or not a DataFrame"), pd.DataFrame()
        
        if data.empty:
            return False, self._create_failed_summary("DataFrame is empty"), data
        
        # Store original data for potential fixes
        cleaned_data = data.copy()
        validation_results = []
        
        # Apply validation rules
        for rule in self.validation_rules:
            try:
                result = self._apply_validation_rule(rule, cleaned_data, {
                    'required_columns': required_columns,
                    'lookback_range': lookback_range,
                    'target_columns': target_columns
                })
                validation_results.append(result)
                
                # Apply auto-fix if available and needed
                if result.status == ValidationStatus.FAILED and rule.auto_fix and rule.fix_func:
                    try:
                        cleaned_data = rule.fix_func(cleaned_data, result.details or {})
                        result.auto_fixed = True
                        result.status = ValidationStatus.PASSED
                        result.fix_applied = f"Auto-fixed using {rule.name}"
                        tprint_debug(f"✅ Auto-fixed validation issue: {rule.name}")
                    except Exception as fix_error:
                        tprint_warning(f"⚠️ Auto-fix failed for {rule.name}: {fix_error}")
                        
            except Exception as e:
                tprint_error(f"❌ Validation rule {rule.name} failed: {e}")
                validation_results.append(ValidationResult(
                    rule_name=rule.name,
                    status=ValidationStatus.FAILED,
                    level=rule.level,
                    message=f"Validation rule failed: {str(e)}"
                ))
        
        # Create validation summary
        summary = self._create_validation_summary(validation_results)
        
        # Determine overall validity
        is_valid = summary.overall_status in [ValidationStatus.PASSED, ValidationStatus.WARNING]
        
        if is_valid:
            tprint_success(f"✅ Data validation passed (quality score: {summary.quality_score:.2f})")
        else:
            tprint_error(f"❌ Data validation failed: {summary.recommendations}")
        
        return is_valid, summary, cleaned_data

    def _apply_validation_rule(self, rule: ValidationRule, data: pd.DataFrame, 
                              context: Dict[str, Any]) -> ValidationResult:
        """Apply a single validation rule."""
        try:
            is_valid, message, details = rule.validator_func(data, context)
            
            if is_valid:
                status = ValidationStatus.PASSED
            elif rule.required:
                status = ValidationStatus.FAILED
            else:
                status = ValidationStatus.WARNING
                
            return ValidationResult(
                rule_name=rule.name,
                status=status,
                level=rule.level,
                message=message,
                details=details
            )
            
        except Exception as e:
            return ValidationResult(
                rule_name=rule.name,
                status=ValidationStatus.FAILED,
                level=rule.level,
                message=f"Validation error: {str(e)}"
            )

    def _validate_dataframe_not_empty(self, data: pd.DataFrame, context: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate that DataFrame is not empty."""
        if data.empty:
            return False, "DataFrame is empty", {"rows": 0, "columns": 0}
        return True, "DataFrame is not empty", {"rows": len(data), "columns": len(data.columns)}

    def _validate_required_columns(self, data: pd.DataFrame, context: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate that required columns are present."""
        required_columns = context.get('required_columns', ['open', 'high', 'low', 'close', 'volume'])
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}", {"missing_columns": missing_columns}
        
        return True, "All required columns present", {"required_columns": required_columns}

    def _validate_data_types(self, data: pd.DataFrame, context: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate data types are appropriate."""
        invalid_types = []
        for col in data.columns:
            if data[col].dtype == 'object':
                # Check if object column can be converted to numeric
                try:
                    pd.to_numeric(data[col], errors='raise')
                except (ValueError, TypeError):
                    invalid_types.append(col)
        
        if invalid_types:
            return False, f"Invalid data types in columns: {invalid_types}", {"invalid_columns": invalid_types}
        
        return True, "All data types are valid", {"data_types": {col: str(dtype) for col, dtype in data.dtypes.items()}}

    def _validate_no_infinite_values(self, data: pd.DataFrame, context: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate no infinite values."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        infinite_cols = []
        
        for col in numeric_cols:
            if np.isinf(data[col]).any():
                infinite_cols.append(col)
        
        if infinite_cols:
            return False, f"Infinite values found in columns: {infinite_cols}", {"infinite_columns": infinite_cols}
        
        return True, "No infinite values found", {"checked_columns": list(numeric_cols)}

    def _validate_no_nan_values(self, data: pd.DataFrame, context: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate no NaN values."""
        nan_cols = data.columns[data.isnull().any()].tolist()
        
        if nan_cols:
            return False, f"NaN values found in columns: {nan_cols}", {"nan_columns": nan_cols}
        
        return True, "No NaN values found", {"checked_columns": list(data.columns)}

    def _validate_sufficient_data_length(self, data: pd.DataFrame, context: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate sufficient data length for optimization."""
        lookback_range = context.get('lookback_range', (5, 100))
        min_lookback, max_lookback = lookback_range
        
        if len(data) < max_lookback:
            return False, f"Insufficient data: {len(data)} rows < {max_lookback} required", {
                "data_length": len(data),
                "required_length": max_lookback
            }
        
        return True, f"Sufficient data length: {len(data)} rows", {
            "data_length": len(data),
            "min_required": max_lookback
        }

    def _validate_target_columns(self, data: pd.DataFrame, context: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate target columns are present."""
        target_columns = context.get('target_columns', [])
        if not target_columns:
            return True, "No target columns required", {}
        
        missing_targets = [col for col in target_columns if col not in data.columns]
        if missing_targets:
            return False, f"Missing target columns: {missing_targets}", {"missing_targets": missing_targets}
        
        return True, "All target columns present", {"target_columns": target_columns}

    def _validate_feature_columns(self, data: pd.DataFrame, context: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate feature columns are valid."""
        feature_cols = [col for col in data.columns if not any(pattern in col.lower() for pattern in ['target', 'label', 'confidence'])]
        
        if len(feature_cols) == 0:
            return False, "No feature columns found", {"feature_count": 0}
        
        return True, f"Found {len(feature_cols)} feature columns", {"feature_count": len(feature_cols)}

    def _fix_infinite_values(self, data: pd.DataFrame, details: Dict[str, Any]) -> pd.DataFrame:
        """Fix infinite values by replacing with NaN."""
        infinite_cols = details.get('infinite_columns', [])
        fixed_data = data.copy()
        
        for col in infinite_cols:
            fixed_data[col] = fixed_data[col].replace([np.inf, -np.inf], np.nan)
        
        return fixed_data

    def _fix_nan_values(self, data: pd.DataFrame, details: Dict[str, Any]) -> pd.DataFrame:
        """Fix NaN values by forward filling."""
        nan_cols = details.get('nan_columns', [])
        fixed_data = data.copy()
        
        for col in nan_cols:
            fixed_data[col] = fixed_data[col].fillna(method='ffill').fillna(method='bfill')
        
        return fixed_data

    def _create_validation_summary(self, results: List[ValidationResult]) -> ValidationSummary:
        """Create validation summary from results."""
        total_rules = len(results)
        passed = sum(1 for r in results if r.status == ValidationStatus.PASSED)
        failed = sum(1 for r in results if r.status == ValidationStatus.FAILED)
        warnings = sum(1 for r in results if r.status == ValidationStatus.WARNING)
        skipped = sum(1 for r in results if r.status == ValidationStatus.SKIPPED)
        critical_failures = sum(1 for r in results if r.status == ValidationStatus.FAILED and r.level == ValidationLevel.CRITICAL)
        
        # Determine overall status
        if critical_failures > 0:
            overall_status = ValidationStatus.FAILED
        elif failed > 0:
            overall_status = ValidationStatus.WARNING
        else:
            overall_status = ValidationStatus.PASSED
        
        # Calculate quality score
        quality_score = (passed + warnings * 0.5) / total_rules if total_rules > 0 else 0.0
        
        # Generate recommendations
        recommendations = []
        for result in results:
            if result.status == ValidationStatus.FAILED:
                recommendations.append(f"Fix {result.rule_name}: {result.message}")
            elif result.status == ValidationStatus.WARNING:
                recommendations.append(f"Consider fixing {result.rule_name}: {result.message}")
        
        return ValidationSummary(
            total_rules=total_rules,
            passed=passed,
            failed=failed,
            warnings=warnings,
            skipped=skipped,
            critical_failures=critical_failures,
            overall_status=overall_status,
            quality_score=quality_score,
            recommendations=recommendations
        )

    def _create_failed_summary(self, message: str) -> ValidationSummary:
        """Create a failed validation summary."""
        return ValidationSummary(
            total_rules=1,
            passed=0,
            failed=1,
            warnings=0,
            skipped=0,
            critical_failures=1,
            overall_status=ValidationStatus.FAILED,
            quality_score=0.0,
            recommendations=[message]
        )

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            'total_validations': len(self.validation_rules),
            'validation_rules': [rule.name for rule in self.validation_rules]
        }