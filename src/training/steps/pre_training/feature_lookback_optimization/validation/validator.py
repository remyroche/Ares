"""
Input Validation Framework for Feature Lookback Optimization.

This module provides comprehensive validation for data quality,
optimization parameters, and pipeline state validation.
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
from src.utils.tprint import tprint, tprint_error, tprint_warning, tprint_success, tprint_debug

from ..constants import VALIDATION_CONSTANTS, QUALITY_CONSTANTS
from ..dependency_manager import get_dependency

# Get dependencies with fallbacks
np, _ = get_dependency('numpy')
pd, _ = get_dependency('pandas')


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


class InputValidator:
    """
    Comprehensive input validation framework.

    Provides structured validation for data quality, optimization parameters,
    and pipeline state with automatic fixing capabilities.
    """

    def __init__(self, logger=None):
        """Initialize the input validator."""
        self.logger = logger or logging.getLogger(__name__)
        self.common_utils = CommonUtilities()
        self.serializer = UniversalSerializer()

        # Define validation rules
        self.validation_rules = self._initialize_validation_rules()

    def _initialize_validation_rules(self) -> List[ValidationRule]:
        """Initialize all validation rules."""
        return [
            ValidationRule(
                name="dataframe_type",
                description="Input must be a pandas DataFrame",
                level=ValidationLevel.CRITICAL,
                validator_func=self._validate_dataframe_type,
                required=True,
                auto_fix=False
            ),
            ValidationRule(
                name="required_columns",
                description="DataFrame must contain required columns",
                level=ValidationLevel.CRITICAL,
                validator_func=self._validate_required_columns,
                required=True,
                auto_fix=False
            ),
            ValidationRule(
                name="data_completeness",
                description="Data must have sufficient completeness",
                level=ValidationLevel.HIGH,
                validator_func=self._validate_data_completeness,
                required=True,
                auto_fix=True,
                fix_func=self._fix_data_completeness
            ),
            ValidationRule(
                name="data_quality",
                description="Data quality must meet minimum standards",
                level=ValidationLevel.HIGH,
                validator_func=self._validate_data_quality,
                required=True,
                auto_fix=True,
                fix_func=self._fix_data_quality
            ),
            ValidationRule(
                name="finite_values",
                description="All numeric values must be finite",
                level=ValidationLevel.CRITICAL,
                validator_func=self._validate_finite_values,
                required=True,
                auto_fix=True,
                fix_func=self._fix_finite_values
            ),
            ValidationRule(
                name="lookback_range",
                description="Lookback range must be valid",
                level=ValidationLevel.MEDIUM,
                validator_func=self._validate_lookback_range,
                required=False,
                auto_fix=True,
                fix_func=self._fix_lookback_range
            )
        ]

    def validate_data(
        self,
        data: Any,
        required_columns: List[str] = None,
        lookback_range: Tuple[int, int] = None
    ) -> Tuple[bool, ValidationSummary, pd.DataFrame]:
        """
        Validate input data comprehensively.

        Args:
            data: Input data to validate
            required_columns: List of required column names
            lookback_range: Tuple of (min_lookback, max_lookback)

        Returns:
            Tuple of (is_valid, validation_summary, cleaned_data)
        """
        try:
            results = []
            cleaned_data = data

            # Run all validation rules
            for rule in self.validation_rules:
                try:
                    result = self._run_validation_rule(rule, data, required_columns, lookback_range)

                    # Apply auto-fix if available and needed
                    if result.status != ValidationStatus.PASSED and rule.auto_fix and rule.fix_func:
                        try:
                            cleaned_data, fix_result = rule.fix_func(cleaned_data, result.details)
                            if fix_result:
                                result.auto_fixed = True
                                result.fix_applied = rule.name
                                result.status = ValidationStatus.WARNING
                        except Exception as e:
                            self.logger.warning(f"Auto-fix failed for rule {rule.name}: {e}")

                    results.append(result)

                except Exception as e:
                    self.logger.error(f"Validation rule {rule.name} failed: {e}")
                    results.append(ValidationResult(
                        rule_name=rule.name,
                        status=ValidationStatus.FAILED,
                        level=rule.level,
                        message=f"Validation rule execution failed: {e}"
                    ))

            # Create validation summary
            summary = self._create_validation_summary(results)

            # Determine overall validity
            is_valid = (
                summary.critical_failures == 0 and
                (summary.failed == 0 or not any(r.required for r in results if r.status == ValidationStatus.FAILED))
            )

            return is_valid, summary, cleaned_data

        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            return False, self._create_failed_summary(), data

    def _run_validation_rule(
        self,
        rule: ValidationRule,
        data: Any,
        required_columns: List[str],
        lookback_range: Tuple[int, int]
    ) -> ValidationResult:
        """Run a single validation rule."""
        try:
            result = rule.validator_func(data, required_columns, lookback_range)

            return ValidationResult(
                rule_name=rule.name,
                status=ValidationStatus.PASSED if result[0] else ValidationStatus.FAILED,
                level=rule.level,
                message=result[1],
                details=result[2] if len(result) > 2 else None
            )

        except Exception as e:
            return ValidationResult(
                rule_name=rule.name,
                status=ValidationStatus.FAILED,
                level=rule.level,
                message=f"Validation rule execution failed: {e}"
            )

    def _validate_dataframe_type(self, data: Any, *args) -> Tuple[bool, str, Optional[Dict]]:
        """Validate that input is a pandas DataFrame."""
        if not isinstance(data, pd.DataFrame):
            return False, "Input must be a pandas DataFrame", {"input_type": type(data).__name__}
        return True, "Input is a valid DataFrame", {"shape": data.shape}

    def _validate_required_columns(self, data: pd.DataFrame, required_columns: List[str], *args) -> Tuple[bool, str, Optional[Dict]]:
        """Validate that required columns exist."""
        if required_columns is None:
            required_columns = VALIDATION_CONSTANTS.REQUIRED_OHLCV_COLUMNS

        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}", {"missing_columns": missing_columns}

        return True, f"All required columns present: {required_columns}", {"available_columns": list(data.columns)}

    def _validate_data_completeness(self, data: pd.DataFrame, *args) -> Tuple[bool, str, Optional[Dict]]:
        """Validate data completeness."""
        null_counts = data.isnull().sum()
        total_cells = data.shape[0] * data.shape[1]
        null_ratio = null_counts.sum() / total_cells

        if null_ratio > VALIDATION_CONSTANTS.MAX_NULL_RATIO:
            return False, f"Too many null values: {null_ratio:.2%}", {
                "null_ratio": null_ratio,
                "null_counts": null_counts.to_dict()
            }

        return True, f"Data completeness acceptable: {null_ratio:.2%} nulls", {"null_ratio": null_ratio}

    def _validate_data_quality(self, data: pd.DataFrame, *args) -> Tuple[bool, str, Optional[Dict]]:
        """Validate overall data quality."""
        # Simple quality check based on data statistics
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        quality_score = 0.0

        if len(numeric_columns) > 0:
            # Check for reasonable variance in numeric columns
            variances = data[numeric_columns].var()
            zero_variance = (variances == 0).sum()
            quality_score = 1.0 - (zero_variance / len(numeric_columns))

        if quality_score < VALIDATION_CONSTANTS.MIN_DATA_QUALITY_SCORE:
            return False, f"Low data quality score: {quality_score:.2f}", {"quality_score": quality_score}

        return True, f"Data quality acceptable: {quality_score:.2f}", {"quality_score": quality_score}

    def _validate_finite_values(self, data: pd.DataFrame, *args) -> Tuple[bool, str, Optional[Dict]]:
        """Validate that all numeric values are finite."""
        numeric_data = data.select_dtypes(include=[np.number])
        finite_mask = np.isfinite(numeric_data)

        if not finite_mask.all().all():
            non_finite = (~finite_mask).sum().sum()
            return False, f"Found {non_finite} non-finite values", {"non_finite_count": int(non_finite)}

        return True, "All numeric values are finite", None

    def _validate_lookback_range(self, data: pd.DataFrame, *args, lookback_range: Tuple[int, int] = None) -> Tuple[bool, str, Optional[Dict]]:
        """Validate lookback range parameters."""
        if lookback_range is None:
            return True, "No lookback range specified", None

        min_lookback, max_lookback = lookback_range

        if not (isinstance(min_lookback, int) and isinstance(max_lookback, int)):
            return False, "Lookback values must be integers", {"min_lookback": min_lookback, "max_lookback": max_lookback}

        if min_lookback <= 0:
            return False, f"Minimum lookback must be positive: {min_lookback}", {"min_lookback": min_lookback}

        if max_lookback <= min_lookback:
            return False, f"Maximum lookback must be greater than minimum: {max_lookback} <= {min_lookback}", {
                "min_lookback": min_lookback, "max_lookback": max_lookback
            }

        if max_lookback > len(data):
            return False, f"Maximum lookback exceeds data length: {max_lookback} > {len(data)}", {
                "max_lookback": max_lookback, "data_length": len(data)
            }

        return True, f"Lookback range valid: {min_lookback}-{max_lookback}", {"lookback_range": lookback_range}

    def _fix_data_completeness(self, data: pd.DataFrame, details: Dict) -> Tuple[pd.DataFrame, bool]:
        """Fix data completeness issues."""
        try:
            # Simple forward fill for missing values
            fixed_data = data.ffill().bfill()
            return fixed_data, True
        except Exception as e:
            self.logger.error(f"Failed to fix data completeness: {e}")
            return data, False

    def _fix_data_quality(self, data: pd.DataFrame, details: Dict) -> Tuple[pd.DataFrame, bool]:
        """Fix data quality issues."""
        try:
            # Remove columns with zero variance
            numeric_columns = data.select_dtypes(include=[np.number])
            variances = numeric_columns.var()
            zero_variance_cols = variances[variances == 0].index.tolist()

            if zero_variance_cols:
                data = data.drop(columns=zero_variance_cols)
                self.logger.info(f"Removed columns with zero variance: {zero_variance_cols}")

            return data, True
        except Exception as e:
            self.logger.error(f"Failed to fix data quality: {e}")
            return data, False

    def _fix_finite_values(self, data: pd.DataFrame, details: Dict) -> Tuple[pd.DataFrame, bool]:
        """Fix non-finite values."""
        try:
            # Replace non-finite values with zeros
            numeric_data = data.select_dtypes(include=[np.number])
            finite_mask = np.isfinite(numeric_data)
            numeric_data[~finite_mask] = 0.0
            data[numeric_data.columns] = numeric_data

            return data, True
        except Exception as e:
            self.logger.error(f"Failed to fix finite values: {e}")
            return data, False

    def _fix_lookback_range(self, data: pd.DataFrame, details: Dict, lookback_range: Tuple[int, int] = None) -> Tuple[pd.DataFrame, bool]:
        """Fix lookback range issues."""
        if lookback_range is None:
            return data, False

        min_lookback, max_lookback = lookback_range
        data_length = len(data)

        # Clamp values to valid range
        min_lookback = max(1, min(min_lookback, data_length))
        max_lookback = min(data_length, max(min_lookback, max_lookback))

        return data, True

    def _create_validation_summary(self, results: List[ValidationResult]) -> ValidationSummary:
        """Create a summary of validation results."""
        total_rules = len(results)
        passed = sum(1 for r in results if r.status == ValidationStatus.PASSED)
        failed = sum(1 for r in results if r.status == ValidationStatus.FAILED)
        warnings = sum(1 for r in results if r.status == ValidationStatus.WARNING)
        skipped = sum(1 for r in results if r.status == ValidationStatus.SKIPPED)
        critical_failures = sum(1 for r in results if r.level == ValidationLevel.CRITICAL and r.status == ValidationStatus.FAILED)

        # Calculate quality score
        if total_rules > 0:
            quality_score = (passed + warnings * 0.5) / total_rules
        else:
            quality_score = 0.0

        # Determine overall status
        if critical_failures > 0:
            overall_status = ValidationStatus.FAILED
        elif failed > 0:
            overall_status = ValidationStatus.WARNING
        else:
            overall_status = ValidationStatus.PASSED

        # Generate recommendations
        recommendations = []
        for result in results:
            if result.status != ValidationStatus.PASSED:
                if result.level == ValidationLevel.CRITICAL:
                    recommendations.append(f"CRITICAL: {result.message}")
                elif result.level in [ValidationLevel.HIGH, ValidationLevel.MEDIUM]:
                    recommendations.append(f"Consider: {result.message}")

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

    def _create_failed_summary(self) -> ValidationSummary:
        """Create a failed validation summary."""
        return ValidationSummary(
            total_rules=0,
            passed=0,
            failed=1,
            warnings=0,
            skipped=0,
            critical_failures=1,
            overall_status=ValidationStatus.FAILED,
            quality_score=0.0,
            recommendations=["Validation framework failed to execute"]
        )
