"""
Advanced Validation Framework - Cleaned Version

This module provides consolidated validation infrastructure with fast-fail patterns,
removed duplicates, and improved validation reporting.

Key improvements:
- Consolidated validation classes (single source of truth)
- Removed duplicate validation methods
- Implemented fast-fail patterns instead of silent errors
- Improved validation reporting and error messages
- Streamlined validation rules and checks
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Import utility modules
from src.utils.common_utilities import (
    CommonUtilities, safe_dataframe_operation, validate_dataframe_columns,
    analyze_nan_values_detailed, format_nan_analysis_report, 
    calculate_data_quality_metrics, create_data_quality_report,
    safe_convert_dtypes, safe_merge_dataframes, safe_drop_columns,
    safe_rename_columns, get_dataframe_info, safe_filter_dataframe
)
from src.utils.serialization_utils import UniversalSerializer

# Centralized tprint import
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

# Import features_common validation utilities
try:
    from src.features_common import (
        ValidationMixin, MonitoringMixin, PerformanceMixin,
        validate_input_data, safe_execute, validate_configuration,
        check_system_health, get_logger, log_operation,
        ValidationError, ConfigurationError, SilentFailureError
    )
    FEATURES_COMMON_VALIDATION_AVAILABLE = True
except ImportError:
    FEATURES_COMMON_VALIDATION_AVAILABLE = False

# Import feature_generation validation utilities
try:
    from src.feature_generation.utils import (
        validate_feature_quality, validate_features_dataframe,
        feature_validation_decorator
    )
    FEATURE_GENERATION_VALIDATION_AVAILABLE = True
except ImportError:
    FEATURE_GENERATION_VALIDATION_AVAILABLE = False

import numpy as np
import pandas as pd

# Centralized enums - single source of truth
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

# Data classes
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
    recommendations: List[str]
    quality_score: float
    is_valid: bool

class AdvancedValidator:
    """Advanced validation framework with fast-fail patterns."""
    
    def __init__(self, component_name: str = "AdvancedValidator", logger: Optional[logging.Logger] = None):
        self.component_name = component_name
        self.logger = logger or logging.getLogger(f"{__name__}.{component_name}")
        self.validation_rules = []
        self.validation_history = []
        self.performance_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'critical_failures': 0
        }
        
        # Initialize validation rules
        self._initialize_validation_rules()
    
    def _initialize_validation_rules(self):
        """Initialize validation rules for different data types."""
        self.validation_rules = [
            ValidationRule(
                name="dataframe_not_empty",
                description="DataFrame must not be empty",
                level=ValidationLevel.CRITICAL,
                validator_func=self._validate_dataframe_not_empty,
                required=True
            ),
            ValidationRule(
                name="required_columns",
                description="DataFrame must have required columns",
                level=ValidationLevel.CRITICAL,
                validator_func=self._validate_required_columns,
                required=True
            ),
            ValidationRule(
                name="data_types",
                description="DataFrame columns must have correct data types",
                level=ValidationLevel.HIGH,
                validator_func=self._validate_data_types,
                required=True
            ),
            ValidationRule(
                name="no_infinite_values",
                description="DataFrame must not contain infinite values",
                level=ValidationLevel.HIGH,
                validator_func=self._validate_no_infinite_values,
                required=True
            ),
            ValidationRule(
                name="no_nan_values",
                description="DataFrame must not contain excessive NaN values",
                level=ValidationLevel.MEDIUM,
                validator_func=self._validate_no_nan_values,
                required=False
            ),
            ValidationRule(
                name="sufficient_data_length",
                description="DataFrame must have sufficient data length",
                level=ValidationLevel.MEDIUM,
                validator_func=self._validate_sufficient_data_length,
                required=True
            ),
            ValidationRule(
                name="target_columns",
                description="Target columns must be valid",
                level=ValidationLevel.HIGH,
                validator_func=self._validate_target_columns,
                required=False
            ),
            ValidationRule(
                name="feature_columns",
                description="Feature columns must be valid",
                level=ValidationLevel.MEDIUM,
                validator_func=self._validate_feature_columns,
                required=False
            )
        ]
    
    def validate_data(self, data: pd.DataFrame, 
                     required_columns: List[str] = None,
                     target_columns: List[str] = None,
                     feature_columns: List[str] = None,
                     validation_level: ValidationLevel = ValidationLevel.STANDARD) -> Tuple[bool, ValidationSummary, pd.DataFrame]:
        """
        Validate data with fast-fail patterns.
        
        Args:
            data: DataFrame to validate
            required_columns: List of required columns
            target_columns: List of target columns
            feature_columns: List of feature columns
            validation_level: Level of validation to perform
            
        Returns:
            Tuple of (is_valid, validation_summary, cleaned_data)
            
        Raises:
            ValueError: If critical validation fails
        """
        if data is None:
            raise ValueError("Data cannot be None")
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        
        # Initialize validation context
        context = {
            'data': data,
            'required_columns': required_columns or [],
            'target_columns': target_columns or [],
            'feature_columns': feature_columns or [],
            'validation_level': validation_level
        }
        
        # Run validation rules
        results = []
        cleaned_data = data.copy()
        
        for rule in self.validation_rules:
            try:
                result = self._run_validation_rule(rule, context)
                results.append(result)
                
                # Fast fail for critical failures
                if result.status == ValidationStatus.FAILED and rule.level == ValidationLevel.CRITICAL:
                    raise ValueError(f"Critical validation failed: {result.message}")
                
                # Apply auto-fix if available
                if result.status == ValidationStatus.FAILED and rule.auto_fix and rule.fix_func:
                    try:
                        cleaned_data = rule.fix_func(cleaned_data, context)
                        result.auto_fixed = True
                        result.fix_applied = f"Applied {rule.name} fix"
                        result.status = ValidationStatus.PASSED
                    except Exception as fix_error:
                        self.logger.warning(f"Auto-fix failed for {rule.name}: {fix_error}")
                
            except Exception as e:
                # Fast fail for critical errors
                if rule.level == ValidationLevel.CRITICAL:
                    raise ValueError(f"Critical validation error in {rule.name}: {e}") from e
                
                # Log non-critical errors
                self.logger.warning(f"Validation rule {rule.name} failed: {e}")
                results.append(ValidationResult(
                    rule_name=rule.name,
                    status=ValidationStatus.FAILED,
                    level=rule.level,
                    message=f"Validation error: {e}",
                    details={'error': str(e)}
                ))
        
        # Create validation summary
        summary = self._create_validation_summary(results)
        
        # Update performance stats
        self.performance_stats['total_validations'] += 1
        if summary.is_valid:
            self.performance_stats['successful_validations'] += 1
        else:
            self.performance_stats['failed_validations'] += 1
            if summary.critical_failures > 0:
                self.performance_stats['critical_failures'] += 1
        
        # Store validation history
        self.validation_history.append({
            'timestamp': pd.Timestamp.now(),
            'summary': summary,
            'results': results
        })
        
        return summary.is_valid, summary, cleaned_data
    
    def _run_validation_rule(self, rule: ValidationRule, context: Dict[str, Any]) -> ValidationResult:
        """Run a single validation rule."""
        try:
            is_valid, message, details = rule.validator_func(context['data'], context)
            
            if is_valid:
                status = ValidationStatus.PASSED
            elif rule.level == ValidationLevel.CRITICAL:
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
                message=f"Validation error: {e}",
                details={'error': str(e)}
            )
    
    def _validate_dataframe_not_empty(self, data: pd.DataFrame, context: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate DataFrame is not empty."""
        if data.empty:
            return False, "DataFrame is empty", {'shape': data.shape}
        return True, "DataFrame is not empty", {'shape': data.shape}
    
    def _validate_required_columns(self, data: pd.DataFrame, context: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate required columns exist."""
        required_columns = context.get('required_columns', [])
        if not required_columns:
            return True, "No required columns specified", {}
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}", {
                'missing_columns': missing_columns,
                'available_columns': list(data.columns)
            }
        
        return True, f"All required columns present: {required_columns}", {
            'required_columns': required_columns
        }
    
    def _validate_data_types(self, data: pd.DataFrame, context: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate data types are appropriate."""
        issues = []
        dtypes = {}
        
        for col in data.columns:
            dtype = str(data[col].dtype)
            dtypes[col] = dtype
            
            # Check for object dtype (often indicates issues)
            if dtype == 'object':
                # Check if it's actually numeric
                try:
                    pd.to_numeric(data[col], errors='raise')
                except (ValueError, TypeError):
                    issues.append(f"Column {col} has object dtype but is not numeric")
        
        if issues:
            return False, f"Data type issues found: {issues}", {
                'issues': issues,
                'dtypes': dtypes
            }
        
        return True, "Data types are appropriate", {'dtypes': dtypes}
    
    def _validate_no_infinite_values(self, data: pd.DataFrame, context: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate no infinite values exist."""
        inf_columns = []
        inf_counts = {}
        
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                inf_count = np.isinf(data[col]).sum()
                if inf_count > 0:
                    inf_columns.append(col)
                    inf_counts[col] = int(inf_count)
        
        if inf_columns:
            return False, f"Columns with infinite values: {inf_columns}", {
                'inf_columns': inf_columns,
                'inf_counts': inf_counts
            }
        
        return True, "No infinite values found", {}
    
    def _validate_no_nan_values(self, data: pd.DataFrame, context: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate no excessive NaN values exist."""
        nan_threshold = 0.5  # 50% threshold
        nan_columns = []
        nan_percentages = {}
        
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                nan_percentage = data[col].isna().sum() / len(data)
                nan_percentages[col] = float(nan_percentage)
                
                if nan_percentage > nan_threshold:
                    nan_columns.append(col)
        
        if nan_columns:
            return False, f"Columns with excessive NaN values: {nan_columns}", {
                'nan_columns': nan_columns,
                'nan_percentages': nan_percentages,
                'threshold': nan_threshold
            }
        
        return True, "NaN values are within acceptable limits", {
            'nan_percentages': nan_percentages,
            'threshold': nan_threshold
        }
    
    def _validate_sufficient_data_length(self, data: pd.DataFrame, context: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate sufficient data length."""
        min_length = 10
        actual_length = len(data)
        
        if actual_length < min_length:
            return False, f"Data length {actual_length} is less than minimum {min_length}", {
                'actual_length': actual_length,
                'min_length': min_length
            }
        
        return True, f"Data length {actual_length} is sufficient", {
            'actual_length': actual_length,
            'min_length': min_length
        }
    
    def _validate_target_columns(self, data: pd.DataFrame, context: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate target columns."""
        target_columns = context.get('target_columns', [])
        if not target_columns:
            return True, "No target columns specified", {}
        
        missing_targets = [col for col in target_columns if col not in data.columns]
        if missing_targets:
            return False, f"Missing target columns: {missing_targets}", {
                'missing_targets': missing_targets,
                'available_columns': list(data.columns)
            }
        
        return True, f"All target columns present: {target_columns}", {
            'target_columns': target_columns
        }
    
    def _validate_feature_columns(self, data: pd.DataFrame, context: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate feature columns."""
        feature_columns = context.get('feature_columns', [])
        if not feature_columns:
            return True, "No feature columns specified", {}
        
        missing_features = [col for col in feature_columns if col not in data.columns]
        if missing_features:
            return False, f"Missing feature columns: {missing_features}", {
                'missing_features': missing_features,
                'available_columns': list(data.columns)
            }
        
        return True, f"All feature columns present: {feature_columns}", {
            'feature_columns': feature_columns
        }
    
    def _create_validation_summary(self, results: List[ValidationResult]) -> ValidationSummary:
        """Create validation summary from results."""
        total_rules = len(results)
        passed = sum(1 for r in results if r.status == ValidationStatus.PASSED)
        failed = sum(1 for r in results if r.status == ValidationStatus.FAILED)
        warnings = sum(1 for r in results if r.status == ValidationStatus.WARNING)
        skipped = sum(1 for r in results if r.status == ValidationStatus.SKIPPED)
        critical_failures = sum(1 for r in results if r.status == ValidationStatus.FAILED and r.level == ValidationLevel.CRITICAL)
        
        # Calculate quality score
        quality_score = (passed / total_rules) * 100 if total_rules > 0 else 0
        
        # Generate recommendations
        recommendations = []
        for result in results:
            if result.status == ValidationStatus.FAILED:
                recommendations.append(f"Fix {result.rule_name}: {result.message}")
            elif result.status == ValidationStatus.WARNING:
                recommendations.append(f"Consider fixing {result.rule_name}: {result.message}")
        
        is_valid = critical_failures == 0 and failed == 0
        
        return ValidationSummary(
            total_rules=total_rules,
            passed=passed,
            failed=failed,
            warnings=warnings,
            skipped=skipped,
            critical_failures=critical_failures,
            recommendations=recommendations,
            quality_score=quality_score,
            is_valid=is_valid
        )
    
    def get_validation_history(self) -> List[Dict[str, Any]]:
        """Get validation history."""
        return self.validation_history.copy()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'critical_failures': 0
        }

# Convenience functions
def create_advanced_validator(component_name: str = "AdvancedValidator", 
                            logger: Optional[logging.Logger] = None) -> AdvancedValidator:
    """Create an advanced validator instance."""
    return AdvancedValidator(component_name, logger)

# Export main classes and functions
__all__ = [
    'ValidationLevel',
    'ValidationStatus',
    'ValidationRule',
    'ValidationResult',
    'ValidationSummary',
    'AdvancedValidator',
    'create_advanced_validator'
]