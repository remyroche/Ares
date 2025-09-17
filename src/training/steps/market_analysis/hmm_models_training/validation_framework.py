"""
HMM Training Validation Framework

Comprehensive validation and error handling framework for HMM models training.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings

from src.utils.tprint import tprint
from src.utils.common_operations import safe_divide, safe_float, safe_int
from src.utils.math_validation import validate_finite, validate_positive, validate_range
from src.utils.common_utilities import safe_convert_dtypes, calculate_data_quality_metrics

# Using tprint for all logging - no logger needed


class ValidationLevel(Enum):
    """Validation levels for different strictness."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"


class ValidationResult(Enum):
    """Validation result types."""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


@dataclass
class ValidationCheck:
    """Individual validation check result."""
    name: str
    result: ValidationResult
    message: str
    details: Optional[Dict[str, Any]] = None
    severity: str = "medium"  # low, medium, high, critical


@dataclass
class ValidationReport:
    """Complete validation report."""
    overall_result: ValidationResult
    checks: List[ValidationCheck]
    summary: Dict[str, Any]
    recommendations: List[str]
    timestamp: str


class HMMTrainingValidator:
    """
    Enhanced comprehensive validator for HMM training inputs and processes.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD, early_exit: bool = True):
        """
        Initialize enhanced validator.
        
        Args:
            validation_level: Level of validation strictness
            early_exit: Whether to exit early on critical failures
        """
        self.validation_level = validation_level
        self.early_exit = early_exit
        
        # Validation thresholds based on level
        self.thresholds = self._get_thresholds()
        
        tprint(f"✅ Enhanced HMM Training Validator initialized (level: {validation_level.value}, early_exit: {early_exit})")
    
    def _get_thresholds(self) -> Dict[str, Any]:
        """Get validation thresholds based on validation level."""
        if self.validation_level == ValidationLevel.BASIC:
            return {
                'min_samples': 100,
                'min_samples_per_regime': 5,
                'max_missing_ratio': 0.5,
                'max_infinite_ratio': 0.1,
                'min_feature_variance': 1e-8,
                'max_correlation_threshold': 0.99
            }
        elif self.validation_level == ValidationLevel.STANDARD:
            return {
                'min_samples': 1000,
                'min_samples_per_regime': 50,
                'max_missing_ratio': 0.1,
                'max_infinite_ratio': 0.01,
                'min_feature_variance': 1e-6,
                'max_correlation_threshold': 0.95
            }
        else:  # STRICT
            return {
                'min_samples': 5000,
                'min_samples_per_regime': 200,
                'max_missing_ratio': 0.01,
                'max_infinite_ratio': 0.001,
                'min_feature_variance': 1e-4,
                'max_correlation_threshold': 0.9
            }
    
    def validate_inputs(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> ValidationReport:
        """
        Comprehensive input validation.
        
        Args:
            X: Input features
            y: Target values
            regime_labels: Regime labels
            feature_names: Optional feature names
            
        Returns:
            ValidationReport with detailed results
        """
        tprint("🔄 Starting comprehensive input validation...")
        
        checks = []
        
        # Basic data type checks
        checks.extend(self._validate_data_types(X, y, regime_labels, feature_names))
        
        # Shape and size checks
        checks.extend(self._validate_shapes_and_sizes(X, y, regime_labels))
        
        # Data quality checks
        checks.extend(self._validate_data_quality(X, y, regime_labels))
        
        # Statistical checks
        checks.extend(self._validate_statistical_properties(X, y, regime_labels))
        
        # Regime-specific checks
        checks.extend(self._validate_regime_properties(regime_labels))
        
        # Feature-specific checks
        checks.extend(self._validate_feature_properties(X, feature_names))
        
        # Early exit on critical failures if enabled
        if self.early_exit:
            critical_failures = [check for check in checks if check.result == ValidationResult.FAIL and check.severity == "critical"]
            if critical_failures:
                tprint(f"❌ Early exit due to critical failures: {[f.name for f in critical_failures]}")
                return ValidationReport(
                    overall_result=ValidationResult.FAIL,
                    checks=critical_failures,
                    summary={"early_exit": True, "critical_failures": len(critical_failures)},
                    recommendations=[f"CRITICAL: {f.message}" for f in critical_failures],
                    timestamp=pd.Timestamp.now().isoformat()
                )
        
        # Generate report
        report = self._generate_validation_report(checks)
        
        tprint(f"✅ Enhanced input validation completed: {report.overall_result.value}")
        return report
    
    def _validate_data_types(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]]
    ) -> List[ValidationCheck]:
        """Validate data types."""
        checks = []
        
        # Check X type
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            checks.append(ValidationCheck(
                name="X_data_type",
                result=ValidationResult.FAIL,
                message=f"X must be numpy array or DataFrame, got {type(X)}",
                severity="critical"
            ))
        else:
            checks.append(ValidationCheck(
                name="X_data_type",
                result=ValidationResult.PASS,
                message=f"X data type is valid: {type(X)}"
            ))
        
        # Check y type
        if not isinstance(y, np.ndarray):
            checks.append(ValidationCheck(
                name="y_data_type",
                result=ValidationResult.FAIL,
                message=f"y must be numpy array, got {type(y)}",
                severity="critical"
            ))
        else:
            checks.append(ValidationCheck(
                name="y_data_type",
                result=ValidationResult.PASS,
                message=f"y data type is valid: {type(y)}"
            ))
        
        # Check regime_labels type
        if not isinstance(regime_labels, np.ndarray):
            checks.append(ValidationCheck(
                name="regime_labels_data_type",
                result=ValidationResult.FAIL,
                message=f"regime_labels must be numpy array, got {type(regime_labels)}",
                severity="critical"
            ))
        else:
            checks.append(ValidationCheck(
                name="regime_labels_data_type",
                result=ValidationResult.PASS,
                message=f"regime_labels data type is valid: {type(regime_labels)}"
            ))
        
        # Check feature_names type
        if feature_names is not None and not isinstance(feature_names, list):
            checks.append(ValidationCheck(
                name="feature_names_data_type",
                result=ValidationResult.WARNING,
                message=f"feature_names should be list, got {type(feature_names)}",
                severity="low"
            ))
        else:
            checks.append(ValidationCheck(
                name="feature_names_data_type",
                result=ValidationResult.PASS,
                message="feature_names data type is valid"
            ))
        
        return checks
    
    def _validate_shapes_and_sizes(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        regime_labels: np.ndarray
    ) -> List[ValidationCheck]:
        """Validate shapes and sizes."""
        checks = []
        
        # Check if data is empty
        if len(X) == 0:
            checks.append(ValidationCheck(
                name="X_empty",
                result=ValidationResult.FAIL,
                message="X is empty",
                severity="critical"
            ))
        else:
            checks.append(ValidationCheck(
                name="X_empty",
                result=ValidationResult.PASS,
                message=f"X has {len(X)} samples"
            ))
        
        # Check minimum sample size
        if len(X) < self.thresholds['min_samples']:
            checks.append(ValidationCheck(
                name="min_samples",
                result=ValidationResult.WARNING if self.validation_level == ValidationLevel.BASIC else ValidationResult.FAIL,
                message=f"X has {len(X)} samples, minimum recommended: {self.thresholds['min_samples']}",
                severity="high" if self.validation_level == ValidationLevel.STRICT else "medium"
            ))
        else:
            checks.append(ValidationCheck(
                name="min_samples",
                result=ValidationResult.PASS,
                message=f"X has sufficient samples: {len(X)}"
            ))
        
        # Check shape consistency
        if len(X) != len(y):
            checks.append(ValidationCheck(
                name="X_y_shape_consistency",
                result=ValidationResult.FAIL,
                message=f"X length ({len(X)}) != y length ({len(y)})",
                severity="critical"
            ))
        else:
            checks.append(ValidationCheck(
                name="X_y_shape_consistency",
                result=ValidationResult.PASS,
                message="X and y have consistent lengths"
            ))
        
        if len(X) != len(regime_labels):
            checks.append(ValidationCheck(
                name="X_regime_shape_consistency",
                result=ValidationResult.FAIL,
                message=f"X length ({len(X)}) != regime_labels length ({len(regime_labels)})",
                severity="critical"
            ))
        else:
            checks.append(ValidationCheck(
                name="X_regime_shape_consistency",
                result=ValidationResult.PASS,
                message="X and regime_labels have consistent lengths"
            ))
        
        return checks
    
    def _validate_data_quality(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        regime_labels: np.ndarray
    ) -> List[ValidationCheck]:
        """Validate data quality."""
        checks = []
        
        # Check for NaN values in X
        if isinstance(X, np.ndarray):
            nan_count = np.isnan(X).sum()
            nan_ratio = nan_count / X.size
        else:  # DataFrame
            nan_count = X.isnull().sum().sum()
            nan_ratio = nan_count / (X.shape[0] * X.shape[1])
        
        if nan_ratio > self.thresholds['max_missing_ratio']:
            checks.append(ValidationCheck(
                name="X_nan_values",
                result=ValidationResult.FAIL,
                message=f"X has {nan_ratio:.2%} NaN values, maximum allowed: {self.thresholds['max_missing_ratio']:.2%}",
                severity="high",
                details={"nan_count": int(nan_count), "nan_ratio": float(nan_ratio)}
            ))
        elif nan_ratio > 0:
            checks.append(ValidationCheck(
                name="X_nan_values",
                result=ValidationResult.WARNING,
                message=f"X has {nan_ratio:.2%} NaN values",
                severity="medium",
                details={"nan_count": int(nan_count), "nan_ratio": float(nan_ratio)}
            ))
        else:
            checks.append(ValidationCheck(
                name="X_nan_values",
                result=ValidationResult.PASS,
                message="X has no NaN values"
            ))
        
        # Check for infinite values in X
        if isinstance(X, np.ndarray):
            inf_count = np.isinf(X).sum()
            inf_ratio = inf_count / X.size
        else:  # DataFrame
            numeric_X = X.select_dtypes(include=[np.number])
            inf_count = np.isinf(numeric_X).sum().sum()
            inf_ratio = inf_count / (numeric_X.shape[0] * numeric_X.shape[1])
        
        if inf_ratio > self.thresholds['max_infinite_ratio']:
            checks.append(ValidationCheck(
                name="X_infinite_values",
                result=ValidationResult.FAIL,
                message=f"X has {inf_ratio:.2%} infinite values, maximum allowed: {self.thresholds['max_infinite_ratio']:.2%}",
                severity="high",
                details={"inf_count": int(inf_count), "inf_ratio": float(inf_ratio)}
            ))
        elif inf_ratio > 0:
            checks.append(ValidationCheck(
                name="X_infinite_values",
                result=ValidationResult.WARNING,
                message=f"X has {inf_ratio:.2%} infinite values",
                severity="medium",
                details={"inf_count": int(inf_count), "inf_ratio": float(inf_ratio)}
            ))
        else:
            checks.append(ValidationCheck(
                name="X_infinite_values",
                result=ValidationResult.PASS,
                message="X has no infinite values"
            ))
        
        # Check for NaN values in y
        y_nan_count = np.isnan(y).sum()
        y_nan_ratio = y_nan_count / len(y)
        
        if y_nan_ratio > 0:
            checks.append(ValidationCheck(
                name="y_nan_values",
                result=ValidationResult.FAIL,
                message=f"y has {y_nan_ratio:.2%} NaN values",
                severity="critical",
                details={"nan_count": int(y_nan_count), "nan_ratio": float(y_nan_ratio)}
            ))
        else:
            checks.append(ValidationCheck(
                name="y_nan_values",
                result=ValidationResult.PASS,
                message="y has no NaN values"
            ))
        
        # Check for infinite values in y
        y_inf_count = np.isinf(y).sum()
        y_inf_ratio = y_inf_count / len(y)
        
        if y_inf_ratio > 0:
            checks.append(ValidationCheck(
                name="y_infinite_values",
                result=ValidationResult.FAIL,
                message=f"y has {y_inf_ratio:.2%} infinite values",
                severity="critical",
                details={"inf_count": int(y_inf_count), "inf_ratio": float(y_inf_ratio)}
            ))
        else:
            checks.append(ValidationCheck(
                name="y_infinite_values",
                result=ValidationResult.PASS,
                message="y has no infinite values"
            ))
        
        # Check for NaN values in regime_labels
        regime_nan_count = np.isnan(regime_labels).sum()
        regime_nan_ratio = regime_nan_count / len(regime_labels)
        
        if regime_nan_ratio > 0:
            checks.append(ValidationCheck(
                name="regime_labels_nan_values",
                result=ValidationResult.FAIL,
                message=f"regime_labels has {regime_nan_ratio:.2%} NaN values",
                severity="critical",
                details={"nan_count": int(regime_nan_count), "nan_ratio": float(regime_nan_ratio)}
            ))
        else:
            checks.append(ValidationCheck(
                name="regime_labels_nan_values",
                result=ValidationResult.PASS,
                message="regime_labels has no NaN values"
            ))
        
        return checks
    
    def _validate_statistical_properties(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        regime_labels: np.ndarray
    ) -> List[ValidationCheck]:
        """Validate statistical properties."""
        checks = []
        
        # Check feature variance
        if isinstance(X, np.ndarray):
            feature_variances = np.var(X, axis=0)
        else:  # DataFrame
            feature_variances = X.var()
        
        low_variance_features = np.sum(feature_variances < self.thresholds['min_feature_variance'])
        low_variance_ratio = low_variance_features / len(feature_variances)
        
        if low_variance_ratio > 0.1:  # More than 10% of features have low variance
            checks.append(ValidationCheck(
                name="feature_variance",
                result=ValidationResult.WARNING,
                message=f"{low_variance_ratio:.1%} of features have low variance (< {self.thresholds['min_feature_variance']})",
                severity="medium",
                details={"low_variance_features": int(low_variance_features), "total_features": len(feature_variances)}
            ))
        else:
            checks.append(ValidationCheck(
                name="feature_variance",
                result=ValidationResult.PASS,
                message="Feature variances are acceptable"
            ))
        
        # Check target distribution
        if len(np.unique(y)) < 2:
            checks.append(ValidationCheck(
                name="target_diversity",
                result=ValidationResult.FAIL,
                message="Target variable has only one unique value",
                severity="critical"
            ))
        else:
            checks.append(ValidationCheck(
                name="target_diversity",
                result=ValidationResult.PASS,
                message=f"Target variable has {len(np.unique(y))} unique values"
            ))
        
        return checks
    
    def _validate_regime_properties(self, regime_labels: np.ndarray) -> List[ValidationCheck]:
        """Validate regime-specific properties."""
        checks = []
        
        unique_regimes, regime_counts = np.unique(regime_labels, return_counts=True)
        n_regimes = len(unique_regimes)
        
        # Check minimum number of regimes
        if n_regimes < 2:
            checks.append(ValidationCheck(
                name="min_regimes",
                result=ValidationResult.FAIL,
                message=f"Need at least 2 regimes, found {n_regimes}",
                severity="critical"
            ))
        else:
            checks.append(ValidationCheck(
                name="min_regimes",
                result=ValidationResult.PASS,
                message=f"Found {n_regimes} regimes"
            ))
        
        # Check minimum samples per regime
        min_samples_per_regime = np.min(regime_counts)
        if min_samples_per_regime < self.thresholds['min_samples_per_regime']:
            checks.append(ValidationCheck(
                name="min_samples_per_regime",
                result=ValidationResult.WARNING if self.validation_level == ValidationLevel.BASIC else ValidationResult.FAIL,
                message=f"Minimum samples per regime: {min_samples_per_regime}, recommended: {self.thresholds['min_samples_per_regime']}",
                severity="high" if self.validation_level == ValidationLevel.STRICT else "medium",
                details={"min_samples": int(min_samples_per_regime), "regime_distribution": dict(zip(unique_regimes, regime_counts))}
            ))
        else:
            checks.append(ValidationCheck(
                name="min_samples_per_regime",
                result=ValidationResult.PASS,
                message=f"All regimes have sufficient samples (minimum: {min_samples_per_regime})"
            ))
        
        # Check regime balance
        regime_balance = np.min(regime_counts) / np.max(regime_counts)
        if regime_balance < 0.1:  # Most imbalanced regime has less than 10% of the most balanced
            checks.append(ValidationCheck(
                name="regime_balance",
                result=ValidationResult.WARNING,
                message=f"Regime balance is poor: {regime_balance:.2f} (closer to 1.0 is better)",
                severity="medium",
                details={"balance_ratio": float(regime_balance), "regime_distribution": dict(zip(unique_regimes, regime_counts))}
            ))
        else:
            checks.append(ValidationCheck(
                name="regime_balance",
                result=ValidationResult.PASS,
                message=f"Regime balance is acceptable: {regime_balance:.2f}"
            ))
        
        return checks
    
    def _validate_feature_properties(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]]
    ) -> List[ValidationCheck]:
        """Validate feature-specific properties."""
        checks = []
        
        # Check feature names consistency
        if isinstance(X, pd.DataFrame) and feature_names is not None:
            if list(X.columns) != feature_names:
                checks.append(ValidationCheck(
                    name="feature_names_consistency",
                    result=ValidationResult.WARNING,
                    message="Feature names don't match DataFrame columns",
                    severity="low"
                ))
            else:
                checks.append(ValidationCheck(
                    name="feature_names_consistency",
                    result=ValidationResult.PASS,
                    message="Feature names are consistent with DataFrame columns"
                ))
        
        # Check for duplicate features
        if isinstance(X, pd.DataFrame):
            duplicate_features = X.columns.duplicated().sum()
            if duplicate_features > 0:
                checks.append(ValidationCheck(
                    name="duplicate_features",
                    result=ValidationResult.WARNING,
                    message=f"Found {duplicate_features} duplicate feature names",
                    severity="medium"
                ))
            else:
                checks.append(ValidationCheck(
                    name="duplicate_features",
                    result=ValidationResult.PASS,
                    message="No duplicate feature names found"
                ))
        
        # Check feature correlation
        if isinstance(X, pd.DataFrame) and X.shape[1] > 1:
            try:
                corr_matrix = X.corr().abs()
                high_corr_pairs = (corr_matrix > self.thresholds['max_correlation_threshold']).sum().sum() - X.shape[1]  # Subtract diagonal
                high_corr_ratio = high_corr_pairs / (X.shape[1] * (X.shape[1] - 1) / 2)
                
                if high_corr_ratio > 0.1:  # More than 10% of feature pairs are highly correlated
                    checks.append(ValidationCheck(
                        name="feature_correlation",
                        result=ValidationResult.WARNING,
                        message=f"{high_corr_ratio:.1%} of feature pairs are highly correlated (> {self.thresholds['max_correlation_threshold']})",
                        severity="medium",
                        details={"high_corr_pairs": int(high_corr_pairs), "total_pairs": X.shape[1] * (X.shape[1] - 1) // 2}
                    ))
                else:
                    checks.append(ValidationCheck(
                        name="feature_correlation",
                        result=ValidationResult.PASS,
                        message="Feature correlations are acceptable"
                    ))
            except Exception as e:
                checks.append(ValidationCheck(
                    name="feature_correlation",
                    result=ValidationResult.WARNING,
                    message=f"Could not compute feature correlations: {e}",
                    severity="low"
                ))
        
        return checks
    
    def _generate_validation_report(self, checks: List[ValidationCheck]) -> ValidationReport:
        """Generate comprehensive validation report."""
        # Determine overall result
        if any(check.result == ValidationResult.FAIL for check in checks):
            overall_result = ValidationResult.FAIL
        elif any(check.result == ValidationResult.WARNING for check in checks):
            overall_result = ValidationResult.WARNING
        else:
            overall_result = ValidationResult.PASS
        
        # Generate summary
        summary = {
            "total_checks": len(checks),
            "passed": sum(1 for check in checks if check.result == ValidationResult.PASS),
            "warnings": sum(1 for check in checks if check.result == ValidationResult.WARNING),
            "failed": sum(1 for check in checks if check.result == ValidationResult.FAIL),
            "validation_level": self.validation_level.value
        }
        
        # Generate recommendations
        recommendations = []
        
        failed_checks = [check for check in checks if check.result == ValidationResult.FAIL]
        warning_checks = [check for check in checks if check.result == ValidationResult.WARNING]
        
        if failed_checks:
            recommendations.append("Address all failed validation checks before proceeding")
            for check in failed_checks:
                if check.severity == "critical":
                    recommendations.append(f"CRITICAL: {check.message}")
        
        if warning_checks:
            recommendations.append("Consider addressing warning validation checks for better results")
            for check in warning_checks:
                if check.severity == "high":
                    recommendations.append(f"HIGH PRIORITY: {check.message}")
        
        if overall_result == ValidationResult.PASS:
            recommendations.append("All validations passed - data is ready for training")
        
        return ValidationReport(
            overall_result=overall_result,
            checks=checks,
            summary=summary,
            recommendations=recommendations,
            timestamp=pd.Timestamp.now().isoformat()
        )
    
    def validate_model_results(self, model_results: Dict[str, Any]) -> ValidationReport:
        """
        Validate model training results.
        
        Args:
            model_results: Dictionary of model training results
            
        Returns:
            ValidationReport for model results
        """
        tprint("🔄 Validating model training results...")
        
        checks = []
        
        # Check if any models were trained
        if not model_results:
            checks.append(ValidationCheck(
                name="no_models_trained",
                result=ValidationResult.FAIL,
                message="No models were trained",
                severity="critical"
            ))
        else:
            checks.append(ValidationCheck(
                name="models_trained",
                result=ValidationResult.PASS,
                message=f"{len(model_results)} models were trained"
            ))
        
        # Check model performance
        successful_models = 0
        failed_models = 0
        accuracies = []
        
        for model_name, result in model_results.items():
            if hasattr(result, 'metrics'):
                if result.metrics.error_message is None:
                    successful_models += 1
                    if hasattr(result.metrics, 'accuracy'):
                        accuracies.append(result.metrics.accuracy)
                else:
                    failed_models += 1
        
        if successful_models == 0:
            checks.append(ValidationCheck(
                name="no_successful_models",
                result=ValidationResult.FAIL,
                message="No models trained successfully",
                severity="critical"
            ))
        else:
            checks.append(ValidationCheck(
                name="successful_models",
                result=ValidationResult.PASS,
                message=f"{successful_models} models trained successfully"
            ))
        
        if failed_models > 0:
            checks.append(ValidationCheck(
                name="failed_models",
                result=ValidationResult.WARNING,
                message=f"{failed_models} models failed to train",
                severity="medium"
            ))
        
        # Check performance quality
        if accuracies:
            avg_accuracy = np.mean(accuracies)
            max_accuracy = np.max(accuracies)
            
            if avg_accuracy < 0.5:
                checks.append(ValidationCheck(
                    name="low_average_accuracy",
                    result=ValidationResult.WARNING,
                    message=f"Average accuracy is low: {avg_accuracy:.3f}",
                    severity="medium"
                ))
            else:
                checks.append(ValidationCheck(
                    name="average_accuracy",
                    result=ValidationResult.PASS,
                    message=f"Average accuracy is acceptable: {avg_accuracy:.3f}"
                ))
            
            if max_accuracy < 0.7:
                checks.append(ValidationCheck(
                    name="low_best_accuracy",
                    result=ValidationResult.WARNING,
                    message=f"Best accuracy is low: {max_accuracy:.3f}",
                    severity="medium"
                ))
            else:
                checks.append(ValidationCheck(
                    name="best_accuracy",
                    result=ValidationResult.PASS,
                    message=f"Best accuracy is good: {max_accuracy:.3f}"
                ))
        
        # Generate report
        report = self._generate_validation_report(checks)
        
        tprint(f"✅ Model results validation completed: {report.overall_result.value}")
        return report


# Convenience functions
def validate_hmm_training_inputs(
    X: Union[np.ndarray, pd.DataFrame],
    y: np.ndarray,
    regime_labels: np.ndarray,
    validation_level: ValidationLevel = ValidationLevel.STANDARD,
    feature_names: Optional[List[str]] = None
) -> ValidationReport:
    """
    Convenience function for input validation.
    
    Args:
        X: Input features
        y: Target values
        regime_labels: Regime labels
        validation_level: Validation strictness level
        feature_names: Optional feature names
        
    Returns:
        ValidationReport with detailed results
    """
    validator = HMMTrainingValidator(validation_level)
    return validator.validate_inputs(X, y, regime_labels, feature_names)


def validate_hmm_training_results(
    model_results: Dict[str, Any],
    validation_level: ValidationLevel = ValidationLevel.STANDARD
) -> ValidationReport:
    """
    Convenience function for results validation.
    
    Args:
        model_results: Model training results
        validation_level: Validation strictness level
        
    Returns:
        ValidationReport with detailed results
    """
    validator = HMMTrainingValidator(validation_level)
    return validator.validate_model_results(model_results)