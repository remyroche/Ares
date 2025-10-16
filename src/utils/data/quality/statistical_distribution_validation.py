"""
Statistical distribution validation utilities.
"""

import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

# Setup logging
logger = logging.getLogger(__name__)

class ValidationStatus(Enum):
    """Validation status."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"

@dataclass
class ValidationResult:
    """Validation result."""
    status: ValidationStatus
    message: str
    details: Dict[str, Any] = None
    feature_name: str = ""

class StatisticalValidator:
    """Validator for statistical distributions."""

    def __init__(self):
        self.validation_results: List[ValidationResult] = []

    def validate_normality(self, data: np.ndarray) -> ValidationResult:
        """Validate normality of data."""
        try:
            clean_data = data[~np.isnan(data)]
            if len(clean_data) < 3:
                return ValidationResult(
                    status=ValidationStatus.FAIL,
                    message="Insufficient data for normality validation",
                    feature_name="normality"
                )

            # Simple normality check using skewness and kurtosis
            skewness = np.mean((clean_data - np.mean(clean_data))**3) / (np.std(clean_data)**3)
            kurtosis = np.mean((clean_data - np.mean(clean_data))**4) / (np.std(clean_data)**4) - 3

            if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
                return ValidationResult(
                    status=ValidationStatus.PASS,
                    message="Data appears to be normally distributed",
                    feature_name="normality",
                    details={'skewness': skewness, 'kurtosis': kurtosis}
                )
            else:
                return ValidationResult(
                    status=ValidationStatus.WARNING,
                    message=f"Data may not be normally distributed (skewness: {skewness:.2f}, kurtosis: {kurtosis:.2f})",
                    feature_name="normality",
                    details={'skewness': skewness, 'kurtosis': kurtosis}
                )

        except Exception as e:
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message=f"Error in normality validation: {str(e)}",
                feature_name="normality"
            )

    def validate_statistical_properties(self, data: np.ndarray) -> ValidationResult:
        """Validate statistical properties of data."""
        try:
            clean_data = data[~np.isnan(data)]
            if len(clean_data) < 3:
                return ValidationResult(
                    status=ValidationStatus.FAIL,
                    message="Insufficient data for statistical validation",
                    feature_name="statistical_properties"
                )

            # Calculate basic statistics
            mean_val = np.mean(clean_data)
            std_val = np.std(clean_data)
            skewness = np.mean((clean_data - mean_val)**3) / (std_val**3)
            kurtosis = np.mean((clean_data - mean_val)**4) / (std_val**4) - 3

            # Check for outliers using IQR method
            q1, q3 = np.percentile(clean_data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = np.sum((clean_data < lower_bound) | (clean_data > upper_bound))
            outlier_percentage = (outliers / len(clean_data)) * 100

            # Validate properties
            issues = []
            if abs(skewness) > 2:
                issues.append(f"High skewness: {skewness:.2f}")
            if abs(kurtosis) > 3:
                issues.append(f"High kurtosis: {kurtosis:.2f}")
            if outlier_percentage > 10:
                issues.append(f"High outlier percentage: {outlier_percentage:.2f}%")

            if issues:
                return ValidationResult(
                    status=ValidationStatus.WARNING,
                    message=f"Statistical issues found: {', '.join(issues)}",
                    feature_name="statistical_properties",
                    details={
                        'mean': mean_val,
                        'std': std_val,
                        'skewness': skewness,
                        'kurtosis': kurtosis,
                        'outlier_percentage': outlier_percentage,
                        'issues': issues
                    }
                )
            else:
                return ValidationResult(
                    status=ValidationStatus.PASS,
                    message="Statistical properties validation passed",
                    feature_name="statistical_properties",
                    details={
                        'mean': mean_val,
                        'std': std_val,
                        'skewness': skewness,
                        'kurtosis': kurtosis,
                        'outlier_percentage': outlier_percentage
                    }
                )

        except Exception as e:
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message=f"Error in statistical properties validation: {str(e)}",
                feature_name="statistical_properties"
            )

    def run_comprehensive_validation(self, data: np.ndarray) -> List[ValidationResult]:
        """Run comprehensive statistical validation."""
        results = []

        # Validate normality
        norm_result = self.validate_normality(data)
        results.append(norm_result)

        # Validate statistical properties
        stats_result = self.validate_statistical_properties(data)
        results.append(stats_result)

        self.validation_results.extend(results)
        return results

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        total_validations = len(self.validation_results)
        passed = len([r for r in self.validation_results if r.status == ValidationStatus.PASS])
        failed = len([r for r in self.validation_results if r.status == ValidationStatus.FAIL])
        warnings = len([r for r in self.validation_results if r.status == ValidationStatus.WARNING])

        return {
            'total_validations': total_validations,
            'passed': passed,
            'failed': failed,
            'warnings': warnings,
            'success_rate': passed / total_validations if total_validations > 0 else 0,
            'results': self.validation_results
        }

    def clear_validation_results(self):
        """Clear validation results."""
        self.validation_results.clear()
        logger.info("Cleared validation results")
