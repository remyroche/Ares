"""
Cross-step validation utilities.

This module provides validation utilities for cross-step operations
and data consistency checks across different pipeline steps.
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
    step_name: str = ""
    timestamp: str = ""

class CrossStepValidator:
    """Validator for cross-step operations."""

    def __init__(self):
        self.validation_results: List[ValidationResult] = []
        self.step_data: Dict[str, Any] = {}

    def validate_data_consistency(self, step1_data: Any, step2_data: Any, tolerance: float = 1e-6) -> ValidationResult:
        """Validate data consistency between steps."""
        try:
            if isinstance(step1_data, pd.DataFrame) and isinstance(step2_data, pd.DataFrame):
                # Check shape consistency
                if step1_data.shape != step2_data.shape:
                    return ValidationResult(
                        status=ValidationStatus.FAIL,
                        message=f"DataFrame shape mismatch: {step1_data.shape} vs {step2_data.shape}",
                        step_name="data_consistency"
                    )

                # Check column consistency
                if not step1_data.columns.equals(step2_data.columns):
                    return ValidationResult(
                        status=ValidationStatus.FAIL,
                        message="DataFrame column mismatch",
                        step_name="data_consistency"
                    )

                # Check data consistency for numeric columns
                numeric_cols = step1_data.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    diff = np.abs(step1_data[col] - step2_data[col])
                    max_diff = diff.max()
                    if max_diff > tolerance:
                        return ValidationResult(
                            status=ValidationStatus.WARNING,
                            message=f"Data inconsistency in column {col}: max difference {max_diff}",
                            step_name="data_consistency"
                        )

                return ValidationResult(
                    status=ValidationStatus.PASS,
                    message="Data consistency validation passed",
                    step_name="data_consistency"
                )

            else:
                return ValidationResult(
                    status=ValidationStatus.WARNING,
                    message="Non-DataFrame data types, skipping consistency check",
                    step_name="data_consistency"
                )

        except Exception as e:
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message=f"Error in data consistency validation: {str(e)}",
                step_name="data_consistency"
            )

    def validate_step_dependencies(self, current_step: str, required_steps: List[str]) -> ValidationResult:
        """Validate that required steps have been completed."""
        try:
            missing_steps = []
            for step in required_steps:
                if step not in self.step_data:
                    missing_steps.append(step)

            if missing_steps:
                return ValidationResult(
                    status=ValidationStatus.FAIL,
                    message=f"Missing required steps: {missing_steps}",
                    step_name=current_step
                )

            return ValidationResult(
                status=ValidationStatus.PASS,
                message="All required steps completed",
                step_name=current_step
            )

        except Exception as e:
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message=f"Error validating step dependencies: {str(e)}",
                step_name=current_step
            )

    def validate_data_quality(self, data: Any, step_name: str) -> ValidationResult:
        """Validate data quality for a step."""
        try:
            if isinstance(data, pd.DataFrame):
                # Check for missing values
                missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100

                if missing_pct > 50:
                    return ValidationResult(
                        status=ValidationStatus.FAIL,
                        message=f"High missing value percentage: {missing_pct:.2f}%",
                        step_name=step_name
                    )
                elif missing_pct > 20:
                    return ValidationResult(
                        status=ValidationStatus.WARNING,
                        message=f"Moderate missing value percentage: {missing_pct:.2f}%",
                        step_name=step_name
                    )

                # Check for duplicate rows
                duplicate_pct = (data.duplicated().sum() / len(data)) * 100
                if duplicate_pct > 10:
                    return ValidationResult(
                        status=ValidationStatus.WARNING,
                        message=f"High duplicate percentage: {duplicate_pct:.2f}%",
                        step_name=step_name
                    )

                # Check for constant columns
                constant_cols = [col for col in data.columns if data[col].nunique() <= 1]
                if constant_cols:
                    return ValidationResult(
                        status=ValidationStatus.WARNING,
                        message=f"Constant columns found: {constant_cols}",
                        step_name=step_name
                    )

                return ValidationResult(
                    status=ValidationStatus.PASS,
                    message="Data quality validation passed",
                    step_name=step_name
                )

            else:
                return ValidationResult(
                    status=ValidationStatus.PASS,
                    message="Non-DataFrame data, skipping quality check",
                    step_name=step_name
                )

        except Exception as e:
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message=f"Error in data quality validation: {str(e)}",
                step_name=step_name
            )

    def validate_feature_consistency(self, features: List[str], expected_features: List[str]) -> ValidationResult:
        """Validate feature consistency."""
        try:
            missing_features = set(expected_features) - set(features)
            extra_features = set(features) - set(expected_features)

            if missing_features:
                return ValidationResult(
                    status=ValidationStatus.FAIL,
                    message=f"Missing expected features: {list(missing_features)}",
                    step_name="feature_consistency"
                )

            if extra_features:
                return ValidationResult(
                    status=ValidationStatus.WARNING,
                    message=f"Extra features found: {list(extra_features)}",
                    step_name="feature_consistency"
                )

            return ValidationResult(
                status=ValidationStatus.PASS,
                message="Feature consistency validation passed",
                step_name="feature_consistency"
            )

        except Exception as e:
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message=f"Error in feature consistency validation: {str(e)}",
                step_name="feature_consistency"
            )

    def register_step_data(self, step_name: str, data: Any):
        """Register data for a step."""
        self.step_data[step_name] = data
        logger.info(f"Registered data for step: {step_name}")

    def get_step_data(self, step_name: str) -> Optional[Any]:
        """Get data for a step."""
        return self.step_data.get(step_name)

    def run_cross_step_validation(self, current_step: str, required_steps: List[str] = None) -> List[ValidationResult]:
        """Run comprehensive cross-step validation."""
        results = []

        # Validate step dependencies
        if required_steps:
            dep_result = self.validate_step_dependencies(current_step, required_steps)
            results.append(dep_result)

        # Validate current step data quality
        current_data = self.get_step_data(current_step)
        if current_data is not None:
            quality_result = self.validate_data_quality(current_data, current_step)
            results.append(quality_result)

        # Validate data consistency with previous steps
        if required_steps:
            for prev_step in required_steps:
                prev_data = self.get_step_data(prev_step)
                if prev_data is not None and current_data is not None:
                    consistency_result = self.validate_data_consistency(prev_data, current_data)
                    results.append(consistency_result)

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
