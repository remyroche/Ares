"""
Feature engineering validation utilities.

This module provides validation utilities for feature engineering operations
including feature validation, data quality checks, and engineering pipeline validation.
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
    timestamp: str = ""

class FeatureEngineeringValidator:
    """Validator for feature engineering operations."""
    
    def __init__(self):
        self.validation_results: List[ValidationResult] = []
        self.feature_metadata: Dict[str, Dict[str, Any]] = {}
    
    def validate_feature_data_types(self, df: pd.DataFrame, feature_specs: Dict[str, str]) -> List[ValidationResult]:
        """Validate feature data types."""
        results = []
        
        for feature, expected_type in feature_specs.items():
            if feature not in df.columns:
                results.append(ValidationResult(
                    status=ValidationStatus.FAIL,
                    message=f"Feature {feature} not found in DataFrame",
                    feature_name=feature
                ))
                continue
            
            actual_type = str(df[feature].dtype)
            if expected_type not in actual_type:
                results.append(ValidationResult(
                    status=ValidationStatus.WARNING,
                    message=f"Feature {feature} type mismatch: expected {expected_type}, got {actual_type}",
                    feature_name=feature
                ))
            else:
                results.append(ValidationResult(
                    status=ValidationStatus.PASS,
                    message=f"Feature {feature} type validation passed",
                    feature_name=feature
                ))
        
        return results
    
    def validate_feature_ranges(self, df: pd.DataFrame, feature_ranges: Dict[str, Tuple[float, float]]) -> List[ValidationResult]:
        """Validate feature value ranges."""
        results = []
        
        for feature, (min_val, max_val) in feature_ranges.items():
            if feature not in df.columns:
                results.append(ValidationResult(
                    status=ValidationStatus.FAIL,
                    message=f"Feature {feature} not found in DataFrame",
                    feature_name=feature
                ))
                continue
            
            feature_data = df[feature]
            if not pd.api.types.is_numeric_dtype(feature_data):
                results.append(ValidationResult(
                    status=ValidationStatus.WARNING,
                    message=f"Feature {feature} is not numeric, skipping range validation",
                    feature_name=feature
                ))
                continue
            
            actual_min = feature_data.min()
            actual_max = feature_data.max()
            
            if actual_min < min_val or actual_max > max_val:
                results.append(ValidationResult(
                    status=ValidationStatus.WARNING,
                    message=f"Feature {feature} out of range: [{actual_min:.2f}, {actual_max:.2f}] vs expected [{min_val:.2f}, {max_val:.2f}]",
                    feature_name=feature
                ))
            else:
                results.append(ValidationResult(
                    status=ValidationStatus.PASS,
                    message=f"Feature {feature} range validation passed",
                    feature_name=feature
                ))
        
        return results
    
    def validate_feature_distribution(self, df: pd.DataFrame, feature: str, expected_distribution: str = "normal") -> ValidationResult:
        """Validate feature distribution."""
        try:
            if feature not in df.columns:
                return ValidationResult(
                    status=ValidationStatus.FAIL,
                    message=f"Feature {feature} not found in DataFrame",
                    feature_name=feature
                )
            
            feature_data = df[feature].dropna()
            if len(feature_data) == 0:
                return ValidationResult(
                    status=ValidationStatus.FAIL,
                    message=f"Feature {feature} has no valid data",
                    feature_name=feature
                )
            
            # Calculate distribution statistics
            skewness = feature_data.skew()
            kurtosis = feature_data.kurtosis()
            
            # Check for normality
            if expected_distribution == "normal":
                if abs(skewness) > 2 or abs(kurtosis) > 3:
                    return ValidationResult(
                        status=ValidationStatus.WARNING,
                        message=f"Feature {feature} may not be normally distributed (skewness: {skewness:.2f}, kurtosis: {kurtosis:.2f})",
                        feature_name=feature,
                        details={'skewness': skewness, 'kurtosis': kurtosis}
                    )
            
            return ValidationResult(
                status=ValidationStatus.PASS,
                message=f"Feature {feature} distribution validation passed",
                feature_name=feature,
                details={'skewness': skewness, 'kurtosis': kurtosis}
            )
        
        except Exception as e:
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message=f"Error validating feature distribution: {str(e)}",
                feature_name=feature
            )
    
    def validate_feature_correlation(self, df: pd.DataFrame, max_correlation: float = 0.95) -> List[ValidationResult]:
        """Validate feature correlations."""
        results = []
        
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                return [ValidationResult(
                    status=ValidationStatus.WARNING,
                    message="No numeric features found for correlation validation",
                    feature_name="correlation"
                )]
            
            correlation_matrix = numeric_df.corr()
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > max_correlation:
                        high_corr_pairs.append((
                            correlation_matrix.columns[i],
                            correlation_matrix.columns[j],
                            corr_value
                        ))
            
            if high_corr_pairs:
                results.append(ValidationResult(
                    status=ValidationStatus.WARNING,
                    message=f"Found {len(high_corr_pairs)} highly correlated feature pairs",
                    feature_name="correlation",
                    details={'high_corr_pairs': high_corr_pairs}
                ))
            else:
                results.append(ValidationResult(
                    status=ValidationStatus.PASS,
                    message="Feature correlation validation passed",
                    feature_name="correlation"
                ))
        
        except Exception as e:
            results.append(ValidationResult(
                status=ValidationStatus.FAIL,
                message=f"Error validating feature correlations: {str(e)}",
                feature_name="correlation"
            ))
        
        return results
    
    def validate_feature_engineering_pipeline(self, input_df: pd.DataFrame, output_df: pd.DataFrame) -> List[ValidationResult]:
        """Validate feature engineering pipeline."""
        results = []
        
        try:
            # Check if output has more features than input
            if len(output_df.columns) <= len(input_df.columns):
                results.append(ValidationResult(
                    status=ValidationStatus.WARNING,
                    message=f"Output has same or fewer features than input ({len(output_df.columns)} vs {len(input_df.columns)})",
                    feature_name="pipeline"
                ))
            
            # Check for new features
            new_features = set(output_df.columns) - set(input_df.columns)
            if new_features:
                results.append(ValidationResult(
                    status=ValidationStatus.PASS,
                    message=f"Successfully created {len(new_features)} new features",
                    feature_name="pipeline",
                    details={'new_features': list(new_features)}
                ))
            
            # Check data integrity
            if len(output_df) != len(input_df):
                results.append(ValidationResult(
                    status=ValidationStatus.FAIL,
                    message=f"Row count mismatch: input {len(input_df)}, output {len(output_df)}",
                    feature_name="pipeline"
                ))
            
            # Check for excessive missing values in new features
            if new_features:
                for feature in new_features:
                    missing_pct = (output_df[feature].isnull().sum() / len(output_df)) * 100
                    if missing_pct > 50:
                        results.append(ValidationResult(
                            status=ValidationStatus.WARNING,
                            message=f"New feature {feature} has high missing value percentage: {missing_pct:.2f}%",
                            feature_name=feature
                        ))
        
        except Exception as e:
            results.append(ValidationResult(
                status=ValidationStatus.FAIL,
                message=f"Error validating feature engineering pipeline: {str(e)}",
                feature_name="pipeline"
            ))
        
        return results
    
    def validate_feature_importance(self, feature_importance: Dict[str, float], min_importance: float = 0.01) -> List[ValidationResult]:
        """Validate feature importance scores."""
        results = []
        
        try:
            low_importance_features = [f for f, imp in feature_importance.items() if imp < min_importance]
            
            if low_importance_features:
                results.append(ValidationResult(
                    status=ValidationStatus.WARNING,
                    message=f"Found {len(low_importance_features)} features with low importance",
                    feature_name="importance",
                    details={'low_importance_features': low_importance_features}
                ))
            else:
                results.append(ValidationResult(
                    status=ValidationStatus.PASS,
                    message="All features have adequate importance scores",
                    feature_name="importance"
                ))
        
        except Exception as e:
            results.append(ValidationResult(
                status=ValidationStatus.FAIL,
                message=f"Error validating feature importance: {str(e)}",
                feature_name="importance"
            ))
        
        return results
    
    def run_comprehensive_validation(self, df: pd.DataFrame, feature_specs: Dict[str, str] = None, 
                                   feature_ranges: Dict[str, Tuple[float, float]] = None) -> List[ValidationResult]:
        """Run comprehensive feature engineering validation."""
        results = []
        
        # Validate data types
        if feature_specs:
            results.extend(self.validate_feature_data_types(df, feature_specs))
        
        # Validate ranges
        if feature_ranges:
            results.extend(self.validate_feature_ranges(df, feature_ranges))
        
        # Validate correlations
        results.extend(self.validate_feature_correlation(df))
        
        # Validate distributions for numeric features
        numeric_features = df.select_dtypes(include=[np.number]).columns
        for feature in numeric_features:
            dist_result = self.validate_feature_distribution(df, feature)
            results.append(dist_result)
        
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
