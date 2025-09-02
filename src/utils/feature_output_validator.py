"""
Feature Output Validator for Feature Engineering
Detects corrupted, invalid, or problematic feature engineering outputs.
"""

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.utils.warning_symbols import critical

warnings.filterwarnings("ignore")


class OutputValidationLevel(str, Enum):
    """Validation levels for feature output issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class OutputValidationIssue:
    """Represents a feature output validation issue."""
    feature_name: str
    issue_type: str
    level: OutputValidationLevel
    description: str
    count: int = 0
    percentage: float = 0.0
    details: Optional[Dict[str, Any]] = None


class FeatureOutputValidator:
    """
    Comprehensive validator for feature engineering outputs.
    Detects corrupted, invalid, or problematic feature results.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the validator with configuration."""
        self.logger = system_logger.getChild("FeatureOutputValidator")
        self.config = config or self._get_default_config()
        self.issues: List[OutputValidationIssue] = []

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for validation thresholds."""
        return {
            # Critical thresholds that indicate corrupted output - made more lenient for financial data
            "critical_thresholds": {
                "max_nan_percentage": 0.3,
                "max_infinite_percentage": 0.05,
                "max_zero_variance_percentage": 0.7,
                "max_constant_percentage": 0.9,
                "max_extreme_values_percentage": 0.1,
                "min_feature_count": 1,
                "max_feature_count": 10000,
            },
            # Warning thresholds - made more lenient for financial data
            "warning_thresholds": {
                "max_nan_percentage": 0.15,
                "max_infinite_percentage": 0.01,
                "max_zero_variance_percentage": 0.5,
                "max_constant_percentage": 0.8,
                "max_extreme_values_percentage": 0.05,
                "max_correlation_threshold": 0.99,
                "max_duplicate_features_percentage": 0.1,
            },
            # Feature type specific thresholds - made more lenient for financial data
            "feature_type_thresholds": {
                "wavelet_features": {
                    "max_nan_percentage": 0.4,
                    "max_infinite_percentage": 0.1,
                    "description": "Wavelet features naturally have edge effects",
                },
                "microstructure_features": {
                    "max_nan_percentage": 0.2,
                    "max_infinite_percentage": 0.05,
                    "description": "Microstructure features should be mostly complete",
                },
                "technical_indicators": {
                    "max_nan_percentage": 0.1,
                    "max_infinite_percentage": 0.01,
                    "description": "Technical indicators should be reliable",
                },
                "price_features": {
                    "max_nan_percentage": 0.01,
                    "max_infinite_percentage": 0.001,
                    "description": "Price-based features should be nearly complete",
                },
            },
            # Validation checks
            "validation_checks": {
                "check_nan_values": True,
                "check_infinite_values": True,
                "check_zero_variance": True,
                "check_constant_values": True,
                "check_extreme_values": True,
                "check_data_types": True,
                "check_feature_correlations": True,
                "check_duplicate_features": True,
                "check_feature_names": True,
                "check_output_structure": True,
            },
        }

    def validate_feature_output(
        self,
        features: Union[pd.DataFrame, Dict[str, Any], np.ndarray],
        method_name: str,
        input_data_shape: Optional[Tuple[int, ...]] = None,
    ) -> Dict[str, Any]:
        """
        Validate feature engineering output for quality and compatibility.
        
        Args:
            features: Feature output to validate
            method_name: Name of the feature engineering method
            input_data_shape: Shape of input data for consistency checks
            
        Returns:
            Dictionary with validation results and recommendations
        """
        print(f"🔍 [FEATURE OUTPUT VALIDATION] Starting validation for {method_name}")
        self.logger.info(
            f"🔍 [FEATURE OUTPUT VALIDATION] Starting validation for {method_name}",
        )

        if input_data_shape:
            print(f"   📊 Input data shape: {input_data_shape}")
            self.logger.info(f"   📊 Input data shape: {input_data_shape}")
        
        print(f"   📊 Features type: {type(features)}")
        self.logger.info(f"   📊 Features type: {type(features)}")

        self.issues.clear()

        validation_results: Dict[str, Any] = {
            "method_name": method_name,
            "validation_passed": True,
            "critical_issues": [],
            "warnings": [],
            "recommendations": [],
            "output_quality_score": 0.0,
            "feature_statistics": {},
            "detailed_analysis": {},
        }

        # Convert features to DataFrame if it's a dict
        print(
            f"🔍 [FEATURE OUTPUT VALIDATION] Converting features to DataFrame for {method_name}",
        )
        self.logger.info(
            f"🔍 [FEATURE OUTPUT VALIDATION] Converting features to DataFrame for {method_name}",
        )

        try:
            features_df = self._convert_features_to_dataframe(features)
            if features_df is None:
                validation_results["validation_passed"] = False
                validation_results["critical_issues"].append("Failed to convert features to DataFrame")
                return validation_results

            # Basic structure validation
            if not self._validate_output_structure(features_df, method_name):
                validation_results["validation_passed"] = False
                validation_results["critical_issues"].append("Output structure validation failed")

            # Data type validation
            if not self._validate_data_types(features_df):
                validation_results["validation_passed"] = False
                validation_results["critical_issues"].append("Data type validation failed")

            # Get method-specific thresholds
            thresholds = self._get_method_specific_thresholds(method_name)

            # Feature value validation
            value_issues = self._validate_feature_values(features_df, thresholds)
            if value_issues:
                validation_results["warnings"].extend(value_issues)

            # Feature relationship validation
            if not self._validate_feature_relationships(features_df, thresholds):
                validation_results["warnings"].append("Feature relationship validation failed")

            # Input-output consistency validation
            if input_data_shape and not self._validate_input_output_consistency(
                features_df, input_data_shape
            ):
                validation_results["warnings"].append("Input-output consistency validation failed")

            # Downstream compatibility validation
            if not self._validate_downstream_compatibility(features_df):
                validation_results["warnings"].append("Downstream compatibility validation failed")

            # Calculate quality score
            validation_results["output_quality_score"] = self._calculate_quality_score(
                validation_results
            )

            # Generate recommendations
            validation_results["recommendations"] = self._generate_output_recommendations(
                validation_results
            )

            # Final validation status
            validation_results["validation_passed"] = (
                len(validation_results["critical_issues"]) == 0
                and validation_results["output_quality_score"] >= 0.7
            )

            print(f"✅ [FEATURE OUTPUT VALIDATION] Validation completed for {method_name}")
            self.logger.info(f"✅ [FEATURE OUTPUT VALIDATION] Validation completed for {method_name}")

        except Exception as e:
            error_msg = f"Validation failed with error: {str(e)}"
            print(f"❌ [FEATURE OUTPUT VALIDATION] {error_msg}")
            self.logger.exception(f"❌ [FEATURE OUTPUT VALIDATION] {error_msg}")
            validation_results["validation_passed"] = False
            validation_results["critical_issues"].append(error_msg)

        return validation_results

    def _convert_features_to_dataframe(
        self, features: Union[pd.DataFrame, Dict[str, Any], np.ndarray]
    ) -> Optional[pd.DataFrame]:
        """Convert various feature formats to DataFrame."""
        # Handle None input
        if features is None:
            self.logger.error("Features input is None")
            return None

        try:
            if isinstance(features, pd.DataFrame):
                return features.copy()
            elif isinstance(features, dict):
                # Convert dict to DataFrame
                if all(isinstance(v, (pd.Series, np.ndarray)) for v in features.values()):
                    return pd.DataFrame(features)
                else:
                    # Try to create DataFrame from dict values
                    return pd.DataFrame.from_dict(features, orient='index').T
            elif isinstance(features, np.ndarray):
                return pd.DataFrame(features)
            else:
                self.logger.error(f"Unsupported features type: {type(features)}")
                return None
        except Exception as e:
            self.logger.exception(f"Error converting features to DataFrame: {e}")
            return None

    def _validate_output_structure(
        self, features_df: pd.DataFrame, method_name: str
    ) -> bool:
        """Validate the structure of feature output."""
        self.logger.info("Validating output structure...")
        
        try:
            # Check if DataFrame is empty
            if features_df.empty:
                self.logger.warning("Feature DataFrame is empty")
                return False

            # Check for required columns
            if features_df.columns.empty:
                self.logger.warning("Feature DataFrame has no columns")
                return False

            # Check for reasonable number of features
            feature_count = len(features_df.columns)
            if feature_count < self.config["critical_thresholds"]["min_feature_count"]:
                self.logger.warning(f"Too few features: {feature_count}")
                return False

            if feature_count > self.config["critical_thresholds"]["max_feature_count"]:
                self.logger.warning(f"Too many features: {feature_count}")
                return False

            # Check for reasonable number of samples
            sample_count = len(features_df)
            if sample_count == 0:
                self.logger.warning("No samples in feature DataFrame")
                return False

            self.logger.info(f"Output structure validation passed: {feature_count} features, {sample_count} samples")
            return True

        except Exception as e:
            self.logger.exception(f"Error in output structure validation: {e}")
            return False

    def _validate_data_types(self, features_df: pd.DataFrame) -> bool:
        """Validate data types of features."""
        self.logger.info("Validating data types...")
        
        try:
            # Check for non-numeric columns
            non_numeric_cols = []
            for col in features_df.columns:
                if not pd.api.types.is_numeric_dtype(features_df[col]):
                    non_numeric_cols.append(col)

            if non_numeric_cols:
                self.logger.warning(f"Non-numeric columns found: {non_numeric_cols}")
                # This is a warning, not critical for financial features

            # Check for object columns that should be numeric
            object_cols = features_df.select_dtypes(include=['object']).columns
            if len(object_cols) > 0:
                self.logger.warning(f"Object columns found: {list(object_cols)}")

            self.logger.info("Data type validation completed")
            return True

        except Exception as e:
            self.logger.exception(f"Error in data type validation: {e}")
            return False

    def _get_method_specific_thresholds(self, method_name: str) -> Dict[str, Any]:
        """Get method-specific validation thresholds."""
        method_lower = method_name.lower()
        
        # Default to general thresholds
        thresholds = self.config["warning_thresholds"].copy()
        
        # Apply method-specific thresholds
        for feature_type, type_config in self.config["feature_type_thresholds"].items():
            if feature_type in method_lower:
                thresholds.update(type_config)
                break
        
        return thresholds

    def _validate_feature_values(
        self, features_df: pd.DataFrame, thresholds: Dict[str, Any]
    ) -> List[str]:
        """Validate individual feature values."""
        self.logger.info("Validating feature values...")
        
        issues = []
        
        try:
            for col in features_df.columns:
                col_data = features_df[col]
                
                # Check for NaN values
                nan_percentage = col_data.isna().sum() / len(col_data)
                if nan_percentage > thresholds.get("max_nan_percentage", 0.15):
                    issues.append(f"High NaN percentage in {col}: {nan_percentage:.2%}")

                # Check for infinite values
                inf_count = np.isinf(col_data).sum()
                if inf_count > 0:
                    inf_percentage = inf_count / len(col_data)
                    if inf_percentage > thresholds.get("max_infinite_percentage", 0.01):
                        issues.append(f"High infinite values in {col}: {inf_percentage:.2%}")

                # Check for zero variance
                if col_data.std() == 0:
                    issues.append(f"Zero variance in feature {col}")

                # Check for constant values
                unique_ratio = col_data.nunique() / len(col_data)
                if unique_ratio < (1 - thresholds.get("max_constant_percentage", 0.8)):
                    issues.append(f"Low variance in feature {col}: {unique_ratio:.2%}")

        except Exception as e:
            self.logger.exception(f"Error in feature value validation: {e}")
            issues.append(f"Feature value validation error: {str(e)}")

        return issues

    def _validate_feature_relationships(
        self, features_df: pd.DataFrame, thresholds: Dict[str, Any]
    ) -> bool:
        """Validate relationships between features."""
        self.logger.info("Validating feature relationships...")
        
        try:
            # Check for highly correlated features
            corr_matrix = features_df.corr()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_value = abs(corr_matrix.iloc[i, j])
                    if corr_value > thresholds.get("max_correlation_threshold", 0.99):
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        high_corr_pairs.append(f"{col1} <-> {col2} ({corr_value:.3f})")

            if high_corr_pairs:
                self.logger.warning(f"High correlation pairs found: {high_corr_pairs}")

            # Check for duplicate features
            duplicate_features = features_df.columns[features_df.columns.duplicated()].tolist()
            if duplicate_features:
                self.logger.warning(f"Duplicate feature names found: {duplicate_features}")

            return True

        except Exception as e:
            self.logger.exception(f"Error in feature relationship validation: {e}")
            return False

    def _validate_input_output_consistency(
        self, features_df: pd.DataFrame, input_data_shape: Tuple[int, ...]
    ) -> bool:
        """Validate consistency between input and output."""
        self.logger.info("Validating input-output consistency...")
        
        try:
            # Check if number of samples matches
            if len(features_df) != input_data_shape[0]:
                self.logger.warning(
                    f"Sample count mismatch: input has {input_data_shape[0]}, "
                    f"output has {len(features_df)}"
                )
                return False

            return True

        except Exception as e:
            self.logger.exception(f"Error in input-output consistency validation: {e}")
            return False

    def _validate_downstream_compatibility(self, features_df: pd.DataFrame) -> bool:
        """Validate compatibility with downstream ML operations."""
        self.logger.info("Validating downstream compatibility...")
        
        try:
            # Check for sklearn compatibility
            try:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                scaler.fit(features_df)
                scaler.transform(features_df)
            except Exception as e:
                self.logger.warning(f"Sklearn compatibility issue: {e}")
                return False

            # Check for reasonable feature names
            invalid_names = []
            for col in features_df.columns:
                if not isinstance(col, str) or len(col) == 0:
                    invalid_names.append(col)

            if invalid_names:
                self.logger.warning(f"Invalid feature names found: {invalid_names}")

            return True

        except Exception as e:
            self.logger.exception(f"Error in downstream compatibility validation: {e}")
            return False

    def _calculate_quality_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall quality score."""
        score = 1.0

        # Deduct for critical issues
        score -= len(results["critical_issues"]) * 0.3

        # Deduct for warnings
        score -= len(results["warnings"]) * 0.05

        # Deduct for feature count issues
        if "feature_statistics" in results:
            total_features = results["feature_statistics"].get("total_features", 0)
            if total_features == 0:
                score -= 0.5
            elif total_features > 1000:
                score -= 0.1

        # Deduct for downstream compatibility issues
        downstream_warnings = [
            w for w in results["warnings"]
            if any(
                keyword in w.lower()
                for keyword in [
                    "sklearn", "scaling", "model", "selection",
                    "regime", "temporal", "volatility",
                ]
            )
        ]
        score -= len(downstream_warnings) * 0.02

        return float(max(0.0, min(1.0, score)))

    def _generate_output_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations: List[str] = []

        if results["output_quality_score"] < 0.8:
            recommendations.append("Consider reviewing feature engineering logic")

        if results["warnings"]:
            recommendations.append("Review warnings before using features")

        if "feature_statistics" in results:
            total_features = results["feature_statistics"].get("total_features", 0)
            if total_features > 1000:
                recommendations.append(
                    "Consider feature selection to reduce dimensionality"
                )

        # Downstream compatibility recommendations
        sklearn_warnings = [w for w in results["warnings"] if "sklearn" in w.lower()]
        if sklearn_warnings:
            recommendations.append(
                "Fix sklearn compatibility issues before model training"
            )

        scaling_warnings = [w for w in results["warnings"] if "scaling" in w.lower()]
        if scaling_warnings:
            recommendations.append(
                "Address scaling issues to prevent StandardScaler problems"
            )

        model_warnings = [w for w in results["warnings"] if "model" in w.lower()]
        if model_warnings:
            recommendations.append("Review model training compatibility issues")

        selection_warnings = [
            w for w in results["warnings"] if "selection" in w.lower()
        ]
        if selection_warnings:
            recommendations.append(
                "Consider feature engineering improvements for better selection"
            )

        regime_warnings = [w for w in results["warnings"] if "regime" in w.lower()]
        if regime_warnings:
            recommendations.append(
                "Add regime-specific features for better model performance"
            )

        temporal_warnings = [w for w in results["warnings"] if "temporal" in w.lower()]
        if temporal_warnings:
            recommendations.append("Include temporal features for time series analysis")

        volatility_warnings = [
            w for w in results["warnings"] if "volatility" in w.lower()
        ]
        if volatility_warnings:
            recommendations.append("Add volatility features for regime detection")

        return recommendations


# Convenience function for easy integration
def validate_feature_output(
    features: Union[pd.DataFrame, Dict[str, Any], np.ndarray],
    method_name: str,
    input_data_shape: Optional[Tuple[int, ...]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convenience function to validate feature output."""
    validator = FeatureOutputValidator(config)
    return validator.validate_feature_output(features, method_name, input_data_shape)
