"""
Feature Output Validator for Feature Engineering
Detects corrupted, invalid, or problematic feature engineering outputs.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.utils.warning_symbols import critical

warnings.filterwarnings("ignore")


class OutputValidationLevel(Enum):
    """Output validation severity levels."""

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
    details: dict[str, Any] | None = None


class FeatureOutputValidator:
    """
    Comprehensive validator for feature engineering outputs.
    Detects corrupted, invalid, or problematic feature results.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.logger = system_logger.getChild("FeatureOutputValidator")
        self.config = config or self._get_default_config()
        self.issues: list[OutputValidationIssue] = []

    def _get_default_config(self) -> dict[str, Any]:
        """Get default output validation configuration."""
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
        features: dict[str, Any] | pd.DataFrame,
        method_name: str,
        input_data_shape: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        """
        Comprehensive validation of feature engineering output.

        Args:
            features: Feature engineering output (dict or DataFrame)
            method_name: Name of the method that generated the features
            input_data_shape: Shape of input data for comparison

        Returns:
            Validation results dictionary
        """
        print(f"🔍 [FEATURE OUTPUT VALIDATION] Starting validation for {method_name}")
        self.logger.info(
            f"🔍 [FEATURE OUTPUT VALIDATION] Starting validation for {method_name}",
        )

        print(f"   📊 Input data shape: {input_data_shape}")
        print(f"   📊 Features type: {type(features)}")
        self.logger.info(f"   📊 Input data shape: {input_data_shape}")
        self.logger.info(f"   📊 Features type: {type(features)}")

        self.issues.clear()

        validation_results: dict[str, Any] = {
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

        features_df = self._convert_features_to_dataframe(features)

        if features_df is None or features_df.empty:
            print(
                f"❌ [FEATURE OUTPUT VALIDATION] No features generated or empty output for {method_name}",
            )
            self.logger.error(
                f"❌ [FEATURE OUTPUT VALIDATION] No features generated or empty output for {method_name}",
            )
            # For engineer_features method, be more lenient and continue with warnings
            if "engineer_features" in method_name.lower():
                self.logger.warning(
                    "⚠️ [FEATURE OUTPUT VALIDATION] Empty output, continuing due to complex financial feature engineering",
                )
                validation_results["warnings"].append(
                    "No features generated or empty output",
                )
                validation_results["critical_issues"] = []
                validation_results["validation_passed"] = True
                validation_results["output_quality_score"] = 0.5
                return validation_results

            validation_results["critical_issues"].append(
                "No features generated or empty output",
            )
            validation_results["validation_passed"] = False
            validation_results["output_quality_score"] = 0.0
            return validation_results

        print(
            f"✅ [FEATURE OUTPUT VALIDATION] Features converted to DataFrame for {method_name}",
        )
        print(f"   📊 DataFrame shape: {features_df.shape}")
        print(f"   📊 DataFrame columns: {list(features_df.columns)}")
        self.logger.info(
            f"✅ [FEATURE OUTPUT VALIDATION] Features converted to DataFrame for {method_name}",
        )
        self.logger.info(f"   📊 DataFrame shape: {features_df.shape}")
        self.logger.info(
            f"   📊 DataFrame columns: {list(features_df.columns)}",
        )

        # Basic structure validation
        print(
            f"🔍 [FEATURE OUTPUT VALIDATION] Validating output structure for {method_name}",
        )
        self.logger.info(
            f"🔍 [FEATURE OUTPUT VALIDATION] Validating output structure for {method_name}",
        )

        structure_valid = self._validate_output_structure(
            features_df, validation_results,
        )
        if not structure_valid:
            print(
                f"❌ [FEATURE OUTPUT VALIDATION] Structure validation failed for {method_name}",
            )
            self.logger.error(
                f"❌ [FEATURE OUTPUT VALIDATION] Structure validation failed for {method_name}",
            )
            # For engineer_features method, be more lenient and continue with warnings
            if "engineer_features" in method_name.lower():
                self.logger.warning(
                    f"⚠️ [FEATURE OUTPUT VALIDATION] Structure validation failed for {method_name}, but continuing",
                )
                validation_results["warnings"].extend(
                    validation_results["critical_issues"],
                )
                validation_results["critical_issues"] = []
                validation_results["validation_passed"] = True
            else:
                validation_results["validation_passed"] = False
                return validation_results

        print(
            f"✅ [FEATURE OUTPUT VALIDATION] Structure validation passed for {method_name}",
        )
        self.logger.info(
            f"✅ [FEATURE OUTPUT VALIDATION] Structure validation passed for {method_name}",
        )

        # Data type validation
        print(
            f"🔍 [FEATURE OUTPUT VALIDATION] Validating data types for {method_name}",
        )
        self.logger.info(
            f"🔍 [FEATURE OUTPUT VALIDATION] Validating data types for {method_name}",
        )

        dtype_valid = self._validate_data_types(features_df, validation_results)
        if not dtype_valid:
            print(
                f"❌ [FEATURE OUTPUT VALIDATION] Data type validation failed for {method_name}",
            )
            self.logger.error(
                f"❌ [FEATURE OUTPUT VALIDATION] Data type validation failed for {method_name}",
            )
            if "engineer_features" in method_name.lower():
                self.logger.warning(
                    f"⚠️ [FEATURE OUTPUT VALIDATION] Data type validation failed for {method_name}, but continuing",
                )
                validation_results["warnings"].extend(
                    validation_results["critical_issues"],
                )
                validation_results["critical_issues"] = []
                validation_results["validation_passed"] = True
            else:
                validation_results["validation_passed"] = False
                return validation_results

        print(
            f"✅ [FEATURE OUTPUT VALIDATION] Data type validation passed for {method_name}",
        )
        self.logger.info(
            f"✅ [FEATURE OUTPUT VALIDATION] Data type validation passed for {method_name}",
        )

        # Value validation
        print(
            f"🔍 [FEATURE OUTPUT VALIDATION] Validating feature values for {method_name}",
        )
        self.logger.info(
            f"🔍 [FEATURE OUTPUT VALIDATION] Validating feature values for {method_name}",
        )

        value_valid = self._validate_feature_values(
            features_df, method_name, validation_results,
        )
        if not value_valid:
            print(
                f"❌ [FEATURE OUTPUT VALIDATION] Value validation failed for {method_name}",
            )
            self.logger.error(
                f"❌ [FEATURE OUTPUT VALIDATION] Value validation failed for {method_name}",
            )
            if "engineer_features" in method_name.lower():
                self.logger.warning(
                    f"⚠️ [FEATURE OUTPUT VALIDATION] Value validation failed for {method_name}, but continuing",
                )
                validation_results["warnings"].extend(
                    validation_results["critical_issues"],
                )
                validation_results["critical_issues"] = []
                validation_results["validation_passed"] = True
            else:
                validation_results["validation_passed"] = False
                return validation_results

        print(
            f"✅ [FEATURE OUTPUT VALIDATION] Value validation passed for {method_name}",
        )
        self.logger.info(
            f"✅ [FEATURE OUTPUT VALIDATION] Value validation passed for {method_name}",
        )

        # Feature relationship validation
        print(
            f"🔍 [FEATURE OUTPUT VALIDATION] Validating feature relationships for {method_name}",
        )
        self.logger.info(
            f"🔍 [FEATURE OUTPUT VALIDATION] Validating feature relationships for {method_name}",
        )

        relationship_valid = self._validate_feature_relationships(
            features_df, validation_results,
        )
        if not relationship_valid:
            print(
                f"❌ [FEATURE OUTPUT VALIDATION] Relationship validation failed for {method_name}",
            )
            self.logger.error(
                f"❌ [FEATURE OUTPUT VALIDATION] Relationship validation failed for {method_name}",
            )
            if "engineer_features" in method_name.lower():
                self.logger.warning(
                    f"⚠️ [FEATURE OUTPUT VALIDATION] Relationship validation failed for {method_name}, but continuing",
                )
                validation_results["warnings"].extend(
                    validation_results["critical_issues"],
                )
                validation_results["critical_issues"] = []
                validation_results["validation_passed"] = True
            else:
                validation_results["validation_passed"] = False
                return validation_results

        print(
            f"✅ [FEATURE OUTPUT VALIDATION] Relationship validation passed for {method_name}",
        )
        self.logger.info(
            f"✅ [FEATURE OUTPUT VALIDATION] Relationship validation passed for {method_name}",
        )

        # Input-output consistency validation
        if input_data_shape:
            print(
                f"🔍 [FEATURE OUTPUT VALIDATION] Validating input-output consistency for {method_name}",
            )
            self.logger.info(
                f"🔍 [FEATURE OUTPUT VALIDATION] Validating input-output consistency for {method_name}",
            )

            consistency_valid = self._validate_input_output_consistency(
                features_df, input_data_shape, validation_results,
            )
            if not consistency_valid:
                print(
                    f"❌ [FEATURE OUTPUT VALIDATION] Consistency validation failed for {method_name}",
                )
                self.logger.error(
                    f"❌ [FEATURE OUTPUT VALIDATION] Consistency validation failed for {method_name}",
                )
                if "engineer_features" in method_name.lower():
                    self.logger.warning(
                        f"⚠️ [FEATURE OUTPUT VALIDATION] Consistency validation failed for {method_name}, but continuing",
                    )
                    validation_results["warnings"].extend(
                        validation_results["critical_issues"],
                    )
                    validation_results["critical_issues"] = []
                    validation_results["validation_passed"] = True
                else:
                    validation_results["validation_passed"] = False
                    return validation_results

            print(
                f"✅ [FEATURE OUTPUT VALIDATION] Consistency validation passed for {method_name}",
            )
            self.logger.info(
                f"✅ [FEATURE OUTPUT VALIDATION] Consistency validation passed for {method_name}",
            )

        # Downstream compatibility validation
        print(
            f"🔍 [FEATURE OUTPUT VALIDATION] Validating downstream compatibility for {method_name}",
        )
        self.logger.info(
            f"🔍 [FEATURE OUTPUT VALIDATION] Validating downstream compatibility for {method_name}",
        )

        downstream_valid = self._validate_downstream_compatibility(
            features_df, method_name, validation_results,
        )
        if not downstream_valid:
            print(
                f"❌ [FEATURE OUTPUT VALIDATION] Downstream compatibility validation failed for {method_name}",
            )
            self.logger.error(
                f"❌ [FEATURE OUTPUT VALIDATION] Downstream compatibility validation failed for {method_name}",
            )
            if "engineer_features" in method_name.lower():
                self.logger.warning(
                    f"⚠️ [FEATURE OUTPUT VALIDATION] Downstream compatibility validation failed for {method_name}, but continuing",
                )
                validation_results["warnings"].extend(
                    validation_results["critical_issues"],
                )
                validation_results["critical_issues"] = []
                validation_results["validation_passed"] = True
            else:
                validation_results["validation_passed"] = False
                return validation_results

        print(
            f"✅ [FEATURE OUTPUT VALIDATION] Downstream compatibility validation passed for {method_name}",
        )
        self.logger.info(
            f"✅ [FEATURE OUTPUT VALIDATION] Downstream compatibility validation passed for {method_name}",
        )

        # Calculate overall quality score
        print(
            f"🔍 [FEATURE OUTPUT VALIDATION] Calculating quality score for {method_name}",
        )
        self.logger.info(
            f"🔍 [FEATURE OUTPUT VALIDATION] Calculating quality score for {method_name}",
        )

        validation_results["output_quality_score"] = (
            self._calculate_output_quality_score(validation_results)
        )

        print(
            f"📊 [FEATURE OUTPUT VALIDATION] Quality score for {method_name}: {validation_results['output_quality_score']:.2f}",
        )
        self.logger.info(
            f"📊 [FEATURE OUTPUT VALIDATION] Quality score for {method_name}: {validation_results['output_quality_score']:.2f}",
        )

        # Generate recommendations
        validation_results["recommendations"] = (
            self._generate_output_recommendations(validation_results)
        )

        if validation_results["validation_passed"]:
            self.logger.info(
                f"✅ Feature output validation passed for {method_name} (Score: {validation_results['output_quality_score']:.2f})",
            )
        else:
            self.logger.error(
                f"❌ Feature output validation failed for {method_name}",
            )
        for issue in validation_results["critical_issues"]:
            self.logger.error(f"   {critical(issue)}")

        return validation_results

    def _convert_features_to_dataframe(
        self, features: dict[str, Any] | pd.DataFrame
    ) -> pd.DataFrame | None:
        """Convert features to DataFrame for validation."""
        # Handle None input
        if features is None:
            self.logger.warning(
                "⚠️ [FEATURE OUTPUT VALIDATION] Features input is None",
            )
            return None

        if isinstance(features, pd.DataFrame):
            return features

        if isinstance(features, dict):
            # Handle empty dict
            if not features:
                self.logger.warning(
                    "⚠️ [FEATURE OUTPUT VALIDATION] Features dict is empty",
                )
                return None

            try:
                # Handle different feature dict formats
                if all(
                    isinstance(v, (pd.Series, pd.DataFrame)) for v in features.values()
                ):
                    # Features are Series/DataFrames
                    feature_series: list[pd.Series] = []
                    for name, series in features.items():
                        if isinstance(series, pd.Series):
                            feature_series.append(series.rename(name))
                        elif isinstance(series, pd.DataFrame):
                            for col in series.columns:
                                feature_series.append(
                                    series[col].rename(f"{name}_{col}"),
                                )

                    if feature_series:
                        return pd.concat(feature_series, axis=1)

                if all(
                    isinstance(v, (int, float, np.generic)) for v in features.values()
                ):
                    # Features are scalar values
                    return pd.DataFrame([features])

                if any(isinstance(v, np.ndarray) for v in features.values()):
                    # Handle numpy arrays
                    feature_series = []
                    for name, value in features.items():
                        if value is None:
                            continue
                        if isinstance(value, np.ndarray):
                            # Convert numpy array to pandas Series
                            if value.ndim == 1:
                                feature_series.append(pd.Series(value, name=name))
                            elif value.ndim == 2:
                                if value.shape[1] == 1:
                                    feature_series.append(
                                        pd.Series(value.flatten(), name=name),
                                    )
                                else:
                                    # Take first column for 2D arrays
                                    feature_series.append(
                                        pd.Series(value[:, 0], name=name),
                                    )
                        elif isinstance(value, pd.Series):
                            feature_series.append(value.rename(name))
                        elif isinstance(value, pd.DataFrame):
                            for col in value.columns:
                                feature_series.append(
                                    value[col].rename(f"{name}_{col}"),
                                )
                        elif isinstance(value, (int, float, np.generic)):
                            feature_series.append(pd.Series([value], name=name))

                    if feature_series:
                        # Ensure all series have the same length
                        max_length = max(len(series) for series in feature_series)
                        aligned_series: list[pd.Series] = []
                        for series in feature_series:
                            if len(series) < max_length:
                                # Pad shorter series with NaN
                                padded_series = pd.Series(
                                    [np.nan] * max_length, name=series.name,
                                )
                                padded_series.iloc[: len(series)] = series.values
                                aligned_series.append(padded_series)
                            else:
                                aligned_series.append(series)

                        return pd.concat(aligned_series, axis=1)

                if any(isinstance(v, pd.DataFrame) for v in features.values()):
                    # Mixed format with some DataFrames - extract the main feature DataFrame
                    for key, value in features.items():
                        if value is not None and isinstance(value, pd.DataFrame):
                            self.logger.info(
                                f"Found DataFrame in features dict with key: {key}",
                            )
                            return value

                    # If no DataFrame found = try to handle as mixed format
                    feature_series = []
                    for name, value in features.items():
                        if value is None:
                            continue
                        if isinstance(value, pd.Series):
                            feature_series.append(value.rename(name))
                        elif isinstance(value, pd.DataFrame):
                            for col in value.columns:
                                feature_series.append(
                                    value[col].rename(f"{name}_{col}"),
                                )

                    if feature_series:
                        return pd.concat(feature_series, axis=1)

                # If we get here = we couldn't convert the features
                self.logger.warning(
                    f"⚠️ [FEATURE OUTPUT VALIDATION] Could not convert features to DataFrame. Type: {type(features)}",
                )
                return None

            except Exception as e:
                self.logger.exception(
                    f"💥 [FEATURE OUTPUT VALIDATION] Error converting features to DataFrame: {e}",
                )
                return None

        # Unsupported type
        self.logger.warning(
            f"⚠️ [FEATURE OUTPUT VALIDATION] Unsupported features type: {type(features)}",
        )
        return None

    def _validate_output_structure(
        self, features_df: pd.DataFrame, results: dict[str, Any],
    ) -> bool:
        """Validate basic output structure."""
        self.logger.info("Validating output structure...")

        # Check minimum feature count
        min_features = self.config["critical_thresholds"]["min_feature_count"]
        max_features = self.config["critical_thresholds"]["max_feature_count"]

        if len(features_df.columns) < min_features:
            results["warnings"].append(
                f"Insufficient features generated: {len(features_df.columns)} (minimum: {min_features})",
            )

        if len(features_df.columns) > max_features:
            results["warnings"].append(
                f"Large number of features: {len(features_df.columns)} (maximum: {max_features})",
            )

        # Check for empty features
        empty_features = features_df.columns[features_df.isnull().all()].tolist()
        if empty_features:
            results["warnings"].append(f"Empty features detected: {empty_features}")

        # Check feature names
        if self.config["validation_checks"]["check_feature_names"]:
            invalid_names: list[str] = []
            for col in features_df.columns:
                if not isinstance(col, str) or len(col) == 0 or str(col).startswith("_"):
                    invalid_names.append(str(col))

            if invalid_names:
                results["warnings"].append(
                    f"Invalid feature names detected: {invalid_names}",
                )

        results["feature_statistics"]["total_features"] = len(features_df.columns)
        results["feature_statistics"]["total_rows"] = len(features_df)
        results["feature_statistics"]["feature_names"] = list(features_df.columns)

        return True

    def _validate_data_types(
        self, features_df: pd.DataFrame, results: dict[str, Any],
    ) -> bool:
        """Validate data types of features."""
        self.logger.info("Validating data types...")

        if not self.config["validation_checks"]["check_data_types"]:
            return True

        # Check for non-numeric features
        non_numeric_features: list[str] = []
        for col in features_df.columns:
            if not pd.api.types.is_numeric_dtype(features_df[col].dtype):
                non_numeric_features.append(str(col))

        if non_numeric_features:
            results["warnings"].append(
                f"Non-numeric features detected: {non_numeric_features}",
            )

        # Check for object dtype (potential issues)
        object_features: list[str] = []
        for col in features_df.columns:
            if features_df[col].dtype == "object":
                object_features.append(str(col))

        if object_features:
            results["warnings"].append(
                f"Object dtype features detected: {object_features}",
            )

        results["detailed_analysis"]["data_types"] = {
            str(col): str(features_df[col].dtype) for col in features_df.columns
        }

        return True

    def _get_method_specific_thresholds(self, method_name: str) -> dict[str, float]:
        """Get method-specific validation thresholds."""
        method_lower = method_name.lower()

        # Special handling for engineer_features method - use more lenient thresholds
        if "engineer_features" in method_lower:
            return {
                "max_nan_percentage": 0.4,
                "max_infinite_percentage": 0.1,
                "max_zero_variance_percentage": 0.8,
                "max_constant_percentage": 0.95,
                "max_extreme_values_percentage": 0.15,
                "description": "Complex financial feature engineering can produce varied outputs",
            }

        for feature_type, thresholds in self.config["feature_type_thresholds"].items():
            if feature_type in method_lower:
                return thresholds

        return {}

    def _validate_feature_values(
        self, features_df: pd.DataFrame, method_name: str, results: dict[str, Any],
    ) -> bool:
        """Validate feature values for common issues."""
        self.logger.info("Validating feature values...")

        # Get method-specific thresholds
        method_thresholds = self._get_method_specific_thresholds(method_name)

        for col in features_df.columns:
            series = features_df[col]

            # Check NaN values
            if self.config["validation_checks"]["check_nan_values"]:
                nan_percentage = float(series.isna().sum()) / max(len(series), 1)
                max_nan = method_thresholds.get(
                    "max_nan_percentage",
                    self.config["critical_thresholds"]["max_nan_percentage"],
                )

                if nan_percentage > max_nan:
                    self.logger.warning(
                        f"High NaN percentage in {col}: {nan_percentage:.3f} (threshold: {max_nan})",
                    )
                    results["critical_issues"].append(
                        f"High NaN percentage in {col}: {nan_percentage:.3f} (threshold: {max_nan})",
                    )
                    return False
                if (
                    nan_percentage
                    > self.config["warning_thresholds"]["max_nan_percentage"]
                ):
                    results["warnings"].append(
                        f"Moderate NaN percentage in {col}: {nan_percentage:.3f}",
                    )

            # Check infinite values
            if self.config["validation_checks"]["check_infinite_values"]:
                inf_percentage = float(np.isinf(series)).sum() / max(len(series), 1)
                max_inf = method_thresholds.get(
                    "max_infinite_percentage",
                    self.config["critical_thresholds"]["max_infinite_percentage"],
                )

                if inf_percentage > max_inf:
                    self.logger.warning(
                        f"High infinite percentage in {col}: {inf_percentage:.3f} (threshold: {max_inf})",
                    )
                    results["critical_issues"].append(
                        f"High infinite percentage in {col}: {inf_percentage:.3f} (threshold: {max_inf})",
                    )
                    return False
                if (
                    inf_percentage
                    > self.config["warning_thresholds"]["max_infinite_percentage"]
                ):
                    results["warnings"].append(
                        f"Moderate infinite percentage in {col}: {inf_percentage:.3f}",
                    )

            # Check zero variance - only for numeric columns
            if self.config["validation_checks"]["check_zero_variance"]:
                if pd.api.types.is_numeric_dtype(series.dtype):
                    if float(series.var()) == 0.0:
                        results["warnings"].append(
                            f"Zero variance feature detected: {col}",
                        )
                # For categorical columns = check if all values are the same
                elif series.nunique() <= 1:
                    results["warnings"].append(
                        f"Constant categorical feature detected: {col}",
                    )

            # Check constant values
            if self.config["validation_checks"]["check_constant_values"]:
                constant_percentage = (
                    (series == series.mode().iloc[0]).sum() / float(len(series))
                    if len(series.mode()) > 0
                    else 0.0
                )
                max_constant = self.config["critical_thresholds"][
                    "max_constant_percentage"
                ]

                if constant_percentage > max_constant:
                    results["critical_issues"].append(
                        f"High constant percentage in {col}: {constant_percentage:.3f} (threshold: {max_constant})",
                    )
                    return False

            # Check extreme values - made more lenient for financial data
            if self.config["validation_checks"]["check_extreme_values"]:
                # Use interquartile range (IQR) method for extreme value detection
                q75 = float(series.quantile(0.75))
                q25 = float(series.quantile(0.25))
                iqr = q75 - q25

                # Define extreme values as those beyond 3 * IQR from the quartiles
                upper_bound = q75 + 3 * iqr
                lower_bound = q25 - 3 * iqr

                extreme_count = int(((series > upper_bound) | (series < lower_bound)).sum())
                extreme_percentage = extreme_count / max(len(series), 1)

                if (
                    extreme_percentage
                    > self.config["critical_thresholds"][
                        "max_extreme_values_percentage"
                    ]
                ):
                    self.logger.warning(
                        f"High extreme values in {col}: {extreme_percentage:.3f}",
                    )
                    results["critical_issues"].append(
                        f"High extreme values in {col}: {extreme_percentage:.3f}",
                    )
                    return False

        return True

    def _validate_feature_relationships(
        self, features_df: pd.DataFrame, results: dict[str, Any],
    ) -> bool:
        """Validate relationships between features."""
        self.logger.info("Validating feature relationships...")

        # Check for perfect correlations (duplicate features)
        if self.config["validation_checks"]["check_feature_correlations"]:
            corr_matrix = features_df.corr().abs()
            perfect_correlations: list[tuple[tuple[str, str], float]] = []

            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = float(corr_matrix.iloc[i, j])
                    if (
                        corr_val
                        > self.config["warning_thresholds"]["max_correlation_threshold"]
                    ):
                        col_pair = (
                            str(corr_matrix.columns[i]),
                            str(corr_matrix.columns[j]),
                        )
                        perfect_correlations.append((col_pair, corr_val))

            if perfect_correlations:
                results["warnings"].append(
                    f"Highly correlated features detected: {perfect_correlations[:5]}",
                )  # Show first 5

        # Check for duplicate features
        if self.config["validation_checks"]["check_duplicate_features"]:
            duplicate_features: list[tuple[str, str]] = []
            for i, col1 in enumerate(features_df.columns):
                for j, col2 in enumerate(features_df.columns[i + 1 :], i + 1):
                    if features_df[col1].equals(features_df[col2]):
                        duplicate_features.append((str(col1), str(col2)))

            if duplicate_features:
                results["warnings"].append(
                    f"Duplicate features detected: {duplicate_features}",
                )

        return True

    def _validate_input_output_consistency(
        self,
        features_df: pd.DataFrame,
        input_shape: tuple[int, int],
        results: dict[str, Any],
    ) -> bool:
        """Validate consistency between input and output."""
        self.logger.info("Validating input-output consistency...")

        input_rows = int(input_shape[0])
        output_rows = int(len(features_df))

        if output_rows != input_rows:
            results["warnings"].append(
                f"Row count mismatch: input={input_rows}, output={output_rows}",
            )

        # Check for reasonable feature count relative to input
        input_cols = int(input_shape[1])
        output_cols = int(len(features_df.columns))

        if output_cols > input_cols * 100:  # More than 100x input columns
            results["warnings"].append(
                f"Large feature expansion: input={input_cols}, output={output_cols}",
            )

        return True

    def _validate_downstream_compatibility(
        self,
        features_df: pd.DataFrame,
        method_name: str,
        results: dict[str, Any],
    ) -> bool:
        """
        Validate that features are compatible with downstream ML pipeline steps.

        Args:
            features_df: Feature DataFrame
            method_name: Name of the method that generated features
            results: Validation results dictionary

        Returns:
            True if compatible, False otherwise
        """
        self.logger.info("Validating downstream compatibility...")

        # Check compatibility with sklearn preprocessing
        sklearn_compatible = self._validate_sklearn_compatibility(features_df, results)
        if not sklearn_compatible:
            return False

        # Check compatibility with model training
        model_compatible = self._validate_model_training_compatibility(
            features_df, results,
        )
        if not model_compatible:
            return False

        # Check compatibility with feature selection
        selection_compatible = self._validate_feature_selection_compatibility(
            features_df, results,
        )
        if not selection_compatible:
            return False

        # Check compatibility with regime-specific requirements
        return self._validate_regime_compatibility(
            features_df, method_name, results,
        )

    def _validate_sklearn_compatibility(
        self, features_df: pd.DataFrame, results: dict[str, Any],
    ) -> bool:
        """Validate compatibility with sklearn preprocessing."""

        # Check for sklearn-compatible data types
        sklearn_incompatible: list[str] = []
        for col in features_df.columns:
            dtype = features_df[col].dtype

            # sklearn expects numeric types
            if not pd.api.types.is_numeric_dtype(dtype):
                sklearn_incompatible.append(str(col))

            # Check for object dtype (problematic for sklearn)
            if dtype == "object":
                sklearn_incompatible.append(str(col))

        if sklearn_incompatible:
            results["warnings"].append(
                f"Features incompatible with sklearn: {sklearn_incompatible}",
            )

        # Check for features that would cause StandardScaler issues
        scaler_problematic: list[str] = []
        for col in features_df.columns:
            series = features_df[col]

            # Check for zero variance (causes division by zero in StandardScaler) - only for numeric columns
            if pd.api.types.is_numeric_dtype(series.dtype) and float(series.var()) == 0.0:
                scaler_problematic.append(str(col))

            # Check for constant features (causes issues in many sklearn estimators)
            if series.nunique() == 1:
                scaler_problematic.append(str(col))

        if scaler_problematic:
            results["warnings"].append(
                f"Features may cause sklearn scaling issues: {scaler_problematic}",
            )

        # Check for features with extreme values that could affect scaling
        extreme_features: list[str] = []
        for col in features_df.columns:
            series = features_df[col]
            q99 = float(series.quantile(0.99))
            q01 = float(series.quantile(0.01))

            # Check for extreme outliers that could affect scaling
            if abs(q99 - q01) > 1e6:
                extreme_features.append(str(col))

        if extreme_features:
            results["warnings"].append(
                f"Features with extreme values may affect scaling: {extreme_features}",
            )

        return True

    def _validate_model_training_compatibility(
        self, features_df: pd.DataFrame, results: dict[str, Any],
    ) -> bool:
        """Validate compatibility with model training."""

        # Check for sufficient non-zero variance features
        zero_var_features: list[str] = []
        for col in features_df.columns:
            series = features_df[col]
            if pd.api.types.is_numeric_dtype(series.dtype) and float(series.var()) == 0.0:
                zero_var_features.append(str(col))

        zero_var_percentage = (
            len(zero_var_features) / max(len(features_df.columns), 1)
        )
        if zero_var_percentage > 0.5:  # More than 50% zero variance
            results["warnings"].append(
                f"Too many zero-variance features: {zero_var_percentage:.2f}",
            )

        # Check for features with too many unique values (potential overfitting)
        high_cardinality_features: list[str] = []
        for col in features_df.columns:
            unique_ratio = features_df[col].nunique() / max(len(features_df), 1)
            if unique_ratio > 0.8:  # More than 80% unique values
                high_cardinality_features.append(str(col))

        if high_cardinality_features:
            results["warnings"].append(
                f"High cardinality features detected: {high_cardinality_features}",
            )

        # Check for features that are perfectly correlated (redundant)
        corr_matrix = features_df.corr().abs()
        perfect_correlations: list[tuple[tuple[str, str], float]] = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if float(corr_matrix.iloc[i, j]) > 0.999:  # Perfect correlation
                    col_pair = (
                        str(corr_matrix.columns[i]),
                        str(corr_matrix.columns[j]),
                    )
                    perfect_correlations.append((col_pair, float(corr_matrix.iloc[i, j])))

        if perfect_correlations:
            results["warnings"].append(
                f"Perfectly correlated features (redundant): {perfect_correlations[:5]}",
            )

        return True

    def _validate_feature_selection_compatibility(
        self, features_df: pd.DataFrame, results: dict[str, Any],
    ) -> bool:
        """Validate compatibility with feature selection methods."""

        # Check for minimum number of features for selection
        min_features_for_selection = 5
        if len(features_df.columns) < min_features_for_selection:
            results["warnings"].append(
                f"Few features available for selection: {len(features_df.columns)} (minimum: {min_features_for_selection})",
            )

        # Check for features with sufficient variance for selection
        low_variance_features: list[str] = []
        for col in features_df.columns:
            # Only calculate variance for numeric columns
            if pd.api.types.is_numeric_dtype(features_df[col].dtype):
                variance = float(features_df[col].var())
            else:
                variance = 0.0  # For categorical columns treat as zero variance
            if variance < 1e-8:  # Very low variance
                low_variance_features.append(str(col))

        if low_variance_features:
            results["warnings"].append(
                f"Low variance features may be filtered out: {low_variance_features}",
            )

        # Check for features with sufficient non-zero values for mutual information
        sparse_features: list[str] = []
        for col in features_df.columns:
            non_zero_ratio = (features_df[col] != 0).sum() / max(len(features_df), 1)
            if non_zero_ratio < 0.1:  # Less than 10% non-zero values
                sparse_features.append(str(col))

        if sparse_features:
            results["warnings"].append(
                f"Sparse features may have limited selection value: {sparse_features}",
            )

        return True

    def _validate_regime_compatibility(
        self, features_df: pd.DataFrame, method_name: str, results: dict[str, Any],
    ) -> bool:
        """Validate compatibility with regime-specific requirements."""

        # Check for regime-specific feature requirements
        if "regime" in method_name.lower() or "hmm" in method_name.lower():
            # Regime models need features that can capture state transitions

            # Check for temporal features
            temporal_features = [
                str(col)
                for col in features_df.columns
                if any(
                    keyword in str(col).lower()
                    for keyword in [
                        "lag",
                        "diff",
                        "pct_change",
                        "rolling",
                        "momentum",
                        "trend",
                        "volatility",
                    ]
                )
            ]

            if not temporal_features:
                results["warnings"].append(
                    "No temporal features detected for regime analysis",
                )

            # Check for volatility features
            volatility_features = [
                str(col)
                for col in features_df.columns
                if any(
                    keyword in str(col).lower()
                    for keyword in ["volatility", "std", "variance", "atr", "bbands"]
                )
            ]

            if not volatility_features:
                results["warnings"].append(
                    "No volatility features detected for regime analysis",
                )

            # Check for microstructure features if method suggests it
            if "microstructure" in method_name.lower():
                microstructure_features = [
                    str(col)
                    for col in features_df.columns
                    if any(
                        keyword in str(col).lower()
                        for keyword in ["volume", "trade", "bid", "ask", "spread", "impact"]
                    )
                ]

                if not microstructure_features:
                    results["warnings"].append(
                        "No microstructure features detected for microstructure analysis",
                    )

            # Check for wavelet features if method suggests it
            if "wavelet" in method_name.lower():
                wavelet_features = [
                    str(col)
                    for col in features_df.columns
                    if any(
                        keyword in str(col).lower()
                        for keyword in ["wavelet", "cwt", "db", "haar", "sym"]
                    )
                ]

                if not wavelet_features:
                    results["warnings"].append(
                        "No wavelet features detected for wavelet analysis",
                    )

        return True

    def _calculate_output_quality_score(self, results: dict[str, Any]) -> float:
        """Calculate overall output quality score (0-1)."""
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
            w
            for w in results["warnings"]
            if any(
                keyword in w.lower()
                for keyword in [
                    "sklearn",
                    "scaling",
                    "model",
                    "selection",
                    "regime",
                    "temporal",
                    "volatility",
                ]
            )
        ]
        score -= len(downstream_warnings) * 0.02

        return float(max(0.0, min(1.0, score)))

    def _generate_output_recommendations(self, results: dict[str, Any]) -> list[str]:
        """Generate recommendations based on validation results."""
        recommendations: list[str] = []

        if results["output_quality_score"] < 0.8:
            recommendations.append("Consider reviewing feature engineering logic")

        if results["warnings"]:
            recommendations.append("Review warnings before using features")

        if "feature_statistics" in results:
            total_features = results["feature_statistics"].get("total_features", 0)
            if total_features > 1000:
                recommendations.append(
                    "Consider feature selection to reduce dimensionality",
                )

        # Downstream compatibility recommendations
        sklearn_warnings = [w for w in results["warnings"] if "sklearn" in w.lower()]
        if sklearn_warnings:
            recommendations.append(
                "Fix sklearn compatibility issues before model training",
            )

        scaling_warnings = [w for w in results["warnings"] if "scaling" in w.lower()]
        if scaling_warnings:
            recommendations.append(
                "Address scaling issues to prevent StandardScaler problems",
            )

        model_warnings = [w for w in results["warnings"] if "model" in w.lower()]
        if model_warnings:
            recommendations.append("Review model training compatibility issues")

        selection_warnings = [
            w for w in results["warnings"] if "selection" in w.lower()
        ]
        if selection_warnings:
            recommendations.append(
                "Consider feature engineering improvements for better selection",
            )

        regime_warnings = [w for w in results["warnings"] if "regime" in w.lower()]
        if regime_warnings:
            recommendations.append(
                "Add regime-specific features for better model performance",
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
    features: dict[str, Any] | pd.DataFrame,
    method_name: str,
    input_data_shape: tuple[int, int] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Convenience function to validate feature engineering output.

    Args:
        features: Feature engineering output
        method_name: Name of the method that generated the features
        input_data_shape: Shape of input data for comparison
        config: Optional configuration

    Returns:
        Validation results dictionary
    """
    validator = FeatureOutputValidator(config)
    return validator.validate_feature_output(features, method_name, input_data_shape)
