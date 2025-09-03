"""
Feature Engineering Validation Module

This module provides comprehensive validation for engineered features,
including value range checks, NaN propagation analysis, and feature correctness verification.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.utils.pipeline_standards import (
    DataQualityLevel,
    ValidationIssue,
    ValidationResult,
)


class FeatureEngineeringValidator:
    """Validates engineered features for quality and correctness."""

    def __init__(self, logger=None):
        self.logger = logger or system_logger.getChild("FeatureEngineeringValidator")

        # Feature validation configurations
        self.feature_bounds = {
            # Price-based features
            "returns": (-0.5, 0.5),  # -50% to +50% for single period
            "log_returns": (-0.7, 0.7),  # Approximate log scale
            "price_ratio": (0.5, 2.0),  # Half to double

            # Volume features
            "volume_ratio": (0.0, 10.0),  # 0 to 10x normal
            "volume_ma_ratio": (0.1, 5.0),  # 10% to 5x MA

            # Technical indicators
            "rsi": (0.0, 100.0),  # RSI bounds
            "stochastic": (0.0, 100.0),  # Stochastic bounds
            "macd": (-1.0, 1.0),  # Normalized MACD
            "bollinger_position": (-3.0, 3.0),  # Standard deviations

            # Statistical features
            "z_score": (-5.0, 5.0),  # 5 standard deviations
            "percentile_rank": (0.0, 1.0),  # Percentile bounds
            "correlation": (-1.0, 1.0),  # Correlation bounds

            # Normalized features
            "normalized": (-3.0, 3.0),  # After standardization
            "min_max_scaled": (0.0, 1.0),  # Min-max scaling
        }

        # Feature calculation registry for validation
        self.feature_calculations = {}
        self._register_standard_calculations()

    def validate_engineered_features(
        self,
        original_df: pd.DataFrame,
        features_df: pd.DataFrame,
        feature_config: dict[str, Any],
        validate_calculations: bool = True,
        check_dependencies: bool = True,
    ) -> ValidationResult:
        """
        Comprehensive validation of engineered features.

        Args:
            original_df: Original data before feature engineering
            features_df: DataFrame with engineered features
            feature_config: Configuration used for feature engineering
            validate_calculations: Whether to validate feature calculations
            check_dependencies: Whether to check feature dependencies

        Returns:
            ValidationResult with detailed findings
        """
        self.logger.info("🔧 Validating engineered features")

        result = ValidationResult(passed=True)

        # Basic validation
        if features_df is None or features_df.empty:
            result.passed = False
            result.issues.append(ValidationIssue(
                severity=DataQualityLevel.CRITICAL,
                message="Features DataFrame is None or empty",
            ))
            return result

        validation_summary = {
            "total_features": len(features_df.columns),
            "original_columns": len(original_df.columns),
            "new_features": len(features_df.columns) - len(original_df.columns),
            "feature_validations": {},
        }

        # 1. Validate feature completeness
        self._validate_feature_completeness(
            original_df, features_df, feature_config, result, validation_summary,
        )

        # 2. Validate feature value ranges
        self._validate_feature_ranges(features_df, result, validation_summary)

        # 3. Check NaN propagation
        nan_analysis = self._analyze_nan_propagation(original_df, features_df)
        validation_summary["nan_analysis"] = nan_analysis

        if nan_analysis["excessive_nan_features"]:
            result.warnings.append(ValidationIssue(
                severity=DataQualityLevel.WARNING,
                message=f"{len(nan_analysis['excessive_nan_features'])} features have excessive NaN values",
                details=nan_analysis,
            ))

        # 4. Validate feature calculations (spot checks)
        if validate_calculations:
            calc_validation = self._validate_feature_calculations(
                original_df, features_df, sample_size=100,
            )
            validation_summary["calculation_validation"] = calc_validation

            if calc_validation["failed_validations"]:
                result.issues.append(ValidationIssue(
                    severity=DataQualityLevel.CRITICAL,
                    message=f"{len(calc_validation['failed_validations'])} features failed calculation validation",
                    details=calc_validation["failed_validations"],
                ))
                result.passed = False

        # 5. Check feature dependencies and consistency
        if check_dependencies:
            dep_validation = self._validate_feature_dependencies(features_df)
            validation_summary["dependency_validation"] = dep_validation

            if dep_validation["inconsistent_features"]:
                result.warnings.append(ValidationIssue(
                    severity=DataQualityLevel.WARNING,
                    message="Feature dependency inconsistencies detected",
                    details=dep_validation,
                ))

        # 6. Validate feature importance/relevance
        relevance_check = self._validate_feature_relevance(features_df)
        validation_summary["relevance_check"] = relevance_check

        if relevance_check["zero_variance_features"]:
            result.warnings.append(ValidationIssue(
                severity=DataQualityLevel.WARNING,
                message=f"{len(relevance_check['zero_variance_features'])} features have zero variance",
                details={"features": relevance_check["zero_variance_features"][:10]},
            ))

        # 7. Check for feature leakage
        leakage_check = self._check_feature_leakage(original_df, features_df)
        if leakage_check["potential_leakage"]:
            result.issues.append(ValidationIssue(
                severity=DataQualityLevel.CRITICAL,
                message="Potential feature leakage detected",
                details=leakage_check,
            ))
            result.passed = False

        # Calculate quality score
        critical_issues = len([i for i in result.issues if i.severity == DataQualityLevel.CRITICAL])
        warning_issues = len(result.warnings)

        result.quality_score = max(0, 1 - (critical_issues * 0.2 + warning_issues * 0.05))
        result.metadata["validation_summary"] = validation_summary

        return result

    def _validate_feature_completeness(
        self,
        original_df: pd.DataFrame,
        features_df: pd.DataFrame,
        feature_config: dict[str, Any],
        result: ValidationResult,
        summary: dict[str, Any],
    ) -> None:
        """Validate that all expected features are present."""
        # Check if original columns are preserved
        missing_original = set(original_df.columns) - set(features_df.columns)
        if missing_original:
            result.issues.append(ValidationIssue(
                severity=DataQualityLevel.CRITICAL,
                message=f"Original columns missing in features: {missing_original}",
                details={"missing_columns": list(missing_original)},
            ))
            result.passed = False

        # Check expected features based on config
        expected_features = self._get_expected_features(feature_config)
        missing_expected = expected_features - set(features_df.columns)

        if missing_expected:
            # Some features might be optional
            result.warnings.append(ValidationIssue(
                severity=DataQualityLevel.WARNING,
                message=f"{len(missing_expected)} expected features not found",
                details={"missing_features": list(missing_expected)[:20]},
            ))

        summary["expected_features"] = len(expected_features)
        summary["missing_features"] = len(missing_expected)

    def _validate_feature_ranges(
        self,
        features_df: pd.DataFrame,
        result: ValidationResult,
        summary: dict[str, Any],
    ) -> None:
        """Validate feature values are within expected ranges."""
        out_of_range_features = {}

        for column in features_df.columns:
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(features_df[column]):
                continue

            col_data = features_df[column].dropna()
            if len(col_data) == 0:
                continue

            # Check against known feature type bounds
            feature_type = self._identify_feature_type(column)
            if feature_type in self.feature_bounds:
                min_bound, max_bound = self.feature_bounds[feature_type]

                out_of_range_low = (col_data < min_bound).sum()
                out_of_range_high = (col_data > max_bound).sum()

                if out_of_range_low > 0 or out_of_range_high > 0:
                    out_of_range_features[column] = {
                        "type": feature_type,
                        "expected_range": (min_bound, max_bound),
                        "actual_range": (float(col_data.min()), float(col_data.max())),
                        "out_of_range_count": int(out_of_range_low + out_of_range_high),
                        "out_of_range_percentage": float((out_of_range_low + out_of_range_high) / len(col_data) * 100),
                    }

            # General sanity checks
            if np.isinf(col_data).any():
                result.issues.append(ValidationIssue(
                    severity=DataQualityLevel.CRITICAL,
                    message=f"Feature '{column}' contains infinite values",
                    column=column,
                    details={"infinite_count": int(np.isinf(col_data).sum())},
                ))
                result.passed = False

        if out_of_range_features:
            # Determine severity based on percentage out of range
            max_out_of_range_pct = max(f["out_of_range_percentage"] for f in out_of_range_features.values())
            severity = DataQualityLevel.CRITICAL if max_out_of_range_pct > 10 else DataQualityLevel.WARNING

            result.issues.append(ValidationIssue(
                severity=severity,
                message=f"{len(out_of_range_features)} features have values outside expected ranges",
                details={"features": out_of_range_features},
            ))

            if severity == DataQualityLevel.CRITICAL:
                result.passed = False

        summary["out_of_range_features"] = len(out_of_range_features)

    def _analyze_nan_propagation(
        self,
        original_df: pd.DataFrame,
        features_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Analyze how NaN values propagate through feature engineering."""
        original_nan_counts = original_df.isnull().sum()
        feature_nan_counts = features_df.isnull().sum()

        # Identify features with excessive NaN values
        excessive_nan_features = []
        nan_propagation_map = {}

        for column in features_df.columns:
            if column not in original_df.columns:  # New feature
                nan_percentage = feature_nan_counts[column] / len(features_df) * 100

                if nan_percentage > 50:  # More than 50% NaN
                    excessive_nan_features.append({
                        "feature": column,
                        "nan_percentage": float(nan_percentage),
                        "nan_count": int(feature_nan_counts[column]),
                    })

                # Try to identify source of NaN propagation
                potential_sources = self._identify_nan_sources(column, original_df, features_df)
                if potential_sources:
                    nan_propagation_map[column] = potential_sources

        return {
            "total_original_nans": int(original_nan_counts.sum()),
            "total_feature_nans": int(feature_nan_counts.sum()),
            "excessive_nan_features": excessive_nan_features,
            "nan_propagation_map": nan_propagation_map,
            "nan_increase_ratio": float(feature_nan_counts.sum() / max(original_nan_counts.sum(), 1)),
        }

    def _validate_feature_calculations(
        self,
        original_df: pd.DataFrame,
        features_df: pd.DataFrame,
        sample_size: int = 100,
    ) -> dict[str, Any]:
        """Validate feature calculations by recomputing a sample."""
        validation_results = {
            "validated_features": [],
            "failed_validations": [],
            "skipped_features": [],
        }

        # Sample indices for validation
        sample_indices = np.random.choice(
            len(features_df),
            size=min(sample_size, len(features_df)),
            replace=False,
        )

        for feature_name, calc_func in self.feature_calculations.items():
            if feature_name not in features_df.columns:
                continue

            try:
                # Recompute feature for sample
                expected_values = calc_func(original_df.iloc[sample_indices])
                actual_values = features_df[feature_name].iloc[sample_indices]

                # Compare values (with tolerance for floating point)
                if pd.api.types.is_numeric_dtype(expected_values):
                    close_matches = np.allclose(
                        expected_values.values,
                        actual_values.values,
                        rtol=1e-5,
                        atol=1e-8,
                        equal_nan=True,
                    )

                    if close_matches:
                        validation_results["validated_features"].append(feature_name)
                    else:
                        max_diff = np.nanmax(np.abs(expected_values.values - actual_values.values))
                        validation_results["failed_validations"].append({
                            "feature": feature_name,
                            "max_difference": float(max_diff),
                            "sample_size": len(sample_indices),
                        })
                else:
                    # For non-numeric features, check exact match
                    matches = (expected_values == actual_values).all()
                    if matches:
                        validation_results["validated_features"].append(feature_name)
                    else:
                        validation_results["failed_validations"].append({
                            "feature": feature_name,
                            "type": "non_numeric",
                            "match_percentage": float((expected_values == actual_values).mean() * 100),
                        })

            except Exception as e:
                validation_results["skipped_features"].append({
                    "feature": feature_name,
                    "error": str(e),
                })

        return validation_results

    def _validate_feature_dependencies(
        self,
        features_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Validate logical dependencies between features."""
        inconsistencies = []

        # Check OHLC relationships
        if all(col in features_df.columns for col in ["high", "low", "close"]):
            invalid_hlc = (features_df["high"] < features_df["low"]) | \
                         (features_df["close"] > features_df["high"]) | \
                         (features_df["close"] < features_df["low"])

            if invalid_hlc.any():
                inconsistencies.append({
                    "type": "ohlc_relationship",
                    "invalid_count": int(invalid_hlc.sum()),
                    "invalid_percentage": float(invalid_hlc.mean() * 100),
                })

        # Check return calculations
        if "returns" in features_df.columns and "log_returns" in features_df.columns:
            # Returns and log returns should be consistent
            expected_log_returns = np.log1p(features_df["returns"])
            log_return_diff = np.abs(expected_log_returns - features_df["log_returns"])

            if (log_return_diff > 0.001).any():
                inconsistencies.append({
                    "type": "return_consistency",
                    "max_difference": float(log_return_diff.max()),
                    "inconsistent_count": int((log_return_diff > 0.001).sum()),
                })

        # Check moving average relationships
        ma_columns = [col for col in features_df.columns if "ma_" in col or "ema_" in col]
        if len(ma_columns) >= 2:
            # Shorter MAs should be more volatile than longer MAs
            ma_pairs = []
            for i in range(len(ma_columns)):
                for j in range(i + 1, len(ma_columns)):
                    ma_pairs.append((ma_columns[i], ma_columns[j]))

            for ma1, ma2 in ma_pairs[:5]:  # Check first 5 pairs
                std1 = features_df[ma1].std()
                std2 = features_df[ma2].std()

                # Identify which should be shorter based on name
                if self._extract_ma_period(ma1) < self._extract_ma_period(ma2):
                    if std1 < std2 * 0.9:  # Shorter MA less volatile (with margin)
                        inconsistencies.append({
                            "type": "ma_volatility",
                            "features": [ma1, ma2],
                            "std_values": [float(std1), float(std2)],
                        })

        return {
            "inconsistent_features": inconsistencies,
            "consistency_score": 1.0 - min(len(inconsistencies) * 0.1, 1.0),
        }

    def _validate_feature_relevance(
        self,
        features_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Check feature relevance and quality."""
        zero_variance_features = []
        constant_features = []
        highly_correlated_pairs = []

        numeric_features = features_df.select_dtypes(include=[np.number]).columns

        for column in numeric_features:
            col_data = features_df[column].dropna()

            if len(col_data) == 0:
                continue

            # Check for zero variance
            if col_data.std() < 1e-10:
                zero_variance_features.append(column)

            # Check for constant features (all same value)
            if col_data.nunique() == 1:
                constant_features.append(column)

        # Check for highly correlated features (redundancy)
        if len(numeric_features) > 1 and len(features_df) > 100:
            corr_matrix = features_df[numeric_features].corr()

            for i in range(len(numeric_features)):
                for j in range(i + 1, len(numeric_features)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.95:  # Very high correlation
                        highly_correlated_pairs.append({
                            "feature1": numeric_features[i],
                            "feature2": numeric_features[j],
                            "correlation": float(corr_value),
                        })

        return {
            "zero_variance_features": zero_variance_features,
            "constant_features": constant_features,
            "highly_correlated_pairs": highly_correlated_pairs[:10],  # Top 10
            "redundancy_score": len(highly_correlated_pairs) / max(len(numeric_features), 1),
        }

    def _check_feature_leakage(
        self,
        original_df: pd.DataFrame,
        features_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Check for potential feature leakage."""
        potential_leakage = []

        # Check for future-looking features
        future_keywords = ["future", "next", "forward", "target", "label"]
        for column in features_df.columns:
            if any(keyword in column.lower() for keyword in future_keywords):
                # Verify if this is actually a feature (not a target)
                if column not in ["target", "label", "y"]:  # Common target names
                    potential_leakage.append({
                        "feature": column,
                        "reason": "suspicious_name",
                        "keywords_found": [k for k in future_keywords if k in column.lower()],
                    })

        # Check for perfect predictors (too good to be true)
        if "target" in features_df.columns:
            target = features_df["target"]

            for column in features_df.select_dtypes(include=[np.number]).columns:
                if column == "target":
                    continue

                # Check correlation with target
                try:
                    corr = features_df[column].corr(target)
                    if abs(corr) > 0.99:  # Near perfect correlation
                        potential_leakage.append({
                            "feature": column,
                            "reason": "perfect_correlation",
                            "correlation": float(corr),
                        })
                except:
                    pass

        return {
            "potential_leakage": len(potential_leakage) > 0,
            "suspicious_features": potential_leakage,
        }

    def _identify_feature_type(self, feature_name: str) -> str | None:
        """Identify feature type based on name patterns."""
        feature_lower = feature_name.lower()

        if "return" in feature_lower and "log" in feature_lower:
            return "log_returns"
        if "return" in feature_lower:
            return "returns"
        if "rsi" in feature_lower:
            return "rsi"
        if "macd" in feature_lower:
            return "macd"
        if "stochastic" in feature_lower or "stoch" in feature_lower:
            return "stochastic"
        if "bollinger" in feature_lower:
            return "bollinger_position"
        if "volume_ratio" in feature_lower:
            return "volume_ratio"
        if "z_score" in feature_lower or "zscore" in feature_lower:
            return "z_score"
        if "normalized" in feature_lower:
            return "normalized"
        if "scaled" in feature_lower and ("min" in feature_lower or "max" in feature_lower):
            return "min_max_scaled"
        if "corr" in feature_lower:
            return "correlation"
        if "percentile" in feature_lower or "pctl" in feature_lower:
            return "percentile_rank"

        return None

    def _identify_nan_sources(
        self,
        feature_name: str,
        original_df: pd.DataFrame,
        features_df: pd.DataFrame,
    ) -> list[str]:
        """Try to identify which original columns might cause NaN in a feature."""
        potential_sources = []

        # Simple heuristic: check which original columns have NaN patterns
        # that correlate with the feature's NaN pattern
        feature_nan_mask = features_df[feature_name].isnull()

        for col in original_df.columns:
            if col in ["timestamp", "exchange", "symbol"]:  # Skip metadata
                continue

            col_nan_mask = original_df[col].isnull()

            # Check if NaN patterns overlap significantly
            if col_nan_mask.any():
                overlap = (feature_nan_mask & col_nan_mask).sum()
                if overlap > len(features_df) * 0.1:  # More than 10% overlap
                    potential_sources.append(col)

        return potential_sources

    def _extract_ma_period(self, ma_column_name: str) -> int:
        """Extract period from MA column name."""
        import re
        numbers = re.findall(r"\d+", ma_column_name)
        return int(numbers[0]) if numbers else 999

    def _register_standard_calculations(self) -> None:
        """Register standard feature calculations for validation."""
        # Returns
        self.feature_calculations["returns"] = lambda df: df["close"].pct_change()
        self.feature_calculations["log_returns"] = lambda df: np.log1p(df["close"].pct_change())

        # Price ratios
        self.feature_calculations["high_low_ratio"] = lambda df: df["high"] / df["low"]
        self.feature_calculations["close_open_ratio"] = lambda df: df["close"] / df["open"]

        # Volume features
        self.feature_calculations["volume_ma_ratio"] = lambda df: df["volume"] / df["volume"].rolling(20).mean()

        # Add more standard calculations as needed

    def register_custom_calculation(
        self,
        feature_name: str,
        calculation_func: Callable[[pd.DataFrame], pd.Series],
    ) -> None:
        """Register a custom feature calculation for validation."""
        self.feature_calculations[feature_name] = calculation_func
