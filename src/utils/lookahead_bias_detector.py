"""
Lookahead Bias Detection System

This module provides comprehensive detection and prevention of lookahead bias
in financial machine learning pipelines.

Detects various types of lookahead bias:
    1. Future information leakage in features
    2. Improper temporal alignment
    3. Incorrect train / test splits
    4. Feature - target correlation issues
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.utils.error_handler import handle_data_processing_errors, handle_errors


class LookaheadBiasDetector:
    """
    Comprehensive lookahead bias detection and prevention system.
    
    This class provides methods to detect various forms of lookahead bias
    that commonly occur in financial machine learning applications.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the LookaheadBiasDetector with configuration."""
        self.config = config or {}
        self.logger = system_logger.getChild("LookaheadBiasDetector")
        self.detected_issues: List[str] = []
        self.critical_issues: List[str] = []

        # Configuration for detection strictness
        self.strict_mode: bool = self.config.get("strict_mode", False)
        self.warning_threshold: int = self.config.get("warning_threshold", 50)
        self.correlation_threshold: float = self.config.get("correlation_threshold", 0.8)
        self.temporal_tolerance: int = self.config.get("temporal_tolerance", 1)  # days

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="lookaheadbiasdetector initialization",
    )
    async def initialize(self) -> bool:
        """Initialize LookaheadBiasDetector."""
        try:
            self.logger.info("🚀 Initializing LookaheadBiasDetector...")
            self.is_initialized = True
            self.logger.info("✅ LookaheadBiasDetector initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing LookaheadBiasDetector: {e}")
            return False

    @handle_data_processing_errors(
        default_return={}, 
        context="LookaheadBiasDetector.detect_feature_lookahead_bias"
    )
    def detect_feature_lookahead_bias(
        self,
        features_df: pd.DataFrame,
        target_series: pd.Series,
        timestamp_col: Optional[str] = None,
        feature_engineering_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Detect lookahead bias in features.
        
        Args:
            features_df: DataFrame containing features
            target_series: Series containing target variable
            timestamp_col: Optional column name for timestamps
            feature_engineering_code: Optional code to analyze for implementation issues
            
        Returns:
            Dictionary containing detection results and recommendations
        """
        results: Dict[str, Any] = {
            "lookahead_bias_detected": False,
            "critical_issues": [],
            "warnings": [],
            "feature_correlations": {},
            "temporal_issues": [],
            "recommendations": [],
            "implementation_analysis": {},
        }

        try:
            # 1. Check for perfect correlations (indicator of lookahead bias)
            self._check_perfect_correlations(features_df, target_series, results)

            # 2. Check temporal alignment if timestamps available
            if timestamp_col and timestamp_col in features_df.columns:
                self._check_temporal_alignment(
                    features_df,
                    target_series,
                    timestamp_col,
                    results,
                )

            # 3. Check for suspicious feature importance patterns
            self._check_feature_importance_patterns(features_df, target_series, results)

            # 4. Enhanced rolling window analysis
            self._check_rolling_window_issues(features_df, results)

            # 5. Analyze actual implementation if code provided
            if feature_engineering_code:
                self._analyze_implementation(
                    feature_engineering_code,
                    features_df,
                    results,
                )

            # 6. Generate recommendations
            self._generate_recommendations(results)

            # Log results
            if results["critical_issues"]:
                self.logger.critical(
                    f"🚨 LOOKAHEAD BIAS DETECTED: {len(results['critical_issues'])} critical issues"
                )
                for issue in results["critical_issues"]:
                    self.logger.critical(f"   ❌ {issue}")

            if results["warnings"]:
                self.logger.warning(
                    f"⚠️ LOOKAHEAD BIAS WARNINGS: {len(results['warnings'])} warnings"
                )
                for warning_msg in results["warnings"]:
                    self.logger.warning(f"   ⚠️ {warning_msg}")

            return results

        except Exception as e:
            self.logger.exception(f"Error in lookahead bias detection: {e}")
            results["error"] = str(e)
            return results

    def _check_perfect_correlations(
        self, 
        features_df: pd.DataFrame, 
        target_series: pd.Series, 
        results: Dict[str, Any]
    ) -> None:
        """Check for suspicious correlations between features and target."""
        # Calculate correlations with target
        correlations: Dict[str, float] = {}
        for col in features_df.columns:
            if col != target_series.name:
                try:
                    corr = float(features_df[col].corr(target_series))
                    if not pd.isna(corr):
                        correlations[col] = corr
                except Exception:
                    continue

        # Check for suspicious correlations
        for feature, corr in correlations.items():
            abs_corr = abs(corr)

            if abs_corr > 0.98:
                results["critical_issues"].append(
                    f"PERFECT CORRELATION: {feature} has {corr:.4f} correlation with target "
                    f"(indicates lookahead bias)"
                )
                results["lookahead_bias_detected"] = True

            elif abs_corr > 0.9:
                results["warnings"].append(
                    f"HIGH CORRELATION: {feature} has {corr:.4f} correlation with target "
                    f"(potential lookahead bias)"
                )

            elif abs_corr > 0.7:
                results["warnings"].append(
                    f"MODERATE CORRELATION: {feature} has {corr:.4f} correlation with target "
                    f"(investigate further)"
                )

        results["feature_correlations"] = correlations

    def _check_temporal_alignment(
        self,
        features_df: pd.DataFrame,
        target_series: pd.Series,
        timestamp_col: str,
        results: Dict[str, Any],
    ) -> None:
        """Check temporal alignment between features and target."""
        try:
            # Ensure timestamps are datetime
            timestamps = pd.to_datetime(features_df[timestamp_col])

            # Check if features and target have same lengths
            if len(features_df) != len(target_series):
                results["critical_issues"].append(
                    "TEMPORAL MISMATCH: Features and target have different lengths"
                )
                results["lookahead_bias_detected"] = True
                return

            # Check for future information leakage in rolling features
            self._check_rolling_feature_timing(features_df, timestamps, results)

        except Exception as e:
            results["warnings"].append(
                f"Could not perform temporal alignment check: {e}"
            )

    def _check_rolling_feature_timing(
        self,
        features_df: pd.DataFrame,
        timestamps: pd.Series,
        results: Dict[str, Any],
    ) -> None:
        """Check for suspicious rolling feature timing patterns."""
        # Look for common rolling feature patterns
        rolling_patterns = [
            "volatility_",
            "momentum_",
            "rsi_",
            "ma_",
            "ema_",
            "rolling_",
            "std_",
            "mean_",
            "corr_",
        ]

        suspicious_features: List[str] = []
        for col in features_df.columns:
            for pattern in rolling_patterns:
                if pattern in col.lower():
                    suspicious_features.append(col)
                    break

        if suspicious_features:
            results["warnings"].append(
                f"ROLLING FEATURES DETECTED: {len(suspicious_features)} features may need lagging. "
                f"Check: {suspicious_features[:5]}"
            )

    def _check_feature_importance_patterns(
        self,
        features_df: pd.DataFrame,
        target_series: pd.Series,
        results: Dict[str, Any],
    ) -> None:
        """Check for suspicious feature importance patterns."""
        # Calculate feature importance using correlation as proxy
        correlations: Dict[str, float] = results.get("feature_correlations", {})

        if not correlations:
            return

        # Sort by absolute correlation
        sorted_features = sorted(
            correlations.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        # Check for dominance by few features
        if len(sorted_features) >= 2:
            top_corr = abs(sorted_features[0][1])
            second_corr = abs(sorted_features[1][1])

            # If top 2 features have very high correlations
            if top_corr > 0.8 and second_corr > 0.8:
                results["critical_issues"].append(
                    f"FEATURE DOMINANCE: Top 2 features have correlations {top_corr:.4f} and {second_corr:.4f} "
                    f"({sorted_features[0][0]}, {sorted_features[1][0]}) - likely lookahead bias"
                )
                results["lookahead_bias_detected"] = True

            # If top feature dominates
            if top_corr > 0.9:
                results["critical_issues"].append(
                    f"FEATURE DOMINANCE: Top feature {sorted_features[0][0]} has correlation {top_corr:.4f} "
                    f"- likely lookahead bias"
                )
                results["lookahead_bias_detected"] = True

    def _check_rolling_window_issues(
        self, 
        features_df: pd.DataFrame, 
        results: Dict[str, Any]
    ) -> None:
        """Enhanced rolling window analysis for lookahead bias detection."""
        # Enhanced patterns for different types of features
        rolling_patterns: Dict[str, List[str]] = {
            "volatility": ["volatility", "std", "atr"],
            "momentum": ["momentum", "roc", "rsi", "stoch"],
            "moving_averages": ["ma", "ema", "sma"],
            "volume": ["volume", "obv", "vwap"],
            "depth": ["depth", "spread", "bid", "ask"],
            "technical": ["macd", "bb", "cci", "mfi", "williams"],
        }

        # Features that are inherently lagged by design
        inherently_lagged_patterns = [
            "slope",
            "returns",
            "change",
            "diff",
            "momentum",
            "acceleration",
        ]

        # Features that should be investigated but may be legitimate
        suspicious_features: List[Dict[str, Any]] = []
        potentially_legitimate_features: List[Dict[str, Any]] = []

        # Enhanced legitimate patterns for common technical indicators
        enhanced_legitimate_patterns = [
            "lag",
            "shift",
            "prev",
            "diff",
            "change",
            "slope",
            "returns",
            "pct_change",
            "impact",
            "correlation",
            "spread",
            "ratio",
            "zscore",
            "upper",
            "lower",
            "momentum",
        ]

        for col in features_df.columns:
            col_lower = col.lower()
            feature_analysis = {
                "feature": col,
                "risk_level": "unknown",
                "reasoning": "",
                "recommendation": "",
            }

            # Check if feature has rolling window indicators
            has_rolling_window = any(
                pattern in col_lower for pattern in ["rolling", "window", "period"]
            )

            # Check for specific rolling patterns
            for category, patterns in rolling_patterns.items():
                if any(pattern in col_lower for pattern in patterns):
                    if has_rolling_window:
                        feature_analysis["risk_level"] = "high"
                        feature_analysis["reasoning"] = f"Rolling {category} feature detected"
                        feature_analysis["recommendation"] = "Ensure proper lagging is applied"
                        suspicious_features.append(feature_analysis)
                    else:
                        feature_analysis["risk_level"] = "medium"
                        feature_analysis["reasoning"] = f"{category} feature without explicit rolling window"
                        feature_analysis["recommendation"] = "Verify calculation method"
                        potentially_legitimate_features.append(feature_analysis)

            # Check for inherently lagged features
            if any(pattern in col_lower for pattern in inherently_lagged_patterns):
                feature_analysis["risk_level"] = "low"
                feature_analysis["reasoning"] = "Inherently lagged feature type"
                feature_analysis["recommendation"] = "Likely safe, but verify implementation"
                potentially_legitimate_features.append(feature_analysis)

            # Check for legitimate lagging patterns
            if any(pattern in col_lower for pattern in enhanced_legitimate_patterns):
                feature_analysis["risk_level"] = "low"
                feature_analysis["reasoning"] = "Explicit lagging pattern detected"
                feature_analysis["recommendation"] = "Likely safe"
                potentially_legitimate_features.append(feature_analysis)

        # Add to results for reference
        results["legitimate_features"] = potentially_legitimate_features
        results["suspicious_features"] = suspicious_features

        # Generate warnings for high-risk features
        high_risk_count = len([f for f in suspicious_features if f["risk_level"] == "high"])
        if high_risk_count > 0:
            results["warnings"].append(
                f"Found {high_risk_count} high-risk rolling features that may have lookahead bias"
            )

    def _identify_lagging_type(self, feature_name: str) -> str:
        """Identify the type of lagging applied to a feature."""
        feature_lower = feature_name.lower()
        
        if any(pattern in feature_lower for pattern in ["lag_", "_lag", "shift_", "_shift"]):
            return "explicit_lagging"
        elif any(pattern in feature_lower for pattern in ["prev_", "_prev", "last_", "_last"]):
            return "previous_value"
        elif any(pattern in feature_lower for pattern in ["rolling_", "window_", "period_"]):
            return "rolling_window"
        elif any(pattern in feature_lower for pattern in ["returns", "pct_change", "diff"]):
            return "inherently_lagged"
        else:
            return "unknown_lagging"

    def _generate_recommendations(self, results: Dict[str, Any]) -> None:
        """Generate actionable recommendations based on findings."""
        recommendations: List[str] = []

        if results["lookahead_bias_detected"]:
            recommendations.append(
                "🚨 CRITICAL: Lookahead bias detected. Review feature engineering pipeline immediately."
            )
            recommendations.append(
                "🔍 Investigate all features with correlation > 0.8 with target variable"
            )
            recommendations.append(
                "⏰ Verify temporal alignment between features and target variables"
            )

        if results["warnings"]:
            recommendations.append(
                "⚠️ Review rolling window features for proper lagging implementation"
            )
            recommendations.append(
                "📊 Analyze feature importance distribution for suspicious patterns"
            )

        # Specific recommendations based on findings
        if results.get("suspicious_features"):
            high_risk_features = [
                f for f in results["suspicious_features"] if f["risk_level"] == "high"
            ]
            if high_risk_features:
                recommendations.append(
                    f"🎯 Focus on {len(high_risk_features)} high-risk features first"
                )

        if results.get("feature_correlations"):
            high_corr_features = [
                f for f, c in results["feature_correlations"].items() if abs(c) > 0.7
            ]
            if high_corr_features:
                recommendations.append(
                    f"🔍 Investigate {len(high_corr_features)} features with high target correlation"
                )

        # General best practices
        recommendations.extend([
            "📚 Review financial ML literature on lookahead bias prevention",
            "🧪 Test features with walk-forward analysis",
            "⏱️ Ensure all features use only past information",
            "📈 Implement proper train/test split with temporal boundaries"
        ])

        results["recommendations"] = recommendations

    @handle_errors(default_return=None, context="LookaheadBiasDetector.validate_train_test_split")
    def validate_train_test_split(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        timestamp_col: str,
        split_date: Optional[datetime] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Validate that train/test split doesn't introduce lookahead bias.
        
        Args:
            train_data: Training data
            test_data: Test data
            timestamp_col: Column containing timestamps
            split_date: Explicit split date (optional)
            
        Returns:
            Validation results dictionary
        """
        results: Dict[str, Any] = {
            "split_valid": True,
            "issues": [],
            "recommendations": []
        }

        try:
            # Ensure timestamps are datetime
            train_timestamps = pd.to_datetime(train_data[timestamp_col])
            test_timestamps = pd.to_datetime(test_data[timestamp_col])

            # Check for temporal overlap
            max_train_time = train_timestamps.max()
            min_test_time = test_timestamps.min()

            if max_train_time >= min_test_time:
                results["split_valid"] = False
                results["issues"].append(
                    f"TEMPORAL OVERLAP: Training data extends to {max_train_time} "
                    f"but test data starts at {min_test_time}"
                )

            # Check for gaps that might indicate data leakage
            time_diff = min_test_time - max_train_time
            if time_diff.days > self.temporal_tolerance:
                results["warnings"].append(
                    f"LARGE GAP: {time_diff.days} days between train and test data"
                )

            # Check for suspicious feature patterns in test data
            if results["split_valid"]:
                test_validation = self.detect_feature_lookahead_bias(
                    test_data.drop(columns=[timestamp_col]),
                    pd.Series([1] * len(test_data)),  # Dummy target for validation
                    timestamp_col
                )
                
                if test_validation.get("lookahead_bias_detected"):
                    results["issues"].append(
                        "TEST DATA ISSUES: Lookahead bias detected in test data features"
                    )

            # Generate recommendations
            if not results["split_valid"]:
                results["recommendations"].append(
                    "🔴 Fix temporal overlap between train and test data"
                )
                results["recommendations"].append(
                    "⏰ Ensure test data timestamps are strictly after training data"
                )

            return results

        except Exception as e:
            self.logger.exception(f"Error validating train/test split: {e}")
            results["error"] = str(e)
            return results

    def add_lagging_to_features(
        self,
        features_df: pd.DataFrame,
        lag_periods: List[int] = None,
        timestamp_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Add lagging to features to prevent lookahead bias.
        
        Args:
            features_df: Original features DataFrame
            lag_periods: List of lag periods to apply
            timestamp_col: Optional timestamp column for temporal ordering
            
        Returns:
            DataFrame with lagged features added
        """
        if lag_periods is None:
            lag_periods = [1, 2, 3, 5, 10]  # Common lag periods

        lagged_features = features_df.copy()

        # Sort by timestamp if available
        if timestamp_col and timestamp_col in features_df.columns:
            lagged_features = lagged_features.sort_values(timestamp_col).reset_index(drop=True)

        # Add lagged versions of numeric features
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            for lag in lag_periods:
                lagged_features[f"{col}_lag_{lag}"] = lagged_features[col].shift(lag)

        # Add rolling statistics with proper lagging
        for col in numeric_columns:
            # Rolling mean with lag
            lagged_features[f"{col}_rolling_mean_5_lag_1"] = (
                lagged_features[col].rolling(window=5).mean().shift(1)
            )
            
            # Rolling std with lag
            lagged_features[f"{col}_rolling_std_5_lag_1"] = (
                lagged_features[col].rolling(window=5).std().shift(1)
            )

        # Remove rows with NaN values from lagging
        lagged_features = lagged_features.dropna()

        return lagged_features

    def _analyze_implementation(
        self,
        feature_engineering_code: str,
        features_df: pd.DataFrame,
        results: Dict[str, Any],
    ) -> None:
        """
        Analyze feature engineering code for potential lookahead bias.
        
        Args:
            feature_engineering_code: Code string to analyze
            features_df: Features DataFrame for context
            results: Results dictionary to update
        """
        implementation_analysis: Dict[str, Any] = {
            "properly_lagged_features": [],
            "suspicious_patterns": [],
            "code_quality_issues": [],
            "recommendations": []
        }

        try:
            # Check for common lookahead bias patterns in code
            suspicious_patterns = [
                r"\.shift\(-?\d+\)",  # Negative shifts (future data)
                r"\.iloc\[-?\d+:\]",  # Future indexing
                r"\.tail\(\d+\)",     # Future data access
                r"\.nlargest\(\d+\)", # Future-based selection
                r"\.nsmallest\(\d+\)", # Future-based selection
            ]

            for pattern in suspicious_patterns:
                matches = re.findall(pattern, feature_engineering_code)
                if matches:
                    implementation_analysis["suspicious_patterns"].extend(matches)

            # Check for proper lagging patterns
            proper_lagging_patterns = [
                r"\.shift\(\d+\)",    # Positive shifts (past data)
                r"\.iloc\[:\d+\]",    # Past indexing
                r"\.head\(\d+\)",     # Past data access
            ]

            for pattern in proper_lagging_patterns:
                matches = re.findall(pattern, feature_engineering_code)
                if matches:
                    implementation_analysis["properly_lagged_features"].extend(matches)

            # Check for rolling window patterns
            rolling_patterns = [
                r"\.rolling\(window=\d+\)",
                r"\.rolling\(\d+\)",
                r"\.ewm\(span=\d+\)",
            ]

            for pattern in rolling_patterns:
                matches = re.findall(pattern, feature_engineering_code)
                if matches:
                    # Check if these are properly lagged
                    for match in matches:
                        if f"{match}.shift(1)" in feature_engineering_code:
                            implementation_analysis["properly_lagged_features"].append(match)
                        else:
                            implementation_analysis["suspicious_patterns"].append(
                                f"{match} (no explicit lagging)"
                            )

            # Generate code-specific recommendations
            if implementation_analysis["suspicious_patterns"]:
                implementation_analysis["recommendations"].append(
                    "🔴 Review suspicious patterns that may access future data"
                )

            if implementation_analysis["properly_lagged_features"]:
                implementation_analysis["recommendations"].append(
                    "✅ Good: Proper lagging patterns detected in code"
                )

            # Update results
            results["implementation_analysis"] = implementation_analysis

            # Log findings
            if implementation_analysis["suspicious_patterns"]:
                self.logger.warning(
                    f"⚠️ Found {len(implementation_analysis['suspicious_patterns'])} suspicious patterns in code"
                )

            self.logger.info(
                f"✅ Implementation analysis: {len(implementation_analysis['properly_lagged_features'])} features have proper lagging"
            )

        except Exception as e:
            self.logger.error(f"Error analyzing implementation code: {e}")
            implementation_analysis["error"] = str(e)

    def _check_feature_lagging_in_code(
        self,
        feature_name: str,
        feature_engineering_code: str,
    ) -> Dict[str, Any]:
        """
        Check if a specific feature has proper lagging in the code.
        
        Args:
            feature_name: Name of the feature to check
            feature_engineering_code: Code string to analyze
            
        Returns:
            Dictionary with lagging analysis results
        """
        feature_lower = feature_name.lower()
        
        # Look for feature-specific code patterns
        feature_patterns = [
            rf"{re.escape(feature_name)}\s*=",
            rf"{re.escape(feature_name)}\s*\.",
            rf"['\"]{re.escape(feature_name)}['\"]",
        ]

        lagging_analysis = {
            "feature": feature_name,
            "has_code_definition": False,
            "properly_lagged": False,
            "lagging_method": "unknown",
            "risk_level": "unknown",
            "recommendations": []
        }

        try:
            # Check if feature is defined in code
            for pattern in feature_patterns:
                if re.search(pattern, feature_engineering_code):
                    lagging_analysis["has_code_definition"] = True
                    break

            if not lagging_analysis["has_code_definition"]:
                lagging_analysis["risk_level"] = "low"
                lagging_analysis["recommendations"].append("Feature not defined in provided code")
                return lagging_analysis

            # Check for proper lagging patterns
            if f"{feature_name}.shift(" in feature_engineering_code:
                lagging_analysis["properly_lagged"] = True
                lagging_analysis["lagging_method"] = "explicit_shift"
                lagging_analysis["risk_level"] = "low"
                lagging_analysis["recommendations"].append("✅ Proper lagging with .shift() detected")
            elif f"{feature_name}.iloc[" in feature_engineering_code:
                lagging_analysis["properly_lagged"] = True
                lagging_analysis["lagging_method"] = "indexing"
                lagging_analysis["risk_level"] = "low"
                lagging_analysis["recommendations"].append("✅ Proper lagging with indexing detected")
            else:
                lagging_analysis["risk_level"] = "high"
                lagging_analysis["recommendations"].append("🔴 No explicit lagging detected - potential lookahead bias")

            return lagging_analysis

        except Exception as e:
            lagging_analysis["error"] = str(e)
            return lagging_analysis


def apply_feature_lagging(
    features_df: pd.DataFrame,
    lag_periods: List[int] = None,
    timestamp_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience function to apply feature lagging.
    
    Args:
        features_df: Features DataFrame
        lag_periods: List of lag periods
        timestamp_col: Optional timestamp column
        
    Returns:
        DataFrame with lagged features
    """
    detector = LookaheadBiasDetector()
    return detector.add_lagging_to_features(features_df, lag_periods, timestamp_col)
