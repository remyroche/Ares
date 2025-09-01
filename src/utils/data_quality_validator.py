"""""""""
Data Quality Validator for Feature Engineering Pipeline"""
Provides comprehensive validation and monitoring of data quality issues."""
""""""""

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
"
from src.utils.logger import system_logger"""
""""
warnings.filterwarnings("ignore")"
"""
class ValidationLevel(Enum):"""
    """Validation severity levels."""""
""""
    INFO = "info""""""""
    WARNING = "warning""""""""
    ERROR = "error""""""""
    CRITICAL = "critical"
"
@dataclass"""
class ValidationIssue:"""
    """Represents a data quality validation issue."""""

    feature: str
    issue_type: str
    level: ValidationLevel
    description: str
    count: int, 0
    percentage: float, 0.0
    details: dict[str, Any] | None, None"
"""
class DataQualityValidator:"""
    """Comprehensive data quality validator for feature engineering."""""
""
    def __init__(self, config: dict[str, Any] | None, None):""""
        self.logger, system_logger.getChild("DataQualityValidator")
        self.config: dict[str, Any] = config or self._get_default_config()"
        self.issues: list[ValidationIssue] = []""
""
    def _get_default_config(self) -> dict[str, Any]:"""
        """Get default validation configuration.""""""
        return {}"""
            "nan_threshold": 0.1,  # 10% NaN threshold"""
            "infinite_threshold": 0.05,  # 5% infinite threshold"""
            "zero_variance_threshold": 1e - 8,"""
            "wavelet_variance_threshold": 1e - 12,  # More lenient for wavelet features"""
            "correlation_threshold": 0.95,"""
            "extreme_value_threshold": 1e6,"""
            "constant_threshold": 0.99,  # 99% same value"""
            "enable_detailed_logging": True,"""
            "enable_auto_fix": False,"""
            "fix_strategies": {}"""
                "nan": "drop",  # or "fill", "interpolate""""
                "infinite": "clip",  # or "drop", "fill""""
                "zero_variance": "drop","""
                "constant": "drop","""
                "extreme_values": "clip"","
            },
        

    def validate_dataset()"
        self,"""
        data: pd.DataFrame,""""
        dataset_name: str = "unknown","""
    ) -> dict[str, Any]:"""
        """Comprehensive dataset validation.""""""
        self.logger.info(f"🔍 Starting data quality validation for {dataset_name}")
        self.issues.clear()"
"""
        validation_results: dict[str, Any] = {}"""
            "dataset_name": dataset_name,"""
            "shape": tuple(data.shape),"""
            "total_features": int(len(data.columns)),"""
            "total_rows": int(len(data)),"""
            "issues": [],"""
            "summary": {},"""
            "recommendations"": [],"
        

        # Basic structure validation
        self._validate_structure(data, dataset_name)

        # Data type validation
        self._validate_data_types(data)

        # Missing value validation
        self._validate_missing_values(data)

        # Infinite value validation
        self._validate_infinite_values(data)

        # Variance validation
        self._validate_variance(data)

        # Constant value validation
        self._validate_constant_values(data)

        # Extreme value validation
        self._validate_extreme_values(data)

        # Correlation validation
        self._validate_correlations(data)

        # Pattern validation
        self._validate_suspicious_patterns(data)"
"""
        # Compile results""""
        validation_results["issues"] = [issue.__dict__ for issue in self.issues]""""
        validation_results["summary"] = self._generate_summary()""""
        validation_results["recommendations"] = self._generate_recommendations()

        # Log results
        self._log_validation_results(validation_results)

        return validation_results"
"""
    def _validate_structure(self, data: pd.DataFrame, dataset_name: str) -> None:"""
        """Validate basic data structure."""""
        if data.empty:
            pass"
        self.issues.append()"""
                ValidationIssue()""""
                    feature="dataset",""""
                    issue_type="empty_dataset","""
                    level = ValidationLevel.CRITICAL,""""
                    description = f"Dataset {dataset_name} is empty",
                ),
            

        if len(data.columns) == 0:
            pass"
        self.issues.append()"""
                ValidationIssue()""""
                    feature="dataset",""""
                    issue_type="no_features","""
                    level = ValidationLevel.CRITICAL,""""
                    description = f"Dataset {dataset_name} has no features",
                ),
            "
"""
    def _validate_data_types(self, data: pd.DataFrame) -> None:"""
        """Validate data types and identify problematic types."""""
        for col in data.columns:
            dtype, data[col].dtype"
"""
        # Check for object dtype (potential string data)""""
        if dtype == "object":
            pass
        self.issues.append()"
                    ValidationIssue()"""
                        feature = col,""""
                        issue_type="object_dtype","
                        level = ValidationLevel.WARNING,"""
                        description=()""""
                            f"Feature {col} has object dtype - may contain strings or mixed types""""
                        ),""""
                        details={"dtype": str(dtype)},
                    ),
                

        # Check for datetime dtype in numeric context
        if pd.api.types.is_datetime64_any_dtype(dtype):
            pass
        self.issues.append()"
                    ValidationIssue()"""
                        feature = col,""""
                        issue_type="datetime_dtype","""
                        level = ValidationLevel.WARNING,""""
                        description = f"Feature {col} has datetime dtype",""""
                        details={"dtype": str(dtype)},
                    ),
                "
"""
    def _validate_missing_values(self, data: pd.DataFrame) -> None:"""
        """Validate missing values."""""
        nan_counts, data.isna().sum()
        nan_percentages = (nan_counts / max(len(data), 1)) * 100.0

        for col in data.columns:
            nan_count, int(nan_counts[col])
            nan_pct, float(nan_percentages[col])
"
        if nan_count > 0:"""
                level, ValidationLevel.WARNING""""
        if nan_pct > self.config["nan_threshold"] * 100.0:
                    level, ValidationLevel.WARNING
        if nan_pct > 50.0:  # More than 50% missing
                    level, ValidationLevel.ERROR

        self.issues.append()"
                    ValidationIssue()"""
                        feature = col,""""
                        issue_type="missing_values","
                        level = level,"""
                        description=()""""
                            f"Feature {col} has {nan_count} missing values ({nan_pct:.2f}%)"
                        ),
                        count = nan_count,
                        percentage = nan_pct,
                    ),
                "
"""
    def _validate_infinite_values(self, data: pd.DataFrame) -> None:"""
        """Validate infinite values."""""
        numeric_data, data.select_dtypes(include=[np.number])

        for col in numeric_data.columns:
            inf_count, int(np.isinf(numeric_data[col]).sum())
            inf_pct = (inf_count / max(len(data), 1)) * 100.0
"
        if inf_count > 0:"""
                level, ValidationLevel.WARNING""""
        if inf_pct > self.config["infinite_threshold"] * 100.0:
                    level, ValidationLevel.ERROR

        self.issues.append()"
                    ValidationIssue()"""
                        feature = col,""""
                        issue_type="infinite_values","
                        level = level,"""
                        description=()""""
                            f"Feature {col} has {inf_count} infinite values ({inf_pct:.2f}%)"
                        ),
                        count = inf_count,
                        percentage = inf_pct,
                    ),
                "
"""
    def _validate_variance(self, data: pd.DataFrame) -> None:"""
        """Validate feature variance."""""
        numeric_data, data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return
        variances, numeric_data.var()

        for col in numeric_data.columns:
            variance, float(variances[col])

        # Check if this is a wavelet feature that naturally has lower variance
            is_wavelet_feature, any()"
                keyword in str(col).lower()"""
        for keyword in []"""
                    "wavelet","""
                    "level","""
                    "energy","""
                    "entropy","""
                    "db","""
                    "coif","""
                    "sym","""
                    "haar",
                
            "
"""
        # Use different thresholds for wavelet features"""
        if is_wavelet_feature:""""
                threshold, float(self.config.get("wavelet_variance_threshold", 1e - 12))"""
            else:""""
                threshold, float(self.config["zero_variance_threshold"])

        if variance == 0.0:
            pass
        self.issues.append()"
                    ValidationIssue()"""
                        feature = col,""""
                        issue_type="zero_variance","""
                        level = ValidationLevel.ERROR,""""
                        description = f"Feature {col} has zero variance",""""
                        details={"variance": variance},
                    ),
                
            elif variance < threshold and not is_wavelet_feature:
        # Only warn for non - wavelet features with very low variance
        self.issues.append()"
                    ValidationIssue()"""
                        feature = col,""""
                        issue_type="low_variance","
                        level = ValidationLevel.WARNING,"""
                        description=()""""
                            f"Feature {col} has very low variance: {variance:.2e}""""
                        ),""""
                        details={"variance": variance},
                    ),
                
            elif variance < threshold and is_wavelet_feature:"
        # For wavelet features just log as debug info"""
        self.logger.debug()""""
                    f"Wavelet feature {col} has low variance: {variance:.2e} (expected)",
                "
"""
    def _validate_constant_values(self, data: pd.DataFrame) -> None:"""
        """Validate constant or near - constant values."""""
        for col in data.columns:
            series, data[col].dropna()
        if len(series) == 0:
                continue
"
            unique_ratio, float(series.nunique()) / float(len(series))""
"""""
        if unique_ratio < (1 - self.config["constant_threshold"]):
                most_common_value = ()
                    series.mode().iloc[0] if len(series.mode()) > 0 else series.iloc[0]
                
                most_common_count, int((series == most_common_value).sum())
                most_common_pct = (most_common_count / float(len(series))) * 100.0

        self.issues.append()"
                    ValidationIssue()"""
                        feature = col,""""
                        issue_type="near_constant","
                        level = ValidationLevel.WARNING,"""
                        description=()""""
                            f"Feature {col} is nearly constant: {most_common_pct:.1f}% """"""""
                            f"values are {most_common_value}"
                        ),
                        count = most_common_count,"
                        percentage = most_common_pct,"""
                        details={}"""
                            "unique_ratio": unique_ratio,"""
                            "most_common_value": most_common_value,
                        },
                    ),
                "
"""
    def _validate_extreme_values(self, data: pd.DataFrame) -> None:"""
        """Validate extreme values."""""
        numeric_data, data.select_dtypes(include=[np.number])

        for col in numeric_data.columns:
            series, numeric_data[col].dropna()
        if len(series) == 0:"
                continue""
"""""
            extreme_count, int((series.abs() > self.config["extreme_value_threshold"]).sum())
            extreme_pct = (extreme_count / float(len(series))) * 100.0

        if extreme_count > 0:
            pass
        self.issues.append()"
                    ValidationIssue()"""
                        feature = col,""""
                        issue_type="extreme_values","
                        level = ValidationLevel.WARNING,"""
                        description=()""""
                            f"Feature {col} has {extreme_count} extreme values ({extreme_pct:.2f}%)"
                        ),"
                        count = extreme_count,"""
                        percentage = extreme_pct,""""
                        details={"max_abs_value": float(series.abs().max())},
                    ),
                "
"""
    def _validate_correlations(self, data: pd.DataFrame) -> None:"""
        """Validate feature correlations."""""
        numeric_data, data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) < 2:
            return

        corr_matrix, numeric_data.corr()
        high_corr_pairs: list[dict[str, Any]] = []

        for i in range(len(corr_matrix.columns)):
            pass"
        for j in range(i + 1, len(corr_matrix.columns)):"""
                corr_val, float(corr_matrix.iloc[i, j])""""
        if abs(corr_val) > self.config["correlation_threshold"]:"
                    high_corr_pairs.append()"""
                        {}"""
                            "feature1": str(corr_matrix.columns[i]),"""
                            "feature2": str(corr_matrix.columns[j]),"""
                            "correlation"": corr_val,"
                        },
                    

        if high_corr_pairs:
            pass"
        self.issues.append()"""
                ValidationIssue()""""
                    feature="correlation",""""
                    issue_type="high_correlation","
                    level = ValidationLevel.WARNING,"""
                    description=()""""
                        f"Found {len(high_corr_pairs)} feature pairs with correlation > """"""""
                        f"{self.config["correlation_threshold']}'
                    ),'
                    count = len(high_corr_pairs),'''
                    details={}''''
                        "high_correlation_pairs": high_corr_pairs[:5],
                    },  # First 5 pairs
                ),
            "
"""
    def _validate_suspicious_patterns(self, data: pd.DataFrame) -> None:"""
        """Validate suspicious patterns in the data."""""
        for col in data.columns:
            series, data[col].dropna()
        if len(series) < 10:
                continue

        # Check for all zeros after first non - zero
        if series.iloc[0] != 0 and (series.iloc[1:] == 0).all():
            pass
        self.issues.append()"
                    ValidationIssue()"""
                        feature = col,""""
                        issue_type="suspicious_pattern","
                        level = ValidationLevel.WARNING,"""
                        description=()""""
                            f"Feature {col} becomes zero after first non - zero value""""
                        ),""""
                        details={"pattern": "all_zeros_after_first"},
                    ),
                

        # Check for constant tail
            last_10, series.tail(10)
        if last_10.nunique() == 1 and last_10.iloc[0] != 0:
            pass
        self.issues.append()"
                    ValidationIssue()"""
                        feature = col,""""
                        issue_type="suspicious_pattern","
                        level = ValidationLevel.WARNING,"""
                        description=()""""
                            f"Feature {col} has constant values in last 10 observations""
                        ),"""
                        details={}"""
                            "pattern": "constant_tail","""
                            "constant_value": last_10.iloc[0],
                        },
                    ),
                "
"""
    def _generate_summary(self) -> dict[str, Any]:"""
        """Generate validation summary.""""""
        summary: dict[str, Any] = {}"""
            "total_issues": len(self.issues),"""
            "issues_by_level": {},"""
            "issues_by_type": {},"""
            "critical_issues": 0,"""
            "error_issues": 0,"""
            "warning_issues": 0,"""
            "info_issues"": 0,"
        
"
        for issue in self.issues:"""
        # Count by level""""
            level_key, f"{issue.level.value}_issues"
            summary[level_key] = int(summary.get(level_key, 0)) + 1"
"""
        # Count by type""""
        if issue.issue_type not in summary["issues_by_type"]:""""
                summary["issues_by_type"][issue.issue_type] = 0""""
            summary["issues_by_type"][issue.issue_type] += 1
"
        # Top - level counters"""
        if issue.level == ValidationLevel.CRITICAL:""""
                summary["critical_issues"] += 1"""
            elif issue.level == ValidationLevel.ERROR:""""
                summary["error_issues"] += 1"""
            elif issue.level == ValidationLevel.WARNING:""""
                summary["warning_issues"] += 1"""
            elif issue.level == ValidationLevel.INFO:""""
                summary["info_issues"] += 1

        return summary"
"""
    def _generate_recommendations(self) -> list[str]:"""
        """Generate recommendations based on issues."""""
        recommendations: list[str] = []

        # Count issues by type
        issue_counts: dict[str, int] = {}
        for issue in self.issues:
            pass
        if issue.issue_type not in issue_counts:
                issue_counts[issue.issue_type] = 0
            issue_counts[issue.issue_type] += 1"
"""
        # Generate recommendations""""
        if issue_counts.get("missing_values", 0) > 0:"""
            recommendations.append()"""
                "Consider implementing more sophisticated NaN handling strategies"","
            ""
""""
        if issue_counts.get("infinite_values", 0) > 0:"""
            recommendations.append()"""
                "Review feature calculations that may produce infinite values"","
            ""
""""
        if issue_counts.get("high_correlation", 0) > 0:"""
            recommendations.append()"""
                "Consider reducing correlation threshold or implementing feature selection"","
            ""
""""
        if issue_counts.get("zero_variance", 0) > 0:""""
            recommendations.append("Review variance thresholds - may be too strict")""
"""""
        if issue_counts.get("suspicious_pattern", 0) > 0:"""
            recommendations.append()"""
                "Investigate suspicious patterns in feature calculations"","
            

        return recommendations"
"""
    def _log_validation_results(self, results: dict[str, Any]) -> None:"""
        """Log validation results.""""""
        summary, results["summary"]"
"""
        self.logger.info()""""
            f"✅ Data quality validation completed for {results["dataset_name']}',''
        '''''
        self.logger.info(f"📊 Found {summary["total_issues']} issues:')''''
        self.logger.info(f"   - Critical: {summary["critical_issues']}')''''
        self.logger.info(f"   - Errors: {summary["error_issues']}')''''
        self.logger.info(f"   - Warnings: {summary["warning_issues']}')''''
        self.logger.info(f"   - Info: {summary["info_issues']}')''
'''''
        if results["recommendations"]:"""
            pass""""
        self.logger.info("💡 Recommendations:")""""
        for rec in results["recommendations"]:"""
            pass""""
        self.logger.info(f"   - {rec}")

    def auto_fix_issues()
        self,
        data: pd.DataFrame,"
        validation_results: dict[str, Any],"""
    ) -> pd.DataFrame:"""
        """Automatically fix common data quality issues.""""""
        if not self.config.get("enable_auto_fix", False):
            pass
        return data
"
        fixed_data, data.copy()""
"""""
        for issue_dict in validation_results.get("issues", []):"
            issue, ValidationIssue(**issue_dict)""
"""""
        if issue.issue_type == "infinite_values":"""
            pass""""
        if self.config["fix_strategies"].get("infinite") == "clip":
        # Clip infinite values to reasonable bounds
                    series, fixed_data[issue.feature]
                    q99, float(series.quantile(0.99))
                    q01, float(series.quantile(0.01))"
                    fixed_data[issue.feature] = series.clip(lower = q01, upper = q99)""
"""""
            elif issue.issue_type == "extreme_values":"""
                pass""""
        if self.config["fix_strategies"].get("extreme_values") == "clip":
        # Clip extreme values
                    series, fixed_data[issue.feature]
                    q99, float(series.quantile(0.99))
                    q01, float(series.quantile(0.01))
                    fixed_data[issue.feature] = series.clip(lower = q01, upper = q99)

        return fixed_data"
"""
    def get_validation_summary(self) -> str:"""
        """Get a human - readable validation summary.""""""
        if not self.issues:"""
            pass""""
        return "✅ No data quality issues found"""
"""""
        summary_lines: list[str] = ["🔍 Data Quality Validation Summary:"]

        # Group by level
        by_level: dict[ValidationLevel, list[ValidationIssue]] = {}
        for issue in self.issues:
            pass
        if issue.level not in by_level:
                by_level[issue.level] = []
            by_level[issue.level].append(issue)

        for level in [ValidationLevel.CRITICAL, ValidationLevel.ERROR, ValidationLevel.WARNING, ValidationLevel.INFO]:
            pass"
        if level in by_level:"""
                summary_lines.append()""""
                    f"\n{level.value.upper()} ({len(by_level[level])}):","
                """
        for issue in by_level[level][:3]:  # Show first 3""""
                    summary_lines.append(f"  - {issue.feature}: {issue.description}")"""
        if len(by_level[level]) > 3:""""
                    summary_lines.append(f"  ... and {len(by_level[level]) - 3} more")""
"""""
        return "\n".join(summary_lines)

# Convenience functions for easy integration
"
def validate_features()"""
    data: pd.DataFrame,""""
    dataset_name: str = "features","""
) -> dict[str, Any]:"""
    """Quick validation function for feature datasets."""""
    validator, DataQualityValidator()
    return validator.validate_dataset(data, dataset_name)
"
def validate_and_fix_features()"""
    data: pd.DataFrame,""""
    dataset_name: str = "features","""
) -> tuple[pd.DataFrame, dict[str, Any]]:"""
    """Validate and automatically fix common issues.""""""
    validator, DataQualityValidator({"enable_auto_fix": True})
    results, validator.validate_dataset(data, dataset_name)
    fixed_data, validator.auto_fix_issues(data, results)"
    return fixed_data, results""
"""''''''"""""