"""
Advanced ML Data Quality Validation System

This module provides comprehensive data quality validation specifically designed
for machine learning training, including statistical analysis, drift detection,
feature correlation analysis, and quality scoring.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, NamedTuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging

import numpy as np
import pandas as pd
from scipy import stats

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.logger import system_logger
from src.utils.comprehensive_file_validation import (
    ValidationSeverity,
    ValidationIssue,
    FileValidationResult
)

# Type aliases for better readability
DataFrame = pd.DataFrame
Series = pd.Series
ConfigDict = Dict[str, Any]
ValidationIssues = List[str]
QualityScore = NamedTuple('QualityScore', [('overall', float), ('grade', str)])

@dataclass
class ValidationResult:
    """Base class for validation results."""
    is_valid: bool = False
    issues: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def add_issue(self, issue: str) -> None:
        """Add a validation issue."""
        self.issues.append(issue)
        self.is_valid = False

@dataclass
class DriftReport(ValidationResult):
    """Report for data drift detection."""
    drift_score: float = 0.0
    drifted_features: List[str] = field(default_factory=list)
    reference_data_info: Dict[str, Any] = field(default_factory=dict)
    current_data_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MLValidationResult(ValidationResult):
    """Comprehensive ML validation result."""
    quality_score: Optional[QualityScore] = None
    distribution_issues: List[str] = field(default_factory=list)
    outlier_issues: List[str] = field(default_factory=list)
    time_series_issues: List[str] = field(default_factory=list)
    financial_issues: List[str] = field(default_factory=list)
    correlation_issues: List[str] = field(default_factory=list)
    target_issues: List[str] = field(default_factory=list)
    drift_report: Optional[DriftReport] = None
    summary: Dict[str, Any] = field(default_factory=dict)

class StatisticalDataValidator:
    """Validates statistical properties of data."""
    
    def __init__(self, config: Optional[ConfigDict] = None):
        self.config = config or self._get_default_config()
        self.logger = system_logger.getChild("StatisticalDataValidator")
        self.is_initialized = False
        
    def _get_default_config(self) -> ConfigDict:
        """Get default configuration for statistical validation."""
        return {
            "distribution_tolerance": 0.1,
            "outlier_threshold": 3.0,
            "min_variance": 1e-6,
            "max_skewness": 10.0,
            "max_kurtosis": 20.0
        }
    
    async def initialize(self) -> bool:
        """Initialize StatisticalDataValidator."""
        try:
            self.logger.info("🚀 Initializing StatisticalDataValidator...")
            self.is_initialized = True
            self.logger.info("✅ StatisticalDataValidator initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing StatisticalDataValidator: {e}")
            return False
    
    def validate_data_distributions(self, df: DataFrame, 
                                  numeric_columns: Optional[List[str]] = None) -> ValidationIssues:
        """Validate statistical distributions of numeric columns."""
        issues = []
        
        if not self.is_initialized:
            issues.append("Validator not initialized")
            return issues
        
        if df.empty:
            issues.append("DataFrame is empty")
            return issues
        
        # Get numeric columns
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_columns:
            if col not in df.columns:
                continue
                
            series = df[col].dropna()
            if len(series) < 10:
                issues.append(f"Column {col}: insufficient data for distribution analysis (< 10 samples)")
                continue
            
            # Check variance
            variance = series.var()
            if variance < self.config["min_variance"]:
                issues.append(f"Column {col}: very low variance ({variance:.6f})")
            
            # Check skewness
            skewness = abs(series.skew())
            if skewness > self.config["max_skewness"]:
                issues.append(f"Column {col}: highly skewed distribution (skewness: {skewness:.3f})")
            
            # Check kurtosis
            kurtosis = abs(series.kurtosis())
            if kurtosis > self.config["max_kurtosis"]:
                issues.append(f"Column {col}: extreme kurtosis ({kurtosis:.3f})")
        
        return issues
    
    def validate_outliers(self, df: DataFrame, 
                         numeric_columns: Optional[List[str]] = None) -> ValidationIssues:
        """Validate outliers using statistical methods."""
        issues = []
        
        if not self.is_initialized:
            issues.append("Validator not initialized")
            return issues
        
        if df.empty:
            issues.append("DataFrame is empty")
            return issues
        
        # Get numeric columns
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_columns:
            if col not in df.columns:
                continue
                
            series = df[col].dropna()
            if len(series) < 10:
                continue
            
            # Use IQR method for outlier detection
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.config["outlier_threshold"] * IQR
            upper_bound = Q3 + self.config["outlier_threshold"] * IQR
            
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            outlier_ratio = len(outliers) / len(series)
            
            if outlier_ratio > 0.1:  # More than 10% outliers
                issues.append(f"Column {col}: high outlier ratio ({outlier_ratio:.2%})")
        
        return issues

class TimeSeriesValidator:
    """Validates time series data quality."""
    
    def __init__(self, config: Optional[ConfigDict] = None):
        self.config = config or self._get_default_config()
        self.logger = system_logger.getChild("TimeSeriesValidator")
        self.is_initialized = False
        
    def _get_default_config(self) -> ConfigDict:
        """Get default configuration for time series validation."""
        return {
            "max_gap_multiplier": 2.0,
            "max_duplicate_ratio": 0.01,
            "min_timestamp_span": 3600,  # 1 hour in seconds
            "future_tolerance_minutes": 5
        }
    
    async def initialize(self) -> bool:
        """Initialize TimeSeriesValidator."""
        try:
            self.logger.info("🚀 Initializing TimeSeriesValidator...")
            self.is_initialized = True
            self.logger.info("✅ TimeSeriesValidator initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing TimeSeriesValidator: {e}")
            return False
    
    def validate_time_series_quality(self, df: DataFrame, 
                                   timestamp_col: str,
                                   expected_interval: Optional[int] = None) -> ValidationIssues:
        """Validate time series data quality."""
        issues = []
        
        if not self.is_initialized:
            issues.append("Validator not initialized")
            return issues
        
        if df.empty or timestamp_col not in df.columns:
            issues.append(f"Invalid data or missing timestamp column: {timestamp_col}")
            return issues
        
        try:
            # Convert timestamp to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            
            # Sort by timestamp
            df_sorted = df.sort_values(timestamp_col).copy()
            
            # Check for future timestamps
            now = pd.Timestamp.now()
            future_timestamps = df_sorted[timestamp_col] > now + pd.Timedelta(minutes=self.config["future_tolerance_minutes"])
            if future_timestamps.any():
                issues.append(f"Found {future_timestamps.sum()} future timestamps")
            
            # Check for duplicates
            duplicate_ratio = df_sorted[timestamp_col].duplicated().sum() / len(df_sorted)
            if duplicate_ratio > self.config["max_duplicate_ratio"]:
                issues.append(f"High duplicate timestamp ratio: {duplicate_ratio:.2%}")
            
            # Check for gaps if expected interval is provided
            if expected_interval:
                time_diffs = df_sorted[timestamp_col].diff().dt.total_seconds()
                max_gap = time_diffs.max()
                if max_gap > expected_interval * self.config["max_gap_multiplier"]:
                    issues.append(f"Large time gap detected: {max_gap:.0f}s (expected: ~{expected_interval}s)")
            
            # Check timestamp span
            total_span = (df_sorted[timestamp_col].max() - df_sorted[timestamp_col].min()).total_seconds()
            if total_span < self.config["min_timestamp_span"]:
                issues.append(f"Timestamp span too short: {total_span:.0f}s")
                
        except Exception as e:
            self.logger.exception(f"Error validating time series: {e}")
            issues.append(f"Error during time series validation: {e}")
        
        return issues

class FinancialDataValidator:
    """Validates financial data quality."""
    
    def __init__(self, config: Optional[ConfigDict] = None):
        self.config = config or self._get_default_config()
        self.logger = system_logger.getChild("FinancialDataValidator")
        self.is_initialized = False
        
    def _get_default_config(self) -> ConfigDict:
        """Get default configuration for financial data validation."""
        return {
            "max_price_change_ratio": 0.5,  # 50% max price change
            "min_volume_threshold": 0.0,
            "max_zero_volume_ratio": 0.1,
            "min_price_threshold": 0.01,
            "max_price_threshold": 1e6
        }
    
    async def initialize(self) -> bool:
        """Initialize FinancialDataValidator."""
        try:
            self.logger.info("🚀 Initializing FinancialDataValidator...")
            self.is_initialized = True
            self.logger.info("✅ FinancialDataValidator initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing FinancialDataValidator: {e}")
            return False
    
    def validate_financial_data(self, df: DataFrame) -> ValidationIssues:
        """Validate financial data quality."""
        issues = []
        
        if not self.is_initialized:
            issues.append("Validator not initialized")
            return issues
        
        if df.empty:
            issues.append("DataFrame is empty")
            return issues
        
        # Common financial column patterns
        price_columns = [col for col in df.columns if any(term in col.lower() for term in ['price', 'close', 'open', 'high', 'low'])]
        volume_columns = [col for col in df.columns if 'volume' in col.lower()]
        
        # Validate price columns
        for col in price_columns:
            if col not in df.columns:
                continue
                
            series = df[col].dropna()
            if len(series) == 0:
                continue
            
            # Check price range
            if series.min() < self.config["min_price_threshold"]:
                issues.append(f"Column {col}: prices below minimum threshold ({series.min():.6f})")
            
            if series.max() > self.config["max_price_threshold"]:
                issues.append(f"Column {col}: prices above maximum threshold ({series.max():.2f})")
            
            # Check for extreme price changes
            if len(series) > 1:
                price_changes = series.pct_change().abs()
                max_change = price_changes.max()
                if max_change > self.config["max_price_change_ratio"]:
                    issues.append(f"Column {col}: extreme price change detected ({max_change:.2%})")
        
        # Validate volume columns
        for col in volume_columns:
            if col not in df.columns:
                continue
                
            series = df[col].dropna()
            if len(series) == 0:
                continue
            
            # Check for zero volumes
            zero_volume_ratio = (series == 0).sum() / len(series)
            if zero_volume_ratio > self.config["max_zero_volume_ratio"]:
                issues.append(f"Column {col}: high zero volume ratio ({zero_volume_ratio:.2%})")
            
            # Check minimum volume
            if series.min() < self.config["min_volume_threshold"]:
                issues.append(f"Column {col}: volumes below minimum threshold")
        
        return issues

class FeatureCorrelationValidator:
    """Validates feature correlations and multicollinearity."""
    
    def __init__(self, config: Optional[ConfigDict] = None):
        self.config = config or self._get_default_config()
        self.logger = system_logger.getChild("FeatureCorrelationValidator")
        self.is_initialized = False
        
    def _get_default_config(self) -> ConfigDict:
        """Get default configuration for correlation validation."""
        return {
            "max_correlation": 0.95,
            "max_multicollinearity_vif": 10.0,
            "min_correlation_for_removal": 0.8
        }
    
    async def initialize(self) -> bool:
        """Initialize FeatureCorrelationValidator."""
        try:
            self.logger.info("🚀 Initializing FeatureCorrelationValidator...")
            self.is_initialized = True
            self.logger.info("✅ FeatureCorrelationValidator initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing FeatureCorrelationValidator: {e}")
            return False
    
    def validate_feature_correlations(self, df: DataFrame, 
                                    target_col: Optional[str] = None) -> ValidationIssues:
        """Validate feature correlations and multicollinearity."""
        issues = []
        
        if not self.is_initialized:
            issues.append("Validator not initialized")
            return issues
        
        if df.empty:
            issues.append("DataFrame is empty")
            return issues
        
        # Get numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        if target_col and target_col in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=[target_col])
        
        if len(numeric_df.columns) < 2:
            return issues
        
        try:
            # Calculate correlation matrix
            corr_matrix = numeric_df.corr()
            
            # Check for high correlations
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = abs(corr_matrix.iloc[i, j])
                    if corr_value > self.config["max_correlation"]:
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        high_corr_pairs.append((col1, col2, corr_value))
            
            if high_corr_pairs:
                for col1, col2, corr_value in high_corr_pairs:
                    issues.append(f"High correlation between {col1} and {col2}: {corr_value:.3f}")
            
            # Check multicollinearity using VIF approximation
            multicollinearity_issues = self._check_multicollinearity(numeric_df)
            issues.extend(multicollinearity_issues)
            
        except Exception as e:
            self.logger.exception(f"Error validating correlations: {e}")
            issues.append(f"Error during correlation validation: {e}")
        
        return issues
    
    def _check_multicollinearity(self, df: DataFrame) -> ValidationIssues:
        """Check for multicollinearity using VIF approximation."""
        issues = []
        
        try:
            # Simple VIF approximation using correlation matrix
            corr_matrix = df.corr()
            vif_scores = {}
            
            for col in df.columns:
                # Calculate VIF using correlation with other features
                other_cols = [c for c in df.columns if c != col]
                if len(other_cols) > 0:
                    # Use R² as VIF approximation
                    corr_with_others = corr_matrix.loc[col, other_cols]
                    r_squared = (corr_with_others ** 2).max()
                    vif = 1 / (1 - r_squared) if r_squared < 1 else float('inf')
                    vif_scores[col] = vif
            
            # Report high VIF features
            high_vif_features = [(col, vif) for col, vif in vif_scores.items() 
                               if vif > self.config["max_multicollinearity_vif"]]
            
            for col, vif in high_vif_features:
                issues.append(f"High multicollinearity in {col}: VIF ≈ {vif:.2f}")
                
        except Exception as e:
            self.logger.exception(f"Error checking multicollinearity: {e}")
            issues.append(f"Error during multicollinearity check: {e}")
        
        return issues

class TargetVariableValidator:
    """Validates target variable quality."""
    
    def __init__(self, config: Optional[ConfigDict] = None):
        self.config = config or self._get_default_config()
        self.logger = system_logger.getChild("TargetVariableValidator")
        self.is_initialized = False
        
    def _get_default_config(self) -> ConfigDict:
        """Get default configuration for target validation."""
        return {
            "class_imbalance_threshold": 0.1,
            "target_leakage_threshold": 0.9,
            "min_target_variance": 1e-6
        }
    
    async def initialize(self) -> bool:
        """Initialize TargetVariableValidator."""
        try:
            self.logger.info("🚀 Initializing TargetVariableValidator...")
            self.is_initialized = True
            self.logger.info("✅ TargetVariableValidator initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing TargetVariableValidator: {e}")
            return False
    
    def validate_target_variable(self, df: DataFrame, 
                               target_col: str,
                               timestamp_col: Optional[str] = None) -> ValidationIssues:
        """Validate target variable quality."""
        issues = []
        
        if not self.is_initialized:
            issues.append("Validator not initialized")
            return issues
        
        if df.empty or target_col not in df.columns:
            issues.append(f"Invalid data or missing target column: {target_col}")
            return issues
        
        target = df[target_col]
        
        # Check for missing target values
        missing_target = target.isnull().sum()
        if missing_target > 0:
            issues.append(f"Found {missing_target} missing target values ({missing_target / len(target):.2%})")
        
        # Check for class imbalance (categorical targets)
        if target.dtype in ['object', 'category'] or target.nunique() < 10:
            class_counts = target.value_counts()
            min_class_ratio = class_counts.min() / class_counts.max()
            
            if min_class_ratio < self.config['class_imbalance_threshold']:
                issues.append(
                    f"Severe class imbalance: {min_class_ratio:.3f} "
                    f"(minority class: {class_counts.min()}, majority class: {class_counts.max()})"
                )
        
        # Check for target variance (regression targets)
        if target.dtype in [np.number] and target.nunique() > 10:
            target_variance = target.var()
            if target_variance < self.config['min_target_variance']:
                issues.append(
                    f"Low target variance: {target_variance:.6f} "
                    f"(threshold: {self.config['min_target_variance']})"
                )
        
        # Check for target leakage with time-based features
        if timestamp_col and timestamp_col in df.columns:
            time_leakage_issues = self._check_time_based_leakage(df, target_col, timestamp_col)
            issues.extend(time_leakage_issues)
        
        # Check for target leakage with other features
        feature_leakage_issues = self._check_feature_based_leakage(df, target_col)
        issues.extend(feature_leakage_issues)
        
        return issues
    
    def _check_time_based_leakage(self, df: DataFrame, 
                                 target_col: str, 
                                 timestamp_col: str) -> ValidationIssues:
        """Check for time-based target leakage."""
        issues = []
        
        try:
            # Convert timestamp to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            
            # Create time-based features
            df_copy = df.copy()
            df_copy['hour'] = df_copy[timestamp_col].dt.hour
            df_copy['day_of_week'] = df_copy[timestamp_col].dt.dayofweek
            df_copy['month'] = df_copy[timestamp_col].dt.month
            
            # Check correlation with target
            time_features = ['hour', 'day_of_week', 'month']
            for feature in time_features:
                if feature in df_copy.columns:
                    corr = abs(df_copy[feature].corr(df_copy[target_col]))
                    if corr > self.config['target_leakage_threshold']:
                        issues.append(
                            f"Potential time-based target leakage with {feature}: corr={corr:.3f}"
                        )
                        
        except Exception as e:
            self.logger.exception(f"Error checking time-based leakage: {e}")
            issues.append(f"Error checking time-based leakage: {e}")
        
        return issues
    
    def _check_feature_based_leakage(self, df: DataFrame, target_col: str) -> ValidationIssues:
        """Check for feature-based target leakage."""
        issues = []
        
        try:
            # Check for perfect or near-perfect correlations
            numeric_df = df.select_dtypes(include=[np.number])
            if target_col in numeric_df.columns:
                numeric_df = numeric_df.drop(columns=[target_col])
            
            for col in numeric_df.columns:
                corr = abs(numeric_df[col].corr(df[target_col]))
                if corr > self.config['target_leakage_threshold']:
                    issues.append(
                        f"Potential target leakage with {col}: corr={corr:.3f}"
                    )
                    
        except Exception as e:
            self.logger.exception(f"Error checking feature-based leakage: {e}")
            issues.append(f"Error checking feature-based leakage: {e}")
        
        return issues

class DataDriftDetector:
    """Detects data drift between reference and current datasets."""
    
    def __init__(self, reference_data: DataFrame):
        self.reference_data = reference_data
        self.logger = system_logger.getChild("DataDriftDetector")
        self.is_initialized = False
        
    async def initialize(self) -> bool:
        """Initialize DataDriftDetector."""
        try:
            self.logger.info("🚀 Initializing DataDriftDetector...")
            self.is_initialized = True
            self.logger.info("✅ DataDriftDetector initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing DataDriftDetector: {e}")
            return False
    
    def detect_drift(self, current_data: DataFrame) -> DriftReport:
        """Detect drift between reference and current data."""
        report = DriftReport()
        
        if not self.is_initialized:
            report.add_issue("Detector not initialized")
            return report
        
        if self.reference_data.empty or current_data.empty:
            report.add_issue("Reference or current data is empty")
            return report
        
        try:
            # Get common numeric columns
            ref_numeric = self.reference_data.select_dtypes(include=[np.number])
            curr_numeric = current_data.select_dtypes(include=[np.number])
            common_columns = list(set(ref_numeric.columns) & set(curr_numeric.columns))
            
            if not common_columns:
                report.add_issue("No common numeric columns for drift detection")
                return report
            
            drift_scores = {}
            drifted_features = []
            
            for col in common_columns:
                # Calculate distribution drift using KS test
                ref_series = ref_numeric[col].dropna()
                curr_series = curr_numeric[col].dropna()
                
                if len(ref_series) < 10 or len(curr_series) < 10:
                    continue
                
                try:
                    # Perform Kolmogorov-Smirnov test
                    ks_statistic, p_value = stats.ks_2samp(ref_series, curr_series)
                    
                    # Calculate drift score (0 = no drift, 1 = complete drift)
                    drift_score = 1 - p_value
                    drift_scores[col] = drift_score
                    
                    # Mark as drifted if p-value is very low
                    if p_value < 0.01:  # 1% significance level
                        drifted_features.append(col)
                        
                except Exception as e:
                    self.logger.warning(f"Could not calculate drift for {col}: {e}")
                    continue
            
            # Calculate overall drift score
            if drift_scores:
                report.drift_score = np.mean(list(drift_scores.values()))
                report.drifted_features = drifted_features
                
                # Add drift issues
                if drifted_features:
                    report.add_issue(f"Detected drift in {len(drifted_features)} features")
                    for feature in drifted_features[:5]:  # Limit to first 5
                        report.add_issue(f"Feature {feature} shows significant drift")
                
                # Add summary info
                report.reference_data_info = {
                    "shape": self.reference_data.shape,
                    "columns": len(self.reference_data.columns)
                }
                report.current_data_info = {
                    "shape": current_data.shape,
                    "columns": len(current_data.columns)
                }
            else:
                report.add_issue("Could not calculate drift scores for any features")
                
        except Exception as e:
            self.logger.exception(f"Error during drift detection: {e}")
            report.add_issue(f"Error during drift detection: {e}")
        
        return report

class DataQualityScorer:
    """Scores data quality based on validation results."""
    
    def __init__(self, weights: Optional[ConfigDict] = None):
        self.weights = weights or self._get_default_weights()
        self.logger = system_logger.getChild("DataQualityScorer")
        self.is_initialized = False
        
    def _get_default_weights(self) -> ConfigDict:
        """Get default weights for quality scoring."""
        return {
            "distribution": 0.2,
            "outliers": 0.15,
            "time_series": 0.15,
            "financial": 0.1,
            "correlations": 0.15,
            "target": 0.25
        }
    
    async def initialize(self) -> bool:
        """Initialize DataQualityScorer."""
        try:
            self.logger.info("🚀 Initializing DataQualityScorer...")
            self.is_initialized = True
            self.logger.info("✅ DataQualityScorer initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing DataQualityScorer: {e}")
            return False
    
    def calculate_quality_score(self, df: DataFrame, 
                              validation_result: MLValidationResult) -> QualityScore:
        """Calculate overall data quality score."""
        if not self.is_initialized:
            return QualityScore(0.0, "F")
        
        try:
            # Calculate component scores
            scores = {}
            
            # Distribution score
            if validation_result.distribution_issues:
                scores["distribution"] = max(0.0, 1.0 - len(validation_result.distribution_issues) * 0.1)
            else:
                scores["distribution"] = 1.0
            
            # Outlier score
            if validation_result.outlier_issues:
                scores["outliers"] = max(0.0, 1.0 - len(validation_result.outlier_issues) * 0.1)
            else:
                scores["outliers"] = 1.0
            
            # Time series score
            if validation_result.time_series_issues:
                scores["time_series"] = max(0.0, 1.0 - len(validation_result.time_series_issues) * 0.1)
            else:
                scores["time_series"] = 1.0
            
            # Financial score
            if validation_result.financial_issues:
                scores["financial"] = max(0.0, 1.0 - len(validation_result.financial_issues) * 0.1)
            else:
                scores["financial"] = 1.0
            
            # Correlation score
            if validation_result.correlation_issues:
                scores["correlations"] = max(0.0, 1.0 - len(validation_result.correlation_issues) * 0.1)
            else:
                scores["correlations"] = 1.0
            
            # Target score
            if validation_result.target_issues:
                scores["target"] = max(0.0, 1.0 - len(validation_result.target_issues) * 0.1)
            else:
                scores["target"] = 1.0
            
            # Calculate weighted average
            overall_score = sum(scores[component] * self.weights[component] 
                              for component in scores)
            
            # Determine grade
            if overall_score >= 0.9:
                grade = "A"
            elif overall_score >= 0.8:
                grade = "B"
            elif overall_score >= 0.7:
                grade = "C"
            elif overall_score >= 0.6:
                grade = "D"
            else:
                grade = "F"
            
            return QualityScore(overall_score, grade)
            
        except Exception as e:
            self.logger.exception(f"Error calculating quality score: {e}")
            return QualityScore(0.0, "F")

class AdvancedMLValidator:
    """Main class for comprehensive ML data validation."""
    
    def __init__(self, config: Optional[ConfigDict] = None):
        self.config = config or self._get_default_config()
        self.logger = system_logger.getChild("AdvancedMLValidator")
        self.is_initialized = False
        
        # Initialize validators
        self.statistical_validator = StatisticalDataValidator(self.config.get("statistical", {}))
        self.time_series_validator = TimeSeriesValidator(self.config.get("time_series", {}))
        self.financial_validator = FinancialDataValidator(self.config.get("financial", {}))
        self.correlation_validator = FeatureCorrelationValidator(self.config.get("correlation", {}))
        self.target_validator = TargetVariableValidator(self.config.get("target", {}))
        self.quality_scorer = DataQualityScorer(self.config.get("scoring", {}))
        self.drift_detector = None
        
    def _get_default_config(self) -> ConfigDict:
        """Get default configuration for ML validation."""
        return {
            "target_column": None,
            "timestamp_column": None,
            "expected_interval": 3600,  # 1 hour in seconds
            "validate_distributions": True,
            "validate_outliers": True,
            "validate_time_series": True,
            "validate_financial": True,
            "validate_correlations": True,
            "validate_target": True,
            "detect_drift": False
        }
    
    async def initialize(self) -> bool:
        """Initialize AdvancedMLValidator."""
        try:
            self.logger.info("🚀 Initializing AdvancedMLValidator...")
            
            # Initialize all validators
            await self.statistical_validator.initialize()
            await self.time_series_validator.initialize()
            await self.financial_validator.initialize()
            await self.correlation_validator.initialize()
            await self.target_validator.initialize()
            await self.quality_scorer.initialize()
            
            self.is_initialized = True
            self.logger.info("✅ AdvancedMLValidator initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing AdvancedMLValidator: {e}")
            return False
    
    def set_reference_data(self, reference_data: DataFrame) -> None:
        """Set reference data for drift detection."""
        self.drift_detector = DataDriftDetector(reference_data)
        self.config["reference_data"] = reference_data
    
    def validate_ml_data(self, df: DataFrame, 
                        target_col: Optional[str] = None,
                        timestamp_col: Optional[str] = None) -> MLValidationResult:
        """Validate ML data comprehensively."""
        self.logger.info("🔍 Starting comprehensive ML data validation...")
        
        # Use config defaults if not provided
        target_col = target_col or self.config["target_column"]
        timestamp_col = timestamp_col or self.config["timestamp_column"]
        
        # Initialize result
        result = MLValidationResult(is_valid=True, quality_score=None)
        
        # Statistical validation
        if self.config["validate_distributions"]:
            result.distribution_issues = self.statistical_validator.validate_data_distributions(df)
        
        if self.config["validate_outliers"]:
            result.outlier_issues = self.statistical_validator.validate_outliers(df)
        
        # Time series validation
        if self.config["validate_time_series"] and timestamp_col and timestamp_col in df.columns:
            result.time_series_issues = self.time_series_validator.validate_time_series_quality(
                df, timestamp_col, self.config["expected_interval"]
            )
        
        # Financial data validation
        if self.config["validate_financial"]:
            result.financial_issues = self.financial_validator.validate_financial_data(df)
        
        # Feature correlation validation
        if self.config["validate_correlations"]:
            result.correlation_issues = self.correlation_validator.validate_feature_correlations(df, target_col)
        
        # Target variable validation
        if self.config["validate_target"] and target_col:
            result.target_issues = self.target_validator.validate_target_variable(df, target_col, timestamp_col)
        
        # Drift detection
        if self.config["detect_drift"] and self.drift_detector:
            result.drift_report = self.drift_detector.detect_drift(df)
        
        # Calculate quality score
        result.quality_score = self.quality_scorer.calculate_quality_score(df, result)
        
        # Determine overall validity
        total_issues = (
            len(result.correlation_issues) +
            len(result.target_issues) +
            len(result.distribution_issues) +
            len(result.outlier_issues) +
            len(result.time_series_issues) +
            len(result.financial_issues)
        )
        
        if result.drift_report:
            total_issues += len(result.drift_report.issues)
        
        result.is_valid = total_issues == 0
        result.summary = {
            "total_issues": total_issues,
            "quality_score": result.quality_score.overall,
            "quality_grade": result.quality_score.grade,
            "drift_detected": result.drift_report is not None
        }
        
        # Log results
        if result.is_valid:
            self.logger.info(f"✅ ML data validation passed (Score: {result.quality_score.overall:.3f}, Grade: {result.quality_score.grade})")
        else:
            self.logger.warning(f"⚠️ ML data validation found {total_issues} issues (Score: {result.quality_score.overall:.3f}, Grade: {result.quality_score.grade})")
        
        return result

# Convenience functions for easy usage
def validate_ml_data_quality(df: DataFrame, 
                            target_col: Optional[str] = None,
                            timestamp_col: Optional[str] = None,
                            config: Optional[ConfigDict] = None) -> MLValidationResult:
    """Convenience function for ML data quality validation."""
    validator = AdvancedMLValidator(config)
    return validator.validate_ml_data(df, target_col, timestamp_col)

def detect_data_drift(reference_data: DataFrame, 
                     current_data: DataFrame) -> DriftReport:
    """Convenience function for data drift detection."""
    detector = DataDriftDetector(reference_data)
    return detector.detect_drift(current_data)

def calculate_data_quality_score(df: DataFrame, 
                               validation_result: MLValidationResult,
                               weights: Optional[ConfigDict] = None) -> QualityScore:
    """Convenience function for data quality scoring."""
    scorer = DataQualityScorer(weights)
    return scorer.calculate_quality_score(df, validation_result)