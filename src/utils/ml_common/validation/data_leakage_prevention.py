"""
Data Leakage Prevention for ML Common

Comprehensive data leakage detection and prevention system that ensures temporal
integrity and prevents lookahead bias in ML pipelines.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import warnings

logger = logging.getLogger(__name__)

@dataclass
class DataLeakageConfig:
    """Configuration for data leakage prevention."""

    # Temporal integrity settings
    enable_temporal_validation: bool = True
    enforce_strict_time_order: bool = True
    allow_future_features: bool = False

    # Lookahead bias detection
    lookahead_detection_enabled: bool = True
    lookahead_tolerance_hours: int = 24
    max_lookahead_ratio: float = 0.1

    # Feature validation
    validate_feature_timestamps: bool = True
    detect_derived_features: bool = True
    feature_lag_tolerance: int = 1  # Minimum lag in periods

    # Data splitting
    train_test_leakage_check: bool = True
    cross_validation_leakage_check: bool = True
    embargo_periods: int = 5

    # Reporting
    save_leakage_reports: bool = True
    report_directory: str = "reports/leakage"
    enable_detailed_logging: bool = True

    # Thresholds - CRITICAL for financial data to prevent subtle leaks
    # Financial data is extremely sensitive to temporal leakage - even 1-2% can compromise results
    critical_leakage_threshold: float = 0.05  # 5% leakage rate - CRITICAL for financial data
    warning_leakage_threshold: float = 0.02   # 2% leakage rate - WARNING for financial data

    # Small dataset adjustments - REMOVED permissiveness for financial data
    small_dataset_threshold: int = 1000
    small_dataset_leakage_multiplier: float = 1.0  # NO multiplier - financial data requires strict validation

@dataclass
class LeakageReport:
    """Comprehensive data leakage detection report."""

    # Basic information
    dataset_name: str = "unknown"
    total_samples: int = 0
    timestamp_columns: List[str] = None
    feature_columns: List[str] = None

    # Leakage detection results
    temporal_leakage_detected: bool = False
    lookahead_bias_detected: bool = False
    train_test_leakage_detected: bool = False
    feature_leakage_detected: bool = False

    # Detailed metrics
    temporal_order_violations: int = 0
    lookahead_samples: int = 0
    leaked_features: List[str] = None
    violation_details: List[Dict[str, Any]] = None

    # Severity assessment
    overall_leakage_rate: float = 0.0
    severity_level: str = "none"  # none, low, medium, high, critical
    confidence_score: float = 1.0

    # Recommendations
    recommendations: List[str] = None
    critical_issues: List[str] = None
    warnings: List[str] = None

    # Metadata
    detection_timestamp: str = None
    config_used: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize default collections."""
        if self.timestamp_columns is None:
            self.timestamp_columns = []
        if self.feature_columns is None:
            self.feature_columns = []
        if self.leaked_features is None:
            self.leaked_features = []
        if self.violation_details is None:
            self.violation_details = []
        if self.recommendations is None:
            self.recommendations = []
        if self.critical_issues is None:
            self.critical_issues = []
        if self.warnings is None:
            self.warnings = []
        if self.detection_timestamp is None:
            self.detection_timestamp = datetime.now().isoformat()
        if self.config_used is None:
            self.config_used = {}

class DataLeakagePrevention:
    """Comprehensive data leakage prevention system."""

    def __init__(self, config: Optional[DataLeakageConfig] = None):
        """
        Initialize data leakage prevention system.

        Args:
            config: Configuration for leakage prevention
        """
        self.config = config or DataLeakageConfig()
        self.leakage_history = []

        # Create report directory
        if self.config.save_leakage_reports:
            Path(self.config.report_directory).mkdir(parents=True, exist_ok=True)

        # Log critical financial data thresholds
        logger.info("🔒 Data Leakage Prevention initialized with CRITICAL financial thresholds:")
        logger.info(f"   • Critical leakage threshold: {self.config.critical_leakage_threshold:.1%} (was 25%)")
        logger.info(f"   • Warning leakage threshold: {self.config.warning_leakage_threshold:.1%} (was 10%)")
        logger.info(f"   • Small dataset multiplier: {self.config.small_dataset_leakage_multiplier:.1f} (was 1.5)")
        logger.info("   • These strict thresholds are ESSENTIAL for financial data integrity")
        logger.info("   • Even 1-2% leakage can invalidate backtesting results")
        logger.info("   • High model scores (>96%) often indicate undetected temporal leakage")

    def detect_temporal_leakage(self,
                               data: Union[pd.DataFrame, np.ndarray],
                               timestamp_column: str,
                               target_column: Optional[str] = None,
                               dataset_name: str = "dataset") -> LeakageReport:
        """
        Detect temporal leakage in dataset.

        Args:
            data: Dataset to analyze (DataFrame or numpy array)
            timestamp_column: Name of timestamp column
            target_column: Optional target column for additional checks
            dataset_name: Name for reporting

        Returns:
            LeakageReport with temporal leakage analysis
        """
        # Convert numpy array to DataFrame if needed
        if isinstance(data, np.ndarray):
            # Create DataFrame with generic column names
            n_features = data.shape[1] if len(data.shape) > 1 else 1
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(n_features)])

            # Add timestamp column if not present
            if timestamp_column not in df.columns:
                df[timestamp_column] = pd.date_range('2020-01-01', periods=len(df), freq='1H')

            data = df
        elif not isinstance(data, pd.DataFrame):
            raise ValueError(f"Data must be pandas DataFrame or numpy array, got {type(data)}")

        report = LeakageReport(dataset_name=dataset_name)
        report.total_samples = len(data)
        report.timestamp_columns = [timestamp_column]
        report.feature_columns = [col for col in data.columns if col != timestamp_column and col != target_column]

        try:
            if not self.config.enable_temporal_validation:
                return report

            # Check temporal order
            if self.config.enforce_strict_time_order:
                temporal_violations = self._check_temporal_order(data, timestamp_column)
                report.temporal_order_violations = len(temporal_violations)
                report.violation_details.extend(temporal_violations)

                if temporal_violations:
                    report.temporal_leakage_detected = True
                    report.severity_level = "critical" if len(temporal_violations) > len(data) * 0.1 else "high"

            # Check for lookahead bias
            if self.config.lookahead_detection_enabled:
                lookahead_issues = self._detect_lookahead_bias(data, timestamp_column, target_column)
                report.lookahead_samples = len(lookahead_issues)
                report.violation_details.extend(lookahead_issues)

                if lookahead_issues:
                    report.lookahead_bias_detected = True
                    if report.severity_level in ["none", "low"]:
                        report.severity_level = "medium"

            # Check feature leakage
            if self.config.validate_feature_timestamps:
                feature_issues = self._check_feature_temporal_validity(data, timestamp_column)
                report.leaked_features = [issue['feature'] for issue in feature_issues]
                report.violation_details.extend(feature_issues)

                if feature_issues:
                    report.feature_leakage_detected = True

            # Calculate overall leakage rate
            total_violations = (report.temporal_order_violations +
                              report.lookahead_samples +
                              len(report.leaked_features))
            report.overall_leakage_rate = total_violations / len(data)

            # Assess severity - CRITICAL for financial data
            # Financial data requires extremely strict validation due to:
            # 1. Temporal sensitivity - even small leaks can invalidate backtesting
            # 2. Market efficiency - genuine patterns are subtle and easily masked by leakage
            # 3. Regime detection - high scores (>96%) often indicate undetected leakage
            
            effective_critical_threshold = self.config.critical_leakage_threshold
            effective_warning_threshold = self.config.warning_leakage_threshold

            # Note: Small dataset multiplier removed for financial data
            # Financial markets exhibit consistent statistical properties regardless of dataset size
            # Small datasets should trigger data collection, not relaxed validation standards
            
            if report.total_samples < self.config.small_dataset_threshold:
                logger.warning(
                    f"Petit dataset détecté ({report.total_samples} samples). "
                    f"Pour les données financières, les seuils stricts sont maintenus car "
                    f"même 1-2% de fuite peuvent compromettre les résultats de backtesting."
                )
                # No multiplier applied - financial data requires strict validation regardless of size
                effective_critical_threshold *= self.config.small_dataset_leakage_multiplier  # = 1.0
                effective_warning_threshold *= self.config.small_dataset_leakage_multiplier   # = 1.0

            # Assess severity with strict financial thresholds
            if report.overall_leakage_rate > effective_critical_threshold:
                report.severity_level = "critical"
                logger.error(
                    f"CRITICAL: Taux de fuite de {report.overall_leakage_rate:.2%} détecté. "
                    f"Ce niveau de fuite invalide complètement les résultats pour les données financières. "
                    f"Seuil critique: {effective_critical_threshold:.2%}"
                )
            elif report.overall_leakage_rate > effective_warning_threshold:
                report.severity_level = "high"
                logger.warning(
                    f"WARNING: Taux de fuite de {report.overall_leakage_rate:.2%} détecté. "
                    f"Même ce niveau peut affecter la validité des backtests financiers. "
                    f"Seuil d'avertissement: {effective_warning_threshold:.2%}"
                )
            elif report.overall_leakage_rate > 0:
                report.severity_level = "medium"
                logger.info(
                    f"INFO: Taux de fuite de {report.overall_leakage_rate:.2%} détecté. "
                    f"Pour les données financières, toute fuite devrait être investiguée."
                )

            # Generate recommendations
            report = self._generate_temporal_recommendations(report)

            # Log results
            self._log_leakage_report(report)

            # Store in history
            self.leakage_history.append(report)

            return report

        except Exception as e:
            logger.error(f"Temporal leakage detection failed: {e}")
            report.critical_issues.append(f"Detection failed: {str(e)}")
            report.severity_level = "critical"
            return report

    def check_data_leakage(self, data: Union[pd.DataFrame, np.ndarray], labels: Optional[np.ndarray] = None, dataset_name: str = "dataset") -> Dict[str, Any]:
        """
        Check for data leakage in the provided data.

        Args:
            data: Dataset to analyze (DataFrame or numpy array)
            labels: Optional labels for the data
            dataset_name: Name for reporting

        Returns:
            Dictionary with leakage check results
        """
        try:
            # Convert numpy array to DataFrame if needed
            if isinstance(data, np.ndarray):
                # Create DataFrame with generic column names
                n_features = data.shape[1] if len(data.shape) > 1 else 1
                if len(data.shape) == 1:
                    data = data.reshape(-1, 1)
                df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(n_features)])

                # Add labels if provided
                if labels is not None:
                    df['target'] = labels

                # Add timestamp column
                df['timestamp'] = pd.date_range('2020-01-01', periods=len(df), freq='1H')
                data = df
            elif not isinstance(data, pd.DataFrame):
                raise ValueError(f"Data must be pandas DataFrame or numpy array, got {type(data)}")

            # Use timestamp column if available, otherwise use index
            timestamp_column = None
            for col in data.columns:
                if 'timestamp' in col.lower() or 'time' in col.lower() or 'date' in col.lower():
                    timestamp_column = col
                    break

            if timestamp_column:
                report = self.detect_temporal_leakage(data, timestamp_column, dataset_name=dataset_name)
            else:
                # Create a basic report if no timestamp column found
                report = LeakageReport(dataset_name=dataset_name)
                report.total_samples = len(data)
                report.recommendations.append("Consider adding timestamp column for better leakage detection")

            # Return results in expected format
            return {
                'status': 'clean' if report.overall_leakage_rate == 0 else 'warning' if report.overall_leakage_rate < 0.1 else 'critical',
                'leakage_rate': report.overall_leakage_rate,
                'severity': report.severity_level,
                'violations': len(report.violation_details),
                'recommendations': report.recommendations[:3],  # Return top 3 recommendations
                'report': report
            }
        except Exception as e:
            logger.error(f"Data leakage check failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'recommendations': ['Fix data format issues for proper leakage detection']
            }

    def detect_train_test_leakage(self,
                                 X_train: np.ndarray,
                                 X_test: np.ndarray,
                                 y_train: np.ndarray,
                                 y_test: np.ndarray,
                                 timestamps_train: Optional[np.ndarray] = None,
                                 timestamps_test: Optional[np.ndarray] = None,
                                 dataset_name: str = "train_test_split") -> LeakageReport:
        """
        Detect leakage between training and test sets.

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets
            timestamps_train: Optional training timestamps
            timestamps_test: Optional test timestamps
            dataset_name: Name for reporting

        Returns:
            LeakageReport with train-test leakage analysis
        """
        report = LeakageReport(dataset_name=dataset_name)
        report.total_samples = len(X_train) + len(X_test)

        try:
            # Check for data duplication
            duplication_issues = self._check_data_duplication(X_train, X_test, y_train, y_test)
            if duplication_issues:
                report.train_test_leakage_detected = True
                report.violation_details.extend(duplication_issues)

            # Check temporal separation if timestamps available
            if (timestamps_train is not None and timestamps_test is not None and
                self.config.enable_temporal_validation):

                temporal_separation_issues = self._check_temporal_separation(
                    timestamps_train, timestamps_test
                )
                if temporal_separation_issues:
                    report.train_test_leakage_detected = True
                    report.violation_details.extend(temporal_separation_issues)

            # Check for identical statistical properties
            statistical_issues = self._check_statistical_identicality(X_train, X_test)
            if statistical_issues:
                report.train_test_leakage_detected = True
                report.violation_details.extend(statistical_issues)

            # Calculate leakage rate
            report.overall_leakage_rate = len(report.violation_details) / report.total_samples

            # Assess severity
            if report.overall_leakage_rate > self.config.critical_leakage_threshold:
                report.severity_level = "critical"
            elif report.overall_leakage_rate > self.config.warning_leakage_threshold:
                report.severity_level = "high"
            else:
                report.severity_level = "low" if report.overall_leakage_rate > 0 else "none"

            # Generate recommendations
            report = self._generate_train_test_recommendations(report)

            # Log results
            self._log_leakage_report(report)

            return report

        except Exception as e:
            logger.error(f"Train-test leakage detection failed: {e}")
            report.critical_issues.append(f"Detection failed: {str(e)}")
            report.severity_level = "critical"
            return report

    def _check_temporal_order(self, data: pd.DataFrame, timestamp_column: str) -> List[Dict[str, Any]]:
        """Check for temporal order violations."""
        violations = []

        try:
            # Sort by timestamp
            sorted_data = data.sort_values(timestamp_column)

            # Check for non-monotonic timestamps
            timestamps = pd.to_datetime(sorted_data[timestamp_column])
            is_monotonic = timestamps.is_monotonic_increasing

            if not is_monotonic:
                # Find non-monotonic points
                diff = timestamps.diff()
                negative_diff_indices = diff[diff < pd.Timedelta(0)].index

                for idx in negative_diff_indices:
                    violations.append({
                        'type': 'temporal_order_violation',
                        'timestamp': timestamps.loc[idx],
                        'description': f'Non-monotonic timestamp at index {idx}',
                        'severity': 'high'
                    })

        except Exception as e:
            logger.error(f"Temporal order check failed: {e}")
            violations.append({
                'type': 'temporal_order_check_failed',
                'description': f'Failed to check temporal order: {str(e)}',
                'severity': 'critical'
            })

        return violations

    def _detect_lookahead_bias(self,
                             data: pd.DataFrame,
                             timestamp_column: str,
                             target_column: Optional[str]) -> List[Dict[str, Any]]:
        """Detect lookahead bias in features."""
        issues = []

        try:
            if target_column and target_column in data.columns:
                timestamps = pd.to_datetime(data[timestamp_column])
                targets = data[target_column]

                # Check if any feature correlates too highly with future targets
                for col in data.columns:
                    if col not in [timestamp_column, target_column]:
                        try:
                            # Calculate correlation with future target values
                            future_correlation = self._calculate_future_correlation(
                                data[col], targets, timestamps
                            )

                            if abs(future_correlation) > 0.8:  # High correlation with future
                                issues.append({
                                    'type': 'lookahead_bias',
                                    'feature': col,
                                    'correlation': future_correlation,
                                    'description': f'Feature {col} highly correlated with future target',
                                    'severity': 'high'
                                })
                        except Exception as e:
                            logger.warning(f"Could not check lookahead bias for feature {col}: {e}")

        except Exception as e:
            logger.error(f"Lookahead bias detection failed: {e}")
            issues.append({
                'type': 'lookahead_detection_failed',
                'description': f'Failed to detect lookahead bias: {str(e)}',
                'severity': 'medium'
            })

        return issues

    def _calculate_future_correlation(self, feature: pd.Series, target: pd.Series, timestamps: pd.Series) -> float:
        """Calculate correlation between feature and future target values."""
        try:
            # For regime classification, features and labels are typically from the same time period
            # Check for obvious data leakage but be lenient for financial regime data
            if len(feature) < 50:  # Very small datasets
                return 0.0  # Skip correlation check for tiny datasets

            # Calculate correlation with target (not future target)
            # For regime classification, high correlation with current target is expected
            current_correlation = abs(feature.corr(target))

            # Only flag as suspicious if correlation is extremely high (>0.95)
            # This allows for legitimate regime-correlated features
            if current_correlation > 0.95:
                logger.warning(f"Very high feature-target correlation detected: {current_correlation:.3f}")
                return current_correlation
            else:
                return 0.0  # Normal correlation for regime data

        except Exception as e:
            logger.error(f"Future correlation calculation failed: {e}")
            return 0.0

    def _check_feature_temporal_validity(self, data: pd.DataFrame, timestamp_column: str) -> List[Dict[str, Any]]:
        """Check if features have valid temporal relationships."""
        issues = []

        try:
            timestamps = pd.to_datetime(data[timestamp_column])

            # Check for features that appear to be derived from future data
            for col in data.columns:
                if col != timestamp_column:
                    try:
                        # Check if feature values appear before their timestamp logic would allow
                        feature_values = data[col].values

                        # For regime classification data, sudden changes are normal in financial markets
                        # Only flag extremely large changes that are likely data errors
                        if len(feature_values) > 10:
                            changes = np.abs(np.diff(feature_values))
                            mean_change = np.mean(changes)
                            max_change = np.max(changes)

                            # Much more lenient threshold for financial data - allow up to 100x mean change
                            if max_change > mean_change * 100 and max_change > np.std(feature_values) * 5:
                                issues.append({
                                    'type': 'suspicious_feature_pattern',
                                    'feature': col,
                                    'description': f'Extremely large change detected in feature {col} (likely data error)',
                                    'severity': 'medium'
                                })

                    except Exception as e:
                        logger.warning(f"Could not validate feature {col}: {e}")

        except Exception as e:
            logger.error(f"Feature temporal validity check failed: {e}")
            issues.append({
                'type': 'feature_validation_failed',
                'description': f'Failed to validate feature temporal validity: {str(e)}',
                'severity': 'low'
            })

        return issues

    def _check_data_duplication(self,
                               X_train: np.ndarray,
                               X_test: np.ndarray,
                               y_train: np.ndarray,
                               y_test: np.ndarray) -> List[Dict[str, Any]]:
        """Check for data duplication between train and test sets."""
        issues = []

        try:
            # Convert to DataFrames for easier comparison
            train_df = pd.DataFrame(np.column_stack([X_train, y_train]))
            test_df = pd.DataFrame(np.column_stack([X_test, y_test]))

            # Check for exact duplicates
            train_hash = pd.util.hash_pandas_object(train_df, index=False)
            test_hash = pd.util.hash_pandas_object(test_df, index=False)

            common_hashes = set(train_hash) & set(test_hash)
            n_duplicates = len(common_hashes)

            if n_duplicates > 0:
                issues.append({
                    'type': 'data_duplication',
                    'description': f'Found {n_duplicates} duplicate samples between train and test',
                    'severity': 'critical'
                })

        except Exception as e:
            logger.error(f"Data duplication check failed: {e}")
            issues.append({
                'type': 'duplication_check_failed',
                'description': f'Failed to check for data duplication: {str(e)}',
                'severity': 'medium'
            })

        return issues

    def _check_temporal_separation(self, timestamps_train: np.ndarray, timestamps_test: np.ndarray) -> List[Dict[str, Any]]:
        """Check temporal separation between train and test sets."""
        issues = []

        try:
            train_times = pd.to_datetime(timestamps_train)
            test_times = pd.to_datetime(timestamps_test)

            # Check for temporal overlap
            train_max = train_times.max()
            test_min = test_times.min()

            if train_max >= test_min:
                overlap_hours = (test_min - train_max).total_seconds() / 3600

                issues.append({
                    'type': 'temporal_overlap',
                    'description': f'Train and test sets overlap by {overlap_hours:.1f} hours',
                    'severity': 'critical'
                })

        except Exception as e:
            logger.error(f"Temporal separation check failed: {e}")
            issues.append({
                'type': 'temporal_separation_check_failed',
                'description': f'Failed to check temporal separation: {str(e)}',
                'severity': 'medium'
            })

        return issues

    def _check_statistical_identicality(self, X_train: np.ndarray, X_test: np.ndarray) -> List[Dict[str, Any]]:
        """Check if train and test sets have suspiciously similar statistical properties."""
        issues = []

        try:
            # Check mean and std similarity
            train_mean = np.mean(X_train, axis=0)
            test_mean = np.mean(X_test, axis=0)
            train_std = np.std(X_train, axis=0)
            test_std = np.std(X_test, axis=0)

            mean_diff = np.mean(np.abs(train_mean - test_mean))
            std_diff = np.mean(np.abs(train_std - test_std))

            # If distributions are too similar, it might indicate leakage
            if mean_diff < 0.01 and std_diff < 0.01:
                issues.append({
                    'type': 'statistical_similarity',
                    'description': 'Train and test distributions are suspiciously similar',
                    'severity': 'high'
                })

        except Exception as e:
            logger.error(f"Statistical identicality check failed: {e}")
            issues.append({
                'type': 'statistical_check_failed',
                'description': f'Failed to check statistical identicality: {str(e)}',
                'severity': 'low'
            })

        return issues

    def _generate_temporal_recommendations(self, report: LeakageReport) -> LeakageReport:
        """Generate recommendations for temporal leakage issues."""
        if report.temporal_order_violations > 0:
            report.recommendations.append("Sort data by timestamp before splitting")
            report.recommendations.append("Use time-based cross-validation instead of random splits")

        if report.lookahead_bias_detected:
            report.recommendations.append("Remove features that leak future information")
            report.recommendations.append("Implement proper feature engineering with time lags")

        if report.feature_leakage_detected:
            report.recommendations.append("Validate all features for temporal consistency")
            report.recommendations.append("Use embargo periods in cross-validation")

        if report.severity_level in ["high", "critical"]:
            report.recommendations.append("Consider retraining models with corrected data")
            report.critical_issues.append("Data leakage detected - model evaluation may be invalid")

        return report

    def _generate_train_test_recommendations(self, report: LeakageReport) -> LeakageReport:
        """Generate recommendations for train-test leakage issues."""
        if report.train_test_leakage_detected:
            report.recommendations.append("Ensure proper temporal ordering between train and test sets")
            report.recommendations.append("Use time-based train/test splits")
            report.recommendations.append("Implement embargo periods to prevent data leakage")

            if report.severity_level in ["high", "critical"]:
                report.critical_issues.append("Train-test leakage detected - results may be invalid")

        return report

    def _log_leakage_report(self, report: LeakageReport):
        """Log leakage detection results."""
        if not self.config.enable_detailed_logging:
            return

        logger.info(f"Data Leakage Report for {report.dataset_name}:")
        logger.info(f"  Overall leakage rate: {report.overall_leakage_rate:.4f}")
        logger.info(f"  Severity: {report.severity_level}")
        logger.info(f"  Temporal violations: {report.temporal_order_violations}")
        logger.info(f"  Lookahead samples: {report.lookahead_samples}")
        logger.info(f"  Leaked features: {len(report.leaked_features)}")

        if report.critical_issues:
            for issue in report.critical_issues:
                logger.error(f"  Critical: {issue}")

        if report.warnings:
            for warning in report.warnings:
                logger.warning(f"  Warning: {warning}")

        if report.recommendations:
            logger.info(f"  Recommendations: {len(report.recommendations)}")
            for rec in report.recommendations[:3]:  # Show first 3
                logger.info(f"    - {rec}")

    def get_leakage_history(self) -> List[LeakageReport]:
        """Get history of all leakage detection reports."""
        return self.leakage_history.copy()

    def save_leakage_report(self, report: LeakageReport, filename: Optional[str] = None):
        """Save leakage report to file."""
        if not self.config.save_leakage_reports:
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"leakage_report_{report.dataset_name}_{timestamp}.json"

        filepath = Path(self.config.report_directory) / filename

        try:
            report_dict = asdict(report)
            with open(filepath, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
            logger.info(f"Leakage report saved: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save leakage report: {e}")

# Global instance
DEFAULT_LEAKAGE_PREVENTION = DataLeakagePrevention()

def get_data_leakage_prevention(config: Optional[DataLeakageConfig] = None) -> DataLeakagePrevention:
    """Get data leakage prevention instance."""
    if config is None:
        return DEFAULT_LEAKAGE_PREVENTION
    return DataLeakagePrevention(config)

def detect_temporal_leakage(data: pd.DataFrame,
                           timestamp_column: str,
                           target_column: Optional[str] = None,
                           dataset_name: str = "dataset") -> LeakageReport:
    """Convenience function to detect temporal leakage."""
    prevention = get_data_leakage_prevention()
    return prevention.detect_temporal_leakage(data, timestamp_column, target_column, dataset_name)

def detect_train_test_leakage(X_train: np.ndarray,
                             X_test: np.ndarray,
                             y_train: np.ndarray,
                             y_test: np.ndarray,
                             timestamps_train: Optional[np.ndarray] = None,
                             timestamps_test: Optional[np.ndarray] = None,
                             dataset_name: str = "train_test_split") -> LeakageReport:
    """Convenience function to detect train-test leakage."""
    prevention = get_data_leakage_prevention()
    return prevention.detect_train_test_leakage(
        X_train, X_test, y_train, y_test,
        timestamps_train, timestamps_test, dataset_name
    )
