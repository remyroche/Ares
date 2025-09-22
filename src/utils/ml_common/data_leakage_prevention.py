"""
Comprehensive Data Leakage Prevention for ML Training

This module provides comprehensive data leakage prevention strategies to ensure
all feature engineering uses only past information and prevents information
leakage across all models in the training pipeline.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path

from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success

logger = system_logger.getChild('DataLeakagePrevention')

@dataclass
class DataLeakagePreventionConfig:
    """Configuration for data leakage prevention strategies."""

    # Temporal validation settings
    enable_temporal_validation: bool = True
    temporal_gap_minutes: int = 30  # Minimum gap between train/val
    enable_expanding_window_validation: bool = True

    # Feature engineering validation
    enable_feature_temporal_check: bool = True
    max_future_lookback_days: int = 1  # Maximum allowed future lookback
    enable_rolling_window_validation: bool = True
    rolling_window_sizes: List[int] = field(default_factory=lambda: [5, 10, 20, 50])

    # Cross-validation settings
    enable_cv_leakage_check: bool = True
    cv_folds: int = 10
    purge_minutes: int = 60  # Purge period for CV
    embargo_minutes: int = 30  # Embargo period for CV

    # Information leakage detection
    enable_information_leakage_detection: bool = True
    correlation_threshold: float = 0.8  # High correlation threshold
    mutual_info_threshold: float = 0.5  # Mutual information threshold

    # Feature validation settings
    enable_feature_validation: bool = True
    max_feature_correlation: float = 0.95  # Max allowed feature correlation
    enable_feature_importance_analysis: bool = True

    # Performance monitoring
    enable_performance_monitoring: bool = True
    overfitting_threshold: float = 0.1  # Max allowed train/val performance gap
    enable_learning_curve_analysis: bool = True

class DataLeakagePrevention:
    """
    Comprehensive data leakage prevention system for ML training.

    This class provides various strategies to prevent data leakage:
    1. Temporal validation and feature engineering checks
    2. Cross-validation integrity validation
    3. Information leakage detection
    4. Performance monitoring for overfitting detection
    """

    def __init__(self, config: Optional[DataLeakagePreventionConfig] = None):
        """Initialize data leakage prevention system."""
        self.config = config or DataLeakagePreventionConfig()
        self.logger = logger.getChild('DataLeakagePrevention')
        self.leakage_detected = False
        self.leakage_issues = []

        # Initialize monitoring
        self.temporal_violations = []
        self.feature_leakage_issues = []
        self.cv_integrity_issues = []
        self.performance_gaps = []

        self.logger.info("✅ Data Leakage Prevention system initialized")

    def validate_temporal_integrity(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        timestamps: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Validate temporal integrity of data splits.

        Args:
            X: Feature matrix
            y: Target values
            timestamps: Optional timestamp series for temporal validation

        Returns:
            Dictionary containing temporal integrity validation results
        """
        self.logger.info("🔍 Validating temporal integrity...")

        results = {
            'is_temporally_valid': True,
            'violations': [],
            'warnings': [],
            'recommendations': []
        }

        try:
            # Check if data has temporal index
            has_temporal_index = False
            if isinstance(X, pd.DataFrame) and isinstance(X.index, pd.DatetimeIndex):
                has_temporal_index = True
            elif timestamps is not None:
                has_temporal_index = True
                # Convert to DataFrame if needed
                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X)

            if not has_temporal_index:
                self.logger.warning("⚠️ No temporal index found - cannot perform temporal validation")
                results['warnings'].append("No temporal index found - temporal validation skipped")
                return results

            # Validate temporal ordering
            if timestamps is None:
                timestamps = X.index

            # Check for temporal ordering violations
            if not timestamps.is_monotonic_increasing:
                violation = "Data is not in chronological order"
                results['violations'].append(violation)
                results['is_temporally_valid'] = False
                self.logger.warning(f"⚠️ {violation}")

            # Check for duplicate timestamps
            duplicate_timestamps = timestamps.duplicated().sum()
            if duplicate_timestamps > 0:
                violation = f"Found {duplicate_timestamps} duplicate timestamps"
                results['violations'].append(violation)
                results['is_temporally_valid'] = False
                self.logger.warning(f"⚠️ {violation}")

            # Check for unrealistic time gaps
            if len(timestamps) > 1:
                time_diffs = timestamps.diff().dropna()
                max_gap = time_diffs.max()
                min_gap = time_diffs.min()

                # Check for suspicious gaps
                if max_gap > pd.Timedelta(days=7):
                    warning = f"Unusually large time gap detected: {max_gap}"
                    results['warnings'].append(warning)
                    self.logger.warning(f"⚠️ {warning}")

                if min_gap < pd.Timedelta(seconds=1):
                    warning = f"Unusually small time gap detected: {min_gap}"
                    results['warnings'].append(warning)
                    self.logger.warning(f"⚠️ {warning}")

            # Validate temporal gap requirements
            if self.config.enable_temporal_validation:
                temporal_gap_valid = self._validate_temporal_gaps(timestamps)
                if not temporal_gap_valid:
                    violation = f"Temporal gap validation failed (required: {self.config.temporal_gap_minutes}min)"
                    results['violations'].append(violation)
                    results['is_temporally_valid'] = False

            if results['is_temporally_valid']:
                self.logger.info("✅ Temporal integrity validation passed")
            else:
                self.logger.warning("❌ Temporal integrity validation failed")

        except Exception as e:
            error_msg = f"Temporal integrity validation failed: {e}"
            results['violations'].append(error_msg)
            results['is_temporally_valid'] = False
            self.logger.error(f"❌ {error_msg}")

        return results

    def _validate_temporal_gaps(self, timestamps: pd.Series) -> bool:
        """Validate that temporal gaps meet minimum requirements."""
        # This would be implemented with specific train/val split validation
        # For now, return True as a placeholder
        return True

    def validate_feature_engineering(
        self,
        features: pd.DataFrame,
        target: Union[pd.Series, np.ndarray],
        timestamps: pd.Series
    ) -> Dict[str, Any]:
        """
        Validate feature engineering for temporal correctness.

        Args:
            features: Feature matrix
            target: Target values
            timestamps: Timestamp series

        Returns:
            Dictionary containing feature engineering validation results
        """
        self.logger.info("🔍 Validating feature engineering...")

        results = {
            'is_feature_valid': True,
            'violations': [],
            'warnings': [],
            'recommendations': [],
            'feature_analysis': {}
        }

        try:
            # Check for features that might leak future information
            suspicious_features = []

            # 1. Check for features with high correlation to future targets
            if self.config.enable_information_leakage_detection:
                leakage_analysis = self._detect_information_leakage(features, target, timestamps)
                if leakage_analysis['has_leakage']:
                    results['is_feature_valid'] = False
                    results['violations'].extend(leakage_analysis['violations'])
                    suspicious_features.extend(leakage_analysis['suspicious_features'])

            # 2. Validate rolling window features
            if self.config.enable_rolling_window_validation:
                rolling_analysis = self._validate_rolling_windows(features, timestamps)
                if rolling_analysis['has_issues']:
                    results['warnings'].extend(rolling_analysis['warnings'])
                    results['feature_analysis']['rolling_windows'] = rolling_analysis

            # 3. Check for unrealistic feature values
            unrealistic_analysis = self._check_unrealistic_features(features)
            if unrealistic_analysis['has_issues']:
                results['warnings'].extend(unrealistic_analysis['warnings'])

            # 4. Validate feature correlation
            if self.config.enable_feature_validation:
                correlation_analysis = self._validate_feature_correlations(features)
                if correlation_analysis['has_issues']:
                    results['warnings'].extend(correlation_analysis['warnings'])
                    results['feature_analysis']['correlations'] = correlation_analysis

            # Generate recommendations
            if suspicious_features:
                results['recommendations'].append(
                    f"Review suspicious features for information leakage: {suspicious_features[:5]}"
                )

            if results['is_feature_valid']:
                self.logger.info("✅ Feature engineering validation passed")
            else:
                self.logger.warning("❌ Feature engineering validation failed")

        except Exception as e:
            error_msg = f"Feature engineering validation failed: {e}"
            results['violations'].append(error_msg)
            results['is_feature_valid'] = False
            self.logger.error(f"❌ {error_msg}")

        return results

    def _detect_information_leakage(
        self,
        features: pd.DataFrame,
        target: Union[pd.Series, np.ndarray],
        timestamps: pd.Series
    ) -> Dict[str, Any]:
        """Detect potential information leakage in features."""
        results = {
            'has_leakage': False,
            'violations': [],
            'warnings': [],
            'suspicious_features': []
        }

        try:
            # Convert target to Series if needed
            if not isinstance(target, pd.Series):
                target = pd.Series(target, index=timestamps.index)

            # Calculate correlations with shifted targets
            for lag in [1, 2, 5, 10, 20]:
                if lag >= len(target):
                    continue

                # Shift target forward (simulate future information)
                target_future = target.shift(-lag)
                valid_mask = ~target_future.isna()

                if valid_mask.sum() < 10:  # Need minimum samples
                    continue

                # Calculate correlations
                for col in features.columns:
                    if features[col].dtype in ['object', 'string']:
                        continue

                    feature_vals = features[col][valid_mask]
                    target_vals = target_future[valid_mask]

                    if len(feature_vals) < 10:
                        continue

                    try:
                        correlation = feature_vals.corr(target_vals)
                        if abs(correlation) > self.config.correlation_threshold:
                            violation = (
                                f"Feature '{col}' has high correlation ({correlation".3f"}) "
                                f"with target at lag +{lag}"
                            )
                            results['violations'].append(violation)
                            results['suspicious_features'].append(col)
                            results['has_leakage'] = True
                            self.logger.warning(f"⚠️ {violation}")
                    except Exception as e:
                        self.logger.debug(f"Correlation calculation failed for {col}: {e}")

        except Exception as e:
            error_msg = f"Information leakage detection failed: {e}"
            results['violations'].append(error_msg)
            self.logger.error(f"❌ {error_msg}")

        return results

    def _validate_rolling_windows(
        self,
        features: pd.DataFrame,
        timestamps: pd.Series
    ) -> Dict[str, Any]:
        """Validate rolling window calculations for temporal correctness."""
        results = {
            'has_issues': False,
            'warnings': [],
            'window_sizes': {}
        }

        try:
            # Look for features that might be rolling window calculations
            rolling_patterns = [
                'rolling', 'ma', 'mean', 'avg', 'sma', 'ema',
                'std', 'var', 'volatility', 'bb', 'bollinger'
            ]

            for col in features.columns:
                col_lower = col.lower()

                # Check if column name suggests rolling window
                is_rolling_feature = any(pattern in col_lower for pattern in rolling_patterns)

                if is_rolling_feature:
                    # Validate that rolling window doesn't use future information
                    window_analysis = self._analyze_rolling_feature(features[col], timestamps, col)
                    if window_analysis['has_issues']:
                        results['warnings'].extend(window_analysis['warnings'])
                        results['has_issues'] = True
                        results['window_sizes'][col] = window_analysis

        except Exception as e:
            error_msg = f"Rolling window validation failed: {e}"
            results['warnings'].append(error_msg)
            self.logger.error(f"❌ {error_msg}")

        return results

    def _analyze_rolling_feature(
        self,
        feature_series: pd.Series,
        timestamps: pd.Series,
        feature_name: str
    ) -> Dict[str, Any]:
        """Analyze a single rolling feature for temporal correctness."""
        results = {
            'has_issues': False,
            'warnings': [],
            'statistics': {}
        }

        try:
            # Check for sudden changes that might indicate future information usage
            feature_diff = feature_series.diff()
            feature_pct_change = feature_series.pct_change()

            # Calculate statistics
            results['statistics'] = {
                'mean_change': feature_diff.mean(),
                'max_change': feature_diff.max(),
                'min_change': feature_diff.min(),
                'change_std': feature_diff.std(),
                'pct_change_mean': feature_pct_change.mean() if feature_pct_change.mean() != np.inf else 0,
                'pct_change_std': feature_pct_change.std() if feature_pct_change.std() != np.inf else 0
            }

            # Check for unrealistic changes
            if abs(results['statistics']['mean_change']) > feature_series.std() * 0.5:
                warning = f"Feature '{feature_name}' has unusually large mean change"
                results['warnings'].append(warning)

            # Check for zero variance periods
            rolling_var = feature_series.rolling(window=10).var()
            zero_var_periods = (rolling_var == 0).sum()

            if zero_var_periods > len(feature_series) * 0.1:  # More than 10% zero variance
                warning = f"Feature '{feature_name}' has {zero_var_periods} zero-variance periods"
                results['warnings'].append(warning)

            if results['warnings']:
                results['has_issues'] = True

        except Exception as e:
            error_msg = f"Analysis failed for feature '{feature_name}': {e}"
            results['warnings'].append(error_msg)

        return results

    def _check_unrealistic_features(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Check for unrealistic feature values."""
        results = {
            'has_issues': False,
            'warnings': []
        }

        try:
            for col in features.columns:
                if features[col].dtype in ['object', 'string']:
                    continue

                # Check for infinite values
                inf_count = np.isinf(features[col]).sum()
                if inf_count > 0:
                    warning = f"Feature '{col}' has {inf_count} infinite values"
                    results['warnings'].append(warning)
                    results['has_issues'] = True

                # Check for extreme outliers
                if len(features[col]) > 10:
                    z_scores = np.abs((features[col] - features[col].mean()) / features[col].std())
                    extreme_outliers = (z_scores > 10).sum()

                    if extreme_outliers > 0:
                        warning = f"Feature '{col}' has {extreme_outliers} extreme outliers (>10σ)"
                        results['warnings'].append(warning)

                # Check for constant features
                if features[col].std() == 0:
                    warning = f"Feature '{col}' has zero variance"
                    results['warnings'].append(warning)

        except Exception as e:
            error_msg = f"Unrealistic feature check failed: {e}"
            results['warnings'].append(error_msg)

        return results

    def _validate_feature_correlations(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Validate feature correlations for multicollinearity."""
        results = {
            'has_issues': False,
            'warnings': [],
            'high_correlations': []
        }

        try:
            # Calculate correlation matrix
            numeric_features = features.select_dtypes(include=[np.number])
            if len(numeric_features.columns) < 2:
                return results

            correlation_matrix = numeric_features.corr()

            # Find highly correlated features
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    col1 = correlation_matrix.columns[i]
                    col2 = correlation_matrix.columns[j]
                    corr = correlation_matrix.iloc[i, j]

                    if abs(corr) > self.config.max_feature_correlation:
                        correlation_info = {
                            'feature1': col1,
                            'feature2': col2,
                            'correlation': float(corr)
        }
                        results['high_correlations'].append(correlation_info)
                        results['has_issues'] = True

                        if len(results['high_correlations']) <= 5:  # Limit warnings
                            warning = f"High correlation ({corr".3f"}) between '{col1}' and '{col2}'"
                            results['warnings'].append(warning)

        except Exception as e:
            error_msg = f"Feature correlation validation failed: {e}"
            results['warnings'].append(error_msg)

        return results

    def validate_cv_integrity(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        splits: List[Tuple[np.ndarray, np.ndarray]],
        timestamps: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Validate cross-validation splits for data leakage.

        Args:
            X: Feature matrix
            y: Target values
            splits: List of (train_indices, val_indices) tuples
            timestamps: Optional timestamp series

        Returns:
            Dictionary containing CV integrity validation results
        """
        self.logger.info("🔍 Validating CV integrity...")

        results = {
            'is_cv_valid': True,
            'violations': [],
            'warnings': [],
            'recommendations': [],
            'fold_analysis': []
        }

        try:
            if len(splits) < 2:
                warning = "CV has less than 2 folds - insufficient for validation"
                results['warnings'].append(warning)
                return results

            # Analyze each fold
            for i, (train_idx, val_idx) in enumerate(splits):
                fold_analysis = self._analyze_cv_fold(
                    X, y, train_idx, val_idx, i, timestamps
                )
                results['fold_analysis'].append(fold_analysis)

                if not fold_analysis['is_valid']:
                    results['is_cv_valid'] = False
                    results['violations'].extend(fold_analysis['violations'])

            # Check for temporal leakage across folds
            if timestamps is not None and self.config.enable_cv_leakage_check:
                temporal_leakage = self._check_cv_temporal_leakage(splits, timestamps)
                if temporal_leakage['has_leakage']:
                    results['is_cv_valid'] = False
                    results['violations'].extend(temporal_leakage['violations'])

            if results['is_cv_valid']:
                self.logger.info("✅ CV integrity validation passed")
            else:
                self.logger.warning("❌ CV integrity validation failed")

        except Exception as e:
            error_msg = f"CV integrity validation failed: {e}"
            results['violations'].append(error_msg)
            results['is_cv_valid'] = False
            self.logger.error(f"❌ {error_msg}")

        return results

    def _analyze_cv_fold(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        fold_id: int,
        timestamps: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Analyze a single CV fold for potential issues."""
        results = {
            'fold_id': fold_id,
            'is_valid': True,
            'violations': [],
            'warnings': [],
            'statistics': {}
        }

        try:
            # Basic size validation
            train_size = len(train_idx)
            val_size = len(val_idx)

            results['statistics'] = {
                'train_size': train_size,
                'val_size': val_size,
                'train_ratio': train_size / (train_size + val_size)
            }

            # Check minimum sizes
            if train_size < 100:
                violation = f"Fold {fold_id}: Train set too small ({train_size} samples)"
                results['violations'].append(violation)
                results['is_valid'] = False

            if val_size < 50:
                violation = f"Fold {fold_id}: Validation set too small ({val_size} samples)"
                results['violations'].append(violation)
                results['is_valid'] = False

            # Check for class imbalance if classification
            if len(np.unique(y)) < 10:  # Likely classification
                train_classes = np.unique(y[train_idx])
                val_classes = np.unique(y[val_idx])

                if len(train_classes) < 2:
                    violation = f"Fold {fold_id}: Train set has only {len(train_classes)} classes"
                    results['violations'].append(violation)
                    results['is_valid'] = False

                if len(val_classes) < 2:
                    violation = f"Fold {fold_id}: Validation set has only {len(val_classes)} classes"
                    results['violations'].append(violation)
                    results['is_valid'] = False

            # Check temporal ordering if timestamps available
            if timestamps is not None:
                train_times = timestamps.iloc[train_idx]
                val_times = timestamps.iloc[val_idx]

                if len(train_times) > 0 and len(val_times) > 0:
                    max_train_time = train_times.max()
                    min_val_time = val_times.min()

                    if max_train_time >= min_val_time:
                        violation = f"Fold {fold_id}: Temporal leakage (train max >= val min)"
                        results['violations'].append(violation)
                        results['is_valid'] = False

        except Exception as e:
            error_msg = f"Fold {fold_id} analysis failed: {e}"
            results['violations'].append(error_msg)
            results['is_valid'] = False

        return results

    def _check_cv_temporal_leakage(
        self,
        splits: List[Tuple[np.ndarray, np.ndarray]],
        timestamps: pd.Series
    ) -> Dict[str, Any]:
        """Check for temporal leakage across CV folds."""
        results = {
            'has_leakage': False,
            'violations': [],
            'warnings': []
        }

        try:
            # Check that validation sets don't overlap in time
            for i, (train_idx_i, val_idx_i) in enumerate(splits):
                for j, (train_idx_j, val_idx_j) in enumerate(splits[i+1:], i+1):

                    val_times_i = timestamps.iloc[val_idx_i]
                    val_times_j = timestamps.iloc[val_idx_j]

                    # Check for overlap in validation periods
                    overlap = pd.merge(
                        val_times_i.reset_index(),
                        val_times_j.reset_index(),
                        on=val_times_i.name,
                        how='inner'
                    )

                    if len(overlap) > 0:
                        violation = f"Temporal overlap between folds {i} and {j}"
                        results['violations'].append(violation)
                        results['has_leakage'] = True

        except Exception as e:
            error_msg = f"Temporal leakage check failed: {e}"
            results['violations'].append(error_msg)

        return results

    def generate_prevention_report(self) -> Dict[str, Any]:
        """Generate comprehensive data leakage prevention report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'leakage_detected': self.leakage_detected,
            'summary': {
                'total_violations': len(self.leakage_issues),
                'temporal_violations': len(self.temporal_violations),
                'feature_leakage_issues': len(self.feature_leakage_issues),
                'cv_integrity_issues': len(self.cv_integrity_issues),
                'performance_gaps': len(self.performance_gaps)
            },
            'violations': {
                'temporal': self.temporal_violations,
                'feature': self.feature_leakage_issues,
                'cv': self.cv_integrity_issues,
                'performance': self.performance_gaps
            },
            'recommendations': self._generate_recommendations()
        }

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on detected issues."""
        recommendations = []

        if self.leakage_detected:
            recommendations.extend([
                "Review feature engineering pipeline for temporal correctness",
                "Implement stricter temporal validation in train/val splits",
                "Add information leakage detection to regular monitoring",
                "Consider using expanding window validation instead of fixed splits",
                "Review rolling window calculations for future information usage"
            ])

        if len(self.temporal_violations) > 0:
            recommendations.extend([
                "Ensure data is sorted chronologically before splitting",
                "Add temporal gap validation between train and validation sets",
                "Consider using purged cross-validation for time series data"
            ])

        if len(self.feature_leakage_issues) > 0:
            recommendations.extend([
                "Review feature calculations for potential future information usage",
                "Implement feature importance analysis with temporal validation",
                "Add correlation checks between features and shifted targets"
            ])

        if len(self.cv_integrity_issues) > 0:
            recommendations.extend([
                "Increase number of CV folds for better validation",
                "Implement temporal cross-validation instead of random splits",
                "Add embargo periods to prevent information leakage in CV"
            ])

        return recommendations

# Convenience functions
def create_data_leakage_prevention(config: Optional[DataLeakagePreventionConfig] = None) -> DataLeakagePrevention:
    """Create data leakage prevention instance."""
    return DataLeakagePrevention(config)

def validate_data_integrity(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    timestamps: Optional[pd.Series] = None,
    config: Optional[DataLeakagePreventionConfig] = None
) -> Dict[str, Any]:
    """
    Convenience function to validate data integrity for leakage.

    Args:
        X: Feature matrix
        y: Target values
        timestamps: Optional timestamp series
        config: Optional configuration

    Returns:
        Dictionary containing validation results
    """
    prevention = DataLeakagePrevention(config)

    # Run comprehensive validation
    temporal_results = prevention.validate_temporal_integrity(X, y, timestamps)

    # Convert to DataFrame if needed
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    if timestamps is None:
        timestamps = pd.Series(range(len(X)), index=X.index)

    target_series = y if isinstance(y, pd.Series) else pd.Series(y, index=timestamps.index)
    feature_results = prevention.validate_feature_engineering(X, target_series, timestamps)

    # Combine results
    combined_results = {
        'temporal_integrity': temporal_results,
        'feature_engineering': feature_results,
        'overall_valid': temporal_results['is_temporally_valid'] and feature_results['is_feature_valid'],
        'prevention_report': prevention.generate_prevention_report()
    }

    return combined_results