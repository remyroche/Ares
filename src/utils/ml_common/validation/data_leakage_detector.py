"""
Data Leakage Detector for ML Common

Provides comprehensive data leakage detection and prevention capabilities
for machine learning pipelines, especially for time series and financial data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
import warnings

logger = logging.getLogger(__name__)


@dataclass
class DataLeakageReport:
    """Report containing data leakage detection results."""

    has_leakage: bool
    leakage_score: float
    temporal_violations: List[str]
    feature_contamination: List[str]
    lookahead_bias: List[str]
    recommendations: List[str]


class DataLeakageDetector:
    """Comprehensive data leakage detection system.

    This class provides methods to detect various types of data leakage
    including temporal leakage, lookahead bias, and feature contamination.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize data leakage detector.

        Args:
            config: Configuration dictionary for detection parameters
        """
        self.config = {
            'temporal_tolerance': 1,  # Minimum time gap between train/test
            'lookahead_tolerance': 24,  # Hours of lookahead tolerance
            'feature_contamination_threshold': 0.1,  # Max allowed contamination
            'enable_strict_mode': True,
            **(config or {})
        }

    def detect_temporal_leakage(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                               y_train: Optional[pd.Series] = None,
                               y_test: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Detect temporal data leakage between train and test sets.

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training targets (optional)
            y_test: Test targets (optional)

        Returns:
            Dictionary containing temporal leakage analysis
        """
        results = {
            'temporal_violations': [],
            'max_future_leakage': 0,
            'min_time_gap': float('inf'),
            'recommendations': []
        }

        try:
            # Check for datetime columns
            datetime_cols = []
            for col in X_train.columns:
                if pd.api.types.is_datetime64_any_dtype(X_train[col]):
                    datetime_cols.append(col)

            if not datetime_cols:
                return results

            # Analyze temporal relationships
            for col in datetime_cols:
                if col in X_test.columns:
                    train_times = X_train[col].dropna()
                    test_times = X_test[col].dropna()

                    if len(train_times) > 0 and len(test_times) > 0:
                        # Check for future data in training set
                        max_train_time = train_times.max()
                        min_test_time = test_times.min()

                        if max_train_time > min_test_time:
                            violation = f"Training data contains future timestamps beyond test period in column '{col}'"
                            results['temporal_violations'].append(violation)

                            # Calculate time gap
                            if hasattr(max_train_time, 'tz_localize'):
                                # Handle timezone-aware timestamps
                                try:
                                    time_gap = (min_test_time - max_train_time).total_seconds() / 3600
                                except:
                                    time_gap = 0
                            else:
                                time_gap = (min_test_time - max_train_time).total_seconds() / 3600

                            results['max_future_leakage'] = max(results['max_future_leakage'], abs(time_gap))
                            results['min_time_gap'] = min(results['min_time_gap'], time_gap)

            # Generate recommendations
            if results['temporal_violations']:
                results['recommendations'].append(
                    "Remove or correct temporal violations before model training"
                )
                results['recommendations'].append(
                    "Ensure strict chronological order: train < validation < test"
                )

        except Exception as e:
            logger.warning(f"Error in temporal leakage detection: {e}")

        return results

    def detect_lookahead_bias(self, features: pd.DataFrame,
                             target: pd.Series,
                             target_column: str) -> Dict[str, Any]:
        """Detect potential lookahead bias in feature-target relationships.

        Args:
            features: Feature matrix
            target: Target variable
            target_column: Name of the target column

        Returns:
            Dictionary containing lookahead bias analysis
        """
        results = {
            'lookahead_violations': [],
            'suspicious_features': [],
            'risk_score': 0.0,
            'recommendations': []
        }

        try:
            # Check for features that might contain future information
            suspicious_features = []

            # Look for features with very high correlation to target
            # (might indicate data leakage)
            for col in features.columns:
                if features[col].dtype in ['float64', 'int64']:
                    try:
                        correlation = features[col].corr(target)
                        if abs(correlation) > 0.95:  # Very high correlation
                            suspicious_features.append({
                                'feature': col,
                                'correlation': correlation,
                                'risk': 'high'
                            })
                    except:
                        continue

            results['suspicious_features'] = suspicious_features

            # Calculate overall risk score
            if suspicious_features:
                high_risk_count = sum(1 for f in suspicious_features if f['risk'] == 'high')
                results['risk_score'] = min(1.0, high_risk_count / len(features.columns))

                if results['risk_score'] > 0.1:
                    results['recommendations'].append(
                        "Review features with high correlation to target for potential data leakage"
                    )

        except Exception as e:
            logger.warning(f"Error in lookahead bias detection: {e}")

        return results

    def detect_feature_contamination(self, train_features: pd.DataFrame,
                                   test_features: pd.DataFrame) -> Dict[str, Any]:
        """Detect feature contamination between train and test sets.

        Args:
            train_features: Training feature matrix
            test_features: Test feature matrix

        Returns:
            Dictionary containing feature contamination analysis
        """
        results = {
            'contaminated_features': [],
            'contamination_score': 0.0,
            'recommendations': []
        }

        try:
            common_features = set(train_features.columns) & set(test_features.columns)

            for feature in common_features:
                # Check for identical or very similar distributions
                train_vals = train_features[feature].dropna()
                test_vals = test_features[feature].dropna()

                if len(train_vals) > 0 and len(test_vals) > 0:
                    # Simple contamination check based on distribution overlap
                    try:
                        # Check if distributions are suspiciously similar
                        contamination_score = self._calculate_distribution_similarity(
                            train_vals, test_vals
                        )

                        if contamination_score > self.config['feature_contamination_threshold']:
                            results['contaminated_features'].append({
                                'feature': feature,
                                'contamination_score': contamination_score
                            })
                    except:
                        continue

            # Overall contamination score
            if results['contaminated_features']:
                results['contamination_score'] = np.mean([
                    f['contamination_score'] for f in results['contaminated_features']
                ])

                if results['contamination_score'] > 0.3:
                    results['recommendations'].append(
                        "High feature contamination detected - review data splitting strategy"
                    )

        except Exception as e:
            logger.warning(f"Error in feature contamination detection: {e}")

        return results

    def _calculate_distribution_similarity(self, train_vals: pd.Series,
                                         test_vals: pd.Series) -> float:
        """Calculate similarity between two distributions.

        Args:
            train_vals: Training values
            test_vals: Test values

        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Simple overlap coefficient based on value ranges
            train_min, train_max = train_vals.min(), train_vals.max()
            test_min, test_max = test_vals.min(), test_vals.max()

            # Calculate overlap
            overlap_start = max(train_min, test_min)
            overlap_end = min(train_max, test_max)

            if overlap_start >= overlap_end:
                return 0.0

            overlap_length = overlap_end - overlap_start
            union_length = max(train_max, test_max) - min(train_min, test_min)

            return overlap_length / union_length if union_length > 0 else 0.0

        except:
            return 0.0

    def generate_report(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                       y_train: Optional[pd.Series] = None,
                       y_test: Optional[pd.Series] = None,
                       features: Optional[pd.DataFrame] = None,
                       target: Optional[pd.Series] = None,
                       target_column: str = 'target') -> DataLeakageReport:
        """Generate comprehensive data leakage report.

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets
            features: Feature matrix for lookahead analysis
            target: Target variable for lookahead analysis
            target_column: Name of target column

        Returns:
            Comprehensive data leakage report
        """
        # Run all detection methods
        temporal_results = self.detect_temporal_leakage(X_train, X_test, y_train, y_test)

        lookahead_results = {}
        if features is not None and target is not None:
            lookahead_results = self.detect_lookahead_bias(features, target, target_column)

        contamination_results = self.detect_feature_contamination(X_train, X_test)

        # Combine results
        all_violations = (
            temporal_results['temporal_violations'] +
            lookahead_results.get('lookahead_violations', []) +
            [f"Feature contamination: {f['feature']}" for f in contamination_results['contaminated_features']]
        )

        all_recommendations = (
            temporal_results['recommendations'] +
            lookahead_results.get('recommendations', []) +
            contamination_results['recommendations']
        )

        # Calculate overall leakage score
        leakage_score = 0.0
        if temporal_results['max_future_leakage'] > 0:
            leakage_score += 0.5
        if lookahead_results.get('risk_score', 0) > 0.1:
            leakage_score += 0.3
        if contamination_results['contamination_score'] > 0.3:
            leakage_score += 0.2

        has_leakage = leakage_score > 0.1

        return DataLeakageReport(
            has_leakage=has_leakage,
            leakage_score=min(1.0, leakage_score),
            temporal_violations=temporal_results['temporal_violations'],
            feature_contamination=[f['feature'] for f in contamination_results['contaminated_features']],
            lookahead_bias=lookahead_results.get('suspicious_features', []),
            recommendations=all_recommendations
        )
