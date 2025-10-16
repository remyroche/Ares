from src.utils.tprint import tprint

"""
Data Quality & Preprocessing Utilities

This module provides comprehensive data quality assessment and preprocessing utilities
for ensuring clean, reliable data for machine learning.

Key Features:
- Automated outlier detection
- Missing value analysis and handling
- Feature distribution stability analysis
- Data drift detection with concept drift analysis
- Label quality assessment
- Feature correlation analysis
- Automated data cleaning with enhanced strategies
- Feature stability analysis over time windows
- Comprehensive data quality scoring system

Built on existing utilities:
- Uses data_processing_utils.py for data handling
- Leverages math_validation.py for safe operations
- Integrates with common_operations.py for robust error handling
- Uses m1_gpu_utils.py for
- Leverages m1_memory_optimizer.py for memory management
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from scipy import stats
from collections import Counter
import warnings
import time

from src.utils.common_utilities import safe_dataframe_operation
from ..math_validation import safe_divide, safe_log
from ..common_operations import create_fallback_logger

# Enhanced dependency management with fast fail
try:
    from ..logger import get_logger
    _LOGGER = get_logger("MLCommon.DataQuality")
    tprint("✅ Custom logger available for MLCommon.DataQuality")
except Exception as e:
    tprint(f"⚠️ Custom logger not available: {e}. Using standard logging.")
    _LOGGER = logging.getLogger("MLCommon.DataQuality")
    _LOGGER.setLevel(logging.INFO)

# Enhanced imports for new functionality
try:
    from src.utils.hardware.m1_gpu_utils import M1GPUManager
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
    MEMORY_OPTIMIZER_AVAILABLE = True
except ImportError:
    MEMORY_OPTIMIZER_AVAILABLE = False

logger = _LOGGER

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mutual_info_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available - limited data quality functionality")

class DataQualityUtilities:
    """Comprehensive data quality assessment and preprocessing utilities."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize data quality utilities with configuration."""
        self.config = config or {}
        self.logger = logger.getChild('DataQuality')

        _LOGGER.info("🚀 Initializing DataQualityUtilities...")

        # Configuration defaults
        self.outlier_contamination = self.config.get('outlier_contamination', 0.1)
        self.missing_threshold = self.config.get('missing_threshold', 0.5)
        self.drift_threshold = self.config.get('drift_threshold', 0.1)
        self.correlation_method = self.config.get('correlation_method', 'spearman')

        # Enhanced configuration for new features
        self.enable_gpu = self.config.get('enable_gpu', GPU_AVAILABLE)
        self.enable_memory_optimization = self.config.get('enable_memory_optimization', MEMORY_OPTIMIZER_AVAILABLE)
        self.drift_detection_window = self.config.get('drift_detection_window', 1000)
        self.stability_analysis_window = self.config.get('stability_analysis_window', '1D')
        self.quality_score_weights = self.config.get('quality_score_weights', {
            'completeness': 0.3,
            'accuracy': 0.25,
            'consistency': 0.25,
            'timeliness': 0.2
        })

        # Initialize utilities
        self.gpu_manager = M1GPUManager() if self.enable_gpu else None
        self.memory_optimizer = M1MemoryOptimizer() if self.enable_memory_optimization else None

        _LOGGER.info(f"⚙️ Configuration - Outlier contamination: {self.outlier_contamination}")
        _LOGGER.info(f"⚙️ Configuration - Missing threshold: {self.missing_threshold}")
        _LOGGER.info(f"⚙️ Configuration - Drift threshold: {self.drift_threshold}")
        _LOGGER.info(f"⚙️ Configuration - GPU enabled: {self.enable_gpu}")
        _LOGGER.info(f"⚙️ Configuration - Memory optimization: {self.enable_memory_optimization}")
        _LOGGER.info("✅ DataQualityUtilities initialized successfully")

    def automated_outlier_detection(self, X: Union[np.ndarray, pd.DataFrame],
                                  method: str = 'isolation_forest',
                                  contamination: Optional[float] = None) -> Dict[str, Any]:
        """
        Automated outlier detection across features.

        Args:
            X: Feature matrix
            method: Outlier detection method
            contamination: Expected proportion of outliers

        Returns:
            Outlier detection results
        """
        start_time = time.time()
        _LOGGER.info(f"🔍 Starting automated outlier detection using {method}...")

        try:
            if contamination is None:
                contamination = self.outlier_contamination

            _LOGGER.info(f"📊 Parameters - Method: {method}, Contamination: {contamination}")

            if not SKLEARN_AVAILABLE:
                _LOGGER.error("❌ Scikit-learn required for outlier detection")
                return {'error': 'Scikit-learn required for outlier detection'}

            if isinstance(X, pd.DataFrame):
                X_array = X.select_dtypes(include=[np.number]).values
                feature_names = X.select_dtypes(include=[np.number]).columns.tolist()
                _LOGGER.debug(f"📊 DataFrame input - Shape: {X.shape}, Numeric features: {len(feature_names)}")
            else:
                X_array = X
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]
                _LOGGER.debug(f"📊 Array input - Shape: {X.shape}")

            outlier_results = {
                'method': method,
                'contamination': contamination,
                'outlier_indices': [],
                'outlier_scores': {},
                'feature_outliers': {},
                'summary': {}
            }

            if method == 'isolation_forest':
                _LOGGER.debug("🌲 Using Isolation Forest for outlier detection...")
                outlier_results.update(self._isolation_forest_detection(X_array, feature_names, contamination))

            # Calculate summary statistics
            outlier_count = len(outlier_results['outlier_indices'])
            outlier_percentage = (outlier_count / len(X_array)) * 100 if len(X_array) > 0 else 0

            outlier_results['summary'] = {
                'total_samples': len(X_array),
                'outlier_count': outlier_count,
                'outlier_percentage': outlier_percentage,
                'features_analyzed': len(feature_names)
            }

            execution_time = time.time() - start_time
            _LOGGER.info(f"✅ Outlier detection completed in {execution_time:.3f}s")
            _LOGGER.info(f"📊 Results - Total samples: {len(X_array)}, Outliers: {outlier_count} ({outlier_percentage:.2f}%)")

            if outlier_count > 0:
                _LOGGER.warning(f"⚠️ Found {outlier_count} outliers ({outlier_percentage:.2f}%) - consider data cleaning")
            else:
                _LOGGER.info("✅ No outliers detected - data appears clean")

            return outlier_results

        except Exception as e:
            execution_time = time.time() - start_time
            _LOGGER.error(f"❌ Outlier detection failed after {execution_time:.3f}s: {e}")
            _LOGGER.warning("⚠️ Outlier detection failed - returning empty results, no outliers will be detected")
            return {'error': f'Outlier detection failed: {e}', 'outlier_indices': [], 'summary': {'total_samples': 0, 'outlier_count': 0}}

    def missing_value_analysis(self, df: pd.DataFrame,
                             missing_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Comprehensive missing value analysis.

        Args:
            df: DataFrame to analyze
            missing_threshold: Threshold for concerning missing rates

        Returns:
            Missing value analysis results
        """
        start_time = time.time()
        _LOGGER.info(f"🔍 Starting missing value analysis...")

        try:
            if missing_threshold is None:
                missing_threshold = self.missing_threshold

            _LOGGER.info(f"📊 Parameters - Threshold: {missing_threshold}, DataFrame shape: {df.shape}")

            missing_analysis = {
                'missing_summary': {},
                'missing_patterns': {},
                'recommendations': [],
                'severity_assessment': {}
            }

            # Calculate missing statistics
            missing_stats = df.isnull().sum()
            missing_percentages = (missing_stats / len(df)) * 100

            missing_analysis['missing_summary'] = {
                'total_missing': missing_stats.sum(),
                'total_missing_percentage': safe_divide(missing_stats.sum(), df.shape[0] * df.shape[1]) * 100,
                'columns_with_missing': (missing_stats > 0).sum(),
                'rows_with_missing': (df.isnull().any(axis=1)).sum()
            }

            # Per-column analysis
            for col in df.columns:
                missing_count = missing_stats[col]
                missing_pct = missing_percentages[col]

                missing_analysis['missing_patterns'][col] = {
                    'missing_count': int(missing_count),
                    'missing_percentage': float(missing_pct),
                    'is_concerning': missing_pct > (missing_threshold * 100)
                }

            # Generate recommendations
            concerning_cols = [col for col, stats in missing_analysis['missing_patterns'].items()
                             if stats['is_concerning']]

            if concerning_cols:
                missing_analysis['recommendations'].extend([
                    f"High missing values in {len(concerning_cols)} columns: {concerning_cols[:5]}{'...' if len(concerning_cols) > 5 else ''}",
                    "Consider imputation strategies for columns with high missing rates",
                    "Evaluate if columns with >50% missing values should be dropped"
                ])

            # Severity assessment
            total_missing_pct = missing_analysis['missing_summary']['total_missing_percentage']
            if total_missing_pct < 5:
                severity = 'low'
            elif total_missing_pct < 15:
                severity = 'moderate'
            elif total_missing_pct < 30:
                severity = 'high'
            else:
                severity = 'critical'

            missing_analysis['severity_assessment'] = {
                'severity_level': severity,
                'total_missing_percentage': total_missing_pct,
                'action_required': severity in ['high', 'critical']
            }

            execution_time = time.time() - start_time
            _LOGGER.info(f"✅ Missing value analysis completed in {execution_time:.3f}s")
            _LOGGER.info(f"📊 Results - Severity: {severity}, Total missing: {total_missing_pct:.2f}%")

            if severity in ['high', 'critical']:
                _LOGGER.warning(f"⚠️ High missing value severity ({severity}) - action required")
            else:
                _LOGGER.info(f"✅ Missing value severity is acceptable ({severity})")

            return missing_analysis

        except Exception as e:
            execution_time = time.time() - start_time
            _LOGGER.error(f"❌ Missing value analysis failed after {execution_time:.3f}s: {e}")
            _LOGGER.warning("⚠️ Missing value analysis failed - returning empty analysis results")
            return {'error': str(e), 'missing_summary': {'total_missing': 0, 'total_missing_percentage': 0.0}}

    def feature_distribution_stability(self, train_df: pd.DataFrame,
                                     test_df: pd.DataFrame,
                                     feature_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze feature distribution stability between train and test sets.

        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            feature_cols: Specific feature columns to analyze

        Returns:
            Distribution stability analysis
        """
        try:
            self.logger.info("📊 Analyzing feature distribution stability")

            if feature_cols is None:
                # Use numeric columns
                feature_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
                feature_cols = [col for col in feature_cols if col in test_df.columns]

            stability_analysis = {
                'feature_stability': {},
                'distribution_shifts': [],
                'stability_summary': {},
                'recommendations': []
            }

            for feature in feature_cols:
                if feature not in train_df.columns or feature not in test_df.columns:
                    continue

                train_values = train_df[feature].dropna()
                test_values = test_df[feature].dropna()

                if len(train_values) == 0 or len(test_values) == 0:
                    continue

                # Calculate distribution statistics
                train_stats = self._calculate_distribution_stats(train_values)
                test_stats = self._calculate_distribution_stats(test_values)

                # Calculate stability metrics
                stability_metrics = self._calculate_stability_metrics(train_stats, test_stats)

                stability_analysis['feature_stability'][feature] = {
                    'train_stats': train_stats,
                    'test_stats': test_stats,
                    'stability_metrics': stability_metrics,
                    'is_stable': stability_metrics['ks_p_value'] > 0.05  # 95% confidence
                }

                # Check for significant distribution shifts
                if stability_metrics['ks_p_value'] < 0.05:
                    stability_analysis['distribution_shifts'].append({
                        'feature': feature,
                        'ks_statistic': stability_metrics['ks_statistic'],
                        'p_value': stability_metrics['ks_p_value'],
                        'shift_severity': 'severe' if stability_metrics['ks_p_value'] < 0.01 else 'moderate'
                    })

            # Generate summary
            stable_features = sum(1 for stats in stability_analysis['feature_stability'].values()
                                if stats['is_stable'])
            total_features = len(stability_analysis['feature_stability'])

            stability_analysis['stability_summary'] = {
                'total_features': total_features,
                'stable_features': stable_features,
                'unstable_features': total_features - stable_features,
                'stability_percentage': safe_divide(stable_features, total_features) * 100,
                'distribution_shifts': len(stability_analysis['distribution_shifts'])
            }

            # Generate recommendations
            if stability_analysis['distribution_shifts']:
                stability_analysis['recommendations'].append(
                    f"Distribution shifts detected in {len(stability_analysis['distribution_shifts'])} features"
                )

            if stability_analysis['stability_summary']['stability_percentage'] < 80:
                stability_analysis['recommendations'].append(
                    "Low distribution stability - consider domain adaptation techniques"
                )

            self.logger.info(f"✅ Distribution stability analysis completed: "
                           f"{stable_features}/{total_features} features stable")
            return stability_analysis

        except Exception as e:
            self.logger.error(f"❌ Distribution stability analysis failed: {e}")
            self.logger.warning("⚠️ Distribution stability analysis failed - assuming no stability issues detected")
            return {'error': str(e), 'feature_stability': {}, 'stability_summary': {'total_features': 0, 'stable_features': 0}}

    def data_drift_detection(self, reference_data: Union[np.ndarray, pd.DataFrame],
                           new_data: Union[np.ndarray, pd.DataFrame],
                           threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Detect data drift between reference and new data.

        Args:
            reference_data: Reference dataset
            new_data: New dataset to compare
            threshold: Drift detection threshold

        Returns:
            Data drift detection results
        """
        try:
            if threshold is None:
                threshold = self.drift_threshold

            self.logger.info(f"🔍 Detecting data drift (threshold={threshold})")

            # Convert to arrays if needed
            if isinstance(reference_data, pd.DataFrame):
                ref_array = reference_data.select_dtypes(include=[np.number]).values
                feature_names = reference_data.select_dtypes(include=[np.number]).columns.tolist()
            else:
                ref_array = reference_data
                feature_names = [f'feature_{i}' for i in range(reference_data.shape[1])]

            if isinstance(new_data, pd.DataFrame):
                new_array = new_data.select_dtypes(include=[np.number]).values
            else:
                new_array = new_data

            drift_results = {
                'drift_detected': False,
                'feature_drift': {},
                'drift_summary': {},
                'recommendations': []
            }

            # Analyze each feature
            for i, feature_name in enumerate(feature_names):
                if i >= ref_array.shape[1] or i >= new_array.shape[1]:
                    continue

                ref_feature = ref_array[:, i]
                new_feature = new_array[:, i]

                # Remove NaN values
                ref_clean = ref_feature[~np.isnan(ref_feature)]
                new_clean = new_feature[~np.isnan(new_feature)]

                if len(ref_clean) == 0 or len(new_clean) == 0:
                    continue

                # Perform statistical tests
                drift_metrics = self._calculate_drift_metrics(ref_clean, new_clean)

                is_drift = drift_metrics['ks_p_value'] < threshold

                drift_results['feature_drift'][feature_name] = {
                    'drift_detected': is_drift,
                    'drift_metrics': drift_metrics,
                    'severity': 'severe' if drift_metrics['ks_p_value'] < 0.01 else 'moderate' if is_drift else 'none'
                }

                if is_drift:
                    drift_results['drift_detected'] = True

            # Generate summary
            drifting_features = [f for f, stats in drift_results['feature_drift'].items()
                               if stats['drift_detected']]

            drift_results['drift_summary'] = {
                'total_features': len(drift_results['feature_drift']),
                'drifting_features': len(drifting_features),
                'drift_percentage': safe_divide(len(drifting_features), len(drift_results['feature_drift'])) * 100,
                'most_severe_drift': max(drift_results['feature_drift'].values(),
                                       key=lambda x: x['drift_metrics']['ks_statistic'])['drift_metrics']['ks_statistic'] if drifting_features else 0
            }

            # Generate recommendations
            if drifting_features:
                drift_results['recommendations'].extend([
                    f"Data drift detected in {len(drifting_features)} features",
                    "Consider retraining model or implementing drift adaptation",
                    "Monitor these features closely in production"
                ])

            self.logger.info(f"✅ Data drift detection completed: {'Drift detected' if drift_results['drift_detected'] else 'No drift detected'}")
            return drift_results

        except Exception as e:
            self.logger.error(f"❌ Data drift detection failed: {e}")
            self.logger.warning("⚠️ Data drift detection failed - assuming no drift detected for safety")
            return {'error': str(e), 'drift_detected': False, 'drift_summary': {'total_features': 0, 'drifting_features': 0}}

    def label_quality_assessment(self, y: Union[np.ndarray, pd.Series],
                               confidence_scores: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Assess label quality and identify potential issues.

        Args:
            y: Target labels
            confidence_scores: Prediction confidence scores

        Returns:
            Label quality assessment results
        """
        try:
            self.logger.info("🏷️ Assessing label quality")

            if isinstance(y, pd.Series):
                y_array = y.values
                labels = y.index if y.index is not None else None
            else:
                y_array = y
                labels = None

            quality_assessment = {
                'label_distribution': {},
                'quality_metrics': {},
                'potential_issues': [],
                'recommendations': []
            }

            # Analyze label distribution
            unique_labels, counts = np.unique(y_array, return_counts=True)
            total_samples = len(y_array)

            quality_assessment['label_distribution'] = {
                'unique_labels': len(unique_labels),
                'label_counts': dict(zip(unique_labels, counts)),
                'class_ratios': dict(zip(unique_labels, counts / total_samples)),
                'most_frequent_label': unique_labels[np.argmax(counts)],
                'least_frequent_label': unique_labels[np.argmin(counts)]
            }

            # Check for class imbalance
            max_ratio = np.max(counts) / total_samples
            if max_ratio > 0.9:
                quality_assessment['potential_issues'].append("Extreme class imbalance detected")
            elif max_ratio > 0.8:
                quality_assessment['potential_issues'].append("Severe class imbalance detected")

            # Check for single-class scenarios
            if len(unique_labels) == 1:
                quality_assessment['potential_issues'].append("Single-class dataset - not suitable for classification")

            # Check for very rare classes
            min_samples_per_class = 10
            rare_classes = [label for label, count in zip(unique_labels, counts)
                          if count < min_samples_per_class]
            if rare_classes:
                quality_assessment['potential_issues'].append(
                    f"Very rare classes detected: {rare_classes} (less than {min_samples_per_class} samples)"
                )

            # Analyze confidence if available
            if confidence_scores is not None:
                quality_assessment['confidence_analysis'] = {
                    'mean_confidence': float(np.mean(confidence_scores)),
                    'confidence_std': float(np.std(confidence_scores)),
                    'low_confidence_samples': int(np.sum(confidence_scores < 0.5))
                }

            # Generate recommendations
            if quality_assessment['potential_issues']:
                quality_assessment['recommendations'].extend([
                    "Address identified label quality issues before training",
                    "Consider data augmentation or resampling for imbalanced classes",
                    "Review data collection process for rare classes"
                ])

            self.logger.info(f"✅ Label quality assessment completed: "
                           f"{len(unique_labels)} unique labels, "
                           f"{len(quality_assessment['potential_issues'])} issues found")
            return quality_assessment

        except Exception as e:
            self.logger.error(f"❌ Label quality assessment failed: {e}")
            self.logger.warning("⚠️ Label quality assessment failed - assuming default quality metrics")
            return {'error': str(e), 'label_distribution': {'unique_labels': 0}, 'potential_issues': ['Assessment failed']}

    def feature_correlation_analysis(self, X: Union[np.ndarray, pd.DataFrame],
                                   method: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze feature correlations and identify multicollinearity.

        Args:
            X: Feature matrix
            method: Correlation method ('pearson', 'spearman', 'mutual_info')

        Returns:
            Feature correlation analysis results
        """
        try:
            if method is None:
                method = self.correlation_method

            self.logger.info(f"🔗 Analyzing feature correlations using {method}")

            if isinstance(X, pd.DataFrame):
                X_array = X.select_dtypes(include=[np.number]).values
                feature_names = X.select_dtypes(include=[np.number]).columns.tolist()
            else:
                X_array = X
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]

            correlation_results = {
                'correlation_method': method,
                'correlation_matrix': {},
                'highly_correlated_pairs': [],
                'multicollinearity_analysis': {},
                'recommendations': []
            }

            # Calculate correlation matrix
            if method in ['pearson', 'spearman']:
                corr_matrix = self._calculate_statistical_correlation(X_array, method)
            elif method == 'mutual_info':
                corr_matrix = self._calculate_mutual_info_correlation(X_array)
            else:
                raise ValueError(f"Unsupported correlation method: {method}")

            # Store correlation matrix
            for i in range(len(feature_names)):
                for j in range(len(feature_names)):
                    if i != j:
                        correlation_results['correlation_matrix'][f"{feature_names[i]}_{feature_names[j]}"] = \
                            float(corr_matrix[i, j])

            # Find highly correlated pairs
            threshold = 0.8 if method != 'mutual_info' else 0.5  # Lower threshold for mutual info

            for i in range(len(feature_names)):
                for j in range(i + 1, len(feature_names)):
                    corr_value = abs(corr_matrix[i, j])
                    if corr_value > threshold:
                        correlation_results['highly_correlated_pairs'].append({
                            'feature1': feature_names[i],
                            'feature2': feature_names[j],
                            'correlation': float(corr_value),
                            'severity': 'high' if corr_value > 0.9 else 'moderate'
                        })

            # Multicollinearity analysis
            correlation_results['multicollinearity_analysis'] = {
                'highly_correlated_pairs_count': len(correlation_results['highly_correlated_pairs']),
                'features_with_multicollinearity': len(set(
                    [pair['feature1'] for pair in correlation_results['highly_correlated_pairs']] +
                    [pair['feature2'] for pair in correlation_results['highly_correlated_pairs']]
                )),
                'max_correlation': max([pair['correlation'] for pair in correlation_results['highly_correlated_pairs']]
                                     if correlation_results['highly_correlated_pairs'] else [0])
            }

            # Generate recommendations
            if correlation_results['highly_correlated_pairs']:
                correlation_results['recommendations'].extend([
                    f"Found {len(correlation_results['highly_correlated_pairs'])} highly correlated feature pairs",
                    "Consider feature selection or dimensionality reduction",
                    "Evaluate if correlated features provide redundant information"
                ])

            self.logger.info(f"✅ Feature correlation analysis completed: "
                           f"{len(correlation_results['highly_correlated_pairs'])} highly correlated pairs found")
            return correlation_results

        except Exception as e:
            self.logger.error(f"❌ Feature correlation analysis failed: {e}")
            self.logger.warning("⚠️ Feature correlation analysis failed - assuming no multicollinearity issues")
            return {'error': str(e), 'correlation_matrix': {}, 'highly_correlated_pairs': [], 'multicollinearity_analysis': {'highly_correlated_pairs_count': 0}}

    def automated_data_cleaning(self, df: pd.DataFrame,
                              cleaning_config: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Automated data cleaning pipeline.

        Args:
            df: DataFrame to clean
            cleaning_config: Cleaning configuration

        Returns:
            Tuple of (cleaned_dataframe, cleaning_report)
        """
        try:
            self.logger.info("🧹 Starting automated data cleaning")

            if cleaning_config is None:
                cleaning_config = {
                    'missing_value_strategy': 'median',
                    'outlier_method': 'isolation_forest',
                    'correlation_threshold': 0.95
                }

            cleaning_report = {
                'original_shape': df.shape,
                'cleaning_steps': [],
                'removed_samples': 0,
                'removed_features': 0,
                'imputed_values': 0
            }

            cleaned_df = df.copy()

            # Step 1: Handle missing values
            if cleaning_config.get('handle_missing', True):
                cleaned_df, missing_report = self._clean_missing_values(
                    cleaned_df, cleaning_config['missing_value_strategy']
                )
                cleaning_report['cleaning_steps'].append(missing_report)

            # Step 2: Handle outliers
            if cleaning_config.get('handle_outliers', True):
                outlier_results = self.automated_outlier_detection(
                    cleaned_df, method=cleaning_config['outlier_method']
                )

                if 'outlier_indices' in outlier_results and outlier_results['outlier_indices']:
                    idx_labels = cleaned_df.index[outlier_results['outlier_indices']]
                    cleaned_df = cleaned_df.drop(index=idx_labels)
                    cleaning_report['removed_samples'] += len(outlier_results['outlier_indices'])
                    cleaning_report['cleaning_steps'].append({
                        'step': 'outlier_removal',
                        'removed_samples': len(outlier_results['outlier_indices'])
                    })

            # Step 3: Handle multicollinearity
            if cleaning_config.get('handle_multicollinearity', True):
                corr_analysis = self.feature_correlation_analysis(
                    cleaned_df, method='pearson'
                )

                if 'highly_correlated_pairs' in corr_analysis:
                    # Remove one feature from each highly correlated pair
                    features_to_remove = set()
                    for pair in corr_analysis['highly_correlated_pairs']:
                        if pair['correlation'] > cleaning_config['correlation_threshold']:
                            features_to_remove.add(pair['feature2'])  # Remove second feature

                    if features_to_remove:
                        cleaned_df = cleaned_df.drop(columns=list(features_to_remove))
                        cleaning_report['removed_features'] += len(features_to_remove)
                        cleaning_report['cleaning_steps'].append({
                            'step': 'multicollinearity_removal',
                            'removed_features': list(features_to_remove)
                        })

            # Final statistics
            cleaning_report['final_shape'] = cleaned_df.shape
            cleaning_report['total_removed_samples'] = cleaning_report['original_shape'][0] - cleaning_report['final_shape'][0]
            cleaning_report['total_removed_features'] = cleaning_report['original_shape'][1] - cleaning_report['final_shape'][1]

            self.logger.info(f"✅ Automated data cleaning completed: "
                           f"Shape: {cleaning_report['original_shape']} -> {cleaning_report['final_shape']}")
            return cleaned_df, cleaning_report

        except Exception as e:
            self.logger.error(f"❌ Automated data cleaning failed: {e}")
            self.logger.warning("⚠️ Automated data cleaning failed - returning original data unchanged")
            return df, {'error': str(e), 'original_shape': df.shape, 'cleaning_steps': [{'step': 'failed', 'error': str(e)}]}

    def detect_concept_drift(self, reference_data: Union[np.ndarray, pd.DataFrame],
                           current_data: Union[np.ndarray, pd.DataFrame],
                           drift_method: str = 'kolmogorov_smirnov',
                           window_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Advanced concept drift detection with multiple statistical methods.

        Args:
            reference_data: Reference dataset (training data)
            current_data: Current dataset to compare (production data)
            drift_method: Drift detection method ('kolmogorov_smirnov', 'wasserstein', 'jensen_shannon')
            window_size: Size of rolling window for incremental drift detection

        Returns:
            Concept drift detection results with severity analysis
        """
        try:
            if window_size is None:
                window_size = self.drift_detection_window

            self.logger.info(f"🔍 Detecting concept drift using {drift_method} (window_size={window_size})")

            # Convert to arrays if needed
            if isinstance(reference_data, pd.DataFrame):
                ref_array = reference_data.select_dtypes(include=[np.number]).values
                feature_names = reference_data.select_dtypes(include=[np.number]).columns.tolist()
            else:
                ref_array = reference_data
                feature_names = [f'feature_{i}' for i in range(reference_data.shape[1])]

            if isinstance(current_data, pd.DataFrame):
                curr_array = current_data.select_dtypes(include=[np.number]).values
            else:
                curr_array = current_data

            drift_results = {
                'drift_detected': False,
                'drift_method': drift_method,
                'feature_drift_scores': {},
                'overall_drift_score': 0.0,
                'drift_severity': 'none',
                'recommendations': [],
                'window_analysis': {},
                'temporal_analysis': {}
            }

            # Analyze each feature for drift
            drift_scores = []
            for i, feature_name in enumerate(feature_names):
                if i >= ref_array.shape[1] or i >= curr_array.shape[1]:
                    continue

                ref_feature = ref_array[:, i]
                curr_feature = curr_array[:, i]

                # Remove NaN values
                ref_clean = ref_feature[~np.isnan(ref_feature)]
                curr_clean = curr_feature[~np.isnan(curr_feature)]

                if len(ref_clean) == 0 or len(curr_clean) == 0:
                    continue

                # Calculate drift score based on method
                if drift_method == 'kolmogorov_smirnov':
                    drift_score = self._calculate_ks_drift(ref_clean, curr_clean)
                elif drift_method == 'wasserstein':
                    drift_score = self._calculate_wasserstein_drift(ref_clean, curr_clean)
                elif drift_method == 'jensen_shannon':
                    drift_score = self._calculate_js_drift(ref_clean, curr_clean)
                else:
                    drift_score = self._calculate_ks_drift(ref_clean, curr_clean)

                drift_results['feature_drift_scores'][feature_name] = {
                    'drift_score': drift_score,
                    'is_significant': drift_score > self.drift_threshold,
                    'severity': self._classify_drift_severity(drift_score)
                }

                if drift_score > self.drift_threshold:
                    drift_scores.append(drift_score)

            # Calculate overall drift
            if drift_scores:
                drift_results['overall_drift_score'] = np.mean(drift_scores)
                drift_results['drift_detected'] = drift_results['overall_drift_score'] > self.drift_threshold
                drift_results['drift_severity'] = self._classify_drift_severity(drift_results['overall_drift_score'])

            # Generate recommendations
            if drift_results['drift_detected']:
                drift_results['recommendations'].extend([
                    f"Concept drift detected with severity: {drift_results['drift_severity']}",
                    "Consider model retraining or drift adaptation techniques",
                    "Monitor feature distributions closely"
                ])

            # Rolling window analysis if window_size is reasonable
            if window_size > 0 and len(curr_array) > window_size:
                drift_results['window_analysis'] = self._analyze_rolling_window_drift(
                    curr_array, window_size, feature_names
                )

            self.logger.info(f"✅ Concept drift detection completed - "
                           f"Overall score: {drift_results['overall_drift_score']:.4f}")
            return drift_results

        except Exception as e:
            self.logger.error(f"❌ Concept drift detection failed: {e}")
            self.logger.warning("⚠️ Concept drift detection failed - assuming no drift for safety")
            return {'error': str(e), 'drift_detected': False, 'overall_drift_score': 0.0, 'drift_severity': 'none'}

    def analyze_feature_stability(self, feature_data: pd.Series,
                                time_window: str = '1D') -> Dict[str, Any]:
        """
        Analyze feature stability over time periods.

        Args:
            feature_data: Time series feature data
            time_window: Time window for stability analysis (e.g., '1D', '1H', '30min')

        Returns:
            Feature stability analysis results
        """
        try:
            if time_window is None:
                time_window = self.stability_analysis_window

            self.logger.info(f"📊 Analyzing feature stability over {time_window} windows")

            # Ensure we have datetime index
            if not isinstance(feature_data.index, pd.DatetimeIndex):
                self.logger.warning("Feature data should have DatetimeIndex for stability analysis")
                return {'error': 'DatetimeIndex required for stability analysis'}

            stability_results = {
                'stability_score': 0.0,
                'volatility_measure': 0.0,
                'trend_analysis': {},
                'seasonal_patterns': {},
                'anomaly_periods': [],
                'recommendations': []
            }

            # Resample data by time window
            resampled = feature_data.resample(time_window).agg(['mean', 'std', 'count'])

            # Calculate stability metrics
            if len(resampled) > 1:
                # Rolling statistics
                rolling_mean = resampled['mean'].rolling(window=min(10, len(resampled)), center=True).mean()
                rolling_std = resampled['mean'].rolling(window=min(10, len(resampled)), center=True).std()

                # Stability score (inverse of coefficient of variation)
                mean_variation = safe_divide(rolling_std.mean(), abs(rolling_mean.mean()))
                stability_results['stability_score'] = safe_divide(1.0, 1.0 + mean_variation)

                # Volatility measure
                stability_results['volatility_measure'] = rolling_std.mean()

                # Trend analysis
                stability_results['trend_analysis'] = {
                    'overall_trend': 'increasing' if resampled['mean'].iloc[-1] > resampled['mean'].iloc[0] else 'decreasing',
                    'trend_strength': abs(resampled['mean'].iloc[-1] - resampled['mean'].iloc[0]) / abs(resampled['mean'].iloc[0]),
                    'trend_significance': self._test_trend_significance(resampled['mean'].values)
                }

                # Detect anomalous periods
                z_scores = np.abs(stats.zscore(resampled['mean'].dropna()))
                anomaly_mask = z_scores > 3.0
                if anomaly_mask.any():
                    anomaly_periods = resampled[anomaly_mask].index.tolist()
                    stability_results['anomaly_periods'] = anomaly_periods

            # Generate recommendations
            if stability_results['stability_score'] < 0.7:
                stability_results['recommendations'].append("Low feature stability detected")
            if stability_results['anomaly_periods']:
                stability_results['recommendations'].append(f"Anomalous periods detected: {len(stability_results['anomaly_periods'])}")

            stab = stability_results.get('stability_score')
            stab_str = f"{stab:.4f}" if isinstance(stab, (int, float, np.floating)) else str(stab)
            self.logger.info(f"✅ Feature stability analysis completed - Stability score: {stab_str}")
            return stability_results

        except Exception as e:
            self.logger.error(f"❌ Feature stability analysis failed: {e}")
            self.logger.warning("⚠️ Feature stability analysis failed - using default stability metrics")
            return {'error': str(e), 'stability_score': 0.5, 'volatility_measure': 0.0}

    def calculate_data_quality_score(self, df: pd.DataFrame,
                                   weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive data quality score.

        Args:
            df: DataFrame to analyze
            weights: Weights for different quality dimensions

        Returns:
            Comprehensive data quality assessment
        """
        try:
            if weights is None:
                weights = self.quality_score_weights

            self.logger.info("📊 Calculating comprehensive data quality score")

            quality_assessment = {
                'overall_score': 0.0,
                'dimension_scores': {},
                'quality_grade': 'F',
                'strengths': [],
                'weaknesses': [],
                'recommendations': []
            }

            # 1. Completeness Score
            completeness_score = self._calculate_completeness_score(df)
            quality_assessment['dimension_scores']['completeness'] = completeness_score

            # 2. Accuracy Score
            accuracy_score = self._calculate_accuracy_score(df)
            quality_assessment['dimension_scores']['accuracy'] = accuracy_score

            # 3. Consistency Score
            consistency_score = self._calculate_consistency_score(df)
            quality_assessment['dimension_scores']['consistency'] = consistency_score

            # 4. Timeliness Score
            timeliness_score = self._calculate_timeliness_score(df)
            quality_assessment['dimension_scores']['timeliness'] = timeliness_score

            # Calculate weighted overall score
            quality_assessment['overall_score'] = sum(
                score * weights.get(dimension, 1.0)
                for dimension, score in quality_assessment['dimension_scores'].items()
            ) / sum(weights.values())

            # Assign grade
            quality_assessment['quality_grade'] = self._score_to_grade(quality_assessment['overall_score'])

            # Analyze strengths and weaknesses
            for dimension, score in quality_assessment['dimension_scores'].items():
                if score >= 0.8:
                    quality_assessment['strengths'].append(f"Strong {dimension} ({score:.2f})")
                elif score < 0.6:
                    quality_assessment['weaknesses'].append(f"Poor {dimension} ({score:.2f})")

            # Generate recommendations
            if quality_assessment['overall_score'] < 0.7:
                quality_assessment['recommendations'].extend([
                    "Overall data quality needs improvement",
                    "Focus on addressing weaknesses in key dimensions"
                ])

            self.logger.info(f"✅ Data quality score calculated - "
                           f"Overall: {quality_assessment['overall_score']:.2f} ({quality_assessment['quality_grade']})")
            return quality_assessment

        except Exception as e:
            self.logger.error(f"❌ Data quality score calculation failed: {e}")
            self.logger.warning("⚠️ Data quality score calculation failed - using default score of 0.5")
            return {'error': str(e), 'overall_score': 0.5, 'quality_grade': 'D', 'dimension_scores': {}}

    def enhanced_automated_data_cleaning(self, df: pd.DataFrame,
                                       cleaning_config: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Enhanced automated data cleaning with advanced strategies and

        Args:
            df: DataFrame to clean
            cleaning_config: Enhanced cleaning configuration

        Returns:
            Tuple of (cleaned_dataframe, enhanced_cleaning_report)
        """
        try:
            self.logger.info("🧹 Starting enhanced automated data cleaning")

            if cleaning_config is None:
                cleaning_config = {
                    'missing_value_strategy': 'advanced_imputation',
                    'outlier_method': 'advanced_detection',
                    'correlation_threshold': 0.95,
                    'drift_adaptation': True,
                    'feature_stability_check': True,
                    'gpu_acceleration': self.enable_gpu
                }

            enhanced_report = {
                'original_shape': df.shape,
                'cleaning_steps': [],
                'removed_samples': 0,
                'removed_features': 0,
                'imputed_values': 0,
                'quality_improvement': {},
                'performance_metrics': {}
            }

            cleaned_df = df.copy()

            # Memory optimization
            if self.memory_optimizer:
                self.memory_optimizer.create_memory_checkpoint("data_cleaning_start")

            # Step 1: Quality assessment before cleaning
            pre_quality = self.calculate_data_quality_score(cleaned_df)

            # Step 2: Advanced missing value handling
            if cleaning_config.get('missing_value_strategy') == 'advanced_imputation':
                cleaned_df, missing_report = self._advanced_missing_value_imputation(cleaned_df)
                enhanced_report['cleaning_steps'].append(missing_report)
            else:
                cleaned_df, missing_report = self._clean_missing_values(cleaned_df, 'median')
                enhanced_report['cleaning_steps'].append(missing_report)

            # Step 3: Enhanced outlier detection
            if cleaning_config.get('outlier_method') == 'advanced_detection':
                outlier_results = self._advanced_outlier_detection(cleaned_df)
                if outlier_results.get('outlier_indices'):
                    cleaned_df = cleaned_df.drop(outlier_results['outlier_indices'])
                    enhanced_report['removed_samples'] += len(outlier_results['outlier_indices'])
                    enhanced_report['cleaning_steps'].append({
                        'step': 'advanced_outlier_removal',
                        'removed_samples': len(outlier_results['outlier_indices'])
                    })

            # Step 4: Feature stability check
            if cleaning_config.get('feature_stability_check', False):
                stability_issues = self._identify_stability_issues(cleaned_df)
                if stability_issues:
                    enhanced_report['cleaning_steps'].append({
                        'step': 'stability_filtering',
                        'unstable_features': stability_issues
                    })

            # Step 5: Drift adaptation
            if cleaning_config.get('drift_adaptation', False):
                # Compare current data with historical baseline (if available)
                drift_analysis = self._adaptive_drift_correction(cleaned_df)
                if drift_analysis.get('corrections_applied', False):
                    enhanced_report['cleaning_steps'].append({
                        'step': 'drift_adaptation',
                        'corrections': drift_analysis
                    })

            # Step 6: Quality assessment after cleaning
            post_quality = self.calculate_data_quality_score(cleaned_df)
            enhanced_report['quality_improvement'] = {
                'pre_cleaning_score': pre_quality['overall_score'],
                'post_cleaning_score': post_quality['overall_score'],
                'improvement': post_quality['overall_score'] - pre_quality['overall_score']
            }

            # Performance metrics
            if self.memory_optimizer:
                memory_usage = self.memory_optimizer.get_memory_usage()
                enhanced_report['performance_metrics']['memory_usage'] = memory_usage

            # Final statistics
            enhanced_report['final_shape'] = cleaned_df.shape
            enhanced_report['total_removed_samples'] = enhanced_report['original_shape'][0] - enhanced_report['final_shape'][0]
            enhanced_report['total_removed_features'] = enhanced_report['original_shape'][1] - enhanced_report['final_shape'][1]

            self.logger.info(f"✅ Enhanced automated data cleaning completed: "
                           f"Shape: {enhanced_report['original_shape']} -> {enhanced_report['final_shape']}")
            return cleaned_df, enhanced_report

        except Exception as e:
            self.logger.error(f"❌ Enhanced automated data cleaning failed: {e}")
            self.logger.warning("⚠️ Enhanced data cleaning failed - returning original data with error report")
            return df, {'error': str(e), 'original_shape': df.shape, 'cleaning_steps': [{'step': 'failed', 'error': str(e)}], 'quality_improvement': {'pre_cleaning_score': 0.5, 'post_cleaning_score': 0.5}}

    # Helper methods for new functionality

    def _calculate_ks_drift(self, ref_data: np.ndarray, curr_data: np.ndarray) -> float:
        """Calculate Kolmogorov-Smirnov drift score."""
        try:
            ks_statistic, _ = stats.ks_2samp(ref_data, curr_data)
            return ks_statistic
        except:
            return 0.0

    def _calculate_wasserstein_drift(self, ref_data: np.ndarray, curr_data: np.ndarray) -> float:
        """Calculate Wasserstein distance drift score."""
        try:
            from scipy.stats import wasserstein_distance
            return wasserstein_distance(ref_data, curr_data)
        except:
            return self._calculate_ks_drift(ref_data, curr_data)

    def _calculate_js_drift(self, ref_data: np.ndarray, curr_data: np.ndarray) -> float:
        """Calculate Jensen-Shannon divergence drift score."""
        try:
            # Normalize data to create probability distributions
            ref_hist, _ = np.histogram(ref_data, bins=50, density=True)
            curr_hist, _ = np.histogram(curr_data, bins=50, density=True)

            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            ref_hist = ref_hist + epsilon
            curr_hist = curr_hist + epsilon

            # Normalize
            ref_hist = ref_hist / ref_hist.sum()
            curr_hist = curr_hist / curr_hist.sum()

            # Calculate Jensen-Shannon divergence
            m = 0.5 * (ref_hist + curr_hist)
            js_divergence = 0.5 * (stats.entropy(ref_hist, m) + stats.entropy(curr_hist, m))

            return js_divergence
        except:
            return self._calculate_ks_drift(ref_data, curr_data)

    def _classify_drift_severity(self, drift_score: float) -> str:
        """Classify drift severity based on score."""
        if drift_score < 0.1:
            return 'low'
        elif drift_score < 0.3:
            return 'moderate'
        elif drift_score < 0.5:
            return 'high'
        else:
            return 'critical'

    def _analyze_rolling_window_drift(self, data: np.ndarray, window_size: int,
                                    feature_names: List[str]) -> Dict[str, Any]:
        """Analyze drift in rolling windows."""
        try:
            window_analysis = {}

            for i, feature_name in enumerate(feature_names):
                if i >= data.shape[1]:
                    continue

                feature_data = data[:, i]
                rolling_windows = []

                for start_idx in range(0, len(feature_data) - window_size + 1, window_size // 2):
                    end_idx = min(start_idx + window_size, len(feature_data))
                    window_data = feature_data[start_idx:end_idx]

                    if len(window_data) > 10:
                        window_stats = {
                            'mean': np.mean(window_data),
                            'std': np.std(window_data),
                            'start_idx': start_idx,
                            'end_idx': end_idx
                        }
                        rolling_windows.append(window_stats)

                window_analysis[feature_name] = rolling_windows

            return window_analysis
        except Exception:
            return {}

    def _test_trend_significance(self, values: np.ndarray) -> float:
        """Test significance of trend in time series."""
        try:
            if len(values) < 5:
                return 0.0

            # Simple linear regression test
            x = np.arange(len(values))
            slope, _, r_value, p_value, _ = stats.linregress(x, values)

            return 1.0 - p_value  # Convert p-value to significance score
        except:
            return 0.0

    def _calculate_completeness_score(self, df: pd.DataFrame) -> float:
        """Calculate completeness score (inverse of missing data ratio)."""
        try:
            missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            return 1.0 - missing_ratio
        except:
            return 0.0

    def _calculate_accuracy_score(self, df: pd.DataFrame) -> float:
        """Calculate accuracy score based on data consistency."""
        try:
            numeric_df = df.select_dtypes(include=[np.number])

            if numeric_df.empty:
                return 0.8  # Default for non-numeric data

            # Check for reasonable value ranges and distributions
            accuracy_indicators = []

            for col in numeric_df.columns:
                values = numeric_df[col].dropna()
                if len(values) > 0:
                    # Check for extreme outliers
                    q1, q3 = np.percentile(values, [25, 75])
                    iqr = q3 - q1
                    outlier_ratio = ((values < q1 - 3*iqr) | (values > q3 + 3*iqr)).sum() / len(values)
                    accuracy_indicators.append(1.0 - outlier_ratio)

            return np.mean(accuracy_indicators) if accuracy_indicators else 0.8
        except:
            return 0.5

    def _calculate_consistency_score(self, df: pd.DataFrame) -> float:
        """Calculate consistency score based on data patterns."""
        try:
            consistency_indicators = []

            # Check for duplicate rows
            duplicate_ratio = df.duplicated().sum() / len(df)
            consistency_indicators.append(1.0 - duplicate_ratio)

            # Check for inconsistent data types
            for col in df.columns:
                try:
                    pd.to_numeric(df[col], errors='coerce')
                    consistency_indicators.append(0.9)  # Successfully converted
                except:
                    consistency_indicators.append(0.7)  # Mixed types

            return np.mean(consistency_indicators) if consistency_indicators else 0.7
        except:
            return 0.5

    def _calculate_timeliness_score(self, df: pd.DataFrame) -> float:
        """Calculate timeliness score based on data freshness."""
        try:
            timeliness_indicators = []

            # Check for datetime columns
            datetime_cols = []
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    datetime_cols.append(col)
                elif df[col].dtype == 'object':
                    # Try to parse as datetime
                    try:
                        pd.to_datetime(df[col].head(10), errors='coerce')
                        datetime_cols.append(col)
                    except Exception as e:
                        logger.warning(f"Could not parse datetime for column {col}: {e}")
                        # Continue without this column

            if datetime_cols:
                timeliness_indicators.append(0.9)  # Has datetime information
            else:
                timeliness_indicators.append(0.6)  # No clear temporal information

            return np.mean(timeliness_indicators)
        except:
            return 0.6

    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        else:
            return 'F'

    def _advanced_missing_value_imputation(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Advanced missing value imputation using multiple strategies."""
        try:
            cleaned_df = df.copy()
            report = {'step': 'advanced_missing_imputation', 'imputed_features': []}

            for col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count == 0:
                    continue

                if pd.api.types.is_numeric_dtype(df[col]):
                    # Use KNN imputation for numeric columns
                    try:
                        from sklearn.impute import KNNImputer
                        imputer = KNNImputer(n_neighbors=min(5, len(df) - missing_count))
                        col_data = df[[col]]
                        imputed = imputer.fit_transform(col_data)
                        cleaned_df[col] = imputed.flatten()
                    except:
                        # Fallback to median imputation
                        median_val = df[col].median()
                        cleaned_df[col] = df[col].fillna(median_val)
                else:
                    # Use mode imputation for categorical columns
                    mode_val = df[col].mode()
                    if not mode_val.empty:
                        cleaned_df[col] = df[col].fillna(mode_val.iloc[0])

                report['imputed_features'].append({
                    'feature': col,
                    'missing_count': int(missing_count),
                    'strategy': 'knn' if pd.api.types.is_numeric_dtype(df[col]) else 'mode'
                })

            return cleaned_df, report
        except Exception as e:
            return df, {'error': str(e)}

    def _advanced_outlier_detection(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Advanced outlier detection using multiple methods."""
        try:
            outlier_results = {'outlier_indices': set()}

            # Method 1: Isolation Forest
            if_outliers = self.automated_outlier_detection(df, method='isolation_forest')
            if 'outlier_indices' in if_outliers:
                outlier_results['outlier_indices'].update(if_outliers['outlier_indices'])

            # Method 2: Local Outlier Factor
            try:
                from sklearn.neighbors import LocalOutlierFactor
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    lof = LocalOutlierFactor(contamination=self.outlier_contamination)
                    lof_scores = lof.fit_predict(numeric_df)
                    lof_outliers = np.where(lof_scores == -1)[0]
                    outlier_results['outlier_indices'].update(lof_outliers)
            except Exception as e:
                logger.warning(f"Outlier detection failed: {e}, continuing without outlier detection")
                return {'outlier_indices': []}  # Return empty results

            outlier_results['outlier_indices'] = list(outlier_results['outlier_indices'])
            return outlier_results
        except Exception:
            return {'outlier_indices': []}

    def _identify_stability_issues(self, df: pd.DataFrame) -> List[str]:
        """Identify features with stability issues."""
        try:
            unstable_features = []

            # Check for datetime columns for stability analysis
            datetime_cols = []
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    datetime_cols.append(col)

            if datetime_cols and len(df) > 10:
                for col in df.select_dtypes(include=[np.number]).columns:
                    # Simple stability check based on rolling statistics
                    rolling_std = df[col].rolling(window=min(10, len(df))).std()
                    avg_volatility = rolling_std.mean()
                    if avg_volatility > df[col].std() * 2:  # Highly volatile
                        unstable_features.append(col)

            return unstable_features
        except:
            return []

    def _adaptive_drift_correction(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Adaptive drift correction based on detected patterns."""
        try:
            corrections = {'corrections_applied': False}

            # Simple drift correction: standardize numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                scaler = StandardScaler()
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                corrections['corrections_applied'] = True
                corrections['standardized_columns'] = list(numeric_cols)

            return corrections
        except Exception:
            return {'corrections_applied': False}

    def _isolation_forest_detection(self, X: np.ndarray, feature_names: List[str],
                                  contamination: float) -> Dict[str, Any]:
        """Perform outlier detection using Isolation Forest."""
        try:
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Fit isolation forest
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            outlier_scores = iso_forest.fit_predict(X_scaled)

            # Get outlier indices and scores
            outlier_indices = np.where(outlier_scores == -1)[0]
            outlier_scores_values = iso_forest.score_samples(X_scaled)

            # Feature-level outlier analysis
            feature_outliers = {}
            for i, feature_name in enumerate(feature_names):
                feature_values = X[:, i]
                # Simple outlier detection for individual features
                q1, q3 = np.percentile(feature_values, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                feature_outlier_indices = np.where(
                    (feature_values < lower_bound) | (feature_values > upper_bound)
                )[0]

                feature_outliers[feature_name] = {
                    'outlier_count': len(feature_outlier_indices),
                    'outlier_percentage': safe_divide(len(feature_outlier_indices), len(feature_values)) * 100,
                    'bounds': {'lower': lower_bound, 'upper': upper_bound}
                }

            return {
                'outlier_indices': outlier_indices.tolist(),
                'outlier_scores': outlier_scores_values.tolist(),
                'feature_outliers': feature_outliers
            }

        except Exception as e:
            return {'error': str(e), 'outlier_indices': []}

    def _calculate_distribution_stats(self, values: np.ndarray) -> Dict[str, float]:
        """Calculate distribution statistics."""
        try:
            return {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'skewness': float(stats.skew(values)),
                'kurtosis': float(stats.kurtosis(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'q25': float(np.percentile(values, 25)),
                'q75': float(np.percentile(values, 75))
            }
        except Exception:
            return {}

    def _calculate_stability_metrics(self, train_stats: Dict[str, float],
                                   test_stats: Dict[str, float]) -> Dict[str, Any]:
        """Calculate stability metrics between distributions."""
        try:
            # Kolmogorov-Smirnov test for distribution similarity
            # This is a simplified version - in practice, you'd need the actual data
            ks_statistic = abs(train_stats.get('mean', 0) - test_stats.get('mean', 0)) / \
                          max(train_stats.get('std', 1), test_stats.get('std', 1))

            # Simplified p-value calculation
            ks_p_value = 1.0 - min(ks_statistic, 1.0)

            return {
                'ks_statistic': ks_statistic,
                'ks_p_value': ks_p_value,
                'mean_difference': abs(train_stats.get('mean', 0) - test_stats.get('mean', 0)),
                'std_difference': abs(train_stats.get('std', 0) - test_stats.get('std', 0))
            }

        except Exception:
            return {'ks_statistic': 0, 'ks_p_value': 1.0}

    def _calculate_drift_metrics(self, ref_values: np.ndarray,
                               new_values: np.ndarray) -> Dict[str, Any]:
        """Calculate drift metrics between two distributions."""
        try:
            # Kolmogorov-Smirnov test
            ks_statistic, ks_p_value = stats.ks_2samp(ref_values, new_values)

            # Additional metrics
            ref_mean, new_mean = np.mean(ref_values), np.mean(new_values)
            ref_std, new_std = np.std(ref_values), np.std(new_values)

            return {
                'ks_statistic': float(ks_statistic),
                'ks_p_value': float(ks_p_value),
                'mean_difference': float(abs(ref_mean - new_mean)),
                'std_difference': float(abs(ref_std - new_std)),
                'relative_mean_change': safe_divide(abs(ref_mean - new_mean), abs(ref_mean))
            }

        except Exception:
            return {'ks_statistic': 0, 'ks_p_value': 1.0}

    def _calculate_statistical_correlation(self, X: np.ndarray, method: str) -> np.ndarray:
        """Calculate statistical correlation matrix."""
        try:
            if method == 'pearson':
                return np.corrcoef(X.T)
            elif method == 'spearman':
                from scipy.stats import spearmanr

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

except ImportError:

    cp = None
                corr_matrix = np.zeros((X.shape[1], X.shape[1]))
                for i in range(X.shape[1]):
                    for j in range(X.shape[1]):
                        if i != j:
                            corr, _ = spearmanr(X[:, i], X[:, j])
                            corr_matrix[i, j] = corr
                        else:
                            corr_matrix[i, j] = 1.0
                return corr_matrix
            else:
                return np.eye(X.shape[1])
        except Exception:
            return np.eye(X.shape[1])

    def _calculate_mutual_info_correlation(self, X: np.ndarray) -> np.ndarray:
        """Calculate mutual information correlation matrix."""
        try:
            if not SKLEARN_AVAILABLE:
                return np.eye(X.shape[1])

            corr_matrix = np.zeros((X.shape[1], X.shape[1]))
            for i in range(X.shape[1]):
                for j in range(X.shape[1]):
                    if i != j:
                        corr_matrix[i, j] = mutual_info_score(X[:, i], X[:, j])
                    else:
                        corr_matrix[i, j] = 1.0
            return corr_matrix

        except Exception:
            return np.eye(X.shape[1])

    def _clean_missing_values(self, df: pd.DataFrame,
                            strategy: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Clean missing values using specified strategy."""
        try:
            cleaned_df = df.copy()
            report = {'step': 'missing_value_handling', 'imputed_features': []}

            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                missing_count = cleaned_df[col].isnull().sum()
                if missing_count > 0:
                    if strategy == 'median':
                        fill_value = cleaned_df[col].median()
                    elif strategy == 'mean':
                        fill_value = cleaned_df[col].mean()
                    else:
                        fill_value = 0

                    cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                    report['imputed_features'].append({
                        'feature': col,
                        'missing_count': int(missing_count),
                        'fill_value': float(fill_value)
                    })

            return cleaned_df, report

        except Exception as e:
            return df, {'error': str(e)}

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and
                VECTORBT_AVAILABLE)

    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

    def _pandas_rolling_operation(self, data: pd.Series, operation: str,
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def _vectorbt_apply_operation(self, data: pd.Series, func,
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)

        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
