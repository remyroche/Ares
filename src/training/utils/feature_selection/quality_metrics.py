from src.utils.tprint import tprint

"""
Quality Metrics Component

This module provides comprehensive quality assessment metrics for feature selection,
including redundancy, relevance, stability, interpretability, and performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime
import time
from collections import defaultdict

# Enhanced dependency management
try:
    from src.utils.logger import get_logger
    _LOGGER = get_logger("FeatureSelection.QualityMetrics")
    tprint("✅ Custom logger available for FeatureSelection.QualityMetrics")
except Exception as e:
    tprint(f"⚠️ Custom logger not available: {e}. Using standard logging.")
    _LOGGER = logging.getLogger("FeatureSelection.QualityMetrics")
    _LOGGER.setLevel(logging.INFO)

logger = _LOGGER

try:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available - limited quality metrics functionality")

class QualityMetricsCalculator:
    """Comprehensive quality metrics calculator for feature selection."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize quality metrics calculator."""
        self.config = config or {}
        self.logger = logger.getChild('QualityMetricsCalculator')

        # Quality metric weights
        self.weights = {
            'redundancy': self.config.get('redundancy_weight', 0.2),
            'relevance': self.config.get('relevance_weight', 0.3),
            'stability': self.config.get('stability_weight', 0.2),
            'interpretability': self.config.get('interpretability_weight', 0.1),
            'performance': self.config.get('performance_weight', 0.2)
        }

        # Metric thresholds
        self.correlation_threshold = self.config.get('correlation_threshold', 0.8)
        self.performance_threshold = self.config.get('performance_threshold', 0.7)

        _LOGGER.info("📊 QualityMetricsCalculator initialized")
        _LOGGER.info(f"⚙️ Metric weights: {self.weights}")
        _LOGGER.info(f"⚙️ Correlation threshold: {self.correlation_threshold}")

    def calculate_comprehensive_quality_metrics(self, X: np.ndarray, y: np.ndarray,
                                               selected_features: List[str],
                                               feature_names: List[str],
                                               pipeline_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics for feature selection."""
        start_time = time.time()
        _LOGGER.info(f"📊 Starting comprehensive quality metrics calculation...")
        _LOGGER.info(f"📊 Parameters - Selected features: {len(selected_features)}, Data shape: {X.shape}")

        try:
            # Get selected feature indices
            selected_indices = [feature_names.index(feat) for feat in selected_features if feat in feature_names]
            X_selected = X[:, selected_indices]

            # Calculate individual metric categories
            redundancy_metrics = self.calculate_redundancy_metrics(X_selected, selected_features)
            relevance_metrics = self.calculate_relevance_metrics(X_selected, y, selected_features)
            stability_metrics = self.calculate_stability_metrics(pipeline_results) if pipeline_results else {}
            interpretability_metrics = self.calculate_interpretability_metrics(selected_features)
            performance_metrics = self.calculate_performance_metrics(X_selected, y, selected_features)

            # Calculate overall quality score
            overall_score = self.calculate_overall_quality_score({
                'redundancy': redundancy_metrics,
                'relevance': relevance_metrics,
                'stability': stability_metrics,
                'interpretability': interpretability_metrics,
                'performance': performance_metrics
            })

            execution_time = time.time() - start_time

            result = {
                'overall_quality_score': overall_score,
                'redundancy_metrics': redundancy_metrics,
                'relevance_metrics': relevance_metrics,
                'stability_metrics': stability_metrics,
                'interpretability_metrics': interpretability_metrics,
                'performance_metrics': performance_metrics,
                'metric_weights': self.weights,
                'execution_time': execution_time,
                'success': True
            }

            _LOGGER.info(f"✅ Quality metrics calculation completed in {execution_time:.3f}s")
            _LOGGER.info(f"📊 Overall quality score: {overall_score:.3f}")

            return result

        except Exception as e:
            _LOGGER.error(f"❌ Quality metrics calculation failed: {e}")
            return {
                'overall_quality_score': 0.0,
                'redundancy_metrics': {},
                'relevance_metrics': {},
                'stability_metrics': {},
                'interpretability_metrics': {},
                'performance_metrics': {},
                'error': str(e),
                'success': False
            }

    def calculate_redundancy_metrics(self, X_selected: np.ndarray, selected_features: List[str]) -> Dict[str, float]:
        """Calculate redundancy metrics for selected features."""
        _LOGGER.debug("📊 Calculating redundancy metrics...")

        try:
            n_features = X_selected.shape[1]
            if n_features < 2:
                return {
                    'mean_correlation': 0.0,
                    'max_correlation': 0.0,
                    'high_correlation_pairs': 0,
                    'redundancy_score': 1.0  # Perfect score for single feature
                }

            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(X_selected.T)

            # Remove diagonal (self-correlation)
            mask = np.ones_like(correlation_matrix, dtype=bool)
            np.fill_diagonal(mask, False)
            correlations = correlation_matrix[mask]

            # Calculate metrics
            mean_correlation = np.mean(np.abs(correlations))
            max_correlation = np.max(np.abs(correlations))
            high_correlation_pairs = np.sum(np.abs(correlations) > self.correlation_threshold)

            # Redundancy score (lower is better, so invert)
            redundancy_score = 1.0 - mean_correlation

            metrics = {
                'mean_correlation': mean_correlation,
                'max_correlation': max_correlation,
                'high_correlation_pairs': high_correlation_pairs,
                'redundancy_score': max(0.0, redundancy_score)
            }

            _LOGGER.debug(f"📊 Redundancy metrics - Mean correlation: {mean_correlation:.3f}, "
                         f"Max correlation: {max_correlation:.3f}, High corr pairs: {high_correlation_pairs}")

            return metrics

        except Exception as e:
            _LOGGER.warning(f"⚠️ Redundancy metrics calculation failed: {e}")
            return {'error': str(e)}

    def calculate_relevance_metrics(self, X_selected: np.ndarray, y: np.ndarray,
                                   selected_features: List[str]) -> Dict[str, float]:
        """Calculate relevance metrics for selected features."""
        _LOGGER.debug("📊 Calculating relevance metrics...")

        try:
            n_features = X_selected.shape[1]
            if n_features == 0:
                return {'relevance_score': 0.0}

            # Calculate individual feature relevances
            feature_relevances = []

            for i in range(n_features):
                feature_data = X_selected[:, i]

                # Calculate correlation with target
                corr = np.corrcoef(feature_data, y)[0, 1]
                if not np.isnan(corr):
                    feature_relevances.append(abs(corr))
                else:
                    feature_relevances.append(0.0)

            # Calculate mutual information if available
            mutual_information_scores = []
            if SKLEARN_AVAILABLE:
                try:
                    # Determine if classification or regression
                    if len(np.unique(y)) <= 10:  # Classification
                        mi_scores = mutual_info_classif(X_selected, y)
                    else:  # Regression
                        mi_scores = mutual_info_regression(X_selected, y)

                    mutual_information_scores = mi_scores.tolist()
                except Exception as e:
                    _LOGGER.debug(f"⚠️ Mutual information calculation failed: {e}")
                    mutual_information_scores = [0.0] * n_features

            # Calculate metrics
            mean_correlation = np.mean(feature_relevances)
            mean_mutual_information = np.mean(mutual_information_scores) if mutual_information_scores else 0.0

            # Relevance score (higher is better)
            relevance_score = (mean_correlation + mean_mutual_information) / 2.0

            metrics = {
                'mean_correlation': mean_correlation,
                'mean_mutual_information': mean_mutual_information,
                'feature_correlations': feature_relevances,
                'feature_mutual_information': mutual_information_scores,
                'relevance_score': relevance_score
            }

            _LOGGER.debug(f"📊 Relevance metrics - Mean correlation: {mean_correlation:.3f}, "
                         f"Mean MI: {mean_mutual_information:.3f}, Relevance score: {relevance_score:.3f}")

            return metrics

        except Exception as e:
            _LOGGER.warning(f"⚠️ Relevance metrics calculation failed: {e}")
            return {'error': str(e)}

    def calculate_stability_metrics(self, pipeline_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate stability metrics from pipeline results."""
        _LOGGER.debug("📊 Calculating stability metrics...")

        try:
            if not pipeline_results:
                return {'stability_score': 0.0}

            # Extract stability information from pipeline results
            stability_scores = []

            # Check for bootstrap stability results
            if 'bootstrap_stability' in pipeline_results:
                bootstrap_results = pipeline_results['bootstrap_stability']
                if bootstrap_results.get('success', False):
                    stability_scores.extend(bootstrap_results.get('stability_scores', {}).values())

            # Check for temporal stability results
            if 'temporal_stability' in pipeline_results:
                temporal_results = pipeline_results['temporal_stability']
                if temporal_results.get('success', False):
                    temporal_metrics = temporal_results.get('temporal_stability_metrics', {})
                    if 'mean_temporal_consistency' in temporal_metrics:
                        stability_scores.append(temporal_metrics['mean_temporal_consistency'])

            # Check for cross-dataset stability results
            if 'cross_dataset_stability' in pipeline_results:
                cross_dataset_results = pipeline_results['cross_dataset_stability']
                if cross_dataset_results.get('success', False):
                    cross_dataset_metrics = cross_dataset_results.get('cross_dataset_metrics', {})
                    if 'mean_cross_dataset_consistency' in cross_dataset_metrics:
                        stability_scores.append(cross_dataset_metrics['mean_cross_dataset_consistency'])

            # Calculate overall stability score
            if stability_scores:
                stability_score = np.mean(stability_scores)
            else:
                stability_score = 0.0

            metrics = {
                'stability_score': stability_score,
                'individual_stability_scores': stability_scores,
                'n_stability_analyses': len(stability_scores)
            }

            _LOGGER.debug(f"📊 Stability metrics - Stability score: {stability_score:.3f}, "
                         f"Analyses: {len(stability_scores)}")

            return metrics

        except Exception as e:
            _LOGGER.warning(f"⚠️ Stability metrics calculation failed: {e}")
            return {'error': str(e)}

    def calculate_interpretability_metrics(self, selected_features: List[str]) -> Dict[str, float]:
        """Calculate interpretability metrics for selected features."""
        _LOGGER.debug("📊 Calculating interpretability metrics...")

        try:
            n_features = len(selected_features)

            # Feature name complexity (simple heuristic)
            name_complexities = []
            for feature in selected_features:
                # Simple complexity based on name length and special characters
                complexity = len(feature) + feature.count('_') + feature.count('-') + feature.count('.')
                name_complexities.append(complexity)

            mean_name_complexity = np.mean(name_complexities)

            # Feature count (fewer features are more interpretable)
            feature_count_score = max(0.0, 1.0 - (n_features / 50.0))  # Penalty for >50 features

            # Interpretability score (higher is better)
            interpretability_score = (1.0 / (1.0 + mean_name_complexity / 10.0)) * feature_count_score

            metrics = {
                'n_features': n_features,
                'mean_name_complexity': mean_name_complexity,
                'feature_count_score': feature_count_score,
                'interpretability_score': interpretability_score
            }

            _LOGGER.debug(f"📊 Interpretability metrics - N features: {n_features}, "
                         f"Mean complexity: {mean_name_complexity:.3f}, Score: {interpretability_score:.3f}")

            return metrics

        except Exception as e:
            _LOGGER.warning(f"⚠️ Interpretability metrics calculation failed: {e}")
            return {'error': str(e)}

    def calculate_performance_metrics(self, X_selected: np.ndarray, y: np.ndarray,
                                     selected_features: List[str]) -> Dict[str, float]:
        """Calculate performance metrics for selected features."""
        _LOGGER.debug("📊 Calculating performance metrics...")

        try:
            if not SKLEARN_AVAILABLE:
                return {'performance_score': 0.0, 'error': 'Scikit-learn not available'}

            n_features = X_selected.shape[1]
            if n_features == 0:
                return {'performance_score': 0.0}

            # Determine if classification or regression
            is_classification = len(np.unique(y)) <= 10

            # Use appropriate model
            if is_classification:
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                scoring = 'accuracy'
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                cv = KFold(n_splits=3, shuffle=True, random_state=42)
                scoring = 'r2'

            # Preprocess data to handle infinity and NaN values before cross-validation
            try:
                from .selection_methods import preprocess_features_for_ml
                X_processed = preprocess_features_for_ml(X_selected, "quality_metrics_cross_validation")

                # Additional NaN handling with detailed analysis
                nan_mask = np.isnan(X_processed)
                if np.any(nan_mask):
                    nan_count = np.sum(nan_mask)

                    # Import the detailed NaN analysis function
                    from src.utils.common_utilities import analyze_nan_values_detailed, format_nan_analysis_report

                    # Perform detailed NaN analysis
                    nan_analysis = analyze_nan_values_detailed(X_processed)
                    detailed_report = format_nan_analysis_report(nan_analysis, "[QUALITY_METRICS] ")

                    _LOGGER.warning(f"⚠️ Found {nan_count} NaN values in features after preprocessing, filling with column means")
                    _LOGGER.warning(detailed_report)

                    # Fill NaN values with column means
                    for col in range(X_processed.shape[1]):
                        col_data = X_processed[:, col]
                        finite_mask = np.isfinite(col_data)
                        if np.any(finite_mask):
                            col_mean = np.mean(col_data[finite_mask])
                            X_processed[np.isnan(col_data), col] = col_mean
                        else:
                            X_processed[np.isnan(col_data), col] = 0.0

                # Validate target variable
                if np.any(np.isnan(y)):
                    nan_target_count = np.sum(np.isnan(y))
                    _LOGGER.warning(f"⚠️ Found {nan_target_count} NaN values in target variable, cannot perform cross-validation")
                    return {
                        'performance_score': 0.0,
                        'error': f"Target variable contains {nan_target_count} NaN values",
                        'target_has_nan': True
                    }

                if np.all(y == y[0]):
                    _LOGGER.warning(f"⚠️ All target values are identical ({y[0]}), cannot perform meaningful cross-validation")
                    return {
                        'performance_score': 0.0,
                        'error': f"All target values are identical ({y[0]})",
                        'constant_target': True
                    }

            except ImportError:
                _LOGGER.warning("⚠️ Could not import preprocessing function, using basic validation")
                X_processed = X_selected

                # Basic NaN handling
                if np.any(np.isnan(X_processed)) or np.any(np.isnan(y)):
                    _LOGGER.warning("⚠️ NaN values found in data, cannot perform cross-validation")
                    return {
                        'performance_score': 0.0,
                        'error': "NaN values found in features or target",
                        'has_nan': True
                    }

            # Perform cross-validation
            try:
                cv_scores = cross_val_score(model, X_processed, y, cv=cv, scoring=scoring)
                mean_cv_score = np.mean(cv_scores)
                std_cv_score = np.std(cv_scores)

                # Performance score (higher is better)
                performance_score = max(0.0, mean_cv_score)

                metrics = {
                    'mean_cv_score': mean_cv_score,
                    'std_cv_score': std_cv_score,
                    'cv_scores': cv_scores.tolist(),
                    'performance_score': performance_score,
                    'scoring_method': scoring,
                    'is_classification': is_classification
                }

            except Exception as e:
                _LOGGER.warning(f"⚠️ Cross-validation failed: {e}")
                # Enhanced error diagnostics
                error_diagnostics = {
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'data_shape': X_processed.shape,
                    'target_shape': y.shape,
                    'target_unique_values': len(np.unique(y)),
                    'target_dtype': str(y.dtype),
                    'features_have_nan': np.any(np.isnan(X_processed)),
                    'features_have_inf': np.any(np.isinf(X_processed)),
                    'target_has_nan': np.any(np.isnan(y)),
                    'target_has_inf': np.any(np.isinf(y)),
                    'model_type': type(model).__name__,
                    'cv_type': type(cv).__name__,
                    'scoring_metric': scoring
                }

                # Try to provide more specific guidance
                if "Input contains NaN" in str(e):
                    error_diagnostics['guidance'] = "Data contains NaN values that weren't properly handled"
                elif "Input contains infinity" in str(e):
                    error_diagnostics['guidance'] = "Data contains infinity values that weren't properly handled"
                elif "classification" in str(e).lower() and is_classification:
                    error_diagnostics['guidance'] = "Classification task failed - check target variable format"
                elif "regression" in str(e).lower() and not is_classification:
                    error_diagnostics['guidance'] = "Regression task failed - check target variable format"
                else:
                    error_diagnostics['guidance'] = "General cross-validation failure - check data quality"

                metrics = {
                    'performance_score': 0.0,
                    'error': str(e),
                    'error_diagnostics': error_diagnostics,
                    'scoring_method': scoring,
                    'is_classification': is_classification
                }

            _LOGGER.debug(f"📊 Performance metrics - CV score: {metrics.get('mean_cv_score', 0):.3f}, "
                         f"Performance score: {metrics.get('performance_score', 0):.3f}")

            return metrics

        except Exception as e:
            _LOGGER.warning(f"⚠️ Performance metrics calculation failed: {e}")
            return {'error': str(e)}

    def calculate_overall_quality_score(self, quality_metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score from individual metrics."""
        try:
            overall_score = 0.0
            total_weight = 0.0

            # Redundancy score (lower redundancy is better)
            if 'redundancy' in quality_metrics and 'redundancy_score' in quality_metrics['redundancy']:
                redundancy_score = quality_metrics['redundancy']['redundancy_score']
                overall_score += self.weights['redundancy'] * redundancy_score
                total_weight += self.weights['redundancy']

            # Relevance score (higher relevance is better)
            if 'relevance' in quality_metrics and 'relevance_score' in quality_metrics['relevance']:
                relevance_score = quality_metrics['relevance']['relevance_score']
                overall_score += self.weights['relevance'] * relevance_score
                total_weight += self.weights['relevance']

            # Stability score (higher stability is better)
            if 'stability' in quality_metrics and 'stability_score' in quality_metrics['stability']:
                stability_score = quality_metrics['stability']['stability_score']
                overall_score += self.weights['stability'] * stability_score
                total_weight += self.weights['stability']

            # Interpretability score (higher interpretability is better)
            if 'interpretability' in quality_metrics and 'interpretability_score' in quality_metrics['interpretability']:
                interpretability_score = quality_metrics['interpretability']['interpretability_score']
                overall_score += self.weights['interpretability'] * interpretability_score
                total_weight += self.weights['interpretability']

            # Performance score (higher performance is better)
            if 'performance' in quality_metrics and 'performance_score' in quality_metrics['performance']:
                performance_score = quality_metrics['performance']['performance_score']
                overall_score += self.weights['performance'] * performance_score
                total_weight += self.weights['performance']

            # Normalize by total weight
            if total_weight > 0:
                overall_score = overall_score / total_weight

            return overall_score

        except Exception as e:
            _LOGGER.warning(f"⚠️ Overall quality score calculation failed: {e}")
            return 0.0

    def generate_quality_report(self, quality_metrics: Dict[str, Any]) -> str:
        """Generate comprehensive quality report."""
        try:
            overall_score = quality_metrics.get('overall_quality_score', 0.0)
            weights = quality_metrics.get('metric_weights', {})

            report = f"""
=== Feature Selection Quality Report ===
Generated: {datetime.now().isoformat()}

=== Overall Quality Score ===
Overall Score: {overall_score:.3f} / 1.0

=== Metric Weights ===
Redundancy: {weights.get('redundancy', 0):.1%}
Relevance: {weights.get('relevance', 0):.1%}
Stability: {weights.get('stability', 0):.1%}
Interpretability: {weights.get('interpretability', 0):.1%}
Performance: {weights.get('performance', 0):.1%}

=== Individual Metrics ===
"""

            # Add redundancy metrics
            redundancy = quality_metrics.get('redundancy_metrics', {})
            if redundancy and 'error' not in redundancy:
                report += f"""
Redundancy Metrics:
  Redundancy Score: {redundancy.get('redundancy_score', 0):.3f}
  Mean Correlation: {redundancy.get('mean_correlation', 0):.3f}
  Max Correlation: {redundancy.get('max_correlation', 0):.3f}
  High Correlation Pairs: {redundancy.get('high_correlation_pairs', 0)}
"""

            # Add relevance metrics
            relevance = quality_metrics.get('relevance_metrics', {})
            if relevance and 'error' not in relevance:
                report += f"""
Relevance Metrics:
  Relevance Score: {relevance.get('relevance_score', 0):.3f}
  Mean Correlation: {relevance.get('mean_correlation', 0):.3f}
  Mean Mutual Information: {relevance.get('mean_mutual_information', 0):.3f}
"""

            # Add stability metrics
            stability = quality_metrics.get('stability_metrics', {})
            if stability and 'error' not in stability:
                report += f"""
Stability Metrics:
  Stability Score: {stability.get('stability_score', 0):.3f}
  Stability Analyses: {stability.get('n_stability_analyses', 0)}
"""

            # Add interpretability metrics
            interpretability = quality_metrics.get('interpretability_metrics', {})
            if interpretability and 'error' not in interpretability:
                report += f"""
Interpretability Metrics:
  Interpretability Score: {interpretability.get('interpretability_score', 0):.3f}
  Number of Features: {interpretability.get('n_features', 0)}
  Mean Name Complexity: {interpretability.get('mean_name_complexity', 0):.3f}
"""

            # Add performance metrics
            performance = quality_metrics.get('performance_metrics', {})
            if performance and 'error' not in performance:
                report += f"""
Performance Metrics:
  Performance Score: {performance.get('performance_score', 0):.3f}
  Mean CV Score: {performance.get('mean_cv_score', 0):.3f}
  CV Score Std: {performance.get('std_cv_score', 0):.3f}
  Scoring Method: {performance.get('scoring_method', 'unknown')}
"""

            return report

        except Exception as e:
            _LOGGER.error(f"❌ Failed to generate quality report: {e}")
            return f"Error generating quality report: {e}"
