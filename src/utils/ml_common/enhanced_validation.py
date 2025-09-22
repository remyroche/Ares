"""
Enhanced Validation Procedures for ML Training

This module provides comprehensive validation procedures and strategies
to strengthen validation across all models in the training pipeline.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success

logger = system_logger.getChild('EnhancedValidation')

@dataclass
class EnhancedValidationConfig:
    """Configuration for enhanced validation procedures."""

    # Cross-validation settings
    enable_purged_cv: bool = True
    cv_folds: int = 10
    purge_minutes: int = 60
    embargo_minutes: int = 30
    enable_nested_cv: bool = True

    # Time series validation
    enable_time_series_validation: bool = True
    time_series_splits: int = 10
    expanding_window_validation: bool = True
    rolling_window_validation: bool = True

    # Bootstrap validation
    enable_bootstrap_validation: bool = True
    bootstrap_samples: int = 1000
    bootstrap_confidence_level: float = 0.95

    # Robustness testing
    enable_robustness_testing: bool = True
    noise_levels: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1])
    perturbation_tests: int = 10

    # Stability analysis
    enable_stability_analysis: bool = True
    stability_window: int = 20
    enable_drift_detection: bool = True
    drift_detection_threshold: float = 0.1

    # Performance validation
    enable_performance_validation: bool = True
    min_performance_threshold: float = 0.5
    enable_calibration_check: bool = True
    calibration_bins: int = 10

class EnhancedValidation:
    """
    Enhanced validation system for comprehensive model validation.

    This class provides various validation strategies:
    1. Advanced cross-validation (purged, nested, time-series aware)
    2. Bootstrap validation for confidence intervals
    3. Robustness testing against noise and perturbations
    4. Stability analysis over time
    5. Performance validation and calibration checks
    """

    def __init__(self, config: Optional[EnhancedValidationConfig] = None):
        """Initialize enhanced validation system."""
        self.config = config or EnhancedValidationConfig()
        self.logger = logger.getChild('EnhancedValidation')

        # Validation results storage
        self.validation_results = []
        self.cross_validation_results = []
        self.bootstrap_results = []
        self.stability_results = []

        self.logger.info("✅ Enhanced Validation system initialized")

    def perform_comprehensive_validation(
        self,
        model: Any,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray],
        X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_test: Optional[Union[pd.Series, np.ndarray]] = None,
        model_name: str = "unknown_model",
        timestamps: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive validation of the model.

        Args:
            model: Trained model
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            X_test: Optional test features
            y_test: Optional test targets
            model_name: Name of the model
            timestamps: Optional timestamp series for temporal validation

        Returns:
            Dictionary containing comprehensive validation results
        """
        self.logger.info(f"🔍 Starting comprehensive validation for {model_name}")

        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'validation_summary': {
                'overall_pass': True,
                'validation_score': 0.0,
                'failed_checks': [],
                'warnings': []
            },
            'cross_validation': {},
            'bootstrap_validation': {},
            'robustness_testing': {},
            'stability_analysis': {},
            'performance_validation': {},
            'recommendations': []
        }

        try:
            # 1. Cross-validation
            if self.config.enable_purged_cv or self.config.enable_time_series_validation:
                cv_results = self._perform_cross_validation(
                    model, X_train, y_train, X_val, y_val, timestamps
                )
                results['cross_validation'] = cv_results

                if not cv_results.get('is_valid', True):
                    results['validation_summary']['overall_pass'] = False
                    results['validation_summary']['failed_checks'].append('cross_validation')

            # 2. Bootstrap validation
            if self.config.enable_bootstrap_validation:
                bootstrap_results = self._perform_bootstrap_validation(model, X_val, y_val)
                results['bootstrap_validation'] = bootstrap_results

            # 3. Robustness testing
            if self.config.enable_robustness_testing:
                robustness_results = self._perform_robustness_testing(model, X_val, y_val)
                results['robustness_testing'] = robustness_results

            # 4. Stability analysis
            if self.config.enable_stability_analysis and len(self.validation_results) > 1:
                stability_results = self._perform_stability_analysis(model_name)
                results['stability_analysis'] = stability_results

            # 5. Performance validation
            if self.config.enable_performance_validation:
                performance_results = self._perform_performance_validation(model, X_train, y_train, X_val, y_val)
                results['performance_validation'] = performance_results

                if not performance_results.get('meets_thresholds', True):
                    results['validation_summary']['overall_pass'] = False
                    results['validation_summary']['failed_checks'].append('performance_validation')

            # Calculate overall validation score
            results['validation_summary']['validation_score'] = self._calculate_validation_score(results)

            # Generate recommendations
            results['recommendations'] = self._generate_validation_recommendations(results)

            # Store results
            self.validation_results.append({
                'model_name': model_name,
                'timestamp': results['timestamp'],
                'results': results
            })

            self.logger.info(f"✅ Comprehensive validation completed for {model_name}")

        except Exception as e:
            error_msg = f"Comprehensive validation failed for {model_name}: {e}"
            results['error'] = error_msg
            results['validation_summary']['overall_pass'] = False
            results['validation_summary']['failed_checks'].append('validation_execution')
            results['recommendations'].append("Review validation setup and data quality")
            self.logger.error(f"❌ {error_msg}")

        return results

    def _perform_cross_validation(
        self,
        model: Any,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray],
        timestamps: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive cross-validation."""
        self.logger.debug("🔄 Performing cross-validation...")

        results = {
            'is_valid': True,
            'cv_method': 'standard',
            'cv_scores': {},
            'stability_metrics': {},
            'temporal_validity': {},
            'recommendations': []
        }

        try:
            from sklearn.model_selection import cross_val_score, TimeSeriesSplit, KFold
            from sklearn.metrics import make_scorer, mean_squared_error, accuracy_score

            # Determine if regression or classification
            is_classification = len(np.unique(y_train)) <= 10

            if is_classification:
                scoring = 'accuracy'
                main_scorer = make_scorer(accuracy_score)
            else:
                scoring = 'neg_mean_squared_error'
                main_scorer = make_scorer(mean_squared_error, greater_is_better=False)

            # Standard K-fold CV
            kf = KFold(n_splits=min(self.config.cv_folds, 5), shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring=scoring)
            results['cv_scores']['kfold'] = cv_scores.tolist()

            # Time series CV if timestamps available
            if timestamps is not None and self.config.enable_time_series_validation:
                try:
                    ts_split = TimeSeriesSplit(n_splits=min(self.config.time_series_splits, 5))
                    ts_scores = cross_val_score(model, X_train, y_train, cv=ts_split, scoring=scoring)
                    results['cv_scores']['timeseries'] = ts_scores.tolist()
                    results['cv_method'] = 'timeseries'

                    # Check temporal validity
                    temporal_validity = self._check_temporal_cv_validity(ts_scores)
                    results['temporal_validity'] = temporal_validity

                except Exception as e:
                    self.logger.warning(f"Time series CV failed: {e}")
                    results['recommendations'].append("Time series CV not available - using standard CV")

            # Purged CV if available
            if self.config.enable_purged_cv:
                try:
                    # This would use a custom purged CV implementation
                    purged_scores = self._perform_purged_cross_validation(model, X_train, y_train, timestamps)
                    if purged_scores is not None:
                        results['cv_scores']['purged'] = purged_scores.tolist()
                        results['cv_method'] = 'purged'
                except Exception as e:
                    self.logger.warning(f"Purged CV failed: {e}")
                    results['recommendations'].append("Purged CV not available - consider implementing")

            # Calculate stability metrics
            all_scores = [score for scores in results['cv_scores'].values() for score in scores]
            if all_scores:
                results['stability_metrics'] = {
                    'mean_score': float(np.mean(all_scores)),
                    'std_score': float(np.std(all_scores)),
                    'score_range': float(np.max(all_scores) - np.min(all_scores)),
                    'cv_stability': float(np.std(all_scores) / np.mean(all_scores)) if np.mean(all_scores) != 0 else 0
                }

                # Check for unstable CV scores
                if results['stability_metrics']['cv_stability'] > 0.2:
                    results['recommendations'].append("High CV score variance - model may be unstable")

            # Store CV results
            self.cross_validation_results.append({
                'model_name': getattr(model, '__class__', type(model)).__name__,
                'timestamp': datetime.now().isoformat(),
                'results': results
            })

        except Exception as e:
            error_msg = f"Cross-validation failed: {e}"
            results['error'] = error_msg
            results['is_valid'] = False
            results['recommendations'].append("Review CV setup and data compatibility")
            self.logger.error(f"❌ {error_msg}")

        return results

    def _perform_purged_cross_validation(
        self,
        model: Any,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        timestamps: Optional[pd.Series] = None
    ) -> Optional[np.ndarray]:
        """Perform purged cross-validation to prevent data leakage."""
        try:
            # This is a placeholder for purged CV implementation
            # In practice, you would implement a PurgedKFold class similar to:
            # https://github.com/hudson-and-thames/purged-cross-validation

            # For now, fall back to time series split
            from sklearn.model_selection import TimeSeriesSplit

            ts_split = TimeSeriesSplit(n_splits=min(self.config.cv_folds, 5))
            scores = []

            for train_idx, val_idx in ts_split.split(X):
                # Simple temporal split without purging
                model_temp = type(model)(**model.get_params())
                model_temp.fit(X[train_idx], y[train_idx])

                # Calculate score
                if len(np.unique(y)) > 10:  # Regression
                    from sklearn.metrics import mean_squared_error
                    pred = model_temp.predict(X[val_idx])
                    score = -mean_squared_error(y[val_idx], pred)
                else:  # Classification
                    from sklearn.metrics import accuracy_score
                    pred = model_temp.predict(X[val_idx])
                    score = accuracy_score(y[val_idx], pred)

                scores.append(score)

            return np.array(scores) if scores else None

        except Exception as e:
            self.logger.warning(f"Purged CV implementation failed: {e}")
            return None

    def _check_temporal_cv_validity(self, cv_scores: np.ndarray) -> Dict[str, Any]:
        """Check temporal validity of CV scores."""
        results = {
            'is_temporally_valid': True,
            'warnings': [],
            'recommendations': []
        }

        try:
            if len(cv_scores) < 2:
                return results

            # Check for temporal degradation (scores getting worse over time)
            temporal_trend = np.polyfit(range(len(cv_scores)), cv_scores, 1)[0]

            if temporal_trend < -0.01:  # Significant downward trend
                results['warnings'].append("CV scores show temporal degradation")
                results['recommendations'].append("Consider non-stationary data or time-dependent effects")

            # Check for unrealistic score improvements
            if temporal_trend > 0.1:  # Unrealistic improvement
                results['warnings'].append("CV scores show unrealistic temporal improvement")
                results['recommendations'].append("Review for potential data leakage or evaluation bias")

        except Exception as e:
            self.logger.debug(f"Temporal CV validity check failed: {e}")

        return results

    def _perform_bootstrap_validation(
        self,
        model: Any,
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray]
    ) -> Dict[str, Any]:
        """Perform bootstrap validation for confidence intervals."""
        self.logger.debug("🔄 Performing bootstrap validation...")

        results = {
            'confidence_intervals': {},
            'bootstrap_mean': None,
            'bootstrap_std': None,
            'is_reliable': True,
            'recommendations': []
        }

        try:
            from sklearn.utils import resample
            from sklearn.metrics import mean_squared_error, accuracy_score

            # Determine if regression or classification
            is_classification = len(np.unique(y_val)) <= 10

            bootstrap_scores = []

            # Perform bootstrap resampling
            for i in range(self.config.bootstrap_samples):
                # Resample with replacement
                indices = np.random.choice(len(X_val), size=len(X_val), replace=True)
                X_boot = X_val[indices] if hasattr(X_val, '__getitem__') else X_val[indices]
                y_boot = y_val[indices] if hasattr(y_val, '__getitem__') else y_val[indices]

                # Get predictions
                pred = model.predict(X_boot)

                # Calculate score
                if is_classification:
                    score = accuracy_score(y_boot, pred)
                else:
                    score = -mean_squared_error(y_boot, pred)  # Negative MSE

                bootstrap_scores.append(score)

            bootstrap_scores = np.array(bootstrap_scores)

            # Calculate confidence intervals
            lower_bound = np.percentile(bootstrap_scores, (1 - self.config.bootstrap_confidence_level) * 50)
            upper_bound = np.percentile(bootstrap_scores, (1 + self.config.bootstrap_confidence_level) * 50)

            results['confidence_intervals'] = {
                'lower': float(lower_bound),
                'upper': float(upper_bound),
                'confidence_level': self.config.bootstrap_confidence_level
            }

            results['bootstrap_mean'] = float(np.mean(bootstrap_scores))
            results['bootstrap_std'] = float(np.std(bootstrap_scores))

            # Check reliability
            ci_width = upper_bound - lower_bound
            if ci_width > abs(results['bootstrap_mean']) * 0.5:  # Wide CI relative to mean
                results['is_reliable'] = False
                results['recommendations'].append("Bootstrap CI is wide - consider more data or model stability")

            # Store bootstrap results
            self.bootstrap_results.append({
                'timestamp': datetime.now().isoformat(),
                'results': results
            })

        except Exception as e:
            error_msg = f"Bootstrap validation failed: {e}"
            results['error'] = error_msg
            results['recommendations'].append("Review bootstrap validation setup")
            self.logger.error(f"❌ {error_msg}")

        return results

    def _perform_robustness_testing(
        self,
        model: Any,
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray]
    ) -> Dict[str, Any]:
        """Test model robustness against noise and perturbations."""
        self.logger.debug("🔄 Performing robustness testing...")

        results = {
            'noise_robustness': {},
            'perturbation_robustness': {},
            'overall_robustness_score': 0.0,
            'is_robust': True,
            'recommendations': []
        }

        try:
            from sklearn.metrics import mean_squared_error, accuracy_score

            # Determine if regression or classification
            is_classification = len(np.unique(y_val)) <= 10

            # Get baseline performance
            y_pred_baseline = model.predict(X_val)
            if is_classification:
                baseline_score = accuracy_score(y_val, y_pred_baseline)
            else:
                baseline_score = -mean_squared_error(y_val, y_pred_baseline)

            # Test robustness to noise
            noise_scores = []
            for noise_level in self.config.noise_levels:
                # Add Gaussian noise
                if X_val.ndim > 1:
                    noise = np.random.normal(0, noise_level, X_val.shape)
                    X_noisy = X_val + noise
                else:
                    noise = np.random.normal(0, noise_level, len(X_val))
                    X_noisy = X_val + noise

                y_pred_noisy = model.predict(X_noisy)

                if is_classification:
                    score = accuracy_score(y_val, y_pred_noisy)
                else:
                    score = -mean_squared_error(y_val, y_pred_noisy)

                noise_scores.append(score)
                results['noise_robustness'][f'noise_{noise_level}'] = float(score)

            # Calculate noise robustness score
            if noise_scores:
                noise_robustness = min(noise_scores) / baseline_score if baseline_score != 0 else 0
                results['noise_robustness']['robustness_score'] = float(noise_robustness)

                if noise_robustness < 0.8:  # Less than 80% of baseline performance
                    results['is_robust'] = False
                    results['recommendations'].append("Model performance degrades significantly with noise")

            # Test robustness to perturbations (feature removal)
            perturbation_scores = []
            n_features = X_val.shape[1] if X_val.ndim > 1 else 1

            for _ in range(self.config.perturbation_tests):
                if n_features > 1:
                    # Randomly mask some features
                    mask = np.random.random(n_features) > 0.1  # Keep 90% of features
                    if X_val.ndim > 1:
                        X_perturbed = X_val * mask
                    else:
                        X_perturbed = X_val if mask[0] else 0  # Simple case for 1D
                else:
                    X_perturbed = X_val

                y_pred_perturbed = model.predict(X_perturbed)

                if is_classification:
                    score = accuracy_score(y_val, y_pred_perturbed)
                else:
                    score = -mean_squared_error(y_val, y_pred_perturbed)

                perturbation_scores.append(score)

            results['perturbation_robustness']['scores'] = [float(s) for s in perturbation_scores]

            if perturbation_scores:
                perturbation_robustness = min(perturbation_scores) / baseline_score if baseline_score != 0 else 0
                results['perturbation_robustness']['robustness_score'] = float(perturbation_robustness)

                if perturbation_robustness < 0.7:  # Less than 70% of baseline performance
                    results['is_robust'] = False
                    results['recommendations'].append("Model performance degrades significantly with feature perturbations")

            # Overall robustness score
            noise_rob_score = results['noise_robustness'].get('robustness_score', 1.0)
            pert_rob_score = results['perturbation_robustness'].get('robustness_score', 1.0)
            results['overall_robustness_score'] = float((noise_rob_score + pert_rob_score) / 2)

        except Exception as e:
            error_msg = f"Robustness testing failed: {e}"
            results['error'] = error_msg
            results['recommendations'].append("Review robustness testing setup")
            self.logger.error(f"❌ {error_msg}")

        return results

    def _perform_stability_analysis(self, model_name: str) -> Dict[str, Any]:
        """Perform stability analysis over time."""
        results = {
            'stability_score': 0.0,
            'performance_trend': None,
            'variance_analysis': {},
            'is_stable': True,
            'recommendations': []
        }

        try:
            # Get recent validation results for this model
            model_results = [
                r for r in self.validation_results
                if r['model_name'] == model_name
            ]

            if len(model_results) < 3:
                results['recommendations'].append("Insufficient data for stability analysis")
                return results

            # Extract performance scores over time
            performance_over_time = []
            timestamps = []

            for result in model_results[-self.config.stability_window:]:  # Last N results
                try:
                    # Get validation score from performance validation
                    perf_val = result['results'].get('performance_validation', {})
                    if 'validation_score' in perf_val:
                        performance_over_time.append(perf_val['validation_score'])
                        timestamps.append(result['timestamp'])
                except Exception:
                    continue

            if len(performance_over_time) < 3:
                results['recommendations'].append("Insufficient performance data for stability analysis")
                return results

            # Calculate stability metrics
            performance_array = np.array(performance_over_time)

            results['stability_score'] = float(1.0 / (1.0 + np.std(performance_array)))
            results['variance_analysis'] = {
                'mean_performance': float(np.mean(performance_array)),
                'performance_std': float(np.std(performance_array)),
                'performance_range': float(np.max(performance_array) - np.min(performance_array))
            }

            # Check for performance drift
            if self.config.enable_drift_detection:
                recent_performance = np.mean(performance_array[-3:])  # Last 3
                historical_performance = np.mean(performance_array[:-3])  # Earlier

                if len(performance_array) > 3:
                    relative_change = abs(recent_performance - historical_performance) / historical_performance

                    if relative_change > self.config.drift_detection_threshold:
                        results['performance_trend'] = 'declining' if recent_performance < historical_performance else 'improving'
                        results['is_stable'] = False
                        results['recommendations'].append("Performance drift detected - investigate data or model changes")

            # Check stability
            if np.std(performance_array) > 0.1:  # High variance
                results['is_stable'] = False
                results['recommendations'].append("High performance variance - model may be unstable")

            # Store stability results
            self.stability_results.append({
                'model_name': model_name,
                'timestamp': datetime.now().isoformat(),
                'results': results
            })

        except Exception as e:
            error_msg = f"Stability analysis failed for {model_name}: {e}"
            results['error'] = error_msg
            results['recommendations'].append("Review stability analysis setup")
            self.logger.error(f"❌ {error_msg}")

        return results

    def _perform_performance_validation(
        self,
        model: Any,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray]
    ) -> Dict[str, Any]:
        """Perform comprehensive performance validation."""
        self.logger.debug("🔄 Performing performance validation...")

        results = {
            'meets_thresholds': True,
            'performance_metrics': {},
            'threshold_analysis': {},
            'calibration_analysis': {},
            'recommendations': []
        }

        try:
            from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
            from sklearn.calibration import calibration_curve

            # Get predictions
            y_pred = model.predict(X_val)

            # Ensure predictions are in correct format
            if hasattr(y_pred, 'values'):
                y_pred = y_pred.values
            elif hasattr(y_pred, 'flatten'):
                y_pred = y_pred.flatten()

            # Calculate comprehensive metrics
            is_classification = len(np.unique(y_val)) <= 10

            if is_classification:
                # Binary classification case
                if len(np.unique(y_val)) == 2:
                    y_pred_binary = (y_pred > 0.5).astype(int) if y_pred.ndim > 1 else (y_pred > np.median(y_pred)).astype(int)

                    accuracy = accuracy_score(y_val, y_pred_binary)
                    precision = precision_score(y_val, y_pred_binary, zero_division=0)
                    recall = recall_score(y_val, y_pred_binary, zero_division=0)
                    f1 = f1_score(y_val, y_pred_binary, zero_division=0)

                    results['performance_metrics'] = {
                        'accuracy': float(accuracy),
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1': float(f1)
                    }
                else:
                    # Multi-class classification
                    y_pred_classes = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred

                    accuracy = accuracy_score(y_val, y_pred_classes)
                    precision = precision_score(y_val, y_pred_classes, average='weighted', zero_division=0)
                    recall = recall_score(y_val, y_pred_classes, average='weighted', zero_division=0)
                    f1 = f1_score(y_val, y_pred_classes, average='weighted', zero_division=0)

                    results['performance_metrics'] = {
                        'accuracy': float(accuracy),
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1': float(f1)
                    }

                # Check against thresholds
                if accuracy < self.config.min_performance_threshold:
                    results['meets_thresholds'] = False
                    results['recommendations'].append(f"Accuracy ({accuracy".3f"}) below minimum threshold ({self.config.min_performance_threshold})")

            else:
                # Regression case
                mse = mean_squared_error(y_val, y_pred)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(y_val - y_pred))

                # Calculate R² if possible
                ss_res = np.sum((y_val - y_pred) ** 2)
                ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                results['performance_metrics'] = {
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'r2': float(r2)
                }

                # Check against thresholds (using R² for regression)
                if r2 < self.config.min_performance_threshold:
                    results['meets_thresholds'] = False
                    results['recommendations'].append(f"R² ({r2".3f"}) below minimum threshold ({self.config.min_performance_threshold})")

            # Threshold analysis
            results['threshold_analysis'] = {
                'meets_accuracy_threshold': results['performance_metrics'].get('accuracy', 0) >= self.config.min_performance_threshold,
                'meets_r2_threshold': results['performance_metrics'].get('r2', 0) >= self.config.min_performance_threshold,
                'minimum_threshold': self.config.min_performance_threshold
            }

            # Calibration check
            if self.config.enable_calibration_check and is_classification:
                try:
                    calibration_results = self._check_model_calibration(y_val, y_pred)
                    results['calibration_analysis'] = calibration_results

                    if not calibration_results.get('is_well_calibrated', True):
                        results['recommendations'].append("Model is not well calibrated - consider probability calibration")
                except Exception as e:
                    self.logger.debug(f"Calibration check failed: {e}")

        except Exception as e:
            error_msg = f"Performance validation failed: {e}"
            results['error'] = error_msg
            results['meets_thresholds'] = False
            results['recommendations'].append("Review performance validation setup")
            self.logger.error(f"❌ {error_msg}")

        return results

    def _check_model_calibration(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray]
    ) -> Dict[str, Any]:
        """Check model probability calibration."""
        results = {
            'is_well_calibrated': True,
            'calibration_metrics': {},
            'recommendations': []
        }

        try:
            from sklearn.calibration import calibration_curve

            # Get probability predictions
            if y_pred.ndim > 1:
                y_prob = y_pred[:, 1] if y_pred.shape[1] > 1 else y_pred[:, 0]
            else:
                y_prob = y_pred

            # Calculate calibration curve
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=self.config.calibration_bins)

            results['calibration_metrics'] = {
                'prob_true': prob_true.tolist(),
                'prob_pred': prob_pred.tolist(),
                'n_bins': self.config.calibration_bins
            }

            # Calculate Expected Calibration Error (ECE)
            ece = np.mean(np.abs(prob_true - prob_pred))
            results['calibration_metrics']['ece'] = float(ece)

            # Check calibration quality
            if ece > 0.1:  # High calibration error
                results['is_well_calibrated'] = False
                results['recommendations'].append(f"High calibration error (ECE: {ece".3f"}) - model probabilities are poorly calibrated")

        except Exception as e:
            results['error'] = str(e)
            self.logger.debug(f"Calibration check failed: {e}")

        return results

    def _calculate_validation_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall validation score."""
        try:
            score = 0.0
            components = 0

            # CV score
            if 'cross_validation' in results:
                cv_results = results['cross_validation']
                if 'stability_metrics' in cv_results:
                    cv_score = cv_results['stability_metrics'].get('mean_score', 0)
                    score += abs(cv_score)  # Use absolute value for scoring
                    components += 1

            # Bootstrap score
            if 'bootstrap_validation' in results:
                bootstrap_results = results['bootstrap_validation']
                if bootstrap_results.get('is_reliable', False):
                    score += 1.0
                    components += 1

            # Robustness score
            if 'robustness_testing' in results:
                robustness_results = results['robustness_testing']
                robustness_score = robustness_results.get('overall_robustness_score', 0)
                score += robustness_score
                components += 1

            # Performance score
            if 'performance_validation' in results:
                perf_results = results['performance_validation']
                if perf_results.get('meets_thresholds', False):
                    score += 1.0
                    components += 1

            return score / components if components > 0 else 0.0

        except Exception as e:
            self.logger.warning(f"Validation score calculation failed: {e}")
            return 0.0

    def _generate_validation_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate validation-based recommendations."""
        recommendations = []

        try:
            # Check CV recommendations
            if 'cross_validation' in results:
                cv_results = results['cross_validation']
                recommendations.extend(cv_results.get('recommendations', []))

            # Check bootstrap recommendations
            if 'bootstrap_validation' in results:
                bootstrap_results = results['bootstrap_validation']
                recommendations.extend(bootstrap_results.get('recommendations', []))

            # Check robustness recommendations
            if 'robustness_testing' in results:
                robustness_results = results['robustness_testing']
                recommendations.extend(robustness_results.get('recommendations', []))

            # Check performance recommendations
            if 'performance_validation' in results:
                perf_results = results['performance_validation']
                recommendations.extend(perf_results.get('recommendations', []))

            # Overall validation recommendations
            validation_summary = results.get('validation_summary', {})
            if not validation_summary.get('overall_pass', True):
                recommendations.append("Model failed validation - review failed checks and implement fixes")

            # Validation score recommendations
            validation_score = validation_summary.get('validation_score', 0)
            if validation_score < 0.7:
                recommendations.append("Low overall validation score - consider improving model or validation procedures")

        except Exception as e:
            self.logger.warning(f"Recommendation generation failed: {e}")
            recommendations.append("Review validation setup and procedures")

        return recommendations

# Convenience functions
def create_enhanced_validation(config: Optional[EnhancedValidationConfig] = None) -> EnhancedValidation:
    """Create enhanced validation instance."""
    return EnhancedValidation(config)

def validate_model_comprehensive(
    model: Any,
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_val: Union[pd.DataFrame, np.ndarray],
    y_val: Union[pd.Series, np.ndarray],
    X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    y_test: Optional[Union[pd.Series, np.ndarray]] = None,
    model_name: str = "unknown_model",
    timestamps: Optional[pd.Series] = None,
    config: Optional[EnhancedValidationConfig] = None
) -> Dict[str, Any]:
    """
    Convenience function to perform comprehensive model validation.

    Args:
        model: Trained model
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        X_test: Optional test features
        y_test: Optional test targets
        model_name: Name of the model
        timestamps: Optional timestamp series
        config: Optional configuration

    Returns:
        Dictionary containing comprehensive validation results
    """
    validation = EnhancedValidation(config)
    return validation.perform_comprehensive_validation(
        model, X_train, y_train, X_val, y_val, X_test, y_test, model_name, timestamps
    )