"""
Cross-Validation Utilities

This module provides comprehensive cross-validation utilities with temporal integrity,
walk-forward validation, and advanced CV metrics for time series data.

Key Features:
- Temporal cross-validation with gap support
- Walk-forward validation (expanding and rolling windows)
- Comprehensive CV metrics with stability assessment
- Out-of-sample performance evaluation
- Time series split utilities
- CV result aggregation and analysis

Built on existing utilities:
- Uses math_validation.py for safe mathematical operations
- Integrates with m1_gpu_utils.py for GPU acceleration
- Leverages common_operations.py for robust error handling
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.utils.class_weight import compute_sample_weight
import warnings

from ..math_validation import safe_divide, safe_log
from ..common_operations import create_fallback_logger
from ..m1_gpu_utils import M1GPUManager
from ..parallel_processing_optimizer import ParallelProcessor

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - GPU acceleration disabled")

try:
    from sklearn.model_selection import cross_val_score, cross_validate
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available - limited CV functionality")


class CrossValidationUtilities:
    """Comprehensive cross-validation utilities with temporal integrity."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize CV utilities with configuration."""
        self.config = config or {}
        self.logger = logger.getChild('CVUtils')
        self.gpu_manager = M1GPUManager() if TORCH_AVAILABLE else None
        self.parallel_processor = ParallelProcessor()

        # Configuration defaults
        self.enable_gpu = self.config.get('enable_gpu', TORCH_AVAILABLE)
        self.enable_parallel = self.config.get('enable_parallel', True)
        self.max_workers = self.config.get('max_workers', 4)
        self.memory_threshold = self.config.get('memory_threshold', 0.8)

    def perform_temporal_cv(self, X: np.ndarray, y: np.ndarray,
                          model: Any, n_splits: int = 5,
                          gap: int = 0, test_size: Optional[int] = None,
                          scorer: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Perform temporal cross-validation with gap support.

        Args:
            X: Feature matrix
            y: Target array
            model: ML model with fit/predict methods
            n_splits: Number of CV splits
            gap: Gap between train and test sets (in samples)
            test_size: Size of test set for each split
            scorer: Custom scoring function

        Returns:
            Dictionary with CV results and metrics
        """
        try:
            self.logger.info(f"🔄 Starting temporal CV with {n_splits} splits, gap={gap}")

            # Validate inputs
            if len(X) != len(y):
                raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")

            if len(X) < (n_splits + 1) * 10:  # Minimum samples check
                raise ValueError(f"Insufficient data for {n_splits} splits: {len(X)} samples")

            # Create time series split with gap
            if test_size is None:
                test_size = max(1, len(X) // (n_splits + 1))

            tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)

            # Initialize results storage
            cv_results = {
                'fold_results': [],
                'predictions': [],
                'true_values': [],
                'metrics': {},
                'fold_metrics': [],
                'feature_importance': [],
                'training_times': [],
                'prediction_times': []
            }

            # Perform cross-validation
            for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
                try:
                    self.logger.info(f"📊 Processing fold {fold_idx + 1}/{n_splits}")

                    # Split data
                    X_train_fold, X_test_fold = X[train_idx], X[test_idx]
                    y_train_fold, y_test_fold = y[train_idx], y[test_idx]

                    # Handle class imbalance (guarded for estimator support)
                    sample_weight = None
                    try:
                        sample_weight = self._compute_sample_weights(y_train_fold)
                        # Only pass if estimator supports sample_weight in fit signature
                        if hasattr(model, 'fit'):
                            import inspect
                            sig = inspect.signature(model.fit)
                            if 'sample_weight' not in sig.parameters:
                                sample_weight = None
                    except Exception:
                        sample_weight = None

                    # Train model
                    start_time = datetime.now()
                    model_copy = self._clone_model(model)
                    if sample_weight is not None:
                        model_copy.fit(X_train_fold, y_train_fold, sample_weight=sample_weight)
                    else:
                        model_copy.fit(X_train_fold, y_train_fold)
                    training_time = (datetime.now() - start_time).total_seconds()

                    # Make predictions
                    start_time = datetime.now()
                    if hasattr(model_copy, 'predict_proba'):
                        y_pred_proba = model_copy.predict_proba(X_test_fold)
                        y_pred = model_copy.predict(X_test_fold)
                    else:
                        y_pred = model_copy.predict(X_test_fold)
                        y_pred_proba = None
                    prediction_time = (datetime.now() - start_time).total_seconds()

                    # Calculate fold metrics
                    fold_metrics = self._calculate_fold_metrics(
                        y_test_fold, y_pred, y_pred_proba
                    )

                    # Store fold results
                    fold_result = {
                        'fold_idx': fold_idx,
                        'train_samples': len(train_idx),
                        'test_samples': len(test_idx),
                        'predictions': y_pred,
                        'true_values': y_test_fold,
                        'metrics': fold_metrics,
                        'training_time': training_time,
                        'prediction_time': prediction_time
                    }

                    # Extract feature importance if available
                    if hasattr(model_copy, 'feature_importances_'):
                        fold_result['feature_importance'] = model_copy.feature_importances_
                        cv_results['feature_importance'].append(model_copy.feature_importances_)

                    cv_results['fold_results'].append(fold_result)
                    cv_results['fold_metrics'].append(fold_metrics)
                    cv_results['training_times'].append(training_time)
                    cv_results['prediction_times'].append(prediction_time)

                    self.logger.info(f"✅ Fold {fold_idx + 1} completed - "
                                   f"Accuracy: {fold_metrics.get('accuracy', 'N/A'):.4f}")

                except Exception as fold_e:
                    self.logger.warning(f"⚠️ Fold {fold_idx + 1} failed: {fold_e}")
                    continue

            # Aggregate results
            if cv_results['fold_results']:
                cv_results['metrics'] = self._aggregate_cv_metrics(cv_results['fold_metrics'])
                cv_results['summary'] = self._create_cv_summary(cv_results)

                self.logger.info(f"✅ Temporal CV completed: "
                               f"Mean accuracy: {cv_results['metrics'].get('accuracy_mean', 'N/A'):.4f}")
            else:
                self.logger.error("❌ No folds completed successfully")
                cv_results['error'] = "No successful folds"

            return cv_results

        except Exception as e:
            self.logger.error(f"❌ Temporal CV failed: {e}")
            return {'error': str(e), 'fold_results': []}

    def walk_forward_validation(self, X: np.ndarray, y: np.ndarray,
                              model: Any, initial_train_size: int = 1000,
                              test_size: int = 100, step_size: int = 50,
                              expanding_window: bool = True) -> Dict[str, Any]:
        """
        Perform walk-forward validation with expanding or rolling windows.

        Args:
            X: Feature matrix
            y: Target array
            model: ML model
            initial_train_size: Initial training set size
            test_size: Test set size for each iteration
            step_size: Step size for moving window
            expanding_window: Use expanding window (True) or rolling window (False)

        Returns:
            Walk-forward validation results
        """
        try:
            self.logger.info(f"🚶 Starting walk-forward validation "
                           f"(window: {'expanding' if expanding_window else 'rolling'})")

            if len(X) < initial_train_size + test_size:
                raise ValueError(f"Insufficient data: {len(X)} < {initial_train_size + test_size}")

            wfv_results = {
                'iterations': [],
                'predictions': [],
                'true_values': [],
                'metrics': {},
                'training_times': [],
                'prediction_times': []
            }

            # Perform walk-forward validation
            current_position = initial_train_size

            while current_position + test_size <= len(X):
                try:
                    # Define training window
                    if expanding_window:
                        train_start = 0
                    else:
                        train_start = max(0, current_position - initial_train_size)

                    train_end = current_position
                    test_end = current_position + test_size

                    # Split data
                    X_train = X[train_start:train_end]
                    y_train = y[train_start:train_end]
                    X_test = X[current_position:test_end]
                    y_test = y[current_position:test_end]

                    # Handle class imbalance
                    sample_weight = self._compute_sample_weights(y_train)

                    # Train model
                    start_time = datetime.now()
                    model_copy = self._clone_model(model)
                    model_copy.fit(X_train, y_train, sample_weight=sample_weight)
                    training_time = (datetime.now() - start_time).total_seconds()

                    # Make predictions
                    start_time = datetime.now()
                    if hasattr(model_copy, 'predict_proba'):
                        y_pred_proba = model_copy.predict_proba(X_test)
                        y_pred = model_copy.predict(X_test)
                    else:
                        y_pred = model_copy.predict(X_test)
                        y_pred_proba = None
                    prediction_time = (datetime.now() - start_time).total_seconds()

                    # Calculate metrics
                    metrics = self._calculate_fold_metrics(y_test, y_pred, y_pred_proba)

                    # Store results
                    iteration_result = {
                        'iteration': len(wfv_results['iterations']),
                        'train_start': train_start,
                        'train_end': train_end,
                        'test_start': current_position,
                        'test_end': test_end,
                        'train_samples': len(X_train),
                        'test_samples': len(X_test),
                        'predictions': y_pred,
                        'true_values': y_test,
                        'metrics': metrics,
                        'training_time': training_time,
                        'prediction_time': prediction_time
                    }

                    wfv_results['iterations'].append(iteration_result)
                    wfv_results['training_times'].append(training_time)
                    wfv_results['prediction_times'].append(prediction_time)

                    # Move to next position
                    current_position += step_size

                    self.logger.info(f"✅ Iteration {len(wfv_results['iterations'])} completed - "
                                   f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")

                except Exception as iter_e:
                    self.logger.warning(f"⚠️ Walk-forward iteration failed: {iter_e}")
                    current_position += step_size
                    continue

            # Aggregate results
            if wfv_results['iterations']:
                all_metrics = [iter['metrics'] for iter in wfv_results['iterations']]
                wfv_results['metrics'] = self._aggregate_cv_metrics(all_metrics)
                wfv_results['summary'] = self._create_wfv_summary(wfv_results)

                self.logger.info(f"✅ Walk-forward validation completed: "
                               f"{len(wfv_results['iterations'])} iterations, "
                               f"Mean accuracy: {wfv_results['metrics'].get('accuracy_mean', 'N/A'):.4f}")
            else:
                self.logger.error("❌ No walk-forward iterations completed")

            return wfv_results

        except Exception as e:
            self.logger.error(f"❌ Walk-forward validation failed: {e}")
            return {'error': str(e), 'iterations': []}

    def cross_validation_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                               y_prob: Optional[np.ndarray] = None,
                               task_type: str = 'classification') -> Dict[str, Any]:
        """
        Calculate comprehensive cross-validation metrics.

        Args:
            y_true: True values
            y_pred: Predicted values
            y_prob: Prediction probabilities (for classification)
            task_type: 'classification' or 'regression'

        Returns:
            Dictionary of metrics
        """
        try:
            metrics = {}

            if task_type == 'classification':
                # Basic classification metrics
                metrics['accuracy'] = accuracy_score(y_true, y_pred)
                metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
                metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
                metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
                metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
                metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')

                # ROC AUC if probabilities available
                if y_prob is not None:
                    try:
                        if y_prob.shape[1] == 2:  # Binary classification
                            metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
                        else:  # Multi-class
                            metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
                    except Exception as auc_e:
                        self.logger.warning(f"ROC AUC calculation failed: {auc_e}")

            elif task_type == 'regression':
                # Regression metrics
                metrics['mse'] = mean_squared_error(y_true, y_pred)
                metrics['mae'] = mean_absolute_error(y_true, y_pred)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                metrics['r2'] = r2_score(y_true, y_pred)

                # Additional regression metrics
                metrics['mape'] = self._calculate_mape(y_true, y_pred)
                metrics['smape'] = self._calculate_smape(y_true, y_pred)

            else:
                raise ValueError(f"Unsupported task type: {task_type}")

            # Add stability metrics
            metrics.update(self._calculate_stability_metrics(y_true, y_pred))

            return metrics

        except Exception as e:
            self.logger.error(f"❌ Metrics calculation failed: {e}")
            return {'error': str(e)}

    def stability_assessment(self, cv_results: Dict[str, Any],
                           threshold: float = 0.1) -> Dict[str, Any]:
        """
        Assess stability of cross-validation results.

        Args:
            cv_results: Cross-validation results dictionary
            threshold: Stability threshold

        Returns:
            Stability assessment results
        """
        try:
            if 'fold_metrics' not in cv_results or not cv_results['fold_metrics']:
                return {'error': 'No fold metrics available for stability assessment'}

            fold_metrics = cv_results['fold_metrics']

            # Calculate stability metrics
            stability_results = {}

            # Get all metric names
            metric_names = set()
            for fold_metric in fold_metrics:
                metric_names.update(fold_metric.keys())

            # Calculate stability for each metric
            for metric_name in metric_names:
                if metric_name == 'error':
                    continue

                metric_values = []
                for fold_metric in fold_metrics:
                    if metric_name in fold_metric:
                        metric_values.append(fold_metric[metric_name])

                if len(metric_values) > 1:
                    metric_array = np.array(metric_values)
                    stability_results[metric_name] = {
                        'mean': np.mean(metric_array),
                        'std': np.std(metric_array),
                        'cv': safe_divide(np.std(metric_array), np.mean(metric_array)),
                        'min': np.min(metric_array),
                        'max': np.max(metric_array),
                        'stable': np.std(metric_array) < threshold
                    }

            # Overall stability assessment
            if stability_results:
                cv_values = [v['cv'] for v in stability_results.values() if 'cv' in v]
                stability_results['overall_stability'] = {
                    'mean_cv': np.mean(cv_values),
                    'stable_metrics': sum(1 for v in stability_results.values()
                                        if isinstance(v, dict) and v.get('stable', False)),
                    'total_metrics': len([v for v in stability_results.values() if isinstance(v, dict)]),
                    'is_stable': np.mean(cv_values) < threshold
                }

            return stability_results

        except Exception as e:
            self.logger.error(f"❌ Stability assessment failed: {e}")
            return {'error': str(e)}

    def out_of_sample_performance(self, model: Any, X_train: np.ndarray,
                                y_train: np.ndarray, X_test: np.ndarray,
                                y_test: np.ndarray, task_type: str = 'classification') -> Dict[str, Any]:
        """
        Evaluate out-of-sample performance.

        Args:
            model: Trained ML model
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            task_type: Task type ('classification' or 'regression')

        Returns:
            Out-of-sample performance metrics
        """
        try:
            self.logger.info("📊 Evaluating out-of-sample performance")

            # Handle class imbalance in training
            sample_weight = self._compute_sample_weights(y_train)

            # Train model on full training set
            start_time = datetime.now()
            model.fit(X_train, y_train, sample_weight=sample_weight)
            training_time = (datetime.now() - start_time).total_seconds()

            # Make predictions on test set
            start_time = datetime.now()
            if hasattr(model, 'predict_proba') and task_type == 'classification':
                y_pred_proba = model.predict_proba(X_test)
                y_pred = model.predict(X_test)
            else:
                y_pred = model.predict(X_test)
                y_pred_proba = None
            prediction_time = (datetime.now() - start_time).total_seconds()

            # Calculate comprehensive metrics
            metrics = self.cross_validation_metrics(y_test, y_pred, y_pred_proba, task_type)

            # Add training/prediction times
            metrics['training_time'] = training_time
            metrics['prediction_time'] = prediction_time
            metrics['samples_train'] = len(X_train)
            metrics['samples_test'] = len(X_test)

            # Data quality assessment
            metrics['data_quality'] = {
                'train_test_ratio': safe_divide(len(X_train), len(X_test)),
                'feature_dimensionality': X_train.shape[1],
                'test_set_size': len(X_test)
            }

            self.logger.info(f"✅ Out-of-sample evaluation completed - "
                           f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")

            return metrics

        except Exception as e:
            self.logger.error(f"❌ Out-of-sample evaluation failed: {e}")
            return {'error': str(e)}

    def _compute_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """Compute sample weights for imbalanced classes."""
        try:
            if len(np.unique(y)) > 1:
                return compute_sample_weight('balanced', y)
            else:
                return np.ones(len(y))
        except Exception:
            return np.ones(len(y))

    def _clone_model(self, model: Any) -> Any:
        """Clone a model for CV folds."""
        try:
            if hasattr(model, 'clone'):
                return model.clone()
            elif hasattr(model, '__class__'):
                # Create new instance with same parameters
                model_class = model.__class__
                if hasattr(model, 'get_params'):
                    params = model.get_params()
                    return model_class(**params)
                else:
                    return model_class()
            else:
                # Fallback - return original model
                return model
        except Exception as e:
            self.logger.warning(f"Model cloning failed: {e}")
            return model

    def _calculate_fold_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                              y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate metrics for a single fold."""
        try:
            metrics = {}

            # Determine task type from data
            unique_values = np.unique(y_true)
            if len(unique_values) <= 10 and all(isinstance(v, (int, np.integer)) for v in unique_values):
                task_type = 'classification'
            else:
                task_type = 'regression'

            # Calculate appropriate metrics
            if task_type == 'classification':
                metrics['accuracy'] = accuracy_score(y_true, y_pred)
                metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

                if len(unique_values) == 2:  # Binary classification
                    metrics['f1'] = f1_score(y_true, y_pred, average='binary')
                    if y_pred_proba is not None:
                        try:
                            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                        except:
                            pass
                else:  # Multi-class
                    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
                    if y_pred_proba is not None:
                        try:
                            metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
                        except:
                            pass
            else:  # Regression
                metrics['mae'] = mean_absolute_error(y_true, y_pred)
                metrics['mse'] = mean_squared_error(y_true, y_pred)
                metrics['r2'] = r2_score(y_true, y_pred)

            return metrics

        except Exception as e:
            self.logger.warning(f"Fold metrics calculation failed: {e}")
            return {'error': str(e)}

    def _aggregate_cv_metrics(self, fold_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across CV folds."""
        try:
            if not fold_metrics:
                return {}

            # Get all metric names
            all_metrics = {}
            for fold_metric in fold_metrics:
                for metric_name, value in fold_metric.items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        all_metrics[metric_name].append(value)

            # Calculate aggregated statistics
            aggregated = {}
            for metric_name, values in all_metrics.items():
                if values:
                    values_array = np.array(values)
                    aggregated[f"{metric_name}_mean"] = np.mean(values_array)
                    aggregated[f"{metric_name}_std"] = np.std(values_array)
                    aggregated[f"{metric_name}_min"] = np.min(values_array)
                    aggregated[f"{metric_name}_max"] = np.max(values_array)

            return aggregated

        except Exception as e:
            self.logger.warning(f"Metrics aggregation failed: {e}")
            return {}

    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        try:
            mask = y_true != 0
            if not mask.any():
                return np.nan
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        except:
            return np.nan

    def _calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error."""
        try:
            numerator = np.abs(y_true - y_pred)
            denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
            mask = denominator != 0
            if not mask.any():
                return np.nan
            return np.mean(numerator[mask] / denominator[mask]) * 100
        except:
            return np.nan

    def _calculate_stability_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate prediction stability metrics."""
        try:
            stability_metrics = {}

            # Prediction variance
            if len(y_pred) > 1:
                stability_metrics['prediction_std'] = np.std(y_pred)
                stability_metrics['prediction_variance'] = np.var(y_pred)

            # Error distribution metrics
            errors = y_true - y_pred
            stability_metrics['error_mean'] = np.mean(errors)
            stability_metrics['error_std'] = np.std(errors)
            stability_metrics['error_skewness'] = self._calculate_skewness(errors)

            return stability_metrics

        except Exception:
            return {}

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        try:
            if len(data) < 3:
                return 0.0
            mean_val = np.mean(data)
            std_val = np.std(data)
            if std_val == 0:
                return 0.0
            return np.mean(((data - mean_val) / std_val) ** 3)
        except:
            return 0.0

    def _create_cv_summary(self, cv_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of CV results."""
        try:
            summary = {
                'total_folds': len(cv_results['fold_results']),
                'successful_folds': len([f for f in cv_results['fold_results']
                                       if 'error' not in f]),
                'avg_training_time': np.mean(cv_results['training_times']),
                'avg_prediction_time': np.mean(cv_results['prediction_times']),
                'total_training_time': sum(cv_results['training_times']),
                'total_prediction_time': sum(cv_results['prediction_times'])
            }

            if cv_results['metrics']:
                summary['best_metric'] = max(cv_results['metrics'].items(),
                                           key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)

            return summary

        except Exception:
            return {}

    def _create_wfv_summary(self, wfv_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of walk-forward validation results."""
        try:
            summary = {
                'total_iterations': len(wfv_results['iterations']),
                'successful_iterations': len([i for i in wfv_results['iterations']
                                            if 'error' not in i]),
                'avg_training_time': np.mean(wfv_results['training_times']),
                'avg_prediction_time': np.mean(wfv_results['prediction_times']),
                'total_training_time': sum(wfv_results['training_times']),
                'total_prediction_time': sum(wfv_results['prediction_times'])
            }

            # Calculate performance trend
            if wfv_results['iterations']:
                accuracies = []
                for iteration in wfv_results['iterations']:
                    if 'metrics' in iteration and 'accuracy' in iteration['metrics']:
                        accuracies.append(iteration['metrics']['accuracy'])

                if accuracies:
                    summary['performance_trend'] = {
                        'first_half_avg': np.mean(accuracies[:len(accuracies)//2]),
                        'second_half_avg': np.mean(accuracies[len(accuracies)//2:]),
                        'overall_trend': 'improving' if accuracies[-1] > accuracies[0] else 'declining'
                    }

            return summary

        except Exception:
            return {}
