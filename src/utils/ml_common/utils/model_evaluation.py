"""
Model Evaluation Utilities

This module provides comprehensive utilities for model evaluation with memory-aware operations.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

logger = logging.getLogger(__name__)

class ModelEvaluationUtilities:
    """Model evaluation utilities with memory management."""

    def __init__(self):
        """Initialize model evaluation utilities."""
        self.logger = logger.getChild('ModelEvaluationUtilities')
        self.logger.info("🚀 Initializing ModelEvaluationUtilities")

    def multi_metric_evaluation(
        self,
        y_true: Union[np.ndarray, pd.Series, List],
        y_pred: Union[np.ndarray, pd.Series, List],
        task_type: str = 'regression',
        additional_metrics: Optional[List[Callable]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform comprehensive multi-metric evaluation.

        Args:
            y_true: True target values
            y_pred: Predicted target values
            task_type: Type of ML task ('regression' or 'classification')
            additional_metrics: List of additional metric functions

        Returns:
            Dictionary containing evaluation results
        """
        self.logger.info(f"🔍 Starting multi-metric evaluation for {task_type} task")

        start_time = time.time()

        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Initialize results
        results = {
            'task_type': task_type,
            'n_samples': len(y_true),
            'evaluation_time': None,
            'metrics': {},
            'warnings': []
        }

        try:
            if task_type.lower() == 'regression':
                results['metrics'] = self._evaluate_regression_metrics(y_true, y_pred)
            elif task_type.lower() == 'classification':
                results['metrics'] = self._evaluate_classification_metrics(y_true, y_pred, **kwargs)
            else:
                results['warnings'].append(f"Unknown task type: {task_type}, defaulting to regression")
                results['metrics'] = self._evaluate_regression_metrics(y_true, y_pred)

            # Add additional metrics
            if additional_metrics:
                for i, metric_func in enumerate(additional_metrics):
                    try:
                        metric_name = f'custom_metric_{i}'
                        metric_value = metric_func(y_true, y_pred)
                        results['metrics'][metric_name] = metric_value
                    except Exception as e:
                        results['warnings'].append(f"Failed to calculate custom metric {i}: {e}")

            # Add summary statistics
            results['summary'] = self._calculate_summary_statistics(y_true, y_pred, task_type)

            results['success'] = True

        except Exception as e:
            self.logger.error(f"❌ Multi-metric evaluation failed: {e}")
            results['error'] = str(e)
            results['success'] = False
            results['warnings'].append(f"Evaluation failed: {e}")

        results['evaluation_time'] = time.time() - start_time

        self.logger.info(f"✅ Multi-metric evaluation completed in {results['evaluation_time']:.3f}s")
        return results

    def _evaluate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression evaluation metrics."""
        metrics = {}

        try:
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)

            # Additional regression metrics
            metrics['mean_absolute_percentage_error'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            metrics['median_absolute_error'] = np.median(np.abs(y_true - y_pred))

            # Explained variance
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics['explained_variance'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        except Exception as e:
            self.logger.warning(f"⚠️ Some regression metrics failed: {e}")
            metrics['error'] = str(e)

        return metrics

    def _evaluate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = 'weighted',
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate classification evaluation metrics."""
        metrics = {}

        try:
            # Basic classification metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()

            # Classification report
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            metrics['classification_report'] = report

            # ROC-AUC (if binary classification)
            if len(np.unique(y_true)) == 2:
                try:
                    # For binary classification, try to get probability predictions
                    # If not available, use decision function
                    if hasattr(kwargs.get('model', {}), 'predict_proba'):
                        y_prob = kwargs['model'].predict_proba(y_pred.reshape(-1, 1))[:, 1]
                        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
                    else:
                        # Use predictions directly for AUC calculation
                        metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
                except Exception as e:
                    self.logger.warning(f"⚠️ ROC-AUC calculation failed: {e}")
                    metrics['roc_auc'] = None

        except Exception as e:
            self.logger.warning(f"⚠️ Some classification metrics failed: {e}")
            metrics['error'] = str(e)

        return metrics

    def _calculate_summary_statistics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: str
    ) -> Dict[str, Any]:
        """Calculate summary statistics for evaluation."""
        summary = {}

        try:
            summary['target_stats'] = {
                'mean': float(np.mean(y_true)),
                'std': float(np.std(y_true)),
                'min': float(np.min(y_true)),
                'max': float(np.max(y_true)),
                'median': float(np.median(y_true))
            }

            summary['prediction_stats'] = {
                'mean': float(np.mean(y_pred)),
                'std': float(np.std(y_pred)),
                'min': float(np.min(y_pred)),
                'max': float(np.max(y_pred)),
                'median': float(np.median(y_pred))
            }

            summary['residual_stats'] = {
                'mean': float(np.mean(y_true - y_pred)),
                'std': float(np.std(y_true - y_pred)),
                'min': float(np.min(y_true - y_pred)),
                'max': float(np.max(y_true - y_pred)),
                'median': float(np.median(y_true - y_pred))
            }

            if task_type.lower() == 'regression':
                summary['performance_category'] = self._categorize_regression_performance(
                    summary['target_stats']['std'],
                    summary['residual_stats']['std']
                )

        except Exception as e:
            self.logger.warning(f"⚠️ Summary statistics calculation failed: {e}")
            summary['error'] = str(e)

        return summary

    def _categorize_regression_performance(self, target_std: float, residual_std: float) -> str:
        """Categorize regression performance based on residual vs target variability."""
        if residual_std == 0:
            return 'perfect'

        ratio = residual_std / target_std

        if ratio < 0.1:
            return 'excellent'
        elif ratio < 0.25:
            return 'good'
        elif ratio < 0.5:
            return 'fair'
        elif ratio < 0.75:
            return 'poor'
        else:
            return 'very_poor'

    def evaluate_model_stability(
        self,
        models: List[Any],
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series, List],
        task_type: str = 'regression',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate model stability across multiple model instances.

        Args:
            models: List of trained models
            X_test: Test features
            y_test: Test targets
            task_type: Type of ML task

        Returns:
            Dictionary containing stability evaluation results
        """
        self.logger.info(f"🔍 Evaluating stability for {len(models)} models")

        start_time = time.time()

        predictions = []
        metrics = []

        for i, model in enumerate(models):
            try:
                if hasattr(model, 'predict'):
                    y_pred = model.predict(X_test)
                else:
                    # Assume model is a function
                    y_pred = model(X_test)

                predictions.append(y_pred)

                # Calculate metrics for this model
                model_metrics = self.multi_metric_evaluation(
                    y_test, y_pred, task_type=task_type, **kwargs
                )
                metrics.append(model_metrics)

            except Exception as e:
                self.logger.error(f"❌ Model {i} prediction failed: {e}")
                predictions.append(None)
                metrics.append({'error': str(e), 'success': False})

        # Calculate stability metrics
        stability_results = self._calculate_stability_metrics(predictions, metrics, task_type)

        evaluation_time = time.time() - start_time

        result = {
            'n_models': len(models),
            'predictions': predictions,
            'individual_metrics': metrics,
            'stability_metrics': stability_results,
            'evaluation_time': evaluation_time,
            'success': stability_results['success']
        }

        self.logger.info(f"✅ Model stability evaluation completed in {evaluation_time:.3f}s")
        return result

    def _calculate_stability_metrics(
        self,
        predictions: List[np.ndarray],
        metrics: List[Dict],
        task_type: str
    ) -> Dict[str, Any]:
        """Calculate stability metrics from multiple predictions."""
        stability = {'success': False}

        try:
            # Filter successful predictions
            valid_predictions = [p for p in predictions if p is not None]
            valid_metrics = [m for m in metrics if m.get('success', False)]

            if not valid_predictions:
                stability['error'] = 'No valid predictions'
                return stability

            # Convert to numpy array
            pred_array = np.array(valid_predictions)

            # Calculate prediction variability
            stability['prediction_std'] = float(np.std(pred_array, axis=0).mean())
            stability['prediction_mean'] = float(np.mean(pred_array, axis=0).mean())
            stability['coefficient_of_variation'] = (
                stability['prediction_std'] / abs(stability['prediction_mean'])
                if stability['prediction_mean'] != 0 else float('inf')
            )

            # Calculate metric stability
            if valid_metrics and len(valid_metrics) > 1:
                metric_values = {}
                for metric_dict in valid_metrics:
                    for key, value in metric_dict.get('metrics', {}).items():
                        if isinstance(value, (int, float)) and not np.isnan(value):
                            if key not in metric_values:
                                metric_values[key] = []
                            metric_values[key].append(value)

                stability['metric_stability'] = {}
                for metric_name, values in metric_values.items():
                    if len(values) > 1:
                        stability['metric_stability'][metric_name] = {
                            'std': float(np.std(values)),
                            'mean': float(np.mean(values)),
                            'cv': float(np.std(values) / abs(np.mean(values))) if np.mean(values) != 0 else float('inf')
                        }

            stability['success'] = True

        except Exception as e:
            self.logger.error(f"❌ Stability calculation failed: {e}")
            stability['error'] = str(e)

        return stability

    def cross_validate_and_evaluate(
        self,
        model_function: Callable,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, List],
        cv_splits: int = 5,
        task_type: str = 'regression',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform cross-validation and comprehensive evaluation.

        Args:
            model_function: Function that returns a fitted model
            X: Feature matrix
            y: Target vector
            cv_splits: Number of cross-validation splits
            task_type: Type of ML task

        Returns:
            Dictionary containing cross-validation and evaluation results
        """
        self.logger.info(f"🔍 Starting cross-validation evaluation with {cv_splits} splits")

        start_time = time.time()

        from sklearn.model_selection import KFold

        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

        cv_results = []
        all_predictions = []
        all_actuals = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            self.logger.debug(f"📈 Processing CV fold {fold + 1}/{cv_splits}")

            X_train, X_test = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx], \
                             X.iloc[test_idx] if hasattr(X, 'iloc') else X[test_idx]
            y_train, y_test = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx], \
                             y.iloc[test_idx] if hasattr(y, 'iloc') else y[test_idx]

            try:
                # Train model
                model = model_function(X_train, y_train, **kwargs)

                # Make predictions
                if hasattr(model, 'predict'):
                    y_pred = model.predict(X_test)
                else:
                    y_pred = model(X_test)

                # Evaluate fold
                fold_evaluation = self.multi_metric_evaluation(
                    y_test, y_pred, task_type=task_type, **kwargs
                )

                fold_result = {
                    'fold': fold + 1,
                    'evaluation': fold_evaluation,
                    'train_size': len(X_train),
                    'test_size': len(X_test)
                }
                cv_results.append(fold_result)

                # Collect predictions and actuals
                all_predictions.extend(y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred))
                all_actuals.extend(y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test))

            except Exception as e:
                self.logger.error(f"❌ CV fold {fold + 1} failed: {e}")
                fold_result = {
                    'fold': fold + 1,
                    'error': str(e),
                    'success': False
                }
                cv_results.append(fold_result)

        # Overall evaluation
        overall_evaluation = None
        if all_predictions and all_actuals:
            overall_evaluation = self.multi_metric_evaluation(
                all_actuals, all_predictions, task_type=task_type, **kwargs
            )

        evaluation_time = time.time() - start_time

        result = {
            'cv_results': cv_results,
            'overall_evaluation': overall_evaluation,
            'cv_splits': cv_splits,
            'task_type': task_type,
            'total_samples': len(X),
            'evaluation_time': evaluation_time,
            'success': overall_evaluation is not None and overall_evaluation.get('success', False)
        }

        self.logger.info(f"✅ Cross-validation evaluation completed in {evaluation_time:.3f}s")
        return result


# Global instance for easy access
_evaluation_instance = None

def get_model_evaluation_utilities() -> ModelEvaluationUtilities:
    """Get global model evaluation utilities instance."""
    global _evaluation_instance
    if _evaluation_instance is None:
        _evaluation_instance = ModelEvaluationUtilities()
    return _evaluation_instance

# Export key classes and functions
__all__ = ['ModelEvaluationUtilities', 'get_model_evaluation_utilities']
