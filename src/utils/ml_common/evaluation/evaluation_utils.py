"""
Evaluation Utilities

Common evaluation patterns shared across all training modules.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, log_loss, roc_auc_score
)
import logging

logger = logging.getLogger(__name__)


class EvaluationUtils:
    """Common evaluation utilities."""
    
    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_pred_proba: Optional[np.ndarray] = None,
        metrics: List[str] = None,
        is_classification: bool = True
    ) -> Dict[str, float]:
        """
        Calculate common metrics for model evaluation.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            y_pred_proba: Predicted probabilities (for classification)
            metrics: List of metrics to calculate
            is_classification: Whether this is a classification task
            
        Returns:
            Dictionary containing calculated metrics
        """
        if metrics is None:
            if is_classification:
                metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            else:
                metrics = ['mse', 'mae', 'r2', 'mape', 'smape']
        
        calculated_metrics = {}
        
        if is_classification:
            # Classification metrics
            if 'accuracy' in metrics:
                calculated_metrics['accuracy'] = accuracy_score(y_true, y_pred)
            
            if 'precision' in metrics:
                calculated_metrics['precision'] = precision_score(
                    y_true, y_pred, average='weighted', zero_division=0
                )
            
            if 'recall' in metrics:
                calculated_metrics['recall'] = recall_score(
                    y_true, y_pred, average='weighted', zero_division=0
                )
            
            if 'f1_score' in metrics:
                calculated_metrics['f1_score'] = f1_score(
                    y_true, y_pred, average='weighted', zero_division=0
                )
            
            if 'classification_report' in metrics:
                calculated_metrics['classification_report'] = classification_report(
                    y_true, y_pred, output_dict=True
                )
            
            if 'confusion_matrix' in metrics:
                calculated_metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
            
            if y_pred_proba is not None:
                if 'log_loss' in metrics:
                    calculated_metrics['log_loss'] = log_loss(y_true, y_pred_proba)
                
                if 'roc_auc' in metrics and len(np.unique(y_true)) == 2:
                    calculated_metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        else:
            # Regression metrics
            if 'mse' in metrics:
                calculated_metrics['mse'] = mean_squared_error(y_true, y_pred)
            
            if 'rmse' in metrics:
                calculated_metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            
            if 'mae' in metrics:
                calculated_metrics['mae'] = mean_absolute_error(y_true, y_pred)
            
            if 'r2' in metrics:
                calculated_metrics['r2'] = r2_score(y_true, y_pred)
            
            if 'mape' in metrics:
                # Avoid division by zero
                mask = y_true != 0
                if np.any(mask):
                    calculated_metrics['mape'] = np.mean(
                        np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
                    ) * 100
                else:
                    calculated_metrics['mape'] = 0.0
            
            if 'smape' in metrics:
                calculated_metrics['smape'] = np.mean(
                    2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))
                ) * 100
            
            if 'explained_variance' in metrics:
                calculated_metrics['explained_variance'] = 1 - np.var(y_true - y_pred) / np.var(y_true)
        
        return calculated_metrics
    
    @staticmethod
    def evaluate_model_performance(
        model: Any, 
        X: np.ndarray, 
        y: np.ndarray, 
        metrics: List[str] = None,
        is_classification: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model performance with common metrics.
        
        Args:
            model: Trained model
            X: Input features
            y: True target values
            metrics: List of metrics to calculate
            is_classification: Whether this is a classification task
            
        Returns:
            Dictionary containing calculated metrics
        """
        # Make predictions
        y_pred = model.predict(X)
        
        # Get probabilities if available and needed
        y_pred_proba = None
        if is_classification and hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X)
        
        # Calculate metrics
        return EvaluationUtils.calculate_metrics(
            y, y_pred, y_pred_proba, metrics, is_classification
        )
    
    @staticmethod
    def evaluate_ensemble_performance(
        ensemble: Any, 
        X: np.ndarray, 
        y: np.ndarray, 
        metrics: List[str] = None,
        is_classification: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate ensemble performance with common metrics.
        
        Args:
            ensemble: Trained ensemble model
            X: Input features
            y: True target values
            metrics: List of metrics to calculate
            is_classification: Whether this is a classification task
            
        Returns:
            Dictionary containing calculated metrics
        """
        # Make predictions
        y_pred = ensemble.predict(X)
        
        # Get probabilities if available and needed
        y_pred_proba = None
        if is_classification and hasattr(ensemble, 'predict_proba'):
            y_pred_proba = ensemble.predict_proba(X)
        
        # Calculate metrics
        return EvaluationUtils.calculate_metrics(
            y, y_pred, y_pred_proba, metrics, is_classification
        )
    
    @staticmethod
    def evaluate_regime_performance(
        models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        metrics: List[str] = None,
        is_classification: bool = True
    ) -> Dict[int, Dict[str, Dict[str, float]]]:
        """
        Evaluate model performance per regime.
        
        Args:
            models: Dictionary of trained models
            X: Input features
            y: True target values
            regime_labels: Array of regime labels
            metrics: List of metrics to calculate
            is_classification: Whether this is a classification task
            
        Returns:
            Dictionary containing evaluation results per regime per model
        """
        evaluation_results = {}
        
        for regime in np.unique(regime_labels):
            regime_mask = regime_labels == regime
            regime_X = X[regime_mask]
            regime_y = y[regime_mask]
            
            regime_evaluation = {}
            
            for model_name, model in models.items():
                try:
                    metrics_dict = EvaluationUtils.evaluate_model_performance(
                        model, regime_X, regime_y, metrics, is_classification
                    )
                    regime_evaluation[model_name] = metrics_dict
                except Exception as e:
                    logger.warning(f"⚠️ Failed to evaluate {model_name} for regime {regime}: {e}")
                    regime_evaluation[model_name] = {'error': str(e)}
            
            evaluation_results[regime] = regime_evaluation
        
        return evaluation_results
    
    @staticmethod
    def analyze_regime_distribution(
        y_train: np.ndarray, 
        y_test: np.ndarray, 
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze regime distribution and model performance per regime.
        
        Args:
            y_train: Training regime labels
            y_test: Test regime labels
            results: Training results containing model performance
            
        Returns:
            Dictionary containing regime analysis results
        """
        # Overall regime distribution
        unique_regimes, train_counts = np.unique(y_train, return_counts=True)
        _, test_counts = np.unique(y_test, return_counts=True)
        
        regime_analysis = {
            'n_regimes': len(unique_regimes),
            'regime_distribution_train': dict(zip(unique_regimes, train_counts)),
            'regime_distribution_test': dict(zip(unique_regimes, test_counts)),
            'regime_balance_train': np.std(train_counts) / np.mean(train_counts) if len(train_counts) > 1 else 0.0,
            'regime_balance_test': np.std(test_counts) / np.mean(test_counts) if len(test_counts) > 1 else 0.0
        }
        
        # Model performance per regime
        for model_name, metrics in results.get('performance', {}).items():
            if 'confusion_matrix' in metrics:
                cm = np.array(metrics['confusion_matrix'])
                regime_precision = np.diag(cm) / np.sum(cm, axis=0)
                regime_recall = np.diag(cm) / np.sum(cm, axis=1)
                regime_f1 = 2 * (regime_precision * regime_recall) / (regime_precision + regime_recall)
                
                regime_analysis[f'{model_name}_regime_performance'] = {
                    'precision_per_regime': regime_precision.tolist(),
                    'recall_per_regime': regime_recall.tolist(),
                    'f1_per_regime': regime_f1.tolist()
                }
        
        return regime_analysis
    
    @staticmethod
    def get_best_model(results: Dict[str, Any], metric: str = 'accuracy') -> Optional[str]:
        """
        Get the best model based on a specific metric.
        
        Args:
            results: Training results containing model performance
            metric: Metric to use for comparison
            
        Returns:
            Name of the best model, or None if not found
        """
        best_model = None
        best_score = -np.inf
        
        for model_name, metrics in results.get('performance', {}).items():
            if metric in metrics and metrics[metric] > best_score:
                best_score = metrics[metric]
                best_model = model_name
        
        return best_model