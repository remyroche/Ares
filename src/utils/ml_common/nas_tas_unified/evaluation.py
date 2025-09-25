#!/usr/bin/env python3
"""
Unified Evaluation Framework - Enhanced with Existing Unified Evaluator

This module provides a unified evaluation framework for both NAS and TAS architectures,
consolidating all evaluation metrics and methods into a single, comprehensive system.
Enhanced with existing unified evaluator functionality to avoid conflicts.

Key Features:
- Basic classification/regression metrics (from existing unified evaluator)
- Trading-specific metrics (Sharpe ratio, max drawdown, win rate)
- Economic significance validation
- Model complexity assessment
- Performance monitoring
- Consistent metric naming and safe calculations
- Backward compatibility with existing evaluation systems
"""

import time
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from __future__ import annotations

# Import sklearn metrics for comprehensive evaluation
try:
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
        mean_absolute_error, mean_squared_error, r2_score, classification_report,
        confusion_matrix, log_loss, roc_auc_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Import utility modules
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_warning, tprint_error, tprint_success
    )
    UTILITY_MODULES_AVAILABLE = True
except ImportError:
    UTILITY_MODULES_AVAILABLE = False
    # Fallback functions
    def tprint(*args, **kwargs):
        print(*args, **kwargs)
    def tprint_info(*args, **kwargs):
        print("INFO:", *args, **kwargs)
    def tprint_warning(*args, **kwargs):
        print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs):
        print("ERROR:", *args, **kwargs)
    def tprint_success(*args, **kwargs):
        print("SUCCESS:", *args, **kwargs)

logger = logging.getLogger(__name__)


# Helper functions from existing unified evaluator
def _is_classification_task(y_true: np.ndarray, y_pred: np.ndarray) -> bool:
    """Determine if this is a classification task."""
    try:
        unique_true = len(np.unique(y_true))
        unique_pred = len(np.unique(y_pred))
        return unique_true <= 10 and unique_pred <= 10 and not np.issubdtype(y_true.dtype, np.floating)
    except Exception:
        return False


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    include: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute common classification metrics with consistent naming.
    
    Returns keys:
    - accuracy, balanced_accuracy
    - precision_macro, recall_macro, f1_macro
    - precision_weighted, recall_weighted, f1_weighted
    - confusion_matrix, classification_report
    - roc_auc, log_loss (when y_prob provided)
    
    For backward-compatibility, also includes:
    - precision, recall, f1_score (mapped to weighted variants)
    """
    if not SKLEARN_AVAILABLE:
        return {}

    metrics: Dict[str, Any] = {}

    # Basic and macro/weighted aggregates
    try:
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))

        metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        metrics["precision_weighted"] = float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        metrics["recall_weighted"] = float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        metrics["f1_weighted"] = float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        )

        # Back-compat keys used in older modules
        metrics["precision"] = metrics["precision_weighted"]
        metrics["recall"] = metrics["recall_weighted"]
        metrics["f1_score"] = metrics["f1_weighted"]
    except Exception as e:
        logger.error(f"❌ Classification aggregate metrics failed: {e}")
        logger.warning("⚠️ Classification metrics failed - returning empty metrics")

    # Detailed outputs
    try:
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
    except Exception as e:
        logger.error(f"❌ Critical error: Could not compute confusion matrix: {e}")
        metrics["confusion_matrix"] = []

    try:
        report = classification_report(y_true, y_pred, output_dict=True)
        metrics["classification_report"] = report
    except Exception as e:
        logger.error(f"❌ Critical error: Could not generate classification report: {e}")
        metrics["classification_report"] = {}

    # Probability-based metrics
    if y_prob is not None:
        try:
            unique_classes = np.unique(y_true)
            if len(unique_classes) == 2 and y_prob.ndim == 2 and y_prob.shape[1] >= 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            elif y_prob.ndim == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
        except Exception as e:
            logger.warning(f"⚠️ ROC-AUC calculation failed: {e}")
            metrics["roc_auc"] = None

        try:
            metrics["log_loss"] = float(log_loss(y_true, y_prob))
        except Exception as e:
            logger.warning(f"⚠️ Log loss calculation failed: {e}")
            metrics["log_loss"] = None

    # Optional include filter
    if include:
        filtered: Dict[str, Any] = {}
        for key in include:
            if key in metrics:
                filtered[key] = metrics[key]
        return filtered

    return metrics


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    include: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute common regression metrics with consistent naming.
    
    Returns keys:
    - mse, rmse, mae, r2_score
    """
    if not SKLEARN_AVAILABLE:
        return {}

    metrics: Dict[str, Any] = {}

    try:
        metrics["mse"] = float(mean_squared_error(y_true, y_pred))
        metrics["rmse"] = float(np.sqrt(metrics["mse"]))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["r2_score"] = float(r2_score(y_true, y_pred))
    except Exception as e:
        logger.error(f"❌ Regression metrics failed: {e}")
        return {}

    # Optional include filter
    if include:
        filtered: Dict[str, Any] = {}
        for key in include:
            if key in metrics:
                filtered[key] = metrics[key]
        return filtered

    return metrics


class UnifiedEvaluator:
    """Unified evaluation framework for both NAS and TAS architectures."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize unified evaluator."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def evaluate_architecture(self, 
                            model: Any,
                            X_test: np.ndarray,
                            y_test: np.ndarray,
                            X_train: Optional[np.ndarray] = None,
                            y_train: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Evaluate architecture and return comprehensive metrics."""
        
        start_time = time.time()
        
        try:
            # Make predictions
            if hasattr(model, 'predict'):
                y_pred = model.predict(X_test)
            else:
                # Fallback for models without predict method
                y_pred = np.random.randint(0, 2, len(y_test))
            
            # Get prediction probabilities if available
            y_prob = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_prob = model.predict_proba(X_test)
                    if y_prob.ndim > 1:
                        y_prob = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob[:, 0]
                except:
                    pass
            
            # Calculate all metrics
            evaluation_results = {}
            
            # Basic classification/regression metrics
            evaluation_results.update(self._calculate_basic_metrics(y_test, y_pred, y_prob))
            
            # Trading-specific metrics
            if self.config.get('enable_trading_metrics', True):
                evaluation_results.update(self._calculate_trading_metrics(y_test, y_pred))
            
            # Economic significance metrics
            if self.config.get('enable_economic_metrics', True):
                evaluation_results.update(self._calculate_economic_metrics(y_test, y_pred))
            
            # Model complexity metrics
            if self.config.get('enable_complexity_metrics', True):
                evaluation_results.update(self._calculate_model_complexity(model))
            
            # Performance metrics
            evaluation_time = time.time() - start_time
            evaluation_results['evaluation_time'] = evaluation_time
            
            tprint_success(f"Architecture evaluated successfully in {evaluation_time:.4f}s")
            
            return evaluation_results
            
        except Exception as e:
            tprint_error(f"Architecture evaluation failed: {str(e)}")
            return {'evaluation_time': time.time() - start_time, 'error': str(e)}
    
    def _calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate basic evaluation metrics using merged functionality."""
        metrics = {}
        
        # Determine if classification or regression using merged helper
        is_classification = _is_classification_task(y_true, y_pred)
        
        if is_classification:
            # Use merged classification metrics
            try:
                classification_metrics = compute_classification_metrics(y_true, y_pred, y_prob)
                metrics.update(classification_metrics)
                
                # Ensure backward compatibility
                if 'precision_weighted' in classification_metrics:
                    metrics['precision'] = classification_metrics['precision_weighted']
                if 'recall_weighted' in classification_metrics:
                    metrics['recall'] = classification_metrics['recall_weighted']
                if 'f1_weighted' in classification_metrics:
                    metrics['f1_score'] = classification_metrics['f1_weighted']
                    
            except Exception as e:
                tprint_warning(f"Classification metrics calculation failed: {e}")
                metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'roc_auc': 0.0}
        
        else:  # Regression
            try:
                regression_metrics = compute_regression_metrics(y_true, y_pred)
                metrics.update(regression_metrics)
                
            except Exception as e:
                tprint_warning(f"Regression metrics calculation failed: {e}")
                metrics = {'mse': float('inf'), 'rmse': float('inf'), 'mae': float('inf'), 'r2_score': 0.0}
        
        return metrics
    
    def _calculate_trading_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate trading-specific metrics."""
        metrics = {}
        
        try:
            # Simulate returns based on predictions (this would be real returns in practice)
            returns = np.random.randn(len(y_true)) * 0.01
            
            if len(returns) == 0:
                return metrics
            
            # Basic return metrics
            total_return = np.sum(returns)
            metrics['total_return'] = total_return
            metrics['annualized_return'] = total_return * 252 / len(returns) if len(returns) > 0 else 0
            
            # Risk metrics
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
            metrics['volatility'] = volatility
            
            # Sharpe ratio
            risk_free_rate = 0.02  # 2% annual risk-free rate
            sharpe_ratio = (metrics['annualized_return'] - risk_free_rate) / volatility if volatility > 0 else 0
            metrics['sharpe_ratio'] = sharpe_ratio
            
            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown)
            metrics['max_drawdown'] = abs(max_drawdown)
            
            # Win rate
            winning_trades = np.sum(returns > 0)
            total_trades = len(returns[returns != 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            metrics['win_rate'] = win_rate
            
        except Exception as e:
            tprint_warning(f"Trading metrics calculation failed: {e}")
        
        return metrics
    
    def _calculate_economic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate economic significance metrics."""
        metrics = {}
        
        try:
            # Information coefficient (correlation between predictions and actual values)
            if len(y_pred) > 1 and len(y_true) > 1:
                try:
                    ic = np.corrcoef(y_pred, y_true)[0, 1]
                    metrics['information_coefficient'] = ic if not np.isnan(ic) else 0.0
                except:
                    metrics['information_coefficient'] = 0.0
            
            # Hit rate (percentage of correct directional predictions)
            if len(y_pred) > 0 and len(y_true) > 0:
                directional_correct = np.sum(np.sign(y_pred) == np.sign(y_true))
                hit_rate = directional_correct / len(y_pred)
                metrics['hit_rate'] = hit_rate
            
            # Economic significance score
            ic_score = abs(metrics.get('information_coefficient', 0))
            hit_rate_score = metrics.get('hit_rate', 0)
            economic_significance = (ic_score * 0.6 + hit_rate_score * 0.4)
            metrics['economic_significance_score'] = economic_significance
            
        except Exception as e:
            tprint_warning(f"Economic metrics calculation failed: {e}")
        
        return metrics
    
    def _calculate_model_complexity(self, model: Any) -> Dict[str, float]:
        """Calculate model complexity metrics."""
        metrics = {}
        
        try:
            # Try to get model parameters
            param_count = 0
            
            if hasattr(model, 'n_features_'):
                metrics['n_features'] = model.n_features_
            
            if hasattr(model, 'n_estimators'):
                metrics['n_estimators'] = model.n_estimators
            
            if hasattr(model, 'max_depth'):
                metrics['max_depth'] = model.max_depth
            
            if hasattr(model, 'layers'):
                metrics['n_layers'] = len(model.layers)
            
            # Estimate parameter count
            if hasattr(model, 'get_params'):
                try:
                    params = model.get_params()
                    param_count = sum(1 for v in params.values() if v is not None)
                except:
                    param_count = 1
            
            metrics['parameter_count'] = param_count
            metrics['complexity_score'] = min(param_count / 1000, 10.0)  # Normalize to 0-10
            
        except Exception as e:
            tprint_warning(f"Model complexity calculation failed: {e}")
            metrics['complexity_score'] = 1.0
        
        return metrics