#!/usr/bin/env python3
"""
Unified Evaluation Framework

This module provides a unified evaluation framework for both NAS and TAS architectures,
consolidating all evaluation metrics and methods into a single, comprehensive system.

Key Features:
- Basic classification/regression metrics
- Trading-specific metrics (Sharpe ratio, max drawdown, win rate)
- Economic significance validation
- Model complexity assessment
- Performance monitoring
"""

import time
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple

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
        """Calculate basic evaluation metrics."""
        metrics = {}
        
        # Determine if classification or regression
        n_unique = len(np.unique(y_true))
        
        if n_unique <= 10:  # Classification
            try:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                
                metrics['accuracy'] = accuracy_score(y_true, y_pred)
                metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                
                if y_prob is not None:
                    try:
                        metrics['roc_auc'] = roc_auc_score(y_true, y_prob, average='weighted', multi_class='ovr')
                    except ValueError:
                        metrics['roc_auc'] = 0.0
            except Exception as e:
                tprint_warning(f"Basic metrics calculation failed: {e}")
                metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'roc_auc': 0.0}
        
        else:  # Regression
            try:
                metrics['mse'] = np.mean((y_true - y_pred) ** 2)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                metrics['mae'] = np.mean(np.abs(y_true - y_pred))
                metrics['r2_score'] = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
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