"""
Bayesian Optimization for Attention Network Parameters
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import logging
import time

from src.utils.logger import system_logger

try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler
    from optuna.pruners import MedianPruner, HyperbandPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    from sklearn.metrics import make_scorer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class AttentionOptimizationConfig:
    """Configuration for Attention Network Bayesian optimization."""
    
    # Optimization parameters
    n_trials: int = 100
    timeout_seconds: Optional[int] = None
    n_startup_trials: int = 10
    n_warmup_steps: int = 5
    
    # Attention network parameter ranges
    attention_dim_range: Tuple[int, int] = (32, 256)
    num_heads_range: Tuple[int, int] = (4, 16)
    dropout_rate_range: Tuple[float, float] = (0.1, 0.5)
    learning_rate_range: Tuple[float, float] = (0.0001, 0.01)
    regularization_range: Tuple[float, float] = (0.001, 0.1)
    
    # Hidden layer configuration
    hidden_layers_options: List[List[int]] = None
    activation_options: List[str] = None
    
    # Training parameters
    batch_size_range: Tuple[int, int] = (16, 128)
    epochs_range: Tuple[int, int] = (50, 200)
    early_stopping_patience_range: Tuple[int, int] = (5, 20)
    
    # Cross-validation
    cv_folds: int = 5
    scoring: str = 'neg_mean_squared_error'
    
    # Early stopping
    early_stopping_patience: int = 20
    improvement_threshold: float = 0.001
    
    def __post_init__(self):
        if self.hidden_layers_options is None:
            self.hidden_layers_options = [
                [64, 32],
                [128, 64, 32],
                [256, 128, 64],
                [128, 64],
                [64, 32, 16]
            ]
        
        if self.activation_options is None:
            self.activation_options = ['relu', 'tanh', 'sigmoid', 'gelu']


class AttentionBayesianOptimizer:
    """Bayesian optimization for Attention Network parameters."""
    
    def __init__(self, config: AttentionOptimizationConfig):
        """Initialize Attention Bayesian optimizer."""
        self.config = config
        self.logger = system_logger.getChild('AttentionBayesianOptimizer')
        
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available. Install with: pip install optuna")
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn not available. Install with: pip install scikit-learn")
        
        # Initialize study
        self.study = None
        self.best_params = None
        self.best_score = None
        self.optimization_history = []
        
    def optimize(self, X: np.ndarray, y: np.ndarray, 
                 objective_func: Optional[Callable] = None) -> Dict[str, Any]:
        """Optimize Attention Network parameters using Bayesian optimization."""
        self.logger.info("Starting Attention Network Bayesian optimization")
        
        # Create study
        sampler = TPESampler(
            n_startup_trials=self.config.n_startup_trials,
            n_warmup_steps=self.config.n_warmup_steps
        )
        
        pruner = MedianPruner(
            n_startup_trials=self.config.n_startup_trials,
            n_warmup_steps=self.config.n_warmup_steps
        )
        
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner
        )
        
        # Define objective function
        if objective_func is None:
            objective_func = self._default_objective
        
        # Optimize
        start_time = time.time()
        
        self.study.optimize(
            lambda trial: objective_func(trial, X, y),
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_seconds
        )
        
        optimization_time = time.time() - start_time
        
        # Store results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        self.logger.info(f"Attention optimization completed in {optimization_time:.2f} seconds")
        self.logger.info(f"Best score: {self.best_score:.4f}")
        self.logger.info(f"Best parameters: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_time': optimization_time,
            'n_trials': len(self.study.trials),
            'study': self.study
        }
    
    def _default_objective(self, trial, X: np.ndarray, y: np.ndarray) -> float:
        """Default objective function for Attention Network optimization."""
        try:
            # Sample attention network parameters
            attention_dim = trial.suggest_int('attention_dim', *self.config.attention_dim_range)
            num_heads = trial.suggest_int('num_heads', *self.config.num_heads_range)
            dropout_rate = trial.suggest_float('dropout_rate', *self.config.dropout_rate_range)
            learning_rate = trial.suggest_float('learning_rate', *self.config.learning_rate_range)
            regularization = trial.suggest_float('regularization', *self.config.regularization_range)
            
            # Sample hidden layer configuration
            hidden_layers = trial.suggest_categorical('hidden_layers', self.config.hidden_layers_options)
            activation = trial.suggest_categorical('activation', self.config.activation_options)
            
            # Sample training parameters
            batch_size = trial.suggest_int('batch_size', *self.config.batch_size_range)
            epochs = trial.suggest_int('epochs', *self.config.epochs_range)
            early_stopping_patience = trial.suggest_int('early_stopping_patience', *self.config.early_stopping_patience_range)
            
            # Create attention configuration
            attention_config = {
                'attention_dim': attention_dim,
                'num_heads': num_heads,
                'dropout_rate': dropout_rate,
                'learning_rate': learning_rate,
                'regularization': regularization,
                'hidden_layers': hidden_layers,
                'activation': activation,
                'batch_size': batch_size,
                'epochs': epochs,
                'early_stopping_patience': early_stopping_patience
            }
            
            # Evaluate attention network
            score = self._evaluate_attention_network(X, y, attention_config)
            
            return score
            
        except Exception as e:
            self.logger.warning(f"Trial failed: {e}")
            return -np.inf
    
    def _evaluate_attention_network(self, X: np.ndarray, y: np.ndarray, config: Dict[str, Any]) -> float:
        """Evaluate attention network with given configuration."""
        try:
            # Import attention mechanisms
            from src.training.steps.model_training.attention_mechanisms import TreeAttentionMechanism, AttentionConfig
            
            # Create attention configuration
            attention_config = AttentionConfig(
                attention_dim=config['attention_dim'],
                num_heads=config['num_heads'],
                dropout_rate=config['dropout_rate'],
                learning_rate=config['learning_rate'],
                regularization=config['regularization'],
                hidden_layers=config['hidden_layers'],
                activation=config['activation'],
                batch_size=config['batch_size'],
                epochs=config['epochs'],
                early_stopping_patience=config['early_stopping_patience']
            )
            
            # Create attention mechanism
            attention_mechanism = TreeAttentionMechanism(attention_config)
            
            # Fit attention mechanism
            attention_mechanism.fit(X, y)
            
            # Calculate evaluation metrics
            metrics = self._calculate_attention_metrics(attention_mechanism, X, y)
            
            # Combine metrics into single score
            score = self._combine_metrics(metrics)
            
            return score
            
        except Exception as e:
            self.logger.warning(f"Attention network evaluation failed: {e}")
            return -np.inf
    
    def _calculate_attention_metrics(self, attention_mechanism, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate attention network evaluation metrics."""
        metrics = {}
        
        try:
            # Get predictions
            predictions = attention_mechanism.predict(X, np.zeros((len(X), 1)))
            
            # Calculate prediction metrics
            mse = np.mean((predictions - y) ** 2)
            mae = np.mean(np.abs(predictions - y))
            r2 = 1 - (np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2))
            
            metrics['mse'] = mse
            metrics['mae'] = mae
            metrics['r2'] = r2
            
            # Calculate attention quality metrics
            attention_weights = attention_mechanism.get_attention_weights(X)
            
            # Attention weight diversity
            weight_entropy = self._calculate_attention_entropy(attention_weights)
            metrics['attention_entropy'] = weight_entropy
            
            # Attention weight stability
            weight_stability = self._calculate_attention_stability(attention_weights)
            metrics['attention_stability'] = weight_stability
            
            # Feature importance quality
            feature_importance = attention_mechanism.get_feature_importance(X)
            importance_quality = self._calculate_importance_quality(feature_importance)
            metrics['importance_quality'] = importance_quality
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate attention metrics: {e}")
            return {'mse': np.inf, 'mae': np.inf, 'r2': -np.inf}
    
    def _calculate_attention_entropy(self, attention_weights: np.ndarray) -> float:
        """Calculate attention weight entropy."""
        try:
            # Normalize weights
            normalized_weights = attention_weights / np.sum(attention_weights, axis=1, keepdims=True)
            
            # Calculate entropy
            entropy = -np.sum(normalized_weights * np.log(normalized_weights + 1e-8), axis=1)
            mean_entropy = np.mean(entropy)
            
            return mean_entropy
            
        except:
            return 0.0
    
    def _calculate_attention_stability(self, attention_weights: np.ndarray) -> float:
        """Calculate attention weight stability."""
        try:
            # Calculate weight variance across samples
            weight_variance = np.var(attention_weights, axis=0)
            mean_variance = np.mean(weight_variance)
            
            # Stability is inverse of variance
            stability = 1.0 / (1.0 + mean_variance)
            
            return stability
            
        except:
            return 0.0
    
    def _calculate_importance_quality(self, feature_importance: np.ndarray) -> float:
        """Calculate feature importance quality."""
        try:
            # Calculate importance diversity
            normalized_importance = feature_importance / np.sum(feature_importance)
            entropy = -np.sum(normalized_importance * np.log(normalized_importance + 1e-8))
            
            # Calculate importance concentration
            max_importance = np.max(feature_importance)
            concentration = max_importance / np.sum(feature_importance)
            
            # Quality is balance between diversity and concentration
            quality = entropy * (1 - concentration)
            
            return quality
            
        except:
            return 0.0
    
    def _combine_metrics(self, metrics: Dict[str, float]) -> float:
        """Combine multiple metrics into single score."""
        # Weighted combination of metrics
        weights = {
            'r2': 0.4,  # Prediction accuracy
            'attention_entropy': 0.2,  # Attention diversity
            'attention_stability': 0.2,  # Attention stability
            'importance_quality': 0.2  # Feature importance quality
        }
        
        score = 0.0
        for metric, weight in weights.items():
            if metric in metrics:
                score += weight * metrics[metric]
        
        return score
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary."""
        if self.study is None:
            return {}
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(self.study.trials),
            'optimization_history': [
                {
                    'trial': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': trial.state.name
                }
                for trial in self.study.trials
            ],
            'parameter_importance': self._calculate_parameter_importance()
        }
    
    def _calculate_parameter_importance(self) -> Dict[str, float]:
        """Calculate parameter importance."""
        if self.study is None:
            return {}
        
        try:
            importance = optuna.importance.get_param_importances(self.study)
            return importance
        except:
            return {}
    
    def plot_optimization_history(self, filepath: Optional[str] = None) -> None:
        """Plot optimization history."""
        if self.study is None:
            self.logger.warning("No study available for plotting")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # Plot optimization history
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot 1: Optimization history
            trials = self.study.trials
            values = [trial.value for trial in trials if trial.value is not None]
            ax1.plot(values)
            ax1.set_title('Attention Network Optimization History')
            ax1.set_xlabel('Trial')
            ax1.set_ylabel('Score')
            ax1.grid(True)
            
            # Plot 2: Parameter importance
            importance = self._calculate_parameter_importance()
            if importance:
                params = list(importance.keys())
                scores = list(importance.values())
                ax2.barh(params, scores)
                ax2.set_title('Parameter Importance')
                ax2.set_xlabel('Importance')
            
            plt.tight_layout()
            
            if filepath:
                plt.savefig(filepath)
                self.logger.info(f"Optimization plot saved to {filepath}")
            else:
                plt.show()
                
        except ImportError:
            self.logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            self.logger.warning(f"Failed to create plot: {e}")
    
    def save_study(self, filepath: str) -> None:
        """Save optimization study."""
        if self.study is None:
            self.logger.warning("No study to save")
            return
        
        try:
            import joblib
            joblib.dump(self.study, filepath)
            self.logger.info(f"Study saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save study: {e}")
    
    def load_study(self, filepath: str) -> None:
        """Load optimization study."""
        try:
            import joblib
            self.study = joblib.load(filepath)
            self.best_params = self.study.best_params
            self.best_score = self.study.best_value
            self.logger.info(f"Study loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load study: {e}")