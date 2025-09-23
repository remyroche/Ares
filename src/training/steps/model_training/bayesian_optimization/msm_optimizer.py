"""
Bayesian Optimization for MSM Parameters
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
class MSMOptimizationConfig:
    """Configuration for MSM Bayesian optimization."""
    
    # Optimization parameters
    n_trials: int = 100
    timeout_seconds: Optional[int] = None
    n_startup_trials: int = 10
    n_warmup_steps: int = 5
    
    # MSM parameter ranges
    n_regimes_range: Tuple[int, int] = (2, 8)
    min_segment_length_range: Tuple[int, int] = (20, 100)
    break_penalty_options: List[str] = None
    
    # Clustering parameters
    clustering_method_options: List[str] = None
    n_components_range: Tuple[int, int] = (2, 10)
    covariance_type_options: List[str] = None
    
    # Feature engineering
    feature_selection: bool = True
    max_features: int = 1000
    
    # Cross-validation
    cv_folds: int = 5
    scoring: str = 'neg_mean_squared_error'
    
    # Early stopping
    early_stopping_patience: int = 20
    improvement_threshold: float = 0.001
    
    def __post_init__(self):
        if self.break_penalty_options is None:
            self.break_penalty_options = ['bic', 'aic', 'hannan_quinn']
        
        if self.clustering_method_options is None:
            self.clustering_method_options = ['gaussian_mixture', 'kmeans']
        
        if self.covariance_type_options is None:
            self.covariance_type_options = ['full', 'tied', 'diag', 'spherical']


class MSMBayesianOptimizer:
    """Bayesian optimization for MSM parameters."""
    
    def __init__(self, config: MSMOptimizationConfig):
        """Initialize MSM Bayesian optimizer."""
        self.config = config
        self.logger = system_logger.getChild('MSMBayesianOptimizer')
        
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
        """Optimize MSM parameters using Bayesian optimization."""
        self.logger.info("Starting MSM Bayesian optimization")
        
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
        
        self.logger.info(f"MSM optimization completed in {optimization_time:.2f} seconds")
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
        """Default objective function for MSM optimization."""
        try:
            # Sample MSM parameters
            n_regimes = trial.suggest_int('n_regimes', *self.config.n_regimes_range)
            min_segment_length = trial.suggest_int('min_segment_length', *self.config.min_segment_length_range)
            break_penalty = trial.suggest_categorical('break_penalty', self.config.break_penalty_options)
            clustering_method = trial.suggest_categorical('clustering_method', self.config.clustering_method_options)
            
            # Sample clustering parameters
            if clustering_method == 'gaussian_mixture':
                n_components = trial.suggest_int('n_components', *self.config.n_components_range)
                covariance_type = trial.suggest_categorical('covariance_type', self.config.covariance_type_options)
            else:
                n_components = trial.suggest_int('n_components', *self.config.n_components_range)
                covariance_type = 'full'  # Not used for KMeans
            
            # Create MSM configuration
            msm_config = {
                'n_regimes': n_regimes,
                'min_segment_length': min_segment_length,
                'break_penalty': break_penalty,
                'clustering_method': clustering_method,
                'n_components': n_components,
                'covariance_type': covariance_type
            }
            
            # Evaluate MSM model
            score = self._evaluate_msm_model(X, y, msm_config)
            
            return score
            
        except Exception as e:
            self.logger.warning(f"Trial failed: {e}")
            return -np.inf
    
    def _evaluate_msm_model(self, X: np.ndarray, y: np.ndarray, config: Dict[str, Any]) -> float:
        """Evaluate MSM model with given configuration."""
        try:
            # Import MSM clustering
            from src.training.steps.market_analysis.msm_clustering import MSMOptimizedClusterer
            
            # Create MSM clusterer
            clusterer = MSMOptimizedClusterer(config)
            
            # Fit MSM model
            result = clusterer.fit(X)
            
            # Calculate evaluation metrics
            metrics = self._calculate_msm_metrics(result, y)
            
            # Combine metrics into single score
            score = self._combine_metrics(metrics)
            
            return score
            
        except Exception as e:
            self.logger.warning(f"MSM evaluation failed: {e}")
            return -np.inf
    
    def _calculate_msm_metrics(self, result, y: np.ndarray) -> Dict[str, float]:
        """Calculate MSM evaluation metrics."""
        metrics = {}
        
        # Clustering quality metrics
        metrics['silhouette_score'] = result.silhouette_score
        metrics['calinski_harabasz_score'] = result.calinski_harabasz_score
        metrics['davies_bouldin_score'] = result.davies_bouldin_score
        
        # Model performance metrics
        metrics['log_likelihood'] = result.log_likelihood
        metrics['aic'] = result.aic
        metrics['bic'] = result.bic
        
        # Regime stability metrics
        metrics['regime_stability'] = self._calculate_regime_stability(result)
        metrics['transition_stability'] = self._calculate_transition_stability(result)
        
        return metrics
    
    def _calculate_regime_stability(self, result) -> float:
        """Calculate regime stability metric."""
        try:
            # Calculate regime duration variance
            durations = list(result.regime_durations.values())
            if len(durations) > 1:
                duration_variance = np.var(durations)
                stability = 1.0 / (1.0 + duration_variance)
            else:
                stability = 1.0
            
            return stability
            
        except:
            return 0.0
    
    def _calculate_transition_stability(self, result) -> float:
        """Calculate transition matrix stability."""
        try:
            # Calculate transition matrix entropy
            transition_matrix = result.transition_matrix
            
            # Remove zero probabilities
            non_zero_probs = transition_matrix[transition_matrix > 0]
            
            if len(non_zero_probs) > 0:
                entropy = -np.sum(non_zero_probs * np.log(non_zero_probs))
                stability = 1.0 / (1.0 + entropy)
            else:
                stability = 1.0
            
            return stability
            
        except:
            return 0.0
    
    def _combine_metrics(self, metrics: Dict[str, float]) -> float:
        """Combine multiple metrics into single score."""
        # Weighted combination of metrics
        weights = {
            'silhouette_score': 0.3,
            'calinski_harabasz_score': 0.2,
            'davies_bouldin_score': -0.1,  # Lower is better
            'log_likelihood': 0.2,
            'regime_stability': 0.1,
            'transition_stability': 0.1
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
            ax1.set_title('Optimization History')
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