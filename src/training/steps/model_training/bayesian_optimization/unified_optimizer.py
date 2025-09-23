"""
Unified Bayesian Optimization for All ML Components
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import logging
import time

from .msm_optimizer import MSMBayesianOptimizer, MSMOptimizationConfig
from .attention_optimizer import AttentionBayesianOptimizer, AttentionOptimizationConfig

from src.utils.logger import system_logger

try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler
    from optuna.pruners import MedianPruner, HyperbandPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


@dataclass
class UnifiedOptimizationConfig:
    """Configuration for unified Bayesian optimization."""
    
    # Optimization parameters
    n_trials: int = 200
    timeout_seconds: Optional[int] = None
    n_startup_trials: int = 20
    n_warmup_steps: int = 10
    
    # Component optimization weights
    msm_weight: float = 0.3
    attention_weight: float = 0.3
    ensemble_weight: float = 0.2
    deepscaler_weight: float = 0.2
    
    # MSM configuration
    msm_config: MSMOptimizationConfig = None
    
    # Attention configuration
    attention_config: AttentionOptimizationConfig = None
    
    # Ensemble configuration
    ensemble_config: Dict[str, Any] = None
    
    # DeepScaler configuration
    deepscaler_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.msm_config is None:
            self.msm_config = MSMOptimizationConfig()
        
        if self.attention_config is None:
            self.attention_config = AttentionOptimizationConfig()
        
        if self.ensemble_config is None:
            self.ensemble_config = {
                'n_trials': 50,
                'weight_range': (0.0, 1.0),
                'meta_learner_options': ['linear', 'ridge', 'lasso', 'elastic_net']
            }
        
        if self.deepscaler_config is None:
            self.deepscaler_config = {
                'n_trials': 50,
                'learning_rate_range': (0.0001, 0.01),
                'hidden_layers_range': (2, 8),
                'dropout_rate_range': (0.1, 0.5)
            }


class UnifiedBayesianOptimizer:
    """Unified Bayesian optimization for all ML components."""
    
    def __init__(self, config: UnifiedOptimizationConfig):
        """Initialize unified Bayesian optimizer."""
        self.config = config
        self.logger = system_logger.getChild('UnifiedBayesianOptimizer')
        
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available. Install with: pip install optuna")
        
        # Initialize component optimizers
        self.msm_optimizer = MSMBayesianOptimizer(self.config.msm_config)
        self.attention_optimizer = AttentionBayesianOptimizer(self.config.attention_config)
        
        # Initialize study
        self.study = None
        self.best_params = None
        self.best_score = None
        self.optimization_history = []
        
    def optimize(self, X: np.ndarray, y: np.ndarray, 
                 components: List[str] = None) -> Dict[str, Any]:
        """Optimize all ML components using unified Bayesian optimization."""
        self.logger.info("Starting unified Bayesian optimization")
        
        if components is None:
            components = ['msm', 'attention', 'ensemble', 'deepscaler']
        
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
        
        # Optimize
        start_time = time.time()
        
        self.study.optimize(
            lambda trial: self._unified_objective(trial, X, y, components),
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_seconds
        )
        
        optimization_time = time.time() - start_time
        
        # Store results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        self.logger.info(f"Unified optimization completed in {optimization_time:.2f} seconds")
        self.logger.info(f"Best score: {self.best_score:.4f}")
        self.logger.info(f"Best parameters: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_time': optimization_time,
            'n_trials': len(self.study.trials),
            'study': self.study
        }
    
    def _unified_objective(self, trial, X: np.ndarray, y: np.ndarray, components: List[str]) -> float:
        """Unified objective function for all components."""
        try:
            total_score = 0.0
            component_scores = {}
            
            # Optimize MSM if included
            if 'msm' in components:
                msm_score = self._optimize_msm_component(trial, X, y)
                total_score += self.config.msm_weight * msm_score
                component_scores['msm'] = msm_score
            
            # Optimize Attention if included
            if 'attention' in components:
                attention_score = self._optimize_attention_component(trial, X, y)
                total_score += self.config.attention_weight * attention_score
                component_scores['attention'] = attention_score
            
            # Optimize Ensemble if included
            if 'ensemble' in components:
                ensemble_score = self._optimize_ensemble_component(trial, X, y)
                total_score += self.config.ensemble_weight * ensemble_score
                component_scores['ensemble'] = ensemble_score
            
            # Optimize DeepScaler if included
            if 'deepscaler' in components:
                deepscaler_score = self._optimize_deepscaler_component(trial, X, y)
                total_score += self.config.deepscaler_weight * deepscaler_score
                component_scores['deepscaler'] = deepscaler_score
            
            # Store component scores for analysis
            trial.set_user_attr('component_scores', component_scores)
            
            return total_score
            
        except Exception as e:
            self.logger.warning(f"Trial failed: {e}")
            return -np.inf
    
    def _optimize_msm_component(self, trial, X: np.ndarray, y: np.ndarray) -> float:
        """Optimize MSM component."""
        try:
            # Sample MSM parameters
            n_regimes = trial.suggest_int('msm_n_regimes', *self.config.msm_config.n_regimes_range)
            min_segment_length = trial.suggest_int('msm_min_segment_length', *self.config.msm_config.min_segment_length_range)
            break_penalty = trial.suggest_categorical('msm_break_penalty', self.config.msm_config.break_penalty_options)
            clustering_method = trial.suggest_categorical('msm_clustering_method', self.config.msm_config.clustering_method_options)
            
            # Create MSM configuration
            msm_config = {
                'n_regimes': n_regimes,
                'min_segment_length': min_segment_length,
                'break_penalty': break_penalty,
                'clustering_method': clustering_method
            }
            
            # Evaluate MSM model
            score = self.msm_optimizer._evaluate_msm_model(X, y, msm_config)
            
            return score
            
        except Exception as e:
            self.logger.warning(f"MSM optimization failed: {e}")
            return -np.inf
    
    def _optimize_attention_component(self, trial, X: np.ndarray, y: np.ndarray) -> float:
        """Optimize Attention component."""
        try:
            # Sample attention parameters
            attention_dim = trial.suggest_int('attention_dim', *self.config.attention_config.attention_dim_range)
            num_heads = trial.suggest_int('attention_num_heads', *self.config.attention_config.num_heads_range)
            dropout_rate = trial.suggest_float('attention_dropout_rate', *self.config.attention_config.dropout_rate_range)
            learning_rate = trial.suggest_float('attention_learning_rate', *self.config.attention_config.learning_rate_range)
            regularization = trial.suggest_float('attention_regularization', *self.config.attention_config.regularization_range)
            
            # Create attention configuration
            attention_config = {
                'attention_dim': attention_dim,
                'num_heads': num_heads,
                'dropout_rate': dropout_rate,
                'learning_rate': learning_rate,
                'regularization': regularization
            }
            
            # Evaluate attention network
            score = self.attention_optimizer._evaluate_attention_network(X, y, attention_config)
            
            return score
            
        except Exception as e:
            self.logger.warning(f"Attention optimization failed: {e}")
            return -np.inf
    
    def _optimize_ensemble_component(self, trial, X: np.ndarray, y: np.ndarray) -> float:
        """Optimize Ensemble component."""
        try:
            # Sample ensemble parameters
            n_models = trial.suggest_int('ensemble_n_models', 3, 10)
            meta_learner = trial.suggest_categorical('ensemble_meta_learner', self.config.ensemble_config['meta_learner_options'])
            
            # Sample model weights
            model_weights = []
            for i in range(n_models):
                weight = trial.suggest_float(f'ensemble_weight_{i}', *self.config.ensemble_config['weight_range'])
                model_weights.append(weight)
            
            # Normalize weights
            model_weights = np.array(model_weights)
            model_weights = model_weights / np.sum(model_weights)
            
            # Create ensemble configuration
            ensemble_config = {
                'n_models': n_models,
                'meta_learner': meta_learner,
                'model_weights': model_weights.tolist()
            }
            
            # Evaluate ensemble model
            score = self._evaluate_ensemble_model(X, y, ensemble_config)
            
            return score
            
        except Exception as e:
            self.logger.warning(f"Ensemble optimization failed: {e}")
            return -np.inf
    
    def _optimize_deepscaler_component(self, trial, X: np.ndarray, y: np.ndarray) -> float:
        """Optimize DeepScaler component."""
        try:
            # Sample DeepScaler parameters
            learning_rate = trial.suggest_float('deepscaler_learning_rate', *self.config.deepscaler_config['learning_rate_range'])
            hidden_layers = trial.suggest_int('deepscaler_hidden_layers', *self.config.deepscaler_config['hidden_layers_range'])
            dropout_rate = trial.suggest_float('deepscaler_dropout_rate', *self.config.deepscaler_config['dropout_rate_range'])
            
            # Create DeepScaler configuration
            deepscaler_config = {
                'learning_rate': learning_rate,
                'hidden_layers': hidden_layers,
                'dropout_rate': dropout_rate
            }
            
            # Evaluate DeepScaler model
            score = self._evaluate_deepscaler_model(X, y, deepscaler_config)
            
            return score
            
        except Exception as e:
            self.logger.warning(f"DeepScaler optimization failed: {e}")
            return -np.inf
    
    def _evaluate_ensemble_model(self, X: np.ndarray, y: np.ndarray, config: Dict[str, Any]) -> float:
        """Evaluate ensemble model."""
        try:
            # This is a simplified implementation
            # In practice, you would create and evaluate the actual ensemble model
            
            # Simulate ensemble evaluation
            n_models = config['n_models']
            model_weights = np.array(config['model_weights'])
            
            # Calculate weighted ensemble score
            base_score = 0.8  # Simulated base score
            weight_penalty = np.sum(model_weights ** 2)  # Penalty for uneven weights
            
            score = base_score - 0.1 * weight_penalty
            
            return score
            
        except Exception as e:
            self.logger.warning(f"Ensemble evaluation failed: {e}")
            return -np.inf
    
    def _evaluate_deepscaler_model(self, X: np.ndarray, y: np.ndarray, config: Dict[str, Any]) -> float:
        """Evaluate DeepScaler model."""
        try:
            # This is a simplified implementation
            # In practice, you would create and evaluate the actual DeepScaler model
            
            # Simulate DeepScaler evaluation
            learning_rate = config['learning_rate']
            hidden_layers = config['hidden_layers']
            dropout_rate = config['dropout_rate']
            
            # Calculate DeepScaler score
            base_score = 0.7  # Simulated base score
            complexity_penalty = 0.01 * hidden_layers  # Penalty for complexity
            dropout_penalty = 0.1 * abs(dropout_rate - 0.2)  # Penalty for suboptimal dropout
            
            score = base_score - complexity_penalty - dropout_penalty
            
            return score
            
        except Exception as e:
            self.logger.warning(f"DeepScaler evaluation failed: {e}")
            return -np.inf
    
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
                    'state': trial.state.name,
                    'component_scores': trial.user_attrs.get('component_scores', {})
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
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Optimization history
            trials = self.study.trials
            values = [trial.value for trial in trials if trial.value is not None]
            ax1.plot(values)
            ax1.set_title('Unified Optimization History')
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
            
            # Plot 3: Component scores over time
            component_scores = {}
            for trial in trials:
                if trial.user_attrs.get('component_scores'):
                    for component, score in trial.user_attrs['component_scores'].items():
                        if component not in component_scores:
                            component_scores[component] = []
                        component_scores[component].append(score)
            
            for component, scores in component_scores.items():
                ax3.plot(scores, label=component)
            ax3.set_title('Component Scores Over Time')
            ax3.set_xlabel('Trial')
            ax3.set_ylabel('Score')
            ax3.legend()
            ax3.grid(True)
            
            # Plot 4: Score distribution
            ax4.hist(values, bins=20, alpha=0.7)
            ax4.set_title('Score Distribution')
            ax4.set_xlabel('Score')
            ax4.set_ylabel('Frequency')
            ax4.grid(True)
            
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