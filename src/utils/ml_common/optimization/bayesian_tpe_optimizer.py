"""
Bayesian TPE (Tree-structured Parzen Estimator) Optimizer

This module provides Bayesian optimization using TPE for hyperparameter tuning
in machine learning models, particularly for regime detection systems.
"""

import numpy as np
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
import warnings

# Optional dependencies
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not available - using fallback optimization")

try:
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import make_scorer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available - using fallback optimization")

logger = logging.getLogger(__name__)

@dataclass
class BayesianTPEConfig:
    """Configuration for Bayesian TPE optimization."""
    
    n_trials: int = 100
    timeout: Optional[float] = None
    random_state: Optional[int] = None
    n_startup_trials: int = 10
    n_ei_candidates: int = 24
    gamma: float = 0.25
    prior_weight: float = 1.0
    n_jobs: int = 1
    verbose: bool = False

class BayesianTPEOptimizer:
    """
    Bayesian TPE optimizer for hyperparameter tuning.
    
    This optimizer uses Tree-structured Parzen Estimator (TPE) for efficient
    hyperparameter optimization, particularly suitable for regime detection systems.
    """
    
    def __init__(self, 
                 n_trials: int = 100,
                 timeout: Optional[float] = None,
                 random_state: Optional[int] = None,
                 n_startup_trials: int = 10,
                 n_ei_candidates: int = 24,
                 gamma: float = 0.25,
                 prior_weight: float = 1.0,
                 n_jobs: int = 1,
                 verbose: bool = False):
        """
        Initialize Bayesian TPE optimizer.
        
        Args:
            n_trials: Number of optimization trials
            timeout: Maximum time for optimization (seconds)
            random_state: Random seed for reproducibility
            n_startup_trials: Number of random trials before TPE
            n_ei_candidates: Number of candidates for expected improvement
            gamma: Gamma parameter for TPE
            prior_weight: Weight for prior distribution
            n_jobs: Number of parallel jobs
            verbose: Whether to print optimization progress
        """
        self.config = BayesianTPEConfig(
            n_trials=n_trials,
            timeout=timeout,
            random_state=random_state,
            n_startup_trials=n_startup_trials,
            n_ei_candidates=n_ei_candidates,
            gamma=gamma,
            prior_weight=prior_weight,
            n_jobs=n_jobs,
            verbose=verbose
        )
        
        self.study = None
        self.best_params = None
        self.best_value = None
        self.optimization_history = []
        
        logger.info("✅ Bayesian TPE optimizer initialized")

    def optimize(self, 
                objective: Callable,
                search_space: Dict[str, Any],
                direction: str = 'maximize') -> Dict[str, Any]:
        """
        Optimize hyperparameters using Bayesian TPE.
        
        Args:
            objective: Objective function to optimize
            search_space: Search space definition
            direction: Optimization direction ('maximize' or 'minimize')
            
        Returns:
            Best parameters found
        """
        try:
            logger.info("🔬 Starting Bayesian TPE optimization...")
            start_time = time.time()
            
            if OPTUNA_AVAILABLE:
                return self._optimize_with_optuna(objective, search_space, direction)
            else:
                return self._optimize_fallback(objective, search_space, direction)
                
        except Exception as e:
            logger.error(f"❌ Bayesian TPE optimization failed: {e}")
            return {}

    def _optimize_with_optuna(self, 
                             objective: Callable,
                             search_space: Dict[str, Any],
                             direction: str) -> Dict[str, Any]:
        """Optimize using Optuna TPE sampler with enhanced TAS regime detection integration."""
        try:
            # Create study with enhanced configuration for regime detection
            self.study = optuna.create_study(
                direction=direction,
                sampler=optuna.samplers.TPESampler(
                    n_startup_trials=self.config.n_startup_trials,
                    n_ei_candidates=self.config.n_ei_candidates,
                    gamma=self.config.gamma,
                    prior_weight=self.config.prior_weight,
                    seed=self.config.random_state
                ),
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=5,
                    n_warmup_steps=10,
                    interval_steps=1
                )
            )
            
            # Define objective function with search space
            def optuna_objective(trial):
                params = {}
                
                for param_name, param_config in search_space.items():
                    if param_config['type'] == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name, 
                            param_config['low'], 
                            param_config['high']
                        )
                    elif param_config['type'] == 'float':
                        if param_config.get('log', False):
                            params[param_name] = trial.suggest_loguniform(
                                param_name,
                                param_config['low'],
                                param_config['high']
                            )
                        else:
                            params[param_name] = trial.suggest_uniform(
                                param_name,
                                param_config['low'],
                                param_config['high']
                            )
                    elif param_config['type'] == 'categorical':
                        params[param_name] = trial.suggest_categorical(
                            param_name,
                            param_config['choices']
                        )
                
                return objective(params)
            
            # Run optimization
            self.study.optimize(
                optuna_objective,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout,
                n_jobs=self.config.n_jobs,
                show_progress_bar=self.config.verbose
            )
            
            # Get results
            self.best_params = self.study.best_params
            self.best_value = self.study.best_value
            
            optimization_time = time.time() - start_time
            logger.info(f"✅ Bayesian TPE optimization completed in {optimization_time:.2f}s")
            logger.info(f"   Best value: {self.best_value:.4f}")
            logger.info(f"   Best parameters: {self.best_params}")
            
            return self.best_params
            
        except Exception as e:
            logger.error(f"❌ Optuna optimization failed: {e}")
            return self._optimize_fallback(objective, search_space, direction)

    def _optimize_fallback(self, 
                          objective: Callable,
                          search_space: Dict[str, Any],
                          direction: str) -> Dict[str, Any]:
        """Fallback optimization using random search."""
        try:
            logger.warning("⚠️ Using fallback random search optimization")
            
            best_params = None
            best_value = float('-inf') if direction == 'maximize' else float('inf')
            
            for trial in range(self.config.n_trials):
                # Generate random parameters
                params = {}
                for param_name, param_config in search_space.items():
                    if param_config['type'] == 'int':
                        params[param_name] = np.random.randint(
                            param_config['low'], 
                            param_config['high'] + 1
                        )
                    elif param_config['type'] == 'float':
                        if param_config.get('log', False):
                            params[param_name] = np.exp(np.random.uniform(
                                np.log(param_config['low']),
                                np.log(param_config['high'])
                            ))
                        else:
                            params[param_name] = np.random.uniform(
                                param_config['low'],
                                param_config['high']
                            )
                    elif param_config['type'] == 'categorical':
                        params[param_name] = np.random.choice(param_config['choices'])
                
                # Evaluate objective
                try:
                    value = objective(params)
                    
                    # Update best if better
                    if direction == 'maximize' and value > best_value:
                        best_value = value
                        best_params = params.copy()
                    elif direction == 'minimize' and value < best_value:
                        best_value = value
                        best_params = params.copy()
                        
                except Exception as e:
                    logger.warning(f"⚠️ Trial {trial} failed: {e}")
                    continue
            
            self.best_params = best_params
            self.best_value = best_value
            
            optimization_time = time.time() - start_time
            logger.info(f"✅ Fallback optimization completed in {optimization_time:.2f}s")
            logger.info(f"   Best value: {self.best_value:.4f}")
            logger.info(f"   Best parameters: {self.best_params}")
            
            return best_params or {}
            
        except Exception as e:
            logger.error(f"❌ Fallback optimization failed: {e}")
            return {}

    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters found."""
        return self.best_params or {}

    def get_best_value(self) -> float:
        """Get best value found."""
        return self.best_value or 0.0

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        if self.study:
            return [
                {
                    'trial': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': trial.state.name
                }
                for trial in self.study.trials
            ]
        else:
            return self.optimization_history

    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Plot optimization history."""
        try:
            if not OPTUNA_AVAILABLE or not self.study:
                logger.warning("⚠️ Cannot plot optimization history - Optuna not available")
                return
            
            import matplotlib.pyplot as plt
            
            # Plot optimization history
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot 1: Optimization history
            trials = self.study.trials
            values = [trial.value for trial in trials if trial.value is not None]
            ax1.plot(values)
            ax1.set_xlabel('Trial')
            ax1.set_ylabel('Objective Value')
            ax1.set_title('Optimization History')
            ax1.grid(True)
            
            # Plot 2: Parameter importance (if available)
            try:
                importance = optuna.importance.get_param_importances(self.study)
                if importance:
                    params = list(importance.keys())
                    importances = list(importance.values())
                    ax2.barh(params, importances)
                    ax2.set_xlabel('Importance')
                    ax2.set_title('Parameter Importance')
            except Exception as e:
                logger.warning(f"⚠️ Could not plot parameter importance: {e}")
                ax2.text(0.5, 0.5, 'Parameter importance not available', 
                        ha='center', va='center', transform=ax2.transAxes)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"✅ Optimization plot saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.warning(f"⚠️ Could not plot optimization history: {e}")

    def suggest_parameters(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest parameters for next trial."""
        try:
            if self.study and len(self.study.trials) > 0:
                # Use TPE to suggest next parameters
                trial = self.study.ask()
                return trial.params
            else:
                # Random suggestion for first trial
                params = {}
                for param_name, param_config in search_space.items():
                    if param_config['type'] == 'int':
                        params[param_name] = np.random.randint(
                            param_config['low'], 
                            param_config['high'] + 1
                        )
                    elif param_config['type'] == 'float':
                        params[param_name] = np.random.uniform(
                            param_config['low'],
                            param_config['high']
                        )
                    elif param_config['type'] == 'categorical':
                        params[param_name] = np.random.choice(param_config['choices'])
                return params
                
        except Exception as e:
            logger.warning(f"⚠️ Parameter suggestion failed: {e}")
            return {}

    def prune_trials(self, n_trials: int = 10):
        """Prune worst performing trials."""
        try:
            if self.study and len(self.study.trials) > n_trials:
                # Get worst trials to prune
                trials_with_values = [
                    (trial, trial.value) for trial in self.study.trials 
                    if trial.value is not None
                ]
                trials_with_values.sort(key=lambda x: x[1])
                
                # Prune worst trials
                for trial, _ in trials_with_values[:-n_trials]:
                    trial.state = optuna.trial.TrialState.PRUNED
                    
                logger.info(f"✅ Pruned {len(trials_with_values) - n_trials} worst trials")
                
        except Exception as e:
            logger.warning(f"⚠️ Trial pruning failed: {e}")

    def get_study_summary(self) -> Dict[str, Any]:
        """Get study summary statistics."""
        try:
            if not self.study:
                return {}
            
            trials = self.study.trials
            values = [trial.value for trial in trials if trial.value is not None]
            
            if not values:
                return {}
            
            return {
                'n_trials': len(trials),
                'n_completed_trials': len(values),
                'best_value': self.study.best_value,
                'best_params': self.study.best_params,
                'mean_value': np.mean(values),
                'std_value': np.std(values),
                'min_value': np.min(values),
                'max_value': np.max(values),
                'optimization_direction': self.study.direction.name
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Could not get study summary: {e}")
            return {}

    def save_study(self, filepath: str):
        """Save study to file."""
        try:
            if self.study:
                import pickle
                with open(filepath, 'wb') as f:
                    pickle.dump(self.study, f)
                logger.info(f"✅ Study saved to {filepath}")
            else:
                logger.warning("⚠️ No study to save")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not save study: {e}")

    def load_study(self, filepath: str):
        """Load study from file."""
        try:
            import pickle
            with open(filepath, 'rb') as f:
                self.study = pickle.load(f)
            logger.info(f"✅ Study loaded from {filepath}")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not load study: {e}")

    def resume_optimization(self, 
                          objective: Callable,
                          search_space: Dict[str, Any],
                          direction: str = 'maximize') -> Dict[str, Any]:
        """Resume optimization from existing study."""
        try:
            if not self.study:
                logger.warning("⚠️ No study to resume")
                return self.optimize(objective, search_space, direction)
            
            logger.info("🔄 Resuming Bayesian TPE optimization...")
            
            # Continue optimization
            self.study.optimize(
                objective,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout,
                n_jobs=self.config.n_jobs,
                show_progress_bar=self.config.verbose
            )
            
            self.best_params = self.study.best_params
            self.best_value = self.study.best_value
            
            logger.info(f"✅ Resumed optimization completed")
            logger.info(f"   Best value: {self.best_value:.4f}")
            logger.info(f"   Best parameters: {self.best_params}")
            
            return self.best_params
            
        except Exception as e:
            logger.error(f"❌ Resume optimization failed: {e}")
            return self.optimize(objective, search_space, direction)

    def optimize_tas_regime_detection(self, 
                                     market_data: np.ndarray,
                                     timestamps: Optional[np.ndarray] = None,
                                     base_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize TAS regime detection hyperparameters.
        
        Args:
            market_data: Market data for optimization
            timestamps: Optional timestamps
            base_config: Base configuration to optimize from
            
        Returns:
            Optimized parameters for TAS regime detection
        """
        try:
            logger.info("🎯 Starting TAS regime detection optimization...")
            
            # Define TAS-specific search space
            tas_search_space = {
                'n_regimes': {'type': 'int', 'low': 3, 'high': 12},
                'tree_depth': {'type': 'int', 'low': 4, 'high': 12},
                'n_estimators': {'type': 'int', 'low': 100, 'high': 2000},
                'min_samples_split': {'type': 'int', 'low': 5, 'high': 50},
                'min_samples_leaf': {'type': 'int', 'low': 2, 'high': 20},
                'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', 'auto', 0.5, 0.8]},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3},
                'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0}
            }
            
            # Create objective function for TAS regime detection
            def tas_objective(params):
                try:
                    # Import TAS detector here to avoid circular imports
                    from src.training.steps.market_analysis.tas_regime.core.tas_regime_detector import TASRegimeDetector
                    from src.training.steps.market_analysis.tas_regime.core.tas_regime_config import TASRegimeConfig
                    
                    # Create config with optimized parameters
                    config = TASRegimeConfig(**params) if base_config else TASRegimeConfig()
                    
                    # Initialize detector
                    detector = TASRegimeDetector(config)
                    
                    # Run regime detection
                    result = detector.detect_regimes(
                        market_data=market_data,
                        timestamps=timestamps,
                        enable_cross_validation=True,
                        enable_out_of_sample_validation=True
                    )
                    
                    if result.success:
                        # Return negative of mean stability score for minimization
                        stability_score = np.mean(result.regime_stability_scores)
                        economic_score = np.mean(result.economic_significance_scores)
                        trading_score = np.mean(result.trading_viability_scores)
                        
                        # Combined score (higher is better, so we return negative for minimization)
                        combined_score = -(0.4 * stability_score + 0.3 * economic_score + 0.3 * trading_score)
                        return combined_score
                    else:
                        return float('inf')
                        
                except Exception as e:
                    logger.warning(f"TAS objective evaluation failed: {e}")
                    return float('inf')
            
            # Run optimization
            best_params = self.optimize(tas_objective, tas_search_space, 'minimize')
            
            logger.info(f"✅ TAS regime detection optimization completed")
            logger.info(f"Best parameters: {best_params}")
            
            return best_params
            
        except Exception as e:
            logger.error(f"❌ TAS regime detection optimization failed: {e}")
            return {}