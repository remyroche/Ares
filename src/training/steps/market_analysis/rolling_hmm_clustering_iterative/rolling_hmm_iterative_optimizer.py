"""
Iterative Rolling HMM Clustering Optimizer

This module implements an automated iterative optimization approach for Rolling HMM clustering.
Instead of traditional grid search, it uses a reduced initial parameter set and then
iteratively optimizes each parameter using 20% increments until convergence.

Optimization Order:
1. n_components (model structure)
2. pca_components (dimensionality reduction)
3. covariance_type (model structure)
4. n_iter (training parameters)
5. kmeans_init (initialization)
6. use_sticky_priors (regularization)
7. ewma_short (feature engineering - short window)
8. ewma_long (feature engineering - long window) 
9. min_covar (regularization)
10. kappa (regularization)
11. post_fit_regularization (regularization)

Each parameter is optimized using 20% increments until score stagnates or decreases.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Set
import logging
from dataclasses import dataclass
import time
import threading
import ctypes
import traceback

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_debug
from src.training.steps.market_analysis.clusters.clustering_optimization_goals import (
    DEFAULT_CLUSTERING_GOALS,
    DEFAULT_OPTIMIZATION_TARGETS
)
from src.training.steps.market_analysis.rolling_hmm_clustering_iterative.feature_engineering import (
    EWMAConfig
)

logger = logging.getLogger(__name__)

CV_RATIO_EPS = 1e-9
FORWARD_RETURN_HORIZON = 2
SHARPE_EPS = 1e-9


@dataclass
class IterativeHPOConfig:
    """Configuration for iterative hyperparameter optimization."""
    
    # Initial reduced parameters
    initial_n_components: int = 5
    initial_ewma_short: int = 6
    initial_ewma_long: int = 20
    initial_min_covar: float = 0.005
    initial_kappa: float = 2.0
    
    # Optimization settings
    increment_ratio: float = 0.2  # 20% increments
    improvement_threshold: float = 0.01  # 1% minimum improvement
    max_iterations: int = 20
    max_parameter_iterations: int = 10  # Max iterations per parameter
    
    # Convergence criteria
    convergence_patience: int = 3  # No improvement for N cycles
    min_score_improvement: float = 0.005  # Minimum improvement to continue
    
    # Cross-validation
    cv_folds: int = 5
    
    # Objective function weights
    weight_between_within_cv: float = 0.40
    weight_temporal: float = 0.20
    weight_economic: float = 0.40
    
    # Early stopping
    enable_early_stopping: bool = True
    early_stop_min_score: float = 0.05
    early_stop_min_quality_score: float = 0.1
    early_stop_min_temporal_smoothness: float = 0.1
    
    # Resource management
    enable_resource_monitoring: bool = True
    timeout_seconds: int = 60
    
    # Logging
    verbose: bool = True
    save_optimization_history: bool = True


class RollingHMMIterativeOptimizer:
    """
    Iterative optimizer for Rolling HMM clustering parameters.
    
    Uses reduced initial parameters and iterative refinement with 20% increments
    until convergence is achieved.
    """
    
    def __init__(self, config: IterativeHPOConfig):
        """
        Initialize iterative optimizer.
        
        Args:
            config: Iterative HPO configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        tprint_info("🧠 Initializing RollingHMMIterativeOptimizer")
        
        # Optimization state
        self.current_iteration = 0
        self.best_score = -np.inf
        self.best_params: Dict[str, Any] = {}
        self.optimization_history: List[Dict[str, Any]] = []
        self.convergence_counter = 0
        
        # Parameter optimization order
        self.optimization_order = [
            'n_components',
            'pca_components',
            'covariance_type',
            'n_iter',
            'kmeans_init',
            'use_sticky_priors',
            'ewma_short', 
            'ewma_long',
            'min_covar',
            'kappa',
            'post_fit_regularization'
        ]
        
        # Current parameters
        self.current_params = {
            'n_components': self.config.initial_n_components,
            'pca_components': None,
            'covariance_type': 'diag',
            'n_iter': 75,
            'kmeans_init': True,
            'use_sticky_priors': True,
            'ewma_short': self.config.initial_ewma_short,
            'ewma_long': self.config.initial_ewma_long,
            'min_covar': self.config.initial_min_covar,
            'kappa': self.config.initial_kappa,
            'post_fit_regularization': True
        }
        
        # Parameter bounds
        self.param_bounds = {
            'n_components': (4, 6),
            'ewma_short': (4, 12),
            'ewma_long': (16, 30),
            'min_covar': (1e-4, 1e-1),
            'kappa': (0.1, 10.0),
            'n_iter': (50, 150)
        }
        
        # Categorical parameter choices
        self.categorical_choices = {
            'covariance_type': ['diag', 'full', 'tied', 'spherical'],
            'kmeans_init': [True, False],
            'use_sticky_priors': [True, False],
            'post_fit_regularization': [True, False]
        }
        
        # Trial tracking
        self.total_trials = 0
        
    def create_objective_function(
        self,
        market_data: pd.DataFrame,
        feature_engineer,
        hmm_model_class,
        quality_assessor
    ):
        """
        Create objective function for iterative optimization.
        
        Args:
            market_data: Market data DataFrame
            feature_engineer: Feature engineering instance
            hmm_model_class: Sticky HMM model class
            quality_assessor: Cluster quality assessor instance
            
        Returns:
            Objective function callable
        """
        tprint_debug("Creating objective function for iterative optimization")
        
        def objective(params: Dict[str, Any]) -> Tuple[float, Optional[Any]]:
            """
            Objective function for iterative optimization.
            
            Args:
                params: Parameter dictionary
                
            Returns:
                Objective score (higher is better) and metrics
            """
            try:
                # Extract parameters
                n_components = int(params.get('n_components', 5))
                ewma_short = int(params.get('ewma_short', 6))
                ewma_long = int(params.get('ewma_long', 20))
                pca_components = params.get('pca_components')
                covariance_type = str(params.get('covariance_type', 'diag'))
                n_iter = int(params.get('n_iter', 75))
                kmeans_init = bool(params.get('kmeans_init', True))
                use_sticky_priors = bool(params.get('use_sticky_priors', True))
                min_covar = float(params.get('min_covar', 0.005))
                kappa = float(params.get('kappa', 2.0))
                post_fit_regularization = bool(params.get('post_fit_regularization', True))
                
                # Ensure ewma_short < ewma_long
                if ewma_short >= ewma_long:
                    ewma_long = ewma_short + 2
                
                # Create dynamic EWMA config
                ewma_config = EWMAConfig(
                    short_window=ewma_short,
                    long_window=ewma_long,
                    name=f"{ewma_short}+{ewma_long}"
                )
                
                # Generate features
                features = feature_engineer.generate_features(market_data, ewma_config)
                
                if len(features) < 50:
                    return -1e6, None
                
                # Extract economic features
                features_economic = feature_engineer.extract_economic_features(
                    features,
                    market_data,
                    ewma_config
                )
                
                # Create HMM config
                from src.training.steps.market_analysis.rolling_hmm_clustering_iterative.sticky_hmm_model import (
                    StickyHMMConfig
                )
                
                hmm_config = StickyHMMConfig(
                    n_components=n_components,
                    min_covar=min_covar,
                    kappa=kappa,
                    n_iter=n_iter,
                    covariance_type=covariance_type,
                    kmeans_init=kmeans_init,
                    use_sticky_priors=use_sticky_priors,
                    post_fit_regularization=post_fit_regularization,
                    early_stopping_enabled=True,
                    early_stopping_patience=3
                )
                
                # Fit HMM model with timeout
                tprint_debug(f"  🔧 Trial {self.total_trials}: Fitting HMM with n_components={n_components}, ewma={ewma_short}+{ewma_long}, min_covar={min_covar:.4f}, kappa={kappa:.2f}")
                
                hmm_model = hmm_model_class(hmm_config)
                
                fit_completed = threading.Event()
                fit_result = {'success': False, 'error': None, 'traceback': None}
                
                def fit_with_monitoring():
                    """Fit in thread with resource monitoring."""
                    try:
                        hmm_model.fit(
                            features_economic.values,
                            ewma_config_name=ewma_config.name,
                            pca_components=pca_components
                        )
                        fit_result['success'] = True
                        tprint_debug(f"  ✅ Trial {self.total_trials}: HMM fit completed successfully")
                    except Exception as e:
                        fit_result['error'] = str(e)
                        fit_result['traceback'] = traceback.format_exc()
                        tprint_error(f"  ❌ Trial {self.total_trials}: HMM fit failed: {str(e)}")
                    finally:
                        fit_completed.set()
                
                # Start fitting in background thread
                fit_thread = threading.Thread(target=fit_with_monitoring, daemon=True)
                start_time = time.time()
                fit_thread.start()
                
                # Monitor with timeout
                timeout_seconds = self.config.timeout_seconds
                if self.config.enable_resource_monitoring:
                    try:
                        from src.utils.hardware.unified_hardware_manager import get_unified_hardware_manager
                        hw_manager = get_unified_hardware_manager()
                        cpu_usage = hw_manager.get_cpu_usage()
                        memory_pressure = hw_manager.get_memory_pressure()
                        
                        if cpu_usage > 90 or memory_pressure > 0.8:
                            timeout_seconds = int(timeout_seconds * 1.5)
                            tprint_warning(f"  ⚠️ High resource pressure (CPU: {cpu_usage:.1f}%, Memory: {memory_pressure:.2f}) - Extending timeout to {timeout_seconds}s")
                    except Exception:
                        pass
                
                # Wait for completion or timeout
                if not fit_completed.wait(timeout=timeout_seconds):
                    tprint_warning(f"  ⏱️  Trial {self.total_trials} TIMEOUT after {timeout_seconds}s")
                    try:
                        # Forceful thread termination
                        thread_id = fit_thread.ident
                        if thread_id:
                            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                                ctypes.c_ulong(thread_id),
                                ctypes.py_object(SystemError("HMM fitting timeout"))
                            )
                    except Exception:
                        pass
                    return -1e6, None
                
                # Check if fitting succeeded
                if not fit_result['success']:
                    error_msg = fit_result['error'] or 'Unknown error'
                    tprint_warning(f"  ⚠️  Trial {self.total_trials} HMM fit failed: {error_msg[:80]}")
                    return -1e6, None
                
                # Predict regime labels
                regime_labels = hmm_model.predict(features_economic.values)
                
                # Calculate regime distribution
                unique_regimes, regime_counts = np.unique(regime_labels, return_counts=True)
                regime_distribution = regime_counts / len(regime_labels)
                
                # 5% minimum regime size constraint
                min_regime_size = 0.05
                violates_constraint = np.any(regime_distribution < min_regime_size)
                
                # Size penalty for tiny regimes
                size_penalty = 0.0
                if violates_constraint:
                    violations = regime_distribution[regime_distribution < min_regime_size]
                    size_penalty = np.sum((min_regime_size - violations) / min_regime_size) * 2.0
                
                # Balance penalty using entropy
                n_regimes = len(unique_regimes)
                current_entropy = -np.sum(regime_distribution * np.log(regime_distribution + 1e-9))
                max_entropy = np.log(n_regimes)
                balance_score = current_entropy / max_entropy if max_entropy > 0 else 1.0
                balance_penalty = (1.0 - balance_score) * 1.5
                
                # Get transition matrix
                transition_matrix = hmm_model.get_transition_matrix()
                
                # Calculate forward returns
                forward_returns = (
                    market_data['close'].pct_change(FORWARD_RETURN_HORIZON)
                    .shift(-FORWARD_RETURN_HORIZON)
                )
                forward_returns = forward_returns.loc[features_economic.index]
                
                # Assess quality with fast mode
                metrics = quality_assessor.assess_hmm_regime_quality(
                    regime_labels=regime_labels,
                    feature_data=features_economic,
                    transition_matrix=transition_matrix,
                    hmm_model=None,
                    forward_returns=forward_returns,
                    timestamps=features_economic.index,
                    timeframe='1h',
                    min_regime_size=10,
                    run_validators=False,
                    temporal_sensitivity_mode="standard",
                    fast_mode=True
                )
                
                # Calculate objective score components
                between_cv = getattr(metrics, 'between_regime_cv', 0.0)
                within_cv = getattr(metrics, 'within_regime_cv', 0.0)
                cv_ratio = between_cv / (within_cv + CV_RATIO_EPS) if within_cv is not None else 0.0
                normalized_cv_ratio = cv_ratio / (cv_ratio + 1.0) if cv_ratio > 0 else 0.0
                
                silhouette_raw = getattr(metrics, 'silhouette_score', 0.0)
                silhouette_norm = float(np.clip((silhouette_raw + 1.0) / 2.0, 0.0, 1.0))
                
                stat_score = float(np.clip((normalized_cv_ratio + silhouette_norm) / 2.0, 0.0, 1.0))
                score_statistical = stat_score * self.config.weight_between_within_cv
                
                temporal_smoothness = getattr(metrics, 'temporal_smoothness', 0.0)
                temporal_score = getattr(metrics, 'comprehensive_temporal_score', temporal_smoothness)
                score_temporal = (
                    temporal_smoothness * 0.5 + temporal_score * 0.5
                ) * self.config.weight_temporal
                
                # Economic components
                economic_components: List[float] = []
                
                if hasattr(metrics, 'economic_cv_metrics') and metrics.economic_cv_metrics:
                    economic_cv_ratio = metrics.economic_cv_metrics.get('economic_cv_ratio_mean_return', 0.0) or 0.0
                    if economic_cv_ratio > 0.0:
                        economic_components.append(
                            float(np.clip(economic_cv_ratio / (economic_cv_ratio + 1.0), 0.0, 1.0))
                        )
                
                if hasattr(metrics, 'economic_validation') and metrics.economic_validation:
                    mean_returns = [
                        regime_data.get('mean_return')
                        for regime_data in metrics.economic_validation.values()
                        if isinstance(regime_data, dict)
                    ]
                    volatilities = [
                        regime_data.get('volatility')
                        for regime_data in metrics.economic_validation.values()
                        if isinstance(regime_data, dict)
                    ]
                    mean_returns = [m for m in mean_returns if m is not None]
                    volatilities = [v for v in volatilities if v is not None]
                    
                    if mean_returns and volatilities:
                        avg_mean = float(np.mean(mean_returns))
                        avg_vol = float(np.mean(volatilities))
                        if avg_vol > SHARPE_EPS:
                            normalized_sharpe = float(np.clip((avg_mean / avg_vol + 2.0) / 6.0, 0.0, 1.0))
                            economic_components.append(normalized_sharpe)
                
                if economic_components:
                    economic_signal = float(np.mean(economic_components))
                else:
                    economic_signal = 0.0
                
                score_economic = economic_signal * self.config.weight_economic
                
                # Persistence penalty
                persistence_penalty = 0.0
                if transition_matrix is not None:
                    diag_mean = float(np.mean(np.diag(transition_matrix)))
                    persistence_penalty += max(0.0, diag_mean - 0.8) * 2.0
                    diag_variance = float(np.var(np.diag(transition_matrix)))
                    persistence_penalty += diag_variance * 0.5
                
                if metrics.regime_persistence is not None:
                    normalized_persistence = min(1.0, metrics.regime_persistence / 30.0)
                    persistence_penalty += normalized_persistence * 0.2
                
                # Economic constraint penalties
                economic_penalty = 0.0
                
                if hasattr(metrics, 'economic_validation') and metrics.economic_validation:
                    volatilities = [
                        regime_data.get('volatility')
                        for regime_data in metrics.economic_validation.values()
                        if isinstance(regime_data, dict) and regime_data.get('volatility') is not None
                    ]
                    if len(volatilities) >= 2:
                        vol_std = float(np.std(volatilities))
                        vol_mean = float(np.mean(volatilities))
                        vol_cv = vol_std / (vol_mean + 1e-8)
                        economic_penalty += max(0.0, 1.0 - vol_cv * 2.0)
                
                if hasattr(metrics, 'economic_validation') and metrics.economic_validation:
                    mean_returns = [
                        regime_data.get('mean_return')
                        for regime_data in metrics.economic_validation.values()
                        if isinstance(regime_data, dict) and regime_data.get('mean_return') is not None
                    ]
                    if len(mean_returns) >= 2:
                        return_range = float(np.max(mean_returns) - np.min(mean_returns))
                        normalized_range = min(1.0, return_range / 0.02)
                        economic_penalty += (1.0 - normalized_range) * 2.0
                
                if hasattr(metrics, 'economic_validation') and metrics.economic_validation:
                    sharpe_proxies = []
                    for regime_data in metrics.economic_validation.values():
                        if isinstance(regime_data, dict):
                            mean_ret = regime_data.get('mean_return')
                            vol = regime_data.get('volatility')
                            if mean_ret is not None and vol is not None and vol > 1e-8:
                                sharpe_proxies.append(mean_ret / vol)
                    
                    if len(sharpe_proxies) >= 2:
                        sharpe_std = float(np.std(sharpe_proxies))
                        normalized_diversity = min(1.0, sharpe_std / 1.0)
                        economic_penalty += (1.0 - normalized_diversity) * 1.5
                
                # Final objective score
                objective_score = (
                    score_statistical
                    + score_temporal
                    + score_economic
                    - persistence_penalty * self.config.weight_temporal
                    - size_penalty * 0.4
                    - balance_penalty * 0.2
                    - economic_penalty * 0.4
                )
                
                return objective_score, metrics
                
            except Exception as e:
                self.logger.warning(f"Objective function failed: {e}")
                tprint_warning(f"  ⚠️  Trial {self.total_trials} failed: {str(e)[:80]}")
                return -1e6, None
        
        return objective
    
    def optimize(
        self,
        market_data: pd.DataFrame,
        feature_engineer,
        hmm_model_class,
        quality_assessor
    ) -> Dict[str, Any]:
        """
        Run iterative parameter optimization.
        
        Args:
            market_data: Market data DataFrame
            feature_engineer: Feature engineering instance
            hmm_model_class: Sticky HMM model class
            quality_assessor: Cluster quality assessor instance
            
        Returns:
            Optimization result dictionary
        """
        tprint("🔍 Starting Iterative Parameter Optimization")
        tprint(f"📊 Initial parameters: {self.current_params}")
        
        # Create objective function
        objective_func = self.create_objective_function(
            market_data,
            feature_engineer,
            hmm_model_class,
            quality_assessor
        )
        
        # Evaluate initial parameters
        tprint_info("🎯 Evaluating initial parameters...")
        self.total_trials += 1
        initial_score, initial_metrics = objective_func(self.current_params)
        
        if initial_score <= -1e6:
            tprint_error("❌ Initial parameter evaluation failed")
            return {
                'success': False,
                'best_score': -np.inf,
                'best_params': {},
                'optimization_history': [],
                'n_trials': 0
            }
        
        self.best_score = initial_score
        self.best_params = self.current_params.copy()
        
        tprint_info(f"✅ Initial score: {initial_score:.4f}")
        
        # Start iterative optimization
        optimization_active = True
        iteration = 0
        
        while optimization_active and iteration < self.config.max_iterations:
            iteration += 1
            self.current_iteration = iteration
            
            tprint("")
            tprint(f"🔄 Iteration {iteration}/{self.config.max_iterations}")
            tprint(f"📊 Current best score: {self.best_score:.4f}")
            tprint(f"📊 Current best params: {self.best_params}")
            
            iteration_improved = False
            
            # Optimize each parameter in order
            for param_name in self.optimization_order:
                tprint_info(f"  🔧 Optimizing parameter: {param_name}")
                
                current_value = self.current_params[param_name]
                param_improved = False
                
                # Handle categorical parameters differently
                if param_name in self.categorical_choices:
                    # For categorical parameters, try all choices
                    best_choice = current_value
                    best_choice_score = -np.inf
                    
                    for choice in self.categorical_choices[param_name]:
                        if choice == current_value:
                            continue
                        
                        # Create test parameters
                        test_params = self.current_params.copy()
                        test_params[param_name] = choice
                        
                        # Evaluate
                        self.total_trials += 1
                        choice_score, _ = objective_func(test_params)
                        
                        if choice_score > best_choice_score:
                            best_choice = choice
                            best_choice_score = choice_score
                    
                    # If we found a better choice, use it
                    if best_choice_score > self.best_score * (1 + self.config.improvement_threshold):
                        self.current_params[param_name] = best_choice
                        self.best_score = best_choice_score
                        self.best_params = self.current_params.copy()
                        param_improved = True
                        iteration_improved = True
                        
                        tprint_info(f"    ✅ {param_name}: {current_value} → {best_choice} (score: {best_choice_score:.4f})")
                else:
                    # For continuous parameters, use the existing logic
                    # Try positive direction first
                    new_value, new_score = self._try_parameter_change(
                        param_name, current_value, initial_score, objective_func, direction='positive'
                    )
                    
                    if new_score > self.best_score * (1 + self.config.improvement_threshold):
                        # Continue in positive direction
                        new_value, new_score = self._continue_optimization_direction(
                            param_name, new_value, new_score, objective_func, direction='positive'
                        )
                        
                        if new_score > self.best_score:
                            self.current_params[param_name] = new_value
                            self.best_score = new_score
                            self.best_params = self.current_params.copy()
                            param_improved = True
                            iteration_improved = True
                            
                            tprint_info(f"    ✅ {param_name}: {current_value} → {new_value} (score: {new_score:.4f})")
                    else:
                        # Try negative direction
                        new_value, new_score = self._try_parameter_change(
                            param_name, current_value, initial_score, objective_func, direction='negative'
                        )
                        
                        if new_score > self.best_score * (1 + self.config.improvement_threshold):
                            # Continue in negative direction
                            new_value, new_score = self._continue_optimization_direction(
                                param_name, new_value, new_score, objective_func, direction='negative'
                            )
                            
                            if new_score > self.best_score:
                                self.current_params[param_name] = new_value
                                self.best_score = new_score
                                self.best_params = self.current_params.copy()
                                param_improved = True
                                iteration_improved = True
                                
                                tprint_info(f"    ✅ {param_name}: {current_value} → {new_value} (score: {new_score:.4f})")
                
                if not param_improved:
                    tprint_debug(f"    ⏭️  {param_name}: No improvement found")
            
            # Check convergence
            if iteration_improved:
                self.convergence_counter = 0
                tprint_info(f"📈 Iteration {iteration} improved score to {self.best_score:.4f}")
            else:
                self.convergence_counter += 1
                tprint_info(f"📊 Iteration {iteration} - No improvement (convergence counter: {self.convergence_counter}/{self.config.convergence_patience})")
                
                if self.convergence_counter >= self.config.convergence_patience:
                    tprint_info(f"✅ Convergence reached after {iteration} iterations")
                    optimization_active = False
        
        # Final result
        tprint("")
        tprint(f"🎯 Iterative optimization complete!")
        tprint(f"📊 Final best score: {self.best_score:.4f}")
        tprint(f"📊 Final best parameters: {self.best_params}")
        tprint(f"📊 Total trials: {self.total_trials}")
        
        return {
            'success': True,
            'best_score': self.best_score,
            'best_params': self.best_params,
            'optimization_history': self.optimization_history,
            'n_trials': self.total_trials,
            'iterations': iteration,
            'converged': self.convergence_counter >= self.config.convergence_patience
        }
    
    def _try_parameter_change(
        self,
        param_name: str,
        current_value: float,
        current_score: float,
        objective_func,
        direction: str = 'positive'
    ) -> Tuple[float, float]:
        """
        Try a single parameter change.
        
        Args:
            param_name: Parameter name
            current_value: Current parameter value
            current_score: Current score
            objective_func: Objective function
            direction: 'positive' or 'negative'
            
        Returns:
            Tuple of (new_value, new_score)
        """
        if direction == 'positive':
            new_value = current_value * (1 + self.config.increment_ratio)
        else:
            new_value = current_value * (1 - self.config.increment_ratio)
        
        # Apply bounds
        new_value = self._apply_parameter_bounds(param_name, new_value)
        
        # Skip if value hasn't changed (due to bounds)
        if abs(new_value - current_value) < 1e-8:
            return current_value, current_score
        
        # Create test parameters
        test_params = self.current_params.copy()
        test_params[param_name] = new_value
        
        # Evaluate
        self.total_trials += 1
        new_score, _ = objective_func(test_params)
        
        if new_score <= -1e6:
            return current_value, current_score
        
        tprint_debug(f"    📊 {param_name} {direction}: {current_value:.4f} → {new_value:.4f}, score: {current_score:.4f} → {new_score:.4f}")
        
        # Record in optimization history
        if self.config.save_optimization_history:
            self.optimization_history.append({
                'iteration': self.current_iteration,
                'trial': self.total_trials,
                'parameter': param_name,
                'direction': direction,
                'old_value': current_value,
                'new_value': new_value,
                'old_score': current_score,
                'new_score': new_score,
                'improvement': new_score > current_score
            })
        
        return new_value, new_score
    
    def _continue_optimization_direction(
        self,
        param_name: str,
        start_value: float,
        start_score: float,
        objective_func,
        direction: str = 'positive',
        max_iterations: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Continue optimization in a direction until no more improvement.
        
        Args:
            param_name: Parameter name
            start_value: Starting value
            start_score: Starting score
            objective_func: Objective function
            direction: 'positive' or 'negative'
            max_iterations: Maximum iterations (uses config if None)
            
        Returns:
            Tuple of (best_value, best_score)
        """
        if max_iterations is None:
            max_iterations = self.config.max_parameter_iterations
        
        best_value = start_value
        best_score = start_score
        current_value = start_value
        current_score = start_score
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            
            if direction == 'positive':
                new_value = current_value * (1 + self.config.increment_ratio)
            else:
                new_value = current_value * (1 - self.config.increment_ratio)
            
            # Apply bounds
            new_value = self._apply_parameter_bounds(param_name, new_value)
            
            # Skip if value hasn't changed (due to bounds)
            if abs(new_value - current_value) < 1e-8:
                break
            
            # Create test parameters
            test_params = self.current_params.copy()
            test_params[param_name] = new_value
            
            # Evaluate
            self.total_trials += 1
            new_score, _ = objective_func(test_params)
            
            if new_score <= -1e6:
                break
            
            tprint_debug(f"      📊 {param_name} {direction} iter {iterations}: {current_value:.4f} → {new_value:.4f}, score: {current_score:.4f} → {new_score:.4f}")
            
            # Record in optimization history
            if self.config.save_optimization_history:
                self.optimization_history.append({
                    'iteration': self.current_iteration,
                    'trial': self.total_trials,
                    'parameter': param_name,
                    'direction': direction,
                    'old_value': current_value,
                    'new_value': new_value,
                    'old_score': current_score,
                    'new_score': new_score,
                    'improvement': new_score > current_score
                })
            
            # Check for improvement
            if new_score > best_score * (1 + self.config.min_score_improvement):
                best_value = new_value
                best_score = new_score
                current_value = new_value
                current_score = new_score
            else:
                break
        
        return best_value, best_score
    
    def _apply_parameter_bounds(self, param_name: str, value: float) -> float:
        """
        Apply bounds to parameter values.
        
        Args:
            param_name: Parameter name
            value: Parameter value
            
        Returns:
            Bounded parameter value
        """
        if param_name not in self.param_bounds:
            return value
        
        min_val, max_val = self.param_bounds[param_name]
        
        if param_name in ['n_components', 'ewma_short', 'ewma_long']:
            # Integer parameters
            value = int(round(value))
            return max(min_val, min(max_val, value))
        else:
            # Float parameters
            return max(min_val, min(max_val, value))
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get summary of optimization process.
        
        Returns:
            Summary dictionary
        """
        return {
            'total_iterations': self.current_iteration,
            'total_trials': self.total_trials,
            'best_score': self.best_score,
            'best_params': self.best_params,
            'convergence_counter': self.convergence_counter,
            'optimization_history_length': len(self.optimization_history),
            'converged': self.convergence_counter >= self.config.convergence_patience
        }


# Default iterative HPO configuration
DEFAULT_ITERATIVE_HPO_CONFIG = IterativeHPOConfig(
    initial_n_components=5,
    initial_ewma_short=6,
    initial_ewma_long=20,
    initial_min_covar=0.005,
    initial_kappa=2.0,
    increment_ratio=0.2,
    improvement_threshold=0.01,
    max_iterations=20,
    max_parameter_iterations=10,
    convergence_patience=3,
    min_score_improvement=0.005,
    cv_folds=5,
    weight_between_within_cv=0.40,
    weight_temporal=0.20,
    weight_economic=0.40,
    enable_early_stopping=True,
    early_stop_min_score=0.05,
    early_stop_min_quality_score=0.1,
    early_stop_min_temporal_smoothness=0.1,
    enable_resource_monitoring=True,
    timeout_seconds=60,
    verbose=True,
    save_optimization_history=True
)
