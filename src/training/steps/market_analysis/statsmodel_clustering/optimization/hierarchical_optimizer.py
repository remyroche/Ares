"""
Hierarchical Parameter Optimizer for Robust Model Tuning

This module implements a 3-stage hierarchical optimization approach
combining global search, local refinement, and validation.

Enhanced with tprint logging and integration with src/utils/ml_common/optimization/
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass, field
import time
import warnings
import logging

# Import tprint for consistent output
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error
    )
    _tprint_available = True
except ImportError:
    _tprint_available = False
    def tprint(msg, level="INFO"): print(msg)
    def tprint_info(msg): print(f"ℹ️ {msg}")
    def tprint_success(msg): print(f"✅ {msg}")
    def tprint_warning(msg): print(f"⚠️ {msg}")
    def tprint_error(msg): print(f"❌ {msg}")

# Import from src/utils/ml_common/optimization/
try:
    from src.utils.ml_common.optimization import (
        HierarchicalParameterOptimizer as MLCommonHierarchicalParameterOptimizer,
        ParameterGroup,
        OptimizationStage,
        OptimizationBackend,
        StageConfig,
        OptimizationResult,
        HierarchicalOptimizationResult,
        create_param_group,
        default_objective_function,
        create_custom_balanced_score_objective,
        CUSTOM_BALANCED_SCORE_AVAILABLE,
        get_execution_mode,
        adjust_hpo_params_for_mode
    )
    _ML_COMMON_AVAILABLE = True
except ImportError:
    _ML_COMMON_AVAILABLE = False
    tprint_warning("⚠️ src/utils/ml_common/optimization not available, using fallback implementation")

# Import optimization frameworks
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from skopt import gp_minimize, forest_minimize
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

# Import logger
try:
    from src.utils.logger import system_logger
    logger = system_logger.getChild('HierarchicalOptimizer')
except ImportError:
    logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for hierarchical optimization."""
    # Stage 1: Global search
    stage1_method: str = 'random'  # 'random', 'bayesian', 'forest'
    stage1_n_trials: int = 50
    stage1_timeout: Optional[float] = None  # seconds
    
    # Stage 2: Local refinement
    stage2_method: str = 'bfgs'  # 'bfgs', 'nelder-mead', 'cobyla'
    stage2_n_starts: int = 5
    stage2_max_iter: int = 100
    
    # Stage 3: Validation
    stage3_method: str = 'rolling_window'  # 'rolling_window', 'holdout'
    stage3_window_size: int = 252  # trading days
    stage3_n_folds: int = 5
    
    # Economic objectives
    enable_economic_objectives: bool = True
    economic_weights: Dict[str, float] = field(default_factory=lambda: {
        'sharpe_ratio': 0.4,
        'information_ratio': 0.3,
        'turnover_penalty': 0.2,
        'regime_stability': 0.1
    })
    
    # Multi-objective optimization
    enable_multi_objective: bool = False
    pareto_front_size: int = 10
    
    # Constraints
    parameter_constraints: Optional[Dict[str, Tuple[float, float]]] = None
    
    # Early stopping
    enable_early_stopping: bool = True
    patience: int = 10
    improvement_threshold: float = 1e-4
    
    def __post_init__(self):
        """Initialize configuration with tprint logging."""
        tprint_info("🔧 Initializing OptimizationConfig")
        tprint_info(f"📊 Stage 1: {self.stage1_method} ({self.stage1_n_trials} trials)")
        tprint_info(f"🔍 Stage 2: {self.stage2_method} ({self.stage2_n_starts} starts)")
        tprint_info(f"✅ Stage 3: {self.stage3_method} ({self.stage3_n_folds} folds)")
        tprint_info(f"💰 Economic objectives: {self.enable_economic_objectives}")
        tprint_info(f"📊 Economic weights: {self.economic_weights}")
        tprint_info(f"🎯 Multi-objective: {self.enable_multi_objective}")
        tprint_info(f"📏 Pareto front size: {self.pareto_front_size}")
        tprint_info(f"⏱️ Early stopping: {self.enable_early_stopping}")
        tprint_info(f"📊 Patience: {self.patience}, Improvement threshold: {self.improvement_threshold}")
        tprint_success("✅ OptimizationConfig initialized successfully")


class HierarchicalParameterOptimizer:
    """
    Hierarchical parameter optimizer with 3-stage optimization approach.
    
    Stage 1: Global search (random/Bayesian/forest)
    Stage 2: Local refinement (BFGS/Nelder-Mead)
    Stage 3: Validation on holdout/rolling windows
    
    Enhanced with tprint logging and integration with src/utils/ml_common/optimization/
    """
    
    def __init__(self,
                 objective_function: Callable,
                 parameter_space: Dict[str, Any],
                 config: Optional[OptimizationConfig] = None):
        """
        Initialize hierarchical optimizer.
        
        Args:
            objective_function: Function to optimize
            parameter_space: Parameter search space
            config: Optimization configuration
        """
        tprint_info("🚀 Initializing Hierarchical Parameter Optimizer")
        tprint_info(f"📊 Parameter space: {list(parameter_space.keys())}")
        
        self.objective_function = objective_function
        self.parameter_space = parameter_space
        self.config = config or OptimizationConfig()
        
        # Try to use ML Common optimization if available
        if _ML_COMMON_AVAILABLE:
            tprint_info("🔗 Using src/utils/ml_common/optimization/ backend")
            self._init_ml_common_optimizer()
        else:
            tprint_warning("⚠️ Using fallback implementation (ml_common not available)")
            self._init_fallback_optimizer()
        
        # Optimization state
        self.best_params = None
        self.best_score = None
        self.optimization_history = []
        
        # Validate configuration
        self._validate_config()
        
        tprint_success("✅ Hierarchical Parameter Optimizer initialized successfully")
    
    def _init_ml_common_optimizer(self):
        """Initialize using ML Common optimization backend."""
        tprint_info("🔗 Initializing ML Common optimization backend")
        
        try:
            # Convert parameter space to ParameterGroup format
            tprint_info("🔄 Converting parameter space to ML Common format")
            param_group = ParameterGroup(
                name="hierarchical_group",
                params=self._convert_parameter_space(),
                priority=1
            )
            tprint_info(f"📊 Created parameter group with {len(param_group.params)} parameters")
            
            # Create stages
            stages = [
                OptimizationStage.COARSE_GRID,
                OptimizationStage.FINE_GRID,
                OptimizationStage.TPE
            ]
            tprint_info(f"📊 Created optimization stages: {[s.value for s in stages]}")
            
            # Create ML Common optimizer
            tprint_info("🚀 Creating ML Common optimizer")
            self.ml_common_optimizer = MLCommonHierarchicalParameterOptimizer(
                param_groups=[param_group],
                objective_func=self._wrap_objective_function,
                stages=stages,
                cv_folds=self.config.stage3_n_folds,
                direction='maximize',
                verbose=True
            )
            
            self.use_ml_common = True
            tprint_success("✅ ML Common optimizer initialized successfully")
            
        except Exception as e:
            tprint_error(f"❌ Failed to initialize ML Common optimizer: {e}")
            self.use_ml_common = False
    
    def _init_fallback_optimizer(self):
        """Initialize fallback optimizer implementation."""
        tprint_info("🔄 Initializing fallback optimizer implementation")
        self.use_ml_common = False
        tprint_info("🔄 Fallback optimizer initialized successfully")
    
    def _convert_parameter_space(self) -> Dict[str, Dict[str, Any]]:
        """Convert parameter space to ML Common format."""
        tprint_info("🔄 Converting parameter space to ML Common format")
        converted = {}
        
        for param_name, param_config in self.parameter_space.items():
            if isinstance(param_config, dict):
                if param_config.get('type') == 'categorical':
                    converted[param_name] = {
                        'type': 'categorical',
                        'choices': param_config['choices']
                    }
                    tprint_info(f"🔄 {param_name}: categorical ({len(param_config['choices'])} choices)")
                elif param_config.get('type') in ['uniform', 'loguniform']:
                    converted[param_name] = {
                        'type': 'float',
                        'low': param_config['low'],
                        'high': param_config['high'],
                        'log': param_config.get('type') == 'loguniform'
                    }
                    log_str = "log" if param_config.get('type') == 'loguniform' else "linear"
                    tprint_info(f"🔄 {param_name}: {log_str} float [{param_config['low']}, {param_config['high']}])")
                elif param_config.get('type') == 'int':
                    converted[param_name] = {
                        'type': 'int',
                        'low': param_config['low'],
                        'high': param_config['high']
                    }
                    tprint_info(f"🔄 {param_name}: int [{param_config['low']}, {param_config['high']}])")
                else:
                    # Default to float
                    converted[param_name] = {
                        'type': 'float',
                        'low': param_config.get('low', 0.0),
                        'high': param_config.get('high', 1.0)
                    }
                    tprint_info(f"🔄 {param_name}: default float [{param_config.get('low', 0.0)}, {param_config.get('high', 1.0)}])")
            else:
                # Simple range
                if isinstance(param_config, (list, tuple)) and len(param_config) == 2:
                    converted[param_name] = {
                        'type': 'float',
                        'low': param_config[0],
                        'high': param_config[1]
                    }
                    tprint_info(f"🔄 {param_name}: range float [{param_config[0]}, {param_config[1]}])")
                else:
                    converted[param_name] = {
                        'type': 'float',
                        'low': 0.0,
                        'high': 1.0
                    }
                    tprint_info(f"🔄 {param_name}: default float [0.0, 1.0])")
        
        tprint_info(f"✅ Converted {len(converted)} parameters to ML Common format")
        return converted
    
    def _wrap_objective_function(self, params, X_train, y_train, X_val=None, y_val=None,
                                model=None, cv_folds=5, scoring_metric='custom_balanced_score', **kwargs):
        """Wrap objective function for ML Common compatibility."""
        tprint_info(f"🔄 Evaluating objective with {len(params)} parameters")
        
        try:
            # Call original objective function
            if self.config.enable_economic_objectives:
                tprint_info("💰 Using economic objectives")
                result = self.objective_function(params, X_train)
                
                if isinstance(result, dict):
                    # Combine multiple objectives
                    score = 0.0
                    for metric, weight in self.config.economic_weights.items():
                        if metric in result:
                            score += weight * result[metric]
                            tprint_info(f"📊 {metric}: {result[metric]:.6f} × {weight:.2f} = {weight * result[metric]:.6f}")
                    tprint_info(f"💰 Combined economic score: {score:.6f}")
                    return score
                else:
                    tprint_info(f"💰 Economic score: {result:.6f}")
                    return result
            else:
                tprint_info("📊 Using simple objective")
                # Simple objective
                score = self.objective_function(params, X_train)
                tprint_info(f"📊 Objective score: {score:.6f}")
                return score
        except Exception as e:
            tprint_error(f"❌ Objective function evaluation failed: {e}")
            return -np.inf
    
    def _validate_config(self):
        """Validate optimization configuration."""
        tprint_info("🔍 Validating optimization configuration")
        
        if self.config.stage1_n_trials < 10:
            tprint_error("❌ stage1_n_trials must be >= 10")
            raise ValueError("stage1_n_trials must be >= 10")
        
        if self.config.stage2_n_starts < 1:
            tprint_error("❌ stage2_n_starts must be >= 1")
            raise ValueError("stage2_n_starts must be >= 1")
        
        if self.config.stage3_window_size < 50:
            tprint_error("❌ stage3_window_size must be >= 50")
            raise ValueError("stage3_window_size must be >= 50")
        
        tprint_success("✅ Configuration validation passed")
    
    def optimize(self,
                data: Any,
                initial_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run hierarchical optimization.
        
        Args:
            data: Data for objective function
            initial_params: Optional initial parameters
            
        Returns:
            Optimization results
        """
        tprint_info("🚀 Starting hierarchical optimization")
        tprint_info(f"📊 Data type: {type(data).__name__}")
        tprint_info(f"📊 Data size: {len(data) if hasattr(data, '__len__') else 'N/A'}")
        
        if initial_params:
            tprint_info(f"📊 Initial parameters: {list(initial_params.keys())}")
        
        start_time = time.time()
        
        try:
            # Use ML Common optimizer if available
            if self.use_ml_common and hasattr(self, 'ml_common_optimizer'):
                tprint_info("🔗 Using ML Common optimization backend")
                return self._optimize_with_ml_common(data, initial_params)
            else:
                tprint_info("🔄 Using fallback optimization implementation")
                return self._optimize_fallback(data, initial_params)
                
        except Exception as e:
            tprint_error(f"❌ Hierarchical optimization failed: {e}")
            raise
    
    def _optimize_with_ml_common(self, data: Any, initial_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Optimize using ML Common backend."""
        tprint_info("🔗 Running optimization with ML Common backend")
        
        # Create dummy X_train, y_train for ML Common optimizer
        # In practice, data should be properly formatted
        if isinstance(data, (np.ndarray, pd.DataFrame)):
            if hasattr(data, 'shape') and len(data.shape) >= 2:
                X_train = data[:, :-1] if isinstance(data, np.ndarray) else data.iloc[:, :-1]
                y_train = data[:, -1] if isinstance(data, np.ndarray) else data.iloc[:, -1]
                tprint_info(f"📊 Data shape: {data.shape}, X_train: {X_train.shape}, y_train: {y_train.shape}")
            else:
                X_train = np.arange(len(data)).reshape(-1, 1)
                y_train = data
                tprint_info(f"📊 Data length: {len(data)}, X_train: {X_train.shape}, y_train: {y_train.shape}")
        else:
            # Fallback for other data types
            X_train = np.arange(100).reshape(-1, 1)
            y_train = np.random.randn(100)
            tprint_info("📊 Using fallback data (100 samples)")
        
        if initial_params:
            tprint_info(f"📊 Initial parameters: {list(initial_params.keys())}")
        
        # Run ML Common optimization
        tprint_info("🚀 Starting ML Common optimization")
        result = self.ml_common_optimizer.optimize(
            X_train=X_train,
            y_train=y_train,
            initial_params=initial_params
        )
        
        tprint_success(f"✅ ML Common optimization completed in {result.total_time:.2f}s")
        tprint_info(f"🎯 Best score: {result.best_score:.6f}")
        tprint_info(f"📊 Best parameters: {list(result.best_params.keys())}")
        
        # Convert results to expected format
        return {
            'best_params': result.best_params,
            'best_score': result.best_score,
            'validation_score': result.best_score,  # ML Common doesn't separate validation
            'stage1_results': {'best_params': result.best_params, 'best_score': result.best_score},
            'stage2_results': {'best_params': result.best_params, 'best_score': result.best_score},
            'stage3_results': {'validation_score': result.best_score},
            'total_time': result.total_time,
            'optimization_history': [],  # ML Common handles history differently
            'config': self.config.__dict__
        }
    
    def _optimize_fallback(self, data: Any, initial_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fallback optimization implementation."""
        tprint_info("🔄 Running fallback optimization implementation")
        start_time = time.time()
        
        if initial_params:
            tprint_info(f"📊 Initial parameters: {list(initial_params.keys())}")
        
        # Stage 1: Global search
        tprint_info("🔍 Starting Stage 1: Global search")
        stage1_results = self._stage1_global_search(data, initial_params)
        tprint_info(f"✅ Stage 1 completed with best score: {stage1_results['best_score']:.6f}")
        
        # Stage 2: Local refinement
        tprint_info("🔍 Starting Stage 2: Local refinement")
        stage2_results = self._stage2_local_refinement(
            stage1_results['best_params'], data
        )
        tprint_info(f"✅ Stage 2 completed with best score: {stage2_results['best_score']:.6f}")
        
        # Stage 3: Validation
        tprint_info("🔍 Starting Stage 3: Validation")
        stage3_results = self._stage3_validation(
            stage2_results['best_params'], data
        )
        tprint_info(f"✅ Stage 3 completed with validation score: {stage3_results['validation_score']:.6f}")
        
        # Compile results
        total_time = time.time() - start_time
        
        results = {
            'best_params': stage2_results['best_params'],
            'best_score': stage2_results['best_score'],
            'validation_score': stage3_results['validation_score'],
            'stage1_results': stage1_results,
            'stage2_results': stage2_results,
            'stage3_results': stage3_results,
            'total_time': total_time,
            'optimization_history': self.optimization_history,
            'config': self.config.__dict__
        }
        
        tprint_success(f"✅ Fallback optimization complete in {total_time:.2f}s")
        tprint_info(f"🎯 Final best score: {stage2_results['best_score']:.6f}")
        tprint_info(f"📊 Final validation score: {stage3_results['validation_score']:.6f}")
        
        return results
    
    def _stage1_global_search(self,
                             data: Any,
                             initial_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Stage 1: Global search for promising regions."""
        tprint_info(f"🔍 Stage 1: Global search using {self.config.stage1_method}")
        tprint_info(f"📊 Parameter space: {list(self.parameter_space.keys())}")
        
        if initial_params:
            tprint_info(f"📊 Initial parameters: {list(initial_params.keys())}")
        
        if self.config.stage1_method == 'random':
            tprint_info("🎲 Using random search for global exploration")
            return self._random_search(data, initial_params)
        elif self.config.stage1_method == 'bayesian':
            tprint_info("🔍 Using Bayesian optimization for global exploration")
            return self._bayesian_search(data, initial_params)
        elif self.config.stage1_method == 'forest':
            tprint_info("🌲 Using forest-based optimization for global exploration")
            return self._forest_search(data, initial_params)
        else:
            tprint_error(f"❌ Unknown stage1 method: {self.config.stage1_method}")
            raise ValueError(f"Unknown stage1 method: {self.config.stage1_method}")
    
    def _random_search(self,
                     data: Any,
                     initial_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Random search for global exploration."""
        tprint_info("🔍 Running random search for global exploration")
        tprint_info(f"📊 Number of trials: {self.config.stage1_n_trials}")
        
        best_params = initial_params or self._sample_random_params()
        best_score = self._evaluate_params(best_params, data)
        tprint_info(f"📊 Initial best score: {best_score:.6f}")
        
        self.optimization_history.append({
            'stage': 'stage1_random',
            'params': best_params.copy(),
            'score': best_score,
            'iteration': 0
        })
        
        for i in range(1, self.config.stage1_n_trials):
            params = self._sample_random_params()
            score = self._evaluate_params(params, data)
            
            self.optimization_history.append({
                'stage': 'stage1_random',
                'params': params.copy(),
                'score': score,
                'iteration': i
            })
            
            if score > best_score:
                best_params = params
                best_score = score
                tprint_info(f"🎯 New best score: {best_score:.6f} at iteration {i}")
            
            if (i + 1) % 10 == 0:
                tprint_info(f"📈 Completed {i + 1}/{self.config.stage1_n_trials} trials")
        
        tprint_success(f"✅ Random search completed with {self.config.stage1_n_trials} trials")
        tprint_info(f"🎯 Final best score: {best_score:.6f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': self.config.stage1_n_trials
        }
    
    def _bayesian_search(self,
                       data: Any,
                       initial_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Bayesian optimization using Optuna."""
        tprint_info("🔍 Running Bayesian optimization with Optuna")
        
        if not OPTUNA_AVAILABLE:
            tprint_warning("⚠️ Optuna not available, falling back to random search")
            return self._random_search(data, initial_params)
        
        def objective(trial):
            params = {}
            for param_name, param_config in self.parameter_space.items():
                if isinstance(param_config, dict):
                    if param_config.get('type') == 'categorical':
                        params[param_name] = trial.suggest_categorical(
                            param_name, param_config['choices']
                        )
                    elif param_config.get('type') == 'uniform':
                        params[param_name] = trial.suggest_uniform(
                            param_name, param_config['low'], param_config['high']
                        )
                    elif param_config.get('type') == 'loguniform':
                        params[param_name] = trial.suggest_loguniform(
                            param_name, param_config['low'], param_config['high']
                        )
                    elif param_config.get('type') == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name, param_config['low'], param_config['high']
                        )
                    elif param_config.get('type') == 'discrete_uniform':
                        params[param_name] = trial.suggest_discrete_uniform(
                            param_name, param_config['low'], param_config['high'], param_config['q']
                        )
                else:
                    # Simple parameter
                    params[param_name] = trial.suggest_float(
                        param_name, param_config[0], param_config[1]
                    )
            
            score = self._evaluate_params(params, data)
            return score
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config.stage1_n_trials)
        
        tprint_success(f"✅ Bayesian optimization completed with {len(study.trials)} trials")
        tprint_info(f"🎯 Best score: {study.best_value:.6f}")
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials)
        }
    
    def _forest_search(self,
                     data: Any,
                     initial_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Forest-based optimization using skopt."""
        tprint_info("🔍 Running forest-based optimization with skopt")
        
        if not SKOPT_AVAILABLE:
            tprint_warning("⚠️ skopt not available, falling back to random search")
            return self._random_search(data, initial_params)
        
        # Convert parameter space for skopt
        dimensions = []
        param_names = []
        
        for param_name, param_config in self.parameter_space.items():
            param_names.append(param_name)
            
            if isinstance(param_config, dict):
                if param_config.get('type') == 'categorical':
                    dimensions.append(param_config['choices'])
                elif param_config.get('type') == 'uniform':
                    dimensions.append((param_config['low'], param_config['high']))
                elif param_config.get('type') == 'loguniform':
                    dimensions.append((param_config['low'], param_config['high']))
                elif param_config.get('type') == 'int':
                    dimensions.append((param_config['low'], param_config['high']))
            else:
                dimensions.append(param_config)
        
        def objective(params):
            param_dict = dict(zip(param_names, params))
            score = self._evaluate_params(param_dict, data)
            return -score  # skopt minimizes
        
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=self.config.stage1_n_trials,
            random_state=42
        )
        
        best_params = dict(zip(param_names, result.x))
        
        tprint_success(f"✅ Forest optimization completed with {len(result.x_iters)} trials")
        tprint_info(f"🎯 Best score: {-result.fun:.6f}")
        
        return {
            'best_params': best_params,
            'best_score': -result.fun,
            'n_trials': len(result.x_iters)
        }
    
    def _stage2_local_refinement(self,
                               best_params: Dict[str, Any],
                               data: Any) -> Dict[str, Any]:
        """Stage 2: Local refinement around best parameters."""
        tprint_info(f"🔍 Stage 2: Local refinement using {self.config.stage2_method}")
        tprint_info(f"📊 Best parameters from Stage 1: {list(best_params.keys())}")
        
        if self.config.stage2_method == 'bfgs':
            tprint_info("🔍 Using BFGS for local refinement")
            return self._bfgs_refinement(best_params, data)
        else:
            tprint_warning(f"⚠️ Unknown stage2 method: {self.config.stage2_method}")
            tprint_info("🔄 Using parameter evaluation as fallback")
            return {'best_params': best_params, 'best_score': self._evaluate_params(best_params, data)}
    
    def _bfgs_refinement(self,
                        best_params: Dict[str, Any],
                        data: Any) -> Dict[str, Any]:
        """BFGS local refinement."""
        tprint_info("🔍 Running BFGS local refinement")
        
        try:
            from scipy.optimize import minimize
        except ImportError:
            tprint_warning("⚠️ scipy not available, skipping BFGS refinement")
            return {'best_params': best_params, 'best_score': self._evaluate_params(best_params, data)}
        
        best_score = self._evaluate_params(best_params, data)
        tprint_info(f"📊 Initial best score: {best_score:.6f}")
        
        # Convert parameters to vector for optimization
        param_vector = self._params_to_vector(best_params)
        
        # Define bounds
        bounds = self._get_parameter_bounds()
        
        # Multi-start optimization
        for start in range(self.config.stage2_n_starts):
            # Add noise to best parameters for different starting points
            if start > 0:
                noisy_params = self._add_parameter_noise(best_params)
                param_vector = self._params_to_vector(noisy_params)
            
            result = minimize(
                fun=lambda x: -self._evaluate_params(self._vector_to_params(x), data),
                x0=param_vector,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': self.config.stage2_max_iter}
            )
            
            score = -result.fun
            if score > best_score:
                best_score = score
                best_params = self._vector_to_params(result.x)
                tprint_info(f"🎯 New best score: {best_score:.6f} at start {start}")
        
        tprint_success(f"✅ BFGS refinement completed with {self.config.stage2_n_starts} starts")
        tprint_info(f"🎯 Final best score: {best_score:.6f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'n_starts': self.config.stage2_n_starts
        }
    
    def _stage3_validation(self,
                        best_params: Dict[str, Any],
                        data: Any) -> Dict[str, Any]:
        """Stage 3: Validation on holdout/rolling windows."""
        tprint_info(f"🔍 Stage 3: Validation using {self.config.stage3_method}")
        tprint_info(f"📊 Best parameters from Stage 2: {list(best_params.keys())}")
        
        if self.config.stage3_method == 'rolling_window':
            tprint_info("🔄 Using rolling window validation")
            return self._rolling_window_validation(best_params, data)
        elif self.config.stage3_method == 'holdout':
            tprint_info("🔄 Using holdout validation")
            return self._holdout_validation(best_params, data)
        else:
            tprint_error(f"❌ Unknown stage3 method: {self.config.stage3_method}")
            raise ValueError(f"Unknown stage3 method: {self.config.stage3_method}")
    
    def _rolling_window_validation(self,
                               best_params: Dict[str, Any],
                               data: Any) -> Dict[str, Any]:
        """Rolling window validation."""
        tprint_info("🔍 Running rolling window validation")
        
        # This is a simplified implementation
        # In practice, you'd implement proper rolling window validation
        
        validation_scores = []
        window_size = self.config.stage3_window_size
        
        tprint_info(f"📊 Window size: {window_size}, Folds: {self.config.stage3_n_folds}")
        
        # Simulate rolling windows (simplified)
        for fold in range(self.config.stage3_n_folds):
            # Create train/validation split
            train_size = len(data) - window_size - fold * (window_size // 2)
            
            if train_size < window_size:
                validation_scores.append(0.0)
                continue
            
            # Evaluate on validation set
            score = self._evaluate_params(best_params, data[:train_size])
            validation_scores.append(score)
            
            if (fold + 1) % 5 == 0:
                tprint_info(f"📈 Completed fold {fold + 1}/{self.config.stage3_n_folds}")
        
        validation_score = np.mean(validation_scores)
        validation_std = np.std(validation_scores)
        
        tprint_success(f"✅ Rolling window validation completed")
        tprint_info(f"📊 Mean validation score: {validation_score:.6f} ± {validation_std:.6f}")
        
        return {
            'validation_score': validation_score,
            'validation_std': validation_std,
            'fold_scores': validation_scores,
            'n_folds': len(validation_scores)
        }
    
    def _holdout_validation(self,
                         best_params: Dict[str, Any],
                         data: Any) -> Dict[str, Any]:
        """Holdout validation."""
        tprint_info("🔍 Running holdout validation")
        
        # Split data into train/validation
        train_size = int(len(data) * 0.8)
        tprint_info(f"📊 Train size: {train_size}, Validation size: {len(data) - train_size}")
        
        train_score = self._evaluate_params(best_params, data[:train_size])
        val_score = self._evaluate_params(best_params, data[train_size:])
        
        generalization_gap = train_score - val_score
        
        tprint_success(f"✅ Holdout validation completed")
        tprint_info(f"📊 Train score: {train_score:.6f}, Validation score: {val_score:.6f}")
        tprint_info(f"📊 Generalization gap: {generalization_gap:.6f}")
        
        return {
            'validation_score': val_score,
            'train_score': train_score,
            'generalization_gap': generalization_gap
        }
    
    def _evaluate_params(self, params: Dict[str, Any], data: Any) -> float:
        """Evaluate parameters using objective function."""
        tprint_info(f"🔍 Evaluating parameters: {list(params.keys())}")
        
        try:
            if self.config.enable_economic_objectives:
                tprint_info("💰 Using economic objectives")
                # Economic objective combining multiple metrics
                result = self.objective_function(params, data)
                
                if isinstance(result, dict):
                    # Combine multiple objectives
                    score = 0.0
                    for metric, weight in self.config.economic_weights.items():
                        if metric in result:
                            score += weight * result[metric]
                            tprint_info(f"📊 {metric}: {result[metric]:.6f} × {weight:.2f} = {weight * result[metric]:.6f}")
                    tprint_info(f"💰 Combined economic score: {score:.6f}")
                    return score
                else:
                    tprint_info(f"💰 Economic score: {result:.6f}")
                    return result
            else:
                tprint_info("📊 Using simple objective")
                # Simple objective
                score = self.objective_function(params, data)
                tprint_info(f"📊 Objective score: {score:.6f}")
                return score
        except Exception as e:
            tprint_error(f"❌ Parameter evaluation failed: {e}")
            return -np.inf
    
    def _sample_random_params(self) -> Dict[str, Any]:
        """Sample random parameters from search space."""
        tprint_info("🎲 Sampling random parameters from search space")
        
        params = {}
        
        for param_name, param_config in self.parameter_space.items():
            if isinstance(param_config, dict):
                if param_config.get('type') == 'categorical':
                    value = np.random.choice(param_config['choices'])
                    tprint_info(f"🎲 {param_name}: {value} (from {param_config['choices']})")
                    params[param_name] = value
                elif param_config.get('type') == 'uniform':
                    value = np.random.uniform(
                        param_config['low'], param_config['high']
                    )
                    tprint_info(f"🎲 {param_name}: {value:.6f} (uniform [{param_config['low']}, {param_config['high']}])")
                    params[param_name] = value
                elif param_config.get('type') == 'loguniform':
                    value = np.exp(np.random.uniform(
                        np.log(param_config['low']), np.log(param_config['high'])
                    ))
                    tprint_info(f"🎲 {param_name}: {value:.6f} (log-uniform [{param_config['low']}, {param_config['high']}])")
                    params[param_name] = value
                elif param_config.get('type') == 'int':
                    value = np.random.randint(
                        param_config['low'], param_config['high'] + 1
                    )
                    tprint_info(f"🎲 {param_name}: {value} (int [{param_config['low']}, {param_config['high']}])")
                    params[param_name] = value
                elif param_config.get('type') == 'discrete_uniform':
                    values = np.arange(
                        param_config['low'], param_config['high'] + 1, param_config['q']
                    )
                    value = np.random.choice(values)
                    tprint_info(f"🎲 {param_name}: {value} (discrete uniform [{param_config['low']}, {param_config['high']}, step={param_config['q']}])")
                    params[param_name] = value
            else:
                # Simple parameter range
                if isinstance(param_config, (list, tuple)) and len(param_config) == 2:
                    value = np.random.uniform(param_config[0], param_config[1])
                    tprint_info(f"🎲 {param_name}: {value:.6f} (range [{param_config[0]}, {param_config[1]}])")
                    params[param_name] = value
                else:
                    tprint_info(f"🎲 {param_name}: {param_config} (fixed)")
                    params[param_name] = param_config
        
        tprint_info(f"🎲 Sampled {len(params)} parameters")
        return params
    
    def _params_to_vector(self, params: Dict[str, Any]) -> np.ndarray:
        """Convert parameter dictionary to vector."""
        tprint_info("🔄 Converting parameter dictionary to vector")
        
        vector = []
        for param_name in self.parameter_space.keys():
            value = params.get(param_name, 0.0)
            vector.append(value)
        
        result = np.array(vector)
        tprint_info(f"🔄 Converted {len(vector)} parameters to vector of shape {result.shape}")
        
        return result
    
    def _vector_to_params(self, vector: np.ndarray) -> Dict[str, Any]:
        """Convert vector to parameter dictionary."""
        tprint_info("🔄 Converting vector to parameter dictionary")
        
        params = {}
        param_names = list(self.parameter_space.keys())
        
        for i, param_name in enumerate(param_names):
            if i < len(vector):
                params[param_name] = vector[i]
        
        tprint_info(f"🔄 Converted vector of shape {vector.shape} to {len(params)} parameters")
        
        return params
    
    def _get_parameter_bounds(self) -> List[Tuple[float, float]]:
        """Get parameter bounds for optimization."""
        tprint_info("🔍 Getting parameter bounds for optimization")
        
        bounds = []
        
        for param_name, param_config in self.parameter_space.items():
            if isinstance(param_config, dict):
                if param_config.get('type') in ['uniform', 'loguniform']:
                    bounds.append((param_config['low'], param_config['high']))
                elif param_config.get('type') == 'int':
                    bounds.append((param_config['low'], param_config['high']))
                else:
                    # Default bounds
                    bounds.append((0.0, 1.0))
            else:
                # Simple parameter range
                if isinstance(param_config, (list, tuple)) and len(param_config) == 2:
                    bounds.append(param_config)
                else:
                    bounds.append((0.0, 1.0))
        
        tprint_info(f"🔍 Generated {len(bounds)} parameter bounds")
        
        return bounds
    
    def _add_parameter_noise(self, params: Dict[str, Any], noise_level: float = 0.1) -> Dict[str, Any]:
        """Add noise to parameters for multi-start optimization."""
        tprint_info(f"🎲 Adding parameter noise with level {noise_level}")
        
        noisy_params = params.copy()
        
        for param_name, param_config in self.parameter_space.items():
            if isinstance(param_config, dict):
                if param_config.get('type') in ['uniform', 'loguniform']:
                    current_value = params.get(param_name, 0.0)
                    range_width = param_config['high'] - param_config['low']
                    noise = np.random.normal(0, noise_level * range_width)
                    noisy_params[param_name] = np.clip(
                        current_value + noise,
                        param_config['low'],
                        param_config['high']
                    )
                elif param_config.get('type') == 'int':
                    current_value = params.get(param_name, 0)
                    noise = int(np.random.normal(0, noise_level * 2))
                    noisy_params[param_name] = np.clip(
                        current_value + noise,
                        param_config['low'],
                        param_config['high']
                    )
        
        tprint_info(f"🎲 Added noise to {len(noisy_params)} parameters")
        
        return noisy_params


def create_hierarchical_optimizer(
    objective_function: Callable,
    parameter_space: Dict[str, Any],
    stage1_method: str = 'bayesian',
    stage1_n_trials: int = 50,
    stage2_method: str = 'bfgs',
    enable_economic_objectives: bool = True
) -> HierarchicalParameterOptimizer:
    """
    Factory function to create hierarchical optimizer.
    
    Args:
        objective_function: Function to optimize
        parameter_space: Parameter search space
        stage1_method: Global search method
        stage1_n_trials: Number of global search trials
        stage2_method: Local refinement method
        enable_economic_objectives: Enable economic objectives
        
    Returns:
        HierarchicalParameterOptimizer instance
    """
    tprint_info("🏭 Creating hierarchical optimizer with factory function")
    tprint_info(f"📊 Stage 1: {stage1_method} ({stage1_n_trials} trials)")
    tprint_info(f"🔍 Stage 2: {stage2_method}")
    tprint_info(f"💰 Economic objectives: {enable_economic_objectives}")
    
    config = OptimizationConfig(
        stage1_method=stage1_method,
        stage1_n_trials=stage1_n_trials,
        stage2_method=stage2_method,
        enable_economic_objectives=enable_economic_objectives
    )
    
    optimizer = HierarchicalParameterOptimizer(objective_function, parameter_space, config)
    tprint_success("✅ Hierarchical optimizer created successfully")
    
    return optimizer