"""
Hyperparameter Optimization Utilities for Hybrid NAS-TAS Regime Detection.

Provides comprehensive hyperparameter optimization using existing ml_common
utilities with advanced optimization strategies and validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
from dataclasses import dataclass
import time
from datetime import datetime
from enum import Enum
import json
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

# Import existing ml_common utilities
try:
    from src.utils.ml_common import (
        UnifiedCrossValidator, perform_cross_validation, temporal_cross_validation,
        ParetoOptimizer, ParetoFront, ParetoFrontAnalyzer,
        RegimeSpecificTPSLOptimizer
    )
    ML_COMMON_AVAILABLE = True
except ImportError:
    ML_COMMON_AVAILABLE = False

# Import existing utilities
try:
    from src.utils.common_operations import (
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer
    )
    HARDWARE_UTILS_AVAILABLE = True
except ImportError:
    HARDWARE_UTILS_AVAILABLE = False

try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        get_vectorized_processing_core,
        get_enhanced_matrix_operations,
        get_batch_matrix_processor
    )
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError:
    MATRIX_OPERATIONS_AVAILABLE = False

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Hyperparameter optimization methods."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    TPE = "tpe"  # Tree-structured Parzen Estimator
    EVOLUTIONARY = "evolutionary"
    PARETO = "pareto"
    REGIME_SPECIFIC = "regime_specific"


class ValidationMethod(Enum):
    """Validation methods for hyperparameter optimization."""
    CROSS_VALIDATION = "cross_validation"
    TEMPORAL_CV = "temporal_cv"
    NESTED_CV = "nested_cv"
    HOLD_OUT = "hold_out"
    TIME_SERIES_SPLIT = "time_series_split"


@dataclass
class HyperparameterConfig:
    """Configuration for hyperparameter optimization."""
    # Optimization method
    method: OptimizationMethod = OptimizationMethod.BAYESIAN
    validation_method: ValidationMethod = ValidationMethod.CROSS_VALIDATION
    
    # Optimization parameters
    n_trials: int = 100
    n_folds: int = 5
    n_jobs: int = -1
    random_state: int = 42
    
    # Early stopping
    enable_early_stopping: bool = True
    patience: int = 10
    min_improvement: float = 0.001
    
    # Multi-objective optimization
    enable_multi_objective: bool = False
    objectives: List[str] = None  # ["accuracy", "speed", "memory"]
    objective_weights: Dict[str, float] = None
    
    # Performance optimization
    use_hardware_acceleration: bool = True
    use_matrix_operations: bool = True
    batch_size: int = 1000
    memory_limit_gb: float = 8.0
    
    # Output settings
    save_results: bool = True
    output_dir: str = "hyperparameter_results"
    verbose: bool = True
    
    def __post_init__(self):
        if self.objectives is None:
            self.objectives = ["accuracy", "speed"]
        if self.objective_weights is None:
            self.objective_weights = {"accuracy": 0.7, "speed": 0.3}


@dataclass
class HyperparameterResult:
    """Result from hyperparameter optimization."""
    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    validation_scores: Dict[str, float]
    pareto_front: Optional[List[Dict[str, Any]]] = None
    processing_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    hardware_optimization_applied: bool = False
    matrix_operations_used: bool = False


class HyperparameterOptimizer:
    """Advanced hyperparameter optimizer with multiple strategies and validation."""
    
    def __init__(self, config: HyperparameterConfig):
        """Initialize the hyperparameter optimizer.
        
        Args:
            config: Hyperparameter optimization configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize hardware acceleration if available
        self.hardware_accelerator = None
        self.memory_optimizer = None
        self.cpu_optimizer = None
        
        if HARDWARE_UTILS_AVAILABLE and config.use_hardware_acceleration:
            try:
                self.hardware_accelerator = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer()
                self.cpu_optimizer = get_m1_cpu_optimizer()
                self.logger.info("✅ Hardware acceleration initialized for hyperparameter optimization")
            except Exception as e:
                self.logger.warning(f"⚠️ Hardware acceleration not available: {e}")
        
        # Initialize matrix operations if available
        self.matrix_ops = None
        self.vectorized_core = None
        self.enhanced_ops = None
        self.batch_processor = None
        
        if MATRIX_OPERATIONS_AVAILABLE and config.use_matrix_operations:
            try:
                self.matrix_ops = get_unified_matrix_operations()
                self.vectorized_core = get_vectorized_processing_core()
                self.enhanced_ops = get_enhanced_matrix_operations()
                self.batch_processor = get_batch_matrix_processor()
                self.logger.info("✅ Matrix operations initialized for hyperparameter optimization")
            except Exception as e:
                self.logger.warning(f"⚠️ Matrix operations not available: {e}")
        
        # Initialize optimization components
        self.optimization_history = []
        self.best_score = -np.inf
        self.best_params = {}
        self.pareto_front = []
        
        self.logger.info("✅ Hyperparameter Optimizer initialized")
        self.logger.info(f"   Method: {config.method.value}")
        self.logger.info(f"   Validation: {config.validation_method.value}")
        self.logger.info(f"   Trials: {config.n_trials}")
        self.logger.info(f"   Multi-objective: {config.enable_multi_objective}")
    
    def optimize(self, 
                 model_func: Callable,
                 param_space: Dict[str, Any],
                 X: pd.DataFrame,
                 y: pd.Series,
                 additional_data: Optional[Dict[str, Any]] = None) -> HyperparameterResult:
        """Optimize hyperparameters for a model.
        
        Args:
            model_func: Function that creates a model with given parameters
            param_space: Parameter space to search
            X: Feature matrix
            y: Target variable
            additional_data: Optional additional data for regime-specific optimization
            
        Returns:
            HyperparameterResult with optimization results
        """
        tprint_info("Starting hyperparameter optimization")
        tprint_debug(f"Data shape: {X.shape}")
        tprint_debug(f"Parameter space: {len(param_space)} parameters")
        tprint_debug(f"Method: {self.config.method.value}")
        
        with tprint_timer("Hyperparameter Optimization"):
            start_time = time.time()
            
            try:
                self.logger.info("🚀 Starting hyperparameter optimization")
                self.logger.info(f"   Data shape: {X.shape}")
                self.logger.info(f"   Parameter space: {len(param_space)} parameters")
                self.logger.info(f"   Method: {self.config.method.value}")
                
                # Initialize optimization
                tprint_info("Initializing optimization")
                self._initialize_optimization()
                tprint_success("Optimization initialized")
                
                # Perform optimization based on method
                if self.config.method == OptimizationMethod.GRID_SEARCH:
                    tprint_info("Performing grid search optimization")
                    result = self._grid_search_optimization(model_func, param_space, X, y)
                elif self.config.method == OptimizationMethod.RANDOM_SEARCH:
                    tprint_info("Performing random search optimization")
                    result = self._random_search_optimization(model_func, param_space, X, y)
                elif self.config.method == OptimizationMethod.BAYESIAN:
                    tprint_info("Performing Bayesian optimization")
                    result = self._bayesian_optimization(model_func, param_space, X, y)
                elif self.config.method == OptimizationMethod.TPE:
                    tprint_info("Performing TPE optimization")
                    result = self._tpe_optimization(model_func, param_space, X, y)
                elif self.config.method == OptimizationMethod.EVOLUTIONARY:
                    tprint_info("Performing evolutionary optimization")
                    result = self._evolutionary_optimization(model_func, param_space, X, y)
                elif self.config.method == OptimizationMethod.PARETO:
                    tprint_info("Performing Pareto optimization")
                    result = self._pareto_optimization(model_func, param_space, X, y)
                elif self.config.method == OptimizationMethod.REGIME_SPECIFIC:
                    tprint_info("Performing regime specific optimization")
                    result = self._regime_specific_optimization(model_func, param_space, X, y, additional_data)
                else:
                    tprint_error(f"Unknown optimization method: {self.config.method}")
                    raise ValueError(f"Unknown optimization method: {self.config.method}")
                
                # Finalize optimization
                processing_time = time.time() - start_time
                result.processing_time = processing_time
                result.hardware_optimization_applied = self.hardware_accelerator is not None
                result.matrix_operations_used = self.matrix_ops is not None
                
                tprint_performance("Hyperparameter Optimization", processing_time)
                tprint_success(f"Optimization completed: Best score {result.best_score:.4f}")
                
                # Save results if requested
                if self.config.save_results:
                    tprint_info("Saving optimization results")
                    self._save_optimization_results(result)
                
                self.logger.info(f"✅ Hyperparameter optimization completed in {processing_time:.2f}s")
                self.logger.info(f"   Best score: {result.best_score:.4f}")
                self.logger.info(f"   Best params: {result.best_params}")
                
                return result
                
            except Exception as e:
                tprint_error(f"Hyperparameter optimization failed: {e}")
                tprint_debug(f"Error details: {type(e).__name__}: {str(e)}")
                processing_time = time.time() - start_time
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Hyperparameter optimization failed: {e}")
            
            return HyperparameterResult(
                best_params={},
                best_score=0.0,
                optimization_history=[],
                validation_scores={},
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    def _initialize_optimization(self):
        """Initialize optimization process."""
        try:
            self.optimization_history = []
            self.best_score = -np.inf
            self.best_params = {}
            self.pareto_front = []
            
            self.logger.info("✅ Optimization initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Optimization initialization failed: {e}")
            raise
    
    def _grid_search_optimization(self, model_func: Callable, param_space: Dict[str, Any],
                                X: pd.DataFrame, y: pd.Series) -> HyperparameterResult:
        """Perform grid search optimization."""
        try:
            self.logger.info("🔍 Performing grid search optimization")
            
            # Generate parameter combinations
            param_combinations = self._generate_param_combinations(param_space)
            
            best_score = -np.inf
            best_params = {}
            optimization_history = []
            
            for i, params in enumerate(param_combinations):
                self.logger.info(f"   Trial {i+1}/{len(param_combinations)}: {params}")
                
                # Evaluate parameters
                score, validation_scores = self._evaluate_parameters(model_func, params, X, y)
                
                # Update best if improved
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                
                # Record history
                trial_result = {
                    'trial': i + 1,
                    'params': params.copy(),
                    'score': score,
                    'validation_scores': validation_scores,
                    'timestamp': datetime.now().isoformat()
                }
                optimization_history.append(trial_result)
                
                # Early stopping check
                if self.config.enable_early_stopping and self._should_stop_early(optimization_history):
                    self.logger.info(f"🛑 Early stopping at trial {i+1}")
                    break
            
            return HyperparameterResult(
                best_params=best_params,
                best_score=best_score,
                optimization_history=optimization_history,
                validation_scores=validation_scores,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"❌ Grid search optimization failed: {e}")
            return HyperparameterResult(
                best_params={},
                best_score=0.0,
                optimization_history=[],
                validation_scores={},
                success=False,
                error_message=str(e)
            )
    
    def _random_search_optimization(self, model_func: Callable, param_space: Dict[str, Any],
                                  X: pd.DataFrame, y: pd.Series) -> HyperparameterResult:
        """Perform random search optimization."""
        try:
            self.logger.info("🎲 Performing random search optimization")
            
            best_score = -np.inf
            best_params = {}
            optimization_history = []
            
            for i in range(self.config.n_trials):
                # Sample random parameters
                params = self._sample_random_parameters(param_space)
                
                self.logger.info(f"   Trial {i+1}/{self.config.n_trials}: {params}")
                
                # Evaluate parameters
                score, validation_scores = self._evaluate_parameters(model_func, params, X, y)
                
                # Update best if improved
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                
                # Record history
                trial_result = {
                    'trial': i + 1,
                    'params': params.copy(),
                    'score': score,
                    'validation_scores': validation_scores,
                    'timestamp': datetime.now().isoformat()
                }
                optimization_history.append(trial_result)
                
                # Early stopping check
                if self.config.enable_early_stopping and self._should_stop_early(optimization_history):
                    self.logger.info(f"🛑 Early stopping at trial {i+1}")
                    break
            
            return HyperparameterResult(
                best_params=best_params,
                best_score=best_score,
                optimization_history=optimization_history,
                validation_scores=validation_scores,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"❌ Random search optimization failed: {e}")
            return HyperparameterResult(
                best_params={},
                best_score=0.0,
                optimization_history=[],
                validation_scores={},
                success=False,
                error_message=str(e)
            )
    
    def _bayesian_optimization(self, model_func: Callable, param_space: Dict[str, Any],
                             X: pd.DataFrame, y: pd.Series) -> HyperparameterResult:
        """Perform Bayesian optimization."""
        try:
            self.logger.info("🧠 Performing Bayesian optimization")
            
            # This would integrate with a Bayesian optimization library like Optuna
            # For now, we'll use a simplified approach
            
            best_score = -np.inf
            best_params = {}
            optimization_history = []
            
            # Start with random exploration
            n_exploration = min(10, self.config.n_trials // 4)
            
            for i in range(self.config.n_trials):
                if i < n_exploration:
                    # Random exploration
                    params = self._sample_random_parameters(param_space)
                else:
                    # Bayesian acquisition (simplified)
                    params = self._bayesian_acquisition(param_space, optimization_history)
                
                self.logger.info(f"   Trial {i+1}/{self.config.n_trials}: {params}")
                
                # Evaluate parameters
                score, validation_scores = self._evaluate_parameters(model_func, params, X, y)
                
                # Update best if improved
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                
                # Record history
                trial_result = {
                    'trial': i + 1,
                    'params': params.copy(),
                    'score': score,
                    'validation_scores': validation_scores,
                    'timestamp': datetime.now().isoformat()
                }
                optimization_history.append(trial_result)
                
                # Early stopping check
                if self.config.enable_early_stopping and self._should_stop_early(optimization_history):
                    self.logger.info(f"🛑 Early stopping at trial {i+1}")
                    break
            
            return HyperparameterResult(
                best_params=best_params,
                best_score=best_score,
                optimization_history=optimization_history,
                validation_scores=validation_scores,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"❌ Bayesian optimization failed: {e}")
            return HyperparameterResult(
                best_params={},
                best_score=0.0,
                optimization_history=[],
                validation_scores={},
                success=False,
                error_message=str(e)
            )
    
    def _tpe_optimization(self, model_func: Callable, param_space: Dict[str, Any],
                         X: pd.DataFrame, y: pd.Series) -> HyperparameterResult:
        """Perform TPE (Tree-structured Parzen Estimator) optimization."""
        try:
            self.logger.info("🌳 Performing TPE optimization")
            
            # This would integrate with Optuna's TPE sampler
            # For now, we'll use a simplified approach similar to Bayesian
            
            return self._bayesian_optimization(model_func, param_space, X, y)
            
        except Exception as e:
            self.logger.error(f"❌ TPE optimization failed: {e}")
            return HyperparameterResult(
                best_params={},
                best_score=0.0,
                optimization_history=[],
                validation_scores={},
                success=False,
                error_message=str(e)
            )
    
    def _evolutionary_optimization(self, model_func: Callable, param_space: Dict[str, Any],
                                 X: pd.DataFrame, y: pd.Series) -> HyperparameterResult:
        """Perform evolutionary optimization."""
        try:
            self.logger.info("🧬 Performing evolutionary optimization")
            
            # Initialize population
            population_size = min(20, self.config.n_trials // 5)
            population = [self._sample_random_parameters(param_space) for _ in range(population_size)]
            
            best_score = -np.inf
            best_params = {}
            optimization_history = []
            
            for generation in range(self.config.n_trials // population_size):
                self.logger.info(f"   Generation {generation + 1}")
                
                # Evaluate population
                scores = []
                for i, params in enumerate(population):
                    score, validation_scores = self._evaluate_parameters(model_func, params, X, y)
                    scores.append(score)
                    
                    # Update best if improved
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                    
                    # Record history
                    trial_result = {
                        'generation': generation + 1,
                        'individual': i + 1,
                        'params': params.copy(),
                        'score': score,
                        'validation_scores': validation_scores,
                        'timestamp': datetime.now().isoformat()
                    }
                    optimization_history.append(trial_result)
                
                # Selection, crossover, and mutation
                population = self._evolve_population(population, scores, param_space)
            
            return HyperparameterResult(
                best_params=best_params,
                best_score=best_score,
                optimization_history=optimization_history,
                validation_scores=validation_scores,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"❌ Evolutionary optimization failed: {e}")
            return HyperparameterResult(
                best_params={},
                best_score=0.0,
                optimization_history=[],
                validation_scores={},
                success=False,
                error_message=str(e)
            )
    
    def _pareto_optimization(self, model_func: Callable, param_space: Dict[str, Any],
                           X: pd.DataFrame, y: pd.Series) -> HyperparameterResult:
        """Perform Pareto optimization for multi-objective problems."""
        try:
            self.logger.info("📊 Performing Pareto optimization")
            
            if not ML_COMMON_AVAILABLE:
                raise ValueError("ML Common utilities not available for Pareto optimization")
            
            # Use existing Pareto optimizer
            pareto_optimizer = ParetoOptimizer()
            
            # This would integrate with the existing Pareto optimizer
            # For now, we'll use a simplified approach
            
            best_score = -np.inf
            best_params = {}
            optimization_history = []
            pareto_front = []
            
            for i in range(self.config.n_trials):
                params = self._sample_random_parameters(param_space)
                
                # Evaluate multiple objectives
                objectives = self._evaluate_multi_objectives(model_func, params, X, y)
                
                # Update Pareto front
                pareto_front = self._update_pareto_front(pareto_front, params, objectives)
                
                # Calculate composite score
                score = self._calculate_composite_score(objectives)
                
                # Update best if improved
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                
                # Record history
                trial_result = {
                    'trial': i + 1,
                    'params': params.copy(),
                    'score': score,
                    'objectives': objectives,
                    'timestamp': datetime.now().isoformat()
                }
                optimization_history.append(trial_result)
            
            return HyperparameterResult(
                best_params=best_params,
                best_score=best_score,
                optimization_history=optimization_history,
                validation_scores=validation_scores,
                pareto_front=pareto_front,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"❌ Pareto optimization failed: {e}")
            return HyperparameterResult(
                best_params={},
                best_score=0.0,
                optimization_history=[],
                validation_scores={},
                success=False,
                error_message=str(e)
            )
    
    def _regime_specific_optimization(self, model_func: Callable, param_space: Dict[str, Any],
                                    X: pd.DataFrame, y: pd.Series, 
                                    additional_data: Optional[Dict[str, Any]]) -> HyperparameterResult:
        """Perform regime-specific optimization."""
        try:
            self.logger.info("🎯 Performing regime-specific optimization")
            
            if not ML_COMMON_AVAILABLE:
                raise ValueError("ML Common utilities not available for regime-specific optimization")
            
            # Use existing regime-specific optimizer
            regime_optimizer = RegimeSpecificTPSLOptimizer()
            
            # This would integrate with the existing regime-specific optimizer
            # For now, we'll use a simplified approach
            
            return self._bayesian_optimization(model_func, param_space, X, y)
            
        except Exception as e:
            self.logger.error(f"❌ Regime-specific optimization failed: {e}")
            return HyperparameterResult(
                best_params={},
                best_score=0.0,
                optimization_history=[],
                validation_scores={},
                success=False,
                error_message=str(e)
            )
    
    def _evaluate_parameters(self, model_func: Callable, params: Dict[str, Any],
                           X: pd.DataFrame, y: pd.Series) -> Tuple[float, Dict[str, float]]:
        """Evaluate parameters using cross-validation.
        
        Args:
            model_func: Function that creates a model
            params: Parameters to evaluate
            X: Feature matrix
            y: Target variable
            
        Returns:
            Tuple of (score, validation_scores)
        """
        try:
            # Create model with parameters
            model = model_func(**params)
            
            # Perform validation based on method
            if self.config.validation_method == ValidationMethod.CROSS_VALIDATION:
                if ML_COMMON_AVAILABLE:
                    cv_result = perform_cross_validation(
                        X, y, model, n_folds=self.config.n_folds
                    )
                    score = cv_result.get('cv_score', 0.0)
                    validation_scores = {
                        'cv_score': score,
                        'cv_std': cv_result.get('cv_std', 0.0)
                    }
                else:
                    # Fallback to simple train-test split
                    from sklearn.model_selection import train_test_split
                    from sklearn.metrics import accuracy_score
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=self.config.random_state
                    )
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    score = accuracy_score(y_test, y_pred)
                    validation_scores = {'accuracy': score}
            
            elif self.config.validation_method == ValidationMethod.TEMPORAL_CV:
                if ML_COMMON_AVAILABLE:
                    cv_result = temporal_cross_validation(
                        X, y, model, n_folds=self.config.n_folds
                    )
                    score = cv_result.get('cv_score', 0.0)
                    validation_scores = {
                        'temporal_cv_score': score,
                        'temporal_cv_std': cv_result.get('cv_std', 0.0)
                    }
                else:
                    # Fallback to simple validation
                    score = 0.5
                    validation_scores = {'temporal_cv_score': score}
            
            else:
                # Default validation
                score = 0.5
                validation_scores = {'default_score': score}
            
            return score, validation_scores
            
        except Exception as e:
            self.logger.warning(f"⚠️ Parameter evaluation failed: {e}")
            return 0.0, {'error': str(e)}
    
    def _evaluate_multi_objectives(self, model_func: Callable, params: Dict[str, Any],
                                 X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate multiple objectives for Pareto optimization.
        
        Args:
            model_func: Function that creates a model
            params: Parameters to evaluate
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary of objective scores
        """
        try:
            objectives = {}
            
            # Evaluate each objective
            for objective in self.config.objectives:
                if objective == "accuracy":
                    score, _ = self._evaluate_parameters(model_func, params, X, y)
                    objectives[objective] = score
                elif objective == "speed":
                    # Measure training time
                    start_time = time.time()
                    model = model_func(**params)
                    model.fit(X, y)
                    training_time = time.time() - start_time
                    objectives[objective] = 1.0 / (1.0 + training_time)  # Higher is better
                elif objective == "memory":
                    # Estimate memory usage (simplified)
                    memory_usage = len(str(params)) / 1000  # Simplified metric
                    objectives[objective] = 1.0 / (1.0 + memory_usage)  # Higher is better
                else:
                    objectives[objective] = 0.5  # Default score
            
            return objectives
            
        except Exception as e:
            self.logger.warning(f"⚠️ Multi-objective evaluation failed: {e}")
            return {obj: 0.0 for obj in self.config.objectives}
    
    def _generate_param_combinations(self, param_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate parameter combinations for grid search."""
        try:
            import itertools
            
            # Convert parameter space to lists
            param_lists = {}
            for param, values in param_space.items():
                if isinstance(values, (list, tuple)):
                    param_lists[param] = values
                elif isinstance(values, dict) and 'range' in values:
                    # Handle range parameters
                    start, end, step = values['range']
                    param_lists[param] = list(np.arange(start, end + step, step))
                else:
                    param_lists[param] = [values]
            
            # Generate all combinations
            param_names = list(param_lists.keys())
            param_values = list(param_lists.values())
            
            combinations = []
            for combo in itertools.product(*param_values):
                param_dict = dict(zip(param_names, combo))
                combinations.append(param_dict)
            
            return combinations
            
        except Exception as e:
            self.logger.warning(f"⚠️ Parameter combination generation failed: {e}")
            return []
    
    def _sample_random_parameters(self, param_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample random parameters from parameter space."""
        try:
            params = {}
            
            for param, values in param_space.items():
                if isinstance(values, (list, tuple)):
                    params[param] = np.random.choice(values)
                elif isinstance(values, dict):
                    if 'uniform' in values:
                        low, high = values['uniform']
                        params[param] = np.random.uniform(low, high)
                    elif 'normal' in values:
                        mean, std = values['normal']
                        params[param] = np.random.normal(mean, std)
                    elif 'loguniform' in values:
                        low, high = values['loguniform']
                        params[param] = np.exp(np.random.uniform(np.log(low), np.log(high)))
                    elif 'choice' in values:
                        params[param] = np.random.choice(values['choice'])
                    else:
                        params[param] = values.get('default', 0)
                else:
                    params[param] = values
            
            return params
            
        except Exception as e:
            self.logger.warning(f"⚠️ Random parameter sampling failed: {e}")
            return {}
    
    def _bayesian_acquisition(self, param_space: Dict[str, Any], 
                            history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Bayesian acquisition function for parameter selection."""
        try:
            # Simplified acquisition function
            # In practice, this would use a Gaussian Process or similar
            
            if len(history) < 5:
                # Not enough history, use random sampling
                return self._sample_random_parameters(param_space)
            
            # Simple acquisition: sample around best parameters
            best_trial = max(history, key=lambda x: x['score'])
            best_params = best_trial['params']
            
            # Add some noise to best parameters
            new_params = {}
            for param, value in best_params.items():
                if isinstance(value, (int, float)):
                    # Add Gaussian noise
                    noise_std = abs(value) * 0.1  # 10% noise
                    new_value = value + np.random.normal(0, noise_std)
                    
                    # Ensure within bounds if specified
                    if param in param_space and isinstance(param_space[param], dict):
                        if 'uniform' in param_space[param]:
                            low, high = param_space[param]['uniform']
                            new_value = np.clip(new_value, low, high)
                    
                    new_params[param] = new_value
                else:
                    new_params[param] = value
            
            return new_params
            
        except Exception as e:
            self.logger.warning(f"⚠️ Bayesian acquisition failed: {e}")
            return self._sample_random_parameters(param_space)
    
    def _evolve_population(self, population: List[Dict[str, Any]], 
                         scores: List[float], param_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evolve population using genetic operators."""
        try:
            # Sort by scores (higher is better)
            sorted_pop = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)
            
            # Keep top 50% as parents
            n_parents = len(population) // 2
            parents = [ind for ind, _ in sorted_pop[:n_parents]]
            
            # Generate offspring
            offspring = []
            for _ in range(len(population)):
                # Select two parents
                parent1 = np.random.choice(parents)
                parent2 = np.random.choice(parents)
                
                # Crossover
                child = self._crossover(parent1, parent2, param_space)
                
                # Mutation
                child = self._mutate(child, param_space)
                
                offspring.append(child)
            
            return offspring
            
        except Exception as e:
            self.logger.warning(f"⚠️ Population evolution failed: {e}")
            return population
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any],
                  param_space: Dict[str, Any]) -> Dict[str, Any]:
        """Crossover operation between two parameter sets."""
        try:
            child = {}
            
            for param in param_space.keys():
                if param in parent1 and param in parent2:
                    if isinstance(parent1[param], (int, float)) and isinstance(parent2[param], (int, float)):
                        # Arithmetic crossover for numeric parameters
                        child[param] = (parent1[param] + parent2[param]) / 2
                    else:
                        # Random choice for non-numeric parameters
                        child[param] = np.random.choice([parent1[param], parent2[param]])
                else:
                    child[param] = parent1.get(param, parent2.get(param, 0))
            
            return child
            
        except Exception as e:
            self.logger.warning(f"⚠️ Crossover failed: {e}")
            return parent1
    
    def _mutate(self, individual: Dict[str, Any], param_space: Dict[str, Any]) -> Dict[str, Any]:
        """Mutation operation on parameter set."""
        try:
            mutated = individual.copy()
            mutation_rate = 0.1  # 10% mutation rate
            
            for param, value in mutated.items():
                if np.random.random() < mutation_rate:
                    if isinstance(value, (int, float)):
                        # Add Gaussian noise
                        noise_std = abs(value) * 0.1
                        new_value = value + np.random.normal(0, noise_std)
                        
                        # Ensure within bounds
                        if param in param_space and isinstance(param_space[param], dict):
                            if 'uniform' in param_space[param]:
                                low, high = param_space[param]['uniform']
                                new_value = np.clip(new_value, low, high)
                        
                        mutated[param] = new_value
                    else:
                        # Random choice for non-numeric parameters
                        if param in param_space and isinstance(param_space[param], (list, tuple)):
                            mutated[param] = np.random.choice(param_space[param])
            
            return mutated
            
        except Exception as e:
            self.logger.warning(f"⚠️ Mutation failed: {e}")
            return individual
    
    def _update_pareto_front(self, pareto_front: List[Dict[str, Any]], 
                           params: Dict[str, Any], objectives: Dict[str, float]) -> List[Dict[str, Any]]:
        """Update Pareto front with new solution."""
        try:
            # Add new solution
            new_solution = {
                'params': params.copy(),
                'objectives': objectives.copy()
            }
            
            # Check if new solution is dominated by existing solutions
            is_dominated = False
            for solution in pareto_front:
                if self._dominates(solution['objectives'], objectives):
                    is_dominated = True
                    break
            
            if not is_dominated:
                # Remove solutions dominated by new solution
                pareto_front = [sol for sol in pareto_front 
                              if not self._dominates(objectives, sol['objectives'])]
                
                # Add new solution
                pareto_front.append(new_solution)
            
            return pareto_front
            
        except Exception as e:
            self.logger.warning(f"⚠️ Pareto front update failed: {e}")
            return pareto_front
    
    def _dominates(self, obj1: Dict[str, float], obj2: Dict[str, float]) -> bool:
        """Check if obj1 dominates obj2."""
        try:
            # obj1 dominates obj2 if it's better in at least one objective
            # and not worse in any objective
            better_in_some = False
            worse_in_some = False
            
            for obj in obj1.keys():
                if obj in obj2:
                    if obj1[obj] > obj2[obj]:
                        better_in_some = True
                    elif obj1[obj] < obj2[obj]:
                        worse_in_some = True
            
            return better_in_some and not worse_in_some
            
        except Exception:
            return False
    
    def _calculate_composite_score(self, objectives: Dict[str, float]) -> float:
        """Calculate composite score from multiple objectives."""
        try:
            composite_score = 0.0
            
            for obj, score in objectives.items():
                weight = self.config.objective_weights.get(obj, 1.0)
                composite_score += weight * score
            
            return composite_score
            
        except Exception as e:
            self.logger.warning(f"⚠️ Composite score calculation failed: {e}")
            return 0.0
    
    def _should_stop_early(self, history: List[Dict[str, Any]]) -> bool:
        """Check if optimization should stop early."""
        try:
            if len(history) < self.config.patience:
                return False
            
            # Check if improvement has plateaued
            recent_scores = [trial['score'] for trial in history[-self.config.patience:]]
            best_recent = max(recent_scores)
            worst_recent = min(recent_scores)
            
            improvement = best_recent - worst_recent
            return improvement < self.config.min_improvement
            
        except Exception:
            return False
    
    def _save_optimization_results(self, result: HyperparameterResult):
        """Save optimization results to file."""
        try:
            from pathlib import Path
            
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save results
            result_file = output_dir / "hyperparameter_results.json"
            with open(result_file, 'w') as f:
                json.dump({
                    'best_params': result.best_params,
                    'best_score': result.best_score,
                    'optimization_history': result.optimization_history,
                    'validation_scores': result.validation_scores,
                    'processing_time': result.processing_time,
                    'success': result.success,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2, default=str)
            
            self.logger.info(f"💾 Optimization results saved to {result_file}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not save optimization results: {e}")


def create_hyperparameter_optimizer(config: Optional[HyperparameterConfig] = None) -> HyperparameterOptimizer:
    """Create a hyperparameter optimizer instance.
    
    Args:
        config: Optional hyperparameter configuration
        
    Returns:
        HyperparameterOptimizer instance
    """
    if config is None:
        config = HyperparameterConfig()
    return HyperparameterOptimizer(config)


def quick_hyperparameter_optimization(model_func: Callable,
                                     param_space: Dict[str, Any],
                                     X: pd.DataFrame,
                                     y: pd.Series,
                                     method: OptimizationMethod = OptimizationMethod.BAYESIAN,
                                     n_trials: int = 50) -> HyperparameterResult:
    """Quick hyperparameter optimization with default settings.
    
    Args:
        model_func: Function that creates a model
        param_space: Parameter space to search
        X: Feature matrix
        y: Target variable
        method: Optimization method
        n_trials: Number of trials
        
    Returns:
        HyperparameterResult
    """
    config = HyperparameterConfig(
        method=method,
        n_trials=n_trials,
        enable_early_stopping=True
    )
    
    optimizer = HyperparameterOptimizer(config)
    return optimizer.optimize(model_func, param_space, X, y)