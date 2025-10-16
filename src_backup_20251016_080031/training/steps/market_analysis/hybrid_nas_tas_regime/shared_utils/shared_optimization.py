"""
Shared Optimization Utilities for Hybrid NAS-TAS Regime Detection.

Provides common optimization utilities that can be used by both NAS and TAS systems
for architecture search, hyperparameter optimization, and performance tuning.
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

# Import existing utilities
try:
    from src.utils.ml_common import (
        ParetoOptimizer, ParetoFront, ParetoFrontAnalyzer,
        RegimeSpecificTPSLOptimizer
    )
    ML_COMMON_AVAILABLE = True
except ImportError:
    ML_COMMON_AVAILABLE = False

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


class OptimizationType(Enum):
    """Types of optimization available."""
    SINGLE_OBJECTIVE = "single_objective"
    MULTI_OBJECTIVE = "multi_objective"
    PARETO = "pareto"
    REGIME_SPECIFIC = "regime_specific"
    EVOLUTIONARY = "evolutionary"
    BAYESIAN = "bayesian"


class OptimizationAlgorithm(Enum):
    """Optimization algorithms available."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    TPE = "tpe"
    NSGA_II = "nsga_ii"
    SPEA2 = "spea2"
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"
    SIMULATED_ANNEALING = "simulated_annealing"


@dataclass
class SharedOptimizationConfig:
    """Configuration for shared optimization utilities."""
    # Optimization type
    optimization_type: OptimizationType = OptimizationType.SINGLE_OBJECTIVE
    algorithm: OptimizationAlgorithm = OptimizationAlgorithm.BAYESIAN_OPTIMIZATION
    
    # Optimization parameters
    n_trials: int = 100
    n_generations: int = 50
    population_size: int = 20
    n_folds: int = 5
    
    # Multi-objective parameters
    objectives: List[str] = None
    objective_weights: Dict[str, float] = None
    
    # Early stopping
    enable_early_stopping: bool = True
    patience: int = 10
    min_improvement: float = 0.001
    
    # Performance optimization
    use_hardware_acceleration: bool = True
    use_matrix_operations: bool = True
    batch_size: int = 1000
    memory_limit_gb: float = 8.0
    
    # Output settings
    save_results: bool = True
    output_dir: str = "optimization_results"
    verbose: bool = True
    
    def __post_init__(self):
        if self.objectives is None:
            self.objectives = ["accuracy", "speed"]
        if self.objective_weights is None:
            self.objective_weights = {"accuracy": 0.7, "speed": 0.3}


@dataclass
class SharedOptimizationResult:
    """Result from shared optimization."""
    best_solution: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    pareto_front: Optional[List[Dict[str, Any]]] = None
    convergence_metrics: Dict[str, float] = None
    processing_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    hardware_optimization_applied: bool = False
    matrix_operations_used: bool = False


class SharedOptimizer:
    """Shared optimizer for both NAS and TAS systems."""
    
    def __init__(self, config: SharedOptimizationConfig):
        """Initialize the shared optimizer.
        
        Args:
            config: Shared optimization configuration
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
                self.logger.info("✅ Hardware acceleration initialized for shared optimization")
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
                self.logger.info("✅ Matrix operations initialized for shared optimization")
            except Exception as e:
                self.logger.warning(f"⚠️ Matrix operations not available: {e}")
        
        # Initialize optimization components
        self.pareto_optimizer = None
        self.regime_optimizer = None
        
        if ML_COMMON_AVAILABLE:
            try:
                if config.optimization_type == OptimizationType.PARETO:
                    self.pareto_optimizer = ParetoOptimizer()
                elif config.optimization_type == OptimizationType.REGIME_SPECIFIC:
                    self.regime_optimizer = RegimeSpecificTPSLOptimizer()
                self.logger.info("✅ ML Common optimization components initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ ML Common optimization components not available: {e}")
        
        self.logger.info("✅ Shared Optimizer initialized")
        self.logger.info(f"   Type: {config.optimization_type.value}")
        self.logger.info(f"   Algorithm: {config.algorithm.value}")
        self.logger.info(f"   Trials: {config.n_trials}")
    
    def optimize(self, 
                 objective_func: Callable,
                 param_space: Dict[str, Any],
                 additional_data: Optional[Dict[str, Any]] = None) -> SharedOptimizationResult:
        """Optimize using the configured algorithm.
        
        Args:
            objective_func: Objective function to optimize
            param_space: Parameter space to search
            additional_data: Optional additional data for optimization
            
        Returns:
            SharedOptimizationResult with optimization results
        """
        tprint_info("Starting shared optimization")
        tprint_debug(f"Optimization type: {self.config.optimization_type.value}")
        tprint_debug(f"Algorithm: {self.config.algorithm.value}")
        tprint_debug(f"Parameter space: {len(param_space)} parameters")
        
        with tprint_timer("Shared Optimization"):
            start_time = time.time()
            
            try:
                self.logger.info("🚀 Starting shared optimization")
                self.logger.info(f"   Type: {self.config.optimization_type.value}")
                self.logger.info(f"   Algorithm: {self.config.algorithm.value}")
                self.logger.info(f"   Parameter space: {len(param_space)} parameters")
                
                # Perform optimization based on type and algorithm
                if self.config.optimization_type == OptimizationType.SINGLE_OBJECTIVE:
                    tprint_info("Performing single objective optimization")
                    result = self._single_objective_optimization(objective_func, param_space)
                elif self.config.optimization_type == OptimizationType.MULTI_OBJECTIVE:
                    tprint_info("Performing multi objective optimization")
                    result = self._multi_objective_optimization(objective_func, param_space)
                elif self.config.optimization_type == OptimizationType.PARETO:
                    tprint_info("Performing Pareto optimization")
                    result = self._pareto_optimization(objective_func, param_space)
                elif self.config.optimization_type == OptimizationType.REGIME_SPECIFIC:
                    tprint_info("Performing regime specific optimization")
                    result = self._regime_specific_optimization(objective_func, param_space, additional_data)
                elif self.config.optimization_type == OptimizationType.EVOLUTIONARY:
                    tprint_info("Performing evolutionary optimization")
                    result = self._evolutionary_optimization(objective_func, param_space)
                elif self.config.optimization_type == OptimizationType.BAYESIAN:
                    tprint_info("Performing Bayesian optimization")
                    result = self._bayesian_optimization(objective_func, param_space)
                else:
                    tprint_error(f"Unknown optimization type: {self.config.optimization_type}")
                    raise ValueError(f"Unknown optimization type: {self.config.optimization_type}")
                
                # Finalize optimization
                processing_time = time.time() - start_time
                result.processing_time = processing_time
                result.hardware_optimization_applied = self.hardware_accelerator is not None
                result.matrix_operations_used = self.matrix_ops is not None
                
                tprint_performance("Shared Optimization", processing_time)
                tprint_success(f"Optimization completed: Best score {result.best_score:.4f}")
                
                # Save results if requested
                if self.config.save_results:
                    tprint_info("Saving optimization results")
                    self._save_optimization_results(result)
                
                self.logger.info(f"✅ Shared optimization completed in {processing_time:.2f}s")
                self.logger.info(f"   Best score: {result.best_score:.4f}")
            self.logger.info(f"   Best solution: {result.best_solution}")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Shared optimization failed: {e}")
            
            return SharedOptimizationResult(
                best_solution={},
                best_score=0.0,
                optimization_history=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    def _single_objective_optimization(self, objective_func: Callable, param_space: Dict[str, Any]) -> SharedOptimizationResult:
        """Perform single-objective optimization."""
        try:
            self.logger.info("🎯 Performing single-objective optimization")
            
            best_score = -np.inf
            best_solution = {}
            optimization_history = []
            
            for trial in range(self.config.n_trials):
                # Sample parameters
                params = self._sample_parameters(param_space)
                
                # Evaluate objective
                score = objective_func(params)
                
                # Update best if improved
                if score > best_score:
                    best_score = score
                    best_solution = params.copy()
                
                # Record history
                trial_result = {
                    'trial': trial + 1,
                    'params': params.copy(),
                    'score': score,
                    'timestamp': datetime.now().isoformat()
                }
                optimization_history.append(trial_result)
                
                # Early stopping check
                if self.config.enable_early_stopping and self._should_stop_early(optimization_history):
                    self.logger.info(f"🛑 Early stopping at trial {trial + 1}")
                    break
            
            return SharedOptimizationResult(
                best_solution=best_solution,
                best_score=best_score,
                optimization_history=optimization_history,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"❌ Single-objective optimization failed: {e}")
            return SharedOptimizationResult(
                best_solution={},
                best_score=0.0,
                optimization_history=[],
                success=False,
                error_message=str(e)
            )
    
    def _multi_objective_optimization(self, objective_func: Callable, param_space: Dict[str, Any]) -> SharedOptimizationResult:
        """Perform multi-objective optimization."""
        try:
            self.logger.info("🎯 Performing multi-objective optimization")
            
            best_solution = {}
            best_score = 0.0
            optimization_history = []
            pareto_front = []
            
            for trial in range(self.config.n_trials):
                # Sample parameters
                params = self._sample_parameters(param_space)
                
                # Evaluate multiple objectives
                objectives = objective_func(params)
                
                # Calculate composite score
                composite_score = self._calculate_composite_score(objectives)
                
                # Update best if improved
                if composite_score > best_score:
                    best_score = composite_score
                    best_solution = params.copy()
                
                # Update Pareto front
                pareto_front = self._update_pareto_front(pareto_front, params, objectives)
                
                # Record history
                trial_result = {
                    'trial': trial + 1,
                    'params': params.copy(),
                    'objectives': objectives,
                    'composite_score': composite_score,
                    'timestamp': datetime.now().isoformat()
                }
                optimization_history.append(trial_result)
                
                # Early stopping check
                if self.config.enable_early_stopping and self._should_stop_early(optimization_history):
                    self.logger.info(f"🛑 Early stopping at trial {trial + 1}")
                    break
            
            return SharedOptimizationResult(
                best_solution=best_solution,
                best_score=best_score,
                optimization_history=optimization_history,
                pareto_front=pareto_front,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"❌ Multi-objective optimization failed: {e}")
            return SharedOptimizationResult(
                best_solution={},
                best_score=0.0,
                optimization_history=[],
                success=False,
                error_message=str(e)
            )
    
    def _pareto_optimization(self, objective_func: Callable, param_space: Dict[str, Any]) -> SharedOptimizationResult:
        """Perform Pareto optimization."""
        try:
            self.logger.info("📊 Performing Pareto optimization")
            
            if self.pareto_optimizer:
                # Use existing Pareto optimizer
                result = self.pareto_optimizer.optimize(objective_func, param_space)
                return SharedOptimizationResult(
                    best_solution=result.get('best_solution', {}),
                    best_score=result.get('best_score', 0.0),
                    optimization_history=result.get('optimization_history', []),
                    pareto_front=result.get('pareto_front', []),
                    success=True
                )
            else:
                # Fallback to multi-objective optimization
                return self._multi_objective_optimization(objective_func, param_space)
                
        except Exception as e:
            self.logger.error(f"❌ Pareto optimization failed: {e}")
            return SharedOptimizationResult(
                best_solution={},
                best_score=0.0,
                optimization_history=[],
                success=False,
                error_message=str(e)
            )
    
    def _regime_specific_optimization(self, objective_func: Callable, param_space: Dict[str, Any], 
                                   additional_data: Optional[Dict[str, Any]]) -> SharedOptimizationResult:
        """Perform regime-specific optimization."""
        try:
            self.logger.info("🎯 Performing regime-specific optimization")
            
            if self.regime_optimizer:
                # Use existing regime-specific optimizer
                result = self.regime_optimizer.optimize(objective_func, param_space, additional_data)
                return SharedOptimizationResult(
                    best_solution=result.get('best_solution', {}),
                    best_score=result.get('best_score', 0.0),
                    optimization_history=result.get('optimization_history', []),
                    success=True
                )
            else:
                # Fallback to single-objective optimization
                return self._single_objective_optimization(objective_func, param_space)
                
        except Exception as e:
            self.logger.error(f"❌ Regime-specific optimization failed: {e}")
            return SharedOptimizationResult(
                best_solution={},
                best_score=0.0,
                optimization_history=[],
                success=False,
                error_message=str(e)
            )
    
    def _evolutionary_optimization(self, objective_func: Callable, param_space: Dict[str, Any]) -> SharedOptimizationResult:
        """Perform evolutionary optimization."""
        try:
            self.logger.info("🧬 Performing evolutionary optimization")
            
            # Initialize population
            population = [self._sample_parameters(param_space) for _ in range(self.config.population_size)]
            
            best_solution = {}
            best_score = -np.inf
            optimization_history = []
            
            for generation in range(self.config.n_generations):
                self.logger.info(f"   Generation {generation + 1}/{self.config.n_generations}")
                
                # Evaluate population
                scores = []
                for individual in population:
                    score = objective_func(individual)
                    scores.append(score)
                    
                    # Update best if improved
                    if score > best_score:
                        best_score = score
                        best_solution = individual.copy()
                
                # Record generation results
                generation_result = {
                    'generation': generation + 1,
                    'best_score': max(scores),
                    'avg_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'timestamp': datetime.now().isoformat()
                }
                optimization_history.append(generation_result)
                
                # Selection, crossover, and mutation
                population = self._evolve_population(population, scores, param_space)
                
                # Early stopping check
                if self.config.enable_early_stopping and self._should_stop_early(optimization_history):
                    self.logger.info(f"🛑 Early stopping at generation {generation + 1}")
                    break
            
            return SharedOptimizationResult(
                best_solution=best_solution,
                best_score=best_score,
                optimization_history=optimization_history,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"❌ Evolutionary optimization failed: {e}")
            return SharedOptimizationResult(
                best_solution={},
                best_score=0.0,
                optimization_history=[],
                success=False,
                error_message=str(e)
            )
    
    def _bayesian_optimization(self, objective_func: Callable, param_space: Dict[str, Any]) -> SharedOptimizationResult:
        """Perform Bayesian optimization."""
        try:
            self.logger.info("🧠 Performing Bayesian optimization")
            
            best_solution = {}
            best_score = -np.inf
            optimization_history = []
            
            # Start with random exploration
            n_exploration = min(10, self.config.n_trials // 4)
            
            for trial in range(self.config.n_trials):
                if trial < n_exploration:
                    # Random exploration
                    params = self._sample_parameters(param_space)
                else:
                    # Bayesian acquisition
                    params = self._bayesian_acquisition(param_space, optimization_history)
                
                # Evaluate objective
                score = objective_func(params)
                
                # Update best if improved
                if score > best_score:
                    best_score = score
                    best_solution = params.copy()
                
                # Record history
                trial_result = {
                    'trial': trial + 1,
                    'params': params.copy(),
                    'score': score,
                    'timestamp': datetime.now().isoformat()
                }
                optimization_history.append(trial_result)
                
                # Early stopping check
                if self.config.enable_early_stopping and self._should_stop_early(optimization_history):
                    self.logger.info(f"🛑 Early stopping at trial {trial + 1}")
                    break
            
            return SharedOptimizationResult(
                best_solution=best_solution,
                best_score=best_score,
                optimization_history=optimization_history,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"❌ Bayesian optimization failed: {e}")
            return SharedOptimizationResult(
                best_solution={},
                best_score=0.0,
                optimization_history=[],
                success=False,
                error_message=str(e)
            )
    
    def _sample_parameters(self, param_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample parameters from parameter space."""
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
            self.logger.warning(f"⚠️ Parameter sampling failed: {e}")
            return {}
    
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
    
    def _bayesian_acquisition(self, param_space: Dict[str, Any], 
                            history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Bayesian acquisition function for parameter selection."""
        try:
            # Simplified acquisition function
            # In practice, this would use a Gaussian Process or similar
            
            if len(history) < 5:
                # Not enough history, use random sampling
                return self._sample_parameters(param_space)
            
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
            return self._sample_parameters(param_space)
    
    def _should_stop_early(self, history: List[Dict[str, Any]]) -> bool:
        """Check if optimization should stop early."""
        try:
            if len(history) < self.config.patience:
                return False
            
            # Check if improvement has plateaued
            recent_scores = [trial.get('score', trial.get('best_score', 0.0)) for trial in history[-self.config.patience:]]
            best_recent = max(recent_scores)
            worst_recent = min(recent_scores)
            
            improvement = best_recent - worst_recent
            return improvement < self.config.min_improvement
            
        except Exception:
            return False
    
    def _save_optimization_results(self, result: SharedOptimizationResult):
        """Save optimization results to file."""
        try:
            from pathlib import Path
            
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save results
            result_file = output_dir / "shared_optimization_results.json"
            with open(result_file, 'w') as f:
                json.dump({
                    'best_solution': result.best_solution,
                    'best_score': result.best_score,
                    'optimization_history': result.optimization_history,
                    'pareto_front': result.pareto_front,
                    'processing_time': result.processing_time,
                    'success': result.success,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2, default=str)
            
            self.logger.info(f"💾 Optimization results saved to {result_file}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not save optimization results: {e}")


def create_shared_optimizer(config: Optional[SharedOptimizationConfig] = None) -> SharedOptimizer:
    """Create a shared optimizer instance.
    
    Args:
        config: Optional shared optimization configuration
        
    Returns:
        SharedOptimizer instance
    """
    if config is None:
        config = SharedOptimizationConfig()
    return SharedOptimizer(config)


def quick_shared_optimization(objective_func: Callable,
                             param_space: Dict[str, Any],
                             optimization_type: OptimizationType = OptimizationType.SINGLE_OBJECTIVE,
                             n_trials: int = 50) -> SharedOptimizationResult:
    """Quick shared optimization with default settings.
    
    Args:
        objective_func: Objective function to optimize
        param_space: Parameter space to search
        optimization_type: Type of optimization
        n_trials: Number of trials
        
    Returns:
        SharedOptimizationResult
    """
    config = SharedOptimizationConfig(
        optimization_type=optimization_type,
        n_trials=n_trials,
        enable_early_stopping=True
    )
    
    optimizer = SharedOptimizer(config)
    return optimizer.optimize(objective_func, param_space)