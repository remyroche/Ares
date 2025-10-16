"""
Unified Search Algorithms for Hybrid NAS-TAS Regime System

This module provides unified search algorithms that combine the best of both
TAS (Tree Architecture Search) and NAS (Neural Architecture Search) approaches.
These algorithms are shared between all regime detection systems.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
import time
from datetime import datetime
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import random
from collections import defaultdict
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    from sklearn.model_selection import ParameterGrid
    BAYESIAN_OPTIMIZATION_AVAILABLE = True
except ImportError:
    BAYESIAN_OPTIMIZATION_AVAILABLE = False

try:
    from scipy.optimize import minimize
    SCIPY_OPTIMIZATION_AVAILABLE = True
except ImportError:
    SCIPY_OPTIMIZATION_AVAILABLE = False

logger = logging.getLogger(__name__)


class SearchAlgorithmType(Enum):
    """Types of search algorithms available."""
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    EVOLUTIONARY_ALGORITHM = "evolutionary_algorithm"
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    TREE_BASED_SEARCH = "tree_based_search"
    NEURAL_ARCHITECTURE_SEARCH = "neural_architecture_search"
    HYBRID_SEARCH = "hybrid_search"


@dataclass
class SearchResult:
    """Result from search algorithm."""
    best_solution: Any
    best_score: float
    search_history: List[Dict[str, Any]]
    convergence_info: Dict[str, Any]
    algorithm_used: str
    execution_time: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class ArchitectureCandidate:
    """Architecture candidate for search algorithms."""
    architecture_id: str
    parameters: Dict[str, Any]
    fitness_score: float
    complexity_score: float
    efficiency_score: float
    regime_accuracy: float
    economic_significance: float
    trading_viability: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedSearchAlgorithm(ABC):
    """Abstract base class for unified search algorithms."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the search algorithm.
        
        Args:
            config: Configuration dictionary
        """
        tprint_info("Initializing Unified Search Algorithm")
        tprint_debug(f"Configuration: {config}")
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.search_history = []
        self.best_solution = None
        self.best_score = -np.inf
        tprint_success("Unified Search Algorithm initialized successfully")
    
    @abstractmethod
    def search(self, 
               objective_function: Callable,
               parameter_space: Dict[str, Any],
               n_iterations: int = 100) -> SearchResult:
        """Perform search optimization.
        
        Args:
            objective_function: Function to optimize
            parameter_space: Parameter space definition
            n_iterations: Number of search iterations
            
        Returns:
            SearchResult with optimization results
        """
        pass
    
    def _evaluate_candidate(self, 
                          candidate: ArchitectureCandidate,
                          objective_function: Callable) -> float:
        """Evaluate a candidate architecture."""
        try:
            score = objective_function(candidate.parameters)
            candidate.fitness_score = score
            return score
        except Exception as e:
            self.logger.warning(f"Candidate evaluation failed: {e}")
            return 0.0
    
    def _record_search_step(self, 
                          iteration: int,
                          candidate: ArchitectureCandidate,
                          score: float):
        """Record a search step."""
        self.search_history.append({
            'iteration': iteration,
            'candidate_id': candidate.architecture_id,
            'score': score,
            'parameters': candidate.parameters.copy(),
            'timestamp': datetime.now().isoformat()
        })
        
        if score > self.best_score:
            self.best_score = score
            self.best_solution = candidate


class BayesianOptimizationSearch(UnifiedSearchAlgorithm):
    """Bayesian optimization search algorithm."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Bayesian optimization search."""
        super().__init__(config)
        
        if not BAYESIAN_OPTIMIZATION_AVAILABLE:
            self.logger.warning("⚠️ Bayesian optimization not available - scikit-learn required")
        
        self.n_initial_points = config.get('n_initial_points', 10)
        self.acquisition_function = config.get('acquisition_function', 'expected_improvement')
        self.exploration_weight = config.get('exploration_weight', 0.1)
        
        self.logger.info("✅ Bayesian Optimization Search initialized")
    
    def search(self, 
               objective_function: Callable,
               parameter_space: Dict[str, Any],
               n_iterations: int = 100) -> SearchResult:
        """Perform Bayesian optimization search."""
        try:
            self.logger.info("🔍 Starting Bayesian optimization search...")
            start_time = time.time()
            
            if not BAYESIAN_OPTIMIZATION_AVAILABLE:
                raise ImportError("Bayesian optimization requires scikit-learn")
            
            # Generate initial points
            initial_candidates = self._generate_initial_candidates(parameter_space)
            
            # Evaluate initial candidates
            for candidate in initial_candidates:
                score = self._evaluate_candidate(candidate, objective_function)
                self._record_search_step(len(self.search_history), candidate, score)
            
            # Bayesian optimization loop
            for iteration in range(self.n_initial_points, n_iterations):
                # Fit Gaussian Process
                gp = self._fit_gaussian_process()
                
                # Select next candidate using acquisition function
                next_candidate = self._select_next_candidate(gp, parameter_space)
                
                # Evaluate candidate
                score = self._evaluate_candidate(next_candidate, objective_function)
                self._record_search_step(iteration, next_candidate, score)
                
                # Check convergence
                if self._check_convergence():
                    self.logger.info(f"✅ Convergence reached at iteration {iteration}")
                    break
            
            execution_time = time.time() - start_time
            
            return SearchResult(
                best_solution=self.best_solution,
                best_score=self.best_score,
                search_history=self.search_history,
                convergence_info=self._get_convergence_info(),
                algorithm_used="bayesian_optimization",
                execution_time=execution_time,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Bayesian optimization failed: {e}")
            return SearchResult(
                best_solution=None,
                best_score=-np.inf,
                search_history=self.search_history,
                convergence_info={'error': str(e)},
                algorithm_used="bayesian_optimization",
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    def _generate_initial_candidates(self, parameter_space: Dict[str, Any]) -> List[ArchitectureCandidate]:
        """Generate initial candidates for Bayesian optimization."""
        candidates = []
        np.random.seed(self.config.get('random_state', 42))
        
        for i in range(self.n_initial_points):
            parameters = {}
            for param_name, param_config in parameter_space.items():
                if isinstance(param_config, dict):
                    if param_config['type'] == 'continuous':
                        min_val = param_config['min']
                        max_val = param_config['max']
                        parameters[param_name] = np.random.uniform(min_val, max_val)
                    elif param_config['type'] == 'discrete':
                        choices = param_config['choices']
                        parameters[param_name] = np.random.choice(choices)
                    elif param_config['type'] == 'integer':
                        min_val = param_config['min']
                        max_val = param_config['max']
                        parameters[param_name] = np.random.randint(min_val, max_val + 1)
                else:
                    # Simple range
                    if isinstance(param_config, (list, tuple)) and len(param_config) == 2:
                        min_val, max_val = param_config
                        parameters[param_name] = np.random.uniform(min_val, max_val)
                    else:
                        parameters[param_name] = param_config
            
            candidate = ArchitectureCandidate(
                architecture_id=f"bayesian_init_{i}",
                parameters=parameters,
                fitness_score=0.0,
                complexity_score=0.0,
                efficiency_score=0.0,
                regime_accuracy=0.0,
                economic_significance=0.0,
                trading_viability=0.0
            )
            candidates.append(candidate)
        
        return candidates
    
    def _fit_gaussian_process(self) -> Optional[GaussianProcessRegressor]:
        """Fit Gaussian Process to search history."""
        try:
            if len(self.search_history) < 2:
                return None
            
            # Extract parameters and scores
            X = []
            y = []
            
            for record in self.search_history:
                param_vector = []
                for param_name in sorted(record['parameters'].keys()):
                    param_vector.append(record['parameters'][param_name])
                X.append(param_vector)
                y.append(record['score'])
            
            X = np.array(X)
            y = np.array(y)
            
            # Create kernel
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
            
            # Fit Gaussian Process
            gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
            gp.fit(X, y)
            
            return gp
            
        except Exception as e:
            self.logger.warning(f"⚠️ Gaussian Process fitting failed: {e}")
            return None
    
    def _select_next_candidate(self, 
                             gp: Optional[GaussianProcessRegressor],
                             parameter_space: Dict[str, Any]) -> ArchitectureCandidate:
        """Select next candidate using acquisition function."""
        try:
            if gp is None:
                # Fallback to random selection
                candidates = self._generate_initial_candidates(parameter_space)
                return candidates[0]
            
            # Generate candidate points
            n_candidates = 1000
            candidates = self._generate_initial_candidates(parameter_space)
            candidates = candidates * (n_candidates // len(candidates) + 1)
            candidates = candidates[:n_candidates]
            
            # Evaluate acquisition function
            best_acquisition = -np.inf
            best_candidate = candidates[0]
            
            for candidate in candidates:
                # Convert to vector
                param_vector = []
                for param_name in sorted(candidate.parameters.keys()):
                    param_vector.append(candidate.parameters[param_name])
                param_vector = np.array(param_vector).reshape(1, -1)
                
                # Get GP predictions
                mean, std = gp.predict(param_vector, return_std=True)
                
                # Calculate acquisition function
                if self.acquisition_function == "expected_improvement":
                    acquisition = self._expected_improvement(mean, std)
                elif self.acquisition_function == "upper_confidence_bound":
                    acquisition = self._upper_confidence_bound(mean, std)
                else:
                    acquisition = mean[0]  # Fallback to mean
                
                if acquisition > best_acquisition:
                    best_acquisition = acquisition
                    best_candidate = candidate
            
            # Update candidate ID
            best_candidate.architecture_id = f"bayesian_opt_{len(self.search_history)}"
            return best_candidate
            
        except Exception as e:
            self.logger.warning(f"⚠️ Next candidate selection failed: {e}")
            candidates = self._generate_initial_candidates(parameter_space)
            return candidates[0]
    
    def _expected_improvement(self, mean: np.ndarray, std: np.ndarray) -> float:
        """Calculate expected improvement acquisition function."""
        try:
            if not self.search_history:
                return mean[0]
            
            # Get best score so far
            best_score = max(record['score'] for record in self.search_history)
            
            # Calculate expected improvement
            improvement = mean[0] - best_score
            z = improvement / (std[0] + 1e-8)
            
            # Expected improvement formula
            from scipy.stats import norm
            ei = improvement * norm.cdf(z) + std[0] * norm.pdf(z)
            
            return ei
            
        except Exception:
            return mean[0]
    
    def _upper_confidence_bound(self, mean: np.ndarray, std: np.ndarray) -> float:
        """Calculate upper confidence bound acquisition function."""
        try:
            return mean[0] + self.exploration_weight * std[0]
        except Exception:
            return mean[0]
    
    def _check_convergence(self) -> bool:
        """Check if optimization has converged."""
        try:
            if len(self.search_history) < 10:
                return False
            
            # Check if improvement is below threshold
            recent_scores = [record['score'] for record in self.search_history[-10:]]
            score_std = np.std(recent_scores)
            
            convergence_threshold = self.config.get('convergence_threshold', 1e-6)
            return score_std < convergence_threshold
            
        except Exception:
            return False
    
    def _get_convergence_info(self) -> Dict[str, Any]:
        """Get convergence information."""
        return {
            'total_iterations': len(self.search_history),
            'convergence_reached': len(self.search_history) < self.config.get('max_iterations', 100),
            'score_improvement': self.best_score - self.search_history[0]['score'] if self.search_history else 0.0
        }


class EvolutionaryAlgorithmSearch(UnifiedSearchAlgorithm):
    """Evolutionary algorithm search."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize evolutionary algorithm search."""
        super().__init__(config)
        
        self.population_size = config.get('population_size', 50)
        self.mutation_rate = config.get('mutation_rate', 0.1)
        self.crossover_rate = config.get('crossover_rate', 0.8)
        self.tournament_size = config.get('tournament_size', 3)
        self.elite_size = config.get('elite_size', 5)
        
        self.logger.info("✅ Evolutionary Algorithm Search initialized")
    
    def search(self, 
               objective_function: Callable,
               parameter_space: Dict[str, Any],
               n_iterations: int = 100) -> SearchResult:
        """Perform evolutionary algorithm search."""
        try:
            self.logger.info("🔍 Starting evolutionary algorithm search...")
            start_time = time.time()
            
            # Initialize population
            population = self._initialize_population(parameter_space)
            
            # Evaluate initial population
            for candidate in population:
                score = self._evaluate_candidate(candidate, objective_function)
                self._record_search_step(len(self.search_history), candidate, score)
            
            # Evolution loop
            for generation in range(n_iterations):
                # Select parents
                parents = self._tournament_selection(population)
                
                # Create offspring
                offspring = []
                for i in range(0, len(parents), 2):
                    if i + 1 < len(parents):
                        parent1 = parents[i]
                        parent2 = parents[i + 1]
                        
                        # Crossover
                        if random.random() < self.crossover_rate:
                            child1, child2 = self._crossover(parent1, parent2, parameter_space)
                        else:
                            child1, child2 = parent1, parent2
                        
                        # Mutation
                        child1 = self._mutate(child1, parameter_space)
                        child2 = self._mutate(child2, parameter_space)
                        
                        offspring.extend([child1, child2])
                
                # Evaluate offspring
                for candidate in offspring:
                    score = self._evaluate_candidate(candidate, objective_function)
                    self._record_search_step(len(self.search_history), candidate, score)
                
                # Create next generation
                population = self._create_next_generation(population, offspring)
                
                # Check convergence
                if self._check_convergence():
                    self.logger.info(f"✅ Convergence reached at generation {generation}")
                    break
            
            execution_time = time.time() - start_time
            
            return SearchResult(
                best_solution=self.best_solution,
                best_score=self.best_score,
                search_history=self.search_history,
                convergence_info=self._get_convergence_info(),
                algorithm_used="evolutionary_algorithm",
                execution_time=execution_time,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Evolutionary algorithm failed: {e}")
            return SearchResult(
                best_solution=None,
                best_score=-np.inf,
                search_history=self.search_history,
                convergence_info={'error': str(e)},
                algorithm_used="evolutionary_algorithm",
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    def _initialize_population(self, parameter_space: Dict[str, Any]) -> List[ArchitectureCandidate]:
        """Initialize population of candidates."""
        population = []
        np.random.seed(self.config.get('random_state', 42))
        
        for i in range(self.population_size):
            parameters = {}
            for param_name, param_config in parameter_space.items():
                if isinstance(param_config, dict):
                    if param_config['type'] == 'continuous':
                        min_val = param_config['min']
                        max_val = param_config['max']
                        parameters[param_name] = np.random.uniform(min_val, max_val)
                    elif param_config['type'] == 'discrete':
                        choices = param_config['choices']
                        parameters[param_name] = np.random.choice(choices)
                    elif param_config['type'] == 'integer':
                        min_val = param_config['min']
                        max_val = param_config['max']
                        parameters[param_name] = np.random.randint(min_val, max_val + 1)
                else:
                    # Simple range
                    if isinstance(param_config, (list, tuple)) and len(param_config) == 2:
                        min_val, max_val = param_config
                        parameters[param_name] = np.random.uniform(min_val, max_val)
                    else:
                        parameters[param_name] = param_config
            
            candidate = ArchitectureCandidate(
                architecture_id=f"evo_init_{i}",
                parameters=parameters,
                fitness_score=0.0,
                complexity_score=0.0,
                efficiency_score=0.0,
                regime_accuracy=0.0,
                economic_significance=0.0,
                trading_viability=0.0
            )
            population.append(candidate)
        
        return population
    
    def _tournament_selection(self, population: List[ArchitectureCandidate]) -> List[ArchitectureCandidate]:
        """Tournament selection for parent selection."""
        parents = []
        
        for _ in range(self.population_size):
            # Random tournament
            tournament = random.sample(population, min(self.tournament_size, len(population)))
            winner = max(tournament, key=lambda x: x.fitness_score)
            parents.append(winner)
        
        return parents
    
    def _crossover(self, 
                  parent1: ArchitectureCandidate,
                  parent2: ArchitectureCandidate,
                  parameter_space: Dict[str, Any]) -> Tuple[ArchitectureCandidate, ArchitectureCandidate]:
        """Crossover operation between two candidates."""
        try:
            # Simple crossover: combine parameters from both parents
            child1_params = {}
            child2_params = {}
            
            for param_name in parameter_space.keys():
                if random.random() < 0.5:
                    child1_params[param_name] = parent1.parameters[param_name]
                    child2_params[param_name] = parent2.parameters[param_name]
                else:
                    child1_params[param_name] = parent2.parameters[param_name]
                    child2_params[param_name] = parent1.parameters[param_name]
            
            # Create child candidates
            child1 = ArchitectureCandidate(
                architecture_id=f"evo_child_{len(self.search_history)}_1",
                parameters=child1_params,
                fitness_score=0.0,
                complexity_score=0.0,
                efficiency_score=0.0,
                regime_accuracy=0.0,
                economic_significance=0.0,
                trading_viability=0.0
            )
            
            child2 = ArchitectureCandidate(
                architecture_id=f"evo_child_{len(self.search_history)}_2",
                parameters=child2_params,
                fitness_score=0.0,
                complexity_score=0.0,
                efficiency_score=0.0,
                regime_accuracy=0.0,
                economic_significance=0.0,
                trading_viability=0.0
            )
            
            return child1, child2
            
        except Exception as e:
            self.logger.warning(f"Crossover failed: {e}")
            return parent1, parent2
    
    def _mutate(self, 
               candidate: ArchitectureCandidate,
               parameter_space: Dict[str, Any]) -> ArchitectureCandidate:
        """Mutation operation on candidate."""
        try:
            mutated_params = candidate.parameters.copy()
            
            for param_name, param_config in parameter_space.items():
                if random.random() < self.mutation_rate:
                    if isinstance(param_config, dict):
                        if param_config['type'] == 'continuous':
                            min_val = param_config['min']
                            max_val = param_config['max']
                            # Gaussian mutation
                            noise = np.random.normal(0, (max_val - min_val) * 0.1)
                            mutated_params[param_name] = np.clip(
                                mutated_params[param_name] + noise, min_val, max_val
                            )
                        elif param_config['type'] == 'discrete':
                            choices = param_config['choices']
                            mutated_params[param_name] = np.random.choice(choices)
                        elif param_config['type'] == 'integer':
                            min_val = param_config['min']
                            max_val = param_config['max']
                            mutated_params[param_name] = np.random.randint(min_val, max_val + 1)
                    else:
                        # Simple range mutation
                        if isinstance(param_config, (list, tuple)) and len(param_config) == 2:
                            min_val, max_val = param_config
                            noise = np.random.normal(0, (max_val - min_val) * 0.1)
                            mutated_params[param_name] = np.clip(
                                mutated_params[param_name] + noise, min_val, max_val
                            )
            
            # Create mutated candidate
            mutated_candidate = ArchitectureCandidate(
                architecture_id=f"evo_mutated_{len(self.search_history)}",
                parameters=mutated_params,
                fitness_score=0.0,
                complexity_score=0.0,
                efficiency_score=0.0,
                regime_accuracy=0.0,
                economic_significance=0.0,
                trading_viability=0.0
            )
            
            return mutated_candidate
            
        except Exception as e:
            self.logger.warning(f"Mutation failed: {e}")
            return candidate
    
    def _create_next_generation(self, 
                              population: List[ArchitectureCandidate],
                              offspring: List[ArchitectureCandidate]) -> List[ArchitectureCandidate]:
        """Create next generation combining population and offspring."""
        try:
            # Combine population and offspring
            combined = population + offspring
            
            # Sort by fitness
            combined.sort(key=lambda x: x.fitness_score, reverse=True)
            
            # Keep elite and select rest
            next_generation = combined[:self.elite_size]
            
            # Select remaining from combined population
            remaining = combined[self.elite_size:]
            if len(remaining) > self.population_size - self.elite_size:
                # Tournament selection for remaining
                for _ in range(self.population_size - self.elite_size):
                    tournament = random.sample(remaining, min(self.tournament_size, len(remaining)))
                    winner = max(tournament, key=lambda x: x.fitness_score)
                    next_generation.append(winner)
            else:
                next_generation.extend(remaining)
            
            return next_generation[:self.population_size]
            
        except Exception as e:
            self.logger.warning(f"Next generation creation failed: {e}")
            return population
    
    def _check_convergence(self) -> bool:
        """Check if evolution has converged."""
        try:
            if len(self.search_history) < 20:
                return False
            
            # Check if improvement is below threshold
            recent_scores = [record['score'] for record in self.search_history[-20:]]
            score_std = np.std(recent_scores)
            
            convergence_threshold = self.config.get('convergence_threshold', 1e-6)
            return score_std < convergence_threshold
            
        except Exception:
            return False
    
    def _get_convergence_info(self) -> Dict[str, Any]:
        """Get convergence information."""
        return {
            'total_generations': len(self.search_history),
            'convergence_reached': len(self.search_history) < self.config.get('max_iterations', 100),
            'score_improvement': self.best_score - self.search_history[0]['score'] if self.search_history else 0.0
        }


class UnifiedSearchManager:
    """Manager for coordinating different search algorithms."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the search manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize search algorithms
        self.search_algorithms = {}
        
        if config.get('enable_bayesian_optimization', True) and BAYESIAN_OPTIMIZATION_AVAILABLE:
            self.search_algorithms['bayesian'] = BayesianOptimizationSearch(config)
        
        if config.get('enable_evolutionary_algorithm', True):
            self.search_algorithms['evolutionary'] = EvolutionaryAlgorithmSearch(config)
        
        self.logger.info("✅ Unified Search Manager initialized")
        self.logger.info(f"   Available algorithms: {list(self.search_algorithms.keys())}")
    
    def search_with_algorithm(self, 
                            algorithm_type: str,
                            objective_function: Callable,
                            parameter_space: Dict[str, Any],
                            n_iterations: int = 100) -> SearchResult:
        """Search using specified algorithm.
        
        Args:
            algorithm_type: Type of algorithm to use
            objective_function: Function to optimize
            parameter_space: Parameter space definition
            n_iterations: Number of iterations
            
        Returns:
            SearchResult with optimization results
        """
        try:
            if algorithm_type not in self.search_algorithms:
                raise ValueError(f"Algorithm {algorithm_type} not available")
            
            self.logger.info(f"🔍 Starting search with {algorithm_type} algorithm...")
            
            algorithm = self.search_algorithms[algorithm_type]
            result = algorithm.search(objective_function, parameter_space, n_iterations)
            
            self.logger.info(f"✅ Search completed with {algorithm_type}")
            self.logger.info(f"   Best score: {result.best_score:.4f}")
            self.logger.info(f"   Execution time: {result.execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Search with {algorithm_type} failed: {e}")
            return SearchResult(
                best_solution=None,
                best_score=-np.inf,
                search_history=[],
                convergence_info={'error': str(e)},
                algorithm_used=algorithm_type,
                execution_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    def compare_algorithms(self, 
                         objective_function: Callable,
                         parameter_space: Dict[str, Any],
                         n_iterations: int = 100) -> Dict[str, SearchResult]:
        """Compare different search algorithms.
        
        Args:
            objective_function: Function to optimize
            parameter_space: Parameter space definition
            n_iterations: Number of iterations
            
        Returns:
            Dictionary mapping algorithm names to results
        """
        try:
            results = {}
            
            for algorithm_name, algorithm in self.search_algorithms.items():
                self.logger.info(f"🔍 Running {algorithm_name} algorithm...")
                result = algorithm.search(objective_function, parameter_space, n_iterations)
                results[algorithm_name] = result
            
            # Compare results
            if results:
                best_algorithm = max(results.keys(), key=lambda k: results[k].best_score)
                self.logger.info(f"✅ Best algorithm: {best_algorithm} (score: {results[best_algorithm].best_score:.4f})")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Algorithm comparison failed: {e}")
            return {}
    
    def get_available_algorithms(self) -> List[str]:
        """Get list of available algorithms."""
        return list(self.search_algorithms.keys())


def create_unified_search_manager(config: Dict[str, Any]) -> UnifiedSearchManager:
    """Create a unified search manager instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        UnifiedSearchManager instance
    """
    return UnifiedSearchManager(config)


def create_search_algorithm(algorithm_type: str, config: Dict[str, Any]) -> UnifiedSearchAlgorithm:
    """Create a specific search algorithm instance.
    
    Args:
        algorithm_type: Type of algorithm to create
        config: Configuration dictionary
        
    Returns:
        UnifiedSearchAlgorithm instance
    """
    if algorithm_type == "bayesian_optimization":
        return BayesianOptimizationSearch(config)
    elif algorithm_type == "evolutionary_algorithm":
        return EvolutionaryAlgorithmSearch(config)
    else:
        raise ValueError(f"Unknown algorithm type: {algorithm_type}")