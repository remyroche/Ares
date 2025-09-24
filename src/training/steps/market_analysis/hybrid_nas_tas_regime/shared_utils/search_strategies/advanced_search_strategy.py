"""
Advanced Search Strategy for Regime Detection Systems.

This module provides advanced search strategies that combine multiple approaches
for both NAS and TAS regime detection systems.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from src.utils.logger import system_logger


class SearchStrategyType(Enum):
    """Types of search strategies available."""
    RANDOM = "random"
    BAYESIAN = "bayesian"
    EVOLUTIONARY = "evolutionary"
    REINFORCEMENT = "reinforcement"
    META_LEARNING = "meta_learning"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


@dataclass
class SearchConfig:
    """Configuration for search strategies."""
    strategy_type: SearchStrategyType = SearchStrategyType.HYBRID
    population_size: int = 50
    max_generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 5
    tournament_size: int = 3
    convergence_threshold: float = 0.01
    max_stagnation_generations: int = 10
    enable_parallel_evaluation: bool = True
    max_parallel_workers: int = 4


@dataclass
class SearchResult:
    """Result of a search operation."""
    best_solution: Any
    best_score: float
    search_history: List[Dict[str, Any]]
    convergence_info: Dict[str, Any]
    execution_time: float
    success: bool
    error_message: Optional[str] = None


class AdvancedSearchStrategy:
    """
    Advanced search strategy combining multiple approaches.

    This class provides sophisticated search strategies that can be used by
    both NAS and TAS systems for finding optimal regimes and architectures.
    """

    def __init__(self, config: SearchConfig):
        """
        Initialize the advanced search strategy.

        Args:
            config: Search configuration
        """
        self.logger = system_logger.getChild('AdvancedSearchStrategy')
        self.config = config

        # Initialize search components
        self._initialize_search_components()

        # Search state
        self.population = []
        self.best_solution = None
        self.best_score = float('-inf')
        self.generation = 0
        self.search_history = []

        self.logger.info(f"✅ Advanced Search Strategy initialized: {config.strategy_type.value}")

    def _initialize_search_components(self):
        """Initialize search strategy components."""
        try:
            self.search_components = {}

            # Initialize Bayesian optimizer
            if self.config.strategy_type in [SearchStrategyType.BAYESIAN, SearchStrategyType.HYBRID]:
                from ..optimization.bayesian_optimizer import BayesianOptimizer
                self.search_components['bayesian'] = BayesianOptimizer()

            # Initialize evolutionary optimizer
            if self.config.strategy_type in [SearchStrategyType.EVOLUTIONARY, SearchStrategyType.HYBRID]:
                from ..optimization.evolutionary_optimizer import EvolutionaryOptimizer
                self.search_components['evolutionary'] = EvolutionaryOptimizer(self.config)

            # Initialize reinforcement learning optimizer
            if self.config.strategy_type in [SearchStrategyType.REINFORCEMENT, SearchStrategyType.HYBRID]:
                from ..optimization.reinforcement_optimizer import ReinforcementOptimizer
                self.search_components['reinforcement'] = ReinforcementOptimizer()

            # Initialize meta-learning component
            if self.config.strategy_type in [SearchStrategyType.META_LEARNING, SearchStrategyType.HYBRID]:
                from ..optimization.meta_learning_optimizer import MetaLearningOptimizer
                self.search_components['meta_learning'] = MetaLearningOptimizer()

            self.logger.info(f"✅ Search components initialized: {list(self.search_components.keys())}")

        except Exception as e:
            self.logger.error(f"❌ Search components initialization failed: {e}")

    def search(self,
               objective_function: Callable,
               search_space: Dict[str, Any],
               constraints: Optional[Dict[str, Any]] = None) -> SearchResult:
        """
        Perform advanced search using configured strategy.

        Args:
            objective_function: Function to evaluate solutions
            search_space: Definition of search space
            constraints: Optional constraints on solutions

        Returns:
            SearchResult with best solution found
        """
        try:
            self.logger.info(f"🔍 Starting {self.config.strategy_type.value} search")
            self.logger.info(f"   Population size: {self.config.population_size}")
            self.logger.info(f"   Max generations: {self.config.max_generations}")

            # Initialize search
            self._initialize_search(objective_function, search_space, constraints)

            # Perform search based on strategy
            if self.config.strategy_type == SearchStrategyType.RANDOM:
                result = self._random_search(objective_function)
            elif self.config.strategy_type == SearchStrategyType.BAYESIAN:
                result = self._bayesian_search(objective_function)
            elif self.config.strategy_type == SearchStrategyType.EVOLUTIONARY:
                result = self._evolutionary_search(objective_function)
            elif self.config.strategy_type == SearchStrategyType.REINFORCEMENT:
                result = self._reinforcement_search(objective_function)
            elif self.config.strategy_type == SearchStrategyType.META_LEARNING:
                result = self._meta_learning_search(objective_function)
            elif self.config.strategy_type == SearchStrategyType.HYBRID:
                result = self._hybrid_search(objective_function)
            elif self.config.strategy_type == SearchStrategyType.ADAPTIVE:
                result = self._adaptive_search(objective_function)
            else:
                raise ValueError(f"Unknown search strategy: {self.config.strategy_type}")

            self.logger.info(f"✅ Search completed: best_score={result.best_score".4f"}")
            return result

        except Exception as e:
            self.logger.error(f"❌ Search failed: {e}")
            return SearchResult(
                best_solution=None,
                best_score=float('-inf'),
                search_history=[],
                convergence_info={},
                execution_time=0.0,
                success=False,
                error_message=str(e)
            )

    def _initialize_search(self,
                          objective_function: Callable,
                          search_space: Dict[str, Any],
                          constraints: Optional[Dict[str, Any]]):
        """
        Initialize search with population and parameters.

        Args:
            objective_function: Function to evaluate solutions
            search_space: Definition of search space
            constraints: Optional constraints
        """
        try:
            # Create initial population
            self.population = []
            for i in range(self.config.population_size):
                solution = self._create_random_solution(search_space)
                if constraints:
                    solution = self._apply_constraints(solution, constraints)
                self.population.append(solution)

            # Evaluate initial population
            self._evaluate_population(objective_function)

            # Initialize best solution
            if self.population:
                self.best_solution = max(self.population, key=lambda x: x.get('fitness', float('-inf')))
                self.best_score = self.best_solution.get('fitness', float('-inf'))

            self.generation = 0
            self.search_history = []

        except Exception as e:
            self.logger.error(f"❌ Search initialization failed: {e}")

    def _create_random_solution(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a random solution within the search space.

        Args:
            search_space: Definition of search space

        Returns:
            Random solution dictionary
        """
        try:
            solution = {}

            for param_name, param_config in search_space.items():
                if isinstance(param_config, dict):
                    # Parameter with range specification
                    param_type = param_config.get('type', 'continuous')
                    param_range = param_config.get('range', [0, 1])

                    if param_type == 'continuous':
                        value = np.random.uniform(param_range[0], param_range[1])
                    elif param_type == 'integer':
                        value = np.random.randint(param_range[0], param_range[1] + 1)
                    elif param_type == 'categorical':
                        choices = param_config.get('choices', param_range)
                        value = np.random.choice(choices)
                    else:
                        value = np.random.uniform(param_range[0], param_range[1])
                else:
                    # Simple range specification
                    if isinstance(param_config, (list, tuple)) and len(param_config) == 2:
                        value = np.random.uniform(param_config[0], param_config[1])
                    else:
                        value = param_config  # Fixed value

                solution[param_name] = value

            return solution

        except Exception as e:
            self.logger.warning(f"⚠️ Random solution creation failed: {e}")
            return {}

    def _apply_constraints(self, solution: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply constraints to a solution.

        Args:
            solution: Solution to constrain
            constraints: Constraint definitions

        Returns:
            Constrained solution
        """
        try:
            constrained_solution = solution.copy()

            for constraint_name, constraint_config in constraints.items():
                constraint_type = constraint_config.get('type', 'inequality')

                if constraint_type == 'inequality':
                    # Inequality constraint: g(x) <= 0
                    constraint_func = constraint_config.get('function')
                    if constraint_func:
                        while not constraint_func(constrained_solution):
                            # Modify solution to satisfy constraint
                            constrained_solution = self._repair_solution(constrained_solution, constraint_config)
                elif constraint_type == 'equality':
                    # Equality constraint: h(x) = 0
                    constraint_func = constraint_config.get('function')
                    if constraint_func:
                        constrained_solution = self._satisfy_equality_constraint(
                            constrained_solution, constraint_func, constraint_config
                        )

            return constrained_solution

        except Exception as e:
            self.logger.warning(f"⚠️ Constraint application failed: {e}")
            return solution

    def _repair_solution(self, solution: Dict[str, Any], constraint_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Repair a solution that violates constraints.

        Args:
            solution: Solution to repair
            constraint_config: Constraint configuration

        Returns:
            Repaired solution
        """
        try:
            # Simple repair: randomly modify parameters until constraint satisfied
            max_attempts = 10
            attempts = 0

            while attempts < max_attempts:
                # Randomly select parameter to modify
                param_name = np.random.choice(list(solution.keys()))
                param_value = solution[param_name]

                # Apply random perturbation
                if isinstance(param_value, (int, float)):
                    perturbation = np.random.normal(0, abs(param_value) * 0.1)
                    solution[param_name] = max(0, param_value + perturbation)
                elif isinstance(param_value, str):
                    # For categorical parameters, change to different value
                    pass  # Skip for now

                attempts += 1

            return solution

        except Exception as e:
            self.logger.warning(f"⚠️ Solution repair failed: {e}")
            return solution

    def _satisfy_equality_constraint(self,
                                   solution: Dict[str, Any],
                                   constraint_func: Callable,
                                   constraint_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modify solution to satisfy equality constraint.

        Args:
            solution: Solution to modify
            constraint_func: Constraint function
            constraint_config: Constraint configuration

        Returns:
            Solution satisfying constraint
        """
        try:
            # For now, use simple gradient-based adjustment
            return solution  # Placeholder

        except Exception as e:
            self.logger.warning(f"⚠️ Equality constraint satisfaction failed: {e}")
            return solution

    def _evaluate_population(self, objective_function: Callable):
        """
        Evaluate fitness of all solutions in population.

        Args:
            objective_function: Function to evaluate solutions
        """
        try:
            for solution in self.population:
                if 'fitness' not in solution:
                    try:
                        solution['fitness'] = objective_function(solution)
                    except Exception as e:
                        self.logger.warning(f"⚠️ Solution evaluation failed: {e}")
                        solution['fitness'] = float('-inf')

        except Exception as e:
            self.logger.error(f"❌ Population evaluation failed: {e}")

    def _random_search(self, objective_function: Callable) -> SearchResult:
        """
        Perform random search.

        Args:
            objective_function: Function to evaluate solutions

        Returns:
            SearchResult
        """
        try:
            import time
            start_time = time.time()

            # Simple random search
            for generation in range(self.config.max_generations):
                # Create new random solutions
                new_population = []
                for _ in range(self.config.population_size):
                    solution = self._create_random_solution({})
                    solution['fitness'] = objective_function(solution)
                    new_population.append(solution)

                # Update population
                self.population = new_population

                # Update best solution
                current_best = max(self.population, key=lambda x: x.get('fitness', float('-inf')))
                current_best_score = current_best.get('fitness', float('-inf'))

                if current_best_score > self.best_score:
                    self.best_solution = current_best
                    self.best_score = current_best_score

                # Record history
                self._record_search_history(generation)

                self.generation = generation + 1

            execution_time = time.time() - start_time
            return self._create_search_result(execution_time)

        except Exception as e:
            self.logger.error(f"❌ Random search failed: {e}")
            return self._create_error_result(str(e))

    def _bayesian_search(self, objective_function: Callable) -> SearchResult:
        """
        Perform Bayesian optimization search.

        Args:
            objective_function: Function to evaluate solutions

        Returns:
            SearchResult
        """
        try:
            if 'bayesian' not in self.search_components:
                return self._create_error_result("Bayesian optimizer not available")

            import time
            start_time = time.time()

            bayesian_optimizer = self.search_components['bayesian']

            # Perform Bayesian optimization
            best_params, best_value, history = bayesian_optimizer.optimize(
                objective_function, {}, self.config.max_generations
            )

            self.best_solution = best_params
            self.best_score = best_value
            self.search_history = history

            execution_time = time.time() - start_time
            return self._create_search_result(execution_time)

        except Exception as e:
            self.logger.error(f"❌ Bayesian search failed: {e}")
            return self._create_error_result(str(e))

    def _evolutionary_search(self, objective_function: Callable) -> SearchResult:
        """
        Perform evolutionary algorithm search.

        Args:
            objective_function: Function to evaluate solutions

        Returns:
            SearchResult
        """
        try:
            if 'evolutionary' not in self.search_components:
                return self._create_error_result("Evolutionary optimizer not available")

            import time
            start_time = time.time()

            evolutionary_optimizer = self.search_components['evolutionary']

            # Perform evolutionary search
            result = evolutionary_optimizer.optimize(
                objective_function, {}, self.config.max_generations
            )

            self.best_solution = result.best_solution
            self.best_score = result.best_score
            self.search_history = result.history

            execution_time = time.time() - start_time
            return self._create_search_result(execution_time)

        except Exception as e:
            self.logger.error(f"❌ Evolutionary search failed: {e}")
            return self._create_error_result(str(e))

    def _reinforcement_search(self, objective_function: Callable) -> SearchResult:
        """
        Perform reinforcement learning search.

        Args:
            objective_function: Function to evaluate solutions

        Returns:
            SearchResult
        """
        try:
            if 'reinforcement' not in self.search_components:
                return self._create_error_result("Reinforcement optimizer not available")

            import time
            start_time = time.time()

            # Placeholder for reinforcement learning search
            # This would implement RL-based search strategies

            execution_time = time.time() - start_time
            return self._create_search_result(execution_time)

        except Exception as e:
            self.logger.error(f"❌ Reinforcement search failed: {e}")
            return self._create_error_result(str(e))

    def _meta_learning_search(self, objective_function: Callable) -> SearchResult:
        """
        Perform meta-learning based search.

        Args:
            objective_function: Function to evaluate solutions

        Returns:
            SearchResult
        """
        try:
            if 'meta_learning' not in self.search_components:
                return self._create_error_result("Meta-learning optimizer not available")

            import time
            start_time = time.time()

            # Placeholder for meta-learning search
            # This would use learned search strategies

            execution_time = time.time() - start_time
            return self._create_search_result(execution_time)

        except Exception as e:
            self.logger.error(f"❌ Meta-learning search failed: {e}")
            return self._create_error_result(str(e))

    def _hybrid_search(self, objective_function: Callable) -> SearchResult:
        """
        Perform hybrid search combining multiple strategies.

        Args:
            objective_function: Function to evaluate solutions

        Returns:
            SearchResult
        """
        try:
            import time
            start_time = time.time()

            # Hybrid approach: start with evolutionary, then refine with Bayesian
            if 'evolutionary' in self.search_components:
                evolutionary_result = self._evolutionary_search(objective_function)

                if evolutionary_result.success and 'bayesian' in self.search_components:
                    # Use evolutionary result as starting point for Bayesian optimization
                    bayesian_optimizer = self.search_components['bayesian']
                    # Refine around the best solution found
                    # This is a simplified hybrid approach

            execution_time = time.time() - start_time
            return self._create_search_result(execution_time)

        except Exception as e:
            self.logger.error(f"❌ Hybrid search failed: {e}")
            return self._create_error_result(str(e))

    def _adaptive_search(self, objective_function: Callable) -> SearchResult:
        """
        Perform adaptive search that switches strategies based on progress.

        Args:
            objective_function: Function to evaluate solutions

        Returns:
            SearchResult
        """
        try:
            import time
            start_time = time.time()

            # Adaptive approach: monitor progress and switch strategies
            # Start with random search, then switch to evolutionary if progress slows

            execution_time = time.time() - start_time
            return self._create_search_result(execution_time)

        except Exception as e:
            self.logger.error(f"❌ Adaptive search failed: {e}")
            return self._create_error_result(str(e))

    def _record_search_history(self, generation: int):
        """
        Record search statistics for current generation.

        Args:
            generation: Current generation number
        """
        try:
            if self.population:
                fitness_values = [sol.get('fitness', 0) for sol in self.population]
                avg_fitness = np.mean(fitness_values)
                max_fitness = np.max(fitness_values)
                min_fitness = np.min(fitness_values)
                fitness_std = np.std(fitness_values)

                history_entry = {
                    'generation': generation,
                    'avg_fitness': avg_fitness,
                    'max_fitness': max_fitness,
                    'min_fitness': min_fitness,
                    'fitness_std': fitness_std,
                    'best_score': self.best_score,
                    'population_size': len(self.population)
                }

                self.search_history.append(history_entry)

        except Exception as e:
            self.logger.warning(f"⚠️ Search history recording failed: {e}")

    def _create_search_result(self, execution_time: float) -> SearchResult:
        """
        Create successful search result.

        Args:
            execution_time: Time taken for search

        Returns:
            SearchResult
        """
        try:
            convergence_info = self._analyze_convergence()

            return SearchResult(
                best_solution=self.best_solution,
                best_score=self.best_score,
                search_history=self.search_history,
                convergence_info=convergence_info,
                execution_time=execution_time,
                success=True
            )

        except Exception as e:
            self.logger.warning(f"⚠️ Search result creation failed: {e}")
            return self._create_error_result(str(e))

    def _create_error_result(self, error_message: str) -> SearchResult:
        """
        Create error search result.

        Args:
            error_message: Error description

        Returns:
            SearchResult
        """
        return SearchResult(
            best_solution=None,
            best_score=float('-inf'),
            search_history=[],
            convergence_info={},
            execution_time=0.0,
            success=False,
            error_message=error_message
        )

    def _analyze_convergence(self) -> Dict[str, Any]:
        """
        Analyze search convergence from history.

        Returns:
            Dictionary with convergence information
        """
        try:
            if not self.search_history:
                return {'converged': False, 'convergence_reason': 'No history'}

            recent_history = self.search_history[-5:]  # Last 5 generations

            if len(recent_history) < 2:
                return {'converged': False, 'convergence_reason': 'Insufficient history'}

            # Check for convergence (fitness not improving significantly)
            recent_fitness = [entry['max_fitness'] for entry in recent_history]
            fitness_improvement = np.diff(recent_fitness)
            avg_improvement = np.mean(fitness_improvement)

            converged = abs(avg_improvement) < self.config.convergence_threshold
            stagnation_count = sum(1 for imp in fitness_improvement if abs(imp) < self.config.convergence_threshold)

            return {
                'converged': converged,
                'convergence_reason': 'Fitness not improving' if converged else 'Still improving',
                'stagnation_generations': stagnation_count,
                'avg_recent_improvement': avg_improvement,
                'total_generations': len(self.search_history)
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Convergence analysis failed: {e}")
            return {'converged': False, 'convergence_reason': 'Analysis failed'}