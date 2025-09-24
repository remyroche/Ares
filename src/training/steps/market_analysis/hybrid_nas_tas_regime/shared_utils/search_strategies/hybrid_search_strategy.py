"""
Hybrid Search Strategy for Regime Detection Systems.

This module provides hybrid search strategies that combine multiple optimization
approaches for both NAS and TAS regime detection systems.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from src.utils.logger import system_logger
from .advanced_search_strategy import SearchConfig, SearchResult, SearchStrategyType


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search strategies."""
    primary_strategy: SearchStrategyType = SearchStrategyType.EVOLUTIONARY
    secondary_strategies: List[SearchStrategyType] = field(default_factory=lambda: [SearchStrategyType.BAYESIAN])
    switch_threshold: float = 0.05  # Improvement threshold to switch strategies
    switch_interval: int = 10  # Generations between strategy switches
    ensemble_size: int = 5  # Number of solutions to ensemble
    diversity_weight: float = 0.3  # Weight for solution diversity
    exploration_weight: float = 0.7  # Weight for exploration vs exploitation


class HybridSearchStrategy:
    """
    Hybrid search strategy that combines multiple optimization approaches.

    This class implements sophisticated hybrid search strategies that can switch
    between different optimization methods and ensemble solutions for better
    performance in both NAS and TAS systems.
    """

    def __init__(self, config: HybridSearchConfig):
        """
        Initialize the hybrid search strategy.

        Args:
            config: Hybrid search configuration
        """
        self.logger = system_logger.getChild('HybridSearchStrategy')
        self.config = config

        # Initialize component search strategies
        self._initialize_component_strategies()

        # Hybrid search state
        self.current_strategy = config.primary_strategy
        self.strategy_history = []
        self.solution_ensemble = []
        self.diversity_metrics = []

        self.logger.info(f"✅ Hybrid Search Strategy initialized")
        self.logger.info(f"   Primary: {config.primary_strategy.value}")
        self.logger.info(f"   Secondary: {[s.value for s in config.secondary_strategies]}")

    def _initialize_component_strategies(self):
        """Initialize component search strategies."""
        try:
            self.strategies = {}

            # Initialize primary strategy
            base_config = SearchConfig(strategy_type=self.config.primary_strategy)
            from .advanced_search_strategy import AdvancedSearchStrategy
            self.strategies[self.config.primary_strategy] = AdvancedSearchStrategy(base_config)

            # Initialize secondary strategies
            for strategy_type in self.config.secondary_strategies:
                if strategy_type != self.config.primary_strategy:
                    base_config = SearchConfig(strategy_type=strategy_type)
                    self.strategies[strategy_type] = AdvancedSearchStrategy(base_config)

            self.logger.info(f"✅ Component strategies initialized: {list(self.strategies.keys())}")

        except Exception as e:
            self.logger.error(f"❌ Component strategies initialization failed: {e}")

    def search(self,
               objective_function: Callable,
               search_space: Dict[str, Any],
               constraints: Optional[Dict[str, Any]] = None) -> SearchResult:
        """
        Perform hybrid search combining multiple strategies.

        Args:
            objective_function: Function to evaluate solutions
            search_space: Definition of search space
            constraints: Optional constraints on solutions

        Returns:
            SearchResult with best solution found
        """
        try:
            self.logger.info("🔍 Starting hybrid search")
            self.logger.info(f"   Ensemble size: {self.config.ensemble_size}")
            self.logger.info(f"   Switch threshold: {self.config.switch_threshold}")

            import time
            start_time = time.time()

            # Initialize search state
            self._initialize_hybrid_search(objective_function, search_space, constraints)

            # Perform hybrid search
            for generation in range(100):  # Fixed generations for hybrid search
                self.logger.debug(f"🔄 Hybrid generation {generation + 1}")

                # Execute current strategy
                strategy_result = self._execute_current_strategy(objective_function, generation)

                # Update ensemble
                self._update_solution_ensemble(strategy_result)

                # Check for strategy switching
                should_switch = self._should_switch_strategy(generation)
                if should_switch:
                    self._switch_strategy(generation)

                # Record hybrid search history
                self._record_hybrid_history(generation, strategy_result)

            # Final ensemble selection
            best_solution = self._select_best_from_ensemble()

            execution_time = time.time() - start_time

            self.logger.info(f"✅ Hybrid search completed: best_score={best_solution.get('fitness', float('-inf'))".4f"}")
            return SearchResult(
                best_solution=best_solution,
                best_score=best_solution.get('fitness', float('-inf')),
                search_history=self.strategy_history,
                convergence_info=self._analyze_hybrid_convergence(),
                execution_time=execution_time,
                success=True
            )

        except Exception as e:
            self.logger.error(f"❌ Hybrid search failed: {e}")
            return SearchResult(
                best_solution=None,
                best_score=float('-inf'),
                search_history=[],
                convergence_info={},
                execution_time=0.0,
                success=False,
                error_message=str(e)
            )

    def _initialize_hybrid_search(self,
                                objective_function: Callable,
                                search_space: Dict[str, Any],
                                constraints: Optional[Dict[str, Any]]):
        """
        Initialize hybrid search state.

        Args:
            objective_function: Function to evaluate solutions
            search_space: Definition of search space
            constraints: Optional constraints
        """
        try:
            # Initialize ensemble
            self.solution_ensemble = []

            # Create initial diverse population
            initial_population = []
            for i in range(self.config.ensemble_size * 2):
                solution = self._create_diverse_solution(search_space, i)
                if constraints:
                    solution = self._apply_hybrid_constraints(solution, constraints)
                solution['fitness'] = objective_function(solution)
                initial_population.append(solution)

            # Initialize ensemble with best solutions
            initial_population.sort(key=lambda x: x.get('fitness', float('-inf')), reverse=True)
            self.solution_ensemble = initial_population[:self.config.ensemble_size]

            # Initialize strategy history
            self.strategy_history = []

            self.logger.info(f"✅ Hybrid search initialized with {len(self.solution_ensemble)} ensemble solutions")

        except Exception as e:
            self.logger.error(f"❌ Hybrid search initialization failed: {e}")

    def _create_diverse_solution(self, search_space: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Create a diverse solution for ensemble initialization.

        Args:
            search_space: Definition of search space
            index: Solution index for diversity

        Returns:
            Diverse solution dictionary
        """
        try:
            solution = {}

            for param_name, param_config in search_space.items():
                if isinstance(param_config, dict):
                    param_type = param_config.get('type', 'continuous')
                    param_range = param_config.get('range', [0, 1])

                    # Use index to create diversity
                    if param_type == 'continuous':
                        # Spread solutions across parameter space
                        base_value = np.random.uniform(param_range[0], param_range[1])
                        diversity_offset = (index * 0.1) % 1.0  # Small systematic variation
                        value = np.clip(base_value + diversity_offset - 0.5, param_range[0], param_range[1])
                    elif param_type == 'integer':
                        value = np.random.randint(param_range[0], param_range[1] + 1)
                    elif param_type == 'categorical':
                        choices = param_config.get('choices', param_range)
                        value = choices[index % len(choices)]
                    else:
                        value = np.random.uniform(param_range[0], param_range[1])
                else:
                    # Simple range specification
                    if isinstance(param_config, (list, tuple)) and len(param_config) == 2:
                        value = np.random.uniform(param_config[0], param_config[1])
                    else:
                        value = param_config

                solution[param_name] = value

            return solution

        except Exception as e:
            self.logger.warning(f"⚠️ Diverse solution creation failed: {e}")
            return self._create_random_solution(search_space)

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
                    if isinstance(param_config, (list, tuple)) and len(param_config) == 2:
                        value = np.random.uniform(param_config[0], param_config[1])
                    else:
                        value = param_config

                solution[param_name] = value

            return solution

        except Exception as e:
            self.logger.warning(f"⚠️ Random solution creation failed: {e}")
            return {}

    def _apply_hybrid_constraints(self, solution: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply constraints to a solution in hybrid context.

        Args:
            solution: Solution to constrain
            constraints: Constraint definitions

        Returns:
            Constrained solution
        """
        try:
            # Apply constraints while maintaining diversity
            constrained_solution = solution.copy()

            for constraint_name, constraint_config in constraints.items():
                constraint_type = constraint_config.get('type', 'inequality')

                if constraint_type == 'inequality':
                    constraint_func = constraint_config.get('function')
                    if constraint_func and not constraint_func(constrained_solution):
                        # Repair while preserving diversity
                        constrained_solution = self._hybrid_repair_solution(
                            constrained_solution, constraint_config
                        )

            return constrained_solution

        except Exception as e:
            self.logger.warning(f"⚠️ Hybrid constraint application failed: {e}")
            return solution

    def _hybrid_repair_solution(self, solution: Dict[str, Any], constraint_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Repair a solution that violates constraints while maintaining diversity.

        Args:
            solution: Solution to repair
            constraint_config: Constraint configuration

        Returns:
            Repaired solution
        """
        try:
            # Hybrid repair: try to fix constraint violation while maintaining solution structure
            repaired_solution = solution.copy()

            # Try multiple repair attempts
            max_attempts = 5
            for attempt in range(max_attempts):
                # Apply small random changes to parameters
                for param_name in repaired_solution:
                    if isinstance(repaired_solution[param_name], (int, float)):
                        # Small perturbation
                        perturbation = np.random.normal(0, abs(repaired_solution[param_name]) * 0.05)
                        repaired_solution[param_name] = max(0, repaired_solution[param_name] + perturbation)

                # Check if repair successful
                constraint_func = constraint_config.get('function')
                if constraint_func and constraint_func(repaired_solution):
                    return repaired_solution

            return solution  # Return original if repair fails

        except Exception as e:
            self.logger.warning(f"⚠️ Hybrid solution repair failed: {e}")
            return solution

    def _execute_current_strategy(self, objective_function: Callable, generation: int) -> Dict[str, Any]:
        """
        Execute the current search strategy.

        Args:
            objective_function: Function to evaluate solutions
            generation: Current generation number

        Returns:
            Strategy execution result
        """
        try:
            current_strategy = self.strategies[self.current_strategy]

            # Create search configuration for current strategy
            search_config = SearchConfig(
                strategy_type=self.current_strategy,
                population_size=self.config.ensemble_size,
                max_generations=1  # Single generation per hybrid step
            )

            # Create search space for current strategy
            search_space = self._create_strategy_search_space(generation)

            # Execute strategy
            result = current_strategy.search(objective_function, search_space)

            return {
                'strategy': self.current_strategy,
                'generation': generation,
                'best_solution': result.best_solution,
                'best_score': result.best_score,
                'search_time': result.execution_time
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Current strategy execution failed: {e}")
            return {
                'strategy': self.current_strategy,
                'generation': generation,
                'best_solution': None,
                'best_score': float('-inf'),
                'search_time': 0.0
            }

    def _create_strategy_search_space(self, generation: int) -> Dict[str, Any]:
        """
        Create search space for current strategy based on ensemble diversity.

        Args:
            generation: Current generation number

        Returns:
            Search space dictionary
        """
        try:
            # Base search space from ensemble
            if self.solution_ensemble:
                # Create search space centered around ensemble solutions
                search_space = {}

                # Extract parameter ranges from ensemble
                for param_name in self.solution_ensemble[0].keys():
                    if param_name != 'fitness':  # Skip fitness score
                        param_values = [sol[param_name] for sol in self.solution_ensemble if param_name in sol]

                        if param_values:
                            min_val = min(param_values)
                            max_val = max(param_values)

                            # Add some margin for exploration
                            margin = (max_val - min_val) * 0.2
                            search_space[param_name] = {
                                'type': 'continuous',
                                'range': [max(0, min_val - margin), max_val + margin]
                            }

                return search_space
            else:
                # Fallback to default search space
                return {'param1': {'type': 'continuous', 'range': [0, 1]}}

        except Exception as e:
            self.logger.warning(f"⚠️ Strategy search space creation failed: {e}")
            return {'param1': {'type': 'continuous', 'range': [0, 1]}}

    def _update_solution_ensemble(self, strategy_result: Dict[str, Any]):
        """
        Update the solution ensemble with new results.

        Args:
            strategy_result: Result from strategy execution
        """
        try:
            if strategy_result['best_solution'] and strategy_result['best_score'] > float('-inf'):
                # Add new solution to ensemble
                new_solution = strategy_result['best_solution'].copy()
                new_solution['fitness'] = strategy_result['best_score']

                self.solution_ensemble.append(new_solution)

                # Maintain ensemble size
                if len(self.solution_ensemble) > self.config.ensemble_size:
                    # Remove worst solution
                    self.solution_ensemble.sort(key=lambda x: x.get('fitness', float('-inf')), reverse=True)
                    self.solution_ensemble = self.solution_ensemble[:self.config.ensemble_size]

                # Calculate diversity metrics
                self._update_diversity_metrics()

        except Exception as e:
            self.logger.warning(f"⚠️ Solution ensemble update failed: {e}")

    def _update_diversity_metrics(self):
        """
        Update diversity metrics for the current ensemble.
        """
        try:
            if len(self.solution_ensemble) < 2:
                self.diversity_metrics.append(0.0)
                return

            # Calculate average pairwise distance
            diversity_scores = []
            for i, sol1 in enumerate(self.solution_ensemble):
                for j, sol2 in enumerate(self.solution_ensemble[i+1:], i+1):
                    distance = self._calculate_solution_distance(sol1, sol2)
                    diversity_scores.append(distance)

            avg_diversity = np.mean(diversity_scores) if diversity_scores else 0.0
            self.diversity_metrics.append(avg_diversity)

        except Exception as e:
            self.logger.warning(f"⚠️ Diversity metrics update failed: {e}")
            self.diversity_metrics.append(0.0)

    def _calculate_solution_distance(self, solution1: Dict[str, Any], solution2: Dict[str, Any]) -> float:
        """
        Calculate distance between two solutions.

        Args:
            solution1: First solution
            solution2: Second solution

        Returns:
            Distance between solutions
        """
        try:
            distance = 0.0
            common_params = set(solution1.keys()) & set(solution2.keys())

            for param in common_params:
                if param != 'fitness':
                    val1 = solution1[param]
                    val2 = solution2[param]

                    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                        # Normalize distance by parameter range
                        param_range = max(abs(val1), abs(val2), 1.0)
                        distance += abs(val1 - val2) / param_range

            return distance / max(len(common_params), 1)

        except Exception as e:
            self.logger.warning(f"⚠️ Solution distance calculation failed: {e}")
            return 0.0

    def _should_switch_strategy(self, generation: int) -> bool:
        """
        Determine if strategy should be switched.

        Args:
            generation: Current generation number

        Returns:
            True if strategy should be switched
        """
        try:
            # Switch strategy based on progress and diversity
            should_switch = False

            # Switch if no improvement in recent generations
            if len(self.strategy_history) >= 5:
                recent_scores = [entry.get('best_score', float('-inf')) for entry in self.strategy_history[-5:]]
                if max(recent_scores) - min(recent_scores) < self.config.switch_threshold:
                    should_switch = True

            # Switch based on generation interval
            if generation > 0 and generation % self.config.switch_interval == 0:
                should_switch = True

            # Switch if diversity is too low
            if self.diversity_metrics and len(self.diversity_metrics) >= 3:
                recent_diversity = np.mean(self.diversity_metrics[-3:])
                if recent_diversity < 0.1:  # Low diversity threshold
                    should_switch = True

            return should_switch

        except Exception as e:
            self.logger.warning(f"⚠️ Strategy switch decision failed: {e}")
            return False

    def _switch_strategy(self, generation: int):
        """
        Switch to a different search strategy.

        Args:
            generation: Current generation number
        """
        try:
            # Select next strategy
            available_strategies = list(self.strategies.keys())
            current_index = available_strategies.index(self.current_strategy)
            next_index = (current_index + 1) % len(available_strategies)
            new_strategy = available_strategies[next_index]

            old_strategy = self.current_strategy
            self.current_strategy = new_strategy

            self.logger.info(f"🔄 Switching strategy: {old_strategy.value} → {new_strategy.value} (gen {generation})")

        except Exception as e:
            self.logger.warning(f"⚠️ Strategy switch failed: {e}")

    def _record_hybrid_history(self, generation: int, strategy_result: Dict[str, Any]):
        """
        Record hybrid search history.

        Args:
            generation: Current generation number
            strategy_result: Result from strategy execution
        """
        try:
            history_entry = {
                'generation': generation,
                'strategy': strategy_result['strategy'].value,
                'best_score': strategy_result['best_score'],
                'ensemble_size': len(self.solution_ensemble),
                'avg_diversity': self.diversity_metrics[-1] if self.diversity_metrics else 0.0,
                'current_best_fitness': self.solution_ensemble[0].get('fitness', float('-inf')) if self.solution_ensemble else float('-inf')
            }

            self.strategy_history.append(history_entry)

        except Exception as e:
            self.logger.warning(f"⚠️ Hybrid history recording failed: {e}")

    def _select_best_from_ensemble(self) -> Dict[str, Any]:
        """
        Select the best solution from the ensemble.

        Returns:
            Best solution from ensemble
        """
        try:
            if not self.solution_ensemble:
                return {}

            # Return the solution with highest fitness
            best_solution = max(self.solution_ensemble, key=lambda x: x.get('fitness', float('-inf')))
            return best_solution

        except Exception as e:
            self.logger.warning(f"⚠️ Ensemble selection failed: {e}")
            return {}

    def _analyze_hybrid_convergence(self) -> Dict[str, Any]:
        """
        Analyze convergence of hybrid search.

        Returns:
            Dictionary with convergence information
        """
        try:
            if not self.strategy_history:
                return {'converged': False, 'convergence_reason': 'No history'}

            # Analyze strategy switches and progress
            strategy_switches = len(set(entry['strategy'] for entry in self.strategy_history))

            recent_history = self.strategy_history[-5:]
            recent_scores = [entry['best_score'] for entry in recent_history]

            if len(recent_scores) < 2:
                return {'converged': False, 'convergence_reason': 'Insufficient data'}

            # Check for convergence
            score_improvement = np.diff(recent_scores)
            avg_improvement = np.mean(score_improvement)
            converged = abs(avg_improvement) < self.config.switch_threshold

            return {
                'converged': converged,
                'convergence_reason': 'Progress stalled' if converged else 'Still improving',
                'strategy_switches': strategy_switches,
                'final_ensemble_size': len(self.solution_ensemble),
                'avg_recent_improvement': avg_improvement,
                'total_generations': len(self.strategy_history)
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Hybrid convergence analysis failed: {e}")
            return {'converged': False, 'convergence_reason': 'Analysis failed'}