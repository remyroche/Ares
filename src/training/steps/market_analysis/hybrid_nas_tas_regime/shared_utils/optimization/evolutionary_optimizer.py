"""
Evolutionary Optimizer for Regime Detection Systems.

This module provides evolutionary optimization algorithms including NSGA-II and SPEA2
that can be used by both NAS and TAS regime detection systems.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from copy import deepcopy
from src.utils.logger import system_logger
from .bayesian_optimizer import SearchConfig


@dataclass
class EvolutionaryConfig:
    """Configuration for evolutionary optimization."""
    algorithm: str = 'nsga2'  # 'nsga2', 'spea2', 'moead'
    population_size: int = 100
    max_generations: int = 50
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1
    tournament_size: int = 3
    elite_size: int = 10
    mutation_strength: float = 0.1
    enable_parallel: bool = True
    max_workers: int = 4


class Individual:
    """Individual in evolutionary population."""

    def __init__(self, genes: np.ndarray, objectives: List[float] = None):
        """
        Initialize individual.

        Args:
            genes: Gene array
            objectives: Objective function values
        """
        self.genes = genes
        self.objectives = objectives or []
        self.fitness = 0.0
        self.rank = 0
        self.crowding_distance = 0.0
        self.dominated_by = 0
        self.dominates = []

    def dominates(self, other: 'Individual') -> bool:
        """
        Check if this individual dominates another.

        Args:
            other: Other individual to compare with

        Returns:
            True if this individual dominates the other
        """
        if not self.objectives or not other.objectives:
            return False

        # Check Pareto dominance
        at_least_one_better = False
        for i in range(len(self.objectives)):
            if self.objectives[i] > other.objectives[i]:  # Assuming maximization
                at_least_one_better = True
            elif self.objectives[i] < other.objectives[i]:
                return False

        return at_least_one_better

    def __lt__(self, other: 'Individual') -> bool:
        """Less than comparison for sorting."""
        return self.rank < other.rank or (self.rank == other.rank and self.crowding_distance > other.crowding_distance)


class EvolutionaryOptimizer:
    """
    Evolutionary optimizer implementing NSGA-II and SPEA2 algorithms.

    This class provides multi-objective evolutionary optimization that can be
    used by both NAS and TAS systems for finding optimal solutions in complex
    search spaces.
    """

    def __init__(self, config: EvolutionaryConfig):
        """
        Initialize the evolutionary optimizer.

        Args:
            config: Evolutionary optimization configuration
        """
        self.logger = system_logger.getChild('EvolutionaryOptimizer')
        self.config = config

        # Population state
        self.population = []
        self.pareto_front = []
        self.offspring = []

        # Algorithm-specific components
        if config.algorithm == 'nsga2':
            self.selection_method = self._nsga2_selection
            self.replacement_method = self._nsga2_replacement
        elif config.algorithm == 'spea2':
            self.selection_method = self._spea2_selection
            self.replacement_method = self._spea2_replacement
        else:
            self.selection_method = self._nsga2_selection  # Default
            self.replacement_method = self._nsga2_replacement

        self.logger.info(f"✅ Evolutionary Optimizer initialized: {config.algorithm}")
        self.logger.info(f"   Population size: {config.population_size}")
        self.logger.info(f"   Max generations: {config.max_generations}")

    def optimize(self,
                objective_function: Callable,
                search_space: Dict[str, Any],
                max_generations: Optional[int] = None) -> Any:
        """
        Perform evolutionary optimization.

        Args:
            objective_function: Multi-objective function to optimize
            search_space: Definition of search space
            max_generations: Maximum number of generations

        Returns:
            Optimization result with Pareto front
        """
        try:
            self.logger.info("🔍 Starting evolutionary optimization")
            import time
            start_time = time.time()

            max_gens = max_generations or self.config.max_generations

            # Initialize population
            self._initialize_population(search_space)

            # Evolutionary loop
            for generation in range(max_gens):
                self.logger.debug(f"🔄 Generation {generation + 1}/{max_gens}")

                # Evaluate population
                self._evaluate_population(objective_function)

                # Update Pareto front
                self._update_pareto_front()

                # Create offspring
                offspring = self._create_offspring()

                # Evaluate offspring
                self._evaluate_population(objective_function, offspring)

                # Environmental selection
                self.population = self.replacement_method(self.population + offspring)

                # Record generation statistics
                self._record_generation_stats(generation)

            execution_time = time.time() - start_time

            # Final Pareto front
            self._evaluate_population(objective_function)
            self._update_pareto_front()

            result = self._create_optimization_result(execution_time)

            self.logger.info(f"✅ Evolutionary optimization completed in {execution_time:.2f}s")
            self.logger.info(f"🏆 Pareto front size: {len(self.pareto_front)}")

            return result

        except Exception as e:
            self.logger.error(f"❌ Evolutionary optimization failed: {e}")
            return self._create_error_result(str(e))

    def _initialize_population(self, search_space: Dict[str, Any]):
        """
        Initialize population with random individuals.

        Args:
            search_space: Definition of search space
        """
        try:
            self.population = []

            for _ in range(self.config.population_size):
                genes = self._create_random_genes(search_space)
                individual = Individual(genes)
                self.population.append(individual)

            self.logger.info(f"✅ Population initialized with {len(self.population)} individuals")

        except Exception as e:
            self.logger.error(f"❌ Population initialization failed: {e}")

    def _create_random_genes(self, search_space: Dict[str, Any]) -> np.ndarray:
        """
        Create random genes within search space.

        Args:
            search_space: Definition of search space

        Returns:
            Random gene array
        """
        try:
            genes = []

            for param, param_config in search_space.items():
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
                    # Simple range specification
                    if isinstance(param_config, (list, tuple)) and len(param_config) == 2:
                        value = np.random.uniform(param_config[0], param_config[1])
                    else:
                        value = param_config

                genes.append(value)

            return np.array(genes)

        except Exception as e:
            self.logger.warning(f"⚠️ Random genes creation failed: {e}")
            return np.array([0.5] * len(search_space))

    def _evaluate_population(self, objective_function: Callable, population: List[Individual] = None):
        """
        Evaluate objective functions for population.

        Args:
            objective_function: Multi-objective function
            population: Population to evaluate (default: self.population)
        """
        try:
            target_population = population or self.population

            for individual in target_population:
                if individual.objectives is None:
                    try:
                        # Convert genes to parameter dictionary
                        params = self._genes_to_params(individual.genes)

                        # Evaluate objectives
                        individual.objectives = objective_function(params)

                        # Calculate fitness (for single-objective compatibility)
                        if isinstance(individual.objectives, (list, tuple)):
                            individual.fitness = sum(individual.objectives)
                        else:
                            individual.fitness = individual.objectives

                    except Exception as e:
                        self.logger.warning(f"⚠️ Individual evaluation failed: {e}")
                        individual.objectives = [0.0]  # Default objectives
                        individual.fitness = 0.0

        except Exception as e:
            self.logger.error(f"❌ Population evaluation failed: {e}")

    def _genes_to_params(self, genes: np.ndarray) -> Dict[str, Any]:
        """
        Convert genes array back to parameter dictionary.

        Args:
            genes: Gene array

        Returns:
            Parameter dictionary
        """
        try:
            # This is a simplified implementation
            # In practice, you would need to track which gene corresponds to which parameter
            return {'param_' + str(i): genes[i] for i in range(len(genes))}

        except Exception as e:
            self.logger.warning(f"⚠️ Genes to params conversion failed: {e}")
            return {}

    def _update_pareto_front(self):
        """
        Update Pareto front from current population.
        """
        try:
            # Non-dominated sorting
            fronts = self._non_dominated_sorting(self.population)

            # Update Pareto front (first front)
            if fronts:
                self.pareto_front = fronts[0]

                # Assign ranks and crowding distances
                for front in fronts:
                    self._calculate_crowding_distance(front)

        except Exception as e:
            self.logger.warning(f"⚠️ Pareto front update failed: {e}")

    def _non_dominated_sorting(self, population: List[Individual]) -> List[List[Individual]]:
        """
        Perform non-dominated sorting (NSGA-II style).

        Args:
            population: Population to sort

        Returns:
            List of Pareto fronts
        """
        try:
            fronts = []
            remaining_population = population.copy()

            while remaining_population:
                # Find non-dominated individuals
                non_dominated = []

                for i, individual in enumerate(remaining_population):
                    is_dominated = False

                    for j, other in enumerate(remaining_population):
                        if i != j and other.dominates(individual):
                            is_dominated = True
                            break

                    if not is_dominated:
                        non_dominated.append(individual)

                # Remove from remaining population
                for individual in non_dominated:
                    individual.rank = len(fronts)
                    remaining_population.remove(individual)

                fronts.append(non_dominated)

            return fronts

        except Exception as e:
            self.logger.warning(f"⚠️ Non-dominated sorting failed: {e}")
            return [population]  # Return single front as fallback

    def _calculate_crowding_distance(self, front: List[Individual]):
        """
        Calculate crowding distance for individuals in a front.

        Args:
            front: Pareto front to calculate distances for
        """
        try:
            if len(front) <= 2:
                for individual in front:
                    individual.crowding_distance = float('inf')
                return

            n_objectives = len(front[0].objectives) if front[0].objectives else 1

            # Initialize distances
            for individual in front:
                individual.crowding_distance = 0.0

            # Calculate distance for each objective
            for obj_idx in range(n_objectives):
                # Sort by objective value
                front.sort(key=lambda x: x.objectives[obj_idx])

                # Boundary points have infinite distance
                front[0].crowding_distance = float('inf')
                front[-1].crowding_distance = float('inf')

                # Calculate distance for middle points
                obj_range = front[-1].objectives[obj_idx] - front[0].objectives[obj_idx]
                if obj_range == 0:
                    continue

                for i in range(1, len(front) - 1):
                    distance = (front[i + 1].objectives[obj_idx] - front[i - 1].objectives[obj_idx]) / obj_range
                    front[i].crowding_distance += distance

        except Exception as e:
            self.logger.warning(f"⚠️ Crowding distance calculation failed: {e}")

    def _create_offspring(self) -> List[Individual]:
        """
        Create offspring through crossover and mutation.

        Returns:
            List of offspring individuals
        """
        try:
            offspring = []

            # Create offspring until we have enough
            while len(offspring) < self.config.population_size:
                # Select parents
                parent1 = self.selection_method()
                parent2 = self.selection_method()

                # Crossover
                if np.random.random() < self.config.crossover_rate:
                    child1_genes, child2_genes = self._crossover(parent1.genes, parent2.genes)
                else:
                    child1_genes, child2_genes = parent1.genes.copy(), parent2.genes.copy()

                # Mutation
                if np.random.random() < self.config.mutation_rate:
                    child1_genes = self._mutate(child1_genes)
                if np.random.random() < self.config.mutation_rate:
                    child2_genes = self._mutate(child2_genes)

                # Create offspring individuals
                offspring.append(Individual(child1_genes))
                if len(offspring) < self.config.population_size:
                    offspring.append(Individual(child2_genes))

            return offspring[:self.config.population_size]

        except Exception as e:
            self.logger.warning(f"⚠️ Offspring creation failed: {e}")
            return []

    def _crossover(self, parent1_genes: np.ndarray, parent2_genes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform crossover between two parents.

        Args:
            parent1_genes: Genes of first parent
            parent2_genes: Genes of second parent

        Returns:
            Tuple of child gene arrays
        """
        try:
            if len(parent1_genes) != len(parent2_genes):
                return parent1_genes.copy(), parent2_genes.copy()

            # Single-point crossover
            crossover_point = np.random.randint(1, len(parent1_genes))

            child1_genes = np.concatenate([parent1_genes[:crossover_point], parent2_genes[crossover_point:]])
            child2_genes = np.concatenate([parent2_genes[:crossover_point], parent1_genes[crossover_point:]])

            return child1_genes, child2_genes

        except Exception as e:
            self.logger.warning(f"⚠️ Crossover failed: {e}")
            return parent1_genes.copy(), parent2_genes.copy()

    def _mutate(self, genes: np.ndarray) -> np.ndarray:
        """
        Perform mutation on genes.

        Args:
            genes: Gene array to mutate

        Returns:
            Mutated gene array
        """
        try:
            mutated_genes = genes.copy()

            for i in range(len(mutated_genes)):
                if np.random.random() < self.config.mutation_rate:
                    # Gaussian mutation
                    mutation = np.random.normal(0, self.config.mutation_strength)
                    mutated_genes[i] += mutation

                    # Clip to reasonable bounds (this is simplified)
                    mutated_genes[i] = np.clip(mutated_genes[i], 0, 1)

            return mutated_genes

        except Exception as e:
            self.logger.warning(f"⚠️ Mutation failed: {e}")
            return genes

    def _nsga2_selection(self) -> Individual:
        """
        NSGA-II tournament selection.

        Returns:
            Selected individual
        """
        try:
            # Tournament selection
            candidates = np.random.choice(self.population, self.config.tournament_size, replace=False)

            # Select best individual
            best = candidates[0]
            for candidate in candidates[1:]:
                if candidate < best:  # Using the __lt__ method
                    best = candidate

            return best

        except Exception as e:
            self.logger.warning(f"⚠️ NSGA-II selection failed: {e}")
            return self.population[0] if self.population else None

    def _nsga2_replacement(self, combined_population: List[Individual]) -> List[Individual]:
        """
        NSGA-II environmental selection.

        Args:
            combined_population: Combined parent and offspring population

        Returns:
            New population
        """
        try:
            # Non-dominated sorting
            fronts = self._non_dominated_sorting(combined_population)

            # Select individuals from fronts
            new_population = []
            for front in fronts:
                if len(new_population) + len(front) <= self.config.population_size:
                    new_population.extend(front)
                else:
                    # Sort remaining front by crowding distance
                    remaining = sorted(front, reverse=True)  # Higher crowding distance first
                    needed = self.config.population_size - len(new_population)
                    new_population.extend(remaining[:needed])
                    break

            return new_population

        except Exception as e:
            self.logger.warning(f"⚠️ NSGA-II replacement failed: {e}")
            return combined_population[:self.config.population_size]

    def _spea2_selection(self) -> Individual:
        """
        SPEA2 selection method.

        Returns:
            Selected individual
        """
        try:
            # Simplified SPEA2 selection
            return self._nsga2_selection()  # Use tournament selection for now

        except Exception as e:
            self.logger.warning(f"⚠️ SPEA2 selection failed: {e}")
            return self.population[0] if self.population else None

    def _spea2_replacement(self, combined_population: List[Individual]) -> List[Individual]:
        """
        SPEA2 environmental selection.

        Args:
            combined_population: Combined parent and offspring population

        Returns:
            New population
        """
        try:
            # Simplified SPEA2 replacement
            return self._nsga2_replacement(combined_population)

        except Exception as e:
            self.logger.warning(f"⚠️ SPEA2 replacement failed: {e}")
            return combined_population[:self.config.population_size]

    def _record_generation_stats(self, generation: int):
        """
        Record statistics for current generation.

        Args:
            generation: Current generation number
        """
        try:
            if self.population:
                fitness_values = [ind.fitness for ind in self.population if ind.fitness is not None]
                if fitness_values:
                    avg_fitness = np.mean(fitness_values)
                    max_fitness = np.max(fitness_values)
                    min_fitness = np.min(fitness_values)

                    # This would be stored in a history object in practice
                    self.logger.debug(f"   Gen {generation}: avg={avg_fitness".4f"}, max={max_fitness".4f"}, min={min_fitness".4f"}")

        except Exception as e:
            self.logger.warning(f"⚠️ Generation statistics recording failed: {e}")

    def _create_optimization_result(self, execution_time: float) -> Any:
        """
        Create optimization result.

        Args:
            execution_time: Time taken for optimization

        Returns:
            Optimization result object
        """
        try:
            # Convert Pareto front to parameter dictionaries
            pareto_solutions = []
            for individual in self.pareto_front:
                params = self._genes_to_params(individual.genes)
                params['objectives'] = individual.objectives
                params['fitness'] = individual.fitness
                pareto_solutions.append(params)

            result = {
                'best_solution': pareto_solutions[0] if pareto_solutions else None,
                'best_score': max(pareto_solutions, key=lambda x: x.get('fitness', 0)).get('fitness', 0) if pareto_solutions else 0,
                'pareto_front': pareto_solutions,
                'algorithm': self.config.algorithm,
                'execution_time': execution_time,
                'total_generations': self.config.max_generations,
                'population_size': self.config.population_size,
                'success': True
            }

            return result

        except Exception as e:
            self.logger.warning(f"⚠️ Optimization result creation failed: {e}")
            return self._create_error_result(str(e))

    def _create_error_result(self, error_message: str) -> Any:
        """
        Create error result.

        Args:
            error_message: Error description

        Returns:
            Error result object
        """
        return {
            'best_solution': None,
            'best_score': 0.0,
            'pareto_front': [],
            'algorithm': self.config.algorithm,
            'execution_time': 0.0,
            'total_generations': 0,
            'population_size': 0,
            'success': False,
            'error_message': error_message
        }