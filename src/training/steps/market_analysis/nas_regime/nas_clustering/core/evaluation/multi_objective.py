"""
Multi-Objective Optimization for NAS Clustering

Provides NSGA-II based multi-objective optimization for neural architecture search.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
import logging
from dataclasses import dataclass
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Result of multi-objective optimization."""
    pareto_frontier: List[Any]
    best_solutions: List[Any]
    metrics: Dict[str, float]
    execution_time: float

class Objective(ABC):
    """Abstract base class for optimization objectives."""
    
    @abstractmethod
    def evaluate(self, individual: Any) -> float:
        """Evaluate an individual and return objective value."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this objective."""
        pass

class FitnessObjective(Objective):
    """Objective for maximizing fitness score."""
    
    def evaluate(self, individual: Any) -> float:
        """Evaluate fitness score."""
        if hasattr(individual, 'fitness_score'):
            return individual.fitness_score
        return 0.0
    
    def get_name(self) -> str:
        return "fitness"

class ComplexityObjective(Objective):
    """Objective for minimizing complexity."""
    
    def evaluate(self, individual: Any) -> float:
        """Evaluate complexity (inverted for minimization)."""
        if hasattr(individual, 'complexity_score'):
            return 1.0 - individual.complexity_score  # Invert for minimization
        return 1.0
    
    def get_name(self) -> str:
        return "complexity"

class EfficiencyObjective(Objective):
    """Objective for maximizing efficiency."""
    
    def evaluate(self, individual: Any) -> float:
        """Evaluate efficiency score."""
        if hasattr(individual, 'efficiency_score'):
            return individual.efficiency_score
        return 0.0
    
    def get_name(self) -> str:
        return "efficiency"

def create_nas_objectives() -> List[Objective]:
    """Create default NAS objectives."""
    return [
        FitnessObjective(),
        ComplexityObjective(),
        EfficiencyObjective()
    ]

class NSGAIIOptimizer:
    """
    NSGA-II Multi-Objective Optimizer for NAS clustering.
    
    Implements the Non-dominated Sorting Genetic Algorithm II for
    multi-objective optimization of neural architectures.
    """
    
    def __init__(self, objectives: List[Objective], population_size: int = 50):
        """
        Initialize NSGA-II Optimizer.
        
        Args:
            objectives: List of objective functions to optimize
            population_size: Size of the population
        """
        self.objectives = objectives
        self.population_size = population_size
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # NSGA-II parameters
        self.crossover_rate = 0.9
        self.mutation_rate = 0.1
        self.tournament_size = 3
        
        self.logger.info(f"NSGA-II Optimizer initialized with {len(objectives)} objectives")
    
    def optimize(self, objectives: List[str], population: List[Any]) -> OptimizationResult:
        """
        Perform multi-objective optimization.
        
        Args:
            objectives: List of objective names to optimize
            population: Initial population of individuals
            
        Returns:
            OptimizationResult with optimization results
        """
        start_time = time.time()
        self.logger.info(f"Starting NSGA-II optimization with population size {len(population)}")
        
        try:
            # Filter objectives based on requested objectives
            filtered_objectives = [obj for obj in self.objectives if obj.get_name() in objectives]
            
            if not filtered_objectives:
                self.logger.warning("No valid objectives found, using default objectives")
                filtered_objectives = self.objectives
            
            # Initialize population if needed
            if not population:
                population = self._generate_initial_population()
            
            # Perform NSGA-II optimization
            pareto_frontier = self._nsga_ii_algorithm(population, filtered_objectives)
            
            # Select best solutions
            best_solutions = self._select_best_solutions(pareto_frontier, filtered_objectives)
            
            # Calculate optimization metrics
            metrics = self._calculate_optimization_metrics(pareto_frontier, filtered_objectives)
            
            execution_time = time.time() - start_time
            
            result = OptimizationResult(
                pareto_frontier=pareto_frontier,
                best_solutions=best_solutions,
                metrics=metrics,
                execution_time=execution_time
            )
            
            self.logger.info(f"NSGA-II optimization completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"NSGA-II optimization failed: {e}")
            execution_time = time.time() - start_time
            return OptimizationResult(
                pareto_frontier=[],
                best_solutions=[],
                metrics={'error': str(e)},
                execution_time=execution_time
            )
    
    def _nsga_ii_algorithm(self, population: List[Any], objectives: List[Objective]) -> List[Any]:
        """Implement NSGA-II algorithm."""
        try:
            current_population = population.copy()
            
            # Evaluate initial population
            self._evaluate_population(current_population, objectives)
            
            # Main NSGA-II loop
            for generation in range(50):  # Fixed number of generations
                # Create offspring through crossover and mutation
                offspring = self._create_offspring(current_population)
                
                # Evaluate offspring
                self._evaluate_population(offspring, objectives)
                
                # Combine parent and offspring populations
                combined_population = current_population + offspring
                
                # Non-dominated sorting
                fronts = self._non_dominated_sorting(combined_population, objectives)
                
                # Select next generation
                current_population = self._select_next_generation(fronts, objectives)
                
                self.logger.debug(f"Generation {generation + 1}: Population size {len(current_population)}")
            
            # Return final Pareto frontier
            return self._non_dominated_sorting(current_population, objectives)[0]
            
        except Exception as e:
            self.logger.warning(f"NSGA-II algorithm failed: {e}")
            return population[:min(10, len(population))]
    
    def _evaluate_population(self, population: List[Any], objectives: List[Objective]):
        """Evaluate population against objectives."""
        for individual in population:
            if not hasattr(individual, 'objective_values'):
                individual.objective_values = []
            
            individual.objective_values.clear()
            for objective in objectives:
                value = objective.evaluate(individual)
                individual.objective_values.append(value)
    
    def _non_dominated_sorting(self, population: List[Any], objectives: List[Objective]) -> List[List[Any]]:
        """Perform non-dominated sorting."""
        fronts = []
        remaining_population = population.copy()
        
        while remaining_population:
            current_front = []
            
            for individual in remaining_population:
                is_dominated = False
                
                for other in remaining_population:
                    if individual != other and self._dominates(other, individual, objectives):
                        is_dominated = True
                        break
                
                if not is_dominated:
                    current_front.append(individual)
            
            if current_front:
                fronts.append(current_front)
                for individual in current_front:
                    remaining_population.remove(individual)
            else:
                break
        
        return fronts
    
    def _dominates(self, individual1: Any, individual2: Any, objectives: List[Objective]) -> bool:
        """Check if individual1 dominates individual2."""
        try:
            values1 = individual1.objective_values if hasattr(individual1, 'objective_values') else []
            values2 = individual2.objective_values if hasattr(individual2, 'objective_values') else []
            
            if len(values1) != len(values2) or len(values1) != len(objectives):
                return False
            
            at_least_one_better = False
            
            for i, objective in enumerate(objectives):
                if objective.get_name() == "complexity":
                    # Complexity should be minimized
                    if values1[i] > values2[i]:
                        return False
                    elif values1[i] < values2[i]:
                        at_least_one_better = True
                else:
                    # Other objectives should be maximized
                    if values1[i] < values2[i]:
                        return False
                    elif values1[i] > values2[i]:
                        at_least_one_better = True
            
            return at_least_one_better
            
        except Exception as e:
            self.logger.warning(f"Dominance check failed: {e}")
            return False
    
    def _create_offspring(self, population: List[Any]) -> List[Any]:
        """Create offspring through crossover and mutation."""
        try:
            offspring = []
            
            while len(offspring) < len(population):
                # Tournament selection for parents
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                
                # Crossover
                if np.random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2
                
                # Mutation
                if np.random.random() < self.mutation_rate:
                    child1 = self._mutate(child1)
                if np.random.random() < self.mutation_rate:
                    child2 = self._mutate(child2)
                
                offspring.extend([child1, child2])
            
            return offspring[:len(population)]
            
        except Exception as e:
            self.logger.warning(f"Offspring creation failed: {e}")
            return population[:len(population)//2]
    
    def _tournament_selection(self, population: List[Any]) -> Any:
        """Tournament selection."""
        try:
            tournament = np.random.choice(population, size=self.tournament_size, replace=False)
            
            # Select best individual from tournament
            best_individual = tournament[0]
            best_score = self._calculate_fitness_sum(best_individual)
            
            for individual in tournament[1:]:
                score = self._calculate_fitness_sum(individual)
                if score > best_score:
                    best_individual = individual
                    best_score = score
            
            return best_individual
            
        except Exception as e:
            self.logger.warning(f"Tournament selection failed: {e}")
            return population[0] if population else None
    
    def _crossover(self, parent1: Any, parent2: Any) -> Tuple[Any, Any]:
        """Perform crossover between two parents."""
        try:
            # Simple crossover for architectures
            child1 = self._copy_individual(parent1)
            child2 = self._copy_individual(parent2)
            
            # Exchange some parameters
            if hasattr(child1, 'parameters_count') and hasattr(child2, 'parameters_count'):
                # Swap parameter counts
                temp = child1.parameters_count
                child1.parameters_count = child2.parameters_count
                child2.parameters_count = temp
            
            return child1, child2
            
        except Exception as e:
            self.logger.warning(f"Crossover failed: {e}")
            return parent1, parent2
    
    def _mutate(self, individual: Any) -> Any:
        """Perform mutation on an individual."""
        try:
            mutated = self._copy_individual(individual)
            
            # Mutate parameters
            if hasattr(mutated, 'parameters_count'):
                # Add random variation to parameter count
                variation = np.random.normal(0, mutated.parameters_count * 0.1)
                mutated.parameters_count = max(100, int(mutated.parameters_count + variation))
            
            if hasattr(mutated, 'fitness_score'):
                # Add small random variation to fitness
                variation = np.random.normal(0, 0.05)
                mutated.fitness_score = max(0.0, min(1.0, mutated.fitness_score + variation))
            
            return mutated
            
        except Exception as e:
            self.logger.warning(f"Mutation failed: {e}")
            return individual
    
    def _select_next_generation(self, fronts: List[List[Any]], objectives: List[Objective]) -> List[Any]:
        """Select next generation using crowding distance."""
        try:
            next_generation = []
            
            for front in fronts:
                if len(next_generation) + len(front) <= self.population_size:
                    next_generation.extend(front)
                else:
                    # Use crowding distance for the last front
                    remaining_slots = self.population_size - len(next_generation)
                    if remaining_slots > 0:
                        sorted_front = self._crowding_distance_sort(front, objectives)
                        next_generation.extend(sorted_front[:remaining_slots])
                    break
            
            return next_generation
            
        except Exception as e:
            self.logger.warning(f"Next generation selection failed: {e}")
            return fronts[0][:self.population_size] if fronts else []
    
    def _crowding_distance_sort(self, front: List[Any], objectives: List[Objective]) -> List[Any]:
        """Sort front by crowding distance."""
        try:
            # Calculate crowding distance for each individual
            for individual in front:
                individual.crowding_distance = 0.0
            
            for obj_idx in range(len(objectives)):
                # Sort by objective value
                front.sort(key=lambda x: x.objective_values[obj_idx])
                
                # Set boundary points
                front[0].crowding_distance = float('inf')
                front[-1].crowding_distance = float('inf')
                
                # Calculate crowding distance
                obj_range = front[-1].objective_values[obj_idx] - front[0].objective_values[obj_idx]
                if obj_range > 0:
                    for i in range(1, len(front) - 1):
                        distance = (front[i+1].objective_values[obj_idx] - front[i-1].objective_values[obj_idx]) / obj_range
                        front[i].crowding_distance += distance
            
            # Sort by crowding distance (descending)
            front.sort(key=lambda x: x.crowding_distance, reverse=True)
            
            return front
            
        except Exception as e:
            self.logger.warning(f"Crowding distance sort failed: {e}")
            return front
    
    def _select_best_solutions(self, pareto_frontier: List[Any], objectives: List[Objective]) -> List[Any]:
        """Select best solutions from Pareto frontier."""
        try:
            if not pareto_frontier:
                return []
            
            # Sort by combined objective score
            for individual in pareto_frontier:
                individual.combined_score = self._calculate_fitness_sum(individual)
            
            pareto_frontier.sort(key=lambda x: x.combined_score, reverse=True)
            
            # Return top 10 solutions
            return pareto_frontier[:min(10, len(pareto_frontier))]
            
        except Exception as e:
            self.logger.warning(f"Best solution selection failed: {e}")
            return pareto_frontier[:5]
    
    def _calculate_fitness_sum(self, individual: Any) -> float:
        """Calculate sum of fitness scores."""
        try:
            if hasattr(individual, 'objective_values'):
                return sum(individual.objective_values)
            elif hasattr(individual, 'fitness_score'):
                return individual.fitness_score
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def _calculate_optimization_metrics(self, pareto_frontier: List[Any], objectives: List[Objective]) -> Dict[str, float]:
        """Calculate optimization metrics."""
        try:
            metrics = {
                'pareto_frontier_size': len(pareto_frontier),
                'diversity_score': 0.0,
                'convergence_score': 0.0
            }
            
            if len(pareto_frontier) > 1:
                # Calculate diversity score
                objective_ranges = []
                for obj_idx in range(len(objectives)):
                    values = [ind.objective_values[obj_idx] for ind in pareto_frontier]
                    obj_range = max(values) - min(values)
                    objective_ranges.append(obj_range)
                
                metrics['diversity_score'] = np.mean(objective_ranges)
                
                # Calculate convergence score (average fitness)
                fitness_scores = [self._calculate_fitness_sum(ind) for ind in pareto_frontier]
                metrics['convergence_score'] = np.mean(fitness_scores)
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Metrics calculation failed: {e}")
            return {'error': str(e)}
    
    def _generate_initial_population(self) -> List[Any]:
        """Generate initial population if none provided."""
        try:
            population = []
            
            for i in range(self.population_size):
                individual = type('Individual', (), {
                    'fitness_score': np.random.random(),
                    'complexity_score': np.random.random(),
                    'efficiency_score': np.random.random(),
                    'parameters_count': np.random.randint(100, 10000)
                })()
                population.append(individual)
            
            return population
            
        except Exception as e:
            self.logger.warning(f"Initial population generation failed: {e}")
            return []
    
    def _copy_individual(self, individual: Any) -> Any:
        """Create a copy of an individual."""
        try:
            copied = type('Individual', (), {})()
            
            # Copy attributes
            for attr in dir(individual):
                if not attr.startswith('_'):
                    setattr(copied, attr, getattr(individual, attr))
            
            return copied
            
        except Exception as e:
            self.logger.warning(f"Individual copying failed: {e}")
            return individual
