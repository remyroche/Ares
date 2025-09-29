"""
Advanced Evolutionary Algorithms for Hybrid NAS-TAS Regime Detection.

Provides common evolutionary algorithm utilities including NSGA-II and SPEA2.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
from dataclasses import dataclass
import time
from datetime import datetime
from abc import ABC, abstractmethod
import random
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

logger = logging.getLogger(__name__)


@dataclass
class EvolutionaryAlgorithmConfig:
    """Configuration for evolutionary algorithm operations."""
    population_size: int = 100
    max_generations: int = 50
    crossover_probability: float = 0.8
    mutation_probability: float = 0.1
    tournament_size: int = 3
    elitism_size: int = 10
    convergence_threshold: float = 1e-6
    random_state: int = 42
    use_nsga2: bool = True
    use_spea2: bool = True


@dataclass
class Individual:
    """Individual in the evolutionary algorithm population."""
    parameters: Dict[str, Any]
    objectives: List[float]
    rank: int = 0
    crowding_distance: float = 0.0
    fitness: float = 0.0
    dominated_count: int = 0
    dominated_solutions: List[int] = None
    
    def __post_init__(self):
        if self.dominated_solutions is None:
            self.dominated_solutions = []


@dataclass
class EvolutionaryResult:
    """Result from evolutionary algorithm operations."""
    best_individuals: List[Individual]
    pareto_front: List[Individual]
    optimization_history: List[Dict[str, Any]]
    convergence_info: Dict[str, Any]
    execution_time: float
    success: bool
    error_message: Optional[str] = None


class EvolutionaryAlgorithm(ABC):
    """Abstract base class for evolutionary algorithms."""
    
    def __init__(self, config: EvolutionaryAlgorithmConfig):
        """Initialize the evolutionary algorithm.
        
        Args:
            config: Evolutionary algorithm configuration
        """
        tprint_info("Initializing Evolutionary Algorithm")
        tprint_debug(f"Configuration: {config}")
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        random.seed(config.random_state)
        np.random.seed(config.random_state)
        tprint_success("Evolutionary Algorithm initialized successfully")
    
    @abstractmethod
    def optimize(self, objective_functions: List[Callable], parameter_space: Dict[str, Any]) -> EvolutionaryResult:
        """Optimize multiple objective functions.
        
        Args:
            objective_functions: List of objective functions to optimize
            parameter_space: Parameter space definition
            
        Returns:
            EvolutionaryResult with optimization results
        """
        pass
    
    def _initialize_population(self, parameter_space: Dict[str, Any]) -> List[Individual]:
        """Initialize random population."""
        try:
            tprint_info(f"Initializing population of size {self.config.population_size}")
            tprint_debug(f"Parameter space: {parameter_space}")
            population = []
            
            for i in range(self.config.population_size):
                if i % 10 == 0:  # Progress every 10 individuals
                    tprint_progress(i, self.config.population_size, "Initializing population")
                parameters = self._generate_random_parameters(parameter_space)
                individual = Individual(
                    parameters=parameters,
                    objectives=[0.0] * len(parameter_space)  # Placeholder
                )
                population.append(individual)
            
            return population
            
        except Exception as e:
            self.logger.warning(f"⚠️ Population initialization failed: {e}")
            return []
    
    def _generate_random_parameters(self, parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Generate random parameters within the parameter space."""
        try:
            parameters = {}
            
            for param_name, param_config in parameter_space.items():
                if isinstance(param_config, dict):
                    if param_config['type'] == 'continuous':
                        min_val = param_config['min']
                        max_val = param_config['max']
                        parameters[param_name] = random.uniform(min_val, max_val)
                    elif param_config['type'] == 'discrete':
                        choices = param_config['choices']
                        parameters[param_name] = random.choice(choices)
                    elif param_config['type'] == 'integer':
                        min_val = param_config['min']
                        max_val = param_config['max']
                        parameters[param_name] = random.randint(min_val, max_val)
                else:
                    # Simple range
                    if isinstance(param_config, (list, tuple)) and len(param_config) == 2:
                        min_val, max_val = param_config
                        parameters[param_name] = random.uniform(min_val, max_val)
                    else:
                        parameters[param_name] = param_config
            
            return parameters
            
        except Exception as e:
            self.logger.warning(f"⚠️ Random parameter generation failed: {e}")
            return {}
    
    def _evaluate_objectives(self, individual: Individual, objective_functions: List[Callable]) -> List[float]:
        """Evaluate objective functions for an individual."""
        try:
            objectives = []
            
            for obj_func in objective_functions:
                try:
                    score = obj_func(individual.parameters)
                    objectives.append(score)
                except Exception as e:
                    self.logger.warning(f"⚠️ Objective evaluation failed: {e}")
                    objectives.append(0.0)
            
            return objectives
            
        except Exception as e:
            self.logger.warning(f"⚠️ Objective evaluation failed: {e}")
            return [0.0] * len(objective_functions)
    
    def _crossover(self, parent1: Individual, parent2: Individual, parameter_space: Dict[str, Any]) -> Tuple[Individual, Individual]:
        """Perform crossover between two parents."""
        try:
            if random.random() > self.config.crossover_probability:
                return parent1, parent2
            
            child1_params = parent1.parameters.copy()
            child2_params = parent2.parameters.copy()
            
            # Single-point crossover for each parameter
            for param_name in parameter_space.keys():
                if random.random() < 0.5:
                    child1_params[param_name], child2_params[param_name] = child2_params[param_name], child1_params[param_name]
            
            child1 = Individual(parameters=child1_params, objectives=[])
            child2 = Individual(parameters=child2_params, objectives=[])
            
            return child1, child2
            
        except Exception as e:
            self.logger.warning(f"⚠️ Crossover failed: {e}")
            return parent1, parent2
    
    def _mutate(self, individual: Individual, parameter_space: Dict[str, Any]) -> Individual:
        """Perform mutation on an individual."""
        try:
            if random.random() > self.config.mutation_probability:
                return individual
            
            mutated_params = individual.parameters.copy()
            
            # Mutate each parameter with small probability
            for param_name, param_config in parameter_space.items():
                if random.random() < 0.1:  # 10% chance to mutate each parameter
                    if isinstance(param_config, dict):
                        if param_config['type'] == 'continuous':
                            min_val = param_config['min']
                            max_val = param_config['max']
                            # Gaussian mutation
                            sigma = (max_val - min_val) * 0.1
                            mutated_params[param_name] = max(min_val, min(max_val, 
                                mutated_params[param_name] + random.gauss(0, sigma)))
                        elif param_config['type'] == 'discrete':
                            choices = param_config['choices']
                            mutated_params[param_name] = random.choice(choices)
                        elif param_config['type'] == 'integer':
                            min_val = param_config['min']
                            max_val = param_config['max']
                            mutated_params[param_name] = random.randint(min_val, max_val)
            
            mutated_individual = Individual(parameters=mutated_params, objectives=[])
            return mutated_individual
            
        except Exception as e:
            self.logger.warning(f"⚠️ Mutation failed: {e}")
            return individual
    
    def _tournament_selection(self, population: List[Individual], tournament_size: int = None) -> Individual:
        """Select individual using tournament selection."""
        try:
            if tournament_size is None:
                tournament_size = self.config.tournament_size
            
            tournament = random.sample(population, min(tournament_size, len(population)))
            
            # For multi-objective, use Pareto ranking
            best_individual = tournament[0]
            for individual in tournament[1:]:
                if self._dominates(individual, best_individual):
                    best_individual = individual
            
            return best_individual
            
        except Exception as e:
            self.logger.warning(f"⚠️ Tournament selection failed: {e}")
            return population[0] if population else Individual(parameters={}, objectives=[])
    
    def _dominates(self, individual1: Individual, individual2: Individual) -> bool:
        """Check if individual1 dominates individual2."""
        try:
            if len(individual1.objectives) != len(individual2.objectives):
                return False
            
            # At least one objective is better
            better_in_one = False
            
            for obj1, obj2 in zip(individual1.objectives, individual2.objectives):
                if obj1 < obj2:  # Assuming minimization
                    return False
                elif obj1 > obj2:
                    better_in_one = True
            
            return better_in_one
            
        except Exception:
            return False


class NSGA2Optimizer(EvolutionaryAlgorithm):
    """NSGA-II (Non-dominated Sorting Genetic Algorithm II) optimizer."""
    
    def __init__(self, config: EvolutionaryAlgorithmConfig):
        """Initialize the NSGA-II optimizer.
        
        Args:
            config: Evolutionary algorithm configuration
        """
        super().__init__(config)
        self.logger.info("✅ NSGA-II Optimizer initialized")
    
    def optimize(self, objective_functions: List[Callable], parameter_space: Dict[str, Any]) -> EvolutionaryResult:
        """Optimize using NSGA-II algorithm.
        
        Args:
            objective_functions: List of objective functions to optimize
            parameter_space: Parameter space definition
            
        Returns:
            EvolutionaryResult with optimization results
        """
        try:
            self.logger.info("🧬 Starting NSGA-II optimization...")
            start_time = time.time()
            
            # Initialize population
            population = self._initialize_population(parameter_space)
            
            # Evaluate initial population
            for individual in population:
                individual.objectives = self._evaluate_objectives(individual, objective_functions)
            
            optimization_history = []
            
            # Evolution loop
            for generation in range(self.config.max_generations):
                # Create offspring
                offspring = []
                
                while len(offspring) < self.config.population_size:
                    # Tournament selection
                    parent1 = self._tournament_selection(population)
                    parent2 = self._tournament_selection(population)
                    
                    # Crossover
                    child1, child2 = self._crossover(parent1, parent2, parameter_space)
                    
                    # Mutation
                    child1 = self._mutate(child1, parameter_space)
                    child2 = self._mutate(child2, parameter_space)
                    
                    # Evaluate offspring
                    child1.objectives = self._evaluate_objectives(child1, objective_functions)
                    child2.objectives = self._evaluate_objectives(child2, objective_functions)
                    
                    offspring.extend([child1, child2])
                
                # Combine parent and offspring populations
                combined_population = population + offspring
                
                # Non-dominated sorting
                fronts = self._non_dominated_sorting(combined_population)
                
                # Crowding distance assignment
                for front in fronts:
                    self._assign_crowding_distance(front)
                
                # Select next generation
                new_population = []
                for front in fronts:
                    if len(new_population) + len(front) <= self.config.population_size:
                        new_population.extend(front)
                    else:
                        # Sort by crowding distance and select best
                        front.sort(key=lambda x: x.crowding_distance, reverse=True)
                        remaining = self.config.population_size - len(new_population)
                        new_population.extend(front[:remaining])
                        break
                
                population = new_population
                
                # Record generation statistics
                generation_stats = {
                    'generation': generation,
                    'population_size': len(population),
                    'pareto_front_size': len(fronts[0]) if fronts else 0,
                    'best_objectives': [min(ind.objectives[i] for ind in population) for i in range(len(objective_functions))]
                }
                optimization_history.append(generation_stats)
                
                # Check convergence
                if self._check_convergence(optimization_history):
                    self.logger.info(f"✅ Convergence reached at generation {generation}")
                    break
            
            execution_time = time.time() - start_time
            
            # Final non-dominated sorting
            fronts = self._non_dominated_sorting(population)
            pareto_front = fronts[0] if fronts else []
            
            # Create convergence info
            convergence_info = {
                'total_generations': len(optimization_history),
                'convergence_reached': len(optimization_history) < self.config.max_generations,
                'final_pareto_front_size': len(pareto_front),
                'best_objectives': [min(ind.objectives[i] for ind in population) for i in range(len(objective_functions))]
            }
            
            self.logger.info(f"✅ NSGA-II optimization completed in {execution_time:.2f}s")
            self.logger.info(f"   Pareto front size: {len(pareto_front)}")
            
            return EvolutionaryResult(
                best_individuals=population,
                pareto_front=pareto_front,
                optimization_history=optimization_history,
                convergence_info=convergence_info,
                execution_time=execution_time,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ NSGA-II optimization failed: {e}")
            return EvolutionaryResult(
                best_individuals=[],
                pareto_front=[],
                optimization_history=[],
                convergence_info={'error': str(e)},
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    def _non_dominated_sorting(self, population: List[Individual]) -> List[List[Individual]]:
        """Perform non-dominated sorting."""
        try:
            # Reset domination information
            for individual in population:
                individual.dominated_count = 0
                individual.dominated_solutions = []
            
            # Calculate domination relationships
            for i, individual1 in enumerate(population):
                for j, individual2 in enumerate(population):
                    if i != j:
                        if self._dominates(individual1, individual2):
                            individual1.dominated_solutions.append(j)
                        elif self._dominates(individual2, individual1):
                            individual1.dominated_count += 1
            
            # Create fronts
            fronts = []
            current_front = []
            
            # First front: non-dominated solutions
            for individual in population:
                if individual.dominated_count == 0:
                    individual.rank = 0
                    current_front.append(individual)
            
            fronts.append(current_front)
            
            # Subsequent fronts
            front_index = 0
            while fronts[front_index]:
                next_front = []
                
                for individual in fronts[front_index]:
                    for dominated_index in individual.dominated_solutions:
                        dominated_individual = population[dominated_index]
                        dominated_individual.dominated_count -= 1
                        
                        if dominated_individual.dominated_count == 0:
                            dominated_individual.rank = front_index + 1
                            next_front.append(dominated_individual)
                
                front_index += 1
                fronts.append(next_front)
            
            return fronts
            
        except Exception as e:
            self.logger.warning(f"⚠️ Non-dominated sorting failed: {e}")
            return [population]
    
    def _assign_crowding_distance(self, front: List[Individual]):
        """Assign crowding distance to individuals in a front."""
        try:
            if len(front) <= 2:
                for individual in front:
                    individual.crowding_distance = float('inf')
                return
            
            # Initialize crowding distance
            for individual in front:
                individual.crowding_distance = 0.0
            
            # Calculate crowding distance for each objective
            num_objectives = len(front[0].objectives)
            
            for obj_index in range(num_objectives):
                # Sort by objective value
                front.sort(key=lambda x: x.objectives[obj_index])
                
                # Set boundary points to infinity
                front[0].crowding_distance = float('inf')
                front[-1].crowding_distance = float('inf')
                
                # Calculate range
                obj_values = [individual.objectives[obj_index] for individual in front]
                obj_range = max(obj_values) - min(obj_values)
                
                if obj_range > 0:
                    # Update crowding distance
                    for i in range(1, len(front) - 1):
                        distance = (front[i + 1].objectives[obj_index] - front[i - 1].objectives[obj_index]) / obj_range
                        front[i].crowding_distance += distance
            
        except Exception as e:
            self.logger.warning(f"⚠️ Crowding distance assignment failed: {e}")
    
    def _check_convergence(self, optimization_history: List[Dict[str, Any]]) -> bool:
        """Check if optimization has converged."""
        try:
            if len(optimization_history) < 10:
                return False
            
            # Check if Pareto front size has stabilized
            recent_front_sizes = [record['pareto_front_size'] for record in optimization_history[-10:]]
            front_size_std = np.std(recent_front_sizes)
            
            return front_size_std < 1.0  # Front size has stabilized
            
        except Exception:
            return False


class SPEA2Optimizer(EvolutionaryAlgorithm):
    """SPEA2 (Strength Pareto Evolutionary Algorithm 2) optimizer."""
    
    def __init__(self, config: EvolutionaryAlgorithmConfig):
        """Initialize the SPEA2 optimizer.
        
        Args:
            config: Evolutionary algorithm configuration
        """
        super().__init__(config)
        self.logger.info("✅ SPEA2 Optimizer initialized")
    
    def optimize(self, objective_functions: List[Callable], parameter_space: Dict[str, Any]) -> EvolutionaryResult:
        """Optimize using SPEA2 algorithm.
        
        Args:
            objective_functions: List of objective functions to optimize
            parameter_space: Parameter space definition
            
        Returns:
            EvolutionaryResult with optimization results
        """
        try:
            self.logger.info("🧬 Starting SPEA2 optimization...")
            start_time = time.time()
            
            # Initialize population
            population = self._initialize_population(parameter_space)
            
            # Evaluate initial population
            for individual in population:
                individual.objectives = self._evaluate_objectives(individual, objective_functions)
            
            optimization_history = []
            
            # Evolution loop
            for generation in range(self.config.max_generations):
                # Create offspring
                offspring = []
                
                while len(offspring) < self.config.population_size:
                    # Tournament selection
                    parent1 = self._tournament_selection(population)
                    parent2 = self._tournament_selection(population)
                    
                    # Crossover
                    child1, child2 = self._crossover(parent1, parent2, parameter_space)
                    
                    # Mutation
                    child1 = self._mutate(child1, parameter_space)
                    child2 = self._mutate(child2, parameter_space)
                    
                    # Evaluate offspring
                    child1.objectives = self._evaluate_objectives(child1, objective_functions)
                    child2.objectives = self._evaluate_objectives(child2, objective_functions)
                    
                    offspring.extend([child1, child2])
                
                # Combine parent and offspring populations
                combined_population = population + offspring
                
                # Calculate fitness using SPEA2
                self._calculate_spea2_fitness(combined_population)
                
                # Environmental selection
                population = self._environmental_selection(combined_population)
                
                # Record generation statistics
                generation_stats = {
                    'generation': generation,
                    'population_size': len(population),
                    'best_objectives': [min(ind.objectives[i] for ind in population) for i in range(len(objective_functions))]
                }
                optimization_history.append(generation_stats)
                
                # Check convergence
                if self._check_convergence(optimization_history):
                    self.logger.info(f"✅ Convergence reached at generation {generation}")
                    break
            
            execution_time = time.time() - start_time
            
            # Get Pareto front
            pareto_front = [ind for ind in population if ind.fitness < 1.0]
            
            # Create convergence info
            convergence_info = {
                'total_generations': len(optimization_history),
                'convergence_reached': len(optimization_history) < self.config.max_generations,
                'final_pareto_front_size': len(pareto_front),
                'best_objectives': [min(ind.objectives[i] for ind in population) for i in range(len(objective_functions))]
            }
            
            self.logger.info(f"✅ SPEA2 optimization completed in {execution_time:.2f}s")
            self.logger.info(f"   Pareto front size: {len(pareto_front)}")
            
            return EvolutionaryResult(
                best_individuals=population,
                pareto_front=pareto_front,
                optimization_history=optimization_history,
                convergence_info=convergence_info,
                execution_time=execution_time,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ SPEA2 optimization failed: {e}")
            return EvolutionaryResult(
                best_individuals=[],
                pareto_front=[],
                optimization_history=[],
                convergence_info={'error': str(e)},
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    def _calculate_spea2_fitness(self, population: List[Individual]):
        """Calculate SPEA2 fitness for population."""
        try:
            # Calculate raw fitness (domination count)
            for individual in population:
                individual.fitness = 0
                for other in population:
                    if self._dominates(other, individual):
                        individual.fitness += 1
            
            # Calculate density estimation
            for individual in population:
                distances = []
                for other in population:
                    if other != individual:
                        distance = self._calculate_distance(individual, other)
                        distances.append(distance)
                
                distances.sort()
                if len(distances) >= self.config.population_size:
                    kth_distance = distances[self.config.population_size - 1]
                    individual.fitness += 1.0 / (kth_distance + 2.0)
                else:
                    individual.fitness += 1.0 / (distances[-1] + 2.0) if distances else 0.0
            
        except Exception as e:
            self.logger.warning(f"⚠️ SPEA2 fitness calculation failed: {e}")
    
    def _calculate_distance(self, individual1: Individual, individual2: Individual) -> float:
        """Calculate Euclidean distance between two individuals."""
        try:
            distance = 0.0
            for obj1, obj2 in zip(individual1.objectives, individual2.objectives):
                distance += (obj1 - obj2) ** 2
            return np.sqrt(distance)
        except Exception:
            return float('inf')
    
    def _environmental_selection(self, population: List[Individual]) -> List[Individual]:
        """Perform environmental selection."""
        try:
            # Sort by fitness
            population.sort(key=lambda x: x.fitness)
            
            # Select best individuals
            selected = population[:self.config.population_size]
            
            return selected
            
        except Exception as e:
            self.logger.warning(f"⚠️ Environmental selection failed: {e}")
            return population[:self.config.population_size]
    
    def _check_convergence(self, optimization_history: List[Dict[str, Any]]) -> bool:
        """Check if optimization has converged."""
        try:
            if len(optimization_history) < 10:
                return False
            
            # Check if best objectives have stabilized
            recent_best = [record['best_objectives'] for record in optimization_history[-10:]]
            if not recent_best:
                return False
            
            # Calculate variance in best objectives
            best_objectives_array = np.array(recent_best)
            objective_vars = np.var(best_objectives_array, axis=0)
            
            return np.all(objective_vars < self.config.convergence_threshold)
            
        except Exception:
            return False


class EvolutionaryAlgorithmManager:
    """Manager for coordinating different evolutionary algorithms."""
    
    def __init__(self, config: EvolutionaryAlgorithmConfig):
        """Initialize the evolutionary algorithm manager.
        
        Args:
            config: Evolutionary algorithm configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize optimizers
        self.nsga2_optimizer = None
        self.spea2_optimizer = None
        
        if config.use_nsga2:
            self.nsga2_optimizer = NSGA2Optimizer(config)
        
        if config.use_spea2:
            self.spea2_optimizer = SPEA2Optimizer(config)
        
        self.logger.info("✅ Evolutionary Algorithm Manager initialized")
    
    def optimize_with_algorithm(self, objective_functions: List[Callable], parameter_space: Dict[str, Any], 
                               algorithm: str = "auto") -> EvolutionaryResult:
        """Optimize using specified algorithm.
        
        Args:
            objective_functions: List of objective functions to optimize
            parameter_space: Parameter space definition
            algorithm: Optimization algorithm ("nsga2", "spea2", "auto")
            
        Returns:
            EvolutionaryResult with optimization results
        """
        try:
            self.logger.info(f"🧬 Starting optimization with algorithm: {algorithm}")
            
            if algorithm == "nsga2" or (algorithm == "auto" and self.nsga2_optimizer is not None):
                if self.nsga2_optimizer is None:
                    raise ValueError("NSGA-II optimizer not available")
                return self.nsga2_optimizer.optimize(objective_functions, parameter_space)
            
            elif algorithm == "spea2" or (algorithm == "auto" and self.spea2_optimizer is not None):
                if self.spea2_optimizer is None:
                    raise ValueError("SPEA2 optimizer not available")
                return self.spea2_optimizer.optimize(objective_functions, parameter_space)
            
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
                
        except Exception as e:
            self.logger.error(f"❌ Algorithm optimization failed: {e}")
            return EvolutionaryResult(
                best_individuals=[],
                pareto_front=[],
                optimization_history=[],
                convergence_info={'error': str(e)},
                execution_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    def compare_algorithms(self, objective_functions: List[Callable], parameter_space: Dict[str, Any]) -> Dict[str, EvolutionaryResult]:
        """Compare different evolutionary algorithms.
        
        Args:
            objective_functions: List of objective functions to optimize
            parameter_space: Parameter space definition
            
        Returns:
            Dictionary mapping algorithm names to results
        """
        try:
            results = {}
            
            # Run NSGA-II if available
            if self.nsga2_optimizer is not None:
                self.logger.info("🧬 Running NSGA-II optimization...")
                results['nsga2'] = self.nsga2_optimizer.optimize(objective_functions, parameter_space)
            
            # Run SPEA2 if available
            if self.spea2_optimizer is not None:
                self.logger.info("🧬 Running SPEA2 optimization...")
                results['spea2'] = self.spea2_optimizer.optimize(objective_functions, parameter_space)
            
            # Compare results
            if results:
                # Compare by Pareto front size and diversity
                best_algorithm = None
                best_score = -1
                
                for algorithm, result in results.items():
                    if result.success:
                        # Score based on Pareto front size and execution time
                        score = len(result.pareto_front) / (result.execution_time + 1.0)
                        if score > best_score:
                            best_score = score
                            best_algorithm = algorithm
                
                if best_algorithm:
                    self.logger.info(f"✅ Best algorithm: {best_algorithm} (score: {best_score:.4f})")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Algorithm comparison failed: {e}")
            return {}


def create_evolutionary_algorithm_manager(config: EvolutionaryAlgorithmConfig) -> EvolutionaryAlgorithmManager:
    """Create an evolutionary algorithm manager instance.
    
    Args:
        config: Evolutionary algorithm configuration
        
    Returns:
        EvolutionaryAlgorithmManager instance
    """
    return EvolutionaryAlgorithmManager(config)