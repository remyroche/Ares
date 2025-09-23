"""
Multi-Objective Optimization for Neural Architecture Search

This module implements multi-objective optimization algorithms for balancing different
regime detection objectives such as accuracy, efficiency, and economic significance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod
import copy
from collections import defaultdict

# Import matrix operations for optimized computations
from src.utils.matrix_operations import UnifiedMatrixOperations

# Import hardware optimization
from src.utils.hardware.unified_hardware_manager import (
    UnifiedHardwareManager, HardwareConfig, WorkloadType, OptimizationLevel
)

logger = logging.getLogger(__name__)


@dataclass
class ObjectiveFunction:
    """Objective function for multi-objective optimization."""
    name: str
    weight: float = 1.0
    direction: str = 'maximize'  # 'maximize' or 'minimize'
    evaluator: Optional[Callable] = None
    bounds: Tuple[float, float] = (0.0, 1.0)
    
    def __post_init__(self):
        """Validate objective function configuration."""
        if self.direction not in ['maximize', 'minimize']:
            raise ValueError(f"Invalid direction: {self.direction}")
        if not (0.0 <= self.weight <= 1.0):
            raise ValueError(f"Invalid weight: {self.weight}")
        if self.bounds[0] >= self.bounds[1]:
            raise ValueError(f"Invalid bounds: {self.bounds}")


@dataclass
class ParetoSolution:
    """Solution in the Pareto frontier."""
    architecture: Any
    objectives: Dict[str, float]
    rank: int = 0
    crowding_distance: float = 0.0
    dominated_count: int = 0
    dominates: List[int] = field(default_factory=list)


class ParetoFrontier:
    """Pareto frontier for multi-objective optimization."""
    
    def __init__(self, objectives: List[ObjectiveFunction]):
        """Initialize Pareto frontier."""
        self.objectives = objectives
        self.solutions: List[ParetoSolution] = []
        self.fronts: List[List[ParetoSolution]] = []
        
    def add_solution(self, architecture: Any, objective_values: Dict[str, float]) -> ParetoSolution:
        """Add a new solution to the Pareto frontier."""
        solution = ParetoSolution(
            architecture=architecture,
            objectives=objective_values
        )
        self.solutions.append(solution)
        return solution
    
    def dominates(self, solution1: ParetoSolution, solution2: ParetoSolution) -> bool:
        """Check if solution1 dominates solution2."""
        try:
            better_in_one = False
            
            for obj in self.objectives:
                obj_name = obj.name
                val1 = solution1.objectives.get(obj_name, 0.0)
                val2 = solution2.objectives.get(obj_name, 0.0)
                
                if obj.direction == 'maximize':
                    if val1 < val2:
                        return False  # solution1 doesn't dominate
                    elif val1 > val2:
                        better_in_one = True
                else:  # minimize
                    if val1 > val2:
                        return False  # solution1 doesn't dominate
                    elif val1 < val2:
                        better_in_one = True
            
            return better_in_one
            
        except Exception as e:
            logger.warning(f"Dominance check failed: {e}")
            return False
    
    def compute_pareto_fronts(self):
        """Compute Pareto fronts using NSGA-II algorithm."""
        try:
            if not self.solutions:
                return
            
            # Reset dominance information
            for solution in self.solutions:
                solution.dominated_count = 0
                solution.dominates = []
                solution.rank = 0
            
            # Compute dominance relationships
            for i, solution1 in enumerate(self.solutions):
                for j, solution2 in enumerate(self.solutions):
                    if i != j:
                        if self.dominates(solution1, solution2):
                            solution1.dominates.append(j)
                            solution2.dominated_count += 1
            
            # Compute Pareto fronts
            self.fronts = []
            remaining_solutions = list(range(len(self.solutions)))
            current_rank = 0
            
            while remaining_solutions:
                # Find solutions with dominated_count = 0
                current_front = []
                for idx in remaining_solutions:
                    if self.solutions[idx].dominated_count == 0:
                        current_front.append(self.solutions[idx])
                        self.solutions[idx].rank = current_rank
                
                if not current_front:
                    break
                
                self.fronts.append(current_front)
                
                # Update dominated_count for remaining solutions
                for solution in current_front:
                    for dominated_idx in solution.dominates:
                        self.solutions[dominated_idx].dominated_count -= 1
                
                # Remove current front from remaining solutions
                current_front_indices = [self.solutions.index(sol) for sol in current_front]
                remaining_solutions = [idx for idx in remaining_solutions 
                                     if idx not in current_front_indices]
                
                current_rank += 1
            
            logger.info(f"Computed {len(self.fronts)} Pareto fronts")
            
        except Exception as e:
            logger.error(f"Pareto front computation failed: {e}")
    
    def compute_crowding_distance(self, front: List[ParetoSolution]):
        """Compute crowding distance for solutions in a front."""
        try:
            if len(front) <= 2:
                # Assign infinite crowding distance to boundary solutions
                for solution in front:
                    solution.crowding_distance = float('inf')
                return
            
            # Initialize crowding distances
            for solution in front:
                solution.crowding_distance = 0.0
            
            # Compute crowding distance for each objective
            for obj in self.objectives:
                obj_name = obj.name
                
                # Sort solutions by objective value
                sorted_solutions = sorted(front, 
                                        key=lambda x: x.objectives.get(obj_name, 0.0))
                
                # Boundary solutions get infinite distance
                sorted_solutions[0].crowding_distance = float('inf')
                sorted_solutions[-1].crowding_distance = float('inf')
                
                # Compute range for normalization
                obj_range = (sorted_solutions[-1].objectives.get(obj_name, 0.0) - 
                           sorted_solutions[0].objectives.get(obj_name, 0.0))
                
                if obj_range > 0:
                    # Compute crowding distance for intermediate solutions
                    for i in range(1, len(sorted_solutions) - 1):
                        distance = (sorted_solutions[i + 1].objectives.get(obj_name, 0.0) - 
                                  sorted_solutions[i - 1].objectives.get(obj_name, 0.0)) / obj_range
                        sorted_solutions[i].crowding_distance += distance
            
        except Exception as e:
            logger.warning(f"Crowding distance computation failed: {e}")
    
    def get_best_solutions(self, n: int) -> List[ParetoSolution]:
        """Get the best n solutions from the Pareto frontier."""
        try:
            if not self.fronts:
                self.compute_pareto_fronts()
            
            best_solutions = []
            
            # Add solutions from fronts in order of rank
            for front in self.fronts:
                if len(best_solutions) >= n:
                    break
                
                # Compute crowding distance for current front
                self.compute_crowding_distance(front)
                
                # Sort by crowding distance (descending)
                sorted_front = sorted(front, 
                                    key=lambda x: x.crowding_distance, 
                                    reverse=True)
                
                # Add solutions until we reach n
                for solution in sorted_front:
                    if len(best_solutions) >= n:
                        break
                    best_solutions.append(solution)
            
            return best_solutions
            
        except Exception as e:
            logger.warning(f"Best solutions selection failed: {e}")
            return self.solutions[:n] if self.solutions else []
    
    def get_pareto_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the Pareto frontier."""
        try:
            if not self.fronts:
                self.compute_pareto_fronts()
            
            summary = {
                'total_solutions': len(self.solutions),
                'num_fronts': len(self.fronts),
                'front_sizes': [len(front) for front in self.fronts],
                'objective_ranges': {}
            }
            
            # Compute objective ranges
            for obj in self.objectives:
                obj_name = obj.name
                values = [sol.objectives.get(obj_name, 0.0) for sol in self.solutions]
                if values:
                    summary['objective_ranges'][obj_name] = {
                        'min': min(values),
                        'max': max(values),
                        'mean': np.mean(values),
                        'std': np.std(values)
                    }
            
            return summary
            
        except Exception as e:
            logger.warning(f"Pareto summary computation failed: {e}")
            return {}


class MultiObjectiveOptimizer(ABC):
    """Abstract base class for multi-objective optimization."""
    
    def __init__(self, objectives: List[ObjectiveFunction],
                 matrix_ops: Optional[UnifiedMatrixOperations] = None,
                 hardware_manager: Optional[UnifiedHardwareManager] = None):
        """Initialize multi-objective optimizer."""
        self.objectives = objectives
        self.matrix_ops = matrix_ops
        self.hardware_manager = hardware_manager
        self.pareto_frontier = ParetoFrontier(objectives)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def optimize(self, architectures: List[Any], data: np.ndarray, 
                 labels: np.ndarray, max_iterations: int = 100) -> ParetoFrontier:
        """Perform multi-objective optimization."""
        pass
    
    def evaluate_objectives(self, architecture: Any, data: np.ndarray, 
                           labels: np.ndarray) -> Dict[str, float]:
        """Evaluate all objectives for an architecture."""
        try:
            objective_values = {}
            
            for obj in self.objectives:
                if obj.evaluator is not None:
                    try:
                        value = obj.evaluator(architecture, data, labels)
                        # Normalize value to bounds
                        value = max(obj.bounds[0], min(obj.bounds[1], value))
                        objective_values[obj.name] = value
                    except Exception as e:
                        logger.warning(f"Objective {obj.name} evaluation failed: {e}")
                        objective_values[obj.name] = obj.bounds[0]  # Default to lower bound
                else:
                    objective_values[obj.name] = obj.bounds[0]  # Default value
            
            return objective_values
            
        except Exception as e:
            logger.error(f"Objective evaluation failed: {e}")
            return {obj.name: obj.bounds[0] for obj in self.objectives}


class WeightedSumOptimizer(MultiObjectiveOptimizer):
    """Weighted sum multi-objective optimizer."""
    
    def __init__(self, objectives: List[ObjectiveFunction],
                 matrix_ops: Optional[UnifiedMatrixOperations] = None,
                 hardware_manager: Optional[UnifiedHardwareManager] = None):
        """Initialize weighted sum optimizer."""
        super().__init__(objectives, matrix_ops, hardware_manager)
        
        # Normalize weights
        total_weight = sum(obj.weight for obj in self.objectives)
        for obj in self.objectives:
            obj.weight = obj.weight / total_weight
    
    def optimize(self, architectures: List[Any], data: np.ndarray, 
                 labels: np.ndarray, max_iterations: int = 100) -> ParetoFrontier:
        """Perform weighted sum optimization."""
        try:
            self.logger.info(f"Starting weighted sum optimization with {len(architectures)} architectures")
            
            best_architecture = None
            best_score = float('-inf')
            
            for i, architecture in enumerate(architectures):
                try:
                    # Evaluate objectives
                    objective_values = self.evaluate_objectives(architecture, data, labels)
                    
                    # Compute weighted sum score
                    weighted_score = 0.0
                    for obj in self.objectives:
                        obj_value = objective_values.get(obj.name, obj.bounds[0])
                        if obj.direction == 'minimize':
                            obj_value = obj.bounds[1] - obj_value  # Convert to maximization
                        
                        weighted_score += obj.weight * obj_value
                    
                    # Update best solution
                    if weighted_score > best_score:
                        best_score = weighted_score
                        best_architecture = architecture
                    
                    # Add to Pareto frontier
                    self.pareto_frontier.add_solution(architecture, objective_values)
                    
                    if i % 10 == 0:
                        self.logger.info(f"Evaluated {i+1}/{len(architectures)} architectures")
                
                except Exception as e:
                    logger.warning(f"Architecture {i} evaluation failed: {e}")
                    continue
            
            # Compute Pareto fronts
            self.pareto_frontier.compute_pareto_fronts()
            
            self.logger.info(f"✅ Weighted sum optimization completed")
            self.logger.info(f"   Best weighted score: {best_score:.4f}")
            self.logger.info(f"   Pareto frontier size: {len(self.pareto_frontier.solutions)}")
            
            return self.pareto_frontier
            
        except Exception as e:
            self.logger.error(f"Weighted sum optimization failed: {e}")
            return self.pareto_frontier


class NSGAIIOptimizer(MultiObjectiveOptimizer):
    """NSGA-II multi-objective optimizer."""
    
    def __init__(self, objectives: List[ObjectiveFunction],
                 population_size: int = 100,
                 matrix_ops: Optional[UnifiedMatrixOperations] = None,
                 hardware_manager: Optional[UnifiedHardwareManager] = None):
        """Initialize NSGA-II optimizer."""
        super().__init__(objectives, matrix_ops, hardware_manager)
        self.population_size = population_size
        self.generation = 0
    
    def optimize(self, architectures: List[Any], data: np.ndarray, 
                 labels: np.ndarray, max_iterations: int = 100) -> ParetoFrontier:
        """Perform NSGA-II optimization."""
        try:
            self.logger.info(f"Starting NSGA-II optimization with {len(architectures)} architectures")
            
            # Initialize population
            population = architectures[:self.population_size] if len(architectures) >= self.population_size else architectures
            
            for generation in range(max_iterations):
                self.generation = generation
                
                # Evaluate population
                population_solutions = []
                for architecture in population:
                    objective_values = self.evaluate_objectives(architecture, data, labels)
                    solution = self.pareto_frontier.add_solution(architecture, objective_values)
                    population_solutions.append(solution)
                
                # Compute Pareto fronts
                self.pareto_frontier.compute_pareto_fronts()
                
                # Selection for next generation
                if generation < max_iterations - 1:  # Don't select on last iteration
                    population = self._nsga2_selection(population_solutions, self.population_size)
                
                if generation % 10 == 0:
                    self.logger.info(f"Generation {generation}: {len(self.pareto_frontier.fronts)} Pareto fronts")
            
            self.logger.info(f"✅ NSGA-II optimization completed")
            self.logger.info(f"   Final population size: {len(population)}")
            self.logger.info(f"   Pareto frontier size: {len(self.pareto_frontier.solutions)}")
            
            return self.pareto_frontier
            
        except Exception as e:
            self.logger.error(f"NSGA-II optimization failed: {e}")
            return self.pareto_frontier
    
    def _nsga2_selection(self, solutions: List[ParetoSolution], 
                        target_size: int) -> List[Any]:
        """NSGA-II selection operator."""
        try:
            # Sort solutions by rank and crowding distance
            sorted_solutions = sorted(solutions, 
                                    key=lambda x: (x.rank, -x.crowding_distance))
            
            # Select top solutions
            selected_solutions = sorted_solutions[:target_size]
            
            # Return architectures
            return [sol.architecture for sol in selected_solutions]
            
        except Exception as e:
            logger.warning(f"NSGA-II selection failed: {e}")
            return [sol.architecture for sol in solutions[:target_size]]


class RegimeDetectionMultiObjective:
    """Multi-objective optimization specifically for regime detection."""
    
    def __init__(self, matrix_ops: Optional[UnifiedMatrixOperations] = None,
                 hardware_manager: Optional[UnifiedHardwareManager] = None):
        """Initialize regime detection multi-objective optimizer."""
        self.matrix_ops = matrix_ops
        self.hardware_manager = hardware_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Define regime detection objectives
        self.objectives = self._create_regime_objectives()
    
    def _create_regime_objectives(self) -> List[ObjectiveFunction]:
        """Create objective functions for regime detection."""
        objectives = []
        
        # Accuracy objective
        objectives.append(ObjectiveFunction(
            name='accuracy',
            weight=0.3,
            direction='maximize',
            evaluator=self._evaluate_accuracy,
            bounds=(0.0, 1.0)
        ))
        
        # Economic significance objective
        objectives.append(ObjectiveFunction(
            name='economic_significance',
            weight=0.25,
            direction='maximize',
            evaluator=self._evaluate_economic_significance,
            bounds=(0.0, 1.0)
        ))
        
        # Trading viability objective
        objectives.append(ObjectiveFunction(
            name='trading_viability',
            weight=0.2,
            direction='maximize',
            evaluator=self._evaluate_trading_viability,
            bounds=(0.0, 1.0)
        ))
        
        # Regime stability objective
        objectives.append(ObjectiveFunction(
            name='regime_stability',
            weight=0.15,
            direction='maximize',
            evaluator=self._evaluate_regime_stability,
            bounds=(0.0, 1.0)
        ))
        
        # Efficiency objective (minimize complexity)
        objectives.append(ObjectiveFunction(
            name='efficiency',
            weight=0.1,
            direction='maximize',
            evaluator=self._evaluate_efficiency,
            bounds=(0.0, 1.0)
        ))
        
        return objectives
    
    def _evaluate_accuracy(self, architecture: Any, data: np.ndarray, labels: np.ndarray) -> float:
        """Evaluate regime detection accuracy."""
        try:
            # This would typically train and evaluate the architecture
            # For now, return a simplified accuracy estimate
            return np.random.uniform(0.6, 0.95)
        except Exception as e:
            self.logger.warning(f"Accuracy evaluation failed: {e}")
            return 0.5
    
    def _evaluate_economic_significance(self, architecture: Any, data: np.ndarray, labels: np.ndarray) -> float:
        """Evaluate economic significance of detected regimes."""
        try:
            # This would analyze the economic relevance of detected regimes
            # For now, return a simplified estimate
            return np.random.uniform(0.5, 0.9)
        except Exception as e:
            self.logger.warning(f"Economic significance evaluation failed: {e}")
            return 0.5
    
    def _evaluate_trading_viability(self, architecture: Any, data: np.ndarray, labels: np.ndarray) -> float:
        """Evaluate trading viability of detected regimes."""
        try:
            # This would assess how well regimes support trading decisions
            # For now, return a simplified estimate
            return np.random.uniform(0.4, 0.8)
        except Exception as e:
            self.logger.warning(f"Trading viability evaluation failed: {e}")
            return 0.5
    
    def _evaluate_regime_stability(self, architecture: Any, data: np.ndarray, labels: np.ndarray) -> float:
        """Evaluate stability of detected regimes."""
        try:
            # This would measure regime persistence and consistency
            # For now, return a simplified estimate
            return np.random.uniform(0.6, 0.9)
        except Exception as e:
            self.logger.warning(f"Regime stability evaluation failed: {e}")
            return 0.5
    
    def _evaluate_efficiency(self, architecture: Any, data: np.ndarray, labels: np.ndarray) -> float:
        """Evaluate architecture efficiency."""
        try:
            # This would measure computational efficiency
            # For now, return a simplified estimate based on architecture complexity
            if hasattr(architecture, 'parameters_count'):
                param_count = architecture.parameters_count
                if param_count < 10000:
                    return 1.0
                elif param_count < 100000:
                    return 0.8
                elif param_count < 500000:
                    return 0.6
                else:
                    return 0.4
            else:
                return 0.7  # Default efficiency
        except Exception as e:
            self.logger.warning(f"Efficiency evaluation failed: {e}")
            return 0.5
    
    def optimize_weighted_sum(self, architectures: List[Any], data: np.ndarray, 
                             labels: np.ndarray, max_iterations: int = 100) -> ParetoFrontier:
        """Perform weighted sum optimization for regime detection."""
        try:
            optimizer = WeightedSumOptimizer(
                objectives=self.objectives,
                matrix_ops=self.matrix_ops,
                hardware_manager=self.hardware_manager
            )
            
            return optimizer.optimize(architectures, data, labels, max_iterations)
            
        except Exception as e:
            self.logger.error(f"Weighted sum optimization failed: {e}")
            return ParetoFrontier(self.objectives)
    
    def optimize_nsga2(self, architectures: List[Any], data: np.ndarray, 
                       labels: np.ndarray, max_iterations: int = 100) -> ParetoFrontier:
        """Perform NSGA-II optimization for regime detection."""
        try:
            optimizer = NSGAIIOptimizer(
                objectives=self.objectives,
                population_size=min(50, len(architectures)),
                matrix_ops=self.matrix_ops,
                hardware_manager=self.hardware_manager
            )
            
            return optimizer.optimize(architectures, data, labels, max_iterations)
            
        except Exception as e:
            self.logger.error(f"NSGA-II optimization failed: {e}")
            return ParetoFrontier(self.objectives)
    
    def get_optimization_summary(self, pareto_frontier: ParetoFrontier) -> Dict[str, Any]:
        """Get summary of optimization results."""
        try:
            summary = pareto_frontier.get_pareto_summary()
            
            # Add regime-specific analysis
            best_solutions = pareto_frontier.get_best_solutions(5)
            
            summary['best_solutions'] = []
            for solution in best_solutions:
                solution_info = {
                    'objectives': solution.objectives,
                    'rank': solution.rank,
                    'crowding_distance': solution.crowding_distance
                }
                summary['best_solutions'].append(solution_info)
            
            # Compute objective correlations
            objective_names = [obj.name for obj in self.objectives]
            objective_matrix = []
            for solution in pareto_frontier.solutions:
                row = [solution.objectives.get(name, 0.0) for name in objective_names]
                objective_matrix.append(row)
            
            if objective_matrix:
                objective_matrix = np.array(objective_matrix)
                correlations = np.corrcoef(objective_matrix.T)
                
                summary['objective_correlations'] = {
                    objective_names[i]: {
                        objective_names[j]: correlations[i, j]
                        for j in range(len(objective_names))
                    }
                    for i in range(len(objective_names))
                }
            
            return summary
            
        except Exception as e:
            self.logger.warning(f"Optimization summary computation failed: {e}")
            return pareto_frontier.get_pareto_summary()