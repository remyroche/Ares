"""
Multi-Objective Optimizer for Hybrid NAS-TAS Regime Detection.

This module provides multi-objective optimization for hybrid regime detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import logging

# Import tprint for comprehensive logging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

logger = logging.getLogger(__name__)

class ObjectiveType(Enum):
    """Types of optimization objectives."""
    ACCURACY = "accuracy"
    EFFICIENCY = "efficiency"
    STABILITY = "stability"
    PROFITABILITY = "profitability"

@dataclass
class MultiObjectiveConfig:
    """Configuration for multi-objective optimization."""
    objectives: List[ObjectiveType]
    weights: List[float]
    max_iterations: int = 100
    convergence_threshold: float = 0.01

@dataclass
class OptimizationResult:
    """Multi-objective optimization result."""
    best_parameters: Dict[str, Any]
    best_scores: Dict[ObjectiveType, float]
    optimization_history: List[Dict[ObjectiveType, float]]
    convergence_achieved: bool

class TradingMultiObjectiveOptimizer:
    """Multi-objective optimizer for trading regime detection."""
    
    def __init__(self, config: MultiObjectiveConfig):
        """Initialize the multi-objective optimizer."""
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def optimize(
        self,
        objective_functions: Dict[ObjectiveType, Callable],
        parameter_bounds: Dict[str, Tuple[float, float]]
    ) -> OptimizationResult:
        """Optimize multiple objectives simultaneously."""
        try:
            # Initialize results
            best_scores = {obj: float('-inf') for obj in self.config.objectives}
            best_parameters = {}
            optimization_history = []
            
            # Generate parameter combinations
            param_combinations = self._generate_parameter_combinations(parameter_bounds)
            
            for i, params in enumerate(param_combinations):
                # Calculate scores for each objective
                scores = {}
                for obj_type in self.config.objectives:
                    if obj_type in objective_functions:
                        try:
                            score = objective_functions[obj_type](params)
                            scores[obj_type] = score
                        except Exception as e:
                            self.logger.warning(f"Objective {obj_type} failed: {e}")
                            scores[obj_type] = 0.0
                    else:
                        scores[obj_type] = 0.0
                
                # Store optimization history
                optimization_history.append(scores.copy())
                
                # Check if this is the best combination
                if self._is_better_solution(scores, best_scores):
                    best_scores = scores.copy()
                    best_parameters = params.copy()
                
                if i >= self.config.max_iterations:
                    break
            
            # Check convergence
            convergence_achieved = self._check_convergence(optimization_history)
            
            return OptimizationResult(
                best_parameters=best_parameters,
                best_scores=best_scores,
                optimization_history=optimization_history,
                convergence_achieved=convergence_achieved
            )
            
        except Exception as e:
            self.logger.error(f"Error in multi-objective optimization: {e}")
            return OptimizationResult({}, {}, [], False)
    
    def _generate_parameter_combinations(
        self, 
        parameter_bounds: Dict[str, Tuple[float, float]]
    ) -> List[Dict[str, Any]]:
        """Generate parameter combinations for optimization."""
        combinations = []
        
        # Simple random sampling
        for i in range(self.config.max_iterations):
            params = {}
            for name, (min_val, max_val) in parameter_bounds.items():
                params[name] = np.random.uniform(min_val, max_val)
            combinations.append(params)
        
        return combinations
    
    def _is_better_solution(
        self, 
        current_scores: Dict[ObjectiveType, float], 
        best_scores: Dict[ObjectiveType, float]
    ) -> bool:
        """Check if current solution is better than best solution."""
        # Weighted sum comparison
        current_weighted = sum(
            self.config.weights[i] * current_scores.get(obj, 0.0) 
            for i, obj in enumerate(self.config.objectives)
        )
        best_weighted = sum(
            self.config.weights[i] * best_scores.get(obj, 0.0) 
            for i, obj in enumerate(self.config.objectives)
        )
        
        return current_weighted > best_weighted
    
    def _check_convergence(
        self, 
        optimization_history: List[Dict[ObjectiveType, float]]
    ) -> bool:
        """Check if optimization has converged."""
        if len(optimization_history) < 10:
            return False
        
        # Check if recent improvements are below threshold
        recent_scores = optimization_history[-10:]
        improvements = []
        
        for i in range(1, len(recent_scores)):
            current_weighted = sum(
                self.config.weights[j] * recent_scores[i].get(obj, 0.0) 
                for j, obj in enumerate(self.config.objectives)
            )
            previous_weighted = sum(
                self.config.weights[j] * recent_scores[i-1].get(obj, 0.0) 
                for j, obj in enumerate(self.config.objectives)
            )
            improvement = current_weighted - previous_weighted
            improvements.append(improvement)
        
        avg_improvement = np.mean(improvements)
        return avg_improvement < self.config.convergence_threshold
