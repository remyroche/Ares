"""
ML Common - Optimization Parallel Processing Module

This module provides parallel processing utilities for optimization.
"""

from src.utils.parallel_processing_optimizer import ParallelProcessor


class ParallelProcessingCoordinator:
    """Coordinator for parallel processing in optimization."""

    def __init__(self):
        self.parallel_processor = ParallelProcessor()

    def optimize_parallel(self, optimization_function, parameter_sets, n_jobs=-1):
        """Execute optimization in parallel."""
        # Simple implementation - just run sequentially for now
        results = []
        for params in parameter_sets:
            result = optimization_function(params)
            results.append(result)
        return results

    def evaluate_population_parallel(self, population, fitness_function, n_jobs=-1):
        """Evaluate population in parallel."""
        # Simple implementation - just run sequentially for now
        fitness_scores = []
        for individual in population:
            score = fitness_function(individual)
            fitness_scores.append(score)
        return fitness_scores


__all__ = ['ParallelProcessingCoordinator']

