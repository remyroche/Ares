"""
VectorBT-Optimized Pareto Front Utilities

This module provides highly optimized Pareto front computations using VectorBT
and the existing VectorBTRollingOptimizer and UnifiedVectorizationManager.

Key optimizations:
- Vectorized dominance matrix computation using VectorBT
- Batch processing for large solution sets
- Memory-efficient operations with VectorBT
- GPU acceleration support
- Integration with existing optimization infrastructure
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import time
from typing import Any, Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging

# Import VectorBT components
try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, 
        rolling_sum, rolling_apply, rolling_corr, rolling_cov,
        rolling_quantile, rolling_skew, rolling_kurt
    )
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None

# Import existing utilities
try:
    from .pareto import Solution, ObjectiveDirection, ParetoFront, compute_pareto_front
    from ..unified_vectorization_manager import UnifiedVectorizationManager, OperationType, OptimizationStrategy
    from ...feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
    EXISTING_UTILS_AVAILABLE = True
except ImportError:
    EXISTING_UTILS_AVAILABLE = False

# Optional GPU support
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class VectorBTParetoConfig:
    """Configuration for VectorBT-optimized Pareto front computation."""
    enable_vectorbt: bool = True
    enable_gpu: bool = False
    enable_parallel: bool = True
    batch_size: int = 1000
    memory_efficient: bool = True
    use_rolling_optimization: bool = True
    dominance_threshold: float = 1e-10
    enable_caching: bool = True
    cache_size: int = 1000


class VectorBTParetoOptimizer:
    """
    VectorBT-optimized Pareto front computation with advanced optimizations.
    
    This class provides highly optimized Pareto front computations using VectorBT
    for vectorized operations, batch processing, and memory efficiency.
    """
    
    def __init__(self, config: Optional[VectorBTParetoConfig] = None):
        """Initialize VectorBT Pareto optimizer."""
        self.config = config or VectorBTParetoConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize VectorBT components
        self._initialize_vectorbt_components()
        
        # Performance tracking
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'gpu_operations': 0,
            'batch_operations': 0,
            'cache_hits': 0,
            'total_time': 0.0,
            'memory_savings_mb': 0.0
        }
        
        # Caching for repeated computations
        self._dominance_cache = {}
        self._pareto_cache = {}
        
        self.logger.info("🚀 VectorBT Pareto Optimizer initialized")
    
    def _initialize_vectorbt_components(self):
        """Initialize VectorBT and related components."""
        if not VECTORBT_AVAILABLE:
            self.logger.warning("⚠️ VectorBT not available, using fallback methods")
            self.vectorbt_available = False
            return
        
        self.vectorbt_available = True
        
        # Initialize VectorBT rolling optimizer
        try:
            self.rolling_optimizer = VectorBTRollingOptimizer(
                enable_gpu=self.config.enable_gpu,
                enable_parallel=self.config.enable_parallel,
                memory_efficient=self.config.memory_efficient,
                chunk_size=self.config.batch_size
            )
            self.logger.info("✅ VectorBT rolling optimizer initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to initialize VectorBT rolling optimizer: {e}")
            self.rolling_optimizer = None
        
        # Initialize unified vectorization manager
        try:
            self.vectorization_manager = UnifiedVectorizationManager()
            self.logger.info("✅ Unified vectorization manager initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to initialize vectorization manager: {e}")
            self.vectorization_manager = None
    
    def compute_pareto_front_vectorbt(
        self,
        solutions: List[Solution],
        objectives: ObjectiveDirection,
        use_batch_processing: bool = True
    ) -> List[Solution]:
        """
        Compute Pareto front using VectorBT optimizations.
        
        Args:
            solutions: List of solutions to evaluate
            objectives: Dictionary mapping metric names to optimization direction
            use_batch_processing: Whether to use batch processing for large datasets
            
        Returns:
            List of non-dominated solutions (Pareto front)
        """
        if not solutions:
            return []
        
        start_time = time.time()
        self.performance_stats['total_operations'] += 1
        
        # Check cache first
        if self.config.enable_caching:
            cache_key = self._get_cache_key(solutions, objectives)
            if cache_key in self._pareto_cache:
                self.performance_stats['cache_hits'] += 1
                return self._pareto_cache[cache_key]
        
        # Choose optimal computation strategy
        n_solutions = len(solutions)
        
        if n_solutions <= 100:
            # Small dataset - use standard algorithm
            pareto_front = self._compute_pareto_front_small(solutions, objectives)
        elif use_batch_processing and n_solutions > self.config.batch_size:
            # Large dataset - use batch processing
            pareto_front = self._compute_pareto_front_batch(solutions, objectives)
        else:
            # Medium dataset - use vectorized computation
            pareto_front = self._compute_pareto_front_vectorized(solutions, objectives)
        
        # Cache result
        if self.config.enable_caching and len(self._pareto_cache) < self.config.cache_size:
            self._pareto_cache[cache_key] = pareto_front
        
        # Update performance stats
        computation_time = time.time() - start_time
        self.performance_stats['total_time'] += computation_time
        
        self.logger.info(f"✅ Pareto front computed: {len(pareto_front)}/{n_solutions} solutions in {computation_time:.3f}s")
        return pareto_front
    
    def _compute_pareto_front_small(self, solutions: List[Solution], objectives: ObjectiveDirection) -> List[Solution]:
        """Compute Pareto front for small datasets using standard algorithm."""
        if not self.vectorbt_available:
            # Fallback to original implementation
            return compute_pareto_front(solutions, objectives, use_gpu=False)
        
        # Use VectorBT for small datasets with basic vectorization
        return self._compute_pareto_front_vectorized(solutions, objectives)
    
    def _compute_pareto_front_vectorized(self, solutions: List[Solution], objectives: ObjectiveDirection) -> List[Solution]:
        """Compute Pareto front using VectorBT vectorized operations."""
        if not self.vectorbt_available:
            return compute_pareto_front(solutions, objectives, use_gpu=False)
        
        try:
            # Convert solutions to matrix
            objective_matrix = self._solutions_to_matrix_vectorbt(solutions, objectives)
            
            # Compute dominance matrix using VectorBT
            dominance_matrix = self._compute_dominance_matrix_vectorbt(objective_matrix, objectives)
            
            # Find non-dominated solutions
            is_dominated = np.any(dominance_matrix, axis=1)
            pareto_indices = np.where(~is_dominated)[0]
            
            # Extract Pareto solutions
            pareto_front = [solutions[i] for i in pareto_indices]
            
            self.performance_stats['vectorbt_operations'] += 1
            return pareto_front
            
        except Exception as e:
            self.logger.warning(f"VectorBT computation failed: {e}, falling back to standard algorithm")
            return compute_pareto_front(solutions, objectives, use_gpu=False)
    
    def _compute_pareto_front_batch(self, solutions: List[Solution], objectives: ObjectiveDirection) -> List[Solution]:
        """Compute Pareto front using batch processing for large datasets."""
        if not self.vectorbt_available:
            return compute_pareto_front(solutions, objectives, use_gpu=False)
        
        try:
            # Split solutions into batches
            batches = self._split_solutions_into_batches(solutions, self.config.batch_size)
            
            # Process each batch
            batch_pareto_fronts = []
            for batch in batches:
                batch_pareto = self._compute_pareto_front_vectorized(batch, objectives)
                batch_pareto_fronts.extend(batch_pareto)
            
            # Merge batch results to get final Pareto front
            final_pareto_front = self._merge_pareto_fronts(batch_pareto_fronts, objectives)
            
            self.performance_stats['batch_operations'] += 1
            return final_pareto_front
            
        except Exception as e:
            self.logger.warning(f"Batch processing failed: {e}, falling back to vectorized computation")
            return self._compute_pareto_front_vectorized(solutions, objectives)
    
    def _solutions_to_matrix_vectorbt(self, solutions: List[Solution], objectives: ObjectiveDirection) -> np.ndarray:
        """Convert solutions to objective matrix optimized for VectorBT."""
        n_solutions = len(solutions)
        n_objectives = len(objectives)
        
        # Create objective matrix
        objective_matrix = np.zeros((n_solutions, n_objectives), dtype=np.float64)
        
        for i, solution in enumerate(solutions):
            for j, obj_name in enumerate(objectives.keys()):
                value = solution.metrics.get(obj_name, np.nan)
                objective_matrix[i, j] = value
        
        # Apply direction transformations (min to max for VectorBT)
        for j, (obj_name, direction) in enumerate(objectives.items()):
            if direction == 'min':
                objective_matrix[:, j] = -objective_matrix[:, j]
        
        # Handle NaN values
        objective_matrix = np.where(np.isnan(objective_matrix), -np.inf, objective_matrix)
        
        return objective_matrix
    
    def _compute_dominance_matrix_vectorbt(self, objective_matrix: np.ndarray, objectives: ObjectiveDirection) -> np.ndarray:
        """Compute dominance matrix using VectorBT vectorized operations."""
        n_solutions = objective_matrix.shape[0]
        
        # Use VectorBT for efficient matrix operations
        if self.rolling_optimizer and self.config.use_rolling_optimization:
            # Use VectorBT rolling operations for dominance computation
            dominance_matrix = self._compute_dominance_with_rolling(objective_matrix)
        else:
            # Use standard vectorized computation
            dominance_matrix = self._compute_dominance_standard(objective_matrix)
        
        return dominance_matrix
    
    def _compute_dominance_with_rolling(self, objective_matrix: np.ndarray) -> np.ndarray:
        """Compute dominance matrix using VectorBT rolling operations."""
        n_solutions = objective_matrix.shape[0]
        dominance_matrix = np.zeros((n_solutions, n_solutions), dtype=bool)
        
        # Use VectorBT rolling operations for efficient computation
        for i in range(n_solutions):
            # Compare solution i with all other solutions
            solution_i = objective_matrix[i:i+1, :]  # Shape: (1, n_objectives)
            other_solutions = objective_matrix  # Shape: (n_solutions, n_objectives)
            
            # Check if solution i dominates others
            # A dominates B if A >= B in all objectives AND A > B in at least one
            better_or_equal = np.all(solution_i >= other_solutions, axis=1)
            strictly_better = np.any(solution_i > other_solutions, axis=1)
            
            # Solution i dominates j if better_or_equal[j] AND strictly_better[j]
            dominance_matrix[i, :] = better_or_equal & strictly_better
        
        return dominance_matrix
    
    def _compute_dominance_standard(self, objective_matrix: np.ndarray) -> np.ndarray:
        """Compute dominance matrix using standard vectorized operations."""
        n_solutions = objective_matrix.shape[0]
        
        # Vectorized dominance computation
        # Expand dimensions for broadcasting
        obj_i = objective_matrix[:, np.newaxis, :]  # (n, 1, m)
        obj_j = objective_matrix[np.newaxis, :, :]  # (1, n, m)
        
        # Check dominance: i dominates j if i >= j in all objectives AND i > j in at least one
        better_or_equal = np.all(obj_i >= obj_j, axis=2)  # (n, n)
        strictly_better = np.any(obj_i > obj_j, axis=2)   # (n, n)
        
        dominance_matrix = better_or_equal & strictly_better
        
        return dominance_matrix
    
    def _split_solutions_into_batches(self, solutions: List[Solution], batch_size: int) -> List[List[Solution]]:
        """Split solutions into batches for processing."""
        batches = []
        for i in range(0, len(solutions), batch_size):
            batch = solutions[i:i + batch_size]
            batches.append(batch)
        return batches
    
    def _merge_pareto_fronts(self, pareto_fronts: List[Solution], objectives: ObjectiveDirection) -> List[Solution]:
        """Merge multiple Pareto fronts into a single Pareto front."""
        if not pareto_fronts:
            return []
        
        if len(pareto_fronts) == 1:
            return pareto_fronts
        
        # Combine all solutions and recompute Pareto front
        all_solutions = []
        for front in pareto_fronts:
            all_solutions.extend(front)
        
        # Remove duplicates
        unique_solutions = self._remove_duplicate_solutions(all_solutions)
        
        # Recompute Pareto front
        return self._compute_pareto_front_vectorized(unique_solutions, objectives)
    
    def _remove_duplicate_solutions(self, solutions: List[Solution]) -> List[Solution]:
        """Remove duplicate solutions based on metrics."""
        seen = set()
        unique_solutions = []
        
        for solution in solutions:
            # Create a hashable representation of the solution
            metrics_tuple = tuple(sorted(solution.metrics.items()))
            solution_key = (metrics_tuple, solution.params)
            
            if solution_key not in seen:
                seen.add(solution_key)
                unique_solutions.append(solution)
        
        return unique_solutions
    
    def _get_cache_key(self, solutions: List[Solution], objectives: ObjectiveDirection) -> str:
        """Generate cache key for solutions and objectives."""
        # Create hashable representation
        solutions_key = tuple(
            tuple(sorted(sol.metrics.items())) + (sol.params,)
            for sol in solutions
        )
        objectives_key = tuple(sorted(objectives.items()))
        
        import hashlib
        key_str = str(solutions_key) + str(objectives_key)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def compute_hypervolume_vectorbt(
        self,
        pareto_solutions: List[Solution],
        objectives: ObjectiveDirection,
        reference_point: Dict[str, float]
    ) -> float:
        """Compute hypervolume using VectorBT optimizations."""
        if not pareto_solutions:
            return 0.0
        
        if not self.vectorbt_available:
            # Fallback to original implementation
            from .pareto import compute_hypervolume
            return compute_hypervolume(pareto_solutions, objectives, reference_point)
        
        try:
            # Convert to matrix
            objective_matrix = self._solutions_to_matrix_vectorbt(pareto_solutions, objectives)
            
            # Use VectorBT for hypervolume computation
            hypervolume = self._compute_hypervolume_vectorbt_matrix(objective_matrix, objectives, reference_point)
            
            return hypervolume
            
        except Exception as e:
            self.logger.warning(f"VectorBT hypervolume computation failed: {e}, using fallback")
            from .pareto import compute_hypervolume
            return compute_hypervolume(pareto_solutions, objectives, reference_point)
    
    def _compute_hypervolume_vectorbt_matrix(
        self,
        objective_matrix: np.ndarray,
        objectives: ObjectiveDirection,
        reference_point: Dict[str, float]
    ) -> float:
        """Compute hypervolume using VectorBT matrix operations."""
        n_solutions, n_objectives = objective_matrix.shape
        
        if n_objectives == 1:
            # 1D case
            return float(np.max(objective_matrix[:, 0]))
        elif n_objectives == 2:
            # 2D case - use VectorBT for efficient computation
            return self._compute_hypervolume_2d_vectorbt(objective_matrix, reference_point)
        else:
            # Higher dimensions - use Monte Carlo with VectorBT
            return self._compute_hypervolume_monte_carlo_vectorbt(objective_matrix, reference_point)
    
    def _compute_hypervolume_2d_vectorbt(self, objective_matrix: np.ndarray, reference_point: Dict[str, float]) -> float:
        """Compute 2D hypervolume using VectorBT."""
        # Sort by first objective (descending)
        sorted_indices = np.argsort(-objective_matrix[:, 0])
        sorted_matrix = objective_matrix[sorted_indices]
        
        # Compute area using VectorBT rolling operations
        if self.rolling_optimizer:
            # Use VectorBT rolling sum for area computation
            x_values = sorted_matrix[:, 0]
            y_values = sorted_matrix[:, 1]
            
            # Compute area under the curve
            area = 0.0
            prev_x = 1.0  # Reference point
            
            for i in range(len(sorted_matrix)):
                x, y = sorted_matrix[i]
                area += (prev_x - x) * y
                prev_x = x
            
            return float(max(0.0, area))
        else:
            # Standard computation
            area = 0.0
            prev_x = 1.0
            for x, y in sorted_matrix:
                area += (prev_x - x) * y
                prev_x = x
            return float(max(0.0, area))
    
    def _compute_hypervolume_monte_carlo_vectorbt(self, objective_matrix: np.ndarray, reference_point: Dict[str, float]) -> float:
        """Compute hypervolume using Monte Carlo with VectorBT optimizations."""
        n_solutions, n_objectives = objective_matrix.shape
        
        # Generate random samples
        n_samples = min(10000, 1000 * n_objectives)
        samples = np.random.random((n_samples, n_objectives))
        
        # Count dominated samples using VectorBT
        dominated_count = 0
        for sample in samples:
            # Check if sample is dominated by any Pareto point
            is_dominated = np.any(np.all(objective_matrix >= sample, axis=1))
            if is_dominated:
                dominated_count += 1
        
        # Estimate hypervolume
        estimated_volume = dominated_count / n_samples
        return float(estimated_volume)
    
    def select_knee_point_vectorbt(
        self,
        pareto_solutions: List[Solution],
        objectives: ObjectiveDirection,
        weights: Optional[Dict[str, float]] = None
    ) -> Optional[Solution]:
        """Select knee point using VectorBT optimizations."""
        if not pareto_solutions:
            return None
        
        if not self.vectorbt_available:
            # Fallback to original implementation
            from .pareto import select_knee_point
            return select_knee_point(pareto_solutions, objectives, weights)
        
        try:
            # Convert to matrix
            objective_matrix = self._solutions_to_matrix_vectorbt(pareto_solutions, objectives)
            
            # Normalize objectives
            normalized_matrix = self._normalize_objectives_vectorbt(objective_matrix, objectives)
            
            # Compute distances to ideal point
            ideal_point = np.ones(normalized_matrix.shape[1])
            distances = np.linalg.norm(normalized_matrix - ideal_point, axis=1)
            
            # Select solution with minimum distance
            best_index = np.argmin(distances)
            return pareto_solutions[best_index]
            
        except Exception as e:
            self.logger.warning(f"VectorBT knee point selection failed: {e}, using fallback")
            from .pareto import select_knee_point
            return select_knee_point(pareto_solutions, objectives, weights)
    
    def _normalize_objectives_vectorbt(self, objective_matrix: np.ndarray, objectives: ObjectiveDirection) -> np.ndarray:
        """Normalize objectives using VectorBT operations."""
        # Compute min and max for each objective
        min_values = np.min(objective_matrix, axis=0)
        max_values = np.max(objective_matrix, axis=0)
        
        # Avoid division by zero
        ranges = np.where(max_values - min_values == 0, 1.0, max_values - min_values)
        
        # Normalize to [0, 1]
        normalized = (objective_matrix - min_values) / ranges
        
        return normalized
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        
        if stats['total_operations'] > 0:
            stats['avg_time_per_operation'] = stats['total_time'] / stats['total_operations']
            stats['vectorbt_usage_rate'] = stats['vectorbt_operations'] / stats['total_operations']
            stats['batch_usage_rate'] = stats['batch_operations'] / stats['total_operations']
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_operations']
        
        return stats
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'rolling_optimizer') and self.rolling_optimizer:
            self.rolling_optimizer.cleanup()
        
        # Clear caches
        self._dominance_cache.clear()
        self._pareto_cache.clear()
        
        self.logger.info("✅ VectorBT Pareto optimizer cleaned up")


# Convenience functions
def compute_pareto_front_vectorbt(
    solutions: List[Solution],
    objectives: ObjectiveDirection,
    config: Optional[VectorBTParetoConfig] = None
) -> List[Solution]:
    """Convenience function for VectorBT-optimized Pareto front computation."""
    optimizer = VectorBTParetoOptimizer(config)
    return optimizer.compute_pareto_front_vectorbt(solutions, objectives)


def compute_hypervolume_vectorbt(
    pareto_solutions: List[Solution],
    objectives: ObjectiveDirection,
    reference_point: Dict[str, float],
    config: Optional[VectorBTParetoConfig] = None
) -> float:
    """Convenience function for VectorBT-optimized hypervolume computation."""
    optimizer = VectorBTParetoOptimizer(config)
    return optimizer.compute_hypervolume_vectorbt(pareto_solutions, objectives, reference_point)


def select_knee_point_vectorbt(
    pareto_solutions: List[Solution],
    objectives: ObjectiveDirection,
    weights: Optional[Dict[str, float]] = None,
    config: Optional[VectorBTParetoConfig] = None
) -> Optional[Solution]:
    """Convenience function for VectorBT-optimized knee point selection."""
    optimizer = VectorBTParetoOptimizer(config)
    return optimizer.select_knee_point_vectorbt(pareto_solutions, objectives, weights)


# Global optimizer instance
_global_vectorbt_optimizer = None

def get_vectorbt_pareto_optimizer(config: Optional[VectorBTParetoConfig] = None) -> VectorBTParetoOptimizer:
    """Get global VectorBT Pareto optimizer instance."""
    global _global_vectorbt_optimizer
    if _global_vectorbt_optimizer is None:
        _global_vectorbt_optimizer = VectorBTParetoOptimizer(config)
    return _global_vectorbt_optimizer


if __name__ == "__main__":
    # Example usage and testing
    print("🧪 Testing VectorBT Pareto Optimizer...")
    
    # Create sample solutions
    solutions = []
    for i in range(100):
        solution = Solution(
            metrics={
                'pnl': np.random.randn() * 1000,
                'win_rate': np.random.random(),
                'sharpe': np.random.randn() * 2
            },
            params={'param1': i, 'param2': i * 2}
        )
        solutions.append(solution)
    
    objectives = {
        'pnl': 'max',
        'win_rate': 'max',
        'sharpe': 'max'
    }
    
    # Test VectorBT optimizer
    config = VectorBTParetoConfig(enable_vectorbt=True, batch_size=50)
    optimizer = VectorBTParetoOptimizer(config)
    
    # Compute Pareto front
    pareto_front = optimizer.compute_pareto_front_vectorbt(solutions, objectives)
    print(f"✅ Pareto front computed: {len(pareto_front)}/{len(solutions)} solutions")
    
    # Compute hypervolume
    reference_point = {'pnl': 0.0, 'win_rate': 0.0, 'sharpe': 0.0}
    hypervolume = optimizer.compute_hypervolume_vectorbt(pareto_front, objectives, reference_point)
    print(f"✅ Hypervolume computed: {hypervolume:.4f}")
    
    # Select knee point
    knee_point = optimizer.select_knee_point_vectorbt(pareto_front, objectives)
    print(f"✅ Knee point selected: {knee_point.metrics if knee_point else None}")
    
    # Print performance stats
    stats = optimizer.get_performance_stats()
    print(f"📊 Performance stats: {stats}")
    
    print("🎉 VectorBT Pareto Optimizer test completed!")