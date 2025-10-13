"""
Pareto Front VectorBT Integration Module

This module provides seamless integration of VectorBT optimizations into the
existing Pareto front utilities, offering automatic optimization selection
and enhanced performance for large datasets.

Key features:
- Automatic optimization selection based on data size
- Seamless integration with existing Pareto front utilities
- VectorBT acceleration for large datasets
- Backward compatibility with existing code
- Performance monitoring and statistics
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import time
from typing import Any, Dict, List, Tuple, Optional, Union, Callable
import logging

# Import original Pareto utilities
from .pareto import (
    Solution, ObjectiveDirection, ParetoFront, compute_pareto_front,
    select_knee_point, compute_hypervolume, scalarize_financial_goals,
    filter_by_constraints, DEFAULT_FINANCIAL_WEIGHTS, get_pareto_front
)

# Import VectorBT optimizations
try:
    from .vectorbt_pareto_optimizer import (
        VectorBTParetoOptimizer, VectorBTParetoConfig,
        compute_pareto_front_vectorbt, compute_hypervolume_vectorbt,
        select_knee_point_vectorbt, get_vectorbt_pareto_optimizer
    )
    VECTORBT_PARETO_AVAILABLE = True
except ImportError:
    VECTORBT_PARETO_AVAILABLE = False

# Import enhanced Pareto front
try:
    from .enhanced_pareto_front import (
        EnhancedParetoFront, EnhancedParetoConfig,
        compute_pareto_front_enhanced, compute_hypervolume_enhanced,
        select_knee_point_enhanced, get_enhanced_pareto_front
    )
    ENHANCED_PARETO_AVAILABLE = True
except ImportError:
    ENHANCED_PARETO_AVAILABLE = False

logger = logging.getLogger(__name__)


class ParetoVectorBTIntegration:
    """
    Integration class for VectorBT optimizations with Pareto front utilities.
    
    This class provides a unified interface that automatically selects the best
    optimization strategy and integrates seamlessly with existing code.
    """
    
    def __init__(self, enable_vectorbt: bool = True, vectorbt_threshold: int = 1000):
        """Initialize Pareto VectorBT integration."""
        self.enable_vectorbt = enable_vectorbt and VECTORBT_PARETO_AVAILABLE
        self.vectorbt_threshold = vectorbt_threshold
        self.logger = logging.getLogger(__name__)
        
        # Initialize VectorBT optimizer
        if self.enable_vectorbt:
            try:
                self.vectorbt_optimizer = get_vectorbt_pareto_optimizer()
                self.logger.info("✅ VectorBT Pareto optimizer initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize VectorBT optimizer: {e}")
                self.vectorbt_optimizer = None
                self.enable_vectorbt = False
        else:
            self.vectorbt_optimizer = None
        
        # Performance tracking
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'standard_operations': 0,
            'total_time': 0.0,
            'optimization_selections': {}
        }
        
        self.logger.info("🚀 Pareto VectorBT Integration initialized")
    
    def compute_pareto_front(
        self,
        solutions: List[Solution],
        objectives: ObjectiveDirection,
        use_gpu: bool = True,
        use_vectorbt: Optional[bool] = None
    ) -> List[Solution]:
        """
        Compute Pareto front with automatic VectorBT optimization.
        
        Args:
            solutions: List of solutions to evaluate
            objectives: Dictionary mapping metric names to optimization direction
            use_gpu: Whether to use GPU acceleration if available
            use_vectorbt: Whether to use VectorBT (None for auto-selection)
            
        Returns:
            List of non-dominated solutions (Pareto front)
        """
        if not solutions:
            return []
        
        start_time = time.time()
        self.performance_stats['total_operations'] += 1
        
        # Determine whether to use VectorBT
        if use_vectorbt is None:
            use_vectorbt = (self.enable_vectorbt and 
                          len(solutions) >= self.vectorbt_threshold)
        
        try:
            if use_vectorbt and self.vectorbt_optimizer:
                # Use VectorBT optimization
                pareto_front = self.vectorbt_optimizer.compute_pareto_front_vectorbt(solutions, objectives)
                self.performance_stats['vectorbt_operations'] += 1
                self.performance_stats['optimization_selections']['vectorbt'] = \
                    self.performance_stats['optimization_selections'].get('vectorbt', 0) + 1
            else:
                # Use standard implementation
                pareto_front = compute_pareto_front(solutions, objectives, use_gpu)
                self.performance_stats['standard_operations'] += 1
                self.performance_stats['optimization_selections']['standard'] = \
                    self.performance_stats['optimization_selections'].get('standard', 0) + 1
            
            # Update performance stats
            computation_time = time.time() - start_time
            self.performance_stats['total_time'] += computation_time
            
            self.logger.info(f"✅ Pareto front computed: {len(pareto_front)}/{len(solutions)} solutions "
                           f"in {computation_time:.3f}s using {'VectorBT' if use_vectorbt else 'standard'}")
            
            return pareto_front
            
        except Exception as e:
            self.logger.error(f"❌ Pareto front computation failed: {e}")
            # Fallback to standard implementation
            self.logger.warning("⚠️ Falling back to standard implementation")
            return compute_pareto_front(solutions, objectives, use_gpu)
    
    def compute_hypervolume(
        self,
        pareto_solutions: List[Solution],
        objectives: ObjectiveDirection,
        reference_point: Dict[str, float],
        use_vectorbt: Optional[bool] = None
    ) -> float:
        """Compute hypervolume with automatic VectorBT optimization."""
        if not pareto_solutions:
            return 0.0
        
        # Determine whether to use VectorBT
        if use_vectorbt is None:
            use_vectorbt = (self.enable_vectorbt and 
                          len(pareto_solutions) >= self.vectorbt_threshold)
        
        try:
            if use_vectorbt and self.vectorbt_optimizer:
                return self.vectorbt_optimizer.compute_hypervolume_vectorbt(
                    pareto_solutions, objectives, reference_point
                )
            else:
                return compute_hypervolume(pareto_solutions, objectives, reference_point)
        except Exception as e:
            self.logger.warning(f"Hypervolume computation failed: {e}, using fallback")
            return compute_hypervolume(pareto_solutions, objectives, reference_point)
    
    def select_knee_point(
        self,
        pareto_solutions: List[Solution],
        objectives: ObjectiveDirection,
        weights: Optional[Dict[str, float]] = None,
        use_vectorbt: Optional[bool] = None
    ) -> Optional[Solution]:
        """Select knee point with automatic VectorBT optimization."""
        if not pareto_solutions:
            return None
        
        # Determine whether to use VectorBT
        if use_vectorbt is None:
            use_vectorbt = (self.enable_vectorbt and 
                          len(pareto_solutions) >= self.vectorbt_threshold)
        
        try:
            if use_vectorbt and self.vectorbt_optimizer:
                return self.vectorbt_optimizer.select_knee_point_vectorbt(
                    pareto_solutions, objectives, weights
                )
            else:
                return select_knee_point(pareto_solutions, objectives, weights)
        except Exception as e:
            self.logger.warning(f"Knee point selection failed: {e}, using fallback")
            return select_knee_point(pareto_solutions, objectives, weights)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        
        if stats['total_operations'] > 0:
            stats['avg_time_per_operation'] = stats['total_time'] / stats['total_operations']
            stats['vectorbt_usage_rate'] = stats['vectorbt_operations'] / stats['total_operations']
            stats['standard_usage_rate'] = stats['standard_operations'] / stats['total_operations']
        
        return stats
    
    def cleanup(self):
        """Clean up resources."""
        if self.vectorbt_optimizer:
            self.vectorbt_optimizer.cleanup()
        
        self.logger.info("✅ Pareto VectorBT Integration cleaned up")


# Global integration instance
_global_integration = None

def get_pareto_vectorbt_integration() -> ParetoVectorBTIntegration:
    """Get global Pareto VectorBT integration instance."""
    global _global_integration
    if _global_integration is None:
        _global_integration = ParetoVectorBTIntegration()
    return _global_integration


# Enhanced wrapper functions that automatically use VectorBT optimizations
def compute_pareto_front_optimized(
    solutions: List[Solution],
    objectives: ObjectiveDirection,
    use_gpu: bool = True,
    use_vectorbt: Optional[bool] = None,
    vectorbt_threshold: int = 1000
) -> List[Solution]:
    """
    Compute Pareto front with automatic VectorBT optimization.
    
    This function automatically selects the best optimization strategy
    based on data size and available hardware.
    """
    integration = get_pareto_vectorbt_integration()
    integration.vectorbt_threshold = vectorbt_threshold
    return integration.compute_pareto_front(solutions, objectives, use_gpu, use_vectorbt)


def compute_hypervolume_optimized(
    pareto_solutions: List[Solution],
    objectives: ObjectiveDirection,
    reference_point: Dict[str, float],
    use_vectorbt: Optional[bool] = None,
    vectorbt_threshold: int = 1000
) -> float:
    """Compute hypervolume with automatic VectorBT optimization."""
    integration = get_pareto_vectorbt_integration()
    integration.vectorbt_threshold = vectorbt_threshold
    return integration.compute_hypervolume(pareto_solutions, objectives, reference_point, use_vectorbt)


def select_knee_point_optimized(
    pareto_solutions: List[Solution],
    objectives: ObjectiveDirection,
    weights: Optional[Dict[str, float]] = None,
    use_vectorbt: Optional[bool] = None,
    vectorbt_threshold: int = 1000
) -> Optional[Solution]:
    """Select knee point with automatic VectorBT optimization."""
    integration = get_pareto_vectorbt_integration()
    integration.vectorbt_threshold = vectorbt_threshold
    return integration.select_knee_point(pareto_solutions, objectives, weights, use_vectorbt)


# Monkey patching for seamless integration (optional)
def _patch_original_functions():
    """Patch original Pareto functions to use VectorBT optimizations."""
    if not VECTORBT_PARETO_AVAILABLE:
        return
    
    # Store original functions
    original_compute_pareto_front = compute_pareto_front
    original_compute_hypervolume = compute_hypervolume
    original_select_knee_point = select_knee_point
    
    # Create patched versions
    def patched_compute_pareto_front(solutions, objectives, use_gpu=True):
        return compute_pareto_front_optimized(solutions, objectives, use_gpu)
    
    def patched_compute_hypervolume(pareto_solutions, objectives, reference_point):
        return compute_hypervolume_optimized(pareto_solutions, objectives, reference_point)
    
    def patched_select_knee_point(pareto_solutions, objectives, weights=None):
        return select_knee_point_optimized(pareto_solutions, objectives, weights)
    
    # Apply patches
    import sys
    current_module = sys.modules[__name__]
    
    # Patch in the pareto module
    try:
        from . import pareto
        pareto.compute_pareto_front = patched_compute_pareto_front
        pareto.compute_hypervolume = patched_compute_hypervolume
        pareto.select_knee_point = patched_select_knee_point
        logger.info("✅ Original Pareto functions patched with VectorBT optimizations")
    except ImportError:
        logger.warning("⚠️ Could not patch original Pareto functions")


# Auto-patch if enabled
AUTO_PATCH_ENABLED = True
if AUTO_PATCH_ENABLED and VECTORBT_PARETO_AVAILABLE:
    _patch_original_functions()


# Performance monitoring utilities
def benchmark_pareto_optimizations(
    solutions: List[Solution],
    objectives: ObjectiveDirection,
    trials: int = 3
) -> Dict[str, Any]:
    """Benchmark different Pareto front optimization strategies."""
    if not solutions:
        return {}
    
    results = {}
    
    # Test standard implementation
    standard_times = []
    for trial in range(trials):
        start_time = time.time()
        pareto_front = compute_pareto_front(solutions, objectives, use_gpu=False)
        end_time = time.time()
        standard_times.append(end_time - start_time)
    
    results['standard'] = {
        'avg_time': np.mean(standard_times),
        'std_time': np.std(standard_times),
        'avg_pareto_size': len(pareto_front)
    }
    
    # Test VectorBT implementation if available
    if VECTORBT_PARETO_AVAILABLE:
        vectorbt_times = []
        for trial in range(trials):
            start_time = time.time()
            pareto_front = compute_pareto_front_vectorbt(solutions, objectives)
            end_time = time.time()
            vectorbt_times.append(end_time - start_time)
        
        results['vectorbt'] = {
            'avg_time': np.mean(vectorbt_times),
            'std_time': np.std(vectorbt_times),
            'avg_pareto_size': len(pareto_front)
        }
    
    # Test enhanced implementation if available
    if ENHANCED_PARETO_AVAILABLE:
        enhanced_times = []
        for trial in range(trials):
            start_time = time.time()
            pareto_front = compute_pareto_front_enhanced(solutions, objectives)
            end_time = time.time()
            enhanced_times.append(end_time - start_time)
        
        results['enhanced'] = {
            'avg_time': np.mean(enhanced_times),
            'std_time': np.std(enhanced_times),
            'avg_pareto_size': len(pareto_front)
        }
    
    return results


if __name__ == "__main__":
    # Example usage and testing
    print("🧪 Testing Pareto VectorBT Integration...")
    
    # Create sample solutions
    solutions = []
    for i in range(2000):
        solution = Solution(
            metrics={
                'pnl': np.random.randn() * 1000,
                'win_rate': np.random.random(),
                'sharpe': np.random.randn() * 2,
                'drawdown': abs(np.random.randn()) * 100
            },
            params={'param1': i, 'param2': i * 2}
        )
        solutions.append(solution)
    
    objectives = {
        'pnl': 'max',
        'win_rate': 'max',
        'sharpe': 'max',
        'drawdown': 'min'
    }
    
    # Test integration
    integration = ParetoVectorBTIntegration(enable_vectorbt=True, vectorbt_threshold=1000)
    
    # Compute Pareto front
    pareto_front = integration.compute_pareto_front(solutions, objectives)
    print(f"✅ Pareto front computed: {len(pareto_front)}/{len(solutions)} solutions")
    
    # Compute hypervolume
    reference_point = {'pnl': 0.0, 'win_rate': 0.0, 'sharpe': 0.0, 'drawdown': 1000.0}
    hypervolume = integration.compute_hypervolume(pareto_front, objectives, reference_point)
    print(f"✅ Hypervolume computed: {hypervolume:.4f}")
    
    # Select knee point
    knee_point = integration.select_knee_point(pareto_front, objectives)
    print(f"✅ Knee point selected: {knee_point.metrics if knee_point else None}")
    
    # Benchmark optimizations
    benchmark_results = benchmark_pareto_optimizations(solutions, objectives)
    print(f"📊 Benchmark results: {benchmark_results}")
    
    # Print performance stats
    stats = integration.get_performance_stats()
    print(f"📊 Performance stats: {stats}")
    
    print("🎉 Pareto VectorBT Integration test completed!")