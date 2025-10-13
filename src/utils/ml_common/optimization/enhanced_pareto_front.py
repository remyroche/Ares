"""
Enhanced Pareto Front Utilities with VectorBT Integration

This module provides an enhanced interface that combines the original Pareto front
utilities with VectorBT optimizations, offering automatic optimization selection
and seamless integration with existing code.

Key features:
- Automatic optimization selection based on data size and available hardware
- Seamless integration with existing Pareto front utilities
- VectorBT acceleration for large datasets
- Fallback to original implementation for compatibility
- Performance monitoring and statistics
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import time
from typing import Any, Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import logging

# Import original Pareto utilities
from .pareto import (
    Solution, ObjectiveDirection, ParetoFront, compute_pareto_front,
    select_knee_point, compute_hypervolume, scalarize_financial_goals,
    filter_by_constraints, DEFAULT_FINANCIAL_WEIGHTS
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
    VectorBTParetoOptimizer = None
    VectorBTParetoConfig = None

# Import unified vectorization manager
try:
    from ..unified_vectorization_manager import UnifiedVectorizationManager, OperationType
    UNIFIED_MANAGER_AVAILABLE = True
except ImportError:
    UNIFIED_MANAGER_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class EnhancedParetoConfig:
    """Configuration for enhanced Pareto front computation."""
    # Optimization selection
    auto_select_optimization: bool = True
    prefer_vectorbt: bool = True
    vectorbt_threshold: int = 1000  # Use VectorBT for datasets >= this size
    
    # VectorBT configuration
    vectorbt_config: Optional[VectorBTParetoConfig] = None
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    enable_caching: bool = True
    
    # Fallback behavior
    fallback_on_error: bool = True
    log_performance: bool = True


class EnhancedParetoFront:
    """
    Enhanced Pareto front utilities with automatic optimization selection.
    
    This class provides a unified interface that automatically selects the best
    optimization strategy based on data size, available hardware, and performance
    requirements.
    """
    
    def __init__(self, config: Optional[EnhancedParetoConfig] = None):
        """Initialize enhanced Pareto front utilities."""
        self.config = config or EnhancedParetoConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
        
        # Performance tracking
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'standard_operations': 0,
            'fallback_operations': 0,
            'total_time': 0.0,
            'optimization_selections': {},
            'error_count': 0
        }
        
        self.logger.info("🚀 Enhanced Pareto Front initialized")
    
    def _initialize_components(self):
        """Initialize optimization components."""
        # Initialize VectorBT optimizer if available
        if VECTORBT_PARETO_AVAILABLE and self.config.prefer_vectorbt:
            try:
                self.vectorbt_optimizer = get_vectorbt_pareto_optimizer(self.config.vectorbt_config)
                self.logger.info("✅ VectorBT Pareto optimizer initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize VectorBT optimizer: {e}")
                self.vectorbt_optimizer = None
        else:
            self.vectorbt_optimizer = None
        
        # Initialize unified vectorization manager
        if UNIFIED_MANAGER_AVAILABLE:
            try:
                self.vectorization_manager = UnifiedVectorizationManager()
                self.logger.info("✅ Unified vectorization manager initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize vectorization manager: {e}")
                self.vectorization_manager = None
        else:
            self.vectorization_manager = None
    
    def compute_pareto_front(
        self,
        solutions: List[Solution],
        objectives: ObjectiveDirection,
        use_gpu: bool = True,
        use_nonlinear_transforms: bool = True,
        force_optimization: Optional[str] = None
    ) -> List[Solution]:
        """
        Compute Pareto front with automatic optimization selection.
        
        Args:
            solutions: List of solutions to evaluate
            objectives: Dictionary mapping metric names to optimization direction
            use_gpu: Whether to use GPU acceleration if available
            use_nonlinear_transforms: Whether to use non-linear transformations
            force_optimization: Force specific optimization ('vectorbt', 'standard', 'gpu')
            
        Returns:
            List of non-dominated solutions (Pareto front)
        """
        if not solutions:
            return []
        
        start_time = time.time()
        self.performance_stats['total_operations'] += 1
        
        try:
            # Select optimization strategy
            if force_optimization:
                strategy = force_optimization
            elif self.config.auto_select_optimization:
                strategy = self._select_optimization_strategy(solutions, objectives, use_gpu)
            else:
                strategy = 'standard'
            
            # Execute with selected strategy
            pareto_front = self._execute_pareto_computation(
                strategy, solutions, objectives, use_gpu, use_nonlinear_transforms
            )
            
            # Update performance stats
            computation_time = time.time() - start_time
            self.performance_stats['total_time'] += computation_time
            self.performance_stats['optimization_selections'][strategy] = \
                self.performance_stats['optimization_selections'].get(strategy, 0) + 1
            
            if self.config.log_performance:
                self.logger.info(f"✅ Pareto front computed using {strategy}: "
                               f"{len(pareto_front)}/{len(solutions)} solutions in {computation_time:.3f}s")
            
            return pareto_front
            
        except Exception as e:
            self.performance_stats['error_count'] += 1
            self.logger.error(f"❌ Pareto front computation failed: {e}")
            
            if self.config.fallback_on_error:
                self.logger.warning("⚠️ Falling back to standard implementation")
                return self._fallback_computation(solutions, objectives, use_gpu)
            else:
                raise
    
    def _select_optimization_strategy(
        self,
        solutions: List[Solution],
        objectives: ObjectiveDirection,
        use_gpu: bool
    ) -> str:
        """Select optimal optimization strategy based on data characteristics."""
        n_solutions = len(solutions)
        n_objectives = len(objectives)
        
        # VectorBT strategy selection
        if (self.vectorbt_optimizer and 
            n_solutions >= self.config.vectorbt_threshold and
            self.config.prefer_vectorbt):
            return 'vectorbt'
        
        # GPU strategy selection
        if use_gpu and n_solutions > 500:
            return 'gpu'
        
        # Standard strategy for small datasets
        return 'standard'
    
    def _execute_pareto_computation(
        self,
        strategy: str,
        solutions: List[Solution],
        objectives: ObjectiveDirection,
        use_gpu: bool,
        use_nonlinear_transforms: bool
    ) -> List[Solution]:
        """Execute Pareto front computation with selected strategy."""
        if strategy == 'vectorbt' and self.vectorbt_optimizer:
            self.performance_stats['vectorbt_operations'] += 1
            return self.vectorbt_optimizer.compute_pareto_front_vectorbt(solutions, objectives)
        
        elif strategy == 'gpu':
            # Use GPU-accelerated computation
            self.performance_stats['standard_operations'] += 1
            return self._compute_pareto_front_gpu(solutions, objectives, use_nonlinear_transforms)
        
        else:
            # Use standard computation
            self.performance_stats['standard_operations'] += 1
            return compute_pareto_front(solutions, objectives, use_gpu)
    
    def _compute_pareto_front_gpu(
        self,
        solutions: List[Solution],
        objectives: ObjectiveDirection,
        use_nonlinear_transforms: bool
    ) -> List[Solution]:
        """Compute Pareto front with GPU acceleration."""
        # Use the original ParetoFront class with GPU support
        pareto_front = ParetoFront()
        return pareto_front.compute_pareto_front_gpu(
            solutions, objectives, use_gpu=True, use_nonlinear_transforms=use_nonlinear_transforms
        )
    
    def _fallback_computation(
        self,
        solutions: List[Solution],
        objectives: ObjectiveDirection,
        use_gpu: bool
    ) -> List[Solution]:
        """Fallback computation using standard implementation."""
        self.performance_stats['fallback_operations'] += 1
        return compute_pareto_front(solutions, objectives, use_gpu)
    
    def compute_hypervolume(
        self,
        pareto_solutions: List[Solution],
        objectives: ObjectiveDirection,
        reference_point: Dict[str, float],
        use_vectorbt: bool = True
    ) -> float:
        """Compute hypervolume with automatic optimization selection."""
        if not pareto_solutions:
            return 0.0
        
        try:
            if (use_vectorbt and self.vectorbt_optimizer and 
                len(pareto_solutions) >= self.config.vectorbt_threshold):
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
        use_vectorbt: bool = True
    ) -> Optional[Solution]:
        """Select knee point with automatic optimization selection."""
        if not pareto_solutions:
            return None
        
        try:
            if (use_vectorbt and self.vectorbt_optimizer and 
                len(pareto_solutions) >= self.config.vectorbt_threshold):
                return self.vectorbt_optimizer.select_knee_point_vectorbt(
                    pareto_solutions, objectives, weights
                )
            else:
                return select_knee_point(pareto_solutions, objectives, weights)
        except Exception as e:
            self.logger.warning(f"Knee point selection failed: {e}, using fallback")
            return select_knee_point(pareto_solutions, objectives, weights)
    
    def scalarize_financial_goals(
        self,
        metrics: Dict[str, float],
        weights: Optional[Dict[str, float]] = None,
        fallback_objectives: Optional[ObjectiveDirection] = None,
        use_nonlinear_scaling: bool = True
    ) -> float:
        """Scalarize financial goals with enhanced optimization."""
        return scalarize_financial_goals(
            metrics, weights, fallback_objectives, use_nonlinear_scaling
        )
    
    def filter_by_constraints(
        self,
        solutions: List[Solution],
        constraints: Dict[str, Any]
    ) -> List[Solution]:
        """Filter solutions by constraints with enhanced performance."""
        return filter_by_constraints(solutions, constraints)
    
    def benchmark_optimizations(
        self,
        solutions: List[Solution],
        objectives: ObjectiveDirection,
        trials: int = 3
    ) -> Dict[str, Any]:
        """Benchmark different optimization strategies."""
        if not solutions:
            return {}
        
        results = {}
        strategies = ['standard']
        
        if self.vectorbt_optimizer:
            strategies.append('vectorbt')
        
        if VECTORBT_PARETO_AVAILABLE:
            strategies.append('gpu')
        
        for strategy in strategies:
            strategy_times = []
            strategy_sizes = []
            
            for trial in range(trials):
                try:
                    start_time = time.time()
                    pareto_front = self._execute_pareto_computation(
                        strategy, solutions, objectives, use_gpu=True, use_nonlinear_transforms=True
                    )
                    end_time = time.time()
                    
                    strategy_times.append(end_time - start_time)
                    strategy_sizes.append(len(pareto_front))
                    
                except Exception as e:
                    self.logger.warning(f"Strategy {strategy} failed in trial {trial}: {e}")
                    continue
            
            if strategy_times:
                results[strategy] = {
                    'avg_time': np.mean(strategy_times),
                    'std_time': np.std(strategy_times),
                    'avg_pareto_size': np.mean(strategy_sizes),
                    'trials_completed': len(strategy_times)
                }
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        
        if stats['total_operations'] > 0:
            stats['avg_time_per_operation'] = stats['total_time'] / stats['total_operations']
            stats['vectorbt_usage_rate'] = stats['vectorbt_operations'] / stats['total_operations']
            stats['standard_usage_rate'] = stats['standard_operations'] / stats['total_operations']
            stats['fallback_usage_rate'] = stats['fallback_operations'] / stats['total_operations']
            stats['error_rate'] = stats['error_count'] / stats['total_operations']
        
        # Add VectorBT stats if available
        if self.vectorbt_optimizer:
            vectorbt_stats = self.vectorbt_optimizer.get_performance_stats()
            stats['vectorbt_details'] = vectorbt_stats
        
        return stats
    
    def cleanup(self):
        """Clean up resources."""
        if self.vectorbt_optimizer:
            self.vectorbt_optimizer.cleanup()
        
        if self.vectorization_manager:
            # Cleanup if available
            pass
        
        self.logger.info("✅ Enhanced Pareto Front cleaned up")


# Convenience functions for backward compatibility
def compute_pareto_front_enhanced(
    solutions: List[Solution],
    objectives: ObjectiveDirection,
    use_gpu: bool = True,
    config: Optional[EnhancedParetoConfig] = None
) -> List[Solution]:
    """Enhanced Pareto front computation with automatic optimization."""
    enhanced_pareto = EnhancedParetoFront(config)
    return enhanced_pareto.compute_pareto_front(solutions, objectives, use_gpu)


def compute_hypervolume_enhanced(
    pareto_solutions: List[Solution],
    objectives: ObjectiveDirection,
    reference_point: Dict[str, float],
    use_vectorbt: bool = True,
    config: Optional[EnhancedParetoConfig] = None
) -> float:
    """Enhanced hypervolume computation with automatic optimization."""
    enhanced_pareto = EnhancedParetoFront(config)
    return enhanced_pareto.compute_hypervolume(pareto_solutions, objectives, reference_point, use_vectorbt)


def select_knee_point_enhanced(
    pareto_solutions: List[Solution],
    objectives: ObjectiveDirection,
    weights: Optional[Dict[str, float]] = None,
    use_vectorbt: bool = True,
    config: Optional[EnhancedParetoConfig] = None
) -> Optional[Solution]:
    """Enhanced knee point selection with automatic optimization."""
    enhanced_pareto = EnhancedParetoFront(config)
    return enhanced_pareto.select_knee_point(pareto_solutions, objectives, weights, use_vectorbt)


# Global enhanced Pareto front instance
_global_enhanced_pareto = None

def get_enhanced_pareto_front(config: Optional[EnhancedParetoConfig] = None) -> EnhancedParetoFront:
    """Get global enhanced Pareto front instance."""
    global _global_enhanced_pareto
    if _global_enhanced_pareto is None:
        _global_enhanced_pareto = EnhancedParetoFront(config)
    return _global_enhanced_pareto


if __name__ == "__main__":
    # Example usage and testing
    print("🧪 Testing Enhanced Pareto Front...")
    
    # Create sample solutions
    solutions = []
    for i in range(1000):
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
    
    # Test enhanced Pareto front
    config = EnhancedParetoConfig(
        auto_select_optimization=True,
        prefer_vectorbt=True,
        vectorbt_threshold=500
    )
    
    enhanced_pareto = EnhancedParetoFront(config)
    
    # Compute Pareto front
    pareto_front = enhanced_pareto.compute_pareto_front(solutions, objectives)
    print(f"✅ Pareto front computed: {len(pareto_front)}/{len(solutions)} solutions")
    
    # Compute hypervolume
    reference_point = {'pnl': 0.0, 'win_rate': 0.0, 'sharpe': 0.0, 'drawdown': 1000.0}
    hypervolume = enhanced_pareto.compute_hypervolume(pareto_front, objectives, reference_point)
    print(f"✅ Hypervolume computed: {hypervolume:.4f}")
    
    # Select knee point
    knee_point = enhanced_pareto.select_knee_point(pareto_front, objectives)
    print(f"✅ Knee point selected: {knee_point.metrics if knee_point else None}")
    
    # Benchmark optimizations
    benchmark_results = enhanced_pareto.benchmark_optimizations(solutions, objectives)
    print(f"📊 Benchmark results: {benchmark_results}")
    
    # Print performance stats
    stats = enhanced_pareto.get_performance_stats()
    print(f"📊 Performance stats: {stats}")
    
    print("🎉 Enhanced Pareto Front test completed!")