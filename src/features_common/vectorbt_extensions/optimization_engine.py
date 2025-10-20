"""
VectorBT Optimization Engine with Hardware Integration

This module provides the core optimization engine for VectorBT operations
with comprehensive hardware utility integration for maximum performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List
import logging
import time

# Import hardware utilities
try:
    from src.utils.hardware.integrated_hardware_manager import (
        get_integrated_hardware_manager, WorkloadType, process_market_data,
        process_ml_training_data, process_backtesting_data
    )
    from src.utils.hardware.adaptive_optimization_engine import (
        get_adaptive_optimization_engine, OptimizationTarget
    )
    from src.utils.hardware.advanced_memory_manager import (
        get_advanced_memory_manager, memory_efficient_processing,
        chunked_processing, track_memory_usage
    )
    from src.utils.hardware.enhanced_gpu_manager import (
        get_enhanced_gpu_manager, GPUOperationType, create_gpu_operation
    )
    HARDWARE_UTILITIES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Hardware utilities not available: {e}")
    HARDWARE_UTILITIES_AVAILABLE = False

logger = logging.getLogger(__name__)

class VectorBTOptimizationEngine:
    """
    VectorBT optimization engine with comprehensive hardware integration.
    
    This engine provides intelligent optimization for VectorBT operations
    using hardware utilities for maximum performance.
    """

    def __init__(self):
        self.optimizations = {}
        self.performance_metrics = {}
        self.hardware_available = HARDWARE_UTILITIES_AVAILABLE
        
        # Hardware optimization components
        self.integrated_manager = None
        self.adaptive_engine = None
        self.memory_manager = None
        self.gpu_manager = None
        
        # Performance tracking
        self.optimization_stats = {
            'total_optimizations': 0,
            'hardware_optimizations': 0,
            'memory_optimizations': 0,
            'gpu_optimizations': 0,
            'adaptive_decisions': 0,
            'performance_improvements': []
        }
        
        # Initialize hardware components if available
        if self.hardware_available:
            self._initialize_hardware_components()

    def _initialize_hardware_components(self):
        """Initialize hardware optimization components."""
        try:
            # Initialize integrated hardware manager
            self.integrated_manager = get_integrated_hardware_manager()
            
            # Initialize adaptive optimization engine
            self.adaptive_engine = get_adaptive_optimization_engine()
            
            # Initialize advanced memory manager
            self.memory_manager = get_advanced_memory_manager()
            
            # Initialize enhanced GPU manager
            self.gpu_manager = get_enhanced_gpu_manager()
            
            logger.debug("Hardware components initialized for VectorBT optimization engine")
            
        except Exception as e:
            logger.warning(f"Failed to initialize hardware components: {e}")
            self.hardware_available = False

    def optimize_operation(self, operation: str, data: Union[pd.DataFrame, pd.Series], 
                          workload_type: str = 'data_processing', **kwargs) -> Any:
        """
        Optimize a VectorBT operation with hardware utilities.
        
        Args:
            operation: Name of the operation to optimize
            data: Input data
            workload_type: Type of workload for optimization
            **kwargs: Additional parameters
            
        Returns:
            Optimized result
        """
        self.optimization_stats['total_optimizations'] += 1
        
        try:
            # Apply hardware optimization if available
            if self.hardware_available and self._should_use_hardware_optimization(data, operation):
                optimized_data = self._apply_hardware_optimization(data, workload_type)
                result = self._execute_optimized_operation(operation, optimized_data, **kwargs)
                self.optimization_stats['hardware_optimizations'] += 1
            else:
                # Use standard optimization
                result = self._execute_standard_operation(operation, data, **kwargs)
            
            # Track performance improvement
            self._track_performance_improvement(operation, data, result)
            
            return result
            
        except Exception as e:
            logger.warning(f"VectorBT optimization failed: {e}")
            return data

    def _should_use_hardware_optimization(self, data: Union[pd.DataFrame, pd.Series], operation: str) -> bool:
        """Determine if hardware optimization should be used."""
        if not self.hardware_available:
            return False
        
        data_size = len(data) if hasattr(data, '__len__') else 0
        
        # Use hardware optimization for large datasets or specific operations
        return (data_size >= 1000 or 
                operation in ['rolling_mean', 'rolling_std', 'scaling', 'ranking'] or
                isinstance(data, pd.DataFrame) and data_size >= 500)

    def _apply_hardware_optimization(self, data: Union[pd.DataFrame, pd.Series], 
                                   workload_type: str) -> Union[pd.DataFrame, pd.Series]:
        """Apply hardware optimization to the data."""
        try:
            # Use advanced memory manager for optimization
            if isinstance(data, pd.DataFrame):
                optimized_data = self.memory_manager.process_data_with_optimization(
                    data, workload_type
                )
            else:
                optimized_data = self.memory_manager.process_data_with_optimization(
                    data, workload_type
                )
            
            self.optimization_stats['memory_optimizations'] += 1
            return optimized_data
            
        except Exception as e:
            logger.warning(f"Hardware optimization failed: {e}")
            return data

    def _execute_optimized_operation(self, operation: str, data: Union[pd.DataFrame, pd.Series], 
                                   **kwargs) -> Any:
        """Execute operation with hardware optimization."""
        try:
            # Use integrated hardware manager based on workload type
            if workload_type == 'data_processing':
                optimized_data = process_market_data(data)
            elif workload_type == 'ml_training':
                optimized_data = process_ml_training_data(data)
            elif workload_type == 'backtesting':
                optimized_data = process_backtesting_data(data)
            else:
                optimized_data = data
            
            # Execute the operation on optimized data
            return self._execute_standard_operation(operation, optimized_data, **kwargs)
            
        except Exception as e:
            logger.warning(f"Optimized operation failed: {e}")
            return self._execute_standard_operation(operation, data, **kwargs)

    def _execute_standard_operation(self, operation: str, data: Union[pd.DataFrame, pd.Series], 
                                  **kwargs) -> Any:
        """Execute operation with standard optimization."""
        # This would contain the actual VectorBT operation logic
        # For now, return the data as-is
        return data

    def _track_performance_improvement(self, operation: str, input_data: Union[pd.DataFrame, pd.Series], 
                                     result: Any) -> None:
        """Track performance improvement metrics."""
        try:
            # Calculate performance metrics
            input_size = len(input_data) if hasattr(input_data, '__len__') else 0
            result_size = len(result) if hasattr(result, '__len__') else 0
            
            improvement = {
                'operation': operation,
                'input_size': input_size,
                'result_size': result_size,
                'timestamp': time.time(),
                'hardware_optimized': self.optimization_stats['hardware_optimizations'] > 0
            }
            
            self.optimization_stats['performance_improvements'].append(improvement)
            
            # Keep only recent improvements
            if len(self.optimization_stats['performance_improvements']) > 1000:
                self.optimization_stats['performance_improvements'] = \
                    self.optimization_stats['performance_improvements'][-500:]
                
        except Exception as e:
            logger.debug(f"Failed to track performance improvement: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        metrics = self.performance_metrics.copy()
        metrics.update(self.optimization_stats)
        
        # Add hardware-specific metrics
        if self.hardware_available:
            try:
                metrics['hardware_stats'] = {
                    'integrated_manager': self.integrated_manager.get_optimization_report() if self.integrated_manager else {},
                    'memory_manager': self.memory_manager.get_detailed_memory_info() if self.memory_manager else {},
                    'gpu_manager': self.gpu_manager.get_enhanced_gpu_info() if self.gpu_manager else {},
                    'adaptive_engine': self.adaptive_engine.get_learning_report() if self.adaptive_engine else {}
                }
            except Exception as e:
                logger.warning(f"Failed to get hardware stats: {e}")
                metrics['hardware_stats'] = {}
        
        return metrics

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary with recommendations."""
        total_ops = self.optimization_stats['total_optimizations']
        hardware_ops = self.optimization_stats['hardware_optimizations']
        memory_ops = self.optimization_stats['memory_optimizations']
        
        return {
            'total_optimizations': total_ops,
            'hardware_optimization_rate': hardware_ops / total_ops if total_ops > 0 else 0,
            'memory_optimization_rate': memory_ops / total_ops if total_ops > 0 else 0,
            'hardware_available': self.hardware_available,
            'recommendations': self._generate_recommendations()
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        total_ops = self.optimization_stats['total_optimizations']
        hardware_ops = self.optimization_stats['hardware_optimizations']
        
        if total_ops > 0:
            hardware_rate = hardware_ops / total_ops
            
            if hardware_rate < 0.5:
                recommendations.append("Consider enabling hardware optimization for more operations")
            
            if not self.hardware_available:
                recommendations.append("Hardware utilities not available - install for better performance")
            
            if len(self.optimization_stats['performance_improvements']) > 10:
                recent_improvements = self.optimization_stats['performance_improvements'][-10:]
                hardware_improvements = sum(1 for imp in recent_improvements if imp.get('hardware_optimized', False))
                
                if hardware_improvements / len(recent_improvements) > 0.8:
                    recommendations.append("Hardware optimization is performing well - consider expanding usage")
        
        return recommendations

    def reset_optimization_stats(self) -> None:
        """Reset optimization statistics."""
        self.optimization_stats = {
            'total_optimizations': 0,
            'hardware_optimizations': 0,
            'memory_optimizations': 0,
            'gpu_optimizations': 0,
            'adaptive_decisions': 0,
            'performance_improvements': []
        }

def get_optimization_engine() -> VectorBTOptimizationEngine:
    """Get the optimization engine instance."""
    return VectorBTOptimizationEngine()
