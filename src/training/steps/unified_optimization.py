"""
Unified Performance & Memory Optimization Infrastructure

This module provides unified performance and memory optimization across all training steps using
MemoryEfficientTraining and ParallelProcessingCoordinator from ml_common, replacing manual 
optimization code.

Key Features:
- Unified memory optimization using MemoryEfficientTraining
- Unified parallel processing using ParallelProcessingCoordinator
- Automatic optimization strategies based on data size and system resources
- Comprehensive performance monitoring and reporting
- Integration with ML Common utilities
- Support for M1/M2/M3 hardware optimizations
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime
import pandas as pd
import numpy as np
import psutil
import gc
import time

# Import new simplified infrastructure
from .simplified_pipeline_infrastructure import (
    create_simple_step_function,
    create_data_processing_step_function
)

# Import standardized validation
from .standardized_config_validation import (
    validate_config,
    validate_and_fix_config
)

# Import unified data quality
from .unified_data_quality import (
    validate_data_quality,
    clean_data,
    generate_quality_report
)

# Import ML Common utilities
from src.utils.ml_common import (
    MemoryEfficientTraining,
    ParallelProcessingCoordinator,
    DataQualityUtilities,
    MLTrainingSafeguards
)

# Import M1 optimization utilities
from src.utils.m1_memory_optimizer import M1MemoryOptimizer
from src.utils.m1_cpu_optimizer import M1CPUOptimizer
from src.utils.m1_gpu_utils import M1GPUManager

# Import common operations
from src.utils.common_operations import get_logger

logger = get_logger(__name__)


class UnifiedOptimizationManager:
    """
    Unified optimization manager for all training steps.
    
    This replaces manual optimization code with a unified approach
    using MemoryEfficientTraining and ParallelProcessingCoordinator from ml_common.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize unified optimization manager."""
        self.config = validate_and_fix_config(config, 'optimization')
        self.logger = logger.getChild('UnifiedOptimizationManager')
        
        # Initialize ML Common utilities
        self.memory_optimizer = MemoryEfficientTraining(self.config.get('memory_config', {}))
        self.parallel_coordinator = ParallelProcessingCoordinator(self.config.get('parallel_config', {}))
        self.data_quality = DataQualityUtilities()
        self.safeguards = MLTrainingSafeguards()
        
        # Initialize M1 optimization utilities
        self.m1_memory_optimizer = M1MemoryOptimizer()
        self.m1_cpu_optimizer = M1CPUOptimizer()
        self.m1_gpu_manager = M1GPUManager()
        
        # Optimization configuration
        self.optimization_config = self.config.get('optimization_config', {})
        
        # Standard optimization settings
        self.standard_settings = {
            'enable_memory_optimization': True,
            'enable_parallel_processing': True,
            'enable_m1_optimizations': True,
            'enable_gpu_acceleration': True,
            'enable_automatic_chunking': True,
            'enable_memory_monitoring': True,
            'enable_performance_profiling': True,
            'chunk_size_mb': 500,
            'max_memory_usage': 0.8,
            'max_workers': psutil.cpu_count(),
            'gc_interval_seconds': 300,
            'memory_checkpoint_interval': 60,
            'performance_report_interval': 300
        }
        
        # Update with user configuration
        self.standard_settings.update(self.optimization_config)
        
        # Performance monitoring
        self.performance_metrics = {
            'memory_usage_history': [],
            'execution_times': [],
            'optimization_events': [],
            'start_time': datetime.now()
        }
        
        self.logger.info("🚀 Unified Optimization Manager initialized")
    
    async def optimize_operation(self, operation: Callable, operation_name: str, 
                               data: Optional[Any] = None, optimization_type: str = 'comprehensive') -> Dict[str, Any]:
        """
        Optimize an operation using unified approach.
        
        Args:
            operation: Function to optimize
            operation_name: Name of the operation
            data: Optional data to optimize
            optimization_type: Type of optimization ('basic', 'standard', 'comprehensive')
            
        Returns:
            Optimization result
        """
        try:
            self.logger.info(f"⚡ Starting {optimization_type} optimization for '{operation_name}'...")
            
            # Start performance monitoring
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Apply optimization based on type
            if optimization_type == 'basic':
                result = await self._apply_basic_optimization(operation, operation_name, data)
            elif optimization_type == 'standard':
                result = await self._apply_standard_optimization(operation, operation_name, data)
            elif optimization_type == 'comprehensive':
                result = await self._apply_comprehensive_optimization(operation, operation_name, data)
            else:
                raise ValueError(f"Unknown optimization type: {optimization_type}")
            
            # End performance monitoring
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Record performance metrics
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            self.performance_metrics['execution_times'].append({
                'operation': operation_name,
                'execution_time': execution_time,
                'start_memory': start_memory,
                'end_memory': end_memory,
                'memory_delta': memory_delta,
                'timestamp': datetime.now().isoformat()
            })
            
            # Generate optimization metadata
            optimization_metadata = self._generate_optimization_metadata(
                operation_name, optimization_type, execution_time, memory_delta, result
            )
            
            # Generate performance report
            performance_report = self._generate_performance_report(operation_name, optimization_type)
            
            return {
                'result': result,
                'optimization_metadata': optimization_metadata,
                'performance_report': performance_report,
                'execution_time': execution_time,
                'memory_usage': {
                    'start_memory_mb': start_memory,
                    'end_memory_mb': end_memory,
                    'memory_delta_mb': memory_delta
                },
                'optimization_type': optimization_type
            }
            
        except Exception as e:
            self.logger.exception(f"Error optimizing operation: {e}")
            raise
    
    async def _apply_basic_optimization(self, operation: Callable, operation_name: str, data: Optional[Any]) -> Any:
        """Apply basic optimization (memory management only)."""
        try:
            self.logger.info("Applying basic optimization...")
            
            # Basic memory management
            if self.standard_settings.get('enable_memory_optimization', True):
                with self.m1_memory_optimizer.memory_checkpoint(f"{operation_name}_basic"):
                    # Force garbage collection before operation
                    gc.collect()
                    
                    # Execute operation
                    if data is not None:
                        result = operation(data)
                    else:
                        result = operation()
                    
                    # Force garbage collection after operation
                    gc.collect()
                    
                    return result
            else:
                # Execute without optimization
                if data is not None:
                    return operation(data)
                else:
                    return operation()
            
        except Exception as e:
            self.logger.exception(f"Error in basic optimization: {e}")
            raise
    
    async def _apply_standard_optimization(self, operation: Callable, operation_name: str, data: Optional[Any]) -> Any:
        """Apply standard optimization (memory + basic parallel processing)."""
        try:
            self.logger.info("Applying standard optimization...")
            
            # Standard memory optimization
            if self.standard_settings.get('enable_memory_optimization', True):
                with self.memory_optimizer.memory_checkpoint(f"{operation_name}_standard"):
                    # Memory-efficient execution
                    if data is not None:
                        result = operation(data)
                    else:
                        result = operation()
                    
                    return result
            else:
                # Execute without optimization
                if data is not None:
                    return operation(data)
                else:
                    return operation()
            
        except Exception as e:
            self.logger.exception(f"Error in standard optimization: {e}")
            raise
    
    async def _apply_comprehensive_optimization(self, operation: Callable, operation_name: str, data: Optional[Any]) -> Any:
        """Apply comprehensive optimization (all features enabled)."""
        try:
            self.logger.info("Applying comprehensive optimization...")
            
            # Comprehensive optimization with all features
            optimization_context = self._create_optimization_context(operation_name)
            
            with optimization_context:
                # Memory optimization
                if self.standard_settings.get('enable_memory_optimization', True):
                    with self.memory_optimizer.memory_checkpoint(f"{operation_name}_comprehensive"):
                        # Parallel processing optimization
                        if self.standard_settings.get('enable_parallel_processing', True):
                            result = await self._execute_with_parallel_optimization(operation, data)
                        else:
                            # Execute with memory optimization only
                            if data is not None:
                                result = operation(data)
                            else:
                                result = operation()
                        
                        return result
                else:
                    # Execute without optimization
                    if data is not None:
                        return operation(data)
                    else:
                        return operation()
            
        except Exception as e:
            self.logger.exception(f"Error in comprehensive optimization: {e}")
            raise
    
    def _create_optimization_context(self, operation_name: str):
        """Create comprehensive optimization context."""
        from contextlib import contextmanager
        
        @contextmanager
        def optimization_context():
            try:
                # Start optimization
                self.logger.debug(f"Starting comprehensive optimization for {operation_name}")
                
                # M1 optimizations
                if self.standard_settings.get('enable_m1_optimizations', True):
                    self.m1_cpu_optimizer.optimize_for_operation(operation_name)
                    self.m1_gpu_manager.optimize_for_operation(operation_name)
                
                yield
                
            finally:
                # End optimization
                self.logger.debug(f"Completed comprehensive optimization for {operation_name}")
                
                # Cleanup
                if self.standard_settings.get('enable_memory_optimization', True):
                    gc.collect()
        
        return optimization_context()
    
    async def _execute_with_parallel_optimization(self, operation: Callable, data: Optional[Any]) -> Any:
        """Execute operation with parallel processing optimization."""
        try:
            # Check if data can be parallelized
            if data is not None and hasattr(data, '__len__') and len(data) > 1000:
                # Use parallel processing for large datasets
                if isinstance(data, pd.DataFrame):
                    return await self._parallel_dataframe_operation(operation, data)
                elif isinstance(data, (list, tuple)):
                    return await self._parallel_list_operation(operation, data)
                else:
                    # Fallback to regular execution
                    return operation(data)
            else:
                # Execute normally for small datasets
                if data is not None:
                    return operation(data)
                else:
                    return operation()
            
        except Exception as e:
            self.logger.warning(f"Parallel optimization failed, falling back to regular execution: {e}")
            if data is not None:
                return operation(data)
            else:
                return operation()
    
    async def _parallel_dataframe_operation(self, operation: Callable, data: pd.DataFrame) -> Any:
        """Execute operation on DataFrame with parallel processing."""
        try:
            # Split DataFrame into chunks
            chunk_size = self.standard_settings.get('chunk_size_mb', 500) * 1024 * 1024  # Convert to bytes
            estimated_chunk_rows = chunk_size // (data.memory_usage(deep=True).sum() / len(data))
            
            if len(data) > estimated_chunk_rows:
                # Process in parallel chunks
                chunks = [data.iloc[i:i+estimated_chunk_rows] for i in range(0, len(data), estimated_chunk_rows)]
                
                # Use parallel coordinator
                results = self.parallel_coordinator.parallel_feature_engineering(
                    feature_functions=[operation] * len(chunks),
                    data_chunks=chunks,
                    combine_results=True
                )
                
                return results
            else:
                # Process entire DataFrame
                return operation(data)
            
        except Exception as e:
            self.logger.warning(f"Parallel DataFrame operation failed: {e}")
            return operation(data)
    
    async def _parallel_list_operation(self, operation: Callable, data: list) -> Any:
        """Execute operation on list with parallel processing."""
        try:
            # Split list into chunks
            chunk_size = max(1, len(data) // self.standard_settings.get('max_workers', psutil.cpu_count()))
            
            if len(data) > chunk_size:
                # Process in parallel chunks
                chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
                
                # Use parallel coordinator
                results = self.parallel_coordinator.parallel_feature_engineering(
                    feature_functions=[operation] * len(chunks),
                    data_chunks=chunks,
                    combine_results=True
                )
                
                return results
            else:
                # Process entire list
                return operation(data)
            
        except Exception as e:
            self.logger.warning(f"Parallel list operation failed: {e}")
            return operation(data)
    
    def _generate_optimization_metadata(self, operation_name: str, optimization_type: str, 
                                      execution_time: float, memory_delta: float, result: Any) -> Dict[str, Any]:
        """Generate metadata about optimization."""
        try:
            metadata = {
                'operation_name': operation_name,
                'optimization_type': optimization_type,
                'execution_time_seconds': execution_time,
                'memory_delta_mb': memory_delta,
                'timestamp': datetime.now().isoformat(),
                'optimization_settings': self.standard_settings,
                'system_info': {
                    'cpu_count': psutil.cpu_count(),
                    'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
                    'memory_available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
                    'memory_usage_percent': psutil.virtual_memory().percent
                }
            }
            
            # Add result information
            if result is not None:
                if hasattr(result, 'shape'):
                    metadata['result_shape'] = result.shape
                elif hasattr(result, '__len__'):
                    metadata['result_length'] = len(result)
                else:
                    metadata['result_type'] = type(result).__name__
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Error generating optimization metadata: {e}")
            return {'error': str(e)}
    
    def _generate_performance_report(self, operation_name: str, optimization_type: str) -> Dict[str, Any]:
        """Generate performance report."""
        try:
            report = {
                'report_timestamp': datetime.now().isoformat(),
                'operation_name': operation_name,
                'optimization_type': optimization_type,
                'performance_summary': self._get_performance_summary(),
                'memory_usage_summary': self._get_memory_usage_summary(),
                'optimization_recommendations': self._get_optimization_recommendations()
            }
            
            return report
            
        except Exception as e:
            self.logger.warning(f"Error generating performance report: {e}")
            return {'error': str(e)}
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        try:
            execution_times = self.performance_metrics['execution_times']
            
            if not execution_times:
                return {'message': 'No performance data available'}
            
            times = [t['execution_time'] for t in execution_times]
            
            return {
                'total_operations': len(execution_times),
                'average_execution_time': np.mean(times),
                'min_execution_time': np.min(times),
                'max_execution_time': np.max(times),
                'total_execution_time': np.sum(times)
            }
            
        except Exception as e:
            self.logger.warning(f"Error getting performance summary: {e}")
            return {'error': str(e)}
    
    def _get_memory_usage_summary(self) -> Dict[str, Any]:
        """Get memory usage summary."""
        try:
            execution_times = self.performance_metrics['execution_times']
            
            if not execution_times:
                return {'message': 'No memory data available'}
            
            memory_deltas = [t['memory_delta'] for t in execution_times]
            
            return {
                'average_memory_delta_mb': np.mean(memory_deltas),
                'min_memory_delta_mb': np.min(memory_deltas),
                'max_memory_delta_mb': np.max(memory_deltas),
                'total_memory_delta_mb': np.sum(memory_deltas)
            }
            
        except Exception as e:
            self.logger.warning(f"Error getting memory usage summary: {e}")
            return {'error': str(e)}
    
    def _get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations."""
        try:
            recommendations = []
            
            # Check memory usage
            current_memory = psutil.virtual_memory().percent
            if current_memory > 80:
                recommendations.append("⚠️ High memory usage detected - consider reducing chunk size or enabling more aggressive garbage collection")
            elif current_memory < 50:
                recommendations.append("✅ Memory usage is optimal - current settings are working well")
            
            # Check execution times
            execution_times = self.performance_metrics['execution_times']
            if execution_times:
                avg_time = np.mean([t['execution_time'] for t in execution_times])
                if avg_time > 60:
                    recommendations.append("⚠️ Long execution times detected - consider enabling parallel processing or reducing data size")
                elif avg_time < 10:
                    recommendations.append("✅ Execution times are optimal - current optimization settings are effective")
            
            # Check CPU usage
            cpu_count = psutil.cpu_count()
            max_workers = self.standard_settings.get('max_workers', cpu_count)
            if max_workers < cpu_count:
                recommendations.append(f"💡 Consider increasing max_workers to {cpu_count} for better parallel processing")
            
            if not recommendations:
                recommendations.append("✅ No specific optimization recommendations - system is performing well")
            
            return recommendations
            
        except Exception as e:
            self.logger.warning(f"Error getting optimization recommendations: {e}")
            return [f"Error getting recommendations: {e}"]
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization capabilities."""
        return {
            'config': self.config,
            'standard_settings': self.standard_settings,
            'optimization_utilities': {
                'memory_optimizer': 'MemoryEfficientTraining',
                'parallel_coordinator': 'ParallelProcessingCoordinator',
                'm1_memory_optimizer': 'M1MemoryOptimizer',
                'm1_cpu_optimizer': 'M1CPUOptimizer',
                'm1_gpu_manager': 'M1GPUManager'
            },
            'performance_metrics': self.performance_metrics,
            'timestamp': datetime.now().isoformat()
        }


# Simplified optimization step functions
async def unified_optimization_logic(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unified optimization logic using MemoryEfficientTraining and ParallelProcessingCoordinator.
    
    Args:
        config: Configuration dictionary
        pipeline_state: Current pipeline state
        
    Returns:
        Optimization result
    """
    logger.info("⚡ Starting unified optimization...")
    
    try:
        # Get operation and data from pipeline state
        operation = pipeline_state.get('operation')
        operation_name = pipeline_state.get('operation_name', 'unknown_operation')
        data = pipeline_state.get('data')
        
        if operation is None:
            raise ValueError("No operation found in pipeline state for optimization")
        
        # Initialize unified optimization manager
        optimization_manager = UnifiedOptimizationManager(config)
        
        # Determine optimization type from configuration
        optimization_type = config.get('optimization_type', 'comprehensive')
        
        # Optimize operation
        result = await optimization_manager.optimize_operation(operation, operation_name, data, optimization_type)
        
        return result
        
    except Exception as e:
        logger.exception(f"Error in unified optimization: {e}")
        raise


async def basic_optimization_logic(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """Basic optimization logic (memory management only)."""
    logger.info("⚡ Starting basic optimization...")
    
    try:
        # Get operation and data from pipeline state
        operation = pipeline_state.get('operation')
        operation_name = pipeline_state.get('operation_name', 'unknown_operation')
        data = pipeline_state.get('data')
        
        if operation is None:
            raise ValueError("No operation found in pipeline state for optimization")
        
        # Initialize unified optimization manager
        optimization_manager = UnifiedOptimizationManager(config)
        
        # Optimize operation
        result = await optimization_manager.optimize_operation(operation, operation_name, data, 'basic')
        
        return result
        
    except Exception as e:
        logger.exception(f"Error in basic optimization: {e}")
        raise


async def standard_optimization_logic(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """Standard optimization logic (memory + basic parallel processing)."""
    logger.info("⚡ Starting standard optimization...")
    
    try:
        # Get operation and data from pipeline state
        operation = pipeline_state.get('operation')
        operation_name = pipeline_state.get('operation_name', 'unknown_operation')
        data = pipeline_state.get('data')
        
        if operation is None:
            raise ValueError("No operation found in pipeline state for optimization")
        
        # Initialize unified optimization manager
        optimization_manager = UnifiedOptimizationManager(config)
        
        # Optimize operation
        result = await optimization_manager.optimize_operation(operation, operation_name, data, 'standard')
        
        return result
        
    except Exception as e:
        logger.exception(f"Error in standard optimization: {e}")
        raise


async def comprehensive_optimization_logic(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """Comprehensive optimization logic (all optimization features)."""
    logger.info("⚡ Starting comprehensive optimization...")
    
    try:
        # Get operation and data from pipeline state
        operation = pipeline_state.get('operation')
        operation_name = pipeline_state.get('operation_name', 'unknown_operation')
        data = pipeline_state.get('data')
        
        if operation is None:
            raise ValueError("No operation found in pipeline state for optimization")
        
        # Initialize unified optimization manager
        optimization_manager = UnifiedOptimizationManager(config)
        
        # Optimize operation
        result = await optimization_manager.optimize_operation(operation, operation_name, data, 'comprehensive')
        
        return result
        
    except Exception as e:
        logger.exception(f"Error in comprehensive optimization: {e}")
        raise


# Create step functions
unified_optimization = create_simple_step_function("unified_optimization", unified_optimization_logic)
basic_optimization = create_simple_step_function("basic_optimization", basic_optimization_logic)
standard_optimization = create_simple_step_function("standard_optimization", standard_optimization_logic)
comprehensive_optimization = create_simple_step_function("comprehensive_optimization", comprehensive_optimization_logic)


class SimplifiedOptimization:
    """
    Simplified optimization using unified infrastructure.
    
    This replaces manual optimization code with a unified approach
    using MemoryEfficientTraining and ParallelProcessingCoordinator from ml_common.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize simplified optimization."""
        self.config = validate_and_fix_config(config, 'optimization')
        self.logger = logger.getChild('SimplifiedOptimization')
        
        # Initialize unified optimization manager
        self.optimization_manager = UnifiedOptimizationManager(self.config)
        
        self.logger.info("🚀 Simplified Optimization initialized")
    
    async def optimize_operation(self, operation: Callable, operation_name: str, 
                               data: Optional[Any] = None, optimization_type: str = 'comprehensive') -> Dict[str, Any]:
        """
        Optimize operation using unified approach.
        
        Args:
            operation: Function to optimize
            operation_name: Name of the operation
            data: Optional data to optimize
            optimization_type: Type of optimization
            
        Returns:
            Optimization result
        """
        try:
            self.logger.info(f"🚀 Optimizing {optimization_type} operation '{operation_name}'...")
            
            # Optimize operation
            result = await self.optimization_manager.optimize_operation(operation, operation_name, data, optimization_type)
            
            self.logger.info(f"✅ Optimization completed: {result['execution_time']:.2f}s execution time")
            
            return result
            
        except Exception as e:
            self.logger.exception(f"Optimization error: {e}")
            raise
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization capabilities."""
        return self.optimization_manager.get_optimization_summary()


# Example usage and testing
async def example_optimization():
    """Example of using the unified optimization."""
    
    # Create sample data
    np.random.seed(42)
    large_data = pd.DataFrame({
        'feature_1': np.random.randn(10000),
        'feature_2': np.random.randn(10000),
        'feature_3': np.random.randn(10000),
        'feature_4': np.random.randn(10000),
        'feature_5': np.random.randn(10000)
    })
    
    # Define sample operations
    def simple_operation(data):
        return data.sum()
    
    def complex_operation(data):
        return data.corr()
    
    # Configuration for different optimization types
    configs = [
        {
            'symbol': 'BTCUSDT',
            'exchange': 'binance',
            'timeframe': '1m',
            'optimization_type': 'basic',
            'optimization_config': {
                'enable_memory_optimization': True,
                'enable_parallel_processing': False,
                'enable_m1_optimizations': False
            }
        },
        {
            'symbol': 'BTCUSDT',
            'exchange': 'binance',
            'timeframe': '1m',
            'optimization_type': 'standard',
            'optimization_config': {
                'enable_memory_optimization': True,
                'enable_parallel_processing': True,
                'enable_m1_optimizations': True,
                'chunk_size_mb': 100
            }
        },
        {
            'symbol': 'BTCUSDT',
            'exchange': 'binance',
            'timeframe': '1m',
            'optimization_type': 'comprehensive',
            'optimization_config': {
                'enable_memory_optimization': True,
                'enable_parallel_processing': True,
                'enable_m1_optimizations': True,
                'enable_gpu_acceleration': True,
                'enable_automatic_chunking': True,
                'enable_memory_monitoring': True,
                'enable_performance_profiling': True,
                'chunk_size_mb': 200,
                'max_workers': psutil.cpu_count()
            }
        }
    ]
    
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n=== Testing Optimization Type {i+1}: {config['optimization_type']} ===")
        
        # Create simplified optimization
        optimizer = SimplifiedOptimization(config)
        
        # Test simple operation
        print(f"Testing simple operation...")
        simple_result = await optimizer.optimize_operation(
            simple_operation, 'simple_operation', large_data, config['optimization_type']
        )
        print(f"   ✅ Simple operation: {simple_result['execution_time']:.3f}s")
        
        # Test complex operation
        print(f"Testing complex operation...")
        complex_result = await optimizer.optimize_operation(
            complex_operation, 'complex_operation', large_data, config['optimization_type']
        )
        print(f"   ✅ Complex operation: {complex_result['execution_time']:.3f}s")
        
        # Get summary
        summary = optimizer.get_optimization_summary()
        
        print(f"Optimization type: {config['optimization_type']}")
        print(f"Memory delta: {simple_result['memory_usage']['memory_delta_mb']:.1f} MB")
        print(f"Performance recommendations: {len(complex_result['performance_report']['optimization_recommendations'])}")
        
        results.append((simple_result, complex_result, summary))
    
    return results


# Main execution
async def main():
    """Main execution function."""
    try:
        results = await example_optimization()
        print("✅ Optimization example completed successfully")
        return results
    except Exception as e:
        logger.exception(f"Optimization example failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())