"""
Performance Benchmarking and Monitoring for features_common.

This module provides comprehensive performance benchmarking and monitoring
capabilities for the features_common system with hardware optimization.
"""

import logging
import time
import json
import statistics
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Result of a performance benchmark."""
    operation_name: str
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: float
    data_size: int
    optimization_strategy: str
    hardware_optimized: bool
    success: bool
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceReport:
    """Comprehensive performance report."""
    benchmark_results: List[BenchmarkResult]
    summary_stats: Dict[str, Any]
    hardware_stats: Dict[str, Any]
    optimization_effectiveness: Dict[str, Any]
    recommendations: List[str]
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

class PerformanceBenchmark:
    """
    Performance benchmarking system for features_common operations.
    
    This class provides comprehensive benchmarking capabilities including
    execution time measurement, memory usage tracking, and optimization
    effectiveness analysis.
    """

    def __init__(self, output_dir: str = "benchmark_results"):
        """Initialize performance benchmark system."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Benchmark results storage
        self.benchmark_results: List[BenchmarkResult] = []
        self.benchmark_lock = threading.Lock()
        
        # Performance monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Benchmark configurations
        self.benchmark_configs = {
            'data_sizes': [100, 1000, 10000, 100000, 1000000],
            'operations': [
                'rolling_mean', 'rolling_std', 'rolling_var',
                'rolling_min', 'rolling_max', 'rolling_sum',
                'scale', 'rank', 'zscore', 'winsorize'
            ],
            'scaling_methods': ['zscore', 'minmax', 'robust', 'quantile'],
            'iterations': 5,  # Number of iterations per benchmark
            'warmup_iterations': 2  # Warmup iterations
        }
        
        logger.info("Performance benchmark system initialized")

    def benchmark_operation(self,
                          operation_func: Callable,
                          data: Union[pd.Series, pd.DataFrame],
                          operation_name: str,
                          optimization_strategy: str = 'standard',
                          hardware_optimized: bool = False,
                          iterations: int = 5) -> BenchmarkResult:
        """
        Benchmark a single operation.
        
        Args:
            operation_func: Function to benchmark
            data: Input data
            operation_name: Name of the operation
            optimization_strategy: Optimization strategy used
            hardware_optimized: Whether hardware optimization was used
            iterations: Number of iterations to run
            
        Returns:
            BenchmarkResult with performance metrics
        """
        logger.info(f"Benchmarking {operation_name} with {optimization_strategy} strategy")
        
        # Warmup iterations
        for _ in range(self.benchmark_configs['warmup_iterations']):
            try:
                operation_func(data)
            except Exception:
                pass
        
        # Benchmark iterations
        execution_times = []
        memory_usages = []
        cpu_usages = []
        gpu_usages = []
        success_count = 0
        
        for i in range(iterations):
            try:
                # Measure execution time
                start_time = time.time()
                result = operation_func(data)
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
                
                # Measure memory usage
                memory_usage = self._measure_memory_usage()
                memory_usages.append(memory_usage)
                
                # Measure CPU usage
                cpu_usage = self._measure_cpu_usage()
                cpu_usages.append(cpu_usage)
                
                # Measure GPU usage
                gpu_usage = self._measure_gpu_usage()
                gpu_usages.append(gpu_usage)
                
                success_count += 1
                
            except Exception as e:
                logger.warning(f"Benchmark iteration {i} failed: {e}")
                continue
        
        # Calculate statistics
        if execution_times:
            avg_execution_time = statistics.mean(execution_times)
            std_execution_time = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
        else:
            avg_execution_time = 0
            std_execution_time = 0
        
        avg_memory_usage = statistics.mean(memory_usages) if memory_usages else 0
        avg_cpu_usage = statistics.mean(cpu_usages) if cpu_usages else 0
        avg_gpu_usage = statistics.mean(gpu_usages) if gpu_usages else 0
        
        # Create benchmark result
        result = BenchmarkResult(
            operation_name=operation_name,
            execution_time=avg_execution_time,
            memory_usage_mb=avg_memory_usage,
            cpu_usage_percent=avg_cpu_usage,
            gpu_usage_percent=avg_gpu_usage,
            data_size=len(data) if hasattr(data, '__len__') else 0,
            optimization_strategy=optimization_strategy,
            hardware_optimized=hardware_optimized,
            success=success_count > 0,
            additional_metrics={
                'iterations': iterations,
                'success_count': success_count,
                'std_execution_time': std_execution_time,
                'min_execution_time': min(execution_times) if execution_times else 0,
                'max_execution_time': max(execution_times) if execution_times else 0
            }
        )
        
        # Store result
        with self.benchmark_lock:
            self.benchmark_results.append(result)
        
        logger.info(f"Benchmark completed: {operation_name} - {avg_execution_time:.4f}s")
        return result

    def benchmark_scaling_operations(self, data_sizes: Optional[List[int]] = None) -> List[BenchmarkResult]:
        """Benchmark scaling operations with different data sizes."""
        if data_sizes is None:
            data_sizes = self.benchmark_configs['data_sizes']
        
        results = []
        
        for size in data_sizes:
            logger.info(f"Benchmarking scaling operations with data size {size}")
            
            # Generate test data
            test_data = self._generate_test_data(size)
            
            # Test different scaling methods
            for method in self.benchmark_configs['scaling_methods']:
                try:
                    # Import scalers
                    from ..transforms.vectorbt_scaler import VectorBTScaler
                    from ..transforms.hardware_optimized_scaler import HardwareOptimizedVectorBTScaler
                    
                    # Test standard scaler
                    standard_scaler = VectorBTScaler(method=method)
                    result = self.benchmark_operation(
                        standard_scaler.fit_transform,
                        test_data,
                        f"scaling_{method}_standard",
                        optimization_strategy='standard',
                        hardware_optimized=False
                    )
                    results.append(result)
                    
                    # Test hardware-optimized scaler
                    hardware_scaler = HardwareOptimizedVectorBTScaler(method=method)
                    result = self.benchmark_operation(
                        hardware_scaler.fit_transform,
                        test_data,
                        f"scaling_{method}_hardware",
                        optimization_strategy='hardware_optimized',
                        hardware_optimized=True
                    )
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Failed to benchmark scaling method {method}: {e}")
                    continue
        
        return results

    def benchmark_vectorbt_operations(self, data_sizes: Optional[List[int]] = None) -> List[BenchmarkResult]:
        """Benchmark VectorBT operations with different data sizes."""
        if data_sizes is None:
            data_sizes = self.benchmark_configs['data_sizes']
        
        results = []
        
        for size in data_sizes:
            logger.info(f"Benchmarking VectorBT operations with data size {size}")
            
            # Generate test data
            test_data = self._generate_test_dataframe(size)
            
            # Test different operations
            for operation in self.benchmark_configs['operations']:
                try:
                    # Import managers
                    from ..vectorbt_extensions.unified_manager import get_unified_vectorbt_manager
                    from ..vectorbt_extensions.hardware_optimized_manager import get_hardware_optimized_vectorbt_manager
                    
                    # Test standard manager
                    standard_manager = get_unified_vectorbt_manager()
                    result = self.benchmark_operation(
                        lambda data: standard_manager.execute_operation(operation, data, window=20),
                        test_data,
                        f"vectorbt_{operation}_standard",
                        optimization_strategy='standard',
                        hardware_optimized=False
                    )
                    results.append(result)
                    
                    # Test hardware-optimized manager
                    hardware_manager = get_hardware_optimized_vectorbt_manager()
                    result = self.benchmark_operation(
                        lambda data: hardware_manager.execute_operation(operation, data, window=20),
                        test_data,
                        f"vectorbt_{operation}_hardware",
                        optimization_strategy='hardware_optimized',
                        hardware_optimized=True
                    )
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Failed to benchmark VectorBT operation {operation}: {e}")
                    continue
        
        return results

    def benchmark_memory_optimization(self, data_sizes: Optional[List[int]] = None) -> List[BenchmarkResult]:
        """Benchmark memory optimization effectiveness."""
        if data_sizes is None:
            data_sizes = self.benchmark_configs['data_sizes']
        
        results = []
        
        for size in data_sizes:
            logger.info(f"Benchmarking memory optimization with data size {size}")
            
            # Generate test data
            test_data = self._generate_test_dataframe(size)
            
            try:
                # Test without memory optimization
                def standard_operation(data):
                    return data.rolling(20).mean()
                
                result = self.benchmark_operation(
                    standard_operation,
                    test_data,
                    f"memory_standard_{size}",
                    optimization_strategy='standard',
                    hardware_optimized=False
                )
                results.append(result)
                
                # Test with memory optimization
                from ..optimization.hardware_optimized_mixin import HardwareOptimizedMixin
                
                class MemoryTestMixin(HardwareOptimizedMixin):
                    def __init__(self):
                        super().__init__()
                        self._memory_optimization_enabled = True
                
                mixin = MemoryTestMixin()
                
                def memory_optimized_operation(data):
                    return mixin.memory_efficient_operation(standard_operation, data)
                
                result = self.benchmark_operation(
                    memory_optimized_operation,
                    test_data,
                    f"memory_optimized_{size}",
                    optimization_strategy='memory_optimized',
                    hardware_optimized=True
                )
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Failed to benchmark memory optimization: {e}")
                continue
        
        return results

    def run_comprehensive_benchmark(self) -> PerformanceReport:
        """Run comprehensive benchmark suite."""
        logger.info("Starting comprehensive benchmark suite")
        
        # Run all benchmark categories
        scaling_results = self.benchmark_scaling_operations()
        vectorbt_results = self.benchmark_vectorbt_operations()
        memory_results = self.benchmark_memory_optimization()
        
        # Combine all results
        all_results = scaling_results + vectorbt_results + memory_results
        
        # Generate performance report
        report = self._generate_performance_report(all_results)
        
        # Save report
        self._save_performance_report(report)
        
        logger.info("Comprehensive benchmark suite completed")
        return report

    def _generate_test_data(self, size: int) -> pd.Series:
        """Generate test data for benchmarking."""
        np.random.seed(42)  # For reproducible results
        return pd.Series(np.random.randn(size), name='test_data')

    def _generate_test_dataframe(self, size: int, columns: int = 5) -> pd.DataFrame:
        """Generate test DataFrame for benchmarking."""
        np.random.seed(42)  # For reproducible results
        data = {}
        for i in range(columns):
            data[f'col_{i}'] = np.random.randn(size)
        return pd.DataFrame(data)

    def _measure_memory_usage(self) -> float:
        """Measure current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # Convert to MB
        except ImportError:
            return 0.0

    def _measure_cpu_usage(self) -> float:
        """Measure current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent()
        except ImportError:
            return 0.0

    def _measure_gpu_usage(self) -> float:
        """Measure current GPU usage percentage."""
        try:
            # This would need to be implemented based on the specific GPU monitoring library
            # For now, return 0
            return 0.0
        except ImportError:
            return 0.0

    def _generate_performance_report(self, results: List[BenchmarkResult]) -> PerformanceReport:
        """Generate comprehensive performance report."""
        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(results)
        
        # Calculate hardware statistics
        hardware_stats = self._calculate_hardware_stats(results)
        
        # Calculate optimization effectiveness
        optimization_effectiveness = self._calculate_optimization_effectiveness(results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results)
        
        return PerformanceReport(
            benchmark_results=results,
            summary_stats=summary_stats,
            hardware_stats=hardware_stats,
            optimization_effectiveness=optimization_effectiveness,
            recommendations=recommendations
        )

    def _calculate_summary_stats(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Calculate summary statistics from benchmark results."""
        if not results:
            return {}
        
        # Group by operation type
        operation_groups = {}
        for result in results:
            op_type = result.operation_name.split('_')[0]
            if op_type not in operation_groups:
                operation_groups[op_type] = []
            operation_groups[op_type].append(result)
        
        # Calculate statistics for each group
        group_stats = {}
        for op_type, group_results in operation_groups.items():
            execution_times = [r.execution_time for r in group_results if r.success]
            memory_usages = [r.memory_usage_mb for r in group_results if r.success]
            
            group_stats[op_type] = {
                'count': len(group_results),
                'success_rate': sum(1 for r in group_results if r.success) / len(group_results),
                'avg_execution_time': statistics.mean(execution_times) if execution_times else 0,
                'std_execution_time': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                'avg_memory_usage': statistics.mean(memory_usages) if memory_usages else 0,
                'min_execution_time': min(execution_times) if execution_times else 0,
                'max_execution_time': max(execution_times) if execution_times else 0
            }
        
        return {
            'total_benchmarks': len(results),
            'successful_benchmarks': sum(1 for r in results if r.success),
            'success_rate': sum(1 for r in results if r.success) / len(results),
            'operation_groups': group_stats,
            'overall_avg_execution_time': statistics.mean([r.execution_time for r in results if r.success]),
            'overall_avg_memory_usage': statistics.mean([r.memory_usage_mb for r in results if r.success])
        }

    def _calculate_hardware_stats(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Calculate hardware-specific statistics."""
        hardware_results = [r for r in results if r.hardware_optimized]
        standard_results = [r for r in results if not r.hardware_optimized]
        
        if not hardware_results or not standard_results:
            return {'message': 'Insufficient data for hardware comparison'}
        
        # Compare hardware vs standard performance
        hardware_times = [r.execution_time for r in hardware_results if r.success]
        standard_times = [r.execution_time for r in standard_results if r.success]
        
        if not hardware_times or not standard_times:
            return {'message': 'No successful benchmarks for comparison'}
        
        avg_hardware_time = statistics.mean(hardware_times)
        avg_standard_time = statistics.mean(standard_times)
        
        speedup = avg_standard_time / avg_hardware_time if avg_hardware_time > 0 else 0
        
        return {
            'hardware_benchmarks': len(hardware_results),
            'standard_benchmarks': len(standard_results),
            'avg_hardware_time': avg_hardware_time,
            'avg_standard_time': avg_standard_time,
            'speedup_factor': speedup,
            'performance_improvement_percent': (speedup - 1) * 100 if speedup > 0 else 0
        }

    def _calculate_optimization_effectiveness(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Calculate optimization effectiveness metrics."""
        # Group by optimization strategy
        strategy_groups = {}
        for result in results:
            strategy = result.optimization_strategy
            if strategy not in strategy_groups:
                strategy_groups[strategy] = []
            strategy_groups[strategy].append(result)
        
        # Calculate effectiveness for each strategy
        strategy_effectiveness = {}
        for strategy, group_results in strategy_groups.items():
            successful_results = [r for r in group_results if r.success]
            if not successful_results:
                continue
            
            execution_times = [r.execution_time for r in successful_results]
            memory_usages = [r.memory_usage_mb for r in successful_results]
            
            strategy_effectiveness[strategy] = {
                'count': len(group_results),
                'success_rate': len(successful_results) / len(group_results),
                'avg_execution_time': statistics.mean(execution_times),
                'avg_memory_usage': statistics.mean(memory_usages),
                'consistency': 1 - (statistics.stdev(execution_times) / statistics.mean(execution_times)) if execution_times else 0
            }
        
        return strategy_effectiveness

    def _generate_recommendations(self, results: List[BenchmarkResult]) -> List[str]:
        """Generate performance recommendations based on benchmark results."""
        recommendations = []
        
        # Analyze performance patterns
        hardware_results = [r for r in results if r.hardware_optimized and r.success]
        standard_results = [r for r in results if not r.hardware_optimized and r.success]
        
        if hardware_results and standard_results:
            avg_hardware_time = statistics.mean([r.execution_time for r in hardware_results])
            avg_standard_time = statistics.mean([r.execution_time for r in standard_results])
            
            if avg_hardware_time < avg_standard_time:
                speedup = avg_standard_time / avg_hardware_time
                recommendations.append(f"Hardware optimization provides {speedup:.2f}x speedup - consider enabling for all operations")
            else:
                recommendations.append("Hardware optimization may not be beneficial for current workload - consider disabling")
        
        # Analyze memory usage patterns
        high_memory_results = [r for r in results if r.memory_usage_mb > 100]
        if high_memory_results:
            recommendations.append("High memory usage detected - consider enabling memory optimization and chunking")
        
        # Analyze execution time patterns
        slow_results = [r for r in results if r.execution_time > 1.0]
        if slow_results:
            recommendations.append("Slow operations detected - consider using batch processing or GPU acceleration")
        
        # Analyze success rates
        low_success_ops = [r for r in results if not r.success]
        if low_success_ops:
            recommendations.append("Some operations are failing - check error handling and fallback mechanisms")
        
        return recommendations

    def _save_performance_report(self, report: PerformanceReport) -> None:
        """Save performance report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"performance_report_{timestamp}.json"
        
        # Convert report to dictionary for JSON serialization
        report_dict = {
            'benchmark_results': [
                {
                    'operation_name': r.operation_name,
                    'execution_time': r.execution_time,
                    'memory_usage_mb': r.memory_usage_mb,
                    'cpu_usage_percent': r.cpu_usage_percent,
                    'gpu_usage_percent': r.gpu_usage_percent,
                    'data_size': r.data_size,
                    'optimization_strategy': r.optimization_strategy,
                    'hardware_optimized': r.hardware_optimized,
                    'success': r.success,
                    'error_message': r.error_message,
                    'timestamp': r.timestamp,
                    'additional_metrics': r.additional_metrics
                }
                for r in report.benchmark_results
            ],
            'summary_stats': report.summary_stats,
            'hardware_stats': report.hardware_stats,
            'optimization_effectiveness': report.optimization_effectiveness,
            'recommendations': report.recommendations,
            'generated_at': report.generated_at
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        logger.info(f"Performance report saved to {report_file}")

    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmark results."""
        with self.benchmark_lock:
            if not self.benchmark_results:
                return {'message': 'No benchmark results available'}
            
            return self._calculate_summary_stats(self.benchmark_results)

    def clear_benchmark_results(self) -> None:
        """Clear all benchmark results."""
        with self.benchmark_lock:
            self.benchmark_results.clear()
        logger.info("Benchmark results cleared")

# Global benchmark instance
_global_benchmark: Optional[PerformanceBenchmark] = None

def get_performance_benchmark() -> PerformanceBenchmark:
    """Get the global performance benchmark instance."""
    global _global_benchmark
    if _global_benchmark is None:
        _global_benchmark = PerformanceBenchmark()
    return _global_benchmark

def run_quick_benchmark() -> PerformanceReport:
    """Run a quick benchmark with default settings."""
    benchmark = get_performance_benchmark()
    return benchmark.run_comprehensive_benchmark()