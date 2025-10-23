"""
Computation Toolbox - Out-of-the-box Optimized Computations

This module provides a comprehensive toolbox for optimized computations
that automatically detects and utilizes available hardware for maximum performance.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import time

# Conditional imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Import hardware integration
try:
    from .hardware_integration import (
        get_hardware_optimized_processor,
        HardwareConfig,
        optimize_matrix_operation
    )
    HARDWARE_INTEGRATION_AVAILABLE = True
except ImportError:
    HARDWARE_INTEGRATION_AVAILABLE = False
    get_hardware_optimized_processor = None
    HardwareConfig = None
    optimize_matrix_operation = None

logger = logging.getLogger(__name__)

@dataclass
class ComputationConfig:
    """Configuration for computation toolbox."""
    # Performance settings
    enable_gpu: bool = True
    enable_parallel: bool = True
    max_memory_gb: float = 8.0

    # Optimization settings
    auto_optimize_dtypes: bool = True
    auto_chunk_large_data: bool = True
    chunk_size_threshold: int = 100000

    # Monitoring settings
    enable_performance_monitoring: bool = True
    log_performance_metrics: bool = True

    # Trading indicators settings
    default_indicator_config: Optional[Dict[str, Any]] = None

class ComputationToolbox:
    """
    Comprehensive computation toolbox that provides out-of-the-box
    optimized computations for various data processing tasks.
    """

    def __init__(self, config: Optional[ComputationConfig] = None):
        self.config = config or ComputationConfig()
        self.logger = logger.getChild('ComputationToolbox')

        # Initialize hardware processor
        self._initialize_hardware_processor()

        # Performance tracking
        self.performance_history = []

        self.logger.info("🛠️ Computation Toolbox initialized")

    def _initialize_hardware_processor(self):
        """Initialize hardware-optimized processor."""
        if HARDWARE_INTEGRATION_AVAILABLE and get_hardware_optimized_processor:
            try:
                hardware_config = HardwareConfig(
                    enable_gpu=self.config.enable_gpu,
                    max_memory_gb=self.config.max_memory_gb,
                    auto_optimize_dtypes=self.config.auto_optimize_dtypes,
                    auto_chunk_large_data=self.config.auto_chunk_large_data,
                    chunk_size_threshold=self.config.chunk_size_threshold,
                    enable_performance_monitoring=self.config.enable_performance_monitoring,
                    log_performance_metrics=self.config.log_performance_metrics
                )

                self.hardware_processor = get_hardware_optimized_processor(hardware_config)
                self.hardware_optimization_enabled = True
                self.logger.info("✅ Hardware optimization enabled")
            except Exception as e:
                self.logger.warning(f"⚠️ Hardware optimization initialization failed: {e}")
                self.hardware_processor = None
                self.hardware_optimization_enabled = False
        else:
            self.hardware_processor = None
            self.hardware_optimization_enabled = False
            self.logger.info("ℹ️ Hardware optimization not available")

    def compute_trading_indicators(self, data: 'pd.DataFrame',
                                 config: Optional[Dict[str, Any]] = None) -> 'pd.DataFrame':
        """
        Compute comprehensive trading indicators with hardware optimization.

        Args:
            data: DataFrame with OHLCV data
            config: Configuration for indicators

        Returns:
            DataFrame with computed indicators
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("Pandas is required for trading indicators computation")

        start_time = time.time()

        try:
            # Import vectorized core
            from .vectorized_core import get_vectorized_processing_core

            core = get_vectorized_processing_core()
            result = core.compute_trading_indicators(data, config)

            # Track performance
            execution_time = time.time() - start_time
            self._track_performance("trading_indicators", execution_time, len(data))

            return result

        except Exception as e:
            self.logger.error(f"❌ Trading indicators computation failed: {e}")
            raise

    def matrix_multiply(self, a: 'np.ndarray', b: 'np.ndarray',
                       use_gpu: bool = True) -> 'np.ndarray':
        """
        Optimized matrix multiplication with hardware acceleration.

        Args:
            a: First matrix
            b: Second matrix
            use_gpu: Whether to use GPU acceleration

        Returns:
            Result matrix
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for matrix operations")

        start_time = time.time()

        try:
            if self.hardware_optimization_enabled and use_gpu:
                # Use hardware-optimized matrix multiplication
                def multiply_func(x, y):
                    return np.dot(x, y)

                result = self.hardware_processor.process_with_hardware_optimization(
                    a, multiply_func, b
                )
            else:
                # Standard matrix multiplication
                result = np.dot(a, b)

            # Track performance
            execution_time = time.time() - start_time
            self._track_performance("matrix_multiply", execution_time, a.size)

            return result

        except Exception as e:
            self.logger.error(f"❌ Matrix multiplication failed: {e}")
            raise

    def correlation_analysis(self, data: Union['np.ndarray', 'pd.DataFrame'],
                           method: str = 'pearson') -> Tuple['np.ndarray', Optional['pd.DataFrame']]:
        """
        Optimized correlation analysis with hardware acceleration.

        Args:
            data: Input data
            method: Correlation method ('pearson', 'spearman', 'kendall')

        Returns:
            Tuple of (correlation_matrix, feature_importance)
        """
        start_time = time.time()

        try:
            if self.hardware_optimization_enabled:
                # Use hardware-optimized correlation analysis

                core = get_vectorized_processing_core()
                if isinstance(data, pd.DataFrame):
                    corr_matrix, feature_importance = core.matrix_correlation_analysis(data, method)
                else:
                    # Convert numpy array to DataFrame for processing
                    df = pd.DataFrame(data)
                    corr_matrix, feature_importance = core.matrix_correlation_analysis(df, method)
            else:
                # Standard correlation analysis
                if isinstance(data, pd.DataFrame):
                    corr_matrix = data.corr(method=method).values
                    feature_importance = None
                else:
                    corr_matrix = np.corrcoef(data.T)
                    feature_importance = None

            # Track performance
            execution_time = time.time() - start_time
            self._track_performance("correlation_analysis", execution_time, data.size if hasattr(data, 'size') else len(data))

            return corr_matrix, feature_importance

        except Exception as e:
            self.logger.error(f"❌ Correlation analysis failed: {e}")
            raise

    def batch_process(self, data: Union['np.ndarray', 'pd.DataFrame'],
                     operation_func: Callable,
                     batch_size: Optional[int] = None,
                     *args, **kwargs) -> Any:
        """
        Process data in optimized batches with hardware acceleration.

        Args:
            data: Input data
            operation_func: Function to apply to each batch
            batch_size: Size of each batch (auto-determined if None)
            *args, **kwargs: Additional arguments for operation_func

        Returns:
            Processed result
        """
        start_time = time.time()

        try:
            if self.hardware_optimization_enabled:
                # Use hardware-optimized batch processing
                result = self.hardware_processor.process_with_hardware_optimization(
                    data, operation_func, *args, **kwargs
                )
            else:
                # Standard batch processing
                if batch_size is None:
                    batch_size = self.config.chunk_size_threshold

                if isinstance(data, pd.DataFrame):
                    results = []
                    for i in range(0, len(data), batch_size):
                        batch = data.iloc[i:i + batch_size]
                        result = operation_func(batch, *args, **kwargs)
                        results.append(result)

                    if isinstance(results[0], pd.DataFrame):
                        result = pd.concat(results, ignore_index=True)
                    else:
                        result = results
                else:
                    results = []
                    for i in range(0, len(data), batch_size):
                        batch = data[i:i + batch_size]
                        result = operation_func(batch, *args, **kwargs)
                        results.append(result)

                    result = np.concatenate(results, axis=0)

            # Track performance
            execution_time = time.time() - start_time
            self._track_performance("batch_process", execution_time, data.size if hasattr(data, 'size') else len(data))

            return result

        except Exception as e:
            self.logger.error(f"❌ Batch processing failed: {e}")
            raise

    def optimize_dataframe(self, data: 'pd.DataFrame') -> 'pd.DataFrame':
        """
        Optimize DataFrame for processing with hardware acceleration.

        Args:
            data: Input DataFrame

        Returns:
            Optimized DataFrame
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("Pandas is required for DataFrame optimization")

        start_time = time.time()

        try:
            if self.hardware_optimization_enabled:
                # Use hardware-optimized DataFrame optimization
                optimized_data = self.hardware_processor.optimize_data_for_processing(data)
            else:
                # Standard optimization
                optimized_data = data.copy()

                # Optimize dtypes
                if self.config.auto_optimize_dtypes:
                    for col in optimized_data.select_dtypes(include=['int64']).columns:
                        if optimized_data[col].min() >= 0:
                            if optimized_data[col].max() < 255:
                                optimized_data[col] = optimized_data[col].astype('uint8')
                            elif optimized_data[col].max() < 65535:
                                optimized_data[col] = optimized_data[col].astype('uint16')
                            elif optimized_data[col].max() < 4294967295:
                                optimized_data[col] = optimized_data[col].astype('uint32')
                        else:
                            if optimized_data[col].min() > -128 and optimized_data[col].max() < 127:
                                optimized_data[col] = optimized_data[col].astype('int8')
                            elif optimized_data[col].min() > -32768 and optimized_data[col].max() < 32767:
                                optimized_data[col] = optimized_data[col].astype('int16')
                            elif optimized_data[col].min() > -2147483648 and optimized_data[col].max() < 2147483647:
                                optimized_data[col] = optimized_data[col].astype('int32')

                # Optimize float dtypes
                for col in optimized_data.select_dtypes(include=['float64']).columns:
                    optimized_data[col] = optimized_data[col].astype('float32')

            # Track performance
            execution_time = time.time() - start_time
            self._track_performance("optimize_dataframe", execution_time, len(data))

            return optimized_data

        except Exception as e:
            self.logger.error(f"❌ DataFrame optimization failed: {e}")
            raise

    def _track_performance(self, operation: str, execution_time: float, data_size: int):
        """Track performance metrics."""
        if self.config.enable_performance_monitoring:
            performance_record = {
                'operation': operation,
                'execution_time': execution_time,
                'data_size': data_size,
                'timestamp': time.time(),
                'hardware_optimization_enabled': self.hardware_optimization_enabled
            }

            self.performance_history.append(performance_record)

            if self.config.log_performance_metrics:
                self.logger.info(
                    f"📊 {operation}: {execution_time:.3f}s, "
                    f"Data size: {data_size:,}, "
                    f"Hardware optimization: {self.hardware_optimization_enabled}"
                )

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            'toolbox_config': {
                'enable_gpu': self.config.enable_gpu,
                'enable_parallel': self.config.enable_parallel,
                'max_memory_gb': self.config.max_memory_gb,
                'auto_optimize_dtypes': self.config.auto_optimize_dtypes,
                'auto_chunk_large_data': self.config.auto_chunk_large_data,
                'chunk_size_threshold': self.config.chunk_size_threshold
            },
            'hardware_optimization_enabled': self.hardware_optimization_enabled,
            'performance_history': self.performance_history.copy(),
            'summary': {}
        }

        # Calculate summary statistics
        if self.performance_history:
            total_operations = len(self.performance_history)
            total_time = sum(record['execution_time'] for record in self.performance_history)
            avg_time = total_time / total_operations

            # Group by operation
            operation_stats = {}
            for record in self.performance_history:
                op = record['operation']
                if op not in operation_stats:
                    operation_stats[op] = {
                        'count': 0,
                        'total_time': 0.0,
                        'avg_time': 0.0,
                        'data_sizes': []
                    }

                operation_stats[op]['count'] += 1
                operation_stats[op]['total_time'] += record['execution_time']
                operation_stats[op]['data_sizes'].append(record['data_size'])

            # Calculate averages
            for op in operation_stats:
                stats = operation_stats[op]
                stats['avg_time'] = stats['total_time'] / stats['count']
                stats['avg_data_size'] = sum(stats['data_sizes']) / len(stats['data_sizes'])
                del stats['data_sizes']  # Remove raw data sizes

            report['summary'] = {
                'total_operations': total_operations,
                'total_time': total_time,
                'average_time': avg_time,
                'operation_stats': operation_stats
            }

        # Add hardware performance report if available
        if self.hardware_optimization_enabled and self.hardware_processor:
            report['hardware_performance'] = self.hardware_processor.get_performance_report()

        return report

    def cleanup(self):
        """Cleanup resources."""
        if self.hardware_optimization_enabled and self.hardware_processor:
            self.hardware_processor.cleanup()

        self.logger.info("🧹 Computation Toolbox cleanup completed")

# Global instance
_computation_toolbox = None

def get_computation_toolbox(config: Optional[ComputationConfig] = None) -> ComputationToolbox:
    """Get or create the global computation toolbox."""
    global _computation_toolbox
    if _computation_toolbox is None:
        _computation_toolbox = ComputationToolbox(config)
    return _computation_toolbox

# Convenience functions for easy access
def compute_trading_indicators_optimized(data: 'pd.DataFrame',
                                       config: Optional[Dict[str, Any]] = None) -> 'pd.DataFrame':
    """Compute trading indicators with full hardware optimization."""
    toolbox = get_computation_toolbox()
    return toolbox.compute_trading_indicators(data, config)

def matrix_multiply_optimized(a: 'np.ndarray', b: 'np.ndarray',
                            use_gpu: bool = True) -> 'np.ndarray':
    """Optimized matrix multiplication with hardware acceleration."""
    toolbox = get_computation_toolbox()
    return toolbox.matrix_multiply(a, b, use_gpu)

def correlation_analysis_optimized(data: Union['np.ndarray', 'pd.DataFrame'],
                                 method: str = 'pearson') -> Tuple['np.ndarray', Optional['pd.DataFrame']]:
    """Optimized correlation analysis with hardware acceleration."""
    toolbox = get_computation_toolbox()
    return toolbox.correlation_analysis(data, method)

def batch_process_optimized(data: Union['np.ndarray', 'pd.DataFrame'],
                          operation_func: Callable,
                          batch_size: Optional[int] = None,
                          *args, **kwargs) -> Any:
    """Process data in optimized batches with hardware acceleration."""
    toolbox = get_computation_toolbox()
    return toolbox.batch_process(data, operation_func, batch_size, *args, **kwargs)

def optimize_dataframe_optimized(data: 'pd.DataFrame') -> 'pd.DataFrame':
    """Optimize DataFrame for processing with hardware acceleration."""
    toolbox = get_computation_toolbox()
    return toolbox.optimize_dataframe(data)

def get_toolbox_performance_report() -> Dict[str, Any]:
    """Get comprehensive performance report from computation toolbox."""
    toolbox = get_computation_toolbox()
    return toolbox.get_performance_report()

def cleanup_toolbox_resources():
    """Cleanup computation toolbox resources."""
    global _computation_toolbox
    if _computation_toolbox:
        _computation_toolbox.cleanup()
        _computation_toolbox = None
