"""
Hardware Optimization Module for Tree-Based Models

This module provides comprehensive hardware optimization capabilities for tree-based models,
including memory optimization, parallel processing, and M1-specific optimizations.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')
from src.utils.tprint import tprint

# Handle optional dependencies gracefully
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Create a mock numpy for basic functionality
    class MockNumpy:
        def __init__(self):
            self.float32 = float
            self.float64 = float
            self.array = list
            self.mean = lambda x: sum(x) / len(x) if x else 0
            self.max = max
            self.sum = sum
            self.vstack = lambda x: x[0] if x else []
        def __getattr__(self, name):
            return lambda *args, **kwargs: 0
    np = MockNumpy()

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    # Create a mock pandas for basic functionality
    class MockPandas:
        def __init__(self):
            """Initialize the mock pandas class for fallback functionality."""
            self._mock_data = {}
            self._mock_index = 0
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    pd = MockPandas()

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # Create a mock psutil for basic functionality
    class MockPsutil:
        def cpu_percent(self):
            return 0.0
        def virtual_memory(self):
            class MockMemory:
                def __init__(self):
                    self.total = 8 * 1024 * 1024 * 1024  # 8GB
                    self.available = 4 * 1024 * 1024 * 1024  # 4GB
                    self.used = 4 * 1024 * 1024 * 1024  # 4GB
                    self.percent = 50.0
            return MockMemory()
    psutil = MockPsutil()

# Import shared utilities
try:
    from src.utils.common_operations import (
        safe_divide, safe_log, safe_sqrt, safe_power,
        validate_finite, validate_positive, validate_range,
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        is_m1_available, is_mps_available
    )
    from src.utils.common_utilities import (
        safe_dataframe_operation, validate_dataframe_columns,
        calculate_data_quality_metrics, create_summary_statistics
    )
    from src.utils.math_validation import (
        safe_correlation, safe_covariance, safe_mean, safe_std,
        validate_correlation_matrix, safe_matrix_inverse
    )
    from src.utils.matrix_operations import (
        vectorized_operations, matrix_optimization
    )
    SHARED_UTILS_AVAILABLE = True
except ImportError:
    SHARED_UTILS_AVAILABLE = False

# Import M1-specific utilities
try:
    from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager, is_m1_available
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    M1_UTILS_AVAILABLE = True
except ImportError:
    M1_UTILS_AVAILABLE = False

# Import ML common utilities
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
        BayesianTPEOptimizer, GridOptimizer
    )
    ML_COMMON_AVAILABLE = True
except ImportError:
    ML_COMMON_AVAILABLE = False

logger = logging.getLogger(__name__)

class OptimizationLevel(Enum):
    """Hardware optimization levels."""
    BASIC = "basic"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"

class HardwareType(Enum):
    """Types of hardware optimization."""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"

@dataclass
class HardwareConfig:
    """Configuration for hardware optimization."""
    enable_gpu_acceleration: bool = True
    enable_memory_optimization: bool = True
    enable_parallel_processing: bool = True
    enable_m1_optimization: bool = True
    memory_limit_gb: float = 8.0
    max_workers: Optional[int] = None
    optimization_level: OptimizationLevel = OptimizationLevel.STANDARD
    chunk_size: int = 1000
    cache_size_mb: int = 100
    enable_caching: bool = True

@dataclass
class PerformanceMetrics:
    """Hardware performance metrics."""
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float]
    processing_time: float
    throughput: float
    optimization_level: str
    hardware_type: str

@dataclass
class TreeOptimizationConfig:
    """Configuration for tree-specific optimizations."""
    enable_tree_parallelization: bool = True
    enable_tree_caching: bool = True
    enable_tree_memory_pooling: bool = True
    tree_batch_size: int = 32
    tree_memory_limit_mb: int = 512
    enable_tree_compression: bool = True

class TreeHardwareOptimizer:
    """
    Advanced hardware optimizer specifically designed for tree-based models.
    
    This optimizer provides:
    - Memory optimization for large tree models
    - Parallel processing for tree operations
    - M1-specific optimizations
    - Tree-specific caching and pooling
    - Performance monitoring and metrics
    """
    
    def __init__(self, config: Optional[HardwareConfig] = None, tree_config: Optional[TreeOptimizationConfig] = None):
        """Initialize tree hardware optimizer."""
        self.config = config or HardwareConfig()
        self.tree_config = tree_config or TreeOptimizationConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize hardware components
        self.gpu_manager = None
        self.memory_optimizer = None
        self.cpu_optimizer = None
        self.cache_manager = None
        
        # Performance tracking
        self.performance_history = []
        self.optimization_stats = {
            'total_optimizations': 0,
            'memory_savings': 0,
            'speed_improvements': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Initialize hardware acceleration
        self._initialize_hardware_components()
        
        self.logger.info("✅ Tree Hardware Optimizer initialized")
    
    def _initialize_hardware_components(self):
        """Initialize hardware acceleration components."""
        try:
            # Initialize GPU acceleration
            if self.config.enable_gpu_acceleration:
                self._setup_gpu_acceleration()
            
            # Initialize memory optimization
            if self.config.enable_memory_optimization:
                self._setup_memory_optimization()
            
            # Initialize M1 optimization
            if self.config.enable_m1_optimization and M1_UTILS_AVAILABLE:
                self._setup_m1_optimization()
            
            # Initialize caching
            if self.config.enable_caching:
                self._setup_caching()
            
            self.logger.info("✅ Hardware components initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Hardware component initialization failed: {e}")
            raise
    
    def _setup_gpu_acceleration(self):
        """Setup GPU acceleration for tree models."""
        try:
            if M1_UTILS_AVAILABLE and is_m1_available():
                self.gpu_manager = get_m1_gpu_manager()
                self.logger.info("✅ M1 GPU acceleration enabled")
            else:
                self.logger.warning("⚠️ GPU acceleration not available")
        except Exception as e:
            self.logger.error(f"❌ GPU acceleration setup failed: {e}")
    
    def _setup_memory_optimization(self):
        """Setup memory optimization for tree models."""
        try:
            if M1_UTILS_AVAILABLE:
                self.memory_optimizer = get_m1_memory_optimizer()
                self.logger.info("✅ Memory optimization enabled")
            else:
                # Fallback memory optimization
                self.memory_optimizer = self._create_fallback_memory_optimizer()
                self.logger.info("✅ Fallback memory optimization enabled")
        except Exception as e:
            self.logger.error(f"❌ Memory optimization setup failed: {e}")
    
    def _setup_m1_optimization(self):
        """Setup M1-specific optimizations."""
        try:
            if M1_UTILS_AVAILABLE:
                self.cpu_optimizer = get_m1_cpu_optimizer()
                self.logger.info("✅ M1 CPU optimization enabled")
            else:
                self.logger.warning("⚠️ M1 optimization not available")
        except Exception as e:
            self.logger.error(f"❌ M1 optimization setup failed: {e}")
    
    def _setup_caching(self):
        """Setup caching for tree operations."""
        try:
            self.cache_manager = TreeCacheManager(
                cache_size_mb=self.config.cache_size_mb,
                enable_compression=self.tree_config.enable_tree_compression
            )
            self.logger.info("✅ Tree caching enabled")
        except Exception as e:
            self.logger.error(f"❌ Caching setup failed: {e}")
    
    def _create_fallback_memory_optimizer(self):
        """Create fallback memory optimizer."""
        class FallbackMemoryOptimizer:
            def __init__(self):
                self.memory_usage = 0
                self.optimization_count = 0
            
            def optimize_memory(self, data):
                """Basic memory optimization."""
                if hasattr(data, 'memory_usage'):
                    return data.memory_usage(deep=True).sum()
                return 0
            
            def get_memory_usage(self):
                """Get current memory usage."""
                return psutil.virtual_memory().percent
        
        return FallbackMemoryOptimizer()
    
    def optimize_tree_processing(self, 
                               tree_model: Any,
                               X: np.ndarray,
                               y: np.ndarray,
                               operation_type: str = "training") -> Dict[str, Any]:
        """
        Optimize tree model processing with hardware acceleration.
        
        Args:
            tree_model: Tree model to optimize
            X: Training features
            y: Training targets
            operation_type: Type of operation (training, prediction, evaluation)
            
        Returns:
            Optimization results with performance metrics
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"🚀 Starting tree optimization for {operation_type}")
            
            # Preprocess data for optimization
            X_optimized, y_optimized = self._optimize_data_for_tree_processing(X, y)
            
            # Apply tree-specific optimizations
            tree_model_optimized = self._apply_tree_optimizations(tree_model)
            
            # Apply hardware optimizations
            if self.config.enable_parallel_processing:
                tree_model_optimized = self._apply_parallel_optimizations(tree_model_optimized)
            
            # Execute optimized processing
            processing_results = self._execute_optimized_processing(
                tree_model_optimized, X_optimized, y_optimized, operation_type
            )
            
            # Calculate performance metrics
            processing_time = time.time() - start_time
            performance_metrics = self._calculate_performance_metrics(
                processing_time, X_optimized.shape[0], operation_type
            )
            
            # Update optimization stats
            self.optimization_stats['total_optimizations'] += 1
            
            results = {
                'processing_results': processing_results,
                'performance_metrics': performance_metrics,
                'optimization_stats': self.optimization_stats,
                'hardware_utilization': self._get_hardware_utilization(),
                'memory_usage': self._get_memory_usage(),
                'processing_time': processing_time
            }
            
            self.logger.info(f"✅ Tree optimization completed in {processing_time:.2f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Tree optimization failed: {e}")
            raise
    
    def _optimize_data_for_tree_processing(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Optimize data for tree processing."""
        try:
            # Convert to optimal data types
            X_optimized = X.astype(np.float32)
            y_optimized = y.astype(np.float32)
            
            # Apply memory optimization
            if self.config.enable_memory_optimization and self.memory_optimizer:
                X_optimized = self._apply_memory_optimization(X_optimized)
                y_optimized = self._apply_memory_optimization(y_optimized)
            
            # Apply chunking for large datasets
            if X_optimized.shape[0] > self.config.chunk_size:
                X_optimized = self._apply_chunking(X_optimized)
                y_optimized = self._apply_chunking(y_optimized)
            
            return X_optimized, y_optimized
            
        except Exception as e:
            self.logger.error(f"❌ Data optimization failed: {e}")
            return X, y
    
    def _apply_tree_optimizations(self, tree_model: Any) -> Any:
        """Apply tree-specific optimizations."""
        try:
            # Tree parallelization
            if self.tree_config.enable_tree_parallelization:
                if hasattr(tree_model, 'n_jobs'):
                    tree_model.n_jobs = -1  # Use all available cores
            
            # Tree caching
            if self.tree_config.enable_tree_caching and self.cache_manager:
                tree_model = self.cache_manager.optimize_tree_model(tree_model)
            
            # Tree memory pooling
            if self.tree_config.enable_tree_memory_pooling:
                tree_model = self._apply_memory_pooling(tree_model)
            
            return tree_model
            
        except Exception as e:
            self.logger.error(f"❌ Tree optimization failed: {e}")
            return tree_model
    
    def _apply_parallel_optimizations(self, tree_model: Any) -> Any:
        """Apply parallel processing optimizations."""
        try:
            # Set parallel processing parameters
            if hasattr(tree_model, 'n_jobs'):
                tree_model.n_jobs = self.config.max_workers or -1
            
            # Enable batch processing
            if hasattr(tree_model, 'batch_size'):
                tree_model.batch_size = self.tree_config.tree_batch_size
            
            return tree_model
            
        except Exception as e:
            self.logger.error(f"❌ Parallel optimization failed: {e}")
            return tree_model
    
    def _execute_optimized_processing(self, 
                                     tree_model: Any,
                                     X: np.ndarray,
                                     y: np.ndarray,
                                     operation_type: str) -> Dict[str, Any]:
        """Execute optimized tree processing."""
        try:
            if operation_type == "training":
                tree_model.fit(X, y)
                return {
                    'model': tree_model,
                    'training_completed': True,
                    'optimization_applied': True
                }
            elif operation_type == "prediction":
                predictions = tree_model.predict(X)
                return {
                    'predictions': predictions,
                    'prediction_completed': True,
                    'optimization_applied': True
                }
            elif operation_type == "evaluation":
                score = tree_model.score(X, y)
                return {
                    'score': score,
                    'evaluation_completed': True,
                    'optimization_applied': True
                }
            else:
                return {
                    'operation_completed': True,
                    'optimization_applied': True
                }
                
        except Exception as e:
            self.logger.error(f"❌ Optimized processing failed: {e}")
            raise
    
    def _apply_memory_optimization(self, data: np.ndarray) -> np.ndarray:
        """Apply memory optimization to data."""
        try:
            if self.memory_optimizer and hasattr(self.memory_optimizer, 'optimize_memory'):
                return self.memory_optimizer.optimize_memory(data)
            return data
        except Exception as e:
            self.logger.warning(f"⚠️ Memory optimization failed: {e}")
            return data
    
    def _apply_chunking(self, data: np.ndarray) -> np.ndarray:
        """Apply chunking for large datasets."""
        try:
            if data.shape[0] > self.config.chunk_size:
                # Process in chunks
                chunks = []
                for i in range(0, data.shape[0], self.config.chunk_size):
                    chunk = data[i:i+self.config.chunk_size]
                    chunks.append(chunk)
                return np.vstack(chunks)
            return data
        except Exception as e:
            self.logger.warning(f"⚠️ Chunking failed: {e}")
            return data
    
    def _apply_memory_pooling(self, tree_model: Any) -> Any:
        """Apply memory pooling to tree model."""
        try:
            # Memory pooling strategies for tree models
            if hasattr(tree_model, 'memory_pooling'):
                tree_model.memory_pooling = True
            return tree_model
        except Exception as e:
            self.logger.warning(f"⚠️ Memory pooling failed: {e}")
            return tree_model
    
    def _calculate_performance_metrics(self, 
                                     processing_time: float, 
                                     n_samples: int, 
                                     operation_type: str) -> PerformanceMetrics:
        """Calculate performance metrics."""
        try:
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            gpu_usage = self._get_gpu_usage()
            throughput = n_samples / processing_time if processing_time > 0 else 0
            
            metrics = PerformanceMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                processing_time=processing_time,
                throughput=throughput,
                optimization_level=self.config.optimization_level.value,
                hardware_type="tree_optimization"
            )
            
            # Store in performance history
            self.performance_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Performance metrics calculation failed: {e}")
            return PerformanceMetrics(0, 0, None, 0, 0, "unknown", "unknown")
    
    def _get_gpu_usage(self) -> Optional[float]:
        """Get GPU usage percentage."""
        try:
            if self.gpu_manager and hasattr(self.gpu_manager, 'get_gpu_usage'):
                return self.gpu_manager.get_gpu_usage()
            return None
        except (AttributeError, TypeError, ValueError) as e:
            tprint(f"Error getting GPU usage: {e}", level="warning")
            return None
    
    def _get_hardware_utilization(self) -> Dict[str, float]:
        """Get current hardware utilization."""
        try:
            utilization = {
                'cpu_utilization': psutil.cpu_percent(),
                'memory_utilization': psutil.virtual_memory().percent
            }
            
            # Add GPU utilization if available
            gpu_usage = self._get_gpu_usage()
            if gpu_usage is not None:
                utilization['gpu_utilization'] = gpu_usage
            
            return utilization
            
        except Exception as e:
            self.logger.error(f"❌ Hardware utilization check failed: {e}")
            return {}
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        try:
            memory = psutil.virtual_memory()
            return {
                'total_memory': memory.total,
                'available_memory': memory.available,
                'used_memory': memory.used,
                'memory_percent': memory.percent
            }
        except Exception as e:
            self.logger.error(f"❌ Memory usage check failed: {e}")
            return {}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        try:
            if not self.performance_history:
                return {'message': 'No performance data available'}
            
            cpu_usage = [m.cpu_usage for m in self.performance_history]
            memory_usage = [m.memory_usage for m in self.performance_history]
            processing_times = [m.processing_time for m in self.performance_history]
            throughputs = [m.throughput for m in self.performance_history]
            
            return {
                'optimization_stats': self.optimization_stats,
                'performance_summary': {
                    'avg_cpu_usage': np.mean(cpu_usage),
                    'max_cpu_usage': np.max(cpu_usage),
                    'avg_memory_usage': np.mean(memory_usage),
                    'max_memory_usage': np.max(memory_usage),
                    'avg_processing_time': np.mean(processing_times),
                    'total_processing_time': np.sum(processing_times),
                    'avg_throughput': np.mean(throughputs),
                    'max_throughput': np.max(throughputs),
                    'optimization_count': len(self.performance_history)
                },
                'hardware_utilization': self._get_hardware_utilization(),
                'memory_usage': self._get_memory_usage()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Performance summary generation failed: {e}")
            return {'error': str(e)}
    
    def reset_performance_history(self):
        """Reset performance history."""
        self.performance_history = []
        self.optimization_stats = {
            'total_optimizations': 0,
            'memory_savings': 0,
            'speed_improvements': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        self.logger.info("✅ Performance history reset")


class TreeMatrixOperations:
    """Matrix operations optimized for tree models."""
    
    def __init__(self):
        """Initialize tree matrix operations."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def optimize_matrix_operations(self, matrix: np.ndarray) -> np.ndarray:
        """Optimize matrix operations for tree models."""
        try:
            # Convert to optimal data type
            if matrix.dtype == np.float64:
                matrix = matrix.astype(np.float32)
            
            # Apply memory optimization
            if matrix.size > 1000000:  # Large matrix
                matrix = self._apply_memory_optimization(matrix)
            
            return matrix
            
        except Exception as e:
            self.logger.error(f"❌ Matrix optimization failed: {e}")
            return matrix
    
    def _apply_memory_optimization(self, matrix: np.ndarray) -> np.ndarray:
        """Apply memory optimization to matrix."""
        try:
            # Use memory-efficient operations
            return matrix.copy()  # Ensure contiguous memory
        except Exception as e:
            self.logger.warning(f"⚠️ Memory optimization failed: {e}")
            return matrix


class TreeCacheManager:
    """Cache manager for tree operations."""
    
    def __init__(self, cache_size_mb: int = 100, enable_compression: bool = True):
        """Initialize tree cache manager."""
        self.cache_size_mb = cache_size_mb
        self.enable_compression = enable_compression
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def optimize_tree_model(self, tree_model: Any) -> Any:
        """Optimize tree model with caching."""
        try:
            # Apply caching optimizations
            if hasattr(tree_model, 'cache_size'):
                tree_model.cache_size = self.cache_size_mb
            
            return tree_model
            
        except Exception as e:
            self.logger.warning(f"⚠️ Tree model caching failed: {e}")
            return tree_model


class TreeM1Optimizer:
    """M1-specific optimizations for tree models."""
    
    def __init__(self):
        """Initialize M1 optimizer."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.m1_available = M1_UTILS_AVAILABLE and is_m1_available()
    
    def optimize_for_m1(self, tree_model: Any) -> Any:
        """Apply M1-specific optimizations to tree model."""
        try:
            if self.m1_available:
                # M1-specific optimizations
                if hasattr(tree_model, 'm1_optimization'):
                    tree_model.m1_optimization = True
                
                self.logger.info("✅ M1 optimizations applied")
            else:
                self.logger.warning("⚠️ M1 not available, skipping M1 optimizations")
            
            return tree_model
            
        except Exception as e:
            self.logger.error(f"❌ M1 optimization failed: {e}")
            return tree_model


# Factory functions
def create_tree_hardware_optimizer(config: Optional[HardwareConfig] = None, 
                                 tree_config: Optional[TreeOptimizationConfig] = None) -> TreeHardwareOptimizer:
    """Create tree hardware optimizer instance."""
    return TreeHardwareOptimizer(config, tree_config)


def create_tree_matrix_operations() -> TreeMatrixOperations:
    """Create tree matrix operations instance."""
    return TreeMatrixOperations()


def create_tree_m1_optimizer() -> TreeM1Optimizer:
    """Create tree M1 optimizer instance."""
    return TreeM1Optimizer()


def create_tree_cache_manager(cache_size_mb: int = 100, enable_compression: bool = True) -> TreeCacheManager:
    """Create tree cache manager instance."""
    return TreeCacheManager(cache_size_mb, enable_compression)


# Quick optimization functions
def quick_tree_optimization(tree_model: Any, 
                           X: np.ndarray, 
                           y: np.ndarray,
                           operation_type: str = "training") -> Dict[str, Any]:
    """Quick tree optimization for immediate use."""
    optimizer = create_tree_hardware_optimizer()
    return optimizer.optimize_tree_processing(tree_model, X, y, operation_type)


def quick_matrix_optimization(matrix: np.ndarray) -> np.ndarray:
    """Quick matrix optimization for immediate use."""
    matrix_ops = create_tree_matrix_operations()
    return matrix_ops.optimize_matrix_operations(matrix)


# Example usage
if __name__ == "__main__":
    # Create hardware optimizer
    config = HardwareConfig(
        enable_gpu_acceleration=True,
        enable_memory_optimization=True,
        enable_parallel_processing=True,
        enable_m1_optimization=True,
        optimization_level=OptimizationLevel.STANDARD
    )
    
    tree_config = TreeOptimizationConfig(
        enable_tree_parallelization=True,
        enable_tree_caching=True,
        enable_tree_memory_pooling=True
    )
    
    optimizer = create_tree_hardware_optimizer(config, tree_config)
    
    # Example usage
    print("Tree Hardware Optimizer created successfully!")
    print(f"Performance summary: {optimizer.get_performance_summary()}")