"""
Core Engine for Unified Matrix Operations

This module provides the AresOptimizer class that consolidates all matrix and vector
operations from across the codebase while retaining ALL existing capabilities.
"""

import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import warnings

# Optional dependencies
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

# Import all existing capabilities
try:
    from ..matrix_operations import get_unified_matrix_operations as _get_legacy_matrix_ops
    from ..enhanced_matrix_operations import get_enhanced_matrix_operations as _get_enhanced_matrix_ops
    from ..batch_matrix_operations import get_batch_matrix_processor as _get_batch_processor
    from ..vectorized_processing_core import get_vectorized_processing_core as _get_vectorized_core
    from ..unified_vectorization_manager import get_unified_vectorization_manager as _get_vectorization_manager
    from ..matrix_cross_validation import matrix_cross_validate as _matrix_cross_validate
    LEGACY_MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Some legacy modules not available: {e}")
    LEGACY_MODULES_AVAILABLE = False

# Import configuration
from .configuration import UnifiedConfiguration

logger = logging.getLogger(__name__)

class OperationMode(Enum):
    """Operation execution modes."""
    AUTO = "auto"
    GPU = "gpu"
    CPU = "cpu"
    PARALLEL = "parallel"
    MEMORY_OPTIMIZED = "memory_optimized"

class OptimizationTarget(Enum):
    """Optimization targets."""
    PERFORMANCE = "performance"
    MEMORY = "memory"
    ACCURACY = "accuracy"
    BALANCED = "balanced"

@dataclass
class PerformanceStats:
    """Performance statistics tracking."""
    total_operations: int = 0
    gpu_operations: int = 0
    cpu_operations: int = 0
    parallel_operations: int = 0
    memory_optimized_operations: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    peak_memory_usage_mb: float = 0.0
    optimization_mode: str = "auto"

class AresOptimizer:
    """
    Unified Matrix and Vector Operations Engine.
    
    This class consolidates ALL matrix and vector operations from across the codebase,
    providing a single, optimized interface while retaining all existing capabilities.
    
    Features:
    - GPU acceleration (MPS/CUDA) with automatic fallback
    - Memory optimization and chunked processing
    - Parallel processing and vectorization
    - Cross-validation with matrix optimization
    - M1 hardware optimization
    - Automatic configuration based on hardware and data
    - Performance monitoring and optimization
    """
    
    def __init__(self, config: Optional[Dict] = None, optimization_target: str = "balanced"):
        """
        Initialize the AresOptimizer.
        
        Args:
            config: Optional configuration dictionary
            optimization_target: Optimization target ("performance", "memory", "accuracy", "balanced")
        """
        self.logger = logger.getChild('AresOptimizer')
        
        # Load configuration
        if config is None:
            config = UnifiedConfiguration.create_optimal_config(optimization_target)
        self.config = config
        
        # Initialize performance tracking
        self.performance_stats = PerformanceStats(
            optimization_mode=optimization_target
        )
        
        # Initialize all engines
        self._initialize_engines()
        
        # Track initialization
        self._log_initialization()
    
    def _initialize_engines(self):
        """Initialize all optimization engines."""
        try:
            # Initialize legacy engines for full capability retention
            if LEGACY_MODULES_AVAILABLE:
                # Matrix operations engines
                self._legacy_matrix_ops = _get_legacy_matrix_ops(
                    enable_gpu=self.config.get('enable_gpu', True),
                    enable_memory_optimization=self.config.get('enable_memory_optimization', True),
                    enable_parallel=self.config.get('enable_parallel_processing', True)
                )
                
                self._enhanced_matrix_ops = _get_enhanced_matrix_ops(
                    use_gpu=self.config.get('enable_gpu', True),
                    memory_efficient=self.config.get('enable_memory_optimization', True)
                )
                
                self._batch_processor = _get_batch_processor(
                    chunk_size_mb=self.config.get('chunk_size_mb', 256),
                    enable_gpu=self.config.get('enable_gpu', True),
                    enable_parallel=self.config.get('enable_parallel_processing', True)
                )
                
                # Vectorization engines
                self._vectorized_core = _get_vectorized_core(
                    chunk_size=self.config.get('chunk_size', 50000),
                    max_memory_gb=self.config.get('max_memory_gb', 8.0),
                    enable_gpu=self.config.get('enable_gpu', True)
                )
                
                self._vectorization_manager = _get_vectorization_manager()
                
                self.logger.info("✅ All legacy engines initialized successfully")
            else:
                self.logger.warning("⚠️ Legacy modules not available - using fallback implementations")
                self._initialize_fallback_engines()
                
        except Exception as e:
            self.logger.error(f"❌ Error initializing engines: {e}")
            self._initialize_fallback_engines()
    
    def _initialize_fallback_engines(self):
        """Initialize fallback engines when legacy modules are not available."""
        self._legacy_matrix_ops = None
        self._enhanced_matrix_ops = None
        self._batch_processor = None
        self._vectorized_core = None
        self._vectorization_manager = None
        self.logger.info("ℹ️ Using fallback implementations")
    
    def _log_initialization(self):
        """Log initialization details."""
        self.logger.info("🚀 AresOptimizer initialized")
        self.logger.info(f"📊 Configuration: {self.config.get('optimization_mode', 'auto')}")
        self.logger.info(f"🖥️ GPU enabled: {self.config.get('enable_gpu', False)}")
        self.logger.info(f"🧠 Memory optimization: {self.config.get('enable_memory_optimization', False)}")
        self.logger.info(f"⚡ Parallel processing: {self.config.get('enable_parallel_processing', False)}")
        self.logger.info(f"📈 Legacy engines available: {LEGACY_MODULES_AVAILABLE}")
    
    # ============================================================================
    # MATRIX OPERATIONS - Retaining ALL existing capabilities
    # ============================================================================
    
    def matrix_multiply(self, a, b, use_gpu: Optional[bool] = None):
        """
        Optimized matrix multiplication with GPU acceleration and automatic fallback.
        
        Retains capabilities from:
        - matrix_operations.py
        - enhanced_matrix_operations.py
        - batch_matrix_operations.py
        """
        start_time = time.time()
        
        try:
            # Try enhanced matrix operations first (best performance)
            if self._enhanced_matrix_ops:
                result = self._enhanced_matrix_ops.matrix_multiply(a, b, use_gpu)
                self.performance_stats.gpu_operations += 1
            # Fallback to legacy matrix operations
            elif self._legacy_matrix_ops:
                result = self._legacy_matrix_ops.matrix_multiply(a, b)
                self.performance_stats.gpu_operations += 1
            else:
                # Fallback to basic operations
                if NUMPY_AVAILABLE and hasattr(a, '__matmul__'):
                    result = a @ b
                else:
                    # Ultimate fallback - this shouldn't happen in practice
                    raise RuntimeError("No suitable matrix multiplication implementation available")
                self.performance_stats.cpu_operations += 1
            
            self._update_performance_stats(start_time)
            return result
            
        except Exception as e:
            self.logger.warning(f"⚠️ Matrix multiplication failed, using fallback: {e}")
            # Ultimate fallback
            if NUMPY_AVAILABLE and hasattr(a, '__matmul__'):
                result = a @ b
            else:
                raise RuntimeError(f"Matrix multiplication failed and no fallback available: {e}")
            self.performance_stats.cpu_operations += 1
            self._update_performance_stats(start_time)
            return result
    
    def correlation_matrix(self, data, method: str = 'pearson'):
        """
        Compute correlation matrix with optimization.
        
        Retains capabilities from:
        - matrix_operations.py (safe_correlation_matrix)
        - enhanced_matrix_operations.py (correlation_matrix)
        - vectorized_processing_core.py (matrix_correlation_analysis)
        """
        start_time = time.time()
        
        try:
            # Try legacy matrix operations first
            if self._legacy_matrix_ops:
                result = self._legacy_matrix_ops.safe_correlation_matrix(data, method)
            # Try enhanced matrix operations
            elif self._enhanced_matrix_ops:
                result = self._enhanced_matrix_ops.correlation_matrix(data, method)
            # Try vectorized core
            elif self._vectorized_core:
                result, _ = self._vectorized_core.matrix_correlation_analysis(data, method)
            else:
                # Fallback to pandas/numpy
                if PANDAS_AVAILABLE and hasattr(data, 'corr'):
                    result = data.corr(method=method).values
                elif NUMPY_AVAILABLE:
                    result = np.corrcoef(data.T)
                else:
                    raise RuntimeError("No correlation computation implementation available")
            
            self._update_performance_stats(start_time)
            return result
            
        except Exception as e:
            self.logger.warning(f"⚠️ Correlation matrix computation failed, using fallback: {e}")
            # Ultimate fallback
            if PANDAS_AVAILABLE and hasattr(data, 'corr'):
                result = data.corr(method=method).values
            elif NUMPY_AVAILABLE:
                result = np.corrcoef(data.T)
            else:
                raise RuntimeError(f"Correlation computation failed and no fallback available: {e}")
            self._update_performance_stats(start_time)
            return result
    
    def svd_decomposition(self, matrix, k: Optional[int] = None,
                         use_gpu: Optional[bool] = None):
        """
        SVD decomposition with GPU acceleration.
        
        Retains capabilities from:
        - matrix_operations.py (svd_decomposition)
        - enhanced_matrix_operations.py (svd_decomposition)
        """
        start_time = time.time()
        
        try:
            # Try enhanced matrix operations first
            if self._enhanced_matrix_ops:
                result = self._enhanced_matrix_ops.svd_decomposition(matrix, k, use_gpu)
            # Try legacy matrix operations
            elif self._legacy_matrix_ops:
                result = self._legacy_matrix_ops.svd_decomposition(matrix, k)
            else:
                # Fallback to numpy
                result = np.linalg.svd(matrix, full_matrices=False)
                if k is not None:
                    U, s, V = result
                    result = (U[:, :k], s[:k], V[:k, :])
            
            self._update_performance_stats(start_time)
            return result
            
        except Exception as e:
            self.logger.warning(f"⚠️ SVD decomposition failed, using fallback: {e}")
            # Ultimate fallback
            result = np.linalg.svd(matrix, full_matrices=False)
            if k is not None:
                U, s, V = result
                result = (U[:, :k], s[:k], V[:k, :])
            self._update_performance_stats(start_time)
            return result
    
    def eigendecomposition(self, matrix,
                          use_gpu: Optional[bool] = None):
        """
        Eigendecomposition with GPU acceleration.
        
        Retains capabilities from:
        - matrix_operations.py (eigendecomposition)
        - enhanced_matrix_operations.py (eigendecomposition)
        """
        start_time = time.time()
        
        try:
            # Try enhanced matrix operations first
            if self._enhanced_matrix_ops:
                result = self._enhanced_matrix_ops.eigendecomposition(matrix, use_gpu)
            # Try legacy matrix operations
            elif self._legacy_matrix_ops:
                result = self._legacy_matrix_ops.eigendecomposition(matrix)
            else:
                # Fallback to numpy
                result = np.linalg.eig(matrix)
            
            self._update_performance_stats(start_time)
            return result
            
        except Exception as e:
            self.logger.warning(f"⚠️ Eigendecomposition failed, using fallback: {e}")
            # Ultimate fallback
            result = np.linalg.eig(matrix)
            self._update_performance_stats(start_time)
            return result
    
    def matrix_inverse(self, matrix,
                      use_gpu: Optional[bool] = None):
        """
        Matrix inversion with GPU acceleration.
        
        Retains capabilities from:
        - matrix_operations.py (matrix_inverse)
        - enhanced_matrix_operations.py (matrix_inverse)
        """
        start_time = time.time()
        
        try:
            # Try enhanced matrix operations first
            if self._enhanced_matrix_ops:
                result = self._enhanced_matrix_ops.matrix_inverse(matrix, use_gpu)
            # Try legacy matrix operations
            elif self._legacy_matrix_ops:
                result = self._legacy_matrix_ops.matrix_inverse(matrix)
            else:
                # Fallback to numpy
                result = np.linalg.inv(matrix)
            
            self._update_performance_stats(start_time)
            return result
            
        except Exception as e:
            self.logger.warning(f"⚠️ Matrix inversion failed, using fallback: {e}")
            # Ultimate fallback
            result = np.linalg.inv(matrix)
            self._update_performance_stats(start_time)
            return result
    
    def batch_matrix_multiply(self, matrices_a: List,
                            matrices_b: List,
                            batch_size: Optional[int] = None):
        """
        Batch matrix multiplication with optimization.
        
        Retains capabilities from:
        - batch_matrix_operations.py (batch_matrix_multiply)
        - enhanced_matrix_operations.py (batch_matrix_multiply)
        """
        start_time = time.time()
        
        try:
            # Try batch processor first
            if self._batch_processor:
                result = self._batch_processor.batch_matrix_multiply(matrices_a, matrices_b)
                self.performance_stats.parallel_operations += 1
            # Try enhanced matrix operations
            elif self._enhanced_matrix_ops:
                result = self._enhanced_matrix_ops.batch_matrix_multiply(matrices_a, matrices_b, batch_size)
                self.performance_stats.parallel_operations += 1
            else:
                # Fallback to sequential processing
                result = []
                for a, b in zip(matrices_a, matrices_b):
                    result.append(self.matrix_multiply(a, b))
                self.performance_stats.cpu_operations += len(matrices_a)
            
            self._update_performance_stats(start_time)
            return result
            
        except Exception as e:
            self.logger.warning(f"⚠️ Batch matrix multiplication failed, using fallback: {e}")
            # Ultimate fallback
            result = []
            for a, b in zip(matrices_a, matrices_b):
                result.append(np.dot(a, b))
            self.performance_stats.cpu_operations += len(matrices_a)
            self._update_performance_stats(start_time)
            return result
    
    # ============================================================================
    # VECTORIZATION OPERATIONS - Retaining ALL existing capabilities
    # ============================================================================
    
    def vectorize_features(self, data,
                          windows: List[int] = None,
                          features: List[str] = None):
        """
        Vectorized feature engineering.
        
        Retains capabilities from:
        - vectorized_processing_core.py (vectorized_rolling_features)
        - unified_vectorization_manager.py
        """
        start_time = time.time()
        
        if windows is None:
            windows = [5, 10, 20, 50]
        
        try:
            # Try vectorized core first
            if self._vectorized_core:
                result = self._vectorized_core.vectorized_rolling_features(data, windows, features)
                self.performance_stats.parallel_operations += 1
            # Try vectorization manager
            elif self._vectorization_manager:
                result = self._vectorization_manager.optimize_operation(
                    operation_type=self._vectorization_manager.OperationType.FEATURE_ENGINEERING,
                    data=data
                ).result
                self.performance_stats.parallel_operations += 1
            else:
                # Fallback to pandas
                result = data.copy()
                for window in windows:
                    for col in (features or data.columns):
                        if col in data.columns:
                            result[f'{col}_rolling_mean_{window}'] = data[col].rolling(window).mean()
                            result[f'{col}_rolling_std_{window}'] = data[col].rolling(window).std()
            
            self._update_performance_stats(start_time)
            return result
            
        except Exception as e:
            self.logger.warning(f"⚠️ Vectorized feature engineering failed, using fallback: {e}")
            # Ultimate fallback
            result = data.copy()
            for window in windows:
                for col in (features or data.columns):
                    if col in data.columns:
                        result[f'{col}_rolling_mean_{window}'] = data[col].rolling(window).mean()
                        result[f'{col}_rolling_std_{window}'] = data[col].rolling(window).std()
            self._update_performance_stats(start_time)
            return result
    
    def optimize_dataframe(self, df):
        """
        Optimize DataFrame for processing.
        
        Retains capabilities from:
        - vectorized_processing_core.py (optimize_dataframe_for_processing)
        """
        start_time = time.time()
        
        try:
            # Try vectorized core first
            if self._vectorized_core:
                result = self._vectorized_core.optimize_dataframe_for_processing(df)
                self.performance_stats.memory_optimized_operations += 1
            else:
                # Fallback implementation
                result = df.copy()
                # Basic optimization
                for col in df.select_dtypes(include=['object']):
                    if df[col].nunique() / len(df) < 0.5:
                        result[col] = df[col].astype('category')
            
            self._update_performance_stats(start_time)
            return result
            
        except Exception as e:
            self.logger.warning(f"⚠️ DataFrame optimization failed, using fallback: {e}")
            result = df.copy()
            self._update_performance_stats(start_time)
            return result
    
    # ============================================================================
    # CROSS-VALIDATION OPERATIONS - Retaining ALL existing capabilities
    # ============================================================================
    
    def cross_validate(self, X,
                      y,
                      model_class: Any,
                      model_params: Dict[str, Any] = None,
                      n_splits: int = 5,
                      parallel: bool = True,
                      max_workers: int = 4):
        """
        Cross-validation with matrix optimization.
        
        Retains capabilities from:
        - matrix_cross_validation.py (matrix_cross_validate)
        - unified_vectorization_manager.py (optimize_cross_validation)
        """
        start_time = time.time()
        
        try:
            # Try matrix cross-validation first
            if LEGACY_MODULES_AVAILABLE:
                result = _matrix_cross_validate(
                    X, y, model_class, model_params,
                    n_splits=n_splits,
                    use_gpu=self.config.get('enable_gpu', True),
                    parallel=parallel,
                    max_workers=max_workers
                )
                self.performance_stats.parallel_operations += 1
            # Try vectorization manager
            elif self._vectorization_manager:
                result = self._vectorization_manager.optimize_operation(
                    operation_type=self._vectorization_manager.OperationType.CROSS_VALIDATION,
                    data={'X': X, 'y': y, 'model_class': model_class}
                ).result
                self.performance_stats.parallel_operations += 1
            else:
                # Fallback to basic cross-validation
                from sklearn.model_selection import cross_val_score
                model = model_class(**(model_params or {}))
                scores = cross_val_score(model, X, y, cv=n_splits)
                result = {
                    'scores': scores,
                    'mean_score': scores.mean(),
                    'std_score': scores.std()
                }
            
            self._update_performance_stats(start_time)
            return result
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cross-validation failed, using fallback: {e}")
            # Ultimate fallback
            from sklearn.model_selection import cross_val_score
            model = model_class(**(model_params or {}))
            scores = cross_val_score(model, X, y, cv=n_splits)
            result = {
                'scores': scores,
                'mean_score': scores.mean(),
                'std_score': scores.std()
            }
            self._update_performance_stats(start_time)
            return result
    
    # ============================================================================
    # MEMORY AND PERFORMANCE MANAGEMENT
    # ============================================================================
    
    def optimize_memory(self):
        """
        Optimize memory usage.
        
        Retains capabilities from:
        - matrix_operations.py (optimize_memory_usage)
        - vectorized_processing_core.py (memory optimization)
        """
        start_time = time.time()
        
        try:
            # Try legacy matrix operations first
            if self._legacy_matrix_ops:
                result = self._legacy_matrix_ops.optimize_memory_usage()
            # Try vectorized core
            elif self._vectorized_core and hasattr(self._vectorized_core, 'memory_optimizer'):
                if self._vectorized_core.memory_optimizer:
                    result = self._vectorized_core.memory_optimizer.optimize_memory()
                else:
                    result = {'status': 'fallback_gc'}
            else:
                # Fallback to garbage collection
                import gc
                gc.collect()
                result = {'status': 'fallback_gc', 'freed_mb': 0}
            
            self.performance_stats.memory_optimized_operations += 1
            self._update_performance_stats(start_time)
            return result
            
        except Exception as e:
            self.logger.warning(f"⚠️ Memory optimization failed: {e}")
            # Ultimate fallback
            import gc
            gc.collect()
            result = {'status': 'fallback_gc', 'freed_mb': 0}
            self._update_performance_stats(start_time)
            return result
    
    def get_performance_stats(self):
        """Get comprehensive performance statistics."""
        stats = {
            'ares_optimizer': {
                'total_operations': self.performance_stats.total_operations,
                'gpu_operations': self.performance_stats.gpu_operations,
                'cpu_operations': self.performance_stats.cpu_operations,
                'parallel_operations': self.performance_stats.parallel_operations,
                'memory_optimized_operations': self.performance_stats.memory_optimized_operations,
                'total_execution_time': self.performance_stats.total_execution_time,
                'average_execution_time': self.performance_stats.average_execution_time,
                'peak_memory_usage_mb': self.performance_stats.peak_memory_usage_mb,
                'optimization_mode': self.performance_stats.optimization_mode
            },
            'configuration': self.config.copy(),
            'legacy_engines_available': LEGACY_MODULES_AVAILABLE
        }
        
        # Add stats from legacy engines if available
        if self._legacy_matrix_ops:
            try:
                stats['legacy_matrix_ops'] = self._legacy_matrix_ops.get_performance_stats()
            except:
                pass
        
        if self._enhanced_matrix_ops:
            try:
                stats['enhanced_matrix_ops'] = self._enhanced_matrix_ops.get_performance_stats()
            except:
                pass
        
        if self._vectorized_core:
            try:
                stats['vectorized_core'] = self._vectorized_core.get_processing_stats()
            except:
                pass
        
        return stats
    
    def _update_performance_stats(self, start_time: float):
        """Update performance statistics."""
        execution_time = time.time() - start_time
        
        self.performance_stats.total_operations += 1
        self.performance_stats.total_execution_time += execution_time
        self.performance_stats.average_execution_time = (
            self.performance_stats.total_execution_time / self.performance_stats.total_operations
        )
    
    # ============================================================================
    # CONVENIENCE METHODS - Making common operations easy
    # ============================================================================
    
    def multiply(self, a, b):
        """Alias for matrix_multiply."""
        return self.matrix_multiply(a, b)
    
    def correlate(self, data, method: str = 'pearson'):
        """Alias for correlation_matrix."""
        return self.correlation_matrix(data, method)
    
    def svd(self, matrix, k: Optional[int] = None):
        """Alias for svd_decomposition."""
        return self.svd_decomposition(matrix, k)
    
    def eigen(self, matrix):
        """Alias for eigendecomposition."""
        return self.eigendecomposition(matrix)
    
    def inv(self, matrix):
        """Alias for matrix_inverse."""
        return self.matrix_inverse(matrix)
    
    def cv(self, X, y, model_class, **kwargs):
        """Alias for cross_validate."""
        return self.cross_validate(X, y, model_class, **kwargs)
    
    def vectorize(self, data, **kwargs):
        """Alias for vectorize_features."""
        return self.vectorize_features(data, **kwargs)
    
    def optimize(self, data="memory"):
        """Optimize data or memory."""
        if PANDAS_AVAILABLE and hasattr(data, 'corr'):  # Check if it's a DataFrame-like object
            return self.optimize_dataframe(data)
        elif data == "memory":
            return self.optimize_memory()
        else:
            raise ValueError(f"Unknown optimization target: {data}")
    
    # ============================================================================
    # CONTEXT MANAGER SUPPORT
    # ============================================================================
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if exc_type is not None:
            self.logger.error(f"Error in AresOptimizer context: {exc_val}")
        
        # Perform cleanup
        try:
            self.optimize_memory()
        except:
            pass
        
        return False  # Don't suppress exceptions