"""
Batch Matrix Operations - Unified Implementation

This module consolidates batch matrix operations functionality from scattered sources
into a single, unified interface with backwards compatibility.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from concurrent.futures import ThreadPoolExecutor
import time
import gc

# Conditional imports for optional dependencies
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

# Import existing utilities
try:
    from .unified_operations import get_unified_matrix_operations, UnifiedMatrixOperations
    from ..hardware.m1_gpu_utils import M1GPUManager
    from ..hardware.m1_memory_optimizer import M1MemoryOptimizer
    from ..hardware.m1_cpu_optimizer import M1CPUOptimizer
    UTILITIES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Some utilities not available: {e}")
    UTILITIES_AVAILABLE = False

# Import VectorBT optimizations
try:
    from .vectorbt_optimizations import get_vectorbt_optimized_operations
    VECTORBT_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"VectorBT optimizations not available: {e}")
    VECTORBT_OPTIMIZATIONS_AVAILABLE = False
    get_vectorbt_optimized_operations = None

# Import VectorBT Rolling Optimizer and Unified Vectorization Manager
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
    VECTORBT_ROLLING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"VectorBTRollingOptimizer not available: {e}")
    VECTORBT_ROLLING_AVAILABLE = False
    get_vectorbt_rolling_optimizer = None

try:
    from src.feature_generation.utils.unified_vectorization_manager import get_unified_vectorization_manager
    UNIFIED_VECTORIZATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"UnifiedVectorizationManager not available: {e}")
    UNIFIED_VECTORIZATION_AVAILABLE = False
    get_unified_vectorization_manager = None

logger = logging.getLogger(__name__)

class BatchMatrixProcessor:
    """
    High-performance batch matrix processor for data transformations.

    Optimized for:
    - Large-scale feature engineering
    - Batch data preprocessing
    - Matrix-based transformations
    - Memory-efficient processing
    """

    def __init__(self,
                 chunk_size_mb: int = 256,
                 enable_gpu: bool = True,
                 enable_parallel: bool = True,
                 max_workers: int = None):
        """Initialize batch matrix processor."""
        self.chunk_size_mb = chunk_size_mb
        self.enable_gpu = enable_gpu and UTILITIES_AVAILABLE
        self.enable_parallel = enable_parallel and UTILITIES_AVAILABLE
        self.max_workers = max_workers

        # Initialize utilities
        self.matrix_ops = None
        self.gpu_manager = None
        self.memory_optimizer = None
        self.cpu_optimizer = None

        if UTILITIES_AVAILABLE:
            try:
                self.matrix_ops = get_unified_matrix_operations()
                self.gpu_manager = M1GPUManager()
                self.memory_optimizer = M1MemoryOptimizer()
                self.cpu_optimizer = M1CPUOptimizer()
                logger.info("✅ Batch matrix processor initialized with M1 optimization")
            except Exception as e:
                logger.warning(f"⚠️ Error initializing utilities: {e}")

        # Initialize VectorBT managers
        self._initialize_vectorbt_managers()

    def _initialize_vectorbt_managers(self):
        """Initialize VectorBT managers for optimized operations."""
        try:
            # Initialize VectorBTRollingOptimizer
            if VECTORBT_ROLLING_AVAILABLE:
                self.rolling_optimizer = get_vectorbt_rolling_optimizer()
                if self.rolling_optimizer:
                    logger.debug("✅ VectorBTRollingOptimizer initialized")
                else:
                    logger.info("ℹ️ VectorBTRollingOptimizer not available")
            else:
                self.rolling_optimizer = None

            # Initialize UnifiedVectorizationManager
            if UNIFIED_VECTORIZATION_AVAILABLE:
                self.vectorization_manager = get_unified_vectorization_manager()
                if self.vectorization_manager:
                    logger.debug("✅ UnifiedVectorizationManager initialized")
                else:
                    logger.info("ℹ️ UnifiedVectorizationManager not available")
            else:
                self.vectorization_manager = None

        except Exception as e:
            logger.error(f"❌ Error initializing VectorBT managers: {e}")
            self.rolling_optimizer = None
            self.vectorization_manager = None

    def batch_matrix_multiply(self,
                            matrices_a: List['np.ndarray'],
                            matrices_b: List['np.ndarray']) -> List['np.ndarray']:
        """Batch matrix multiplication with VectorBT and automatic optimization."""
        if len(matrices_a) != len(matrices_b):
            raise ValueError("Matrices A and B must have same length")

        start_time = time.time()
        logger.info(f"🔄 Processing {len(matrices_a)} matrix multiplications")

        # Try VectorBT optimization first if available
        if VECTORBT_OPTIMIZATIONS_AVAILABLE:
            try:
                vectorbt_ops = get_vectorbt_optimized_operations()
                results = vectorbt_ops.batch_process(
                    matrices_a, 'batch_matrix_multiply', matrices_b=matrices_b
                )
                processing_time = time.time() - start_time
                logger.info(f"✅ VectorBT batch processing completed in {processing_time:.2f}s")
                return results
            except Exception as e:
                logger.warning(f"⚠️ VectorBT batch processing failed: {e}, falling back to standard method")

        # Determine processing strategy
        total_memory_mb = sum(a.nbytes + b.nbytes for a, b in zip(matrices_a, matrices_b)) / (1024 * 1024)

        if total_memory_mb > self.chunk_size_mb * 10:  # Large batch
            results = self._batch_process_large(matrices_a, matrices_b)
        elif self.enable_parallel and len(matrices_a) > 4:
            results = self._batch_process_parallel(matrices_a, matrices_b)
        else:
            results = self._batch_process_sequential(matrices_a, matrices_b)

        processing_time = time.time() - start_time
        logger.info(f"✅ Batch processing completed in {processing_time:.2f}s")

        return results

    def _batch_process_sequential(self,
                                matrices_a: List['np.ndarray'],
                                matrices_b: List['np.ndarray']) -> List['np.ndarray']:
        """Process matrices sequentially."""
        results = []
        for i, (a, b) in enumerate(zip(matrices_a, matrices_b)):
            if self.matrix_ops:
                result = self.matrix_ops.matrix_multiply(a, b)
            else:
                result = a @ b
            results.append(result)

            if (i + 1) % 10 == 0:
                logger.debug(f"📊 Processed {i + 1}/{len(matrices_a)} matrices")

        return results

    def _batch_process_parallel(self,
                               matrices_a: List['np.ndarray'],
                               matrices_b: List['np.ndarray']) -> List['np.ndarray']:
        """Process matrices in parallel."""
        logger.info("⚡ Using parallel processing for matrix multiplications")

        def multiply_pair(args):
            a, b = args
            if self.matrix_ops:
                return self.matrix_ops.matrix_multiply(a, b)
            else:
                return a @ b

        pairs = list(zip(matrices_a, matrices_b))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(multiply_pair, pairs))

        return results

    def _batch_process_large(self,
                           matrices_a: List['np.ndarray'],
                           matrices_b: List['np.ndarray']) -> List['np.ndarray']:
        """Process large batches with memory management."""
        logger.info("🧠 Using memory-managed processing for large batch")

        results = []
        chunk_size = max(1, int(self.chunk_size_mb * 1024 * 1024 /
                               (matrices_a[0].nbytes / len(matrices_a[0]))))

        for i in range(0, len(matrices_a), chunk_size):
            end_idx = min(i + chunk_size, len(matrices_a))
            chunk_a = matrices_a[i:end_idx]
            chunk_b = matrices_b[i:end_idx]

            chunk_results = self._batch_process_sequential(chunk_a, chunk_b)
            results.extend(chunk_results)

            # Memory cleanup
            if self.memory_optimizer:
                self.memory_optimizer.optimize_memory()
            else:
                gc.collect()

            logger.debug(f"📊 Processed chunk {i//chunk_size + 1}/{(len(matrices_a) + chunk_size - 1)//chunk_size}")

        return results

    def batch_feature_transformation(self,
                                   data: Union['np.ndarray', 'pd.DataFrame'],
                                   transformations: List[Dict[str, Any]]) -> Union['np.ndarray', 'pd.DataFrame']:
        """Apply batch feature transformations efficiently."""
        logger.info(f"🔄 Applying {len(transformations)} transformations to {data.shape[0]} samples")

        if isinstance(data, pd.DataFrame):
            result_df = data.copy()
        else:
            result_df = pd.DataFrame(data)

        for transform_spec in transformations:
            transform_type = transform_spec.get('type', 'identity')
            columns = transform_spec.get('columns', result_df.columns.tolist())
            params = transform_spec.get('params', {})

            if transform_type == 'standardize':
                result_df[columns] = self._batch_standardize(result_df[columns])
            elif transform_type == 'normalize':
                result_df[columns] = self._batch_normalize(result_df[columns])
            elif transform_type == 'robust_scale':
                result_df[columns] = self._batch_robust_scale(result_df[columns])
            elif transform_type == 'power_transform':
                result_df[columns] = self._batch_power_transform(result_df[columns], **params)
            elif transform_type == 'quantile_transform':
                result_df[columns] = self._batch_quantile_transform(result_df[columns], **params)

        if isinstance(data, np.ndarray):
            return result_df.values
        return result_df

    def _batch_standardize(self, data: 'pd.DataFrame') -> 'pd.DataFrame':
        """Vectorized standardization (z-score normalization)."""
        # VECTORIZED: Compute mean and std in single operations
        means = data.mean()
        stds = data.std()

        # VECTORIZED: Apply transformation
        standardized = (data - means) / stds.replace(0, 1)  # Avoid division by zero

        return standardized

    def _batch_normalize(self, data: 'pd.DataFrame') -> 'pd.DataFrame':
        """Vectorized min-max normalization."""
        # VECTORIZED: Compute min and max in single operations
        mins = data.min()
        maxs = data.max()

        # VECTORIZED: Apply transformation
        normalized = (data - mins) / (maxs - mins).replace(0, 1)  # Avoid division by zero

        return normalized

    def _batch_robust_scale(self, data: 'pd.DataFrame') -> 'pd.DataFrame':
        """Vectorized robust scaling using median and IQR."""
        # VECTORIZED: Compute median and IQR
        medians = data.median()
        q75 = data.quantile(0.75)
        q25 = data.quantile(0.25)
        iqrs = q75 - q25

        # VECTORIZED: Apply transformation
        robust_scaled = (data - medians) / iqrs.replace(0, 1)  # Avoid division by zero

        return robust_scaled

    def _batch_power_transform(self, data: 'pd.DataFrame', method: str = 'yeo-johnson') -> 'pd.DataFrame':
        """Vectorized power transformation."""
        from sklearn.preprocessing import PowerTransformer

        transformer = PowerTransformer(method=method)
        transformed = transformer.fit_transform(data.values)
        return pd.DataFrame(transformed, columns=data.columns, index=data.index)

    def _batch_quantile_transform(self, data: 'pd.DataFrame', n_quantiles: int = 1000) -> 'pd.DataFrame':
        """Vectorized quantile transformation."""
        from sklearn.preprocessing import QuantileTransformer

        transformer = QuantileTransformer(n_quantiles=n_quantiles, output_distribution='normal')
        transformed = transformer.fit_transform(data.values)
        return pd.DataFrame(transformed, columns=data.columns, index=data.index)

    def batch_correlation_analysis(self,
                                 data: Union['np.ndarray', 'pd.DataFrame'],
                                 method: str = 'pearson') -> Tuple['np.ndarray', 'np.ndarray']:
        """Batch correlation analysis with memory optimization."""
        logger.info(f"📊 Computing {method} correlations for {data.shape[1]} features")

        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = data

        # Handle large datasets with chunking
        n_features = data_array.shape[1]
        memory_mb = data_array.nbytes / (1024 * 1024)

        if memory_mb > self.chunk_size_mb:
            logger.info("🧠 Using chunked correlation analysis for large dataset")
            return self._chunked_correlation_analysis(data_array, method)
        else:
            return self._direct_correlation_analysis(data_array, method)

    def _direct_correlation_analysis(self, data: 'np.ndarray', method: str) -> Tuple['np.ndarray', 'np.ndarray']:
        """Direct correlation analysis for smaller datasets."""
        if self.matrix_ops:
            # Use our optimized correlation matrix
            corr_matrix = self.matrix_ops.safe_correlation_matrix(data)
        else:
            # Fallback to numpy
            corr_matrix = np.corrcoef(data.T)

        # For p-values, we'll use a simplified approach
        n = data.shape[0]
        p_values = np.full_like(corr_matrix, 0.0)

        # Calculate p-values using t-distribution approximation
        t_stats = corr_matrix * np.sqrt((n - 2) / (1 - corr_matrix**2))
        p_values = 2 * (1 - np.array([self._t_cdf(abs(t), n-2) for t in t_stats.flatten()]).reshape(corr_matrix.shape))

        return corr_matrix, p_values

    def _chunked_correlation_analysis(self, data: 'np.ndarray', method: str) -> Tuple['np.ndarray', 'np.ndarray']:
        """Chunked correlation analysis for large datasets."""
        n_features = data.shape[1]
        chunk_size = min(1000, n_features // 4 + 1)  # Adaptive chunk size

        corr_matrix = np.zeros((n_features, n_features))
        p_values = np.zeros((n_features, n_features))

        for i in range(0, n_features, chunk_size):
            end_i = min(i + chunk_size, n_features)
            for j in range(0, n_features, chunk_size):
                end_j = min(j + chunk_size, n_features)

                # Compute correlation for this chunk
                chunk_data_i = data[:, i:end_i]
                chunk_data_j = data[:, j:end_j]

                if i == j:
                    # Diagonal chunk - use full correlation
                    chunk_corr = np.corrcoef(chunk_data_i.T)
                    chunk_p = np.zeros_like(chunk_corr)
                else:
                    # Off-diagonal chunk - compute cross-correlations
                    chunk_corr = np.zeros((end_i - i, end_j - j))
                    chunk_p = np.zeros((end_i - i, end_j - j))

                    for ii in range(end_i - i):
                        for jj in range(end_j - j):
                            corr = np.corrcoef(chunk_data_i[:, ii], chunk_data_j[:, jj])[0, 1]
                            chunk_corr[ii, jj] = corr
                            # Simplified p-value calculation
                            n = len(chunk_data_i)
                            t_stat = abs(corr) * np.sqrt((n - 2) / (1 - corr**2))
                            chunk_p[ii, jj] = 2 * (1 - self._t_cdf(t_stat, n-2))

                corr_matrix[i:end_i, j:end_j] = chunk_corr
                p_values[i:end_i, j:end_j] = chunk_p

                # Symmetric matrix
                if i != j:
                    corr_matrix[j:end_j, i:end_i] = chunk_corr.T
                    p_values[j:end_j, i:end_i] = chunk_p.T

        return corr_matrix, p_values

    def _t_cdf(self, t: float, df: int) -> float:
        """Approximate t-distribution CDF for p-value calculation."""
        # Simplified approximation using normal distribution for large df
        if df > 30:
            from scipy.stats import norm
            return norm.cdf(t)
        else:
            # More accurate approximation for small df
            return 0.5 * (1 + np.sign(t) * np.sqrt(1 - np.exp(-2 * t**2 / np.pi)))

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'chunk_size_mb': self.chunk_size_mb,
            'gpu_enabled': self.enable_gpu,
            'parallel_enabled': self.enable_parallel,
            'utilities_available': UTILITIES_AVAILABLE,
            'matrix_ops_available': self.matrix_ops is not None,
            'gpu_manager_available': self.gpu_manager is not None,
            'memory_optimizer_available': self.memory_optimizer is not None,
            'cpu_optimizer_available': self.cpu_optimizer is not None
        }

# Convenience functions
def get_batch_matrix_processor(chunk_size_mb: int = 256,
                             enable_gpu: bool = True,
                             enable_parallel: bool = True) -> BatchMatrixProcessor:
    """Factory function to create batch matrix processor."""
    return BatchMatrixProcessor(
        chunk_size_mb=chunk_size_mb,
        enable_gpu=enable_gpu,
        enable_parallel=enable_parallel
    )

def batch_matrix_multiply(matrices_a: List['np.ndarray'], matrices_b: List['np.ndarray']) -> List['np.ndarray']:
    """Convenience function for batch matrix multiplication."""
    processor = get_batch_matrix_processor()
    return processor.batch_matrix_multiply(matrices_a, matrices_b)

def batch_feature_transformation(data: Union['np.ndarray', 'pd.DataFrame'],
                               transformations: List[Dict[str, Any]]) -> Union['np.ndarray', 'pd.DataFrame']:
    """Convenience function for batch feature transformation."""
    processor = get_batch_matrix_processor()
    return processor.batch_feature_transformation(data, transformations)

def batch_correlation_analysis(data: Union['np.ndarray', 'pd.DataFrame'],
                             method: str = 'pearson') -> Tuple['np.ndarray', 'np.ndarray']:
    """Convenience function for batch correlation analysis."""
    processor = get_batch_matrix_processor()
    return processor.batch_correlation_analysis(data, method)

# Alias for backwards compatibility
BatchMatrixOperations = BatchMatrixProcessor
