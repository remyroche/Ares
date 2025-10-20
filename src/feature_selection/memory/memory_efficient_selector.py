"""
Memory-Efficient Feature Selection

This module provides memory-efficient feature selection operations using
hardware optimization tools for handling large datasets.
"""

import logging
import gc
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse

# Import hardware optimization tools
from src.utils.hardware import (
    get_integrated_hardware_manager,
    memory_efficient,
    performance_tracked,
    smart_cache,
    auto_optimize,
    WorkloadType,
    OptimizationLevel
)
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_performance, tprint_debug

logger = logging.getLogger(__name__)

@dataclass
class MemoryConfig:
    """Configuration for memory-efficient feature selection."""
    # Memory limits
    memory_limit_gb: float = 8.0
    chunk_size: int = 10000
    max_memory_usage: float = 0.8  # 80% of available memory

    # Optimization settings
    enable_memory_monitoring: bool = True
    enable_garbage_collection: bool = True
    enable_compression: bool = True

    # Chunked processing
    enable_chunked_processing: bool = True
    adaptive_chunk_size: bool = True
    min_chunk_size: int = 1000
    max_chunk_size: int = 50000

    # Sparse matrix support
    enable_sparse_support: bool = True
    sparse_threshold: float = 0.1  # Use sparse if >10% zeros

class MemoryEfficientFeatureSelector:
    """Memory-efficient feature selector with hardware optimization."""

    def __init__(self, config: Optional[MemoryConfig] = None):
        """Initialize memory-efficient feature selector."""
        self.config = config or MemoryConfig()
        self.logger = logger.getChild('MemoryEfficientFeatureSelector')

        # Initialize hardware tools
        if self.config.enable_memory_monitoring:
            self.hardware_manager = get_integrated_hardware_manager()
        else:
            self.hardware_manager = None

        # Memory tracking
        self.memory_stats = {
            'peak_usage_mb': 0,
            'current_usage_mb': 0,
            'chunks_processed': 0,
            'memory_optimizations': 0
        }

        tprint_success("🧠 MemoryEfficientFeatureSelector initialized")

    def _check_memory_usage(self) -> float:
        """Check current memory usage."""
        try:
            if self.hardware_manager:
                memory_report = self.hardware_manager.get_memory_report()
                return memory_report.get('total_memory_usage_mb', 0) / (1024 * 1024)  # Convert to ratio
            else:
                import psutil
                return psutil.virtual_memory().percent / 100.0
        except Exception as mem_e:
            tprint_debug(f"⚠️ Memory check failed: {mem_e}")
            return 0.5  # Default to 50% if can't check

    def _optimize_memory_if_needed(self) -> bool:
        """Optimize memory if usage is too high."""
        memory_pressure = self._check_memory_usage()

        if memory_pressure > self.config.max_memory_usage:
            tprint_warning(f"⚠️ High memory usage: {memory_pressure:.1%}")

            if self.config.enable_garbage_collection:
                # Force garbage collection
                collected = gc.collect()
                tprint_debug(f"🗑️ Garbage collected {collected} objects")

            if self.hardware_manager:
                # Apply memory optimizations
                self.hardware_manager.clear_all_caches()
                self.memory_stats['memory_optimizations'] += 1
                tprint_success("🧠 Memory optimized")
                return True

            return False
        return True

    def _get_optimal_chunk_size(self, data_size: int) -> int:
        """Calculate optimal chunk size based on data size and memory."""
        if not self.config.adaptive_chunk_size:
            return self.config.chunk_size

        # Base chunk size
        base_chunk = self.config.chunk_size

        # Adjust based on data size
        if data_size < 10000:
            chunk_size = min(base_chunk, data_size)
        elif data_size < 100000:
            chunk_size = base_chunk
        else:
            # For very large datasets, use smaller chunks
            chunk_size = max(self.config.min_chunk_size, base_chunk // 2)

        # Adjust based on memory pressure
        memory_pressure = self._check_memory_usage()
        if memory_pressure > 0.7:
            chunk_size = max(self.config.min_chunk_size, chunk_size // 2)

        return min(chunk_size, self.config.max_chunk_size)

    def _process_chunk(self, X_chunk: np.ndarray, y_chunk: np.ndarray,
                      method: str, **kwargs) -> Dict[str, Any]:
        """Process a single chunk of data."""
        try:
            # Import here to avoid circular imports
            from src.feature_selection import select_features

            # Process chunk
            result = select_features(X_chunk, y_chunk, method=method, **kwargs)

            # Update memory stats
            self.memory_stats['chunks_processed'] += 1

            return result

        except Exception as e:
            self.logger.warning(f"Chunk processing failed: {e}")
            return {'success': False, 'error': str(e)}

    @memory_efficient(memory_threshold_mb=200.0, auto_cleanup=True)
    @performance_tracked(log_performance=True, track_memory=True)
    def select_features_chunked(self, X: np.ndarray, y: np.ndarray,
                               method: str = 'comprehensive', **kwargs) -> Dict[str, Any]:
        """Select features using chunked processing for memory efficiency."""
        tprint_info(f"🧠 Starting chunked feature selection: {X.shape}")

        # Check if chunked processing is needed
        if not self.config.enable_chunked_processing or X.shape[0] <= self.config.chunk_size:
            # Process normally
            from src.feature_selection import select_features
            return select_features(X, y, method=method, **kwargs)

        # Calculate optimal chunk size
        chunk_size = self._get_optimal_chunk_size(X.shape[0])
        tprint_debug(f"📦 Using chunk size: {chunk_size}")

        # Process in chunks
        chunk_results = []
        n_chunks = (X.shape[0] + chunk_size - 1) // chunk_size

        for i in range(0, X.shape[0], chunk_size):
            chunk_idx = i // chunk_size + 1
            end_idx = min(i + chunk_size, X.shape[0])

            tprint_debug(f"📦 Processing chunk {chunk_idx}/{n_chunks}: rows {i}-{end_idx}")

            # Get chunk
            X_chunk = X[i:end_idx]
            y_chunk = y[i:end_idx]

            # Process chunk
            chunk_result = self._process_chunk(X_chunk, y_chunk, method, **kwargs)
            chunk_results.append(chunk_result)

            # Memory optimization
            if not self._optimize_memory_if_needed():
                tprint_warning("⚠️ Memory optimization failed, continuing...")

            # Update progress
            if chunk_idx % 10 == 0:
                tprint_performance(f"📊 Processed {chunk_idx}/{n_chunks} chunks")

        # Combine results
        combined_result = self._combine_chunk_results(chunk_results)

        tprint_success(f"✅ Chunked processing completed: {len(combined_result.get('selected_features', []))} features")
        return combined_result

    def _combine_chunk_results(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from multiple chunks."""
        try:
            # Filter successful results
            successful_results = [r for r in chunk_results if r.get('success', False)]

            if not successful_results:
                return {'success': False, 'error': 'All chunks failed'}

            # Combine selected features (intersection of all chunks)
            all_selected = [set(r.get('selected_features', [])) for r in successful_results]
            if not all_selected:
                return {'success': False, 'error': 'No features selected'}

            # Find common features across chunks
            common_features = set.intersection(*all_selected)

            # If no common features, use union
            if not common_features:
                common_features = set.union(*all_selected)
                tprint_warning("⚠️ No common features across chunks, using union")

            # Calculate average scores if available
            feature_scores = {}
            if 'feature_scores' in successful_results[0]:
                all_scores = [r.get('feature_scores', {}) for r in successful_results]
                for feature in common_features:
                    scores = [s.get(feature, 0) for s in all_scores if feature in s]
                    if scores:
                        feature_scores[feature] = np.mean(scores)

            return {
                'success': True,
                'selected_features': list(common_features),
                'feature_scores': feature_scores,
                'n_chunks_processed': len(successful_results),
                'total_chunks': len(chunk_results),
                'method': 'chunked'
            }

        except Exception as e:
            self.logger.error(f"Error combining chunk results: {e}")
            return {'success': False, 'error': str(e)}

class ChunkedFeatureProcessor:
    """Processor for chunked feature selection operations."""

    def __init__(self, memory_config: Optional[MemoryConfig] = None):
        """Initialize chunked processor."""
        self.memory_config = memory_config or MemoryConfig()
        self.processor = MemoryEfficientFeatureSelector(self.memory_config)

    def process_large_dataset(self, X: np.ndarray, y: np.ndarray,
                            processor_func: Callable, **kwargs) -> Dict[str, Any]:
        """Process large dataset using chunked processing."""
        return self.processor.select_features_chunked(X, y, **kwargs)

class SparseFeatureSelector:
    """Feature selector optimized for sparse matrices."""

    def __init__(self, sparse_threshold: float = 0.1):
        """Initialize sparse feature selector."""
        self.sparse_threshold = sparse_threshold
        self.logger = logger.getChild('SparseFeatureSelector')

        tprint_success("📊 SparseFeatureSelector initialized")

    def _should_use_sparse(self, X: np.ndarray) -> bool:
        """Determine if data should be treated as sparse."""
        if issparse(X):
            return True

        # Check sparsity
        zero_ratio = np.count_nonzero(X == 0) / X.size
        return zero_ratio > self.sparse_threshold

    @memory_efficient(memory_threshold_mb=100.0, auto_cleanup=True)
    @performance_tracked(log_performance=True, track_memory=True)
    def select_features_sparse(self, X: Union[np.ndarray, csr_matrix],
                              y: np.ndarray, method: str = 'comprehensive',
                              **kwargs) -> Dict[str, Any]:
        """Select features using sparse matrix operations."""
        tprint_info(f"📊 Sparse feature selection: {X.shape}")

        # Convert to sparse if needed
        if not issparse(X) and self._should_use_sparse(X):
            X = csr_matrix(X)
            tprint_debug("📊 Converted to sparse matrix")

        if issparse(X):
            # Use sparse-aware selection
            return self._sparse_selection(X, y, method, **kwargs)
        else:
            # Use regular selection
            from src.feature_selection import select_features
            return select_features(X, y, method=method, **kwargs)

    def _sparse_selection(self, X: csr_matrix, y: np.ndarray,
                         method: str, **kwargs) -> Dict[str, Any]:
        """Perform feature selection on sparse matrix."""
        try:
            # For sparse matrices, we need to convert to dense for most sklearn methods
            # This is a limitation of current sklearn implementations
            tprint_warning("⚠️ Converting sparse matrix to dense for selection")

            X_dense = X.toarray()

            # Use regular selection on dense matrix
            from src.feature_selection import select_features
            result = select_features(X_dense, y, method=method, **kwargs)

            # Add sparse information to result
            result['sparse_original'] = True
            result['sparsity_ratio'] = 1.0 - (X.nnz / X.size)

            return result

        except Exception as e:
            self.logger.error(f"Sparse selection failed: {e}")
            return {'success': False, 'error': str(e)}

def create_memory_efficient_selector(config: Optional[MemoryConfig] = None) -> MemoryEfficientFeatureSelector:
    """Create a memory-efficient feature selector."""
    return MemoryEfficientFeatureSelector(config)
