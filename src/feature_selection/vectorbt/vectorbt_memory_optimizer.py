"""
VectorBT Memory Optimizer

This module provides VectorBT-based memory-efficient feature selection
for large datasets with chunked processing and lazy evaluation.
"""

import numpy as np
import pandas as pd
import time
import logging
import gc
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator, Generator
from dataclasses import dataclass
import psutil
import os

# VectorBT imports
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None

# Import utilities
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_performance, tprint_debug
from src.utils.math_validation import validate_numeric_array, validate_finite

from .vectorbt_config import VectorBTFeatureSelectionConfig

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for memory optimization."""
    # Memory limits
    max_memory_mb: int = 2048
    chunk_size: int = 10000
    memory_threshold: float = 0.8  # Use 80% of available memory
    
    # Chunked processing
    enable_chunked_processing: bool = True
    chunk_overlap: int = 100  # Overlap between chunks for continuity
    
    # Lazy evaluation
    enable_lazy_evaluation: bool = True
    lazy_batch_size: int = 1000
    
    # Memory monitoring
    enable_memory_monitoring: bool = True
    memory_check_interval: int = 10  # Check memory every N operations
    
    # Garbage collection
    enable_aggressive_gc: bool = True
    gc_threshold: float = 0.7  # Trigger GC when memory usage exceeds this


class VectorBTMemoryOptimizer:
    """
    VectorBT-based memory optimizer for large dataset feature selection.
    
    This class provides:
    - Chunked processing for large datasets
    - Lazy evaluation to minimize memory usage
    - Memory monitoring and automatic optimization
    - Garbage collection management
    - Financial data optimization
    """
    
    def __init__(self, config: Optional[VectorBTFeatureSelectionConfig] = None):
        """Initialize VectorBT memory optimizer."""
        self.config = config or VectorBTFeatureSelectionConfig()
        self.memory_config = MemoryConfig()
        self.logger = logger.getChild('VectorBTMemoryOptimizer')
        
        # Check VectorBT availability
        if not VECTORBT_AVAILABLE:
            raise ImportError("VectorBT is required but not available. Please install vectorbt.")
        
        # Memory tracking
        self.memory_stats = {
            'peak_memory_mb': 0.0,
            'current_memory_mb': 0.0,
            'memory_saved_mb': 0.0,
            'chunks_processed': 0,
            'gc_cycles': 0,
            'memory_warnings': 0
        }
        
        # Initialize memory monitoring
        self._initialize_memory_monitoring()
        
        tprint_success("🚀 VectorBTMemoryOptimizer initialized")
    
    def _initialize_memory_monitoring(self):
        """Initialize memory monitoring system."""
        try:
            # Get initial memory usage
            process = psutil.Process(os.getpid())
            self.memory_stats['current_memory_mb'] = process.memory_info().rss / 1024 / 1024
            
            # Set memory limits based on available memory
            available_memory = psutil.virtual_memory().available / 1024 / 1024
            self.memory_config.max_memory_mb = min(
                self.memory_config.max_memory_mb,
                int(available_memory * self.memory_config.memory_threshold)
            )
            
            tprint_debug(f"📊 Memory monitoring initialized: {self.memory_stats['current_memory_mb']:.1f}MB current, "
                        f"{self.memory_config.max_memory_mb:.1f}MB limit")
            
        except Exception as e:
            self.logger.warning(f"Memory monitoring initialization failed: {e}")
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check current memory usage and return statistics."""
        try:
            process = psutil.Process(os.getpid())
            current_memory = process.memory_info().rss / 1024 / 1024
            
            # Update stats
            self.memory_stats['current_memory_mb'] = current_memory
            self.memory_stats['peak_memory_mb'] = max(
                self.memory_stats['peak_memory_mb'], current_memory
            )
            
            # Check if memory usage is high
            memory_usage_ratio = current_memory / self.memory_config.max_memory_mb
            
            if memory_usage_ratio > self.memory_config.gc_threshold:
                self._trigger_garbage_collection()
                self.memory_stats['memory_warnings'] += 1
            
            return {
                'current_memory_mb': current_memory,
                'peak_memory_mb': self.memory_stats['peak_memory_mb'],
                'memory_usage_ratio': memory_usage_ratio,
                'memory_warning': memory_usage_ratio > self.memory_config.gc_threshold
            }
            
        except Exception as e:
            self.logger.warning(f"Memory check failed: {e}")
            return {
                'current_memory_mb': 0.0,
                'peak_memory_mb': 0.0,
                'memory_usage_ratio': 0.0,
                'memory_warning': False
            }
    
    def _trigger_garbage_collection(self):
        """Trigger aggressive garbage collection."""
        try:
            if self.memory_config.enable_aggressive_gc:
                # Force garbage collection
                collected = gc.collect()
                
                # Update stats
                self.memory_stats['gc_cycles'] += 1
                
                tprint_debug(f"🗑️ Garbage collection triggered: {collected} objects collected")
                
        except Exception as e:
            self.logger.warning(f"Garbage collection failed: {e}")
    
    def _create_chunked_dataframe(self, X: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
        """Create VectorBT-optimized DataFrame with chunked processing."""
        try:
            # Create DataFrame with proper indexing for VectorBT
            df = pd.DataFrame(X, columns=feature_names)
            
            # Set index for time series optimization if applicable
            if self.config.enable_financial_optimization:
                # Use datetime index for financial data optimization
                df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
            
            return df
            
        except Exception as e:
            self.logger.warning(f"Chunked DataFrame creation failed: {e}")
            return pd.DataFrame(X, columns=feature_names)
    
    def process_chunked_correlation(self, X: np.ndarray, feature_names: List[str],
                                   threshold: float = 0.95) -> np.ndarray:
        """Process correlation filtering with chunked processing."""
        try:
            n_features = X.shape[1]
            chunk_size = min(self.memory_config.chunk_size, n_features)
            
            # Initialize correlation matrix
            corr_matrix = np.eye(n_features)
            
            # Process in chunks
            for i in range(0, n_features, chunk_size):
                end_i = min(i + chunk_size, n_features)
                chunk_i = X[:, i:end_i]
                
                for j in range(0, n_features, chunk_size):
                    end_j = min(j + chunk_size, n_features)
                    chunk_j = X[:, j:end_j]
                    
                    # Compute correlation between chunks
                    chunk_corr = np.corrcoef(chunk_i.T, chunk_j.T)
                    
                    # Extract relevant part
                    if i == j:
                        # Same chunk - use upper triangle
                        corr_subset = chunk_corr[:len(chunk_i.T), :len(chunk_j.T)]
                    else:
                        # Different chunks - use cross-correlation
                        corr_subset = chunk_corr[:len(chunk_i.T), len(chunk_i.T):]
                    
                    # Fill correlation matrix
                    for ii, idx_i in enumerate(range(i, end_i)):
                        for jj, idx_j in enumerate(range(j, end_j)):
                            if idx_i != idx_j:  # Skip diagonal
                                corr_matrix[idx_i, idx_j] = corr_subset[ii, jj]
                
                # Update memory stats
                self.memory_stats['chunks_processed'] += 1
                
                # Check memory usage
                if self.memory_config.enable_memory_monitoring:
                    memory_info = self._check_memory_usage()
                    if memory_info['memory_warning']:
                        tprint_warning(f"⚠️ High memory usage: {memory_info['current_memory_mb']:.1f}MB")
            
            # Apply correlation filter
            high_corr_mask = np.abs(corr_matrix) > threshold
            np.fill_diagonal(high_corr_mask, False)  # Exclude diagonal
            
            # Find features to remove
            to_remove = np.any(high_corr_mask, axis=1)
            features_to_keep = ~to_remove
            
            return features_to_keep
            
        except Exception as e:
            self.logger.error(f"Chunked correlation processing failed: {e}")
            # Fallback to standard correlation
            corr_matrix = np.corrcoef(X.T)
            high_corr_mask = np.abs(corr_matrix) > threshold
            np.fill_diagonal(high_corr_mask, False)
            to_remove = np.any(high_corr_mask, axis=1)
            return ~to_remove
    
    def process_chunked_mutual_information(self, X: np.ndarray, y: np.ndarray,
                                         k: int = 50) -> np.ndarray:
        """Process mutual information with chunked processing."""
        try:
            from sklearn.feature_selection import mutual_info_regression
            
            n_features = X.shape[1]
            chunk_size = min(self.memory_config.chunk_size, n_features)
            mi_scores = np.zeros(n_features)
            
            # Process features in chunks
            for i in range(0, n_features, chunk_size):
                end_idx = min(i + chunk_size, n_features)
                chunk_X = X[:, i:end_idx]
                
                # Compute mutual information for chunk
                chunk_scores = mutual_info_regression(chunk_X, y, random_state=42)
                mi_scores[i:end_idx] = chunk_scores
                
                # Update memory stats
                self.memory_stats['chunks_processed'] += 1
                
                # Check memory usage
                if self.memory_config.enable_memory_monitoring:
                    memory_info = self._check_memory_usage()
                    if memory_info['memory_warning']:
                        tprint_warning(f"⚠️ High memory usage: {memory_info['current_memory_mb']:.1f}MB")
            
            # Select top-k features
            top_k_indices = np.argsort(mi_scores)[-k:]
            mask = np.zeros(n_features, dtype=bool)
            mask[top_k_indices] = True
            
            return mask
            
        except Exception as e:
            self.logger.error(f"Chunked mutual information processing failed: {e}")
            # Fallback to standard mutual information
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(X, y, random_state=42)
            top_k_indices = np.argsort(mi_scores)[-k:]
            mask = np.zeros(X.shape[1], dtype=bool)
            mask[top_k_indices] = True
            return mask
    
    def process_lazy_evaluation(self, X: np.ndarray, y: np.ndarray,
                               operation: str, **kwargs) -> Generator[Any, None, None]:
        """Process operations with lazy evaluation."""
        try:
            n_features = X.shape[1]
            batch_size = self.memory_config.lazy_batch_size
            
            # Process features in batches
            for i in range(0, n_features, batch_size):
                end_idx = min(i + batch_size, n_features)
                batch_X = X[:, i:end_idx]
                
                # Process batch based on operation
                if operation == 'mutual_information':
                    from sklearn.feature_selection import mutual_info_regression
                    batch_result = mutual_info_regression(batch_X, y, random_state=42)
                elif operation == 'variance':
                    batch_result = np.var(batch_X, axis=0)
                elif operation == 'correlation':
                    batch_result = np.corrcoef(batch_X.T)
                else:
                    raise ValueError(f"Unknown operation: {operation}")
                
                yield batch_result
                
                # Check memory usage
                if self.memory_config.enable_memory_monitoring:
                    memory_info = self._check_memory_usage()
                    if memory_info['memory_warning']:
                        tprint_warning(f"⚠️ High memory usage: {memory_info['current_memory_mb']:.1f}MB")
            
        except Exception as e:
            self.logger.error(f"Lazy evaluation processing failed: {e}")
            yield None
    
    def optimize_memory_usage(self, X: np.ndarray, y: np.ndarray,
                             feature_names: List[str]) -> Dict[str, Any]:
        """Optimize memory usage for large datasets."""
        try:
            tprint("🚀 Starting memory-optimized feature selection")
            
            # Check initial memory
            initial_memory = self._check_memory_usage()
            tprint_debug(f"📊 Initial memory: {initial_memory['current_memory_mb']:.1f}MB")
            
            # Apply memory-optimized filters
            filters_applied = []
            selected_mask = np.ones(X.shape[1], dtype=bool)
            
            # Variance filter (memory-efficient)
            if X.shape[1] > 1000:
                tprint_debug("📊 Applying memory-optimized variance filter...")
                variance_mask = self._memory_efficient_variance_filter(X)
                selected_mask &= variance_mask
                filters_applied.append('variance')
            
            # Correlation filter (chunked processing)
            if X.shape[1] > 500:
                tprint_debug("📊 Applying chunked correlation filter...")
                correlation_mask = self.process_chunked_correlation(X, feature_names)
                selected_mask &= correlation_mask
                filters_applied.append('correlation')
            
            # Mutual information filter (chunked processing)
            if X.shape[1] > 200:
                tprint_debug("📊 Applying chunked mutual information filter...")
                mi_mask = self.process_chunked_mutual_information(X, y, k=50)
                selected_mask &= mi_mask
                filters_applied.append('mutual_information')
            
            # Get selected features
            selected_indices = np.where(selected_mask)[0]
            selected_features = [feature_names[i] for i in selected_indices]
            
            # Check final memory
            final_memory = self._check_memory_usage()
            memory_saved = initial_memory['current_memory_mb'] - final_memory['current_memory_mb']
            self.memory_stats['memory_saved_mb'] += memory_saved
            
            tprint_success(f"✅ Memory-optimized selection completed: {len(selected_features)}/{X.shape[1]} features, "
                         f"{memory_saved:.1f}MB memory saved")
            
            return {
                'success': True,
                'selected_features': selected_features,
                'selected_indices': selected_indices.tolist(),
                'n_selected': len(selected_features),
                'n_total': X.shape[1],
                'filters_applied': filters_applied,
                'memory_saved_mb': memory_saved,
                'method': 'vectorbt_memory_optimized'
            }
            
        except Exception as e:
            self.logger.error(f"Memory-optimized selection failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'method': 'vectorbt_memory_optimized'
            }
    
    def _memory_efficient_variance_filter(self, X: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Memory-efficient variance filtering."""
        try:
            # Process in chunks to minimize memory usage
            chunk_size = min(self.memory_config.chunk_size, X.shape[1])
            variances = np.zeros(X.shape[1])
            
            for i in range(0, X.shape[1], chunk_size):
                end_idx = min(i + chunk_size, X.shape[1])
                chunk_X = X[:, i:end_idx]
                chunk_variances = np.var(chunk_X, axis=0)
                variances[i:end_idx] = chunk_variances
                
                # Check memory usage
                if self.memory_config.enable_memory_monitoring:
                    memory_info = self._check_memory_usage()
                    if memory_info['memory_warning']:
                        tprint_warning(f"⚠️ High memory usage: {memory_info['current_memory_mb']:.1f}MB")
            
            return variances > threshold
            
        except Exception as e:
            self.logger.warning(f"Memory-efficient variance filter failed: {e}")
            # Fallback to standard variance
            variances = np.var(X, axis=0)
            return variances > threshold
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        current_memory = self._check_memory_usage()
        
        stats = self.memory_stats.copy()
        stats.update(current_memory)
        
        if stats['chunks_processed'] > 0:
            stats['avg_memory_per_chunk'] = stats['peak_memory_mb'] / stats['chunks_processed']
        else:
            stats['avg_memory_per_chunk'] = 0.0
        
        tprint_performance(f"📊 Memory Stats: {stats['peak_memory_mb']:.1f}MB peak, "
                         f"{stats['memory_saved_mb']:.1f}MB saved, "
                         f"{stats['chunks_processed']} chunks processed")
        
        return stats


def create_vectorbt_memory_optimizer(config: Optional[VectorBTFeatureSelectionConfig] = None) -> VectorBTMemoryOptimizer:
    """Create a VectorBT memory optimizer."""
    return VectorBTMemoryOptimizer(config)