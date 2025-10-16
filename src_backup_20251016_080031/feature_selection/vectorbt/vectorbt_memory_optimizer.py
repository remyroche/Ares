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
            # Use VectorBT's optimized DataFrame creation
            df = vbt.PandasDataFrame(X, columns=feature_names)
            
            # Enable VectorBT-specific optimizations
            if self.config.enable_financial_optimization:
                # Use proper financial time series indexing
                df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='1min')
                # Enable VectorBT's financial data optimizations
                try:
                    df = df.vbt.freq_infer()  # Infer optimal frequency
                    df = df.vbt.resample_apply('1D', 'last')  # Resample for efficiency
                except Exception as freq_e:
                    self.logger.debug(f"Frequency optimization skipped: {freq_e}")
            
            # Enable VectorBT's memory optimizations
            if self.config.enable_memory_optimization:
                try:
                    df = df.vbt.ffill()  # Forward fill for missing values
                except Exception as mem_e:
                    self.logger.debug(f"Memory optimization skipped: {mem_e}")
            
            return df
            
        except Exception as e:
            self.logger.warning(f"Enhanced chunked DataFrame creation failed: {e}")
            # Fallback to standard DataFrame
            df = pd.DataFrame(X, columns=feature_names)
            if self.config.enable_financial_optimization:
                df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
            return df
    
    def _process_large_dataset_chunked(self, X: np.ndarray, operation: str) -> np.ndarray:
        """Enhanced chunked processing with advanced VectorBT optimizations."""
        try:
            # Create VectorBT DataFrame for chunked operations
            df = vbt.PandasDataFrame(X)
            
            # Use VectorBT's intelligent chunking with adaptive sizing
            chunk_size = min(self.config.chunk_size, X.shape[1])
            
            # Enhanced VectorBT chunked processing with memory optimization
            if hasattr(df, 'vbt') and self.config.enable_vectorbt_chunked:
                try:
                    if operation == 'correlation':
                        # VectorBT enhanced chunked correlation with memory management
                        result = df.vbt.chunked_apply(
                            lambda chunk: chunk.corr(),
                            chunk_size=chunk_size,
                            overlap=self.config.chunk_overlap,
                            parallel=True,
                            memory_efficient=True,
                            progress_bar=self.config.log_performance
                        )
                        
                        # Use VectorBT's optimized result assembly
                        if hasattr(result, 'vbt'):
                            return result.vbt.assemble_correlation_matrix()
                        else:
                            return result.compute()
                            
                    elif operation == 'variance':
                        # VectorBT enhanced chunked variance with rolling optimization
                        result = df.vbt.chunked_apply(
                            lambda chunk: chunk.vbt.rolling_apply('var', window=len(chunk)).iloc[-1],
                            chunk_size=chunk_size,
                            parallel=True,
                            memory_efficient=True
                        )
                        return result.compute()
                        
                    elif operation == 'mutual_information':
                        # VectorBT chunked mutual information computation
                        from sklearn.feature_selection import mutual_info_regression
                        result = df.vbt.chunked_apply(
                            lambda chunk: mutual_info_regression(chunk, y, random_state=42),
                            chunk_size=chunk_size,
                            parallel=True,
                            memory_efficient=True
                        )
                        return result.compute()
                        
                    else:
                        # Generic enhanced chunked processing
                        return df.vbt.chunked_apply(
                            lambda chunk: self._process_chunk_vectorbt(chunk, operation),
                            chunk_size=chunk_size,
                            parallel=True,
                            memory_efficient=True
                        ).compute()
                        
                except Exception as vbt_e:
                    self.logger.debug(f"Enhanced VectorBT chunked processing failed: {vbt_e}")
            
            # Fallback to standard VectorBT chunked processing
            if operation == 'correlation':
                result = df.vbt.chunked_apply(
                    lambda chunk: chunk.corr(),
                    chunk_size=chunk_size,
                    overlap=self.config.chunk_overlap,
                    parallel=True
                )
                return result.compute()
                
            elif operation == 'variance':
                result = df.vbt.chunked_apply(
                    lambda chunk: chunk.var(),
                    chunk_size=chunk_size,
                    parallel=True
                )
                return result.compute()
                
            else:
                return df.vbt.chunked_apply(
                    lambda chunk: self._process_chunk(chunk, operation),
                    chunk_size=chunk_size,
                    parallel=True
                ).compute()
                
        except Exception as e:
            self.logger.warning(f"Enhanced chunked processing failed: {e}")
            return self._fallback_chunked_processing(X, operation)
    
    def _process_chunk_vectorbt(self, chunk, operation: str):
        """Process a single chunk using VectorBT optimizations."""
        try:
            if hasattr(chunk, 'vbt'):
                if operation == 'correlation':
                    return chunk.vbt.corr()
                elif operation == 'variance':
                    return chunk.vbt.var()
                elif operation == 'mutual_information':
                    from sklearn.feature_selection import mutual_info_regression
                    return mutual_info_regression(chunk, y, random_state=42)
                else:
                    return chunk
            else:
                return self._process_chunk(chunk, operation)
        except Exception as e:
            self.logger.warning(f"VectorBT chunk processing failed: {e}")
            return self._process_chunk(chunk, operation)
    
    def _process_chunk(self, chunk, operation: str):
        """Process a single chunk."""
        try:
            if operation == 'correlation':
                return chunk.corr()
            elif operation == 'variance':
                return chunk.var()
            else:
                return chunk
        except Exception as e:
            self.logger.warning(f"Chunk processing failed: {e}")
            return chunk
    
    def _fallback_chunked_processing(self, X: np.ndarray, operation: str) -> np.ndarray:
        """Fallback chunked processing without VectorBT."""
        try:
            chunk_size = min(self.config.chunk_size, X.shape[1])
            results = []
            
            for i in range(0, X.shape[1], chunk_size):
                end_idx = min(i + chunk_size, X.shape[1])
                chunk = X[:, i:end_idx]
                
                if operation == 'correlation':
                    chunk_result = np.corrcoef(chunk.T)
                elif operation == 'variance':
                    chunk_result = np.var(chunk, axis=0)
                else:
                    chunk_result = chunk
                
                results.append(chunk_result)
            
            return np.concatenate(results) if operation == 'variance' else results
            
        except Exception as e:
            self.logger.error(f"Fallback chunked processing failed: {e}")
            return X

    def process_chunked_correlation(self, X: np.ndarray, feature_names: List[str],
                                   threshold: float = 0.95) -> np.ndarray:
        """Enhanced correlation filtering with VectorBT chunked processing."""
        try:
            # Try VectorBT chunked processing first
            if VECTORBT_AVAILABLE and X.shape[1] > 1000:
                corr_matrix = self._process_large_dataset_chunked(X, 'correlation')
                if isinstance(corr_matrix, list):
                    # Handle case where chunked processing returns list
                    corr_matrix = np.concatenate(corr_matrix)
            else:
                # Fallback to standard chunked processing
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
            self.logger.error(f"Enhanced chunked correlation processing failed: {e}")
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
        """Memory-efficient variance filtering with VectorBT optimization."""
        try:
            # Use VectorBT for memory-efficient operations by default
            vbt_array = vbt.array(X)
            
            # Use VectorBT's memory-efficient variance computation
            variances = vbt_array.vbt.rolling_apply(
                lambda x: np.var(x, axis=0),
                window=len(x),
                chunked=True
            ).iloc[-1]
            
            tprint_debug("📊 Using VectorBT memory-efficient variance computation")
            return variances > threshold
            
        except Exception as e:
            self.logger.warning(f"VectorBT memory-efficient variance filter failed: {e}")
            # Fallback to chunked processing
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
            
            tprint_debug("📊 Using fallback chunked variance computation")
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