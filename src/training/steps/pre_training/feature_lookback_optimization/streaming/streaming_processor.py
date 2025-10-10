#!/usr/bin/env python3
"""
Streaming processor for very large datasets with memory-efficient chunking.
"""

import numpy as np
import pandas as pd
import gc
import psutil
from typing import Iterator, Tuple, Dict, Any, Optional, List, Union
from dataclasses import dataclass
import logging

@dataclass
class StreamingConfig:
    """Configuration for streaming processing."""
    chunk_size: int = 10000  # Number of rows per chunk
    memory_limit_mb: int = 1024  # Memory limit in MB
    overlap_size: int = 100  # Overlap between chunks for continuity
    enable_gc: bool = True  # Enable garbage collection between chunks
    progress_interval: int = 10  # Log progress every N chunks

class StreamingProcessor:
    """
    Memory-efficient streaming processor for very large datasets.
    
    Features:
    - Chunked processing with configurable chunk sizes
    - Memory monitoring and automatic garbage collection
    - Overlap handling for continuous features
    - Progress tracking and logging
    - Automatic memory management
    """
    
    def __init__(self, config: Optional[StreamingConfig] = None, logger: Optional[logging.Logger] = None):
        """Initialize streaming processor."""
        self.config = config or StreamingConfig()
        self.logger = logger or logging.getLogger('StreamingProcessor')
        self.memory_monitor = MemoryMonitor()
        
    def process_large_dataset(
        self,
        data: pd.DataFrame,
        feature_names: List[str],
        target_column: str,
        lookback_range: Tuple[int, int],
        optimization_method: str = "coarse_to_refine",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process very large dataset using streaming approach.
        
        Args:
            data: Input DataFrame
            feature_names: List of feature names to optimize
            target_column: Target column name
            lookback_range: (min_lookback, max_lookback) tuple
            optimization_method: Optimization method to use
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with optimization results
        """
        total_rows = len(data)
        self.logger.info(f"🚀 Starting streaming processing for {total_rows:,} rows")
        self.logger.info(f"   → Chunk size: {self.config.chunk_size:,}")
        self.logger.info(f"   → Memory limit: {self.config.memory_limit_mb} MB")
        self.logger.info(f"   → Features: {len(feature_names)}")
        
        # Calculate number of chunks needed
        num_chunks = (total_rows + self.config.chunk_size - 1) // self.config.chunk_size
        self.logger.info(f"   → Processing in {num_chunks} chunks")
        
        # Initialize results storage
        all_results = {}
        chunk_results = []
        
        try:
            # Process data in chunks
            for chunk_idx, chunk_data in enumerate(self._create_data_chunks(data)):
                self.logger.info(f"📦 Processing chunk {chunk_idx + 1}/{num_chunks}")
                
                # Process this chunk
                chunk_result = self._process_chunk(
                    chunk_data,
                    feature_names,
                    target_column,
                    lookback_range,
                    optimization_method,
                    chunk_idx,
                    **kwargs
                )
                
                chunk_results.append(chunk_result)
                
                # Memory management
                if self.config.enable_gc:
                    self._manage_memory()
                
                # Progress logging
                if (chunk_idx + 1) % self.config.progress_interval == 0:
                    self._log_progress(chunk_idx + 1, num_chunks)
            
            # Merge results from all chunks
            all_results = self._merge_chunk_results(chunk_results, feature_names)
            
            self.logger.info("✅ Streaming processing completed successfully")
            return all_results
            
        except Exception as e:
            self.logger.error(f"❌ Streaming processing failed: {e}")
            raise
    
    def _create_data_chunks(self, data: pd.DataFrame) -> Iterator[pd.DataFrame]:
        """Create overlapping data chunks for processing with memory optimization."""
        total_rows = len(data)
        chunk_size = self.config.chunk_size
        overlap = self.config.overlap_size
        
        for start_idx in range(0, total_rows, chunk_size - overlap):
            end_idx = min(start_idx + chunk_size, total_rows)
            
            # Create chunk with overlap using memory-efficient operations
            if start_idx == 0:
                # First chunk: no overlap at start
                chunk = data.iloc[start_idx:end_idx].copy()
            else:
                # Subsequent chunks: include overlap
                chunk_start = max(0, start_idx - overlap)
                chunk = data.iloc[chunk_start:end_idx].copy()
            
            # Optimize chunk memory usage
            chunk = self._optimize_chunk_memory(chunk)
            
            # Add chunk metadata
            chunk.attrs['chunk_start'] = start_idx
            chunk.attrs['chunk_end'] = end_idx
            chunk.attrs['overlap_start'] = max(0, start_idx - overlap)
            
            yield chunk
    
    def _optimize_chunk_memory(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Optimize memory usage for a data chunk."""
        try:
            # Convert object columns to category if they have low cardinality
            for col in chunk.select_dtypes(include=['object']).columns:
                if chunk[col].nunique() / len(chunk) < 0.5:
                    chunk[col] = chunk[col].astype('category')
            
            # Convert float64 to float32 if precision allows
            for col in chunk.select_dtypes(include=['float64']).columns:
                if chunk[col].min() >= np.finfo(np.float32).min and chunk[col].max() <= np.finfo(np.float32).max:
                    chunk[col] = chunk[col].astype('float32')
            
            # Convert int64 to int32 if range allows
            for col in chunk.select_dtypes(include=['int64']).columns:
                if chunk[col].min() >= np.iinfo(np.int32).min and chunk[col].max() <= np.iinfo(np.int32).max:
                    chunk[col] = chunk[col].astype('int32')
            
            return chunk
            
        except Exception as e:
            self.logger.warning(f"⚠️ Chunk memory optimization failed: {e}")
            return chunk
    
    def _process_chunk(
        self,
        chunk_data: pd.DataFrame,
        feature_names: List[str],
        target_column: str,
        lookback_range: Tuple[int, int],
        optimization_method: str,
        chunk_idx: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Process a single chunk of data."""
        try:
            # Import here to avoid circular imports
            from src.training.steps.pre_training.feature_lookback_optimization.core.optimizer import CoreOptimizer
            
            # Create optimizer instance for this chunk
            optimizer = CoreOptimizer()
            
            # Process features in this chunk
            chunk_results = {}
            for feature_name in feature_names:
                try:
                    # Optimize single feature for this chunk
                    result = optimizer._optimize_single_feature(
                        chunk_data,
                        feature_name,
                        target_column,
                        lookback_range,
                        optimization_method,
                        **kwargs
                    )
                    
                    chunk_results[feature_name] = result
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Feature {feature_name} failed in chunk {chunk_idx}: {e}")
                    chunk_results[feature_name] = None
            
            return {
                'chunk_idx': chunk_idx,
                'chunk_start': chunk_data.attrs.get('chunk_start', 0),
                'chunk_end': chunk_data.attrs.get('chunk_end', len(chunk_data)),
                'results': chunk_results
            }
            
        except Exception as e:
            self.logger.error(f"❌ Chunk {chunk_idx} processing failed: {e}")
            return {
                'chunk_idx': chunk_idx,
                'chunk_start': chunk_data.attrs.get('chunk_start', 0),
                'chunk_end': chunk_data.attrs.get('chunk_end', len(chunk_data)),
                'results': {},
                'error': str(e)
            }
    
    def _merge_chunk_results(
        self,
        chunk_results: List[Dict[str, Any]],
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Merge results from all chunks into final results."""
        self.logger.info("🔄 Merging chunk results...")
        
        merged_results = {}
        
        for feature_name in feature_names:
            feature_results = []
            
            # Collect results for this feature from all chunks
            for chunk_result in chunk_results:
                if 'results' in chunk_result and feature_name in chunk_result['results']:
                    feature_result = chunk_result['results'][feature_name]
                    if feature_result is not None:
                        feature_results.append(feature_result)
            
            if feature_results:
                # Merge results for this feature
                merged_result = self._merge_feature_results(feature_results, feature_name)
                merged_results[feature_name] = merged_result
            else:
                self.logger.warning(f"⚠️ No valid results for feature {feature_name}")
                merged_results[feature_name] = None
        
        self.logger.info(f"✅ Merged results for {len(merged_results)} features")
        return merged_results
    
    def _merge_feature_results(
        self,
        feature_results: List[Any],
        feature_name: str
    ) -> Dict[str, Any]:
        """Merge results for a single feature across chunks."""
        if not feature_results:
            return None
        
        # For now, use the result from the chunk with the best score
        # In a more sophisticated implementation, we could:
        # - Weight results by chunk size
        # - Use ensemble methods
        # - Apply statistical aggregation
        
        best_result = max(feature_results, key=lambda x: x.best_score if hasattr(x, 'best_score') else 0)
        
        return {
            'best_lookback_period': best_result.best_lookback_period,
            'best_score': best_result.best_score,
            'optimization_method': best_result.optimization_method,
            'total_trials': sum(getattr(r, 'total_trials', 0) for r in feature_results),
            'optimization_time': sum(getattr(r, 'optimization_time', 0) for r in feature_results),
            'convergence_achieved': any(getattr(r, 'convergence_achieved', False) for r in feature_results),
            'chunk_count': len(feature_results)
        }
    
    def _manage_memory(self):
        """Manage memory usage and perform garbage collection."""
        memory_usage = self.memory_monitor.get_memory_usage()
        
        if memory_usage > self.config.memory_limit_mb:
            self.logger.warning(f"⚠️ Memory usage {memory_usage:.1f} MB exceeds limit {self.config.memory_limit_mb} MB")
            
            # Force garbage collection
            gc.collect()
            
            # Check memory after GC
            memory_after_gc = self.memory_monitor.get_memory_usage()
            self.logger.info(f"   → Memory after GC: {memory_after_gc:.1f} MB")
    
    def _log_progress(self, current_chunk: int, total_chunks: int):
        """Log processing progress."""
        progress_pct = (current_chunk / total_chunks) * 100
        memory_usage = self.memory_monitor.get_memory_usage()
        
        self.logger.info(
            f"📊 Progress: {current_chunk}/{total_chunks} chunks ({progress_pct:.1f}%) - "
            f"Memory: {memory_usage:.1f} MB"
        )

class MemoryMonitor:
    """Monitor memory usage for streaming processing."""
    
    def __init__(self):
        self.process = psutil.Process()
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except Exception:
            return 0.0
    
    def get_memory_percent(self) -> float:
        """Get current memory usage as percentage of system memory."""
        try:
            return self.process.memory_percent()
        except Exception:
            return 0.0
    
    def is_memory_available(self, required_mb: float) -> bool:
        """Check if required memory is available."""
        try:
            available_memory = psutil.virtual_memory().available / 1024 / 1024
            return available_memory > required_mb
        except Exception:
            return True

class AdaptiveChunking:
    """Adaptive chunking based on memory usage and data characteristics."""
    
    def __init__(self, base_chunk_size: int = 10000, memory_limit_mb: int = 1024):
        self.base_chunk_size = base_chunk_size
        self.memory_limit_mb = memory_limit_mb
        self.memory_monitor = MemoryMonitor()
    
    def calculate_optimal_chunk_size(self, data: pd.DataFrame) -> int:
        """Calculate optimal chunk size based on data characteristics and memory."""
        # Base chunk size
        chunk_size = self.base_chunk_size
        
        # Adjust based on data size
        if len(data) > 1000000:  # Very large dataset
            chunk_size = min(chunk_size, 5000)
        elif len(data) > 100000:  # Large dataset
            chunk_size = min(chunk_size, 8000)
        
        # Adjust based on memory usage
        current_memory = self.memory_monitor.get_memory_usage()
        if current_memory > self.memory_limit_mb * 0.8:
            chunk_size = max(chunk_size // 2, 1000)  # Reduce chunk size
        elif current_memory < self.memory_limit_mb * 0.3:
            chunk_size = min(chunk_size * 2, 20000)  # Increase chunk size
        
        return chunk_size

def create_streaming_processor(
    chunk_size: int = 10000,
    memory_limit_mb: int = 1024,
    overlap_size: int = 100,
    enable_gc: bool = True
) -> StreamingProcessor:
    """Create a streaming processor with specified configuration."""
    config = StreamingConfig(
        chunk_size=chunk_size,
        memory_limit_mb=memory_limit_mb,
        overlap_size=overlap_size,
        enable_gc=enable_gc
    )
    return StreamingProcessor(config)