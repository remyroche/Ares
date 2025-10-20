"""
Chunked Processing for Large Datasets

This module provides chunked processing capabilities for large datasets
with memory-efficient algorithms and hardware optimization.
"""

import logging
import time
import gc
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd

# Import hardware optimization tools
from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager, HardwareConfig
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_performance, tprint_debug
from src.utils.hardware import (
    get_integrated_hardware_manager, 
    get_comprehensive_optimizer,
    memory_optimized, 
    comprehensive_memory_optimization,
    optimize_dataframe, 
    optimize_array,
    m1_optimized,
    WorkloadCategory,
    MemoryOptimizationLevel
)

logger = logging.getLogger(__name__)

@dataclass
class ChunkedConfig:
    """Configuration for chunked processing."""
    # Chunking settings
    chunk_size: int = 10000
    min_chunk_size: int = 1000
    max_chunk_size: int = 50000
    adaptive_chunking: bool = True

    # Memory management
    memory_limit_gb: float = 8.0
    memory_threshold: float = 0.8
    enable_garbage_collection: bool = True

    # Processing settings
    enable_parallel_chunks: bool = True
    max_parallel_chunks: int = 4
    enable_progress_tracking: bool = True

    # Hardware optimization
    enable_hardware_optimization: bool = True
    enable_memory_monitoring: bool = True

class ChunkedFeatureProcessor:
    """Processor for chunked feature selection operations."""

    def __init__(self, config: Optional[ChunkedConfig] = None):
        """Initialize chunked processor."""
        self.config = config or ChunkedConfig()
        self.logger = logger.getChild('ChunkedFeatureProcessor')

        # Initialize hardware tools
        if self.config.enable_hardware_optimization:
            self.memory_optimizer = get_integrated_hardware_manager().memory_manager(self.config.memory_limit_gb)
            if self.config.enable_memory_monitoring:
                self.memory_optimizer.start_monitoring()

            hw_config = HardwareConfig(
                memory_limit_gb=self.config.memory_limit_gb,
                memory_optimization_level='aggressive'
            )
            self.hardware_manager = UnifiedHardwareManager(hw_config)
        else:
            self.memory_optimizer = None
            self.hardware_manager = None

        # Processing statistics
        self.processing_stats = {
            'total_chunks': 0,
            'successful_chunks': 0,
            'failed_chunks': 0,
            'total_time': 0.0,
            'memory_optimizations': 0,
            'garbage_collections': 0
        }

        tprint_success("📦 ChunkedFeatureProcessor initialized")

    def _calculate_optimal_chunk_size(self, data_size: int, memory_usage: float = 0.5) -> int:
        """Calculate optimal chunk size based on data size and memory usage."""
        if not self.config.adaptive_chunking:
            return self.config.chunk_size

        # Base chunk size
        base_chunk = self.config.chunk_size

        # Adjust based on data size
        if data_size < 10000:
            chunk_size = min(base_chunk, data_size)
        elif data_size < 100000:
            chunk_size = base_chunk
        elif data_size < 1000000:
            chunk_size = base_chunk // 2
        else:
            chunk_size = base_chunk // 4

        # Adjust based on memory usage
        if memory_usage > 0.7:
            chunk_size = max(self.config.min_chunk_size, chunk_size // 2)
        elif memory_usage < 0.3:
            chunk_size = min(self.config.max_chunk_size, chunk_size * 2)

        return max(self.config.min_chunk_size, min(chunk_size, self.config.max_chunk_size))

    def _check_memory_usage(self) -> float:
        """Check current memory usage."""
        try:
            if self.memory_optimizer:
                return self.memory_optimizer.get_memory_pressure()
            else:
                import psutil
                return psutil.virtual_memory().percent / 100.0
        except Exception as mem_e:
            tprint_debug(f"⚠️ Memory check failed: {mem_e}")
            return 0.5  # Default to 50% if can't check

    def _optimize_memory_if_needed(self) -> bool:
        """Optimize memory if usage is too high."""
        memory_usage = self._check_memory_usage()

        if memory_usage > self.config.memory_threshold:
            tprint_warning(f"⚠️ High memory usage: {memory_usage:.1%}")

            if self.config.enable_garbage_collection:
                # Force garbage collection
                collected = gc.collect()
                self.processing_stats['garbage_collections'] += 1
                tprint_debug(f"🗑️ Garbage collected {collected} objects")

            if self.memory_optimizer:
                # Apply memory optimizations
                optimization_result = self.memory_optimizer.get_integrated_hardware_manager().clear_all_caches()
                if optimization_result.get('optimized', False):
                    self.processing_stats['memory_optimizations'] += 1
                    tprint_success("🧠 Memory optimized")
                    return True

            return False
        return True

    def _create_chunks(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, int, int]]:
        """Create chunks from data."""
        n_samples = X.shape[0]
        memory_usage = self._check_memory_usage()
        chunk_size = self._calculate_optimal_chunk_size(n_samples, memory_usage)

        chunks = []
        for i in range(0, n_samples, chunk_size):
            end_idx = min(i + chunk_size, n_samples)
            X_chunk = X[i:end_idx]
            y_chunk = y[i:end_idx]
            chunks.append((X_chunk, y_chunk, i, end_idx))

        tprint_debug(f"📦 Created {len(chunks)} chunks of size {chunk_size}")
        return chunks

    def _process_chunk(self, X_chunk: np.ndarray, y_chunk: np.ndarray,
                      processor_func: Callable, **kwargs) -> Dict[str, Any]:
        """Process a single chunk."""
        try:
            # Process chunk
            result = processor_func(X_chunk, y_chunk, **kwargs)

            # Update stats
            self.processing_stats['successful_chunks'] += 1

            return {
                'success': True,
                'result': result,
                'chunk_size': X_chunk.shape[0]
            }

        except Exception as e:
            self.logger.warning(f"Chunk processing failed: {e}")
            self.processing_stats['failed_chunks'] += 1

            return {
                'success': False,
                'error': str(e),
                'chunk_size': X_chunk.shape[0]
            }

    def _combine_chunk_results(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from multiple chunks."""
        try:
            # Filter successful results
            successful_results = [r for r in chunk_results if r.get('success', False)]

            if not successful_results:
                return {'success': False, 'error': 'All chunks failed'}

            # Extract results
            results = [r['result'] for r in successful_results]

            # Combine selected features (intersection of all chunks)
            all_selected = []
            for result in results:
                if 'selected_features' in result and result['selected_features']:
                    all_selected.append(set(result['selected_features']))

            if not all_selected:
                return {'success': False, 'error': 'No features selected in any chunk'}

            # Find common features across chunks
            common_features = set.intersection(*all_selected)

            # If no common features, use union
            if not common_features:
                common_features = set.union(*all_selected)
                tprint_warning("⚠️ No common features across chunks, using union")

            # Calculate average scores if available
            feature_scores = {}
            for result in results:
                if 'feature_scores' in result:
                    for feature, score in result['feature_scores'].items():
                        if feature in common_features:
                            if feature not in feature_scores:
                                feature_scores[feature] = []
                            feature_scores[feature].append(score)

            # Average scores
            for feature in feature_scores:
                feature_scores[feature] = np.mean(feature_scores[feature])

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

    def process_large_dataset(self, X: np.ndarray, y: np.ndarray,
                            processor_func: Callable, **kwargs) -> Dict[str, Any]:
        """Process large dataset using chunked processing."""
        tprint_info(f"📦 Starting chunked processing: {X.shape}")

        start_time = time.time()

        try:
            # Create chunks
            chunks = self._create_chunks(X, y)
            self.processing_stats['total_chunks'] = len(chunks)

            # Process chunks
            chunk_results = []
            for i, (X_chunk, y_chunk, start_idx, end_idx) in enumerate(chunks):
                if self.config.enable_progress_tracking:
                    tprint_debug(f"📦 Processing chunk {i+1}/{len(chunks)}: rows {start_idx}-{end_idx}")

                # Process chunk
                chunk_result = self._process_chunk(X_chunk, y_chunk, processor_func, **kwargs)
                chunk_result['chunk_index'] = i
                chunk_result['start_idx'] = start_idx
                chunk_result['end_idx'] = end_idx
                chunk_results.append(chunk_result)

                # Memory optimization
                if not self._optimize_memory_if_needed():
                    tprint_warning("⚠️ Memory optimization failed, continuing...")

                # Progress update
                if (i + 1) % 10 == 0:
                    tprint_performance(f"📊 Processed {i+1}/{len(chunks)} chunks")

            # Combine results
            combined_result = self._combine_chunk_results(chunk_results)

            end_time = time.time()
            execution_time = end_time - start_time
            self.processing_stats['total_time'] += execution_time

            # Add processing statistics
            combined_result['processing_stats'] = {
                'total_chunks': self.processing_stats['total_chunks'],
                'successful_chunks': self.processing_stats['successful_chunks'],
                'failed_chunks': self.processing_stats['failed_chunks'],
                'execution_time': execution_time,
                'memory_optimizations': self.processing_stats['memory_optimizations'],
                'garbage_collections': self.processing_stats['garbage_collections']
            }

            tprint_success(f"✅ Chunked processing completed: {len(combined_result.get('selected_features', []))} features "
                         f"in {execution_time:.2f}s")

            return combined_result

        except Exception as e:
            self.logger.error(f"Chunked processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.processing_stats.copy()

        if stats['total_chunks'] > 0:
            stats['success_rate'] = stats['successful_chunks'] / stats['total_chunks']
            stats['avg_time_per_chunk'] = stats['total_time'] / stats['total_chunks']
        else:
            stats['success_rate'] = 0.0
            stats['avg_time_per_chunk'] = 0.0

        tprint_performance(f"📊 Chunked Processing Stats: {stats['success_rate']:.1%} success rate, "
                         f"{stats['avg_time_per_chunk']:.3f}s avg per chunk")

        return stats

class AdaptiveChunkProcessor:
    """Adaptive chunk processor that adjusts chunk size based on performance."""

    def __init__(self, config: Optional[ChunkedConfig] = None):
        """Initialize adaptive chunk processor."""
        self.config = config or ChunkedConfig()
        self.processor = ChunkedFeatureProcessor(self.config)

        # Adaptive parameters
        self.performance_history = []
        self.optimal_chunk_size = self.config.chunk_size

        tprint_success("🔄 AdaptiveChunkProcessor initialized")

    def _update_optimal_chunk_size(self, chunk_size: int, execution_time: float, memory_usage: float):
        """Update optimal chunk size based on performance."""
        performance_score = execution_time / (chunk_size / 1000)  # Time per 1000 samples

        self.performance_history.append({
            'chunk_size': chunk_size,
            'execution_time': execution_time,
            'memory_usage': memory_usage,
            'performance_score': performance_score
        })

        # Keep only recent history
        if len(self.performance_history) > 10:
            self.performance_history = self.performance_history[-10:]

        # Calculate optimal chunk size
        if len(self.performance_history) >= 3:
            # Find chunk size with best performance
            best_performance = min(self.performance_history, key=lambda x: x['performance_score'])
            self.optimal_chunk_size = best_performance['chunk_size']

            tprint_debug(f"🔄 Updated optimal chunk size: {self.optimal_chunk_size}")

    def process_with_adaptation(self, X: np.ndarray, y: np.ndarray,
                              processor_func: Callable, **kwargs) -> Dict[str, Any]:
        """Process dataset with adaptive chunk sizing."""
        tprint_info(f"🔄 Starting adaptive chunked processing: {X.shape}")

        # Update processor config with optimal chunk size
        self.processor.config.chunk_size = self.optimal_chunk_size

        # Process dataset
        result = self.processor.process_large_dataset(X, y, processor_func, **kwargs)

        # Update optimal chunk size based on results
        if 'processing_stats' in result:
            stats = result['processing_stats']
            if stats['total_chunks'] > 0:
                avg_time_per_chunk = stats['execution_time'] / stats['total_chunks']
                memory_usage = self.processor._check_memory_usage()
                self._update_optimal_chunk_size(
                    self.optimal_chunk_size,
                    avg_time_per_chunk,
                    memory_usage
                )

        return result

def create_chunked_processor(config: Optional[ChunkedConfig] = None) -> ChunkedFeatureProcessor:
    """Create a chunked feature processor."""
    return ChunkedFeatureProcessor(config)
