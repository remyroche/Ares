"""
Memory-Optimized Feature Processing

Implements memory-efficient processing using memmap, tile-based operations,
and hardware-optimized memory management for large feature datasets.
"""

import os
import gc
import tempfile
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from contextlib import contextmanager

# Import hardware optimization tools
from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
from src.utils.hardware.advanced_memory_optimizer import get_advanced_memory_optimizer
from src.utils.hardware.unified_hardware_manager import WorkloadType, OptimizationLevel

from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
from src.utils.logger import get_logger


@dataclass
class MemoryConfig:
    """Configuration for memory-optimized processing."""
    # Memory limits
    max_memory_gb: float = 8.0
    tile_size_mb: int = 64  # Size of each processing tile
    enable_memmap: bool = True
    enable_compression: bool = True
    
    # Hardware optimization
    enable_hardware_optimization: bool = True
    memory_optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    
    # Processing settings
    enable_online_estimation: bool = True  # Use online algorithms for large datasets
    enable_welford_algorithm: bool = True  # Use Welford's algorithm for variance
    enable_batch_processing: bool = True
    batch_size: int = 1000


class MemoryOptimizedProcessor:
    """
    Memory-optimized processor for large feature datasets.
    
    Uses memmap, tile-based processing, and hardware optimization
    to handle datasets larger than available memory.
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None, logger=None):
        """Initialize the memory-optimized processor."""
        self.config = config or MemoryConfig()
        self.logger = logger or get_logger('MemoryOptimizedProcessor')
        
        tprint("🧠 Initializing Memory-Optimized Processor")
        
        # Initialize hardware memory optimizer
        if self.config.enable_hardware_optimization:
            self.memory_optimizer = M1MemoryOptimizer(self.config.max_memory_gb)
            self.advanced_memory_optimizer = get_advanced_memory_optimizer(self.config.max_memory_gb)
        else:
            self.memory_optimizer = None
            self.advanced_memory_optimizer = None
        
        # Memory tracking
        self.memory_usage = {
            'current_mb': 0.0,
            'peak_mb': 0.0,
            'tiles_processed': 0,
            'memmap_files': 0
        }
        
        # Temporary directory for memmap files
        self.temp_dir = Path(tempfile.mkdtemp(prefix="feature_lookback_memmap_"))
        
        tprint_info(f"   → Max memory: {self.config.max_memory_gb}GB")
        tprint_info(f"   → Tile size: {self.config.tile_size_mb}MB")
        tprint_info(f"   → Memmap enabled: {self.config.enable_memmap}")
        tprint_info(f"   → Hardware optimization: {self.config.enable_hardware_optimization}")
        tprint_info(f"   → Temp directory: {self.temp_dir}")
        
        tprint_success("✅ Memory-Optimized Processor initialized")
    
    def process_large_dataset(self, 
                            data: pd.DataFrame,
                            feature_columns: List[str],
                            target_column: str,
                            lookback_range: range,
                            chunk_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Process large dataset with memory optimization.
        
        Args:
            data: Input DataFrame
            feature_columns: List of feature column names
            target_column: Target column name
            lookback_range: Range of lookback periods to test
            chunk_size: Optional chunk size for processing
            
        Returns:
            Dictionary of optimization results
        """
        tprint("🚀 Starting memory-optimized dataset processing")
        tprint_info(f"   → Dataset shape: {data.shape}")
        tprint_info(f"   → Features: {len(feature_columns)}")
        tprint_info(f"   → Lookback range: {lookback_range.start}-{lookback_range.stop}")
        
        # Determine optimal chunk size
        if chunk_size is None:
            chunk_size = self._calculate_optimal_chunk_size(data, feature_columns)
        
        tprint_info(f"   → Chunk size: {chunk_size}")
        
        # Optimize data for hardware if available
        if self.memory_optimizer:
            data = self._optimize_dataframe_for_hardware(data)
        
        # Process in chunks
        results = {}
        total_chunks = (len(data) + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(0, len(data), chunk_size):
            chunk_end = min(chunk_idx + chunk_size, len(data))
            chunk_data = data.iloc[chunk_idx:chunk_end]
            
            tprint_debug(f"Processing chunk {chunk_idx//chunk_size + 1}/{total_chunks}")
            
            # Process chunk
            chunk_results = self._process_chunk(
                chunk_data, feature_columns, target_column, lookback_range
            )
            
            # Merge results
            for feature, feature_results in chunk_results.items():
                if feature not in results:
                    results[feature] = {
                        'scores': [],
                        'lookbacks': [],
                        'chunks_processed': 0
                    }
                
                results[feature]['scores'].extend(feature_results['scores'])
                results[feature]['lookbacks'].extend(feature_results['lookbacks'])
                results[feature]['chunks_processed'] += 1
            
            # Update memory tracking
            self._update_memory_tracking()
            
            # Force garbage collection if memory pressure is high
            if self._check_memory_pressure():
                gc.collect()
        
        # Aggregate results across chunks
        final_results = self._aggregate_chunk_results(results, lookback_range)
        
        tprint_success(f"✅ Memory-optimized processing completed")
        tprint_info(f"   → Peak memory usage: {self.memory_usage['peak_mb']:.1f}MB")
        tprint_info(f"   → Tiles processed: {self.memory_usage['tiles_processed']}")
        
        return final_results
    
    def _process_chunk(self, 
                      chunk_data: pd.DataFrame,
                      feature_columns: List[str],
                      target_column: str,
                      lookback_range: range) -> Dict[str, Dict[str, List]]:
        """Process a single chunk of data."""
        results = {}
        
        for feature in feature_columns:
            if feature not in chunk_data.columns:
                continue
            
            feature_data = chunk_data[feature].values
            target_data = chunk_data[target_column].values
            
            feature_results = {
                'scores': [],
                'lookbacks': []
            }
            
            for lookback in lookback_range:
                try:
                    # Apply lookback with memory optimization
                    feature_with_lookback = self._apply_lookback_optimized(
                        feature_data, lookback
                    )
                    
                    # Calculate score with online estimation if enabled
                    if self.config.enable_online_estimation:
                        score = self._calculate_ic_online(
                            feature_with_lookback, target_data
                        )
                    else:
                        score = self._calculate_ic_standard(
                            feature_with_lookback, target_data
                        )
                    
                    feature_results['scores'].append(score)
                    feature_results['lookbacks'].append(lookback)
                    
                except Exception as e:
                    tprint_debug(f"Lookback {lookback} failed for {feature}: {e}")
                    feature_results['scores'].append(-np.inf)
                    feature_results['lookbacks'].append(lookback)
            
            results[feature] = feature_results
        
        return results
    
    def _apply_lookback_optimized(self, feature_data: np.ndarray, lookback: int) -> np.ndarray:
        """Apply lookback period with memory optimization."""
        if lookback <= 0:
            return feature_data
        
        # Use memmap for large arrays
        if self.config.enable_memmap and len(feature_data) > 10000:
            return self._apply_lookback_memmap(feature_data, lookback)
        else:
            return self._apply_lookback_standard(feature_data, lookback)
    
    def _apply_lookback_standard(self, feature_data: np.ndarray, lookback: int) -> np.ndarray:
        """Standard lookback application."""
        result = np.full_like(feature_data, np.nan)
        for i in range(lookback - 1, len(feature_data)):
            result[i] = np.mean(feature_data[i - lookback + 1:i + 1])
        return result
    
    def _apply_lookback_memmap(self, feature_data: np.ndarray, lookback: int) -> np.ndarray:
        """Apply lookback using memory-mapped arrays."""
        # Create memmap file
        memmap_file = self.temp_dir / f"lookback_{lookback}_{id(feature_data)}.npy"
        
        try:
            # Create result array as memmap
            result = np.memmap(memmap_file, dtype=feature_data.dtype, 
                             mode='w+', shape=feature_data.shape)
            
            # Process in tiles to avoid memory spikes
            tile_size = self.config.tile_size_mb * 1024 * 1024 // (feature_data.dtype.itemsize * 2)
            tile_size = max(1000, min(tile_size, len(feature_data)))
            
            for i in range(0, len(feature_data), tile_size):
                end_idx = min(i + tile_size, len(feature_data))
                
                # Calculate rolling mean for this tile
                for j in range(i, end_idx):
                    if j >= lookback - 1:
                        start_idx = max(0, j - lookback + 1)
                        result[j] = np.mean(feature_data[start_idx:j+1])
                    else:
                        result[j] = np.nan
            
            # Flush to disk
            result.flush()
            self.memory_usage['memmap_files'] += 1
            
            return result
            
        except Exception as e:
            tprint_warning(f"Memmap processing failed, falling back to standard: {e}")
            return self._apply_lookback_standard(feature_data, lookback)
    
    def _calculate_ic_online(self, feature_data: np.ndarray, target_data: np.ndarray) -> float:
        """Calculate IC using online estimation for large datasets."""
        if self.config.enable_welford_algorithm:
            return self._calculate_ic_welford(feature_data, target_data)
        else:
            return self._calculate_ic_standard(feature_data, target_data)
    
    def _calculate_ic_welford(self, feature_data: np.ndarray, target_data: np.ndarray) -> float:
        """Calculate IC using Welford's online algorithm."""
        # Remove NaN values
        mask = ~(np.isnan(feature_data) | np.isnan(target_data))
        if np.sum(mask) < 10:
            return -np.inf
        
        feature_clean = feature_data[mask]
        target_clean = target_data[mask]
        
        # Use Welford's algorithm for online correlation
        n = len(feature_clean)
        if n < 2:
            return -np.inf
        
        # Initialize
        sum_x = feature_clean[0]
        sum_y = target_clean[0]
        sum_xy = feature_clean[0] * target_clean[0]
        sum_x2 = feature_clean[0] ** 2
        sum_y2 = target_clean[0] ** 2
        
        # Online update
        for i in range(1, n):
            x = feature_clean[i]
            y = target_clean[i]
            
            sum_x += x
            sum_y += y
            sum_xy += x * y
            sum_x2 += x ** 2
            sum_y2 += y ** 2
        
        # Calculate correlation
        mean_x = sum_x / n
        mean_y = sum_y / n
        
        numerator = sum_xy - n * mean_x * mean_y
        denominator = np.sqrt((sum_x2 - n * mean_x ** 2) * (sum_y2 - n * mean_y ** 2))
        
        if denominator == 0:
            return -np.inf
        
        correlation = numerator / denominator
        return correlation if not np.isnan(correlation) else -np.inf
    
    def _calculate_ic_standard(self, feature_data: np.ndarray, target_data: np.ndarray) -> float:
        """Standard IC calculation."""
        # Remove NaN values
        mask = ~(np.isnan(feature_data) | np.isnan(target_data))
        if np.sum(mask) < 10:
            return -np.inf
        
        feature_clean = feature_data[mask]
        target_clean = target_data[mask]
        
        # Calculate correlation
        correlation = np.corrcoef(feature_clean, target_clean)[0, 1]
        return correlation if not np.isnan(correlation) else -np.inf
    
    def _calculate_optimal_chunk_size(self, data: pd.DataFrame, feature_columns: List[str]) -> int:
        """Calculate optimal chunk size based on available memory."""
        # Estimate memory usage per row
        bytes_per_row = data[feature_columns].memory_usage(deep=True).sum() / len(data)
        
        # Calculate chunk size based on available memory
        available_memory_bytes = self.config.max_memory_gb * 1024 * 1024 * 1024 * 0.8  # Use 80% of available
        optimal_chunk_size = int(available_memory_bytes / bytes_per_row)
        
        # Ensure reasonable bounds
        optimal_chunk_size = max(1000, min(optimal_chunk_size, 50000))
        
        tprint_info(f"   → Optimal chunk size: {optimal_chunk_size} rows")
        return optimal_chunk_size
    
    def _optimize_dataframe_for_hardware(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for hardware processing."""
        if self.memory_optimizer:
            try:
                optimized_data = self.memory_optimizer.optimize_dataframe(data)
                tprint_debug("✅ DataFrame optimized for M1 memory architecture")
                return optimized_data
            except Exception as e:
                tprint_debug(f"Hardware optimization failed: {e}")
        
        return data
    
    def _update_memory_tracking(self):
        """Update memory usage tracking."""
        if self.memory_optimizer:
            try:
                current_usage = self.memory_optimizer.get_memory_usage()
                self.memory_usage['current_mb'] = current_usage / (1024 * 1024)
                self.memory_usage['peak_mb'] = max(
                    self.memory_usage['peak_mb'], 
                    self.memory_usage['current_mb']
                )
            except Exception as e:
                tprint_debug(f"Memory tracking update failed: {e}")
        
        self.memory_usage['tiles_processed'] += 1
    
    def _check_memory_pressure(self) -> bool:
        """Check if memory pressure is high."""
        if self.memory_optimizer:
            try:
                pressure = self.memory_optimizer.memory_pressure
                return pressure > 0.8  # High pressure threshold
            except Exception:
                pass
        
        return False
    
    def _aggregate_chunk_results(self, 
                                chunk_results: Dict[str, Dict[str, List]], 
                                lookback_range: range) -> Dict[str, Any]:
        """Aggregate results from multiple chunks."""
        final_results = {}
        
        for feature, results in chunk_results.items():
            if not results['scores']:
                continue
            
            # Convert to numpy arrays for easier processing
            scores = np.array(results['scores'])
            lookbacks = np.array(results['lookbacks'])
            
            # Find best lookback period
            valid_scores = ~np.isnan(scores) & (scores != -np.inf)
            if not np.any(valid_scores):
                continue
            
            valid_scores_array = scores[valid_scores]
            valid_lookbacks_array = lookbacks[valid_scores]
            
            best_idx = np.argmax(valid_scores_array)
            best_lookback = valid_lookbacks_array[best_idx]
            best_score = valid_scores_array[best_idx]
            
            # Calculate stability metrics
            stability_score = self._calculate_stability_score(valid_scores_array)
            
            final_results[feature] = {
                'best_lookback': int(best_lookback),
                'best_score': float(best_score),
                'stability_score': float(stability_score),
                'chunks_processed': results['chunks_processed'],
                'all_scores': scores.tolist(),
                'all_lookbacks': lookbacks.tolist()
            }
        
        return final_results
    
    def _calculate_stability_score(self, scores: np.ndarray) -> float:
        """Calculate stability score for lookback optimization."""
        if len(scores) < 2:
            return 0.0
        
        # Calculate coefficient of variation (lower is more stable)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        if mean_score == 0:
            return 0.0
        
        cv = std_score / abs(mean_score)
        stability_score = max(0.0, 1.0 - cv)  # Convert to 0-1 scale
        
        return stability_score
    
    def cleanup(self):
        """Cleanup resources and temporary files."""
        tprint("🧹 Cleaning up memory-optimized processor...")
        
        # Stop memory monitoring
        if self.memory_optimizer:
            self.memory_optimizer.stop_monitoring()
        
        # Cleanup temporary files
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                tprint_debug(f"✅ Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            tprint_warning(f"Failed to cleanup temp directory: {e}")
        
        # Force garbage collection
        gc.collect()
        
        tprint_success("✅ Memory-optimized processor cleanup completed")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return self.memory_usage.copy()


# Convenience function
def process_large_dataset_memory_optimized(data: pd.DataFrame,
                                         feature_columns: List[str],
                                         target_column: str,
                                         lookback_range: range,
                                         config: Optional[MemoryConfig] = None) -> Dict[str, Any]:
    """
    Convenience function for memory-optimized processing.
    
    Args:
        data: Input DataFrame
        feature_columns: List of feature column names
        target_column: Target column name
        lookback_range: Range of lookback periods to test
        config: Memory configuration
        
    Returns:
        Dictionary of optimization results
    """
    processor = MemoryOptimizedProcessor(config)
    try:
        return processor.process_large_dataset(data, feature_columns, target_column, lookback_range)
    finally:
        processor.cleanup()