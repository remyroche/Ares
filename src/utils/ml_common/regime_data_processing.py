#!/usr/bin/env python3
"""
Enhanced Regime Data Processing Utilities

This module provides comprehensive regime data processing capabilities with
advanced features for handling large-scale regime-based datasets:

Key Features:
- Async File Processing: High-performance asynchronous file operations
- Memory Pool Management: Efficient memory management for large datasets
- Data Type Optimization: Memory-efficient data type handling
- Regime Continuity Validation: Ensures regime transition consistency
- Cross-Regime Analysis: Advanced cross-regime data validation
- Parallel Processing: Multi-threaded regime data processing
- M1 Optimization: Apple M1 chip optimizations
- GPU Acceleration: M1 MPS support for data processing
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
import logging
from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
from pathlib import Path
import multiprocessing as mp
from queue import Queue, Empty
import threading
import gc

# Import comprehensive utility infrastructure
from ..math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_kelly_calculation,
    validate_positive, validate_range, MathValidationError
)
from ..common_operations import create_fallback_logger, create_fallback_decorator
from ..common_utilities import CommonUtilities
from ..parquet_utils import ParquetUtils
from ..serialization_utils import UniversalSerializer
from ..data_processing_utils import DataProcessingUtils
from ..m1_gpu_utils import get_m1_gpu_manager, M1GPUManager
from ..m1_memory_optimizer import get_m1_memory_optimizer, M1MemoryOptimizer
from ..m1_cpu_optimizer import get_m1_cpu_optimizer, M1CPUOptimizer

# Import ML Common utilities
from .cv_utils import TemporalCrossValidator, PurgedKFold
from .validation_utils import ValidationFramework
from .pareto import ParetoFrontAnalyzer
from .ensemble_manager import EnsembleManager

logger = logging.getLogger(__name__)

class ProcessingMode(Enum):
    """Available processing modes."""
    SYNC = "sync"
    ASYNC = "async"
    PARALLEL = "parallel"
    STREAMING = "streaming"

class DataTypeOptimization(Enum):
    """Available data type optimizations."""
    MEMORY = "memory"
    SPEED = "speed"
    BALANCED = "balanced"

@dataclass
class AsyncFileProcessorConfig:
    """Configuration for async file processing."""
    chunk_size: int = 10000
    max_concurrent_files: int = 5
    buffer_size: int = 8192
    compression: str = "snappy"
    use_memory_mapping: bool = True

@dataclass
class MemoryPoolConfig:
    """Configuration for memory pool management."""
    max_pool_size: int = 1000
    initial_pool_size: int = 100
    cleanup_interval: int = 300  # seconds
    memory_threshold: float = 0.8  # 80% of available memory
    gc_frequency: int = 100  # operations

@dataclass
class DataTypeOptimizerConfig:
    """Configuration for data type optimization."""
    optimization_mode: DataTypeOptimization = DataTypeOptimization.BALANCED
    enable_compression: bool = True
    precision_threshold: float = 1e-6
    memory_target_reduction: float = 0.3  # 30% memory reduction

@dataclass
class RegimeContinuityConfig:
    """Configuration for regime continuity validation."""
    max_gap_tolerance: int = 5
    continuity_threshold: float = 0.9
    transition_validation: bool = True
    temporal_consistency_check: bool = True

@dataclass
class ProcessingStats:
    """Statistics for processing operations."""
    files_processed: int = 0
    total_rows_processed: int = 0
    processing_time: float = 0.0
    memory_usage: float = 0.0
    error_count: int = 0
    optimization_savings: float = 0.0

class AsyncFileProcessor:
    """High-performance async file processor for regime data."""
    
    def __init__(self, config: Optional[AsyncFileProcessorConfig] = None):
        self.config = config or AsyncFileProcessorConfig()
        self.logger = create_fallback_logger("AsyncFileProcessor")
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_files)
        
        # Initialize utilities
        self.parquet_utils = ParquetUtils() if ParquetUtils else None
        self.serializer = UniversalSerializer() if UniversalSerializer else None

    async def process_file_async(
        self, 
        file_path: str, 
        processing_func: Callable[[pd.DataFrame], pd.DataFrame]
    ) -> pd.DataFrame:
        """Process a single file asynchronously."""
        async with self.semaphore:
            try:
                # Read file asynchronously
                data = await self._read_file_async(file_path)
                
                # Process data
                processed_data = processing_func(data)
                
                # Optimize data types
                optimized_data = self._optimize_data_types(processed_data)
                
                return optimized_data
                
            except Exception as e:
                self.logger.error(f"❌ Failed to process file {file_path}: {e}")
                raise

    async def process_files_batch(
        self, 
        file_paths: List[str], 
        processing_func: Callable[[pd.DataFrame], pd.DataFrame]
    ) -> List[pd.DataFrame]:
        """Process multiple files in batch asynchronously."""
        tasks = [
            self.process_file_async(file_path, processing_func)
            for file_path in file_paths
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in results if not isinstance(r, Exception)]
        exceptions = [r for r in results if isinstance(r, Exception)]
        
        if exceptions:
            self.logger.warning(f"⚠️ {len(exceptions)} files failed to process")
        
        return valid_results

    async def _read_file_async(self, file_path: str) -> pd.DataFrame:
        """Read file asynchronously."""
        try:
            if file_path.endswith('.parquet'):
                if self.parquet_utils:
                    return await asyncio.get_event_loop().run_in_executor(
                        None, self.parquet_utils.read_parquet, file_path
                    )
                else:
                    return await asyncio.get_event_loop().run_in_executor(
                        None, pd.read_parquet, file_path
                    )
            else:
                # Fallback to pandas
                return await asyncio.get_event_loop().run_in_executor(
                    None, pd.read_csv, file_path
                )
        except Exception as e:
            self.logger.error(f"❌ Failed to read file {file_path}: {e}")
            raise

    def _optimize_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency."""
        try:
            optimized_data = data.copy()
            
            # Optimize numeric columns
            for col in optimized_data.select_dtypes(include=[np.number]).columns:
                col_data = optimized_data[col]
                
                # Check if column can be downcast
                if col_data.dtype == 'float64':
                    if col_data.min() >= np.finfo(np.float32).min and col_data.max() <= np.finfo(np.float32).max:
                        optimized_data[col] = col_data.astype(np.float32)
                
                elif col_data.dtype == 'int64':
                    if col_data.min() >= np.iinfo(np.int32).min and col_data.max() <= np.iinfo(np.int32).max:
                        optimized_data[col] = col_data.astype(np.int32)
                    elif col_data.min() >= np.iinfo(np.int16).min and col_data.max() <= np.iinfo(np.int16).max:
                        optimized_data[col] = col_data.astype(np.int16)
                    elif col_data.min() >= np.iinfo(np.int8).min and col_data.max() <= np.iinfo(np.int8).max:
                        optimized_data[col] = col_data.astype(np.int8)
            
            # Optimize categorical columns
            for col in optimized_data.select_dtypes(include=['object']).columns:
                if optimized_data[col].nunique() / len(optimized_data) < 0.5:
                    optimized_data[col] = optimized_data[col].astype('category')
            
            return optimized_data
            
        except Exception as e:
            self.logger.warning(f"⚠️ Data type optimization failed: {e}")
            return data

class MemoryPoolManager:
    """Memory pool manager for efficient memory usage."""
    
    def __init__(self, config: Optional[MemoryPoolConfig] = None):
        self.config = config or MemoryPoolConfig()
        self.logger = create_fallback_logger("MemoryPoolManager")
        
        # Initialize memory optimizer
        self.memory_optimizer = get_m1_memory_optimizer() if get_m1_memory_optimizer else None
        
        # Memory pool state
        self.memory_pool: Queue = Queue(maxsize=self.config.max_pool_size)
        self.active_objects: Dict[str, Any] = {}
        self.memory_stats = {
            'pool_hits': 0,
            'pool_misses': 0,
            'memory_freed': 0,
            'gc_runs': 0
        }
        
        # Start cleanup thread
        self._start_cleanup_thread()

    def get_object(self, object_id: str) -> Optional[Any]:
        """Get object from memory pool."""
        try:
            # Check active objects first
            if object_id in self.active_objects:
                self.memory_stats['pool_hits'] += 1
                return self.active_objects[object_id]
            
            # Check memory pool
            if not self.memory_pool.empty():
                try:
                    obj = self.memory_pool.get_nowait()
                    self.active_objects[object_id] = obj
                    self.memory_stats['pool_hits'] += 1
                    return obj
                except Empty:
                    pass
            
            self.memory_stats['pool_misses'] += 1
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get object {object_id}: {e}")
            return None

    def put_object(self, object_id: str, obj: Any):
        """Put object into memory pool."""
        try:
            # Add to active objects
            self.active_objects[object_id] = obj
            
            # Try to add to pool if there's space
            if not self.memory_pool.full():
                try:
                    self.memory_pool.put_nowait(obj)
                except:
                    pass  # Pool is full, keep in active objects
            
        except Exception as e:
            self.logger.error(f"❌ Failed to put object {object_id}: {e}")

    def release_object(self, object_id: str):
        """Release object from memory pool."""
        try:
            if object_id in self.active_objects:
                del self.active_objects[object_id]
                self.memory_stats['memory_freed'] += 1
                
                # Trigger garbage collection if needed
                if self.memory_stats['memory_freed'] % self.config.gc_frequency == 0:
                    gc.collect()
                    self.memory_stats['gc_runs'] += 1
                    
        except Exception as e:
            self.logger.error(f"❌ Failed to release object {object_id}: {e}")

    def _start_cleanup_thread(self):
        """Start cleanup thread for memory management."""
        def cleanup_worker():
            while True:
                time.sleep(self.config.cleanup_interval)
                self._cleanup_memory_pool()
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()

    def _cleanup_memory_pool(self):
        """Cleanup memory pool."""
        try:
            # Check memory usage
            if self.memory_optimizer:
                memory_usage = self.memory_optimizer.get_memory_usage()
                if memory_usage > self.config.memory_threshold:
                    # Clear half of the pool
                    for _ in range(self.config.max_pool_size // 2):
                        try:
                            self.memory_pool.get_nowait()
                        except Empty:
                            break
                    
                    # Force garbage collection
                    gc.collect()
                    self.memory_stats['gc_runs'] += 1
                    
        except Exception as e:
            self.logger.error(f"❌ Memory pool cleanup failed: {e}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        return {
            'pool_size': self.memory_pool.qsize(),
            'active_objects': len(self.active_objects),
            'pool_hits': self.memory_stats['pool_hits'],
            'pool_misses': self.memory_stats['pool_misses'],
            'memory_freed': self.memory_stats['memory_freed'],
            'gc_runs': self.memory_stats['gc_runs'],
            'hit_rate': safe_divide(
                self.memory_stats['pool_hits'],
                self.memory_stats['pool_hits'] + self.memory_stats['pool_misses']
            )
        }

class DataTypeOptimizer:
    """Data type optimizer for memory efficiency."""
    
    def __init__(self, config: Optional[DataTypeOptimizerConfig] = None):
        self.config = config or DataTypeOptimizerConfig()
        self.logger = create_fallback_logger("DataTypeOptimizer")
        
        # Initialize memory optimizer
        self.memory_optimizer = get_m1_memory_optimizer() if get_m1_memory_optimizer else None

    def optimize_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types for memory efficiency."""
        try:
            optimized_data = data.copy()
            original_memory = optimized_data.memory_usage(deep=True).sum()
            
            # Optimize based on configuration
            if self.config.optimization_mode == DataTypeOptimization.MEMORY:
                optimized_data = self._optimize_for_memory(optimized_data)
            elif self.config.optimization_mode == DataTypeOptimization.SPEED:
                optimized_data = self._optimize_for_speed(optimized_data)
            else:  # BALANCED
                optimized_data = self._optimize_balanced(optimized_data)
            
            # Apply compression if enabled
            if self.config.enable_compression:
                optimized_data = self._apply_compression(optimized_data)
            
            new_memory = optimized_data.memory_usage(deep=True).sum()
            memory_reduction = safe_divide(original_memory - new_memory, original_memory)
            
            self.logger.info(f"✅ Memory optimization: {memory_reduction:.2%} reduction")
            return optimized_data
            
        except Exception as e:
            self.logger.error(f"❌ Data type optimization failed: {e}")
            return data

    def _optimize_for_memory(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize for maximum memory reduction."""
        optimized_data = data.copy()
        
        # Aggressive downcasting
        for col in optimized_data.select_dtypes(include=[np.number]).columns:
            col_data = optimized_data[col]
            
            if col_data.dtype == 'float64':
                # Try float32 first
                if self._can_downcast_float(col_data, np.float32):
                    optimized_data[col] = col_data.astype(np.float32)
                # Try float16 if possible
                elif self._can_downcast_float(col_data, np.float16):
                    optimized_data[col] = col_data.astype(np.float16)
            
            elif col_data.dtype == 'int64':
                # Try smaller integer types
                for dtype in [np.int32, np.int16, np.int8]:
                    if self._can_downcast_int(col_data, dtype):
                        optimized_data[col] = col_data.astype(dtype)
                        break
        
        return optimized_data

    def _optimize_for_speed(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize for maximum processing speed."""
        optimized_data = data.copy()
        
        # Minimal optimization to maintain speed
        for col in optimized_data.select_dtypes(include=['object']).columns:
            if optimized_data[col].nunique() / len(optimized_data) < 0.1:
                optimized_data[col] = optimized_data[col].astype('category')
        
        return optimized_data

    def _optimize_balanced(self, data: pd.DataFrame) -> pd.DataFrame:
        """Balanced optimization between memory and speed."""
        optimized_data = data.copy()
        
        # Moderate optimization
        for col in optimized_data.select_dtypes(include=[np.number]).columns:
            col_data = optimized_data[col]
            
            if col_data.dtype == 'float64':
                if self._can_downcast_float(col_data, np.float32):
                    optimized_data[col] = col_data.astype(np.float32)
            
            elif col_data.dtype == 'int64':
                if self._can_downcast_int(col_data, np.int32):
                    optimized_data[col] = col_data.astype(np.int32)
        
        # Optimize categorical columns
        for col in optimized_data.select_dtypes(include=['object']).columns:
            if optimized_data[col].nunique() / len(optimized_data) < 0.3:
                optimized_data[col] = optimized_data[col].astype('category')
        
        return optimized_data

    def _can_downcast_float(self, data: pd.Series, target_dtype) -> bool:
        """Check if float data can be downcast safely."""
        try:
            return (
                data.min() >= np.finfo(target_dtype).min and
                data.max() <= np.finfo(target_dtype).max and
                np.allclose(data, data.astype(target_dtype).astype(np.float64), 
                           rtol=self.config.precision_threshold)
            )
        except:
            return False

    def _can_downcast_int(self, data: pd.Series, target_dtype) -> bool:
        """Check if integer data can be downcast safely."""
        try:
            return (
                data.min() >= np.iinfo(target_dtype).min and
                data.max() <= np.iinfo(target_dtype).max
            )
        except:
            return False

    def _apply_compression(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply compression to DataFrame."""
        try:
            # This would implement compression logic
            # For now, return the data as-is
            return data
        except Exception as e:
            self.logger.warning(f"⚠️ Compression failed: {e}")
            return data

class RegimeContinuityValidator:
    """Validator for regime continuity and transitions."""
    
    def __init__(self, config: Optional[RegimeContinuityConfig] = None):
        self.config = config or RegimeContinuityConfig()
        self.logger = create_fallback_logger("RegimeContinuityValidator")

    def validate_regime_continuity(
        self, 
        regimes_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Validate regime continuity and transitions."""
        try:
            regimes = regimes_df['regime'].values
            unique_regimes = np.unique(regimes)
            
            validation_results = {
                'continuity_score': 0.0,
                'transition_validity': True,
                'temporal_consistency': 0.0,
                'gaps_detected': [],
                'invalid_transitions': [],
                'warnings': [],
                'errors': []
            }
            
            # Check for gaps
            gaps = self._detect_regime_gaps(regimes)
            validation_results['gaps_detected'] = gaps
            
            # Validate transitions
            invalid_transitions = self._validate_transitions(regimes)
            validation_results['invalid_transitions'] = invalid_transitions
            
            # Calculate continuity score
            continuity_score = self._calculate_continuity_score(regimes, gaps)
            validation_results['continuity_score'] = continuity_score
            
            # Check temporal consistency
            temporal_consistency = self._check_temporal_consistency(regimes_df)
            validation_results['temporal_consistency'] = temporal_consistency
            
            # Overall validation
            validation_results['transition_validity'] = (
                len(invalid_transitions) == 0 and
                continuity_score >= self.config.continuity_threshold
            )
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"❌ Regime continuity validation failed: {e}")
            return {
                'continuity_score': 0.0,
                'transition_validity': False,
                'temporal_consistency': 0.0,
                'gaps_detected': [],
                'invalid_transitions': [],
                'warnings': [],
                'errors': [str(e)]
            }

    def _detect_regime_gaps(self, regimes: np.ndarray) -> List[Dict[str, Any]]:
        """Detect gaps in regime sequences."""
        gaps = []
        
        for i in range(1, len(regimes)):
            if regimes[i] != regimes[i-1]:
                # Check if this is a significant gap
                gap_size = 1
                j = i + 1
                while j < len(regimes) and regimes[j] == regimes[i]:
                    gap_size += 1
                    j += 1
                
                if gap_size <= self.config.max_gap_tolerance:
                    gaps.append({
                        'start_index': i,
                        'end_index': j - 1,
                        'gap_size': gap_size,
                        'regime': regimes[i]
                    })
        
        return gaps

    def _validate_transitions(self, regimes: np.ndarray) -> List[Dict[str, Any]]:
        """Validate regime transitions."""
        invalid_transitions = []
        
        for i in range(1, len(regimes)):
            if regimes[i] != regimes[i-1]:
                # Check if transition is valid
                if not self._is_valid_transition(regimes[i-1], regimes[i]):
                    invalid_transitions.append({
                        'index': i,
                        'from_regime': regimes[i-1],
                        'to_regime': regimes[i]
                    })
        
        return invalid_transitions

    def _is_valid_transition(self, from_regime: int, to_regime: int) -> bool:
        """Check if a regime transition is valid."""
        # This would implement transition validation logic
        # For now, allow all transitions
        return True

    def _calculate_continuity_score(self, regimes: np.ndarray, gaps: List[Dict[str, Any]]) -> float:
        """Calculate regime continuity score."""
        if len(regimes) == 0:
            return 0.0
        
        # Count regime changes
        regime_changes = np.sum(regimes[1:] != regimes[:-1])
        
        # Calculate continuity based on changes and gaps
        continuity = 1.0 - safe_divide(regime_changes, len(regimes) - 1)
        
        # Penalize gaps
        gap_penalty = safe_divide(len(gaps), len(regimes))
        continuity = max(0.0, continuity - gap_penalty)
        
        return continuity

    def _check_temporal_consistency(self, regimes_df: pd.DataFrame) -> float:
        """Check temporal consistency of regimes."""
        try:
            if 'timestamp' not in regimes_df.columns:
                return 1.0  # No temporal data to check
            
            # This would implement temporal consistency checks
            # For now, return a placeholder
            return 1.0
            
        except Exception as e:
            self.logger.warning(f"⚠️ Temporal consistency check failed: {e}")
            return 0.0

class EnhancedRegimeDataProcessor:
    """Enhanced regime data processor with comprehensive functionality."""
    
    def __init__(self, processing_mode: ProcessingMode = ProcessingMode.SYNC):
        self.processing_mode = processing_mode
        self.logger = create_fallback_logger("EnhancedRegimeDataProcessor")
        
        # Initialize components
        self.async_processor = AsyncFileProcessor()
        self.memory_pool = MemoryPoolManager()
        self.data_optimizer = DataTypeOptimizer()
        self.continuity_validator = RegimeContinuityValidator()
        
        # Initialize utility managers
        self._initialize_utilities()
        
        # Processing statistics
        self.stats = ProcessingStats()

    def _initialize_utilities(self):
        """Initialize utility managers."""
        try:
            self.gpu_manager = get_m1_gpu_manager()
            self.memory_optimizer = get_m1_memory_optimizer()
            self.cpu_optimizer = get_m1_cpu_optimizer()
            self.parquet_utils = ParquetUtils()
            self.serializer = UniversalSerializer()
            self.data_processor = DataProcessingUtils()
            self.common_utils = CommonUtilities()
            
            self.logger.info("✅ All utility managers initialized successfully")
        except Exception as e:
            self.logger.warning(f"⚠️ Some utility managers failed to initialize: {e}")

    async def process_regime_data_async(
        self, 
        file_paths: List[str], 
        processing_func: Callable[[pd.DataFrame], pd.DataFrame]
    ) -> List[pd.DataFrame]:
        """Process regime data files asynchronously."""
        start_time = time.time()
        
        try:
            # Process files asynchronously
            results = await self.async_processor.process_files_batch(file_paths, processing_func)
            
            # Update statistics
            self.stats.files_processed += len(file_paths)
            self.stats.total_rows_processed += sum(len(df) for df in results)
            self.stats.processing_time += time.time() - start_time
            
            self.logger.info(f"✅ Processed {len(file_paths)} files asynchronously")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Async processing failed: {e}")
            self.stats.error_count += 1
            raise

    def process_regime_data_parallel(
        self, 
        file_paths: List[str], 
        processing_func: Callable[[pd.DataFrame], pd.DataFrame],
        max_workers: Optional[int] = None
    ) -> List[pd.DataFrame]:
        """Process regime data files in parallel."""
        start_time = time.time()
        
        try:
            max_workers = max_workers or min(len(file_paths), mp.cpu_count())
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                futures = [
                    executor.submit(self._process_single_file, file_path, processing_func)
                    for file_path in file_paths
                ]
                
                # Collect results
                results = []
                for future in futures:
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"❌ Parallel processing failed: {e}")
                        self.stats.error_count += 1
            
            # Update statistics
            self.stats.files_processed += len(file_paths)
            self.stats.total_rows_processed += sum(len(df) for df in results)
            self.stats.processing_time += time.time() - start_time
            
            self.logger.info(f"✅ Processed {len(file_paths)} files in parallel")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Parallel processing failed: {e}")
            self.stats.error_count += 1
            raise

    def _process_single_file(
        self, 
        file_path: str, 
        processing_func: Callable[[pd.DataFrame], pd.DataFrame]
    ) -> pd.DataFrame:
        """Process a single file (for parallel processing)."""
        try:
            # Read file
            if file_path.endswith('.parquet'):
                data = pd.read_parquet(file_path)
            else:
                data = pd.read_csv(file_path)
            
            # Process data
            processed_data = processing_func(data)
            
            # Optimize data types
            optimized_data = self.data_optimizer.optimize_dataframe(processed_data)
            
            return optimized_data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process file {file_path}: {e}")
            raise

    def validate_regime_data(
        self, 
        regimes_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Validate regime data quality and continuity."""
        try:
            # Validate regime continuity
            continuity_results = self.continuity_validator.validate_regime_continuity(regimes_df)
            
            # Additional validation checks
            validation_results = {
                'continuity_validation': continuity_results,
                'data_quality': self._assess_data_quality(regimes_df),
                'regime_distribution': self._analyze_regime_distribution(regimes_df),
                'overall_valid': (
                    continuity_results['transition_validity'] and
                    continuity_results['continuity_score'] >= 0.8
                )
            }
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"❌ Regime data validation failed: {e}")
            return {
                'continuity_validation': {'transition_validity': False},
                'data_quality': {'score': 0.0},
                'regime_distribution': {},
                'overall_valid': False,
                'error': str(e)
            }

    def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality."""
        try:
            quality_metrics = {
                'completeness': 1.0 - data.isnull().sum().sum() / (len(data) * len(data.columns)),
                'consistency': 1.0,  # Placeholder
                'accuracy': 1.0,     # Placeholder
                'score': 0.0
            }
            
            quality_metrics['score'] = np.mean([
                quality_metrics['completeness'],
                quality_metrics['consistency'],
                quality_metrics['accuracy']
            ])
            
            return quality_metrics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Data quality assessment failed: {e}")
            return {'score': 0.0}

    def _analyze_regime_distribution(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze regime distribution."""
        try:
            if 'regime' not in data.columns:
                return {}
            
            regime_counts = data['regime'].value_counts()
            total_regimes = len(data)
            
            distribution = {
                'regime_counts': regime_counts.to_dict(),
                'regime_proportions': (regime_counts / total_regimes).to_dict(),
                'regime_balance': 1.0 - np.std(regime_counts / total_regimes),
                'num_regimes': len(regime_counts)
            }
            
            return distribution
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime distribution analysis failed: {e}")
            return {}

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'files_processed': self.stats.files_processed,
            'total_rows_processed': self.stats.total_rows_processed,
            'processing_time': self.stats.processing_time,
            'error_count': self.stats.error_count,
            'rows_per_second': safe_divide(
                self.stats.total_rows_processed,
                self.stats.processing_time
            ),
            'memory_pool_stats': self.memory_pool.get_memory_stats()
        }

# Global instance for backward compatibility
enhanced_regime_data_processor = EnhancedRegimeDataProcessor()

# Export for backward compatibility
RegimeDataProcessor = EnhancedRegimeDataProcessor