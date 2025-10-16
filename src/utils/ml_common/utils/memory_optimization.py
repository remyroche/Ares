from __future__ import annotations

"""
Memory-Efficient ML Training Utilities

This module provides comprehensive memory optimization utilities for large-scale
machine learning training with chunking, streaming, and GPU memory management.

Key Features:
- Data chunking strategies for large datasets
- Streaming feature engineering
- Incremental model training
- Memory-mapped arrays
- GPU memory pool management
- Garbage collection scheduling
- Memory usage monitoring

Built on existing utilities:
- Extends m1_memory_optimizer.py capabilities
- Uses m1_gpu_utils.py for GPU memory management
- Leverages common_operations.py for robust error handling
- Integrates with data_processing_utils.py for data handling
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Iterator, Generator
from datetime import datetime, timedelta
import logging
import gc
import time
import psutil
import os
from pathlib import Path
from contextlib import contextmanager
import tempfile

from ..math_validation import safe_divide
from ..common_operations import create_fallback_logger
from src.utils.logger import system_logger
from src.utils.hardware.m1_gpu_utils import M1GPUManager
from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
from src.utils.common_utilities import safe_dataframe_operation

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - limited GPU memory optimization")

try:
    import dask.array as da
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    logger.warning("Dask not available - limited distributed computing capabilities")

class MemoryEfficientTraining:
    """Memory-efficient ML training utilities."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize memory-efficient training utilities with configuration."""
        self.logger = logger.getChild('MemoryOptimization')
        self.logger.info("🚀 Initializing MemoryEfficientTraining...")
        start_time = time.time()

        self.config = config or {}
        self.logger.info(f"📊 Configuration loaded with {len(self.config)} parameters")

        # Configuration defaults
        self.chunk_size_mb = self.config.get('chunk_size_mb', 500)
        self.max_memory_usage = self.config.get('max_memory_usage', 0.8)  # 80% of available memory
        self.enable_gpu_memory_pool = self.config.get('enable_gpu_memory_pool', True)
        self.gc_interval_seconds = self.config.get('gc_interval_seconds', 300)
        self.temp_dir = self.config.get('temp_dir', tempfile.gettempdir())

        self.logger.info(f"📊 Chunk size: {self.chunk_size_mb} MB")
        self.logger.info(f"📊 Max memory usage: {self.max_memory_usage*100:.1f}%")
        self.logger.info(f"📊 GPU memory pool: {self.enable_gpu_memory_pool}")
        self.logger.info(f"📊 GC interval: {self.gc_interval_seconds}s")

        # Initialize utilities
        self.logger.debug("🔧 Initializing GPU manager...")
        self.gpu_manager = M1GPUManager() if TORCH_AVAILABLE else None
        if self.gpu_manager:
            self.logger.debug("✅ GPU manager initialized")
        else:
            self.logger.debug("ℹ️ GPU manager not initialized (PyTorch not available)")

        self.logger.debug("🔧 Initializing memory optimizer...")
        self.memory_optimizer = M1MemoryOptimizer() if TORCH_AVAILABLE else None
        if self.memory_optimizer:
            self.logger.debug("✅ Memory optimizer initialized")
        else:
            self.logger.debug("ℹ️ Memory optimizer not initialized (PyTorch not available)")

        # Memory tracking
        self.memory_usage_history = []
        self.last_gc_time = datetime.now()
        self.logger.debug("✅ Memory tracking initialized")

        # GPU memory pool (if available)
        self.logger.debug("🔧 Initializing GPU memory pool...")
        self.gpu_memory_pool = self._initialize_gpu_memory_pool()
        if self.gpu_memory_pool:
            self.logger.debug("✅ GPU memory pool initialized")
        else:
            self.logger.debug("ℹ️ GPU memory pool not initialized")

        init_time = time.time() - start_time
        self.logger.info(f"✅ MemoryEfficientTraining initialized in {init_time:.3f}s")

    def memory_checkpoint(self, checkpoint_name: str):
        """Create a memory checkpoint context manager for compatibility with M1 memory optimizer."""

        @contextmanager
        def checkpoint_context():
            try:
                # Log memory usage at checkpoint
                self.logger.debug(f'🧠 Memory checkpoint: {checkpoint_name} - start')
                yield
            finally:
                # Log memory usage after checkpoint
                self.logger.debug(f'🧠 Memory checkpoint: {checkpoint_name} - end')

        return checkpoint_context()

    def data_chunking_strategy(self, X: Union[np.ndarray, pd.DataFrame],
                             y: Optional[Union[np.ndarray, pd.Series]] = None,
                             chunk_size_mb: Optional[int] = None) -> Iterator[Tuple[Any, Optional[Any]]]:
        """
        Create optimal data chunking strategy for large datasets.

        Args:
            X: Feature matrix or DataFrame
            y: Target array or Series (optional)
            chunk_size_mb: Target chunk size in MB

        Yields:
            Tuples of (X_chunk, y_chunk)
        """
        try:
            if chunk_size_mb is None:
                chunk_size_mb = self.chunk_size_mb

            self.logger.info(f"📦 Starting data chunking strategy (chunk_size={chunk_size_mb}MB)")

            # Calculate optimal chunk size
            optimal_chunk_size = self._calculate_optimal_chunk_size(X, chunk_size_mb)

            # Handle different data types
            if isinstance(X, pd.DataFrame):
                yield from self._chunk_pandas_dataframe(X, y, optimal_chunk_size)
            elif isinstance(X, np.ndarray):
                yield from self._chunk_numpy_array(X, y, optimal_chunk_size)
            elif hasattr(X, '__len__') and hasattr(X, '__getitem__'):
                yield from self._chunk_generic_sequence(X, y, optimal_chunk_size)
            else:
                self.logger.warning(f"Unsupported data type for chunking: {type(X)}")
                yield X, y

        except Exception as e:
            self.logger.error(f"❌ Data chunking strategy failed: {e}")
            yield X, y

    def streaming_feature_engineering(self, data_generator: Iterator[Any],
                                    feature_functions: List[Callable],
                                    batch_size: int = 1000) -> Iterator[pd.DataFrame]:
        """
        Perform streaming feature engineering on data streams.

        Args:
            data_generator: Generator yielding data chunks
            feature_functions: List of feature engineering functions
            batch_size: Batch size for processing

        Yields:
            DataFrames with engineered features
        """
        try:
            self.logger.info(f"🌊 Starting streaming feature engineering with {len(feature_functions)} functions")

            for chunk_idx, data_chunk in enumerate(data_generator):
                try:
                    self.logger.debug(f"Processing chunk {chunk_idx}")

                    # Convert to DataFrame if needed
                    if not isinstance(data_chunk, pd.DataFrame):
                        if isinstance(data_chunk, np.ndarray):
                            data_chunk = pd.DataFrame(data_chunk)
                        else:
                            data_chunk = pd.DataFrame([data_chunk])

                    # Apply feature engineering functions
                    engineered_features = []

                    for func in feature_functions:
                        try:
                            feature_result = func(data_chunk)
                            if isinstance(feature_result, pd.DataFrame):
                                engineered_features.append(feature_result)
                            elif isinstance(feature_result, pd.Series):
                                engineered_features.append(feature_result.to_frame())
                            else:
                                # Convert to Series
                                feature_name = getattr(func, '__name__', f'feature_{len(engineered_features)}')
                                engineered_features.append(pd.DataFrame({feature_name: feature_result}))
                        except Exception as func_e:
                            self.logger.warning(f"Feature function failed: {func_e}")
                            continue

                    # Combine original data with engineered features
                    if engineered_features:
                        combined_features = pd.concat([data_chunk] + engineered_features, axis=1)
                    else:
                        combined_features = data_chunk

                    # Memory cleanup
                    self._periodic_memory_cleanup()

                    yield combined_features

                except Exception as chunk_e:
                    self.logger.warning(f"Chunk processing failed: {chunk_e}")
                    yield data_chunk

        except Exception as e:
            self.logger.error(f"❌ Streaming feature engineering failed: {e}")

    def incremental_model_training(self, model_factory: Callable,
                                 data_stream: Iterator[Tuple[np.ndarray, np.ndarray]],
                                 batch_size: int = 1000,
                                 validation_stream: Optional[Iterator[Tuple[np.ndarray, np.ndarray]]] = None) -> Iterator[Dict[str, Any]]:
        """
        Perform incremental model training on data streams.

        Args:
            model_factory: Function that creates new model instances
            data_stream: Stream of (X, y) training data
            batch_size: Batch size for incremental training
            validation_stream: Optional validation data stream

        Yields:
            Training results for each increment
        """
        try:
            self.logger.info("🔄 Starting incremental model training")

            model = None
            training_history = []

            for batch_idx, (X_batch, y_batch) in enumerate(data_stream):
                try:
                    # Create or update model
                    if model is None or batch_idx % 10 == 0:  # Recreate model periodically
                        model = model_factory()

                    # Incremental training
                    if hasattr(model, 'partial_fit'):
                        # Online learning models
                        model.partial_fit(X_batch, y_batch)
                    else:
                        # Standard models - fit on accumulated data
                        model.fit(X_batch, y_batch)

                    # Evaluate current model
                    training_result = {
                        'batch_idx': batch_idx,
                        'samples_processed': (batch_idx + 1) * len(X_batch),
                        'model_type': type(model).__name__,
                        'training_timestamp': datetime.now().isoformat()
                    }

                    # Validation if available
                    if validation_stream is not None:
                        try:
                            val_results = self._evaluate_on_validation_stream(model, validation_stream)
                            training_result.update(val_results)
                        except Exception as val_e:
                            self.logger.warning(f"Validation failed: {val_e}")

                    training_history.append(training_result)

                    # Memory cleanup
                    self._periodic_memory_cleanup()

                    yield training_result

                except Exception as batch_e:
                    self.logger.warning(f"Batch {batch_idx} training failed: {batch_e}")
                    continue

        except Exception as e:
            self.logger.error(f"❌ Incremental model training failed: {e}")

    def memory_mapped_arrays(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                           file_path: Optional[str] = None,
                           mode: str = 'r+') -> Tuple[Any, Optional[Any]]:
        """
        Create memory-mapped arrays for large datasets.

        Args:
            X: Feature array
            y: Target array (optional)
            file_path: Path for memory-mapped file
            mode: File mode ('r', 'r+', 'w+', 'c')

        Returns:
            Tuple of memory-mapped arrays (X_mmap, y_mmap)
        """
        try:
            if file_path is None:
                file_path = os.path.join(self.temp_dir, f'memory_map_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

            self.logger.info(f"🗺️ Creating memory-mapped arrays at {file_path}")

            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Create memory-mapped array for features
            if mode == 'c':  # Copy-on-write mode
                X_mmap = np.memmap(str(file_path) + '_X.npy', dtype=X.dtype, mode=mode, shape=X.shape)
                X_mmap[:] = X[:]
            else:
                # Save array to disk first
                np.save(str(file_path) + '_X.npy', X)
                X_mmap = np.load(str(file_path) + '_X.npy', mmap_mode='r+')

            # Create memory-mapped array for targets
            y_mmap = None
            if y is not None:
                if mode == 'c':
                    y_mmap = np.memmap(str(file_path) + '_y.npy', dtype=y.dtype, mode=mode, shape=y.shape)
                    y_mmap[:] = y[:]
                else:
                    np.save(str(file_path) + '_y.npy', y)
                    y_mmap = np.load(str(file_path) + '_y.npy', mmap_mode='r+')

            self.logger.info(f"✅ Memory-mapped arrays created - X: {X_mmap.shape}, y: {y_mmap.shape if y_mmap is not None else None}")
            return X_mmap, y_mmap

        except Exception as e:
            self.logger.error(f"❌ Memory-mapped array creation failed: {e}")
            return X, y

    def gpu_memory_pool_manager(self, max_memory_gb: float = 8.0) -> 'GPUMemoryPool':
        """
        Create GPU memory pool manager for efficient GPU memory usage.

        Args:
            max_memory_gb: Maximum GPU memory to allocate

        Returns:
            GPU memory pool manager instance
        """
        try:
            if not TORCH_AVAILABLE or not self.gpu_manager:
                self.logger.warning("GPU memory pool not available - PyTorch/M1 GPU manager required")
                return None

            memory_pool = GPUMemoryPool(max_memory_gb, self.gpu_manager)
            self.logger.info(f"🎮 GPU memory pool created (max: {max_memory_gb}GB)")

            return memory_pool

        except Exception as e:
            self.logger.error(f"❌ GPU memory pool creation failed: {e}")
            return None

    def garbage_collection_scheduler(self, gc_interval_seconds: Optional[int] = None) -> None:
        """
        Schedule automatic garbage collection.

        Args:
            gc_interval_seconds: Interval between GC runs
        """
        try:
            if gc_interval_seconds is None:
                gc_interval_seconds = self.gc_interval_seconds

            current_time = datetime.now()
            time_since_last_gc = (current_time - self.last_gc_time).total_seconds()

            if time_since_last_gc >= gc_interval_seconds:
                self.logger.debug("🗑️ Running scheduled garbage collection")

                # Force garbage collection
                collected = gc.collect()

                # Clear memory caches if available
                if hasattr(gc, 'clear_cache'):
                    gc.clear_cache()

                self.last_gc_time = current_time

                self.logger.debug(f"✅ Garbage collection completed - {collected} objects collected")

        except Exception as e:
            self.logger.warning(f"Garbage collection scheduling failed: {e}")

    def memory_usage_monitoring(self, process_name: str = 'ml_training',
                              alert_threshold_mb: int = 1000) -> Dict[str, Any]:
        """
        Monitor memory usage and provide alerts.

        Args:
            process_name: Name for logging
            alert_threshold_mb: Memory threshold for alerts (MB)

        Returns:
            Memory usage statistics
        """
        try:
            # Get current memory usage
            process = psutil.Process()
            memory_info = process.memory_info()

            memory_stats = {
                'rss_mb': memory_info.rss / (1024 * 1024),  # Resident Set Size
                'vms_mb': memory_info.vms / (1024 * 1024),  # Virtual Memory Size
                'timestamp': datetime.now().isoformat(),
                'process_name': process_name
            }

            # Get system memory info
            system_memory = psutil.virtual_memory()
            memory_stats.update({
                'system_total_mb': system_memory.total / (1024 * 1024),
                'system_available_mb': system_memory.available / (1024 * 1024),
                'system_used_percent': system_memory.percent
            })

            # GPU memory info (if available)
            if TORCH_AVAILABLE and torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0)
                memory_stats['gpu_total_mb'] = gpu_memory.total_memory / (1024 * 1024)
                memory_stats['gpu_allocated_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
                memory_stats['gpu_reserved_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)

            # Alert if memory usage is high
            if memory_stats['rss_mb'] > alert_threshold_mb:
                self.logger.warning(f"⚠️ High memory usage alert: {memory_stats['rss_mb']:.1f}MB "
                                  f"(threshold: {alert_threshold_mb}MB)")

            # Store in history
            self.memory_usage_history.append(memory_stats)

            # Keep only recent history
            if len(self.memory_usage_history) > 100:
                self.memory_usage_history = self.memory_usage_history[-50:]

            return memory_stats

        except Exception as e:
            self.logger.error(f"❌ Memory usage monitoring failed: {e}")
            return {'error': str(e)}

    @contextmanager
    def memory_efficient_context(self, max_memory_usage: Optional[float] = None):
        """
        Context manager for memory-efficient operations.

        Args:
            max_memory_usage: Maximum memory usage ratio (0-1)
        """
        if max_memory_usage is None:
            max_memory_usage = self.max_memory_usage

        # Pre-context setup
        initial_memory = self.memory_usage_monitoring('context_start')

        try:
            yield

        finally:
            # Post-context cleanup
            final_memory = self.memory_usage_monitoring('context_end')

            # Force cleanup if memory usage is high
            if final_memory.get('rss_mb', 0) > initial_memory.get('rss_mb', 0) * 1.5:
                self.logger.info("🧹 Context cleanup: high memory usage detected, running GC")
                gc.collect()

    def _calculate_optimal_chunk_size(self, X: Union[np.ndarray, pd.DataFrame],
                                    target_mb: int) -> int:
        """Calculate optimal chunk size based on data characteristics."""
        try:
            # Estimate memory usage per sample
            if isinstance(X, pd.DataFrame):
                sample_size = X.memory_usage(deep=True).sum() / len(X)
            else:
                sample_size = X.nbytes / X.shape[0]

            # Convert target MB to bytes
            target_bytes = target_mb * 1024 * 1024

            # Calculate chunk size
            optimal_size = int(target_bytes / sample_size)

            # Apply bounds
            optimal_size = max(100, min(optimal_size, len(X)))

            self.logger.debug(f"Calculated optimal chunk size: {optimal_size} samples")
            return optimal_size

        except Exception as e:
            self.logger.warning(f"Optimal chunk size calculation failed: {e}")
            return min(1000, len(X) if hasattr(X, '__len__') else 1000)

    def _chunk_pandas_dataframe(self, X: pd.DataFrame, y: Optional[pd.Series],
                              chunk_size: int) -> Iterator[Tuple[pd.DataFrame, Optional[pd.Series]]]:
        """Chunk pandas DataFrame."""
        for start_idx in range(0, len(X), chunk_size):
            end_idx = min(start_idx + chunk_size, len(X))

            X_chunk = X.iloc[start_idx:end_idx]
            y_chunk = y.iloc[start_idx:end_idx] if y is not None else None

            yield X_chunk, y_chunk

    def _chunk_numpy_array(self, X: np.ndarray, y: Optional[np.ndarray],
                          chunk_size: int) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Chunk numpy array."""
        for start_idx in range(0, len(X), chunk_size):
            end_idx = min(start_idx + chunk_size, len(X))

            X_chunk = X[start_idx:end_idx]
            y_chunk = y[start_idx:end_idx] if y is not None else None

            yield X_chunk, y_chunk

    def _chunk_generic_sequence(self, X: Any, y: Optional[Any],
                              chunk_size: int) -> Iterator[Tuple[Any, Optional[Any]]]:
        """Chunk generic sequence."""
        total_length = len(X)
        for start_idx in range(0, total_length, chunk_size):
            end_idx = min(start_idx + chunk_size, total_length)

            X_chunk = X[start_idx:end_idx]
            y_chunk = y[start_idx:end_idx] if y is not None else None

            yield X_chunk, y_chunk

    def _periodic_memory_cleanup(self) -> None:
        """Perform periodic memory cleanup."""
        try:
            # Run garbage collection
            self.garbage_collection_scheduler()

            # Clear any cached data
            if hasattr(self, 'memory_optimizer') and self.memory_optimizer:
                self.memory_optimizer.optimize_memory()

        except Exception as e:
            self.logger.debug(f"Periodic memory cleanup failed: {e}")

    def _evaluate_on_validation_stream(self, model: Any,
                                     validation_stream: Iterator[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Evaluate model on validation stream."""
        try:
            validation_scores = []

            # Evaluate on a few validation batches
            for val_batch_idx, (X_val, y_val) in enumerate(validation_stream):
                if val_batch_idx >= 3:  # Limit validation batches
                    break

                try:
                    if hasattr(model, 'predict_proba') and len(np.unique(y_val)) == 2:
                        y_pred_proba = model.predict_proba(X_val)
                        from sklearn.metrics import roc_auc_score
                        score = roc_auc_score(y_val, y_pred_proba[:, 1])
                    else:
                        y_pred = model.predict(X_val)
                        from sklearn.metrics import accuracy_score
                        score = accuracy_score(y_val, y_pred)

                    validation_scores.append(score)

                except Exception as val_e:
                    self.logger.debug(f"Validation batch {val_batch_idx} failed: {val_e}")
                    continue

            if validation_scores:
                return {
                    'validation_score_mean': np.mean(validation_scores),
                    'validation_score_std': np.std(validation_scores),
                    'validation_batches': len(validation_scores)
                }
            else:
                return {'validation_error': 'No validation scores available'}

        except Exception as e:
            return {'validation_error': str(e)}

    def _initialize_gpu_memory_pool(self) -> Optional['GPUMemoryPool']:
        """Initialize GPU memory pool if available."""
        try:
            if not TORCH_AVAILABLE or not self.enable_gpu_memory_pool:
                return None

            return GPUMemoryPool(8.0, self.gpu_manager)  # 8GB default

        except Exception as e:
            self.logger.warning(f"GPU memory pool initialization failed: {e}")
            return None

class MemoryEfficientProcessor:
    """Memory efficient processor for feature selection operations."""

    def __init__(self):
        """Initialize memory efficient processor."""
        self.logger = system_logger.getChild('MemoryEfficientProcessor')
        self.memory_optimizer = M1MemoryOptimizer()

    def process_dataframe(self, df: pd.DataFrame, operation: str = "optimize") -> pd.DataFrame:
        """Process dataframe with memory optimization."""
        try:
            if operation == "optimize":
                return self.memory_optimizer.optimize_dataframe_memory(df)
            else:
                return df
        except Exception as e:
            self.logger.warning(f"Memory optimization failed: {e}")
            return df

    def batch_process(self, data: List[pd.DataFrame], batch_size: int = 1000) -> List[pd.DataFrame]:
        """Process data in batches for memory efficiency."""
        results = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            processed_batch = [self.process_dataframe(df) for df in batch]
            results.extend(processed_batch)
        return results

class GPUMemoryPool:
    """GPU memory pool manager for efficient memory usage."""

    def __init__(self, max_memory_gb: float, gpu_manager: M1GPUManager):
        """Initialize GPU memory pool."""
        self.max_memory_gb = max_memory_gb
        self.gpu_manager = gpu_manager
        self.allocated_tensors = []
        self.memory_usage = 0.0

    def allocate_tensor(self, shape: Tuple[int, ...], dtype: Any = None) -> Any:
        """
        Allocate tensor from memory pool.

        Args:
            shape: Tensor shape
            dtype: Tensor data type

        Returns:
            Allocated tensor
        """
        try:
            # Default dtype if torch available
            if dtype is None:
                try:
                    import torch as _torch
                    dtype = _torch.float32
                except Exception:
                    dtype = None

            # Check if allocation would exceed limit
            tensor_size_mb = self._calculate_tensor_size_mb(shape, dtype)

            if self.memory_usage + tensor_size_mb > self.max_memory_gb * 1024:
                # Force cleanup
                self.cleanup()

            # Allocate tensor on best available device (cuda > mps > cpu)
            try:
                device = 'cuda' if _torch.cuda.is_available() else ('mps' if _torch.backends.mps.is_available() else 'cpu')
                used_dtype = dtype if dtype is not None else _torch.float32
                tensor = _torch.zeros(shape, dtype=used_dtype, device=device)
            except Exception:
                # Fallback to numpy array when torch unavailable
                tensor = np.zeros(shape, dtype=np.float32)

            # Track allocation
            self.allocated_tensors.append(tensor)
            self.memory_usage += tensor_size_mb

            return tensor

        except Exception as e:
            logger.warning(f"GPU tensor allocation failed: {e}")
            try:
                used_dtype = dtype if dtype is not None else _torch.float32
                return _torch.zeros(shape, dtype=used_dtype)
            except Exception:
                return np.zeros(shape, dtype=np.float32)

    def cleanup(self) -> None:
        """Clean up GPU memory pool."""
        try:
            # Delete all tracked tensors
            for tensor in self.allocated_tensors:
                del tensor

            self.allocated_tensors.clear()

            # Force GPU memory cleanup
            try:
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
                elif _torch.backends.mps.is_available():
                    _torch.mps.empty_cache()
            except Exception:
                pass

            self.memory_usage = 0.0

        except Exception as e:
            logger.warning(f"GPU memory cleanup failed: {e}")

    def _calculate_tensor_size_mb(self, shape: Tuple[int, ...], dtype: Any) -> float:
        """Calculate tensor size in MB."""
        try:
            try:
                element_size = _torch.tensor([], dtype=(dtype or _torch.float32)).element_size()
            except Exception:
                # Fallback to numpy dtype size
                element_size = np.dtype(np.float32).itemsize
            total_elements = np.prod(shape)
            size_bytes = element_size * total_elements
            size_mb = size_bytes / (1024 * 1024)
            return size_mb

        except Exception:
            return 0.0

    def test_method(self):
        """Simple test method to verify file is working."""
        return "test"

    def _get_memory_usage(self) -> float:
        """
        Get current memory usage in GB.

        Returns:
            Current memory usage in GB
        """
        try:
            process = psutil.Process(os.getpid())
            memory_gb = process.memory_info().rss / (1024 ** 3)
            return memory_gb
        except Exception as e:
            logger.warning(f"Could not get memory usage: {e}")
            return 0.0

    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()
