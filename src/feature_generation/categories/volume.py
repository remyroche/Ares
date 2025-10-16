"""
Advanced Volume Feature Generator

This module provides comprehensive volume feature generators for quantitative finance,
including advanced volume analysis, OBV, AD, MFI, VWAP, and volume profile analysis.

Key Features:
- On-Balance Volume (OBV) with VectorBT optimization
- Accumulation/Distribution Line (AD) with advanced metrics
- Money Flow Index (MFI) and related indicators
- Volume Rate of Change and momentum indicators
- Volume-weighted average price (VWAP) with VectorBT
- Volume profile analysis and clustering
- VectorBT-optimized rolling operations
- Memory-efficient processing
-
- Comprehensive volume indicators
"""

import numpy as np
import pandas as pd
import warnings
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory
from ..core.vectorbt_feature_generator import VectorBTFeatureGenerator
from ..core.vectorbt_optimization_mixin import VectorBTOptimizationMixin

# VectorBT imports for optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    from vectorbt.indicators import OBV, AD, MFI, ADOSC, AROONOSC
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    OBV = None
    AD = None
    MFI = None
    ADOSC = None
    AROONOSC = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Import tprint for consistent logging
try:
    from src.utils.tprint import tprint
except ImportError:
    def tprint(*args, **kwargs):
        print(*args, **kwargs)

# VectorBT Scaler removed to avoid circular imports - using direct scaling instead

# PyTorch for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# GPU acceleration removed - CuPy not supported on all platforms
CUPY_AVAILABLE = False

# VectorBT Rolling Optimizer - NOW USING NEW OPTIMIZED VERSION
try:
    from ..utils.consolidated_rolling_optimizer import (
        ConsolidatedRollingOptimizer as VectorBTRollingOptimizer,
        get_global_rolling_optimizer as get_vectorbt_rolling_optimizer,
        RollingOperationConfig,
        RollingOperationType
    )
    from ..utils.statistical_calculations_optimizer import (
        StatisticalCalculationsOptimizer as VectorizationOptimizer,
        get_global_statistical_optimizer as get_vectorization_optimizer,
        StatisticalOperationConfig,
        StatisticalOperationType
    )
    VECTORBT_OPTIMIZER_AVAILABLE = True
    ROLLING_OPTIMIZER_AVAILABLE = True
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    # Fallback to legacy if new version not available
    try:
        from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
        VECTORBT_OPTIMIZER_AVAILABLE = True
        ROLLING_OPTIMIZER_AVAILABLE = True
        OPTIMIZATION_AVAILABLE = False
    except ImportError:
        VECTORBT_OPTIMIZER_AVAILABLE = False
        ROLLING_OPTIMIZER_AVAILABLE = False
        OPTIMIZATION_AVAILABLE = False
        get_vectorbt_rolling_optimizer = None
        VectorBTRollingOptimizer = None

# Unified Vectorization Manager
try:
    from ...utils.ml_common.unified_vectorization_manager import (
        get_unified_vectorization_manager,
        UnifiedVectorizationManager,
        OperationType,
        OptimizationStrategy,
        OperationConfig
    )
    UNIFIED_MANAGER_AVAILABLE = True
except ImportError:
    UNIFIED_MANAGER_AVAILABLE = False
    get_unified_vectorization_manager = None
    UnifiedVectorizationManager = None
    OperationType = None
    OptimizationStrategy = None
    OperationConfig = None

# Optimization utilities
try:
    from ..utils.vectorization_optimizer import get_vectorization_optimizer
    from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

from ..base_calculations import (
    BaseCalculationType,
    create_base_calculator
)

logger = logging.getLogger(__name__)

class VolumeFeatureGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Advanced feature generator for volume-based features with VectorBT optimization."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

        # Initialize VectorBT rolling optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.rolling_optimizer = None

        # Initialize Unified Vectorization Manager
        if UNIFIED_MANAGER_AVAILABLE:
            self.unified_manager = get_unified_vectorization_manager()
        else:
            self.unified_manager = None

        # Performance tracking
        self.performance_stats = {
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'unified_manager_operations': 0,
            'batch_operations': 0,
            'memory_optimizations': 0,
            'gpu_operations': 0,
            'total_operations': 0,
            'total_time': 0.0,
            'average_operation_time': 0.0,
            'peak_memory_usage': 0.0,
            'chunk_operations': 0
        }

        # Performance monitoring
        self.operation_times = []
        self.memory_usage_history = []

        # Memory management
        self.memory_threshold_mb = 512.0  # 512MB threshold
        self.chunk_size = 10000  # Process data in chunks
        self.enable_memory_optimization = True

        # GPU settings
        self.enable_gpu = False  # GPU support disabled by default
        self.gpu_available = self._check_gpu_availability()
        self.gpu_threshold = 50000  # Use GPU for datasets larger than 50k rows

    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="advanced_volume_features",
            category=FeatureCategory.VOLUME,
            description="Advanced volume features with VectorBT optimization including moving averages, ratios, and rate of change",
            required_columns=["volume"],
            optional_columns=["close", "high", "low", "open"],
            default_lookback=20,
            min_lookback=1,
            max_lookback=100,
            parameters={
                "volume_windows": [5, 10, 20, 50],
                "ratio_windows": [10, 20, 50],
                "roc_windows": [1, 5, 10, 20]
            },
            matrix_optimized=True,
            gpu_accelerated=True
        )

    @classmethod
    def create_default(cls) -> 'VolumeFeatureGenerator':
        return cls()

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate comprehensive volume features using VectorBT optimization."""
        if data.empty or 'volume' not in data.columns:
            return pd.Series(dtype=float, index=data.index, name='volume_features')

        # Optimize DataFrame for processing and memory usage
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        # Apply memory optimization
        data = self._optimize_memory_usage(data)

        volume = data['volume']

        # Use Unified Vectorization Manager for intelligent optimization
        if self.unified_manager and self._should_use_unified_manager(volume):
            try:
                # Use Unified Vectorization Manager for optimized rolling operations
                volume_result = self.unified_manager.optimize_operation(
                    OperationType.TECHNICAL_INDICATORS,
                    {
                        'data': volume,
                        'operation': 'rolling_mean',
                        'window': 20,
                        'indicator_configs': {'rolling_mean': {'window': 20}}
                    },
                    OperationConfig(
                        operation_type=OperationType.TECHNICAL_INDICATORS,
                        data_size=len(volume),
                        data_dimensions=volume.shape,
                        memory_budget_mb=256.0
                    )
                )
                volume_sma = volume_result.result
                self.performance_stats['unified_manager_operations'] += 1
                return volume_sma
            except Exception as e:
                logger.warning(f"Unified Vectorization Manager failed: {e}, using VectorBT fallback")
                self.performance_stats['pandas_fallbacks'] += 1

        # Use VectorBT rolling optimizer if available
        if self.rolling_optimizer and self._should_use_vectorbt(volume):
            try:
                # Calculate volume SMA using VectorBT rolling optimizer
                volume_sma = self.rolling_optimizer.rolling_mean(volume, window=20)
                self.performance_stats['vectorbt_operations'] += 1
                return volume_sma
            except Exception as e:
                logger.warning(f"VectorBT rolling optimizer failed: {e}, using fallback")
                self.performance_stats['pandas_fallbacks'] += 1

        # Fallback to VectorBT direct operations
        if VECTORBT_AVAILABLE:
            try:
                volume_sma = rolling_mean(volume, window=20)
                self.performance_stats['vectorbt_operations'] += 1
                return volume_sma
            except Exception as e:
                logger.warning(f"VectorBT volume calculation failed: {e}, using pandas fallback")
                self.performance_stats['pandas_fallbacks'] += 1

        # Final fallback to pandas
        return volume.rolling(window=20).mean()

# Volume Simple Moving Average

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (VECTORBT_AVAILABLE and
                len(data) >= getattr(self, 'vectorbt_threshold', 1000))

    def _should_use_unified_manager(self, data) -> bool:
        """Determine if Unified Vectorization Manager should be used."""
        return (UNIFIED_MANAGER_AVAILABLE and
                len(data) >= getattr(self, 'unified_manager_threshold', 5000))

    def _generate_with_unified_manager(self, volume: pd.Series, data: pd.DataFrame) -> pd.Series:
        """Generate volume features using Unified Vectorization Manager."""
        try:
            # Use the unified manager's rolling operation method for better optimization
            volume_sma = self.unified_manager.rolling_operation(volume, 'mean', window=20)

            self.performance_stats['unified_manager_operations'] += 1
            return volume_sma

        except Exception as e:
            logger.warning(f"Unified manager volume generation failed: {e}")
            # Fallback to VectorBT rolling optimizer
            if self.rolling_optimizer:
                return self.rolling_optimizer.rolling_mean(volume, window=20)
            else:
                return volume.rolling(window=20).mean()

    def generate_batch_volume_features(self, data: pd.DataFrame, feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Generate multiple volume features in batch using VectorBT optimization.

        Args:
            data: OHLCV data
            feature_configs: List of feature configuration dictionaries

        Returns:
            DataFrame with generated features
        """
        if data.empty or 'volume' not in data.columns:
            return pd.DataFrame(index=data.index)

        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        results = {}
        volume = data['volume']

        # Use Unified Vectorization Manager for batch processing if available
        if self.unified_manager and self._should_use_unified_manager(volume):
            try:
                # Use Unified Vectorization Manager for batch processing
                batch_result = self.unified_manager.optimize_operation(
                    OperationType.FEATURE_ENGINEERING,
                    {
                        'data': data,
                        'feature_configs': feature_configs,
                        'operation_type': 'volume_batch'
                    },
                    OperationConfig(
                        operation_type=OperationType.FEATURE_ENGINEERING,
                        data_size=len(data),
                        data_dimensions=data.shape,
                        memory_budget_mb=1024.0
                    )
                )
                return batch_result.result
            except Exception as e:
                logger.warning(f"Unified manager batch processing failed: {e}, using VectorBT fallback")
                self.performance_stats['pandas_fallbacks'] += 1

        # Use VectorBT rolling optimizer for batch processing
        if self.rolling_optimizer and self._should_use_vectorbt(volume):
            try:
                return self._generate_batch_with_vectorbt(data, feature_configs)
            except Exception as e:
                logger.warning(f"VectorBT batch processing failed: {e}, using individual processing")
                self.performance_stats['pandas_fallbacks'] += 1

        # Fallback to individual processing
        for config in feature_configs:
            feature_name = config.get('name', 'volume_feature')
            feature_type = config.get('type', 'sma')
            period = config.get('period', 20)

            try:
                if feature_type == 'sma':
                    result = volume.rolling(window=period).mean()
                elif feature_type == 'ema':
                    result = volume.ewm(span=period).mean()
                elif feature_type == 'std':
                    result = volume.rolling(window=period).std()
                elif feature_type == 'var':
                    result = volume.rolling(window=period).var()
                elif feature_type == 'min':
                    result = volume.rolling(window=period).min()
                elif feature_type == 'max':
                    result = volume.rolling(window=period).max()
                elif feature_type == 'sum':
                    result = volume.rolling(window=period).sum()
                else:
                    logger.warning(f"Unknown feature type: {feature_type}")
                    continue

                results[feature_name] = result

            except Exception as e:
                logger.warning(f"Feature {feature_name} generation failed: {e}")
                continue

        return pd.DataFrame(results, index=data.index)

    def _generate_batch_with_unified_manager(self, data: pd.DataFrame, feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Generate batch features using Unified Vectorization Manager."""
        try:
            # Configure operation for batch volume feature generation
            operation_config = {
                'operation_type': OperationType.TECHNICAL_INDICATORS,
                'data_size': len(data),
                'data_dimensions': (len(data), len(feature_configs)),
                'memory_budget_mb': 1024.0,
                'time_budget_seconds': 120.0,
                'precision_requirement': 'medium'
            }

            # Create batch operation function
            def batch_operation():
                return self._generate_batch_with_vectorbt(data, feature_configs)

            # Use unified manager for batch processing
            result = self.unified_manager.optimize_operation(batch_operation, operation_config)

            self.performance_stats['unified_manager_operations'] += 1
            self.performance_stats['batch_operations'] += 1
            return result.result

        except Exception as e:
            logger.warning(f"Unified manager batch generation failed: {e}")
            raise

    def _generate_batch_with_vectorbt(self, data: pd.DataFrame, feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Generate batch features using VectorBT rolling optimizer."""
        results = {}
        volume = data['volume']

        for config in feature_configs:
            feature_name = config.get('name', 'volume_feature')
            feature_type = config.get('type', 'sma')
            period = config.get('period', 20)

            try:
                if feature_type == 'sma':
                    result = self.rolling_optimizer.rolling_mean(volume, window=period)
                elif feature_type == 'std':
                    result = self.rolling_optimizer.rolling_std(volume, window=period)
                elif feature_type == 'var':
                    result = self.rolling_optimizer.rolling_var(volume, window=period)
                elif feature_type == 'min':
                    result = self.rolling_optimizer.rolling_min(volume, window=period)
                elif feature_type == 'max':
                    result = self.rolling_optimizer.rolling_max(volume, window=period)
                elif feature_type == 'sum':
                    result = self.rolling_optimizer.rolling_sum(volume, window=period)
                else:
                    logger.warning(f"Unknown feature type: {feature_type}")
                    continue

                results[feature_name] = result
                self.performance_stats['vectorbt_operations'] += 1

            except Exception as e:
                logger.warning(f"VectorBT feature {feature_name} generation failed: {e}")
                continue

        self.performance_stats['batch_operations'] += 1
        return pd.DataFrame(results, index=data.index)

    def _optimize_memory_usage(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage for large datasets."""
        if not self.enable_memory_optimization:
            return data

        try:
            # Calculate memory usage
            memory_usage_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)

            if memory_usage_mb > self.memory_threshold_mb:
                logger.info(f"Memory usage {memory_usage_mb:.2f}MB exceeds threshold {self.memory_threshold_mb}MB, optimizing...")

                # Optimize data types
                optimized_data = data.copy()

                for column in optimized_data.columns:
                    if optimized_data[column].dtype == 'float64':
                        # Use float32 if precision allows
                        if (optimized_data[column].min() >= np.finfo(np.float32).min and
                            optimized_data[column].max() <= np.finfo(np.float32).max):
                            optimized_data[column] = optimized_data[column].astype(np.float32)

                    elif optimized_data[column].dtype == 'int64':
                        # Use int32 if range allows
                        if (optimized_data[column].min() >= np.iinfo(np.int32).min and
                            optimized_data[column].max() <= np.iinfo(np.int32).max):
                            optimized_data[column] = optimized_data[column].astype(np.int32)

                # Calculate new memory usage
                new_memory_usage_mb = optimized_data.memory_usage(deep=True).sum() / (1024 * 1024)
                memory_saved_mb = memory_usage_mb - new_memory_usage_mb

                logger.info(f"Memory optimization saved {memory_saved_mb:.2f}MB ({memory_saved_mb/memory_usage_mb*100:.1f}%)")
                self.performance_stats['memory_optimizations'] += 1

                return optimized_data

            return data

        except Exception as e:
            logger.warning(f"Memory optimization failed: {e}")
            return data

    def _process_in_chunks(self, data: pd.DataFrame, operation_func: Callable,
                          chunk_size: Optional[int] = None) -> pd.DataFrame:
        """Process large datasets in chunks to manage memory usage."""
        if chunk_size is None:
            chunk_size = self.chunk_size

        if len(data) <= chunk_size:
            return operation_func(data)

        try:
            logger.info(f"Processing {len(data)} rows in chunks of {chunk_size}")

            results = []
            for i in range(0, len(data), chunk_size):
                chunk = data.iloc[i:i + chunk_size]
                chunk_result = operation_func(chunk)
                results.append(chunk_result)

                # Force garbage collection after each chunk
                import gc
                gc.collect()

            # Combine results
            final_result = pd.concat(results, ignore_index=False)
            self.performance_stats['memory_optimizations'] += 1

            return final_result

        except Exception as e:
            logger.warning(f"Chunk processing failed: {e}, falling back to full processing")
            return operation_func(data)

    def _should_use_chunking(self, data: pd.DataFrame) -> bool:
        """Determine if chunking should be used based on data size and memory."""
        if not self.enable_memory_optimization:
            return False

        memory_usage_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
        return memory_usage_mb > self.memory_threshold_mb or len(data) > self.chunk_size * 2

    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available for processing."""
        if not self.enable_gpu:
            return False

        try:
            # Check for GPU availability
            if False:  # GPU support removed
                return True

            # Check for PyTorch with CUDA support
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return True

            return False
        except Exception as e:
            logger.warning(f"GPU availability check failed: {e}")
            return False

    def _should_use_gpu(self, data: pd.DataFrame) -> bool:
        """Determine if GPU should be used based on data size and availability."""
        return (self.gpu_available and
                len(data) >= self.gpu_threshold and
                self.enable_gpu)

    def _convert_to_gpu(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert DataFrame to GPU memory if available."""
        if not self._should_use_gpu(data):
            return data

        try:
            if False:  # GPU support removed
                # Convert to GPU arrays
                gpu_data = data.copy()
                for column in gpu_data.columns:
                    if gpu_data[column].dtype in ['float32', 'float64', 'int32', 'int64']:
                        gpu_data[column] = np.asarray(gpu_data[column].values)
                return gpu_data
            elif TORCH_AVAILABLE and torch.cuda.is_available():
                # Convert to PyTorch tensors on GPU
                gpu_data = data.copy()
                for column in gpu_data.columns:
                    if gpu_data[column].dtype in ['float32', 'float64', 'int32', 'int64']:
                        gpu_data[column] = torch.tensor(gpu_data[column].values, device='cuda')
                return gpu_data
            else:
                return data
        except Exception as e:
            logger.warning(f"GPU conversion failed: {e}")
            return data

    def _convert_from_gpu(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert DataFrame from GPU memory back to CPU."""
        try:
            if False:  # GPU support removed
                # Convert from
                cpu_data = data.copy()
                for column in cpu_data.columns:
                    if hasattr(cpu_data[column].iloc[0], 'get'):  #
                        cpu_data[column] = cpu_data[column].apply(lambda x: x.get() if hasattr(x, 'get') else x)
                return cpu_data
            elif TORCH_AVAILABLE and torch.cuda.is_available():
                # Convert from PyTorch tensors
                cpu_data = data.copy()
                for column in cpu_data.columns:
                    if hasattr(cpu_data[column].iloc[0], 'cpu'):  # PyTorch tensor
                        cpu_data[column] = cpu_data[column].apply(lambda x: x.cpu().numpy() if hasattr(x, 'cpu') else x)
                return cpu_data
            else:
                return data
        except Exception as e:
            logger.warning(f"GPU to CPU conversion failed: {e}")
            return data

    def _monitor_operation(self, operation_name: str, operation_func: Callable, *args, **kwargs):
        """Monitor operation performance and memory usage."""
        import time
        import psutil
        import os

        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # MB

        try:
            result = operation_func(*args, **kwargs)

            end_time = time.time()
            end_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # MB

            operation_time = end_time - start_time
            memory_used = end_memory - start_memory

            # Update performance stats
            self.performance_stats['total_operations'] += 1
            self.performance_stats['total_time'] += operation_time
            self.performance_stats['average_operation_time'] = (
                self.performance_stats['total_time'] / self.performance_stats['total_operations']
            )
            self.performance_stats['peak_memory_usage'] = max(
                self.performance_stats['peak_memory_usage'],
                end_memory
            )

            # Store operation details
            self.operation_times.append({
                'operation': operation_name,
                'time': operation_time,
                'memory_used': memory_used,
                'timestamp': start_time
            })

            self.memory_usage_history.append({
                'timestamp': start_time,
                'memory_mb': end_memory
            })

            # Log performance if significant
            if operation_time > 1.0:  # Log operations taking more than 1 second
                logger.info(f"Operation '{operation_name}' took {operation_time:.2f}s, used {memory_used:.2f}MB")

            return result

        except Exception as e:
            end_time = time.time()
            operation_time = end_time - start_time
            logger.error(f"Operation '{operation_name}' failed after {operation_time:.2f}s: {e}")
            raise

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            'performance_stats': self.performance_stats.copy(),
            'recent_operations': self.operation_times[-10:] if self.operation_times else [],
            'memory_usage_trend': self.memory_usage_history[-20:] if self.memory_usage_history else [],
            'optimization_effectiveness': self._calculate_optimization_effectiveness()
        }

        return summary

    def _calculate_optimization_effectiveness(self) -> Dict[str, float]:
        """Calculate effectiveness of different optimization strategies."""
        total_ops = self.performance_stats['total_operations']
        if total_ops == 0:
            return {}

        effectiveness = {
            'vectorbt_usage_rate': self.performance_stats['vectorbt_operations'] / total_ops,
            'unified_manager_usage_rate': self.performance_stats['unified_manager_operations'] / total_ops,
            'batch_processing_rate': self.performance_stats['batch_operations'] / total_ops,
            'memory_optimization_rate': self.performance_stats['memory_optimizations'] / total_ops,
            'gpu_usage_rate': self.performance_stats['gpu_operations'] / total_ops,
            'fallback_rate': self.performance_stats['pandas_fallbacks'] / total_ops
        }

        return effectiveness

    def reset_performance_stats(self):
        """Reset all performance statistics."""
        self.performance_stats = {
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'unified_manager_operations': 0,
            'batch_operations': 0,
            'memory_optimizations': 0,
            'gpu_operations': 0,
            'total_operations': 0,
            'total_time': 0.0,
            'average_operation_time': 0.0,
            'peak_memory_usage': 0.0,
            'chunk_operations': 0
        }
        self.operation_times.clear()
        self.memory_usage_history.clear()

    def log_performance_report(self):
        """Log a comprehensive performance report."""
        summary = self.get_performance_summary()

        logger.info("=== Volume Feature Generation Performance Report ===")
        logger.info(f"Total Operations: {summary['performance_stats']['total_operations']}")
        logger.info(f"Total Time: {summary['performance_stats']['total_time']:.2f}s")
        logger.info(f"Average Operation Time: {summary['performance_stats']['average_operation_time']:.3f}s")
        logger.info(f"Peak Memory Usage: {summary['performance_stats']['peak_memory_usage']:.2f}MB")

        effectiveness = summary['optimization_effectiveness']
        logger.info("=== Optimization Effectiveness ===")
        logger.info(f"VectorBT Usage Rate: {effectiveness.get('vectorbt_usage_rate', 0):.1%}")
        logger.info(f"Unified Manager Usage Rate: {effectiveness.get('unified_manager_usage_rate', 0):.1%}")
        logger.info(f"Batch Processing Rate: {effectiveness.get('batch_processing_rate', 0):.1%}")
        logger.info(f"Memory Optimization Rate: {effectiveness.get('memory_optimization_rate', 0):.1%}")
        logger.info(f"GPU Usage Rate: {effectiveness.get('gpu_usage_rate', 0):.1%}")
        logger.info(f"Fallback Rate: {effectiveness.get('fallback_rate', 0):.1%}")

        logger.info("=== Recent Operations ===")
        for op in summary['recent_operations'][-5:]:  # Show last 5 operations
            logger.info(f"  {op['operation']}: {op['time']:.3f}s, {op['memory_used']:.2f}MB")

    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

        try:
            if operation == 'mean':
                return self.vectorbt_optimizer.rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return self.vectorbt_optimizer.rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return self.vectorbt_optimizer.rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return self.vectorbt_optimizer.rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return self.vectorbt_optimizer.rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return self.vectorbt_optimizer.rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            self.logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

    def _pandas_rolling_operation(self, data: pd.Series, operation: str,
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return self._optimized_rolling_operation(data, "mean", window)
        elif operation == 'std':
            return self._optimized_rolling_operation(data, "std", window)
        elif operation == 'var':
            return self._optimized_rolling_operation(data, "var", window)
        elif operation == 'min':
            return self._optimized_rolling_operation(data, "min", window)
        elif operation == 'max':
            return self._optimized_rolling_operation(data, "max", window)
        elif operation == 'sum':
            return self._optimized_rolling_operation(data, "sum", window)
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def generate_volume_indicators_batch(self, data: pd.DataFrame,
                                       sma_windows: List[int] = None,
                                       ema_windows: List[int] = None,
                                       ratio_windows: List[int] = None,
                                       roc_windows: List[int] = None) -> pd.DataFrame:
        """
        Generate comprehensive volume indicators in batch.

        Args:
            data: OHLCV data
            sma_windows: SMA window sizes (default: [5, 10, 20, 50])
            ema_windows: EMA window sizes (default: [12, 26])
            ratio_windows: Volume ratio window sizes (default: [5, 10, 20])
            roc_windows: Rate of change window sizes (default: [5, 10, 20])

        Returns:
            DataFrame with all volume indicators
        """
        if sma_windows is None:
            sma_windows = [5, 10, 20, 50]
        if ema_windows is None:
            ema_windows = [12, 26]
        if ratio_windows is None:
            ratio_windows = [5, 10, 20]
        if roc_windows is None:
            roc_windows = [5, 10, 20]

        feature_configs = []

        # Add volume SMA features
        for window in sma_windows:
            feature_configs.append({
                'name': f'volume_sma_{window}',
                'type': 'rolling',
                'params': {'operation': 'mean', 'window': window, 'column': 'volume'}
            })

        # Add volume EMA features (custom implementation)
        for window in ema_windows:
            feature_configs.append({
                'name': f'volume_ema_{window}',
                'type': 'custom',
                'params': {
                    'function': lambda df, w=window: df['volume'].ewm(span=w).mean(),
                    'window': window
                }
            })

        # Add volume ratio features
        for window in ratio_windows:
            feature_configs.append({
                'name': f'volume_ratio_{window}',
                'type': 'custom',
                'params': {
                    'function': lambda df, w=window: df['volume'] / df['volume'].rolling(w).mean(),
                    'window': window
                }
            })

        # Add volume ROC features
        for window in roc_windows:
            feature_configs.append({
                'name': f'volume_roc_{window}',
                'type': 'custom',
                'params': {
                    'function': lambda df, w=window: df['volume'].pct_change(w),
                    'window': window
                }
            })

        # Process all features using UnifiedVectorizationManager
        if self.unified_manager:
            try:
                return self.unified_manager.batch_process_features(data, feature_configs)
            except Exception as e:
                self.logger.warning(f"Unified manager batch processing failed: {e}, using individual processing")

        # Fallback to individual processing
        return self._process_volume_features_individually(data, feature_configs)

    def generate_volume_correlation_features_batch(self, data: pd.DataFrame,
                                                 windows: List[int] = None,
                                                 price_columns: List[str] = None) -> pd.DataFrame:
        """
        Generate volume-price correlation features in batch.

        Args:
            data: OHLCV data
            windows: Correlation window sizes (default: [10, 20, 50])
            price_columns: Price columns to correlate with volume (default: ['close', 'high', 'low'])

        Returns:
            DataFrame with volume correlation features
        """
        if windows is None:
            windows = [10, 20, 50]
        if price_columns is None:
            price_columns = ['close', 'high', 'low']

        feature_configs = []

        for window in windows:
            for price_col in price_columns:
                if price_col in data.columns:
                    feature_configs.append({
                        'name': f'volume_{price_col}_corr_{window}',
                        'type': 'rolling',
                        'params': {
                            'operation': 'corr',
                            'window': window,
                            'column': 'volume',
                            'other_column': price_col
                        }
                    })

        # Process using UnifiedVectorizationManager
        if self.unified_manager:
            try:
                return self.unified_manager.batch_process_features(data, feature_configs)
            except Exception as e:
                self.logger.warning(f"Unified manager correlation processing failed: {e}, using individual processing")

        # Fallback to individual processing
        return self._process_volume_features_individually(data, feature_configs)

    def generate_vwap_features_batch(self, data: pd.DataFrame,
                                   vwap_windows: List[int] = None) -> pd.DataFrame:
        """
        Generate VWAP-related features in batch.

        Args:
            data: OHLCV data
            vwap_windows: VWAP window sizes (default: [20, 50])

        Returns:
            DataFrame with VWAP features
        """
        if vwap_windows is None:
            vwap_windows = [20, 50]

        feature_configs = []

        for window in vwap_windows:
            # VWAP calculation
            feature_configs.append({
                'name': f'vwap_{window}',
                'type': 'custom',
                'params': {
                    'function': lambda df, w=window: self._calculate_vwap(df, w),
                    'window': window
                }
            })

            # VWAP deviation
            feature_configs.append({
                'name': f'vwap_deviation_{window}',
                'type': 'custom',
                'params': {
                    'function': lambda df, w=window: self._calculate_vwap_deviation(df, w),
                    'window': window
                }
            })

        # Process using UnifiedVectorizationManager
        if self.unified_manager:
            try:
                return self.unified_manager.batch_process_features(data, feature_configs)
            except Exception as e:
                self.logger.warning(f"Unified manager VWAP processing failed: {e}, using individual processing")

        # Fallback to individual processing
        return self._process_volume_features_individually(data, feature_configs)

    def _calculate_vwap(self, data: pd.DataFrame, window: int) -> pd.Series:
        """Calculate VWAP for given window."""
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        volume = data['volume'].values

        vwap = np.full(len(data), np.nan)

        for i in range(window - 1, len(data)):
            window_high = high[i - window + 1:i + 1]
            window_low = low[i - window + 1:i + 1]
            window_close = close[i - window + 1:i + 1]
            window_volume = volume[i - window + 1:i + 1]

            typical_price = (window_high + window_low + window_close) / 3
            vwap[i] = np.sum(typical_price * window_volume) / np.sum(window_volume)

        return pd.Series(vwap, index=data.index)

    def _calculate_vwap_deviation(self, data: pd.DataFrame, window: int) -> pd.Series:
        """Calculate VWAP deviation for given window."""
        vwap = self._calculate_vwap(data, window)
        return (data['close'] - vwap) / vwap

    def _process_volume_features_individually(self, data: pd.DataFrame, feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process volume features individually as fallback."""
        results = {}

        for config in feature_configs:
            feature_name = config['name']
            feature_type = config.get('type', 'rolling')
            params = config.get('params', {})

            try:
                if feature_type == 'rolling':
                    operation = params.get('operation', 'mean')
                    window = params.get('window', 20)
                    column = params.get('column', 'volume')
                    other_column = params.get('other_column')

                    if column in data.columns:
                        series_data = data[column]

                        if operation == 'corr' and other_column and other_column in data.columns:
                            results[feature_name] = self.rolling_optimizer.rolling_corr(
                                series_data, data[other_column], window
                            )
                        else:
                            # Use VectorBTRollingOptimizer for other operations
                            if hasattr(self.rolling_optimizer, f'rolling_{operation}'):
                                results[feature_name] = getattr(self.rolling_optimizer, f'rolling_{operation}')(
                                    series_data, window
                                )
                            else:
                                # Fallback to pandas
                                rolling_obj = series_data.rolling(window=window)
                                results[feature_name] = getattr(rolling_obj, operation)()

                elif feature_type == 'custom':
                    func = params.get('function')
                    if callable(func):
                        results[feature_name] = func(data)
                    else:
                        results[feature_name] = pd.Series(np.nan, index=data.index)

            except Exception as e:
                self.logger.warning(f"Volume feature {feature_name} failed: {e}")
                results[feature_name] = pd.Series(np.nan, index=data.index)

        return pd.DataFrame(results, index=data.index)

class VolumeSMAGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for Volume Simple Moving Average with VectorBT optimization."""

    def __init__(self, period: int = 20):
        config = FeatureConfig(
            name=f"volume_sma_{period}",
            category=FeatureCategory.VOLUME,
            description=f"Volume Simple Moving Average over {period} periods with VectorBT optimization",
            required_columns=["volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={'period': period},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period

        # Initialize VectorBT rolling optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.rolling_optimizer = None

        # Initialize Unified Vectorization Manager
        if UNIFIED_MANAGER_AVAILABLE:
            self.unified_manager = get_unified_vectorization_manager()
        else:
            self.unified_manager = None

        # Performance tracking
        self.performance_stats = {
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'unified_manager_operations': 0
        }

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volume SMA using VectorBT optimization."""
        if data.empty or 'volume' not in data.columns:
            return pd.Series(dtype=float, index=data.index, name=f'volume_sma_{self.period}')

        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        volume = data['volume']

        # Use Unified Vectorization Manager for intelligent optimization
        if self.unified_manager and self._should_use_unified_manager(volume):
            try:
                return self._generate_sma_with_unified_manager(volume)
            except Exception as e:
                logger.warning(f"Unified Vectorization Manager failed: {e}, using VectorBT fallback")
                self.performance_stats['pandas_fallbacks'] += 1

        # Use VectorBT rolling optimizer if available
        if self.rolling_optimizer and self._should_use_vectorbt(volume):
            try:
                volume_sma = self.rolling_optimizer.rolling_mean(volume, window=self.period)
                self.performance_stats['vectorbt_operations'] += 1
                return volume_sma
            except Exception as e:
                logger.warning(f"VectorBT rolling optimizer failed: {e}, using fallback")
                self.performance_stats['pandas_fallbacks'] += 1

        # Fallback to VectorBT direct operations
        if VECTORBT_AVAILABLE:
            try:
                volume_sma = rolling_mean(volume, window=self.period)
                self.performance_stats['vectorbt_operations'] += 1
                return volume_sma
            except Exception as e:
                logger.warning(f"VectorBT volume SMA calculation failed: {e}, using pandas fallback")
                self.performance_stats['pandas_fallbacks'] += 1

        # Final fallback to pandas
        return volume.rolling(window=self.period).mean()

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (VECTORBT_AVAILABLE and
                len(data) >= getattr(self, 'vectorbt_threshold', 1000))

    def _should_use_unified_manager(self, data) -> bool:
        """Determine if Unified Vectorization Manager should be used."""
        return (UNIFIED_MANAGER_AVAILABLE and
                len(data) >= getattr(self, 'unified_manager_threshold', 5000))

    def _generate_sma_with_unified_manager(self, volume: pd.Series) -> pd.Series:
        """Generate SMA using Unified Vectorization Manager."""
        try:
            # Configure operation for volume SMA calculation
            operation_config = {
                'operation_type': OperationType.TECHNICAL_INDICATORS,
                'data_size': len(volume),
                'data_dimensions': (len(volume),),
                'memory_budget_mb': 256.0,
                'time_budget_seconds': 30.0,
                'precision_requirement': 'medium'
            }

            # Use unified manager for rolling mean calculation
            result = self.unified_manager.optimize_operation(
                lambda: self.rolling_optimizer.rolling_mean(volume, window=self.period),
                operation_config
            )

            self.performance_stats['unified_manager_operations'] += 1
            return result.result

        except Exception as e:
            logger.warning(f"Unified manager SMA generation failed: {e}")
            raise

# Volume Exponential Moving Average

class VolumeEMAGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for Volume Exponential Moving Average with VectorBT optimization."""

    def __init__(self, period: int = 20, alpha: Optional[float] = None):
        if alpha is None:
            alpha = 2.0 / (period + 1)

        config = FeatureConfig(
            name=f"volume_ema_{period}",
            category=FeatureCategory.VOLUME,
            description=f"Volume Exponential Moving Average over {period} periods with VectorBT optimization",
            required_columns=["volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={'period': period, 'alpha': alpha},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.alpha = alpha

        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_optimizer = None
        # Initialize VectorBT rolling optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.rolling_optimizer = None

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volume EMA using VectorBT optimization."""
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        volume = data['volume']

        # Use VectorBT for volume EMA calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(volume):
            return self._optimized_rolling_operation(volume, "mean", self.period)
        else:
            return self._optimized_rolling_operation(volume, "mean", self.period)

        if data.empty or 'volume' not in data.columns:
            return pd.Series(dtype=float, index=data.index, name=f'volume_ema_{self.period}')

        volume = data['volume']

        # Use VectorBT rolling optimizer for EMA calculation
        if self.rolling_optimizer:
            try:
                # Use VectorBT rolling apply for EMA calculation
                def ema_func(x):
                    return x.ewm(alpha=self.alpha, adjust=False).mean().iloc[-1]

                volume_ema = self.rolling_optimizer.rolling_apply(volume, window=self.period, func=ema_func)
                self.performance_stats['vectorbt_operations'] += 1
                return volume_ema
            except Exception as e:
                self.logger.warning(f"VectorBT rolling optimizer failed: {e}, using fallback")
                self.performance_stats['pandas_fallbacks'] += 1

        # Fallback to pandas EMA (VectorBT doesn't have direct EMA support)
        try:
            volume_ema = volume.ewm(alpha=self.alpha, adjust=False).mean()
            self.performance_stats['pandas_fallbacks'] += 1
            return volume_ema
        except Exception as e:
            self.logger.warning(f"Volume EMA calculation failed: {e}")
            return pd.Series(np.nan, index=data.index, name=f'volume_ema_{self.period}')

# Volume Ratio

class VolumeRatioGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for Volume Ratio (current volume vs average volume) with VectorBT optimization."""

    def __init__(self, period: int = 20):
        config = FeatureConfig(
            name=f"volume_ratio_{period}",
            category=FeatureCategory.VOLUME,
            description=f"Volume ratio (current volume / average volume) over {period} periods with VectorBT optimization",
            required_columns=["volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={'period': period},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period

        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_optimizer = None
        # Initialize VectorBT rolling optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.rolling_optimizer = None

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volume Ratio using VectorBT optimization."""
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        volume = data['volume']

        # Use VectorBT for volume ratio calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(volume):
            try:
                avg_volume = self.vectorbt_optimizer.rolling_mean(volume, window=self.period)
                # Use safe division - replace zeros and NaN with forward fill, then small positive number
                avg_volume_safe = avg_volume.replace(0, np.nan).fillna(method='ffill').fillna(volume.mean())
                return volume / avg_volume_safe
            except Exception as e:
                # Fallback to pandas
                avg_volume = volume.rolling(window=self.period).mean()
                # Use safe division - replace zeros and NaN with forward fill, then small positive number
                avg_volume_safe = avg_volume.replace(0, np.nan).fillna(method='ffill').fillna(volume.mean())
                return volume / avg_volume_safe
        else:
            # Fallback to pandas
            avg_volume = volume.rolling(window=self.period).mean()
            # Use safe division - replace zeros and NaN with forward fill, then small positive number
            avg_volume_safe = avg_volume.replace(0, np.nan).fillna(method='ffill').fillna(volume.mean())
            return volume / avg_volume_safe

        if data.empty or 'volume' not in data.columns:
            return pd.Series(dtype=float, index=data.index, name=f'volume_ratio_{self.period}')

        volume = data['volume']

        # Use VectorBT rolling optimizer if available
        if self.rolling_optimizer:
            try:
                avg_volume = self.rolling_optimizer.rolling_mean(volume, window=self.period)
                volume_ratio = volume / avg_volume.replace(0, np.nan).fillna(method='ffill').fillna(volume.mean())  # Avoid division by zero
                self.performance_stats['vectorbt_operations'] += 1
                return volume_ratio
            except Exception as e:
                self.logger.warning(f"VectorBT rolling optimizer failed: {e}, using fallback")
                self.performance_stats['pandas_fallbacks'] += 1

        # Fallback to VectorBT direct operations
        if VECTORBT_AVAILABLE:
            try:
                avg_volume = rolling_mean(volume, window=self.period)
                volume_ratio = volume / avg_volume.replace(0, np.nan).fillna(method='ffill').fillna(volume.mean())  # Avoid division by zero
                self.performance_stats['vectorbt_operations'] += 1
                return volume_ratio
            except Exception as e:
                self.logger.warning(f"VectorBT volume ratio calculation failed: {e}, using pandas fallback")
                self.performance_stats['pandas_fallbacks'] += 1

        # Final fallback to pandas
        avg_volume = volume.rolling(window=self.period).mean()
        return volume / avg_volume.replace(0, np.nan).fillna(method='ffill').fillna(volume.mean())  # Avoid division by zero

# Volume Rate of Change

class VolumeROCGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for Volume Rate of Change with VectorBT optimization."""

    def __init__(self, period: int = 10):
        config = FeatureConfig(
            name=f"volume_roc_{period}",
            category=FeatureCategory.VOLUME,
            description=f"Volume Rate of Change over {period} periods with VectorBT optimization",
            required_columns=["volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={'period': period},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period

        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_optimizer = None
        # Initialize VectorBT rolling optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.rolling_optimizer = None

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volume ROC using VectorBT optimization."""
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        volume = data['volume']

        if data.empty or 'volume' not in data.columns:
            return pd.Series(dtype=float, index=data.index, name=f'volume_roc_{self.period}')

        volume = data['volume']

        # Use VectorBT rolling optimizer for ROC calculation
        if self.rolling_optimizer:
            try:
                # Use VectorBT rolling apply for ROC calculation
                def roc_func(x):
                    if len(x) < self.period + 1:
                        return np.nan
                    return (x.iloc[-1] / x.iloc[0] - 1) * 100 if x.iloc[0] != 0 else np.nan

                volume_roc = self.rolling_optimizer.rolling_apply(volume, window=self.period + 1, func=roc_func)
                self.performance_stats['vectorbt_operations'] += 1
                return volume_roc
            except Exception as e:
                self.logger.warning(f"VectorBT rolling optimizer failed: {e}, using fallback")
                self.performance_stats['pandas_fallbacks'] += 1

        # Fallback to VectorBT direct operations
        if VECTORBT_AVAILABLE:
            try:
                # VectorBT doesn't have direct pct_change, so we calculate it manually
                roc = (volume / volume.shift(self.period) - 1) * 100
                self.performance_stats['vectorbt_operations'] += 1
                return roc
            except Exception as e:
                self.logger.warning(f"VectorBT volume ROC calculation failed: {e}, using pandas fallback")
                self.performance_stats['pandas_fallbacks'] += 1

        # Final fallback to pandas
        return volume.pct_change(periods=self.period) * 100

# Volume Standard Deviation

class VolumeStdGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for Volume Standard Deviation with VectorBT optimization."""

    def __init__(self, period: int = 20):
        config = FeatureConfig(
            name=f"volume_std_{period}",
            category=FeatureCategory.VOLUME,
            description=f"Volume Standard Deviation over {period} periods with VectorBT optimization",
            required_columns=["volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={'period': period},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period

        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_optimizer = None
        # Initialize VectorBT rolling optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.rolling_optimizer = None

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volume Standard Deviation using VectorBT optimization."""
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        volume = data['volume']

        if data.empty or 'volume' not in data.columns:
            return pd.Series(dtype=float, index=data.index, name=f'volume_std_{self.period}')

        volume = data['volume']

        # Use VectorBT rolling optimizer if available
        if self.rolling_optimizer:
            try:
                volume_std = self.rolling_optimizer.rolling_std(volume, window=self.period)
                self.performance_stats['vectorbt_operations'] += 1
                return volume_std
            except Exception as e:
                self.logger.warning(f"VectorBT rolling optimizer failed: {e}, using fallback")
                self.performance_stats['pandas_fallbacks'] += 1

        # Fallback to VectorBT direct operations
        if VECTORBT_AVAILABLE:
            try:
                volume_std = rolling_std(volume, window=self.period)
                self.performance_stats['vectorbt_operations'] += 1
                return volume_std
            except Exception as e:
                self.logger.warning(f"VectorBT volume std calculation failed: {e}, using pandas fallback")
                self.performance_stats['pandas_fallbacks'] += 1

        # Final fallback to pandas
        return volume.rolling(window=self.period).std()

# Volume Percentile Rank

class VolumePercentileGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for Volume Percentile Rank with VectorBT optimization."""

    def __init__(self, period: int = 50):
        config = FeatureConfig(
            name=f"volume_percentile_{period}",
            category=FeatureCategory.VOLUME,
            description=f"Volume Percentile Rank over {period} periods",
            required_columns=["volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={'period': period},
            gpu_accelerated=True
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.rolling_optimizer = VectorBTRollingOptimizer()
        self.period = period

        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_optimizer = None

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        volume = data['volume']

        """Generate Volume Percentile Rank using VectorBT."""
        volume = data['volume']

        # Try VectorBT rolling optimizer first
        if self.rolling_optimizer:
            try:
                # Use custom function for percentile rank calculation
                def percentile_rank_func(x):
                    return x.rank(pct=True).iloc[-1] * 100

                percentile_rank = self.rolling_optimizer.rolling_apply(volume, window=self.period, func=percentile_rank_func)
                self.performance_stats['vectorbt_operations'] += 1
                return percentile_rank
            except Exception as e:
                self.logger.warning(f"VectorBT volume percentile calculation failed: {e}, using fallback")
                self.performance_stats['pandas_fallbacks'] += 1

        # Fallback to VectorBT direct operations
        if VECTORBT_AVAILABLE:
            try:
                # VectorBT doesn't have direct rank, so we use pandas rank
                return volume.rolling(window=self.period).rank(pct=True) * 100
            except Exception as e:
                self.logger.warning(f"VectorBT volume percentile calculation failed: {e}, using pandas fallback")
                self.performance_stats['pandas_fallbacks'] += 1

        # Final fallback to pandas
        return volume.rolling(window=self.period).rank(pct=True) * 100

# Volume Trend Strength

class VolumeTrendStrengthGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for Volume Trend Strength with VectorBT optimization."""

    def __init__(self, short_period: int = 10, long_period: int = 30):
        config = FeatureConfig(
            name=f"volume_trend_strength_{short_period}_{long_period}",
            category=FeatureCategory.VOLUME,
            description=f"Volume Trend Strength using {short_period} and {long_period} periods",
            required_columns=["volume"],
            default_lookback=long_period,
            min_lookback=long_period,
            max_lookback=long_period,
            parameters={'short_period': short_period, 'long_period': long_period},
            gpu_accelerated=True
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.rolling_optimizer = VectorBTRollingOptimizer()
        self.short_period = short_period
        self.long_period = long_period

        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_optimizer = None

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        volume = data['volume']

        """Generate Volume Trend Strength using VectorBT."""
        volume = data['volume']

        # Try VectorBT rolling optimizer first
        if self.rolling_optimizer:
            try:
                short_ma = self.rolling_optimizer.rolling_mean(volume, window=self.short_period)
                long_ma = self.rolling_optimizer.rolling_mean(volume, window=self.long_period)
                self.performance_stats['vectorbt_operations'] += 2
                return (short_ma - long_ma) / long_ma.replace(0, 1) * 100
            except Exception as e:
                self.logger.warning(f"VectorBT volume trend strength calculation failed: {e}, using fallback")
                self.performance_stats['pandas_fallbacks'] += 1

        # Fallback to VectorBT direct operations
        if VECTORBT_AVAILABLE:
            try:
                short_ma = rolling_mean(volume, window=self.short_period)
                long_ma = rolling_mean(volume, window=self.long_period)
                self.performance_stats['vectorbt_operations'] += 2
                return (short_ma - long_ma) / long_ma.replace(0, 1) * 100
            except Exception as e:
                self.logger.warning(f"VectorBT volume trend strength calculation failed: {e}, using pandas fallback")
                self.performance_stats['pandas_fallbacks'] += 1

        # Final fallback to pandas
        short_ma = volume.rolling(window=self.short_period).mean()
        long_ma = volume.rolling(window=self.long_period).mean()
        return (short_ma - long_ma) / long_ma.replace(0, 1) * 100

# Volume Oscillator

class VolumeOscillatorGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for Volume Oscillator with VectorBT optimization."""

    def __init__(self, short_period: int = 10, long_period: int = 20):
        config = FeatureConfig(
            name=f"volume_oscillator_{short_period}_{long_period}",
            category=FeatureCategory.VOLUME,
            description=f"Volume Oscillator using {short_period} and {long_period} periods",
            required_columns=["volume"],
            default_lookback=long_period,
            min_lookback=long_period,
            max_lookback=long_period,
            parameters={'short_period': short_period, 'long_period': long_period},
            gpu_accelerated=True
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.rolling_optimizer = VectorBTRollingOptimizer()
        self.short_period = short_period
        self.long_period = long_period

        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_optimizer = None

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        volume = data['volume']

        """Generate Volume Oscillator using VectorBT."""
        volume = data['volume']

        # Try VectorBT rolling optimizer first
        if self.rolling_optimizer:
            try:
                short_ma = self.rolling_optimizer.rolling_mean(volume, window=self.short_period)
                long_ma = self.rolling_optimizer.rolling_mean(volume, window=self.long_period)
                self.performance_stats['vectorbt_operations'] += 2
                return short_ma - long_ma
            except Exception as e:
                self.logger.warning(f"VectorBT volume oscillator calculation failed: {e}, using fallback")
                self.performance_stats['pandas_fallbacks'] += 1

        # Fallback to VectorBT direct operations
        if VECTORBT_AVAILABLE:
            try:
                short_ma = rolling_mean(volume, window=self.short_period)
                long_ma = rolling_mean(volume, window=self.long_period)
                self.performance_stats['vectorbt_operations'] += 2
                return short_ma - long_ma
            except Exception as e:
                self.logger.warning(f"VectorBT volume oscillator calculation failed: {e}, using pandas fallback")
                self.performance_stats['pandas_fallbacks'] += 1

        # Final fallback to pandas
        short_ma = volume.rolling(window=self.short_period).mean()
        long_ma = volume.rolling(window=self.long_period).mean()
        return short_ma - long_ma

# Volume Momentum

class VolumeMomentumGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for Volume Momentum with VectorBT optimization."""

    def __init__(self, period: int = 10):
        config = FeatureConfig(
            name=f"volume_momentum_{period}",
            category=FeatureCategory.VOLUME,
            description=f"Volume Momentum over {period} periods",
            required_columns=["volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={'period': period},
            gpu_accelerated=True
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.rolling_optimizer = VectorBTRollingOptimizer()
        self.period = period

        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_optimizer = None

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        volume = data['volume']

        """Generate Volume Momentum using VectorBT."""
        volume = data['volume']

        # Try VectorBT rolling optimizer first
        if self.rolling_optimizer:
            try:
                # Use custom function for momentum calculation
                def momentum_func(x):
                    if len(x) > self.period:
                        return x.iloc[-1] - x.iloc[-self.period-1]
                    return 0

                momentum = self.rolling_optimizer.rolling_apply(volume, window=self.period + 1, func=momentum_func)
                self.performance_stats['vectorbt_operations'] += 1
                return momentum
            except Exception as e:
                self.logger.warning(f"VectorBT volume momentum calculation failed: {e}, using fallback")
                self.performance_stats['pandas_fallbacks'] += 1

        # Fallback to VectorBT direct operations
        if VECTORBT_AVAILABLE:
            try:
                # VectorBT doesn't have direct shift, so we use pandas shift
                return volume - volume.shift(self.period)
            except Exception as e:
                self.logger.warning(f"VectorBT volume momentum calculation failed: {e}, using pandas fallback")
                self.performance_stats['pandas_fallbacks'] += 1

        # Final fallback to pandas
        return volume - volume.shift(self.period)

# Volume Weighted Average Price (VWAP)

class VolumeVWAPGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for Volume Weighted Average Price with VectorBT optimization."""

    def __init__(self, period: int = 20):
        config = FeatureConfig(
            name=f"volume_vwap_{period}",
            category=FeatureCategory.VOLUME,
            description=f"Volume Weighted Average Price over {period} periods",
            required_columns=["close", "volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={'period': period},
            gpu_accelerated=True
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period

        # Initialize VectorBT rolling optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.rolling_optimizer = None

        # Initialize Unified Vectorization Manager
        if UNIFIED_MANAGER_AVAILABLE:
            self.unified_manager = get_unified_vectorization_manager()
        else:
            self.unified_manager = None

        # Performance tracking
        self.performance_stats = {
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'unified_manager_operations': 0
        }

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volume VWAP using VectorBT optimization."""
        if data.empty or 'close' not in data.columns or 'volume' not in data.columns:
            return pd.Series(dtype=float, index=data.index, name=f'volume_vwap_{self.period}')

        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        volume = data['volume']

        # Use Unified Vectorization Manager for intelligent optimization
        if self.unified_manager and self._should_use_unified_manager(volume):
            try:
                return self._generate_vwap_with_unified_manager(close, volume)
            except Exception as e:
                logger.warning(f"Unified Vectorization Manager failed: {e}, using VectorBT fallback")
                self.performance_stats['pandas_fallbacks'] += 1

        # Use VectorBT rolling optimizer if available
        if self.rolling_optimizer and self._should_use_vectorbt(volume):
            try:
                price_volume = close * volume
                price_volume_sum = self.rolling_optimizer.rolling_sum(price_volume, window=self.period)
                volume_sum = self.rolling_optimizer.rolling_sum(volume, window=self.period)
                self.performance_stats['vectorbt_operations'] += 2
                return price_volume_sum / volume_sum
            except Exception as e:
                logger.warning(f"VectorBT volume VWAP calculation failed: {e}, using fallback")
                self.performance_stats['pandas_fallbacks'] += 1

        # Fallback to VectorBT direct operations
        if VECTORBT_AVAILABLE:
            try:
                price_volume = close * volume
                price_volume_sum = rolling_sum(price_volume, window=self.period)
                volume_sum = rolling_sum(volume, window=self.period)
                self.performance_stats['vectorbt_operations'] += 2
                return price_volume_sum / volume_sum
            except Exception as e:
                logger.warning(f"VectorBT volume VWAP calculation failed: {e}, using pandas fallback")
                self.performance_stats['pandas_fallbacks'] += 1

        # Final fallback to pandas
        return (close * volume).rolling(window=self.period).sum() / volume.rolling(window=self.period).sum()

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (VECTORBT_AVAILABLE and
                len(data) >= getattr(self, 'vectorbt_threshold', 1000))

    def _should_use_unified_manager(self, data) -> bool:
        """Determine if Unified Vectorization Manager should be used."""
        return (UNIFIED_MANAGER_AVAILABLE and
                len(data) >= getattr(self, 'unified_manager_threshold', 5000))

    def _generate_vwap_with_unified_manager(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Generate VWAP using Unified Vectorization Manager."""
        try:
            # Configure operation for volume VWAP calculation
            operation_config = {
                'operation_type': OperationType.TECHNICAL_INDICATORS,
                'data_size': len(volume),
                'data_dimensions': (len(volume),),
                'memory_budget_mb': 512.0,
                'time_budget_seconds': 60.0,
                'precision_requirement': 'medium'
            }

            # Use unified manager for VWAP calculation
            def vwap_operation():
                price_volume = close * volume
                price_volume_sum = self.rolling_optimizer.rolling_sum(price_volume, window=self.period)
                volume_sum = self.rolling_optimizer.rolling_sum(volume, window=self.period)
                return price_volume_sum / volume_sum

            result = self.unified_manager.optimize_operation(vwap_operation, operation_config)

            self.performance_stats['unified_manager_operations'] += 1
            return result.result

        except Exception as e:
            logger.warning(f"Unified manager VWAP generation failed: {e}")
            raise

# Volume Price Trend (VPT)

class VolumePriceTrendGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for Volume Price Trend with VectorBT optimization."""

    def __init__(self):
        config = FeatureConfig(
            name="volume_price_trend",
            category=FeatureCategory.VOLUME,
            description="Volume Price Trend indicator",
            required_columns=["close", "volume"],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            parameters={}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_optimizer = None

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        volume = data['volume']

        # Use VectorBT for volume price trend calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(volume):
            try:
                price_change = close.pct_change()
                vpt = (price_change * volume).cumsum()
                return vpt
            except Exception as e:
                self.logger.warning(f"VectorBT volume price trend calculation failed: {e}, using pandas fallback")
                price_change = close.pct_change()
                vpt = (price_change * volume).cumsum()
                return vpt
        else:
            price_change = close.pct_change()
            vpt = (price_change * volume).cumsum()
            return vpt

# Volume Accumulation/Distribution

class VolumeAccumulationDistributionGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for Volume Accumulation/Distribution with VectorBT optimization."""

    def __init__(self):
        config = FeatureConfig(
            name="volume_accumulation_distribution",
            category=FeatureCategory.VOLUME,
            description="Volume Accumulation/Distribution indicator",
            required_columns=["close", "high", "low", "volume"],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1,
            parameters={}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_optimizer = None

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']

        # Use VectorBT for volume accumulation/distribution calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(volume):
            try:
                # Calculate Money Flow Multiplier
                mfm = ((close - low) - (high - close)) / (high - low).replace(0, 1)
                mfm = mfm.clip(-1, 1)  # Clamp between -1 and 1

                # Calculate Money Flow Volume
                mfv = mfm * volume

                return mfv.cumsum()
            except Exception as e:
                self.logger.warning(f"VectorBT volume accumulation/distribution calculation failed: {e}, using pandas fallback")
                # Calculate Money Flow Multiplier
                mfm = ((close - low) - (high - close)) / (high - low).replace(0, 1)
                mfm = mfm.clip(-1, 1)  # Clamp between -1 and 1

                # Calculate Money Flow Volume
                mfv = mfm * volume

                return mfv.cumsum()
        else:
            # Calculate Money Flow Multiplier
            mfm = ((close - low) - (high - close)) / (high - low).replace(0, 1)
            mfm = mfm.clip(-1, 1)  # Clamp between -1 and 1

            # Calculate Money Flow Volume
            mfv = mfm * volume

            return mfv.cumsum()

class VolumePriceCorrelationGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for volume-price correlation features with VectorBT optimization."""

    def __init__(self, period: int = 20):
        config = FeatureConfig(
            name=f"volume_price_correlation_{period}",
            category=FeatureCategory.VOLUME,
            description=f"Correlation between volume and price over {period} periods",
            required_columns=["close", "volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period * 2,
            parameters={"period": period},
            gpu_accelerated=True
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period

        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_optimizer = None
        self.rolling_optimizer = VectorBTRollingOptimizer()

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        volume = data['volume']

        # Use VectorBT for volume-price correlation calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(volume):
            try:
                # Rolling correlation between price returns and volume
                price_returns = close.pct_change()
                correlation = self.vectorbt_optimizer.rolling_corr(price_returns, volume, window=self.period)
                return correlation
            except Exception as e:
                self.logger.warning(f"VectorBT volume-price correlation calculation failed: {e}, using pandas fallback")
                price_returns = close.pct_change()
                correlation = price_returns.rolling(window=self.period).corr(volume)
                return correlation
        else:
            # Rolling correlation between price returns and volume
            price_returns = close.pct_change()
            correlation = price_returns.rolling(window=self.period).corr(volume)
            return correlation
        """Calculate volume-price correlation using VectorBT."""
        close = data['close']
        volume = data['volume']

        # Try VectorBT rolling optimizer first
        if self.rolling_optimizer:
            try:
                # Rolling correlation between price returns and volume
                price_returns = close.pct_change()
                correlation = self.rolling_optimizer.rolling_corr(price_returns, volume, window=self.config.parameters["period"])
                self.performance_stats['vectorbt_operations'] += 1
                return correlation
            except Exception as e:
                self.logger.warning(f"VectorBT volume-price correlation calculation failed: {e}, using fallback")
                self.performance_stats['pandas_fallbacks'] += 1

        # Fallback to VectorBT direct operations
        if VECTORBT_AVAILABLE:
            try:
                # Rolling correlation between price returns and volume
                price_returns = close.pct_change()
                correlation = rolling_corr(price_returns, volume, window=self.config.parameters["period"])
                self.performance_stats['vectorbt_operations'] += 1
                return correlation
            except Exception as e:
                self.logger.warning(f"VectorBT volume-price correlation calculation failed: {e}, using pandas fallback")
                self.performance_stats['pandas_fallbacks'] += 1

        # Final fallback to pandas
        price_returns = close.pct_change()
        correlation = price_returns.rolling(window=self.config.parameters["period"]).corr(volume)
        return correlation

class VolumePriceDivergenceGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for volume-price divergence features with VectorBT optimization."""

    def __init__(self, period: int = 20):
        config = FeatureConfig(
            name=f"volume_price_divergence_{period}",
            category=FeatureCategory.VOLUME,
            description=f"Volume-price divergence indicator over {period} periods",
            required_columns=["close", "volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period * 2,
            parameters={"period": period}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period

        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_optimizer = None

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        volume = data['volume']

        # Use VectorBT for volume-price divergence calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(volume):
            try:
                # Price momentum with regime smoothing
                price_ma = self.vectorbt_optimizer.rolling_mean(close, window=self.period)
                price_momentum = (close - price_ma) / (price_ma + 1e-8)  # Avoid division by zero

                # Volume momentum with regime smoothing
                volume_ma = self.vectorbt_optimizer.rolling_mean(volume, window=self.period)
                volume_momentum = (volume - volume_ma) / (volume_ma + 1e-8)  # Avoid division by zero

                # Enhanced divergence with regime persistence
                divergence = price_momentum * volume_momentum

                # Add regime stability measure
                price_volatility = self.vectorbt_optimizer.rolling_std(close, window=self.period)
                volume_volatility = self.vectorbt_optimizer.rolling_std(volume, window=self.period)

                # Regime strength indicator (higher when both price and volume show consistent trends)
                regime_strength = np.abs(divergence) / (price_volatility * volume_volatility + 1e-8)

                # Combine divergence with regime strength for better clustering
                enhanced_divergence = divergence * (1 + regime_strength)

                return enhanced_divergence
            except Exception as e:
                self.logger.warning(f"VectorBT volume-price divergence calculation failed: {e}, using pandas fallback")
                # Price momentum with regime smoothing
                price_ma = close.rolling(window=self.period).mean()
                price_momentum = (close - price_ma) / (price_ma + 1e-8)  # Avoid division by zero

                # Volume momentum with regime smoothing
                volume_ma = volume.rolling(window=self.period).mean()
                volume_momentum = (volume - volume_ma) / (volume_ma + 1e-8)  # Avoid division by zero

                # Enhanced divergence with regime persistence
                divergence = price_momentum * volume_momentum

                # Add regime stability measure
                price_volatility = close.rolling(window=self.period).std()
                volume_volatility = volume.rolling(window=self.period).std()

                # Regime strength indicator (higher when both price and volume show consistent trends)
                regime_strength = np.abs(divergence) / (price_volatility * volume_volatility + 1e-8)

                # Combine divergence with regime strength for better clustering
                enhanced_divergence = divergence * (1 + regime_strength)

                return enhanced_divergence
        else:
            # Price momentum with regime smoothing
            price_ma = close.rolling(window=self.period).mean()
            price_momentum = (close - price_ma) / (price_ma + 1e-8)  # Avoid division by zero

            # Volume momentum with regime smoothing
            volume_ma = volume.rolling(window=self.period).mean()
            volume_momentum = (volume - volume_ma) / (volume_ma + 1e-8)  # Avoid division by zero

            # Enhanced divergence with regime persistence
            divergence = price_momentum * volume_momentum

            # Add regime stability measure
            price_volatility = close.rolling(window=self.period).std()
            volume_volatility = volume.rolling(window=self.period).std()

            # Regime strength indicator (higher when both price and volume show consistent trends)
            regime_strength = np.abs(divergence) / (price_volatility * volume_volatility + 1e-8)

            # Combine divergence with regime strength for better clustering
            enhanced_divergence = divergence * (1 + regime_strength)

            return enhanced_divergence

class PriceVolumeOscillatorGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for price-volume oscillator features with VectorBT optimization."""

    def __init__(self, fast_period: int = 10, slow_period: int = 20):
        config = FeatureConfig(
            name=f"price_volume_oscillator_{fast_period}_{slow_period}",
            category=FeatureCategory.VOLUME,
            description=f"Price-volume oscillator ({fast_period}/{slow_period})",
            required_columns=["close", "volume"],
            default_lookback=slow_period,
            min_lookback=slow_period,
            max_lookback=slow_period * 2,
            parameters={"fast_period": fast_period, "slow_period": slow_period}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.fast_period = fast_period
        self.slow_period = slow_period

        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_optimizer = None

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        volume = data['volume']

        # Use VectorBT for price-volume oscillator calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(volume):
            try:
                # Price oscillator
                fast_ma = self.vectorbt_optimizer.rolling_mean(close, window=self.fast_period)
                slow_ma = self.vectorbt_optimizer.rolling_mean(close, window=self.slow_period)
                price_osc = (fast_ma - slow_ma) / slow_ma

                # Volume oscillator
                volume_fast_ma = self.vectorbt_optimizer.rolling_mean(volume, window=self.fast_period)
                volume_slow_ma = self.vectorbt_optimizer.rolling_mean(volume, window=self.slow_period)
                volume_osc = (volume_fast_ma - volume_slow_ma) / volume_slow_ma

                # Combined oscillator
                combined_osc = price_osc * volume_osc

                return combined_osc
            except Exception as e:
                self.logger.warning(f"VectorBT price-volume oscillator calculation failed: {e}, using pandas fallback")
                # Price oscillator
                fast_ma = close.rolling(window=self.fast_period).mean()
                slow_ma = close.rolling(window=self.slow_period).mean()
                price_osc = (fast_ma - slow_ma) / slow_ma

                # Volume oscillator
                volume_fast_ma = volume.rolling(window=self.fast_period).mean()
                volume_slow_ma = volume.rolling(window=self.slow_period).mean()
                volume_osc = (volume_fast_ma - volume_slow_ma) / volume_slow_ma

                # Combined oscillator
                combined_osc = price_osc * volume_osc

                return combined_osc
        else:
            # Price oscillator
            fast_ma = close.rolling(window=self.fast_period).mean()
            slow_ma = close.rolling(window=self.slow_period).mean()
            price_osc = (fast_ma - slow_ma) / slow_ma

            # Volume oscillator
            volume_fast_ma = volume.rolling(window=self.fast_period).mean()
            volume_slow_ma = volume.rolling(window=self.slow_period).mean()
            volume_osc = (volume_fast_ma - volume_slow_ma) / volume_slow_ma

            # Combined oscillator
            combined_osc = price_osc * volume_osc

            return combined_osc

def create_default_volume_generators() -> List[FeatureGenerator]:
    """Create default volume feature generators."""
    generators = []

    # Volume moving averages
    for period in [5, 10, 20, 50]:
        generators.append(VolumeSMAGenerator(period))
        generators.append(VolumeEMAGenerator(period))

    # Volume ratios
    for period in [10, 20, 50]:
        generators.append(VolumeRatioGenerator(period))

    # Volume rate of change
    for period in [1, 5, 10, 20]:
        generators.append(VolumeROCGenerator(period))

    # Volume standard deviation
    for period in [10, 20, 50]:
        generators.append(VolumeStdGenerator(period))

    # Volume percentile rank
    for period in [20, 50, 100]:
        generators.append(VolumePercentileGenerator(period))

    # Volume trend strength
    generators.append(VolumeTrendStrengthGenerator(10, 30))
    generators.append(VolumeTrendStrengthGenerator(20, 50))

    # Volume oscillator
    generators.append(VolumeOscillatorGenerator(10, 20))
    generators.append(VolumeOscillatorGenerator(5, 15))

    # Volume momentum
    for period in [5, 10, 20]:
        generators.append(VolumeMomentumGenerator(period))

    # Volume VWAP
    for period in [10, 20, 50]:
        generators.append(VolumeVWAPGenerator(period))

    # Volume Price Trend
    generators.append(VolumePriceTrendGenerator())

    # Volume Accumulation/Distribution
    generators.append(VolumeAccumulationDistributionGenerator())

    # Volume-Price Divergence features for regime identification
    for period in [10, 20]:
        generators.append(VolumePriceCorrelationGenerator(period))
        generators.append(VolumePriceDivergenceGenerator(period))

    # Price-Volume Oscillator
    generators.append(PriceVolumeOscillatorGenerator(10, 20))
    generators.append(PriceVolumeOscillatorGenerator(5, 15))

    # Analyst Features - Volume patterns
    generators.append(AnalystVolumePressureGenerator())
    generators.append(AnalystVolumeTrendGenerator())

    # NEW FEATURES - Enhanced Volume Analysis
    # Volume z-score generators
    for short_window in [60]:
        for long_window in [252]:
            generators.append(VolumeZScoreGenerator(short_window, long_window))

    # Volume MA ratios generators
    for ma_period in [20]:
        for surprise_window in [10]:
            generators.append(VolumeMARatiosGenerator(ma_period, surprise_window))

    # CMF generators
    for period in [20]:
        generators.append(CMFGenerator(period))

    # VWAP deviations generators
    for vwap_window in [20]:
        generators.append(VWAPDeviationsGenerator(vwap_window))

    # Order flow imbalance generators
    for window in [20]:
        generators.append(OrderFlowImbalanceGenerator(window))

    # Volume-volatility elasticity generators
    for window in [20]:
        generators.append(VolumeVolatilityElasticityGenerator(window))

    # Enhanced OBV and AD line versions using VectorBTRollingOptimizer
    if VECTORBT_AVAILABLE:
        # Enhanced OBV generators
        for period in [10, 20, 50]:
            generators.append(VectorBTEnhancedOBVGenerator(period))

        # Enhanced AD line generators
        for period in [10, 20, 50]:
            generators.append(VectorBTEnhancedADLineGenerator(period))

        # Volume-weighted AD line generators
        for period in [10, 20, 50]:
            generators.append(VectorBTVolumeWeightedADLineGenerator(period))

        # Smoothed OBV generators
        for period in [10, 20, 50]:
            generators.append(VectorBTSmoothedOBVGenerator(period))

    return generators

class OptimizedVolumeFeatureFactory:
    """Factory for creating optimized volume feature generators with VectorBT and UnifiedVectorizationManager."""

    def __init__(self, enable_gpu: bool = False, enable_parallel: bool = True):
        """
        Initialize the optimized volume feature factory.

        Args:
            enable_gpu: Whether to enable GPU processing
            enable_parallel: Whether to enable parallel processing
        """
        self.enable_gpu = enable_gpu
        self.enable_parallel = enable_parallel

        # Initialize optimizers
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=enable_gpu,
                enable_parallel=enable_parallel
            )
        else:
            self.rolling_optimizer = None

        if UNIFIED_MANAGER_AVAILABLE:
            self.unified_manager = get_unified_vectorization_manager()
        else:
            self.unified_manager = None

    def create_volume_sma_generator(self, period: int = 20) -> VolumeSMAGenerator:
        """Create an optimized Volume SMA generator."""
        generator = VolumeSMAGenerator(period)
        generator.rolling_optimizer = self.rolling_optimizer
        generator.unified_manager = self.unified_manager
        return generator

    def create_volume_vwap_generator(self, period: int = 20) -> VolumeVWAPGenerator:
        """Create an optimized Volume VWAP generator."""
        generator = VolumeVWAPGenerator(period)
        generator.rolling_optimizer = self.rolling_optimizer
        generator.unified_manager = self.unified_manager
        return generator

    def create_batch_volume_generator(self) -> VolumeFeatureGenerator:
        """Create an optimized batch volume feature generator."""
        generator = VolumeFeatureGenerator()
        generator.rolling_optimizer = self.rolling_optimizer
        generator.unified_manager = self.unified_manager
        return generator

    def generate_comprehensive_volume_features(self, data: pd.DataFrame,
                                             periods: List[int] = None) -> pd.DataFrame:
        """
        Generate comprehensive volume features using batch processing.

        Args:
            data: OHLCV data
            periods: List of periods to use for features

        Returns:
            DataFrame with comprehensive volume features
        """
        if periods is None:
            periods = [5, 10, 20, 50]

        # Create feature configurations for batch processing
        feature_configs = []

        # Volume SMAs
        for period in periods:
            feature_configs.append({
                'name': f'volume_sma_{period}',
                'type': 'sma',
                'period': period
            })

        # Volume EMAs
        for period in periods:
            feature_configs.append({
                'name': f'volume_ema_{period}',
                'type': 'ema',
                'period': period
            })

        # Volume standard deviations
        for period in periods:
            feature_configs.append({
                'name': f'volume_std_{period}',
                'type': 'std',
                'period': period
            })

        # Volume VWAPs
        for period in periods:
            feature_configs.append({
                'name': f'volume_vwap_{period}',
                'type': 'vwap',
                'period': period
            })

        # Use batch generator
        batch_generator = self.create_batch_volume_generator()
        return batch_generator.generate_batch_volume_features(data, feature_configs)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics from all optimizers."""
        stats = {}

        if self.rolling_optimizer:
            stats['rolling_optimizer'] = self.rolling_optimizer.get_performance_stats()

        if self.unified_manager:
            stats['unified_manager'] = self.unified_manager.get_performance_stats()

        return stats

def create_optimized_volume_factory(enable_gpu: bool = False,
                                  enable_parallel: bool = True) -> OptimizedVolumeFeatureFactory:
    """
    Create an optimized volume feature factory.

    Args:
        enable_gpu: Whether to enable GPU processing
        enable_parallel: Whether to enable parallel processing

    Returns:
        OptimizedVolumeFeatureFactory instance
    """
    return OptimizedVolumeFeatureFactory(enable_gpu=enable_gpu, enable_parallel=enable_parallel)

class VectorBTEnhancedOBVGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized Enhanced OBV generator with smoothing and trend analysis."""

    def __init__(self, period: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.period = period

        # Initialize VectorBT rolling optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=True, enable_parallel=True)
        else:
            self.rolling_optimizer = None

    @classmethod
    def _create_default_config(cls, period: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_enhanced_obv_{period}",
            category=FeatureCategory.VOLUME,
            description=f"VectorBT-optimized Enhanced OBV with smoothing over {period} periods",
            required_columns=["close", "volume"],
            optional_columns=["high", "low", "open"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=True
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Enhanced OBV using VectorBT."""
        if data.empty or not all(col in data.columns for col in ['close', 'volume']):
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_enhanced_obv_{self.period}')

        try:
            close = data['close']
            volume = data['volume']

            # Calculate price direction
            price_direction = np.where(close > close.shift(1), 1,
                                     np.where(close < close.shift(1), -1, 0))

            # Calculate basic OBV
            obv = (price_direction * volume).cumsum()
            obv_series = pd.Series(obv, index=close.index)

            # Apply smoothing using VectorBT rolling optimizer
            if self.rolling_optimizer:
                try:
                    # Smooth OBV with EMA
                    smoothed_obv = obv_series.ewm(span=self.period).mean()

                    # Calculate OBV trend strength
                    obv_trend = self.rolling_optimizer.rolling_mean(smoothed_obv.diff(), window=self.period)

                    # Calculate OBV momentum
                    obv_momentum = smoothed_obv - smoothed_obv.shift(self.period)

                    # Combine features
                    enhanced_obv = smoothed_obv + obv_trend + obv_momentum

                    return enhanced_obv.rename(f'vectorbt_enhanced_obv_{self.period}')
                except Exception as e:
                    self.logger.warning(f"VectorBT rolling optimizer failed: {e}, using fallback")

            # Fallback to VectorBT direct operations
            if VECTORBT_AVAILABLE:
                try:
                    # Smooth OBV with EMA
                    smoothed_obv = obv_series.ewm(span=self.period).mean()

                    # Calculate OBV trend strength
                    obv_trend = rolling_mean(smoothed_obv.diff(), window=self.period)

                    # Calculate OBV momentum
                    obv_momentum = smoothed_obv - smoothed_obv.shift(self.period)

                    # Combine features
                    enhanced_obv = smoothed_obv + obv_trend + obv_momentum

                    return enhanced_obv.rename(f'vectorbt_enhanced_obv_{self.period}')
                except Exception as e:
                    self.logger.warning(f"VectorBT Enhanced OBV calculation failed: {e}, using pandas fallback")

            # Final fallback to pandas
            smoothed_obv = obv_series.ewm(span=self.period).mean()
            obv_trend = smoothed_obv.diff().rolling(window=self.period).mean()
            obv_momentum = smoothed_obv - smoothed_obv.shift(self.period)
            enhanced_obv = smoothed_obv + obv_trend + obv_momentum

            return enhanced_obv.rename(f'vectorbt_enhanced_obv_{self.period}')

        except Exception as e:
            self.logger.error(f"Error generating Enhanced OBV: {e}")
            return pd.Series(np.nan, index=data.index, name=f'vectorbt_enhanced_obv_{self.period}')

class VectorBTEnhancedADLineGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized Enhanced AD Line generator with advanced features."""

    def __init__(self, period: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.period = period

        # Initialize VectorBT rolling optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=True, enable_parallel=True)
        else:
            self.rolling_optimizer = None

    @classmethod
    def _create_default_config(cls, period: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_enhanced_ad_line_{period}",
            category=FeatureCategory.VOLUME,
            description=f"VectorBT-optimized Enhanced AD Line with smoothing over {period} periods",
            required_columns=["high", "low", "close", "volume"],
            optional_columns=["open"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=True
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Enhanced AD Line using VectorBT."""
        if data.empty or not all(col in data.columns for col in ['high', 'low', 'close', 'volume']):
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_enhanced_ad_line_{self.period}')

        try:
            high = data['high']
            low = data['low']
            close = data['close']
            volume = data['volume']

            # Calculate Money Flow Multiplier
            mfm = ((close - low) - (high - close)) / (high - low)
            mfm = mfm.fillna(0)  # Handle division by zero

            # Calculate Money Flow Volume
            mfv = mfm * volume

            # Calculate basic AD Line
            ad_line = mfv.cumsum()

            # Apply smoothing and enhancement using VectorBT rolling optimizer
            if self.rolling_optimizer:
                try:
                    # Smooth AD Line with EMA
                    smoothed_ad = ad_line.ewm(span=self.period).mean()

                    # Calculate AD Line trend strength
                    ad_trend = self.rolling_optimizer.rolling_mean(smoothed_ad.diff(), window=self.period)

                    # Calculate AD Line momentum
                    ad_momentum = smoothed_ad - smoothed_ad.shift(self.period)

                    # Calculate AD Line volatility
                    ad_volatility = self.rolling_optimizer.rolling_std(smoothed_ad, window=self.period)

                    # Combine features
                    enhanced_ad = smoothed_ad + ad_trend + ad_momentum - ad_volatility

                    return enhanced_ad.rename(f'vectorbt_enhanced_ad_line_{self.period}')
                except Exception as e:
                    self.logger.warning(f"VectorBT rolling optimizer failed: {e}, using fallback")

            # Fallback to VectorBT direct operations
            if VECTORBT_AVAILABLE:
                try:
                    # Smooth AD Line with EMA
                    smoothed_ad = ad_line.ewm(span=self.period).mean()

                    # Calculate AD Line trend strength
                    ad_trend = rolling_mean(smoothed_ad.diff(), window=self.period)

                    # Calculate AD Line momentum
                    ad_momentum = smoothed_ad - smoothed_ad.shift(self.period)

                    # Calculate AD Line volatility
                    ad_volatility = rolling_std(smoothed_ad, window=self.period)

                    # Combine features
                    enhanced_ad = smoothed_ad + ad_trend + ad_momentum - ad_volatility

                    return enhanced_ad.rename(f'vectorbt_enhanced_ad_line_{self.period}')
                except Exception as e:
                    self.logger.warning(f"VectorBT Enhanced AD Line calculation failed: {e}, using pandas fallback")

            # Final fallback to pandas
            smoothed_ad = ad_line.ewm(span=self.period).mean()
            ad_trend = smoothed_ad.diff().rolling(window=self.period).mean()
            ad_momentum = smoothed_ad - smoothed_ad.shift(self.period)
            ad_volatility = smoothed_ad.rolling(window=self.period).std()
            enhanced_ad = smoothed_ad + ad_trend + ad_momentum - ad_volatility

            return enhanced_ad.rename(f'vectorbt_enhanced_ad_line_{self.period}')

        except Exception as e:
            self.logger.error(f"Error generating Enhanced AD Line: {e}")
            return pd.Series(np.nan, index=data.index, name=f'vectorbt_enhanced_ad_line_{self.period}')

class VectorBTVolumeWeightedADLineGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized Volume-Weighted AD Line generator."""

    def __init__(self, period: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.period = period

        # Initialize VectorBT rolling optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=True, enable_parallel=True)
        else:
            self.rolling_optimizer = None

    @classmethod
    def _create_default_config(cls, period: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_volume_weighted_ad_line_{period}",
            category=FeatureCategory.VOLUME,
            description=f"VectorBT-optimized Volume-Weighted AD Line over {period} periods",
            required_columns=["high", "low", "close", "volume"],
            optional_columns=["open"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=True
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Volume-Weighted AD Line using VectorBT."""
        if data.empty or not all(col in data.columns for col in ['high', 'low', 'close', 'volume']):
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_volume_weighted_ad_line_{self.period}')

        try:
            high = data['high']
            low = data['low']
            close = data['close']
            volume = data['volume']

            # Calculate Money Flow Multiplier
            mfm = ((close - low) - (high - close)) / (high - low)
            mfm = mfm.fillna(0)  # Handle division by zero

            # Calculate Volume-Weighted Money Flow Volume
            vw_mfv = mfm * volume * volume  # Square volume for emphasis

            # Calculate Volume-Weighted AD Line
            vw_ad_line = vw_mfv.cumsum()

            # Apply smoothing using VectorBT rolling optimizer
            if self.rolling_optimizer:
                try:
                    # Smooth with volume-weighted moving average
                    volume_sum = self.rolling_optimizer.rolling_sum(volume, window=self.period)
                    vw_ad_sum = self.rolling_optimizer.rolling_sum(vw_ad_line, window=self.period)
                    smoothed_vw_ad = vw_ad_sum / volume_sum

                    return smoothed_vw_ad.rename(f'vectorbt_volume_weighted_ad_line_{self.period}')
                except Exception as e:
                    self.logger.warning(f"VectorBT rolling optimizer failed: {e}, using fallback")

            # Fallback to VectorBT direct operations
            if VECTORBT_AVAILABLE:
                try:
                    # Smooth with volume-weighted moving average
                    volume_sum = rolling_sum(volume, window=self.period)
                    vw_ad_sum = rolling_sum(vw_ad_line, window=self.period)
                    smoothed_vw_ad = vw_ad_sum / volume_sum

                    return smoothed_vw_ad.rename(f'vectorbt_volume_weighted_ad_line_{self.period}')
                except Exception as e:
                    self.logger.warning(f"VectorBT Volume-Weighted AD Line calculation failed: {e}, using pandas fallback")

            # Final fallback to pandas
            volume_sum = volume.rolling(window=self.period).sum()
            vw_ad_sum = vw_ad_line.rolling(window=self.period).sum()
            smoothed_vw_ad = vw_ad_sum / volume_sum

            return smoothed_vw_ad.rename(f'vectorbt_volume_weighted_ad_line_{self.period}')

        except Exception as e:
            self.logger.error(f"Error generating Volume-Weighted AD Line: {e}")
            return pd.Series(np.nan, index=data.index, name=f'vectorbt_volume_weighted_ad_line_{self.period}')

class VectorBTSmoothedOBVGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized Smoothed OBV generator with multiple smoothing techniques."""

    def __init__(self, period: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.period = period

        # Initialize VectorBT rolling optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=True, enable_parallel=True)
        else:
            self.rolling_optimizer = None

    @classmethod
    def _create_default_config(cls, period: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_smoothed_obv_{period}",
            category=FeatureCategory.VOLUME,
            description=f"VectorBT-optimized Smoothed OBV with multiple smoothing techniques over {period} periods",
            required_columns=["close", "volume"],
            optional_columns=["high", "low", "open"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=True
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Smoothed OBV using VectorBT."""
        if data.empty or not all(col in data.columns for col in ['close', 'volume']):
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_smoothed_obv_{self.period}')

        try:
            close = data['close']
            volume = data['volume']

            # Calculate price direction
            price_direction = np.where(close > close.shift(1), 1,
                                     np.where(close < close.shift(1), -1, 0))

            # Calculate basic OBV
            obv = (price_direction * volume).cumsum()
            obv_series = pd.Series(obv, index=close.index)

            # Apply multiple smoothing techniques using VectorBT rolling optimizer
            if self.rolling_optimizer:
                try:
                    # EMA smoothing
                    ema_smoothed = obv_series.ewm(span=self.period).mean()

                    # SMA smoothing
                    sma_smoothed = self.rolling_optimizer.rolling_mean(obv_series, window=self.period)

                    # WMA smoothing (weighted moving average)
                    weights = np.arange(1, self.period + 1)
                    wma_smoothed = obv_series.rolling(window=self.period).apply(
                        lambda x: np.average(x, weights=weights)
                    )

                    # Combine different smoothing techniques
                    smoothed_obv = (ema_smoothed + sma_smoothed + wma_smoothed) / 3

                    return smoothed_obv.rename(f'vectorbt_smoothed_obv_{self.period}')
                except Exception as e:
                    self.logger.warning(f"VectorBT rolling optimizer failed: {e}, using fallback")

            # Fallback to VectorBT direct operations
            if VECTORBT_AVAILABLE:
                try:
                    # EMA smoothing
                    ema_smoothed = obv_series.ewm(span=self.period).mean()

                    # SMA smoothing
                    sma_smoothed = rolling_mean(obv_series, window=self.period)

                    # WMA smoothing (weighted moving average)
                    weights = np.arange(1, self.period + 1)
                    wma_smoothed = obv_series.rolling(window=self.period).apply(
                        lambda x: np.average(x, weights=weights)
                    )

                    # Combine different smoothing techniques
                    smoothed_obv = (ema_smoothed + sma_smoothed + wma_smoothed) / 3

                    return smoothed_obv.rename(f'vectorbt_smoothed_obv_{self.period}')
                except Exception as e:
                    self.logger.warning(f"VectorBT Smoothed OBV calculation failed: {e}, using pandas fallback")

            # Final fallback to pandas
            ema_smoothed = obv_series.ewm(span=self.period).mean()
            sma_smoothed = obv_series.rolling(window=self.period).mean()
            weights = np.arange(1, self.period + 1)
            wma_smoothed = obv_series.rolling(window=self.period).apply(
                lambda x: np.average(x, weights=weights)
            )
            smoothed_obv = (ema_smoothed + sma_smoothed + wma_smoothed) / 3

            return smoothed_obv.rename(f'vectorbt_smoothed_obv_{self.period}')

        except Exception as e:
            self.logger.error(f"Error generating Smoothed OBV: {e}")
            return pd.Series(np.nan, index=data.index, name=f'vectorbt_smoothed_obv_{self.period}')

# Analyst Features - Volume pattern generators

class AnalystVolumePressureGenerator(VectorizedFeatureGenerator):
    """Generator for volume pressure feature."""

    def __init__(self):
        config = FeatureConfig(
            name="analyst_volume_pressure",
            category=FeatureCategory.VOLUME,
            description="Analyst volume pressure ((buy_volume - sell_volume) / total_volume)",
            required_columns=["volume", "close"],
            default_lookback=20,
            min_lookback=10,
            max_lookback=100,
            parameters={}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate volume pressure feature."""
        volume = data['volume']
        price_change = data['close'].pct_change()

        # Use VectorBT for volume pressure calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(volume):
            try:
                # Use price movement direction as proxy for buy/sell pressure
                volume_up = volume.where(price_change > 0, 0)
                volume_down = volume.where(price_change < 0, 0)

                volume_pressure = (volume_up - volume_down) / volume.replace(0, 1)
                return volume_pressure
            except Exception as e:
                self.logger.warning(f"VectorBT volume pressure calculation failed: {e}, using pandas fallback")
                # Use price movement direction as proxy for buy/sell pressure
                volume_up = volume.where(price_change > 0, 0)
                volume_down = volume.where(price_change < 0, 0)

                volume_pressure = (volume_up - volume_down) / volume.replace(0, 1)
                return volume_pressure
        else:
            # Use price movement direction as proxy for buy/sell pressure
            volume_up = volume.where(price_change > 0, 0)
            volume_down = volume.where(price_change < 0, 0)

            volume_pressure = (volume_up - volume_down) / volume.replace(0, 1)
            return volume_pressure

class AnalystVolumeTrendGenerator(VectorizedFeatureGenerator):
    """Generator for volume trend using linear regression."""

    def __init__(self, lookback: int = 20):
        config = FeatureConfig(
            name="analyst_volume_trend",
            category=FeatureCategory.VOLUME,
            description="Analyst volume trend using linear regression slope",
            required_columns=["volume"],
            default_lookback=lookback,
            min_lookback=10,
            max_lookback=100,
            parameters={"lookback": lookback}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.lookback = lookback

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate volume trend feature."""
        volume = data['volume']

        # Use VectorBT for volume trend calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(volume):
            try:
                def volume_trend(x):
                    if len(x) < 10:
                        return 0.0
                    try:
                        from scipy.stats import linregress
                        slope, _, _, _, _ = linregress(range(len(x)), x.values)
                        return slope
                    except:
                        return 0.0

                volume_trend_values = volume.rolling(self.lookback).apply(volume_trend)
                return volume_trend_values
            except Exception as e:
                self.logger.warning(f"VectorBT volume trend calculation failed: {e}, using pandas fallback")
                def volume_trend(x):
                    if len(x) < 10:
                        return 0.0
                    try:
                        from scipy.stats import linregress
                        slope, _, _, _, _ = linregress(range(len(x)), x.values)
                        return slope
                    except:
                        return 0.0

                volume_trend_values = volume.rolling(self.lookback).apply(volume_trend)
                return volume_trend_values
        else:
            def volume_trend(x):
                if len(x) < 10:
                    return 0.0
                try:
                    from scipy.stats import linregress
                    slope, _, _, _, _ = linregress(range(len(x)), x.values)
                    return slope
                except:
                    return 0.0

            volume_trend_values = volume.rolling(self.lookback).apply(volume_trend)
            return volume_trend_values

__all__ = [
    'VolumeFeatureGenerator',
    'VolumeSMAGenerator',
    'VolumeEMAGenerator',
    'VolumeRatioGenerator',
    'VolumeROCGenerator',
    'VolumeStdGenerator',
    'VolumePercentileGenerator',
    'VolumeTrendStrengthGenerator',
    'VolumeOscillatorGenerator',
    'VolumeMomentumGenerator',
    'VolumeVWAPGenerator',
    'VolumePriceTrendGenerator',
    'VolumeAccumulationDistributionGenerator',
    'VolumePriceCorrelationGenerator',
    'VolumePriceDivergenceGenerator',
    'PriceVolumeOscillatorGenerator',
    'AnalystVolumePressureGenerator',
    'AnalystVolumeTrendGenerator',
    'create_default_volume_generators'
]

# NEW FEATURES - Enhanced Volume Analysis

class VolumeZScoreGenerator(VectorizedFeatureGenerator):
    """Generator for volume z-score vs 60/252-bar history."""

    def __init__(self, short_window: int = 60, long_window: int = 252):
        config = FeatureConfig(
            name=f"volume_zscore_{short_window}_{long_window}",
            category=FeatureCategory.VOLUME,
            description=f"Volume z-score vs {short_window}/{long_window}-bar history",
            required_columns=["volume"],
            default_lookback=long_window,
            min_lookback=long_window,
            max_lookback=long_window,
            parameters={'short_window': short_window, 'long_window': long_window},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.short_window = short_window
        self.long_window = long_window

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        volume = data['volume'].values
        if len(volume) < self.long_window:
            return pd.Series(np.full(len(volume), np.nan), index=data.index)

        # Calculate volume z-score
        volume_zscore = np.full(len(volume), np.nan)
        for i in range(self.long_window - 1, len(volume)):
            # Short-term mean and std
            short_window_vol = volume[i - self.short_window + 1:i + 1]
            short_mean = np.mean(short_window_vol)
            short_std = np.std(short_window_vol, ddof=1)

            # Long-term mean and std
            long_window_vol = volume[i - self.long_window + 1:i + 1]
            long_mean = np.mean(long_window_vol)
            long_std = np.std(long_window_vol, ddof=1)

            if long_std > 0:
                volume_zscore[i] = (volume[i] - long_mean) / long_std

        return pd.Series(volume_zscore, index=data.index)

class VolumeMARatiosGenerator(VectorizedFeatureGenerator):
    """Generator for volume MA ratios and volume surprise."""

    def __init__(self, ma_period: int = 20, surprise_window: int = 10):
        config = FeatureConfig(
            name=f"volume_ma_ratios_{ma_period}_{surprise_window}",
            category=FeatureCategory.VOLUME,
            description=f"Volume MA ratios and surprise over {ma_period}/{surprise_window} periods",
            required_columns=["volume"],
            default_lookback=ma_period + surprise_window,
            min_lookback=ma_period + surprise_window,
            max_lookback=ma_period + surprise_window,
            parameters={'ma_period': ma_period, 'surprise_window': surprise_window},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.ma_period = ma_period
        self.surprise_window = surprise_window

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        volume = data['volume'].values
        if len(volume) < self.ma_period + self.surprise_window:
            return pd.Series(np.full(len(volume), np.nan), index=data.index)

        # Calculate volume MA ratios
        volume_ma_ratios = np.full(len(volume), np.nan)
        volume_surprise = np.full(len(volume), np.nan)

        for i in range(self.ma_period + self.surprise_window - 1, len(volume)):
            # Volume MA ratio
            ma_window = volume[i - self.ma_period + 1:i + 1]
            ma_volume = np.mean(ma_window)
            if ma_volume > 0:
                volume_ma_ratios[i] = volume[i] / ma_volume

            # Volume surprise (actual - expected)
            if i >= self.surprise_window:
                expected_window = volume[i - self.surprise_window:i]
                expected_volume = np.mean(expected_window)
                volume_surprise[i] = volume[i] - expected_volume

        return pd.Series(volume_ma_ratios, index=data.index)

class CMFGenerator(VectorizedFeatureGenerator):
    """Generator for Chaikin Money Flow (CMF)."""

    def __init__(self, period: int = 20):
        config = FeatureConfig(
            name=f"cmf_{period}",
            category=FeatureCategory.VOLUME,
            description=f"Chaikin Money Flow over {period} periods",
            required_columns=["close", "high", "low", "volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={'period': period},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        volume = data['volume'].values

        if len(close) < self.period:
            return pd.Series(np.full(len(close), np.nan), index=data.index)

        # Calculate CMF
        cmf = np.full(len(close), np.nan)
        for i in range(self.period - 1, len(close)):
            # Money Flow Multiplier
            mfm = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
            mfm = np.nan_to_num(mfm, nan=0.0)  # Handle division by zero

            # Money Flow Volume
            mfv = mfm * volume[i]

            # CMF = sum(MFV) / sum(Volume) over period
            period_mfv = []
            period_vol = []
            for j in range(i - self.period + 1, i + 1):
                if high[j] != low[j]:  # Avoid division by zero
                    period_mfm = ((close[j] - low[j]) - (high[j] - close[j])) / (high[j] - low[j])
                    period_mfm = np.nan_to_num(period_mfm, nan=0.0)
                    period_mfv.append(period_mfm * volume[j])
                    period_vol.append(volume[j])

            if len(period_vol) > 0 and sum(period_vol) > 0:
                cmf[i] = sum(period_mfv) / sum(period_vol)

        return pd.Series(cmf, index=data.index)

class VWAPDeviationsGenerator(VectorizedFeatureGenerator):
    """Generator for VWAP deviations and closing-VWAP gap."""

    def __init__(self, vwap_window: int = 20):
        config = FeatureConfig(
            name=f"vwap_deviations_{vwap_window}",
            category=FeatureCategory.VOLUME,
            description=f"VWAP deviations and closing-VWAP gap over {vwap_window} periods",
            required_columns=["close", "high", "low", "volume"],
            default_lookback=vwap_window,
            min_lookback=vwap_window,
            max_lookback=vwap_window,
            parameters={'vwap_window': vwap_window},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.vwap_window = vwap_window

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        volume = data['volume'].values

        if len(close) < self.vwap_window:
            return pd.Series(np.full(len(close), np.nan), index=data.index)

        # Calculate VWAP deviations
        vwap_deviations = np.full(len(close), np.nan)
        closing_vwap_gap = np.full(len(close), np.nan)

        for i in range(self.vwap_window - 1, len(close)):
            # Calculate VWAP for the window
            window_high = high[i - self.vwap_window + 1:i + 1]
            window_low = low[i - self.vwap_window + 1:i + 1]
            window_close = close[i - self.vwap_window + 1:i + 1]
            window_volume = volume[i - self.vwap_window + 1:i + 1]

            # Typical price
            typical_price = (window_high + window_low + window_close) / 3

            # VWAP
            vwap = np.sum(typical_price * window_volume) / np.sum(window_volume)

            if vwap > 0:
                # VWAP deviation
                vwap_deviations[i] = (close[i] - vwap) / vwap

                # Closing-VWAP gap
                closing_vwap_gap[i] = close[i] - vwap

        return pd.Series(vwap_deviations, index=data.index)

class OrderFlowImbalanceGenerator(VectorizedFeatureGenerator):
    """Generator for order flow imbalance (signed volume)."""

    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"order_flow_imbalance_{window}",
            category=FeatureCategory.VOLUME,
            description=f"Order flow imbalance over {window} periods",
            required_columns=["close", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close'].values
        volume = data['volume'].values

        if len(close) < self.window + 1:
            return pd.Series(np.full(len(close), np.nan), index=data.index)

        # Calculate order flow imbalance
        ofi = np.full(len(close), np.nan)

        for i in range(1, len(close)):
            # Price change direction
            price_change = close[i] - close[i-1]

            # Signed volume (positive for buying pressure, negative for selling pressure)
            if price_change > 0:
                signed_volume = volume[i]
            elif price_change < 0:
                signed_volume = -volume[i]
            else:
                signed_volume = 0

            # Rolling sum of signed volume
            if i >= self.window:
                window_signed_vol = []
                for j in range(i - self.window + 1, i + 1):
                    if j > 0:
                        price_chg = close[j] - close[j-1]
                        if price_chg > 0:
                            window_signed_vol.append(volume[j])
                        elif price_chg < 0:
                            window_signed_vol.append(-volume[j])
                        else:
                            window_signed_vol.append(0)

                ofi[i] = sum(window_signed_vol)

        return pd.Series(ofi, index=data.index)

class VolumeVolatilityElasticityGenerator(VectorizedFeatureGenerator):
    """Generator for volume-volatility elasticity."""

    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"volume_volatility_elasticity_{window}",
            category=FeatureCategory.VOLUME,
            description=f"Volume-volatility elasticity over {window} periods",
            required_columns=["close", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close'].values
        volume = data['volume'].values

        if len(close) < self.window + 1:
            return pd.Series(np.full(len(close), np.nan), index=data.index)

        # Calculate returns and absolute returns
        returns = np.diff(close) / close[:-1]
        abs_returns = np.abs(returns)
        returns = np.concatenate([[np.nan], returns])
        abs_returns = np.concatenate([[np.nan], abs_returns])

        # Calculate volume-volatility elasticity
        elasticity = np.full(len(close), np.nan)

        for i in range(self.window, len(close)):
            window_abs_returns = abs_returns[i - self.window + 1:i + 1]
            window_volume = volume[i - self.window + 1:i + 1]

            # Filter out NaN values
            valid_mask = np.isfinite(window_abs_returns) & np.isfinite(window_volume)
            if np.sum(valid_mask) > 1:
                valid_abs_returns = window_abs_returns[valid_mask]
                valid_volume = window_volume[valid_mask]

                # Calculate correlation
                if len(valid_abs_returns) > 1 and np.std(valid_abs_returns) > 0 and np.std(valid_volume) > 0:
                    correlation = np.corrcoef(valid_abs_returns, valid_volume)[0, 1]
                    if not np.isnan(correlation):
                        elasticity[i] = correlation

        return pd.Series(elasticity, index=data.index)

    def _optimized_rolling_operation(self, data: pd.Series, operation: str,
                                   window: int, **kwargs) -> pd.Series:
        """Perform rolling operation using centralized VectorBTRollingOptimizer."""
        if not hasattr(self, 'rolling_optimizer'):
            self.rolling_optimizer = get_vectorbt_rolling_optimizer()

        try:
            if operation == 'mean':
                return self.rolling_optimizer.rolling_mean(data, window, **kwargs)
            elif operation == 'std':
                return self.rolling_optimizer.rolling_std(data, window, **kwargs)
            elif operation == 'var':
                return self.rolling_optimizer.rolling_var(data, window, **kwargs)
            elif operation == 'min':
                return self.rolling_optimizer.rolling_min(data, window, **kwargs)
            elif operation == 'max':
                return self.rolling_optimizer.rolling_max(data, window, **kwargs)
            elif operation == 'sum':
                return self.rolling_optimizer.rolling_sum(data, window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT rolling operation failed: {e}, using fallback")
            return self._fallback_rolling_operation(data, operation, window, **kwargs)

    def _fallback_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        rolling_obj = data.rolling(window=window, **kwargs)

        if operation == 'mean':
            return rolling_obj.mean()
        elif operation == 'std':
            return rolling_obj.std()
        elif operation == 'var':
            return rolling_obj.var()
        elif operation == 'min':
            return rolling_obj.min()
        elif operation == 'max':
            return rolling_obj.max()
        elif operation == 'sum':
            return rolling_obj.sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def _normalize_feature(self, data: pd.Series, method: str = 'zscore') -> pd.Series:
        """Normalize feature using direct scaling to avoid circular imports."""
        try:
            if method == 'zscore':
                return (data - data.mean()) / data.std()
            elif method == 'minmax':
                return (data - data.min()) / (data.max() - data.min())
            elif method == 'robust':
                median = data.median()
                mad = (data - median).abs().median()
                return (data - median) / mad
            else:
                logger.warning(f"Unsupported normalization method: {method}, using zscore")
                return (data - data.mean()) / data.std()
        except Exception as e:
            logger.warning(f"Normalization failed: {e}, using simple zscore")
            return (data - data.mean()) / data.std()

    def _fallback_normalize(self, data: pd.Series, method: str = 'zscore') -> pd.Series:
        """Fallback normalization using pandas/numpy."""
        if method == 'zscore':
            return (data - data.mean()) / data.std()
        elif method == 'minmax':
            return (data - data.min()) / (data.max() - data.min())
        elif method == 'robust':
            median = data.median()
            mad = (data - median).abs().median()
            return (data - median) / mad
        else:
            return data

# ============================================================================
# ADVANCED VOLUME FEATURES
# ============================================================================

@dataclass
class VolumeConfig:
    """Configuration for advanced volume features."""
    enable_obv: bool = True
    enable_ad: bool = True
    enable_mfi: bool = True
    enable_vwap: bool = True
    enable_volume_profile: bool = True
    obv_window: int = 20
    ad_window: int = 20
    mfi_window: int = 14
    vwap_window: int = 20
    volume_profile_bins: int = 10

class AdvancedVolumeFeatures(VectorizedFeatureGenerator):
    """Advanced volume features with VectorBT optimization.

    This generator creates comprehensive volume features including OBV, AD, MFI,
    VWAP, and volume profile analysis with full VectorBT optimization.

    Key Features:
    - On-Balance Volume (OBV) with VectorBT optimization
    - Accumulation/Distribution Line (AD) with advanced metrics
    - Money Flow Index (MFI) and related indicators
    - Volume Rate of Change and momentum indicators
    - Volume-weighted average price (VWAP) with VectorBT
    - Volume profile analysis and clustering

    Parameters:
    - config: VolumeConfig object with generator parameters

    Returns:
    - Dict[str, np.ndarray]: Dictionary of advanced volume features

    Example:
        >>> config = VolumeConfig(enable_obv=True, enable_vwap=True)
        >>> generator = AdvancedVolumeFeatures(config)
        >>> features = generator.generate_features(data)
        >>> print(f"Generated {len(features)} advanced volume features")
    """

    def __init__(self, config: Optional[VolumeConfig] = None):
        if config is None:
            config = VolumeConfig()

        self.volume_config = config

        feature_config = FeatureConfig(
            name="advanced_volume_features",
            category=FeatureCategory.VOLUME,
            description="Advanced volume features with VectorBT optimization",
            required_columns=["close", "volume"],
            optional_columns=["high", "low", "open"],
            default_lookback=20,
            min_lookback=10,
            max_lookback=100,
            parameters={}
        )

        super().__init__(feature_config, enable_matrix_ops=True, enable_vectorization_optimization=True)

        # Initialize VectorBT optimizer
        self.vectorbt_optimizer = None
        if ROLLING_OPTIMIZER_AVAILABLE:
            try:
                self.vectorbt_optimizer = get_vectorbt_rolling_optimizer()
            except Exception as e:
                tprint(f"⚠️ VectorBT optimizer initialization failed: {e}")

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate advanced volume features."""
        features = self.generate_features(data, **kwargs)

        # Return the first feature as representative
        if features:
            first_feature_name = list(features.keys())[0]
            return pd.Series(features[first_feature_name], index=data.index[:len(features[first_feature_name])])
        else:
            return pd.Series(np.zeros(len(data)), index=data.index)

    def generate_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate all advanced volume features."""
        features = {}

        try:
            # On-Balance Volume (OBV)
            if self.volume_config.enable_obv and VECTORBT_AVAILABLE and OBV is not None:
                try:
                    obv = OBV.run(data['close'], data['volume'])
                    features['obv'] = obv.values
                    features['obv_sma'] = rolling_mean(obv, window=self.volume_config.obv_window).values
                except Exception as e:
                    tprint(f"⚠️ OBV calculation failed: {e}")

            # Accumulation/Distribution Line (AD)
            if self.volume_config.enable_ad and VECTORBT_AVAILABLE and AD is not None:
                try:
                    ad = AD.run(data['high'], data['low'], data['close'], data['volume'])
                    features['ad'] = ad.values
                    features['ad_sma'] = rolling_mean(ad, window=self.volume_config.ad_window).values
                except Exception as e:
                    tprint(f"⚠️ AD calculation failed: {e}")

            # Money Flow Index (MFI)
            if self.volume_config.enable_mfi and VECTORBT_AVAILABLE and MFI is not None:
                try:
                    mfi = MFI.run(data['high'], data['low'], data['close'], data['volume'], window=self.volume_config.mfi_window)
                    features['mfi'] = mfi.values
                except Exception as e:
                    tprint(f"⚠️ MFI calculation failed: {e}")

            # Volume-Weighted Average Price (VWAP)
            if self.volume_config.enable_vwap and VECTORBT_AVAILABLE:
                try:
                    typical_price = (data['high'] + data['low'] + data['close']) / 3
                    vwap = (typical_price * data['volume']).rolling(window=self.volume_config.vwap_window).sum() / data['volume'].rolling(window=self.volume_config.vwap_window).sum()
                    features['vwap'] = vwap.values
                    features['vwap_ratio'] = (data['close'] / vwap).values
                except Exception as e:
                    tprint(f"⚠️ VWAP calculation failed: {e}")

            # Volume Rate of Change
            try:
                volume_roc = data['volume'].pct_change(periods=5)
                features['volume_roc'] = volume_roc.values
            except Exception as e:
                tprint(f"⚠️ Volume ROC calculation failed: {e}")

            # Volume Profile Analysis
            if self.volume_config.enable_volume_profile:
                try:
                    volume_profile = self._calculate_volume_profile(data)
                    features.update(volume_profile)
                except Exception as e:
                    tprint(f"⚠️ Volume profile calculation failed: {e}")

        except Exception as e:
            tprint(f"⚠️ Advanced volume features generation failed: {e}")

        return features

    def _calculate_volume_profile(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate volume profile features."""
        features = {}

        try:
            if 'high' not in data.columns or 'low' not in data.columns:
                return features

            # Price range
            price_range = data['high'] - data['low']
            price_center = (data['high'] + data['low']) / 2

            # Volume-weighted price
            volume_weighted_price = (data['close'] * data['volume']).rolling(window=20).sum() / data['volume'].rolling(window=20).sum()

            # Volume profile features
            features['volume_profile_center'] = price_center.values
            features['volume_profile_range'] = price_range.values
            features['volume_profile_vwp'] = volume_weighted_price.values

            # Volume clustering
            volume_ma = data['volume'].rolling(window=20).mean()
            volume_clustering = (data['volume'] / volume_ma).values
            features['volume_clustering'] = volume_clustering

        except Exception as e:
            tprint(f"⚠️ Volume profile calculation failed: {e}")

        return features

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_advanced_volume_generators() -> List[FeatureGenerator]:
    """Create advanced volume feature generators."""
    generators = []

    # Different volume configurations
    configs = [
        VolumeConfig(enable_obv=True, enable_ad=True, enable_mfi=True, enable_vwap=True),
        VolumeConfig(enable_obv=True, enable_ad=False, enable_mfi=True, enable_vwap=True),
        VolumeConfig(enable_obv=False, enable_ad=True, enable_mfi=True, enable_vwap=True),
    ]

    for config in configs:
        generators.append(AdvancedVolumeFeatures(config))

    return generators

def process_advanced_volume_features_batch(data: pd.DataFrame,
                                         generators: Optional[List[FeatureGenerator]] = None,
                                         use_vectorbt: bool = True,
                                         **kwargs) -> pd.DataFrame:
    """
    Process advanced volume features in batch using VectorBT optimizations.

    Args:
        data: Input OHLCV data
        generators: List of feature generators (uses default if None)
        use_vectorbt: Whether to use VectorBT batch processing
        **kwargs: Additional parameters

    Returns:
        DataFrame with generated advanced volume features
    """
    if generators is None:
        generators = create_advanced_volume_generators()

    if use_vectorbt and OPTIMIZATION_AVAILABLE:
        try:
            # Use unified optimization system for batch processing
            from ..utils.unified_optimization_system import get_unified_optimization_system
            unified_optimizer = get_unified_optimization_system()

            # Process features in batch
            result = unified_optimizer.process_features_batch(data, generators, **kwargs)
            return result

        except Exception as e:
            warnings.warn(f"VectorBT batch processing failed: {e}, using sequential processing")
            return _process_advanced_volume_features_sequential(data, generators, **kwargs)
    else:
        return _process_advanced_volume_features_sequential(data, generators, **kwargs)

def _process_advanced_volume_features_sequential(data: pd.DataFrame,
                                               generators: List[FeatureGenerator],
                                               **kwargs) -> pd.DataFrame:
    """Process advanced volume features sequentially (fallback).

    Args:
        data: Input OHLCV data
        generators: List of feature generators to process
        **kwargs: Additional arguments for generators

    Returns:
        DataFrame with processed features
    """

    results = []

    for generator in generators:
        try:
            feature_result = generator._generate_feature(data, **kwargs)
            if not feature_result.empty:
                results.append(feature_result)
        except Exception as e:
            warnings.warn(f"Generator {generator.__class__.__name__} failed: {e}")
            continue

    if results:
        return pd.concat(results, axis=1)
    else:
        return pd.DataFrame(index=data.index)
