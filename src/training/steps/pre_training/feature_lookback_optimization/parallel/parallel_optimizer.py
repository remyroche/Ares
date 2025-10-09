"""
Parallel Feature Lookback Optimizer

Implements high-performance parallel processing for feature lookback optimization
using multiprocessing.Pool and joblib for CPU-bound numeric work.
"""

import os
import time
import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc

import numpy as np
import pandas as pd

# Try to import joblib for better parallel processing
try:
    from joblib import Parallel, delayed, Memory
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    Parallel = None
    delayed = None
    Memory = None

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# Import hardware optimization tools
from src.utils.hardware.unified_hardware_manager import (
    UnifiedHardwareManager, WorkloadType, OptimizationLevel, HardwareConfig
)
from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
from src.utils.hardware.m1_gpu_utils import M1GPUManager
from src.utils.hardware.advanced_memory_optimizer import get_advanced_memory_optimizer

from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
from src.utils.logger import get_logger
from src.utils.matrix_operations import get_unified_matrix_operations, get_batch_matrix_processor


@dataclass
class ParallelConfig:
    """Configuration for parallel processing with hardware optimization."""
    max_workers: int = None
    chunk_size: int = 1000
    use_joblib: bool = True
    use_gpu: bool = False
    memory_mapping: bool = True
    cache_dir: str = "parallel_cache"
    enable_compression: bool = True
    pin_memory: bool = True  # For GPU transfers
    l3_cache_size_mb: int = 32  # L3 cache size for chunk tuning
    
    # Hardware optimization settings
    workload_type: WorkloadType = WorkloadType.FEATURE_ENGINEERING
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    enable_hardware_optimization: bool = True
    enable_memory_pooling: bool = True
    enable_gpu_acceleration: bool = False
    memory_limit_gb: float = 8.0
    enable_thermal_monitoring: bool = True
    enable_adaptive_optimization: bool = True


@dataclass
class FeatureGroup:
    """Represents a group of features for parallel processing."""
    features: List[str]
    group_id: str
    priority: int = 0
    estimated_complexity: float = 1.0
    dependencies: List[str] = None


class ParallelFeatureOptimizer:
    """
    High-performance parallel feature lookback optimizer.
    
    Uses multiprocessing.Pool and joblib for CPU-bound work,
    with GPU acceleration for matrix operations when available.
    """
    
    def __init__(self, config: Optional[ParallelConfig] = None, logger=None):
        """Initialize the parallel optimizer with hardware optimization."""
        self.config = config or ParallelConfig()
        self.logger = logger or get_logger('ParallelFeatureOptimizer')
        
        tprint(f"🚀 Initializing Hardware-Optimized Parallel Feature Optimizer")
        
        # Initialize hardware optimization components
        if self.config.enable_hardware_optimization:
            self._initialize_hardware_optimization()
        else:
            self.hardware_manager = None
            self.cpu_optimizer = None
            self.memory_optimizer = None
            self.gpu_manager = None
        
        # Determine optimal number of workers using hardware optimization
        if self.config.max_workers is None:
            if self.cpu_optimizer:
                self.config.max_workers = self.cpu_optimizer.get_optimal_worker_count()
            else:
                self.config.max_workers = min(mp.cpu_count(), 8)  # Cap at 8 for memory reasons
        
        # Optimize chunk size based on L3 cache
        if self.config.l3_cache_size_mb > 0:
            self.config.chunk_size = self._optimize_chunk_size()
        
        tprint_info(f"   → Max workers: {self.config.max_workers}")
        tprint_info(f"   → Chunk size: {self.config.chunk_size}")
        tprint_info(f"   → Joblib available: {JOBLIB_AVAILABLE}")
        tprint_info(f"   → CuPy available: {CUPY_AVAILABLE}")
        tprint_info(f"   → Hardware optimization: {self.config.enable_hardware_optimization}")
        
        # Initialize joblib memory cache if available
        if JOBLIB_AVAILABLE and self.config.use_joblib:
            self.memory = Memory(self.config.cache_dir, verbose=0)
            self._cached_optimize_feature = self.memory.cache(self._optimize_single_feature)
        else:
            self.memory = None
            self._cached_optimize_feature = self._optimize_single_feature
        
        # Initialize matrix operations with hardware optimization
        self.matrix_ops = get_unified_matrix_operations()
        self.batch_processor = get_batch_matrix_processor()
        
        # Performance tracking
        self.performance_stats = {
            'total_features': 0,
            'successful_features': 0,
            'failed_features': 0,
            'total_time': 0.0,
            'parallel_efficiency': 0.0,
            'memory_peak_mb': 0.0,
            'gpu_utilization': 0.0,
            'cpu_utilization': 0.0,
            'thermal_state': 'normal'
        }
        
        # Feature groups for intelligent batching
        self.feature_groups: List[FeatureGroup] = []
        
        tprint_success("✅ Hardware-Optimized Parallel Feature Optimizer initialized")
    
    def _initialize_hardware_optimization(self):
        """Initialize hardware optimization components."""
        tprint("🔧 Initializing hardware optimization components...")
        
        try:
            # Create hardware configuration
            hw_config = HardwareConfig(
                cpu_optimization_level=self.config.optimization_level,
                gpu_optimization_level=self.config.optimization_level,
                memory_optimization_level=self.config.optimization_level,
                memory_limit_gb=self.config.memory_limit_gb,
                enable_gpu_memory_pooling=self.config.enable_memory_pooling,
                enable_thermal_monitoring=self.config.enable_thermal_monitoring,
                enable_adaptive_optimization=self.config.enable_adaptive_optimization
            )
            
            # Initialize unified hardware manager
            self.hardware_manager = UnifiedHardwareManager(hw_config)
            
            # Initialize individual optimizers
            self.cpu_optimizer = M1CPUOptimizer()
            self.memory_optimizer = M1MemoryOptimizer(self.config.memory_limit_gb)
            
            if self.config.enable_gpu_acceleration:
                self.gpu_manager = M1GPUManager()
            else:
                self.gpu_manager = None
            
            # Optimize for feature engineering workload
            if self.hardware_manager:
                self.hardware_manager.optimize_for_workload(
                    self.config.workload_type, 
                    self.config.optimization_level
                )
            
            # Start memory monitoring
            if self.memory_optimizer:
                self.memory_optimizer.start_monitoring()
            
            tprint_success("✅ Hardware optimization components initialized")
            
        except Exception as e:
            tprint_warning(f"⚠️ Hardware optimization initialization failed: {e}")
            self.hardware_manager = None
            self.cpu_optimizer = None
            self.memory_optimizer = None
            self.gpu_manager = None
    
    def _optimize_chunk_size(self) -> int:
        """Optimize chunk size based on L3 cache size and data characteristics."""
        # Estimate optimal chunk size based on L3 cache
        # Assume each feature uses ~8 bytes (float64) + overhead
        bytes_per_feature = 8 * 1.5  # 8 bytes + 50% overhead
        optimal_chunk_size = int((self.config.l3_cache_size_mb * 1024 * 1024) / bytes_per_feature)
        
        # Ensure reasonable bounds
        optimal_chunk_size = max(100, min(optimal_chunk_size, 10000))
        
        tprint_info(f"   → Optimized chunk size: {optimal_chunk_size} (L3 cache: {self.config.l3_cache_size_mb}MB)")
        return optimal_chunk_size
    
    def create_feature_groups(self, features: List[str], 
                            group_size: int = 10,
                            complexity_estimator: Optional[Callable] = None) -> List[FeatureGroup]:
        """
        Create intelligent feature groups for parallel processing.
        
        Groups features by complexity and dependencies for optimal parallel execution.
        """
        tprint("📊 Creating intelligent feature groups...")
        
        if complexity_estimator is None:
            # Simple complexity estimator based on feature name patterns
            def complexity_estimator(feature: str) -> float:
                complexity = 1.0
                if 'interaction' in feature.lower():
                    complexity *= 2.0
                if 'cross_timeframe' in feature.lower():
                    complexity *= 1.5
                if 'regime' in feature.lower():
                    complexity *= 1.3
                if 'wavelet' in feature.lower():
                    complexity *= 1.8
                return complexity
        
        # Sort features by complexity (descending)
        features_with_complexity = [
            (feature, complexity_estimator(feature)) 
            for feature in features
        ]
        features_with_complexity.sort(key=lambda x: x[1], reverse=True)
        
        # Create groups with balanced complexity
        groups = []
        current_group = []
        current_complexity = 0.0
        group_id = 0
        
        for feature, complexity in features_with_complexity:
            if len(current_group) >= group_size or current_complexity + complexity > 10.0:
                if current_group:
                    groups.append(FeatureGroup(
                        features=current_group,
                        group_id=f"group_{group_id}",
                        estimated_complexity=current_complexity
                    ))
                    group_id += 1
                    current_group = []
                    current_complexity = 0.0
            
            current_group.append(feature)
            current_complexity += complexity
        
        # Add remaining features
        if current_group:
            groups.append(FeatureGroup(
                features=current_group,
                group_id=f"group_{group_id}",
                estimated_complexity=current_complexity
            ))
        
        self.feature_groups = groups
        tprint_success(f"✅ Created {len(groups)} feature groups")
        
        for group in groups:
            tprint_info(f"   → {group.group_id}: {len(group.features)} features, complexity={group.estimated_complexity:.2f}")
        
        return groups
    
    def optimize_features_parallel(self, 
                                 features: List[str],
                                 lookback_range: range,
                                 data: pd.DataFrame,
                                 labels: pd.Series,
                                 method: str = "grid_search",
                                 **kwargs) -> Dict[str, Any]:
        """
        Optimize features in parallel using multiprocessing or joblib.
        
        Args:
            features: List of feature names to optimize
            lookback_range: Range of lookback periods to test
            data: Feature data
            labels: Target labels
            method: Optimization method
            **kwargs: Additional optimization parameters
            
        Returns:
            Dictionary of optimization results
        """
        start_time = time.time()
        tprint(f"🚀 Starting parallel feature optimization")
        tprint_info(f"   → Features: {len(features)}")
        tprint_info(f"   → Lookback range: {lookback_range.start}-{lookback_range.stop}")
        tprint_info(f"   → Method: {method}")
        tprint_info(f"   → Workers: {self.config.max_workers}")
        
        # Create feature groups
        groups = self.create_feature_groups(features)
        
        # Prepare data for parallel processing
        data_hash = self._compute_data_hash(data, labels)
        shared_data = self._prepare_shared_data(data, labels, data_hash)
        
        # Track performance
        self.performance_stats['total_features'] = len(features)
        self.performance_stats['total_time'] = 0.0
        
        results = {}
        successful_features = 0
        failed_features = 0
        
        try:
            if self.config.use_joblib and JOBLIB_AVAILABLE:
                # Use joblib for better memory management
                results = self._optimize_with_joblib(groups, lookback_range, shared_data, method, **kwargs)
            else:
                # Use multiprocessing.Pool
                results = self._optimize_with_multiprocessing(groups, lookback_range, shared_data, method, **kwargs)
            
            # Count successes and failures
            for group_results in results.values():
                for feature_result in group_results.values():
                    if feature_result.get('success', False):
                        successful_features += 1
                    else:
                        failed_features += 1
            
        except Exception as e:
            tprint_error(f"❌ Parallel optimization failed: {e}")
            self.logger.error(f"Parallel optimization error: {e}", exc_info=True)
            return {'error': str(e), 'success': False}
        
        # Update performance stats
        total_time = time.time() - start_time
        self.performance_stats['total_time'] = total_time
        self.performance_stats['successful_features'] = successful_features
        self.performance_stats['failed_features'] = failed_features
        self.performance_stats['parallel_efficiency'] = successful_features / (total_time * self.config.max_workers)
        
        tprint_success(f"✅ Parallel optimization completed")
        tprint_info(f"   → Total time: {total_time:.2f}s")
        tprint_info(f"   → Successful: {successful_features}")
        tprint_info(f"   → Failed: {failed_features}")
        tprint_info(f"   → Parallel efficiency: {self.performance_stats['parallel_efficiency']:.2f}")
        
        return {
            'results': results,
            'performance_stats': self.performance_stats,
            'success': True
        }
    
    def _optimize_with_joblib(self, 
                            groups: List[FeatureGroup],
                            lookback_range: range,
                            shared_data: Dict[str, Any],
                            method: str,
                            **kwargs) -> Dict[str, Any]:
        """Optimize using joblib for better memory management."""
        tprint("🔧 Using joblib for parallel optimization...")
        
        # Prepare tasks for joblib
        tasks = []
        for group in groups:
            for feature in group.features:
                task = delayed(self._optimize_single_feature)(
                    feature, lookback_range, shared_data, method, **kwargs
                )
                tasks.append((group.group_id, feature, task))
        
        # Execute in parallel
        parallel = Parallel(
            n_jobs=self.config.max_workers,
            backend='multiprocessing',
            batch_size=max(1, len(tasks) // (self.config.max_workers * 4)),
            verbose=1
        )
        
        # Execute tasks
        task_results = parallel(tasks)
        
        # Organize results by group
        results = {}
        for (group_id, feature, result) in task_results:
            if group_id not in results:
                results[group_id] = {}
            results[group_id][feature] = result
        
        return results
    
    def _optimize_with_multiprocessing(self, 
                                     groups: List[FeatureGroup],
                                     lookback_range: range,
                                     shared_data: Dict[str, Any],
                                     method: str,
                                     **kwargs) -> Dict[str, Any]:
        """Optimize using hardware-optimized multiprocessing.Pool."""
        tprint("🔧 Using hardware-optimized multiprocessing.Pool for parallel optimization...")
        
        results = {}
        
        # Use hardware-optimized process pool if available
        if self.cpu_optimizer:
            executor = self.cpu_optimizer.create_optimized_process_pool(self.config.max_workers)
        else:
            executor = ProcessPoolExecutor(max_workers=self.config.max_workers)
        
        try:
            with executor:
                # Submit all tasks
                future_to_group = {}
                for group in groups:
                    for feature in group.features:
                        future = executor.submit(
                            self._optimize_single_feature,
                            feature, lookback_range, shared_data, method, **kwargs
                        )
                        future_to_group[future] = (group.group_id, feature)
                
                # Collect results with hardware monitoring
                for future in as_completed(future_to_group):
                    group_id, feature = future_to_group[future]
                    try:
                        result = future.result()
                        if group_id not in results:
                            results[group_id] = {}
                        results[group_id][feature] = result
                        
                        # Update performance stats
                        self._update_performance_stats()
                        
                    except Exception as e:
                        tprint_error(f"❌ Feature {feature} optimization failed: {e}")
                        if group_id not in results:
                            results[group_id] = {}
                        results[group_id][feature] = {'success': False, 'error': str(e)}
        
        except Exception as e:
            tprint_error(f"❌ Multiprocessing execution failed: {e}")
            raise
        
        return results
    
    def _optimize_single_feature(self, 
                               feature: str,
                               lookback_range: range,
                               shared_data: Dict[str, Any],
                               method: str,
                               **kwargs) -> Dict[str, Any]:
        """
        Optimize a single feature (runs in worker process).
        
        This method is designed to be called in parallel and should be
        self-contained with minimal shared state.
        """
        try:
            # Reconstruct data from shared data
            data = self._reconstruct_data(shared_data)
            
            # Extract feature data
            if feature not in data.columns:
                return {'success': False, 'error': f'Feature {feature} not found in data'}
            
            feature_data = data[feature].values
            labels = shared_data['labels']
            
            # Optimize lookback period
            best_lookback = None
            best_score = -np.inf
            scores = []
            
            for lookback in lookback_range:
                try:
                    # Calculate feature with this lookback
                    feature_with_lookback = self._apply_lookback(feature_data, lookback)
                    
                    # Calculate score (e.g., information coefficient)
                    score = self._calculate_ic(feature_with_lookback, labels)
                    scores.append(score)
                    
                    if score > best_score:
                        best_score = score
                        best_lookback = lookback
                        
                except Exception as e:
                    tprint_debug(f"Lookback {lookback} failed for {feature}: {e}")
                    scores.append(-np.inf)
            
            return {
                'success': True,
                'feature': feature,
                'best_lookback': best_lookback,
                'best_score': best_score,
                'all_scores': scores,
                'lookback_range': list(lookback_range)
            }
            
        except Exception as e:
            return {
                'success': False,
                'feature': feature,
                'error': str(e)
            }
    
    def _compute_data_hash(self, data: pd.DataFrame, labels: pd.Series) -> str:
        """Compute hash of data for caching."""
        data_str = f"{data.shape}_{data.columns.tolist()}_{labels.shape}_{labels.name}"
        return hashlib.md5(data_str.encode()).hexdigest()[:16]
    
    def _prepare_shared_data(self, data: pd.DataFrame, labels: pd.Series, data_hash: str) -> Dict[str, Any]:
        """Prepare data for sharing across processes."""
        return {
            'data_hash': data_hash,
            'data_shape': data.shape,
            'data_columns': data.columns.tolist(),
            'data_values': data.values.tobytes() if hasattr(data.values, 'tobytes') else data.values.tolist(),
            'labels': labels.values.tolist(),
            'labels_name': labels.name
        }
    
    def _reconstruct_data(self, shared_data: Dict[str, Any]) -> pd.DataFrame:
        """Reconstruct DataFrame from shared data."""
        values = np.frombuffer(shared_data['data_values'], dtype=np.float64).reshape(shared_data['data_shape'])
        return pd.DataFrame(values, columns=shared_data['data_columns'])
    
    def _apply_lookback(self, feature_data: np.ndarray, lookback: int) -> np.ndarray:
        """Apply lookback period to feature data."""
        if lookback <= 0:
            return feature_data
        
        # Simple rolling mean as example - replace with actual feature calculation
        result = np.full_like(feature_data, np.nan)
        for i in range(lookback - 1, len(feature_data)):
            result[i] = np.mean(feature_data[i - lookback + 1:i + 1])
        
        return result
    
    def _calculate_ic(self, feature_data: np.ndarray, labels: np.ndarray) -> float:
        """Calculate information coefficient between feature and labels with hardware optimization."""
        # Remove NaN values
        mask = ~(np.isnan(feature_data) | np.isnan(labels))
        if np.sum(mask) < 10:  # Need minimum samples
            return -np.inf
        
        feature_clean = feature_data[mask]
        labels_clean = labels[mask]
        
        # Use GPU acceleration if available
        if self.gpu_manager and CUPY_AVAILABLE:
            try:
                # Convert to GPU arrays
                feature_gpu = cp.asarray(feature_clean)
                labels_gpu = cp.asarray(labels_clean)
                
                # Calculate correlation on GPU
                correlation = cp.corrcoef(feature_gpu, labels_gpu)[0, 1]
                correlation = float(correlation)  # Convert back to CPU
                
                return correlation if not np.isnan(correlation) else -np.inf
            except Exception as e:
                tprint_debug(f"GPU correlation calculation failed, falling back to CPU: {e}")
        
        # Fallback to CPU calculation
        correlation = np.corrcoef(feature_clean, labels_clean)[0, 1]
        return correlation if not np.isnan(correlation) else -np.inf
    
    def _update_performance_stats(self):
        """Update performance statistics with hardware monitoring."""
        if self.hardware_manager:
            try:
                # Get current hardware metrics
                metrics = self.hardware_manager.get_current_metrics()
                
                # Update performance stats
                self.performance_stats['cpu_utilization'] = metrics.cpu_usage
                self.performance_stats['memory_peak_mb'] = metrics.memory_usage_mb
                self.performance_stats['gpu_utilization'] = metrics.gpu_usage
                self.performance_stats['thermal_state'] = metrics.thermal_state
                
            except Exception as e:
                tprint_debug(f"Failed to update performance stats: {e}")
    
    def _optimize_data_for_hardware(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for hardware processing."""
        if self.memory_optimizer:
            try:
                # Use M1 memory optimizer
                optimized_data = self.memory_optimizer.optimize_dataframe(data)
                tprint_debug("✅ Data optimized for M1 memory architecture")
                return optimized_data
            except Exception as e:
                tprint_debug(f"Memory optimization failed, using original data: {e}")
        
        return data
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return self.performance_stats.copy()
    
    def cleanup(self):
        """Cleanup resources and stop hardware monitoring."""
        tprint("🧹 Cleaning up hardware-optimized parallel optimizer...")
        
        # Stop hardware monitoring
        if self.memory_optimizer:
            self.memory_optimizer.stop_monitoring()
        
        # Cleanup hardware managers
        if self.hardware_manager:
            try:
                self.hardware_manager.cleanup()
            except Exception as e:
                tprint_warning(f"Hardware manager cleanup failed: {e}")
        
        # Cleanup joblib memory
        if self.memory:
            self.memory.clear()
        
        # Force garbage collection
        gc.collect()
        
        tprint_success("✅ Hardware-optimized parallel optimizer cleanup completed")


# Convenience function for easy usage
def optimize_features_parallel(features: List[str],
                             lookback_range: range,
                             data: pd.DataFrame,
                             labels: pd.Series,
                             config: Optional[ParallelConfig] = None,
                             **kwargs) -> Dict[str, Any]:
    """
    Convenience function for parallel feature optimization.
    
    Args:
        features: List of feature names to optimize
        lookback_range: Range of lookback periods to test
        data: Feature data
        labels: Target labels
        config: Parallel processing configuration
        **kwargs: Additional optimization parameters
        
    Returns:
        Dictionary of optimization results
    """
    optimizer = ParallelFeatureOptimizer(config)
    try:
        return optimizer.optimize_features_parallel(features, lookback_range, data, labels, **kwargs)
    finally:
        optimizer.cleanup()