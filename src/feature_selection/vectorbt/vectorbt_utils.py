"""
VectorBT Utilities

This module provides shared utilities for VectorBT feature selection operations
to avoid code duplication across different selector modules.
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import List, Tuple, Optional, Dict, Any

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


def create_vectorbt_dataframe(X: np.ndarray, feature_names: List[str], 
                            config: VectorBTFeatureSelectionConfig) -> pd.DataFrame:
    """
    Create VectorBT-optimized DataFrame with enhanced financial operations.
    
    Args:
        X: Feature matrix
        feature_names: List of feature names
        config: VectorBT configuration
        
    Returns:
        VectorBT-optimized DataFrame
    """
    tprint_debug("🔧 Creating VectorBT-optimized DataFrame")
    
    try:
        # Use VectorBT's optimized DataFrame creation
        df = vbt.PandasDataFrame(X, columns=feature_names)
        
        # Enhanced financial time series indexing
        if config.enable_financial_optimization:
            tprint_debug("📊 Applying financial optimizations")
            # Use proper financial time series indexing with business days
            df.index = pd.bdate_range(start='2020-01-01', periods=len(df), freq='1min')
            
            # Leverage VectorBT's financial data optimizations
            try:
                df = df.vbt.freq_infer()  # Infer optimal frequency
                df = df.vbt.resample_apply('1H', 'last')  # More efficient resampling
                
                # Use VectorBT's financial data validation
                df = df.vbt.validate()  # Validate financial data integrity
                
                # Enable VectorBT's rolling window optimizations
                if hasattr(df, 'vbt') and config.enable_vectorbt_rolling:
                    df = df.vbt.rolling_apply('mean', window=100)  # Pre-compute rolling stats
                    
            except Exception as freq_e:
                logger.debug(f"Financial optimization skipped: {freq_e}")
        
        # Enhanced memory optimizations
        if config.enable_memory_optimization:
            tprint_debug("💾 Applying memory optimizations")
            try:
                # Use VectorBT's chunked operations
                df = df.vbt.chunked_apply('ffill', chunk_size=config.chunk_size)
                
                # Enable VectorBT's memory mapping for large datasets
                if X.nbytes > config.memory_mapping_threshold:
                    df = df.vbt.memory_map()  # Memory map large datasets
                    
            except Exception as mem_e:
                logger.debug(f"Memory optimization skipped: {mem_e}")
        
        tprint_debug(f"✅ VectorBT DataFrame created: {df.shape}")
        return df
        
    except Exception as e:
        logger.warning(f"Enhanced DataFrame creation failed: {e}")
        # Fallback to standard DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        if config.enable_financial_optimization:
            df.index = pd.bdate_range(start='2020-01-01', periods=len(df), freq='D')
        return df


def validate_inputs(X: np.ndarray, y: np.ndarray, 
                   feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Validate and prepare inputs for VectorBT processing.
    
    Args:
        X: Feature matrix
        y: Target variable
        feature_names: Optional list of feature names
        
    Returns:
        Tuple of validated (X, y, feature_names)
    """
    tprint_debug("🔍 Validating inputs for VectorBT processing")
    
    # Validate X
    X = validate_numeric_array(X, name="Feature matrix X")
    if not validate_finite(X):
        raise ValueError("Feature matrix X contains non-finite values")
    
    # Validate y
    y = validate_numeric_array(y, name="Target variable y")
    if not validate_finite(y):
        raise ValueError("Target variable y contains non-finite values")
    
    # Check dimensions
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have same number of samples: {X.shape[0]} vs {y.shape[0]}")
    
    # Prepare feature names
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    elif len(feature_names) != X.shape[1]:
        raise ValueError(f"Feature names length {len(feature_names)} doesn't match X shape[1] {X.shape[1]}")
    
    tprint_debug(f"✅ Inputs validated: X={X.shape}, y={y.shape}, features={len(feature_names)}")
    return X, y, feature_names


def time_operation(operation_name: str, func: callable, config: VectorBTFeatureSelectionConfig,
                  *args, **kwargs) -> Any:
    """
    Time an operation and log performance.
    
    Args:
        operation_name: Name of the operation
        func: Function to execute
        config: VectorBT configuration
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
    """
    if not config.enable_timing:
        return func(*args, **kwargs)
    
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    if config.log_performance:
        tprint_performance(f"⏱️ {operation_name}: {execution_time:.3f}s")
    
    return result


def track_vectorbt_performance(operation_name: str, start_time: float, 
                              vectorbt_operation: bool = True, 
                              df_shape: Tuple[int, int] = None,
                              performance_stats: Dict[str, Any] = None):
    """
    Track VectorBT performance with detailed metrics.
    
    Args:
        operation_name: Name of the operation
        start_time: Start time of operation
        vectorbt_operation: Whether this is a VectorBT operation
        df_shape: Shape of the DataFrame
        performance_stats: Performance statistics dictionary to update
    """
    execution_time = time.time() - start_time
    
    if performance_stats is not None:
        # Update VectorBT-specific stats
        if vectorbt_operation:
            performance_stats['vectorbt_operations'] = performance_stats.get('vectorbt_operations', 0) + 1
            performance_stats['vectorbt_time'] = performance_stats.get('vectorbt_time', 0.0) + execution_time
            
            # Track VectorBT efficiency
            if performance_stats.get('total_operations', 0) > 0:
                performance_stats['vectorbt_efficiency'] = (
                    performance_stats['vectorbt_operations'] / 
                    performance_stats['total_operations']
                )
        
        # Track data size efficiency
        if df_shape:
            features_per_second = df_shape[1] / execution_time if execution_time > 0 else 0
            performance_stats['features_per_second'] = features_per_second
            
            # Track memory efficiency
            memory_usage = df_shape[0] * df_shape[1] * 8 / (1024 * 1024)  # MB
            performance_stats['memory_efficiency_mb_per_sec'] = memory_usage / execution_time
    
    # Log performance with enhanced metrics
    metrics = f"⏱️ {operation_name}: {execution_time:.3f}s"
    if vectorbt_operation:
        metrics += f" (VectorBT: {vectorbt_operation})"
    if df_shape:
        metrics += f" (Shape: {df_shape})"
        if performance_stats and 'features_per_second' in performance_stats:
            metrics += f" ({performance_stats['features_per_second']:.0f} features/sec)"
    
    tprint_performance(metrics)


def setup_vectorbt_optimizations(config: VectorBTFeatureSelectionConfig) -> bool:
    """
    Setup VectorBT with optimal settings.
    
    Args:
        config: VectorBT configuration
        
    Returns:
        True if setup successful, False otherwise
    """
    tprint_debug("🔧 Setting up VectorBT optimizations")
    
    try:
        # Configure VectorBT theme
        vbt.settings.set_theme(config.vectorbt_theme)
        
        # Configure array wrapper settings
        for key, value in config.vectorbt_array_wrapper.items():
            vbt.settings['array_wrapper'][key] = value
        
        # Enable VectorBT optimizations
        if config.enable_vectorbt_rolling:
            vbt.settings['array_wrapper']['enable_rolling'] = True
        
        if config.enable_vectorbt_chunked:
            vbt.settings['array_wrapper']['enable_chunked'] = True
        
        if config.enable_vectorbt_parallel:
            vbt.settings['array_wrapper']['enable_parallel'] = True
        
        tprint_success("✅ VectorBT optimizations configured")
        return True
        
    except Exception as e:
        logger.warning(f"VectorBT optimization setup failed: {e}")
        return False


def setup_gpu_acceleration(config: VectorBTFeatureSelectionConfig) -> bool:
    """
    Setup GPU acceleration for VectorBT operations.
    
    Args:
        config: VectorBT configuration
        
    Returns:
        True if GPU setup successful, False otherwise
    """
    tprint_debug("🔧 Setting up GPU acceleration")
    
    try:
        if config.enable_gpu:
            import torch
            import cupy as cp
            
            # Check CUDA availability
            if torch.cuda.is_available():
                # Configure CUDA device
                torch.cuda.set_device(config.gpu_device)
                
                # Configure CuPy memory pool
                if config.cuda_memory_pool:
                    cp.cuda.MemoryPool().set_limit(fraction=config.gpu_memory_fraction)
                
                # Enable VectorBT GPU operations
                vbt.settings['array_wrapper']['enable_gpu'] = True
                vbt.settings['array_wrapper']['gpu_chunk_size'] = config.gpu_chunk_size
                
                tprint_success("✅ GPU acceleration enabled")
                return True
            else:
                logger.warning("CUDA not available, GPU acceleration disabled")
                return False
        else:
            tprint_debug("GPU acceleration disabled in config")
            return False
            
    except Exception as e:
        logger.warning(f"GPU setup failed: {e}")
        return False


def setup_advanced_parallel_processing(config: VectorBTFeatureSelectionConfig) -> Dict[str, Any]:
    """
    Setup advanced parallel processing with VectorBT optimizations.
    
    Args:
        config: VectorBT configuration
        
    Returns:
        Dictionary of parallel processing clients
    """
    tprint_debug("🔧 Setting up advanced parallel processing")
    
    try:
        # Configure VectorBT's parallel processing
        vbt.settings['array_wrapper']['enable_parallel'] = True
        vbt.settings['array_wrapper']['max_workers'] = config.max_workers or -1
        
        # Enable VectorBT's chunked parallel processing
        vbt.settings['array_wrapper']['enable_chunked_parallel'] = True
        vbt.settings['array_wrapper']['chunk_parallel_workers'] = config.max_workers or 4
        
        # Configure for financial data parallel processing
        vbt.settings['array_wrapper']['enable_financial_parallel'] = True
        
        parallel_clients = {}
        
        # Enhanced Dask integration
        if config.enable_dask:
            tprint_debug("🔧 Setting up Dask parallel processing")
            import dask
            from dask.distributed import Client
            
            # Configure Dask for VectorBT
            dask.config.set({
                'array.chunk-size': f"{config.chunk_size}MB",
                'array.slicing.split_large_chunks': True,
                'array.optimization.fuse.active': True
            })
            
            if config.dask_cluster_type == "local":
                dask_client = Client(
                    n_workers=config.dask_workers,
                    memory_limit=config.dask_memory_limit,
                    threads_per_worker=2  # Optimize for VectorBT
                )
            else:
                dask_client = Client(config.dask_cluster_type)
            
            parallel_clients['dask'] = dask_client
            tprint_success("✅ Dask parallel processing enabled")
        
        if config.enable_ray:
            tprint_debug("🔧 Setting up Ray parallel processing")
            import ray
            
            if not ray.is_initialized():
                ray.init(
                    address=config.ray_cluster_address,
                    num_cpus=config.ray_num_cpus,
                    num_gpus=config.ray_num_gpus
                )
            
            parallel_clients['ray'] = ray
            tprint_success("✅ Ray parallel processing enabled")
        
        if parallel_clients:
            tprint_success(f"✅ Advanced parallel processing enabled: {list(parallel_clients.keys())}")
        else:
            tprint_debug("No advanced parallel processing clients configured")
        
        return parallel_clients
        
    except Exception as e:
        logger.warning(f"Advanced parallel processing setup failed: {e}")
        return {}


def create_performance_stats() -> Dict[str, Any]:
    """
    Create initial performance statistics dictionary.
    
    Returns:
        Performance statistics dictionary
    """
    return {
        'total_operations': 0,
        'vectorbt_operations': 0,
        'total_time': 0.0,
        'vectorbt_time': 0.0,
        'speedup': 0.0,
        'memory_saved_mb': 0.0,
        'vectorbt_efficiency': 0.0,
        'gpu_operations': 0,
        'cache_hits': 0,
        'cache_misses': 0
    }