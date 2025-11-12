"""
Feature Engineering Common Utilities

This module provides shared utilities for feature engineering operations including
caching, monitoring, validation, and optimization utilities.
"""

# Import caching utilities - these are the core features
from .caching import (
    SharedComputationCache, 
    FeatureCache,
    HardwareAwareCache,
    OptimizedFeatureCache,
    OptimizedCacheConfig,
    CacheConfig,
    CacheStrategy,
    CompressionType,
    get_default_shared_cache,
    get_default_feature_cache,
    cache_context,
    optimize_cache_for_operation
)

# Import monitoring and validation utilities
try:
    from .monitoring import FeaturePerformanceMonitor, ResourceTracker
except ImportError:
    FeaturePerformanceMonitor = None
    ResourceTracker = None

try:
    from .validation import FeatureDataValidator, DataLeakageDetector
except ImportError:
    FeatureDataValidator = None
    DataLeakageDetector = None

# Import volume transformations
try:
    from .volume_transforms import (
        stabilize_volume,
        log_volume,
        rolling_median_log_volume,
        calculate_mad,
        robust_z_score,
        calculate_true_range,
        calculate_atr,
        volume_normalized_by_tr,
        calculate_robust_volume_features,
    )
except ImportError:
    stabilize_volume = None
    log_volume = None
    rolling_median_log_volume = None
    calculate_mad = None
    robust_z_score = None
    calculate_true_range = None
    calculate_atr = None
    volume_normalized_by_tr = None
    calculate_robust_volume_features = None

__all__ = [
    # Caching utilities (optimized, hardware-aware)
    'SharedComputationCache',
    'FeatureCache',
    'HardwareAwareCache',
    'OptimizedFeatureCache',
    'OptimizedCacheConfig',
    'CacheConfig',
    'CacheStrategy',
    'CompressionType',
    'get_default_shared_cache',
    'get_default_feature_cache',
    'cache_context',
    'optimize_cache_for_operation',
]

# Add monitoring and validation if available
if FeaturePerformanceMonitor is not None:
    __all__.extend([
        'FeaturePerformanceMonitor',
        'ResourceTracker'
    ])

if FeatureDataValidator is not None:
    __all__.extend([
        'FeatureDataValidator',
        'DataLeakageDetector'
    ])

# Add volume transformations if available
if stabilize_volume is not None:
    __all__.extend([
        'stabilize_volume',
        'log_volume',
        'rolling_median_log_volume',
        'calculate_mad',
        'robust_z_score',
        'calculate_true_range',
        'calculate_atr',
        'volume_normalized_by_tr',
        'calculate_robust_volume_features',
    ])
