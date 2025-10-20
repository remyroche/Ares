"""
VectorBT Feature Selection Core Framework

This module provides the core feature selection framework using VectorBT
for high-performance feature selection operations with hardware optimization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
import logging

# Import VectorBT framework
from ..vectorbt_extensions.vectorbt_unified_framework import VectorBTUnifiedFramework, create_vectorbt_unified_framework
from ..vectorbt_extensions.vectorbt_config import VectorBTFeatureSelectionConfig

# Import hardware optimization tools
from src.utils.hardware import (
    get_integrated_hardware_manager,
    memory_efficient,
    performance_tracked,
    smart_cache,
    auto_optimize,
    WorkloadType,
    OptimizationLevel
)

# Import tprint for consistent logging
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_performance, tprint_debug

logger = logging.getLogger(__name__)

# Global VectorBT framework instance
_GLOBAL_VECTORBT_FRAMEWORK: Optional[VectorBTUnifiedFramework] = None

def get_feature_selection_framework(config: Optional[Dict[str, Any]] = None) -> VectorBTUnifiedFramework:
    """
    Get a global instance of the VectorBT feature selection framework.

    Args:
        config: Optional configuration dictionary

    Returns:
        VectorBTUnifiedFramework instance
    """
    tprint("🚀 Getting VectorBT feature selection framework")
    global _GLOBAL_VECTORBT_FRAMEWORK

    if _GLOBAL_VECTORBT_FRAMEWORK is None:
        tprint("🔧 Initializing new VectorBT feature selection framework")

        # Convert dict config to VectorBT config if provided
        vectorbt_config = None
        if config:
            vectorbt_config = VectorBTFeatureSelectionConfig.from_dict(config)

        _GLOBAL_VECTORBT_FRAMEWORK = create_vectorbt_unified_framework(vectorbt_config)
        tprint_success("✅ VectorBT framework initialized successfully")

    return _GLOBAL_VECTORBT_FRAMEWORK

def _ensure_feature_names(X: Union[np.ndarray, pd.DataFrame], feature_names: Optional[List[str]]) -> Tuple[np.ndarray, List[str]]:
    """Ensure feature names are available."""
    tprint_debug("🔍 Ensuring feature names are available")

    if hasattr(X, "values"):
        X_np = X.values
        names = feature_names or list(getattr(X, "columns"))
    else:
        X_np = np.asarray(X)
        names = feature_names or [f"feature_{i}" for i in range(X_np.shape[1])]

    tprint_debug(f"📊 Feature matrix shape: {X_np.shape}, {len(names)} feature names")
    return X_np, names

@memory_efficient(memory_threshold_mb=500.0, auto_cleanup=True)
@performance_tracked(log_performance=True, track_memory=True)
@smart_cache(ttl=3600)  # Cache for 1 hour
def select_features(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series, List[float], List[int]],
    method: str = "auto",
    max_features: Optional[int] = None,
    is_classification: Optional[bool] = None,
    feature_names: Optional[List[str]] = None,
    framework_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Unified VectorBT feature selection API.

    Args:
        X: Feature matrix (np.ndarray or pandas DataFrame)
        y: Target vector
        method: Selection method ('auto', 'comprehensive', 'correlation', 'mrmr', etc.)
        max_features: Maximum features to select
        is_classification: Whether this is classification (inferred if not provided)
        feature_names: Optional list of feature names
        framework_config: Optional configuration for the VectorBT framework
        **kwargs: Additional method-specific parameters

    Returns:
        Dictionary with selection results
    """
    tprint(f"🚀 Starting VectorBT feature selection: method={method}, max_features={max_features}")

    try:
        # Get integrated hardware manager for optimization
        hardware_manager = get_integrated_hardware_manager()
        
        # Determine workload type based on data size and method
        data_size_mb = X.nbytes / (1024 * 1024) if hasattr(X, 'nbytes') else 0
        if data_size_mb > 1000 or method in ['comprehensive', 'stability_selection']:
            workload_type = WorkloadType.ML_TRAINING
            optimization_level = OptimizationLevel.AGGRESSIVE
        elif method in ['correlation', 'mutual_information']:
            workload_type = WorkloadType.DATA_PROCESSING
            optimization_level = OptimizationLevel.BALANCED
        else:
            workload_type = WorkloadType.GENERAL
            optimization_level = OptimizationLevel.MINIMAL
        
        # Optimize hardware for workload
        hardware_manager.optimize_for_workload(workload_type, optimization_level)
        
        # Pre-optimize data with hardware manager
        X_optimized = hardware_manager.process_data_with_optimization(X, workload_type)
        y_optimized = hardware_manager.process_data_with_optimization(y, workload_type)

        # Get VectorBT framework
        framework = get_feature_selection_framework(framework_config)

        # Normalize inputs
        X_np, names = _ensure_feature_names(X_optimized, feature_names)
        y_arr = np.asarray(y_optimized)

        # Map legacy method names to VectorBT methods
        method_mapping = {
            "auto": "auto",
            "comprehensive": "comprehensive",
            "filter": "correlation",
            "correlation": "correlation",
            "mutual_info": "mutual_information",
            "mrmr": "mrmr",
            "stability": "stability_selection",
            "lasso": "lasso",
            "elasticnet": "elasticnet",
            "rfe": "rfe",
            "adaptive": "adaptive"
        }

        vectorbt_method = method_mapping.get(method, method)
        tprint_debug(f"📊 Mapped method '{method}' to VectorBT method '{vectorbt_method}'")

        # Perform feature selection using VectorBT
        result = framework.select_features(
            X=X_np,
            y=y_arr,
            method=vectorbt_method,
            k=max_features,
            feature_names=names,
            **kwargs
        )

        # Convert VectorBT result to expected format
        if result.success:
            tprint_success(f"✅ VectorBT selection completed: {result.n_selected}/{result.n_total} features selected")

            return {
                'success': True,
                'selected_features': result.selected_features,
                'selected_indices': result.selected_indices,
                'feature_scores': result.feature_scores,
                'n_selected': result.n_selected,
                'n_total': result.n_total,
                'method': result.method,
                'execution_time': result.execution_time,
                'performance_stats': result.performance_stats,
                'metadata': result.metadata
            }
        else:
            tprint_warning(f"⚠️ VectorBT selection failed: {result.error}")
            return {
                'success': False,
                'error': result.error,
                'selected_features': [],
                'selected_indices': [],
                'feature_scores': {},
                'n_selected': 0,
                'n_total': X_np.shape[1],
                'method': result.method,
                'execution_time': result.execution_time
            }

    except Exception as e:
        tprint_warning(f"⚠️ Feature selection failed: {e}")
        logger.error(f"Feature selection error: {e}")

        return {
            'success': False,
            'error': str(e),
            'selected_features': [],
            'selected_indices': [],
            'feature_scores': {},
            'n_selected': 0,
            'n_total': X_np.shape[1] if 'X_np' in locals() else 0,
            'method': method,
            'execution_time': 0.0
        }

@memory_efficient(memory_threshold_mb=1000.0, auto_cleanup=True)
@performance_tracked(log_performance=True, track_memory=True)
def benchmark_methods(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series, List[float], List[int]],
    max_features: Optional[int] = None,
    feature_names: Optional[List[str]] = None,
    framework_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Benchmark all VectorBT feature selection methods.

    Args:
        X: Feature matrix
        y: Target vector
        max_features: Maximum features to select
        feature_names: Optional list of feature names
        framework_config: Optional configuration

    Returns:
        Dictionary with benchmark results
    """
    tprint("🚀 Starting VectorBT method benchmarking")

    try:
        # Get VectorBT framework
        framework = get_feature_selection_framework(framework_config)

        # Normalize inputs
        X_np, names = _ensure_feature_names(X, feature_names)
        y_arr = np.asarray(y)

        # Run benchmark
        result = framework.benchmark_methods(
            X=X_np,
            y=y_arr,
            k=max_features,
            feature_names=names
        )

        if result['success']:
            tprint_success(f"✅ Benchmarking completed: {result['n_successful']}/{result['n_methods_tested']} methods successful")
        else:
            tprint_warning(f"⚠️ Benchmarking failed: {result.get('error', 'Unknown error')}")

        return result

    except Exception as e:
        tprint_warning(f"⚠️ Benchmarking failed: {e}")
        logger.error(f"Benchmarking error: {e}")

        return {
            'success': False,
            'error': str(e),
            'benchmark_results': {}
        }

def get_performance_stats() -> Dict[str, Any]:
    """Get performance statistics from the VectorBT framework."""
    tprint("📊 Getting VectorBT performance statistics")

    try:
        framework = get_feature_selection_framework()
        stats = framework.get_performance_stats()

        tprint_performance(f"📊 VectorBT Performance: {stats['total_selections']} total selections, "
                         f"{stats['success_rate']:.2%} success rate")

        return stats

    except Exception as e:
        tprint_warning(f"⚠️ Failed to get performance stats: {e}")
        logger.error(f"Performance stats error: {e}")

        return {
            'total_selections': 0,
            'successful_selections': 0,
            'failed_selections': 0,
            'success_rate': 0.0,
            'avg_execution_time': 0.0,
            'error': str(e)
        }

def reset_framework():
    """Reset the global framework instance."""
    tprint("🔄 Resetting VectorBT framework")
    global _GLOBAL_VECTORBT_FRAMEWORK
    _GLOBAL_VECTORBT_FRAMEWORK = None
    tprint_success("✅ Framework reset complete")

# Legacy compatibility functions
def get_enhanced_framework(config: Optional[Dict[str, Any]] = None) -> VectorBTUnifiedFramework:
    """Legacy compatibility function."""
    tprint("⚠️ Using legacy get_enhanced_framework - consider using get_feature_selection_framework")
    return get_feature_selection_framework(config)

def enhanced_select_features(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series, List[float], List[int]],
    method: str = "comprehensive",
    max_features: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """Legacy compatibility function."""
    tprint("⚠️ Using legacy enhanced_select_features - consider using select_features")
    return select_features(X, y, method, max_features, **kwargs)

def run_comprehensive_feature_selection(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series, List[float], List[int]],
    feature_names: Optional[List[str]] = None,
    target_count: Optional[int] = None,
    model_type: str = 'default',
    enable_all_optimizations: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Run comprehensive feature selection with all optimizations enabled.

    This is a convenience function that wraps select_features with comprehensive
    settings for maximum effectiveness across all feature selection methods.

    Args:
        X: Feature matrix (np.ndarray or pandas DataFrame)
        y: Target vector
        feature_names: Optional list of feature names
        target_count: Target number of features to select (if None, auto-determined)
        model_type: Type of model for feature selection ('classification', 'regression', 'default')
        enable_all_optimizations: Whether to enable all optimizations
        **kwargs: Additional parameters passed to select_features

    Returns:
        Dictionary with comprehensive selection results
    """
    tprint("🚀 Starting comprehensive feature selection...")

    # Determine if this is classification or regression
    y_array = np.asarray(y)
    is_classification = model_type == 'classification' or (
        model_type == 'default' and (
            len(np.unique(y_array)) < 20 or
            (np.issubdtype(y_array.dtype, np.integer) and np.max(y_array) < 100)
        )
    )

    tprint(f"📊 Input: {X.shape[0]} samples, {X.shape[1]} features, "
           f"classification: {is_classification}, model_type: {model_type}")

    # Set comprehensive method as default
    method = kwargs.get('method', 'comprehensive')

    # Configure comprehensive settings
    comprehensive_config = {
        'method': method,
        'is_classification': is_classification,
        'enable_performance_monitoring': enable_all_optimizations,
        'enable_early_stopping': enable_all_optimizations,
        'enable_parallel_processing': enable_all_optimizations,
        'max_features': target_count,
        **kwargs
    }

    # Run selection
    result = select_features(
        X=X,
        y=y,
        feature_names=feature_names,
        **comprehensive_config
    )

    if result['success']:
        tprint_success(f"✅ Comprehensive selection completed: {result['n_selected']} features selected")
    else:
        tprint_warning(f"⚠️ Comprehensive selection failed: {result.get('error', 'Unknown error')}")

    return result
