#!/usr/bin/env python3
"""
Modular Final Feature Selection Pipeline

This module provides a streamlined interface to the modular feature selection
system, replacing the original monolithic implementation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import time
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
)

# Import modular components
from .feature_selection import (
    MultiStageFeatureSelector,
    FeatureSelector,
    FeatureSelectionOptimizer,
    FeatureSelectionConfig,
    FeatureSelectionResult,
    MemoryManager,
    VectorBTManager,
    PerformanceMonitor,
    ConfigLoader,
    ModelProfileManager,
    ConfigValidator,
    DataValidator
)


def run_final_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    timeframe: str = "15m",
    config: Optional[Dict[str, Any]] = None
) -> FeatureSelectionResult:
    """
    Run final feature selection using the modular system.
    
    This function provides a clean interface to the modular feature selection
    system, replacing the original monolithic implementation.
    
    Args:
        X: Feature matrix (samples x features)
        y: Target variable (samples,)
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        config: Optional configuration dictionary
        
    Returns:
        FeatureSelectionResult with selected features and metrics
    """
    tprint("🚀 Starting Modular Final Feature Selection")
    start_time = time.time()
    
    try:
        # Initialize configuration
        tprint_info("🔧 Initializing configuration")
        config_loader = ConfigLoader()
        config_result = config_loader.load_config('feature_selection', 'default')
        
        if not config_result.success:
            raise ValueError(f"Failed to load configuration: {config_result.error_message}")
        
        # Create feature selection configuration
        fs_config = FeatureSelectionConfig(**config_result.config)
        
        # Apply custom config if provided
        if config:
            for key, value in config.items():
                if hasattr(fs_config, key):
                    setattr(fs_config, key, value)
        
        # Initialize data validator
        tprint_info("🔍 Validating input data")
        data_validator = DataValidator()
        validation_result = data_validator.validate_feature_selection_input(X, y)
        
        if not validation_result.is_valid:
            tprint_error(f"❌ Data validation failed: {validation_result.errors}")
            return FeatureSelectionResult(
                selected_features=[],
                feature_importance={},
                feature_scores={},
                performance_metrics={},
                validation_scores={},
                config_used=fs_config,
                execution_time=time.time() - start_time,
                memory_usage={},
                success=False,
                error_message=f"Data validation failed: {validation_result.errors}"
            )
        
        # Use validated data if available
        if validation_result.validated_data is not None:
            X = validation_result.validated_data
        
        # Initialize performance monitoring
        tprint_info("📊 Initializing performance monitoring")
        performance_monitor = PerformanceMonitor(enable_monitoring=True)
        performance_monitor.start_monitoring()
        
        # Initialize memory manager
        tprint_info("🧠 Initializing memory management")
        memory_manager = MemoryManager(
            memory_limit_gb=fs_config.vectorbt_chunk_size / 1000,  # Convert to GB
            strategy='aggressive'
        )
        
        # Check memory pressure
        memory_stats = memory_manager.get_memory_stats()
        if memory_stats.cleanup_recommended:
            tprint_warning("🧹 Memory pressure detected, performing cleanup")
            memory_manager.perform_cleanup()
        
        # Initialize VectorBT manager
        tprint_info("⚡ Initializing VectorBT optimization")
        vectorbt_config = {
            'enable_gpu': fs_config.vectorbt_enable_gpu,
            'enable_parallel': fs_config.vectorbt_enable_parallel,
            'memory_efficient': fs_config.vectorbt_memory_efficient,
            'chunk_size': fs_config.vectorbt_chunk_size
        }
        vectorbt_manager = VectorBTManager(vectorbt_config)
        
        # Optimize data with VectorBT if available
        if vectorbt_manager.is_available():
            tprint_info("⚡ Applying VectorBT optimizations")
            X = vectorbt_manager.optimize_dataframe(X)
        
        # Initialize feature selector
        tprint_info("🎯 Initializing feature selector")
        selector = MultiStageFeatureSelector(fs_config)
        
        # Run feature selection with performance monitoring
        with performance_monitor.monitor_operation("feature_selection", X.shape):
            result = selector.select_features(X, y, symbol, exchange, timeframe)
        
        # Get performance statistics
        performance_stats = performance_monitor.get_performance_summary()
        vectorbt_stats = vectorbt_manager.get_performance_stats()
        memory_stats_final = memory_manager.get_memory_stats()
        
        # Update result with performance metrics
        result.performance_metrics.update({
            'performance_monitor': performance_stats,
            'vectorbt_stats': vectorbt_stats,
            'memory_stats': memory_stats_final,
            'data_validation': validation_result.quality_metrics
        })
        
        # Cleanup resources
        tprint_info("🧹 Cleaning up resources")
        performance_monitor.cleanup()
        memory_manager.cleanup_resources()
        vectorbt_manager.cleanup()
        
        execution_time = time.time() - start_time
        result.execution_time = execution_time
        
        if result.success:
            tprint_success(f"✅ Feature selection completed successfully in {execution_time:.2f}s")
            tprint(f"   📊 Selected {len(result.selected_features)} features from {X.shape[1]}")
            tprint(f"   📊 Performance score: {performance_stats.get('average_execution_time', 0):.3f}s avg")
            tprint(f"   📊 Memory efficiency: {memory_stats_final.memory_efficiency:.2f}")
        else:
            tprint_error(f"❌ Feature selection failed: {result.error_message}")
        
        return result
        
    except Exception as e:
        tprint_error(f"❌ Feature selection failed with exception: {e}")
        return FeatureSelectionResult(
            selected_features=[],
            feature_importance={},
            feature_scores={},
            performance_metrics={},
            validation_scores={},
            config_used=fs_config if 'fs_config' in locals() else FeatureSelectionConfig(),
            execution_time=time.time() - start_time,
            memory_usage={},
            success=False,
            error_message=str(e)
        )


def get_final_features(
    X: pd.DataFrame,
    y: pd.Series,
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    timeframe: str = "15m",
    config: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Get final selected features using the modular system.
    
    This is a convenience function that returns only the selected feature names.
    
    Args:
        X: Feature matrix
        y: Target variable
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        config: Optional configuration
        
    Returns:
        List of selected feature names
    """
    result = run_final_feature_selection(X, y, symbol, exchange, timeframe, config)
    return result.selected_features if result.success else []


# Backward compatibility aliases
MultiStageFeatureSelector = MultiStageFeatureSelector
FeatureSelectionConfig = FeatureSelectionConfig
FeatureSelectionResult = FeatureSelectionResult