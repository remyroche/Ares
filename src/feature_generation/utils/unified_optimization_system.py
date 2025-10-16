"""
Unified Optimization System

This module provides a unified, non-redundant optimization system that consolidates
all normalization, scaling, and vectorization optimizations into a single, efficient interface.
"""

import logging
import time
import warnings
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
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
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Import existing utilities (avoid duplication)
from ...utils.intensity_scaler import get_intensity_config, apply_intensity_scaling
from ...training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.data_normalization import (
    create_data_normalizer, NormalizationConfig, NormalizationMethod
)
from .vectorization_optimizer import get_vectorization_optimizer, VectorizationConfig
from ...utils.hardware.unified_hardware_manager import get_unified_hardware_manager, WorkloadType, OptimizationLevel

logger = logging.getLogger(__name__)

@dataclass
class UnifiedOptimizationConfig:
    """Unified configuration for all optimization components."""
    # Normalization Configuration
    enable_normalization: bool = True
    normalization_method: str = "zscore"  # "zscore", "minmax", "robust", "quantile"
    normalization_exclude_categories: List[str] = field(default_factory=list)
    normalization_exclude_features: List[str] = field(default_factory=list)

    # Scaling Configuration
    enable_scaling: bool = True
    intensity_percentage: Optional[float] = None  # Auto-detect from environment

    # Vectorization Configuration
    enable_vectorization: bool = True
    chunk_size: int = 10000
    adaptive_chunking: bool = True
    memory_limit_gb: float = 8.0

    # Hardware Configuration
    enable_hardware_optimization: bool = True
    workload_type: WorkloadType = WorkloadType.FEATURE_ENGINEERING
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED

    # Performance Configuration
    enable_caching: bool = True
    enable_memory_pooling: bool = True
    enable_lazy_loading: bool = True

@dataclass
class OptimizationResult:
    """Result from unified optimization processing."""
    data: pd.DataFrame
    normalization_params: Dict[str, Any] = field(default_factory=dict)
    scaling_params: Dict[str, Any] = field(default_factory=dict)
    vectorization_stats: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    memory_usage: float = 0.0
    success: bool = True
    error_message: Optional[str] = None

class UnifiedOptimizationSystem:
    """
    Unified optimization system that consolidates all optimization utilities
    without redundancy.
    """

    def __init__(self, config: Optional[UnifiedOptimizationConfig] = None):
        """Initialize the unified optimization system."""
        self.config = config or UnifiedOptimizationConfig()
        self.logger = logger.getChild('UnifiedOptimizationSystem')

        # Initialize components (lazy loading if enabled)
        self._normalizer = None
        self._scaler = None
        self._vectorization_optimizer = None
        self._hardware_manager = None

        # Performance tracking
        self.performance_stats = {
            'total_operations': 0,
            'normalization_operations': 0,
            'scaling_operations': 0,
            'vectorization_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_processing_time': 0.0,
            'memory_savings': 0.0
        }

        # Cache for parameters
        self._parameter_cache = {} if self.config.enable_caching else None

        self.logger.info("✅ Unified Optimization System initialized")

    @property
    def normalizer(self):
        """Lazy-loaded normalizer."""
        if self._normalizer is None and self.config.enable_normalization:
            try:
                normalization_config = NormalizationConfig(
                    method=NormalizationMethod(self.config.normalization_method.upper()),
                    use_hardware_acceleration=self.config.enable_hardware_optimization,
                    use_matrix_operations=True,
                    memory_limit_gb=self.config.memory_limit_gb
                )
                self._normalizer = create_data_normalizer(normalization_config)
                self.logger.debug("✅ Normalizer initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Normalizer not available: {e}")
        return self._normalizer

    @property
    def scaler(self):
        """Lazy-loaded scaler."""
        if self._scaler is None and self.config.enable_scaling:
            try:
                self._scaler = get_intensity_config(self.config.intensity_percentage)
                self.logger.debug("✅ Scaler initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Scaler not available: {e}")
        return self._scaler

    @property
    def vectorization_optimizer(self):
        """Lazy-loaded vectorization optimizer."""
        if self._vectorization_optimizer is None and self.config.enable_vectorization:
            try:
                vectorization_config = VectorizationConfig(
                    chunk_size=self.config.chunk_size,
                    adaptive_chunking=self.config.adaptive_chunking,
                    memory_limit_gb=self.config.memory_limit_gb,
                    enable_memory_pooling=self.config.enable_memory_pooling
                )
                self._vectorization_optimizer = get_vectorization_optimizer(vectorization_config)
                self.logger.debug("✅ Vectorization optimizer initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Vectorization optimizer not available: {e}")
        return self._vectorization_optimizer

    @property
    def hardware_manager(self):
        """Lazy-loaded hardware manager."""
        if self._hardware_manager is None and self.config.enable_hardware_optimization:
            try:
                self._hardware_manager = get_unified_hardware_manager()
                self._hardware_manager.optimize_for_workload(
                    self.config.workload_type,
                    self.config.optimization_level
                )
                self.logger.debug("✅ Hardware manager initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Hardware manager not available: {e}")
        return self._hardware_manager

    def optimize_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame for processing using vectorization optimizer.

        Args:
            data: Input DataFrame

        Returns:
            Optimized DataFrame
        """
        if self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data

    def process_features_unified(self,
                               data: pd.DataFrame,
                               categories: Optional[List[str]] = None,
                               features: Optional[List[str]] = None,
                               target_column: Optional[str] = None,
                               **kwargs) -> OptimizationResult:
        """
        Process features through the unified optimization system.

        Args:
            data: Input DataFrame
            categories: List of feature categories
            features: List of specific features
            target_column: Target column for optimization
            **kwargs: Additional parameters

        Returns:
            OptimizationResult with processed data and metadata
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()

        try:
            self.logger.info("🚀 Starting unified feature processing")

            # Step 1: Optimize DataFrame
            optimized_data = self.optimize_dataframe(data)

            # Step 2: Apply normalization if enabled
            normalization_params = {}
            if self.config.enable_normalization and self.normalizer:
                normalized_data, normalization_params = self._apply_normalization_unified(
                    optimized_data, categories
                )
                optimized_data = normalized_data
                self.performance_stats['normalization_operations'] += 1

            # Step 3: Apply scaling if enabled
            scaling_params = {}
            if self.config.enable_scaling and self.scaler:
                scaled_data, scaling_params = self._apply_scaling_unified(optimized_data)
                optimized_data = scaled_data
                self.performance_stats['scaling_operations'] += 1

            # Step 4: Final vectorization optimization
            if self.vectorization_optimizer:
                optimized_data = self.vectorization_optimizer.optimize_dataframe_processing(optimized_data)
                self.performance_stats['vectorization_operations'] += 1

            processing_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - start_memory

            # Update performance stats
            self.performance_stats['total_operations'] += 1
            self.performance_stats['total_processing_time'] += processing_time

            self.logger.info(f"✅ Unified processing completed in {processing_time:.3f}s")

            return OptimizationResult(
                data=optimized_data,
                normalization_params=normalization_params,
                scaling_params=scaling_params,
                vectorization_stats=self._get_vectorization_stats(),
                processing_time=processing_time,
                memory_usage=memory_usage,
                success=True
            )

        except Exception as e:
            processing_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - start_memory

            self.logger.error(f"❌ Unified processing failed: {e}")

            return OptimizationResult(
                data=data,
                processing_time=processing_time,
                memory_usage=memory_usage,
                success=False,
                error_message=str(e)
            )

    def _apply_normalization_unified(self,
                                   data: pd.DataFrame,
                                   categories: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply normalization using the unified normalizer."""
        try:
            # Select features for normalization
            target_columns = self._select_normalization_targets(data, categories)

            if not target_columns:
                return data, {}

            # Use the unified normalizer
            result = self.normalizer.normalize_data(data, target_columns=target_columns)

            if result.success:
                return result.normalized_data, result.normalization_params
            else:
                self.logger.warning(f"Normalization failed: {result.error_message}")
                return data, {}

        except Exception as e:
            self.logger.error(f"Normalization error: {e}")
            return data, {}

    def _apply_scaling_unified(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply scaling using the unified scaler."""
        try:
            # Apply intensity scaling to configuration
            scaling_params = {
                'intensity_percentage': self.scaler.intensity_percentage,
                'training_mode': self.scaler.training_mode,
                'scaled_parameters': {}
            }

            # Note: Intensity scaling typically affects configuration parameters,
            # not the data itself, so we return the data unchanged
            return data, scaling_params

        except Exception as e:
            self.logger.error(f"Scaling error: {e}")
            return data, {}

    def _select_normalization_targets(self,
                                    data: pd.DataFrame,
                                    categories: Optional[List[str]] = None) -> List[str]:
        """Select which features should be normalized."""
        target_columns = []

        # Get numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_columns:
            # Skip excluded features
            if col in self.config.normalization_exclude_features:
                continue

            # Skip features from excluded categories
            if categories and self._is_feature_in_excluded_category(col, categories):
                continue

            # Only normalize features that are not already normalized
            if not self._is_already_normalized(col):
                target_columns.append(col)

        return target_columns

    def _is_feature_in_excluded_category(self, feature_name: str, categories: List[str]) -> bool:
        """Check if a feature belongs to an excluded category."""
        excluded_indicators = ['zscore', 'normalized', 'scaled', 'rank']
        return any(indicator in feature_name.lower() for indicator in excluded_indicators)

    def _is_already_normalized(self, feature_name: str) -> bool:
        """Check if a feature is already normalized."""
        normalized_indicators = [
            'rsi', 'stoch', 'williams', 'macd_hist', 'bb_percent',
            'adx', 'cci', 'momentum', 'roc', 'zscore', 'normalized'
        ]
        return any(indicator in feature_name.lower() for indicator in normalized_indicators)

    def _get_vectorization_stats(self) -> Dict[str, Any]:
        """Get vectorization statistics."""
        if self.vectorization_optimizer:
            return self.vectorization_optimizer.get_performance_report()
        return {}

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            return psutil.virtual_memory().used / (1024 * 1024)
        except ImportError:
            return 0.0

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            'unified_stats': self.performance_stats.copy(),
            'component_status': {
                'normalizer_available': self.normalizer is not None,
                'scaler_available': self.scaler is not None,
                'vectorization_optimizer_available': self.vectorization_optimizer is not None,
                'hardware_manager_available': self.hardware_manager is not None
            },
            'config': {
                'enable_normalization': self.config.enable_normalization,
                'enable_scaling': self.config.enable_scaling,
                'enable_vectorization': self.config.enable_vectorization,
                'enable_hardware_optimization': self.config.enable_hardware_optimization,
                'enable_caching': self.config.enable_caching
            }
        }

    def cleanup(self):
        """Cleanup resources."""
        try:
            if self.hardware_manager:
                self.hardware_manager.shutdown()
            if self.vectorization_optimizer:
                self.vectorization_optimizer.cleanup()
            if self._parameter_cache:
                self._parameter_cache.clear()
            self.logger.info("🧹 Unified optimization system cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

# Global instance
_unified_optimization_system: Optional[UnifiedOptimizationSystem] = None

def get_unified_optimization_system(config: Optional[UnifiedOptimizationConfig] = None) -> UnifiedOptimizationSystem:
    """Get or create the global unified optimization system."""
    global _unified_optimization_system

    if _unified_optimization_system is None:
        _unified_optimization_system = UnifiedOptimizationSystem(config)

    return _unified_optimization_system

def optimize_features_unified(data: pd.DataFrame,
                            categories: Optional[List[str]] = None,
                            features: Optional[List[str]] = None,
                            target_column: Optional[str] = None,
                            config: Optional[UnifiedOptimizationConfig] = None,
                            **kwargs) -> OptimizationResult:
    """Convenience function for unified feature optimization."""
    system = get_unified_optimization_system(config)
    return system.process_features_unified(data, categories, features, target_column, **kwargs)
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and
                VECTORBT_AVAILABLE)

    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

    def _pandas_rolling_operation(self, data: pd.Series, operation: str,
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
