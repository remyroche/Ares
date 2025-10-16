"""
import warnings
Optimized Feature Engineering Pipeline

This module provides a unified, hardware-optimized feature engineering pipeline
that ensures full compatibility between the feature bank, normalizer, and scaler
components with maximum vectorization and hardware utilization.
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Import core components
from ..core.feature_bank import FeatureBank, FeatureBankConfig, get_global_feature_bank
# Note: Removed direct import of NormalizationFeatureGenerator to avoid circular import
# Will use lazy import pattern instead
from ...utils.intensity_scaler import get_intensity_config, apply_intensity_scaling
from ...utils.matrix_operations import get_unified_matrix_operations
from ...utils.hardware.unified_hardware_manager import get_unified_hardware_manager, WorkloadType, OptimizationLevel
from ...utils.matrix_operations.vectorized_core import get_vectorized_processing_core

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for the optimized feature pipeline."""
    # Feature Bank Configuration
    enable_matrix_operations: bool = True
    enable_gpu_acceleration: bool = True
    enable_lookback_optimization: bool = True
    enable_parallel_processing: bool = True
    max_workers: int = 4
    chunk_size: int = 1000
    memory_efficient: bool = True
    cache_results: bool = True
    
    # Normalization Configuration
    auto_normalize: bool = True
    normalization_method: str = "zscore"  # "zscore", "minmax", "robust", "quantile"
    normalization_exclude_categories: List[str] = field(default_factory=list)
    normalization_exclude_features: List[str] = field(default_factory=list)
    normalization_rolling_windows: List[int] = field(default_factory=lambda: [20, 50, 100])
    
    # Scaling Configuration
    enable_intensity_scaling: bool = True
    intensity_percentage: Optional[float] = None  # Auto-detect from environment
    
    # Hardware Optimization
    enable_hardware_optimization: bool = True
    workload_type: WorkloadType = WorkloadType.FEATURE_ENGINEERING
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    
    # Performance Monitoring
    enable_performance_monitoring: bool = True
    enable_memory_tracking: bool = True
    enable_profiling: bool = False

@dataclass
class PipelineResult:
    """Result from feature pipeline execution."""
    features: pd.DataFrame
    normalization_params: Dict[str, Any]
    scaling_params: Dict[str, Any]
    performance_stats: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    processing_time: float = 0.0
    memory_usage: float = 0.0

class OptimizedFeaturePipeline:
    """
    Optimized feature engineering pipeline with full hardware acceleration
    and vectorization support.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the optimized feature pipeline."""
        self.config = config or PipelineConfig()
        self.logger = logger.getChild('OptimizedFeaturePipeline')
        
        # Initialize components
        self.feature_bank = None
        self.normalizer = None
        self.scaler = None
        self.matrix_ops = None
        self.hardware_manager = None
        self.vectorized_core = None
        
        # Performance tracking
        self.performance_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_processing_time': 0.0,
            'peak_memory_usage': 0.0,
            'hardware_accelerations': 0,
            'vectorized_operations': 0
        }
        
        # Initialize all components
        self._initialize_components()
        
        self.logger.info("✅ Optimized Feature Pipeline initialized")
    
    def _initialize_components(self):
        """Initialize all pipeline components with optimal configuration."""
        try:
            # Initialize Feature Bank with optimized configuration
            feature_bank_config = FeatureBankConfig(
                enable_matrix_operations=self.config.enable_matrix_operations,
                enable_gpu_acceleration=self.config.enable_gpu_acceleration,
                enable_lookback_optimization=self.config.enable_lookback_optimization,
                enable_parallel_processing=self.config.enable_parallel_processing,
                max_workers=self.config.max_workers,
                chunk_size=self.config.chunk_size,
                memory_efficient=self.config.memory_efficient,
                cache_results=self.config.cache_results,
                auto_normalize=False,  # We'll handle normalization separately
                normalization_method=self.config.normalization_method,
                normalization_exclude_categories=self.config.normalization_exclude_categories,
                normalization_exclude_features=self.config.normalization_exclude_features,
                normalization_rolling_windows=self.config.normalization_rolling_windows
            )
            
            self.feature_bank = FeatureBank(feature_bank_config)
            self.logger.info("✅ Feature Bank initialized")
            
            # Initialize Normalizer - using lazy import to avoid circular dependencies
            try:
                from ...training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.data_normalization import (
                    NormalizationConfig, NormalizationMethod, create_data_normalizer
                )
            except ImportError:
                # Fallback: use features_common normalization
                from src.features_common.normalization import create_data_normalizer
                # Create a simple config class for fallback
                class NormalizationConfig:
                    def __init__(self, method="ZSCORE", use_hardware_acceleration=False, 
                               use_matrix_operations=False, batch_size=1000, memory_limit_gb=8.0):
                        self.method = method
                        self.use_hardware_acceleration = use_hardware_acceleration
                        self.use_matrix_operations = use_matrix_operations
                        self.batch_size = batch_size
                        self.memory_limit_gb = memory_limit_gb
                
                class NormalizationMethod:
                    ZSCORE = "ZSCORE"
            
            normalization_config = NormalizationConfig(
                method=getattr(NormalizationMethod, self.config.normalization_method.upper(), NormalizationMethod.ZSCORE),
                use_hardware_acceleration=self.config.enable_hardware_optimization,
                use_matrix_operations=self.config.enable_matrix_operations,
                batch_size=self.config.chunk_size,
                memory_limit_gb=8.0
            )
            
            self.normalizer = create_data_normalizer(normalization_config)
            self.logger.info("✅ Normalizer initialized")
            
            # Initialize Scaler (Intensity Scaler)
            if self.config.enable_intensity_scaling:
                self.scaler = get_intensity_config(self.config.intensity_percentage)
                self.logger.info("✅ Intensity Scaler initialized")
            
            # Initialize Matrix Operations
            if self.config.enable_matrix_operations:
                self.matrix_ops = get_unified_matrix_operations()
                self.logger.info("✅ Matrix Operations initialized")
            
            # Initialize Hardware Manager
            if self.config.enable_hardware_optimization:
                self.hardware_manager = get_unified_hardware_manager()
                self.hardware_manager.optimize_for_workload(
                    self.config.workload_type, 
                    self.config.optimization_level
                )
                self.logger.info("✅ Hardware Manager initialized")
            
            # Initialize Vectorized Core
            self.vectorized_core = get_vectorized_processing_core()
            self.logger.info("✅ Vectorized Core initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize components: {e}")
            raise
    
    @contextmanager
    def _hardware_optimization_context(self):
        """Context manager for hardware optimization."""
        if self.hardware_manager:
            with self.hardware_manager.optimization_context(
                self.config.workload_type, 
                self.config.optimization_level
            ):
                yield
        else:
            yield
    
    def process_features(self, 
                       data: pd.DataFrame,
                       categories: Optional[List[str]] = None,
                       features: Optional[List[str]] = None,
                       target_column: Optional[str] = None,
                       **kwargs) -> PipelineResult:
        """
        Process features through the complete pipeline with hardware optimization.
        
        Args:
            data: Input DataFrame
            categories: List of feature categories to generate
            features: List of specific features to generate
            target_column: Target column for lookback optimization
            **kwargs: Additional parameters
            
        Returns:
            PipelineResult with processed features and metadata
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            self.logger.info("🚀 Starting optimized feature processing pipeline")
            self.logger.info(f"   Input shape: {data.shape}")
            self.logger.info(f"   Categories: {categories}")
            self.logger.info(f"   Features: {features}")
            
            with self._hardware_optimization_context():
                # Step 1: Generate features using Feature Bank
                features_df = self._generate_features_optimized(
                    data, categories, features, target_column, **kwargs
                )
                
                # Step 2: Apply normalization
                normalized_df, normalization_params = self._apply_normalization_optimized(
                    features_df, categories
                )
                
                # Step 3: Apply scaling
                scaled_df, scaling_params = self._apply_scaling_optimized(
                    normalized_df
                )
                
                # Step 4: Final optimization
                final_df = self._finalize_features(scaled_df)
                
                processing_time = time.time() - start_time
                memory_usage = self._get_memory_usage() - start_memory
                
                # Update performance stats
                self._update_performance_stats(processing_time, memory_usage, True)
                
                self.logger.info(f"✅ Feature processing completed in {processing_time:.3f}s")
                self.logger.info(f"   Generated features: {len(final_df.columns)}")
                self.logger.info(f"   Memory usage: {memory_usage:.2f}MB")
                
                return PipelineResult(
                    features=final_df,
                    normalization_params=normalization_params,
                    scaling_params=scaling_params,
                    performance_stats=self.performance_stats.copy(),
                    success=True,
                    processing_time=processing_time,
                    memory_usage=memory_usage
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - start_memory
            
            self.logger.error(f"❌ Feature processing failed: {e}")
            self._update_performance_stats(processing_time, memory_usage, False)
            
            return PipelineResult(
                features=pd.DataFrame(),
                normalization_params={},
                scaling_params={},
                performance_stats=self.performance_stats.copy(),
                success=False,
                error_message=str(e),
                processing_time=processing_time,
                memory_usage=memory_usage
            )
    
    def _generate_features_optimized(self, 
                                   data: pd.DataFrame,
                                   categories: Optional[List[str]] = None,
                                   features: Optional[List[str]] = None,
                                   target_column: Optional[str] = None,
                                   **kwargs) -> pd.DataFrame:
        """Generate features with hardware optimization."""
        try:
            # Use vectorized core for optimization
            if self.vectorized_core:
                data = self.vectorized_core.optimize_dataframe_for_processing(data)
            
            # Generate features using Feature Bank
            features_df = self.feature_bank.generate_features(
                data=data,
                categories=categories,
                features=features,
                lookback_optimization=bool(target_column),
                target_column=target_column,
                **kwargs
            )
            
            self.performance_stats['vectorized_operations'] += 1
            return features_df
            
        except Exception as e:
            self.logger.error(f"Feature generation failed: {e}")
            raise
    
    def _apply_normalization_optimized(self, 
                                     features_df: pd.DataFrame,
                                     categories: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply normalization with hardware optimization."""
        try:
            if not self.config.auto_normalize or features_df.empty:
                return features_df, {}
            
            # Select features for normalization
            target_columns = self._select_normalization_targets(features_df, categories)
            
            if not target_columns:
                return features_df, {}
            
            # Apply normalization using the normalizer
            normalization_result = self.normalizer.normalize_data(
                features_df, target_columns=target_columns
            )
            
            if normalization_result.success:
                self.performance_stats['hardware_accelerations'] += 1
                return normalization_result.normalized_data, normalization_result.normalization_params
            else:
                self.logger.warning("Normalization failed, returning original features")
                return features_df, {}
                
        except Exception as e:
            self.logger.error(f"Normalization failed: {e}")
            return features_df, {}
    
    def _apply_scaling_optimized(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply intensity scaling with optimization."""
        try:
            if not self.config.enable_intensity_scaling or features_df.empty:
                return features_df, {}
            
            # Apply intensity scaling to configuration
            scaling_params = {
                'intensity_percentage': self.scaler.intensity_percentage,
                'training_mode': self.scaler.training_mode,
                'scaled_parameters': {}
            }
            
            # Scale feature generation parameters if needed
            if hasattr(self.feature_bank, 'config'):
                scaled_config = apply_intensity_scaling(
                    self.feature_bank.config.__dict__, 
                    self.scaler.intensity_percentage
                )
                scaling_params['scaled_parameters'] = scaled_config
            
            return features_df, scaling_params
            
        except Exception as e:
            self.logger.error(f"Scaling failed: {e}")
            return features_df, {}
    
    def _finalize_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Finalize features with additional optimizations."""
        try:
            if features_df.empty:
                return features_df
            
            # Apply final vectorized optimizations
            if self.vectorized_core:
                features_df = self.vectorized_core.optimize_dataframe_for_processing(features_df)
            
            # Remove any remaining NaN values
            features_df = features_df.fillna(0)
            
            # Ensure numeric types are optimized
            for col in features_df.select_dtypes(include=[np.number]).columns:
                if features_df[col].dtype == np.float64:
                    # Check if float32 is sufficient
                    if (features_df[col].max() < np.finfo(np.float32).max and
                        features_df[col].min() > np.finfo(np.float32).min):
                        features_df[col] = features_df[col].astype(np.float32)
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Feature finalization failed: {e}")
            return features_df
    
    def _select_normalization_targets(self, 
                                    features_df: pd.DataFrame,
                                    categories: Optional[List[str]] = None) -> List[str]:
        """Select which features should be normalized."""
        target_columns = []
        
        # Get numeric columns
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns.tolist()
        
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
        # Simple heuristic - in practice, you'd maintain a proper mapping
        excluded_indicators = ['zscore', 'normalized', 'scaled', 'rank']
        return any(indicator in feature_name.lower() for indicator in excluded_indicators)
    
    def _is_already_normalized(self, feature_name: str) -> bool:
        """Check if a feature is already normalized."""
        normalized_indicators = [
            'rsi', 'stoch', 'williams', 'macd_hist', 'bb_percent',
            'adx', 'cci', 'momentum', 'roc', 'zscore', 'normalized'
        ]
        return any(indicator in feature_name.lower() for indicator in normalized_indicators)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
        except Exception:
            return 0.0

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

except ImportError:
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")
    
    cp = None
    
    def _update_performance_stats(self, processing_time: float, memory_usage: float, success: bool):
        """Update performance statistics."""
        self.performance_stats['total_executions'] += 1
        if success:
            self.performance_stats['successful_executions'] += 1
        else:
            self.performance_stats['failed_executions'] += 1
        
        # Update average processing time
        total_time = self.performance_stats['average_processing_time'] * (self.performance_stats['total_executions'] - 1)
        self.performance_stats['average_processing_time'] = (total_time + processing_time) / self.performance_stats['total_executions']
        
        # Update peak memory usage
        self.performance_stats['peak_memory_usage'] = max(
            self.performance_stats['peak_memory_usage'], 
            memory_usage
        )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            'pipeline_stats': self.performance_stats.copy(),
            'component_status': {
                'feature_bank': self.feature_bank is not None,
                'normalizer': self.normalizer is not None,
                'scaler': self.scaler is not None,
                'matrix_ops': self.matrix_ops is not None,
                'hardware_manager': self.hardware_manager is not None,
                'vectorized_core': self.vectorized_core is not None
            },
            'config': {
                'auto_normalize': self.config.auto_normalize,
                'normalization_method': self.config.normalization_method,
                'enable_hardware_optimization': self.config.enable_hardware_optimization,
                'enable_matrix_operations': self.config.enable_matrix_operations,
                'workload_type': self.config.workload_type.value,
                'optimization_level': self.config.optimization_level.value
            }
        }
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            if self.hardware_manager:
                self.hardware_manager.shutdown()
            if self.feature_bank:
                self.feature_bank.clear_cache()
            self.logger.info("🧹 Pipeline cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

# Global instance
_optimized_pipeline: Optional[OptimizedFeaturePipeline] = None

def get_optimized_feature_pipeline(config: Optional[PipelineConfig] = None) -> OptimizedFeaturePipeline:
    """Get or create the global optimized feature pipeline instance."""
    global _optimized_pipeline
    
    if _optimized_pipeline is None:
        _optimized_pipeline = OptimizedFeaturePipeline(config)
    
    return _optimized_pipeline

def process_features_optimized(data: pd.DataFrame,
                             categories: Optional[List[str]] = None,
                             features: Optional[List[str]] = None,
                             target_column: Optional[str] = None,
                             config: Optional[PipelineConfig] = None,
                             **kwargs) -> PipelineResult:
    """
    Convenience function to process features through the optimized pipeline.
    
    Args:
        data: Input DataFrame
        categories: List of feature categories to generate
        features: List of specific features to generate
        target_column: Target column for lookback optimization
        config: Optional pipeline configuration
        **kwargs: Additional parameters
        
    Returns:
        PipelineResult with processed features and metadata
    """
    pipeline = get_optimized_feature_pipeline(config)
    return pipeline.process_features(data, categories, features, target_column, **kwargs)
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
