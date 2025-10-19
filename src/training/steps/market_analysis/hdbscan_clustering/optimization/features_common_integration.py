"""
Features Common Integration for HDBSCAN Clustering System

This module integrates the features_common/ systems with the HDBSCAN clustering
pipeline to provide enhanced normalization, optimization, and performance monitoring.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass

# Import features_common systems
from src.features_common import (
    UnifiedVectorizationManager, get_unified_vectorization_manager,
    VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer,
    OptimizationMixin, PerformanceMixin, VectorBTMixin,
    ValidationMixin, CachingMixin, MonitoringMixin
)

from src.features_common.transforms.scaling_normalization import ScalingNormalizer
from src.features_common.transforms.vectorbt_scaler import VectorBTScaler
from src.features_common.normalization import NormalizationFeatureGenerator
from src.features_common.config import get_unified_config, get_optimization_config

# Import feature generation systems
from src.feature_generation.categories.entropy import create_default_entropy_generators
from src.feature_generation.categories.spectral_wavelet import create_default_spectral_wavelet_generators
from src.feature_generation.categories.regime_features import create_default_regime_generators

logger = logging.getLogger(__name__)

@dataclass
class FeaturesCommonIntegrationConfig:
    """Configuration for features_common integration."""
    # Core features_common settings
    enable_unified_vectorization: bool = True
    enable_vectorbt_optimization: bool = True
    enable_automatic_scaling: bool = True
    enable_performance_monitoring: bool = True
    enable_caching: bool = True
    
    # Optimization settings
    optimization_level: str = "high"  # "high", "medium", "low"
    auto_tuning: bool = True
    adaptive_parameters: bool = True
    
    # Memory and performance
    memory_efficient: bool = True
    max_memory_gb: float = 8.0
    chunk_size: int = 1000
    
    # Feature generation
    enable_entropy_features: bool = True
    enable_spectral_features: bool = True
    enable_regime_features: bool = True
    enable_normalization_features: bool = True

class FeaturesCommonHDBSCANIntegration:
    """
    Enhanced HDBSCAN integration using features_common systems.
    
    This class integrates the comprehensive features_common infrastructure
    with the HDBSCAN clustering pipeline for maximum performance and optimization.
    """
    
    def __init__(self, config: Optional[FeaturesCommonIntegrationConfig] = None):
        """Initialize the features_common integration."""
        self.config = config or FeaturesCommonIntegrationConfig()
        
        # Initialize features_common components
        self._initialize_features_common_components()
        
        # Initialize feature generators
        self._initialize_feature_generators()
        
        # Performance tracking
        self.performance_stats = {
            'total_processing_time': 0.0,
            'vectorbt_operations': 0,
            'normalization_operations': 0,
            'caching_hits': 0,
            'optimization_improvements': 0,
            'memory_optimizations': 0
        }
        
        logger.info("✅ FeaturesCommonHDBSCANIntegration initialized")
    
    def _initialize_features_common_components(self):
        """Initialize features_common components."""
        # Get unified configuration
        self.unified_config = get_unified_config()
        self.optimization_config = get_optimization_config()
        
        # Initialize unified vectorization manager
        if self.config.enable_unified_vectorization:
            self.vectorization_manager = get_unified_vectorization_manager(
                config=self.unified_config
            )
        else:
            self.vectorization_manager = None
        
        # Initialize VectorBT rolling optimizer
        if self.config.enable_vectorbt_optimization:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=False,  # Can be configured
                enable_parallel=True,
                memory_efficient=self.config.memory_efficient,
                chunk_size=self.config.chunk_size
            )
        else:
            self.rolling_optimizer = None
        
        # Initialize scaling normalizer
        if self.config.enable_automatic_scaling:
            self.scaling_normalizer = ScalingNormalizer({
                'method': 'robust',  # Robust scaling for financial data
                'exclude_outliers': True,
                'outlier_threshold': 3.0,
                'use_vectorbt': self.config.enable_vectorbt_optimization
            })
        else:
            self.scaling_normalizer = None
        
        # Initialize VectorBT scaler
        if self.config.enable_vectorbt_optimization:
            self.vectorbt_scaler = VectorBTScaler(
                method='robust',
                enable_gpu=False,
                memory_efficient=True
            )
        else:
            self.vectorbt_scaler = None
    
    def _initialize_feature_generators(self):
        """Initialize feature generators with features_common integration."""
        self.feature_generators = []
        
        # Entropy features
        if self.config.enable_entropy_features:
            entropy_generators = create_default_entropy_generators()
            self.feature_generators.extend(entropy_generators)
        
        # Spectral features
        if self.config.enable_spectral_features:
            spectral_generators = create_default_spectral_wavelet_generators()
            self.feature_generators.extend(spectral_generators)
        
        # Regime features
        if self.config.enable_regime_features:
            regime_generators = create_default_regime_generators()
            self.feature_generators.extend(regime_generators)
        
        # Normalization features
        if self.config.enable_normalization_features:
            normalization_generator = NormalizationFeatureGenerator()
            self.feature_generators.append(normalization_generator)
    
    def process_data_with_features_common(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process data using features_common systems."""
        import time
        start_time = time.time()
        
        logger.info(f"🚀 Processing data with features_common: {data.shape}")
        
        # Step 1: Generate features using feature_generation/ system
        features_df = self._generate_features_with_optimization(data)
        
        # Step 2: Apply features_common normalization
        if self.scaling_normalizer:
            features_df = self._apply_features_common_normalization(features_df)
        
        # Step 3: Apply VectorBT optimization
        if self.vectorization_manager:
            features_df = self._apply_vectorbt_optimization(features_df)
        
        # Step 4: Final memory optimization
        features_df = self._apply_memory_optimization(features_df)
        
        # Update performance stats
        processing_time = time.time() - start_time
        self.performance_stats['total_processing_time'] += processing_time
        
        logger.info(f"✅ Data processing completed: {processing_time:.2f}s")
        
        return features_df
    
    def _generate_features_with_optimization(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate features with features_common optimization."""
        features_df = data.copy()
        
        for generator in self.feature_generators:
            try:
                # Use features_common optimization if available
                if hasattr(generator, 'generate_with_optimization'):
                    feature_result = generator.generate_with_optimization(
                        data, 
                        vectorization_manager=self.vectorization_manager,
                        rolling_optimizer=self.rolling_optimizer
                    )
                else:
                    feature_result = generator.generate(data)
                
                # Handle different result types
                if isinstance(feature_result, pd.DataFrame):
                    features_df = pd.concat([features_df, feature_result], axis=1)
                elif isinstance(feature_result, pd.Series):
                    features_df[feature_result.name] = feature_result
                
                self.performance_stats['vectorbt_operations'] += 1
                
            except Exception as e:
                logger.warning(f"⚠️ Feature generation failed for {generator.__class__.__name__}: {e}")
                continue
        
        return features_df
    
    def _apply_features_common_normalization(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Apply features_common normalization."""
        if not self.scaling_normalizer:
            return features_df
        
        try:
            # Use features_common scaling normalizer
            normalized_df = self.scaling_normalizer.fit_transform(features_df)
            
            # Apply VectorBT scaler if available
            if self.vectorbt_scaler:
                normalized_df = self.vectorbt_scaler.fit_transform(normalized_df)
            
            self.performance_stats['normalization_operations'] += 1
            
            logger.info("✅ Features_common normalization applied")
            
            return normalized_df
            
        except Exception as e:
            logger.warning(f"⚠️ Features_common normalization failed: {e}")
            return features_df
    
    def _apply_vectorbt_optimization(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Apply VectorBT optimization using features_common."""
        if not self.vectorization_manager:
            return features_df
        
        try:
            # Use unified vectorization manager for optimization
            optimized_df = self.vectorization_manager.optimize_dataframe(
                features_df,
                operation_type='clustering',
                enable_caching=self.config.enable_caching,
                memory_efficient=self.config.memory_efficient
            )
            
            self.performance_stats['optimization_improvements'] += 1
            
            logger.info("✅ VectorBT optimization applied")
            
            return optimized_df
            
        except Exception as e:
            logger.warning(f"⚠️ VectorBT optimization failed: {e}")
            return features_df
    
    def _apply_memory_optimization(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Apply memory optimization using features_common."""
        try:
            # Use features_common memory optimization
            if hasattr(self.vectorization_manager, 'optimize_memory'):
                optimized_df = self.vectorization_manager.optimize_memory(features_df)
            else:
                # Fallback to basic optimization
                optimized_df = features_df.copy()
                for col in optimized_df.select_dtypes(include=['float64']).columns:
                    optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
                for col in optimized_df.select_dtypes(include=['int64']).columns:
                    optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')
            
            self.performance_stats['memory_optimizations'] += 1
            
            return optimized_df
            
        except Exception as e:
            logger.warning(f"⚠️ Memory optimization failed: {e}")
            return features_df
    
    def get_features_common_benefits(self) -> Dict[str, Any]:
        """Get benefits of using features_common integration."""
        return {
            'unified_vectorization': {
                'description': 'Unified vectorization management across all operations',
                'benefits': [
                    'Automatic optimization selection',
                    'VectorBT integration when available',
                    'Fallback to pandas/numpy when needed',
                    'Memory-efficient processing'
                ]
            },
            'automatic_scaling': {
                'description': 'Intelligent scaling and normalization',
                'benefits': [
                    'Robust scaling for financial data',
                    'Outlier detection and handling',
                    'VectorBT-optimized scaling operations',
                    'Automatic method selection'
                ]
            },
            'performance_monitoring': {
                'description': 'Comprehensive performance monitoring',
                'benefits': [
                    'Real-time performance tracking',
                    'Automatic optimization decisions',
                    'Memory usage monitoring',
                    'Caching optimization'
                ]
            },
            'optimization_mixins': {
                'description': 'Automatic optimization capabilities',
                'benefits': [
                    'Adaptive parameter tuning',
                    'Performance-based optimization selection',
                    'Automatic fallback mechanisms',
                    'Intelligent caching strategies'
                ]
            }
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        
        # Add features_common specific stats
        if self.vectorization_manager:
            vectorization_stats = self.vectorization_manager.get_performance_stats()
            stats['vectorization_stats'] = vectorization_stats
        
        if self.rolling_optimizer:
            rolling_stats = self.rolling_optimizer.get_performance_stats()
            stats['rolling_optimizer_stats'] = rolling_stats
        
        if self.scaling_normalizer:
            scaling_stats = self.scaling_normalizer.get_performance_stats()
            stats['scaling_stats'] = scaling_stats
        
        return stats
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_processing_time': 0.0,
            'vectorbt_operations': 0,
            'normalization_operations': 0,
            'caching_hits': 0,
            'optimization_improvements': 0,
            'memory_optimizations': 0
        }

# Convenience function
def create_features_common_hdbscan_integration(
    enable_unified_vectorization: bool = True,
    enable_vectorbt_optimization: bool = True,
    enable_automatic_scaling: bool = True,
    enable_performance_monitoring: bool = True,
    enable_caching: bool = True,
    optimization_level: str = "high",
    memory_efficient: bool = True,
    max_memory_gb: float = 8.0
) -> FeaturesCommonHDBSCANIntegration:
    """
    Create a features_common HDBSCAN integration with specified configuration.
    
    Args:
        enable_unified_vectorization: Enable unified vectorization management
        enable_vectorbt_optimization: Enable VectorBT optimization
        enable_automatic_scaling: Enable automatic scaling and normalization
        enable_performance_monitoring: Enable performance monitoring
        enable_caching: Enable caching optimization
        optimization_level: Optimization level ("high", "medium", "low")
        memory_efficient: Enable memory optimization
        max_memory_gb: Maximum memory usage in GB
        
    Returns:
        FeaturesCommonHDBSCANIntegration instance
    """
    config = FeaturesCommonIntegrationConfig(
        enable_unified_vectorization=enable_unified_vectorization,
        enable_vectorbt_optimization=enable_vectorbt_optimization,
        enable_automatic_scaling=enable_automatic_scaling,
        enable_performance_monitoring=enable_performance_monitoring,
        enable_caching=enable_caching,
        optimization_level=optimization_level,
        memory_efficient=memory_efficient,
        max_memory_gb=max_memory_gb
    )
    
    return FeaturesCommonHDBSCANIntegration(config)
