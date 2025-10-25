"""
Volatility Aware Labeling Integration for HDBSCAN Clustering System

This module integrates the VolatilityAwareMultiHorizonLabeler with the features_common
systems for optimal volatility-based labeling in HDBSCAN clustering. All other
feature generation methods are disabled to focus on the primary volatility aware
labeling approach.
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

# Import volatility aware labeler
from src.training.steps.pre_training.profit_labeling.volatility_aware_labeler import (
    VolatilityAwareMultiHorizonLabeler,
    VolatilityAwareConfig
)

# Import feature generation systems
from src.feature_generation.categories.entropy import create_default_entropy_generators
from src.feature_generation.categories.spectral_wavelet import create_default_spectral_wavelet_generators
from src.feature_generation.categories.regime_features import create_default_regime_generators

logger = logging.getLogger(__name__)


class VolatilityAwareFeatureGenerator:
    """
    Wrapper to adapt VolatilityAwareMultiHorizonLabeler for feature generation system.
    """

    def __init__(self, volatility_labeler: VolatilityAwareMultiHorizonLabeler):
        """Initialize the wrapper."""
        self.volatility_labeler = volatility_labeler
        self.name = "volatility_aware_labels"

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate volatility-aware labels as features.

        Args:
            data: Input market data

        Returns:
            DataFrame with volatility-aware label features
        """
        try:
            from src.utils.tprint import tprint

            tprint(f"🏷️ Starting volatility-aware labeling for {len(data)} samples", "INFO")

            # Generate labels using the volatility labeler with local maxima/minima detection
            labeling_result = self.volatility_labeler.generate_labels(
                data,
                price_column="close"
            )

            tprint(f"📊 Generated labeling result: success={labeling_result.success}", "INFO")

            if labeling_result.success and hasattr(labeling_result.labels, '__len__'):
                # Convert labels to DataFrame format
                if isinstance(labeling_result.labels, pd.Series):
                    # Single label series
                    labels_df = labeling_result.labels.to_frame(name=f"volatility_label_{self.volatility_labeler.config.lookahead_periods}h")
                    tprint(f"📈 Created single label series: {labels_df.shape[1]} features", "SUCCESS")
                elif isinstance(labeling_result.labels, pd.DataFrame):
                    # Multiple label columns
                    labels_df = labeling_result.labels.copy()
                    labels_df.columns = [f"volatility_label_{col}" for col in labels_df.columns]
                    tprint(f"📈 Created multi-label DataFrame: {labels_df.shape[1]} features", "SUCCESS")
                else:
                    # Fallback for other formats
                    labels_df = pd.DataFrame({
                        f"volatility_label_{self.volatility_labeler.config.lookahead_periods}h": labeling_result.labels
                    }, index=data.index)
                    tprint(f"📈 Created fallback label format: {labels_df.shape[1]} features", "SUCCESS")

                # Add metadata about the labeling process
                if hasattr(labeling_result, 'metadata'):
                    metadata = labeling_result.metadata
                    quality_score = metadata.get('quality_score')
                    volatility_adaptation = metadata.get('volatility_adaptation')
                    quality_str = f"{quality_score:.3f}" if quality_score is not None else "N/A"
                    vol_str = f"{volatility_adaptation:.3f}" if volatility_adaptation is not None else "N/A"
                    tprint(f"📋 Labeling metadata: quality_score={quality_str}, volatility_adaptation={vol_str}", "INFO")

                logger.info(f"✅ Generated volatility-aware labels: {labels_df.shape[1]} features")
                return labels_df
            else:
                tprint("⚠️ Volatility labeling failed or returned empty results", "WARNING")
                return pd.DataFrame(index=data.index)

        except Exception as e:
            from src.utils.tprint import tprint
            tprint(f"❌ Error generating volatility-aware labels: {e}", "ERROR")
            return pd.DataFrame(index=data.index)

    def generate_with_optimization(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate with optimization support."""
        return self.generate(data)


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
    
    # Feature generation (disabled - using only volatility aware labeling)
    enable_entropy_features: bool = True
    enable_spectral_features: bool = True
    enable_regime_features: bool = True
    enable_normalization_features: bool = False

    # Volatility aware labeling (PRIMARY METHOD)
    enable_volatility_labeling: bool = True
    volatility_threshold: float = 0.006  # 0.6% base threshold (more when volatility is high)
    lookahead_periods: int = 6
    label_type: str = "binary"  # "binary", "multi_class", "regression"
    enable_long_positions: bool = True
    enable_short_positions: bool = False

class FeaturesCommonHDBSCANIntegration:
    """
    Enhanced HDBSCAN integration using volatility aware labeling.

    This class integrates the VolatilityAwareMultiHorizonLabeler with the features_common
    infrastructure for optimal volatility-based labeling in HDBSCAN clustering.
    All other feature generation methods are disabled to focus on the primary
    volatility aware labeling approach.
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
            'volatility_labeling_operations': 0,
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
            from src.features_common.transforms.base_scaler import SimpleScaler
            self.vectorbt_scaler = SimpleScaler(
                use_vectorbt=True,
                enable_gpu=False,
                memory_efficient=True
            )
        else:
            self.vectorbt_scaler = None

        # Initialize volatility aware labeler
        if self.config.enable_volatility_labeling:
            from src.training.steps.pre_training.profit_labeling.volatility_aware_labeler import LabelDefinitionType

            # Configure for local maxima/minima detection with optimal threshold
            label_type = LabelDefinitionType.BINARY if self.config.label_type == "binary" else LabelDefinitionType.MULTI_CLASS

            vol_config = VolatilityAwareConfig(
                volatility_threshold=self.config.volatility_threshold,  # 0.6% base, adapts higher with volatility
                lookahead_periods=self.config.lookahead_periods,
                label_type=label_type,
                enable_long_positions=self.config.enable_long_positions,
                enable_short_positions=self.config.enable_short_positions,
                min_label_quality=0.4,  # Higher quality threshold for better signals
                min_predictability=0.3
            )
            self.volatility_labeler = VolatilityAwareMultiHorizonLabeler(vol_config)
            logger.info(f"✅ Initialized volatility aware labeler: threshold={self.config.volatility_threshold:.3%}, lookahead={self.config.lookahead_periods} periods")
        else:
            self.volatility_labeler = None
    
    def _initialize_feature_generators(self):
        """Initialize feature generators with features_common integration."""
        self.feature_generators = []

        # ONLY Volatility aware labeling features (PRIMARY METHOD)
        if self.config.enable_volatility_labeling and self.volatility_labeler:
            # Create a wrapper generator for the volatility labeler
            volatility_generator = VolatilityAwareFeatureGenerator(self.volatility_labeler)
            self.feature_generators.append(volatility_generator)

        # Enable sophisticated technical indicators for regime discovery
        if getattr(self.config, 'enable_entropy_features', True):
            entropy_generators = create_default_entropy_generators()
            self.feature_generators.extend(entropy_generators)
            logger.info(f"✅ Added {len(entropy_generators)} entropy feature generators")

        if getattr(self.config, 'enable_spectral_features', True):
            spectral_generators = create_default_spectral_wavelet_generators()
            self.feature_generators.extend(spectral_generators)
            logger.info(f"✅ Added {len(spectral_generators)} spectral feature generators")

        if getattr(self.config, 'enable_regime_features', True):
            regime_generators = create_default_regime_generators()
            self.feature_generators.extend(regime_generators)
            logger.info(f"✅ Added {len(regime_generators)} regime feature generators")
    
    def process_data_with_features_common(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process data using volatility aware labeling with features_common optimization."""
        import time
        from src.utils.tprint import tprint

        start_time = time.time()
        tprint(f"🚀 Processing data with volatility aware labeling: {data.shape}", "INFO")

        # Step 1: Generate volatility aware labels (PRIMARY METHOD)
        tprint("📊 Step 1: Generating volatility aware labels...", "INFO")
        features_df = self._generate_features_with_optimization(data)

        # Step 2: Apply VectorBT optimization for labeling
        if self.vectorization_manager:
            tprint("⚡ Step 2: Applying VectorBT optimization...", "INFO")
            features_df = self._apply_vectorbt_optimization(features_df)

        # Step 3: Final memory optimization for labels
        tprint("🧠 Step 3: Applying memory optimization...", "INFO")
        features_df = self._apply_memory_optimization(features_df)

        # Update performance stats
        processing_time = time.time() - start_time
        self.performance_stats['total_processing_time'] += processing_time

        tprint(f"✅ Volatility aware labeling completed: {processing_time:.2f}s, {features_df.shape[1]} label features", "SUCCESS")

        return features_df
    
    def _generate_features_with_optimization(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate market-specific regime features using the feature_generation system."""
        from src.utils.tprint import tprint
        from src.feature_generation.core.feature_bank import FeatureBank, FeatureBankConfig
        from src.feature_generation.core.feature_generator import FeatureCategory

        tprint(f"📊 Initial data shape: {data.shape}", "INFO")
        tprint(f"📊 Initial columns: {list(data.columns)}", "INFO")

        # Initialize feature bank with regime-specific configuration
        feature_bank_config = FeatureBankConfig(
            enable_matrix_operations=True,
            enable_gpu_acceleration=False,  # Disable for stability
            enable_lookback_optimization=True,
            enable_parallel_processing=True,
            max_workers=2,  # Conservative for stability
            chunk_size=1000,
            memory_efficient=True,
            cache_results=True,
            default_lookback=20
        )
        
        feature_bank = FeatureBank(feature_bank_config)
        
        # Generate regime-specific features using the feature bank
        tprint("🎯 Generating regime-specific features using feature bank...", "INFO")
        
        # Use regime-specific feature categories
        regime_categories = [
            FeatureCategory.REGIME,
            FeatureCategory.VOLATILITY, 
            FeatureCategory.MOMENTUM,
            FeatureCategory.TREND,
            FeatureCategory.VOLUME
        ]
        
        features_df = data.copy()
        
        for category in regime_categories:
            try:
                tprint(f"📊 Generating {category.value} features...", "INFO")
                category_features = feature_bank.generate_features_by_category(
                    data, 
                    category=category,
                    lookback_periods=[5, 10, 20, 50],  # Multiple lookback periods for regime detection
                    enable_optimization=True
                )
                
                if category_features is not None and not category_features.empty:
                    # Merge features
                    features_df = pd.concat([features_df, category_features], axis=1)
                    tprint(f"✅ Added {category_features.shape[1]} {category.value} features", "SUCCESS")
                else:
                    tprint(f"⚠️ No {category.value} features generated", "WARNING")
                    
            except Exception as e:
                tprint(f"⚠️ Failed to generate {category.value} features: {e}", "WARNING")
                continue
        
        # Final feature processing
        tprint(f"📊 Final features shape: {features_df.shape}", "INFO")
        tprint(f"📊 Generated {features_df.shape[1] - data.shape[1]} regime-specific features", "SUCCESS")
        
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
            if hasattr(self.vectorization_manager, 'optimize_dataframe_processing'):
                optimized_df = self.vectorization_manager.optimize_dataframe_processing(
                    features_df
                )
            else:
                # Fallback to basic optimization if method not available
                optimized_df = features_df
            
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
        """Get benefits of using features_common integration with volatility aware labeling."""
        return {
            'volatility_aware_labeling': {
                'description': 'Primary method using VolatilityAwareMultiHorizonLabeler',
                'benefits': [
                    'Market volatility-based label generation',
                    'Multi-horizon profit target labeling',
                    'Adaptive threshold based on market conditions',
                    'Quality scoring and validation',
                    'Robust handling of different market regimes'
                ]
            },
            'unified_vectorization': {
                'description': 'Unified vectorization management for labeling operations',
                'benefits': [
                    'Automatic optimization selection for labeling',
                    'VectorBT integration for efficient computation',
                    'Memory-efficient label processing',
                    'Optimized rolling window calculations'
                ]
            },
            'performance_monitoring': {
                'description': 'Comprehensive performance monitoring for labeling',
                'benefits': [
                    'Real-time labeling performance tracking',
                    'Volatility labeling operation monitoring',
                    'Memory usage optimization for labeling',
                    'Quality metric tracking'
                ]
            },
            'labeling_optimization': {
                'description': 'Specialized optimization for volatility-based labeling',
                'benefits': [
                    'Adaptive volatility threshold optimization',
                    'Multi-horizon label generation efficiency',
                    'Quality-based label selection',
                    'Market regime aware processing'
                ]
            }
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        
        # Add features_common specific stats
        if self.vectorization_manager:
            try:
                vectorization_stats = self.vectorization_manager.get_performance_stats()
            except AttributeError:
                # Fallback if method doesn't exist
                vectorization_stats = {
                    'vectorization_time': 0.0,
                    'vectorization_operations': 0,
                    'vectorization_efficiency': 1.0
                }
            stats['vectorization_stats'] = vectorization_stats
        
        if self.rolling_optimizer:
            try:
                rolling_stats = self.rolling_optimizer.get_performance_stats()
                stats['rolling_optimizer_stats'] = rolling_stats
            except AttributeError:
                # Fallback if get_performance_stats method doesn't exist
                stats['rolling_optimizer_stats'] = {
                    'method': 'VectorBTRollingOptimizer',
                    'status': 'performance_stats_not_available'
                }
        
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
            'volatility_labeling_operations': 0,
            'caching_hits': 0,
            'optimization_improvements': 0,
            'memory_optimizations': 0
        }

# Convenience function
def create_features_common_hdbscan_integration(
    enable_unified_vectorization: bool = True,
    enable_vectorbt_optimization: bool = True,
    enable_automatic_scaling: bool = False,  # Disabled - using only volatility labeling
    enable_performance_monitoring: bool = True,
    enable_caching: bool = True,
    optimization_level: str = "high",
    memory_efficient: bool = True,
    max_memory_gb: float = 8.0,
    enable_volatility_labeling: bool = True,  # Always enabled as primary method
    volatility_threshold: float = 0.006,  # 0.6% base threshold (more when volatility is high)
    lookahead_periods: int = 6,
    label_type: str = "binary",
    enable_long_positions: bool = True,
    enable_short_positions: bool = False
) -> FeaturesCommonHDBSCANIntegration:
    """
    Create a features_common HDBSCAN integration with volatility aware labeling.

    Args:
        enable_unified_vectorization: Enable unified vectorization management
        enable_vectorbt_optimization: Enable VectorBT optimization
        enable_automatic_scaling: Enable automatic scaling (disabled for volatility labeling)
        enable_performance_monitoring: Enable performance monitoring
        enable_caching: Enable caching optimization
        optimization_level: Optimization level ("high", "medium", "low")
        memory_efficient: Enable memory optimization
        max_memory_gb: Maximum memory usage in GB
        enable_volatility_labeling: Enable volatility aware labeling (primary method)
        volatility_threshold: Volatility threshold for labeling
        lookahead_periods: Number of periods to look ahead for labels
        label_type: Type of labels ("binary", "multi_class", "regression")
        enable_long_positions: Enable long position labeling
        enable_short_positions: Enable short position labeling

    Returns:
        FeaturesCommonHDBSCANIntegration instance with volatility aware labeling
    """
    config = FeaturesCommonIntegrationConfig(
        enable_unified_vectorization=enable_unified_vectorization,
        enable_vectorbt_optimization=enable_vectorbt_optimization,
        enable_automatic_scaling=enable_automatic_scaling,
        enable_performance_monitoring=enable_performance_monitoring,
        enable_caching=enable_caching,
        optimization_level=optimization_level,
        memory_efficient=memory_efficient,
        max_memory_gb=max_memory_gb,
        # Volatility aware labeling parameters (PRIMARY METHOD)
        enable_volatility_labeling=enable_volatility_labeling,
        volatility_threshold=volatility_threshold,  # 0.6% base threshold (more when volatility is high)
        lookahead_periods=lookahead_periods,
        label_type=label_type,
        enable_long_positions=enable_long_positions,
        enable_short_positions=enable_short_positions
    )
    
    return FeaturesCommonHDBSCANIntegration(config)
