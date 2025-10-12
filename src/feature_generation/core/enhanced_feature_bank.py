"""
Enhanced Feature Bank System

This module provides an enhanced FeatureBank class that integrates all centralized
utilities, eliminating code duplication and improving performance.
"""

import copy
import logging
import time
from typing import Any, Dict, List, Optional, Union, Set
from dataclasses import dataclass
import pandas as pd
import numpy as np

from .feature_generator import FeatureGenerator, FeatureCategory, FeatureResult, FeatureConfig
from .feature_registry import FeatureRegistry
from .unified_feature_generator import UnifiedFeatureConfig
from ..utils.centralized_rolling_manager import get_centralized_rolling_manager
from ..utils.scaler_factory import get_scaler_factory
from ..utils.common_operations import get_common_operations
from ..utils.unified_vectorization_manager import get_unified_vectorization_manager
from src.utils.unified_cache import UnifiedCache

logger = logging.getLogger(__name__)

@dataclass
class EnhancedFeatureBankConfig:
    """Enhanced configuration for the feature bank."""
    # Core functionality
    enable_matrix_operations: bool = True
    enable_gpu_acceleration: bool = True
    enable_lookback_optimization: bool = True
    enable_parallel_processing: bool = True
    max_workers: int = 4
    chunk_size: int = 1000
    memory_efficient: bool = True
    cache_results: bool = True
    default_lookback: int = 20
    
    # Centralized utilities configuration
    enable_centralized_rolling: bool = True
    enable_centralized_scaling: bool = True
    enable_common_operations: bool = True
    enable_unified_vectorization: bool = True
    
    # Performance optimization
    enable_batch_processing: bool = True
    enable_memory_optimization: bool = True
    enable_performance_tracking: bool = True
    
    # Normalization configuration
    auto_normalize: bool = True
    normalization_method: str = 'zscore'
    normalization_feature_type: str = 'default'
    normalization_exclude_categories: List[str] = None
    normalization_exclude_features: List[str] = None
    normalization_rolling_windows: List[int] = None
    
    # State management
    persist_generator_state: bool = True
    state_cache_dir: str = "data_cache/feature_states"
    state_cache_namespace: str = "enhanced_feature_bank"
    state_cache_ttl_seconds: Optional[int] = None

class EnhancedFeatureBank:
    """
    Enhanced feature bank that integrates all centralized utilities.
    
    This class eliminates code duplication by using centralized rolling operations,
    scaling, and common operations utilities.
    """
    
    VERSION = "2024.09.enhanced"
    
    def __init__(self, config: Optional[EnhancedFeatureBankConfig] = None):
        """
        Initialize the enhanced feature bank.
        
        Args:
            config: Enhanced feature bank configuration
        """
        self.config = config or EnhancedFeatureBankConfig()
        self.logger = logger.getChild('EnhancedFeatureBank')

        # Initialize feature registry
        self.registry = FeatureRegistry()
        
        # Initialize centralized utilities
        self.rolling_manager = None
        self.scaler_factory = None
        self.common_operations = None
        self.vectorization_manager = None
        
        if self.config.enable_centralized_rolling:
            self.rolling_manager = get_centralized_rolling_manager()
            self.logger.debug("✅ Centralized rolling manager enabled")
        
        if self.config.enable_centralized_scaling:
            self.scaler_factory = get_scaler_factory()
            self.logger.debug("✅ Centralized scaler factory enabled")
        
        if self.config.enable_common_operations:
            self.common_operations = get_common_operations()
            self.logger.debug("✅ Common operations enabled")
        
        if self.config.enable_unified_vectorization:
            self.vectorization_manager = get_unified_vectorization_manager()
            self.logger.debug("✅ Unified vectorization manager enabled")
        
        # Initialize matrix operations if enabled
        self.matrix_ops = None
        self.matrix_accelerator = None

        if self.config.enable_matrix_operations:
            try:
                from ...utils.matrix_operations import get_unified_matrix_operations
                self.matrix_ops = get_unified_matrix_operations()
                self.logger.debug("✅ Matrix operations enabled")
            except ImportError:
                self.logger.warning("⚠️ Matrix operations not available")

            # Initialize enhanced matrix accelerator
            try:
                from ..utils.enhanced_matrix_accelerator import get_enhanced_matrix_accelerator
                self.matrix_accelerator = get_enhanced_matrix_accelerator(
                    enable_gpu=self.config.enable_gpu_acceleration,
                    enable_parallel=self.config.enable_parallel_processing
                )
                self.logger.debug("✅ Enhanced matrix accelerator enabled")
            except ImportError:
                self.logger.warning("⚠️ Enhanced matrix accelerator not available")
        
        # Initialize lookback optimizer if enabled
        self.lookback_optimizer = None
        if self.config.enable_lookback_optimization:
            try:
                from ..utils.optimization import LookbackOptimizer
                self.lookback_optimizer = LookbackOptimizer()
                self.logger.info("✅ Lookback optimization enabled")
            except ImportError:
                self.logger.warning("⚠️ Lookback optimization not available")
        
        # Performance tracking
        self.performance_stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'categories_used': set(),
            'features_generated': 0,
            'average_generation_time': 0.0,
            'total_generation_time': 0.0,
            'normalization_applied': 0,
            'matrix_accelerations': 0,
            'centralized_rolling_operations': 0,
            'centralized_scaling_operations': 0,
            'common_operations_used': 0,
            'batch_operations': 0,
            'memory_optimizations': 0
        }

        # Normalization configuration
        self.normalization_config = {
            'exclude_categories': self.config.normalization_exclude_categories or [],
            'exclude_features': self.config.normalization_exclude_features or [],
            'rolling_windows': self.config.normalization_rolling_windows or [20, 50, 100]
        }
        
        # Cache for generated features
        self.feature_cache = {} if self.config.cache_results else None

        # Persistent state cache for generator-level rolling state
        self.persist_generator_state = self.config.persist_generator_state
        if self.persist_generator_state:
            self.state_cache = UnifiedCache(
                cache_dir=self.config.state_cache_dir,
                namespace=self.config.state_cache_namespace,
                default_ttl_seconds=self.config.state_cache_ttl_seconds,
                enable_disk=True,
                enable_compression=True
            )
        else:
            self.state_cache = None
        
        # Auto-register default generators
        self._auto_register_generators()

        self.logger.info("✅ Enhanced FeatureBank initialized")
        self.logger.info(f"📊 Centralized utilities: rolling={self.config.enable_centralized_rolling}, "
                        f"scaling={self.config.enable_centralized_scaling}, "
                        f"common_ops={self.config.enable_common_operations}")

    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """Helper to fetch values from dataclass or dict configs."""
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

    def _auto_register_generators(self) -> None:
        """Auto-register default feature generators from all categories."""
        try:
            # List of categories to auto-register
            categories_to_register = [
                FeatureCategory.MOMENTUM,
                FeatureCategory.VOLATILITY,
                FeatureCategory.TREND,
                FeatureCategory.VOLUME,
                FeatureCategory.SUPPORT_RESISTANCE,
                FeatureCategory.RETURNS,
                FeatureCategory.OSCILLATOR,
                FeatureCategory.CANDLESTICK_PATTERN,
                FeatureCategory.ENTROPY,
                FeatureCategory.ORDER_FLOW,
                FeatureCategory.ACCELERATION,
                FeatureCategory.CROSS_TIMEFRAME,
                FeatureCategory.AUTOENCODER,
                FeatureCategory.INTERACTION,
                FeatureCategory.MICROSTRUCTURE,
                FeatureCategory.REGIME,
                FeatureCategory.TIME,
                FeatureCategory.NORMALIZATION,
                FeatureCategory.REPRESENTATION_LEARNING,
                FeatureCategory.ADVANCED_STATISTICAL,
                FeatureCategory.SPECTRAL_WAVELET
            ]

            registered_count = 0
            total_categories = len(categories_to_register)
            self.logger.info(f"🚀 Initializing {total_categories} feature categories...")
            
            for i, category in enumerate(categories_to_register, 1):
                try:
                    self.logger.debug(f"📁 Registering category {i}/{total_categories}: {category.value}")
                    category_count = self.registry.register_category(category)
                    registered_count += category_count
                    self.logger.debug(f"✅ Registered {category_count} generators for {category.value}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to register category {category.value}: {e}")
            
            self.logger.info(f"✅ Auto-registration complete: {registered_count} generators registered")
            
        except Exception as e:
            self.logger.error(f"❌ Auto-registration failed: {e}")
            raise

    def generate_features_optimized(self, data: pd.DataFrame, 
                                   categories: Optional[List[FeatureCategory]] = None,
                                   features: Optional[List[str]] = None,
                                   **kwargs) -> Dict[str, pd.Series]:
        """
        Generate features using centralized utilities for better performance.
        
        Args:
            data: Input DataFrame with OHLCV data
            categories: List of feature categories to generate
            features: List of specific features to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary mapping feature names to generated series
        """
        start_time = time.time()
        
        try:
            # Optimize DataFrame for processing
            if self.config.enable_memory_optimization and self.vectorization_manager:
                data = self.vectorization_manager.optimize_dataframe(data)
                self.performance_stats['memory_optimizations'] += 1
            
            # Get generators to use
            generators = self._get_generators_for_categories(categories, features)
            
            # Generate features using centralized utilities
            results = {}
            
            if self.config.enable_batch_processing and len(generators) > 1:
                results = self._batch_generate_features(data, generators, **kwargs)
            else:
                results = self._sequential_generate_features(data, generators, **kwargs)
            
            # Apply normalization if enabled
            if self.config.auto_normalize:
                results = self._apply_normalization(results, categories, features)
            
            # Update performance stats
            self._update_performance_stats(start_time, len(results), True)
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Feature generation failed: {e}")
            self._update_performance_stats(start_time, 0, False)
            raise

    def _get_generators_for_categories(self, categories: Optional[List[FeatureCategory]] = None,
                                      features: Optional[List[str]] = None) -> List[FeatureGenerator]:
        """Get generators for specified categories and features."""
        generators = []
        
        if categories:
            for category in categories:
                category_generators = self.registry.get_generators_by_category(category)
                generators.extend(category_generators)
        elif features:
            for feature_name in features:
                generator = self.registry.get_generator_by_name(feature_name)
                if generator:
                    generators.append(generator)
        else:
            # Get all generators
            generators = self.registry.get_all_generators()
        
        return generators

    def _batch_generate_features(self, data: pd.DataFrame, 
                                generators: List[FeatureGenerator], **kwargs) -> Dict[str, pd.Series]:
        """Generate features in batch using centralized utilities."""
        results = {}
        
        try:
            # Group generators by type for batch processing
            unified_generators = [g for g in generators if hasattr(g, 'rolling_manager')]
            regular_generators = [g for g in generators if not hasattr(g, 'rolling_manager')]
            
            # Process unified generators in batch
            if unified_generators:
                batch_results = self._process_unified_generators_batch(data, unified_generators, **kwargs)
                results.update(batch_results)
                self.performance_stats['batch_operations'] += 1
            
            # Process regular generators individually
            for generator in regular_generators:
                try:
                    generator_results = self._generate_single_feature(data, generator, **kwargs)
                    results.update(generator_results)
                except Exception as e:
                    self.logger.warning(f"Failed to generate feature with {generator.__class__.__name__}: {e}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch feature generation failed: {e}")
            return {}

    def _process_unified_generators_batch(self, data: pd.DataFrame, 
                                         generators: List[FeatureGenerator], **kwargs) -> Dict[str, pd.Series]:
        """Process unified generators in batch for efficiency."""
        results = {}
        
        try:
            # Use common operations for batch processing
            if self.common_operations:
                # Create batch configuration
                batch_configs = []
                for generator in generators:
                    config = {
                        'name': generator.name,
                        'operation': 'technical_indicator',
                        'indicator': generator.name,
                        'params': getattr(generator, 'config', {}).__dict__ if hasattr(generator, 'config') else {}
                    }
                    batch_configs.append(config)
                
                # Process in batch
                batch_results = self.common_operations.batch_process_features(data, batch_configs)
                results.update(batch_results)
                self.performance_stats['common_operations_used'] += 1
            
            return results
            
        except Exception as e:
            self.logger.warning(f"Batch processing failed: {e}")
            return {}

    def _sequential_generate_features(self, data: pd.DataFrame, 
                                    generators: List[FeatureGenerator], **kwargs) -> Dict[str, pd.Series]:
        """Generate features sequentially using centralized utilities."""
        results = {}
        
        for generator in generators:
            try:
                generator_results = self._generate_single_feature(data, generator, **kwargs)
                results.update(generator_results)
            except Exception as e:
                self.logger.warning(f"Failed to generate feature with {generator.__class__.__name__}: {e}")
        
        return results

    def _generate_single_feature(self, data: pd.DataFrame, 
                                generator: FeatureGenerator, **kwargs) -> Dict[str, pd.Series]:
        """Generate a single feature using the generator."""
        try:
            # Use centralized utilities if available
            if hasattr(generator, 'rolling_manager') and self.rolling_manager:
                # Update generator's rolling manager reference
                generator.rolling_manager = self.rolling_manager
                self.performance_stats['centralized_rolling_operations'] += 1
            
            if hasattr(generator, 'scaler_factory') and self.scaler_factory:
                # Update generator's scaler factory reference
                generator.scaler_factory = self.scaler_factory
                self.performance_stats['centralized_scaling_operations'] += 1
            
            # Generate the feature
            result = generator.generate(data, **kwargs)
            
            if isinstance(result, FeatureResult):
                return {result.name: result.values}
            elif isinstance(result, pd.Series):
                return {generator.name: result}
            elif isinstance(result, dict):
                return result
            else:
                self.logger.warning(f"Unexpected result type from {generator.__class__.__name__}")
                return {}
                
        except Exception as e:
            self.logger.warning(f"Feature generation failed for {generator.__class__.__name__}: {e}")
            return {}

    def _apply_normalization(self, results: Dict[str, pd.Series], 
                           categories: Optional[List[FeatureCategory]] = None,
                           features: Optional[List[str]] = None) -> Dict[str, pd.Series]:
        """Apply normalization using centralized scaler factory."""
        if not self.scaler_factory:
            return results
        
        normalized_results = {}
        
        for feature_name, series in results.items():
            try:
                # Check if feature should be excluded
                if self._should_exclude_from_normalization(feature_name, categories, features):
                    normalized_results[feature_name] = series
                    continue
                
                # Determine feature type for appropriate scaler selection
                feature_type = self._determine_feature_type(feature_name)
                
                # Get appropriate scaler
                scaler = self.scaler_factory.get_scaler_for_feature_type(feature_type)
                
                # Apply normalization
                normalized_series = scaler.fit_transform(series)
                normalized_results[feature_name] = normalized_series
                
                self.performance_stats['normalization_applied'] += 1
                self.performance_stats['centralized_scaling_operations'] += 1
                
            except Exception as e:
                self.logger.warning(f"Normalization failed for {feature_name}: {e}")
                normalized_results[feature_name] = series
        
        return normalized_results

    def _should_exclude_from_normalization(self, feature_name: str, 
                                         categories: Optional[List[FeatureCategory]] = None,
                                         features: Optional[List[str]] = None) -> bool:
        """Check if feature should be excluded from normalization."""
        # Check feature name exclusions
        if feature_name in self.normalization_config['exclude_features']:
            return True
        
        # Check category exclusions (if we can determine the category)
        # This is a simplified check - in practice, you might want to track
        # which category each feature belongs to
        return False

    def _determine_feature_type(self, feature_name: str) -> str:
        """Determine feature type for appropriate scaler selection."""
        feature_name_lower = feature_name.lower()
        
        if any(keyword in feature_name_lower for keyword in ['price', 'close', 'open', 'high', 'low']):
            return 'price'
        elif any(keyword in feature_name_lower for keyword in ['volume', 'vol']):
            return 'volume'
        elif any(keyword in feature_name_lower for keyword in ['return', 'pct_change']):
            return 'returns'
        elif any(keyword in feature_name_lower for keyword in ['volatility', 'std', 'var']):
            return 'volatility'
        elif any(keyword in feature_name_lower for keyword in ['momentum', 'rsi', 'macd']):
            return 'momentum'
        elif any(keyword in feature_name_lower for keyword in ['trend', 'sma', 'ema']):
            return 'trend'
        elif any(keyword in feature_name_lower for keyword in ['oscillator', 'stoch']):
            return 'oscillator'
        else:
            return 'default'

    def _update_performance_stats(self, start_time: float, features_count: int, success: bool):
        """Update performance statistics."""
        generation_time = time.time() - start_time
        
        self.performance_stats['total_generations'] += 1
        if success:
            self.performance_stats['successful_generations'] += 1
            self.performance_stats['features_generated'] += features_count
        else:
            self.performance_stats['failed_generations'] += 1
        
        self.performance_stats['total_generation_time'] += generation_time
        self.performance_stats['average_generation_time'] = (
            self.performance_stats['total_generation_time'] / 
            max(self.performance_stats['total_generations'], 1)
        )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        
        # Add centralized utilities stats
        if self.rolling_manager:
            rolling_stats = self.rolling_manager.get_performance_stats()
            stats['rolling_manager_stats'] = rolling_stats
        
        if self.scaler_factory:
            scaler_stats = self.scaler_factory.get_performance_stats()
            stats['scaler_factory_stats'] = scaler_stats
        
        if self.common_operations:
            common_ops_stats = self.common_operations.get_performance_stats()
            stats['common_operations_stats'] = common_ops_stats
        
        return stats

    def reset_performance_stats(self):
        """Reset all performance statistics."""
        self.performance_stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'categories_used': set(),
            'features_generated': 0,
            'average_generation_time': 0.0,
            'total_generation_time': 0.0,
            'normalization_applied': 0,
            'matrix_accelerations': 0,
            'centralized_rolling_operations': 0,
            'centralized_scaling_operations': 0,
            'common_operations_used': 0,
            'batch_operations': 0,
            'memory_optimizations': 0
        }
        
        # Reset centralized utilities stats
        if self.rolling_manager:
            self.rolling_manager.reset_performance_stats()
        if self.scaler_factory:
            self.scaler_factory.reset_performance_stats()
        if self.common_operations:
            self.common_operations.reset_performance_stats()

    # Delegate other methods to the registry
    def get_generators_by_category(self, category: FeatureCategory) -> List[FeatureGenerator]:
        """Get generators by category."""
        return self.registry.get_generators_by_category(category)
    
    def get_generator_by_name(self, name: str) -> Optional[FeatureGenerator]:
        """Get generator by name."""
        return self.registry.get_generator_by_name(name)
    
    def get_all_generators(self) -> List[FeatureGenerator]:
        """Get all registered generators."""
        return self.registry.get_all_generators()
    
    def get_categories(self) -> List[FeatureCategory]:
        """Get all available categories."""
        return self.registry.get_categories()

# Global instance
_enhanced_feature_bank = None

def get_enhanced_feature_bank() -> EnhancedFeatureBank:
    """Get the global enhanced feature bank instance."""
    global _enhanced_feature_bank
    if _enhanced_feature_bank is None:
        _enhanced_feature_bank = EnhancedFeatureBank()
    return _enhanced_feature_bank