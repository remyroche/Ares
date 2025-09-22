"""
Feature Bank System

This module provides the FeatureBank class, which serves as the central registry
and management system for all feature generators. It allows scripts to easily
select and generate features by category, with support for lookback optimization
and matrix operations integration.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union, Set
from dataclasses import dataclass
import pandas as pd
import numpy as np

from .feature_generator import FeatureGenerator, FeatureCategory, FeatureResult, FeatureConfig
from .feature_registry import FeatureRegistry

logger = logging.getLogger(__name__)

@dataclass
class FeatureBankConfig:
    """Configuration for the feature bank."""
    enable_matrix_operations: bool = True
    enable_gpu_acceleration: bool = True
    enable_lookback_optimization: bool = True
    enable_parallel_processing: bool = True
    max_workers: int = 4
    chunk_size: int = 1000
    memory_efficient: bool = True
    cache_results: bool = True
    default_lookback: int = 20

class FeatureBank:
    """
    Central feature bank that manages all feature generators and provides
    a unified interface for feature generation by category.
    
    The FeatureBank serves as the single source of truth for feature generation,
    allowing scripts to easily select and generate features based on categories
    like returns, momentum, volume, support/resistance, etc.
    """
    
    def __init__(self, config: Optional[FeatureBankConfig] = None):
        """
        Initialize the feature bank.
        
        Args:
            config: Feature bank configuration
        """
        self.config = config or FeatureBankConfig()
        self.logger = logger.getChild('FeatureBank')
        
        # Initialize feature registry
        self.registry = FeatureRegistry()
        
        # Initialize matrix operations if enabled
        self.matrix_ops = None
        if self.config.enable_matrix_operations:
            try:
                from ...utils.matrix_operations import get_unified_matrix_operations
                self.matrix_ops = get_unified_matrix_operations()
                self.logger.info("✅ Matrix operations enabled")
            except ImportError:
                self.logger.warning("⚠️ Matrix operations not available")
        
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
            'total_generation_time': 0.0
        }
        
        # Cache for generated features
        self.feature_cache = {} if self.config.cache_results else None
        
        # Auto-register default generators
        self._auto_register_generators()

        self.logger.info("✅ FeatureBank initialized")
        self.logger.info(f"📊 Matrix ops: {self.config.enable_matrix_operations}, "
                        f"GPU: {self.config.enable_gpu_acceleration}, "
                        f"Lookback opt: {self.config.enable_lookback_optimization}")

    def _auto_register_generators(self) -> None:
        """
        Auto-register default feature generators from all categories.
        """
        try:
            from .feature_generator import FeatureCategory

            # List of categories to auto-register (excluding cross_timeframe, wavelet, candlestick, autoencoder, interaction, microstructure, time)
            categories_to_register = [
                FeatureCategory.MOMENTUM,
                FeatureCategory.VOLATILITY,
                FeatureCategory.TREND,
                FeatureCategory.VOLUME,
                FeatureCategory.SUPPORT_RESISTANCE,
                FeatureCategory.RETURNS,
                FeatureCategory.OSCILLATOR,
                FeatureCategory.CANDLESTICK_PATTERN,
                FeatureCategory.HMM_REGIME,
                FeatureCategory.ENTROPY,
                FeatureCategory.ORDER_FLOW,
                FeatureCategory.ACCELERATION
            ]

            registered_count = 0
            for category in categories_to_register:
                try:
                    generators = self._create_default_generators_for_category(category)
                    for generator in generators:
                        self.register_generator(generator)
                        registered_count += 1
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to register {category.value} generators: {e}")

            self.logger.info(f"✅ Auto-registered {registered_count} generators from {len(categories_to_register)} categories")

        except Exception as e:
            self.logger.warning(f"⚠️ Auto-registration failed: {e}")

    def _create_default_generators_for_category(self, category: FeatureCategory) -> List[FeatureGenerator]:
        """
        Create default generators for a given category using existing factory functions.
        """
        try:
            # Import all the factory functions from categories
            from ..categories import (
                create_acceleration_generators,
                create_interaction_generators,
                create_cross_timeframe_generators,
                create_entropy_generators,
                create_default_legacy_generators,
                create_default_time_generators
            )

            # Map categories to their creation functions
            category_creators = {
                FeatureCategory.MOMENTUM: self._create_momentum_generators,
                FeatureCategory.VOLATILITY: self._create_volatility_generators,
                FeatureCategory.TREND: self._create_trend_generators,
                FeatureCategory.VOLUME: self._create_volume_generators,
                FeatureCategory.SUPPORT_RESISTANCE: self._create_sr_generators,
                FeatureCategory.RETURNS: self._create_returns_generators,
                FeatureCategory.OSCILLATOR: self._create_oscillator_generators,
                FeatureCategory.CANDLESTICK_PATTERN: self._create_pattern_generators,
                FeatureCategory.HMM_REGIME: self._create_hmm_regime_generators,
                FeatureCategory.ENTROPY: self._create_entropy_generators,
                FeatureCategory.ORDER_FLOW: self._create_order_flow_generators,
                FeatureCategory.ACCELERATION: self._create_acceleration_generators
            }

            creator_func = category_creators.get(category)
            if creator_func:
                generators = creator_func()
                self.logger.info(f"✅ Created {len(generators)} generators for {category.value}")
                return generators
            else:
                self.logger.warning(f"⚠️ No creator function available for category: {category.value}")
                return []

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create generators for {category.value}: {e}")
            return []

    def _create_momentum_generators(self) -> List[FeatureGenerator]:
        """Create momentum-specific feature generators."""
        generators = []
        try:
            # First try to create advanced momentum generators
            from ..categories.momentum import create_default_momentum_generators
            advanced_generators = create_default_momentum_generators()
            generators.extend(advanced_generators)

            # Fallback to legacy generators if advanced ones fail
            if not generators:
                from ..categories.legacy import create_default_legacy_generators
                legacy_generators = create_default_legacy_generators()

                # Filter momentum-related generators from legacy set
                momentum_names = ['rsi', 'macd', 'stochastic', 'williams_r']
                for gen in legacy_generators:
                    if any(name in gen.config.name for name in momentum_names):
                        # Update the category to momentum
                        gen.config.category = FeatureCategory.MOMENTUM
                        generators.append(gen)

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create momentum generators: {e}")

        return generators

    def _create_volatility_generators(self) -> List[FeatureGenerator]:
        """Create volatility-specific feature generators."""
        generators = []
        try:
            # First try to create advanced volatility generators
            from ..categories.volatility import create_default_volatility_generators
            advanced_generators = create_default_volatility_generators()
            generators.extend(advanced_generators)

            # Fallback to legacy generators if advanced ones fail
            if not generators:
                from ..categories.legacy import create_default_legacy_generators
                legacy_generators = create_default_legacy_generators()

                # Filter volatility-related generators from legacy set
                volatility_names = ['bollinger', 'atr']
                for gen in legacy_generators:
                    if any(name in gen.config.name for name in volatility_names):
                        # Update the category to volatility
                        gen.config.category = FeatureCategory.VOLATILITY
                        generators.append(gen)

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create volatility generators: {e}")

        return generators

    def _create_trend_generators(self) -> List[FeatureGenerator]:
        """Create trend-specific feature generators."""
        generators = []
        try:
            # First try to create advanced trend generators
            from ..categories.trend import create_default_trend_generators
            advanced_generators = create_default_trend_generators()
            generators.extend(advanced_generators)

            # Fallback to legacy generators if advanced ones fail
            if not generators:
                from ..categories.legacy import create_default_legacy_generators
                legacy_generators = create_default_legacy_generators()

                # Filter trend-related generators from legacy set
                trend_names = ['sma', 'ema']
                for gen in legacy_generators:
                    if any(name in gen.config.name for name in trend_names):
                        # Update the category to trend
                        gen.config.category = FeatureCategory.TREND
                        generators.append(gen)

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create trend generators: {e}")

        return generators

    def _create_volume_generators(self) -> List[FeatureGenerator]:
        """Create volume-specific feature generators."""
        generators = []
        try:
            # First try to create advanced volume generators
            from ..categories.volume import create_default_volume_generators
            advanced_generators = create_default_volume_generators()
            generators.extend(advanced_generators)

            # Fallback to legacy generators if advanced ones fail
            if not generators:
                from ..categories.legacy import create_default_legacy_generators
                legacy_generators = create_default_legacy_generators()

                # Filter volume-related generators from legacy set
                volume_names = ['obv']
                for gen in legacy_generators:
                    if any(name in gen.config.name for name in volume_names):
                        # Update the category to volume
                        gen.config.category = FeatureCategory.VOLUME
                        generators.append(gen)

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create volume generators: {e}")

        return generators

    def _create_sr_generators(self) -> List[FeatureGenerator]:
        """Create support/resistance-specific feature generators."""
        generators = []
        try:
            # Try to create advanced support/resistance generators
            from ..categories.support_resistance import create_default_support_resistance_generators
            advanced_generators = create_default_support_resistance_generators()
            generators.extend(advanced_generators)

            # Fallback to legacy generators if advanced ones fail
            if not generators:
                from ..categories.legacy import create_default_legacy_generators
                legacy_generators = create_default_legacy_generators()

                # Filter support/resistance-related generators from legacy set
                sr_names = ['pivot']  # Add more as needed
                for gen in legacy_generators:
                    if any(name in gen.config.name for name in sr_names):
                        # Update the category to support_resistance
                        gen.config.category = FeatureCategory.SUPPORT_RESISTANCE
                        generators.append(gen)

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create support/resistance generators: {e}")

        return generators

    def _create_returns_generators(self) -> List[FeatureGenerator]:
        """Create returns-specific feature generators."""
        generators = []
        try:
            # Try to create advanced returns generators
            from ..categories.returns import create_default_returns_generators
            advanced_generators = create_default_returns_generators()
            generators.extend(advanced_generators)

            # Fallback to legacy generators if advanced ones fail
            if not generators:
                from ..categories.legacy import create_default_legacy_generators
                legacy_generators = create_default_legacy_generators()

                # Filter returns-related generators from legacy set
                returns_names = ['returns', 'log_returns']
                for gen in legacy_generators:
                    if any(name in gen.config.name for name in returns_names):
                        # Update the category to returns
                        gen.config.category = FeatureCategory.RETURNS
                        generators.append(gen)

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create returns generators: {e}")

        return generators

    def _create_oscillator_generators(self) -> List[FeatureGenerator]:
        """Create oscillator-specific feature generators."""
        generators = []
        try:
            # Try to create advanced oscillator generators
            from ..categories.oscillator import create_default_oscillator_generators
            advanced_generators = create_default_oscillator_generators()
            generators.extend(advanced_generators)

            # Fallback to legacy generators if advanced ones fail
            if not generators:
                from ..categories.legacy import create_default_legacy_generators
                legacy_generators = create_default_legacy_generators()

                # Filter oscillator-related generators from legacy set
                oscillator_names = ['cci', 'mfi']
                for gen in legacy_generators:
                    if any(name in gen.config.name for name in oscillator_names):
                        # Update the category to oscillator
                        gen.config.category = FeatureCategory.OSCILLATOR
                        generators.append(gen)

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create oscillator generators: {e}")

        return generators

    def _create_pattern_generators(self) -> List[FeatureGenerator]:
        """Create candlestick pattern-specific feature generators."""
        generators = []
        try:
            # Try to create advanced candlestick pattern generators
            # Note: This might not exist yet, so we'll handle the exception
            try:
                from ..categories.candlestick import create_default_candlestick_generators
                advanced_generators = create_default_candlestick_generators()
                generators.extend(advanced_generators)
            except ImportError:
                # Candlestick patterns might not be implemented yet
                pass

            # Fallback to legacy generators if advanced ones fail or don't exist
            if not generators:
                from ..categories.legacy import create_default_legacy_generators
                legacy_generators = create_default_legacy_generators()

                # Filter pattern-related generators from legacy set
                pattern_names = ['doji', 'hammer', 'pattern']
                for gen in legacy_generators:
                    if any(name in gen.config.name for name in pattern_names):
                        # Update the category to candlestick_pattern
                        gen.config.category = FeatureCategory.CANDLESTICK_PATTERN
                        generators.append(gen)

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create pattern generators: {e}")

        return generators

    def _create_hmm_regime_generators(self) -> List[FeatureGenerator]:
        """Create HMM regime-specific feature generators."""
        generators = []
        try:
            # Try to create advanced HMM regime generators
            from ..categories.hmm_regime import create_default_hmm_regime_generators
            advanced_generators = create_default_hmm_regime_generators()
            generators.extend(advanced_generators)

            # Try performance metrics generators
            from ..categories.hmm_performance_metrics import create_default_hmm_performance_metrics_generators
            try:
                performance_generators = create_default_hmm_performance_metrics_generators()
                generators.extend(performance_generators)
            except ImportError:
                pass

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create HMM regime generators: {e}")

        return generators

    def _create_entropy_generators(self) -> List[FeatureGenerator]:
        """Create entropy-specific feature generators."""
        generators = []
        try:
            from ..categories.entropy import create_default_entropy_generators
            advanced_generators = create_default_entropy_generators()
            generators.extend(advanced_generators)

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create entropy generators: {e}")

        return generators

    def _create_order_flow_generators(self) -> List[FeatureGenerator]:
        """Create order flow-specific feature generators."""
        generators = []
        try:
            from ..categories.order_flow import create_default_order_flow_generators
            advanced_generators = create_default_order_flow_generators()
            generators.extend(advanced_generators)

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create order flow generators: {e}")

        return generators

    def _create_acceleration_generators(self) -> List[FeatureGenerator]:
        """Create acceleration-specific feature generators."""
        generators = []
        try:
            from ..categories.acceleration import create_default_acceleration_generators
            advanced_generators = create_default_acceleration_generators()
            generators.extend(advanced_generators)

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create acceleration generators: {e}")

        return generators

    def register_generator(self, generator: FeatureGenerator) -> None:
        """
        Register a feature generator.
        
        Args:
            generator: Feature generator to register
        """
        self.registry.register(generator)
        self.logger.info(f"Registered generator: {generator.config.name} ({generator.config.category.value})")
    
    def register_generators(self, generators: List[FeatureGenerator]) -> None:
        """
        Register multiple feature generators.
        
        Args:
            generators: List of feature generators to register
        """
        for generator in generators:
            self.register_generator(generator)
    
    def get_generators_by_category(self, category: FeatureCategory) -> List[FeatureGenerator]:
        """
        Get all generators for a specific category.
        
        Args:
            category: Feature category
            
        Returns:
            List of generators for the category
        """
        return self.registry.get_by_category(category)
    
    def get_generator_by_name(self, name: str) -> Optional[FeatureGenerator]:
        """
        Get a generator by name.
        
        Args:
            name: Generator name
            
        Returns:
            Generator or None if not found
        """
        return self.registry.get_by_name(name)
    
    def list_categories(self) -> List[FeatureCategory]:
        """
        List all available categories.
        
        Returns:
            List of available categories
        """
        return self.registry.list_categories()
    
    def list_features(self, category: Optional[FeatureCategory] = None) -> List[str]:
        """
        List all available features.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of feature names
        """
        return self.registry.list_features(category)
    
    def generate_features(self, 
                         data: pd.DataFrame,
                         categories: Optional[List[Union[str, FeatureCategory]]] = None,
                         features: Optional[List[str]] = None,
                         lookback_optimization: bool = False,
                         target_column: Optional[str] = None,
                         **kwargs) -> pd.DataFrame:
        """
        Generate features by category or specific feature names.
        
        Args:
            data: Input data DataFrame
            categories: List of categories to generate features for
            features: List of specific feature names to generate
            lookback_optimization: Whether to optimize lookback periods
            target_column: Target column for lookback optimization
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with generated features
        """
        start_time = time.time()
        self.logger.info(f"🎯 Starting feature generation...")
        
        if data.empty:
            self.logger.warning("Empty data provided")
            return pd.DataFrame()
        
        # Determine which generators to use
        generators_to_use = self._select_generators(categories, features)
        
        if not generators_to_use:
            self.logger.warning("No generators selected")
            return pd.DataFrame()
        
        self.logger.info(f"📊 Selected {len(generators_to_use)} generators")
        
        # Optimize lookbacks if requested
        if lookback_optimization and target_column and self.lookback_optimizer:
            generators_to_use = self._optimize_lookbacks(generators_to_use, data, target_column)
        
        # Generate features
        results = self._generate_features_parallel(generators_to_use, data, **kwargs)
        
        # Combine results
        feature_df = self._combine_results(results, data.index)
        
        # Update performance stats
        generation_time = time.time() - start_time
        self._update_performance_stats(generation_time, len(results), categories)
        
        self.logger.info(f"✅ Feature generation completed in {generation_time:.3f}s")
        self.logger.info(f"📊 Generated {len(feature_df.columns)} features")
        
        return feature_df
    
    def generate_features_by_category(self, 
                                    data: pd.DataFrame,
                                    category: Union[str, FeatureCategory],
                                    **kwargs) -> pd.DataFrame:
        """
        Generate all features for a specific category.
        
        Args:
            data: Input data DataFrame
            category: Feature category
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with generated features
        """
        if isinstance(category, str):
            try:
                category = FeatureCategory(category)
            except ValueError:
                self.logger.error(f"Invalid category: {category}")
                return pd.DataFrame()
        
        return self.generate_features(data, categories=[category], **kwargs)
    
    def generate_specific_features(self, 
                                 data: pd.DataFrame,
                                 feature_names: List[str],
                                 **kwargs) -> pd.DataFrame:
        """
        Generate specific features by name.
        
        Args:
            data: Input data DataFrame
            feature_names: List of feature names to generate
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with generated features
        """
        return self.generate_features(data, features=feature_names, **kwargs)
    
    def _select_generators(self, 
                          categories: Optional[List[Union[str, FeatureCategory]]] = None,
                          features: Optional[List[str]] = None) -> List[FeatureGenerator]:
        """
        Select generators based on categories or feature names.
        
        Args:
            categories: List of categories
            features: List of feature names
            
        Returns:
            List of selected generators
        """
        generators = []
        
        if features:
            # Select by specific feature names
            for feature_name in features:
                generator = self.get_generator_by_name(feature_name)
                if generator:
                    generators.append(generator)
                else:
                    self.logger.warning(f"Generator not found: {feature_name}")
        
        elif categories:
            # Select by categories
            for category in categories:
                if isinstance(category, str):
                    try:
                        category = FeatureCategory(category)
                    except ValueError:
                        self.logger.warning(f"Invalid category: {category}")
                        continue
                
                category_generators = self.get_generators_by_category(category)
                generators.extend(category_generators)
        
        else:
            # Select all generators
            generators = self.registry.get_all()
        
        return generators
    
    def _optimize_lookbacks(self, 
                           generators: List[FeatureGenerator],
                           data: pd.DataFrame,
                           target_column: str) -> List[FeatureGenerator]:
        """
        Optimize lookback periods for generators that support it.
        
        Args:
            generators: List of generators
            data: Input data
            target_column: Target column for optimization
            
        Returns:
            List of generators with optimized lookbacks
        """
        if not self.lookback_optimizer:
            return generators
        
        self.logger.info("🔧 Optimizing lookback periods...")
        
        optimized_generators = []
        for generator in generators:
            if generator.supports_lookback_optimization():
                try:
                    # Optimize lookback for this generator
                    optimal_lookback = self.lookback_optimizer.optimize_lookback(
                        generator, data, target_column
                    )
                    
                    # Create a new generator with optimized lookback
                    optimized_config = generator.config
                    optimized_config.default_lookback = optimal_lookback
                    
                    # Create new generator instance (this is a simplified approach)
                    # In practice, you might want to modify the existing generator
                    optimized_generators.append(generator)
                    
                except Exception as e:
                    self.logger.warning(f"Lookback optimization failed for {generator.config.name}: {e}")
                    optimized_generators.append(generator)
            else:
                optimized_generators.append(generator)
        
        return optimized_generators
    
    def _generate_features_parallel(self, 
                                   generators: List[FeatureGenerator],
                                   data: pd.DataFrame,
                                   **kwargs) -> List[FeatureResult]:
        """
        Generate features using parallel processing if enabled.
        
        Args:
            generators: List of generators
            data: Input data
            **kwargs: Additional parameters
            
        Returns:
            List of feature results
        """
        if self.config.enable_parallel_processing and len(generators) > 1:
            return self._generate_features_parallel_impl(generators, data, **kwargs)
        else:
            return self._generate_features_sequential(generators, data, **kwargs)
    
    def _generate_features_sequential(self, 
                                    generators: List[FeatureGenerator],
                                    data: pd.DataFrame,
                                    **kwargs) -> List[FeatureResult]:
        """
        Generate features sequentially.
        
        Args:
            generators: List of generators
            data: Input data
            **kwargs: Additional parameters
            
        Returns:
            List of feature results
        """
        results = []
        
        for generator in generators:
            try:
                # Check cache first
                cache_key = self._get_cache_key(generator, data, **kwargs)
                if self.feature_cache and cache_key in self.feature_cache:
                    self.logger.debug(f"Using cached result for {generator.config.name}")
                    results.append(self.feature_cache[cache_key])
                    continue
                
                # Generate feature
                result = generator.generate(data, **kwargs)
                results.append(result)
                
                # Cache result
                if self.feature_cache:
                    self.feature_cache[cache_key] = result
                
            except Exception as e:
                self.logger.error(f"Error generating {generator.config.name}: {e}")
                # Create failed result
                failed_result = FeatureResult(
                    name=generator.config.name,
                    data=pd.Series(dtype=float, index=data.index),
                    config=generator.config,
                    computation_time=0.0,
                    success=False,
                    error_message=str(e)
                )
                results.append(failed_result)
        
        return results
    
    def _generate_features_parallel_impl(self, 
                                       generators: List[FeatureGenerator],
                                       data: pd.DataFrame,
                                       **kwargs) -> List[FeatureResult]:
        """
        Generate features using parallel processing.
        
        Args:
            generators: List of generators
            data: Input data
            **kwargs: Additional parameters
            
        Returns:
            List of feature results
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_generator = {
                executor.submit(self._generate_single_feature, generator, data, **kwargs): generator
                for generator in generators
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_generator):
                generator = future_to_generator[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error in parallel generation for {generator.config.name}: {e}")
                    # Create failed result
                    failed_result = FeatureResult(
                        name=generator.config.name,
                        data=pd.Series(dtype=float, index=data.index),
                        config=generator.config,
                        computation_time=0.0,
                        success=False,
                        error_message=str(e)
                    )
                    results.append(failed_result)
        
        return results
    
    def _generate_single_feature(self, 
                               generator: FeatureGenerator,
                               data: pd.DataFrame,
                               **kwargs) -> FeatureResult:
        """
        Generate a single feature (for parallel processing).
        
        Args:
            generator: Feature generator
            data: Input data
            **kwargs: Additional parameters
            
        Returns:
            Feature result
        """
        try:
            # Check cache first
            cache_key = self._get_cache_key(generator, data, **kwargs)
            if self.feature_cache and cache_key in self.feature_cache:
                return self.feature_cache[cache_key]
            
            # Generate feature
            result = generator.generate(data, **kwargs)
            
            # Cache result
            if self.feature_cache:
                self.feature_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            return FeatureResult(
                name=generator.config.name,
                data=pd.Series(dtype=float, index=data.index),
                config=generator.config,
                computation_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _combine_results(self, results: List[FeatureResult], index: pd.Index) -> pd.DataFrame:
        """
        Combine feature results into a single DataFrame.
        
        Args:
            results: List of feature results
            index: Index for the DataFrame
            
        Returns:
            Combined features DataFrame
        """
        feature_data = {}
        successful_features = 0
        
        for result in results:
            if result.success:
                feature_data[result.name] = result.data
                successful_features += 1
            else:
                self.logger.warning(f"Feature {result.name} failed: {result.error_message}")
        
        if not feature_data:
            self.logger.warning("No features were successfully generated")
            return pd.DataFrame(index=index)
        
        feature_df = pd.DataFrame(feature_data, index=index)
        self.logger.info(f"✅ Successfully generated {successful_features}/{len(results)} features")
        
        return feature_df
    
    def _get_cache_key(self, generator: FeatureGenerator, data: pd.DataFrame, **kwargs) -> str:
        """
        Generate cache key for a feature generation request.
        
        Args:
            generator: Feature generator
            data: Input data
            **kwargs: Additional parameters
            
        Returns:
            Cache key string
        """
        # Simple cache key based on generator name, data shape, and parameters
        data_hash = hash((data.shape, tuple(data.columns)))
        params_hash = hash(tuple(sorted(kwargs.items())))
        return f"{generator.config.name}_{data_hash}_{params_hash}"
    
    def _update_performance_stats(self, 
                                 generation_time: float,
                                 num_results: int,
                                 categories: Optional[List[Union[str, FeatureCategory]]] = None) -> None:
        """
        Update performance statistics.
        
        Args:
            generation_time: Time taken for generation
            num_results: Number of results
            categories: Categories used
        """
        self.performance_stats['total_generations'] += 1
        self.performance_stats['features_generated'] += num_results
        self.performance_stats['total_generation_time'] += generation_time
        self.performance_stats['average_generation_time'] = (
            self.performance_stats['total_generation_time'] / 
            self.performance_stats['total_generations']
        )
        
        if categories:
            for category in categories:
                if isinstance(category, str):
                    try:
                        category = FeatureCategory(category)
                    except ValueError:
                        continue
                self.performance_stats['categories_used'].add(category.value)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance statistics
        """
        stats = self.performance_stats.copy()
        stats['categories_used'] = list(stats['categories_used'])
        return stats
    
    def clear_cache(self) -> None:
        """Clear the feature cache."""
        if self.feature_cache:
            self.feature_cache.clear()
            self.logger.info("Feature cache cleared")
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """
        Get a summary of available features.
        
        Returns:
            Dictionary with feature summary
        """
        summary = {
            'total_generators': len(self.registry.get_all()),
            'categories': {},
            'features_by_category': {}
        }
        
        for category in self.list_categories():
            generators = self.get_generators_by_category(category)
            summary['categories'][category.value] = len(generators)
            summary['features_by_category'][category.value] = [
                gen.config.name for gen in generators
            ]
        
        return summary

# Global feature bank instance
_global_feature_bank: Optional[FeatureBank] = None

def get_global_feature_bank() -> FeatureBank:
    """
    Get the global feature bank instance.
    
    Returns:
        Global feature bank instance
    """
    global _global_feature_bank
    if _global_feature_bank is None:
        _global_feature_bank = FeatureBank()
    return _global_feature_bank

def set_global_feature_bank(bank: FeatureBank) -> None:
    """
    Set the global feature bank instance.
    
    Args:
        bank: Feature bank instance
    """
    global _global_feature_bank
    _global_feature_bank = bank