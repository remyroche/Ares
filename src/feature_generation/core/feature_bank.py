"""
Feature Bank System

This module provides the FeatureBank class, which serves as the central registry
and management system for all feature generators. It allows scripts to easily
select and generate features by category, with support for lookback optimization
and matrix operations integration.
"""

import copy
import logging
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np

from .feature_generator import FeatureGenerator, FeatureCategory, FeatureResult, FeatureConfig
from .auto_optimized_feature_generator import AutoOptimizedFeatureGenerator
from .auto_optimization_config import AutoOptimizationConfig, OptimizationLevel
from .generator_factory import GeneratorFactory
from .feature_registry import FeatureRegistry
from src.utils.unified_cache import UnifiedCache
from src.utils.tprint import tprint

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
    persist_generator_state: bool = True
    state_cache_dir: str = "data_cache/feature_states"
    state_cache_namespace: str = "feature_bank"
    state_cache_ttl_seconds: Optional[int] = None
    
    # Auto-optimization settings
    enable_auto_optimization: bool = True
    default_optimization_level: str = "balanced"  # "conservative", "balanced", "aggressive"
    auto_optimization_config: Optional[AutoOptimizationConfig] = None

class FeatureBank:
    """
    Central feature bank that manages all feature generators and provides
    a unified interface for feature generation by category.

    The FeatureBank serves as the single source of truth for feature generation,
    allowing scripts to easily select and generate features based on categories
    like returns, momentum, volume, support/resistance, etc.
    """

    VERSION = "2024.09"
    
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
        
        # Initialize generator factory
        self.generator_factory = GeneratorFactory()
        
        # Initialize auto-optimization configuration
        if self.config.auto_optimization_config is None:
            self.config.auto_optimization_config = AutoOptimizationConfig()
            self.config.auto_optimization_config.optimization_level = OptimizationLevel(self.config.default_optimization_level)
        
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
            'matrix_accelerations': 0
        }

        # Normalization configuration (use helper method for dict/Pydantic compatibility)
        self.auto_normalize = self._get_config_value('auto_normalize', True)
        self.normalization_method = self._get_config_value('normalization_method', 'zscore')
        self.normalization_config = {
            'exclude_categories': self._get_config_value('normalization_exclude_categories', []),
            'exclude_features': self._get_config_value('normalization_exclude_features', []),
            'rolling_windows': self._get_config_value('normalization_rolling_windows', [20, 50, 100])
        }
        
        # Cache for generated features
        self.feature_cache = {} if self._get_config_value('cache_results', True) else None

        # Persistent state cache for generator-level rolling state
        self.persist_generator_state = self._get_config_value('persist_generator_state', True)
        if self.persist_generator_state:
            state_cache_dir = self._get_config_value('state_cache_dir', 'data_cache/feature_states')
            state_cache_namespace = self._get_config_value('state_cache_namespace', 'feature_bank')
            state_cache_ttl = self._get_config_value('state_cache_ttl_seconds', None)
            self.state_cache = UnifiedCache(
                cache_dir=state_cache_dir,
                namespace=state_cache_namespace,
                default_ttl_seconds=state_cache_ttl,
                enable_disk=True,
                enable_compression=True
            )
        else:
            self.state_cache = None
        
        # Auto-register default generators
        tprint("🔧 Auto-registering feature generators...")
        self._auto_register_generators()

        # Set as global feature bank if no global instance exists
        global _global_feature_bank
        if _global_feature_bank is None:
            _global_feature_bank = self
            tprint("✅ FeatureBank set as global instance")

        self.logger.info("✅ FeatureBank initialized")
        self.logger.info(f"📊 Matrix ops: {self.config.enable_matrix_operations}, "
                        f"GPU: {self.config.enable_gpu_acceleration}, "
                        f"Lookback opt: {self.config.enable_lookback_optimization}")

    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """Helper to fetch values from dataclass or dict configs."""
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

    def _auto_register_generators(self) -> None:
        """
        Auto-register default feature generators from all categories.
        """
        tprint("🔧 Starting auto-registration of feature generators...")
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
            tprint(f"🚀 Initializing {total_categories} feature categories...")
            
            for i, category in enumerate(categories_to_register, 1):
                try:
                    tprint(f"🔧 Processing category {i}/{total_categories}: {category.value}")
                    generators = self._create_default_generators_for_category(category)
                    tprint(f"✅ Created {len(generators)} generators for {category.value}")
                    for generator in generators:
                        self.register_generator(generator)
                        registered_count += 1
                    tprint(f"📊 Progress: {i}/{total_categories} categories completed")
                except Exception as e:
                    tprint(f"⚠️ Failed to register {category.value} generators: {e}")
                    self.logger.warning(f"⚠️ Failed to register {category.value} generators: {e}")

            tprint(f"✅ Auto-registration completed. Registered {registered_count} generators")
            self.logger.info(f"✅ Auto-registered {registered_count} generators from {len(categories_to_register)} categories")

        except Exception as e:
            tprint(f"❌ Auto-registration failed: {e}")
            self.logger.warning(f"⚠️ Auto-registration failed: {e}")

    def _create_default_generators_for_category(self, category: FeatureCategory) -> List[FeatureGenerator]:
        """
        Create default generators for a given category using auto-optimized generators.
        """
        tprint(f"🔧 Creating auto-optimized generators for category: {category.value}")
        try:
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
                FeatureCategory.ENTROPY: self._create_entropy_generators,
                FeatureCategory.ORDER_FLOW: self._create_order_flow_generators,
                FeatureCategory.ACCELERATION: self._create_acceleration_generators,
                FeatureCategory.CROSS_TIMEFRAME: self._create_cross_timeframe_generators,
                FeatureCategory.AUTOENCODER: self._create_autoencoder_generators,
                FeatureCategory.INTERACTION: self._create_interaction_generators,
                FeatureCategory.MICROSTRUCTURE: self._create_microstructure_generators,
                FeatureCategory.REGIME: self._create_regime_generators,
                FeatureCategory.TIME: self._create_time_generators,
                FeatureCategory.NORMALIZATION: self._create_normalization_generators,
                FeatureCategory.REPRESENTATION_LEARNING: self._create_representation_learning_generators,
                FeatureCategory.ADVANCED_STATISTICAL: self._create_advanced_statistical_generators,
                FeatureCategory.SPECTRAL_WAVELET: self._create_spectral_wavelet_generators
            }

            creator_func = category_creators.get(category)
            if creator_func:
                tprint(f"🔧 Creating {category.value} features with auto-optimization...")
                generators = creator_func()
                
                # Convert generators to auto-optimized versions if auto-optimization is enabled
                if self.config.enable_auto_optimization:
                    auto_optimized_generators = []
                    for generator in generators:
                        auto_optimized_gen = self._convert_to_auto_optimized(generator)
                        auto_optimized_generators.append(auto_optimized_gen)
                    generators = auto_optimized_generators
                    tprint(f"✅ Created {len(generators)} auto-optimized generators for {category.value}")
                else:
                    tprint(f"✅ Created {len(generators)} generators for {category.value}")
                
                return generators
            else:
                tprint(f"⚠️ No creator function available for category: {category.value}")
                self.logger.warning(f"⚠️ No creator function available for category: {category.value}")
                return []

        except Exception as e:
            tprint(f"❌ Failed to create generators for {category.value}: {e}")
            self.logger.warning(f"⚠️ Failed to create generators for {category.value}: {e}")
            return []

    def _convert_to_auto_optimized(self, generator: FeatureGenerator) -> AutoOptimizedFeatureGenerator:
        """
        Convert a regular generator to an auto-optimized generator.
        
        Args:
            generator: Original generator
            
        Returns:
            Auto-optimized generator
        """
        try:
            tprint(f"🔄 Converting generator '{generator.config.name}' to auto-optimized...")
            
            # Create auto-optimized generator with same config
            tprint("📝 Creating auto-optimized generator with same config...")
            auto_optimized_gen = AutoOptimizedFeatureGenerator(
                config=generator.config,
                auto_optimization_config=self.config.auto_optimization_config
            )
            tprint("✅ Auto-optimized generator created")
            
            # Copy any additional state from original generator
            if hasattr(generator, 'get_state'):
                tprint("📦 Copying state from original generator...")
                state = generator.get_state()
                if state and hasattr(auto_optimized_gen, 'load_state'):
                    auto_optimized_gen.load_state(state)
                    tprint("✅ State copied successfully")
                else:
                    tprint("⚠️ No state to copy or load_state not available")
            else:
                tprint("⚠️ Original generator has no get_state method")
            
            tprint(f"✅ Generator '{generator.config.name}' converted to auto-optimized successfully")
            return auto_optimized_gen
            
        except Exception as e:
            tprint(f"❌ Failed to convert '{generator.config.name}' to auto-optimized: {e}")
            self.logger.warning(f"Failed to convert {generator.config.name} to auto-optimized: {e}")
            tprint("🔄 Returning original generator as fallback")
            # Return original generator if conversion fails
            return generator

    def _create_momentum_generators(self) -> List[FeatureGenerator]:
        """Create momentum-specific feature generators."""
        tprint("🔧 Creating momentum generators...")
        generators = []
        try:
            from ..categories.momentum import create_default_momentum_generators
            advanced_generators = create_default_momentum_generators()
            generators.extend(advanced_generators)
            tprint(f"✅ Created {len(advanced_generators)} momentum generators")
        except Exception as e:
            tprint(f"⚠️ Failed to create momentum generators: {e}")
            self.logger.warning(f"⚠️ Failed to create momentum generators: {e}")

        return generators

    def _create_volatility_generators(self) -> List[FeatureGenerator]:
        """Create volatility-specific feature generators."""
        tprint("🔧 Creating volatility generators...")
        generators = []
        try:
            from ..categories.volatility import create_default_volatility_generators
            advanced_generators = create_default_volatility_generators()
            generators.extend(advanced_generators)
            tprint(f"✅ Created {len(advanced_generators)} volatility generators")
        except Exception as e:
            tprint(f"⚠️ Failed to create volatility generators: {e}")
            self.logger.warning(f"⚠️ Failed to create volatility generators: {e}")

        return generators

    def _create_trend_generators(self) -> List[FeatureGenerator]:
        """Create trend-specific feature generators."""
        tprint("🔧 Creating trend generators...")
        generators = []
        try:
            from ..categories.trend import create_default_trend_generators
            advanced_generators = create_default_trend_generators()
            generators.extend(advanced_generators)
            tprint(f"✅ Created {len(advanced_generators)} trend generators")
        except Exception as e:
            tprint(f"⚠️ Failed to create trend generators: {e}")
            self.logger.warning(f"⚠️ Failed to create trend generators: {e}")

        return generators

    def _create_volume_generators(self) -> List[FeatureGenerator]:
        """Create volume-specific feature generators."""
        tprint("🔧 Creating volume generators...")
        generators = []
        try:
            from ..categories.volume import create_default_volume_generators
            advanced_generators = create_default_volume_generators()
            generators.extend(advanced_generators)
            tprint(f"✅ Created {len(advanced_generators)} volume generators")
        except Exception as e:
            tprint(f"⚠️ Failed to create volume generators: {e}")
            self.logger.warning(f"⚠️ Failed to create volume generators: {e}")

        return generators

    def _create_sr_generators(self) -> List[FeatureGenerator]:
        """Create support/resistance-specific feature generators."""
        tprint("🔧 Creating support/resistance generators...")
        generators = []
        try:
            from ..categories.support_resistance import create_default_support_resistance_generators
            advanced_generators = create_default_support_resistance_generators()
            generators.extend(advanced_generators)
            tprint(f"✅ Created {len(advanced_generators)} support/resistance generators")
        except Exception as e:
            tprint(f"⚠️ Failed to create support/resistance generators: {e}")
            self.logger.warning(f"⚠️ Failed to create support/resistance generators: {e}")

        return generators

    def _create_returns_generators(self) -> List[FeatureGenerator]:
        """Create returns-specific feature generators."""
        tprint("🔧 Creating returns generators...")
        generators = []
        try:
            from ..categories.returns import create_default_returns_generators
            advanced_generators = create_default_returns_generators()
            generators.extend(advanced_generators)
            tprint(f"✅ Created {len(advanced_generators)} returns generators")
        except Exception as e:
            tprint(f"⚠️ Failed to create returns generators: {e}")
            self.logger.warning(f"⚠️ Failed to create returns generators: {e}")

        return generators

    def _create_oscillator_generators(self) -> List[FeatureGenerator]:
        """Create oscillator-specific feature generators."""
        tprint("🔧 Creating oscillator generators...")
        generators = []
        try:
            from ..categories.oscillator import create_default_oscillator_generators
            advanced_generators = create_default_oscillator_generators()
            generators.extend(advanced_generators)
            tprint(f"✅ Created {len(advanced_generators)} oscillator generators")
        except Exception as e:
            tprint(f"⚠️ Failed to create oscillator generators: {e}")
            self.logger.warning(f"⚠️ Failed to create oscillator generators: {e}")

        return generators

    def _create_pattern_generators(self) -> List[FeatureGenerator]:
        """Create candlestick pattern-specific feature generators."""
        tprint("🔧 Creating candlestick pattern generators...")
        generators = []
        try:
            from ..categories.candlestick_pattern import create_default_candlestick_pattern_generators
            advanced_generators = create_default_candlestick_pattern_generators()
            generators.extend(advanced_generators)
            tprint(f"✅ Created {len(advanced_generators)} candlestick pattern generators")
        except Exception as e:
            tprint(f"⚠️ Failed to create candlestick pattern generators: {e}")
            self.logger.warning(f"⚠️ Failed to create candlestick pattern generators: {e}")

        return generators


    def _create_entropy_generators(self) -> List[FeatureGenerator]:
        """Create entropy-specific feature generators."""
        tprint("🔧 Creating entropy generators...")
        generators = []
        try:
            from ..categories.entropy import create_default_entropy_generators
            advanced_generators = create_default_entropy_generators()
            generators.extend(advanced_generators)
            tprint(f"✅ Created {len(advanced_generators)} entropy generators")
        except Exception as e:
            tprint(f"⚠️ Failed to create entropy generators: {e}")
            self.logger.warning(f"⚠️ Failed to create entropy generators: {e}")

        return generators

    def _create_order_flow_generators(self) -> List[FeatureGenerator]:
        """Create order flow-specific feature generators."""
        tprint("🔧 Creating order flow generators...")
        generators = []
        try:
            from ..categories.order_flow import create_default_order_flow_generators
            advanced_generators = create_default_order_flow_generators()
            generators.extend(advanced_generators)
            tprint(f"✅ Created {len(advanced_generators)} order flow generators")
        except Exception as e:
            tprint(f"⚠️ Failed to create order flow generators: {e}")
            self.logger.warning(f"⚠️ Failed to create order flow generators: {e}")

        return generators

    def _create_acceleration_generators(self) -> List[FeatureGenerator]:
        """Create acceleration-specific feature generators."""
        tprint("🔧 Creating acceleration generators...")
        generators = []
        try:
            from ..categories.acceleration import create_default_acceleration_generators
            advanced_generators = create_default_acceleration_generators()
            generators.extend(advanced_generators)
            tprint(f"✅ Created {len(advanced_generators)} acceleration generators")
        except Exception as e:
            tprint(f"⚠️ Failed to create acceleration generators: {e}")
            self.logger.warning(f"⚠️ Failed to create acceleration generators: {e}")

        return generators

    def _create_cross_timeframe_generators(self) -> List[FeatureGenerator]:
        """Create cross-timeframe-specific feature generators."""
        generators = []
        try:
            from ..categories.cross_timeframe import create_default_cross_timeframe_generators
            advanced_generators = create_default_cross_timeframe_generators()
            generators.extend(advanced_generators)

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create cross-timeframe generators: {e}")

        return generators

    def _create_autoencoder_generators(self) -> List[FeatureGenerator]:
        """Create autoencoder-specific feature generators."""
        tprint("🔧 Creating autoencoder generators...")
        generators = []
        try:
            from ..categories.autoencoder import create_default_autoencoder_generators
            advanced_generators = create_default_autoencoder_generators()
            generators.extend(advanced_generators)
            tprint(f"✅ Created {len(advanced_generators)} autoencoder generators")
        except Exception as e:
            tprint(f"⚠️ Failed to create autoencoder generators: {e}")
            self.logger.warning(f"⚠️ Failed to create autoencoder generators: {e}")

        return generators

    def _create_interaction_generators(self) -> List[FeatureGenerator]:
        """Create interaction-specific feature generators."""
        tprint("🔧 Creating interaction generators...")
        generators = []
        try:
            from ..categories.interaction import create_default_interaction_generators
            advanced_generators = create_default_interaction_generators()
            generators.extend(advanced_generators)
            tprint(f"✅ Created {len(advanced_generators)} interaction generators")
        except Exception as e:
            tprint(f"⚠️ Failed to create interaction generators: {e}")
            self.logger.warning(f"⚠️ Failed to create interaction generators: {e}")

        return generators

    def _create_microstructure_generators(self) -> List[FeatureGenerator]:
        """Create microstructure-specific feature generators."""
        tprint("🔧 Creating microstructure generators...")
        generators = []
        try:
            from ..categories.microstructure import create_default_microstructure_generators
            advanced_generators = create_default_microstructure_generators()
            generators.extend(advanced_generators)
            tprint(f"✅ Created {len(advanced_generators)} microstructure generators")
        except Exception as e:
            tprint(f"⚠️ Failed to create microstructure generators: {e}")
            self.logger.warning(f"⚠️ Failed to create microstructure generators: {e}")

        return generators

    def _create_regime_generators(self) -> List[FeatureGenerator]:
        """Create regime-specific feature generators."""
        generators = []
        try:
            # Try to create regime volatility generators
            try:
                from ..categories.regime_volatility import RegimeVolatilityFeatureGenerator
                # Create a regime volatility generator (single instance for now)
                generators.append(RegimeVolatilityFeatureGenerator())
            except ImportError:
                pass

            # Try to create regime statistical generators
            try:
                from ..categories.regime_statistical import RegimeStatisticalFeatureGenerator
                generators.append(RegimeStatisticalFeatureGenerator())
            except ImportError:
                pass

            # Regime feature integration is now part of regime_features.py
            # No separate import needed

            # Try to create regime structural trend generators
            try:
                from ..categories.regime_structural_trend import RegimeStructuralTrendGenerator
                generators.append(RegimeStructuralTrendGenerator())
            except ImportError:
                pass

            # Try to create regime volume generators
            try:
                from ..categories.regime_volume import RegimeVolumeFeatureGenerator
                generators.append(RegimeVolumeFeatureGenerator())
            except ImportError:
                pass

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create regime generators: {e}")

        return generators

    def _create_time_generators(self) -> List[FeatureGenerator]:
        """Create time-specific feature generators."""
        tprint("🔧 Creating time generators...")
        generators = []
        try:
            from ..categories.time import create_default_time_generators
            advanced_generators = create_default_time_generators()
            generators.extend(advanced_generators)
            tprint(f"✅ Created {len(advanced_generators)} time generators")
        except Exception as e:
            tprint(f"⚠️ Failed to create time generators: {e}")
            self.logger.warning(f"⚠️ Failed to create time generators: {e}")

        return generators

    def _create_normalization_generators(self) -> List[FeatureGenerator]:
        """Create normalization-specific feature generators."""
        generators = []
        # Normalization features are now handled by individual feature generators
        # No separate normalization generators needed
        return generators

    def _create_representation_learning_generators(self) -> List[FeatureGenerator]:
        """Create representation learning feature generators."""
        generators = []
        try:
            # Try enhanced representation learning generators first
            try:
                from ..categories.enhanced_representation_learning import create_enhanced_representation_learning_generators
                enhanced_generators = create_enhanced_representation_learning_generators()
                generators.extend(enhanced_generators)
                self.logger.info(f"✅ Added {len(enhanced_generators)} enhanced representation learning generators")
            except ImportError:
                self.logger.debug("ℹ️ Enhanced representation learning generators not available (optional)")

            # Fallback to standard representation learning generators
            try:
                from ..categories.representation_learning import create_default_representation_learning_generators
                standard_generators = create_default_representation_learning_generators()
                generators.extend(standard_generators)
            except ImportError:
                pass

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create representation learning generators: {e}")

        return generators

    def _create_advanced_statistical_generators(self) -> List[FeatureGenerator]:
        """Create advanced statistical feature generators."""
        generators = []
        try:
            from ..categories.advanced_statistical import create_default_advanced_statistical_generators
            advanced_generators = create_default_advanced_statistical_generators()
            generators.extend(advanced_generators)
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create advanced statistical generators: {e}")
        return generators

    def _create_spectral_wavelet_generators(self) -> List[FeatureGenerator]:
        """Create spectral/wavelet feature generators."""
        generators = []
        try:
            from ..categories.spectral_wavelet import create_default_spectral_wavelet_generators
            spectral_generators = create_default_spectral_wavelet_generators()
            generators.extend(spectral_generators)
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create spectral/wavelet generators: {e}")
        return generators

    def register_generator(self, generator: FeatureGenerator) -> None:
        """
        Register a feature generator.
        
        Args:
            generator: Feature generator to register
        """
        tprint(f"📝 Registering generator: {generator.config.name}")
        self.registry.register(generator)
        tprint(f"✅ Successfully registered generator: {generator.config.name}")
        self.logger.debug(f"Registered generator: {generator.config.name} ({generator.config.category.value})")
    
    def register_generators(self, generators: List[FeatureGenerator]) -> None:
        """
        Register multiple feature generators.
        
        Args:
            generators: List of feature generators to register
        """
        tprint(f"📝 Registering {len(generators)} generators...")
        for generator in generators:
            self.register_generator(generator)
        tprint(f"✅ Successfully registered {len(generators)} generators")
    
    def get_generators_by_category(self, category: FeatureCategory) -> List[FeatureGenerator]:
        """
        Get all generators for a specific category.
        
        Args:
            category: Feature category
            
        Returns:
            List of generators for the category
        """
        tprint(f"🔍 Looking up generators for category: {category.value}")
        generators = self.registry.get_by_category(category)
        tprint(f"✅ Found {len(generators)} generators for category: {category.value}")
        return generators
    
    def get_generator_by_name(self, name: str) -> Optional[FeatureGenerator]:
        """
        Get a generator by name.
        
        Args:
            name: Generator name
            
        Returns:
            Generator or None if not found
        """
        tprint(f"🔍 Looking up generator: {name}")
        generator = self.registry.get_by_name(name)
        if generator:
            tprint(f"✅ Found generator: {name}")
        else:
            tprint(f"❌ Generator not found: {name}")
        return generator
    
    def list_categories(self) -> List[FeatureCategory]:
        """
        List all available categories.
        
        Returns:
            List of available categories
        """
        tprint("🔍 Listing available categories...")
        categories = self.registry.list_categories()
        tprint(f"✅ Found {len(categories)} categories")
        return categories
    
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
                         use_optimized_pipeline: bool = True,
                         **kwargs) -> pd.DataFrame:
        """
        Generate features by category or specific feature names.
        
        Args:
            data: Input data DataFrame
            categories: List of categories to generate features for
            features: List of specific feature names to generate
            lookback_optimization: Whether to optimize lookback periods
            target_column: Target column for lookback optimization
            use_optimized_pipeline: Whether to use the optimized pipeline
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with generated features
        """
        start_time = time.time()
        tprint("🚀 Starting feature generation...")
        self.logger.info(f"🎯 Starting feature generation...")
        
        if data.empty:
            tprint("⚠️ Empty data provided")
            self.logger.warning("Empty data provided")
            return pd.DataFrame()
        
        # Use optimized pipeline if requested and available
        if use_optimized_pipeline:
            try:
                from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
                pipeline = get_optimized_feature_pipeline()
                
                # Convert categories to strings if needed
                category_strings = []
                if categories:
                    for cat in categories:
                        if isinstance(cat, FeatureCategory):
                            category_strings.append(cat.value)
                        else:
                            category_strings.append(cat)
                
                result = pipeline.process_features(
                    data=data,
                    categories=category_strings if category_strings else None,
                    features=features,
                    target_column=target_column,
                    **kwargs
                )
                
                if result.success:
                    tprint(f"✅ Optimized pipeline completed in {result.processing_time:.3f}s")
                    tprint(f"📊 Generated {len(result.features.columns)} features")
                    self.logger.info(f"✅ Optimized pipeline completed in {result.processing_time:.3f}s")
                    self.logger.info(f"📊 Generated {len(result.features.columns)} features")
                    return result.features
                else:
                    tprint(f"⚠️ Optimized pipeline failed: {result.error_message}")
                    self.logger.warning(f"Optimized pipeline failed: {result.error_message}")
                    # Fall back to standard generation
            except ImportError:
                tprint("⚠️ Optimized pipeline not available, using standard generation")
                self.logger.warning("Optimized pipeline not available, using standard generation")
            except Exception as e:
                tprint(f"⚠️ Optimized pipeline error: {e}, using standard generation")
                self.logger.warning(f"Optimized pipeline error: {e}, using standard generation")
        
        # Standard feature generation (fallback)
        tprint("🔧 Using standard feature generation...")
        # Determine which generators to use
        generators_to_use = self._select_generators(categories, features)
        
        if not generators_to_use:
            tprint("⚠️ No generators selected")
            self.logger.warning("No generators selected")
            return pd.DataFrame()
        
        tprint(f"📊 Selected {len(generators_to_use)} generators")
        self.logger.info(f"📊 Selected {len(generators_to_use)} generators")
        
        # Optimize lookbacks if requested
        if lookback_optimization and target_column and self.lookback_optimizer:
            generators_to_use = self._optimize_lookbacks(generators_to_use, data, target_column)
        
        # Generate features
        results = self._generate_features_parallel(generators_to_use, data, **kwargs)
        
        # Combine results
        feature_df = self._combine_results(results, data.index)

        # Apply automatic normalization if enabled
        if self.auto_normalize and not feature_df.empty:
            feature_df = self._apply_automatic_normalization(feature_df, categories)

        # Update performance stats
        generation_time = time.time() - start_time
        self._update_performance_stats(generation_time, len(results), categories)

        self.logger.info(f"✅ Feature generation completed in {generation_time:.3f}s")
        self.logger.info(f"📊 Generated {len(feature_df.columns)} features")

        return feature_df

    def _apply_automatic_normalization(self, feature_df: pd.DataFrame,
                                     categories: Optional[List[Union[str, FeatureCategory]]] = None) -> pd.DataFrame:
        """
        Apply automatic normalization to generated features.

        Args:
            feature_df: DataFrame with generated features
            categories: Categories that were generated (for exclusion logic)

        Returns:
            Normalized feature DataFrame
        """
        if feature_df.empty:
            return feature_df

        normalized_df = feature_df.copy()

        try:
            # Determine which features to normalize
            features_to_normalize = self._select_features_for_normalization(feature_df, categories)

            if not features_to_normalize:
                self.logger.debug("No features selected for normalization")
                return normalized_df

            self.logger.info(f"🔧 Applying {self.normalization_method} normalization to {len(features_to_normalize)} features")

            # Apply normalization using matrix accelerator if available
            if self.matrix_accelerator:
                transformations = [{
                    'type': self.normalization_method,
                    'params': {'columns': features_to_normalize}
                }]

                normalized_df = self.matrix_accelerator.vectorized_feature_transformations(
                    normalized_df, transformations
                )

                self.performance_stats['normalization_applied'] += 1
                self.performance_stats['matrix_accelerations'] += 1

            else:
                # Fallback to manual normalization
                for feature in features_to_normalize:
                    if feature in normalized_df.columns:
                        if self.normalization_method == 'zscore':
                            mean_val = normalized_df[feature].mean()
                            std_val = normalized_df[feature].std()
                            if std_val > 0:
                                normalized_df[feature] = (normalized_df[feature] - mean_val) / std_val

                        elif self.normalization_method == 'minmax':
                            min_val = normalized_df[feature].min()
                            max_val = normalized_df[feature].max()
                            if max_val > min_val:
                                normalized_df[feature] = (normalized_df[feature] - min_val) / (max_val - min_val)

                        elif self.normalization_method == 'robust':
                            median_val = normalized_df[feature].median()
                            mad_val = (normalized_df[feature] - median_val).abs().median()
                            if mad_val > 0:
                                normalized_df[feature] = (normalized_df[feature] - median_val) / mad_val

                self.performance_stats['normalization_applied'] += 1

            self.logger.info(f"✅ Normalization applied to {len(features_to_normalize)} features")

        except Exception as e:
            self.logger.error(f"Error applying automatic normalization: {e}")
            # Return original dataframe if normalization fails

        return normalized_df

    def _select_features_for_normalization(self, feature_df: pd.DataFrame,
                                         categories: Optional[List[Union[str, FeatureCategory]]] = None) -> List[str]:
        """Select which features should be normalized."""
        features_to_normalize = []

        # Get numeric columns
        numeric_columns = feature_df.select_dtypes(include=[np.number]).columns.tolist()

        for feature in numeric_columns:
            # Skip excluded features
            if feature in self.normalization_config['exclude_features']:
                continue

            # Skip features from excluded categories if categories are specified
            if categories:
                feature_category = self._get_feature_category(feature)
                if feature_category and str(feature_category) in self.normalization_config['exclude_categories']:
                    continue

            # Only normalize features that are not already normalized or bounded
            if not self._is_already_normalized(feature):
                features_to_normalize.append(feature)

        return features_to_normalize

    def _get_feature_category(self, feature_name: str) -> Optional[FeatureCategory]:
        """Get the category of a feature based on its name."""
        # This is a simple heuristic - in practice, we'd maintain a registry
        if 'zscore' in feature_name.lower() or 'normalized' in feature_name.lower():
            return FeatureCategory.NORMALIZATION
        elif 'rsi' in feature_name.lower() or 'momentum' in feature_name.lower():
            return FeatureCategory.MOMENTUM
        elif 'volume' in feature_name.lower():
            return FeatureCategory.VOLUME
        elif 'volatility' in feature_name.lower() or 'atr' in feature_name.lower():
            return FeatureCategory.VOLATILITY
        elif 'trend' in feature_name.lower() or 'ma_' in feature_name.lower():
            return FeatureCategory.TREND
        else:
            return None

    def _is_already_normalized(self, feature_name: str) -> bool:
        """Check if a feature is already normalized or bounded."""
        # Features that are already bounded or normalized
        normalized_indicators = [
            'rsi', 'stoch', 'williams', 'macd_hist', 'bb_percent',
            'adx', 'cci', 'momentum', 'roc', 'zscore', 'normalized'
        ]

        return any(indicator in feature_name.lower() for indicator in normalized_indicators)

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
                    # Skip problematic generators that are already excluded from lookback optimization
                    if self._should_exclude_generator(generator):
                        continue
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

                # Skip problematic categories that are already excluded from lookback optimization
                if self._should_exclude_category(category):
                    continue

                category_generators = self.get_generators_by_category(category)
                generators.extend(category_generators)
        
        else:
            # Select all generators but filter out problematic ones
            all_generators = self.registry.get_all()
            generators = [gen for gen in all_generators if not self._should_exclude_generator(gen)]
        
        return generators

    def _should_exclude_generator(self, generator: FeatureGenerator) -> bool:
        """
        Check if a generator should be excluded from feature generation.
        These are generators that are already excluded from lookback optimization
        and cause technical issues.

        Args:
            generator: The generator to check

        Returns:
            True if generator should be excluded
        """
        generator_name = generator.config.name.lower()

        # Exclude autoencoder generators (technical issues)
        if 'autoencoder' in generator_name:
            return True

        # Exclude cross-timeframe generators (complexity issues)
        if 'cross_timeframe' in generator_name:
            return True

        # Exclude bid-ask generators (missing data)
        if 'bid_ask' in generator_name or 'bidask' in generator_name:
            return True

        # Exclude interaction generators (complexity)
        if 'interaction' in generator_name:
            return True

        # Exclude regime-specific generators (context-dependent)
        if 'regime_' in generator_name:
            return True

        return False

    def _should_exclude_category(self, category: FeatureCategory) -> bool:
        """
        Check if a category should be excluded from feature generation.

        Args:
            category: The category to check

        Returns:
            True if category should be excluded
        """
        # Exclude autoencoder category (technical issues)
        if category == FeatureCategory.AUTOENCODER:
            return True

        # Exclude cross-timeframe category (complexity issues)
        if category == FeatureCategory.CROSS_TIMEFRAME:
            return True

        # Exclude interaction category (complexity)
        if category == FeatureCategory.INTERACTION:
            return True

        return False

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
                    cached_result = self.feature_cache[cache_key]
                    results.append(cached_result)
                    if self.persist_generator_state:
                        self._store_generator_state(generator, self._extract_state_from_result(generator, cached_result))
                    continue

                state_payload = self._load_generator_state(generator)
                if state_payload:
                    generator.load_state(state_payload)

                # Generate feature
                result = generator.generate(data, **kwargs)
                results.append(result)

                # Cache result
                if self.feature_cache:
                    self.feature_cache[cache_key] = result

                if self.persist_generator_state:
                    self._store_generator_state(generator, self._extract_state_from_result(generator, result))
                
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
                cached_result = self.feature_cache[cache_key]
                if self.persist_generator_state:
                    self._store_generator_state(generator, self._extract_state_from_result(generator, cached_result))
                return cached_result

            state_payload = self._load_generator_state(generator)
            if state_payload:
                generator.load_state(state_payload)

            # Generate feature
            result = generator.generate(data, **kwargs)

            # Cache result
            if self.feature_cache:
                self.feature_cache[cache_key] = result

            if self.persist_generator_state:
                self._store_generator_state(generator, self._extract_state_from_result(generator, result))

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

    def _extract_state_from_result(self, generator: FeatureGenerator, result: FeatureResult) -> Optional[Dict[str, Any]]:
        if result and result.metadata:
            state_payload = result.metadata.get('state')
            if state_payload is not None:
                return copy.deepcopy(state_payload)
        # Fallback to generator's current state snapshot
        current_state = generator.get_state()
        return copy.deepcopy(current_state) if current_state else None

    def _state_cache_key(self, generator: FeatureGenerator) -> str:
        return f"{generator.__class__.__name__}:{generator.config.name}"

    def _load_generator_state(self, generator: FeatureGenerator) -> Optional[Dict[str, Any]]:
        if not self.state_cache:
            return None
        cache_key = self._state_cache_key(generator)
        cached_state = self.state_cache.get(cache_key)
        if cached_state is None:
            return None
        return copy.deepcopy(cached_state)

    def _store_generator_state(self, generator: FeatureGenerator, state: Optional[Dict[str, Any]]) -> None:
        if not self.state_cache or state is None:
            return
        cache_key = self._state_cache_key(generator)
        self.state_cache.set(cache_key, copy.deepcopy(state), persist=True)

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
            'features_by_category': {},
            'auto_optimization_enabled': self.config.enable_auto_optimization,
            'optimization_level': self.config.default_optimization_level
        }
        
        for category in self.list_categories():
            generators = self.get_generators_by_category(category)
            summary['categories'][category.value] = len(generators)
            summary['features_by_category'][category.value] = [
                gen.config.name for gen in generators
            ]
        
        return summary
    
    def create_auto_optimized_generator(self, name: str, category: FeatureCategory,
                                      required_columns: List[str], 
                                      optimization_level: Optional[str] = None,
                                      **kwargs) -> Optional[AutoOptimizedFeatureGenerator]:
        """
        Create an auto-optimized generator using the factory.
        
        Args:
            name: Generator name
            category: Feature category
            required_columns: Required input columns
            optimization_level: Optimization level (uses default if None)
            **kwargs: Additional parameters
            
        Returns:
            Auto-optimized generator or None if creation fails
        """
        try:
            tprint(f"🔧 Creating auto-optimized generator via FeatureBank: {name}")
            tprint(f"📊 Category: {category.value}")
            
            if optimization_level is None:
                optimization_level = self.config.default_optimization_level
                tprint(f"📊 Using default optimization level: {optimization_level}")
            else:
                tprint(f"📊 Using specified optimization level: {optimization_level}")
            
            tprint("🚀 Delegating to GeneratorFactory...")
            generator = self.generator_factory.create_auto_optimized_generator(
                name=name,
                category=category,
                required_columns=required_columns,
                optimization_level=optimization_level,
                auto_optimization_config=self.config.auto_optimization_config,
                **kwargs
            )
            
            if generator:
                tprint(f"✅ Auto-optimized generator '{name}' created successfully via FeatureBank")
            else:
                tprint(f"❌ Failed to create auto-optimized generator '{name}' via FeatureBank")
            
            return generator
            
        except Exception as e:
            tprint(f"❌ Error creating auto-optimized generator '{name}' via FeatureBank: {e}")
            self.logger.error(f"Error creating auto-optimized generator '{name}': {e}")
            return None
    
    def create_auto_optimized_generators_by_category(self, category: FeatureCategory,
                                                   optimization_level: Optional[str] = None,
                                                   **kwargs) -> List[AutoOptimizedFeatureGenerator]:
        """
        Create auto-optimized generators for a specific category.
        
        Args:
            category: Feature category
            optimization_level: Optimization level (uses default if None)
            **kwargs: Additional parameters
            
        Returns:
            List of auto-optimized generators
        """
        if optimization_level is None:
            optimization_level = self.config.default_optimization_level
        
        # Get existing generators for the category
        existing_generators = self.get_generators_by_category(category)
        
        # Convert to auto-optimized versions
        auto_optimized_generators = []
        for generator in existing_generators:
            auto_optimized_gen = self._convert_to_auto_optimized(generator)
            auto_optimized_generators.append(auto_optimized_gen)
        
        return auto_optimized_generators
    
    def set_optimization_level(self, level: str) -> None:
        """
        Set the default optimization level for all generators.
        
        Args:
            level: Optimization level ("conservative", "balanced", "aggressive")
        """
        try:
            self.config.default_optimization_level = level
            self.config.auto_optimization_config.optimization_level = OptimizationLevel(level)
            self.logger.info(f"Optimization level set to: {level}")
        except ValueError:
            self.logger.error(f"Invalid optimization level: {level}")
    
    def enable_auto_optimization(self, enabled: bool = True) -> None:
        """
        Enable or disable auto-optimization for the feature bank.
        
        Args:
            enabled: Whether to enable auto-optimization
        """
        self.config.enable_auto_optimization = enabled
        self.logger.info(f"Auto-optimization {'enabled' if enabled else 'disabled'}")
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Get optimization statistics from all generators.
        
        Returns:
            Dictionary with optimization statistics
        """
        stats = {
            'total_generators': 0,
            'auto_optimized_generators': 0,
            'total_optimizations': 0,
            'total_optimization_time': 0.0,
            'memory_savings_mb': 0.0,
            'optimization_levels': {}
        }
        
        for generator in self.registry.get_all():
            stats['total_generators'] += 1
            
            if isinstance(generator, AutoOptimizedFeatureGenerator):
                stats['auto_optimized_generators'] += 1
                
                # Get optimization stats from this generator
                gen_stats = generator.get_auto_optimization_stats()
                stats['total_optimizations'] += gen_stats.get('total_optimizations', 0)
                stats['total_optimization_time'] += gen_stats.get('total_optimization_time', 0.0)
                stats['memory_savings_mb'] += gen_stats.get('memory_savings_mb', 0.0)
                
                # Track optimization levels
                level = gen_stats.get('strategy_used', 'unknown')
                stats['optimization_levels'][level] = stats['optimization_levels'].get(level, 0) + 1
        
        return stats

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
    
    # Ensure the feature bank is properly initialized with generators
    if len(_global_feature_bank.registry.get_all()) == 0:
        # Force re-initialization if no generators are found
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
    tprint("✅ Global feature bank instance set")
