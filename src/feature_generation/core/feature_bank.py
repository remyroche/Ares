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
from ..utils.vectorbt_operation_batcher import VectorBTOperationBatcher, get_global_batcher
from ..utils.memory_pool_optimizer import MemoryPoolOptimizer, get_global_memory_pool
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
    chunk_size: int = 3000
    memory_efficient: bool = True
    cache_results: bool = True
    default_lookback: int = 20
    persist_generator_state: bool = True
    state_cache_dir: str = "data_cache/feature_states"
    state_cache_namespace: str = "feature_bank"
    state_cache_ttl_seconds: Optional[int] = None
    
    # Memory management settings
    max_cache_size: int = 1000  # Maximum cache entries
    max_cache_memory_mb: float = 500.0  # Maximum cache memory in MB
    cleanup_frequency: int = 5  # Cleanup every N batches
    aggressive_cleanup: bool = True  # Enable aggressive memory cleanup

    # Auto-optimization settings (enabled by default for better performance)
    enable_auto_optimization: bool = True
    default_optimization_level: str = "balanced"  # "conservative", "balanced", "aggressive"
    auto_optimization_config: Optional[AutoOptimizationConfig] = None

    # Regime feature settings (ONLY enable for regime-specific training)
    enable_regime_features: bool = False  # Disabled by default, enable only for regime models training

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

        # Initialize auto-optimization configuration (enabled by default)
        if self.config.enable_auto_optimization:
            if self.config.auto_optimization_config is None:
                self.config.auto_optimization_config = AutoOptimizationConfig()
                self.config.auto_optimization_config.optimization_level = OptimizationLevel(self.config.default_optimization_level)
            tprint("✅ Auto-optimization enabled for FeatureBank (default)")
        else:
            tprint("ℹ️ Auto-optimization disabled for FeatureBank (explicitly disabled)")

        # Initialize VectorBTRollingOptimizer if enabled
        self.vectorbt_rolling_optimizer = None
        self.unified_vectorization_manager = None
        
        # Initialize VectorBT operation batcher and memory pool optimizer
        self.vectorbt_batcher = get_global_batcher()
        self.memory_pool = get_global_memory_pool()

        if self.config.enable_matrix_operations:
            try:
                from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
                self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer()
                self.logger.debug("✅ VectorBTRollingOptimizer enabled")
            except ImportError:
                self.logger.warning("⚠️ VectorBTRollingOptimizer not available")

            # Initialize UnifiedVectorizationManager
            try:
                from src.utils.ml_common.unified_vectorization_manager import UnifiedVectorizationManager, get_unified_vectorization_manager, OperationType
                self.unified_vectorization_manager = get_unified_vectorization_manager()
                self.logger.debug("✅ UnifiedVectorizationManager enabled")
            except ImportError:
                self.logger.warning("⚠️ UnifiedVectorizationManager not available")

        # Initialize lookback optimizer if enabled
        self.lookback_optimizer = None
        if self.config.enable_lookback_optimization:
            # Check if we're in Tactician mode (use complementary optimizer) or Analyst mode (use regular optimizer)
            try:
                # Try to detect mode from environment or config
                import os
                mode = os.environ.get('ARES_MODE', 'analyst').lower()
                
                if mode == 'tactician':
                    # Use complementary lookback optimizer for Tactician mode
                    from src.feature_generation.utils.optimization.complementary_lookback_optimizer import ComplementaryLookbackOptimizer
                    self.lookback_optimizer = ComplementaryLookbackOptimizer()
                    self.logger.info("✅ Complementary lookback optimization enabled (Tactician mode)")
                else:
                    # Try to use regular lookback optimizer for Analyst mode
                    try:
                        from src.feature_generation.utils.optimization.lookback_optimizer import LookbackOptimizer
                        self.lookback_optimizer = LookbackOptimizer()
                        self.logger.info("✅ Regular lookback optimization enabled (Analyst mode)")
                    except ImportError:
                        # Fallback to complementary optimizer if regular is not available
                        from src.feature_generation.utils.optimization.complementary_lookback_optimizer import ComplementaryLookbackOptimizer
                        self.lookback_optimizer = ComplementaryLookbackOptimizer()
                        self.logger.info("✅ Complementary lookback optimization enabled (fallback from regular)")
            except ImportError as e:
                self.logger.warning(f"⚠️ Lookback optimization not available: {e}")

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
            'vectorization_optimizations': 0,
            'batch_count': 0,
            'periodic_cleanups': 0
        }
        
        # Periodic cleanup settings
        self.cleanup_frequency = self._get_config_value('cleanup_frequency', 5)  # Cleanup every 5 batches

        # Normalization configuration (use helper method for dict/Pydantic compatibility)
        self.auto_normalize = self._get_config_value('auto_normalize', True)
        self.normalization_method = self._get_config_value('normalization_method', 'zscore')
        self.normalization_config = {
            'exclude_categories': self._get_config_value('normalization_exclude_categories', []),
            'exclude_features': self._get_config_value('normalization_exclude_features', []),
            'rolling_windows': self._get_config_value('normalization_rolling_windows', [20, 50, 100])
        }

        # Cache for generated features with size limits
        self.feature_cache = {} if self._get_config_value('cache_results', True) else None
        self.max_cache_size = self._get_config_value('max_cache_size', 1000)  # Limit cache entries
        self.max_cache_memory_mb = self._get_config_value('max_cache_memory_mb', 500)  # Limit cache memory

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

        # Set as global feature bank if no global instance exists
        global _global_feature_bank
        if _global_feature_bank is None:
            _global_feature_bank = self
            tprint("✅ FeatureBank set as global instance")
            # Auto-register default generators only for the global instance
            tprint("🔧 Auto-registering feature generators...")
            self._auto_register_generators()
        else:
            # Use existing global feature bank instance - copy its registry
            self.logger.debug("🔄 Using existing global feature bank instance")
            # Copy registry and other state from global instance to maintain consistency
            self.registry = _global_feature_bank.registry
            self.feature_cache = _global_feature_bank.feature_cache
            self.state_cache = _global_feature_bank.state_cache
            self.persist_generator_state = _global_feature_bank.persist_generator_state
            # Copy the auto-registration completion flag from global instance
            if hasattr(_global_feature_bank, '_auto_registration_completed'):
                self._auto_registration_completed = _global_feature_bank._auto_registration_completed
            # If REGIME features are requested but not present in registry, register them now
            try:
                if self.config.enable_regime_features:
                    from .feature_generator import FeatureCategory
                    existing = self.registry.get_by_category(FeatureCategory.REGIME)
                    if not existing:
                        tprint("🎯 [FEATURE_BANK] Enabling REGIME generators on existing global bank", color="green")
                        gens = self._create_default_generators_for_category(FeatureCategory.REGIME)
                        if gens:
                            self.register_generators(gens)
                            tprint(f"✅ Registered {len(gens)} REGIME generators on existing bank", color="green")
            except Exception:
                pass

        # Check if this is a repeated initialization
        from src.utils.initialization_guard import check_initialization_status
        if check_initialization_status("FeatureBank"):
            self.logger.debug("🔄 FeatureBank already initialized, skipping duplicate initialization")
            return
        
        # Reduced verbosity - only log once per session
        if not hasattr(FeatureBank, '_logged_initialization'):
            self.logger.info("✅ FeatureBank initialized")
            self.logger.info(f"📊 Matrix ops: {self.config.enable_matrix_operations}, "
                            f"GPU: {self.config.enable_gpu_acceleration}, "
                        f"Lookback opt: {self.config.enable_lookback_optimization}")
            FeatureBank._logged_initialization = True
        
        # Mark as initialized to prevent duplicate initialization
        from src.utils.initialization_guard import init_guard
        init_guard.mark_initialized("FeatureBank")

    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """Helper to fetch values from dataclass or dict configs."""
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

    def _auto_register_generators(self) -> None:
        """
        Auto-register default feature generators from all categories.
        """
        global _global_feature_bank

        # Check if auto-registration has already been completed globally to prevent duplicates
        if _global_feature_bank and hasattr(_global_feature_bank, '_auto_registration_completed') and _global_feature_bank._auto_registration_completed:
            self.logger.debug("🔄 Auto-registration already completed globally, skipping")
            # Mark as completed on current instance too
            self._auto_registration_completed = True
            return

        # Also check if we already have generators registered (defensive check)
        if self.registry and len(self.registry.get_all()) > 0:
            self.logger.debug(f"🔄 Registry already has {len(self.registry.get_all())} generators, skipping auto-registration")
            # Mark as completed on current instance too
            self._auto_registration_completed = True
            return

        tprint("🔧 Starting auto-registration of feature generators...")
        try:
            # List of categories to auto-register
            # NOTE: CUSTOM_SUPPORT_RESISTANCE is intentionally excluded (disabled by default)
            # Enable it manually by registering custom SR generators explicitly
            # Build categories list - REGIME only if explicitly enabled
            categories_to_register = []

            # Add REGIME category ONLY if explicitly enabled (for regime models training)
            if self.config.enable_regime_features:
                categories_to_register.append(FeatureCategory.REGIME)
                tprint("🎯 [FEATURE_BANK] REGIME features ENABLED (regime models training mode)", color="green")
            else:
                tprint("ℹ️ [FEATURE_BANK] REGIME features DISABLED (standard mode)", color="cyan")

            # Add standard categories (always enabled)
            categories_to_register.extend([
                FeatureCategory.MOMENTUM,
                FeatureCategory.VOLATILITY,
                FeatureCategory.TREND,
                FeatureCategory.VOLUME,
                FeatureCategory.SUPPORT_RESISTANCE,  # Pre-created SR levels only
                # FeatureCategory.CUSTOM_SUPPORT_RESISTANCE,  # DISABLED by default - custom SR features
                FeatureCategory.RETURNS,
                FeatureCategory.OSCILLATOR,
                FeatureCategory.CANDLESTICK_PATTERN,
                FeatureCategory.ENTROPY,
                FeatureCategory.ORDER_FLOW,
                FeatureCategory.ACCELERATION,
                FeatureCategory.CROSS_TIMEFRAME,
                FeatureCategory.INTERACTION,
                FeatureCategory.MICROSTRUCTURE,
                FeatureCategory.ADVANCED_STATISTICAL,
                FeatureCategory.SPECTRAL_WAVELET
            ])

            registered_count = 0
            total_categories = len(categories_to_register)
            tprint(f"🚀 Initializing {total_categories} feature categories...")

            for i, category in enumerate(categories_to_register, 1):
                try:
                    # Debug: Check if category is the right type
                    if isinstance(category, str):
                        self.logger.error(f"❌ Category is a string: {category} (type: {type(category)})")
                        tprint(f"❌ Category is a string: {category} (type: {type(category)})")
                        continue

                    self.logger.debug(f"Processing category {i}/{total_categories}: {category.value}")
                    generators = self._create_default_generators_for_category(category)
                    self.logger.debug(f"Created {len(generators)} generators for {category.value}")
                    for generator in generators:
                        self.register_generator(generator)
                        registered_count += 1
                    self.logger.debug(f"Progress: {i}/{total_categories} categories completed")
                except Exception as e:
                    # Debug: Check category type in exception handler
                    if isinstance(category, str):
                        tprint(f"⚠️ Failed to register {category} generators (category is string): {e}")
                        self.logger.warning(f"⚠️ Failed to register {category} generators (category is string): {e}")
                    else:
                        tprint(f"⚠️ Failed to register {category.value} generators: {e}")
                        self.logger.warning(f"⚠️ Failed to register {category.value} generators: {e}")

            tprint(f"✅ Auto-registration completed. Registered {registered_count} generators")
            self.logger.info(f"✅ Auto-registered {registered_count} generators from {len(categories_to_register)} categories")

            # Populate the generator factory with the registered generators
            try:
                self.generator_factory.populate_from_feature_bank(self)
                tprint("✅ Generator factory populated with registered generators")
            except Exception as e:
                tprint(f"⚠️ Failed to populate generator factory: {e}")
                self.logger.warning(f"Failed to populate generator factory: {e}")

            # Mark auto-registration as completed globally to prevent duplicates
            self._auto_registration_completed = True
            if _global_feature_bank:
                _global_feature_bank._auto_registration_completed = True

        except Exception as e:
            tprint(f"❌ Auto-registration failed: {e}")
            self.logger.warning(f"⚠️ Auto-registration failed: {e}")

    def _create_default_generators_for_category(self, category: FeatureCategory) -> List[FeatureGenerator]:
        """
        Create default generators for a given category using auto-optimized generators.
        """
        # Debug: Check if category is the right type
        if isinstance(category, str):
            self.logger.error(f"❌ _create_default_generators_for_category received string: {category} (type: {type(category)})")
            tprint(f"❌ _create_default_generators_for_category received string: {category} (type: {type(category)})")
            return []
        
        self.logger.debug(f"Creating auto-optimized generators for category: {category.value}")
        try:
            # Map categories to their creation functions
            category_creators = {
                FeatureCategory.REGIME: self._create_regime_generators,  # CRITICAL: Added for regime classification
                FeatureCategory.MOMENTUM: self._create_momentum_generators,
                FeatureCategory.VOLATILITY: self._create_volatility_generators,
                FeatureCategory.TREND: self._create_trend_generators,
                FeatureCategory.VOLUME: self._create_volume_generators,
                FeatureCategory.SUPPORT_RESISTANCE: self._create_sr_generators,
                FeatureCategory.CUSTOM_SUPPORT_RESISTANCE: self._create_custom_sr_generators,
                FeatureCategory.RETURNS: self._create_returns_generators,
                FeatureCategory.OSCILLATOR: self._create_oscillator_generators,
                FeatureCategory.CANDLESTICK_PATTERN: self._create_pattern_generators,
                FeatureCategory.ENTROPY: self._create_entropy_generators,
                FeatureCategory.ORDER_FLOW: self._create_order_flow_generators,
                FeatureCategory.ACCELERATION: self._create_acceleration_generators,
                FeatureCategory.CROSS_TIMEFRAME: self._create_cross_timeframe_generators,
                FeatureCategory.INTERACTION: self._create_interaction_generators,
                FeatureCategory.MICROSTRUCTURE: self._create_microstructure_generators,
                FeatureCategory.ADVANCED_STATISTICAL: self._create_advanced_statistical_generators,
                FeatureCategory.SPECTRAL_WAVELET: self._create_spectral_wavelet_generators
            }

            creator_func = category_creators.get(category)
            if creator_func:
                self.logger.debug(f"Creating {category.value} features with auto-optimization...")
                generators = creator_func()

                # Convert generators to auto-optimized versions if auto-optimization is enabled
                if self.config.enable_auto_optimization:
                    self.logger.debug(f"Converting {len(generators)} generators to auto-optimized versions...")
                    auto_optimized_generators = []
                    for generator in generators:
                        auto_optimized_gen = self._convert_to_auto_optimized(generator)
                        auto_optimized_generators.append(auto_optimized_gen)
                    generators = auto_optimized_generators
                    self.logger.debug(f"Created {len(generators)} auto-optimized generators for {category.value}")
                else:
                    self.logger.debug(f"Created {len(generators)} standard generators for {category.value} (auto-optimization disabled)")

                return generators
            else:
                tprint(f"⚠️ No creator function available for category: {category.value}")
                self.logger.warning(f"⚠️ No creator function available for category: {category.value}")
                return []

        except Exception as e:
            # Debug: Check category type in exception handler
            if isinstance(category, str):
                tprint(f"❌ Failed to create generators for {category} (category is string): {e}")
                self.logger.warning(f"⚠️ Failed to create generators for {category} (category is string): {e}")
            else:
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
            self.logger.debug(f"Converting generator '{generator.config.name}' to auto-optimized...")

            # Create a modified config with a unique name for the auto-optimized version
            self.logger.debug("Creating auto-optimized generator with modified config...")
            original_config = generator.config

            # Create a new config with a modified name to avoid naming conflicts
            # Use the existing config as a base and override specific fields
            modified_config = FeatureConfig(
                name=original_config.name,  # Keep the original meaningful name
                category=original_config.category,
                description=f"Auto-optimized version of {original_config.description}",
                required_columns=original_config.required_columns,
                optional_columns=original_config.optional_columns,
                default_lookback=original_config.default_lookback,
                min_lookback=original_config.min_lookback,
                max_lookback=original_config.max_lookback,
                parameters=original_config.parameters.copy() if original_config.parameters else {},
                dependencies=original_config.dependencies.copy() if original_config.dependencies else [],
                matrix_optimized=original_config.matrix_optimized,
                gpu_accelerated=original_config.gpu_accelerated,
                enable_feature_selection=original_config.enable_feature_selection,
                use_vectorbt=original_config.use_vectorbt,
                vectorbt_threshold=original_config.vectorbt_threshold
            )

            # Add auto-optimization metadata to the parameters if supported
            if modified_config.parameters is None:
                modified_config.parameters = {}
            modified_config.parameters['auto_optimized'] = True
            modified_config.parameters['original_name'] = original_config.name
            modified_config.parameters['optimization_level'] = self.config.default_optimization_level

            # Create a custom AutoOptimizedFeatureGenerator that preserves the original generator's logic
            class CustomAutoOptimizedFeatureGenerator(AutoOptimizedFeatureGenerator):
                def __init__(self, original_generator, config, auto_optimization_config):
                    super().__init__(config, auto_optimization_config)
                    self.original_generator = original_generator
                    # Initialize logger
                    import logging
                    self.logger = logging.getLogger(self.__class__.__name__)

                @staticmethod
                def _ensure_series(values: Any, index: pd.Index, feature_name: str) -> pd.Series:
                    """Convert arbitrary outputs into a numeric Series aligned to the provided index."""
                    try:
                        if isinstance(values, pd.Series):
                            series = values.copy()
                        elif isinstance(values, pd.DataFrame):
                            series = values.iloc[:, 0].copy() if not values.empty else pd.Series(dtype=float)
                        elif isinstance(values, dict):
                            for key in ('result', 'data', 'series'):
                                if key in values:
                                    return CustomAutoOptimizedFeatureGenerator._ensure_series(values[key], index, feature_name)
                            series = pd.Series(values)
                        elif hasattr(values, '__iter__') and not np.isscalar(values):
                            series = pd.Series(list(values))
                        else:
                            series = pd.Series([values], dtype=float)

                        series = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan)

                        if len(series) == len(index):
                            series.index = index
                        else:
                            series = series.reset_index(drop=True)
                            series = series.reindex(range(len(index)), fill_value=np.nan)
                            series.index = index

                        series = series.fillna(0.0)
                        series.name = feature_name
                        return series
                    except Exception as conversion_error:
                        logging.getLogger(__name__).warning(
                            "Failed to coerce feature output to Series (%s): %s", feature_name, conversion_error
                        )
                        return pd.Series(0.0, index=index, name=feature_name)
                
                def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                    """Use the original generator's feature generation logic."""
                    try:
                        # Apply auto-optimization if enabled
                        if self.auto_optimization_config.enable_auto_optimization:
                            data = self._auto_optimize_data(data)

                        # Use the original generator's feature generation logic
                        if hasattr(self.original_generator, '_generate_feature'):
                            feature_series = self.original_generator._generate_feature(data, **kwargs)
                        else:
                            # Fallback to the original generator's generate method
                            result = self.original_generator.generate(data, **kwargs)
                            # Handle case where result.data might be a scalar
                            if hasattr(result, 'data'):
                                feature_series = result.data
                            else:
                                feature_series = result

                        # Ensure feature_series is a proper pandas Series
                        if not isinstance(feature_series, pd.Series):
                            if isinstance(feature_series, (int, float)):
                                # Convert scalar to Series
                                feature_series = pd.Series([feature_series] * len(data), index=data.index)
                            else:
                                # Try to convert other types to Series
                                try:
                                    feature_series = pd.Series(feature_series)
                                except Exception:
                                    # If conversion fails, create a Series of NaN
                                    feature_series = pd.Series([np.nan] * len(data), index=data.index)

                        # Apply post-processing optimization
                        if self.auto_optimization_config.enable_auto_optimization:
                            feature_series = self._optimize_feature_series(feature_series)

                        feature_series = CustomAutoOptimizedFeatureGenerator._ensure_series(feature_series, data.index, self.config.name)
                        return feature_series

                    except Exception as e:
                        self.logger.error(f"Error in custom auto-optimized feature generation: {e}")
                        # Fallback to parent implementation
                        fallback = super()._generate_feature(data, **kwargs)
                        return CustomAutoOptimizedFeatureGenerator._ensure_series(fallback, data.index, self.config.name)

            auto_optimized_gen = CustomAutoOptimizedFeatureGenerator(
                generator, modified_config, self.config.auto_optimization_config
            )
            self.logger.debug("Auto-optimized generator created with preserved logic")

            # Copy any additional state from original generator
            if hasattr(generator, 'get_state'):
                self.logger.debug("Copying state from original generator...")
                state = generator.get_state()
                if state and hasattr(auto_optimized_gen, 'load_state'):
                    auto_optimized_gen.load_state(state)
                    self.logger.debug("State copied successfully")
                else:
                    self.logger.debug("No state to copy or load_state not available")
            else:
                self.logger.debug("Original generator has no get_state method")

            self.logger.debug(f"Generator '{generator.config.name}' converted to auto-optimized with preserved logic")
            return auto_optimized_gen

        except Exception as e:
            tprint(f"❌ Failed to convert '{generator.config.name}' to auto-optimized: {e}")
            self.logger.warning(f"Failed to convert {generator.config.name} to auto-optimized: {e}")
            tprint("🔄 Returning original generator as fallback")
            # Return original generator if conversion fails
            return generator

    def _create_momentum_generators(self) -> List[FeatureGenerator]:
        """Create momentum-specific feature generators."""
        self.logger.debug("Creating momentum generators...")
        generators = []
        try:
            from ..categories.momentum import create_default_momentum_generators
            advanced_generators = create_default_momentum_generators()
            generators.extend(advanced_generators)
            self.logger.debug(f"Created {len(advanced_generators)} momentum generators")
        except Exception as e:
            tprint(f"⚠️ Failed to create momentum generators: {e}")
            self.logger.warning(f"⚠️ Failed to create momentum generators: {e}")

        return generators

    def _create_volatility_generators(self) -> List[FeatureGenerator]:
        """Create volatility-specific feature generators."""
        self.logger.debug("Creating volatility generators...")
        generators = []
        try:
            from ..categories.volatility import create_default_volatility_generators
            advanced_generators = create_default_volatility_generators()
            
            # Debug: Check each generator's category
            for i, gen in enumerate(advanced_generators):
                if hasattr(gen, 'config') and hasattr(gen.config, 'category'):
                    if isinstance(gen.config.category, str):
                        self.logger.error(f"❌ Generator {i} has string category: {gen.config.category} (type: {type(gen.config.category)})")
                        tprint(f"❌ Generator {i} has string category: {gen.config.category} (type: {type(gen.config.category)})")
                    else:
                        self.logger.debug(f"✅ Generator {i} category: {gen.config.category.value}")
            
            generators.extend(advanced_generators)
            self.logger.debug(f"Created {len(advanced_generators)} volatility generators")
        except Exception as e:
            tprint(f"⚠️ Failed to create volatility generators: {e}")
            self.logger.warning(f"⚠️ Failed to create volatility generators: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")

        return generators

    def _create_trend_generators(self) -> List[FeatureGenerator]:
        """Create trend-specific feature generators."""
        self.logger.debug("Creating trend generators...")
        generators = []
        try:
            from ..categories.trend import create_default_trend_generators
            advanced_generators = create_default_trend_generators()
            generators.extend(advanced_generators)
            self.logger.debug(f"Created {len(advanced_generators)} trend generators")
        except Exception as e:
            tprint(f"⚠️ Failed to create trend generators: {e}")
            self.logger.warning(f"⚠️ Failed to create trend generators: {e}")

        return generators

    def _create_volume_generators(self) -> List[FeatureGenerator]:
        """Create volume-specific feature generators."""
        self.logger.debug("Creating volume generators...")
        generators = []
        try:
            from ..categories.volume import create_default_volume_generators
            advanced_generators = create_default_volume_generators()
            generators.extend(advanced_generators)
            self.logger.debug(f"Created {len(advanced_generators)} volume generators")
        except Exception as e:
            tprint(f"⚠️ Failed to create volume generators: {e}")
            self.logger.warning(f"⚠️ Failed to create volume generators: {e}")

        return generators

    def _create_sr_generators(self) -> List[FeatureGenerator]:
        """Create support/resistance-specific feature generators (pre-created SR levels only)."""
        self.logger.debug("Creating support/resistance generators...")
        generators = []
        try:
            from ..categories.support_resistance import create_default_support_resistance_generators
            advanced_generators = create_default_support_resistance_generators()
            generators.extend(advanced_generators)
            self.logger.debug(f"Created {len(advanced_generators)} support/resistance generators")
        except Exception as e:
            tprint(f"⚠️ Failed to create support/resistance generators: {e}")
            self.logger.warning(f"⚠️ Failed to create support/resistance generators: {e}")

        return generators

    def _create_custom_sr_generators(self) -> List[FeatureGenerator]:
        """Create custom support/resistance feature generators (strength, distance, touches, etc.)."""
        self.logger.debug("Creating custom SR generators...")
        generators = []
        try:
            from ..categories.custom_support_resistance import create_default_custom_sr_generators
            custom_generators = create_default_custom_sr_generators()
            generators.extend(custom_generators)
            self.logger.debug(f"Created {len(custom_generators)} custom SR generators")
            tprint(f"✅ Created {len(custom_generators)} custom SR generators (strength, distance, touches, etc.)")
        except Exception as e:
            tprint(f"⚠️ Failed to create custom SR generators: {e}")
            self.logger.warning(f"⚠️ Failed to create custom SR generators: {e}")

        return generators

    def _create_returns_generators(self) -> List[FeatureGenerator]:
        """Create returns-specific feature generators."""
        self.logger.debug("Creating returns generators...")
        generators = []
        try:
            from ..categories.returns import create_default_returns_generators
            advanced_generators = create_default_returns_generators()
            generators.extend(advanced_generators)
            self.logger.debug(f"Created {len(advanced_generators)} returns generators")
        except Exception as e:
            tprint(f"⚠️ Failed to create returns generators: {e}")
            self.logger.warning(f"⚠️ Failed to create returns generators: {e}")

        return generators

    def _create_oscillator_generators(self) -> List[FeatureGenerator]:
        """Create oscillator-specific feature generators."""
        self.logger.debug("Creating oscillator generators...")
        generators = []
        try:
            from ..categories.oscillator import create_default_oscillator_generators
            advanced_generators = create_default_oscillator_generators()
            generators.extend(advanced_generators)
            self.logger.debug(f"Created {len(advanced_generators)} oscillator generators")
        except Exception as e:
            tprint(f"⚠️ Failed to create oscillator generators: {e}")
            self.logger.warning(f"⚠️ Failed to create oscillator generators: {e}")

        return generators

    def _create_pattern_generators(self) -> List[FeatureGenerator]:
        """Create candlestick pattern-specific feature generators."""
        self.logger.debug("Creating candlestick pattern generators...")
        generators = []
        try:
            from ..categories.candlestick_pattern import create_default_candlestick_pattern_generators
            advanced_generators = create_default_candlestick_pattern_generators()
            generators.extend(advanced_generators)
            self.logger.debug(f"Created {len(advanced_generators)} candlestick pattern generators")
        except Exception as e:
            tprint(f"⚠️ Failed to create candlestick pattern generators: {e}")
            self.logger.warning(f"⚠️ Failed to create candlestick pattern generators: {e}")

        return generators

    def _create_entropy_generators(self) -> List[FeatureGenerator]:
        """Create entropy-specific feature generators."""
        self.logger.debug("Creating entropy generators...")
        generators = []
        try:
            from ..categories.entropy import create_default_entropy_generators
            advanced_generators = create_default_entropy_generators()
            generators.extend(advanced_generators)
            self.logger.debug(f"Created {len(advanced_generators)} entropy generators")
        except Exception as e:
            tprint(f"⚠️ Failed to create entropy generators: {e}")
            self.logger.warning(f"⚠️ Failed to create entropy generators: {e}")

        return generators

    def _create_order_flow_generators(self) -> List[FeatureGenerator]:
        """Create order flow-specific feature generators."""
        self.logger.debug("Creating order flow generators...")
        generators = []
        try:
            from ..categories.microstructure_features import create_default_microstructure_generators
            advanced_generators = create_default_microstructure_generators()
            generators.extend(advanced_generators)
            self.logger.debug(f"Created {len(advanced_generators)} order flow generators")
        except Exception as e:
            tprint(f"⚠️ Failed to create order flow generators: {e}")
            self.logger.warning(f"⚠️ Failed to create order flow generators: {e}")

        return generators

    def _create_acceleration_generators(self) -> List[FeatureGenerator]:
        """Create acceleration-specific feature generators."""
        self.logger.debug("Creating acceleration generators...")
        generators = []
        try:
            from ..categories.vectorbt_acceleration import create_default_acceleration_generators
            advanced_generators = create_default_acceleration_generators()
            generators.extend(advanced_generators)
            self.logger.debug(f"Created {len(advanced_generators)} acceleration generators")
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
        self.logger.debug("Creating autoencoder generators...")
        generators = []
        try:
            from ..categories.autoencoder import create_default_autoencoder_generators
            advanced_generators = create_default_autoencoder_generators()
            generators.extend(advanced_generators)
            self.logger.debug(f"Created {len(advanced_generators)} autoencoder generators")
        except Exception as e:
            tprint(f"⚠️ Failed to create autoencoder generators: {e}")
            self.logger.warning(f"⚠️ Failed to create autoencoder generators: {e}")

        return generators

    def _create_interaction_generators(self) -> List[FeatureGenerator]:
        """Create interaction-specific feature generators."""
        self.logger.debug("Creating interaction generators...")
        generators = []
        try:
            from ..categories.interaction import create_default_interaction_generators
            advanced_generators = create_default_interaction_generators()
            generators.extend(advanced_generators)
            self.logger.debug(f"Created {len(advanced_generators)} interaction generators")
        except Exception as e:
            tprint(f"⚠️ Failed to create interaction generators: {e}")
            self.logger.warning(f"⚠️ Failed to create interaction generators: {e}")

        return generators

    def _create_microstructure_generators(self) -> List[FeatureGenerator]:
        """Create microstructure-specific feature generators."""
        self.logger.debug("Creating microstructure generators...")
        generators = []
        try:
            from ..categories.microstructure_features import create_default_microstructure_generators
            advanced_generators = create_default_microstructure_generators()
            generators.extend(advanced_generators)
            self.logger.debug(f"Created {len(advanced_generators)} microstructure generators")
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
                from ..categories.regime_features import RegimeVolatilityFeatureGenerator
                # Create a regime volatility generator (single instance for now)
                generators.append(RegimeVolatilityFeatureGenerator())
            except ImportError:
                pass

            # Try to create regime statistical generators
            try:
                from ..categories.regime_features import RegimeStatisticalFeatureGenerator
                generators.append(RegimeStatisticalFeatureGenerator())
            except ImportError:
                pass

            # Regime feature integration is now part of regime_features.py
            # No separate import needed

            # Try to create regime structural trend generators
            try:
                from ..categories.regime_features import RegimeStructuralTrendFeatureGenerator
                generators.append(RegimeStructuralTrendFeatureGenerator())
            except ImportError:
                pass

            # Try to create regime volume generators (new robust version)
            try:
                from ..categories.regime_volume import RegimeVolumeFeatureGenerator, create_regime_volume_generators
                # Use the factory function to create generators for multiple windows
                generators.extend(create_regime_volume_generators(windows=[14, 20, 30]))
            except ImportError:
                # Fallback to old regime_features version
                try:
                    from ..categories.regime_features import RegimeVolumeFeatureGenerator
                    generators.append(RegimeVolumeFeatureGenerator())
                except ImportError:
                    pass

            # NEW: Register regime analysis generators
            try:
                from ..categories.market_structure import MarketStructureGenerator
                generators.append(MarketStructureGenerator())
            except ImportError:
                pass
            try:
                from ..categories.regime_persistence import RegimePersistenceGenerator
                generators.append(RegimePersistenceGenerator())
            except ImportError:
                pass
            try:
                from ..categories.regime_probability import RegimeProbabilityGenerator
                generators.append(RegimeProbabilityGenerator())
            except ImportError:
                pass
            try:
                from ..categories.regime_transitions import RegimeTransitionGenerator
                generators.append(RegimeTransitionGenerator())
            except ImportError:
                pass
            try:
                from ..categories.regime_uncertainty import RegimeUncertaintyGenerator
                generators.append(RegimeUncertaintyGenerator())
            except ImportError:
                pass

            # Multi-timeframe EWMA generators (inspired by rolling_hmm_clustering)
            try:
                from ..categories.multi_timeframe_ewma import (
                    MultiTimeframeEWMAReturnsGenerator,
                    MultiTimeframeEWMAVolatilityGenerator,
                    MultiTimeframeEWMATrendGenerator,
                    MultiTimeframeEWMAVolumeGenerator,
                )
                # Add EWMA generators with multiple timeframes (faster, more responsive)
                windows = [3, 8, 20]
                generators.append(MultiTimeframeEWMAReturnsGenerator(windows=windows))
                generators.append(MultiTimeframeEWMAVolatilityGenerator(windows=windows))
                generators.append(MultiTimeframeEWMATrendGenerator(windows=windows))
                generators.append(MultiTimeframeEWMAVolumeGenerator(windows=windows))
                self.logger.info(f"✅ Added 4 multi-timeframe EWMA generators with windows {windows}")
            except ImportError as e:
                self.logger.debug(f"ℹ️ Multi-timeframe EWMA generators not available: {e}")
                pass

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create regime generators: {e}")

        # Normalize category and names to REGIME for all created generators
        try:
            normalized = []
            for gen in generators:
                if hasattr(gen, 'config'):
                    gen.config.category = FeatureCategory.REGIME
                    if isinstance(gen.config.name, str) and not gen.config.name.lower().startswith('regime_'):
                        gen.config.name = f"regime_{gen.config.name}"
                normalized.append(gen)
            generators = normalized
            tprint(f"✅ Normalized {len(generators)} generators to REGIME category", color="green")
        except Exception as _:
            pass

        return generators

    def _create_time_generators(self) -> List[FeatureGenerator]:
        """Create time-specific feature generators."""
        self.logger.debug("Creating time generators...")
        generators = []
        try:
            from ..categories.time import create_default_time_generators
            advanced_generators = create_default_time_generators()
            generators.extend(advanced_generators)
            self.logger.debug(f"Created {len(advanced_generators)} time generators")
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
                from ..categories.representation_learning import create_default_representation_learning_generators
                enhanced_generators = create_default_representation_learning_generators()
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
            from ..categories.spectral_features import create_default_spectral_generators
            spectral_generators = create_default_spectral_generators()
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
        self.logger.debug(f"Registering generator: {generator.config.name}")
        
        # Debug: Check if category is the right type
        if hasattr(generator.config, 'category'):
            if isinstance(generator.config.category, str):
                self.logger.error(f"❌ Generator {generator.config.name} has string category: {generator.config.category} (type: {type(generator.config.category)})")
                tprint(f"❌ Generator {generator.config.name} has string category: {generator.config.category} (type: {type(generator.config.category)})")
                return
            else:
                self.logger.debug(f"✅ Generator {generator.config.name} category: {generator.config.category.value}")
        
        self.registry.register(generator)
        self.logger.debug(f"Successfully registered generator: {generator.config.name}")
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
        if isinstance(category, FeatureCategory):
            tprint(f"🔍 Looking up generators for category: {category.value}")
        else:
            tprint(f"🔍 Looking up generators for category: {category}")
        generators = self.registry.get_by_category(category)
        if isinstance(category, FeatureCategory):
            tprint(f"✅ Found {len(generators)} generators for category: {category.value}")
        else:
            tprint(f"✅ Found {len(generators)} generators for category: {category}")
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

    def get_available_features(self, categories: Optional[List[str]] = None, 
                              features: Optional[List[str]] = None) -> List[Dict]:
        """
        Get all available features as dictionaries with metadata.
        
        Args:
            categories: Optional list of category names to filter by
            features: Optional list of specific feature names to filter by
            
        Returns:
            List of dictionaries containing feature metadata
        """
        feature_list = []
        
        # Get all registered generators
        all_generators = []
        if categories:
            # Filter by categories
            for cat_name in categories:
                try:
                    if isinstance(cat_name, str):
                        cat = FeatureCategory(cat_name)
                    else:
                        cat = cat_name
                    cat_generators = self.get_generators_by_category(cat)
                    all_generators.extend(cat_generators)
                except (ValueError, AttributeError):
                    # Skip invalid category names
                    continue
        else:
            # Get all generators from all categories
            for category in FeatureCategory:
                all_generators.extend(self.get_generators_by_category(category))
        
        # Convert generators to feature dictionaries
        for generator in all_generators:
            feature_dict = {
                'name': generator.config.name,
                'category': generator.config.category.value if generator.config.category else 'unknown',
                'description': generator.config.description or '',
                'required_columns': generator.config.required_columns or [],
                'generator': generator
            }
            
            # Apply feature name filter if specified
            if features is None or generator.config.name in features:
                feature_list.append(feature_dict)
        
        return feature_list

    def generate_features(self,
                         data: pd.DataFrame,
                         categories: Optional[List[Union[str, FeatureCategory]]] = None,
                         features: Optional[List[str]] = None,
                         lookback_optimization: bool = False,
                         target_column: Optional[str] = None,
                         use_optimized_pipeline: bool = True,
                         progressive_loading: bool = True,
                         batch_size: Optional[int] = None,
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
            progressive_loading: Whether to load features in batches for memory efficiency
            batch_size: Number of features to process per batch (default: adaptive based on memory)
            **kwargs: Additional parameters

        Returns:
            DataFrame with generated features
        """
        start_time = time.time()
        tprint("🚀 Starting feature generation...")
        self.logger.info(f"🎯 Starting feature generation...")

        if len(data) == 0:
            tprint("⚠️ Empty data provided")
            self.logger.warning("Empty data provided")
            return pd.DataFrame()

        # ================= Execution mode handling + date range reporting =================
        # Determine execution mode (prefer kwargs; fallback to env)
        import os
        exec_mode = kwargs.get('execution_mode') or kwargs.get('intensity') or kwargs.get('mode') or os.environ.get('ARES_EXECUTION_MODE') or os.environ.get('EXECUTION_MODE')
        exec_mode_str = str(getattr(exec_mode, 'value', getattr(exec_mode, 'name', exec_mode))).lower() if exec_mode else None

        # Extract timestamp series (column preferred, fallback to index if datetime-like)
        ts_series = None
        if 'timestamp' in data.columns:
            ts_col = data['timestamp']
            try:
                if np.issubdtype(ts_col.dtype, np.datetime64):
                    ts_series = pd.to_datetime(ts_col)
                elif np.issubdtype(ts_col.dtype, np.integer):
                    # Heuristic: seconds vs milliseconds
                    unit = 'ms' if ts_col.dropna().astype(np.int64).median() > 10**12 else 's'
                    ts_series = pd.to_datetime(ts_col, unit=unit, errors='coerce')
                else:
                    ts_series = pd.to_datetime(ts_col, errors='coerce')
            except Exception:
                ts_series = None
        
        # Also check for other common timestamp column names
        if ts_series is None:
            for col_name in ['time', 'datetime', 'date', 'open_time', 'close_time']:
                if col_name in data.columns:
                    try:
                        ts_col = data[col_name]
                        # Check if it's already a datetime type
                        if pd.api.types.is_datetime64_any_dtype(ts_col):
                            ts_series = ts_col
                        # Check if it's numeric (Unix timestamp)
                        elif pd.api.types.is_numeric_dtype(ts_col):
                            # Determine if it's seconds or milliseconds
                            sample_val = ts_col.dropna().iloc[0] if not ts_col.empty else 0
                            if sample_val > 1e12:  # Likely milliseconds
                                ts_series = pd.to_datetime(ts_col, unit='ms', errors='coerce')
                            else:  # Likely seconds
                                ts_series = pd.to_datetime(ts_col, unit='s', errors='coerce')
                        else:
                            # Try as string
                            ts_series = pd.to_datetime(ts_col, errors='coerce')
                        
                        if not ts_series.empty and not ts_series.isna().all() and ts_series.max() > pd.Timestamp('2020-01-01'):
                            self.logger.info(f"📅 Using timestamp column: {col_name}")
                            break
                        else:
                            ts_series = None
                    except Exception as e:
                        self.logger.debug(f"Failed to parse timestamp column {col_name}: {e}")
                        continue
        
        if ts_series is None and isinstance(data.index, pd.DatetimeIndex):
            ts_series = data.index.to_series()

        # Light mode: restrict to last 180 days if we have timestamps
        if exec_mode_str == 'light':
            if ts_series is not None and not ts_series.empty:
                end_dt = ts_series.max()
                start_dt_full = ts_series.min()
                cutoff_dt = end_dt - pd.Timedelta(days=180)
                mask = ts_series >= cutoff_dt
                if mask.any() and mask.sum() < len(ts_series):
                    rows_before = len(data)
                    data = data.loc[mask.values]
                    rows_after = len(data)
                    days_used = (end_dt.normalize() - cutoff_dt.normalize()).days + 1
                    tprint(f"📅 LIGHT mode: restricting to last 180 days: {cutoff_dt.date()} → {end_dt.date()} ({days_used} days, {rows_after}/{rows_before} rows)")
                    self.logger.info(f"📅 LIGHT mode: 180-day cap applied: {cutoff_dt} → {end_dt} | rows {rows_after}/{rows_before}")
                else:
                    # If mask doesn't reduce, still report range
                    days_range = (end_dt.normalize() - start_dt_full.normalize()).days + 1
                    tprint(f"📅 LIGHT mode: data already within {days_range} days: {start_dt_full.date()} → {end_dt.date()}")
                    self.logger.info(f"📅 LIGHT mode: data range {start_dt_full} → {end_dt} (~{days_range} days)")
            else:
                # Fallback: if no timestamp available, restrict to last ~30% of data
                # This is a rough approximation for light mode with 180 days
                total_rows = len(data)
                if total_rows > 1000:  # Only apply if we have enough data
                    rows_to_keep = max(1000, int(total_rows * 0.3))  # Keep at least 1000 rows or 30%
                    data = data.tail(rows_to_keep)
                    tprint(f"📅 LIGHT mode: no timestamp found, restricting to last {rows_to_keep} rows (from {total_rows} total)")
                    self.logger.info(f"📅 LIGHT mode: no timestamp available, using last {rows_to_keep}/{total_rows} rows")
                else:
                    tprint(f"📅 LIGHT mode: no timestamp found, data already small ({total_rows} rows)")
                    self.logger.info(f"📅 LIGHT mode: no timestamp available, data already small ({total_rows} rows)")
        else:
            # Report full date range when timestamps available
            if ts_series is not None and not ts_series.empty:
                start_dt = ts_series.min()
                end_dt = ts_series.max()
                days_range = (end_dt.normalize() - start_dt.normalize()).days + 1
                tprint(f"📅 Processing date range: {start_dt.date()} → {end_dt.date()} ({days_range} days)")
                self.logger.info(f"📅 Processing date range: {start_dt} → {end_dt} (~{days_range} days)")

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
        
        # Add detailed progress monitoring
        self.logger.info("🔄 Feature generation process started")
        self.logger.info(f"📈 Processing {len(data)} rows of data")
        self.logger.info(f"🧮 Memory usage: {data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # Log generator breakdown by category
        category_counts = {}
        for gen in generators_to_use:
            if hasattr(gen, 'config') and hasattr(gen.config, 'category'):
                cat_name = getattr(gen.config.category, 'value', str(gen.config.category))
                category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
        
        self.logger.info("📋 Generator breakdown by category:")
        for cat, count in sorted(category_counts.items()):
            self.logger.info(f"  • {cat}: {count} generators")

        # Optimize lookbacks if requested
        if lookback_optimization and target_column and self.lookback_optimizer:
            # Check if we're using complementary optimizer (Tactician mode)
            optimizer_type = type(self.lookback_optimizer).__name__
            if optimizer_type == 'ComplementaryLookbackOptimizer':
                # Extract analyst signals and regime information from kwargs for Tactician mode
                analyst_signals = kwargs.get('analyst_signals', None)
                regime_series = kwargs.get('regime_series', None)
                generators_to_use = self._optimize_lookbacks(
                    generators_to_use, data, target_column, analyst_signals, regime_series
                )
            else:
                # Use standard optimization for Analyst mode
                generators_to_use = self._optimize_lookbacks(generators_to_use, data, target_column)

        # Generate features with progressive loading if enabled
        if progressive_loading and len(generators_to_use) > 10:  # Only use progressive loading for large feature sets
            feature_df = self._generate_features_progressive(generators_to_use, data, batch_size, **kwargs)
            results = []  # Progressive loading doesn't return individual results
        else:
            # Generate features normally
            results = self._generate_features_parallel(generators_to_use, data, **kwargs)
            # Combine results
            feature_df = self._combine_results(results, data.index)
            # Proactively release memory held by individual results now that we have the combined frame
            try:
                results.clear()
                import gc as _gc
                _gc.collect()
            except Exception:
                pass

        # Apply automatic normalization if enabled
        if self.auto_normalize and not feature_df.empty:
            feature_df = self._apply_automatic_normalization(feature_df, categories)

        # Update performance stats
        generation_time = time.time() - start_time
        self._update_performance_stats(generation_time, len(results), categories)

        self.logger.info(f"✅ Feature generation completed in {generation_time:.3f}s")
        self.logger.info(f"📊 Generated {len(feature_df.columns)} features")

        return feature_df

    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the optimization systems.
        
        Returns:
            Dictionary of optimization statistics
        """
        stats = {
            'feature_bank': self.performance_stats,
            'memory_pool': self.memory_pool.get_stats(),
            'vectorbt_batcher': {
                'queue_size': len(self.vectorbt_batcher.operations_queue),
                'cache_size': len(self.vectorbt_batcher.results_cache)
            }
        }
        
        return stats

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

            # Apply normalization using UnifiedVectorizationManager if available
            if self.unified_vectorization_manager:
                from src.utils.ml_common.unified_vectorization_manager import OperationType
                transformations = [{
                    'type': self.normalization_method,
                    'params': {'columns': features_to_normalize}
                }]

                # Use the correct method name
                optimization_result = self.unified_vectorization_manager.optimize_operation(
                    operation_type=OperationType.NORMALIZATION,
                    data=normalized_df,
                    transformations=transformations
                )
                # Extract the actual DataFrame from the OptimizationResult
                normalized_df = optimization_result.result

                self.performance_stats['normalization_applied'] += 1
                self.performance_stats['vectorization_optimizations'] += 1

            else:
                # Fallback to manual normalization
                for feature in features_to_normalize:
                    if feature in normalized_df.columns:
                        if self.normalization_method == 'zscore':
                            mean_val = normalized_df[feature].mean()
                            std_val = normalized_df[feature].std()
                            if std_val > 0:  # Check for division by zero
                                normalized_df[feature] = (normalized_df[feature] - mean_val) / std_val
                            else:
                                self.logger.warning(f"⚠️ Standard deviation is zero for feature {feature}, skipping z-score normalization")

                        elif self.normalization_method == 'minmax':
                            min_val = normalized_df[feature].min()
                            max_val = normalized_df[feature].max()
                            if max_val > min_val:  # Check for division by zero
                                normalized_df[feature] = (normalized_df[feature] - min_val) / (max_val - min_val)
                            else:
                                self.logger.warning(f"⚠️ Min and max values are equal for feature {feature}, skipping minmax normalization")

                        elif self.normalization_method == 'robust':
                            median_val = normalized_df[feature].median()
                            mad_val = (normalized_df[feature] - median_val).abs().median()
                            if mad_val > 0:  # Check for division by zero
                                normalized_df[feature] = (normalized_df[feature] - median_val) / mad_val
                            else:
                                self.logger.warning(f"⚠️ MAD value is zero for feature {feature}, skipping robust normalization")

                self.performance_stats['normalization_applied'] += 1

            self.logger.info(f"✅ Normalization applied to {len(features_to_normalize)} features")

        except Exception as e:
            import traceback
            self.logger.error(f"Error applying automatic normalization: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
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
                if feature_category is not None:
                    # Check if feature_category is actually a FeatureCategory enum
                    if isinstance(feature_category, FeatureCategory):
                        category_str = feature_category.value
                    else:
                        # If it's a string, use it directly
                        category_str = str(feature_category)
                    
                    if category_str in self.normalization_config['exclude_categories']:
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

    def _create_feature_generator(self, config: FeatureConfig) -> Optional[FeatureGenerator]:
        """
        Create a feature generator from a configuration.
        
        Args:
            config: Feature configuration
            
        Returns:
            Feature generator instance or None if creation fails
        """
        try:
            # For individual feature generation, we don't create new generators
            # Instead, we should use the existing generators from the registry
            # This method is not the right approach for generating individual features
            # The system should use generate_features() with the appropriate category instead
            
            # Return None to indicate that individual feature generation via this path is not supported
            self.logger.debug(f"Individual feature generation for {config.name} should use generate_features() instead")
            return None
        except Exception as e:
            self.logger.error(f"Error creating feature generator for {config.name}: {e}")
            return None

    def generate_single_feature(self, data: pd.DataFrame, config: FeatureConfig) -> Optional[pd.Series]:
        """
        Generate a single feature using the specified configuration.
        
        Args:
            data: Input data DataFrame
            config: Feature configuration
            
        Returns:
            Generated feature as a pandas Series, or None if generation fails
        """
        try:
            # Individual feature generation is not supported through this path
            # Features should be generated through generate_features() with the appropriate category
            # This method is kept for backward compatibility but returns None
            self.logger.debug(f"Individual feature generation for {config.name} not supported - use generate_features() instead")
            return None
                
        except Exception as e:
            self.logger.error(f"Error generating single feature {config.name}: {e}")
            return None

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
            self.logger.info(f"🔍 Processing {len(categories)} categories: {categories}")
            for category in categories:
                if isinstance(category, str):
                    try:
                        category_enum = FeatureCategory(category)
                        self.logger.info(f"🔍 Converting string '{category}' to enum: {category_enum}")
                    except ValueError:
                        self.logger.warning(f"Invalid category: {category}")
                        continue
                else:
                    category_enum = category
                    self.logger.info(f"🔍 Using enum category: {category_enum}")

                # Skip problematic categories that are already excluded from lookback optimization
                if self._should_exclude_category(category_enum):
                    self.logger.info(f"🔍 Skipping excluded category: {category_enum}")
                    if isinstance(category_enum, FeatureCategory):
                        tprint(f"⏭️  Skipping category: {category_enum.value} (excluded)")
                    else:
                        tprint(f"⏭️  Skipping category: {category_enum} (excluded)")
                    continue

                category_generators = self.get_generators_by_category(category_enum)
                if isinstance(category_enum, FeatureCategory):
                    tprint(f"📦 Starting category: {category_enum.value} ({len(category_generators)} features)")
                else:
                    tprint(f"📦 Starting category: {category_enum} ({len(category_generators)} features)")
                self.logger.info(f"🔍 Found {len(category_generators)} generators for category {category_enum}")
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

        # FIXED: Do NOT exclude regime-specific generators when regime features are enabled
        # These generators are critical for regime detection models
        if 'regime_' in generator_name and not self.config.enable_regime_features:
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
        # Exclude removed categories (matching the exclusions from feature generation step)
        excluded_categories = {
            FeatureCategory.AUTOENCODER,
            FeatureCategory.REPRESENTATION_LEARNING,
            FeatureCategory.TIME,
            # FeatureCategory.REGIME,  # REMOVED: Enable regime features for regime models training
            FeatureCategory.NORMALIZATION,  # Not a feature category, it's a transform
            # Additional exclusions as specified in feature generation step
            FeatureCategory.ORDER_FLOW,
            # FeatureCategory.MICROSTRUCTURE,  # ENABLED: Microstructure features now enabled (no orderbook dependency)
            # FeatureCategory.ADVANCED_STATISTICAL,  # REMOVED: Allow advanced statistical features
            # Exclude empty categories that have 0 generators but still consume processing time
            FeatureCategory.CROSS_TIMEFRAME,  # 0 generators but still processed
            FeatureCategory.LEGACY,           # 0 generators but still processed
            FeatureCategory.CUSTOM            # 0 generators but still processed
        }
        
        if category in excluded_categories:
            return True

        return False

    def _optimize_lookbacks(self,
                           generators: List[FeatureGenerator],
                           data: pd.DataFrame,
                           target_column: str,
                           analyst_signals: Optional[pd.Series] = None,
                           regime_series: Optional[pd.Series] = None) -> List[FeatureGenerator]:
        """
        Optimize lookback periods for generators using appropriate optimizer based on mode.

        Args:
            generators: List of generators
            data: Input data
            target_column: Target column for optimization
            analyst_signals: Optional analyst signals for complementary scoring (Tactician mode)
            regime_series: Optional regime assignments as pd.Series for regime-invariant optimization (Tactician mode)

        Returns:
            List of generators with optimized lookbacks
        """
        if not self.lookback_optimizer:
            return generators

        # Detect optimizer type
        optimizer_type = type(self.lookback_optimizer).__name__
        
        if optimizer_type == 'ComplementaryLookbackOptimizer':
            self.logger.info("🔧 Optimizing lookback periods using complementary scoring (Tactician mode)...")
        else:
            self.logger.info("🔧 Optimizing lookback periods using standard optimization (Analyst mode)...")

        optimized_generators = []
        for generator in generators:
            if generator.supports_lookback_optimization():
                try:
                    # Use appropriate optimization method based on optimizer type
                    if optimizer_type == 'ComplementaryLookbackOptimizer':
                        # Use complementary scoring for Tactician mode
                        optimal_lookback = self.lookback_optimizer.optimize_lookback(
                            generator, data, target_column, analyst_signals, regime_series
                        )
                    else:
                        # Use standard optimization for Analyst mode
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

    def _prepare_generator_kwargs_with_lookback(self,
                                               generator: FeatureGenerator,
                                               base_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare per-generator kwargs, applying per-feature lookback overrides when available.

        This uses the per_feature_lookbacks mapping (final feature name -> [lb1, lb2, ...])
        to override the generator's default lookback and to pass explicit lookback-related
        parameters down to the generator in a generic way.
        """
        if not base_kwargs:
            return {}

        kwargs = dict(base_kwargs)
        per_feature_lookbacks = kwargs.get('per_feature_lookbacks')
        if not isinstance(per_feature_lookbacks, dict):
            return kwargs

        name = getattr(getattr(generator, 'config', None), 'name', None)
        if not name or name not in per_feature_lookbacks:
            return kwargs

        lookbacks = per_feature_lookbacks.get(name)
        if not isinstance(lookbacks, (list, tuple)) or not lookbacks:
            return kwargs

        try:
            optimal_lookback = int(lookbacks[0])
        except Exception:
            return kwargs

        # Update generator config bounds defensively so generators that rely on
        # default_lookback / min_lookback / max_lookback can pick up the override.
        try:
            if hasattr(generator, 'config'):
                if hasattr(generator.config, 'default_lookback'):
                    generator.config.default_lookback = optimal_lookback
                if hasattr(generator.config, 'min_lookback'):
                    try:
                        current_min = int(getattr(generator.config, 'min_lookback', optimal_lookback))
                    except Exception:
                        current_min = optimal_lookback
                    generator.config.min_lookback = min(current_min, optimal_lookback)
                if hasattr(generator.config, 'max_lookback'):
                    try:
                        current_max = int(getattr(generator.config, 'max_lookback', optimal_lookback))
                    except Exception:
                        current_max = optimal_lookback
                    generator.config.max_lookback = max(current_max, optimal_lookback)
        except Exception:
            # Config updates are best-effort; generators that cannot be updated
            # will simply ignore the override and rely on kwargs instead.
            pass

        # Expose lookback parameters to generators that want to consume them explicitly
        kwargs.setdefault('lookback', optimal_lookback)
        kwargs.setdefault('lookback_period', optimal_lookback)
        kwargs.setdefault('lookback_periods', lookbacks)

        # Keep a per-feature mapping available for any downstream components that
        # want the full triple per feature.
        optimized_map = kwargs.get('optimized_lookbacks_per_feature') or {}
        optimized_map = dict(optimized_map)
        optimized_map[name] = list(lookbacks)
        kwargs['optimized_lookbacks_per_feature'] = optimized_map

        return kwargs

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
        total_generators = len(generators)
        processed_generators = 0

        for generator in generators:
            processed_generators += 1
            
            # Progress update every 10 generators
            if processed_generators % 10 == 0 or processed_generators == total_generators:
                progress_pct = (processed_generators / total_generators) * 100
                tprint(f"🔄 Processing generator {processed_generators}/{total_generators} ({progress_pct:.1f}%): {generator.config.name}")
                self.logger.info(f"🔄 Processing generator {processed_generators}/{total_generators} ({progress_pct:.1f}%): {generator.config.name}")
            
            try:
                # Prepare per-generator kwargs with any per-feature lookback overrides
                generator_kwargs = self._prepare_generator_kwargs_with_lookback(generator, kwargs)

                # Check cache first (exclude unhashable entries like mapping dicts)
                cache_key_kwargs = {
                    k: v for k, v in generator_kwargs.items()
                    if k not in ('per_feature_lookbacks', 'optimized_lookbacks_per_feature')
                }
                cache_key = self._get_cache_key(generator, data, **cache_key_kwargs)
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
                result = generator.generate(data, **generator_kwargs)
                results.append(result)

                # Cache result with size management
                if self.feature_cache:
                    self._add_to_cache(cache_key, result)

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
        # Choose workers adaptively based on CPU and workload size
        try:
            import multiprocessing as _mp
            cpu_workers = max(1, (_mp.cpu_count() or 2) - 0)
        except Exception:
            cpu_workers = 2
        max_workers = min(self.config.max_workers, cpu_workers, max(1, len(generators)))

        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
            # Prepare per-generator kwargs with any per-feature lookback overrides
            generator_kwargs = self._prepare_generator_kwargs_with_lookback(generator, kwargs)

            # Check cache first (exclude unhashable entries like mapping dicts)
            cache_key_kwargs = {
                k: v for k, v in generator_kwargs.items()
                if k not in ('per_feature_lookbacks', 'optimized_lookbacks_per_feature')
            }
            cache_key = self._get_cache_key(generator, data, **cache_key_kwargs)
            if self.feature_cache and cache_key in self.feature_cache:
                cached_result = self.feature_cache[cache_key]
                if self.persist_generator_state:
                    self._store_generator_state(generator, self._extract_state_from_result(generator, cached_result))
                return cached_result

            state_payload = self._load_generator_state(generator)
            if state_payload:
                generator.load_state(state_payload)

            # Generate feature
            result = generator.generate(data, **generator_kwargs)

            # Cache result with size management
            if self.feature_cache:
                self._add_to_cache(cache_key, result)

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

    def _generate_features_progressive(self,
                                     generators: List[FeatureGenerator],
                                     data: pd.DataFrame,
                                     batch_size: Optional[int] = None,
                                     **kwargs) -> pd.DataFrame:
        """
        Generate features using progressive loading in batches for memory efficiency.

        Args:
            generators: List of generators to process
            data: Input data
            batch_size: Number of features per batch (adaptive if None)
            **kwargs: Additional parameters

        Returns:
            Combined features DataFrame
        """
        total_generators = len(generators)
        
        # Determine adaptive batch size based on available memory
        if batch_size is None:
            batch_size = self._calculate_adaptive_batch_size(total_generators, data)
        
        tprint(f"🔄 Progressive loading: processing {total_generators} generators in batches of {batch_size}")
        self.logger.info(f"🔄 Progressive loading: processing {total_generators} generators in batches of {batch_size}")
        
        all_feature_data = {}
        processed_count = 0
        
        # Process generators in batches
        for i in range(0, total_generators, batch_size):
            batch_generators = generators[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_generators + batch_size - 1) // batch_size
            
            tprint(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch_generators)} generators)")
            self.logger.info(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch_generators)} generators)")
            
            try:
                # Generate features for this batch
                batch_results = self._generate_features_parallel(batch_generators, data, **kwargs)
                
                # Extract feature data from batch results
                for result in batch_results:
                    if result.success:
                        all_feature_data[result.name] = result.data
                        processed_count += 1
                
                # Memory cleanup after each batch
                self._cleanup_batch_memory(batch_results)
                
                # Increment batch counter
                self.performance_stats['batch_count'] += 1
                
                # Periodic cleanup every N batches
                if self.performance_stats['batch_count'] % self.cleanup_frequency == 0:
                    self._perform_periodic_cleanup()
                
                # Progress update
                progress_pct = (processed_count / total_generators) * 100
                tprint(f"✅ Batch {batch_num} completed. Progress: {processed_count}/{total_generators} ({progress_pct:.1f}%)")
                
            except Exception as e:
                self.logger.error(f"Error processing batch {batch_num}: {e}")
                tprint(f"⚠️ Batch {batch_num} failed: {e}")
                continue
        
        # Create final DataFrame
        if all_feature_data:
            feature_df = pd.DataFrame(all_feature_data, index=data.index)
            tprint(f"✅ Progressive loading completed: {len(feature_df.columns)} features generated")
            return feature_df
        else:
            tprint("⚠️ No features were successfully generated")
            return pd.DataFrame(index=data.index)
    
    def _calculate_adaptive_batch_size(self, total_generators: int, data: pd.DataFrame) -> int:
        """
        Calculate adaptive batch size based on available memory and data size.
        
        Args:
            total_generators: Total number of generators
            data: Input data to estimate memory requirements
            
        Returns:
            Optimal batch size
        """
        # Base batch size
        base_batch_size = 20
        
        # Adjust based on data size
        data_size_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
        if data_size_mb > 100:  # Large datasets
            base_batch_size = min(base_batch_size, 10)
        elif data_size_mb > 50:  # Medium datasets
            base_batch_size = min(base_batch_size, 15)
        
        # Adjust based on total generators
        if total_generators > 200:
            base_batch_size = min(base_batch_size, 15)
        elif total_generators > 100:
            base_batch_size = min(base_batch_size, 20)
        
        # Ensure reasonable bounds
        batch_size = max(5, min(base_batch_size, total_generators))
        
        self.logger.info(f"📊 Adaptive batch size: {batch_size} (data: {data_size_mb:.1f}MB, generators: {total_generators})")
        return batch_size
    
    def _cleanup_batch_memory(self, batch_results: List[FeatureResult]) -> None:
        """
        Clean up memory after processing a batch.
        
        Args:
            batch_results: Results from the batch to clean up
        """
        # Clear ALL intermediate results, not just large ones
        for result in batch_results:
            if hasattr(result, 'data'):
                # Clear all result data, regardless of size
                del result.data
            
            # Clear any other attributes that might hold references
            if hasattr(result, 'metadata'):
                del result.metadata
            if hasattr(result, 'config'):
                del result.config
        
        # Clear the batch results list itself
        batch_results.clear()
        
        # Force garbage collection multiple times for better cleanup
        import gc
        for _ in range(3):
            gc.collect()
        
        # Log memory cleanup
        self.logger.debug(f"🧹 Batch memory cleanup completed: {len(batch_results)} results cleaned")
    
    def _perform_periodic_cleanup(self) -> None:
        """Perform periodic memory cleanup every N batches."""
        try:
            # Clear feature cache if it's getting large
            if self.feature_cache and len(self.feature_cache) > self.max_cache_size * 0.8:
                self.clear_cache()
                self.logger.info("🧹 Periodic cleanup: Feature cache cleared")
            
            # Clear VectorBT batcher queue
            if hasattr(self, 'vectorbt_batcher') and self.vectorbt_batcher:
                self.vectorbt_batcher.operations_queue.clear()
                self.vectorbt_batcher.results_cache.clear()
            
            # Force garbage collection
            import gc
            collected = 0
            for _ in range(3):
                collected += gc.collect()
            
            # Clear memory usage history in components
            if hasattr(self, 'vectorbt_rolling_optimizer') and self.vectorbt_rolling_optimizer:
                if hasattr(self.vectorbt_rolling_optimizer, '_memory_usage_history'):
                    self.vectorbt_rolling_optimizer._memory_usage_history = self.vectorbt_rolling_optimizer._memory_usage_history[-10:]
            
            self.performance_stats['periodic_cleanups'] += 1
            self.logger.info(f"🧹 Periodic cleanup completed: {collected} objects collected, batch #{self.performance_stats['batch_count']}")
            
        except Exception as e:
            self.logger.error(f"Periodic cleanup failed: {e}")
    
    def _batch_vectorbt_operations(self, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Batch multiple VectorBT operations for improved performance.
        
        Args:
            operations: List of operation dictionaries with 'name', 'func', 'args', 'kwargs'
            
        Returns:
            Dictionary of operation results
        """
        if not operations:
            return {}
        
        # Add operations to the batcher
        for op in operations:
            self.vectorbt_batcher.add_operation(
                op['name'],
                op['func'],
                *op.get('args', ()),
                priority=op.get('priority', 0),
                memory_weight=op.get('memory_weight', 1.0),
                **op.get('kwargs', {})
            )
        
        # Execute the batch
        return self.vectorbt_batcher.execute_batch()
    
    def _optimize_memory_usage(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize memory usage of a DataFrame using the memory pool.
        
        Args:
            data: DataFrame to optimize
            
        Returns:
            Memory-optimized DataFrame
        """
        # Use memory pool context manager for temporary operations
        with self.memory_pool.get_dataframe(data.shape[0], data.shape[1]) as temp_df:
            # Copy data to optimized DataFrame
            temp_df = data.copy()
            
            # Apply memory optimizations
            temp_df = self.memory_pool._optimize_dataframe_memory(temp_df)
            
            return temp_df.copy()

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
        features_by_category = {}

        for result in results:
            if result.success:
                feature_data[result.name] = result.data
                successful_features += 1
                
                # Group features by category for logging
                if hasattr(result.config, 'category'):
                    if hasattr(result.config.category, 'value'):
                        category = result.config.category.value
                    else:
                        category = str(result.config.category)
                else:
                    category = 'unknown'
                if category not in features_by_category:
                    features_by_category[category] = []
                features_by_category[category].append(result.name)
            else:
                self.logger.warning(f"Feature {result.name} failed: {result.error_message}")

        if not feature_data:
            self.logger.warning("No features were successfully generated")
            return pd.DataFrame(index=index)

        # Log features by category with progress updates
        total_features = len(feature_data)
        processed_categories = 0
        total_categories = len(features_by_category)
        
        for category, feature_names in features_by_category.items():
            feature_count = len(feature_names)
            # Only show first 3 feature names to reduce verbosity
            feature_list = ', '.join(feature_names[:3])
            if len(feature_names) > 3:
                feature_list += f" (+{len(feature_names)-3} more)"
            processed_categories += 1
            
            # Progress update every category (reduced verbosity)
            progress_pct = (processed_categories / total_categories) * 100
            tprint(f"📊 [{processed_categories}/{total_categories}] ({progress_pct:.1f}%) Generated {feature_count} {category} features: {feature_list}")
            self.logger.info(f"📊 [{processed_categories}/{total_categories}] ({progress_pct:.1f}%) Generated {feature_count} {category} features: {feature_list}")
            
            # Additional progress update every 10 categories (reduced frequency)
            if processed_categories % 10 == 0:
                tprint(f"🔄 Progress: {processed_categories}/{total_categories} categories processed ({progress_pct:.1f}%)")
                self.logger.info(f"🔄 Progress: {processed_categories}/{total_categories} categories processed ({progress_pct:.1f}%)")

        feature_df = pd.DataFrame(feature_data, index=index)

        # Downcast numeric columns to float32 to reduce memory footprint
        try:
            num_cols = feature_df.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                feature_df[num_cols] = feature_df[num_cols].astype(np.float32, copy=False)
        except Exception:
            # Keep robust if downcast fails for any reason
            pass
        
        # Final summary
        tprint(f"🎉 FEATURE GENERATION COMPLETE!")
        tprint(f"📊 Total features generated: {successful_features}/{len(results)}")
        tprint(f"📊 Categories processed: {total_categories}")
        tprint(f"📊 Data shape: {feature_df.shape}")
        tprint(f"📊 Memory usage: {feature_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        self.logger.info(f"✅ Successfully generated {successful_features}/{len(results)} features")
        self.logger.info(f"📊 Final data shape: {feature_df.shape}")
        self.logger.info(f"📊 Memory usage: {feature_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

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
        # Fast, in-process cache key based on generator name, data id, and hashable parameters
        # Using id(data) avoids expensive per-call tuple materialization and is stable within a run
        data_hash = hash((data.shape, id(data)))

        # Only include hashable (key, value) pairs to avoid TypeError when values are
        # complex objects like dicts or lists (e.g. optimized lookback mappings).
        safe_items = []
        for item in sorted(kwargs.items()):
            try:
                hash(item)
            except TypeError:
                continue
            else:
                safe_items.append(item)

        try:
            params_hash = hash(tuple(safe_items))
        except Exception:
            params_hash = 0

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
                # Only add if it's a FeatureCategory enum
                if isinstance(category, FeatureCategory):
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

    def _add_to_cache(self, cache_key: str, result: Any) -> None:
        """Add result to cache with size management."""
        if not self.feature_cache:
            return
        
        # Check cache size limits
        self._enforce_cache_limits()
        
        # Add to cache
        self.feature_cache[cache_key] = result
        
        # Log cache size periodically
        if len(self.feature_cache) % 100 == 0:
            self.logger.debug(f"📦 Cache size: {len(self.feature_cache)} entries")
    
    def _enforce_cache_limits(self) -> None:
        """Enforce cache size and memory limits."""
        if not self.feature_cache:
            return
        
        # Check entry count limit
        if len(self.feature_cache) >= self.max_cache_size:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self.feature_cache.keys())[:len(self.feature_cache) - self.max_cache_size + 100]
            for key in keys_to_remove:
                del self.feature_cache[key]
            self.logger.info(f"🧹 Cache size limit reached, removed {len(keys_to_remove)} entries")
        
        # Check memory limit
        try:
            import sys
            cache_memory_mb = sum(sys.getsizeof(v) for v in self.feature_cache.values()) / (1024 * 1024)
            if cache_memory_mb >= self.max_cache_memory_mb:
                # Remove half the cache entries
                keys_to_remove = list(self.feature_cache.keys())[:len(self.feature_cache) // 2]
                for key in keys_to_remove:
                    del self.feature_cache[key]
                self.logger.info(f"🧹 Cache memory limit reached ({cache_memory_mb:.1f}MB), removed {len(keys_to_remove)} entries")
        except Exception as e:
            self.logger.debug(f"Cache memory check failed: {e}")

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
            if isinstance(category, FeatureCategory):
                summary['categories'][category.value] = len(generators)
                summary['features_by_category'][category.value] = [
                    gen.config.name for gen in generators
                ]
            else:
                summary['categories'][str(category)] = len(generators)
                summary['features_by_category'][str(category)] = [
                    gen.config.name for gen in generators
                ]

        return summary

    def get_sr_levels(self, symbol: str = None, exchange: str = None, 
                     timeframe: str = None, direction: str = None) -> Dict[str, Any]:
        """
        Get SR levels dictionary from the artifact manager.
        
        This method provides access to the SR levels dictionary that was saved
        by the SR clustering component, making it available to feature generators
        and training scripts.
        
        Args:
            symbol: Trading symbol to filter by (optional)
            exchange: Exchange to filter by (optional)
            timeframe: Timeframe to filter by (optional)
            direction: Trading direction to filter by (optional)
            
        Returns:
            Dictionary containing SR levels with scores and metadata
        """
        try:
            from src.utils.artifact_manager import ArtifactManager
            
            # Initialize artifact manager
            artifact_manager = ArtifactManager(config={})
            
            # Set context for artifact retrieval (step_name is required)
            artifact_manager.set_context(
                step_name='feature_bank',
                symbol=symbol,
                exchange=exchange,
                direction=direction or 'long'
            )
            
            # Retrieve the SR levels dictionary
            sr_levels_dict = artifact_manager.get_artifact(
                artifact_name='sr_levels_dictionary',
                artifact_type='data'
            )
            
            # Handle case where artifact manager returns DataFrame instead of dict
            import pandas as pd
            if isinstance(sr_levels_dict, pd.DataFrame):
                self.logger.warning("SR levels returned as DataFrame, expected dict - skipping")
                return {
                    'levels': [],
                    'summary': {'total_levels': 0, 'total_clusters': 0},
                    'error': 'SR levels returned as DataFrame instead of dict'
                }
            
            if sr_levels_dict is None or (isinstance(sr_levels_dict, dict) and not sr_levels_dict):
                self.logger.warning("SR levels dictionary not found in artifacts")
                return {
                    'levels': [],
                    'summary': {'total_levels': 0, 'total_clusters': 0},
                    'error': 'SR levels dictionary not found'
                }
            
            # Apply filters if specified (only apply if any filter is a non-empty string)
            should_filter = any([
                symbol is not None and isinstance(symbol, str) and symbol != '',
                exchange is not None and isinstance(exchange, str) and exchange != '',
                timeframe is not None and isinstance(timeframe, str) and timeframe != '',
                direction is not None and isinstance(direction, str) and direction != ''
            ])
            
            if should_filter:
                filtered_levels = []
                for level in sr_levels_dict.get('levels', []):
                    level_metadata = level.get('metadata', {})
                    
                    # Check filters (only if they are valid strings)
                    if symbol and isinstance(symbol, str) and level_metadata.get('symbol') != symbol:
                        continue
                    if exchange and isinstance(exchange, str) and level_metadata.get('exchange') != exchange:
                        continue
                    if timeframe and isinstance(timeframe, str) and level_metadata.get('timeframe') != timeframe:
                        continue
                    if direction and isinstance(direction, str) and level_metadata.get('direction') != direction:
                        continue
                    
                    filtered_levels.append(level)
                
                # Update the dictionary with filtered levels
                sr_levels_dict = sr_levels_dict.copy()
                sr_levels_dict['levels'] = filtered_levels
                sr_levels_dict['summary']['total_levels'] = len(filtered_levels)
            
            self.logger.info(f"Retrieved SR levels dictionary with {len(sr_levels_dict.get('levels', []))} levels")
            return sr_levels_dict
            
        except Exception as e:
            self.logger.error(f"Failed to get SR levels: {e}")
            return {
                'levels': [],
                'summary': {'total_levels': 0, 'total_clusters': 0},
                'error': str(e)
            }

    def get_sr_levels_summary(self) -> Dict[str, Any]:
        """
        Get a summary of available SR levels without loading the full dictionary.
        
        Returns:
            Dictionary containing summary information about SR levels
        """
        try:
            sr_levels = self.get_sr_levels()
            return sr_levels.get('summary', {})
        except Exception as e:
            self.logger.error(f"Failed to get SR levels summary: {e}")
            return {'error': str(e)}

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
            if isinstance(category, FeatureCategory):
                tprint(f"📊 Category: {category.value}")
            else:
                tprint(f"📊 Category: {category}")

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
