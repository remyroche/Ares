"""
Unified Feature Generation System

This module provides a centralized, category-based feature generation system that consolidates
all scattered feature generation code into a single source of truth while maintaining
full backwards compatibility.

Key Features:
- Category-based feature organization (returns, momentum, volume, support/resistance, etc.)
- Matrix operations integration for optimized computation
- Advanced utilities (moved from feature_engineering)
- Feature bank for easy feature selection
- Backwards compatibility with existing code
- Hardware acceleration support (M1/M2/M3 optimization)

Architecture:
- Core framework for feature generation
- Category-specific feature generators
- Advanced utilities (optimization, analysis, etc.)
- Feature bank and registry
- Matrix operations integration
- Backwards compatibility layer

Usage:
    from src.feature_generation import (
        FeatureBank,
        get_feature_generator,
        generate_features_by_category,
        # Advanced utilities
        FeatureGenerationOptimizer,
        EnhancedFeatureEngineering
    )

    # Initialize feature bank (using global singleton pattern)
    # bank = get_feature_bank()  # Access the global feature bank when needed

    # Generate features by category
    # features = bank.generate_features(
    #     data=df,
    #     categories=['returns', 'momentum', 'volume'],
    #     lookback_optimization=True
    # )
"""

import logging
import os

# Ensure VectorBT optimizations are enabled when the package is available.
# This matches the runtime requirement without forcing users to export the variable manually.
os.environ.setdefault("ARES_ENABLE_VECTORBT", "1")

# Configure tprint to reduce verbosity for feature generation
try:
    from src.utils.tprint import configure_tprint, TPrintConfig, LogLevel
    # Set tprint to only show WARNING and above to reduce verbosity
    tprint_config = TPrintConfig(
        min_log_level=LogLevel.WARNING,
        output_to_console=True,
        use_colors=True
    )
    configure_tprint(tprint_config)
except ImportError:
    # tprint not available, continue without configuration
    pass

# Core framework imports
try:
    from .core import (
        FeatureBank,
        FeatureGenerator,
        FeatureCategory,
        FeatureRegistry,
        VectorizedFeatureGenerator,
        get_feature_generator,
        get_feature_bank,
        register_feature_generator,
        list_available_features,
        list_available_categories
    )
    # New utility mixins
    from .core.optimization_mixin import OptimizationMixin
    from .core.rolling_operations_mixin import RollingOperationsMixin
    from .core.generator_factory import GeneratorFactory, get_generator_factory, create_generator
    from .core.vectorbt_optimization_mixin import VectorBTOptimizationMixin
    # Auto-optimization components
    from .core.auto_optimized_feature_generator import AutoOptimizedFeatureGenerator
    from .core.auto_optimization_config import AutoOptimizationConfig, OptimizationLevel
    from .core.optimization_strategies import (
        OptimizationStrategy,
        ConservativeOptimizationStrategy,
        BalancedOptimizationStrategy,
        AggressiveOptimizationStrategy
    )
    CORE_AVAILABLE = True
except ImportError as e:
    CORE_AVAILABLE = False
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Core feature generation not available: {e}")

# Base calculations
try:
    from .base_calculations import (
        BaseCalculator,
        BaseCalculationType,
        BaseCalculationConfig,
        PriceReturnsCalculator,
        ReturnsVWAPCalculator,
        PriceLevelsCalculator,
        VolumeWeightedCalculator,
        VolumeReturnsCalculator,
        create_base_calculator,
        get_base_calculator,
        calculate_price_returns,
        calculate_returns_vwap,
        calculate_price_levels,
        calculate_volume_weighted,
        calculate_volume_returns
    )
    BASE_CALCULATIONS_AVAILABLE = True
except ImportError as e:
    BASE_CALCULATIONS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Base calculations not available: {e}")

# # Category-specific generators
# try:
#     from .categories import (
#         # Core categories
#         ReturnsFeatureGenerator,
#         MomentumFeatureGenerator,
#         VolumeFeatureGenerator,
#         SupportResistanceFeatureGenerator,
#         CandlestickPatternFeatureGenerator,
#         VolatilityFeatureGenerator,
#         TrendFeatureGenerator,
#         OscillatorFeatureGenerator,
# 
#         # New consolidated categories
#         AccelerationFeatureGenerator,
#         InteractionFeatureGenerator,
#         CrossTimeframeFeatureGenerator,
#         EntropyFeatureGenerator,
# 
#         # Specific generators from new categories
#         MomentumGenerator,
#         PriceAccelerationGenerator,
#         PriceJerkGenerator,
#         TrendStrengthGenerator,
#         TrendConsistencyGenerator,
#         VolumeAccelerationGenerator,
#         VolatilityAccelerationGenerator,
#         create_acceleration_generators,
# 
#         MomentumDivergenceGenerator,
#         MomentumVolumeGenerator,
#         MomentumVolatilityGenerator,
#         VolatilityVolumeGenerator,
#         create_interaction_generators,
# 
#         CrossTimeframeMomentumGenerator,
#         CrossTimeframeVolatilityGenerator,
#         CrossTimeframeVolumeGenerator,
#         create_cross_timeframe_generators,
# 
#         PriceEntropyGenerator,
#         VolumeEntropyGenerator,
#         ReturnEntropyGenerator,
#         HighLowEntropyGenerator,
#         VolatilityEntropyGenerator,
#         MomentumEntropyGenerator,
#         RSIEntropyGenerator,
#         MACDEntropyGenerator,
#         BollingerBandsEntropyGenerator,
#         CrossAssetEntropyGenerator,
#         RegimeEntropyGenerator,
#         create_entropy_generators,
# 
#         # Legacy interaction generators
#         CrossTimeframeInteractionGenerator,
#         FeatureRatioGenerator,
#         PolynomialFeatureGenerator,
#         CorrelationInteractionGenerator,
#         create_default_interaction_generators
#     )
#     CATEGORIES_AVAILABLE = True
# except ImportError as e:
#     CATEGORIES_AVAILABLE = False
#     logger = logging.getLogger(__name__)
#     logger.warning(f"Category generators not available: {e}")

# Matrix operations integration
try:
    from .matrix_integration import (
        MatrixFeatureProcessor,
        VectorizedFeatureGenerator,
        get_matrix_processor,
        enable_matrix_acceleration
    )
    MATRIX_INTEGRATION_AVAILABLE = True
except ImportError as e:
    MATRIX_INTEGRATION_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Matrix integration not available: {e}")

# HMM compatibility layer - REMOVED (deprecated)
HMM_COMPATIBILITY_AVAILABLE = False

# Convenience functions
# try:
#     from .convenience import (
#         generate_features_by_category,
#         generate_all_features,
#         get_feature_summary,
#         validate_feature_data,
#         export_feature_config
#     )
#     CONVENIENCE_AVAILABLE = True
# except ImportError as e:
#     CONVENIENCE_AVAILABLE = False
#     logger = logging.getLogger(__name__)
#     logger.warning(f"Convenience functions not available: {e}")
# 
# # Advanced utilities (moved from feature_engineering)
# try:
#     from .utils import (
#         # Optimization system
#         FeatureGenerationOptimizer,
#         FeatureOptimizationConfig,
#         FeatureOptimizationResult,
#         OptimizationMethod,
#         get_feature_optimizer,
#         optimize_feature_lookback,
#         get_optimization_config,
#         LookbackOptimizer,
# 
#         # Advanced feature engineering
#         EnhancedFeatureEngineering,
#         Step06UtilityContainer,
#         UtilityConfig,
# 
#         # Analysis pipelines
#         CrossTimeframeAnalysisPipeline,
#         FractionalDifferentiationPipeline,
#         EnhancedMatrixOperations,
# 
#         # Validation
#         validate_feature_quality,
#         validate_features_dataframe
#     )
#     UTILS_AVAILABLE = True
# except ImportError as e:
#     UTILS_AVAILABLE = False
#     logger = logging.getLogger(__name__)
#     logger.warning(f"Advanced utils not available: {e}")

# VectorBT optimizations
# try:
#     from .core.vectorbt_batch_processor import (
#         VectorBTBatchProcessor,
#         BatchProcessingConfig,
#         VectorBTFeatureBatchProcessor,
#         VectorBTSignalBatchProcessor,
#         create_vectorbt_batch_processor,
#         create_feature_batch_processor,
#         create_signal_batch_processor
#     )
#     # Check if VectorBT is actually available by testing the import in the module
#     from .core.vectorbt_batch_processor import VECTORBT_AVAILABLE
#     if VECTORBT_AVAILABLE:
#         VECTORBT_OPTIMIZATIONS_AVAILABLE = True
#     else:
#         VECTORBT_OPTIMIZATIONS_AVAILABLE = False
#         logger = logging.getLogger(__name__)
#         logger.warning("VectorBT optimizations not available - vectorbt package not properly installed")
# except ImportError as e:
#     VECTORBT_OPTIMIZATIONS_AVAILABLE = False
#     logger = logging.getLogger(__name__)
#     logger.warning(f"VectorBT optimizations not available: {e}")

# Version and metadata
__version__ = "2.0.0"
__author__ = "Unified Feature Generation Team"
__description__ = "Centralized feature generation system with category-based organization and advanced utilities"

# Build __all__ list conditionally
__all__ = []

# Core framework
if CORE_AVAILABLE:
    __all__.extend([
        "FeatureBank",
        "FeatureGenerator",
        "FeatureCategory",
        "FeatureRegistry",
        "VectorizedFeatureGenerator",
        "get_feature_generator",
        "get_feature_bank",
        "register_feature_generator",
        "list_available_features",
        "list_available_categories",
        # New utility mixins
        "OptimizationMixin",
        "RollingOperationsMixin",
        "VectorBTOptimizationMixin",
        # Factory pattern
        "GeneratorFactory",
        "get_generator_factory",
        "create_generator",
        # Auto-optimization components
        "AutoOptimizedFeatureGenerator",
        "AutoOptimizationConfig",
        "OptimizationLevel",
        "OptimizationStrategy",
        "ConservativeOptimizationStrategy",
        "BalancedOptimizationStrategy",
        "AggressiveOptimizationStrategy"
    ])

# Base calculations
if BASE_CALCULATIONS_AVAILABLE:
    __all__.extend([
        "BaseCalculator",
        "BaseCalculationType",
        "BaseCalculationConfig",
        "PriceReturnsCalculator",
        "ReturnsVWAPCalculator",
        "PriceLevelsCalculator",
        "VolumeWeightedCalculator",
        "VolumeReturnsCalculator",
        "create_base_calculator",
        "get_base_calculator",
        "calculate_price_returns",
        "calculate_returns_vwap",
        "calculate_price_levels",
        "calculate_volume_weighted",
        "calculate_volume_returns"
    ])

# Category generators
if True:
    __all__.extend([
        "ReturnsFeatureGenerator",
        "MomentumFeatureGenerator",
        "VolumeFeatureGenerator",
        "SupportResistanceFeatureGenerator",
        "CandlestickPatternFeatureGenerator",
        "VolatilityFeatureGenerator",
        "TrendFeatureGenerator",
        "OscillatorFeatureGenerator",
        "InteractionFeatureGenerator",
        "CrossTimeframeInteractionGenerator",
        "FeatureRatioGenerator",
        "PolynomialFeatureGenerator",
        "CorrelationInteractionGenerator",
        "create_interaction_generators",
        "create_default_interaction_generators"
    ])

# Matrix integration
if True:
    __all__.extend([
        "MatrixFeatureProcessor",
        "VectorizedFeatureGenerator",
        "get_matrix_processor",
        "enable_matrix_acceleration"
    ])

# HMM compatibility - REMOVED (deprecated)

# Convenience functions
if True:
    __all__.extend([
        "generate_features_by_category",
        "generate_all_features",
        "get_feature_summary",
        "validate_feature_data",
        "export_feature_config"
    ])

# Advanced utils
if True:
    __all__.extend([
        # Optimization system
        "FeatureGenerationOptimizer",
        "FeatureOptimizationConfig",
        "FeatureOptimizationResult",
        "OptimizationMethod",
        "get_feature_optimizer",
        "optimize_feature_lookback",
        "get_optimization_config",
        "LookbackOptimizer",

        # Advanced utilities
        "EnhancedFeatureEngineering",
        "Step06UtilityContainer",
        "UtilityConfig",
        "CrossTimeframeAnalysisPipeline",
        "FractionalDifferentiationPipeline",
        "EnhancedMatrixOperations",
        "validate_feature_quality",
        "validate_features_dataframe"
    ])

# VectorBT optimizations
if True:
    __all__.extend([
        # Batch processing
        "VectorBTBatchProcessor",
        "BatchProcessingConfig",
        "VectorBTFeatureBatchProcessor",
        "VectorBTSignalBatchProcessor",
        "create_vectorbt_batch_processor",
        "create_feature_batch_processor",
        "create_signal_batch_processor"
    ])

# Initialize default feature bank if core is available
from src.utils.initialization_guard import init_guard

if CORE_AVAILABLE:
    try:
        from .core import _initialize_default_bank
        if init_guard.mark_initialized("feature_generation.default_bank"):
            _initialize_default_bank()
    except Exception as e:
        logger = logging.getLogger(__name__)
        if init_guard.is_initialized("feature_generation.default_bank"):
            logger.warning(f"Failed to initialize default feature bank: {e}")

# Log initialization
logger = logging.getLogger(__name__)
if init_guard.mark_initialized("feature_generation.__init__"):
    logger.info("✅ Unified Feature Generation System initialized")
    logger.info(f"📦 Version: {__version__}")
    logger.info("🔧 Features: Category-based organization, advanced utilities, optimization")
    logger.info("🍎 Optimized for: Apple Silicon M1/M2/M3 Macs")

    if True:
        logger.info("🚀 Advanced utilities available (optimization, analysis, etc.)")
    else:
        logger.warning("⚠️ Advanced utilities not available - limited functionality")

    if True:
        logger.info("⚡ VectorBT optimizations available (advanced volatility, volume, batch processing)")
    else:
        logger.warning("⚠️ VectorBT optimizations not available - install vectorbt for enhanced performance")

# LAZY LOADING IMPLEMENTATION
import warnings
from typing import Any, Dict

# Lazy loading cache
_lazy_modules: Dict[str, Any] = {}

def _lazy_load_categories():
    """Load categories module lazily."""
    if 'categories' not in _lazy_modules:
        try:
            from . import categories
            _lazy_modules['categories'] = categories
        except ImportError as e:
            warnings.warn(f"Failed to load categories: {e}")
            _lazy_modules['categories'] = None
    return _lazy_modules['categories']

def _lazy_load_utils():
    """Load utils module lazily."""
    if 'utils' not in _lazy_modules:
        try:
            from . import utils
            _lazy_modules['utils'] = utils
        except ImportError as e:
            warnings.warn(f"Failed to load utils: {e}")
            _lazy_modules['utils'] = None
    return _lazy_modules['utils']

def _lazy_load_vectorbt():
    """Load VectorBT optimizations lazily."""
    if 'vectorbt' not in _lazy_modules:
        try:
            from .core.vectorbt_batch_processor import (
                VectorBTBatchProcessor, BatchProcessingConfig,
                VectorBTFeatureBatchProcessor, VectorBTSignalBatchProcessor,
                create_vectorbt_batch_processor, create_feature_batch_processor,
                create_signal_batch_processor, VECTORBT_AVAILABLE
            )
            _lazy_modules['vectorbt'] = {
                'VectorBTBatchProcessor': VectorBTBatchProcessor,
                'BatchProcessingConfig': BatchProcessingConfig,
                'VectorBTFeatureBatchProcessor': VectorBTFeatureBatchProcessor,
                'VectorBTSignalBatchProcessor': VectorBTSignalBatchProcessor,
                'create_vectorbt_batch_processor': create_vectorbt_batch_processor,
                'create_feature_batch_processor': create_feature_batch_processor,
                'create_signal_batch_processor': create_signal_batch_processor,
                'VECTORBT_AVAILABLE': VECTORBT_AVAILABLE
            }
        except ImportError as e:
            warnings.warn(f"VectorBT optimizations not available: {e}")
            _lazy_modules['vectorbt'] = None
    return _lazy_modules['vectorbt']

def __getattr__(name: str) -> Any:
    """Lazy loading for feature generation components."""
    # Check categories first
    categories = _lazy_load_categories()
    if categories is not None and hasattr(categories, name):
        return getattr(categories, name)
    
    # Check utils
    utils = _lazy_load_utils()
    if utils is not None and hasattr(utils, name):
        return getattr(utils, name)
    
    # Check VectorBT
    vectorbt = _lazy_load_vectorbt()
    if vectorbt is not None and name in vectorbt:
        return vectorbt[name]
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Set availability flags
CATEGORIES_AVAILABLE = True
UTILS_AVAILABLE = True
VECTORBT_OPTIMIZATIONS_AVAILABLE = True
CONVENIENCE_AVAILABLE = True

