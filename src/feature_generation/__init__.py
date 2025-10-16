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

    # Initialize feature bank
    bank = FeatureBank()

    # Generate features by category
    features = bank.generate_features(
        data=df,
        categories=['returns', 'momentum', 'volume'],
        lookback_optimization=True
    )
"""

import logging

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

# Category-specific generators
try:
    from .categories import (
        # Core categories
        ReturnsFeatureGenerator,
        MomentumFeatureGenerator,
        VolumeFeatureGenerator,
        SupportResistanceFeatureGenerator,
        CandlestickPatternFeatureGenerator,
        VolatilityFeatureGenerator,
        TrendFeatureGenerator,
        OscillatorFeatureGenerator,

        # New consolidated categories
        AccelerationFeatureGenerator,
        InteractionFeatureGenerator,
        CrossTimeframeFeatureGenerator,
        EntropyFeatureGenerator,

        # Specific generators from new categories
        MomentumGenerator,
        PriceAccelerationGenerator,
        PriceJerkGenerator,
        TrendStrengthGenerator,
        TrendConsistencyGenerator,
        VolumeAccelerationGenerator,
        VolatilityAccelerationGenerator,
        create_acceleration_generators,

        MomentumDivergenceGenerator,
        MomentumVolumeGenerator,
        MomentumVolatilityGenerator,
        VolatilityVolumeGenerator,
        create_interaction_generators,

        CrossTimeframeMomentumGenerator,
        CrossTimeframeVolatilityGenerator,
        CrossTimeframeVolumeGenerator,
        create_cross_timeframe_generators,

        PriceEntropyGenerator,
        VolumeEntropyGenerator,
        ReturnEntropyGenerator,
        HighLowEntropyGenerator,
        VolatilityEntropyGenerator,
        MomentumEntropyGenerator,
        RSIEntropyGenerator,
        MACDEntropyGenerator,
        BollingerBandsEntropyGenerator,
        CrossAssetEntropyGenerator,
        RegimeEntropyGenerator,
        create_entropy_generators,

        # Legacy interaction generators
        CrossTimeframeInteractionGenerator,
        FeatureRatioGenerator,
        PolynomialFeatureGenerator,
        CorrelationInteractionGenerator,
        create_default_interaction_generators
    )
    CATEGORIES_AVAILABLE = True
except ImportError as e:
    CATEGORIES_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Category generators not available: {e}")

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
try:
    from .convenience import (
        generate_features_by_category,
        generate_all_features,
        get_feature_summary,
        validate_feature_data,
        export_feature_config
    )
    CONVENIENCE_AVAILABLE = True
except ImportError as e:
    CONVENIENCE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Convenience functions not available: {e}")

# Advanced utilities (moved from feature_engineering)
try:
    from .utils import (
        # Optimization system
        FeatureGenerationOptimizer,
        FeatureOptimizationConfig,
        FeatureOptimizationResult,
        OptimizationMethod,
        get_feature_optimizer,
        optimize_feature_lookback,
        get_optimization_config,
        LookbackOptimizer,

        # Advanced feature engineering
        EnhancedFeatureEngineering,
        Step06UtilityContainer,
        UtilityConfig,

        # Analysis pipelines
        CrossTimeframeAnalysisPipeline,
        FractionalDifferentiationPipeline,
        EnhancedMatrixOperations,

        # Validation
        validate_feature_quality,
        validate_features_dataframe
    )
    UTILS_AVAILABLE = True
except ImportError as e:
    UTILS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Advanced utils not available: {e}")

# VectorBT optimizations
try:
    from .core.vectorbt_batch_processor import (
        VectorBTBatchProcessor,
        BatchProcessingConfig,
        VectorBTFeatureBatchProcessor,
        VectorBTSignalBatchProcessor,
        create_vectorbt_batch_processor,
        create_feature_batch_processor,
        create_signal_batch_processor
    )
    # Check if VectorBT is actually available by testing the import in the module
    from .core.vectorbt_batch_processor import VECTORBT_AVAILABLE
    if VECTORBT_AVAILABLE:
        VECTORBT_OPTIMIZATIONS_AVAILABLE = True
    else:
        VECTORBT_OPTIMIZATIONS_AVAILABLE = False
        logger = logging.getLogger(__name__)
        logger.warning("VectorBT optimizations not available - vectorbt package not properly installed")
except ImportError as e:
    VECTORBT_OPTIMIZATIONS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"VectorBT optimizations not available: {e}")

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
if CATEGORIES_AVAILABLE:
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
if MATRIX_INTEGRATION_AVAILABLE:
    __all__.extend([
        "MatrixFeatureProcessor",
        "VectorizedFeatureGenerator",
        "get_matrix_processor",
        "enable_matrix_acceleration"
    ])

# HMM compatibility - REMOVED (deprecated)

# Convenience functions
if CONVENIENCE_AVAILABLE:
    __all__.extend([
        "generate_features_by_category",
        "generate_all_features",
        "get_feature_summary",
        "validate_feature_data",
        "export_feature_config"
    ])

# Advanced utils
if UTILS_AVAILABLE:
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
if VECTORBT_OPTIMIZATIONS_AVAILABLE:
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
if CORE_AVAILABLE:
    try:
        from .core import _initialize_default_bank
        _initialize_default_bank()
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to initialize default feature bank: {e}")

# Log initialization
logger = logging.getLogger(__name__)
logger.info("✅ Unified Feature Generation System initialized")
logger.info(f"📦 Version: {__version__}")
logger.info("🔧 Features: Category-based organization, advanced utilities, optimization")
logger.info("🍎 Optimized for: Apple Silicon M1/M2/M3 Macs")

if UTILS_AVAILABLE:
    logger.info("🚀 Advanced utilities available (optimization, analysis, etc.)")
else:
    logger.warning("⚠️ Advanced utilities not available - limited functionality")

if VECTORBT_OPTIMIZATIONS_AVAILABLE:
    logger.info("⚡ VectorBT optimizations available (advanced volatility, volume, batch processing)")
else:
    logger.warning("⚠️ VectorBT optimizations not available - install vectorbt for enhanced performance")
