"""
Unified Feature Generation System

This module provides a centralized, category-based feature generation system that consolidates
all scattered feature generation code into a single source of truth while maintaining
full backwards compatibility.

Key Features:
- Category-based feature organization (returns, momentum, volume, support/resistance, etc.)
- Matrix operations integration for optimized computation
- Lookback optimization system
- Feature bank for easy feature selection
- Backwards compatibility with existing code
- Hardware acceleration support (M1/M2/M3 optimization)

Architecture:
- Core framework for feature generation
- Category-specific feature generators
- Lookback optimization system
- Feature bank and registry
- Matrix operations integration
- Backwards compatibility layer

Usage:
    from src.feature_generation import (
        FeatureBank,
        get_feature_generator,
        generate_features_by_category,
        optimize_feature_lookbacks
    )
    
    # Initialize feature bank
    bank = FeatureBank()
    
    # Generate features by category
    features = bank.generate_features(
        data=df,
        categories=['returns', 'momentum', 'volume'],
        lookback_optimization=True
    )
    
    # Get specific feature generator
    generator = get_feature_generator('momentum')
    momentum_features = generator.generate(df, lookback_periods=[5, 10, 20])
"""

# Core framework imports
try:
    from .core import (
        FeatureBank,
        FeatureGenerator,
        FeatureCategory,
        FeatureRegistry,
        get_feature_generator,
        get_feature_bank,
        register_feature_generator,
        list_available_features,
        list_available_categories
    )
    CORE_AVAILABLE = True
except ImportError as e:
    CORE_AVAILABLE = False
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Core feature generation not available: {e}")

# Category-specific generators
try:
    from .categories import (
        ReturnsFeatureGenerator,
        MomentumFeatureGenerator,
        VolumeFeatureGenerator,
        SupportResistanceFeatureGenerator,
        CandlestickPatternFeatureGenerator,
        HMMRegimeFeatureGenerator,
        VolatilityFeatureGenerator,
        TrendFeatureGenerator,
        OscillatorFeatureGenerator
    )
    CATEGORIES_AVAILABLE = True
except ImportError as e:
    CATEGORIES_AVAILABLE = False
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Category generators not available: {e}")

# Lookback optimization system
try:
    from .optimization import (
        LookbackOptimizer,
        FeatureOptimizationConfig,
        FeatureOptimizationResult,
        optimize_feature_lookbacks,
        get_optimization_config
    )
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    OPTIMIZATION_AVAILABLE = False
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Lookback optimization not available: {e}")

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
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Matrix integration not available: {e}")

# Backwards compatibility layer
try:
    from .compatibility import (
        LegacyFeatureAdapter,
        migrate_legacy_features,
        get_legacy_adapter,
        enable_legacy_compatibility
    )
    COMPATIBILITY_AVAILABLE = True
except ImportError as e:
    COMPATIBILITY_AVAILABLE = False
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Backwards compatibility not available: {e}")

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
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Convenience functions not available: {e}")

# Version and metadata
__version__ = "1.0.0"
__author__ = "Unified Feature Generation Team"
__description__ = "Centralized feature generation system with category-based organization"

# Build __all__ list conditionally
__all__ = []

# Always available
if CORE_AVAILABLE:
    __all__.extend([
        "FeatureBank",
        "FeatureGenerator", 
        "FeatureCategory",
        "FeatureRegistry",
        "get_feature_generator",
        "get_feature_bank",
        "register_feature_generator",
        "list_available_features",
        "list_available_categories"
    ])

# Category generators
if CATEGORIES_AVAILABLE:
    __all__.extend([
        "ReturnsFeatureGenerator",
        "MomentumFeatureGenerator", 
        "VolumeFeatureGenerator",
        "SupportResistanceFeatureGenerator",
        "CandlestickPatternFeatureGenerator",
        "HMMRegimeFeatureGenerator",
        "VolatilityFeatureGenerator",
        "TrendFeatureGenerator",
        "OscillatorFeatureGenerator"
    ])

# Optimization
if OPTIMIZATION_AVAILABLE:
    __all__.extend([
        "LookbackOptimizer",
        "FeatureOptimizationConfig",
        "FeatureOptimizationResult", 
        "optimize_feature_lookbacks",
        "get_optimization_config"
    ])

# Matrix integration
if MATRIX_INTEGRATION_AVAILABLE:
    __all__.extend([
        "MatrixFeatureProcessor",
        "VectorizedFeatureGenerator",
        "get_matrix_processor",
        "enable_matrix_acceleration"
    ])

# Backwards compatibility
if COMPATIBILITY_AVAILABLE:
    __all__.extend([
        "LegacyFeatureAdapter",
        "migrate_legacy_features",
        "get_legacy_adapter", 
        "enable_legacy_compatibility"
    ])

# Convenience functions
if CONVENIENCE_AVAILABLE:
    __all__.extend([
        "generate_features_by_category",
        "generate_all_features",
        "get_feature_summary",
        "validate_feature_data",
        "export_feature_config"
    ])

# Initialize default feature bank if core is available
if CORE_AVAILABLE:
    try:
        from .core import _initialize_default_bank
        _initialize_default_bank()
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to initialize default feature bank: {e}")

# Log initialization
import logging
logger = logging.getLogger(__name__)
logger.info("✅ Unified Feature Generation System initialized")
logger.info(f"📦 Version: {__version__}")
logger.info("🔧 Features: Category-based organization, matrix operations integration, lookback optimization")
logger.info("🍎 Optimized for: Apple Silicon M1/M2/M3 Macs")