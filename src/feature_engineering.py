"""
Backwards Compatibility Wrapper for feature_engineering

This module provides backwards compatibility for code that imports from
the old 'feature_engineering' module, which has been renamed to
'feature_engineering_roadmap'.

DEPRECATED: Please update your imports to use feature_engineering_roadmap directly.

Old import:
    from feature_engineering.feature_registry import FeatureRegistry

New import:
    from src.feature_engineering_roadmap.feature_registry import FeatureRegistry

This wrapper will remain for backwards compatibility but may be removed in future versions.
"""

import warnings
import sys

# Show deprecation warning
warnings.warn(
    "Importing from 'feature_engineering' is deprecated. "
    "Please update to 'feature_engineering_roadmap'. "
    "This compatibility wrapper will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from feature_engineering_roadmap for backwards compatibility
try:
    from src.feature_engineering_roadmap.feature_registry import (
        FeatureRegistry,
        FeatureFamily,
        FeatureMetadata,
        PriceReturnsFeatures,
        VolatilityFeatures,
        MeanReversionFeatures,
        LiquidityMicroFeatures,
        AnchorsTODFeatures,
        ContextFeatures
    )
    
    from src.feature_engineering_roadmap.interactions import (
        InteractionType,
        InteractionConfig,
        RegimeFlags,
        InteractionEngine,
        create_default_interaction_config
    )
    
    from src.feature_engineering_roadmap.transforms import (
        TransformType,
        TransformConfig,
        OnlineEWZ,
        TODRank,
        SignedLog,
        MADScaler,
        Winsorization,
        TransformRouter,
        create_default_transform_config,
        apply_winsorization
    )
    
    from src.feature_engineering_roadmap.lookback_selection import (
        SelectionCriteria,
        LookbackChoice,
        LookbackMenu,
        LookbackSelector,
        LookbackOptimizer,
        create_feature_families
    )
    
    from src.feature_engineering_roadmap.assembly_dag import *
    from src.feature_engineering_roadmap.data_contracts import *
    from src.feature_engineering_roadmap.disagreement_meta_features import *
    from src.feature_engineering_roadmap.ensemble_meta_features import *
    
    __all__ = [
        # feature_registry
        'FeatureRegistry',
        'FeatureFamily',
        'FeatureMetadata',
        'PriceReturnsFeatures',
        'VolatilityFeatures',
        'MeanReversionFeatures',
        'LiquidityMicroFeatures',
        'AnchorsTODFeatures',
        'ContextFeatures',
        # interactions
        'InteractionType',
        'InteractionConfig',
        'RegimeFlags',
        'InteractionEngine',
        'create_default_interaction_config',
        # transforms
        'TransformType',
        'TransformConfig',
        'OnlineEWZ',
        'TODRank',
        'SignedLog',
        'MADScaler',
        'Winsorization',
        'TransformRouter',
        'create_default_transform_config',
        'apply_winsorization',
        # lookback
        'SelectionCriteria',
        'LookbackChoice',
        'LookbackMenu',
        'LookbackSelector',
        'LookbackOptimizer',
        'create_feature_families',
    ]

except ImportError as e:
    warnings.warn(
        f"Failed to import from feature_engineering_roadmap: {e}. "
        "Please ensure feature_engineering_roadmap module is available.",
        ImportWarning
    )
    raise
