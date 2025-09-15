"""
Unified Feature Generation System

This package provides a single source of truth for all feature generation
across the Ares trading system, with full backwards compatibility.

Key Components:
- FeatureGenerator: Base interface for all feature generators
- FeatureRegistry: Dynamic discovery and management of feature generators
- FeatureOrchestrator: Unified coordination of feature generation
- BackwardsCompatibilityLayer: Seamless migration from existing systems

Usage:
    from src.feature_engineering.unified import FeatureOrchestrator
    
    orchestrator = FeatureOrchestrator(config)
    features = await orchestrator.generate_features(data)
"""

from .core import (
    FeatureGenerator,
    FeatureGeneratorConfig,
    FeatureGenerationResult,
    FeatureCategory,
    FeaturePriority
)

from .registry import (
    FeatureRegistry,
    register_feature_generator,
    get_feature_generator,
    list_available_generators
)

from .orchestrator import (
    FeatureOrchestrator,
    OrchestrationConfig,
    FeaturePipeline
)

from .compatibility import (
    BackwardsCompatibilityLayer,
    LegacyFeatureAdapter
)

from .validation import (
    FeatureValidator,
    FeatureConsistencyChecker,
    FeatureQualityMetrics
)

__version__ = "1.0.0"
__all__ = [
    # Core interfaces
    "FeatureGenerator",
    "FeatureGeneratorConfig", 
    "FeatureGenerationResult",
    "FeatureCategory",
    "FeaturePriority",
    
    # Registry system
    "FeatureRegistry",
    "register_feature_generator",
    "get_feature_generator", 
    "list_available_generators",
    
    # Orchestration
    "FeatureOrchestrator",
    "OrchestrationConfig",
    "FeaturePipeline",
    
    # Compatibility
    "BackwardsCompatibilityLayer",
    "LegacyFeatureAdapter",
    
    # Validation
    "FeatureValidator",
    "FeatureConsistencyChecker", 
    "FeatureQualityMetrics"
]