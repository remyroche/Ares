"""
Search Space Optimization Framework for UnifiedDataDrivenPipeline

Provides comprehensive search space optimization to prevent explosion and
improve computational efficiency in feature engineering.

Key Features:
- Hereditary interactions (A×B only if A and B survive pre-selection)
- Advanced screening (HSIC, distance correlation)
- Collinearity filtering
- Condition number monitoring
- Aggressive pruning strategies
"""

from .hereditary_interactions import (
    HereditaryInteractionGenerator,
    HereditaryInteractionConfig,
    HereditaryInteractionResult,
    InteractionType
)

from .advanced_screening import (
    AdvancedScreeningFramework,
    AdvancedScreeningConfig,
    AdvancedScreeningResult,
    ScreeningMethod
)

from .collinearity_filter import (
    CollinearityFilter,
    CollinearityFilterConfig,
    CollinearityFilterResult,
    CollinearityMetric
)

from .search_space_pruner import (
    SearchSpacePruner,
    SearchSpacePrunerConfig,
    SearchSpacePrunerResult,
    PruningStrategy
)

__all__ = [
    # Hereditary interactions
    'HereditaryInteractionGenerator',
    'HereditaryInteractionConfig',
    'HereditaryInteractionResult',
    'InteractionType',
    
    # Advanced screening
    'AdvancedScreeningFramework',
    'AdvancedScreeningConfig',
    'AdvancedScreeningResult',
    'ScreeningMethod',
    
    # Collinearity filter
    'CollinearityFilter',
    'CollinearityFilterConfig',
    'CollinearityFilterResult',
    'CollinearityMetric',
    
    # Search space pruner
    'SearchSpacePruner',
    'SearchSpacePrunerConfig',
    'SearchSpacePrunerResult',
    'PruningStrategy'
]