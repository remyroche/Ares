"""
Step08 Utilities Bank

This package contains all the utilities that were previously part of step08.
These utilities are now available as a bank of reusable components that can be
imported and used by other parts of the system without being part of the main
pipeline.

Available utilities:
- step08_advanced_feature_selection_wrapper: BaseStep wrapper for advanced feature selection
- step08_advanced_feature_selection: Main advanced feature selection implementation
- step08_advanced_feature_selection_per_regime: Regime-specific feature selection
- step08_optimized_*: Various optimized implementations
- step08_unified_*: Unified implementations with different approaches
- step08_regime_data_splitting: Regime-specific data splitting utilities

Usage:
    from src.utils.step08_utilities import (
        AdvancedFeatureSelectionStep,
        Step08AdvancedFeatureSelection,
        Step08AdvancedFeatureSelectionPerRegime
    )
"""

# Import main utility classes for easy access
from .step08_advanced_feature_selection_wrapper import (
    AdvancedFeatureSelectionStep
)

from .step08_advanced_feature_selection import (
    Step08AdvancedFeatureSelection
)

from .step08_advanced_feature_selection_per_regime import (
    Step08AdvancedFeatureSelectionPerRegime
)

# Import optimized implementations
from .step08_optimized_class import (
    Step08OptimizedClass
)

from .step08_optimized_execution import (
    Step08OptimizedExecution
)

from .step08_optimized_methods import (
    Step08OptimizedMethods
)

from .step08_optimized import (
    Step08Optimized
)

# Import unified implementations
from .step08_unified_class import (
    Step08UnifiedClass
)

from .step08_unified_complete import (
    Step08UnifiedComplete
)

from .step08_unified_final import (
    Step08UnifiedFinal
)

from .step08_unified_methods import (
    Step08UnifiedMethods
)

from .step08_unified_risk import (
    Step08UnifiedRisk
)

from .step08_unified import (
    Step08Unified
)

# Import regime-specific utilities
from .step08_regime_data_splitting import (
    Step08RegimeDataSplitting
)

__all__ = [
    'AdvancedFeatureSelectionStep',
    'Step08AdvancedFeatureSelection',
    'Step08AdvancedFeatureSelectionPerRegime',
    'Step08OptimizedClass',
    'Step08OptimizedExecution',
    'Step08OptimizedMethods',
    'Step08Optimized',
    'Step08UnifiedClass',
    'Step08UnifiedComplete',
    'Step08UnifiedFinal',
    'Step08UnifiedMethods',
    'Step08UnifiedRisk',
    'Step08Unified',
    'Step08RegimeDataSplitting'
]

__version__ = "1.0.0"
__author__ = "Ares Trading Bot Team"
__description__ = "Step08 Utilities Bank - Reusable components for advanced feature selection"