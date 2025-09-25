"""
Evolutionary Architecture Search (EAS) for Neural Architecture Search (NAS).

This module provides a simplified interface to the shared evolutionary search utilities.
It imports and re-exports the main classes from the shared implementation.
"""

import warnings

warnings.warn(
    "nas_search.evolutionary_search is deprecated; use src.utils.nas_tas.evolutionary_search instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Import from shared utilities
from src.utils.nas_tas.evolutionary_search import (
    EvolutionaryTreeSearch,
    TreeGeneticAlgorithm,
    TreeNSGA2,
    EvolutionaryConfig
)

# Re-export for backward compatibility
__all__ = [
    'EvolutionaryTreeSearch',
    'TreeGeneticAlgorithm', 
    'TreeNSGA2',
    'EvolutionaryConfig'
]