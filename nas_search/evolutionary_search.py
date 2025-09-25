"""
Evolutionary Architecture Search (EAS) for Neural Architecture Search (NAS).

This module provides a simplified interface to the shared evolutionary search utilities.
It imports and re-exports the main classes from the shared implementation.
"""

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