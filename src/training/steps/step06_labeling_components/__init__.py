from __future__ import annotations

# src/training/steps/step06_labeling_components/__init__.py

"""Step 06 Labeling Components.

This module contains the components for triple barrier labeling used in the training pipeline.
Feature engineering has been moved to separate components.
"""

from .fractional_triple_barrier_labeling import FractionalTripleBarrierLabeling
from .optimized_triple_barrier_labeling import OptimizedTripleBarrierLabeling

# Import regime-specific triple barrier optimizer
from .regime_specific_triple_barrier_optimizer import (
    RegimeSpecificTripleBarrierOptimizer,
    create_regime_specific_triple_barrier_optimizer,
)

__all__ = [
    "OptimizedTripleBarrierLabeling",
    "FractionalTripleBarrierLabeling",
    "RegimeSpecificTripleBarrierOptimizer",
    "create_regime_specific_triple_barrier_optimizer",
]

# Version information
__version__ = "1.0.0"
__author__ = "Ares Trading System"
__description__ = (
    "Optimized triple barrier labeling components for step 06"
)
