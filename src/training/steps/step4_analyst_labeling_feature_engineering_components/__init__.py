# src/training/steps/step4_analyst_labeling_feature_engineering_components/__init__.py

"""Step 4 Analyst Labeling and Feature Engineering Components.

This module contains the components for triple barrier labeling and feature engineering
used in the analyst training pipeline.
"""

from .optimized_triple_barrier_labeling import OptimizedTripleBarrierLabeling
from .fractional_triple_barrier_labeling import FractionalTripleBarrierLabeling

__all__ = [
    "OptimizedTripleBarrierLabeling",
    "FractionalTripleBarrierLabeling",
]

# Version information
__version__ = "1.0.0"
__author__ = "Ares Trading System"
__description__ = "Optimized triple barrier labeling and feature engineering components for step 4"