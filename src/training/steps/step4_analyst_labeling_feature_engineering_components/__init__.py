# src/training/steps/step4_analyst_labeling_feature_engineering_components/__init__.py

"""
Step 4 Analyst Labeling Feature Engineering Components

This module contains optimized components for triple barrier labeling and feature engineering
used in step 4 of the training pipeline.
"""

from .optimized_triple_barrier_labeling import (
    OptimizedTripleBarrierLabeling,
    benchmark_triple_barrier_methods,
)

__all__ = [
    "OptimizedTripleBarrierLabeling",
    "benchmark_triple_barrier_methods",
]

# Version information
__version__ = "1.0.0"
__author__ = "Ares Trading System"
__description__ = "Optimized triple barrier labeling and feature engineering components for step 4"