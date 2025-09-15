"""
PID-Based Feature Generation Module

This module provides data-driven feature generation using Partial Information Decomposition (PID)
to create the most relevant interaction, polynomial, and cross-timeframe features.

Key Components:
- InteractionFeatureGenerator: Generates up to 100 interaction features using PID analysis
- PolynomialFeatureGenerator: Generates up to 50 polynomial features using PID analysis  
- CrossTimeframeFeatureGenerator: Generates up to 50 cross-timeframe features using PID analysis
- PIDBasedFeatureOrchestrator: Orchestrates all feature generation processes
- OptimizedLookbackIntegration: Integrates optimized lookback periods from feature_lookback_optimization

Author: Market Analysis Team
Version: 2.0.0
"""

from .interaction_feature_generator import InteractionFeatureGenerator, InteractionConfig
from .polynomial_feature_generator import PolynomialFeatureGenerator, PolynomialConfig
from .cross_timeframe_feature_generator import CrossTimeframeFeatureGenerator, CrossTimeframeConfig
from .pid_based_feature_orchestrator import PIDBasedFeatureOrchestrator, OrchestratorConfig
from .optimized_lookback_integration import OptimizedLookbackIntegration

__all__ = [
    'InteractionFeatureGenerator',
    'InteractionConfig', 
    'PolynomialFeatureGenerator',
    'PolynomialConfig',
    'CrossTimeframeFeatureGenerator', 
    'CrossTimeframeConfig',
    'PIDBasedFeatureOrchestrator',
    'OrchestratorConfig',
    'OptimizedLookbackIntegration'
]

__version__ = '2.0.0'