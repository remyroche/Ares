"""
PID-Based Feature Generation Module

This module provides data-driven feature generation using Partial Information Decomposition (PID)
to create the most relevant interaction and cross-timeframe features.

Key Components:
- InteractionFeatureGenerator: Generates up to 100 interaction features using PID analysis
- CrossTimeframeFeatureGenerator: Generates up to 50 cross-timeframe features using PID analysis
- PIDBasedFeatureOrchestrator: Orchestrates all feature generation processes
- OptimizedLookbackIntegration: Integrates optimized lookback periods from feature_lookback_optimization

Author: Market Analysis Team
Version: 2.0.0
"""

from src.utils.tprint import tprint

tprint("🔧 Loading PID-based feature generation module...")

from .interaction_feature_generator import InteractionFeatureGenerator, InteractionConfig
# PolynomialFeatureGenerator removed due to empty except blocks - use feature_engineering bank instead
from .cross_timeframe_feature_generator import CrossTimeframeFeatureGenerator, CrossTimeframeConfig
from .pid_based_feature_orchestrator import PIDBasedFeatureOrchestrator, OrchestratorConfig
from .optimized_lookback_integration import OptimizedLookbackIntegration
from .feature_selection_mechanism import FeatureSelectionMechanism, FeatureSelectionConfig, SelectionStrategy

tprint("✅ PID-based feature generation module components loaded")

tprint("📋 Setting up module exports...")
__all__ = [
    'InteractionFeatureGenerator',
    'InteractionConfig', 
    'CrossTimeframeFeatureGenerator', 
    'CrossTimeframeConfig',
    'PIDBasedFeatureOrchestrator',
    'OrchestratorConfig',
    'OptimizedLookbackIntegration',
    'FeatureSelectionMechanism',
    'FeatureSelectionConfig',
    'SelectionStrategy'
]

__version__ = '2.0.0'
tprint("✅ PID-based feature generation module fully loaded")