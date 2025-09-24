"""
Hybrid NAS TAS Regime Module

This module combines the outputs from NAS regime detection and TAS regime detection
to create a coherent regime modeling system with economic and financial relevance.

Key Features:
- Integrates TAS and NAS regime detection outputs
- Creates coherent regime modeling with economic significance
- Performs clustering based on combined TAS & NAS inputs
- Tags existing data with regime information
- Replaces hmm_clustering functionality

Superior Architecture:
- Core: Main hybrid regime detector
- Components: TAS and NAS integration components
- Evaluation: Economic and financial evaluation
- Integration: Main orchestrator (replaces HMM)
- Tagging: Data tagging and labeling functionality
- Config: Configuration management
- Tests: Comprehensive test suite
"""

# Main orchestrator (replaces HMM clustering)
from .integration.hybrid_orchestrator import HybridOrchestrator

# Core components
from .core.hybrid_regime_detector import HybridRegimeDetector

# Integration components
from .components.tas_integration import TASIntegration
from .components.nas_integration import NASIntegration

# Evaluation components
from .evaluation.economic_evaluator import EconomicEvaluator

# Tagging components
from .tagging.regime_tagger import RegimeTagger

# Configuration
from .config.hybrid_config import (
    HybridRegimeConfig, HybridNASConfig, HybridTASConfig,
    ClusteringMethod, IntegrationStrategy
)

__version__ = "1.0.0"
__author__ = "Hybrid NAS TAS Regime System"
__description__ = "Hybrid regime detection combining NAS and TAS with economic and financial relevance"

__all__ = [
    # Main orchestrator (HMM replacement)
    'HybridOrchestrator',
    
    # Core components
    'HybridRegimeDetector',
    
    # Integration components
    'TASIntegration',
    'NASIntegration',
    
    # Evaluation components
    'EconomicEvaluator',
    
    # Tagging components
    'RegimeTagger',
    
    # Configuration
    'HybridRegimeConfig',
    'HybridNASConfig',
    'HybridTASConfig',
    'ClusteringMethod',
    'IntegrationStrategy'
]