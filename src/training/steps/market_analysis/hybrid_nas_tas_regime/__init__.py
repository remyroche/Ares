"""
Hybrid NAS-TAS Regime System

This module replaces the HMM clustering system with a hybrid approach that combines:
- Neural Architecture Search (NAS) from nas_regime/
- Tree Architecture Search (TAS) from ml_common TAS system
- Economic and financial relevance evaluation
- Regime tagging for existing data

The system creates coherent regime modeling with economic significance and replaces
the HMM-based clustering entirely.
"""

from src.utils.nas_tas.config.hybrid_regime_config import HybridRegimeConfig
from src.utils.nas_tas.core.hybrid_regime_detector import HybridNASTASRegimeDetector
from src.training.steps.market_analysis.hybrid_nas_tas_regime.integration.hybrid_orchestrator import (
    HybridRegimeOrchestrator,
)
from src.training.steps.market_analysis.hybrid_nas_tas_regime.components.tas_integration import (
    TASIntegrationComponent,
)
from src.training.steps.market_analysis.hybrid_nas_tas_regime.components.nas_integration import (
    NASIntegrationComponent,
)
from src.training.steps.market_analysis.hybrid_nas_tas_regime.evaluation.economic_evaluator import (
    EconomicRegimeEvaluator,
)
from src.utils.nas_tas.tagging.regime_tagger import RegimeTagger

__all__ = [
    'HybridNASTASRegimeDetector',
    'HybridRegimeConfig',
    'HybridRegimeOrchestrator',
    'TASIntegrationComponent',
    'NASIntegrationComponent',
    'EconomicRegimeEvaluator',
    'RegimeTagger'
]