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

try:
    from .core.hybrid_regime_detector import HybridNASTASRegimeDetector
except ImportError:  # pragma: no cover - optional dependency guard for tests
    HybridNASTASRegimeDetector = None

from .config.hybrid_regime_config import HybridRegimeConfig

try:
    from .integration.hybrid_orchestrator import HybridRegimeOrchestrator
except ImportError:  # pragma: no cover - optional dependency guard
    HybridRegimeOrchestrator = None

try:
    from .components.tas_integration import TASIntegrationComponent
except ImportError:  # pragma: no cover - optional dependency guard
    TASIntegrationComponent = None

try:
    from .components.nas_integration import NASIntegrationComponent
except ImportError:  # pragma: no cover - optional dependency guard
    NASIntegrationComponent = None

try:
    from .evaluation.economic_evaluator import EconomicRegimeEvaluator
except ImportError:  # pragma: no cover - optional dependency guard
    EconomicRegimeEvaluator = None

try:
    from .tagging.regime_tagger import RegimeTagger
except ImportError:  # pragma: no cover - optional dependency guard
    RegimeTagger = None

__all__ = [
    'HybridNASTASRegimeDetector',
    'HybridRegimeConfig',
    'HybridRegimeOrchestrator',
    'TASIntegrationComponent',
    'NASIntegrationComponent',
    'EconomicRegimeEvaluator',
    'RegimeTagger'
]
