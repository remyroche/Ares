"""
Tests for Hybrid NAS-TAS Regime System
"""

__all__ = []

try:  # pragma: no cover - optional test imports for environments with full dependencies
    from .test_hybrid_regime_detector import TestHybridRegimeDetector
    __all__.append('TestHybridRegimeDetector')
except ImportError:
    TestHybridRegimeDetector = None

try:
    from .test_tas_integration import TestTASIntegration
    __all__.append('TestTASIntegration')
except ImportError:
    TestTASIntegration = None

try:
    from .test_nas_integration import TestNASIntegration
    __all__.append('TestNASIntegration')
except ImportError:
    TestNASIntegration = None

try:
    from .test_economic_evaluator import TestEconomicEvaluator
    __all__.append('TestEconomicEvaluator')
except ImportError:
    TestEconomicEvaluator = None

try:
    from .test_regime_tagger import TestRegimeTagger
    __all__.append('TestRegimeTagger')
except ImportError:
    TestRegimeTagger = None

from .test_hybrid_orchestrator import TestHybridOrchestrator
__all__.append('TestHybridOrchestrator')
