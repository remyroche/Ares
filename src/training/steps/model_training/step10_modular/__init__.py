"""Step 10 Modular Unified Regime Intelligence System.

This is the modularized version of Step 10, broken down into focused,
maintainable components for better code organization and testing.
"""

from .orchestrator import UnifiedRegimeIntelligenceOrchestrator
from .config import Step10Config, DEFAULT_CONFIG
from .models import MultiTimeframeHMMEncoder

__version__ = "1.0.0"
__all__ = [
    'UnifiedRegimeIntelligenceOrchestrator',
    'Step10Config',
    'DEFAULT_CONFIG',
    'MultiTimeframeHMMEncoder',
]
