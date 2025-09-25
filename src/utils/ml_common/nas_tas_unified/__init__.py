"""
Unified NAS-TAS Regime Detection System

This module provides unified components for both NAS and TAS regime detection systems,
eliminating code duplication and providing consistent interfaces.
"""

from .unified_regime_detector import UnifiedRegimeDetector
from .unified_regime_config import UnifiedRegimeConfig, RegimeSystemType
from .unified_result import UnifiedRegimeResult

__all__ = [
    'UnifiedRegimeDetector',
    'UnifiedRegimeConfig', 
    'RegimeSystemType',
    'UnifiedRegimeResult'
]