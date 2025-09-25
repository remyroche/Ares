"""
TAS Adaptation Module

This module provides continuous adaptation capabilities for TAS architectures.
"""

from .continuous_adaptation_system import (
    ContinuousAdaptationSystem,
    ContinuousAdaptationConfig,
    RegimeChangeDetector,
    PerformanceAdaptationTrigger,
    CLVSAAdaptationEngine,
    AdaptationTrigger,
    AdaptationResult,
    create_continuous_adaptation_system,
    create_regime_change_detector,
    create_cvlsa_adaptation_engine
)

__all__ = [
    'ContinuousAdaptationSystem',
    'ContinuousAdaptationConfig',
    'RegimeChangeDetector',
    'PerformanceAdaptationTrigger',
    'CLVSAAdaptationEngine',
    'AdaptationTrigger',
    'AdaptationResult',
    'create_continuous_adaptation_system',
    'create_regime_change_detector',
    'create_cvlsa_adaptation_engine'
]