"""
Tree-Driven Advanced Statistics (TAS) Regime Detection System

This package provides a fully implemented, production-ready regime detection system
that combines tree-based learning with advanced statistical methods, CLVSA architecture,
and comprehensive tool integration.

Main components:
- TASRegimeConfig: Configuration for TAS regime detection
- TASRegimeDetector: Main regime detection system
- TASArchitectureType: Available architecture types
- TASOptimizationLevel: Optimization levels

Example usage:
    from tas_regime import TASRegimeConfig, TASRegimeDetector
    config = TASRegimeConfig.create_short_term_trading_config()
    detector = TASRegimeDetector(config)
    result = detector.detect_regimes(market_data, timestamps)
"""

from .core.tas_regime_config import TASRegimeConfig, TASArchitectureType, TASOptimizationLevel
from .core.tas_regime_detector import TASRegimeDetector, TASRegimeResult

__version__ = "1.0.0"
__author__ = "Ares Trading System"
__description__ = "Tree-Driven Advanced Statistics Regime Detection System"

__all__ = [
    'TASRegimeConfig',
    'TASRegimeDetector',
    'TASRegimeResult',
    'TASArchitectureType',
    'TASOptimizationLevel'
]