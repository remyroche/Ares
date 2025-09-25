#!/usr/bin/env python3
"""
NAS-TAS Unified Components Package - Enhanced with Comprehensive Features

This package provides unified components for both NAS and TAS systems,
organized into logical modules for better maintainability and organization.
Enhanced with comprehensive hardware management and merged evaluation capabilities.

Modules:
- evaluation: Enhanced unified evaluation framework with merged functionality
- hardware: Comprehensive hardware optimization with performance monitoring
- search: Search algorithms with Bayesian TPE integration
- data_processing: Unified data processing pipeline
- manager: Unified component manager
"""

from .evaluation import UnifiedEvaluator, compute_classification_metrics, compute_regression_metrics
from .hardware import (
    UnifiedHardwareOptimizer, HardwareConfig, PerformanceMetrics, 
    HardwarePerformanceMonitor, WorkloadType, OptimizationLevel
)
from .search import UnifiedSearchEngine
from .data_processing import UnifiedDataProcessor
from .manager import UnifiedComponentManager

__all__ = [
    # Main components
    'UnifiedEvaluator',
    'UnifiedHardwareOptimizer', 
    'UnifiedSearchEngine',
    'UnifiedDataProcessor',
    'UnifiedComponentManager',
    
    # Enhanced evaluation functions
    'compute_classification_metrics',
    'compute_regression_metrics',
    
    # Hardware management classes
    'HardwareConfig',
    'PerformanceMetrics',
    'HardwarePerformanceMonitor',
    'WorkloadType',
    'OptimizationLevel'
]

__version__ = "2.0.0"
__author__ = "AI Assistant"
__description__ = "Enhanced unified components with comprehensive hardware management and merged evaluation capabilities"
=======
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
