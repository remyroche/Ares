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
- regime_detector: Unified regime detection interface
"""

from .evaluation import UnifiedEvaluator, compute_classification_metrics, compute_regression_metrics
from .hardware import (
    UnifiedHardwareOptimizer, HardwareConfig, PerformanceMetrics, 
    HardwarePerformanceMonitor, WorkloadType, OptimizationLevel
)
from .search import UnifiedSearchEngine
from .data_processing import UnifiedDataProcessor
from .manager import UnifiedComponentManager

# Unified regime detection components
from .unified_regime_detector import UnifiedRegimeDetector
from .unified_regime_config import UnifiedRegimeConfig, RegimeSystemType, ArchitectureType
from .unified_result import UnifiedRegimeResult
from .regime_detector import (
    RegimeDetector, 
    create_tas_regime_detector, 
    create_nas_regime_detector,
    create_hybrid_regime_detector,
    create_unified_regime_detector
)

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
    'OptimizationLevel',
    
    # Unified regime detection
    'UnifiedRegimeDetector',
    'UnifiedRegimeConfig', 
    'RegimeSystemType',
    'ArchitectureType',
    'UnifiedRegimeResult',
    'RegimeDetector',
    'create_tas_regime_detector',
    'create_nas_regime_detector',
    'create_hybrid_regime_detector',
    'create_unified_regime_detector'
]

__version__ = "2.1.0"
__author__ = "AI Assistant"
__description__ = "Enhanced unified components with comprehensive hardware management, merged evaluation capabilities, and unified regime detection"