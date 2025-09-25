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

# Unified regime detection components (primary focus)
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

# Other components (imported conditionally to avoid circular imports)
try:
    from .evaluation import UnifiedEvaluator, compute_classification_metrics, compute_regression_metrics
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False

try:
    from .hardware import (
        UnifiedHardwareOptimizer, HardwareConfig, PerformanceMetrics, 
        HardwarePerformanceMonitor, WorkloadType, OptimizationLevel
    )
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False

try:
    from .search import UnifiedSearchEngine
    SEARCH_AVAILABLE = True
except ImportError:
    SEARCH_AVAILABLE = False

try:
    from .data_processing import UnifiedDataProcessor
    DATA_PROCESSING_AVAILABLE = True
except ImportError:
    DATA_PROCESSING_AVAILABLE = False

try:
    from .manager import UnifiedComponentManager
    MANAGER_AVAILABLE = True
except ImportError:
    MANAGER_AVAILABLE = False

__all__ = [
    # Unified regime detection (always available)
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

# Add other components if available
if EVALUATION_AVAILABLE:
    __all__.extend(['UnifiedEvaluator', 'compute_classification_metrics', 'compute_regression_metrics'])

if HARDWARE_AVAILABLE:
    __all__.extend(['UnifiedHardwareOptimizer', 'HardwareConfig', 'PerformanceMetrics', 
                   'HardwarePerformanceMonitor', 'WorkloadType', 'OptimizationLevel'])

if SEARCH_AVAILABLE:
    __all__.append('UnifiedSearchEngine')

if DATA_PROCESSING_AVAILABLE:
    __all__.append('UnifiedDataProcessor')

if MANAGER_AVAILABLE:
    __all__.append('UnifiedComponentManager')

__version__ = "2.1.0"
__author__ = "AI Assistant"
__description__ = "Enhanced unified components with comprehensive hardware management, merged evaluation capabilities, and unified regime detection"