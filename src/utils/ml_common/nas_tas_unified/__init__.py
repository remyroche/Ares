#!/usr/bin/env python3
"""
NAS-TAS Unified Components Package

This package provides unified components for both NAS and TAS systems,
organized into logical modules for better maintainability and organization.

Modules:
- evaluation: Unified evaluation framework
- hardware: Hardware optimization using existing tools
- search: Search algorithms with Bayesian TPE integration
- data_processing: Unified data processing pipeline
- manager: Unified component manager
"""

from .evaluation import UnifiedEvaluator
from .hardware import UnifiedHardwareOptimizer
from .search import UnifiedSearchEngine
from .data_processing import UnifiedDataProcessor
from .manager import UnifiedComponentManager

__all__ = [
    'UnifiedEvaluator',
    'UnifiedHardwareOptimizer', 
    'UnifiedSearchEngine',
    'UnifiedDataProcessor',
    'UnifiedComponentManager'
]

__version__ = "1.0.0"
__author__ = "AI Assistant"