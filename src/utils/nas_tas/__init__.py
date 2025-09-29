"""
Neural Architecture Search and Trading Architecture Search (NAS/TAS) Module

This module provides comprehensive neural architecture search and trading architecture
search capabilities with extensive integration of utility modules for optimal performance.

Key Features:
- Neural Architecture Search (NAS) for optimal model architectures
- Trading Architecture Search (TAS) for optimal trading strategies
- Extensive integration with common utilities for data processing
- M1 hardware optimization support
- Comprehensive logging and monitoring
- Advanced optimization algorithms (Grid Search + Bayesian TPE)
- Matrix operations for high-performance computations
"""

from .core.nas_engine import NASEngine
from .core.tas_engine import TASEngine
from .optimization.architecture_search import ArchitectureSearchOptimizer
from .optimization.strategy_search import StrategySearchOptimizer
from .data.data_manager import NASDataManager
from .evaluation.evaluator import ArchitectureEvaluator
from .utils.nas_utilities import NASUtilities
from .utils.tas_utilities import TASUtilities

__version__ = "1.0.0"
__author__ = "NAS/TAS Development Team"

__all__ = [
    "NASEngine",
    "TASEngine", 
    "ArchitectureSearchOptimizer",
    "StrategySearchOptimizer",
    "NASDataManager",
    "ArchitectureEvaluator",
    "NASUtilities",
    "TASUtilities"
]