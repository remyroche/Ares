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

# Import convenience functions
from .core.nas_engine import create_nas_engine
from .core.tas_engine import create_tas_engine
from .optimization.architecture_search import create_architecture_search_optimizer
from .optimization.strategy_search import create_strategy_search_optimizer
from .data.data_manager import create_nas_data_manager
from .evaluation.evaluator import create_architecture_evaluator
from .utils.nas_utilities import create_nas_utilities
from .utils.tas_utilities import create_tas_utilities

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
    "TASUtilities",
    # Convenience functions
    "create_nas_engine",
    "create_tas_engine",
    "create_architecture_search_optimizer",
    "create_strategy_search_optimizer",
    "create_nas_data_manager",
    "create_architecture_evaluator",
    "create_nas_utilities",
    "create_tas_utilities"
]