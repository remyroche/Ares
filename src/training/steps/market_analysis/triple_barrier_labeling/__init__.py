"""
Triple Barrier Labeling Package

This package provides a unified, robust implementation of triple barrier labeling
for the market analysis pipeline.

Key Features:
- Unified configuration and execution
- Explicit error handling (no silent failures)
- Comprehensive validation framework
- Enhanced reporting and metrics
- Performance optimization with proper fallbacks
- Regime-aware labeling support

Main Classes:
- UnifiedTripleBarrierLabeler: Main labeling class
- TripleBarrierConfig: Configuration management
- TripleBarrierResult: Execution results
- DataValidator: Data validation framework
- HardwareManager: Hardware optimization management

Convenience Functions:
- apply_triple_barrier_labeling: Direct labeling function
- create_triple_barrier_labeler: Labeler factory function
"""

from .unified_labeler import (
    UnifiedTripleBarrierLabeler,
    TripleBarrierConfig,
    TripleBarrierResult,
    ValidationResult,
    DataValidator,
    HardwareManager,
    MetricsCollector,
    ProgressReporter,
    # Exception classes
    TripleBarrierError,
    ValidationError,
    ConfigurationError,
    HardwareOptimizationError,
    DataQualityError,
    # Convenience functions
    create_triple_barrier_labeler,
    apply_triple_barrier_labeling
)

# Version information
__version__ = "1.0.0"
__author__ = "Market Analysis Team"
__description__ = "Unified Triple Barrier Labeling Implementation"

# Public API
__all__ = [
    # Main classes
    "UnifiedTripleBarrierLabeler",
    "TripleBarrierConfig", 
    "TripleBarrierResult",
    "ValidationResult",
    "DataValidator",
    "HardwareManager",
    "MetricsCollector",
    "ProgressReporter",
    
    # Exception classes
    "TripleBarrierError",
    "ValidationError",
    "ConfigurationError", 
    "HardwareOptimizationError",
    "DataQualityError",
    
    # Convenience functions
    "create_triple_barrier_labeler",
    "apply_triple_barrier_labeling",
    
    # Version info
    "__version__",
    "__author__",
    "__description__"
]