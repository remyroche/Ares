"""
NAS/TAS Shared Results Utilities

This module provides unified result handling, comparison, and serialization
for both NAS and TAS systems, consolidating result management logic.
"""

from .result_manager import (
    ResultManager,
    UnifiedArchitectureResult,
    ArchitectureResult,
    ExecutionInfo,
    ComparisonResult
)

from .comparison_utils import (
    ResultComparison,
    ArchitectureComparison,
    PerformanceComparison,
    FinancialComparison,
    RegimeComparison
)

__all__ = [
    # Result management
    'ResultManager',
    'UnifiedArchitectureResult',
    'ArchitectureResult',
    'ExecutionInfo',
    'ComparisonResult',

    # Comparison utilities
    'ResultComparison',
    'ArchitectureComparison',
    'PerformanceComparison',
    'FinancialComparison',
    'RegimeComparison'
]
