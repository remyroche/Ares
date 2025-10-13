"""
Data-Driven Period Selection for Cross-Timeframe Features (Refactored)

This module implements intelligent period selection based on data characteristics
rather than using hardcoded periods. It analyzes the data to determine optimal
periods for cross-timeframe feature generation.

Key Features:
- Analyzes data frequency and length
- Detects natural market cycles
- Optimizes periods for feature diversity
- Considers computational constraints
- Adapts to different timeframes (5m, 15m, 60m)
- VectorBT-optimized rolling operations
- Memory-efficient batch processing
- Parallel period analysis

Refactored Architecture:
- PeriodAnalyzer: Handles data analysis and pattern detection
- PeriodValidator: Handles filtering, ranking, and validation
- PeriodSelector: Coordinates the selection process
- PeriodAnalysisUtils: Common utilities to eliminate code duplication
"""

# Import the refactored implementation
from .data_driven_periods_refactored import (
    DataDrivenPeriodSelector,
    PeriodAnalysisResult,
    get_data_driven_periods,
    get_data_driven_periods_with_stats,
    benchmark_period_selector,
    PeriodAnalyzer,
    PeriodValidator,
    PeriodSelector,
    PeriodAnalysisUtils,
    ValidationError,
    AnalysisError
)

# Re-export everything for backward compatibility
__all__ = [
    'DataDrivenPeriodSelector',
    'PeriodAnalysisResult',
    'get_data_driven_periods',
    'get_data_driven_periods_with_stats',
    'benchmark_period_selector',
    'PeriodAnalyzer',
    'PeriodValidator',
    'PeriodSelector',
    'PeriodAnalysisUtils',
    'ValidationError',
    'AnalysisError'
]