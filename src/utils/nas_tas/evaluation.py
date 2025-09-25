#!/usr/bin/env python3
"""
Unified Evaluation Framework - Redirected to New Implementation

This module redirects to the new unified evaluation framework.
"""

# Import directly from unified_evaluator to avoid circular imports
from .unified_evaluator import (
    UnifiedEvaluator,
    EvaluationConfig,
    EvaluationResult,
    ModelType,
    EvaluationMode,
    MetricType
)

# Maintain backward compatibility
__all__ = [
    'UnifiedEvaluator',
    'EvaluationConfig',
    'EvaluationResult', 
    'ModelType',
    'EvaluationMode',
    'MetricType'
]