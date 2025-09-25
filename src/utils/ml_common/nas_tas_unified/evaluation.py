#!/usr/bin/env python3
"""
Unified Evaluation Framework - Redirected to New Implementation

This module redirects to the new unified evaluation framework.
"""

# Redirect to the new unified evaluator
from src.utils.nas_tas import (
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