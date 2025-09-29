"""
NAS/TAS Evaluation

This module provides comprehensive evaluation for Neural Architecture Search
and Trading Architecture Search with extensive utility integration.
"""

from .evaluator import ArchitectureEvaluator, create_architecture_evaluator

__all__ = ["ArchitectureEvaluator", "create_architecture_evaluator"]