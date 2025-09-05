"""
Core framework components for code quality analysis.

This module contains the base classes and core functionality that other
components inherit from and build upon.
"""

from .config import CodeQualityConfig, get_default_config, AnalysisConfig

__all__ = [
    'CodeQualityConfig',
    'get_default_config',
    'AnalysisConfig'
]