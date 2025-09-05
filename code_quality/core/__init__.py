"""
Core framework components for code quality analysis.

This module contains the base classes and core functionality that other
components inherit from and build upon.
"""

from .config import CodeQualityConfig
from .base_analyzer import BaseAnalyzer
from .base_fixer import BaseFixer
from .pipeline_base import BasePipeline

__all__ = [
    'CodeQualityConfig',
    'BaseAnalyzer', 
    'BaseFixer',
    'BasePipeline'
]