"""
Code Complexity Analysis Pipeline

A comprehensive code complexity analysis pipeline that combines PyExamine, Radon, and Xenon
tools to provide detailed complexity metrics for Python codebases.

Features:
- Multi-tool analysis combining PyExamine, Radon, and Xenon
- Per-file and per-directory complexity analysis
- Multiple output formats (JSON, HTML, Markdown, Summary)
- Configurable analysis parameters
- Command line interface
- Tool availability checking

Usage:
    from code_complexity import ComplexityPipeline
    
    pipeline = ComplexityPipeline()
    results = pipeline.run_full_analysis('/path/to/code')
"""

__version__ = "1.0.0"
__author__ = "Code Quality Team"
__email__ = "code-quality@example.com"

from .complexity_pipeline import ComplexityPipeline, ComplexityMetrics, DirectoryMetrics
from .config.complexity_config import ComplexityConfig

__all__ = [
    'ComplexityPipeline',
    'ComplexityMetrics', 
    'DirectoryMetrics',
    'ComplexityConfig'
]