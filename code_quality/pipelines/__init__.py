"""
Code Quality Pipelines

This module contains various pipelines for running code quality tools:

- pipeline_unified_enhanced: Enhanced unified pipeline with comprehensive reporting
- pipeline_unified_integrated: Integrated pipeline with direct imports
- pipeline_unified_standalone: Standalone pipeline using subprocess (no imports)
- pipeline_syntax_imports: Pipeline for syntax and import fixes
- pipeline_syntax_imports_enhanced: Enhanced syntax/import pipeline with unified reporting
- pipeline_async_types: Pipeline for async and type hint fixes
- pipeline_analysis: Pipeline for code analysis and validation
"""

from pathlib import Path

# Pipeline directory
PIPELINE_DIR = Path(__file__).parent

# Code quality root directory
CODE_QUALITY_DIR = PIPELINE_DIR.parent

# Scripts directory (where individual tools are located)
SCRIPTS_DIR = CODE_QUALITY_DIR / 'scripts'

# Reports directory
REPORTS_DIR = CODE_QUALITY_DIR / 'reports'

__all__ = [
    'PIPELINE_DIR',
    'CODE_QUALITY_DIR', 
    'SCRIPTS_DIR',
    'REPORTS_DIR'
]