"""
Code Quality Analysis Functions

This module provides simple, direct functions for code quality analysis.
The pipeline abstraction has been removed in favor of simple functions that
directly call analyzers without unnecessary complexity.

Main functions:
- run_import_verification: Check which files are imported by others
- run_enhanced_import_analysis: Comprehensive import analysis
- run_dead_code_analysis: Find unused code
- run_complexity_analysis: Measure code complexity
- run_dependency_analysis: Understand module dependencies
- run_all_analyses: Run all available analyses
- run_sequential_fixes: Run automated code fixes

Usage:
    from analysis_functions import run_import_verification
    results = run_import_verification("/path/to/project")
"""

from pathlib import Path

# Pipeline directory
PIPELINE_DIR = Path(__file__).parent

# Code quality root directory
CODE_QUALITY_DIR = PIPELINE_DIR.parent

# Scripts directory (where individual tools are located)
SCRIPTS_DIR = CODE_QUALITY_DIR / "scripts"

# Reports directory
REPORTS_DIR = CODE_QUALITY_DIR / "reports"

__all__ = [
    "PIPELINE_DIR",
    "CODE_QUALITY_DIR", 
    "SCRIPTS_DIR",
    "REPORTS_DIR",
]
