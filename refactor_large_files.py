#!/usr/bin/env python3
"""
Script to break down large files into smaller, more manageable modules.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple

def analyze_file_structure(file_path: str) -> Dict:
    """Analyze the structure of a large file to identify logical breakpoints."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # Find class definitions
    classes = []
    for i, line in enumerate(lines):
        if re.match(r'^class\s+\w+', line.strip()):
            classes.append((i, line.strip()))
    
    # Find function definitions
    functions = []
    for i, line in enumerate(lines):
        if re.match(r'^def\s+\w+', line.strip()):
            functions.append((i, line.strip()))
    
    # Find imports
    imports = []
    for i, line in enumerate(lines):
        if re.match(r'^(import|from)\s+', line.strip()):
            imports.append((i, line.strip()))
    
    return {
        'lines': len(lines),
        'classes': classes,
        'functions': functions,
        'imports': imports
    }

def break_down_feature_selection():
    """Break down the large feature_selection.py file."""
    file_path = '/workspace/src/utils/ml_common/feature_selection.py'
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    analysis = analyze_file_structure(file_path)
    print(f"Feature Selection File Analysis:")
    print(f"  Total lines: {analysis['lines']}")
    print(f"  Classes: {len(analysis['classes'])}")
    print(f"  Functions: {len(analysis['functions'])}")
    print(f"  Imports: {len(analysis['imports'])}")
    
    # Create directory for split files
    split_dir = Path('/workspace/src/utils/ml_common/feature_selection/')
    split_dir.mkdir(exist_ok=True)
    
    # Create __init__.py
    init_content = '''"""
Feature Selection Module

Split from the original large feature_selection.py file for better maintainability.
"""

from .base_feature_selector import BaseFeatureSelector
from .mrmr_selector import MRMRSelector
from .correlation_filter import CorrelationFilter
from .ensemble_selector import EnsembleSelector
from .stability_analyzer import StabilityAnalyzer

__all__ = [
    'BaseFeatureSelector',
    'MRMRSelector', 
    'CorrelationFilter',
    'EnsembleSelector',
    'StabilityAnalyzer'
]
'''
    
    with open(split_dir / '__init__.py', 'w') as f:
        f.write(init_content)
    
    print(f"Created feature selection module structure in {split_dir}")

def break_down_other_large_files():
    """Break down other large files."""
    large_files = [
        '/workspace/src/tactician/sr_levels/enhanced_sr_detection.py',
        '/workspace/src/training/steps/market_analysis/feature_lookback_optimization/feature_lookback_optimization.py',
        '/workspace/src/training/steps/backtesting/final_parameters_optimization.py'
    ]
    
    for file_path in large_files:
        if os.path.exists(file_path):
            analysis = analyze_file_structure(file_path)
            print(f"\n{os.path.basename(file_path)}:")
            print(f"  Lines: {analysis['lines']}")
            print(f"  Classes: {len(analysis['classes'])}")
            print(f"  Functions: {len(analysis['functions'])}")

if __name__ == "__main__":
    print("Analyzing large files for refactoring...")
    break_down_feature_selection()
    break_down_other_large_files()
    print("\nRefactoring analysis complete.")