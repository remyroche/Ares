#!/usr/bin/env python3
"""
Script to analyze undefined names report and prioritize fixes.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path


def analyze_undefined_names_report(report_path: str):
    """Analyze the undefined names report and provide prioritized fixes."""
    
    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)
    
    # Collect all undefined names
    undefined_names = Counter()
    files_with_errors = defaultdict(list)
    error_patterns = defaultdict(int)
    
    for file_path, file_data in report.get('files', {}).items():
        if file_data.get('status') == 'success' and file_data.get('total_errors', 0) > 0:
            for error in file_data.get('errors', []):
                if error.get('error_type') == 'undefined_name':
                    name = error.get('name', '')
                    undefined_names[name] += 1
                    files_with_errors[name].append({
                        'file': file_path,
                        'line': error.get('line', 0),
                        'context': error.get('context', '')
                    })
                    
                    # Categorize error patterns
                    context = error.get('context', '')
                    if 'attribute access' in context:
                        error_patterns['attribute_access'] += 1
                    elif 'function call' in context:
                        error_patterns['function_call'] += 1
                    elif 'in Dict' in context:
                        error_patterns['type_annotation'] += 1
                    elif 'in Subscript' in context:
                        error_patterns['type_annotation'] += 1
                    else:
                        error_patterns['other'] += 1
    
    print("="*80)
    print("UNDEFINED NAMES ANALYSIS REPORT")
    print("="*80)
    
    print(f"\nTotal undefined names: {len(undefined_names)}")
    print(f"Total occurrences: {sum(undefined_names.values())}")
    print(f"Files with errors: {len([f for f in report.get('files', {}).values() if f.get('total_errors', 0) > 0])}")
    
    print(f"\nError patterns:")
    for pattern, count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pattern}: {count}")
    
    print(f"\nTop 30 undefined names:")
    for name, count in undefined_names.most_common(30):
        print(f"  {name}: {count} occurrences")
    
    print(f"\nFiles with most errors:")
    file_error_counts = []
    for file_path, file_data in report.get('files', {}).items():
        if file_data.get('total_errors', 0) > 0:
            file_error_counts.append((file_path, file_data.get('total_errors', 0)))
    
    for file_path, count in sorted(file_error_counts, key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {file_path}: {count} errors")
    
    # Identify common import patterns
    print(f"\nCommon import patterns to fix:")
    common_imports = {
        'pd': 'import pandas as pd',
        'np': 'import numpy as np',
        'datetime': 'from datetime import datetime',
        'Dict': 'from typing import Dict',
        'List': 'from typing import List',
        'Optional': 'from typing import Optional',
        'Any': 'from typing import Any',
        'Tuple': 'from typing import Tuple',
        'Union': 'from typing import Union',
        'Set': 'from typing import Set',
        'Callable': 'from typing import Callable',
        'dataclass': 'from dataclasses import dataclass',
        'cached': 'from functools import cached_property',
        'error': 'from logging import error',
        'info': 'from logging import info',
        'warning': 'from logging import warning',
        'debug': 'from logging import debug',
    }
    
    for name, import_stmt in common_imports.items():
        if name in undefined_names:
            print(f"  {name} ({undefined_names[name]} occurrences): {import_stmt}")
    
    return {
        'undefined_names': dict(undefined_names),
        'files_with_errors': dict(files_with_errors),
        'error_patterns': dict(error_patterns),
        'file_error_counts': file_error_counts
    }


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze undefined names report")
    parser.add_argument("--report", "-r", 
                       default="code_quality/reports/undefined_names_current.json",
                       help="Path to undefined names report")
    
    args = parser.parse_args()
    
    if not Path(args.report).exists():
        print(f"Report file not found: {args.report}")
        return 1
    
    analyze_undefined_names_report(args.report)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
