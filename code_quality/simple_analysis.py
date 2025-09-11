#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Simple analysis script to demonstrate pipeline findings.
"""

import ast
import os
from pathlib import Path
from collections import defaultdict
import numpy as np

def analyze_file_complexity(file_path):
    """Analyze cyclomatic complexity of a Python file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        complexity = 1  # Base complexity
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.FunctionDef):
                func_complexity = 1
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler)):
                        func_complexity += 1
                functions.append({
                    'name': node.name,
                    'line': node.lineno,
                    'complexity': func_complexity
                })
        
        return {
            'file': file_path,
            'total_complexity': complexity,
            'functions': functions,
            'lines_of_code': len(content.splitlines())
        }
    except Exception as e:
        return {'file': file_path, 'error': str(e)}

def find_dead_imports(file_path):
    """Find potentially unused imports."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        imports = []
        used_names = set()
        
        # Find all imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'type': 'import',
                        'name': alias.name,
                        'line': node.lineno,
                        'asname': alias.asname
                    })
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imports.append({
                        'type': 'from_import',
                        'name': alias.name,
                        'line': node.lineno,
                        'module': node.module,
                        'asname': alias.asname
                    })
        
        # Find all name usages
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                used_names.add(node.attr)
        
        # Check for unused imports
        unused_imports = []
        for imp in imports:
            name_to_check = imp['asname'] if imp['asname'] else imp['name']
            if name_to_check not in used_names:
                unused_imports.append(imp)
        
        return unused_imports
    except Exception as e:
        return [{'error': str(e)}]

def main():
    """Run simple analysis on the codebase."""
    tprint("🔍 SIMPLE CODE QUALITY ANALYSIS")
    tprint("=" * 50)
    
    # Analyze some key files
    files_to_analyze = [
        '/workspace/code_quality/analyzers/complexity_analyzer.py',
        '/workspace/code_quality/pipelines/complexity_pipeline.py',
        '/workspace/code_quality/analyzers/enhanced_dead_code_analyzer.py',
        '/workspace/code_quality/fixers/auto_fixer.py',
        '/workspace/code_quality/core/config.py'
    ]
    
    tprint("\n📊 COMPLEXITY ANALYSIS")
    tprint("-" * 30)
    
    complexity_results = []
    for file_path in files_to_analyze:
        if os.path.exists(file_path):
            result = analyze_file_complexity(file_path)
            if 'error' not in result:
                complexity_results.append(result)
                tprint(f"📁 {os.path.basename(file_path)}")
                tprint(f"   Total Complexity: {result['total_complexity']}")
                tprint(f"   Lines of Code: {result['lines_of_code']}")
                tprint(f"   Functions: {len(result['functions'])}")
                
                # Show most complex functions
                if result['functions']:
                    complex_funcs = sorted(result['functions'], key=lambda x: x['complexity'], reverse=True)[:3]
                    for func in complex_funcs:
                        if func['complexity'] > 5:  # Only show complex functions
                            tprint(f"   ⚠️  {func['name']} (line {func['line']}): complexity {func['complexity']}")
                tprint()
    
    tprint("\n🔍 DEAD IMPORT ANALYSIS")
    tprint("-" * 30)
    
    dead_imports_found = 0
    for file_path in files_to_analyze:
        if os.path.exists(file_path):
            unused = find_dead_imports(file_path)
            if unused and not any('error' in item for item in unused):
                dead_imports_found += len(unused)
                tprint(f"📁 {os.path.basename(file_path)}: {len(unused)} unused imports")
                for imp in unused[:3]:  # Show first 3
                    tprint(f"   - Line {imp['line']}: {imp['name']}")
    
    tprint(f"\n📈 SUMMARY")
    tprint("-" * 30)
    tprint(f"Files analyzed: {len(complexity_results)}")
    tprint(f"Total unused imports found: {dead_imports_found}")
    
    # Find most complex files
    if complexity_results:
        most_complex = max(complexity_results, key=lambda x: x['total_complexity'])
        tprint(f"Most complex file: {os.path.basename(most_complex['file'])} (complexity: {most_complex['total_complexity']})")
        
        # Find files with many functions
        most_functions = max(complexity_results, key=lambda x: len(x['functions']))
        tprint(f"File with most functions: {os.path.basename(most_functions['file'])} ({len(most_functions['functions'])} functions)")

if __name__ == "__main__":
    main()