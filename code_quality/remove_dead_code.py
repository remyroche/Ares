#!/usr/bin/env python3
"""
Remove Dead Code and Legacy Functions
Systematically removes dead code and legacy functions based on analysis findings
"""

import os
import ast
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple
import argparse


class DeadCodeRemover:
    """Removes dead code and legacy functions from Python files."""
    
    def __init__(self, exclusions_file: str = None):
        self.exclusions = self._load_exclusions(exclusions_file)
        self.removed_items = []
        
    def _load_exclusions(self, exclusions_file: str) -> Set[str]:
        """Load exclusion patterns from file."""
        exclusions = set()
        if exclusions_file and os.path.exists(exclusions_file):
            with open(exclusions_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        exclusions.add(line)
        return exclusions
    
    def _should_exclude(self, filepath: str) -> bool:
        """Check if file should be excluded based on patterns."""
        for pattern in self.exclusions:
            if pattern in filepath or filepath.endswith(pattern.replace('*', '')):
                return True
        return False
    
    def _is_test_function(self, func_name: str) -> bool:
        """Check if a function is a test function."""
        return func_name.startswith('test_')
    
    def _is_main_or_init(self, func_name: str) -> bool:
        """Check if function is main or __init__."""
        return func_name in ['main', '__init__', '__main__']
    
    def _is_public_api(self, func_name: str) -> bool:
        """Check if function might be part of public API."""
        # Functions that might be called externally
        public_indicators = [
            'run', 'start', 'execute', 'process', 'analyze', 'train', 'predict',
            'validate', 'check', 'verify', 'test', 'main', '__init__'
        ]
        
        func_lower = func_name.lower()
        return any(indicator in func_lower for indicator in public_indicators)
    
    def analyze_file(self, filepath: str) -> Dict[str, List[Dict]]:
        """Analyze a file for dead code and legacy functions."""
        if self._should_exclude(filepath):
            return {}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            lines = content.split('\n')
            
            dead_code = {
                'unused_functions': [],
                'legacy_functions': [],
                'unreachable_code': []
            }
            
            # Find all function definitions
            function_names = set()
            called_functions = set()
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    function_names.add(node.name)
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        called_functions.add(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        called_functions.add(node.func.attr)
            
            # Check for string usage of function names
            for line in content.split('\n'):
                for func_name in function_names:
                    if f"'{func_name}'" in line or f'"{func_name}"' in line:
                        called_functions.add(func_name)
            
            # Identify unused and legacy functions
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_name = node.name
                    
                    # Skip certain types of functions
                    if (self._is_main_or_init(func_name) or 
                        self._is_public_api(func_name) or
                        self._is_test_function(func_name)):
                        continue
                    
                    # Check if function is unused
                    if func_name not in called_functions:
                        dead_code['unused_functions'].append({
                            'name': func_name,
                            'line': node.lineno,
                            'type': 'unused_function'
                        })
                    
                    # Check if function appears to be legacy
                    if self._is_legacy_function(func_name):
                        dead_code['legacy_functions'].append({
                            'name': func_name,
                            'line': node.lineno,
                            'type': 'legacy_function'
                        })
            
            # Find unreachable code after return statements
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for i, stmt in enumerate(node.body):
                        if isinstance(stmt, ast.Return):
                            # Check if there are statements after return
                            if i < len(node.body) - 1:
                                dead_code['unreachable_code'].append({
                                    'function': node.name,
                                    'line': node.body[i + 1].lineno,
                                    'type': 'unreachable_after_return'
                                })
            
            return dead_code
            
        except Exception as e:
            print(f"Error analyzing {filepath}: {e}")
            return {}
    
    def process_directory(self, directory: str, dry_run: bool = True) -> Dict[str, List]:
        """Process all Python files in a directory."""
        results = {
            'files_processed': 0,
            'files_modified': 0,
            'total_removed': 0
        }
        
        for root, dirs, files in os.walk(directory):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not self._should_exclude(os.path.join(root, d))]
            
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    
                    if self._should_exclude(filepath):
                        continue
                    
                    print(f"Processing: {filepath}")
                    results['files_processed'] += 1
                    
                    if self.remove_dead_code_from_file(filepath, dry_run):
                        results['files_modified'] += 1
                        results['total_removed'] += 1
        
        return results
    
    def generate_report(self, results: Dict) -> str:
        """Generate a report of removed items."""
        report = []
        report.append("=" * 80)
        report.append("DEAD CODE REMOVAL REPORT")
        report.append("=" * 80)
        report.append(f"Files processed: {results['files_processed']}")
        report.append(f"Files modified: {results['files_modified']}")
        report.append(f"Total items removed: {results['total_removed']}")
        report.append("")
        
        if self.removed_items:
            report.append("Removed Items:")
            report.append("-" * 50)
            
            by_type = {}
            for item in self.removed_items:
                item_type = item['type']
                if item_type not in by_type:
                    by_type[item_type] = []
                by_type[item_type].append(item)
            
            for item_type, items in by_type.items():
                report.append(f"\n{item_type.title()} ({len(items)}):")
                for item in items:
                    report.append(f"  - {item['name']} in {item['file']} (line {item['line']})")
        else:
            report.append("No items were removed.")
        
        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Remove dead code and legacy functions')
    parser.add_argument('directory', help='Directory to process')
    parser.add_argument('--exclusions', help='Exclusions file path')
    parser.add_argument('--no-dry-run', action='store_true', help='Actually remove code (not just preview)')
    parser.add_argument('--output', help='Output report to file')
    
    args = parser.parse_args()
    
    remover = DeadCodeRemover(args.exclusions)
    results = remover.process_directory(args.directory, dry_run=not args.no_dry_run)
    
    report = remover.generate_report(results)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == '__main__':
    main()