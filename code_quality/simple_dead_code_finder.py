#!/usr/bin/env python3
"""
Simple Dead Code Finder

A simplified version of dead code detection that works without external dependencies.
This script performs basic AST-based dead code analysis.
"""

import ast
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Any
from datetime import datetime


class SimpleDeadCodeFinder:
    """Simple dead code finder using AST analysis."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.defined_functions = set()
        self.defined_classes = set()
        self.defined_imports = set()
        self.used_functions = set()
        self.used_classes = set()
        self.used_imports = set()
        self.function_definitions = {}
        self.class_definitions = {}
        self.import_definitions = {}
        
    def analyze(self) -> Dict[str, Any]:
        """Analyze the project for dead code."""
        print(f"Analyzing project: {self.project_root}")
        
        # First pass: collect all definitions
        self._collect_definitions()
        
        # Second pass: collect all usage
        self._collect_usage()
        
        # Find dead code
        dead_code = self._find_dead_code()
        
        return dead_code
    
    def _collect_definitions(self):
        """Collect all function, class, and import definitions."""
        print("Collecting definitions...")
        
        for py_file in self.project_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                self._extract_definitions(tree, py_file)
                
            except Exception as e:
                print(f"Warning: Could not parse {py_file}: {e}")
    
    def _collect_usage(self):
        """Collect all function, class, and import usage."""
        print("Collecting usage patterns...")
        
        for py_file in self.project_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                self._extract_usage(tree, py_file)
                
            except Exception as e:
                print(f"Warning: Could not parse {py_file}: {e}")
    
    def _extract_definitions(self, tree: ast.AST, file_path: Path):
        """Extract definitions from AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                self.defined_functions.add(func_name)
                self.function_definitions[func_name] = {
                    'file': str(file_path),
                    'line': node.lineno,
                    'is_public': not func_name.startswith('_'),
                    'is_special': func_name.startswith('__') and func_name.endswith('__')
                }
            
            elif isinstance(node, ast.ClassDef):
                class_name = node.name
                self.defined_classes.add(class_name)
                self.class_definitions[class_name] = {
                    'file': str(file_path),
                    'line': node.lineno,
                    'is_public': not class_name.startswith('_'),
                    'is_special': class_name.startswith('__') and class_name.endswith('__')
                }
            
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    import_name = alias.name
                    self.defined_imports.add(import_name)
                    self.import_definitions[import_name] = {
                        'file': str(file_path),
                        'line': node.lineno,
                        'module': getattr(node, 'module', None)
                    }
    
    def _extract_usage(self, tree: ast.AST, file_path: Path):
        """Extract usage patterns from AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Function calls
                if isinstance(node.func, ast.Name):
                    self.used_functions.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    # Method calls like obj.method()
                    if isinstance(node.func.value, ast.Name):
                        self.used_functions.add(f"{node.func.value.id}.{node.func.attr}")
            
            elif isinstance(node, ast.Name):
                # Variable/function/class references
                if isinstance(node.ctx, ast.Load):
                    self.used_functions.add(node.id)
                    self.used_classes.add(node.id)
            
            elif isinstance(node, ast.Attribute):
                # Attribute access like obj.attr
                if isinstance(node.ctx, ast.Load):
                    if isinstance(node.value, ast.Name):
                        self.used_classes.add(node.value.id)
    
    def _find_dead_code(self) -> Dict[str, Any]:
        """Find dead code based on definitions vs usage."""
        print("Finding dead code...")
        
        dead_functions = []
        dead_classes = []
        dead_imports = []
        
        # Find dead functions
        for func_name, info in self.function_definitions.items():
            if not self._is_function_used(func_name, info):
                dead_functions.append({
                    'name': func_name,
                    'file': info['file'],
                    'line': info['line'],
                    'is_public': info['is_public'],
                    'is_special': info['is_special'],
                    'reason': self._get_dead_reason(func_name, 'function')
                })
        
        # Find dead classes
        for class_name, info in self.class_definitions.items():
            if not self._is_class_used(class_name, info):
                dead_classes.append({
                    'name': class_name,
                    'file': info['file'],
                    'line': info['line'],
                    'is_public': info['is_public'],
                    'is_special': info['is_special'],
                    'reason': self._get_dead_reason(class_name, 'class')
                })
        
        # Find dead imports
        for import_name, info in self.import_definitions.items():
            if not self._is_import_used(import_name, info):
                dead_imports.append({
                    'name': import_name,
                    'file': info['file'],
                    'line': info['line'],
                    'module': info['module'],
                    'reason': self._get_dead_reason(import_name, 'import')
                })
        
        return {
            'timestamp': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'total_definitions': {
                'functions': len(self.defined_functions),
                'classes': len(self.defined_classes),
                'imports': len(self.defined_imports)
            },
            'total_usage': {
                'functions': len(self.used_functions),
                'classes': len(self.used_classes),
                'imports': len(self.used_imports)
            },
            'dead_code': {
                'functions': dead_functions,
                'classes': dead_classes,
                'imports': dead_imports
            },
            'statistics': {
                'dead_functions': len(dead_functions),
                'dead_classes': len(dead_classes),
                'dead_imports': len(dead_imports),
                'total_dead_items': len(dead_functions) + len(dead_classes) + len(dead_imports)
            }
        }
    
    def _is_function_used(self, func_name: str, info: Dict[str, Any]) -> bool:
        """Check if a function is used."""
        # Special functions are considered used
        if info['is_special']:
            return True
        
        # Check direct usage
        if func_name in self.used_functions:
            return True
        
        # Check if it's a main function
        if func_name == 'main':
            return True
        
        # Check if it's a test function
        if func_name.startswith('test_'):
            return True
        
        # Check if it's a setup/teardown function
        if func_name.startswith(('setup_', 'teardown_', 'fixture_')):
            return True
        
        return False
    
    def _is_class_used(self, class_name: str, info: Dict[str, Any]) -> bool:
        """Check if a class is used."""
        # Special classes are considered used
        if info['is_special']:
            return True
        
        # Check direct usage
        if class_name in self.used_classes:
            return True
        
        # Check if it's a base class
        if class_name in ['Base', 'Abstract', 'Interface', 'Protocol']:
            return True
        
        # Check if it's a test class
        if class_name.startswith('Test'):
            return True
        
        return False
    
    def _is_import_used(self, import_name: str, info: Dict[str, Any]) -> bool:
        """Check if an import is used."""
        # Check if the imported name is used
        if import_name in self.used_functions or import_name in self.used_classes:
            return True
        
        # Check if it's a common import that might be used dynamically
        common_imports = ['os', 'sys', 'json', 'datetime', 'pathlib', 'typing']
        if import_name in common_imports:
            return True
        
        return False
    
    def _get_dead_reason(self, name: str, item_type: str) -> str:
        """Get the reason why an item is considered dead."""
        if item_type == 'function':
            if name.startswith('_'):
                return "private_function_not_used"
            else:
                return "public_function_not_used"
        elif item_type == 'class':
            if name.startswith('_'):
                return "private_class_not_used"
            else:
                return "public_class_not_used"
        elif item_type == 'import':
            return "import_not_used"
        else:
            return "not_used"
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if a file should be skipped."""
        skip_patterns = [
            '__pycache__',
            '.git',
            'venv',
            'env',
            '.pytest_cache',
            'node_modules',
            '.tox',
            'build',
            'dist'
        ]
        
        file_str = str(file_path)
        return any(pattern in file_str for pattern in skip_patterns)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Dead Code Finder")
    parser.add_argument("--project-root", type=str, default=".", help="Project root directory")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Initialize finder
    project_root = Path(args.project_root)
    finder = SimpleDeadCodeFinder(project_root)
    
    # Run analysis
    results = finder.analyze()
    
    # Print results
    print(f"\n{'='*60}")
    print("DEAD CODE ANALYSIS RESULTS")
    print(f"{'='*60}")
    
    stats = results['statistics']
    print(f"Total definitions: {results['total_definitions']}")
    print(f"Total usage: {results['total_usage']}")
    print(f"\nDead code found:")
    print(f"  - Dead functions: {stats['dead_functions']}")
    print(f"  - Dead classes: {stats['dead_classes']}")
    print(f"  - Dead imports: {stats['dead_imports']}")
    print(f"  - Total dead items: {stats['total_dead_items']}")
    
    # Show details
    if args.verbose or stats['total_dead_items'] > 0:
        print(f"\n{'='*60}")
        print("DETAILED RESULTS")
        print(f"{'='*60}")
        
        # Dead functions
        if results['dead_code']['functions']:
            print(f"\nDead Functions ({len(results['dead_code']['functions'])}):")
            for func in results['dead_code']['functions']:
                print(f"  - {func['name']} in {func['file']}:{func['line']} ({func['reason']})")
        
        # Dead classes
        if results['dead_code']['classes']:
            print(f"\nDead Classes ({len(results['dead_code']['classes'])}):")
            for cls in results['dead_code']['classes']:
                print(f"  - {cls['name']} in {cls['file']}:{cls['line']} ({cls['reason']})")
        
        # Dead imports
        if results['dead_code']['imports']:
            print(f"\nDead Imports ({len(results['dead_code']['imports'])}):")
            for imp in results['dead_code']['imports']:
                print(f"  - {imp['name']} in {imp['file']}:{imp['line']} ({imp['reason']})")
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    return results


if __name__ == "__main__":
    main()