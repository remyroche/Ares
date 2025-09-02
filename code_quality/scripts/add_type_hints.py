#!/usr/bin/env python3
"""
Script to add type hints to functions and classes in the codebase.
"""

import ast
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import re


class TypeHintAdder(ast.NodeTransformer):
    """AST transformer to add type hints to functions."""
    
    def __init__(self):
        self.changes_made = []
        
    def visit_FunctionDef(self, node):
        """Add type hints to function definitions."""
        self.generic_visit(node)
        
        # Check if function already has type hints
        has_return_type = node.returns is not None
        has_arg_types = any(arg.annotation is not None for arg in node.args.args)
        
        if not has_return_type or not has_arg_types:
            # Infer types based on function name and parameters
            inferred_hints = self._infer_type_hints(node)
            
            # Add parameter type hints
            for i, arg in enumerate(node.args.args):
                if arg.annotation is None and i < len(inferred_hints['params']):
                    param_type = inferred_hints['params'][i]
                    if param_type:
                        arg.annotation = ast.parse(param_type).body[0].value
                        self.changes_made.append(f"Added type hint for parameter {arg.arg}")
            
            # Add return type hint
            if not has_return_type and inferred_hints['return']:
                node.returns = ast.parse(inferred_hints['return']).body[0].value
                self.changes_made.append(f"Added return type hint for {node.name}")
        
        return node
    
    def visit_AsyncFunctionDef(self, node):
        """Add type hints to async function definitions."""
        # Treat async functions the same way
        return self.visit_FunctionDef(node)
    
    def _infer_type_hints(self, node) -> Dict[str, any]:
        """Infer type hints based on function name and parameters."""
        hints = {'params': [], 'return': None}
        
        # Common patterns for parameter names
        param_patterns = {
            'df': 'pd.DataFrame',
            'data': 'pd.DataFrame',
            'config': 'Dict[str, Any]',
            'path': 'Union[str, Path]',
            'file_path': 'Union[str, Path]',
            'symbol': 'str',
            'symbols': 'List[str]',
            'start_date': 'datetime',
            'end_date': 'datetime',
            'amount': 'float',
            'price': 'float',
            'count': 'int',
            'index': 'int',
            'size': 'int',
            'window': 'int',
            'threshold': 'float',
            'model': 'Any',
            'features': 'np.ndarray',
            'labels': 'np.ndarray',
            'X': 'np.ndarray',
            'y': 'np.ndarray',
            'params': 'Dict[str, Any]',
            'kwargs': 'Dict[str, Any]',
        }
        
        # Infer parameter types
        for arg in node.args.args:
            if arg.arg == 'self' or arg.arg == 'cls':
                continue
            
            param_type = None
            # Check exact matches
            if arg.arg in param_patterns:
                param_type = param_patterns[arg.arg]
            # Check patterns
            elif arg.arg.endswith('_id') or arg.arg.endswith('_ids'):
                param_type = 'List[str]' if arg.arg.endswith('_ids') else 'str'
            elif arg.arg.startswith('is_') or arg.arg.startswith('has_'):
                param_type = 'bool'
            elif arg.arg.endswith('_list') or arg.arg.endswith('_array'):
                param_type = 'List[Any]'
            elif arg.arg.endswith('_dict'):
                param_type = 'Dict[str, Any]'
            
            hints['params'].append(param_type)
        
        # Infer return type based on function name
        func_name = node.name
        if func_name.startswith('get_') or func_name.startswith('fetch_'):
            if 'dataframe' in func_name.lower() or 'df' in func_name.lower():
                hints['return'] = 'pd.DataFrame'
            elif 'config' in func_name.lower():
                hints['return'] = 'Dict[str, Any]'
            elif 'list' in func_name.lower():
                hints['return'] = 'List[Any]'
            else:
                hints['return'] = 'Any'
        elif func_name.startswith('is_') or func_name.startswith('has_') or func_name.startswith('check_'):
            hints['return'] = 'bool'
        elif func_name.startswith('calculate_') or func_name.startswith('compute_'):
            hints['return'] = 'float'
        elif func_name.startswith('create_') or func_name.startswith('build_'):
            hints['return'] = 'Any'
        elif func_name == 'initialize' or func_name.startswith('setup_'):
            hints['return'] = 'None'
        elif func_name.startswith('validate_'):
            hints['return'] = 'bool'
        elif func_name.startswith('run_') or func_name.startswith('execute_'):
            hints['return'] = 'None'
        
        return hints


class TypeHintAnalyzer:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.files_needing_hints = []
        
    def analyze_file(self, file_path: Path) -> Dict:
        """Analyze a file for missing type hints."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            functions_without_hints = []
            total_functions = 0
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    total_functions += 1
                    
                    # Skip private methods and special methods
                    if node.name.startswith('_') and not node.name.startswith('__'):
                        continue
                    
                    # Check for missing type hints
                    has_return_type = node.returns is not None
                    has_any_param_type = any(
                        arg.annotation is not None 
                        for arg in node.args.args 
                        if arg.arg not in ['self', 'cls']
                    )
                    
                    if not has_return_type or not has_any_param_type:
                        functions_without_hints.append({
                            'name': node.name,
                            'line': node.lineno,
                            'has_return_type': has_return_type,
                            'has_param_types': has_any_param_type,
                            'params': [arg.arg for arg in node.args.args if arg.arg not in ['self', 'cls']]
                        })
            
            return {
                'file': str(file_path),
                'total_functions': total_functions,
                'functions_without_hints': functions_without_hints,
                'coverage': 1 - (len(functions_without_hints) / total_functions) if total_functions > 0 else 1.0
            }
            
        except Exception as e:
            return {
                'file': str(file_path),
                'error': str(e)
            }
    
    def analyze_project(self) -> Dict:
        """Analyze the entire project for type hint coverage."""
        print("Analyzing type hint coverage...")
        
        # Find Python files
        python_files = list(self.project_root.rglob("*.py"))
        
        # Filter out excluded directories
        python_files = [
            f for f in python_files 
            if '__pycache__' not in str(f) and '.venv' not in str(f)
        ]
        
        results = {
            'total_files': len(python_files),
            'files_analyzed': 0,
            'total_functions': 0,
            'functions_with_hints': 0,
            'files_needing_hints': []
        }
        
        for file_path in python_files[:50]:  # Analyze first 50 files as sample
            analysis = self.analyze_file(file_path)
            
            if 'error' not in analysis:
                results['files_analyzed'] += 1
                results['total_functions'] += analysis['total_functions']
                results['functions_with_hints'] += int(
                    analysis['total_functions'] * analysis['coverage']
                )
                
                if analysis['functions_without_hints']:
                    results['files_needing_hints'].append(analysis)
        
        results['overall_coverage'] = (
            results['functions_with_hints'] / results['total_functions']
            if results['total_functions'] > 0 else 0
        )
        
        return results
    
    def suggest_type_hints_for_file(self, file_path: str) -> List[Dict]:
        """Generate type hint suggestions for a specific file."""
        suggestions = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            adder = TypeHintAdder()
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    hints = adder._infer_type_hints(node)
                    
                    if hints['params'] or hints['return']:
                        suggestion = {
                            'function': node.name,
                            'line': node.lineno,
                            'suggested_signature': self._build_signature(node, hints)
                        }
                        suggestions.append(suggestion)
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
        
        return suggestions
    
    def _build_signature(self, node, hints) -> str:
        """Build a function signature with type hints."""
        params = []
        
        for i, arg in enumerate(node.args.args):
            if arg.arg in ['self', 'cls']:
                params.append(arg.arg)
            elif i < len(hints['params']) and hints['params'][i]:
                params.append(f"{arg.arg}: {hints['params'][i]}")
            else:
                params.append(arg.arg)
        
        param_str = ', '.join(params)
        return_str = f" -> {hints['return']}" if hints['return'] else ""
        
        return f"def {node.name}({param_str}){return_str}:"


def create_type_stub_file(module_path: str, output_path: str):
    """Create a .pyi stub file with type hints."""
    analyzer = TypeHintAnalyzer('/')
    
    try:
        with open(module_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        stub_content = []
        
        # Add imports
        stub_content.append("from typing import Any, Dict, List, Optional, Union, Tuple")
        stub_content.append("import pandas as pd")
        stub_content.append("import numpy as np")
        stub_content.append("from pathlib import Path")
        stub_content.append("import datetime")
        stub_content.append("")
        
        # Process classes and functions
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                stub_content.append(f"class {node.name}:")
                # Add class methods
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        adder = TypeHintAdder()
                        hints = adder._infer_type_hints(item)
                        sig = analyzer._build_signature(item, hints)
                        stub_content.append(f"    {sig} ...")
                stub_content.append("")
            
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                adder = TypeHintAdder()
                hints = adder._infer_type_hints(node)
                sig = analyzer._build_signature(node, hints)
                prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
                stub_content.append(f"{prefix}{sig} ...")
                stub_content.append("")
        
        # Write stub file
        with open(output_path, 'w') as f:
            f.write('\n'.join(stub_content))
        
        return True
    
    except Exception as e:
        print(f"Error creating stub file: {e}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Add type hints to Python code')
    parser.add_argument('--project-root', default='/workspace/src',
                       help='Root directory to analyze')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze type hint coverage')
    parser.add_argument('--suggest', type=str,
                       help='Suggest type hints for a specific file')
    parser.add_argument('--create-stub', type=str,
                       help='Create a .pyi stub file for a module')
    
    args = parser.parse_args()
    
    analyzer = TypeHintAnalyzer(args.project_root)
    
    if args.analyze:
        results = analyzer.analyze_project()
        
        print("\nType Hint Coverage Analysis")
        print("=" * 60)
        print(f"Files analyzed: {results['files_analyzed']} / {results['total_files']}")
        print(f"Total functions: {results['total_functions']}")
        print(f"Functions with hints: {results['functions_with_hints']}")
        print(f"Overall coverage: {results['overall_coverage']:.1%}")
        
        print("\nFiles needing type hints (top 10):")
        for file_info in results['files_needing_hints'][:10]:
            print(f"\n{Path(file_info['file']).name}:")
            print(f"  Coverage: {file_info['coverage']:.1%}")
            print(f"  Functions without hints: {len(file_info['functions_without_hints'])}")
            for func in file_info['functions_without_hints'][:3]:
                print(f"    - {func['name']} (line {func['line']})")
        
        # Save report
        report_path = '/workspace/code_quality/type_hints_report.json'
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nFull report saved to: {report_path}")
    
    elif args.suggest:
        suggestions = analyzer.suggest_type_hints_for_file(args.suggest)
        
        print(f"\nType hint suggestions for {args.suggest}:")
        print("=" * 60)
        
        for sugg in suggestions[:10]:
            print(f"\nLine {sugg['line']}: {sugg['function']}")
            print(f"Suggested: {sugg['suggested_signature']}")
    
    elif args.create_stub:
        output_path = args.create_stub.replace('.py', '.pyi')
        if create_type_stub_file(args.create_stub, output_path):
            print(f"Created stub file: {output_path}")
        else:
            print("Failed to create stub file")


if __name__ == '__main__':
    main()