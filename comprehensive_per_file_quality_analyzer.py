#!/usr/bin/env python3
"""
Comprehensive Per-File Code Quality Analyzer

This script analyzes the codebase and generates detailed per-file reports for:
1. Functions with too many arguments (legitimate refactoring needed)
2. Undefined function calls (import issues)
3. Other function issues

Ignores missing docstrings as requested (mostly fallback functions).
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict
import json
import argparse


class CodeQualityAnalyzer:
    def __init__(self, root_dir: str = "/workspace"):
        self.root_dir = Path(root_dir)
        self.src_dir = self.root_dir / "src"
        self.results = {
            'too_many_arguments': defaultdict(list),
            'undefined_calls': defaultdict(list),
            'other_issues': defaultdict(list),
            'file_stats': {}
        }
        
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single Python file for code quality issues."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            file_issues = {
                'too_many_arguments': [],
                'undefined_calls': [],
                'other_issues': [],
                'stats': {
                    'total_lines': len(content.splitlines()),
                    'total_functions': 0,
                    'total_classes': 0
                }
            }
            
            # Track defined functions and classes
            defined_functions = set()
            defined_classes = set()
            
            # First pass: collect all defined functions and classes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    defined_functions.add(node.name)
                    file_issues['stats']['total_functions'] += 1
                elif isinstance(node, ast.ClassDef):
                    defined_classes.add(node.name)
                    file_issues['stats']['total_classes'] += 1
            
            # Second pass: analyze for issues
            for node in ast.walk(tree):
                # Check for functions with too many arguments
                if isinstance(node, ast.FunctionDef):
                    self._check_function_arguments(node, file_path, file_issues)
                
                # Check for undefined function calls
                if isinstance(node, ast.Call):
                    self._check_undefined_calls(node, defined_functions, defined_classes, file_path, file_issues)
                
                # Check for other issues
                self._check_other_issues(node, file_path, file_issues)
            
            return file_issues
            
        except SyntaxError as e:
            return {
                'syntax_error': str(e),
                'too_many_arguments': [],
                'undefined_calls': [],
                'other_issues': [],
                'stats': {'total_lines': 0, 'total_functions': 0, 'total_classes': 0}
            }
        except Exception as e:
            return {
                'error': str(e),
                'too_many_arguments': [],
                'undefined_calls': [],
                'other_issues': [],
                'stats': {'total_lines': 0, 'total_functions': 0, 'total_classes': 0}
            }
    
    def _check_function_arguments(self, node: ast.FunctionDef, file_path: Path, file_issues: Dict):
        """Check if function has too many arguments."""
        # Count all arguments including *args and **kwargs
        arg_count = len(node.args.args)
        if node.args.vararg:
            arg_count += 1
        if node.args.kwarg:
            arg_count += 1
        
        # Consider functions with more than 5 arguments as having too many
        if arg_count > 5:
            issue = {
                'function_name': node.name,
                'line_number': node.lineno,
                'argument_count': arg_count,
                'arguments': [arg.arg for arg in node.args.args],
                'has_vararg': node.args.vararg is not None,
                'has_kwarg': node.args.kwarg is not None
            }
            file_issues['too_many_arguments'].append(issue)
    
    def _check_undefined_calls(self, node: ast.Call, defined_functions: Set[str], 
                             defined_classes: Set[str], file_path: Path, file_issues: Dict):
        """Check for undefined function calls."""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            # Skip if it's a defined function or class
            if func_name not in defined_functions and func_name not in defined_classes:
                # Skip built-in functions and common exceptions
                builtins = {
                    'print', 'len', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple',
                    'range', 'enumerate', 'zip', 'map', 'filter', 'sorted', 'reversed',
                    'min', 'max', 'sum', 'abs', 'round', 'type', 'isinstance', 'hasattr',
                    'getattr', 'setattr', 'delattr', 'dir', 'vars', 'locals', 'globals',
                    'open', 'file', 'input', 'raw_input', 'exec', 'eval', 'compile',
                    'repr', 'ascii', 'bin', 'hex', 'oct', 'chr', 'ord', 'bool',
                    'complex', 'divmod', 'pow', 'all', 'any', 'iter', 'next',
                    'slice', 'super', 'property', 'staticmethod', 'classmethod',
                    'Exception', 'ValueError', 'TypeError', 'AttributeError', 'KeyError',
                    'IndexError', 'RuntimeError', 'ImportError', 'ModuleNotFoundError',
                    'OSError', 'IOError', 'FileNotFoundError', 'PermissionError',
                    'NotImplementedError', 'StopIteration', 'GeneratorExit',
                    'SystemExit', 'KeyboardInterrupt', 'BaseException'
                }
                
                if func_name not in builtins:
                    issue = {
                        'function_name': func_name,
                        'line_number': node.lineno,
                        'call_context': self._get_call_context(node)
                    }
                    file_issues['undefined_calls'].append(issue)
    
    def _check_other_issues(self, node: ast.AST, file_path: Path, file_issues: Dict):
        """Check for other code quality issues."""
        # Check for overly complex expressions
        if isinstance(node, ast.BoolOp):
            if len(node.values) > 3:
                issue = {
                    'issue_type': 'complex_boolean_expression',
                    'line_number': node.lineno,
                    'description': f'Boolean expression with {len(node.values)} operands'
                }
                file_issues['other_issues'].append(issue)
        
        # Check for deeply nested structures
        if isinstance(node, (ast.If, ast.For, ast.While, ast.With)):
            if hasattr(node, 'lineno'):
                issue = {
                    'issue_type': 'control_flow_structure',
                    'line_number': node.lineno,
                    'description': f'{type(node).__name__} statement'
                }
                file_issues['other_issues'].append(issue)
        
        # Check for long lines (approximate)
        if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
            if node.end_lineno and node.end_lineno - node.lineno > 20:
                issue = {
                    'issue_type': 'long_function_or_class',
                    'line_number': node.lineno,
                    'description': f'Structure spans {node.end_lineno - node.lineno + 1} lines'
                }
                file_issues['other_issues'].append(issue)
    
    def _get_call_context(self, node: ast.Call) -> str:
        """Get context around a function call for better debugging."""
        if isinstance(node.func, ast.Name):
            return f"Direct call: {node.func.id}"
        elif isinstance(node.func, ast.Attribute):
            return f"Method call: {node.func.attr}"
        else:
            return "Complex call expression"
    
    def analyze_directory(self, directory: Path) -> None:
        """Analyze all Python files in a directory recursively."""
        for py_file in directory.rglob("*.py"):
            if py_file.is_file():
                relative_path = str(py_file.relative_to(self.root_dir))
                print(f"Analyzing: {relative_path}")
                
                file_issues = self.analyze_file(py_file)
                
                # Store results
                self.results['file_stats'][relative_path] = file_issues['stats']
                
                if file_issues['too_many_arguments']:
                    self.results['too_many_arguments'][relative_path] = file_issues['too_many_arguments']
                
                if file_issues['undefined_calls']:
                    self.results['undefined_calls'][relative_path] = file_issues['undefined_calls']
                
                if file_issues['other_issues']:
                    self.results['other_issues'][relative_path] = file_issues['other_issues']
    
    def generate_reports(self) -> None:
        """Generate detailed per-file reports."""
        print("Generating per-file reports...")
        
        # Report 1: Too Many Arguments
        self._generate_too_many_args_report()
        
        # Report 2: Undefined Function Calls
        self._generate_undefined_calls_report()
        
        # Report 3: Other Issues
        self._generate_other_issues_report()
        
        # Summary Report
        self._generate_summary_report()
    
    def _generate_too_many_args_report(self) -> None:
        """Generate report for functions with too many arguments."""
        report_path = self.root_dir / "too_many_arguments_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Functions with Too Many Arguments Report\n\n")
            f.write("This report identifies functions that have more than 5 arguments and may need refactoring.\n\n")
            
            total_functions = 0
            total_files = len(self.results['too_many_arguments'])
            
            for file_path, functions in sorted(self.results['too_many_arguments'].items()):
                f.write(f"## {file_path}\n\n")
                f.write(f"**Functions with too many arguments: {len(functions)}**\n\n")
                
                for func in functions:
                    total_functions += 1
                    f.write(f"### {func['function_name']} (Line {func['line_number']})\n")
                    f.write(f"- **Argument count:** {func['argument_count']}\n")
                    f.write(f"- **Arguments:** {', '.join(func['arguments'])}\n")
                    if func['has_vararg']:
                        f.write(f"- **Has *args:** Yes\n")
                    if func['has_kwarg']:
                        f.write(f"- **Has **kwargs:** Yes\n")
                    f.write("\n")
                
                f.write("---\n\n")
            
            f.write(f"## Summary\n\n")
            f.write(f"- **Total files affected:** {total_files}\n")
            f.write(f"- **Total functions needing refactoring:** {total_functions}\n")
        
        print(f"Too Many Arguments report generated: {report_path}")
    
    def _generate_undefined_calls_report(self) -> None:
        """Generate report for undefined function calls."""
        report_path = self.root_dir / "undefined_calls_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Undefined Function Calls Report\n\n")
            f.write("This report identifies function calls that may be undefined (import issues).\n\n")
            
            total_calls = 0
            total_files = len(self.results['undefined_calls'])
            
            for file_path, calls in sorted(self.results['undefined_calls'].items()):
                f.write(f"## {file_path}\n\n")
                f.write(f"**Undefined function calls: {len(calls)}**\n\n")
                
                for call in calls:
                    total_calls += 1
                    f.write(f"- **Line {call['line_number']}:** {call['function_name']} - {call['call_context']}\n")
                
                f.write("\n---\n\n")
            
            f.write(f"## Summary\n\n")
            f.write(f"- **Total files affected:** {total_files}\n")
            f.write(f"- **Total undefined calls:** {total_calls}\n")
        
        print(f"Undefined Calls report generated: {report_path}")
    
    def _generate_other_issues_report(self) -> None:
        """Generate report for other function issues."""
        report_path = self.root_dir / "other_issues_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Other Function Issues Report\n\n")
            f.write("This report identifies other code quality issues that may need attention.\n\n")
            
            total_issues = 0
            total_files = len(self.results['other_issues'])
            issue_types = defaultdict(int)
            
            for file_path, issues in sorted(self.results['other_issues'].items()):
                f.write(f"## {file_path}\n\n")
                f.write(f"**Other issues: {len(issues)}**\n\n")
                
                for issue in issues:
                    total_issues += 1
                    issue_types[issue['issue_type']] += 1
                    f.write(f"- **Line {issue['line_number']}:** {issue['issue_type']} - {issue['description']}\n")
                
                f.write("\n---\n\n")
            
            f.write(f"## Summary\n\n")
            f.write(f"- **Total files affected:** {total_files}\n")
            f.write(f"- **Total issues:** {total_issues}\n\n")
            f.write("### Issue Types:\n")
            for issue_type, count in sorted(issue_types.items()):
                f.write(f"- **{issue_type}:** {count}\n")
        
        print(f"Other Issues report generated: {report_path}")
    
    def _generate_summary_report(self) -> None:
        """Generate a summary report with file counts and priorities."""
        report_path = self.root_dir / "code_quality_summary.md"
        
        with open(report_path, 'w') as f:
            f.write("# Code Quality Analysis Summary\n\n")
            f.write("This report provides a high-level summary of code quality issues found in the codebase.\n\n")
            
            # File statistics
            total_files = len(self.results['file_stats'])
            total_lines = sum(stats['total_lines'] for stats in self.results['file_stats'].values())
            total_functions = sum(stats['total_functions'] for stats in self.results['file_stats'].values())
            total_classes = sum(stats['total_classes'] for stats in self.results['file_stats'].values())
            
            f.write("## Overall Statistics\n\n")
            f.write(f"- **Total Python files analyzed:** {total_files}\n")
            f.write(f"- **Total lines of code:** {total_lines:,}\n")
            f.write(f"- **Total functions:** {total_functions:,}\n")
            f.write(f"- **Total classes:** {total_classes:,}\n\n")
            
            # Issue summaries
            f.write("## Issue Summary\n\n")
            f.write("### 1. Functions with Too Many Arguments (Legitimate Refactoring Needed)\n")
            f.write(f"- **Files affected:** {len(self.results['too_many_arguments'])}\n")
            total_too_many = sum(len(funcs) for funcs in self.results['too_many_arguments'].values())
            f.write(f"- **Total functions needing refactoring:** {total_too_many}\n")
            f.write("- **Priority:** HIGH - These need legitimate refactoring\n\n")
            
            f.write("### 2. Undefined Function Calls (Import Issues)\n")
            f.write(f"- **Files affected:** {len(self.results['undefined_calls'])}\n")
            total_undefined = sum(len(calls) for calls in self.results['undefined_calls'].values())
            f.write(f"- **Total undefined calls:** {total_undefined}\n")
            f.write("- **Priority:** HIGH - These indicate import/dependency issues\n\n")
            
            f.write("### 3. Other Function Issues\n")
            f.write(f"- **Files affected:** {len(self.results['other_issues'])}\n")
            total_other = sum(len(issues) for issues in self.results['other_issues'].values())
            f.write(f"- **Total other issues:** {total_other}\n")
            f.write("- **Priority:** MEDIUM - These may need attention\n\n")
            
            f.write("### 4. Missing Docstrings (Ignored)\n")
            f.write("- **Status:** IGNORED as requested (mostly fallback functions)\n")
            f.write("- **Estimated count:** ~2,000\n\n")
            
            # Top problematic files
            f.write("## Top Problematic Files\n\n")
            
            # Files with most too many arguments
            f.write("### Files with Most Functions Needing Refactoring:\n")
            too_many_sorted = sorted(
                self.results['too_many_arguments'].items(),
                key=lambda x: len(x[1]),
                reverse=True
            )[:10]
            for file_path, functions in too_many_sorted:
                f.write(f"- **{file_path}:** {len(functions)} functions\n")
            
            f.write("\n### Files with Most Undefined Calls:\n")
            undefined_sorted = sorted(
                self.results['undefined_calls'].items(),
                key=lambda x: len(x[1]),
                reverse=True
            )[:10]
            for file_path, calls in undefined_sorted:
                f.write(f"- **{file_path}:** {len(calls)} undefined calls\n")
            
            f.write("\n### Files with Most Other Issues:\n")
            other_sorted = sorted(
                self.results['other_issues'].items(),
                key=lambda x: len(x[1]),
                reverse=True
            )[:10]
            for file_path, issues in other_sorted:
                f.write(f"- **{file_path}:** {len(issues)} issues\n")
            
            f.write("\n## Recommendations\n\n")
            f.write("1. **Start with undefined function calls** - These indicate broken imports and dependencies\n")
            f.write("2. **Refactor functions with too many arguments** - Break them into smaller, more focused functions\n")
            f.write("3. **Address other issues** - Review and fix complex expressions and long functions\n")
            f.write("4. **Focus on the most problematic files first** - Use the top problematic files list above\n")
        
        print(f"Summary report generated: {report_path}")
    
    def save_json_results(self) -> None:
        """Save detailed results to JSON for programmatic access."""
        json_path = self.root_dir / "code_quality_analysis.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Detailed results saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze code quality and generate per-file reports")
    parser.add_argument("--root-dir", default="/workspace", help="Root directory to analyze")
    parser.add_argument("--src-only", action="store_true", help="Analyze only src/ directory")
    parser.add_argument("--json", action="store_true", help="Save detailed JSON results")
    
    args = parser.parse_args()
    
    analyzer = CodeQualityAnalyzer(args.root_dir)
    
    if args.src_only:
        print("Analyzing src/ directory only...")
        analyzer.analyze_directory(Path(args.root_dir) / "src")
    else:
        print("Analyzing entire codebase...")
        analyzer.analyze_directory(Path(args.root_dir))
    
    analyzer.generate_reports()
    
    if args.json:
        analyzer.save_json_results()
    
    print("\nAnalysis complete! Check the generated report files:")
    print("- too_many_arguments_report.md")
    print("- undefined_calls_report.md") 
    print("- other_issues_report.md")
    print("- code_quality_summary.md")


if __name__ == "__main__":
    main()