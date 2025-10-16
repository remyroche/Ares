#!/usr/bin/env python3
"""
Standalone dead code analysis script to find unused code.
This script avoids circular import issues by implementing a simplified analyzer.
"""

import ast
import os
import json
from pathlib import Path
from collections import defaultdict
import re
from typing import Dict, List, Any, Set, Tuple

class SimpleDeadCodeAnalyzer:
    """Simple dead code analyzer that finds unused functions and classes."""
    
    def __init__(self):
        self.issues = []
        self.function_definitions = {}
        self.class_definitions = {}
        self.function_calls = set()
        self.class_usage = set()
        
    def analyze_directory(self, directory: str) -> Dict[str, Any]:
        """Analyze a directory for dead code."""
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")
        
        print(f"Analyzing directory: {directory}")
        
        # Find all Python files
        python_files = list(directory.rglob("*.py"))
        print(f"Found {len(python_files)} Python files")
        
        # First pass: collect all definitions
        for file_path in python_files:
            if self._should_skip_file(file_path):
                continue
            self._collect_definitions(file_path)
        
        # Second pass: collect all usage
        for file_path in python_files:
            if self._should_skip_file(file_path):
                continue
            self._collect_usage(file_path)
        
        # Third pass: find dead code
        self._find_dead_code()
        
        return self._generate_report()
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        skip_patterns = [
            '__pycache__',
            '.git',
            'venv',
            'env',
            'node_modules',
            '.pytest_cache',
            'test_',
            '_test',
            'tests/',
            '/test/',
            'conftest.py'
        ]
        
        file_str = str(file_path)
        return any(pattern in file_str for pattern in skip_patterns)
    
    def _collect_definitions(self, file_path: Path) -> None:
        """Collect function and class definitions from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_key = f"{file_path}::{node.name}"
                    self.function_definitions[func_key] = {
                        'file_path': str(file_path),
                        'line_number': node.lineno,
                        'name': node.name,
                        'is_async': False,
                        'has_docstring': ast.get_docstring(node) is not None,
                        'decorators': [self._get_decorator_name(dec) for dec in node.decorator_list]
                    }
                elif isinstance(node, ast.AsyncFunctionDef):
                    func_key = f"{file_path}::{node.name}"
                    self.function_definitions[func_key] = {
                        'file_path': str(file_path),
                        'line_number': node.lineno,
                        'name': node.name,
                        'is_async': True,
                        'has_docstring': ast.get_docstring(node) is not None,
                        'decorators': [self._get_decorator_name(dec) for dec in node.decorator_list]
                    }
                elif isinstance(node, ast.ClassDef):
                    class_key = f"{file_path}::{node.name}"
                    self.class_definitions[class_key] = {
                        'file_path': str(file_path),
                        'line_number': node.lineno,
                        'name': node.name,
                        'has_docstring': ast.get_docstring(node) is not None,
                        'decorators': [self._get_decorator_name(dec) for dec in node.decorator_list]
                    }
                    
        except Exception as e:
            print(f"Warning: Failed to parse {file_path}: {e}")
    
    def _collect_usage(self, file_path: Path) -> None:
        """Collect function and class usage from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            self._extract_usage_from_ast(tree, file_path)
            
        except Exception as e:
            print(f"Warning: Failed to analyze usage in {file_path}: {e}")
    
    def _extract_usage_from_ast(self, tree: ast.AST, file_path: Path) -> None:
        """Extract usage patterns from AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    # Direct function call
                    self.function_calls.add(f"{file_path}::{node.func.id}")
                elif isinstance(node.func, ast.Attribute):
                    # Method call
                    self.function_calls.add(f"{file_path}::{node.func.attr}")
            elif isinstance(node, ast.Name):
                # Variable/class usage
                if node.id in [cls['name'] for cls in self.class_definitions.values()]:
                    self.class_usage.add(f"{file_path}::{node.id}")
            elif isinstance(node, ast.Attribute):
                # Attribute access
                if node.attr in [cls['name'] for cls in self.class_definitions.values()]:
                    self.class_usage.add(f"{file_path}::{node.attr}")
    
    def _find_dead_code(self) -> None:
        """Find dead code based on collected definitions and usage."""
        # Find unused functions
        for func_key, func_info in self.function_definitions.items():
            if not self._is_function_used(func_key, func_info):
                self.issues.append({
                    'type': 'unused_function',
                    'file_path': func_info['file_path'],
                    'line_number': func_info['line_number'],
                    'name': func_info['name'],
                    'description': f"Function '{func_info['name']}' is defined but never called",
                    'confidence': self._calculate_confidence(func_info),
                    'severity': self._calculate_severity(func_info),
                    'is_async': func_info['is_async'],
                    'has_docstring': func_info['has_docstring'],
                    'decorators': func_info['decorators']
                })
        
        # Find unused classes
        for class_key, class_info in self.class_definitions.items():
            if not self._is_class_used(class_key, class_info):
                self.issues.append({
                    'type': 'unused_class',
                    'file_path': class_info['file_path'],
                    'line_number': class_info['line_number'],
                    'name': class_info['name'],
                    'description': f"Class '{class_info['name']}' is defined but never used",
                    'confidence': self._calculate_confidence(class_info),
                    'severity': self._calculate_severity(class_info),
                    'has_docstring': class_info['has_docstring'],
                    'decorators': class_info['decorators']
                })
    
    def _is_function_used(self, func_key: str, func_info: Dict) -> bool:
        """Check if a function is used."""
        func_name = func_info['name']
        
        # Skip special functions
        if self._is_special_function(func_name):
            return True
        
        # Check if function is called
        if func_key in self.function_calls:
            return True
        
        # Check for string references (dynamic usage)
        file_path = Path(func_info['file_path'])
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for string references
            string_patterns = [f'"{func_name}"', f"'{func_name}'"]
            for pattern in string_patterns:
                if pattern in content:
                    return True
                    
        except Exception:
            pass
        
        return False
    
    def _is_class_used(self, class_key: str, class_info: Dict) -> bool:
        """Check if a class is used."""
        class_name = class_info['name']
        
        # Skip special classes
        if class_name.startswith('_'):
            return True
        
        # Check if class is used
        if class_key in self.class_usage:
            return True
        
        # Check for string references
        file_path = Path(class_info['file_path'])
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            string_patterns = [f'"{class_name}"', f"'{class_name}'"]
            for pattern in string_patterns:
                if pattern in content:
                    return True
                    
        except Exception:
            pass
        
        return False
    
    def _is_special_function(self, func_name: str) -> bool:
        """Check if function is special (should not be considered dead code)."""
        special_patterns = [
            r'^__\w+__$',  # Special methods
            r'^test_',     # Test functions
            r'^setup_',    # Setup functions
            r'^teardown_', # Teardown functions
            r'^main$',     # Main function
        ]
        
        for pattern in special_patterns:
            if re.match(pattern, func_name):
                return True
        
        return False
    
    def _calculate_confidence(self, item_info: Dict) -> float:
        """Calculate confidence score for an issue."""
        confidence = 80.0  # Base confidence
        
        # Reduce confidence for documented items
        if item_info.get('has_docstring', False):
            confidence -= 10.0
        
        # Reduce confidence for items with decorators
        if item_info.get('decorators'):
            confidence -= 5.0
        
        return max(0.0, min(100.0, confidence))
    
    def _calculate_severity(self, item_info: Dict) -> str:
        """Calculate severity for an issue."""
        if item_info.get('has_docstring', False):
            return 'low'
        elif item_info.get('decorators'):
            return 'medium'
        else:
            return 'high'
    
    def _get_decorator_name(self, decorator: ast.AST) -> str:
        """Get decorator name."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return decorator.attr
        else:
            return str(decorator)
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate analysis report."""
        # Group issues by type
        issues_by_type = defaultdict(int)
        issues_by_file = defaultdict(list)
        issues_by_severity = defaultdict(list)
        
        for issue in self.issues:
            issues_by_type[issue['type']] += 1
            issues_by_file[issue['file_path']].append(issue)
            issues_by_severity[issue['severity']].append(issue)
        
        # Calculate confidence distribution
        high_confidence = len([i for i in self.issues if i['confidence'] >= 80])
        medium_confidence = len([i for i in self.issues if 60 <= i['confidence'] < 80])
        low_confidence = len([i for i in self.issues if i['confidence'] < 60])
        
        return {
            'total_issues': len(self.issues),
            'issues_by_type': dict(issues_by_type),
            'issues_by_file': dict(issues_by_file),
            'issues_by_severity': dict(issues_by_severity),
            'confidence_distribution': {
                'high': high_confidence,
                'medium': medium_confidence,
                'low': low_confidence
            },
            'detailed_issues': self.issues,
            'summary': {
                'total_functions_analyzed': len(self.function_definitions),
                'total_classes_analyzed': len(self.class_definitions),
                'unused_functions': len([i for i in self.issues if i['type'] == 'unused_function']),
                'unused_classes': len([i for i in self.issues if i['type'] == 'unused_class'])
            }
        }

def main():
    """Main function to run the analysis."""
    print("🔍 DEAD CODE ANALYSIS")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = SimpleDeadCodeAnalyzer()
    
    # Run analysis on src directory
    try:
        report = analyzer.analyze_directory('/workspace/src')
        
        # Print summary
        print(f"\n📊 ANALYSIS SUMMARY")
        print("-" * 30)
        print(f"Total issues found: {report['total_issues']}")
        print(f"Unused functions: {report['summary']['unused_functions']}")
        print(f"Unused classes: {report['summary']['unused_classes']}")
        print(f"Total functions analyzed: {report['summary']['total_functions_analyzed']}")
        print(f"Total classes analyzed: {report['summary']['total_classes_analyzed']}")
        
        # Print issues by type
        print(f"\n📋 ISSUES BY TYPE")
        print("-" * 30)
        for issue_type, count in report['issues_by_type'].items():
            print(f"{issue_type}: {count}")
        
        # Print issues by severity
        print(f"\n⚠️ ISSUES BY SEVERITY")
        print("-" * 30)
        for severity, issues in report['issues_by_severity'].items():
            print(f"{severity}: {len(issues)}")
        
        # Print confidence distribution
        print(f"\n🎯 CONFIDENCE DISTRIBUTION")
        print("-" * 30)
        for level, count in report['confidence_distribution'].items():
            print(f"{level}: {count}")
        
        # Print top files with issues
        print(f"\n📁 TOP FILES WITH ISSUES")
        print("-" * 30)
        file_issue_counts = [(file_path, len(issues)) for file_path, issues in report['issues_by_file'].items()]
        file_issue_counts.sort(key=lambda x: x[1], reverse=True)
        for file_path, count in file_issue_counts[:10]:
            print(f"{file_path}: {count} issues")
        
        # Print some example issues
        print(f"\n🔍 EXAMPLE ISSUES (High Confidence)")
        print("-" * 30)
        high_conf_issues = [i for i in report['detailed_issues'] if i['confidence'] >= 80]
        for issue in high_conf_issues[:5]:
            print(f"• {issue['file_path']}:{issue['line_number']} - {issue['description']} (confidence: {issue['confidence']:.1f}%)")
        
        # Save detailed report
        output_file = '/workspace/dead_code_analysis_report.json'
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()