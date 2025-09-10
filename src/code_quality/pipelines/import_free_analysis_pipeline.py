#!/usr/bin/env python3
"""
Import-Free Analysis Pipeline

This pipeline performs comprehensive code analysis without requiring external imports,
focusing on static analysis of Python code structure, method references, and attribute access patterns.

Usage:
    python src/code_quality/pipelines/import_free_analysis_pipeline.py --analysis-type all
"""

import ast
import os
import re
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any, Optional
import argparse
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class ImportFreeAnalyzer:
    """Performs comprehensive analysis without external dependencies."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        self.results = {}
        
    def analyze_file(self, file_path: Path, analysis_types: List[str]) -> Dict[str, Any]:
        """Analyze a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            file_results = {
                'file': str(file_path),
                'timestamp': datetime.now().isoformat(),
                'analysis_types': analysis_types,
                'results': {}
            }
            
            for analysis_type in analysis_types:
                if analysis_type == 'method_references':
                    file_results['results']['method_references'] = self._analyze_method_references(tree, file_path, content)
                elif analysis_type == 'attribute_access':
                    file_results['results']['attribute_access'] = self._analyze_attribute_access(tree, file_path, content)
                elif analysis_type == 'import_analysis':
                    file_results['results']['import_analysis'] = self._analyze_imports(tree, file_path)
                elif analysis_type == 'class_structure':
                    file_results['results']['class_structure'] = self._analyze_class_structure(tree, file_path)
                elif analysis_type == 'function_complexity':
                    file_results['results']['function_complexity'] = self._analyze_function_complexity(tree, file_path)
                elif analysis_type == 'enhanced_method_references':
                    file_results['results']['enhanced_method_references'] = self._run_enhanced_method_analyzer(file_path)
                elif analysis_type == 'enhanced_attribute_access':
                    file_results['results']['enhanced_attribute_access'] = self._run_enhanced_attribute_analyzer(file_path)
            
            return file_results
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"Error analyzing {file_path}: {e}")
            return {
                'file': str(file_path),
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _analyze_method_references(self, tree: ast.AST, file_path: Path, content: str) -> Dict[str, Any]:
        """Analyze method references in the file."""
        issues = []
        classes = {}
        
        # Extract classes
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes[node.name] = node
        
        # Analyze each class
        for class_name, class_node in classes.items():
            # Get defined methods
            defined_methods = set()
            for node in class_node.body:
                if isinstance(node, ast.FunctionDef):
                    defined_methods.add(node.name)
            
            # Find method calls
            method_calls = []
            for node in ast.walk(class_node):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        if isinstance(node.func.value, ast.Name) and node.func.value.id == 'self':
                            method_name = node.func.attr
                            method_calls.append({
                                'method': method_name,
                                'line': node.lineno,
                                'defined': method_name in defined_methods
                            })
            
            # Check for missing methods
            for call in method_calls:
                if not call['defined']:
                    issues.append({
                        'type': 'missing_method',
                        'class': class_name,
                        'method': call['method'],
                        'line': call['line'],
                        'severity': 'error'
                    })
        
        return {
            'total_classes': len(classes),
            'total_method_calls': sum(len(self._get_method_calls(c)) for c in classes.values()),
            'missing_methods': len(issues),
            'issues': issues
        }
    
    def _analyze_attribute_access(self, tree: ast.AST, file_path: Path, content: str) -> Dict[str, Any]:
        """Analyze attribute access patterns."""
        issues = []
        classes = {}
        
        # Extract classes
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes[node.name] = node
        
        # Analyze each class
        for class_name, class_node in classes.items():
            # Get assigned attributes
            assigned_attrs = set()
            for node in class_node.body:
                if isinstance(node, ast.FunctionDef) and node.name == '__init__':
                    for init_node in ast.walk(node):
                        if isinstance(init_node, ast.Assign):
                            for target in init_node.targets:
                                if isinstance(target, ast.Attribute):
                                    if isinstance(target.value, ast.Name) and target.value.id == 'self':
                                        assigned_attrs.add(target.attr)
            
            # Find attribute access
            accessed_attrs = set()
            for node in ast.walk(class_node):
                if isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name) and node.value.id == 'self':
                        accessed_attrs.add(node.attr)
            
            # Check for unsafe access
            unsafe_access = accessed_attrs - assigned_attrs
            for attr in unsafe_access:
                issues.append({
                    'type': 'unsafe_attribute_access',
                    'class': class_name,
                    'attribute': attr,
                    'severity': 'warning'
                })
        
        return {
            'total_classes': len(classes),
            'unsafe_access_count': len(issues),
            'issues': issues
        }
    
    def _analyze_imports(self, tree: ast.AST, file_path: Path) -> Dict[str, Any]:
        """Analyze import statements."""
        imports = []
        import_issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'type': 'import',
                        'module': alias.name,
                        'alias': alias.asname,
                        'line': node.lineno
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append({
                        'type': 'from_import',
                        'module': module,
                        'name': alias.name,
                        'alias': alias.asname,
                        'line': node.lineno
                    })
        
        # Check for common import issues
        for imp in imports:
            if imp['module'].startswith('...'):
                import_issues.append({
                    'type': 'relative_import',
                    'import': imp,
                    'severity': 'warning'
                })
        
        return {
            'total_imports': len(imports),
            'import_issues': len(import_issues),
            'imports': imports,
            'issues': import_issues
        }
    
    def _analyze_class_structure(self, tree: ast.AST, file_path: Path) -> Dict[str, Any]:
        """Analyze class structure and inheritance."""
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    'name': node.name,
                    'line': node.lineno,
                    'bases': [base.id if isinstance(base, ast.Name) else str(base) for base in node.bases],
                    'methods': [],
                    'attributes': []
                }
                
                # Extract methods and attributes
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        class_info['methods'].append({
                            'name': item.name,
                            'line': item.lineno,
                            'args': len(item.args.args),
                            'is_private': item.name.startswith('_')
                        })
                    elif isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                class_info['attributes'].append(target.id)
                
                classes.append(class_info)
        
        return {
            'total_classes': len(classes),
            'classes': classes
        }
    
    def _analyze_function_complexity(self, tree: ast.AST, file_path: Path) -> Dict[str, Any]:
        """Analyze function complexity."""
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self._calculate_complexity(node)
                functions.append({
                    'name': node.name,
                    'line': node.lineno,
                    'complexity': complexity,
                    'lines': node.end_lineno - node.lineno if node.end_lineno else 0,
                    'args': len(node.args.args)
                })
        
        # Find complex functions
        complex_functions = [f for f in functions if f['complexity'] > 10]
        
        return {
            'total_functions': len(functions),
            'complex_functions': len(complex_functions),
            'functions': functions,
            'complex_function_details': complex_functions
        }
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _run_enhanced_method_analyzer(self, file_path: Path) -> Dict[str, Any]:
        """Run the enhanced method reference analyzer."""
        try:
            # Import and run our enhanced method analyzer
            import sys
            project_root = Path(__file__).parent.parent.parent.parent
            sys.path.insert(0, str(project_root))
            from data_quality.method_reference_analyzer import MethodReferenceAnalyzer
            
            analyzer = MethodReferenceAnalyzer(verbose=self.verbose)
            result = analyzer.analyze_file(file_path)
            
            return {
                'enhanced_analysis': True,
                'total_issues': result.get('summary', {}).get('total_issues', 0),
                'errors': result.get('summary', {}).get('errors', 0),
                'warnings': result.get('summary', {}).get('warnings', 0),
                'issues': result.get('issues', [])
            }
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"Enhanced method analyzer failed: {e}")
            return {
                'enhanced_analysis': False,
                'error': str(e),
                'total_issues': 0,
                'errors': 0,
                'warnings': 0
            }
    
    def _run_enhanced_attribute_analyzer(self, file_path: Path) -> Dict[str, Any]:
        """Run the enhanced attribute access analyzer."""
        try:
            # Import and run our enhanced attribute analyzer
            import sys
            project_root = Path(__file__).parent.parent.parent.parent
            sys.path.insert(0, str(project_root))
            from data_quality.attribute_access_analyzer import AttributeAccessAnalyzer
            
            analyzer = AttributeAccessAnalyzer(verbose=self.verbose)
            result = analyzer.analyze_file(file_path)
            
            return {
                'enhanced_analysis': True,
                'total_issues': result.get('summary', {}).get('total_issues', 0),
                'errors': result.get('summary', {}).get('errors', 0),
                'warnings': result.get('summary', {}).get('warnings', 0),
                'issues': result.get('issues', [])
            }
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"Enhanced attribute analyzer failed: {e}")
            return {
                'enhanced_analysis': False,
                'error': str(e),
                'total_issues': 0,
                'errors': 0,
                'warnings': 0
            }
    
    def _get_method_calls(self, class_node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Get all method calls in a class."""
        method_calls = []
        for node in ast.walk(class_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == 'self':
                        method_calls.append({
                            'method': node.func.attr,
                            'line': node.lineno
                        })
        return method_calls
    
    def analyze_directory(self, directory: Path, analysis_types: List[str]) -> Dict[str, Any]:
        """Analyze all Python files in a directory."""
        all_results = {
            'directory': str(directory),
            'timestamp': datetime.now().isoformat(),
            'analysis_types': analysis_types,
            'files': [],
            'summary': {}
        }
        
        total_issues = 0
        total_files = 0
        
        for py_file in directory.rglob("*.py"):
            if self.verbose:
                print(f"Analyzing {py_file}")
            
            file_result = self.analyze_file(py_file, analysis_types)
            all_results['files'].append(file_result)
            total_files += 1
            
            # Count issues
            if 'results' in file_result:
                for analysis_type in analysis_types:
                    if analysis_type in file_result['results']:
                        result = file_result['results'][analysis_type]
                        if 'issues' in result:
                            total_issues += len(result['issues'])
        
        all_results['summary'] = {
            'total_files': total_files,
            'total_issues': total_issues,
            'analysis_types': analysis_types
        }
        
        return all_results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive report."""
        report = []
        report.append("🔍 Import-Free Analysis Report")
        report.append("=" * 50)
        report.append(f"Analysis Time: {results['timestamp']}")
        report.append(f"Directory: {results['directory']}")
        report.append(f"Analysis Types: {', '.join(results['analysis_types'])}")
        report.append("")
        
        summary = results['summary']
        report.append("📊 SUMMARY:")
        report.append(f"  Total Files: {summary['total_files']}")
        report.append(f"  Total Issues: {summary['total_issues']}")
        report.append("")
        
        # Detailed results by analysis type
        for analysis_type in results['analysis_types']:
            report.append(f"📋 {analysis_type.upper().replace('_', ' ')}:")
            
            type_issues = 0
            type_files = 0
            
            for file_result in results['files']:
                if 'results' in file_result and analysis_type in file_result['results']:
                    type_files += 1
                    result = file_result['results'][analysis_type]
                    if 'issues' in result:
                        type_issues += len(result['issues'])
            
            report.append(f"  Files Analyzed: {type_files}")
            report.append(f"  Issues Found: {type_issues}")
            report.append("")
        
        return "\n".join(report)


def main():
    """Main entry point for the import-free analysis pipeline."""
    parser = argparse.ArgumentParser(description="Import-Free Analysis Pipeline")
    parser.add_argument("--path", default="src", help="Path to analyze")
    parser.add_argument("--analysis-type", default="all", 
                       choices=["all", "method_references", "attribute_access", "import_analysis", 
                               "class_structure", "function_complexity", "enhanced_method_references", 
                               "enhanced_attribute_access"],
                       help="Type of analysis to perform")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", "-o", help="Output file for results")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    
    # Determine analysis types
    if args.analysis_type == "all":
        analysis_types = ["method_references", "attribute_access", "import_analysis", 
                         "class_structure", "function_complexity", "enhanced_method_references", 
                         "enhanced_attribute_access"]
    else:
        analysis_types = [args.analysis_type]
    
    # Create analyzer
    analyzer = ImportFreeAnalyzer(verbose=args.verbose)
    
    # Analyze directory
    scan_path = Path(args.path)
    if not scan_path.exists():
        print(f"❌ Path {scan_path} does not exist")
        sys.exit(1)
    
    print(f"🔍 Starting import-free analysis of {scan_path}...")
    results = analyzer.analyze_directory(scan_path, analysis_types)
    
    # Generate report
    report = analyzer.generate_report(results)
    print(report)
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📄 Results saved to {args.output}")
    
    # Exit with appropriate code
    if results['summary']['total_issues'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
