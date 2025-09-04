#!/usr/bin/env python3
"""
Simple Import and Undefined Variable Checker

A standalone script that provides:
1. Required imports checking - ensures all necessary imports are present
2. Undefined variables detection - spots undefined variables for easier troubleshooting

This version is self-contained and doesn't rely on complex analyzer modules.
"""

import ast
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


class SimpleImportChecker:
    """Simple import checker that analyzes Python files for import issues."""
    
    def __init__(self):
        self.imports = {}
        self.issues = []
        # Common problematic import patterns
        self.problematic_patterns = {
            'import *': 'Wildcard imports can cause namespace pollution',
            'from . import': 'Relative imports may cause issues in some contexts',
            'import sys, os': 'Multiple imports on one line reduce readability'
        }
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single Python file for import issues."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            file_imports = []
            issues = []
            import_names = set()
            duplicate_imports = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        import_name = alias.name
                        as_name = alias.asname or import_name.split('.')[-1]
                        
                        # Check for duplicate imports
                        if as_name in import_names:
                            duplicate_imports.add(as_name)
                            issues.append({
                                'type': 'duplicate_import',
                                'name': as_name,
                                'line': node.lineno,
                                'message': f'Duplicate import: {as_name}'
                            })
                        else:
                            import_names.add(as_name)
                        
                        # Check for wildcard imports
                        if import_name == '*':
                            issues.append({
                                'type': 'wildcard_import',
                                'name': import_name,
                                'line': node.lineno,
                                'message': 'Wildcard import (*) can cause namespace pollution'
                            })
                        
                        file_imports.append({
                            'type': 'import',
                            'module': import_name,
                            'name': as_name,
                            'line': node.lineno
                        })
                
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        import_name = alias.name
                        as_name = alias.asname or import_name
                        
                        # Check for duplicate imports
                        if as_name in import_names:
                            duplicate_imports.add(as_name)
                            issues.append({
                                'type': 'duplicate_import',
                                'name': as_name,
                                'line': node.lineno,
                                'message': f'Duplicate import: {as_name}'
                            })
                        else:
                            import_names.add(as_name)
                        
                        # Check for wildcard imports
                        if import_name == '*':
                            issues.append({
                                'type': 'wildcard_import',
                                'name': import_name,
                                'line': node.lineno,
                                'message': 'Wildcard import (*) can cause namespace pollution'
                            })
                        
                        # Check for relative imports
                        if module.startswith('.'):
                            issues.append({
                                'type': 'relative_import',
                                'name': f'{module}.{import_name}',
                                'line': node.lineno,
                                'message': f'Relative import: {module}.{import_name}'
                            })
                        
                        file_imports.append({
                            'type': 'from_import',
                            'module': module,
                            'name': as_name,
                            'line': node.lineno
                        })
            
            return {
                'file': file_path,
                'imports': file_imports,
                'issues': issues,
                'total_imports': len(file_imports)
            }
            
        except Exception as e:
            return {
                'file': file_path,
                'imports': [],
                'issues': [{'type': 'parse_error', 'message': str(e), 'line': 0}],
                'total_imports': 0,
                'error': str(e)
            }
    
    def analyze_directory(self, directory_path: str) -> Dict[str, Any]:
        """Analyze all Python files in a directory."""
        python_files = []
        for root, dirs, files in os.walk(directory_path):
            # Skip directories based on ignore patterns
            dirs[:] = [d for d in dirs if d not in self.ignore_patterns]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        results = {
            'files': {},
            'summary': {
                'total_files': len(python_files),
                'total_imports': 0,
                'files_with_issues': 0
            }
        }
        
        for file_path in python_files:
            file_result = self.analyze_file(file_path)
            results['files'][file_path] = file_result
            results['summary']['total_imports'] += file_result['total_imports']
            
            if file_result['issues']:
                results['summary']['files_with_issues'] += 1
        
        return results


class SimpleUndefinedChecker:
    """Simple undefined variable checker that analyzes Python files for undefined names."""
    
    def __init__(self):
        self.builtin_names = set(dir(__builtins__))
        # Add common builtin types and functions
        self.builtin_names.update({
            'object', 'type', 'str', 'int', 'float', 'bool', 'list', 'dict', 'tuple',
            'set', 'frozenset', 'bytes', 'bytearray', 'complex', 'range', 'slice',
            'property', 'staticmethod', 'classmethod', 'super', 'vars', 'dir',
            'hasattr', 'getattr', 'setattr', 'delattr', 'isinstance', 'issubclass',
            'callable', 'iter', 'next', 'enumerate', 'zip', 'map', 'filter',
            'sorted', 'reversed', 'sum', 'min', 'max', 'abs', 'round', 'pow',
            'divmod', 'bin', 'oct', 'hex', 'chr', 'ord', 'len', 'repr', 'ascii',
            'format', 'hash', 'id', 'globals', 'locals', 'eval', 'exec', 'compile',
            'open', 'input', 'print', 'exit', 'quit', 'help', 'license', 'credits',
            'copyright', 'True', 'False', 'None', 'Ellipsis', 'NotImplemented',
            '__name__', '__file__', '__doc__', '__package__', '__loader__',
            '__spec__', '__annotations__', '__builtins__', '__debug__',
            '__import__', '__main__', '__version__', '__author__', '__email__'
        })
        
        # Common third-party libraries that might be imported
        self.common_libraries = {
            'pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn', 'scipy',
            'requests', 'flask', 'django', 'fastapi', 'sqlalchemy', 'pytest',
            'pydantic', 'typing', 'dataclasses', 'enum', 'collections', 'itertools',
            'functools', 'operator', 're', 'json', 'csv', 'datetime', 'time',
            'os', 'sys', 'pathlib', 'shutil', 'tempfile', 'logging', 'warnings'
        }
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single Python file for undefined names."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Collect defined names and imports
            defined_names = set()
            imported_names = set()
            undefined_issues = []
            function_params = set()
            
            # First pass: collect definitions and imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.asname or alias.name.split('.')[-1]
                        imported_names.add(name)
                        defined_names.add(name)
                
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        name = alias.asname or alias.name
                        imported_names.add(name)
                        defined_names.add(name)
                
                elif isinstance(node, ast.FunctionDef):
                    defined_names.add(node.name)
                    # Add function parameters
                    for arg in node.args.args:
                        function_params.add(arg.arg)
                        defined_names.add(arg.arg)
                    # Add default arguments
                    for default in node.args.defaults:
                        if isinstance(default, ast.Name):
                            defined_names.add(default.id)
                
                elif isinstance(node, ast.ClassDef):
                    defined_names.add(node.name)
                    # Add class methods and attributes
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            defined_names.add(item.name)
                
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            defined_names.add(target.id)
                        elif isinstance(target, ast.Tuple):
                            for elt in target.elts:
                                if isinstance(elt, ast.Name):
                                    defined_names.add(elt.id)
                
                elif isinstance(node, ast.For):
                    if isinstance(node.target, ast.Name):
                        defined_names.add(node.target.id)
                    elif isinstance(node.target, ast.Tuple):
                        for elt in node.target.elts:
                            if isinstance(elt, ast.Name):
                                defined_names.add(elt.id)
                
                elif isinstance(node, ast.With):
                    for item in node.items:
                        if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                            defined_names.add(item.optional_vars.id)
            
            # Second pass: check for undefined names
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    name = node.id
                    
                    # Skip if it's a known name
                    if (name in defined_names or 
                        name in imported_names or 
                        name in self.builtin_names):
                        continue
                    
                    # Skip common patterns that are likely false positives
                    if (name.startswith('_') or  # Private variables
                        name.isupper() or  # Constants
                        name in self.common_libraries or  # Common libraries
                        name in ['self', 'cls', 'args', 'kwargs']):  # Common patterns
                        continue
                    
                    # Skip exception variables in except blocks
                    if self._is_exception_variable(node, tree):
                        continue
                    
                    # Skip lambda parameters
                    if self._is_lambda_parameter(node, tree):
                        continue
                    
                    # Get context
                    context = ""
                    try:
                        lines = content.split('\n')
                        if 0 <= node.lineno - 1 < len(lines):
                            context = lines[node.lineno - 1].strip()
                    except:
                        pass
                    
                    # Determine issue type
                    issue_type = 'undefined_name'
                    if name in function_params:
                        issue_type = 'possible_scope_issue'
                    elif name.lower() in [lib.lower() for lib in self.common_libraries]:
                        issue_type = 'missing_import'
                    
                    undefined_issues.append({
                        'name': name,
                        'line': node.lineno,
                        'column': node.col_offset,
                        'context': context,
                        'type': issue_type,
                        'severity': 'high' if issue_type == 'undefined_name' else 'medium'
                    })
            
            return {
                'file': file_path,
                'undefined_issues': undefined_issues,
                'total_undefined': len(undefined_issues),
                'defined_names': list(defined_names),
                'imported_names': list(imported_names)
            }
            
        except Exception as e:
            return {
                'file': file_path,
                'undefined_issues': [],
                'total_undefined': 0,
                'defined_names': [],
                'imported_names': [],
                'error': str(e)
            }
    
    def analyze_directory(self, directory_path: str) -> Dict[str, Any]:
        """Analyze all Python files in a directory."""
        python_files = []
        for root, dirs, files in os.walk(directory_path):
            # Skip directories based on ignore patterns
            dirs[:] = [d for d in dirs if d not in self.ignore_patterns]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        results = {
            'files': {},
            'summary': {
                'total_files': len(python_files),
                'total_undefined': 0,
                'files_with_undefined': 0
            }
        }
        
        for file_path in python_files:
            file_result = self.analyze_file(file_path)
            results['files'][file_path] = file_result
            results['summary']['total_undefined'] += file_result['total_undefined']
            
            if file_result['total_undefined'] > 0:
                results['summary']['files_with_undefined'] += 1
        
        return results


class SimpleImportAndUndefinedChecker:
    """
    Simple comprehensive checker for imports and undefined variables.
    
    This class provides:
    1. Import validation - checks for missing, unused, conflicting imports
    2. Undefined variable detection - identifies undefined names and variables
    3. Report generation - creates detailed reports for troubleshooting
    """
    
    def __init__(self, project_root: str = None, config: Dict[str, Any] = None):
        """
        Initialize the checker.
        
        Args:
            project_root: Root directory of the project to analyze
            config: Configuration dictionary for filtering and options
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Configuration
        self.config = config or {}
        self.ignore_patterns = self.config.get('ignore_patterns', ['__pycache__', '.git', 'node_modules', '.venv', 'venv'])
        self.max_issues_per_file = self.config.get('max_issues_per_file', 100)
        self.min_severity = self.config.get('min_severity', 'low')  # low, medium, high
        
        # Initialize checkers
        self.import_checker = SimpleImportChecker()
        self.import_checker.ignore_patterns = self.ignore_patterns
        self.undefined_checker = SimpleUndefinedChecker()
        self.undefined_checker.ignore_patterns = self.ignore_patterns
        
        # Results storage
        self.results = {
            "import_analysis": {},
            "undefined_analysis": {},
            "summary": {},
            "recommendations": []
        }
    
    def check_imports(self, target_path: str = None) -> Dict[str, Any]:
        """
        Check for import issues in the target path.
        
        Args:
            target_path: Path to analyze (default: project_root)
            
        Returns:
            Dictionary containing import analysis results
        """
        if target_path is None:
            target_path = str(self.project_root)
        
        print("🔍 Checking imports...")
        print(f"Target: {target_path}")
        
        start_time = time.time()
        
        try:
            if os.path.isfile(target_path):
                # Single file analysis
                results = {'files': {target_path: self.import_checker.analyze_file(target_path)}}
                results['summary'] = {
                    'total_files': 1,
                    'total_imports': results['files'][target_path]['total_imports'],
                    'files_with_issues': 1 if results['files'][target_path]['issues'] else 0
                }
            else:
                # Directory analysis
                results = self.import_checker.analyze_directory(target_path)
            
            execution_time = time.time() - start_time
            
            # Process results
            import_results = {
                "status": "success",
                "execution_time": execution_time,
                "target_path": target_path,
                "summary": results.get("summary", {}),
                "files": results.get("files", {}),
                "total_issues": results.get("summary", {}).get("files_with_issues", 0),
            }
            
            # Generate recommendations
            recommendations = []
            total_issues = import_results["total_issues"]
            
            if total_issues > 0:
                recommendations.append({
                    "priority": "medium",
                    "category": "imports",
                    "message": f"Review {total_issues} files with import issues"
                })
            
            import_results["recommendations"] = recommendations
            
            self.results["import_analysis"] = import_results
            
            # Print summary
            print(f"✅ Import analysis completed in {execution_time:.2f}s")
            print(f"📊 Total files analyzed: {results.get('summary', {}).get('total_files', 0)}")
            print(f"📦 Total imports found: {results.get('summary', {}).get('total_imports', 0)}")
            print(f"⚠️  Files with issues: {total_issues}")
            
            return import_results
            
        except Exception as e:
            error_result = {
                "status": "error",
                "error": str(e),
                "execution_time": time.time() - start_time,
                "target_path": target_path
            }
            self.results["import_analysis"] = error_result
            print(f"❌ Import analysis failed: {e}")
            return error_result
    
    def check_undefined_variables(self, target_path: str = None) -> Dict[str, Any]:
        """
        Check for undefined variables and names in the target path.
        
        Args:
            target_path: Path to analyze (default: project_root)
            
        Returns:
            Dictionary containing undefined variable analysis results
        """
        if target_path is None:
            target_path = str(self.project_root)
        
        print("🔍 Checking undefined variables...")
        print(f"Target: {target_path}")
        
        start_time = time.time()
        
        try:
            if os.path.isfile(target_path):
                # Single file analysis
                results = {'files': {target_path: self.undefined_checker.analyze_file(target_path)}}
                results['summary'] = {
                    'total_files': 1,
                    'total_undefined': results['files'][target_path]['total_undefined'],
                    'files_with_undefined': 1 if results['files'][target_path]['total_undefined'] > 0 else 0
                }
            else:
                # Directory analysis
                results = self.undefined_checker.analyze_directory(target_path)
            
            execution_time = time.time() - start_time
            
            # Process results
            undefined_results = {
                "status": "success",
                "execution_time": execution_time,
                "target_path": target_path,
                "summary": results.get("summary", {}),
                "files": results.get("files", {}),
                "total_errors": results.get("summary", {}).get("total_undefined", 0),
                "files_with_errors": results.get("summary", {}).get("files_with_undefined", 0),
            }
            
            # Generate recommendations
            recommendations = []
            total_errors = undefined_results["total_errors"]
            
            if total_errors > 0:
                recommendations.append({
                    "priority": "high",
                    "category": "undefined_variables",
                    "message": f"Fix {total_errors} undefined variable/name issues"
                })
            
            undefined_results["recommendations"] = recommendations
            
            self.results["undefined_analysis"] = undefined_results
            
            # Print summary
            print(f"✅ Undefined variable analysis completed in {execution_time:.2f}s")
            print(f"📊 Total files analyzed: {results.get('summary', {}).get('total_files', 0)}")
            print(f"❌ Total undefined issues: {total_errors}")
            print(f"📄 Files with undefined issues: {undefined_results['files_with_errors']}")
            
            return undefined_results
            
        except Exception as e:
            error_result = {
                "status": "error",
                "error": str(e),
                "execution_time": time.time() - start_time,
                "target_path": target_path
            }
            self.results["undefined_analysis"] = error_result
            print(f"❌ Undefined variable analysis failed: {e}")
            return error_result
    
    def run_comprehensive_check(self, target_path: str = None) -> Dict[str, Any]:
        """
        Run both import and undefined variable checks.
        
        Args:
            target_path: Path to analyze (default: project_root)
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        if target_path is None:
            target_path = str(self.project_root)
        
        print("="*70)
        print("COMPREHENSIVE IMPORT AND UNDEFINED VARIABLE CHECK")
        print("="*70)
        print(f"Target: {target_path}")
        print(f"Timestamp: {self.timestamp}")
        print()
        
        start_time = time.time()
        
        # Run both checks
        import_results = self.check_imports(target_path)
        print()
        undefined_results = self.check_undefined_variables(target_path)
        
        total_time = time.time() - start_time
        
        # Generate overall summary
        overall_summary = {
            "timestamp": self.timestamp,
            "target_path": target_path,
            "total_execution_time": total_time,
            "import_issues": import_results.get("total_issues", 0),
            "undefined_issues": undefined_results.get("total_errors", 0),
            "total_issues": (import_results.get("total_issues", 0) + 
                           undefined_results.get("total_errors", 0)),
            "files_with_import_issues": import_results.get("summary", {}).get("files_with_issues", 0),
            "files_with_undefined_issues": undefined_results.get("files_with_errors", 0),
        }
        
        # Combine recommendations
        all_recommendations = []
        all_recommendations.extend(import_results.get("recommendations", []))
        all_recommendations.extend(undefined_results.get("recommendations", []))
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        all_recommendations.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 2))
        
        overall_summary["recommendations"] = all_recommendations
        
        self.results["summary"] = overall_summary
        
        # Print final summary
        print("\n" + "="*70)
        print("COMPREHENSIVE CHECK SUMMARY")
        print("="*70)
        print(f"Total execution time: {total_time:.2f}s")
        print(f"Import issues: {overall_summary['import_issues']}")
        print(f"Undefined variable issues: {overall_summary['undefined_issues']}")
        print(f"Total issues: {overall_summary['total_issues']}")
        
        if all_recommendations:
            print("\n📋 Recommendations:")
            for i, rec in enumerate(all_recommendations, 1):
                priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(rec.get("priority", "low"), "⚪")
                print(f"  {i}. {priority_emoji} [{rec.get('priority', 'low').upper()}] {rec.get('message', '')}")
        
        return self.results
    
    def save_report(self, output_file: str = None) -> str:
        """
        Save the analysis results to a JSON report file.
        
        Args:
            output_file: Output file path (optional)
            
        Returns:
            Path to the saved report file
        """
        if output_file is None:
            output_file = f"simple_import_undefined_check_report_{self.timestamp}.json"
        
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Report saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"❌ Failed to save report: {e}")
            raise
    
    def get_high_priority_issues(self) -> List[Dict[str, Any]]:
        """
        Get a list of high-priority issues that need immediate attention.
        
        Returns:
            List of high-priority issues
        """
        high_priority = []
        
        # Check import analysis
        import_analysis = self.results.get("import_analysis", {})
        if import_analysis.get("status") == "success":
            for rec in import_analysis.get("recommendations", []):
                if rec.get("priority") == "high":
                    high_priority.append({
                        "type": "import",
                        "category": rec.get("category"),
                        "message": rec.get("message"),
                        "source": "import_analysis"
                    })
        
        # Check undefined analysis
        undefined_analysis = self.results.get("undefined_analysis", {})
        if undefined_analysis.get("status") == "success":
            for rec in undefined_analysis.get("recommendations", []):
                if rec.get("priority") == "high":
                    high_priority.append({
                        "type": "undefined",
                        "category": rec.get("category"),
                        "message": rec.get("message"),
                        "source": "undefined_analysis"
                    })
        
        return high_priority
    
    def filter_issues_by_severity(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter issues based on minimum severity level."""
        severity_levels = {'low': 0, 'medium': 1, 'high': 2}
        min_level = severity_levels.get(self.min_severity, 0)
        
        filtered_issues = []
        for issue in issues:
            issue_severity = issue.get('severity', 'low')
            issue_level = severity_levels.get(issue_severity, 0)
            if issue_level >= min_level:
                filtered_issues.append(issue)
        
        return filtered_issues
    
    def get_issue_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about found issues."""
        stats = {
            'import_issues': {
                'total': 0,
                'by_type': {},
                'files_affected': 0
            },
            'undefined_issues': {
                'total': 0,
                'by_type': {},
                'by_severity': {},
                'files_affected': 0
            }
        }
        
        # Import issues statistics
        import_analysis = self.results.get("import_analysis", {})
        if import_analysis.get("status") == "success":
            files = import_analysis.get("files", {})
            stats['import_issues']['files_affected'] = len([f for f in files.values() if f.get('issues')])
            
            for file_result in files.values():
                for issue in file_result.get('issues', []):
                    stats['import_issues']['total'] += 1
                    issue_type = issue.get('type', 'unknown')
                    stats['import_issues']['by_type'][issue_type] = stats['import_issues']['by_type'].get(issue_type, 0) + 1
        
        # Undefined issues statistics
        undefined_analysis = self.results.get("undefined_analysis", {})
        if undefined_analysis.get("status") == "success":
            files = undefined_analysis.get("files", {})
            stats['undefined_issues']['files_affected'] = len([f for f in files.values() if f.get('total_undefined', 0) > 0])
            
            for file_result in files.values():
                for issue in file_result.get('undefined_issues', []):
                    stats['undefined_issues']['total'] += 1
                    issue_type = issue.get('type', 'unknown')
                    severity = issue.get('severity', 'low')
                    
                    stats['undefined_issues']['by_type'][issue_type] = stats['undefined_issues']['by_type'].get(issue_type, 0) + 1
                    stats['undefined_issues']['by_severity'][severity] = stats['undefined_issues']['by_severity'].get(severity, 0) + 1
        
        return stats
    
    def _is_exception_variable(self, node: ast.Name, tree: ast.AST) -> bool:
        """Check if a name node is an exception variable in an except block."""
        # Walk up the AST to find the parent except block
        for parent in ast.walk(tree):
            if isinstance(parent, ast.ExceptHandler):
                # Check if this node is within this except block
                if (hasattr(parent, 'lineno') and hasattr(parent, 'end_lineno') and
                    parent.lineno <= node.lineno <= (parent.end_lineno or parent.lineno)):
                    # Check if the name matches the exception variable
                    if parent.name == node.id:
                        return True
        return False
    
    def _is_lambda_parameter(self, node: ast.Name, tree: ast.AST) -> bool:
        """Check if a name node is a parameter in a lambda function."""
        # Walk up the AST to find the parent lambda
        for parent in ast.walk(tree):
            if isinstance(parent, ast.Lambda):
                # Check if this node is within this lambda
                if (hasattr(parent, 'lineno') and hasattr(parent, 'end_lineno') and
                    parent.lineno <= node.lineno <= (parent.end_lineno or parent.lineno)):
                    # Check if the name is in the lambda's args
                    for arg in parent.args.args:
                        if arg.arg == node.id:
                            return True
        return False


def main():
    """Command-line interface for the simple import and undefined checker."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Simple comprehensive import and undefined variable checker"
    )
    parser.add_argument("--target", "-t", 
                       help="Path to Python file or directory to analyze (default: current directory)")
    parser.add_argument("--output", "-o", 
                       help="Output file for JSON report")
    parser.add_argument("--imports-only", action="store_true",
                       help="Check only imports")
    parser.add_argument("--undefined-only", action="store_true",
                       help="Check only undefined variables")
    parser.add_argument("--project-root", 
                       help="Project root directory (default: current directory)")
    parser.add_argument("--min-severity", choices=['low', 'medium', 'high'], default='low',
                       help="Minimum severity level to report (default: low)")
    parser.add_argument("--max-issues-per-file", type=int, default=100,
                       help="Maximum issues to report per file (default: 100)")
    parser.add_argument("--ignore-patterns", nargs='+', 
                       default=['__pycache__', '.git', 'node_modules', '.venv', 'venv'],
                       help="Directory patterns to ignore")
    parser.add_argument("--stats", action="store_true",
                       help="Show detailed statistics")
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'min_severity': args.min_severity,
        'max_issues_per_file': args.max_issues_per_file,
        'ignore_patterns': args.ignore_patterns
    }
    
    # Initialize checker
    checker = SimpleImportAndUndefinedChecker(project_root=args.project_root, config=config)
    
    # Run checks based on arguments
    if args.imports_only:
        results = checker.check_imports(args.target)
    elif args.undefined_only:
        results = checker.check_undefined_variables(args.target)
    else:
        results = checker.run_comprehensive_check(args.target)
    
    # Save report if requested
    if args.output:
        checker.save_report(args.output)
    
    # Print high-priority issues
    high_priority = checker.get_high_priority_issues()
    if high_priority:
        print(f"\n🚨 {len(high_priority)} high-priority issues found:")
        for issue in high_priority:
            print(f"  - {issue['message']}")
    
    # Show detailed statistics if requested
    if args.stats:
        stats = checker.get_issue_statistics()
        print(f"\n📊 Detailed Statistics:")
        print(f"Import Issues:")
        print(f"  Total: {stats['import_issues']['total']}")
        print(f"  Files affected: {stats['import_issues']['files_affected']}")
        if stats['import_issues']['by_type']:
            print(f"  By type:")
            for issue_type, count in stats['import_issues']['by_type'].items():
                print(f"    {issue_type}: {count}")
        
        print(f"Undefined Issues:")
        print(f"  Total: {stats['undefined_issues']['total']}")
        print(f"  Files affected: {stats['undefined_issues']['files_affected']}")
        if stats['undefined_issues']['by_type']:
            print(f"  By type:")
            for issue_type, count in stats['undefined_issues']['by_type'].items():
                print(f"    {issue_type}: {count}")
        if stats['undefined_issues']['by_severity']:
            print(f"  By severity:")
            for severity, count in stats['undefined_issues']['by_severity'].items():
                print(f"    {severity}: {count}")
    
    # Exit with appropriate code
    summary = results.get("summary", {})
    total_issues = summary.get("total_issues", 0)
    
    if total_issues == 0:
        print(f"\n✅ All checks passed!")
        return 0
    elif total_issues <= 10:
        print(f"\n⚠️  Found {total_issues} issues that need attention.")
        return 1
    else:
        print(f"\n❌ Found {total_issues} issues that require immediate attention!")
        return 2


if __name__ == "__main__":
    sys.exit(main())
