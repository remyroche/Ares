#!/usr/bin/env python3
"""
Comprehensive Repository Analysis
Analyzes the entire codebase for code quality issues without external dependencies.
"""

import ast
import os
import json
import time
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

class RepositoryAnalyzer:
    def __init__(self, root_dir="."):
        self.root_dir = Path(root_dir)
        self.results = {
            "analysis_timestamp": datetime.now().isoformat(),
            "root_directory": str(self.root_dir),
            "files_analyzed": 0,
            "total_lines": 0,
            "complexity_analysis": {},
            "import_analysis": {},
            "function_analysis": {},
            "class_analysis": {},
            "issues_found": [],
            "summary": {}
        }
        
        # Directories to skip
        self.skip_dirs = {
            "__pycache__", ".git", "venv", "env", "node_modules", 
            ".pytest_cache", "build", "dist", "mlruns", "artifacts", 
            "logs", "log", ".mypy_cache", ".coverage", "htmlcov"
        }
    
    def should_skip_file(self, file_path):
        """Check if file should be skipped."""
        # Skip hidden files and common non-source files
        if file_path.name.startswith('.') or file_path.name.startswith('_'):
            return True
        
        # Skip test files for now to focus on main code
        if 'test' in file_path.name.lower():
            return True
            
        return False
    
    def find_python_files(self):
        """Find all Python files in the repository."""
        python_files = []
        
        for root, dirs, files in os.walk(self.root_dir):
            # Remove directories we want to skip
            dirs[:] = [d for d in dirs if d not in self.skip_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    if not self.should_skip_file(file_path):
                        python_files.append(file_path)
        
        return python_files
    
    def analyze_file_complexity(self, file_path):
        """Analyze cyclomatic complexity of a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            lines = content.splitlines()
            
            complexity = 1  # Base complexity
            functions = []
            classes = []
            imports = []
            
            # Count complexity factors
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
                        'complexity': func_complexity,
                        'args': len(node.args.args),
                        'is_async': isinstance(node, ast.AsyncFunctionDef)
                    })
                elif isinstance(node, ast.ClassDef):
                    classes.append({
                        'name': node.name,
                        'line': node.lineno,
                        'methods': len([n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))])
                    })
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append({
                                'type': 'import',
                                'name': alias.name,
                                'line': node.lineno,
                                'asname': alias.asname
                            })
                    else:
                        for alias in node.names:
                            imports.append({
                                'type': 'from_import',
                                'name': alias.name,
                                'line': node.lineno,
                                'module': node.module,
                                'asname': alias.asname
                            })
            
            return {
                'file': str(file_path),
                'total_complexity': complexity,
                'lines_of_code': len(lines),
                'non_empty_lines': len([line for line in lines if line.strip()]),
                'functions': functions,
                'classes': classes,
                'imports': imports,
                'success': True
            }
        except Exception as e:
            return {
                'file': str(file_path),
                'error': str(e),
                'success': False
            }
    
    def find_potential_issues(self, file_analysis):
        """Find potential code quality issues."""
        issues = []
        
        if not file_analysis.get('success', False):
            return issues
        
        file_path = file_analysis['file']
        
        # Check for high complexity
        if file_analysis['total_complexity'] > 20:
            issues.append({
                'type': 'high_complexity',
                'severity': 'high',
                'file': file_path,
                'message': f"File has high complexity: {file_analysis['total_complexity']}",
                'value': file_analysis['total_complexity']
            })
        
        # Check for long files
        if file_analysis['lines_of_code'] > 500:
            issues.append({
                'type': 'long_file',
                'severity': 'medium',
                'file': file_path,
                'message': f"File is very long: {file_analysis['lines_of_code']} lines",
                'value': file_analysis['lines_of_code']
            })
        
        # Check for complex functions
        for func in file_analysis['functions']:
            if func['complexity'] > 10:
                issues.append({
                    'type': 'complex_function',
                    'severity': 'high',
                    'file': file_path,
                    'function': func['name'],
                    'line': func['line'],
                    'message': f"Function '{func['name']}' has high complexity: {func['complexity']}",
                    'value': func['complexity']
                })
        
        # Check for functions with many parameters
        for func in file_analysis['functions']:
            if func['args'] > 5:
                issues.append({
                    'type': 'many_parameters',
                    'severity': 'medium',
                    'file': file_path,
                    'function': func['name'],
                    'line': func['line'],
                    'message': f"Function '{func['name']}' has many parameters: {func['args']}",
                    'value': func['args']
                })
        
        return issues
    
    def analyze_repository(self):
        """Run comprehensive analysis on the repository."""
        print("🔍 COMPREHENSIVE REPOSITORY ANALYSIS")
        print("=" * 60)
        
        # Find all Python files
        print("📁 Finding Python files...")
        python_files = self.find_python_files()
        print(f"Found {len(python_files)} Python files to analyze")
        
        # Analyze each file
        print("\n📊 Analyzing files...")
        file_analyses = []
        total_issues = []
        
        for i, file_path in enumerate(python_files):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(python_files)} files analyzed")
            
            analysis = self.analyze_file_complexity(file_path)
            file_analyses.append(analysis)
            
            if analysis.get('success', False):
                self.results['files_analyzed'] += 1
                self.results['total_lines'] += analysis['lines_of_code']
                
                # Find issues in this file
                issues = self.find_potential_issues(analysis)
                total_issues.extend(issues)
        
        self.results['issues_found'] = total_issues
        
        # Generate summary statistics
        self.generate_summary(file_analyses)
        
        return self.results
    
    def generate_summary(self, file_analyses):
        """Generate summary statistics."""
        successful_analyses = [a for a in file_analyses if a.get('success', False)]
        
        if not successful_analyses:
            return
        
        # Complexity statistics
        complexities = [a['total_complexity'] for a in successful_analyses]
        line_counts = [a['lines_of_code'] for a in successful_analyses]
        
        # Function statistics
        all_functions = []
        for analysis in successful_analyses:
            all_functions.extend(analysis['functions'])
        
        function_complexities = [f['complexity'] for f in all_functions]
        
        # Issue statistics
        issue_types = Counter(issue['type'] for issue in self.results['issues_found'])
        issue_severities = Counter(issue['severity'] for issue in self.results['issues_found'])
        
        self.results['summary'] = {
            'total_files': len(successful_analyses),
            'total_lines': sum(line_counts),
            'average_complexity': sum(complexities) / len(complexities) if complexities else 0,
            'max_complexity': max(complexities) if complexities else 0,
            'average_lines_per_file': sum(line_counts) / len(line_counts) if line_counts else 0,
            'max_lines_per_file': max(line_counts) if line_counts else 0,
            'total_functions': len(all_functions),
            'average_function_complexity': sum(function_complexities) / len(function_complexities) if function_complexities else 0,
            'max_function_complexity': max(function_complexities) if function_complexities else 0,
            'total_issues': len(self.results['issues_found']),
            'issues_by_type': dict(issue_types),
            'issues_by_severity': dict(issue_severities)
        }
    
    def print_results(self):
        """Print analysis results in a readable format."""
        summary = self.results['summary']
        
        print("\n" + "=" * 60)
        print("📈 ANALYSIS SUMMARY")
        print("=" * 60)
        
        print(f"📁 Files analyzed: {summary['total_files']}")
        print(f"📄 Total lines of code: {summary['total_lines']:,}")
        print(f"📊 Average lines per file: {summary['average_lines_per_file']:.1f}")
        print(f"🔧 Total functions: {summary['total_functions']}")
        
        print(f"\n🎯 COMPLEXITY METRICS")
        print(f"   Average file complexity: {summary['average_complexity']:.1f}")
        print(f"   Maximum file complexity: {summary['max_complexity']}")
        print(f"   Average function complexity: {summary['average_function_complexity']:.1f}")
        print(f"   Maximum function complexity: {summary['max_function_complexity']}")
        
        print(f"\n⚠️  ISSUES FOUND")
        print(f"   Total issues: {summary['total_issues']}")
        
        if summary['issues_by_severity']:
            print(f"   High severity: {summary['issues_by_severity'].get('high', 0)}")
            print(f"   Medium severity: {summary['issues_by_severity'].get('medium', 0)}")
            print(f"   Low severity: {summary['issues_by_severity'].get('low', 0)}")
        
        if summary['issues_by_type']:
            print(f"\n📋 Issues by type:")
            for issue_type, count in summary['issues_by_type'].items():
                print(f"   {issue_type}: {count}")
        
        # Show top issues
        if self.results['issues_found']:
            print(f"\n🚨 TOP ISSUES")
            print("-" * 40)
            
            # Sort by severity and value
            severity_order = {'high': 3, 'medium': 2, 'low': 1}
            sorted_issues = sorted(
                self.results['issues_found'],
                key=lambda x: (severity_order.get(x['severity'], 0), x.get('value', 0)),
                reverse=True
            )
            
            for issue in sorted_issues[:10]:  # Show top 10
                print(f"   {issue['severity'].upper()}: {issue['message']}")
                if 'function' in issue:
                    print(f"      Function: {issue['function']} (line {issue['line']})")
                print(f"      File: {Path(issue['file']).name}")
                print()
    
    def save_results(self, output_file=None):
        """Save results to JSON file."""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"comprehensive_analysis_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: {output_file}")
        return output_file

def main():
    """Main function."""
    start_time = time.time()
    
    # Initialize analyzer
    analyzer = RepositoryAnalyzer()
    
    # Run analysis
    results = analyzer.analyze_repository()
    
    # Print results
    analyzer.print_results()
    
    # Save results
    output_file = analyzer.save_results()
    
    end_time = time.time()
    print(f"\n⏱️  Analysis completed in {end_time - start_time:.2f} seconds")
    
    return 0

if __name__ == "__main__":
    exit(main())