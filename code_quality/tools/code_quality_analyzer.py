#!/usr/bin/env python3
"""
Custom Code Quality Analyzer
Analyzes Python files for unused imports, dead code, and other quality issues.
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict
import argparse


class CodeQualityAnalyzer:
    """Analyzes Python code for quality issues."""
    
    def __init__(self, exclusions_file: str = None):
        self.exclusions = self._load_exclusions(exclusions_file)
        self.issues = defaultdict(list)
        self.stats = {
            'files_analyzed': 0,
            'unused_imports_found': 0,
            'dead_code_found': 0,
            'formatting_issues': 0
        }
    
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
    
    def analyze_file(self, filepath: str) -> Dict[str, Any]:
        """Analyze a single Python file for quality issues."""
        if self._should_exclude(filepath):
            return {}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            issues = {
                'unused_imports': self._find_unused_imports(tree, content),
                'dead_code': self._find_dead_code(tree, content),
                'formatting_issues': self._find_formatting_issues(content),
                'duplicate_imports': self._find_duplicate_imports(tree),
                'long_lines': self._find_long_lines(content)
            }
            
            self.stats['files_analyzed'] += 1
            if issues['unused_imports']:
                self.stats['unused_imports_found'] += len(issues['unused_imports'])
            if issues['dead_code']:
                self.stats['dead_code_found'] += len(issues['dead_code'])
            if issues['formatting_issues']:
                self.stats['formatting_issues'] += len(issues['formatting_issues'])
            
            return issues
            
        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"Error analyzing {filepath}: {e}")
            return {}
    
    def _find_unused_imports(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Find unused imports in the AST."""
        imports = []
        used_names = set()
        
        # Collect all imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'type': 'import',
                        'name': alias.name,
                        'asname': alias.asname,
                        'lineno': node.lineno,
                        'used_name': alias.asname or alias.name.split('.')[0]
                    })
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imports.append({
                        'type': 'from_import',
                        'module': node.module,
                        'name': alias.name,
                        'asname': alias.asname,
                        'lineno': node.lineno,
                        'used_name': alias.asname or alias.name
                    })
        
        # Collect all used names
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                # For module.attr usage
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)
        
        # Check for string usage (some imports used in strings)
        for line in content.split('\n'):
            # Look for quoted usage
            for imp in imports:
                if f"'{imp['used_name']}'" in line or f'"{imp["used_name"]}"' in line:
                    used_names.add(imp['used_name'])
        
        # Find unused imports
        unused = []
        for imp in imports:
            if imp['used_name'] not in used_names and imp['name'] != '*':
                unused.append(imp)
        
        return unused
    
    def _find_dead_code(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Find potentially dead code."""
        dead_code = []
        
        # Find unreachable code after return statements
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for i, stmt in enumerate(node.body):
                    if isinstance(stmt, ast.Return):
                        # Check if there are statements after return
                        if i < len(node.body) - 1:
                            dead_code.append({
                                'type': 'unreachable_after_return',
                                'function': node.name,
                                'lineno': node.body[i + 1].lineno,
                                'description': f"Code after return in function '{node.name}'"
                            })
        
        # Find unused functions (basic detection)
        function_names = set()
        called_functions = set()
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith('_') and node.name not in ['main', '__init__']:
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
        
        for func_name in function_names:
            if func_name not in called_functions:
                # Find the line number of the function
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
                        dead_code.append({
                            'type': 'unused_function',
                            'function': func_name,
                            'lineno': node.lineno,
                            'description': f"Function '{func_name}' appears to be unused"
                        })
                        break
        
        return dead_code
    
    def _find_formatting_issues(self, content: str) -> List[Dict[str, Any]]:
        """Find basic formatting issues."""
        issues = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Trailing whitespace
            if line.rstrip() != line:
                issues.append({
                    'type': 'trailing_whitespace',
                    'lineno': i,
                    'description': 'Line has trailing whitespace'
                })
            
            # Mixed tabs and spaces
            if '\t' in line and '    ' in line:
                issues.append({
                    'type': 'mixed_indentation',
                    'lineno': i,
                    'description': 'Mixed tabs and spaces'
                })
        
        return issues
    
    def _find_duplicate_imports(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Find duplicate imports."""
        imports = defaultdict(list)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    key = f"import {alias.name}"
                    imports[key].append(node.lineno)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    key = f"from {node.module} import {alias.name}"
                    imports[key].append(node.lineno)
        
        duplicates = []
        for import_str, lines in imports.items():
            if len(lines) > 1:
                duplicates.append({
                    'type': 'duplicate_import',
                    'import': import_str,
                    'lines': lines,
                    'description': f"Import '{import_str}' appears on lines: {', '.join(map(str, lines))}"
                })
        
        return duplicates
    
    def _find_long_lines(self, content: str, max_length: int = 120) -> List[Dict[str, Any]]:
        """Find lines that are too long."""
        long_lines = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            if len(line) > max_length:
                long_lines.append({
                    'type': 'long_line',
                    'lineno': i,
                    'length': len(line),
                    'description': f'Line is {len(line)} characters (max {max_length})'
                })
        
        return long_lines
    
    def fix_unused_imports(self, filepath: str, dry_run: bool = True) -> bool:
        """Remove unused imports from a file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            unused_imports = self._find_unused_imports(tree, content)
            
            if not unused_imports:
                return False
            
            lines = content.split('\n')
            lines_to_remove = set()
            
            for imp in unused_imports:
                lines_to_remove.add(imp['lineno'] - 1)  # Convert to 0-based indexing
            
            # Remove lines in reverse order to maintain line numbers
            for line_idx in sorted(lines_to_remove, reverse=True):
                if line_idx < len(lines):
                    print(f"Removing unused import at line {line_idx + 1}: {lines[line_idx].strip()}")
                    if not dry_run:
                        lines.pop(line_idx)
            
            if not dry_run and lines_to_remove:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                return True
            
            return bool(lines_to_remove)
            
        except Exception as e:
            print(f"Error fixing {filepath}: {e}")
            return False
    
    def analyze_directory(self, directory: str, fix_imports: bool = False, dry_run: bool = True) -> Dict[str, Any]:
        """Analyze all Python files in a directory."""
        results = {}
        
        for root, dirs, files in os.walk(directory):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not self._should_exclude(os.path.join(root, d))]
            
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    
                    if self._should_exclude(filepath):
                        continue
                    
                    print(f"Analyzing: {filepath}")
                    issues = self.analyze_file(filepath)
                    
                    if issues:
                        results[filepath] = issues
                    
                    if fix_imports and issues.get('unused_imports'):
                        self.fix_unused_imports(filepath, dry_run=dry_run)
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a quality report."""
        report = []
        report.append("=" * 80)
        report.append("CODE QUALITY ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Files analyzed: {self.stats['files_analyzed']}")
        report.append(f"Unused imports found: {self.stats['unused_imports_found']}")
        report.append(f"Dead code issues found: {self.stats['dead_code_found']}")
        report.append(f"Formatting issues found: {self.stats['formatting_issues']}")
        report.append("")
        
        if not results:
            report.append("No issues found!")
            return "\n".join(report)
        
        for filepath, issues in results.items():
            if any(issues.values()):
                report.append(f"\nFile: {filepath}")
                report.append("-" * len(filepath))
                
                for issue_type, issue_list in issues.items():
                    if issue_list:
                        report.append(f"\n{issue_type.replace('_', ' ').title()}:")
                        for issue in issue_list:
                            if isinstance(issue, dict):
                                line_info = f" (line {issue.get('lineno', 'unknown')})" if 'lineno' in issue else ""
                                desc = issue.get('description', str(issue))
                                report.append(f"  - {desc}{line_info}")
                            else:
                                report.append(f"  - {issue}")
        
        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Analyze Python code quality')
    parser.add_argument('directory', help='Directory to analyze')
    parser.add_argument('--fix-imports', action='store_true', help='Fix unused imports')
    parser.add_argument('--no-dry-run', action='store_true', help='Actually apply fixes (not just preview)')
    parser.add_argument('--exclusions', help='Exclusions file path')
    parser.add_argument('--output', help='Output report to file')
    
    args = parser.parse_args()
    
    analyzer = CodeQualityAnalyzer(args.exclusions)
    results = analyzer.analyze_directory(
        args.directory, 
        fix_imports=args.fix_imports,
        dry_run=not args.no_dry_run
    )
    
    report = analyzer.generate_report(results)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == '__main__':
    main()