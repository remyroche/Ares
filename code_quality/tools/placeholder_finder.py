#!/usr/bin/env python3
"""
Enhanced Placeholder Finder
Finds placeholders and functions that need to be implemented in Python code.
This enhanced version provides comprehensive detection of various placeholder patterns.
"""

import os
import re
import argparse
import ast
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
from pathlib import Path
import logging
from datetime import datetime, timezone


class PlaceholderFinder:
    """Enhanced placeholder finder with comprehensive detection capabilities."""

    def __init__(self, exclusions_file: Optional[str] = None):
        """Initialize the PlaceholderFinder.
        
        Args:
            exclusions_file: Path to file containing exclusion patterns
        """
        self.exclusions = self._load_exclusions(exclusions_file)
        self.placeholders = defaultdict(list)
        self.stats = {
            'files_analyzed': 0,
            'total_placeholders': 0,
            'pass_statements': 0,
            'todo_comments': 0,
            'raise_notimplemented': 0,
            'placeholder_functions': 0,
            'empty_classes': 0,
            'stub_functions': 0,
            'unimplemented_methods': 0,
            'placeholder_variables': 0,
            'incomplete_implementations': 0
        }
        
        # Enhanced patterns for detection
        self.todo_patterns = [
            r'#\s*(TODO|FIXME|HACK|XXX|BUG|NOTE|REFACTOR|CLEANUP|REVIEW):?\s*(.+)',
            r'"""\s*(TODO|FIXME|HACK|XXX|BUG|NOTE|REFACTOR|CLEANUP|REVIEW):?\s*(.+)',
            r"'''\s*(TODO|FIXME|HACK|XXX|BUG|NOTE|REFACTOR|CLEANUP|REVIEW):?\s*(.+)",
            r'#\s*(.+?)\s*#\s*(TODO|FIXME|HACK|XXX|BUG|NOTE|REFACTOR|CLEANUP|REVIEW)',
        ]
        
        self.placeholder_patterns = [
            r'#\s*placeholder',
            r'#\s*implement\s+later',
            r'#\s*to\s+be\s+implemented',
            r'#\s*not\s+implemented',
            r'#\s*stub',
            r'#\s*empty\s+for\s+now',
            r'#\s*work\s+in\s+progress',
            r'#\s*wip',
        ]
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Record analysis start time
        self.analysis_start_time = datetime.now(timezone.utc)

    def _load_exclusions(self, exclusions_file: Optional[str]) -> Set[str]:
        """Load exclusion patterns from file.
        
        Args:
            exclusions_file: Path to exclusions file
            
        Returns:
            Set of exclusion patterns
        """
        exclusions = set()
        if exclusions_file and os.path.exists(exclusions_file):
            try:
                with open(exclusions_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            exclusions.add(line)
            except Exception as e:
                self.logger.warning(f"Could not load exclusions file: {e}")
        return exclusions

    def _should_exclude(self, filepath: str) -> bool:
        """Check if file should be excluded from analysis.
        
        Args:
            filepath: Path to file to check
            
        Returns:
            True if file should be excluded
        """
        for pattern in self.exclusions:
            if pattern in filepath or filepath.endswith(pattern.replace('*', '')):
                return True
        return False

    def analyze_file(self, filepath: str) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze a single Python file for placeholders.
        
        Args:
            filepath: Path to Python file to analyze
            
        Returns:
            Dictionary of placeholder issues found
        """
        if self._should_exclude(filepath):
            return {}

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Skip empty files
            if not content.strip():
                return {}

            issues = {
                'pass_statements': self._find_pass_statements(content, filepath),
                'todo_comments': self._find_todo_comments(content, filepath),
                'raise_notimplemented': self._find_raise_notimplemented(content, filepath),
                'placeholder_functions': self._find_placeholder_functions(content, filepath),
                'empty_classes': self._find_empty_classes(content, filepath),
                'stub_functions': self._find_stub_functions(content, filepath),
                'unimplemented_methods': self._find_unimplemented_methods(content, filepath),
                'placeholder_variables': self._find_placeholder_variables(content, filepath),
                'incomplete_implementations': self._find_incomplete_implementations(content, filepath)
            }

            self.stats['files_analyzed'] += 1
            
            # Update statistics
            for key in issues:
                if key in self.stats:
                    self.stats[key] += len(issues[key])

            total_issues = sum(len(issues[key]) for key in issues)
            self.stats['total_placeholders'] += total_issues

            if total_issues > 0:
                self.placeholders[filepath] = issues

            return issues

        except (UnicodeDecodeError, PermissionError, SyntaxError) as e:
            self.logger.warning(f"Error analyzing {filepath}: {e}")
            return {}

    def _find_pass_statements(self, content: str, filepath: str) -> List[Dict[str, Any]]:
        """Find pass statements that might be placeholders.
        
        Args:
            content: File content
            filepath: Path to file
            
        Returns:
            List of pass statement issues
        """
        pass_statements = []
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped == 'pass':
                # Check if there's a TODO comment nearby or if it's in a try/except block
                has_todo = False
                in_try_except = False
                context_lines = []

                # Check nearby lines for TODO comments and context
                for j in range(max(0, i-3), min(len(lines), i+2)):
                    if 'TODO' in lines[j] or 'FIXME' in lines[j]:
                        has_todo = True
                    context_lines.append(lines[j])

                # Check if we're in a try/except block
                for j in range(max(0, i-5), min(len(lines), i+5)):
                    if 'try:' in lines[j] or 'except' in lines[j]:
                        in_try_except = True
                        break

                # Only flag if there's a TODO nearby, it's in a try/except block, or it's isolated
                if has_todo or in_try_except or self._is_isolated_pass(lines, i):
                    pass_statements.append({
                        'type': 'pass_statement',
                        'line': i,
                        'content': line,
                        'has_todo': has_todo,
                        'in_try_except': in_try_except,
                        'context': self._get_context(lines, i, 3)
                    })

        return pass_statements

    def _is_isolated_pass(self, lines: List[str], line_num: int) -> bool:
        """Check if a pass statement is isolated (not in a try/except block).
        
        Args:
            lines: All lines in the file
            line_num: Line number of the pass statement
            
        Returns:
            True if pass is isolated
        """
        # Look for surrounding context
        start = max(0, line_num - 5)
        end = min(len(lines), line_num + 5)
        
        # Check if we're in a try/except block
        in_try_except = False
        for i in range(start, end):
            if 'try:' in lines[i] or 'except' in lines[i]:
                in_try_except = True
                break
        
        if in_try_except:
            return False
            
        # Check if we're in a function or class definition
        for i in range(start, line_num):
            if lines[i].strip().startswith(('def ', 'class ', 'async def ')):
                return True
                
        return False

    def _find_todo_comments(self, content: str, filepath: str) -> List[Dict[str, Any]]:
        """Find TODO and related comments.
        
        Args:
            content: File content
            filepath: Path to file
            
        Returns:
            List of TODO comment issues
        """
        todo_comments = []
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Check all TODO patterns
            for pattern in self.todo_patterns:
                match = re.search(pattern, stripped, re.IGNORECASE)
                if match:
                    # Skip if it's just a comment describing what the code does
                    if not any(phrase in stripped.lower() for phrase in [
                        'find todo', 'find fixme', 'todo comment', 'fixme comment',
                        'todo comments', 'fixme comments', 'example todo'
                    ]):
                        todo_comments.append({
                            'type': 'todo_comment',
                            'line': i,
                            'content': line.strip(),
                            'category': match.group(1).upper(),
                            'description': match.group(2) if len(match.groups()) > 1 else '',
                            'context': self._get_context(lines, i, 2)
                        })
                        break

            # Check placeholder patterns
            for pattern in self.placeholder_patterns:
                if re.search(pattern, stripped, re.IGNORECASE):
                    todo_comments.append({
                        'type': 'placeholder_comment',
                        'line': i,
                        'content': line.strip(),
                        'category': 'PLACEHOLDER',
                        'description': 'Placeholder comment',
                        'context': self._get_context(lines, i, 2)
                    })
                    break

        return todo_comments

    def _find_raise_notimplemented(self, content: str, filepath: str) -> List[Dict[str, Any]]:
        """Find NotImplementedError and NotImplemented raises.
        
        Args:
            content: File content
            filepath: Path to file
            
        Returns:
            List of NotImplemented issues
        """
        not_implemented = []
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            if 'raise NotImplementedError' in line or 'raise NotImplemented' in line:
                not_implemented.append({
                    'type': 'not_implemented',
                    'line': i,
                    'content': line.strip(),
                    'context': self._get_context(lines, i, 3)
                })

        return not_implemented

    def _find_placeholder_functions(self, content: str, filepath: str) -> List[Dict[str, Any]]:
        """Find functions with minimal content that might be placeholders.
        
        Args:
            content: File content
            filepath: Path to file
            
        Returns:
            List of placeholder function issues
        """
        placeholder_functions = []
        lines = content.split('\n')

        # Look for function definitions with minimal content
        for i, line in enumerate(lines, 1):
            if line.strip().startswith('def ') or line.strip().startswith('async def '):
                # Check if function body is minimal (just pass, docstring, or TODO)
                body_start = i
                body_end = self._find_function_end(lines, i)

                if body_end > body_start:
                    body_lines = [l.strip() for l in lines[body_start:body_end] if l.strip()]

                    # Check if body is minimal
                    if len(body_lines) <= 3:  # Just pass, docstring, or TODO
                        is_placeholder = False
                        for body_line in body_lines:
                            if (body_line == 'pass' or
                                body_line.startswith('"""') or
                                body_line.startswith("'''") or
                                'TODO' in body_line or
                                'FIXME' in body_line or
                                'raise NotImplementedError' in body_line or
                                'raise NotImplemented' in body_line):
                                is_placeholder = True
                                break

                        if is_placeholder:
                            placeholder_functions.append({
                                'type': 'placeholder_function',
                                'line': i,
                                'content': line.strip(),
                                'body_lines': body_end - body_start,
                                'context': self._get_context(lines, i, 3)
                            })

        return placeholder_functions

    def _find_empty_classes(self, content: str, filepath: str) -> List[Dict[str, Any]]:
        """Find empty classes that might be placeholders.
        
        Args:
            content: File content
            filepath: Path to file
            
        Returns:
            List of empty class issues
        """
        empty_classes = []
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            if line.strip().startswith('class '):
                body_start = i
                body_end = self._find_class_end(lines, i)

                if body_end > body_start:
                    body_lines = [l.strip() for l in lines[body_start:body_end] if l.strip()]
                    
                    # Check if class is empty or just has pass/docstring
                    if len(body_lines) <= 2:
                        is_empty = True
                        for body_line in body_lines:
                            if not (body_line == 'pass' or 
                                   body_line.startswith('"""') or 
                                   body_line.startswith("'''")):
                                is_empty = False
                                break
                        
                        if is_empty:
                            empty_classes.append({
                                'type': 'empty_class',
                                'line': i,
                                'content': line.strip(),
                                'body_lines': body_end - body_start,
                                'context': self._get_context(lines, i, 3)
                            })

        return empty_classes

    def _find_stub_functions(self, content: str, filepath: str) -> List[Dict[str, Any]]:
        """Find stub functions (functions with just ellipsis).
        
        Args:
            content: File content
            filepath: Path to file
            
        Returns:
            List of stub function issues
        """
        stub_functions = []
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            if line.strip().startswith('def ') or line.strip().startswith('async def '):
                body_start = i
                body_end = self._find_function_end(lines, i)

                if body_end > body_start:
                    body_lines = [l.strip() for l in lines[body_start:body_end] if l.strip()]
                    
                    # Check if function body is just ellipsis
                    if len(body_lines) == 1 and body_lines[0] == '...':
                        stub_functions.append({
                            'type': 'stub_function',
                            'line': i,
                            'content': line.strip(),
                            'context': self._get_context(lines, i, 3)
                        })

        return stub_functions

    def _find_unimplemented_methods(self, content: str, filepath: str) -> List[Dict[str, Any]]:
        """Find methods that are explicitly marked as unimplemented.
        
        Args:
            content: File content
            filepath: Path to file
            
        Returns:
            List of unimplemented method issues
        """
        unimplemented_methods = []
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            if line.strip().startswith('def ') or line.strip().startswith('async def '):
                # Check if method name suggests it's not implemented
                method_name = line.strip().split('(')[0].split()[-1]
                if any(keyword in method_name.lower() for keyword in [
                    'placeholder', 'stub', 'todo', 'unimplemented', 'not_implemented'
                ]):
                    unimplemented_methods.append({
                        'type': 'unimplemented_method',
                        'line': i,
                        'content': line.strip(),
                        'method_name': method_name,
                        'context': self._get_context(lines, i, 3)
                    })

        return unimplemented_methods

    def _find_placeholder_variables(self, content: str, filepath: str) -> List[Dict[str, Any]]:
        """Find variables that might be placeholders.
        
        Args:
            content: File content
            filepath: Path to file
            
        Returns:
            List of placeholder variable issues
        """
        placeholder_variables = []
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if '=' in stripped:
                var_name = stripped.split('=')[0].strip()
                value = stripped.split('=', 1)[1].strip()
                
                # Check for placeholder values
                if any(placeholder in value.lower() for placeholder in [
                    'placeholder', 'todo', 'fixme', 'implement', 'stub', 'temp', 'dummy'
                ]):
                    placeholder_variables.append({
                        'type': 'placeholder_variable',
                        'line': i,
                        'content': line.strip(),
                        'variable_name': var_name,
                        'value': value,
                        'context': self._get_context(lines, i, 2)
                    })

        return placeholder_variables

    def _find_incomplete_implementations(self, content: str, filepath: str) -> List[Dict[str, Any]]:
        """Find functions that appear to have incomplete implementations.
        
        Args:
            content: File content
            filepath: Path to file
            
        Returns:
            List of incomplete implementation issues
        """
        incomplete_implementations = []
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            if line.strip().startswith('def ') or line.strip().startswith('async def '):
                body_start = i
                body_end = self._find_function_end(lines, i)

                if body_end > body_start:
                    body_lines = [l.strip() for l in lines[body_start:body_end] if l.strip()]
                    
                    # Check for incomplete patterns
                    incomplete_patterns = [
                        'raise Exception', 'raise RuntimeError', 'raise ValueError',
                        'print(', 'logging.warning', 'logging.error',
                        'assert False', 'return None', 'return False'
                    ]
                    
                    for body_line in body_lines:
                        if any(pattern in body_line for pattern in incomplete_patterns):
                            incomplete_implementations.append({
                                'type': 'incomplete_implementation',
                                'line': i,
                                'content': line.strip(),
                                'incomplete_line': body_line,
                                'context': self._get_context(lines, i, 3)
                            })
                            break

        return incomplete_implementations

    def _find_function_end(self, lines: List[str], start_line: int) -> int:
        """Find the end of a function definition.
        
        Args:
            lines: All lines in the file
            start_line: Line number where function starts
            
        Returns:
            Line number where function ends
        """
        indent_level = None
        for i in range(start_line, len(lines)):
            line = lines[i]
            if i == start_line:
                # Get indentation level of function body
                stripped = line.strip()
                if stripped.endswith(':'):
                    # Find the first indented line
                    for j in range(i + 1, len(lines)):
                        if lines[j].strip():
                            indent_level = len(lines[j]) - len(lines[j].lstrip())
                            break
                    continue

            if indent_level is not None:
                if line.strip() and len(line) - len(line.lstrip()) <= indent_level:
                    return i

        return len(lines)

    def _find_class_end(self, lines: List[str], start_line: int) -> int:
        """Find the end of a class definition.
        
        Args:
            lines: All lines in the file
            start_line: Line number where class starts
            
        Returns:
            Line number where class ends
        """
        return self._find_function_end(lines, start_line)

    def _get_context(self, lines: List[str], line_num: int, context_lines: int = 3) -> List[str]:
        """Get context lines around a specific line.
        
        Args:
            lines: All lines in the file
            line_num: Line number to get context for
            context_lines: Number of lines of context on each side
            
        Returns:
            List of context lines
        """
        start = max(0, line_num - context_lines - 1)
        end = min(len(lines), line_num + context_lines)
        return lines[start:end]

    def analyze_directory(self, directory: str) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Analyze all Python files in a directory recursively.
        
        Args:
            directory: Directory path to analyze
            
        Returns:
            Dictionary mapping file paths to their issues
        """
        results = {}

        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    file_results = self.analyze_file(filepath)
                    if file_results:
                        results[filepath] = file_results

        return results

    def generate_report(self, results: Optional[Dict[str, Dict[str, List[Dict[str, Any]]]]] = None) -> str:
        """Generate a comprehensive report of all findings.
        
        Args:
            results: Optional results to report on (uses self.placeholders if None)
            
        Returns:
            Formatted report string
        """
        if results is None:
            results = self.placeholders

        # Calculate analysis duration
        analysis_end_time = datetime.now(timezone.utc)
        analysis_duration = analysis_end_time - self.analysis_start_time

        report = []
        report.append("=" * 80)
        report.append("ENHANCED PLACEHOLDER FINDER REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Add timestamp information
        report.append("ANALYSIS TIMESTAMP:")
        report.append(f"  Started:  {self.analysis_start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report.append(f"  Completed: {analysis_end_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report.append(f"  Duration:  {analysis_duration}")
        report.append("")

        # Summary statistics
        report.append("SUMMARY STATISTICS:")
        report.append(f"  Files analyzed: {self.stats['files_analyzed']}")
        report.append(f"  Total placeholders found: {self.stats['total_placeholders']}")
        report.append(f"  Pass statements: {self.stats['pass_statements']}")
        report.append(f"  TODO comments: {self.stats['todo_comments']}")
        report.append(f"  NotImplementedError raises: {self.stats['raise_notimplemented']}")
        report.append(f"  Placeholder functions: {self.stats['placeholder_functions']}")
        report.append(f"  Empty classes: {self.stats['empty_classes']}")
        report.append(f"  Stub functions: {self.stats['stub_functions']}")
        report.append(f"  Unimplemented methods: {self.stats['unimplemented_methods']}")
        report.append(f"  Placeholder variables: {self.stats['placeholder_variables']}")
        report.append(f"  Incomplete implementations: {self.stats['incomplete_implementations']}")
        report.append("")

        # Per directory statistics
        dir_stats = defaultdict(lambda: {'files': 0, 'placeholders': 0})
        for filepath, issues in results.items():
            directory = os.path.dirname(filepath)
            dir_stats[directory]['files'] += 1
            dir_stats[directory]['placeholders'] += sum(len(issues[key]) for key in issues)

        report.append("PER DIRECTORY BREAKDOWN:")
        for directory, stats in sorted(dir_stats.items()):
            report.append(f"  {directory}: {stats['files']} files, {stats['placeholders']} placeholders")
        report.append("")

        # Per file breakdown
        report.append("PER FILE BREAKDOWN:")
        for filepath, issues in sorted(results.items()):
            total_issues = sum(len(issues[key]) for key in issues)
            report.append(f"  {filepath}: {total_issues} placeholders")

            # Breakdown by type
            for issue_type, issue_list in issues.items():
                if issue_list:
                    report.append(f"    - {issue_type}: {len(issue_list)}")
        report.append("")

        # Detailed findings
        report.append("DETAILED FINDINGS:")
        for filepath, issues in sorted(results.items()):
            report.append(f"")
            report.append(f"File: {filepath}")
            report.append("-" * len(f"File: {filepath}"))

            for issue_type, issue_list in issues.items():
                if issue_list:
                    report.append(f"")
                    report.append(f"  {issue_type.upper()}:")
                    for issue in issue_list:
                        report.append(f"    Line {issue['line']}: {issue['content']}")
                        if 'context' in issue:
                            report.append("      Context:")
                            for ctx_line in issue['context']:
                                report.append(f"        {ctx_line}")

        return "\n".join(report)

    def export_json(self, results: Optional[Dict[str, Dict[str, List[Dict[str, Any]]]]] = None) -> str:
        """Export results as JSON for programmatic use.
        
        Args:
            results: Optional results to export (uses self.placeholders if None)
            
        Returns:
            JSON string representation
        """
        import json
        
        if results is None:
            results = self.placeholders
            
        # Calculate analysis duration
        analysis_end_time = datetime.now(timezone.utc)
        analysis_duration = analysis_end_time - self.analysis_start_time
            
        export_data = {
            'metadata': {
                'tool': 'Enhanced Placeholder Finder',
                'version': '2.0.0',
                'analysis_start_time': self.analysis_start_time.isoformat(),
                'analysis_end_time': analysis_end_time.isoformat(),
                'analysis_duration_seconds': analysis_duration.total_seconds(),
                'analysis_duration_formatted': str(analysis_duration),
                'timestamp_utc': analysis_end_time.isoformat(),
                'working_directory': str(Path().cwd())
            },
            'summary': self.stats,
            'results': results
        }
        
        return json.dumps(export_data, indent=2, default=str)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Enhanced placeholder finder for Python code')
    parser.add_argument('path', help='Path to analyze (file or directory)')
    parser.add_argument('--exclusions', help='Path to exclusions file')
    parser.add_argument('--output', help='Output file for report')
    parser.add_argument('--json', help='Output file for JSON export')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    finder = PlaceholderFinder(args.exclusions)

    if os.path.isfile(args.path):
        results = {args.path: finder.analyze_file(args.path)}
    else:
        results = finder.analyze_directory(args.path)

    # Generate report
    report = finder.generate_report(results)

    # Auto-append datetime to output filenames if they don't already contain a timestamp
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.output:
        # Check if filename already contains a timestamp pattern
        if not re.search(r'\d{8}_\d{6}', args.output):
            # Split filename and extension
            name, ext = os.path.splitext(args.output)
            output_file = f"{name}_{current_time}{ext}"
        else:
            output_file = args.output
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report written to {output_file}")
    else:
        print(report)

    # Export JSON if requested
    if args.json:
        json_data = finder.export_json(results)
        
        # Check if filename already contains a timestamp pattern
        if not re.search(r'\d{8}_\d{6}', args.json):
            # Split filename and extension
            name, ext = os.path.splitext(args.json)
            json_file = f"{name}_{current_time}{ext}"
        else:
            json_file = args.json
        
        with open(json_file, 'w', encoding='utf-8') as f:
            f.write(json_data)
        print(f"JSON export written to {json_file}")


if __name__ == '__main__':
    main()