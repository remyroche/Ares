#!/usr/bin/env python3
"""
Simple Placeholder Finder
Finds placeholders and functions that need to be implemented in Python code.
"""

import os
import re
import argparse
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict


class SimplePlaceholderFinder:
    """Finds placeholders and functions that need to be implemented."""

    def __init__(self, exclusions_file: str = None):
        self.exclusions = self._load_exclusions(exclusions_file)
        self.placeholders = defaultdict(list)
        self.stats = {
            'files_analyzed': 0,
            'total_placeholders': 0,
            'pass_statements': 0,
            'todo_comments': 0,
            'raise_notimplemented': 0,
            'placeholder_functions': 0
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
        """Check if file should be excluded from analysis."""
        for pattern in self.exclusions:
            if pattern in filepath or filepath.endswith(pattern.replace('*', '')):
                return True
        return False

    def analyze_file(self, filepath: str) -> Dict[str, List[Tuple[int, str]]]:
        """Analyze a single file for placeholders."""
        if self._should_exclude(filepath):
            return {}

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            issues = {
                'pass_statements': self._find_pass_statements(content, filepath),
                'todo_comments': self._find_todo_comments(content, filepath),
                'raise_notimplemented': self._find_raise_notimplemented(content, filepath),
                'placeholder_functions': self._find_placeholder_functions(content, filepath)
            }

            self.stats['files_analyzed'] += 1
            self.stats['pass_statements'] += len(issues['pass_statements'])
            self.stats['todo_comments'] += len(issues['todo_comments'])
            self.stats['raise_notimplemented'] += len(issues['raise_notimplemented'])
            self.stats['placeholder_functions'] += len(issues['placeholder_functions'])

            total_issues = sum(len(issues[key]) for key in issues)
            self.stats['total_placeholders'] += total_issues

            if total_issues > 0:
                self.placeholders[filepath] = issues

            return issues

        except (UnicodeDecodeError, PermissionError) as e:
            print(f"Error analyzing {filepath}: {e}")
            return {}

    def _find_pass_statements(self, content: str, filepath: str) -> List[Tuple[int, str]]:
        """Find standalone pass statements that might be placeholders."""
        pass_statements = []
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped == 'pass':
                # Check if there's a TODO comment nearby or if it's in a try/except block
                has_todo = False
                in_try_except = False

                # Check nearby lines for TODO comments
                for j in range(max(0, i-3), min(len(lines), i+2)):
                    if 'TODO' in lines[j] or 'FIXME' in lines[j]:
                        has_todo = True
                        break

                # Check if we're in a try/except block
                for j in range(max(0, i-5), min(len(lines), i+5)):
                    if 'try:' in lines[j] or 'except' in lines[j]:
                        in_try_except = True
                        break

                # If it's not in a try/except block or has a TODO, it's likely a placeholder
                if not in_try_except or has_todo:
                    pass_statements.append((i, f"Standalone pass statement (likely placeholder)"))

        return pass_statements

    def _find_todo_comments(self, content: str, filepath: str) -> List[Tuple[int, str]]:
        """Find TODO and FIXME comments."""
        todo_comments = []
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            if 'TODO' in line or 'FIXME' in line:
                todo_comments.append((i, line.strip()))

        return todo_comments

    def _find_raise_notimplemented(self, content: str, filepath: str) -> List[Tuple[int, str]]:
        """Find NotImplementedError raises."""
        not_implemented = []
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            if 'NotImplementedError' in line or 'raise NotImplemented' in line:
                not_implemented.append((i, line.strip()))

        return not_implemented

    def _find_placeholder_functions(self, content: str, filepath: str) -> List[Tuple[int, str]]:
        """Find functions that are likely placeholders."""
        placeholder_functions = []
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            # Look for function definitions that might be placeholders
            if re.match(r'^\s*def\s+\w+\s*\([^)]*\)\s*:\s*$', line):
                # Check if the function body is just pass or raise NotImplementedError
                function_start = i
                function_end = self._find_function_end(lines, i)
                
                function_body = lines[function_start:function_end]
                body_text = '\n'.join(function_body).strip()
                
                if (body_text == 'pass' or 
                    'NotImplementedError' in body_text or
                    'TODO' in body_text or
                    'FIXME' in body_text):
                    placeholder_functions.append((i, f"Placeholder function: {line.strip()}"))

        return placeholder_functions

    def _find_function_end(self, lines: List[str], start_line: int) -> int:
        """Find the end of a function definition."""
        indent_level = None
        for i in range(start_line, len(lines)):
            line = lines[i]
            if i == start_line:
                # Get the indentation level of the function body
                indent_level = len(line) - len(line.lstrip())
                continue
            
            if line.strip() == '':
                continue
                
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent_level:
                return i
                
        return len(lines)

    def analyze_directory(self, directory: str) -> Dict[str, Any]:
        """Analyze all Python files in a directory."""
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    self.analyze_file(filepath)

        return {
            'stats': self.stats,
            'placeholders': dict(self.placeholders)
        }

    def generate_report(self, output_file: str = None):
        """Generate a report of findings."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("PLACEHOLDER FINDER REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Summary statistics
        report_lines.append("SUMMARY STATISTICS:")
        report_lines.append("-" * 40)
        for key, value in self.stats.items():
            report_lines.append(f"{key.replace('_', ' ').title()}: {value}")
        report_lines.append("")
        
        # Detailed findings
        if self.placeholders:
            report_lines.append("DETAILED FINDINGS:")
            report_lines.append("-" * 40)
            
            for filepath, issues in self.placeholders.items():
                report_lines.append(f"\nFile: {filepath}")
                report_lines.append("-" * len(f"File: {filepath}"))
                
                for issue_type, items in issues.items():
                    if items:
                        report_lines.append(f"\n  {issue_type.replace('_', ' ').title()}:")
                        for line_num, description in items:
                            report_lines.append(f"    Line {line_num}: {description}")
        else:
            report_lines.append("No placeholders found!")
        
        report_content = '\n'.join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_content)
            print(f"Report written to: {output_file}")
        else:
            print(report_content)


def main():
    parser = argparse.ArgumentParser(description='Find placeholders in Python code')
    parser.add_argument('directory', help='Directory to analyze')
    parser.add_argument('--exclusions', help='Path to exclusions file')
    parser.add_argument('--output', help='Output file for report')
    
    args = parser.parse_args()
    
    finder = SimplePlaceholderFinder(args.exclusions)
    finder.analyze_directory(args.directory)
    finder.generate_report(args.output)


if __name__ == '__main__':
    main()