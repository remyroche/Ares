#!/usr/bin/env python3
"""
Placeholder Finder
Finds placeholders and functions that need to be implemented in Python code.
"""

import os
import argparse
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict


class PlaceholderFinder:
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

    def analyze_file(self, filepath: str) -> Dict[str, List[Dict]]:
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

    def _find_pass_statements(self, content: str, filepath: str) -> List[Dict]:
        """Find pass statements that might be placeholders."""
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

                # Only flag if there's a TODO nearby or it's in a try/except block
                if has_todo or in_try_except:
                    pass_statements.append({
                        'type': 'pass_statement',
                        'line': i,
                        'content': line,
                        'context': self._get_context(lines, i)
                    })

        return pass_statements

    def _find_todo_comments(self, content: str, filepath: str) -> List[Dict]:
        """Find TODO and FIXME comments."""
        todo_comments = []
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            if 'TODO' in line or 'FIXME' in line:
                todo_comments.append({
                    'type': 'todo_comment',
                    'line': i,
                    'content': line,
                    'context': self._get_context(lines, i)
                })

        return todo_comments

    def _find_raise_notimplemented(self, content: str, filepath: str) -> List[Dict]:
        """Find raise NotImplementedError statements."""
        raise_notimplemented = []
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            if 'raise NotImplementedError' in line:
                raise_notimplemented.append({
                    'type': 'raise_notimplemented',
                    'line': i,
                    'content': line,
                    'context': self._get_context(lines, i)
                })

        return raise_notimplemented

    def _find_placeholder_functions(self, content: str, filepath: str) -> List[Dict]:
        """Find functions that are likely placeholders."""
        placeholder_functions = []
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith('def ') and stripped.endswith(':'):
                # Check if function body is just pass or raise NotImplementedError
                func_start = i
                func_end = self._find_function_end(lines, i)
                
                func_body = lines[func_start:func_end]
                if self._is_placeholder_function(func_body):
                    placeholder_functions.append({
                        'type': 'placeholder_function',
                        'line': i,
                        'content': line,
                        'context': self._get_context(lines, i)
                    })

        return placeholder_functions

    def _find_function_end(self, lines: List[str], start_line: int) -> int:
        """Find the end of a function definition."""
        indent_level = len(lines[start_line - 1]) - len(lines[start_line - 1].lstrip())
        
        for i in range(start_line, len(lines)):
            line = lines[i]
            if line.strip() == '':
                continue
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent_level and line.strip():
                return i
        
        return len(lines)

    def _is_placeholder_function(self, func_body: List[str]) -> bool:
        """Check if function body is just a placeholder."""
        non_empty_lines = [line.strip() for line in func_body if line.strip()]
        
        if len(non_empty_lines) == 0:
            return True
        
        if len(non_empty_lines) == 1 and non_empty_lines[0] == 'pass':
            return True
        
        if len(non_empty_lines) == 1 and 'raise NotImplementedError' in non_empty_lines[0]:
            return True
        
        return False

    def _get_context(self, lines: List[str], line_num: int, context_lines: int = 3) -> List[str]:
        """Get context around a specific line."""
        start = max(0, line_num - context_lines - 1)
        end = min(len(lines), line_num + context_lines)
        return lines[start:end]

    def analyze_directory(self, directory: str) -> Dict[str, Dict[str, List[Dict]]]:
        """Analyze all Python files in a directory."""
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    self.analyze_file(filepath)
        
        return dict(self.placeholders)

    def generate_report(self) -> str:
        """Generate a comprehensive report of findings."""
        report = []
        report.append("=" * 80)
        report.append("PLACEHOLDER FINDER REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary statistics
        report.append("SUMMARY STATISTICS:")
        report.append(f"Files analyzed: {self.stats['files_analyzed']}")
        report.append(f"Total placeholders found: {self.stats['total_placeholders']}")
        report.append(f"Pass statements: {self.stats['pass_statements']}")
        report.append(f"TODO comments: {self.stats['todo_comments']}")
        report.append(f"NotImplementedError raises: {self.stats['raise_notimplemented']}")
        report.append(f"Placeholder functions: {self.stats['placeholder_functions']}")
        report.append("")
        
        # Detailed findings
        for filepath, issues in self.placeholders.items():
            report.append(f"FILE: {filepath}")
            report.append("-" * 60)
            
            for issue_type, items in issues.items():
                if items:
                    report.append(f"\n{issue_type.upper()}:")
                    for item in items:
                        report.append(f"  Line {item['line']}: {item['content'].strip()}")
                        if 'context' in item:
                            report.append("  Context:")
                            for ctx_line in item['context']:
                                report.append(f"    {ctx_line}")
                        report.append("")
            
            report.append("")
        
        return "\n".join(report)


def main():
    """Main function to run the placeholder finder."""
    parser = argparse.ArgumentParser(description='Find placeholders in Python code')
    parser.add_argument('path', help='File or directory to analyze')
    parser.add_argument('--exclusions', '-e', help='File containing exclusion patterns')
    parser.add_argument('--output', '-o', help='Output file for report')
    
    args = parser.parse_args()
    
    finder = PlaceholderFinder(args.exclusions)
    
    if os.path.isfile(args.path):
        finder.analyze_file(args.path)
    else:
        finder.analyze_directory(args.path)
    
    report = finder.generate_report()
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
    else:
        print(report)


if __name__ == '__main__':
    main()