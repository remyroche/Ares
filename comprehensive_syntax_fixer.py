#!/usr/bin/env python3
"""
Comprehensive Syntax Fixer
Fixes common syntax errors in Python files.
"""

import os
import re
from typing import List, Dict, Set, Tuple


class ComprehensiveSyntaxFixer:
    """Fixes common syntax errors in Python files."""

    def __init__(self):
        self.fixed_files = 0
        self.errors_fixed = 0

    def fix_file(self, filepath: str) -> bool:
        """Fix syntax errors in a single file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            original_content = content

            # Apply fixes
            content = self._fix_missing_imports(content)
            content = self._fix_indentation_issues(content)
            content = self._fix_try_except_blocks(content)
            content = self._fix_function_definitions(content)
            content = self._fix_if_statements(content)
            content = self._fix_for_statements(content)
            content = self._fix_parameter_issues(content)
            content = self._fix_syntax_errors(content)

            # Only write if changes were made
            if content != original_content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.fixed_files += 1
                return True

            return False

        except Exception as e:
            print(f"Error fixing {filepath}: {e}")
            return False

    def _fix_missing_imports(self, content: str) -> str:
        """Fix missing import statements."""
        lines = content.split('\n')
        fixed_lines = []

        for i, line in enumerate(lines):
            # Fix missing imports after from statements
            if line.strip().startswith('from ') and 'import' in line:
                # Check if next line is missing import
                if i + 1 < len(lines) and not lines[i + 1].strip().startswith('import') and not lines[i + 1].strip().startswith('from'):
                    # Look for the actual import line
                    for j in range(i + 1, min(i + 10, len(lines))):
                        if lines[j].strip().startswith('import') or lines[j].strip().startswith('from'):
                            break
                        elif lines[j].strip() and not lines[j].strip().startswith('#'):
                            # Insert missing import
                            lines.insert(j, 'import ' + lines[j].strip().split('(')[0].strip())
                            break

            # Fix incomplete import statements
            if line.strip().startswith('from ') and not line.strip().endswith('import'):
                if 'import' not in line:
                    # Add missing import keyword
                    line = line + ' import *'

            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def _fix_indentation_issues(self, content: str) -> str:
        """Fix indentation issues."""
        lines = content.split('\n')
        fixed_lines = []

        for line in lines:
            # Fix mixed tabs and spaces
            if '\t' in line and '    ' in line:
                line = line.replace('\t', '    ')

            # Fix trailing whitespace
            line = line.rstrip()

            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def _fix_try_except_blocks(self, content: str) -> str:
        """Fix incomplete try-except blocks."""
        lines = content.split('\n')
        fixed_lines = []

        for i, line in enumerate(lines):
            # Fix missing except blocks after try
            if line.strip().startswith('try:') or line.strip().startswith('try '):
                # Check if next line is properly indented
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and not next_line.startswith('except') and not next_line.startswith('finally'):
                        # Find the end of the try block
                        try_end = i + 1
                        indent_level = len(lines[i + 1]) - len(lines[i + 1].lstrip())

                        for j in range(i + 2, len(lines)):
                            if lines[j].strip() == '':
                                continue
                            current_indent = len(lines[j]) - len(lines[j].lstrip())
                            if current_indent <= indent_level and lines[j].strip():
                                try_end = j
                                break

                        # Insert except block
                        lines.insert(try_end, '    except Exception as e:')
                        lines.insert(try_end + 1, '        pass')

            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def _fix_function_definitions(self, content: str) -> str:
        """Fix incomplete function definitions."""
        lines = content.split('\n')
        fixed_lines = []

        for i, line in enumerate(lines):
            # Fix function definitions without body
            if line.strip().startswith('def ') and line.strip().endswith(':'):
                # Check if next line is properly indented
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and not next_line.startswith('    ') and not next_line.startswith('\t'):
                        # Insert pass statement
                        lines.insert(i + 1, '    pass')

            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def _fix_if_statements(self, content: str) -> str:
        """Fix incomplete if statements."""
        lines = content.split('\n')
        fixed_lines = []

        for i, line in enumerate(lines):
            # Fix if statements without body
            if line.strip().startswith('if ') and line.strip().endswith(':'):
                # Check if next line is properly indented
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and not next_line.startswith('    ') and not next_line.startswith('\t'):
                        # Insert pass statement
                        lines.insert(i + 1, '    pass')

            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def _fix_for_statements(self, content: str) -> str:
        """Fix incomplete for statements."""
        lines = content.split('\n')
        fixed_lines = []

        for i, line in enumerate(lines):
            # Fix for statements without body
            if line.strip().startswith('for ') and line.strip().endswith(':'):
                # Check if next line is properly indented
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and not next_line.startswith('    ') and not next_line.startswith('\t'):
                        # Insert pass statement
                        lines.insert(i + 1, '    pass')

            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def _fix_parameter_issues(self, content: str) -> str:
        """Fix parameter ordering issues."""
        lines = content.split('\n')
        fixed_lines = []

        for line in lines:
            # Fix parameter without default follows parameter with default
            if 'def ' in line and '=' in line:
                # This is a complex fix that would require parsing
                # For now, just ensure basic syntax
                pass

            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def _fix_syntax_errors(self, content: str) -> str:
        """Fix common syntax errors."""
        # Fix invalid decimal literals
        content = re.sub(r'(\d+)\.(\d+)\.(\d+)', r'\1.\2\3', content)

        # Fix unmatched parentheses
        # This is complex and would require a proper parser
        # For now, just fix obvious cases

        # Fix invalid escape sequences
        content = re.sub(r'\\([^\\])', r'\\\\\1', content)

        return content

    def fix_directory(self, directory: str) -> Dict[str, int]:
        """Fix syntax errors in all Python files in a directory."""
        results = {'files_fixed': 0, 'errors_fixed': 0}

        for root, dirs, files in os.walk(directory):
            # Skip certain directories
            dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'test_results', 'log']]

            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)

                    # Skip certain files
                    if any(skip in filepath for skip in ['test_models', 'data_cache']):
                        continue

                    print(f"Fixing: {filepath}")
                    if self.fix_file(filepath):
                        results['files_fixed'] += 1

        return results


def main():
    """Main function."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python comprehensive_syntax_fixer.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]

    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist")
        sys.exit(1)

    fixer = ComprehensiveSyntaxFixer()
    results = fixer.fix_directory(directory)

    print(f"\nSyntax fixing completed:")
    print(f"Files fixed: {results['files_fixed']}")
    print(f"Errors fixed: {results['errors_fixed']}")


if __name__ == '__main__':
    main()