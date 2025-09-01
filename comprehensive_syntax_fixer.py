#!/usr/bin/env python3
"""
Comprehensive Syntax Fixer
Fixes common syntax errors in Python files.
"""

import os
import re
import ast
from typing import List, Dict, Tuple


class ComprehensiveSyntaxFixer:
    """Fixes common syntax errors in Python files."""

    def __init__(self):
        self.fixed_files = []
        self.error_files = []

    def fix_file(self, filepath: str) -> bool:
        """Fix syntax errors in a single file."""
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            content = self._fix_indentation_errors(content)
            content = self._fix_missing_blocks(content)
            content = self._fix_unmatched_parentheses(content)
            content = self._fix_invalid_decimal_literals(content)
            content = self._fix_parameter_order(content)
            content = self._fix_invalid_syntax(content)

            # Test if the file can be parsed
            try:
                ast.parse(content)
                if content != original_content:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    self.fixed_files.append(filepath)
                    print(f"Fixed: {filepath}")
                    return True
                return False
            except SyntaxError as e:
                print(f"Still has syntax errors after fixing: {filepath} - {e}")
                self.error_files.append(filepath)
                return False

        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            self.error_files.append(filepath)
            return False

    def _fix_indentation_errors(self, content: str) -> str:
        """Fix indentation errors."""
        lines = content.split('\n')
        fixed_lines = []

        for i, line in enumerate(lines):
            # Fix mixed tabs and spaces
            if '\t' in line and '    ' in line:
                line = line.replace('\t', '    ')

            # Fix inconsistent indentation
            stripped = line.lstrip()
            if stripped and not line.startswith(' '):
                # Count leading spaces/tabs
                leading = len(line) - len(stripped)
                if leading % 4 != 0:
                    # Round to nearest 4-space increment
                    leading = (leading // 4) * 4
                    line = ' ' * leading + stripped

            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def _fix_missing_blocks(self, content: str) -> str:
        """Fix missing indented blocks after control structures."""
        lines = content.split('\n')
        fixed_lines = []

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Check for control structures that need indented blocks
            if (stripped.startswith(('if ', 'for ', 'while ', 'try:', 'def ', 'class ')) and
                not stripped.endswith(':')):
                # Add missing colon
                if not stripped.endswith(':'):
                    line = line + ':'

            # Check for missing indented block after control structure
            if (stripped.endswith(':') and
                i + 1 < len(lines) and
                lines[i + 1].strip() and
                not lines[i + 1].startswith(' ') and
                not lines[i + 1].strip().startswith(('#', '"""', "'''"))):
                # Add pass statement
                fixed_lines.append(line)
                fixed_lines.append('    pass')
                i += 1
                continue

            fixed_lines.append(line)
            i += 1

        return '\n'.join(fixed_lines)

    def _fix_unmatched_parentheses(self, content: str) -> str:
        """Fix unmatched parentheses."""
        # Count parentheses
        open_parens = content.count('(')
        close_parens = content.count(')')

        if open_parens > close_parens:
            # Add missing closing parentheses
            content += ')' * (open_parens - close_parens)
        elif close_parens > open_parens:
            # Remove extra closing parentheses from the end
            while content.endswith(')') and close_parens > open_parens:
                content = content[:-1]
                close_parens -= 1

        return content

    def _fix_invalid_decimal_literals(self, content: str) -> str:
        """Fix invalid decimal literals."""
        # Fix numbers with leading zeros (e.g., 01.5 -> 1.5)
        content = re.sub(r'\b0+(\d+\.\d+)\b', r'\1', content)
        # Fix numbers with multiple decimal points
        content = re.sub(r'(\d+)\.(\d+)\.(\d+)', r'\1.\2', content)
        return content

    def _fix_parameter_order(self, content: str) -> str:
        """Fix parameter order issues (parameters with defaults must come after those without)."""
        # This is a complex fix that would require parsing the function definitions
        # For now, we'll just add a comment to flag these issues
        lines = content.split('\n')
        fixed_lines = []

        for line in lines:
            if 'def ' in line and '=' in line:
                # Check if there are parameters with defaults before those without
                # This is a simplified check
                if re.search(r'def \w+\s*\([^)]*=\s*[^,)]*[^=]*\w+\s*[^=,)]*[^=]*\w+\s*=', line):
                    # Add a comment about parameter order
                    fixed_lines.append(line + '  # TODO: Fix parameter order')
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def _fix_invalid_syntax(self, content: str) -> str:
        """Fix other invalid syntax issues."""
        # Fix unterminated strings
        content = re.sub(r'(["\'])([^"\']*)$', r'\1\2\1', content)

        # Fix missing colons after function/class definitions
        content = re.sub(r'(def \w+\s*\([^)]*\))\s*$', r'\1:', content, flags=re.MULTILINE)
        content = re.sub(r'(class \w+[^:]*)\s*$', r'\1:', content, flags=re.MULTILINE)

        # Fix missing except/finally blocks
        lines = content.split('\n')
        fixed_lines = []

        i = 0
        while i < len(lines):
            line = lines[i]
            if line.strip().startswith('try:'):
                # Look for the next non-indented line
                j = i + 1
                while j < len(lines) and (not lines[j].strip() or lines[j].startswith(' ')):
                    j += 1

                if j < len(lines) and not lines[j].strip().startswith(('except', 'finally')):
                    # Add missing except block
                    fixed_lines.append(line)
                    i += 1
                    while i < j:
                        fixed_lines.append(lines[i])
                        i += 1
                    fixed_lines.append('    except Exception as e:')
                    fixed_lines.append('        pass')
                    continue

            fixed_lines.append(line)
            i += 1

        return '\n'.join(fixed_lines)

    def fix_directory(self, directory: str) -> Dict[str, List[str]]:
        """Fix syntax errors in all Python files in a directory."""
        results = {
            'fixed': [],
            'errors': []
        }

        for root, dirs, files in os.walk(directory):
            # Skip certain directories
            dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'test_results']]

            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    if self.fix_file(filepath):
                        results['fixed'].append(filepath)
                    else:
                        results['errors'].append(filepath)

        return results


def main():
    """Main function to run the syntax fixer."""
    import argparse

    parser = argparse.ArgumentParser(description='Fix syntax errors in Python files')
    parser.add_argument('directory', help='Directory to fix')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be fixed without making changes')

    args = parser.parse_args()

    fixer = ComprehensiveSyntaxFixer()

    if args.dry_run:
        print("DRY RUN: Would fix syntax errors in the following files:")
        # Just scan for files with syntax errors
        for root, dirs, files in os.walk(args.directory):
            dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'test_results']]
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            ast.parse(f.read())
                    except SyntaxError:
                        print(f"  {filepath}")
    else:
        results = fixer.fix_directory(args.directory)

        print(f"\nFixed {len(results['fixed'])} files:")
        for filepath in results['fixed']:
            print(f"  {filepath}")

        if results['errors']:
            print(f"\n{len(results['errors'])} files still have errors:")
            for filepath in results['errors']:
                print(f"  {filepath}")


if __name__ == '__main__':
    main()