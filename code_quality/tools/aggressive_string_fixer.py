#!/usr/bin/env python3
"""
Aggressive String Fixer
Fixes persistent unterminated string literal errors using more sophisticated pattern matching.
"""

import os
import re
from typing import List, Tuple, Dict, Any


class AggressiveStringFixer:
    """Fixes unterminated string literals using aggressive pattern matching."""

    def __init__(self):
        self.fixes_applied = 0
        self.files_processed = 0
        self.files_fixed = 0

    def fix_file(self, filepath: str, dry_run: bool = True) -> bool:
        """Fix string issues in a single file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            content = self._fix_unterminated_strings_aggressive(content)
            content = self._fix_malformed_strings(content)
            content = self._fix_string_continuation_issues(content)

            if content != original_content:
                if not dry_run:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"Fixed: {filepath}")
                else:
                    print(f"Would fix: {filepath}")
                self.files_fixed += 1
                return True

            return False

        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return False

    def _fix_unterminated_strings_aggressive(self, content: str) -> str:
        """Fix unterminated strings using aggressive pattern matching."""
        lines = content.split('\n')
        fixed_lines = []

        for i, line in enumerate(lines):
            # Skip comments and empty lines
            stripped = line.strip()
            if stripped.startswith('#') or not stripped:
                fixed_lines.append(line)
                continue

            # Fix common unterminated string patterns
            line = self._fix_single_quotes(line)
            line = self._fix_double_quotes(line)
            line = self._fix_triple_quotes(line)
            line = self._fix_f_strings(line)
            line = self._fix_raw_strings(line)

            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def _fix_single_quotes(self, line: str) -> str:
        """Fix single quote issues."""
        # Count single quotes
        single_quotes = line.count("'")
        
        if single_quotes % 2 == 1:
            # Find the last single quote
            last_quote = line.rfind("'")
            if last_quote != -1:
                # Check if it's part of a valid string
                before_quote = line[:last_quote]
                after_quote = line[last_quote + 1:]
                
                # If there's content after the quote, add a closing quote
                if after_quote.strip():
                    line = line[:last_quote + 1] + "'" + after_quote
                else:
                    # Add closing quote at the end
                    line += "'"
        
        return line

    def _fix_double_quotes(self, line: str) -> str:
        """Fix double quote issues."""
        # Count double quotes
        double_quotes = line.count('"')
        
        if double_quotes % 2 == 1:
            # Find the last double quote
            last_quote = line.rfind('"')
            if last_quote != -1:
                # Check if it's part of a valid string
                before_quote = line[:last_quote]
                after_quote = line[last_quote + 1:]
                
                # If there's content after the quote, add a closing quote
                if after_quote.strip():
                    line = line[:last_quote + 1] + '"' + after_quote
                else:
                    # Add closing quote at the end
                    line += '"'
        
        return line

    def _fix_triple_quotes(self, line: str) -> str:
        """Fix triple quote issues."""
        # Count triple quotes
        triple_single = line.count("'''")
        triple_double = line.count('"""')
        
        # Fix triple single quotes
        if triple_single % 2 == 1:
            if "'''" in line:
                # Find the last occurrence
                last_triple = line.rfind("'''")
                if last_triple != -1:
                    after_triple = line[last_triple + 3:]
                    if after_triple.strip():
                        line = line[:last_triple + 3] + "'''" + after_triple
                    else:
                        line += "'''"
        
        # Fix triple double quotes
        if triple_double % 2 == 1:
            if '"""' in line:
                # Find the last occurrence
                last_triple = line.rfind('"""')
                if last_triple != -1:
                    after_triple = line[last_triple + 3:]
                    if after_triple.strip():
                        line = line[:last_triple + 3] + '"""' + after_triple
                    else:
                        line += '"""'
        
        return line

    def _fix_f_strings(self, line: str) -> str:
        """Fix f-string issues."""
        # Look for unterminated f-strings
        f_string_pattern = r'f["\']([^"\']*)$'
        match = re.search(f_string_pattern, line)
        if match:
            # Add closing quote
            if line.endswith('"'):
                line += '"'
            elif line.endswith("'"):
                line += "'"
        
        return line

    def _fix_raw_strings(self, line: str) -> str:
        """Fix raw string issues."""
        # Look for unterminated r-strings
        r_string_pattern = r'r["\']([^"\']*)$'
        match = re.search(r_string_pattern, line)
        if match:
            # Add closing quote
            if line.endswith('"'):
                line += '"'
            elif line.endswith("'"):
                line += "'"
        
        return line

    def _fix_malformed_strings(self, content: str) -> str:
        """Fix malformed string patterns."""
        # Fix common malformed patterns
        patterns = [
            # Fix strings that end with backslash
            (r'(["\'])([^"\']*)\\(?!["\'])$', r'\1\2\1'),
            # Fix strings with mismatched quotes
            (r'(["\'])([^"\']*)(["\'])(?!["\'])', r'\1\2\1'),
            # Fix strings that start but don't end properly
            (r'(["\'])([^"\']*)$', r'\1\2\1'),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        
        return content

    def _fix_string_continuation_issues(self, content: str) -> str:
        """Fix string continuation issues."""
        lines = content.split('\n')
        fixed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check for string continuation issues
            if self._has_unterminated_string(line):
                # Look for continuation in next lines
                j = i + 1
                while j < len(lines) and self._is_string_continuation(lines[j]):
                    line += '\n' + lines[j]
                    j += 1
                
                # Fix the combined line
                line = self._fix_single_quotes(line)
                line = self._fix_double_quotes(line)
                line = self._fix_triple_quotes(line)
                
                # Split back into lines
                fixed_parts = line.split('\n')
                fixed_lines.extend(fixed_parts)
                i = j
            else:
                fixed_lines.append(line)
                i += 1
        
        return '\n'.join(fixed_lines)

    def _has_unterminated_string(self, line: str) -> bool:
        """Check if line has an unterminated string."""
        single_quotes = line.count("'")
        double_quotes = line.count('"')
        triple_single = line.count("'''")
        triple_double = line.count('"""')
        
        # Adjust for triple quotes
        single_quotes -= triple_single * 3
        double_quotes -= triple_double * 3
        
        return (single_quotes % 2 == 1) or (double_quotes % 2 == 1)

    def _is_string_continuation(self, line: str) -> bool:
        """Check if line is a string continuation."""
        stripped = line.strip()
        return (stripped.startswith('"') or stripped.startswith("'") or 
                stripped.startswith('"""') or stripped.startswith("'''"))

    def fix_directory(self, directory: str, dry_run: bool = True) -> Dict[str, Any]:
        """Fix string issues in all Python files in a directory."""
        results = {
            'files_processed': 0,
            'files_fixed': 0,
            'errors': []
        }

        for root, dirs, files in os.walk(directory):
            # Skip certain directories
            dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'test_results', 'log']]

            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    results['files_processed'] += 1

                    try:
                        if self.fix_file(filepath, dry_run):
                            results['files_fixed'] += 1
                    except Exception as e:
                        results['errors'].append(f"{filepath}: {e}")

        return results


def main():
    """Main function to run the aggressive string fixer."""
    import argparse

    parser = argparse.ArgumentParser(description='Fix unterminated strings aggressively')
    parser.add_argument('directory', help='Directory to process')
    parser.add_argument('--no-dry-run', action='store_true', help='Actually apply fixes')
    parser.add_argument('--output', help='Output report to file')

    args = parser.parse_args()

    fixer = AggressiveStringFixer()
    results = fixer.fix_directory(args.directory, dry_run=not args.no_dry_run)

    # Generate report
    report = f"""
Aggressive String Fixer Report
=============================

Files processed: {results['files_processed']}
Files fixed: {results['files_fixed']}
Errors encountered: {len(results['errors'])}

"""

    if results['errors']:
        report += "Errors:\n"
        for error in results['errors']:
            report += f"  - {error}\n"

    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == '__main__':
    main()