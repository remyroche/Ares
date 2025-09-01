#!/usr/bin/env python3
"""
Comprehensive String Fixer
Fixes complex unterminated string literal errors including triple-quoted strings and multi-line strings.
"""

import os
import re
from typing import List, Tuple, Dict, Any


class ComprehensiveStringFixer:
    """Fixes complex unterminated string literals using comprehensive pattern matching."""

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
            content = self._fix_complex_string_issues(content)
            content = self._fix_triple_quoted_strings(content)
            content = self._fix_multi_line_strings(content)
            content = self._fix_remaining_string_issues(content)

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

    def _fix_complex_string_issues(self, content: str) -> str:
        """Fix complex string issues using regex patterns."""
        # Fix unterminated strings at end of lines
        content = re.sub(r'(["\'])([^"\']*)$', r'\1\2\1', content, flags=re.MULTILINE)
        
        # Fix unterminated f-strings
        content = re.sub(r'f(["\'])([^"\']*)$', r'f\1\2\1', content, flags=re.MULTILINE)
        
        # Fix unterminated r-strings
        content = re.sub(r'r(["\'])([^"\']*)$', r'r\1\2\1', content, flags=re.MULTILINE)
        
        # Fix unterminated b-strings
        content = re.sub(r'b(["\'])([^"\']*)$', r'b\1\2\1', content, flags=re.MULTILINE)
        
        # Fix unterminated u-strings
        content = re.sub(r'u(["\'])([^"\']*)$', r'u\1\2\1', content, flags=re.MULTILINE)
        
        return content

    def _fix_triple_quoted_strings(self, content: str) -> str:
        """Fix triple-quoted string issues."""
        lines = content.split('\n')
        fixed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check for unterminated triple quotes
            triple_single_count = line.count("'''")
            triple_double_count = line.count('"""')
            
            if triple_single_count % 2 == 1:
                # Find the start of the triple quote
                start_pos = line.find("'''")
                if start_pos != -1:
                    # Look for the end in subsequent lines
                    j = i + 1
                    found_end = False
                    while j < len(lines) and not found_end:
                        if "'''" in lines[j]:
                            found_end = True
                        j += 1
                    
                    if not found_end:
                        # Add closing triple quote at the end of the last line
                        if j > i + 1:
                            lines[j - 1] += "'''"
                        else:
                            line += "'''"
            
            if triple_double_count % 2 == 1:
                # Find the start of the triple quote
                start_pos = line.find('"""')
                if start_pos != -1:
                    # Look for the end in subsequent lines
                    j = i + 1
                    found_end = False
                    while j < len(lines) and not found_end:
                        if '"""' in lines[j]:
                            found_end = True
                        j += 1
                    
                    if not found_end:
                        # Add closing triple quote at the end of the last line
                        if j > i + 1:
                            lines[j - 1] += '"""'
                        else:
                            line += '"""'
            
            fixed_lines.append(line)
            i += 1
        
        return '\n'.join(fixed_lines)

    def _fix_multi_line_strings(self, content: str) -> str:
        """Fix multi-line string issues."""
        lines = content.split('\n')
        fixed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check for unterminated strings that span multiple lines
            if self._has_unterminated_string(line):
                # Look for continuation in next lines
                j = i + 1
                continuation_lines = []
                while j < len(lines) and self._is_string_continuation(lines[j]):
                    continuation_lines.append(lines[j])
                    j += 1
                
                if continuation_lines:
                    # Combine the lines and fix
                    combined = line + '\n' + '\n'.join(continuation_lines)
                    fixed_combined = self._fix_single_line_strings(combined)
                    
                    # Split back and replace
                    fixed_parts = fixed_combined.split('\n')
                    fixed_lines.extend(fixed_parts)
                    i = j
                else:
                    # No continuation found, fix the single line
                    line = self._fix_single_line_strings(line)
                    fixed_lines.append(line)
                    i += 1
            else:
                fixed_lines.append(line)
                i += 1
        
        return '\n'.join(fixed_lines)

    def _fix_single_line_strings(self, line: str) -> str:
        """Fix single line string issues."""
        # Count quotes
        single_quotes = line.count("'")
        double_quotes = line.count('"')
        triple_single = line.count("'''")
        triple_double = line.count('"""')
        
        # Adjust for triple quotes
        single_quotes -= triple_single * 3
        double_quotes -= triple_double * 3
        
        # Fix single quotes
        if single_quotes % 2 == 1:
            if line.endswith("'"):
                line += "'"
            else:
                line += "'"
        
        # Fix double quotes
        if double_quotes % 2 == 1:
            if line.endswith('"'):
                line += '"'
            else:
                line += '"'
        
        return line

    def _fix_remaining_string_issues(self, content: str) -> str:
        """Fix any remaining string issues."""
        # Fix strings that end with backslash
        content = re.sub(r'(["\'])([^"\']*)\\(?!["\'])$', r'\1\2\1', content, flags=re.MULTILINE)
        
        # Fix strings with mismatched quotes
        content = re.sub(r'(["\'])([^"\']*)(["\'])(?!["\'])', r'\1\2\1', content, flags=re.MULTILINE)
        
        # Fix strings that start but don't end properly
        content = re.sub(r'(["\'])([^"\']*)$', r'\1\2\1', content, flags=re.MULTILINE)
        
        return content

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
                stripped.startswith('"""') or stripped.startswith("'''") or
                stripped.endswith('\\'))

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
    """Main function to run the comprehensive string fixer."""
    import argparse

    parser = argparse.ArgumentParser(description='Fix complex unterminated strings')
    parser.add_argument('directory', help='Directory to process')
    parser.add_argument('--no-dry-run', action='store_true', help='Actually apply fixes')
    parser.add_argument('--output', help='Output report to file')

    args = parser.parse_args()

    fixer = ComprehensiveStringFixer()
    results = fixer.fix_directory(args.directory, dry_run=not args.no_dry_run)

    # Generate report
    report = f"""
Comprehensive String Fixer Report
================================

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