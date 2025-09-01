#!/usr/bin/env python3
"""
Targeted String and Parentheses Fixer
Fixes specific issues with unterminated string literals and unmatched parentheses.
"""

import os
import re
from typing import List, Tuple, Dict, Any


class TargetedStringFixer:
    """Fixes unterminated string literals and unmatched parentheses."""
    
    def __init__(self):
        self.fixes_applied = 0
        self.files_processed = 0
        self.files_fixed = 0
        
    def fix_file(self, filepath: str, dry_run: bool = True) -> bool:
        """Fix string and parentheses issues in a single file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            original_content = content
            content = self._fix_unterminated_strings(content)
            content = self._fix_unmatched_parentheses(content)
            content = self._fix_mismatched_brackets(content)
            
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
    
    def _fix_unterminated_strings(self, content: str) -> str:
        """Fix unterminated string literals."""
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Count quotes in the line
            single_quotes = line.count("'")
            double_quotes = line.count('"')
            
            # If odd number of quotes, add a closing quote
            if single_quotes % 2 == 1:
                # Find the last single quote
                last_single = line.rfind("'")
                if last_single != -1:
                    # Add closing quote after the last single quote
                    line = line[:last_single + 1] + "'" + line[last_single + 1:]
                else:
                    line += "'"
            
            if double_quotes % 2 == 1:
                # Find the last double quote
                last_double = line.rfind('"')
                if last_double != -1:
                    # Add closing quote after the last double quote
                    line = line[:last_double + 1] + '"' + line[last_double + 1:]
                else:
                    line += '"'
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _fix_unmatched_parentheses(self, content: str) -> str:
        """Fix unmatched parentheses."""
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Count parentheses
            open_parens = line.count('(')
            close_parens = line.count(')')
            
            # If more opening than closing, add missing closing parentheses
            if open_parens > close_parens:
                missing = open_parens - close_parens
                line += ')' * missing
            
            # If more closing than opening, remove extra closing parentheses from the end
            elif close_parens > open_parens:
                extra = close_parens - open_parens
                # Remove extra closing parentheses from the end of the line
                while extra > 0 and line.rstrip().endswith(')'):
                    line = line.rstrip().rstrip(')')
                    extra -= 1
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _fix_mismatched_brackets(self, content: str) -> str:
        """Fix mismatched brackets and braces."""
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Count brackets
            open_brackets = line.count('[')
            close_brackets = line.count(']')
            
            # Fix brackets
            if open_brackets > close_brackets:
                missing = open_brackets - close_brackets
                line += ']' * missing
            elif close_brackets > open_brackets:
                extra = close_brackets - open_brackets
                while extra > 0 and line.rstrip().endswith(']'):
                    line = line.rstrip().rstrip(']')
                    extra -= 1
            
            # Count braces
            open_braces = line.count('{')
            close_braces = line.count('}')
            
            # Fix braces
            if open_braces > close_braces:
                missing = open_braces - close_braces
                line += '}' * missing
            elif close_braces > open_braces:
                extra = close_braces - open_braces
                while extra > 0 and line.rstrip().endswith('}'):
                    line = line.rstrip().rstrip('}')
                    extra -= 1
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def fix_directory(self, directory: str, dry_run: bool = True) -> Dict[str, Any]:
        """Fix string and parentheses issues in all Python files in a directory."""
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
    """Main function to run the targeted string fixer."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix unterminated strings and unmatched parentheses')
    parser.add_argument('directory', help='Directory to process')
    parser.add_argument('--no-dry-run', action='store_true', help='Actually apply fixes')
    parser.add_argument('--output', help='Output report to file')
    
    args = parser.parse_args()
    
    fixer = TargetedStringFixer()
    results = fixer.fix_directory(args.directory, dry_run=not args.no_dry_run)
    
    # Generate report
    report = f"""
Targeted String and Parentheses Fixer Report
==========================================

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