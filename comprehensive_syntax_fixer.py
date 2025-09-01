#!/usr/bin/env python3
"""
Comprehensive Syntax Fixer
Fixes common syntax errors in Python files including:
- Indentation issues
- Missing try/except blocks
- Missing indented blocks after control statements
- Unmatched parentheses
- Invalid decimal literals
"""

import os
import re
import ast
from pathlib import Path
from typing import List, Tuple, Dict, Any


class ComprehensiveSyntaxFixer:
    """Fixes common syntax errors in Python files."""
    
    def __init__(self):
        self.fixes_applied = 0
        self.files_processed = 0
        self.files_fixed = 0
        
    def fix_file(self, filepath: str, dry_run: bool = True) -> bool:
        """Fix syntax errors in a single file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            original_content = content
            content = self._fix_indentation_issues(content)
            content = self._fix_missing_try_except_blocks(content)
            content = self._fix_missing_indented_blocks(content)
            content = self._fix_unmatched_parentheses(content)
            content = self._fix_invalid_decimal_literals(content)
            content = self._fix_parameter_order_issues(content)
            content = self._fix_unterminated_strings(content)
            
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
    
    def _fix_indentation_issues(self, content: str) -> str:
        """Fix common indentation issues."""
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            # Fix mixed tabs and spaces
            if '\t' in line and '    ' in line:
                # Replace tabs with 4 spaces
                line = line.replace('\t', '    ')
            
            # Fix inconsistent indentation
            stripped = line.lstrip()
            if stripped and not line.startswith('#'):
                # Count leading spaces
                leading_spaces = len(line) - len(stripped)
                # Round to nearest 4-space increment
                if leading_spaces % 4 != 0:
                    new_spaces = (leading_spaces // 4) * 4
                    line = ' ' * new_spaces + stripped
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _fix_missing_try_except_blocks(self, content: str) -> str:
        """Fix missing except or finally blocks after try statements."""
        lines = content.split('\n')
        fixed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check for try statement without except/finally
            if line.strip().startswith('try:') or line.strip().startswith('try :'):
                # Look ahead to see if there's an except or finally
                has_except_or_finally = False
                for j in range(i + 1, min(i + 10, len(lines))):
                    next_line = lines[j].strip()
                    if next_line.startswith('except') or next_line.startswith('finally'):
                        has_except_or_finally = True
                        break
                    elif next_line and not next_line.startswith('#'):
                        # Found non-empty, non-comment line that's not except/finally
                        break
                
                if not has_except_or_finally:
                    # Add a basic except block
                    indent = len(line) - len(line.lstrip())
                    fixed_lines.append(line)
                    fixed_lines.append(' ' * (indent + 4) + 'except Exception as e:')
                    fixed_lines.append(' ' * (indent + 8) + 'pass')
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
            
            i += 1
        
        return '\n'.join(fixed_lines)
    
    def _fix_missing_indented_blocks(self, content: str) -> str:
        """Fix missing indented blocks after control statements."""
        lines = content.split('\n')
        fixed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check for control statements that need indented blocks
            control_keywords = ['if', 'for', 'while', 'def', 'class', 'try', 'except', 'finally', 'else', 'elif']
            line_stripped = line.strip()
            
            needs_block = False
            for keyword in control_keywords:
                if line_stripped.startswith(f'{keyword} '):
                    needs_block = True
                    break
            
            if needs_block and line_stripped.endswith(':'):
                # Check if next line is properly indented
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    next_stripped = next_line.strip()
                    
                    if next_stripped and not next_stripped.startswith('#'):
                        # Check if next line is not indented
                        current_indent = len(line) - len(line.lstrip())
                        next_indent = len(next_line) - len(next_line.lstrip())
                        
                        if next_indent <= current_indent:
                            # Add a pass statement
                            indent = len(line) - len(line.lstrip())
                            fixed_lines.append(line)
                            fixed_lines.append(' ' * (indent + 4) + 'pass')
                            i += 1
                            continue
            
            fixed_lines.append(line)
            i += 1
        
        return '\n'.join(fixed_lines)
    
    def _fix_unmatched_parentheses(self, content: str) -> str:
        """Fix unmatched parentheses."""
        # Simple fix for common cases
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Count parentheses
            open_parens = line.count('(') + line.count('[') + line.count('{')
            close_parens = line.count(')') + line.count(']') + line.count('}')
            
            if open_parens > close_parens:
                # Add missing closing parentheses
                missing = open_parens - close_parens
                line += ')' * missing
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _fix_invalid_decimal_literals(self, content: str) -> str:
        """Fix invalid decimal literals like 1.2.3."""
        # Replace invalid decimal literals with valid ones
        # Pattern: number.number.number (invalid)
        pattern = r'(\d+\.\d+)\.(\d+)'
        
        def replace_invalid_decimal(match):
            first_part = match.group(1)
            second_part = match.group(2)
            # Convert to a valid float representation
            return f"{first_part}_{second_part}"
        
        return re.sub(pattern, replace_invalid_decimal, content)
    
    def _fix_parameter_order_issues(self, content: str) -> str:
        """Fix parameter order issues (parameter without default follows parameter with default)."""
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Look for function definitions with parameters
            if 'def ' in line and '(' in line and ')' in line:
                # Extract parameters
                start = line.find('(')
                end = line.find(')')
                if start != -1 and end != -1:
                    params_str = line[start+1:end]
                    params = [p.strip() for p in params_str.split(',') if p.strip()]
                    
                    # Check for parameters with defaults
                    has_defaults = []
                    no_defaults = []
                    
                    for param in params:
                        if '=' in param:
                            has_defaults.append(param)
                        else:
                            no_defaults.append(param)
                    
                    # If we have both types, reorder them
                    if has_defaults and no_defaults:
                        new_params = no_defaults + has_defaults
                        new_params_str = ', '.join(new_params)
                        line = line[:start+1] + new_params_str + line[end:]
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _fix_unterminated_strings(self, content: str) -> str:
        """Fix unterminated string literals."""
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Count quotes
            single_quotes = line.count("'")
            double_quotes = line.count('"')
            
            # If odd number of quotes, add a closing quote
            if single_quotes % 2 == 1:
                line += "'"
            if double_quotes % 2 == 1:
                line += '"'
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def fix_directory(self, directory: str, dry_run: bool = True) -> Dict[str, Any]:
        """Fix syntax errors in all Python files in a directory."""
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
    """Main function to run the syntax fixer."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix common syntax errors in Python files')
    parser.add_argument('directory', help='Directory to process')
    parser.add_argument('--no-dry-run', action='store_true', help='Actually apply fixes')
    parser.add_argument('--output', help='Output report to file')
    
    args = parser.parse_args()
    
    fixer = ComprehensiveSyntaxFixer()
    results = fixer.fix_directory(args.directory, dry_run=not args.no_dry_run)
    
    # Generate report
    report = f"""
Comprehensive Syntax Fixer Report
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