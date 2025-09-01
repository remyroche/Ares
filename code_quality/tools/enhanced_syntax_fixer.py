#!/usr/bin/env python3
"""
Enhanced Python Syntax Fixer
Automatically fixes complex Python syntax errors including broken imports,
missing indentation, incomplete try/except blocks, and other common issues.
"""

import os
import re
import ast
from typing import List, Dict, Tuple
import argparse


class EnhancedSyntaxFixer:
    """Enhanced syntax fixer for complex Python syntax errors."""
    
    def __init__(self):
        self.fixes_applied = 0
        self.files_fixed = 0
        
    def fix_file(self, filepath: str, dry_run: bool = True) -> bool:
        """Fix syntax errors in a single file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            content = self._fix_complex_errors(content)
            
            if content != original_content:
                if not dry_run:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"✅ Fixed: {filepath}")
                else:
                    print(f"🔧 Would fix: {filepath}")
                self.fixes_applied += 1
                return True
                
            return False
            
        except Exception as e:
            print(f"❌ Error processing {filepath}: {e}")
            return False
    
    def _fix_complex_errors(self, content: str) -> str:
        """Apply complex syntax fixes."""
        lines = content.split('\n')
        fixed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Fix 1: Broken import statements
            if self._is_broken_import(line, lines, i):
                fixed_imports = self._fix_broken_imports(lines, i)
                fixed_lines.extend(fixed_imports)
                i += len(fixed_imports)
                continue
            
            # Fix 2: Missing indentation in class methods
            if self._needs_class_method_indentation(line, fixed_lines):
                line = '    ' + line
            
            # Fix 3: Missing try/except blocks
            if line.strip().startswith('try:') and not self._has_except_block(lines, i):
                fixed_lines.append(line)
                fixed_lines.append('    pass  # TODO: Add proper exception handling')
                fixed_lines.append('except Exception as e:')
                fixed_lines.append('    pass  # TODO: Add proper exception handling')
                i += 1
                continue
            
            # Fix 4: Incomplete method definitions
            if line.strip().startswith('def ') and not line.strip().endswith(':'):
                line = line.rstrip() + ':'
            
            # Fix 5: Missing indentation after colons
            if line.strip().endswith(':') and i + 1 < len(lines):
                next_line = lines[i + 1]
                if next_line.strip() and not next_line.startswith(' ') and not next_line.startswith('\t'):
                    # Add pass statement
                    fixed_lines.append(line)
                    fixed_lines.append('    pass  # TODO: Add implementation')
                    i += 1
                    continue
            
            # Fix 6: Fix self references without proper indentation
            if line.strip().startswith('self.') and not line.startswith('    '):
                line = '    ' + line
            
            # Fix 7: Fix missing return statements
            if line.strip().startswith('return ') and not line.startswith('    '):
                line = '    ' + line
            
            # Fix 8: Fix broken decorators
            if line.strip().startswith('@') and not line.strip().endswith(')'):
                # Look for the closing parenthesis
                j = i + 1
                while j < len(lines) and not lines[j].strip().endswith(')'):
                    j += 1
                if j < len(lines):
                    # Reconstruct the decorator
                    decorator_lines = [lines[k] for k in range(i, j + 1)]
                    fixed_decorator = self._fix_decorator(decorator_lines)
                    fixed_lines.extend(fixed_decorator)
                    i = j + 1
                    continue
            
            fixed_lines.append(line)
            i += 1
        
        return '\n'.join(fixed_lines)
    
    def _is_broken_import(self, line: str, lines: List[str], index: int) -> bool:
        """Check if this line is part of a broken import statement."""
        if 'from' in line and 'import' in line:
            # Check if the import is incomplete
            if not line.strip().endswith(')'):
                return True
        return False
    
    def _fix_broken_imports(self, lines: List[str], start_index: int) -> List[str]:
        """Fix broken import statements."""
        fixed_imports = []
        i = start_index
        
        # Find the complete import block
        while i < len(lines):
            line = lines[i]
            if 'from' in line and 'import' in line:
                # Start of import
                if '(' in line:
                    # Multi-line import
                    fixed_imports.append(line)
                    i += 1
                    while i < len(lines) and not lines[i].strip().endswith(')'):
                        fixed_imports.append(lines[i])
                        i += 1
                    if i < len(lines):
                        fixed_imports.append(lines[i])
                else:
                    # Single line import
                    fixed_imports.append(line)
                break
            i += 1
        
        return fixed_imports
    
    def _needs_class_method_indentation(self, line: str, fixed_lines: List[str]) -> bool:
        """Check if a line needs class method indentation."""
        if line.strip().startswith('def ') and fixed_lines:
            # Check if we're inside a class
            for prev_line in reversed(fixed_lines):
                if prev_line.strip().startswith('class '):
                    return True
                elif prev_line.strip() and not prev_line.startswith('    '):
                    break
        return False
    
    def _has_except_block(self, lines: List[str], try_index: int) -> bool:
        """Check if a try block has a corresponding except block."""
        for i in range(try_index + 1, min(try_index + 20, len(lines))):
            if lines[i].strip().startswith('except'):
                return True
            elif lines[i].strip() and not lines[i].startswith(' '):
                break
        return False
    
    def _fix_decorator(self, decorator_lines: List[str]) -> List[str]:
        """Fix broken decorator syntax."""
        # Simple fix: join lines and ensure proper formatting
        decorator_text = ' '.join([line.strip() for line in decorator_lines])
        return [decorator_text]
    
    def fix_directory(self, directory: str, dry_run: bool = True) -> Dict[str, int]:
        """Fix syntax errors in all Python files in a directory."""
        results = {'files_processed': 0, 'files_fixed': 0, 'total_fixes': 0}
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    results['files_processed'] += 1
                    
                    if self.fix_file(filepath, dry_run):
                        results['files_fixed'] += 1
                        results['total_fixes'] += self.fixes_applied
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Fix complex Python syntax errors')
    parser.add_argument('directory', help='Directory to fix')
    parser.add_argument('--no-dry-run', action='store_true', help='Actually apply fixes')
    parser.add_argument('--output', help='Output report to file')
    
    args = parser.parse_args()
    
    fixer = EnhancedSyntaxFixer()
    results = fixer.fix_directory(args.directory, dry_run=not args.no_dry_run)
    
    report = f"""
Enhanced Syntax Fix Report
==========================
Files processed: {results['files_processed']}
Files fixed: {results['files_fixed']}
Total fixes applied: {results['total_fixes']}
"""
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == '__main__':
    main()