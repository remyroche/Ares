#!/usr/bin/env python3
"""
Scripts-specific syntax fixer for common indentation and syntax issues
found in the scripts/ directory.
"""

import os
import re
from typing import List, Dict, Tuple
import argparse


class ScriptsSyntaxFixer:
    """Specialized syntax fixer for scripts directory issues."""
    
    def __init__(self):
        self.fixes_applied = 0
        self.files_fixed = 0
        
    def fix_file(self, filepath: str, dry_run: bool = True) -> bool:
        """Fix syntax errors in a single file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            content = self._fix_scripts_errors(content)
            
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
    
    def _fix_scripts_errors(self, content: str) -> str:
        """Apply scripts-specific syntax fixes."""
        lines = content.split('\n')
        fixed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Fix 1: Class method indentation issues
            if self._is_class_method_with_wrong_indentation(line, fixed_lines):
                line = '    ' + line.strip()
            
            # Fix 2: Function parameter trailing comma
            if line.strip().endswith(', )'):
                line = line.replace(', )', ')')
            
            # Fix 3: Missing indentation after class definition
            if line.strip().startswith('def ') and not line.startswith('    ') and self._is_in_class(fixed_lines):
                line = '    ' + line
            
            # Fix 4: Fix method definitions with wrong indentation
            if line.strip().startswith('def ') and line.startswith('        ') and self._is_in_class(fixed_lines):
                line = '    ' + line.strip()
            
            # Fix 5: Fix variable assignments with wrong indentation in methods
            if (line.strip() and not line.startswith(' ') and not line.startswith('\t') and 
                not line.strip().startswith('#') and not line.strip().startswith('"""') and
                not line.strip().startswith("'''") and self._is_in_method(fixed_lines)):
                line = '        ' + line
            
            # Fix 6: Fix return statements with wrong indentation
            if line.strip().startswith('return ') and not line.startswith('        ') and self._is_in_method(fixed_lines):
                line = '        ' + line
            
            # Fix 7: Fix if/for/while statements with wrong indentation in methods
            if (line.strip().startswith(('if ', 'for ', 'while ', 'try:', 'except', 'finally:', 'with ')) and 
                not line.startswith('        ') and self._is_in_method(fixed_lines)):
                line = '        ' + line
            
            fixed_lines.append(line)
            i += 1
        
        return '\n'.join(fixed_lines)
    
    def _is_class_method_with_wrong_indentation(self, line: str, fixed_lines: List[str]) -> bool:
        """Check if this is a class method with wrong indentation."""
        if line.strip().startswith('def ') and line.startswith('        '):
            # Check if we're in a class
            for prev_line in reversed(fixed_lines):
                if prev_line.strip().startswith('class '):
                    return True
                elif prev_line.strip() and not prev_line.startswith('    '):
                    break
        return False
    
    def _is_in_class(self, fixed_lines: List[str]) -> bool:
        """Check if we're currently inside a class definition."""
        in_class = False
        for line in reversed(fixed_lines):
            if line.strip().startswith('class '):
                in_class = True
                break
            elif line.strip() and not line.startswith('    ') and not line.startswith('\t'):
                break
        return in_class
    
    def _is_in_method(self, fixed_lines: List[str]) -> bool:
        """Check if we're currently inside a method definition."""
        in_method = False
        for line in reversed(fixed_lines):
            if line.strip().startswith('def '):
                in_method = True
                break
            elif line.strip() and not line.startswith('    ') and not line.startswith('\t'):
                break
        return in_method
    
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
    parser = argparse.ArgumentParser(description='Fix scripts-specific Python syntax errors')
    parser.add_argument('directory', help='Directory to fix')
    parser.add_argument('--no-dry-run', action='store_true', help='Actually apply fixes')
    parser.add_argument('--output', help='Output report to file')
    
    args = parser.parse_args()
    
    fixer = ScriptsSyntaxFixer()
    results = fixer.fix_directory(args.directory, dry_run=not args.no_dry_run)
    
    report = f"""
Scripts Syntax Fix Report
========================
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