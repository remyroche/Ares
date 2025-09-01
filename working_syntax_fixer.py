#!/usr/bin/env python3
"""
Working Syntax Fixer
Fixes common Python syntax errors including indentation issues.
"""

import os
import re
import ast
from typing import List, Dict, Tuple
import argparse


class WorkingSyntaxFixer:
    """Fixes common Python syntax errors."""
    
    def __init__(self):
        self.fixes_applied = 0
        self.files_fixed = 0
        
    def fix_file(self, filepath: str, dry_run: bool = False) -> bool:
        """Fix syntax errors in a single file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            content = self._fix_common_errors(content)
            
            if content != original_content:
                if not dry_run:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"✅ Fixed: {filepath}")
                else:
                    print(f"🔧 Would fix: {filepath}")
                self.fixes_applied += 1
                self.files_fixed += 1
                return True
                
            return False
            
        except Exception as e:
            print(f"❌ Error processing {filepath}: {e}")
            return False
    
    def _fix_common_errors(self, content: str) -> str:
        """Fix common syntax errors in Python code."""
        lines = content.split('\n')
        fixed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Fix 1: Fix unterminated triple-quoted strings
            if '"""' in line and line.count('"""') % 2 == 1:
                # Look for the closing quote in next lines
                j = i + 1
                while j < len(lines) and '"""' not in lines[j]:
                    j += 1
                if j >= len(lines):
                    # Add closing quote
                    fixed_lines.append(line + '"""')
                    i += 1
                    continue
            
            # Fix 2: Fix indentation issues
            stripped = line.strip()
            if stripped and not line.startswith(' ') and not line.startswith('\t'):
                # Check if this should be indented (previous line ends with :)
                if fixed_lines and fixed_lines[-1].strip().endswith(':'):
                    # This line should be indented
                    line = '    ' + line
            
            # Fix 3: Fix missing indented blocks after try/except/if/for/while
            if (stripped.endswith(':') and 
                any(stripped.startswith(keyword) for keyword in ['try', 'except', 'if', 'for', 'while', 'def', 'class'])):
                # Check if next line is not indented
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and not lines[i + 1].startswith(' ') and not lines[i + 1].startswith('\t'):
                        # Add pass statement
                        fixed_lines.append(line)
                        fixed_lines.append('    pass')
                        i += 1
                        continue
            
            # Fix 4: Fix inconsistent indentation
            if line.startswith(' ') or line.startswith('\t'):
                # Normalize indentation to 4 spaces
                indent_level = 0
                for char in line:
                    if char == ' ':
                        indent_level += 1
                    elif char == '\t':
                        indent_level += 4
                    else:
                        break
                
                # Convert to 4-space indentation
                new_indent = '    ' * (indent_level // 4)
                line = new_indent + line.lstrip()
            
            fixed_lines.append(line)
            i += 1
        
        return '\n'.join(fixed_lines)
    
    def fix_directory(self, directory: str, dry_run: bool = False) -> Dict[str, int]:
        """Fix syntax errors in all Python files in a directory."""
        stats = {'files_processed': 0, 'files_fixed': 0, 'total_fixes': 0}
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    stats['files_processed'] += 1
                    
                    if self.fix_file(filepath, dry_run):
                        stats['files_fixed'] += 1
                        stats['total_fixes'] += self.fixes_applied
        
        return stats


def main():
    parser = argparse.ArgumentParser(description='Fix Python syntax errors')
    parser.add_argument('directory', help='Directory to fix')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be fixed without making changes')
    
    args = parser.parse_args()
    
    fixer = WorkingSyntaxFixer()
    stats = fixer.fix_directory(args.directory, args.dry_run)
    
    print(f"\n📊 Summary:")
    print(f"Files processed: {stats['files_processed']}")
    print(f"Files fixed: {stats['files_fixed']}")
    print(f"Total fixes applied: {stats['total_fixes']}")


if __name__ == '__main__':
    main()