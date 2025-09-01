#!/usr/bin/env python3
"""
Comprehensive Final Fixer
Fixes all remaining syntax issues in one pass.
"""

import os
import re
from typing import List, Dict, Tuple
import argparse


class ComprehensiveFinalFixer:
    """Fixes all remaining syntax issues in one comprehensive pass."""
    
    def __init__(self):
        self.fixes_applied = 0
        self.files_fixed = 0
        
    def fix_file(self, filepath: str, dry_run: bool = False) -> bool:
        """Fix all syntax errors in a single file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            content = self._fix_all_issues(content)
            
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
    
    def _fix_all_issues(self, content: str) -> str:
        """Fix all syntax issues in content."""
        lines = content.split('\n')
        fixed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Fix 1: Fix unterminated triple-quoted strings
            if '"""' in line:
                quote_count = line.count('"""')
                if quote_count % 2 == 1:
                    # Look for closing quote in next lines
                    j = i + 1
                    found_closing = False
                    while j < len(lines):
                        if '"""' in lines[j]:
                            found_closing = True
                            break
                        j += 1
                    
                    if not found_closing:
                        # Add closing quote at end of current line
                        line = line + '"""'
            
            # Fix 2: Fix empty triple-quoted strings
            if stripped == '""""""':
                line = '"""'
            
            # Fix 3: Fix unmatched parentheses
            if stripped.count('(') > stripped.count(')'):
                # Look ahead to find the matching closing parenthesis
                j = i + 1
                while j < len(lines):
                    if ')' in lines[j]:
                        # Found the closing parenthesis, no fix needed
                        break
                    j += 1
                else:
                    # No closing parenthesis found, add one
                    line = line + ')'
            
            # Fix 4: Fix missing function/class/control structure bodies
            if (stripped.endswith(':') and 
                any(stripped.startswith(keyword) for keyword in ['def', 'class', 'try', 'except', 'if', 'for', 'while', 'else', 'elif'])):
                # Check if next line is missing or not indented
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if not next_line or (next_line and not lines[i + 1].startswith(' ') and not lines[i + 1].startswith('\t')):
                        # Add pass statement
                        fixed_lines.append(line)
                        fixed_lines.append('    pass')
                        i += 1
                        continue
            
            # Fix 5: Fix indentation issues
            if stripped:
                # Remove leading whitespace and re-indent properly
                if line.startswith(' ') or line.startswith('\t'):
                    # Count current indentation
                    indent_count = 0
                    for char in line:
                        if char == ' ':
                            indent_count += 1
                        elif char == '\t':
                            indent_count += 4
                        else:
                            break
                    
                    # Normalize to 4-space indentation
                    indent_level = max(0, indent_count // 4)
                    line = '    ' * indent_level + stripped
                else:
                    # Check if this line should be indented
                    should_indent = False
                    indent_level = 0
                    
                    # Look back to find the proper indentation level
                    for prev_line in reversed(fixed_lines):
                        prev_stripped = prev_line.strip()
                        if prev_stripped.endswith(':'):
                            if any(prev_stripped.startswith(keyword) for keyword in 
                                   ['try', 'except', 'if', 'for', 'while', 'def', 'class', 'else', 'elif']):
                                should_indent = True
                                indent_level = 1
                                break
                            elif prev_stripped.startswith('@'):
                                # Decorator - next line should be indented
                                should_indent = True
                                indent_level = 1
                                break
                        elif prev_stripped and not prev_stripped.startswith('#'):
                            # Non-empty, non-comment line
                            break
                    
                    # Apply proper indentation
                    if should_indent:
                        line = '    ' * indent_level + stripped
            
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
    parser = argparse.ArgumentParser(description='Comprehensively fix all Python syntax errors')
    parser.add_argument('directory', help='Directory to fix')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be fixed without making changes')
    
    args = parser.parse_args()
    
    fixer = ComprehensiveFinalFixer()
    stats = fixer.fix_directory(args.directory, args.dry_run)
    
    print(f"\n📊 Summary:")
    print(f"Files processed: {stats['files_processed']}")
    print(f"Files fixed: {stats['files_fixed']}")
    print(f"Total fixes applied: {stats['total_fixes']}")


if __name__ == '__main__':
    main()