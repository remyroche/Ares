#!/usr/bin/env python3
"""
Enhanced Python Syntax Fixer
Automatically fixes common Python syntax errors found in the codebase.
"""

import os
import re
import ast
from typing import List, Dict, Tuple
import argparse


class EnhancedSyntaxFixer:
    """Enhanced syntax fixer for common Python syntax errors."""
    
    def __init__(self):
        self.fixes_applied = 0
        self.files_fixed = 0
        
    def fix_file(self, filepath: str, dry_run: bool = True) -> bool:
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
                return True
                
            return False
            
        except Exception as e:
            print(f"❌ Error processing {filepath}: {e}")
            return False
    
    def _fix_common_errors(self, content: str) -> str:
        """Apply common syntax fixes."""
        lines = content.split('\n')
        fixed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Fix 1: Missing indented block after function definition
            if re.match(r'^\s*def\s+\w+\([^)]*\)\s*:\s*$', line.strip()):
                # Function definition with no body
                fixed_lines.append(line)
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if next_line.strip() and not next_line.startswith(' ') and not next_line.startswith('\t'):
                        # Missing indented block
                        fixed_lines.append('    pass  # TODO: Add implementation')
                        i += 1
                        continue
            
            # Fix 2: Missing indented block after if/for/while/try
            if line.strip().endswith(':') and i + 1 < len(lines):
                next_line = lines[i + 1]
                if next_line.strip() and not next_line.startswith(' ') and not next_line.startswith('\t'):
                    # Missing indented block
                    fixed_lines.append(line)
                    fixed_lines.append('    pass  # TODO: Add implementation')
                    i += 1
                    continue
            
            # Fix 3: Fix indentation issues
            if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                # Check if this should be indented (previous line ends with :)
                if i > 0 and fixed_lines and fixed_lines[-1].strip().endswith(':'):
                    # This line should be indented
                    line = '    ' + line
            
            # Fix 4: Fix unmatched parentheses
            if line.strip().endswith('(') and i + 1 < len(lines):
                next_line = lines[i + 1]
                if next_line.strip().startswith(')') and not next_line.strip().startswith('))'):
                    # Missing closing parenthesis
                    fixed_lines.append(line)
                    fixed_lines.append('    )  # TODO: Add proper closing')
                    i += 1
                    continue
            
            # Fix 5: Fix invalid decimal literals
            if re.search(r'\b\d+\.\d+\.\d+\b', line):
                # Fix invalid decimal like 1.2.3
                line = re.sub(r'\b(\d+)\.(\d+)\.(\d+)\b', r'\1_\2_\3', line)
            
            # Fix 6: Fix parameter order issues
            if 'def ' in line and '=' in line:
                # Check for parameters with defaults before parameters without defaults
                if re.search(r'def \w+\([^)]*[^=,]+=[^,)]*[^=,]+[^)]*\)', line):
                    # This is a complex fix that would require parsing - skip for now
                    pass
            
            # Fix 7: Fix missing except blocks after try
            if line.strip().startswith('try:') and i + 1 < len(lines):
                next_line = lines[i + 1]
                # Check if next line is not indented properly or missing except
                if not next_line.strip().startswith('except') and not next_line.strip().startswith('finally'):
                    # Look ahead to see if there's an except block
                    has_except = False
                    for j in range(i + 1, min(i + 10, len(lines))):
                        if lines[j].strip().startswith('except'):
                            has_except = True
                            break
                        elif lines[j].strip() and not lines[j].startswith(' '):
                            break
                    
                    if not has_except:
                        # Add a basic except block
                        fixed_lines.append(line)
                        fixed_lines.append('    pass  # TODO: Add proper exception handling')
                        fixed_lines.append('except Exception as e:')
                        fixed_lines.append('    pass  # TODO: Add proper exception handling')
                        i += 1
                        continue
            
            # Fix 8: Fix missing async def blocks
            if re.match(r'^\s*async\s+def\s+\w+\([^)]*\)\s*:\s*$', line.strip()):
                # Async function definition with no body
                fixed_lines.append(line)
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if next_line.strip() and not next_line.startswith(' ') and not next_line.startswith('\t'):
                        # Missing indented block
                        fixed_lines.append('    pass  # TODO: Add implementation')
                        i += 1
                        continue
            
            # Fix 9: Fix missing class definitions
            if re.match(r'^\s*class\s+\w+[^:]*:\s*$', line.strip()):
                # Class definition with no body
                fixed_lines.append(line)
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if next_line.strip() and not next_line.startswith(' ') and not next_line.startswith('\t'):
                        # Missing indented block
                        fixed_lines.append('    pass  # TODO: Add implementation')
                        i += 1
                        continue
            
            fixed_lines.append(line)
            i += 1
        
        return '\n'.join(fixed_lines)
    
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
    parser = argparse.ArgumentParser(description='Fix Python syntax errors')
    parser.add_argument('directory', help='Directory to fix')
    parser.add_argument('--no-dry-run', action='store_true', help='Actually apply fixes')
    parser.add_argument('--output', help='Output report to file')
    
    args = parser.parse_args()
    
    fixer = EnhancedSyntaxFixer()
    results = fixer.fix_directory(args.directory, dry_run=not args.no_dry_run)
    
    report = f"""
Enhanced Syntax Fix Report
=========================
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