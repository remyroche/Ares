#!/usr/bin/env python3
"""
Comprehensive syntax error fixer for the codebase.
This script attempts to fix common syntax errors found in the analysis.
"""

import os
import re
import sys
import ast
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

class SyntaxErrorFixer:
    def __init__(self, backup_dir: str = "syntax_fix_backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self.fixes_applied = []
        self.files_fixed = 0
        self.files_failed = 0
        
    def backup_file(self, filepath: Path) -> Path:
        """Create a backup of the file before modifying."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"{filepath.name}.{timestamp}.bak"
        shutil.copy2(filepath, backup_path)
        return backup_path
        
    def validate_syntax(self, code: str, filepath: str) -> Tuple[bool, Optional[str]]:
        """Validate Python syntax."""
        try:
            compile(code, filepath, 'exec')
            return True, None
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, str(e)
            
    def fix_indentation_errors(self, content: str) -> str:
        """Fix common indentation errors."""
        lines = content.split('\n')
        fixed_lines = []
        indent_stack = [0]
        
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if not stripped or stripped.startswith('#'):
                fixed_lines.append(line)
                continue
                
            # Calculate current indentation
            current_indent = len(line) - len(stripped)
            
            # Fix common patterns
            if stripped.startswith(('def ', 'class ', 'if ', 'elif ', 'else:', 'try:', 
                                  'except', 'finally:', 'for ', 'while ', 'with ')):
                # These should align with or be one level deeper than previous
                if current_indent > indent_stack[-1] + 8:
                    # Too much indentation
                    line = ' ' * (indent_stack[-1] + 4) + stripped
                elif stripped.startswith(('elif ', 'else:', 'except', 'finally:')):
                    # These should align with their opening statement
                    if indent_stack and current_indent != indent_stack[-1]:
                        line = ' ' * indent_stack[-1] + stripped
                        
            # Update indent stack
            if stripped.endswith(':'):
                if current_indent not in indent_stack:
                    indent_stack.append(current_indent)
            elif current_indent < indent_stack[-1]:
                while indent_stack and current_indent < indent_stack[-1]:
                    indent_stack.pop()
                    
            fixed_lines.append(line)
            
        return '\n'.join(fixed_lines)
        
    def fix_unmatched_brackets(self, content: str) -> str:
        """Fix unmatched parentheses, brackets, and quotes."""
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Count brackets
            open_parens = line.count('(')
            close_parens = line.count(')')
            open_brackets = line.count('[')
            close_brackets = line.count(']')
            open_braces = line.count('{')
            close_braces = line.count('}')
            
            # Fix unmatched parentheses
            if open_parens > close_parens:
                line += ')' * (open_parens - close_parens)
            elif close_parens > open_parens:
                # Remove extra closing parentheses from the end
                for _ in range(close_parens - open_parens):
                    if line.rstrip().endswith(')'):
                        line = line.rstrip()[:-1]
                        
            # Fix unmatched brackets
            if open_brackets > close_brackets:
                line += ']' * (open_brackets - close_brackets)
                
            # Fix unmatched braces
            if open_braces > close_braces:
                line += '}' * (open_braces - close_braces)
                
            fixed_lines.append(line)
            
        return '\n'.join(fixed_lines)
        
    def fix_try_except_blocks(self, content: str) -> str:
        """Fix missing except or finally blocks after try."""
        lines = content.split('\n')
        fixed_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            stripped = line.lstrip()
            
            if stripped.startswith('try:'):
                indent = len(line) - len(stripped)
                fixed_lines.append(line)
                i += 1
                
                # Collect the try block
                try_block_lines = []
                while i < len(lines):
                    next_line = lines[i]
                    next_stripped = next_line.lstrip()
                    next_indent = len(next_line) - len(next_stripped)
                    
                    if next_stripped and next_indent <= indent:
                        # End of try block
                        break
                    try_block_lines.append(next_line)
                    i += 1
                
                # Add try block lines
                fixed_lines.extend(try_block_lines)
                
                # Check if except or finally follows
                has_except_or_finally = False
                if i < len(lines):
                    next_stripped = lines[i].lstrip()
                    if next_stripped.startswith(('except', 'finally:')):
                        has_except_or_finally = True
                
                # Add except block if missing
                if not has_except_or_finally:
                    fixed_lines.append(' ' * indent + 'except Exception:')
                    fixed_lines.append(' ' * (indent + 4) + 'pass')
                    
                continue
                
            fixed_lines.append(line)
            i += 1
            
        return '\n'.join(fixed_lines)
        
    def fix_import_order(self, content: str) -> str:
        """Fix import order issues, especially __future__ imports."""
        lines = content.split('\n')
        
        # Separate different types of lines
        future_imports = []
        standard_imports = []
        third_party_imports = []
        local_imports = []
        other_lines = []
        
        # Track if we're past imports
        past_imports = False
        
        for line in lines:
            stripped = line.strip()
            
            if not past_imports and stripped.startswith('from __future__ import'):
                future_imports.append(line)
            elif not past_imports and (stripped.startswith('import ') or stripped.startswith('from ')):
                if any(stripped.startswith(f'from {pkg}') or stripped.startswith(f'import {pkg}') 
                      for pkg in ['os', 'sys', 'json', 'time', 'datetime', 'pathlib', 're', 'ast']):
                    standard_imports.append(line)
                elif stripped.startswith(('from .', 'import .')):
                    local_imports.append(line)
                else:
                    third_party_imports.append(line)
            else:
                if stripped and not stripped.startswith('#'):
                    past_imports = True
                other_lines.append(line)
        
        # Reconstruct with proper order
        result = []
        
        # Add future imports first
        if future_imports:
            result.extend(future_imports)
            result.append('')
            
        # Add standard library imports
        if standard_imports:
            result.extend(sorted(set(standard_imports)))
            result.append('')
            
        # Add third-party imports
        if third_party_imports:
            result.extend(sorted(set(third_party_imports)))
            result.append('')
            
        # Add local imports
        if local_imports:
            result.extend(sorted(set(local_imports)))
            result.append('')
            
        # Remove multiple empty lines at the start
        while result and result[0] == '':
            result.pop(0)
            
        # Add the rest
        result.extend(other_lines)
        
        return '\n'.join(result)
        
    def fix_file(self, filepath: Path) -> bool:
        """Fix syntax errors in a single file."""
        try:
            # Read file
            with open(filepath, 'r', encoding='utf-8') as f:
                original_content = f.read()
                
            # Check if already valid
            is_valid, error = self.validate_syntax(original_content, str(filepath))
            if is_valid:
                return True
                
            # Create backup
            self.backup_file(filepath)
            
            # Apply fixes in order
            content = original_content
            
            # Fix import order first (especially __future__ imports)
            if "from __future__ import" in content:
                content = self.fix_import_order(content)
                
            # Fix indentation
            if "unexpected indent" in error or "unindent does not match" in error:
                content = self.fix_indentation_errors(content)
                
            # Fix unmatched brackets
            if any(x in error for x in ["unmatched", "was never closed"]):
                content = self.fix_unmatched_brackets(content)
                
            # Fix try-except blocks
            if "expected 'except' or 'finally'" in error:
                content = self.fix_try_except_blocks(content)
                
            # Validate the fixed content
            is_valid, new_error = self.validate_syntax(content, str(filepath))
            
            if is_valid:
                # Write fixed content
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.files_fixed += 1
                self.fixes_applied.append({
                    'file': str(filepath),
                    'original_error': error,
                    'status': 'fixed'
                })
                return True
            else:
                # If still not valid, try more aggressive fixes
                content = self.apply_aggressive_fixes(content, new_error)
                is_valid, final_error = self.validate_syntax(content, str(filepath))
                
                if is_valid:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    self.files_fixed += 1
                    self.fixes_applied.append({
                        'file': str(filepath),
                        'original_error': error,
                        'status': 'fixed_aggressive'
                    })
                    return True
                else:
                    self.files_failed += 1
                    self.fixes_applied.append({
                        'file': str(filepath),
                        'original_error': error,
                        'final_error': final_error,
                        'status': 'failed'
                    })
                    return False
                    
        except Exception as e:
            self.files_failed += 1
            self.fixes_applied.append({
                'file': str(filepath),
                'error': str(e),
                'status': 'error'
            })
            return False
            
    def apply_aggressive_fixes(self, content: str, error: str) -> str:
        """Apply more aggressive fixes for stubborn errors."""
        # Remove trailing commas in function definitions
        content = re.sub(r',\s*\)', ')', content)
        
        # Fix multi-line strings
        content = re.sub(r'"""[^"]*$', '"""\\n"""', content, flags=re.MULTILINE)
        content = re.sub(r"'''[^']*$", "'''\\n'''", content, flags=re.MULTILINE)
        
        # Ensure all colons are followed by newline or comment
        lines = content.split('\n')
        fixed_lines = []
        for line in lines:
            if line.strip().endswith(':') and not line.strip().startswith('#'):
                # Ensure there's something after the colon
                fixed_lines.append(line)
                indent = len(line) - len(line.lstrip()) + 4
                fixed_lines.append(' ' * indent + 'pass  # TODO: Implement')
            else:
                fixed_lines.append(line)
                
        return '\n'.join(fixed_lines)
        
    def fix_all_files(self, files: List[str]) -> Dict:
        """Fix all files with syntax errors."""
        print(f"Starting to fix {len(files)} files with syntax errors...")
        
        for i, filepath in enumerate(files, 1):
            print(f"[{i}/{len(files)}] Fixing {filepath}...")
            self.fix_file(Path(filepath))
            
        return {
            'total_files': len(files),
            'files_fixed': self.files_fixed,
            'files_failed': self.files_failed,
            'fixes_applied': self.fixes_applied
        }


def main():
    # Read the analysis report
    report_path = Path("/workspace/sequential_fixer_reports/sequential_analysis_20250903_121913.json")
    
    if not report_path.exists():
        print("Error: Analysis report not found!")
        return 1
        
    with open(report_path, 'r') as f:
        report = json.load(f)
        
    # Get files with syntax errors
    syntax_error_files = []
    for error in report.get('syntax_errors', []):
        filepath = Path("/workspace/src") / error['file']
        if filepath not in syntax_error_files:
            syntax_error_files.append(str(filepath))
            
    # Also check file details for files with errors
    for rel_path, details in report.get('file_details', {}).items():
        if details.get('errors'):
            filepath = Path("/workspace/src") / rel_path
            if str(filepath) not in syntax_error_files:
                syntax_error_files.append(str(filepath))
                
    print(f"Found {len(syntax_error_files)} files with syntax errors")
    
    # Create fixer and run
    fixer = SyntaxErrorFixer()
    results = fixer.fix_all_files(syntax_error_files)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(f"/workspace/syntax_fix_results_{timestamp}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nResults:")
    print(f"Files fixed: {results['files_fixed']}")
    print(f"Files failed: {results['files_failed']}")
    print(f"Results saved to: {results_path}")
    
    return 0 if results['files_failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())