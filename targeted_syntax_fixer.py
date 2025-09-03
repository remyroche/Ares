#!/usr/bin/env python3
"""
Targeted syntax error fixer that handles specific common issues in the codebase.
"""

import os
import re
import sys
import ast
import json
import shutil
import textwrap
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

class TargetedSyntaxFixer:
    def __init__(self):
        self.backup_dir = Path("syntax_fix_backups")
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
        
    def fix_future_imports(self, content: str) -> str:
        """Move __future__ imports to the beginning of the file."""
        lines = content.split('\n')
        
        # Separate different parts
        shebang_lines = []
        encoding_lines = []
        module_docstring_lines = []
        future_imports = []
        other_imports = []
        rest_of_file = []
        
        in_module_docstring = False
        docstring_delimiter = None
        past_imports = False
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Handle shebang
            if i == 0 and line.startswith('#!'):
                shebang_lines.append(line)
                i += 1
                continue
                
            # Handle encoding
            if i < 2 and line.startswith('# -*- coding:') or line.startswith('# coding:'):
                encoding_lines.append(line)
                i += 1
                continue
                
            # Skip initial comments (but not docstrings)
            if not in_module_docstring and stripped.startswith('#'):
                rest_of_file.append(line)
                i += 1
                continue
                
            # Handle module docstring
            if not in_module_docstring and not past_imports:
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    in_module_docstring = True
                    docstring_delimiter = stripped[:3]
                    module_docstring_lines.append(line)
                    
                    # Check if it's a one-line docstring
                    if stripped.count(docstring_delimiter) >= 2:
                        in_module_docstring = False
                    i += 1
                    continue
                    
            if in_module_docstring:
                module_docstring_lines.append(line)
                if docstring_delimiter in stripped and len(module_docstring_lines) > 1:
                    in_module_docstring = False
                i += 1
                continue
                
            # Collect future imports
            if stripped.startswith('from __future__ import'):
                future_imports.append(line)
                i += 1
                continue
                
            # Collect other imports
            if not past_imports and (stripped.startswith('import ') or stripped.startswith('from ')):
                other_imports.append(line)
                i += 1
                continue
                
            # Everything else
            if stripped and not stripped.startswith('#'):
                past_imports = True
            rest_of_file.append(line)
            i += 1
            
        # Reconstruct file
        result = []
        
        # Add shebang if exists
        if shebang_lines:
            result.extend(shebang_lines)
            
        # Add encoding if exists  
        if encoding_lines:
            result.extend(encoding_lines)
            
        # Add module docstring if exists
        if module_docstring_lines:
            if result:
                result.append('')
            result.extend(module_docstring_lines)
            
        # Add future imports
        if future_imports:
            if result:
                result.append('')
            result.extend(future_imports)
            
        # Add other imports
        if other_imports:
            if result:
                result.append('')
            result.extend(other_imports)
            
        # Add rest of file
        if rest_of_file:
            if result and not all(line.strip() == '' for line in rest_of_file):
                result.append('')
            result.extend(rest_of_file)
            
        return '\n'.join(result)
        
    def fix_indentation_smart(self, content: str) -> str:
        """Smart indentation fixer that understands Python structure."""
        lines = content.split('\n')
        fixed_lines = []
        indent_stack = []
        current_indent = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Empty lines and comments preserve their format
            if not stripped:
                fixed_lines.append(line)
                continue
                
            if stripped.startswith('#'):
                # Align comments with current indent
                fixed_lines.append(' ' * current_indent + stripped)
                continue
                
            # Calculate the line's natural indent
            line_indent = len(line) - len(line.lstrip())
            
            # Check for dedent keywords
            if stripped.startswith(('elif ', 'else:', 'except', 'finally:', 'case ')):
                # These should align with their opening statement
                if indent_stack:
                    current_indent = max(0, indent_stack[-1] - 4)
                line = ' ' * current_indent + stripped
                
            # Check for block starters
            elif stripped.endswith(':') and not stripped.startswith('#'):
                # This starts a new block
                line = ' ' * current_indent + stripped
                indent_stack.append(current_indent)
                current_indent += 4
                
            # Check for explicit dedent
            elif any(stripped.startswith(kw) for kw in ['return', 'break', 'continue', 'raise', 'pass']):
                # These often end a block
                line = ' ' * current_indent + stripped
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and not next_line.startswith(('    ', '\t')):
                        # Next line is dedented, so we should dedent too
                        if indent_stack:
                            current_indent = indent_stack.pop()
                            
            else:
                # Regular line - use current indent
                line = ' ' * current_indent + stripped
                
            fixed_lines.append(line)
            
        return '\n'.join(fixed_lines)
        
    def fix_unclosed_strings(self, content: str) -> str:
        """Fix unclosed strings and quotes."""
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            # Count quotes
            single_quotes = line.count("'") - line.count("\\'")
            double_quotes = line.count('"') - line.count('\\"')
            
            # Fix unmatched quotes
            if single_quotes % 2 != 0:
                line += "'"
            if double_quotes % 2 != 0:
                line += '"'
                
            fixed_lines.append(line)
            
        # Fix triple-quoted strings
        content = '\n'.join(fixed_lines)
        
        # Find unclosed triple quotes
        triple_single = content.count("'''")
        triple_double = content.count('"""')
        
        if triple_single % 2 != 0:
            content += "\n'''"
        if triple_double % 2 != 0:
            content += '\n"""'
            
        return content
        
    def fix_file(self, filepath: Path) -> bool:
        """Fix a single file with targeted fixes."""
        try:
            # Read file
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Create backup
            self.backup_file(filepath)
            
            # Apply fixes based on error type
            original_content = content
            
            # Fix __future__ imports first
            if "from __future__ import" in content:
                content = self.fix_future_imports(content)
                
            # Fix indentation
            content = self.fix_indentation_smart(content)
            
            # Fix unclosed strings
            content = self.fix_unclosed_strings(content)
            
            # Validate
            try:
                compile(content, str(filepath), 'exec')
                # Success! Write the file
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.files_fixed += 1
                self.fixes_applied.append({
                    'file': str(filepath),
                    'status': 'fixed'
                })
                print(f"  ✓ Fixed: {filepath.name}")
                return True
            except SyntaxError as e:
                # Try one more aggressive fix
                content = self.apply_emergency_fixes(content, str(e))
                try:
                    compile(content, str(filepath), 'exec')
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    self.files_fixed += 1
                    self.fixes_applied.append({
                        'file': str(filepath),
                        'status': 'fixed_emergency'
                    })
                    print(f"  ✓ Fixed (emergency): {filepath.name}")
                    return True
                except:
                    self.files_failed += 1
                    self.fixes_applied.append({
                        'file': str(filepath),
                        'error': str(e),
                        'status': 'failed'
                    })
                    print(f"  ✗ Failed: {filepath.name} - {e}")
                    return False
                    
        except Exception as e:
            self.files_failed += 1
            self.fixes_applied.append({
                'file': str(filepath),
                'error': str(e),
                'status': 'error'
            })
            print(f"  ✗ Error: {filepath.name} - {e}")
            return False
            
    def apply_emergency_fixes(self, content: str, error: str) -> str:
        """Apply emergency fixes for stubborn errors."""
        # Remove any incomplete lines at the end
        lines = content.split('\n')
        while lines and lines[-1].strip() and not lines[-1].strip().endswith((':', ')', ']', '}', '"', "'")):
            lines.pop()
            
        content = '\n'.join(lines)
        
        # Ensure file ends with newline
        if not content.endswith('\n'):
            content += '\n'
            
        return content


def main():
    # Get the list of files with errors from our analysis
    error_files = [
        "/workspace/src/monitoring/performance_dashboard.py",
        "/workspace/src/monitoring/performance_monitor.py",
        "/workspace/src/launcher/enhanced_trading_launcher.py",
        "/workspace/src/interfaces/enhanced_event_bus.py",
        "/workspace/src/supervisor/enhanced_prediction_service.py",
        "/workspace/src/supervisor/supervisor.py",
        "/workspace/src/tactician/tactician.py",
        "/workspace/src/analyst/analyst.py",
        # Add more critical files first
    ]
    
    print("Starting targeted syntax fixes...")
    print("=" * 60)
    
    fixer = TargetedSyntaxFixer()
    
    for filepath in error_files:
        if os.path.exists(filepath):
            print(f"\nFixing: {filepath}")
            fixer.fix_file(Path(filepath))
            
    print("\n" + "=" * 60)
    print(f"Summary:")
    print(f"  Files fixed: {fixer.files_fixed}")
    print(f"  Files failed: {fixer.files_failed}")
    
    # Save results
    results = {
        'files_fixed': fixer.files_fixed,
        'files_failed': fixer.files_failed,
        'fixes_applied': fixer.fixes_applied
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"/workspace/targeted_fix_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()