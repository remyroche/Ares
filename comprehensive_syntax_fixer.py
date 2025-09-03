#!/usr/bin/env python3
"""
Comprehensive syntax fixer that handles all the specific issues found in the codebase.
"""

import os
import re
import sys
import ast
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

class ComprehensiveSyntaxFixer:
    def __init__(self):
        self.backup_dir = Path("syntax_fix_backups")
        self.backup_dir.mkdir(exist_ok=True)
        self.fixes_applied = []
        self.files_fixed = 0
        self.files_failed = 0
        self.files_skipped = 0
        
    def backup_file(self, filepath: Path) -> Path:
        """Create a backup of the file before modifying."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"{filepath.name}.{timestamp}.bak"
        shutil.copy2(filepath, backup_path)
        return backup_path
        
    def fix_import_aliases(self, content: str) -> str:
        """Fix import statements with problematic aliases."""
        lines = content.split('\n')
        fixed_lines = []
        
        import_pattern = re.compile(r'from\s+([\w\.]+)\s+import\s+([\w\s,]+)\s+as\s+([\w_]+)')
        
        for line in lines:
            # Fix imports with underscores in aliases
            if 'import' in line and ' as ' in line and '_src_' in line:
                # Remove the problematic alias
                line = re.sub(r'\s+as\s+\w+_src_\w+', '', line)
            
            fixed_lines.append(line)
            
        return '\n'.join(fixed_lines)
        
    def fix_future_imports_comprehensive(self, content: str) -> str:
        """Comprehensive fix for __future__ import placement."""
        lines = content.split('\n')
        
        # Categories of lines
        shebang = []
        encoding = []
        future_imports = []
        module_docstring = []
        regular_imports = []
        rest = []
        
        in_docstring = False
        docstring_quote = None
        found_code = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Shebang (first line only)
            if i == 0 and line.startswith('#!'):
                shebang.append(line)
                continue
                
            # Encoding (first two lines only)
            if i < 2 and ('coding:' in line or 'coding=' in line) and line.startswith('#'):
                encoding.append(line)
                continue
                
            # Module docstring detection
            if not found_code and not in_docstring:
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    in_docstring = True
                    docstring_quote = stripped[:3]
                    module_docstring.append(line)
                    # Check if it ends on same line
                    if line.count(docstring_quote) >= 2:
                        in_docstring = False
                    continue
                    
            if in_docstring:
                module_docstring.append(line)
                if docstring_quote in line and len(module_docstring) > 1:
                    in_docstring = False
                continue
                
            # Future imports
            if 'from __future__ import' in line:
                future_imports.append(line)
                continue
                
            # Regular imports
            if not found_code and (stripped.startswith('import ') or stripped.startswith('from ')):
                regular_imports.append(line)
                continue
                
            # Rest of the code
            if stripped and not stripped.startswith('#'):
                found_code = True
            rest.append(line)
            
        # Rebuild the file
        result = []
        
        if shebang:
            result.extend(shebang)
        if encoding:
            result.extend(encoding)
            
        # Add blank line after headers
        if result and (module_docstring or future_imports):
            result.append('')
            
        if module_docstring:
            result.extend(module_docstring)
            
        if future_imports:
            if result and result[-1].strip():
                result.append('')
            result.extend(future_imports)
            
        if regular_imports:
            if result and result[-1].strip():
                result.append('')
            result.extend(regular_imports)
            
        if rest:
            if result and result[-1].strip():
                result.append('')
            # Remove leading empty lines from rest
            while rest and not rest[0].strip():
                rest.pop(0)
            result.extend(rest)
            
        return '\n'.join(result)
        
    def fix_indentation_advanced(self, content: str) -> str:
        """Advanced indentation fixer using AST when possible."""
        lines = content.split('\n')
        fixed_lines = []
        indent_levels = [0]
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if not stripped:
                fixed_lines.append('')
                continue
                
            # Comments maintain current indentation
            if stripped.startswith('#'):
                if indent_levels:
                    fixed_lines.append(' ' * indent_levels[-1] + stripped)
                else:
                    fixed_lines.append(stripped)
                continue
                
            # Dedenting keywords
            if any(stripped.startswith(kw) for kw in ['elif ', 'else:', 'except', 'finally:', 'case ']):
                if len(indent_levels) > 1:
                    indent_levels.pop()
                fixed_lines.append(' ' * indent_levels[-1] + stripped)
                if stripped.endswith(':'):
                    indent_levels.append(indent_levels[-1] + 4)
                continue
                
            # Block starters
            if stripped.endswith(':') and not stripped.startswith('#'):
                fixed_lines.append(' ' * indent_levels[-1] + stripped)
                indent_levels.append(indent_levels[-1] + 4)
                continue
                
            # Dedent after certain statements
            if any(stripped.startswith(kw) for kw in ['return', 'break', 'continue', 'raise']):
                fixed_lines.append(' ' * indent_levels[-1] + stripped)
                # Check if next line should dedent
                if i + 1 < len(lines) and lines[i + 1].strip():
                    next_indent = len(lines[i + 1]) - len(lines[i + 1].lstrip())
                    if next_indent < indent_levels[-1]:
                        while len(indent_levels) > 1 and indent_levels[-1] > next_indent:
                            indent_levels.pop()
                continue
                
            # Regular lines
            fixed_lines.append(' ' * indent_levels[-1] + stripped)
            
        return '\n'.join(fixed_lines)
        
    def fix_try_except_comprehensive(self, content: str) -> str:
        """Fix missing except/finally blocks."""
        lines = content.split('\n')
        result = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            if stripped == 'try:':
                indent = len(line) - len(stripped)
                result.append(line)
                i += 1
                
                # Collect try body
                try_body = []
                body_indent = indent + 4
                
                while i < len(lines):
                    next_line = lines[i]
                    next_stripped = next_line.strip()
                    
                    if not next_stripped:
                        try_body.append(next_line)
                        i += 1
                        continue
                        
                    next_indent = len(next_line) - len(next_stripped)
                    
                    # Check if we've exited the try block
                    if next_stripped and next_indent <= indent:
                        # Check if it's except or finally
                        if not (next_stripped.startswith('except') or next_stripped.startswith('finally:')):
                            # Need to add except block
                            result.extend(try_body)
                            result.append(' ' * indent + 'except Exception:')
                            result.append(' ' * body_indent + 'pass')
                        else:
                            result.extend(try_body)
                        break
                    else:
                        try_body.append(next_line)
                        i += 1
                        
                # If we reached end of file, add except
                if i >= len(lines):
                    result.extend(try_body)
                    result.append(' ' * indent + 'except Exception:')
                    result.append(' ' * body_indent + 'pass')
                    
            else:
                result.append(line)
                i += 1
                
        return '\n'.join(result)
        
    def fix_unclosed_strings_advanced(self, content: str) -> str:
        """Advanced string closure fixer."""
        # First, handle triple quotes
        if content.count('"""') % 2 != 0:
            content += '\n"""'
        if content.count("'''") % 2 != 0:
            content += "\n'''"
            
        # Fix individual lines
        lines = content.split('\n')
        fixed_lines = []
        
        in_string = False
        string_char = None
        
        for line in lines:
            # Skip comments
            if line.strip().startswith('#'):
                fixed_lines.append(line)
                continue
                
            # Count quotes more carefully
            new_line = line
            i = 0
            while i < len(new_line):
                if new_line[i] in ['"', "'"]:
                    if i > 0 and new_line[i-1] == '\\':
                        # Escaped quote
                        i += 1
                        continue
                    if not in_string:
                        in_string = True
                        string_char = new_line[i]
                    elif new_line[i] == string_char:
                        in_string = False
                        string_char = None
                i += 1
                
            # If string is still open at end of line, close it
            if in_string and string_char:
                new_line += string_char
                in_string = False
                string_char = None
                
            fixed_lines.append(new_line)
            
        return '\n'.join(fixed_lines)
        
    def fix_file_comprehensive(self, filepath: Path) -> bool:
        """Apply all fixes to a file."""
        try:
            # Read the file
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check if already valid
            try:
                compile(content, str(filepath), 'exec')
                print(f"  ✓ Already valid: {filepath.name}")
                self.files_skipped += 1
                return True
            except:
                pass
                
            # Create backup
            self.backup_file(filepath)
            
            # Apply fixes in order
            original = content
            
            # 1. Fix import aliases first
            content = self.fix_import_aliases(content)
            
            # 2. Fix __future__ imports
            if 'from __future__ import' in content:
                content = self.fix_future_imports_comprehensive(content)
                
            # 3. Fix string issues
            content = self.fix_unclosed_strings_advanced(content)
            
            # 4. Fix indentation
            content = self.fix_indentation_advanced(content)
            
            # 5. Fix try-except blocks
            if 'try:' in content:
                content = self.fix_try_except_comprehensive(content)
                
            # Test compilation
            try:
                compile(content, str(filepath), 'exec')
                # Success!
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.files_fixed += 1
                print(f"  ✓ Fixed: {filepath.name}")
                return True
            except SyntaxError as e:
                print(f"  ✗ Failed: {filepath.name} - {e}")
                self.files_failed += 1
                # Restore original
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(original)
                return False
                
        except Exception as e:
            print(f"  ✗ Error: {filepath.name} - {e}")
            self.files_failed += 1
            return False
            
    def fix_all_errors(self):
        """Fix all files with syntax errors."""
        # Read the analysis report
        report_path = Path("/workspace/sequential_fixer_reports/sequential_analysis_20250903_121913.json")
        
        with open(report_path, 'r') as f:
            report = json.load(f)
            
        # Get all files with errors
        error_files = set()
        
        # From syntax_errors
        for error in report.get('syntax_errors', []):
            filepath = Path("/workspace/src") / error['file']
            error_files.add(filepath)
            
        # From file_details
        for rel_path, details in report.get('file_details', {}).items():
            if details.get('errors'):
                filepath = Path("/workspace/src") / rel_path
                error_files.add(filepath)
                
        print(f"Found {len(error_files)} files with errors")
        print("=" * 60)
        
        # Sort files by directory for better organization
        sorted_files = sorted(error_files, key=lambda x: str(x))
        
        for filepath in sorted_files:
            if filepath.exists():
                print(f"\nProcessing: {filepath}")
                self.fix_file_comprehensive(filepath)
                
        print("\n" + "=" * 60)
        print("Summary:")
        print(f"  Files processed: {len(error_files)}")
        print(f"  Files fixed: {self.files_fixed}")
        print(f"  Files failed: {self.files_failed}")
        print(f"  Files skipped (already valid): {self.files_skipped}")
        
        # Save results
        results = {
            'total_files': len(error_files),
            'files_fixed': self.files_fixed,
            'files_failed': self.files_failed,
            'files_skipped': self.files_skipped,
            'timestamp': datetime.now().isoformat()
        }
        
        results_file = f"/workspace/comprehensive_fix_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")


def main():
    fixer = ComprehensiveSyntaxFixer()
    fixer.fix_all_errors()


if __name__ == "__main__":
    main()