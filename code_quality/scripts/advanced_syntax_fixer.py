#!/usr/bin/env python3
"""
Advanced syntax fixer for Python files with complex syntax errors.
This script attempts to fix common syntax errors that prevent files from being parsed.
"""

import re
import ast
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import subprocess
from collections import defaultdict


class AdvancedSyntaxFixer:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.fixed_files = []
        self.failed_files = []
        self.syntax_errors = defaultdict(list)
        
    def diagnose_syntax_error(self, file_path: str) -> Optional[Dict]:
        """Diagnose syntax error in a file using Python's parser."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to parse and catch the specific error
            try:
                ast.parse(content)
                return None  # No syntax error
            except SyntaxError as e:
                return {
                    'type': type(e).__name__,
                    'msg': str(e.msg),
                    'line': e.lineno,
                    'offset': e.offset,
                    'text': e.text
                }
        except Exception as e:
            return {'type': 'Unknown', 'msg': str(e), 'line': 0}
    
    def fix_common_syntax_errors(self, file_path: str) -> bool:
        """Apply common syntax fixes to a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            lines = content.split('\n')
            
            # Get syntax error details
            error = self.diagnose_syntax_error(file_path)
            if not error:
                return False  # No syntax error to fix
            
            self.syntax_errors[error['msg']].append(file_path)
            
            # Apply fixes based on error type
            if "expected 'except' or 'finally' block" in error['msg']:
                content = self.fix_try_except_blocks(content)
            
            elif "unexpected indent" in error['msg'] and error['line']:
                lines = self.fix_indentation_error(lines, error['line'] - 1)
                content = '\n'.join(lines)
            
            elif "invalid syntax" in error['msg']:
                content = self.fix_invalid_syntax(content, error)
            
            elif "expected ':'" in error['msg']:
                content = self.fix_missing_colon(content, error)
            
            # Additional common fixes
            content = self.apply_common_fixes(content)
            
            # Only write if changes were made and syntax is now valid
            if content != original_content:
                # Verify the fix worked
                try:
                    ast.parse(content)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    self.fixed_files.append(file_path)
                    return True
                except SyntaxError:
                    # Fix didn't work, don't save
                    pass
            
            self.failed_files.append(file_path)
            return False
            
        except Exception as e:
            self.failed_files.append(file_path)
            return False
    
    def fix_try_except_blocks(self, content: str) -> str:
        """Fix try blocks without except/finally."""
        lines = content.split('\n')
        fixed_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Detect try block
            if line.strip().startswith('try:'):
                try_indent = len(line) - len(line.lstrip())
                block_lines = [line]
                i += 1
                
                # Collect try block content
                while i < len(lines):
                    current_line = lines[i]
                    current_indent = len(current_line) - len(current_line.lstrip())
                    
                    if current_line.strip() and current_indent <= try_indent:
                        # End of try block
                        break
                    
                    block_lines.append(current_line)
                    i += 1
                
                # Check if except/finally follows
                has_handler = False
                if i < len(lines):
                    next_line = lines[i].strip()
                    if next_line.startswith(('except', 'finally')):
                        has_handler = True
                
                # Add the try block
                fixed_lines.extend(block_lines)
                
                # Add except block if missing
                if not has_handler:
                    except_line = ' ' * try_indent + 'except Exception as e:'
                    pass_line = ' ' * (try_indent + 4) + 'pass'
                    fixed_lines.extend([except_line, pass_line])
                
                continue
            
            fixed_lines.append(line)
            i += 1
        
        return '\n'.join(fixed_lines)
    
    def fix_indentation_error(self, lines: List[str], error_line: int) -> List[str]:
        """Fix indentation errors."""
        if 0 <= error_line < len(lines):
            # Get expected indentation from previous lines
            expected_indent = 0
            
            # Look for the previous non-empty line
            for i in range(error_line - 1, -1, -1):
                if lines[i].strip():
                    prev_indent = len(lines[i]) - len(lines[i].lstrip())
                    # If previous line ends with :, increase indent
                    if lines[i].rstrip().endswith(':'):
                        expected_indent = prev_indent + 4
                    else:
                        expected_indent = prev_indent
                    break
            
            # Fix the indentation
            if error_line < len(lines):
                line = lines[error_line]
                stripped = line.lstrip()
                if stripped:
                    lines[error_line] = ' ' * expected_indent + stripped
        
        return lines
    
    def fix_invalid_syntax(self, content: str, error: Dict) -> str:
        """Fix various invalid syntax patterns."""
        # Fix asyncio.run(await ...)
        content = re.sub(r'asyncio\.run\s*\(\s*await\s+', 'asyncio.run(', content)
        
        # Fix missing commas in lists/dicts
        content = re.sub(r'(["\'])\s*\n\s*(["\'])', r'\1,\n    \2', content)
        
        # Fix f-string errors
        content = re.sub(r'f"([^"]*)\{([^}]*)\}"', r'f"\1{\2}"', content)
        
        # Fix import statement errors
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('import') and line.count('import') > 1:
                # Fix multiple imports on one line
                parts = line.split('import')
                if len(parts) > 2:
                    lines[i] = parts[0] + 'import' + parts[1]
        
        return '\n'.join(lines)
    
    def fix_missing_colon(self, content: str, error: Dict) -> str:
        """Fix missing colons after function/class definitions."""
        lines = content.split('\n')
        
        if error['line'] and 0 < error['line'] <= len(lines):
            line_idx = error['line'] - 1
            line = lines[line_idx]
            
            # Check for function/class definition without colon
            if re.match(r'^\s*(def|class|if|elif|else|while|for|with|try|except|finally)\s+.*[^:]$', line):
                lines[line_idx] = line + ':'
        
        return '\n'.join(lines)
    
    def apply_common_fixes(self, content: str) -> str:
        """Apply additional common fixes."""
        # Fix None comparison
        content = re.sub(r'(\s+)if\s+(\w+)\s+==\s+None:', r'\1if \2 is None:', content)
        content = re.sub(r'(\s+)if\s+(\w+)\s+!=\s+None:', r'\1if \2 is not None:', content)
        
        # Fix print statements (Python 2 to 3)
        content = re.sub(r'print\s+"([^"]*)"$', r'print("\1")', content, flags=re.MULTILINE)
        content = re.sub(r'print\s+\'([^\']*)\'$', r'print(\'\1\')', content, flags=re.MULTILINE)
        
        # Fix except statements
        content = re.sub(r'except\s+(\w+),\s*(\w+):', r'except \1 as \2:', content)
        
        return content
    
    def fix_all_syntax_errors(self, dry_run: bool = True):
        """Fix syntax errors in all Python files."""
        python_files = list(self.project_root.rglob("*.py"))
        
        # Filter out excluded directories
        python_files = [
            f for f in python_files 
            if '__pycache__' not in str(f) and 
               '.venv' not in str(f) and
               'venv' not in str(f)
        ]
        
        print(f"Checking {len(python_files)} Python files for syntax errors...")
        
        # First, identify files with syntax errors
        files_with_errors = []
        for file_path in python_files:
            error = self.diagnose_syntax_error(str(file_path))
            if error:
                files_with_errors.append((file_path, error))
        
        print(f"Found {len(files_with_errors)} files with syntax errors")
        
        if dry_run:
            # Show error types
            error_types = defaultdict(int)
            for _, error in files_with_errors:
                error_types[error['msg']] += 1
            
            print("\nError types found:")
            for error_msg, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  {error_msg}: {count} files")
            
            print("\nSample files with errors:")
            for file_path, error in files_with_errors[:10]:
                print(f"  {file_path.name}: {error['msg']} (line {error.get('line', '?')})")
            
            return {
                'dry_run': True,
                'files_with_errors': len(files_with_errors),
                'error_types': dict(error_types)
            }
        else:
            # Actually fix the files
            for file_path, _ in files_with_errors:
                self.fix_common_syntax_errors(str(file_path))
            
            print(f"\nFixed {len(self.fixed_files)} files")
            print(f"Failed to fix {len(self.failed_files)} files")
            
            # Show common error patterns that couldn't be fixed
            if self.syntax_errors:
                print("\nCommon unfixed error patterns:")
                for error_msg, files in sorted(self.syntax_errors.items(), 
                                             key=lambda x: len(x[1]), reverse=True)[:5]:
                    print(f"  {error_msg}: {len(files)} files")
            
            return {
                'fixed': len(self.fixed_files),
                'failed': len(self.failed_files),
                'fixed_files': self.fixed_files[:10],
                'error_patterns': {k: len(v) for k, v in self.syntax_errors.items()}
            }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix syntax errors in Python files')
    parser.add_argument('--project-root', default='/workspace/src',
                       help='Root directory of the project')
    parser.add_argument('--fix', action='store_true',
                       help='Actually fix the files (default is dry run)')
    
    args = parser.parse_args()
    
    fixer = AdvancedSyntaxFixer(args.project_root)
    result = fixer.fix_all_syntax_errors(dry_run=not args.fix)
    
    # Save report
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f'/workspace/code_quality/reports/syntax_fixes_report_{timestamp}.json'
    Path(report_file).parent.mkdir(exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"\nReport saved to: {report_file}")


if __name__ == '__main__':
    main()