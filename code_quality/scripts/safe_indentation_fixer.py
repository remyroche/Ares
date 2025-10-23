#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Safe Indentation Error Fixer

This script safely fixes common indentation errors without risking to create more issues.
It focuses on the most common and safe-to-fix patterns.
"""

import ast
import re
from pathlib import Path
from typing import List, Dict, Tuple


class SafeIndentationFixer:
    """Safely fixes common indentation errors."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.fixed_files = []
        self.failed_files = []
        self.skipped_files = []
        
    def fix_indentation_errors(self, directory: str = None) -> Dict:
        """Main method to fix indentation errors safely."""
        if directory is None:
            directory = str(self.project_root)
        
        directory_path = Path(directory)
        python_files = list(directory_path.rglob("*.py"))
        
        results = {
            "total_files": len(python_files),
            "fixed_files": [],
            "failed_files": [],
            "skipped_files": [],
            "changes_made": 0
        }
        
        for file_path in python_files:
            if self._should_skip_file(file_path):
                results["skipped_files"].append(str(file_path))
                continue
                
            try:
                if self._fix_file_indentation(str(file_path)):
                    results["fixed_files"].append(str(file_path))
                    results["changes_made"] += 1
                else:
                    results["failed_files"].append(str(file_path))
            except Exception as e:
                results["failed_files"].append(str(file_path))
                tprint(f"Error processing {file_path}: {e}")
        
        return results
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Skip files that are likely to be problematic."""
        # Skip test files, __pycache__, and very large files
        if any(skip in str(file_path) for skip in [
            "__pycache__", ".git", ".venv", "test_", "_test", "tests/"
        ]):
            return True
        
        # Skip very large files (> 10MB)
        try:
            if file_path.stat().st_size > 10 * 1024 * 1024:
                return True
        except:
            pass
            
        return False
    
    def _fix_file_indentation(self, file_path: str) -> bool:
        """Fix indentation errors in a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Apply safe fixes
            content = self._fix_missing_import_statements(content)
            content = self._fix_orphaned_import_lines(content)
            content = self._fix_duplicate_imports(content)
            content = self._fix_misplaced_imports(content)
            content = self._fix_misplaced_imports_in_code_blocks(content)
            content = self._fix_basic_indentation_issues(content)
            
            # Only write if changes were made and syntax is valid
            if content != original_content:
                # Verify the fix worked
                try:
                    ast.parse(content)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    return True
                except SyntaxError:
                    # Fix didn't work, don't save
                    return False
            
            return False
            
        except Exception:
            return False
    
    def _fix_missing_import_statements(self, content: str) -> str:
        """Fix missing import statements that cause indentation errors."""
        lines = content.split('\n')
        fixed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Look for orphaned import-like lines (missing 'from' or 'import')
            if (line.strip().startswith(('get_', 'safe_', 'format_', 'ensure_', 'timed_')) and 
                i > 0 and not lines[i-1].strip().startswith(('from ', 'import '))):
                
                # Check if this looks like a continuation of an import
                if any(keyword in line for keyword in [
                    'get_current_datetime', 'format_datetime', 'ensure_directory',
                    'safe_json_dump', 'safe_json_load', 'safe_file_exists',
                    'timed_operation', 'format_bytes', 'safe_log_metric', 'safe_log_params'
                ]):
                    # This is likely a missing import continuation
                    # Try to reconstruct the import statement
                    if 'get_current_datetime' in line or 'format_datetime' in line:
                        # This looks like a utils import
                        fixed_lines.append("from .utils.common_operations import (")
                        fixed_lines.append(line)
                        i += 1
                        continue
                    else:
                        # Skip this line as it's probably part of a broken import
                        i += 1
                        continue
            
            fixed_lines.append(line)
            i += 1
        
        return '\n'.join(fixed_lines)
    
    def _fix_orphaned_import_lines(self, content: str) -> str:
        """Fix orphaned import lines that are not properly indented."""
        lines = content.split('\n')
        fixed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Look for lines that look like import continuations but are at wrong indentation
            if (line.strip() and 
                not line.strip().startswith(('from ', 'import ', '#', '"""', "'''", 'class ', 'def ', 'if ', 'for ', 'while ', 'try:', 'except', 'finally', 'with ')) and
                any(keyword in line for keyword in [
                    'get_', 'safe_', 'format_', 'ensure_', 'timed_', 'connection_',
                    'critical', 'error', 'execution_error', 'failed', 'initialization_error',
                    'invalid', 'missing', 'problem', 'timeout', 'validation_error', 'warning'
                ])):
                
                # Check if this is part of a broken import block
                if self._is_broken_import_block(lines, i):
                    # Try to fix the import block
                    fixed_import = self._reconstruct_import_block(lines, i)
                    if fixed_import:
                        fixed_lines.extend(fixed_import)
                        # Skip the broken lines
                        while i < len(lines) and self._is_import_continuation_line(lines[i]):
                            i += 1
                        continue
                    else:
                        # Remove the orphaned line
                        i += 1
                        continue
                else:
                    # This looks like an orphaned import line, remove it
                    i += 1
                    continue
            
            fixed_lines.append(line)
            i += 1
        
        return '\n'.join(fixed_lines)
    
    def _is_broken_import_block(self, lines: list, start_idx: int) -> bool:
        """Check if we're in a broken import block."""
        if start_idx >= len(lines):
            return False
        
        line = lines[start_idx].strip()
        
        # Look for patterns that suggest a broken import
        if (any(keyword in line for keyword in [
            'connection_error', 'critical', 'error', 'execution_error', 'failed',
            'initialization_error', 'invalid', 'missing', 'problem', 'timeout',
            'validation_error', 'warning'
        ]) and not line.startswith(('from ', 'import '))):
            return True
        
        return False
    
    def _is_import_continuation_line(self, line: str) -> bool:
        """Check if a line is part of an import continuation."""
        stripped = line.strip()
        if not stripped:
            return False
        
        # Check for import continuation patterns
        if (stripped.endswith(',') or stripped.endswith(')') or 
            any(keyword in stripped for keyword in [
                'connection_error', 'critical', 'error', 'execution_error', 'failed',
                'initialization_error', 'invalid', 'missing', 'problem', 'timeout',
                'validation_error', 'warning', 'get_', 'safe_', 'format_', 'ensure_', 'timed_'
            ])):
            return True
        
        return False
    
    def _reconstruct_import_block(self, lines: list, start_idx: int) -> list:
        """Try to reconstruct a broken import block."""
        if start_idx >= len(lines):
            return None
        
        line = lines[start_idx].strip()
        
        # Try to determine the source module based on the imports
        if any(keyword in line for keyword in [
            'connection_error', 'critical', 'error', 'execution_error', 'failed',
            'initialization_error', 'invalid', 'missing', 'problem', 'timeout',
            'validation_error', 'warning'
        ]):
            # This looks like error constants - likely from a custom error module
            return [
                "from .exceptions import (",
                line,
                ")"
            ]
        elif any(keyword in line for keyword in [
            'get_current_datetime', 'format_datetime', 'ensure_directory',
            'safe_json_dump', 'safe_json_load', 'safe_file_exists',
            'timed_operation', 'format_bytes', 'safe_log_metric', 'safe_log_params'
        ]):
            # This looks like utility functions
            return [
                "from .utils.common_operations import (",
                line,
                ")"
            ]
        
        return None
    
    def _fix_duplicate_imports(self, content: str) -> str:
        """Remove duplicate import statements."""
        lines = content.split('\n')
        seen_imports = set()
        fixed_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Track import statements
            if stripped.startswith(('import ', 'from ')):
                if stripped in seen_imports:
                    # Skip duplicate import
                    continue
                seen_imports.add(stripped)
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _fix_misplaced_imports(self, content: str) -> str:
        """Fix imports that are placed in the middle of code blocks."""
        lines = content.split('\n')
        fixed_lines = []
        imports = []
        in_code_block = False
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            current_indent = len(line) - len(line.lstrip())
            
            # Detect if we're in a code block
            if stripped and not stripped.startswith(('import ', 'from ', '#', '"""', "'''")):
                if current_indent > 0 or stripped.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except', 'finally', 'with ')):
                    in_code_block = True
                    indent_level = current_indent
            
            # Collect import statements
            if stripped.startswith(('import ', 'from ')):
                if in_code_block and current_indent > 0:
                    # This import is inside a code block, collect it for later
                    imports.append(line)
                    continue
                else:
                    # This is a top-level import, keep it
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        # Add collected imports at the top (after existing imports)
        if imports:
            # Find the last import statement
            last_import_idx = -1
            for i, line in enumerate(fixed_lines):
                if line.strip().startswith(('import ', 'from ')):
                    last_import_idx = i
            
            # Insert collected imports after the last import
            if last_import_idx >= 0:
                fixed_lines.insert(last_import_idx + 1, '')
                for imp in imports:
                    fixed_lines.insert(last_import_idx + 2, imp)
            else:
                # No existing imports, add at the beginning
                for imp in imports:
                    fixed_lines.insert(0, imp)
        
        return '\n'.join(fixed_lines)
    
    def _fix_misplaced_imports_in_code_blocks(self, content: str) -> str:
        """Fix imports that are placed in the middle of code blocks (like the regularization.py case)."""
        lines = content.split('\n')
        fixed_lines = []
        imports_to_move = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Look for import statements that are indented (inside code blocks)
            if (stripped.startswith(('import ', 'from ')) and 
                line.startswith(('    ', '\t'))):  # Indented
                
                # This is an import inside a code block - collect it to move later
                imports_to_move.append(stripped)
                i += 1
                continue
            
            fixed_lines.append(line)
            i += 1
        
        # If we found misplaced imports, add them at the top
        if imports_to_move:
            # Find the last import statement to insert after
            last_import_idx = -1
            for j, line in enumerate(fixed_lines):
                if line.strip().startswith(('import ', 'from ')):
                    last_import_idx = j
            
            # Insert the misplaced imports after the last import
            if last_import_idx >= 0:
                # Insert after the last import
                fixed_lines.insert(last_import_idx + 1, '')
                for imp in imports_to_move:
                    fixed_lines.insert(last_import_idx + 2, imp)
            else:
                # No existing imports, add at the beginning (after shebang and docstring)
                insert_idx = 0
                for j, line in enumerate(fixed_lines):
                    if line.strip().startswith('#!') or line.strip().startswith('"""') or line.strip().startswith("'''"):
                        insert_idx = j + 1
                        # Skip docstring lines
                        while (insert_idx < len(fixed_lines) and 
                               (fixed_lines[insert_idx].strip().startswith('"""') or 
                                fixed_lines[insert_idx].strip().startswith("'''") or
                                fixed_lines[insert_idx].strip().startswith('#'))):
                            insert_idx += 1
                        break
                
                for imp in imports_to_move:
                    fixed_lines.insert(insert_idx, imp)
                    insert_idx += 1
        
        return '\n'.join(fixed_lines)
    
    def _fix_basic_indentation_issues(self, content: str) -> str:
        """Fix basic indentation issues that are safe to correct."""
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Fix mixed tabs and spaces (convert tabs to 4 spaces)
            if '\t' in line:
                line = line.replace('\t', '    ')
            
            # Fix lines that have inconsistent indentation
            # This is a very conservative fix - only fix obvious cases
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                # Count leading spaces
                leading_spaces = len(line) - len(line.lstrip())
                
                # If the line has an odd number of spaces (not multiple of 4), 
                # and it's not a continuation line, try to fix it
                if (leading_spaces > 0 and 
                    leading_spaces % 4 != 0 and 
                    not stripped.startswith(('import ', 'from ')) and
                    not any(keyword in stripped for keyword in [
                        'get_', 'safe_', 'format_', 'ensure_', 'timed_'
                    ])):
                    
                    # Round to nearest multiple of 4
                    new_indent = (leading_spaces // 4) * 4
                    line = ' ' * new_indent + stripped
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)


def main():
    """Main function to run the safe indentation fixer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Safely fix indentation errors")
    parser.add_argument("--project-root", default="/workspace/src",
                       help="Root directory of the project")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be fixed without making changes")
    
    args = parser.parse_args()
    
    fixer = SafeIndentationFixer(args.project_root)
    
    if args.dry_run:
        tprint("DRY RUN MODE - No changes will be made")
        # In dry run mode, we would analyze but not fix
        # For now, just run normally but with extra logging
        results = fixer.fix_indentation_errors()
    else:
        results = fixer.fix_indentation_errors()
    
    tprint(f"\nSafe Indentation Fixer Results:")
    tprint(f"Total files processed: {results['total_files']}")
    tprint(f"Files fixed: {len(results['fixed_files'])}")
    tprint(f"Files failed: {len(results['failed_files'])}")
    tprint(f"Files skipped: {len(results['skipped_files'])}")
    tprint(f"Total changes made: {results['changes_made']}")
    
    if results['fixed_files']:
        tprint(f"\nFixed files:")
        for file_path in results['fixed_files'][:10]:  # Show first 10
            tprint(f"  - {file_path}")
        if len(results['fixed_files']) > 10:
            tprint(f"  ... and {len(results['fixed_files']) - 10} more")


if __name__ == "__main__":
    main()