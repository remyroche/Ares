#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Script to fix incorrectly placed imports that were added in the middle of functions.
"""

import ast
import re
from pathlib import Path
from typing import List, Tuple
import numpy as np


def find_incorrect_imports(file_path: str) -> List[Tuple[int, str]]:
    """Find imports that are incorrectly placed in the middle of functions or incomplete import statements."""
    incorrect_imports = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Look for import statements that are clearly in the middle of code
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('import numpy as np') or stripped.startswith('import pandas as pd') or stripped.startswith('import warnings'):
                # Check if this import is in the middle of an incomplete from import
                is_in_incomplete_from = False
                if i > 0:
                    prev_line = lines[i-1].strip()
                    if prev_line.endswith('(') and 'from ' in prev_line:
                        is_in_incomplete_from = True
                
                # Check if this import is not in the first 50 lines (should be at top)
                if i > 50:
                    incorrect_imports.append((i, stripped))
                elif is_in_incomplete_from:
                    # This is definitely wrong - import in middle of from statement
                    incorrect_imports.append((i, stripped))
                else:
                    # Check if this import is in the middle of a try block or other control structure
                    is_in_control_structure = False
                    if i > 0:
                        # Look backwards to see if we're inside a try/except/if/for/while block
                        brace_count = 0
                        for j in range(i-1, max(0, i-20), -1):  # Look back up to 20 lines
                            line = lines[j].strip()
                            if line.startswith('try:') or line.startswith('except') or line.startswith('if ') or line.startswith('for ') or line.startswith('while '):
                                is_in_control_structure = True
                                break
                            elif line.startswith('class ') or line.startswith('def '):
                                break  # We're in a class/function, not a control structure
                    
                    if is_in_control_structure:
                        incorrect_imports.append((i, stripped))
                    else:
                        # Check if the previous line is not an import or blank
                        if i > 0:
                            prev_line = lines[i-1].strip()
                            if prev_line and not prev_line.startswith('import') and not prev_line.startswith('from') and not prev_line.startswith('#'):
                                # Check if the next line is not an import or blank
                                if i < len(lines) - 1:
                                    next_line = lines[i+1].strip()
                                    if next_line and not next_line.startswith('import') and not next_line.startswith('from') and not next_line.startswith('#'):
                                        incorrect_imports.append((i, stripped))
                                else:
                                    # This is the last line, so it's definitely misplaced
                                    incorrect_imports.append((i, stripped))
                        else:
                            # This is the first line, so it's definitely misplaced
                            incorrect_imports.append((i, stripped))
    
    except Exception as e:
        tprint(f"Error analyzing {file_path}: {e}")
    
    return incorrect_imports


def fix_incorrect_imports(file_path: str) -> bool:
    """Fix incorrectly placed imports by moving them to the top and fixing incomplete imports."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Find incorrect imports
        incorrect_imports = find_incorrect_imports(file_path)
        
        if not incorrect_imports:
            return False
        
        tprint(f"Found {len(incorrect_imports)} incorrect imports in {file_path}")
        
        # Collect the imports to move and fix incomplete imports
        imports_to_move = []
        lines_to_remove = []
        incomplete_imports_to_fix = []
        
        for line_num, import_stmt in incorrect_imports:
            # Check if this is in the middle of an incomplete from import
            if line_num > 0:
                prev_line = lines[line_num-1].strip()
                if prev_line.endswith('(') and 'from ' in prev_line:
                    # This is in an incomplete from import - we need to fix the from import too
                    incomplete_imports_to_fix.append((line_num-1, prev_line, import_stmt))
                    lines_to_remove.append(line_num-1)  # Remove the incomplete from line
                    lines_to_remove.append(line_num)    # Remove the misplaced import
                    
                    # Also remove any dangling import items that follow
                    i = line_num + 1
                    while i < len(lines):
                        next_line = lines[i].strip()
                        if next_line and (next_line.endswith(',') or next_line.endswith(')')):
                            lines_to_remove.append(i)
                            if next_line.endswith(')'):
                                break
                        elif next_line and not next_line.startswith('import') and not next_line.startswith('from') and not next_line.startswith('class') and not next_line.startswith('def'):
                            # This might be a dangling import item - check if it looks like an import item
                            if (',' in next_line or 
                                next_line in ['execution_error', 'initialization_error', 'invalid', 'validation_error'] or
                                next_line.endswith(',') or
                                (next_line and not next_line.startswith('#') and not next_line.startswith('"""') and not next_line.startswith("'''"))):
                                lines_to_remove.append(i)
                            else:
                                break
                        else:
                            break
                        i += 1
                    continue
            
            imports_to_move.append(import_stmt)
            lines_to_remove.append(line_num)
        
        # Remove the incorrect imports (in reverse order to maintain line numbers)
        for line_num in sorted(lines_to_remove, reverse=True):
            del lines[line_num]
        
        # Re-read the file to get updated line numbers after removals
        # This is needed because we might have moved imports that are still in wrong places
        lines = lines  # Keep the current state
        
        # Find the right place to insert imports at the top
        insert_line = 0
        
        # Handle shebang
        if lines and lines[0].startswith('#!'):
            insert_line = 1
        
        # Handle module docstring
        if lines and (lines[0].startswith('"""') or lines[0].startswith("'''")):
            for i, line in enumerate(lines[1:], 1):
                if line.strip().endswith('"""') or line.strip().endswith("'''"):
                    insert_line = i + 1
                    break
        
        # Find existing imports - be more careful about multi-line imports
        last_import_line = insert_line
        in_multiline_import = False
        
        for i, line in enumerate(lines[insert_line:], insert_line):
            stripped = line.strip()
            
            if stripped.startswith('import ') or stripped.startswith('from '):
                if '(' in stripped:  # Multi-line import
                    in_multiline_import = True
                elif ')' in stripped:  # End of multi-line import
                    in_multiline_import = False
                    last_import_line = i + 1
                elif not in_multiline_import:
                    last_import_line = i + 1
            elif stripped and not stripped.startswith('#') and not in_multiline_import:
                break
        
        # Insert the imports
        for import_stmt in sorted(set(imports_to_move)):
            lines.insert(last_import_line, import_stmt + '\n')
            last_import_line += 1
        
        # Handle incomplete imports by creating proper separate imports
        for incomplete_line_num, incomplete_from, misplaced_import in incomplete_imports_to_fix:
            # Extract the module from the incomplete from statement
            from_match = re.match(r'from\s+([^\s]+)\s+import\s*\(', incomplete_from)
            if from_match:
                module = from_match.group(1)
                # Create proper import statements
                lines.insert(last_import_line, f"import {module}\n")
                last_import_line += 1
                lines.insert(last_import_line, misplaced_import + '\n')
                last_import_line += 1
        
        # Add blank line after imports if needed
        if last_import_line < len(lines) and lines[last_import_line].strip():
            lines.insert(last_import_line, '\n')
        
        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        return True
        
    except Exception as e:
        tprint(f"Error fixing {file_path}: {e}")
        return False


def main():
    """Main function to fix incorrect imports in all files."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix incorrectly placed imports")
    parser.add_argument("--project-root", default="/workspace/src", help="Project root directory")
    parser.add_argument("--file-pattern", default="**/*.py", help="File pattern to match")
    parser.add_argument("--fix", action="store_true", help="Actually fix the files")
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root)
    file_paths = list(project_root.glob(args.file_pattern))
    
    tprint(f"Analyzing {len(file_paths)} Python files...")
    
    files_with_issues = []
    for file_path in file_paths:
        incorrect_imports = find_incorrect_imports(str(file_path))
        if incorrect_imports:
            files_with_issues.append((str(file_path), incorrect_imports))
    
    tprint(f"\nFound {len(files_with_issues)} files with incorrect imports:")
    
    for file_path, imports in files_with_issues:
        tprint(f"\n{file_path}:")
        for line_num, import_stmt in imports:
            tprint(f"  Line {line_num + 1}: {import_stmt}")
    
    if args.fix and files_with_issues:
        tprint(f"\nFixing {len(files_with_issues)} files...")
        fixed = 0
        for file_path, _ in files_with_issues:
            if fix_incorrect_imports(file_path):
                fixed += 1
                tprint(f"✓ Fixed {file_path}")
        
        tprint(f"\nFixed {fixed} files")
    elif not args.fix:
        tprint(f"\nRun with --fix to actually fix the files")


if __name__ == "__main__":
    main()