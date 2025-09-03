#!/usr/bin/env python3
"""Advanced syntax error fixer for Python files."""

import ast
import json
import re
from pathlib import Path
from typing import Optional, Tuple, List
import subprocess


def get_syntax_error_details(file_path: Path) -> Optional[dict]:
    """Get detailed syntax error information using Python's parser."""
    try:
        result = subprocess.run(
            ['python3', '-m', 'py_compile', str(file_path)],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            # Parse error message
            error_msg = result.stderr
            
            # Extract line number and error type
            match = re.search(r'File ".*", line (\d+)', error_msg)
            if match:
                line_no = int(match.group(1))
                
                # Extract error message
                if "expected 'except' or 'finally' block" in error_msg:
                    return {"line": line_no, "type": "missing_except_finally"}
                elif "unterminated string literal" in error_msg:
                    return {"line": line_no, "type": "unterminated_string"}
                elif "expected an indented block" in error_msg:
                    return {"line": line_no, "type": "missing_indent"}
                elif "unexpected indent" in error_msg:
                    return {"line": line_no, "type": "unexpected_indent"}
                elif "unmatched ')'" in error_msg:
                    return {"line": line_no, "type": "unmatched_paren"}
                elif "'(' was never closed" in error_msg:
                    return {"line": line_no, "type": "unclosed_paren"}
                elif "invalid syntax" in error_msg:
                    return {"line": line_no, "type": "invalid_syntax"}
                else:
                    return {"line": line_no, "type": "other", "msg": error_msg}
                    
        return None
        
    except Exception as e:
        return {"line": 0, "type": "error", "msg": str(e)}


def fix_missing_except_finally_advanced(file_path: Path, error_line: int) -> bool:
    """Fix missing except/finally blocks with better heuristics."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Find all try blocks
        try_blocks = []
        for i, line in enumerate(lines):
            if re.match(r'^\s*try:\s*$', line):
                indent = len(line) - len(line.lstrip())
                try_blocks.append((i, indent))
        
        # Find which try block needs fixing
        for try_idx, try_indent in try_blocks:
            # Check if this try has except/finally
            has_handler = False
            
            for j in range(try_idx + 1, len(lines)):
                line = lines[j]
                if line.strip() == '':
                    continue
                    
                line_indent = len(line) - len(line.lstrip())
                
                # If we hit something at same or less indent, we're done with this try
                if line_indent <= try_indent and line.strip():
                    break
                    
                # Check for except/finally at correct indent
                if line_indent == try_indent and re.match(r'^\s*(except|finally)', line):
                    has_handler = True
                    break
            
            # If no handler and this is near our error line, fix it
            if not has_handler and abs(try_idx - error_line) < 50:
                # Find where to insert except
                insert_idx = try_idx + 1
                
                # Skip over the try block content
                for j in range(try_idx + 1, len(lines)):
                    if j >= len(lines):
                        break
                    line = lines[j]
                    line_indent = len(line) - len(line.lstrip())
                    
                    # Stop at same or less indent
                    if line.strip() and line_indent <= try_indent:
                        insert_idx = j
                        break
                    insert_idx = j + 1
                
                # Insert except block
                indent_str = ' ' * try_indent
                lines.insert(insert_idx, f"{indent_str}except Exception as e:\n")
                lines.insert(insert_idx + 1, f"{indent_str}    pass  # TODO: Handle exception\n")
                
                # Write back
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                
                return True
                
        return False
        
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False


def fix_invalid_syntax_imports(file_path: Path) -> bool:
    """Fix invalid syntax caused by imports in wrong places."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        # Find imports that are not at the beginning or in wrong places
        import_lines = []
        fixed_lines = []
        
        in_function = False
        function_indent = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Track if we're in a function/method
            if re.match(r'^(\s*)def\s+\w+.*:\s*$', line) or re.match(r'^(\s*)class\s+\w+.*:\s*$', line):
                in_function = True
                function_indent = len(line) - len(line.lstrip())
            elif in_function and line.strip() and (len(line) - len(line.lstrip())) <= function_indent:
                in_function = False
            
            # Check for import statements
            if (stripped.startswith('import ') or stripped.startswith('from ')) and not stripped.startswith('from __future__'):
                # If it's inside a function/try block and not properly indented, it's wrong
                if in_function or (i > 0 and i < len(lines) - 1 and not lines[i-1].strip() and not lines[i+1].strip()):
                    # This import is likely in the wrong place
                    if line.strip():  # Don't add empty lines
                        import_lines.append(line.strip())
                    continue
            
            fixed_lines.append(line)
        
        # If we found misplaced imports, move them to the top
        if import_lines:
            # Find where to insert (after initial imports)
            insert_idx = 0
            for i, line in enumerate(fixed_lines):
                if line.strip() and not (line.strip().startswith('import ') or 
                                       line.strip().startswith('from ') or 
                                       line.strip().startswith('#') or
                                       line.strip().startswith('"""') or
                                       line.strip().startswith("'''")):
                    insert_idx = i
                    break
            
            # Insert the imports
            for imp in reversed(import_lines):
                fixed_lines.insert(insert_idx, imp)
            
            # Write back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(fixed_lines))
            
            return True
            
        return False
        
    except Exception as e:
        print(f"Error fixing imports in {file_path}: {e}")
        return False


def fix_file_advanced(file_path: Path) -> bool:
    """Fix a file using advanced techniques."""
    error_info = get_syntax_error_details(file_path)
    
    if not error_info:
        return True
    
    error_type = error_info.get("type")
    error_line = error_info.get("line", 0)
    
    print(f"Fixing {file_path} - {error_type} at line {error_line}")
    
    if error_type == "missing_except_finally":
        if fix_missing_except_finally_advanced(file_path, error_line):
            # Verify fix
            if get_syntax_error_details(file_path) is None:
                print(f"✓ Fixed {file_path}")
                return True
    
    # Try fixing import issues for any syntax error
    if fix_invalid_syntax_imports(file_path):
        # Verify fix
        if get_syntax_error_details(file_path) is None:
            print(f"✓ Fixed {file_path} (import issues)")
            return True
    
    # If still not fixed, try the specific error type
    if error_type == "invalid_syntax":
        # Additional invalid syntax fixes can go here
        pass
    
    return False


def main():
    """Main function to fix syntax errors."""
    # Get fresh list of files with errors
    print("Scanning for files with syntax errors...")
    
    src_dir = Path('/workspace/src')
    error_files = []
    
    for py_file in src_dir.rglob('*.py'):
        if get_syntax_error_details(py_file):
            error_files.append(py_file)
    
    print(f"\nFound {len(error_files)} files with syntax errors")
    
    fixed_count = 0
    for file_path in error_files:
        if fix_file_advanced(file_path):
            fixed_count += 1
    
    print(f"\nSummary:")
    print(f"  Fixed: {fixed_count}")
    print(f"  Remaining: {len(error_files) - fixed_count}")


if __name__ == "__main__":
    main()