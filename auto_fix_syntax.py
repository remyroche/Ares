#!/usr/bin/env python3
"""Automatically fix common syntax errors in Python files."""

import ast
import json
import re
from pathlib import Path
from typing import Optional, Tuple


def detect_syntax_error(file_path: Path) -> Optional[Tuple[str, int, str]]:
    """Detect syntax error in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        return None
    except SyntaxError as e:
        return (str(file_path), e.lineno or 0, e.msg or "Unknown error")
    except Exception as e:
        return (str(file_path), 0, str(e))


def fix_missing_except_finally(content: str, error_line: int) -> Optional[str]:
    """Fix missing except or finally block."""
    lines = content.split('\n')
    
    # Find the try block before the error line
    try_line = -1
    indent_level = 0
    
    for i in range(error_line - 1, -1, -1):
        line = lines[i]
        stripped = line.lstrip()
        if stripped.startswith('try:'):
            try_line = i
            indent_level = len(line) - len(stripped)
            break
    
    if try_line == -1:
        return None
    
    # Check if there's already an except or finally
    has_except_finally = False
    for i in range(try_line + 1, min(len(lines), error_line + 10)):
        line = lines[i]
        stripped = line.lstrip()
        line_indent = len(line) - len(stripped)
        
        if line_indent == indent_level and (stripped.startswith('except') or stripped.startswith('finally')):
            has_except_finally = True
            break
    
    if not has_except_finally:
        # Add a generic except block
        indent = ' ' * indent_level
        lines.insert(error_line, f"{indent}except Exception as e:")
        lines.insert(error_line + 1, f"{indent}    pass  # TODO: Handle exception properly")
    
    return '\n'.join(lines)


def fix_unterminated_string(content: str, error_line: int) -> Optional[str]:
    """Fix unterminated string literal."""
    lines = content.split('\n')
    if error_line <= 0 or error_line > len(lines):
        return None
    
    line = lines[error_line - 1]
    
    # Count quotes
    single_quotes = line.count("'") - line.count("\\'")
    double_quotes = line.count('"') - line.count('\\"')
    
    # Add missing quote at the end
    if single_quotes % 2 == 1:
        lines[error_line - 1] = line.rstrip() + "'"
    elif double_quotes % 2 == 1:
        lines[error_line - 1] = line.rstrip() + '"'
    
    return '\n'.join(lines)


def fix_missing_indent_after_colon(content: str, error_line: int) -> Optional[str]:
    """Fix missing indented block after if/try/for/while/def/class statements."""
    lines = content.split('\n')
    
    # Find the statement with colon before error line
    colon_line = -1
    indent_level = 0
    
    for i in range(max(0, error_line - 5), error_line):
        if i < len(lines):
            line = lines[i]
            stripped = line.lstrip()
            if stripped.endswith(':') and any(stripped.startswith(kw) for kw in ['if', 'try', 'for', 'while', 'def', 'class', 'except', 'finally', 'elif', 'else']):
                colon_line = i
                indent_level = len(line) - len(stripped)
                break
    
    if colon_line == -1:
        return None
    
    # Check if next line is properly indented
    next_line = colon_line + 1
    if next_line < len(lines):
        next_line_content = lines[next_line]
        if next_line_content.strip() == '' or (len(next_line_content) - len(next_line_content.lstrip())) <= indent_level:
            # Add pass statement
            indent = ' ' * (indent_level + 4)
            lines.insert(next_line, f"{indent}pass  # TODO: Implement")
    
    return '\n'.join(lines)


def fix_unexpected_indent(content: str, error_line: int) -> Optional[str]:
    """Fix unexpected indent by aligning with previous line."""
    lines = content.split('\n')
    if error_line <= 1 or error_line > len(lines):
        return None
    
    # Get the problematic line
    problem_line = lines[error_line - 1]
    problem_indent = len(problem_line) - len(problem_line.lstrip())
    
    # Find previous non-empty line
    prev_indent = 0
    for i in range(error_line - 2, -1, -1):
        if lines[i].strip():
            prev_indent = len(lines[i]) - len(lines[i].lstrip())
            break
    
    # Adjust indent
    lines[error_line - 1] = ' ' * prev_indent + problem_line.lstrip()
    
    return '\n'.join(lines)


def fix_file(file_path: Path) -> bool:
    """Fix syntax errors in a single file."""
    error_info = detect_syntax_error(file_path)
    if not error_info:
        return True
    
    _, line_no, error_msg = error_info
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        fixed_content = None
        
        if "expected 'except' or 'finally' block" in error_msg:
            fixed_content = fix_missing_except_finally(content, line_no)
        elif "unterminated string literal" in error_msg:
            fixed_content = fix_unterminated_string(content, line_no)
        elif "expected an indented block" in error_msg:
            fixed_content = fix_missing_indent_after_colon(content, line_no)
        elif "unexpected indent" in error_msg:
            fixed_content = fix_unexpected_indent(content, line_no)
        
        if fixed_content and fixed_content != content:
            # Verify the fix
            try:
                ast.parse(fixed_content)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                print(f"✓ Fixed {file_path}: {error_msg}")
                return True
            except SyntaxError:
                print(f"✗ Fix failed for {file_path}: {error_msg}")
                return False
        else:
            print(f"✗ Could not fix {file_path}: {error_msg}")
            return False
            
    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}")
        return False


def main():
    """Main function to fix all syntax errors."""
    # Load the list of files with errors
    with open('/workspace/syntax_errors_detailed.json', 'r') as f:
        errors = json.load(f)
    
    print(f"Processing {len(errors)} files with syntax errors...\n")
    
    fixed_count = 0
    failed_count = 0
    
    for error in errors:
        file_path = Path(error['file'])
        if file_path.exists():
            if fix_file(file_path):
                fixed_count += 1
            else:
                failed_count += 1
    
    print(f"\nSummary:")
    print(f"  Fixed: {fixed_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Total: {len(errors)}")


if __name__ == "__main__":
    main()