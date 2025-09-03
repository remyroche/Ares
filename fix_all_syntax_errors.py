#!/usr/bin/env python3
"""Comprehensive syntax error fixer."""

import re
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List


def find_all_syntax_errors(file_path: Path) -> List[Tuple[int, str]]:
    """Find all syntax errors in a file by repeatedly running the parser."""
    errors = []
    max_iterations = 10  # Prevent infinite loops
    
    for _ in range(max_iterations):
        result = subprocess.run(
            ['python3', '-m', 'py_compile', str(file_path)],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            break
            
        # Parse error message
        match = re.search(r'File ".*", line (\d+)', result.stderr)
        if match:
            line_no = int(match.group(1))
            errors.append((line_no, result.stderr))
        else:
            break
            
    return errors


def fix_missing_parentheses(file_path: Path) -> int:
    """Fix missing closing parentheses in function calls."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_count = 0
    i = 0
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Look for patterns like 'self.logger.error(' without matching ')'
        if re.search(r'\.(error|warning|info|debug|critical)\s*\($', stripped):
            # Count parentheses in this and following lines
            paren_count = line.count('(') - line.count(')')
            j = i + 1
            
            while j < len(lines) and paren_count > 0:
                paren_count += lines[j].count('(') - lines[j].count(')')
                j += 1
            
            # If unmatched, add closing paren
            if paren_count > 0 and j > i + 1:
                # Add closing paren to the line before the next statement
                insert_line = j - 1
                while insert_line > i and lines[insert_line].strip() == '':
                    insert_line -= 1
                    
                if not lines[insert_line].rstrip().endswith(')'):
                    lines[insert_line] = lines[insert_line].rstrip() + ')\n'
                    fixed_count += 1
        
        i += 1
    
    if fixed_count > 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    
    return fixed_count


def fix_unclosed_function_calls(file_path: Path) -> int:
    """Fix unclosed function calls and method definitions."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    fixed_count = 0
    
    # Track open parentheses
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check for function/method calls that span multiple lines
        if '(' in line and ')' not in line:
            # Count parentheses
            open_count = line.count('(')
            close_count = line.count(')')
            j = i + 1
            
            # Find the matching closing parenthesis
            while j < len(lines) and open_count > close_count:
                open_count += lines[j].count('(')
                close_count += lines[j].count(')')
                j += 1
            
            # If we didn't find enough closing parens
            if open_count > close_count:
                # Look for the next line that starts a new statement
                for k in range(j, min(j + 10, len(lines))):
                    if lines[k].strip() and not lines[k].startswith(' '):
                        # Insert closing paren before this line
                        if k > 0 and not lines[k-1].rstrip().endswith(')'):
                            lines[k-1] = lines[k-1].rstrip() + ')\n'
                            fixed_count += 1
                        break
        
        i += 1
    
    if fixed_count > 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    return fixed_count


def fix_file_comprehensively(file_path: Path) -> bool:
    """Apply all fixes to a file."""
    print(f"Fixing {file_path}...")
    
    # Get initial errors
    errors = find_all_syntax_errors(file_path)
    if not errors:
        return True
    
    print(f"  Found {len(errors)} syntax errors")
    
    # Apply fixes
    total_fixes = 0
    
    # Fix missing parentheses
    fixes = fix_missing_parentheses(file_path)
    if fixes > 0:
        print(f"  Fixed {fixes} missing parentheses")
        total_fixes += fixes
    
    # Fix unclosed function calls
    fixes = fix_unclosed_function_calls(file_path)
    if fixes > 0:
        print(f"  Fixed {fixes} unclosed function calls")
        total_fixes += fixes
    
    # Check if fixed
    remaining_errors = find_all_syntax_errors(file_path)
    if not remaining_errors:
        print(f"  ✓ All syntax errors fixed!")
        return True
    else:
        print(f"  ✗ {len(remaining_errors)} errors remain")
        return False


def main():
    """Main function."""
    # Get list of files with syntax errors
    src_dir = Path('/workspace/src')
    error_files = []
    
    print("Scanning for files with syntax errors...")
    for py_file in src_dir.rglob('*.py'):
        result = subprocess.run(
            ['python3', '-m', 'py_compile', str(py_file)],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            error_files.append(py_file)
    
    print(f"\nFound {len(error_files)} files with syntax errors")
    
    # Focus on specific problematic files first
    priority_files = [
        '/workspace/src/launcher/enhanced_trading_launcher.py',
        '/workspace/src/interfaces/enhanced_event_bus.py',
        '/workspace/src/tactician/position_sizer.py',
    ]
    
    fixed_count = 0
    
    # Fix priority files first
    for file_path in priority_files:
        if Path(file_path).exists() and Path(file_path) in error_files:
            if fix_file_comprehensively(Path(file_path)):
                fixed_count += 1
    
    print(f"\nFixed {fixed_count} files")


if __name__ == "__main__":
    main()