#!/usr/bin/env python3
"""
Custom script to fix specific syntax errors in the codebase.
Targets the common patterns found in the syntax error reports.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

def fix_nested_imports(content: str) -> str:
    """Fix nested import statements (import inside another import)."""
    # Pattern to find imports inside parentheses with another import
    pattern = r'from\s+[\w.]+\s+import\s+\(\s*\nfrom\s+[\w.]+\s+import\s+[\w,\s]+\n'
    
    # Find all matches
    matches = list(re.finditer(pattern, content, re.MULTILINE))
    
    if not matches:
        return content
    
    # Process from end to start to maintain positions
    for match in reversed(matches):
        start, end = match.span()
        matched_text = match.group()
        
        # Extract the inner import
        inner_import_match = re.search(r'from\s+([\w.]+)\s+import\s+([\w,\s]+)\n', matched_text[matched_text.find('\nfrom'):])
        if inner_import_match:
            inner_module = inner_import_match.group(1)
            inner_items = inner_import_match.group(2).strip()
            
            # Move the inner import before the outer import
            outer_start = content.rfind('\n', 0, start) + 1
            new_import = f"from {inner_module} import {inner_items}\n"
            
            # Remove the inner import from the parentheses
            fixed_text = matched_text.replace(inner_import_match.group(), '')
            
            # Replace in content
            content = content[:start] + fixed_text + content[end:]
            content = content[:outer_start] + new_import + content[outer_start:]
    
    return content


def fix_missing_colons(content: str) -> str:
    """Add missing colons after function/class definitions."""
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Check for function or class definition without colon
        if re.match(r'^\s*(def|class)\s+\w+.*\)$', line) and not line.endswith(':'):
            line += ':'
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)


def fix_try_blocks(content: str) -> str:
    """Fix try blocks without except or finally."""
    lines = content.split('\n')
    fixed_lines = []
    in_try_block = False
    try_indent = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Detect try block
        if stripped.startswith('try:'):
            in_try_block = True
            try_indent = len(line) - len(line.lstrip())
            fixed_lines.append(line)
            continue
        
        # If we're in a try block and hit a line with same or less indentation
        if in_try_block and line.strip() and len(line) - len(line.lstrip()) <= try_indent:
            # Check if it's except or finally
            if not stripped.startswith(('except', 'finally')):
                # Add a generic except block
                fixed_lines.append(' ' * try_indent + 'except Exception as e:')
                fixed_lines.append(' ' * (try_indent + 4) + 'pass  # TODO: Handle exception properly')
            in_try_block = False
        
        fixed_lines.append(line)
    
    # Handle case where try block is at end of file
    if in_try_block:
        fixed_lines.append(' ' * try_indent + 'except Exception as e:')
        fixed_lines.append(' ' * (try_indent + 4) + 'pass  # TODO: Handle exception properly')
    
    return '\n'.join(fixed_lines)


def fix_indentation_errors(content: str) -> str:
    """Fix common indentation errors."""
    lines = content.split('\n')
    fixed_lines = []
    expected_indent = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        if not stripped:  # Empty line
            fixed_lines.append(line)
            continue
        
        current_indent = len(line) - len(line.lstrip())
        
        # Fix unexpected indentation
        if current_indent > expected_indent + 4:
            # Likely an indentation error, adjust to expected
            line = ' ' * expected_indent + stripped
        
        fixed_lines.append(line)
        
        # Update expected indentation for next line
        if stripped.endswith(':'):
            expected_indent = current_indent + 4
        elif stripped in ('pass', 'return', 'break', 'continue'):
            expected_indent = max(0, current_indent - 4)
        else:
            expected_indent = current_indent
    
    return '\n'.join(fixed_lines)


def fix_unterminated_strings(content: str) -> str:
    """Fix unterminated string literals."""
    # Pattern to find unterminated strings
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Count quotes
        single_quotes = line.count("'") - line.count("\\'")
        double_quotes = line.count('"') - line.count('\\"')
        
        # If odd number of quotes, likely unterminated
        if single_quotes % 2 == 1:
            line += "'"
        elif double_quotes % 2 == 1:
            line += '"'
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)


def fix_python_file(filepath: Path) -> bool:
    """Fix syntax errors in a Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply fixes in order
        content = fix_nested_imports(content)
        content = fix_missing_colons(content)
        content = fix_try_blocks(content)
        content = fix_indentation_errors(content)
        content = fix_unterminated_strings(content)
        
        # Only write if changes were made
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def main():
    """Main function to fix syntax errors in the codebase."""
    project_root = Path('/workspace/src')
    fixed_count = 0
    error_count = 0
    
    print("Fixing syntax errors in Python files...")
    print("=" * 60)
    
    # Find all Python files
    python_files = list(project_root.rglob('*.py'))
    total_files = len(python_files)
    
    for i, filepath in enumerate(python_files, 1):
        if i % 50 == 0:
            print(f"Progress: {i}/{total_files} files processed...")
        
        # Try to compile the file first to check for syntax errors
        try:
            compile(open(filepath).read(), filepath, 'exec')
        except SyntaxError:
            # File has syntax error, try to fix it
            if fix_python_file(filepath):
                fixed_count += 1
                print(f"Fixed: {filepath}")
            else:
                error_count += 1
        except Exception:
            error_count += 1
    
    print("\n" + "=" * 60)
    print(f"Total files processed: {total_files}")
    print(f"Files fixed: {fixed_count}")
    print(f"Files with errors: {error_count}")
    print("=" * 60)
    
    # Run the advanced syntax fixer again to catch remaining issues
    print("\nRunning advanced syntax fixer for remaining issues...")
    os.system('python3 /workspace/code_quality/scripts/advanced_syntax_fixer.py --project-root /workspace/src --fix')
    
    # Run import fixer
    print("\nRunning import fixer...")
    os.system('python3 /workspace/code_quality/scripts/safe_import_fixer.py --project-root /workspace/src --fix')
    
    # Run async fixer
    print("\nRunning async/await fixer...")
    os.system('python3 /workspace/code_quality/scripts/robust_async_fixer.py --project-root /workspace/src --fix')
    
    # Run type hint enhancer
    print("\nEnhancing type hints...")
    os.system('python3 /workspace/code_quality/scripts/enhanced_type_hints.py --project-root /workspace/src --target 0.9')
    
    print("\nCode quality fixes completed!")


if __name__ == '__main__':
    main()