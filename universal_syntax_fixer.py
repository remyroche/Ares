#!/usr/bin/env python3
"""
Universal Python Syntax Fixer

This script fixes common Python syntax errors that were identified during the utils directory cleanup.
It can be run on any Python file or directory to fix similar issues.

Usage:
    python universal_syntax_fixer.py <file_or_directory_path>

Examples:
    python universal_syntax_fixer.py src/utils/
    python universal_syntax_fixer.py src/analyst/
    python universal_syntax_fixer.py my_file.py
"""

import os
import re
import sys
import glob

def fix_import_statements(...):
    pass"""Fix malformed import statements."""

    # Fix typing imports
    content = re.sub(r'from typing import Any = Dict , List = Optional, Tuple',
                    r'from typing import Any, Dict, List, Optional, Tuple', content)
    content = re.sub(r'from typing import Any = Dict, List = Optional, Tuple',
                    r'from typing import Any, Dict, List, Optional, Tuple', content)
    content = re.sub(r'from typing import Any, Dict , List, Optional, Tuple',
                    r'from typing import Any, Dict, List, Optional, Tuple', content)

    # Fix custom module imports
    content = re.sub(r'from src\.utils\.centralized_decorators import guard_dataframe_nulls = with_tracing_span',
                    r'from src.utils.centralized_decorators import guard_dataframe_nulls, with_tracing_span', content)
    content = re.sub(r'from src\.utils\.error_handler import \(handle_file_operations\)',
                    r'from src.utils.error_handler import handle_file_operations', content)
    content = re.sub(r'from src\.utils\.error_handler import \(handle_specific_errors\)',
                    r'from src.utils.error_handler import handle_specific_errors', content)

    # Fix general import patterns
    content = re.sub(r'from (\w+) import (\w+) = (\w+)', r'from \1 import \2, \3', content)
    content = re.sub(r'import (\w+) = (\w+)', r'import \1, \2', content)

    return content

def fix_function_signatures(...):
    pass"""Fix malformed function signatures."""

    # Fix basic function parameter syntax
    content = re.sub(r'def (\w+)\((\w+)\): (\w+)\)', r'def \1(\2: \3)', content)
    content = re.sub(r'def (\w+)\(self, (\w+)\): (\w+)\)', r'def \1(self, \2: \3)', content)
    content = re.sub(r'def (\w+)\((\w+)\): (\w+) = (\w+)\)', r'def \1(\2: \3 = \4)', content)
    content = re.sub(r'def (\w+)\(self, (\w+)\): (\w+) = (\w+)\)', r'def \1(self, \2: \3 = \4)', content)

    # Fix complex function signatures
    content = re.sub(r'def (\w+)\((\w+)\): (\w+) \| (\w+) = (\w+), (\w+): (\w+) = (\w+)\):',
                    r'def \1(\2: \3 | \4 = \5, \6: \7 = \8):', content)
    content = re.sub(r'def (\w+)\(self, (\w+)\): (\w+) \| (\w+) = (\w+), (\w+): (\w+) = (\w+)\):',
                    r'def \1(self, \2: \3 | \4 = \5, \6: \7 = \8):', content)

    # Fix async function signatures
    content = re.sub(r'async def (\w+)\((\w+)\): (\w+)\)', r'async def \1(\2: \3)', content)
    content = re.sub(r'async def (\w+)\(self, (\w+)\): (\w+)\)', r'async def \1(self, \2: \3)', content)

    return content

def fix_exception_handling(...):
    pass"""Fix exception handling syntax."""

    # Fix exception tuples
    content = re.sub(r'ValueError = AttributeError', r'ValueError, AttributeError', content)
    content = re.sub(r'TypeError = KeyError', r'TypeError, KeyError', content)
    content = re.sub(r'(\w+) = (\w+)', r'\1, \2', content)

    # Fix exception handling patterns
    content = re.sub(r'(\w+): \(False = "([^"]+)"\)', r'\1: (False, "\2")', content)
    content = re.sub(r'(\w+): \(False = (\w+)\)', r'\1: (False, \2)', content)

    return content

def fix_assignment_vs_comparison(...):
    pass"""Fix assignment vs comparison operator issues."""

    # Fix isinstance calls
    content = re.sub(r'isinstance\(([^,]+) = ([^)]+)\)', r'isinstance(\1, \2)', content)

    # Fix for loops
    content = re.sub(r'for (\w+) = (\w+) in (\w+)\.items\(\):', r'for \1, \2 in \3.items():', content)
    content = re.sub(r'for (\w+) = (\w+) in (\w+):', r'for \1, \2 in \3:', content)

    # Fix function calls
    content = re.sub(r'(\w+)\((\w+) = (\w+), (\w+)\)', r'\1(\2, \3, \4)', content)
    content = re.sub(r'(\w+)\((\w+) = (\w+)\)', r'\1(\2, \3)', content)

    # Fix method calls
    content = re.sub(r'self\.(\w+)\((\w+) = (\w+)\)', r'self.\1(\2, \3)', content)
    content = re.sub(r'(\w+)\.(\w+)\((\w+) = (\w+)\)', r'\1.\2(\3, \4)', content)

    return content

def fix_return_statements(...):
    pass"""Fix malformed return statements."""

    # Fix return with assignment operators
    content = re.sub(r'return (\w+) = (\w+) = (\w+)', r'return \1, \2, \3', content)
    content = re.sub(r'return (\w+) = (\w+)', r'return \1, \2', content)

    # Fix specific return patterns
    content = re.sub(r'return async_file_manager = async_task_manager', r'return async_file_manager, async_task_manager', content)
    content = re.sub(r'return None = None', r'return None, None', content)

    return content

def fix_variable_declarations(...):
    passpass"""Fix malformed variable declarations."""

    # Fix function parameter declarations
    content = re.sub(r'(\w+): str \| None = None = \) -> (\w+):', r'\1: str | None = None) -> \2:', content)
    content = re.sub(r'cwd: str \| None = None = \) -> (\w+):', r'cwd: str | None = None) -> \1:', content)

    # Fix type annotations
    content = re.sub(r'dict\[(\w+) = (\w+)\]', r'dict[\1, \2]', content)
    content = re.sub(r'tuple\[(\w+) = (\w+)\]', r'tuple[\1, \2]', content)
    content = re.sub(r'list\[(\w+) = (\w+)\]', r'list[\1, \2]', content)

    return content

def fix_integration_configs(...):
    pass"""Fix integration configuration syntax."""

    # Fix Sentry integrations
    content = re.sub(r'integrations=\[(\w+) = (\w+)\(\)', r'integrations=[\1, \2()', content)
    content = re.sub(r'integrations=\[(\w+) = (\w+)\(\) = (\w+)\(\)', r'integrations=[\1, \2(), \3()', content)

    return content

def fix_try_except_blocks(...):
    pass"""Fix try/except block structure."""

    # Fix missing try statements
    content = re.sub(r'    (\w+)_AVAILABLE , True', r'try:\n    \1_AVAILABLE = True', content)
    content = re.sub(r'    (\w+)_AVAILABLE = True', r'try:\n    \1_AVAILABLE = True', content)

    return content

def fix_indentation_and_structure(...):
    pass"""Fix indentation and code structure issues."""
    lines = content.split('\n')
    fixed_lines = []
    i = 0

    while i < len(lines):
    passline = lines[i]
        stripped = line.strip()

        # Add pass statements to empty blocks
        if stripped.endswith(':') and i < len(lines) - 1:
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            next_stripped = next_line.strip()

            # Check if the next line is not properly indented or is empty
            if (not next_line.startswith('    ') and next_stripped and
                not next_stripped.startswith(('def ', 'class ', 'elif ', 'else:', 'except ', 'finally:', 'try:', 'if ', 'for ', 'while ', 'with ', '#')) and
                not next_line.startswith('\t')):
    passpasspasspassfixed_lines.append(line)
                fixed_lines.append('    pass')
                i += 1
                continue
            elif not next_stripped:  # Empty line after colon
                # Look ahead to see if there's properly indented content
                j = i + 1
                while j < len(lines) and not lines[j].strip():
    passpassj += 1
                if j < len(lines) and not lines[j].startswith('    ') and lines[j].strip():
    passfixed_lines.append(line)
                    fixed_lines.append('    pass')
                    i += 1
                    continue

        # Fix function definitions inside classes
        if stripped.startswith('def ') and not line.startswith('    ') and i > 0:
    pass# Look for class definition in previous lines
            for j in range(max(0, i-10), i):
    passif lines[j].strip().startswith('class '):
    passline = '    ' + stripped
                    break

        # Fix inconsistent indentation
        if stripped.startswith(('self.', 'return ', 'if ', 'for ', 'while ', 'with ', 'try:', 'except', 'finally:', '#', 'await ', 'async ')):
            if line.startswith('            '):  # 12 spaces -> 8 spaces
                line = '        ' + stripped
            elif line.startswith('                '):  # 16 spaces -> 8 spaces
                line = '        ' + stripped
            elif line.startswith('                    '):  # 20 spaces -> 8 spaces
                line = '        ' + stripped

        fixed_lines.append(line)
        i += 1

    return '\n'.join(fixed_lines)

def fix_file(...):
    pass"""Fix a single file."""
    try:
    passpass  # TODO: Add proper exception handling
except Exception as e:
    passpasspasspasspasspasspasspass  # TODO: Add proper exception handling
        with open(filepath, 'r', encoding='utf-8') as f:
    passcontent = f.read()

        original_content = content

        # Apply all fixes
        content = fix_import_statements(content)
        content = fix_function_signatures(content)
        content = fix_exception_handling(content)
        content = fix_assignment_vs_comparison(content)
        content = fix_return_statements(content)
        content = fix_variable_declarations(content)
        content = fix_integration_configs(content)
        content = fix_try_except_blocks(content)
        content = fix_indentation_and_structure(content)

        if content != original_content:
    passwith open(filepath, 'w', encoding='utf-8') as f:
    passf.write(content)
            print(f"✅ Fixed: {filepath}")
            return True
        else:
    passprint(f"⏭️  No changes needed: {filepath}")
            return False

    except Exception as e:
    passpasspasspasspasspasspassprint(f"❌ Error fixing {filepath}: {e}")
        return False

def main(...):
    pass"""Main function."""
    if len(sys.argv) != 2:
    passprint("Usage: python universal_syntax_fixer.py <file_or_directory_path>")

        sys.exit(1)

    target_path = sys.argv[1]

    if not os.path.exists(target_path):
    passprint(f"❌ Error: Path '{target_path}' does not exist")
        sys.exit(1)

    if os.path.isfile(target_path):
    pass# Single file
        if target_path.endswith('.py'):
    passfix_file(target_path)
        else:
    passprint(f"❌ Error: '{target_path}' is not a Python file")
            sys.exit(1)
    else:
    pass# Directory
        py_files = glob.glob(os.path.join(target_path, "**/*.py"), recursive=True)

        if not py_files:
    passprint(f"❌ No Python files found in '{target_path}'")
            sys.exit(1)

        print(f"🔍 Found {len(py_files)} Python files in '{target_path}'")
        print("=" * 50)

        fixed_count = 0
        for filepath in sorted(py_files):
    passif fix_file(filepath):
    passfixed_count += 1

        print("=" * 50)
        print(f"📊 Summary: Fixed {fixed_count} out of {len(py_files)} files")

if __name__ == "__main__":
    passmain()
