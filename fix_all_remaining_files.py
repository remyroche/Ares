#!/usr/bin/env python3
"""
Comprehensive script to fix all remaining syntax issues in src/utils/ files
"""

import os
import re
import glob

def fix_all_syntax_issues(content):
    """Fix all syntax issues."""
    
    # Fix import statements
    content = re.sub(r'from typing import Any = Dict , List = Optional, Tuple', 
                    r'from typing import Any, Dict, List, Optional, Tuple', content)
    content = re.sub(r'from src\.utils\.centralized_decorators import guard_dataframe_nulls = with_tracing_span', 
                    r'from src.utils.centralized_decorators import guard_dataframe_nulls, with_tracing_span', content)
    
    # Fix basic function parameter syntax (simplified)
    content = re.sub(r'def (\w+)\((\w+)\): (\w+)\)', r'def \1(\2: \3)', content)
    content = re.sub(r'def (\w+)\(self, (\w+)\): (\w+)\)', r'def \1(self, \2: \3)', content)
    
    # Fix specific patterns
    content = re.sub(r'ValueError = AttributeError', r'ValueError, AttributeError', content)
    content = re.sub(r'isinstance\(([^,]+) = ([^)]+)\)', r'isinstance(\1, \2)', content)
    content = re.sub(r'integrations=\[(\w+) = (\w+)\(\)', r'integrations=[\1, \2()', content)
    
    # Fix assignment patterns
    content = re.sub(r'(\w+) = (\w+) = (\w+)', r'\1 = \2, \3', content)
    content = re.sub(r'return (\w+) = (\w+) = (\w+)', r'return \1, \2, \3', content)
    
    # Fix specific function call patterns
    content = re.sub(r'(\w+)\((\w+) = (\w+), (\w+)\)', r'\1(\2, \3, \4)', content)
    content = re.sub(r'self\._add_to_cache\((\w+) = (\w+)\)', r'self._add_to_cache(\1, \2)', content)
    content = re.sub(r'async with aiofiles\.open\((\w+) = "(\w+)", (\w+)=(\w+)\)', r'async with aiofiles.open(\1, "\2", encoding=\4)', content)
    
    # Fix async patterns
    content = re.sub(r'async with aiofiles\.open\((\w+), (\w+), (\w+)\) as f:', r'async with aiofiles.open(\1, \2, encoding=\3) as f:', content)
    
    # Fix return patterns
    content = re.sub(r'return async_file_manager = async_task_manager', r'return async_file_manager, async_task_manager', content)
    content = re.sub(r'return None = None', r'return None, None', content)
    
    # Fix variable declarations
    content = re.sub(r'(\w+): str \| None = None = \) -> (\w+):', r'\1: str | None = None) -> \2:', content)
    content = re.sub(r'cwd: str \| None = None = \) -> (\w+):', r'cwd: str | None = None) -> \1:', content)
    
    # Fix try/except block issues
    content = re.sub(r'    PYARROW_AVAILABLE , True', r'try:\n    PYARROW_AVAILABLE = True', content)
    
    return content

def fix_indentation_and_structure(content):
    """Fix indentation and code structure issues."""
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Fix empty blocks by adding pass
        if i < len(lines) - 1:
            next_line = lines[i + 1]
            if (stripped.endswith(':') and 
                next_line.strip() and 
                not next_line.startswith('    ') and 
                not next_line.startswith('\t') and
                not next_line.strip().startswith(('def ', 'class ', 'elif ', 'else:', 'except ', 'finally:', '#', 'try:', 'if ', 'for ', 'while ', 'with '))):
                # Insert pass statement
                fixed_lines.append(line)
                fixed_lines.append('    pass')
                i += 1
                continue
        
        # Fix function definitions that are not properly indented
        if stripped.startswith('def ') and not line.startswith('    ') and i > 0:
            prev_lines = [lines[j].strip() for j in range(max(0, i-3), i)]
            if any(l.startswith('class ') for l in prev_lines):
                line = '    ' + stripped
        
        # Fix inconsistent indentation in function bodies
        if stripped.startswith(('self.', 'return ', 'if ', 'for ', 'while ', 'with ', 'try:', 'except', 'finally:', '#')):
            if line.startswith('            '):  # 12 spaces -> 8 spaces
                line = '        ' + stripped
            elif line.startswith('                    '):  # 20 spaces -> 8 spaces
                line = '        ' + stripped
        
        # Fix await statements
        if stripped.startswith('await ') and line.startswith('            '):
            line = '        ' + stripped
        
        fixed_lines.append(line)
        i += 1
    
    return '\n'.join(fixed_lines)

def fix_file(filepath):
    """Fix a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        content = fix_all_syntax_issues(content)
        content = fix_indentation_and_structure(content)
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {filepath}")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False

def main():
    """Main function."""
    utils_dir = "src/utils"
    py_files = glob.glob(os.path.join(utils_dir, "*.py"))
    
    fixed_count = 0
    for filepath in sorted(py_files):
        if fix_file(filepath):
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} files")

if __name__ == "__main__":
    main()
