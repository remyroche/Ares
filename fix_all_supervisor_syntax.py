#!/usr/bin/env python3
"""
Comprehensive script to fix all syntax and indentation issues in supervisor files.
"""

import os
import re
import glob
import subprocess

def fix_import_statements(content):
    """Fix malformed import statements with extra commas and parentheses."""
    # Fix the common import pattern with extra commas and unmatched parentheses
    content = re.sub(r'from src\.utils\.supervisor_error_handler import \(([^)]+)\)\)', 
                     r'from src.utils.supervisor_error_handler import (\1)', content)
    
    # Fix individual imports with extra commas
    content = re.sub(r',,', ',', content)
    content = re.sub(r',\s*\)', ')', content)
    
    # Fix unmatched parentheses at the end of import statements
    content = re.sub(r'\)\s*\n\s*\)', ')', content)
    content = re.sub(r'\)\s*\n\s*\)\s*\n', ')\n', content)
    
    return content

def fix_module_docstrings(content):
    """Fix module-level docstring indentation."""
    # Fix module docstrings that are over-indented
    content = re.sub(r'^\s{8,}"""', '"""', content, flags=re.MULTILINE)
    content = re.sub(r'^\s{8,}"""\s*\n\s{8,}', '"""\n', content, flags=re.MULTILINE)
    
    return content

def fix_class_docstrings(content):
    """Fix class docstring indentation."""
    # Fix class docstrings that are over-indented
    content = re.sub(r'class \w+:\s*\n\s{8,}"""', r'class \g<0>:\n    """', content)
    content = re.sub(r'class \w+:\s*\n\s{4,}"""\s*\n\s{8,}', r'class \g<0>:\n    """\n    ', content)
    
    return content

def fix_method_docstrings(content):
    """Fix method docstring indentation."""
    # Fix method docstrings that are over-indented
    content = re.sub(r'def \w+\([^)]*\):\s*\n\s{8,}"""', r'def \g<0>:\n        """', content)
    content = re.sub(r'def \w+\([^)]*\):\s*\n\s{4,}"""\s*\n\s{8,}', r'def \g<0>:\n        """\n        ', content)
    
    return content

def fix_decorators(content):
    """Fix decorator indentation."""
    # Fix decorators that are over-indented
    content = re.sub(r'^\s{8,}@', '    @', content, flags=re.MULTILINE)
    
    return content

def fix_method_definitions(content):
    """Fix method definition indentation."""
    # Fix method definitions that are over-indented
    content = re.sub(r'^\s{8,}(async )?def ', r'    \1def ', content, flags=re.MULTILINE)
    
    return content

def fix_class_attributes(content):
    """Fix class attribute indentation."""
    # Fix class attributes that are over-indented
    content = re.sub(r'^\s{8,}self\.\w+:', r'        self.\g<0>:', content, flags=re.MULTILINE)
    
    return content

def fix_try_except_blocks(content):
    """Fix malformed try-except blocks."""
    # Fix the pattern: try:\n    pass  # TODO: Add proper exception handling\n    except Exception as e:\n    pass  # TODO: Add proper exception handling
    pattern = r'try:\s*\n\s*pass\s*# TODO: Add proper exception handling\s*\nexcept Exception as e:\s*\n\s*pass\s*# TODO: Add proper exception handling\s*\n'
    replacement = 'try:\n'
    content = re.sub(pattern, replacement, content)
    
    return content

def fix_logger_calls(content):
    """Fix malformed logger calls with extra function calls."""
    # Fix pattern: self.logger.warning(warning(...))
    content = re.sub(r'self\.logger\.warning\(warning\(', 'self.logger.warning(', content)
    content = re.sub(r'self\.logger\.error\(error\(', 'self.logger.error(', content)
    content = re.sub(r'self\.logger\.info\(info\(', 'self.logger.info(', content)
    content = re.sub(r'self\.logger\.debug\(debug\(', 'self.logger.debug(', content)
    
    return content

def fix_indentation_issues(content):
    """Fix general indentation issues."""
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Fix class attribute indentation
        if re.match(r'^self\.\w+:', line):
            if not line.startswith('        '):
                line = '        ' + line.lstrip()
        
        # Fix method decorator indentation
        if line.strip().startswith('@'):
            if not line.startswith('    '):
                line = '    ' + line.lstrip()
        
        # Fix method definition indentation
        if re.match(r'^(async )?def \w+\(', line.strip()):
            if not line.startswith('    '):
                line = '    ' + line.lstrip()
        
        # Fix docstring indentation
        if line.strip().startswith('"""') or line.strip().startswith("'''"):
            if not line.startswith('        '):
                line = '        ' + line.lstrip()
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_supervisor_file(file_path):
    """Fix syntax errors in a supervisor file."""
    print(f"Fixing {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Apply fixes
        content = fix_import_statements(content)
        content = fix_module_docstrings(content)
        content = fix_class_docstrings(content)
        content = fix_method_docstrings(content)
        content = fix_decorators(content)
        content = fix_method_definitions(content)
        content = fix_class_attributes(content)
        content = fix_try_except_blocks(content)
        content = fix_logger_calls(content)
        content = fix_indentation_issues(content)
        
        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Test compilation
        result = subprocess.run(['python3', '-m', 'py_compile', file_path],
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {file_path} - Fixed successfully")
            return True
        else:
            print(f"❌ {file_path} - Still has errors: {result.stderr}")
            return False
    
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False

def main():
    """Main function to fix all supervisor files."""
    supervisor_dir = "src/supervisor"
    
    if not os.path.exists(supervisor_dir):
        print(f"Directory {supervisor_dir} not found")
        return
    
    # Get all Python files in supervisor directory
    python_files = glob.glob(os.path.join(supervisor_dir, "*.py"))
    
    print(f"Found {len(python_files)} Python files in {supervisor_dir}")
    
    fixed_count = 0
    total_count = len(python_files)
    
    for file_path in python_files:
        if fix_supervisor_file(file_path):
            fixed_count += 1
    
    print(f"\nSummary: Fixed {fixed_count}/{total_count} files")

if __name__ == "__main__":
    main()