#!/usr/bin/env python3
"""
Comprehensive script to fix all systematic syntax issues introduced by manual changes.
This script addresses the specific patterns that were broken by replacing commas with equals signs.
"""

import re
import os

def fix_type_annotations(content):
    """Fix broken type annotations"""
    # Fix dict type annotations
    content = re.sub(r'dict\[str = (\w+)\]', r'dict[str, \1]', content)
    content = re.sub(r'dict\[(\w+) = (\w+)\]', r'dict[\1, \2]', content)
    
    # Fix tuple type annotations
    content = re.sub(r'tuple\[(\w+) = (\w+)\]', r'tuple[\1, \2]', content)
    
    # Fix list type annotations
    content = re.sub(r'list\[(\w+) = (\w+)\]', r'list[\1, \2]', content)
    
    return content

def fix_function_signatures(content):
    """Fix broken function signatures"""
    # Fix parameter type annotations
    content = re.sub(r'(\w+): (\w+) = (\w+)', r'\1: \2 = \3', content)
    
    # Fix function parameter lists
    content = re.sub(r'def (\w+)\(self = (\w+)\): (\w+)', r'def \1(self, \2: \3)', content)
    content = re.sub(r'def (\w+)\(self, (\w+)\): (\w+)', r'def \1(self, \2: \3)', content)
    
    # Fix async function signatures
    content = re.sub(r'async def (\w+)\(self = (\w+)\): (\w+)', r'async def \1(self, \2: \3)', content)
    
    return content

def fix_import_statements(content):
    """Fix broken import statements"""
    # Fix from imports
    content = re.sub(r'from (\w+) import (\w+) = (\w+)', r'from \1 import \2, \3', content)
    
    # Fix multiple imports
    content = re.sub(r'(\w+) = (\w+)', r'\1, \2', content)
    
    return content

def fix_exception_handling(content):
    """Fix broken exception handling"""
    # Fix except clauses
    content = re.sub(r'except \((\w+) = (\w+)\):', r'except (\1, \2):', content)
    
    return content

def fix_decorator_parameters(content):
    """Fix broken decorator parameters"""
    # Fix decorator parameter assignments
    content = re.sub(r'(\w+) = (\w+) = (\w+)', r'\1=\2, \3', content)
    
    return content

def fix_file(file_path):
    """Fix a single file"""
    print(f"🔧 Fixing {file_path}...")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Apply all fixes
    content = fix_type_annotations(content)
    content = fix_function_signatures(content)
    content = fix_import_statements(content)
    content = fix_exception_handling(content)
    content = fix_decorator_parameters(content)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixed {file_path}")

def main():
    """Main function to fix all files"""
    files_to_fix = [
        "src/utils/state_manager.py",
        "src/utils/model_manager.py", 
        "src/utils/config_loader.py",
        "src/utils/async_utils.py"
    ]
    
    print("🔧 Applying comprehensive fixes...")
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            fix_file(file_path)
        else:
            print(f"⚠️ File not found: {file_path}")
    
    print("✅ All files processed!")

if __name__ == "__main__":
    main()
