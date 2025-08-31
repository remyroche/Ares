#!/usr/bin/env python3
"""
Script to fix decorator registration issues in src/utils/decorators.py
"""

import re

def fix_decorators_file():
    """Remove all @_register_decorator_if_available decorators from decorators.py"""
    
    with open('src/utils/decorators.py', 'r') as f:
        content = f.read()
    
    # Remove all @_register_decorator_if_available decorators and their parameters
    pattern = r'@_register_decorator_if_available\(\s*[^)]*\)\s*\n'
    content = re.sub(pattern, '', content)
    
    with open('src/utils/decorators.py', 'w') as f:
        f.write(content)
    
    print("✅ Fixed decorator registration issues in src/utils/decorators.py")

if __name__ == "__main__":
    fix_decorators_file()