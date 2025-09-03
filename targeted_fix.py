#!/usr/bin/env python3
"""
Targeted script to fix specific syntax issues without breaking import statements.
"""

import re


def fix_state_manager():
    """Fix src/utils/state_manager.py with targeted patterns"""
    file_path="src/utils/state_manager.py"

    with open(file_path) as f:
        content=f.read()

    # Fix specific patterns only
    # Fix variable assignments in class methods (self.var, value)
    content = re.sub(r"(self\.[A-Za-z_][A-Za-z0-9_]*)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)", r"\\1=\\2", content)

    # Fix decorator parameters
    content = re.sub(r"default_return\s*,\s*([A-Za-z_][A-Za-z0-9_]*)", r"default_return=\\1", content)

    # Fix function parameter syntax (param: type, default)
    content = re.sub(r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)", r"\\1: \\2=\\3", content)

    with open(file_path, "w") as f:
        f.write(content)

    print(f"✅ Fixed {file_path}")

def fix_model_manager():
    """Fix src/utils/model_manager.py with targeted patterns"""
    file_path="src/utils/model_manager.py"

    with open(file_path) as f:
        content=f.read()

    # Fix specific patterns only
    # Fix variable assignments
    content = re.sub(r"([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)", r"\\1=\\2", content)

    # Fix function parameter syntax
    content = re.sub(r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)", r"\\1: \\2=\\3", content)

    # Fix getattr calls
    content = re.sub(r"getattr\(([^,\)]+)\)\s*=\s*([^\)\s]+)", r"getattr(\\1, \\2)", content)

    with open(file_path, "w") as f:
        f.write(content)

    print(f"✅ Fixed {file_path}")

def fix_config_loader():
    """Fix src/utils/config_loader.py with targeted patterns"""
    file_path="src/utils/config_loader.py"

    with open(file_path) as f:
        content=f.read()

    # Fix specific patterns only
    # Fix function parameter syntax
    content = re.sub(r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)", r"\\1: \\2=\\3", content)

    # Fix decorator parameters
    content = re.sub(r"default_return\s*,\s*([A-Za-z_][A-Za-z0-9_]*)", r"default_return=\\1", content)

    with open(file_path, "w") as f:
        f.write(content)

    print(f"✅ Fixed {file_path}")

def fix_async_utils():
    """Fix src/utils/async_utils.py with targeted patterns"""
    file_path="src/utils/async_utils.py"

    with open(file_path) as f:
        content=f.read()

    # Fix specific patterns only
    # Fix variable assignments
    content = re.sub(r"([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)", r"\\1=\\2", content)

    # Fix decorator parameters
    content = re.sub(r"default_return\s*,\s*([A-Za-z_][A-Za-z0-9_]*)", r"default_return=\\1", content)

    with open(file_path, "w") as f:
        f.write(content)

    print(f"✅ Fixed {file_path}")

def main():
    """Main function to fix all remaining files"""
    print("🔧 Applying targeted fixes...")

    # Fix each file
    fix_state_manager()
    fix_model_manager()
    fix_config_loader()
    fix_async_utils()

    print("✅ All files processed!")

if __name__== "__main__":
    main()
