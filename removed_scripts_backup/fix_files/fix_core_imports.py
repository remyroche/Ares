#!/usr/bin/env python3
"""Fix remaining core import issues in utils directory."""

import os
import re

def fix_core_imports():
    root_dir = "/Users/remyroche/Documents/Ares"
    
    # Patterns to fix
    patterns = [
        (r"from \.core\.decorators import", r"from src.core.decorators import"),
        (r"from \.core\.exceptions import", r"from src.core.exceptions import"),
        (r"from \.utils\.file_operations import", r"from src.utils.file_operations import"),
        (r"from \.utils\.warning_symbols import", r"from src.utils.warning_symbols import"),
    ]
    
    fixed_files = []
    
    # Only fix files in the utils directory
    utils_dir = os.path.join(root_dir, "src", "utils")
    
    for dirpath, _, filenames in os.walk(utils_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                filepath = os.path.join(dirpath, filename)
                try:
                    with open(filepath, "r") as f:
                        content = f.read()
                    
                    new_content = content
                    for pattern, replacement in patterns:
                        new_content = re.sub(pattern, replacement, new_content)
                    
                    if new_content != content:
                        with open(filepath, "w") as f:
                            f.write(new_content)
                        fixed_files.append(filepath)
                        print(f"✅ Fixed: {filepath}")
                        
                except Exception as e:
                    print(f"❌ Error processing {filepath}: {e}")
    
    print(f"\n🎉 Fixed {len(fixed_files)} files in utils directory")
    return fixed_files

if __name__ == "__main__":
    fix_core_imports()
