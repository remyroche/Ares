#!/usr/bin/env python3
"""Fix remaining utils import issues."""

import os
import re

def fix_utils_imports():
    root_dir = "/Users/remyroche/Documents/Ares"
    
    # Fix the specific pattern in utils directory
    pattern = r"from \.utils\.logger import system_logger"
    replacement = r"from .logger import system_logger"
    
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
                    
                    new_content = re.sub(pattern, replacement, content)
                    
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
    fix_utils_imports()
