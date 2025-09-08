#!/usr/bin/env python3
"""Fix remaining relative import issues in training directory."""

import os
import re

def fix_training_imports():
    root_dir = "/Users/remyroche/Documents/Ares"
    
    # Patterns to fix
    patterns = [
        (r"from \.\.\.utils\.logger import system_logger", r"from src.utils.logger import system_logger"),
        (r"from \.\.\.core\.decorators import handles_errors", r"from src.core.decorators import handles_errors"),
        (r"from \.\.\.utils\.core\.decorators import handles_errors", r"from src.core.decorators import handles_errors"),
        (r"from \.\.\.utils\.core\.decorators import traced", r"from src.core.decorators import traced"),
        (r"from \.\.\.utils\.core\.decorators import validates", r"from src.core.decorators import validates"),
    ]
    
    fixed_files = []
    
    # Only fix files in the training directory
    training_dir = os.path.join(root_dir, "src", "training")
    
    for dirpath, _, filenames in os.walk(training_dir):
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
    
    print(f"\n🎉 Fixed {len(fixed_files)} files in training directory")
    return fixed_files

if __name__ == "__main__":
    fix_training_imports()
