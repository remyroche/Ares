#!/usr/bin/env python3
"""
Script to fix syntax errors in training steps files.
This script addresses common syntax issues found in the placeholder analysis.
"""

import os
import re
from pathlib import Path

def fix_syntax_errors(file_path):
    """Fix common syntax errors in Python files."""
    print(f"Fixing syntax errors in: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Fix common syntax patterns
    
    # 1. Fix type hints with = instead of ,
    content = re.sub(r'from typing import ([^=]+) = ([^=]+)', r'from typing import \1, \2', content)
    content = re.sub(r'([A-Za-z_][A-Za-z0-9_]*): ([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)', r'\1: \2, \3', content)
    
    # 2. Fix assignment syntax with = instead of =
    content = re.sub(r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)', r'\1 = \2, \3', content)
    
    # 3. Fix import statements
    content = re.sub(r'from ([^=]+) import ([^=]+) = ([^=]+)', r'from \1 import \2, \3', content)
    
    # 4. Fix function parameters
    content = re.sub(r'def ([^(]+)\(([^)]*)\):\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)', r'def \1(\2):\n    \3 = \4', content)
    
    # 5. Fix if statements without colons
    content = re.sub(r'if ([^:]+):\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^:]+)', r'if \1:\n    \2 = \3', content)
    
    # 6. Fix lambda expressions
    content = re.sub(r'lambda \* args = \*\*kwargs', r'lambda *args, **kwargs', content)
    content = re.sub(r'lambda \* args = \*\*kwargs', r'lambda *args, **kwargs', content)
    
    # 7. Fix dictionary assignments
    content = re.sub(r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)', r'\1 = \2, \3', content)
    
    # 8. Fix list comprehensions
    content = re.sub(r'for ([^=]+) = ([^=]+) in ([^:]+):', r'for \1, \2 in \3:', content)
    
    # 9. Fix function calls with = instead of ,
    content = re.sub(r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)', r'\1 = \2, \3', content)
    
    # 10. Fix class method definitions
    content = re.sub(r'def ([^(]+)\(self = ([^)]*)\):', r'def \1(self, \2):', content)
    
    # 11. Fix logging.basicConfig
    content = re.sub(r'logging\.basicConfig\(level = logging\.INFO\)', r'logging.basicConfig(level=logging.INFO)', content)
    
    # 12. Fix sys.path.insert
    content = re.sub(r'sys\.path\.insert\(0 = str\(([^)]+)\)\)', r'sys.path.insert(0, str(\1))', content)
    
    # 13. Fix PipelineStandards import
    content = re.sub(r'from src\.utils\.pipeline_standards import PipelineStandards = pipeline_standards', 
                    r'from src.utils.pipeline_standards import PipelineStandards, pipeline_standards', content)
    
    # 14. Fix decorator assignments
    content = re.sub(r'([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)', 
                    r'\1 = \2.\3', content)
    
    # 15. Fix file path comments
    content = re.sub(r'# src / training / steps /', r'# src/training/steps/', content)
    
    # 16. Fix dictionary key-value pairs
    content = re.sub(r'"([^"]+)"\s*=\s*([^,]+)\s*=\s*([^,]+)', r'"\1": \2, \3', content)
    
    # 17. Fix function calls with missing commas
    content = re.sub(r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)', 
                    r'\1 = \2, \3', content)
    
    # 18. Fix list assignments
    content = re.sub(r'\[([^=]+) = ([^=]+)\]', r'[\1, \2]', content)
    
    # 19. Fix tuple assignments
    content = re.sub(r'\(([^=]+) = ([^=]+)\)', r'(\1, \2)', content)
    
    # 20. Fix numpy operations
    content = re.sub(r'np\.([^(]+)\(([^=]+) = ([^=]+)\)', r'np.\1(\2, \3)', content)
    
    # Write back if changes were made
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Fixed syntax errors in: {file_path}")
        return True
    else:
        print(f"ℹ️  No syntax errors found in: {file_path}")
        return False

def main():
    """Main function to fix syntax errors in training steps files."""
    training_steps_dir = Path("src/training/steps")
    
    if not training_steps_dir.exists():
        print(f"❌ Training steps directory not found: {training_steps_dir}")
        return
    
    # Get all Python files in the training steps directory
    python_files = list(training_steps_dir.rglob("*.py"))
    
    print(f"🔍 Found {len(python_files)} Python files to check")
    
    fixed_count = 0
    for file_path in python_files:
        try:
            if fix_syntax_errors(file_path):
                fixed_count += 1
        except Exception as e:
            print(f"❌ Error fixing {file_path}: {e}")
    
    print(f"\n📊 Summary:")
    print(f"   - Total files checked: {len(python_files)}")
    print(f"   - Files fixed: {fixed_count}")
    print(f"   - Files unchanged: {len(python_files) - fixed_count}")

if __name__ == "__main__":
    main()