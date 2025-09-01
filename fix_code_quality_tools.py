#!/usr/bin/env python3
"""
Fix indentation issues in code quality tools
"""

import os
import re
from pathlib import Path

def fix_code_quality_tools():
    """Fix indentation issues in code quality tools."""
    tools_dir = Path("code_quality/tools")
    
    if not tools_dir.exists():
        print("Code quality tools directory not found")
        return
    
    files_fixed = 0
    total_files = 0
    
    for py_file in tools_dir.glob("*.py"):
        total_files += 1
        print(f"Processing {py_file}...")
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Fix the specific pattern: try: followed by unindented pass statements
            content = re.sub(
                r'try:\n\s*pass  # TODO: Add proper exception handling\nexcept Exception as e:\n\s*pass  # TODO: Add proper exception handling',
                'try:',
                content
            )
            
            with open(py_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            files_fixed += 1
            print(f"Fixed {py_file}")
        except Exception as e:
            print(f"Failed to fix {py_file}: {e}")
    
    print(f"\nSummary: Fixed {files_fixed}/{total_files} files")

if __name__ == "__main__":
    fix_code_quality_tools()