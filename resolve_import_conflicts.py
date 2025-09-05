#!/usr/bin/env python3
"""
Script to automatically resolve import conflicts in merge.
The conflicts are typically between:
- Our branch: Has numpy/pandas imports added by auto-fixer
- Main branch: Has additional imports that our branch doesn't have

This script merges both sets of imports.
"""

import os
import re
from pathlib import Path


def resolve_import_conflicts(file_path: str) -> bool:
    """Resolve import conflicts in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if file has conflicts
        if '<<<<<<< HEAD' not in content:
            return False
        
        print(f"Resolving conflicts in {file_path}")
        
        # Pattern to match import conflicts
        # This handles the common pattern where our branch has numpy/pandas imports
        # and main branch has additional imports
        conflict_pattern = r'<<<<<<< HEAD\n(.*?)\n=======\n(.*?)\n>>>>>>> origin/main'
        
        def resolve_conflict(match):
            our_content = match.group(1).strip()
            main_content = match.group(2).strip()
            
            # If our content is just import statements and main content has more imports
            if (our_content.startswith('import ') or our_content.startswith('from ')) and \
               (main_content.startswith('import ') or main_content.startswith('from ') or main_content == ''):
                
                # Merge both sets of imports
                if main_content:
                    return f"{our_content}\n{main_content}"
                else:
                    return our_content
            
            # For other conflicts, prefer main branch content
            return main_content
        
        # Resolve all conflicts
        resolved_content = re.sub(conflict_pattern, resolve_conflict, content, flags=re.DOTALL)
        
        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(resolved_content)
        
        return True
        
    except Exception as e:
        print(f"Error resolving conflicts in {file_path}: {e}")
        return False


def main():
    """Main function to resolve all import conflicts."""
    # Get list of conflicted files from git status
    import subprocess
    
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True, cwd='/workspace')
        
        conflicted_files = []
        for line in result.stdout.split('\n'):
            if line.startswith('UU ') or line.startswith('AA ') or line.startswith('DD '):
                file_path = line[3:].strip()
                if file_path.endswith('.py'):
                    conflicted_files.append(file_path)
        
        print(f"Found {len(conflicted_files)} Python files with conflicts")
        
        resolved_count = 0
        for file_path in conflicted_files:
            if resolve_import_conflicts(file_path):
                resolved_count += 1
        
        print(f"Resolved conflicts in {resolved_count} files")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()