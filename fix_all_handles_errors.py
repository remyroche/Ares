#!/usr/bin/env python3
"""
Script to fix all remaining handles_errors import issues.
"""

import os
import re
from pathlib import Path

def fix_handles_errors_imports(file_path):
    """Fix handles_errors imports in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Calculate relative path depth
        parts = file_path.parts
        src_index = None
        for i, part in enumerate(parts):
            if part == 'src':
                src_index = i
                break
        
        if src_index is None:
            return False
        
        # Calculate depth from src
        depth = len(parts) - src_index - 2  # -2 for src and filename
        relative_prefix = '.' * depth if depth > 0 else '.'
        
        # Fix various incorrect import patterns
        patterns_to_fix = [
            # Remove duplicate imports
            (r'from src\.core\.decorators import handles_errors\n', ''),
            (r'from src\.utils\.decorators\.errors import handles_errors\n', ''),
            (r'from src\.utils\.decorators import handles_errors\n', ''),
            (r'from \.core\.decorators\.errors import handles_errors\n', ''),
            (r'from \.decorators\.errors import handles_errors\n', ''),
            (r'from \.error_handler import handles_errors\n', ''),
            (r'from \.errors import handles_errors\n', ''),
            (r'from errors import handles_errors\n', ''),
            
            # Fix correct imports
            (r'from core\.decorators import handles_errors', f'from {relative_prefix}core.decorators import handles_errors'),
        ]
        
        for pattern, replacement in patterns_to_fix:
            content = re.sub(pattern, replacement, content)
        
        # Add correct import if handles_errors is used but not imported
        if 'handles_errors' in content and 'from ' + relative_prefix + 'core.decorators import handles_errors' not in content:
            # Find the best place to add the import
            lines = content.split('\n')
            import_section_end = 0
            
            for i, line in enumerate(lines):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    import_section_end = i + 1
                elif line.strip() and not line.strip().startswith('#'):
                    break
            
            # Insert the import
            lines.insert(import_section_end, f'from {relative_prefix}core.decorators import handles_errors')
            content = '\n'.join(lines)
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function to fix all handles_errors imports."""
    src_dir = Path('/Users/remyroche/Documents/Ares/src')
    fixed_count = 0
    
    # Get list of Python files
    python_files = list(src_dir.rglob('*.py'))
    
    print(f"Processing {len(python_files)} Python files...")
    
    for file_path in python_files:
        if fix_handles_errors_imports(file_path):
            print(f"Fixed: {file_path}")
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} files")

if __name__ == '__main__':
    main()
