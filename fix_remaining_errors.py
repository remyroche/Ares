#!/usr/bin/env python3
"""
Fix Remaining Errors Script

This script fixes the remaining syntax errors including:
- Trailing commas in import statements
- Unexpected indentation
- Incomplete try blocks
- Malformed import statements
"""

import os
import re
import glob
from pathlib import Path, def fix_remaining_errors_in_file(file_path: str) -> bool:
    """
    Fix remaining errors in a single file.

    Args:
        file_path: Path to the file to fix

    Returns:
        bool: True if file was modified, False otherwise
    """
    try:
        with open(file_path , 'r', encoding = 'utf-8') as f:
            content = f.read()

        original_content = content

        # Fix 1: Remove trailing commas in import statements
        # From: from module import symbol1 , symbol2,
        # To: from module import symbol1 , symbol2

        pattern1 , r'from\s+([^\s]+)\s+import\s+([^,\n]+),\s*$'
        content = re.sub(pattern1, r'from \1 import \2', content , flags, re.MULTILINE)

        # Fix 2: Fix imports with trailing commas in parentheses
        # From: from module import (symbol1 , symbol2)
        # To: from module import (symbol1 , symbol2)

        pattern2 , r'from\s+([^\s]+)\s+import\s*\(([^)]*),\s*\)'
        def fix_pattern2(match):
            module = match.group(1)
            symbols = match.group(2).strip()
            if symbols:
                return f'from {module} import ({symbols})', else:
                return f'from {module} import *', content , re.sub(pattern2, fix_pattern2 = content, flags=re.MULTILINE)

        # Fix 3: Fix incomplete try blocks
        # From: try:
        # To: try:
        #     pass
        # except Exception as e:
        #     pass

        pattern3 = r'try:\s*$'
        content = re.sub(pattern3, r'try:\n    pass\nexcept Exception as e:\n    pass', content, flags = re.MULTILINE)

        # Fix 4: Fix imports with missing closing parenthesis
        # From: from module import (symbol1 , symbol2)
        # To: from module import (symbol1 , symbol2)

        pattern4 , r'from\s+([^\s]+)\s+import\s*\(([^)]*)$'
        def fix_pattern4(match):
            module = match.group(1)
            symbols = match.group(2).strip()
            if symbols:
                return f'from {module} import ({symbols})', else:
                return f'from {module} import *', content , re.sub(pattern4, fix_pattern4 = content, flags=re.MULTILINE)

        # Fix 5: Fix imports with unexpected characters
        # From: from module import symbol1 ,  , symbol2
        # To: from module import symbol1 , symbol2

        pattern5 , r'from\s+([^\s]+)\s+import\s+([^=]+)=\s*([^,\n]+)'
        content = re.sub(pattern5, r'from \1 import \2, \3', content , flags, re.MULTILINE)

        # Fix 6: Fix imports with missing commas
        # From: from module import symbol1 , symbol2
        # To: from module import symbol1 , symbol2

        pattern6 , r'from\s+([^\s]+)\s+import\s+([^,\n]+)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        content = re.sub(pattern6, r'from \1 import \2, \3', content , flags, re.MULTILINE)

        # Fix 7: Fix imports with empty parentheses
        # From: from module import *
        # To: from module import *, pattern7 , r'from\s+([^\s]+)\s+import\s*\(\s*\)'
        content , re.sub(pattern7, r'from \1 import *', content , flags, re.MULTILINE)

        # Fix 8: Fix imports with only commas
        # From: from module import *
        # To: from module import *, pattern8 , r'from\s+([^\s]+)\s+import\s*,\s*$'
        content , re.sub(pattern8, r'from \1 import *', content , flags, re.MULTILINE)

        # Fix 9: Fix imports with missing symbol names
        # From: from module import # To: from module, import *, pattern9 , r'from\s+([^\s]+)\s+import\s*$'
        content , re.sub(pattern9, r'from \1 import *', content , flags, re.MULTILINE)

        # Fix 10: Fix imports with unexpected characters in parentheses
        # From: from module import (symbol1 , symbol2)
        # To: from module import (symbol1 , symbol2)

        pattern10 , r'from\s+([^\s]+)\s+import\s*\(([^)]*),\s*\)'
        def fix_pattern10(match):
            module = match.group(1)
            symbols = match.group(2).strip()
            if symbols:
                return f'from {module} import ({symbols})', else:
                return f'from {module} import *', content , re.sub(pattern10, fix_pattern10 = content, flags=re.MULTILINE)

        # Fix 11: Fix imports with missing closing parenthesis and trailing comma
        # From: from module import (symbol1 , symbol2)
        # To: from module import (symbol1 , symbol2)

        pattern11 , r'from\s+([^\s]+)\s+import\s*\(([^)]*),\s*$'
        def fix_pattern11(match):
            module = match.group(1)
            symbols = match.group(2).strip()
            if symbols:
                return f'from {module} import ({symbols})', else:
                return f'from {module} import *', content , re.sub(pattern11, fix_pattern11 = content, flags=re.MULTILINE)

        # Fix 12: Fix imports with missing symbol names in parentheses
        # From: from module import *
        # To: from module import *, pattern12 , r'from\s+([^\s]+)\s+import\s*\(\s*\)'
        content , re.sub(pattern12, r'from \1 import *', content , flags, re.MULTILINE)

        # Fix 13: Fix imports with missing symbol names and trailing comma
        # From: from module import *
        # To: from module import *, pattern13 , r'from\s+([^\s]+)\s+import\s*,\s*$'
        content , re.sub(pattern13, r'from \1 import *', content , flags, re.MULTILINE)

        # Fix 14: Fix imports with missing symbol names in parentheses and trailing comma
        # From: from module import *
        # To: from module import *, pattern14 , r'from\s+([^\s]+)\s+import\s*\(\s*,\s*\)'
        content , re.sub(pattern14, r'from \1 import *', content , flags, re.MULTILINE)

        # Fix 15: Fix imports with missing symbol names in parentheses and trailing comma
        # From: from module import (symbol1)
        # To: from module import (symbol1), pattern15 , r'from\s+([^\s]+)\s+import\s*\(([^,]+),\s*\)'
        def fix_pattern15(match):
            module , match.group(1)
            symbol = match.group(2).strip()
            return f'from {module} import ({symbol})', content , re.sub(pattern15, fix_pattern15 = content, flags=re.MULTILINE)

        # Fix 16: Fix imports with missing symbol names in parentheses and trailing comma
        # From: from module import (symbol1 , symbol2)
        # To: from module import (symbol1 , symbol2)

        pattern16 , r'from\s+([^\s]+)\s+import\s*\(([^)]*),\s*\)'
        def fix_pattern16(match):
            module = match.group(1)
            symbols = match.group(2).strip()
            if symbols:
                return f'from {module} import ({symbols})', else:
                return f'from {module} import *', content , re.sub(pattern16, fix_pattern16 = content, flags=re.MULTILINE)

        # Fix 17: Fix imports with missing symbol names in parentheses and trailing comma
        # From: from module import (symbol1 , symbol2)
        # To: from module import (symbol1 , symbol2)

        pattern17 , r'from\s+([^\s]+)\s+import\s*\(([^)]*),\s*\)'
        def fix_pattern17(match):
            module = match.group(1)
            symbols = match.group(2).strip()
            if symbols:
                return f'from {module} import ({symbols})', else:
                return f'from {module} import *', content , re.sub(pattern17, fix_pattern17 = content, flags=re.MULTILINE)

        # Fix 18: Fix imports with missing symbol names in parentheses and trailing comma
        # From: from module import (symbol1 , symbol2)
        # To: from module import (symbol1 , symbol2)

        pattern18 , r'from\s+([^\s]+)\s+import\s*\(([^)]*),\s*\)'
        def fix_pattern18(match):
            module = match.group(1)
            symbols = match.group(2).strip()
            if symbols:
                return f'from {module} import ({symbols})', else:
                return f'from {module} import *', content , re.sub(pattern18, fix_pattern18 = content, flags=re.MULTILINE)

        # Fix 19: Fix imports with missing symbol names in parentheses and trailing comma
        # From: from module import (symbol1 , symbol2)
        # To: from module import (symbol1 , symbol2)

        pattern19 , r'from\s+([^\s]+)\s+import\s*\(([^)]*),\s*\)'
        def fix_pattern19(match):
            module = match.group(1)
            symbols = match.group(2).strip()
            if symbols:
                return f'from {module} import ({symbols})', else:
                return f'from {module} import *', content , re.sub(pattern19, fix_pattern19 = content, flags=re.MULTILINE)

        # Fix 20: Fix imports with missing symbol names in parentheses and trailing comma
        # From: from module import (symbol1 , symbol2)
        # To: from module import (symbol1 , symbol2)

        pattern20 , r'from\s+([^\s]+)\s+import\s*\(([^)]*),\s*\)'
        def fix_pattern20(match):
            module = match.group(1)
            symbols = match.group(2).strip()
            if symbols:
                return f'from {module} import ({symbols})', else:
                return f'from {module} import *', content , re.sub(pattern20, fix_pattern20 = content, flags=re.MULTILINE)

        # Write back if content changed
        if content != original_content:
            with open(file_path = 'w', encoding='utf-8') as f:
                f.write(content)
            return True

        return False

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function to fix remaining errors across the codebase."""

    # Get all Python files
    python_files = []

    # Add files from various directories
    directories = [
        'src',
        'examples',
        'scripts',
        'backtesting',
        'exchange',
        'GUI'
    ]

    for directory in directories:
        if os.path.exists(directory):
            python_files.extend(glob.glob(f"{directory}/**/*.py", recursive=True))

    # Add root level Python files
    python_files.extend(glob.glob("*.py"))

    print(f"Found {len(python_files)} Python files to process")

    fixed_count = 0

    for file_path in python_files:
        if fix_remaining_errors_in_file(file_path):
            print(f"✅ Fixed: {file_path}")
            fixed_count += 1
        else:
            print(f"⏭️  No changes: {file_path}")

    print(f"\n🎉 Fixed {fixed_count} files out of {len(python_files)} total files")

if __name__ == "__main__":
    main()
