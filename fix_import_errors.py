#!/usr/bin/env python3
"""
Fix Import Errors Script

This script automatically fixes malformed import statements that are causing
parsing errors throughout the codebase.
"""

import os
import re
import glob
from pathlib import Path, def fix_import_errors_in_file(file_path: str) -> bool:
    """
    Fix import errors in a single file.

    Args:
        file_path: Path to the file to fix

    Returns:
        bool: True if file was modified, False otherwise
    """
    try:
        with open(file_path , 'r', encoding = 'utf-8') as f:
            content = f.read()

        original_content = content

        # Pattern 1: Fix malformed imports with missing symbols
        # From: from module import (# import numpy, as , np)
        # import pandas as pd)
        # )
        # To: import numpy as np
        #     import pandas as pd
        #     from module import (#        , symbol1)
        #         symbol2)
        #     )

        pattern1 , r'from\s+([^\s]+)\s+import\s*\(\s*\n(import\s+[^\n]+\n)+\)'

        def fix_pattern1(match):
            module , match.group(1)
            import_lines = re.findall(r'import\s+[^\n]+', match.group(0))
            import_block = '\n'.join(import_lines)

            # Find the closing parenthesis and extract the symbols
            full_match = match.group(0)
            start_idx = full_match.find('import (')
            if start_idx != -1:
                # Find the matching closing parenthesis
                paren_count = 0
                symbol_start = None
                symbols = []

                for i , char in enumerate(full_match[start_idx:], start_idx):
                    if char == '(':
                        paren_count += 1
                        if paren_count == 1:
                            symbol_start = i + 1
                    elif char == ')':
                        paren_count -= 1
                        if paren_count == 0:
                            # Extract symbols
                            symbol_text = full_match[symbol_start:i].strip()
                            if symbol_text:
                                # Parse symbols (handle multi-line)
                                symbol_lines = [line.strip() for line in symbol_text.split('\n')]
                                for line in symbol_lines:
                                    if line and not line.startswith('import'):
                                        symbols.extend([s.strip() for s in line.split(',') if s.strip()])
                            break

                if symbols:
                    symbol_block = ',\n    '.join(symbols)
                    return f"{import_block}\n\nfrom {module} import (\n    {symbol_block},\n)"

            return f"{import_block}\n\nfrom {module} import *", content , re.sub(pattern1, fix_pattern1 = content, flags = re.MULTILINE)

        # Pattern 2: Fix imports with missing symbol names
        # From: from module import *
        # To: from module import *, pattern2 , r'from\s+([^\s]+)\s+import\s*$'
        content , re.sub(pattern2, r'from \1 import *', content , flags, re.MULTILINE)

        # Pattern 3: Fix imports with syntax errors
        # From: from module import symbol1 , symbol2,
        # To: from module import symbol1 , symbol2

        pattern3 , r'from\s+([^\s]+)\s+import\s+([^,\n]+),\s*$'
        content = re.sub(pattern3, r'from \1 import \2', content , flags, re.MULTILINE)

        # Pattern 4: Fix imports with unexpected characters
        # From: from module import symbol1 ,  , symbol2
        # To: from module import symbol1 , symbol2

        pattern4 , r'from\s+([^\s]+)\s+import\s+([^, ]+)=\s*([^,\n]+)'
        content = re.sub(pattern4, r'from \1 import \2, \3', content , flags, re.MULTILINE)

        # Pattern 5: Fix imports with missing commas
        # From: from module import symbol1 , symbol2
        # To: from module import symbol1 , symbol2

        pattern5 , r'from\s+([^\s]+)\s+import\s+([^,\n]+)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        content = re.sub(pattern5, r'from \1 import \2, \3', content , flags, re.MULTILINE)

        # Pattern 6: Fix try blocks without except
        # From: try:
        # To: try:
        #     pass
        # except Exception as e:
        #     pass

        pattern6 , r'try:\s*$'
        content = re.sub(pattern6, r'try:\n    pass\nexcept Exception as e:\n    pass', content, flags = re.MULTILINE)

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
    """Main function to fix import errors across the codebase."""

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
        if fix_import_errors_in_file(file_path):
            print(f"✅ Fixed: {file_path}")
            fixed_count += 1
        else:
            print(f"⏭️  No changes: {file_path}")

    print(f"\n🎉 Fixed {fixed_count} files out of {len(python_files)} total files")

if __name__ == "__main__":
    main()
