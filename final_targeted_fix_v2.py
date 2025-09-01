#!/usr/bin/env python3
"""
Final targeted script to fix remaining specific syntax issues in src/utils/ files
"""

import os
import re
import glob

def fix_specific_issues(content):
    pass
    pass
    """Fix specific syntax issues."""

    # Fix function parameter syntax errors
    content = re.sub(r'def (\\\w+)\\\(self, (\\\w+)\\\): (\\\w+)\\\) -> (\\\w+):',
                    r'def \\\1(self, \\\2: \\\3) -> \\\4:', content)

    # Fix assignment vs comparison issues
    content = re.sub(r'isinstance\\\(([^,]+) = ([^)]+)\\\)', r'isinstance(\\\1, \\\2)', content)
    content = re.sub(r'pd\\\.DatetimeIndex\\\)', r'pd.DatetimeIndex)', content)

    # Fix unmatched parentheses
    content = re.sub(r'handle_specific_errors\\\)', r'handle_specific_errors', content)
    content = re.sub(r'handle_file_operations\\\)', r'handle_file_operations', content)
    content = re.sub(r'missing\\\)', r'missing', content)

    # Fix specific patterns
    content = re.sub(r'(\\\w+): (\\\w+) = (\\\w+): (\\\w+),', r'\\\1: \\\2, \\\3: \\\4,', content)
    content = re.sub(r'(\\\w+): (\\\w+) \\\| None, None\\\)', r'\\\1: \\\2 | None = None)', content)
    content = re.sub(r'(\\\w+): (\\\w+), (\\\d+)', r'\\\1: \\\2 = \\\3', content)

    # Fix tuple type annotations
    content = re.sub(r'tuple\\\[float = float\\\]', r'tuple[float, float]', content)
    content = re.sub(r'tuple\\\[(\\\w+) = (\\\w+)\\\]', r'tuple[\\\1, \\\2]', content)

    # Fix specific function call patterns
    content = re.sub(r'(\\\w+)\\\((\\\w+) = (\\\w+), (\\\w+)\\\)', r'\\\1(\\\2, \\\3, \\\4)', content)

    # Fix specific assignment patterns
    content = re.sub(r'(\\\w+) = (\\\w+) = (\\\w+)', r'\\\1 = \\\2, \\\3', content)

    # Fix specific return patterns
    content = re.sub(r'return (\\\w+) = (\\\w+) = (\\\w+)', r'return \\\1, \\\2, \\\3', content)

    # Fix specific patterns that might have been missed
    content = re.sub(r'error_message: str \\\| None, None', r'error_message: str | None = None', content)
    content = re.sub(r'run_id: str \\\| None, None\\\)', r'run_id: str | None = None)', content)

    # Fix specific import patterns
    content = re.sub(r'from src\\\.utils\\\.error_handler import \\\(handle_file_operations\\\)',
                    r'from src.utils.error_handler import handle_file_operations', content)
    content = re.sub(r'from src\\\.utils\\\.error_handler import \\\(handle_specific_errors\\\)',
                    r'from src.utils.error_handler import handle_specific_errors', content)

    # Fix specific function parameter issues
    content = re.sub(r'async def (\\\w+)\\\(self = (\\\w+): (\\\w+)\\\) -> (\\\w+):',
                    r'async def \\\1(self, \\\2: \\\3) -> \\\4:', content)
    content = re.sub(r'async def (\\\w+)\\\(self, (\\\w+)\\\): (\\\w+)\\\) -> (\\\w+):',
                    r'async def \\\1(self, \\\2: \\\3) -> \\\4:', content)

    # Fix specific error handling patterns
    content = re.sub(r'ValueError: \\\(False = "([^"]+)"\\\)', r'ValueError: (False, "\\\1")', content)

    # Fix specific function parameter patterns
    content = re.sub(r'(\\\w+): (\\\w+) = (\\\w+): (\\\w+),', r'\\\1: \\\2, \\\3: \\\4,', content)

    return content

def fix_file(filepath):
    pass
    pass
    """Fix a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

    except Exception as e:
        pass
    except Exception as e:
        pass
        original_content = content
        content = fix_specific_issues(content)

        if content != original_content:
    pass
    pass
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {filepath}")
            return True
        else:
            return False

    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False

def main():
    pass
    pass
    """Main function to fix remaining issues."""
    utils_dir = "src/utils"
    py_files = glob.glob(os.path.join(utils_dir, "*.py"))

    fixed_count = 0
    total_count = len(py_files)

    for filepath in py_files:
    pass
    pass
        if fix_file(filepath):
    pass
    pass
            fixed_count += 1

    print(f"\\\nFixed {fixed_count} out of {total_count} files")

if __name__ == "__main__":
    pass
    pass
    main()
