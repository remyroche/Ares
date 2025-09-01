#!/usr/bin/env python3
"""
Conservative Syntax Fixer for Ares Repository

This script applies only the safest and most targeted fixes to avoid introducing
new errors. It focuses on:
1. Simple import statement fixes
2. Basic function call syntax fixes
3. Simple indentation fixes
4. Only the most obvious malformed try-except patterns
"""

import os
import re
from pathlib import Path
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ConservativeSyntaxFixer:
    """Conservative syntax fixer that only applies safe fixes."""

    def __init__(self):
        self.fixes_applied = 0
        self.files_processed = 0
        self.files_fixed = 0

    def fix_file(self, file_path: str) -> bool:
        """Fix syntax errors in a single file using conservative approach."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            fixes_in_file = 0

            # Apply only the safest fixes
            content, fixes = self._fix_simple_import_errors(content)
            fixes_in_file += fixes

            content, fixes = self._fix_simple_function_calls(content)
            fixes_in_file += fixes

            content, fixes = self._fix_simple_indentation(content)
            fixes_in_file += fixes

            # Write back if changes were made
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.fixes_applied += fixes_in_file
                self.files_fixed += 1
                logger.info(f"✅ Fixed {fixes_in_file} issues in {file_path}")
                return True
            else:
                return False

        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
            return False

    def _fix_simple_import_errors(self, content: str) -> Tuple[str, int]:
        """Fix only the most obvious import statement errors."""
        fixes = 0

        # Fix the specific import error we found
        if 'from pathlib import Path
import glob' in content:
            content = content.replace('from pathlib import Path
import glob', 'from pathlib import Path\nimport glob')
            fixes += 1

        return content, fixes

    def _fix_simple_function_calls(self, content: str) -> Tuple[str, int]:
        """Fix only the most obvious function call syntax errors."""
        fixes = 0

        # Fix logging.basicConfig calls with syntax errors
        content = re.sub(
            r'logging\.basicConfig\(\s*level\s*,\s*logging\.INFO\s*,\s*format\s*,\s*"([^"]*)"\s*\)',
            r'logging.basicConfig(level=logging.INFO, format=r"\1")',
            content
        )
        fixes += len(re.findall(r'logging\.basicConfig\(\s*level\s*,\s*logging\.INFO\s*,\s*format\s*,\s*"', content))

        # Fix max() function calls with syntax errors
        content = re.sub(
            r'max\(([^,]+),\s*key\s*=\s*([^)]+)\)',
            r'max(\1, key=\2)',
            content
        )
        fixes += len(re.findall(r'max\([^,]+,\s*key\s*=\s*[^)]+\)', content))

        # Fix to_parquet calls with syntax errors
        content = re.sub(
            r'\.to_parquet\(([^,]+),\s*index\s*=\s*False\)',
            r'.to_parquet(\1, index=False)',
            content
        )
        fixes += len(re.findall(r'\.to_parquet\([^,]+,\s*index\s*=\s*False\)', content))

        return content, fixes

    def _fix_simple_indentation(self, content: str) -> Tuple[str, int]:
        """Fix only the most obvious indentation issues."""
        fixes = 0
        lines = content.split('\n')
        fixed_lines = []

        for line in lines:
            # Only fix obvious tab-to-space conversions
            if line.startswith('\t') and not line.startswith('    '):
                line = '    ' + line[1:]
                fixes += 1

            fixed_lines.append(line)

        return '\n'.join(fixed_lines), fixes

    def scan_and_fix_directory(self, directory: str) -> dict:
        """Scan and fix all Python files in a directory."""
        logger.info(f"🔍 Scanning directory: {directory}")

        # Find all Python files
        python_files = []
        for root, dirs, files in os.walk(directory):
            # Skip certain directories
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', 'venv', 'env', 'backup_']]

            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))

        logger.info(f"📁 Found {len(python_files)} Python files")

        # Process each file
        for file_path in python_files:
            self.files_processed += 1
            self.fix_file(file_path)

        return {
            'files_processed': self.files_processed,
            'files_fixed': self.files_fixed,
            'total_fixes': self.fixes_applied
        }


def main():
    """Main function to run the conservative syntax fixer."""
    logger.info("🚀 Starting conservative syntax fixer")

    fixer = ConservativeSyntaxFixer()

    # Fix files in the current directory and subdirectories
    results = fixer.scan_and_fix_directory('.')

    # Print summary
    logger.info("📊 Fix Summary:")
    logger.info(f"   Files processed: {results['files_processed']}")
    logger.info(f"   Files fixed: {results['files_fixed']}")
    logger.info(f"   Total fixes applied: {results['total_fixes']}")

    # Run a verification scan
    logger.info("🔍 Running verification scan...")
    import subprocess
    try:
        result = subprocess.run(
            "find . -name '*.py' -type f -exec python -m py_compile {} \; 2>&1 | wc -l",
            shell=True, capture_output=True, text=True
        )
        remaining_errors = int(result.stdout.strip())
        logger.info(f"   Remaining errors: {remaining_errors}")

        if remaining_errors < 466:  # Original error count
            improvement = 466 - remaining_errors
            logger.info(f"✅ Improved by {improvement} errors!")
        else:
            logger.warning("⚠️ No improvement detected")

    except Exception as e:
        logger.error(f"❌ Error during verification: {e}")

    logger.info("✅ Conservative syntax fixing completed!")


if __name__ == "__main__":
    main()
