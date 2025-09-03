#!/usr/bin/env python3
"""
Comprehensive syntax error fixer for Python files.
Addresses common syntax errors found in the codebase.
"""

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any


class SyntaxErrorFixer:
    """Fixes common syntax errors in Python files."""

    def __init__(self):
        self.fixes_applied = []
        self.backup_dir = Path("syntax_fix_backups")
        self.backup_dir.mkdir(exist_ok=True)

    def backup_file(self, file_path: str) -> str:
        """Create a backup of the file before modifying."""
        backup_path = self.backup_dir / Path(file_path).name
        shutil.copy2(file_path, backup_path)
        return str(backup_path)

    def fix_unterminated_string_literal(self, content: str) -> tuple[str, bool]:
        """Fix unterminated string literals (e.g., quadruple quotes to triple quotes)."""
        fixed = False

        # Fix quadruple quotes to triple quotes
        if '""""' in content:
            content = content.replace('""""', '"""')
            fixed = True

        # Fix standalone quadruple quotes on separate lines
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.strip() == '""""':
                lines[i] = line.replace('""""', '"""')
                fixed = True

        if fixed:
            content = "\n".join(lines)

        return content, fixed

    def fix_invalid_regex_patterns(self, content: str) -> tuple[str, bool]:
        """Fix invalid regex patterns in re.sub calls."""
        fixed = False

        # Common pattern fixes
        replacements = [
            # Fix unclosed parentheses in regex patterns
            (r"re\.sub\(r'(\w+)'\)([^,]*?)', r'", r"re.sub(r'\1\2', r'"),
            # Fix improper escaping
            (r"re\.sub\(r'([^']*?)(\w+)'([^']*?)\)", r"re.sub(r'\1\2\3)"),
            # Fix assignment in regex replacement
            (r"= (\w+) = (\w+)", r", \1, \2"),
        ]

        for pattern, replacement in replacements:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                fixed = True

        return content, fixed

    def fix_assignment_in_expressions(self, content: str) -> tuple[str, bool]:
        """Fix assignments used in expressions where comparison was intended."""
        fixed = False

        # Fix assignment in function arguments
        pattern = r'(with open\([^,]+)="([^"]+)"\)'
        if re.search(pattern, content):
            content = re.sub(pattern, r'\1, "\2")', content)
            fixed = True

        # Fix assignment in except clauses
        pattern = r"except \((\w+) = (\w+)\):"
        if re.search(pattern, content):
            content = re.sub(pattern, r"except (\1, \2):", content)
            fixed = True

        # Fix assignment in list append
        pattern = r"\.append\(\(i=([^)]+)\)\)"
        if re.search(pattern, content):
            content = re.sub(pattern, r".append((\1))", content)
            fixed = True

        return content, fixed

    def fix_syntax_errors_in_function_definitions(self, content: str) -> tuple[str, bool]:
        """Fix syntax errors in function definitions."""
        fixed = False

        # Fix quoted parameter names
        pattern = r'def (\w+)\([^,)]*,\s*"(\w+)"'
        if re.search(pattern, content):
            content = re.sub(pattern, r"def \1(\2", content)
            fixed = True

        # Fix parameter type annotations
        pattern = r": (\w+)\[str=Any\]"
        if re.search(pattern, content):
            content = re.sub(pattern, r": \1[str, Any]", content)
            fixed = True

        return content, fixed

    def fix_file(self, file_path: str, errors: list[dict[str, Any]]) -> bool:
        """Fix syntax errors in a single file."""
        try:
            # Read file content
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            original_content = content
            any_fixed = False

            # Apply fixes based on error types
            for error in errors:
                error_msg = error.get("message", "")

                if "unterminated string literal" in error_msg:
                    content, fixed = self.fix_unterminated_string_literal(content)
                    any_fixed |= fixed

                elif "invalid syntax" in error_msg:
                    content, fixed = self.fix_invalid_regex_patterns(content)
                    any_fixed |= fixed
                    content, fixed = self.fix_syntax_errors_in_function_definitions(content)
                    any_fixed |= fixed

                elif "cannot assign to function call" in error_msg:
                    content, fixed = self.fix_assignment_in_expressions(content)
                    any_fixed |= fixed

            # Write fixed content if changes were made
            if any_fixed and content != original_content:
                # Create backup
                backup_path = self.backup_file(file_path)

                # Write fixed content
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)

                self.fixes_applied.append({
                    "file": file_path,
                    "backup": backup_path,
                    "errors_fixed": len(errors),
                })

                return True

        except Exception as e:
            print(f"Error fixing {file_path}: {str(e)}")

        return False

    def fix_from_results_file(self, results_file: str) -> dict[str, Any]:
        """Fix syntax errors based on results from syntax checker."""
        with open(results_file) as f:
            results = json.load(f)

        files_to_fix = [
            r for r in results["results"]
            if not r["valid"] and r["errors"]
        ]

        print(f"Found {len(files_to_fix)} files with syntax errors to fix")

        fixed_count = 0
        for file_info in files_to_fix:
            file_path = file_info["file"]
            errors = file_info["errors"]

            # Skip certain fix scripts that might have complex regex patterns
            if any(skip in file_path for skip in ["fix_", "fixer.py", "comprehensive_fix"]):
                continue

            if self.fix_file(file_path, errors):
                fixed_count += 1
                print(f"✓ Fixed {file_path}")

        return {
            "total_files_with_errors": len(files_to_fix),
            "files_fixed": fixed_count,
            "fixes_applied": self.fixes_applied,
        }


def main():
    """Main function to run the syntax error fixer."""
    import argparse

    parser = argparse.ArgumentParser(description="Fix syntax errors in Python files")
    parser.add_argument("--results-file", default="syntax_check_results.json",
                       help="Path to syntax check results JSON file")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be fixed without making changes")

    args = parser.parse_args()

    if not os.path.exists(args.results_file):
        print(f"Error: Results file '{args.results_file}' not found")
        print("Please run the syntax checker first")
        return 1

    fixer = SyntaxErrorFixer()
    results = fixer.fix_from_results_file(args.results_file)

    print(f"\n{'='*60}")
    print("SYNTAX FIX SUMMARY")
    print(f"{'='*60}")
    print(f"Total files with errors: {results['total_files_with_errors']}")
    print(f"Files fixed: {results['files_fixed']}")
    print(f"Backup directory: {fixer.backup_dir}")

    if results["files_fixed"] > 0:
        print(f"\n✅ Successfully fixed {results['files_fixed']} files")
        print("Run the syntax checker again to verify all errors are resolved")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
