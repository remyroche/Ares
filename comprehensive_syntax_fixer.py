#!/usr/bin/env python3
"""
Comprehensive syntax fixer that addresses all common Python syntax errors.
"""

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any


class ComprehensiveSyntaxFixer:
    """Fixes all types of syntax errors found in Python files."""

    def __init__(self):
        self.fixes_applied = []
        self.backup_dir = Path("syntax_fix_backups_v2")
        self.backup_dir.mkdir(exist_ok=True)

    def backup_file(self, file_path: str) -> str:
        """Create a backup of the file before modifying."""
        # Create subdirectory structure in backup
        rel_path = Path(file_path).relative_to(".")
        backup_path = self.backup_dir / rel_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, backup_path)
        return str(backup_path)

    def fix_unterminated_string_literal(self, content: str, error_line: int = None) -> tuple[str, bool]:
        """Fix unterminated string literals."""
        fixed = False
        lines = content.split("\n")

        # Fix quadruple quotes
        for i, line in enumerate(lines):
            if '""""' in line:
                lines[i] = line.replace('""""', '"""')
                fixed = True

        # Fix specific pattern for model name extraction
        pattern = r'\.split\("\'>"\)\[0\]\''
        for i, line in enumerate(lines):
            if pattern in line:
                lines[i] = line.replace(pattern, '.split("\'>")[0]')
                fixed = True

        # Fix unterminated f-strings
        for i, line in enumerate(lines):
            if "str(e}" in line:
                lines[i] = line.replace("str(e}", "str(e)}")
                fixed = True

        if fixed:
            content = "\n".join(lines)

        return content, fixed

    def fix_invalid_syntax(self, content: str, error_line: int = None) -> tuple[str, bool]:
        """Fix various invalid syntax patterns."""
        fixed = False

        # Fix function definition with quoted parameter
        pattern = r'def (\w+)\([^,)]*,\s*"(\w+)"'
        if re.search(pattern, content):
            content = re.sub(pattern, r"def \1(\2", content)
            fixed = True

        # Fix incorrect await usage
        pattern = r"return (\w+)\.await (\w+)\("
        if re.search(pattern, content):
            content = re.sub(pattern, r"return await \1.\2(", content)
            fixed = True

        # Fix incorrect await usage 2
        pattern = r"(\w+) = self\.await (\w+)\("
        if re.search(pattern, content):
            content = re.sub(pattern, r"\1 = await self.\2(", content)
            fixed = True

        # Fix extra colon in return statement
        pattern = r'return print\(f"([^"]+)"\):'
        if re.search(pattern, content):
            content = re.sub(pattern, r'return print(f"\1")', content)
            fixed = True

        # Fix missing import statement after docstring
        lines = content.split("\n")
        for i in range(len(lines) - 1):
            if lines[i].strip() == '"""' and i > 0:
                # Check if next non-empty line is an import
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines) and lines[j].strip().startswith("import "):
                    # Check if there's content before the closing quotes
                    if i > 2 and lines[i-1].strip() and not lines[i-2].strip():
                        # Move import to after the docstring
                        import_line = lines[j]
                        lines[j] = ""
                        lines.insert(i + 1, "")
                        lines.insert(i + 2, import_line)
                        fixed = True
                        break

        if fixed:
            content = "\n".join(lines)

        return content, fixed

    def fix_unexpected_indent(self, content: str, error_line: int = None) -> tuple[str, bool]:
        """Fix unexpected indentation errors."""
        fixed = False
        lines = content.split("\n")

        if error_line and 0 < error_line <= len(lines):
            # Check the line and adjust indentation
            line_idx = error_line - 1
            current_line = lines[line_idx]

            if current_line.strip():
                # Find the expected indentation by looking at previous lines
                expected_indent = 0
                for i in range(line_idx - 1, max(0, line_idx - 10), -1):
                    prev_line = lines[i].rstrip()
                    if prev_line and not prev_line.strip().startswith("#"):
                        if prev_line.endswith(":"):
                            # Next line should be indented
                            expected_indent = len(prev_line) - len(prev_line.lstrip()) + 4
                            break
                        # Same level as previous line
                        expected_indent = len(prev_line) - len(prev_line.lstrip())
                        break

                # Apply the expected indentation
                lines[line_idx] = " " * expected_indent + current_line.lstrip()
                fixed = True

        if fixed:
            content = "\n".join(lines)

        return content, fixed

    def fix_import_errors(self, content: str) -> tuple[str, bool]:
        """Fix import-related syntax errors."""
        fixed = False

        # Fix "import os.path" to "import os"
        if "import os.path" in content:
            content = content.replace("import os.path", "import os")
            fixed = True

        # Fix "import copy" when it appears right after docstring
        lines = content.split("\n")
        for i in range(len(lines) - 1):
            if lines[i].strip() == '"""' and i < len(lines) - 1:
                if lines[i+1].strip() == "import copy":
                    # Ensure there's a blank line after docstring
                    if i > 0 and lines[i-1].strip():
                        lines.insert(i+1, "")
                        fixed = True
                        break

        if fixed:
            content = "\n".join(lines)

        return content, fixed

    def fix_expected_except_or_finally(self, content: str, error_line: int = None) -> tuple[str, bool]:
        """Fix missing except or finally blocks."""
        fixed = False
        lines = content.split("\n")

        if error_line and 0 < error_line <= len(lines):
            # Look backwards for a try block
            for i in range(error_line - 1, max(0, error_line - 20), -1):
                if lines[i].strip().startswith("try:"):
                    # Check if there's already an except or finally
                    has_except = False
                    for j in range(i + 1, min(len(lines), error_line + 5)):
                        if lines[j].strip().startswith(("except", "finally")):
                            has_except = True
                            break

                    if not has_except:
                        # Add a generic except block before the error line
                        indent = len(lines[i]) - len(lines[i].lstrip())
                        lines.insert(error_line - 1, " " * indent + "except Exception:")
                        lines.insert(error_line, " " * (indent + 4) + "pass")
                        fixed = True
                        break

        if fixed:
            content = "\n".join(lines)

        return content, fixed

    def fix_file(self, file_path: str, errors: list[dict[str, Any]]) -> bool:
        """Fix syntax errors in a single file."""
        # Skip backup files
        if "syntax_fix_backups" in file_path:
            return False

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            original_content = content
            any_fixed = False

            for error in errors:
                error_msg = error.get("message", "")
                error_line = error.get("line")

                if "unterminated string literal" in error_msg:
                    content, fixed = self.fix_unterminated_string_literal(content, error_line)
                    any_fixed |= fixed

                elif "invalid syntax" in error_msg:
                    content, fixed = self.fix_invalid_syntax(content, error_line)
                    any_fixed |= fixed
                    content, fixed = self.fix_import_errors(content)
                    any_fixed |= fixed

                elif "unexpected indent" in error_msg:
                    content, fixed = self.fix_unexpected_indent(content, error_line)
                    any_fixed |= fixed

                elif "expected an indented block" in error_msg:
                    # Add pass statement
                    lines = content.split("\n")
                    if error_line and 0 < error_line <= len(lines):
                        prev_line = lines[error_line - 2] if error_line > 1 else ""
                        if prev_line.rstrip().endswith(":"):
                            indent = len(prev_line) - len(prev_line.lstrip()) + 4
                            lines.insert(error_line - 1, " " * indent + "pass")
                            content = "\n".join(lines)
                            any_fixed = True

                elif "expected 'except' or 'finally' block" in error_msg:
                    content, fixed = self.fix_expected_except_or_finally(content, error_line)
                    any_fixed |= fixed

            if any_fixed and content != original_content:
                backup_path = self.backup_file(file_path)

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
            if not r["valid"] and r["errors"] and "syntax_fix_backups" not in r["file"]
        ]

        print(f"Found {len(files_to_fix)} files with syntax errors to fix")

        fixed_count = 0
        for file_info in files_to_fix:
            file_path = file_info["file"]
            errors = file_info["errors"]

            if self.fix_file(file_path, errors):
                fixed_count += 1
                print(f"✓ Fixed {file_path}")

        return {
            "total_files_with_errors": len(files_to_fix),
            "files_fixed": fixed_count,
            "fixes_applied": self.fixes_applied,
        }


def main():
    """Main function to run the comprehensive syntax fixer."""
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive syntax error fixer")
    parser.add_argument("--results-file", default="syntax_check_results_after_fix.json",
                       help="Path to syntax check results JSON file")

    args = parser.parse_args()

    if not os.path.exists(args.results_file):
        print(f"Error: Results file '{args.results_file}' not found")
        return 1

    fixer = ComprehensiveSyntaxFixer()
    results = fixer.fix_from_results_file(args.results_file)

    print(f"\n{'='*60}")
    print("COMPREHENSIVE SYNTAX FIX SUMMARY")
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
