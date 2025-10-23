#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Conservative code fixer that only makes safe, minimal changes.
This fixer is designed to be extremely safe and avoid generating new issues.
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class ConservativeFixer:
    """Conservative, safe code fixer for common issues."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.fixes_applied = 0
        self.files_processed = 0

    def fix_file(self, file_path: Path) -> Dict[str, Any]:
        """Fix common issues in a single file with extreme caution."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()

            # Validate that the file is syntactically correct first
            try:
                ast.parse(original_content)
            except SyntaxError as e:
                logger.warning(f"File {file_path} has syntax errors, skipping: {e}")
                return {
                    "file_path": str(file_path),
                    "fixed": False,
                    "status": "skipped",
                    "reason": "syntax_errors"
                }

            # Apply only the safest fixes
            fixed_content = original_content
            fixes_made = []

            # 1. Detect empty except blocks (report as issues, don't mask them)
            empty_except_issues = self._detect_empty_except_blocks(fixed_content)
            if empty_except_issues:
                fixes_made.append(f"detected_{len(empty_except_issues)}_empty_except_blocks")

            # 2. Fix basic spacing around = operator (very safe)
            fixed_content, spacing_fixed = self._fix_basic_spacing(fixed_content)
            if spacing_fixed:
                fixes_made.append("fixed_basic_spacing")

            # 3. Fix spacing around other operators (safe)
            fixed_content, operator_spacing_fixed = self._fix_operator_spacing(fixed_content)
            if operator_spacing_fixed:
                fixes_made.append("fixed_operator_spacing")

            # Validate the result is still syntactically correct
            try:
                ast.parse(fixed_content)
            except SyntaxError as e:
                logger.warning(f"Fix would introduce syntax errors in {file_path}, reverting: {e}")
                return {
                    "file_path": str(file_path),
                    "fixed": False,
                    "status": "reverted",
                    "reason": "would_introduce_syntax_errors"
                }

            # Only write if content changed
            if fixed_content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                self.fixes_applied += 1
                return {
                    "file_path": str(file_path),
                    "fixed": True,
                    "fixes_applied": fixes_made,
                    "issues_detected": empty_except_issues,
                    "status": "success"
                }
            else:
                return {
                    "file_path": str(file_path),
                    "fixed": False,
                    "fixes_applied": [],
                    "issues_detected": empty_except_issues,
                    "status": "success"
                }

        except Exception as e:
            logger.error(f"Error fixing file {file_path}: {e}")
            return {
                "file_path": str(file_path),
                "fixed": False,
                "status": "error",
                "error": str(e)
            }

    def _detect_empty_except_blocks(self, content: str) -> List[Dict[str, Any]]:
        """Detect empty except blocks and report them as issues instead of masking them."""
        lines = content.split('\n')
        empty_except_issues = []

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if stripped.startswith('except'):
                current_indent = len(line) - len(line.lstrip())
                
                # Check if the except block only contains pass statements or is empty
                is_only_pass = self._is_except_block_only_pass(lines, i, current_indent)
                
                if is_only_pass:
                    empty_except_issues.append({
                        'line_number': i + 1,
                        'line_content': line.strip(),
                        'issue_type': 'empty_except_block',
                        'message': 'Except block only contains pass statement - this masks errors and should be fixed',
                        'suggestion': 'Add proper error handling, logging, or re-raise the exception'
                    })

            i += 1

        return empty_except_issues

    def _is_except_block_only_pass(self, lines: List[str], except_line_index: int, except_indent: int) -> bool:
        """Check if an except block only contains pass statements and comments."""
        # Look at lines after the except statement
        for i in range(except_line_index + 1, len(lines)):
            line = lines[i]
            stripped = line.strip()
            
            # If we hit a line with same or less indentation, we're out of the except block
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= except_indent and stripped:
                break
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                continue
            
            # If we find anything other than 'pass', it's not just a pass block
            if stripped != 'pass':
                return False
        
        return True

    def _fix_basic_spacing(self, content: str) -> Tuple[str, bool]:
        """Fix basic spacing issues around = operator - very conservative."""
        fixed = False

        # Only fix very obvious spacing issues around = (not == or !=)
        # Pattern: word=word (but not word==word or word!=word)
        pattern = r'(\b\w+)=(\w+\b)'
        
        def replacement(match):
            # Make sure it's not == or !=
            if match.group(0) in ['==', '!=', '<=', '>=', '+=', '-=', '*=', '/=']:
                return match.group(0)
            return f"{match.group(1)} = {match.group(2)}"

        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            fixed = True

        return new_content, fixed

    def _fix_operator_spacing(self, content: str) -> Tuple[str, bool]:
        """Fix spacing around operators - conservative approach."""
        fixed = False

        # Fix spacing around assignment operators
        patterns = [
            (r'(\w+)\+=(\w+)', r'\1 += \2'),  # +=
            (r'(\w+)-=(\w+)', r'\1 -= \2'),  # -=
            (r'(\w+)\*=(\w+)', r'\1 *= \2'),  # *=
            (r'(\w+)/=(\w+)', r'\1 /= \2'),  # /=
            (r'(\w+)%=(\w+)', r'\1 %= \2'),  # %=
            (r'(\w+)\*\*=(\w+)', r'\1 **= \2'),  # **=
            (r'(\w+)//=(\w+)', r'\1 //= \2'),  # //=
        ]

        for pattern, replacement in patterns:
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                content = new_content
                fixed = True

        # Fix spacing around comparison operators (but be careful with strings)
        comparison_patterns = [
            (r'(\w+)==(\w+)', r'\1 == \2'),  # ==
            (r'(\w+)!=(\w+)', r'\1 != \2'),  # !=
            (r'(\w+)<=(\w+)', r'\1 <= \2'),  # <=
            (r'(\w+)>=(\w+)', r'\1 >= \2'),  # >=
        ]

        for pattern, replacement in comparison_patterns:
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                content = new_content
                fixed = True

        return content, fixed

    def fix_directory(self, directory: Path) -> Dict[str, Any]:
        """Fix all Python files in a directory."""
        self.fixes_applied = 0
        self.files_processed = 0

        total_files = 0
        files_fixed = 0
        files_skipped = 0
        results = []

        for py_file in directory.rglob("*.py"):
            if "venv" in str(py_file) or ".git" in str(py_file) or "build" in str(py_file):
                continue

            total_files += 1
            result = self.fix_file(py_file)
            results.append(result)

            if result.get("fixed", False):
                files_fixed += 1
            elif result.get("status") == "skipped":
                files_skipped += 1

        return {
            "total_files": total_files,
            "files_fixed": files_fixed,
            "files_skipped": files_skipped,
            "total_fixes": self.fixes_applied,
            "results": results,
            "status": "completed"
        }


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Conservative, safe code fixer.")
    parser.add_argument("--file", type=str, help="Path to a single Python file to fix.")
    parser.add_argument("--directory", type=str, help="Path to a directory to fix all Python files.")

    args = parser.parse_args()

    fixer = ConservativeFixer()

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            tprint(f"Error: File not found at {file_path}")
            return

        result = fixer.fix_file(file_path)
        if result.get("fixed", False):
            tprint(f"✅ Fixed issues in {file_path}")
            tprint(f"   Fixes applied: {', '.join(result['fixes_applied'])}")
        elif result.get("status") == "skipped":
            tprint(f"⏭️  Skipped {file_path}: {result.get('reason', 'unknown')}")
        else:
            tprint(f"ℹ️  No fixes applied to {file_path}")
        
        # Show detected issues (like empty except blocks)
        issues = result.get("issues_detected", [])
        if issues:
            tprint(f"⚠️  Detected {len(issues)} issues that need attention:")
            for issue in issues:
                tprint(f"   Line {issue['line_number']}: {issue['message']}")
                tprint(f"   Suggestion: {issue['suggestion']}")

    elif args.directory:
        directory_path = Path(args.directory)
        if not directory_path.exists() or not directory_path.is_dir():
            tprint(f"Error: Directory not found at {directory_path}")
            return

        results = fixer.fix_directory(directory_path)
        tprint(f"Fix results for {directory_path}:")
        tprint(f"  Total files processed: {results['total_files']}")
        tprint(f"  Files fixed: {results['files_fixed']}")
        tprint(f"  Files skipped: {results['files_skipped']}")
        tprint(f"  Total fixes applied: {results['total_fixes']}")
        
        # Show files with detected issues
        files_with_issues = [r for r in results['results'] if r.get('issues_detected')]
        if files_with_issues:
            tprint(f"\n⚠️  Files with issues that need attention: {len(files_with_issues)}")
            for result in files_with_issues[:5]:  # Show first 5 files
                issues = result.get('issues_detected', [])
                tprint(f"  {result['file_path']}: {len(issues)} issues")
                for issue in issues[:2]:  # Show first 2 issues per file
                    tprint(f"    Line {issue['line_number']}: {issue['message']}")
            if len(files_with_issues) > 5:
                tprint(f"  ... and {len(files_with_issues) - 5} more files with issues")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
