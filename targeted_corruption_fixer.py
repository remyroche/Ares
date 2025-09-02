#!/usr/bin/env python3
"""
Conservative Targeted Corruption Fixer - Safe fixer for specific corruption patterns
found in the codebase.

This fixer is designed to handle only the most obvious and safe corruption patterns
while maintaining safety and not creating new problems.
"""

import re
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("targeted_fixer.log"),
    ],
)
logger = logging.getLogger(__name__)


class ConservativeTargetedCorruptionFixer:
    """
    A conservative targeted fixer for specific corruption patterns found in the codebase.
    Only applies the safest fixes that are very unlikely to cause new problems.
    """

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.stats = {
            "files_processed": 0,
            "files_fixed": 0,
            "total_fixes": 0,
            "fixes_by_type": {
                "git_conflicts": 0,
                "placeholder_fixes": 0,
                "pass_patterns": 0,
                "string_literals": 0,
                "typing_imports": 0,
                "import_statements": 0,
            },
        }

        # ONLY THE SAFEST PATTERNS - ordered by safety
        self.fix_patterns = {
            # SAFEST PATTERNS - These are very unlikely to cause issues
            "git_conflicts": [
                # Remove git conflict markers
                (r"<<<<<<<.*?\n(.*?)\n======\n(.*?)\n>>>>>>>.*?\n", r"\1\n"),
                (r"<<<<<<<.*?\n", r""),
                (r"======\n", r""),
                (r">>>>>>>.*?\n", r""),
            ],
            
            # VERY SAFE PATTERNS - Simple text replacements
            "placeholder_fixes": [
                # Fix: """..."""
                (r'"""\.\.\."""', r'"""Implementation placeholder - needs specific logic"""'),
                # Fix: pass...
                (r"pass\.\.\.", r"pass"),
            ],
            
            # SAFE PATTERNS - Well-defined replacements
            "pass_patterns": [
                # Fix: passpasspass
                (r"passpasspass", r"pass"),
                # Fix: passpass
                (r"passpass", r"pass"),
                # Fix: pass followed by specific keywords (very safe)
                (r"passself\.", r"pass\n        self."),
                (r"passlogger\.", r"pass\n        logger."),
                (r"passtry:", r"pass\n        try:"),
                (r"passexcept", r"pass\n        except"),
                (r"passif", r"pass\n        if"),
                (r"passfor", r"pass\n        for"),
                (r"passwhile", r"pass\n        while"),
                (r"passdef", r"pass\n        def"),
                (r"passclass", r"pass\n        class"),
                (r"passimport", r"pass\n        import"),
                (r"passfrom", r"pass\n        from"),
                (r"passreturn", r"pass\n        return"),
                (r"passraise", r"pass\n        raise"),
                (r"passbreak", r"pass\n        break"),
                (r"passcontinue", r"pass\n        continue"),
                (r"passawait", r"pass\n        await"),
            ],
            
            # MODERATELY SAFE PATTERNS
            "string_literals": [
                # Fix: pass"""docstring"""
                (r'pass"""([^"]+)"""', r'"""\1"""'),
                # Fix: pass'docstring'
                (r"pass'([^']+)'", r"'\1'"),
                # Fix: pass"docstring"
                (r'pass"([^"]+)"', r'"\1"'),
            ],
            
            # IMPORT PATTERNS - Generally safe
            "typing_imports": [
                # Fix: from typing import Any = Dict + List = Optional
                (
                    r"from typing import (\w+)\s*=\s*(\w+)\s*\+\s*(\w+)\s*=\s*(\w+)",
                    r"from typing import \1, \2, \3, \4",
                ),
                # Fix: from typing import Any = Dict + List
                (
                    r"from typing import (\w+)\s*=\s*(\w+)\s*\+\s*(\w+)",
                    r"from typing import \1, \2, \3",
                ),
                # Fix: dict[str = Any]
                (r"dict\[(\w+)\s*=\s*(\w+)\]", r"dict[\1, \2]"),
                # Fix: List[str = Any]
                (r"List\[(\w+)\s*=\s*(\w+)\]", r"List[\1, \2]"),
                # Fix: Tuple[str = Any]
                (r"Tuple\[(\w+)\s*=\s*(\w+)\]", r"Tuple[\1, \2]"),
            ],
            
            "import_statements": [
                # Fix: import statements with equals
                (
                    r"import\s+(\w+)\s*=\s*(\w+)",
                    r"import \1, \2",
                ),
                # Fix: from import with equals
                (
                    r"from\s+(\S+)\s+import\s+(\w+)\s*=\s*(\w+)",
                    r"from \1 import \2, \3",
                ),
                # Fix: import statements with plus
                (
                    r"import\s+(\w+)\s*\+\s*(\w+)",
                    r"import \1, \2",
                ),
                # Fix: from import with plus
                (
                    r"from\s+(\S+)\s+import\s+(\w+)\s*\+\s*(\w+)",
                    r"from \1 import \2, \3",
                ),
            ],
        }

    def _is_safe_to_fix(
        self, filepath: str, original_content: str, fixed_content: str
    ) -> Tuple[bool, str]:
        """
        Very conservative validation that a fix is safe to apply.
        Returns (is_safe, reason)
        """
        # Check if content changed
        if original_content == fixed_content:
            return True, "No changes made"

        # Check if we're not removing too much content
        if len(fixed_content) < len(original_content) * 0.95:
            return False, "Fix would remove too much content (>5%)"

        # Check if we're not adding too much content
        if len(fixed_content) > len(original_content) * 1.2:
            return False, "Fix would add too much content (>20%)"

        # Check for dangerous patterns that could indicate corruption
        dangerous_patterns = [
            r"======",  # Git conflict markers
            r"<<<<<<<",  # Git conflict markers
            r">>>>>>>",  # Git conflict markers
            r"^\s*:\s*$",  # Lone colons
            r"^\s*,\s*$",  # Lone commas
            r"^\s*=\s*$",  # Lone equals
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, fixed_content, re.MULTILINE):
                return False, f"Fix would create dangerous pattern: {pattern}"

        # Check for balanced parentheses and braces
        if fixed_content.count('(') != fixed_content.count(')') or \
           fixed_content.count('[') != fixed_content.count(']') or \
           fixed_content.count('{') != fixed_content.count('}'):
            return False, "Fix would create unbalanced brackets/parentheses"

        # Check for obvious syntax issues
        if re.search(r'^\s*[^#\n]*\s*=\s*[^#\n]*\s*=\s*[^#\n]*\s*$', fixed_content, re.MULTILINE):
            return False, "Fix would create double equals assignment"

        return True, "Fix appears safe"

    def _apply_fixes(self, content: str, filepath: str) -> Tuple[str, Dict[str, int]]:
        """
        Apply fixes to the content in order of safety.
        Returns (fixed_content, fixes_applied)
        """
        original_content = content
        fixes_applied = {k: 0 for k in self.fix_patterns.keys()}
        changes_log = []

        # Apply each pattern type in order (safest first)
        for pattern_type, patterns in self.fix_patterns.items():
            for pattern, replacement in patterns:
                # Handle string-based replacements
                new_content = re.sub(
                    pattern, replacement, content, flags=re.MULTILINE
                )

                if new_content != content:
                    # Validate the fix is safe
                    is_safe, reason = self._is_safe_to_fix(
                        filepath, original_content, new_content
                    )
                    if is_safe:
                        # Log the specific change
                        change_info = self._log_specific_change(
                            content, new_content, pattern, replacement, pattern_type
                        )
                        changes_log.append(change_info)

                        content = new_content
                        fixes_applied[pattern_type] += 1
                        logger.info(
                            f"Applied {pattern_type} fix: {pattern} -> {replacement}"
                        )
                    else:
                        logger.warning(f"Skipped unsafe fix: {reason}")

        # Log all changes made
        if changes_log:
            logger.info(f"\n📝 CHANGES MADE IN {filepath}:")
            for i, change in enumerate(changes_log, 1):
                logger.info(f"  {i}. {change}")
            logger.info("")

        return content, fixes_applied

    def _log_specific_change(
        self,
        old_content: str,
        new_content: str,
        pattern: str,
        replacement: str,
        pattern_type: str,
    ) -> str:
        """Log the specific change made, showing before/after."""
        old_lines = old_content.split("\n")
        new_lines = new_content.split("\n")

        # Find the first line that changed
        for i, (old_line, new_line) in enumerate(zip(old_lines, new_lines)):
            if old_line != new_line:
                line_num = i + 1
                # Truncate long lines for readability
                old_display = (
                    old_line[:100] + "..." if len(old_line) > 100 else old_line
                )
                new_display = (
                    new_line[:100] + "..." if len(new_line) > 100 else new_line
                )

                return f"{pattern_type}: Line {line_num} - '{old_display}' → '{new_display}'"

        return f"{pattern_type}: Pattern '{pattern}' → '{replacement}'"

    def fix_file(self, filepath: str) -> bool:
        """
        Fix a single file.
        Returns True if fixes were applied, False otherwise.
        """
        try:
            logger.info(f"Processing file: {filepath}")

            # Read the file
            with open(filepath, "r", encoding="utf-8") as f:
                original_content = f.read()

            # Skip empty files
            if not original_content.strip():
                logger.warning(f"Skipping empty file: {filepath}")
                return False

            # Apply fixes
            fixed_content, fixes_applied = self._apply_fixes(original_content, filepath)

            # Check if any fixes were applied
            total_fixes = sum(fixes_applied.values())
            if total_fixes == 0:
                logger.info(f"No fixes needed for: {filepath}")
                return False

            # Final validation
            is_safe, reason = self._is_safe_to_fix(
                filepath, original_content, fixed_content
            )
            if not is_safe:
                logger.error(f"Final validation failed for {filepath}: {reason}")
                return False

            # Apply fixes if not in dry run mode
            if not self.dry_run:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(fixed_content)
                logger.info(f"Applied {total_fixes} fixes to: {filepath}")
            else:
                logger.info(f"[DRY RUN] Would apply {total_fixes} fixes to: {filepath}")

            # Update statistics
            self.stats["files_processed"] += 1
            if total_fixes > 0:
                self.stats["files_fixed"] += 1
                self.stats["total_fixes"] += total_fixes
                for fix_type, count in fixes_applied.items():
                    if count > 0:
                        self.stats["fixes_by_type"][fix_type] += count

            return True

        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            return False

    def fix_directory(self, directory: str) -> None:
        """Fix all Python files in a directory."""
        directory_path = Path(directory)
        if not directory_path.exists():
            logger.error(f"Directory does not exist: {directory}")
            return

        logger.info(f"Starting to fix Python files in: {directory}")

        # Find all Python files
        python_files = list(directory_path.rglob("*.py"))
        logger.info(f"Found {len(python_files)} Python files")

        # Process each file
        for filepath in python_files:
            if self._should_process_file(filepath):
                self.fix_file(str(filepath))

        logger.info("Directory processing complete")

    def _should_process_file(self, filepath: Path) -> bool:
        """Check if a file should be processed."""
        # Skip certain directories
        skip_dirs = {
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "env",
            "node_modules",
            ".pytest_cache",
            ".ruff_cache",
        }

        for part in filepath.parts:
            if part in skip_dirs:
                return False

        # Skip certain file patterns
        skip_patterns = [
            r"\.pyc$",
            r"\.pyo$",
            r"\.pyd$",
            r"\.bak$",
            r"\.backup$",
            r"\.orig$",
        ]

        for pattern in skip_patterns:
            if re.search(pattern, str(filepath)):
                return False

        return True

    def print_summary(self) -> None:
        """Print a summary of the fixes applied."""
        print("\n" + "=" * 60)
        print("CONSERVATIVE TARGETED CORRUPTION FIXER SUMMARY")
        print("=" * 60)
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Files fixed: {self.stats['files_fixed']}")
        print(f"Total fixes applied: {self.stats['total_fixes']}")
        print("\nFixes by type:")
        for fix_type, count in self.stats["fixes_by_type"].items():
            if count > 0:
                print(f"  {fix_type}: {count}")

        if self.stats["files_fixed"] > 0:
            print(f"\n🎯 Successfully fixed {self.stats['files_fixed']} files")
            print(
                f"📊 Average fixes per file: {self.stats['total_fixes'] / self.stats['files_fixed']:.1f}"
            )

        print("=" * 60)

    def get_fix_summary(self) -> str:
        """Get a detailed summary of fixes for reporting."""
        summary = []
        summary.append("CONSERVATIVE TARGETED CORRUPTION FIXER RESULTS")
        summary.append("=" * 50)
        summary.append(f"Files processed: {self.stats['files_processed']}")
        summary.append(f"Files fixed: {self.stats['files_fixed']}")
        summary.append(f"Total fixes applied: {self.stats['total_fixes']}")
        summary.append("")
        summary.append("Fixes by type:")
        for fix_type, count in self.stats["fixes_by_type"].items():
            if count > 0:
                summary.append(f"  {fix_type}: {count}")
        return "\n".join(summary)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Conservative Targeted Corruption Fixer - Fix only the safest corruption patterns found in the codebase"
    )
    parser.add_argument("target", help="File or directory to fix")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without making changes",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create fixer
    fixer = ConservativeTargetedCorruptionFixer(dry_run=args.dry_run)

    # Process target
    target_path = Path(args.target)
    if target_path.is_file():
        if target_path.suffix == ".py":
            fixer.fix_file(str(target_path))
        else:
            logger.error(f"Target is not a Python file: {target_path}")
    elif target_path.is_dir():
        fixer.fix_directory(str(target_path))
    else:
        logger.error(f"Target does not exist: {target_path}")
        return

    # Print summary
    fixer.print_summary()


if __name__ == "__main__":
    main()
