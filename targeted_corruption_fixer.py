#!/usr/bin/env python3
"""
Targeted Corruption Fixer - Specialized fixer for specific corruption patterns
found in the codebase.

This fixer is designed to handle the specific corruption patterns we've identified
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


class TargetedCorruptionFixer:
    """
    A targeted fixer for specific corruption patterns found in the codebase.
    """

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.stats = {
            "files_processed": 0,
            "files_fixed": 0,
            "total_fixes": 0,
            "fixes_by_type": {
                "typing_imports": 0,
                "assignment_operators": 0,
                "pass_patterns": 0,
                "function_definitions": 0,
                "decorators": 0,
                "git_conflicts": 0,
                "complex_patterns": 0,
                "complex_imports": 0,
                "decorator_imports": 0,
                "function_placeholders": 0,
                "await_fixes": 0,
                "remaining_patterns": 0,
            },
        }

        # Specific patterns found in the codebase
        self.fix_patterns = {
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
            "assignment_operators": [
                # Fix: sys.path.insert(0 = str(project_root))
                (
                    r"sys\.path\.insert\s*\(\s*(\d+)\s*=\s*([^)]+)\)",
                    r"sys.path.insert(\1, \2)",
                ),
                # Fix: hasattr(obj = 'attr')
                (
                    r'hasattr\s*\(\s*(\w+)\s*=\s*[\'"](\w+)[\'"]\s*\)',
                    r"hasattr(\1, \2)",
                ),
                # Fix: comprehensive_data_validation = handle_errors + memory_efficient
                (r"(\w+)\s*=\s*(\w+)\s*\+\s*(\w+)", r"\1 = \2 + \3"),
            ],
            "pass_patterns": [
                # Fix: passpasspass
                (r"passpasspass", r"pass"),
                # Fix: pasself.
                (r"pasself\.", r"pass\n        self."),
                # Fix: pass#
                (r"pass#", r"pass\n        #"),
                # Fix: passtry:
                (r"passtry:", r"pass\n        try:"),
                # Fix: passawait
                (r"passawait", r"pass\n        await"),
                # Fix: pass followed by any word
                (r"pass(\w+)", r"pass\n        \1"),
            ],
            "function_definitions": [
                # Fix: def __init__(self: config: dict[str = Any])
                (
                    r"def\s+(\w+)\s*\(([^)]*:\s*\w+\s*:\s*\w+[^)]*)\)",
                    self._fix_function_definition,
                ),
                # Fix: def __init__(self, config: dict[str = Any])
                (
                    r"def\s+(\w+)\s*\(([^)]*:\s*\w+\s*=\s*\w+[^)]*)\)",
                    self._fix_function_definition,
                ),
            ],
            "decorators": [
                # Fix: @handle_errors(exceptions=(Exception,), default_return, False)
                (
                    r"@(\w+)\s*\(\s*([^)]*default_return\s*,\s*False[^)]*)\)",
                    self._fix_decorator,
                ),
                # Fix: @handle_errors(exceptions=(Exception,), default_return = False)
                (
                    r"@(\w+)\s*\(\s*([^)]*default_return\s*=\s*False[^)]*)\)",
                    self._fix_decorator,
                ),
            ],
            "git_conflicts": [
                # Remove git conflict markers
                (r"<<<<<<<.*?\n(.*?)\n======\n(.*?)\n>>>>>>>.*?\n", r"\1\n"),
                (r"<<<<<<<.*?\n", r""),
                (r"======\n", r""),
                (r">>>>>>>.*?\n", r""),
            ],
            "complex_patterns": [
                # Fix: sr_config["sr_breakout_predictor"], sr_config.get("sr_breakout_predictor", {})
                (
                    r'(\w+)\["(\w+)"\],\s*\1\.get\("(\w+)",\s*\{\}\)',
                    r'\1["\2"] = \1.get("\3", {})',
                ),
                # Fix: sr_config["sr_breakout_predictor"]["enable_detailed_reporting"], True
                (r'(\w+)\["(\w+)"\]\["(\w+)"\],\s*(\w+)', r'\1["\2"]["\3"] = \4'),
                # Fix: if hasattr(self.sr_data_integration = 'initialize'):
                (
                    r'if\s+hasattr\s*\(\s*(\w+\.\w+)\s*=\s*[\'"](\w+)[\'"]\s*\):',
                    r"if hasattr(\1, \2):",
                ),
            ],
            "complex_imports": [
                # Fix complex import statements with multiple equals and plus operators
                (
                    r"from\s+(\S+)\s+import\s+([^=]+)\s*=\s*([^=]+)\s*\+\s*([^=]+)\s*=\s*([^=]+)",
                    r"from \1 import \2, \3, \4, \5",
                ),
                (
                    r"from\s+(\S+)\s+import\s+([^=]+)\s*\+\s*([^=]+)\s*=\s*([^=]+)",
                    r"from \1 import \2, \3, \4",
                ),
                (
                    r"from\s+(\S+)\s+import\s+([^=]+)\s*=\s*([^=]+)",
                    r"from \1 import \2, \3",
                ),
                # Fix: SRDataIntegrationSimple = create_sr_data_integration_simple
                (
                    r"from\s+(\S+)\s+import\s+(\w+)\s*=\s*(\w+)",
                    r"from \1 import \2, \3",
                ),
            ],
            "decorator_imports": [
                # Fix complex decorator imports with multiple equals and plus operators
                (
                    r"comprehensive_data_validation\s*=\s*handle_errors\s*\+\s*memory_efficient\s*=\s*resource_monitor\s*=\s*secure_data_processing\s*=\s*validate_data_structure\s*\+\s*with_tracing_span\s*=\s*quality_gate\s*=\s*monitor_feature_engineering\s*=\s*ensure_data_integrity\s*+\s*monitor_step_execution\s*=\s*secure_step_execution\s*=\s*validate_pipeline_step",
                    r"comprehensive_data_validation, handle_errors, memory_efficient, resource_monitor, secure_data_processing, validate_data_structure, with_tracing_span, quality_gate, monitor_feature_engineering, ensure_data_integrity, monitor_step_execution, secure_step_execution, validate_pipeline_step",
                ),
                # Fix: with_enhanced_mlflow_logging = log_step_report + create_detailed_step_report = log_step_metrics = log_step_artifact_with_standardized_name
                (
                    r"with_enhanced_mlflow_logging\s*=\s*log_step_report\s*\+\s*create_detailed_step_report\s*=\s*log_step_metrics\s*=\s*log_step_artifact_with_standardized_name",
                    r"with_enhanced_mlflow_logging, log_step_report, create_detailed_step_report, log_step_metrics, log_step_artifact_with_standardized_name",
                ),
            ],
            "function_placeholders": [
                # Fix function placeholders with ...
                (
                    r"def\s+(\w+)\s*\(\.\.\.\)\s*->\s*\.\.\.:",
                    r"def \1(self):\n        pass",
                ),
            ],
            "await_fixes": [
                # Fix await statements that are missing proper structure
                (r"await\s+(\w+)\s*\(\s*([^)]+)\s*\)", r"await \1(\2)"),
            ],
            "remaining_patterns": [
                # Fix remaining complex patterns
                (r'pass"""([^"]+)"""', r'"""\1"""'),
                (r"pass([^#\n]+)", r"pass\n        \1"),
                (r"(\w+)\s*=\s*(\w+)\s*\+\s*(\w+)\s*=\s*(\w+)", r"\1 = \2 + \3"),
                (r"(\w+)\s*,\s*(\w+)\s*=\s*(\w+)", r"\1, \2"),
                (r"(\w+)\s*=\s*(\w+)\s*,\s*(\w+)", r"\1, \2"),
            ],
        }

    def _fix_function_definition(self, match) -> str:
        """Fix malformed function definitions."""
        func_name = match.group(1)
        params = match.group(2)

        # Fix parameter syntax issues
        # Replace : with , in parameter lists
        fixed_params = re.sub(r":\s*(\w+)\s*:", r", \1: ", params)
        fixed_params = re.sub(r":\s*(\w+)\s*=", r", \1=", fixed_params)

        return f"def {func_name}({fixed_params})"

    def _fix_decorator(self, match) -> str:
        """Fix malformed decorators."""
        decorator_name = match.group(1)
        args = match.group(2)

        # Fix common decorator issues
        args = re.sub(r"default_return\s*,\s*False", r"default_return=False", args)
        args = re.sub(r"default_return\s*=\s*False", r"default_return=False", args)

        return f"@{decorator_name}({args})"

    def _is_safe_to_fix(
        self, filepath: str, original_content: str, fixed_content: str
    ) -> Tuple[bool, str]:
        """
        Validate that a fix is safe to apply.
        Returns (is_safe, reason)
        """
        # Check if content changed
        if original_content == fixed_content:
            return True, "No changes made"

        # Check if we're not removing too much content
        if len(fixed_content) < len(original_content) * 0.8:
            return False, "Fix would remove too much content (>20%)"

        # Check if we're not adding too much content
        if len(fixed_content) > len(original_content) * 1.5:
            return False, "Fix would add too much content (>50%)"

        # Check for only the most dangerous patterns
        dangerous_patterns = [
            r"======",  # Git conflict markers
            r"<<<<<<<",  # Git conflict markers
            r">>>>>>>",  # Git conflict markers
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, fixed_content):
                return False, f"Fix would create dangerous pattern: {pattern}"

        return True, "Fix appears safe"

    def _apply_fixes(self, content: str, filepath: str) -> Tuple[str, Dict[str, int]]:
        """
        Apply fixes to the content.
        Returns (fixed_content, fixes_applied)
        """
        original_content = content
        fixes_applied = {k: 0 for k in self.fix_patterns.keys()}
        changes_log = []

        # Apply each pattern type
        for pattern_type, patterns in self.fix_patterns.items():
            for pattern, replacement in patterns:
                if callable(replacement):
                    # Handle function-based replacements
                    new_content = re.sub(
                        pattern, replacement, content, flags=re.MULTILINE
                    )
                else:
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
        print("TARGETED CORRUPTION FIXER SUMMARY")
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
        summary.append("TARGETED CORRUPTION FIXER RESULTS")
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
        description="Targeted Corruption Fixer - Fix specific corruption patterns found in the codebase"
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
    fixer = TargetedCorruptionFixer(dry_run=args.dry_run)

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
