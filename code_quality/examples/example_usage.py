#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Example usage of Code Quality Tools with Ruff integration and SequentialFixer.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import collections

from code_quality import (
    AutoFixer,
    LinterAnalyzer,
    SequentialFixer,
    SyntaxValidator,
    get_default_config,
)


def example_sequential_fix():
    """Example of using the SequentialFixer pipeline."""
    tprint("="*60)
    tprint("EXAMPLE: Sequential Auto-Fix Pipeline")
    tprint("="*60)

    # Create a SequentialFixer instance
    fixer = SequentialFixer()

    # Example 1: Fix a single file
    tprint("\n1. Fixing a single file:")
    try:
        # This would be a real file path in practice
        results = fixer.run_pipeline(
            target="example_file.py",
            output_dir="reports/",
            create_backups = True,
        )
        tprint(f"Pipeline completed with status: {results['summary']['overall_status']}")
    except FileNotFoundError:
        tprint("Example file not found - this is expected in the demo")

    # Example 2: Fix a directory
    tprint("\n2. Fixing a directory:")
    try:
        results = fixer.run_pipeline(
            target="example_directory/",
            output_dir="reports/",
            create_backups = True,
        )
        tprint(f"Pipeline completed with status: {results['summary']['overall_status']}")
    except FileNotFoundError:
        tprint("Example directory not found - this is expected in the demo")

    # Example 3: Fix multiple specific files
    tprint("\n3. Fixing multiple specific files:")
    try:
        results = fixer.run_pipeline(
            target=["file1.py", "file2.py", "file3.py"],
            output_dir="reports/",
            create_backups = True,
        )
        tprint(f"Pipeline completed with status: {results['summary']['overall_status']}")
    except FileNotFoundError:
        tprint("Example files not found - this is expected in the demo")


def example_ruff_integration():
    """Example of using Ruff in the auto-fixer."""
    tprint("\n" + "="*60)
    tprint("EXAMPLE: Ruff Integration in AutoFixer")
    tprint("="*60)

    # Create an AutoFixer instance
    fixer = AutoFixer()

    # Example: Fix a single file with Ruff
    tprint("\nFixing a single file with Ruff:")
    try:
        results = fixer.fix_file("example_file.py")
        tprint("Auto-fix completed!")

        # Check if Ruff was successful
        if "ruff" in results:
            ruff_result = results["ruff"]
            tprint(f"Ruff status: {ruff_result['status']}")
            if ruff_result["status"] == "success":
                tprint("Ruff formatting and checking completed successfully!")
            elif ruff_result["status"] == "skipped":
                tprint("Ruff not available - install with: pip install ruff")
            else:
                tprint(f"Ruff encountered issues: {ruff_result.get('error', 'Unknown error')}")

    except FileNotFoundError:
        tprint("Example file not found - this is expected in the demo")


def example_individual_tools():
    """Example of using individual analysis tools."""
    tprint("\n" + "="*60)
    tprint("EXAMPLE: Individual Analysis Tools")
    tprint("="*60)

    # Example 1: Syntax validation
    tprint("\n1. Syntax validation:")
    try:
        validator = SyntaxValidator()
        results = validator.validate_directory("example_directory/")
        tprint(f"Syntax validation completed. Valid files: {results['summary']['valid_files']}")
    except FileNotFoundError:
        tprint("Example directory not found - this is expected in the demo")

    # Example 2: Linter analysis
    tprint("\n2. Linter analysis:")
    try:
        linter = LinterAnalyzer()
        results = linter.analyze_directory("example_directory/")
        tprint(f"Linter analysis completed. Total issues: {results['total_issues']}")
    except FileNotFoundError:
        tprint("Example directory not found - this is expected in the demo")


def example_configuration():
    """Example of custom configuration."""
    tprint("\n" + "="*60)
    tprint("EXAMPLE: Custom Configuration")
    tprint("="*60)

    # Get default configuration
    config = get_default_config()

    # Customize the configuration
    config.auto_fix.tools = ["black", "isort", "ruff"]  # Use Ruff instead of autopep8
    config.auto_fix.max_line_length = 100  # Custom line length
    config.auto_fix.aggressive = True  # Enable aggressive fixing

    config.analysis.linters = ["flake8", "ruff", "mypy"]  # Include Ruff in linters
    config.analysis.complexity_threshold = 15  # Higher complexity threshold

    tprint("Custom configuration created:")
    tprint(f"  Auto-fix tools: {config.auto_fix.tools}")
    tprint(f"  Max line length: {config.auto_fix.max_line_length}")
    tprint(f"  Aggressive mode: {config.auto_fix.aggressive}")
    tprint(f"  Linters: {config.analysis.linters}")
    tprint(f"  Complexity threshold: {config.analysis.complexity_threshold}")

    # Use the custom configuration
    SequentialFixer(config)
    tprint("\nSequentialFixer created with custom configuration!")


def main():
    """Run all examples."""
    tprint("CODE QUALITY TOOLS - EXAMPLE USAGE")
    tprint("This script demonstrates the new features:")
    tprint("1. SequentialFixer pipeline")
    tprint("2. Ruff integration")
    tprint("3. Individual tool usage")
    tprint("4. Custom configuration")

    try:
        example_sequential_fix()
        example_ruff_integration()
        example_individual_tools()
        example_configuration()

        tprint("\n" + "="*60)
        tprint("All examples completed successfully!")
        tprint("="*60)

    except Exception as e:
        tprint(f"\nError running examples: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
