#!/usr/bin/env python3
"""
Example usage of Code Quality Tools with Ruff integration and SequentialFixer.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from code_quality import (
    SequentialFixer, 
    AutoFixer, 
    SyntaxValidator,
    LinterAnalyzer,
    get_default_config
)


def example_sequential_fix():
    """Example of using the SequentialFixer pipeline."""
    print("="*60)
    print("EXAMPLE: Sequential Auto-Fix Pipeline")
    print("="*60)
    
    # Create a SequentialFixer instance
    fixer = SequentialFixer()
    
    # Example 1: Fix a single file
    print("\n1. Fixing a single file:")
    try:
        # This would be a real file path in practice
        results = fixer.run_pipeline(
            target="example_file.py",
            output_dir="reports/",
            create_backups=True
        )
        print(f"Pipeline completed with status: {results['summary']['overall_status']}")
    except FileNotFoundError:
        print("Example file not found - this is expected in the demo")
    
    # Example 2: Fix a directory
    print("\n2. Fixing a directory:")
    try:
        results = fixer.run_pipeline(
            target="example_directory/",
            output_dir="reports/",
            create_backups=True
        )
        print(f"Pipeline completed with status: {results['summary']['overall_status']}")
    except FileNotFoundError:
        print("Example directory not found - this is expected in the demo")
    
    # Example 3: Fix multiple specific files
    print("\n3. Fixing multiple specific files:")
    try:
        results = fixer.run_pipeline(
            target=["file1.py", "file2.py", "file3.py"],
            output_dir="reports/",
            create_backups=True
        )
        print(f"Pipeline completed with status: {results['summary']['overall_status']}")
    except FileNotFoundError:
        print("Example files not found - this is expected in the demo")


def example_ruff_integration():
    """Example of using Ruff in the auto-fixer."""
    print("\n" + "="*60)
    print("EXAMPLE: Ruff Integration in AutoFixer")
    print("="*60)
    
    # Create an AutoFixer instance
    fixer = AutoFixer()
    
    # Example: Fix a single file with Ruff
    print("\nFixing a single file with Ruff:")
    try:
        results = fixer.fix_file("example_file.py")
        print("Auto-fix completed!")
        
        # Check if Ruff was successful
        if "ruff" in results:
            ruff_result = results["ruff"]
            print(f"Ruff status: {ruff_result['status']}")
            if ruff_result['status'] == 'success':
                print("Ruff formatting and checking completed successfully!")
            elif ruff_result['status'] == 'skipped':
                print("Ruff not available - install with: pip install ruff")
            else:
                print(f"Ruff encountered issues: {ruff_result.get('error', 'Unknown error')}")
                
    except FileNotFoundError:
        print("Example file not found - this is expected in the demo")


def example_individual_tools():
    """Example of using individual analysis tools."""
    print("\n" + "="*60)
    print("EXAMPLE: Individual Analysis Tools")
    print("="*60)
    
    # Example 1: Syntax validation
    print("\n1. Syntax validation:")
    try:
        validator = SyntaxValidator()
        results = validator.validate_directory("example_directory/")
        print(f"Syntax validation completed. Valid files: {results['summary']['valid_files']}")
    except FileNotFoundError:
        print("Example directory not found - this is expected in the demo")
    
    # Example 2: Linter analysis
    print("\n2. Linter analysis:")
    try:
        linter = LinterAnalyzer()
        results = linter.analyze_directory("example_directory/")
        print(f"Linter analysis completed. Total issues: {results['total_issues']}")
    except FileNotFoundError:
        print("Example directory not found - this is expected in the demo")


def example_configuration():
    """Example of custom configuration."""
    print("\n" + "="*60)
    print("EXAMPLE: Custom Configuration")
    print("="*60)
    
    # Get default configuration
    config = get_default_config()
    
    # Customize the configuration
    config.auto_fix.tools = ["black", "isort", "ruff"]  # Use Ruff instead of autopep8
    config.auto_fix.max_line_length = 100  # Custom line length
    config.auto_fix.aggressive = True  # Enable aggressive fixing
    
    config.analysis.linters = ["flake8", "ruff", "mypy"]  # Include Ruff in linters
    config.analysis.complexity_threshold = 15  # Higher complexity threshold
    
    print("Custom configuration created:")
    print(f"  Auto-fix tools: {config.auto_fix.tools}")
    print(f"  Max line length: {config.auto_fix.max_line_length}")
    print(f"  Aggressive mode: {config.auto_fix.aggressive}")
    print(f"  Linters: {config.analysis.linters}")
    print(f"  Complexity threshold: {config.analysis.complexity_threshold}")
    
    # Use the custom configuration
    fixer = SequentialFixer(config)
    print("\nSequentialFixer created with custom configuration!")


def main():
    """Run all examples."""
    print("CODE QUALITY TOOLS - EXAMPLE USAGE")
    print("This script demonstrates the new features:")
    print("1. SequentialFixer pipeline")
    print("2. Ruff integration")
    print("3. Individual tool usage")
    print("4. Custom configuration")
    
    try:
        example_sequential_fix()
        example_ruff_integration()
        example_individual_tools()
        example_configuration()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())