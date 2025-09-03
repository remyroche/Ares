#!/usr/bin/env python3
"""
Quick Start Script for Code Quality Validation

This script provides a simple way to get started with code quality validation.
Run it from your project root to quickly validate your code.
"""

import os
import sys
from pathlib import Path

# Add the code_quality directory to the path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def quick_validate():
    """Run a quick validation on the current project."""

    print("🚀 Code Quality Validation - Quick Start")
    print("=" * 50)

    # Check if we're in a Python project
    if not any(Path().glob("*.py")):
        print("❌ No Python files found in current directory.")
        print("   Please run this script from your project root.")
        return False

    print("✅ Python project detected!")
    print(f"📁 Current directory: {os.getcwd()}")

    # Create reports directory
    reports_dir = Path("./reports")
    reports_dir.mkdir(exist_ok=True)

    try:
        # Import the function validator
        from function_validator import FunctionValidator

        print("\n🔍 Running function validation...")
        validator = FunctionValidator(".")
        output_file = validator.generate_report(str(reports_dir / "quick_validation.json"))

        print("✅ Function validation completed!")
        print(f"📊 Report: {output_file}")

        # Show summary
        summary_file = output_file.replace(".json", "_summary.txt")
        if os.path.exists(summary_file):
            print(f"📋 Summary: {summary_file}")

            # Display key findings
            with open(summary_file) as f:
                lines = f.readlines()
                for line in lines[:20]:  # Show first 20 lines
                    if line.strip():
                        print(f"   {line.rstrip()}")

        print("\n🎯 Quick validation completed!")
        print(f"📁 Reports saved to: {reports_dir}")
        print("🔧 For comprehensive analysis, run: python code_quality/run_validation.py")

        return True

    except ImportError as e:
        print(f"❌ Error importing validation tools: {e}")
        print("   Make sure you're running from the project root.")
        return False
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        return False


def show_help():
    """Show help information."""
    print("""
Code Quality Validation - Quick Start

Usage:
  python code_quality/quick_start.py

This script will:
1. Detect if you're in a Python project
2. Run a quick function validation
3. Generate a report in ./reports/
4. Show a summary of findings

For more options, see:
  python code_quality/run_validation.py --help
  python code_quality/function_validator.py --help
  python code_quality/comprehensive_code_review.py --help

Documentation: code_quality/README.md
""")


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help", "help"]:
        show_help()
        return

    success = quick_validate()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
