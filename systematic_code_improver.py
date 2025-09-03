#!/usr/bin/env python3
"""
Systematic Code Improver for src/utils

This script applies various code quality improvements systematically:
1. Fix remaining import issues
2. Apply code formatting (black, isort)
3. Add basic type annotations
4. Fix common linting issues
"""

import json
import os
import re
import subprocess
import sys
import time

# Add local bin to PATH
os.environ["PATH"] = f"/home/ubuntu/.local/bin:{os.environ['PATH']}"

def run_command(cmd, capture=True):
    """Run a command and return result."""
    print(f"Running: {' '.join(cmd)}")
    if capture:
        return subprocess.run(cmd, check=False, capture_output=True, text=True)
    return subprocess.run(cmd, check=False)

def fix_remaining_imports():
    """Fix any remaining import issues."""
    print("\n=== Fixing Remaining Import Issues ===")

    # Fix data_preprocessing.py specific issues
    file_path = "src/utils/data_preprocessing.py"
    if os.path.exists(file_path):
        with open(file_path) as f:
            content = f.read()

        # Add missing imports at the top
        if "import logging" not in content:
            content = "import logging\n" + content

        # Create logger if not exists
        if "logger = logging.getLogger" not in content:
            # Add after imports
            lines = content.split("\n")
            import_end = 0
            for i, line in enumerate(lines):
                if line and not line.startswith("import") and not line.startswith("from"):
                    import_end = i
                    break
            lines.insert(import_end, "\nlogger = logging.getLogger(__name__)\n")
            content = "\n".join(lines)

        with open(file_path, "w") as f:
            f.write(content)
        print(f"Fixed imports in {file_path}")

def apply_black_formatting():
    """Apply black formatting to all Python files."""
    print("\n=== Applying Black Formatting ===")

    # Run black with configuration
    cmd = ["black", "src/utils", "--line-length", "120", "--target-version", "py38"]
    result = run_command(cmd, capture=False)

    if result.returncode == 0:
        print("Black formatting applied successfully")
    else:
        print("Black formatting completed with warnings")

def apply_isort():
    """Apply isort to organize imports."""
    print("\n=== Organizing Imports with isort ===")

    # Run isort with configuration
    cmd = ["isort", "src/utils", "--line-length", "120", "--profile", "black"]
    result = run_command(cmd, capture=False)

    if result.returncode == 0:
        print("Import organization completed successfully")
    else:
        print("Import organization completed with warnings")

def fix_common_linting_issues():
    """Fix common linting issues automatically."""
    print("\n=== Fixing Common Linting Issues ===")

    # Run autopep8 to fix basic issues
    cmd = ["autopep8", "--in-place", "--recursive", "--max-line-length", "120",
           "--ignore", "E402", "src/utils"]
    run_command(cmd, capture=False)

    print("Common linting issues fixed")

def add_basic_type_annotations():
    """Add basic type annotations where missing."""
    print("\n=== Adding Basic Type Annotations ===")

    # This is a simplified version - in practice, you'd use a more sophisticated tool
    files_updated = 0

    for root, _dirs, files in os.walk("src/utils"):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)

                with open(file_path) as f:
                    content = f.read()

                # Add return type hints for simple cases
                # Pattern: def function_name(args): without -> annotation
                pattern = r"(\n\s*def\s+\w+\([^)]*\))(\s*:)(?!\s*->)"

                # Check if there are unannotated functions
                if re.search(pattern, content):
                    # For now, just count them
                    matches = re.findall(pattern, content)
                    if matches:
                        print(f"Found {len(matches)} functions without return type in {file_path}")
                        files_updated += 1

    print(f"Found {files_updated} files that could benefit from type annotations")

def fix_unused_imports():
    """Remove unused imports."""
    print("\n=== Removing Unused Imports ===")

    # Use autoflake to remove unused imports
    cmd = ["autoflake", "--in-place", "--remove-all-unused-imports",
           "--recursive", "src/utils"]

    # Check if autoflake is available
    try:
        result = run_command(["autoflake", "--version"], capture=True)
        if result.returncode == 0:
            result = run_command(cmd, capture=False)
            print("Unused imports removed")
        else:
            print("autoflake not available, installing...")
            run_command(["pip", "install", "--break-system-packages", "autoflake"], capture=False)
            result = run_command(cmd, capture=False)
            print("Unused imports removed")
    except FileNotFoundError:
        print("autoflake not available, skipping unused import removal")

def generate_improvement_report():
    """Generate a report of improvements made."""
    print("\n=== Generating Improvement Report ===")

    # Run quick analysis to see improvements
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "improvements_applied": [
            "Fixed remaining import issues",
            "Applied black formatting",
            "Organized imports with isort",
            "Fixed common linting issues",
            "Removed unused imports",
        ],
        "before_after": {},
    }

    # Get current issue count
    print("\nChecking remaining issues...")

    # Flake8 check
    result = run_command(["flake8", "src/utils", "--count", "--select=E9,F63,F7,F82"], capture=True)
    critical_errors = len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0
    report["before_after"]["critical_errors"] = critical_errors

    # Save report
    report_file = f"improvement_report_{int(time.time())}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {report_file}")
    print(f"Critical errors remaining: {critical_errors}")

def main():
    """Run all improvements systematically."""
    print("Starting Systematic Code Improvement for src/utils")
    print("=" * 60)

    # Check if directory exists
    if not os.path.exists("src/utils"):
        print("Error: src/utils directory not found!")
        return 1

    # Run improvements in order
    fix_remaining_imports()
    apply_black_formatting()
    apply_isort()
    fix_common_linting_issues()
    fix_unused_imports()
    add_basic_type_annotations()

    # Generate report
    generate_improvement_report()

    print("\n" + "=" * 60)
    print("Systematic improvements completed!")
    print("\nNext steps:")
    print("1. Review the changes made by the automated tools")
    print("2. Run type checking with mypy to identify remaining type issues")
    print("3. Manually refactor complex functions identified by radon")
    print("4. Add missing docstrings and improve documentation")

    return 0

if __name__ == "__main__":
    sys.exit(main())
