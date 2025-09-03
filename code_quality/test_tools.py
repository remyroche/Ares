#!/usr/bin/env python3
"""
Test script for the code quality validation tools.

This script creates a small test project with known issues to verify
that the validation tools correctly identify them.
"""

import os
import tempfile
from pathlib import Path


def create_test_project():
    """Create a test project with known code quality issues."""

    # Create temporary directory
    test_dir = Path(tempfile.mkdtemp(prefix="code_quality_test_"))
    print(f"Created test project at: {test_dir}")

    # Create test Python files with known issues

    # File 1: Missing await for async function
    async_issue_file = test_dir / "async_issues.py"
    async_issue_file.write_text('''#!/usr/bin/env python3
"""
Test file with async/await issues.
"""

import asyncio
import aiohttp

async def fetch_data(url):
    """Fetch data from URL."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

def main():
    """Main function with missing await."""
    url = "https://api.example.com/data"
    # ISSUE: Missing await for async function
    result = fetch_data(url)  # Should be: result = await fetch_data(url)
    print(result)

if __name__ == "__main__":
    main()
''')

    # File 2: Undefined function call
    undefined_func_file = test_dir / "undefined_function.py"
    undefined_func_file.write_text('''#!/usr/bin/env python3
"""
Test file with undefined function call.
"""

def existing_function():
    """This function exists."""
    return "Hello"

def main():
    """Main function calling undefined function."""
    # ISSUE: Calling undefined function
    result = nonexistent_function()  # This function is not defined
    print(result)

    # This call is fine
    result2 = existing_function()
    print(result2)
''')

    # File 3: Missing docstrings and other style issues
    style_issues_file = test_dir / "style_issues.py"
    style_issues_file.write_text('''#!/usr/bin/env python3
"""
Test file with style and documentation issues.
"""

class TestClass:  # ISSUE: Missing docstring
    def __init__(self):
        self.value = 42

    def method_without_docstring(self):  # ISSUE: Missing docstring
        return self.value * 2

def function_with_many_args(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8):  # ISSUE: Too many arguments
    """Function with too many arguments."""
    return arg1 + arg2 + arg3 + arg4 + arg5 + arg6 + arg7 + arg8

def function_with_magic_numbers():
    """Function with magic numbers."""
    # ISSUE: Magic numbers
    timeout = 5000  # Should be a named constant
    retry_count = 3  # Should be a named constant
    return timeout * retry_count

def function_with_trailing_whitespace():
    """Function with trailing whitespace."""
    return "test"
''')

    # File 4: Import issues
    import_issues_file = test_dir / "import_issues.py"
    import_issues_file.write_text('''#!/usr/bin/env python3
"""
Test file with import issues.
"""

import os
import sys
from pathlib import Path

# ISSUE: Unused import
import json  # This import is never used

def main():
    """Main function."""
    path = Path("test")
    print(path.exists())
''')

    # File 5: Security issues
    security_issues_file = test_dir / "security_issues.py"
    security_issues_file.write_text('''#!/usr/bin/env python3
"""
Test file with security issues.
"""

import sqlite3

def unsafe_query(user_input):
    """Function with potential SQL injection."""
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    # ISSUE: Potential SQL injection
    query = "SELECT * FROM users WHERE name = '" + user_input + "'"
    cursor.execute(query)  # Should use parameterized queries

    return cursor.fetchall()

def hardcoded_secret():
    """Function with hardcoded secret."""
    # ISSUE: Hardcoded secret
    api_key = "sk-1234567890abcdef"  # This should be in environment variables
    return api_key
''')

    return test_dir


def run_validation_tools(test_dir):
    """Run the validation tools on the test project."""

    print("\n" + "="*60)
    print("RUNNING CODE QUALITY VALIDATION TOOLS")
    print("="*60)

    # Change to the test directory
    original_dir = os.getcwd()
    os.chdir(test_dir)

    try:
        # Run the comprehensive review
        print("\n1. Running comprehensive code review...")
        os.system(f"python {Path(__file__).parent}/comprehensive_code_review.py --project-root . --output comprehensive_report.json")

        # Run the function validator
        print("\n2. Running function validator...")
        os.system(f"python {Path(__file__).parent}/function_validator.py --project-root . --output function_report.json")

        # Run the runner script
        print("\n3. Running runner script...")
        os.system(f"python {Path(__file__).parent}/run_validation.py --mode both --output-dir ./reports")

        # Check if reports were generated
        print("\n" + "="*60)
        print("VALIDATION RESULTS")
        print("="*60)

        report_files = [
            "comprehensive_report.json",
            "function_report.json",
            "reports/comprehensive_review_*.json",
            "reports/function_validation_*.json",
        ]

        for pattern in report_files:
            import glob
            files = glob.glob(pattern)
            for file in files:
                if os.path.exists(file):
                    print(f"✅ Generated: {file}")
                    # Try to read and show summary
                    try:
                        with open(file) as f:
                            import json
                            data = json.load(f)
                            if "summary" in data:
                                summary = data["summary"]
                                print(f"   - Files processed: {summary.get('files_processed', 'N/A')}")
                                print(f"   - Total issues: {summary.get('total_issues', 'N/A')}")
                                print(f"   - Errors: {summary.get('errors', 'N/A')}")
                                print(f"   - Warnings: {summary.get('warnings', 'N/A')}")
                    except Exception as e:
                        print(f"   - Error reading report: {e}")
                else:
                    print(f"❌ Missing: {pattern}")

        # Show summary files
        print("\nSummary files:")
        summary_files = glob.glob("reports/*_summary.txt")
        for file in summary_files:
            if os.path.exists(file):
                print(f"✅ Summary: {file}")
                # Show first few lines
                try:
                    with open(file) as f:
                        lines = f.readlines()[:10]
                        print("   First 10 lines:")
                        for line in lines:
                            print(f"   {line.rstrip()}")
                except Exception as e:
                    print(f"   - Error reading summary: {e}")

    finally:
        # Change back to original directory
        os.chdir(original_dir)

    return test_dir


def main():
    """Main test function."""
    print("Code Quality Tools Test")
    print("=" * 40)

    # Create test project
    test_dir = create_test_project()

    try:
        # Run validation tools
        run_validation_tools(test_dir)

        print(f"\nTest completed! Test project created at: {test_dir}")
        print("You can manually inspect the generated reports to verify the tools work correctly.")
        print("To clean up, delete the test directory when you're done.")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

    return test_dir


if __name__ == "__main__":
    test_project_dir = main()
