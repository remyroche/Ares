#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Example usage of the enhanced validation tools.

This script demonstrates how to use:
1. Enhanced Validator (for function arguments and data access)
2. Integrated Validator (combining function_validator and enhanced_validator)
"""

import sys
from pathlib import Path

# Add code_quality to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_validator import EnhancedValidator
from integrated_validator import IntegratedValidator


def example_enhanced_validator():
    """Example of using the enhanced validator directly."""
    tprint("=" * 60)
    tprint("ENHANCED VALIDATOR EXAMPLE")
    tprint("=" * 60)

    # Initialize validator
    validator = EnhancedValidator(
        project_root=".",
        exclude_patterns=["__pycache__", "*.pyc", ".git", "venv", "test_*"],
    )

    # Run validation
    tprint("\nRunning enhanced validation...")
    report = validator.validate_project()

    # Display summary
    summary = report["summary"]
    tprint("\nValidation Summary:")
    tprint(f"  Files processed: {summary['files_processed']}")
    tprint(f"  Total issues: {summary['total_issues']}")
    tprint(f"  - Argument mismatches: {summary['argument_mismatches']}")
    tprint(f"  - Unsafe data access: {summary['unsafe_data_access']}")
    tprint(f"  - Missing null checks: {summary['missing_null_checks']}")
    tprint(f"  - Type inconsistencies: {summary['type_inconsistencies']}")

    # Show top issues
    if report["issues"]:
        tprint("\nTop Issues Found:")
        for issue in report["issues"][:5]:
            tprint(f"  {issue['severity'].upper()}: {issue['file_path']}:{issue['line_number']}")
            tprint(f"    {issue['message']}")
            if issue.get("suggestion"):
                tprint(f"    → {issue['suggestion']}")

    return report


def example_integrated_validator():
    """Example of using the integrated validator."""
    tprint("\n" + "=" * 60)
    tprint("INTEGRATED VALIDATOR EXAMPLE")
    tprint("=" * 60)

    # Initialize integrated validator
    validator = IntegratedValidator(
        project_root=".",
        exclude_patterns=["__pycache__", "*.pyc", ".git", "venv", "test_*"],
    )

    # Generate comprehensive reports
    tprint("\nGenerating integrated validation reports...")
    report_files = validator.generate_report(output_dir="code_quality/reports")

    tprint("\nGenerated reports:")
    for report_type, file_path in report_files.items():
        tprint(f"  - {report_type.capitalize()}: {file_path}")

    # Read and display summary
    with open(report_files["summary"]) as f:
        summary_content = f.read()

    tprint("\n" + "-" * 40)
    tprint("SUMMARY PREVIEW:")
    tprint("-" * 40)
    # Show first 30 lines of summary
    summary_lines = summary_content.split("\n")[:30]
    tprint("\n".join(summary_lines))
    if len(summary_content.split("\n")) > 30:
        tprint("... (truncated)")


def example_code_with_issues():
    """Create example code files with various issues for testing."""
    tprint("\n" + "=" * 60)
    tprint("CREATING TEST FILES WITH ISSUES")
    tprint("=" * 60)

    test_dir = Path("code_quality/test_examples")
    test_dir.mkdir(exist_ok = True)

    # Example 1: Function with argument issues
    example1 = '''#!/usr/bin/env python3
"""Example with function argument issues."""

def process_data(data, threshold, normalize = True):
    """Process data with threshold."""
    if data:
        return data * threshold
    return 0

def main():
    # Missing required argument
    result1 = process_data(100)  # Missing 'threshold'

    # Too many arguments
    result2 = process_data(100, 0.5, True, False)  # Extra argument

    # Wrong keyword argument
    result3 = process_data(100, 0.5, normalized = True)  # Wrong keyword

if __name__ == "__main__":
    main()
'''

    with open(test_dir / "argument_issues.py", "w") as f:
        f.write(example1)

    # Example 2: Unsafe data access
    example2 = '''#!/usr/bin/env python3
"""Example with unsafe data access."""

def process_config(config):
    """Process configuration dictionary."""
    # Unsafe dictionary access
    server = config["server"]  # No check if 'server' exists
    port = config["port"]      # No check if 'port' exists

    # Unsafe attribute access
    response = get_response()
    data = response.data       # No null check
    status = response.status   # No null check

    # Unsafe list access
    items = get_items()
    first = items[0]          # No bounds check
    last = items[-1]          # No bounds check

    return {"server": server, "port": port}

def safe_process_config(config):
    """Process configuration safely."""
    # Safe dictionary access
    server = config.get("server", "localhost")
    port = config.get("port", 8080)

    # Safe attribute access
    response = get_response()
    if response is not None:
        data = response.data
        status = response.status

    # Safe list access
    items = get_items()
    if items:
        first = items[0]
        last = items[-1]

    return {"server": server, "port": port}

def get_response():
    """Mock function."""
    return None

def get_items():
    """Mock function."""
    return []
'''

    with open(test_dir / "data_access_issues.py", "w") as f:
        f.write(example2)

    # Example 3: Mixed issues
    example3 = '''#!/usr/bin/env python3
"""Example with mixed validation issues."""

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.data = None

    def process(self, input_data, options):
        """Process input data with options."""
        # Unsafe access without checks
        mode = options["mode"]
        threshold = options["threshold"]

        # Calling method on potentially None object
        result = input_data.transform()

        # Accessing nested attributes unsafely
        value = self.config.settings.default_value

        return result * threshold

    async def async_process(self, data):
        """Async processing method."""
        # Should be awaited but isn't
        result = self.fetch_data(data)
        return result

    async def fetch_data(self, data):
        """Fetch data asynchronously."""
        return data * 2

def use_processor():
    """Use the processor with issues."""
    processor = DataProcessor(None)  # Passing None

    # Missing required arguments
    result = processor.process({"value": 10})  # Missing 'options'

    # Accessing result unsafely
    tprint(result.value)  # result might be None
'''

    with open(test_dir / "mixed_issues.py", "w") as f:
        f.write(example3)

    tprint(f"\nCreated test files in {test_dir}:")
    for file in test_dir.glob("*.py"):
        tprint(f"  - {file.name}")

    return test_dir


def validate_test_files(test_dir):
    """Run validation on the test files."""
    tprint("\n" + "=" * 60)
    tprint("VALIDATING TEST FILES")
    tprint("=" * 60)

    # Use enhanced validator on test directory
    validator = EnhancedValidator(
        project_root = str(test_dir),
        exclude_patterns=["__pycache__", "*.pyc"],
    )

    tprint(f"\nValidating test directory: {test_dir}")
    report = validator.validate_project()

    # Display all issues found
    tprint(f"\nFound {len(report['issues'])} issues:")

    # Group by file
    from collections import defaultdict
    issues_by_file = defaultdict(list)
    for issue in report["issues"]:
        file_name = Path(issue["file_path"]).name
        issues_by_file[file_name].append(issue)

    for file_name, issues in sorted(issues_by_file.items()):
        tprint(f"\n{file_name}:")
        for issue in issues:
            tprint(f"  Line {issue['line_number']}: {issue['message']}")
            if issue.get("suggestion"):
                tprint(f"    → {issue['suggestion']}")


def main():
    """Run all examples."""
    tprint("Enhanced Validation Examples")
    tprint("=" * 60)
    tprint("\nThis script demonstrates the enhanced validation capabilities:")
    tprint("1. Function argument validation")
    tprint("2. Data access validation")
    tprint("3. Integrated validation with multiple checkers")

    # Run enhanced validator example
    example_enhanced_validator()

    # Run integrated validator example
    example_integrated_validator()

    # Create and validate test files
    test_dir = example_code_with_issues()
    validate_test_files(test_dir)

    tprint("\n" + "=" * 60)
    tprint("Examples completed!")
    tprint("\nTo use these validators in your project:")
    tprint("1. Enhanced Validator: python code_quality/enhanced_validator.py --project-root /path/to/project")
    tprint("2. Integrated Validator: python code_quality/integrated_validator.py --project-root /path/to/project")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
