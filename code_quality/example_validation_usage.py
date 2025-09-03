#!/usr/bin/env python3
"""
Example usage of the enhanced validation tools.

This script demonstrates how to use:
1. Enhanced Validator (for function arguments and data access)
2. Integrated Validator (combining function_validator and enhanced_validator)
"""

import os
import sys
from pathlib import Path

# Add code_quality to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_validator import EnhancedValidator
from integrated_validator import IntegratedValidator


def example_enhanced_validator():
    """Example of using the enhanced validator directly."""
    print("=" * 60)
    print("ENHANCED VALIDATOR EXAMPLE")
    print("=" * 60)
    
    # Initialize validator
    validator = EnhancedValidator(
        project_root='.',
        exclude_patterns=['__pycache__', '*.pyc', '.git', 'venv', 'test_*']
    )
    
    # Run validation
    print("\nRunning enhanced validation...")
    report = validator.validate_project()
    
    # Display summary
    summary = report['summary']
    print(f"\nValidation Summary:")
    print(f"  Files processed: {summary['files_processed']}")
    print(f"  Total issues: {summary['total_issues']}")
    print(f"  - Argument mismatches: {summary['argument_mismatches']}")
    print(f"  - Unsafe data access: {summary['unsafe_data_access']}")
    print(f"  - Missing null checks: {summary['missing_null_checks']}")
    print(f"  - Type inconsistencies: {summary['type_inconsistencies']}")
    
    # Show top issues
    if report['issues']:
        print("\nTop Issues Found:")
        for issue in report['issues'][:5]:
            print(f"  {issue['severity'].upper()}: {issue['file_path']}:{issue['line_number']}")
            print(f"    {issue['message']}")
            if issue.get('suggestion'):
                print(f"    → {issue['suggestion']}")
    
    return report


def example_integrated_validator():
    """Example of using the integrated validator."""
    print("\n" + "=" * 60)
    print("INTEGRATED VALIDATOR EXAMPLE")
    print("=" * 60)
    
    # Initialize integrated validator
    validator = IntegratedValidator(
        project_root='.',
        exclude_patterns=['__pycache__', '*.pyc', '.git', 'venv', 'test_*']
    )
    
    # Generate comprehensive reports
    print("\nGenerating integrated validation reports...")
    report_files = validator.generate_report(output_dir='code_quality/reports')
    
    print("\nGenerated reports:")
    for report_type, file_path in report_files.items():
        print(f"  - {report_type.capitalize()}: {file_path}")
    
    # Read and display summary
    with open(report_files['summary'], 'r') as f:
        summary_content = f.read()
    
    print("\n" + "-" * 40)
    print("SUMMARY PREVIEW:")
    print("-" * 40)
    # Show first 30 lines of summary
    summary_lines = summary_content.split('\n')[:30]
    print('\n'.join(summary_lines))
    if len(summary_content.split('\n')) > 30:
        print("... (truncated)")


def example_code_with_issues():
    """Create example code files with various issues for testing."""
    print("\n" + "=" * 60)
    print("CREATING TEST FILES WITH ISSUES")
    print("=" * 60)
    
    test_dir = Path('code_quality/test_examples')
    test_dir.mkdir(exist_ok=True)
    
    # Example 1: Function with argument issues
    example1 = '''#!/usr/bin/env python3
"""Example with function argument issues."""

def process_data(data, threshold, normalize=True):
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
    result3 = process_data(100, 0.5, normalized=True)  # Wrong keyword

if __name__ == "__main__":
    main()
'''
    
    with open(test_dir / 'argument_issues.py', 'w') as f:
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
    
    with open(test_dir / 'data_access_issues.py', 'w') as f:
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
    print(result.value)  # result might be None
'''
    
    with open(test_dir / 'mixed_issues.py', 'w') as f:
        f.write(example3)
    
    print(f"\nCreated test files in {test_dir}:")
    for file in test_dir.glob('*.py'):
        print(f"  - {file.name}")
    
    return test_dir


def validate_test_files(test_dir):
    """Run validation on the test files."""
    print("\n" + "=" * 60)
    print("VALIDATING TEST FILES")
    print("=" * 60)
    
    # Use enhanced validator on test directory
    validator = EnhancedValidator(
        project_root=str(test_dir),
        exclude_patterns=['__pycache__', '*.pyc']
    )
    
    print(f"\nValidating test directory: {test_dir}")
    report = validator.validate_project()
    
    # Display all issues found
    print(f"\nFound {len(report['issues'])} issues:")
    
    # Group by file
    from collections import defaultdict
    issues_by_file = defaultdict(list)
    for issue in report['issues']:
        file_name = Path(issue['file_path']).name
        issues_by_file[file_name].append(issue)
    
    for file_name, issues in sorted(issues_by_file.items()):
        print(f"\n{file_name}:")
        for issue in issues:
            print(f"  Line {issue['line_number']}: {issue['message']}")
            if issue.get('suggestion'):
                print(f"    → {issue['suggestion']}")


def main():
    """Run all examples."""
    print("Enhanced Validation Examples")
    print("=" * 60)
    print("\nThis script demonstrates the enhanced validation capabilities:")
    print("1. Function argument validation")
    print("2. Data access validation")
    print("3. Integrated validation with multiple checkers")
    
    # Run enhanced validator example
    example_enhanced_validator()
    
    # Run integrated validator example
    example_integrated_validator()
    
    # Create and validate test files
    test_dir = example_code_with_issues()
    validate_test_files(test_dir)
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("\nTo use these validators in your project:")
    print("1. Enhanced Validator: python code_quality/enhanced_validator.py --project-root /path/to/project")
    print("2. Integrated Validator: python code_quality/integrated_validator.py --project-root /path/to/project")


if __name__ == '__main__':
    main()