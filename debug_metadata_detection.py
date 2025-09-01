#!/usr/bin/env python3
"""
Debug script to test metadata detection in step files.
"""

import re
from pathlib import Path

def test_metadata_detection(...):
    pass"""Test metadata detection in a step file."""
    print(f"\n🔍 Testing metadata detection in {file_path.name}")

    try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
        with open(file_path, 'r', encoding='utf-8') as f:
    passcontent = f.read()

        # Extract step number
        step_match = re.search(r'step(\d+(?:_\d+)?)', file_path.name)
        if not step_match:
    passprint("❌ Could not extract step number")
            return

        step_num = step_match.group(1)
        method_name = f"_log_step{step_num}_artifacts_and_report"

        print(f"   Looking for method: {method_name}")

        if method_name not in content:
    passprint(f"❌ Method {method_name} not found")
            return

        # Find method content
        method_start = content.find(method_name)
        if method_start == -1:
    passprint("❌ Could not find method start")
            return

        # Find the opening brace of the method
        brace_start = content.find(":", method_start)
        if brace_start == -1:
    passprint("❌ Could not find method opening brace")
            return

        # Find method end by looking for the next method or end of file
        # Look for the next method that's at the same indentation level
        lines = content.split('\n')
        method_start_line = content[:method_start].count('\n')

        method_end = len(content)
        for i in range(method_start_line + 1, len(lines)):
    passline = lines[i]
            if line.strip().startswith('def ') and line.strip() != method_name:
    pass# Found next method, calculate end position
                method_end = content.find(line, method_start)
                break

        # Get the entire method content from the method name to the end
        method_content = content[method_start:method_end]

        print(f"   Method content length: {len(method_content)} characters")
        print(f"   First 200 characters: {method_content[:200]}")
        print(f"   Last 200 characters: {method_content[-200:]}")

        # Test metadata field detection
        metadata_fields = ["asset", "lookback_period", "project_version", "date"]

        for field in metadata_fields:
    pass# Look for different patterns
            patterns = [
                f'"{field}"',
                f"'{field}'",
                field
            ]

            found = False
            for pattern in patterns:
    passif pattern in method_content:
    passfound = True
                    print(f"   ✅ Found {field} with pattern: {pattern}")
                    break

            if not found:
    passprint(f"   ❌ Not found: {field}")

        # Test specific patterns we added
        specific_patterns = [
            '"asset": symbol',
            'lookback_period": self.config.get("lookback_days"',
            'project_version": self.config.get("project_version"',
            'datetime.now()'
        ]

        print(f"\n   Testing specific patterns:")
        for pattern in specific_patterns:
    passif pattern in method_content:
    passprint(f"   ✅ Found: {pattern}")
            else:
    passprint(f"   ❌ Not found: {pattern}")

    except Exception as e:
    passpasspasspasspasspasspassprint(f"❌ Error testing {file_path.name}: {e}")

def main(...):
    pass"""Main function to test metadata detection."""
    steps_dir = Path("src/training/steps")

    # Test a few files
    test_files = [
        "step2_data_reading.py",
        "step8_regime_data_splitting.py",
        "step1_data_collection.py"
    ]

    for test_file in test_files:
    passfile_path = steps_dir / test_file
        if file_path.exists():
    passtest_metadata_detection(file_path)
        else:
    passprint(f"❌ File not found: {test_file}")

if __name__ == "__main__":
    passmain()