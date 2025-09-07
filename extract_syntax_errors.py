#!/usr/bin/env python3
"""
Extract and organize syntax errors from the enhanced dead code pipeline demo.
This script runs the demo and captures all syntax errors for organized reporting.
"""

import sys
import re
from pathlib import Path
from collections import defaultdict
import subprocess

def run_demo_and_capture_errors():
    """Run the demo and capture syntax errors."""
    print("Running enhanced dead code pipeline demo to capture syntax errors...")
    print("="*80)

    # Run the demo and capture output
    try:
        result = subprocess.run([
            sys.executable,
            "demo_enhanced_dead_code_pipeline.py"
            # Run full demo to get all syntax errors from interaction mapping analysis
        ], capture_output=True, text=True, cwd=Path.cwd())

        return result.stdout + result.stderr
    except Exception as e:
        print(f"Error running demo: {e}")
        return ""

def extract_syntax_errors(output):
    """Extract and organize syntax errors from demo output."""
    errors_by_file = defaultdict(list)
    errors_by_type = defaultdict(list)

    print("🔍 Analyzing demo output for syntax errors...")
    print(f"Output length: {len(output)} characters")

    # Look for syntax error patterns
    for line in output.split('\n'):
        # Look for lines containing syntax errors
        if 'Failed to analyze' in line:
            # Extract file path and error
            parts = line.split('Failed to analyze ')
            if len(parts) > 1:
                file_and_error = parts[1]
                if ': ' in file_and_error:
                    file_path, error_part = file_and_error.split(': ', 1)

                    # Clean up file path
                    if file_path.startswith('/Users/remyroche/Documents/Ares/'):
                        file_path = file_path.replace('/Users/remyroche/Documents/Ares/', '')

                    # Extract error description
                    if '(' in error_part and ')' in error_part:
                        error_desc = error_part.split(' (')[0].strip()
                    else:
                        error_desc = error_part.strip()

                    if error_desc and 'SyntaxError:' in error_desc or 'IndentationError:' in error_desc:
                        # Clean up error description
                        error_desc = error_desc.replace('SyntaxError: ', '').replace('IndentationError: ', '')

                        errors_by_file[file_path].append(error_desc)
                        errors_by_type[error_desc].append(file_path)
                        print(f"📄 Found error in {file_path}: {error_desc}")

    print(f"📊 Found {len(errors_by_file)} files with syntax errors")
    print(f"🔧 Found {len(errors_by_type)} unique error types")

    return errors_by_file, errors_by_type

def print_organized_errors(errors_by_file, errors_by_type):
    """Print syntax errors in organized format."""

    print("\n" + "="*100)
    print("📋 SYNTAX ERRORS FOUND IN CODEBASE")
    print("="*100)

    print(f"\n📊 SUMMARY:")
    print(f"   📁 Total files with syntax errors: {len(errors_by_file)}")
    print(f"   🔧 Total syntax errors: {sum(len(errors) for errors in errors_by_file.values())}")
    print(f"   📋 Unique error types: {len(errors_by_type)}")

    # Group errors by directory for better organization
    errors_by_directory = defaultdict(lambda: defaultdict(list))

    for file_path, errors in errors_by_file.items():
        directory = str(Path(file_path).parent)
        for error in errors:
            errors_by_directory[directory][error].append(file_path)

    print("\n" + "="*100)
    print("📂 ERRORS BY DIRECTORY")
    print("="*100)

    for directory, dir_errors in sorted(errors_by_directory.items()):
        print(f"\n📁 {directory}/")
        print("-" * (len(directory) + 3))

        for error_type, files in dir_errors.items():
            print(f"  🔴 {error_type}")
            for file in sorted(set(files)):
                print(f"     📄 {file}")
            print()

    print("\n" + "="*100)
    print("🔧 ERRORS BY TYPE")
    print("="*100)

    for error_type, files in sorted(errors_by_type.items()):
        print(f"\n🔧 {error_type}")
        print("-" * (len(error_type) + 3))
        unique_files = sorted(set(files))
        for file in unique_files:
            print(f"   📄 {file}")
        print(f"   📊 Affected files: {len(unique_files)}")

    print("\n" + "="*100)
    print("💡 RECOMMENDATIONS")
    print("="*100)
    print("""
1. 🔧 Fix indentation errors first - these are usually quick fixes
2. 📝 Address unterminated strings - check for missing quotes/brackets
3. 🏗️ Fix function/class definitions - ensure proper indentation blocks
4. 🔍 Review import statements - check for syntax issues
5. 📊 Prioritize by directory - fix critical directories first

Most Common Fixes:
• Add missing indentation after function/class definitions
• Close unterminated string literals
• Fix import statement syntax
• Complete try/except blocks properly
• Fix bracket/parentheses matching
    """)

def main():
    """Main function."""
    # Run demo and capture output
    output = run_demo_and_capture_errors()

    if not output:
        print("❌ No output captured from demo run")
        return

    # Extract and organize errors
    errors_by_file, errors_by_type = extract_syntax_errors(output)

    if not errors_by_file:
        print("✅ No syntax errors found!")
        return

    # Print organized results
    print_organized_errors(errors_by_file, errors_by_type)

    print(f"\n{'='*100}")
    print("🎯 NEXT STEPS")
    print(f"{'='*100}")
    print("1. Use the organized list above to prioritize fixes")
    print("2. Start with the most critical directories (src/, core/ files)")
    print("3. Fix indentation errors first (usually quickest)")
    print("4. Re-run the enhanced dead code pipeline after fixes")
    print("5. Monitor improvement in false positive detection")

if __name__ == "__main__":
    main()
