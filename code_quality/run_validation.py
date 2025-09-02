#!/usr/bin/env python3
"""
Simple runner script for code quality validation tools.

This script provides easy access to both the comprehensive code quality review
and the focused function validation tools.
"""

import sys
import os
from pathlib import Path

# Add the code_quality directory to the path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from comprehensive_code_review import CodeQualityReviewer
from function_validator import FunctionValidator


def run_comprehensive_review(project_root: str = ".", output_file: str = None):
    """Run the comprehensive code quality review."""
    print("Running comprehensive code quality review...")
    
    reviewer = CodeQualityReviewer(project_root)
    output_file = reviewer.generate_report(output_file)
    
    print(f"Comprehensive review completed!")
    print(f"Report: {output_file}")
    print(f"Summary: {output_file.replace('.json', '_summary.txt')}")
    
    return output_file


def run_function_validation(project_root: str = ".", output_file: str = None):
    """Run the focused function validation."""
    print("Running function validation...")
    
    validator = FunctionValidator(project_root)
    output_file = validator.generate_report(output_file)
    
    print(f"Function validation completed!")
    print(f"Report: {output_file}")
    print(f"Summary: {output_file.replace('.json', '_summary.txt')}")
    
    return output_file


def main():
    """Main entry point with command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Code Quality Validation Runner')
    parser.add_argument('--mode', choices=['comprehensive', 'function', 'both'], 
                       default='both', help='Validation mode to run')
    parser.add_argument('--project-root', default='.', help='Project root directory')
    parser.add_argument('--output-dir', default='./reports', help='Output directory for reports')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if args.verbose:
        print(f"Project root: {args.project_root}")
        print(f"Output directory: {output_dir}")
    
    results = {}
    
    if args.mode in ['comprehensive', 'both']:
        output_file = output_dir / f"comprehensive_review_{int(time.time())}.json"
        results['comprehensive'] = run_comprehensive_review(args.project_root, str(output_file))
    
    if args.mode in ['function', 'both']:
        output_file = output_dir / f"function_validation_{int(time.time())}.json"
        results['function'] = run_function_validation(args.project_root, str(output_file))
    
    print("\n" + "="*50)
    print("VALIDATION COMPLETED")
    print("="*50)
    
    for mode, output_file in results.items():
        print(f"{mode.title()}: {output_file}")
        summary_file = output_file.replace('.json', '_summary.txt')
        if os.path.exists(summary_file):
            print(f"Summary: {summary_file}")
    
    return results


if __name__ == '__main__':
    import time
    main()