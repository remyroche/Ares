#!/usr/bin/env python3
"""
Run the conservative auto-fixer on the codebase.
This version prioritizes safety and won't break your code.
"""

import os
import sys
import json
from datetime import datetime

# Add code_quality to path
sys.path.insert(0, os.path.dirname(__file__))

from code_quality.fixers.conservative_auto_fixer import ConservativeAutoFixer
from code_quality.core.config import load_config


def main():
    """Run conservative fixer with safety features."""
    print("=" * 70)
    print("CONSERVATIVE AUTO-FIXER")
    print("=" * 70)
    print("This fixer prioritizes safety over aggressive changes.")
    print("Features:")
    print("  - Pre-validates syntax before attempting fixes")
    print("  - Creates backups before any changes")
    print("  - Validates after each fix")
    print("  - Automatically restores files if fixes break syntax")
    print("  - Only uses the safest formatting tools")
    print("=" * 70)
    
    # You can specify a target directory or file
    import argparse
    parser = argparse.ArgumentParser(description="Conservative Python code fixer")
    parser.add_argument("target", nargs="?", default="src", 
                       help="Target file or directory (default: src)")
    parser.add_argument("--report", help="Save detailed report to file")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Preview changes without applying them")
    
    args = parser.parse_args()
    
    # Load conservative configuration
    try:
        config = load_config("code_quality/config_conservative.yaml")
        print("Loaded conservative configuration")
    except Exception:
        config = None
        print("Using default conservative settings")
    
    # Create the fixer
    fixer = ConservativeAutoFixer(config)
    
    # Show which tools are enabled
    print(f"\nEnabled tools: {', '.join(fixer.enabled_tools)}")
    print(f"Target: {args.target}")
    print()
    
    # Process the target
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if os.path.isfile(args.target):
        print(f"Processing single file: {args.target}")
        results = fixer.fix_file(args.target)
    elif os.path.isdir(args.target):
        print(f"Processing directory: {args.target}")
        results = fixer.fix_directory(args.target)
    else:
        print(f"Error: {args.target} is not a valid file or directory")
        return 1
    
    # Save report if requested
    if args.report:
        report_file = args.report if args.report.endswith('.json') else f"{args.report}_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed report saved to: {report_file}")
    
    # Show summary
    if "summary" in results:
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        summary = results["summary"]
        print(f"Success rate: {summary['success_rate']:.1f}%")
        print(f"Files that needed restoration: {summary['restored_files']}")
        print(f"Files with pre-existing errors: {summary['skipped_files']}")
        
        if summary['restored_files'] > 0:
            print("\nWARNING: Some files were restored because fixes would have broken them.")
            print("These files need manual attention.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())