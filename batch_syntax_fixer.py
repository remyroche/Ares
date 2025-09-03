#!/usr/bin/env python3
"""
Batch syntax fixer for common issues in the codebase.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

def run_autopep8(filepath):
    """Run autopep8 on a file to fix basic syntax issues."""
    try:
        # Run autopep8 with aggressive mode
        cmd = ['python3', '-m', 'autopep8', '--aggressive', '--aggressive', '--in-place', str(filepath)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"  autopep8 failed: {e}")
        return False

def run_black(filepath):
    """Run black formatter on a file."""
    try:
        cmd = ['python3', '-m', 'black', '--line-length', '120', str(filepath)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"  black failed: {e}")
        return False

def validate_syntax(filepath):
    """Check if a file has valid Python syntax."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            compile(f.read(), str(filepath), 'exec')
        return True
    except SyntaxError:
        return False
    except Exception:
        return False

def process_files_with_errors():
    """Process all files with syntax errors."""
    # Read the analysis report
    report_path = Path("/workspace/sequential_fixer_reports/sequential_analysis_20250903_121913.json")
    
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    # Get all files with errors
    error_files = []
    for error in report.get('syntax_errors', []):
        filepath = Path("/workspace/src") / error['file']
        if filepath.exists():
            error_files.append(filepath)
    
    print(f"Found {len(error_files)} files with syntax errors")
    print("Attempting to fix with automated tools...")
    print("=" * 60)
    
    fixed = 0
    still_broken = 0
    
    for i, filepath in enumerate(error_files, 1):
        print(f"\n[{i}/{len(error_files)}] Processing: {filepath.name}")
        
        # Check current status
        if validate_syntax(filepath):
            print("  ✓ Already fixed")
            fixed += 1
            continue
        
        # Try autopep8 first
        print("  → Trying autopep8...")
        if run_autopep8(filepath):
            if validate_syntax(filepath):
                print("  ✓ Fixed with autopep8")
                fixed += 1
                continue
        
        # Try black
        print("  → Trying black...")
        if run_black(filepath):
            if validate_syntax(filepath):
                print("  ✓ Fixed with black")
                fixed += 1
                continue
        
        print("  ✗ Still has errors")
        still_broken += 1
    
    print("\n" + "=" * 60)
    print(f"Summary:")
    print(f"  Total files: {len(error_files)}")
    print(f"  Fixed: {fixed}")
    print(f"  Still broken: {still_broken}")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_files': len(error_files),
        'fixed': fixed,
        'still_broken': still_broken
    }
    
    results_file = f"/workspace/batch_fix_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    # First check if we have the required tools
    print("Checking for required tools...")
    
    # Try to install autopep8 if not available
    try:
        import autopep8
        print("✓ autopep8 is available")
    except ImportError:
        print("✗ autopep8 not found - Some fixes may not work")
    
    # Try to install black if not available
    try:
        import black
        print("✓ black is available")
    except ImportError:
        print("✗ black not found - Some fixes may not work")
    
    print()
    process_files_with_errors()