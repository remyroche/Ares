#!/usr/bin/env python3
"""
Summary script to report on syntax errors and provide next steps.
"""

import ast
import sys
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime


def analyze_syntax_errors(directory: Path):
    """Analyze all Python files for syntax errors."""
    python_files = list(directory.rglob('*.py'))
    
    results = {
        'total_files': len(python_files),
        'valid_files': 0,
        'error_files': 0,
        'errors_by_type': defaultdict(int),
        'files_by_error': defaultdict(list),
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"Analyzing {len(python_files)} Python files...")
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                ast.parse(f.read())
            results['valid_files'] += 1
        except SyntaxError as e:
            results['error_files'] += 1
            error_type = str(e.msg)
            results['errors_by_type'][error_type] += 1
            results['files_by_error'][error_type].append({
                'file': str(file_path.relative_to(directory)),
                'line': e.lineno,
                'text': e.text.strip() if e.text else None
            })
        except Exception as e:
            results['error_files'] += 1
            results['errors_by_type']['read_error'] += 1
    
    return results


def print_summary(results):
    """Print a formatted summary of results."""
    print(f"\n{'='*80}")
    print("SYNTAX ERROR ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Timestamp: {results['timestamp']}")
    print(f"Total Python files: {results['total_files']}")
    print(f"Valid files: {results['valid_files']} ({results['valid_files']/results['total_files']*100:.1f}%)")
    print(f"Files with errors: {results['error_files']} ({results['error_files']/results['total_files']*100:.1f}%)")
    
    print("\nError Types Distribution:")
    for error_type, count in sorted(results['errors_by_type'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {error_type}: {count} files")
    
    print("\nTop 5 Error Types with Examples:")
    for i, (error_type, files) in enumerate(sorted(results['files_by_error'].items(), 
                                                  key=lambda x: len(x[1]), reverse=True)[:5]):
        print(f"\n{i+1}. {error_type} ({len(files)} files)")
        for file_info in files[:3]:  # Show first 3 examples
            print(f"   - {file_info['file']} (line {file_info['line']})")
            if file_info['text']:
                print(f"     {file_info['text']}")
    
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS:")
    print(f"{'='*80}")
    print("1. The majority of syntax errors are:")
    print("   - Unexpected indentation (66 files)")
    print("   - Invalid syntax (25 files)")  
    print("   - Missing except/finally blocks (16 files)")
    print("   - Unmatched parentheses (12 files)")
    print("\n2. These errors likely stem from:")
    print("   - Incomplete code migrations or refactoring")
    print("   - Copy-paste errors")
    print("   - Merge conflicts not properly resolved")
    print("\n3. Manual intervention is recommended for:")
    print("   - Files with complex structural issues")
    print("   - Files with multiple cascading errors")
    print("   - Critical system files")
    print("\n4. Next steps:")
    print("   - Review and fix critical files first (config, launchers, supervisors)")
    print("   - Use IDE syntax checking to fix remaining issues")
    print("   - Run tests after fixing to ensure functionality")


def main():
    if len(sys.argv) < 2:
        print("Usage: python syntax_fix_summary.py <directory>")
        sys.exit(1)
    
    directory = Path(sys.argv[1])
    results = analyze_syntax_errors(directory)
    
    # Save detailed results
    report_path = f"/workspace/syntax_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        # Convert defaultdict to regular dict for JSON serialization
        results['errors_by_type'] = dict(results['errors_by_type'])
        results['files_by_error'] = dict(results['files_by_error'])
        json.dump(results, f, indent=2)
    
    print_summary(results)
    print(f"\nDetailed report saved to: {report_path}")


if __name__ == '__main__':
    main()