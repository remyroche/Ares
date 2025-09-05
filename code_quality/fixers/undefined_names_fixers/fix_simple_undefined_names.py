#!/usr/bin/env python3
"""
Script to fix simple undefined names by adding missing imports and variable definitions.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Any

def get_simple_fixes() -> Dict[str, str]:
    """Get simple fixes for common undefined names."""
    return {
        # Missing imports
        'pd': 'import pandas as pd',
        'np': 'import numpy as np',
        'cached': 'from functools import cached_property',
        'logger': 'from src.utils.logger import logger',
        'error': 'from src.utils.logger import error',
        'handles_errors': 'from src.core.decorators.errors import handles_errors',
        
        # Common variables that might need initialization
        'symbol': 'symbol = ""',
        'exchange': 'exchange = ""',
        'timeframe': 'timeframe = ""',
        'data_dir': 'data_dir = ""',
        'force_rerun': 'force_rerun = False',
        'failed': 'failed = False',
        'traced': 'traced = False',
        'size': 'size = 0',
        'regime_id': 'regime_id = ""',
        'decision_id': 'decision_id = ""',
        'timeframe_name': 'timeframe_name = ""',
        'strategy': 'strategy = ""',
    }

def analyze_file_for_simple_fixes(file_path: str, report_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Analyze a file for simple undefined names that can be fixed."""
    fixes = []
    
    if file_path not in report_data.get('files', {}):
        return fixes
    
    file_data = report_data['files'][file_path]
    if file_data.get('status') != 'success' or file_data.get('total_errors', 0) == 0:
        return fixes
    
    simple_fixes = get_simple_fixes()
    
    for error in file_data.get('errors', []):
        if error.get('error_type') == 'undefined_name':
            name = error.get('name', '')
            line_num = error.get('line', 0)
            
            if name in simple_fixes:
                fixes.append({
                    'name': name,
                    'line': line_num,
                    'fix': simple_fixes[name],
                    'context': error.get('context', '')
                })
    
    return fixes

def fix_simple_undefined_names_in_file(file_path: str, fixes: List[Dict[str, Any]], dry_run: bool = True) -> int:
    """Fix simple undefined names in a file."""
    if not fixes:
        return 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        fixed_count = 0
        
        # Group fixes by type
        import_fixes = []
        variable_fixes = []
        
        for fix in fixes:
            if fix['fix'].startswith('import ') or fix['fix'].startswith('from '):
                import_fixes.append(fix)
            else:
                variable_fixes.append(fix)
        
        # Handle import fixes first
        if import_fixes:
            # Find the best place to add imports (after existing imports)
            import_end_line = 0
            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ')):
                    import_end_line = i + 1
                elif line.strip() and not line.strip().startswith('#'):
                    break
            
            # Add unique imports
            imports_to_add = set()
            for fix in import_fixes:
                imports_to_add.add(fix['fix'])
            
            if dry_run:
                for imp in imports_to_add:
                    print(f"  Would add import: {imp}")
            else:
                # Insert imports
                for imp in sorted(imports_to_add):
                    lines.insert(import_end_line, imp)
                    import_end_line += 1
                    fixed_count += 1
        
        # Handle variable fixes
        if variable_fixes:
            # Sort by line number (descending) to avoid line number shifts
            variable_fixes.sort(key=lambda x: x['line'], reverse=True)
            
            for fix in variable_fixes:
                line_idx = fix['line'] - 1  # Convert to 0-based index
                if 0 <= line_idx < len(lines):
                    line = lines[line_idx]
                    name = fix['name']
                    fix_text = fix['fix']
                    
                    # Check if this is a simple variable assignment context
                    if (name in line and 
                        ('=' in line or line.strip().startswith(name) or 
                         line.strip().endswith(name) or 
                         name in line.split())):
                        
                        if dry_run:
                            print(f"  Would fix line {fix['line']}: {name} -> {fix_text}")
                        else:
                            # Try to add variable initialization
                            if '=' not in line or line.strip().startswith(name):
                                # Add variable initialization before the line
                                lines.insert(line_idx, f"    {fix_text}")
                                fixed_count += 1
        
        if not dry_run and fixed_count > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
        
        return fixed_count
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0

def main():
    """Main function to fix simple undefined names."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix simple undefined names')
    parser.add_argument('--report', default='code_quality/reports/undefined_names_current_scan.json',
                       help='Path to undefined names report')
    parser.add_argument('--dry-run', action='store_true', default=True,
                       help='Show what would be fixed without making changes')
    parser.add_argument('--apply', action='store_true',
                       help='Actually apply the fixes')
    
    args = parser.parse_args()
    
    if args.apply:
        args.dry_run = False
    
    # Load report
    try:
        with open(args.report, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
    except FileNotFoundError:
        print(f"Report file not found: {args.report}")
        return
    except json.JSONDecodeError as e:
        print(f"Error parsing report: {e}")
        return
    
    print(f"Analyzing undefined names report: {args.report}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'APPLYING FIXES'}")
    print("=" * 80)
    
    total_fixes = 0
    files_processed = 0
    
    # Process files with most errors first
    files_with_errors = []
    for file_path, file_data in report_data.get('files', {}).items():
        if file_data.get('status') == 'success' and file_data.get('total_errors', 0) > 0:
            files_with_errors.append((file_path, file_data.get('total_errors', 0)))
    
    # Sort by error count (descending)
    files_with_errors.sort(key=lambda x: x[1], reverse=True)
    
    for file_path, error_count in files_with_errors[:10]:  # Process top 10 files
        if not os.path.exists(file_path):
            continue
            
        fixes = analyze_file_for_simple_fixes(file_path, report_data)
        if fixes:
            print(f"\n📁 {file_path} ({error_count} errors)")
            print(f"   Found {len(fixes)} simple fixes:")
            
            fixed_count = fix_simple_undefined_names_in_file(file_path, fixes, args.dry_run)
            total_fixes += fixed_count
            files_processed += 1
    
    print("\n" + "=" * 80)
    print(f"Summary:")
    print(f"  Files processed: {files_processed}")
    print(f"  Total fixes: {total_fixes}")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'APPLIED'}")
    
    if args.dry_run and total_fixes > 0:
        print(f"\nTo apply these fixes, run with --apply flag")

if __name__ == "__main__":
    main()
