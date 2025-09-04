#!/usr/bin/env python3
"""
Script to fix missing imports for the most common undefined names.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Any

def get_import_fixes() -> Dict[str, str]:
    """Get import fixes for common undefined names."""
    return {
        # Missing imports
        'pd': 'import pandas as pd',
        'np': 'import numpy as np',
        'cached': 'from functools import cached_property',
        'logger': 'from src.utils.logger import logger',
        'error': 'from src.utils.logger import error',
        'handles_errors': 'from src.core.decorators.errors import handles_errors',
        'Any': 'from typing import Any',
        'Dict': 'from typing import Dict',
        'List': 'from typing import List',
        'Optional': 'from typing import Optional',
        'Tuple': 'from typing import Tuple',
        'Set': 'from typing import Set',
        'Union': 'from typing import Union',
        'Callable': 'from typing import Callable',
        'datetime': 'from datetime import datetime',
        'dataclass': 'from dataclasses import dataclass',
    }

def analyze_file_for_missing_imports(file_path: str, report_data: Dict[str, Any]) -> Set[str]:
    """Analyze a file for missing imports."""
    missing_imports = set()
    
    if file_path not in report_data.get('files', {}):
        return missing_imports
    
    file_data = report_data['files'][file_path]
    if file_data.get('status') != 'success' or file_data.get('total_errors', 0) == 0:
        return missing_imports
    
    import_fixes = get_import_fixes()
    
    for error in file_data.get('errors', []):
        if error.get('error_type') == 'undefined_name':
            name = error.get('name', '')
            if name in import_fixes:
                missing_imports.add(import_fixes[name])
    
    return missing_imports

def add_missing_imports_to_file(file_path: str, missing_imports: Set[str], dry_run: bool = True) -> int:
    """Add missing imports to a file."""
    if not missing_imports:
        return 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        # Find the best place to add imports (after existing imports)
        import_end_line = 0
        for i, line in enumerate(lines):
            if line.strip().startswith(('import ', 'from ')):
                import_end_line = i + 1
            elif line.strip() and not line.strip().startswith('#'):
                break
        
        if dry_run:
            print(f"  Would add {len(missing_imports)} imports:")
            for imp in sorted(missing_imports):
                print(f"    {imp}")
        else:
            # Insert imports
            for imp in sorted(missing_imports):
                lines.insert(import_end_line, imp)
                import_end_line += 1
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
        
        return len(missing_imports)
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0

def main():
    """Main function to fix missing imports."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix missing imports')
    parser.add_argument('--report', default='code_quality/reports/undefined_names_after_import_fixes.json',
                       help='Path to undefined names report')
    parser.add_argument('--dry-run', action='store_true', default=True,
                       help='Show what would be fixed without making changes')
    parser.add_argument('--apply', action='store_true',
                       help='Actually apply the fixes')
    parser.add_argument('--max-files', type=int, default=10,
                       help='Maximum number of files to process')
    
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
    print(f"Max files to process: {args.max_files}")
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
    
    for file_path, error_count in files_with_errors[:args.max_files]:
        if not os.path.exists(file_path):
            continue
            
        missing_imports = analyze_file_for_missing_imports(file_path, report_data)
        if missing_imports:
            print(f"\n📁 {file_path} ({error_count} errors)")
            
            fixed_count = add_missing_imports_to_file(file_path, missing_imports, args.dry_run)
            total_fixes += fixed_count
            files_processed += 1
    
    print("\n" + "=" * 80)
    print(f"Summary:")
    print(f"  Files processed: {files_processed}")
    print(f"  Total imports added: {total_fixes}")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'APPLIED'}")
    
    if args.dry_run and total_fixes > 0:
        print(f"\nTo apply these fixes, run with --apply flag")

if __name__ == "__main__":
    main()
