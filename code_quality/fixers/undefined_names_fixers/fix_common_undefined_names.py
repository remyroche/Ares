#!/usr/bin/env python3
"""
Script to fix the most common undefined names by adding missing imports.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Any


def get_common_imports() -> Dict[str, str]:
    """Get the most common import patterns to fix."""
    return {
        'pd': 'import pandas as pd',
        'np': 'import numpy as np',
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
        'cached': 'from functools import cached_property',
        'error': 'from logging import error',
        'info': 'from logging import info',
        'warning': 'from logging import warning',
        'debug': 'from logging import debug',
    }


def find_import_position(content: str) -> int:
    """Find the position to insert imports after existing imports."""
    lines = content.split('\n')
    
    # Find the last import line
    last_import_line = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(('import ', 'from ')) and not stripped.startswith('#'):
            last_import_line = i
    
    # If no imports found, find the first non-comment, non-docstring line
    if last_import_line == -1:
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and not stripped.startswith('"""') and not stripped.startswith("'''"):
                return i
    
    return last_import_line + 1


def add_imports_to_file(file_path: str, undefined_names: Set[str], dry_run: bool = False) -> bool:
    """Add missing imports to a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        common_imports = get_common_imports()
        imports_to_add = []
        
        # Check which undefined names we can fix
        for name in undefined_names:
            if name in common_imports:
                import_stmt = common_imports[name]
                if import_stmt not in content:
                    imports_to_add.append(import_stmt)
        
        if not imports_to_add:
            return False
        
        if dry_run:
            print(f"Would add to {file_path}:")
            for imp in imports_to_add:
                print(f"  {imp}")
            return True
        
        # Find position to insert imports
        insert_pos = find_import_position(content)
        lines = content.split('\n')
        
        # Insert imports
        for i, import_stmt in enumerate(imports_to_add):
            lines.insert(insert_pos + i, import_stmt)
        
        # Add blank line after imports if needed
        if insert_pos + len(imports_to_add) < len(lines):
            next_line = lines[insert_pos + len(imports_to_add)].strip()
            if next_line and not next_line.startswith('#'):
                lines.insert(insert_pos + len(imports_to_add), '')
        
        # Write back to file
        new_content = '\n'.join(lines)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"✅ Fixed {file_path}: Added {len(imports_to_add)} imports")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False


def fix_undefined_names_from_report(report_path: str, dry_run: bool = False) -> None:
    """Fix undefined names based on the report."""
    
    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)
    
    files_fixed = 0
    total_imports_added = 0
    
    print(f"🔍 Analyzing report: {report_path}")
    print(f"📊 Total files with errors: {len([f for f in report.get('files', {}).values() if f.get('total_errors', 0) > 0])}")
    
    for file_path, file_data in report.get('files', {}).items():
        if file_data.get('status') == 'success' and file_data.get('total_errors', 0) > 0:
            # Extract undefined names from errors
            undefined_names = set()
            for error in file_data.get('errors', []):
                if error.get('error_type') == 'undefined_name':
                    name = error.get('name', '')
                    if name:
                        undefined_names.add(name)
            
            if undefined_names:
                print(f"\n📁 Processing: {file_path}")
                print(f"   Undefined names: {sorted(undefined_names)}")
                
                if add_imports_to_file(file_path, undefined_names, dry_run):
                    files_fixed += 1
                    common_imports = get_common_imports()
                    imports_added = sum(1 for name in undefined_names if name in common_imports)
                    total_imports_added += imports_added
    
    print(f"\n📈 Summary:")
    print(f"   Files processed: {files_fixed}")
    print(f"   Total imports added: {total_imports_added}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix common undefined names by adding imports")
    parser.add_argument("--report", "-r", 
                       default="code_quality/reports/undefined_names_latest.json",
                       help="Path to undefined names report")
    parser.add_argument("--dry-run", "-d", action="store_true",
                       help="Show what would be fixed without making changes")
    
    args = parser.parse_args()
    
    if not Path(args.report).exists():
        print(f"❌ Report file not found: {args.report}")
        return 1
    
    fix_undefined_names_from_report(args.report, args.dry_run)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
