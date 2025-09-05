#!/usr/bin/env python3
"""
Script to fix the most common missing imports based on the fixed analyzer results.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Any
from collections import defaultdict

def get_common_imports() -> Dict[str, str]:
    """Get the most common import patterns to fix based on analysis."""
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
        'cached': 'from functools import cached_property',
        'warning': 'from logging import warning',
    }

def add_import_to_file(file_path: Path, import_statement: str, dry_run: bool = True) -> bool:
    """Adds an import statement to a Python file if it's not already present."""
    try:
        content = file_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"  ❌ Error reading file: {e}")
        return False
        
    if import_statement in content:
        return False  # Already exists

    # Find the best place to insert the import:
    lines = content.splitlines()
    insert_index = 0
    found_imports = False

    # Skip shebang and encoding
    if lines and lines[0].startswith('#!'):
        insert_index += 1
    if lines and insert_index < len(lines) and 'coding:' in lines[insert_index]:
        insert_index += 1

    # Skip docstring
    if lines and insert_index < len(lines) and lines[insert_index].strip().startswith('"""'):
        while insert_index < len(lines) and not lines[insert_index].strip().endswith('"""'):
            insert_index += 1
        if insert_index < len(lines):  # Found end of docstring
            insert_index += 1

    # Find last existing import
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if line.startswith('import ') or line.startswith('from '):
            insert_index = i + 1
            found_imports = True
            break
    
    # If no imports found, insert after docstring/shebang
    if not found_imports:
        # Ensure there's a blank line after docstring/shebang if inserting at top
        if insert_index > 0 and lines[insert_index-1].strip() != '':
            lines.insert(insert_index, '')
            insert_index += 1
            
    # Insert the new import statement
    lines.insert(insert_index, import_statement)
    
    # Ensure there's a blank line after the new import if it's not already there
    if insert_index + 1 < len(lines) and lines[insert_index + 1].strip() != '':
        lines.insert(insert_index + 1, '')

    new_content = "\n".join(lines)
    if not dry_run:
        try:
            file_path.write_text(new_content, encoding='utf-8')
        except Exception as e:
            print(f"  ❌ Error writing file: {e}")
            return False
    return True

def fix_common_imports(report_path: str, dry_run: bool = True):
    """
    Fixes common missing imports based on the report.
    """
    print(f"\n🔧 Fixing common missing imports")
    print(f"📊 Report: {report_path}")
    print(f"🎯 Mode: {'DRY RUN' if dry_run else 'APPLYING FIXES'}")
    print("=" * 80)

    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

    common_imports = get_common_imports()
    total_imports_added = 0
    files_processed = 0
    
    files_to_process: Dict[Path, Set[str]] = defaultdict(set)

    # Collect files and imports to add
    for file_path_str, file_data in report.get('files', {}).items():
        if file_data.get('status') == 'success' and file_data.get('total_errors', 0) > 0:
            file_imports = set()
            for error in file_data.get('errors', []):
                if error.get('error_type') == 'undefined_name':
                    name = error.get('name')
                    if name in common_imports:
                        file_imports.add(common_imports[name])
            
            if file_imports:
                files_to_process[Path(file_path_str)] = file_imports

    # Process each file
    for file_path, imports_to_add in files_to_process.items():
        if not file_path.exists():
            print(f"⚠️  File not found: {file_path}")
            continue
            
        print(f"\n📁 {file_path} ({len(imports_to_add)} imports)")
        imports_added_for_file = 0
        
        for import_statement in sorted(list(imports_to_add)):
            if add_import_to_file(file_path, import_statement, dry_run):
                print(f"  {'Would add' if dry_run else '✅ Added'} import: {import_statement}")
                imports_added_for_file += 1
            else:
                print(f"  ⏭️  Already exists: {import_statement}")
                
        if imports_added_for_file > 0:
            total_imports_added += imports_added_for_file
            files_processed += 1

    print("\n" + "=" * 80)
    print("📊 Summary:")
    print(f"  📁 Files processed: {files_processed}")
    print(f"  📦 Total imports added: {total_imports_added}")
    print(f"  🎯 Mode: {'DRY RUN' if dry_run else 'APPLIED'}")

    if dry_run:
        print("\n💡 To apply these fixes, run with --apply flag")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fix common missing imports based on undefined names analysis.")
    parser.add_argument("--report", type=str, default="code_quality/reports/undefined_names_fixed_analyzer.json",
                        help="Path to the undefined names JSON report.")
    parser.add_argument("--apply", action="store_true", help="Apply fixes to files (default is dry run).")
    args = parser.parse_args()

    fix_common_imports(args.report, dry_run=not args.apply)
