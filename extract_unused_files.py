#!/usr/bin/env python3
"""
Extract unused files from import analysis results.
"""

import json
from pathlib import Path

def extract_unused_files():
    """Extract all unused files from the import analysis."""
    json_file = Path("/Users/remyroche/Documents/Ares/imports_analysis_20250910_193741/import_verifier_analysis.json")

    with open(json_file, 'r') as f:
        data = json.load(f)

    unused_files = []
    for file_path, info in data['import_status'].items():
        if not info.get('is_imported', True):
            unused_files.append({
                'path': file_path,
                'module_name': info.get('module_name', ''),
                'import_count': info.get('import_count', 0)
            })

    # Sort by file path for consistency
    unused_files.sort(key=lambda x: x['path'])

    return unused_files

if __name__ == "__main__":
    unused_files = extract_unused_files()
    print(f"Found {len(unused_files)} unused files:")
    for i, file_info in enumerate(unused_files, 1):
        print(f"{i:2d}. {file_info['path']} ({file_info['module_name']})")
