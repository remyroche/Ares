#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Quick verification of unused files to identify truly dead code.
Focuses on the most important checks for efficiency.
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Set

def extract_unused_files() -> List[Dict]:
    """Extract unused files from import analysis results."""
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

    return unused_files

def grep_search(pattern: str, search_path: str = "/Users/remyroche/Documents/Ares/src") -> List[str]:
    """Use grep to search for patterns in the codebase."""
    try:
        result = subprocess.run(
            ['grep', '-r', '-l', pattern, search_path],
            capture_output=True,
            text=True,
            cwd="/Users/remyroche/Documents/Ares"
        )
        if result.returncode == 0:
            return result.stdout.strip().split('\n')
        return []
    except Exception:
        return []

def verify_file_is_dead(unused_file: Dict) -> Dict:
    """Quick verification if file is truly dead."""
    file_path = unused_file['path']
    module_name = unused_file['module_name']
    file_name = Path(file_path).name

    tprint(f"🔍 Checking: {Path(file_path).name}")

    # Check 1: Direct imports
    direct_import_pattern = f"import {module_name}"
    from_import_pattern = f"from {module_name}"

    direct_files = grep_search(direct_import_pattern)
    from_files = grep_search(from_import_pattern)

    # Remove self-references
    direct_files = [f for f in direct_files if f != file_path]
    from_files = [f for f in from_files if f != file_path]

    # Check 2: Dynamic imports
    dynamic_patterns = [
        f"__import__.*{module_name}",
        f"importlib.*{module_name}",
        f"imp.*{module_name}"
    ]

    dynamic_files = []
    for pattern in dynamic_patterns:
        dynamic_files.extend(grep_search(pattern))

    # Remove self-references
    dynamic_files = [f for f in dynamic_files if f != file_path]

    # Check 3: String references
    string_patterns = [
        f"'{module_name}'",
        f'"{module_name}"',
        f"'{file_name}'",
        f'"{file_name}"'
    ]

    string_files = []
    for pattern in string_patterns:
        string_files.extend(grep_search(pattern))

    # Remove self-references
    string_files = [f for f in string_files if f != file_path]

    # Combine all references
    all_references = set(direct_files + from_files + dynamic_files + string_files)

    # Special check for __init__.py files
    if file_name == '__init__.py':
        # Check if parent directory is imported
        parent_dir = str(Path(file_path).parent)
        parent_name = module_name.rsplit('.', 1)[0] if '.' in module_name else module_name
        parent_files = grep_search(f"import {parent_name}")
        parent_from_files = grep_search(f"from {parent_name}")
        all_references.update(parent_files + parent_from_files)

    is_dead = len(all_references) == 0

    result = {
        'file': file_path,
        'module_name': module_name,
        'is_dead': is_dead,
        'total_references': len(all_references),
        'referencing_files': list(all_references)[:5],  # Limit for readability
        'checks': {
            'direct_imports': len(direct_files),
            'from_imports': len(from_files),
            'dynamic_imports': len(dynamic_files),
            'string_references': len(string_files)
        }
    }

    if is_dead:
        tprint("   ✅ DEAD - No references found")
    else:
        tprint(f"   ⚠️  NOT DEAD - {len(all_references)} references found")

    return result

def main():
    """Main verification process."""
    tprint("🔬 QUICK DEAD CODE VERIFICATION")
    tprint("=" * 80)

    # Get list of unused files
    unused_files = extract_unused_files()
    tprint(f"Found {len(unused_files)} unused files to verify")

    # Quick verification
    results = []
    dead_files = []
    not_dead_files = []

    for i, unused_file in enumerate(unused_files, 1):
        tprint(f"\n[{i:3d}/{len(unused_files)}] ", end="")
        result = verify_file_is_dead(unused_file)
        results.append(result)

        if result['is_dead']:
            dead_files.append(result)
        else:
            not_dead_files.append(result)

    # Generate summary
    tprint("\n📊 VERIFICATION SUMMARY")
    tprint("=" * 80)
    tprint(f"Total files verified: {len(results)}")
    tprint(f"Truly dead files: {len(dead_files)}")
    tprint(f"Files with references: {len(not_dead_files)}")

    # Save results
    output_file = Path("/Users/remyroche/Documents/Ares/quick_verification_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': str(output_file.stat().st_mtime),
            'summary': {
                'total_verified': len(results),
                'dead_files': len(dead_files),
                'not_dead_files': len(not_dead_files)
            },
            'dead_files': [{'file': f['file'], 'module_name': f['module_name']} for f in dead_files],
            'not_dead_files': [{'file': f['file'], 'module_name': f['module_name'], 'references': f['total_references']} for f in not_dead_files]
        }, f, indent=2)

    tprint(f"\n📁 Results saved to: {output_file}")

    # Print dead files (limit to first 30)
    if dead_files:
        tprint("\n✅ TRULY DEAD FILES:")
        for file in dead_files[:30]:
            tprint(f"  • {Path(file['file']).name} ({file['module_name']})")

        if len(dead_files) > 30:
            tprint(f"  ... and {len(dead_files) - 30} more")

    # Print files with references (limit to first 20)
    if not_dead_files:
        tprint("\n⚠️  FILES WITH REFERENCES:")
        for file in not_dead_files[:20]:
            tprint(f"  • {Path(file['file']).name} ({file['module_name']}) - {file['total_references']} refs")

        if len(not_dead_files) > 20:
            tprint(f"  ... and {len(not_dead_files) - 20} more")

    tprint("\n🎯 RECOMMENDATIONS:")
    tprint("=" * 50)
    if dead_files:
        tprint("• Safe to remove dead files after backup")
        tprint("• Consider git history before deletion")
        tprint("• Check if files are needed for future features")

    if not_dead_files:
        tprint("• Files with references should be kept")
        tprint("• Review references to understand dependencies")
        tprint("• Some may be conditionally imported")

if __name__ == "__main__":
    main()
