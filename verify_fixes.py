#!/usr/bin/env python3
"""
Script to verify the fixes applied to import conflicts and signature issues.
"""

import os
import json
from pathlib import Path
from datetime import datetime
import subprocess


def count_python_files(exclude_dirs=None):
    """Count Python files in the codebase."""
    if exclude_dirs is None:
        exclude_dirs = {'syntax_fix_backups', 'syntax_fix_backups_v2', '__pycache__', '.git'}
    
    count = 0
    for root, dirs, files in os.walk('.'):
        # Remove excluded directories from traversal
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if file.endswith('.py'):
                count += 1
    
    return count


def check_import_conflicts():
    """Quick check for common import conflict patterns."""
    conflicts = {
        'system_logger': 0,
        'run_step': 0,
        'handles_errors': 0,
        'get_default_config': 0,
        'Callable': 0
    }
    
    exclude_dirs = {'syntax_fix_backups', 'syntax_fix_backups_v2', '__pycache__', '.git'}
    
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for each conflict pattern
                    for name in conflicts:
                        # Count import statements for this name
                        import_count = content.count(f'import {name}')
                        if import_count > 1:
                            conflicts[name] += 1
                
                except Exception:
                    pass
    
    return conflicts


def analyze_fix_reports():
    """Analyze the fix reports generated."""
    reports = []
    
    # Find all fix reports
    for file in os.listdir('.'):
        if file.endswith('_fix_report_') and file.endswith('.json'):
            reports.append(file)
    
    reports.sort()
    
    total_fixes = 0
    for report_file in reports:
        try:
            with open(report_file, 'r') as f:
                data = json.load(f)
            
            if 'files_fixed' in data:
                fixes = data['files_fixed']
            elif 'import_fixes' in data:
                fixes = data['import_fixes']
            else:
                fixes = 0
            
            total_fixes += fixes
            print(f"\nReport: {report_file}")
            print(f"  Timestamp: {data.get('timestamp', 'Unknown')}")
            print(f"  Files fixed: {fixes}")
            
        except Exception as e:
            print(f"Error reading {report_file}: {e}")
    
    return total_fixes


def main():
    """Main verification function."""
    print("Import and Signature Fix Verification")
    print("=" * 50)
    
    # Count Python files
    total_files = count_python_files()
    print(f"\nTotal Python files (excluding backups): {total_files}")
    
    # Check for remaining conflicts
    print("\nChecking for remaining import patterns...")
    conflicts = check_import_conflicts()
    
    print("\nPotential remaining conflicts:")
    for name, count in conflicts.items():
        print(f"  {name}: {count} files")
    
    # Analyze fix reports
    print("\nFix Reports Summary:")
    total_fixes = analyze_fix_reports()
    
    print(f"\nTotal fixes applied: {total_fixes}")
    
    # Check backup directories
    backup_dirs = [
        'import_fix_backups',
        'signature_fix_backups',
        'comprehensive_import_fix_backups'
    ]
    
    print("\nBackup directories created:")
    for backup_dir in backup_dirs:
        if os.path.exists(backup_dir):
            backup_count = len([f for f in os.listdir(backup_dir) if f.endswith('.backup')])
            print(f"  {backup_dir}: {backup_count} backups")
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"  Python files processed: {total_files}")
    print(f"  Total fixes applied: {total_fixes}")
    print(f"  Fix rate: {(total_fixes / total_files * 100):.1f}%")
    
    if sum(conflicts.values()) > 0:
        print("\n⚠️  Some potential conflicts may remain. Manual review recommended for:")
        for name, count in conflicts.items():
            if count > 0:
                print(f"    - {name} imports ({count} files)")
    else:
        print("\n✅  No obvious import conflicts detected!")


if __name__ == "__main__":
    main()