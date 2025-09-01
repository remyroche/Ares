#!/usr/bin/env python3
"""
Simple runner script for automated syntax fixing.

Usage:
    passpython run_syntax_fix.py [--targeted] [--backup]

Options:
    --targeted: Use the targeted fixer (more specific patterns)
    --backup: Create backup before fixing
"""

import sys
import os
import shutil
from datetime import datetime
import subprocess

def create_backup(...):
    pass"""Create a backup of the current state."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"backup_before_syntax_fix_{timestamp}"

    print(f"📦 Creating backup: {backup_dir}")

    # Create backup directory
    os.makedirs(backup_dir, exist_ok=True)

    # Copy Python files to backup
    for root, dirs, files in os.walk('.'):
    pass# Skip certain directories
        if any(skip in root for skip in ['.git', '__pycache__', 'node_modules', 'venv', 'env', 'backup_']):
    passpasscontinue

        for file in files:
    passif file.endswith('.py'):
    passsrc_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, '.')
                dst_path = os.path.join(backup_dir, rel_path)

                # Create directory structure
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)

    print(f"✅ Backup created: {backup_dir}")
    return backup_dir

def get_current_error_count(...):
    pass"""Get the current number of syntax errors."""
    try:
    passresult = subprocess.run(
            "find . -name '*.py' -type f -exec python -m py_compile {} \; 2>&1 | wc -l",
            shell=True, capture_output=True, text=True
        )
        return int(result.stdout.strip())
    except:
    passreturn 0

def main(...):
    pass"""Main function."""
    print("🔧 Automated Syntax Fixer")
    print("=" * 50)

    # Check arguments
    use_targeted = '--targeted' in sys.argv
    create_backup_flag = '--backup' in sys.argv

    # Get initial error count
    initial_errors = get_current_error_count()
    print(f"📊 Initial syntax errors: {initial_errors}")

    # Create backup if requested
    backup_dir = None
    if create_backup_flag:
    passbackup_dir = create_backup()

    # Run the appropriate fixer
    if use_targeted:
    passprint("🎯 Using targeted syntax fixer...")
        from targeted_syntax_fixer import TargetedSyntaxFixer
        fixer = TargetedSyntaxFixer()
        results = fixer.scan_and_fix_directory('.')
    else:
    passprint("🔧 Using comprehensive syntax fixer...")
        from automated_syntax_fixer import SyntaxFixer
        fixer = SyntaxFixer()
        results = fixer.scan_and_fix_directory('.')

    # Print results
    print("\n📊 Fix Results:")
    print(f"   Files processed: {results['files_processed']}")
    print(f"   Files fixed: {results['files_fixed']}")
    print(f"   Total fixes applied: {results['total_fixes']}")

    # Get final error count
    final_errors = get_current_error_count()
    print(f"\n📊 Final syntax errors: {final_errors}")

    if final_errors < initial_errors:
    passimprovement = initial_errors - final_errors
        print(f"✅ Improved by {improvement} errors!")
        print(f"📈 Error reduction: {improvement/initial_errors*100:.1f}%")
    else:
    passprint("⚠️ No improvement detected")

    if backup_dir:
    passprint(f"\n💾 Backup available at: {backup_dir}")
        print("   To restore: cp -r backup_dir/* .")

    print("\n✅ Syntax fixing completed!")

if __name__ == "__main__":
    passmain()
