#!/usr/bin/env python3
"""
Script to safely remove unused test, demo, and fix files.
"""

import json
import os
import shutil
from pathlib import Path

def load_unused_files():
    """Load the categorized unused files."""
    with open('/workspace/unused_scripts_analysis.json', 'r') as f:
        data = json.load(f)
    
    # Get the full list and categorize
    unused_files = data['unused_files']
    
    test_files = []
    demo_files = []
    fix_files = []
    
    for file_path in unused_files:
        filename = os.path.basename(file_path)
        
        if filename.startswith('test_') or 'test' in filename.lower():
            test_files.append(file_path)
        elif filename.startswith('demo_') or 'demo' in filename.lower():
            demo_files.append(file_path)
        elif any(word in filename.lower() for word in ['fix', 'repair', 'correct']):
            fix_files.append(file_path)
    
    return test_files, demo_files, fix_files

def create_backup():
    """Create a backup directory for removed files."""
    backup_dir = "/workspace/removed_scripts_backup"
    os.makedirs(backup_dir, exist_ok=True)
    return backup_dir

def remove_files_safely(files, category, backup_dir):
    """Safely remove files with backup."""
    removed_count = 0
    failed_count = 0
    
    print(f"\n🗑️  Removing {category} files...")
    print("=" * 50)
    
    for file_path in files:
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"⚠️  File not found: {file_path}")
                continue
            
            # Create backup
            backup_path = os.path.join(backup_dir, category, file_path.lstrip('/'))
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            shutil.copy2(file_path, backup_path)
            
            # Remove original file
            os.remove(file_path)
            print(f"✅ Removed: {file_path}")
            removed_count += 1
            
        except Exception as e:
            print(f"❌ Failed to remove {file_path}: {e}")
            failed_count += 1
    
    print(f"\n📊 {category} removal summary:")
    print(f"  ✅ Successfully removed: {removed_count}")
    print(f"  ❌ Failed to remove: {failed_count}")
    
    return removed_count, failed_count

def main():
    print("🚀 Starting unused script removal process")
    print("=" * 60)
    
    # Load files to remove
    test_files, demo_files, fix_files = load_unused_files()
    
    print(f"📁 Files to remove:")
    print(f"  🧪 Test files: {len(test_files)}")
    print(f"  📚 Demo files: {len(demo_files)}")
    print(f"  🔧 Fix files: {len(fix_files)}")
    print(f"  📊 Total: {len(test_files) + len(demo_files) + len(fix_files)}")
    
    # Create backup directory
    backup_dir = create_backup()
    print(f"\n💾 Backup directory: {backup_dir}")
    
    # Remove files
    total_removed = 0
    total_failed = 0
    
    if test_files:
        removed, failed = remove_files_safely(test_files, "test_files", backup_dir)
        total_removed += removed
        total_failed += failed
    
    if demo_files:
        removed, failed = remove_files_safely(demo_files, "demo_files", backup_dir)
        total_removed += removed
        total_failed += failed
    
    if fix_files:
        removed, failed = remove_files_safely(fix_files, "fix_files", backup_dir)
        total_removed += removed
        total_failed += failed
    
    print(f"\n{'='*60}")
    print("🎯 REMOVAL COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Total files removed: {total_removed}")
    print(f"❌ Total files failed: {total_failed}")
    print(f"💾 Backup location: {backup_dir}")
    
    if total_failed == 0:
        print("\n🎉 All files removed successfully!")
    else:
        print(f"\n⚠️  {total_failed} files could not be removed. Check the output above.")

if __name__ == "__main__":
    main()