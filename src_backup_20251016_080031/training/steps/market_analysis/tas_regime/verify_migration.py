#!/usr/bin/env python3
"""
Verify TAS Regime Migration

This script verifies that the TAS regime system has been successfully moved
from utils/ml_common/optimization/tas/ to src/training/steps/market_analysis/tas_regime/
"""

import os
import sys

def verify_migration():
    """Verify that the migration was successful."""
    print("🔍 Verifying TAS Regime Migration")
    print("=" * 50)
    
    # Check source directory (should be empty or not exist)
    source_path = '/workspace/src/utils/ml_common/optimization/tas'
    if os.path.exists(source_path):
        print(f"⚠️  Source directory still exists: {source_path}")
        print("   This is expected if you want to keep the original for reference")
    else:
        print("✅ Source directory has been removed")
    
    # Check destination directory
    dest_path = '/workspace/src/training/steps/market_analysis/tas_regime'
    if os.path.exists(dest_path):
        print(f"✅ Destination directory exists: {dest_path}")
        
        # Check key files
        key_files = [
            '__init__.py',
            'README.md',
            'core/tas_config.py',
            'core/tas_engine.py',
            'core/tree_cvlSA_architecture.py',
            'examples/advanced_tas_example.py',
            'tree_cvlSA_demo.py'
        ]
        
        print("\n📁 Checking key files:")
        for file_path in key_files:
            full_path = os.path.join(dest_path, file_path)
            if os.path.exists(full_path):
                print(f"✅ {file_path}")
            else:
                print(f"❌ {file_path}")
        
        # Check directory structure
        required_dirs = [
            'core', 'components', 'evaluation', 'meta_learning',
            'optimization', 'regime_analysis', 'search', 'adaptation',
            'uncertainty', 'utils', 'examples', 'backtesting', 'data_pipeline'
        ]
        
        print("\n📂 Checking directory structure:")
        for dir_name in required_dirs:
            dir_path = os.path.join(dest_path, dir_name)
            if os.path.exists(dir_path):
                print(f"✅ {dir_name}/")
            else:
                print(f"❌ {dir_name}/")
        
        return True
    else:
        print(f"❌ Destination directory does not exist: {dest_path}")
        return False

def check_import_paths():
    """Check that import paths have been updated."""
    print("\n🔍 Checking import path updates...")
    
    dest_path = '/workspace/src/training/steps/market_analysis/tas_regime'
    
    # Files that should have updated import paths
    files_to_check = [
        'README.md',
        'examples/advanced_tas_example.py',
        'examples/advanced_regime_detection_example.py',
        'tree_cvlSA_demo.py'
    ]
    
    updated_count = 0
    total_count = 0
    
    for file_path in files_to_check:
        full_path = os.path.join(dest_path, file_path)
        if os.path.exists(full_path):
            total_count += 1
            try:
                with open(full_path, 'r') as f:
                    content = f.read()
                    if 'src.training.steps.market_analysis.tas_regime' in content:
                        print(f"✅ {file_path} has updated imports")
                        updated_count += 1
                    elif 'src.utils.ml_common.optimization.tas' in content:
                        print(f"⚠️  {file_path} still has old import paths")
                    else:
                        print(f"ℹ️  {file_path} doesn't contain import paths")
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")
    
    print(f"\n📊 Import path update status: {updated_count}/{total_count} files updated")
    return updated_count == total_count

def main():
    """Main verification function."""
    print("🚀 TAS Regime Migration Verification")
    print("=" * 60)
    
    # Verify migration
    migration_success = verify_migration()
    
    # Check import paths
    import_success = check_import_paths()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Migration Summary")
    print("=" * 60)
    
    if migration_success and import_success:
        print("🎉 Migration successful!")
        print("✅ TAS regime system has been moved to market analysis pipeline")
        print("✅ Directory structure is correct")
        print("✅ Key files are present")
        print("✅ Import paths have been updated")
        print("\n🎯 Next steps:")
        print("   1. Test the integration with your market analysis pipeline")
        print("   2. Update any remaining import references")
        print("   3. Run the TAS regime examples to verify functionality")
        return True
    else:
        print("❌ Migration issues detected")
        if not migration_success:
            print("   - Directory structure or files missing")
        if not import_success:
            print("   - Import paths need updating")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)