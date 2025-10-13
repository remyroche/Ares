#!/usr/bin/env python3
"""
Cleanup Validation Script

This script validates that the duplicate cleanup was successful and that
the base class methods are properly available.
"""

import os
import sys
import re
from pathlib import Path

def check_duplicate_methods():
    """Check for remaining duplicate methods in categories."""
    print("🔍 Checking for remaining duplicate methods...")
    
    categories_dir = Path("src/feature_generation/categories")
    if not categories_dir.exists():
        print("❌ Categories directory not found")
        return False
    
    duplicate_count = 0
    files_with_duplicates = []
    
    for py_file in categories_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for duplicate methods
            optimize_matches = len(re.findall(r'def optimize_dataframe_processing', content))
            rolling_matches = len(re.findall(r'def vectorized_rolling_operations', content))
            
            if optimize_matches > 0 or rolling_matches > 0:
                files_with_duplicates.append({
                    'file': py_file.name,
                    'optimize': optimize_matches,
                    'rolling': rolling_matches
                })
                duplicate_count += optimize_matches + rolling_matches
                
        except Exception as e:
            print(f"⚠️ Error reading {py_file.name}: {e}")
    
    if duplicate_count == 0:
        print("✅ No duplicate methods found in categories")
        return True
    else:
        print(f"❌ Found {duplicate_count} duplicate methods in {len(files_with_duplicates)} files:")
        for file_info in files_with_duplicates:
            print(f"  - {file_info['file']}: {file_info['optimize']} optimize, {file_info['rolling']} rolling")
        return False

def check_base_class_methods():
    """Check that base class has the required methods."""
    print("\n🔍 Checking base class methods...")
    
    base_class_file = Path("src/feature_generation/core/feature_generator.py")
    if not base_class_file.exists():
        print("❌ Base class file not found")
        return False
    
    try:
        with open(base_class_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for required methods
        has_optimize = 'def optimize_dataframe_processing(' in content
        has_rolling = 'def vectorized_rolling_operations(' in content
        
        if has_optimize and has_rolling:
            print("✅ Base class has required methods")
            return True
        else:
            print("❌ Base class missing required methods:")
            if not has_optimize:
                print("  - optimize_dataframe_processing")
            if not has_rolling:
                print("  - vectorized_rolling_operations")
            return False
            
    except Exception as e:
        print(f"❌ Error reading base class file: {e}")
        return False

def check_utility_mixins():
    """Check that utility mixins were created."""
    print("\n🔍 Checking utility mixins...")
    
    mixin_files = [
        "src/feature_generation/core/optimization_mixin.py",
        "src/feature_generation/core/rolling_operations_mixin.py"
    ]
    
    all_exist = True
    for mixin_file in mixin_files:
        if Path(mixin_file).exists():
            print(f"✅ {mixin_file} exists")
        else:
            print(f"❌ {mixin_file} missing")
            all_exist = False
    
    return all_exist

def check_factory_pattern():
    """Check that factory pattern was implemented."""
    print("\n🔍 Checking factory pattern...")
    
    factory_file = Path("src/feature_generation/core/generator_factory.py")
    if factory_file.exists():
        print("✅ Generator factory exists")
        return True
    else:
        print("❌ Generator factory missing")
        return False

def check_documentation():
    """Check that documentation was updated."""
    print("\n🔍 Checking documentation...")
    
    doc_files = [
        "src/feature_generation/MIGRATION_GUIDE.md",
        "src/feature_generation/CLEANUP_RESULTS.md"
    ]
    
    all_exist = True
    for doc_file in doc_files:
        if Path(doc_file).exists():
            print(f"✅ {doc_file} exists")
        else:
            print(f"❌ {doc_file} missing")
            all_exist = False
    
    return all_exist

def check_file_sizes():
    """Check file sizes to estimate cleanup impact."""
    print("\n🔍 Checking file sizes...")
    
    categories_dir = Path("src/feature_generation/categories")
    if not categories_dir.exists():
        print("❌ Categories directory not found")
        return False
    
    total_lines = 0
    file_count = 0
    
    for py_file in categories_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
            total_lines += lines
            file_count += 1
        except Exception as e:
            print(f"⚠️ Error reading {py_file.name}: {e}")
    
    print(f"📊 Categories: {file_count} files, {total_lines:,} total lines")
    
    # Check base class size
    base_class_file = Path("src/feature_generation/core/feature_generator.py")
    if base_class_file.exists():
        with open(base_class_file, 'r', encoding='utf-8') as f:
            base_lines = len(f.readlines())
        print(f"📊 Base class: {base_lines:,} lines")
    
    return True

def main():
    """Run all validation checks."""
    print("🚀 Feature Generation Cleanup Validation")
    print("=" * 50)
    
    checks = [
        ("Duplicate Methods", check_duplicate_methods),
        ("Base Class Methods", check_base_class_methods),
        ("Utility Mixins", check_utility_mixins),
        ("Factory Pattern", check_factory_pattern),
        ("Documentation", check_documentation),
        ("File Sizes", check_file_sizes)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Error in {name}: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 50)
    print("📋 Validation Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All validation checks passed! Cleanup was successful.")
        return 0
    else:
        print("⚠️ Some validation checks failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())