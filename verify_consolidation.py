#!/usr/bin/env python3
"""
Verification Script: Check consolidation completeness

This script verifies that the consolidation is complete and all references are correct.
"""

import os
from pathlib import Path

def check_structure():
    """Check that the new structure exists."""
    print("🏗️ Checking structure...")
    
    checks = [
        ("src/feature_generation/utils", "Utils directory exists"),
        ("src/feature_generation/utils/optimization", "Optimization directory exists"),
        ("src/feature_generation/utils/step06_utility_container.py", "Step06 utilities moved"),
        ("src/feature_generation/utils/feature_generators.py", "Feature generators moved"),
        ("src/feature_generation/utils/math_validation.py", "Math validation moved"),
        ("src/feature_generation/utils/optimization/unified_optimizer.py", "Unified optimizer exists"),
        ("src/feature_generation/compatibility/hmm_compatibility.py", "HMM compatibility exists"),
    ]
    
    all_good = True
    for path_str, description in checks:
        if Path(path_str).exists():
            print(f"  ✅ {description}")
        else:
            print(f"  ❌ {description}")
            all_good = False
    
    return all_good

def check_imports():
    """Check for remaining feature_engineering imports."""
    print("\n🔍 Checking for remaining feature_engineering imports...")
    
    problematic_files = []
    
    for root, dirs, files in os.walk("src"):
        # Skip backup directories
        if 'backup' in root:
            continue
            
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for problematic imports
                    if 'from src.feature_engineering' in content or 'import src.feature_engineering' in content:
                        problematic_files.append(file_path)
                
                except Exception:
                    pass
    
    if problematic_files:
        print(f"  ⚠️ Found {len(problematic_files)} files with feature_engineering imports:")
        for file_path in problematic_files:
            print(f"    - {file_path}")
        return False
    else:
        print("  ✅ No problematic imports found")
        return True

def check_key_integrations():
    """Check key integration points."""
    print("\n🔗 Checking key integration points...")
    
    integrations = [
        ("src/feature_generation/__init__.py", "Main package init"),
        ("src/feature_generation/utils/__init__.py", "Utils package init"),
        ("src/feature_generation/core/feature_bank.py", "Feature bank integration"),
        ("src/feature_generation/compatibility/hmm_compatibility.py", "HMM compatibility"),
    ]
    
    all_good = True
    for file_path_str, description in integrations:
        file_path = Path(file_path_str)
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for proper utils references
                if 'feature_generation.utils' in content or 'from .utils' in content or 'from ..utils' in content:
                    print(f"  ✅ {description} - properly references utils")
                else:
                    print(f"  ⚠️ {description} - may need utils references")
            except Exception as e:
                print(f"  ❌ {description} - error reading: {e}")
                all_good = False
        else:
            print(f"  ❌ {description} - file missing")
            all_good = False
    
    return all_good

def main():
    """Main verification function."""
    print("🚀 Starting Consolidation Verification")
    print("=" * 40)
    
    structure_ok = check_structure()
    imports_ok = check_imports()
    integrations_ok = check_key_integrations()
    
    print("\n📊 Verification Summary:")
    print(f"  Structure: {'✅ OK' if structure_ok else '❌ Issues'}")
    print(f"  Imports: {'✅ OK' if imports_ok else '❌ Issues'}")
    print(f"  Integrations: {'✅ OK' if integrations_ok else '❌ Issues'}")
    
    if structure_ok and imports_ok and integrations_ok:
        print("\n🎉 Consolidation verification PASSED!")
        print("All feature_engineering references have been properly updated.")
        return True
    else:
        print("\n⚠️ Consolidation verification found issues.")
        print("Some references may need manual fixing.")
        return False

if __name__ == "__main__":
    main()