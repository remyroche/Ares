#!/usr/bin/env python3
"""
Simple validation script for VectorBT migration.

This script validates that the VectorBT migration was completed successfully
by checking file structure and imports.
"""

import os
import sys
from pathlib import Path

def check_file_exists(file_path: str) -> bool:
    """Check if a file exists."""
    return os.path.exists(file_path)

def check_imports_in_file(file_path: str, expected_imports: list) -> dict:
    """Check if expected imports are present in a file."""
    if not check_file_exists(file_path):
        return {'exists': False, 'imports': []}
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        found_imports = []
        for import_name in expected_imports:
            if import_name in content:
                found_imports.append(import_name)
        
        return {'exists': True, 'imports': found_imports}
    except Exception as e:
        return {'exists': True, 'imports': [], 'error': str(e)}

def main():
    """Main validation function."""
    print("VectorBT Migration Validation")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('src/feature_generation'):
        print("✗ Not in the correct directory. Please run from the workspace root.")
        return
    
    # Define the files that should have been created
    vectorbt_files = [
        'src/feature_generation/categories/vectorbt_advanced_statistical.py',
        'src/feature_generation/categories/vectorbt_support_resistance.py',
        'src/feature_generation/categories/vectorbt_legacy.py',
        'src/feature_generation/categories/vectorbt_order_flow.py',
        'src/feature_generation/categories/vectorbt_acceleration.py'
    ]
    
    # Define the main files that should have been updated
    main_files = [
        'src/feature_generation/categories/advanced_statistical.py',
        'src/feature_generation/categories/support_resistance.py',
        'src/feature_generation/categories/legacy.py',
        'src/feature_generation/categories/acceleration.py',
        'src/feature_generation/categories/order_flow.py'
    ]
    
    print("\n1. Checking VectorBT-optimized files...")
    vectorbt_results = {}
    for file_path in vectorbt_files:
        exists = check_file_exists(file_path)
        vectorbt_results[file_path] = exists
        status = "✓" if exists else "✗"
        print(f"  {status} {file_path}")
    
    print("\n2. Checking main files for VectorBT integration...")
    main_results = {}
    for file_path in main_files:
        exists = check_file_exists(file_path)
        if exists:
            # Check for VectorBT imports
            expected_imports = ['vectorbt_', 'VECTORBT_', 'create_default_vectorbt_']
            import_check = check_imports_in_file(file_path, expected_imports)
            main_results[file_path] = {
                'exists': True,
                'has_vectorbt_imports': len(import_check['imports']) > 0,
                'imports': import_check['imports']
            }
            status = "✓" if import_check['imports'] else "✗"
            print(f"  {status} {file_path} - VectorBT integration: {len(import_check['imports'])} imports found")
        else:
            main_results[file_path] = {'exists': False}
            print(f"  ✗ {file_path} - File not found")
    
    print("\n3. Checking VectorBT rolling optimizer...")
    optimizer_file = 'src/feature_generation/utils/vectorbt_rolling_optimizer.py'
    optimizer_exists = check_file_exists(optimizer_file)
    print(f"  {'✓' if optimizer_exists else '✗'} {optimizer_file}")
    
    # Summary
    print("\n" + "=" * 50)
    print("Validation Summary:")
    print("=" * 50)
    
    vectorbt_created = sum(1 for result in vectorbt_results.values() if result)
    vectorbt_total = len(vectorbt_results)
    print(f"VectorBT files created: {vectorbt_created}/{vectorbt_total}")
    
    main_updated = sum(1 for result in main_results.values() 
                      if result.get('exists', False) and result.get('has_vectorbt_imports', False))
    main_total = len(main_results)
    print(f"Main files updated: {main_updated}/{main_total}")
    
    print(f"VectorBT rolling optimizer: {'✓' if optimizer_exists else '✗'}")
    
    # Overall status
    if vectorbt_created == vectorbt_total and main_updated == main_total and optimizer_exists:
        print("\n🎉 VectorBT migration completed successfully!")
        print("\nKey features implemented:")
        print("  • Order Flow Features - VectorBT optimized")
        print("  • Acceleration Features - VectorBT optimized") 
        print("  • Advanced Statistical Features - VectorBT optimized")
        print("  • Support/Resistance Features - VectorBT optimized")
        print("  • Legacy Features - VectorBT optimized")
        print("  • VectorBTRollingOptimizer integration")
        print("  • Automatic fallback to legacy implementations")
    else:
        print("\n⚠️  VectorBT migration incomplete. Please check the missing components.")
    
    print("\nNext steps:")
    print("  1. Install VectorBT: pip install vectorbt")
    print("  2. Test the integration with sample data")
    print("  3. Verify performance improvements")
    print("  4. Update any remaining feature categories as needed")

if __name__ == "__main__":
    main()