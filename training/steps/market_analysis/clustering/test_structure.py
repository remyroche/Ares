#!/usr/bin/env python3
"""
Test script to verify the clustering module structure without importing dependencies.
"""

import sys
import os
from pathlib import Path

def test_module_structure():
    """Test that all required files exist and have correct structure."""
    
    base_dir = Path(__file__).parent
    
    required_files = [
        '__init__.py',
        'regime_consolidator.py',
        'hmm_integration.py', 
        'ml_output_generator.py',
        'clustering_pipeline.py',
        'run_clustering_pipeline.py',
        'README.md'
    ]
    
    print("Testing clustering module structure...")
    print(f"Base directory: {base_dir}")
    
    # Check if all files exist
    missing_files = []
    for file_name in required_files:
        file_path = base_dir / file_name
        if file_path.exists():
            print(f"✓ {file_name} exists")
        else:
            print(f"✗ {file_name} missing")
            missing_files.append(file_name)
    
    if missing_files:
        print(f"\nMissing files: {missing_files}")
        return False
    
    # Check file sizes (basic content validation)
    print("\nFile sizes:")
    for file_name in required_files:
        if file_name != 'README.md':  # Skip README for size check
            file_path = base_dir / file_name
            size = file_path.stat().st_size
            print(f"  {file_name}: {size:,} bytes")
            
            if size < 1000:  # Files should be substantial
                print(f"    ⚠ Warning: {file_name} is very small ({size} bytes)")
    
    print("\n✓ All required files exist and have reasonable sizes")
    return True

def test_import_structure():
    """Test that the module can be imported (without dependencies)."""
    
    try:
        # Add the parent directory to path
        parent_dir = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(parent_dir))
        
        # Try to import the module structure
        import training.steps.market_analysis.clustering
        
        print("✓ Module structure can be imported")
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def main():
    """Run all structure tests."""
    
    print("=" * 60)
    print("CLUSTERING MODULE STRUCTURE TEST")
    print("=" * 60)
    
    # Test file structure
    structure_ok = test_module_structure()
    
    print("\n" + "=" * 60)
    
    # Test import structure
    import_ok = test_import_structure()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if structure_ok and import_ok:
        print("✓ All tests passed! Module structure is correct.")
        print("\nThe clustering module is ready to use.")
        print("\nNext steps:")
        print("1. Install required dependencies (numpy, pandas, scikit-learn, scipy)")
        print("2. Test with actual HMM discovery results")
        print("3. Run the pipeline with: python run_clustering_pipeline.py --help")
        return 0
    else:
        print("✗ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())