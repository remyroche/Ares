"""
Simple test script for the feature comparison framework.
This script tests the basic functionality without requiring external dependencies.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from feature_comparison_utils import FeatureComparisonUtils
        print("✓ FeatureComparisonUtils imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import FeatureComparisonUtils: {e}")
        return False
    
    try:
        from feature_versions import FeatureVersions
        print("✓ FeatureVersions imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import FeatureVersions: {e}")
        return False
    
    try:
        from relevance_analyzer import RelevanceAnalyzer
        print("✓ RelevanceAnalyzer imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import RelevanceAnalyzer: {e}")
        return False
    
    try:
        from comparison_report import ComparisonReport
        print("✓ ComparisonReport imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ComparisonReport: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    print("\nTesting basic functionality...")
    
    try:
        from feature_comparison_utils import FeatureComparisonUtils
        
        # Test initialization
        utils = FeatureComparisonUtils()
        print("✓ FeatureComparisonUtils initialized successfully")
        
        # Test module paths
        print(f"✓ Feature modules configured: {list(utils.feature_modules.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in basic functionality test: {e}")
        return False

def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting file structure...")
    
    required_files = [
        '__init__.py',
        'feature_comparison_utils.py',
        'feature_versions.py',
        'relevance_analyzer.py',
        'comparison_report.py',
        'run_comparison.py',
        'example_usage.py',
        'README.md',
        'requirements.txt'
    ]
    
    all_exist = True
    for file_name in required_files:
        file_path = Path(__file__).parent / file_name
        if file_path.exists():
            print(f"✓ {file_name} exists")
        else:
            print(f"✗ {file_name} missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests."""
    print("Feature Comparison Framework - Test Suite")
    print("=" * 50)
    
    # Test file structure
    structure_ok = test_file_structure()
    
    # Test imports (will fail without dependencies)
    imports_ok = test_imports()
    
    # Test basic functionality
    functionality_ok = test_basic_functionality()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"File Structure: {'PASS' if structure_ok else 'FAIL'}")
    print(f"Imports: {'PASS' if imports_ok else 'FAIL (requires dependencies)'}")
    print(f"Basic Functionality: {'PASS' if functionality_ok else 'FAIL'}")
    
    if structure_ok and functionality_ok:
        print("\n✓ Framework structure is correct!")
        print("To run full analysis, install dependencies:")
        print("pip install -r requirements.txt")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())