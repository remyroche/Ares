#!/usr/bin/env python3
"""
Test script to verify that feature selection results are saved to outcomes/ directory.
"""
import sys
from pathlib import Path
import tempfile
import os

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_feature_selection_paths():
    """Test that feature selection modules use outcomes/ directory."""

    print("🧪 Testing feature selection path configuration...")

    # Test 1: Check final_feature_selection_pipeline.py configuration
    try:
        from src.training.steps.market_analysis.final_feature_selection_pipeline import FeatureSelectionConfig

        config = FeatureSelectionConfig()
        print(f"✅ FeatureSelectionConfig output_directory: {config.output_directory}")

        if config.output_directory == "outcomes":
            print("✅ final_feature_selection_pipeline.py correctly configured to use outcomes/")
        else:
            print("❌ final_feature_selection_pipeline.py still uses old directory structure")
            return False

    except Exception as e:
        print(f"❌ Error testing final_feature_selection_pipeline.py: {e}")
        return False

    # Test 2: Check main_framework.py path generation
    try:
        from src.training.utils.feature_selection.main_framework import FeatureSelectionFramework

        # Check that the _generate_csv_output method would create paths in outcomes/
        import inspect
        source = inspect.getsource(FeatureSelectionFramework._generate_csv_output)

        if "outcomes" in source and "market_analysis_feature_selection_outcome" in source:
            print("✅ main_framework.py correctly configured to save to outcomes/")
        else:
            print("❌ main_framework.py still uses old directory structure")
            return False

    except Exception as e:
        print(f"❌ Error testing main_framework.py: {e}")
        return False

    print("✅ All feature selection path tests passed!")
    return True

def test_outcomes_directory():
    """Test that outcomes directory exists and is writable."""
    outcomes_dir = Path("outcomes")

    if outcomes_dir.exists():
        print("✅ outcomes/ directory exists")
    else:
        print("📁 outcomes/ directory does not exist, creating it...")
        outcomes_dir.mkdir(parents=True, exist_ok=True)
        print("✅ outcomes/ directory created")

    # Test writing a small file
    try:
        test_file = outcomes_dir / "test_feature_selection_path.txt"
        with open(test_file, 'w') as f:
            f.write("Test file to verify outcomes/ directory is writable\n")
        print("✅ outcomes/ directory is writable")

        # Clean up test file
        test_file.unlink()
        print("✅ Test file cleaned up")

    except Exception as e:
        print(f"❌ Error testing outcomes/ directory writability: {e}")
        return False

    return True

if __name__ == "__main__":
    print("🔍 Testing feature selection output path configuration...\n")

    success = True
    success &= test_feature_selection_paths()
    success &= test_outcomes_directory()

    if success:
        print("\n🎉 All tests passed! Feature selection results will be saved to outcomes/ directory.")
    else:
        print("\n❌ Some tests failed. Please check the configuration.")
        sys.exit(1)
