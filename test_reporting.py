#!/usr/bin/env python3
"""
Test script to verify that the reporting functionality works correctly.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_reporting():
    """Test the reporting functionality."""
    print("🧪 Testing reporting functionality...")

    try:
        # Test import
        from src.training.reports import save_training_report, CentralizedReportManager
        print("✅ Import successful")

        # Test basic report saving
        test_data = {
            "test": "data",
            "timestamp": "2025-09-07",
            "status": "success"
        }

        # Save a test report
        result_path = save_training_report(
            data=test_data,
            step_name="test_step",
            report_type="test_report",
            symbol="TEST",
            file_format="json"
        )

        if result_path:
            print(f"✅ Test report saved: {result_path}")

            # Verify file exists
            if os.path.exists(result_path):
                print("✅ File exists and is accessible")
                return True
            else:
                print("❌ File was not created")
                return False
        else:
            print("❌ Report saving returned None")
            return False

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_reporting()
    if success:
        print("🎉 Reporting functionality test PASSED")
        sys.exit(0)
    else:
        print("💥 Reporting functionality test FAILED")
        sys.exit(1)
