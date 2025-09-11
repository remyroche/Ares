#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Test script for directory analysis functionality.

This script tests the directory analysis capabilities of the UnifiedQualityOrchestrator.
"""

import os
import sys
import tempfile
from pathlib import Path


# Add the parent directory to the path to import the orchestrator
sys.path.append(str(Path(__file__).parent))

from unified_quality_orchestrator import UnifiedQualityOrchestrator
import numpy as np
import pandas as pd


def create_test_data():
    """Create simple test data."""
    # Create sample data
    data = pd.DataFrame({
        "timestamp": range(100),
        "value1": np.random.randn(100),
        "value2": np.random.randn(100),
        "value3": np.random.randn(100),
    })

    # Add some quality issues
    data.loc[10:15, "value1"] = np.nan
    data.loc[20, "value2"] = np.inf
    data["constant"] = 42  # Constant column

    return data


def test_directory_analysis():
    """Test directory analysis functionality."""
    tprint("🧪 Testing directory analysis functionality...")

    # Create temporary directory with test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test data
        test_data = create_test_data()

        # Save files in different formats
        csv_file = temp_path / "test_data.csv"
        parquet_file = temp_path / "test_data.parquet"
        json_file = temp_path / "test_data.json"

        test_data.to_csv(csv_file, index=False)
        test_data.to_parquet(parquet_file, index=False)
        test_data.to_json(json_file, orient="records")

        tprint(f"✅ Created test files in: {temp_path}")
        tprint(f"   - {csv_file.name}")
        tprint(f"   - {parquet_file.name}")
        tprint(f"   - {json_file.name}")

        # Initialize orchestrator
        orchestrator = UnifiedQualityOrchestrator()

        # Test 1: Quick directory scan
        tprint("\n📊 Test 1: Quick directory scan")
        try:
            scan_summary = orchestrator.get_directory_summary(str(temp_path), recursive=True)

            assert scan_summary["total_files"] == 3, f"Expected 3 files, got {scan_summary['total_files']}"
            assert scan_summary["file_types"][".csv"]["count"] == 1, "Expected 1 CSV file"
            assert scan_summary["file_types"][".parquet"]["count"] == 1, "Expected 1 Parquet file"
            assert scan_summary["file_types"][".json"]["count"] == 1, "Expected 1 JSON file"

            tprint("   ✅ Quick directory scan passed")
            tprint(f"   - Total files: {scan_summary['total_files']}")
            tprint(f"   - Total size: {scan_summary['total_size_mb']:.2f} MB")

        except Exception as e:
            tprint(f"   ❌ Quick directory scan failed: {e}")
            return False

        # Test 2: Full directory analysis
        tprint("\n📊 Test 2: Full directory analysis")
        try:
            directory_report = orchestrator.analyze_directory(str(temp_path), recursive=True)

            if "error" in directory_report:
                tprint(f"   ❌ Directory analysis failed: {directory_report['error']}")
                return False

            summary = directory_report.get("summary", {})

            assert summary["total_files"] == 3, f"Expected 3 files, got {summary['total_files']}"
            assert summary["successful_analyses"] == 3, f"Expected 3 successful analyses, got {summary['successful_analyses']}"
            assert summary["failed_analyses"] == 0, f"Expected 0 failed analyses, got {summary['failed_analyses']}"

            tprint("   ✅ Full directory analysis passed")
            tprint(f"   - Total files: {summary['total_files']}")
            tprint(f"   - Successful analyses: {summary['successful_analyses']}")
            tprint(f"   - Success rate: {summary['success_rate']:.1%}")
            tprint(f"   - Overall Quality: {summary['overall_quality'].upper()}")

        except Exception as e:
            tprint(f"   ❌ Full directory analysis failed: {e}")
            return False

        # Test 3: File pattern filtering
        tprint("\n📊 Test 3: File pattern filtering")
        try:
            csv_only_report = orchestrator.analyze_directory(str(temp_path), file_pattern="*.csv", recursive=True)

            if "error" in csv_only_report:
                tprint(f"   ❌ CSV-only analysis failed: {csv_only_report['error']}")
                return False

            csv_summary = csv_only_report.get("summary", {})
            assert csv_summary["total_files"] == 1, f"Expected 1 CSV file, got {csv_summary['total_files']}"

            tprint("   ✅ File pattern filtering passed")
            tprint(f"   - CSV files found: {csv_summary['total_files']}")

        except Exception as e:
            tprint(f"   ❌ File pattern filtering failed: {e}")
            return False

        # Test 4: Batch file analysis
        tprint("\n📊 Test 4: Batch file analysis")
        try:
            file_paths = [str(csv_file), str(parquet_file), str(json_file)]
            batch_report = orchestrator.analyze_file_batch(file_paths)

            batch_summary = batch_report.get("summary", {})
            assert batch_summary["total_files"] == 3, f"Expected 3 files, got {batch_summary['total_files']}"
            assert batch_summary["successful_analyses"] == 3, f"Expected 3 successful analyses, got {batch_summary['successful_analyses']}"

            tprint("   ✅ Batch file analysis passed")
            tprint(f"   - Total files: {batch_summary['total_files']}")
            tprint(f"   - Successful analyses: {batch_summary['successful_analyses']}")

        except Exception as e:
            tprint(f"   ❌ Batch file analysis failed: {e}")
            return False

        # Test 5: Save directory report
        tprint("\n📊 Test 5: Save directory report")
        try:
            output_file = orchestrator.save_report(directory_report, "test_directory_report.json")

            # Verify file was created
            assert Path(output_file).exists(), f"Report file was not created: {output_file}"

            tprint("   ✅ Directory report save passed")
            tprint(f"   - Report saved to: {output_file}")

            # Clean up test report
            os.remove(output_file)
            tprint("   🧹 Cleaned up test report")

        except Exception as e:
            tprint(f"   ❌ Directory report save failed: {e}")
            return False

    tprint("\n🎉 All directory analysis tests passed!")
    return True


def main():
    """Main test function."""
    tprint("🚀 TESTING DIRECTORY ANALYSIS FUNCTIONALITY")
    tprint("="*60)

    try:
        success = test_directory_analysis()

        if success:
            tprint("\n" + "="*60)
            tprint("✅ ALL TESTS PASSED SUCCESSFULLY!")
            tprint("="*60)
            tprint("\n💡 The directory analysis functionality is working correctly.")
            tprint("   You can now use the orchestrator to analyze both single files and directories.")
        else:
            tprint("\n" + "="*60)
            tprint("❌ SOME TESTS FAILED!")
            tprint("="*60)
            tprint("\n🔧 Please check the error messages above and fix any issues.")

    except Exception as e:
        tprint(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
