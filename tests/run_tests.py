#!/usr/bin/env python3
# tests/run_tests.py

"""
Test Runner for Dual Model System
Executes comprehensive tests and generates reports.
"""

import asyncio
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def run_comprehensive_tests():
    """Run comprehensive test suite."""
    print("🚀 Running Comprehensive Test Suite...")

    # Test files to run
    test_files = [
        "tests/test_dual_model_system.py",
        "tests/test_system_validation.py",
        "tests/test_enhanced_error_handling.py",
        "tests/test_comprehensive_integration.py",
    ]

    results = {}
    total_passed = 0
    total_failed = 0
    total_skipped = 0

    for test_file in test_files:
        test_path = project_root / test_file

        if not test_path.exists():
            print(f"⚠️  Test file not found: {test_file}")
            continue

        print(f"\n📋 Running {test_file}...")

        try:
            # Run pytest
            cmd = [sys.executable, "-m", "pytest", str(test_path), "-v", "--tb=short"]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )

            # Parse results
            output_lines = result.stdout.split("\n")
            passed = sum(1 for line in output_lines if "PASSED" in line)
            failed = sum(1 for line in output_lines if "FAILED" in line)
            skipped = sum(1 for line in output_lines if "SKIPPED" in line)

            results[test_file] = {
                "status": "completed" if result.returncode == 0 else "failed",
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "total": passed + failed + skipped,
            }

            total_passed += passed
            total_failed += failed
            total_skipped += skipped

            status_icon = "✅" if result.returncode == 0 else "❌"
            print(
                f"{status_icon} {test_file}: {passed} passed, {failed} failed, {skipped} skipped",
            )

        except subprocess.TimeoutExpired:
            print(f"⏰ {test_file}: Timeout")
            results[test_file] = {"status": "timeout"}
        except Exception as e:
            print(f"💥 {test_file}: Error - {e}")
            results[test_file] = {"status": "error", "error": str(e)}

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total_passed + total_failed + total_skipped}")
    print(f"✅ Passed: {total_passed}")
    print(f"❌ Failed: {total_failed}")
    print(f"⏭️  Skipped: {total_skipped}")

    if total_passed + total_failed > 0:
        success_rate = (total_passed / (total_passed + total_failed)) * 100
        print(f"📈 Success Rate: {success_rate:.1f}%")

    # Save results
    report_file = (
        project_root
        / "test_reports"
        / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    report_file.parent.mkdir(exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_passed": total_passed,
            "total_failed": total_failed,
            "total_skipped": total_skipped,
            "success_rate": success_rate if total_passed + total_failed > 0 else 0,
        },
        "results": results,
    }

    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n📊 Report saved to: {report_file}")

    # Exit with appropriate code
    if total_failed > 0:
        print(f"\n❌ {total_failed} tests failed")
        return False
    print("\n✅ All tests passed!")
    return True


async def run_validation_tests():
    """Run validation tests specifically."""
    print("🔍 Running Validation Tests...")

    test_file = "tests/test_system_validation.py"
    test_path = project_root / test_file

    if not test_path.exists():
        print(f"❌ Validation test file not found: {test_file}")
        return False

    try:
        cmd = [sys.executable, "-m", "pytest", str(test_path), "-v", "--tb=short"]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )

        # Parse results
        output_lines = result.stdout.split("\n")
        passed = sum(1 for line in output_lines if "PASSED" in line)
        failed = sum(1 for line in output_lines if "FAILED" in line)
        skipped = sum(1 for line in output_lines if "SKIPPED" in line)

        print("\n📊 Validation Results:")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⏭️  Skipped: {skipped}")

        return result.returncode == 0

    except Exception as e:
        print(f"💥 Validation test error: {e}")
        return False


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Run comprehensive tests")
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Run only validation tests",
    )
    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Run comprehensive test suite",
    )

    args = parser.parse_args()

    if args.validation_only:
        success = asyncio.run(run_validation_tests())
    elif args.comprehensive:
        success = asyncio.run(run_comprehensive_tests())
    else:
        # Run both by default
        print("Running both comprehensive and validation tests...")
        success = asyncio.run(run_comprehensive_tests())
        if success:
            success = asyncio.run(run_validation_tests())

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
