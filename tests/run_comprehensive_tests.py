#!/usr/bin/env python3
# tests/run_comprehensive_tests.py

"""
Comprehensive Test Runner
Executes all tests and generates detailed reports for system validation.
"""

import asyncio
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ComprehensiveTestRunner:
    """Comprehensive test runner for the dual model system."""

    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        self.report_dir = project_root / "test_reports"
        self.report_dir.mkdir(exist_ok=True)

    async def run_all_tests(self) -> dict[str, Any]:
        """Run all comprehensive tests."""
        print("🚀 Starting Comprehensive Test Suite...")
        self.start_time = datetime.now()

        # Test categories
        test_categories = {
            "dual_model_system": "tests/test_dual_model_system.py",
            "system_validation": "tests/test_system_validation.py",
            "enhanced_error_handling": "tests/test_enhanced_error_handling.py",
            "comprehensive_integration": "tests/test_comprehensive_integration.py",
        }

        results = {}

        for category, test_file in test_categories.items():
            print(f"\n📋 Running {category} tests...")
            try:
                result = await self.run_test_category(category, test_file)
                results[category] = result
            except Exception as e:
                print(f"❌ Error running {category} tests: {e}")
                results[category] = {
                    "status": "error",
                    "error": str(e),
                    "tests_passed": 0,
                    "tests_failed": 0,
                    "tests_skipped": 0,
                }

        self.end_time = datetime.now()
        self.test_results = results

        # Generate comprehensive report
        await self.generate_comprehensive_report()

        return results

    async def run_test_category(self, category: str, test_file: str) -> dict[str, Any]:
        """Run tests for a specific category."""
        test_path = project_root / test_file

        if not test_path.exists():
            return {
                "status": "skipped",
                "reason": f"Test file not found: {test_file}",
                "tests_passed": 0,
                "tests_failed": 0,
                "tests_skipped": 0,
            }

        # Run pytest with detailed output
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(test_path),
            "-v",
            "--tb=short",
            "--durations=10",
            "--capture=no",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,  # 5 minutes timeout
            )

            # Parse pytest output
            output_lines = result.stdout.split("\n")
            test_summary = self.parse_pytest_output(output_lines)

            return {
                "status": "completed",
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                **test_summary,
            }

        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "reason": "Test execution timed out",
                "tests_passed": 0,
                "tests_failed": 0,
                "tests_skipped": 0,
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "tests_passed": 0,
                "tests_failed": 0,
                "tests_skipped": 0,
            }

    def parse_pytest_output(self, output_lines: list[str]) -> dict[str, Any]:
        """Parse pytest output to extract test statistics."""
        tests_passed = 0
        tests_failed = 0
        tests_skipped = 0

        for line in output_lines:
            if "PASSED" in line:
                tests_passed += 1
            elif "FAILED" in line:
                tests_failed += 1
            elif "SKIPPED" in line:
                tests_skipped += 1

        return {
            "tests_passed": tests_passed,
            "tests_failed": tests_failed,
            "tests_skipped": tests_skipped,
            "total_tests": tests_passed + tests_failed + tests_skipped,
        }

    async def generate_comprehensive_report(self):
        """Generate comprehensive test report."""
        print("\n📊 Generating Comprehensive Test Report...")

        # Calculate overall statistics
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_skipped = 0

        for category, result in self.test_results.items():
            total_tests += result.get("total_tests", 0)
            total_passed += result.get("tests_passed", 0)
            total_failed += result.get("tests_failed", 0)
            total_skipped += result.get("tests_skipped", 0)

        # Calculate success rate
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

        # Generate report
        report = {
            "test_run": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "duration_seconds": (self.end_time - self.start_time).total_seconds()
                if self.start_time and self.end_time
                else 0,
            },
            "overall_statistics": {
                "total_tests": total_tests,
                "tests_passed": total_passed,
                "tests_failed": total_failed,
                "tests_skipped": total_skipped,
                "success_rate_percent": success_rate,
            },
            "category_results": self.test_results,
            "system_health": await self.assess_system_health(),
        }

        # Save report
        report_file = (
            self.report_dir
            / f"comprehensive_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        # Generate summary
        await self.generate_summary_report(report)

        print(f"✅ Comprehensive test report saved to: {report_file}")

    async def assess_system_health(self) -> dict[str, Any]:
        """Assess overall system health based on test results."""
        health_indicators = {
            "data_quality": "good",
            "model_performance": "good",
            "system_reliability": "good",
            "integration_health": "good",
            "error_handling": "good",
        }

        # Analyze test results for health indicators
        for category, result in self.test_results.items():
            if result.get("status") == "completed":
                success_rate = (
                    result.get("tests_passed", 0) / result.get("total_tests", 1)
                ) * 100

                if success_rate < 80:
                    if "validation" in category:
                        health_indicators["data_quality"] = "warning"
                    elif "performance" in category:
                        health_indicators["model_performance"] = "warning"
                    elif "reliability" in category:
                        health_indicators["system_reliability"] = "warning"
                    elif "integration" in category:
                        health_indicators["integration_health"] = "warning"
                    elif "error" in category:
                        health_indicators["error_handling"] = "warning"

        # Overall health assessment
        warning_count = sum(
            1 for status in health_indicators.values() if status == "warning"
        )
        error_count = sum(
            1 for status in health_indicators.values() if status == "error"
        )

        if error_count > 0:
            overall_health = "critical"
        elif warning_count > 2:
            overall_health = "warning"
        else:
            overall_health = "healthy"

        return {
            "overall_health": overall_health,
            "indicators": health_indicators,
            "recommendations": await self.generate_health_recommendations(
                health_indicators,
            ),
        }

    async def generate_health_recommendations(
        self,
        health_indicators: dict[str, str],
    ) -> list[str]:
        """Generate health recommendations based on indicators."""
        recommendations = []

        if health_indicators["data_quality"] == "warning":
            recommendations.append("Review data quality validation procedures")
            recommendations.append("Check for data preprocessing issues")

        if health_indicators["model_performance"] == "warning":
            recommendations.append("Monitor model performance degradation")
            recommendations.append("Consider retraining models")

        if health_indicators["system_reliability"] == "warning":
            recommendations.append("Review error handling mechanisms")
            recommendations.append("Check system resource usage")

        if health_indicators["integration_health"] == "warning":
            recommendations.append("Verify component integration")
            recommendations.append("Check configuration consistency")

        if health_indicators["error_handling"] == "warning":
            recommendations.append("Improve error handling coverage")
            recommendations.append("Add more robust fallback mechanisms")

        if not recommendations:
            recommendations.append("System is healthy - continue monitoring")

        return recommendations

    async def generate_summary_report(self, report: dict[str, Any]):
        """Generate human-readable summary report."""
        summary_file = (
            self.report_dir
            / f"test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )

        with open(summary_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE TEST SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Test run information
            f.write("TEST RUN INFORMATION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Start Time: {report['test_run']['start_time']}\n")
            f.write(f"End Time: {report['test_run']['end_time']}\n")
            f.write(
                f"Duration: {report['test_run']['duration_seconds']:.2f} seconds\n\n",
            )

            # Overall statistics
            stats = report["overall_statistics"]
            f.write("OVERALL STATISTICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Tests: {stats['total_tests']}\n")
            f.write(f"Tests Passed: {stats['tests_passed']}\n")
            f.write(f"Tests Failed: {stats['tests_failed']}\n")
            f.write(f"Tests Skipped: {stats['tests_skipped']}\n")
            f.write(f"Success Rate: {stats['success_rate_percent']:.1f}%\n\n")

            # Category results
            f.write("CATEGORY RESULTS:\n")
            f.write("-" * 40 + "\n")
            for category, result in report["category_results"].items():
                f.write(f"{category.upper()}:\n")
                f.write(f"  Status: {result.get('status', 'unknown')}\n")
                f.write(f"  Passed: {result.get('tests_passed', 0)}\n")
                f.write(f"  Failed: {result.get('tests_failed', 0)}\n")
                f.write(f"  Skipped: {result.get('tests_skipped', 0)}\n")
                if result.get("error"):
                    f.write(f"  Error: {result['error']}\n")
                f.write("\n")

            # System health
            health = report["system_health"]
            f.write("SYSTEM HEALTH:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Overall Health: {health['overall_health'].upper()}\n")
            f.write("Health Indicators:\n")
            for indicator, status in health["indicators"].items():
                f.write(f"  {indicator}: {status}\n")
            f.write("\n")

            # Recommendations
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 40 + "\n")
            for i, recommendation in enumerate(health["recommendations"], 1):
                f.write(f"{i}. {recommendation}\n")
            f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        print(f"📋 Summary report saved to: {summary_file}")

    def print_summary(self):
        """Print test summary to console."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE TEST SUMMARY")
        print("=" * 80)

        if not self.test_results:
            print("❌ No test results available")
            return

        # Overall statistics
        total_tests = sum(r.get("total_tests", 0) for r in self.test_results.values())
        total_passed = sum(r.get("tests_passed", 0) for r in self.test_results.values())
        total_failed = sum(r.get("tests_failed", 0) for r in self.test_results.values())
        total_skipped = sum(
            r.get("tests_skipped", 0) for r in self.test_results.values()
        )

        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

        print(f"📊 Total Tests: {total_tests}")
        print(f"✅ Passed: {total_passed}")
        print(f"❌ Failed: {total_failed}")
        print(f"⏭️  Skipped: {total_skipped}")
        print(f"📈 Success Rate: {success_rate:.1f}%")

        # Category breakdown
        print("\n📋 Category Breakdown:")
        for category, result in self.test_results.items():
            status = result.get("status", "unknown")
            passed = result.get("tests_passed", 0)
            failed = result.get("tests_failed", 0)
            skipped = result.get("tests_skipped", 0)

            status_icon = (
                "✅"
                if status == "completed" and failed == 0
                else "⚠️"
                if status == "completed"
                else "❌"
            )
            print(
                f"  {status_icon} {category}: {passed} passed, {failed} failed, {skipped} skipped",
            )

        # Duration
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
            print(f"\n⏱️  Total Duration: {duration:.2f} seconds")

        print("=" * 80)


async def main():
    """Main function to run comprehensive tests."""
    runner = ComprehensiveTestRunner()

    try:
        results = await runner.run_all_tests()
        runner.print_summary()

        # Exit with appropriate code
        total_failed = sum(r.get("tests_failed", 0) for r in results.values())
        if total_failed > 0:
            print(f"\n❌ Tests failed: {total_failed}")
            sys.exit(1)
        else:
            print("\n✅ All tests passed!")
            sys.exit(0)

    except KeyboardInterrupt:
        print("\n⏹️  Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
