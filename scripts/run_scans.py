#!/usr/bin/env python3
"""
Run Scans - Code Quality and Analysis Tool

This script provides a comprehensive mapping of available code analysis features
and their corresponding functions. It can run various code quality checks, static analysis, and maintainability assessments.
"""

import argparse
import logging
import subprocess
import sys

from dataclasses import dataclass
from enum import Enum
from src.utils.warning_symbols import error, failed, warning

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ScanType(Enum):
    """Enumeration of available scan types"""

    FORMATTING = "formatting"
    LINTING = "linting"
    TYPE_CHECKING = "type_checking"
    COMPLEXITY = "complexity"
    MAINTAINABILITY = "maintainability"
    DEAD_CODE = "dead_code"
    CIRCULAR_IMPORTS = "circular_imports"
    SECURITY = "security"
    PERFORMANCE = "performance"
    DOCUMENTATION = "documentation"
    ALL = "all"


@dataclass
class ScanFeature:
    """Represents a scan feature with its configuration"""

    name: str
    description: str
    command: str
    enabled: bool = True
    timeout: int = 300  # 5 minutes default
    ignore_errors: bool = False


class ScanManager:
    """Manages the execution of various code analysis scans"""

    def __init__(self):
        self.features: dict[str, ScanFeature] = self._initialize_features()

    def _initialize_features(self) -> dict[str, ScanFeature]:
        """Initialize the feature mapping with all available scans"""
        return {
            ScanType.FORMATTING.value: ScanFeature(
                name="Code Formatting",
                description="Format code using ruff formatter",
                command="poetry run ruff format .",
                enabled=True
            ),
            ScanType.LINTING.value: ScanFeature(
                name="Code Linting",
                description="Check code style and potential issues using ruff",
                command="poetry run ruff check . --fix",
                enabled=True
            ),
            ScanType.TYPE_CHECKING.value: ScanFeature(
                name="Static Type Checking",
                description="Perform static type checking using mypy",
                command="poetry run mypy --ignore-missing-imports --package src",
                enabled=True, ignore_errors=True,
            ),
            ScanType.COMPLEXITY.value: ScanFeature(
                name="Cyclomatic Complexity Analysis",
                description="Analyze code complexity using radon",
                command="poetry run radon cc src/ -s -nc",
                enabled=True, ignore_errors=True,
            ),
            ScanType.MAINTAINABILITY.value: ScanFeature(
                name="Maintainability Index",
                description="Calculate maintainability index using radon",
                command="poetry run radon mi src/ -s -nc",
                enabled=True, ignore_errors=True,
            ),
            ScanType.DEAD_CODE.value: ScanFeature(
                name="Dead Code Detection",
                description="Find unused code using vulture",
                command="poetry run vulture src/",
                enabled=True, ignore_errors=True,
            ),
            ScanType.CIRCULAR_IMPORTS.value: ScanFeature(
                name="Circular Import Detection",
                description="Detect circular imports using pylint",
                command="poetry run pylint --disable=all --enable=cyclic-import src/",
                enabled=True, ignore_errors=True,
            ),
            ScanType.SECURITY.value: ScanFeature(
                name="Security Analysis",
                description="Check for security vulnerabilities using bandit",
                command="poetry run bandit -r src/ -f json",
                enabled=True, ignore_errors=True,
            ),
            ScanType.PERFORMANCE.value: ScanFeature(
                name="Performance Analysis",
                description="Analyze performance issues using pyflakes",
                command="poetry run pyflakes src/",
                enabled=True, ignore_errors=True,
            ),
            ScanType.DOCUMENTATION.value: ScanFeature(
                name="Documentation Check",
                description="Check documentation coverage using pydocstyle",
                command="poetry run pydocstyle src/",
                enabled=True, ignore_errors=True,
            ),
        }

    def run_scan(self, scan_type: str, verbose: bool = False) -> bool:
        """Run a specific scan"""
        if scan_type not in self.features:
            print(error(f"Unknown scan type: {scan_type}"))
            return False

        feature = self.features[scan_type]

        if not feature.enabled:
            print(warning(f"Feature '{feature.name}' is disabled"))
            return False

        logger.info(f"Running {feature.name}...")

        if verbose:
            print(f"\n--- {feature.name} ---")
            print(f"Command: {feature.command}")
            print(f"Description: {feature.description}")
            print("-" * 50)

        try:
            result = subprocess.run(
                feature.command.split(),
                capture_output=True,
                text=True,
                timeout=feature.timeout,
                check=False,
            )

            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)

            if result.returncode == 0:
                logger.info(f"✓ {feature.name} completed successfully")
                return True
            if feature.ignore_errors:
                logger.warning(
                    f"⚠ {feature.name} completed with warnings (ignored)",
                )
                return True
            logger.error(
                f"✗ {feature.name} failed with return code {result.returncode}",
            )
            return False

        except subprocess.TimeoutExpired:
            logger.exception(
                f"✗ {feature.name} timed out after {feature.timeout} seconds",
            )
            return False
        except Exception as e:
            print(failed(f"✗ {feature.name} failed with error: {e}"))
            return False

    def run_all_scans(self, verbose: bool = False) -> dict[str, bool]:
        """Run all enabled scans and return results"""
        results = {}

        logger.info("Starting comprehensive code analysis...")

        for scan_type, feature in self.features.items():
            if feature.enabled:
                results[scan_type] = self.run_scan(scan_type, verbose)
            else:
                results[scan_type] = False

        return results

    def enable_feature(self, scan_type: str) -> bool:
        """Enable a specific feature"""
        if scan_type in self.features:
            self.features[scan_type].enabled = True
            logger.info(f"Enabled feature: {self.features[scan_type].name}")
            return True
        print(error(f"Unknown feature: {scan_type}"))
        return False

    def disable_feature(self, scan_type: str) -> bool:
        """Disable a specific feature"""
        if scan_type in self.features:
            self.features[scan_type].enabled = False
            logger.info(f"Disabled feature: {self.features[scan_type].name}")
            return True
        print(error(f"Unknown feature: {scan_type}"))
        return False

    def get_feature_info(self, scan_type: str) -> ScanFeature | None:
        """Get detailed information about a specific feature"""
        return self.features.get(scan_type)

    def list_features(self) -> None:
        """List all available features and their status"""
        print("\nAvailable Code Analysis Features:")
        print("=" * 50)
        for scan_type, feature in self.features.items():
            status = "✓ Enabled" if feature.enabled else "✗ Disabled"
            print(f"{scan_type:20} {status}")
            print(f"{'':20} {feature.description}")
            print()

    def get_summary(self, results: dict[str, bool]) -> str:
        """Generate a summary of scan results"""
        total = len(results)
        passed = sum(1 for result in results.values() if result)
        failed = total - passed

        summary = f"\nScan Summary:\n"
        summary += f"Total scans: {total}\n"
        summary += f"Passed: {passed}\n"
        summary += f"Failed: {failed}\n"
        summary += f"Success rate: {(passed/total)*100:.1f}%\n"

        if failed > 0:
            summary += f"\nFailed scans:\n"
            for scan_type, result in results.items():
                if not result:
                    summary += f"  - {scan_type}\n"

        return summary


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run comprehensive code analysis scans"
    )
    parser.add_argument(
        "scan_type",
        nargs="?",
        choices=[t.value for t in ScanType],
        help="Type of scan to run (default: all)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available features",
    )
    parser.add_argument(
        "--enable", "-e",
        help="Enable a specific feature",
    )
    parser.add_argument(
        "--disable", "-d",
        help="Disable a specific feature",
    )

    args = parser.parse_args()

    scan_manager = ScanManager()

    # Handle list command
    if args.list:
        scan_manager.list_features()
        return

    # Handle enable/disable commands
    if args.enable:
        if scan_manager.enable_feature(args.enable):
            print(f"✓ Enabled feature: {args.enable}")
        return

    if args.disable:
        if scan_manager.disable_feature(args.disable):
            print(f"✓ Disabled feature: {args.disable}")
        return

    # Run scans
    if args.scan_type:
        if args.scan_type == ScanType.ALL.value:
            results = scan_manager.run_all_scans(args.verbose)
        else:
            success = scan_manager.run_scan(args.scan_type, args.verbose)
            results = {args.scan_type: success}
    else:
        # Default to running all scans
        results = scan_manager.run_all_scans(args.verbose)

    # Print summary
    print(scan_manager.get_summary(results))

    # Exit with appropriate code
    if all(results.values()):
        print("✓ All scans passed!")
        sys.exit(0)
    else:
        print("✗ Some scans failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
