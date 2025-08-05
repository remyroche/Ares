#!/usr/bin/env python3
"""
Run Scans - Code Quality and Analysis Tool

This script provides a comprehensive mapping of available code analysis features
and their corresponding functions. It can run various code quality checks,
static analysis, and maintainability assessments.
"""

import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum

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
                enabled=True,
            ),
            ScanType.LINTING.value: ScanFeature(
                name="Code Linting",
                description="Check code style and potential issues using ruff",
                command="poetry run ruff check . --fix",
                enabled=True,
            ),
            ScanType.TYPE_CHECKING.value: ScanFeature(
                name="Static Type Checking",
                description="Perform static type checking using mypy",
                command="poetry run mypy --ignore-missing-imports --package src",
                enabled=True,
                ignore_errors=True,
            ),
            ScanType.COMPLEXITY.value: ScanFeature(
                name="Cyclomatic Complexity Analysis",
                description="Analyze code complexity using radon",
                command="poetry run radon cc src/ -s -nc",
                enabled=True,
                ignore_errors=True,
            ),
            ScanType.MAINTAINABILITY.value: ScanFeature(
                name="Maintainability Index",
                description="Calculate maintainability index using radon",
                command="poetry run radon mi src/ -s -nc",
                enabled=True,
                ignore_errors=True,
            ),
            ScanType.DEAD_CODE.value: ScanFeature(
                name="Dead Code Detection",
                description="Find unused code using vulture",
                command="poetry run vulture src/",
                enabled=True,
                ignore_errors=True,
            ),
            ScanType.CIRCULAR_IMPORTS.value: ScanFeature(
                name="Circular Import Detection",
                description="Detect circular imports using pylint",
                command="poetry run pylint --disable=all --enable=cyclic-import src/",
                enabled=True,
                ignore_errors=True,
            ),
            ScanType.SECURITY.value: ScanFeature(
                name="Security Analysis",
                description="Run security checks using bandit",
                command="poetry run bandit -r src/",
                enabled=False,  # Disabled by default, requires bandit
                ignore_errors=True,
            ),
            ScanType.PERFORMANCE.value: ScanFeature(
                name="Performance Analysis",
                description="Analyze performance using py-spy",
                command="poetry run py-spy top -- python -c 'import time; time.sleep(1)'",
                enabled=False,  # Disabled by default, requires py-spy
                ignore_errors=True,
            ),
            ScanType.DOCUMENTATION.value: ScanFeature(
                name="Documentation Coverage",
                description="Check documentation coverage using pydocstyle",
                command="poetry run pydocstyle src/",
                enabled=False,  # Disabled by default, requires pydocstyle
                ignore_errors=True,
            ),
        }

    def list_features(self) -> None:
        """List all available features with their status"""
        print("\n=== Available Scan Features ===")
        print(f"{'Feature':<25} {'Status':<12} {'Description'}")
        print("-" * 70)

        for scan_type, feature in self.features.items():
            status = "✓ Enabled" if feature.enabled else "✗ Disabled"
            print(f"{feature.name:<25} {status:<12} {feature.description}")

        print(f"\nTotal features: {len(self.features)}")
        print(
            f"Enabled features: {sum(1 for f in self.features.values() if f.enabled)}",
        )

    def run_scan(self, scan_type: str, verbose: bool = False) -> bool:
        """Run a specific scan type"""
        if scan_type not in self.features:
            logger.error(f"Unknown scan type: {scan_type}")
            return False

        feature = self.features[scan_type]
        if not feature.enabled:
            logger.warning(f"Feature '{feature.name}' is disabled")
            return False

        logger.info(f"Running {feature.name}...")

        try:
            if verbose:
                print(f"\n--- {feature.name} ---")
                print(f"Command: {feature.command}")
                print(f"Description: {feature.description}")
                print("-" * 50)

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
            logger.error(f"✗ {feature.name} timed out after {feature.timeout} seconds")
            return False
        except Exception as e:
            logger.error(f"✗ {feature.name} failed with error: {e}")
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
        logger.error(f"Unknown feature: {scan_type}")
        return False

    def disable_feature(self, scan_type: str) -> bool:
        """Disable a specific feature"""
        if scan_type in self.features:
            self.features[scan_type].enabled = False
            logger.info(f"Disabled feature: {self.features[scan_type].name}")
            return True
        logger.error(f"Unknown feature: {scan_type}")
        return False

    def get_feature_info(self, scan_type: str) -> ScanFeature | None:
        """Get detailed information about a specific feature"""
        return self.features.get(scan_type)

    def print_summary(self, results: dict[str, bool]) -> None:
        """Print a summary of scan results"""
        print("\n=== Scan Results Summary ===")

        successful = sum(1 for success in results.values() if success)
        total = len(results)

        print(f"Total scans: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {total - successful}")

        if results:
            print("\nDetailed Results:")
            for scan_type, success in results.items():
                feature = self.features[scan_type]
                status = "✓ PASS" if success else "✗ FAIL"
                print(f"  {feature.name:<25} {status}")


def main():
    """Main entry point for the scan runner"""
    parser = argparse.ArgumentParser(
        description="Run comprehensive code analysis and quality scans",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_scans.py --list                    # List all available features
  python run_scans.py --all                     # Run all enabled scans
  python run_scans.py --scan linting            # Run only linting
  python run_scans.py --scan formatting --verbose  # Run formatting with verbose output
  python run_scans.py --enable security         # Enable security scanning
  python run_scans.py --info type_checking      # Get info about type checking
        """,
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available scan features",
    )

    parser.add_argument("--all", action="store_true", help="Run all enabled scans")

    parser.add_argument("--scan", type=str, help="Run a specific scan type")

    parser.add_argument("--enable", type=str, help="Enable a specific feature")

    parser.add_argument("--disable", type=str, help="Disable a specific feature")

    parser.add_argument(
        "--info",
        type=str,
        help="Get detailed information about a specific feature",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Initialize scan manager
    manager = ScanManager()

    # Handle different command modes
    if args.list:
        manager.list_features()
        return

    if args.enable:
        manager.enable_feature(args.enable)
        return

    if args.disable:
        manager.disable_feature(args.disable)
        return

    if args.info:
        feature = manager.get_feature_info(args.info)
        if feature:
            print(f"\n=== Feature Information: {feature.name} ===")
            print(f"Description: {feature.description}")
            print(f"Command: {feature.command}")
            print(f"Enabled: {feature.enabled}")
            print(f"Timeout: {feature.timeout} seconds")
            print(f"Ignore Errors: {feature.ignore_errors}")
        else:
            logger.error(f"Unknown feature: {args.info}")
        return

    # Run scans
    if args.all:
        results = manager.run_all_scans(verbose=args.verbose)
        manager.print_summary(results)
    elif args.scan:
        success = manager.run_scan(args.scan, verbose=args.verbose)
        if success:
            logger.info(f"Scan '{args.scan}' completed successfully")
        else:
            logger.error(f"Scan '{args.scan}' failed")
            sys.exit(1)
    else:
        # Default behavior: run all scans
        results = manager.run_all_scans(verbose=args.verbose)
        manager.print_summary(results)


if __name__ == "__main__":
    main()
