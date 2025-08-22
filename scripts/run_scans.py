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

        for feature in self.features.values():
            status = "✓ Enabled" if feature.enabled else "✗ Disabled"
            print(f"{feature.name:<25} {status:<12} {feature.description}")

        print(f"\nTotal features: {len(self.features)}")
        print(
            f"Enabled features: {sum(1 for f in self.features.values() if f.enabled)}",
        )

    def run_scan(self, scan_type: str, verbose: bool = False) -> bool:
        """Run a specific scan type"""
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
        except subprocess.TimeoutExpired as e:
            logger.exception(
                f"✗ {feature.name} timed out after {feature.timeout} seconds",
            )
            return False
        except Exception as e:  # noqa: BLE001
            print(failed(f"✗ {feature.name} failed with error: {e}"))
            return False

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

    def run_all_scans(self, verbose: bool = False) -> dict[str, bool]:
        """Run all enabled scans and return results"""
        results = {}

        for key, feature in self.features.items():
            if feature.enabled:
                results[feature.name] = self.run_scan(key, verbose=verbose)
        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run code quality scans")
    parser.add_argument(
        "--scan",
        choices=[t.value for t in ScanType],
        default=ScanType.ALL.value,
        help="Type of scan to run",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available scans",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manager = ScanManager()

    if args.list:
        manager.list_features()
        return 0

    if args.scan == ScanType.ALL.value:
        results = manager.run_all_scans(verbose=args.verbose)
        success = all(results.values()) if results else True
        return 0 if success else 1

    ok = manager.run_scan(args.scan, verbose=args.verbose)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
