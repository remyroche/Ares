#!/usr/bin/env python3
"""
Attribute Checker Pipeline

Specialized pipeline for detecting missing methods and attributes in Python classes.
Uses enhanced attribute access analysis to identify potential issues with reduced false positives.

Features:
- Detects missing method calls and attribute accesses
- Filters out known external attributes and parent class methods
- Provides severity classification (errors vs warnings)
- Generates detailed reports with context information

Usage:
    python pipelines/attribute_checker_pipeline.py --target-file src/file.py --class-name ClassName
    python pipelines/attribute_checker_pipeline.py --target-dir src/ --exclude-common
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
grandparent_dir = parent_dir.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(grandparent_dir))

# Import attribute checker
from attribute_checker import AttributeChecker, check_file

# Import core components
from core.config import get_default_config

# Import standardized base pipeline
from base_pipeline import BasePipeline
import logging


class AttributeCheckerPipeline(BasePipeline):
    """Specialized pipeline for attribute access analysis with standardized initialization."""

    def __init__(self, project_root: str = None, enable_plugins: bool = True):
        # Use standardized initialization from base class
        super().__init__(project_root=project_root, enable_plugins=enable_plugins,
                        pipeline_name="attribute_checker")

        # Setup pipeline-specific paths
        self.setup_pipeline_paths()

        # Initialize configuration
        self.config = get_default_config()

        # Initialize analysis results
        self.analysis_results = {}

        # Initialize plugin system with standardized registration
        if self.enable_plugins:
            self._register_attribute_checker_plugins()

    def _register_attribute_checker_plugins(self):
        """Register attribute checker related plugins using standardized batch registration."""
        try:
            # Import attribute checker and related plugins
            from plugins.attribute_checker_analyzer import AttributeCheckerAnalyzer
            from plugins.flake8_analyzer import Flake8Analyzer

            # Use standardized batch registration
            plugin_classes = [AttributeCheckerAnalyzer, Flake8Analyzer]
            self.register_plugins_batch(plugin_classes)

        except ImportError as e:
            self.logger.warning(f"Could not import attribute checker plugins: {e}")

    def analyze_file(self, file_path: str, class_name: str) -> Dict[str, Any]:
        """Analyze a single file for attribute access issues."""
        print(f"\n🔍 Analyzing {file_path} for class '{class_name}'...")

        try:
            result = check_file(file_path, class_name)

            if result['status'] == 'error':
                print(f"❌ Analysis failed: {result['error']}")
                return result

            # Store results
            self.analysis_results[file_path] = result

            # Print summary
            total_accesses = result['total_accesses']
            missing_count = result['missing_count']
            error_count = result['error_count']
            warning_count = result['warning_count']

            print(f"📊 Found {total_accesses} attribute accesses")
            print(f"⚠️  {warning_count} warnings, 🚨 {error_count} errors")

            if error_count > 0:
                print("🚨 CRITICAL ISSUES FOUND - Manual review required")
            elif warning_count > 0:
                print("⚠️  Potential issues found - Review recommended")

            return result

        except Exception as e:
            error_result = {
                'status': 'error',
                'error': str(e),
                'file_path': file_path,
                'class_name': class_name
            }
            print(f"❌ Analysis failed: {e}")
            return error_result

    def analyze_directory(self, directory_path: str, exclude_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze all Python files in a directory."""
        print(f"\n🔍 Analyzing directory: {directory_path}")

        if exclude_patterns is None:
            exclude_patterns = ['__pycache__', '.git', 'node_modules', '*.pyc', '.pytest_cache']

        directory = Path(directory_path)
        if not directory.exists():
            return {'status': 'error', 'error': f'Directory not found: {directory_path}'}

        results = {
            'status': 'success',
            'directory': str(directory),
            'files_analyzed': 0,
            'total_errors': 0,
            'total_warnings': 0,
            'files_with_issues': 0,
            'file_results': {}
        }

        # Find Python files
        python_files = []
        for py_file in directory.rglob("*.py"):
            should_exclude = False
            for pattern in exclude_patterns:
                if pattern in str(py_file):
                    should_exclude = True
                    break

            if not should_exclude:
                python_files.append(py_file)

        print(f"📁 Found {len(python_files)} Python files to analyze")

        # Analyze each file
        for py_file in python_files:
            try:
                # Extract classes from the file
                classes = self._extract_classes_from_file(str(py_file))

                if classes:
                    results['files_analyzed'] += 1

                    for class_name in classes:
                        file_result = self.analyze_file(str(py_file), class_name)

                        if file_result['status'] == 'success':
                            results['file_results'][str(py_file)] = file_result
                            results['total_errors'] += file_result.get('error_count', 0)
                            results['total_warnings'] += file_result.get('warning_count', 0)

                            if file_result.get('missing_count', 0) > 0:
                                results['files_with_issues'] += 1
                        else:
                            results['file_results'][str(py_file)] = file_result

            except Exception as e:
                print(f"❌ Failed to analyze {py_file}: {e}")
                results['file_results'][str(py_file)] = {'status': 'error', 'error': str(e)}

        return results

    def _extract_classes_from_file(self, file_path: str) -> List[str]:
        """Extract class names from a Python file."""
        try:
            import ast

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)
            classes = []

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)

            return classes

        except Exception as e:
            print(f"Warning: Could not extract classes from {file_path}: {e}")
            return []

    def generate_report(self, output_file: Optional[str] = None) -> Path:
        """Generate a comprehensive analysis report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        report_data = {
            'timestamp': timestamp,
            'pipeline_type': 'AttributeCheckerPipeline',
            'project_root': str(self.project_root),
            'analysis_results': self.analysis_results,
            'summary': {
                'total_files_analyzed': len(self.analysis_results),
                'total_errors': sum(r.get('error_count', 0) for r in self.analysis_results.values() if isinstance(r, dict)),
                'total_warnings': sum(r.get('warning_count', 0) for r in self.analysis_results.values() if isinstance(r, dict)),
                'files_with_issues': sum(1 for r in self.analysis_results.values() if isinstance(r, dict) and r.get('missing_count', 0) > 0)
            }
        }

        # Save report
        if output_file is None:
            output_file = f"attribute_checker_report_{timestamp}.json"

        report_path = self.reports_dir / output_file
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        print(f"📄 Report saved to: {report_path}")
        return report_path

    def run_analysis(self, target: str, class_name: Optional[str] = None,
                    exclude_common: bool = True) -> Dict[str, Any]:
        """Run the complete attribute checker analysis."""
        self._print_pipeline_header("ATTRIBUTE CHECKER ANALYSIS")

        try:
            target_path = Path(target)

            if target_path.is_file():
                # Analyze single file
                if not class_name:
                    print("❌ Class name required for single file analysis")
                    return {'status': 'error', 'error': 'Class name required for single file analysis'}

                result = self.analyze_file(target, class_name)

            elif target_path.is_dir():
                # Analyze directory
                result = self.analyze_directory(target)

            else:
                return {'status': 'error', 'error': f'Target not found: {target}'}

            # Generate report
            self.generate_report()

            # Print final summary
            if result['status'] == 'success':
                self._print_analysis_summary(result)

            return result

        except Exception as e:
            error_result = {'status': 'error', 'error': str(e)}
            print(f"❌ Analysis failed: {e}")
            return error_result

    def _print_analysis_summary(self, result: Dict[str, Any]) -> None:
        """Print a formatted analysis summary."""
        print(f"\n{'='*80}")
        print("ATTRIBUTE CHECKER ANALYSIS SUMMARY")
        print(f"{'='*80}")

        if 'file_results' in result:
            # Directory analysis summary
            files_analyzed = result.get('files_analyzed', 0)
            total_errors = result.get('total_errors', 0)
            total_warnings = result.get('total_warnings', 0)
            files_with_issues = result.get('files_with_issues', 0)

            print(f"Files analyzed: {files_analyzed}")
            print(f"Files with issues: {files_with_issues}")
            print(f"Total errors: {total_errors}")
            print(f"Total warnings: {total_warnings}")

            if total_errors > 0:
                print("🚨 CRITICAL: Errors found that need immediate attention")
            elif total_warnings > 0:
                print("⚠️  WARNING: Potential issues found - review recommended")
            else:
                print("✅ SUCCESS: No attribute access issues found")

        elif 'missing_count' in result:
            # Single file analysis summary
            missing_count = result.get('missing_count', 0)
            error_count = result.get('error_count', 0)
            warning_count = result.get('warning_count', 0)

            print(f"Attribute accesses analyzed: {result.get('total_accesses', 0)}")
            print(f"Potential issues: {missing_count}")
            print(f"Errors: {error_count}")
            print(f"Warnings: {warning_count}")

            if error_count > 0:
                print("🚨 CRITICAL: Errors found that need immediate attention")
            elif warning_count > 0:
                print("⚠️  WARNING: Potential issues found - review recommended")
            else:
                print("✅ SUCCESS: No attribute access issues found")

    def health_check(self) -> Dict[str, Any]:
        """Return pipeline health status with attribute checker specific information."""
        base_health = super().health_check()

        # Add attribute checker specific health information
        base_health.update({
            'attribute_checker_available': True,
            'analysis_results_count': len(self.analysis_results),
            'last_analysis_timestamp': getattr(self, 'timestamp', None)
        })

        return base_health


def main():
    """Command-line interface for the Attribute Checker Pipeline."""
    parser = argparse.ArgumentParser(
        description="Enhanced Attribute Checker Pipeline - Detect missing methods and attributes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single file
  python attribute_checker_pipeline.py --target-file src/my_file.py --class-name MyClass

  # Analyze directory
  python attribute_checker_pipeline.py --target-dir src/

  # Analyze with custom exclusions
  python attribute_checker_pipeline.py --target-dir src/ --exclude-patterns "__pycache__,test_*"
        """
    )

    parser.add_argument(
        '--target-file',
        help='Path to single Python file to analyze'
    )

    parser.add_argument(
        '--target-dir',
        help='Path to directory containing Python files to analyze'
    )

    parser.add_argument(
        '--class-name',
        help='Class name to analyze (required for single file analysis)'
    )

    parser.add_argument(
        '--exclude-patterns',
        default='__pycache__,*.pyc,.git,node_modules',
        help='Comma-separated patterns to exclude from analysis'
    )

    parser.add_argument(
        '--exclude-common',
        action='store_true',
        default=True,
        help='Exclude common external attributes (recommended)'
    )

    parser.add_argument(
        '--output-file',
        help='Output file name for the report'
    )

    parser.add_argument(
        '--project-root',
        help='Project root directory (defaults to current directory)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.target_file and not args.target_dir:
        parser.error("Either --target-file or --target-dir must be specified")

    if args.target_file and not args.class_name:
        parser.error("--class-name is required when analyzing a single file")

    # Determine target
    if args.target_file:
        target = args.target_file
    else:
        target = args.target_dir

    # Parse exclude patterns
    exclude_patterns = [p.strip() for p in args.exclude_patterns.split(',')]

    try:
        # Initialize pipeline
        pipeline = AttributeCheckerPipeline(
            project_root=args.project_root,
            enable_plugins=True
        )

        # Run analysis
        result = pipeline.run_analysis(
            target=target,
            class_name=args.class_name,
            exclude_common=args.exclude_common
        )

        # Exit with appropriate code
        if result.get('status') == 'error':
            sys.exit(1)
        elif isinstance(result, dict) and result.get('error_count', 0) > 0:
            sys.exit(1)  # Exit with error if critical issues found
        else:
            sys.exit(0)

    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
