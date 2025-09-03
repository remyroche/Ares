#!/usr/bin/env python3
"""
Integrated Validator - Combines function_validator.py with enhanced_validator.py

This module integrates both validators to provide comprehensive code quality checks:
- Function existence and import validation (from function_validator)
- Function argument validation (from enhanced_validator)
- Data access validation (from enhanced_validator)
- Async/await usage verification (from function_validator)
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from enhanced_validator import EnhancedValidator

# Import both validators
from function_validator import FunctionValidator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class IntegratedValidator:
    """Integrates multiple validators for comprehensive code quality checking."""

    def __init__(self, project_root: str = ".", exclude_patterns: list[str] | None = None):
        self.project_root = Path(project_root).resolve()
        self.exclude_patterns = exclude_patterns or ["__pycache__", "*.pyc", ".git", "venv", ".env"]

        # Initialize sub-validators
        self.function_validator = FunctionValidator(project_root, exclude_patterns)
        self.enhanced_validator = EnhancedValidator(project_root, exclude_patterns)

        # Combined statistics
        self.combined_stats = {}

    def validate_project(self) -> dict[str, Any]:
        """Run all validators and combine results."""
        logger.info(f"Starting integrated validation for project: {self.project_root}")

        # Run function validator
        logger.info("Running function validation...")
        function_report = self.function_validator.validate_project()

        # Run enhanced validator
        logger.info("Running enhanced validation...")
        enhanced_report = self.enhanced_validator.validate_project()

        # Combine reports
        return self._combine_reports(function_report, enhanced_report)


    def _combine_reports(self, function_report: dict[str, Any],
                        enhanced_report: dict[str, Any]) -> dict[str, Any]:
        """Combine reports from different validators."""

        # Combine summaries
        combined_summary = {
            "project_root": str(self.project_root),
            "validation_timestamp": datetime.now().isoformat(),
            "files_processed": max(
                function_report["summary"]["files_processed"],
                enhanced_report["summary"]["files_processed"],
            ),
            "function_validation": {
                "total_issues": function_report["summary"]["total_issues"],
                "undefined_functions": function_report["summary"]["undefined_functions"],
                "missing_await": function_report["summary"]["missing_await"],
                "parameter_mismatches": function_report["summary"]["parameter_mismatches"],
            },
            "enhanced_validation": {
                "total_issues": enhanced_report["summary"]["total_issues"],
                "argument_mismatches": enhanced_report["summary"]["argument_mismatches"],
                "unsafe_data_access": enhanced_report["summary"]["unsafe_data_access"],
                "missing_null_checks": enhanced_report["summary"]["missing_null_checks"],
                "type_inconsistencies": enhanced_report["summary"]["type_inconsistencies"],
            },
            "total_issues": (
                function_report["summary"]["total_issues"] +
                enhanced_report["summary"]["total_issues"]
            ),
        }

        # Combine issues
        all_issues = []

        # Add source field to differentiate issues
        for issue in function_report["issues"]:
            issue["source"] = "function_validator"
            all_issues.append(issue)

        for issue in enhanced_report["issues"]:
            issue["source"] = "enhanced_validator"
            all_issues.append(issue)

        # Sort issues by severity and file
        severity_order = {"error": 0, "warning": 1, "info": 2}
        all_issues.sort(key=lambda x: (
            severity_order.get(x["severity"], 3),
            x["file_path"],
            x["line_number"],
        ))

        return {
            "summary": combined_summary,
            "issues": all_issues,
            "function_analysis": function_report.get("function_analysis", {}),
            "data_access_summary": enhanced_report.get("data_access_summary", {}),
            "function_signatures": enhanced_report.get("function_signatures", {}),
        }

    def generate_report(self, output_dir: str | None = None) -> dict[str, str]:
        """Generate comprehensive reports in multiple formats."""
        if not output_dir:
            output_dir = "code_quality/reports"

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Run validation
        report = self.validate_project()

        # Save JSON report
        json_file = os.path.join(output_dir, f"integrated_validation_{timestamp}.json")
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # Generate human-readable report
        txt_file = os.path.join(output_dir, f"integrated_validation_summary_{timestamp}.txt")
        self._generate_summary_report(report, txt_file)

        # Generate markdown report
        md_file = os.path.join(output_dir, f"integrated_validation_report_{timestamp}.md")
        self._generate_markdown_report(report, md_file)

        return {
            "json": json_file,
            "summary": txt_file,
            "markdown": md_file,
        }

    def _generate_summary_report(self, report: dict[str, Any], output_file: str) -> None:
        """Generate a concise summary report."""
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("INTEGRATED CODE QUALITY VALIDATION SUMMARY\n")
            f.write("=" * 60 + "\n\n")

            summary = report["summary"]
            f.write(f"Project: {summary['project_root']}\n")
            f.write(f"Timestamp: {summary['validation_timestamp']}\n")
            f.write(f"Files processed: {summary['files_processed']}\n")
            f.write(f"Total issues found: {summary['total_issues']}\n\n")

            # Function validation summary
            fv = summary["function_validation"]
            f.write("FUNCTION VALIDATION\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total issues: {fv['total_issues']}\n")
            f.write(f"  - Undefined functions: {fv['undefined_functions']}\n")
            f.write(f"  - Missing await: {fv['missing_await']}\n")
            f.write(f"  - Parameter mismatches: {fv['parameter_mismatches']}\n\n")

            # Enhanced validation summary
            ev = summary["enhanced_validation"]
            f.write("ENHANCED VALIDATION\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total issues: {ev['total_issues']}\n")
            f.write(f"  - Argument mismatches: {ev['argument_mismatches']}\n")
            f.write(f"  - Unsafe data access: {ev['unsafe_data_access']}\n")
            f.write(f"  - Missing null checks: {ev['missing_null_checks']}\n")
            f.write(f"  - Type inconsistencies: {ev['type_inconsistencies']}\n\n")

            # Top issues by severity
            f.write("TOP ISSUES BY SEVERITY\n")
            f.write("-" * 30 + "\n")

            errors = [i for i in report["issues"] if i["severity"] == "error"]
            warnings = [i for i in report["issues"] if i["severity"] == "warning"]

            f.write(f"\nERRORS ({len(errors)}):\n")
            f.writelines(f"  {issue['file_path']}:{issue['line_number']} - {issue['message']}\n" for issue in errors[:5])
            if len(errors) > 5:
                f.write(f"  ... and {len(errors) - 5} more errors\n")

            f.write(f"\nWARNINGS ({len(warnings)}):\n")
            f.writelines(f"  {issue['file_path']}:{issue['line_number']} - {issue['message']}\n" for issue in warnings[:5])
            if len(warnings) > 5:
                f.write(f"  ... and {len(warnings) - 5} more warnings\n")

    def _generate_markdown_report(self, report: dict[str, Any], output_file: str) -> None:
        """Generate a detailed markdown report."""
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# Integrated Code Quality Validation Report\n\n")

            summary = report["summary"]
            f.write("## Summary\n\n")
            f.write(f"- **Project**: `{summary['project_root']}`\n")
            f.write(f"- **Timestamp**: {summary['validation_timestamp']}\n")
            f.write(f"- **Files Processed**: {summary['files_processed']}\n")
            f.write(f"- **Total Issues**: {summary['total_issues']}\n\n")

            # Validation summaries
            f.write("## Validation Results\n\n")

            f.write("### Function Validation\n\n")
            fv = summary["function_validation"]
            f.write("| Issue Type | Count |\n")
            f.write("|------------|-------|\n")
            f.write(f"| Undefined Functions | {fv['undefined_functions']} |\n")
            f.write(f"| Missing Await | {fv['missing_await']} |\n")
            f.write(f"| Parameter Mismatches | {fv['parameter_mismatches']} |\n")
            f.write(f"| **Total** | **{fv['total_issues']}** |\n\n")

            f.write("### Enhanced Validation\n\n")
            ev = summary["enhanced_validation"]
            f.write("| Issue Type | Count |\n")
            f.write("|------------|-------|\n")
            f.write(f"| Argument Mismatches | {ev['argument_mismatches']} |\n")
            f.write(f"| Unsafe Data Access | {ev['unsafe_data_access']} |\n")
            f.write(f"| Missing Null Checks | {ev['missing_null_checks']} |\n")
            f.write(f"| Type Inconsistencies | {ev['type_inconsistencies']} |\n")
            f.write(f"| **Total** | **{ev['total_issues']}** |\n\n")

            # Data access summary
            if "data_access_summary" in report:
                das = report["data_access_summary"]
                f.write("### Data Access Analysis\n\n")
                f.write(f"- Total Accesses: {das['total_accesses']}\n")
                f.write(f"- Safe Accesses: {das['safe_accesses']}\n")
                f.write(f"- Unsafe Accesses: {das['unsafe_accesses']}\n\n")

            # Detailed issues
            f.write("## Detailed Issues\n\n")

            # Group by file
            from collections import defaultdict
            issues_by_file = defaultdict(list)
            for issue in report["issues"]:
                issues_by_file[issue["file_path"]].append(issue)

            for file_path, file_issues in sorted(issues_by_file.items()):
                f.write(f"### {file_path}\n\n")

                # Sort by line number
                file_issues.sort(key=lambda x: x["line_number"])

                for issue in file_issues:
                    severity_icon = {
                        "error": "🔴",
                        "warning": "🟡",
                        "info": "ℹ️",
                    }.get(issue["severity"], "❓")

                    f.write(f"{severity_icon} **Line {issue['line_number']}**: {issue['message']}\n")

                    if issue.get("suggestion"):
                        f.write(f"  - *Suggestion*: {issue['suggestion']}\n")

                    if issue.get("code_snippet"):
                        f.write(f"  - *Code*: `{issue['code_snippet'].strip()}`\n")

                    f.write("\n")

                f.write("\n")


def main():
    """Main entry point for integrated validation."""
    import argparse

    parser = argparse.ArgumentParser(description="Integrated Code Quality Validator")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--output-dir", help="Output directory for reports")
    parser.add_argument("--exclude", nargs="*", help="Patterns to exclude")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize integrated validator
    validator = IntegratedValidator(args.project_root, args.exclude)

    # Generate reports
    report_files = validator.generate_report(args.output_dir)

    print("\nIntegrated validation completed!")
    print("\nGenerated reports:")
    print(f"  - JSON report: {report_files['json']}")
    print(f"  - Summary: {report_files['summary']}")
    print(f"  - Markdown report: {report_files['markdown']}")

    # Show brief summary
    with open(report_files["summary"]) as f:
        summary_lines = f.readlines()[:20]  # Show first 20 lines
        print("\n" + "".join(summary_lines))
        if len(f.readlines()) > 20:
            print("... (see full report for details)")


if __name__ == "__main__":
    main()
