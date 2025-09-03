#!/usr/bin/env python3
"""
Merge Conflict Detector - Specialized tool for detecting issues after merges.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .analyzers.import_analyzer import ImportAnalyzer
from .analyzers.linter_analyzer import LinterAnalyzer
from .analyzers.signature_analyzer import SignatureAnalyzer
from .analyzers.syntax_validator import SyntaxValidator
from .core.config import CodeQualityConfig, get_default_config


class MergeConflictDetector:
    """
    Specialized tool for detecting merge conflicts and compatibility issues.

    This tool focuses on the most common issues that arise after merges:
    1. Import conflicts and circular dependencies
    2. Function signature changes and compatibility issues
    3. Syntax errors and compilation issues
    4. Linter violations
    """

    def __init__(self, config: CodeQualityConfig | None = None):
        self.config = config or get_default_config()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}

    def detect_conflicts(self, target: str, output_dir: str | None = None) -> dict[str, Any]:
        """
        Detect merge conflicts in the specified target.

        Args:
            target: File, directory, or comma-separated list of files
            output_dir: Optional output directory for reports

        Returns:
            Conflict detection results
        """
        print("="*70)
        print("MERGE CONFLICT DETECTOR")
        print("="*70)
        print(f"Target: {target}")
        print(f"Timestamp: {self.timestamp}")

        # Normalize target to list of files
        if isinstance(target, str):
            if "," in target:
                # Comma-separated list
                target_files = [f.strip() for f in target.split(",")]
            elif os.path.isfile(target):
                target_files = [target]
            else:
                # Directory - find all Python files
                target_files = self._find_python_files(target)
        else:
            target_files = target

        print(f"Files to analyze: {len(target_files)}")

        # Initialize results
        self.results = {
            "merge_conflict_detection": {
                "target": target,
                "total_files": len(target_files),
                "timestamp": self.timestamp,
                "start_time": datetime.now().isoformat(),
            },
            "conflicts": {},
            "summary": {},
        }

        try:
            # Step 1: Import Conflict Detection
            print("\n" + "-"*50)
            print("STEP 1: DETECTING IMPORT CONFLICTS")
            print("-"*50)
            import_conflicts = self._detect_import_conflicts(target_files)
            self.results["conflicts"]["import_conflicts"] = import_conflicts

            # Step 2: Function Signature Compatibility
            print("\n" + "-"*50)
            print("STEP 2: DETECTING FUNCTION SIGNATURE CHANGES")
            print("-"*50)
            signature_conflicts = self._detect_signature_conflicts(target_files)
            self.results["conflicts"]["signature_conflicts"] = signature_conflicts

            # Step 3: Syntax and Compilation Issues
            print("\n" + "-"*50)
            print("STEP 3: DETECTING SYNTAX AND COMPILATION ISSUES")
            print("-"*50)
            syntax_conflicts = self._detect_syntax_conflicts(target_files)
            self.results["conflicts"]["syntax_conflicts"] = syntax_conflicts

            # Step 4: Linter Violations
            print("\n" + "-"*50)
            print("STEP 4: DETECTING LINTER VIOLATIONS")
            print("-"*50)
            linter_conflicts = self._detect_linter_conflicts(target_files)
            self.results["conflicts"]["linter_conflicts"] = linter_conflicts

            # Step 5: Generate Summary
            print("\n" + "-"*50)
            print("STEP 5: GENERATING CONFLICT SUMMARY")
            print("-"*50)
            summary = self._generate_conflict_summary()
            self.results["summary"] = summary

            # Step 6: Save Reports
            if output_dir:
                self._save_conflict_reports(output_dir)

            # Step 7: Print Final Summary
            self._print_conflict_summary()

        except Exception as e:
            print(f"\nERROR: Conflict detection failed: {e}")
            self.results["error"] = str(e)
            raise

        finally:
            self.results["merge_conflict_detection"]["end_time"] = datetime.now().isoformat()

        return self.results

    def _find_python_files(self, directory: str) -> list[str]:
        """Find all Python files in a directory."""
        python_files = []
        for root, dirs, files in os.walk(directory):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.config.analysis.exclude_patterns]

            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))

        return python_files

    def _detect_import_conflicts(self, files: list[str]) -> dict[str, Any]:
        """Detect import conflicts and circular dependencies."""
        print(f"Analyzing imports in {len(files)} files...")

        try:
            # Find common parent directory
            if len(files) == 1:
                target_dir = str(Path(files[0]).parent)
            else:
                paths = [Path(f) for f in files]
                common_prefix = Path(os.path.commonpath([str(p) for p in paths]))
                target_dir = str(common_prefix)

            analyzer = ImportAnalyzer(self.config)
            results = analyzer.analyze_directory(target_dir)

            # Filter to only our target files
            filtered_results = {
                "summary": {
                    "total_files_analyzed": len(files),
                    "total_imports": 0,
                    "total_issues": 0,
                    "duplicate_imports": 0,
                    "circular_dependencies": 0,
                    "conflicting_imports": 0,
                },
                "critical_issues": [],
                "warnings": [],
            }

            # Categorize issues by severity
            for issue_type in ["duplicate_imports", "circular_dependencies", "conflicting_imports"]:
                if issue_type in results.get("issues", {}):
                    for issue in results["issues"][issue_type]:
                        if issue.get("file") in files:
                            filtered_results["summary"][f"{issue_type}"] += 1
                            filtered_results["summary"]["total_issues"] += 1

                            # Categorize by severity
                            if issue.get("severity") == "error" or issue_type == "circular_dependencies":
                                filtered_results["critical_issues"].append(issue)
                            else:
                                filtered_results["warnings"].append(issue)

            # Get total imports for our files
            for file_path in files:
                if file_path in results.get("files", {}):
                    filtered_results["summary"]["total_imports"] += results["files"][file_path].get("total_imports", 0)

            return {
                "status": "success",
                "results": filtered_results,
                "full_results": results,
            }

        except Exception as e:
            print(f"Error detecting import conflicts: {e}")
            return {"status": "error", "error": str(e)}

    def _detect_signature_conflicts(self, files: list[str]) -> dict[str, Any]:
        """Detect function signature changes and compatibility issues."""
        print(f"Analyzing function signatures in {len(files)} files...")

        try:
            # Find common parent directory
            if len(files) == 1:
                target_dir = str(Path(files[0]).parent)
            else:
                paths = [Path(f) for f in files]
                common_prefix = Path(os.path.commonpath([str(p) for p in paths]))
                target_dir = str(common_prefix)

            analyzer = SignatureAnalyzer(self.config)
            results = analyzer.analyze_directory(target_dir)

            # Filter to only our target files
            filtered_results = {
                "summary": {
                    "total_files_analyzed": len(files),
                    "total_functions": 0,
                    "total_function_calls": 0,
                    "total_issues": 0,
                    "signature_changes": 0,
                    "compatibility_issues": 0,
                    "missing_functions": 0,
                    "unused_functions": 0,
                },
                "critical_issues": [],
                "warnings": [],
            }

            # Categorize issues by severity
            for issue_type in ["signature_changes", "compatibility_issues", "missing_functions", "unused_functions"]:
                if issue_type in results.get("issues", {}):
                    for issue in results["issues"][issue_type]:
                        if issue.get("file") in files:
                            filtered_results["summary"][f"{issue_type}"] += 1
                            filtered_results["summary"]["total_issues"] += 1

                            # Categorize by severity
                            if issue_type in ["compatibility_issues", "missing_functions"]:
                                filtered_results["critical_issues"].append(issue)
                            else:
                                filtered_results["warnings"].append(issue)

            # Get function counts for our files
            for file_path in files:
                if file_path in results.get("functions", {}):
                    filtered_results["summary"]["total_functions"] += len(results["functions"][file_path])

                if file_path in results.get("calls", {}):
                    filtered_results["summary"]["total_function_calls"] += len(results["calls"][file_path])

            return {
                "status": "success",
                "results": filtered_results,
                "full_results": results,
            }

        except Exception as e:
            print(f"Error detecting signature conflicts: {e}")
            return {"status": "error", "error": str(e)}

    def _detect_syntax_conflicts(self, files: list[str]) -> dict[str, Any]:
        """Detect syntax and compilation issues."""
        print(f"Validating syntax in {len(files)} files...")

        try:
            # Find common parent directory
            if len(files) == 1:
                target_dir = str(Path(files[0]).parent)
            else:
                paths = [Path(f) for f in files]
                common_prefix = Path(os.path.commonpath([str(p) for p in paths]))
                target_dir = str(common_prefix)

            validator = SyntaxValidator(self.config)
            results = validator.validate_directory(target_dir)

            # Filter to only our target files
            filtered_results = {
                "summary": {
                    "total_files": len(files),
                    "valid_files": 0,
                    "invalid_files": 0,
                    "ast_parseable_files": 0,
                    "compilable_files": 0,
                    "total_errors": 0,
                    "total_ast_nodes": 0,
                },
                "critical_issues": [],
                "warnings": [],
            }

            for file_path in files:
                if file_path in results.get("file_details", {}):
                    file_details = results["file_details"][file_path]

                    if file_details.get("syntax_valid", False):
                        filtered_results["summary"]["valid_files"] += 1
                    else:
                        filtered_results["summary"]["invalid_files"] += 1
                        # Add to critical issues
                        filtered_results["critical_issues"].append({
                            "file": file_path,
                            "type": "syntax_error",
                            "message": "File has syntax errors that prevent compilation",
                        })

                    if file_details.get("ast_parseable", False):
                        filtered_results["summary"]["ast_parseable_files"] += 1

                    if file_details.get("compilable", False):
                        filtered_results["summary"]["compilable_files"] += 1

                    # Count errors for this file
                    file_errors = results.get("errors_by_file", {}).get(file_path, [])
                    filtered_results["summary"]["total_errors"] += len(file_errors)

                    # Add syntax errors to critical issues
                    for error in file_errors:
                        filtered_results["critical_issues"].append({
                            "file": file_path,
                            "type": "syntax_error",
                            "message": error.get("message", "Unknown syntax error"),
                            "line": error.get("line", 0),
                        })

            # Get total AST nodes
            filtered_results["summary"]["total_ast_nodes"] = results.get("summary", {}).get("total_ast_nodes", 0)

            return {
                "status": "success",
                "results": filtered_results,
                "full_results": results,
            }

        except Exception as e:
            print(f"Error detecting syntax conflicts: {e}")
            return {"status": "error", "error": str(e)}

    def _detect_linter_conflicts(self, files: list[str]) -> dict[str, Any]:
        """Detect linter violations."""
        print(f"Running linter analysis on {len(files)} files...")

        try:
            # Find common parent directory
            if len(files) == 1:
                target_dir = str(Path(files[0]).parent)
            else:
                paths = [Path(f) for f in files]
                common_prefix = Path(os.path.commonpath([str(p) for p in paths]))
                target_dir = str(common_prefix)

            analyzer = LinterAnalyzer(self.config)
            results = analyzer.analyze_directory(target_dir)

            # Filter to only our target files
            filtered_results = {
                "summary": {
                    "total_issues": 0,
                    "total_files_with_issues": 0,
                    "total_errors": 0,
                    "total_warnings": 0,
                },
                "critical_issues": [],
                "warnings": [],
            }

            for file_path in files:
                if file_path in results.get("by_file", {}):
                    file_issues = results["by_file"][file_path]
                    filtered_results["summary"]["total_issues"] += len(file_issues)
                    filtered_results["summary"]["total_files_with_issues"] += 1

                    for issue in file_issues:
                        if issue.get("severity") == "error":
                            filtered_results["summary"]["total_errors"] += 1
                            filtered_results["critical_issues"].append(issue)
                        else:
                            filtered_results["summary"]["total_warnings"] += 1
                            filtered_results["warnings"].append(issue)

            return {
                "status": "success",
                "results": filtered_results,
                "full_results": results,
            }

        except Exception as e:
            print(f"Error detecting linter conflicts: {e}")
            return {"status": "error", "error": str(e)}

    def _generate_conflict_summary(self) -> dict[str, Any]:
        """Generate a summary of all detected conflicts."""
        summary = {
            "overall_status": "clean",
            "total_conflicts": 0,
            "critical_conflicts": 0,
            "warning_conflicts": 0,
            "conflict_categories": {},
            "recommendations": [],
        }

        # Count conflicts by category
        for conflict_type, conflict_data in self.results["conflicts"].items():
            if conflict_data.get("status") == "success":
                results = conflict_data.get("results", {})

                # Count critical issues
                critical_count = len(results.get("critical_issues", []))
                summary["critical_conflicts"] += critical_count

                # Count warnings
                warning_count = len(results.get("warnings", []))
                summary["warning_conflicts"] += warning_count

                # Store by category
                summary["conflict_categories"][conflict_type] = {
                    "critical": critical_count,
                    "warnings": warning_count,
                    "total": critical_count + warning_count,
                }

        summary["total_conflicts"] = summary["critical_conflicts"] + summary["warning_conflicts"]

        # Determine overall status
        if summary["critical_conflicts"] > 0:
            summary["overall_status"] = "critical"
        elif summary["warning_conflicts"] > 0:
            summary["overall_status"] = "warnings"
        else:
            summary["overall_status"] = "clean"

        # Generate recommendations
        if summary["critical_conflicts"] > 0:
            summary["recommendations"].append({
                "priority": "high",
                "message": f"Fix {summary['critical_conflicts']} critical conflicts before proceeding",
            })

        if summary["warning_conflicts"] > 0:
            summary["recommendations"].append({
                "priority": "medium",
                "message": f"Address {summary['warning_conflicts']} warnings for better code quality",
            })

        # Category-specific recommendations
        for category, counts in summary["conflict_categories"].items():
            if counts["critical"] > 0:
                summary["recommendations"].append({
                    "priority": "high",
                    "category": category,
                    "message": f"Resolve {counts['critical']} critical {category.replace('_', ' ')}",
                })

        return summary

    def _save_conflict_reports(self, output_dir: str) -> None:
        """Save conflict detection reports."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\nSaving conflict detection reports to: {output_path}")

        # Save main conflict report
        import json
        conflict_file = output_path / f"merge_conflict_detection_{self.timestamp}.json"
        with open(conflict_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Conflict detection report saved: {conflict_file}")

        # Save individual conflict reports
        for conflict_type, conflict_data in self.results["conflicts"].items():
            if conflict_data.get("status") == "success":
                conflict_report_file = output_path / f"{conflict_type}_conflicts_{self.timestamp}.json"
                with open(conflict_report_file, "w") as f:
                    json.dump(conflict_data, f, indent=2)
                print(f"{conflict_type} conflicts report saved: {conflict_report_file}")

    def _print_conflict_summary(self) -> None:
        """Print the final conflict summary."""
        summary = self.results["summary"]

        print("\n" + "="*70)
        print("MERGE CONFLICT DETECTION COMPLETED")
        print("="*70)
        print(f"Overall Status: {summary['overall_status'].upper()}")
        print(f"Timestamp: {self.timestamp}")

        print("\nConflict Summary:")
        print(f"  Total conflicts: {summary['total_conflicts']}")
        print(f"  Critical conflicts: {summary['critical_conflicts']}")
        print(f"  Warning conflicts: {summary['warning_conflicts']}")

        print("\nConflicts by Category:")
        for category, counts in summary["conflict_categories"].items():
            print(f"  {category.replace('_', ' ').title()}: {counts['total']} (Critical: {counts['critical']}, Warnings: {counts['warnings']})")

        if summary["recommendations"]:
            print("\nRecommendations:")
            for i, rec in enumerate(summary["recommendations"], 1):
                priority = rec.get("priority", "medium").upper()
                category = rec.get("category", "")
                message = rec["message"]

                if category:
                    print(f"  {i}. [{priority}] [{category}] {message}")
                else:
                    print(f"  {i}. [{priority}] {message}")

        # Print status message
        if summary["overall_status"] == "clean":
            print("\n✅ MERGE IS CLEAN - No conflicts detected!")
        elif summary["overall_status"] == "warnings":
            print("\n⚠️  MERGE HAS WARNINGS - Review and address warnings before proceeding")
        else:
            print("\n🚨 MERGE HAS CRITICAL CONFLICTS - Fix critical issues before proceeding!")


def main():
    """Command-line interface for merge conflict detection."""
    parser = argparse.ArgumentParser(
        description="Merge Conflict Detector - Detect issues after merges",
    )
    parser.add_argument("--target", required=True,
                       help="Path to Python file, directory, or comma-separated list of files")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--output", help="Output directory for reports")

    args = parser.parse_args()

    # Load configuration
    if args.config:
        from .core.config import load_config
        config = load_config(args.config)
    else:
        config = get_default_config()

    # Run conflict detection
    detector = MergeConflictDetector(config)
    results = detector.detect_conflicts(
        target=args.target,
        output_dir=args.output,
    )

    # Exit with appropriate code
    if results["summary"]["overall_status"] == "clean":
        return 0
    if results["summary"]["overall_status"] == "warnings":
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
