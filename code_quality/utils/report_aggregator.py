#!/usr/bin/env python3
"""
Report Aggregator for Code Quality Tools

This module provides functionality to aggregate results from multiple
code quality tools into unified reports with per-file and per-directory
information.
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


class ReportAggregator:
    """Aggregates code quality reports with per-file and per-directory analysis."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.file_issues = defaultdict(lambda: {
            "syntax_errors": [],
            "import_issues": [],
            "async_issues": [],
            "type_issues": [],
            "function_issues": [],
            "circular_imports": [],
            "security_issues": [],
            "performance_issues": [],
            "total_issues": 0,
            "fixed_issues": 0,
            "lines_of_code": 0,
        })
        self.directory_summary = defaultdict(lambda: {
            "total_files": 0,
            "files_with_issues": 0,
            "total_issues": 0,
            "fixed_issues": 0,
            "issue_breakdown": defaultdict(int),
        })
        self.overall_summary = {
            "total_files": 0,
            "total_directories": 0,
            "total_issues": 0,
            "fixed_issues": 0,
            "issue_breakdown": defaultdict(int),
            "critical_files": [],  # Files with most issues
            "clean_files": [],      # Files with no issues
        }

    def add_syntax_results(self, results: dict[str, Any]):
        """Add syntax checker results."""
        for file_info in results.get("fixed_files", []):
            file_path = self._normalize_path(file_info)
            self.file_issues[file_path]["syntax_errors"].extend(
                file_info.get("errors", []),
            )
            self.file_issues[file_path]["fixed_issues"] += 1

        for file_info in results.get("failed_files", []):
            file_path = self._normalize_path(file_info)
            self.file_issues[file_path]["syntax_errors"].append({
                "type": "syntax_error",
                "message": file_info.get("error", "Unknown syntax error"),
                "severity": "error",
            })

    def add_import_results(self, results: dict[str, Any]):
        """Add import checker results."""
        for file_info in results.get("fixed_files", []):
            file_path = self._normalize_path(file_info)
            self.file_issues[file_path]["import_issues"].extend(
                file_info.get("imports_added", []),
            )
            self.file_issues[file_path]["fixed_issues"] += len(
                file_info.get("imports_added", []),
            )

        import_errors = results.get("import_errors", {})
        for file_path, errors in import_errors.items():
            norm_path = self._normalize_path(file_path)
            for error in errors:
                self.file_issues[norm_path]["import_issues"].append({
                    "type": "import_error",
                    "message": error,
                    "severity": "error",
                })

    def add_async_results(self, results: dict[str, Any]):
        """Add async checker results."""
        for file_info in results.get("fixed_files", []):
            file_path = self._normalize_path(file_info)
            self.file_issues[file_path]["async_issues"].extend(
                file_info.get("changes", []),
            )
            self.file_issues[file_path]["fixed_issues"] += len(
                file_info.get("changes", []),
            )

    def add_type_results(self, results: dict[str, Any]):
        """Add type hint results."""
        for file_info in results.get("fixed_files", []):
            file_path = self._normalize_path(file_info["file"])
            self.file_issues[file_path]["type_issues"].extend(
                file_info.get("changes", []),
            )
            self.file_issues[file_path]["fixed_issues"] += len(
                file_info.get("changes", []),
            )

    def add_function_validation_results(self, results: dict[str, Any]):
        """Add function validation results."""
        for issue in results.get("issues", []):
            file_path = self._normalize_path(issue["file_path"])
            self.file_issues[file_path]["function_issues"].append({
                "type": issue["issue_type"],
                "message": issue["message"],
                "line": issue["line_number"],
                "severity": issue["severity"],
            })

    def add_circular_import_results(self, results: dict[str, Any]):
        """Add circular import detection results."""
        for cycle in results.get("circular_imports", []):
            # Add to each file in the cycle
            for module in cycle:
                if module.endswith(".py"):
                    file_path = self._normalize_path(module)
                else:
                    # Convert module path to file path
                    file_path = self._normalize_path(
                        module.replace(".", "/") + ".py",
                    )
                self.file_issues[file_path]["circular_imports"].append({
                    "cycle": cycle,
                    "severity": "warning",
                })

    def add_comprehensive_review_results(self, results: dict[str, Any]):
        """Add comprehensive review results."""
        for issue in results.get("issues", []):
            file_path = self._normalize_path(issue["file_path"])
            issue_type = issue["issue_type"]

            if "security" in issue_type.lower():
                self.file_issues[file_path]["security_issues"].append(issue)
            elif "performance" in issue_type.lower():
                self.file_issues[file_path]["performance_issues"].append(issue)
            # Add to appropriate category based on issue type
            elif "import" in issue_type:
                self.file_issues[file_path]["import_issues"].append(issue)
            elif "async" in issue_type or "await" in issue_type:
                self.file_issues[file_path]["async_issues"].append(issue)
            elif "type" in issue_type:
                self.file_issues[file_path]["type_issues"].append(issue)
            else:
                self.file_issues[file_path]["function_issues"].append(issue)

    def add_enhanced_validation_results(self, results: dict[str, Any]):
        """Add enhanced validation results (argument and data access validation)."""
        for issue in results.get("issues", []):
            file_path = self._normalize_path(issue["file_path"])

            # Categorize enhanced validation issues
            if "argument" in issue["issue_type"] or "parameter" in issue["issue_type"]:
                self.file_issues[file_path]["function_issues"].append({
                    "type": issue["issue_type"],
                    "line": issue["line_number"],
                    "message": issue["message"],
                    "severity": issue["severity"],
                    "suggestion": issue.get("suggestion", ""),
                    "source": "enhanced_validator",
                })
            elif "access" in issue["issue_type"] or "null" in issue["issue_type"] or "none" in issue["issue_type"].lower():
                # Data access issues
                self.file_issues[file_path]["security_issues"].append({
                    "type": issue["issue_type"],
                    "line": issue["line_number"],
                    "message": issue["message"],
                    "severity": issue["severity"],
                    "suggestion": issue.get("suggestion", ""),
                    "source": "enhanced_validator",
                })
            else:
                # Other issues
                self.file_issues[file_path]["syntax_errors"].append({
                    "type": issue["issue_type"],
                    "line": issue["line_number"],
                    "message": issue["message"],
                    "severity": issue["severity"],
                    "suggestion": issue.get("suggestion", ""),
                    "source": "enhanced_validator",
                })

    def _normalize_path(self, path: Any) -> str:
        """Normalize file path for consistency."""
        if isinstance(path, dict):
            path = path.get("file", path.get("file_path", ""))

        path = str(path)
        if path.startswith("/workspace/"):
            return path
        if path.startswith("src/"):
            return f"/workspace/{path}"
        return f"/workspace/src/{path}"

    def _count_lines_of_code(self, file_path: str) -> int:
        """Count lines of code in a file."""
        try:
            path = Path(file_path)
            if path.exists() and path.suffix == ".py":
                with open(path, encoding="utf-8") as f:
                    return len(f.readlines())
        except:
            pass
        return 0

    def generate_unified_report(self) -> dict[str, Any]:
        """Generate unified report with per-file and per-directory information."""
        # Calculate totals for each file
        for file_path, issues in self.file_issues.items():
            # Count total issues
            total_issues = (
                len(issues["syntax_errors"]) +
                len(issues["import_issues"]) +
                len(issues["async_issues"]) +
                len(issues["type_issues"]) +
                len(issues["function_issues"]) +
                len(issues["circular_imports"]) +
                len(issues["security_issues"]) +
                len(issues["performance_issues"])
            )
            issues["total_issues"] = total_issues
            issues["lines_of_code"] = self._count_lines_of_code(file_path)

            # Update directory summary
            dir_path = str(Path(file_path).parent)
            self.directory_summary[dir_path]["total_files"] += 1
            if total_issues > 0:
                self.directory_summary[dir_path]["files_with_issues"] += 1
            self.directory_summary[dir_path]["total_issues"] += total_issues
            self.directory_summary[dir_path]["fixed_issues"] += issues["fixed_issues"]

            # Update issue breakdown
            for issue_type in ["syntax_errors", "import_issues", "async_issues",
                             "type_issues", "function_issues", "circular_imports",
                             "security_issues", "performance_issues"]:
                count = len(issues[issue_type])
                if count > 0:
                    self.directory_summary[dir_path]["issue_breakdown"][issue_type] += count
                    self.overall_summary["issue_breakdown"][issue_type] += count

        # Calculate overall summary
        self.overall_summary["total_files"] = len(self.file_issues)
        self.overall_summary["total_directories"] = len(self.directory_summary)
        self.overall_summary["total_issues"] = sum(
            f["total_issues"] for f in self.file_issues.values()
        )
        self.overall_summary["fixed_issues"] = sum(
            f["fixed_issues"] for f in self.file_issues.values()
        )

        # Find critical and clean files
        files_by_issues = sorted(
            self.file_issues.items(),
            key=lambda x: x[1]["total_issues"],
            reverse=True,
        )

        # Top 10 files with most issues
        self.overall_summary["critical_files"] = [
            {
                "file": f[0],
                "issues": f[1]["total_issues"],
                "fixed": f[1]["fixed_issues"],
            }
            for f in files_by_issues[:10] if f[1]["total_issues"] > 0
        ]

        # Files with no issues
        self.overall_summary["clean_files"] = [
            f[0] for f in files_by_issues if f[1]["total_issues"] == 0
        ]

        return {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "overall_summary": dict(self.overall_summary),
            "directory_summary": dict(self.directory_summary),
            "file_details": dict(self.file_issues),
        }

    def generate_markdown_report(self) -> str:
        """Generate a human-readable markdown report."""
        report = self.generate_unified_report()

        md = []
        md.append("# Code Quality Unified Report")
        md.append(f"\n**Generated:** {report['timestamp']}")
        md.append(f"**Project Root:** {report['project_root']}")

        # Overall Summary
        summary = report["overall_summary"]
        md.append("\n## Overall Summary")
        md.append(f"- **Total Files Analyzed:** {summary['total_files']}")
        md.append(f"- **Total Directories:** {summary['total_directories']}")
        md.append(f"- **Total Issues Found:** {summary['total_issues']}")
        md.append(f"- **Issues Fixed:** {summary['fixed_issues']}")

        # Issue Breakdown
        md.append("\n### Issue Breakdown")
        for issue_type, count in summary["issue_breakdown"].items():
            md.append(f"- **{issue_type.replace('_', ' ').title()}:** {count}")

        # Critical Files
        if summary["critical_files"]:
            md.append("\n## Critical Files (Most Issues)")
            md.append("| File | Total Issues | Fixed |")
            md.append("|------|--------------|-------|")
            for file_info in summary["critical_files"]:
                file_name = Path(file_info["file"]).name
                md.append(f"| {file_name} | {file_info['issues']} | {file_info['fixed']} |")

        # Directory Summary
        md.append("\n## Directory Summary")
        sorted_dirs = sorted(
            report["directory_summary"].items(),
            key=lambda x: x[1]["total_issues"],
            reverse=True,
        )

        md.append("| Directory | Files | Files with Issues | Total Issues | Fixed |")
        md.append("|-----------|-------|-------------------|--------------|-------|")
        for dir_path, info in sorted_dirs[:20]:  # Top 20 directories
            dir_name = Path(dir_path).name
            md.append(
                f"| {dir_name} | {info['total_files']} | "
                f"{info['files_with_issues']} | {info['total_issues']} | "
                f"{info['fixed_issues']} |",
            )

        # File Details (Top files with issues)
        md.append("\n## File Details (Top 20)")
        files_with_issues = [
            (f, d) for f, d in report["file_details"].items()
            if d["total_issues"] > 0
        ]
        sorted_files = sorted(
            files_with_issues,
            key=lambda x: x[1]["total_issues"],
            reverse=True,
        )[:20]

        for file_path, details in sorted_files:
            md.append(f"\n### {Path(file_path).name}")
            md.append(f"**Path:** `{file_path}`")
            md.append(f"**Lines of Code:** {details['lines_of_code']}")
            md.append(f"**Total Issues:** {details['total_issues']} (Fixed: {details['fixed_issues']})")

            # Issue details
            for issue_type in ["syntax_errors", "import_issues", "async_issues",
                             "type_issues", "function_issues", "circular_imports",
                             "security_issues", "performance_issues"]:
                if details[issue_type]:
                    md.append(f"\n**{issue_type.replace('_', ' ').title()}:**")
                    for _i, issue in enumerate(details[issue_type][:3]):  # First 3 issues
                        if isinstance(issue, dict):
                            msg = issue.get("message", issue.get("type", str(issue)))
                        else:
                            msg = str(issue)
                        md.append(f"- {msg}")
                    if len(details[issue_type]) > 3:
                        md.append(f"- ... and {len(details[issue_type]) - 3} more")

        # Clean Files Summary
        if summary["clean_files"]:
            md.append("\n## Clean Files")
            md.append(f"**{len(summary['clean_files'])} files with no issues found**")

        return "\n".join(md)

    def save_reports(self, output_dir: Path, base_name: str = "unified_report"):
        """Save both JSON and Markdown reports."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON report
        json_path = output_dir / f"{base_name}_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(self.generate_unified_report(), f, indent=2, default=str)

        # Save Markdown report
        md_path = output_dir / f"{base_name}_{timestamp}.md"
        with open(md_path, "w") as f:
            f.write(self.generate_markdown_report())

        return json_path, md_path
