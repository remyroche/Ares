#!/usr/bin/env python3
"""
Function Validator Wrapper - Provides compatibility methods for pipeline integration
"""

from collections import defaultdict

from function_validator import FunctionValidator as BaseValidator


class FunctionValidator(BaseValidator):
    """Extended FunctionValidator with pipeline-compatible methods."""

    def __init__(self, project_root: str, exclude_patterns=None):
        super().__init__(project_root, exclude_patterns)
        self.files_analyzed = []

    def validate_all_files(self) -> None:
        """Validate all files in the project (pipeline compatibility method)."""
        # This calls the actual validation method
        self.validate_project()

        # Extract files analyzed from the processed files
        self.files_analyzed = list({issue.file_path for issue in self.issues})

    def get_issue_summary(self) -> dict[str, int]:
        """Get a summary of issues by type."""
        summary = defaultdict(int)

        for issue in self.issues:
            summary[issue.issue_type] += 1

        # Also include the stats
        summary.update({
            "total_issues": len(self.issues),
            "files_processed": self.stats["files_processed"],
            "undefined_functions": self.stats["undefined_functions"],
            "missing_await": self.stats["missing_await"],
            "parameter_mismatches": self.stats["parameter_mismatches"],
        })

        return dict(summary)
