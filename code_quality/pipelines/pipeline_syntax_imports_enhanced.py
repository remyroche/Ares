#!/usr/bin/env python3
"""
Enhanced Syntax and Import Pipeline with Unified Reporting

This pipeline handles all syntax-related fixes and import management
with comprehensive per-file and per-directory reporting.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.advanced_syntax_fixer import AdvancedSyntaxFixer
from scripts.detect_circular_imports import CircularImportDetector
from scripts.safe_import_fixer import SafeImportFixer
from utils.report_aggregator import ReportAggregator


class SyntaxImportPipelineEnhanced:
    """Enhanced pipeline for syntax and import-related fixes with unified reporting."""

    def __init__(self, project_root: str = "/workspace/src"):
        self.project_root = Path(project_root)
        self.reports_dir = Path("/workspace/code_quality/reports")
        self.reports_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {
            "syntax_fixes": {},
            "import_fixes": {},
            "circular_imports": {},
            "summary": {},
        }
        self.report_aggregator = ReportAggregator(project_root)

    def run_syntax_fixes(self) -> dict[str, Any]:
        """Run advanced syntax fixes."""
        print("\n" + "="*60)
        print("Running Advanced Syntax Fixes")
        print("="*60)

        fixer = AdvancedSyntaxFixer(str(self.project_root))
        fixer.fix_all_files()

        result = {
            "fixed_files": fixer.fixed_files,
            "failed_files": fixer.failed_files,
            "syntax_errors": dict(fixer.syntax_errors),
            "total_fixed": len(fixer.fixed_files),
            "total_failed": len(fixer.failed_files),
        }

        # Add to aggregator
        self.report_aggregator.add_syntax_results(result)

        # Save individual report
        report_path = self.reports_dir / f"syntax_fixes_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)

        self.results["syntax_fixes"] = result
        return result

    def run_import_fixes(self) -> dict[str, Any]:
        """Run import fixes."""
        print("\n" + "="*60)
        print("Running Import Fixes")
        print("="*60)

        fixer = SafeImportFixer(str(self.project_root))
        fixer.fix_all_files()

        result = {
            "fixed_files": fixer.fixed_files,
            "failed_files": fixer.failed_files,
            "import_errors": dict(fixer.import_errors),
            "total_fixed": len(fixer.fixed_files),
            "total_failed": len(fixer.failed_files),
        }

        # Add to aggregator
        self.report_aggregator.add_import_results(result)

        # Save individual report
        report_path = self.reports_dir / f"import_fixes_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)

        self.results["import_fixes"] = result
        return result

    def detect_circular_imports(self) -> dict[str, Any]:
        """Detect circular imports."""
        print("\n" + "="*60)
        print("Detecting Circular Imports")
        print("="*60)

        detector = CircularImportDetector(str(self.project_root))
        cycles = detector.find_circular_imports()

        result = {
            "circular_imports": cycles,
            "total_cycles": len(cycles),
            "affected_modules": list({
                module for cycle in cycles
                for module in cycle
            }),
        }

        # Add to aggregator
        self.report_aggregator.add_circular_import_results(result)

        # Save individual report
        report_path = self.reports_dir / f"circular_imports_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)

        self.results["circular_imports"] = result
        return result

    def run_full_pipeline(self) -> dict[str, Any]:
        """Run the complete syntax and import pipeline with unified reporting."""
        print("\n" + "="*80)
        print("SYNTAX AND IMPORT PIPELINE - ENHANCED")
        print("="*80)
        print(f"Project root: {self.project_root}")
        print(f"Timestamp: {self.timestamp}")

        # Run each step
        syntax_result = self.run_syntax_fixes()
        import_result = self.run_import_fixes()
        circular_result = self.detect_circular_imports()

        # Create summary
        self.results["summary"] = {
            "timestamp": self.timestamp,
            "project_root": str(self.project_root),
            "syntax_fixes": {
                "fixed": syntax_result["total_fixed"],
                "failed": syntax_result["total_failed"],
            },
            "import_fixes": {
                "fixed": import_result["total_fixed"],
                "failed": import_result["total_failed"],
            },
            "circular_imports": {
                "cycles_found": circular_result["total_cycles"],
                "affected_modules": len(circular_result["affected_modules"]),
            },
        }

        # Save pipeline report
        report_path = self.reports_dir / f"syntax_import_pipeline_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)

        # Generate and save unified reports
        print("\n" + "="*60)
        print("Generating Unified Reports")
        print("="*60)

        json_report, md_report = self.report_aggregator.save_reports(
            self.reports_dir,
            "syntax_import_unified_report",
        )

        print("\nUnified reports saved:")
        print(f"  JSON: {json_report}")
        print(f"  Markdown: {md_report}")

        # Print summary
        self._print_summary()

        # Print aggregated summary
        self._print_aggregated_summary()

        return self.results

    def _print_summary(self):
        """Print pipeline execution summary."""
        print("\n" + "="*80)
        print("PIPELINE SUMMARY")
        print("="*80)
        print(f"Syntax fixes: {self.results['syntax_fixes']['total_fixed']} fixed, "
              f"{self.results['syntax_fixes']['total_failed']} failed")
        print(f"Import fixes: {self.results['import_fixes']['total_fixed']} fixed, "
              f"{self.results['import_fixes']['total_failed']} failed")
        print(f"Circular imports: {self.results['circular_imports']['total_cycles']} cycles found")

    def _print_aggregated_summary(self):
        """Print unified report summary."""
        report = self.report_aggregator.generate_unified_report()
        summary = report["overall_summary"]

        print("\n" + "="*80)
        print("UNIFIED REPORT SUMMARY")
        print("="*80)
        print(f"Total Files Analyzed: {summary['total_files']}")
        print(f"Total Issues Found: {summary['total_issues']}")
        print(f"Issues Fixed: {summary['fixed_issues']}")

        # Show breakdown for relevant issue types
        print("\nIssue Breakdown:")
        for issue_type in ["syntax_errors", "import_issues", "circular_imports"]:
            if issue_type in summary["issue_breakdown"]:
                count = summary["issue_breakdown"][issue_type]
                if count > 0:
                    print(f"  {issue_type.replace('_', ' ').title()}: {count}")

        # Show top problematic files
        if summary["critical_files"]:
            print("\nTop Files with Issues:")
            for i, file_info in enumerate(summary["critical_files"][:5]):
                file_name = Path(file_info["file"]).name
                print(f"  {i+1}. {file_name}: {file_info['issues']} issues")

        print(f"\nReports saved to: {self.reports_dir}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run enhanced syntax and import fixes pipeline with unified reporting",
    )
    parser.add_argument("--project-root", default="/workspace/src",
                        help="Project root directory")
    parser.add_argument("--syntax-only", action="store_true",
                        help="Run only syntax fixes")
    parser.add_argument("--imports-only", action="store_true",
                        help="Run only import fixes")
    parser.add_argument("--circular-only", action="store_true",
                        help="Run only circular import detection")

    args = parser.parse_args()

    pipeline = SyntaxImportPipelineEnhanced(args.project_root)

    if args.syntax_only:
        pipeline.run_syntax_fixes()
        # Still generate unified report for partial run
        pipeline.report_aggregator.save_reports(
            pipeline.reports_dir,
            "syntax_only_report",
        )
    elif args.imports_only:
        pipeline.run_import_fixes()
        pipeline.report_aggregator.save_reports(
            pipeline.reports_dir,
            "imports_only_report",
        )
    elif args.circular_only:
        pipeline.detect_circular_imports()
        pipeline.report_aggregator.save_reports(
            pipeline.reports_dir,
            "circular_only_report",
        )
    else:
        pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
