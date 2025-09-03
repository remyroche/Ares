#!/usr/bin/env python3
"""
Command-line interface for Code Quality Tools.
"""

import argparse
import os
import sys
from pathlib import Path

from .analyzers.call_graph_analyzer import CallGraphAnalyzer
from .analyzers.dependency_analyzer import DependencyAnalyzer
from .analyzers.import_analyzer import ImportAnalyzer
from .analyzers.linter_analyzer import LinterAnalyzer
from .analyzers.signature_analyzer import SignatureAnalyzer
from .analyzers.syntax_validator import SyntaxValidator
from .core.config import get_default_config, load_config
from .fixers.auto_fixer import AutoFixer
from .fixers.sequential_fixer import SequentialFixer
from .merge_conflict_detector import MergeConflictDetector
from .reporters.quality_reporter import QualityReporter


def _run_auto_fix(args, config):
    """Run auto-fix on files or directories."""
    print("Running auto-fix...")

    if os.path.isfile(args.path):
        # Single file
        fixer = AutoFixer(config)
        fixer.fix_file(args.path)
        print("Auto-fix completed!")
        return 0
    # Directory
    fixer = AutoFixer(config)
    fixer.fix_all(args.path)

    # Print summary
    summary = fixer.get_fix_summary()
    print("\nAuto-fix Summary:")
    print(f"  Tools run: {', '.join(summary['tools_run'])}")
    print(f"  Successful: {', '.join(summary['successful_tools'])}")
    if summary["failed_tools"]:
        print(f"  Failed: {', '.join(summary['failed_tools'])}")

    print("Auto-fix completed!")
    return 0


def _run_linter_analysis(args, config):
    """Run linter analysis."""
    print("Running linter analysis...")

    analyzer = LinterAnalyzer(config)
    results = analyzer.analyze_directory(args.path)

    print("\nLinter Analysis Results:")
    print(f"  Total issues: {results['total_issues']}")
    print(f"  Files with issues: {results['total_files_with_issues']}")
    print(f"  Errors: {results['total_errors']}")
    print(f"  Warnings: {results['total_warnings']}")

    if args.output:
        import json
        output_path = Path(args.output)
        output_path.mkdir(exist_ok=True)

        report_file = output_path / "linter_analysis_report.json"
        with open(report_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Report saved to: {report_file}")

    return 0


def _run_syntax_validation(args, config):
    """Run syntax validation."""
    print("Running syntax validation...")

    validator = SyntaxValidator(config)
    results = validator.validate_directory(args.path)

    print("\nSyntax Validation Results:")
    print(f"  Total files: {results['summary']['total_files']}")
    print(f"  Valid files: {results['summary']['valid_files']}")
    print(f"  Invalid files: {results['summary']['invalid_files']}")
    print(f"  AST parseable: {results['summary']['ast_parseable_files']}")
    print(f"  Compilable: {results['summary']['compilable_files']}")
    print(f"  Total errors: {results['summary']['total_errors']}")

    if args.output:
        import json
        output_path = Path(args.output)
        output_path.mkdir(exist_ok=True)

        report_file = output_path / "syntax_validation_report.json"
        with open(report_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Report saved to: {report_file}")

    return 0


def _run_call_graph_analysis(args, config):
    """Run call graph analysis."""
    print("Running call graph analysis...")

    analyzer = CallGraphAnalyzer(config)
    results = analyzer.analyze_directory(args.path)

    print("\nCall Graph Analysis Results:")
    print(f"  Total files: {results['summary']['total_files']}")
    print(f"  Total functions: {results['summary']['total_functions']}")
    print(f"  Total calls: {results['summary']['total_calls']}")
    print(f"  Dead code candidates: {results['summary']['dead_code_candidates']}")
    print(f"  Circular dependencies: {results['summary']['circular_dependencies']}")

    if args.output:
        import json
        output_path = Path(args.output)
        output_path.mkdir(exist_ok=True)

        # Save main report
        report_file = output_path / "call_graph_analysis_report.json"
        with open(report_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Report saved to: {report_file}")

        # Export graph if requested
        if args.export_graph:
            graph_file = output_path / "call_graph.gml"
            analyzer.export_graph(graph_file)
            print(f"Graph exported to: {graph_file}")

    return 0


def _run_dependency_analysis(args, config):
    """Run dependency analysis."""
    print("Running dependency analysis...")

    analyzer = DependencyAnalyzer(config)
    results = analyzer.analyze_directory(args.path)

    print("\nDependency Analysis Results:")
    print(f"  Total dependencies: {results['summary']['total_dependencies']}")
    print(f"  Missing dependencies: {results['summary']['missing_dependencies']}")
    print(f"  Unused dependencies: {results['summary']['unused_dependencies']}")
    print(f"  Security issues: {results['summary']['security_issues']}")

    if args.output:
        import json
        output_path = Path(args.output)
        output_path.mkdir(exist_ok=True)

        report_file = output_path / "dependency_analysis_report.json"
        with open(report_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Report saved to: {report_file}")

    return 0


def _run_import_analysis(args, config):
    """Run import analysis."""
    print("Running import analysis...")

    analyzer = ImportAnalyzer(config)
    results = analyzer.analyze_directory(args.path)

    print("\nImport Analysis Results:")
    print(f"  Total files analyzed: {results['summary']['total_files_analyzed']}")
    print(f"  Total imports: {results['summary']['total_imports']}")
    print(f"  Total issues: {results['summary']['total_issues']}")
    print(f"  Duplicate imports: {results['summary']['duplicate_imports']}")
    print(f"  Circular dependencies: {results['summary']['circular_dependencies']}")
    print(f"  Conflicting imports: {results['summary']['conflicting_imports']}")

    if args.output:
        import json
        output_path = Path(args.output)
        output_path.mkdir(exist_ok=True)

        # Save main report
        report_file = output_path / "import_analysis_report.json"
        with open(report_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Report saved to: {report_file}")

        # Export import graph if requested
        if args.export_graph:
            graph_file = output_path / "import_graph.png"
            analyzer.visualize_import_graph(graph_file)
            print(f"Import graph saved to: {graph_file}")

    return 0


def _run_signature_analysis(args, config):
    """Run function signature analysis."""
    print("Running function signature analysis...")

    analyzer = SignatureAnalyzer(config)
    results = analyzer.analyze_directory(args.path)

    print("\nFunction Signature Analysis Results:")
    print(f"  Total files analyzed: {results['summary']['total_files_analyzed']}")
    print(f"  Total functions: {results['summary']['total_functions']}")
    print(f"  Total function calls: {results['summary']['total_function_calls']}")
    print(f"  Total issues: {results['summary']['total_issues']}")
    print(f"  Signature changes: {results['summary']['signature_changes']}")
    print(f"  Compatibility issues: {results['summary']['compatibility_issues']}")
    print(f"  Missing functions: {results['summary']['missing_functions']}")
    print(f"  Unused functions: {results['summary']['unused_functions']}")

    if args.output:
        import json
        output_path = Path(args.output)
        output_path.mkdir(exist_ok=True)

        report_file = output_path / "signature_analysis_report.json"
        with open(report_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Report saved to: {report_file}")

    return 0


def _run_sequential_fix(args, config):
    """Run sequential auto-fix pipeline."""
    print("Running sequential auto-fix pipeline...")

    # Parse target
    if "," in args.target:
        # Comma-separated list of files
        target = [f.strip() for f in args.target.split(",")]
    else:
        target = args.target

    fixer = SequentialFixer(config)
    results = fixer.run_pipeline(
        target=target,
        output_dir=args.output,
        create_backups=not args.no_backups,
    )

    print("Sequential fix pipeline completed!")

    # Return appropriate exit code
    if results["summary"]["overall_status"] == "success":
        return 0
    if results["summary"]["overall_status"] == "partial":
        return 1
    return 2


def _run_merge_conflict_detection(args, config):
    """Run merge conflict detection."""
    print("Running merge conflict detection...")

    detector = MergeConflictDetector(config)
    results = detector.detect_conflicts(
        target=args.target,
        output_dir=args.output,
    )

    print("Merge conflict detection completed!")

    # Return appropriate exit code
    if results["summary"]["overall_status"] == "clean":
        return 0
    if results["summary"]["overall_status"] == "warnings":
        return 1
    return 2


def _run_quality_report(args, config):
    """Run comprehensive quality report."""
    print("Running comprehensive quality report...")

    reporter = QualityReporter(config)
    results = reporter.generate_comprehensive_report(args.path)

    print("\nQuality Report Generated:")
    print(f"  Overall score: {results['overall_score']}/100")
    print(f"  Files analyzed: {results['summary']['total_files']}")
    print(f"  Issues found: {results['summary']['total_issues']}")

    if args.output:
        print(f"Reports saved to: {args.output}")

    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Code Quality Tools - Comprehensive Python code analysis and fixing",
    )
    parser.add_argument("--config", help="Path to configuration file")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Auto-fix command
    auto_fix_parser = subparsers.add_parser("auto-fix", help="Auto-fix Python code")
    auto_fix_parser.add_argument("--path", required=True, help="Path to Python file or directory")

    # Linter analysis command
    linter_parser = subparsers.add_parser("linter", help="Run linter analysis")
    linter_parser.add_argument("--path", required=True, help="Path to Python file or directory")
    linter_parser.add_argument("--output", help="Output directory for reports")

    # Syntax validation command
    syntax_parser = subparsers.add_parser("syntax", help="Run syntax validation")
    syntax_parser.add_argument("--path", required=True, help="Path to Python file or directory")
    syntax_parser.add_argument("--output", help="Output directory for reports")

    # Call graph analysis command
    call_graph_parser = subparsers.add_parser("call-graph", help="Run call graph analysis")
    call_graph_parser.add_argument("--path", required=True, help="Path to Python file or directory")
    call_graph_parser.add_argument("--output", help="Output directory for reports")
    call_graph_parser.add_argument("--export-graph", action="store_true", help="Export call graph")

    # Dependency analysis command
    dependency_parser = subparsers.add_parser("dependencies", help="Run dependency analysis")
    dependency_parser.add_argument("--path", required=True, help="Path to Python file or directory")
    dependency_parser.add_argument("--output", help="Output directory for reports")

    # Import analysis command
    import_parser = subparsers.add_parser("imports", help="Run import analysis")
    import_parser.add_argument("--path", required=True, help="Path to Python file or directory")
    import_parser.add_argument("--output", help="Output directory for reports")
    import_parser.add_argument("--export-graph", action="store_true", help="Export import graph")

    # Signature analysis command
    signature_parser = subparsers.add_parser("signatures", help="Run function signature analysis")
    signature_parser.add_argument("--path", required=True, help="Path to Python file or directory")
    signature_parser.add_argument("--output", help="Output directory for reports")

    # Sequential fix command
    sequential_parser = subparsers.add_parser("sequential-fix", help="Run sequential auto-fix pipeline")
    sequential_parser.add_argument("--target", required=True,
                                 help="Path to Python file, directory, or comma-separated list of files")
    sequential_parser.add_argument("--config", help="Path to configuration file")
    sequential_parser.add_argument("--output", help="Output directory for reports")
    sequential_parser.add_argument("--no-backups", action="store_true", help="Disable backup creation")

    # Merge conflict detection command
    merge_conflict_parser = subparsers.add_parser("merge-conflicts", help="Detect merge conflicts and compatibility issues")
    merge_conflict_parser.add_argument("--target", required=True,
                                     help="Path to Python file, directory, or comma-separated list of files")
    merge_conflict_parser.add_argument("--config", help="Path to configuration file")
    merge_conflict_parser.add_argument("--output", help="Output directory for reports")

    # Quality report command
    quality_parser = subparsers.add_parser("quality-report", help="Generate comprehensive quality report")
    quality_parser.add_argument("--path", required=True, help="Path to Python file or directory")
    quality_parser.add_argument("--output", help="Output directory for reports")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Load configuration
    config = load_config(args.config) if args.config else get_default_config()

    # Dispatch to appropriate handler
    if args.command == "auto-fix":
        return _run_auto_fix(args, config)
    if args.command == "linter":
        return _run_linter_analysis(args, config)
    if args.command == "syntax":
        return _run_syntax_validation(args, config)
    if args.command == "call-graph":
        return _run_call_graph_analysis(args, config)
    if args.command == "dependencies":
        return _run_dependency_analysis(args, config)
    if args.command == "imports":
        return _run_import_analysis(args, config)
    if args.command == "signatures":
        return _run_signature_analysis(args, config)
    if args.command == "sequential-fix":
        return _run_sequential_fix(args, config)
    if args.command == "merge-conflicts":
        return _run_merge_conflict_detection(args, config)
    if args.command == "quality-report":
        return _run_quality_report(args, config)
    print(f"Unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
