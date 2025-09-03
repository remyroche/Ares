#!/usr/bin/env python3
"""
Simplified Comprehensive Code Quality Analysis

This script runs all the built-in analyzers we created sequentially:
1. Syntax validation
2. Complexity analysis
3. Dead code detection
4. Dependency analysis
5. Import analysis
6. Call graph analysis
7. Signature analysis
8. Type checking
9. Advanced AST analysis
10. Architecture analysis
11. Code duplication analysis
12. Error handling analysis
13. Concurrency analysis

Provides:
- Number of files analyzed per directory
- Analysis results per category
- Global metrics and summary
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Import minimal modules
from minimal_config import get_default_config

# Add code_quality to path
code_quality_path = Path(__file__).parent / "code_quality"
sys.path.insert(0, str(code_quality_path))

try:
    from code_quality.analyzers.advanced_ast_analyzer import AdvancedASTAnalyzer
    from code_quality.analyzers.architecture_analyzer import ArchitectureAnalyzer
    from code_quality.analyzers.call_graph_analyzer import CallGraphAnalyzer
    from code_quality.analyzers.code_duplication_analyzer import CodeDuplicationAnalyzer
    from code_quality.analyzers.complexity_analyzer import ComplexityAnalyzer
    from code_quality.analyzers.concurrency_analyzer import ConcurrencyAnalyzer
    from code_quality.analyzers.dead_code_analyzer import DeadCodeAnalyzer
    from code_quality.analyzers.dependency_analyzer import DependencyAnalyzer
    from code_quality.analyzers.error_handling_analyzer import ErrorHandlingAnalyzer
    from code_quality.analyzers.import_analyzer import ImportAnalyzer
    from code_quality.analyzers.signature_analyzer import SignatureAnalyzer
    from code_quality.analyzers.syntax_validator import SyntaxValidator
    from code_quality.analyzers.type_checker import TypeChecker
    ANALYZERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some analyzers not available: {e}")
    ANALYZERS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("comprehensive_analysis_simplified.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("ComprehensiveAnalysisSimplified")


@dataclass
class AnalysisResult:
    """Container for analysis results."""
    file_path: str
    directory: str
    analyzer_name: str
    category: str
    issues_found: int
    issues_fixed: int
    details: dict[str, Any]
    processing_time: float
    status: str  # 'success', 'error', 'skipped'


@dataclass
class DirectorySummary:
    """Summary of analysis for a directory."""
    directory: str
    total_files: int
    files_analyzed: int
    total_issues: int
    total_fixed: int
    analyzers_run: list[str]
    categories_covered: list[str]
    processing_time: float


@dataclass
class GlobalMetrics:
    """Global metrics across all analysis."""
    total_directories: int
    total_files: int
    total_analyzers_run: int
    total_issues_found: int
    total_issues_fixed: int
    total_processing_time: float
    success_rate: float
    categories_covered: list[str]
    top_issues: list[tuple[str, int]]


class ComprehensiveAnalysisSimplified:
    """Runs all built-in code quality analyzers sequentially."""

    def __init__(self, project_root: str = ".", config_path: str | None = None):
        self.project_root = Path(project_root).resolve()
        self.config = get_default_config() if ANALYZERS_AVAILABLE else {}

        # Results storage
        self.analysis_results: list[AnalysisResult] = []
        self.directory_summaries: dict[str, DirectorySummary] = {}
        self.global_metrics = None

        # Available analyzers
        self.analyzers = {}
        self._initialize_analyzers()

        # Analysis categories
        self.categories = {
            "syntax": "Syntax and AST Analysis",
            "complexity": "Code Complexity Metrics",
            "dead_code": "Dead Code Detection",
            "dependencies": "Import and Dependency Analysis",
            "call_graph": "Function Call Graph Analysis",
            "signatures": "Function Signature Analysis",
            "type_checking": "Type Checking and Validation",
            "advanced_ast": "Advanced AST Analysis",
            "architecture": "Code Architecture Analysis",
            "code_duplication": "Code Duplication Detection",
            "error_handling": "Error Handling Analysis",
            "concurrency": "Concurrency Analysis",
        }

    def _initialize_analyzers(self):
        """Initialize all available analyzers."""
        if not ANALYZERS_AVAILABLE:
            return

        try:
            # Initialize all analyzers
            self.analyzers = {
                "syntax": SyntaxValidator(self.config),
                "complexity": ComplexityAnalyzer(self.config),
                "dead_code": DeadCodeAnalyzer(self.config),
                "dependencies": DependencyAnalyzer(self.config),
                "imports": ImportAnalyzer(self.config),
                "call_graph": CallGraphAnalyzer(self.config),
                "signatures": SignatureAnalyzer(self.config),
                "type_checking": TypeChecker(self.config),
                "advanced_ast": AdvancedASTAnalyzer(self.config),
                "architecture": ArchitectureAnalyzer(self.config),
                "code_duplication": CodeDuplicationAnalyzer(self.config),
                "error_handling": ErrorHandlingAnalyzer(self.config),
                "concurrency": ConcurrencyAnalyzer(self.config),
            }

            logger.info(f"✅ Initialized {len(self.analyzers)} analyzers")

        except Exception as e:
            logger.exception(f"❌ Failed to initialize some analyzers: {e}")

    def run_comprehensive_analysis(self) -> dict[str, Any]:
        """Run the complete analysis suite."""
        logger.info("🚀 Starting comprehensive code quality analysis...")
        start_time = time.time()

        # Find all Python files organized by directory
        python_files_by_dir = self._find_python_files_by_directory()
        logger.info(f"📁 Found Python files in {len(python_files_by_dir)} directories")

        # Run all analyzers
        logger.info("🔍 Running all analyzers...")
        self._run_all_analyzers(python_files_by_dir)

        # Generate summaries
        self._generate_directory_summaries(python_files_by_dir)
        self._generate_global_metrics(start_time)

        # Generate comprehensive report
        report = self._generate_comprehensive_report()

        total_time = time.time() - start_time
        logger.info(f"✅ Comprehensive analysis completed in {total_time:.2f} seconds")

        return report

    def _find_python_files_by_directory(self) -> dict[str, list[Path]]:
        """Find all Python files organized by directory."""
        python_files_by_dir = defaultdict(list)

        for root, dirs, files in os.walk(self.project_root):
            # Skip hidden directories and common exclusions
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ["__pycache__", "node_modules", "venv", "env"]]

            for file in files:
                if file.endswith(".py"):
                    file_path = Path(root) / file
                    relative_dir = str(file_path.parent.relative_to(self.project_root))
                    python_files_by_dir[relative_dir].append(file_path)

        return dict(python_files_by_dir)

    def _run_all_analyzers(self, python_files_by_dir: dict[str, list[Path]]):
        """Run all analyzers on all files."""
        total_files = sum(len(files) for files in python_files_by_dir.values())
        processed_files = 0

        for directory, files in python_files_by_dir.items():
            logger.info(f"📂 Analyzing directory: {directory} ({len(files)} files)")

            for file_path in files:
                processed_files += 1
                logger.info(f"  📄 Processing {file_path.name} ({processed_files}/{total_files})")

                # Run each analyzer on this file
                for analyzer_name, analyzer in self.analyzers.items():
                    try:
                        start_time = time.time()

                        # Check if analyzer can handle this file
                        if hasattr(analyzer, "can_analyze") and not analyzer.can_analyze(str(file_path)):
                            continue

                        # Run the analyzer
                        if hasattr(analyzer, "analyze"):
                            result = analyzer.analyze(str(file_path))
                        elif hasattr(analyzer, "validate_file"):
                            result = analyzer.validate_file(str(file_path))
                        else:
                            logger.warning(f"Analyzer {analyzer_name} has no analyze method")
                            continue

                        processing_time = time.time() - start_time

                        # Create analysis result
                        analysis_result = AnalysisResult(
                            file_path=str(file_path),
                            directory=directory,
                            analyzer_name=analyzer_name,
                            category=self._get_category_for_analyzer(analyzer_name),
                            issues_found=result.get("issues_found", 0),
                            issues_fixed=result.get("issues_fixed", 0),
                            details=result.get("details", {}),
                            processing_time=processing_time,
                            status=result.get("status", "success"),
                        )

                        self.analysis_results.append(analysis_result)

                        if result.get("issues_found", 0) > 0:
                            logger.info(f"    ⚠️  {analyzer_name}: {result['issues_found']} issues found")

                    except Exception as e:
                        logger.exception(f"    ❌ Error running {analyzer_name} on {file_path}: {e}")

                        # Create error result
                        error_result = AnalysisResult(
                            file_path=str(file_path),
                            directory=directory,
                            analyzer_name=analyzer_name,
                            category=self._get_category_for_analyzer(analyzer_name),
                            issues_found=0,
                            issues_fixed=0,
                            details={"error": str(e)},
                            processing_time=0,
                            status="error",
                        )

                        self.analysis_results.append(error_result)

    def _get_category_for_analyzer(self, analyzer_name: str) -> str:
        """Map analyzer name to category."""
        category_mapping = {
            "syntax": "syntax",
            "complexity": "complexity",
            "dead_code": "dead_code",
            "dependencies": "dependencies",
            "imports": "dependencies",
            "call_graph": "call_graph",
            "signatures": "signatures",
            "type_checking": "type_checking",
            "advanced_ast": "advanced_ast",
            "architecture": "architecture",
            "code_duplication": "code_duplication",
            "error_handling": "error_handling",
            "concurrency": "concurrency",
        }
        return category_mapping.get(analyzer_name, "other")

    def _generate_directory_summaries(self, python_files_by_dir: dict[str, list[Path]]):
        """Generate summaries for each directory."""
        logger.info("📊 Generating directory summaries...")

        for directory, files in python_files_by_dir.items():
            # Get results for this directory
            dir_results = [r for r in self.analysis_results if r.directory == directory]

            # Calculate metrics
            total_files = len(files)
            files_analyzed = len({r.file_path for r in dir_results})
            total_issues = sum(r.issues_found for r in dir_results)
            total_fixed = sum(r.issues_fixed for r in dir_results)
            analyzers_run = list({r.analyzer_name for r in dir_results})
            categories_covered = list({r.category for r in dir_results})
            processing_time = sum(r.processing_time for r in dir_results)

            # Create directory summary
            dir_summary = DirectorySummary(
                directory=directory,
                total_files=total_files,
                files_analyzed=files_analyzed,
                total_issues=total_issues,
                total_fixed=total_fixed,
                analyzers_run=analyzers_run,
                categories_covered=categories_covered,
                processing_time=processing_time,
            )

            self.directory_summaries[directory] = dir_summary

    def _generate_global_metrics(self, start_time: float):
        """Generate global metrics across all analysis."""
        logger.info("🌍 Generating global metrics...")

        # Calculate totals
        total_directories = len(self.directory_summaries)
        total_files = sum(s.total_files for s in self.directory_summaries.values())
        total_analyzers_run = len({r.analyzer_name for r in self.analysis_results})
        total_issues_found = sum(r.issues_found for r in self.analysis_results)
        total_issues_fixed = sum(r.issues_fixed for r in self.analysis_results)
        total_processing_time = time.time() - start_time

        # Calculate success rate
        successful_runs = len([r for r in self.analysis_results if r.status == "success"])
        total_runs = len(self.analysis_results)
        success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0

        # Get categories covered
        categories_covered = list({r.category for r in self.analysis_results})

        # Get top issues by category
        issues_by_category = Counter(r.category for r in self.analysis_results if r.issues_found > 0)
        top_issues = issues_by_category.most_common(10)

        self.global_metrics = GlobalMetrics(
            total_directories=total_directories,
            total_files=total_files,
            total_analyzers_run=total_analyzers_run,
            total_issues_found=total_issues_found,
            total_issues_fixed=total_issues_fixed,
            total_processing_time=total_processing_time,
            success_rate=success_rate,
            categories_covered=categories_covered,
            top_issues=top_issues,
        )

    def _generate_comprehensive_report(self) -> dict[str, Any]:
        """Generate a comprehensive report of all analysis results."""
        logger.info("📋 Generating comprehensive report...")

        # Convert dataclasses to dictionaries
        analysis_results_dict = [asdict(r) for r in self.analysis_results]
        directory_summaries_dict = {k: asdict(v) for k, v in self.directory_summaries.items()}
        global_metrics_dict = asdict(self.global_metrics) if self.global_metrics else {}

        # Group results by directory and category
        results_by_directory = defaultdict(lambda: defaultdict(list))
        for result in self.analysis_results:
            results_by_directory[result.directory][result.category].append(result)

        # Create the comprehensive report
        return {
            "metadata": {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "project_root": str(self.project_root),
                "total_analyzers": len(self.analyzers),
                "categories_analyzed": list(self.categories.keys()),
            },
            "global_metrics": global_metrics_dict,
            "directory_summaries": directory_summaries_dict,
            "analysis_results": analysis_results_dict,
            "results_by_directory": dict(results_by_directory),
            "category_summary": self._generate_category_summary(),
        }


    def _generate_category_summary(self) -> dict[str, Any]:
        """Generate summary for each analysis category."""
        category_summary = {}

        for category_name, category_description in self.categories.items():
            category_results = [r for r in self.analysis_results if r.category == category_name]

            if category_results:
                total_issues = sum(r.issues_found for r in category_results)
                total_fixed = sum(r.issues_fixed for r in category_results)
                files_analyzed = len({r.file_path for r in category_results})

                category_summary[category_name] = {
                    "description": category_description,
                    "files_analyzed": files_analyzed,
                    "total_issues": total_issues,
                    "total_fixed": total_fixed,
                    "success_rate": len([r for r in category_results if r.status == "success"]) / len(category_results) * 100,
                }

        return category_summary

    def save_report(self, report: dict[str, Any], output_file: str = None, text_summary: str = None):
        """Save the analysis report to files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save JSON report
        if output_file is None:
            output_file = f"comprehensive_analysis_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"💾 JSON report saved to: {output_file}")

        # Save text summary
        if text_summary is None:
            text_summary = f"comprehensive_analysis_{timestamp}.txt"

        with open(text_summary, "w") as f:
            f.write(self._generate_text_summary(report))

        logger.info(f"📝 Text summary saved to: {text_summary}")

    def _generate_text_summary(self, report: dict[str, Any]) -> str:
        """Generate a human-readable text summary."""
        lines = []
        lines.append("=" * 80)
        lines.append("COMPREHENSIVE CODE QUALITY ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {report['metadata']['generated_at']}")
        lines.append(f"Project: {report['metadata']['project_root']}")
        lines.append("")

        # Global metrics
        metrics = report["global_metrics"]
        lines.append("🌍 GLOBAL METRICS")
        lines.append("-" * 40)
        lines.append(f"Total Directories: {metrics['total_directories']}")
        lines.append(f"Total Files: {metrics['total_files']}")
        lines.append(f"Analyzers Run: {metrics['total_analyzers_run']}")
        lines.append(f"Total Issues Found: {metrics['total_issues_found']}")
        lines.append(f"Total Issues Fixed: {metrics['total_issues_fixed']}")
        lines.append(f"Success Rate: {metrics['success_rate']:.1f}%")
        lines.append(f"Processing Time: {metrics['total_processing_time']:.2f}s")
        lines.append("")

        # Category summary
        lines.append("📊 ANALYSIS CATEGORIES")
        lines.append("-" * 40)
        for summary in report["category_summary"].values():
            lines.append(f"{summary['description']}:")
            lines.append(f"  Files: {summary['files_analyzed']}, Issues: {summary['total_issues']}, Fixed: {summary['total_fixed']}")
            lines.append(f"  Success Rate: {summary['success_rate']:.1f}%")
            lines.append("")

        # Directory summaries
        lines.append("📁 DIRECTORY SUMMARIES")
        lines.append("-" * 40)
        for directory, summary in report["directory_summaries"].items():
            lines.append(f"Directory: {directory}")
            lines.append(f"  Files: {summary['total_files']}, Analyzed: {summary['files_analyzed']}")
            lines.append(f"  Issues: {summary['total_issues']}, Fixed: {summary['total_fixed']}")
            lines.append(f"  Categories: {', '.join(summary['categories_covered'])}")
            lines.append("")

        return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Code Quality Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run analysis on current directory
  python comprehensive_analysis_simplified.py

  # Run analysis on specific directory
  python comprehensive_analysis_simplified.py --project-root /path/to/project

  # Custom output file
  python comprehensive_analysis_simplified.py --output my_analysis.json

  # Verbose logging
  python comprehensive_analysis_simplified.py --verbose
        """,
    )

    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory to analyze (default: current)",
    )

    parser.add_argument(
        "--output",
        help="Output file for the JSON report",
    )

    parser.add_argument(
        "--text-summary",
        help="Output file for text summary",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check if analyzers are available
    if not ANALYZERS_AVAILABLE:
        print("❌ Required analyzers not available!")
        print("Please ensure the code_quality module is properly installed.")
        sys.exit(1)

    try:
        # Create analyzer and run analysis
        analyzer = ComprehensiveAnalysisSimplified(args.project_root)
        report = analyzer.run_comprehensive_analysis()

        # Save report
        analyzer.save_report(report, args.output, args.text_summary)

        print("\n✅ Analysis completed successfully!")
        print(f"📊 Found {report['global_metrics']['total_issues_found']} total issues")
        print(f"🔧 Fixed {report['global_metrics']['total_issues_fixed']} issues")
        print(f"📁 Analyzed {report['global_metrics']['total_files']} files in {report['global_metrics']['total_directories']} directories")

    except Exception as e:
        logger.exception(f"❌ Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
