#!/usr/bin/env python3
"""
Comprehensive Professional Code Quality Analysis

This script runs ALL fixers, plugins, and advanced analyzers sequentially:
1. Auto-fixers (Black, isort, autopep8, etc.)
2. All plugins (fixers and analyzers)
3. Advanced analyzers (complexity, dead code, dependencies, etc.)

Provides:
- Number of files analyzed per directory
- Auto-fix results per directory
- Full analysis per category
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

# Add code_quality to path
code_quality_path = Path(__file__).parent / "code_quality"
sys.path.insert(0, str(code_quality_path))

try:
    from code_quality.analyzers.advanced_ast_analyzer import AdvancedASTAnalyzer as AdvancedASTAnalyzer_2
    from code_quality.analyzers.architecture_analyzer import ArchitectureAnalyzer
    from code_quality.analyzers.call_graph_analyzer import CallGraphAnalyzer
    from code_quality.analyzers.code_duplication_analyzer import CodeDuplicationAnalyzer
    from code_quality.analyzers.complexity_analyzer import ComplexityAnalyzer as ComplexityAnalyzer_3
    from code_quality.analyzers.concurrency_analyzer import ConcurrencyAnalyzer as ConcurrencyAnalyzer_2
    from code_quality.analyzers.dead_code_analyzer import DeadCodeAnalyzer as DeadCodeAnalyzer_3
    from code_quality.analyzers.dependency_analyzer import DependencyAnalyzer
    from code_quality.analyzers.error_handling_analyzer import ErrorHandlingAnalyzer as ErrorHandlingAnalyzer_2
    from code_quality.analyzers.import_analyzer import ImportAnalyzer
    from code_quality.analyzers.signature_analyzer import SignatureAnalyzer as SignatureAnalyzer_2
    from code_quality.analyzers.syntax_validator import SyntaxValidator as SyntaxValidator_3
    from code_quality.analyzers.type_checker import TypeChecker as TypeChecker_code_quality_analyzers_type_checker
    from code_quality.core.config import get_default_config as get_default_config_code_quality_core_config
    from code_quality.core.plugins import PluginManager as PluginManager_code_quality_core_plugins
    from code_quality.fixers.auto_fixer import AutoFixer as AutoFixer_code_quality_fixers_auto_fixer
    from code_quality.fixers.sequential_fixer import SequentialFixer as SequentialFixer_3
    CODE_QUALITY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Code quality tools not available: {e}")
    CODE_QUALITY_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("comprehensive_analysis.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("ComprehensiveProfessionalAnalysis")


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


class ComprehensiveProfessionalAnalyzer:
    """Runs all professional code quality tools sequentially."""

    def __init__(self, project_root: str = ".", config_path: str | None = None):
        self.project_root = Path(project_root).resolve()
        self.config = get_default_config() if CODE_QUALITY_AVAILABLE else {}

        # Results storage
        self.analysis_results: list[AnalysisResult] = []
        self.directory_summaries: dict[str, DirectorySummary] = {}
        self.global_metrics = None

        # Plugin manager
        self.plugin_manager = None
        if CODE_QUALITY_AVAILABLE:
            self.plugin_manager = PluginManager()

        # Available analyzers
        self.analyzers = {}
        self.fixers = {}
        self._initialize_tools()

        # Analysis categories - comprehensive code quality analysis
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
            "auto_fixing": "Automatic Code Fixing",
        }

    def _initialize_tools(self):
        """Initialize all available tools."""
        if not CODE_QUALITY_AVAILABLE:
            return

        try:
            # Initialize analyzers - comprehensive code quality analysis
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

            # Initialize fixers
            self.fixers = {
                "auto_fixer": AutoFixer(self.config),
                "sequential_fixer": SequentialFixer(self.config),
            }

            logger.info(f"Initialized {len(self.analyzers)} analyzers and {len(self.fixers)} fixers")

        except Exception as e:
            logger.exception(f"Failed to initialize some tools: {e}")

    def run_comprehensive_analysis(self) -> dict[str, Any]:
        """Run the complete professional analysis suite."""
        logger.info("🚀 Starting comprehensive professional code quality analysis...")
        start_time = time.time()

        # Find all Python files organized by directory
        python_files_by_dir = self._find_python_files_by_directory()
        logger.info(f"Found Python files in {len(python_files_by_dir)} directories")

        # Phase 1: Auto-fixing
        logger.info("🔧 Phase 1: Running auto-fixers...")
        self._run_auto_fixers(python_files_by_dir)

        # Phase 2: Advanced analysis
        logger.info("🔍 Phase 2: Running advanced analyzers...")
        self._run_advanced_analyzers(python_files_by_dir)

        # Phase 3: Plugin analysis
        logger.info("🔌 Phase 3: Running plugin analysis...")
        self._run_plugin_analysis(python_files_by_dir)

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

        exclude_patterns = [
            "__pycache__", ".git", "venv", "env", "node_modules",
            ".pytest_cache", "code_quality_env", ".venv",
        ]

        for root, dirs, files in os.walk(self.project_root):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_patterns]

            for file in files:
                if file.endswith(".py"):
                    file_path = Path(root) / file
                    # Get relative directory from project root
                    rel_dir = str(file_path.parent.relative_to(self.project_root))
                    if rel_dir == ".":
                        rel_dir = "root"
                    python_files_by_dir[rel_dir].append(file_path)

        return dict(python_files_by_dir)

    def _run_auto_fixers(self, python_files_by_dir: dict[str, list[Path]]):
        """Run all auto-fixers on Python files."""
        if not self.fixers:
            logger.warning("No fixers available, skipping auto-fixing phase")
            return

        logger.info("🔧 Running auto-fixers...")

        for fixer_name, fixer in self.fixers.items():
            logger.info(f"  Running {fixer_name}...")

            for directory, files in python_files_by_dir.items():
                for file_path in files:
                    try:
                        start_time = time.time()

                        # Check if fixer can handle this file
                        if hasattr(fixer, "can_fix") and not fixer.can_fix(str(file_path)):
                            continue

                        # Run the fixer
                        if hasattr(fixer, "fix"):
                            result = fixer.fix(str(file_path))
                        else:
                            result = {"status": "no_fix_method", "issues_fixed": 0}

                        processing_time = time.time() - start_time

                        # Record result
                        analysis_result = AnalysisResult(
                            file_path=str(file_path),
                            directory=directory,
                            analyzer_name=fixer_name,
                            category="auto_fixing",
                            issues_found=result.get("issues_found", 0),
                            issues_fixed=result.get("issues_fixed", 0),
                            details=result,
                            processing_time=processing_time,
                            status="success",
                        )
                        self.analysis_results.append(analysis_result)

                    except Exception as e:
                        logger.exception(f"Error running {fixer_name} on {file_path}: {e}")
                        analysis_result = AnalysisResult(
                            file_path=str(file_path),
                            directory=directory,
                            analyzer_name=fixer_name,
                            category="auto_fixing",
                            issues_found=0,
                            issues_fixed=0,
                            details={"error": str(e)},
                            processing_time=0,
                            status="error",
                        )
                        self.analysis_results.append(analysis_result)

    def _run_advanced_analyzers(self, python_files_by_dir: dict[str, list[Path]]):
        """Run all advanced analyzers on Python files."""
        if not self.analyzers:
            logger.warning("No analyzers available, skipping analysis phase")
            return

        logger.info("🔍 Running advanced analyzers...")

        for analyzer_name, analyzer in self.analyzers.items():
            logger.info(f"  Running {analyzer_name}...")

            for directory, files in python_files_by_dir.items():
                for file_path in files:
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
                            result = {"status": "no_analysis_method", "issues_found": 0}

                        processing_time = time.time() - start_time

                        # Record result
                        analysis_result = AnalysisResult(
                            file_path=str(file_path),
                            directory=directory,
                            analyzer_name=analyzer_name,
                            category=self._get_category_for_analyzer(analyzer_name),
                            issues_found=result.get("issues_found", 0),
                            issues_fixed=result.get("issues_fixed", 0),
                            details=result,
                            processing_time=processing_time,
                            status="success",
                        )
                        self.analysis_results.append(analysis_result)

                    except Exception as e:
                        logger.exception(f"Error running {analyzer_name} on {file_path}: {e}")
                        analysis_result = AnalysisResult(
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
                        self.analysis_results.append(analysis_result)

    def _run_plugin_analysis(self, python_files_by_dir: dict[str, list[Path]]):
        """Run plugin-based analysis if available."""
        if not self.plugin_manager:
            logger.info("No plugin manager available, skipping plugin analysis")
            return

        logger.info("🔌 Running plugin analysis...")

        try:
            # Get all available plugins
            plugins = self.plugin_manager.get_all_plugins()
            logger.info(f"Found {len(plugins)} plugins")

            for plugin_name, plugin in plugins.items():
                logger.info(f"  Running plugin {plugin_name}...")

                for directory, files in python_files_by_dir.items():
                    for file_path in files:
                        try:
                            start_time = time.time()

                            # Check if plugin can handle this file
                            if hasattr(plugin, "can_analyze") and not plugin.can_analyze(str(file_path)):
                                continue

                            # Run the plugin
                            if hasattr(plugin, "analyze"):
                                result = plugin.analyze(str(file_path))
                            elif hasattr(plugin, "fix"):
                                result = plugin.fix(str(file_path))
                            else:
                                result = {"status": "no_plugin_method", "issues_found": 0}

                            processing_time = time.time() - start_time

                            # Record result
                            analysis_result = AnalysisResult(
                                file_path=str(file_path),
                                directory=directory,
                                analyzer_name=plugin_name,
                                category="plugin_analysis",
                                issues_found=result.get("issues_found", 0),
                                issues_fixed=result.get("issues_fixed", 0),
                                details=result,
                                processing_time=processing_time,
                                status="success",
                            )
                            self.analysis_results.append(analysis_result)

                        except Exception as e:
                            logger.exception(f"Error running plugin {plugin_name} on {file_path}: {e}")
                            analysis_result = AnalysisResult(
                                file_path=str(file_path),
                                directory=directory,
                                analyzer_name=plugin_name,
                                category="plugin_analysis",
                                issues_found=0,
                                issues_fixed=0,
                                details={"error": str(e)},
                                processing_time=0,
                                status="error",
                            )
                            self.analysis_results.append(analysis_result)

        except Exception as e:
            logger.exception(f"Error in plugin analysis: {e}")

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
        [asdict(r) for r in self.analysis_results]
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
                "analysis_duration": self.global_metrics.total_processing_time if self.global_metrics else 0,
            },
            "global_metrics": global_metrics_dict,
            "directory_summaries": directory_summaries_dict,
            "detailed_results": {
                "by_directory": {
                    directory: {
                        category: [asdict(r) for r in results]
                        for category, results in categories.items()
                    }
                    for directory, categories in results_by_directory.items()
                },
                "by_category": {
                    category: [asdict(r) for r in results]
                    for category, results in self._group_results_by_category().items()
                },
                "by_analyzer": {
                    analyzer: [asdict(r) for r in results]
                    for analyzer, results in self._group_results_by_analyzer().items()
                },
            },
            "summary": {
                "total_analysis_runs": len(self.analysis_results),
                "successful_runs": len([r for r in self.analysis_results if r.status == "success"]),
                "failed_runs": len([r for r in self.analysis_results if r.status == "error"]),
                "categories_analyzed": len({r.category for r in self.analysis_results}),
                "analyzers_used": len({r.analyzer_name for r in self.analysis_results}),
            },
        }


    def _group_results_by_category(self) -> dict[str, list[AnalysisResult]]:
        """Group analysis results by category."""
        grouped = defaultdict(list)
        for result in self.analysis_results:
            grouped[result.category].append(result)
        return dict(grouped)

    def _group_results_by_analyzer(self) -> dict[str, list[AnalysisResult]]:
        """Group analysis results by analyzer."""
        grouped = defaultdict(list)
        for result in self.analysis_results:
            grouped[result.analyzer_name].append(result)
        return dict(grouped)

    def save_report(self, report: dict[str, Any], output_file: str = None) -> str:
        """Save the comprehensive report to a file."""
        if output_file is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"comprehensive_professional_analysis_{timestamp}.json"

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)

        logger.info(f"📄 Comprehensive report saved to: {output_path}")
        return str(output_path)

    def generate_text_summary(self, report: dict[str, Any]) -> str:
        """Generate a human-readable text summary."""
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append("COMPREHENSIVE PROFESSIONAL CODE QUALITY ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Generated: {report['metadata']['generated_at']}")
        lines.append(f"Project Root: {report['metadata']['project_root']}")
        lines.append(f"Analysis Duration: {report['metadata']['analysis_duration']:.2f} seconds")
        lines.append("")

        # Global Metrics
        global_metrics = report["global_metrics"]
        lines.append("🌍 GLOBAL METRICS")
        lines.append("-" * 50)
        lines.append(f"Total Directories: {global_metrics['total_directories']}")
        lines.append(f"Total Files: {global_metrics['total_files']}")
        lines.append(f"Total Analyzers Run: {global_metrics['total_analyzers_run']}")
        lines.append(f"Total Issues Found: {global_metrics['total_issues_found']}")
        lines.append(f"Total Issues Fixed: {global_metrics['total_issues_fixed']}")
        lines.append(f"Success Rate: {global_metrics['success_rate']:.1f}%")
        lines.append(f"Categories Covered: {', '.join(global_metrics['categories_covered'])}")
        lines.append("")

        # Top Issues
        if global_metrics["top_issues"]:
            lines.append("🚨 TOP ISSUES BY CATEGORY")
            lines.append("-" * 50)
            for category, count in global_metrics["top_issues"]:
                lines.append(f"• {category}: {count} issues")
            lines.append("")

        # Directory Summaries
        lines.append("📁 DIRECTORY ANALYSIS SUMMARIES")
        lines.append("=" * 80)
        lines.append("")

        for directory, summary in report["directory_summaries"].items():
            lines.append(f"📂 {directory}/")
            lines.append(f"   Files: {summary['total_files']} (analyzed: {summary['files_analyzed']})")
            lines.append(f"   Issues: {summary['total_issues']} (fixed: {summary['total_fixed']})")
            lines.append(f"   Analyzers: {len(summary['analyzers_run'])}")
            lines.append(f"   Categories: {', '.join(summary['categories_covered'])}")
            lines.append(f"   Processing Time: {summary['processing_time']:.2f}s")
            lines.append("")

        # Category Analysis
        lines.append("🔍 ANALYSIS BY CATEGORY")
        lines.append("=" * 80)
        lines.append("")

        for category, results in report["detailed_results"]["by_category"].items():
            total_issues = sum(r["issues_found"] for r in results)
            total_fixed = sum(r["issues_fixed"] for r in results)
            files_analyzed = len({r["file_path"] for r in results})

            lines.append(f"📊 {category.upper()}")
            lines.append(f"   Files Analyzed: {files_analyzed}")
            lines.append(f"   Issues Found: {total_issues}")
            lines.append(f"   Issues Fixed: {total_fixed}")
            lines.append("")

        # Analyzer Performance
        lines.append("⚡ ANALYZER PERFORMANCE")
        lines.append("=" * 80)
        lines.append("")

        for analyzer, results in report["detailed_results"]["by_analyzer"].items():
            total_time = sum(r["processing_time"] for r in results)
            total_issues = sum(r["issues_found"] for r in results)
            success_count = len([r for r in results if r["status"] == "success"])
            total_count = len(results)

            lines.append(f"🔧 {analyzer}")
            lines.append(f"   Files Processed: {total_count}")
            lines.append(f"   Success Rate: {(success_count/total_count*100):.1f}%")
            lines.append(f"   Issues Found: {total_issues}")
            lines.append(f"   Total Time: {total_time:.2f}s")
            lines.append("")

        # Footer
        lines.append("=" * 80)
        lines.append("END OF COMPREHENSIVE PROFESSIONAL ANALYSIS REPORT")
        lines.append("=" * 80)

        return "\n".join(lines)


def main():
    """Main function to run the comprehensive professional analysis."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Professional Code Quality Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run analysis on current directory
  python comprehensive_professional_analysis.py

  # Run analysis on specific directory
  python comprehensive_professional_analysis.py --project-root /path/to/project

  # Custom output file
  python comprehensive_professional_analysis.py --output my_analysis.json

  # Verbose logging
  python comprehensive_professional_analysis.py --verbose
        """,
    )

    parser.add_argument("--project-root", default=".",
                       help="Project root directory to analyze (default: current)")
    parser.add_argument("--output", help="Output file for the JSON report")
    parser.add_argument("--text-summary", help="Output file for text summary")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not CODE_QUALITY_AVAILABLE:
        print("❌ Code quality tools not available!")
        print("Please install the required dependencies:")
        print("pip install -r code_quality/requirements.txt")
        return 1

    try:
        # Initialize analyzer
        analyzer = ComprehensiveProfessionalAnalyzer(args.project_root)

        # Run comprehensive analysis
        report = analyzer.run_comprehensive_analysis()

        # Save JSON report
        analyzer.save_report(report, args.output)

        # Generate and save text summary
        text_summary = analyzer.generate_text_summary(report)
        if args.text_summary:
            text_file = args.text_summary
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            text_file = f"comprehensive_professional_analysis_{timestamp}.txt"

        with open(text_file, "w", encoding="utf-8") as f:
            f.write(text_summary)

        print(f"\n📄 Text summary saved to: {text_file}")

        # Print summary to console
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE PROFESSIONAL ANALYSIS COMPLETE")
        print("="*80)

        global_metrics = report["global_metrics"]
        print(f"🌍 Total Files: {global_metrics['total_files']}")
        print(f"🔍 Total Issues Found: {global_metrics['total_issues_found']}")
        print(f"🔧 Total Issues Fixed: {global_metrics['total_issues_fixed']}")
        print(f"✅ Success Rate: {global_metrics['success_rate']:.1f}%")
        print(f"📁 Directories Analyzed: {global_metrics['total_directories']}")
        print(f"⚡ Total Processing Time: {global_metrics['total_processing_time']:.2f}s")
        print("="*80)

        return 0

    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
