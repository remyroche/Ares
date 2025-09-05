#!/usr/bin/env python3
"""
Comprehensive Code Quality Analysis Pipeline

This pipeline provides the most comprehensive code quality analysis available,
including all analyzers, visualizations, advanced analysis, and plugin support.
It's designed to be the ultimate code quality assessment tool.
"""

import ast
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all analyzers
from analyzers.architecture_analyzer import ArchitectureAnalyzer
from analyzers.call_graph_analyzer import CallGraphAnalyzer
from analyzers.code_duplication_analyzer import CodeDuplicationAnalyzer
from analyzers.code_smell_detector import CodeSmellDetector
from analyzers.complexity_analyzer import ComplexityAnalyzer
from analyzers.concurrency_analyzer import ConcurrencyAnalyzer
from analyzers.configuration_analyzer import ConfigurationAnalyzer
from analyzers.data_flow_analyzer import DataFlowAnalyzer
from analyzers.dependency_analyzer import DependencyAnalyzer
from analyzers.documentation_analyzer import DocumentationAnalyzer
from analyzers.error_handling_analyzer import ErrorHandlingAnalyzer
from analyzers.import_analyzer import ImportAnalyzer
from analyzers.linter_analyzer import LinterAnalyzer
from analyzers.metrics_analyzer import MetricsAnalyzer
from analyzers.performance_analyzer import PerformanceAnalyzer
from analyzers.signature_analyzer import SignatureAnalyzer
from analyzers.static_analysis_analyzer import StaticAnalysisAnalyzer
from analyzers.syntax_validator import SyntaxValidator
from analyzers.test_coverage_analyzer import TestCoverageAnalyzer
from analyzers.type_checker import TypeChecker
from analyzers.undefined_names_analyzer import UndefinedNamesAnalyzer
from analyzers.dead_code_analyzer import DeadCodeAnalyzer
from analyzers.improved_dead_code_analyzer import ImprovedDeadCodeAnalyzer
from analyzers.enhanced_dead_code_analyzer import EnhancedDeadCodeAnalyzer

# Import fixers and scripts
from fixers.sequential_fixer import SequentialFixer
from fixers.auto_fixer import AutoFixer
from fixers.conservative_auto_fixer import ConservativeAutoFixer
from scripts.advanced_syntax_fixer import AdvancedSyntaxFixer
from scripts.enhanced_type_hints import TypeHintEnhancer
from scripts.robust_async_fixer import RobustAsyncFixer
from scripts.detect_circular_imports import CircularImportDetector
from scripts.add_type_hints import TypeHintAdder
from scripts.fix_missing_imports import MissingImportFixer
from scripts.bulk_syntax_cleanup import BulkSyntaxCleanup
from scripts.apply_all_fixes import ApplyAllFixes
from scripts.master_code_quality import MasterCodeQuality

# Import visualizers
from visualizers.dashboard_generator import DashboardGenerator
from visualizers.complexity_heatmap import ComplexityHeatmap
from visualizers.dependency_graph import DependencyGraph
from visualizers.interaction_network import InteractionNetwork
from visualizers.code_visualizer import CodeVisualizer

# Import core components
from comprehensive_code_review import CodeQualityReviewer
from enhanced_validator import EnhancedValidator
from function_validator import FunctionValidator
from map_code_interactions import CodeInteractionMapper
from enhanced_map_code_interactions import EnhancedCodeInteractionMapper
from visualize_interactions import InteractionVisualizer

# Import plugin system
from plugins.plugin_manager import PluginManager
from plugins.plugin_registry import PluginRegistry

# Import utilities
from utils.report_aggregator import ReportAggregator
from utils.file_utils import find_python_files
from core.config import get_default_config


class ComprehensiveAnalysisPipeline:
    """
    Comprehensive code quality analysis pipeline that includes:
    - All available analyzers
    - Advanced visualization
    - Plugin support
    - Consolidated fix scripts
    - Dead code analysis
    - Architecture analysis
    - Performance analysis
    - Security analysis
    """

    def __init__(self, project_root: str = "/workspace/src", enable_plugins: bool = True):
        self.project_root = Path(project_root)
        self.reports_dir = Path("/workspace/code_quality/reports")
        self.reports_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize configuration
        self.config = get_default_config()
        
        # Initialize results structure
        self.results = {
            "syntax_imports": {},
            "async_types": {},
            "basic_analysis": {},
            "advanced_analysis": {},
            "architecture_analysis": {},
            "performance_analysis": {},
            "security_analysis": {},
            "dead_code_analysis": {},
            "visualization": {},
            "consolidated_fixes": {},
            "plugin_results": {},
            "comprehensive_review": {},
            "summary": {},
        }
        
        # Initialize report aggregator
        self.report_aggregator = ReportAggregator(project_root)
        
        # Initialize plugin system
        self.enable_plugins = enable_plugins
        if self.enable_plugins:
            self.plugin_registry = PluginRegistry()
            self.plugin_manager = PluginManager(self.plugin_registry)
            self._register_all_plugins()
        
        # Initialize all analyzers
        self._initialize_all_analyzers()
        
        # Initialize visualizers
        self._initialize_visualizers()

    def _register_all_plugins(self):
        """Register all available plugins."""
        try:
            # Production plugins
            from plugins.production.syntax_fixer import SyntaxFixerPlugin
            from plugins.production.import_fixer import ImportFixerPlugin
            from plugins.production.dead_code_fixer import DeadCodeFixerPlugin
            from plugins.production.linter_runner import LinterRunnerPlugin
            from plugins.production.security_scanner import SecurityScannerPlugin
            
            # Code quality plugins
            from plugins.black_fixer import BlackFixer
            from plugins.isort_fixer import IsortFixer
            from plugins.autopep8_fixer import Autopep8Fixer
            from plugins.autoflake_fixer import AutoflakeFixer
            from plugins.flake8_analyzer import Flake8Analyzer
            from plugins.ruff_analyzer import RuffAnalyzer
            from plugins.ruff_fixer import RuffFixer
            
            # Register plugins
            self.plugin_registry.register_plugin("syntax_fixer", SyntaxFixerPlugin())
            self.plugin_registry.register_plugin("import_fixer", ImportFixerPlugin())
            self.plugin_registry.register_plugin("dead_code_fixer", DeadCodeFixerPlugin())
            self.plugin_registry.register_plugin("linter_runner", LinterRunnerPlugin())
            self.plugin_registry.register_plugin("security_scanner", SecurityScannerPlugin())
            self.plugin_registry.register_plugin("black_fixer", BlackFixer())
            self.plugin_registry.register_plugin("isort_fixer", IsortFixer())
            self.plugin_registry.register_plugin("autopep8_fixer", Autopep8Fixer())
            self.plugin_registry.register_plugin("autoflake_fixer", AutoflakeFixer())
            self.plugin_registry.register_plugin("flake8_analyzer", Flake8Analyzer())
            self.plugin_registry.register_plugin("ruff_analyzer", RuffAnalyzer())
            self.plugin_registry.register_plugin("ruff_fixer", RuffFixer())
            
            print("✓ All plugins registered successfully")
        except ImportError as e:
            print(f"⚠ Warning: Could not register some plugins: {e}")

    def _initialize_all_analyzers(self):
        """Initialize all available analyzers."""
        try:
            self.analyzers = {
                # Basic analyzers
                "syntax_validator": SyntaxValidator(self.config),
                "linter": LinterAnalyzer(self.config),
                "import_analyzer": ImportAnalyzer(self.config),
                "undefined_names": UndefinedNamesAnalyzer(self.config),
                "type_checker": TypeChecker(self.config),
                "static_analysis": StaticAnalysisAnalyzer(self.config),
                
                # Advanced analyzers
                "architecture": ArchitectureAnalyzer(self.config),
                "call_graph": CallGraphAnalyzer(self.config),
                "code_duplication": CodeDuplicationAnalyzer(self.config),
                "complexity": ComplexityAnalyzer(self.config),
                "concurrency": ConcurrencyAnalyzer(self.config),
                "dependency": DependencyAnalyzer(self.config),
                "error_handling": ErrorHandlingAnalyzer(self.config),
                "signature": SignatureAnalyzer(self.config),
                
                # Quality analyzers
                "code_smell": CodeSmellDetector(self.config),
                "configuration": ConfigurationAnalyzer(self.config),
                "data_flow": DataFlowAnalyzer(self.config),
                "documentation": DocumentationAnalyzer(self.config),
                "metrics": MetricsAnalyzer(self.config),
                "performance": PerformanceAnalyzer(self.config),
                "test_coverage": TestCoverageAnalyzer(self.config),
                
                # Dead code analyzers
                "dead_code": DeadCodeAnalyzer(self.config),
                "improved_dead_code": ImprovedDeadCodeAnalyzer(self.config),
                "enhanced_dead_code": EnhancedDeadCodeAnalyzer(self.config),
            }
            print("✓ All analyzers initialized successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not initialize some analyzers: {e}")
            self.analyzers = {}

    def _initialize_visualizers(self):
        """Initialize all visualizers."""
        try:
            self.visualizers = {
                "dashboard": DashboardGenerator(str(self.project_root)),
                "complexity_heatmap": ComplexityHeatmap(str(self.project_root)),
                "dependency_graph": DependencyGraph(str(self.project_root)),
                "interaction_network": InteractionNetwork(str(self.project_root)),
                "code_visualizer": CodeVisualizer(str(self.project_root)),
            }
            print("✓ All visualizers initialized successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not initialize some visualizers: {e}")
            self.visualizers = {}

    def run_syntax_import_analysis(self) -> dict[str, Any]:
        """Run comprehensive syntax and import analysis."""
        print("\n" + "="*60)
        print("Running Syntax and Import Analysis")
        print("="*60)

        start_time = time.time()
        results = {}

        # Sequential Fixer (most comprehensive)
        print("Running Sequential Fixer...")
        try:
            sequential_fixer = SequentialFixer(self.config)
            fixer_results = sequential_fixer.run_analysis(str(self.project_root))
            results["sequential_fixer"] = fixer_results
        except Exception as e:
            results["sequential_fixer"] = {"error": str(e)}

        # Advanced Syntax Fixer
        print("Running Advanced Syntax Fixer...")
        try:
            syntax_fixer = AdvancedSyntaxFixer(str(self.project_root))
            syntax_fixer.fix_all_syntax_errors(dry_run=False)
            results["advanced_syntax_fixer"] = {
                "fixed_files": syntax_fixer.fixed_files,
                "failed_files": syntax_fixer.failed_files,
                "syntax_errors": dict(syntax_fixer.syntax_errors)
            }
        except Exception as e:
            results["advanced_syntax_fixer"] = {"error": str(e)}

        # Import Analysis
        if "import_analyzer" in self.analyzers:
            print("Running Import Analysis...")
            try:
                import_results = self.analyzers["import_analyzer"].analyze_directory(str(self.project_root))
                results["import_analysis"] = import_results
            except Exception as e:
                results["import_analysis"] = {"error": str(e)}

        # Circular Import Detection
        print("Running Circular Import Detection...")
        try:
            circular_detector = CircularImportDetector(str(self.project_root))
            circular_results = circular_detector.detect_circular_imports()
            results["circular_imports"] = circular_results
        except Exception as e:
            results["circular_imports"] = {"error": str(e)}

        results["execution_time"] = time.time() - start_time
        return results

    def run_advanced_analysis(self) -> dict[str, Any]:
        """Run advanced analysis including architecture, complexity, and design patterns."""
        print("\n" + "="*60)
        print("Running Advanced Analysis")
        print("="*60)

        start_time = time.time()
        results = {}

        # Architecture Analysis
        if "architecture" in self.analyzers:
            print("Running Architecture Analysis...")
            try:
                arch_results = self.analyzers["architecture"].analyze_directory(str(self.project_root))
                results["architecture"] = arch_results
            except Exception as e:
                results["architecture"] = {"error": str(e)}

        # Call Graph Analysis
        if "call_graph" in self.analyzers:
            print("Running Call Graph Analysis...")
            try:
                call_graph_results = self.analyzers["call_graph"].analyze_directory(str(self.project_root))
                results["call_graph"] = call_graph_results
            except Exception as e:
                results["call_graph"] = {"error": str(e)}

        # Code Duplication Analysis
        if "code_duplication" in self.analyzers:
            print("Running Code Duplication Analysis...")
            try:
                dup_results = self.analyzers["code_duplication"].analyze_directory(str(self.project_root))
                results["code_duplication"] = dup_results
            except Exception as e:
                results["code_duplication"] = {"error": str(e)}

        # Complexity Analysis
        if "complexity" in self.analyzers:
            print("Running Complexity Analysis...")
            try:
                complexity_results = self.analyzers["complexity"].analyze_directory(str(self.project_root))
                results["complexity"] = complexity_results
            except Exception as e:
                results["complexity"] = {"error": str(e)}

        # Concurrency Analysis
        if "concurrency" in self.analyzers:
            print("Running Concurrency Analysis...")
            try:
                concurrency_results = self.analyzers["concurrency"].analyze_directory(str(self.project_root))
                results["concurrency"] = concurrency_results
            except Exception as e:
                results["concurrency"] = {"error": str(e)}

        # Error Handling Analysis
        if "error_handling" in self.analyzers:
            print("Running Error Handling Analysis...")
            try:
                error_handling_results = self.analyzers["error_handling"].analyze_directory(str(self.project_root))
                results["error_handling"] = error_handling_results
            except Exception as e:
                results["error_handling"] = {"error": str(e)}

        results["execution_time"] = time.time() - start_time
        return results

    def run_dead_code_analysis(self) -> dict[str, Any]:
        """Run comprehensive dead code analysis using all available analyzers."""
        print("\n" + "="*60)
        print("Running Dead Code Analysis")
        print("="*60)

        start_time = time.time()
        results = {}

        # Basic Dead Code Analysis
        if "dead_code" in self.analyzers:
            print("Running Basic Dead Code Analysis...")
            try:
                dead_code_results = self.analyzers["dead_code"].analyze_directory(str(self.project_root))
                results["basic_dead_code"] = dead_code_results
            except Exception as e:
                results["basic_dead_code"] = {"error": str(e)}

        # Improved Dead Code Analysis
        if "improved_dead_code" in self.analyzers:
            print("Running Improved Dead Code Analysis...")
            try:
                improved_results = self.analyzers["improved_dead_code"].analyze_directory(str(self.project_root))
                results["improved_dead_code"] = improved_results
            except Exception as e:
                results["improved_dead_code"] = {"error": str(e)}

        # Enhanced Dead Code Analysis
        if "enhanced_dead_code" in self.analyzers:
            print("Running Enhanced Dead Code Analysis...")
            try:
                enhanced_results = self.analyzers["enhanced_dead_code"].analyze_directory(str(self.project_root))
                results["enhanced_dead_code"] = enhanced_results
            except Exception as e:
                results["enhanced_dead_code"] = {"error": str(e)}

        # Code Interaction Mapping for Dead Code Detection
        print("Running Code Interaction Mapping...")
        try:
            interaction_mapper = CodeInteractionMapper(str(self.project_root))
            interaction_results = interaction_mapper.map_interactions()
            results["interaction_mapping"] = interaction_results
        except Exception as e:
            results["interaction_mapping"] = {"error": str(e)}

        results["execution_time"] = time.time() - start_time
        return results

    def run_visualization_analysis(self) -> dict[str, Any]:
        """Run comprehensive visualization analysis."""
        print("\n" + "="*60)
        print("Running Visualization Analysis")
        print("="*60)

        start_time = time.time()
        results = {}

        # Dashboard Generation
        if "dashboard" in self.visualizers:
            print("Generating Dashboard...")
            try:
                dashboard_results = self.visualizers["dashboard"].generate_dashboard()
                results["dashboard"] = dashboard_results
            except Exception as e:
                results["dashboard"] = {"error": str(e)}

        # Complexity Heatmap
        if "complexity_heatmap" in self.visualizers:
            print("Generating Complexity Heatmap...")
            try:
                heatmap_results = self.visualizers["complexity_heatmap"].generate_heatmap()
                results["complexity_heatmap"] = heatmap_results
            except Exception as e:
                results["complexity_heatmap"] = {"error": str(e)}

        # Dependency Graph
        if "dependency_graph" in self.visualizers:
            print("Generating Dependency Graph...")
            try:
                dep_graph_results = self.visualizers["dependency_graph"].generate_graph()
                results["dependency_graph"] = dep_graph_results
            except Exception as e:
                results["dependency_graph"] = {"error": str(e)}

        # Interaction Network
        if "interaction_network" in self.visualizers:
            print("Generating Interaction Network...")
            try:
                network_results = self.visualizers["interaction_network"].generate_network()
                results["interaction_network"] = network_results
            except Exception as e:
                results["interaction_network"] = {"error": str(e)}

        # Enhanced Code Interaction Mapping
        print("Running Enhanced Code Interaction Mapping...")
        try:
            enhanced_mapper = EnhancedCodeInteractionMapper(str(self.project_root))
            enhanced_results = enhanced_mapper.map_interactions()
            results["enhanced_interactions"] = enhanced_results
        except Exception as e:
            results["enhanced_interactions"] = {"error": str(e)}

        # Interaction Visualization
        print("Running Interaction Visualization...")
        try:
            interaction_viz = InteractionVisualizer(str(self.project_root))
            viz_results = interaction_viz.generate_visualizations()
            results["interaction_visualization"] = viz_results
        except Exception as e:
            results["interaction_visualization"] = {"error": str(e)}

        results["execution_time"] = time.time() - start_time
        return results

    def run_consolidated_fixes(self) -> dict[str, Any]:
        """Run all consolidated fix scripts."""
        print("\n" + "="*60)
        print("Running Consolidated Fix Scripts")
        print("="*60)

        start_time = time.time()
        results = {}

        # Master Code Quality
        print("Running Master Code Quality...")
        try:
            master_quality = MasterCodeQuality(str(self.project_root))
            master_results = master_quality.run_all_quality_checks()
            results["master_code_quality"] = master_results
        except Exception as e:
            results["master_code_quality"] = {"error": str(e)}

        # Apply All Fixes
        print("Running Apply All Fixes...")
        try:
            apply_fixes = ApplyAllFixes(str(self.project_root))
            apply_results = apply_fixes.apply_all_fixes()
            results["apply_all_fixes"] = apply_results
        except Exception as e:
            results["apply_all_fixes"] = {"error": str(e)}

        # Bulk Syntax Cleanup
        print("Running Bulk Syntax Cleanup...")
        try:
            bulk_cleanup = BulkSyntaxCleanup(str(self.project_root))
            bulk_results = bulk_cleanup.cleanup_all_files()
            results["bulk_syntax_cleanup"] = bulk_results
        except Exception as e:
            results["bulk_syntax_cleanup"] = {"error": str(e)}

        # Type Hint Enhancement
        print("Running Type Hint Enhancement...")
        try:
            type_enhancer = TypeHintEnhancer(str(self.project_root))
            type_results = type_enhancer.enhance_type_hints()
            results["type_hint_enhancement"] = type_results
        except Exception as e:
            results["type_hint_enhancement"] = {"error": str(e)}

        # Async/Await Fixes
        print("Running Async/Await Fixes...")
        try:
            async_fixer = RobustAsyncFixer(str(self.project_root))
            async_results = async_fixer.fix_async_issues()
            results["async_fixes"] = async_results
        except Exception as e:
            results["async_fixes"] = {"error": str(e)}

        results["execution_time"] = time.time() - start_time
        return results

    def run_plugin_analysis(self) -> dict[str, Any]:
        """Run all registered plugins."""
        if not self.enable_plugins:
            return {"status": "disabled", "message": "Plugins are disabled"}

        print("\n" + "="*60)
        print("Running Plugin Analysis")
        print("="*60)

        start_time = time.time()
        results = {}

        # Run each registered plugin
        for plugin_name in self.plugin_registry.list_plugins():
            print(f"Running plugin: {plugin_name}")
            try:
                # Create plugin context
                from plugins.base_plugin import PluginContext
                context = PluginContext(
                    project_root=str(self.project_root),
                    config={},
                    files=list(self.project_root.rglob("*.py"))
                )
                
                # Execute plugin
                plugin_result = self.plugin_manager.execute_plugin(plugin_name, context)
                results[plugin_name] = {
                    "status": "success",
                    "result": plugin_result.to_dict()
                }
            except Exception as e:
                results[plugin_name] = {
                    "status": "error",
                    "error": str(e)
                }

        results["execution_time"] = time.time() - start_time
        return results

    def run_comprehensive_review(self) -> dict[str, Any]:
        """Run comprehensive code review."""
        print("\n" + "="*60)
        print("Running Comprehensive Code Review")
        print("="*60)

        start_time = time.time()
        results = {}

        # Comprehensive Code Review
        print("Running Comprehensive Code Review...")
        try:
            reviewer = CodeQualityReviewer(str(self.project_root))
            review_results = reviewer.run_comprehensive_review()
            results["comprehensive_review"] = review_results
        except Exception as e:
            results["comprehensive_review"] = {"error": str(e)}

        # Enhanced Validation
        print("Running Enhanced Validation...")
        try:
            validator = EnhancedValidator(str(self.project_root))
            validation_results = validator.run_validation()
            results["enhanced_validation"] = validation_results
        except Exception as e:
            results["enhanced_validation"] = {"error": str(e)}

        # Function Validation
        print("Running Function Validation...")
        try:
            func_validator = FunctionValidator(str(self.project_root))
            func_results = func_validator.validate_functions()
            results["function_validation"] = func_results
        except Exception as e:
            results["function_validation"] = {"error": str(e)}

        results["execution_time"] = time.time() - start_time
        return results

    def run_all(self) -> dict[str, Any]:
        """Run the complete comprehensive analysis pipeline."""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE CODE QUALITY ANALYSIS PIPELINE")
        print(f"{'='*80}")
        print(f"Project root: {self.project_root}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Plugins enabled: {self.enable_plugins}")

        total_start = time.time()

        # Run all analysis categories
        self.results["syntax_imports"] = self.run_syntax_import_analysis()
        self.results["advanced_analysis"] = self.run_advanced_analysis()
        self.results["dead_code_analysis"] = self.run_dead_code_analysis()
        self.results["visualization"] = self.run_visualization_analysis()
        self.results["consolidated_fixes"] = self.run_consolidated_fixes()
        self.results["plugin_results"] = self.run_plugin_analysis()
        self.results["comprehensive_review"] = self.run_comprehensive_review()

        # Generate summary
        self.results["summary"] = self._generate_summary(time.time() - total_start)

        # Save comprehensive report
        report_path = self.reports_dir / f"comprehensive_analysis_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)

        # Generate unified reports
        print("\n" + "="*60)
        print("Generating Unified Reports")
        print("="*60)

        json_report, md_report = self.report_aggregator.save_reports(
            self.reports_dir,
            "comprehensive_analysis_report",
        )

        print(f"\n{'='*80}")
        print("COMPREHENSIVE ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"Total execution time: {time.time() - total_start:.2f} seconds")
        print(f"Reports saved to: {self.reports_dir}")
        print(f"Main report: {report_path}")
        print(f"Unified JSON report: {json_report}")
        print(f"Unified Markdown report: {md_report}")

        return self.results

    def _generate_summary(self, total_time: float) -> dict[str, Any]:
        """Generate comprehensive summary of all analysis results."""
        summary = {
            "execution_time": total_time,
            "timestamp": self.timestamp,
            "project_root": str(self.project_root),
            "categories_completed": len(self.results) - 1,  # Exclude summary itself
            "analyzers_used": len(self.analyzers),
            "visualizers_used": len(self.visualizers),
            "plugins_enabled": self.enable_plugins,
            "plugin_count": len(self.plugin_registry.list_plugins()) if self.enable_plugins else 0,
            "total_issues_found": 0,
            "critical_issues": 0,
            "warnings": 0,
            "recommendations": [],
        }

        # Aggregate issues from all categories
        for category, results in self.results.items():
            if category == "summary":
                continue
                
            if isinstance(results, dict):
                # Count issues in this category
                category_issues = self._count_issues_in_category(results)
                summary["total_issues_found"] += category_issues["total"]
                summary["critical_issues"] += category_issues["critical"]
                summary["warnings"] += category_issues["warnings"]

        # Generate recommendations
        if summary["critical_issues"] > 0:
            summary["recommendations"].append({
                "priority": "high",
                "category": "critical_issues",
                "message": f"Address {summary['critical_issues']} critical issues immediately"
            })

        if summary["warnings"] > 10:
            summary["recommendations"].append({
                "priority": "medium",
                "category": "warnings",
                "message": f"Review {summary['warnings']} warnings for potential improvements"
            })

        return summary

    def _count_issues_in_category(self, category_results: dict) -> dict[str, int]:
        """Count issues in a category of results."""
        total = 0
        critical = 0
        warnings = 0

        for key, value in category_results.items():
            if isinstance(value, dict):
                if "error" in value:
                    continue  # Skip error entries
                
                # Look for common issue counting patterns
                if "total_issues" in value:
                    total += value["total_issues"]
                if "critical_issues" in value:
                    critical += value["critical_issues"]
                if "warnings" in value:
                    warnings += value["warnings"]
                if "issues" in value and isinstance(value["issues"], list):
                    total += len(value["issues"])

        return {"total": total, "critical": critical, "warnings": warnings}


def main():
    """Main entry point for the comprehensive analysis pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Comprehensive Code Quality Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run comprehensive analysis on default project
  python pipeline_comprehensive_analysis.py
  
  # Run on specific project directory
  python pipeline_comprehensive_analysis.py --project-root /path/to/project
  
  # Disable plugins
  python pipeline_comprehensive_analysis.py --no-plugins
  
  # Run specific categories only
  python pipeline_comprehensive_analysis.py --categories syntax_imports,advanced_analysis
        """
    )

    parser.add_argument("--project-root", "-p", 
                       default="/workspace/src",
                       help="Project root directory to analyze")
    parser.add_argument("--no-plugins", action="store_true",
                       help="Disable plugin system")
    parser.add_argument("--categories", "-c",
                       help="Comma-separated list of categories to run")

    args = parser.parse_args()

    # Create and run pipeline
    pipeline = ComprehensiveAnalysisPipeline(
        project_root=args.project_root,
        enable_plugins=not args.no_plugins
    )

    if args.categories:
        # Run specific categories
        categories = [cat.strip() for cat in args.categories.split(",")]
        results = {}
        for category in categories:
            if hasattr(pipeline, f"run_{category}"):
                method = getattr(pipeline, f"run_{category}")
                results[category] = method()
            else:
                print(f"Warning: Unknown category '{category}'")
        
        # Save partial results
        report_path = pipeline.reports_dir / f"partial_analysis_{pipeline.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Partial analysis complete. Results saved to: {report_path}")
    else:
        # Run complete analysis
        pipeline.run_all()


if __name__ == "__main__":
    main()